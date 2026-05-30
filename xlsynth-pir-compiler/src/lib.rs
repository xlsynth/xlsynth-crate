// SPDX-License-Identifier: Apache-2.0

//! Native JIT compilation for the supported subset of PIR functions.

use std::collections::{HashMap, HashSet};
use std::ptr;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Type as ClifType, Value, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use thiserror::Error;
use xlsynth::IrValue;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils::{get_topological, operands};

type NativeEntrypoint =
    unsafe extern "C" fn(inputs: *const *const u8, output: *mut u8, scratch: *mut u8) -> i32;

/// Describes the native scalar carrier used for one bits-typed PIR value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarLayout {
    /// Number of semantically meaningful PIR bits.
    pub bit_count: usize,
    /// Size of the native carrier in bytes.
    pub byte_count: usize,
}

impl ScalarLayout {
    fn from_type(ty: &Type) -> Result<Self, JitError> {
        let Type::Bits(bit_count) = ty else {
            return Err(JitError::UnsupportedType(ty.to_string()));
        };
        if !(1..=64).contains(bit_count) {
            return Err(JitError::UnsupportedType(format!(
                "bits[{bit_count}] (initial JIT supports bits[1..=64])"
            )));
        }
        let byte_count = match bit_count {
            1..=8 => 1,
            9..=16 => 2,
            17..=32 => 4,
            33..=64 => 8,
            _ => unreachable!("bit width was validated above"),
        };
        Ok(Self {
            bit_count: *bit_count,
            byte_count,
        })
    }

    fn clif_type(self) -> ClifType {
        match self.byte_count {
            1 => types::I8,
            2 => types::I16,
            4 => types::I32,
            8 => types::I64,
            _ => unreachable!("scalar carrier size is fixed by construction"),
        }
    }

    fn storage_bit_count(self) -> usize {
        self.byte_count * 8
    }

    fn mask(self) -> u64 {
        if self.bit_count == 64 {
            u64::MAX
        } else {
            (1u64 << self.bit_count) - 1
        }
    }

    fn validate_value(self, value: u64) -> Result<(), JitError> {
        if value & !self.mask() != 0 {
            Err(JitError::InvalidArgument(format!(
                "value {value:#x} does not fit bits[{}]",
                self.bit_count
            )))
        } else {
            Ok(())
        }
    }
}

/// Describes the native in-memory layout used for one supported PIR value.
///
/// Arrays use the same contiguous element layout as a Rust or C array whose
/// leaf type is the corresponding native scalar carrier. Tuples use C struct
/// field ordering, padding, and alignment, so callers may use corresponding
/// Rust structs annotated with `#[repr(C)]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeValueLayout {
    /// A bits value carried as a native integer type.
    Scalar(ScalarLayout),
    /// A contiguous native array with recursively described elements.
    Array {
        /// Layout of one array element.
        element: Box<NativeValueLayout>,
        /// Number of array elements.
        element_count: usize,
    },
    /// A native C-compatible struct with recursively described fields.
    Tuple {
        /// Layout and byte offset of each field.
        fields: Vec<NativeTupleFieldLayout>,
        /// Size of the complete struct, including tail padding.
        byte_count: usize,
        /// Required alignment for the struct.
        alignment: usize,
    },
}

/// Describes one field in a native C-compatible PIR tuple layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTupleFieldLayout {
    /// Native layout of the field value.
    pub layout: Box<NativeValueLayout>,
    /// Byte offset of the field from the containing tuple address.
    pub offset: usize,
}

impl NativeValueLayout {
    /// Constructs the native representation for a currently supported PIR type.
    pub fn from_type(ty: &Type) -> Result<Self, JitError> {
        match ty {
            Type::Bits(_) => Ok(Self::Scalar(ScalarLayout::from_type(ty)?)),
            Type::Array(array) => {
                if array.element_count == 0 {
                    return Err(JitError::UnsupportedType(
                        "zero-length native C arrays are unsupported".into(),
                    ));
                }
                let element = Self::from_type(array.element_type.as_ref())?;
                element
                    .byte_count()
                    .checked_mul(array.element_count)
                    .ok_or_else(|| {
                        JitError::UnsupportedType(format!("native layout size overflow for {ty}"))
                    })?;
                Ok(Self::Array {
                    element: Box::new(element),
                    element_count: array.element_count,
                })
            }
            Type::Tuple(field_types) => {
                let mut fields = Vec::with_capacity(field_types.len());
                let mut byte_count = 0usize;
                let mut alignment = 1usize;
                for field_ty in field_types {
                    let layout = Self::from_type(field_ty)?;
                    byte_count = align_up(byte_count, layout.alignment())?;
                    fields.push(NativeTupleFieldLayout {
                        layout: Box::new(layout.clone()),
                        offset: byte_count,
                    });
                    byte_count = byte_count.checked_add(layout.byte_count()).ok_or_else(|| {
                        JitError::UnsupportedType(format!(
                            "native tuple layout size overflow for {ty}"
                        ))
                    })?;
                    alignment = alignment.max(layout.alignment());
                }
                byte_count = align_up(byte_count, alignment)?;
                Ok(Self::Tuple {
                    fields,
                    byte_count,
                    alignment,
                })
            }
            Type::Token => Err(JitError::UnsupportedType(ty.to_string())),
        }
    }

    /// Returns this layout's native size in bytes.
    pub fn byte_count(&self) -> usize {
        match self {
            Self::Scalar(layout) => layout.byte_count,
            Self::Array {
                element,
                element_count,
            } => element.byte_count() * element_count,
            Self::Tuple { byte_count, .. } => *byte_count,
        }
    }

    /// Returns this layout's native alignment in bytes.
    pub fn alignment(&self) -> usize {
        match self {
            Self::Scalar(layout) => layout.byte_count,
            Self::Array { element, .. } => element.alignment(),
            Self::Tuple { alignment, .. } => *alignment,
        }
    }

    /// Returns the element stride for a native array layout.
    pub fn element_stride(&self) -> Option<usize> {
        match self {
            Self::Array { element, .. } => Some(element.byte_count()),
            Self::Scalar(_) | Self::Tuple { .. } => None,
        }
    }

    /// Returns the scalar layout, if this layout represents bits directly.
    pub fn as_scalar(&self) -> Option<ScalarLayout> {
        match self {
            Self::Scalar(layout) => Some(*layout),
            Self::Array { .. } | Self::Tuple { .. } => None,
        }
    }
}

/// Error produced while compiling or invoking an initial PIR JIT function.
#[derive(Debug, Error)]
pub enum JitError {
    /// The PIR function is malformed for compilation.
    #[error("invalid PIR function: {0}")]
    InvalidFunction(String),
    /// A PIR type is outside the currently implemented type subset.
    #[error("unsupported PIR type: {0}")]
    UnsupportedType(String),
    /// A PIR operation is outside the currently implemented lowering subset.
    #[error("unsupported PIR node: {0}")]
    UnsupportedNode(String),
    /// Cranelift rejected or failed to finalize generated code.
    #[error("Cranelift backend error: {0}")]
    Backend(String),
    /// Invocation arguments do not conform to the native layout contract.
    #[error("invalid JIT argument: {0}")]
    InvalidArgument(String),
    /// A transitional dynamic value adapter operation failed.
    #[error("value conversion error: {0}")]
    Value(String),
    /// Generated code returned a non-success status.
    #[error("JIT execution returned status {0}")]
    ExecutionFailed(i32),
}

impl JitError {
    /// Returns whether compilation rejected an as-yet unsupported PIR
    /// construct.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::UnsupportedType(_) | Self::UnsupportedNode(_))
    }
}

/// Executable native code for one supported PIR function.
///
/// Bits values use native integer carrier storage sized to the next one of
/// `u8`, `u16`, `u32`, or `u64`. Arrays of supported values use native
/// contiguous array storage; tuples use `#[repr(C)]`-compatible struct
/// storage.
pub struct PirFunctionJit {
    module: Option<JITModule>,
    entrypoint: NativeEntrypoint,
    param_layouts: Vec<NativeValueLayout>,
    result_layout: NativeValueLayout,
    scratch_byte_count: usize,
    scratch_alignment: usize,
}

impl PirFunctionJit {
    /// Compiles the reachable portion of a PIR function into native host code.
    pub fn compile(function: &ir::Fn) -> Result<Self, JitError> {
        function
            .check_pir_layout_invariants()
            .map_err(JitError::InvalidFunction)?;
        xlsynth_pir::ir_verify::verify_function(function)
            .map_err(|e| JitError::InvalidFunction(e.to_string()))?;

        let param_layouts = function
            .params
            .iter()
            .map(|param| NativeValueLayout::from_type(&param.ty))
            .collect::<Result<Vec<_>, _>>()?;
        let result_layout = NativeValueLayout::from_type(&function.ret_ty)?;
        let order = reachable_topological_order(function)?;
        for node_ref in &order {
            NativeValueLayout::from_type(&function.get_node(*node_ref).ty)?;
        }
        let scratch_plan = ScratchPlan::for_function(function, &order)?;

        let builder = JITBuilder::new(default_libcall_names())
            .map_err(|error| JitError::Backend(error.to_string()))?;
        let mut module = JITModule::new(builder);
        let pointer_type = module.target_config().pointer_type();
        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(pointer_type));
        signature.params.push(AbiParam::new(pointer_type));
        signature.params.push(AbiParam::new(pointer_type));
        signature.returns.push(AbiParam::new(types::I32));

        let function_id = module
            .declare_function("xlsynth_pir_entry", Linkage::Export, &signature)
            .map_err(|error| JitError::Backend(error.to_string()))?;
        let mut context = module.make_context();
        context.func.signature = signature;

        let mut function_builder_context = FunctionBuilderContext::new();
        {
            let mut function_builder =
                FunctionBuilder::new(&mut context.func, &mut function_builder_context);
            lower_function(
                function,
                &order,
                &param_layouts,
                &scratch_plan,
                pointer_type,
                &mut function_builder,
            )?;
            function_builder.finalize();
        }

        module
            .define_function(function_id, &mut context)
            .map_err(|error| JitError::Backend(error.to_string()))?;
        module.clear_context(&mut context);
        module
            .finalize_definitions()
            .map_err(|error| JitError::Backend(error.to_string()))?;

        let entrypoint_ptr = module.get_finalized_function(function_id);
        // SAFETY: `entrypoint_ptr` denotes the finalized function just defined
        // with the exact `NativeEntrypoint` signature above.
        let entrypoint: NativeEntrypoint = unsafe { std::mem::transmute(entrypoint_ptr) };
        Ok(Self {
            module: Some(module),
            entrypoint,
            param_layouts,
            result_layout,
            scratch_byte_count: scratch_plan.byte_count,
            scratch_alignment: scratch_plan.alignment,
        })
    }

    /// Returns the native layouts for function parameters.
    pub fn param_layouts(&self) -> &[NativeValueLayout] {
        &self.param_layouts
    }

    /// Returns the native layout for the function result.
    pub fn result_layout(&self) -> &NativeValueLayout {
        &self.result_layout
    }

    /// Returns the required size of the aggregate-intermediate scratch slab.
    pub fn scratch_byte_count(&self) -> usize {
        self.scratch_byte_count
    }

    /// Returns the required alignment of the aggregate-intermediate scratch
    /// slab.
    pub fn scratch_alignment(&self) -> usize {
        self.scratch_alignment
    }

    /// Runs generated code directly against caller-owned native storage.
    ///
    /// No parameter or result values are copied by this method. The generated
    /// code reads from the provided input pointers and writes the result at
    /// `output`. This convenience entrypoint allocates its scratch slab for
    /// each call; use [`Self::run_native_with_scratch`] to reuse scratch
    /// storage across repeated executions.
    ///
    /// # Safety
    ///
    /// Each input pointer must be non-null, properly aligned, and readable for
    /// the native storage described by the corresponding [`NativeValueLayout`].
    /// `output` must be non-null, properly aligned, and writable for the result
    /// layout. Input and output values must obey their bits-width invariants.
    /// When an aggregate result is copied from an input aggregate, input and
    /// output storage must not partially overlap.
    pub unsafe fn run_native(&self, inputs: &[*const u8], output: *mut u8) -> Result<(), JitError> {
        debug_assert!(self.scratch_alignment <= std::mem::align_of::<u64>());
        let mut scratch_words =
            vec![0u64; self.scratch_byte_count.div_ceil(std::mem::size_of::<u64>())];
        let scratch = if scratch_words.is_empty() {
            ptr::null_mut()
        } else {
            scratch_words.as_mut_ptr().cast::<u8>()
        };
        // SAFETY: the caller upholds the native argument/result contract; the
        // scratch vector is aligned and remains alive for this call.
        unsafe {
            self.run_native_with_scratch(
                inputs,
                output,
                scratch,
                scratch_words.len() * std::mem::size_of::<u64>(),
            )
        }
    }

    /// Runs generated code using caller-owned inputs, result, and scratch
    /// storage.
    ///
    /// This is the allocation-free execution entrypoint for repeated calls.
    /// A nonempty scratch buffer must have at least
    /// [`Self::scratch_byte_count`] bytes and satisfy
    /// [`Self::scratch_alignment`].
    ///
    /// # Safety
    ///
    /// The input and output requirements are the same as for
    /// [`Self::run_native`]. When scratch storage is required, `scratch` must
    /// be non-null, properly aligned, writable for `scratch_byte_count` bytes,
    /// and remain alive for this call. The scratch and output storage must not
    /// overlap any storage read during execution.
    pub unsafe fn run_native_with_scratch(
        &self,
        inputs: &[*const u8],
        output: *mut u8,
        scratch: *mut u8,
        scratch_byte_count: usize,
    ) -> Result<(), JitError> {
        if inputs.len() != self.param_layouts.len() {
            return Err(JitError::InvalidArgument(format!(
                "expected {} input pointers, got {}",
                self.param_layouts.len(),
                inputs.len()
            )));
        }
        if inputs.iter().any(|pointer| pointer.is_null()) || output.is_null() {
            return Err(JitError::InvalidArgument(
                "native input/output pointers must be non-null".to_string(),
            ));
        }
        if scratch_byte_count < self.scratch_byte_count {
            return Err(JitError::InvalidArgument(format!(
                "scratch buffer has {scratch_byte_count} bytes, requires {}",
                self.scratch_byte_count
            )));
        }
        if self.scratch_byte_count != 0
            && (scratch.is_null() || (scratch as usize) % self.scratch_alignment != 0)
        {
            return Err(JitError::InvalidArgument(format!(
                "scratch buffer must be non-null and aligned to {} bytes",
                self.scratch_alignment
            )));
        }
        // SAFETY: the caller upholds the pointer and layout contract stated
        // above; the function was finalized with this exact ABI.
        let status = unsafe { (self.entrypoint)(inputs.as_ptr(), output, scratch) };
        if status == 0 {
            Ok(())
        } else {
            Err(JitError::ExecutionFailed(status))
        }
    }

    /// Runs the scalar JIT through a convenient integer adapter.
    pub fn run_u64(&self, args: &[u64]) -> Result<u64, JitError> {
        if args.len() != self.param_layouts.len() {
            return Err(JitError::InvalidArgument(format!(
                "expected {} arguments, got {}",
                self.param_layouts.len(),
                args.len()
            )));
        }
        let param_layouts = self
            .param_layouts
            .iter()
            .map(require_scalar_layout)
            .collect::<Result<Vec<_>, _>>()?;
        let result_layout = require_scalar_layout(&self.result_layout)?;
        let inputs = args
            .iter()
            .zip(&param_layouts)
            .map(|(value, layout)| NativeScalar::new(*layout, *value))
            .collect::<Result<Vec<_>, _>>()?;
        let pointers = inputs.iter().map(NativeScalar::as_ptr).collect::<Vec<_>>();
        let mut output = NativeScalar::new(result_layout, 0)?;
        // SAFETY: `NativeScalar` allocates the native carrier selected by its
        // layout and remains alive for the generated call.
        unsafe { self.run_native(&pointers, output.as_mut_ptr())? };
        Ok(output.value() & result_layout.mask())
    }

    /// Transitional dynamic-value adapter used by differential tests and
    /// fuzzing.
    pub fn run_ir_values(&self, args: &[IrValue]) -> Result<IrValue, JitError> {
        if args.len() != self.param_layouts.len() {
            return Err(JitError::InvalidArgument(format!(
                "expected {} arguments, got {}",
                self.param_layouts.len(),
                args.len()
            )));
        }
        let inputs = args
            .iter()
            .zip(&self.param_layouts)
            .map(|(arg, layout)| NativeValueStorage::from_ir_value(layout, arg))
            .collect::<Result<Vec<_>, _>>()?;
        let pointers = inputs
            .iter()
            .map(NativeValueStorage::as_ptr)
            .collect::<Vec<_>>();
        let mut output = NativeValueStorage::zeroed(&self.result_layout);
        // SAFETY: each `NativeValueStorage` owns aligned storage with exactly
        // the corresponding published native layout and lives across the call.
        unsafe { self.run_native(&pointers, output.as_mut_ptr())? };
        output.to_ir_value(&self.result_layout)
    }
}

impl Drop for PirFunctionJit {
    fn drop(&mut self) {
        let Some(module) = self.module.take() else {
            return;
        };
        // SAFETY: the entrypoint is private and all safe calls borrow `self`,
        // so dropping this owner cannot overlap an invocation of its code.
        unsafe { module.free_memory() };
    }
}

fn require_scalar_layout(layout: &NativeValueLayout) -> Result<ScalarLayout, JitError> {
    layout.as_scalar().ok_or_else(|| {
        JitError::UnsupportedType("this operation requires a native scalar value".into())
    })
}

enum NativeScalar {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

/// Aligned native storage used by the dynamic-value test/fuzz adapter.
struct NativeValueStorage {
    words: Vec<u64>,
}

impl NativeValueStorage {
    fn zeroed(layout: &NativeValueLayout) -> Self {
        Self {
            words: vec![0; layout.byte_count().div_ceil(std::mem::size_of::<u64>())],
        }
    }

    fn from_ir_value(layout: &NativeValueLayout, value: &IrValue) -> Result<Self, JitError> {
        let mut storage = Self::zeroed(layout);
        // SAFETY: storage is sized from `layout` and aligned to at least all
        // currently supported scalar and aggregate layouts.
        unsafe { write_ir_value_to_native(storage.as_mut_ptr(), layout, value)? };
        Ok(storage)
    }

    fn as_ptr(&self) -> *const u8 {
        self.words.as_ptr().cast()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.words.as_mut_ptr().cast()
    }

    fn to_ir_value(&self, layout: &NativeValueLayout) -> Result<IrValue, JitError> {
        // SAFETY: storage remains live and was allocated for `layout`.
        unsafe { read_ir_value_from_native(self.as_ptr(), layout) }
    }
}

unsafe fn write_ir_value_to_native(
    destination: *mut u8,
    layout: &NativeValueLayout,
    value: &IrValue,
) -> Result<(), JitError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let bits = value
                .to_bits()
                .map_err(|error| JitError::Value(error.to_string()))?;
            if bits.get_bit_count() != scalar.bit_count {
                return Err(JitError::InvalidArgument(format!(
                    "expected bits[{}] argument, got bits[{}]",
                    scalar.bit_count,
                    bits.get_bit_count()
                )));
            }
            let integer = bits
                .to_u64()
                .map_err(|error| JitError::Value(error.to_string()))?;
            let bytes = integer.to_ne_bytes();
            // SAFETY: the caller provides storage for this scalar layout.
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), destination, scalar.byte_count);
            }
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let elements = value
                .get_elements()
                .map_err(|error| JitError::InvalidArgument(error.to_string()))?;
            if elements.len() != *element_count {
                return Err(JitError::InvalidArgument(format!(
                    "expected array with {element_count} elements, got {}",
                    elements.len()
                )));
            }
            for (index, child) in elements.iter().enumerate() {
                // SAFETY: each child lies in its recursively described region.
                unsafe {
                    write_ir_value_to_native(
                        destination.add(index * element.byte_count()),
                        element,
                        child,
                    )?;
                }
            }
        }
        NativeValueLayout::Tuple { fields, .. } => {
            let elements = value
                .get_elements()
                .map_err(|error| JitError::InvalidArgument(error.to_string()))?;
            if elements.len() != fields.len() {
                return Err(JitError::InvalidArgument(format!(
                    "expected tuple with {} fields, got {}",
                    fields.len(),
                    elements.len()
                )));
            }
            for (field, child) in fields.iter().zip(elements.iter()) {
                // SAFETY: each field lies at its published C-compatible offset.
                unsafe {
                    write_ir_value_to_native(
                        destination.add(field.offset),
                        field.layout.as_ref(),
                        child,
                    )?;
                }
            }
        }
    }
    Ok(())
}

unsafe fn read_ir_value_from_native(
    source: *const u8,
    layout: &NativeValueLayout,
) -> Result<IrValue, JitError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let mut bytes = [0u8; std::mem::size_of::<u64>()];
            // SAFETY: the caller provides readable storage for this scalar layout.
            unsafe {
                ptr::copy_nonoverlapping(source, bytes.as_mut_ptr(), scalar.byte_count);
            }
            IrValue::make_ubits(scalar.bit_count, u64::from_ne_bytes(bytes) & scalar.mask())
                .map_err(|error| JitError::Value(error.to_string()))
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let mut elements = Vec::with_capacity(*element_count);
            for index in 0..*element_count {
                // SAFETY: each child lies in its recursively described region.
                elements.push(unsafe {
                    read_ir_value_from_native(source.add(index * element.byte_count()), element)?
                });
            }
            IrValue::make_array(&elements).map_err(|error| JitError::Value(error.to_string()))
        }
        NativeValueLayout::Tuple { fields, .. } => {
            let mut elements = Vec::with_capacity(fields.len());
            for field in fields {
                // SAFETY: each field lies at its published C-compatible offset.
                elements.push(unsafe {
                    read_ir_value_from_native(source.add(field.offset), field.layout.as_ref())?
                });
            }
            Ok(IrValue::make_tuple(&elements))
        }
    }
}

impl NativeScalar {
    fn new(layout: ScalarLayout, value: u64) -> Result<Self, JitError> {
        layout.validate_value(value)?;
        Ok(match layout.byte_count {
            1 => Self::U8(value as u8),
            2 => Self::U16(value as u16),
            4 => Self::U32(value as u32),
            8 => Self::U64(value),
            _ => unreachable!("scalar carrier size is fixed by construction"),
        })
    }

    fn as_ptr(&self) -> *const u8 {
        match self {
            Self::U8(value) => ptr::from_ref(value).cast(),
            Self::U16(value) => ptr::from_ref(value).cast(),
            Self::U32(value) => ptr::from_ref(value).cast(),
            Self::U64(value) => ptr::from_ref(value).cast(),
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::U8(value) => ptr::from_mut(value).cast(),
            Self::U16(value) => ptr::from_mut(value).cast(),
            Self::U32(value) => ptr::from_mut(value).cast(),
            Self::U64(value) => ptr::from_mut(value).cast(),
        }
    }

    fn value(&self) -> u64 {
        match self {
            Self::U8(value) => u64::from(*value),
            Self::U16(value) => u64::from(*value),
            Self::U32(value) => u64::from(*value),
            Self::U64(value) => *value,
        }
    }
}

#[derive(Debug)]
struct ScratchPlan {
    offsets: HashMap<NodeRef, usize>,
    byte_count: usize,
    alignment: usize,
}

impl ScratchPlan {
    /// Assigns native scratch slots for materialized intermediate aggregates.
    fn for_function(function: &ir::Fn, order: &[NodeRef]) -> Result<Self, JitError> {
        let mut offsets = HashMap::new();
        let mut byte_count = 0usize;
        let mut alignment = 1usize;
        for node_ref in order {
            let node = function.get_node(*node_ref);
            let layout = NativeValueLayout::from_type(&node.ty)?;
            if !needs_aggregate_destination(node)
                || function
                    .ret_node_ref
                    .is_some_and(|result| result == *node_ref)
            {
                continue;
            }
            byte_count = align_up(byte_count, layout.alignment())?;
            offsets.insert(*node_ref, byte_count);
            byte_count = byte_count.checked_add(layout.byte_count()).ok_or_else(|| {
                JitError::UnsupportedType(format!("scratch size overflow for {}", node.ty))
            })?;
            alignment = alignment.max(layout.alignment());
        }
        Ok(Self {
            offsets,
            byte_count,
            alignment,
        })
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, JitError> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| JitError::UnsupportedType("native layout size overflow".into()))
}

fn needs_aggregate_destination(node: &ir::Node) -> bool {
    matches!(node.ty, Type::Array(_) | Type::Tuple(_))
        && matches!(
            &node.payload,
            NodePayload::Literal(_)
                | NodePayload::Array(_)
                | NodePayload::Tuple(_)
                | NodePayload::ArrayConcat(_)
                | NodePayload::ArraySlice { .. }
                | NodePayload::ArrayUpdate { .. }
                | NodePayload::Binop(Binop::Gate | Binop::Umulp | Binop::Smulp, _, _,)
                | NodePayload::Sel { .. }
                | NodePayload::PrioritySel { .. }
                | NodePayload::OneHotSel { .. }
                | NodePayload::ExtNormalizeLeft { .. }
        )
}

#[derive(Clone, Copy)]
enum ComputedValue {
    Scalar(Value),
    Address(Value),
}

fn reachable_topological_order(function: &ir::Fn) -> Result<Vec<NodeRef>, JitError> {
    let return_node = function.ret_node_ref.ok_or_else(|| {
        JitError::InvalidFunction(format!("function '{}' has no return node", function.name))
    })?;
    let mut stack = vec![return_node];
    let mut reachable = HashSet::new();
    while let Some(node_ref) = stack.pop() {
        if node_ref.index >= function.nodes.len() {
            return Err(JitError::InvalidFunction(format!(
                "node reference {} is out of bounds",
                node_ref.index
            )));
        }
        if reachable.insert(node_ref) {
            stack.extend(operands(&function.get_node(node_ref).payload));
        }
    }
    Ok(get_topological(function)
        .into_iter()
        .filter(|node_ref| reachable.contains(node_ref))
        .collect())
}

fn lower_function(
    function: &ir::Fn,
    order: &[NodeRef],
    param_layouts: &[NativeValueLayout],
    scratch_plan: &ScratchPlan,
    pointer_type: ClifType,
    builder: &mut FunctionBuilder<'_>,
) -> Result<(), JitError> {
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    let inputs_pointer = builder.block_params(entry)[0];
    let output_pointer = builder.block_params(entry)[1];
    let scratch_pointer = builder.block_params(entry)[2];
    let return_node = function.ret_node_ref.ok_or_else(|| {
        JitError::InvalidFunction(format!("function '{}' has no return node", function.name))
    })?;

    let parameter_indices = function
        .params
        .iter()
        .enumerate()
        .map(|(index, parameter)| (parameter.id, index))
        .collect::<HashMap<_, _>>();
    let mut values = vec![None; function.nodes.len()];

    for node_ref in order {
        let node = function.get_node(*node_ref);
        let layout = NativeValueLayout::from_type(&node.ty)?;
        let value = match &node.payload {
            NodePayload::GetParam(param_id) => {
                let param_index = parameter_indices.get(param_id).copied().ok_or_else(|| {
                    JitError::InvalidFunction(format!("unknown parameter id in {}", node.text_id))
                })?;
                let param_pointer = builder.ins().load(
                    pointer_type,
                    MemFlags::new(),
                    inputs_pointer,
                    (param_index * std::mem::size_of::<*const u8>()) as i32,
                );
                load_value_from_storage(builder, param_pointer, &param_layouts[param_index])
            }
            NodePayload::Literal(literal) => match &layout {
                NativeValueLayout::Scalar(scalar) => {
                    ComputedValue::Scalar(lower_scalar_literal(builder, literal, *scalar)?)
                }
                NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. } => {
                    let destination = materialized_destination(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                    )?;
                    lower_literal_to_storage(builder, destination, literal, &layout)?;
                    ComputedValue::Address(destination)
                }
            },
            NodePayload::Unop(Unop::Identity, arg)
                if matches!(
                    layout,
                    NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. }
                ) =>
            {
                computed_value_for(&values, *arg)?
            }
            NodePayload::Unop(op, arg) => {
                let scalar_layout = require_scalar_layout(&layout)?;
                let arg_value = scalar_value_for(&values, *arg)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                let raw = match op {
                    Unop::Identity => arg_value,
                    Unop::Not => builder.ins().bnot(arg_value),
                    Unop::Neg => builder.ins().ineg(arg_value),
                    Unop::Reverse => {
                        let reversed = builder.ins().bitrev(arg_value);
                        let padding = arg_layout.storage_bit_count() - arg_layout.bit_count;
                        if padding == 0 {
                            reversed
                        } else {
                            builder.ins().ushr_imm(reversed, padding as i64)
                        }
                    }
                    Unop::OrReduce => builder.ins().icmp_imm(IntCC::NotEqual, arg_value, 0),
                    Unop::AndReduce => {
                        builder
                            .ins()
                            .icmp_imm(IntCC::Equal, arg_value, arg_layout.mask() as i64)
                    }
                    Unop::XorReduce => {
                        let population = builder.ins().popcnt(arg_value);
                        let parity = builder.ins().band_imm(population, 1);
                        resize_unsigned(builder, parity, arg_layout, scalar_layout)
                    }
                };
                ComputedValue::Scalar(mask_value(builder, raw, scalar_layout))
            }
            NodePayload::Nary(op, args) => {
                let scalar_layout = require_scalar_layout(&layout)?;
                ComputedValue::Scalar(lower_nary(
                    builder,
                    function,
                    node,
                    *op,
                    args,
                    &values,
                    scalar_layout,
                )?)
            }
            NodePayload::ArrayConcat(args) => {
                let destination = materialized_destination(
                    builder,
                    *node_ref,
                    return_node,
                    output_pointer,
                    scratch_pointer,
                    scratch_plan,
                )?;
                lower_array_concat(builder, function, destination, args, &values, &layout)?;
                ComputedValue::Address(destination)
            }
            NodePayload::Binop(Binop::Gate, predicate, gated) => lower_gate(
                builder,
                *node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
                scalar_value_for(&values, *predicate)?,
                computed_value_for(&values, *gated)?,
                &layout,
            )?,
            NodePayload::Binop(op @ (Binop::Umulp | Binop::Smulp), lhs, rhs) => {
                let destination = materialized_destination(
                    builder,
                    *node_ref,
                    return_node,
                    output_pointer,
                    scratch_pointer,
                    scratch_plan,
                )?;
                lower_mulp(
                    builder,
                    function,
                    destination,
                    *op,
                    *lhs,
                    *rhs,
                    &values,
                    &layout,
                )?;
                ComputedValue::Address(destination)
            }
            NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), lhs, rhs)
                if matches!(function.get_node(*lhs).ty, Type::Array(_) | Type::Tuple(_)) =>
            {
                let result_layout = require_scalar_layout(&layout)?;
                let operand_layout = NativeValueLayout::from_type(&function.get_node(*lhs).ty)?;
                let equal = lower_value_equality(
                    builder,
                    computed_value_for(&values, *lhs)?,
                    computed_value_for(&values, *rhs)?,
                    &operand_layout,
                )?;
                let compared = if *op == Binop::Eq {
                    equal
                } else {
                    builder.ins().icmp_imm(IntCC::Equal, equal, 0)
                };
                ComputedValue::Scalar(mask_value(builder, compared, result_layout))
            }
            NodePayload::Binop(op, lhs, rhs) => {
                let scalar_layout = require_scalar_layout(&layout)?;
                ComputedValue::Scalar(lower_binop(
                    builder,
                    function,
                    node,
                    *op,
                    *lhs,
                    *rhs,
                    &values,
                    scalar_layout,
                )?)
            }
            NodePayload::ZeroExt { arg, .. } => {
                let layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                let resized = resize_unsigned(
                    builder,
                    scalar_value_for(&values, *arg)?,
                    arg_layout,
                    layout,
                );
                ComputedValue::Scalar(mask_value(builder, resized, layout))
            }
            NodePayload::SignExt { arg, .. } => {
                let layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                let signed = signed_value(builder, scalar_value_for(&values, *arg)?, arg_layout);
                let resized = resize_signed(builder, signed, arg_layout, layout);
                ComputedValue::Scalar(mask_value(builder, resized, layout))
            }
            NodePayload::BitSlice { arg, start, width } => {
                let layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                if start.saturating_add(*width) > arg_layout.bit_count {
                    return Err(JitError::UnsupportedNode(format!(
                        "out-of-bounds bit_slice at node {}",
                        node.text_id
                    )));
                }
                let shifted = if *start == 0 {
                    scalar_value_for(&values, *arg)?
                } else {
                    builder
                        .ins()
                        .ushr_imm(scalar_value_for(&values, *arg)?, *start as i64)
                };
                let resized = resize_unsigned(builder, shifted, arg_layout, layout);
                ComputedValue::Scalar(mask_value(builder, resized, layout))
            }
            NodePayload::DynamicBitSlice {
                arg,
                start,
                width: _,
            } => {
                let result_layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                let start_layout = ScalarLayout::from_type(&function.get_node(*start).ty)?;
                ComputedValue::Scalar(lower_dynamic_bit_slice(
                    builder,
                    scalar_value_for(&values, *arg)?,
                    arg_layout,
                    scalar_value_for(&values, *start)?,
                    start_layout,
                    result_layout,
                ))
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                let result_layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                let start_layout = ScalarLayout::from_type(&function.get_node(*start).ty)?;
                let update_layout = ScalarLayout::from_type(&function.get_node(*update_value).ty)?;
                ComputedValue::Scalar(lower_bit_slice_update(
                    builder,
                    scalar_value_for(&values, *arg)?,
                    arg_layout,
                    scalar_value_for(&values, *start)?,
                    start_layout,
                    scalar_value_for(&values, *update_value)?,
                    update_layout,
                    result_layout,
                ))
            }
            NodePayload::ExtCarryOut { lhs, rhs, c_in } => {
                let result_layout = require_scalar_layout(&layout)?;
                let operand_layout = ScalarLayout::from_type(&function.get_node(*lhs).ty)?;
                let c_in_layout = ScalarLayout::from_type(&function.get_node(*c_in).ty)?;
                ComputedValue::Scalar(lower_ext_carry_out(
                    builder,
                    scalar_value_for(&values, *lhs)?,
                    scalar_value_for(&values, *rhs)?,
                    scalar_value_for(&values, *c_in)?,
                    operand_layout,
                    c_in_layout,
                    result_layout,
                ))
            }
            NodePayload::ExtPrioEncode { arg, lsb_prio } => {
                let result_layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                ComputedValue::Scalar(lower_ext_prio_encode(
                    builder,
                    scalar_value_for(&values, *arg)?,
                    arg_layout,
                    result_layout,
                    *lsb_prio,
                ))
            }
            NodePayload::ExtClz { arg, offset, .. } => {
                let result_layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                ComputedValue::Scalar(lower_ext_clz(
                    builder,
                    scalar_value_for(&values, *arg)?,
                    arg_layout,
                    result_layout,
                    *offset,
                ))
            }
            NodePayload::ExtNormalizeLeft {
                arg,
                shift_offset,
                normalized_bit_count: _,
                clz_bit_count: _,
            } => lower_ext_normalize_left(
                builder,
                function,
                node,
                *node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
                *arg,
                *shift_offset,
                scalar_value_for(&values, *arg)?,
                &layout,
            )?,
            NodePayload::ExtMaskLow { count } => {
                let result_layout = require_scalar_layout(&layout)?;
                let count_layout = ScalarLayout::from_type(&function.get_node(*count).ty)?;
                ComputedValue::Scalar(lower_ext_mask_low(
                    builder,
                    scalar_value_for(&values, *count)?,
                    count_layout,
                    result_layout,
                ))
            }
            NodePayload::ExtNaryAdd { terms, arch: _ } => {
                let result_layout = require_scalar_layout(&layout)?;
                ComputedValue::Scalar(lower_ext_nary_add(
                    builder,
                    function,
                    terms,
                    &values,
                    result_layout,
                )?)
            }
            NodePayload::Array(args) => {
                let destination = materialized_destination(
                    builder,
                    *node_ref,
                    return_node,
                    output_pointer,
                    scratch_pointer,
                    scratch_plan,
                )?;
                lower_array_construction(builder, destination, args, &values, &layout)?;
                ComputedValue::Address(destination)
            }
            NodePayload::Tuple(args) => {
                let destination = materialized_destination(
                    builder,
                    *node_ref,
                    return_node,
                    output_pointer,
                    scratch_pointer,
                    scratch_plan,
                )?;
                lower_tuple_construction(builder, destination, args, &values, &layout)?;
                ComputedValue::Address(destination)
            }
            NodePayload::TupleIndex { tuple, index } => {
                lower_tuple_index(builder, *tuple, *index, &values, &layout, function, node)?
            }
            NodePayload::ArrayIndex {
                array,
                indices,
                assumed_in_bounds,
            } => lower_array_index(
                builder,
                function,
                node,
                *array,
                indices,
                *assumed_in_bounds,
                &values,
                &layout,
                pointer_type,
            )?,
            NodePayload::ArraySlice {
                array,
                start,
                width: _,
            } => {
                let destination = materialized_destination(
                    builder,
                    *node_ref,
                    return_node,
                    output_pointer,
                    scratch_pointer,
                    scratch_plan,
                )?;
                lower_array_slice(
                    builder,
                    function,
                    destination,
                    *array,
                    *start,
                    &values,
                    &layout,
                    pointer_type,
                )?;
                ComputedValue::Address(destination)
            }
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                assumed_in_bounds,
            } => lower_array_update(
                builder,
                function,
                node,
                *node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
                *array,
                *value,
                indices,
                *assumed_in_bounds,
                &values,
                &layout,
                pointer_type,
            )?,
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => lower_sel(
                builder,
                *node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
                *selector,
                cases,
                *default,
                &values,
                &layout,
            )?,
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => lower_priority_sel(
                builder,
                function,
                node,
                *node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
                *selector,
                cases,
                *default,
                &values,
                &layout,
            )?,
            NodePayload::OneHotSel { selector, cases } => lower_one_hot_sel(
                builder,
                function,
                node,
                *node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
                *selector,
                cases,
                &values,
                &layout,
            )?,
            NodePayload::OneHot { arg, lsb_prio } => {
                let result_layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                ComputedValue::Scalar(lower_one_hot(
                    builder,
                    scalar_value_for(&values, *arg)?,
                    arg_layout,
                    result_layout,
                    *lsb_prio,
                ))
            }
            NodePayload::Encode { arg } => {
                let result_layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                ComputedValue::Scalar(lower_encode(
                    builder,
                    scalar_value_for(&values, *arg)?,
                    arg_layout,
                    result_layout,
                ))
            }
            NodePayload::Decode { arg, width: _ } => {
                let result_layout = require_scalar_layout(&layout)?;
                let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                ComputedValue::Scalar(lower_decode(
                    builder,
                    scalar_value_for(&values, *arg)?,
                    arg_layout,
                    result_layout,
                ))
            }
            _ => return Err(unsupported_node(node)),
        };
        values[node_ref.index] = Some(value);
    }

    let result = computed_value_for(&values, return_node)?;
    store_value_to_storage(
        builder,
        output_pointer,
        result,
        &NativeValueLayout::from_type(&function.ret_ty)?,
    )?;
    let success = builder.ins().iconst(types::I32, 0);
    builder.ins().return_(&[success]);
    Ok(())
}

fn lower_nary(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    op: NaryOp,
    args: &[NodeRef],
    values: &[Option<ComputedValue>],
    layout: ScalarLayout,
) -> Result<Value, JitError> {
    if matches!(op, NaryOp::Concat) {
        return lower_concat(builder, function, node, args, values, layout);
    }
    let Some((first, rest)) = args.split_first() else {
        return Err(unsupported_node(node));
    };
    let mut result = scalar_value_for(values, *first)?;
    for arg in rest {
        let rhs = scalar_value_for(values, *arg)?;
        result = match op {
            NaryOp::And | NaryOp::Nand => builder.ins().band(result, rhs),
            NaryOp::Or | NaryOp::Nor => builder.ins().bor(result, rhs),
            NaryOp::Xor => builder.ins().bxor(result, rhs),
            NaryOp::Concat => unreachable!("concat is handled before bitwise reduction"),
        };
    }
    result = match op {
        NaryOp::Nand | NaryOp::Nor => builder.ins().bnot(result),
        _ => result,
    };
    Ok(mask_value(builder, result, layout))
}

fn lower_binop(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    op: Binop,
    lhs: NodeRef,
    rhs: NodeRef,
    values: &[Option<ComputedValue>],
    layout: ScalarLayout,
) -> Result<Value, JitError> {
    let lhs_layout = ScalarLayout::from_type(&function.get_node(lhs).ty)?;
    let rhs_layout = ScalarLayout::from_type(&function.get_node(rhs).ty)?;
    let lhs_value = scalar_value_for(values, lhs)?;
    let rhs_value = scalar_value_for(values, rhs)?;
    let raw = match op {
        Binop::Add => builder.ins().iadd(lhs_value, rhs_value),
        Binop::Sub => builder.ins().isub(lhs_value, rhs_value),
        Binop::Umul => {
            let lhs_value = resize_unsigned(builder, lhs_value, lhs_layout, layout);
            let rhs_value = resize_unsigned(builder, rhs_value, rhs_layout, layout);
            builder.ins().imul(lhs_value, rhs_value)
        }
        Binop::Smul => {
            let lhs_value = signed_value(builder, lhs_value, lhs_layout);
            let rhs_value = signed_value(builder, rhs_value, rhs_layout);
            let lhs_value = resize_signed(builder, lhs_value, lhs_layout, layout);
            let rhs_value = resize_signed(builder, rhs_value, rhs_layout, layout);
            builder.ins().imul(lhs_value, rhs_value)
        }
        Binop::Eq => builder.ins().icmp(IntCC::Equal, lhs_value, rhs_value),
        Binop::Ne => builder.ins().icmp(IntCC::NotEqual, lhs_value, rhs_value),
        Binop::Ugt => builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThan, lhs_value, rhs_value),
        Binop::Uge => builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs_value, rhs_value),
        Binop::Ult => builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, lhs_value, rhs_value),
        Binop::Ule => builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, lhs_value, rhs_value),
        Binop::Sgt | Binop::Sge | Binop::Slt | Binop::Sle => {
            let lhs_signed = signed_value(builder, lhs_value, lhs_layout);
            let rhs_signed = signed_value(builder, rhs_value, rhs_layout);
            let condition = match op {
                Binop::Sgt => IntCC::SignedGreaterThan,
                Binop::Sge => IntCC::SignedGreaterThanOrEqual,
                Binop::Slt => IntCC::SignedLessThan,
                Binop::Sle => IntCC::SignedLessThanOrEqual,
                _ => unreachable!("signed comparison branch selected above"),
            };
            builder.ins().icmp(condition, lhs_signed, rhs_signed)
        }
        Binop::Shll => lower_shift(builder, lhs_value, rhs_value, lhs_layout, rhs_layout, op),
        Binop::Shrl | Binop::Shra => {
            lower_shift(builder, lhs_value, rhs_value, lhs_layout, rhs_layout, op)
        }
        Binop::Udiv | Binop::Umod | Binop::Sdiv | Binop::Smod => {
            lower_divmod(builder, lhs_value, rhs_value, lhs_layout, op)
        }
        _ => return Err(unsupported_node(node)),
    };
    Ok(mask_value(builder, raw, layout))
}

fn lower_shift(
    builder: &mut FunctionBuilder<'_>,
    lhs: Value,
    rhs: Value,
    lhs_layout: ScalarLayout,
    rhs_layout: ScalarLayout,
    op: Binop,
) -> Value {
    let out_of_bounds = builder.ins().icmp_imm(
        IntCC::UnsignedGreaterThanOrEqual,
        rhs,
        lhs_layout.bit_count as i64,
    );
    let shift = resize_unsigned(builder, rhs, rhs_layout, lhs_layout);
    match op {
        Binop::Shll => {
            let shifted = builder.ins().ishl(lhs, shift);
            let zero = builder.ins().iconst(lhs_layout.clif_type(), 0);
            builder.ins().select(out_of_bounds, zero, shifted)
        }
        Binop::Shrl => {
            let shifted = builder.ins().ushr(lhs, shift);
            let zero = builder.ins().iconst(lhs_layout.clif_type(), 0);
            builder.ins().select(out_of_bounds, zero, shifted)
        }
        Binop::Shra => {
            let signed = signed_value(builder, lhs, lhs_layout);
            let shifted = builder.ins().sshr(signed, shift);
            let fill = builder
                .ins()
                .sshr_imm(signed, (lhs_layout.bit_count - 1) as i64);
            builder.ins().select(out_of_bounds, fill, shifted)
        }
        _ => unreachable!("lower_shift is called only for shift operations"),
    }
}

/// Lowers division and remainder while avoiding Cranelift traps for XLS-defined
/// divide-by-zero and signed-overflow cases.
fn lower_divmod(
    builder: &mut FunctionBuilder<'_>,
    lhs: Value,
    rhs: Value,
    layout: ScalarLayout,
    op: Binop,
) -> Value {
    let zero = builder.ins().iconst(layout.clif_type(), 0);
    let one = builder.ins().iconst(layout.clif_type(), 1);
    let divisor_is_zero = builder.ins().icmp_imm(IntCC::Equal, rhs, 0);
    match op {
        Binop::Udiv | Binop::Umod => {
            let safe_rhs = builder.ins().select(divisor_is_zero, one, rhs);
            let computed = if op == Binop::Udiv {
                builder.ins().udiv(lhs, safe_rhs)
            } else {
                builder.ins().urem(lhs, safe_rhs)
            };
            let exceptional = if op == Binop::Udiv {
                builder
                    .ins()
                    .iconst(layout.clif_type(), layout.mask() as i64)
            } else {
                zero
            };
            builder.ins().select(divisor_is_zero, exceptional, computed)
        }
        Binop::Sdiv | Binop::Smod => {
            let signed_lhs = signed_value(builder, lhs, layout);
            let signed_rhs = signed_value(builder, rhs, layout);
            let minimum = builder
                .ins()
                .iconst(layout.clif_type(), (1u64 << (layout.bit_count - 1)) as i64);
            let negative_one = builder
                .ins()
                .iconst(layout.clif_type(), layout.mask() as i64);
            let lhs_is_minimum = builder.ins().icmp(IntCC::Equal, lhs, minimum);
            let rhs_is_negative_one = builder.ins().icmp(IntCC::Equal, rhs, negative_one);
            let signed_overflow = builder.ins().band(lhs_is_minimum, rhs_is_negative_one);
            let exceptional = builder.ins().bor(divisor_is_zero, signed_overflow);
            let safe_rhs = builder.ins().select(exceptional, one, signed_rhs);
            let computed = if op == Binop::Sdiv {
                builder.ins().sdiv(signed_lhs, safe_rhs)
            } else {
                builder.ins().srem(signed_lhs, safe_rhs)
            };
            if op == Binop::Smod {
                builder.ins().select(exceptional, zero, computed)
            } else {
                let sign_is_negative = builder.ins().icmp_imm(IntCC::SignedLessThan, signed_lhs, 0);
                let maximum = builder.ins().iconst(
                    layout.clif_type(),
                    ((1u64 << (layout.bit_count - 1)) - 1) as i64,
                );
                let div_by_zero_result = builder.ins().select(sign_is_negative, minimum, maximum);
                let exceptional_result =
                    builder
                        .ins()
                        .select(signed_overflow, minimum, div_by_zero_result);
                builder
                    .ins()
                    .select(exceptional, exceptional_result, computed)
            }
        }
        _ => unreachable!("lower_divmod is called only for division/remainder operations"),
    }
}

/// Lowers a scalar concatenation into extension, shifting, and or-ing.
fn lower_concat(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    args: &[NodeRef],
    values: &[Option<ComputedValue>],
    layout: ScalarLayout,
) -> Result<Value, JitError> {
    if args.is_empty() {
        return Err(unsupported_node(node));
    }
    let mut remaining_width = layout.bit_count;
    let mut result = builder.ins().iconst(layout.clif_type(), 0);
    for arg in args {
        let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
        remaining_width = remaining_width
            .checked_sub(arg_layout.bit_count)
            .ok_or_else(|| {
                JitError::InvalidFunction(format!("concat width overflow at {}", node.text_id))
            })?;
        let extended =
            resize_unsigned(builder, scalar_value_for(values, *arg)?, arg_layout, layout);
        let positioned = if remaining_width == 0 {
            extended
        } else {
            builder.ins().ishl_imm(extended, remaining_width as i64)
        };
        result = builder.ins().bor(result, positioned);
    }
    if remaining_width != 0 {
        return Err(JitError::InvalidFunction(format!(
            "concat operands do not fill result type at {}",
            node.text_id
        )));
    }
    Ok(mask_value(builder, result, layout))
}

/// Lowers a zero-filled dynamic bit slice without allowing a backend overshift.
fn lower_dynamic_bit_slice(
    builder: &mut FunctionBuilder<'_>,
    arg: Value,
    arg_layout: ScalarLayout,
    start: Value,
    start_layout: ScalarLayout,
    result_layout: ScalarLayout,
) -> Value {
    let out_of_bounds = builder.ins().icmp_imm(
        IntCC::UnsignedGreaterThanOrEqual,
        start,
        arg_layout.bit_count as i64,
    );
    let zero_start = builder.ins().iconst(start_layout.clif_type(), 0);
    let safe_start = builder.ins().select(out_of_bounds, zero_start, start);
    let shift = resize_unsigned(builder, safe_start, start_layout, arg_layout);
    let shifted = builder.ins().ushr(arg, shift);
    let resized = resize_unsigned(builder, shifted, arg_layout, result_layout);
    let zero = builder.ins().iconst(result_layout.clif_type(), 0);
    let selected = builder.ins().select(out_of_bounds, zero, resized);
    mask_value(builder, selected, result_layout)
}

/// Lowers insertion of a dynamic low-to-high bit slice into a bits value.
#[allow(clippy::too_many_arguments)]
fn lower_bit_slice_update(
    builder: &mut FunctionBuilder<'_>,
    arg: Value,
    arg_layout: ScalarLayout,
    start: Value,
    start_layout: ScalarLayout,
    update: Value,
    update_layout: ScalarLayout,
    result_layout: ScalarLayout,
) -> Value {
    let out_of_bounds = builder.ins().icmp_imm(
        IntCC::UnsignedGreaterThanOrEqual,
        start,
        arg_layout.bit_count as i64,
    );
    let zero_start = builder.ins().iconst(start_layout.clif_type(), 0);
    let safe_start = builder.ins().select(out_of_bounds, zero_start, start);
    let shift = resize_unsigned(builder, safe_start, start_layout, arg_layout);
    let resized_update = resize_unsigned(builder, update, update_layout, arg_layout);
    let update_mask = if update_layout.bit_count >= arg_layout.bit_count {
        arg_layout.mask()
    } else {
        update_layout.mask()
    };
    let update_mask = builder
        .ins()
        .iconst(arg_layout.clif_type(), update_mask as i64);
    let shifted_mask = builder.ins().ishl(update_mask, shift);
    let keep_mask = builder.ins().bnot(shifted_mask);
    let retained = builder.ins().band(arg, keep_mask);
    let inserted = builder.ins().ishl(resized_update, shift);
    let updated = builder.ins().bor(retained, inserted);
    let selected = builder.ins().select(out_of_bounds, arg, updated);
    mask_value(builder, selected, result_layout)
}

/// Lowers the extended carry-out operation for native scalar widths.
fn lower_ext_carry_out(
    builder: &mut FunctionBuilder<'_>,
    lhs: Value,
    rhs: Value,
    c_in: Value,
    operand_layout: ScalarLayout,
    c_in_layout: ScalarLayout,
    result_layout: ScalarLayout,
) -> Value {
    let carry = if operand_layout.bit_count == 64 {
        let sum = builder.ins().iadd(lhs, rhs);
        let carry_from_operands = builder.ins().icmp(IntCC::UnsignedLessThan, sum, lhs);
        let c_in = resize_integer_type_unsigned(builder, c_in, c_in_layout, types::I64);
        let sum_with_carry = builder.ins().iadd(sum, c_in);
        let carry_from_input = builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, sum_with_carry, sum);
        builder.ins().bor(carry_from_operands, carry_from_input)
    } else {
        let lhs = resize_integer_type_unsigned(builder, lhs, operand_layout, types::I64);
        let rhs = resize_integer_type_unsigned(builder, rhs, operand_layout, types::I64);
        let c_in = resize_integer_type_unsigned(builder, c_in, c_in_layout, types::I64);
        let sum = builder.ins().iadd(lhs, rhs);
        let sum = builder.ins().iadd(sum, c_in);
        let shifted = builder.ins().ushr_imm(sum, operand_layout.bit_count as i64);
        let bit = builder.ins().band_imm(shifted, 1);
        builder.ins().icmp_imm(IntCC::NotEqual, bit, 0)
    };
    mask_value(builder, carry, result_layout)
}

/// Lowers priority encode while preserving its all-zero sentinel index.
fn lower_ext_prio_encode(
    builder: &mut FunctionBuilder<'_>,
    arg: Value,
    arg_layout: ScalarLayout,
    result_layout: ScalarLayout,
    lsb_prio: bool,
) -> Value {
    let mut result = builder
        .ins()
        .iconst(result_layout.clif_type(), arg_layout.bit_count as i64);
    let indices: Box<dyn Iterator<Item = usize>> = if lsb_prio {
        Box::new((0..arg_layout.bit_count).rev())
    } else {
        Box::new(0..arg_layout.bit_count)
    };
    for index in indices {
        let selected = bit_is_set(builder, arg, arg_layout, index);
        let encoded = builder
            .ins()
            .iconst(result_layout.clif_type(), index as i64);
        result = builder.ins().select(selected, encoded, result);
    }
    mask_value(builder, result, result_layout)
}

/// Computes the leading-zero count for the logical PIR width, excluding
/// padding in its native carrier.
fn lower_logical_clz(
    builder: &mut FunctionBuilder<'_>,
    arg: Value,
    arg_layout: ScalarLayout,
) -> Value {
    let clz = builder.ins().clz(arg);
    let padding = arg_layout.storage_bit_count() - arg_layout.bit_count;
    if padding == 0 {
        clz
    } else {
        builder.ins().iadd_imm(clz, -(padding as i64))
    }
}

/// Lowers leading-zero count plus a static offset modulo the result width.
fn lower_ext_clz(
    builder: &mut FunctionBuilder<'_>,
    arg: Value,
    arg_layout: ScalarLayout,
    result_layout: ScalarLayout,
    offset: usize,
) -> Value {
    let clz = lower_logical_clz(builder, arg, arg_layout);
    let resized = resize_unsigned(builder, clz, arg_layout, result_layout);
    let adjusted = builder
        .ins()
        .iadd_imm(resized, (offset as u64 & result_layout.mask()) as i64);
    mask_value(builder, adjusted, result_layout)
}

/// Lowers normalize-left, materializing the optional `(normalized, clz)`
/// tuple result in native tuple storage.
#[allow(clippy::too_many_arguments)]
fn lower_ext_normalize_left(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    arg: NodeRef,
    shift_offset: usize,
    arg_value: Value,
    result_layout: &NativeValueLayout,
) -> Result<ComputedValue, JitError> {
    let arg_layout = ScalarLayout::from_type(&function.get_node(arg).ty)?;
    let (normalized_layout, clz_layout) = match result_layout {
        NativeValueLayout::Scalar(layout) => (*layout, None),
        NativeValueLayout::Tuple { fields, .. } if fields.len() == 2 => (
            require_scalar_layout(fields[0].layout.as_ref())?,
            Some(require_scalar_layout(fields[1].layout.as_ref())?),
        ),
        _ => {
            return Err(JitError::InvalidFunction(format!(
                "ext_normalize_left has unexpected result layout at {}",
                node.text_id
            )));
        }
    };
    let logical_clz = lower_logical_clz(builder, arg_value, arg_layout);
    let normalized_zero = builder.ins().iconst(normalized_layout.clif_type(), 0);
    let normalized = if shift_offset >= normalized_layout.bit_count {
        normalized_zero
    } else {
        let resized_arg = resize_unsigned(builder, arg_value, arg_layout, normalized_layout);
        let shift = builder.ins().iadd_imm(logical_clz, shift_offset as i64);
        let out_of_bounds = builder.ins().icmp_imm(
            IntCC::UnsignedGreaterThanOrEqual,
            shift,
            normalized_layout.bit_count as i64,
        );
        let zero_shift = builder.ins().iconst(arg_layout.clif_type(), 0);
        let safe_shift = builder.ins().select(out_of_bounds, zero_shift, shift);
        let safe_shift = resize_unsigned(builder, safe_shift, arg_layout, normalized_layout);
        let shifted = builder.ins().ishl(resized_arg, safe_shift);
        builder
            .ins()
            .select(out_of_bounds, normalized_zero, shifted)
    };
    let normalized = mask_value(builder, normalized, normalized_layout);
    let Some(clz_layout) = clz_layout else {
        return Ok(ComputedValue::Scalar(normalized));
    };
    let NativeValueLayout::Tuple { fields, .. } = result_layout else {
        unreachable!("clz layout is present only for a tuple result");
    };
    let destination = materialized_destination(
        builder,
        node_ref,
        return_node,
        output_pointer,
        scratch_pointer,
        scratch_plan,
    )?;
    let normalized_pointer = pointer_at_offset(builder, destination, fields[0].offset);
    store_value_to_storage(
        builder,
        normalized_pointer,
        ComputedValue::Scalar(normalized),
        fields[0].layout.as_ref(),
    )?;
    let clz = resize_unsigned(builder, logical_clz, arg_layout, clz_layout);
    let clz = mask_value(builder, clz, clz_layout);
    let clz_pointer = pointer_at_offset(builder, destination, fields[1].offset);
    store_value_to_storage(
        builder,
        clz_pointer,
        ComputedValue::Scalar(clz),
        fields[1].layout.as_ref(),
    )?;
    Ok(ComputedValue::Address(destination))
}

/// Lowers generation of a low-bit mask with saturating large counts.
fn lower_ext_mask_low(
    builder: &mut FunctionBuilder<'_>,
    count: Value,
    count_layout: ScalarLayout,
    result_layout: ScalarLayout,
) -> Value {
    let saturated = builder.ins().icmp_imm(
        IntCC::UnsignedGreaterThanOrEqual,
        count,
        result_layout.bit_count as i64,
    );
    let zero_count = builder.ins().iconst(count_layout.clif_type(), 0);
    let safe_count = builder.ins().select(saturated, zero_count, count);
    let shift = resize_unsigned(builder, safe_count, count_layout, result_layout);
    let one = builder.ins().iconst(result_layout.clif_type(), 1);
    let power = builder.ins().ishl(one, shift);
    let mask = builder.ins().isub(power, one);
    let all_ones = builder
        .ins()
        .iconst(result_layout.clif_type(), result_layout.mask() as i64);
    let selected = builder.ins().select(saturated, all_ones, mask);
    mask_value(builder, selected, result_layout)
}

/// Lowers signed/unsigned and optionally negated terms into a modular sum.
fn lower_ext_nary_add(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    terms: &[ir::ExtNaryAddTerm],
    values: &[Option<ComputedValue>],
    result_layout: ScalarLayout,
) -> Result<Value, JitError> {
    let mut result = builder.ins().iconst(result_layout.clif_type(), 0);
    for term in terms {
        let term_layout = ScalarLayout::from_type(&function.get_node(term.operand).ty)?;
        let value = scalar_value_for(values, term.operand)?;
        let resized = if term.signed {
            let signed = signed_value(builder, value, term_layout);
            resize_signed(builder, signed, term_layout, result_layout)
        } else {
            resize_unsigned(builder, value, term_layout, result_layout)
        };
        let contribution = if term.negated {
            builder.ins().ineg(resized)
        } else {
            resized
        };
        result = builder.ins().iadd(result, contribution);
    }
    Ok(mask_value(builder, result, result_layout))
}

/// Lowers XLS partial-product multiplies using the same deterministic
/// decomposition selected by the existing LLVM JIT.
#[allow(clippy::too_many_arguments)]
fn lower_mulp(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    destination: Value,
    op: Binop,
    lhs: NodeRef,
    rhs: NodeRef,
    values: &[Option<ComputedValue>],
    result_layout: &NativeValueLayout,
) -> Result<(), JitError> {
    let NativeValueLayout::Tuple { fields, .. } = result_layout else {
        return Err(JitError::InvalidFunction(
            "partial-product multiply did not have tuple result type".into(),
        ));
    };
    if fields.len() != 2 || fields[0].layout != fields[1].layout {
        return Err(JitError::InvalidFunction(
            "partial-product multiply requires two equal scalar tuple fields".into(),
        ));
    }
    let element_layout = require_scalar_layout(fields[0].layout.as_ref())?;
    let lhs_layout = ScalarLayout::from_type(&function.get_node(lhs).ty)?;
    let rhs_layout = ScalarLayout::from_type(&function.get_node(rhs).ty)?;
    let lhs_value = scalar_value_for(values, lhs)?;
    let rhs_value = scalar_value_for(values, rhs)?;
    let (lhs_value, rhs_value) = if op == Binop::Smulp {
        let lhs_signed = signed_value(builder, lhs_value, lhs_layout);
        let rhs_signed = signed_value(builder, rhs_value, rhs_layout);
        (
            resize_signed(builder, lhs_signed, lhs_layout, element_layout),
            resize_signed(builder, rhs_signed, rhs_layout, element_layout),
        )
    } else {
        (
            resize_unsigned(builder, lhs_value, lhs_layout, element_layout),
            resize_unsigned(builder, rhs_value, rhs_layout, element_layout),
        )
    };
    let product = builder.ins().imul(lhs_value, rhs_value);
    let product = mask_value(builder, product, element_layout);
    let offset_value = mulp_offset_for_llvm_jit(element_layout.bit_count);
    let offset = builder
        .ins()
        .iconst(element_layout.clif_type(), offset_value as i64);
    let residual = builder.ins().isub(product, offset);
    let residual = mask_value(builder, residual, element_layout);
    let first = pointer_at_offset(builder, destination, fields[0].offset);
    let second = pointer_at_offset(builder, destination, fields[1].offset);
    store_value_to_storage(
        builder,
        first,
        ComputedValue::Scalar(offset),
        fields[0].layout.as_ref(),
    )?;
    store_value_to_storage(
        builder,
        second,
        ComputedValue::Scalar(residual),
        fields[1].layout.as_ref(),
    )?;
    Ok(())
}

/// Returns XLS LLVM JIT's deterministic partial-product offset.
fn mulp_offset_for_llvm_jit(result_width: usize) -> u64 {
    let low_width = result_width.saturating_sub(2);
    let high_width = result_width - low_width;
    let low_shift = low_width.saturating_sub(1).min(3);
    let low = if low_width == 0 {
        0
    } else {
        ((1u64 << low_width) - 1) >> low_shift
    };
    let high = if high_width == 0 {
        0
    } else {
        (1u64 << (high_width - 1)) - 1
    };
    (high << low_width) | low
}

/// Lowers indexing into a native array, returning either an element scalar or
/// an address into a nested array value.
fn lower_array_index(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    array: NodeRef,
    indices: &[NodeRef],
    assumed_in_bounds: bool,
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
    pointer_type: ClifType,
) -> Result<ComputedValue, JitError> {
    if indices.is_empty() {
        let value = computed_value_for(values, array)?;
        let input_layout = NativeValueLayout::from_type(&function.get_node(array).ty)?;
        if input_layout != *layout {
            return Err(JitError::InvalidFunction(format!(
                "zero-index array_index result layout disagrees with input at {}",
                node.text_id
            )));
        }
        return Ok(value);
    }
    let mut pointer = address_value_for(values, array)?;
    let mut current_layout = NativeValueLayout::from_type(&function.get_node(array).ty)?;
    for index in indices {
        let NativeValueLayout::Array {
            element,
            element_count,
        } = current_layout
        else {
            return Err(JitError::InvalidFunction(format!(
                "array_index exceeds array dimensions at {}",
                node.text_id
            )));
        };
        if element_count == 0 {
            return Err(JitError::UnsupportedType(
                "zero-length native arrays are not supported for indexing".into(),
            ));
        }
        let index_layout = ScalarLayout::from_type(&function.get_node(*index).ty)?;
        let index_value = scalar_value_for(values, *index)?;
        let bounded_index = clamped_array_index(
            builder,
            index_value,
            index_layout,
            element_count,
            assumed_in_bounds,
            node,
        )?;
        let address_index =
            resize_integer_type_unsigned(builder, bounded_index, index_layout, pointer_type);
        let offset = if element.byte_count() == 1 {
            address_index
        } else {
            builder
                .ins()
                .imul_imm(address_index, element.byte_count() as i64)
        };
        pointer = builder.ins().iadd(pointer, offset);
        current_layout = *element;
    }
    if &current_layout != layout {
        return Err(JitError::InvalidFunction(format!(
            "array_index result layout disagrees with result type at {}",
            node.text_id
        )));
    }
    Ok(load_value_from_storage(builder, pointer, layout))
}

fn lower_scalar_literal(
    builder: &mut FunctionBuilder<'_>,
    literal: &IrValue,
    layout: ScalarLayout,
) -> Result<Value, JitError> {
    let value = literal
        .to_u64()
        .map_err(|error| JitError::Value(error.to_string()))?;
    layout.validate_value(value)?;
    Ok(builder.ins().iconst(layout.clif_type(), value as i64))
}

fn lower_literal_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    literal: &IrValue,
    layout: &NativeValueLayout,
) -> Result<(), JitError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let value = lower_scalar_literal(builder, literal, *scalar)?;
            builder.ins().store(MemFlags::new(), value, destination, 0);
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let elements = literal
                .get_elements()
                .map_err(|error| JitError::Value(error.to_string()))?;
            if elements.len() != *element_count {
                return Err(JitError::InvalidFunction(
                    "literal array element count disagrees with its PIR type".into(),
                ));
            }
            for (index, child) in elements.iter().enumerate() {
                let pointer = pointer_at_offset(builder, destination, index * element.byte_count());
                lower_literal_to_storage(builder, pointer, child, element)?;
            }
        }
        NativeValueLayout::Tuple { fields, .. } => {
            let elements = literal
                .get_elements()
                .map_err(|error| JitError::Value(error.to_string()))?;
            if elements.len() != fields.len() {
                return Err(JitError::InvalidFunction(
                    "literal tuple field count disagrees with its PIR type".into(),
                ));
            }
            for (field, child) in fields.iter().zip(elements.iter()) {
                let pointer = pointer_at_offset(builder, destination, field.offset);
                lower_literal_to_storage(builder, pointer, child, field.layout.as_ref())?;
            }
        }
    }
    Ok(())
}

fn lower_array_construction(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    elements: &[NodeRef],
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
) -> Result<(), JitError> {
    let NativeValueLayout::Array {
        element,
        element_count,
    } = layout
    else {
        return Err(JitError::InvalidFunction(
            "array node did not have array result type".into(),
        ));
    };
    if elements.len() != *element_count {
        return Err(JitError::InvalidFunction(
            "array node operand count disagrees with its result type".into(),
        ));
    }
    for (index, node_ref) in elements.iter().enumerate() {
        let pointer = pointer_at_offset(builder, destination, index * element.byte_count());
        store_value_to_storage(
            builder,
            pointer,
            computed_value_for(values, *node_ref)?,
            element,
        )?;
    }
    Ok(())
}

fn lower_tuple_construction(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    elements: &[NodeRef],
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
) -> Result<(), JitError> {
    let NativeValueLayout::Tuple { fields, .. } = layout else {
        return Err(JitError::InvalidFunction(
            "tuple node did not have tuple result type".into(),
        ));
    };
    if elements.len() != fields.len() {
        return Err(JitError::InvalidFunction(
            "tuple node operand count disagrees with its result type".into(),
        ));
    }
    for (node_ref, field) in elements.iter().zip(fields.iter()) {
        let pointer = pointer_at_offset(builder, destination, field.offset);
        store_value_to_storage(
            builder,
            pointer,
            computed_value_for(values, *node_ref)?,
            field.layout.as_ref(),
        )?;
    }
    Ok(())
}

fn lower_tuple_index(
    builder: &mut FunctionBuilder<'_>,
    tuple: NodeRef,
    index: usize,
    values: &[Option<ComputedValue>],
    result_layout: &NativeValueLayout,
    function: &ir::Fn,
    node: &ir::Node,
) -> Result<ComputedValue, JitError> {
    let tuple_layout = NativeValueLayout::from_type(&function.get_node(tuple).ty)?;
    let NativeValueLayout::Tuple { fields, .. } = tuple_layout else {
        return Err(JitError::InvalidFunction(format!(
            "tuple_index operand is not a tuple at {}",
            node.text_id
        )));
    };
    let field = fields.get(index).ok_or_else(|| {
        JitError::InvalidFunction(format!("tuple_index is out of bounds at {}", node.text_id))
    })?;
    if field.layout.as_ref() != result_layout {
        return Err(JitError::InvalidFunction(format!(
            "tuple_index result layout disagrees with result type at {}",
            node.text_id
        )));
    }
    let tuple_pointer = address_value_for(values, tuple)?;
    let field_pointer = pointer_at_offset(builder, tuple_pointer, field.offset);
    Ok(load_value_from_storage(
        builder,
        field_pointer,
        result_layout,
    ))
}

fn lower_array_concat(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    destination: Value,
    args: &[NodeRef],
    values: &[Option<ComputedValue>],
    result_layout: &NativeValueLayout,
) -> Result<(), JitError> {
    let NativeValueLayout::Array {
        element,
        element_count,
    } = result_layout
    else {
        return Err(JitError::InvalidFunction(
            "array_concat node did not have array result type".into(),
        ));
    };
    if args.is_empty() {
        return Err(JitError::InvalidFunction(
            "array_concat requires at least one operand".into(),
        ));
    }
    let mut destination_start = 0usize;
    for arg in args {
        let Type::Array(arg_ty) = &function.get_node(*arg).ty else {
            return Err(JitError::InvalidFunction(
                "array_concat operand did not have array type".into(),
            ));
        };
        copy_array_elements(
            builder,
            destination,
            destination_start,
            expect_address(computed_value_for(values, *arg)?)?,
            arg_ty.element_count,
            element,
        )?;
        destination_start += arg_ty.element_count;
    }
    if destination_start != *element_count {
        return Err(JitError::InvalidFunction(
            "array_concat result length disagrees with operands".into(),
        ));
    }
    Ok(())
}

/// Compares native values recursively without observing tuple padding bytes.
fn lower_value_equality(
    builder: &mut FunctionBuilder<'_>,
    lhs: ComputedValue,
    rhs: ComputedValue,
    layout: &NativeValueLayout,
) -> Result<Value, JitError> {
    match layout {
        NativeValueLayout::Scalar(_) => {
            Ok(builder
                .ins()
                .icmp(IntCC::Equal, expect_scalar(lhs)?, expect_scalar(rhs)?))
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let lhs = expect_address(lhs)?;
            let rhs = expect_address(rhs)?;
            let mut equal = builder.ins().iconst(types::I8, 1);
            for index in 0..*element_count {
                let offset = index * element.byte_count();
                let lhs_pointer = pointer_at_offset(builder, lhs, offset);
                let rhs_pointer = pointer_at_offset(builder, rhs, offset);
                let lhs_child = load_value_from_storage(builder, lhs_pointer, element);
                let rhs_child = load_value_from_storage(builder, rhs_pointer, element);
                let child_equal = lower_value_equality(builder, lhs_child, rhs_child, element)?;
                equal = builder.ins().band(equal, child_equal);
            }
            Ok(equal)
        }
        NativeValueLayout::Tuple { fields, .. } => {
            let lhs = expect_address(lhs)?;
            let rhs = expect_address(rhs)?;
            let mut equal = builder.ins().iconst(types::I8, 1);
            for field in fields {
                let lhs_pointer = pointer_at_offset(builder, lhs, field.offset);
                let rhs_pointer = pointer_at_offset(builder, rhs, field.offset);
                let lhs_child =
                    load_value_from_storage(builder, lhs_pointer, field.layout.as_ref());
                let rhs_child =
                    load_value_from_storage(builder, rhs_pointer, field.layout.as_ref());
                let child_equal =
                    lower_value_equality(builder, lhs_child, rhs_child, field.layout.as_ref())?;
                equal = builder.ins().band(equal, child_equal);
            }
            Ok(equal)
        }
    }
}

fn copy_array_elements(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    destination_start: usize,
    source: Value,
    source_element_count: usize,
    element_layout: &NativeValueLayout,
) -> Result<(), JitError> {
    for index in 0..source_element_count {
        let source_element =
            pointer_at_offset(builder, source, index * element_layout.byte_count());
        let destination_element = pointer_at_offset(
            builder,
            destination,
            (destination_start + index) * element_layout.byte_count(),
        );
        let element = load_value_from_storage(builder, source_element, element_layout);
        store_value_to_storage(builder, destination_element, element, element_layout)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_array_slice(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    destination: Value,
    array: NodeRef,
    start: NodeRef,
    values: &[Option<ComputedValue>],
    result_layout: &NativeValueLayout,
    pointer_type: ClifType,
) -> Result<(), JitError> {
    let NativeValueLayout::Array {
        element: result_element,
        element_count: result_count,
    } = result_layout
    else {
        return Err(JitError::InvalidFunction(
            "array_slice node did not have array result type".into(),
        ));
    };
    let NativeValueLayout::Array {
        element: input_element,
        element_count: input_count,
    } = NativeValueLayout::from_type(&function.get_node(array).ty)?
    else {
        return Err(JitError::InvalidFunction(
            "array_slice operand did not have array type".into(),
        ));
    };
    if input_element.as_ref() != result_element.as_ref() {
        return Err(JitError::InvalidFunction(
            "array_slice element layout disagrees with result type".into(),
        ));
    }
    let source = address_value_for(values, array)?;
    let start_value = scalar_value_for(values, start)?;
    let start_layout = ScalarLayout::from_type(&function.get_node(start).ty)?;
    for output_index in 0..*result_count {
        let source_element = clamped_array_element_pointer(
            builder,
            source,
            start_value,
            start_layout,
            input_count,
            output_index,
            input_element.byte_count(),
            pointer_type,
        )?;
        let destination_element = pointer_at_offset(
            builder,
            destination,
            output_index * result_element.byte_count(),
        );
        let element = load_value_from_storage(builder, source_element, result_element);
        store_value_to_storage(builder, destination_element, element, result_element)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_array_update(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    array: NodeRef,
    value: NodeRef,
    indices: &[NodeRef],
    assumed_in_bounds: bool,
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
    pointer_type: ClifType,
) -> Result<ComputedValue, JitError> {
    if indices.is_empty() {
        return Ok(computed_value_for(values, value)?);
    }
    let destination = materialized_destination(
        builder,
        node_ref,
        return_node,
        output_pointer,
        scratch_pointer,
        scratch_plan,
    )?;
    store_value_to_storage(
        builder,
        destination,
        computed_value_for(values, array)?,
        layout,
    )?;

    let mut target = destination;
    let mut target_layout = layout.clone();
    let mut all_in_bounds = None;
    for index in indices {
        let NativeValueLayout::Array {
            element,
            element_count,
        } = target_layout
        else {
            return Err(JitError::InvalidFunction(format!(
                "array_update exceeds array dimensions at {}",
                node.text_id
            )));
        };
        let index_layout = ScalarLayout::from_type(&function.get_node(*index).ty)?;
        let (bounded_index, is_in_bounds) = update_array_index(
            builder,
            scalar_value_for(values, *index)?,
            index_layout,
            element_count,
            assumed_in_bounds,
            node,
        )?;
        all_in_bounds = Some(match all_in_bounds {
            Some(condition) => builder.ins().band(condition, is_in_bounds),
            None => is_in_bounds,
        });
        target = array_element_pointer(
            builder,
            target,
            bounded_index,
            index_layout,
            element.byte_count(),
            pointer_type,
        );
        target_layout = *element;
    }
    let condition = all_in_bounds.expect("nonempty array update has an index condition");
    let original = load_value_from_storage(builder, target, &target_layout);
    write_selected_value_to_storage(
        builder,
        target,
        condition,
        computed_value_for(values, value)?,
        Some(original),
        &target_layout,
    )?;
    Ok(ComputedValue::Address(destination))
}

#[allow(clippy::too_many_arguments)]
fn lower_gate(
    builder: &mut FunctionBuilder<'_>,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    predicate: Value,
    gated: ComputedValue,
    layout: &NativeValueLayout,
) -> Result<ComputedValue, JitError> {
    let enabled = builder.ins().icmp_imm(IntCC::NotEqual, predicate, 0);
    selected_value(
        builder,
        node_ref,
        return_node,
        output_pointer,
        scratch_pointer,
        scratch_plan,
        enabled,
        gated,
        None,
        layout,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_sel(
    builder: &mut FunctionBuilder<'_>,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    selector: NodeRef,
    cases: &[NodeRef],
    default: Option<NodeRef>,
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
) -> Result<ComputedValue, JitError> {
    let selector_value = scalar_value_for(values, selector)?;
    let initial = default
        .or_else(|| cases.last().copied())
        .ok_or_else(|| JitError::InvalidFunction("sel requires a case or default".into()))?;
    let mut result = computed_value_for(values, initial)?;
    for (index, case) in cases.iter().enumerate().rev() {
        let selected = builder
            .ins()
            .icmp_imm(IntCC::Equal, selector_value, index as i64);
        result = selected_value(
            builder,
            node_ref,
            return_node,
            output_pointer,
            scratch_pointer,
            scratch_plan,
            selected,
            computed_value_for(values, *case)?,
            Some(result),
            layout,
        )?;
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn lower_priority_sel(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    selector: NodeRef,
    cases: &[NodeRef],
    default: Option<NodeRef>,
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
) -> Result<ComputedValue, JitError> {
    let Some(default) = default else {
        return Err(JitError::UnsupportedNode(format!(
            "priority_sel without default at node {}",
            node.text_id
        )));
    };
    let selector_value = scalar_value_for(values, selector)?;
    let selector_layout = ScalarLayout::from_type(&function.get_node(selector).ty)?;
    let mut result = computed_value_for(values, default)?;
    for (index, case) in cases.iter().enumerate().rev() {
        let selected = bit_is_set(builder, selector_value, selector_layout, index);
        result = selected_value(
            builder,
            node_ref,
            return_node,
            output_pointer,
            scratch_pointer,
            scratch_plan,
            selected,
            computed_value_for(values, *case)?,
            Some(result),
            layout,
        )?;
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn lower_one_hot_sel(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    selector: NodeRef,
    cases: &[NodeRef],
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
) -> Result<ComputedValue, JitError> {
    if cases.is_empty() {
        return Err(unsupported_node(node));
    }
    let selector_value = scalar_value_for(values, selector)?;
    let selector_layout = ScalarLayout::from_type(&function.get_node(selector).ty)?;
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let mut result = builder.ins().iconst(scalar.clif_type(), 0);
            for (index, case) in cases.iter().enumerate() {
                let active = bit_is_set(builder, selector_value, selector_layout, index);
                let with_case = builder.ins().bor(result, scalar_value_for(values, *case)?);
                result = builder.ins().select(active, with_case, result);
            }
            Ok(ComputedValue::Scalar(mask_value(builder, result, *scalar)))
        }
        NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. } => {
            let destination = materialized_destination(
                builder,
                node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
            )?;
            write_zero_value_to_storage(builder, destination, layout);
            for (index, case) in cases.iter().enumerate() {
                let active = bit_is_set(builder, selector_value, selector_layout, index);
                write_selected_or_value_to_storage(
                    builder,
                    destination,
                    active,
                    computed_value_for(values, *case)?,
                    layout,
                )?;
            }
            Ok(ComputedValue::Address(destination))
        }
    }
}

fn lower_one_hot(
    builder: &mut FunctionBuilder<'_>,
    arg: Value,
    arg_layout: ScalarLayout,
    result_layout: ScalarLayout,
    lsb_prio: bool,
) -> Value {
    let mut result = builder.ins().iconst(
        result_layout.clif_type(),
        (1u64 << arg_layout.bit_count) as i64,
    );
    let indices: Box<dyn Iterator<Item = usize>> = if lsb_prio {
        Box::new((0..arg_layout.bit_count).rev())
    } else {
        Box::new(0..arg_layout.bit_count)
    };
    for index in indices {
        let selected = bit_is_set(builder, arg, arg_layout, index);
        let one_hot = builder
            .ins()
            .iconst(result_layout.clif_type(), (1u64 << index) as i64);
        result = builder.ins().select(selected, one_hot, result);
    }
    mask_value(builder, result, result_layout)
}

fn lower_encode(
    builder: &mut FunctionBuilder<'_>,
    arg: Value,
    arg_layout: ScalarLayout,
    result_layout: ScalarLayout,
) -> Value {
    let mut result = builder.ins().iconst(result_layout.clif_type(), 0);
    for index in 0..arg_layout.bit_count {
        let selected = bit_is_set(builder, arg, arg_layout, index);
        let encoded = builder
            .ins()
            .iconst(result_layout.clif_type(), index as i64);
        let zero = builder.ins().iconst(result_layout.clif_type(), 0);
        let contribution = builder.ins().select(selected, encoded, zero);
        result = builder.ins().bor(result, contribution);
    }
    mask_value(builder, result, result_layout)
}

fn lower_decode(
    builder: &mut FunctionBuilder<'_>,
    arg: Value,
    arg_layout: ScalarLayout,
    result_layout: ScalarLayout,
) -> Value {
    let in_bounds =
        builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThan, arg, result_layout.bit_count as i64);
    let shift = resize_unsigned(builder, arg, arg_layout, result_layout);
    let one = builder.ins().iconst(result_layout.clif_type(), 1);
    let decoded = builder.ins().ishl(one, shift);
    let zero = builder.ins().iconst(result_layout.clif_type(), 0);
    builder.ins().select(in_bounds, decoded, zero)
}

fn bit_is_set(
    builder: &mut FunctionBuilder<'_>,
    value: Value,
    layout: ScalarLayout,
    index: usize,
) -> Value {
    if index >= layout.bit_count {
        return builder.ins().icmp(IntCC::NotEqual, value, value);
    }
    let masked = builder.ins().band_imm(value, (1u64 << index) as i64);
    builder.ins().icmp_imm(IntCC::NotEqual, masked, 0)
}

#[allow(clippy::too_many_arguments)]
fn selected_value(
    builder: &mut FunctionBuilder<'_>,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    condition: Value,
    when_true: ComputedValue,
    when_false: Option<ComputedValue>,
    layout: &NativeValueLayout,
) -> Result<ComputedValue, JitError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let true_value = expect_scalar(when_true)?;
            let false_value = match when_false {
                Some(value) => expect_scalar(value)?,
                None => builder.ins().iconst(scalar.clif_type(), 0),
            };
            Ok(ComputedValue::Scalar(builder.ins().select(
                condition,
                true_value,
                false_value,
            )))
        }
        NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. } => {
            let destination = materialized_destination(
                builder,
                node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
            )?;
            write_selected_value_to_storage(
                builder,
                destination,
                condition,
                when_true,
                when_false,
                layout,
            )?;
            Ok(ComputedValue::Address(destination))
        }
    }
}

fn write_selected_value_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    condition: Value,
    when_true: ComputedValue,
    when_false: Option<ComputedValue>,
    layout: &NativeValueLayout,
) -> Result<(), JitError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let false_value = match when_false {
                Some(value) => expect_scalar(value)?,
                None => builder.ins().iconst(scalar.clif_type(), 0),
            };
            let selected = builder
                .ins()
                .select(condition, expect_scalar(when_true)?, false_value);
            builder
                .ins()
                .store(MemFlags::new(), selected, destination, 0);
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let true_source = expect_address(when_true)?;
            let false_source = when_false.map(expect_address).transpose()?;
            for index in 0..*element_count {
                let offset = index * element.byte_count();
                let child_destination = pointer_at_offset(builder, destination, offset);
                let true_pointer = pointer_at_offset(builder, true_source, offset);
                let true_child = load_value_from_storage(builder, true_pointer, element);
                let false_child = false_source.map(|source| {
                    let false_pointer = pointer_at_offset(builder, source, offset);
                    load_value_from_storage(builder, false_pointer, element)
                });
                write_selected_value_to_storage(
                    builder,
                    child_destination,
                    condition,
                    true_child,
                    false_child,
                    element,
                )?;
            }
        }
        NativeValueLayout::Tuple { fields, .. } => {
            let true_source = expect_address(when_true)?;
            let false_source = when_false.map(expect_address).transpose()?;
            for field in fields {
                let child_destination = pointer_at_offset(builder, destination, field.offset);
                let true_pointer = pointer_at_offset(builder, true_source, field.offset);
                let true_child =
                    load_value_from_storage(builder, true_pointer, field.layout.as_ref());
                let false_child = false_source.map(|source| {
                    let false_pointer = pointer_at_offset(builder, source, field.offset);
                    load_value_from_storage(builder, false_pointer, field.layout.as_ref())
                });
                write_selected_value_to_storage(
                    builder,
                    child_destination,
                    condition,
                    true_child,
                    false_child,
                    field.layout.as_ref(),
                )?;
            }
        }
    }
    Ok(())
}

fn write_zero_value_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    layout: &NativeValueLayout,
) {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let zero = builder.ins().iconst(scalar.clif_type(), 0);
            builder.ins().store(MemFlags::new(), zero, destination, 0);
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            for index in 0..*element_count {
                let child = pointer_at_offset(builder, destination, index * element.byte_count());
                write_zero_value_to_storage(builder, child, element);
            }
        }
        NativeValueLayout::Tuple { fields, .. } => {
            for field in fields {
                let child = pointer_at_offset(builder, destination, field.offset);
                write_zero_value_to_storage(builder, child, field.layout.as_ref());
            }
        }
    }
}

fn write_selected_or_value_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    condition: Value,
    value: ComputedValue,
    layout: &NativeValueLayout,
) -> Result<(), JitError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let current = builder
                .ins()
                .load(scalar.clif_type(), MemFlags::new(), destination, 0);
            let combined = builder.ins().bor(current, expect_scalar(value)?);
            let selected = builder.ins().select(condition, combined, current);
            builder
                .ins()
                .store(MemFlags::new(), selected, destination, 0);
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let source = expect_address(value)?;
            for index in 0..*element_count {
                let offset = index * element.byte_count();
                let destination_element = pointer_at_offset(builder, destination, offset);
                let source_element = pointer_at_offset(builder, source, offset);
                let value = load_value_from_storage(builder, source_element, element);
                write_selected_or_value_to_storage(
                    builder,
                    destination_element,
                    condition,
                    value,
                    element,
                )?;
            }
        }
        NativeValueLayout::Tuple { fields, .. } => {
            let source = expect_address(value)?;
            for field in fields {
                let destination_field = pointer_at_offset(builder, destination, field.offset);
                let source_field = pointer_at_offset(builder, source, field.offset);
                let value = load_value_from_storage(builder, source_field, field.layout.as_ref());
                write_selected_or_value_to_storage(
                    builder,
                    destination_field,
                    condition,
                    value,
                    field.layout.as_ref(),
                )?;
            }
        }
    }
    Ok(())
}

fn array_element_pointer(
    builder: &mut FunctionBuilder<'_>,
    array_pointer: Value,
    index: Value,
    index_layout: ScalarLayout,
    element_byte_count: usize,
    pointer_type: ClifType,
) -> Value {
    let address_index = resize_integer_type_unsigned(builder, index, index_layout, pointer_type);
    let offset = if element_byte_count == 1 {
        address_index
    } else {
        builder
            .ins()
            .imul_imm(address_index, element_byte_count as i64)
    };
    builder.ins().iadd(array_pointer, offset)
}

#[allow(clippy::too_many_arguments)]
fn clamped_array_element_pointer(
    builder: &mut FunctionBuilder<'_>,
    array_pointer: Value,
    index: Value,
    index_layout: ScalarLayout,
    element_count: usize,
    additional_index: usize,
    element_byte_count: usize,
    pointer_type: ClifType,
) -> Result<Value, JitError> {
    let max_index = element_count - 1;
    if max_index as u128 > i64::MAX as u128 {
        return Err(JitError::UnsupportedType(
            "array dimensions larger than i64::MAX are unsupported".into(),
        ));
    }
    let address_index = resize_integer_type_unsigned(builder, index, index_layout, pointer_type);
    let bounded = if additional_index > max_index {
        builder.ins().iconst(pointer_type, max_index as i64)
    } else {
        let candidate = if additional_index == 0 {
            address_index
        } else {
            builder
                .ins()
                .iadd_imm(address_index, additional_index as i64)
        };
        let max_start = max_index - additional_index;
        if max_start as u128 >= index_layout.mask() as u128 {
            candidate
        } else {
            let out_of_bounds =
                builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedGreaterThan, index, max_start as i64);
            let final_index = builder.ins().iconst(pointer_type, max_index as i64);
            builder.ins().select(out_of_bounds, final_index, candidate)
        }
    };
    let offset = if element_byte_count == 1 {
        bounded
    } else {
        builder.ins().imul_imm(bounded, element_byte_count as i64)
    };
    Ok(builder.ins().iadd(array_pointer, offset))
}

fn update_array_index(
    builder: &mut FunctionBuilder<'_>,
    index: Value,
    layout: ScalarLayout,
    element_count: usize,
    assumed_in_bounds: bool,
    node: &ir::Node,
) -> Result<(Value, Value), JitError> {
    let max_index = element_count - 1;
    if max_index as u128 > i64::MAX as u128 {
        return Err(JitError::UnsupportedType(
            "array dimensions larger than i64::MAX are unsupported".into(),
        ));
    }
    if layout.mask() <= max_index as u64 {
        let in_bounds = builder.ins().icmp(IntCC::Equal, index, index);
        return Ok((index, in_bounds));
    }
    if assumed_in_bounds {
        return Err(JitError::UnsupportedNode(format!(
            "array_update assumed_in_bounds cannot be guaranteed at node {}",
            node.text_id
        )));
    }
    let in_bounds = builder
        .ins()
        .icmp_imm(IntCC::UnsignedLessThan, index, element_count as i64);
    let final_index = builder.ins().iconst(layout.clif_type(), max_index as i64);
    Ok((
        builder.ins().select(in_bounds, index, final_index),
        in_bounds,
    ))
}

fn clamped_array_index(
    builder: &mut FunctionBuilder<'_>,
    index: Value,
    layout: ScalarLayout,
    element_count: usize,
    assumed_in_bounds: bool,
    node: &ir::Node,
) -> Result<Value, JitError> {
    let max_index = element_count - 1;
    if max_index as u128 > i64::MAX as u128 {
        return Err(JitError::UnsupportedType(
            "array dimensions larger than i64::MAX are unsupported".into(),
        ));
    }
    if layout.mask() < element_count as u64 {
        return Ok(index);
    }
    if assumed_in_bounds {
        return Err(JitError::UnsupportedNode(format!(
            "array_index assumed_in_bounds cannot be guaranteed at node {}",
            node.text_id
        )));
    }
    let out_of_bounds = builder.ins().icmp_imm(
        IntCC::UnsignedGreaterThanOrEqual,
        index,
        element_count as i64,
    );
    let final_index = builder.ins().iconst(layout.clif_type(), max_index as i64);
    Ok(builder.ins().select(out_of_bounds, final_index, index))
}

fn resize_integer_type_unsigned(
    builder: &mut FunctionBuilder<'_>,
    value: Value,
    from: ScalarLayout,
    to: ClifType,
) -> Value {
    match from.byte_count.cmp(&(to.bytes() as usize)) {
        std::cmp::Ordering::Less => builder.ins().uextend(to, value),
        std::cmp::Ordering::Equal => value,
        std::cmp::Ordering::Greater => builder.ins().ireduce(to, value),
    }
}

fn materialized_destination(
    builder: &mut FunctionBuilder<'_>,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
) -> Result<Value, JitError> {
    if node_ref == return_node {
        return Ok(output_pointer);
    }
    let offset = scratch_plan
        .offsets
        .get(&node_ref)
        .copied()
        .ok_or_else(|| {
            JitError::InvalidFunction(format!(
                "materialized aggregate node {} has no scratch assignment",
                node_ref.index
            ))
        })?;
    Ok(pointer_at_offset(builder, scratch_pointer, offset))
}

fn pointer_at_offset(builder: &mut FunctionBuilder<'_>, pointer: Value, offset: usize) -> Value {
    if offset == 0 {
        pointer
    } else {
        builder.ins().iadd_imm(pointer, offset as i64)
    }
}

fn load_value_from_storage(
    builder: &mut FunctionBuilder<'_>,
    pointer: Value,
    layout: &NativeValueLayout,
) -> ComputedValue {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let value = builder
                .ins()
                .load(scalar.clif_type(), MemFlags::new(), pointer, 0);
            ComputedValue::Scalar(mask_value(builder, value, *scalar))
        }
        NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. } => {
            ComputedValue::Address(pointer)
        }
    }
}

fn store_value_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    value: ComputedValue,
    layout: &NativeValueLayout,
) -> Result<(), JitError> {
    match layout {
        NativeValueLayout::Scalar(_) => {
            builder
                .ins()
                .store(MemFlags::new(), expect_scalar(value)?, destination, 0);
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let source = expect_address(value)?;
            for index in 0..*element_count {
                let offset = index * element.byte_count();
                let source_element = pointer_at_offset(builder, source, offset);
                let destination_element = pointer_at_offset(builder, destination, offset);
                let child = load_value_from_storage(builder, source_element, element);
                store_value_to_storage(builder, destination_element, child, element)?;
            }
        }
        NativeValueLayout::Tuple { fields, .. } => {
            let source = expect_address(value)?;
            for field in fields {
                let source_field = pointer_at_offset(builder, source, field.offset);
                let destination_field = pointer_at_offset(builder, destination, field.offset);
                let child = load_value_from_storage(builder, source_field, field.layout.as_ref());
                store_value_to_storage(builder, destination_field, child, field.layout.as_ref())?;
            }
        }
    }
    Ok(())
}

fn computed_value_for(
    values: &[Option<ComputedValue>],
    node_ref: NodeRef,
) -> Result<ComputedValue, JitError> {
    values
        .get(node_ref.index)
        .copied()
        .flatten()
        .ok_or_else(|| {
            JitError::InvalidFunction(format!(
                "operand node {} was not lowered before its user",
                node_ref.index
            ))
        })
}

fn scalar_value_for(
    values: &[Option<ComputedValue>],
    node_ref: NodeRef,
) -> Result<Value, JitError> {
    expect_scalar(computed_value_for(values, node_ref)?)
}

fn address_value_for(
    values: &[Option<ComputedValue>],
    node_ref: NodeRef,
) -> Result<Value, JitError> {
    expect_address(computed_value_for(values, node_ref)?)
}

fn expect_scalar(value: ComputedValue) -> Result<Value, JitError> {
    match value {
        ComputedValue::Scalar(value) => Ok(value),
        ComputedValue::Address(_) => Err(JitError::InvalidFunction(
            "array value used as a scalar".into(),
        )),
    }
}

fn expect_address(value: ComputedValue) -> Result<Value, JitError> {
    match value {
        ComputedValue::Address(value) => Ok(value),
        ComputedValue::Scalar(_) => Err(JitError::InvalidFunction(
            "scalar value used as an array".into(),
        )),
    }
}

fn mask_value(builder: &mut FunctionBuilder<'_>, value: Value, layout: ScalarLayout) -> Value {
    if layout.bit_count == layout.storage_bit_count() {
        value
    } else {
        builder.ins().band_imm(value, layout.mask() as i64)
    }
}

fn signed_value(builder: &mut FunctionBuilder<'_>, value: Value, layout: ScalarLayout) -> Value {
    let padding = layout.storage_bit_count() - layout.bit_count;
    if padding == 0 {
        value
    } else {
        let shifted = builder.ins().ishl_imm(value, padding as i64);
        builder.ins().sshr_imm(shifted, padding as i64)
    }
}

fn resize_unsigned(
    builder: &mut FunctionBuilder<'_>,
    value: Value,
    from: ScalarLayout,
    to: ScalarLayout,
) -> Value {
    match from.byte_count.cmp(&to.byte_count) {
        std::cmp::Ordering::Less => builder.ins().uextend(to.clif_type(), value),
        std::cmp::Ordering::Equal => value,
        std::cmp::Ordering::Greater => builder.ins().ireduce(to.clif_type(), value),
    }
}

fn resize_signed(
    builder: &mut FunctionBuilder<'_>,
    value: Value,
    from: ScalarLayout,
    to: ScalarLayout,
) -> Value {
    match from.byte_count.cmp(&to.byte_count) {
        std::cmp::Ordering::Less => builder.ins().sextend(to.clif_type(), value),
        std::cmp::Ordering::Equal => value,
        std::cmp::Ordering::Greater => builder.ins().ireduce(to.clif_type(), value),
    }
}

fn unsupported_node(node: &ir::Node) -> JitError {
    JitError::UnsupportedNode(format!(
        "{} at node {} ({})",
        node.payload.get_operator(),
        node.text_id,
        node.ty
    ))
}
