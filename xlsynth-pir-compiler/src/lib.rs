// SPDX-License-Identifier: Apache-2.0

//! Native compilation and in-memory execution for the supported subset of PIR
//! functions.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::ptr;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    AbiParam, ExtFuncData, ExternalName, FuncRef, InstBuilder, LibCall, MemFlags, Signature,
    Type as ClifType, Value, types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use thiserror::Error;
use xlsynth::{IrBits, IrValue};
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils::{is_observable_effect_root, operands};
pub use xlsynth_pir_compiler_runtime::{
    AssertionFailure, AssumptionFailure, AssumptionFailureKind, CompiledEntrypoint,
    CompiledFunctionMetadata, CoverCount, EventKind, EventSiteMetadata, ExecutionContext,
    ExecutionResult, RawExecutionContext, TraceMessage, TraceTupleFieldLayout, TraceValueLayout,
    WideBinaryOp, WideUnaryOp,
};
use xlsynth_pir_compiler_runtime::{
    xlsynth_pir_record_assert, xlsynth_pir_record_assumption_failure, xlsynth_pir_record_cover,
    xlsynth_pir_record_trace, xlsynth_pir_runtime_wide_binop,
    xlsynth_pir_runtime_wide_bit_slice_update, xlsynth_pir_runtime_wide_dynamic_bit_slice,
    xlsynth_pir_runtime_wide_mulp, xlsynth_pir_runtime_wide_unary_op,
};

type NativeEntrypoint = CompiledEntrypoint;

/// Describes the native scalar carrier used for one bits-typed PIR value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarLayout {
    /// Number of semantically meaningful PIR bits.
    pub bit_count: usize,
    /// Size of the native carrier in bytes.
    pub byte_count: usize,
}

impl ScalarLayout {
    fn from_type(ty: &Type) -> Result<Self, CompilerError> {
        let Type::Bits(bit_count) = ty else {
            return Err(CompilerError::UnsupportedType(ty.to_string()));
        };
        if !(1..=64).contains(bit_count) {
            return Err(CompilerError::UnsupportedType(format!(
                "bits[{bit_count}] is not a native scalar"
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

    fn validate_value(self, value: u64) -> Result<(), CompilerError> {
        if value & !self.mask() != 0 {
            Err(CompilerError::InvalidArgument(format!(
                "value {value:#x} does not fit bits[{}]",
                self.bit_count
            )))
        } else {
            Ok(())
        }
    }
}

/// Describes the native limb storage used for a bitvector wider than 64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideBitsLayout {
    /// Number of semantically meaningful PIR bits.
    pub bit_count: usize,
    /// Number of least-significant-first native `u64` limbs in storage.
    pub limb_count: usize,
}

impl WideBitsLayout {
    fn new(bit_count: usize) -> Self {
        debug_assert!(bit_count > 64);
        Self {
            bit_count,
            limb_count: bit_count.div_ceil(64),
        }
    }

    fn byte_count(self) -> usize {
        self.limb_count * std::mem::size_of::<u64>()
    }

    fn high_mask(self) -> u64 {
        let remainder = self.bit_count % 64;
        if remainder == 0 {
            u64::MAX
        } else {
            (1u64 << remainder) - 1
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
    /// A bits value carried in least-significant-first `u64` limbs.
    WideBits(WideBitsLayout),
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
    /// A zero-sized token used only for event ordering.
    Token,
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
    pub fn from_type(ty: &Type) -> Result<Self, CompilerError> {
        match ty {
            Type::Bits(bit_count) if *bit_count == 0 => Err(CompilerError::UnsupportedType(
                "bits[0] native storage is unsupported".into(),
            )),
            Type::Bits(bit_count) if *bit_count <= 64 => {
                Ok(Self::Scalar(ScalarLayout::from_type(ty)?))
            }
            Type::Bits(bit_count) => Ok(Self::WideBits(WideBitsLayout::new(*bit_count))),
            Type::Array(array) => {
                if array.element_count == 0 {
                    return Err(CompilerError::UnsupportedType(
                        "zero-length native C arrays are unsupported".into(),
                    ));
                }
                let element = Self::from_type(array.element_type.as_ref())?;
                element
                    .byte_count()
                    .checked_mul(array.element_count)
                    .ok_or_else(|| {
                        CompilerError::UnsupportedType(format!(
                            "native layout size overflow for {ty}"
                        ))
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
                        CompilerError::UnsupportedType(format!(
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
            Type::Token => Ok(Self::Token),
        }
    }

    /// Returns this layout's native size in bytes.
    pub fn byte_count(&self) -> usize {
        match self {
            Self::Scalar(layout) => layout.byte_count,
            Self::WideBits(layout) => layout.byte_count(),
            Self::Array {
                element,
                element_count,
            } => element.byte_count() * element_count,
            Self::Tuple { byte_count, .. } => *byte_count,
            Self::Token => 0,
        }
    }

    /// Returns this layout's native alignment in bytes.
    pub fn alignment(&self) -> usize {
        match self {
            Self::Scalar(layout) => layout.byte_count,
            Self::WideBits(_) => std::mem::align_of::<u64>(),
            Self::Array { element, .. } => element.alignment(),
            Self::Tuple { alignment, .. } => *alignment,
            Self::Token => 1,
        }
    }

    /// Returns the element stride for a native array layout.
    pub fn element_stride(&self) -> Option<usize> {
        match self {
            Self::Array { element, .. } => Some(element.byte_count()),
            Self::Scalar(_) | Self::WideBits(_) | Self::Tuple { .. } | Self::Token => None,
        }
    }

    /// Returns the scalar layout, if this layout represents bits directly.
    pub fn as_scalar(&self) -> Option<ScalarLayout> {
        match self {
            Self::Scalar(layout) => Some(*layout),
            Self::WideBits(_) | Self::Array { .. } | Self::Tuple { .. } | Self::Token => None,
        }
    }

    fn is_memory_backed(&self) -> bool {
        matches!(
            self,
            Self::WideBits(_) | Self::Array { .. } | Self::Tuple { .. }
        )
    }
}

/// Error produced while compiling or invoking a PIR function.
#[derive(Debug, Error)]
pub enum CompilerError {
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
    #[error("invalid compiled-function argument: {0}")]
    InvalidArgument(String),
    /// A transitional dynamic value adapter operation failed.
    #[error("value conversion error: {0}")]
    Value(String),
    /// Generated code returned a non-success status.
    #[error("compiled execution returned status {0}")]
    ExecutionFailed(i32),
}

impl CompilerError {
    /// Returns whether compilation rejected an as-yet unsupported PIR
    /// construct.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::UnsupportedType(_) | Self::UnsupportedNode(_))
    }
}

/// Executable native code for one supported PIR function.
///
/// Bits values through width 64 use native integer carrier storage sized to
/// the next one of `u8`, `u16`, `u32`, or `u64`; wider bits values use
/// least-significant-first native `u64` limbs. Arrays of supported values use
/// native contiguous array storage; tuples use `#[repr(C)]`-compatible struct
/// storage.
pub struct PirFunctionCompiler {
    module: Option<JITModule>,
    entrypoint: NativeEntrypoint,
    param_layouts: Vec<NativeValueLayout>,
    result_layout: NativeValueLayout,
    metadata: CompiledFunctionMetadata,
    scratch_byte_count: usize,
    scratch_alignment: usize,
}

/// Value and observable events produced by one dynamic-value execution.
#[derive(Debug, Clone, PartialEq)]
pub struct IrExecutionResult {
    pub value: IrValue,
    pub events: ExecutionResult,
}

impl PirFunctionCompiler {
    /// Compiles the reachable portion of a PIR function into native host code.
    pub fn compile(function: &ir::Fn) -> Result<Self, CompilerError> {
        function
            .check_pir_layout_invariants()
            .map_err(CompilerError::InvalidFunction)?;
        xlsynth_pir::ir_verify::verify_function(function)
            .map_err(|e| CompilerError::InvalidFunction(e.to_string()))?;

        let param_layouts = function
            .params
            .iter()
            .map(|param| NativeValueLayout::from_type(&param.ty))
            .collect::<Result<Vec<_>, _>>()?;
        let result_layout = NativeValueLayout::from_type(&function.ret_ty)?;
        let order = reachable_scheduled_order(function)?;
        for node_ref in &order {
            NativeValueLayout::from_type(&function.get_node(*node_ref).ty)?;
        }
        let scratch_plan = ScratchPlan::for_function(function, &order)?;
        let (metadata, event_sites) = build_event_metadata(function, &order)?;

        let mut builder = JITBuilder::new(default_libcall_names())
            .map_err(|error| CompilerError::Backend(error.to_string()))?;
        builder.symbol(
            "xlsynth_pir_record_assert",
            xlsynth_pir_record_assert as *const u8,
        );
        builder.symbol(
            "xlsynth_pir_record_assumption_failure",
            xlsynth_pir_record_assumption_failure as *const u8,
        );
        builder.symbol(
            "xlsynth_pir_record_cover",
            xlsynth_pir_record_cover as *const u8,
        );
        builder.symbol(
            "xlsynth_pir_record_trace",
            xlsynth_pir_record_trace as *const u8,
        );
        builder.symbol(
            "xlsynth_pir_runtime_wide_binop",
            xlsynth_pir_runtime_wide_binop as *const u8,
        );
        builder.symbol(
            "xlsynth_pir_runtime_wide_dynamic_bit_slice",
            xlsynth_pir_runtime_wide_dynamic_bit_slice as *const u8,
        );
        builder.symbol(
            "xlsynth_pir_runtime_wide_bit_slice_update",
            xlsynth_pir_runtime_wide_bit_slice_update as *const u8,
        );
        builder.symbol(
            "xlsynth_pir_runtime_wide_unary_op",
            xlsynth_pir_runtime_wide_unary_op as *const u8,
        );
        builder.symbol(
            "xlsynth_pir_runtime_wide_mulp",
            xlsynth_pir_runtime_wide_mulp as *const u8,
        );
        let mut module = JITModule::new(builder);
        let pointer_type = module.target_config().pointer_type();
        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(pointer_type));
        signature.params.push(AbiParam::new(pointer_type));
        signature.params.push(AbiParam::new(pointer_type));
        signature.params.push(AbiParam::new(pointer_type));
        signature.returns.push(AbiParam::new(types::I32));

        let function_id = module
            .declare_function("xlsynth_pir_entry", Linkage::Export, &signature)
            .map_err(|error| CompilerError::Backend(error.to_string()))?;
        let mut context = module.make_context();
        context.func.signature = signature;
        let runtime_callbacks =
            declare_runtime_callbacks(&mut module, &mut context.func, pointer_type)?;

        let mut function_builder_context = FunctionBuilderContext::new();
        {
            let mut function_builder =
                FunctionBuilder::new(&mut context.func, &mut function_builder_context);
            lower_function(
                function,
                &order,
                &param_layouts,
                &scratch_plan,
                &event_sites,
                runtime_callbacks,
                pointer_type,
                &mut function_builder,
            )?;
            function_builder.finalize();
        }

        module
            .define_function(function_id, &mut context)
            .map_err(|error| CompilerError::Backend(error.to_string()))?;
        module.clear_context(&mut context);
        module
            .finalize_definitions()
            .map_err(|error| CompilerError::Backend(error.to_string()))?;

        let entrypoint_ptr = module.get_finalized_function(function_id);
        // SAFETY: `entrypoint_ptr` denotes the finalized function just defined
        // with the exact `NativeEntrypoint` signature above.
        let entrypoint: NativeEntrypoint = unsafe { std::mem::transmute(entrypoint_ptr) };
        Ok(Self {
            module: Some(module),
            entrypoint,
            param_layouts,
            result_layout,
            metadata,
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

    /// Returns static metadata for observable event sites in this function.
    pub fn metadata(&self) -> &CompiledFunctionMetadata {
        &self.metadata
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
    pub unsafe fn run_native(
        &self,
        inputs: &[*const u8],
        output: *mut u8,
    ) -> Result<(), CompilerError> {
        let mut context = ExecutionContext::new(&self.metadata);
        // SAFETY: this method forwards the caller's storage contract and owns
        // an active execution context for the duration of generated execution.
        unsafe { self.run_native_with_context(inputs, output, &mut context) }
    }

    /// Runs generated code with a caller-owned observable-event collector.
    ///
    /// # Safety
    ///
    /// Input and output pointer requirements match [`Self::run_native`].
    pub unsafe fn run_native_with_context(
        &self,
        inputs: &[*const u8],
        output: *mut u8,
        context: &mut ExecutionContext<'_>,
    ) -> Result<(), CompilerError> {
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
            self.run_native_with_scratch_and_context(
                inputs,
                output,
                scratch,
                scratch_words.len() * std::mem::size_of::<u64>(),
                context,
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
    ) -> Result<(), CompilerError> {
        let mut context = ExecutionContext::new(&self.metadata);
        // SAFETY: this method forwards the caller's native-storage contract
        // while owning an active context for the generated call.
        unsafe {
            self.run_native_with_scratch_and_context(
                inputs,
                output,
                scratch,
                scratch_byte_count,
                &mut context,
            )
        }
    }

    /// Runs generated code with caller-owned storage and an event collector.
    ///
    /// # Safety
    ///
    /// Input, output, and scratch pointer requirements match
    /// [`Self::run_native_with_scratch`].
    pub unsafe fn run_native_with_scratch_and_context(
        &self,
        inputs: &[*const u8],
        output: *mut u8,
        scratch: *mut u8,
        scratch_byte_count: usize,
        context: &mut ExecutionContext<'_>,
    ) -> Result<(), CompilerError> {
        if inputs.len() != self.param_layouts.len() {
            return Err(CompilerError::InvalidArgument(format!(
                "expected {} input pointers, got {}",
                self.param_layouts.len(),
                inputs.len()
            )));
        }
        if inputs
            .iter()
            .zip(&self.param_layouts)
            .any(|(pointer, layout)| layout.byte_count() != 0 && pointer.is_null())
            || (self.result_layout.byte_count() != 0 && output.is_null())
        {
            return Err(CompilerError::InvalidArgument(
                "nonempty native input/output pointers must be non-null".to_string(),
            ));
        }
        if scratch_byte_count < self.scratch_byte_count {
            return Err(CompilerError::InvalidArgument(format!(
                "scratch buffer has {scratch_byte_count} bytes, requires {}",
                self.scratch_byte_count
            )));
        }
        if self.scratch_byte_count != 0
            && (scratch.is_null() || (scratch as usize) % self.scratch_alignment != 0)
        {
            return Err(CompilerError::InvalidArgument(format!(
                "scratch buffer must be non-null and aligned to {} bytes",
                self.scratch_alignment
            )));
        }
        // SAFETY: the caller upholds the pointer and layout contract stated
        // above; the function was finalized with this exact ABI.
        let mut raw_context = context.raw_context();
        let status =
            unsafe { (self.entrypoint)(inputs.as_ptr(), output, scratch, &mut raw_context) };
        if status == 0 {
            Ok(())
        } else {
            Err(CompilerError::ExecutionFailed(status))
        }
    }

    /// Runs scalar compiled code through a convenient integer adapter.
    pub fn run_u64(&self, args: &[u64]) -> Result<u64, CompilerError> {
        if args.len() != self.param_layouts.len() {
            return Err(CompilerError::InvalidArgument(format!(
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
    pub fn run_ir_values(&self, args: &[IrValue]) -> Result<IrValue, CompilerError> {
        Ok(self.run_ir_values_with_events(args)?.value)
    }

    /// Runs dynamic PIR values and returns the value plus observable events.
    pub fn run_ir_values_with_events(
        &self,
        args: &[IrValue],
    ) -> Result<IrExecutionResult, CompilerError> {
        if args.len() != self.param_layouts.len() {
            return Err(CompilerError::InvalidArgument(format!(
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
        let mut context = ExecutionContext::new(&self.metadata);
        // SAFETY: each `NativeValueStorage` owns aligned storage with exactly
        // the corresponding published native layout and lives across the call.
        unsafe { self.run_native_with_context(&pointers, output.as_mut_ptr(), &mut context)? };
        Ok(IrExecutionResult {
            value: output.to_ir_value(&self.result_layout)?,
            events: context.result(),
        })
    }
}

impl Drop for PirFunctionCompiler {
    fn drop(&mut self) {
        let Some(module) = self.module.take() else {
            return;
        };
        // SAFETY: the entrypoint is private and all safe calls borrow `self`,
        // so dropping this owner cannot overlap an invocation of its code.
        unsafe { module.free_memory() };
    }
}

fn require_scalar_layout(layout: &NativeValueLayout) -> Result<ScalarLayout, CompilerError> {
    layout.as_scalar().ok_or_else(|| {
        CompilerError::UnsupportedType("this operation requires a native scalar value".into())
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

    fn from_ir_value(layout: &NativeValueLayout, value: &IrValue) -> Result<Self, CompilerError> {
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

    fn to_ir_value(&self, layout: &NativeValueLayout) -> Result<IrValue, CompilerError> {
        // SAFETY: storage remains live and was allocated for `layout`.
        unsafe { read_ir_value_from_native(self.as_ptr(), layout) }
    }
}

unsafe fn write_ir_value_to_native(
    destination: *mut u8,
    layout: &NativeValueLayout,
    value: &IrValue,
) -> Result<(), CompilerError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let bits = value
                .to_bits()
                .map_err(|error| CompilerError::Value(error.to_string()))?;
            if bits.get_bit_count() != scalar.bit_count {
                return Err(CompilerError::InvalidArgument(format!(
                    "expected bits[{}] argument, got bits[{}]",
                    scalar.bit_count,
                    bits.get_bit_count()
                )));
            }
            let integer = bits
                .to_u64()
                .map_err(|error| CompilerError::Value(error.to_string()))?;
            let bytes = integer.to_ne_bytes();
            // SAFETY: the caller provides storage for this scalar layout.
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), destination, scalar.byte_count);
            }
        }
        NativeValueLayout::WideBits(wide) => {
            let bits = value
                .to_bits()
                .map_err(|error| CompilerError::Value(error.to_string()))?;
            if bits.get_bit_count() != wide.bit_count {
                return Err(CompilerError::InvalidArgument(format!(
                    "expected bits[{}] argument, got bits[{}]",
                    wide.bit_count,
                    bits.get_bit_count()
                )));
            }
            let bytes = bits
                .to_le_bytes()
                .map_err(|error| CompilerError::Value(error.to_string()))?;
            for limb_index in 0..wide.limb_count {
                let start = limb_index * std::mem::size_of::<u64>();
                let mut limb_bytes = [0u8; std::mem::size_of::<u64>()];
                if start < bytes.len() {
                    let end = bytes.len().min(start + std::mem::size_of::<u64>());
                    limb_bytes[..end - start].copy_from_slice(&bytes[start..end]);
                }
                // SAFETY: storage is sized and aligned for the described limbs.
                unsafe {
                    destination
                        .add(start)
                        .cast::<u64>()
                        .write(u64::from_le_bytes(limb_bytes));
                }
            }
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let elements = value
                .get_elements()
                .map_err(|error| CompilerError::InvalidArgument(error.to_string()))?;
            if elements.len() != *element_count {
                return Err(CompilerError::InvalidArgument(format!(
                    "expected array with {element_count} elements, got {}",
                    elements.len()
                )));
            }
            for (index, child) in elements.iter().enumerate() {
                // SAFETY: each child lies in its recursively described region.
                unsafe {
                    write_ir_value_to_native(
                        destination.wrapping_add(index * element.byte_count()),
                        element,
                        child,
                    )?;
                }
            }
        }
        NativeValueLayout::Tuple { fields, .. } => {
            let elements = value
                .get_elements()
                .map_err(|error| CompilerError::InvalidArgument(error.to_string()))?;
            if elements.len() != fields.len() {
                return Err(CompilerError::InvalidArgument(format!(
                    "expected tuple with {} fields, got {}",
                    fields.len(),
                    elements.len()
                )));
            }
            for (field, child) in fields.iter().zip(elements.iter()) {
                // SAFETY: each field lies at its published C-compatible offset.
                unsafe {
                    write_ir_value_to_native(
                        destination.wrapping_add(field.offset),
                        field.layout.as_ref(),
                        child,
                    )?;
                }
            }
        }
        NativeValueLayout::Token => {
            // A token has no native bytes to write or inspect.
        }
    }
    Ok(())
}

unsafe fn read_ir_value_from_native(
    source: *const u8,
    layout: &NativeValueLayout,
) -> Result<IrValue, CompilerError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let mut bytes = [0u8; std::mem::size_of::<u64>()];
            // SAFETY: the caller provides readable storage for this scalar layout.
            unsafe {
                ptr::copy_nonoverlapping(source, bytes.as_mut_ptr(), scalar.byte_count);
            }
            IrValue::make_ubits(scalar.bit_count, u64::from_ne_bytes(bytes) & scalar.mask())
                .map_err(|error| CompilerError::Value(error.to_string()))
        }
        NativeValueLayout::WideBits(wide) => {
            let mut bytes = Vec::with_capacity(wide.limb_count * std::mem::size_of::<u64>());
            for limb_index in 0..wide.limb_count {
                // SAFETY: storage is sized and aligned for the described limbs.
                let limb = unsafe {
                    source
                        .add(limb_index * std::mem::size_of::<u64>())
                        .cast::<u64>()
                        .read()
                };
                bytes.extend_from_slice(&limb.to_le_bytes());
            }
            bytes.truncate(wide.bit_count.div_ceil(8));
            let high_remainder = wide.bit_count % 8;
            if high_remainder != 0 {
                let mask = (1u8 << high_remainder) - 1;
                *bytes.last_mut().expect("wide bits has bytes") &= mask;
            }
            IrBits::from_le_bytes(wide.bit_count, &bytes)
                .map(|bits| IrValue::from_bits(&bits))
                .map_err(|error| CompilerError::Value(error.to_string()))
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let mut elements = Vec::with_capacity(*element_count);
            for index in 0..*element_count {
                // SAFETY: each child lies in its recursively described region.
                elements.push(unsafe {
                    read_ir_value_from_native(
                        source.wrapping_add(index * element.byte_count()),
                        element,
                    )?
                });
            }
            IrValue::make_array(&elements).map_err(|error| CompilerError::Value(error.to_string()))
        }
        NativeValueLayout::Tuple { fields, .. } => {
            let mut elements = Vec::with_capacity(fields.len());
            for field in fields {
                // SAFETY: each field lies at its published C-compatible offset.
                elements.push(unsafe {
                    read_ir_value_from_native(
                        source.wrapping_add(field.offset),
                        field.layout.as_ref(),
                    )?
                });
            }
            Ok(IrValue::make_tuple(&elements))
        }
        NativeValueLayout::Token => Ok(IrValue::make_token()),
    }
}

impl NativeScalar {
    fn new(layout: ScalarLayout, value: u64) -> Result<Self, CompilerError> {
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
    in_place_array_updates: HashSet<NodeRef>,
    gate_zero_offset: Option<usize>,
    gate_zero_byte_count: usize,
    trace_sites: HashMap<NodeRef, TraceScratchPlan>,
    runtime_scalar_offsets: HashMap<NodeRef, usize>,
    runtime_temporary_offsets: [Option<usize>; 2],
    byte_count: usize,
    alignment: usize,
}

#[derive(Debug)]
struct TraceScratchPlan {
    pointer_array_offset: usize,
    scalar_operand_offsets: Vec<Option<usize>>,
}

#[derive(Debug, Clone, Copy)]
struct ScratchSlot {
    offset: usize,
    byte_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct ActiveScratchSlot {
    slot: ScratchSlot,
    last_use: usize,
}

impl ScratchPlan {
    /// Assigns reusable native scratch slots for materialized intermediates.
    fn for_function(function: &ir::Fn, order: &[NodeRef]) -> Result<Self, CompilerError> {
        let (offsets, in_place_array_updates, mut byte_count, mut alignment) =
            assign_materialized_scratch_offsets(function, order)?;
        let (gate_zero_offset, gate_zero_byte_count) =
            reserve_gate_zero_storage(function, order, &mut byte_count, &mut alignment)?;
        let mut trace_sites = HashMap::new();
        let mut runtime_scalar_offsets = HashMap::new();
        let mut runtime_temporary_offsets = [None; 2];
        for node_ref in order {
            let NodePayload::Trace { operands, .. } = &function.get_node(*node_ref).payload else {
                continue;
            };
            let mut scalar_operand_offsets = Vec::with_capacity(operands.len());
            for operand in operands {
                let layout = NativeValueLayout::from_type(&function.get_node(*operand).ty)?;
                if matches!(layout, NativeValueLayout::Scalar(_)) {
                    byte_count = align_up(byte_count, layout.alignment())?;
                    scalar_operand_offsets.push(Some(byte_count));
                    byte_count = byte_count.checked_add(layout.byte_count()).ok_or_else(|| {
                        CompilerError::UnsupportedType("trace scratch size overflow".into())
                    })?;
                    alignment = alignment.max(layout.alignment());
                } else {
                    scalar_operand_offsets.push(None);
                }
            }
            let pointer_alignment = std::mem::align_of::<*const u8>();
            byte_count = align_up(byte_count, pointer_alignment)?;
            let pointer_array_offset = byte_count;
            byte_count = byte_count
                .checked_add(operands.len() * std::mem::size_of::<*const u8>())
                .ok_or_else(|| {
                    CompilerError::UnsupportedType("trace pointer size overflow".into())
                })?;
            alignment = alignment.max(pointer_alignment);
            trace_sites.insert(
                *node_ref,
                TraceScratchPlan {
                    pointer_array_offset,
                    scalar_operand_offsets,
                },
            );
        }
        if function_uses_runtime_bits_callback(function, order)? {
            for node_ref in order {
                let NativeValueLayout::Scalar(_) =
                    NativeValueLayout::from_type(&function.get_node(*node_ref).ty)?
                else {
                    continue;
                };
                byte_count = align_up(byte_count, std::mem::align_of::<u64>())?;
                runtime_scalar_offsets.insert(*node_ref, byte_count);
                byte_count = byte_count
                    .checked_add(std::mem::size_of::<u64>())
                    .ok_or_else(|| {
                        CompilerError::UnsupportedType("runtime scratch size overflow".into())
                    })?;
                alignment = alignment.max(std::mem::align_of::<u64>());
            }
            for offset in &mut runtime_temporary_offsets {
                byte_count = align_up(byte_count, std::mem::align_of::<u64>())?;
                *offset = Some(byte_count);
                byte_count = byte_count
                    .checked_add(std::mem::size_of::<u64>())
                    .ok_or_else(|| {
                        CompilerError::UnsupportedType("runtime temporary size overflow".into())
                    })?;
                alignment = alignment.max(std::mem::align_of::<u64>());
            }
        }
        Ok(Self {
            offsets,
            in_place_array_updates,
            gate_zero_offset,
            gate_zero_byte_count,
            trace_sites,
            runtime_scalar_offsets,
            runtime_temporary_offsets,
            byte_count,
            alignment,
        })
    }
}

/// Assigns overlapping scratch offsets to materialized values with disjoint
/// lifetimes.
fn assign_materialized_scratch_offsets(
    function: &ir::Fn,
    order: &[NodeRef],
) -> Result<(HashMap<NodeRef, usize>, HashSet<NodeRef>, usize, usize), CompilerError> {
    let return_node = function.ret_node_ref.ok_or_else(|| {
        CompilerError::InvalidFunction(format!("function '{}' has no return node", function.name))
    })?;
    let positions = order
        .iter()
        .enumerate()
        .map(|(position, node_ref)| (*node_ref, position))
        .collect::<HashMap<_, _>>();
    let direct_last_uses = direct_last_uses(function, order, return_node);
    let (materialized, in_place_array_updates, owners) =
        plan_scratch_storage_owners(function, order, return_node, &direct_last_uses)?;
    let mut last_uses = materialized
        .iter()
        .map(|node_ref| (*node_ref, positions[node_ref]))
        .collect::<HashMap<_, _>>();
    for (position, node_ref) in order.iter().enumerate() {
        for operand in operands(&function.get_node(*node_ref).payload) {
            for owner in owners.get(&operand).into_iter().flatten() {
                let owner_last_use =
                    if aggregate_construction_preloads_scalar_operand(function, *node_ref, operand)
                    {
                        position.saturating_sub(1)
                    } else {
                        position
                    };
                last_uses
                    .entry(*owner)
                    .and_modify(|last_use| *last_use = (*last_use).max(owner_last_use));
            }
        }
    }
    for owner in owners.get(&return_node).into_iter().flatten() {
        last_uses
            .entry(*owner)
            .and_modify(|last_use| *last_use = (*last_use).max(order.len()));
    }

    let mut offsets = HashMap::new();
    let mut active = Vec::<ActiveScratchSlot>::new();
    let mut free = Vec::<ScratchSlot>::new();
    let mut byte_count = 0usize;
    let mut alignment = 1usize;
    for (position, node_ref) in order.iter().copied().enumerate() {
        if !materialized.contains(&node_ref) {
            continue;
        }
        let mut still_active = Vec::with_capacity(active.len());
        for allocation in active.drain(..) {
            if allocation.last_use < position {
                free.push(allocation.slot);
            } else {
                still_active.push(allocation);
            }
        }
        active = still_active;

        let node = function.get_node(node_ref);
        let layout = NativeValueLayout::from_type(&node.ty)?;
        let layout_alignment = layout.alignment();
        let layout_byte_count = layout.byte_count();
        let reusable = free
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                slot.byte_count >= layout_byte_count && slot.offset % layout_alignment == 0
            })
            .min_by_key(|(_, slot)| slot.byte_count)
            .map(|(index, _)| index);
        let slot = if let Some(index) = reusable {
            free.swap_remove(index)
        } else {
            byte_count = align_up(byte_count, layout_alignment)?;
            let slot = ScratchSlot {
                offset: byte_count,
                byte_count: layout_byte_count,
            };
            byte_count = byte_count.checked_add(layout_byte_count).ok_or_else(|| {
                CompilerError::UnsupportedType(format!("scratch size overflow for {}", node.ty))
            })?;
            slot
        };
        offsets.insert(node_ref, slot.offset);
        active.push(ActiveScratchSlot {
            slot,
            last_use: last_uses[&node_ref],
        });
        alignment = alignment.max(layout_alignment);
    }
    Ok((offsets, in_place_array_updates, byte_count, alignment))
}

/// Returns whether lowering reads a scalar operand before writing aggregate
/// destination storage, allowing its backing scratch slot to be reused.
fn aggregate_construction_preloads_scalar_operand(
    function: &ir::Fn,
    node_ref: NodeRef,
    operand: NodeRef,
) -> bool {
    matches!(
        function.get_node(node_ref).payload,
        NodePayload::Array(_) | NodePayload::Tuple(_)
    ) && matches!(
        NativeValueLayout::from_type(&function.get_node(operand).ty),
        Ok(NativeValueLayout::Scalar(_))
    )
}

/// Returns the final use position for each node value.
fn direct_last_uses(
    function: &ir::Fn,
    order: &[NodeRef],
    return_node: NodeRef,
) -> HashMap<NodeRef, usize> {
    let mut last_uses = order
        .iter()
        .enumerate()
        .map(|(position, node_ref)| (*node_ref, position))
        .collect::<HashMap<_, _>>();
    for (position, node_ref) in order.iter().enumerate() {
        for operand in operands(&function.get_node(*node_ref).payload) {
            last_uses
                .entry(operand)
                .and_modify(|last_use| *last_use = (*last_use).max(position));
        }
    }
    last_uses
        .entry(return_node)
        .and_modify(|last_use| *last_use = (*last_use).max(order.len()));
    last_uses
}

/// Resolves scratch allocations and storage aliases, including array updates
/// that can consume a dead scratch-owned input in place.
fn plan_scratch_storage_owners(
    function: &ir::Fn,
    order: &[NodeRef],
    return_node: NodeRef,
    direct_last_uses: &HashMap<NodeRef, usize>,
) -> Result<
    (
        HashSet<NodeRef>,
        HashSet<NodeRef>,
        HashMap<NodeRef, Vec<NodeRef>>,
    ),
    CompilerError,
> {
    let mut materialized = HashSet::new();
    let mut in_place_array_updates = HashSet::new();
    let mut owners = HashMap::<NodeRef, Vec<NodeRef>>::new();
    for (position, node_ref) in order.iter().enumerate() {
        let node = function.get_node(*node_ref);
        let layout = NativeValueLayout::from_type(&node.ty)?;
        let can_update_in_place = match &node.payload {
            NodePayload::ArrayUpdate { array, indices, .. } if !indices.is_empty() => {
                let source_owners = owners.get(array).cloned().unwrap_or_default();
                source_owners.len() == 1
                    && scratch_owner_is_dead_after_position(
                        source_owners[0],
                        position,
                        &owners,
                        direct_last_uses,
                    )
            }
            _ => false,
        };
        if can_update_in_place {
            in_place_array_updates.insert(*node_ref);
        } else if *node_ref != return_node && needs_materialized_destination(node, &layout) {
            materialized.insert(*node_ref);
        }
        let mut node_owners = if materialized.contains(node_ref) {
            vec![*node_ref]
        } else if can_update_in_place {
            let NodePayload::ArrayUpdate { array, .. } = &node.payload else {
                unreachable!("in-place update decision requires array_update")
            };
            owners.get(array).cloned().unwrap_or_default()
        } else if layout.is_memory_backed() || scalar_storage_view_payload(&node.payload) {
            aliased_storage_operands(&node.payload)
                .into_iter()
                .flat_map(|operand| owners.get(&operand).into_iter().flatten().copied())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        node_owners.sort_by_key(|owner| owner.index);
        node_owners.dedup();
        owners.insert(*node_ref, node_owners);
    }
    Ok((materialized, in_place_array_updates, owners))
}

/// Returns whether no existing view of a scratch owner survives an update.
fn scratch_owner_is_dead_after_position(
    owner: NodeRef,
    position: usize,
    owners: &HashMap<NodeRef, Vec<NodeRef>>,
    direct_last_uses: &HashMap<NodeRef, usize>,
) -> bool {
    owners.iter().all(|(node_ref, node_owners)| {
        !node_owners.contains(&owner) || direct_last_uses[node_ref] <= position
    })
}

/// Returns whether a scalar result can preserve a view into backing storage.
fn scalar_storage_view_payload(payload: &NodePayload) -> bool {
    matches!(
        payload,
        NodePayload::Unop(Unop::Identity, _)
            | NodePayload::ArrayIndex { .. }
            | NodePayload::TupleIndex { .. }
    )
}

/// Returns operands whose backing storage is preserved by a memory-backed view.
fn aliased_storage_operands(payload: &NodePayload) -> Vec<NodeRef> {
    match payload {
        NodePayload::GetParam(_) => Vec::new(),
        NodePayload::Unop(Unop::Identity, arg) => vec![*arg],
        NodePayload::ArrayIndex { array, .. } => vec![*array],
        NodePayload::TupleIndex { tuple, .. } => vec![*tuple],
        NodePayload::ArrayUpdate { value, indices, .. } if indices.is_empty() => vec![*value],
        NodePayload::Binop(Binop::Gate, _, gated) => vec![*gated],
        NodePayload::Sel { cases, default, .. }
        | NodePayload::PrioritySel { cases, default, .. } => {
            let mut operands = cases.clone();
            operands.extend(default.iter().copied());
            operands
        }
        _ => Vec::new(),
    }
}

/// Reserves one shared zero buffer used by memory-backed gate views.
fn reserve_gate_zero_storage(
    function: &ir::Fn,
    order: &[NodeRef],
    byte_count: &mut usize,
    alignment: &mut usize,
) -> Result<(Option<usize>, usize), CompilerError> {
    let mut gate_zero_byte_count = 0usize;
    let mut gate_zero_alignment = 1usize;
    for node_ref in order {
        let node = function.get_node(*node_ref);
        if !matches!(node.payload, NodePayload::Binop(Binop::Gate, _, _)) {
            continue;
        }
        let layout = NativeValueLayout::from_type(&node.ty)?;
        if layout.is_memory_backed() {
            gate_zero_byte_count = gate_zero_byte_count.max(layout.byte_count());
            gate_zero_alignment = gate_zero_alignment.max(layout.alignment());
        }
    }
    if gate_zero_byte_count == 0 {
        return Ok((None, 0));
    }
    *byte_count = align_up(*byte_count, gate_zero_alignment)?;
    let offset = *byte_count;
    *byte_count = (*byte_count)
        .checked_add(gate_zero_byte_count)
        .ok_or_else(|| CompilerError::UnsupportedType("gate zero scratch size overflow".into()))?;
    *alignment = (*alignment).max(gate_zero_alignment);
    Ok((Some(offset), gate_zero_byte_count))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, CompilerError> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| CompilerError::UnsupportedType("native layout size overflow".into()))
}

fn needs_materialized_destination(node: &ir::Node, layout: &NativeValueLayout) -> bool {
    match layout {
        NativeValueLayout::WideBits(_) => !matches!(
            &node.payload,
            NodePayload::GetParam(_)
                | NodePayload::Unop(Unop::Identity, _)
                | NodePayload::ArrayIndex { .. }
                | NodePayload::TupleIndex { .. }
                | NodePayload::Binop(Binop::Gate, _, _)
                | NodePayload::Sel { .. }
                | NodePayload::PrioritySel { .. }
        ),
        NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. } => match &node.payload {
            NodePayload::ArrayUpdate { indices, .. } => !indices.is_empty(),
            NodePayload::Literal(_)
            | NodePayload::Array(_)
            | NodePayload::Tuple(_)
            | NodePayload::ArrayConcat(_)
            | NodePayload::ArraySlice { .. }
            | NodePayload::Binop(Binop::Umulp | Binop::Smulp, _, _)
            | NodePayload::OneHotSel { .. }
            | NodePayload::ExtNormalizeLeft { .. } => true,
            _ => false,
        },
        NativeValueLayout::Scalar(_) | NativeValueLayout::Token => false,
    }
}

#[derive(Clone)]
enum ComputedValue {
    Scalar(Value),
    ScalarAddress {
        pointer: Value,
        offset: usize,
        layout: ScalarLayout,
    },
    ScalarArrayIndex(Box<DeferredScalarArrayIndex>),
    Address(Value),
    ZeroSized,
}

#[derive(Clone)]
struct DeferredScalarArrayIndex {
    array_pointer: Value,
    dimensions: Vec<DeferredArrayIndexDimension>,
    layout: ScalarLayout,
    assumption_site: Option<DeferredAssumptionSite>,
}

#[derive(Clone)]
struct DeferredArrayIndexDimension {
    index: ComputedValue,
    index_layout: NativeValueLayout,
    element_count: usize,
    element_byte_count: usize,
    statically_in_bounds: bool,
}

#[derive(Clone, Copy)]
struct DeferredAssumptionSite {
    callback: FuncRef,
    execution_context: Value,
    site_id: u32,
}

#[derive(Clone, Copy)]
struct RuntimeCallbacks {
    record_assert: FuncRef,
    record_assumption_failure: FuncRef,
    record_cover: FuncRef,
    record_trace: FuncRef,
    wide_binop: FuncRef,
    wide_bit_slice_update: FuncRef,
    wide_unary_op: FuncRef,
    wide_mulp: FuncRef,
}

fn trace_layout_from_native(layout: &NativeValueLayout) -> TraceValueLayout {
    match layout {
        NativeValueLayout::Scalar(scalar) => TraceValueLayout::Bits {
            bit_count: scalar.bit_count,
            byte_count: scalar.byte_count,
        },
        NativeValueLayout::WideBits(wide) => TraceValueLayout::WideBits {
            bit_count: wide.bit_count,
            limb_count: wide.limb_count,
        },
        NativeValueLayout::Array {
            element,
            element_count,
        } => TraceValueLayout::Array {
            element: Box::new(trace_layout_from_native(element)),
            element_count: *element_count,
        },
        NativeValueLayout::Tuple {
            fields, byte_count, ..
        } => TraceValueLayout::Tuple {
            fields: fields
                .iter()
                .map(|field| TraceTupleFieldLayout {
                    layout: Box::new(trace_layout_from_native(&field.layout)),
                    offset: field.offset,
                })
                .collect(),
            byte_count: *byte_count,
        },
        NativeValueLayout::Token => TraceValueLayout::Token,
    }
}

/// Returns a literal bits value as `usize` when it fits in the host address
/// space.
fn literal_bits_as_usize(value: &IrValue) -> Option<usize> {
    let bytes = value.to_bits().ok()?.to_le_bytes().ok()?;
    let mut result = 0usize;
    for (index, byte) in bytes.into_iter().enumerate() {
        if index >= std::mem::size_of::<usize>() {
            if byte != 0 {
                return None;
            }
            continue;
        }
        result |= usize::from(byte) << (index * 8);
    }
    Some(result)
}

/// Returns whether an index node is provably below an array dimension from its
/// literal value or its bits type.
fn bits_node_is_statically_less_than(
    function: &ir::Fn,
    node_ref: NodeRef,
    upper_bound: usize,
) -> bool {
    if upper_bound == 0 {
        return false;
    }
    let node = function.get_node(node_ref);
    if let NodePayload::Literal(value) = &node.payload {
        return literal_bits_as_usize(value).is_some_and(|value| value < upper_bound);
    }
    let Type::Bits(bit_count) = &node.ty else {
        return false;
    };
    *bit_count < usize::BITS as usize && (1usize << *bit_count) <= upper_bound
}

/// Returns whether every index in a multidimensional array access is
/// statically guaranteed to be in bounds.
fn array_indices_are_statically_in_bounds(
    function: &ir::Fn,
    array: NodeRef,
    indices: &[NodeRef],
) -> bool {
    let mut ty = &function.get_node(array).ty;
    for index in indices {
        let Type::Array(array) = ty else {
            return false;
        };
        if !bits_node_is_statically_less_than(function, *index, array.element_count) {
            return false;
        }
        ty = array.element_type.as_ref();
    }
    true
}

fn build_event_metadata(
    function: &ir::Fn,
    order: &[NodeRef],
) -> Result<(CompiledFunctionMetadata, HashMap<NodeRef, u32>), CompilerError> {
    let mut event_sites = Vec::new();
    let mut site_ids = HashMap::new();
    for node_ref in order {
        let node = function.get_node(*node_ref);
        let site = match &node.payload {
            NodePayload::Cover { label, .. } => Some(EventSiteMetadata {
                node_text_id: node.text_id,
                kind: EventKind::Cover,
                label: Some(label.clone()),
                message: None,
                format: None,
                operand_layouts: Vec::new(),
            }),
            NodePayload::Assert { message, label, .. } => Some(EventSiteMetadata {
                node_text_id: node.text_id,
                kind: EventKind::Assert,
                label: Some(label.clone()),
                message: Some(message.clone()),
                format: None,
                operand_layouts: Vec::new(),
            }),
            NodePayload::ArrayIndex {
                array,
                indices,
                assumed_in_bounds: true,
            } if !array_indices_are_statically_in_bounds(function, *array, indices) => {
                Some(EventSiteMetadata {
                    node_text_id: node.text_id,
                    kind: EventKind::Assumption(AssumptionFailureKind::ArrayIndexOutOfBounds),
                    label: None,
                    message: None,
                    format: None,
                    operand_layouts: Vec::new(),
                })
            }
            NodePayload::ArrayUpdate {
                array,
                indices,
                assumed_in_bounds: true,
                ..
            } if !array_indices_are_statically_in_bounds(function, *array, indices) => {
                Some(EventSiteMetadata {
                    node_text_id: node.text_id,
                    kind: EventKind::Assumption(AssumptionFailureKind::ArrayUpdateOutOfBounds),
                    label: None,
                    message: None,
                    format: None,
                    operand_layouts: Vec::new(),
                })
            }
            NodePayload::ArrayIndex {
                assumed_in_bounds: true,
                ..
            }
            | NodePayload::ArrayUpdate {
                assumed_in_bounds: true,
                ..
            } => None,
            NodePayload::Trace {
                format, operands, ..
            } => Some(EventSiteMetadata {
                node_text_id: node.text_id,
                kind: EventKind::Trace,
                label: None,
                message: None,
                format: Some(format.clone()),
                operand_layouts: operands
                    .iter()
                    .map(|operand| {
                        NativeValueLayout::from_type(&function.get_node(*operand).ty)
                            .map(|layout| trace_layout_from_native(&layout))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            }),
            _ => None,
        };
        if let Some(site) = site {
            let site_id = u32::try_from(event_sites.len())
                .map_err(|_| CompilerError::UnsupportedType("too many event sites".into()))?;
            site_ids.insert(*node_ref, site_id);
            event_sites.push(site);
        }
    }
    Ok((CompiledFunctionMetadata { event_sites }, site_ids))
}

/// Returns the reachable nodes in a pressure-aware topological lowering order.
fn reachable_scheduled_order(function: &ir::Fn) -> Result<Vec<NodeRef>, CompilerError> {
    let return_node = function.ret_node_ref.ok_or_else(|| {
        CompilerError::InvalidFunction(format!("function '{}' has no return node", function.name))
    })?;
    let mut stack = vec![return_node];
    stack.extend(
        function
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                is_observable_effect_root(&node.payload)
                    || match &node.payload {
                        NodePayload::ArrayIndex {
                            array,
                            indices,
                            assumed_in_bounds: true,
                        }
                        | NodePayload::ArrayUpdate {
                            array,
                            indices,
                            assumed_in_bounds: true,
                            ..
                        } => !array_indices_are_statically_in_bounds(function, *array, indices),
                        _ => false,
                    }
            })
            .map(|(index, _)| NodeRef { index }),
    );
    let mut reachable = vec![false; function.nodes.len()];
    while let Some(node_ref) = stack.pop() {
        if node_ref.index >= function.nodes.len() {
            return Err(CompilerError::InvalidFunction(format!(
                "node reference {} is out of bounds",
                node_ref.index
            )));
        }
        if !reachable[node_ref.index] {
            reachable[node_ref.index] = true;
            stack.extend(operands(&function.get_node(node_ref).payload));
        }
    }
    let reachable_count = reachable
        .iter()
        .filter(|is_reachable| **is_reachable)
        .count();
    let dependencies = function
        .nodes
        .iter()
        .enumerate()
        .map(|(index, _)| {
            if reachable[index] {
                unique_operands(function, NodeRef { index })
            } else {
                Vec::new()
            }
        })
        .collect::<Vec<_>>();
    let mut users = vec![Vec::<NodeRef>::new(); function.nodes.len()];
    for (index, is_reachable) in reachable.iter().copied().enumerate() {
        if !is_reachable {
            continue;
        }
        for operand in &dependencies[index] {
            users[operand.index].push(NodeRef { index });
        }
    }
    let mut remaining_dependency_count = dependencies.iter().map(Vec::len).collect::<Vec<_>>();
    let mut remaining_user_count = users.iter().map(Vec::len).collect::<Vec<_>>();
    let is_scalar = function
        .nodes
        .iter()
        .map(|node| scheduling_is_scalar(&node.ty))
        .collect::<Result<Vec<_>, _>>()?;
    let scheduling_priority = |node_ref: NodeRef, remaining_user_count: &[usize]| {
        let scalar_operand_count = dependencies[node_ref.index]
            .iter()
            .filter(|operand| is_scalar[operand.index])
            .count();
        let scalars_freed = dependencies[node_ref.index]
            .iter()
            .filter(|operand| is_scalar[operand.index] && remaining_user_count[operand.index] == 1)
            .count();
        let scalar_result_allocated = usize::from(
            is_scalar[node_ref.index]
                && (remaining_user_count[node_ref.index] != 0 || node_ref == return_node),
        );
        (
            scheduling_gain(scalars_freed, scalar_result_allocated),
            scalar_operand_count,
            Reverse(node_ref.index),
        )
    };
    let mut eligible = reachable
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, is_reachable)| *is_reachable && remaining_dependency_count[*index] == 0)
        .map(|(index, _)| scheduling_priority(NodeRef { index }, &remaining_user_count))
        .collect::<BinaryHeap<_>>();
    let mut scheduled = vec![false; function.nodes.len()];
    let mut order = Vec::with_capacity(reachable_count);
    while let Some(priority @ (_, _, Reverse(index))) = eligible.pop() {
        if scheduled[index] {
            continue;
        }
        let node_ref = NodeRef { index };
        let current_priority = scheduling_priority(node_ref, &remaining_user_count);
        if priority != current_priority {
            eligible.push(current_priority);
            continue;
        }
        scheduled[index] = true;
        order.push(node_ref);
        for operand in &dependencies[index] {
            remaining_user_count[operand.index] -= 1;
            if is_scalar[operand.index] && remaining_user_count[operand.index] == 1 {
                for user in &users[operand.index] {
                    if !scheduled[user.index] && remaining_dependency_count[user.index] == 0 {
                        eligible.push(scheduling_priority(*user, &remaining_user_count));
                    }
                }
            }
        }
        for user in &users[index] {
            remaining_dependency_count[user.index] -= 1;
            if remaining_dependency_count[user.index] == 0 {
                eligible.push(scheduling_priority(*user, &remaining_user_count));
            }
        }
    }
    if order.len() != reachable_count {
        return Err(CompilerError::InvalidFunction(format!(
            "function '{}' contains a dependency cycle",
            function.name
        )));
    }
    Ok(order)
}

/// Returns the estimated scalar-register pressure reduction from one node.
fn scheduling_gain(scalars_freed: usize, scalar_results_allocated: usize) -> i128 {
    let scalars_freed = i128::try_from(scalars_freed).unwrap_or(i128::MAX);
    let scalar_results_allocated = i128::try_from(scalar_results_allocated).unwrap_or(i128::MAX);
    scalars_freed.saturating_sub(scalar_results_allocated)
}

/// Returns the distinct operands of `node_ref` in stable operand order.
fn unique_operands(function: &ir::Fn, node_ref: NodeRef) -> Vec<NodeRef> {
    let mut seen = HashSet::new();
    operands(&function.get_node(node_ref).payload)
        .into_iter()
        .filter(|operand| seen.insert(*operand))
        .collect()
}

/// Returns whether a PIR value occupies one scalar Cranelift SSA register.
fn scheduling_is_scalar(ty: &Type) -> Result<bool, CompilerError> {
    let layout = NativeValueLayout::from_type(ty)?;
    Ok(matches!(layout, NativeValueLayout::Scalar(_)))
}

fn declare_runtime_callbacks<M: Module>(
    module: &mut M,
    function: &mut cranelift_codegen::ir::Function,
    pointer_type: ClifType,
) -> Result<RuntimeCallbacks, CompilerError> {
    let mut site_signature = module.make_signature();
    site_signature.params.push(AbiParam::new(pointer_type));
    site_signature.params.push(AbiParam::new(types::I32));
    let assert_id = module
        .declare_function(
            "xlsynth_pir_record_assert",
            Linkage::Import,
            &site_signature,
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    let assumption_failure_id = module
        .declare_function(
            "xlsynth_pir_record_assumption_failure",
            Linkage::Import,
            &site_signature,
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    let cover_id = module
        .declare_function("xlsynth_pir_record_cover", Linkage::Import, &site_signature)
        .map_err(|error| CompilerError::Backend(error.to_string()))?;

    let mut trace_signature = site_signature;
    trace_signature.params.push(AbiParam::new(pointer_type));
    let trace_id = module
        .declare_function(
            "xlsynth_pir_record_trace",
            Linkage::Import,
            &trace_signature,
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;

    let mut wide_binop_signature = module.make_signature();
    wide_binop_signature.params.extend([
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(types::I32),
    ]);
    let wide_binop_id = module
        .declare_function(
            "xlsynth_pir_runtime_wide_binop",
            Linkage::Import,
            &wide_binop_signature,
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;

    let mut bit_slice_update_signature = module.make_signature();
    for _ in 0..8 {
        bit_slice_update_signature
            .params
            .push(AbiParam::new(pointer_type));
    }
    let wide_bit_slice_update_id = module
        .declare_function(
            "xlsynth_pir_runtime_wide_bit_slice_update",
            Linkage::Import,
            &bit_slice_update_signature,
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    let mut unary_op_signature = module.make_signature();
    unary_op_signature.params.extend([
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(types::I32),
        AbiParam::new(pointer_type),
    ]);
    let wide_unary_op_id = module
        .declare_function(
            "xlsynth_pir_runtime_wide_unary_op",
            Linkage::Import,
            &unary_op_signature,
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    let mut mulp_signature = module.make_signature();
    mulp_signature.params.extend([
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(types::I32),
    ]);
    let wide_mulp_id = module
        .declare_function(
            "xlsynth_pir_runtime_wide_mulp",
            Linkage::Import,
            &mulp_signature,
        )
        .map_err(|error| CompilerError::Backend(error.to_string()))?;
    Ok(RuntimeCallbacks {
        record_assert: module.declare_func_in_func(assert_id, function),
        record_assumption_failure: module.declare_func_in_func(assumption_failure_id, function),
        record_cover: module.declare_func_in_func(cover_id, function),
        record_trace: module.declare_func_in_func(trace_id, function),
        wide_binop: module.declare_func_in_func(wide_binop_id, function),
        wide_bit_slice_update: module.declare_func_in_func(wide_bit_slice_update_id, function),
        wide_unary_op: module.declare_func_in_func(wide_unary_op_id, function),
        wide_mulp: module.declare_func_in_func(wide_mulp_id, function),
    })
}

fn lower_function(
    function: &ir::Fn,
    order: &[NodeRef],
    param_layouts: &[NativeValueLayout],
    scratch_plan: &ScratchPlan,
    event_sites: &HashMap<NodeRef, u32>,
    runtime_callbacks: RuntimeCallbacks,
    pointer_type: ClifType,
    builder: &mut FunctionBuilder<'_>,
) -> Result<(), CompilerError> {
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    let inputs_pointer = builder.block_params(entry)[0];
    let output_pointer = builder.block_params(entry)[1];
    let scratch_pointer = builder.block_params(entry)[2];
    let execution_context = builder.block_params(entry)[3];
    if let Some(offset) = scratch_plan.gate_zero_offset {
        let gate_zero_pointer = pointer_at_offset(builder, scratch_pointer, offset);
        emit_memory_zero(
            builder,
            gate_zero_pointer,
            scratch_plan.gate_zero_byte_count,
        )?;
    }
    let return_node = function.ret_node_ref.ok_or_else(|| {
        CompilerError::InvalidFunction(format!("function '{}' has no return node", function.name))
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
                    CompilerError::InvalidFunction(format!(
                        "unknown parameter id in {}",
                        node.text_id
                    ))
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
                NativeValueLayout::Token => ComputedValue::ZeroSized,
                NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. }
                    if layout.byte_count() == 0 =>
                {
                    ComputedValue::ZeroSized
                }
                NativeValueLayout::WideBits(_)
                | NativeValueLayout::Array { .. }
                | NativeValueLayout::Tuple { .. } => {
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
            NodePayload::AfterAll(_) => ComputedValue::ZeroSized,
            NodePayload::Cover { predicate, .. } => {
                let site_id = event_site_id(event_sites, *node_ref, node)?;
                let predicate = scalar_value_for(builder, &mut values, *predicate)?;
                emit_conditional_site_call(
                    builder,
                    predicate,
                    runtime_callbacks.record_cover,
                    execution_context,
                    site_id,
                );
                ComputedValue::ZeroSized
            }
            NodePayload::Assert {
                activate,
                token: _,
                message: _,
                label: _,
            } => {
                let site_id = event_site_id(event_sites, *node_ref, node)?;
                let condition = scalar_value_for(builder, &mut values, *activate)?;
                let failed = builder.ins().icmp_imm(IntCC::Equal, condition, 0);
                emit_conditional_site_call(
                    builder,
                    failed,
                    runtime_callbacks.record_assert,
                    execution_context,
                    site_id,
                );
                ComputedValue::ZeroSized
            }
            NodePayload::Trace {
                activated,
                operands,
                token: _,
                format: _,
            } => {
                let site_id = event_site_id(event_sites, *node_ref, node)?;
                let operand_pointers = lower_trace_operand_pointers(
                    builder,
                    function,
                    *node_ref,
                    operands,
                    &values,
                    scratch_pointer,
                    scratch_plan,
                    pointer_type,
                )?;
                let activated = scalar_value_for(builder, &mut values, *activated)?;
                emit_conditional_trace_call(
                    builder,
                    activated,
                    runtime_callbacks.record_trace,
                    execution_context,
                    site_id,
                    operand_pointers,
                );
                ComputedValue::ZeroSized
            }
            NodePayload::Unop(Unop::Identity, arg) if layout.is_memory_backed() => {
                computed_value_for(&values, *arg)?
            }
            NodePayload::Unop(op @ (Unop::Not | Unop::Neg | Unop::Reverse), arg)
                if matches!(&layout, NativeValueLayout::WideBits(_)) =>
            {
                let destination = materialized_destination(
                    builder,
                    *node_ref,
                    return_node,
                    output_pointer,
                    scratch_pointer,
                    scratch_plan,
                )?;
                lower_wide_bitwise_unop(
                    builder,
                    destination,
                    computed_value_for(&values, *arg)?,
                    &NativeValueLayout::from_type(&function.get_node(*arg).ty)?,
                    expect_wide_layout(&layout)?,
                    *op,
                )?;
                ComputedValue::Address(destination)
            }
            NodePayload::Unop(op @ (Unop::OrReduce | Unop::AndReduce | Unop::XorReduce), arg)
                if matches!(
                    NativeValueLayout::from_type(&function.get_node(*arg).ty)?,
                    NativeValueLayout::WideBits(_)
                ) =>
            {
                let NativeValueLayout::WideBits(arg_layout) =
                    NativeValueLayout::from_type(&function.get_node(*arg).ty)?
                else {
                    unreachable!("guard checks wide argument")
                };
                ComputedValue::Scalar(lower_wide_reduction(
                    builder,
                    computed_value_for(&values, *arg)?,
                    arg_layout,
                    *op,
                ))
            }
            NodePayload::Unop(op, arg) => {
                let scalar_layout = require_scalar_layout(&layout)?;
                let arg_value = scalar_value_for(builder, &mut values, *arg)?;
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
                if let NativeValueLayout::WideBits(wide) = &layout {
                    let destination = materialized_destination(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                    )?;
                    if *op == NaryOp::Concat {
                        lower_wide_concat(builder, function, destination, args, &values, *wide)?;
                    } else {
                        lower_wide_nary(builder, destination, args, &values, *wide, *op)?;
                    }
                    ComputedValue::Address(destination)
                } else {
                    let scalar_layout = require_scalar_layout(&layout)?;
                    ComputedValue::Scalar(lower_nary(
                        builder,
                        function,
                        node,
                        *op,
                        args,
                        &mut values,
                        scalar_layout,
                    )?)
                }
            }
            NodePayload::ArrayConcat(args) => {
                if layout.byte_count() == 0 {
                    ComputedValue::ZeroSized
                } else {
                    let destination = materialized_destination(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                    )?;
                    lower_array_concat(
                        builder,
                        function,
                        destination,
                        args,
                        &mut values,
                        &layout,
                        *node_ref != return_node,
                    )?;
                    ComputedValue::Address(destination)
                }
            }
            NodePayload::Binop(Binop::Gate, predicate, gated) => {
                let predicate = scalar_value_for(builder, &mut values, *predicate)?;
                lower_gate(
                    builder,
                    scratch_pointer,
                    scratch_plan,
                    predicate,
                    computed_value_for(&values, *gated)?,
                    &layout,
                )?
            }
            NodePayload::Binop(op @ (Binop::Umulp | Binop::Smulp), lhs, rhs) => {
                let destination = materialized_destination(
                    builder,
                    *node_ref,
                    return_node,
                    output_pointer,
                    scratch_pointer,
                    scratch_plan,
                )?;
                let lhs_layout = NativeValueLayout::from_type(&function.get_node(*lhs).ty)?;
                let rhs_layout = NativeValueLayout::from_type(&function.get_node(*rhs).ty)?;
                if layout_contains_wide_bits(&layout)
                    || is_wide_bits(&lhs_layout)
                    || is_wide_bits(&rhs_layout)
                {
                    lower_runtime_wide_mulp(
                        builder,
                        function,
                        destination,
                        *op,
                        *lhs,
                        *rhs,
                        &mut values,
                        &layout,
                        scratch_pointer,
                        scratch_plan,
                        runtime_callbacks.wide_mulp,
                        pointer_type,
                    )?;
                } else {
                    lower_mulp(
                        builder,
                        function,
                        destination,
                        *op,
                        *lhs,
                        *rhs,
                        &mut values,
                        &layout,
                    )?;
                }
                ComputedValue::Address(destination)
            }
            NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), lhs, rhs)
                if NativeValueLayout::from_type(&function.get_node(*lhs).ty)?
                    .is_memory_backed() =>
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
            NodePayload::Binop(
                op @ (Binop::Ugt
                | Binop::Uge
                | Binop::Ult
                | Binop::Ule
                | Binop::Sgt
                | Binop::Sge
                | Binop::Slt
                | Binop::Sle),
                lhs,
                rhs,
            ) if matches!(
                NativeValueLayout::from_type(&function.get_node(*lhs).ty)?,
                NativeValueLayout::WideBits(_)
            ) =>
            {
                let NativeValueLayout::WideBits(operand_layout) =
                    NativeValueLayout::from_type(&function.get_node(*lhs).ty)?
                else {
                    unreachable!("guard checks wide operand")
                };
                ComputedValue::Scalar(lower_wide_comparison(
                    builder,
                    computed_value_for(&values, *lhs)?,
                    computed_value_for(&values, *rhs)?,
                    operand_layout,
                    *op,
                )?)
            }
            NodePayload::Binop(op @ (Binop::Add | Binop::Sub), lhs, rhs)
                if matches!(&layout, NativeValueLayout::WideBits(_)) =>
            {
                let destination = materialized_destination(
                    builder,
                    *node_ref,
                    return_node,
                    output_pointer,
                    scratch_pointer,
                    scratch_plan,
                )?;
                lower_wide_add_sub(
                    builder,
                    destination,
                    computed_value_for(&values, *lhs)?,
                    computed_value_for(&values, *rhs)?,
                    expect_wide_layout(&layout)?,
                    *op == Binop::Sub,
                )?;
                ComputedValue::Address(destination)
            }
            NodePayload::Binop(
                op @ (Binop::Umul
                | Binop::Smul
                | Binop::Udiv
                | Binop::Sdiv
                | Binop::Umod
                | Binop::Smod
                | Binop::Shll
                | Binop::Shrl
                | Binop::Shra),
                lhs,
                rhs,
            ) if is_wide_bits(&layout)
                || is_wide_bits(&NativeValueLayout::from_type(&function.get_node(*lhs).ty)?)
                || is_wide_bits(&NativeValueLayout::from_type(&function.get_node(*rhs).ty)?) =>
            {
                let operation = match op {
                    Binop::Umul => WideBinaryOp::Umul,
                    Binop::Smul => WideBinaryOp::Smul,
                    Binop::Udiv => WideBinaryOp::Udiv,
                    Binop::Sdiv => WideBinaryOp::Sdiv,
                    Binop::Umod => WideBinaryOp::Umod,
                    Binop::Smod => WideBinaryOp::Smod,
                    Binop::Shll => WideBinaryOp::Shll,
                    Binop::Shrl => WideBinaryOp::Shrl,
                    Binop::Shra => WideBinaryOp::Shra,
                    _ => unreachable!("wide runtime binop branch"),
                };
                lower_runtime_wide_binop(
                    builder,
                    function,
                    *node_ref,
                    return_node,
                    output_pointer,
                    scratch_pointer,
                    scratch_plan,
                    runtime_callbacks.wide_binop,
                    pointer_type,
                    *lhs,
                    *rhs,
                    &mut values,
                    &layout,
                    operation,
                )?
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
                    &mut values,
                    scalar_layout,
                )?)
            }
            NodePayload::ZeroExt { arg, .. } => {
                if let NativeValueLayout::WideBits(wide) = &layout {
                    let destination = materialized_destination(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                    )?;
                    lower_wide_resize(
                        builder,
                        destination,
                        computed_value_for(&values, *arg)?,
                        &NativeValueLayout::from_type(&function.get_node(*arg).ty)?,
                        *wide,
                        false,
                    )?;
                    ComputedValue::Address(destination)
                } else {
                    let layout = require_scalar_layout(&layout)?;
                    let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    let resized = resize_unsigned(builder, arg, arg_layout, layout);
                    ComputedValue::Scalar(mask_value(builder, resized, layout))
                }
            }
            NodePayload::SignExt { arg, .. } => {
                if let NativeValueLayout::WideBits(wide) = &layout {
                    let destination = materialized_destination(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                    )?;
                    lower_wide_resize(
                        builder,
                        destination,
                        computed_value_for(&values, *arg)?,
                        &NativeValueLayout::from_type(&function.get_node(*arg).ty)?,
                        *wide,
                        true,
                    )?;
                    ComputedValue::Address(destination)
                } else {
                    let layout = require_scalar_layout(&layout)?;
                    let arg_layout = ScalarLayout::from_type(&function.get_node(*arg).ty)?;
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    let signed = signed_value(builder, arg, arg_layout);
                    let resized = resize_signed(builder, signed, arg_layout, layout);
                    ComputedValue::Scalar(mask_value(builder, resized, layout))
                }
            }
            NodePayload::BitSlice { arg, start, width } => {
                let arg_native_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
                if start.saturating_add(*width) > bits_bit_count(&arg_native_layout)? {
                    return Err(CompilerError::UnsupportedNode(format!(
                        "out-of-bounds bit_slice at node {}",
                        node.text_id
                    )));
                }
                if let NativeValueLayout::WideBits(wide) = &layout {
                    let destination = materialized_destination(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                    )?;
                    lower_wide_static_slice(
                        builder,
                        destination,
                        computed_value_for(&values, *arg)?,
                        &arg_native_layout,
                        *start,
                        *wide,
                    )?;
                    ComputedValue::Address(destination)
                } else if matches!(arg_native_layout, NativeValueLayout::WideBits(_)) {
                    let scalar_layout = require_scalar_layout(&layout)?;
                    let window = load_zero_window(
                        builder,
                        computed_value_for(&values, *arg)?,
                        &arg_native_layout,
                        *start,
                    )?;
                    let resized = if scalar_layout.clif_type() == types::I64 {
                        window
                    } else {
                        builder.ins().ireduce(scalar_layout.clif_type(), window)
                    };
                    ComputedValue::Scalar(mask_value(builder, resized, scalar_layout))
                } else {
                    let scalar_layout = require_scalar_layout(&layout)?;
                    let arg_layout = require_scalar_layout(&arg_native_layout)?;
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    let shifted = if *start == 0 {
                        arg
                    } else {
                        builder.ins().ushr_imm(arg, *start as i64)
                    };
                    let resized = resize_unsigned(builder, shifted, arg_layout, scalar_layout);
                    ComputedValue::Scalar(mask_value(builder, resized, scalar_layout))
                }
            }
            NodePayload::DynamicBitSlice {
                arg,
                start,
                width: _,
            } => {
                let arg_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
                let start_layout = NativeValueLayout::from_type(&function.get_node(*start).ty)?;
                if is_wide_bits(&layout) || is_wide_bits(&arg_layout) || is_wide_bits(&start_layout)
                {
                    lower_direct_dynamic_bit_slice(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                        computed_value_for(&values, *arg)?,
                        &arg_layout,
                        computed_value_for(&values, *start)?,
                        &start_layout,
                        &layout,
                    )?
                } else {
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    let start = scalar_value_for(builder, &mut values, *start)?;
                    ComputedValue::Scalar(lower_dynamic_bit_slice(
                        builder,
                        arg,
                        require_scalar_layout(&arg_layout)?,
                        start,
                        require_scalar_layout(&start_layout)?,
                        require_scalar_layout(&layout)?,
                    ))
                }
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                let arg_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
                let start_layout = NativeValueLayout::from_type(&function.get_node(*start).ty)?;
                let update_layout =
                    NativeValueLayout::from_type(&function.get_node(*update_value).ty)?;
                if is_wide_bits(&layout)
                    || is_wide_bits(&arg_layout)
                    || is_wide_bits(&start_layout)
                    || is_wide_bits(&update_layout)
                {
                    lower_runtime_wide_bit_slice_update(
                        builder,
                        function,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                        runtime_callbacks.wide_bit_slice_update,
                        pointer_type,
                        *arg,
                        *start,
                        *update_value,
                        &values,
                        &layout,
                    )?
                } else {
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    let start = scalar_value_for(builder, &mut values, *start)?;
                    let update_value = scalar_value_for(builder, &mut values, *update_value)?;
                    ComputedValue::Scalar(lower_bit_slice_update(
                        builder,
                        arg,
                        require_scalar_layout(&arg_layout)?,
                        start,
                        require_scalar_layout(&start_layout)?,
                        update_value,
                        require_scalar_layout(&update_layout)?,
                        require_scalar_layout(&layout)?,
                    ))
                }
            }
            NodePayload::ExtCarryOut { lhs, rhs, c_in } => {
                let operand_layout = NativeValueLayout::from_type(&function.get_node(*lhs).ty)?;
                let c_in_layout = ScalarLayout::from_type(&function.get_node(*c_in).ty)?;
                if let NativeValueLayout::WideBits(wide) = operand_layout {
                    let c_in = scalar_value_for(builder, &mut values, *c_in)?;
                    ComputedValue::Scalar(lower_wide_ext_carry_out(
                        builder,
                        computed_value_for(&values, *lhs)?,
                        computed_value_for(&values, *rhs)?,
                        c_in,
                        c_in_layout,
                        wide,
                    )?)
                } else {
                    let lhs = scalar_value_for(builder, &mut values, *lhs)?;
                    let rhs = scalar_value_for(builder, &mut values, *rhs)?;
                    let c_in = scalar_value_for(builder, &mut values, *c_in)?;
                    ComputedValue::Scalar(lower_ext_carry_out(
                        builder,
                        lhs,
                        rhs,
                        c_in,
                        require_scalar_layout(&operand_layout)?,
                        c_in_layout,
                        require_scalar_layout(&layout)?,
                    ))
                }
            }
            NodePayload::ExtPrioEncode { arg, lsb_prio } => {
                let arg_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
                if is_wide_bits(&layout) || is_wide_bits(&arg_layout) {
                    lower_runtime_wide_unary_op(
                        builder,
                        function,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                        runtime_callbacks.wide_unary_op,
                        pointer_type,
                        *arg,
                        &values,
                        &layout,
                        WideUnaryOp::ExtPrioEncode,
                        usize::from(*lsb_prio),
                    )?
                } else {
                    let result_layout = require_scalar_layout(&layout)?;
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    ComputedValue::Scalar(lower_ext_prio_encode(
                        builder,
                        arg,
                        require_scalar_layout(&arg_layout)?,
                        result_layout,
                        *lsb_prio,
                    ))
                }
            }
            NodePayload::ExtClz { arg, offset, .. } => {
                let arg_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
                if is_wide_bits(&layout) || is_wide_bits(&arg_layout) {
                    lower_runtime_wide_unary_op(
                        builder,
                        function,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                        runtime_callbacks.wide_unary_op,
                        pointer_type,
                        *arg,
                        &values,
                        &layout,
                        WideUnaryOp::ExtClz,
                        *offset,
                    )?
                } else {
                    let result_layout = require_scalar_layout(&layout)?;
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    ComputedValue::Scalar(lower_ext_clz(
                        builder,
                        arg,
                        require_scalar_layout(&arg_layout)?,
                        result_layout,
                        *offset,
                    ))
                }
            }
            NodePayload::ExtNormalizeLeft {
                arg,
                shift_offset,
                normalized_bit_count: _,
                clz_bit_count: _,
            } => {
                let arg_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
                if is_wide_bits(&arg_layout) || layout_contains_wide_bits(&layout) {
                    lower_runtime_wide_ext_normalize_left(
                        builder,
                        function,
                        node,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                        runtime_callbacks.wide_unary_op,
                        pointer_type,
                        *arg,
                        *shift_offset,
                        &values,
                        &layout,
                    )?
                } else {
                    let arg_value = scalar_value_for(builder, &mut values, *arg)?;
                    lower_ext_normalize_left(
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
                        arg_value,
                        &layout,
                    )?
                }
            }
            NodePayload::ExtMaskLow { count } => {
                let count_layout = NativeValueLayout::from_type(&function.get_node(*count).ty)?;
                if is_wide_bits(&layout) || is_wide_bits(&count_layout) {
                    lower_runtime_wide_unary_op(
                        builder,
                        function,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                        runtime_callbacks.wide_unary_op,
                        pointer_type,
                        *count,
                        &values,
                        &layout,
                        WideUnaryOp::ExtMaskLow,
                        0,
                    )?
                } else {
                    let result_layout = require_scalar_layout(&layout)?;
                    let count = scalar_value_for(builder, &mut values, *count)?;
                    ComputedValue::Scalar(lower_ext_mask_low(
                        builder,
                        count,
                        require_scalar_layout(&count_layout)?,
                        result_layout,
                    ))
                }
            }
            NodePayload::ExtNaryAdd { terms, arch: _ } => {
                let has_wide_term = terms.iter().try_fold(false, |found, term| {
                    NativeValueLayout::from_type(&function.get_node(term.operand).ty)
                        .map(|term_layout| found || is_wide_bits(&term_layout))
                })?;
                if let NativeValueLayout::WideBits(wide) = &layout {
                    let destination = materialized_destination(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                    )?;
                    lower_wide_ext_nary_add(builder, function, destination, terms, &values, *wide)?;
                    ComputedValue::Address(destination)
                } else if has_wide_term {
                    let result_layout = require_scalar_layout(&layout)?;
                    ComputedValue::Scalar(lower_mixed_ext_nary_add(
                        builder,
                        function,
                        terms,
                        &mut values,
                        result_layout,
                    )?)
                } else {
                    let result_layout = require_scalar_layout(&layout)?;
                    ComputedValue::Scalar(lower_ext_nary_add(
                        builder,
                        function,
                        terms,
                        &mut values,
                        result_layout,
                    )?)
                }
            }
            NodePayload::Array(args) => {
                if layout.byte_count() == 0 {
                    ComputedValue::ZeroSized
                } else {
                    let destination = materialized_destination(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                    )?;
                    lower_array_construction(
                        builder,
                        destination,
                        args,
                        &values,
                        &layout,
                        *node_ref != return_node,
                    )?;
                    ComputedValue::Address(destination)
                }
            }
            NodePayload::Tuple(args) => {
                if layout.byte_count() == 0 {
                    ComputedValue::ZeroSized
                } else {
                    let destination = materialized_destination(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                    )?;
                    lower_tuple_construction(
                        builder,
                        destination,
                        args,
                        &values,
                        &layout,
                        *node_ref != return_node,
                    )?;
                    ComputedValue::Address(destination)
                }
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
                *node_ref,
                *array,
                indices,
                *assumed_in_bounds,
                &mut values,
                &layout,
                pointer_type,
                event_sites,
                runtime_callbacks.record_assumption_failure,
                execution_context,
            )?,
            NodePayload::ArraySlice {
                array,
                start,
                width: _,
            } => {
                if layout.byte_count() == 0 {
                    ComputedValue::ZeroSized
                } else {
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
                        *node_ref != return_node,
                    )?;
                    ComputedValue::Address(destination)
                }
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
                &mut values,
                &layout,
                pointer_type,
                event_sites,
                runtime_callbacks.record_assumption_failure,
                execution_context,
            )?,
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => lower_sel(
                builder,
                scratch_pointer,
                scratch_plan,
                *selector,
                cases,
                *default,
                &mut values,
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
                scratch_pointer,
                scratch_plan,
                *selector,
                cases,
                *default,
                &mut values,
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
                &mut values,
                &layout,
            )?,
            NodePayload::OneHot { arg, lsb_prio } => {
                let arg_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
                if is_wide_bits(&layout) || is_wide_bits(&arg_layout) {
                    lower_runtime_wide_unary_op(
                        builder,
                        function,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                        runtime_callbacks.wide_unary_op,
                        pointer_type,
                        *arg,
                        &values,
                        &layout,
                        WideUnaryOp::OneHot,
                        usize::from(*lsb_prio),
                    )?
                } else {
                    let result_layout = require_scalar_layout(&layout)?;
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    ComputedValue::Scalar(lower_one_hot(
                        builder,
                        arg,
                        require_scalar_layout(&arg_layout)?,
                        result_layout,
                        *lsb_prio,
                    ))
                }
            }
            NodePayload::Encode { arg } => {
                let arg_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
                if is_wide_bits(&layout) || is_wide_bits(&arg_layout) {
                    lower_runtime_wide_unary_op(
                        builder,
                        function,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                        runtime_callbacks.wide_unary_op,
                        pointer_type,
                        *arg,
                        &values,
                        &layout,
                        WideUnaryOp::Encode,
                        0,
                    )?
                } else {
                    let result_layout = require_scalar_layout(&layout)?;
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    ComputedValue::Scalar(lower_encode(
                        builder,
                        arg,
                        require_scalar_layout(&arg_layout)?,
                        result_layout,
                    ))
                }
            }
            NodePayload::Decode { arg, width: _ } => {
                let arg_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
                if is_wide_bits(&layout) || is_wide_bits(&arg_layout) {
                    lower_direct_decode(
                        builder,
                        *node_ref,
                        return_node,
                        output_pointer,
                        scratch_pointer,
                        scratch_plan,
                        computed_value_for(&values, *arg)?,
                        &arg_layout,
                        &layout,
                    )?
                } else {
                    let result_layout = require_scalar_layout(&layout)?;
                    let arg = scalar_value_for(builder, &mut values, *arg)?;
                    ComputedValue::Scalar(lower_decode(
                        builder,
                        arg,
                        require_scalar_layout(&arg_layout)?,
                        result_layout,
                    ))
                }
            }
            _ => return Err(unsupported_node(node)),
        };
        values[node_ref.index] = Some(value);
    }

    let result = computed_value_for(&values, return_node)?;
    let result_layout = NativeValueLayout::from_type(&function.ret_ty)?;
    if result_layout.byte_count() != 0 {
        store_value_to_storage(builder, output_pointer, result, &result_layout)?;
    }
    let success = builder.ins().iconst(types::I32, 0);
    builder.ins().return_(&[success]);
    Ok(())
}

fn event_site_id(
    event_sites: &HashMap<NodeRef, u32>,
    node_ref: NodeRef,
    node: &ir::Node,
) -> Result<u32, CompilerError> {
    event_sites.get(&node_ref).copied().ok_or_else(|| {
        CompilerError::InvalidFunction(format!("missing event metadata for node {}", node.text_id))
    })
}

fn emit_conditional_site_call(
    builder: &mut FunctionBuilder<'_>,
    condition: Value,
    callback: FuncRef,
    execution_context: Value,
    site_id: u32,
) {
    let invoke = builder.create_block();
    let after = builder.create_block();
    builder.set_cold_block(invoke);
    builder.ins().brif(condition, invoke, &[], after, &[]);
    builder.switch_to_block(invoke);
    builder.seal_block(invoke);
    let site_id = builder.ins().iconst(types::I32, i64::from(site_id));
    builder.ins().call(callback, &[execution_context, site_id]);
    builder.ins().jump(after, &[]);
    builder.switch_to_block(after);
    builder.seal_block(after);
}

fn emit_conditional_trace_call(
    builder: &mut FunctionBuilder<'_>,
    condition: Value,
    callback: FuncRef,
    execution_context: Value,
    site_id: u32,
    operand_pointers: Value,
) {
    let invoke = builder.create_block();
    let after = builder.create_block();
    builder.set_cold_block(invoke);
    builder.ins().brif(condition, invoke, &[], after, &[]);
    builder.switch_to_block(invoke);
    builder.seal_block(invoke);
    let site_id = builder.ins().iconst(types::I32, i64::from(site_id));
    builder
        .ins()
        .call(callback, &[execution_context, site_id, operand_pointers]);
    builder.ins().jump(after, &[]);
    builder.switch_to_block(after);
    builder.seal_block(after);
}

#[allow(clippy::too_many_arguments)]
fn lower_trace_operand_pointers(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    trace_node: NodeRef,
    operands: &[NodeRef],
    values: &[Option<ComputedValue>],
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    pointer_type: ClifType,
) -> Result<Value, CompilerError> {
    if operands.is_empty() {
        return Ok(builder.ins().iconst(pointer_type, 0));
    }
    let plan = scratch_plan.trace_sites.get(&trace_node).ok_or_else(|| {
        CompilerError::InvalidFunction("missing trace scratch allocation".to_string())
    })?;
    let pointer_array = pointer_at_offset(builder, scratch_pointer, plan.pointer_array_offset);
    for (index, operand) in operands.iter().enumerate() {
        let layout = NativeValueLayout::from_type(&function.get_node(*operand).ty)?;
        let pointer = match computed_value_for(values, *operand)? {
            ComputedValue::Scalar(value) => {
                let offset = plan.scalar_operand_offsets[index].ok_or_else(|| {
                    CompilerError::InvalidFunction("missing trace scalar scratch allocation".into())
                })?;
                let pointer = pointer_at_offset(builder, scratch_pointer, offset);
                store_value_to_storage(builder, pointer, ComputedValue::Scalar(value), &layout)?;
                pointer
            }
            ComputedValue::ScalarAddress {
                pointer, offset, ..
            } => pointer_at_offset(builder, pointer, offset),
            ComputedValue::ScalarArrayIndex(value) => {
                let value = materialize_scalar(builder, ComputedValue::ScalarArrayIndex(value))?;
                let offset = plan.scalar_operand_offsets[index].ok_or_else(|| {
                    CompilerError::InvalidFunction("missing trace scalar scratch allocation".into())
                })?;
                let pointer = pointer_at_offset(builder, scratch_pointer, offset);
                store_value_to_storage(builder, pointer, ComputedValue::Scalar(value), &layout)?;
                pointer
            }
            ComputedValue::Address(pointer) => pointer,
            ComputedValue::ZeroSized => builder.ins().iconst(pointer_type, 0),
        };
        builder.ins().store(
            MemFlags::new(),
            pointer,
            pointer_array,
            (index * std::mem::size_of::<*const u8>()) as i32,
        );
    }
    Ok(pointer_array)
}

fn expect_wide_layout(layout: &NativeValueLayout) -> Result<WideBitsLayout, CompilerError> {
    match layout {
        NativeValueLayout::WideBits(wide) => Ok(*wide),
        _ => Err(CompilerError::InvalidFunction(
            "operation expected a wide bits result layout".into(),
        )),
    }
}

fn bits_bit_count(layout: &NativeValueLayout) -> Result<usize, CompilerError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => Ok(scalar.bit_count),
        NativeValueLayout::WideBits(wide) => Ok(wide.bit_count),
        _ => Err(CompilerError::InvalidFunction(
            "operation expected a bits-typed value".into(),
        )),
    }
}

fn bits_limb_count(layout: &NativeValueLayout) -> Result<usize, CompilerError> {
    Ok(bits_bit_count(layout)?.div_ceil(64))
}

fn is_wide_bits(layout: &NativeValueLayout) -> bool {
    matches!(layout, NativeValueLayout::WideBits(_))
}

fn layout_contains_wide_bits(layout: &NativeValueLayout) -> bool {
    match layout {
        NativeValueLayout::WideBits(_) => true,
        NativeValueLayout::Array { element, .. } => layout_contains_wide_bits(element),
        NativeValueLayout::Tuple { fields, .. } => fields
            .iter()
            .any(|field| layout_contains_wide_bits(field.layout.as_ref())),
        NativeValueLayout::Scalar(_) | NativeValueLayout::Token => false,
    }
}

/// Returns whether lowering needs scratch storage for a runtime bits callback.
fn function_uses_runtime_bits_callback(
    function: &ir::Fn,
    order: &[NodeRef],
) -> Result<bool, CompilerError> {
    for node_ref in order {
        let node = function.get_node(*node_ref);
        let may_call_runtime = matches!(
            &node.payload,
            NodePayload::Binop(
                Binop::Umul
                    | Binop::Smul
                    | Binop::Udiv
                    | Binop::Sdiv
                    | Binop::Umod
                    | Binop::Smod
                    | Binop::Umulp
                    | Binop::Smulp
                    | Binop::Shll
                    | Binop::Shrl
                    | Binop::Shra,
                _,
                _
            ) | NodePayload::BitSliceUpdate { .. }
                | NodePayload::ExtPrioEncode { .. }
                | NodePayload::ExtClz { .. }
                | NodePayload::ExtNormalizeLeft { .. }
                | NodePayload::ExtMaskLow { .. }
                | NodePayload::OneHot { .. }
                | NodePayload::Encode { .. }
        );
        if !may_call_runtime {
            continue;
        }
        let result_layout = NativeValueLayout::from_type(&node.ty)?;
        if layout_contains_wide_bits(&result_layout) {
            return Ok(true);
        }
        for operand in operands(&node.payload) {
            let operand_layout = NativeValueLayout::from_type(&function.get_node(operand).ty)?;
            if layout_contains_wide_bits(&operand_layout) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn runtime_width_constant(
    builder: &mut FunctionBuilder<'_>,
    pointer_type: ClifType,
    bit_count: usize,
) -> Result<Value, CompilerError> {
    let bit_count = i64::try_from(bit_count).map_err(|_| {
        CompilerError::UnsupportedType("bit width exceeds the compiler runtime ABI".into())
    })?;
    Ok(builder.ins().iconst(pointer_type, bit_count))
}

fn runtime_scalar_pointer(
    builder: &mut FunctionBuilder<'_>,
    node_ref: NodeRef,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
) -> Result<Value, CompilerError> {
    let offset = scratch_plan
        .runtime_scalar_offsets
        .get(&node_ref)
        .copied()
        .ok_or_else(|| {
            CompilerError::InvalidFunction(format!(
                "runtime scalar node {} has no scratch assignment",
                node_ref.index
            ))
        })?;
    Ok(pointer_at_offset(builder, scratch_pointer, offset))
}

fn runtime_temporary_pointer(
    builder: &mut FunctionBuilder<'_>,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    index: usize,
) -> Result<Value, CompilerError> {
    let offset = scratch_plan
        .runtime_temporary_offsets
        .get(index)
        .copied()
        .flatten()
        .ok_or_else(|| CompilerError::InvalidFunction("runtime temporary is unavailable".into()))?;
    Ok(pointer_at_offset(builder, scratch_pointer, offset))
}

fn runtime_bits_operand_pointer(
    builder: &mut FunctionBuilder<'_>,
    node_ref: NodeRef,
    value: ComputedValue,
    layout: &NativeValueLayout,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
) -> Result<Value, CompilerError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let pointer = runtime_scalar_pointer(builder, node_ref, scratch_pointer, scratch_plan)?;
            let value = materialize_scalar(builder, value)?;
            let value = resize_integer_type_unsigned(builder, value, *scalar, types::I64);
            builder.ins().store(MemFlags::new(), value, pointer, 0);
            Ok(pointer)
        }
        NativeValueLayout::WideBits(_) => expect_address(value),
        _ => Err(CompilerError::InvalidFunction(
            "runtime bit operation received a non-bits operand".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn runtime_bits_result_pointer(
    builder: &mut FunctionBuilder<'_>,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    layout: &NativeValueLayout,
) -> Result<Value, CompilerError> {
    match layout {
        NativeValueLayout::Scalar(_) => {
            runtime_scalar_pointer(builder, node_ref, scratch_pointer, scratch_plan)
        }
        NativeValueLayout::WideBits(_) => materialized_destination(
            builder,
            node_ref,
            return_node,
            output_pointer,
            scratch_pointer,
            scratch_plan,
        ),
        _ => Err(CompilerError::InvalidFunction(
            "runtime bit operation has a non-bits result".into(),
        )),
    }
}

fn load_runtime_bits_result(
    builder: &mut FunctionBuilder<'_>,
    pointer: Value,
    layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let raw = builder.ins().load(types::I64, MemFlags::new(), pointer, 0);
            let resized = if scalar.clif_type() == types::I64 {
                raw
            } else {
                builder.ins().ireduce(scalar.clif_type(), raw)
            };
            Ok(ComputedValue::Scalar(mask_value(builder, resized, *scalar)))
        }
        NativeValueLayout::WideBits(_) => Ok(ComputedValue::Address(pointer)),
        _ => Err(CompilerError::InvalidFunction(
            "runtime bit operation has a non-bits result".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_runtime_wide_binop(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    callback: FuncRef,
    pointer_type: ClifType,
    lhs: NodeRef,
    rhs: NodeRef,
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
    operation: WideBinaryOp,
) -> Result<ComputedValue, CompilerError> {
    let lhs_layout = NativeValueLayout::from_type(&function.get_node(lhs).ty)?;
    let rhs_layout = NativeValueLayout::from_type(&function.get_node(rhs).ty)?;
    let destination = runtime_bits_result_pointer(
        builder,
        node_ref,
        return_node,
        output_pointer,
        scratch_pointer,
        scratch_plan,
        layout,
    )?;
    let lhs_pointer = runtime_bits_operand_pointer(
        builder,
        lhs,
        computed_value_for(values, lhs)?,
        &lhs_layout,
        scratch_pointer,
        scratch_plan,
    )?;
    let rhs_pointer = runtime_bits_operand_pointer(
        builder,
        rhs,
        computed_value_for(values, rhs)?,
        &rhs_layout,
        scratch_pointer,
        scratch_plan,
    )?;
    let result_width = runtime_width_constant(builder, pointer_type, bits_bit_count(layout)?)?;
    let lhs_width = runtime_width_constant(builder, pointer_type, bits_bit_count(&lhs_layout)?)?;
    let rhs_width = runtime_width_constant(builder, pointer_type, bits_bit_count(&rhs_layout)?)?;
    let operation = builder.ins().iconst(types::I32, operation as u32 as i64);
    builder.ins().call(
        callback,
        &[
            destination,
            result_width,
            lhs_pointer,
            lhs_width,
            rhs_pointer,
            rhs_width,
            operation,
        ],
    );
    load_runtime_bits_result(builder, destination, layout)
}

#[allow(clippy::too_many_arguments)]
fn lower_runtime_wide_unary_op(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    callback: FuncRef,
    pointer_type: ClifType,
    arg: NodeRef,
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
    operation: WideUnaryOp,
    attribute: usize,
) -> Result<ComputedValue, CompilerError> {
    let arg_layout = NativeValueLayout::from_type(&function.get_node(arg).ty)?;
    let destination = runtime_bits_result_pointer(
        builder,
        node_ref,
        return_node,
        output_pointer,
        scratch_pointer,
        scratch_plan,
        layout,
    )?;
    let arg_pointer = runtime_bits_operand_pointer(
        builder,
        arg,
        computed_value_for(values, arg)?,
        &arg_layout,
        scratch_pointer,
        scratch_plan,
    )?;
    emit_runtime_wide_unary_op(
        builder,
        callback,
        pointer_type,
        destination,
        layout,
        arg_pointer,
        &arg_layout,
        operation,
        attribute,
    )?;
    load_runtime_bits_result(builder, destination, layout)
}

#[allow(clippy::too_many_arguments)]
fn emit_runtime_wide_unary_op(
    builder: &mut FunctionBuilder<'_>,
    callback: FuncRef,
    pointer_type: ClifType,
    destination: Value,
    layout: &NativeValueLayout,
    arg_pointer: Value,
    arg_layout: &NativeValueLayout,
    operation: WideUnaryOp,
    attribute: usize,
) -> Result<(), CompilerError> {
    let result_width = runtime_width_constant(builder, pointer_type, bits_bit_count(layout)?)?;
    let arg_width = runtime_width_constant(builder, pointer_type, bits_bit_count(arg_layout)?)?;
    let operation = builder.ins().iconst(types::I32, operation as u32 as i64);
    let attribute = runtime_width_constant(builder, pointer_type, attribute)?;
    builder.ins().call(
        callback,
        &[
            destination,
            result_width,
            arg_pointer,
            arg_width,
            operation,
            attribute,
        ],
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_runtime_wide_ext_normalize_left(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    callback: FuncRef,
    pointer_type: ClifType,
    arg: NodeRef,
    shift_offset: usize,
    values: &[Option<ComputedValue>],
    result_layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    let arg_layout = NativeValueLayout::from_type(&function.get_node(arg).ty)?;
    let arg_pointer = runtime_bits_operand_pointer(
        builder,
        arg,
        computed_value_for(values, arg)?,
        &arg_layout,
        scratch_pointer,
        scratch_plan,
    )?;
    match result_layout {
        NativeValueLayout::Scalar(_) | NativeValueLayout::WideBits(_) => {
            let destination = runtime_bits_result_pointer(
                builder,
                node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
                result_layout,
            )?;
            emit_runtime_wide_unary_op(
                builder,
                callback,
                pointer_type,
                destination,
                result_layout,
                arg_pointer,
                &arg_layout,
                WideUnaryOp::ExtNormalizeLeft,
                shift_offset,
            )?;
            load_runtime_bits_result(builder, destination, result_layout)
        }
        NativeValueLayout::Tuple { fields, .. } if fields.len() == 2 => {
            let destination = materialized_destination(
                builder,
                node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
            )?;
            for (index, (field, operation, attribute)) in [
                (
                    fields[0].layout.as_ref(),
                    WideUnaryOp::ExtNormalizeLeft,
                    shift_offset,
                ),
                (fields[1].layout.as_ref(), WideUnaryOp::ExtClz, 0),
            ]
            .into_iter()
            .enumerate()
            {
                let field_destination =
                    pointer_at_offset(builder, destination, fields[index].offset);
                let call_destination = if matches!(field, NativeValueLayout::Scalar(_)) {
                    runtime_temporary_pointer(builder, scratch_pointer, scratch_plan, index)?
                } else {
                    field_destination
                };
                emit_runtime_wide_unary_op(
                    builder,
                    callback,
                    pointer_type,
                    call_destination,
                    field,
                    arg_pointer,
                    &arg_layout,
                    operation,
                    attribute,
                )?;
                if matches!(field, NativeValueLayout::Scalar(_)) {
                    let value = load_runtime_bits_result(builder, call_destination, field)?;
                    store_value_to_storage(builder, field_destination, value, field)?;
                }
            }
            Ok(ComputedValue::Address(destination))
        }
        _ => Err(CompilerError::InvalidFunction(format!(
            "ext_normalize_left has unexpected result layout at {}",
            node.text_id
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_runtime_wide_mulp(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    destination: Value,
    op: Binop,
    lhs: NodeRef,
    rhs: NodeRef,
    values: &[Option<ComputedValue>],
    result_layout: &NativeValueLayout,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    callback: FuncRef,
    pointer_type: ClifType,
) -> Result<(), CompilerError> {
    let NativeValueLayout::Tuple { fields, .. } = result_layout else {
        return Err(CompilerError::InvalidFunction(
            "partial-product multiply did not have tuple result type".into(),
        ));
    };
    if fields.len() != 2 || fields[0].layout != fields[1].layout {
        return Err(CompilerError::InvalidFunction(
            "partial-product multiply requires two equal bits tuple fields".into(),
        ));
    }
    let field_layout = fields[0].layout.as_ref();
    let lhs_layout = NativeValueLayout::from_type(&function.get_node(lhs).ty)?;
    let rhs_layout = NativeValueLayout::from_type(&function.get_node(rhs).ty)?;
    let lhs_pointer = runtime_bits_operand_pointer(
        builder,
        lhs,
        computed_value_for(values, lhs)?,
        &lhs_layout,
        scratch_pointer,
        scratch_plan,
    )?;
    let rhs_pointer = runtime_bits_operand_pointer(
        builder,
        rhs,
        computed_value_for(values, rhs)?,
        &rhs_layout,
        scratch_pointer,
        scratch_plan,
    )?;
    let field_pointers = if matches!(field_layout, NativeValueLayout::Scalar(_)) {
        [
            runtime_temporary_pointer(builder, scratch_pointer, scratch_plan, 0)?,
            runtime_temporary_pointer(builder, scratch_pointer, scratch_plan, 1)?,
        ]
    } else {
        [
            pointer_at_offset(builder, destination, fields[0].offset),
            pointer_at_offset(builder, destination, fields[1].offset),
        ]
    };
    let result_width =
        runtime_width_constant(builder, pointer_type, bits_bit_count(field_layout)?)?;
    let lhs_width = runtime_width_constant(builder, pointer_type, bits_bit_count(&lhs_layout)?)?;
    let rhs_width = runtime_width_constant(builder, pointer_type, bits_bit_count(&rhs_layout)?)?;
    let signed = builder
        .ins()
        .iconst(types::I32, i64::from(op == Binop::Smulp));
    builder.ins().call(
        callback,
        &[
            field_pointers[0],
            field_pointers[1],
            result_width,
            lhs_pointer,
            lhs_width,
            rhs_pointer,
            rhs_width,
            signed,
        ],
    );
    if matches!(field_layout, NativeValueLayout::Scalar(_)) {
        for (field, pointer) in fields.iter().zip(field_pointers) {
            let value = load_runtime_bits_result(builder, pointer, field.layout.as_ref())?;
            let field_pointer = pointer_at_offset(builder, destination, field.offset);
            store_value_to_storage(builder, field_pointer, value, field.layout.as_ref())?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_runtime_wide_bit_slice_update(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    callback: FuncRef,
    pointer_type: ClifType,
    arg: NodeRef,
    start: NodeRef,
    update_value: NodeRef,
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    let arg_layout = NativeValueLayout::from_type(&function.get_node(arg).ty)?;
    let start_layout = NativeValueLayout::from_type(&function.get_node(start).ty)?;
    let update_layout = NativeValueLayout::from_type(&function.get_node(update_value).ty)?;
    let destination = runtime_bits_result_pointer(
        builder,
        node_ref,
        return_node,
        output_pointer,
        scratch_pointer,
        scratch_plan,
        layout,
    )?;
    let arg_pointer = runtime_bits_operand_pointer(
        builder,
        arg,
        computed_value_for(values, arg)?,
        &arg_layout,
        scratch_pointer,
        scratch_plan,
    )?;
    let start_pointer = runtime_bits_operand_pointer(
        builder,
        start,
        computed_value_for(values, start)?,
        &start_layout,
        scratch_pointer,
        scratch_plan,
    )?;
    let update_pointer = runtime_bits_operand_pointer(
        builder,
        update_value,
        computed_value_for(values, update_value)?,
        &update_layout,
        scratch_pointer,
        scratch_plan,
    )?;
    let result_width = runtime_width_constant(builder, pointer_type, bits_bit_count(layout)?)?;
    let arg_width = runtime_width_constant(builder, pointer_type, bits_bit_count(&arg_layout)?)?;
    let start_width =
        runtime_width_constant(builder, pointer_type, bits_bit_count(&start_layout)?)?;
    let update_width =
        runtime_width_constant(builder, pointer_type, bits_bit_count(&update_layout)?)?;
    builder.ins().call(
        callback,
        &[
            destination,
            result_width,
            arg_pointer,
            arg_width,
            start_pointer,
            start_width,
            update_pointer,
            update_width,
        ],
    );
    load_runtime_bits_result(builder, destination, layout)
}

fn load_raw_bits_limb(
    builder: &mut FunctionBuilder<'_>,
    value: ComputedValue,
    layout: &NativeValueLayout,
    limb: usize,
) -> Result<Value, CompilerError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            if limb != 0 {
                return Ok(builder.ins().iconst(types::I64, 0));
            }
            let value = materialize_scalar(builder, value)?;
            Ok(resize_integer_type_unsigned(
                builder,
                value,
                *scalar,
                types::I64,
            ))
        }
        NativeValueLayout::WideBits(wide) => {
            if limb >= wide.limb_count {
                return Ok(builder.ins().iconst(types::I64, 0));
            }
            let pointer = expect_address(value)?;
            Ok(builder.ins().load(
                types::I64,
                MemFlags::new(),
                pointer,
                (limb * std::mem::size_of::<u64>()) as i32,
            ))
        }
        _ => Err(CompilerError::InvalidFunction(
            "aggregate value used as bitvector storage".into(),
        )),
    }
}

fn bit_sign_condition(
    builder: &mut FunctionBuilder<'_>,
    value: ComputedValue,
    layout: &NativeValueLayout,
) -> Result<Value, CompilerError> {
    let bit_count = bits_bit_count(layout)?;
    let limb = load_raw_bits_limb(builder, value, layout, (bit_count - 1) / 64)?;
    let mask = 1u64 << ((bit_count - 1) % 64);
    let masked = builder.ins().band_imm(limb, mask as i64);
    Ok(builder.ins().icmp_imm(IntCC::NotEqual, masked, 0))
}

fn load_extended_bits_limb(
    builder: &mut FunctionBuilder<'_>,
    value: ComputedValue,
    layout: &NativeValueLayout,
    limb: usize,
    signed: bool,
) -> Result<Value, CompilerError> {
    let bit_count = bits_bit_count(layout)?;
    let source_limbs = bits_limb_count(layout)?;
    let fill = if signed {
        let sign = bit_sign_condition(builder, value.clone(), layout)?;
        let zero = builder.ins().iconst(types::I64, 0);
        let ones = builder.ins().iconst(types::I64, -1);
        builder.ins().select(sign, ones, zero)
    } else {
        builder.ins().iconst(types::I64, 0)
    };
    if limb >= source_limbs {
        return Ok(fill);
    }
    let raw = load_raw_bits_limb(builder, value, layout, limb)?;
    if !signed || limb + 1 != source_limbs || bit_count % 64 == 0 {
        return Ok(raw);
    }
    let semantic_mask = (1u64 << (bit_count % 64)) - 1;
    let high_fill = builder.ins().band_imm(fill, !semantic_mask as i64);
    Ok(builder.ins().bor(raw, high_fill))
}

fn load_zero_window(
    builder: &mut FunctionBuilder<'_>,
    value: ComputedValue,
    layout: &NativeValueLayout,
    start: usize,
) -> Result<Value, CompilerError> {
    let limb = start / 64;
    let shift = start % 64;
    let low = load_raw_bits_limb(builder, value.clone(), layout, limb)?;
    if shift == 0 {
        return Ok(low);
    }
    let high = load_raw_bits_limb(builder, value, layout, limb + 1)?;
    let low = builder.ins().ushr_imm(low, shift as i64);
    let high = builder.ins().ishl_imm(high, (64 - shift) as i64);
    Ok(builder.ins().bor(low, high))
}

fn store_wide_limb(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    wide: WideBitsLayout,
    limb: usize,
    value: Value,
) {
    let value = if limb + 1 == wide.limb_count && wide.bit_count % 64 != 0 {
        builder.ins().band_imm(value, wide.high_mask() as i64)
    } else {
        value
    };
    builder.ins().store(
        MemFlags::new(),
        value,
        destination,
        (limb * std::mem::size_of::<u64>()) as i32,
    );
}

fn load_wide_limb(builder: &mut FunctionBuilder<'_>, source: Value, limb: usize) -> Value {
    builder.ins().load(
        types::I64,
        MemFlags::new(),
        source,
        (limb * std::mem::size_of::<u64>()) as i32,
    )
}

fn lower_wide_resize(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    input: ComputedValue,
    input_layout: &NativeValueLayout,
    result_layout: WideBitsLayout,
    signed: bool,
) -> Result<(), CompilerError> {
    for limb in 0..result_layout.limb_count {
        let value = load_extended_bits_limb(builder, input.clone(), input_layout, limb, signed)?;
        store_wide_limb(builder, destination, result_layout, limb, value);
    }
    Ok(())
}

fn lower_wide_static_slice(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    input: ComputedValue,
    input_layout: &NativeValueLayout,
    start: usize,
    result_layout: WideBitsLayout,
) -> Result<(), CompilerError> {
    for limb in 0..result_layout.limb_count {
        let value = load_zero_window(builder, input.clone(), input_layout, start + limb * 64)?;
        store_wide_limb(builder, destination, result_layout, limb, value);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DynamicBitOffset {
    low_bits: Value,
    in_bounds: Value,
}

/// Produces the low offset bits and an explicit range check for a dynamic bit
/// offset. Wide offsets are in range only when all limbs above the low limb are
/// zero.
fn lower_dynamic_bit_offset(
    builder: &mut FunctionBuilder<'_>,
    offset: ComputedValue,
    offset_layout: &NativeValueLayout,
    upper_bound: usize,
) -> Result<DynamicBitOffset, CompilerError> {
    let low_bits = load_raw_bits_limb(builder, offset.clone(), offset_layout, 0)?;
    let mut high_bits = builder.ins().iconst(types::I64, 0);
    for limb in 1..bits_limb_count(offset_layout)? {
        let next = load_raw_bits_limb(builder, offset.clone(), offset_layout, limb)?;
        high_bits = builder.ins().bor(high_bits, next);
    }
    let high_is_zero = builder.ins().icmp_imm(IntCC::Equal, high_bits, 0);
    let low_is_in_bounds =
        builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThan, low_bits, upper_bound as i64);
    Ok(DynamicBitOffset {
        low_bits,
        in_bounds: builder.ins().band(high_is_zero, low_is_in_bounds),
    })
}

/// Loads one dynamically selected input limb, returning zero when the limb or
/// the original bit offset is out of range.
fn load_dynamic_zero_limb(
    builder: &mut FunctionBuilder<'_>,
    input: ComputedValue,
    input_layout: &NativeValueLayout,
    limb: Value,
    offset_in_bounds: Value,
) -> Result<Value, CompilerError> {
    let limb_count = bits_limb_count(input_layout)?;
    let limb_is_in_bounds =
        builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThan, limb, limb_count as i64);
    let valid = builder.ins().band(offset_in_bounds, limb_is_in_bounds);
    let raw = match input_layout {
        NativeValueLayout::Scalar(_) => load_raw_bits_limb(builder, input, input_layout, 0)?,
        NativeValueLayout::WideBits(_) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let safe_limb = builder.ins().select(limb_is_in_bounds, limb, zero);
            let pointer = expect_address(input)?;
            let pointer_type = builder.func.dfg.value_type(pointer);
            let safe_limb = if pointer_type == types::I64 {
                safe_limb
            } else {
                builder.ins().ireduce(pointer_type, safe_limb)
            };
            let pointer = array_element_pointer_from_address_index(builder, pointer, safe_limb, 8);
            builder.ins().load(types::I64, MemFlags::new(), pointer, 0)
        }
        _ => {
            return Err(CompilerError::InvalidFunction(
                "dynamic bit slice input is not bits-typed".into(),
            ));
        }
    };
    let zero = builder.ins().iconst(types::I64, 0);
    Ok(builder.ins().select(valid, raw, zero))
}

/// Emits one dynamic zero-filled 64-bit window from a bits value.
fn lower_dynamic_zero_window(
    builder: &mut FunctionBuilder<'_>,
    input: ComputedValue,
    input_layout: &NativeValueLayout,
    offset: DynamicBitOffset,
    output_limb: usize,
) -> Result<Value, CompilerError> {
    let source_limb = builder.ins().ushr_imm(offset.low_bits, 6);
    let source_limb = if output_limb == 0 {
        source_limb
    } else {
        builder.ins().iadd_imm(source_limb, output_limb as i64)
    };
    let low = load_dynamic_zero_limb(
        builder,
        input.clone(),
        input_layout,
        source_limb,
        offset.in_bounds,
    )?;
    let high_limb = builder.ins().iadd_imm(source_limb, 1);
    let high = load_dynamic_zero_limb(builder, input, input_layout, high_limb, offset.in_bounds)?;
    let shift = builder.ins().band_imm(offset.low_bits, 63);
    let low = builder.ins().ushr(low, shift);
    let high_shift = builder.ins().irsub_imm(shift, 64);
    let high = builder.ins().ishl(high, high_shift);
    let shift_is_zero = builder.ins().icmp_imm(IntCC::Equal, shift, 0);
    let zero = builder.ins().iconst(types::I64, 0);
    let high = builder.ins().select(shift_is_zero, zero, high);
    Ok(builder.ins().bor(low, high))
}

#[allow(clippy::too_many_arguments)]
fn lower_direct_dynamic_bit_slice(
    builder: &mut FunctionBuilder<'_>,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    input: ComputedValue,
    input_layout: &NativeValueLayout,
    start: ComputedValue,
    start_layout: &NativeValueLayout,
    result_layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    let offset =
        lower_dynamic_bit_offset(builder, start, start_layout, bits_bit_count(input_layout)?)?;
    match result_layout {
        NativeValueLayout::WideBits(wide) => {
            let destination = materialized_destination(
                builder,
                node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
            )?;
            for limb in 0..wide.limb_count {
                let value =
                    lower_dynamic_zero_window(builder, input.clone(), input_layout, offset, limb)?;
                store_wide_limb(builder, destination, *wide, limb, value);
            }
            Ok(ComputedValue::Address(destination))
        }
        NativeValueLayout::Scalar(scalar) => {
            let value = lower_dynamic_zero_window(builder, input, input_layout, offset, 0)?;
            let value = if scalar.clif_type() == types::I64 {
                value
            } else {
                builder.ins().ireduce(scalar.clif_type(), value)
            };
            Ok(ComputedValue::Scalar(mask_value(builder, value, *scalar)))
        }
        _ => Err(CompilerError::InvalidFunction(
            "dynamic bit slice result is not bits-typed".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_direct_decode(
    builder: &mut FunctionBuilder<'_>,
    node_ref: NodeRef,
    return_node: NodeRef,
    output_pointer: Value,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    input: ComputedValue,
    input_layout: &NativeValueLayout,
    result_layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    let offset =
        lower_dynamic_bit_offset(builder, input, input_layout, bits_bit_count(result_layout)?)?;
    let shift = builder.ins().band_imm(offset.low_bits, 63);
    let one = builder.ins().iconst(types::I64, 1);
    let shifted_one = builder.ins().ishl(one, shift);
    match result_layout {
        NativeValueLayout::WideBits(wide) => {
            let destination = materialized_destination(
                builder,
                node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
            )?;
            let target_limb = builder.ins().ushr_imm(offset.low_bits, 6);
            let zero = builder.ins().iconst(types::I64, 0);
            for limb in 0..wide.limb_count {
                let matches_limb = builder
                    .ins()
                    .icmp_imm(IntCC::Equal, target_limb, limb as i64);
                let selected = builder.ins().band(offset.in_bounds, matches_limb);
                let value = builder.ins().select(selected, shifted_one, zero);
                store_wide_limb(builder, destination, *wide, limb, value);
            }
            Ok(ComputedValue::Address(destination))
        }
        NativeValueLayout::Scalar(scalar) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let value = builder.ins().select(offset.in_bounds, shifted_one, zero);
            let value = if scalar.clif_type() == types::I64 {
                value
            } else {
                builder.ins().ireduce(scalar.clif_type(), value)
            };
            Ok(ComputedValue::Scalar(mask_value(builder, value, *scalar)))
        }
        _ => Err(CompilerError::InvalidFunction(
            "decode result is not bits-typed".into(),
        )),
    }
}

fn lower_wide_bitwise_unop(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    input: ComputedValue,
    input_layout: &NativeValueLayout,
    result_layout: WideBitsLayout,
    op: Unop,
) -> Result<(), CompilerError> {
    match op {
        Unop::Not => {
            for limb in 0..result_layout.limb_count {
                let input = load_raw_bits_limb(builder, input.clone(), input_layout, limb)?;
                let output = builder.ins().bnot(input);
                store_wide_limb(builder, destination, result_layout, limb, output);
            }
        }
        Unop::Neg => {
            let mut carry = builder.ins().iconst(types::I64, 1);
            for limb in 0..result_layout.limb_count {
                let input = load_raw_bits_limb(builder, input.clone(), input_layout, limb)?;
                let inverted = builder.ins().bnot(input);
                let sum = builder.ins().iadd(inverted, carry);
                let overflow = builder.ins().icmp(IntCC::UnsignedLessThan, sum, inverted);
                carry = builder.ins().uextend(types::I64, overflow);
                store_wide_limb(builder, destination, result_layout, limb, sum);
            }
        }
        Unop::Reverse => {
            let padding = result_layout.limb_count * 64 - result_layout.bit_count;
            for limb in 0..result_layout.limb_count {
                let source_limb = result_layout.limb_count - limb - 1;
                let source = load_raw_bits_limb(builder, input.clone(), input_layout, source_limb)?;
                let reversed = builder.ins().bitrev(source);
                let output = if padding == 0 {
                    reversed
                } else {
                    let low = builder.ins().ushr_imm(reversed, padding as i64);
                    let next = if source_limb == 0 {
                        builder.ins().iconst(types::I64, 0)
                    } else {
                        let adjacent = load_raw_bits_limb(
                            builder,
                            input.clone(),
                            input_layout,
                            source_limb - 1,
                        )?;
                        builder.ins().bitrev(adjacent)
                    };
                    let high = builder.ins().ishl_imm(next, (64 - padding) as i64);
                    builder.ins().bor(low, high)
                };
                store_wide_limb(builder, destination, result_layout, limb, output);
            }
        }
        Unop::Identity | Unop::OrReduce | Unop::AndReduce | Unop::XorReduce => {
            return Err(CompilerError::InvalidFunction(
                "unexpected wide stored unary operation".into(),
            ));
        }
    }
    Ok(())
}

fn lower_wide_reduction(
    builder: &mut FunctionBuilder<'_>,
    input: ComputedValue,
    input_layout: WideBitsLayout,
    op: Unop,
) -> Value {
    let mut combined = match op {
        Unop::AndReduce => builder.ins().iconst(types::I64, -1),
        _ => builder.ins().iconst(types::I64, 0),
    };
    for limb in 0..input_layout.limb_count {
        let mut value = load_wide_limb(builder, expect_address(input.clone()).unwrap(), limb);
        if op == Unop::AndReduce
            && limb + 1 == input_layout.limb_count
            && input_layout.bit_count % 64 != 0
        {
            let padding_ones = builder
                .ins()
                .iconst(types::I64, !input_layout.high_mask() as i64);
            value = builder.ins().bor(value, padding_ones);
        }
        combined = match op {
            Unop::OrReduce => builder.ins().bor(combined, value),
            Unop::AndReduce => builder.ins().band(combined, value),
            Unop::XorReduce => builder.ins().bxor(combined, value),
            _ => unreachable!("reduction operation"),
        };
    }
    match op {
        Unop::OrReduce => builder.ins().icmp_imm(IntCC::NotEqual, combined, 0),
        Unop::AndReduce => builder.ins().icmp_imm(IntCC::Equal, combined, -1),
        Unop::XorReduce => {
            let population = builder.ins().popcnt(combined);
            let parity = builder.ins().band_imm(population, 1);
            builder.ins().ireduce(types::I8, parity)
        }
        _ => unreachable!("reduction operation"),
    }
}

fn lower_wide_nary(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    args: &[NodeRef],
    values: &[Option<ComputedValue>],
    layout: WideBitsLayout,
    op: NaryOp,
) -> Result<(), CompilerError> {
    let Some((first, rest)) = args.split_first() else {
        return Err(CompilerError::UnsupportedNode(
            "wide n-ary operation requires an operand".into(),
        ));
    };
    for limb in 0..layout.limb_count {
        let mut result = load_wide_limb(
            builder,
            expect_address(computed_value_for(values, *first)?)?,
            limb,
        );
        for arg in rest {
            let rhs = load_wide_limb(
                builder,
                expect_address(computed_value_for(values, *arg)?)?,
                limb,
            );
            result = match op {
                NaryOp::And | NaryOp::Nand => builder.ins().band(result, rhs),
                NaryOp::Or | NaryOp::Nor => builder.ins().bor(result, rhs),
                NaryOp::Xor => builder.ins().bxor(result, rhs),
                NaryOp::Concat => unreachable!("concat has a dedicated wide lowering"),
            };
        }
        if matches!(op, NaryOp::Nand | NaryOp::Nor) {
            result = builder.ins().bnot(result);
        }
        store_wide_limb(builder, destination, layout, limb, result);
    }
    Ok(())
}

fn lower_wide_add_sub(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    lhs: ComputedValue,
    rhs: ComputedValue,
    layout: WideBitsLayout,
    subtract: bool,
) -> Result<(), CompilerError> {
    let lhs = expect_address(lhs)?;
    let rhs = expect_address(rhs)?;
    let mut carry_or_borrow = builder.ins().iconst(types::I64, 0);
    for limb in 0..layout.limb_count {
        let lhs_limb = load_wide_limb(builder, lhs, limb);
        let rhs_limb = load_wide_limb(builder, rhs, limb);
        let partial = if subtract {
            builder.ins().isub(lhs_limb, rhs_limb)
        } else {
            builder.ins().iadd(lhs_limb, rhs_limb)
        };
        let first_flag = if subtract {
            builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, lhs_limb, rhs_limb)
        } else {
            builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, partial, lhs_limb)
        };
        let result = if subtract {
            builder.ins().isub(partial, carry_or_borrow)
        } else {
            builder.ins().iadd(partial, carry_or_borrow)
        };
        let second_flag = if subtract {
            builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, partial, carry_or_borrow)
        } else {
            builder.ins().icmp(IntCC::UnsignedLessThan, result, partial)
        };
        let combined = builder.ins().bor(first_flag, second_flag);
        carry_or_borrow = builder.ins().uextend(types::I64, combined);
        store_wide_limb(builder, destination, layout, limb, result);
    }
    Ok(())
}

fn lower_wide_ext_carry_out(
    builder: &mut FunctionBuilder<'_>,
    lhs: ComputedValue,
    rhs: ComputedValue,
    c_in: Value,
    c_in_layout: ScalarLayout,
    layout: WideBitsLayout,
) -> Result<Value, CompilerError> {
    let lhs = expect_address(lhs)?;
    let rhs = expect_address(rhs)?;
    let mut carry = resize_integer_type_unsigned(builder, c_in, c_in_layout, types::I64);
    for limb in 0..layout.limb_count {
        let lhs_limb = load_wide_limb(builder, lhs, limb);
        let rhs_limb = load_wide_limb(builder, rhs, limb);
        let partial = builder.ins().iadd(lhs_limb, rhs_limb);
        let partial_carry = builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, partial, lhs_limb);
        let sum = builder.ins().iadd(partial, carry);
        let input_carry = builder.ins().icmp(IntCC::UnsignedLessThan, sum, partial);
        if limb + 1 == layout.limb_count && layout.bit_count % 64 != 0 {
            let semantic_carry_mask = 1u64 << (layout.bit_count % 64);
            let semantic_carry = builder.ins().band_imm(sum, semantic_carry_mask as i64);
            return Ok(builder.ins().icmp_imm(IntCC::NotEqual, semantic_carry, 0));
        }
        let next_carry = builder.ins().bor(partial_carry, input_carry);
        carry = builder.ins().uextend(types::I64, next_carry);
    }
    Ok(builder.ins().ireduce(types::I8, carry))
}

fn lower_wide_unsigned_compare(
    builder: &mut FunctionBuilder<'_>,
    lhs: ComputedValue,
    rhs: ComputedValue,
    layout: WideBitsLayout,
    condition: IntCC,
) -> Result<Value, CompilerError> {
    let lhs = expect_address(lhs)?;
    let rhs = expect_address(rhs)?;
    let mut equal = builder.ins().iconst(types::I8, 1);
    let initial = matches!(
        condition,
        IntCC::UnsignedGreaterThanOrEqual | IntCC::UnsignedLessThanOrEqual
    );
    let mut result = builder.ins().iconst(types::I8, i64::from(initial));
    for limb in (0..layout.limb_count).rev() {
        let lhs_limb = load_wide_limb(builder, lhs, limb);
        let rhs_limb = load_wide_limb(builder, rhs, limb);
        let comparison = builder.ins().icmp(condition, lhs_limb, rhs_limb);
        result = builder.ins().select(equal, comparison, result);
        let limb_equal = builder.ins().icmp(IntCC::Equal, lhs_limb, rhs_limb);
        equal = builder.ins().band(equal, limb_equal);
    }
    Ok(result)
}

fn lower_wide_comparison(
    builder: &mut FunctionBuilder<'_>,
    lhs: ComputedValue,
    rhs: ComputedValue,
    layout: WideBitsLayout,
    op: Binop,
) -> Result<Value, CompilerError> {
    if matches!(op, Binop::Eq | Binop::Ne) {
        let equal = lower_value_equality(builder, lhs, rhs, &NativeValueLayout::WideBits(layout))?;
        return Ok(if op == Binop::Eq {
            equal
        } else {
            builder.ins().icmp_imm(IntCC::Equal, equal, 0)
        });
    }
    let unsigned_condition = match op {
        Binop::Ugt | Binop::Sgt => IntCC::UnsignedGreaterThan,
        Binop::Uge | Binop::Sge => IntCC::UnsignedGreaterThanOrEqual,
        Binop::Ult | Binop::Slt => IntCC::UnsignedLessThan,
        Binop::Ule | Binop::Sle => IntCC::UnsignedLessThanOrEqual,
        _ => {
            return Err(CompilerError::InvalidFunction(
                "non-comparison passed to wide comparison lowering".into(),
            ));
        }
    };
    let unsigned = lower_wide_unsigned_compare(
        builder,
        lhs.clone(),
        rhs.clone(),
        layout,
        unsigned_condition,
    )?;
    if matches!(op, Binop::Ugt | Binop::Uge | Binop::Ult | Binop::Ule) {
        return Ok(unsigned);
    }
    let lhs_sign = bit_sign_condition(builder, lhs, &NativeValueLayout::WideBits(layout))?;
    let rhs_sign = bit_sign_condition(builder, rhs, &NativeValueLayout::WideBits(layout))?;
    let signs_differ = builder.ins().bxor(lhs_sign, rhs_sign);
    let signed_if_different = match op {
        Binop::Slt | Binop::Sle => lhs_sign,
        Binop::Sgt | Binop::Sge => rhs_sign,
        _ => unreachable!("signed operation"),
    };
    Ok(builder
        .ins()
        .select(signs_differ, signed_if_different, unsigned))
}

fn lower_wide_concat(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    destination: Value,
    args: &[NodeRef],
    values: &[Option<ComputedValue>],
    result_layout: WideBitsLayout,
) -> Result<(), CompilerError> {
    write_zero_value_to_storage(
        builder,
        destination,
        &NativeValueLayout::WideBits(result_layout),
    )?;
    let mut offset = 0usize;
    for arg in args.iter().rev() {
        let arg_layout = NativeValueLayout::from_type(&function.get_node(*arg).ty)?;
        let arg_width = bits_bit_count(&arg_layout)?;
        let arg_value = computed_value_for(values, *arg)?;
        for source_limb in 0..arg_width.div_ceil(64) {
            let value = load_raw_bits_limb(builder, arg_value.clone(), &arg_layout, source_limb)?;
            let destination_limb = (offset / 64) + source_limb;
            let shift = offset % 64;
            if destination_limb < result_layout.limb_count {
                let current = load_wide_limb(builder, destination, destination_limb);
                let positioned = if shift == 0 {
                    value
                } else {
                    builder.ins().ishl_imm(value, shift as i64)
                };
                let combined = builder.ins().bor(current, positioned);
                store_wide_limb(
                    builder,
                    destination,
                    result_layout,
                    destination_limb,
                    combined,
                );
            }
            if shift != 0 && destination_limb + 1 < result_layout.limb_count {
                let current = load_wide_limb(builder, destination, destination_limb + 1);
                let positioned = builder.ins().ushr_imm(value, (64 - shift) as i64);
                let combined = builder.ins().bor(current, positioned);
                store_wide_limb(
                    builder,
                    destination,
                    result_layout,
                    destination_limb + 1,
                    combined,
                );
            }
        }
        offset += arg_width;
    }
    if offset != result_layout.bit_count {
        return Err(CompilerError::InvalidFunction(
            "concat operands do not fill wide result type".into(),
        ));
    }
    Ok(())
}

fn lower_wide_ext_nary_add(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    destination: Value,
    terms: &[ir::ExtNaryAddTerm],
    values: &[Option<ComputedValue>],
    result_layout: WideBitsLayout,
) -> Result<(), CompilerError> {
    write_zero_value_to_storage(
        builder,
        destination,
        &NativeValueLayout::WideBits(result_layout),
    )?;
    for term in terms {
        let operand_layout = NativeValueLayout::from_type(&function.get_node(term.operand).ty)?;
        let operand = computed_value_for(values, term.operand)?;
        let mut carry_or_borrow = builder.ins().iconst(types::I64, 0);
        for limb in 0..result_layout.limb_count {
            let current = load_wide_limb(builder, destination, limb);
            let contribution = load_extended_bits_limb(
                builder,
                operand.clone(),
                &operand_layout,
                limb,
                term.signed,
            )?;
            let partial = if term.negated {
                builder.ins().isub(current, contribution)
            } else {
                builder.ins().iadd(current, contribution)
            };
            let first_flag = if term.negated {
                builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, current, contribution)
            } else {
                builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, partial, current)
            };
            let result = if term.negated {
                builder.ins().isub(partial, carry_or_borrow)
            } else {
                builder.ins().iadd(partial, carry_or_borrow)
            };
            let second_flag = if term.negated {
                builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, partial, carry_or_borrow)
            } else {
                builder.ins().icmp(IntCC::UnsignedLessThan, result, partial)
            };
            let flag = builder.ins().bor(first_flag, second_flag);
            carry_or_borrow = builder.ins().uextend(types::I64, flag);
            store_wide_limb(builder, destination, result_layout, limb, result);
        }
    }
    Ok(())
}

fn lower_mixed_ext_nary_add(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    terms: &[ir::ExtNaryAddTerm],
    values: &mut [Option<ComputedValue>],
    result_layout: ScalarLayout,
) -> Result<Value, CompilerError> {
    let mut result = builder.ins().iconst(result_layout.clif_type(), 0);
    for term in terms {
        let operand_layout = NativeValueLayout::from_type(&function.get_node(term.operand).ty)?;
        let resized = match operand_layout {
            NativeValueLayout::Scalar(term_layout) => {
                let value = scalar_value_for(builder, values, term.operand)?;
                if term.signed {
                    let signed = signed_value(builder, value, term_layout);
                    resize_signed(builder, signed, term_layout, result_layout)
                } else {
                    resize_unsigned(builder, value, term_layout, result_layout)
                }
            }
            NativeValueLayout::WideBits(_) => {
                let operand = computed_value_for(values, term.operand)?;
                let raw = load_raw_bits_limb(builder, operand, &operand_layout, 0)?;
                if result_layout.clif_type() == types::I64 {
                    raw
                } else {
                    builder.ins().ireduce(result_layout.clif_type(), raw)
                }
            }
            _ => {
                return Err(CompilerError::InvalidFunction(
                    "ext_nary_add term is not bits-typed".into(),
                ));
            }
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

fn lower_nary(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    node: &ir::Node,
    op: NaryOp,
    args: &[NodeRef],
    values: &mut [Option<ComputedValue>],
    layout: ScalarLayout,
) -> Result<Value, CompilerError> {
    if matches!(op, NaryOp::Concat) {
        return lower_concat(builder, function, node, args, values, layout);
    }
    let Some((first, rest)) = args.split_first() else {
        return Err(unsupported_node(node));
    };
    let mut result = scalar_value_for(builder, values, *first)?;
    for arg in rest {
        let rhs = scalar_value_for(builder, values, *arg)?;
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
    values: &mut [Option<ComputedValue>],
    layout: ScalarLayout,
) -> Result<Value, CompilerError> {
    let lhs_layout = ScalarLayout::from_type(&function.get_node(lhs).ty)?;
    let rhs_layout = ScalarLayout::from_type(&function.get_node(rhs).ty)?;
    let lhs_value = scalar_value_for(builder, values, lhs)?;
    let rhs_value = scalar_value_for(builder, values, rhs)?;
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
    values: &mut [Option<ComputedValue>],
    layout: ScalarLayout,
) -> Result<Value, CompilerError> {
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
                CompilerError::InvalidFunction(format!("concat width overflow at {}", node.text_id))
            })?;
        let arg = scalar_value_for(builder, values, *arg)?;
        let extended = resize_unsigned(builder, arg, arg_layout, layout);
        let positioned = if remaining_width == 0 {
            extended
        } else {
            builder.ins().ishl_imm(extended, remaining_width as i64)
        };
        result = builder.ins().bor(result, positioned);
    }
    if remaining_width != 0 {
        return Err(CompilerError::InvalidFunction(format!(
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
) -> Result<ComputedValue, CompilerError> {
    let arg_layout = ScalarLayout::from_type(&function.get_node(arg).ty)?;
    let (normalized_layout, clz_layout) = match result_layout {
        NativeValueLayout::Scalar(layout) => (*layout, None),
        NativeValueLayout::Tuple { fields, .. } if fields.len() == 2 => (
            require_scalar_layout(fields[0].layout.as_ref())?,
            Some(require_scalar_layout(fields[1].layout.as_ref())?),
        ),
        _ => {
            return Err(CompilerError::InvalidFunction(format!(
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
    values: &mut [Option<ComputedValue>],
    result_layout: ScalarLayout,
) -> Result<Value, CompilerError> {
    let mut result = builder.ins().iconst(result_layout.clif_type(), 0);
    for term in terms {
        let term_layout = ScalarLayout::from_type(&function.get_node(term.operand).ty)?;
        let value = scalar_value_for(builder, values, term.operand)?;
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
    values: &mut [Option<ComputedValue>],
    result_layout: &NativeValueLayout,
) -> Result<(), CompilerError> {
    let NativeValueLayout::Tuple { fields, .. } = result_layout else {
        return Err(CompilerError::InvalidFunction(
            "partial-product multiply did not have tuple result type".into(),
        ));
    };
    if fields.len() != 2 || fields[0].layout != fields[1].layout {
        return Err(CompilerError::InvalidFunction(
            "partial-product multiply requires two equal scalar tuple fields".into(),
        ));
    }
    let element_layout = require_scalar_layout(fields[0].layout.as_ref())?;
    let lhs_layout = ScalarLayout::from_type(&function.get_node(lhs).ty)?;
    let rhs_layout = ScalarLayout::from_type(&function.get_node(rhs).ty)?;
    let lhs_value = scalar_value_for(builder, values, lhs)?;
    let rhs_value = scalar_value_for(builder, values, rhs)?;
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
    node_ref: NodeRef,
    array: NodeRef,
    indices: &[NodeRef],
    assumed_in_bounds: bool,
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
    pointer_type: ClifType,
    event_sites: &HashMap<NodeRef, u32>,
    record_assumption_failure: FuncRef,
    execution_context: Value,
) -> Result<ComputedValue, CompilerError> {
    if indices.is_empty() {
        let value = computed_value_for(values, array)?;
        let input_layout = NativeValueLayout::from_type(&function.get_node(array).ty)?;
        if input_layout != *layout {
            return Err(CompilerError::InvalidFunction(format!(
                "zero-index array_index result layout disagrees with input at {}",
                node.text_id
            )));
        }
        return Ok(value);
    }
    if let NativeValueLayout::Scalar(layout) = layout {
        let value = deferred_scalar_array_index(
            function,
            node,
            node_ref,
            array,
            indices,
            assumed_in_bounds,
            values,
            *layout,
            event_sites,
            record_assumption_failure,
            execution_context,
        )?;
        return if assumed_in_bounds {
            Ok(ComputedValue::Scalar(materialize_scalar(builder, value)?))
        } else {
            Ok(value)
        };
    }
    let mut pointer = if layout.byte_count() == 0 {
        None
    } else {
        Some(address_value_for(values, array)?)
    };
    let mut current_layout = NativeValueLayout::from_type(&function.get_node(array).ty)?;
    let mut all_in_bounds = None;
    for index in indices {
        let NativeValueLayout::Array {
            element,
            element_count,
        } = current_layout
        else {
            return Err(CompilerError::InvalidFunction(format!(
                "array_index exceeds array dimensions at {}",
                node.text_id
            )));
        };
        if element_count == 0 {
            return Err(CompilerError::UnsupportedType(
                "zero-length native arrays are not supported for indexing".into(),
            ));
        }
        let index_layout = NativeValueLayout::from_type(&function.get_node(*index).ty)?;
        let bounded = bounded_array_index(
            builder,
            computed_value_for(values, *index)?,
            &index_layout,
            element_count,
            pointer_type,
            bits_node_is_statically_less_than(function, *index, element_count),
        )?;
        all_in_bounds = combine_optional_conditions(builder, all_in_bounds, bounded.in_bounds);
        if let Some(current_pointer) = pointer {
            pointer = Some(array_element_pointer_from_address_index(
                builder,
                current_pointer,
                bounded.address_index,
                element.byte_count(),
            ));
        }
        current_layout = *element;
    }
    if &current_layout != layout {
        return Err(CompilerError::InvalidFunction(format!(
            "array_index result layout disagrees with result type at {}",
            node.text_id
        )));
    }
    if assumed_in_bounds && let Some(all_in_bounds) = all_in_bounds {
        let site_id = event_site_id(event_sites, node_ref, node)?;
        let failed = builder.ins().icmp_imm(IntCC::Equal, all_in_bounds, 0);
        emit_conditional_site_call(
            builder,
            failed,
            record_assumption_failure,
            execution_context,
            site_id,
        );
    }
    match pointer {
        Some(pointer) => Ok(load_value_from_storage(builder, pointer, layout)),
        None => Ok(ComputedValue::ZeroSized),
    }
}

/// Builds a scalar array-index recipe whose address calculation is emitted
/// only when the selected scalar is consumed.
#[allow(clippy::too_many_arguments)]
fn deferred_scalar_array_index(
    function: &ir::Fn,
    node: &ir::Node,
    node_ref: NodeRef,
    array: NodeRef,
    indices: &[NodeRef],
    assumed_in_bounds: bool,
    values: &[Option<ComputedValue>],
    layout: ScalarLayout,
    event_sites: &HashMap<NodeRef, u32>,
    record_assumption_failure: FuncRef,
    execution_context: Value,
) -> Result<ComputedValue, CompilerError> {
    let array_pointer = address_value_for(values, array)?;
    let mut current_layout = NativeValueLayout::from_type(&function.get_node(array).ty)?;
    let mut dimensions = Vec::with_capacity(indices.len());
    for index in indices {
        let NativeValueLayout::Array {
            element,
            element_count,
        } = current_layout
        else {
            return Err(CompilerError::InvalidFunction(format!(
                "array_index exceeds array dimensions at {}",
                node.text_id
            )));
        };
        if element_count == 0 {
            return Err(CompilerError::UnsupportedType(
                "zero-length native arrays are not supported for indexing".into(),
            ));
        }
        dimensions.push(DeferredArrayIndexDimension {
            index: computed_value_for(values, *index)?,
            index_layout: NativeValueLayout::from_type(&function.get_node(*index).ty)?,
            element_count,
            element_byte_count: element.byte_count(),
            statically_in_bounds: bits_node_is_statically_less_than(
                function,
                *index,
                element_count,
            ),
        });
        current_layout = *element;
    }
    if current_layout != NativeValueLayout::Scalar(layout) {
        return Err(CompilerError::InvalidFunction(format!(
            "array_index result layout disagrees with result type at {}",
            node.text_id
        )));
    }
    let assumption_site = if assumed_in_bounds
        && dimensions
            .iter()
            .any(|dimension| !dimension.statically_in_bounds)
    {
        Some(DeferredAssumptionSite {
            callback: record_assumption_failure,
            execution_context,
            site_id: event_site_id(event_sites, node_ref, node)?,
        })
    } else {
        None
    };
    Ok(ComputedValue::ScalarArrayIndex(Box::new(
        DeferredScalarArrayIndex {
            array_pointer,
            dimensions,
            layout,
            assumption_site,
        },
    )))
}

/// Emits a deferred scalar array-index recipe and returns the selected value.
fn materialize_deferred_scalar_array_index(
    builder: &mut FunctionBuilder<'_>,
    value: DeferredScalarArrayIndex,
) -> Result<Value, CompilerError> {
    let mut pointer = value.array_pointer;
    let mut all_in_bounds = None;
    for dimension in value.dimensions {
        let bounded = bounded_array_index(
            builder,
            dimension.index,
            &dimension.index_layout,
            dimension.element_count,
            builder.func.dfg.value_type(pointer),
            dimension.statically_in_bounds,
        )?;
        all_in_bounds = combine_optional_conditions(builder, all_in_bounds, bounded.in_bounds);
        pointer = array_element_pointer_from_address_index(
            builder,
            pointer,
            bounded.address_index,
            dimension.element_byte_count,
        );
    }
    if let (Some(site), Some(all_in_bounds)) = (value.assumption_site, all_in_bounds) {
        let failed = builder.ins().icmp_imm(IntCC::Equal, all_in_bounds, 0);
        emit_conditional_site_call(
            builder,
            failed,
            site.callback,
            site.execution_context,
            site.site_id,
        );
    }
    let ComputedValue::Scalar(value) =
        load_value_from_storage(builder, pointer, &NativeValueLayout::Scalar(value.layout))
    else {
        unreachable!("scalar array-index recipe has scalar layout")
    };
    Ok(value)
}

fn lower_scalar_literal(
    builder: &mut FunctionBuilder<'_>,
    literal: &IrValue,
    layout: ScalarLayout,
) -> Result<Value, CompilerError> {
    let value = literal
        .to_u64()
        .map_err(|error| CompilerError::Value(error.to_string()))?;
    layout.validate_value(value)?;
    Ok(builder.ins().iconst(layout.clif_type(), value as i64))
}

fn lower_literal_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    literal: &IrValue,
    layout: &NativeValueLayout,
) -> Result<(), CompilerError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let value = lower_scalar_literal(builder, literal, *scalar)?;
            builder.ins().store(MemFlags::new(), value, destination, 0);
        }
        NativeValueLayout::WideBits(wide) => {
            let bits = literal
                .to_bits()
                .map_err(|error| CompilerError::Value(error.to_string()))?;
            let bytes = bits
                .to_le_bytes()
                .map_err(|error| CompilerError::Value(error.to_string()))?;
            for limb_index in 0..wide.limb_count {
                let start = limb_index * std::mem::size_of::<u64>();
                let mut limb_bytes = [0u8; std::mem::size_of::<u64>()];
                if start < bytes.len() {
                    let end = bytes.len().min(start + std::mem::size_of::<u64>());
                    limb_bytes[..end - start].copy_from_slice(&bytes[start..end]);
                }
                let value = builder
                    .ins()
                    .iconst(types::I64, u64::from_le_bytes(limb_bytes) as i64);
                builder.ins().store(
                    MemFlags::new(),
                    value,
                    destination,
                    (limb_index * std::mem::size_of::<u64>()) as i32,
                );
            }
        }
        NativeValueLayout::Array {
            element,
            element_count,
        } => {
            let elements = literal
                .get_elements()
                .map_err(|error| CompilerError::Value(error.to_string()))?;
            if elements.len() != *element_count {
                return Err(CompilerError::InvalidFunction(
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
                .map_err(|error| CompilerError::Value(error.to_string()))?;
            if elements.len() != fields.len() {
                return Err(CompilerError::InvalidFunction(
                    "literal tuple field count disagrees with its PIR type".into(),
                ));
            }
            for (field, child) in fields.iter().zip(elements.iter()) {
                let pointer = pointer_at_offset(builder, destination, field.offset);
                lower_literal_to_storage(builder, pointer, child, field.layout.as_ref())?;
            }
        }
        NativeValueLayout::Token => {}
    }
    Ok(())
}

fn lower_array_construction(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    elements: &[NodeRef],
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
    non_overlapping: bool,
) -> Result<(), CompilerError> {
    let NativeValueLayout::Array {
        element,
        element_count,
    } = layout
    else {
        return Err(CompilerError::InvalidFunction(
            "array node did not have array result type".into(),
        ));
    };
    if elements.len() != *element_count {
        return Err(CompilerError::InvalidFunction(
            "array node operand count disagrees with its result type".into(),
        ));
    }
    let values = preload_scalar_storage_views(builder, elements, values)?;
    for (index, value) in values.into_iter().enumerate() {
        let pointer = pointer_at_offset(builder, destination, index * element.byte_count());
        if non_overlapping {
            store_nonoverlapping_value_to_storage(builder, pointer, value, element)?;
        } else {
            store_value_to_storage(builder, pointer, value, element)?;
        }
    }
    Ok(())
}

fn lower_tuple_construction(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    elements: &[NodeRef],
    values: &[Option<ComputedValue>],
    layout: &NativeValueLayout,
    non_overlapping: bool,
) -> Result<(), CompilerError> {
    let NativeValueLayout::Tuple { fields, .. } = layout else {
        return Err(CompilerError::InvalidFunction(
            "tuple node did not have tuple result type".into(),
        ));
    };
    if elements.len() != fields.len() {
        return Err(CompilerError::InvalidFunction(
            "tuple node operand count disagrees with its result type".into(),
        ));
    }
    let values = preload_scalar_storage_views(builder, elements, values)?;
    for (value, field) in values.into_iter().zip(fields.iter()) {
        let pointer = pointer_at_offset(builder, destination, field.offset);
        if non_overlapping {
            store_nonoverlapping_value_to_storage(builder, pointer, value, field.layout.as_ref())?;
        } else {
            store_value_to_storage(builder, pointer, value, field.layout.as_ref())?;
        }
    }
    Ok(())
}

/// Loads deferred scalar aggregate views before aggregate destination writes.
fn preload_scalar_storage_views(
    builder: &mut FunctionBuilder<'_>,
    elements: &[NodeRef],
    values: &[Option<ComputedValue>],
) -> Result<Vec<ComputedValue>, CompilerError> {
    elements
        .iter()
        .map(|element| {
            let value = computed_value_for(values, *element)?;
            match value {
                ComputedValue::ScalarAddress { .. } | ComputedValue::ScalarArrayIndex(_) => {
                    Ok(ComputedValue::Scalar(materialize_scalar(builder, value)?))
                }
                _ => Ok(value),
            }
        })
        .collect()
}

fn lower_tuple_index(
    builder: &mut FunctionBuilder<'_>,
    tuple: NodeRef,
    index: usize,
    values: &[Option<ComputedValue>],
    result_layout: &NativeValueLayout,
    function: &ir::Fn,
    node: &ir::Node,
) -> Result<ComputedValue, CompilerError> {
    let tuple_layout = NativeValueLayout::from_type(&function.get_node(tuple).ty)?;
    let NativeValueLayout::Tuple { fields, .. } = tuple_layout else {
        return Err(CompilerError::InvalidFunction(format!(
            "tuple_index operand is not a tuple at {}",
            node.text_id
        )));
    };
    let field = fields.get(index).ok_or_else(|| {
        CompilerError::InvalidFunction(format!("tuple_index is out of bounds at {}", node.text_id))
    })?;
    if field.layout.as_ref() != result_layout {
        return Err(CompilerError::InvalidFunction(format!(
            "tuple_index result layout disagrees with result type at {}",
            node.text_id
        )));
    }
    if result_layout.byte_count() == 0 {
        return Ok(ComputedValue::ZeroSized);
    }
    let tuple_pointer = address_value_for(values, tuple)?;
    if let NativeValueLayout::Scalar(layout) = result_layout {
        Ok(ComputedValue::ScalarAddress {
            pointer: tuple_pointer,
            offset: field.offset,
            layout: *layout,
        })
    } else {
        let field_pointer = pointer_at_offset(builder, tuple_pointer, field.offset);
        Ok(deferred_value_from_storage(field_pointer, result_layout))
    }
}

fn lower_array_concat(
    builder: &mut FunctionBuilder<'_>,
    function: &ir::Fn,
    destination: Value,
    args: &[NodeRef],
    values: &[Option<ComputedValue>],
    result_layout: &NativeValueLayout,
    non_overlapping: bool,
) -> Result<(), CompilerError> {
    let NativeValueLayout::Array {
        element,
        element_count,
    } = result_layout
    else {
        return Err(CompilerError::InvalidFunction(
            "array_concat node did not have array result type".into(),
        ));
    };
    if args.is_empty() {
        return Err(CompilerError::InvalidFunction(
            "array_concat requires at least one operand".into(),
        ));
    }
    let mut destination_start = 0usize;
    for arg in args {
        let Type::Array(arg_ty) = &function.get_node(*arg).ty else {
            return Err(CompilerError::InvalidFunction(
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
            non_overlapping,
        )?;
        destination_start += arg_ty.element_count;
    }
    if destination_start != *element_count {
        return Err(CompilerError::InvalidFunction(
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
) -> Result<Value, CompilerError> {
    if layout.byte_count() == 0 {
        return Ok(builder.ins().iconst(types::I8, 1));
    }
    match layout {
        NativeValueLayout::Scalar(_) => {
            let lhs = materialize_scalar(builder, lhs)?;
            let rhs = materialize_scalar(builder, rhs)?;
            Ok(builder.ins().icmp(IntCC::Equal, lhs, rhs))
        }
        NativeValueLayout::WideBits(wide) => {
            let lhs = expect_address(lhs)?;
            let rhs = expect_address(rhs)?;
            let mut equal = builder.ins().iconst(types::I8, 1);
            for limb in 0..wide.limb_count {
                let offset = (limb * std::mem::size_of::<u64>()) as i32;
                let lhs_limb = builder.ins().load(types::I64, MemFlags::new(), lhs, offset);
                let rhs_limb = builder.ins().load(types::I64, MemFlags::new(), rhs, offset);
                let limb_equal = builder.ins().icmp(IntCC::Equal, lhs_limb, rhs_limb);
                equal = builder.ins().band(equal, limb_equal);
            }
            Ok(equal)
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
        NativeValueLayout::Token => Ok(builder.ins().iconst(types::I8, 1)),
    }
}

fn copy_array_elements(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    destination_start: usize,
    source: Value,
    source_element_count: usize,
    element_layout: &NativeValueLayout,
    non_overlapping: bool,
) -> Result<(), CompilerError> {
    let element_byte_count = element_layout.byte_count();
    let destination_offset = destination_start
        .checked_mul(element_byte_count)
        .ok_or_else(|| CompilerError::UnsupportedType("array copy offset overflow".into()))?;
    let byte_count = source_element_count
        .checked_mul(element_byte_count)
        .ok_or_else(|| CompilerError::UnsupportedType("array copy size overflow".into()))?;
    let destination = pointer_at_offset(builder, destination, destination_offset);
    if non_overlapping {
        emit_nonoverlapping_memory_copy(builder, destination, source, byte_count)
    } else {
        emit_memory_copy(builder, destination, source, byte_count)
    }
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
    non_overlapping: bool,
) -> Result<(), CompilerError> {
    let NativeValueLayout::Array {
        element: result_element,
        element_count: result_count,
    } = result_layout
    else {
        return Err(CompilerError::InvalidFunction(
            "array_slice node did not have array result type".into(),
        ));
    };
    let NativeValueLayout::Array {
        element: input_element,
        element_count: input_count,
    } = NativeValueLayout::from_type(&function.get_node(array).ty)?
    else {
        return Err(CompilerError::InvalidFunction(
            "array_slice operand did not have array type".into(),
        ));
    };
    if input_element.as_ref() != result_element.as_ref() {
        return Err(CompilerError::InvalidFunction(
            "array_slice element layout disagrees with result type".into(),
        ));
    }
    let source = address_value_for(values, array)?;
    let start_value = computed_value_for(values, start)?;
    let start_layout = NativeValueLayout::from_type(&function.get_node(start).ty)?;
    for output_index in 0..*result_count {
        let source_element = clamped_array_element_pointer_for_bits(
            builder,
            source,
            start_value.clone(),
            &start_layout,
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
        if non_overlapping {
            store_nonoverlapping_value_to_storage(
                builder,
                destination_element,
                element,
                result_element,
            )?;
        } else {
            store_value_to_storage(builder, destination_element, element, result_element)?;
        }
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
    event_sites: &HashMap<NodeRef, u32>,
    record_assumption_failure: FuncRef,
    execution_context: Value,
) -> Result<ComputedValue, CompilerError> {
    if indices.is_empty() {
        return Ok(computed_value_for(values, value)?);
    }
    let destination = if layout.byte_count() == 0 {
        None
    } else if scratch_plan.in_place_array_updates.contains(&node_ref) {
        Some(address_value_for(values, array)?)
    } else {
        let destination = materialized_destination(
            builder,
            node_ref,
            return_node,
            output_pointer,
            scratch_pointer,
            scratch_plan,
        )?;
        let array = computed_value_for(values, array)?;
        if node_ref != return_node {
            store_nonoverlapping_value_to_storage(builder, destination, array, layout)?;
        } else {
            store_value_to_storage(builder, destination, array, layout)?;
        }
        Some(destination)
    };
    let mut target = destination;
    let mut target_layout = layout.clone();
    let mut all_in_bounds = None;
    for index in indices {
        let NativeValueLayout::Array {
            element,
            element_count,
        } = target_layout
        else {
            return Err(CompilerError::InvalidFunction(format!(
                "array_update exceeds array dimensions at {}",
                node.text_id
            )));
        };
        let index_layout = NativeValueLayout::from_type(&function.get_node(*index).ty)?;
        if element_count == 0 {
            return Err(CompilerError::UnsupportedType(
                "zero-length native arrays are not supported for updates".into(),
            ));
        }
        let bounded = bounded_array_index(
            builder,
            computed_value_for(values, *index)?,
            &index_layout,
            element_count,
            pointer_type,
            bits_node_is_statically_less_than(function, *index, element_count),
        )?;
        all_in_bounds = combine_optional_conditions(builder, all_in_bounds, bounded.in_bounds);
        if let Some(current_target) = target {
            target = Some(array_element_pointer_from_address_index(
                builder,
                current_target,
                bounded.address_index,
                element.byte_count(),
            ));
        }
        target_layout = *element;
    }
    if assumed_in_bounds && let Some(condition) = all_in_bounds {
        let site_id = event_site_id(event_sites, node_ref, node)?;
        let failed = builder.ins().icmp_imm(IntCC::Equal, condition, 0);
        emit_conditional_site_call(
            builder,
            failed,
            record_assumption_failure,
            execution_context,
            site_id,
        );
    }
    let Some(target) = target else {
        return Ok(ComputedValue::ZeroSized);
    };
    let replacement = computed_value_for(values, value)?;
    if let Some(condition) = all_in_bounds {
        let original = load_value_from_storage(builder, target, &target_layout);
        write_selected_value_to_storage(
            builder,
            target,
            condition,
            replacement,
            Some(original),
            &target_layout,
        )?;
    } else {
        store_value_to_storage(builder, target, replacement, &target_layout)?;
    }
    Ok(ComputedValue::Address(destination.expect(
        "nonempty array update has materialized destination",
    )))
}

#[allow(clippy::too_many_arguments)]
fn lower_gate(
    builder: &mut FunctionBuilder<'_>,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    predicate: Value,
    gated: ComputedValue,
    layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    let enabled = builder.ins().icmp_imm(IntCC::NotEqual, predicate, 0);
    selected_value(
        builder,
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
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    selector: NodeRef,
    cases: &[NodeRef],
    default: Option<NodeRef>,
    values: &mut [Option<ComputedValue>],
    layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    let selector_value = scalar_value_for(builder, values, selector)?;
    let initial = default
        .or_else(|| cases.last().copied())
        .ok_or_else(|| CompilerError::InvalidFunction("sel requires a case or default".into()))?;
    let mut result = computed_value_for(values, initial)?;
    for (index, case) in cases.iter().enumerate().rev() {
        let selected = builder
            .ins()
            .icmp_imm(IntCC::Equal, selector_value, index as i64);
        result = selected_value(
            builder,
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
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    selector: NodeRef,
    cases: &[NodeRef],
    default: Option<NodeRef>,
    values: &mut [Option<ComputedValue>],
    layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    let Some(default) = default else {
        return Err(CompilerError::UnsupportedNode(format!(
            "priority_sel without default at node {}",
            node.text_id
        )));
    };
    let selector_value = scalar_value_for(builder, values, selector)?;
    let selector_layout = ScalarLayout::from_type(&function.get_node(selector).ty)?;
    let mut result = computed_value_for(values, default)?;
    for (index, case) in cases.iter().enumerate().rev() {
        let selected = bit_is_set(builder, selector_value, selector_layout, index);
        result = selected_value(
            builder,
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
    values: &mut [Option<ComputedValue>],
    layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    if cases.is_empty() {
        return Err(unsupported_node(node));
    }
    let selector_value = scalar_value_for(builder, values, selector)?;
    let selector_layout = ScalarLayout::from_type(&function.get_node(selector).ty)?;
    if layout.byte_count() == 0 {
        return Ok(ComputedValue::ZeroSized);
    }
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let mut result = builder.ins().iconst(scalar.clif_type(), 0);
            for (index, case) in cases.iter().enumerate() {
                let active = bit_is_set(builder, selector_value, selector_layout, index);
                let case = scalar_value_for(builder, values, *case)?;
                let with_case = builder.ins().bor(result, case);
                result = builder.ins().select(active, with_case, result);
            }
            Ok(ComputedValue::Scalar(mask_value(builder, result, *scalar)))
        }
        NativeValueLayout::WideBits(_)
        | NativeValueLayout::Array { .. }
        | NativeValueLayout::Tuple { .. } => {
            let destination = materialized_destination(
                builder,
                node_ref,
                return_node,
                output_pointer,
                scratch_pointer,
                scratch_plan,
            )?;
            write_zero_value_to_storage(builder, destination, layout)?;
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
        NativeValueLayout::Token => Ok(ComputedValue::ZeroSized),
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
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
    condition: Value,
    when_true: ComputedValue,
    when_false: Option<ComputedValue>,
    layout: &NativeValueLayout,
) -> Result<ComputedValue, CompilerError> {
    if layout.byte_count() == 0 {
        return Ok(ComputedValue::ZeroSized);
    }
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let true_value = materialize_scalar(builder, when_true)?;
            let false_value = match when_false {
                Some(value) => materialize_scalar(builder, value)?,
                None => builder.ins().iconst(scalar.clif_type(), 0),
            };
            Ok(ComputedValue::Scalar(builder.ins().select(
                condition,
                true_value,
                false_value,
            )))
        }
        NativeValueLayout::WideBits(_)
        | NativeValueLayout::Array { .. }
        | NativeValueLayout::Tuple { .. } => {
            let true_source = expect_address(when_true)?;
            let false_source = match when_false {
                Some(value) => expect_address(value)?,
                None => gate_zero_pointer(builder, scratch_pointer, scratch_plan)?,
            };
            Ok(ComputedValue::Address(builder.ins().select(
                condition,
                true_source,
                false_source,
            )))
        }
        NativeValueLayout::Token => Ok(ComputedValue::ZeroSized),
    }
}

fn write_selected_value_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    condition: Value,
    when_true: ComputedValue,
    when_false: Option<ComputedValue>,
    layout: &NativeValueLayout,
) -> Result<(), CompilerError> {
    if layout.byte_count() == 0 {
        return Ok(());
    }
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let false_value = match when_false {
                Some(value) => materialize_scalar(builder, value)?,
                None => builder.ins().iconst(scalar.clif_type(), 0),
            };
            let when_true = materialize_scalar(builder, when_true)?;
            let selected = builder.ins().select(condition, when_true, false_value);
            builder
                .ins()
                .store(MemFlags::new(), selected, destination, 0);
        }
        NativeValueLayout::WideBits(wide) => {
            let true_source = expect_address(when_true)?;
            let false_source = when_false.map(expect_address).transpose()?;
            for limb in 0..wide.limb_count {
                let offset = (limb * std::mem::size_of::<u64>()) as i32;
                let true_limb =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::new(), true_source, offset);
                let false_limb = if let Some(false_source) = false_source {
                    builder
                        .ins()
                        .load(types::I64, MemFlags::new(), false_source, offset)
                } else {
                    builder.ins().iconst(types::I64, 0)
                };
                let selected = builder.ins().select(condition, true_limb, false_limb);
                builder
                    .ins()
                    .store(MemFlags::new(), selected, destination, offset);
            }
        }
        NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. } => {
            let true_source = expect_address(when_true)?;
            let false_source = when_false.map(expect_address).transpose()?;
            emit_selected_memory_copy(
                builder,
                destination,
                condition,
                true_source,
                false_source,
                layout.byte_count(),
            )?;
        }
        NativeValueLayout::Token => {}
    }
    Ok(())
}

fn write_zero_value_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    layout: &NativeValueLayout,
) -> Result<(), CompilerError> {
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let zero = builder.ins().iconst(scalar.clif_type(), 0);
            builder.ins().store(MemFlags::new(), zero, destination, 0);
        }
        NativeValueLayout::WideBits(wide) => {
            let zero = builder.ins().iconst(types::I64, 0);
            for limb in 0..wide.limb_count {
                builder.ins().store(
                    MemFlags::new(),
                    zero,
                    destination,
                    (limb * std::mem::size_of::<u64>()) as i32,
                );
            }
        }
        NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. } => {
            emit_memory_zero(builder, destination, layout.byte_count())?;
        }
        NativeValueLayout::Token => {}
    }
    Ok(())
}

fn write_selected_or_value_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    condition: Value,
    value: ComputedValue,
    layout: &NativeValueLayout,
) -> Result<(), CompilerError> {
    if layout.byte_count() == 0 {
        return Ok(());
    }
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let current = builder
                .ins()
                .load(scalar.clif_type(), MemFlags::new(), destination, 0);
            let value = materialize_scalar(builder, value)?;
            let combined = builder.ins().bor(current, value);
            let selected = builder.ins().select(condition, combined, current);
            builder
                .ins()
                .store(MemFlags::new(), selected, destination, 0);
        }
        NativeValueLayout::WideBits(wide) => {
            let source = expect_address(value)?;
            for limb in 0..wide.limb_count {
                let offset = (limb * std::mem::size_of::<u64>()) as i32;
                let current = builder
                    .ins()
                    .load(types::I64, MemFlags::new(), destination, offset);
                let source_limb = builder
                    .ins()
                    .load(types::I64, MemFlags::new(), source, offset);
                let combined = builder.ins().bor(current, source_limb);
                let selected = builder.ins().select(condition, combined, current);
                builder
                    .ins()
                    .store(MemFlags::new(), selected, destination, offset);
            }
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
        NativeValueLayout::Token => {}
    }
    Ok(())
}

fn array_element_pointer_from_address_index(
    builder: &mut FunctionBuilder<'_>,
    array_pointer: Value,
    address_index: Value,
    element_byte_count: usize,
) -> Value {
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
fn clamped_array_element_pointer_for_bits(
    builder: &mut FunctionBuilder<'_>,
    array_pointer: Value,
    index: ComputedValue,
    index_layout: &NativeValueLayout,
    element_count: usize,
    additional_index: usize,
    element_byte_count: usize,
    pointer_type: ClifType,
) -> Result<Value, CompilerError> {
    let max_index = element_count - 1;
    if max_index as u128 > i64::MAX as u128 {
        return Err(CompilerError::UnsupportedType(
            "array dimensions larger than i64::MAX are unsupported".into(),
        ));
    }
    let bounded = if additional_index > max_index {
        builder.ins().iconst(pointer_type, max_index as i64)
    } else {
        let max_start = max_index - additional_index;
        let bounded_start = bounded_array_index(
            builder,
            index,
            index_layout,
            max_start + 1,
            pointer_type,
            false,
        )?
        .address_index;
        if additional_index == 0 {
            bounded_start
        } else {
            builder
                .ins()
                .iadd_imm(bounded_start, additional_index as i64)
        }
    };
    let offset = if element_byte_count == 1 {
        bounded
    } else {
        builder.ins().imul_imm(bounded, element_byte_count as i64)
    };
    Ok(builder.ins().iadd(array_pointer, offset))
}

#[derive(Clone, Copy)]
struct BoundedArrayIndex {
    address_index: Value,
    in_bounds: Option<Value>,
}

fn combine_optional_conditions(
    builder: &mut FunctionBuilder<'_>,
    lhs: Option<Value>,
    rhs: Option<Value>,
) -> Option<Value> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(builder.ins().band(lhs, rhs)),
        (Some(condition), None) | (None, Some(condition)) => Some(condition),
        (None, None) => None,
    }
}

fn bounded_array_index(
    builder: &mut FunctionBuilder<'_>,
    index: ComputedValue,
    layout: &NativeValueLayout,
    element_count: usize,
    pointer_type: ClifType,
    statically_in_bounds: bool,
) -> Result<BoundedArrayIndex, CompilerError> {
    let max_index = element_count - 1;
    if max_index as u128 > i64::MAX as u128 {
        return Err(CompilerError::UnsupportedType(
            "array dimensions larger than i64::MAX are unsupported".into(),
        ));
    }
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let index = materialize_scalar(builder, index)?;
            let address_index = resize_integer_type_unsigned(builder, index, *scalar, pointer_type);
            if statically_in_bounds || scalar.mask() <= max_index as u64 {
                return Ok(BoundedArrayIndex {
                    address_index,
                    in_bounds: None,
                });
            }
            let in_bounds =
                builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThan, index, element_count as i64);
            let final_index = builder.ins().iconst(pointer_type, max_index as i64);
            Ok(BoundedArrayIndex {
                address_index: builder.ins().select(in_bounds, address_index, final_index),
                in_bounds: Some(in_bounds),
            })
        }
        NativeValueLayout::WideBits(wide) => {
            let address = expect_address(index)?;
            let low = load_wide_limb(builder, address, 0);
            let address_index = if pointer_type == types::I64 {
                low
            } else {
                builder.ins().ireduce(pointer_type, low)
            };
            if statically_in_bounds {
                return Ok(BoundedArrayIndex {
                    address_index,
                    in_bounds: None,
                });
            }
            let mut high = builder.ins().iconst(types::I64, 0);
            for limb in 1..wide.limb_count {
                let next = load_wide_limb(builder, address, limb);
                high = builder.ins().bor(high, next);
            }
            let high_is_zero = builder.ins().icmp_imm(IntCC::Equal, high, 0);
            let low_in_bounds =
                builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThan, low, element_count as i64);
            let in_bounds = builder.ins().band(high_is_zero, low_in_bounds);
            let final_index = builder.ins().iconst(pointer_type, max_index as i64);
            Ok(BoundedArrayIndex {
                address_index: builder.ins().select(in_bounds, address_index, final_index),
                in_bounds: Some(in_bounds),
            })
        }
        _ => Err(CompilerError::InvalidFunction(
            "array index is not bits-typed".into(),
        )),
    }
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
) -> Result<Value, CompilerError> {
    if node_ref == return_node {
        return Ok(output_pointer);
    }
    let offset = scratch_plan
        .offsets
        .get(&node_ref)
        .cloned()
        .ok_or_else(|| {
            CompilerError::InvalidFunction(format!(
                "materialized aggregate node {} has no scratch assignment",
                node_ref.index
            ))
        })?;
    Ok(pointer_at_offset(builder, scratch_pointer, offset))
}

/// Returns the shared zero-valued backing storage used by memory-backed gates.
fn gate_zero_pointer(
    builder: &mut FunctionBuilder<'_>,
    scratch_pointer: Value,
    scratch_plan: &ScratchPlan,
) -> Result<Value, CompilerError> {
    let offset = scratch_plan.gate_zero_offset.ok_or_else(|| {
        CompilerError::InvalidFunction("memory-backed gate has no zero scratch assignment".into())
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

const INLINE_MEMORY_COPY_MAX_BYTE_COUNT: usize = 32;

#[derive(Clone, Copy)]
enum MemoryCopyOverlap {
    MayOverlap,
    NonOverlapping,
}

/// Emits an overlap-safe copy for one contiguous native storage region.
fn emit_memory_copy(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    source: Value,
    byte_count: usize,
) -> Result<(), CompilerError> {
    emit_memory_copy_with_overlap(
        builder,
        destination,
        source,
        byte_count,
        MemoryCopyOverlap::MayOverlap,
    )
}

/// Emits a copy for one contiguous native storage region known not to overlap.
fn emit_nonoverlapping_memory_copy(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    source: Value,
    byte_count: usize,
) -> Result<(), CompilerError> {
    emit_memory_copy_with_overlap(
        builder,
        destination,
        source,
        byte_count,
        MemoryCopyOverlap::NonOverlapping,
    )
}

/// Emits a fixed-size copy, inlining small regions and selecting the
/// appropriate libc helper for larger ones.
fn emit_memory_copy_with_overlap(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    source: Value,
    byte_count: usize,
    overlap: MemoryCopyOverlap,
) -> Result<(), CompilerError> {
    if byte_count == 0 {
        return Ok(());
    }
    if byte_count <= INLINE_MEMORY_COPY_MAX_BYTE_COUNT {
        return emit_inline_memory_copy(builder, destination, source, byte_count);
    }
    let pointer_type = builder.func.dfg.value_type(destination);
    debug_assert_eq!(pointer_type, builder.func.dfg.value_type(source));
    let byte_count = i64::try_from(byte_count).map_err(|_| {
        CompilerError::UnsupportedType("native aggregate copy exceeds i64::MAX bytes".into())
    })?;
    let mut signature = Signature::new(builder.func.signature.call_conv);
    signature.params.extend([
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
    ]);
    signature.returns.push(AbiParam::new(pointer_type));
    let signature = builder.import_signature(signature);
    let libcall = match overlap {
        MemoryCopyOverlap::MayOverlap => LibCall::Memmove,
        MemoryCopyOverlap::NonOverlapping => LibCall::Memcpy,
    };
    let copy = builder.import_function(ExtFuncData {
        name: ExternalName::LibCall(libcall),
        signature,
        colocated: false,
        patchable: false,
    });
    let byte_count = builder.ins().iconst(pointer_type, byte_count);
    builder.ins().call(copy, &[destination, source, byte_count]);
    Ok(())
}

/// Emits an overlap-safe fixed-size copy by loading every source chunk before
/// writing any destination chunk.
fn emit_inline_memory_copy(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    source: Value,
    byte_count: usize,
) -> Result<(), CompilerError> {
    debug_assert!(byte_count <= INLINE_MEMORY_COPY_MAX_BYTE_COUNT);
    let mut chunks = Vec::new();
    let mut offset = 0usize;
    while offset < byte_count {
        let remaining = byte_count - offset;
        let ty = if remaining >= 8 {
            types::I64
        } else if remaining >= 4 {
            types::I32
        } else if remaining >= 2 {
            types::I16
        } else {
            types::I8
        };
        let chunk_offset = i32::try_from(offset).map_err(|_| {
            CompilerError::UnsupportedType("inline aggregate copy offset overflow".into())
        })?;
        chunks.push((
            chunk_offset,
            builder
                .ins()
                .load(ty, MemFlags::new(), source, chunk_offset),
        ));
        offset = offset.checked_add(ty.bytes() as usize).ok_or_else(|| {
            CompilerError::UnsupportedType("inline aggregate copy size overflow".into())
        })?;
    }
    for (offset, value) in chunks {
        builder
            .ins()
            .store(MemFlags::new(), value, destination, offset);
    }
    Ok(())
}

/// Emits a libc zero-fill for one contiguous native storage region.
fn emit_memory_zero(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    byte_count: usize,
) -> Result<(), CompilerError> {
    if byte_count == 0 {
        return Ok(());
    }
    let pointer_type = builder.func.dfg.value_type(destination);
    let byte_count = i64::try_from(byte_count).map_err(|_| {
        CompilerError::UnsupportedType("native aggregate zero-fill exceeds i64::MAX bytes".into())
    })?;
    let mut signature = Signature::new(builder.func.signature.call_conv);
    signature.params.extend([
        AbiParam::new(pointer_type),
        AbiParam::new(types::I32),
        AbiParam::new(pointer_type),
    ]);
    signature.returns.push(AbiParam::new(pointer_type));
    let signature = builder.import_signature(signature);
    let memset = builder.import_function(ExtFuncData {
        name: ExternalName::LibCall(LibCall::Memset),
        signature,
        colocated: false,
        patchable: false,
    });
    let zero = builder.ins().iconst(types::I32, 0);
    let byte_count = builder.ins().iconst(pointer_type, byte_count);
    builder.ins().call(memset, &[destination, zero, byte_count]);
    Ok(())
}

/// Emits one aggregate selection without recursively selecting every leaf.
fn emit_selected_memory_copy(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    condition: Value,
    true_source: Value,
    false_source: Option<Value>,
    byte_count: usize,
) -> Result<(), CompilerError> {
    if let Some(false_source) = false_source {
        let source = builder.ins().select(condition, true_source, false_source);
        return emit_memory_copy(builder, destination, source, byte_count);
    }

    let true_block = builder.create_block();
    let false_block = builder.create_block();
    let done_block = builder.create_block();
    builder
        .ins()
        .brif(condition, true_block, &[], false_block, &[]);

    builder.switch_to_block(true_block);
    emit_memory_copy(builder, destination, true_source, byte_count)?;
    builder.ins().jump(done_block, &[]);
    builder.seal_block(true_block);

    builder.switch_to_block(false_block);
    emit_memory_zero(builder, destination, byte_count)?;
    builder.ins().jump(done_block, &[]);
    builder.seal_block(false_block);

    builder.switch_to_block(done_block);
    builder.seal_block(done_block);
    Ok(())
}

fn load_value_from_storage(
    builder: &mut FunctionBuilder<'_>,
    pointer: Value,
    layout: &NativeValueLayout,
) -> ComputedValue {
    if layout.byte_count() == 0 {
        return ComputedValue::ZeroSized;
    }
    match layout {
        NativeValueLayout::Scalar(scalar) => {
            let value = builder
                .ins()
                .load(scalar.clif_type(), MemFlags::new(), pointer, 0);
            ComputedValue::Scalar(mask_value(builder, value, *scalar))
        }
        NativeValueLayout::WideBits(_)
        | NativeValueLayout::Array { .. }
        | NativeValueLayout::Tuple { .. } => ComputedValue::Address(pointer),
        NativeValueLayout::Token => ComputedValue::ZeroSized,
    }
}

/// Returns a storage-backed value, deferring scalar loads until consumption.
fn deferred_value_from_storage(pointer: Value, layout: &NativeValueLayout) -> ComputedValue {
    if layout.byte_count() == 0 {
        return ComputedValue::ZeroSized;
    }
    match layout {
        NativeValueLayout::Scalar(scalar) => ComputedValue::ScalarAddress {
            pointer,
            offset: 0,
            layout: *scalar,
        },
        NativeValueLayout::WideBits(_)
        | NativeValueLayout::Array { .. }
        | NativeValueLayout::Tuple { .. } => ComputedValue::Address(pointer),
        NativeValueLayout::Token => ComputedValue::ZeroSized,
    }
}

fn store_value_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    value: ComputedValue,
    layout: &NativeValueLayout,
) -> Result<(), CompilerError> {
    store_value_to_storage_with_overlap(
        builder,
        destination,
        value,
        layout,
        MemoryCopyOverlap::MayOverlap,
    )
}

/// Stores a value when memory-backed source and destination regions are known
/// not to overlap.
fn store_nonoverlapping_value_to_storage(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    value: ComputedValue,
    layout: &NativeValueLayout,
) -> Result<(), CompilerError> {
    store_value_to_storage_with_overlap(
        builder,
        destination,
        value,
        layout,
        MemoryCopyOverlap::NonOverlapping,
    )
}

fn store_value_to_storage_with_overlap(
    builder: &mut FunctionBuilder<'_>,
    destination: Value,
    value: ComputedValue,
    layout: &NativeValueLayout,
    overlap: MemoryCopyOverlap,
) -> Result<(), CompilerError> {
    if layout.byte_count() == 0 {
        return Ok(());
    }
    match layout {
        NativeValueLayout::Scalar(_) => {
            let value = materialize_scalar(builder, value)?;
            builder.ins().store(MemFlags::new(), value, destination, 0);
        }
        NativeValueLayout::WideBits(wide) => {
            let source = expect_address(value)?;
            for limb in 0..wide.limb_count {
                let offset = (limb * std::mem::size_of::<u64>()) as i32;
                let value = builder
                    .ins()
                    .load(types::I64, MemFlags::new(), source, offset);
                builder
                    .ins()
                    .store(MemFlags::new(), value, destination, offset);
            }
        }
        NativeValueLayout::Array { .. } | NativeValueLayout::Tuple { .. } => {
            let source = expect_address(value)?;
            emit_memory_copy_with_overlap(
                builder,
                destination,
                source,
                layout.byte_count(),
                overlap,
            )?;
        }
        NativeValueLayout::Token => {}
    }
    Ok(())
}

fn computed_value_for(
    values: &[Option<ComputedValue>],
    node_ref: NodeRef,
) -> Result<ComputedValue, CompilerError> {
    values
        .get(node_ref.index)
        .cloned()
        .flatten()
        .ok_or_else(|| {
            CompilerError::InvalidFunction(format!(
                "operand node {} was not lowered before its user",
                node_ref.index
            ))
        })
}

fn scalar_value_for(
    builder: &mut FunctionBuilder<'_>,
    values: &mut [Option<ComputedValue>],
    node_ref: NodeRef,
) -> Result<Value, CompilerError> {
    let value = materialize_scalar(builder, computed_value_for(values, node_ref)?)?;
    values[node_ref.index] = Some(ComputedValue::Scalar(value));
    Ok(value)
}

fn address_value_for(
    values: &[Option<ComputedValue>],
    node_ref: NodeRef,
) -> Result<Value, CompilerError> {
    expect_address(computed_value_for(values, node_ref)?)
}

fn materialize_scalar(
    builder: &mut FunctionBuilder<'_>,
    value: ComputedValue,
) -> Result<Value, CompilerError> {
    match value {
        ComputedValue::Scalar(value) => Ok(value),
        ComputedValue::ScalarAddress {
            pointer,
            offset,
            layout,
        } => {
            let (pointer, offset) = match i32::try_from(offset) {
                Ok(offset) => (pointer, offset),
                Err(_) => (pointer_at_offset(builder, pointer, offset), 0),
            };
            let value = builder
                .ins()
                .load(layout.clif_type(), MemFlags::new(), pointer, offset);
            Ok(mask_value(builder, value, layout))
        }
        ComputedValue::ScalarArrayIndex(value) => {
            materialize_deferred_scalar_array_index(builder, *value)
        }
        ComputedValue::Address(_) | ComputedValue::ZeroSized => Err(
            CompilerError::InvalidFunction("array value used as a scalar".into()),
        ),
    }
}

fn expect_address(value: ComputedValue) -> Result<Value, CompilerError> {
    match value {
        ComputedValue::Address(value) => Ok(value),
        ComputedValue::Scalar(_)
        | ComputedValue::ScalarAddress { .. }
        | ComputedValue::ScalarArrayIndex(_)
        | ComputedValue::ZeroSized => Err(CompilerError::InvalidFunction(
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

fn unsupported_node(node: &ir::Node) -> CompilerError {
    CompilerError::UnsupportedNode(format!(
        "{} at node {} ({})",
        node.payload.get_operator(),
        node.text_id,
        node.ty
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_pir::ir_utils::get_topological;

    fn parse_function(ir: &str) -> ir::Fn {
        Parser::new(ir)
            .parse_and_validate_package()
            .expect("test PIR should parse and validate")
            .get_fn("f")
            .expect("function f should exist")
            .clone()
    }

    fn assert_is_topological(function: &ir::Fn, order: &[NodeRef]) {
        let positions = order
            .iter()
            .enumerate()
            .map(|(position, node_ref)| (*node_ref, position))
            .collect::<HashMap<_, _>>();
        for node_ref in order {
            for operand in operands(&function.get_node(*node_ref).payload) {
                assert!(
                    positions[&operand] < positions[node_ref],
                    "operand {} should precede user {}",
                    operand.index,
                    node_ref.index
                );
            }
        }
    }

    fn peak_live_scalar_count(function: &ir::Fn, order: &[NodeRef]) -> usize {
        let is_scalar = function
            .nodes
            .iter()
            .map(|node| scheduling_is_scalar(&node.ty).unwrap())
            .collect::<Vec<_>>();
        let mut remaining_user_count = vec![0usize; function.nodes.len()];
        for node_ref in order {
            for operand in unique_operands(function, *node_ref) {
                remaining_user_count[operand.index] += 1;
            }
        }
        let mut live_weight = 0usize;
        let mut peak = 0usize;
        for node_ref in order {
            live_weight += usize::from(is_scalar[node_ref.index]);
            peak = peak.max(live_weight);
            for operand in unique_operands(function, *node_ref) {
                remaining_user_count[operand.index] -= 1;
                if is_scalar[operand.index] && remaining_user_count[operand.index] == 0 {
                    live_weight -= 1;
                }
            }
        }
        peak
    }

    #[test]
    fn pressure_scheduler_delays_independent_parameter_chains() {
        let function = parse_function(
            r#"package test

fn f(a: bits[32] id=1, b: bits[32] id=2, c: bits[32] id=3, d: bits[32] id=4, e: bits[32] id=5) -> bits[32] {
  ab: bits[32] = add(a, b, id=6)
  cd: bits[32] = add(c, d, id=7)
  cde: bits[32] = add(cd, e, id=8)
  ret result: bits[32] = add(ab, cde, id=9)
}
"#,
        );
        let original = get_topological(&function)
            .into_iter()
            .filter(|node_ref| node_ref.index != 0)
            .collect::<Vec<_>>();
        let scheduled = reachable_scheduled_order(&function).unwrap();
        assert_is_topological(&function, &scheduled);
        assert_eq!(scheduled, reachable_scheduled_order(&function).unwrap());
        assert!(
            peak_live_scalar_count(&function, &scheduled)
                < peak_live_scalar_count(&function, &original),
            "pressure-aware order should reduce the peak live scalar count"
        );
    }
}
