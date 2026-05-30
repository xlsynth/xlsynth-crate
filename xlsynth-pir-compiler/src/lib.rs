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
/// leaf type is the corresponding native scalar carrier.
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
            _ => Err(JitError::UnsupportedType(ty.to_string())),
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
        }
    }

    /// Returns this layout's native alignment in bytes.
    pub fn alignment(&self) -> usize {
        match self {
            Self::Scalar(layout) => layout.byte_count,
            Self::Array { element, .. } => element.alignment(),
        }
    }

    /// Returns the element stride for a native array layout.
    pub fn element_stride(&self) -> Option<usize> {
        match self {
            Self::Array { element, .. } => Some(element.byte_count()),
            Self::Scalar(_) => None,
        }
    }

    /// Returns the scalar layout, if this layout represents bits directly.
    pub fn as_scalar(&self) -> Option<ScalarLayout> {
        match self {
            Self::Scalar(layout) => Some(*layout),
            Self::Array { .. } => None,
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
/// contiguous array storage.
pub struct PirFunctionJit {
    _module: JITModule,
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
        xlsynth_pir::ir_verify::verify_fn_types_agree_with_deduction(function)
            .map_err(JitError::InvalidFunction)?;

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
            _module: module,
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

    /// Runs generated code directly against caller-owned native storage.
    ///
    /// No parameter or result values are copied by this method. The generated
    /// code reads from the provided input pointers and writes the result at
    /// `output`.
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
        debug_assert!(self.scratch_alignment <= std::mem::align_of::<u64>());
        let mut scratch_words =
            vec![0u64; self.scratch_byte_count.div_ceil(std::mem::size_of::<u64>())];
        let scratch = if scratch_words.is_empty() {
            ptr::null_mut()
        } else {
            scratch_words.as_mut_ptr().cast::<u8>()
        };
        // SAFETY: the caller upholds the pointer and layout contract stated
        // above; scratch storage is aligned and remains alive for this call;
        // the function was finalized with this exact ABI.
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
        let param_layouts = self
            .param_layouts
            .iter()
            .map(require_scalar_layout)
            .collect::<Result<Vec<_>, _>>()?;
        let result_layout = require_scalar_layout(&self.result_layout)?;
        let scalar_args = args
            .iter()
            .zip(&param_layouts)
            .map(|(arg, layout)| {
                let bits = arg
                    .to_bits()
                    .map_err(|error| JitError::Value(error.to_string()))?;
                if bits.get_bit_count() != layout.bit_count {
                    return Err(JitError::InvalidArgument(format!(
                        "expected bits[{}] argument, got bits[{}]",
                        layout.bit_count,
                        bits.get_bit_count()
                    )));
                }
                bits.to_u64()
                    .map_err(|error| JitError::Value(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result = self.run_u64(&scalar_args)?;
        IrValue::make_ubits(result_layout.bit_count, result)
            .map_err(|error| JitError::Value(error.to_string()))
    }
}

fn require_scalar_layout(layout: &NativeValueLayout) -> Result<ScalarLayout, JitError> {
    layout.as_scalar().ok_or_else(|| {
        JitError::UnsupportedType("dynamic adapter supports scalar function signatures only".into())
    })
}

enum NativeScalar {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
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
    /// Assigns native scratch slots for materialized intermediate arrays.
    fn for_function(function: &ir::Fn, order: &[NodeRef]) -> Result<Self, JitError> {
        let mut offsets = HashMap::new();
        let mut byte_count = 0usize;
        let mut alignment = 1usize;
        for node_ref in order {
            let node = function.get_node(*node_ref);
            let layout = NativeValueLayout::from_type(&node.ty)?;
            if !needs_array_destination(node)
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

fn needs_array_destination(node: &ir::Node) -> bool {
    matches!(
        &node.payload,
        NodePayload::Literal(_) | NodePayload::Array(_)
    ) && matches!(node.ty, Type::Array(_))
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
                NativeValueLayout::Array { .. } => {
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
            NodePayload::Unop(op, arg) => {
                let scalar_layout = require_scalar_layout(&layout)?;
                let arg_value = scalar_value_for(&values, *arg)?;
                let raw = match op {
                    Unop::Identity => arg_value,
                    Unop::Not => builder.ins().bnot(arg_value),
                    Unop::Neg => builder.ins().ineg(arg_value),
                    _ => return Err(unsupported_node(node)),
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
        Binop::Umul | Binop::Smul => builder.ins().imul(lhs_value, rhs_value),
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
            let rhs_signed = signed_value(builder, rhs_value, lhs_layout);
            let condition = match op {
                Binop::Sgt => IntCC::SignedGreaterThan,
                Binop::Sge => IntCC::SignedGreaterThanOrEqual,
                Binop::Slt => IntCC::SignedLessThan,
                Binop::Sle => IntCC::SignedLessThanOrEqual,
                _ => unreachable!("signed comparison branch selected above"),
            };
            builder.ins().icmp(condition, lhs_signed, rhs_signed)
        }
        Binop::Shll => {
            let out_of_bounds = builder.ins().icmp_imm(
                IntCC::UnsignedGreaterThanOrEqual,
                rhs_value,
                lhs_layout.bit_count as i64,
            );
            let shift = resize_unsigned(builder, rhs_value, rhs_layout, lhs_layout);
            let shifted = builder.ins().ishl(lhs_value, shift);
            let zero = builder.ins().iconst(layout.clif_type(), 0);
            builder.ins().select(out_of_bounds, zero, shifted)
        }
        _ => return Err(unsupported_node(node)),
    };
    Ok(mask_value(builder, raw, layout))
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
        return Err(unsupported_node(node));
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
                "materialized array node {} has no scratch assignment",
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
        NativeValueLayout::Array { .. } => ComputedValue::Address(pointer),
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
