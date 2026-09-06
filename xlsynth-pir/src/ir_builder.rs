// SPDX-License-Identifier: Apache-2.0

//! Checked construction of PIR function and block graphs.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::IrValue;
use crate::ir::{self, Binop, NaryOp, Node, NodePayload, NodeRef, Type, Unop};
use crate::ir_deduce::deduce_result_type;
use crate::ir_rebase_ids::{package_max_emitted_node_id, rebase_fn_ids_in_place};
use crate::ir_utils::operands;
use crate::ir_verify::{
    verify_function, verify_function_in_package, verify_function_signature,
    verify_node_xls_semantics, verify_package,
};

mod block;
mod block_cycles;
mod ops;

pub use block::{
    BInstantiation, BRegister, BlockBuilder, BlockState, RegisterWriteOptions, ResetBehavior,
};
pub use ops::{NaryAddOptions, NaryAddTerm, NormalizeLeftOptions};

static NEXT_BUILDER_ID: AtomicUsize = AtomicUsize::new(1);

/// An opaque, cheap-to-copy reference to a value in one builder.
///
/// Handles cannot be constructed externally or used with another builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BValue {
    builder_id: usize,
    node: NodeRef,
}

/// An invalid construction request; rejected requests leave the builder usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuilderError {
    ForeignValue,
    ForeignRegister,
    ForeignInstantiation,
    ParameterAfterBody,
    InvalidName(String),
    DuplicateName(String),
    InvalidOperation(String),
    WidthOverflow,
}

impl std::fmt::Display for BuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForeignValue => write!(f, "value belongs to a different builder"),
            Self::ForeignRegister => write!(f, "register belongs to a different builder"),
            Self::ForeignInstantiation => write!(f, "instantiation belongs to a different builder"),
            Self::ParameterAfterBody => write!(f, "parameters must precede body nodes"),
            Self::InvalidName(name) => write!(f, "invalid IR identifier: {name:?}"),
            Self::DuplicateName(name) => write!(f, "duplicate IR identifier: {name:?}"),
            Self::InvalidOperation(reason) => f.write_str(reason),
            Self::WidthOverflow => write!(f, "IR width or element count overflows usize"),
        }
    }
}

impl std::error::Error for BuilderError {}

mod sealed {
    use super::{BuilderError, NodeRef};

    /// Interface-specific restrictions applied by the shared graph builder.
    pub trait State {
        fn check_name_available(
            &self,
            _name: &str,
            _node: Option<NodeRef>,
        ) -> Result<(), BuilderError> {
            Ok(())
        }
    }
}

/// The function interface carried by a checked builder.
#[doc(hidden)]
#[derive(Default)]
pub struct FunctionState {
    params: Vec<NodeRef>,
}

impl sealed::State for FunctionState {}

/// Builds a function using PIR types, values, and deterministic node IDs.
///
/// Add parameters first, then body nodes. Every operation checks its operand
/// types immediately; `build` consumes the builder and verifies the function.
/// Names are optional on body nodes and can be assigned with `set_name`.
/// Standard operations emit XLS-compatible nodes; methods prefixed with `ext_`
/// explicitly construct PIR extensions, which must be desugared for XLS tools.
///
/// ```
/// use xlsynth_pir::{FnBuilder, ir::Type};
/// let mut builder = FnBuilder::new("add");
/// let lhs = builder.param("lhs", Type::Bits(32))?;
/// let rhs = builder.param("rhs", Type::Bits(32))?;
/// let sum = builder.add(lhs, rhs)?;
/// let function = builder.build(sum)?;
/// assert_eq!(function.ret_ty, Type::Bits(32));
/// # Ok::<(), xlsynth_pir::BuilderError>(())
/// ```
pub type FnBuilder = Builder<FunctionState>;

/// Shared checked graph operations, specialized by function or block interface.
///
/// Construct this through [`FnBuilder`] or [`BlockBuilder`]. Its graph and
/// interface stay together; neither is exposed for unchecked mutation.
pub struct Builder<S: sealed::State> {
    id: usize,
    graph: ir::NodeGraph,
    names: HashMap<String, NodeRef>,
    callees: BTreeMap<String, ir::FunctionType>,
    state: S,
}

macro_rules! binary_op {
    ($name:ident, $op:ident) => {
        #[doc = concat!("Adds a `", stringify!($name), "` operation.")]
        pub fn $name(&mut self, lhs: BValue, rhs: BValue) -> Result<BValue, BuilderError> {
            self.add_node(
                NodePayload::Binop(Binop::$op, self.node(lhs)?, self.node(rhs)?),
                None,
            )
        }
    };
}

macro_rules! bitwise_op {
    ($name:ident, $op:ident) => {
        #[doc = concat!("Adds a two-operand `", stringify!($name), "` operation.")]
        pub fn $name(&mut self, lhs: BValue, rhs: BValue) -> Result<BValue, BuilderError> {
            self.add_node(
                NodePayload::Nary(NaryOp::$op, self.nodes(&[lhs, rhs])?),
                None,
            )
        }
    };
}

macro_rules! unary_op {
    ($name:ident, $op:ident) => {
        #[doc = concat!("Adds a `", stringify!($name), "` operation.")]
        pub fn $name(&mut self, arg: BValue) -> Result<BValue, BuilderError> {
            self.add_node(NodePayload::Unop(Unop::$op, self.node(arg)?), None)
        }
    };
}

/// Checks the identifier subset shared by PIR and XLS, excluding reserved
/// words.
///
/// Adapters importing external names can use this before applying their own
/// legalization and collision-resolution policy.
pub fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let valid_start = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let reserved = matches!(
        name,
        "fn" | "bits"
            | "token"
            | "ret"
            | "package"
            | "proc"
            | "chan"
            | "chan_interface"
            | "reg"
            | "next"
            | "block"
            | "clock"
            | "instantiation"
            | "top"
            | "file_number"
            | "proc_instantiation"
            | "stage"
            | "true"
            | "false"
    );
    valid_start && !reserved && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Validates a name without modifying or silently uniquifying it.
fn check_name(name: &str) -> Result<(), BuilderError> {
    if is_valid_identifier(name) {
        Ok(())
    } else {
        Err(BuilderError::InvalidName(name.to_string()))
    }
}

/// Checks composite size arithmetic before the deducer or interpreter uses it.
fn checked_flat_width(ty: &Type) -> Result<usize, BuilderError> {
    ty.checked_bit_count().ok_or(BuilderError::WidthOverflow)
}

impl<S: sealed::State> Builder<S> {
    /// Creates shared state without exposing a separately mutable graph.
    fn from_state(name: &str, state: S) -> Self {
        let id = NEXT_BUILDER_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("builder identity space exhausted");
        Self {
            id,
            graph: ir::NodeGraph::new(name),
            names: HashMap::new(),
            callees: BTreeMap::new(),
            state,
        }
    }

    fn node(&self, value: BValue) -> Result<NodeRef, BuilderError> {
        if value.builder_id != self.id {
            return Err(BuilderError::ForeignValue);
        }
        Ok(value.node)
    }

    fn nodes(&self, values: &[BValue]) -> Result<Vec<NodeRef>, BuilderError> {
        values.iter().map(|value| self.node(*value)).collect()
    }

    fn bits_width(&self, value: BValue) -> Result<usize, BuilderError> {
        match self.get_type(value)? {
            Type::Bits(width) => Ok(*width),
            ty => Err(BuilderError::InvalidOperation(format!(
                "expected bits, got {ty}"
            ))),
        }
    }

    /// Appends one node transactionally, checking only that node's operands.
    fn add_node(
        &mut self,
        payload: NodePayload,
        explicit_type: Option<Type>,
    ) -> Result<BValue, BuilderError> {
        let operand_types: Vec<Type> = operands(&payload)
            .iter()
            .map(|node| self.graph.get_node_ty(*node).clone())
            .collect();
        let deduced = deduce_result_type(&payload, &operand_types)
            .map_err(|error| BuilderError::InvalidOperation(error.to_string()))?;
        let ty = match (explicit_type, deduced) {
            (Some(explicit), Some(deduced)) if explicit != deduced => {
                return Err(BuilderError::InvalidOperation(format!(
                    "expected result type {explicit}, deduced {deduced}"
                )));
            }
            (Some(explicit), _) => explicit,
            (None, Some(deduced)) => deduced,
            (None, None) => {
                return Err(BuilderError::InvalidOperation(
                    "operation requires an explicit result type".to_string(),
                ));
            }
        };
        checked_flat_width(&ty)?;
        let node = NodeRef {
            index: self.graph.nodes.len(),
        };
        self.graph.nodes.push(Node {
            text_id: node.index,
            name: None,
            ty,
            payload,
            pos: None,
        });
        let validation = verify_node_xls_semantics(&self.graph, node.index)
            .and_then(|()| self.graph.nodes[node.index].payload.validate(&self.graph));
        if let Err(reason) = validation {
            self.graph.nodes.pop();
            return Err(BuilderError::InvalidOperation(reason));
        }
        Ok(BValue {
            builder_id: self.id,
            node,
        })
    }

    /// Adds a literal, including aggregate, token, and typed empty-array
    /// values.
    pub fn literal(&mut self, value: IrValue) -> Result<BValue, BuilderError> {
        let ty = value.type_();
        self.add_node(NodePayload::Literal(value), Some(ty))
    }

    /// Assigns a unique node name while respecting interface reservations.
    pub fn set_name(&mut self, value: BValue, name: &str) -> Result<(), BuilderError> {
        let node = self.node(value)?;
        self.check_node_name(name, Some(node))?;
        let data = self.graph.get_node_mut(node);
        if let Some(old_name) = data.name.replace(name.to_string()) {
            self.names.remove(&old_name);
        }
        self.names.insert(name.to_string(), node);
        Ok(())
    }

    /// Borrows the type of a value belonging to this builder.
    pub fn get_type(&self, value: BValue) -> Result<&Type, BuilderError> {
        Ok(self.graph.get_node_ty(self.node(value)?))
    }

    /// Returns the latest successfully constructed node, if there is one.
    pub fn last_value(&self) -> Option<BValue> {
        (self.graph.nodes.len() > 1).then(|| BValue {
            builder_id: self.id,
            node: NodeRef {
                index: self.graph.nodes.len() - 1,
            },
        })
    }

    /// Checks node names against both graph names and interface reservations.
    fn check_node_name(&self, name: &str, node: Option<NodeRef>) -> Result<(), BuilderError> {
        check_name(name)?;
        if self
            .names
            .get(name)
            .is_some_and(|existing| Some(*existing) != node)
        {
            return Err(BuilderError::DuplicateName(name.to_string()));
        }
        self.state.check_name_available(name, node)
    }

    /// Resolves every signature supplied to a call against its destination
    /// package.
    fn check_callees(&self, package: &ir::Package) -> Result<(), BuilderError> {
        for (name, expected) in &self.callees {
            let actual = package.get_fn(name).ok_or_else(|| {
                BuilderError::InvalidOperation(format!("callee '{name}' is missing from package"))
            })?;
            if actual.get_type() != *expected {
                return Err(BuilderError::InvalidOperation(format!(
                    "callee '{name}' in package has a different signature than the one supplied to the builder"
                )));
            }
        }
        Ok(())
    }

    /// Checks a referenced signature without storing it until the node
    /// succeeds.
    fn check_callee(&self, callee: &ir::Fn) -> Result<(), BuilderError> {
        check_name(&callee.name)?;
        verify_function_signature(callee)
            .map_err(|error| BuilderError::InvalidOperation(error.to_string()))?;
        if callee.name == self.graph.name {
            return Err(BuilderError::InvalidOperation(format!(
                "recursive reference to '{}' is not allowed",
                callee.name
            )));
        }
        if self
            .callees
            .get(&callee.name)
            .is_some_and(|known| *known != callee.get_type())
        {
            return Err(BuilderError::InvalidOperation(format!(
                "conflicting signatures supplied for callee '{}'",
                callee.name
            )));
        }
        Ok(())
    }

    /// Invokes a function using its typed signature, resolving its name at
    /// build.
    pub fn invoke(&mut self, callee: &ir::Fn, args: &[BValue]) -> Result<BValue, BuilderError> {
        self.check_callee(callee)?;
        let args = self.nodes(args)?;
        if args.len() != callee.params.len() {
            return Err(BuilderError::InvalidOperation(format!(
                "callee '{}' expects {} arguments, got {}",
                callee.name,
                callee.params.len(),
                args.len()
            )));
        }
        for (arg, param) in args.iter().zip(callee.param_nodes()) {
            let actual = self.graph.get_node_ty(*arg);
            if actual != &param.ty {
                return Err(BuilderError::InvalidOperation(format!(
                    "argument '{}' of '{}' requires {}, got {}",
                    param.param_name(),
                    callee.name,
                    param.ty,
                    actual
                )));
            }
        }
        let value = self.add_node(
            NodePayload::Invoke {
                to_apply: callee.name.clone(),
                operands: args,
            },
            Some(callee.ret_ty.clone()),
        )?;
        self.callees
            .entry(callee.name.clone())
            .or_insert_with(|| callee.get_type());
        Ok(value)
    }

    /// Adds a counted loop whose body is `(index, carry, invariants...) ->
    /// carry`.
    ///
    /// The unsigned induction parameter must represent every visited index;
    /// zero- and one-trip loops require at least one induction bit, as in XLS.
    pub fn counted_for(
        &mut self,
        init: BValue,
        trip_count: usize,
        stride: usize,
        body: &ir::Fn,
        invariant_args: &[BValue],
    ) -> Result<BValue, BuilderError> {
        self.check_callee(body)?;
        let init = self.node(init)?;
        let invariant_args = self.nodes(invariant_args)?;
        let param_count = invariant_args
            .len()
            .checked_add(2)
            .ok_or(BuilderError::WidthOverflow)?;
        if body.params.len() != param_count {
            return Err(BuilderError::InvalidOperation(format!(
                "counted_for body '{}' requires {} parameters, got {}",
                body.name,
                param_count,
                body.params.len()
            )));
        }
        let max_index = stride
            .checked_mul(trip_count.saturating_sub(1))
            .ok_or(BuilderError::WidthOverflow)?;
        let minimum_width = if trip_count <= 1 {
            1
        } else {
            (usize::BITS - max_index.leading_zeros()) as usize
        };
        if !matches!(body.get_param(0).ty, Type::Bits(width) if width >= minimum_width) {
            return Err(BuilderError::InvalidOperation(format!(
                "counted_for induction parameter must be bits[N] with N >= {minimum_width}"
            )));
        }
        let carry_type = self.graph.get_node_ty(init);
        if &body.get_param(1).ty != carry_type || &body.ret_ty != carry_type {
            return Err(BuilderError::InvalidOperation(format!(
                "counted_for carry parameter and return must have type {carry_type}"
            )));
        }
        for (arg, param) in invariant_args.iter().zip(body.param_nodes().skip(2)) {
            if self.graph.get_node_ty(*arg) != &param.ty {
                return Err(BuilderError::InvalidOperation(format!(
                    "counted_for invariant '{}' requires {}",
                    param.param_name(),
                    param.ty
                )));
            }
        }
        let value = self.add_node(
            NodePayload::CountedFor {
                init,
                trip_count,
                stride,
                body: body.name.clone(),
                invariant_args,
            },
            Some(body.ret_ty.clone()),
        )?;
        self.callees
            .entry(body.name.clone())
            .or_insert_with(|| body.get_type());
        Ok(value)
    }

    binary_op!(add, Add);
    binary_op!(sub, Sub);
    binary_op!(eq, Eq);
    binary_op!(ne, Ne);
    binary_op!(ule, Ule);
    binary_op!(ult, Ult);
    binary_op!(uge, Uge);
    binary_op!(ugt, Ugt);
    binary_op!(sle, Sle);
    binary_op!(slt, Slt);
    binary_op!(sge, Sge);
    binary_op!(sgt, Sgt);
    binary_op!(udiv, Udiv);
    binary_op!(sdiv, Sdiv);
    binary_op!(umod, Umod);
    binary_op!(smod, Smod);
    binary_op!(shll, Shll);
    binary_op!(shrl, Shrl);
    binary_op!(shra, Shra);
    bitwise_op!(and, And);
    bitwise_op!(nand, Nand);
    bitwise_op!(or, Or);
    bitwise_op!(nor, Nor);
    bitwise_op!(xor, Xor);
    unary_op!(not, Not);
    unary_op!(neg, Neg);
    unary_op!(rev, Reverse);
    unary_op!(identity, Identity);
    unary_op!(or_reduce, OrReduce);
    unary_op!(and_reduce, AndReduce);
    unary_op!(xor_reduce, XorReduce);

    /// Sign-extends a nonempty bits value without truncating.
    pub fn sign_extend(
        &mut self,
        arg: BValue,
        new_bit_count: usize,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::SignExt {
                arg: self.node(arg)?,
                new_bit_count,
            },
            None,
        )
    }

    /// Zero-extends a bits value without truncating.
    pub fn zero_extend(
        &mut self,
        arg: BValue,
        new_bit_count: usize,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::ZeroExt {
                arg: self.node(arg)?,
                new_bit_count,
            },
            None,
        )
    }

    /// Selects a constant, in-bounds range of bits.
    pub fn bit_slice(
        &mut self,
        arg: BValue,
        start: usize,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::BitSlice {
                arg: self.node(arg)?,
                start,
                width,
            },
            None,
        )
    }

    /// Selects bits at a dynamic offset, zero-filling positions past the input.
    pub fn dynamic_bit_slice(
        &mut self,
        arg: BValue,
        start: BValue,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::DynamicBitSlice {
                arg: self.node(arg)?,
                start: self.node(start)?,
                width,
            },
            None,
        )
    }

    /// Replaces bits beginning at a dynamic offset, ignoring out-of-range bits.
    pub fn bit_slice_update(
        &mut self,
        arg: BValue,
        start: BValue,
        update: BValue,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::BitSliceUpdate {
                arg: self.node(arg)?,
                start: self.node(start)?,
                update_value: self.node(update)?,
            },
            None,
        )
    }

    /// Concatenates bits in most-significant-first order; an empty list is
    /// bits[0].
    pub fn concat(&mut self, args: &[BValue]) -> Result<BValue, BuilderError> {
        args.iter().try_fold(0usize, |width, arg| {
            width
                .checked_add(self.bits_width(*arg)?)
                .ok_or(BuilderError::WidthOverflow)
        })?;
        self.add_node(NodePayload::Nary(NaryOp::Concat, self.nodes(args)?), None)
    }

    /// Constructs a tuple, including the empty tuple.
    pub fn tuple(&mut self, elements: &[BValue]) -> Result<BValue, BuilderError> {
        elements.iter().try_fold(0usize, |width, element| {
            width
                .checked_add(checked_flat_width(self.get_type(*element)?)?)
                .ok_or(BuilderError::WidthOverflow)
        })?;
        self.add_node(NodePayload::Tuple(self.nodes(elements)?), None)
    }

    /// Extracts a statically selected tuple member.
    pub fn tuple_index(&mut self, tuple: BValue, index: usize) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::TupleIndex {
                tuple: self.node(tuple)?,
                index,
            },
            None,
        )
    }

    /// Multiplies equal-width operands, keeping that width (modular
    /// arithmetic).
    pub fn umul(&mut self, lhs: BValue, rhs: BValue) -> Result<BValue, BuilderError> {
        let width = self.same_bits_width(lhs, rhs)?;
        self.umul_with_width(lhs, rhs, width)
    }

    /// Multiplies equal-width signed operands, keeping that width.
    pub fn smul(&mut self, lhs: BValue, rhs: BValue) -> Result<BValue, BuilderError> {
        let width = self.same_bits_width(lhs, rhs)?;
        self.smul_with_width(lhs, rhs, width)
    }

    fn same_bits_width(&self, lhs: BValue, rhs: BValue) -> Result<usize, BuilderError> {
        let width = self.bits_width(lhs)?;
        if self.bits_width(rhs)? != width {
            return Err(BuilderError::InvalidOperation("default multiply requires equal operand widths; use the explicit-width method otherwise".to_string()));
        }
        Ok(width)
    }

    /// Multiplies unsigned operands of arbitrary widths to the requested width.
    pub fn umul_with_width(
        &mut self,
        lhs: BValue,
        rhs: BValue,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::Binop(Binop::Umul, self.node(lhs)?, self.node(rhs)?),
            Some(Type::Bits(width)),
        )
    }

    /// Multiplies signed operands of arbitrary widths to the requested width.
    pub fn smul_with_width(
        &mut self,
        lhs: BValue,
        rhs: BValue,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::Binop(Binop::Smul, self.node(lhs)?, self.node(rhs)?),
            Some(Type::Bits(width)),
        )
    }

    /// Constructs a homogeneous array, retaining the type of an empty array.
    pub fn array(
        &mut self,
        element_type: Type,
        elements: &[BValue],
    ) -> Result<BValue, BuilderError> {
        let ty = Type::new_array(element_type.clone(), elements.len());
        checked_flat_width(&ty)?;
        if elements.is_empty() {
            let value = IrValue::make_array_typed(element_type, &[])
                .map_err(|error| BuilderError::InvalidOperation(error.to_string()))?;
            return self.literal(value);
        }
        self.add_node(NodePayload::Array(self.nodes(elements)?), Some(ty))
    }

    /// Indexes an array; an out-of-range index selects its last element.
    pub fn array_index(&mut self, array: BValue, index: BValue) -> Result<BValue, BuilderError> {
        self.array_index_multi(array, &[index])
    }

    /// Indexes nested arrays, one dimension per index; zero indices is
    /// identity.
    pub fn array_index_multi(
        &mut self,
        array: BValue,
        indices: &[BValue],
    ) -> Result<BValue, BuilderError> {
        let mut ty = self.get_type(array)?;
        for _ in indices {
            match ty {
                Type::Array(array) if array.element_count != 0 => ty = &array.element_type,
                _ => {
                    return Err(BuilderError::InvalidOperation(
                        "array_index requires a nonempty array at every indexed dimension"
                            .to_string(),
                    ));
                }
            }
        }
        self.add_node(
            NodePayload::ArrayIndex {
                array: self.node(array)?,
                indices: self.nodes(indices)?,
                assumed_in_bounds: false,
            },
            None,
        )
    }

    /// Concatenates one or more arrays of the same element type.
    pub fn array_concat(&mut self, arrays: &[BValue]) -> Result<BValue, BuilderError> {
        arrays
            .iter()
            .try_fold(0usize, |count, array| match self.get_type(*array)? {
                Type::Array(array) => count
                    .checked_add(array.element_count)
                    .ok_or(BuilderError::WidthOverflow),
                _ => Err(BuilderError::InvalidOperation(
                    "array_concat requires array operands".to_string(),
                )),
            })?;
        self.add_node(NodePayload::ArrayConcat(self.nodes(arrays)?), None)
    }

    /// Slices an array, repeating its last element when the slice extends past
    /// it.
    pub fn array_slice(
        &mut self,
        array: BValue,
        start: BValue,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::ArraySlice {
                array: self.node(array)?,
                start: self.node(start)?,
                width,
            },
            None,
        )
    }

    /// Updates a nested array element; out-of-range indices leave the array
    /// unchanged.
    pub fn array_update(
        &mut self,
        array: BValue,
        update: BValue,
        indices: &[BValue],
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::ArrayUpdate {
                array: self.node(array)?,
                value: self.node(update)?,
                indices: self.nodes(indices)?,
                assumed_in_bounds: false,
            },
            None,
        )
    }

    /// Encodes a one-hot input into ceil(log2(input width)) bits.
    pub fn encode(&mut self, arg: BValue) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::Encode {
                arg: self.node(arg)?,
            },
            None,
        )
    }

    /// Decodes an index to one-hot bits, defaulting to 2^(input width) result
    /// bits.
    pub fn decode(&mut self, arg: BValue, width: Option<usize>) -> Result<BValue, BuilderError> {
        let input_width = self.bits_width(arg)?;
        let width = match width {
            Some(width) => width,
            None => u32::try_from(input_width)
                .ok()
                .and_then(|shift| 1usize.checked_shl(shift))
                .ok_or(BuilderError::WidthOverflow)?,
        };
        self.add_node(
            NodePayload::Decode {
                arg: self.node(arg)?,
                width,
            },
            None,
        )
    }

    /// Selects one case by index, requiring a default only for uncovered
    /// indices.
    pub fn select(
        &mut self,
        selector: BValue,
        cases: &[BValue],
        default: Option<BValue>,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::Sel {
                selector: self.node(selector)?,
                cases: self.nodes(cases)?,
                default: default.map(|value| self.node(value)).transpose()?,
            },
            None,
        )
    }

    /// Selects the lowest enabled case, using the default when no bit is
    /// enabled.
    pub fn priority_select(
        &mut self,
        selector: BValue,
        cases: &[BValue],
        default: BValue,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::PrioritySel {
                selector: self.node(selector)?,
                cases: self.nodes(cases)?,
                default: Some(self.node(default)?),
            },
            None,
        )
    }

    /// ORs the cases enabled by the corresponding selector bits.
    pub fn one_hot_select(
        &mut self,
        selector: BValue,
        cases: &[BValue],
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::OneHotSel {
                selector: self.node(selector)?,
                cases: self.nodes(cases)?,
            },
            None,
        )
    }

    /// Produces one-hot priority bits, with an extra high bit for an all-zero
    /// input.
    pub fn one_hot(&mut self, arg: BValue, lsb_is_priority: bool) -> Result<BValue, BuilderError> {
        self.bits_width(arg)?
            .checked_add(1)
            .ok_or(BuilderError::WidthOverflow)?;
        self.add_node(
            NodePayload::OneHot {
                arg: self.node(arg)?,
                lsb_prio: lsb_is_priority,
            },
            None,
        )
    }

    /// Counts leading zeros, returning the count in the input's bit width.
    pub fn clz(&mut self, arg: BValue) -> Result<BValue, BuilderError> {
        let width = self.bits_width(arg)?;
        width.checked_add(1).ok_or(BuilderError::WidthOverflow)?;
        if width == 0 {
            return self.identity(arg);
        }
        let reversed = self.rev(arg)?;
        self.ctz(reversed)
    }

    /// Counts trailing zeros, returning the count in the input's bit width.
    pub fn ctz(&mut self, arg: BValue) -> Result<BValue, BuilderError> {
        let width = self.bits_width(arg)?;
        width.checked_add(1).ok_or(BuilderError::WidthOverflow)?;
        if width == 0 {
            return self.identity(arg);
        }
        let one_hot = self.one_hot(arg, true)?;
        let encoded = self.encode(one_hot)?;
        self.zero_extend(encoded, width)
    }
}

impl Builder<FunctionState> {
    /// Creates an empty builder; the function name is checked when building.
    pub fn new(name: &str) -> Self {
        Self::from_state(name, FunctionState::default())
    }

    /// Adds a uniquely named parameter before any body nodes.
    pub fn param(&mut self, name: &str, ty: Type) -> Result<BValue, BuilderError> {
        if self.graph.nodes.len() != self.state.params.len() + 1 {
            return Err(BuilderError::ParameterAfterBody);
        }
        self.check_node_name(name, None)?;
        checked_flat_width(&ty)?;
        let node = NodeRef {
            index: self.graph.nodes.len(),
        };
        self.state.params.push(node);
        self.graph.nodes.push(Node {
            text_id: node.index,
            name: Some(name.to_string()),
            ty,
            payload: NodePayload::Param,
            pos: None,
        });
        self.names.insert(name.to_string(), node);
        Ok(BValue {
            builder_id: self.id,
            node,
        })
    }

    /// Moves the graph and ordered parameter interface into a function.
    fn into_function(self, return_value: BValue) -> Result<ir::Fn, BuilderError> {
        check_name(&self.graph.name)?;
        let node = self.node(return_value)?;
        let ret_ty = self.graph.get_node_ty(node).clone();
        Ok(ir::Fn {
            graph: self.graph,
            params: self.state.params,
            ret_ty,
            ret_node_ref: Some(node),
        })
    }

    /// Consumes and verifies a standalone function with no package
    /// dependencies.
    ///
    /// Functions containing `invoke` or `counted_for` require
    /// `build_in_package` or `build_into_package`, even if the call's
    /// result is unused.
    pub fn build(self, return_value: BValue) -> Result<ir::Fn, BuilderError> {
        let function = self.into_function(return_value)?;
        verify_function(&function)
            .map_err(|error| BuilderError::InvalidOperation(error.to_string()))?;
        Ok(function)
    }

    /// Builds a one-function package with that function marked as top.
    pub fn build_package(
        self,
        return_value: BValue,
        package_name: &str,
    ) -> Result<ir::Package, BuilderError> {
        check_name(package_name)?;
        let function = self.build(return_value)?;
        let top = Some((function.name.clone(), ir::MemberType::Function));
        Ok(ir::Package {
            name: package_name.to_string(),
            file_table: ir::FileTable::new(),
            members: vec![ir::PackageMember::Function(function)],
            top,
        })
    }

    /// Builds an insertion-ready function in the context of an existing
    /// package.
    ///
    /// Callees must already exist in a valid package and match the signatures
    /// supplied during construction. Rejects duplicate names and recursion;
    /// rebases IDs above all existing graph nodes. No package or function clone
    /// is needed, and the package is never modified.
    pub fn build_in_package(
        self,
        return_value: BValue,
        package: &ir::Package,
    ) -> Result<ir::Fn, BuilderError> {
        check_name(&package.name)?;
        if package
            .members
            .iter()
            .any(|member| member.graph().name == self.graph.name)
        {
            return Err(BuilderError::DuplicateName(self.graph.name.clone()));
        }
        verify_package(package)
            .map_err(|error| BuilderError::InvalidOperation(error.to_string()))?;
        self.check_callees(package)?;
        let mut function = self.into_function(return_value)?;
        verify_function_in_package(&function, package)
            .map_err(|error| BuilderError::InvalidOperation(error.to_string()))?;
        let base = package_max_emitted_node_id(package);
        rebase_fn_ids_in_place(&mut function, base)
            .map_err(|error| BuilderError::InvalidOperation(error.to_string()))?;
        Ok(function)
    }

    /// Appends a context-verified function with unique IDs, preserving the top.
    ///
    /// All checks occur before insertion. A failure leaves the package
    /// unchanged.
    pub fn build_into_package(
        self,
        return_value: BValue,
        package: &mut ir::Package,
    ) -> Result<(), BuilderError> {
        let function = self.build_in_package(return_value, package)?;
        package.members.push(ir::PackageMember::Function(function));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_eval::{FnEvalResult, eval_fn, eval_fn_in_package};
    use crate::ir_parser::Parser;

    fn bits(width: usize, value: u64) -> IrValue {
        IrValue::make_ubits(width, value).unwrap()
    }

    fn evaluate(function: &ir::Fn, args: &[IrValue]) -> IrValue {
        match eval_fn(function, args) {
            FnEvalResult::Success(result) => result.value,
            FnEvalResult::Failure(result) => panic!("unexpected evaluator failure: {result:?}"),
        }
    }

    #[test]
    fn parameter_return_and_last_value() {
        let mut builder = FnBuilder::new("identity");
        assert_eq!(builder.last_value(), None);
        let x = builder.param("x", Type::Bits(32)).unwrap();
        assert_eq!(builder.last_value(), Some(x));
        assert_eq!(builder.get_type(x).unwrap(), &Type::Bits(32));
        let package = builder.build_package(x, "sample").unwrap();
        assert_eq!(
            evaluate(package.get_top_fn().unwrap(), &[IrValue::u32(42)]),
            IrValue::u32(42)
        );
        Parser::new(&package.to_string())
            .parse_and_validate_package()
            .unwrap();
    }

    #[test]
    fn named_nodes_and_reused_handles_have_stable_text() {
        let mut builder = FnBuilder::new("f");
        let x = builder.param("x", Type::Bits(32)).unwrap();
        let y = builder.param("y", Type::Bits(32)).unwrap();
        let both = builder.and(x, y).unwrap();
        builder.set_name(both, "both").unwrap();
        let either = builder.or(x, y).unwrap();
        builder.set_name(either, "either").unwrap();
        let result = builder.tuple(&[both, either, both]).unwrap();
        builder.set_name(result, "result").unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(
            function.to_string(),
            r#"fn f(x: bits[32] id=1, y: bits[32] id=2) -> (bits[32], bits[32], bits[32]) {
  both: bits[32] = and(x, y, id=3)
  either: bits[32] = or(x, y, id=4)
  ret result: (bits[32], bits[32], bits[32]) = tuple(both, either, both, id=5)
}"#
        );
        assert_eq!(
            evaluate(&function, &[IrValue::u32(3), IrValue::u32(5)]),
            IrValue::make_tuple(&[IrValue::u32(1), IrValue::u32(7), IrValue::u32(1)])
        );
    }

    #[test]
    fn scalar_and_aggregate_literals() {
        let values = [
            bits(0, 0),
            bits(129, 7),
            IrValue::make_token(),
            IrValue::make_tuple(&[bits(2, 1), bits(4, 2)]),
            IrValue::make_array(&[bits(4, 1), bits(4, 2)]).unwrap(),
            IrValue::make_array_typed(Type::Bits(7), &[]).unwrap(),
        ];
        for value in values {
            let mut builder = FnBuilder::new("constant");
            let literal = builder.literal(value.clone()).unwrap();
            assert_eq!(builder.get_type(literal).unwrap(), &value.type_());
            let function = builder.build(literal).unwrap();
            assert_eq!(evaluate(&function, &[]), value);
        }
    }

    #[test]
    fn tuple_creation_and_indexing() {
        let mut builder = FnBuilder::new("tuple_index");
        let x = builder.param("x", Type::Bits(2)).unwrap();
        let y = builder.param("y", Type::Bits(4)).unwrap();
        let tuple = builder.tuple(&[x, y]).unwrap();
        let selected = builder.tuple_index(tuple, 1).unwrap();
        assert_eq!(builder.get_type(selected).unwrap(), &Type::Bits(4));
        assert!(builder.tuple_index(tuple, 2).is_err());
        assert!(builder.tuple_index(x, 0).is_err());
        let function = builder.build(selected).unwrap();
        assert_eq!(evaluate(&function, &[bits(2, 1), bits(4, 9)]), bits(4, 9));
    }

    #[test]
    fn empty_tuple_and_array_preserve_types() {
        let mut builder = FnBuilder::new("empty");
        let tuple = builder.tuple(&[]).unwrap();
        let array = builder.array(Type::Bits(17), &[]).unwrap();
        let result = builder.tuple(&[tuple, array]).unwrap();
        let package = builder.build_package(result, "sample").unwrap();
        let function = package.get_top_fn().unwrap();
        assert_eq!(
            evaluate(function, &[]),
            IrValue::make_tuple(&[
                IrValue::make_tuple(&[]),
                IrValue::make_array_typed(Type::Bits(17), &[]).unwrap(),
            ])
        );
        let parsed = Parser::new(&package.to_string())
            .parse_and_validate_package()
            .unwrap();
        assert_eq!(parsed.get_top_fn().unwrap().ret_ty, function.ret_ty);
    }

    #[test]
    fn arithmetic_and_comparisons() {
        let mut builder = FnBuilder::new("arithmetic");
        let a = builder.param("a", Type::Bits(4)).unwrap();
        let b = builder.param("b", Type::Bits(4)).unwrap();
        let values = [
            builder.add(a, b).unwrap(),
            builder.sub(a, b).unwrap(),
            builder.udiv(a, b).unwrap(),
            builder.sdiv(a, b).unwrap(),
            builder.umod(a, b).unwrap(),
            builder.smod(a, b).unwrap(),
            builder.eq(a, b).unwrap(),
            builder.ne(a, b).unwrap(),
            builder.ule(a, b).unwrap(),
            builder.ult(a, b).unwrap(),
            builder.uge(a, b).unwrap(),
            builder.ugt(a, b).unwrap(),
            builder.sle(a, b).unwrap(),
            builder.slt(a, b).unwrap(),
            builder.sge(a, b).unwrap(),
            builder.sgt(a, b).unwrap(),
        ];
        let result = builder.tuple(&values).unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(4, 13), bits(4, 2)]),
            IrValue::make_tuple(&[
                bits(4, 15),
                bits(4, 11),
                bits(4, 6),
                bits(4, 15),
                bits(4, 1),
                bits(4, 15),
                bits(1, 0),
                bits(1, 1),
                bits(1, 0),
                bits(1, 0),
                bits(1, 1),
                bits(1, 1),
                bits(1, 1),
                bits(1, 1),
                bits(1, 0),
                bits(1, 0),
            ])
        );
    }

    #[test]
    fn default_and_explicit_width_multiplication() {
        let mut builder = FnBuilder::new("multiply");
        let a = builder.param("a", Type::Bits(3)).unwrap();
        let b = builder.param("b", Type::Bits(3)).unwrap();
        let c = builder.param("c", Type::Bits(4)).unwrap();
        let values = [
            builder.umul(a, b).unwrap(),
            builder.smul(a, b).unwrap(),
            builder.umul_with_width(a, c, 7).unwrap(),
            builder.smul_with_width(a, c, 7).unwrap(),
        ];
        assert!(builder.umul(a, c).is_err());
        assert!(builder.smul(a, c).is_err());
        let result = builder.tuple(&values).unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(3, 6), bits(3, 2), bits(4, 3)]),
            IrValue::make_tuple(&[bits(3, 4), bits(3, 4), bits(7, 18), bits(7, 122)])
        );
    }

    #[test]
    fn bitwise_operations_and_reductions() {
        let mut builder = FnBuilder::new("bitops");
        let a = builder.param("a", Type::Bits(4)).unwrap();
        let b = builder.param("b", Type::Bits(4)).unwrap();
        let values = [
            builder.xor(a, b).unwrap(),
            builder.and(a, b).unwrap(),
            builder.nand(a, b).unwrap(),
            builder.or(a, b).unwrap(),
            builder.nor(a, b).unwrap(),
            builder.not(a).unwrap(),
            builder.neg(a).unwrap(),
            builder.rev(a).unwrap(),
            builder.and_reduce(a).unwrap(),
            builder.or_reduce(a).unwrap(),
            builder.xor_reduce(a).unwrap(),
        ];
        let result = builder.tuple(&values).unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(4, 3), bits(4, 5)]),
            IrValue::make_tuple(&[
                bits(4, 6),
                bits(4, 1),
                bits(4, 14),
                bits(4, 7),
                bits(4, 8),
                bits(4, 12),
                bits(4, 13),
                bits(4, 12),
                bits(1, 0),
                bits(1, 1),
                bits(1, 0)
            ])
        );
    }

    #[test]
    fn nand_truth_table() {
        let mut builder = FnBuilder::new("nand");
        let a = builder.param("a", Type::Bits(1)).unwrap();
        let b = builder.param("b", Type::Bits(1)).unwrap();
        let result = builder.nand(a, b).unwrap();
        let function = builder.build(result).unwrap();
        for (a, b, expected) in [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert_eq!(
                evaluate(&function, &[bits(1, a), bits(1, b)]),
                bits(1, expected)
            );
        }
    }

    #[test]
    fn concatenation_and_static_dynamic_slices() {
        let mut builder = FnBuilder::new("slices");
        let a = builder.param("a", Type::Bits(2)).unwrap();
        let b = builder.param("b", Type::Bits(2)).unwrap();
        let start = builder.param("start", Type::Bits(3)).unwrap();
        let concat = builder.concat(&[a, b]).unwrap();
        let slice = builder.bit_slice(concat, 1, 2).unwrap();
        let dynamic = builder.dynamic_bit_slice(concat, start, 2).unwrap();
        let updated = builder.bit_slice_update(concat, start, b).unwrap();
        let result = builder.tuple(&[concat, slice, dynamic, updated]).unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(2, 1), bits(2, 2), bits(3, 1)]),
            IrValue::make_tuple(&[bits(4, 6), bits(2, 3), bits(2, 3), bits(4, 4)])
        );
        assert_eq!(
            evaluate(&function, &[bits(2, 1), bits(2, 2), bits(3, 7)]),
            IrValue::make_tuple(&[bits(4, 6), bits(2, 3), bits(2, 0), bits(4, 6)])
        );
    }

    #[test]
    fn shifts_and_extensions() {
        let mut builder = FnBuilder::new("shift");
        let value = builder.param("value", Type::Bits(4)).unwrap();
        let amount = builder.param("amount", Type::Bits(2)).unwrap();
        let values = [
            builder.shra(value, amount).unwrap(),
            builder.shrl(value, amount).unwrap(),
            builder.shll(value, amount).unwrap(),
            builder.sign_extend(value, 8).unwrap(),
            builder.zero_extend(value, 8).unwrap(),
        ];
        let result = builder.tuple(&values).unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(4, 10), bits(2, 2)]),
            IrValue::make_tuple(&[
                bits(4, 14),
                bits(4, 2),
                bits(4, 8),
                bits(8, 250),
                bits(8, 10)
            ])
        );
    }

    #[test]
    fn leading_and_trailing_zeros_use_standard_nodes() {
        for width in [0, 1, 2, 4, 65, 129] {
            let mut builder = FnBuilder::new("count_zeros");
            let x = builder.param("x", Type::Bits(width)).unwrap();
            let clz = builder.clz(x).unwrap();
            let ctz = builder.ctz(x).unwrap();
            let result = builder.tuple(&[clz, ctz]).unwrap();
            let function = builder.build(result).unwrap();
            assert!(
                function
                    .nodes
                    .iter()
                    .all(|node| !node.payload.is_extension_op())
            );
            assert_eq!(
                evaluate(&function, &[bits(width, 0)]),
                IrValue::make_tuple(&[bits(width, width as u64), bits(width, width as u64)])
            );
            if width > 0 {
                assert_eq!(
                    evaluate(&function, &[bits(width, 1)]),
                    IrValue::make_tuple(&[bits(width, (width - 1) as u64), bits(width, 0)])
                );
            }
        }
    }

    #[test]
    fn encode_decode_and_one_hot() {
        let mut builder = FnBuilder::new("encoding");
        let x = builder.param("x", Type::Bits(4)).unwrap();
        let index = builder.param("index", Type::Bits(2)).unwrap();
        let values = [
            builder.encode(x).unwrap(),
            builder.decode(index, None).unwrap(),
            builder.decode(index, Some(2)).unwrap(),
            builder.one_hot(x, true).unwrap(),
            builder.one_hot(x, false).unwrap(),
        ];
        let result = builder.tuple(&values).unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(4, 10), bits(2, 3)]),
            IrValue::make_tuple(&[bits(2, 3), bits(4, 8), bits(2, 0), bits(5, 2), bits(5, 8)])
        );
        assert_eq!(
            evaluate(&function, &[bits(4, 0), bits(2, 1)]),
            IrValue::make_tuple(&[bits(2, 0), bits(4, 2), bits(2, 2), bits(5, 16), bits(5, 16)])
        );
    }

    #[test]
    fn array_construction_and_clamped_indexing() {
        let mut builder = FnBuilder::new("array_index");
        let x = builder.param("x", Type::Bits(4)).unwrap();
        let y = builder.param("y", Type::Bits(4)).unwrap();
        let index = builder.param("index", Type::Bits(3)).unwrap();
        let array = builder.array(Type::Bits(4), &[x, y]).unwrap();
        let selected = builder.array_index(array, index).unwrap();
        let function = builder.build(selected).unwrap();
        for (index, expected) in [(0, 1), (1, 2), (7, 2)] {
            assert_eq!(
                evaluate(&function, &[bits(4, 1), bits(4, 2), bits(3, index)]),
                bits(4, expected)
            );
        }
    }

    #[test]
    fn array_concat_slice_and_update() {
        let mut builder = FnBuilder::new("arrays");
        let a = builder
            .param("a", Type::new_array(Type::Bits(4), 2))
            .unwrap();
        let b = builder
            .param("b", Type::new_array(Type::Bits(4), 2))
            .unwrap();
        let start = builder.param("start", Type::Bits(3)).unwrap();
        let update = builder.param("update", Type::Bits(4)).unwrap();
        let concat = builder.array_concat(&[a, b]).unwrap();
        let slice = builder.array_slice(concat, start, 3).unwrap();
        let updated = builder.array_update(a, update, &[start]).unwrap();
        let result = builder.tuple(&[slice, updated]).unwrap();
        let function = builder.build(result).unwrap();
        let a = IrValue::make_array(&[bits(4, 0), bits(4, 1)]).unwrap();
        let b = IrValue::make_array(&[bits(4, 2), bits(4, 3)]).unwrap();
        assert_eq!(
            evaluate(&function, &[a.clone(), b.clone(), bits(3, 1), bits(4, 9)]),
            IrValue::make_tuple(&[
                IrValue::make_array(&[bits(4, 1), bits(4, 2), bits(4, 3)]).unwrap(),
                IrValue::make_array(&[bits(4, 0), bits(4, 9)]).unwrap(),
            ])
        );
        assert_eq!(
            evaluate(&function, &[a.clone(), b, bits(3, 3), bits(4, 9)]),
            IrValue::make_tuple(&[
                IrValue::make_array(&[bits(4, 3), bits(4, 3), bits(4, 3)]).unwrap(),
                a,
            ])
        );
    }

    #[test]
    fn multidimensional_arrays() {
        let mut builder = FnBuilder::new("nested");
        let array = builder
            .param(
                "array",
                Type::new_array(Type::new_array(Type::Bits(4), 2), 2),
            )
            .unwrap();
        let index = builder.param("index", Type::Bits(1)).unwrap();
        let value = builder.array_index_multi(array, &[index, index]).unwrap();
        let updated = builder.array_update(array, value, &[index, index]).unwrap();
        let same = builder.array_index_multi(updated, &[]).unwrap();
        let function = builder.build(same).unwrap();
        let inner = IrValue::make_array(&[bits(4, 1), bits(4, 2)]).unwrap();
        let array = IrValue::make_array(&[inner.clone(), inner]).unwrap();
        assert_eq!(evaluate(&function, &[array.clone(), bits(1, 1)]), array);
    }

    #[test]
    fn select_with_and_without_default() {
        let mut builder = FnBuilder::new("select");
        let selector = builder.param("selector", Type::Bits(2)).unwrap();
        let a = builder.literal(bits(4, 2)).unwrap();
        let b = builder.literal(bits(4, 3)).unwrap();
        let default = builder.literal(bits(4, 7)).unwrap();
        let partial = builder.select(selector, &[a, b], Some(default)).unwrap();
        let full = builder.select(selector, &[a, b, a, b], None).unwrap();
        let result = builder.tuple(&[partial, full]).unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(2, 1)]),
            IrValue::make_tuple(&[bits(4, 3), bits(4, 3)])
        );
        assert_eq!(
            evaluate(&function, &[bits(2, 3)]),
            IrValue::make_tuple(&[bits(4, 7), bits(4, 3)])
        );
    }

    #[test]
    fn priority_and_one_hot_selection() {
        let mut builder = FnBuilder::new("priority");
        let selector = builder.param("selector", Type::Bits(2)).unwrap();
        let a = builder.literal(bits(4, 2)).unwrap();
        let b = builder.literal(bits(4, 5)).unwrap();
        let default = builder.literal(bits(4, 8)).unwrap();
        let priority = builder.priority_select(selector, &[a, b], default).unwrap();
        let one_hot = builder.one_hot_select(selector, &[a, b]).unwrap();
        let result = builder.tuple(&[priority, one_hot]).unwrap();
        let function = builder.build(result).unwrap();
        for (selector, priority, one_hot) in [(0, 8, 0), (1, 2, 2), (2, 5, 5), (3, 2, 7)] {
            assert_eq!(
                evaluate(&function, &[bits(2, selector)]),
                IrValue::make_tuple(&[bits(4, priority), bits(4, one_hot)])
            );
        }
    }

    #[test]
    fn rejects_foreign_handles_everywhere() {
        let mut first = FnBuilder::new("first");
        let mut second = FnBuilder::new("second");
        let x = first.param("x", Type::Bits(4)).unwrap();
        let y = second.param("y", Type::Bits(4)).unwrap();
        assert_eq!(first.get_type(y), Err(BuilderError::ForeignValue));
        assert_eq!(first.add(x, y), Err(BuilderError::ForeignValue));
        assert_eq!(first.tuple(&[x, y]), Err(BuilderError::ForeignValue));
        assert_eq!(
            first.set_name(y, "foreign"),
            Err(BuilderError::ForeignValue)
        );
        assert!(matches!(first.build(y), Err(BuilderError::ForeignValue)));
    }

    #[test]
    fn rejected_operations_do_not_poison_builder_or_consume_ids() {
        let mut builder = FnBuilder::new("recover");
        let x = builder.param("x", Type::Bits(4)).unwrap();
        let y = builder.param("y", Type::Bits(5)).unwrap();
        assert!(builder.add(x, y).is_err());
        assert_eq!(builder.last_value(), Some(y));
        let sum = builder.add(x, x).unwrap();
        assert_eq!(sum.node.index, 3);
        let function = builder.build(sum).unwrap();
        assert_eq!(evaluate(&function, &[bits(4, 3), bits(5, 0)]), bits(4, 6));
    }

    #[test]
    fn parameter_order_names_and_renaming() {
        let mut builder = FnBuilder::new("names");
        assert!(matches!(
            builder.param("not a name", Type::Bits(1)),
            Err(BuilderError::InvalidName(_))
        ));
        let a = builder.param("a", Type::Bits(1)).unwrap();
        assert_eq!(
            builder.param("a", Type::Bits(1)),
            Err(BuilderError::DuplicateName("a".to_string()))
        );
        let b = builder.param("b", Type::Bits(1)).unwrap();
        assert_eq!(
            builder.set_name(a, "b"),
            Err(BuilderError::DuplicateName("b".to_string()))
        );
        builder.set_name(a, "renamed").unwrap();
        builder.set_name(a, "renamed").unwrap();
        let result = builder.and(a, b).unwrap();
        assert_eq!(
            builder.param("late", Type::Bits(1)),
            Err(BuilderError::ParameterAfterBody)
        );
        builder.set_name(result, "a").unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(function.get_param(0).param_name(), "renamed");
        assert_eq!(function.nodes[1].name.as_deref(), Some("renamed"));
    }

    #[test]
    fn invalid_function_and_package_names() {
        let mut builder = FnBuilder::new("bad name");
        let value = builder.tuple(&[]).unwrap();
        assert!(matches!(
            builder.build(value),
            Err(BuilderError::InvalidName(_))
        ));
        let mut builder = FnBuilder::new("valid");
        let value = builder.tuple(&[]).unwrap();
        assert!(matches!(
            builder.build_package(value, "bad name"),
            Err(BuilderError::InvalidName(_))
        ));
        for name in ["ret", "bits", "token", "fn", "true", "false"] {
            let mut builder = FnBuilder::new(name);
            let value = builder.tuple(&[]).unwrap();
            assert!(matches!(
                builder.build(value),
                Err(BuilderError::InvalidName(_))
            ));
        }
    }

    #[test]
    fn identifier_validation_is_shared_with_name_importers() {
        for valid in ["x", "_", "_42", "add", "add__1", "literal"] {
            assert!(is_valid_identifier(valid));
        }
        for invalid in [
            "", "42", "a.b", "a[0]", "a b", "café", "fn", "token", "false",
        ] {
            assert!(!is_valid_identifier(invalid));
        }
    }

    #[test]
    fn appending_functions_rebases_ids_and_preserves_top() {
        let mut first = FnBuilder::new("first");
        let x = first.param("x", Type::Bits(4)).unwrap();
        let doubled = first.add(x, x).unwrap();
        let mut package = first.build_package(doubled, "sample").unwrap();
        let mut second = FnBuilder::new("second");
        let y = second.param("y", Type::Bits(4)).unwrap();
        let inverted = second.not(y).unwrap();
        second.build_into_package(inverted, &mut package).unwrap();
        assert_eq!(package.get_top_fn().unwrap().name, "first");
        let second = package.get_fn("second").unwrap();
        assert_eq!(second.get_param(0).text_id, 3);
        assert_eq!(second.nodes[1].text_id, 3);
        assert_eq!(second.nodes[2].text_id, 4);
        assert_eq!(evaluate(second, &[bits(4, 3)]), bits(4, 12));
        crate::ir_verify::verify_package(&package).unwrap();
        Parser::new(&package.to_string())
            .parse_and_validate_package()
            .unwrap();
    }

    #[test]
    fn failed_package_insertion_leaves_package_unchanged() {
        let mut first = FnBuilder::new("same");
        let x = first.param("x", Type::Bits(4)).unwrap();
        let mut package = first.build_package(x, "sample").unwrap();
        let before = package.to_string();
        let mut duplicate = FnBuilder::new("same");
        let value = duplicate.tuple(&[]).unwrap();
        assert_eq!(
            duplicate.build_into_package(value, &mut package),
            Err(BuilderError::DuplicateName("same".to_string()))
        );
        assert_eq!(package.to_string(), before);

        package.get_fn_mut("same").unwrap().nodes[1].text_id = usize::MAX;
        let before = package.to_string();
        let mut overflow = FnBuilder::new("another");
        let value = overflow.tuple(&[]).unwrap();
        assert!(overflow.build_into_package(value, &mut package).is_err());
        assert_eq!(package.to_string(), before);
    }

    #[test]
    fn rejects_bad_slices_extensions_and_aggregate_types() {
        let mut builder = FnBuilder::new("invalid");
        let x = builder.param("x", Type::Bits(4)).unwrap();
        let y = builder.param("y", Type::Bits(3)).unwrap();
        let zero = builder.param("zero", Type::Bits(0)).unwrap();
        let tuple = builder.tuple(&[x]).unwrap();
        assert!(builder.bit_slice(x, 3, 2).is_err());
        assert!(builder.bit_slice(x, usize::MAX, 1).is_err());
        assert!(builder.dynamic_bit_slice(x, y, 5).is_err());
        assert!(builder.dynamic_bit_slice(x, tuple, 2).is_err());
        assert!(builder.sign_extend(x, 3).is_err());
        assert!(builder.sign_extend(zero, 1).is_err());
        assert!(builder.zero_extend(x, 3).is_err());
        assert!(builder.array(Type::Bits(4), &[x, y]).is_err());
        assert!(builder.array(Type::Bits(3), &[x]).is_err());
        assert!(builder.not(tuple).is_err());
        assert!(builder.add(tuple, tuple).is_err());
        assert!(builder.decode(y, Some(9)).is_err());
        let empty = builder.array(Type::Bits(4), &[]).unwrap();
        assert!(builder.array_index(empty, x).is_err());
        assert!(builder.array_slice(empty, x, 1).is_err());
        let array = builder.array(Type::Bits(4), &[x]).unwrap();
        assert!(builder.array_update(array, y, &[x]).is_err());
        assert!(builder.array_slice(array, x, 0).is_err());
        builder.build(x).unwrap();
    }

    #[test]
    fn rejects_invalid_selection_contracts() {
        let mut builder = FnBuilder::new("invalid_select");
        let one = builder.param("one", Type::Bits(1)).unwrap();
        let two = builder.param("two", Type::Bits(2)).unwrap();
        let a = builder.param("a", Type::Bits(4)).unwrap();
        let wrong = builder.param("wrong", Type::Bits(3)).unwrap();
        assert!(builder.select(two, &[a], None).is_err());
        assert!(builder.select(one, &[a, a], Some(a)).is_err());
        assert!(builder.select(two, &[a], Some(wrong)).is_err());
        assert!(builder.select(one, &[a, a, a], Some(a)).is_err());
        assert!(builder.priority_select(two, &[a], a).is_err());
        assert!(builder.priority_select(one, &[a], wrong).is_err());
        assert!(builder.one_hot_select(two, &[a]).is_err());
        assert!(builder.one_hot_select(two, &[a, wrong]).is_err());
        assert!(builder.one_hot_select(one, &[]).is_err());
        builder.build(a).unwrap();
    }

    #[test]
    fn zero_width_operations() {
        let mut builder = FnBuilder::new("zero_width");
        let x = builder.param("x", Type::Bits(0)).unwrap();
        let empty = builder.concat(&[]).unwrap();
        let slice = builder.bit_slice(x, 0, 0).unwrap();
        let extended = builder.zero_extend(x, 3).unwrap();
        let one_hot = builder.one_hot(x, true).unwrap();
        let result = builder.tuple(&[empty, slice, extended, one_hot]).unwrap();
        let function = builder.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(0, 0)]),
            IrValue::make_tuple(&[bits(0, 0), bits(0, 0), bits(3, 0), bits(1, 1)])
        );
    }

    #[test]
    fn checked_dimension_arithmetic() {
        let mut builder = FnBuilder::new("overflow");
        let huge = builder.param("huge", Type::Bits(usize::MAX)).unwrap();
        let one = builder.param("one", Type::Bits(1)).unwrap();
        assert_eq!(
            builder.concat(&[huge, one]),
            Err(BuilderError::WidthOverflow)
        );
        assert_eq!(
            builder.tuple(&[huge, one]),
            Err(BuilderError::WidthOverflow)
        );
        assert_eq!(
            builder.one_hot(huge, true),
            Err(BuilderError::WidthOverflow)
        );
        assert_eq!(builder.clz(huge), Err(BuilderError::WidthOverflow));
        assert_eq!(builder.ctz(huge), Err(BuilderError::WidthOverflow));
        assert_eq!(builder.decode(huge, None), Err(BuilderError::WidthOverflow));
        assert_eq!(
            builder.array(Type::Bits(usize::MAX), &[huge, huge]),
            Err(BuilderError::WidthOverflow)
        );
        assert_eq!(builder.last_value(), Some(one));
        builder.build(one).unwrap();
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn decode_checks_large_operand_width_without_truncation() {
        let mut builder = FnBuilder::new("wide_decode");
        let arg = builder.param("arg", Type::Bits(1usize << 32)).unwrap();
        let decoded = builder.decode(arg, Some(2)).unwrap();
        assert_eq!(builder.get_type(decoded).unwrap(), &Type::Bits(2));
        assert_eq!(builder.decode(arg, None), Err(BuilderError::WidthOverflow));
        assert_eq!(builder.last_value(), Some(decoded));
        // This is a type-only test: no enormous bitvector needs allocating.
        builder.build(decoded).unwrap();
    }

    /// Makes a typed callee without coupling a test to manual node
    /// construction.
    fn identity_callee(name: &str, width: usize) -> ir::Fn {
        let mut builder = FnBuilder::new(name);
        let value = builder.param("value", Type::Bits(width)).unwrap();
        builder.build(value).unwrap()
    }

    fn package_for(function: ir::Fn) -> ir::Package {
        ir::Package {
            name: "sample".to_string(),
            members: vec![ir::PackageMember::Function(function)],
            file_table: ir::FileTable::new(),
            top: None,
        }
    }

    #[test]
    fn invoke_checks_arguments_and_resolves_actual_package_callee() {
        let callee = identity_callee("identity", 8);
        let mut package = package_for(callee.clone());
        let mut builder = FnBuilder::new("caller");
        let arg = builder.param("arg", Type::Bits(8)).unwrap();
        let wrong = builder.param("wrong", Type::Bits(4)).unwrap();
        assert!(builder.invoke(&callee, &[]).is_err());
        assert!(builder.invoke(&callee, &[wrong]).is_err());
        assert_eq!(builder.last_value(), Some(wrong));
        let call = builder.invoke(&callee, &[arg]).unwrap();
        builder.build_into_package(call, &mut package).unwrap();
        let caller = package.get_fn("caller").unwrap();
        match eval_fn_in_package(&package, caller, &[bits(8, 42), bits(4, 0)]) {
            FnEvalResult::Success(result) => assert_eq!(result.value, bits(8, 42)),
            FnEvalResult::Failure(result) => panic!("unexpected failure: {result:?}"),
        }
        verify_package(&package).unwrap();
    }

    #[test]
    fn invoke_requires_package_context_and_matching_definition() {
        let callee = identity_callee("identity", 8);
        let mut standalone = FnBuilder::new("caller");
        let arg = standalone.param("arg", Type::Bits(8)).unwrap();
        let call = standalone.invoke(&callee, &[arg]).unwrap();
        assert!(standalone.build(call).is_err());

        for actual in [
            identity_callee("unrelated", 8),
            identity_callee("identity", 4),
        ] {
            let mut package = package_for(actual);
            let before = package.to_string();
            let mut builder = FnBuilder::new("caller");
            let arg = builder.param("arg", Type::Bits(8)).unwrap();
            let call = builder.invoke(&callee, &[arg]).unwrap();
            assert!(builder.build_into_package(call, &mut package).is_err());
            assert_eq!(package.to_string(), before);
        }
    }

    #[test]
    fn rejects_recursive_or_conflicting_callee_declarations() {
        let callee = identity_callee("callee", 8);
        let conflicting = identity_callee("callee", 4);
        let recursive = identity_callee("caller", 8);
        let mut builder = FnBuilder::new("caller");
        let arg = builder.param("arg", Type::Bits(8)).unwrap();
        let small = builder.param("small", Type::Bits(4)).unwrap();
        assert!(builder.invoke(&recursive, &[arg]).is_err());
        let call = builder.invoke(&callee, &[arg]).unwrap();
        assert!(builder.invoke(&conflicting, &[small]).is_err());
        assert_eq!(builder.last_value(), Some(call));
        builder
            .build_in_package(call, &package_for(callee))
            .unwrap();
    }

    fn loop_body(index_width: usize) -> ir::Fn {
        let mut builder = FnBuilder::new("loop_body");
        builder.param("index", Type::Bits(index_width)).unwrap();
        let carry = builder.param("carry", Type::Bits(8)).unwrap();
        let step = builder.param("step", Type::Bits(8)).unwrap();
        let result = builder.add(carry, step).unwrap();
        builder.build(result).unwrap()
    }

    #[test]
    fn counted_loops_check_induction_range_carry_and_invariants() {
        let body = loop_body(3);
        for (trip_count, expected) in [(0, 1), (1, 3), (3, 7)] {
            let mut package = package_for(body.clone());
            let mut builder = FnBuilder::new("caller");
            let init = builder.literal(bits(8, 1)).unwrap();
            let step = builder.literal(bits(8, 2)).unwrap();
            let result = builder
                .counted_for(init, trip_count, 2, &body, &[step])
                .unwrap();
            builder.build_into_package(result, &mut package).unwrap();
            match eval_fn_in_package(&package, package.get_fn("caller").unwrap(), &[]) {
                FnEvalResult::Success(result) => assert_eq!(result.value, bits(8, expected)),
                FnEvalResult::Failure(result) => panic!("unexpected failure: {result:?}"),
            }
        }
        let mut builder = FnBuilder::new("invalid_loop");
        let init = builder.literal(bits(8, 1)).unwrap();
        let small = builder.literal(bits(4, 1)).unwrap();
        assert!(builder.counted_for(init, 4, 3, &body, &[init]).is_err());
        assert!(
            builder
                .counted_for(init, 3, usize::MAX, &body, &[init])
                .is_err()
        );
        assert!(builder.counted_for(init, 3, 1, &body, &[]).is_err());
        assert!(builder.counted_for(init, 3, 1, &body, &[small]).is_err());
        assert!(builder.counted_for(small, 3, 1, &body, &[init]).is_err());
        assert!(
            builder
                .counted_for(init, 0, 1, &loop_body(0), &[init])
                .is_err()
        );
        assert_eq!(builder.last_value(), Some(small));
        builder.build(init).unwrap();
    }
}
