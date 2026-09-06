// SPDX-License-Identifier: Apache-2.0
//
//! Desugars PIR extension ops (e.g. `ext_carry_out`) into upstream-compatible
//! PIR / XLS IR basis operations.
//!
//! ## Design invariant
//! This module is a *semantic projection* from the PIR extension-op set onto
//! the canonical XLS IR opcode basis. It must be deterministic and
//! semantics-preserving; it is **not** where QoR strategies belong.
//!
//! Gate-level QoR strategies belong in `xlsynth-g8r` (gatify), where different
//! circuits can be chosen for the same semantics.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{
    Binop, ExtNaryAddArchitecture, ExtNaryAddTerm, Fn, Node, NodeGraph, NodePayload, NodeRef,
    Package, PackageMember, Type, Unop,
};
use crate::ir_builder::{
    BuilderError, FnBuilder, NaryAddOptions, NaryAddTerm, NormalizeLeftOptions,
};
use crate::ir_rebase_ids::{package_max_emitted_node_id, rebase_fn_ids_in_place};
use crate::ir_utils::{compact_and_toposort_in_place, remap_payload_with};
use crate::math::ceil_log2;
use crate::{IrBits, IrValue};

#[derive(Debug, Clone)]
pub struct DesugarError {
    msg: String,
}

impl DesugarError {
    fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl std::fmt::Display for DesugarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DesugarError: {}", self.msg)
    }
}

impl std::error::Error for DesugarError {}

impl From<BuilderError> for DesugarError {
    fn from(error: BuilderError) -> Self {
        Self::new(format!("constructing extension helper: {error}"))
    }
}

/// Controls how PIR extension ops are emitted as text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionEmitMode {
    /// Emit PIR extension ops directly in the textual IR.
    /// This form is not compatible with XLS.
    /// For example:
    ///   x: bits[3] = ext_carry_out(lhs, rhs, c_in);
    AsExtensionOp,
    /// Replace extension ops with the equivalent inline XLS IR operations.
    Desugared,
    /// Replace extension ops with invokes of synthetic FFI helper functions.
    /// This form can be handled by XLS and the invoke and helper functions are
    /// robust against transformations in the optimization pipeline.
    /// For example:
    ///   x: bits[3] = invoke(__pir_ext__ext_carry_out__w8, lhs, rhs, c_in);
    AsFfiFunction,
}

fn next_text_id(f: &NodeGraph) -> usize {
    f.nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn expect_bits_width(f: &NodeGraph, r: NodeRef, ctx: &str) -> Result<usize, DesugarError> {
    let ty = f.get_node_ty(r);
    match ty {
        Type::Bits(w) => Ok(*w),
        _ => Err(DesugarError::new(format!(
            "{}: expected bits operand, got {}",
            ctx, ty
        ))),
    }
}

#[derive(Debug, Clone, Copy)]
struct ExtCarryOutShape {
    width: usize,
}

/// Validates `ext_carry_out` operands and returns the shared shape info used by
/// both inline lowering and FFI wrapper synthesis.
fn analyze_ext_carry_out(
    f: &NodeGraph,
    lhs: NodeRef,
    rhs: NodeRef,
    c_in: NodeRef,
) -> Result<ExtCarryOutShape, DesugarError> {
    let width = expect_bits_width(f, lhs, "ext_carry_out.lhs")?;
    let rhs_width = expect_bits_width(f, rhs, "ext_carry_out.rhs")?;
    if width != rhs_width {
        return Err(DesugarError::new(format!(
            "ext_carry_out: lhs width {} != rhs width {}",
            width, rhs_width
        )));
    }
    let c_in_width = expect_bits_width(f, c_in, "ext_carry_out.c_in")?;
    if c_in_width != 1 {
        return Err(DesugarError::new(format!(
            "ext_carry_out: c_in must be bits[1], got bits[{}]",
            c_in_width
        )));
    }
    Ok(ExtCarryOutShape { width })
}

/// Appends the basis-op implementation of `ext_carry_out` and returns the
/// lowered carry-out bit node.
fn append_lowered_ext_carry_out(
    f: &mut NodeGraph,
    lhs: NodeRef,
    rhs: NodeRef,
    c_in: NodeRef,
    shape: ExtCarryOutShape,
) -> NodeRef {
    let w1 = shape.width.saturating_add(1);
    let lhs_ext = push_node(
        f,
        Type::Bits(w1),
        NodePayload::ZeroExt {
            arg: lhs,
            new_bit_count: w1,
        },
    );
    let rhs_ext = push_node(
        f,
        Type::Bits(w1),
        NodePayload::ZeroExt {
            arg: rhs,
            new_bit_count: w1,
        },
    );
    let sum_w1 = push_node(
        f,
        Type::Bits(w1),
        NodePayload::Binop(Binop::Add, lhs_ext, rhs_ext),
    );
    let c_in_ext = push_node(
        f,
        Type::Bits(w1),
        NodePayload::ZeroExt {
            arg: c_in,
            new_bit_count: w1,
        },
    );
    let sum_w1_ci = push_node(
        f,
        Type::Bits(w1),
        NodePayload::Binop(Binop::Add, sum_w1, c_in_ext),
    );
    push_node(
        f,
        Type::Bits(1),
        NodePayload::BitSlice {
            arg: sum_w1_ci,
            start: shape.width,
            width: 1,
        },
    )
}

#[derive(Debug, Clone, Copy)]
struct ExtPrioEncodeShape {
    input_width: usize,
    output_width: usize,
    lsb_prio: bool,
}

impl ExtPrioEncodeShape {
    fn new(input_width: usize, lsb_prio: bool) -> Self {
        Self {
            input_width,
            output_width: ceil_log2(input_width.saturating_add(1)),
            lsb_prio,
        }
    }
}

#[derive(Clone, Copy)]
struct ExtClzShape {
    input_width: usize,
    output_width: usize,
    offset: usize,
}

impl ExtClzShape {
    fn new(input_width: usize, output_width: usize, offset: usize) -> Self {
        Self {
            input_width,
            output_width,
            offset,
        }
    }
}

#[derive(Clone, Copy)]
struct ExtNormalizeLeftShape {
    input_width: usize,
    normalized_bit_count: usize,
    shift_offset: usize,
    clz_bit_count: Option<usize>,
}

impl ExtNormalizeLeftShape {
    fn internal_shift_bit_count(self) -> usize {
        ceil_log2(
            self.input_width
                .saturating_add(self.shift_offset)
                .saturating_add(1),
        )
    }
}

/// Validates `ext_prio_encode` operands and returns the shared shape info used
/// by both inline lowering and FFI wrapper synthesis.
fn analyze_ext_prio_encode(
    f: &NodeGraph,
    arg: NodeRef,
    lsb_prio: bool,
) -> Result<ExtPrioEncodeShape, DesugarError> {
    let input_width = expect_bits_width(f, arg, "ext_prio_encode.arg")?;
    Ok(ExtPrioEncodeShape::new(input_width, lsb_prio))
}

/// Validates `ext_clz` operands and returns the shared shape info used by both
/// inline lowering and FFI wrapper synthesis.
fn analyze_ext_clz(
    f: &NodeGraph,
    nr: NodeRef,
    arg: NodeRef,
    offset: usize,
    new_bit_count: usize,
) -> Result<ExtClzShape, DesugarError> {
    let input_width = expect_bits_width(f, arg, "ext_clz.arg")?;
    match &f.get_node(nr).ty {
        Type::Bits(width) if *width == new_bit_count => {
            Ok(ExtClzShape::new(input_width, new_bit_count, offset))
        }
        ty => Err(DesugarError::new(format!(
            "ext_clz result type must be bits[{new_bit_count}], got {ty}"
        ))),
    }
}

/// Validates `ext_normalize_left` operands and returns the shared shape info
/// used by both inline lowering and FFI wrapper synthesis.
fn analyze_ext_normalize_left(
    f: &NodeGraph,
    nr: NodeRef,
    arg: NodeRef,
    shift_offset: usize,
    normalized_bit_count: usize,
    clz_bit_count: Option<usize>,
) -> Result<ExtNormalizeLeftShape, DesugarError> {
    let input_width = expect_bits_width(f, arg, "ext_normalize_left.arg")?;
    if normalized_bit_count < input_width {
        return Err(DesugarError::new(format!(
            "ext_normalize_left normalized width {normalized_bit_count} must be at least input width {input_width}"
        )));
    }
    let expected_ty =
        crate::ir::ext_normalize_left_result_type(normalized_bit_count, clz_bit_count);
    if f.get_node(nr).ty != expected_ty {
        return Err(DesugarError::new(format!(
            "ext_normalize_left result type must be {expected_ty}, got {}",
            f.get_node(nr).ty
        )));
    }
    Ok(ExtNormalizeLeftShape {
        input_width,
        normalized_bit_count,
        shift_offset,
        clz_bit_count,
    })
}

#[derive(Clone, Copy)]
struct ExtMaskLowShape {
    output_width: usize,
    count_width: usize,
}

/// Validates `ext_mask_low` shape and returns the widths needed by both inline
/// lowering and FFI wrapper synthesis.
fn analyze_ext_mask_low(
    f: &NodeGraph,
    result: NodeRef,
    count: NodeRef,
) -> Result<ExtMaskLowShape, DesugarError> {
    let output_width = match f.get_node(result).ty {
        Type::Bits(width) => width,
        ref ty => {
            return Err(DesugarError::new(format!(
                "ext_mask_low: result type must be bits, got {}",
                ty
            )));
        }
    };
    let count_width = expect_bits_width(f, count, "ext_mask_low.count")?;
    Ok(ExtMaskLowShape {
        output_width,
        count_width,
    })
}

/// Appends the basis-op implementation of `ext_prio_encode` and returns the
/// lowered encoded-result node.
fn append_lowered_ext_prio_encode(
    f: &mut NodeGraph,
    arg: NodeRef,
    shape: ExtPrioEncodeShape,
) -> NodeRef {
    let one_hot_width = shape.input_width.saturating_add(1);
    let one_hot = push_node(
        f,
        Type::Bits(one_hot_width),
        NodePayload::OneHot {
            arg,
            lsb_prio: shape.lsb_prio,
        },
    );
    push_node(
        f,
        Type::Bits(shape.output_width),
        NodePayload::Encode { arg: one_hot },
    )
}

/// Appends the basis-op implementation of `ext_clz` and returns the lowered
/// encoded-result node.
fn append_lowered_ext_clz(
    f: &mut NodeGraph,
    arg: NodeRef,
    shape: ExtClzShape,
) -> Result<NodeRef, DesugarError> {
    let reversed = push_node(
        f,
        Type::Bits(shape.input_width),
        NodePayload::Unop(Unop::Reverse, arg),
    );
    let one_hot_width = shape.input_width.saturating_add(1);
    let one_hot = push_node(
        f,
        Type::Bits(one_hot_width),
        NodePayload::OneHot {
            arg: reversed,
            lsb_prio: true,
        },
    );
    let encoded = push_node(
        f,
        Type::Bits(ceil_log2(shape.input_width.saturating_add(1))),
        NodePayload::Encode { arg: one_hot },
    );
    let resized = extend_or_truncate_to_width(
        f,
        encoded,
        shape.output_width,
        /* signed= */ false,
        "ext_clz.encoded",
    )?;
    if shape.offset == 0 {
        return Ok(resized);
    }
    let offset = make_usize_bits_literal(f, shape.output_width, shape.offset);
    Ok(push_node(
        f,
        Type::Bits(shape.output_width),
        NodePayload::Binop(Binop::Add, resized, offset),
    ))
}

/// Appends the basis-op implementation of `ext_normalize_left` and returns the
/// lowered normalized value or `(normalized, raw_clz)` tuple.
fn append_lowered_ext_normalize_left(
    f: &mut NodeGraph,
    arg: NodeRef,
    shape: ExtNormalizeLeftShape,
) -> Result<NodeRef, DesugarError> {
    let normalized_input = extend_or_truncate_to_width(
        f,
        arg,
        shape.normalized_bit_count,
        /* signed= */ false,
        "ext_normalize_left.arg",
    )?;
    let shift_amount = append_lowered_ext_clz(
        f,
        arg,
        ExtClzShape::new(
            shape.input_width,
            shape.internal_shift_bit_count(),
            shape.shift_offset,
        ),
    )?;
    let normalized = push_node(
        f,
        Type::Bits(shape.normalized_bit_count),
        NodePayload::Binop(Binop::Shll, normalized_input, shift_amount),
    );
    let Some(clz_bit_count) = shape.clz_bit_count else {
        return Ok(normalized);
    };
    let raw_clz = append_lowered_ext_clz(
        f,
        arg,
        ExtClzShape::new(shape.input_width, clz_bit_count, /* offset= */ 0),
    )?;
    Ok(push_node(
        f,
        crate::ir::ext_normalize_left_result_type(shape.normalized_bit_count, Some(clz_bit_count)),
        NodePayload::Tuple(vec![normalized, raw_clz]),
    ))
}

fn push_node(f: &mut NodeGraph, ty: Type, payload: NodePayload) -> NodeRef {
    let text_id = next_text_id(f);
    let new_index = f.nodes.len();
    f.nodes.push(Node {
        text_id,
        name: None,
        ty,
        payload,
        pos: None,
    });
    NodeRef { index: new_index }
}

fn make_zero_bits_literal(f: &mut NodeGraph, width: usize) -> NodeRef {
    push_node(
        f,
        Type::Bits(width),
        NodePayload::Literal(IrValue::make_ubits(width, 0).expect("zero bits literal")),
    )
}

fn make_ubits_literal(f: &mut NodeGraph, width: usize, value: u64) -> NodeRef {
    push_node(
        f,
        Type::Bits(width),
        NodePayload::Literal(IrValue::make_ubits(width, value).expect("bits literal")),
    )
}

fn make_usize_bits_literal(f: &mut NodeGraph, width: usize, value: usize) -> NodeRef {
    let mut bits = vec![false; width];
    for (i, bit) in bits.iter_mut().enumerate() {
        if i < usize::BITS as usize {
            *bit = ((value >> i) & 1) == 1;
        }
    }
    push_node(
        f,
        Type::Bits(width),
        NodePayload::Literal(IrValue::from_bits(&IrBits::from_lsb_is_0(&bits))),
    )
}

/// Appends the basis-op implementation of `ext_mask_low` and returns the
/// lowered mask node.
fn append_lowered_ext_mask_low(
    f: &mut NodeGraph,
    count: NodeRef,
    shape: ExtMaskLowShape,
) -> NodeRef {
    if shape.output_width == 0 {
        return make_zero_bits_literal(f, 0);
    }
    let one = make_ubits_literal(f, shape.output_width, 1);
    let shifted = push_node(
        f,
        Type::Bits(shape.output_width),
        NodePayload::Binop(Binop::Shll, one, count),
    );
    push_node(
        f,
        Type::Bits(shape.output_width),
        NodePayload::Binop(Binop::Sub, shifted, one),
    )
}

/// Sign- or zero-extends, or truncates, a bits-typed value to `output_width`.
fn extend_or_truncate_to_width(
    f: &mut NodeGraph,
    operand: NodeRef,
    output_width: usize,
    signed: bool,
    ctx: &str,
) -> Result<NodeRef, DesugarError> {
    let operand_width = expect_bits_width(f, operand, ctx)?;
    if operand_width == output_width {
        Ok(operand)
    } else if signed && operand_width == 0 {
        Ok(make_zero_bits_literal(f, output_width))
    } else if operand_width < output_width {
        Ok(push_node(
            f,
            Type::Bits(output_width),
            if signed {
                NodePayload::SignExt {
                    arg: operand,
                    new_bit_count: output_width,
                }
            } else {
                NodePayload::ZeroExt {
                    arg: operand,
                    new_bit_count: output_width,
                }
            },
        ))
    } else {
        Ok(push_node(
            f,
            Type::Bits(output_width),
            NodePayload::BitSlice {
                arg: operand,
                start: 0,
                width: output_width,
            },
        ))
    }
}

fn desugar_ext_carry_out_in_graph(f: &mut NodeGraph) -> Result<bool, DesugarError> {
    let mut changed = false;

    // Snapshot length so we only visit original nodes; desugaring appends
    // nodes.
    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        let NodePayload::ExtCarryOut { lhs, rhs, c_in } = payload else {
            continue;
        };
        changed = true;

        let shape = analyze_ext_carry_out(f, lhs, rhs, c_in)?;
        let lowered_carry_out = append_lowered_ext_carry_out(f, lhs, rhs, c_in, shape);

        // Overwrite the ext node in-place; compaction/toposort will place deps
        // before this node.
        let node = f.get_node_mut(nr);
        node.ty = Type::Bits(1);
        node.payload = NodePayload::Unop(Unop::Identity, lowered_carry_out);
    }

    Ok(changed)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExtNaryAddShape {
    output_width: usize,
    operand_widths: Vec<usize>,
    operand_signed: Vec<bool>,
    operand_negated: Vec<bool>,
    arch: Option<ExtNaryAddArchitecture>,
}

/// Validates `ext_nary_add` shape and returns the widths needed by both inline
/// lowering and FFI wrapper synthesis.
fn analyze_ext_nary_add(
    f: &NodeGraph,
    result: NodeRef,
    terms: &[ExtNaryAddTerm],
    arch: Option<ExtNaryAddArchitecture>,
) -> Result<ExtNaryAddShape, DesugarError> {
    let output_width = match f.get_node(result).ty {
        Type::Bits(width) => width,
        ref ty => {
            return Err(DesugarError::new(format!(
                "ext_nary_add: result type must be bits, got {}",
                ty
            )));
        }
    };
    let mut operand_widths = Vec::with_capacity(terms.len());
    let mut operand_signed = Vec::with_capacity(terms.len());
    let mut operand_negated = Vec::with_capacity(terms.len());
    for (i, term) in terms.iter().enumerate() {
        operand_widths.push(expect_bits_width(
            f,
            term.operand,
            &format!("ext_nary_add.operand[{i}]"),
        )?);
        operand_signed.push(term.signed);
        operand_negated.push(term.negated);
    }
    Ok(ExtNaryAddShape {
        output_width,
        operand_widths,
        operand_signed,
        operand_negated,
        arch,
    })
}

/// Appends the basis-op implementation of `ext_nary_add` and returns the
/// lowered sum node.
fn append_lowered_ext_nary_add(
    f: &mut NodeGraph,
    terms: &[ExtNaryAddTerm],
    output_width: usize,
) -> Result<NodeRef, DesugarError> {
    if output_width == 0 || terms.is_empty() {
        return Ok(make_zero_bits_literal(f, output_width));
    }

    let mut lowered_terms = Vec::with_capacity(terms.len());
    for term in terms {
        let resized = extend_or_truncate_to_width(
            f,
            term.operand,
            output_width,
            term.signed,
            "ext_nary_add.operand",
        )?;
        let lowered = if term.negated {
            push_node(
                f,
                Type::Bits(output_width),
                NodePayload::Unop(Unop::Neg, resized),
            )
        } else {
            resized
        };
        lowered_terms.push(lowered);
    }

    let mut acc = lowered_terms[0];
    for operand in lowered_terms.into_iter().skip(1) {
        acc = push_node(
            f,
            Type::Bits(output_width),
            NodePayload::Binop(Binop::Add, acc, operand),
        );
    }
    Ok(acc)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum FfiWrapKey {
    ExtCarryOut {
        width: usize,
    },
    ExtClz {
        input_width: usize,
        output_width: usize,
        offset: usize,
    },
    ExtNormalizeLeft {
        input_width: usize,
        normalized_bit_count: usize,
        shift_offset: usize,
        clz_bit_count: Option<usize>,
    },
    ExtMaskLow {
        output_width: usize,
        count_width: usize,
    },
    ExtNaryAdd {
        output_width: usize,
        operand_widths: Vec<usize>,
        operand_signed: Vec<bool>,
        operand_negated: Vec<bool>,
        arch: Option<ExtNaryAddArchitecture>,
    },
    ExtPrioEncode {
        input_width: usize,
        lsb_prio: bool,
    },
}

fn helper_base_name(key: &FfiWrapKey) -> String {
    match key {
        FfiWrapKey::ExtCarryOut { width } => {
            format!("__pir_ext__ext_carry_out__w{width}")
        }
        FfiWrapKey::ExtClz {
            input_width,
            output_width,
            offset,
        } => {
            format!("__pir_ext__ext_clz__inw{input_width}__outw{output_width}__off{offset}")
        }
        FfiWrapKey::ExtNormalizeLeft {
            input_width,
            normalized_bit_count,
            shift_offset,
            clz_bit_count,
        } => {
            let mut name = format!(
                "__pir_ext__ext_normalize_left__inw{input_width}__normw{normalized_bit_count}__off{shift_offset}"
            );
            if let Some(clz_bit_count) = clz_bit_count {
                name.push_str(&format!("__clzw{clz_bit_count}"));
            }
            name
        }
        FfiWrapKey::ExtMaskLow {
            output_width,
            count_width,
        } => {
            format!("__pir_ext__ext_mask_low__outw{output_width}__countw{count_width}")
        }
        FfiWrapKey::ExtNaryAdd {
            output_width,
            operand_widths,
            operand_signed,
            operand_negated,
            arch,
        } => {
            let operand_widths_text = operand_widths
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("_");
            let mut base = format!(
                "__pir_ext__ext_nary_add__outw{}__ops{}",
                output_width, operand_widths_text
            );
            if operand_signed.iter().any(|bit| *bit) {
                let signed_text = operand_signed
                    .iter()
                    .map(|bit| if *bit { "1" } else { "0" })
                    .collect::<Vec<_>>()
                    .join("_");
                base.push_str(&format!("__sgn{signed_text}"));
            }
            if operand_negated.iter().any(|bit| *bit) {
                let negated_text = operand_negated
                    .iter()
                    .map(|bit| if *bit { "1" } else { "0" })
                    .collect::<Vec<_>>()
                    .join("_");
                base.push_str(&format!("__neg{negated_text}"));
            }
            match arch {
                Some(arch) => format!("{base}__arch{arch}"),
                None => base,
            }
        }
        FfiWrapKey::ExtPrioEncode {
            input_width,
            lsb_prio,
        } => {
            format!(
                "__pir_ext__ext_prio_encode__w{}__lsb{}",
                input_width,
                if *lsb_prio { 1 } else { 0 }
            )
        }
    }
}

fn make_unique_helper_name(base: &str, existing_names: &mut BTreeSet<String>) -> String {
    if existing_names.insert(base.to_string()) {
        return base.to_string();
    }
    let mut suffix = 1usize;
    loop {
        let candidate = format!("{base}__{suffix}");
        if existing_names.insert(candidate.clone()) {
            return candidate;
        }
        suffix = suffix.saturating_add(1);
    }
}

fn format_ffi_proto_outer_attr(code_template: &str) -> String {
    format!(
        "#[ffi_proto(\"\"\"code_template: {:?}\n\"\"\")]",
        code_template
    )
}

fn format_bool_list_csv(values: &[bool]) -> String {
    values
        .iter()
        .map(|bit| if *bit { "true" } else { "false" })
        .collect::<Vec<_>>()
        .join(",")
}

fn helper_code_template(key: &FfiWrapKey) -> String {
    match key {
        FfiWrapKey::ExtCarryOut { width } => format!(
            "pir_ext_carry_out {{fn}} (.lhs({{lhs}}), .rhs({{rhs}}), .c_in({{c_in}}), .out({{return}})); /* xlsynth_pir_ext=ext_carry_out;width={width} */"
        ),
        FfiWrapKey::ExtClz {
            input_width,
            output_width,
            offset,
        } => {
            let metadata = format!(
                "xlsynth_pir_ext=ext_clz;width={input_width};out_width={output_width};offset={offset}"
            );
            format!("pir_ext_clz {{fn}} (.arg({{arg}}), .out({{return}})); /* {metadata} */")
        }
        FfiWrapKey::ExtNormalizeLeft {
            input_width,
            normalized_bit_count,
            shift_offset,
            clz_bit_count,
        } => {
            let mut metadata = format!(
                "xlsynth_pir_ext=ext_normalize_left;width={input_width};normalized_width={normalized_bit_count};shift_offset={shift_offset}"
            );
            if let Some(clz_bit_count) = clz_bit_count {
                metadata.push_str(&format!(";clz_width={clz_bit_count}"));
            }
            format!(
                "pir_ext_normalize_left {{fn}} (.arg({{arg}}), .out({{return}})); /* {metadata} */"
            )
        }
        FfiWrapKey::ExtMaskLow {
            output_width,
            count_width,
        } => {
            let metadata = format!(
                "xlsynth_pir_ext=ext_mask_low;out_width={output_width};count_width={count_width}"
            );
            format!(
                "pir_ext_mask_low {{fn}} (.count({{count}}), .out({{return}})); /* {metadata} */"
            )
        }
        FfiWrapKey::ExtNaryAdd {
            output_width,
            operand_widths,
            operand_signed,
            operand_negated,
            arch,
        } => {
            let operand_bindings = operand_widths
                .iter()
                .enumerate()
                .map(|(i, _)| format!(".op{i}({{op{i}}})"))
                .collect::<Vec<_>>()
                .join(", ");
            let port_list = if operand_bindings.is_empty() {
                ".out({return})".to_string()
            } else {
                format!("{operand_bindings}, .out({{return}})")
            };
            let mut metadata_fields = vec![
                "xlsynth_pir_ext=ext_nary_add".to_string(),
                format!("out_width={output_width}"),
                format!("operand_signed={}", format_bool_list_csv(operand_signed)),
                format!("operand_negated={}", format_bool_list_csv(operand_negated)),
            ];
            if let Some(arch) = arch {
                metadata_fields.push(format!("arch={arch}"));
            }
            format!(
                "pir_ext_nary_add {{fn}} ({port_list}); /* {} */",
                metadata_fields.join(";")
            )
        }
        FfiWrapKey::ExtPrioEncode {
            input_width,
            lsb_prio,
        } => {
            let metadata = format!(
                "xlsynth_pir_ext=ext_prio_encode;width={input_width};lsb_prio={}",
                if *lsb_prio { "true" } else { "false" }
            );
            format!(
                "pir_ext_prio_encode {{fn}} (.arg({{arg}}), .out({{return}})); /* {metadata} */"
            )
        }
    }
}

/// Constructs a checked extension helper and reuses the inline lowering path.
fn make_helper_fn(name: String, key: &FfiWrapKey) -> Result<Fn, DesugarError> {
    let mut builder = FnBuilder::new(&name);
    let result = match key {
        FfiWrapKey::ExtCarryOut { width } => {
            let lhs = builder.param("lhs", Type::Bits(*width))?;
            let rhs = builder.param("rhs", Type::Bits(*width))?;
            let carry_in = builder.param("c_in", Type::Bits(1))?;
            builder.ext_carry_out(lhs, rhs, carry_in)?
        }
        FfiWrapKey::ExtNaryAdd {
            output_width,
            operand_widths,
            operand_signed,
            operand_negated,
            arch,
        } => {
            let operands = operand_widths
                .iter()
                .enumerate()
                .map(|(index, width)| builder.param(&format!("op{index}"), Type::Bits(*width)))
                .collect::<Result<Vec<_>, _>>()?;
            let terms = operands
                .into_iter()
                .zip(operand_signed)
                .zip(operand_negated)
                .map(|((operand, signed), negated)| NaryAddTerm {
                    operand,
                    signed: *signed,
                    negated: *negated,
                })
                .collect::<Vec<_>>();
            builder.ext_nary_add(
                &terms,
                NaryAddOptions {
                    bit_count: *output_width,
                    architecture: *arch,
                },
            )?
        }
        FfiWrapKey::ExtClz {
            input_width,
            output_width,
            offset,
        } => {
            let arg = builder.param("arg", Type::Bits(*input_width))?;
            builder.ext_clz(arg, *offset, *output_width)?
        }
        FfiWrapKey::ExtNormalizeLeft {
            input_width,
            normalized_bit_count,
            shift_offset,
            clz_bit_count,
        } => {
            let arg = builder.param("arg", Type::Bits(*input_width))?;
            builder.ext_normalize_left(
                arg,
                NormalizeLeftOptions {
                    normalized_bit_count: *normalized_bit_count,
                    shift_offset: *shift_offset,
                    clz_bit_count: *clz_bit_count,
                },
            )?
        }
        FfiWrapKey::ExtMaskLow {
            output_width,
            count_width,
        } => {
            let count = builder.param("count", Type::Bits(*count_width))?;
            builder.ext_mask_low(count, *output_width)?
        }
        FfiWrapKey::ExtPrioEncode {
            input_width,
            lsb_prio,
        } => {
            let arg = builder.param("arg", Type::Bits(*input_width))?;
            builder.ext_prio_encode(arg, *lsb_prio)?
        }
    };
    let mut helper = builder.build(result)?;
    desugar_extensions_in_fn(&mut helper)?;
    canonicalize_helper_node_order(&mut helper);
    helper
        .outer_attrs
        .push(format_ffi_proto_outer_attr(&helper_code_template(key)));
    Ok(helper)
}

/// Preserves canonical FFI helper IDs and allocation order after lowering.
fn canonicalize_helper_node_order(helper: &mut Fn) {
    // The builder allocated one extension node after the parameters. Lowering
    // replaces it with the return identity and appends its dependencies. Give
    // those dependencies consecutive IDs first, followed by the identity, to
    // retain the established FFI text representation.
    let mut order = (0..helper.nodes.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&index| {
        (
            helper.ret_node_ref == Some(NodeRef { index }),
            helper.nodes[index].text_id,
        )
    });
    let mut mapping = vec![NodeRef { index: 0 }; helper.nodes.len()];
    for (index, old_index) in order.into_iter().enumerate() {
        helper.nodes[old_index].text_id = index;
        mapping[old_index] = NodeRef { index };
    }
    // Compaction may have interleaved independent dependencies. Restore their
    // allocation order and remap references without changing annotations.
    helper.nodes.sort_by_key(|node| node.text_id);
    for node in &mut helper.nodes {
        node.payload = remap_payload_with(&node.payload, |(_, operand)| mapping[operand.index]);
    }
    for parameter in &mut helper.params {
        *parameter = mapping[parameter.index];
    }
    helper.ret_node_ref = helper.ret_node_ref.map(|node| mapping[node.index]);
}

fn max_text_id_in_graph(f: &NodeGraph) -> usize {
    f.nodes.iter().map(|node| node.text_id).max().unwrap_or(0)
}

fn get_or_create_helper_name(
    key: &FfiWrapKey,
    helper_names: &mut BTreeMap<FfiWrapKey, String>,
    helper_fns: &mut Vec<Fn>,
    existing_names: &mut BTreeSet<String>,
    current_max_text_id: &mut usize,
) -> Result<String, DesugarError> {
    if let Some(existing) = helper_names.get(key) {
        return Ok(existing.clone());
    }
    let name = make_unique_helper_name(&helper_base_name(key), existing_names);
    let mut helper = make_helper_fn(name.clone(), key)?;
    rebase_fn_ids_in_place(&mut helper, *current_max_text_id)
        .map_err(|error| DesugarError::new(error.to_string()))?;
    *current_max_text_id = max_text_id_in_graph(&helper);
    helper_names.insert(key.clone(), name.clone());
    helper_fns.push(helper);
    Ok(name)
}

fn wrap_extensions_in_graph(
    f: &mut NodeGraph,
    helper_names: &mut BTreeMap<FfiWrapKey, String>,
    helper_fns: &mut Vec<Fn>,
    existing_names: &mut BTreeSet<String>,
    current_max_text_id: &mut usize,
) -> Result<bool, DesugarError> {
    let mut changed = false;
    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        match payload {
            NodePayload::ExtCarryOut { lhs, rhs, c_in } => {
                let shape = analyze_ext_carry_out(f, lhs, rhs, c_in)?;
                let key = FfiWrapKey::ExtCarryOut { width: shape.width };
                let helper_name = get_or_create_helper_name(
                    &key,
                    helper_names,
                    helper_fns,
                    existing_names,
                    current_max_text_id,
                )?;
                let node = f.get_node_mut(nr);
                node.ty = Type::Bits(1);
                node.payload = NodePayload::Invoke {
                    to_apply: helper_name,
                    operands: vec![lhs, rhs, c_in],
                };
                changed = true;
            }
            NodePayload::ExtClz {
                arg,
                offset,
                new_bit_count,
            } => {
                let shape = analyze_ext_clz(f, nr, arg, offset, new_bit_count)?;
                let key = FfiWrapKey::ExtClz {
                    input_width: shape.input_width,
                    output_width: shape.output_width,
                    offset: shape.offset,
                };
                let helper_name = get_or_create_helper_name(
                    &key,
                    helper_names,
                    helper_fns,
                    existing_names,
                    current_max_text_id,
                )?;
                let node = f.get_node_mut(nr);
                node.ty = Type::Bits(shape.output_width);
                node.payload = NodePayload::Invoke {
                    to_apply: helper_name,
                    operands: vec![arg],
                };
                changed = true;
            }
            NodePayload::ExtNormalizeLeft {
                arg,
                shift_offset,
                normalized_bit_count,
                clz_bit_count,
            } => {
                let shape = analyze_ext_normalize_left(
                    f,
                    nr,
                    arg,
                    shift_offset,
                    normalized_bit_count,
                    clz_bit_count,
                )?;
                let key = FfiWrapKey::ExtNormalizeLeft {
                    input_width: shape.input_width,
                    normalized_bit_count: shape.normalized_bit_count,
                    shift_offset: shape.shift_offset,
                    clz_bit_count: shape.clz_bit_count,
                };
                let helper_name = get_or_create_helper_name(
                    &key,
                    helper_names,
                    helper_fns,
                    existing_names,
                    current_max_text_id,
                )?;
                let node = f.get_node_mut(nr);
                node.ty = crate::ir::ext_normalize_left_result_type(
                    shape.normalized_bit_count,
                    shape.clz_bit_count,
                );
                node.payload = NodePayload::Invoke {
                    to_apply: helper_name,
                    operands: vec![arg],
                };
                changed = true;
            }
            NodePayload::ExtMaskLow { count } => {
                let shape = analyze_ext_mask_low(f, nr, count)?;
                let key = FfiWrapKey::ExtMaskLow {
                    output_width: shape.output_width,
                    count_width: shape.count_width,
                };
                let helper_name = get_or_create_helper_name(
                    &key,
                    helper_names,
                    helper_fns,
                    existing_names,
                    current_max_text_id,
                )?;
                let node = f.get_node_mut(nr);
                node.ty = Type::Bits(shape.output_width);
                node.payload = NodePayload::Invoke {
                    to_apply: helper_name,
                    operands: vec![count],
                };
                changed = true;
            }
            NodePayload::ExtNaryAdd { terms, arch } => {
                let shape = analyze_ext_nary_add(f, nr, &terms, arch)?;
                let key = FfiWrapKey::ExtNaryAdd {
                    output_width: shape.output_width,
                    operand_widths: shape.operand_widths.clone(),
                    operand_signed: shape.operand_signed.clone(),
                    operand_negated: shape.operand_negated.clone(),
                    arch: shape.arch,
                };
                let helper_name = get_or_create_helper_name(
                    &key,
                    helper_names,
                    helper_fns,
                    existing_names,
                    current_max_text_id,
                )?;
                let node = f.get_node_mut(nr);
                node.ty = Type::Bits(shape.output_width);
                node.payload = NodePayload::Invoke {
                    to_apply: helper_name,
                    operands: terms.iter().map(|term| term.operand).collect(),
                };
                changed = true;
            }
            NodePayload::ExtPrioEncode { arg, lsb_prio } => {
                let shape = analyze_ext_prio_encode(f, arg, lsb_prio)?;
                let key = FfiWrapKey::ExtPrioEncode {
                    input_width: shape.input_width,
                    lsb_prio: shape.lsb_prio,
                };
                let helper_name = get_or_create_helper_name(
                    &key,
                    helper_names,
                    helper_fns,
                    existing_names,
                    current_max_text_id,
                )?;
                let node = f.get_node_mut(nr);
                node.ty = Type::Bits(shape.output_width);
                node.payload = NodePayload::Invoke {
                    to_apply: helper_name,
                    operands: vec![arg],
                };
                changed = true;
            }
            _ => {}
        }
    }
    Ok(changed)
}

fn wrap_extensions_in_package(pkg: &mut Package) -> Result<(), DesugarError> {
    let mut existing_names: BTreeSet<String> = pkg
        .members
        .iter()
        .map(|member| member.graph().name.clone())
        .collect();
    let mut helper_names: BTreeMap<FfiWrapKey, String> = BTreeMap::new();
    let mut helper_fns: Vec<Fn> = Vec::new();
    let mut current_max_text_id = package_max_emitted_node_id(pkg);

    for member in pkg.members.iter_mut() {
        let _changed = wrap_extensions_in_graph(
            member.graph_mut(),
            &mut helper_names,
            &mut helper_fns,
            &mut existing_names,
            &mut current_max_text_id,
        )?;
    }

    if !helper_fns.is_empty() {
        let mut new_members: Vec<PackageMember> = helper_fns
            .into_iter()
            .map(PackageMember::Function)
            .collect();
        new_members.extend(std::mem::take(&mut pkg.members));
        pkg.members = new_members;
    }
    Ok(())
}

fn desugar_ext_nary_add_in_graph(f: &mut NodeGraph) -> Result<bool, DesugarError> {
    let mut changed = false;

    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        let NodePayload::ExtNaryAdd { terms, arch } = payload else {
            continue;
        };
        changed = true;

        let shape = analyze_ext_nary_add(f, nr, &terms, arch)?;
        let lowered = append_lowered_ext_nary_add(f, &terms, shape.output_width)?;

        let node = f.get_node_mut(nr);
        node.ty = Type::Bits(shape.output_width);
        node.payload = NodePayload::Unop(Unop::Identity, lowered);
    }

    Ok(changed)
}

fn desugar_ext_prio_encode_in_graph(f: &mut NodeGraph) -> Result<bool, DesugarError> {
    let mut changed = false;

    // Snapshot length so we only visit original nodes; desugaring appends
    // nodes.
    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        let NodePayload::ExtPrioEncode { arg, lsb_prio } = payload else {
            continue;
        };
        changed = true;

        let shape = analyze_ext_prio_encode(f, arg, lsb_prio)?;
        let encoded = append_lowered_ext_prio_encode(f, arg, shape);

        // Overwrite the ext node in-place; compaction/toposort will place deps
        // before this node.
        let node = f.get_node_mut(nr);
        node.ty = Type::Bits(shape.output_width);
        node.payload = NodePayload::Unop(Unop::Identity, encoded);
    }

    Ok(changed)
}

fn desugar_ext_clz_in_graph(f: &mut NodeGraph) -> Result<bool, DesugarError> {
    let mut changed = false;

    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        let NodePayload::ExtClz {
            arg,
            offset,
            new_bit_count,
        } = payload
        else {
            continue;
        };
        changed = true;

        let shape = analyze_ext_clz(f, nr, arg, offset, new_bit_count)?;
        let encoded = append_lowered_ext_clz(f, arg, shape)?;

        let node = f.get_node_mut(nr);
        node.ty = Type::Bits(shape.output_width);
        node.payload = NodePayload::Unop(Unop::Identity, encoded);
    }

    Ok(changed)
}

fn desugar_ext_normalize_left_in_graph(f: &mut NodeGraph) -> Result<bool, DesugarError> {
    let mut changed = false;

    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        let NodePayload::ExtNormalizeLeft {
            arg,
            shift_offset,
            normalized_bit_count,
            clz_bit_count,
        } = payload
        else {
            continue;
        };
        changed = true;

        let shape = analyze_ext_normalize_left(
            f,
            nr,
            arg,
            shift_offset,
            normalized_bit_count,
            clz_bit_count,
        )?;
        let lowered = append_lowered_ext_normalize_left(f, arg, shape)?;

        let node = f.get_node_mut(nr);
        node.ty = crate::ir::ext_normalize_left_result_type(
            shape.normalized_bit_count,
            shape.clz_bit_count,
        );
        node.payload = NodePayload::Unop(Unop::Identity, lowered);
    }

    Ok(changed)
}

fn desugar_ext_mask_low_in_graph(f: &mut NodeGraph) -> Result<bool, DesugarError> {
    let mut changed = false;

    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        let NodePayload::ExtMaskLow { count } = payload else {
            continue;
        };
        changed = true;

        let shape = analyze_ext_mask_low(f, nr, count)?;
        let lowered = append_lowered_ext_mask_low(f, count, shape);

        let node = f.get_node_mut(nr);
        node.ty = Type::Bits(shape.output_width);
        node.payload = NodePayload::Unop(Unop::Identity, lowered);
    }

    Ok(changed)
}

/// Desugars extension ops within `f` into upstream-compatible PIR operations.
///
/// This function also normalizes the node list into a valid topological order.
pub fn desugar_extensions_in_fn(f: &mut Fn) -> Result<(), DesugarError> {
    desugar_extensions_in_graph(&mut f.graph)?;
    compact_and_toposort_in_place(f).map_err(DesugarError::new)?;
    Ok(())
}

/// Lowers extension nodes without imposing function interface or root rules.
fn desugar_extensions_in_graph(f: &mut NodeGraph) -> Result<(), DesugarError> {
    let _changed = desugar_ext_carry_out_in_graph(f)?
        | desugar_ext_clz_in_graph(f)?
        | desugar_ext_normalize_left_in_graph(f)?
        | desugar_ext_mask_low_in_graph(f)?
        | desugar_ext_nary_add_in_graph(f)?
        | desugar_ext_prio_encode_in_graph(f)?;
    Ok(())
}

/// Desugars extension ops within `pkg` into upstream-compatible PIR operations.
pub fn desugar_extensions_in_package(pkg: &mut Package) -> Result<(), DesugarError> {
    for member in pkg.members.iter_mut() {
        match member {
            PackageMember::Function(f) => desugar_extensions_in_fn(f)?,
            PackageMember::Block(block) => {
                desugar_extensions_in_graph(&mut block.graph)?;
                block.compact_and_toposort().map_err(DesugarError::new)?;
            }
        }
    }
    Ok(())
}

/// Emits a package as text using the requested extension-op projection mode.
pub fn emit_package_with_extension_mode(
    pkg: &Package,
    mode: ExtensionEmitMode,
) -> Result<String, DesugarError> {
    match mode {
        ExtensionEmitMode::AsExtensionOp => Ok(pkg.to_string()),
        ExtensionEmitMode::Desugared => {
            let mut desugared = pkg.clone();
            desugar_extensions_in_package(&mut desugared)?;
            Ok(desugared.to_string())
        }
        ExtensionEmitMode::AsFfiFunction => {
            let mut wrapped = pkg.clone();
            wrap_extensions_in_package(&mut wrapped)?;
            Ok(wrapped.to_string())
        }
    }
}

/// Emits upstream-compatible XLS IR text for `pkg` by desugaring extensions
/// first.
pub fn emit_package_as_xls_ir_text(pkg: &Package) -> Result<String, DesugarError> {
    emit_package_with_extension_mode(pkg, ExtensionEmitMode::Desugared)
}

#[cfg(test)]
mod tests {
    use super::{
        FfiWrapKey, canonicalize_helper_node_order, get_or_create_helper_name, helper_base_name,
        make_helper_fn,
    };
    use crate::ir::ExtNaryAddArchitecture;
    use crate::ir_eval::{FnEvalResult, eval_fn};
    use crate::ir_verify::verify_function;
    use crate::{IrBits, IrValue};
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn ffi_helpers_preserve_canonical_text() {
        let cases = [
            FfiWrapKey::ExtCarryOut { width: 8 },
            FfiWrapKey::ExtNaryAdd {
                output_width: 6,
                operand_widths: vec![3, 5, 9],
                operand_signed: vec![false, true, false],
                operand_negated: vec![false, true, false],
                arch: Some(ExtNaryAddArchitecture::BrentKung),
            },
            FfiWrapKey::ExtClz {
                input_width: 4,
                output_width: 3,
                offset: 1,
            },
            FfiWrapKey::ExtNormalizeLeft {
                input_width: 4,
                normalized_bit_count: 8,
                shift_offset: 1,
                clz_bit_count: Some(3),
            },
            FfiWrapKey::ExtMaskLow {
                output_width: 8,
                count_width: 4,
            },
            FfiWrapKey::ExtPrioEncode {
                input_width: 5,
                lsb_prio: true,
            },
        ];
        let text = cases
            .iter()
            .map(|key| {
                let mut helper = make_helper_fn(helper_base_name(key), key).unwrap();
                verify_function(&helper).unwrap();
                let text = helper.to_string();
                for node in &mut helper.nodes {
                    node.text_id *= 3;
                }
                canonicalize_helper_node_order(&mut helper);
                assert_eq!(helper.to_string(), text);
                text
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            format!("{text}\n"),
            include_str!("../tests/goldens/extension_helpers.ir")
        );
    }

    #[test]
    fn ffi_helpers_support_zero_single_bit_and_wide_operands() {
        for width in [0, 1, 129] {
            let zero = IrValue::from_bits(&IrBits::zero(width));
            let cases = [
                (FfiWrapKey::ExtCarryOut { width }, IrValue::bool(false)),
                (
                    FfiWrapKey::ExtNaryAdd {
                        output_width: width,
                        operand_widths: vec![width, width + 1],
                        operand_signed: vec![true, false],
                        operand_negated: vec![false, true],
                        arch: None,
                    },
                    zero.clone(),
                ),
                (
                    FfiWrapKey::ExtNaryAdd {
                        output_width: width,
                        operand_widths: Vec::new(),
                        operand_signed: Vec::new(),
                        operand_negated: Vec::new(),
                        arch: None,
                    },
                    zero.clone(),
                ),
                (
                    FfiWrapKey::ExtClz {
                        input_width: width,
                        output_width: 8,
                        offset: 2,
                    },
                    IrValue::make_ubits(8, (width + 2) as u64).unwrap(),
                ),
                (
                    FfiWrapKey::ExtNormalizeLeft {
                        input_width: width,
                        normalized_bit_count: width + 2,
                        shift_offset: 1,
                        clz_bit_count: Some(8),
                    },
                    IrValue::make_tuple(&[
                        IrValue::from_bits(&IrBits::zero(width + 2)),
                        IrValue::make_ubits(8, width as u64).unwrap(),
                    ]),
                ),
                (
                    FfiWrapKey::ExtMaskLow {
                        output_width: width,
                        count_width: 8,
                    },
                    zero,
                ),
                (
                    FfiWrapKey::ExtPrioEncode {
                        input_width: width,
                        lsb_prio: false,
                    },
                    IrValue::make_ubits(crate::math::ceil_log2(width + 1), width as u64).unwrap(),
                ),
            ];
            for (key, expected) in cases {
                let helper = make_helper_fn(helper_base_name(&key), &key).unwrap();
                verify_function(&helper).unwrap();
                for (index, node) in helper.nodes.iter().enumerate() {
                    assert_eq!(node.text_id, index, "{key:?}");
                    assert!(!node.payload.is_extension_op());
                }
                let args = helper
                    .param_nodes()
                    .map(|node| IrValue::from_bits(&IrBits::zero(node.ty.bit_count())))
                    .collect::<Vec<_>>();
                let FnEvalResult::Success(result) = eval_fn(&helper, &args) else {
                    panic!("helper evaluation failed: {key:?}");
                };
                assert_eq!(result.value, expected, "{key:?}");
            }
        }
    }

    #[test]
    fn ffi_helper_construction_propagates_invalid_name_and_width_errors() {
        let error = make_helper_fn(
            "bad name".to_string(),
            &FfiWrapKey::ExtCarryOut { width: 1 },
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "DesugarError: constructing extension helper: invalid IR identifier: \"bad name\""
        );
        let error = make_helper_fn(
            "overflow".to_string(),
            &FfiWrapKey::ExtCarryOut { width: usize::MAX },
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "DesugarError: constructing extension helper: IR width or element count overflows usize"
        );
    }

    #[test]
    fn ffi_helper_package_id_overflow_is_reported() {
        let mut max_text_id = usize::MAX;
        let error = get_or_create_helper_name(
            &FfiWrapKey::ExtCarryOut { width: 1 },
            &mut BTreeMap::new(),
            &mut Vec::new(),
            &mut BTreeSet::new(),
            &mut max_text_id,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "DesugarError: node ID allocation or rebasing overflows usize"
        );
    }
}
