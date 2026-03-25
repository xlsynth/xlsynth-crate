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
    Binop, ExtNaryAddArchitecture, Fn, Node, NodePayload, NodeRef, Package, PackageMember, Type,
    Unop,
};
use crate::ir::{Param, ParamId};
use crate::ir_rebase_ids::rebase_fn_ids;
use crate::ir_utils::compact_and_toposort_in_place;
use crate::math::ceil_log2;
use xlsynth::IrValue;

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

fn next_text_id(f: &Fn) -> usize {
    f.nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn expect_bits_width(f: &Fn, r: NodeRef, ctx: &str) -> Result<usize, DesugarError> {
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
    f: &Fn,
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
    f: &mut Fn,
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

/// Validates `ext_prio_encode` operands and returns the shared shape info used
/// by both inline lowering and FFI wrapper synthesis.
fn analyze_ext_prio_encode(
    f: &Fn,
    arg: NodeRef,
    lsb_prio: bool,
) -> Result<ExtPrioEncodeShape, DesugarError> {
    let input_width = expect_bits_width(f, arg, "ext_prio_encode.arg")?;
    Ok(ExtPrioEncodeShape::new(input_width, lsb_prio))
}

/// Appends the basis-op implementation of `ext_prio_encode` and returns the
/// lowered encoded-result node.
fn append_lowered_ext_prio_encode(f: &mut Fn, arg: NodeRef, shape: ExtPrioEncodeShape) -> NodeRef {
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

fn push_node(f: &mut Fn, ty: Type, payload: NodePayload) -> NodeRef {
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

fn make_zero_bits_literal(f: &mut Fn, width: usize) -> NodeRef {
    push_node(
        f,
        Type::Bits(width),
        NodePayload::Literal(IrValue::make_ubits(width, 0).expect("zero bits literal")),
    )
}

fn zext_or_truncate_to_width(
    f: &mut Fn,
    operand: NodeRef,
    output_width: usize,
    ctx: &str,
) -> Result<NodeRef, DesugarError> {
    let operand_width = expect_bits_width(f, operand, ctx)?;
    if operand_width == output_width {
        Ok(operand)
    } else if operand_width < output_width {
        Ok(push_node(
            f,
            Type::Bits(output_width),
            NodePayload::ZeroExt {
                arg: operand,
                new_bit_count: output_width,
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

fn desugar_ext_carry_out_in_fn(f: &mut Fn) -> Result<bool, DesugarError> {
    let mut changed = false;

    // Snapshot length so we only visit original nodes; desugaring appends nodes.
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
    arch: ExtNaryAddArchitecture,
}

/// Validates `ext_nary_add` shape and returns the widths needed by both inline
/// lowering and FFI wrapper synthesis.
fn analyze_ext_nary_add(
    f: &Fn,
    result: NodeRef,
    operands: &[NodeRef],
    arch: ExtNaryAddArchitecture,
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
    let mut operand_widths = Vec::with_capacity(operands.len());
    for (i, operand) in operands.iter().enumerate() {
        operand_widths.push(expect_bits_width(
            f,
            *operand,
            &format!("ext_nary_add.operand[{i}]"),
        )?);
    }
    Ok(ExtNaryAddShape {
        output_width,
        operand_widths,
        arch,
    })
}

/// Appends the basis-op implementation of `ext_nary_add` and returns the
/// lowered sum node.
fn append_lowered_ext_nary_add(
    f: &mut Fn,
    operands: &[NodeRef],
    output_width: usize,
) -> Result<NodeRef, DesugarError> {
    if output_width == 0 || operands.is_empty() {
        return Ok(make_zero_bits_literal(f, output_width));
    }

    let mut resized_operands = Vec::with_capacity(operands.len());
    for operand in operands {
        resized_operands.push(zext_or_truncate_to_width(
            f,
            *operand,
            output_width,
            "ext_nary_add.operand",
        )?);
    }

    let mut acc = resized_operands[0];
    for operand in resized_operands.into_iter().skip(1) {
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
    ExtNaryAdd {
        output_width: usize,
        operand_widths: Vec<usize>,
        arch: ExtNaryAddArchitecture,
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
        FfiWrapKey::ExtNaryAdd {
            output_width,
            operand_widths,
            arch,
        } => {
            let operand_widths_text = operand_widths
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("_");
            format!(
                "__pir_ext__ext_nary_add__outw{}__ops{}__arch{}",
                output_width, operand_widths_text, arch
            )
        }
        FfiWrapKey::ExtPrioEncode {
            input_width,
            lsb_prio,
        } => format!(
            "__pir_ext__ext_prio_encode__w{}__lsb{}",
            input_width,
            if *lsb_prio { 1 } else { 0 }
        ),
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

fn helper_code_template(key: &FfiWrapKey) -> String {
    match key {
        FfiWrapKey::ExtCarryOut { width } => format!(
            "pir_ext_carry_out {{fn}} (.lhs({{lhs}}), .rhs({{rhs}}), .c_in({{c_in}}), .out({{return}})); /* xlsynth_pir_ext=ext_carry_out;width={width} */"
        ),
        FfiWrapKey::ExtNaryAdd {
            output_width,
            operand_widths,
            arch,
        } => {
            let operand_bindings = operand_widths
                .iter()
                .enumerate()
                .map(|(i, _)| format!(".op{i}({{op{i}}})"))
                .collect::<Vec<_>>()
                .join(", ");
            let operand_width_list = operand_widths
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let port_list = if operand_bindings.is_empty() {
                ".out({return})".to_string()
            } else {
                format!("{operand_bindings}, .out({{return}})")
            };
            format!(
                "pir_ext_nary_add {{fn}} ({port_list}); /* xlsynth_pir_ext=ext_nary_add;out_width={output_width};operand_widths={operand_width_list};arch={arch} */"
            )
        }
        FfiWrapKey::ExtPrioEncode {
            input_width,
            lsb_prio,
        } => format!(
            "pir_ext_prio_encode {{fn}} (.arg({{arg}}), .out({{return}})); /* xlsynth_pir_ext=ext_prio_encode;width={input_width};lsb_prio={} */",
            if *lsb_prio { "true" } else { "false" }
        ),
    }
}

fn make_reserved_nil_node() -> Node {
    Node {
        text_id: 0,
        name: Some("reserved_zero_node".to_string()),
        ty: Type::nil(),
        payload: NodePayload::Nil,
        pos: None,
    }
}

fn make_helper_with_params(name: String, params: Vec<Param>, ret_ty: Type, key: &FfiWrapKey) -> Fn {
    let mut nodes = vec![make_reserved_nil_node()];
    for param in &params {
        nodes.push(Node {
            text_id: param.id.get_wrapped_id(),
            name: Some(param.name.clone()),
            ty: param.ty.clone(),
            payload: NodePayload::GetParam(param.id),
            pos: None,
        });
    }
    Fn {
        name,
        params,
        ret_ty,
        nodes,
        ret_node_ref: None,
        outer_attrs: vec![format_ffi_proto_outer_attr(&helper_code_template(key))],
        inner_attrs: Vec::new(),
    }
}

fn make_helper_fn(name: String, key: &FfiWrapKey) -> Fn {
    match key {
        FfiWrapKey::ExtCarryOut { width } => {
            let params = vec![
                Param {
                    name: "lhs".to_string(),
                    ty: Type::Bits(*width),
                    id: ParamId::new(1),
                },
                Param {
                    name: "rhs".to_string(),
                    ty: Type::Bits(*width),
                    id: ParamId::new(2),
                },
                Param {
                    name: "c_in".to_string(),
                    ty: Type::Bits(1),
                    id: ParamId::new(3),
                },
            ];
            let mut helper = make_helper_with_params(name, params, Type::Bits(1), key);
            let shape = ExtCarryOutShape { width: *width };
            let lowered = append_lowered_ext_carry_out(
                &mut helper,
                NodeRef { index: 1 },
                NodeRef { index: 2 },
                NodeRef { index: 3 },
                shape,
            );
            let ret_node_ref = push_node(
                &mut helper,
                Type::Bits(1),
                NodePayload::Unop(Unop::Identity, lowered),
            );
            helper.ret_node_ref = Some(ret_node_ref);
            helper
        }
        FfiWrapKey::ExtNaryAdd {
            output_width,
            operand_widths,
            arch: _,
        } => {
            let params = operand_widths
                .iter()
                .enumerate()
                .map(|(i, width)| Param {
                    name: format!("op{i}"),
                    ty: Type::Bits(*width),
                    id: ParamId::new(i.saturating_add(1)),
                })
                .collect::<Vec<_>>();
            let mut helper = make_helper_with_params(name, params, Type::Bits(*output_width), key);
            let operand_refs = operand_widths
                .iter()
                .enumerate()
                .map(|(i, _)| NodeRef {
                    index: i.saturating_add(1),
                })
                .collect::<Vec<_>>();
            let lowered = append_lowered_ext_nary_add(&mut helper, &operand_refs, *output_width)
                .expect("helper ext_nary_add lowering must be well-typed");
            let ret_node_ref = push_node(
                &mut helper,
                Type::Bits(*output_width),
                NodePayload::Unop(Unop::Identity, lowered),
            );
            helper.ret_node_ref = Some(ret_node_ref);
            helper
        }
        FfiWrapKey::ExtPrioEncode {
            input_width,
            lsb_prio,
        } => {
            let params = vec![Param {
                name: "arg".to_string(),
                ty: Type::Bits(*input_width),
                id: ParamId::new(1),
            }];
            let shape = ExtPrioEncodeShape::new(*input_width, *lsb_prio);
            let mut helper =
                make_helper_with_params(name, params, Type::Bits(shape.output_width), key);
            let lowered = append_lowered_ext_prio_encode(&mut helper, NodeRef { index: 1 }, shape);
            let ret_node_ref = push_node(
                &mut helper,
                Type::Bits(shape.output_width),
                NodePayload::Unop(Unop::Identity, lowered),
            );
            helper.ret_node_ref = Some(ret_node_ref);
            helper
        }
    }
}

fn max_text_id_in_fn(f: &Fn) -> usize {
    f.nodes.iter().map(|node| node.text_id).max().unwrap_or(0)
}

fn max_text_id_in_package(pkg: &Package) -> usize {
    pkg.members
        .iter()
        .map(|member| match member {
            PackageMember::Function(f) => max_text_id_in_fn(f),
            PackageMember::Block { func, .. } => max_text_id_in_fn(func),
        })
        .max()
        .unwrap_or(0)
}

fn get_or_create_helper_name(
    key: &FfiWrapKey,
    helper_names: &mut BTreeMap<FfiWrapKey, String>,
    helper_fns: &mut Vec<Fn>,
    existing_names: &mut BTreeSet<String>,
    current_max_text_id: &mut usize,
) -> String {
    if let Some(existing) = helper_names.get(key) {
        return existing.clone();
    }
    let name = make_unique_helper_name(&helper_base_name(key), existing_names);
    let helper = make_helper_fn(name.clone(), key);
    let rebased_helper = if *current_max_text_id == 0 {
        helper
    } else {
        rebase_fn_ids(&helper, *current_max_text_id)
    };
    *current_max_text_id = max_text_id_in_fn(&rebased_helper);
    helper_names.insert(key.clone(), name.clone());
    helper_fns.push(rebased_helper);
    name
}

fn wrap_extensions_in_fn(
    f: &mut Fn,
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
                );
                let node = f.get_node_mut(nr);
                node.ty = Type::Bits(1);
                node.payload = NodePayload::Invoke {
                    to_apply: helper_name,
                    operands: vec![lhs, rhs, c_in],
                };
                changed = true;
            }
            NodePayload::ExtNaryAdd { operands, arch } => {
                let shape = analyze_ext_nary_add(f, nr, &operands, arch)?;
                let key = FfiWrapKey::ExtNaryAdd {
                    output_width: shape.output_width,
                    operand_widths: shape.operand_widths.clone(),
                    arch: shape.arch,
                };
                let helper_name = get_or_create_helper_name(
                    &key,
                    helper_names,
                    helper_fns,
                    existing_names,
                    current_max_text_id,
                );
                let node = f.get_node_mut(nr);
                node.ty = Type::Bits(shape.output_width);
                node.payload = NodePayload::Invoke {
                    to_apply: helper_name,
                    operands,
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
                );
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
        .map(|member| match member {
            PackageMember::Function(f) => f.name.clone(),
            PackageMember::Block { func, .. } => func.name.clone(),
        })
        .collect();
    let mut helper_names: BTreeMap<FfiWrapKey, String> = BTreeMap::new();
    let mut helper_fns: Vec<Fn> = Vec::new();
    let mut current_max_text_id = max_text_id_in_package(pkg);

    for member in pkg.members.iter_mut() {
        match member {
            PackageMember::Function(f) => {
                let _changed = wrap_extensions_in_fn(
                    f,
                    &mut helper_names,
                    &mut helper_fns,
                    &mut existing_names,
                    &mut current_max_text_id,
                )?;
            }
            PackageMember::Block { func, .. } => {
                let _changed = wrap_extensions_in_fn(
                    func,
                    &mut helper_names,
                    &mut helper_fns,
                    &mut existing_names,
                    &mut current_max_text_id,
                )?;
            }
        }
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

fn desugar_ext_nary_add_in_fn(f: &mut Fn) -> Result<bool, DesugarError> {
    let mut changed = false;

    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        let NodePayload::ExtNaryAdd { operands, arch } = payload else {
            continue;
        };
        changed = true;

        let shape = analyze_ext_nary_add(f, nr, &operands, arch)?;
        let lowered = append_lowered_ext_nary_add(f, &operands, shape.output_width)?;

        let node = f.get_node_mut(nr);
        node.ty = Type::Bits(shape.output_width);
        node.payload = NodePayload::Unop(Unop::Identity, lowered);
    }

    Ok(changed)
}

fn desugar_ext_prio_encode_in_fn(f: &mut Fn) -> Result<bool, DesugarError> {
    let mut changed = false;

    // Snapshot length so we only visit original nodes; desugaring appends nodes.
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

/// Desugars extension ops within `f` into upstream-compatible PIR operations.
///
/// This function also normalizes the node list into a valid topological order.
pub fn desugar_extensions_in_fn(f: &mut Fn) -> Result<(), DesugarError> {
    let _changed = desugar_ext_carry_out_in_fn(f)?
        | desugar_ext_nary_add_in_fn(f)?
        | desugar_ext_prio_encode_in_fn(f)?;
    compact_and_toposort_in_place(f).map_err(DesugarError::new)?;
    Ok(())
}

/// Desugars extension ops within `pkg` into upstream-compatible PIR operations.
pub fn desugar_extensions_in_package(pkg: &mut Package) -> Result<(), DesugarError> {
    for member in pkg.members.iter_mut() {
        match member {
            PackageMember::Function(f) => desugar_extensions_in_fn(f)?,
            PackageMember::Block { func, .. } => desugar_extensions_in_fn(func)?,
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
