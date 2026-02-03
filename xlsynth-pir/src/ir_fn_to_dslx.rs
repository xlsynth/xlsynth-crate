// SPDX-License-Identifier: Apache-2.0

//! Converts an XLS IR function into DSLX.

use std::collections::{HashMap, HashSet};

use xlsynth::ir_value::IrFormatPreference;

use crate::ir::{self, Binop, NaryOp, NodePayload, NodeRef, ParamId, Type, Unop};
use crate::ir_parser;
use crate::ir_utils::get_topological;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrFnToDslxResult {
    pub function_name: String,
    pub dslx_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrFnToDslxError {
    ParseOrValidate(String),
    TopFunctionNotFound(String),
    MissingTopFunction,
    MissingReturnNode(String),
    UnsupportedType(String),
    UnsupportedNode(String),
    MissingNodeName(usize),
    Internal(String),
}

impl std::fmt::Display for IrFnToDslxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IrFnToDslxError::ParseOrValidate(e) => write!(f, "Parse/validate error: {}", e),
            IrFnToDslxError::TopFunctionNotFound(name) => {
                write!(f, "Top function not found in package: {}", name)
            }
            IrFnToDslxError::MissingTopFunction => {
                write!(f, "No top function present in package")
            }
            IrFnToDslxError::MissingReturnNode(name) => {
                write!(f, "Function has no return node: {}", name)
            }
            IrFnToDslxError::UnsupportedType(msg) => write!(f, "Unsupported type: {}", msg),
            IrFnToDslxError::UnsupportedNode(msg) => write!(f, "Unsupported node: {}", msg),
            IrFnToDslxError::MissingNodeName(i) => {
                write!(f, "Missing translated DSLX name for node index {}", i)
            }
            IrFnToDslxError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for IrFnToDslxError {}

/// Converts a package IR string into DSLX function text.
///
/// If `top` is `Some`, that function is selected. Otherwise the package top
/// function is used.
pub fn convert_ir_package_fn_to_dslx(
    ir_package_text: &str,
    top: Option<&str>,
) -> Result<IrFnToDslxResult, IrFnToDslxError> {
    let mut parser = ir_parser::Parser::new(ir_package_text);
    let pkg = parser
        .parse_and_validate_package()
        .map_err(|e| IrFnToDslxError::ParseOrValidate(e.to_string()))?;

    let ir_fn = match top {
        Some(name) => pkg
            .get_fn(name)
            .ok_or_else(|| IrFnToDslxError::TopFunctionNotFound(name.to_string()))?,
        None => pkg
            .get_top_fn()
            .ok_or(IrFnToDslxError::MissingTopFunction)?,
    };

    convert_ir_fn_to_dslx(ir_fn)
}

/// Converts an in-memory IR function into DSLX function text.
pub fn convert_ir_fn_to_dslx(func: &ir::Fn) -> Result<IrFnToDslxResult, IrFnToDslxError> {
    for p in &func.params {
        let _ = type_to_dslx(&p.ty)?;
    }
    let ret_ty_str = type_to_dslx(&func.ret_ty)?;
    let ret_node_ref = func
        .ret_node_ref
        .ok_or_else(|| IrFnToDslxError::MissingReturnNode(func.name.clone()))?;

    let mut used_names: HashSet<String> = HashSet::new();
    let mut param_names: Vec<String> = Vec::with_capacity(func.params.len());
    let mut param_name_by_id: HashMap<ParamId, String> = HashMap::with_capacity(func.params.len());
    for p in &func.params {
        let base = sanitize_identifier(&p.name);
        let chosen = make_unique_identifier(&base, &mut used_names);
        param_name_by_id.insert(p.id, chosen.clone());
        param_names.push(chosen);
    }

    let mut node_names: Vec<Option<String>> = vec![None; func.nodes.len()];
    for nr in get_topological(func) {
        let node = func.get_node(nr);
        match node.payload {
            NodePayload::Nil => {}
            NodePayload::GetParam(pid) => {
                let pname = param_name_by_id.get(&pid).ok_or_else(|| {
                    IrFnToDslxError::Internal(format!(
                        "GetParam id {} not found in signature",
                        pid.get_wrapped_id()
                    ))
                })?;
                node_names[nr.index] = Some(pname.clone());
            }
            _ => {
                let base = if let Some(name) = &node.name {
                    sanitize_identifier(name)
                } else {
                    sanitize_identifier(&format!(
                        "{}_{}",
                        node.payload.get_operator(),
                        node.text_id
                    ))
                };
                let chosen = make_unique_identifier(&base, &mut used_names);
                node_names[nr.index] = Some(chosen);
            }
        }
    }

    let mut body_lines: Vec<String> = Vec::new();
    for nr in get_topological(func) {
        let node = func.get_node(nr);
        match node.payload {
            NodePayload::Nil | NodePayload::GetParam(_) => continue,
            _ => {}
        }
        let lhs_name = node_names[nr.index]
            .as_ref()
            .ok_or(IrFnToDslxError::MissingNodeName(nr.index))?;
        let lhs_ty = type_to_dslx(&node.ty)?;
        let rhs_expr = lower_node_payload(func, nr, &node_names)?;
        body_lines.push(format!("  let {}: {} = {};", lhs_name, lhs_ty, rhs_expr));
    }

    let ret_expr = node_names[ret_node_ref.index]
        .as_ref()
        .ok_or(IrFnToDslxError::MissingNodeName(ret_node_ref.index))?
        .clone();

    let fn_name = sanitize_identifier(&func.name);
    let params_str = func
        .params
        .iter()
        .zip(param_names.iter())
        .map(|(p, name)| {
            let ty = type_to_dslx(&p.ty)?;
            Ok(format!("{}: {}", name, ty))
        })
        .collect::<Result<Vec<String>, IrFnToDslxError>>()?
        .join(", ");

    let mut out = String::new();
    out.push_str(&format!(
        "fn {}({}) -> {} {{\n",
        fn_name, params_str, ret_ty_str
    ));
    for line in body_lines {
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str(&format!("  {}\n", ret_expr));
    out.push_str("}\n");

    Ok(IrFnToDslxResult {
        function_name: fn_name,
        dslx_text: out,
    })
}

fn type_to_dslx(ty: &Type) -> Result<String, IrFnToDslxError> {
    match ty {
        Type::Bits(w) => Ok(format!("uN[{}]", w)),
        Type::Token => Ok("token".to_string()),
        Type::Tuple(members) => {
            let members = members
                .iter()
                .map(|m| type_to_dslx(m))
                .collect::<Result<Vec<String>, IrFnToDslxError>>()?;
            if members.is_empty() {
                Ok("()".to_string())
            } else if members.len() == 1 {
                Ok(format!("({},)", members[0]))
            } else {
                Ok(format!("({})", members.join(", ")))
            }
        }
        Type::Array(data) => {
            let elem = type_to_dslx(&data.element_type)?;
            Ok(format!("{}[{}]", elem, data.element_count))
        }
    }
}

fn bits_width(ty: &Type) -> Result<usize, IrFnToDslxError> {
    match ty {
        Type::Bits(w) => Ok(*w),
        _ => Err(IrFnToDslxError::UnsupportedType(format!(
            "only bits types are supported in MVP; got {}",
            ty
        ))),
    }
}

fn node_name(node_names: &[Option<String>], nr: NodeRef) -> Result<&str, IrFnToDslxError> {
    node_names[nr.index]
        .as_deref()
        .ok_or(IrFnToDslxError::MissingNodeName(nr.index))
}

fn render_tuple_expr(elements: &[String]) -> String {
    if elements.is_empty() {
        "()".to_string()
    } else if elements.len() == 1 {
        format!("({},)", elements[0])
    } else {
        format!("({})", elements.join(", "))
    }
}

fn render_array_index_expr(base: &str, indices: &[String]) -> String {
    let mut expr = base.to_string();
    for idx in indices {
        expr.push_str(&format!("[{}]", idx));
    }
    expr
}

fn render_array_update_expr(base: &str, indices: &[String], value: &str) -> String {
    if indices.is_empty() {
        return value.to_string();
    }
    let mut update_expr = value.to_string();
    for depth in (0..indices.len()).rev() {
        let prefix = render_array_index_expr(base, &indices[..depth]);
        update_expr = format!("update({}, {}, {})", prefix, indices[depth], update_expr);
    }
    update_expr
}

fn zero_value_expr_for_type(ty: &Type) -> Result<String, IrFnToDslxError> {
    match ty {
        Type::Bits(w) => Ok(format!("uN[{}]:0", w)),
        Type::Token => Err(IrFnToDslxError::UnsupportedType(
            "cannot synthesize a token zero literal".to_string(),
        )),
        Type::Tuple(members) => {
            let members = members
                .iter()
                .map(|m| zero_value_expr_for_type(m))
                .collect::<Result<Vec<String>, IrFnToDslxError>>()?;
            Ok(render_tuple_expr(&members))
        }
        Type::Array(data) => {
            let elem_zero = zero_value_expr_for_type(&data.element_type)?;
            let elems = std::iter::repeat_n(elem_zero, data.element_count)
                .collect::<Vec<String>>()
                .join(", ");
            let ty_str = type_to_dslx(ty)?;
            Ok(format!("{}:[{}]", ty_str, elems))
        }
    }
}

fn lower_node_payload(
    func: &ir::Fn,
    nr: NodeRef,
    node_names: &[Option<String>],
) -> Result<String, IrFnToDslxError> {
    let node = func.get_node(nr);
    match &node.payload {
        NodePayload::Literal(v) => {
            let w = bits_width(&node.ty)?;
            let value = v
                .to_string_fmt_no_prefix(IrFormatPreference::Default)
                .map_err(|e| IrFnToDslxError::UnsupportedNode(e.to_string()))?;
            Ok(format!("uN[{}]:{}", w, value))
        }
        NodePayload::Unop(op, arg) => {
            let arg_name = node_name(node_names, *arg)?;
            match op {
                Unop::Identity => Ok(arg_name.to_string()),
                Unop::Not => Ok(format!("!{}", arg_name)),
                Unop::Neg => Ok(format!("-{}", arg_name)),
                Unop::OrReduce => Ok(format!("or_reduce({}) as uN[1]", arg_name)),
                Unop::AndReduce => Ok(format!("and_reduce({}) as uN[1]", arg_name)),
                Unop::XorReduce => Ok(format!("xor_reduce({}) as uN[1]", arg_name)),
                Unop::Reverse => Ok(format!("rev({})", arg_name)),
            }
        }
        NodePayload::Binop(op, lhs, rhs) => {
            let lhs_name = node_name(node_names, *lhs)?;
            let rhs_name = node_name(node_names, *rhs)?;
            match op {
                Binop::Add => Ok(format!("{} + {}", lhs_name, rhs_name)),
                Binop::Sub => Ok(format!("{} - {}", lhs_name, rhs_name)),
                Binop::Shll => Ok(format!("{} << {}", lhs_name, rhs_name)),
                Binop::Shrl => Ok(format!("{} >> {}", lhs_name, rhs_name)),
                Binop::Eq => Ok(format!("({} == {}) as uN[1]", lhs_name, rhs_name)),
                Binop::Ne => Ok(format!("({} != {}) as uN[1]", lhs_name, rhs_name)),
                Binop::Uge => Ok(format!("({} >= {}) as uN[1]", lhs_name, rhs_name)),
                Binop::Ugt => Ok(format!("({} > {}) as uN[1]", lhs_name, rhs_name)),
                Binop::Ult => Ok(format!("({} < {}) as uN[1]", lhs_name, rhs_name)),
                Binop::Ule => Ok(format!("({} <= {}) as uN[1]", lhs_name, rhs_name)),
                Binop::Umul => Ok(format!("{} * {}", lhs_name, rhs_name)),
                Binop::Udiv => Ok(format!("{} / {}", lhs_name, rhs_name)),
                Binop::Umod => Ok(format!("{} % {}", lhs_name, rhs_name)),
                Binop::Smul => {
                    let lhs_w = bits_width(&func.get_node(*lhs).ty)?;
                    let rhs_w = bits_width(&func.get_node(*rhs).ty)?;
                    let out_w = bits_width(&node.ty)?;
                    if lhs_w != rhs_w {
                        return Err(IrFnToDslxError::UnsupportedNode(format!(
                            "smul requires equal operand widths in MVP; got {} and {}",
                            lhs_w, rhs_w
                        )));
                    }
                    Ok(format!(
                        "(({} as sN[{}]) * ({} as sN[{}])) as uN[{}]",
                        lhs_name, lhs_w, rhs_name, rhs_w, out_w
                    ))
                }
                Binop::Sdiv => {
                    let lhs_w = bits_width(&func.get_node(*lhs).ty)?;
                    let rhs_w = bits_width(&func.get_node(*rhs).ty)?;
                    let out_w = bits_width(&node.ty)?;
                    if lhs_w != rhs_w {
                        return Err(IrFnToDslxError::UnsupportedNode(format!(
                            "sdiv requires equal operand widths in MVP; got {} and {}",
                            lhs_w, rhs_w
                        )));
                    }
                    Ok(format!(
                        "(({} as sN[{}]) / ({} as sN[{}])) as uN[{}]",
                        lhs_name, lhs_w, rhs_name, rhs_w, out_w
                    ))
                }
                Binop::Smod => {
                    let lhs_w = bits_width(&func.get_node(*lhs).ty)?;
                    let rhs_w = bits_width(&func.get_node(*rhs).ty)?;
                    let out_w = bits_width(&node.ty)?;
                    if lhs_w != rhs_w {
                        return Err(IrFnToDslxError::UnsupportedNode(format!(
                            "smod requires equal operand widths in MVP; got {} and {}",
                            lhs_w, rhs_w
                        )));
                    }
                    Ok(format!(
                        "(({} as sN[{}]) % ({} as sN[{}])) as uN[{}]",
                        lhs_name, lhs_w, rhs_name, rhs_w, out_w
                    ))
                }
                Binop::Sge => {
                    let lhs_w = bits_width(&func.get_node(*lhs).ty)?;
                    let rhs_w = bits_width(&func.get_node(*rhs).ty)?;
                    if lhs_w != rhs_w {
                        return Err(IrFnToDslxError::UnsupportedNode(format!(
                            "sge requires equal operand widths in MVP; got {} and {}",
                            lhs_w, rhs_w
                        )));
                    }
                    Ok(format!(
                        "(({} as sN[{}]) >= ({} as sN[{}])) as uN[1]",
                        lhs_name, lhs_w, rhs_name, rhs_w
                    ))
                }
                Binop::Sgt => {
                    let lhs_w = bits_width(&func.get_node(*lhs).ty)?;
                    let rhs_w = bits_width(&func.get_node(*rhs).ty)?;
                    if lhs_w != rhs_w {
                        return Err(IrFnToDslxError::UnsupportedNode(format!(
                            "sgt requires equal operand widths in MVP; got {} and {}",
                            lhs_w, rhs_w
                        )));
                    }
                    Ok(format!(
                        "(({} as sN[{}]) > ({} as sN[{}])) as uN[1]",
                        lhs_name, lhs_w, rhs_name, rhs_w
                    ))
                }
                Binop::Slt => {
                    let lhs_w = bits_width(&func.get_node(*lhs).ty)?;
                    let rhs_w = bits_width(&func.get_node(*rhs).ty)?;
                    if lhs_w != rhs_w {
                        return Err(IrFnToDslxError::UnsupportedNode(format!(
                            "slt requires equal operand widths in MVP; got {} and {}",
                            lhs_w, rhs_w
                        )));
                    }
                    Ok(format!(
                        "(({} as sN[{}]) < ({} as sN[{}])) as uN[1]",
                        lhs_name, lhs_w, rhs_name, rhs_w
                    ))
                }
                Binop::Sle => {
                    let lhs_w = bits_width(&func.get_node(*lhs).ty)?;
                    let rhs_w = bits_width(&func.get_node(*rhs).ty)?;
                    if lhs_w != rhs_w {
                        return Err(IrFnToDslxError::UnsupportedNode(format!(
                            "sle requires equal operand widths in MVP; got {} and {}",
                            lhs_w, rhs_w
                        )));
                    }
                    Ok(format!(
                        "(({} as sN[{}]) <= ({} as sN[{}])) as uN[1]",
                        lhs_name, lhs_w, rhs_name, rhs_w
                    ))
                }
                Binop::Shra => {
                    let lhs_w = bits_width(&func.get_node(*lhs).ty)?;
                    let out_w = bits_width(&node.ty)?;
                    Ok(format!(
                        "(({} as sN[{}]) >> {}) as uN[{}]",
                        lhs_name, lhs_w, rhs_name, out_w
                    ))
                }
                Binop::ArrayConcat => Ok(format!("{} ++ {}", lhs_name, rhs_name)),
                Binop::Umulp => Ok(format!("umulp({}, {})", lhs_name, rhs_name)),
                Binop::Smulp => {
                    let lhs_w = bits_width(&func.get_node(*lhs).ty)?;
                    let rhs_w = bits_width(&func.get_node(*rhs).ty)?;
                    Ok(format!(
                        "smulp({} as sN[{}], {} as sN[{}])",
                        lhs_name, lhs_w, rhs_name, rhs_w
                    ))
                }
                Binop::Gate => {
                    let zeros = zero_value_expr_for_type(&node.ty)?;
                    Ok(format!(
                        "if {} == u1:1 {{ {} }} else {{ {} }}",
                        lhs_name, rhs_name, zeros
                    ))
                }
            }
        }
        NodePayload::Tuple(elements) => {
            let values = elements
                .iter()
                .map(|nr| node_name(node_names, *nr).map(|s| s.to_string()))
                .collect::<Result<Vec<String>, IrFnToDslxError>>()?;
            Ok(render_tuple_expr(&values))
        }
        NodePayload::TupleIndex { tuple, index } => {
            let tuple_name = node_name(node_names, *tuple)?;
            Ok(format!("{}.{}", tuple_name, index))
        }
        NodePayload::Array(elements) => {
            let values = elements
                .iter()
                .map(|nr| node_name(node_names, *nr).map(|s| s.to_string()))
                .collect::<Result<Vec<String>, IrFnToDslxError>>()?;
            let ty = type_to_dslx(&node.ty)?;
            Ok(format!("{}:[{}]", ty, values.join(", ")))
        }
        NodePayload::ArrayIndex { array, indices, .. } => {
            let array_name = node_name(node_names, *array)?;
            let indices = indices
                .iter()
                .map(|nr| node_name(node_names, *nr).map(|s| s.to_string()))
                .collect::<Result<Vec<String>, IrFnToDslxError>>()?;
            Ok(render_array_index_expr(array_name, &indices))
        }
        NodePayload::ArrayUpdate {
            array,
            value,
            indices,
            ..
        } => {
            let array_name = node_name(node_names, *array)?;
            let value_name = node_name(node_names, *value)?;
            let indices = indices
                .iter()
                .map(|nr| node_name(node_names, *nr).map(|s| s.to_string()))
                .collect::<Result<Vec<String>, IrFnToDslxError>>()?;
            Ok(render_array_update_expr(array_name, &indices, value_name))
        }
        NodePayload::ArraySlice {
            array,
            start,
            width: _,
        } => {
            let array_name = node_name(node_names, *array)?;
            let start_name = node_name(node_names, *start)?;
            let default_value = zero_value_expr_for_type(&node.ty)?;
            Ok(format!(
                "array_slice({}, {}, {})",
                array_name, start_name, default_value
            ))
        }
        NodePayload::Nary(op, nodes) => {
            if nodes.is_empty() {
                return Err(IrFnToDslxError::UnsupportedNode(
                    "n-ary op with zero operands is unsupported".to_string(),
                ));
            }
            let rendered = nodes
                .iter()
                .map(|nr| node_name(node_names, *nr).map(|s| s.to_string()))
                .collect::<Result<Vec<String>, IrFnToDslxError>>()?;
            match op {
                NaryOp::And => Ok(rendered.join(" & ")),
                NaryOp::Or => Ok(rendered.join(" | ")),
                NaryOp::Xor => Ok(rendered.join(" ^ ")),
                NaryOp::Concat => Ok(rendered.join(" ++ ")),
                NaryOp::Nand => Ok(format!("!({})", rendered.join(" & "))),
                NaryOp::Nor => Ok(format!("!({})", rendered.join(" | "))),
            }
        }
        NodePayload::SignExt { arg, new_bit_count } => {
            let arg_name = node_name(node_names, *arg)?;
            let arg_w = bits_width(&func.get_node(*arg).ty)?;
            if *new_bit_count < arg_w {
                return Err(IrFnToDslxError::UnsupportedNode(format!(
                    "sign_ext new_bit_count {} is smaller than argument width {}",
                    new_bit_count, arg_w
                )));
            }
            Ok(format!(
                "((({} as sN[{}]) as sN[{}]) as uN[{}])",
                arg_name, arg_w, new_bit_count, new_bit_count
            ))
        }
        NodePayload::ZeroExt { arg, new_bit_count } => {
            let arg_name = node_name(node_names, *arg)?;
            Ok(format!("({} as uN[{}])", arg_name, new_bit_count))
        }
        NodePayload::BitSlice { arg, start, width } => {
            let arg_name = node_name(node_names, *arg)?;
            Ok(format!("{}[{}+:uN[{}]]", arg_name, start, width))
        }
        NodePayload::DynamicBitSlice { arg, start, width } => {
            let arg_name = node_name(node_names, *arg)?;
            let start_name = node_name(node_names, *start)?;
            Ok(format!("{}[{}+:uN[{}]]", arg_name, start_name, width))
        }
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => {
            let arg_name = node_name(node_names, *arg)?;
            let start_name = node_name(node_names, *start)?;
            let update_name = node_name(node_names, *update_value)?;
            Ok(format!(
                "bit_slice_update({}, {}, {})",
                arg_name, start_name, update_name
            ))
        }
        NodePayload::OneHot { arg, lsb_prio } => {
            let arg_name = node_name(node_names, *arg)?;
            Ok(format!("one_hot({}, {})", arg_name, lsb_prio))
        }
        NodePayload::Decode { arg, width } => {
            let arg_name = node_name(node_names, *arg)?;
            Ok(format!("decode<uN[{}]>({})", width, arg_name))
        }
        NodePayload::Encode { arg } => {
            let arg_name = node_name(node_names, *arg)?;
            let out_w = bits_width(&node.ty)?;
            Ok(format!("encode({}) as uN[{}]", arg_name, out_w))
        }
        NodePayload::OneHotSel { selector, cases } => {
            let _ = bits_width(&node.ty)?;
            let selector_name = node_name(node_names, *selector)?;
            let case_text = cases
                .iter()
                .map(|c| node_name(node_names, *c).map(|s| s.to_string()))
                .collect::<Result<Vec<String>, IrFnToDslxError>>()?
                .join(", ");
            Ok(format!("one_hot_sel({}, [{}])", selector_name, case_text))
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => {
            let _ = bits_width(&node.ty)?;
            let selector_name = node_name(node_names, *selector)?;
            let case_text = cases
                .iter()
                .map(|c| node_name(node_names, *c).map(|s| s.to_string()))
                .collect::<Result<Vec<String>, IrFnToDslxError>>()?
                .join(", ");
            let default_name = default.ok_or_else(|| {
                IrFnToDslxError::UnsupportedNode(
                    "priority_sel without default is unsupported in MVP".to_string(),
                )
            })?;
            let default_name = node_name(node_names, default_name)?;
            Ok(format!(
                "priority_sel({}, [{}], {})",
                selector_name, case_text, default_name
            ))
        }
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            let selector_name = node_name(node_names, *selector)?;
            let selector_w = bits_width(&func.get_node(*selector).ty)?;
            if selector_w == 0 {
                return Err(IrFnToDslxError::UnsupportedNode(
                    "sel selector with width 0 is unsupported".to_string(),
                ));
            }
            if cases.is_empty() {
                return Err(IrFnToDslxError::UnsupportedNode(
                    "sel with zero cases is unsupported".to_string(),
                ));
            }
            let mut arms: Vec<String> = Vec::with_capacity(cases.len() + 1);
            for (i, c) in cases.iter().enumerate() {
                let case_name = node_name(node_names, *c)?;
                arms.push(format!("uN[{}]:{} => {}", selector_w, i, case_name));
            }
            match default {
                Some(d) => {
                    let default_name = node_name(node_names, *d)?;
                    arms.push(format!("_ => {}", default_name));
                }
                None => {
                    let selector_domain =
                        1usize.checked_shl(selector_w as u32).ok_or_else(|| {
                            IrFnToDslxError::UnsupportedNode(format!(
                                "sel selector width {} too large for domain calculation",
                                selector_w
                            ))
                        })?;
                    if cases.len() != selector_domain {
                        return Err(IrFnToDslxError::UnsupportedNode(format!(
                            "sel without default requires exactly 2^selector_width cases; got {} cases and selector width {}",
                            cases.len(),
                            selector_w
                        )));
                    }
                }
            }
            Ok(format!("match {} {{ {} }}", selector_name, arms.join(", ")))
        }
        _ => Err(IrFnToDslxError::UnsupportedNode(format!(
            "operator not yet supported in MVP: {}",
            node.payload.get_operator()
        ))),
    }
}

fn sanitize_identifier(input: &str) -> String {
    let mut out = String::new();
    for c in input.chars() {
        if c == '_' || c.is_ascii_alphanumeric() {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('n');
    }
    if out.chars().next().unwrap().is_ascii_digit() {
        out = format!("n_{}", out);
    }
    if is_dslx_keyword(&out) {
        out.push_str("_v");
    }
    out
}

fn make_unique_identifier(base: &str, used: &mut HashSet<String>) -> String {
    if used.insert(base.to_string()) {
        return base.to_string();
    }
    let mut i = 2usize;
    loop {
        let candidate = format!("{}_{}", base, i);
        if used.insert(candidate.clone()) {
            return candidate;
        }
        i += 1;
    }
}

fn is_dslx_keyword(s: &str) -> bool {
    matches!(
        s,
        "as" | "const"
            | "else"
            | "enum"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "import"
            | "in"
            | "let"
            | "match"
            | "mod"
            | "pub"
            | "struct"
            | "test"
            | "true"
            | "type"
            | "use"
            | "while"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_simple_add_function() {
        let ir_text = r#"package sample

top fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  x: bits[8] = param(name=x, id=1)
  y: bits[8] = param(name=y, id=2)
  ret add.3: bits[8] = add(x, y, id=3)
}
"#;
        let result = convert_ir_package_fn_to_dslx(ir_text, None).unwrap();
        assert!(result.dslx_text.contains("fn f("));
        assert!(result.dslx_text.contains("let add_3"));
        assert!(result.dslx_text.contains("x + y"));
    }

    #[test]
    fn test_name_collision_is_made_unique() {
        let ir_text = r#"package sample

top fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  x: bits[8] = param(name=x, id=1)
  y: bits[8] = param(name=y, id=2)
  add.3: bits[8] = add(x, y, id=3)
  add_3: bits[8] = add(x, y, id=4)
  ret out: bits[8] = add(add.3, add_3, id=5)
}
"#;
        let result = convert_ir_package_fn_to_dslx(ir_text, None).unwrap();
        assert!(result.dslx_text.contains("let add_3: uN[8]"));
        assert!(result.dslx_text.contains("let add_3_2: uN[8]"));
    }

    #[test]
    fn test_convert_tuple_type_and_tuple_index() {
        let ir_text = r#"package sample

top fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  x: bits[8] = param(name=x, id=1)
  y: bits[8] = param(name=y, id=2)
  t: (bits[8], bits[8]) = tuple(x, y, id=3)
  ret out: bits[8] = tuple_index(t, index=1, id=4)
}
"#;
        let result = convert_ir_package_fn_to_dslx(ir_text, None).unwrap();
        assert!(result.dslx_text.contains("let t: (uN[8], uN[8]) = (x, y);"));
        assert!(result.dslx_text.contains("t.1"));
    }

    #[test]
    fn test_convert_array_update_and_slice() {
        let ir_text = r#"package sample

top fn f(a: bits[8][4] id=1, i: bits[2] id=2, v: bits[8] id=3) -> bits[8][2] {
  a: bits[8][4] = param(name=a, id=1)
  i: bits[2] = param(name=i, id=2)
  v: bits[8] = param(name=v, id=3)
  updated: bits[8][4] = array_update(a, v, indices=[i], id=4)
  ret out: bits[8][2] = array_slice(updated, i, width=2, id=5)
}
"#;
        let result = convert_ir_package_fn_to_dslx(ir_text, None).unwrap();
        assert!(
            result
                .dslx_text
                .contains("let updated: uN[8][4] = update(a, i, v);")
        );
        assert!(
            result
                .dslx_text
                .contains("array_slice(updated, i, uN[8][2]:[uN[8]:0, uN[8]:0])")
        );
    }

    #[test]
    fn test_convert_reverse_gate_and_umulp() {
        let ir_text = r#"package sample

top fn f(p: bits[1] id=1, x: bits[8] id=2, y: bits[8] id=3) -> (bits[16], bits[16]) {
  p: bits[1] = param(name=p, id=1)
  x: bits[8] = param(name=x, id=2)
  y: bits[8] = param(name=y, id=3)
  g: bits[8] = gate(p, x, id=4)
  r: bits[8] = reverse(g, id=5)
  ret out: (bits[16], bits[16]) = umulp(r, y, id=6)
}
"#;
        let result = convert_ir_package_fn_to_dslx(ir_text, None).unwrap();
        assert!(
            result
                .dslx_text
                .contains("if p == u1:1 { x } else { uN[8]:0 }")
        );
        assert!(result.dslx_text.contains("rev(g)"));
        assert!(result.dslx_text.contains("umulp(r, y)"));
    }

    #[test]
    fn test_convert_token_param_and_return_type() {
        let ir_text = r#"package sample

top fn f(t: token id=1) -> token {
  ret t: token = param(name=t, id=1)
}
"#;
        let result = convert_ir_package_fn_to_dslx(ir_text, None).unwrap();
        assert!(result.dslx_text.contains("fn f(t: token) -> token"));
    }

    #[test]
    fn test_sign_ext_uses_input_sign_bit() {
        let ir_text = r#"package sample

top fn f(x: bits[8] id=1) -> bits[13] {
  x: bits[8] = param(name=x, id=1)
  ret sx: bits[13] = sign_ext(x, new_bit_count=13, id=2)
}
"#;
        let result = convert_ir_package_fn_to_dslx(ir_text, None).unwrap();
        assert!(
            result
                .dslx_text
                .contains("(((x as sN[8]) as sN[13]) as uN[13])")
        );
    }
}
