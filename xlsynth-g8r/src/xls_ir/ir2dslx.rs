// SPDX-License-Identifier: Apache-2.0

//! Minimal IR→DSLX emitter.
//!
//! This walks IR nodes in their existing order (expected to be topological)
//! and emits a simple, deterministic DSLX function using let-bound
//! expressions, finishing with the returned node.
//!
//! Supported ops (initial):
//! - Params, Literals
//! - Binops: add, sub, umul/smul ("*")
//! - Shifts: shll ("<<"), shrl (">>")
//! - Nary: and/or/xor (folded with &, |, ^)
//! - Tuple/Array construction (basic)
//!
//! For unsupported nodes, we return an error describing the operator.

use crate::xls_ir::ir;

fn ty_to_dslx(t: &ir::Type) -> String {
    match t {
        ir::Type::Token => "token".to_string(),
        ir::Type::Bits(n) => format!("u{}", n),
        ir::Type::Tuple(members) => {
            let parts: Vec<String> = members.iter().map(|m| ty_to_dslx(m)).collect();
            format!("({})", parts.join(", "))
        }
        ir::Type::Array(arr) => {
            let elem = ty_to_dslx(&arr.element_type);
            format!("{}[{}]", elem, arr.element_count)
        }
    }
}

fn literal_to_dslx(value: &xlsynth::IrValue) -> String {
    // Prefer uN:VALUE form in DSLX.
    let bits = value.bit_count().unwrap_or(0);
    // Use IR value printer without the leading type prefix and attach DSLX uN:
    let no_prefix = value
        .to_string_fmt_no_prefix(xlsynth::ir_value::IrFormatPreference::Default)
        .unwrap();
    format!("u{}:{}", bits, no_prefix)
}

fn node_display_name(f: &ir::Fn, nr: ir::NodeRef) -> String {
    let n = f.get_node(nr);
    if let Some(name) = &n.name {
        return name.clone();
    }
    format!("{}_{}", n.payload.get_operator(), n.text_id)
}

fn emit_expr_for_node(f: &ir::Fn, nr: ir::NodeRef) -> Result<String, String> {
    use ir::NodePayload as P;
    let n = f.get_node(nr);
    let name = node_display_name(f, nr);
    let ty = ty_to_dslx(&n.ty);
    let expr = match &n.payload {
        P::Nil => return Ok(String::new()),
        P::GetParam(_) => return Ok(String::new()),
        P::Literal(v) => format!("let {}: {} = {};", name, ty, literal_to_dslx(v)),

        P::Binop(op, a, b) => {
            let lhs = node_display_name(f, *a);
            let rhs = node_display_name(f, *b);
            let op_str = match op {
                ir::Binop::Add => "+",
                ir::Binop::Sub => "-",
                ir::Binop::Umul | ir::Binop::Smul => "*",
                ir::Binop::Shll => "<<",
                ir::Binop::Shrl => ">>",
                _ => return Err(format!("unsupported binop: {}", ir::binop_to_operator(*op))),
            };
            format!("let {}: {} = {} {} {};", name, ty, lhs, op_str, rhs)
        }

        P::Unop(op, a) => {
            let arg = node_display_name(f, *a);
            let s = match op {
                ir::Unop::Not => format!("!{}", arg),
                ir::Unop::Neg => format!("-{}", arg),
                ir::Unop::Identity => format!("{}", arg),
                _ => return Err(format!("unsupported unop: {}", ir::unop_to_operator(*op))),
            };
            format!("let {}: {} = {};", name, ty, s)
        }

        P::Nary(nop, nodes) => {
            let op_str = match nop {
                ir::NaryOp::And => "&",
                ir::NaryOp::Or => "|",
                ir::NaryOp::Xor => "^",
                _ => {
                    return Err(format!(
                        "unsupported n-ary op: {}",
                        ir::nary_op_to_operator(*nop)
                    ));
                }
            };
            let parts: Vec<String> = nodes.iter().map(|r| node_display_name(f, *r)).collect();
            let folded = if parts.is_empty() {
                return Err("empty n-ary node".to_string());
            } else {
                parts.join(&format!(" {} ", op_str))
            };
            format!("let {}: {} = {};", name, ty, folded)
        }

        P::Tuple(elems) => {
            let parts: Vec<String> = elems.iter().map(|r| node_display_name(f, *r)).collect();
            format!("let {}: {} = ({});", name, ty, parts.join(", "))
        }

        P::Array(elems) => {
            let parts: Vec<String> = elems.iter().map(|r| node_display_name(f, *r)).collect();
            format!("let {}: {} = [{}];", name, ty, parts.join(", "))
        }

        // Many nodes are currently unsupported in this minimal emitter.
        other => return Err(format!("unsupported op: {}", other.get_operator())),
    };
    Ok(expr)
}

/// Emits a DSLX function text for the given IR function.
pub fn emit_fn_as_dslx(func: &ir::Fn) -> Result<String, String> {
    let params_str = func
        .params
        .iter()
        .map(|p| format!("{}: {}", p.name, ty_to_dslx(&p.ty)))
        .collect::<Vec<String>>()
        .join(", ");
    let ret_ty = ty_to_dslx(&func.ret_ty);
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("fn {}({}) -> {} {{", func.name, params_str, ret_ty));

    for (i, _node) in func.nodes.iter().enumerate() {
        let nr = ir::NodeRef { index: i };
        if let Ok(s) = emit_expr_for_node(func, nr) {
            if !s.is_empty() {
                lines.push(format!("  {}", s));
            }
        } else {
            let n = func.get_node(nr);
            return Err(format!(
                "unsupported node id {} op {}",
                n.text_id,
                n.payload.get_operator()
            ));
        }
    }

    let ret_ref = func
        .ret_node_ref
        .ok_or_else(|| "function missing ret node ref".to_string())?;
    let ret_name = node_display_name(func, ret_ref);
    lines.push(format!("  {}", ret_name));
    lines.push("}".to_string());
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xls_ir::ir_parser::Parser;
    use test_case::test_case;

    #[test]
    fn test_add_literal() {
        let ir_text = r#"fn plus_one(x: bits[32] id=1) -> bits[32] {
  one: bits[32] = literal(value=1, id=2)
  ret add.3: bits[32] = add(x, one, id=3)
}"#;
        let mut p = Parser::new(ir_text);
        let f = p.parse_fn().unwrap();
        let dslx = emit_fn_as_dslx(&f).unwrap();
        let want = r#"fn plus_one(x: u32) -> u32 {
  let one: u32 = u32:1;
  let add_3: u32 = x + one;
  add_3
}"#;
        assert_eq!(dslx, want);
    }

    fn make_ir_with_binop(op: &str) -> String {
        format!(
            "fn f(x: bits[32] id=1) -> bits[32] {{\n  one: bits[32] = literal(value=1, id=2)\n  ret {}.3: bits[32] = {}(x, one, id=3)\n}}",
            op, op
        )
    }

    #[test_case("sub", "x - one", "sub_3"; "sub")]
    #[test_case("umul", "x * one", "umul_3"; "umul")]
    #[test_case("smul", "x * one", "smul_3"; "smul")]
    #[test_case("shll", "x << one", "shll_3"; "shll")]
    #[test_case("shrl", "x >> one", "shrl_3"; "shrl")]
    fn test_binop_emission(op: &str, expr: &str, name: &str) {
        let ir_text = make_ir_with_binop(op);
        let mut p = Parser::new(&ir_text);
        let f = p.parse_fn().unwrap();
        let dslx = emit_fn_as_dslx(&f).unwrap();
        let want = format!(
            "fn f(x: u32) -> u32 {{\n  let one: u32 = u32:1;\n  let {name}: u32 = {expr};\n  {name}\n}}"
        );
        assert_eq!(dslx, want);
    }
}
