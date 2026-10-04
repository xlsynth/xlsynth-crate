// SPDX-License-Identifier: Apache-2.0

//! Recover full-width additions from a one-bit sum and carry decomposition.

use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef};
use xlsynth_pir::ir_match::{MatchCtx, any, bit_slice, commutative_binop};
use xlsynth_pir::ir_utils::{self, Users};

/// A matched concatenation and the full-width operands its addition recovers.
struct RecoveredAdd {
    root: NodeRef,
    lhs: NodeRef,
    rhs: NodeRef,
}

/// Recovers `x + y` from its one-bit low sum and carry into the upper sum.
///
/// For `x, y: u5`, the recognized DSLX expression is:
/// ```text
/// (x[1:5] + y[1:5] + ((x[0:1] & y[0:1]) as u4))
///     ++ (x[0:1] ^ y[0:1])
/// ```
/// Addition is modulo the original width; no extra carry-out bit is retained.
/// Both upper additions and the low XOR must be exclusive to this expression.
/// Replaces only the concatenation payload, leaving cleanup to the caller.
pub fn rewrite_split_adders(f: &mut ir::Fn) -> usize {
    let ctx = MatchCtx::new(f);
    let users = ir_utils::compute_users(f);
    let recovered: Vec<_> = (0..f.nodes.len())
        .filter_map(|index| match_split_adder(&ctx, &users, NodeRef { index }))
        .collect();
    for matched in &recovered {
        f.nodes[matched.root.index].payload =
            NodePayload::Binop(Binop::Add, matched.lhs, matched.rhs);
    }
    recovered.len()
}

/// Matches all three upper-sum groupings without depending on operand order.
fn match_split_adder(ctx: &MatchCtx, users: &Users, root: NodeRef) -> Option<RecoveredAdd> {
    let width = ctx.bits_width(root)?;
    if width < 2 {
        return None;
    }
    let NodePayload::Nary(NaryOp::Concat, parts) = &ctx.f.get_node(root).payload else {
        return None;
    };
    let [high, low] = parts.as_slice() else {
        return None;
    };
    let high_width = width - 1;
    if ctx.bits_width(*high) != Some(high_width)
        || ctx.bits_width(*low) != Some(1)
        || !has_two_slice_operands(ctx, *low)
    {
        return None;
    }
    let low_match = ctx.commutative_pair(
        *low,
        NaryOp::Xor,
        bit_slice(any("x"), 0, 1),
        bit_slice(any("y"), 0, 1),
    )?;
    let lhs = low_match.get_node("x")?;
    let rhs = low_match.get_node("y")?;
    if ctx.bits_width(lhs) != Some(width) || ctx.bits_width(rhs) != Some(width) {
        return None;
    }
    let x_high = bit_slice(lhs, 1, high_width);
    let y_high = bit_slice(rhs, 1, high_width);
    let carry = any("carry");
    let groupings = [
        commutative_binop(
            Binop::Add,
            commutative_binop(Binop::Add, x_high.clone(), y_high.clone()),
            carry.clone(),
        ),
        commutative_binop(
            Binop::Add,
            commutative_binop(Binop::Add, x_high.clone(), carry.clone()),
            y_high.clone(),
        ),
        commutative_binop(
            Binop::Add,
            commutative_binop(Binop::Add, y_high, carry),
            x_high,
        ),
    ];
    let matched_carry = groupings.into_iter().any(|pattern| {
        ctx.matches(*high, pattern)
            .and_then(|bindings| bindings.get_node("carry"))
            .is_some_and(|node| is_low_carry(ctx, node, high_width, lhs, rhs))
    });
    if !matched_carry
        || !exclusively_used_by(ctx.f, users, *high, root)
        || !exclusively_used_by(ctx.f, users, *low, root)
    {
        return None;
    }

    // The successful pattern has exactly one nested addition. Check its
    // users separately so recovering the root does not duplicate a shared sum.
    let inner = ir_utils::operands(&ctx.f.get_node(*high).payload)
        .into_iter()
        .find(|node| {
            matches!(
                ctx.f.get_node(*node).payload,
                NodePayload::Binop(Binop::Add, ..)
            )
        })?;
    if ctx.bits_width(inner) != Some(high_width) || !exclusively_used_by(ctx.f, users, inner, *high)
    {
        return None;
    }
    Some(RecoveredAdd { root, lhs, rhs })
}

/// Recognizes the zero-extended carry of the same low bits used by the XOR.
fn is_low_carry(ctx: &MatchCtx, node: NodeRef, width: usize, lhs: NodeRef, rhs: NodeRef) -> bool {
    if ctx.bits_width(node) != Some(width) {
        return false;
    }
    let carry = match &ctx.f.get_node(node).payload {
        NodePayload::ZeroExt { arg, new_bit_count } if *new_bit_count == width => *arg,
        NodePayload::Nary(NaryOp::Concat, parts) => {
            let [prefix, carry] = parts.as_slice() else {
                return false;
            };
            if ctx.bits_width(*prefix) != Some(width - 1)
                || !matches!(&ctx.f.get_node(*prefix).payload,
                    NodePayload::Literal(value) if value.bits_equals_u64_value(0))
            {
                return false;
            }
            *carry
        }
        NodePayload::Nary(NaryOp::And, _) if width == 1 => node,
        _ => return false,
    };
    ctx.bits_width(carry) == Some(1)
        && has_two_slice_operands(ctx, carry)
        && ctx
            .commutative_pair(
                carry,
                NaryOp::And,
                bit_slice(lhs, 0, 1),
                bit_slice(rhs, 0, 1),
            )
            .is_some()
}

/// Bounds commutative matching to two slices instead of flattening a whole DAG.
fn has_two_slice_operands(ctx: &MatchCtx, node: NodeRef) -> bool {
    matches!(&ctx.f.get_node(node).payload, NodePayload::Nary(_, operands)
        if operands.len() == 2 && operands.iter().all(|operand|
            matches!(ctx.f.get_node(*operand).payload, NodePayload::BitSlice { .. })))
}

/// Includes the function return reference, which is absent from `Users`.
fn exclusively_used_by(f: &ir::Fn, users: &Users, node: NodeRef, user: NodeRef) -> bool {
    f.ret_node_ref != Some(node)
        && users
            .get(&node)
            .is_some_and(|node_users| node_users == [user])
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir::Type;
    use xlsynth_pir::ir_parser::Parser;

    /// Constructs the exact split identity; individual tests vary its graph.
    fn split_ir(width: usize) -> String {
        let high_width = width - 1;
        format!(
            r#"package split
top fn f(x: bits[{width}] id=1, y: bits[{width}] id=2) -> bits[{width}] {{
  xl: bits[1] = bit_slice(x, start=0, width=1, id=3)
  yl: bits[1] = bit_slice(y, start=0, width=1, id=4)
  xh: bits[{high_width}] = bit_slice(x, start=1, width={high_width}, id=5)
  yh: bits[{high_width}] = bit_slice(y, start=1, width={high_width}, id=6)
  low: bits[1] = xor(xl, yl, id=7)
  carry: bits[1] = and(xl, yl, id=8)
  ext: bits[{high_width}] = zero_ext(carry, new_bit_count={high_width}, id=9)
  pair: bits[{high_width}] = add(xh, yh, id=10)
  high: bits[{high_width}] = add(pair, ext, id=11)
  ret result: bits[{width}] = concat(high, low, id=12)
}}
"#,
        )
    }

    /// Parses a standalone fixture and selects its declared top function.
    fn parse_function(text: &str) -> Result<ir::Fn, String> {
        let package = Parser::new(text)
            .parse_and_validate_package()
            .map_err(|error| error.to_string())?;
        package
            .get_top_fn()
            .cloned()
            .ok_or_else(|| "missing top function".to_string())
    }

    #[test]
    fn commutation_reassociation_and_carry_spellings() {
        for (pair, high) in [
            ("xh, yh", "pair, ext"),
            ("xh, ext", "pair, yh"),
            ("yh, ext", "pair, xh"),
        ] {
            for reverse_pair in [false, true] {
                for reverse_high in [false, true] {
                    for reverse_bits in [false, true] {
                        for concat_carry in [false, true] {
                            let mut text = split_ir(5);
                            let pair = if reverse_pair {
                                pair.rsplit(", ").collect::<Vec<_>>().join(", ")
                            } else {
                                pair.to_string()
                            };
                            let high = if reverse_high {
                                high.rsplit(", ").collect::<Vec<_>>().join(", ")
                            } else {
                                high.to_string()
                            };
                            text = text
                                .replace("add(xh, yh,", &format!("add({pair},"))
                                .replace("add(pair, ext,", &format!("add({high},"));
                            if reverse_bits {
                                text = text.replace("xl, yl,", "yl, xl,");
                            }
                            if concat_carry {
                                text = text.replace(
                                    "  ext: bits[4] = zero_ext(carry, new_bit_count=4, id=9)",
                                    "  zero: bits[3] = literal(value=0, id=13)\n  ext: bits[4] = concat(zero, carry, id=9)",
                                );
                            }
                            let mut f = parse_function(&text).unwrap();
                            let result = f.ret_node_ref.unwrap();
                            let old_nodes = f.nodes.clone();
                            assert_eq!(rewrite_split_adders(&mut f), 1, "{text}");
                            assert!(
                                MatchCtx::new(&f)
                                    .commutative_binop_pair(
                                        result,
                                        Binop::Add,
                                        f.params[0],
                                        f.params[1]
                                    )
                                    .is_some()
                            );
                            for (index, old) in old_nodes.iter().enumerate() {
                                if index != result.index {
                                    assert_eq!(f.nodes[index].payload, old.payload);
                                }
                            }
                            assert_eq!(rewrite_split_adders(&mut f), 0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn width_boundaries_and_repeated_operand() {
        for width in [2, 3, 65, 129] {
            let mut f = parse_function(&split_ir(width)).unwrap();
            assert_eq!(rewrite_split_adders(&mut f), 1);
        }
        for (literal, expected) in [("0", 1), ("0x10000000000000000", 0)] {
            let text = split_ir(129).replace(
                "  ext: bits[128] = zero_ext(carry, new_bit_count=128, id=9)",
                &format!("  prefix: bits[127] = literal(value={literal}, id=13)\n  ext: bits[128] = concat(prefix, carry, id=9)"),
            );
            let mut f = parse_function(&text).unwrap();
            assert_eq!(rewrite_split_adders(&mut f), expected);
        }
        let direct_carry = split_ir(2)
            .replace(
                "  ext: bits[1] = zero_ext(carry, new_bit_count=1, id=9)\n",
                "",
            )
            .replace("add(pair, ext,", "add(pair, carry,");
        let mut f = parse_function(&direct_carry).unwrap();
        assert_eq!(rewrite_split_adders(&mut f), 1);
        let mut f = parse_function(&split_ir(5).replace("bit_slice(y,", "bit_slice(x,")).unwrap();
        assert_eq!(rewrite_split_adders(&mut f), 1);
        assert_eq!(
            f.get_node(f.ret_node_ref.unwrap()).payload,
            NodePayload::Binop(Binop::Add, f.params[0], f.params[0])
        );
    }

    #[test]
    fn rejects_mismatched_or_additional_terms() {
        for text in [
            split_ir(5).replace("and(xl, yl,", "and(xl, xl,"),
            split_ir(5).replace("xor(xl, yl,", "xor(xl, xl,"),
            split_ir(5).replace("and(xl, yl,", "and(xl, yl, xl,"),
            split_ir(5).replace("xor(xl, yl,", "xor(xl, yl, xl,"),
            split_ir(5).replace("zero_ext(carry,", "sign_ext(carry,"),
            split_ir(5).replace("add(pair, ext,", "add(pair, xh,"),
            split_ir(5).replace("bit_slice(x, start=0,", "bit_slice(x, start=1,"),
            split_ir(5).replace("bit_slice(x, start=1,", "bit_slice(x, start=0,"),
            split_ir(5).replace("x: bits[5]", "x: bits[6]"),
            split_ir(5).replace(
                "  high: bits[4] = add(pair, ext, id=11)",
                "  extra: bits[4] = add(pair, xh, id=13)\n  high: bits[4] = add(extra, ext, id=11)",
            ),
            split_ir(5).replace(
                "  ext: bits[4] = zero_ext(carry, new_bit_count=4, id=9)",
                "  nonzero: bits[3] = literal(value=1, id=13)\n  ext: bits[4] = concat(nonzero, carry, id=9)",
            ),
        ] {
            let mut f = parse_function(&text).unwrap();
            let before = f.to_string();
            assert_eq!(rewrite_split_adders(&mut f), 0, "{text}");
            assert_eq!(f.to_string(), before);
        }
    }

    #[test]
    fn rejects_large_shared_boolean_dags_without_flattening() {
        for op in ["xor", "and"] {
            let mut nodes = String::new();
            let mut previous = "xl".to_string();
            for index in 0..64 {
                let name = format!("chain_{index}");
                nodes.push_str(&format!(
                    "  {name}: bits[1] = {op}({previous}, {previous}, id={})\n",
                    13 + index
                ));
                previous = name;
            }
            let text = split_ir(5)
                .replace("  low:", &format!("{nodes}  low:"))
                .replace(
                    &format!("{op}(xl, yl,"),
                    &format!("{op}({previous}, {previous},"),
                );
            let mut f = parse_function(&text).unwrap();
            assert_eq!(rewrite_split_adders(&mut f), 0);
        }
    }

    #[test]
    fn shared_upper_sums_or_low_sum_are_retained() {
        for (name, width, expected) in [
            ("pair", 4, 0),
            ("high", 4, 0),
            ("low", 1, 0),
            ("x", 5, 1),
            ("y", 5, 1),
            ("carry", 1, 1),
        ] {
            let text = split_ir(5)
                .replace("-> bits[5] {", &format!("-> (bits[5], bits[{width}]) {{"))
                .replace("ret result:", "result:")
                .replace(
                    "\n}",
                    &format!(
                        "\n  ret both: (bits[5], bits[{width}]) = tuple(result, {name}, id=13)\n}}"
                    ),
                );
            let mut f = parse_function(&text).unwrap();
            let returned = f.get_node(f.ret_node_ref.unwrap()).payload.clone();
            assert_eq!(rewrite_split_adders(&mut f), expected, "{text}");
            assert_eq!(f.get_node(f.ret_node_ref.unwrap()).payload, returned);
        }
    }

    #[test]
    fn direct_return_counts_as_an_external_user() {
        for name in ["pair", "high", "low"] {
            let mut f = parse_function(&split_ir(5)).unwrap();
            let index = f
                .nodes
                .iter()
                .position(|node| node.name.as_deref() == Some(name))
                .unwrap();
            f.ret_node_ref = Some(NodeRef { index });
            f.ret_ty = f.nodes[index].ty.clone();
            assert_eq!(rewrite_split_adders(&mut f), 0);
        }
    }

    #[test]
    fn rejects_an_extra_carry_out_bit() {
        let text = split_ir(6)
            .replace("x: bits[6]", "x: bits[5]")
            .replace("y: bits[6]", "y: bits[5]")
            .replace("xh: bits[5] = bit_slice(x, start=1, width=5, id=5)",
                "xs: bits[4] = bit_slice(x, start=1, width=4, id=5)\n  xh: bits[5] = zero_ext(xs, new_bit_count=5, id=13)")
            .replace("yh: bits[5] = bit_slice(y, start=1, width=5, id=6)",
                "ys: bits[4] = bit_slice(y, start=1, width=4, id=6)\n  yh: bits[5] = zero_ext(ys, new_bit_count=5, id=14)");
        let mut f = parse_function(&text).unwrap();
        assert_eq!(f.ret_ty, Type::Bits(6));
        assert_eq!(rewrite_split_adders(&mut f), 0);
    }
}
