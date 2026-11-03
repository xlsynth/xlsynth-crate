// SPDX-License-Identifier: Apache-2.0

//! Conservative bit demand for one immutable lowering boundary.

use std::collections::HashMap;

use crate::gatify::wiring;
use xlsynth_pir::{ir, ir_utils};

/// Unions demanded flat bits across consumers. Only the supplied dependency-
/// ordered nodes are traversed; operands outside that boundary remain opaque.
/// Unsupported operations demand all operand bits, without value reasoning.
pub(in crate::gatify) fn analyze(
    f: &ir::Fn,
    nodes: &[ir::NodeRef],
    outputs: &[ir::NodeRef],
) -> HashMap<ir::NodeRef, Vec<bool>> {
    let mut demand = HashMap::new();
    for &output in outputs {
        demand.insert(output, vec![true; f.get_node_ty(output).bit_count()]);
    }
    for &node_ref in nodes.iter().rev() {
        let node = f.get_node(node_ref);
        let mask = demand.get(&node_ref).cloned().unwrap_or_default();
        // Effects may have zero result bits. Conservatively retain their
        // operands even though gate lowering itself models value outputs.
        let effect = ir_utils::is_observable_effect_root(&node.payload);
        if effect {
            for operand in ir_utils::operands(&node.payload) {
                demand.insert(operand, vec![true; f.get_node_ty(operand).bit_count()]);
            }
            continue;
        }
        if !mask.iter().any(|bit| *bit) {
            continue;
        }
        let mut propagate = |operand, offset: usize, bits: &[bool]| {
            let target = demand
                .entry(operand)
                .or_insert_with(|| vec![false; f.get_node_ty(operand).bit_count()]);
            for (target, &needed) in target[offset..offset + bits.len()].iter_mut().zip(bits) {
                *target |= needed;
            }
        };
        if let Some((arg, start)) = wiring::slice_source(f, &node.payload) {
            propagate(arg, start, &mask);
            continue;
        }
        if let Some(parts) = wiring::concat_sources(f, &node.payload) {
            for (arg, range) in parts {
                propagate(arg, 0, &mask[range]);
            }
            continue;
        }
        match &node.payload {
            ir::NodePayload::Unop(ir::Unop::Reverse, arg) => {
                let reversed = mask.iter().rev().copied().collect::<Vec<_>>();
                propagate(*arg, 0, &reversed);
            }
            ir::NodePayload::ZeroExt { arg, .. } | ir::NodePayload::SignExt { arg, .. } => {
                let width = f.get_node_ty(*arg).bit_count();
                propagate(*arg, 0, &mask[..width]);
                if matches!(node.payload, ir::NodePayload::SignExt { .. })
                    && width > 0
                    && mask[width..].iter().any(|bit| *bit)
                {
                    propagate(*arg, width - 1, &[true]);
                }
            }
            ir::NodePayload::Array(args) | ir::NodePayload::ArrayConcat(args) => {
                let mut offset = 0;
                for &arg in args {
                    let width = f.get_node_ty(arg).bit_count();
                    propagate(arg, 0, &mask[offset..offset + width]);
                    offset += width;
                }
            }
            _ => {
                // Arithmetic, dynamic indexing, selects, and shifts are opaque
                // here. Shift-stage dependencies are handled by the emitter.
                for operand in ir_utils::operands(&node.payload) {
                    propagate(operand, 0, &vec![true; f.get_node_ty(operand).bit_count()]);
                }
            }
        }
    }
    demand
}

#[cfg(test)]
mod tests {
    use super::analyze;
    use xlsynth_pir::{ir, ir_parser::Parser, ir_utils};

    #[test]
    fn demanded_bits_union_wiring_and_opaque_consumers() {
        let f = Parser::new(
            r#"fn f(x: bits[29] id=1, amount: bits[5] id=2) -> bits[5] {
  shift: bits[29] = shll(x, amount, id=3)
  low: bits[2] = bit_slice(shift, start=1, width=2, id=4)
  high: bits[3] = bit_slice(shift, start=25, width=3, id=5)
  ret joined: bits[5] = concat(high, low, id=6)
}"#,
        )
        .parse_fn()
        .unwrap();
        let topo = ir_utils::get_topological(&f);
        let masks = analyze(&f, &topo, &[f.ret_node_ref.unwrap()]);
        let shift = topo.iter().find(|r| f.get_node(**r).text_id == 3).unwrap();
        let mut expected = vec![false; 29];
        for bit in [1, 2, 25, 26, 27] {
            expected[bit] = true;
        }
        assert_eq!(masks[shift], expected);
        assert_eq!(masks[&f.params[0]], vec![true; 29]);
        let full = analyze(&f, &topo, &[f.ret_node_ref.unwrap(), *shift]);
        assert_eq!(full[shift], vec![true; 29]);
        // Region roots differ from the containing return, and do not traverse
        // the independently supplied shift input.
        let region = analyze(&f, &[*shift], &[*shift, *shift]);
        assert_eq!(region[shift], vec![true; 29]);
        assert!(!region.contains_key(&f.ret_node_ref.unwrap()));
        assert!(matches!(
            f.get_node(*shift).payload,
            ir::NodePayload::Binop(..)
        ));
    }

    #[test]
    fn demanded_bits_extensions_and_aggregate_layouts() {
        let f = Parser::new(
            r#"fn f(x: bits[3] id=1, y: bits[3] id=2) -> bits[2] {
  signed: bits[5] = sign_ext(x, new_bit_count=5, id=3)
  zeroed: bits[5] = zero_ext(y, new_bit_count=5, id=4)
  tup: (bits[5], bits[5]) = tuple(signed, zeroed, id=5)
  member: bits[5] = tuple_index(tup, index=0, id=6)
  ret high: bits[2] = bit_slice(member, start=3, width=2, id=7)
}"#,
        )
        .parse_fn()
        .unwrap();
        let masks = analyze(
            &f,
            &ir_utils::get_topological(&f),
            &[f.ret_node_ref.unwrap()],
        );
        assert_eq!(masks[&f.params[0]], vec![false, false, true]);
        assert!(!masks.contains_key(&f.params[1]));
    }

    #[test]
    fn nested_tuple_fields_and_repeated_operands_union_their_flat_bits() {
        let f = Parser::new(
            r#"fn f(x: bits[5] id=1, y: bits[3] id=2, empty: () id=3) -> bits[4] {
  inner: (bits[3], (), bits[5]) = tuple(y, empty, x, id=4)
  outer: (bits[5], (bits[3], (), bits[5]), (), bits[5]) = tuple(x, inner, empty, x, id=5)
  head: bits[5] = tuple_index(outer, index=0, id=6)
  middle: (bits[3], (), bits[5]) = tuple_index(outer, index=1, id=7)
  tail: bits[5] = tuple_index(outer, index=3, id=8)
  a: bits[3] = tuple_index(middle, index=0, id=9)
  b: bits[5] = tuple_index(middle, index=2, id=10)
  inverted: bits[5] = not(b, id=11)
  copied: bits[5] = identity(tail, id=12)
  p: bits[1] = bit_slice(head, start=4, width=1, id=13)
  q: bits[1] = bit_slice(a, start=2, width=1, id=14)
  r: bits[1] = bit_slice(inverted, start=2, width=1, id=15)
  s: bits[1] = bit_slice(copied, start=0, width=1, id=16)
  ret out: bits[4] = concat(p, q, r, s, id=17)
}"#,
        )
        .parse_fn()
        .unwrap();
        let topo = ir_utils::get_topological(&f);
        let masks = analyze(&f, &topo, &[f.ret_node_ref.unwrap()]);
        assert_eq!(masks[&f.params[0]], [true, false, true, false, true]);
        assert_eq!(masks[&f.params[1]], [false, false, true]);
        assert!(masks[&f.params[2]].is_empty());

        let outer = topo.iter().find(|r| f.get_node(**r).text_id == 5).unwrap();
        let expected: Vec<_> = (0..18).map(|i| [0, 7, 12, 17].contains(&i)).collect();
        assert_eq!(masks[outer], expected);
    }
}
