// SPDX-License-Identifier: Apache-2.0
//! Enumerate the value of a shared selection DAG on a complete, bounded control
//! domain.

use super::Budget;
use std::collections::{HashMap, HashSet};
use xlsynth_pir::ir::{self, NodePayload, NodeRef, Type};

pub(super) struct Domain {
    pub axes: Vec<(NodeRef, usize)>,
    pub x: NodeRef,
    pub y: NodeRef,
    pub constant: Vec<bool>,
    pub take_x: u64,
    pub take_y: u64,
}

impl Domain {
    pub fn analyze(f: &ir::Fn, root: NodeRef, budget: &mut Budget) -> Option<Self> {
        let mut axes = Vec::new();
        let mut leaves = Vec::new();
        let mut pending = vec![root];
        let mut seen = HashSet::new();
        while let Some(n) = pending.pop() {
            if !seen.insert(n) {
                continue;
            }
            if !budget.take(1) {
                return None;
            }
            if let Some((selector, arms, default, _)) = selection(f, n) {
                let Type::Bits(width) = f.get_node(selector).ty else {
                    return None;
                };
                for bit in 0..width {
                    if !axes.contains(&(selector, bit)) {
                        if axes.len() == 6 {
                            return None;
                        }
                        axes.push((selector, bit));
                    }
                }
                pending.extend(arms.iter().copied());
                pending.extend(default);
            } else {
                leaves.push(n);
            }
        }
        let mut data = Vec::new();
        let mut constant = None;
        for leaf in leaves {
            if let NodePayload::Literal(v) = &f.get_node(leaf).payload {
                let bits = v.to_bits().ok()?;
                let value = (0..bits.get_bit_count())
                    .map(|i| bits.get_bit(i).ok())
                    .collect::<Option<Vec<_>>>()?;
                if constant.as_ref().is_some_and(|c| c != &value) {
                    return None;
                }
                constant = Some(value);
            } else {
                data.push(leaf);
            }
        }
        let [x, y] = data[..] else { return None };
        let constant = constant?;
        let assignments = 1usize << axes.len();
        if !budget.take(seen.len().saturating_mul(assignments)) {
            return None;
        }
        let columns: HashMap<_, _> = axes
            .iter()
            .enumerate()
            .map(|(i, &axis)| (axis, i))
            .collect();
        let mut take_x = 0;
        let mut take_y = 0;
        for row in 0..assignments {
            let mut n = root;
            while let Some((s, arms, default, priority)) = selection(f, n) {
                let Type::Bits(width) = f.get_node(s).ty else {
                    return None;
                };
                let mut code = 0usize;
                for i in 0..width {
                    code |= ((row >> columns[&(s, i)]) & 1) << i;
                }
                n = if priority {
                    if code == 0 {
                        default?
                    } else {
                        *arms.get(code.trailing_zeros() as usize)?
                    }
                } else {
                    arms.get(code).copied().or(default)?
                };
            }
            if n == x {
                take_x |= 1 << row;
            }
            if n == y {
                take_y |= 1 << row;
            }
        }
        if take_x == 0 || take_y == 0 {
            return None;
        }
        Some(Self {
            axes,
            x,
            y,
            constant,
            take_x,
            take_y,
        })
    }
}

fn selection(f: &ir::Fn, n: NodeRef) -> Option<(NodeRef, &[NodeRef], Option<NodeRef>, bool)> {
    match &f.get_node(n).payload {
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => Some((*selector, cases, *default, false)),
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => Some((*selector, cases, *default, true)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn priority_and_shared_nested_choices_preserve_control_order() {
        let pkg = Parser::new(
            "package p
top fn f(x: bits[7] id=1, y: bits[7] id=2, g: bits[2] id=3, s: bits[1] id=4) -> bits[7] {
  k: bits[7] = literal(value=45, id=5)
  z: bits[7] = priority_sel(g, cases=[x, k], default=y, id=6)
  arm: bits[7] = sel(s, cases=[z, y], id=7)
  ret final_word: bits[7] = sel(s, cases=[z, arm], id=8)
}",
        )
        .parse_and_validate_package()
        .unwrap();
        let f = pkg.get_top_fn().unwrap();
        let d = Domain::analyze(f, f.ret_node_ref.unwrap(), &mut Budget::default()).unwrap();
        assert_eq!(
            d.constant,
            vec![true, false, true, true, false, true, false]
        );
        assert_eq!(d.axes.len(), 3);
        for g in 0..4usize {
            for s in 0..2usize {
                let mut row = 0;
                for (i, (n, b)) in d.axes.iter().enumerate() {
                    let v = match f.get_node(*n).text_id {
                        3 => g,
                        4 => s,
                        _ => unreachable!(),
                    };
                    row |= ((v >> b) & 1) << i;
                }
                let actual = if d.take_x >> row & 1 != 0 {
                    f.get_node(d.x).text_id
                } else if d.take_y >> row & 1 != 0 {
                    f.get_node(d.y).text_id
                } else {
                    5
                };
                let expected = if s != 0 {
                    2
                } else if g & 1 != 0 {
                    1
                } else if g & 2 != 0 {
                    5
                } else {
                    2
                };
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn fourth_value_keeps_the_incumbent_word() {
        let pkg = Parser::new(
            "package p
top fn f(x: bits[4] id=1, y: bits[4] id=2, z: bits[4] id=3, s: bits[2] id=4) -> bits[4] {
  k: bits[4] = literal(value=10, id=5)
  ret mux: bits[4] = sel(s, cases=[x, k, y, z], id=6)
}",
        )
        .parse_and_validate_package()
        .unwrap();
        let f = pkg.get_top_fn().unwrap();
        assert!(Domain::analyze(f, f.ret_node_ref.unwrap(), &mut Budget::default()).is_none());
    }
}
