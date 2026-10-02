// SPDX-License-Identifier: Apache-2.0

//! A bounded profitability estimate for PIR bitwise regions. This is not a
//! technology mapper: muxes remain abstract operations, and no gate cleanup or
//! fanout-dependent delay analysis is performed.

use std::collections::HashMap;

use crate::dce::get_dead_nodes;
use crate::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Unop};
use crate::ir_utils::{get_topological, is_observable_effect_root, operands};
use crate::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;

/// Bounds both retained bit expressions and work spent folding expressions.
const MAX_BIT_WORK: usize = 16_384;

/// Estimated cost of the supported bitwise regions of a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalCost {
    /// AND/OR cost one; XOR and full two-input muxes cost three.
    pub and_equivalents: usize,
    /// Maximum logic/mux stages between region boundaries, not AIG depth.
    pub logic_depth: usize,
}

impl LocalCost {
    /// Requires a strict improvement without trading either estimate away.
    pub fn is_pareto_improvement_on(self, other: Self) -> bool {
        self.and_equivalents <= other.and_equivalents
            && self.logic_depth <= other.logic_depth
            && self != other
    }
}

/// Estimates live bitwise logic, retaining sharing, constants, and free wiring.
///
/// Other operations form region boundaries: their operands remain costed roots,
/// and their output bits become independent inputs. Their own area and delay
/// are omitted. Compare alternatives that preserve those operations; this is
/// a heuristic, not a guarantee about final mapped area or Graph LE. Returns
/// `None` if the work budget is exhausted, so callers can retain the incumbent.
pub fn estimate_local_cost(f: &ir::Fn) -> Option<LocalCost> {
    estimate_with_budget(f, MAX_BIT_WORK)
}

/// A bit expression plus a free inversion; zero and one are 0 and 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Bit(usize);

impl Bit {
    fn inverted(self) -> Self {
        Self(self.0 ^ 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Expr {
    Constant,
    Input(NodeRef, usize),
    And(Bit, Bit),
    Xor(Bit, Bit),
    Mux(Bit, Bit, Bit),
}

impl Expr {
    fn inputs(self) -> Vec<Bit> {
        match self {
            Self::Constant | Self::Input(..) => vec![],
            Self::And(a, b) | Self::Xor(a, b) => vec![a, b],
            Self::Mux(s, a, b) => vec![s, a, b],
        }
    }
}

struct Estimator {
    remaining: usize,
    expressions: Vec<Expr>,
    interned: HashMap<Expr, Bit>,
    depths: Vec<usize>,
}

impl Estimator {
    fn spend(&mut self, work: usize) -> Option<()> {
        self.remaining = self.remaining.checked_sub(work)?;
        Some(())
    }

    /// Shares equal expressions without expanding muxes into an AIG.
    fn intern(&mut self, expr: Expr) -> Option<Bit> {
        self.spend(1)?;
        if let Some(bit) = self.interned.get(&expr) {
            return Some(*bit);
        }
        let inputs = expr.inputs();
        let depth = inputs.iter().map(|b| self.depths[b.0 / 2]).max();
        let bit = Bit(self.expressions.len() * 2);
        self.expressions.push(expr);
        self.depths.push(depth.map_or(0, |d| d + 1));
        self.interned.insert(expr, bit);
        Some(bit)
    }

    fn and(&mut self, a: Bit, b: Bit) -> Option<Bit> {
        self.spend(1)?;
        if a == Bit(0) || b == Bit(0) || a == b.inverted() {
            Some(Bit(0))
        } else if a == Bit(1) || a == b {
            Some(b)
        } else if b == Bit(1) {
            Some(a)
        } else {
            self.intern(Expr::And(a.min(b), a.max(b)))
        }
    }

    fn xor(&mut self, a: Bit, b: Bit) -> Option<Bit> {
        self.spend(1)?;
        // Normalize polarity, including XOR with either constant.
        let inverted = (a.0 ^ b.0) & 1;
        let a = Bit(a.0 & !1);
        let b = Bit(b.0 & !1);
        let result = if a == b {
            Bit(0)
        } else if a == Bit(0) {
            b
        } else if b == Bit(0) {
            a
        } else {
            self.intern(Expr::Xor(a.min(b), a.max(b)))?
        };
        Some(Bit(result.0 ^ inverted))
    }

    fn mux(&mut self, selector: Bit, off: Bit, on: Bit) -> Option<Bit> {
        self.spend(1)?;
        if selector.0 & 1 != 0 {
            return self.mux(selector.inverted(), on, off);
        }
        if selector == Bit(0) || off == on {
            Some(off)
        } else if off == Bit(0) || off == selector {
            self.and(selector, on)
        } else if on == Bit(0) || on == selector.inverted() {
            self.and(selector.inverted(), off)
        } else if off == Bit(1) || off == selector.inverted() {
            self.and(selector, on.inverted()).map(Bit::inverted)
        } else if on == Bit(1) || on == selector {
            self.and(selector.inverted(), off.inverted())
                .map(Bit::inverted)
        } else if off == on.inverted() {
            self.xor(selector, off)
        } else {
            self.intern(Expr::Mux(selector, off, on))
        }
    }

    /// Uses balanced bitwise reductions, with inversion represented on edges.
    fn reduce(&mut self, op: NaryOp, mut values: Vec<Bit>) -> Option<Bit> {
        let inverted_inputs = matches!(op, NaryOp::Or | NaryOp::Nor);
        if inverted_inputs {
            values.iter_mut().for_each(|b| *b = b.inverted());
        }
        while values.len() > 1 {
            values = values
                .chunks(2)
                .map(|pair| match pair {
                    [a, b] if op == NaryOp::Xor => self.xor(*a, *b),
                    [a, b] => self.and(*a, *b),
                    [a] => Some(*a),
                    _ => unreachable!("chunks contain one or two bits"),
                })
                .collect::<Option<_>>()?;
        }
        let result =
            values
                .first()
                .copied()
                .unwrap_or(if op == NaryOp::Xor { Bit(0) } else { Bit(1) });
        Some(if matches!(op, NaryOp::Or | NaryOp::Nand) {
            result.inverted()
        } else {
            result
        })
    }

    /// Models ordinary barrel stages and one combined oversized-shift guard.
    fn shift(&mut self, mut data: Vec<Bit>, amount: &[Bit], op: Binop) -> Option<Vec<Bit>> {
        let width = data.len();
        if width == 0 {
            return Some(data);
        }
        let stages = (usize::BITS - (width - 1).leading_zeros()) as usize;
        for (i, control) in amount.iter().take(stages).enumerate() {
            let distance = 1usize << i;
            data = (0..width)
                .map(|bit| {
                    let source = if op == Binop::Shll {
                        bit.checked_sub(distance)
                    } else {
                        bit.checked_add(distance).filter(|n| *n < width)
                    };
                    self.mux(*control, data[bit], source.map_or(Bit(0), |n| data[n]))
                })
                .collect::<Option<_>>()?;
        }
        if amount.len() > stages {
            let oversized = self.reduce(NaryOp::Or, amount[stages..].to_vec())?;
            for bit in &mut data {
                *bit = self.mux(oversized, *bit, Bit(0))?;
            }
        }
        Some(data)
    }

    /// Estimates indexed selects as mux trees, preserving out-of-range
    /// defaults.
    fn select(
        &mut self,
        selector: &[Bit],
        mut cases: Vec<Vec<Bit>>,
        default: Vec<Bit>,
        priority: bool,
    ) -> Option<Vec<Bit>> {
        if cases.is_empty() {
            return Some(default);
        }
        if priority {
            let mut result = default;
            for (control, case) in selector.iter().zip(cases).rev() {
                for (bit, on) in result.iter_mut().zip(case) {
                    *bit = self.mux(*control, *bit, on)?;
                }
            }
            return Some(result);
        }
        for control in selector {
            if cases.len() % 2 != 0 {
                cases.push(default.clone());
            }
            cases = cases
                .chunks(2)
                .map(|pair| {
                    pair[0]
                        .iter()
                        .zip(&pair[1])
                        .map(|(off, on)| self.mux(*control, *off, *on))
                        .collect::<Option<_>>()
                })
                .collect::<Option<_>>()?;
        }
        cases.into_iter().next()
    }

    /// Counts only expressions that survive constant folding and output
    /// slicing.
    fn cost(&self, mut roots: Vec<Bit>) -> LocalCost {
        let logic_depth = roots
            .iter()
            .map(|b| self.depths[b.0 / 2])
            .max()
            .unwrap_or(0);
        let mut seen = vec![false; self.expressions.len()];
        let mut and_equivalents = 0;
        while let Some(bit) = roots.pop() {
            let index = bit.0 / 2;
            if seen[index] {
                continue;
            }
            seen[index] = true;
            let expr = self.expressions[index];
            and_equivalents += match expr {
                Expr::Constant | Expr::Input(..) => 0,
                Expr::And(..) => 1,
                Expr::Xor(..) | Expr::Mux(..) => 3,
            };
            roots.extend(expr.inputs());
        }
        LocalCost {
            and_equivalents,
            logic_depth,
        }
    }
}

/// Keeps all boundary operands as roots, so unsupported users cannot hide logic
/// that remains live after a rewrite (notably shared shift amounts).
fn estimate_with_budget(f: &ir::Fn, budget: usize) -> Option<LocalCost> {
    let mut estimator = Estimator {
        remaining: budget,
        expressions: vec![Expr::Constant],
        interned: HashMap::new(),
        depths: vec![0],
    };
    estimator.spend(f.nodes.len())?;
    let mut live = vec![true; f.nodes.len()];
    for nr in get_dead_nodes(f) {
        live[nr.index] = false;
    }
    let mut bits: Vec<Vec<Bit>> = vec![vec![]; f.nodes.len()];
    let mut roots = Vec::new();
    for nr in get_topological(f) {
        if !live[nr.index] {
            continue;
        }
        let node = f.get_node(nr);
        let width = node.ty.checked_bit_count()?;
        estimator.spend(width)?;
        let args = operands(&node.payload);
        estimator.spend(args.len())?;
        for arg in &args {
            estimator.spend(bits[arg.index].len())?;
        }
        bits[nr.index] = match &node.payload {
            NodePayload::Literal(value) => {
                let mut flat = Vec::new();
                flatten_ir_value_to_lsb0_bits_for_type(value, &node.ty, &mut flat).ok()?;
                flat.into_iter().map(|b| Bit(usize::from(b))).collect()
            }
            NodePayload::Unop(Unop::Identity, arg) => bits[arg.index].clone(),
            NodePayload::Unop(Unop::Not, arg) => {
                bits[arg.index].iter().map(|b| b.inverted()).collect()
            }
            NodePayload::Unop(Unop::Reverse, arg) => {
                bits[arg.index].iter().rev().copied().collect()
            }
            NodePayload::Nary(NaryOp::Concat, args) | NodePayload::Tuple(args) => args
                .iter()
                .rev()
                .flat_map(|arg| bits[arg.index].iter().copied())
                .collect(),
            NodePayload::Array(args) | NodePayload::ArrayConcat(args) => args
                .iter()
                .flat_map(|arg| bits[arg.index].iter().copied())
                .collect(),
            NodePayload::TupleIndex { tuple, index } => {
                let slice = f
                    .get_node_ty(*tuple)
                    .tuple_get_flat_bit_slice_for_index(*index)
                    .ok()?;
                bits[tuple.index][slice.start..slice.limit].to_vec()
            }
            NodePayload::BitSlice { arg, start, width } => {
                bits[arg.index][*start..start + width].to_vec()
            }
            NodePayload::ZeroExt { arg, .. } | NodePayload::SignExt { arg, .. } => {
                let mut value = bits[arg.index].clone();
                let fill = if matches!(node.payload, NodePayload::SignExt { .. }) {
                    value.last().copied().unwrap_or(Bit(0))
                } else {
                    Bit(0)
                };
                value.resize(width, fill);
                value
            }
            NodePayload::Nary(op, args) => (0..width)
                .map(|i| estimator.reduce(*op, args.iter().map(|a| bits[a.index][i]).collect()))
                .collect::<Option<_>>()?,
            NodePayload::Unop(op @ (Unop::AndReduce | Unop::OrReduce | Unop::XorReduce), arg) => {
                let op = match op {
                    Unop::AndReduce => NaryOp::And,
                    Unop::OrReduce => NaryOp::Or,
                    _ => NaryOp::Xor,
                };
                vec![estimator.reduce(op, bits[arg.index].clone())?]
            }
            NodePayload::Binop(op @ (Binop::Shll | Binop::Shrl), data, amount) => {
                estimator.shift(bits[data.index].clone(), &bits[amount.index], *op)?
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            }
            | NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => estimator.select(
                &bits[selector.index],
                cases.iter().map(|c| bits[c.index].clone()).collect(),
                default.map_or_else(|| vec![Bit(0); width], |d| bits[d.index].clone()),
                matches!(node.payload, NodePayload::PrioritySel { .. }),
            )?,
            _ => {
                // Arithmetic, effects, and other unmodeled operations delimit
                // regions. Their inputs still contribute to the total cost.
                for arg in &args {
                    roots.extend_from_slice(&bits[arg.index]);
                }
                (0..width)
                    .map(|i| estimator.intern(Expr::Input(nr, i)))
                    .collect::<Option<_>>()?
            }
        };
        if is_observable_effect_root(&node.payload) {
            roots.extend_from_slice(&bits[nr.index]);
        }
    }
    roots.extend_from_slice(&bits[f.ret_node_ref?.index]);
    Some(estimator.cost(roots))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    fn parse(text: &str) -> ir::Fn {
        Parser::new(text)
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone()
    }

    #[test]
    fn sharing_constants_and_projection_remove_unused_logic() {
        let f = parse(
            r#"package test
top fn main(x: bits[4] id=1, p: bits[1] id=2) -> (bits[2], bits[2]) {
  zero: bits[4] = literal(value=0, id=3)
  selected: bits[4] = sel(p, cases=[zero, x], id=4)
  sliced: bits[2] = bit_slice(selected, start=0, width=2, id=5)
  ret out: (bits[2], bits[2]) = tuple(sliced, sliced, id=6)
}"#,
        );
        assert_eq!(
            estimate_local_cost(&f),
            Some(LocalCost {
                and_equivalents: 2,
                logic_depth: 1
            })
        );
        assert_eq!(estimate_with_budget(&f, 5), None);
    }

    #[test]
    fn unknown_consumer_retains_input_logic_as_a_region_root() {
        let f = parse(
            r#"package test
top fn main(x: bits[4] id=1, y: bits[4] id=2, p: bits[1] id=3) -> bits[4] {
  selected: bits[4] = sel(p, cases=[x, y], id=4)
  ret out: bits[4] = add(selected, x, id=5)
}"#,
        );
        assert_eq!(
            estimate_local_cost(&f),
            Some(LocalCost {
                and_equivalents: 12,
                logic_depth: 1
            })
        );
    }

    #[test]
    fn wide_overshifts_fold_without_host_integer_conversion() {
        let f = parse(
            r#"package test
top fn main(x: bits[4] id=1, p: bits[1] id=2) -> bits[4] {
  zero: bits[65] = literal(value=0, id=3)
  huge: bits[65] = literal(value=18446744073709551616, id=4)
  amount: bits[65] = sel(p, cases=[zero, huge], id=5)
  ret out: bits[4] = shrl(x, amount, id=6)
}"#,
        );
        assert_eq!(
            estimate_local_cost(&f),
            Some(LocalCost {
                and_equivalents: 4,
                logic_depth: 1
            })
        );
    }

    #[test]
    fn default_only_select_preserves_the_cost_of_its_default() {
        let f = parse(
            r#"package test
top fn main(x: bits[2] id=1, y: bits[2] id=2, s: bits[3] id=3) -> bits[2] {
  masked: bits[2] = and(x, y, id=4)
  ret out: bits[2] = sel(s, cases=[], default=masked, id=5)
}"#,
        );
        assert_eq!(
            estimate_local_cost(&f),
            Some(LocalCost {
                and_equivalents: 2,
                logic_depth: 1
            })
        );
    }

    #[test]
    fn repeated_wide_cases_exhaust_the_budget_before_expansion() {
        let cases = vec!["x"; 1_024].join(", ");
        let f = parse(&format!(
            r#"package test
top fn main(x: bits[1024] id=1, s: bits[10] id=2) -> bits[1024] {{
  ret out: bits[1024] = sel(s, cases=[{cases}], id=3)
}}"#,
        ));
        assert_eq!(estimate_local_cost(&f), None);
    }
}
