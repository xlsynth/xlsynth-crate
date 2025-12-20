// SPDX-License-Identifier: Apache-2.0

//! Compact AIG witness for a 4-input single-output Boolean function.

use serde::{Deserialize, Serialize};

use crate::aig::{AigBitVector, GateFn};
use crate::cut_db::npn::NpnTransform;
use crate::cut_db::tt16::TruthTable16;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

pub const INPUT_COUNT: u16 = 4;
pub const CONST0_ID: u16 = INPUT_COUNT;
pub const FIRST_NODE_ID: u16 = CONST0_ID + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Lit {
    pub id: u16,
    pub negated: bool,
}

impl Lit {
    pub const fn new(id: u16, negated: bool) -> Self {
        Self { id, negated }
    }

    pub const fn negate(self) -> Self {
        Self {
            id: self.id,
            negated: !self.negated,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FragmentNode {
    And2 { a: Lit, b: Lit },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GateFnFragment {
    pub nodes: Vec<FragmentNode>,
    pub output: Lit,
}

impl GateFnFragment {
    pub fn const0() -> Self {
        Self {
            nodes: Vec::new(),
            output: Lit::new(CONST0_ID, false),
        }
    }

    pub fn const1() -> Self {
        Self {
            nodes: Vec::new(),
            output: Lit::new(CONST0_ID, true),
        }
    }

    pub fn input(i: u16) -> Self {
        assert!(i < INPUT_COUNT);
        Self {
            nodes: Vec::new(),
            output: Lit::new(i, false),
        }
    }

    /// Returns the AND-count for this fragment.
    pub fn and_count(&self) -> u16 {
        self.nodes.len() as u16
    }

    /// Evaluates this fragment to a `TruthTable16`.
    pub fn eval_tt16(&self) -> TruthTable16 {
        // Build base slots (inputs + const0) in fixed positions.
        let mut values: Vec<TruthTable16> =
            Vec::with_capacity(FIRST_NODE_ID as usize + self.nodes.len());
        values.extend_from_slice(&[
            TruthTable16::var(0),
            TruthTable16::var(1),
            TruthTable16::var(2),
            TruthTable16::var(3),
            TruthTable16::const0(),
        ]);
        debug_assert_eq!(values.len(), FIRST_NODE_ID as usize);

        for node in &self.nodes {
            let tt = match *node {
                FragmentNode::And2 { a, b } => {
                    let mut a_tt = values[a.id as usize];
                    if a.negated {
                        a_tt = a_tt.not();
                    }
                    let mut b_tt = values[b.id as usize];
                    if b.negated {
                        b_tt = b_tt.not();
                    }
                    a_tt.and(b_tt)
                }
            };
            values.push(tt);
        }

        let mut out_tt = values[self.output.id as usize];
        if self.output.negated {
            out_tt = out_tt.not();
        }
        out_tt
    }

    /// Computes the AND-depth of this fragment (inputs/constants at depth 0).
    pub fn depth(&self) -> u16 {
        let mut depths: Vec<u16> = vec![0; FIRST_NODE_ID as usize + self.nodes.len()];
        for (idx, node) in self.nodes.iter().enumerate() {
            let out_id = FIRST_NODE_ID as usize + idx;
            let d = match *node {
                FragmentNode::And2 { a, b } => {
                    let da = depths[a.id as usize];
                    let db = depths[b.id as usize];
                    1 + core::cmp::max(da, db)
                }
            };
            depths[out_id] = d;
        }
        depths[self.output.id as usize]
    }

    /// Returns the per-input depth contribution from each fragment input to the
    /// fragment output.
    ///
    /// This is used to score a cut replacement's impact on global depth:
    /// `newDepthAtRoot = max_i(depth(leaf_i) + input_depths[i])`.
    pub fn input_depths(&self) -> [u16; 4] {
        // Base slots (inputs + const0) in fixed positions.
        let mut deps: Vec<[u16; 4]> = Vec::with_capacity(FIRST_NODE_ID as usize + self.nodes.len());
        deps.extend_from_slice(&[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]);
        // Mark input i as contributing depth 0 for itself (others remain 0).
        deps[0][0] = 0;
        deps[1][1] = 0;
        deps[2][2] = 0;
        deps[3][3] = 0;

        for node in &self.nodes {
            let v = match *node {
                FragmentNode::And2 { a, b } => {
                    let da = deps[a.id as usize];
                    let db = deps[b.id as usize];
                    [
                        1 + core::cmp::max(da[0], db[0]),
                        1 + core::cmp::max(da[1], db[1]),
                        1 + core::cmp::max(da[2], db[2]),
                        1 + core::cmp::max(da[3], db[3]),
                    ]
                }
            };
            deps.push(v);
        }

        deps[self.output.id as usize]
    }

    /// Applies an NPN transform to this fragment.
    ///
    /// This is intended for taking a canonical recipe (canonical input order)
    /// and producing a recipe over original inputs. See
    /// `crate::cut_db::npn` for the transform semantics.
    pub fn apply_npn(&self, xform: NpnTransform) -> Self {
        let remap_lit = |lit: Lit| -> Lit {
            if lit.id < INPUT_COUNT {
                let i = lit.id as usize;
                let out_input = xform.perm.0[i];
                let flip = ((xform.input_neg_mask >> i) & 1) != 0;
                let neg = lit.negated ^ flip;
                Lit::new(out_input as u16, neg)
            } else {
                // const0 and internal nodes are unchanged by input permutations/negations.
                lit
            }
        };

        let mut nodes = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            match *node {
                FragmentNode::And2 { a, b } => nodes.push(FragmentNode::And2 {
                    a: remap_lit(a),
                    b: remap_lit(b),
                }),
            }
        }

        let mut output = remap_lit(self.output);
        if xform.output_neg {
            output = output.negate();
        }

        Self { nodes, output }
    }

    /// Projects this fragment into a `GateFn` using `GateBuilder`.
    pub fn to_gatefn(&self, name: &str) -> GateFn {
        let mut gb = GateBuilder::new(name.to_string(), GateBuilderOptions::opt());

        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let c = gb.add_input("c".to_string(), 1);
        let d = gb.add_input("d".to_string(), 1);

        let mut ops: Vec<crate::aig::AigOperand> =
            Vec::with_capacity(FIRST_NODE_ID as usize + self.nodes.len());
        ops.push(*a.get_lsb(0));
        ops.push(*b.get_lsb(0));
        ops.push(*c.get_lsb(0));
        ops.push(*d.get_lsb(0));
        ops.push(gb.get_false());
        debug_assert_eq!(ops.len(), FIRST_NODE_ID as usize);

        let op_from_lit = |lit: Lit, ops: &[crate::aig::AigOperand]| -> crate::aig::AigOperand {
            let mut op = ops[lit.id as usize];
            if lit.negated {
                op = op.negate();
            }
            op
        };

        for node in &self.nodes {
            let op = match *node {
                FragmentNode::And2 { a, b } => {
                    let a_op = op_from_lit(a, &ops);
                    let b_op = op_from_lit(b, &ops);
                    gb.add_and_binary(a_op, b_op)
                }
            };
            ops.push(op);
        }

        let out_op = op_from_lit(self.output, &ops);
        gb.add_output("o".to_string(), AigBitVector::from_bit(out_op));
        gb.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_sim::gate_sim;
    use xlsynth::IrBits;

    fn gate_fn_to_tt16(g: &GateFn) -> TruthTable16 {
        // Evaluate for all 16 assignments with 1-bit inputs.
        let mut out = TruthTable16::const0();
        for i in 0u8..16 {
            let a = (i & 0b0001) as u64;
            let b = ((i >> 1) & 1) as u64;
            let c = ((i >> 2) & 1) as u64;
            let d = ((i >> 3) & 1) as u64;
            let inputs = vec![
                IrBits::make_ubits(1, a).unwrap(),
                IrBits::make_ubits(1, b).unwrap(),
                IrBits::make_ubits(1, c).unwrap(),
                IrBits::make_ubits(1, d).unwrap(),
            ];
            let sim = gate_sim::eval(g, &inputs, gate_sim::Collect::None);
            assert_eq!(sim.outputs.len(), 1);
            let bit = sim.outputs[0].get_bit(0).unwrap();
            out.set_bit(i, bit);
        }
        out
    }

    #[test]
    fn test_eval_tt16_matches_gate_sim_for_simple_and() {
        let frag = GateFnFragment {
            nodes: vec![FragmentNode::And2 {
                a: Lit::new(0, false),
                b: Lit::new(1, false),
            }],
            output: Lit::new(FIRST_NODE_ID, false),
        };
        let tt = frag.eval_tt16();

        let g = frag.to_gatefn("and_ab");
        let sim_tt = gate_fn_to_tt16(&g);
        assert_eq!(tt, sim_tt);
    }
}
