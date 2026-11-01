// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::gate::AigNode;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_g8r::test_utils::prove_prepared_ir_equiv;
use xlsynth_pir::ir::{Binop, NodePayload};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{IrBits, IrValue};
use xlsynth_prover::prover::types::EquivResult;

#[test]
fn compares_the_wrapped_result_at_either_endpoint_spelling() {
    for (w, k) in [(2, 1), (3, 1), (4, 3), (5, 2)] {
        for signed_cmp in [false, true] {
            for signed_term in [false, true] {
                for relation in [Relation::Gt, Relation::Le, Relation::Ge, Relation::Lt] {
                    for reverse in [false, true] {
                        let case = Case {
                            w,
                            k,
                            signed_cmp,
                            signed_term,
                            relation,
                            reverse,
                            ..Case::base()
                        };
                        // With k = W-1, an unsigned narrow term has its upper
                        // endpoint at signed MAX. The adjacent literal
                        // represents MIN.
                        let signed_discontinuity =
                            signed_cmp && !signed_term && k + 1 == w && relation.adjacent();
                        case.check(
                            if signed_discontinuity {
                                None
                            } else {
                                Some(false)
                            },
                            false,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn signed_max_predicate_is_a_literal_without_gate_folding() {
    for (relation, expected) in [(Relation::Gt, false), (Relation::Le, true)] {
        for reverse in [false, true] {
            let case = Case {
                w: 32,
                k: 31,
                signed_term: false,
                relation,
                reverse,
                ..Case::base()
            };
            let input = Parser::new(&case.ir()).parse_fn().unwrap();
            let gate_fn = gatify(
                &input,
                GatifyOptions {
                    fold: false,
                    enable_rewrite_nary_add: true,
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap()
            .gate_fn;
            let output = gate_fn.outputs[0].bit_vector.get_lsb(0);
            let AigNode::Literal { value, .. } = gate_fn.get(output.node) else {
                panic!("expected a literal output for {case:?}");
            };
            assert_eq!(*value ^ output.negated, expected, "{case:?}");
        }
    }
}

#[test]
fn signed_max_plus_one_is_signed_min_not_the_next_integer() {
    for relation in [Relation::Ge, Relation::Lt] {
        for reverse in [false, true] {
            let case = Case {
                w: 4,
                k: 3,
                signed_cmp: true,
                signed_term: false,
                relation,
                reverse,
                ..Case::base()
            };
            assert_eq!(case.literal(), 8);
            for (x, y) in [(7, 0), (7, 1), (8, 0), (15, 7)] {
                assert_eq!(
                    case.expected(x, y),
                    u128::from(matches!(relation, Relation::Ge))
                );
            }
            // A generic constant folder can also discharge this relation.
            case.check(None, true);
        }
    }
}

#[test]
fn independently_wrapped_inner_and_outer_additions() {
    for signed_cmp in [false, true] {
        for signed_term in [false, true] {
            let offset = 1 + if signed_term { 1 << 2 } else { 0 };
            let case = Case {
                w: 5,
                k: 3,
                signed_cmp,
                signed_term,
                offset,
                ..Case::base()
            };
            case.check(Some(false), false);
        }
    }
    let case = Case {
        w: 5,
        k: 3,
        offset: 5,
        ..Case::base()
    };
    // At y=7 the eight-valued translation wraps to signed -4. High x can
    // still win. With y=4, signed +1 takes MAX to MIN.
    assert_eq!(case.expected(0, 7), 0);
    assert_eq!(case.expected(8, 7), 1);
    assert_eq!(case.expected(15, 4), 0);
    assert_eq!(case.expected(16, 2), 1);
    let unsigned = Case {
        signed_cmp: false,
        ..case
    };
    assert_eq!(unsigned.expected(0, 2), 1); // 0 + (-1) wraps to unsigned MAX.
}

#[test]
fn limb_crossings_preserve_carry_and_all_output_observations() {
    for (w, k) in [(65, 5), (67, 65)] {
        let case = Case {
            w,
            k,
            offset: (1u128 << (k - 1)) + 1,
            ..Case::base()
        };
        case.check(Some(false), true);
    }
}

#[test]
fn preserve_exposed_sums_and_explicit_adder_contracts() {
    for exposure in [Exposure::Sum, Exposure::LowSumBit, Exposure::Narrow] {
        let case = Case {
            exposure,
            offset: 5,
            ..Case::base()
        };
        case.check(Some(!matches!(exposure, Exposure::Narrow)), true);
    }
    // A same-width term's resize signedness does not affect its value.
    // Scheduled adders and explicitly negated terms are native PIR contracts.
    for (wide_signed, scheduled, subtract) in [
        (true, false, false),
        (false, false, false),
        (true, true, false),
        (false, false, true),
    ] {
        let case = Case {
            native: true,
            wide_signed,
            scheduled,
            subtract,
            ..Case::base()
        };
        case.check(Some(scheduled || subtract), true);
    }
    for (relation, literal) in [(Relation::Gt, 4), (Relation::Ge, 3)] {
        Case {
            relation,
            override_literal: Some(literal),
            ..Case::base()
        }
        .check(Some(true), true);
    }
}

#[derive(Clone, Copy, Debug)]
enum Exposure {
    Predicate,
    Sum,
    LowSumBit,
    Narrow,
}

#[derive(Clone, Copy, Debug)]
enum Relation {
    Gt,
    Le,
    Ge,
    Lt,
}

impl Relation {
    fn adjacent(self) -> bool {
        matches!(self, Self::Ge | Self::Lt)
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    w: usize,
    k: usize,
    signed_cmp: bool,
    signed_term: bool,
    relation: Relation,
    reverse: bool,
    offset: u128,
    exposure: Exposure,
    native: bool,
    wide_signed: bool,
    scheduled: bool,
    subtract: bool,
    override_literal: Option<u128>,
}

impl Case {
    fn base() -> Self {
        Self {
            w: 5,
            k: 3,
            signed_cmp: true,
            signed_term: true,
            relation: Relation::Gt,
            reverse: false,
            offset: 0,
            exposure: Exposure::Predicate,
            native: false,
            wide_signed: false,
            scheduled: false,
            subtract: false,
            override_literal: None,
        }
    }

    fn literal(self) -> u128 {
        self.override_literal.unwrap_or(
            (1u128 << (self.k - usize::from(self.signed_term))) - 1
                + u128::from(self.relation.adjacent()),
        )
    }

    fn ir(self) -> String {
        let Self {
            w,
            k,
            reverse,
            offset,
            signed_term,
            ..
        } = self;
        let (extra_bits, output) = match self.exposure {
            Exposure::Predicate => (0, "identity(observation, id=11)"),
            Exposure::Sum => (w, "concat(total, observation, id=11)"),
            Exposure::LowSumBit => (1, "concat(low, observation, id=11)"),
            Exposure::Narrow => (k, "concat(term, observation, id=11)"),
        };
        let mut text = format!(
            "fn test(a: bits[{w}] id=1, b: bits[{k}] id=2) -> bits[{}] {{\n",
            1 + extra_bits
        );
        let term = if offset == 0 {
            "b"
        } else {
            text.push_str(&format!("  offset: bits[{k}] = literal(value={offset:#x}, id=3)\n  term: bits[{k}] = add(b, offset, id=4)\n"));
            "term"
        };
        if self.native {
            let arch = if self.scheduled {
                ", arch=kogge_stone"
            } else {
                ""
            };
            text.push_str(&format!("  total: bits[{w}] = ext_nary_add(a, {term}, signed=[{}, {signed_term}], negated=[false, {}]{arch}, id=6)\n",self.wide_signed,self.subtract));
        } else {
            let ext = if signed_term { "sign_ext" } else { "zero_ext" };
            let terms = if reverse {
                "extended, a"
            } else {
                "a, extended"
            };
            text.push_str(&format!("  extended: bits[{w}] = {ext}({term}, new_bit_count={w}, id=5)\n  total: bits[{w}] = add({terms}, id=6)\n"));
        }
        let rel = match (self.relation, reverse) {
            (Relation::Gt, false) | (Relation::Lt, true) => "gt",
            (Relation::Gt, true) | (Relation::Lt, false) => "lt",
            (Relation::Ge, false) | (Relation::Le, true) => "ge",
            (Relation::Ge, true) | (Relation::Le, false) => "le",
        };
        let operands = if reverse {
            "endpoint, total"
        } else {
            "total, endpoint"
        };
        let sign = if self.signed_cmp { "s" } else { "u" };
        text.push_str(&format!("  endpoint: bits[{w}] = literal(value={:#x}, id=7)\n  observation: bits[1] = {sign}{rel}({operands}, id=8)\n",self.literal()));
        if matches!(self.exposure, Exposure::LowSumBit) {
            text.push_str("  low: bits[1] = bit_slice(total, start=0, width=1, id=9)\n");
        }
        let output = if matches!(self.exposure, Exposure::Narrow) && offset == 0 {
            "concat(b, observation, id=11)"
        } else {
            output
        };
        text.push_str(&format!(
            "  ret result: bits[{}] = {output}\n}}",
            1 + extra_bits
        ));
        text
    }

    // Use mathematical integers to widen the input, then wrap only at the
    // result width. Neither carry decomposition nor bit-level thresholds enter
    // this oracle.
    fn expected(self, x: u128, y: u128) -> u128 {
        let term = (y + self.offset) % (1u128 << self.k);
        let n = interpreted(term, self.k, self.signed_term);
        let sum =
            ((x as i128 + if self.subtract { -n } else { n }).rem_euclid(1i128 << self.w)) as u128;
        let s = interpreted(sum, self.w, self.signed_cmp);
        let c = interpreted(self.literal(), self.w, self.signed_cmp);
        let flag = match self.relation {
            Relation::Gt => s > c,
            Relation::Le => s <= c,
            Relation::Ge => s >= c,
            Relation::Lt => s < c,
        };
        let extra = match self.exposure {
            Exposure::Predicate => 0,
            Exposure::Sum => sum,
            Exposure::LowSumBit => sum & 1,
            Exposure::Narrow => term,
        };
        (extra << 1) | u128::from(flag)
    }

    fn check(self, retains_wide_add: Option<bool>, formal: bool) {
        let text = self.ir();
        let input = Parser::new(&text).parse_fn().unwrap();
        let prepared = prep_for_gatify(
            &input,
            None,
            PrepForGatifyOptions {
                enable_rewrite_nary_add: true,
                ..PrepForGatifyOptions::all_opts_disabled()
            },
        );
        if let Some(expected) = retains_wide_add {
            let wide_add = prepared.nodes.iter().any(|node| {
                node.ty.bit_count() == self.w
                    && matches!(
                        node.payload,
                        NodePayload::ExtNaryAdd { .. }
                            | NodePayload::Binop(Binop::Add | Binop::Sub, _, _)
                    )
            });
            assert_eq!(wide_add, expected, "{self:?}\n{prepared}");
        }
        if formal {
            let proof = prove_prepared_ir_equiv(&input, &prepared);
            assert!(matches!(proof, EquivResult::Proved), "{self:?}: {proof:?}");
        }
        for x in values(self.w) {
            for y in values(self.k) {
                let args = [value(self.w, x), value(self.k, y)];
                let expected = value(
                    input.get_node(input.ret_node_ref.unwrap()).ty.bit_count(),
                    self.expected(x, y),
                );
                for f in [&input, &prepared] {
                    let FnEvalResult::Success(result) = eval_fn(f, &args) else {
                        panic!("failed to evaluate {self:?}")
                    };
                    assert_eq!(result.value, expected, "{self:?}, a={x:#x}, b={y:#x}");
                }
            }
        }
    }
}

fn interpreted(n: u128, width: usize, signed: bool) -> i128 {
    let modulus = 1i128 << width;
    if signed && n >= (modulus / 2) as u128 {
        n as i128 - modulus
    } else {
        n as i128
    }
}

fn values(width: usize) -> Vec<u128> {
    if width <= 5 {
        return (0..1u128 << width).collect();
    }
    let high = 1u128 << (width - 1);
    vec![
        0,
        1,
        2,
        high - 2,
        high - 1,
        high,
        high + 1,
        2 * high - 2,
        2 * high - 1,
    ]
}

fn value(width: usize, data: u128) -> IrValue {
    IrValue::from_bits(&IrBits::from_lsb_fn(width, |i| data & (1u128 << i) != 0))
}
