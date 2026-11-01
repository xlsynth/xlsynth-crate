// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::table::build_area_table_report;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::check_equivalence::validate_same_fn_via_toolchain;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify, gatify_node_as_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::ir::{Binop, NodePayload, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{FnBuilder, IrBits, IrValue};

const WIDENED_SUM: &str = r#"package sample

top fn main(x: bits[8] id=1, y: bits[4] id=2) -> bits[1] {
  xw: bits[9] = zero_ext(x, new_bit_count=9, id=3)
  yw: bits[9] = zero_ext(y, new_bit_count=9, id=4)
  sum: bits[9] = add(xw, yw, id=5)
  bound: bits[9] = literal(value=15, id=6)
  ret cmp: bits[1] = ugt(sum, bound, id=7)
}"#;

#[test]
fn standalone_widened_sum_comparison_uses_its_original_operands() {
    let package = Parser::new(WIDENED_SUM)
        .parse_and_validate_package()
        .unwrap();
    let f = package.get_top_fn().unwrap();
    let gates = gatify_node_as_fn(
        f,
        f.ret_node_ref.unwrap(),
        &GatifyOptions {
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let widths: Vec<_> = gates.inputs.iter().map(|i| i.get_bit_count()).collect();
    assert_eq!(widths, [9, 9]);

    // Both the sum and the literal bound are inputs to the standalone UGT.
    let mut sim = PreparedGateSim::new(&gates);
    for (sum, bound) in [(15, 15), (16, 15), (16, 16), (255, 256), (256, 255)] {
        assert_eq!(
            sim.eval_outputs(&[
                IrBits::make_ubits(9, sum).unwrap(),
                IrBits::make_ubits(9, bound).unwrap(),
            ]),
            vec![IrBits::bool(sum > bound)],
            "sum={sum}, bound={bound}"
        );
    }
}

#[test]
fn inverted_predicate_owns_its_comparison_area() {
    let ir = WIDENED_SUM.replace("ugt(", "ule(");
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let mapped = gatify(
        f,
        GatifyOptions {
            fold: false,
            hash: false,
            track_pir_node_ids: true,
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let area = build_area_table_report(&mapped.gate_fn, f).unwrap();
    let ids: Vec<_> = area.rows.iter().map(|row| row.pir_node_id).collect();
    assert_eq!(ids, [7]);
    assert_eq!(area.unattributed, None);
}

#[test]
fn shared_sum_with_many_endpoint_comparisons() {
    // Every UGT would scan the entire endpoint if the use check came last.
    let width = 65_536;
    let count = 1024;
    let mut b = FnBuilder::new("shared");
    let x = b.param("x", Type::Bits(width - 1)).unwrap();
    let y = b.param("y", Type::Bits(4)).unwrap();
    let xw = b.zero_extend(x, width).unwrap();
    let yw = b.zero_extend(y, width).unwrap();
    let sum = b.add(xw, yw).unwrap();
    let bound = b.literal(IrValue::make_ubits(width, 15).unwrap()).unwrap();
    let predicates: Vec<_> = (0..count).map(|_| b.ugt(sum, bound).unwrap()).collect();
    let tuple = b.tuple(&predicates).unwrap();
    let f = b.build(tuple).unwrap();
    let prepared = prep_for_gatify(
        &f,
        None,
        PrepForGatifyOptions {
            enable_rewrite_nary_add: true,
            ..PrepForGatifyOptions::all_opts_disabled()
        },
    );
    let wide_comparisons = prepared.nodes.iter().filter(|n| {
        matches!(n.payload, NodePayload::Binop(Binop::Ugt, lhs, _)
            if prepared.get_node(lhs).ty == Type::Bits(width))
    });
    assert_eq!(wide_comparisons.count(), count);
}

#[test]
fn sparse_predicate_ids_require_room_for_all_helpers() {
    let package = Parser::new(WIDENED_SUM)
        .parse_and_validate_package()
        .unwrap();
    let mut f = package.get_top_fn().unwrap().clone();
    let root = f.ret_node_ref.unwrap();
    // Five helpers feed the OR in the original predicate slot.
    for (remaining, expected_op) in [(0, "ugt"), (4, "ugt"), (5, "or")] {
        f.get_node_mut(root).text_id = usize::MAX - remaining;
        gatify(&f, GatifyOptions::all_opts_enabled()).unwrap();
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
        let predicate = prepared.get_node(prepared.ret_node_ref.unwrap());
        assert_eq!(predicate.payload.get_operator(), expected_op);
    }
}

#[test]
fn replacing_the_predicate_preserves_only_user_names() {
    let package = Parser::new(WIDENED_SUM)
        .parse_and_validate_package()
        .unwrap();
    let mut f = package.get_top_fn().unwrap().clone();
    let root = f.ret_node_ref.unwrap();
    for (name, expected) in [("ugt.7", None), ("cmp", Some("cmp"))] {
        f.get_node_mut(root).name = Some(name.to_owned());
        gatify(&f, GatifyOptions::all_opts_enabled()).unwrap();
        let prepared = prep_for_gatify(&f, None, PrepForGatifyOptions::all_opts_enabled());
        let predicate = prepared.get_node(prepared.ret_node_ref.unwrap());
        assert_eq!(predicate.payload.get_operator(), "or");
        assert_eq!(predicate.name.as_deref(), expected);
    }
}

#[test]
fn a_wrapping_narrow_increment_needs_only_a_narrow_comparator() {
    for (wide, narrow) in [(17, 5), (32, 8)] {
        let case = Case {
            wide,
            narrow,
            result: wide + 1,
            offset: (1 << (narrow - 1)) + 1,
            ..Case::signed()
        };
        let ir = case.ir();
        case.check_observer(true);
        let gates = lower(&ir);
        let f = Parser::new(&ir).parse_fn().unwrap();
        validate_same_fn_via_toolchain(&f, &gates).unwrap();
        let stats = get_aig_stats(&gates);
        // The wide high bits feed a reduction; the arithmetic comparison uses
        // only the narrow low slice and a guard on the wrapped increment.
        assert!(
            stats.and_nodes <= 2 * wide + 12 * narrow,
            "unexpected full-width arithmetic cost for {case:?}: {stats:?}"
        );
    }
}

#[test]
fn small_widened_predicates_match_modular_arithmetic() {
    for (wide, narrow) in [(3, 1), (4, 3), (5, 3)] {
        for (wide_signed, narrow_signed, signed_cmp) in [
            (true, true, true),
            (true, false, true),
            (false, true, true),
            (false, false, true),
            (false, false, false),
        ] {
            for (literal_first, complement, swap_terms) in [
                (false, false, false),
                (true, false, true),
                (false, true, true),
                (true, true, false),
            ] {
                let case = Case {
                    wide,
                    narrow,
                    result: wide + if signed_cmp && !wide_signed { 2 } else { 1 },
                    wide_signed,
                    narrow_signed,
                    signed_cmp,
                    literal_first,
                    complement,
                    swap_terms,
                    ..Case::signed()
                };
                case.check_all();
            }
        }
    }
}

#[test]
fn the_adjacent_inclusive_boundary_is_the_same_partition() {
    for signed in [true, false] {
        for literal_first in [true, false] {
            for complement in [true, false] {
                let base = Case {
                    wide_signed: signed,
                    narrow_signed: signed,
                    signed_cmp: signed,
                    literal_first,
                    complement,
                    ..Case::signed()
                };
                let case = Case {
                    inclusive: true,
                    limit: Some(base.endpoint() + 1),
                    ..base
                };
                case.check_observer(true);
                case.check_all();
                // At the endpoint itself >= and < partition the values
                // differently.
                let wrong = Case {
                    limit: Some(base.endpoint()),
                    ..case
                };
                wrong.check_observer(false);
                wrong.check_all();
            }
        }
    }
}

#[test]
fn wrapping_narrow_values_and_overflowing_outer_sums_preserve_both_boundaries() {
    let base = Case::signed();
    let wrapped_minus_one = Case { offset: 5, ..base };
    // For k=3 the signed term is (y + 5) mod 8. At y=7 the term is
    // -4: low x=0 must fail, while a positive high x=8 must pass.
    assert_eq!(wrapped_minus_one.expected(0, 7), 0);
    assert_eq!(wrapped_minus_one.expected(8, 7), 1);
    wrapped_minus_one.check_all();

    let truncated = Case {
        result: base.wide,
        ..base
    };
    // A 5-bit signed sum of 15 + 3 overflows to -14.
    assert_eq!(truncated.expected(15, 3), 0);
    truncated.check_observer(true);
    truncated.check_all();

    let unsigned_wide = Case {
        wide_signed: false,
        ..base
    };
    // A 6-bit signed comparison of unsigned 31 + signed 3 sees -30.
    assert_eq!(unsigned_wide.expected(31, 3), 0);
    unsigned_wide.check_observer(false);
    unsigned_wide.check_all();

    let modular_unsigned = Case {
        wide_signed: false,
        narrow_signed: false,
        signed_cmp: false,
        result: base.wide,
        ..base
    };
    assert_eq!(modular_unsigned.expected(31, 7), 0);
    modular_unsigned.check_observer(true);
    modular_unsigned.check_all();
}

#[test]
fn shared_results_and_nonendpoint_constants_preserve_all_observations() {
    let base = Case::signed();
    for shared in [
        Exposure::WholeSum,
        Exposure::LowSumBit,
        Exposure::NarrowTerm,
    ] {
        let case = Case {
            shared,
            offset: 5,
            ..base
        };
        case.check_observer(matches!(shared, Exposure::NarrowTerm));
        case.check_all();
    }
    for limit in [0, 2, 4, 31] {
        let case = Case {
            limit: Some(limit),
            ..base
        };
        case.check_observer(false);
        case.check_all();
    }
    // Unsigned comparison cannot reinterpret a potentially negative term.
    Case {
        signed_cmp: false,
        ..base
    }
    .check_all();
}

#[derive(Clone, Copy, Debug)]
enum Exposure {
    Predicate,
    WholeSum,
    LowSumBit,
    NarrowTerm,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    wide: usize,
    narrow: usize,
    result: usize,
    wide_signed: bool,
    narrow_signed: bool,
    signed_cmp: bool,
    swap_terms: bool,
    literal_first: bool,
    complement: bool,
    inclusive: bool,
    offset: u64,
    limit: Option<u64>,
    shared: Exposure,
}

impl Case {
    fn signed() -> Self {
        Self {
            wide: 5,
            narrow: 3,
            result: 6,
            wide_signed: true,
            narrow_signed: true,
            signed_cmp: true,
            swap_terms: false,
            literal_first: false,
            complement: false,
            inclusive: false,
            offset: 0,
            limit: None,
            shared: Exposure::Predicate,
        }
    }

    fn endpoint(self) -> u64 {
        self.limit
            .unwrap_or((1 << (self.narrow - usize::from(self.narrow_signed))) - 1)
    }

    fn ir(self) -> String {
        let Self {
            wide,
            narrow,
            result,
            offset,
            ..
        } = self;
        let ext = |signed| if signed { "sign_ext" } else { "zero_ext" };
        let op = match (
            self.literal_first ^ self.complement,
            self.inclusive ^ self.complement,
        ) {
            (false, false) => "gt",
            (true, false) => "lt",
            (false, true) => "ge",
            (true, true) => "le",
        };
        let args = if self.literal_first {
            "bound, sum"
        } else {
            "sum, bound"
        };
        let sign = if self.signed_cmp { "s" } else { "u" };
        let terms = if self.swap_terms { "b, a" } else { "a, b" };
        let (extra, ret) = match self.shared {
            Exposure::Predicate => (0, "identity(test, id=11)"),
            Exposure::WholeSum => (result, "concat(sum, test, id=11)"),
            Exposure::LowSumBit => (1, "concat(low, test, id=11)"),
            Exposure::NarrowTerm => (narrow, "concat(term, test, id=11)"),
        };
        let mut ir = format!(
            "fn f(x: bits[{wide}] id=1, y: bits[{narrow}] id=2) -> bits[{}] {{\n",
            1 + extra
        );
        if offset == 0 {
            ir.push_str(&format!("  term: bits[{narrow}] = identity(y, id=4)\n"));
        } else {
            ir.push_str(&format!(
                "  offset: bits[{narrow}] = literal(value={offset}, id=3)\n  term: bits[{narrow}] = add(y, offset, id=4)\n"
            ));
        }
        ir.push_str(&format!(
            "  a: bits[{result}] = {}(x, new_bit_count={result}, id=5)\n  b: bits[{result}] = {}(term, new_bit_count={result}, id=6)\n  sum: bits[{result}] = add({terms}, id=7)\n  bound: bits[{result}] = literal(value={}, id=8)\n  test: bits[1] = {sign}{op}({args}, id=9)\n",
            ext(self.wide_signed), ext(self.narrow_signed), self.endpoint()
        ));
        if matches!(self.shared, Exposure::LowSumBit) {
            ir.push_str("  low: bits[1] = bit_slice(sum, start=0, width=1, id=10)\n");
        }
        ir.push_str(&format!("  ret r: bits[{}] = {ret}\n}}", 1 + extra));
        ir
    }

    fn check_observer(self, projected: bool) {
        let ir = self.ir();
        let f = Parser::new(&ir).parse_fn().unwrap();
        let prepared = prep_for_gatify(
            &f,
            None,
            PrepForGatifyOptions {
                enable_rewrite_nary_add: true,
                ..PrepForGatifyOptions::all_opts_disabled()
            },
        );
        let compares_full_sum = prepared.nodes.iter().any(|node| {
            let NodePayload::Binop(
                Binop::Sgt
                | Binop::Slt
                | Binop::Sle
                | Binop::Sge
                | Binop::Ugt
                | Binop::Ult
                | Binop::Ule
                | Binop::Uge,
                a,
                b,
            ) = node.payload
            else {
                return false;
            };
            [a, b].into_iter().any(|r| {
                let operand = prepared.get_node(r);
                operand.ty.bit_count() == self.result
                    && matches!(
                        operand.payload,
                        NodePayload::ExtNaryAdd { .. } | NodePayload::Binop(Binop::Add, _, _)
                    )
            })
        });
        assert_eq!(!compares_full_sum, projected, "{self:?}: {prepared}");
    }

    // Model each modular boundary explicitly, independently of the mapped gate
    // function.
    fn expected(self, x: u64, y: u64) -> u64 {
        let term = (y + self.offset) & ((1 << self.narrow) - 1);
        let sum = (number(x, self.wide, self.wide_signed)
            + number(term, self.narrow, self.narrow_signed)) as u64
            & ((1 << self.result) - 1);
        let sum_value = number(sum, self.result, self.signed_cmp);
        let bound = number(self.endpoint(), self.result, self.signed_cmp);
        let test = (if self.inclusive {
            sum_value >= bound
        } else {
            sum_value > bound
        }) ^ self.complement;
        let extra = match self.shared {
            Exposure::Predicate => 0,
            Exposure::WholeSum => sum,
            Exposure::LowSumBit => sum & 1,
            Exposure::NarrowTerm => term,
        };
        (extra << 1) | u64::from(test)
    }

    fn check_all(self) {
        let gates = lower(&self.ir());
        let mut sim = PreparedGateSim::new(&gates);
        for x in 0..1 << self.wide {
            for y in 0..1 << self.narrow {
                let value = sim.eval_outputs(&[
                    IrBits::make_ubits(self.wide, x).unwrap(),
                    IrBits::make_ubits(self.narrow, y).unwrap(),
                ]);
                assert_eq!(value.len(), 1);
                assert_eq!(
                    value[0],
                    IrBits::make_ubits(value[0].get_bit_count(), self.expected(x, y)).unwrap(),
                    "{self:?}, x={x}, y={y}"
                );
            }
        }
    }
}

fn number(bits: u64, width: usize, signed: bool) -> i64 {
    if signed && bits & (1 << (width - 1)) != 0 {
        bits as i64 - (1i64 << width)
    } else {
        bits as i64
    }
}

fn lower(ir: &str) -> GateFn {
    let f = Parser::new(ir).parse_fn().unwrap();
    let mapped = gatify(
        &f,
        GatifyOptions {
            fold: true,
            hash: true,
            enable_rewrite_nary_add: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    dce(&mapped.gate_fn)
}
