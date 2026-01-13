// SPDX-License-Identifier: Apache-2.0

use xlsynth::FnBuilder;
use xlsynth::IrPackage;
use xlsynth::IrValue;
use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::aig_serdes::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::test_utils::Opt;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RhsSpec {
    Param,
    Zero,
    AllOnes,
    Pow2 { bit_index: u8 },
    Pow2Minus1 { bit_index: u8 },
}

impl RhsSpec {
    fn name(self) -> &'static str {
        match self {
            RhsSpec::Param => "param",
            RhsSpec::Zero => "0",
            RhsSpec::AllOnes => "all_ones",
            RhsSpec::Pow2 { bit_index: 0 } => "pow2_0",
            RhsSpec::Pow2 { bit_index: 1 } => "pow2_1",
            RhsSpec::Pow2 { bit_index: 2 } => "pow2_2",
            RhsSpec::Pow2 { bit_index: 3 } => "pow2_3",
            RhsSpec::Pow2 { bit_index: 4 } => "pow2_4",
            RhsSpec::Pow2 { bit_index: 5 } => "pow2_5",
            RhsSpec::Pow2 { bit_index: 6 } => "pow2_6",
            RhsSpec::Pow2 { bit_index: 7 } => "pow2_7",
            RhsSpec::Pow2 { bit_index } => panic!("unexpected pow2 bit_index for 8b: {bit_index}"),
            RhsSpec::Pow2Minus1 { bit_index: 1 } => "pow2m1_1",
            RhsSpec::Pow2Minus1 { bit_index: 2 } => "pow2m1_2",
            RhsSpec::Pow2Minus1 { bit_index: 3 } => "pow2m1_3",
            RhsSpec::Pow2Minus1 { bit_index: 4 } => "pow2m1_4",
            RhsSpec::Pow2Minus1 { bit_index: 5 } => "pow2m1_5",
            RhsSpec::Pow2Minus1 { bit_index: 6 } => "pow2m1_6",
            RhsSpec::Pow2Minus1 { bit_index: 7 } => "pow2m1_7",
            RhsSpec::Pow2Minus1 { bit_index: 8 } => "pow2m1_8",
            RhsSpec::Pow2Minus1 { bit_index } => {
                panic!("unexpected pow2minus1 bit_index for 8b: {bit_index}")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CmpRow {
    cmp: &'static str,
    rhs: &'static str,
    live_nodes: usize,
    deepest_path: usize,
}

const CMP_BINOPS: &[ir::Binop] = &[
    ir::Binop::Eq,
    ir::Binop::Ne,
    ir::Binop::Ult,
    ir::Binop::Ule,
    ir::Binop::Ugt,
    ir::Binop::Uge,
    ir::Binop::Slt,
    ir::Binop::Sle,
    ir::Binop::Sgt,
    ir::Binop::Sge,
];

fn build_8b_cmp_ir_text(kind: ir::Binop, rhs_spec: RhsSpec) -> String {
    let mut package = IrPackage::new("sample").expect("create package");
    let kind_name = ir::binop_to_operator(kind);
    let fn_name = format!("cmp_{}_8b_{}", kind_name, rhs_spec.name());
    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);

    let ty_u8 = package.get_bits_type(8);
    let lhs = fb.param("lhs", &ty_u8);

    let rhs = match rhs_spec {
        RhsSpec::Param => fb.param("rhs", &ty_u8),
        RhsSpec::Zero => {
            let v = IrValue::make_ubits(8, 0).expect("make_ubits");
            fb.literal(&v, Some("rhs"))
        }
        RhsSpec::AllOnes => {
            let v = IrValue::make_ubits(8, 0xff).expect("make_ubits");
            fb.literal(&v, Some("rhs"))
        }
        RhsSpec::Pow2 { bit_index } => {
            assert!(bit_index < 8, "bit_index must be < 8 for 8-bit pow2");
            let v = IrValue::make_ubits(8, 1u64 << bit_index).expect("make_ubits");
            fb.literal(&v, Some("rhs"))
        }
        RhsSpec::Pow2Minus1 { bit_index } => {
            assert!(
                (1..=8).contains(&bit_index),
                "bit_index must be in 1..=8 for 8-bit pow2minus1"
            );
            let value = (1u64 << bit_index) - 1;
            let v = IrValue::make_ubits(8, value).expect("make_ubits");
            fb.literal(&v, Some("rhs"))
        }
    };

    let out = match kind {
        ir::Binop::Eq => fb.eq(&lhs, &rhs, Some("cmp")),
        ir::Binop::Ne => fb.ne(&lhs, &rhs, Some("cmp")),
        ir::Binop::Ult => fb.ult(&lhs, &rhs, Some("cmp")),
        ir::Binop::Ule => fb.ule(&lhs, &rhs, Some("cmp")),
        ir::Binop::Ugt => fb.ugt(&lhs, &rhs, Some("cmp")),
        ir::Binop::Uge => fb.uge(&lhs, &rhs, Some("cmp")),
        ir::Binop::Slt => fb.slt(&lhs, &rhs, Some("cmp")),
        ir::Binop::Sle => fb.sle(&lhs, &rhs, Some("cmp")),
        ir::Binop::Sgt => fb.sgt(&lhs, &rhs, Some("cmp")),
        ir::Binop::Sge => fb.sge(&lhs, &rhs, Some("cmp")),
        other => panic!("unexpected binop for cmp sweep: {other:?}"),
    };

    let _ = fb.build_with_return_value(&out).expect("build function");
    package.set_top_by_name(&fn_name).expect("set top");
    package.to_string()
}

fn stats_for_ir_text(ir_text: &str, opt: Opt) -> SummaryStats {
    let mut parser = ir_parser::Parser::new(ir_text);
    let ir_package = parser.parse_and_validate_package().unwrap();
    let ir_fn = ir_package.get_top_fn().unwrap();
    let out = gatify(
        &ir_fn,
        GatifyOptions {
            fold: opt == Opt::Yes,
            check_equivalence: false,
            hash: opt == Opt::Yes,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::RippleCarry,
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
        },
    )
    .unwrap();
    get_summary_stats(&out.gate_fn)
}

fn gather_cmp_rows(rhs_specs: &[RhsSpec]) -> Vec<CmpRow> {
    let mut got: Vec<CmpRow> = Vec::new();
    for &kind in CMP_BINOPS {
        let kind_name = ir::binop_to_operator(kind);
        for &rhs_spec in rhs_specs {
            let ir_text = build_8b_cmp_ir_text(kind, rhs_spec);
            let stats = stats_for_ir_text(&ir_text, Opt::Yes);
            got.push(CmpRow {
                cmp: kind_name,
                rhs: rhs_spec.name(),
                live_nodes: stats.live_nodes,
                deepest_path: stats.deepest_path,
            });
        }
    }
    got
}

#[test]
fn test_cmp_gate_stats_sweep_8b_all_rhs_specs() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut rhs_specs: Vec<RhsSpec> = Vec::new();
    rhs_specs.push(RhsSpec::Param);
    rhs_specs.push(RhsSpec::Zero);
    rhs_specs.push(RhsSpec::AllOnes);
    for bit_index in 0u8..8 {
        rhs_specs.push(RhsSpec::Pow2 { bit_index });
    }
    for bit_index in 1u8..=8 {
        rhs_specs.push(RhsSpec::Pow2Minus1 { bit_index });
    }

    let got = gather_cmp_rows(&rhs_specs);

    // This is a "microbenchmark sweep" style test: lock in the expected gate
    // count + depth so we can notice regressions and improvements.
    // eprintln!("{got:#?}");
    #[rustfmt::skip]
    let want: &[CmpRow] = &[
        // eq
        CmpRow { cmp: "eq", rhs: "param", live_nodes: 47, deepest_path: 6 },
        CmpRow { cmp: "eq", rhs: "0", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "all_ones", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2_0", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2_1", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2_2", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2_3", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2_4", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2_5", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2_6", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2_7", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2m1_1", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2m1_2", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2m1_3", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2m1_4", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2m1_5", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2m1_6", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2m1_7", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "eq", rhs: "pow2m1_8", live_nodes: 15, deepest_path: 4 },

        // ne
        CmpRow { cmp: "ne", rhs: "param", live_nodes: 47, deepest_path: 6 },
        CmpRow { cmp: "ne", rhs: "0", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "all_ones", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2_0", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2_1", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2_2", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2_3", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2_4", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2_5", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2_6", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2_7", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2m1_1", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2m1_2", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2m1_3", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2m1_4", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2m1_5", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2m1_6", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2m1_7", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ne", rhs: "pow2m1_8", live_nodes: 15, deepest_path: 4 },

        // ult
        CmpRow { cmp: "ult", rhs: "param", live_nodes: 63, deepest_path: 10 },
        CmpRow { cmp: "ult", rhs: "0", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "ult", rhs: "all_ones", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ult", rhs: "pow2_0", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ult", rhs: "pow2_1", live_nodes: 13, deepest_path: 4 },
        CmpRow { cmp: "ult", rhs: "pow2_2", live_nodes: 11, deepest_path: 4 },
        CmpRow { cmp: "ult", rhs: "pow2_3", live_nodes: 9, deepest_path: 4 },
        CmpRow { cmp: "ult", rhs: "pow2_4", live_nodes: 7, deepest_path: 3 },
        CmpRow { cmp: "ult", rhs: "pow2_5", live_nodes: 5, deepest_path: 3 },
        CmpRow { cmp: "ult", rhs: "pow2_6", live_nodes: 3, deepest_path: 2 },
        CmpRow { cmp: "ult", rhs: "pow2_7", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "ult", rhs: "pow2m1_1", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ult", rhs: "pow2m1_2", live_nodes: 21, deepest_path: 6 },
        CmpRow { cmp: "ult", rhs: "pow2m1_3", live_nodes: 24, deepest_path: 7 },
        CmpRow { cmp: "ult", rhs: "pow2m1_4", live_nodes: 26, deepest_path: 7 },
        CmpRow { cmp: "ult", rhs: "pow2m1_5", live_nodes: 28, deepest_path: 8 },
        CmpRow { cmp: "ult", rhs: "pow2m1_6", live_nodes: 30, deepest_path: 8 },
        CmpRow { cmp: "ult", rhs: "pow2m1_7", live_nodes: 32, deepest_path: 8 },
        CmpRow { cmp: "ult", rhs: "pow2m1_8", live_nodes: 15, deepest_path: 4 },

        // ule
        CmpRow { cmp: "ule", rhs: "param", live_nodes: 69, deepest_path: 11 },
        CmpRow { cmp: "ule", rhs: "0", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ule", rhs: "all_ones", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "ule", rhs: "pow2_0", live_nodes: 13, deepest_path: 4 },
        CmpRow { cmp: "ule", rhs: "pow2_1", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ule", rhs: "pow2_2", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ule", rhs: "pow2_3", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ule", rhs: "pow2_4", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ule", rhs: "pow2_5", live_nodes: 15, deepest_path: 6 },
        CmpRow { cmp: "ule", rhs: "pow2_6", live_nodes: 15, deepest_path: 6 },
        CmpRow { cmp: "ule", rhs: "pow2_7", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ule", rhs: "pow2m1_1", live_nodes: 13, deepest_path: 4 },
        CmpRow { cmp: "ule", rhs: "pow2m1_2", live_nodes: 11, deepest_path: 4 },
        CmpRow { cmp: "ule", rhs: "pow2m1_3", live_nodes: 9, deepest_path: 4 },
        CmpRow { cmp: "ule", rhs: "pow2m1_4", live_nodes: 7, deepest_path: 3 },
        CmpRow { cmp: "ule", rhs: "pow2m1_5", live_nodes: 5, deepest_path: 3 },
        CmpRow { cmp: "ule", rhs: "pow2m1_6", live_nodes: 3, deepest_path: 2 },
        CmpRow { cmp: "ule", rhs: "pow2m1_7", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "ule", rhs: "pow2m1_8", live_nodes: 1, deepest_path: 1 },

        // ugt
        CmpRow { cmp: "ugt", rhs: "param", live_nodes: 63, deepest_path: 10 },
        CmpRow { cmp: "ugt", rhs: "0", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "ugt", rhs: "all_ones", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "ugt", rhs: "pow2_0", live_nodes: 13, deepest_path: 4 },
        CmpRow { cmp: "ugt", rhs: "pow2_1", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ugt", rhs: "pow2_2", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ugt", rhs: "pow2_3", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ugt", rhs: "pow2_4", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ugt", rhs: "pow2_5", live_nodes: 15, deepest_path: 6 },
        CmpRow { cmp: "ugt", rhs: "pow2_6", live_nodes: 15, deepest_path: 6 },
        CmpRow { cmp: "ugt", rhs: "pow2_7", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "ugt", rhs: "pow2m1_1", live_nodes: 13, deepest_path: 4 },
        CmpRow { cmp: "ugt", rhs: "pow2m1_2", live_nodes: 11, deepest_path: 4 },
        CmpRow { cmp: "ugt", rhs: "pow2m1_3", live_nodes: 9, deepest_path: 4 },
        CmpRow { cmp: "ugt", rhs: "pow2m1_4", live_nodes: 7, deepest_path: 3 },
        CmpRow { cmp: "ugt", rhs: "pow2m1_5", live_nodes: 5, deepest_path: 3 },
        CmpRow { cmp: "ugt", rhs: "pow2m1_6", live_nodes: 3, deepest_path: 2 },
        CmpRow { cmp: "ugt", rhs: "pow2m1_7", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "ugt", rhs: "pow2m1_8", live_nodes: 1, deepest_path: 1 },

        // uge
        CmpRow { cmp: "uge", rhs: "param", live_nodes: 69, deepest_path: 11 },
        CmpRow { cmp: "uge", rhs: "0", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "uge", rhs: "all_ones", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "uge", rhs: "pow2_0", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "uge", rhs: "pow2_1", live_nodes: 13, deepest_path: 4 },
        CmpRow { cmp: "uge", rhs: "pow2_2", live_nodes: 11, deepest_path: 4 },
        CmpRow { cmp: "uge", rhs: "pow2_3", live_nodes: 9, deepest_path: 4 },
        CmpRow { cmp: "uge", rhs: "pow2_4", live_nodes: 7, deepest_path: 3 },
        CmpRow { cmp: "uge", rhs: "pow2_5", live_nodes: 5, deepest_path: 3 },
        CmpRow { cmp: "uge", rhs: "pow2_6", live_nodes: 3, deepest_path: 2 },
        CmpRow { cmp: "uge", rhs: "pow2_7", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "uge", rhs: "pow2m1_1", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "uge", rhs: "pow2m1_2", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "uge", rhs: "pow2m1_3", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "uge", rhs: "pow2m1_4", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "uge", rhs: "pow2m1_5", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "uge", rhs: "pow2m1_6", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "uge", rhs: "pow2m1_7", live_nodes: 15, deepest_path: 5 },
        CmpRow { cmp: "uge", rhs: "pow2m1_8", live_nodes: 15, deepest_path: 4 },

        // slt
        CmpRow { cmp: "slt", rhs: "param", live_nodes: 66, deepest_path: 12 },
        CmpRow { cmp: "slt", rhs: "0", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "slt", rhs: "all_ones", live_nodes: 16, deepest_path: 5 },
        CmpRow { cmp: "slt", rhs: "pow2_0", live_nodes: 17, deepest_path: 6 },
        CmpRow { cmp: "slt", rhs: "pow2_1", live_nodes: 15, deepest_path: 6 },
        CmpRow { cmp: "slt", rhs: "pow2_2", live_nodes: 13, deepest_path: 6 },
        CmpRow { cmp: "slt", rhs: "pow2_3", live_nodes: 11, deepest_path: 6 },
        CmpRow { cmp: "slt", rhs: "pow2_4", live_nodes: 9, deepest_path: 5 },
        CmpRow { cmp: "slt", rhs: "pow2_5", live_nodes: 7, deepest_path: 5 },
        CmpRow { cmp: "slt", rhs: "pow2_6", live_nodes: 5, deepest_path: 4 },
        CmpRow { cmp: "slt", rhs: "pow2_7", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "slt", rhs: "pow2m1_1", live_nodes: 17, deepest_path: 6 },
        CmpRow { cmp: "slt", rhs: "pow2m1_2", live_nodes: 23, deepest_path: 8 },
        CmpRow { cmp: "slt", rhs: "pow2m1_3", live_nodes: 26, deepest_path: 9 },
        CmpRow { cmp: "slt", rhs: "pow2m1_4", live_nodes: 28, deepest_path: 9 },
        CmpRow { cmp: "slt", rhs: "pow2m1_5", live_nodes: 30, deepest_path: 10 },
        CmpRow { cmp: "slt", rhs: "pow2m1_6", live_nodes: 32, deepest_path: 10 },
        CmpRow { cmp: "slt", rhs: "pow2m1_7", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "slt", rhs: "pow2m1_8", live_nodes: 16, deepest_path: 5 },

        // sle
        CmpRow { cmp: "sle", rhs: "param", live_nodes: 72, deepest_path: 13 },
        CmpRow { cmp: "sle", rhs: "0", live_nodes: 16, deepest_path: 5 },
        CmpRow { cmp: "sle", rhs: "all_ones", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "sle", rhs: "pow2_0", live_nodes: 15, deepest_path: 6 },
        CmpRow { cmp: "sle", rhs: "pow2_1", live_nodes: 17, deepest_path: 7 },
        CmpRow { cmp: "sle", rhs: "pow2_2", live_nodes: 17, deepest_path: 7 },
        CmpRow { cmp: "sle", rhs: "pow2_3", live_nodes: 17, deepest_path: 7 },
        CmpRow { cmp: "sle", rhs: "pow2_4", live_nodes: 17, deepest_path: 7 },
        CmpRow { cmp: "sle", rhs: "pow2_5", live_nodes: 17, deepest_path: 8 },
        CmpRow { cmp: "sle", rhs: "pow2_6", live_nodes: 17, deepest_path: 8 },
        CmpRow { cmp: "sle", rhs: "pow2_7", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "sle", rhs: "pow2m1_1", live_nodes: 15, deepest_path: 6 },
        CmpRow { cmp: "sle", rhs: "pow2m1_2", live_nodes: 13, deepest_path: 6 },
        CmpRow { cmp: "sle", rhs: "pow2m1_3", live_nodes: 11, deepest_path: 6 },
        CmpRow { cmp: "sle", rhs: "pow2m1_4", live_nodes: 9, deepest_path: 5 },
        CmpRow { cmp: "sle", rhs: "pow2m1_5", live_nodes: 7, deepest_path: 5 },
        CmpRow { cmp: "sle", rhs: "pow2m1_6", live_nodes: 5, deepest_path: 4 },
        CmpRow { cmp: "sle", rhs: "pow2m1_7", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "sle", rhs: "pow2m1_8", live_nodes: 1, deepest_path: 1 },

        // sgt
        CmpRow { cmp: "sgt", rhs: "param", live_nodes: 72, deepest_path: 13 },
        CmpRow { cmp: "sgt", rhs: "0", live_nodes: 16, deepest_path: 5 },
        CmpRow { cmp: "sgt", rhs: "all_ones", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "sgt", rhs: "pow2_0", live_nodes: 14, deepest_path: 5 },
        CmpRow { cmp: "sgt", rhs: "pow2_1", live_nodes: 16, deepest_path: 6 },
        CmpRow { cmp: "sgt", rhs: "pow2_2", live_nodes: 16, deepest_path: 6 },
        CmpRow { cmp: "sgt", rhs: "pow2_3", live_nodes: 16, deepest_path: 6 },
        CmpRow { cmp: "sgt", rhs: "pow2_4", live_nodes: 16, deepest_path: 6 },
        CmpRow { cmp: "sgt", rhs: "pow2_5", live_nodes: 16, deepest_path: 7 },
        CmpRow { cmp: "sgt", rhs: "pow2_6", live_nodes: 16, deepest_path: 7 },
        CmpRow { cmp: "sgt", rhs: "pow2_7", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "sgt", rhs: "pow2m1_1", live_nodes: 14, deepest_path: 5 },
        CmpRow { cmp: "sgt", rhs: "pow2m1_2", live_nodes: 12, deepest_path: 5 },
        CmpRow { cmp: "sgt", rhs: "pow2m1_3", live_nodes: 10, deepest_path: 5 },
        CmpRow { cmp: "sgt", rhs: "pow2m1_4", live_nodes: 8, deepest_path: 4 },
        CmpRow { cmp: "sgt", rhs: "pow2m1_5", live_nodes: 6, deepest_path: 4 },
        CmpRow { cmp: "sgt", rhs: "pow2m1_6", live_nodes: 4, deepest_path: 3 },
        CmpRow { cmp: "sgt", rhs: "pow2m1_7", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "sgt", rhs: "pow2m1_8", live_nodes: 1, deepest_path: 1 },

        // sge
        CmpRow { cmp: "sge", rhs: "param", live_nodes: 73, deepest_path: 14 },
        CmpRow { cmp: "sge", rhs: "0", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "sge", rhs: "all_ones", live_nodes: 16, deepest_path: 5 },
        CmpRow { cmp: "sge", rhs: "pow2_0", live_nodes: 16, deepest_path: 5 },
        CmpRow { cmp: "sge", rhs: "pow2_1", live_nodes: 14, deepest_path: 5 },
        CmpRow { cmp: "sge", rhs: "pow2_2", live_nodes: 12, deepest_path: 5 },
        CmpRow { cmp: "sge", rhs: "pow2_3", live_nodes: 10, deepest_path: 5 },
        CmpRow { cmp: "sge", rhs: "pow2_4", live_nodes: 8, deepest_path: 4 },
        CmpRow { cmp: "sge", rhs: "pow2_5", live_nodes: 6, deepest_path: 4 },
        CmpRow { cmp: "sge", rhs: "pow2_6", live_nodes: 4, deepest_path: 3 },
        CmpRow { cmp: "sge", rhs: "pow2_7", live_nodes: 1, deepest_path: 1 },
        CmpRow { cmp: "sge", rhs: "pow2m1_1", live_nodes: 16, deepest_path: 5 },
        CmpRow { cmp: "sge", rhs: "pow2m1_2", live_nodes: 16, deepest_path: 6 },
        CmpRow { cmp: "sge", rhs: "pow2m1_3", live_nodes: 16, deepest_path: 6 },
        CmpRow { cmp: "sge", rhs: "pow2m1_4", live_nodes: 16, deepest_path: 5 },
        CmpRow { cmp: "sge", rhs: "pow2m1_5", live_nodes: 16, deepest_path: 6 },
        CmpRow { cmp: "sge", rhs: "pow2m1_6", live_nodes: 16, deepest_path: 6 },
        CmpRow { cmp: "sge", rhs: "pow2m1_7", live_nodes: 15, deepest_path: 4 },
        CmpRow { cmp: "sge", rhs: "pow2m1_8", live_nodes: 16, deepest_path: 5 },
    ];

    assert_eq!(got.as_slice(), want);
}
