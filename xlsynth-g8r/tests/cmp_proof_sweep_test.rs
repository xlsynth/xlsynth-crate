// SPDX-License-Identifier: Apache-2.0

use xlsynth::{FnBuilder, IrBits, IrPackage, IrValue};
use xlsynth_g8r::aig_serdes::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RhsSpec {
    Param,
    Zero,
    AllOnes,
    Pow2 { bit_index: usize },
    Pow2Minus1 { bit_index: usize },
}

impl RhsSpec {
    fn all_for_bit_count(bit_count: usize) -> Vec<RhsSpec> {
        let mut specs = vec![RhsSpec::Param, RhsSpec::Zero, RhsSpec::AllOnes];
        for bit_index in 0..bit_count {
            specs.push(RhsSpec::Pow2 { bit_index });
        }
        for bit_index in 1..=bit_count {
            specs.push(RhsSpec::Pow2Minus1 { bit_index });
        }
        specs
    }

    fn name(self) -> String {
        match self {
            RhsSpec::Param => "param".to_string(),
            RhsSpec::Zero => "0".to_string(),
            RhsSpec::AllOnes => "all_ones".to_string(),
            RhsSpec::Pow2 { bit_index } => format!("pow2_{bit_index}"),
            RhsSpec::Pow2Minus1 { bit_index } => format!("pow2m1_{bit_index}"),
        }
    }

    fn is_literal(self) -> bool {
        !matches!(self, RhsSpec::Param)
    }

    fn literal_value(self, bit_count: usize) -> Option<u64> {
        match self {
            RhsSpec::Param => None,
            RhsSpec::Zero => Some(0),
            RhsSpec::AllOnes => {
                assert!(bit_count <= 64);
                Some((1u64 << bit_count) - 1)
            }
            RhsSpec::Pow2 { bit_index } => {
                assert!(bit_index < bit_count);
                Some(1u64 << bit_index)
            }
            RhsSpec::Pow2Minus1 { bit_index } => {
                assert!(bit_index >= 1 && bit_index <= bit_count);
                Some((1u64 << bit_index) - 1)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct BuiltIr {
    ir_text: String,
    top_name: String,
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

fn build_cmp_ir_text(
    bit_count: usize,
    binop: ir::Binop,
    rhs_spec: RhsSpec,
    swap_const: bool,
) -> BuiltIr {
    let mut package = IrPackage::new("sample").expect("create package");
    let op = ir::binop_to_operator(binop);
    let fn_name = format!(
        "cmp_proof_{}_{}b_{}_{}",
        op,
        bit_count,
        rhs_spec.name(),
        if swap_const { "swap" } else { "normal" }
    );

    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);
    let ty = package.get_bits_type(bit_count as u64);

    let (lhs, rhs) = match (rhs_spec, swap_const) {
        (RhsSpec::Param, _) => {
            let lhs = fb.param("lhs", &ty);
            let rhs = fb.param("rhs", &ty);
            (lhs, rhs)
        }
        (spec, true) => {
            // Literal-on-LHS form: exercise normalization/commutation paths.
            let x = fb.param("x", &ty);
            let c = spec.literal_value(bit_count).expect("literal value");
            let v = IrValue::make_ubits(bit_count, c).expect("make_ubits");
            let lit = fb.literal(&v, Some("c"));
            (lit, x)
        }
        (spec, false) => {
            // Normal literal-on-RHS form.
            let x = fb.param("x", &ty);
            let c = spec.literal_value(bit_count).expect("literal value");
            let v = IrValue::make_ubits(bit_count, c).expect("make_ubits");
            let lit = fb.literal(&v, Some("c"));
            (x, lit)
        }
    };

    let out = match binop {
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
        other => panic!("unexpected binop in proof sweep: {other:?}"),
    };

    let _ = fb.build_with_return_value(&out).expect("build function");
    package.set_top_by_name(&fn_name).expect("set top");
    BuiltIr {
        ir_text: package.to_string(),
        top_name: fn_name,
    }
}

fn assert_ir_gate_equivalent_for_built(
    built: &BuiltIr,
    bit_count: usize,
    rhs_spec: RhsSpec,
    swap_const: bool,
) {
    let mut parser = ir_parser::Parser::new(&built.ir_text);
    let ir_pkg = parser.parse_and_validate_package().unwrap();
    let ir_fn = ir_pkg.get_top_fn().unwrap();

    let gatified = gatify(
        &ir_fn,
        GatifyOptions {
            fold: true,
            check_equivalence: false,
            hash: true,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::RippleCarry,
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
        },
    )
    .unwrap();

    let interp_pkg = IrPackage::parse_ir(&built.ir_text, None).unwrap();
    let interp_fn = interp_pkg.get_function(&built.top_name).unwrap();

    let max_val = 1u64 << bit_count;
    if rhs_spec == RhsSpec::Param {
        for lhs in 0..max_val {
            for rhs in 0..max_val {
                let ir_out = interp_fn
                    .interpret(&[
                        IrValue::make_ubits(bit_count, lhs).unwrap(),
                        IrValue::make_ubits(bit_count, rhs).unwrap(),
                    ])
                    .unwrap();
                let ir_bit = ir_out.to_bits().unwrap().get_bit(0).unwrap();

                let gate_out = gate_sim::eval(
                    &gatified.gate_fn,
                    &[
                        IrBits::make_ubits(bit_count, lhs).unwrap(),
                        IrBits::make_ubits(bit_count, rhs).unwrap(),
                    ],
                    Collect::None,
                );
                assert_eq!(gate_out.outputs.len(), 1);
                let gate_bit = gate_out.outputs[0].get_bit(0).unwrap();

                assert_eq!(
                    ir_bit, gate_bit,
                    "mismatch: bit_count={} rhs_spec={:?} swap_const={} lhs=0x{:x} rhs=0x{:x}\nIR:\n{}",
                    bit_count, rhs_spec, swap_const, lhs, rhs, built.ir_text
                );
            }
        }
    } else {
        for x in 0..max_val {
            let ir_out = interp_fn
                .interpret(&[IrValue::make_ubits(bit_count, x).unwrap()])
                .unwrap();
            let ir_bit = ir_out.to_bits().unwrap().get_bit(0).unwrap();

            let gate_out = gate_sim::eval(
                &gatified.gate_fn,
                &[IrBits::make_ubits(bit_count, x).unwrap()],
                Collect::None,
            );
            assert_eq!(gate_out.outputs.len(), 1);
            let gate_bit = gate_out.outputs[0].get_bit(0).unwrap();

            assert_eq!(
                ir_bit, gate_bit,
                "mismatch: bit_count={} rhs_spec={:?} swap_const={} x=0x{:x}\nIR:\n{}",
                bit_count, rhs_spec, swap_const, x, built.ir_text
            );
        }
    }
}

#[test]
fn test_cmp_gatify_proof_sweep_small_widths() {
    let _ = env_logger::builder().is_test(true).try_init();

    for bit_count in 1usize..=4 {
        for &binop in CMP_BINOPS {
            for rhs_spec in RhsSpec::all_for_bit_count(bit_count) {
                // Normal form: lhs is a param; rhs is param or literal depending on rhs_spec.
                let built =
                    build_cmp_ir_text(bit_count, binop, rhs_spec, /* swap_const= */ false);
                assert_ir_gate_equivalent_for_built(
                    &built, bit_count, rhs_spec, /* swap_const= */ false,
                );

                // Swapped literal: exercise literal-on-LHS normalization/commutation paths.
                if rhs_spec.is_literal() {
                    let built =
                        build_cmp_ir_text(bit_count, binop, rhs_spec, /* swap_const= */ true);
                    assert_ir_gate_equivalent_for_built(
                        &built, bit_count, rhs_spec, /* swap_const= */ true,
                    );
                }
            }
        }
    }
}
