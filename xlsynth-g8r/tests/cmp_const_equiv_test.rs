// SPDX-License-Identifier: Apache-2.0

use xlsynth::{FnBuilder, IrBits, IrPackage, IrValue};
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

#[derive(Debug, Clone)]
struct BuiltIr {
    ir_text: String,
    top_name: String,
}

fn build_cmp_ir(bit_count: usize, binop: ir::Binop, lhs_is_const: bool, rhs_const: u64) -> BuiltIr {
    let mut package = IrPackage::new("sample").expect("create package");
    let op = ir::binop_to_operator(binop);
    let fn_name = format!(
        "cmp_{}_{}b_{}_0x{:x}",
        op,
        bit_count,
        if lhs_is_const {
            "const_lhs"
        } else {
            "const_rhs"
        },
        rhs_const
    );

    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);
    let ty = package.get_bits_type(bit_count as u64);
    let x = fb.param("x", &ty);
    let c = {
        let v = IrValue::make_ubits(bit_count, rhs_const).expect("make_ubits");
        fb.literal(&v, Some("c"))
    };

    let out = match (binop, lhs_is_const) {
        (ir::Binop::Eq, false) => fb.eq(&x, &c, Some("cmp")),
        (ir::Binop::Ne, false) => fb.ne(&x, &c, Some("cmp")),
        (ir::Binop::Ult, false) => fb.ult(&x, &c, Some("cmp")),
        (ir::Binop::Ule, false) => fb.ule(&x, &c, Some("cmp")),
        (ir::Binop::Ugt, false) => fb.ugt(&x, &c, Some("cmp")),
        (ir::Binop::Uge, false) => fb.uge(&x, &c, Some("cmp")),
        (ir::Binop::Slt, false) => fb.slt(&x, &c, Some("cmp")),
        (ir::Binop::Sle, false) => fb.sle(&x, &c, Some("cmp")),
        (ir::Binop::Sgt, false) => fb.sgt(&x, &c, Some("cmp")),
        (ir::Binop::Sge, false) => fb.sge(&x, &c, Some("cmp")),

        (ir::Binop::Eq, true) => fb.eq(&c, &x, Some("cmp")),
        (ir::Binop::Ne, true) => fb.ne(&c, &x, Some("cmp")),
        (ir::Binop::Ult, true) => fb.ult(&c, &x, Some("cmp")),
        (ir::Binop::Ule, true) => fb.ule(&c, &x, Some("cmp")),
        (ir::Binop::Ugt, true) => fb.ugt(&c, &x, Some("cmp")),
        (ir::Binop::Uge, true) => fb.uge(&c, &x, Some("cmp")),
        (ir::Binop::Slt, true) => fb.slt(&c, &x, Some("cmp")),
        (ir::Binop::Sle, true) => fb.sle(&c, &x, Some("cmp")),
        (ir::Binop::Sgt, true) => fb.sgt(&c, &x, Some("cmp")),
        (ir::Binop::Sge, true) => fb.sge(&c, &x, Some("cmp")),
        (other, _) => panic!("unexpected binop for cmp-const test: {other:?}"),
    };

    let _ = fb.build_with_return_value(&out).expect("build function");
    package.set_top_by_name(&fn_name).expect("set top");
    BuiltIr {
        ir_text: package.to_string(),
        top_name: fn_name,
    }
}

fn assert_ir_gate_equivalent_unary_cmp(
    bit_count: usize,
    binop: ir::Binop,
    lhs_is_const: bool,
    c: u64,
) {
    let built = build_cmp_ir(bit_count, binop, lhs_is_const, c);

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
            enable_rewrite_prio_encode: false,
        },
    )
    .unwrap();

    let max_val = 1u64 << bit_count;
    let interp_pkg = IrPackage::parse_ir(&built.ir_text, None).unwrap();
    let interp_fn = interp_pkg.get_function(&built.top_name).unwrap();
    for x in 0..max_val {
        let arg = IrValue::make_ubits(bit_count, x).unwrap();
        let ir_out = interp_fn.interpret(&[arg]).unwrap();
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
            "mismatch: bit_count={} binop={:?} lhs_is_const={} c=0x{:x} x=0x{:x}\nIR:\n{}",
            bit_count, binop, lhs_is_const, c, x, built.ir_text
        );
    }
}

#[test]
fn test_cmp_constant_rewrites_match_ir_interp_small_widths() {
    let _ = env_logger::builder().is_test(true).try_init();

    let binops: &[ir::Binop] = &[
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

    for bit_count in [1usize, 2, 3, 4] {
        let all_ones = if bit_count == 64 {
            u64::MAX
        } else {
            (1u64 << bit_count) - 1
        };
        let int_min = 1u64 << (bit_count - 1);
        let int_max = int_min - 1;

        let mut consts: Vec<u64> = vec![0, all_ones, int_min, int_max];
        for k in 0..bit_count {
            consts.push(1u64 << k);
        }
        for k in 1..=bit_count {
            consts.push((1u64 << k) - 1);
        }
        consts.sort();
        consts.dedup();

        for &binop in binops {
            for &c in &consts {
                assert_ir_gate_equivalent_unary_cmp(bit_count, binop, false, c);
                assert_ir_gate_equivalent_unary_cmp(bit_count, binop, true, c);
            }
        }
    }
}
