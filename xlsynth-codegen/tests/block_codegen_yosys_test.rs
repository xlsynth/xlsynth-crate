// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
#[path = "support/cases.rs"]
mod cases;
mod support;
use cases::*;
use std::collections::BTreeMap;
use std::time::Duration;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_g8r::netlist::yosys::YosysToolchain;
use xlsynth_pir::IrBits;
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::PackageMember;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};

/// Narrowing followed by replacement and concatenation exercises width
/// boundaries.
const NARROW_SLICE_UPDATE: &str = r#"package public_narrow_slice_update
top block narrow_slice_update(in0: bits[3], in1: bits[3], out: bits[5]) {
  in0: bits[3] = input_port(name=in0, id=1)
  in1: bits[3] = input_port(name=in1, id=2)
  dynamic_bit_slice.3: bits[1] = dynamic_bit_slice(in0, in1, width=1, id=3)
  literal.4: bits[4] = literal(value=1, id=4)
  decode.5: bits[2] = decode(in1, width=2, id=5)
  bit_slice_update.6: bits[3] = bit_slice_update(in0, decode.5, dynamic_bit_slice.3, id=6)
  sel.7: bits[4] = sel(in1, cases=[literal.4, literal.4, literal.4, literal.4], default=literal.4, id=7)
  bit_slice.8: bits[1] = bit_slice(decode.5, start=1, width=1, id=8)
  ugt.9: bits[1] = ugt(in0, in1, id=9)
  concat.10: bits[5] = concat(decode.5, bit_slice_update.6, id=10)
  encode.11: bits[3] = encode(concat.10, id=11)
  xor.12: bits[4] = xor(sel.7, id=12)
  shrl.13: bits[5] = shrl(concat.10, literal.4, id=13)
  ne.14: bits[1] = ne(in0, in0, id=14)
  out: () = output_port(shrl.13, name=out, id=15)
}
"#;

/// Exercises multiple outputs, modular arithmetic, and values wider than u64.
#[test]
fn arithmetic_outputs_evaluate_with_yosys() {
    let yosys = YosysToolchain::from_env().expect("required Yosys executable");
    for width in [8, 32, 65] {
        let ir = format!(
            r#"package arithmetic
top block arithmetic(a: bits[{width}], b: bits[{width}], sum: bits[{width}], difference: bits[{width}]) {{
  a: bits[{width}] = input_port(name=a, id=1)
  b: bits[{width}] = input_port(name=b, id=2)
  added: bits[{width}] = add(a, b, id=3)
  subtracted: bits[{width}] = sub(a, b, id=4)
  sum: () = output_port(added, name=sum, id=5)
  difference: () = output_port(subtracted, name=difference, id=6)
}}
"#
        );
        let one = IrBits::make_ubits(width, 1).unwrap();
        let inputs = [
            (IrBits::all_ones(width), one.clone()),
            (IrBits::zero(width), one),
            (
                IrBits::signed_min_value(width),
                IrBits::signed_min_value(width),
            ),
        ]
        .into_iter()
        .map(|(a, b)| BTreeMap::from([("a".into(), a), ("b".into(), b)]))
        .collect::<Vec<_>>();
        let expected = inputs
            .iter()
            .map(|input| {
                BTreeMap::from([
                    ("sum".into(), input["a"].add(&input["b"])),
                    ("difference".into(), input["a"].sub(&input["b"])),
                ])
            })
            .collect::<Vec<_>>();
        let rtl = emit(&ir, &BlockCodegenOptions::default());
        let actual = yosys
            .eval_combinational(
                &rtl,
                "arithmetic",
                &inputs,
                &BTreeMap::from([("sum".into(), width), ("difference".into(), width)]),
                Duration::from_secs(30),
            )
            .unwrap();
        assert_eq!(actual, expected, "width={width}\n{rtl}");
    }
}

/// Confirms evaluation traverses emitted child instances after flattening.
#[test]
fn hierarchical_outputs_evaluate_with_yosys() {
    let yosys = YosysToolchain::from_env().expect("required Yosys executable");
    let inputs = [(0, 255), (165, 90), (15, 15)]
        .into_iter()
        .map(|(a, b)| {
            BTreeMap::from([
                ("first".into(), IrBits::make_ubits(8, a).unwrap()),
                ("second".into(), IrBits::make_ubits(8, b).unwrap()),
            ])
        })
        .collect::<Vec<_>>();
    let expected = inputs
        .iter()
        .map(|input| BTreeMap::from([("result".into(), input["first"].xor(&input["second"]))]))
        .collect::<Vec<_>>();
    let rtl = emit(HIERARCHY, &BlockCodegenOptions::default());
    let actual = yosys
        .eval_combinational(
            &rtl,
            "parent",
            &inputs,
            &BTreeMap::from([("result".into(), 8)]),
            Duration::from_secs(30),
        )
        .unwrap();
    assert_eq!(actual, expected, "{rtl}");
}

/// Retains all-input coverage of the narrow regression without mapped cells.
#[test]
fn narrow_slice_update_matches_pir_for_all_inputs_with_yosys() {
    let yosys = YosysToolchain::from_env().expect("required Yosys executable");
    let package = package(NARROW_SLICE_UPDATE);
    let Some(PackageMember::Block { func, .. }) = package.get_top_block() else {
        panic!("expected top block");
    };
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for a in 0..8 {
        for b in 0..8 {
            let args = [
                IrValue::make_ubits(3, a).unwrap(),
                IrValue::make_ubits(3, b).unwrap(),
            ];
            let a = IrBits::make_ubits(3, a).unwrap();
            let b = IrBits::make_ubits(3, b).unwrap();
            let FnEvalResult::Success(result) = eval_fn(func, &args) else {
                panic!("PIR evaluation failed");
            };
            expected.push(BTreeMap::from([(
                "out".into(),
                result.value.to_bits().unwrap(),
            )]));
            inputs.push(BTreeMap::from([("in0".into(), a), ("in1".into(), b)]));
        }
    }
    let rtl = emit(NARROW_SLICE_UPDATE, &BlockCodegenOptions::default());
    let actual = yosys
        .eval_combinational(
            &rtl,
            "narrow_slice_update",
            &inputs,
            &BTreeMap::from([("out".into(), 5)]),
            Duration::from_secs(30),
        )
        .unwrap();
    assert_eq!(actual, expected, "{rtl}");
}

/// Missing inputs and unknown outputs must never become passing comparisons.
#[test]
fn yosys_eval_rejects_missing_inputs_and_unknown_outputs() {
    let yosys = YosysToolchain::from_env().expect("required Yosys executable");
    let outputs = BTreeMap::from([("y".into(), 1)]);
    for rtl in [
        "module dut(input a, output y); assign y = a; endmodule",
        "module dut(output y); assign y = 1'bx; endmodule",
    ] {
        let error = yosys
            .eval_combinational(
                rtl,
                "dut",
                &[BTreeMap::new()],
                &outputs,
                Duration::from_secs(30),
            )
            .unwrap_err();
        assert!(error.to_string().contains("Yosys program:"), "{error}");
    }
    let rtl = "module dut(input a, output y); assign y = a; endmodule";
    for inputs in [
        BTreeMap::from([("a".into(), IrBits::zero(2))]),
        BTreeMap::from([("y".into(), IrBits::zero(1))]),
    ] {
        assert!(
            yosys
                .eval_combinational(rtl, "dut", &[inputs], &outputs, Duration::from_secs(30))
                .is_err()
        );
    }
}

#[test]
fn one_hot_case_functions_prove_equivalent_to_prefix_logic_with_yosys() {
    let yosys = YosysToolchain::from_env().expect("required Yosys executable");
    let directory = tempfile::tempdir().unwrap();
    for width in [1, 4, 8, 65, 256] {
        let ir = format!(
            r#"package public_hot_proof

top block hot_proof(value: bits[{width}], lsb: bits[{out_width}], msb: bits[{out_width}]) {{
  value: bits[{width}] = input_port(name=value, id=1)
  low: bits[{out_width}] = one_hot(value, lsb_prio=true, id=2)
  high: bits[{out_width}] = one_hot(value, lsb_prio=false, id=3)
  lsb: () = output_port(low, name=lsb, id=4)
  msb: () = output_port(high, name=msb, id=5)
}}
"#,
            out_width = width + 1
        );
        let generated = emit(&ir, &BlockCodegenOptions::default());
        let mut source = format!(
            r#"{generated}
module proof(input logic [{last}:0] value, output logic equal);
  wire [{width}:0] actual_lsb, actual_msb, expected_lsb, expected_msb;
  hot_proof dut(.value(value), .lsb(actual_lsb), .msb(actual_msb));
  assign expected_lsb[{width}] = ~(|value);
  assign expected_msb[{width}] = ~(|value);
  assign equal = actual_lsb == expected_lsb && actual_msb == expected_msb;
"#,
            last = width - 1
        );
        for bit in 0..width {
            let low = if bit == 0 {
                format!("value[{bit}]")
            } else {
                format!("value[{bit}] && !(|value[{}:0])", bit - 1)
            };
            let high = if bit == width - 1 {
                format!("value[{bit}]")
            } else {
                format!("value[{bit}] && !(|value[{}:{}])", width - 1, bit + 1)
            };
            source.push_str(&format!(
                "assign expected_lsb[{bit}] = {low};\nassign expected_msb[{bit}] = {high};\n"
            ));
        }
        source.push_str("endmodule\n");
        std::fs::write(directory.path().join("proof.sv"), source).unwrap();
        yosys.run_script(directory.path(), "read_verilog -sv proof.sv; hierarchy -check -top proof; proc; flatten; opt; sat -verify -prove equal 1 -show-inputs -show-outputs", Duration::from_secs(60)).unwrap_or_else(|e| panic!("one_hot width={width}: {e}"));
    }
}
