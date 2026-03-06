// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::time::SystemTime;

use xlsynth_vastly::compile_combo_module;
use xlsynth_vastly::eval_combo;
use xlsynth_vastly::plan_combo_eval;
use xlsynth_vastly::run_iverilog_combo_and_collect_vcd;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;
use xlsynth_vastly::Vcd;
use vastly_fuzz::codegen_semantics::make_vastly_input_map;
use vastly_fuzz::codegen_semantics::pack_ir_value_to_value4;
use vastly_fuzz::codegen_semantics::packed_signature;
use vastly_fuzz::codegen_semantics::parse_pir_top_fn;
use xlsynth::IrValue;

const COMPOUND_SHAPES_IR: &str = r#"package aot_tests

top fn compound_shapes(lhs: (bits[8], bits[16][2]), rhs: (bits[16], bits[8][2])) -> (bits[16][2], (bits[8], bits[16]), bits[8][2]) {
  lhs_base: bits[8] = tuple_index(lhs, index=0)
  lhs_arr: bits[16][2] = tuple_index(lhs, index=1)
  rhs_base: bits[16] = tuple_index(rhs, index=0)
  rhs_arr: bits[8][2] = tuple_index(rhs, index=1)

  i0: bits[1] = literal(value=0)
  i1: bits[1] = literal(value=1)

  l0: bits[16] = array_index(lhs_arr, indices=[i0])
  l1: bits[16] = array_index(lhs_arr, indices=[i1])
  r0: bits[8] = array_index(rhs_arr, indices=[i0])
  r1: bits[8] = array_index(rhs_arr, indices=[i1])

  r0_wide: bits[16] = zero_ext(r0, new_bit_count=16)
  s0: bits[16] = add(l0, r0_wide)
  s1: bits[16] = add(l1, rhs_base)
  out_wide_arr: bits[16][2] = array(s0, s1)

  out_pair: (bits[8], bits[16]) = tuple(r1, rhs_base)

  out_narrow0: bits[8] = add(lhs_base, r0)
  out_narrow1: bits[8] = add(lhs_base, r1)
  out_narrow_arr: bits[8][2] = array(out_narrow0, out_narrow1)

  ret out: (bits[16][2], (bits[8], bits[16]), bits[8][2]) = tuple(out_wide_arr, out_pair, out_narrow_arr)
}
"#;

fn make_compound_shapes_args() -> Vec<IrValue> {
    let lhs = IrValue::make_tuple(&[
        IrValue::parse_typed("bits[8]:0xaa").unwrap(),
        IrValue::make_array(&[
            IrValue::parse_typed("bits[16]:0x1122").unwrap(),
            IrValue::parse_typed("bits[16]:0x3344").unwrap(),
        ])
        .unwrap(),
    ]);
    let rhs = IrValue::make_tuple(&[
        IrValue::parse_typed("bits[16]:0x5566").unwrap(),
        IrValue::make_array(&[
            IrValue::parse_typed("bits[8]:0x77").unwrap(),
            IrValue::parse_typed("bits[8]:0x88").unwrap(),
        ])
        .unwrap(),
    ]);
    vec![lhs, rhs]
}

fn canonical_compound_shapes_ir() -> String {
    xlsynth::IrPackage::parse_ir(COMPOUND_SHAPES_IR, None)
        .unwrap()
        .to_string()
}

fn codegen_combo(
    ir_text: &str,
    module_name: &str,
    use_system_verilog: bool,
) -> Result<String, xlsynth::XlsynthError> {
    let package = xlsynth::IrPackage::parse_ir(ir_text, None)?;
    let sched_proto = "delay_model: \"unit\"";
    let codegen_proto = format!(
        "register_merge_strategy: STRATEGY_IDENTITY_ONLY\n\
generator: GENERATOR_KIND_COMBINATIONAL\n\
use_system_verilog: {use_system_verilog}\n\
module_name: \"{module_name}\"\n\
add_invariant_assertions: false\n\
codegen_version: 1"
    );
    let result = xlsynth::schedule_and_codegen(&package, sched_proto, &codegen_proto)?;
    result.get_verilog_text()
}

fn eval_codegen_with_vastly(
    src: &str,
    inputs: &BTreeMap<String, Value4>,
) -> Result<Value4, String> {
    let m = compile_combo_module(src).map_err(|e| format!("compile_combo_module failed: {e:?}"))?;
    let plan = plan_combo_eval(&m).map_err(|e| format!("plan_combo_eval failed: {e:?}"))?;
    let env = eval_combo(&m, &plan, inputs).map_err(|e| format!("eval_combo failed: {e:?}"))?;
    if m.output_ports.len() != 1 {
        return Err(format!("expected exactly one output port, got {}", m.output_ports.len()));
    }
    env.get(&m.output_ports[0].name)
        .cloned()
        .ok_or_else(|| format!("missing output `{}`", m.output_ports[0].name))
}

fn eval_codegen_with_iverilog_verilog(
    src: &str,
    inputs: &BTreeMap<String, Value4>,
) -> Result<Value4, String> {
    let m = compile_combo_module(src).map_err(|e| format!("compile_combo_module failed: {e:?}"))?;
    let td = mk_temp_dir().map_err(|e| format!("mk_temp_dir failed: {e}"))?;
    let result = (|| {
        let dut_path = td.join("dut.v");
        let out_vcd_path = td.join("iverilog.vcd");
        std::fs::write(&dut_path, src).map_err(|e| format!("write dut.v failed: {e}"))?;
        run_iverilog_combo_and_collect_vcd(&dut_path, &m, &[inputs.clone()], &out_vcd_path)
            .map_err(|e| format!("run_iverilog_combo_and_collect_vcd failed: {e:?}"))?;
        let vcd_text =
            std::fs::read_to_string(&out_vcd_path).map_err(|e| format!("read VCD failed: {e}"))?;
        let vcd = Vcd::parse(&vcd_text).map_err(|e| format!("parse VCD failed: {e:?}"))?;
        let timeline = vcd
            .materialize()
            .map_err(|e| format!("materialize VCD failed: {e:?}"))?;
        let (_, last_values) = timeline
            .last()
            .ok_or_else(|| "VCD had no timestamped events".to_string())?;
        let out_name = &m.output_ports[0].name;
        let bits = last_values
            .get(&format!("tb.dut.{out_name}"))
            .or_else(|| last_values.get(&format!("tb.{out_name}")))
            .ok_or_else(|| format!("missing output `{out_name}` in VCD"))?;
        value4_from_msb_bits(bits)
    })();
    let _ = std::fs::remove_dir_all(&td);
    result
}

fn value4_from_msb_bits(bits: &str) -> Result<Value4, String> {
    let mut out = Vec::with_capacity(bits.len());
    for c in bits.chars().rev() {
        out.push(match c {
            '0' => LogicBit::Zero,
            '1' => LogicBit::One,
            'x' | 'X' => LogicBit::X,
            'z' | 'Z' => LogicBit::Z,
            _ => return Err(format!("unexpected VCD bit `{c}` in `{bits}`")),
        });
    }
    Ok(Value4::new(out.len() as u32, Signedness::Unsigned, out))
}

fn mk_temp_dir() -> Result<std::path::PathBuf, String> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0u32..1000u32 {
        let p = base.join(format!("vastly_fuzz_test_iv_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return Ok(p),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("create temp dir: {e}")),
        }
    }
    Err("failed to create temp dir".to_string())
}

#[test]
fn packed_signature_accepts_tuple_and_array_top_levels() {
    let ir_text = canonical_compound_shapes_ir();
    let top = parse_pir_top_fn(&ir_text, "compound_shapes").unwrap();
    let sig = packed_signature(&top).unwrap();
    assert_eq!(sig.params.len(), 2);
    assert_eq!(sig.params[0].name, "lhs");
    assert_eq!(sig.params[0].width, 40);
    assert_eq!(sig.params[1].name, "rhs");
    assert_eq!(sig.params[1].width, 32);
    assert_eq!(sig.ret_width, 72);
}

#[test]
fn pack_ir_value_matches_codegen_packed_port_order() {
    let ir_text = canonical_compound_shapes_ir();
    let top = parse_pir_top_fn(&ir_text, "compound_shapes").unwrap();
    let args = make_compound_shapes_args();
    let lhs = pack_ir_value_to_value4(&top.params[0].ty, &args[0]).unwrap();
    let rhs = pack_ir_value_to_value4(&top.params[1].ty, &args[1]).unwrap();

    assert_eq!(lhs.to_hex_string_if_known().unwrap(), "aa33441122");
    assert_eq!(rhs.to_hex_string_if_known().unwrap(), "55668877");
}

#[test]
fn typed_top_level_codegen_matches_ir_and_oracles() {
    let ir_text = canonical_compound_shapes_ir();
    let top = parse_pir_top_fn(&ir_text, "compound_shapes").unwrap();
    let sig = packed_signature(&top).unwrap();
    let args = make_compound_shapes_args();

    let pkg = xlsynth::IrPackage::parse_ir(&ir_text, None).unwrap();
    let f = pkg.get_function("compound_shapes").unwrap();
    let ir_result = f.interpret(&args).unwrap();
    let want = pack_ir_value_to_value4(&sig.ret_ty, &ir_result).unwrap();

    let input_map = make_vastly_input_map(&sig, &args).unwrap();
    let verilog_src = codegen_combo(&ir_text, "compound_shapes_v", false).unwrap();
    let sv_src = codegen_combo(&ir_text, "compound_shapes_sv", true).unwrap();

    let got_v = eval_codegen_with_vastly(&verilog_src, &input_map).unwrap();
    let got_sv = eval_codegen_with_vastly(&sv_src, &input_map).unwrap();
    let got_iv = eval_codegen_with_iverilog_verilog(&verilog_src, &input_map).unwrap();

    assert_eq!(got_v.to_bit_string_msb_first(), want.to_bit_string_msb_first());
    assert_eq!(got_sv.to_bit_string_msb_first(), want.to_bit_string_msb_first());
    assert_eq!(got_iv.to_bit_string_msb_first(), want.to_bit_string_msb_first());
}
