// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Once;
use std::time::SystemTime;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;
use xlsynth::IrValue;
use xlsynth::XlsynthError;
use xlsynth_pir::ir_fuzz::FuzzSampleWithArgs;
use xlsynth_autocov::{generate_ir_fn_corpus_from_ir_text_with_replay, IrFnAutocovGenerateConfig};
use vastly_fuzz::codegen_semantics::make_vastly_input_map;
use vastly_fuzz::codegen_semantics::pack_ir_value_to_value4;
use vastly_fuzz::codegen_semantics::packed_signature;
use vastly_fuzz::codegen_semantics::parse_pir_top_fn;

fuzz_target!(|data: &[u8]| {
    install_panic_context_hook();
    quiet_xls_warnings();
    maybe_enable_backtrace_on_failure();
    let _panic_ctx_guard = begin_panic_context(data);

    let mut u = Unstructured::new(data);
    let fuzz_case = match FuzzSampleWithArgs::arbitrary(&mut u) {
        Ok(v) => v,
        // Invalid arbitrary payload shape; skip rather than fail this sample.
        Err(_) => return,
    };

    let mut package = match xlsynth::IrPackage::new("fuzz_pkg") {
        Ok(p) => p,
        // Package construction failures are infrastructure-level, not sample-specific semantics.
        Err(_) => return,
    };
    let f = match xlsynth_pir::ir_fuzz::generate_ir_fn(fuzz_case.sample.ops.clone(), &mut package, None) {
        Ok(f) => f,
        // Generator rejects some op sequences; treat those as uninteresting corpus points.
        Err(_) => return,
    };
    let fname = f.get_name();
    if package.set_top_by_name(&fname).is_err() {
        // If we cannot mark the generated function as top, we cannot run codegen/eval meaningfully.
        return;
    }

    let ir_text = package.to_string();
    set_panic_context_ir(&ir_text);
    // Skip samples that commonly lower via assertion machinery in non-SV
    // codegen or that the optional autocov engine cannot currently evaluate.
    // This keeps the fuzz signal focused on semantics gaps instead of known
    // tooling limitations/noise.
    if has_sv_only_codegen_noise(&ir_text) {
        return;
    }
    let pir_top = match parse_pir_top_fn(&ir_text, &fname) {
        Ok(v) => v,
        // Parser mismatch on generated text is tracked elsewhere; skip for this semantics target.
        Err(_) => return,
    };
    let sig = match packed_signature(&pir_top) {
        Some(sig) => sig,
        // Non-packable signatures are outside this target's current simulation harness.
        None => return,
    };
    let stimuli = match generate_stimuli(&fuzz_case, &pir_top, &ir_text, data) {
        Ok(v) => v,
        Err(e) => {
            unsupported(
                "autocov could not generate stimuli",
                &ir_text,
                None,
                Some(&e),
                data,
            );
            return;
        }
    };

    let verilog_src = match codegen_combo(&ir_text, "fuzz_codegen_v", false) {
        Ok(v) => v,
        Err(e) => {
            unsupported(
                "verilog-codegen-failed",
                &ir_text,
                None,
                Some(&e.to_string()),
                data,
            );
            return;
        }
    };
    set_panic_context_verilog(&verilog_src);
    let sv_src = match codegen_combo(&ir_text, "fuzz_codegen_sv", true) {
        Ok(v) => v,
        Err(e) => {
            unsupported(
                "systemverilog-codegen-failed",
                &ir_text,
                Some(&verilog_src),
                Some(&e.to_string()),
                data,
            );
            return;
        }
    };
    set_panic_context_sv(&sv_src);

    let mut expected_outputs = Vec::with_capacity(stimuli.len());
    let mut pipeline_input_vectors = Vec::with_capacity(stimuli.len());

    for (stimulus_index, stimulus) in stimuli.iter().enumerate() {
        let stimulus_ordinal = format!("{}/{}", stimulus_index + 1, stimuli.len());
        set_panic_context_stimulus(&stimulus.to_string());

        let args = match tuple_value_to_args(stimulus) {
            Ok(v) => v,
            Err(e) => {
                unsupported(
                    "failed to unpack stimulus tuple",
                    &ir_text,
                    None,
                    Some(&format!(
                        "{e} {}",
                        summarize_stimulus_with_index(stimulus, &stimulus_ordinal)
                    )),
                    data,
                );
                return;
            }
        };
        let ir_result = match f.interpret(&args) {
            Ok(v) => v,
            Err(_) => return,
        };
        let want_bits = match pack_ir_value_to_value4(&sig.ret_ty, &ir_result) {
            Ok(v) => v,
            Err(_) => return,
        };
        if want_bits.width != sig.ret_width {
            maybe_enable_backtrace_on_failure();
            let mut msg = format!(
                "ir-width-mismatch got={} want={} {} {} {}",
                want_bits.width,
                sig.ret_width,
                summarize_ir(&ir_text),
                summarize_stimulus_with_index(stimulus, &stimulus_ordinal),
                summarize_sample(data)
            );
            if verbose_failure_context() {
                msg.push_str("\n--- BEGIN FAILURE CONTEXT ---\n");
                msg.push_str("IR:\n");
                msg.push_str(&ir_text);
                msg.push_str("\nStimulus:\n");
                msg.push_str(&stimulus.to_string());
                msg.push_str("\n--- END FAILURE CONTEXT ---");
            }
            append_verbose_backtrace(&mut msg);
            panic!("{msg}");
        }

        let input_map = match make_vastly_input_map(&sig, &args) {
            Ok(v) => v,
            Err(_) => return,
        };
        expected_outputs.push(want_bits.clone());
        pipeline_input_vectors.push(input_map.clone());

        let got_v = match eval_codegen_with_vastly(&verilog_src, &input_map) {
            Ok(v) => v,
            Err(e) => {
                unsupported(
                    "vastly could not compile/eval generated Verilog",
                    &ir_text,
                    Some(&verilog_src),
                    Some(&format!(
                        "{e} {}",
                        summarize_stimulus_with_index(stimulus, &stimulus_ordinal)
                    )),
                    data,
                );
                return;
            }
        };
        let got_sv = match eval_codegen_with_vastly(&sv_src, &input_map) {
            Ok(v) => v,
            Err(e) => {
                unsupported(
                    "vastly could not compile/eval generated SystemVerilog",
                    &ir_text,
                    Some(&sv_src),
                    Some(&format!(
                        "{e} {}",
                        summarize_stimulus_with_index(stimulus, &stimulus_ordinal)
                    )),
                    data,
                );
                return;
            }
        };
        assert_same_bits(
            "verilog",
            &want_bits,
            &got_v,
            &ir_text,
            &verilog_src,
            data,
            stimulus,
            &stimulus_ordinal,
        );
        assert_same_bits(
            "systemverilog",
            &want_bits,
            &got_sv,
            &ir_text,
            &sv_src,
            data,
            stimulus,
            &stimulus_ordinal,
        );
        assert_same_bits(
            "v_vs_sv",
            &got_v,
            &got_sv,
            &ir_text,
            &sv_src,
            data,
            stimulus,
            &stimulus_ordinal,
        );
        if include_iverilog_oracle() {
            // Keep Icarus on the plain-Verilog path only. XLS can emit
            // SystemVerilog array assignment forms that current Icarus does not
            // support, so the external oracle is: Icarus Verilog, plus an
            // explicit Vastly-V/Vastly-SV equality check.
            let got_iv_v = match eval_codegen_with_iverilog_verilog(&verilog_src, &input_map) {
                Ok(v) => v,
                Err(e) => {
                    unsupported(
                        "iverilog could not compile/eval generated Verilog",
                        &ir_text,
                        Some(&verilog_src),
                        Some(&format!(
                            "{e} {}",
                            summarize_stimulus_with_index(stimulus, &stimulus_ordinal)
                        )),
                        data,
                    );
                    return;
                }
            };

            assert_same_bits(
                "iverilog_verilog",
                &want_bits,
                &got_iv_v,
                &ir_text,
                &verilog_src,
                data,
                stimulus,
                &stimulus_ordinal,
            );
            assert_same_bits(
                "iverilog_v_vs_vastly_v",
                &got_iv_v,
                &got_v,
                &ir_text,
                &verilog_src,
                data,
                stimulus,
                &stimulus_ordinal,
            );
            assert_same_bits(
                "iverilog_v_vs_vastly_sv",
                &got_iv_v,
                &got_sv,
                &ir_text,
                &sv_src,
                data,
                stimulus,
                &stimulus_ordinal,
            );
        }
    }

    let mut stage1_retired: Option<(String, Vec<Value4>)> = None;
    if include_stage1_pipeline_oracle() || include_stage2_pipeline_oracle() {
        let pipeline_src = match codegen_pipeline(&ir_text, "fuzz_codegen_pipe1", 1) {
            Ok(v) => v,
            Err(e) => {
                unsupported(
                    "pipeline-stage1-codegen-failed",
                    &ir_text,
                    None,
                    Some(&e.to_string()),
                    data,
                );
                return;
            }
        };
        let pipeline_outputs = match eval_pipeline_with_vastly(&pipeline_src, 1, &pipeline_input_vectors)
        {
            Ok(v) => v,
            Err(e) => {
                unsupported(
                    "vastly could not compile/eval generated stage1 pipeline",
                    &ir_text,
                    Some(&pipeline_src),
                    Some(&e),
                    data,
                );
                return;
            }
        };
        if pipeline_outputs.len() != expected_outputs.len() {
            maybe_enable_backtrace_on_failure();
            emit_mismatch_context(&ir_text, &pipeline_src);
            let mut msg = format!(
                "pipeline1-retire-count-mismatch want={} got={} {} {} {}",
                expected_outputs.len(),
                pipeline_outputs.len(),
                summarize_ir(&ir_text),
                summarize_generated(&pipeline_src),
                summarize_sample(data)
            );
            append_verbose_backtrace(&mut msg);
            panic!("{msg}");
        }
        for (i, ((want, got), stimulus)) in expected_outputs
            .iter()
            .zip(pipeline_outputs.iter())
            .zip(stimuli.iter())
            .enumerate()
        {
            let stimulus_ordinal = format!("{}/{}", i + 1, stimuli.len());
            set_panic_context_stimulus(&stimulus.to_string());
            assert_same_bits(
                "pipeline1",
                want,
                got,
                &ir_text,
                &pipeline_src,
                data,
                stimulus,
                &stimulus_ordinal,
            );
        }
        stage1_retired = Some((pipeline_src, pipeline_outputs));
    }

    if include_stage2_pipeline_oracle() {
        let pipeline2_src = match codegen_pipeline(&ir_text, "fuzz_codegen_pipe2", 2) {
            Ok(v) => v,
            Err(e) => {
                unsupported(
                    "pipeline-stage2-codegen-failed",
                    &ir_text,
                    None,
                    Some(&e.to_string()),
                    data,
                );
                return;
            }
        };
        let pipeline2_outputs = match eval_pipeline_with_vastly(&pipeline2_src, 2, &pipeline_input_vectors)
        {
            Ok(v) => v,
            Err(e) => {
                unsupported(
                    "vastly could not compile/eval generated stage2 pipeline",
                    &ir_text,
                    Some(&pipeline2_src),
                    Some(&e),
                    data,
                );
                return;
            }
        };
        if pipeline2_outputs.len() != expected_outputs.len() {
            maybe_enable_backtrace_on_failure();
            emit_mismatch_context(&ir_text, &pipeline2_src);
            let mut msg = format!(
                "pipeline2-retire-count-mismatch want={} got={} {} {} {}",
                expected_outputs.len(),
                pipeline2_outputs.len(),
                summarize_ir(&ir_text),
                summarize_generated(&pipeline2_src),
                summarize_sample(data)
            );
            append_verbose_backtrace(&mut msg);
            panic!("{msg}");
        }
        for (i, ((want, got), stimulus)) in expected_outputs
            .iter()
            .zip(pipeline2_outputs.iter())
            .zip(stimuli.iter())
            .enumerate()
        {
            let stimulus_ordinal = format!("{}/{}", i + 1, stimuli.len());
            set_panic_context_stimulus(&stimulus.to_string());
            assert_same_bits(
                "pipeline2",
                want,
                got,
                &ir_text,
                &pipeline2_src,
                data,
                stimulus,
                &stimulus_ordinal,
            );
        }
        if let Some((pipeline1_src, pipeline1_outputs)) = &stage1_retired {
            if pipeline1_outputs.len() != pipeline2_outputs.len() {
                maybe_enable_backtrace_on_failure();
                emit_mismatch_context(&ir_text, &pipeline2_src);
                let mut msg = format!(
                    "pipeline2_vs_pipeline1_retire_count_mismatch stage1={} stage2={} {} {} {} {}",
                    pipeline1_outputs.len(),
                    pipeline2_outputs.len(),
                    summarize_ir(&ir_text),
                    summarize_generated(pipeline1_src),
                    summarize_generated(&pipeline2_src),
                    summarize_sample(data)
                );
                append_verbose_backtrace(&mut msg);
                panic!("{msg}");
            }
            for (i, ((want, got), stimulus)) in pipeline1_outputs
                .iter()
                .zip(pipeline2_outputs.iter())
                .zip(stimuli.iter())
                .enumerate()
            {
                let stimulus_ordinal = format!("{}/{}", i + 1, stimuli.len());
                set_panic_context_stimulus(&stimulus.to_string());
                assert_same_bits(
                    "pipeline2_vs_pipeline1",
                    want,
                    got,
                    &ir_text,
                    &pipeline2_src,
                    data,
                    stimulus,
                    &stimulus_ordinal,
                );
            }
        }
    }
});

fn generate_stimuli(
    fuzz_case: &FuzzSampleWithArgs,
    pir_top: &xlsynth_pir::ir::Fn,
    ir_text: &str,
    sample_data: &[u8],
) -> Result<Vec<IrValue>, String> {
    let base_args = fuzz_case.gen_args_for_fn(pir_top);
    let base_tuple = IrValue::make_tuple(&base_args);
    if !use_autocov_stimuli() {
        return Ok(vec![base_tuple]);
    }

    let cfg = IrFnAutocovGenerateConfig {
        seed: stable_hash_bytes(sample_data),
        max_iters: Some(env_u64_or("VASTLY_FUZZ_AUTOCOV_MAX_ITERS", 4096)),
        max_corpus_len: Some(env_usize_or("VASTLY_FUZZ_AUTOCOV_MAX_CORPUS_LEN", 64)),
        progress_every: Some(env_u64_or("VASTLY_FUZZ_AUTOCOV_PROGRESS_EVERY", 0)),
        threads: Some(env_usize_or("VASTLY_FUZZ_AUTOCOV_THREADS", 1)),
        seed_structured: env_bool_or("VASTLY_FUZZ_AUTOCOV_SEED_STRUCTURED", false),
        seed_two_hot_max_bits: env_usize_or("VASTLY_FUZZ_AUTOCOV_SEED_TWO_HOT_MAX_BITS", 0),
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        generate_ir_fn_corpus_from_ir_text_with_replay(
            ir_text,
            &pir_top.name,
            std::slice::from_ref(&base_tuple),
            cfg,
        )
    }))
    .map_err(|panic_payload| format!("autocov panicked: {}", summarize_panic_payload(panic_payload)))??;
    if result.corpus.is_empty() {
        return Ok(vec![base_tuple]);
    }
    Ok(result.corpus)
}

fn tuple_value_to_args(tuple_value: &IrValue) -> Result<Vec<IrValue>, String> {
    tuple_value
        .get_elements()
        .map_err(|e| format!("stimulus is not a tuple: {e}"))
}

fn codegen_combo(
    ir_text: &str,
    module_name: &str,
    use_system_verilog: bool,
) -> Result<String, XlsynthError> {
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
    let result = xlsynth::schedule_and_codegen(&package, &sched_proto, &codegen_proto)?;
    result.get_verilog_text()
}

fn codegen_pipeline(
    ir_text: &str,
    module_name: &str,
    pipeline_stages: u32,
) -> Result<String, XlsynthError> {
    let package = xlsynth::IrPackage::parse_ir(ir_text, None)?;
    let sched_proto = format!("delay_model: \"unit\"\npipeline_stages: {pipeline_stages}");
    let codegen_proto = format!(
        "register_merge_strategy: STRATEGY_IDENTITY_ONLY\n\
generator: GENERATOR_KIND_PIPELINE\n\
use_system_verilog: true\n\
module_name: \"{module_name}\"\n\
input_valid_signal: \"input_valid\"\n\
output_valid_signal: \"output_valid\"\n\
flop_inputs: false\n\
flop_outputs: false\n\
reset: \"rst\"\n\
reset_active_low: false\n\
reset_asynchronous: false\n\
reset_data_path: true\n\
add_invariant_assertions: false\n\
codegen_version: 1"
    );
    let result = xlsynth::schedule_and_codegen(&package, &sched_proto, &codegen_proto)?;
    result.get_verilog_text()
}

fn eval_codegen_with_vastly(
    src: &str,
    inputs: &BTreeMap<String, Value4>,
) -> Result<Value4, String> {
    let m = xlsynth_vastly::compile_combo_module(src)
        .map_err(|e| format!("compile_combo_module failed: {e:?}"))?;
    if m.output_ports.len() != 1 {
        return Err(format!(
            "expected exactly one output port, got {}",
            m.output_ports.len()
        ));
    }
    if m.input_ports.len() != inputs.len() {
        return Err(format!(
            "input port count mismatch: module has {} inputs but vector has {}",
            m.input_ports.len(),
            inputs.len()
        ));
    }
    for p in &m.input_ports {
        if !inputs.contains_key(&p.name) {
            return Err(format!("missing expected input `{}`", p.name));
        }
    }
    let plan = xlsynth_vastly::plan_combo_eval(&m).map_err(|e| format!("plan_combo_eval failed: {e:?}"))?;
    let env = xlsynth_vastly::eval_combo(&m, &plan, inputs)
        .map_err(|e| format!("eval_combo failed: {e:?}"))?;
    let out_port = &m.output_ports[0];
    env.get(&out_port.name)
        .cloned()
        .ok_or_else(|| format!("output `{}` missing from eval environment", out_port.name))
}

fn eval_codegen_with_iverilog_verilog(
    src: &str,
    inputs: &BTreeMap<String, Value4>,
) -> Result<Value4, String> {
    let m = xlsynth_vastly::compile_combo_module(src)
        .map_err(|e| format!("compile_combo_module failed before iverilog run: {e:?}"))?;
    if m.output_ports.len() != 1 {
        return Err(format!(
            "expected exactly one output port, got {}",
            m.output_ports.len()
        ));
    }
    let out_name = m.output_ports[0].name.clone();
    let td = mk_temp_dir().map_err(|e| format!("mk_temp_dir failed: {e}"))?;
    let result = (|| {
        let dut_path = td.join("dut.v");
        let out_vcd_path = td.join("iverilog.vcd");
        std::fs::write(&dut_path, src).map_err(|e| format!("write dut.v failed: {e}"))?;
        let vectors = vec![inputs.clone()];
        xlsynth_vastly::run_iverilog_combo_and_collect_vcd(&dut_path, &m, &vectors, &out_vcd_path)
            .map_err(|e| format!("run_iverilog_combo_and_collect_vcd failed: {e:?}"))?;
        let vcd_text =
            std::fs::read_to_string(&out_vcd_path).map_err(|e| format!("read VCD failed: {e}"))?;
        let vcd =
            xlsynth_vastly::Vcd::parse(&vcd_text).map_err(|e| format!("parse VCD failed: {e:?}"))?;
        extract_output_from_vcd(&vcd, &out_name)
    })();
    let _ = std::fs::remove_dir_all(&td);
    result
}

fn eval_pipeline_with_vastly(
    src: &str,
    pipeline_stages: usize,
    input_vectors: &[BTreeMap<String, Value4>],
) -> Result<Vec<Value4>, String> {
    let m = xlsynth_vastly::compile_pipeline_module(src)
        .map_err(|e| format!("compile_pipeline_module failed: {e:?}"))?;
    let output_valid_name = "output_valid";
    let data_outputs: Vec<&xlsynth_vastly::Port> = m
        .combo
        .output_ports
        .iter()
        .filter(|p| p.name != output_valid_name)
        .collect();
    if data_outputs.len() != 1 {
        return Err(format!(
            "expected exactly one data output port besides `{output_valid_name}`, got {}",
            data_outputs.len()
        ));
    }
    let data_output_name = data_outputs[0].name.clone();
    let stimulus = build_pipeline_stimulus(&m, input_vectors, pipeline_stages)?;
    let initial_state = m.initial_state_x();
    let cycle_outputs = xlsynth_vastly::run_pipeline_and_collect_outputs(&m, &stimulus, &initial_state)
        .map_err(|e| format!("run_pipeline_and_collect_outputs failed: {e:?}"))?;
    let mut retired = Vec::new();
    for cycle_out in cycle_outputs {
        let valid = cycle_out.get(output_valid_name).ok_or_else(|| {
            format!("pipeline output `{output_valid_name}` missing from collected outputs")
        })?;
        match valid.bits_lsb_first().first().copied() {
            Some(LogicBit::One) => {
                let v = cycle_out.get(&data_output_name).ok_or_else(|| {
                    format!(
                        "pipeline data output `{}` missing from collected outputs",
                        data_output_name
                    )
                })?;
                retired.push(v.clone());
            }
            Some(LogicBit::Zero) => {}
            Some(other) => {
                return Err(format!(
                    "pipeline output_valid was not concrete 0/1: {:?}",
                    other
                ));
            }
            None => {
                return Err("pipeline output_valid had width 0".to_string());
            }
        }
    }
    Ok(retired)
}

fn build_pipeline_stimulus(
    m: &xlsynth_vastly::CompiledPipelineModule,
    input_vectors: &[BTreeMap<String, Value4>],
    pipeline_stages: usize,
) -> Result<xlsynth_vastly::PipelineStimulus, String> {
    let mut zero_cycle = BTreeMap::new();
    for p in &m.combo.input_ports {
        let info = m
            .combo
            .decls
            .get(&p.name)
            .ok_or_else(|| format!("missing decl info for pipeline input `{}`", p.name))?;
        let v = match p.name.as_str() {
            "rst" | "input_valid" => Value4::new(1, Signedness::Unsigned, vec![LogicBit::Zero]),
            _ => Value4::zeros(p.width, info.signedness),
        };
        zero_cycle.insert(p.name.clone(), v);
    }
    if !zero_cycle.contains_key("rst") || !zero_cycle.contains_key("input_valid") {
        return Err(
            "generated pipeline did not expose expected `rst` and `input_valid` ports".to_string(),
        );
    }

    let mut cycles = Vec::with_capacity(input_vectors.len() + 3);
    for _ in 0..2 {
        let mut cycle = zero_cycle.clone();
        cycle.insert(
            "rst".to_string(),
            Value4::new(1, Signedness::Unsigned, vec![LogicBit::One]),
        );
        cycles.push(xlsynth_vastly::PipelineCycle { inputs: cycle });
    }
    for input_map in input_vectors {
        let mut cycle = zero_cycle.clone();
        cycle.insert(
            "input_valid".to_string(),
            Value4::new(1, Signedness::Unsigned, vec![LogicBit::One]),
        );
        for p in &m.combo.input_ports {
            if p.name == "rst" || p.name == "input_valid" {
                continue;
            }
            let v = input_map
                .get(&p.name)
                .ok_or_else(|| format!("missing pipeline data input `{}`", p.name))?;
            cycle.insert(p.name.clone(), v.clone());
        }
        cycles.push(xlsynth_vastly::PipelineCycle { inputs: cycle });
    }
    for _ in 0..pipeline_stages.max(1) {
        cycles.push(xlsynth_vastly::PipelineCycle {
            inputs: zero_cycle.clone(),
        });
    }
    Ok(xlsynth_vastly::PipelineStimulus {
        half_period: 5,
        cycles,
    })
}

fn extract_output_from_vcd(vcd: &xlsynth_vastly::Vcd, out_name: &str) -> Result<Value4, String> {
    let candidates = [format!("tb.dut.{out_name}"), format!("tb.{out_name}")];
    let timeline = vcd
        .materialize()
        .map_err(|e| format!("materialize VCD failed: {e:?}"))?;
    let (_, last_values) = timeline
        .last()
        .ok_or_else(|| "VCD had no timestamped events".to_string())?;
    for candidate in &candidates {
        if let Some(bits) = last_values.get(candidate) {
            return value4_from_msb_bits(bits);
        }
    }
    Err(format!(
        "could not find output `{out_name}` in VCD; candidates={candidates:?}"
    ))
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
        let p = base.join(format!("vastly_fuzz_iv_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return Ok(p),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("create temp dir: {e}")),
        }
    }
    Err("failed to create temp dir".to_string())
}

fn assert_same_bits(
    kind: &str,
    want: &Value4,
    got: &Value4,
    ir_text: &str,
    src: &str,
    sample_data: &[u8],
    stimulus: &IrValue,
    stimulus_ordinal: &str,
) {
    if want.width != got.width || want.to_bit_string_msb_first() != got.to_bit_string_msb_first() {
        maybe_enable_backtrace_on_failure();
        emit_mismatch_context(ir_text, src);

        let mut msg = format!(
            "mismatch kind={kind} want={} got={} {} {} {}",
            summarize_value4("want", want),
            summarize_value4("got", got),
            summarize_ir(ir_text),
            summarize_generated(src),
            summarize_sample(sample_data)
        );
        msg.push(' ');
        msg.push_str(&summarize_stimulus_with_index(stimulus, stimulus_ordinal));
        append_verbose_backtrace(&mut msg);
        panic!("{msg}");
    }
}

fn strict_unsupported() -> bool {
    env_truthy("VASTLY_FUZZ_STRICT_UNSUPPORTED")
}

fn include_iverilog_oracle() -> bool {
    env_truthy("VASTLY_FUZZ_INCLUDE_IVERILOG_ORACLE")
}

fn include_stage1_pipeline_oracle() -> bool {
    env_bool_or("VASTLY_FUZZ_INCLUDE_PIPELINE_STAGE1_ORACLE", true)
}

fn include_stage2_pipeline_oracle() -> bool {
    env_bool_or("VASTLY_FUZZ_INCLUDE_PIPELINE_STAGE2_ORACLE", true)
}

fn use_autocov_stimuli() -> bool {
    env_truthy("VASTLY_FUZZ_USE_AUTOCOV")
}

fn unsupported(
    reason: &str,
    ir_text: &str,
    src: Option<&str>,
    detail: Option<&str>,
    sample_data: &[u8],
) {
    if !strict_unsupported() {
        return;
    }
    maybe_enable_backtrace_on_failure();
    let context = match src {
        Some(src) => format!("{} {}", summarize_ir(ir_text), summarize_generated(src)),
        None => summarize_ir(ir_text),
    };
    let mut msg = match detail {
        Some(detail) => format!(
            "unsupported reason={reason} {context} {} {}",
            summarize_detail(detail),
            summarize_sample(sample_data)
        ),
        None => format!(
            "unsupported reason={reason} {context} {}",
            summarize_sample(sample_data)
        ),
    };
    if verbose_failure_context() {
        msg.push_str("\n--- BEGIN FAILURE CONTEXT ---\n");
        msg.push_str("IR:\n");
        msg.push_str(ir_text);
        if let Some(src) = src {
            msg.push_str("\nGenerated:\n");
            msg.push_str(src);
        }
        if let Some(detail) = detail {
            msg.push_str("\nDetail:\n");
            msg.push_str(detail);
        }
        msg.push_str("\n--- END FAILURE CONTEXT ---");
    }
    append_verbose_backtrace(&mut msg);
    panic!("{msg}");
}

fn summarize_ir(text: &str) -> String {
    summarize_text("ir", text)
}

fn summarize_generated(text: &str) -> String {
    summarize_text("generated", text)
}

fn summarize_detail(text: &str) -> String {
    summarize_text("detail", text)
}

fn summarize_value4(label: &str, value: &Value4) -> String {
    let bits = value.to_bit_string_msb_first();
    format!(
        "{label}_w={} {label}_hash={:016x}",
        value.width,
        stable_hash_bytes(bits.as_bytes())
    )
}

fn summarize_text(label: &str, text: &str) -> String {
    format!(
        "{label}_hash={:016x} {label}_len={}",
        stable_hash_bytes(text.as_bytes()),
        text.len()
    )
}

fn summarize_sample(sample_data: &[u8]) -> String {
    format!(
        "sample_len={} sample_hex={}",
        sample_data.len(),
        hex_bytes(sample_data)
    )
}

fn summarize_panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn summarize_stimulus(stimulus: &IrValue) -> String {
    summarize_text("stimulus", &stimulus.to_string())
}

fn summarize_stimulus_with_index(stimulus: &IrValue, stimulus_ordinal: &str) -> String {
    format!(
        "stimulus_ordinal={} {}",
        stimulus_ordinal,
        summarize_stimulus(stimulus)
    )
}

fn has_sv_only_codegen_noise(ir_text: &str) -> bool {
    const NOISY_PATTERNS: [&str; 1] = ["= assert("];
    NOISY_PATTERNS.iter().any(|p| ir_text.contains(p))
}

fn quiet_xls_warnings() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("GLOG_minloglevel").is_none() {
            std::env::set_var("GLOG_minloglevel", "2");
        }
        if std::env::var_os("GLOG_stderrthreshold").is_none() {
            std::env::set_var("GLOG_stderrthreshold", "2");
        }
    });
}

fn verbose_failure_context() -> bool {
    env_truthy("VASTLY_FUZZ_VERBOSE_FAILURE")
}

fn verbose_backtrace() -> bool {
    env_truthy("VASTLY_FUZZ_VERBOSE_BACKTRACE")
}

fn append_verbose_backtrace(msg: &mut String) {
    if !verbose_backtrace() {
        return;
    }
    let bt = Backtrace::force_capture();
    msg.push_str("\n--- BEGIN BACKTRACE ---\n");
    msg.push_str(&bt.to_string());
    msg.push_str("--- END BACKTRACE ---");
}

fn emit_mismatch_context(ir_text: &str, generated_src: &str) {
    eprintln!("--- BEGIN FAILURE CONTEXT ---");
    eprintln!("IR:");
    eprintln!("{ir_text}");
    eprintln!("Generated:");
    eprintln!("{generated_src}");
    if verbose_backtrace() {
        eprintln!("--- BEGIN BACKTRACE ---");
        eprintln!("{}", Backtrace::force_capture());
        eprintln!("--- END BACKTRACE ---");
    }
    eprintln!("--- END FAILURE CONTEXT ---");
}

#[derive(Clone, Default)]
struct PanicContext {
    sample_hex: String,
    sample_len: usize,
    ir_text: Option<String>,
    verilog_src: Option<String>,
    sv_src: Option<String>,
    stimulus_text: Option<String>,
}

std::thread_local! {
    static PANIC_CONTEXT: RefCell<Option<PanicContext>> = const { RefCell::new(None) };
}

struct PanicContextGuard;

impl Drop for PanicContextGuard {
    fn drop(&mut self) {
        PANIC_CONTEXT.with(|slot| {
            *slot.borrow_mut() = None;
        });
    }
}

fn begin_panic_context(sample_data: &[u8]) -> PanicContextGuard {
    PANIC_CONTEXT.with(|slot| {
        *slot.borrow_mut() = Some(PanicContext {
            sample_hex: hex_bytes(sample_data),
            sample_len: sample_data.len(),
            ir_text: None,
            verilog_src: None,
            sv_src: None,
            stimulus_text: None,
        });
    });
    PanicContextGuard
}

fn set_panic_context_ir(ir_text: &str) {
    PANIC_CONTEXT.with(|slot| {
        if let Some(ctx) = slot.borrow_mut().as_mut() {
            ctx.ir_text = Some(ir_text.to_string());
        }
    });
}

fn set_panic_context_verilog(verilog_src: &str) {
    PANIC_CONTEXT.with(|slot| {
        if let Some(ctx) = slot.borrow_mut().as_mut() {
            ctx.verilog_src = Some(verilog_src.to_string());
        }
    });
}

fn set_panic_context_sv(sv_src: &str) {
    PANIC_CONTEXT.with(|slot| {
        if let Some(ctx) = slot.borrow_mut().as_mut() {
            ctx.sv_src = Some(sv_src.to_string());
        }
    });
}

fn set_panic_context_stimulus(stimulus_text: &str) {
    PANIC_CONTEXT.with(|slot| {
        if let Some(ctx) = slot.borrow_mut().as_mut() {
            ctx.stimulus_text = Some(stimulus_text.to_string());
        }
    });
}

fn install_panic_context_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic_info| {
            if verbose_failure_context() {
                PANIC_CONTEXT.with(|slot| {
                    if let Some(ctx) = slot.borrow().as_ref() {
                        eprintln!("--- BEGIN PANIC CONTEXT ---");
                        eprintln!("sample_len={}", ctx.sample_len);
                        eprintln!("sample_hex={}", ctx.sample_hex);
                        if let Some(ir_text) = &ctx.ir_text {
                            eprintln!("IR:");
                            eprintln!("{ir_text}");
                        }
                        if let Some(verilog_src) = &ctx.verilog_src {
                            eprintln!("Generated Verilog:");
                            eprintln!("{verilog_src}");
                        }
                        if let Some(sv_src) = &ctx.sv_src {
                            eprintln!("Generated SystemVerilog:");
                            eprintln!("{sv_src}");
                        }
                        if let Some(stimulus_text) = &ctx.stimulus_text {
                            eprintln!("Stimulus:");
                            eprintln!("{stimulus_text}");
                        }
                        eprintln!("--- END PANIC CONTEXT ---");
                    }
                });
            }
            prev(panic_info);
        }));
    });
}

fn maybe_enable_backtrace_on_failure() {
    if env_truthy("VASTLY_FUZZ_BACKTRACE_ON_FAILURE") {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
}

fn env_truthy(key: &str) -> bool {
    match std::env::var(key) {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        }
        Err(_) => false,
    }
}

fn env_bool_or(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => {
            let v = v.trim();
            if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes") {
                true
            } else if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no") {
                false
            } else {
                default
            }
        }
        Err(_) => default,
    }
}

fn env_u64_or(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_usize_or(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn stable_hash_bytes(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
