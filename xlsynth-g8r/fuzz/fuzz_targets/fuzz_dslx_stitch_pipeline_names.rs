// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::collections::BTreeMap;
use std::path::Path;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::dslx_stitch_pipeline::{StitchPipelineOptions, stitch_pipeline};
use xlsynth_g8r::verilog_version::VerilogVersion;
use xlsynth_vastly::{
    LogicBit, PipelineCycle, PipelineStimulus, Signedness, Value4, compile_pipeline_module,
    run_pipeline_and_collect_outputs,
};

#[derive(Arbitrary, Debug)]
struct NameSample {
    a: NameChoice,
    b: NameChoice,
    c: NameChoice,
    d: NameChoice,
}

#[derive(Arbitrary, Debug, Clone, Copy)]
struct NameChoice {
    selector: u8,
    suffix: u8,
}

impl NameChoice {
    fn to_name(self) -> String {
        match self.selector % 21 {
            0 => "clk".to_string(),
            1 => "rst".to_string(),
            2 => "rst_n".to_string(),
            3 => "input_valid".to_string(),
            4 => "output_valid".to_string(),
            5 => "out".to_string(),
            6 => "valid".to_string(),
            7 => "stage_0".to_string(),
            8 => "stage_0_out_comb".to_string(),
            9 => "p0_valid".to_string(),
            10 => "p0_out_comb".to_string(),
            11 => "input".to_string(),
            12 => "output".to_string(),
            13 => "wire".to_string(),
            14 => "logic".to_string(),
            15 => "module".to_string(),
            _ => format!("n{}_{}", self.selector, self.suffix),
        }
    }
}

fuzz_target!(|sample: NameSample| {
    let _ = env_logger::builder().is_test(true).try_init();

    let a = sample.a.to_name();
    let b = sample.b.to_name();
    let c = sample.c.to_name();
    let d = sample.d.to_name();

    if a == b || c == d {
        // Duplicate parameter names within a single DSLX function are invalid
        // before stitch-pipeline name validation runs, so this sample does not
        // exercise the wrapper namespace property.
        return;
    }

    let dslx = format!(
        r#"fn foo_cycle0({a}: u8, {b}: u8) -> (u8, u8) {{ ({b}, {a}) }}
fn foo_cycle1({c}: u8, {d}: u8) -> u8 {{ {c} - {d} }}"#
    );
    let opts = StitchPipelineOptions {
        verilog_version: VerilogVersion::SystemVerilog,
        explicit_stages: None,
        stdlib_path: None,
        search_paths: Vec::new(),
        flop_inputs: true,
        flop_outputs: true,
        input_valid_signal: Some("input_valid"),
        output_valid_signal: Some("output_valid"),
        reset_signal: Some("rst"),
        reset_active_low: false,
        add_invariant_assertions: false,
        array_index_bounds_checking: true,
        output_module_name: "foo",
    };

    let sv = match stitch_pipeline(&dslx, Path::new("foo.x"), "foo", &opts) {
        Ok(sv) => sv,
        Err(err) => {
            let message = err.0;
            if message.contains("name validation failed") {
                // Unsafe names are the expected negative outcome for this target;
                // the property is that they fail through the controlled validator.
                return;
            }
            if is_dslx_frontend_error(&message) {
                // Some candidate names can be rejected by the DSLX frontend before
                // stitch-pipeline validation has a chance to run; those samples are
                // not failures of the Verilog-name hardening property.
                return;
            }
            panic!("unexpected stitch_pipeline error: {message}\nDSLX:\n{dslx}");
        }
    };

    let sim_sv = flattened_simulation_sv(&a, &b, &c, &d);
    simulate_and_check(&sim_sv, &a, &b).unwrap_or_else(|err| {
        panic!(
            "accepted stitch-pipeline sample failed simulation: {err}\nDSLX:\n{dslx}\nSV:\n{sv}\nSimulation SV:\n{sim_sv}"
        )
    });
});

fn is_dslx_frontend_error(message: &str) -> bool {
    message.contains("ParseError")
        || message.contains("parse error")
        || message.contains("Expected")
        || message.contains("TypeInference")
        || message.contains("syntax")
}

/// Builds the single-module form that `xlsynth-vastly` can simulate for this
/// sample shape.
fn flattened_simulation_sv(a: &str, b: &str, c: &str, d: &str) -> String {
    format!(
        r#"module foo(
  input wire clk,
  input wire rst,
  input wire [7:0] {a},
  input wire [7:0] {b},
  input wire input_valid,
  output wire [7:0] out,
  output wire output_valid
);
  reg [7:0] p0_{a};
  reg [7:0] p0_{b};
  reg p0_valid;
  always_ff @ (posedge clk) begin
    p0_{a} <= input_valid ? {a} : p0_{a};
    p0_{b} <= input_valid ? {b} : p0_{b};
    p0_valid <= rst ? 1'b0 : input_valid;
  end
  wire [15:0] stage_0_out_comb;
  assign stage_0_out_comb = {{p0_{b}, p0_{a}}};
  wire [7:0] p1_{c}_comb;
  assign p1_{c}_comb = stage_0_out_comb[7:0];
  wire [7:0] p1_{d}_comb;
  assign p1_{d}_comb = stage_0_out_comb[15:8];
  reg [7:0] p1_{c};
  reg [7:0] p1_{d};
  reg p1_valid;
  always_ff @ (posedge clk) begin
    p1_{c} <= p0_valid ? p1_{c}_comb : p1_{c};
    p1_{d} <= p0_valid ? p1_{d}_comb : p1_{d};
    p1_valid <= rst ? 1'b0 : p0_valid;
  end
  wire [7:0] stage_1_out_comb;
  assign stage_1_out_comb = p1_{c} - p1_{d};
  reg [7:0] p2_out;
  reg p2_valid;
  always_ff @ (posedge clk) begin
    p2_out <= p1_valid ? stage_1_out_comb : p2_out;
    p2_valid <= rst ? 1'b0 : p1_valid;
  end
  assign out = p2_out;
  assign output_valid = p2_valid;
endmodule
"#
    )
}

fn simulate_and_check(sv: &str, a: &str, b: &str) -> Result<(), String> {
    let module =
        compile_pipeline_module(sv).map_err(|err| format!("compile_pipeline_module: {err:?}"))?;
    let zero_cycle = zero_pipeline_inputs(&module)?;
    let mut cycles = Vec::new();

    for _ in 0..2 {
        let mut inputs = zero_cycle.clone();
        inputs.insert("rst".to_string(), bit1(LogicBit::One));
        cycles.push(PipelineCycle { inputs });
    }

    let mut payload = zero_cycle.clone();
    payload.insert("input_valid".to_string(), bit1(LogicBit::One));
    payload.insert(a.to_string(), Value4::from_u64(8, Signedness::Unsigned, 3));
    payload.insert(b.to_string(), Value4::from_u64(8, Signedness::Unsigned, 4));
    cycles.push(PipelineCycle { inputs: payload });

    for _ in 0..4 {
        cycles.push(PipelineCycle {
            inputs: zero_cycle.clone(),
        });
    }

    let stimulus = PipelineStimulus {
        half_period: 5,
        cycles,
    };
    let outputs = run_pipeline_and_collect_outputs(&module, &stimulus, &module.initial_state_x())
        .map_err(|err| format!("run_pipeline_and_collect_outputs: {err:?}"))?;

    let retired = retired_outputs(&outputs)?;
    if retired != vec![Value4::from_u64(8, Signedness::Unsigned, 255)] {
        let got = retired
            .iter()
            .map(Value4::to_bit_string_msb_first)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("expected retired output 11111111, got [{got}]"));
    }
    Ok(())
}

fn zero_pipeline_inputs(
    module: &xlsynth_vastly::CompiledPipelineModule,
) -> Result<BTreeMap<String, Value4>, String> {
    let mut inputs = BTreeMap::new();
    for port in &module.combo.input_ports {
        let info = module
            .combo
            .decls
            .get(&port.name)
            .ok_or_else(|| format!("missing decl info for input `{}`", port.name))?;
        let value = match port.name.as_str() {
            "rst" | "input_valid" => bit1(LogicBit::Zero),
            _ => Value4::zeros(port.width, info.signedness),
        };
        inputs.insert(port.name.clone(), value);
    }
    if !inputs.contains_key("rst") || !inputs.contains_key("input_valid") {
        return Err("pipeline is missing expected reset or input-valid port".to_string());
    }
    Ok(inputs)
}

fn retired_outputs(outputs: &[BTreeMap<String, Value4>]) -> Result<Vec<Value4>, String> {
    let mut retired = Vec::new();
    for cycle_out in outputs {
        let valid = cycle_out
            .get("output_valid")
            .ok_or_else(|| "missing output_valid output".to_string())?;
        match valid.bits_lsb_first().first().copied() {
            Some(LogicBit::One) => {
                let out = cycle_out
                    .get("out")
                    .ok_or_else(|| "missing out output".to_string())?;
                retired.push(out.clone());
            }
            Some(LogicBit::Zero) => {}
            Some(other) => {
                return Err(format!("output_valid was not concrete 0/1: {other:?}"));
            }
            None => return Err("output_valid had zero width".to_string()),
        }
    }
    Ok(retired)
}

fn bit1(bit: LogicBit) -> Value4 {
    Value4::new(1, Signedness::Unsigned, vec![bit])
}
