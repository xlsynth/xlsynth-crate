// SPDX-License-Identifier: Apache-2.0

//! Measures functional trace simulation and sequential power replay separately
//! on a two-stage register bank. Model loading and input generation are
//! untimed.

use std::fmt::Write;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig_sim::sequential::SequentialState;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, load_labeled_sequential_netlist_aig_with_liberty,
};
use xlsynth_g8r::netlist::io::load_liberty_from_path;
use xlsynth_g8r::netlist::power::GvSequentialDynamicPowerOptions;
use xlsynth_pir::IrBits;

const WIDTH: usize = 128;
const CYCLES: usize = 257;

const DFF_LIBERTY: &str = r#"
format_magic: 5496997758228255311
nominal_voltage: 1.0
interned_strings: ["D", "CLK", "Q", "IQ"]
lut_shapes: {}
cells: {
  name: "DFF"
  pins: { name_string_id: 1 direction: INPUT capacitance: 1.0 }
  pins: {
    name_string_id: 2 direction: INPUT is_clocking_pin: true capacitance: 1.0
    internal_power: {
      when_string_id: 4
      tables: { transition: POWER_TRANSITION_RISE shape_id: 1 values: 1.0 }
      tables: { transition: POWER_TRANSITION_FALL shape_id: 1 values: 1.0 }
    }
  }
  pins: {
    name_string_id: 3 direction: OUTPUT function_string_id: 4
    timing_arcs: {
      related_pin_string_id: 2 timing_type: TIMING_TYPE_RISING_EDGE
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 1.0 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 1.0 }
    }
  }
  sequential: { state_var: "IQ" next_state: "D" clock_expr: "CLK" kind: SEQUENTIAL_KIND_FF }
}
"#;

/// Builds a mapped pipeline with one DFF per bit per stage.
fn register_bank() -> String {
    let msb = WIDTH - 1;
    let mut netlist = format!(
        "module top (clk, d, q);\ninput clk;\ninput [{msb}:0] d;\nwire [{msb}:0] stage;\noutput [{msb}:0] q;\n"
    );
    for bit in 0..WIDTH {
        writeln!(
            netlist,
            "DFF first_{bit} (.CLK(clk), .D(d[{bit}]), .Q(stage[{bit}]));"
        )
        .unwrap();
        writeln!(
            netlist,
            "DFF second_{bit} (.CLK(clk), .D(stage[{bit}]), .Q(q[{bit}]));"
        )
        .unwrap();
    }
    netlist.push_str("endmodule\n");
    netlist
}

/// Times the two library calls made by sequential gv-eval with power enabled.
fn bench_sequential_gv_eval(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let netlist_path = dir.path().join("register_bank.gv");
    let liberty_path = dir.path().join("dff.textproto");
    std::fs::write(&netlist_path, register_bank()).unwrap();
    std::fs::write(&liberty_path, DFF_LIBERTY).unwrap();
    let library = load_liberty_from_path(&liberty_path).unwrap();
    let model = load_labeled_sequential_netlist_aig_with_liberty(
        &netlist_path,
        &library,
        &GvEvalOptions::default(),
    )
    .unwrap();
    let state = SequentialState::all_zeros(&model.sequential_gate_fn);
    let mut rng = StdRng::seed_from_u64(1);
    let inputs: Vec<_> = (0..CYCLES)
        .map(|_| {
            let mut bytes = [0; WIDTH.div_ceil(8)];
            rng.fill_bytes(&mut bytes);
            vec![IrBits::from_le_bytes(WIDTH, &bytes).unwrap()]
        })
        .collect();

    let mut group = c.benchmark_group("gv_eval_sequential");
    group.throughput(Throughput::Elements(CYCLES as u64));
    group.bench_function("simulate_bits", |b| {
        b.iter_batched(
            || state.clone(),
            |initial_state| {
                black_box(
                    model
                        .simulate_bits(black_box(&inputs), initial_state)
                        .unwrap(),
                );
            },
            BatchSize::PerIteration,
        );
    });
    let trace = model.simulate_bits(&inputs, state).unwrap();
    assert_eq!(&trace.external_outputs()[2..], &inputs[..CYCLES - 2]);
    group.bench_function("analyze_dynamic_power", |b| {
        b.iter(|| {
            black_box(
                model
                    .analyze_dynamic_power(
                        &library,
                        black_box(&trace),
                        GvSequentialDynamicPowerOptions::default(),
                    )
                    .unwrap(),
            );
        });
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bench_sequential_gv_eval
}
criterion_main!(benches);
