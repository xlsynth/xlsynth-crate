// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, load_sequential_gate_fn_from_path};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};

fn linear_and4_design() -> SequentialGateFn {
    let mut builder = GateBuilder::new("and4".to_string(), GateBuilderOptions::opt());
    let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
    let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
    let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
    let d = *builder.add_input("d".to_string(), 1).get_lsb(0);
    let ab = builder.add_and_binary(a, b);
    let abc = builder.add_and_binary(ab, c);
    let abcd = builder.add_and_binary(abc, d);
    builder.add_output("o".to_string(), abcd.into());
    SequentialGateFn::from_gate_fn(builder.build())
}

#[test]
fn g8r_optimize_runs_reassociation_in_isolation() {
    let temp = tempfile::tempdir().expect("create temporary directory");
    let input_path = temp.path().join("input.g8r");
    let output_path = temp.path().join("output.g8rbin");
    let stats_path = temp.path().join("stats.json");
    std::fs::write(&input_path, emit_g8r(&linear_and4_design())).expect("write input g8r");

    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("g8r-optimize")
        .arg(&input_path)
        .arg("--fraig=false")
        .arg("--reassociation=true")
        .arg("--cut-db-rewrite=false")
        .arg("--quiet=true")
        .arg("--bin-out")
        .arg(&output_path)
        .arg("--stats-out")
        .arg(&stats_path)
        .output()
        .expect("run g8r-optimize");
    assert!(
        output.status.success(),
        "g8r-optimize failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());

    let optimized = load_sequential_gate_fn_from_path(&output_path)
        .expect("load optimized g8r")
        .try_into_gate_fn()
        .expect("optimized design should be combinational");
    let optimized_stats = get_aig_stats(&optimized);
    assert_eq!(optimized_stats.and_nodes, 3);
    assert_eq!(optimized_stats.max_depth, 2);

    let stats: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&stats_path).expect("read g8r-optimize stats"))
            .expect("parse g8r-optimize stats");
    assert_eq!(
        stats,
        serde_json::json!({
            "input": {"and_nodes": 3, "levels": 3},
            "output": {"and_nodes": 3, "levels": 2},
            "fraig_pass_stat": null
        })
    );
}
