// SPDX-License-Identifier: Apache-2.0

mod common;

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use xlsynth_g8r::aig::{AigBitVector, GateFn};
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_ir_value::IrBits;
use xlsynth_lean::{
    LeanError, LeanEvaluator, LeanExportOptions, LeanModuleName, export_gate_fn, write_lean_support,
};
use xlsynth_pir::{IrValue, ir_parser::Parser};

static EVALUATOR: OnceLock<LeanEvaluator> = OnceLock::new();

/// Initializes one evaluator for all explicitly enabled Lean integration tests.
fn evaluator() -> &'static LeanEvaluator {
    EVALUATOR.get_or_init(|| {
        let executable = std::env::var_os("XLSYNTH_LEAN")
            .expect("set XLSYNTH_LEAN to the actual Lean executable path");
        LeanEvaluator::new(PathBuf::from(executable)).unwrap()
    })
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn block_trace_observes_old_state_and_carries_simultaneous_writes() {
    // Exercise positional trace construction with a generated-name collision
    // and nonalphabetical declaration order in both input and state records.
    let ir = include_str!("fixtures/block_step.ir")
        .replace("rst", "recOn")
        .replace("reg a(", "reg noConfusion(")
        .replace("register=a", "register=noConfusion");
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let block = package.get_top_block().unwrap();
    let word = |n| IrValue::make_ubits(2, n).unwrap();
    let bit = |n| IrValue::make_ubits(1, n).unwrap();
    // Starting at (2, 3), swap, reset a with enable low, then hold a.
    // b samples old a on all three clocks, including the reset clock.
    let inputs = vec![
        vec![bit(1), bit(1)],
        vec![bit(0), bit(0)],
        vec![bit(1), bit(0)],
    ];
    let trace = evaluator()
        .eval_block_trace(&package, block, &[word(2), word(3)], &inputs)
        .unwrap();
    let expected = [(2, [3, 2]), (3, [1, 3]), (1, [1, 1])];
    assert_eq!(trace.len(), expected.len());
    for (cycle, (output, next)) in trace.iter().zip(expected) {
        assert_eq!(cycle.outputs, [word(output)]);
        assert_eq!(cycle.next_state, next.map(word));
    }
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn block_trace_preserves_bits_above_the_machine_word() {
    let ir = include_str!("fixtures/block_step.ir").replace("bits[2]", "bits[65]");
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let high = IrValue::parse_typed("bits[65]:0x10000000000000000").unwrap();
    let low = IrValue::parse_typed("bits[65]:1").unwrap();
    let enabled = IrValue::make_ubits(1, 1).unwrap();
    let trace = evaluator()
        .eval_block_trace(
            &package,
            package.get_top_block().unwrap(),
            &[high.clone(), low.clone()],
            &[vec![enabled.clone(), enabled]],
        )
        .unwrap();
    assert_eq!(trace[0].outputs, std::slice::from_ref(&high));
    assert_eq!(trace[0].next_state, [low, high]);
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn evaluates_wide_and_non_topological_graph_like_g8r() {
    let graph = common::interface_graph();
    let widths = graph
        .inputs
        .iter()
        .map(|input| input.get_bit_count())
        .collect::<Vec<_>>();
    let mut cases = vec![
        widths
            .iter()
            .map(|width| IrBits::zero(*width))
            .collect::<Vec<_>>(),
        widths
            .iter()
            .map(|width| IrBits::all_ones(*width))
            .collect::<Vec<_>>(),
    ];
    for (active_port, width) in widths.iter().copied().enumerate() {
        for active_bit in 0..width {
            cases.push(
                widths
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(port, width)| {
                        IrBits::from_lsb_fn(width, |bit| port == active_port && bit == active_bit)
                    })
                    .collect(),
            );
        }
    }
    let mut simulator = PreparedGateSim::new(&graph);
    let expected = cases
        .iter()
        .map(|inputs| simulator.eval_outputs(inputs))
        .collect::<Vec<_>>();
    let options = LeanExportOptions {
        module_name: LeanModuleName::new("Generated.def").unwrap(),
    };
    let bundle = export_gate_fn(&graph, &options).unwrap();
    let actual = evaluator().eval_circuit_batch(&bundle, &cases).unwrap();
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn evaluates_empty_interfaces_and_batches() {
    let graph = GateFn {
        name: "empty".to_string(),
        inputs: vec![],
        outputs: vec![],
        gates: vec![],
    };
    assert_eq!(
        evaluator().eval_gate_fn(&graph, &[]).unwrap(),
        Vec::<IrBits>::new()
    );
    assert_eq!(
        evaluator().eval_gate_fn_batch(&graph, &[]).unwrap(),
        Vec::<Vec<IrBits>>::new()
    );
    assert_eq!(
        evaluator()
            .eval_gate_fn_batch(&graph, &[vec![], vec![]])
            .unwrap(),
        vec![Vec::<IrBits>::new(), Vec::<IrBits>::new()]
    );

    let mut builder = GateBuilder::new("zero_width".to_string(), GateBuilderOptions::no_opt());
    builder.add_input("input".to_string(), 0);
    builder.add_output("output".to_string(), AigBitVector::from_lsb_is_index_0(&[]));
    assert_eq!(
        evaluator()
            .eval_gate_fn(&builder.build(), &[IrBits::zero(0)])
            .unwrap(),
        vec![IrBits::zero(0)]
    );
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn rejects_input_shape_mismatches() {
    let graph = common::and3();
    assert!(matches!(
        evaluator().eval_gate_fn(&graph, &[]),
        Err(LeanError::InputCount {
            expected: 3,
            actual: 0,
            ..
        })
    ));
    assert!(matches!(
        evaluator().eval_gate_fn(&graph, &[IrBits::zero(1), IrBits::zero(2), IrBits::zero(1)]),
        Err(LeanError::InputWidth {
            input_index: 1,
            expected: 1,
            actual: 2,
            ..
        })
    ));
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn exported_module_and_lean_regressions_compile() {
    let options = LeanExportOptions {
        module_name: LeanModuleName::new("Generated.And3").unwrap(),
    };
    let bundle = export_gate_fn(&common::and3(), &options).unwrap();
    let directory = tempfile::tempdir().unwrap();
    write_lean_support(directory.path()).unwrap();
    let written = bundle.write_to(directory.path()).unwrap();
    let regression_source = directory.path().join("Tests.lean");
    std::fs::write(&regression_source, include_str!("../lean/Tests.lean")).unwrap();
    for source in [
        directory.path().join("XlsynthLean/Aig.lean"),
        directory.path().join("XlsynthLean/Aag.lean"),
        directory.path().join("XlsynthLean/IR.lean"),
        directory.path().join("XlsynthLean.lean"),
        written.source,
        regression_source,
    ] {
        let output = Command::new(evaluator().lean_executable())
            .arg("-o")
            .arg(source.with_extension("olean"))
            .arg(&source)
            .current_dir(directory.path())
            .env("LEAN_PATH", directory.path())
            .env_remove("LEAN_SYSROOT")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "Lean failed for {}:\n{}\n{}",
            source.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
