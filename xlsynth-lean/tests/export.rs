// SPDX-License-Identifier: Apache-2.0

mod common;

use xlsynth_g8r::aig::{AigBitVector, AigNode, AigOperand, AigRef, GateFn};
use xlsynth_lean::{LeanError, LeanExportOptions, LeanModuleName, export_gate_fn};

#[test]
fn export_matches_golden_files() {
    let options = LeanExportOptions {
        module_name: LeanModuleName::new("Generated.And3").unwrap(),
    };
    let bundle = export_gate_fn(&common::and3(), &options).unwrap();
    let input_ports = bundle
        .interface()
        .inputs()
        .iter()
        .map(|port| (port.name.as_str(), port.bit_count))
        .collect::<Vec<_>>();
    assert_eq!(input_ports, vec![("a", 1), ("b", 1), ("c", 1)]);
    let output_ports = bundle
        .interface()
        .outputs()
        .iter()
        .map(|port| (port.name.as_str(), port.bit_count))
        .collect::<Vec<_>>();
    assert_eq!(output_ports, vec![("result", 1)]);
    pretty_assertions::assert_eq!(bundle.lean_source(), include_str!("goldens/and3.lean"));
    pretty_assertions::assert_eq!(bundle.aag_source(), include_str!("goldens/and3.aag"));
    assert_eq!(
        bundle.aag_relative_path(),
        std::path::PathBuf::from("Generated/And3.aag")
    );
    let repeated = export_gate_fn(&common::and3(), &options).unwrap();
    assert_eq!(bundle.lean_source(), repeated.lean_source());
    assert_eq!(bundle.aag_source(), repeated.aag_source());

    let directory = tempfile::tempdir().unwrap();
    let written = bundle.write_to(directory.path()).unwrap();
    assert_eq!(
        std::fs::read_to_string(written.source).unwrap(),
        bundle.lean_source()
    );
    assert_eq!(
        std::fs::read_to_string(written.aag).unwrap(),
        bundle.aag_source()
    );
}

#[test]
fn accepts_empty_graph_and_non_topological_storage() {
    let empty = GateFn {
        name: "empty".to_string(),
        inputs: vec![],
        outputs: vec![],
        gates: vec![],
    };
    let bundle = export_gate_fn(&empty, &LeanExportOptions::default()).unwrap();
    assert_eq!(bundle.interface().input_bit_count(), 0);
    assert_eq!(bundle.interface().output_bit_count(), 0);
    let graph = common::interface_graph();
    let bundle = export_gate_fn(&graph, &LeanExportOptions::default()).unwrap();
    assert_eq!(bundle.interface().input_bit_count(), 69);
    assert_eq!(bundle.interface().output_bit_count(), 72);
}

#[test]
fn rejects_invalid_module_names() {
    for name in [
        "",
        ".A",
        "A.",
        "A..B",
        "../A",
        "A/B",
        "A-B",
        "9A",
        "A\nB",
        "Init",
        "Lean.Custom",
        "std.Custom",
        "Lake",
        "XlsynthLean",
        "XlsynthLean.Aig",
        "xlsynthlean.Custom",
        "EvalMain",
        "evalmain",
        "EvalMain.Child",
    ] {
        assert!(matches!(
            LeanModuleName::new(name),
            Err(LeanError::InvalidModuleName(_))
        ));
    }
}

#[test]
fn rejects_invalid_references_and_cycles() {
    let invalid = AigOperand {
        node: AigRef { id: 999 },
        negated: false,
    };
    let mut invalid_input = common::and3();
    invalid_input.inputs[0].bit_vector = AigBitVector::from_bit(invalid);
    let mut invalid_output = common::and3();
    invalid_output.outputs[0].bit_vector = AigBitVector::from_bit(invalid);
    let mut invalid_and = common::and3();
    if let AigNode::And2 { a, .. } = &mut invalid_and.gates[4] {
        *a = invalid;
    }
    let mut cycle = common::and3();
    if let AigNode::And2 { a, .. } = &mut cycle.gates[4] {
        a.node.id = 5;
    }
    for graph in [invalid_input, invalid_output, invalid_and, cycle] {
        assert!(matches!(
            export_gate_fn(&graph, &LeanExportOptions::default()),
            Err(LeanError::InvalidGraph(_))
        ));
    }
}

#[test]
fn rejects_ambiguous_input_bindings() {
    let mut negated = common::and3();
    let operand = negated.inputs[0].bit_vector.get_lsb(0).negate();
    negated.inputs[0].bit_vector = AigBitVector::from_bit(operand);
    let mut duplicate = common::and3();
    duplicate.inputs[1].bit_vector = duplicate.inputs[0].bit_vector.clone();
    let mut literal = common::and3();
    literal.inputs[0].bit_vector = AigBitVector::from_bit(AigOperand {
        node: AigRef { id: 0 },
        negated: false,
    });
    let mut hidden = common::and3();
    hidden.gates.push(AigNode::Input {
        name: "hidden".to_string(),
        lsb_index: 0,
        pir_node_ids: Default::default(),
    });
    for graph in [negated, duplicate, literal, hidden] {
        assert!(matches!(
            export_gate_fn(&graph, &LeanExportOptions::default()),
            Err(LeanError::InvalidGraph(_))
        ));
    }
}
