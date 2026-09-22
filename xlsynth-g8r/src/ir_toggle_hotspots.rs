// SPDX-License-Identifier: Apache-2.0

//! Ordered IR and AIG switching activity from a shared `.irvals` stimulus.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use xlsynth_pir::ir::{self, NodePayload, NodeRef};
use xlsynth_pir::ir_eval::{self, EvalObserver, FnEvalResult, SelectEvent};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;
use xlsynth_pir::{IrBits, IrValue};

use crate::aig_sim::count_toggles::{ToggleNodeKind, count_toggle_activity};
use crate::aig_sim::gate_simd::validate_ordered_batch_inputs;
use crate::gatify::ir2gate::{GatifyOptions, gatify};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IrNodeLabel {
    pub id: usize,
    pub name: String,
    pub op: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WordActivity {
    pub ir_node: IrNodeLabel,
    /// Transitions on which at least one bit of this node's value changed.
    pub word_toggles: usize,
    /// Sum of individual bit transitions, including tuple and array members.
    pub bit_toggles: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IrOutputBit {
    pub ir_node: IrNodeLabel,
    /// Least-significant-bit-first index in the flattened IR value.
    pub bit_index: usize,
    /// Whether the IR bit is the inversion of the AIG node's output.
    pub inverted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GateActivity {
    pub node_id: usize,
    pub toggle_count: usize,
    /// IR nodes credited with creating or sharing this AIG node. May be empty.
    pub sources: Vec<IrNodeLabel>,
    /// IR output bits whose lowered signal is precisely this AIG node (possibly
    /// inverted).
    pub ir_output_bits: Vec<IrOutputBit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToggleHotspots {
    pub function: String,
    pub sample_count: usize,
    pub transition_count: usize,
    /// Ranked by word toggles, bit toggles, and then IR node ID.
    pub words: Vec<WordActivity>,
    /// Output-reachable AND2 nodes ranked by toggles, then AIG node ID.
    pub gates: Vec<GateActivity>,
}

fn label(node: &ir::Node) -> IrNodeLabel {
    IrNodeLabel {
        id: node.text_id,
        name: node
            .name
            .clone()
            .unwrap_or_else(|| format!("{}.{}", node.payload.get_operator(), node.text_id)),
        op: node.payload.get_operator().to_string(),
    }
}

struct WordObserver<'a> {
    function: &'a ir::Fn,
    previous: Vec<Option<Vec<bool>>>,
    word_toggles: Vec<usize>,
    bit_toggles: Vec<usize>,
}

impl WordObserver<'_> {
    fn record(&mut self, node_ref: NodeRef, value: &IrValue) -> Result<(), String> {
        let node = self.function.get_node(node_ref);
        let mut bits = Vec::with_capacity(node.ty.bit_count());
        flatten_ir_value_to_lsb0_bits_for_type(value, &node.ty, &mut bits)?;
        if let Some(previous) = &self.previous[node_ref.index] {
            let changes = bits.iter().zip(previous).filter(|(a, b)| a != b).count();
            self.bit_toggles[node_ref.index] += changes;
            self.word_toggles[node_ref.index] += usize::from(changes != 0);
        }
        self.previous[node_ref.index] = Some(bits);
        Ok(())
    }
}

impl EvalObserver for WordObserver<'_> {
    fn on_select(&mut self, _ev: SelectEvent) {
        // Node output values already capture selection-related switching.
    }

    fn on_node_value(&mut self, node_ref: NodeRef, _node_text_id: usize, value: &IrValue) {
        // The evaluator has already checked each node's declared type.
        self.record(node_ref, value)
            .expect("evaluated IR value must match its node type");
    }
}

/// Loads named or positional records and validates them against a function
/// signature.
pub fn load_irvals_samples(path: &Path, function: &ir::Fn) -> Result<Vec<IrValue>, String> {
    let param_names = function
        .param_nodes()
        .map(|param| param.param_name().to_string())
        .collect::<Vec<_>>();
    let samples = xlsynth_pir::parse_ir_values_file(path)
        .map_err(|e| e.to_string())?
        .into_positional_values(&param_names)
        .map_err(|e| e.to_string())?;
    let expected = ir::Type::Tuple(
        function
            .param_nodes()
            .map(|param| Box::new(param.ty.clone()))
            .collect(),
    );
    for (index, sample) in samples.iter().enumerate() {
        if sample.type_() != expected {
            return Err(format!(
                "sample {} type mismatch: expected {}, got {}",
                index + 1,
                expected,
                sample.type_()
            ));
        }
    }
    Ok(samples)
}

/// Reports switching in the selected IR function and its AIG lowering.
///
/// Samples are ordered tuple values, one per cycle; transitions between
/// adjacent samples are counted. Invokes and loops must be inlined before
/// analysis because the evaluator's node observer does not identify nested
/// functions.
pub fn analyze_toggle_hotspots(
    package: &ir::Package,
    function: &ir::Fn,
    samples: &[IrValue],
) -> Result<ToggleHotspots, String> {
    if samples.len() < 2 {
        return Err(format!(
            "toggle stimulus must contain at least two samples; got {}",
            samples.len()
        ));
    }
    if function.nodes.iter().any(|node| {
        matches!(
            node.payload,
            NodePayload::Invoke { .. } | NodePayload::CountedFor { .. }
        )
    }) {
        return Err("inline invokes and counted_for loops before analyzing toggles".to_string());
    }

    let expected = ir::Type::Tuple(
        function
            .param_nodes()
            .map(|param| Box::new(param.ty.clone()))
            .collect(),
    );
    let mut observer = WordObserver {
        function,
        previous: vec![None; function.nodes.len()],
        word_toggles: vec![0; function.nodes.len()],
        bit_toggles: vec![0; function.nodes.len()],
    };
    let mut gate_inputs: Vec<Vec<IrBits>> = Vec::with_capacity(samples.len());
    for (sample_index, sample) in samples.iter().enumerate() {
        if sample.type_() != expected {
            return Err(format!(
                "sample {} type mismatch: expected {}, got {}",
                sample_index + 1,
                expected,
                sample.type_()
            ));
        }
        let args = sample.get_elements().map_err(|e| e.to_string())?;
        let mut input_bits = Vec::with_capacity(args.len());
        for ((&node_ref, param), arg) in function
            .params
            .iter()
            .zip(function.param_nodes())
            .zip(&args)
        {
            observer.record(node_ref, arg)?;
            let mut flat = Vec::with_capacity(param.ty.bit_count());
            flatten_ir_value_to_lsb0_bits_for_type(arg, &param.ty, &mut flat)?;
            input_bits.push(IrBits::from_lsb_is_0(&flat));
        }
        gate_inputs.push(input_bits);
        match ir_eval::eval_fn_in_package_with_observer(
            package,
            function,
            &args,
            Some(&mut observer),
        ) {
            FnEvalResult::Success(_) => {
                // The observer has already recorded every evaluated node value.
            }
            FnEvalResult::Failure(failure) => {
                return Err(format!(
                    "sample {} failed IR evaluation: {:?}",
                    sample_index + 1,
                    failure
                ));
            }
        }
    }

    let mut words = function
        .nodes
        .iter()
        .enumerate()
        .filter(|(index, _)| observer.word_toggles[*index] > 0)
        .map(|(index, node)| WordActivity {
            ir_node: label(node),
            word_toggles: observer.word_toggles[index],
            bit_toggles: observer.bit_toggles[index],
        })
        .collect::<Vec<_>>();
    words.sort_by(|a, b| {
        b.word_toggles
            .cmp(&a.word_toggles)
            .then_with(|| b.bit_toggles.cmp(&a.bit_toggles))
            .then_with(|| a.ir_node.id.cmp(&b.ir_node.id))
    });

    let lowering = gatify(
        function,
        GatifyOptions {
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )?;
    validate_ordered_batch_inputs(&lowering.gate_fn, &gate_inputs)?;
    let activity = count_toggle_activity(&lowering.gate_fn, &gate_inputs);
    let labels: BTreeMap<usize, IrNodeLabel> = function
        .nodes
        .iter()
        .map(|node| (node.text_id, label(node)))
        .collect();
    let mut output_bits: BTreeMap<usize, Vec<IrOutputBit>> = BTreeMap::new();
    for (node_ref, bits) in &lowering.lowering_map {
        let ir_node = label(function.get_node(*node_ref));
        for (bit_index, operand) in bits.iter_lsb_to_msb().enumerate() {
            output_bits
                .entry(operand.node.id)
                .or_default()
                .push(IrOutputBit {
                    ir_node: ir_node.clone(),
                    bit_index,
                    inverted: operand.negated,
                });
        }
    }
    for bits in output_bits.values_mut() {
        bits.sort_by_key(|bit| (bit.ir_node.id, bit.bit_index));
    }
    let mut gates = activity
        .nodes
        .into_iter()
        .filter(|node| node.node_kind == ToggleNodeKind::And2 && node.toggle_count > 0)
        .map(|node| GateActivity {
            node_id: node.node_id,
            toggle_count: node.toggle_count,
            sources: lowering.gate_fn.gates[node.node_id]
                .get_pir_node_ids()
                .iter()
                .filter_map(|id| labels.get(&(*id as usize)).cloned())
                .collect(),
            ir_output_bits: output_bits.remove(&node.node_id).unwrap_or_default(),
        })
        .collect::<Vec<_>>();
    gates.sort_by(|a, b| {
        b.toggle_count
            .cmp(&a.toggle_count)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });

    Ok(ToggleHotspots {
        function: function.name.clone(),
        sample_count: samples.len(),
        transition_count: samples.len() - 1,
        words,
        gates,
    })
}

/// Parses package IR and `.irvals` stimulus, selecting the package top or sole
/// function.
pub fn analyze_toggle_hotspots_from_files(
    ir_text: &str,
    top: Option<&str>,
    irvals_path: &Path,
) -> Result<ToggleHotspots, String> {
    let package = Parser::new(ir_text)
        .parse_and_validate_package()
        .map_err(|e| format!("IR parse/validate failed: {e}"))?;
    let function = match top {
        Some(name) => package.get_named_or_top_or_sole_fn(name)?,
        None => package.get_top_or_sole_fn()?,
    };
    let samples = load_irvals_samples(irvals_path, function)?;
    analyze_toggle_hotspots(&package, function, &samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IR: &str = r#"package test

top fn main(x: bits[2] id=1, y: bits[2] id=2) -> bits[2] {
  ret result: bits[2] = and(x, y, id=3)
}
"#;

    #[test]
    fn shared_stimulus_counts_words_bits_and_exact_gate_output_bits() {
        let dir = tempfile::tempdir().unwrap();
        let positional = dir.path().join("positional.irvals");
        let named = dir.path().join("named.irvals");
        std::fs::write(
            &positional,
            "(bits[2]:0, bits[2]:3)\n(bits[2]:3, bits[2]:3)\n(bits[2]:0, bits[2]:3)\n",
        )
        .unwrap();
        std::fs::write(
            &named,
            "{y: bits[2]:3, x: bits[2]:0}\n{x: bits[2]:3, y: bits[2]:3}\n{y: bits[2]:3, x: bits[2]:0}\n",
        )
        .unwrap();

        let report = analyze_toggle_hotspots_from_files(IR, None, &positional).unwrap();
        assert_eq!(
            report,
            analyze_toggle_hotspots_from_files(IR, Some("main"), &named).unwrap()
        );
        assert_eq!(report.sample_count, 3);
        assert_eq!(report.transition_count, 2);
        assert_eq!(report.words.len(), 2);
        assert_eq!(report.words[0].ir_node.name, "x");
        assert_eq!(report.words[0].word_toggles, 2);
        assert_eq!(report.words[0].bit_toggles, 4);
        assert_eq!(report.words[1].ir_node.name, "result");
        assert_eq!(report.words[1].word_toggles, 2);
        assert_eq!(report.words[1].bit_toggles, 4);
        assert_eq!(report.gates.len(), 2);
        for gate in &report.gates {
            assert_eq!(gate.toggle_count, 2);
            assert_eq!(gate.sources[0].id, 3);
            assert_eq!(gate.ir_output_bits.len(), 1);
            assert_eq!(gate.ir_output_bits[0].ir_node.name, "result");
            assert!(!gate.ir_output_bits[0].inverted);
        }
        assert_eq!(
            report
                .gates
                .iter()
                .map(|gate| gate.ir_output_bits[0].bit_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn requires_valid_ordered_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stimulus.irvals");
        std::fs::write(&path, "(bits[2]:0, bits[2]:3)\n").unwrap();
        assert!(
            analyze_toggle_hotspots_from_files(IR, None, &path)
                .unwrap_err()
                .contains("at least two samples")
        );
        std::fs::write(&path, "(bits[3]:0, bits[2]:3)\n(bits[3]:1, bits[2]:3)\n").unwrap();
        assert!(
            analyze_toggle_hotspots_from_files(IR, None, &path)
                .unwrap_err()
                .contains("type mismatch")
        );
    }
}
