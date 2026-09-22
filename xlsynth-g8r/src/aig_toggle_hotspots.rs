// SPDX-License-Identifier: Apache-2.0

//! Ordered AND2 switching activity for an existing combinational AIG artifact.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;

use crate::aig::GateFn;
use crate::aig_serdes::g8r::load_gate_fn_from_path;
use crate::aig_serdes::gate2ir::{GateFnInterfaceSchema, repack_gate_fn_interface_with_schema};
use crate::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use crate::aig_sim::count_toggles::{ToggleNodeKind, count_toggle_activity};
use crate::aig_sim::gate_simd::validate_ordered_batch_inputs;
use crate::gate_builder::GateBuilderOptions;
use crate::ir_toggle_hotspots::label;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AigSource {
    pub id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GateActivity {
    pub node_id: usize,
    pub toggle_count: usize,
    /// IR node IDs retained in a native g8r artifact, if tracked at lowering.
    pub sources: Vec<AigSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AigToggleHotspots {
    pub sample_count: usize,
    pub transition_count: usize,
    /// Output-reachable AND2 nodes ranked by toggles, then AIG node ID.
    pub gates: Vec<GateActivity>,
}

/// Loads an existing AIGER or native g8r combinational artifact.
pub fn load_aig_artifact(path: &Path) -> Result<GateFn, String> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("aag" | "aig") => load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
            .map(|result| result.gate_fn),
        Some("g8r" | "g8rbin") => load_gate_fn_from_path(path),
        _ => Err(format!(
            "expected a .aag, .aig, .g8r, or .g8rbin artifact: {}",
            path.display()
        )),
    }
}

/// Ranks switching AND2 gates in an already lowered circuit.
///
/// Source labels are looked up only from the explicitly supplied IR function;
/// AIGER itself does not retain IR IDs. No IR bit positions are inferred from
/// the source IDs because an artifact does not preserve the lowering map.
pub fn analyze_aig_toggle_hotspots(
    gate_fn: &GateFn,
    inputs: &[Vec<IrBits>],
    source_fn: Option<&ir::Fn>,
) -> Result<AigToggleHotspots, String> {
    if inputs.len() < 2 {
        return Err(format!(
            "toggle stimulus must contain at least two samples; got {}",
            inputs.len()
        ));
    }
    validate_ordered_batch_inputs(gate_fn, inputs)?;
    let labels = source_fn
        .map(|function| {
            function
                .nodes
                .iter()
                .map(|node| (node.text_id, label(node)))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let activity = count_toggle_activity(gate_fn, inputs);
    let mut gates = activity
        .nodes
        .into_iter()
        .filter(|node| node.node_kind == ToggleNodeKind::And2 && node.toggle_count > 0)
        .map(|node| GateActivity {
            node_id: node.node_id,
            toggle_count: node.toggle_count,
            sources: gate_fn.gates[node.node_id]
                .get_pir_node_ids()
                .iter()
                .map(|&id| {
                    let ir_node = labels.get(&(id as usize));
                    AigSource {
                        id,
                        name: ir_node.map(|node| node.name.clone()),
                        op: ir_node.map(|node| node.op.clone()),
                    }
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    gates.sort_by(|a, b| {
        b.toggle_count
            .cmp(&a.toggle_count)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
    Ok(AigToggleHotspots {
        sample_count: inputs.len(),
        transition_count: inputs.len() - 1,
        gates,
    })
}

/// Loads one artifact and typed ordered stimulus, with optional IR annotations.
pub fn analyze_aig_toggle_hotspots_from_files(
    aig_path: &Path,
    irvals_path: &Path,
    fn_type: Option<&ir::FunctionType>,
    source_ir: Option<(&str, Option<&str>)>,
) -> Result<AigToggleHotspots, String> {
    let package = source_ir
        .map(|(text, _)| {
            Parser::new(text)
                .parse_and_validate_package()
                .map_err(|e| format!("source IR parse/validate failed: {e}"))
        })
        .transpose()?;
    let source_fn = match (package.as_ref(), source_ir) {
        (Some(package), Some((_, Some(top)))) => Some(package.get_named_or_top_or_sole_fn(top)?),
        (Some(package), Some((_, None))) => Some(package.get_top_or_sole_fn()?),
        _ => None,
    };
    if let (Some(function), Some(fn_type)) = (source_fn, fn_type) {
        if function.get_type() != *fn_type {
            return Err(format!(
                "--fn-type {:?} does not match source IR function type {:?}",
                fn_type,
                function.get_type()
            ));
        }
    }

    let mut gate_fn = load_aig_artifact(aig_path)?;
    if let Some(function) = source_fn {
        let schema = GateFnInterfaceSchema::from_pir_fn(function)?;
        gate_fn = repack_gate_fn_interface_with_schema(gate_fn, &schema)?;
    } else if let Some(fn_type) = fn_type {
        let schema = GateFnInterfaceSchema::from_function_type(fn_type)?;
        gate_fn = repack_gate_fn_interface_with_schema(gate_fn, &schema)?;
    }
    let names = gate_fn
        .inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<_>>();
    let samples = xlsynth_pir::parse_ir_values_file(irvals_path)
        .and_then(|file| file.into_positional_values(&names))
        .map_err(|e| e.to_string())?;
    let source_type = source_fn.map(ir::Fn::get_type);
    let flat_type = gate_fn.get_flat_type();
    let param_types = &source_type
        .as_ref()
        .or(fn_type)
        .unwrap_or(&flat_type)
        .param_types;
    let expected = ir::Type::Tuple(param_types.iter().cloned().map(Box::new).collect());
    let mut inputs = Vec::with_capacity(samples.len());
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
        inputs.push(
            args.iter()
                .zip(param_types)
                .map(|(arg, ty)| {
                    let mut bits = Vec::with_capacity(ty.bit_count());
                    flatten_ir_value_to_lsb0_bits_for_type(arg, ty, &mut bits)?;
                    Ok(IrBits::from_lsb_is_0(&bits))
                })
                .collect::<Result<Vec<_>, String>>()?,
        );
    }
    analyze_aig_toggle_hotspots(&gate_fn, &inputs, source_fn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::SequentialGateFn;
    use crate::aig_serdes::emit_aiger::emit_aiger;
    use crate::aig_serdes::emit_aiger_binary::emit_aiger_binary;
    use crate::aig_serdes::g8r::{emit_g8r, encode_g8r_binary};
    use crate::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};

    const IR: &str = r#"package test

top fn main(x: bits[2] id=1, y: bits[2] id=2) -> bits[2] {
  ret result: bits[2] = and(x, y, id=3)
}
"#;

    #[test]
    fn native_artifact_labels_sources_and_aiger_omits_them() {
        let dir = tempfile::tempdir().unwrap();
        let native_path = dir.path().join("design.g8rbin");
        let native_text_path = dir.path().join("design.g8r");
        let aiger_path = dir.path().join("design.aag");
        let binary_aiger_path = dir.path().join("design.aig");
        let stimulus_path = dir.path().join("inputs.irvals");
        let package = Parser::new(IR).parse_and_validate_package().unwrap();
        let function = package.get_top_fn().unwrap();
        let gate_fn = gatify_prepared_fn(
            function,
            GatifyOptions {
                track_pir_node_ids: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap()
        .gate_fn;
        let sequential = SequentialGateFn::from_gate_fn(gate_fn.clone());
        std::fs::write(&native_path, encode_g8r_binary(&sequential).unwrap()).unwrap();
        std::fs::write(&native_text_path, emit_g8r(&sequential)).unwrap();
        std::fs::write(&aiger_path, emit_aiger(&gate_fn, true).unwrap()).unwrap();
        std::fs::write(
            &binary_aiger_path,
            emit_aiger_binary(&gate_fn, true).unwrap(),
        )
        .unwrap();
        std::fs::write(
            &stimulus_path,
            "{y: bits[2]:3, x: bits[2]:0}\n{x: bits[2]:3, y: bits[2]:3}\n{x: bits[2]:0, y: bits[2]:3}\n",
        )
        .unwrap();

        let native = analyze_aig_toggle_hotspots_from_files(
            &native_path,
            &stimulus_path,
            None,
            Some((IR, None)),
        )
        .unwrap();
        assert_eq!(native.sample_count, 3);
        assert_eq!(native.transition_count, 2);
        assert_eq!(native.gates.len(), 2);
        assert_eq!(
            native,
            analyze_aig_toggle_hotspots_from_files(
                &native_text_path,
                &stimulus_path,
                None,
                Some((IR, None)),
            )
            .unwrap()
        );
        for gate in &native.gates {
            assert_eq!(gate.toggle_count, 2);
            assert_eq!(gate.sources.len(), 1);
            assert_eq!(gate.sources[0].id, 3);
            assert_eq!(gate.sources[0].name.as_deref(), Some("result"));
            assert_eq!(gate.sources[0].op.as_deref(), Some("and"));
        }

        let without_ir =
            analyze_aig_toggle_hotspots_from_files(&native_path, &stimulus_path, None, None)
                .unwrap();
        assert_eq!(without_ir.gates[0].sources[0].id, 3);
        assert_eq!(without_ir.gates[0].sources[0].name, None);

        let aiger = analyze_aig_toggle_hotspots_from_files(
            &aiger_path,
            &stimulus_path,
            None,
            Some((IR, None)),
        )
        .unwrap();
        assert_eq!(aiger.gates.len(), 2);
        assert!(aiger.gates.iter().all(|gate| gate.sources.is_empty()));
        assert!(aiger.gates.iter().all(|gate| gate.toggle_count == 2));
        assert_eq!(
            aiger,
            analyze_aig_toggle_hotspots_from_files(
                &binary_aiger_path,
                &stimulus_path,
                None,
                Some((IR, None)),
            )
            .unwrap()
        );

        let aiger_by_type = analyze_aig_toggle_hotspots_from_files(
            &aiger_path,
            &stimulus_path,
            Some(&function.get_type()),
            Some((IR, None)),
        )
        .unwrap();
        assert_eq!(aiger_by_type, aiger);

        std::fs::write(&stimulus_path, "(bits[2]:0, bits[2]:3)\n").unwrap();
        assert!(
            analyze_aig_toggle_hotspots_from_files(&native_path, &stimulus_path, None, None)
                .unwrap_err()
                .contains("at least two samples")
        );
    }
}
