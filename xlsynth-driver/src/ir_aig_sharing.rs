// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::Path;

use clap::ArgMatches;
use xlsynth_g8r::aig::gate::{AigBitVector, Input};
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir_aig_sharing::{
    get_equivalences, prove_equivalence_candidates_varisat, CandidateProofResult,
    IrAigCandidateRhs, IrAigSharingOptions,
};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

use crate::common::parse_bool_flag_or;
use crate::toolchain_config::ToolchainConfig;

fn format_mapping_rhs(rhs: IrAigCandidateRhs) -> String {
    match rhs {
        IrAigCandidateRhs::AigOperand(op) => {
            if op.negated {
                format!("!%{}", op.node.id)
            } else {
                format!("%{}", op.node.id)
            }
        }
        IrAigCandidateRhs::Const(false) => "0".to_string(),
        IrAigCandidateRhs::Const(true) => "1".to_string(),
    }
}

fn choose_preferred_operand(
    a: xlsynth_g8r::aig::gate::AigOperand,
    b: xlsynth_g8r::aig::gate::AigOperand,
) -> xlsynth_g8r::aig::gate::AigOperand {
    // Deterministic choice: smaller node id wins; if tied, non-negated wins.
    match a.node.id.cmp(&b.node.id) {
        std::cmp::Ordering::Less => a,
        std::cmp::Ordering::Greater => b,
        std::cmp::Ordering::Equal => {
            if a.negated == b.negated {
                a
            } else if !a.negated {
                a
            } else {
                b
            }
        }
    }
}

fn choose_preferred_rhs(a: IrAigCandidateRhs, b: IrAigCandidateRhs) -> IrAigCandidateRhs {
    match (a, b) {
        (IrAigCandidateRhs::Const(_), _) => a,
        (_, IrAigCandidateRhs::Const(_)) => b,
        (IrAigCandidateRhs::AigOperand(aop), IrAigCandidateRhs::AigOperand(bop)) => {
            IrAigCandidateRhs::AigOperand(choose_preferred_operand(aop, bop))
        }
    }
}

fn load_aig_gate_fn(path: &Path) -> Result<GateFn, String> {
    load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
        .map(|res| res.gate_fn)
        .map_err(|e| format!("failed to load {}: {}", path.display(), e))
}

fn repack_flat_aig_inputs_to_pir_params(
    pir_fn: &ir::Fn,
    mut gate_fn: GateFn,
) -> Result<GateFn, String> {
    let want_param_count = pir_fn.params.len();
    let want_total_bits: usize = pir_fn.params.iter().map(|p| p.ty.bit_count()).sum();

    let gate_total_bits: usize = gate_fn.inputs.iter().map(|i| i.get_bit_count()).sum();

    // Only repack the common "AIGER loader created one 1-bit input per bit" case.
    let all_one_bit_inputs = gate_fn.inputs.iter().all(|i| i.get_bit_count() == 1);
    if !all_one_bit_inputs
        || gate_fn.inputs.len() != want_total_bits
        || gate_total_bits != want_total_bits
    {
        return Ok(gate_fn);
    }

    // Flatten the underlying input operands in AIGER order.
    let mut flat_ops = Vec::with_capacity(want_total_bits);
    for inp in &gate_fn.inputs {
        flat_ops.push(*inp.bit_vector.get_lsb(0));
    }
    debug_assert_eq!(flat_ops.len(), want_total_bits);

    let mut new_inputs: Vec<Input> = Vec::with_capacity(want_param_count);
    let mut offset = 0usize;
    for p in &pir_fn.params {
        let w = p.ty.bit_count();
        let slice = &flat_ops[offset..offset + w];
        new_inputs.push(Input {
            name: p.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(slice),
        });
        offset += w;
    }
    gate_fn.inputs = new_inputs;
    Ok(gate_fn)
}

fn load_pir_top_fn(ir_path: &Path, top: Option<&str>) -> Result<(ir::Package, ir::Fn), String> {
    let text = std::fs::read_to_string(ir_path)
        .map_err(|e| format!("failed to read {}: {}", ir_path.display(), e))?;
    let mut parser = ir_parser::Parser::new(&text);
    let pkg = parser.parse_and_validate_package().map_err(|e| {
        format!(
            "failed to parse/validate PIR package {}: {}",
            ir_path.display(),
            e
        )
    })?;
    let f = if let Some(name) = top {
        pkg.get_fn(name)
            .ok_or_else(|| format!("top function '{}' not found in {}", name, ir_path.display()))?
            .clone()
    } else {
        pkg.get_top_fn()
            .ok_or_else(|| format!("no top function found in {}", ir_path.display()))?
            .clone()
    };
    Ok((pkg, f))
}

pub fn handle_ir_aig_sharing(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let ir_path = Path::new(matches.get_one::<String>("pir_ir_file").unwrap());
    let aig_path = Path::new(matches.get_one::<String>("aig_file").unwrap());
    let top = matches.get_one::<String>("ir_top").map(|s| s.as_str());

    let sample_count = matches
        .get_one::<String>("sample_count")
        .map(|s| s.parse::<usize>().unwrap_or(256))
        .unwrap_or(256);
    let sample_seed = matches
        .get_one::<String>("sample_seed")
        .map(|s| s.parse::<u64>().unwrap_or(0))
        .unwrap_or(0);
    let exclude_structural = parse_bool_flag_or(
        matches,
        "exclude_structural_pir_nodes",
        /* default_value= */ true,
    );

    let max_proofs = matches
        .get_one::<String>("max_proofs")
        .map(|s| s.parse::<usize>().unwrap_or(0))
        .unwrap_or(0);
    let print_limit = matches
        .get_one::<String>("print")
        .map(|s| s.parse::<usize>().unwrap_or(20))
        .unwrap_or(20);
    let print_mappings_limit = matches
        .get_one::<String>("print_mappings")
        .map(|s| s.parse::<usize>().unwrap_or(20))
        .unwrap_or(20);

    let (pir_pkg, pir_fn) = match load_pir_top_fn(ir_path, top) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ir-aig-sharing error: {}", e);
            std::process::exit(2);
        }
    };
    let gate_fn = match load_aig_gate_fn(aig_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("ir-aig-sharing error: {}", e);
            std::process::exit(2);
        }
    };
    let gate_fn = match repack_flat_aig_inputs_to_pir_params(&pir_fn, gate_fn) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("ir-aig-sharing error: {}", e);
            std::process::exit(2);
        }
    };

    let options = IrAigSharingOptions {
        sample_count,
        sample_seed,
        exclude_structural_pir_nodes: exclude_structural,
    };
    let mut candidates = match get_equivalences(&pir_pkg, &pir_fn, &gate_fn, &options) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ir-aig-sharing error: {}", e);
            std::process::exit(2);
        }
    };

    if max_proofs != 0 && candidates.len() > max_proofs {
        candidates.truncate(max_proofs);
    }

    let gatify_opts = GatifyOptions {
        fold: true,
        hash: true,
        check_equivalence: false,
        adder_mapping: AdderMapping::default(),
        mul_adder_mapping: None,
        range_info: None,
        enable_rewrite_carry_out: false,
    };

    let proofs =
        match prove_equivalence_candidates_varisat(&pir_fn, &gate_fn, &candidates, &gatify_opts) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("ir-aig-sharing error: {}", e);
                std::process::exit(2);
            }
        };

    let mut proved = 0usize;
    let mut disproved = 0usize;
    let mut skipped = 0usize;
    for p in &proofs {
        match &p.result {
            CandidateProofResult::Proved => proved += 1,
            CandidateProofResult::Disproved { .. } => disproved += 1,
            CandidateProofResult::Skipped { .. } => skipped += 1,
        }
    }

    println!(
        "ir-aig-sharing: samples={} seed={} candidates={} proved={} disproved={} skipped={}",
        sample_count,
        sample_seed,
        proofs.len(),
        proved,
        disproved,
        skipped
    );

    for (i, p) in proofs.iter().take(print_limit).enumerate() {
        match &p.result {
            CandidateProofResult::Proved => {
                println!(
                    "proof[{}]: PROVED pir_text_id={} bit={} <-> {}",
                    i,
                    p.candidate.pir_node_text_id,
                    p.candidate.bit_index,
                    format_mapping_rhs(p.candidate.rhs)
                );
            }
            CandidateProofResult::Disproved {
                counterexample_inputs,
            } => {
                println!(
                    "proof[{}]: DISPROVED pir_text_id={} bit={} <-> {} cex_inputs={:?}",
                    i,
                    p.candidate.pir_node_text_id,
                    p.candidate.bit_index,
                    format_mapping_rhs(p.candidate.rhs),
                    counterexample_inputs
                );
            }
            CandidateProofResult::Skipped { reason } => {
                println!(
                    "proof[{}]: SKIPPED pir_text_id={} bit={} reason={}",
                    i, p.candidate.pir_node_text_id, p.candidate.bit_index, reason
                );
            }
        }
    }

    // --- Mapping-style output: PIR nodes in topo order, bits MSB..LSB.
    //
    // Build a deterministic map from (pir node ref, bit_index_lsb0) -> aig operand
    // using only proved candidates.
    let mut proved_map: HashMap<(usize, usize), IrAigCandidateRhs> = HashMap::new();
    for p in &proofs {
        if let CandidateProofResult::Proved = p.result {
            let key = (p.candidate.pir_node_ref.index, p.candidate.bit_index);
            proved_map
                .entry(key)
                .and_modify(|existing| {
                    *existing = choose_preferred_rhs(*existing, p.candidate.rhs);
                })
                .or_insert(p.candidate.rhs);
        }
    }

    let mut printed_nodes = 0usize;
    for node_ref in pir_fn.node_refs() {
        if node_ref.index == 0 {
            continue; // reserved Nil
        }
        if print_mappings_limit != 0 && printed_nodes >= print_mappings_limit {
            break;
        }

        let ty = pir_fn.get_node_ty(node_ref);
        let ir::Type::Bits(w) = ty else {
            continue;
        };

        // Only emit nodes that have at least one proved bit mapping.
        let mut any = false;
        for bit in 0..*w {
            if proved_map.contains_key(&(node_ref.index, bit)) {
                any = true;
                break;
            }
        }
        if !any {
            continue;
        }

        let node_name = xlsynth_pir::ir::node_textual_id(&pir_fn, node_ref);

        let mut items: Vec<String> = Vec::with_capacity(*w);
        // Print MSB..LSB, while proved_map uses LSB=0 indexing.
        for bit_index in (0..*w).rev() {
            if let Some(rhs) = proved_map.get(&(node_ref.index, bit_index)) {
                items.push(format_mapping_rhs(*rhs));
            } else {
                items.push("?".to_string());
            }
        }
        println!("{}: bits[{}] = [{}]", node_name, w, items.join(", "));
        printed_nodes += 1;
    }

    if disproved != 0 {
        std::process::exit(1);
    }
}
