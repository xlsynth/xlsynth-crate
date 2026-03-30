// SPDX-License-Identifier: Apache-2.0

//! Builds per-PIR-node area attribution summaries for live AIG AND nodes.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_pir::ir;

use crate::aig::gate::{AigNode, GateFn};
use crate::aig::get_summary_stats::get_level_critical_path_ands;
use crate::use_count::get_id_to_use_count;

#[derive(Debug, Clone, PartialEq)]
pub struct AreaTableRow {
    pub pir_node_id: u32,
    pub opcode: String,
    pub ir_text: String,
    pub raw_aig_node_count: usize,
    pub weighted_aig_node_count: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpcodeAreaTableRow {
    pub opcode: String,
    pub raw_aig_node_count: usize,
    pub weighted_aig_node_count: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnattributedAreaTableRow {
    pub raw_aig_node_count: usize,
    pub weighted_aig_node_count: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AreaTableReport {
    pub function_name: String,
    pub total_aig_node_count: usize,
    pub selected_aig_node_count: usize,
    pub critical_path_depth_nodes: Option<usize>,
    pub rows: Vec<AreaTableRow>,
    pub unattributed: Option<UnattributedAreaTableRow>,
    pub missing_pir_node_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpcodeAreaTableReport {
    pub function_name: String,
    pub total_aig_node_count: usize,
    pub selected_aig_node_count: usize,
    pub critical_path_depth_nodes: Option<usize>,
    pub rows: Vec<OpcodeAreaTableRow>,
    pub unattributed: Option<UnattributedAreaTableRow>,
    pub missing_pir_node_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
struct PirNodeInfo {
    opcode: String,
    ir_text: String,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Counts {
    raw: usize,
    weighted: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AigNodeSelection {
    AllLiveAnds,
    LevelCriticalPathAnds,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedAndNodes {
    total_aig_node_count: usize,
    selected_and_nodes: BTreeSet<crate::aig::AigRef>,
    critical_path_depth_nodes: Option<usize>,
}

fn as_u32_text_id(id: usize, context: &str) -> Result<u32, String> {
    u32::try_from(id).map_err(|_| format!("{context} text_id {id} does not fit in u32"))
}

/// Formats one PIR node without `id=` or `pos=` noise, but with typed operands.
fn render_ir_node_text(f: &ir::Fn, node: &ir::Node) -> Option<String> {
    node.to_string_with_options(
        f,
        &ir::NodeRenderOptions {
            typed_operands: true,
            inline_literals: true,
            include_id: false,
            include_pos: false,
        },
    )
}

fn collect_pir_info_by_id(f: &ir::Fn) -> Result<BTreeMap<u32, PirNodeInfo>, String> {
    let mut result = BTreeMap::new();

    for param in &f.params {
        let pir_node_id = as_u32_text_id(
            param.id.get_wrapped_id(),
            &format!("parameter '{}'", param.name),
        )?;
        result.insert(
            pir_node_id,
            PirNodeInfo {
                opcode: "param".to_string(),
                ir_text: format!("{}: {}", param.name, param.ty),
            },
        );
    }

    for (index, node) in f.nodes.iter().enumerate() {
        if matches!(
            node.payload,
            ir::NodePayload::Nil | ir::NodePayload::GetParam(_)
        ) {
            continue;
        }
        let pir_node_id = as_u32_text_id(node.text_id, "node")?;
        let Some(node_text) = render_ir_node_text(f, node) else {
            continue;
        };
        let ir_text = if f.ret_node_ref == Some(ir::NodeRef { index }) {
            format!("ret {}", node_text)
        } else {
            node_text
        };
        let pir_node_info = PirNodeInfo {
            opcode: node.payload.get_operator().to_string(),
            ir_text,
        };
        if let Some(existing) = result.insert(pir_node_id, pir_node_info.clone()) {
            return Err(format!(
                "duplicate PIR text_id {} in function '{}': {:?} vs {:?}",
                pir_node_id, f.name, existing, pir_node_info
            ));
        }
    }

    Ok(result)
}

/// Selects which live `And2` nodes should contribute to the attribution table.
fn select_and_nodes(gate_fn: &GateFn, selection: AigNodeSelection) -> SelectedAndNodes {
    let live_nodes: Vec<crate::aig::AigRef> =
        get_id_to_use_count(gate_fn).keys().copied().collect();
    let live_and_nodes: BTreeSet<crate::aig::AigRef> = live_nodes
        .iter()
        .copied()
        .filter(|node_ref| matches!(gate_fn.gates[node_ref.id], AigNode::And2 { .. }))
        .collect();
    let selected_and_nodes = match selection {
        AigNodeSelection::AllLiveAnds => SelectedAndNodes {
            total_aig_node_count: live_and_nodes.len(),
            selected_and_nodes: live_and_nodes.clone(),
            critical_path_depth_nodes: None,
        },
        AigNodeSelection::LevelCriticalPathAnds => {
            let critical_path = get_level_critical_path_ands(gate_fn, &live_nodes);
            SelectedAndNodes {
                total_aig_node_count: live_and_nodes.len(),
                selected_and_nodes: critical_path.and_nodes,
                critical_path_depth_nodes: Some(critical_path.depth_aig_nodes),
            }
        }
    };
    selected_and_nodes
}

/// Builds a per-PIR-node area table for the selected PIR function.
///
/// The counts are over live `AigNode::And2` nodes so the totals align with the
/// AIG area notion already used by `get_aig_stats().and_nodes`.
pub fn build_area_table_report(gate_fn: &GateFn, f: &ir::Fn) -> Result<AreaTableReport, String> {
    build_area_table_report_with_selection(gate_fn, f, AigNodeSelection::AllLiveAnds)
}

/// Builds a per-PIR-node table for the `And2` nodes on at least one max-level
/// input-to-output path.
pub fn build_critical_path_area_table_report(
    gate_fn: &GateFn,
    f: &ir::Fn,
) -> Result<AreaTableReport, String> {
    build_area_table_report_with_selection(gate_fn, f, AigNodeSelection::LevelCriticalPathAnds)
}

fn build_area_table_report_with_selection(
    gate_fn: &GateFn,
    f: &ir::Fn,
    selection: AigNodeSelection,
) -> Result<AreaTableReport, String> {
    let pir_info_by_id = collect_pir_info_by_id(f)?;
    let selected_and_nodes = select_and_nodes(gate_fn, selection);

    let mut counts_by_id: BTreeMap<u32, Counts> = BTreeMap::new();
    let mut missing_pir_node_ids = BTreeSet::new();
    let mut unattributed = Counts::default();

    for node_ref in selected_and_nodes.selected_and_nodes.iter().copied() {
        let node = &gate_fn.gates[node_ref.id];

        let pir_node_ids = node.get_pir_node_ids();
        if pir_node_ids.is_empty() {
            unattributed.raw += 1;
            unattributed.weighted += 1.0;
            continue;
        }

        let per_id_weight = 1.0 / pir_node_ids.len() as f64;
        let mut had_unattributed_share = false;
        for pir_node_id in pir_node_ids {
            if pir_info_by_id.contains_key(pir_node_id) {
                let entry = counts_by_id.entry(*pir_node_id).or_default();
                entry.raw += 1;
                entry.weighted += per_id_weight;
            } else {
                had_unattributed_share = true;
                missing_pir_node_ids.insert(*pir_node_id);
                unattributed.weighted += per_id_weight;
            }
        }
        if had_unattributed_share {
            unattributed.raw += 1;
        }
    }

    let rows = counts_by_id
        .into_iter()
        .map(|(pir_node_id, counts)| AreaTableRow {
            pir_node_id,
            opcode: pir_info_by_id
                .get(&pir_node_id)
                .expect("counted PIR node id should have an IR text entry")
                .opcode
                .clone(),
            ir_text: pir_info_by_id
                .get(&pir_node_id)
                .expect("counted PIR node id should have an IR text entry")
                .ir_text
                .clone(),
            raw_aig_node_count: counts.raw,
            weighted_aig_node_count: counts.weighted,
        })
        .collect();

    let unattributed = (unattributed.raw != 0 || unattributed.weighted != 0.0).then_some(
        UnattributedAreaTableRow {
            raw_aig_node_count: unattributed.raw,
            weighted_aig_node_count: unattributed.weighted,
        },
    );

    Ok(AreaTableReport {
        function_name: f.name.clone(),
        total_aig_node_count: selected_and_nodes.total_aig_node_count,
        selected_aig_node_count: selected_and_nodes.selected_and_nodes.len(),
        critical_path_depth_nodes: selected_and_nodes.critical_path_depth_nodes,
        rows,
        unattributed,
        missing_pir_node_ids: missing_pir_node_ids.into_iter().collect(),
    })
}

/// Builds a per-opcode area table for the selected PIR function.
///
/// Each live `And2` contributes at most one raw count per opcode it references,
/// while weighted attribution preserves the per-provenance-id `1/N` split.
pub fn build_opcode_area_table_report(
    gate_fn: &GateFn,
    f: &ir::Fn,
) -> Result<OpcodeAreaTableReport, String> {
    build_opcode_area_table_report_with_selection(gate_fn, f, AigNodeSelection::AllLiveAnds)
}

/// Builds a per-opcode table for the `And2` nodes on at least one max-level
/// input-to-output path.
pub fn build_critical_path_opcode_area_table_report(
    gate_fn: &GateFn,
    f: &ir::Fn,
) -> Result<OpcodeAreaTableReport, String> {
    build_opcode_area_table_report_with_selection(
        gate_fn,
        f,
        AigNodeSelection::LevelCriticalPathAnds,
    )
}

fn build_opcode_area_table_report_with_selection(
    gate_fn: &GateFn,
    f: &ir::Fn,
    selection: AigNodeSelection,
) -> Result<OpcodeAreaTableReport, String> {
    let pir_info_by_id = collect_pir_info_by_id(f)?;
    let selected_and_nodes = select_and_nodes(gate_fn, selection);

    let mut counts_by_opcode: BTreeMap<String, Counts> = BTreeMap::new();
    let mut missing_pir_node_ids = BTreeSet::new();
    let mut unattributed = Counts::default();

    for node_ref in selected_and_nodes.selected_and_nodes.iter().copied() {
        let node = &gate_fn.gates[node_ref.id];

        let pir_node_ids = node.get_pir_node_ids();
        if pir_node_ids.is_empty() {
            unattributed.raw += 1;
            unattributed.weighted += 1.0;
            continue;
        }

        let per_id_weight = 1.0 / pir_node_ids.len() as f64;
        let mut weighted_share_by_opcode: BTreeMap<String, f64> = BTreeMap::new();
        let mut had_unattributed_share = false;
        for pir_node_id in pir_node_ids {
            if let Some(pir_node_info) = pir_info_by_id.get(pir_node_id) {
                *weighted_share_by_opcode
                    .entry(pir_node_info.opcode.clone())
                    .or_default() += per_id_weight;
            } else {
                had_unattributed_share = true;
                missing_pir_node_ids.insert(*pir_node_id);
                unattributed.weighted += per_id_weight;
            }
        }
        for (opcode, weighted_share) in weighted_share_by_opcode {
            let entry = counts_by_opcode.entry(opcode).or_default();
            entry.raw += 1;
            entry.weighted += weighted_share;
        }
        if had_unattributed_share {
            unattributed.raw += 1;
        }
    }

    let rows = counts_by_opcode
        .into_iter()
        .map(|(opcode, counts)| OpcodeAreaTableRow {
            opcode,
            raw_aig_node_count: counts.raw,
            weighted_aig_node_count: counts.weighted,
        })
        .collect();

    let unattributed = (unattributed.raw != 0 || unattributed.weighted != 0.0).then_some(
        UnattributedAreaTableRow {
            raw_aig_node_count: unattributed.raw,
            weighted_aig_node_count: unattributed.weighted,
        },
    );

    Ok(OpcodeAreaTableReport {
        function_name: f.name.clone(),
        total_aig_node_count: selected_and_nodes.total_aig_node_count,
        selected_aig_node_count: selected_and_nodes.selected_and_nodes.len(),
        critical_path_depth_nodes: selected_and_nodes.critical_path_depth_nodes,
        rows,
        unattributed,
        missing_pir_node_ids: missing_pir_node_ids.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_area_table_report, build_critical_path_area_table_report,
        build_critical_path_opcode_area_table_report, build_opcode_area_table_report,
    };
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use xlsynth_pir::ir_parser;

    #[test]
    fn test_build_area_table_report_counts_live_and_nodes() {
        let ir_text = r#"package sample
top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  add.3: bits[1] = add(a, b, id=3, pos=[(0,1,2)])
  ret not.4: bits[1] = not(add.3, id=4, pos=[(0,3,4)])
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let package = parser.parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();

        let mut builder = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);

        builder.set_current_pir_node_id(Some(3));
        let add_like = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));

        builder.set_current_pir_node_id(Some(4));
        let shared = builder.add_and_binary(*a.get_lsb(0), *a.get_lsb(0));
        builder.add_pir_node_id(shared.node, 3);

        builder.set_current_pir_node_id(None);
        let unattributed = builder.add_and_binary(*b.get_lsb(0), *b.get_lsb(0));

        builder.set_current_pir_node_id(Some(99));
        let missing = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        builder.set_current_pir_node_id(None);

        builder.add_output("o0".to_string(), add_like.into());
        builder.add_output("o1".to_string(), shared.into());
        builder.add_output("o2".to_string(), unattributed.into());
        builder.add_output("o3".to_string(), missing.into());
        let gate_fn = builder.build();

        let report = build_area_table_report(&gate_fn, f).unwrap();

        assert_eq!(report.function_name, "f");
        assert_eq!(report.total_aig_node_count, 4);
        assert_eq!(report.selected_aig_node_count, 4);
        assert_eq!(report.critical_path_depth_nodes, None);
        assert_eq!(report.missing_pir_node_ids, vec![99]);
        assert_eq!(report.rows.len(), 2);

        let add_row = report.rows.iter().find(|row| row.pir_node_id == 3).unwrap();
        assert_eq!(add_row.opcode, "add");
        assert_eq!(
            add_row.ir_text,
            "add.3: bits[1] = add(a: bits[1], b: bits[1])"
        );
        assert_eq!(add_row.raw_aig_node_count, 2);
        assert_eq!(add_row.weighted_aig_node_count, 1.5);

        let not_row = report.rows.iter().find(|row| row.pir_node_id == 4).unwrap();
        assert_eq!(not_row.ir_text, "ret not.4: bits[1] = not(add.3: bits[1])");
        assert_eq!(not_row.raw_aig_node_count, 1);
        assert_eq!(not_row.weighted_aig_node_count, 0.5);

        assert_eq!(
            report.unattributed,
            Some(super::UnattributedAreaTableRow {
                raw_aig_node_count: 2,
                weighted_aig_node_count: 2.0,
            })
        );
    }

    #[test]
    fn test_build_area_table_report_formats_param_rows() {
        let ir_text = r#"package sample
top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret add.3: bits[1] = add(a, b, id=3)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let package = parser.parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();

        let mut builder = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);

        builder.set_current_pir_node_id(Some(1));
        let attributed_to_param = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        builder.set_current_pir_node_id(None);
        builder.add_output("o".to_string(), attributed_to_param.into());
        let gate_fn = builder.build();

        let report = build_area_table_report(&gate_fn, f).unwrap();
        assert_eq!(report.function_name, "f");
        assert_eq!(report.total_aig_node_count, 1);
        assert_eq!(report.selected_aig_node_count, 1);
        assert_eq!(report.critical_path_depth_nodes, None);
        let param_row = report.rows.iter().find(|row| row.pir_node_id == 1).unwrap();
        assert_eq!(param_row.ir_text, "a: bits[1]");
        assert_eq!(param_row.raw_aig_node_count, 1);
        assert_eq!(param_row.weighted_aig_node_count, 1.0);
        assert_eq!(report.unattributed, None);
    }

    #[test]
    fn test_build_opcode_area_table_report_groups_duplicate_opcode_ids_per_node() {
        let ir_text = r#"package sample
top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  add.3: bits[1] = add(a, b, id=3)
  ret add.4: bits[1] = add(add.3, b, id=4)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let package = parser.parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();

        let mut builder = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);

        builder.set_current_pir_node_id(Some(3));
        let shared_add = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        builder.add_pir_node_id(shared_add.node, 4);

        builder.set_current_pir_node_id(Some(1));
        let param_related = builder.add_and_binary(*a.get_lsb(0), *a.get_lsb(0));
        builder.set_current_pir_node_id(None);

        builder.add_output("o0".to_string(), shared_add.into());
        builder.add_output("o1".to_string(), param_related.into());
        let gate_fn = builder.build();

        let report = build_opcode_area_table_report(&gate_fn, f).unwrap();

        assert_eq!(report.function_name, "f");
        assert_eq!(report.total_aig_node_count, 2);
        assert_eq!(report.selected_aig_node_count, 2);
        assert_eq!(report.critical_path_depth_nodes, None);
        assert_eq!(report.missing_pir_node_ids, Vec::<u32>::new());
        assert_eq!(report.rows.len(), 2);

        let add_row = report.rows.iter().find(|row| row.opcode == "add").unwrap();
        assert_eq!(add_row.raw_aig_node_count, 1);
        assert_eq!(add_row.weighted_aig_node_count, 1.0);

        let param_row = report
            .rows
            .iter()
            .find(|row| row.opcode == "param")
            .unwrap();
        assert_eq!(param_row.raw_aig_node_count, 1);
        assert_eq!(param_row.weighted_aig_node_count, 1.0);

        assert_eq!(report.unattributed, None);
    }

    #[test]
    fn test_build_area_table_report_renders_literal_operands_in_hex() {
        let ir_text = r#"package sample
top fn f(x: bits[8] id=1) -> bits[8] {
  literal.2: bits[8] = literal(value=0xab, id=2)
  ret add.3: bits[8] = add(x, literal.2, id=3)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let package = parser.parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();

        let mut builder = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let x = builder.add_input("x".to_string(), 1);
        builder.set_current_pir_node_id(Some(3));
        let attributed = builder.add_and_binary(*x.get_lsb(0), *x.get_lsb(0));
        builder.set_current_pir_node_id(None);
        builder.add_output("o".to_string(), attributed.into());
        let gate_fn = builder.build();

        let report = build_area_table_report(&gate_fn, f).unwrap();
        let add_row = report.rows.iter().find(|row| row.pir_node_id == 3).unwrap();
        assert_eq!(
            add_row.ir_text,
            "ret add.3: bits[8] = add(x: bits[8], literal.2: bits[8]:0xab)"
        );
    }

    #[test]
    fn test_build_critical_path_area_table_report_filters_to_max_level_nodes() {
        let ir_text = r#"package sample
top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  add.3: bits[1] = add(a, b, id=3)
  not.4: bits[1] = not(add.3, id=4)
  and.5: bits[1] = and(not.4, b, id=5)
  ret xor.6: bits[1] = xor(and.5, a, id=6)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let package = parser.parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();

        let mut builder = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);

        builder.set_current_pir_node_id(Some(3));
        let crit0 = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));

        builder.set_current_pir_node_id(Some(4));
        let crit1 = builder.add_and_binary(crit0, *a.get_lsb(0));
        builder.add_pir_node_id(crit1.node, 3);

        builder.set_current_pir_node_id(Some(5));
        let crit2 = builder.add_and_binary(crit1, *b.get_lsb(0));
        builder.add_pir_node_id(crit2.node, 99);

        builder.set_current_pir_node_id(Some(6));
        let non_critical0 = builder.add_and_binary(*a.get_lsb(0), *a.get_lsb(0));
        let non_critical1 = builder.add_and_binary(non_critical0, *b.get_lsb(0));
        builder.set_current_pir_node_id(None);

        builder.add_output("o0".to_string(), crit2.into());
        builder.add_output("o1".to_string(), non_critical1.into());
        let gate_fn = builder.build();

        let report = build_critical_path_area_table_report(&gate_fn, f).unwrap();
        assert_eq!(report.total_aig_node_count, 5);
        assert_eq!(report.selected_aig_node_count, 3);
        assert_eq!(report.critical_path_depth_nodes, Some(3));
        assert_eq!(report.missing_pir_node_ids, vec![99]);
        assert_eq!(report.rows.len(), 3);

        let add_row = report.rows.iter().find(|row| row.pir_node_id == 3).unwrap();
        assert_eq!(add_row.raw_aig_node_count, 2);
        assert_eq!(add_row.weighted_aig_node_count, 1.5);

        let not_row = report.rows.iter().find(|row| row.pir_node_id == 4).unwrap();
        assert_eq!(not_row.raw_aig_node_count, 1);
        assert_eq!(not_row.weighted_aig_node_count, 0.5);

        let and_row = report.rows.iter().find(|row| row.pir_node_id == 5).unwrap();
        assert_eq!(and_row.raw_aig_node_count, 1);
        assert_eq!(and_row.weighted_aig_node_count, 0.5);

        assert_eq!(
            report.unattributed,
            Some(super::UnattributedAreaTableRow {
                raw_aig_node_count: 1,
                weighted_aig_node_count: 0.5,
            })
        );
    }

    #[test]
    fn test_build_critical_path_opcode_area_table_report_filters_to_max_level_nodes() {
        let ir_text = r#"package sample
top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  add.3: bits[1] = add(a, b, id=3)
  not.4: bits[1] = not(add.3, id=4)
  and.5: bits[1] = and(not.4, b, id=5)
  ret xor.6: bits[1] = xor(and.5, a, id=6)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let package = parser.parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();

        let mut builder = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);

        builder.set_current_pir_node_id(Some(3));
        let crit0 = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));

        builder.set_current_pir_node_id(Some(4));
        let crit1 = builder.add_and_binary(crit0, *a.get_lsb(0));
        builder.add_pir_node_id(crit1.node, 3);

        builder.set_current_pir_node_id(Some(5));
        let crit2 = builder.add_and_binary(crit1, *b.get_lsb(0));
        builder.add_pir_node_id(crit2.node, 99);

        builder.set_current_pir_node_id(Some(6));
        let non_critical0 = builder.add_and_binary(*a.get_lsb(0), *a.get_lsb(0));
        let non_critical1 = builder.add_and_binary(non_critical0, *b.get_lsb(0));
        builder.set_current_pir_node_id(None);

        builder.add_output("o0".to_string(), crit2.into());
        builder.add_output("o1".to_string(), non_critical1.into());
        let gate_fn = builder.build();

        let report = build_critical_path_opcode_area_table_report(&gate_fn, f).unwrap();
        assert_eq!(report.total_aig_node_count, 5);
        assert_eq!(report.selected_aig_node_count, 3);
        assert_eq!(report.critical_path_depth_nodes, Some(3));
        assert_eq!(report.missing_pir_node_ids, vec![99]);
        assert_eq!(report.rows.len(), 3);

        let add_row = report.rows.iter().find(|row| row.opcode == "add").unwrap();
        assert_eq!(add_row.raw_aig_node_count, 2);
        assert_eq!(add_row.weighted_aig_node_count, 1.5);

        let and_row = report.rows.iter().find(|row| row.opcode == "and").unwrap();
        assert_eq!(and_row.raw_aig_node_count, 1);
        assert_eq!(and_row.weighted_aig_node_count, 0.5);

        let not_row = report.rows.iter().find(|row| row.opcode == "not").unwrap();
        assert_eq!(not_row.raw_aig_node_count, 1);
        assert_eq!(not_row.weighted_aig_node_count, 0.5);

        assert_eq!(
            report.unattributed,
            Some(super::UnattributedAreaTableRow {
                raw_aig_node_count: 1,
                weighted_aig_node_count: 0.5,
            })
        );
    }
}
