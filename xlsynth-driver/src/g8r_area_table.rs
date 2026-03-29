// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use comfy_table::presets::ASCII_MARKDOWN;
use comfy_table::{CellAlignment, ContentArrangement, Table};
use xlsynth_g8r::aig::area_table::{
    build_area_table_report, build_critical_path_area_table_report,
    build_critical_path_opcode_area_table_report, build_opcode_area_table_report, AreaTableReport,
    OpcodeAreaTableReport, UnattributedAreaTableRow,
};
use xlsynth_g8r::aig::GateFn;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

use crate::toolchain_config::ToolchainConfig;

const AREA_SUBCOMMAND: &str = "g8r-area-table";
const CRITICAL_PATH_SUBCOMMAND: &str = "g8r-critical-path-table";

struct DisplayRow {
    group_key: Option<String>,
    group_label: String,
    ir_text: Option<String>,
    raw_aig_node_count: usize,
    raw_percentage: f64,
    weighted_aig_node_count: f64,
    weighted_percentage: f64,
}

fn load_gate_fn(path: &Path, subcommand: &str) -> Result<GateFn, String> {
    if !path.extension().map(|e| e == "g8rbin").unwrap_or(false) {
        return Err(format!(
            "{} requires a .g8rbin input because text .g8r files do not preserve PIR provenance",
            subcommand
        ));
    }
    let bytes = fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    bincode::deserialize(&bytes).map_err(|e| {
        format!(
            "failed to deserialize GateFn from {}: {}",
            path.display(),
            e
        )
    })
}

fn load_selected_ir_fn(path: &Path, top: Option<&str>) -> Result<ir::Fn, String> {
    let pkg = ir_parser::parse_and_validate_path_to_package(path).map_err(|e| {
        format!(
            "failed to parse/validate PIR package {}: {}",
            path.display(),
            e
        )
    })?;

    if let Some(name) = top {
        if let Some(f) = pkg.get_fn(name) {
            return Ok(f.clone());
        }
        if let Some(ir::PackageMember::Block { func, .. }) = pkg.get_block(name) {
            return Ok(func.clone());
        }
        return Err(format!(
            "top IR member '{}' not found in {}",
            name,
            path.display()
        ));
    }

    if let Some(f) = pkg.get_top_fn() {
        return Ok(f.clone());
    }
    if let Some(ir::PackageMember::Block { func, .. }) = pkg.get_top_block() {
        return Ok(func.clone());
    }
    Err(format!(
        "no top function or block found in {}",
        path.display()
    ))
}

fn render_rows_table(
    function_name: &str,
    total_aig_node_count: usize,
    extra_summary_lines: &[(&str, usize)],
    label_header: &str,
    include_ir_text: bool,
    mut rows: Vec<DisplayRow>,
) -> String {
    fn fixed_point_width(value: f64) -> usize {
        format!("{value:.1}").len()
    }

    let mut table = Table::new();
    table.load_preset(ASCII_MARKDOWN);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    let mut headers = vec![label_header.to_string()];
    if include_ir_text {
        headers.push("ir_op".to_string());
    }
    headers.push("aig_nodes".to_string());
    headers.push("weighted_aig_nodes".to_string());
    table.set_header(headers);
    table
        .column_mut(if include_ir_text { 2 } else { 1 })
        .expect("aig_nodes column should exist")
        .set_cell_alignment(CellAlignment::Right);
    table
        .column_mut(if include_ir_text { 3 } else { 2 })
        .expect("weighted_aig_nodes column should exist")
        .set_cell_alignment(CellAlignment::Right);

    rows.sort_by(|lhs, rhs| {
        rhs.weighted_aig_node_count
            .total_cmp(&lhs.weighted_aig_node_count)
            .then_with(|| rhs.raw_aig_node_count.cmp(&lhs.raw_aig_node_count))
            .then_with(|| match (&lhs.group_key, &rhs.group_key) {
                (Some(lhs_key), Some(rhs_key)) => lhs_key.cmp(rhs_key),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            })
            .then_with(|| lhs.ir_text.cmp(&rhs.ir_text))
    });

    let weighted_width = rows
        .iter()
        .map(|row| fixed_point_width(row.weighted_aig_node_count))
        .max()
        .unwrap_or(3);
    let raw_width = rows
        .iter()
        .map(|row| row.raw_aig_node_count.to_string().len())
        .max()
        .unwrap_or(1);
    let raw_percentage_width = rows
        .iter()
        .map(|row| fixed_point_width(row.raw_percentage))
        .max()
        .unwrap_or(3);
    let percentage_width = rows
        .iter()
        .map(|row| fixed_point_width(row.weighted_percentage))
        .max()
        .unwrap_or(3);

    for row in rows {
        let mut table_row = vec![row.group_label];
        if include_ir_text {
            table_row.push(
                row.ir_text
                    .unwrap_or_else(|| "<no PIR attribution>".to_string()),
            );
        }
        table_row.push(format!(
            "{:>raw_width$} ({:>raw_percentage_width$.1}%)",
            row.raw_aig_node_count, row.raw_percentage
        ));
        table_row.push(format!(
            "{:>weighted_width$.1} ({:>percentage_width$.1}%)",
            row.weighted_aig_node_count, row.weighted_percentage
        ));
        table.add_row(table_row);
    }

    let rendered_table = table.to_string();
    let rendered_table = {
        let lines: Vec<&str> = rendered_table.lines().collect();
        if lines.len() >= 2 {
            format!("{}\n{}", lines[1], rendered_table)
        } else {
            rendered_table
        }
    };
    let mut summary_lines = vec![
        format!("function: {}", function_name),
        format!("total_aig_nodes: {}", total_aig_node_count),
    ];
    for (label, value) in extra_summary_lines {
        summary_lines.push(format!("{}: {}", label, value));
    }

    format!("{}\n\n{}", summary_lines.join("\n"), rendered_table)
}

fn to_display_rows_for_pir_node_report(report: &AreaTableReport) -> Vec<DisplayRow> {
    let row_percentage = |count: f64| {
        if report.selected_aig_node_count == 0 {
            0.0
        } else {
            count * 100.0 / report.selected_aig_node_count as f64
        }
    };

    let mut rows: Vec<DisplayRow> = report
        .rows
        .iter()
        .map(|row| DisplayRow {
            group_key: Some(format!("{:010}", row.pir_node_id)),
            group_label: row.pir_node_id.to_string(),
            ir_text: Some(row.ir_text.clone()),
            raw_aig_node_count: row.raw_aig_node_count,
            raw_percentage: row_percentage(row.raw_aig_node_count as f64),
            weighted_aig_node_count: row.weighted_aig_node_count,
            weighted_percentage: row_percentage(row.weighted_aig_node_count),
        })
        .collect();
    push_unattributed_display_row(
        &mut rows,
        report.unattributed.as_ref(),
        row_percentage,
        /* include_ir_text= */ true,
    );
    rows
}

fn to_display_rows_for_opcode_report(report: &OpcodeAreaTableReport) -> Vec<DisplayRow> {
    let row_percentage = |count: f64| {
        if report.selected_aig_node_count == 0 {
            0.0
        } else {
            count * 100.0 / report.selected_aig_node_count as f64
        }
    };

    let mut rows: Vec<DisplayRow> = report
        .rows
        .iter()
        .map(|row| DisplayRow {
            group_key: Some(row.opcode.clone()),
            group_label: row.opcode.clone(),
            ir_text: None,
            raw_aig_node_count: row.raw_aig_node_count,
            raw_percentage: row_percentage(row.raw_aig_node_count as f64),
            weighted_aig_node_count: row.weighted_aig_node_count,
            weighted_percentage: row_percentage(row.weighted_aig_node_count),
        })
        .collect();
    push_unattributed_display_row(
        &mut rows,
        report.unattributed.as_ref(),
        row_percentage,
        /* include_ir_text= */ false,
    );
    rows
}

fn push_unattributed_display_row(
    rows: &mut Vec<DisplayRow>,
    unattributed: Option<&UnattributedAreaTableRow>,
    weighted_percentage: impl Fn(f64) -> f64,
    include_ir_text: bool,
) {
    if let Some(unattributed) = unattributed {
        rows.push(DisplayRow {
            group_key: None,
            group_label: "unattributed".to_string(),
            ir_text: include_ir_text.then_some("<no PIR attribution>".to_string()),
            raw_aig_node_count: unattributed.raw_aig_node_count,
            raw_percentage: weighted_percentage(unattributed.raw_aig_node_count as f64),
            weighted_aig_node_count: unattributed.weighted_aig_node_count,
            weighted_percentage: weighted_percentage(unattributed.weighted_aig_node_count),
        });
    }
}

fn render_pir_node_table(
    report: &AreaTableReport,
    extra_summary_lines: &[(&str, usize)],
) -> String {
    render_rows_table(
        &report.function_name,
        report.total_aig_node_count,
        extra_summary_lines,
        "pir_node_id",
        /* include_ir_text= */ true,
        to_display_rows_for_pir_node_report(report),
    )
}

fn render_opcode_table(
    report: &OpcodeAreaTableReport,
    extra_summary_lines: &[(&str, usize)],
) -> String {
    render_rows_table(
        &report.function_name,
        report.total_aig_node_count,
        extra_summary_lines,
        "opcode",
        /* include_ir_text= */ false,
        to_display_rows_for_opcode_report(report),
    )
}

enum AreaTableOutput {
    PirNode(AreaTableReport),
    Opcode(OpcodeAreaTableReport),
}

impl AreaTableOutput {
    fn missing_pir_node_ids(&self) -> &[u32] {
        match self {
            AreaTableOutput::PirNode(report) => &report.missing_pir_node_ids,
            AreaTableOutput::Opcode(report) => &report.missing_pir_node_ids,
        }
    }
}

pub fn handle_g8r_area_table(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    handle_g8r_attribution_table(
        matches,
        AREA_SUBCOMMAND,
        /* critical_path_only= */ false,
    );
}

pub fn handle_g8r_critical_path_table(
    matches: &clap::ArgMatches,
    _config: &Option<ToolchainConfig>,
) {
    handle_g8r_attribution_table(
        matches,
        CRITICAL_PATH_SUBCOMMAND,
        /* critical_path_only= */ true,
    );
}

fn handle_g8r_attribution_table(
    matches: &clap::ArgMatches,
    subcommand: &str,
    critical_path_only: bool,
) {
    let g8r_path = Path::new(matches.get_one::<String>("g8r_input_file").unwrap());
    let ir_path = Path::new(matches.get_one::<String>("ir_input_file").unwrap());
    let ir_top = matches.get_one::<String>("ir_top").map(|s| s.as_str());
    let group_by_opcode = matches.get_flag("group_by_opcode");

    let gate_fn = load_gate_fn(g8r_path, subcommand).unwrap_or_else(|e| {
        eprintln!("{} error: {}", subcommand, e);
        std::process::exit(2)
    });
    let ir_fn = load_selected_ir_fn(ir_path, ir_top).unwrap_or_else(|e| {
        eprintln!("{} error: {}", subcommand, e);
        std::process::exit(2)
    });

    let report = if group_by_opcode {
        let report = if critical_path_only {
            build_critical_path_opcode_area_table_report(&gate_fn, &ir_fn)
        } else {
            build_opcode_area_table_report(&gate_fn, &ir_fn)
        };
        AreaTableOutput::Opcode(report.unwrap_or_else(|e| {
            eprintln!("{} error: {}", subcommand, e);
            std::process::exit(2)
        }))
    } else {
        let report = if critical_path_only {
            build_critical_path_area_table_report(&gate_fn, &ir_fn)
        } else {
            build_area_table_report(&gate_fn, &ir_fn)
        };
        AreaTableOutput::PirNode(report.unwrap_or_else(|e| {
            eprintln!("{} error: {}", subcommand, e);
            std::process::exit(2)
        }))
    };

    for pir_node_id in report.missing_pir_node_ids() {
        eprintln!(
            "{} warning: PIR node id {} was referenced by selected AIG provenance but was not found in IR member '{}'; its weight was added to the unattributed row.",
            subcommand, pir_node_id, ir_fn.name
        );
    }

    let extra_summary_lines = match &report {
        AreaTableOutput::PirNode(report) if critical_path_only => vec![
            ("critical_path_aig_nodes", report.selected_aig_node_count),
            (
                "critical_path_depth_nodes",
                report.critical_path_depth_nodes.unwrap_or(0),
            ),
        ],
        AreaTableOutput::Opcode(report) if critical_path_only => vec![
            ("critical_path_aig_nodes", report.selected_aig_node_count),
            (
                "critical_path_depth_nodes",
                report.critical_path_depth_nodes.unwrap_or(0),
            ),
        ],
        _ => Vec::new(),
    };

    match &report {
        AreaTableOutput::PirNode(report) => {
            println!("{}", render_pir_node_table(report, &extra_summary_lines))
        }
        AreaTableOutput::Opcode(report) => {
            println!("{}", render_opcode_table(report, &extra_summary_lines))
        }
    }
}

#[cfg(test)]
mod tests {
    use xlsynth_g8r::aig::area_table::{AreaTableReport, AreaTableRow, UnattributedAreaTableRow};

    use super::render_pir_node_table;

    #[test]
    fn test_render_table_right_justifies_integer_area_counts() {
        let report = AreaTableReport {
            function_name: "f".to_string(),
            total_aig_node_count: 16,
            selected_aig_node_count: 16,
            critical_path_depth_nodes: None,
            rows: vec![
                AreaTableRow {
                    pir_node_id: 7,
                    opcode: "add".to_string(),
                    ir_text: "foo: bits[8] = add(a: bits[8], b: bits[8])".to_string(),
                    raw_aig_node_count: 12,
                    weighted_aig_node_count: 12.0,
                },
                AreaTableRow {
                    pir_node_id: 9,
                    opcode: "not".to_string(),
                    ir_text: "bar: bits[8] = not(foo: bits[8])".to_string(),
                    raw_aig_node_count: 3,
                    weighted_aig_node_count: 3.0,
                },
            ],
            unattributed: Some(UnattributedAreaTableRow {
                raw_aig_node_count: 1,
                weighted_aig_node_count: 1.0,
            }),
            missing_pir_node_ids: vec![],
        };

        let rendered = render_pir_node_table(&report, &[]);
        assert!(rendered.starts_with("function: f\ntotal_aig_nodes: 16\n\n|--------------|"));
        let foo_line = rendered
            .lines()
            .find(|line| line.contains("foo: bits[8] = add("))
            .unwrap();
        let bar_line = rendered
            .lines()
            .find(|line| line.contains("bar: bits[8] = not("))
            .unwrap();
        let unattributed_line = rendered
            .lines()
            .find(|line| line.contains("<no PIR attribution>"))
            .unwrap();

        let foo_raw_cell = foo_line.split('|').nth(3).unwrap().trim_end();
        let bar_raw_cell = bar_line.split('|').nth(3).unwrap().trim_end();
        let unattributed_raw_cell = unattributed_line.split('|').nth(3).unwrap().trim_end();

        assert!(foo_raw_cell.ends_with("12 (75.0%)"));
        assert!(bar_raw_cell.ends_with("3 (18.8%)"));
        assert!(unattributed_raw_cell.ends_with("1 ( 6.2%)"));
        assert_eq!(foo_raw_cell.len(), bar_raw_cell.len());
        assert_eq!(foo_raw_cell.len(), unattributed_raw_cell.len());

        let foo_weighted_cell = foo_line.split('|').nth(4).unwrap().trim_end();
        let bar_weighted_cell = bar_line.split('|').nth(4).unwrap().trim_end();
        let unattributed_weighted_cell = unattributed_line.split('|').nth(4).unwrap().trim_end();

        assert!(foo_weighted_cell.ends_with("12.0 (75.0%)"));
        assert!(bar_weighted_cell.ends_with("3.0 (18.8%)"));
        assert!(unattributed_weighted_cell.ends_with("1.0 ( 6.2%)"));
        assert_eq!(foo_weighted_cell.len(), bar_weighted_cell.len());
        assert_eq!(foo_weighted_cell.len(), unattributed_weighted_cell.len());
    }

    #[test]
    fn test_render_table_uses_selected_aig_nodes_for_percentages_and_preamble() {
        let report = AreaTableReport {
            function_name: "f".to_string(),
            total_aig_node_count: 16,
            selected_aig_node_count: 4,
            critical_path_depth_nodes: Some(4),
            rows: vec![AreaTableRow {
                pir_node_id: 7,
                opcode: "add".to_string(),
                ir_text: "foo: bits[8] = add(a: bits[8], b: bits[8])".to_string(),
                raw_aig_node_count: 3,
                weighted_aig_node_count: 2.5,
            }],
            unattributed: None,
            missing_pir_node_ids: vec![],
        };

        let rendered = render_pir_node_table(
            &report,
            &[
                ("critical_path_aig_nodes", 4),
                ("critical_path_depth_nodes", 4),
            ],
        );
        assert!(rendered
            .starts_with("function: f\ntotal_aig_nodes: 16\ncritical_path_aig_nodes: 4\ncritical_path_depth_nodes: 4\n\n|"));
        assert!(rendered.contains("pir_node_id"));
        assert!(rendered.contains("weighted_aig_nodes"));
        let row_line = rendered
            .lines()
            .find(|line| line.contains("foo: bits[8] = add("))
            .unwrap();
        assert!(row_line.contains("| 3 (75.0%) |"));
        assert!(row_line.contains("2.5 (62.5%)"));
    }
}
