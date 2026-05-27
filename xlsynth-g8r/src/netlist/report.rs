// SPDX-License-Identifier: Apache-2.0

//! Area, timing, and level summaries for parsed gate-level netlists.

use crate::liberty::LibraryWithTimingData;
use crate::liberty_proto::Library;
pub use crate::netlist::io::{resolve_symbol, select_module};
use crate::netlist::parse::{Net, NetlistModule, PortDirection};
use crate::netlist::sta::{
    RegisterPathDelayBreakdown, StaOptions, StaReport, analyze_combinational_max_arrival,
    analyze_register_boundary_max_arrival,
};
use crate::netlist::stages::{StagePartitionStatus, analyze_register_stages};
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// One cell-type contribution to a mapped netlist area summary.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CellAreaRow {
    pub cell: String,
    pub count: usize,
    pub cell_area: f64,
    pub total_area: f64,
}

/// Mapped standard-cell area summary for one selected netlist module.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NetlistAreaReport {
    pub module: String,
    pub cell_count: usize,
    pub area: f64,
    pub cells: Vec<CellAreaRow>,
}

/// One primary-output timing row in a gate-level timing summary.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OutputTimingRow {
    pub output: String,
    pub rise_arrival: f64,
    pub fall_arrival: f64,
    pub rise_transition: f64,
    pub fall_transition: f64,
    pub worst_arrival: f64,
}

/// Combinational timing summary for one selected netlist module.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NetlistStaReport {
    pub module: String,
    pub time_unit: String,
    pub primary_input_transition: f64,
    pub module_output_load: f64,
    pub delay: f64,
    pub cell_levels: usize,
    pub outputs: Vec<OutputTimingRow>,
}

/// Combined mapped area and combinational timing summary for one module.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NetlistReport {
    pub module: String,
    pub time_unit: String,
    pub primary_input_transition: f64,
    pub module_output_load: f64,
    pub cell_area: f64,
    pub max_delay: f64,
    pub max_input_to_register_delay: Option<f64>,
    pub max_input_to_register_delay_breakdown: Option<RegisterPathDelayBreakdown>,
    pub max_register_to_register_delay: Option<f64>,
    pub max_register_to_register_delay_breakdown: Option<RegisterPathDelayBreakdown>,
    pub max_register_to_output_delay: Option<f64>,
    pub max_register_to_output_delay_breakdown: Option<RegisterPathDelayBreakdown>,
    pub cell_count: usize,
    pub cell_levels: usize,
    pub sequential_cell_area: f64,
    pub non_stage_combinational_cell_area: f64,
    pub stage_partition_status: StagePartitionStatus,
    pub stages: Vec<StageReportRow>,
    pub cells: Vec<CellAreaRow>,
    pub outputs: Vec<OutputTimingRow>,
}

/// Timing and combinational area attributed to one adjacent register stage.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StageReportRow {
    pub stage: usize,
    pub max_delay: f64,
    pub max_delay_breakdown: Option<RegisterPathDelayBreakdown>,
    pub combinational_cell_area: f64,
}

/// Builds mapped standard-cell area for one selected module.
pub fn build_area_report(
    module: &NetlistModule,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
) -> Result<NetlistAreaReport> {
    let module_name = resolve_symbol(interner, module.name, "module name")?;
    let mut area_by_cell = HashMap::with_capacity(library.cells.len());
    for cell in &library.cells {
        if !cell.area.is_finite() || cell.area < 0.0 {
            return Err(anyhow!(
                "library cell '{}' has invalid area {}; expected a non-negative finite value",
                cell.name,
                cell.area
            ));
        }
        if area_by_cell.insert(cell.name.clone(), cell.area).is_some() {
            return Err(anyhow!(
                "library defines cell '{}' more than once; duplicate cell names are unsupported in area reporting",
                cell.name
            ));
        }
    }

    let mut counts = BTreeMap::<String, usize>::new();
    for instance in &module.instances {
        let cell_name = resolve_symbol(interner, instance.type_name, "cell type")?;
        if !area_by_cell.contains_key(cell_name.as_str()) {
            let instance_name = resolve_symbol(interner, instance.instance_name, "instance name")
                .unwrap_or_else(|_| "<unknown>".to_string());
            return Err(anyhow!(
                "instance '{}' references unknown cell '{}'",
                instance_name,
                cell_name
            ));
        }
        *counts.entry(cell_name).or_insert(0) += 1;
    }

    let mut cells = Vec::with_capacity(counts.len());
    let mut area = 0.0;
    for (cell, count) in counts {
        let cell_area = area_by_cell[&cell];
        let total_area = cell_area * count as f64;
        area += total_area;
        cells.push(CellAreaRow {
            cell,
            count,
            cell_area,
            total_area,
        });
    }
    cells.sort_by(|lhs, rhs| {
        rhs.total_area
            .total_cmp(&lhs.total_area)
            .then(rhs.count.cmp(&lhs.count))
            .then(lhs.cell.cmp(&rhs.cell))
    });

    Ok(NetlistAreaReport {
        module: module_name,
        cell_count: module.instances.len(),
        area,
        cells,
    })
}

/// Builds combinational timing and level metrics for one selected module.
pub fn build_sta_report(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &LibraryWithTimingData,
    options: StaOptions,
) -> Result<NetlistStaReport> {
    let module_name = resolve_symbol(interner, module.name, "module name")?;
    let report = analyze_combinational_max_arrival(module, nets, interner, library, options)?;
    let outputs = output_timing_rows(module, nets, interner, &report, true)?;

    let time_unit = library
        .units
        .as_ref()
        .map(|u| u.time_unit.as_str())
        .unwrap_or("")
        .to_string();

    Ok(NetlistStaReport {
        module: module_name,
        time_unit,
        primary_input_transition: options.primary_input_transition,
        module_output_load: options.module_output_load,
        delay: report.worst_output_arrival,
        cell_levels: report.cell_levels,
        outputs,
    })
}

fn output_timing_rows(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    report: &StaReport,
    require_all: bool,
) -> Result<Vec<OutputTimingRow>> {
    let mut outputs = Vec::new();
    for port in &module.ports {
        if port.direction != PortDirection::Output {
            continue;
        }

        let output_name = resolve_symbol(interner, port.name, "output port")?;
        let net_idx = module
            .find_net_index(port.name, nets)
            .ok_or_else(|| anyhow!("output '{}' does not resolve to a net", output_name))?;
        let Some(timing) = report.timing_for_net(net_idx) else {
            if require_all {
                return Err(anyhow!("output '{}' has no computed timing", output_name));
            }
            continue;
        };
        outputs.push(OutputTimingRow {
            output: output_name,
            rise_arrival: timing.rise.arrival,
            fall_arrival: timing.fall.arrival,
            rise_transition: timing.rise.transition,
            fall_transition: timing.fall.transition,
            worst_arrival: timing.rise.arrival.max(timing.fall.arrival),
        });
    }
    Ok(outputs)
}

/// Builds mapped area plus combinational timing metrics for one selected
/// module.
pub fn build_netlist_report(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &LibraryWithTimingData,
    options: StaOptions,
) -> Result<NetlistReport> {
    let area = build_area_report(module, interner, library.as_proto())?;
    let stages = analyze_register_stages(module, nets, interner, library.as_proto())?;
    if stages.register_indices.is_empty() {
        let sta = build_sta_report(module, nets, interner, library, options)?;
        return Ok(NetlistReport {
            module: area.module,
            time_unit: sta.time_unit,
            primary_input_transition: sta.primary_input_transition,
            module_output_load: sta.module_output_load,
            cell_area: area.area,
            max_delay: sta.delay,
            max_input_to_register_delay: None,
            max_input_to_register_delay_breakdown: None,
            max_register_to_register_delay: None,
            max_register_to_register_delay_breakdown: None,
            max_register_to_output_delay: None,
            max_register_to_output_delay_breakdown: None,
            cell_count: area.cell_count,
            cell_levels: sta.cell_levels,
            sequential_cell_area: stages.sequential_area,
            non_stage_combinational_cell_area: stages.non_stage_combinational_area,
            stage_partition_status: stages.status,
            stages: Vec::new(),
            cells: area.cells,
            outputs: sta.outputs,
        });
    }

    let input_launch =
        analyze_register_boundary_max_arrival(module, nets, interner, library, options, true, &[])?;
    let register_launch = analyze_register_boundary_max_arrival(
        module,
        nets,
        interner,
        library,
        options,
        false,
        stages.register_indices.as_slice(),
    )?;
    let mut stage_rows = Vec::new();
    if stages.status == StagePartitionStatus::Partitioned {
        for (stage, stage_area) in &stages.stage_areas {
            let launch_registers: Vec<usize> = stages
                .register_indices
                .iter()
                .copied()
                .filter(|idx| stages.register_levels[*idx] == Some(*stage))
                .collect();
            let stage_timing = analyze_register_boundary_max_arrival(
                module,
                nets,
                interner,
                library,
                options,
                false,
                launch_registers.as_slice(),
            )?;
            let mut delay = 0.0;
            let mut max_delay_breakdown = None;
            let mut found_delay = false;
            for idx in stages
                .register_indices
                .iter()
                .copied()
                .filter(|idx| stages.register_levels[*idx] == Some(*stage + 1))
            {
                let Some(candidate_delay) = stage_timing.register_input_arrivals[idx] else {
                    continue;
                };
                if !found_delay || candidate_delay > delay {
                    found_delay = true;
                    delay = candidate_delay;
                    max_delay_breakdown = stage_timing.register_input_breakdowns[idx];
                }
            }
            stage_rows.push(StageReportRow {
                stage: *stage,
                max_delay: delay,
                max_delay_breakdown,
                combinational_cell_area: *stage_area,
            });
        }
    }
    let outputs = output_timing_rows(module, nets, interner, &input_launch, false)?;
    let time_unit = library
        .units
        .as_ref()
        .map(|u| u.time_unit.as_str())
        .unwrap_or("")
        .to_string();

    Ok(NetlistReport {
        module: area.module,
        time_unit,
        primary_input_transition: options.primary_input_transition,
        module_output_load: options.module_output_load,
        cell_area: area.area,
        max_delay: input_launch.worst_output_arrival,
        max_input_to_register_delay: Some(input_launch.worst_register_input_arrival),
        max_input_to_register_delay_breakdown: input_launch.worst_register_input_breakdown,
        max_register_to_register_delay: Some(register_launch.worst_register_input_arrival),
        max_register_to_register_delay_breakdown: register_launch.worst_register_input_breakdown,
        max_register_to_output_delay: Some(register_launch.worst_output_arrival),
        max_register_to_output_delay_breakdown: register_launch.worst_output_breakdown,
        cell_count: area.cell_count,
        cell_levels: input_launch.cell_levels,
        sequential_cell_area: stages.sequential_area,
        non_stage_combinational_cell_area: stages.non_stage_combinational_area,
        stage_partition_status: stages.status,
        stages: stage_rows,
        cells: area.cells,
        outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::{build_area_report, build_netlist_report, select_module};
    use crate::liberty::LibraryWithTimingData;
    use crate::liberty_proto::{
        Cell, Library, Pin, PinDirection, Sequential, SequentialKind, TimingArc, TimingTable,
    };
    use crate::netlist::io::ParsedNetlist;
    use crate::netlist::parse::{Parser as NetlistParser, TokenScanner};
    use crate::netlist::sta::{RegisterPathDelayBreakdown, StaOptions};
    use crate::netlist::stages::StagePartitionStatus;

    fn parse_netlist(src: &'static str) -> ParsedNetlist {
        let scanner = TokenScanner::from_str(src);
        let mut parser = NetlistParser::new(scanner);
        let modules = parser.parse_file().expect("parse netlist");
        ParsedNetlist {
            modules,
            nets: parser.nets,
            interner: parser.interner,
        }
    }

    fn timing_arc(related_pin: &str, rise: f64, fall: f64) -> TimingArc {
        TimingArc {
            related_pin: related_pin.to_string(),
            timing_sense: "positive_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                TimingTable {
                    kind: "cell_rise".to_string(),
                    values: vec![rise],
                    ..Default::default()
                },
                TimingTable {
                    kind: "cell_fall".to_string(),
                    values: vec![fall],
                    ..Default::default()
                },
                TimingTable {
                    kind: "rise_transition".to_string(),
                    values: vec![0.1],
                    ..Default::default()
                },
                TimingTable {
                    kind: "fall_transition".to_string(),
                    values: vec![0.1],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn clock_to_output_arc() -> TimingArc {
        TimingArc {
            related_pin: "CLK".to_string(),
            timing_sense: "non_unate".to_string(),
            timing_type: "rising_edge".to_string(),
            tables: vec![
                TimingTable {
                    kind: "cell_rise".to_string(),
                    values: vec![0.5],
                    ..Default::default()
                },
                TimingTable {
                    kind: "cell_fall".to_string(),
                    values: vec![0.5],
                    ..Default::default()
                },
                TimingTable {
                    kind: "rise_transition".to_string(),
                    values: vec![0.1],
                    ..Default::default()
                },
                TimingTable {
                    kind: "fall_transition".to_string(),
                    values: vec![0.1],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn setup_arc() -> TimingArc {
        TimingArc {
            related_pin: "CLK".to_string(),
            timing_type: "setup_rising".to_string(),
            tables: vec![
                TimingTable {
                    kind: "rise_constraint".to_string(),
                    values: vec![0.25],
                    ..Default::default()
                },
                TimingTable {
                    kind: "fall_constraint".to_string(),
                    values: vec![0.25],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn inv_nand_library() -> Library {
        Library {
            cells: vec![
                Cell {
                    name: "INV".to_string(),
                    pins: vec![
                        Pin {
                            name: "A".to_string(),
                            direction: PinDirection::Input as i32,
                            capacitance: Some(0.0),
                            ..Default::default()
                        },
                        Pin {
                            name: "Y".to_string(),
                            direction: PinDirection::Output as i32,
                            function: "!A".to_string(),
                            timing_arcs: vec![timing_arc("A", 1.0, 1.0)],
                            ..Default::default()
                        },
                    ],
                    area: 1.0,
                    ..Default::default()
                },
                Cell {
                    name: "BUF".to_string(),
                    pins: vec![
                        Pin {
                            name: "A".to_string(),
                            direction: PinDirection::Input as i32,
                            capacitance: Some(0.0),
                            ..Default::default()
                        },
                        Pin {
                            name: "Y".to_string(),
                            direction: PinDirection::Output as i32,
                            function: "A".to_string(),
                            timing_arcs: vec![timing_arc("A", 1.0, 1.0)],
                            ..Default::default()
                        },
                    ],
                    area: 1.0,
                    ..Default::default()
                },
                Cell {
                    name: "NAND2".to_string(),
                    pins: vec![
                        Pin {
                            name: "A".to_string(),
                            direction: PinDirection::Input as i32,
                            capacitance: Some(0.0),
                            ..Default::default()
                        },
                        Pin {
                            name: "B".to_string(),
                            direction: PinDirection::Input as i32,
                            capacitance: Some(0.0),
                            ..Default::default()
                        },
                        Pin {
                            name: "Y".to_string(),
                            direction: PinDirection::Output as i32,
                            function: "!(A*B)".to_string(),
                            timing_arcs: vec![timing_arc("A", 2.0, 2.0), timing_arc("B", 2.0, 2.0)],
                            ..Default::default()
                        },
                    ],
                    area: 2.0,
                    ..Default::default()
                },
                Cell {
                    name: "DFF".to_string(),
                    pins: vec![
                        Pin {
                            name: "D".to_string(),
                            direction: PinDirection::Input as i32,
                            capacitance: Some(0.0),
                            timing_arcs: vec![setup_arc()],
                            ..Default::default()
                        },
                        Pin {
                            name: "CLK".to_string(),
                            direction: PinDirection::Input as i32,
                            is_clocking_pin: true,
                            capacitance: Some(0.0),
                            ..Default::default()
                        },
                        Pin {
                            name: "Q".to_string(),
                            direction: PinDirection::Output as i32,
                            timing_arcs: vec![clock_to_output_arc()],
                            ..Default::default()
                        },
                    ],
                    area: 4.0,
                    sequential: vec![Sequential {
                        state_var: "IQ".to_string(),
                        next_state: "D".to_string(),
                        clock_expr: "CLK".to_string(),
                        kind: SequentialKind::Ff as i32,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn build_area_report_sums_selected_module_cells() {
        let parsed = parse_netlist(
            r#"
module fast (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule

module slow (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  NAND2 u0 (.A(a), .B(a), .Y(n0));
  INV u1 (.A(n0), .Y(y));
endmodule
"#,
        );
        let module = select_module(&parsed, Some("slow")).expect("select slow module");
        let report = build_area_report(module, &parsed.interner, &inv_nand_library())
            .expect("build area report");
        assert_eq!(report.module, "slow");
        assert_eq!(report.cell_count, 2);
        assert_eq!(report.area, 3.0);
        assert_eq!(report.cells[0].cell, "NAND2");
        assert_eq!(report.cells[0].total_area, 2.0);
        assert_eq!(report.cells[1].cell, "INV");
        assert_eq!(report.cells[1].total_area, 1.0);
    }

    #[test]
    fn build_netlist_report_reports_area_delay_and_levels() {
        let parsed = parse_netlist(
            r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  wire n0;
  NAND2 u0 (.A(a), .B(b), .Y(n0));
  INV u1 (.A(n0), .Y(y));
endmodule
"#,
        );
        let module = select_module(&parsed, None).expect("select only module");
        let library = LibraryWithTimingData::from_proto(inv_nand_library());
        let report = build_netlist_report(
            module,
            &parsed.nets,
            &parsed.interner,
            &library,
            StaOptions::default(),
        )
        .expect("build report");
        assert_eq!(report.cell_area, 3.0);
        assert_eq!(report.max_delay, 3.0);
        assert_eq!(report.cell_count, 2);
        assert_eq!(report.cell_levels, 2);
        assert_eq!(report.outputs[0].worst_arrival, 3.0);
    }

    #[test]
    fn build_netlist_report_accepts_yosys_concat_alias_assign() {
        let parsed = parse_netlist(
            r#"
module top (x, y);
  input [31:0] x;
  output y;
  wire [31:0] x;
  wire y;
  wire [8:0] exp_x_signed__2;
  assign exp_x_signed__2 = { 1'h0, x[30:23] };
  BUF u0 (.A(exp_x_signed__2[8]), .Y(y));
endmodule
"#,
        );
        let module = select_module(&parsed, None).expect("select only module");
        let library = LibraryWithTimingData::from_proto(inv_nand_library());
        let report = build_netlist_report(
            module,
            &parsed.nets,
            &parsed.interner,
            &library,
            StaOptions::default(),
        )
        .expect("build report with Yosys concat alias assign");
        assert_eq!(report.cell_area, 1.0);
        assert_eq!(report.max_delay, 1.0);
        assert_eq!(report.cell_count, 1);
        assert_eq!(report.cell_levels, 1);
    }

    #[test]
    fn build_netlist_report_accepts_yosys_tran_alias() {
        let parsed = parse_netlist(
            r#"
module top (x, y);
  input x;
  output y;
  wire x;
  wire y;
  wire y_alias;
  BUF u0 (.A(x), .Y(y_alias));
  tran(y, y_alias);
endmodule
"#,
        );
        let module = select_module(&parsed, None).expect("select only module");
        let library = LibraryWithTimingData::from_proto(inv_nand_library());
        let report = build_netlist_report(
            module,
            &parsed.nets,
            &parsed.interner,
            &library,
            StaOptions::default(),
        )
        .expect("build report with Yosys tran alias");
        assert_eq!(report.cell_area, 1.0);
        assert_eq!(report.max_delay, 1.0);
        assert_eq!(report.cell_count, 1);
        assert_eq!(report.cell_levels, 1);
    }

    #[test]
    fn build_netlist_report_partitions_register_pipeline_and_accounts_area() {
        let parsed = parse_netlist(
            r#"
module top (a, clk, y);
  input a;
  input clk;
  output y;
  wire a;
  wire clk;
  wire y;
  wire d0;
  wire q0;
  wire d1;
  wire q1;
  INV in_logic (.A(a), .Y(d0));
  DFF r0 (.D(d0), .CLK(clk), .Q(q0));
  INV stage_logic (.A(q0), .Y(d1));
  DFF r1 (.D(d1), .CLK(clk), .Q(q1));
  INV out_logic (.A(q1), .Y(y));
endmodule
"#,
        );
        let module = select_module(&parsed, None).expect("select only module");
        let library = LibraryWithTimingData::from_proto(inv_nand_library());
        let report = build_netlist_report(
            module,
            &parsed.nets,
            &parsed.interner,
            &library,
            StaOptions::default(),
        )
        .expect("build registered report");
        assert_eq!(
            report.stage_partition_status,
            StagePartitionStatus::Partitioned
        );
        assert_eq!(report.max_delay, 0.0);
        assert_eq!(report.max_input_to_register_delay, Some(1.25));
        assert_eq!(
            report.max_input_to_register_delay_breakdown,
            Some(RegisterPathDelayBreakdown {
                clock_to_output_delay: 0.0,
                combinational_delay: 1.0,
                setup_delay: 0.25,
            })
        );
        assert_eq!(report.max_register_to_register_delay, Some(1.75));
        assert_eq!(
            report.max_register_to_register_delay_breakdown,
            Some(RegisterPathDelayBreakdown {
                clock_to_output_delay: 0.5,
                combinational_delay: 1.0,
                setup_delay: 0.25,
            })
        );
        assert_eq!(report.max_register_to_output_delay, Some(1.5));
        assert_eq!(
            report.max_register_to_output_delay_breakdown,
            Some(RegisterPathDelayBreakdown {
                clock_to_output_delay: 0.5,
                combinational_delay: 1.0,
                setup_delay: 0.0,
            })
        );
        assert_eq!(report.sequential_cell_area, 8.0);
        assert_eq!(report.non_stage_combinational_cell_area, 2.0);
        assert_eq!(report.stages.len(), 1);
        assert_eq!(report.stages[0].stage, 0);
        assert_eq!(report.stages[0].max_delay, 1.75);
        assert_eq!(
            report.stages[0].max_delay_breakdown,
            Some(RegisterPathDelayBreakdown {
                clock_to_output_delay: 0.5,
                combinational_delay: 1.0,
                setup_delay: 0.25,
            })
        );
        assert_eq!(report.stages[0].combinational_cell_area, 1.0);
        assert_eq!(
            report.cell_area,
            report.sequential_cell_area
                + report.non_stage_combinational_cell_area
                + report
                    .stages
                    .iter()
                    .map(|stage| stage.combinational_cell_area)
                    .sum::<f64>()
        );
    }

    #[test]
    fn build_netlist_report_partitions_multi_cell_register_stage() {
        let parsed = parse_netlist(
            r#"
module top (a, clk, y);
  input a;
  input clk;
  output y;
  wire a;
  wire clk;
  wire y;
  wire q0;
  wire n0;
  wire d1;
  wire q1;
  DFF r0 (.D(a), .CLK(clk), .Q(q0));
  INV stage_logic_0 (.A(q0), .Y(n0));
  INV stage_logic_1 (.A(n0), .Y(d1));
  DFF r1 (.D(d1), .CLK(clk), .Q(q1));
  INV out_logic (.A(q1), .Y(y));
endmodule
"#,
        );
        let module = select_module(&parsed, None).expect("select only module");
        let library = LibraryWithTimingData::from_proto(inv_nand_library());
        let report = build_netlist_report(
            module,
            &parsed.nets,
            &parsed.interner,
            &library,
            StaOptions::default(),
        )
        .expect("build two-cell registered report");
        assert_eq!(
            report.stage_partition_status,
            StagePartitionStatus::Partitioned
        );
        assert_eq!(report.max_register_to_register_delay, Some(2.75));
        assert_eq!(report.stages.len(), 1);
        assert_eq!(report.stages[0].max_delay, 2.75);
        assert_eq!(report.stages[0].combinational_cell_area, 2.0);
        assert_eq!(report.non_stage_combinational_cell_area, 1.0);
    }

    #[test]
    fn build_netlist_report_omits_stages_for_forward_register_bypass() {
        let parsed = parse_netlist(
            r#"
module top (a, clk, y);
  input a;
  input clk;
  output y;
  wire a;
  wire clk;
  wire y;
  wire q0;
  wire d1;
  wire q1;
  wire d2;
  wire q2;
  DFF r0 (.D(a), .CLK(clk), .Q(q0));
  INV stage0 (.A(q0), .Y(d1));
  DFF r1 (.D(d1), .CLK(clk), .Q(q1));
  NAND2 bypass (.A(q0), .B(q1), .Y(d2));
  DFF r2 (.D(d2), .CLK(clk), .Q(q2));
  INV out_logic (.A(q2), .Y(y));
endmodule
"#,
        );
        let module = select_module(&parsed, None).expect("select only module");
        let library = LibraryWithTimingData::from_proto(inv_nand_library());
        let report = build_netlist_report(
            module,
            &parsed.nets,
            &parsed.interner,
            &library,
            StaOptions::default(),
        )
        .expect("build bypass report");
        assert_eq!(
            report.stage_partition_status,
            StagePartitionStatus::NotPartitionable
        );
        assert!(report.stages.is_empty());
        assert_eq!(report.sequential_cell_area, 12.0);
        assert_eq!(report.non_stage_combinational_cell_area, 4.0);
        assert_eq!(report.max_register_to_register_delay, Some(2.75));
        assert_eq!(report.cell_area, 16.0);
    }

    #[test]
    fn build_netlist_report_ignores_self_feedback_for_stage_partitioning() {
        let parsed = parse_netlist(
            r#"
module top (a, clk, y);
  input a;
  input clk;
  output y;
  wire a;
  wire clk;
  wire y;
  wire q0;
  wire d1;
  wire q1;
  DFF r0 (.D(a), .CLK(clk), .Q(q0));
  NAND2 recurrent_stage (.A(q0), .B(q1), .Y(d1));
  DFF r1 (.D(d1), .CLK(clk), .Q(q1));
  INV out_logic (.A(q1), .Y(y));
endmodule
"#,
        );
        let module = select_module(&parsed, None).expect("select only module");
        let library = LibraryWithTimingData::from_proto(inv_nand_library());
        let report = build_netlist_report(
            module,
            &parsed.nets,
            &parsed.interner,
            &library,
            StaOptions::default(),
        )
        .expect("build recurrent report");
        assert_eq!(
            report.stage_partition_status,
            StagePartitionStatus::Partitioned
        );
        assert_eq!(report.max_register_to_register_delay, Some(2.75));
        assert_eq!(report.stages.len(), 1);
        assert_eq!(report.stages[0].stage, 0);
        assert_eq!(report.stages[0].max_delay, 2.75);
        assert_eq!(report.stages[0].combinational_cell_area, 0.0);
        assert_eq!(report.sequential_cell_area, 8.0);
        assert_eq!(report.non_stage_combinational_cell_area, 3.0);
        assert_eq!(report.cell_area, 11.0);
    }

    #[test]
    fn build_area_report_rejects_unknown_cells() {
        let parsed = parse_netlist(
            r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  UNKNOWN u0 (.A(a), .Y(y));
endmodule
"#,
        );
        let module = select_module(&parsed, None).expect("select only module");
        let error = build_area_report(module, &parsed.interner, &inv_nand_library())
            .expect_err("unknown cell should be rejected");
        assert!(error.to_string().contains("unknown cell 'UNKNOWN'"));
    }
}
