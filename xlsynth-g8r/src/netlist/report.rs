// SPDX-License-Identifier: Apache-2.0

//! Area, timing, and level summaries for parsed gate-level netlists.

use crate::liberty::LibraryWithTimingData;
use crate::liberty_proto::Library;
use crate::netlist::io::ParsedNetlist;
use crate::netlist::parse::{Net, NetlistModule, PortDirection, PortId};
use crate::netlist::sta::{StaOptions, analyze_combinational_max_arrival};
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
    pub area: f64,
    pub delay: f64,
    pub cell_count: usize,
    pub cell_levels: usize,
    pub cells: Vec<CellAreaRow>,
    pub outputs: Vec<OutputTimingRow>,
}

/// Resolves one interned netlist symbol with an actionable error.
pub fn resolve_symbol(
    interner: &StringInterner<StringBackend<SymbolU32>>,
    sym: PortId,
    what: &str,
) -> Result<String> {
    interner
        .resolve(sym)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("could not resolve {} symbol", what))
}

/// Selects one module by name, or the only module when no name is provided.
pub fn select_module<'a>(
    parsed: &'a ParsedNetlist,
    module_name: Option<&str>,
) -> Result<&'a NetlistModule> {
    if let Some(name) = module_name {
        for module in &parsed.modules {
            let resolved = resolve_symbol(&parsed.interner, module.name, "module name")?;
            if resolved == name {
                return Ok(module);
            }
        }
        return Err(anyhow!("module '{}' was not found in netlist", name));
    }

    if parsed.modules.len() == 1 {
        return Ok(&parsed.modules[0]);
    }

    let mut names = Vec::with_capacity(parsed.modules.len());
    for module in &parsed.modules {
        names.push(resolve_symbol(
            &parsed.interner,
            module.name,
            "module name",
        )?);
    }
    names.sort();
    Err(anyhow!(
        "netlist contains {} modules; use --module_name; available modules: [{}]",
        parsed.modules.len(),
        names.join(", ")
    ))
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

    let mut outputs = Vec::new();
    for port in &module.ports {
        if port.direction != PortDirection::Output {
            continue;
        }

        let output_name = resolve_symbol(interner, port.name, "output port")?;
        let net_idx = module
            .find_net_index(port.name, nets)
            .ok_or_else(|| anyhow!("output '{}' does not resolve to a net", output_name))?;
        let timing = report
            .timing_for_net(net_idx)
            .ok_or_else(|| anyhow!("output '{}' has no computed timing", output_name))?;
        outputs.push(OutputTimingRow {
            output: output_name,
            rise_arrival: timing.rise.arrival,
            fall_arrival: timing.fall.arrival,
            rise_transition: timing.rise.transition,
            fall_transition: timing.fall.transition,
            worst_arrival: timing.rise.arrival.max(timing.fall.arrival),
        });
    }

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
    let sta = build_sta_report(module, nets, interner, library, options)?;

    Ok(NetlistReport {
        module: area.module,
        time_unit: sta.time_unit,
        primary_input_transition: sta.primary_input_transition,
        module_output_load: sta.module_output_load,
        area: area.area,
        delay: sta.delay,
        cell_count: area.cell_count,
        cell_levels: sta.cell_levels,
        cells: area.cells,
        outputs: sta.outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::{build_area_report, build_netlist_report, select_module};
    use crate::liberty::LibraryWithTimingData;
    use crate::liberty_proto::{Cell, Library, Pin, PinDirection, TimingArc, TimingTable};
    use crate::netlist::io::ParsedNetlist;
    use crate::netlist::parse::{Parser as NetlistParser, TokenScanner};
    use crate::netlist::sta::StaOptions;

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
        assert_eq!(report.area, 3.0);
        assert_eq!(report.delay, 3.0);
        assert_eq!(report.cell_count, 2);
        assert_eq!(report.cell_levels, 2);
        assert_eq!(report.outputs[0].worst_arrival, 3.0);
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
  BUF u0 (.A(a), .Y(y));
endmodule
"#,
        );
        let module = select_module(&parsed, None).expect("select only module");
        let error = build_area_report(module, &parsed.interner, &inv_nand_library())
            .expect_err("unknown cell should be rejected");
        assert!(error.to_string().contains("unknown cell 'BUF'"));
    }
}
