// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use std::cmp::Reverse;
use std::path::Path;
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::liberty::cell_formula::{parse_formula, Term};
use xlsynth_g8r::liberty_proto::{Cell, Library, PinDirection};
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::load_liberty_with_timing_data_from_path;
use xlsynth_g8r::netlist::sta::validate_output_pin_for_basic_sta;
use xlsynth_g8r::netlist::techmap::{map_gatefn_to_structural_netlist, StructuralTechMapOptions};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CellPolicy {
    SmallNormalVt,
    MaxSpeed,
}

impl CellPolicy {
    fn from_cli(value: &str) -> Result<Self, String> {
        match value {
            "small-normal-vt" => Ok(Self::SmallNormalVt),
            "max-speed" => Ok(Self::MaxSpeed),
            _ => Err(format!(
                "unknown --cell_policy '{}'; expected one of: small-normal-vt, max-speed",
                value
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SmallNormalVt => "small-normal-vt",
            Self::MaxSpeed => "max-speed",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SelectedCellBinding {
    cell_name: String,
    input_pin_names: Vec<String>,
    output_pin_name: String,
}

fn input_pin_names(cell: &Cell) -> Vec<String> {
    let mut names = cell
        .pins
        .iter()
        .filter(|p| p.direction == PinDirection::Input as i32)
        .map(|p| p.name.clone())
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn timed_output_pin_names(cell: &Cell) -> Vec<String> {
    let mut names = cell
        .pins
        .iter()
        .filter(|p| p.direction == PinDirection::Output as i32 && !p.timing_arcs.is_empty())
        .map(|p| p.name.clone())
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn output_pin_function<'a>(cell: &'a Cell, output_pin_name: &str) -> Option<&'a str> {
    cell.pins
        .iter()
        .find(|p| p.name == output_pin_name && p.direction == PinDirection::Output as i32)
        .map(|p| p.function.as_str())
}

fn is_inv_formula(term: &Term, input_pin_name: &str) -> bool {
    *term == Term::Negate(Box::new(Term::Input(input_pin_name.to_string())))
}

fn is_nand2_formula(term: &Term, input_pin_names: &[String]) -> bool {
    let [a, b] = input_pin_names else {
        return false;
    };
    // Liberty libraries may spell NAND2 either directly or in equivalent SOP
    // form, so accept both exact two-input AST shapes after parsing.
    let expected_ab = Term::Negate(Box::new(Term::And(
        Box::new(Term::Input(a.clone())),
        Box::new(Term::Input(b.clone())),
    )));
    let expected_ba = Term::Negate(Box::new(Term::And(
        Box::new(Term::Input(b.clone())),
        Box::new(Term::Input(a.clone())),
    )));
    let expected_demorgan_ab = Term::Or(
        Box::new(Term::Negate(Box::new(Term::Input(a.clone())))),
        Box::new(Term::Negate(Box::new(Term::Input(b.clone())))),
    );
    let expected_demorgan_ba = Term::Or(
        Box::new(Term::Negate(Box::new(Term::Input(b.clone())))),
        Box::new(Term::Negate(Box::new(Term::Input(a.clone())))),
    );
    *term == expected_ab
        || *term == expected_ba
        || *term == expected_demorgan_ab
        || *term == expected_demorgan_ba
}

fn cell_binding_if_shape_matches(
    lib: &Library,
    cell: &Cell,
    family_prefix: &str,
    expected_input_count: usize,
) -> Option<SelectedCellBinding> {
    if !cell.name.starts_with(family_prefix) {
        return None;
    }
    let input_pin_names = input_pin_names(cell);
    let output_pin_names = timed_output_pin_names(cell);
    if input_pin_names.len() != expected_input_count || output_pin_names.len() != 1 {
        return None;
    }
    let output_pin_name = &output_pin_names[0];
    let output_function = output_pin_function(cell, output_pin_name)?;
    let output_term = parse_formula(output_function).ok()?;
    let formula_matches = match family_prefix {
        "INV" => is_inv_formula(&output_term, &input_pin_names[0]),
        "NAND2" => is_nand2_formula(&output_term, &input_pin_names),
        _ => false,
    };
    if !formula_matches {
        return None;
    }
    let output_pin = cell
        .pins
        .iter()
        .find(|pin| pin.name == *output_pin_name)
        .expect("timed output pin should still be present");
    if let Err(err) =
        validate_output_pin_for_basic_sta(lib, cell.name.as_str(), output_pin, &input_pin_names)
    {
        log::warn!(
            "skipping {} candidate '{}' because it is incompatible with basic STA: {:#}",
            family_prefix,
            cell.name,
            err
        );
        return None;
    }
    Some(SelectedCellBinding {
        cell_name: cell.name.clone(),
        input_pin_names,
        output_pin_name: output_pin_name.clone(),
    })
}

fn is_inv_candidate(lib: &Library, cell: &Cell) -> bool {
    cell_binding_if_shape_matches(lib, cell, "INV", 1).is_some()
}

fn is_nand2_candidate(lib: &Library, cell: &Cell) -> bool {
    cell_binding_if_shape_matches(lib, cell, "NAND2", 2).is_some()
}

/// Returns the effective VT class for a cell when the library provides
/// ordering.
fn effective_vt_class(lib: &Library, cell: &Cell) -> Result<Option<i32>, String> {
    let group_id = if cell.threshold_voltage_group_id != 0 {
        cell.threshold_voltage_group_id
    } else {
        lib.default_threshold_voltage_group_id
    };
    if group_id == 0 {
        return Ok(None);
    }
    let group_index = group_id as usize - 1;
    if group_index >= lib.threshold_voltage_groups.len() {
        return Err(format!(
            "Cell '{}' effective threshold-voltage group ID {} is out of range",
            cell.name, group_id
        ));
    }
    if lib.threshold_voltage_group_class_indices.is_empty() {
        return Ok(None);
    }
    let Some(class_index) = lib
        .threshold_voltage_group_class_indices
        .get(group_index)
        .copied()
    else {
        return Err(format!(
            "Cell '{}' effective threshold-voltage group '{}' has no class index",
            cell.name, lib.threshold_voltage_groups[group_index]
        ));
    };
    Ok(Some(class_index))
}

fn drive_x1_priority(name: &str, base: &str) -> usize {
    if name == base {
        return 0;
    }
    let needle = format!("{}x1_", base);
    if name.contains(needle.as_str()) || name.ends_with(&format!("{}x1", base)) {
        1
    } else {
        2
    }
}

fn parse_drive_strength_milli(name: &str, base: &str) -> Option<i64> {
    if !name.starts_with(base) {
        return None;
    }
    let rest = name.strip_prefix(base)?;
    let rest = rest.strip_prefix('x')?;
    let end = rest.find('_').unwrap_or(rest.len());
    let token = &rest[..end];
    if token.is_empty() {
        return None;
    }

    if let Some(frac_digits) = token.strip_prefix('p') {
        if frac_digits.is_empty() {
            return None;
        }
        let frac = frac_digits.parse::<i64>().ok()?;
        let denom = 10i64.pow(frac_digits.len() as u32);
        return Some((frac * 1000) / denom);
    }

    let whole = token.parse::<i64>().ok()?;
    Some(whole * 1000)
}

fn best_cell_name<'a>(
    lib: &Library,
    candidates: &[&'a Cell],
    exact_base: &str,
    policy: CellPolicy,
) -> Result<Option<&'a str>, String> {
    let candidates_with_class = candidates
        .iter()
        .map(|cell| effective_vt_class(lib, cell).map(|class_index| (*cell, class_index)))
        .collect::<Result<Vec<_>, _>>()?;
    let has_any_class = candidates_with_class
        .iter()
        .any(|(_, class_index)| class_index.is_some());
    let has_missing_class = candidates_with_class
        .iter()
        .any(|(_, class_index)| class_index.is_none());
    if has_any_class && has_missing_class {
        return Err(format!(
            "{} candidates mix ordered and unordered threshold-voltage metadata",
            exact_base
        ));
    }

    match policy {
        CellPolicy::SmallNormalVt => Ok(candidates_with_class
            .iter()
            .map(|(cell, class_index)| (cell.name.as_str(), *class_index))
            .min_by_key(|(name, class_index)| {
                (
                    if class_index.map_or(true, |class| class == 0) {
                        0usize
                    } else {
                        1usize
                    },
                    if *name == exact_base { 0usize } else { 1usize },
                    drive_x1_priority(name, exact_base),
                    *name,
                )
            })
            .map(|(name, _)| name)),
        CellPolicy::MaxSpeed => Ok(candidates_with_class
            .iter()
            .map(|(cell, class_index)| (cell.name.as_str(), *class_index))
            .min_by_key(|(name, class_index)| {
                let drive = parse_drive_strength_milli(name, exact_base);
                (
                    Reverse(class_index.unwrap_or(0)),
                    if drive.is_some() { 0usize } else { 1usize },
                    Reverse(drive.unwrap_or(0)),
                    *name,
                )
            })
            .map(|(name, _)| name)),
    }
}

fn selected_binding_for_name(
    lib: &Library,
    selected_name: &str,
    family_prefix: &str,
    expected_input_count: usize,
) -> Result<SelectedCellBinding, String> {
    let cell = lib
        .cells
        .iter()
        .find(|cell| cell.name == selected_name)
        .ok_or_else(|| format!("selected cell '{}' disappeared", selected_name))?;
    cell_binding_if_shape_matches(lib, cell, family_prefix, expected_input_count).ok_or_else(|| {
        format!(
            "selected cell '{}' no longer matches expected {} shape",
            selected_name, family_prefix
        )
    })
}

fn select_inv_cell(
    lib: &xlsynth_g8r::liberty_proto::Library,
    policy: CellPolicy,
) -> Result<SelectedCellBinding, String> {
    let candidates: Vec<&Cell> = lib
        .cells
        .iter()
        .filter(|c| is_inv_candidate(lib, c))
        .collect();
    let Some(name) = best_cell_name(lib, candidates.as_slice(), "INV", policy)? else {
        return Err(
            "could not find an INV candidate with one input pin, one timed output pin, and complete combinational timing coverage"
                .to_string(),
        );
    };
    selected_binding_for_name(lib, name, "INV", 1)
}

fn select_nand2_cell(
    lib: &xlsynth_g8r::liberty_proto::Library,
    policy: CellPolicy,
) -> Result<SelectedCellBinding, String> {
    let candidates: Vec<&Cell> = lib
        .cells
        .iter()
        .filter(|c| is_nand2_candidate(lib, c))
        .collect();
    let Some(name) = best_cell_name(lib, candidates.as_slice(), "NAND2", policy)? else {
        return Err(
            "could not find a NAND2 candidate with two input pins, one timed output pin, and complete combinational timing coverage"
                .to_string(),
        );
    };
    selected_binding_for_name(lib, name, "NAND2", 2)
}

#[cfg(test)]
fn select_inv_cell_name(lib: &Library, policy: CellPolicy) -> Result<String, String> {
    select_inv_cell(lib, policy).map(|binding| binding.cell_name)
}

#[cfg(test)]
fn select_nand2_cell_name(lib: &Library, policy: CellPolicy) -> Result<String, String> {
    select_nand2_cell(lib, policy).map(|binding| binding.cell_name)
}

pub fn handle_aig_tech_map(matches: &ArgMatches) {
    let aig_input_file = matches
        .get_one::<String>("aig_input_file")
        .expect("aig_input_file is required");
    let liberty_proto_path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required");
    let netlist_out = matches
        .get_one::<String>("netlist_out")
        .expect("netlist_out is required");
    let module_name = matches.get_one::<String>("module_name").cloned();
    let cell_policy = matches
        .get_one::<String>("cell_policy")
        .map(|s| s.as_str())
        .unwrap_or("small-normal-vt");
    let cell_policy = match CellPolicy::from_cli(cell_policy) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("aig-tech-map error: {}", e);
            std::process::exit(1);
        }
    };

    let load_res =
        load_aiger_auto_from_path(Path::new(aig_input_file), GateBuilderOptions::no_opt())
            .map_err(|e| format!("failed to load AIG '{}': {}", aig_input_file, e));
    let gate_fn = match load_res {
        Ok(r) => r.gate_fn,
        Err(e) => {
            eprintln!("aig-tech-map error: {}", e);
            std::process::exit(1);
        }
    };

    let lib = match load_liberty_with_timing_data_from_path(Path::new(liberty_proto_path)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "aig-tech-map error: failed to load timing-enabled Liberty proto '{}': {:#}",
                liberty_proto_path, e
            );
            std::process::exit(1);
        }
    };

    let inv_cell = match select_inv_cell(&lib, cell_policy) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("aig-tech-map error: {}", e);
            std::process::exit(1);
        }
    };
    let nand2_cell = match select_nand2_cell(&lib, cell_policy) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("aig-tech-map error: {}", e);
            std::process::exit(1);
        }
    };

    let mapped = match map_gatefn_to_structural_netlist(
        &gate_fn,
        &StructuralTechMapOptions {
            module_name,
            nand2_cell_name: nand2_cell.cell_name.clone(),
            nand2_input_pin_names: [
                nand2_cell.input_pin_names[0].clone(),
                nand2_cell.input_pin_names[1].clone(),
            ],
            nand2_output_pin_name: nand2_cell.output_pin_name.clone(),
            inv_cell_name: inv_cell.cell_name.clone(),
            inv_input_pin_name: inv_cell.input_pin_names[0].clone(),
            inv_output_pin_name: inv_cell.output_pin_name.clone(),
        },
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("aig-tech-map error: structural mapping failed: {:#}", e);
            std::process::exit(1);
        }
    };

    let netlist_text =
        match emit_module_as_netlist_text(&mapped.module, mapped.nets.as_slice(), &mapped.interner)
        {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "aig-tech-map error: failed to emit mapped netlist text: {:#}",
                    e
                );
                std::process::exit(1);
            }
        };

    if netlist_out == "-" {
        print!("{}", netlist_text);
    } else if let Err(e) = std::fs::write(netlist_out, netlist_text) {
        eprintln!(
            "aig-tech-map error: failed to write netlist output '{}': {}",
            netlist_out, e
        );
        std::process::exit(1);
    }

    eprintln!(
        "aig-tech-map: mapped '{}' to {} instances / {} nets using policy='{}' INV='{}' NAND2='{}'",
        aig_input_file,
        mapped.module.instances.len(),
        mapped.nets.len(),
        cell_policy.as_str(),
        inv_cell.cell_name,
        nand2_cell.cell_name
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_g8r::liberty_proto::{Pin, TimingTable};

    fn make_pin(name: &str, direction: PinDirection, function: &str) -> Pin {
        Pin {
            name: name.to_string(),
            direction: direction as i32,
            function: function.to_string(),
            ..Default::default()
        }
    }

    fn make_timed_output_pin(name: &str, function: &str, related_pins: &[&str]) -> Pin {
        Pin {
            name: name.to_string(),
            direction: PinDirection::Output as i32,
            function: function.to_string(),
            timing_arcs: related_pins
                .iter()
                .map(|related_pin| xlsynth_g8r::liberty_proto::TimingArc {
                    related_pin: (*related_pin).to_string(),
                    timing_type: "combinational".to_string(),
                    tables: vec![
                        scalar_table("cell_rise", 1.0),
                        scalar_table("cell_fall", 1.0),
                        scalar_table("rise_transition", 0.1),
                        scalar_table("fall_transition", 0.1),
                    ],
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn scalar_table(kind: &str, value: f64) -> TimingTable {
        TimingTable {
            kind: kind.to_string(),
            values: vec![value],
            dimensions: vec![],
            ..Default::default()
        }
    }

    fn make_inv_cell(name: &str, threshold_voltage_group_id: u32) -> Cell {
        make_inv_cell_with_pins(name, threshold_voltage_group_id, "A", "Y")
    }

    fn make_inv_cell_with_pins(
        name: &str,
        threshold_voltage_group_id: u32,
        input_pin_name: &str,
        output_pin_name: &str,
    ) -> Cell {
        Cell {
            name: name.to_string(),
            threshold_voltage_group_id,
            pins: vec![
                make_pin(input_pin_name, PinDirection::Input, ""),
                make_timed_output_pin(
                    output_pin_name,
                    format!("!{}", input_pin_name).as_str(),
                    &[input_pin_name],
                ),
            ],
            ..Default::default()
        }
    }

    fn make_nand2_cell(name: &str, threshold_voltage_group_id: u32) -> Cell {
        make_nand2_cell_with_pins(name, threshold_voltage_group_id, ["A", "B"], "Y")
    }

    fn make_nand2_cell_with_pins(
        name: &str,
        threshold_voltage_group_id: u32,
        input_pin_names: [&str; 2],
        output_pin_name: &str,
    ) -> Cell {
        Cell {
            name: name.to_string(),
            threshold_voltage_group_id,
            pins: vec![
                make_pin(input_pin_names[0], PinDirection::Input, ""),
                make_pin(input_pin_names[1], PinDirection::Input, ""),
                make_timed_output_pin(
                    output_pin_name,
                    format!("!({}*{})", input_pin_names[0], input_pin_names[1]).as_str(),
                    &input_pin_names,
                ),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn select_prefers_exact_base_names_in_small_normal_vt_mode() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![
                make_inv_cell("INV", 1),
                make_inv_cell("INVx1_nominal", 1),
                make_nand2_cell("NAND2", 1),
                make_nand2_cell("NAND2x1_nominal", 1),
            ],
            threshold_voltage_groups: vec!["nominal".to_string()],
            threshold_voltage_group_class_indices: vec![0],
            ..Default::default()
        };

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "INV"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2"
        );
    }

    #[test]
    fn select_prefers_x1_nominal_when_no_exact_base_in_small_normal_vt_mode() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![
                make_inv_cell("INVx2_fast", 2),
                make_inv_cell("INVx1_nominal", 1),
                make_inv_cell("INVx1_faster", 3),
                make_nand2_cell("NAND2x2_fast", 2),
                make_nand2_cell("NAND2x1_nominal", 1),
            ],
            threshold_voltage_groups: vec![
                "nominal".to_string(),
                "fast".to_string(),
                "faster".to_string(),
            ],
            threshold_voltage_group_class_indices: vec![0, 1, 2],
            ..Default::default()
        };

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "INVx1_nominal"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2x1_nominal"
        );
    }

    #[test]
    fn select_prefers_max_drive_in_fastest_vt_in_max_speed_mode() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![
                make_inv_cell("INVx1_nominal", 1),
                make_inv_cell("INVx2_faster", 3),
                make_inv_cell("INVx4_faster", 3),
                make_nand2_cell("NAND2x1_nominal", 1),
                make_nand2_cell("NAND2x2_fast", 2),
                make_nand2_cell("NAND2x4_faster", 3),
            ],
            threshold_voltage_groups: vec![
                "nominal".to_string(),
                "fast".to_string(),
                "faster".to_string(),
            ],
            threshold_voltage_group_class_indices: vec![0, 1, 2],
            ..Default::default()
        };

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "INVx4_faster"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "NAND2x4_faster"
        );
    }

    #[test]
    fn select_uses_drive_strength_with_fractional_suffix_in_max_speed_mode() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![
                make_inv_cell("INVxp75_faster", 1),
                make_inv_cell("INVx1_faster", 1),
                make_nand2_cell("NAND2xp33_faster", 1),
                make_nand2_cell("NAND2x1_faster", 1),
            ],
            threshold_voltage_groups: vec!["faster".to_string()],
            threshold_voltage_group_class_indices: vec![2],
            ..Default::default()
        };

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "INVx1_faster"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "NAND2x1_faster"
        );
    }

    #[test]
    fn select_uses_default_threshold_voltage_group() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![
                make_inv_cell("INVx1_default", 0),
                make_inv_cell("INVx2_fast", 2),
                make_nand2_cell("NAND2x1_default", 0),
            ],
            threshold_voltage_groups: vec!["nominal".to_string(), "fast".to_string()],
            default_threshold_voltage_group_id: 1,
            threshold_voltage_group_class_indices: vec![0, 1],
            ..Default::default()
        };

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "INVx1_default"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2x1_default"
        );
    }

    #[test]
    fn select_discovers_nonstandard_pin_names() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![
                make_inv_cell_with_pins("INV", 0, "I", "ZN"),
                make_nand2_cell_with_pins("NAND2", 0, ["I0", "I1"], "ZN"),
            ],
            ..Default::default()
        };

        assert_eq!(
            select_inv_cell(&lib, CellPolicy::SmallNormalVt).unwrap(),
            SelectedCellBinding {
                cell_name: "INV".to_string(),
                input_pin_names: vec!["I".to_string()],
                output_pin_name: "ZN".to_string(),
            }
        );
        assert_eq!(
            select_nand2_cell(&lib, CellPolicy::SmallNormalVt).unwrap(),
            SelectedCellBinding {
                cell_name: "NAND2".to_string(),
                input_pin_names: vec!["I0".to_string(), "I1".to_string()],
                output_pin_name: "ZN".to_string(),
            }
        );
    }

    #[test]
    fn select_rejects_family_name_with_wrong_formula() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![
                Cell {
                    name: "INV".to_string(),
                    pins: vec![
                        make_pin("A", PinDirection::Input, ""),
                        make_timed_output_pin("Y", "A", &["A"]),
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "NAND2".to_string(),
                    pins: vec![
                        make_pin("A", PinDirection::Input, ""),
                        make_pin("B", PinDirection::Input, ""),
                        make_timed_output_pin("Y", "A*B", &["A", "B"]),
                    ],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        assert!(select_inv_cell(&lib, CellPolicy::SmallNormalVt).is_err());
        assert!(select_nand2_cell(&lib, CellPolicy::SmallNormalVt).is_err());
    }

    #[test]
    fn select_accepts_demorgan_nand2_formula() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    make_pin("A", PinDirection::Input, ""),
                    make_pin("B", PinDirection::Input, ""),
                    make_timed_output_pin("Y", "(!A)+(!B)", &["A", "B"]),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2"
        );
    }

    #[test]
    fn select_is_deterministic_without_vt_metadata() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![
                make_inv_cell("INVx2", 0),
                make_inv_cell("INVx1", 0),
                make_nand2_cell("NAND2x2", 0),
                make_nand2_cell("NAND2x1", 0),
            ],
            ..Default::default()
        };

        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "INVx1"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::SmallNormalVt).unwrap(),
            "NAND2x1"
        );
        assert_eq!(
            select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "INVx2"
        );
        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "NAND2x2"
        );
    }

    #[test]
    fn select_rejects_candidate_missing_timing_for_one_functional_input() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    make_pin("A", PinDirection::Input, ""),
                    make_pin("B", PinDirection::Input, ""),
                    make_timed_output_pin("Y", "!(A*B)", &["A"]),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        assert!(select_nand2_cell(&lib, CellPolicy::SmallNormalVt).is_err());
    }

    #[test]
    fn select_skips_sta_incompatible_candidate_and_uses_valid_alternative() {
        let mut invalid_fast = make_nand2_cell("NAND2x2_fast", 2);
        invalid_fast
            .pins
            .iter_mut()
            .find(|pin| pin.name == "Y")
            .expect("output pin")
            .timing_arcs[0]
            .timing_type = "rising_edge".to_string();
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![invalid_fast, make_nand2_cell("NAND2x1_nominal", 1)],
            threshold_voltage_groups: vec!["nominal".to_string(), "fast".to_string()],
            threshold_voltage_group_class_indices: vec![0, 1],
            ..Default::default()
        };

        assert_eq!(
            select_nand2_cell_name(&lib, CellPolicy::MaxSpeed).unwrap(),
            "NAND2x1_nominal"
        );
    }

    #[test]
    fn select_rejects_mixed_ordered_and_unordered_vt_metadata() {
        let lib = xlsynth_g8r::liberty_proto::Library {
            cells: vec![
                make_inv_cell("INVx1_nominal", 1),
                make_inv_cell("INVx2_unclassified", 0),
            ],
            threshold_voltage_groups: vec!["nominal".to_string()],
            threshold_voltage_group_class_indices: vec![0],
            ..Default::default()
        };

        let err = select_inv_cell_name(&lib, CellPolicy::MaxSpeed).unwrap_err();
        assert!(err.contains("mix ordered and unordered threshold-voltage metadata"));
    }
}
