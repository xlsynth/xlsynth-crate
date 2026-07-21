// SPDX-License-Identifier: Apache-2.0

//! Clean-sheet, final-only technology mapping from choice AIGs to Liberty
//! cells.
//!
//! This module intentionally does not build on the older structural NAND/INV
//! lowering under netlist::techmap. It consumes the final choice-rich AIG once,
//! matches bounded Boolean cuts against arbitrary combinational Liberty
//! functions, runs NF-style delay/area-flow/exact-area cover rounds, and emits
//! a final parsed gate-level netlist. There is no mapping serialization or
//! ABC-loop feedback protocol in this API.

mod cover;
mod cuts;
mod emit;
mod liberty_index;
mod truth;

use crate::aig::{ChoiceAig, GateFn};
use crate::liberty_model::Library;
use crate::netlist::parse::{Net, NetlistModule};
use crate::netlist::report::build_sta_report;
use crate::netlist::sta::StaOptions;
use anyhow::Result;
use std::collections::BTreeMap;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Controller-owned endpoint timing information for one final mapping pass.
///
/// Names use the mapper's flattened scalar port spelling: a one-bit port keeps
/// its source name, while bit i of a wider port is named name_i.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TechMapTimingConstraints {
    pub primary_input_arrivals: BTreeMap<String, f64>,
    pub primary_output_required: BTreeMap<String, f64>,
}

/// Search bounds and final-netlist naming options.
#[derive(Clone, Debug, PartialEq)]
pub struct TechMapOptions {
    pub module_name: Option<String>,
    /// Maximum number of leaves in one truth-table cut; supported range is
    /// 1..=6.
    pub max_cut_size: usize,
    /// Maximum structural cuts retained per AIG node, including its trivial
    /// cut.
    pub max_cuts_per_node: usize,
    /// Maximum non-dominated Liberty variants retained for one cut/state
    /// signature before NF-style mapping rounds begin.
    pub max_frontier_size: usize,
    /// Transition seeded at each primary input for final STA and constrained
    /// mapping, in Liberty time units.
    pub primary_input_transition: f64,
    /// Extra capacitive load applied to each module output for final STA and
    /// constrained mapping.
    pub module_output_load: f64,
}

impl Default for TechMapOptions {
    fn default() -> Self {
        Self {
            module_name: None,
            max_cut_size: 6,
            max_cuts_per_node: 16,
            max_frontier_size: 16,
            primary_input_transition: 0.01,
            module_output_load: 0.0,
        }
    }
}

/// Deterministic diagnostics from one final technology-mapping run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TechMapStats {
    pub choice_class_count: usize,
    pub choice_link_count: usize,
    pub enumerated_cut_count: usize,
    pub indexed_cell_outputs: usize,
    pub indexed_cell_bindings: usize,
    pub skipped_liberty_cells: usize,
    pub matched_candidate_count: usize,
    pub selected_instance_count: usize,
    pub selected_area: f64,
    pub worst_estimated_output_arrival: f64,
}

/// Parsed final netlist plus mapping statistics.
#[derive(Debug)]
pub struct MappedNetlist {
    pub module: NetlistModule,
    pub nets: Vec<Net>,
    pub interner: StringInterner<StringBackend<SymbolU32>>,
    pub stats: TechMapStats,
}

/// Maps a final choice-rich AIG into a deterministic combinational cell
/// netlist.
pub fn map_choice_aig_to_netlist(
    choice_aig: &ChoiceAig,
    library: &Library,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    let analysis = cuts::analyze_choices(choice_aig)?;
    let cell_index = liberty_index::LibertyCellIndex::build_nf(library, options.max_cut_size)?;
    let cuts_by_node = cuts::enumerate_choice_cuts(
        choice_aig,
        &analysis,
        &cell_index,
        options.max_cut_size,
        options.max_cuts_per_node,
    )?;
    let plan = cover::build_cover_plan(
        choice_aig,
        &analysis,
        cuts_by_node.as_slice(),
        &cell_index,
        library,
        options,
        constraints,
    )?;
    let emitted = emit::emit_cover(choice_aig, &plan, &cell_index, options)?;
    let worst_estimated_output_arrival = if emitted.timing_complete {
        build_sta_report(
            &emitted.module,
            emitted.nets.as_slice(),
            &emitted.interner,
            library,
            StaOptions {
                primary_input_transition: options.primary_input_transition,
                module_output_load: options.module_output_load,
            },
        )?
        .delay
    } else {
        plan.output_arrivals
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(0.0)
    };
    let stats = TechMapStats {
        choice_class_count: analysis.classes.len(),
        choice_link_count: choice_aig.sibling_link_count(),
        enumerated_cut_count: cuts::cut_count(cuts_by_node.as_slice()),
        indexed_cell_outputs: cell_index.stats.indexed_cell_outputs,
        indexed_cell_bindings: cell_index.stats.indexed_bindings,
        skipped_liberty_cells: cell_index.stats.skipped_cells,
        matched_candidate_count: plan.matched_candidate_count,
        selected_instance_count: emitted.module.instances.len(),
        selected_area: emitted.area,
        worst_estimated_output_arrival,
    };
    Ok(MappedNetlist {
        module: emitted.module,
        nets: emitted.nets,
        interner: emitted.interner,
        stats,
    })
}

/// Maps an ordinary AIG by wrapping it as a no-choice final AIG.
pub fn map_gatefn_to_netlist(
    graph: GateFn,
    library: &Library,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    let choice_aig = ChoiceAig::without_choices(graph);
    map_choice_aig_to_netlist(&choice_aig, library, constraints, options)
}

pub(super) fn scalar_bit_name(base: &str, bit_index: usize, bit_count: usize) -> String {
    if bit_count == 1 {
        base.to_string()
    } else {
        format!("{}_{}", base, bit_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{AigOperand, ChoiceAig, GateBuilder, GateBuilderOptions};
    use crate::aig_sim::gate_sim::{Collect, eval};
    use crate::liberty_model::{Cell, LibraryBuilder, Pin, PinDirection, TimingTable};
    use crate::liberty_proto::TimingTableKind;
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
    use std::collections::HashSet;
    use xlsynth::IrBits;

    fn pin(
        builder: &mut LibraryBuilder,
        direction: PinDirection,
        name: &str,
        function: &str,
    ) -> Pin {
        Pin {
            direction: direction as i32,
            name: builder.intern_string(name).unwrap(),
            function: builder.intern_string(function).unwrap(),
            ..Default::default()
        }
    }

    fn make_library() -> Library {
        let mut builder = LibraryBuilder::new();
        builder.cells = vec![
            Cell {
                name: "mystery_and".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "A * B"),
                ],
                area: 1.0,
                ..Default::default()
            },
            Cell {
                name: "mystery_inv".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "!A"),
                ],
                area: 0.5,
                ..Default::default()
            },
            Cell {
                name: "mystery_buf".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "A"),
                ],
                area: 0.5,
                ..Default::default()
            },
        ];
        builder.finish()
    }

    fn make_and_graph() -> GateFn {
        let mut builder = GateBuilder::new("and_graph".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let and = builder.add_and_binary(a, b);
        builder.add_output("o".to_string(), and.into());
        builder.build()
    }

    fn make_nand_library() -> Library {
        let mut builder = LibraryBuilder::new();
        builder.cells = vec![
            Cell {
                name: "mystery_nand".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "!(A * B)"),
                ],
                area: 1.0,
                ..Default::default()
            },
            Cell {
                name: "mystery_inv".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "!A"),
                ],
                area: 0.5,
                ..Default::default()
            },
            Cell {
                name: "mystery_buf".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "A"),
                ],
                area: 0.5,
                ..Default::default()
            },
        ];
        builder.finish()
    }

    fn scalar_delay_table(builder: &mut LibraryBuilder, value: f64) -> TimingTable {
        builder
            .add_timing_table_f64(
                TimingTableKind::CellRise,
                0,
                vec![],
                vec![],
                vec![],
                vec![value],
                vec![],
                "",
            )
            .unwrap()
    }

    fn timed_output_pin(
        builder: &mut LibraryBuilder,
        name: &str,
        function: &str,
        related_pins: &[&str],
        delay: f64,
    ) -> Pin {
        let timing_arcs = related_pins
            .iter()
            .map(|related_pin| {
                let tables = vec![scalar_delay_table(builder, delay)];
                builder
                    .add_timing_arc(related_pin, "", "combinational", "", tables)
                    .unwrap()
            })
            .collect();
        Pin {
            direction: PinDirection::Output as i32,
            name: builder.intern_string(name).unwrap(),
            function: builder.intern_string(function).unwrap(),
            timing_arcs,
            ..Default::default()
        }
    }

    fn make_timed_and_library() -> Library {
        let mut builder = LibraryBuilder::new();
        builder.cells = vec![
            Cell {
                name: "slow_small".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    timed_output_pin(&mut builder, "Y", "A * B", &["A", "B"], 5.0),
                ],
                area: 1.0,
                ..Default::default()
            },
            Cell {
                name: "fast_large".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    timed_output_pin(&mut builder, "Y", "A * B", &["A", "B"], 1.0),
                ],
                area: 2.0,
                ..Default::default()
            },
        ];
        builder.finish()
    }

    fn make_three_input_and_graph() -> GateFn {
        let mut builder =
            GateBuilder::new("three_input_and".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let c: AigOperand = builder.add_input("c".to_string(), 1).try_into().unwrap();
        let ab = builder.add_and_binary(a, b);
        let abc = builder.add_and_binary(ab.into(), c);
        builder.add_output("o".to_string(), abc.into());
        builder.build()
    }

    fn make_unit_vs_liberty_timing_library() -> Library {
        let mut builder = LibraryBuilder::new();
        builder.cells = vec![
            Cell {
                name: "and2".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    timed_output_pin(&mut builder, "Y", "A * B", &["A", "B"], 1.0),
                ],
                area: 1.0,
                ..Default::default()
            },
            Cell {
                name: "and3_slow".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    pin(&mut builder, PinDirection::Input, "C", ""),
                    timed_output_pin(&mut builder, "Y", "A * B * C", &["A", "B", "C"], 100.0),
                ],
                area: 1.5,
                ..Default::default()
            },
        ];
        builder.finish()
    }

    #[test]
    fn maps_by_formula_instead_of_cell_family_name() {
        let graph = make_and_graph();
        let mapped = map_gatefn_to_netlist(
            graph,
            &make_library(),
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();

        assert_eq!(mapped.module.instances.len(), 1);
        let cell_name = mapped
            .interner
            .resolve(mapped.module.instances[0].type_name)
            .unwrap();
        assert_eq!(cell_name, "mystery_and");
        assert_eq!(mapped.stats.selected_area, 1.0);
    }

    #[test]
    fn mapping_output_is_deterministic() {
        let library = make_library();
        let first = map_gatefn_to_netlist(
            make_and_graph(),
            &library,
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();
        let second = map_gatefn_to_netlist(
            make_and_graph(),
            &library,
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();
        let first_text =
            emit_module_as_netlist_text(&first.module, first.nets.as_slice(), &first.interner)
                .unwrap();
        let second_text =
            emit_module_as_netlist_text(&second.module, second.nets.as_slice(), &second.interner)
                .unwrap();

        assert_eq!(first_text, second_text);
        assert_eq!(first.stats, second.stats);
    }

    #[test]
    fn emitted_cover_projects_back_to_equivalent_logic() {
        let graph = make_and_graph();
        let library = make_library();
        let mapped = map_gatefn_to_netlist(
            graph.clone(),
            &library,
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();
        let projected = project_gatefn_from_netlist_and_liberty(
            &mapped.module,
            mapped.nets.as_slice(),
            &mapped.interner,
            &library,
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap();

        for assignment in 0..4u64 {
            let inputs = [
                IrBits::make_ubits(1, assignment & 1).unwrap(),
                IrBits::make_ubits(1, (assignment >> 1) & 1).unwrap(),
            ];
            assert_eq!(
                eval(&graph, &inputs, Collect::None).outputs,
                eval(&projected, &inputs, Collect::None).outputs
            );
        }
    }

    #[test]
    fn closes_output_polarity_with_an_inverter() {
        let graph = make_and_graph();
        let mapped = map_gatefn_to_netlist(
            graph,
            &make_nand_library(),
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();

        assert_eq!(mapped.module.instances.len(), 2);
        let cell_names: Vec<&str> = mapped
            .module
            .instances
            .iter()
            .map(|instance| mapped.interner.resolve(instance.type_name).unwrap())
            .collect();
        assert!(cell_names.contains(&"mystery_nand"));
        assert!(cell_names.contains(&"mystery_inv"));
    }

    #[test]
    fn maps_complemented_cut_inputs_without_an_and_not_cell() {
        let mut builder =
            GateBuilder::new("and_not_graph".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let and_not = builder.add_and_binary(a, b.negate());
        builder.add_output("o".to_string(), and_not.into());
        let graph = builder.build();
        let library = make_nand_library();

        let mapped = map_gatefn_to_netlist(
            graph.clone(),
            &library,
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();
        let projected = project_gatefn_from_netlist_and_liberty(
            &mapped.module,
            mapped.nets.as_slice(),
            &mapped.interner,
            &library,
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap();

        for assignment in 0..4u64 {
            let inputs = [
                IrBits::make_ubits(1, assignment & 1).unwrap(),
                IrBits::make_ubits(1, (assignment >> 1) & 1).unwrap(),
            ];
            assert_eq!(
                eval(&graph, &inputs, Collect::None).outputs,
                eval(&projected, &inputs, Collect::None).outputs
            );
        }
    }

    #[test]
    fn maps_primary_input_output_through_a_buffer() {
        let mut builder = GateBuilder::new("identity".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        builder.add_output("o".to_string(), a.into());
        let graph = builder.build();

        let mapped = map_gatefn_to_netlist(
            graph,
            &make_library(),
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();

        assert_eq!(mapped.module.instances.len(), 1);
        assert_eq!(
            mapped
                .interner
                .resolve(mapped.module.instances[0].type_name)
                .unwrap(),
            "mystery_buf"
        );
    }

    #[test]
    fn choice_class_can_replace_an_unmappable_structural_cone() {
        let mut builder = GateBuilder::new(
            "choice_absorption".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let a_or_b = builder.add_or_binary(a, b);
        let absorbed = builder.add_and_binary(a, a_or_b);
        builder.add_output("o".to_string(), absorbed.into());
        let graph = builder.build();
        let mut siblings = vec![None; graph.gates.len()];
        siblings[absorbed.node.id] = Some(a.node);
        let choice_aig = ChoiceAig::new(graph.clone(), siblings).unwrap();
        let options = TechMapOptions {
            max_cut_size: 1,
            ..TechMapOptions::default()
        };

        assert!(
            map_gatefn_to_netlist(
                graph,
                &make_library(),
                &TechMapTimingConstraints::default(),
                &options,
            )
            .is_err()
        );
        let mapped = map_choice_aig_to_netlist(
            &choice_aig,
            &make_library(),
            &TechMapTimingConstraints::default(),
            &options,
        )
        .unwrap();

        assert_eq!(mapped.stats.choice_link_count, 1);
        assert_eq!(mapped.module.instances.len(), 1);
        assert_eq!(
            mapped
                .interner
                .resolve(mapped.module.instances[0].type_name)
                .unwrap(),
            "mystery_buf"
        );
    }

    #[test]
    fn nf_root_library_reports_unmet_required_time() {
        let graph = make_and_graph();
        let library = make_timed_and_library();
        let mut relaxed_constraints = TechMapTimingConstraints::default();
        relaxed_constraints
            .primary_output_required
            .insert("o".to_string(), 10.0);
        let relaxed = map_gatefn_to_netlist(
            graph.clone(),
            &library,
            &relaxed_constraints,
            &TechMapOptions::default(),
        )
        .unwrap();
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_output_required
            .insert("o".to_string(), 2.0);
        let constrained =
            map_gatefn_to_netlist(graph, &library, &constraints, &TechMapOptions::default());

        assert_eq!(
            relaxed
                .interner
                .resolve(relaxed.module.instances[0].type_name)
                .unwrap(),
            "slow_small"
        );
        assert!(
            constrained
                .unwrap_err()
                .to_string()
                .contains("no cover meets required time 2")
        );
    }

    #[test]
    fn unconstrained_search_uses_nf_unit_delay() {
        let graph = make_three_input_and_graph();
        let library = make_unit_vs_liberty_timing_library();
        let unconstrained = map_gatefn_to_netlist(
            graph.clone(),
            &library,
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_output_required
            .insert("o".to_string(), 5.0);
        let constrained =
            map_gatefn_to_netlist(graph, &library, &constraints, &TechMapOptions::default())
                .unwrap();

        assert_eq!(unconstrained.module.instances.len(), 1);
        assert_eq!(
            unconstrained
                .interner
                .resolve(unconstrained.module.instances[0].type_name)
                .unwrap(),
            "and3_slow"
        );
        assert_eq!(constrained.module.instances.len(), 2);
        assert!(
            constrained
                .module
                .instances
                .iter()
                .all(|instance| constrained.interner.resolve(instance.type_name) == Some("and2"))
        );
    }
}
