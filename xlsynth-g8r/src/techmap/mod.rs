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
mod nf;
mod truth;

/// Timing-oriented covers may safely use a stricter tree than the caller's cap.
const BALANCED_TIMING_MAX_FANOUT: usize = 8;

use crate::aig::{ChoiceAig, GateFn};
use crate::liberty_model::Library;
use crate::netlist::buffer::{BufferOptions, BufferStats};
use crate::netlist::optimize::{NetlistOptimizationOptions, optimize_mapped_netlist};
use crate::netlist::parse::{Net, NetlistModule};
use crate::netlist::report::build_sta_report;
use crate::netlist::resize::{ResizeOptions, ResizeStats};
use crate::netlist::sta::StaOptions;
use anyhow::{Result, anyhow};
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

/// Structural-cover objective used before exact final-netlist STA.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TechMapTimingModel {
    /// Explicitly request the previous exact-area/delay portfolio.
    Balanced,
    /// Reserved for internal NF characterization; public selection panics.
    NfUnit,
    /// Preserve NF cover selection while using representative Liberty arcs.
    #[default]
    NfLiberty,
    /// Use native Liberty delay and the intended bounded buffer-tree load.
    BufferedLiberty,
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
    /// Maximum non-dominated Liberty variants retained by the explicitly
    /// selected experimental timing models. Native NF keeps one delay and
    /// one area-flow match per object phase instead of a larger frontier.
    pub max_frontier_size: usize,
    /// Transition seeded at each primary input for final STA and constrained
    /// mapping, in Liberty time units.
    pub primary_input_transition: f64,
    /// Extra capacitive load applied to each module output for final STA and
    /// constrained mapping.
    pub module_output_load: f64,
    /// Lightweight timing objective for unconstrained structural mapping.
    pub timing_model: TechMapTimingModel,
    /// Optional buffer insertion after mapping; `None` disables the pass.
    pub buffer_options: Option<BufferOptions>,
    /// Incremental exact-Liberty cell sizing; `None` disables the pass.
    pub resize_options: Option<ResizeOptions>,
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
            timing_model: TechMapTimingModel::NfLiberty,
            buffer_options: None,
            resize_options: None,
        }
    }
}

/// Reusable Liberty-side state for mapping many AIGs against one library.
///
/// ABC normally reads a preprocessed SCL library before mapping. Keeping the
/// parsed Liberty model and its NF binding index together gives callers the
/// same reuse boundary without weakening the one-shot mapping API.
pub struct PreparedTechMapLibrary<'a> {
    library: &'a Library,
    cell_index: liberty_index::LibertyCellIndex,
    max_cut_size: usize,
}

impl<'a> PreparedTechMapLibrary<'a> {
    /// Builds reusable Liberty matching state for one maximum cut size.
    pub fn new(library: &'a Library, max_cut_size: usize) -> Result<Self> {
        let cell_index = liberty_index::LibertyCellIndex::build_nf(library, max_cut_size)?;
        Ok(Self {
            library,
            cell_index,
            max_cut_size,
        })
    }
}

/// Deterministic diagnostics from one final technology-mapping run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TechMapStats {
    /// Structural search objective used by the selected complete cover.
    pub selected_timing_model: TechMapTimingModel,
    /// Input slew used to characterize the fixed NF Liberty pin delays.
    pub representative_input_transition: Option<f64>,
    /// Two-sink median Liberty load used to characterize the NF pin delays.
    pub representative_output_load: Option<f64>,
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
    pub buffer_stats: Option<BufferStats>,
    pub resize_stats: Option<ResizeStats>,
}

/// Parsed final netlist plus mapping statistics.
#[derive(Debug)]
pub struct MappedNetlist {
    pub module: NetlistModule,
    pub nets: Vec<Net>,
    pub interner: StringInterner<StringBackend<SymbolU32>>,
    pub stats: TechMapStats,
}

/// Prevents callers from accidentally selecting the structural unit-delay mode.
fn assert_supported_timing_model(options: &TechMapOptions) {
    assert!(
        options.timing_model != TechMapTimingModel::NfUnit,
        "nf-unit technology mapping is disabled; use nf-liberty representative Liberty pin delays"
    );
}

/// Maps a final choice-rich AIG into a deterministic combinational cell
/// netlist.
pub fn map_choice_aig_to_netlist(
    choice_aig: &ChoiceAig,
    library: &Library,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    assert_supported_timing_model(options);
    let prepared = PreparedTechMapLibrary::new(library, options.max_cut_size)?;
    map_choice_aig_to_netlist_with_prepared(choice_aig, &prepared, constraints, options)
}

/// Maps one choice-rich AIG using Liberty state prepared for repeated runs.
pub fn map_choice_aig_to_netlist_with_prepared(
    choice_aig: &ChoiceAig,
    prepared: &PreparedTechMapLibrary<'_>,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    assert_supported_timing_model(options);
    if options.max_cut_size != prepared.max_cut_size {
        return Err(anyhow!(
            "prepared techmap library uses max_cut_size {}, but mapping options request {}",
            prepared.max_cut_size,
            options.max_cut_size
        ));
    }
    let cell_index = &prepared.cell_index;
    let analysis = cuts::analyze_choices(choice_aig)?;

    if options.timing_model == TechMapTimingModel::NfLiberty
        && constraints.primary_input_arrivals.is_empty()
        && constraints.primary_output_required.is_empty()
    {
        let cover = nf::build_cover_plan(
            choice_aig,
            &analysis,
            prepared.library,
            cell_index,
            options,
            constraints,
        )?;
        return finish_prepared_choice_cover(
            choice_aig,
            prepared,
            &analysis,
            cover.plan,
            cover.enumerated_cut_count,
            cover.representative_output_load,
            options,
        );
    }

    let cuts_by_node = cuts::enumerate_choice_cuts(
        choice_aig,
        &analysis,
        cell_index,
        options.max_cut_size,
        options.max_cuts_per_node,
    )?;
    if options.timing_model == TechMapTimingModel::Balanced
        && constraints.primary_input_arrivals.is_empty()
        && constraints.primary_output_required.is_empty()
        && cell_index.all_bindings_have_complete_timing()
    {
        let mut nf_options = options.clone();
        nf_options.timing_model = TechMapTimingModel::NfUnit;
        let nf = map_prepared_choice_cover(
            choice_aig,
            prepared,
            &analysis,
            cuts_by_node.as_slice(),
            constraints,
            &nf_options,
        );

        let mut liberty_options = options.clone();
        liberty_options.timing_model = TechMapTimingModel::BufferedLiberty;
        if let Some(buffer_options) = liberty_options.buffer_options.as_mut() {
            buffer_options.max_fanout = buffer_options.max_fanout.min(BALANCED_TIMING_MAX_FANOUT);
        }
        let liberty = map_prepared_choice_cover(
            choice_aig,
            prepared,
            &analysis,
            cuts_by_node.as_slice(),
            constraints,
            &liberty_options,
        );

        return match (nf, liberty) {
            (Ok(nf), Ok(liberty)) => {
                if mapped_area_delay_order(&nf, &liberty).is_le() {
                    Ok(nf)
                } else {
                    Ok(liberty)
                }
            }
            (Ok(mapped), Err(_)) | (Err(_), Ok(mapped)) => Ok(mapped),
            (Err(nf_error), Err(liberty_error)) => Err(anyhow!(
                "both balanced technology-mapping covers failed: NF cover: {nf_error:#}; Liberty cover: {liberty_error:#}"
            )),
        };
    }

    let mut effective_options = options.clone();
    if effective_options.timing_model == TechMapTimingModel::Balanced {
        effective_options.timing_model = TechMapTimingModel::NfUnit;
    }
    map_prepared_choice_cover(
        choice_aig,
        prepared,
        &analysis,
        cuts_by_node.as_slice(),
        constraints,
        &effective_options,
    )
}

/// Builds and exactly scores one cover from shared choice and Liberty state.
fn map_prepared_choice_cover(
    choice_aig: &ChoiceAig,
    prepared: &PreparedTechMapLibrary<'_>,
    analysis: &cuts::ChoiceAnalysis,
    cuts_by_node: &[Vec<cuts::Cut>],
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    let cell_index = &prepared.cell_index;
    // A timing-complete emitted netlist runs exact parsed-netlist STA below.
    // Only incomplete libraries need the approximate structural cover to
    // compute fallback Liberty arrivals.
    let retime_approximate_cover_for_fallback = !cell_index.all_bindings_have_complete_timing();
    let plan = cover::build_cover_plan(
        choice_aig,
        cuts_by_node,
        cell_index,
        prepared.library,
        options,
        constraints,
        retime_approximate_cover_for_fallback,
    )?;
    finish_prepared_choice_cover(
        choice_aig,
        prepared,
        analysis,
        plan,
        cuts::cut_count(cuts_by_node),
        None,
        options,
    )
}

/// Emits, optionally post-processes, and exactly scores one selected cover.
fn finish_prepared_choice_cover(
    choice_aig: &ChoiceAig,
    prepared: &PreparedTechMapLibrary<'_>,
    analysis: &cuts::ChoiceAnalysis,
    plan: cover::CoverPlan,
    enumerated_cut_count: usize,
    representative_output_load: Option<f64>,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    let library = prepared.library;
    let cell_index = &prepared.cell_index;
    let retime_approximate_cover_for_fallback = !cell_index.all_bindings_have_complete_timing();
    let mut emitted = emit::emit_cover(choice_aig, &plan, cell_index, options)?;
    let mut buffer_stats = None;
    let mut resize_stats = None;
    let mut optimized_area = None;
    let mut optimized_delay = None;
    if emitted.timing_complete
        && (options.buffer_options.is_some() || options.resize_options.is_some())
    {
        let optimization = optimize_mapped_netlist(
            &mut emitted.module,
            &mut emitted.nets,
            &mut emitted.interner,
            library,
            &NetlistOptimizationOptions {
                sta_options: StaOptions {
                    primary_input_transition: options.primary_input_transition,
                    module_output_load: options.module_output_load,
                },
                buffer_options: options.buffer_options.clone(),
                resize_options: options.resize_options.clone(),
            },
        )?;
        optimized_area = Some(optimization.final_area);
        optimized_delay = Some(optimization.final_delay);
        buffer_stats = optimization.buffer_stats;
        resize_stats = optimization.resize_stats;
    }
    let worst_estimated_output_arrival = if let Some(delay) = optimized_delay {
        delay
    } else if emitted.timing_complete {
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
        debug_assert!(
            retime_approximate_cover_for_fallback,
            "incomplete emitted timing should retain cover fallback arrivals"
        );
        plan.output_arrivals
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(0.0)
    };
    let stats = TechMapStats {
        selected_timing_model: options.timing_model,
        representative_input_transition: representative_output_load
            .map(|_| options.primary_input_transition),
        representative_output_load,
        choice_class_count: analysis.choice_class_count,
        choice_link_count: choice_aig.sibling_link_count(),
        enumerated_cut_count,
        indexed_cell_outputs: cell_index.stats.indexed_cell_outputs,
        indexed_cell_bindings: cell_index.stats.indexed_bindings,
        skipped_liberty_cells: cell_index.stats.skipped_cells,
        matched_candidate_count: plan.matched_candidate_count,
        selected_instance_count: emitted.module.instances.len(),
        selected_area: optimized_area.unwrap_or(emitted.area),
        worst_estimated_output_arrival,
        buffer_stats,
        resize_stats,
    };
    Ok(MappedNetlist {
        module: emitted.module,
        nets: emitted.nets,
        interner: emitted.interner,
        stats,
    })
}

/// Prefers the best exact area-delay-squared result with stable tie breaks.
fn mapped_area_delay_order(lhs: &MappedNetlist, rhs: &MappedNetlist) -> std::cmp::Ordering {
    let lhs_cost = lhs.stats.selected_area * lhs.stats.worst_estimated_output_arrival.powi(2);
    let rhs_cost = rhs.stats.selected_area * rhs.stats.worst_estimated_output_arrival.powi(2);
    lhs_cost
        .total_cmp(&rhs_cost)
        .then_with(|| {
            lhs.stats
                .worst_estimated_output_arrival
                .total_cmp(&rhs.stats.worst_estimated_output_arrival)
        })
        .then_with(|| lhs.stats.selected_area.total_cmp(&rhs.stats.selected_area))
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

/// Maps an ordinary AIG using Liberty state prepared for repeated runs.
pub fn map_gatefn_to_netlist_with_prepared(
    graph: GateFn,
    prepared: &PreparedTechMapLibrary<'_>,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    let choice_aig = ChoiceAig::without_choices(graph);
    map_choice_aig_to_netlist_with_prepared(&choice_aig, prepared, constraints, options)
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
    use crate::netlist::cell_catalog::test_utils::sizing_library;
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
    use crate::netlist::parse::NetRef;
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

    fn make_asymmetric_phase_library() -> Library {
        let mut builder = LibraryBuilder::new();
        builder.cells = vec![
            Cell {
                name: "AND2".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "A * B"),
                ],
                area: 2.0,
                ..Default::default()
            },
            Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Input, "B", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "!(A * B)"),
                ],
                area: 1.0,
                ..Default::default()
            },
            Cell {
                name: "INV".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "!A"),
                ],
                area: 0.1,
                ..Default::default()
            },
            Cell {
                name: "BUF".to_string(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", ""),
                    pin(&mut builder, PinDirection::Output, "Y", "A"),
                ],
                area: 0.1,
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

    /// Builds complete timing for covers that trade cell count against delay.
    fn make_complete_unit_vs_liberty_timing_library() -> Library {
        let mut builder = LibraryBuilder::new();
        let specs: [(&str, &[&str], &str, f64, f64); 2] = [
            ("AND2", &["A", "B"], "A * B", 1.0, 1.0),
            ("AND3", &["A", "B", "C"], "A * B * C", 1.5, 100.0),
        ];
        let mut cells = Vec::with_capacity(specs.len());
        for (name, inputs, function, area, delay) in specs {
            let mut pins: Vec<Pin> = inputs
                .iter()
                .map(|input| Pin {
                    direction: PinDirection::Input as i32,
                    name: builder.intern_string(input).unwrap(),
                    capacitance: Some(0.1),
                    ..Pin::default()
                })
                .collect();
            let timing_arcs = inputs
                .iter()
                .map(|input| {
                    let tables = [
                        (TimingTableKind::CellRise, delay),
                        (TimingTableKind::CellFall, delay),
                        (TimingTableKind::RiseTransition, 0.1),
                        (TimingTableKind::FallTransition, 0.1),
                    ]
                    .into_iter()
                    .map(|(kind, value)| {
                        builder
                            .add_timing_table_f64(
                                kind,
                                0,
                                vec![],
                                vec![],
                                vec![],
                                vec![value],
                                vec![],
                                "",
                            )
                            .unwrap()
                    })
                    .collect();
                    builder
                        .add_timing_arc(input, "positive_unate", "combinational", "", tables)
                        .unwrap()
                })
                .collect();
            pins.push(Pin {
                direction: PinDirection::Output as i32,
                name: builder.intern_string("Y").unwrap(),
                function: builder.intern_string(function).unwrap(),
                max_capacitance: Some(10.0),
                timing_arcs,
                ..Pin::default()
            });
            cells.push(Cell {
                name: name.to_string(),
                pins,
                area,
                ..Cell::default()
            });
        }
        builder.cells = cells;
        builder.finish()
    }

    #[test]
    fn default_mapping_uses_a_single_representative_liberty_nf_cover() {
        let options = TechMapOptions::default();

        assert_eq!(options.timing_model, TechMapTimingModel::NfLiberty);
        assert!(options.buffer_options.is_none());
        assert!(options.resize_options.is_none());
    }

    #[test]
    fn nf_mapping_preserves_full_precision_liberty_cell_area() {
        let mut library = make_library();
        library.cells[0].area = 1.234_567_89;

        let mapped = map_gatefn_to_netlist(
            make_and_graph(),
            &library,
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("the exact-area NF root should map the two-input AND");

        assert_eq!(
            mapped.stats.selected_timing_model,
            TechMapTimingModel::NfLiberty
        );
        assert_eq!(mapped.stats.selected_area, 1.234_567_89);
        assert!(mapped.stats.buffer_stats.is_none());
        assert!(mapped.stats.resize_stats.is_none());
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
    fn prepared_library_matches_one_shot_mapping() {
        let library = make_library();
        let options = TechMapOptions::default();
        let prepared = PreparedTechMapLibrary::new(&library, options.max_cut_size).unwrap();
        let one_shot = map_gatefn_to_netlist(
            make_and_graph(),
            &library,
            &TechMapTimingConstraints::default(),
            &options,
        )
        .unwrap();
        let reused = map_gatefn_to_netlist_with_prepared(
            make_and_graph(),
            &prepared,
            &TechMapTimingConstraints::default(),
            &options,
        )
        .unwrap();
        let one_shot_text = emit_module_as_netlist_text(
            &one_shot.module,
            one_shot.nets.as_slice(),
            &one_shot.interner,
        )
        .unwrap();
        let reused_text =
            emit_module_as_netlist_text(&reused.module, reused.nets.as_slice(), &reused.interner)
                .unwrap();

        assert_eq!(one_shot_text, reused_text);
        assert_eq!(one_shot.stats, reused.stats);
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
    fn maps_constant_outputs_as_zero_area_assignments() {
        let mut builder =
            GateBuilder::new("constant_outputs".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let and = builder.add_and_binary(a, b);
        builder.add_output("logic".to_string(), and.into());
        let constants = builder.add_literal(&IrBits::make_ubits(4, 0b0101).unwrap());
        for bit_index in 0..constants.get_bit_count() {
            builder.add_output(
                format!("constants_{bit_index}"),
                (*constants.get_lsb(bit_index)).into(),
            );
        }
        let graph = builder.build();
        let library = sizing_library();

        let mapped = map_gatefn_to_netlist(
            graph.clone(),
            &library,
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .expect("constant outputs should map without Liberty tie cells");

        assert_eq!(mapped.module.assigns.len(), 4);
        assert_eq!(mapped.module.instances.len(), 1);
        assert_eq!(mapped.stats.selected_instance_count, 1);
        let emitted =
            emit_module_as_netlist_text(&mapped.module, mapped.nets.as_slice(), &mapped.interner)
                .expect("constant-output mapping should emit");
        assert_eq!(
            emitted,
            "module constant_outputs(a, b, logic, constants_0, constants_1, constants_2, constants_3);\n  input a;\n  input b;\n  output logic;\n  output constants_0;\n  output constants_1;\n  output constants_2;\n  output constants_3;\n  assign constants_0 = 1'b1;\n  assign constants_1 = 1'b0;\n  assign constants_2 = 1'b1;\n  assign constants_3 = 1'b0;\n  AND2 u_tm_0 (.A(a), .B(b), .Y(logic));\nendmodule\n"
        );
        let projected = project_gatefn_from_netlist_and_liberty(
            &mapped.module,
            mapped.nets.as_slice(),
            &mapped.interner,
            &library,
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect("constant-output mapping should project back to an AIG");
        for assignment in 0..4_u64 {
            let inputs = [
                IrBits::make_ubits(1, assignment & 1).unwrap(),
                IrBits::make_ubits(1, (assignment >> 1) & 1).unwrap(),
            ];
            assert_eq!(
                eval(&graph, &inputs, Collect::None).outputs,
                eval(&projected, &inputs, Collect::None).outputs,
            );
        }
    }

    #[test]
    fn sibling_cut_can_replace_an_unmappable_structural_cone() {
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
    fn sibling_choices_keep_distinct_output_mapping_states() {
        let mut builder =
            GateBuilder::new("choice_roots".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let first = builder.add_and_binary(a, b);
        let second = builder.add_and_binary(a, b);
        assert_ne!(first.node, second.node);
        builder.add_output("o0".to_string(), first.into());
        builder.add_output("o1".to_string(), second.into());
        let graph = builder.build();
        let mut siblings = vec![None; graph.gates.len()];
        siblings[second.node.id] = Some(first.node);
        let choice_aig = ChoiceAig::new(graph, siblings).unwrap();

        let mapped = map_choice_aig_to_netlist(
            &choice_aig,
            &make_library(),
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();
        let and_instance_count = mapped
            .module
            .instances
            .iter()
            .filter(|instance| mapped.interner.resolve(instance.type_name) == Some("mystery_and"))
            .count();

        // NF stores matches per GIA object. A sibling contributes cuts to its
        // later representative, but it does not merge two referenced roots
        // into one selected mapping state.
        assert_eq!(and_instance_count, 2);
    }

    #[test]
    fn output_phase_cleanup_keeps_inverter_off_high_fanout_phase() {
        let mut builder = GateBuilder::new(
            "phase_orientation".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let c: AigOperand = builder.add_input("c".to_string(), 1).try_into().unwrap();
        let d: AigOperand = builder.add_input("d".to_string(), 1).try_into().unwrap();
        let e: AigOperand = builder.add_input("e".to_string(), 1).try_into().unwrap();
        let root = builder.add_and_binary(a, b);
        let use0 = builder.add_and_binary(root.into(), c);
        let use1 = builder.add_and_binary(root.into(), d);
        let use2 = builder.add_and_binary(root.into(), e);
        builder.add_output("neg".to_string(), root.negate().into());
        builder.add_output("use0".to_string(), use0.into());
        builder.add_output("use1".to_string(), use1.into());
        builder.add_output("use2".to_string(), use2.into());
        let graph = builder.build();

        let mapped = map_gatefn_to_netlist(
            graph,
            &make_asymmetric_phase_library(),
            &TechMapTimingConstraints::default(),
            &TechMapOptions::default(),
        )
        .unwrap();
        let output_driver = mapped
            .module
            .instances
            .iter()
            .find(|instance| {
                instance.connections.iter().any(|(pin, net_ref)| {
                    mapped.interner.resolve(*pin) == Some("Y")
                        && matches!(
                            net_ref,
                            NetRef::Simple(net)
                                if mapped.interner.resolve(mapped.nets[net.0].name) == Some("neg")
                        )
                })
            })
            .expect("negative output should have a cell driver");
        let inverter_input = output_driver
            .connections
            .iter()
            .find_map(|(pin, net_ref)| {
                if mapped.interner.resolve(*pin) != Some("A") {
                    return None;
                }
                match net_ref {
                    NetRef::Simple(net) => Some(*net),
                    _ => None,
                }
            })
            .expect("output inverter should have a simple input net");
        let internal_driver = mapped
            .module
            .instances
            .iter()
            .find(|instance| {
                instance.connections.iter().any(|(pin, net_ref)| {
                    mapped.interner.resolve(*pin) == Some("Y")
                        && matches!(net_ref, NetRef::Simple(net) if *net == inverter_input)
                })
            })
            .expect("the output inverter input should have a cell driver");

        // Area flow prefers NAND2 plus an inverter for the multiply-referenced
        // positive phase. The NF-like PO cleanup flips that orientation so
        // the output-only negative phase owns the inverter instead.
        assert_eq!(
            mapped.interner.resolve(output_driver.type_name),
            Some("INV")
        );
        assert_eq!(
            mapped.interner.resolve(internal_driver.type_name),
            Some("AND2")
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
    #[should_panic(
        expected = "nf-unit technology mapping is disabled; use nf-liberty representative Liberty pin delays"
    )]
    fn explicit_nf_unit_timing_panics() {
        let _ = map_gatefn_to_netlist(
            make_three_input_and_graph(),
            &make_unit_vs_liberty_timing_library(),
            &TechMapTimingConstraints::default(),
            &TechMapOptions {
                timing_model: TechMapTimingModel::NfUnit,
                ..TechMapOptions::default()
            },
        );
    }

    #[test]
    #[should_panic(
        expected = "nf-unit technology mapping is disabled; use nf-liberty representative Liberty pin delays"
    )]
    fn explicit_nf_unit_timing_panics_with_a_prepared_library() {
        let graph = ChoiceAig::without_choices(make_three_input_and_graph());
        let library = make_unit_vs_liberty_timing_library();
        let options = TechMapOptions {
            timing_model: TechMapTimingModel::NfUnit,
            ..TechMapOptions::default()
        };
        let prepared = PreparedTechMapLibrary::new(&library, options.max_cut_size)
            .expect("the characterization library should prepare");

        let _ = map_choice_aig_to_netlist_with_prepared(
            &graph,
            &prepared,
            &TechMapTimingConstraints::default(),
            &options,
        );
    }

    #[test]
    fn representative_nf_search_avoids_a_slow_shallow_cover() {
        let mapped = map_gatefn_to_netlist(
            make_three_input_and_graph(),
            &make_complete_unit_vs_liberty_timing_library(),
            &TechMapTimingConstraints::default(),
            &TechMapOptions {
                timing_model: TechMapTimingModel::NfLiberty,
                ..TechMapOptions::default()
            },
        )
        .expect("representative Liberty timing should preserve a valid NF cover");

        assert_eq!(
            mapped.stats.selected_timing_model,
            TechMapTimingModel::NfLiberty
        );
        assert_eq!(mapped.stats.representative_input_transition, Some(0.01));
        assert!(
            (mapped
                .stats
                .representative_output_load
                .expect("representative timing should record its load")
                - 0.2)
                .abs()
                < 1e-12
        );
        assert_eq!(mapped.module.instances.len(), 2);
        assert!(
            mapped
                .module
                .instances
                .iter()
                .all(|instance| { mapped.interner.resolve(instance.type_name) == Some("AND2") })
        );
        assert_eq!(mapped.stats.worst_estimated_output_arrival, 2.0);
    }

    #[test]
    fn representative_nf_search_falls_back_for_incomplete_timing() {
        let mapped = map_gatefn_to_netlist(
            make_three_input_and_graph(),
            &make_unit_vs_liberty_timing_library(),
            &TechMapTimingConstraints::default(),
            &TechMapOptions {
                timing_model: TechMapTimingModel::NfLiberty,
                ..TechMapOptions::default()
            },
        )
        .expect("incomplete timing tables should retain scalar NF pin delays");

        assert_eq!(
            mapped.stats.selected_timing_model,
            TechMapTimingModel::NfLiberty
        );
        assert_eq!(mapped.module.instances.len(), 2);
        assert!(
            mapped
                .module
                .instances
                .iter()
                .all(|instance| { mapped.interner.resolve(instance.type_name) == Some("and2") })
        );
    }

    #[test]
    fn buffered_liberty_search_avoids_a_slow_shallow_cover() {
        let mapped = map_gatefn_to_netlist(
            make_three_input_and_graph(),
            &make_unit_vs_liberty_timing_library(),
            &TechMapTimingConstraints::default(),
            &TechMapOptions {
                timing_model: TechMapTimingModel::BufferedLiberty,
                ..TechMapOptions::default()
            },
        )
        .unwrap();

        assert_eq!(mapped.module.instances.len(), 2);
        assert!(
            mapped
                .module
                .instances
                .iter()
                .all(|instance| { mapped.interner.resolve(instance.type_name) == Some("and2") })
        );
    }

    #[test]
    fn balanced_search_selects_the_best_exact_area_delay_cover() {
        let graph = make_three_input_and_graph();
        let library = make_complete_unit_vs_liberty_timing_library();
        let constraints = TechMapTimingConstraints::default();
        let nf_options = TechMapOptions {
            timing_model: TechMapTimingModel::NfUnit,
            ..TechMapOptions::default()
        };
        let choice_aig = ChoiceAig::without_choices(graph.clone());
        let prepared = PreparedTechMapLibrary::new(&library, nf_options.max_cut_size)
            .expect("the complete Liberty library should prepare");
        let analysis = cuts::analyze_choices(&choice_aig)
            .expect("the characterization graph should be topologically valid");
        let cuts_by_node = cuts::enumerate_choice_cuts(
            &choice_aig,
            &analysis,
            &prepared.cell_index,
            nf_options.max_cut_size,
            nf_options.max_cuts_per_node,
        )
        .expect("the characterization cuts should enumerate");
        let nf = map_prepared_choice_cover(
            &choice_aig,
            &prepared,
            &analysis,
            &cuts_by_node,
            &constraints,
            &nf_options,
        )
        .expect("the internal compact NF characterization should map");
        let liberty = map_gatefn_to_netlist(
            graph.clone(),
            &library,
            &constraints,
            &TechMapOptions {
                timing_model: TechMapTimingModel::BufferedLiberty,
                ..TechMapOptions::default()
            },
        )
        .expect("the timing-oriented Liberty cover should map");
        let balanced = map_gatefn_to_netlist(
            graph,
            &library,
            &constraints,
            &TechMapOptions {
                timing_model: TechMapTimingModel::Balanced,
                ..TechMapOptions::default()
            },
        )
        .expect("explicit balanced mapping should select a timing-complete cover");

        assert_eq!(nf.module.instances.len(), 1);
        assert_eq!(liberty.module.instances.len(), 2);
        assert_eq!(balanced.stats, liberty.stats);
        assert_eq!(
            balanced.stats.selected_timing_model,
            TechMapTimingModel::BufferedLiberty
        );
        assert!(
            balanced.stats.selected_area * balanced.stats.worst_estimated_output_arrival.powi(2)
                < nf.stats.selected_area * nf.stats.worst_estimated_output_arrival.powi(2)
        );
    }

    #[test]
    fn complete_mapping_runs_buffering_and_exact_timing_sizing() {
        let mut builder = GateBuilder::new(
            "buffered_sized_mapping".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let root = builder.add_and_binary(a, b);
        for index in 0..9 {
            let input: AigOperand = builder
                .add_input(format!("c{index}"), 1)
                .try_into()
                .unwrap();
            let output = builder.add_and_binary(root.into(), input);
            builder.add_output(format!("o{index}"), output.into());
        }
        let graph = builder.build();
        let library = sizing_library();
        let mapped = map_gatefn_to_netlist(
            graph.clone(),
            &library,
            &TechMapTimingConstraints::default(),
            &TechMapOptions {
                buffer_options: Some(BufferOptions {
                    max_fanout: 8,
                    ..BufferOptions::default()
                }),
                resize_options: Some(ResizeOptions::default()),
                ..TechMapOptions::default()
            },
        )
        .unwrap();

        let buffer_stats = mapped
            .stats
            .buffer_stats
            .as_ref()
            .expect("timing-complete mapping should run buffer insertion");
        let resize_stats = mapped
            .stats
            .resize_stats
            .as_ref()
            .expect("timing-complete mapping should run incremental sizing");
        assert!(buffer_stats.buffers_inserted > 0);
        assert_eq!(buffer_stats.unresolved_overloaded_nets, 0);
        assert!(resize_stats.final_delay <= resize_stats.initial_delay);

        let report = build_sta_report(
            &mapped.module,
            mapped.nets.as_slice(),
            &mapped.interner,
            &library,
            StaOptions::default(),
        )
        .unwrap();
        assert_eq!(mapped.stats.worst_estimated_output_arrival, report.delay);
        assert_eq!(mapped.stats.selected_area, resize_stats.final_area);

        let projected = project_gatefn_from_netlist_and_liberty(
            &mapped.module,
            mapped.nets.as_slice(),
            &mapped.interner,
            &library,
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap();
        for assignment in 0..32_u64 {
            let inputs: Vec<IrBits> = (0..11)
                .map(|bit| IrBits::make_ubits(1, (assignment >> bit) & 1).unwrap())
                .collect();
            assert_eq!(
                eval(&graph, &inputs, Collect::None).outputs,
                eval(&projected, &inputs, Collect::None).outputs
            );
        }
    }
}
