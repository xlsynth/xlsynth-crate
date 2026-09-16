// SPDX-License-Identifier: Apache-2.0

//! Internal operating points and inexpensive, edge-aware Liberty responses.

use super::cover::{CoverPlan, SolutionChoice, SourceKind};
use super::liberty_index::CellBinding;
use super::{TechMapOptions, TechMapTimingConstraints, scalar_bit_name};
use crate::aig::ChoiceAig;
use crate::liberty_model::Library;
use crate::netlist::sta::prepared::PreparedPinTiming;
use crate::netlist::sta::{
    CombinationalOutputLoad, EdgeTiming, SignalTiming, StaOptions, TimingEdge,
    TimingQueryDiagnosticCounts, effective_representative_driver,
    evaluate_combinational_cell_output_timing, evaluate_primary_input_driver_timing,
    evaluate_sequential_cell_output_timing, resolved_module_output_load,
};
use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};

/// A measured internal operating point, separate from primary-IO constraints.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct NfTimingCalibration {
    pub input_transition: f64,
    pub output_load: f64,
    pub sampled_cells: usize,
}

/// Physical Q source information for a temporary transition input.
#[derive(Clone, Debug)]
pub(super) struct RegisterSource {
    pub cell: usize,
    pub output_pin: usize,
    pub tied_inputs: HashMap<String, bool>,
}

/// Distinguishes physical register pins from temporary transition IO ports.
#[derive(Clone, Debug, Default)]
pub(super) struct ElectricalBoundary {
    pub registers: BTreeMap<usize, RegisterSource>,
    pub output_loads: BTreeMap<String, CombinationalOutputLoad>,
}

/// Exact boundary launch timing, including the real Q drive and PI driver.
pub(super) fn source_timing(
    node: usize,
    load: CombinationalOutputLoad,
    graph: &ChoiceAig,
    library: &Library,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
    boundary: &ElectricalBoundary,
) -> Result<SignalTiming> {
    let mut diagnostics = TimingQueryDiagnosticCounts::default();
    if let Some(source) = boundary.registers.get(&node) {
        let cell = &library.cells[source.cell];
        return evaluate_sequential_cell_output_timing(
            library,
            &cell.name,
            &cell.pins[source.output_pin],
            load,
            &source.tied_inputs,
            &mut diagnostics,
        );
    }
    let name = graph
        .graph()
        .inputs
        .iter()
        .find_map(|port| {
            port.bit_vector
                .iter_lsb_to_msb()
                .enumerate()
                .find_map(|(bit, operand)| {
                    (operand.node.id == node)
                        .then(|| scalar_bit_name(&port.name, bit, port.get_bit_count()))
                })
        })
        .ok_or_else(|| anyhow!("unknown mapped input node {node}"))?;
    let driver = effective_representative_driver(library)?;
    evaluate_primary_input_driver_timing(
        library,
        driver.as_ref(),
        StaOptions {
            primary_input_transition: options.primary_input_transition,
            module_output_load: options.module_output_load,
        },
        constraints
            .primary_input_arrivals
            .get(&name)
            .copied()
            .unwrap_or(0.0),
        load,
        &mut diagnostics,
    )
}

/// Resolves ordered output capacitances, replacing synthetic D-port loads.
pub(super) fn output_loads(
    graph: &ChoiceAig,
    library: &Library,
    options: &TechMapOptions,
    boundary: &ElectricalBoundary,
) -> Result<Vec<CombinationalOutputLoad>> {
    let external = resolved_module_output_load(
        library,
        StaOptions {
            primary_input_transition: options.primary_input_transition,
            module_output_load: options.module_output_load,
        },
    )?;
    Ok(graph
        .graph()
        .outputs
        .iter()
        .flat_map(|port| {
            (0..port.get_bit_count()).map(move |bit| {
                let name = scalar_bit_name(&port.name, bit, port.get_bit_count());
                boundary
                    .output_loads
                    .get(&name)
                    .copied()
                    .unwrap_or(external)
            })
        })
        .collect())
}

/// Measures a selected cover; unused AIG alternatives never contribute load.
pub(super) fn calibrate_cover(
    plan: &CoverPlan,
    graph: &ChoiceAig,
    library: &Library,
    options: &TechMapOptions,
    constraints: &TechMapTimingConstraints,
    boundary: &ElectricalBoundary,
) -> Result<NfTimingCalibration> {
    let mut loads = vec![CombinationalOutputLoad::default(); plan.solutions.len()];
    for (id, load) in plan
        .output_solutions
        .iter()
        .zip(output_loads(graph, library, options, boundary)?)
    {
        add_load(&mut loads[id.0], load);
    }
    for solution in &plan.solutions {
        if let SolutionChoice::Cell { binding, inputs } = &solution.choice {
            for (pin, input) in inputs.iter().enumerate() {
                add_load(&mut loads[input.0], binding.input_capacitances[pin]);
            }
        }
    }
    let mut timings = Vec::<SignalTiming>::with_capacity(plan.solutions.len());
    let mut slews = Vec::new();
    let mut samples = Vec::new();
    let mut diagnostics = TimingQueryDiagnosticCounts::default();
    for (index, solution) in plan.solutions.iter().enumerate() {
        let timing = match &solution.choice {
            SolutionChoice::Source(SourceKind::Input(node)) => source_timing(
                node.id,
                loads[index],
                graph,
                library,
                constraints,
                options,
                boundary,
            )?,
            SolutionChoice::Source(SourceKind::Literal(_)) => uniform_timing(0.0, 0.0),
            SolutionChoice::Cell { binding, inputs } => {
                if !binding.has_complete_timing() {
                    bail!(
                        "internal calibration needs complete timing for '{}'",
                        binding.cell_name
                    );
                }
                let inputs = inputs
                    .iter()
                    .enumerate()
                    .map(|(pin, input)| (binding.input_pin_names[pin].as_str(), timings[input.0]))
                    .collect::<Vec<_>>();
                let timing = evaluate_combinational_cell_output_timing(
                    library,
                    &binding.cell_name,
                    binding.output_pin(library),
                    &inputs,
                    loads[index],
                    &HashMap::new(),
                    &mut diagnostics,
                )?;
                // Exclude zero-load aliases and constants from the
                // operating-point sample.
                if max_load(loads[index]) > 0.0 {
                    samples.push(max_load(loads[index]));
                    slews.push(timing.rise.transition.max(timing.fall.transition));
                }
                timing
            }
        };
        timings.push(timing);
    }
    if samples.is_empty() {
        bail!("selected cover has no loaded cells to calibrate");
    }
    let sampled_cells = samples.len();
    let input_transition = median(&mut slews)?;
    let output_load = median(&mut samples)?;
    Ok(NfTimingCalibration {
        input_transition,
        output_load,
        sampled_cells,
    })
}

/// Robust deterministic operating point in the library's native units.
fn median(values: &mut [f64]) -> Result<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite() || *v < 0.0) {
        bail!("invalid internal timing samples");
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len() % 2 == 0 {
        values[middle - 1] / 2.0 + values[middle] / 2.0
    } else {
        values[middle]
    })
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PinQuery {
    cell: usize,
    output: usize,
    input: usize,
    rise: bool,
    slew: u64,
    rise_load: u64,
    fall_load: u64,
}

/// Bounded cache of zero-arrival responses; arrivals remain unquantized.
pub(super) struct PinTimingCache<'a> {
    library: &'a Library,
    context: NfTimingCalibration,
    responses: RefCell<HashMap<PinQuery, [Option<EdgeTiming>; 2]>>,
    prepared_pins: RefCell<HashMap<(usize, usize, usize), PreparedPinTiming<'a>>>,
    pub queries: Cell<usize>,
}

impl<'a> PinTimingCache<'a> {
    pub fn new(library: &'a Library, context: NfTimingCalibration) -> Self {
        Self {
            library,
            context,
            responses: RefCell::new(HashMap::new()),
            prepared_pins: RefCell::new(HashMap::new()),
            queries: Cell::new(0),
        }
    }

    /// Drops stale responses between frozen-load rounds, bounding retained
    /// memory.
    pub fn clear(&self) {
        self.responses.borrow_mut().clear();
    }

    /// Propagates one cell pin with timing sense and conservative output slews.
    pub fn pin(
        &self,
        binding: &CellBinding,
        input: usize,
        timing: SignalTiming,
        load: CombinationalOutputLoad,
    ) -> Result<(SignalTiming, f64)> {
        if !binding.has_complete_timing() {
            bail!(
                "electrical mapping needs complete timing for '{}'",
                binding.cell_name
            );
        }
        let mut result: [Option<EdgeTiming>; 2] = [None, None];
        let mut delay = 0.0_f64;
        for (rise, edge) in [(true, timing.rise), (false, timing.fall)] {
            let slew = quantize_up(edge.transition, self.context.input_transition)?;
            let rise_load = quantize_up(load.rise, self.context.output_load)?;
            let fall_load = quantize_up(load.fall, self.context.output_load)?;
            let key = PinQuery {
                cell: binding.cell_index,
                output: binding.output_pin_index,
                input,
                rise,
                slew: slew.to_bits(),
                rise_load: rise_load.to_bits(),
                fall_load: fall_load.to_bits(),
            };
            let cached = self.responses.borrow().get(&key).copied();
            let response = if let Some(response) = cached {
                response
            } else {
                self.queries.set(self.queries.get() + 1);
                let mut prepared_pins = self.prepared_pins.borrow_mut();
                let pin_key = (binding.cell_index, binding.output_pin_index, input);
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    prepared_pins.entry(pin_key)
                {
                    entry.insert(PreparedPinTiming::new(
                        self.library,
                        &binding.cell_name,
                        binding.output_pin(self.library),
                        &binding.input_pin_names[input],
                    )?);
                }
                let response = prepared_pins[&pin_key].characterize(
                    if rise {
                        TimingEdge::Rise
                    } else {
                        TimingEdge::Fall
                    },
                    slew,
                    CombinationalOutputLoad {
                        rise: rise_load,
                        fall: fall_load,
                    },
                )?;
                let mut cache = self.responses.borrow_mut();
                if cache.len() < 65_536 {
                    cache.insert(key, response);
                }
                response
            };
            for (index, response) in response.into_iter().enumerate() {
                if let Some(response) = response {
                    delay = delay.max(response.arrival);
                    let candidate = EdgeTiming {
                        arrival: edge.arrival + response.arrival,
                        transition: response.transition,
                    };
                    result[index] = Some(match result[index] {
                        None => candidate,
                        Some(previous) => EdgeTiming {
                            arrival: previous.arrival.max(candidate.arrival),
                            transition: previous.transition.max(candidate.transition),
                        },
                    });
                }
            }
        }
        Ok((
            SignalTiming {
                rise: result[0].ok_or_else(|| anyhow!("missing rise response"))?,
                fall: result[1].ok_or_else(|| anyhow!("missing fall response"))?,
            },
            delay,
        ))
    }
}

/// Rounds coordinates upward without clipping high loads or changing arrivals.
fn quantize_up(value: f64, scale: f64) -> Result<f64> {
    if !value.is_finite() || value < 0.0 {
        bail!("invalid electrical timing coordinate");
    }
    if scale <= 0.0 {
        return Ok(value);
    }
    let step = scale / 16.0;
    let rounded = (value / step).ceil() * step;
    Ok(if rounded.is_finite() {
        rounded.max(value)
    } else {
        value
    })
}

/// Creates identical rise/fall timing for an unpolarized source.
pub(super) fn uniform_timing(arrival: f64, transition: f64) -> SignalTiming {
    SignalTiming {
        rise: EdgeTiming {
            arrival,
            transition,
        },
        fall: EdgeTiming {
            arrival,
            transition,
        },
    }
}

/// Accumulates physical sink capacitance separately for both edges.
pub(super) fn add_load(destination: &mut CombinationalOutputLoad, source: CombinationalOutputLoad) {
    destination.rise += source.rise;
    destination.fall += source.fall;
}

/// Reduces edge loads only when a scalar operating point is required.
pub(super) fn max_load(load: CombinationalOutputLoad) -> f64 {
    load.rise.max(load.fall)
}

/// Returns the late edge while retaining full timing elsewhere.
pub(super) fn worst_arrival(timing: SignalTiming) -> f64 {
    timing.rise.arrival.max(timing.fall.arrival)
}

/// Keeps conservative independent arrival and slew envelopes across input pins.
pub(super) fn merge_timing(lhs: SignalTiming, rhs: SignalTiming) -> SignalTiming {
    SignalTiming {
        rise: EdgeTiming {
            arrival: lhs.rise.arrival.max(rhs.rise.arrival),
            transition: lhs.rise.transition.max(rhs.rise.transition),
        },
        fall: EdgeTiming {
            arrival: lhs.fall.arrival.max(rhs.fall.arrival),
            transition: lhs.fall.transition.max(rhs.fall.transition),
        },
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::aig::{AigOperand, GateBuilder, GateBuilderOptions};
    use crate::aig_sim::gate_sim::{Collect, eval};
    use crate::liberty_model::{Cell, LibraryBuilder, LuTableTemplate, Pin, PinDirection};
    use crate::liberty_proto::TimingTableKind;
    use crate::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
    use crate::techmap::{PreparedTechMapLibrary, cuts, nf};
    use std::collections::HashSet;
    use xlsynth::IrBits;

    /// Small complete library with distinct edge delays and slew/load
    /// sensitivity.
    pub(crate) fn electrical_library() -> Library {
        let mut builder = LibraryBuilder::new();
        builder.lu_table_templates.push(LuTableTemplate {
            kind: "lu_table_template".to_string().into(),
            name: "electrical".into(),
            variable_1: "input_net_transition".to_string().into(),
            variable_2: "total_output_net_capacitance".to_string().into(),
            index_1: vec![0.0, 4.0],
            index_2: vec![0.0, 4.0],
            ..Default::default()
        });
        for (name, inputs, function, sense) in [
            ("BUF", vec!["A"], "A", "positive_unate"),
            ("INV", vec!["A"], "!A", "negative_unate"),
            ("AND2", vec!["A", "B"], "A * B", "positive_unate"),
            ("XOR2", vec!["A", "B"], "A ^ B", "non_unate"),
        ] {
            let mut pins = Vec::new();
            let mut arcs = Vec::new();
            for input in inputs {
                pins.push(Pin {
                    name: builder.intern_string(input).unwrap(),
                    direction: PinDirection::Input as i32,
                    capacitance: Some(1.0),
                    ..Default::default()
                });
                let mut tables = Vec::new();
                for (kind, values) in [
                    (TimingTableKind::CellRise, vec![1.0, 9.0, 5.0, 13.0]),
                    (TimingTableKind::CellFall, vec![2.0, 10.0, 6.0, 14.0]),
                    (TimingTableKind::RiseTransition, vec![2.0, 6.0, 4.0, 8.0]),
                    (TimingTableKind::FallTransition, vec![3.0, 7.0, 5.0, 9.0]),
                ] {
                    tables.push(
                        builder
                            .add_timing_table_f64(
                                kind,
                                1,
                                vec![],
                                vec![],
                                vec![],
                                values,
                                vec![2, 2],
                                "",
                            )
                            .unwrap(),
                    );
                }
                arcs.push(
                    builder
                        .add_timing_arc(input, sense, "combinational", "", tables)
                        .unwrap(),
                );
            }
            pins.push(Pin {
                name: builder.intern_string("Y").unwrap(),
                direction: PinDirection::Output as i32,
                function: builder.intern_string(function).unwrap(),
                timing_arcs: arcs,
                ..Default::default()
            });
            builder.cells.push(Cell {
                name: name.into(),
                pins,
                area: 1.0,
                ..Default::default()
            });
        }
        builder.finish()
    }

    /// Fanout includes repeated outputs and dead logic that must not be
    /// charged.
    pub(crate) fn electrical_graph() -> ChoiceAig {
        let mut builder = GateBuilder::new("electrical".into(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".into(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".into(), 1).try_into().unwrap();
        let c: AigOperand = builder.add_input("c".into(), 1).try_into().unwrap();
        let first = builder.add_and_binary(a, b);
        let second = builder.add_and_binary(first, c);
        let _dead = builder.add_and_binary(first, c.negate());
        builder.add_output("o".into(), second.into());
        builder.add_output("n".into(), second.negate().into());
        ChoiceAig::without_choices(builder.build())
    }

    #[test]
    fn cached_pin_responses_preserve_edge_sense_and_unquantized_arrivals() {
        let library = electrical_library();
        let index = super::super::liberty_index::LibertyCellIndex::build_nf(&library, 6).unwrap();
        let cache = PinTimingCache::new(
            &library,
            NfTimingCalibration {
                input_transition: 4.0,
                output_load: 4.0,
                sampled_cells: 1,
            },
        );
        let input = SignalTiming {
            rise: EdgeTiming {
                arrival: 3.123,
                transition: 1.0,
            },
            fall: EdgeTiming {
                arrival: 8.456,
                transition: 2.0,
            },
        };
        let load = CombinationalOutputLoad {
            rise: 1.0,
            fall: 2.0,
        };
        for (size, truth, name) in [(1, 0b10, "BUF"), (1, 0b01, "INV"), (2, 0b0110, "XOR2")] {
            let binding = index
                .matches(size, truth)
                .iter()
                .map(|id| index.binding(*id))
                .find(|b| b.cell_name == name && b.input_negated.iter().all(|negated| !negated))
                .unwrap();
            let expected = evaluate_combinational_cell_output_timing(
                &library,
                name,
                binding.output_pin(&library),
                &binding
                    .input_pin_names
                    .iter()
                    .map(|name| (name.as_str(), input))
                    .collect::<Vec<_>>(),
                load,
                &HashMap::new(),
                &mut TimingQueryDiagnosticCounts::default(),
            )
            .unwrap();
            let (actual, _) = cache.pin(binding, 0, input, load).unwrap();
            assert_eq!(actual, expected);
            let queries = cache.queries.get();
            let shifted = SignalTiming {
                rise: EdgeTiming {
                    arrival: input.rise.arrival + 0.001,
                    ..input.rise
                },
                fall: EdgeTiming {
                    arrival: input.fall.arrival + 0.001,
                    ..input.fall
                },
            };
            let (later, _) = cache.pin(binding, 0, shifted, load).unwrap();
            assert!((later.rise.arrival - actual.rise.arrival - 0.001).abs() < 1e-12);
            assert_eq!(cache.queries.get(), queries);
            let (loaded, _) = cache
                .pin(
                    binding,
                    0,
                    input,
                    CombinationalOutputLoad {
                        rise: 3.0,
                        fall: 3.0,
                    },
                )
                .unwrap();
            assert!(loaded.rise.arrival > actual.rise.arrival);
            assert!(loaded.fall.transition > actual.fall.transition);
        }
    }

    #[test]
    fn operating_point_is_internal_and_dead_logic_is_not_loaded() {
        let graph = electrical_graph();
        let library = electrical_library();
        let options = TechMapOptions {
            primary_input_transition: 0.01,
            module_output_load: 2.0,
            ..Default::default()
        };
        let prepared = PreparedTechMapLibrary::new(&library, 6).unwrap();
        let analysis = cuts::analyze_choices(&graph).unwrap();
        let constraints = TechMapTimingConstraints::default();
        let cover = nf::build_cover_plan(
            &graph,
            &analysis,
            &library,
            &prepared.cell_index,
            &options,
            &constraints,
        )
        .unwrap();
        let context = calibrate_cover(
            &cover.plan,
            &graph,
            &library,
            &options,
            &constraints,
            &ElectricalBoundary::default(),
        )
        .unwrap();
        assert!(context.input_transition > 3.0);
        assert_eq!(context.output_load, 2.0);
        assert_eq!(context.sampled_cells, 3);
        assert_eq!(options.primary_input_transition, 0.01);
        for feedback in [false, true] {
            let mapped = nf::build_electrical_cover(
                &graph,
                &analysis,
                &library,
                &prepared.cell_index,
                &options,
                &constraints,
                context,
                &ElectricalBoundary::default(),
                feedback,
            )
            .unwrap();
            assert_eq!(mapped.representative_output_load, Some(2.0));
            assert_eq!(mapped.plan.output_solutions.len(), 2);
            let netlist = super::super::cover_search::finish_nf_cover(
                &graph, &prepared, &analysis, mapped, &options,
            )
            .unwrap();
            let projected = project_gatefn_from_netlist_and_liberty(
                &netlist.module,
                &netlist.nets,
                &netlist.interner,
                &library,
                &HashSet::new(),
                &HashSet::new(),
            )
            .unwrap();
            for assignment in 0..8 {
                let inputs = (0..3)
                    .map(|bit| IrBits::make_ubits(1, (assignment >> bit) & 1).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    eval(graph.graph(), &inputs, Collect::None).outputs,
                    eval(&projected, &inputs, Collect::None).outputs
                );
            }
        }
    }

    #[test]
    fn coordinate_rounding_and_medians_are_finite_and_deterministic() {
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]).unwrap(), 2.5);
        assert!(median(&mut [f64::NAN]).is_err());
        assert!(median(&mut []).is_err());
        assert_eq!(quantize_up(1.01, 4.0).unwrap(), 1.25);
        assert_eq!(quantize_up(100.01, 4.0).unwrap(), 100.25);
        assert!(quantize_up(f64::INFINITY, 4.0).is_err());
    }
}
