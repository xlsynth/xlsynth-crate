// SPDX-License-Identifier: Apache-2.0

//! Query-invariant Liberty preparation for repeated incremental timing trials.
//! The interpolation kernel, edge frontiers, and predecessor tie-breaks remain
//! shared with ordinary STA; no approximate slew/load quantization is used.

use super::{
    AxisVariable, CombinationalOutputLoad, EdgeTiming, EdgeTimingCandidate, EdgeTimingSet,
    LibertyTableKind, SignalTiming, SignalTimingSet, StaTimingSense, StaTimingTableKind,
    StaTimingType, TimingEdge, TimingPredecessor, TimingQueryDiagnosticCounts, TimingTableLayout,
    TimingTableQuery, TracedCombinationalTiming, build_monotone_timing_table_envelope,
    collapse_signal_timing_set_to_envelope, constant_output_function_value, find_unique_table,
    interpolate_table_corners, resolve_table_query, split_related_pin_names, timing_table_layout,
    update_traced_timing_predecessor, uses_monotone_upper_envelope, validate_non_negative_finite,
    validate_timing_table_structure,
};
use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty::timing_table::TimingTableArrayView;
use crate::liberty_model::{Library, Pin, PinDirection, TimingTable};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::rc::Rc;

/// Immutable, validated table with direct corner indexing and exact envelopes.
struct PreparedTable<'a> {
    layout: TimingTableLayout<'a>,
    variables: [AxisVariable; 3],
    strides: [usize; 3],
    rank: usize,
    is_setup: bool,
    values: Vec<f64>,
    context: String,
}

impl<'a> PreparedTable<'a> {
    /// Resolves immutable axes and repairs the table once, outside hot queries.
    fn new(library: &'a Library, table: &'a TimingTable, context: String) -> Result<Self> {
        validate_timing_table_structure(library, table, &context)?;
        let array = TimingTableArrayView::from_timing_table(library, table)
            .map_err(|error| anyhow!("{context}: {error}"))?;
        let layout = timing_table_layout(library, table, &context)?;
        let mut strides = [0; 3];
        let mut stride = 1;
        for axis in (0..array.rank()).rev() {
            strides[axis] = stride;
            stride *= array.dimensions()[axis] as usize;
        }
        Ok(Self {
            variables: layout.variables.map(AxisVariable::from_raw),
            layout,
            strides,
            rank: array.rank(),
            is_setup: matches!(
                LibertyTableKind::from_raw(table.kind_str()),
                LibertyTableKind::RiseConstraint | LibertyTableKind::FallConstraint
            ),
            values: if uses_monotone_upper_envelope(table) {
                build_monotone_timing_table_envelope(&array)
            } else {
                array.values().iter().copied().map(f64::from).collect()
            },
            context,
        })
    }

    /// Uses the ordinary STA interpolation policy with direct corner lookup.
    fn evaluate(
        &self,
        query: TimingTableQuery,
        counts: &mut TimingQueryDiagnosticCounts,
    ) -> Result<f64> {
        let bounds = self.resolve_query(query, counts)?;
        self.evaluate_bounds(&bounds)
    }

    /// Validates a query and resolves reusable interpolation coordinates.
    fn resolve_query(
        &self,
        query: TimingTableQuery,
        counts: &mut TimingQueryDiagnosticCounts,
    ) -> Result<[(usize, usize, f64); 3]> {
        for (value, name) in [
            (query.input_transition, "input transition query"),
            (query.output_load, "output load query"),
            (
                query.constrained_pin_transition,
                "constrained pin transition query",
            ),
            (query.related_pin_transition, "related pin transition query"),
        ] {
            validate_non_negative_finite(value, name, &self.context)?;
        }
        resolve_table_query(
            self.rank,
            &self.layout,
            self.variables,
            self.is_setup,
            query,
            counts,
            &self.context,
        )
    }

    /// Looks up prepared values without repeating axis selection or bracketing.
    fn evaluate_bounds(&self, bounds: &[(usize, usize, f64); 3]) -> Result<f64> {
        if self.rank == 0 {
            return Ok(self.values[0]);
        }
        let result = interpolate_table_corners(&bounds[..self.rank], |indices| {
            let index: usize = indices
                .iter()
                .zip(self.strides)
                .map(|(i, stride)| i * stride)
                .sum();
            self.values
                .get(index)
                .copied()
                .ok_or_else(|| anyhow!("{}: invalid table index", self.context))
        })?;
        if !result.is_finite() {
            return Err(anyhow!(
                "{}: timing table evaluation produced non-finite result {}",
                self.context,
                result
            ));
        }
        Ok(result)
    }
}

struct PreparedEdge<'a> {
    delay: PreparedTable<'a>,
    slew: PreparedTable<'a>,
    shared_coordinates: bool,
}

impl<'a> PreparedEdge<'a> {
    /// Records whether both surfaces can use exactly the same query brackets.
    fn new(delay: PreparedTable<'a>, slew: PreparedTable<'a>) -> Self {
        let shared_coordinates = delay.rank == slew.rank
            && delay.is_setup == slew.is_setup
            && delay.layout.variables == slew.layout.variables
            && delay.layout.axes == slew.layout.axes;
        Self {
            delay,
            slew,
            shared_coordinates,
        }
    }

    /// Shares coordinate work without changing the per-table diagnostics.
    fn evaluate_tables(
        &self,
        query: TimingTableQuery,
        counts: &mut TimingQueryDiagnosticCounts,
    ) -> Result<(f64, f64)> {
        if !self.shared_coordinates || super::sta_trace_enabled() {
            return Ok((
                self.delay.evaluate(query, counts)?,
                self.slew.evaluate(query, counts)?,
            ));
        }
        let mut query_counts = TimingQueryDiagnosticCounts::default();
        let bounds = self.delay.resolve_query(query, &mut query_counts)?;
        counts.accumulate(query_counts);
        let delay = self.delay.evaluate_bounds(&bounds)?;
        counts.accumulate(query_counts);
        Ok((delay, self.slew.evaluate_bounds(&bounds)?))
    }

    /// Preserves exact source-edge order, slew envelopes, and path accounting.
    fn evaluate(
        &self,
        sources: &EdgeTimingSet,
        load: f64,
        counts: &mut TimingQueryDiagnosticCounts,
    ) -> Result<EdgeTimingSet> {
        let mut outputs = EdgeTimingSet::default();
        for source in sources.iter() {
            let query = TimingTableQuery::combinational(source.timing.transition, load);
            let (delay, transition) = self.evaluate_tables(query, counts)?;
            let transition = transition.max(0.0);
            let arrival = source.timing.arrival + delay;
            if !arrival.is_finite() {
                return Err(anyhow!(
                    "{}: propagated arrival must be finite",
                    self.delay.context
                ));
            }
            outputs.insert(EdgeTimingCandidate {
                timing: EdgeTiming {
                    arrival,
                    transition,
                },
                source_edge: source.source_edge,
                register_path_breakdown: source.register_path_breakdown.map(|mut breakdown| {
                    breakdown.combinational_delay += delay;
                    breakdown
                }),
            });
        }
        Ok(outputs)
    }
}

struct PreparedArc<'a> {
    related: Vec<&'a str>,
    when: Option<Term>,
    sense: StaTimingSense,
    // Defer table errors until an active arc has a connected timing source,
    // just as ordinary STA does for false conditional arcs.
    rise: Result<Option<PreparedEdge<'a>>, String>,
    fall: Result<Option<PreparedEdge<'a>>, String>,
}

/// One output's resolved arcs, parsed predicates, and precomputed table data.
pub(crate) struct PreparedCombinationalOutput<'a> {
    arcs: Vec<PreparedArc<'a>>,
    constant: bool,
    context: String,
}

impl<'a> PreparedCombinationalOutput<'a> {
    /// Parses conditions and prepares each arc's rise/fall tables once.
    fn new(library: &'a Library, cell_name: &str, pin: &'a Pin) -> Result<Self> {
        let context = format!(
            "cell '{cell_name}' output pin '{}'",
            library.resolve_string(&pin.name)
        );
        if pin.direction != PinDirection::Output as i32 {
            return Err(anyhow!("{context}: not an output pin"));
        }
        let constant = pin.timing_arcs.is_empty()
            && constant_output_function_value(library, cell_name, pin)?.is_some();
        let mut arcs = Vec::new();
        for arc in &pin.timing_arcs {
            let related = library.resolve_string(&arc.related_pin);
            let context = format!("{context} timing arc related_pin '{related}'");
            let kind = StaTimingType::from_raw(arc.timing_type_str(library));
            let sense = StaTimingSense::from_raw(arc.timing_sense_str(library));
            if !kind.is_combinational()
                || !matches!(
                    sense,
                    StaTimingSense::PositiveUnate
                        | StaTimingSense::NegativeUnate
                        | StaTimingSense::NonUnate
                        | StaTimingSense::Unspecified
                )
            {
                return Err(anyhow!("{context}: unsupported combinational arc"));
            }
            let when = library.resolve_string(&arc.when);
            let when = if when.is_empty() {
                None
            } else {
                Some(parse_formula(when).map_err(|error| anyhow!("{context}: {error}"))?)
            };
            let prepare =
                |delay: StaTimingTableKind, slew: StaTimingTableKind| -> Result<PreparedEdge<'a>> {
                    Ok(PreparedEdge::new(
                        PreparedTable::new(
                            library,
                            find_unique_table(arc, delay, &context)?,
                            format!("{context} {}", delay.as_raw()),
                        )?,
                        PreparedTable::new(
                            library,
                            find_unique_table(arc, slew, &context)?,
                            format!("{context} {}", slew.as_raw()),
                        )?,
                    ))
                };
            arcs.push(PreparedArc {
                related: split_related_pin_names(related).collect(),
                when,
                sense,
                rise: kind
                    .produces_rise()
                    .then(|| {
                        prepare(
                            StaTimingTableKind::CellRise,
                            StaTimingTableKind::RiseTransition,
                        )
                    })
                    .transpose()
                    .map_err(|error| format!("{error:#}")),
                fall: kind
                    .produces_fall()
                    .then(|| {
                        prepare(
                            StaTimingTableKind::CellFall,
                            StaTimingTableKind::FallTransition,
                        )
                    })
                    .transpose()
                    .map_err(|error| format!("{error:#}")),
            });
        }
        Ok(Self {
            arcs,
            constant,
            context,
        })
    }

    /// Matches ordinary STA's arc order, non-unate frontier, and predecessor
    /// ties.
    pub(crate) fn evaluate(
        &self,
        inputs: &[(&str, SignalTiming)],
        load: CombinationalOutputLoad,
        known: &HashMap<String, bool>,
        counts: &mut TimingQueryDiagnosticCounts,
    ) -> Result<TracedCombinationalTiming> {
        if self.constant {
            let zero = EdgeTiming {
                arrival: 0.0,
                transition: 0.0,
            };
            return Ok(TracedCombinationalTiming {
                timing: SignalTiming {
                    rise: zero,
                    fall: zero,
                },
                rise_predecessor: None,
                fall_predecessor: None,
            });
        }
        let mut accumulated: Option<SignalTimingSet> = None;
        let mut rise_winner: Option<(EdgeTimingCandidate, TimingPredecessor)> = None;
        let mut fall_winner: Option<(EdgeTimingCandidate, TimingPredecessor)> = None;
        for arc in &self.arcs {
            if arc
                .when
                .as_ref()
                .is_some_and(|when| when.evaluate_partial(known) == Some(false))
            {
                continue;
            }
            for related in &arc.related {
                let Some((index, (_, input))) = inputs
                    .iter()
                    .enumerate()
                    .find(|(_, (name, _))| name == related)
                else {
                    continue;
                };
                let input = SignalTimingSet::from_single(*input);
                let mut both = EdgeTimingSet::default();
                if arc.sense.may_use_either_input_edge() {
                    both = input.rise.clone();
                    both.extend_from(&input.fall);
                }
                let mut output = SignalTimingSet::default();
                for (rise, edge, destination, load) in [
                    (true, &arc.rise, &mut output.rise, load.rise),
                    (false, &arc.fall, &mut output.fall, load.fall),
                ] {
                    if let Some(edge) = edge.as_ref().map_err(|error| anyhow!(error.clone()))? {
                        let source = match (arc.sense, rise) {
                            (StaTimingSense::PositiveUnate, true)
                            | (StaTimingSense::NegativeUnate, false) => &input.rise,
                            (StaTimingSense::PositiveUnate, false)
                            | (StaTimingSense::NegativeUnate, true) => &input.fall,
                            _ => &both,
                        };
                        *destination = edge.evaluate(source, load, counts)?;
                    }
                }
                update_traced_timing_predecessor(
                    &mut rise_winner,
                    output.rise.max_arrival_candidate(),
                    index,
                );
                update_traced_timing_predecessor(
                    &mut fall_winner,
                    output.fall.max_arrival_candidate(),
                    index,
                );
                accumulated = Some(match accumulated {
                    Some(previous) => previous.merge(&output),
                    None => output,
                });
            }
        }
        let mut output =
            accumulated.ok_or_else(|| anyhow!("{}: no timing candidates", self.context))?;
        collapse_signal_timing_set_to_envelope(&mut output);
        Ok(TracedCombinationalTiming {
            timing: output
                .as_report_signal_timing()
                .ok_or_else(|| anyhow!("{}: incomplete rise/fall result", self.context))?,
            rise_predecessor: rise_winner.map(|(_, predecessor)| predecessor),
            fall_predecessor: fall_winner.map(|(_, predecessor)| predecessor),
        })
    }
}

/// Query-independent arcs for one mapper input pin, with all other pins
/// unknown.
pub(crate) struct PreparedPinTiming<'a> {
    arcs: Vec<PreparedPinArc<'a>>,
}

struct PreparedPinArc<'a> {
    sense: StaTimingSense,
    rise: Result<Option<PreparedEdge<'a>>, String>,
    fall: Result<Option<PreparedEdge<'a>>, String>,
}

impl<'a> PreparedPinTiming<'a> {
    /// Prepares only the arcs observed by the ordinary single-pin
    /// characterizer.
    pub(crate) fn new(
        library: &'a Library,
        cell_name: &str,
        output: &'a Pin,
        input: &str,
    ) -> Result<Self> {
        let context = format!("cell '{cell_name}' input '{input}' characterization");
        let known = HashMap::new();
        let mut arcs = Vec::new();
        for arc in &output.timing_arcs {
            if !split_related_pin_names(library.resolve_string(&arc.related_pin))
                .any(|related| related == input)
                || !super::arc_when_may_apply(library, arc, &known, &context)?
            {
                continue;
            }
            let kind = StaTimingType::from_raw(arc.timing_type_str(library));
            if !kind.is_combinational() {
                return Err(anyhow!("{context}: non-combinational arc"));
            }
            let sense = StaTimingSense::from_raw(arc.timing_sense_str(library));
            if !matches!(
                sense,
                StaTimingSense::PositiveUnate
                    | StaTimingSense::NegativeUnate
                    | StaTimingSense::NonUnate
                    | StaTimingSense::Unspecified
            ) {
                return Err(anyhow!("{context}: unsupported timing sense"));
            }
            let prepare = |delay, slew| -> Result<PreparedEdge<'a>> {
                Ok(PreparedEdge::new(
                    PreparedTable::new(
                        library,
                        find_unique_table(arc, delay, &context)?,
                        context.clone(),
                    )?,
                    PreparedTable::new(
                        library,
                        find_unique_table(arc, slew, &context)?,
                        context.clone(),
                    )?,
                ))
            };
            arcs.push(PreparedPinArc {
                sense,
                rise: kind
                    .produces_rise()
                    .then(|| {
                        prepare(
                            StaTimingTableKind::CellRise,
                            StaTimingTableKind::RiseTransition,
                        )
                    })
                    .transpose()
                    .map_err(|error| format!("{error:#}")),
                fall: kind
                    .produces_fall()
                    .then(|| {
                        prepare(
                            StaTimingTableKind::CellFall,
                            StaTimingTableKind::FallTransition,
                        )
                    })
                    .transpose()
                    .map_err(|error| format!("{error:#}")),
            });
        }
        Ok(Self { arcs })
    }

    /// Evaluates a zero-arrival edge without constructing temporary frontiers.
    pub(crate) fn characterize(
        &self,
        input_edge: TimingEdge,
        transition: f64,
        load: CombinationalOutputLoad,
    ) -> Result<[Option<EdgeTiming>; 2]> {
        let mut result: [Option<EdgeTiming>; 2] = [None, None];
        let mut counts = TimingQueryDiagnosticCounts::default();
        for arc in &self.arcs {
            for (index, (rise, edge, load)) in
                [(true, &arc.rise, load.rise), (false, &arc.fall, load.fall)]
                    .into_iter()
                    .enumerate()
            {
                let allowed = match arc.sense {
                    StaTimingSense::PositiveUnate => rise == (input_edge == TimingEdge::Rise),
                    StaTimingSense::NegativeUnate => rise != (input_edge == TimingEdge::Rise),
                    _ => true,
                };
                if !allowed {
                    continue;
                }
                let Some(edge) = edge.as_ref().map_err(|error| anyhow!(error.clone()))? else {
                    continue;
                };
                let query = TimingTableQuery::combinational(transition, load);
                let (delay, slew) = edge.evaluate_tables(query, &mut counts)?;
                let response = EdgeTiming {
                    arrival: 0.0 + delay,
                    transition: slew.max(0.0),
                };
                result[index] = Some(match result[index] {
                    None => response,
                    Some(previous) => EdgeTiming {
                        arrival: previous.arrival.max(response.arrival),
                        transition: previous.transition.max(response.transition),
                    },
                });
            }
        }
        Ok(result)
    }
}

/// A library-borrowed cache; no pointer identities survive their source
/// library.
pub(crate) struct PreparedTimingLibrary<'a> {
    library: &'a Library,
    outputs: HashMap<(usize, usize), Rc<PreparedCombinationalOutput<'a>>>,
}

impl<'a> PreparedTimingLibrary<'a> {
    pub(crate) fn new(library: &'a Library) -> Self {
        Self {
            library,
            outputs: HashMap::new(),
        }
    }

    /// Returns shared preparation indexed by immutable cell and output IDs.
    pub(crate) fn output(
        &mut self,
        cell: usize,
        pin: usize,
    ) -> Result<Rc<PreparedCombinationalOutput<'a>>> {
        if let Some(output) = self.outputs.get(&(cell, pin)) {
            return Ok(Rc::clone(output));
        }
        let cell_data = &self.library.cells[cell];
        let output = Rc::new(PreparedCombinationalOutput::new(
            self.library,
            &cell_data.name,
            &cell_data.pins[pin],
        )?);
        self.outputs.insert((cell, pin), Rc::clone(&output));
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_model::{Cell, LibraryBuilder, LuTableTemplate};
    use crate::liberty_proto::TimingTableKind;
    use crate::netlist::sta::{
        MinimumCharacterizedAxis, characterize_combinational_pin_edge,
        evaluate_combinational_cell_output_timing_with_predecessors,
        evaluate_table_with_query_and_diagnostics, interpolate_table_corners,
        interpolate_table_corners_generic,
    };

    #[test]
    fn specialized_interpolation_preserves_generic_rounding() {
        let mut seed = 1234567u64;
        for iteration in 0..1000 {
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 1.0
            };
            let values = [next(), next(), next(), next(), -0.0, 0.0];
            let bounds = [
                (0, usize::from(iteration & 1 != 0), next()),
                (0, usize::from(iteration & 2 != 0), next()),
            ];
            for rank in 1..=2 {
                let lookup = |indices: &[usize]| {
                    Ok(
                        values[(indices[0] * 2 + indices.get(1).copied().unwrap_or(0) + iteration)
                            % values.len()],
                    )
                };
                let expected = interpolate_table_corners_generic(&bounds[..rank], lookup).unwrap();
                let actual = interpolate_table_corners(&bounds[..rank], lookup).unwrap();
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn prepared_pin_defers_inactive_edge_and_unrelated_arc_errors() {
        let mut builder = LibraryBuilder::new();
        let mut tables = Vec::new();
        for kind in [TimingTableKind::CellRise, TimingTableKind::RiseTransition] {
            tables.push(
                builder
                    .add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![1.0], vec![], "")
                    .unwrap(),
            );
        }
        let good = builder
            .add_timing_arc("A", "positive_unate", "combinational", "", tables)
            .unwrap();
        let unrelated = builder
            .add_timing_arc("B", "positive_unate", "combinational", "", vec![])
            .unwrap();
        let inactive = builder
            .add_timing_arc("A", "positive_unate", "combinational", "0", vec![])
            .unwrap();
        let output = Pin {
            name: builder.intern_string("Y").unwrap(),
            direction: PinDirection::Output as i32,
            timing_arcs: vec![good, unrelated, inactive],
            ..Default::default()
        };
        builder.cells.push(Cell {
            name: "GATE".into(),
            pins: vec![output],
            ..Default::default()
        });
        let library = builder.finish();
        let pin = &library.cells[0].pins[0];
        let prepared = PreparedPinTiming::new(&library, "GATE", pin, "A").unwrap();
        let load = CombinationalOutputLoad::default();
        assert_eq!(
            prepared.characterize(TimingEdge::Rise, 0.1, load).unwrap(),
            characterize_combinational_pin_edge(
                &library,
                "GATE",
                pin,
                "A",
                TimingEdge::Rise,
                0.1,
                load
            )
            .unwrap()
        );
        assert!(prepared.characterize(TimingEdge::Fall, 0.1, load).is_err());
        assert!(
            characterize_combinational_pin_edge(
                &library,
                "GATE",
                pin,
                "A",
                TimingEdge::Fall,
                0.1,
                load
            )
            .is_err()
        );
    }

    #[test]
    fn inactive_or_unconnected_arcs_do_not_raise_prepared_table_errors() {
        let mut builder = LibraryBuilder::new();
        let mut tables = Vec::new();
        for kind in [
            TimingTableKind::CellRise,
            TimingTableKind::CellFall,
            TimingTableKind::RiseTransition,
            TimingTableKind::FallTransition,
        ] {
            tables.push(
                builder
                    .add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![1.0], vec![], "")
                    .unwrap(),
            );
        }
        let good = builder
            .add_timing_arc("A", "positive_unate", "combinational", "", tables)
            .unwrap();
        let inactive = builder
            .add_timing_arc("B", "positive_unate", "combinational", "B", vec![])
            .unwrap();
        let pin = Pin {
            name: builder.intern_string("Y").unwrap(),
            direction: PinDirection::Output as i32,
            timing_arcs: vec![good, inactive],
            ..Default::default()
        };
        builder.cells.push(Cell {
            name: "GATE".into(),
            pins: vec![pin],
            ..Default::default()
        });
        let library = builder.finish();
        let mut cache = PreparedTimingLibrary::new(&library);
        let prepared = cache.output(0, 0).unwrap();
        let edge = EdgeTiming {
            arrival: 0.0,
            transition: 1.0,
        };
        let input = SignalTiming {
            rise: edge,
            fall: edge,
        };
        for (inputs, known) in [
            (vec![("A", input)], HashMap::new()),
            (
                vec![("A", input), ("B", input)],
                HashMap::from([("B".into(), false)]),
            ),
        ] {
            let expected = evaluate_combinational_cell_output_timing_with_predecessors(
                &library,
                "GATE",
                &library.cells[0].pins[0],
                &inputs,
                CombinationalOutputLoad::default(),
                &known,
                &mut TimingQueryDiagnosticCounts::default(),
            )
            .unwrap();
            let actual = prepared
                .evaluate(
                    &inputs,
                    CombinationalOutputLoad::default(),
                    &known,
                    &mut TimingQueryDiagnosticCounts::default(),
                )
                .unwrap();
            assert_eq!(actual, expected);
        }
        assert!(
            prepared
                .evaluate(
                    &[("A", input), ("B", input)],
                    CombinationalOutputLoad::default(),
                    &HashMap::new(),
                    &mut TimingQueryDiagnosticCounts::default()
                )
                .is_err()
        );
    }

    #[test]
    fn prepared_tables_exactly_match_all_ranks_clamps_and_extrapolation() {
        for rank in 0..=3 {
            for kind in [
                TimingTableKind::CellRise,
                TimingTableKind::RiseTransition,
                TimingTableKind::RiseConstraint,
            ] {
                let mut builder = LibraryBuilder::new();
                builder.lu_table_templates.push(LuTableTemplate {
                    name: "axes".into(),
                    kind: "lu_table_template".to_string().into(),
                    variable_1: "input_net_transition".to_string().into(),
                    variable_2: "total_output_net_capacitance".to_string().into(),
                    variable_3: "constrained_pin_transition".to_string().into(),
                    index_1: if rank >= 1 { vec![0.25, 1.0] } else { vec![] },
                    index_2: if rank >= 2 { vec![0.25, 1.0] } else { vec![] },
                    index_3: if rank >= 3 { vec![0.25, 1.0] } else { vec![] },
                    ..Default::default()
                });
                let table = builder
                    .add_timing_table_f64(
                        kind,
                        1,
                        vec![],
                        vec![],
                        vec![],
                        (0..1usize << rank)
                            .map(|index| if index % 2 == 0 { 5.0 } else { -0.25 })
                            .collect(),
                        vec![2; rank],
                        "",
                    )
                    .unwrap();
                let library = builder.finish();
                let prepared =
                    PreparedTable::new(&library, &table, "prepared table".into()).unwrap();
                for slew in [0.0, 0.1, 0.25, 0.7, 1.0, 2.5] {
                    for load in [0.0, 0.1, 0.25, 0.7, 1.0, 2.5] {
                        for minimum in [
                            MinimumCharacterizedAxis::None,
                            MinimumCharacterizedAxis::InputTransition,
                            MinimumCharacterizedAxis::RelatedPinTransition,
                        ] {
                            let mut query = TimingTableQuery::combinational(slew, load);
                            query.minimum_characterized_axis = minimum;
                            let mut expected_counts = TimingQueryDiagnosticCounts::default();
                            let mut actual_counts = TimingQueryDiagnosticCounts::default();
                            let expected = evaluate_table_with_query_and_diagnostics(
                                &library,
                                &table,
                                query,
                                &mut expected_counts,
                                "reference",
                            )
                            .unwrap();
                            let actual = prepared.evaluate(query, &mut actual_counts).unwrap();
                            assert_eq!(actual.to_bits(), expected.to_bits());
                            assert_eq!(actual_counts, expected_counts);
                        }
                    }
                }
                assert!(
                    prepared
                        .evaluate(
                            TimingTableQuery::combinational(f64::NAN, 0.0),
                            &mut TimingQueryDiagnosticCounts::default()
                        )
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn prepared_arcs_preserve_conditions_polarities_and_predecessors() {
        for sense in ["positive_unate", "negative_unate", "non_unate", ""] {
            let mut builder = LibraryBuilder::new();
            let mut arcs = Vec::new();
            for (related, when) in [("A", ""), ("B", ""), ("A B", "A & !B")] {
                let mut tables = Vec::new();
                for (kind, values) in [
                    (TimingTableKind::CellRise, vec![1.0, 2.0, 4.0, 3.0]),
                    (TimingTableKind::CellFall, vec![2.0, 4.0, 3.0, 1.0]),
                    (TimingTableKind::RiseTransition, vec![-0.1, 0.2, 0.1, 0.3]),
                    (TimingTableKind::FallTransition, vec![0.2, 0.1, 0.4, 0.3]),
                ] {
                    tables.push(
                        builder
                            .add_timing_table_f64(
                                kind,
                                0,
                                vec![0.25, 1.0],
                                vec![0.25, 1.0],
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
                        .add_timing_arc(related, sense, "combinational", when, tables)
                        .unwrap(),
                );
            }
            let pin = Pin {
                name: builder.intern_string("Y").unwrap(),
                direction: PinDirection::Output as i32,
                timing_arcs: arcs,
                ..Default::default()
            };
            builder.cells.push(Cell {
                name: "GATE".into(),
                pins: vec![pin],
                ..Default::default()
            });
            let library = builder.finish();
            let mut cache = PreparedTimingLibrary::new(&library);
            let prepared = cache.output(0, 0).unwrap();
            assert!(Rc::ptr_eq(&prepared, &cache.output(0, 0).unwrap()));
            for pin in ["A", "B", "unconnected"] {
                let prepared_pin =
                    PreparedPinTiming::new(&library, "GATE", &library.cells[0].pins[0], pin)
                        .unwrap();
                for edge in [TimingEdge::Rise, TimingEdge::Fall] {
                    for slew in [0.0, 0.1, 0.4, 1.0, 3.0] {
                        for load in [
                            CombinationalOutputLoad::default(),
                            CombinationalOutputLoad {
                                rise: 0.7,
                                fall: 2.0,
                            },
                        ] {
                            let expected = characterize_combinational_pin_edge(
                                &library,
                                "GATE",
                                &library.cells[0].pins[0],
                                pin,
                                edge,
                                slew,
                                load,
                            )
                            .unwrap();
                            let actual = prepared_pin.characterize(edge, slew, load).unwrap();
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
            for known in [
                HashMap::new(),
                HashMap::from([("A".into(), true), ("B".into(), false)]),
                HashMap::from([("A".into(), false)]),
            ] {
                for slew in [0.1, 0.4, 1.0, 3.0] {
                    let a = SignalTiming {
                        rise: EdgeTiming {
                            arrival: 5.0,
                            transition: slew,
                        },
                        fall: EdgeTiming {
                            arrival: 3.0,
                            transition: slew / 2.0,
                        },
                    };
                    let b = SignalTiming {
                        rise: a.fall,
                        fall: a.rise,
                    };
                    for inputs in [
                        vec![("A", a), ("B", b)],
                        vec![("B", b), ("A", a)],
                        vec![("A", a)],
                    ] {
                        let load = CombinationalOutputLoad {
                            rise: 0.7,
                            fall: 2.0,
                        };
                        let mut expected_counts = TimingQueryDiagnosticCounts::default();
                        let mut actual_counts = TimingQueryDiagnosticCounts::default();
                        let expected = evaluate_combinational_cell_output_timing_with_predecessors(
                            &library,
                            "GATE",
                            &library.cells[0].pins[0],
                            &inputs,
                            load,
                            &known,
                            &mut expected_counts,
                        )
                        .unwrap();
                        let actual = prepared
                            .evaluate(&inputs, load, &known, &mut actual_counts)
                            .unwrap();
                        assert_eq!(actual, expected);
                        assert_eq!(actual_counts, expected_counts);
                    }
                }
            }
        }
    }
}
