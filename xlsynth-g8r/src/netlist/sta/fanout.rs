// SPDX-License-Identifier: Apache-2.0

//! Scalar cell adapter for topology-changing fanout optimization.
//! Interpolation, arc selection, conditional predicates and coverage checks use
//! shared STA.

use super::*;
use anyhow::ensure;
use std::cell::RefCell;

/// Scalar connection retained by sparse fanout transactions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Source {
    Net(usize),
    Literal(bool),
    Empty,
}

/// Immutable Liberty indexes and reusable table/predicate preparation.
pub(crate) struct SharedTiming<'a> {
    prepared: PreparedStaLibrary<'a>,
    evaluation: RefCell<TimingEvaluationContext<'a>>,
}

impl<'a> SharedTiming<'a> {
    /// Creates one timing session for all cells and candidates of an
    /// optimization.
    pub fn new(library: &'a Library) -> Result<Self> {
        Ok(Self {
            prepared: PreparedStaLibrary::new(library)?,
            evaluation: RefCell::new(TimingEvaluationContext::new(library)),
        })
    }
}

/// Cached pin metadata; the shared STA session owns formula/table preparation.
pub(crate) struct CachedCell {
    index: usize,
    pub names: Vec<String>,
    pub inputs: Vec<bool>,
    pub caps: Vec<[f64; 2]>,
    pub constants: Vec<Option<bool>>,
    related: Vec<Vec<Vec<Option<usize>>>>,
}

/// One cell's timing envelopes, Boolean coverage constants and evaluation work.
pub(crate) struct CellResult {
    pub timing: Vec<Option<SignalTiming>>,
    pub coverage_constants: Vec<Option<bool>>,
    pub diagnostics: TimingQueryDiagnosticCounts,
    pub arcs: usize,
}

impl CachedCell {
    /// Resolves one cell's pin identities and direct related-pin references.
    pub fn new(shared: &SharedTiming<'_>, index: usize) -> Result<Self> {
        let lib = &shared.prepared;
        let cell = &lib.library.cells[index];
        let names: Vec<_> = cell
            .pins
            .iter()
            .map(|p| lib.library.resolve_string(&p.name).to_string())
            .collect();
        let inputs: Vec<_> = cell
            .pins
            .iter()
            .map(|p| p.direction == PinDirection::Input as i32)
            .collect();
        let mut caps = Vec::new();
        let mut constants = Vec::new();
        let mut related = Vec::new();
        for (p, pin) in cell.pins.iter().enumerate() {
            let cap = effective_input_capacitance_by_edge(pin, &cell.name)?;
            caps.push([cap.rise, cap.fall]);
            constants.push(if inputs[p] {
                None
            } else {
                // Unused outputs may have unsupported formulas; selected
                // outputs validate this cached result during evaluation.
                lib.timing_constant(index, p).ok().flatten()
            });
            let mut arcs = Vec::new();
            if !inputs[p] {
                for arc in &pin.timing_arcs {
                    arcs.push(
                        lib.related_pin_names(arc.related_pin)
                            .iter()
                            .map(|n| lib.pin_index(index, n))
                            .collect(),
                    );
                }
            }
            related.push(arcs);
        }
        Ok(Self {
            index,
            names,
            inputs,
            caps,
            constants,
            related,
        })
    }

    /// Infers Boolean constants using the same partial formulas as full STA.
    pub fn coverage_constants(
        &self,
        shared: &SharedTiming<'_>,
        known: &HashMap<String, bool>,
    ) -> Vec<Option<bool>> {
        if !shared.prepared.supports_boolean_analysis(self.index) {
            return vec![None; self.names.len()];
        }
        self.inputs
            .iter()
            .enumerate()
            .map(|(p, input)| {
                if *input {
                    None
                } else {
                    shared
                        .prepared
                        .constant_function(self.index, p)
                        .and_then(|f| f.evaluate_partial(known))
                }
            })
            .collect()
    }

    /// Screens replacement cells with independent input pins; context-dependent
    /// incompleteness is conservatively excluded from the bounded search.
    pub fn supports_independent_inputs(&self, shared: &SharedTiming<'_>) -> bool {
        let timing = SignalTiming {
            rise: EdgeTiming {
                arrival: 0.0,
                transition: 1.0,
            },
            fall: EdgeTiming {
                arrival: 0.0,
                transition: 1.0,
            },
        };
        let inputs: Vec<_> = self
            .inputs
            .iter()
            .map(|input| input.then_some(timing))
            .collect();
        let connections: Vec<_> = (0..self.names.len()).map(|p| (p, Source::Net(p))).collect();
        self.evaluate(
            shared,
            &inputs,
            &HashMap::new(),
            &HashMap::new(),
            &vec![[1.0; 2]; self.names.len()],
            &connections,
        )
        .is_ok()
    }

    /// Converts stable numeric bindings into the shared STA coverage interface.
    fn pin_sources(&self, connections: &[(usize, Source)]) -> HashMap<String, Vec<PinBitSource>> {
        connections
            .iter()
            .map(|(p, s)| {
                let source = match s {
                    Source::Net(n) => PinBitSource::Bit(*n),
                    Source::Literal(v) => PinBitSource::Literal(*v),
                    Source::Empty => PinBitSource::Unknown,
                };
                (self.names[*p].clone(), vec![source])
            })
            .collect()
    }

    /// Uses shared combinational kernels with the candidate's actual inputs and
    /// loads.
    pub fn evaluate(
        &self,
        shared: &SharedTiming<'_>,
        inputs: &[Option<SignalTiming>],
        known: &HashMap<String, bool>,
        coverage_known: &HashMap<String, bool>,
        loads: &[[f64; 2]],
        connections: &[(usize, Source)],
    ) -> Result<CellResult> {
        let lib = &shared.prepared;
        let cell = &lib.library.cells[self.index];
        let pins = self.pin_sources(connections);
        let order = combinational_output_pin_evaluation_order(lib, self.index, &pins)?;
        let mut evaluation = shared.evaluation.borrow_mut();
        evaluation.diagnostics = TimingQueryDiagnosticCounts::default();
        validate_instance_functional_timing_coverage(
            InstanceCoverageInputs {
                lib,
                cell_idx: self.index,
                pin_sources: &pins,
                known_pin_values: known,
                coverage_known_pin_values: coverage_known,
                output_pin_order: &order,
                instance_name: "fanout candidate",
                launched_timing: None,
            },
            &mut evaluation,
        )?;
        let mut timing: Vec<_> = inputs
            .iter()
            .map(|t| t.map(SignalTimingSet::from_single))
            .collect();
        let literal = SignalTimingSet::from_single(SignalTiming {
            rise: EdgeTiming {
                arrival: 0.0,
                transition: 0.0,
            },
            fall: EdgeTiming {
                arrival: 0.0,
                transition: 0.0,
            },
        });
        let mut evaluated = 0;
        let mut coverage_constants = vec![None; self.names.len()];
        for &p in &order {
            coverage_constants[p] = if lib.supports_boolean_analysis(self.index) {
                lib.constant_function(self.index, p)
                    .and_then(|f| f.evaluate_partial(coverage_known))
            } else {
                None
            };
            let pin = &cell.pins[p];
            ensure!(
                !lib.output_has_unsupported_arcs(self.index, p),
                "fanout optimization requires combinational arcs for {}",
                self.names[p]
            );
            if lib.timing_constant(self.index, p)?.is_some() {
                timing[p] = Some(literal.clone());
                continue;
            }
            validate_timing_tables_once(
                lib.library,
                &cell.name,
                &self.names[p],
                pin.timing_arcs.iter(),
                &mut evaluation,
            )?;
            let mut accumulated: Option<SignalTimingSet> = None;
            for (a, arc) in pin.timing_arcs.iter().enumerate() {
                if !arc_when_may_apply(lib.library, arc, known, &mut evaluation, "fanout cell")? {
                    continue;
                }
                for &r in &self.related[p][a] {
                    let r = r.ok_or_else(|| anyhow!("unknown related pin in {}", self.names[p]))?;
                    let input = timing[r]
                        .as_ref()
                        .ok_or_else(|| anyhow!("missing timing input {}", self.names[r]))?;
                    let candidate = evaluate_arc_set(
                        lib.library,
                        arc,
                        input,
                        EdgeLoadCapacitance {
                            rise: loads[p][0],
                            fall: loads[p][1],
                        },
                        &mut evaluation,
                        "fanout cell",
                    )?;
                    evaluated += 1;
                    accumulated = Some(match accumulated {
                        Some(t) => t.merge(&candidate),
                        None => candidate,
                    });
                }
            }
            let mut output = accumulated.ok_or_else(|| anyhow!("no usable output timing arcs"))?;
            collapse_signal_timing_set_to_envelope(&mut output);
            timing[p] = Some(output.clone());
            // Full STA stores each output immediately in the global net array.
            // Mirror that for a net wired back to another pin on this cell.
            if let Some((_, Source::Net(net))) = connections.iter().find(|(pin, _)| *pin == p) {
                for &(input, source) in connections {
                    if self.inputs[input] && source == Source::Net(*net) {
                        timing[input] = Some(output.clone());
                    }
                }
            }
        }
        Ok(CellResult {
            timing: timing
                .iter()
                .map(|t| {
                    t.as_ref()
                        .and_then(SignalTimingSet::as_report_signal_timing)
                })
                .collect(),
            coverage_constants,
            diagnostics: evaluation.diagnostics,
            arcs: evaluated,
        })
    }
}
