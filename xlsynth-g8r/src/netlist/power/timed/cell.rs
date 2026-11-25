// SPDX-License-Identifier: Apache-2.0

use super::super::{
    ConditionValues, InstanceTopology, Library, Pin, PowerInstance, PowerTransitionKind,
    RawLutQuery, RawLutQueryDiagnostics, TimingTableKind, evaluate_power_lut,
    evaluate_timing_lut_raw, internal_power_table, timing_sense_matches, timing_type_allows_output,
};
use super::OutputCauses;

/// One actual instance pin edge with its characterized or supplied slew.
#[derive(Clone, Copy)]
pub(super) struct PinEdge {
    pub(super) time: f64,
    pub(super) round: usize,
    pub(super) pin: usize,
    pub(super) value: bool,
    pub(super) slew: f64,
}

/// Threshold delay and output slew from the same sensitized timing arc.
pub(super) struct CellResponse {
    pub(super) delay: f64,
    pub(super) slew: f64,
}

/// Selectors for input-owned energy use the batch predecessor. Output-owned
/// selectors use states that sensitize the particular committed crossing.
pub(super) enum PowerEvent<'a> {
    Inputs {
        edges: &'a [PinEdge],
        prior: &'a [bool],
    },
    Output {
        edge: PinEdge,
        causes: &'a OutputCauses,
    },
}

/// Liberty characterization and resolved pin selectors for one instance.
pub(super) struct CellContext<'a> {
    pub(super) library: &'a Library,
    pub(super) instance: &'a PowerInstance,
    pub(super) topology: &'a InstanceTopology,
}

impl CellContext<'_> {
    fn pin(&self, index: usize) -> &Pin {
        &self.library.cells[self.topology.cell].pins[self.topology.liberty_pins[index]]
    }

    /// Evaluates a selector on the pins just before the causative input change.
    fn enabled(&self, condition: Option<usize>, prior: &[bool]) -> Result<bool, String> {
        condition.map_or(Ok(true), |index| {
            self.topology.bindings.conditions[index].evaluate(
                self.library,
                ConditionValues {
                    pins: prior,
                    state_aliases: &[],
                },
            )
        })
    }

    /// Finds a changing input that causes the output edge while its Liberty
    /// selector holds. An output's self-related power group can use any cause.
    fn causal_source(
        &self,
        causes: &OutputCauses,
        eligible: impl Fn(PinEdge) -> bool,
        condition: Option<usize>,
    ) -> Result<Option<PinEdge>, String> {
        let mut pins = None;
        for (index, &edge) in causes.edges.iter().enumerate() {
            if !eligible(edge) {
                continue;
            }
            for row in causes.predecessors(index) {
                if condition.is_none() {
                    return Ok(Some(edge));
                }
                let predecessor = if row == 0 {
                    &causes.prior
                } else {
                    let pins = pins.get_or_insert_with(|| causes.prior.clone());
                    causes.apply_row(row, pins);
                    pins
                };
                if self.enabled(condition, predecessor)? {
                    return Ok(Some(edge));
                }
            }
        }
        Ok(None)
    }

    /// Selects the earliest eligible crossing, breaking exact ties by pin slot
    /// and Liberty arc order.
    pub(super) fn response(
        &self,
        output: usize,
        rise: bool,
        causes: &OutputCauses,
        output_load: f64,
    ) -> Result<CellResponse, String> {
        let name = format!(
            "{}.{}",
            self.instance.instance_name, self.instance.pins[output].pin_name
        );
        let mut best: Option<(CellResponse, usize)> = None;
        let (delay_kind, slew_kind) = if rise {
            (TimingTableKind::CellRise, TimingTableKind::RiseTransition)
        } else {
            (TimingTableKind::CellFall, TimingTableKind::FallTransition)
        };
        let bindings = &self.topology.bindings.pins[output];
        for (arc, binding) in self
            .pin(output)
            .timing_arcs
            .iter()
            .zip(&bindings.timing_arcs)
        {
            if !timing_type_allows_output(
                arc.timing_type_str(self.library),
                rise,
                PowerTransitionKind::Combinational,
                false,
            ) {
                continue;
            }
            let related = *binding.related_pin.as_ref().map_err(|pin| {
                format!(
                    "unknown related timing pin '{}' on {name}",
                    self.library.resolve_string(pin)
                )
            })?;
            let sense = arc.timing_sense_str(self.library);
            let Some(edge) = self.causal_source(
                causes,
                |edge| edge.pin == related && timing_sense_matches(sense, edge.value, rise),
                binding.condition,
            )?
            else {
                continue;
            };
            if !matches!(
                sense,
                "" | "non_unate" | "positive_unate" | "negative_unate"
            ) {
                return Err(format!("{name} has unsupported timing sense '{sense}'"));
            }
            let query = RawLutQuery {
                input_transition: edge.slew,
                output_load,
            };
            let lookup = |kind| {
                let mut tables = arc.tables.iter().filter(|t| t.kind == kind);
                let table = tables
                    .next()
                    .ok_or_else(|| format!("missing {kind:?} timing table for {name}"))?;
                if tables.next().is_some() {
                    return Err(format!("ambiguous {kind:?} timing table for {name}"));
                }
                evaluate_timing_lut_raw(
                    self.library,
                    table,
                    query,
                    &mut RawLutQueryDiagnostics::default(),
                    &name,
                )
                .map_err(|e| e.to_string())
            };
            let delay = lookup(delay_kind)?;
            let slew = lookup(slew_kind)?;
            if !delay.is_finite() || delay < 0.0 || !slew.is_finite() || slew < 0.0 {
                return Err(format!(
                    "{name} has unsupported delay {delay} or slew {slew}"
                ));
            }
            if best.as_ref().is_none_or(|(prev, pin)| {
                delay < prev.delay || (delay == prev.delay && related < *pin)
            }) {
                best = Some((CellResponse { delay, slew }, related));
            }
        }
        best.map(|(response, _)| response).ok_or_else(|| {
            format!("no sensitized combinational timing arc on {name} for output value {rise}")
        })
    }

    /// Averages competing selectors per rail. Input-owned groups use the
    /// related edge's direction; output-owned groups use the output direction.
    pub(super) fn internal_energy(
        &self,
        owner: usize,
        event: PowerEvent<'_>,
        output_load: f64,
    ) -> Result<f64, String> {
        let bindings = &self.topology.bindings.pins[owner];
        let pin = self.pin(owner);
        if pin.internal_power.is_empty() {
            return Ok(0.0);
        }
        let name = format!(
            "{}.{}",
            self.instance.instance_name, self.instance.pins[owner].pin_name
        );
        let mut rails = vec![(0.0, 0usize); bindings.rail_count];
        for (group, binding) in pin.internal_power.iter().zip(&bindings.power_groups) {
            if let PowerEvent::Inputs { prior, .. } = &event
                && !self.enabled(binding.condition, prior)?
            {
                continue;
            }
            let related = binding.related_pins.as_ref().map_err(|pin| {
                format!(
                    "unknown related power pin '{}' on {name}",
                    self.library.resolve_string(pin)
                )
            })?;
            for &source in related {
                let (edge, rise) = match &event {
                    PowerEvent::Inputs { edges, .. } => {
                        let Some(edge) = edges.iter().find(|edge| edge.pin == source) else {
                            continue;
                        };
                        (*edge, edge.value)
                    }
                    PowerEvent::Output { edge, causes } => {
                        let Some(input) = self.causal_source(
                            causes,
                            |input| source == owner || source == input.pin,
                            binding.condition,
                        )?
                        else {
                            continue;
                        };
                        (if source == owner { *edge } else { input }, edge.value)
                    }
                };
                let Some(table) = internal_power_table(group, rise, self.instance, owner)? else {
                    continue;
                };
                let energy = evaluate_power_lut(
                    self.library,
                    table,
                    RawLutQuery {
                        input_transition: edge.slew,
                        output_load,
                    },
                    &mut RawLutQueryDiagnostics::default(),
                    &name,
                )
                .map_err(|e| e.to_string())?;
                let rail = &mut rails[binding.rail];
                rail.0 += energy;
                rail.1 += 1;
            }
        }
        Ok(rails
            .into_iter()
            .filter(|(_, n)| *n != 0)
            .map(|(total, n)| total / n as f64)
            .sum())
    }
}
