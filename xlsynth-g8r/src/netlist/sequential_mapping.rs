// SPDX-License-Identifier: Apache-2.0

//! Native sequential mapping policy and shared load-enable recognition.

pub(crate) mod assembly;
mod native;

pub use native::{map_sequential_to_area_netlist, map_sequential_to_native_netlist};

use std::collections::BTreeMap;

use crate::aig::get_summary_stats::get_aig_cone_stats;
use crate::aig::{AigNode, AigOperand, AigRef, GateFn, SequentialGateFn};
use crate::netlist::register_mapping::RegisterCellSelection;

/// Controls register cleanup, physical flop selection, and clock-gate
/// extraction.
#[derive(Clone, Debug)]
pub struct SequentialMappingOptions {
    pub register_selection: RegisterCellSelection,
    /// Run native register cleanup before selecting flops and clock gates.
    pub cleanup_registers: bool,
    /// `None` keeps shared load enables on the flop data paths.
    pub clock_gating: Option<ClockGatingOptions>,
}

impl Default for SequentialMappingOptions {
    fn default() -> Self {
        Self {
            register_selection: RegisterCellSelection::SmallestArea,
            cleanup_registers: false,
            clock_gating: Some(ClockGatingOptions::default()),
        }
    }
}

/// Inclusive size and depth thresholds for extracting an integrated clock gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClockGatingOptions {
    /// Minimum number of physical flop bits sharing the signed enable operand.
    pub min_flops: usize,
    /// Maximum number of AIG AND levels from an input or current Q to the
    /// enable.
    pub max_enable_depth: usize,
}

impl Default for ClockGatingOptions {
    fn default() -> Self {
        Self {
            min_flops: 4,
            max_enable_depth: 8,
        }
    }
}

impl ClockGatingOptions {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.min_flops == 0 {
            return Err("clock-gate minimum flop count must be positive".into());
        }
        Ok(())
    }
}

/// One physical FF bit and its logical load data when the clock is enabled.
#[derive(Clone, Copy, Debug)]
pub(super) struct ClockGateMember {
    pub(super) register_index: usize,
    pub(super) bit_index: usize,
    pub(super) data: AigOperand,
}

/// FF bits sharing the exact same signed, active-high load predicate.
#[derive(Debug)]
pub(super) struct ClockGateGroup {
    pub(super) enable: AigOperand,
    pub(super) members: Vec<ClockGateMember>,
}

/// Groups physically eligible bits of a legalized design by their load
/// predicate, then measures the combined cone of groups meeting the size limit.
pub(super) fn plan_clock_gates(
    design: &SequentialGateFn,
    options: &ClockGatingOptions,
    can_gate: impl Fn(AigOperand, AigOperand) -> bool,
) -> Vec<ClockGateGroup> {
    let graph = &design.transition;
    // Folded constant-data muxes need a literal in the mapping boundary.
    let true_bit = graph.gates.iter().enumerate().find_map(|(id, gate)| {
        if let AigNode::Literal { value, .. } = gate {
            Some(AigOperand {
                node: AigRef { id },
                negated: !value,
            })
        } else {
            None
        }
    });
    let mut groups = BTreeMap::<AigOperand, Vec<ClockGateMember>>::new();
    for (register_index, reg) in design.registers.iter().enumerate() {
        let qs = graph.inputs[reg.q.index()].bit_vector.iter_lsb_to_msb();
        let ds = graph.outputs[reg.d.index()].bit_vector.iter_lsb_to_msb();
        for (bit_index, (&q, &d)) in qs.zip(ds).enumerate() {
            if let Some((enable, data)) = feedback_mux(graph, q, d, true_bit)
                && !matches!(graph.get(enable.node), AigNode::Literal { .. })
                && data != q
                && can_gate(q, data)
            {
                groups.entry(enable).or_default().push(ClockGateMember {
                    register_index,
                    bit_index,
                    data,
                });
            }
        }
    }
    groups.retain(|_, members| members.len() >= options.min_flops);
    let roots = groups.keys().copied().collect::<Vec<_>>();
    let depths = get_aig_cone_stats(&graph.gates, &roots).root_depths;
    groups
        .into_iter()
        .zip(depths)
        .filter_map(|((enable, members), depth)| {
            (depth <= options.max_enable_depth).then_some(ClockGateGroup { enable, members })
        })
        .collect()
}

/// Finds the other input of a positive AND when either input is the specified
/// signed operand.
fn other_factor(graph: &GateFn, term: AigOperand, factor: AigOperand) -> Option<AigOperand> {
    match graph.get(term.non_negated()?) {
        AigNode::And2 { a, b, .. } if *a == factor => Some(*b),
        AigNode::And2 { a, b, .. } if *b == factor => Some(*a),
        _ => None,
    }
}

/// Recovers (enable, data) only from a complete D expression equal to
/// `enable ? data : q`. Looking through an outer reset would lose its priority.
fn feedback_mux(
    graph: &GateFn,
    q: AigOperand,
    d: AigOperand,
    true_bit: Option<AigOperand>,
) -> Option<(AigOperand, AigOperand)> {
    let AigNode::And2 { a, b, .. } = graph.get(d.node) else {
        return None;
    };
    // Regard the root as an OR of two terms by De Morgan. If D exposes the
    // AND directly, both mux arms must be complemented to recover logical D.
    let output_phase = |arm: AigOperand| if d.negated { arm } else { arm.negate() };
    let held = output_phase(q);
    let (left, right) = (a.negate(), b.negate());
    for (hold_term, load_term) in [(left, right), (right, left)] {
        if hold_term == held {
            // Q | enable (set) or, through the output phase, Q & !enable
            // (clear).
            return Some((load_term, output_phase(true_bit?)));
        }
        let Some(hold) = other_factor(graph, hold_term, held) else {
            continue;
        };
        let enable = hold.negate();
        if load_term == enable {
            // The load arm's AND with constant one was folded.
            return Some((enable, output_phase(true_bit?)));
        }
        if let Some(data) = other_factor(graph, load_term, enable) {
            return Some((enable, output_phase(data)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{
        AigBitVector, ClockPort, GateBuilder, GateBuilderOptions, RegisterBinding,
        TransitionInputId, TransitionOutputId,
    };

    #[test]
    fn feedback_requires_the_bits_own_q_and_complementary_selects() {
        let mut b = GateBuilder::new("feedback".into(), GateBuilderOptions::no_opt());
        let inputs = b.add_input("inputs".into(), 4);
        let [q, data, enable, reset] = std::array::from_fn(|i| *inputs.get_lsb(i));
        let normal = b.add_mux2(enable, data, q);
        let swapped = b.add_mux2(enable, q, data);
        let dual = b.add_mux2(enable, data.negate(), q.negate()).negate();
        let wrong_phase = b.add_mux2(enable, data, q.negate());
        let reset_priority = b.add_and_binary(normal, reset.negate());
        let hold = b.add_and_binary(enable.negate(), q);
        let load = b.add_and_binary(data, reset);
        let mismatched_select = b.add_or_binary(hold, load);
        let set = b.add_or_binary(q, enable);
        let clear = b.add_and_binary(q, enable.negate());
        let folded_load = b.add_or_binary(hold, enable);
        let one = b.get_true();
        b.add_output("out".into(), AigBitVector::from_lsb_is_index_0(&[normal]));
        let graph = b.build();
        for (d, expected) in [
            (normal, Some((enable, data))),
            (swapped, Some((enable.negate(), data))),
            (dual, Some((enable, data))),
            (wrong_phase, None),
            (reset_priority, None),
            (mismatched_select, None),
            (set, Some((enable, one))),
            (clear, Some((enable, one.negate()))),
            (folded_load, Some((enable, one))),
        ] {
            assert_eq!(feedback_mux(&graph, q, d, Some(one)), expected);
        }
        assert_eq!(feedback_mux(&graph, reset, normal, Some(one)), None);
    }

    #[test]
    fn signed_groups_span_registers_and_obey_inclusive_size_and_depth() -> Result<(), String> {
        let mut b = GateBuilder::new("thresholds".into(), GateBuilderOptions::no_opt());
        let guards = b.add_input("guards".into(), 10);
        let data = b.add_input("data".into(), 4);
        let depth8 = (1..=8).fold(*guards.get_lsb(0), |prev, i| {
            b.add_and_binary(prev, *guards.get_lsb(i))
        });
        let depth9 = b.add_and_binary(depth8, *guards.get_lsb(9));
        let mut registers = Vec::new();
        for (r, (en, width)) in [(depth8, 2), (depth8, 2), (depth8.negate(), 4), (depth9, 4)]
            .into_iter()
            .enumerate()
        {
            let q = b.add_input(format!("q{r}"), width);
            let ds = (0..width)
                .map(|i| b.add_mux2(en, *data.get_lsb(i), *q.get_lsb(i)))
                .collect::<Vec<_>>();
            b.add_output(format!("d{r}"), AigBitVector::from_lsb_is_index_0(&ds));
            registers.push(RegisterBinding {
                name: format!("r{r}"),
                q: TransitionInputId::new(r + 2),
                d: TransitionOutputId::new(r),
                reset: None,
                enable: None,
            });
        }
        let design = SequentialGateFn::new(
            "thresholds".into(),
            b.build(),
            vec![TransitionInputId::new(0), TransitionInputId::new(1)],
            vec![],
            Some(ClockPort { name: "clk".into() }),
            registers,
        )?;
        let options = ClockGatingOptions::default();
        let groups = plan_clock_gates(&design, &options, |_, _| true);
        assert_eq!(
            groups.iter().map(|g| g.enable).collect::<Vec<_>>(),
            [depth8, depth8.negate()]
        );
        let bits = |group: &ClockGateGroup| {
            group
                .members
                .iter()
                .map(|m| {
                    assert_eq!(m.data, *data.get_lsb(m.bit_index));
                    (m.register_index, m.bit_index)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&groups[0]), [(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(bits(&groups[1]), [(2, 0), (2, 1), (2, 2), (2, 3)]);
        for (min_flops, max_enable_depth, count) in [(5, 9, 0), (4, 7, 0), (4, 9, 3)] {
            let options = ClockGatingOptions {
                min_flops,
                max_enable_depth,
            };
            assert_eq!(
                plan_clock_gates(&design, &options, |_, _| true).len(),
                count
            );
        }
        let rejected_q = *design.transition.inputs[2].bit_vector.get_lsb(1);
        let remaining = plan_clock_gates(&design, &options, |q, _| q != rejected_q);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].enable, depth8.negate());
        assert!(
            ClockGatingOptions {
                min_flops: 0,
                ..options
            }
            .validate()
            .is_err()
        );
        Ok(())
    }
}
