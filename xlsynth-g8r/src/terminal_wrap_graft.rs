// SPDX-License-Identifier: Apache-2.0

//! Form a terminal modular zero test from its signed, pre-optimization operand
//! cones.

mod endpoints;

use std::collections::HashMap;

use crate::aig::dce::dce;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, GateFn, Output};
use crate::aig::import::{import_gate_fn, import_gate_fn_with_observations};
use crate::aig::topo::extract_cone;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::gatify::terminal_wrap::{ReturnBit, TerminalWrapCapture};

/// Owns the input-bound rails while the ordinary optimizer consumes its input.
pub(crate) struct TerminalWrapPlan {
    rails: GateFn,
    observation: TerminalWrapCapture,
    /// Logical word coordinates in the original input ports. Their signed
    /// values can be assigned independently of the carry's entire transitive
    /// support.
    direct_word: Option<Vec<(usize, usize, bool)>>,
}

impl TerminalWrapPlan {
    pub(crate) fn new(pre: &GateFn, observation: TerminalWrapCapture) -> Option<Self> {
        let word = &observation.word_lsb;
        if word.len() < 2
            || !observation
                .outputs
                .iter()
                .flatten()
                .any(|v| matches!(v, ReturnBit::Predicate { .. }))
        {
            return None;
        }
        let mut direct_word = None;
        let computed_carry = matches!(pre.get(observation.carry.node), AigNode::And2 { .. });
        if computed_carry
            && word.iter().all(|bit| {
                matches!(
                    pre.get(bit.node),
                    AigNode::Input { .. } | AigNode::Literal { .. }
                )
            })
        {
            let (_, carry_support) = extract_cone(&[observation.carry.node], &pre.gates);
            if word.iter().all(|bit| !carry_support.contains(&bit.node)) {
                // Both endpoints must be feasible. Repeating a pin is harmless;
                // opposite phases of one pin or a fixed bit rule out an
                // endpoint.
                let mut pins = HashMap::new();
                for bit in word {
                    if !matches!(pre.get(bit.node), AigNode::Input { .. }) {
                        return None;
                    }
                    if pins
                        .insert(bit.node, bit.negated)
                        .is_some_and(|old| old != bit.negated)
                    {
                        return None;
                    }
                }
                let mut coordinates = Vec::with_capacity(pins.len());
                for (port, input) in pre.inputs.iter().enumerate() {
                    for (i, bit) in input.bit_vector.iter_lsb_to_msb().enumerate() {
                        if let Some(phase) = pins.remove(&bit.node) {
                            coordinates.push((port, i, phase ^ bit.negated));
                        }
                    }
                }
                if !pins.is_empty() {
                    return None;
                }
                direct_word = Some(coordinates);
            }
        }
        let mut outputs = vec![Output {
            name: "word".into(),
            bit_vector: AigBitVector::from_lsb_is_index_0(word),
        }];
        if direct_word.is_none() {
            outputs.push(Output {
                name: "carry".into(),
                bit_vector: AigBitVector::from_bit(observation.carry),
            });
        }
        let rails = dce(&GateFn {
            outputs,
            ..pre.clone()
        });
        Some(Self {
            rails,
            observation,
            direct_word,
        })
    }

    pub(crate) fn apply(self, stock: GateFn) -> Result<GateFn, String> {
        let obs = self.observation;
        let compatible = self.rails.inputs.len() == stock.inputs.len()
            && self
                .rails
                .inputs
                .iter()
                .zip(&stock.inputs)
                .all(|(a, b)| a.name == b.name && a.get_bit_count() == b.get_bit_count())
            && obs.outputs.len() == stock.outputs.len()
            && obs
                .outputs
                .iter()
                .zip(&stock.outputs)
                .all(|(k, o)| k.len() == o.get_bit_count());
        if !compatible {
            return Ok(stock);
        }
        let mut builder = GateBuilder::new(stock.name.clone(), GateBuilderOptions::opt());
        let inputs: Vec<_> = stock
            .inputs
            .iter()
            .map(|p| builder.add_input(p.name.clone(), p.get_bit_count()))
            .collect();
        // Seed the ordinary graph's signed ANDs so a borrowed rail can keep its
        // existing factorization and metadata. Only real outputs survive DCE.
        import_gate_fn(&stock, &inputs, &mut builder)?;
        let stock_nodes = builder.gates.len();
        use endpoints::Cut;
        let cuts = if self.direct_word.is_some() {
            // Independent primary pins of consistent phase admit both
            // endpoints.
            [Cut::Unchanged, Cut::Unchanged]
        } else {
            endpoints::analyze(&self.rails, &self.rails.outputs[0].bit_vector)
        };
        let carry = if let Some(word) = self.direct_word {
            let (port, bit, inverse) = obs
                .outputs
                .iter()
                .enumerate()
                .find_map(|(p, bits)| {
                    bits.iter().enumerate().find_map(|(i, b)| match b {
                        ReturnBit::Predicate { inverted } => Some((p, i, *inverted)),
                        ReturnBit::Constant(_) => None,
                    })
                })
                .expect("capture has a variable returned bit");
            // Cofactor imports may fold hypothetical gates onto real prefix
            // operands.
            let trial = builder.begin_append_checkpoint();
            let mut endpoint = |ones: bool| -> Result<AigOperand, String> {
                let mut bindings = inputs.clone();
                for &(p, i, inverted) in &word {
                    let value = if ones ^ inverted {
                        builder.get_true()
                    } else {
                        builder.get_false()
                    };
                    bindings[p].set_lsb(i, value);
                }
                let outputs = import_gate_fn(&stock, &bindings, &mut builder)?;
                let value = *outputs[port].get_lsb(bit);
                Ok(if inverse ^ !obs.equal {
                    value.negate()
                } else {
                    value
                })
            };
            let carry = endpoint(false)?.negate();
            let verified = carry.node.id < stock_nodes && endpoint(true)? == carry;
            builder.rollback_append_checkpoint(trial);
            if !verified {
                return Ok(stock);
            }
            Some(carry)
        } else {
            None
        };
        let extra: Vec<_> = cuts
            .iter()
            .filter_map(|cut| match cut {
                Cut::Literals(bits) => Some(AigBitVector::from_lsb_is_index_0(bits)),
                _ => None,
            })
            .collect();
        let rails = import_gate_fn_with_observations(&self.rails, &inputs, &mut builder, &extra)?;
        let flag = carry.unwrap_or_else(|| *rails[1].get_lsb(0));
        let word: Vec<_> = rails[0].iter_lsb_to_msb().copied().collect();
        let zero: Vec<_> = word.iter().map(|bit| bit.negate()).collect();
        builder.set_current_pir_node_id(obs.predicate_pir_id);
        // Folding or reusing a gate can add source provenance to an existing
        // node. Each construction owns its builder so a declined trial cannot
        // alter the winner.
        let trial = cuts
            .iter()
            .any(|cut| !matches!(cut, Cut::Unchanged))
            .then(|| builder.clone());
        let equal = endpoint_test(
            &mut builder,
            &mut Vec::new(),
            flag,
            Some(&zero),
            Some(&word),
        );
        let ordinary = with_terminal_outputs(builder, equal, &obs, &stock);
        let refactored = trial.map(|mut builder| {
            let mut observed = rails[self.rails.outputs.len()..].iter();
            let mut resolved = [None, None];
            for (i, cut) in cuts.iter().enumerate() {
                resolved[i] = match cut {
                    Cut::Impossible => None,
                    Cut::Unchanged => Some(if i == 0 { zero.clone() } else { word.clone() }),
                    Cut::Literals(_) => Some(
                        observed
                            .next()
                            .expect("each cut has an imported observation")
                            .iter_lsb_to_msb()
                            .copied()
                            .collect(),
                    ),
                };
            }
            let equal = endpoint_test(
                &mut builder,
                &mut Vec::new(),
                flag,
                resolved[0].as_deref(),
                resolved[1].as_deref(),
            );
            with_terminal_outputs(builder, equal, &obs, &stock)
        });
        let incumbent = choose_replacement(stock, ordinary);
        Ok(match refactored {
            Some(graph) => choose_replacement(incumbent, graph),
            None => incumbent,
        })
    }
}

fn with_terminal_outputs(
    mut builder: GateBuilder,
    predicate: AigOperand,
    obs: &TerminalWrapCapture,
    stock: &GateFn,
) -> GateFn {
    if let Some(id) = obs.predicate_pir_id {
        builder.add_pir_node_id(predicate.node, id);
    }
    builder.set_current_pir_node_id(None);
    for (output, continuation) in stock.outputs.iter().zip(&obs.outputs) {
        let bits: Vec<_> = continuation
            .iter()
            .map(|bit| match bit {
                ReturnBit::Constant(true) => builder.get_true(),
                ReturnBit::Constant(false) => builder.get_false(),
                ReturnBit::Predicate { inverted } if *inverted ^ !obs.equal => predicate.negate(),
                ReturnBit::Predicate { .. } => predicate,
            })
            .collect();
        builder.add_output(
            output.name.clone(),
            AigBitVector::from_lsb_is_index_0(&bits),
        );
    }
    builder.build()
}

fn endpoint_test(
    builder: &mut GateBuilder,
    depths: &mut Vec<usize>,
    carry: AigOperand,
    zero: Option<&[AigOperand]>,
    ones: Option<&[AigOperand]>,
) -> AigOperand {
    use crate::aig::reassociation::and_cube_with_arrivals;
    match (zero, ones) {
        (Some(low), Some(high)) => {
            let on = and_cube_with_arrivals(builder, depths, high.iter().copied().chain([carry]));
            let off = and_cube_with_arrivals(
                builder,
                depths,
                low.iter().copied().chain([carry.negate()]),
            );
            builder.add_or_binary(on, off)
        }
        (Some(bits), None) | (None, Some(bits)) => {
            // When one endpoint is impossible, the carry gates the surviving
            // word predicate.
            let data = and_cube_with_arrivals(builder, depths, bits.iter().copied());
            builder.add_and_binary(
                data,
                if zero.is_some() {
                    carry.negate()
                } else {
                    carry
                },
            )
        }
        (None, None) => builder.get_false(),
    }
}

/// Only replace the live output cone when its structural cost improves and
/// its modeled arrival under a unit load per fanout does not increase.
fn choose_replacement(stock: GateFn, proposed: GateFn) -> GateFn {
    use crate::aig::get_summary_stats::get_output_cone_stats;
    use crate::aig::graph_logical_effort::{GraphLogicalEffortOptions, score_graph_logical_effort};

    let stock_stats = get_output_cone_stats(&stock);
    let trial_stats = get_output_cone_stats(&proposed);
    if trial_stats.and_nodes > stock_stats.and_nodes
        || (trial_stats.and_nodes == stock_stats.and_nodes
            && trial_stats.root_depths.iter().max() >= stock_stats.root_depths.iter().max())
    {
        return stock;
    }

    let proposed = dce(&proposed);
    if trial_stats.and_nodes == 0 {
        return proposed;
    }

    // Dead gates have no physical load. Keep the original graph, including
    // its stored metadata, whenever the trial is declined.
    let model = GraphLogicalEffortOptions::default();
    let before = score_graph_logical_effort(&dce(&stock), &model).delay;
    let after = score_graph_logical_effort(&proposed, &model).delay;
    if before.is_finite() && after.is_finite() && after <= before {
        proposed
    } else {
        stock
    }
}

#[cfg(test)]
mod tests;
