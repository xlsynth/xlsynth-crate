// SPDX-License-Identifier: Apache-2.0

//! Solver-independent alias facts and rewrites for array operations.

use std::collections::BTreeMap;

use crate::ir::{self, NodePayload, NodeRef, Unop};

/// Identifies a read and one update in its contiguous array-update chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArrayAccessPair {
    pub read_text_id: usize,
    pub update_text_id: usize,
}

/// Describes whether an array update can affect a particular array read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayAliasRelation {
    NeverAliases,
    AlwaysAliases,
    MayAlias,
    Unknown,
}

/// Records one proved or unresolved read/update relationship.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayAliasFact {
    pub pair: ArrayAccessPair,
    pub chain_depth: usize,
    pub relation: ArrayAliasRelation,
}

/// Summarizes conservative array read/update rewrites.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArrayReadRewriteStats {
    pub array_index_count: usize,
    pub reads_with_update_operand: usize,
    pub rewritten_reads: usize,
    pub bypassed_update_layers: usize,
    pub always_alias_replacements: usize,
    pub fully_bypassed_update_chains: usize,
    pub stopped_at_may_alias: usize,
    pub stopped_at_unknown: usize,
}

/// Holds a rewritten function and deterministic rewrite statistics.
#[derive(Debug, Clone)]
pub struct ArrayReadRewriteOutcome {
    pub rewritten_fn: ir::Fn,
    pub stats: ArrayReadRewriteStats,
}

#[derive(Debug, Clone, Copy)]
enum RewriteAction {
    BypassTo { array: NodeRef, layers: usize },
    ReplaceWithValue { value: NodeRef, layers: usize },
}

/// Rewrites reads using supplied alias facts without modifying shared updates.
pub fn rewrite_array_reads_with_alias_facts(
    function: &ir::Fn,
    facts: &[ArrayAliasFact],
) -> Result<ArrayReadRewriteOutcome, String> {
    let mut facts_by_pair = BTreeMap::new();
    for fact in facts {
        if let Some(previous) = facts_by_pair.insert(fact.pair, fact.relation)
            && previous != fact.relation
        {
            return Err(format!(
                "conflicting alias facts for read {} and update {}",
                fact.pair.read_text_id, fact.pair.update_text_id
            ));
        }
    }

    let mut stats = ArrayReadRewriteStats::default();
    let mut actions: Vec<(NodeRef, RewriteAction)> = Vec::new();

    for (read_index, read_node) in function.nodes.iter().enumerate() {
        let NodePayload::ArrayIndex {
            array,
            indices: read_indices,
            ..
        } = &read_node.payload
        else {
            continue;
        };
        stats.array_index_count += 1;
        if !matches!(
            function.get_node(*array).payload,
            NodePayload::ArrayUpdate { .. }
        ) {
            continue;
        }
        stats.reads_with_update_operand += 1;

        let read_ref = NodeRef { index: read_index };
        let original_array = *array;
        let mut current_array = original_array;
        let mut bypassed_layers = 0usize;
        let mut action = None;

        while let NodePayload::ArrayUpdate {
            array: preceding_array,
            value,
            indices: write_indices,
            ..
        } = &function.get_node(current_array).payload
        {
            if read_indices.len() != write_indices.len() {
                stats.stopped_at_unknown += 1;
                break;
            }
            let pair = ArrayAccessPair {
                read_text_id: read_node.text_id,
                update_text_id: function.get_node(current_array).text_id,
            };
            match facts_by_pair
                .get(&pair)
                .copied()
                .unwrap_or(ArrayAliasRelation::Unknown)
            {
                ArrayAliasRelation::NeverAliases => {
                    bypassed_layers += 1;
                    current_array = *preceding_array;
                }
                ArrayAliasRelation::AlwaysAliases => {
                    action = Some(RewriteAction::ReplaceWithValue {
                        value: *value,
                        layers: bypassed_layers,
                    });
                    break;
                }
                ArrayAliasRelation::MayAlias => {
                    stats.stopped_at_may_alias += 1;
                    break;
                }
                ArrayAliasRelation::Unknown => {
                    stats.stopped_at_unknown += 1;
                    break;
                }
            }
        }

        if action.is_none() && current_array != original_array {
            if !matches!(
                function.get_node(current_array).payload,
                NodePayload::ArrayUpdate { .. }
            ) {
                stats.fully_bypassed_update_chains += 1;
            }
            action = Some(RewriteAction::BypassTo {
                array: current_array,
                layers: bypassed_layers,
            });
        }
        if let Some(action) = action {
            actions.push((read_ref, action));
        }
    }

    let mut rewritten = function.clone();
    for (read_ref, action) in actions {
        let layers = match action {
            RewriteAction::BypassTo { array, layers } => {
                let NodePayload::ArrayIndex {
                    indices,
                    assumed_in_bounds,
                    ..
                } = rewritten.get_node(read_ref).payload.clone()
                else {
                    return Err(format!(
                        "array read {} changed before its rewrite was applied",
                        rewritten.get_node(read_ref).text_id
                    ));
                };
                rewritten.get_node_mut(read_ref).payload = NodePayload::ArrayIndex {
                    array,
                    indices,
                    assumed_in_bounds,
                };
                layers
            }
            RewriteAction::ReplaceWithValue { value, layers } => {
                let read_ty = rewritten.get_node(read_ref).ty.clone();
                let value_ty = rewritten.get_node(value).ty.clone();
                if read_ty != value_ty {
                    return Err(format!(
                        "always-alias replacement type mismatch for read {}: {} vs {}",
                        rewritten.get_node(read_ref).text_id,
                        read_ty,
                        value_ty
                    ));
                }
                rewritten.get_node_mut(read_ref).payload = NodePayload::Unop(Unop::Identity, value);
                stats.always_alias_replacements += 1;
                layers
            }
        };
        stats.rewritten_reads += 1;
        stats.bypassed_update_layers += layers;
    }

    Ok(ArrayReadRewriteOutcome {
        rewritten_fn: rewritten,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    fn parse_top(ir_text: &str) -> ir::Fn {
        let mut parser = Parser::new(ir_text);
        parser
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone()
    }

    fn find_node_ref(function: &ir::Fn, text_id: usize) -> NodeRef {
        function
            .nodes
            .iter()
            .enumerate()
            .find_map(|(index, node)| (node.text_id == text_id).then_some(NodeRef { index }))
            .unwrap()
    }

    #[test]
    fn bypasses_only_the_proved_read_of_a_shared_update() {
        let function = parse_top(
            r#"package test

top fn f(a: bits[8][4] id=1, v: bits[8] id=2, i: bits[2] id=3, j: bits[2] id=4) -> (bits[8], bits[8]) {
  a: bits[8][4] = param(name=a, id=1)
  v: bits[8] = param(name=v, id=2)
  i: bits[2] = param(name=i, id=3)
  j: bits[2] = param(name=j, id=4)
  updated: bits[8][4] = array_update(a, v, indices=[i], id=5)
  bypassed: bits[8] = array_index(updated, indices=[j], id=6)
  retained: bits[8] = array_index(updated, indices=[i], id=7)
  ret result: (bits[8], bits[8]) = tuple(bypassed, retained, id=8)
}
"#,
        );
        let facts = vec![ArrayAliasFact {
            pair: ArrayAccessPair {
                read_text_id: 6,
                update_text_id: 5,
            },
            chain_depth: 1,
            relation: ArrayAliasRelation::NeverAliases,
        }];
        let outcome = rewrite_array_reads_with_alias_facts(&function, &facts).unwrap();

        let bypassed = outcome
            .rewritten_fn
            .get_node(find_node_ref(&outcome.rewritten_fn, 6));
        let retained = outcome
            .rewritten_fn
            .get_node(find_node_ref(&outcome.rewritten_fn, 7));
        let a_ref = find_node_ref(&outcome.rewritten_fn, 1);
        let updated_ref = find_node_ref(&outcome.rewritten_fn, 5);
        assert!(matches!(
            bypassed.payload,
            NodePayload::ArrayIndex { array, .. } if array == a_ref
        ));
        assert!(matches!(
            retained.payload,
            NodePayload::ArrayIndex { array, .. } if array == updated_ref
        ));
        assert_eq!(outcome.stats.rewritten_reads, 1);
        assert_eq!(outcome.stats.bypassed_update_layers, 1);
        assert_eq!(outcome.stats.fully_bypassed_update_chains, 1);
    }

    #[test]
    fn walks_past_never_aliases_and_stops_at_may_alias() {
        let function = parse_top(
            r#"package test

top fn f(a: bits[8][4] id=1, v0: bits[8] id=2, v1: bits[8] id=3, i: bits[2] id=4, j: bits[2] id=5, k: bits[2] id=6) -> bits[8] {
  a: bits[8][4] = param(name=a, id=1)
  v0: bits[8] = param(name=v0, id=2)
  v1: bits[8] = param(name=v1, id=3)
  i: bits[2] = param(name=i, id=4)
  j: bits[2] = param(name=j, id=5)
  k: bits[2] = param(name=k, id=6)
  older: bits[8][4] = array_update(a, v0, indices=[i], id=7)
  newer: bits[8][4] = array_update(older, v1, indices=[j], id=8)
  ret read: bits[8] = array_index(newer, indices=[k], id=9)
}
"#,
        );
        let facts = vec![
            ArrayAliasFact {
                pair: ArrayAccessPair {
                    read_text_id: 9,
                    update_text_id: 8,
                },
                chain_depth: 1,
                relation: ArrayAliasRelation::NeverAliases,
            },
            ArrayAliasFact {
                pair: ArrayAccessPair {
                    read_text_id: 9,
                    update_text_id: 7,
                },
                chain_depth: 2,
                relation: ArrayAliasRelation::MayAlias,
            },
        ];
        let outcome = rewrite_array_reads_with_alias_facts(&function, &facts).unwrap();
        let read = outcome
            .rewritten_fn
            .get_node(find_node_ref(&outcome.rewritten_fn, 9));
        let older_ref = find_node_ref(&outcome.rewritten_fn, 7);
        assert!(matches!(
            read.payload,
            NodePayload::ArrayIndex { array, .. } if array == older_ref
        ));
        assert_eq!(outcome.stats.bypassed_update_layers, 1);
        assert_eq!(outcome.stats.stopped_at_may_alias, 1);
    }

    #[test]
    fn replaces_an_always_aliasing_read_with_the_update_value() {
        let function = parse_top(
            r#"package test

top fn f(a: bits[8][4] id=1, v: bits[8] id=2, i: bits[2] id=3) -> bits[8] {
  a: bits[8][4] = param(name=a, id=1)
  v: bits[8] = param(name=v, id=2)
  i: bits[2] = param(name=i, id=3)
  updated: bits[8][4] = array_update(a, v, indices=[i], id=4)
  ret read: bits[8] = array_index(updated, indices=[i], id=5)
}
"#,
        );
        let facts = vec![ArrayAliasFact {
            pair: ArrayAccessPair {
                read_text_id: 5,
                update_text_id: 4,
            },
            chain_depth: 1,
            relation: ArrayAliasRelation::AlwaysAliases,
        }];
        let outcome = rewrite_array_reads_with_alias_facts(&function, &facts).unwrap();
        let read = outcome
            .rewritten_fn
            .get_node(find_node_ref(&outcome.rewritten_fn, 5));
        let value_ref = find_node_ref(&outcome.rewritten_fn, 2);
        assert!(matches!(
            read.payload,
            NodePayload::Unop(Unop::Identity, value) if value == value_ref
        ));
        assert_eq!(outcome.stats.always_alias_replacements, 1);
    }
}
