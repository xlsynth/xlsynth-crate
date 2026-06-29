// SPDX-License-Identifier: Apache-2.0

//! Formal alias analysis for array reads over contiguous update chains.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use xlsynth_pir::array_alias_rewrite::{
    ArrayAccessPair, ArrayAliasFact, ArrayAliasRelation, ArrayReadRewriteOutcome,
    rewrite_array_reads_with_alias_facts,
};
use xlsynth_pir::ir::{self, NodePayload, NodeRef, Type};

use crate::prover::translate::{get_fn_inputs, ir_to_smt_with_node_terms};
use crate::prover::types::{ProverFn, UfRegistry};
use crate::solver::{BitVec, Response, Solver};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ArrayAliasAnalysisStats {
    pub array_index_count: usize,
    pub read_update_pair_count: usize,
    pub unsupported_depth_pair_count: usize,
    pub cheap_fact_count: usize,
    pub unique_smt_query_count: usize,
    pub cached_smt_query_count: usize,
    pub solver_check_count: usize,
    pub never_alias_count: usize,
    pub always_alias_count: usize,
    pub may_alias_count: usize,
    pub unknown_count: usize,
    pub translation_seconds: f64,
    pub query_seconds: f64,
}

/// Holds deterministic alias facts and statistics for one function.
#[derive(Debug, Clone)]
pub struct ArrayAliasAnalysis {
    pub facts: Vec<ArrayAliasFact>,
    pub stats: ArrayAliasAnalysisStats,
}

/// Holds the formal analysis together with its conservative rewrite result.
#[derive(Debug, Clone)]
pub struct FormalArrayRewriteOutcome {
    pub analysis: ArrayAliasAnalysis,
    pub rewrite: ArrayReadRewriteOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AliasQueryKey {
    read_index_text_ids: Vec<usize>,
    write_index_text_ids: Vec<usize>,
    element_counts: Vec<usize>,
}

/// Returns array dimensions consumed by the given number of indices.
fn dimensions_for_indices(
    function: &ir::Fn,
    array: NodeRef,
    index_count: usize,
) -> Result<Vec<usize>, String> {
    let mut ty = &function.get_node(array).ty;
    let mut dimensions = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        let Type::Array(data) = ty else {
            return Err(format!(
                "array operation operand stopped being an array at type {ty}"
            ));
        };
        if data.element_count == 0 {
            return Err("zero-element arrays are unsupported by array alias analysis".to_string());
        }
        dimensions.push(data.element_count);
        ty = data.element_type.as_ref();
    }
    Ok(dimensions)
}

fn bits_required(value: usize) -> usize {
    if value == 0 {
        1
    } else {
        usize::BITS as usize - value.leading_zeros() as usize
    }
}

fn index_is_inherently_in_bounds(function: &ir::Fn, index: NodeRef, count: usize) -> bool {
    let Type::Bits(width) = function.get_node(index).ty else {
        return false;
    };
    width < usize::BITS as usize && (1usize << width) <= count
}

/// Extracts a bits literal only when its arbitrary-width value fits in usize.
fn literal_as_usize(function: &ir::Fn, node_ref: NodeRef) -> Option<usize> {
    let NodePayload::Literal(value) = &function.get_node(node_ref).payload else {
        return None;
    };
    let bits = value.to_bits().ok()?;
    let bytes = bits.to_le_bytes().ok()?;
    if bytes
        .iter()
        .skip(std::mem::size_of::<usize>())
        .any(|byte| *byte != 0)
    {
        return None;
    }
    let mut result = 0usize;
    for (byte_index, byte) in bytes.iter().take(std::mem::size_of::<usize>()).enumerate() {
        result |= (*byte as usize) << (byte_index * 8);
    }
    Some(result)
}

/// Applies structural, type-width, and literal checks before invoking SMT.
fn classify_cheap(
    function: &ir::Fn,
    read_indices: &[NodeRef],
    write_indices: &[NodeRef],
    element_counts: &[usize],
) -> Option<ArrayAliasRelation> {
    if read_indices.is_empty() {
        return Some(ArrayAliasRelation::AlwaysAliases);
    }
    let mut all_dimensions_always_equal = true;
    for ((&read_index, &write_index), &element_count) in
        read_indices.iter().zip(write_indices).zip(element_counts)
    {
        let read_literal = literal_as_usize(function, read_index);
        let write_literal = literal_as_usize(function, write_index);
        if let Some(write_value) = write_literal
            && write_value >= element_count
        {
            return Some(ArrayAliasRelation::NeverAliases);
        }
        if let (Some(read_value), Some(write_value)) = (read_literal, write_literal) {
            let effective_read = read_value.min(element_count - 1);
            if effective_read != write_value {
                return Some(ArrayAliasRelation::NeverAliases);
            }
            continue;
        }
        if read_index == write_index
            && index_is_inherently_in_bounds(function, write_index, element_count)
        {
            continue;
        }
        all_dimensions_always_equal = false;
    }
    all_dimensions_always_equal.then_some(ArrayAliasRelation::AlwaysAliases)
}

fn in_bounds<S: Solver>(
    solver: &mut S,
    index: &BitVec<S::Term>,
    element_count: usize,
) -> BitVec<S::Term> {
    let index_width = index.get_width();
    if index_width < usize::BITS as usize && (1usize << index_width) <= element_count {
        return solver.one(1);
    }
    let compare_width = index_width.max(bits_required(element_count));
    let extended_index = solver.zero_extend_to(index, compare_width);
    let bound = solver.numerical(compare_width, element_count as u64);
    solver.ult(&extended_index, &bound)
}

/// Builds the exact XLS alias predicate, including read clamping and OOB
/// writes.
fn build_alias_predicate<S: Solver>(
    solver: &mut S,
    function: &ir::Fn,
    node_terms: &BTreeMap<usize, crate::prover::types::IrTypedBitVec<'_, S::Term>>,
    read_indices: &[NodeRef],
    write_indices: &[NodeRef],
    element_counts: &[usize],
) -> Result<BitVec<S::Term>, String> {
    let mut aliases = solver.one(1);
    for ((&read_index, &write_index), &element_count) in
        read_indices.iter().zip(write_indices).zip(element_counts)
    {
        let read_node = function.get_node(read_index);
        let write_node = function.get_node(write_index);
        let read_term = &node_terms
            .get(&read_node.text_id)
            .ok_or_else(|| format!("missing SMT term for read index {}", read_node.text_id))?
            .bitvec;
        let write_term = &node_terms
            .get(&write_node.text_id)
            .ok_or_else(|| format!("missing SMT term for write index {}", write_node.text_id))?
            .bitvec;

        let write_in_bounds = in_bounds(solver, write_term, element_count);
        let read_in_bounds = in_bounds(solver, read_term, element_count);
        let compare_width = read_term
            .get_width()
            .max(write_term.get_width())
            .max(bits_required(element_count - 1));
        let extended_read = solver.zero_extend_to(read_term, compare_width);
        let extended_write = solver.zero_extend_to(write_term, compare_width);
        let last_element = solver.numerical(compare_width, (element_count - 1) as u64);
        let effective_read = solver.ite(&read_in_bounds, &extended_read, &last_element);
        let same_element = solver.eq(&effective_read, &extended_write);
        let aliases_dimension = solver.and(&write_in_bounds, &same_element);
        aliases = solver.and(&aliases, &aliases_dimension);
    }
    Ok(aliases)
}

fn check_predicate<S: Solver>(
    solver: &mut S,
    predicate: &BitVec<S::Term>,
) -> Result<(Response, f64), String> {
    solver.push().map_err(|error| error.to_string())?;
    solver
        .assert(predicate)
        .map_err(|error| error.to_string())?;
    let start = Instant::now();
    let response = solver.check().map_err(|error| error.to_string())?;
    let seconds = start.elapsed().as_secs_f64();
    solver.pop().map_err(|error| error.to_string())?;
    Ok((response, seconds))
}

/// Proves read/update alias relationships with one incremental SMT instance.
pub fn prove_array_index_update_aliases<S: Solver>(
    solver_config: &S::Config,
    package: &ir::Package,
    function: &ir::Fn,
) -> Result<ArrayAliasAnalysis, String> {
    let mut solver = S::new(solver_config).map_err(|error| error.to_string())?;
    let prover_fn = ProverFn::new(function, Some(package));
    let function_inputs = get_fn_inputs(&mut solver, prover_fn, None);
    let uf_map = HashMap::new();
    let uf_registry = UfRegistry {
        ufs: HashMap::new(),
    };
    let translation_start = Instant::now();
    let smt = ir_to_smt_with_node_terms(&mut solver, &function_inputs, &uf_map, &uf_registry);

    let mut stats = ArrayAliasAnalysisStats {
        translation_seconds: translation_start.elapsed().as_secs_f64(),
        ..ArrayAliasAnalysisStats::default()
    };
    let mut facts = Vec::new();
    let mut query_cache: BTreeMap<AliasQueryKey, ArrayAliasRelation> = BTreeMap::new();

    for read_node in &function.nodes {
        let NodePayload::ArrayIndex {
            array,
            indices: read_indices,
            ..
        } = &read_node.payload
        else {
            continue;
        };
        stats.array_index_count += 1;
        let mut current_array = *array;
        let mut chain_depth = 0usize;
        while let NodePayload::ArrayUpdate {
            array: preceding_array,
            indices: write_indices,
            ..
        } = &function.get_node(current_array).payload
        {
            chain_depth += 1;
            stats.read_update_pair_count += 1;
            let update_node = function.get_node(current_array);
            let relation = if read_indices.len() != write_indices.len() {
                stats.unsupported_depth_pair_count += 1;
                ArrayAliasRelation::Unknown
            } else {
                let dimensions =
                    dimensions_for_indices(function, current_array, write_indices.len())?;
                if let Some(relation) =
                    classify_cheap(function, read_indices, write_indices, &dimensions)
                {
                    stats.cheap_fact_count += 1;
                    relation
                } else {
                    let key = AliasQueryKey {
                        read_index_text_ids: read_indices
                            .iter()
                            .map(|index| function.get_node(*index).text_id)
                            .collect(),
                        write_index_text_ids: write_indices
                            .iter()
                            .map(|index| function.get_node(*index).text_id)
                            .collect(),
                        element_counts: dimensions.clone(),
                    };
                    if let Some(relation) = query_cache.get(&key).copied() {
                        stats.cached_smt_query_count += 1;
                        relation
                    } else {
                        stats.unique_smt_query_count += 1;
                        let aliases = build_alias_predicate(
                            &mut solver,
                            function,
                            &smt.node_terms,
                            read_indices,
                            write_indices,
                            &dimensions,
                        )?;
                        let (alias_response, alias_seconds) =
                            check_predicate(&mut solver, &aliases)?;
                        stats.solver_check_count += 1;
                        stats.query_seconds += alias_seconds;
                        let relation = match alias_response {
                            Response::Unsat => ArrayAliasRelation::NeverAliases,
                            Response::Unknown => ArrayAliasRelation::Unknown,
                            Response::Sat => {
                                let does_not_alias = solver.not(&aliases);
                                let (non_alias_response, non_alias_seconds) =
                                    check_predicate(&mut solver, &does_not_alias)?;
                                stats.solver_check_count += 1;
                                stats.query_seconds += non_alias_seconds;
                                match non_alias_response {
                                    Response::Unsat => ArrayAliasRelation::AlwaysAliases,
                                    Response::Sat => ArrayAliasRelation::MayAlias,
                                    Response::Unknown => ArrayAliasRelation::Unknown,
                                }
                            }
                        };
                        query_cache.insert(key, relation);
                        relation
                    }
                }
            };

            match relation {
                ArrayAliasRelation::NeverAliases => stats.never_alias_count += 1,
                ArrayAliasRelation::AlwaysAliases => stats.always_alias_count += 1,
                ArrayAliasRelation::MayAlias => stats.may_alias_count += 1,
                ArrayAliasRelation::Unknown => stats.unknown_count += 1,
            }
            facts.push(ArrayAliasFact {
                pair: ArrayAccessPair {
                    read_text_id: read_node.text_id,
                    update_text_id: update_node.text_id,
                },
                chain_depth,
                relation,
            });
            current_array = *preceding_array;
        }
    }

    Ok(ArrayAliasAnalysis { facts, stats })
}

/// Proves alias facts and applies the conservative solver-independent rewrite.
pub fn prove_and_rewrite_array_reads<S: Solver>(
    solver_config: &S::Config,
    package: &ir::Package,
    function: &ir::Fn,
) -> Result<FormalArrayRewriteOutcome, String> {
    let analysis = prove_array_index_update_aliases::<S>(solver_config, package, function)?;
    let rewrite = rewrite_array_reads_with_alias_facts(function, &analysis.facts)?;
    Ok(FormalArrayRewriteOutcome { analysis, rewrite })
}

#[cfg(all(test, feature = "has-bitwuzla"))]
mod tests {
    use super::*;
    use crate::prover::ir_equiv::prove_ir_fn_equiv;
    use crate::prover::types::{AssertionSemantics, EquivResult};
    use crate::solver::bitwuzla::{Bitwuzla, BitwuzlaOptions};
    use xlsynth_pir::ir_parser::Parser;

    fn parse_package(ir_text: &str) -> ir::Package {
        let mut parser = Parser::new(ir_text);
        parser.parse_and_validate_package().unwrap()
    }

    #[test]
    fn classifies_equal_different_variable_and_oob_aliases() {
        let package = parse_package(
            r#"package test

top fn f(a4: bits[8][4] id=1, a3: bits[8][3] id=2, v: bits[8] id=3, i: bits[2] id=4, j: bits[2] id=5) -> (bits[8], bits[8], bits[8], bits[8], bits[8]) {
  a4: bits[8][4] = param(name=a4, id=1)
  a3: bits[8][3] = param(name=a3, id=2)
  v: bits[8] = param(name=v, id=3)
  i: bits[2] = param(name=i, id=4)
  j: bits[2] = param(name=j, id=5)
  zero: bits[2] = literal(value=0, id=6)
  one: bits[2] = literal(value=1, id=7)
  two: bits[2] = literal(value=2, id=8)
  three: bits[2] = literal(value=3, id=9)
  i_xor_one: bits[2] = xor(i, one, id=10)
  update_i: bits[8][4] = array_update(a4, v, indices=[i], id=11)
  update_i_three: bits[8][3] = array_update(a3, v, indices=[i], id=12)
  update_two: bits[8][3] = array_update(a3, v, indices=[two], id=13)
  update_three: bits[8][3] = array_update(a3, v, indices=[three], id=14)
  read_equal: bits[8] = array_index(update_i, indices=[i], id=15)
  read_equal_or_oob: bits[8] = array_index(update_i_three, indices=[i], id=16)
  read_different: bits[8] = array_index(update_i, indices=[i_xor_one], id=17)
  read_variable: bits[8] = array_index(update_i, indices=[j], id=18)
  read_clamped: bits[8] = array_index(update_two, indices=[i], id=19)
  read_oob_write: bits[8] = array_index(update_three, indices=[zero], id=20)
  ret result: (bits[8], bits[8], bits[8], bits[8], bits[8]) = tuple(read_equal, read_different, read_variable, read_clamped, read_oob_write, id=21)
}
"#,
        );
        let function = package.get_top_fn().unwrap();
        let mut options = BitwuzlaOptions::new();
        options.disable_produce_models();
        let analysis =
            prove_array_index_update_aliases::<Bitwuzla>(&options, &package, function).unwrap();
        let relations: BTreeMap<(usize, usize), ArrayAliasRelation> = analysis
            .facts
            .iter()
            .map(|fact| {
                (
                    (fact.pair.read_text_id, fact.pair.update_text_id),
                    fact.relation,
                )
            })
            .collect();
        assert_eq!(relations[&(15, 11)], ArrayAliasRelation::AlwaysAliases);
        assert_eq!(
            relations[&(16, 12)],
            ArrayAliasRelation::MayAlias,
            "equal raw indices do not alias when the update is out of bounds"
        );
        assert_eq!(relations[&(17, 11)], ArrayAliasRelation::NeverAliases);
        assert_eq!(relations[&(18, 11)], ArrayAliasRelation::MayAlias);
        assert_eq!(relations[&(19, 13)], ArrayAliasRelation::MayAlias);
        assert_eq!(relations[&(20, 14)], ArrayAliasRelation::NeverAliases);
    }

    #[test]
    fn conservative_rewrite_is_formally_equivalent() {
        let package = parse_package(
            r#"package test

top fn f(a: bits[8][4] id=1, v0: bits[8] id=2, v1: bits[8] id=3, i: bits[2] id=4) -> bits[8] {
  a: bits[8][4] = param(name=a, id=1)
  v0: bits[8] = param(name=v0, id=2)
  v1: bits[8] = param(name=v1, id=3)
  i: bits[2] = param(name=i, id=4)
  one: bits[2] = literal(value=1, id=5)
  two: bits[2] = literal(value=2, id=6)
  older_index: bits[2] = xor(i, two, id=7)
  newer_index: bits[2] = xor(i, one, id=8)
  older: bits[8][4] = array_update(a, v0, indices=[older_index], id=9)
  newer: bits[8][4] = array_update(older, v1, indices=[newer_index], id=10)
  ret read: bits[8] = array_index(newer, indices=[i], id=11)
}
"#,
        );
        let original = package.get_top_fn().unwrap();
        let mut options = BitwuzlaOptions::new();
        options.disable_produce_models();
        let outcome =
            prove_and_rewrite_array_reads::<Bitwuzla>(&options, &package, original).unwrap();
        assert_eq!(outcome.rewrite.stats.bypassed_update_layers, 2);

        let lhs = ProverFn::new(original, Some(&package));
        let rhs = ProverFn::new(&outcome.rewrite.rewritten_fn, None);
        let equivalence = prove_ir_fn_equiv::<Bitwuzla>(
            &options,
            &lhs,
            &rhs,
            AssertionSemantics::Ignore,
            None,
            false,
        );
        assert_eq!(equivalence, EquivResult::Proved);
    }

    #[test]
    fn proves_multidimensional_non_alias_when_one_dimension_differs() {
        let package = parse_package(
            r#"package test

top fn f(a: bits[8][4][4] id=1, v: bits[8] id=2, i: bits[2] id=3, j: bits[2] id=4) -> bits[8] {
  a: bits[8][4][4] = param(name=a, id=1)
  v: bits[8] = param(name=v, id=2)
  i: bits[2] = param(name=i, id=3)
  j: bits[2] = param(name=j, id=4)
  one: bits[2] = literal(value=1, id=5)
  different_j: bits[2] = xor(j, one, id=6)
  updated: bits[8][4][4] = array_update(a, v, indices=[i, j], id=7)
  ret read: bits[8] = array_index(updated, indices=[i, different_j], id=8)
}
"#,
        );
        let function = package.get_top_fn().unwrap();
        let mut options = BitwuzlaOptions::new();
        options.disable_produce_models();
        let analysis =
            prove_array_index_update_aliases::<Bitwuzla>(&options, &package, function).unwrap();
        assert_eq!(analysis.facts.len(), 1);
        assert_eq!(analysis.facts[0].relation, ArrayAliasRelation::NeverAliases);
    }
}
