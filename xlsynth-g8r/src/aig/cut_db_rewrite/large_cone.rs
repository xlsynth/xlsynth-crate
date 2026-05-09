// SPDX-License-Identifier: Apache-2.0

//! Bounded large-cone construction for cut-db rewriting.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::aig::dynamic_depth::DynamicDepthState;
use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};

use super::{
    ConstructedLargeConeCandidates, LargeConeCandidate, LargeConeCandidateStats, Replacement,
    ReplacementImpl, collect_internal_and_nodes_under_cut, collect_mffc_nodes_under_cut,
    live_forward_depth,
};

const LARGE_CONE_MAX_LEAVES: usize = 8;
const LARGE_TRUTH_TABLE_WORDS: usize = 4;
const LARGE_CONE_MAX_INTERNAL_NODES: usize = 128;
const LARGE_CONE_FANOUT_STOP: usize = 2;
const SOP_MAX_CUBES_DURING_MERGE: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::aig) struct SopCube {
    pub(in crate::aig) care_mask: u16,
    pub(in crate::aig) value_mask: u16,
}

impl SopCube {
    pub(super) fn literal_count(self) -> usize {
        self.care_mask.count_ones() as usize
    }

    fn covers_cube(self, other: SopCube) -> bool {
        (self.care_mask & !other.care_mask) == 0
            && ((self.value_mask ^ other.value_mask) & self.care_mask) == 0
    }
}

fn ternary_cube_space(nvars: usize) -> usize {
    let mut out = 1usize;
    for _ in 0..nvars {
        out = out
            .checked_mul(3)
            .expect("ternary cube space should fit in usize");
    }
    out
}

fn ternary_cube_index(cube: SopCube, nvars: usize) -> usize {
    let mut index = 0usize;
    let mut place = 1usize;
    for var in 0..nvars {
        let bit = 1u16 << var;
        let digit = if (cube.care_mask & bit) == 0 {
            0usize
        } else if (cube.value_mask & bit) == 0 {
            1usize
        } else {
            2usize
        };
        index += digit * place;
        place *= 3;
    }
    index
}

#[derive(Debug, Clone)]
struct DenseCubeSet {
    nvars: usize,
    present: Vec<bool>,
    cubes: Vec<SopCube>,
}

impl DenseCubeSet {
    /// Creates a deterministic local set for cubes over `nvars` ternary
    /// literals.
    fn new(nvars: usize) -> Self {
        debug_assert!(nvars <= LARGE_CONE_MAX_LEAVES);
        Self {
            nvars,
            present: vec![false; ternary_cube_space(nvars)],
            cubes: Vec::new(),
        }
    }

    fn insert(&mut self, cube: SopCube) -> bool {
        let index = ternary_cube_index(cube, self.nvars);
        if self.present[index] {
            return false;
        }
        self.present[index] = true;
        self.cubes.push(cube);
        true
    }

    fn contains(&self, cube: SopCube) -> bool {
        self.present[ternary_cube_index(cube, self.nvars)]
    }

    fn extend(&mut self, other: DenseCubeSet) {
        debug_assert_eq!(self.nvars, other.nvars);
        for cube in other.cubes {
            self.insert(cube);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &SopCube> {
        self.cubes.iter()
    }

    fn len(&self) -> usize {
        self.cubes.len()
    }

    fn into_sorted_vec(mut self) -> Vec<SopCube> {
        self.cubes.sort();
        self.cubes
    }
}

#[derive(Debug, Clone)]
pub(in crate::aig) struct SopReplacement {
    pub(in crate::aig) nvars: usize,
    pub(in crate::aig) cubes: Vec<SopCube>,
    pub(in crate::aig) factored: FactoredExpr,
    pub(in crate::aig) output_negated: bool,
    pub(in crate::aig) kind: SopVariantKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::aig) enum SopVariantKind {
    Flat,
    ArrivalBalanced,
    Factored,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(in crate::aig) struct FactoredExprId(pub(in crate::aig) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(in crate::aig) enum FactoredExprNode {
    Const(bool),
    Lit {
        var: usize,
        negated: bool,
    },
    And {
        lhs: FactoredExprId,
        rhs: FactoredExprId,
    },
    Or {
        lhs: FactoredExprId,
        rhs: FactoredExprId,
    },
}

impl FactoredExprNode {
    fn children(self) -> Option<(FactoredExprId, FactoredExprId)> {
        match self {
            FactoredExprNode::And { lhs, rhs } | FactoredExprNode::Or { lhs, rhs } => {
                Some((lhs, rhs))
            }
            FactoredExprNode::Const(_) | FactoredExprNode::Lit { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::aig) struct FactoredExpr {
    nodes: Vec<FactoredExprNode>,
    root: FactoredExprId,
}

impl FactoredExpr {
    pub(in crate::aig) fn nodes(&self) -> &[FactoredExprNode] {
        &self.nodes
    }

    pub(in crate::aig) fn root(&self) -> FactoredExprId {
        self.root
    }
}

fn estimate_factored_and_count(expr: &FactoredExpr) -> usize {
    expr.nodes
        .iter()
        .filter(|node| {
            matches!(
                node,
                FactoredExprNode::And { .. } | FactoredExprNode::Or { .. }
            )
        })
        .count()
}

fn sop_depth_from_depths(leaf_ops: &[AigOperand], sop: &SopReplacement, depths: &[usize]) -> usize {
    factored_depth_from_depths(leaf_ops, &sop.factored, depths)
}

fn factored_depth_from_depths(
    leaf_ops: &[AigOperand],
    expr: &FactoredExpr,
    depths: &[usize],
) -> usize {
    let mut node_depths = vec![0usize; expr.nodes.len()];
    for (id, node) in expr.nodes.iter().copied().enumerate() {
        let depth = match node {
            FactoredExprNode::Const(_) => 0,
            FactoredExprNode::Lit { var, .. } => depths[leaf_ops[var].node.id],
            FactoredExprNode::And { lhs, rhs } | FactoredExprNode::Or { lhs, rhs } => {
                node_depths[lhs.0].max(node_depths[rhs.0]) + 1
            }
        };
        node_depths[id] = depth;
    }
    node_depths[expr.root.0]
}

fn factored_node_depth_from_inputs(
    node: FactoredExprNode,
    leaf_ops: &[AigOperand],
    structural_hash_state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
    node_depths: &[usize],
) -> usize {
    match node {
        FactoredExprNode::Const(_) => 0,
        FactoredExprNode::Lit { var, .. } => {
            live_forward_depth(depth_state, structural_hash_state, leaf_ops[var].node)
        }
        FactoredExprNode::And { lhs, rhs } | FactoredExprNode::Or { lhs, rhs } => {
            node_depths[lhs.0].max(node_depths[rhs.0]) + 1
        }
    }
}

pub(super) fn sop_depth_from_inputs(
    leaf_ops: &[AigOperand],
    sop: &SopReplacement,
    structural_hash_state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
) -> usize {
    factored_depth_from_inputs(leaf_ops, &sop.factored, structural_hash_state, depth_state)
}

fn factored_depth_from_inputs(
    leaf_ops: &[AigOperand],
    expr: &FactoredExpr,
    structural_hash_state: &DynamicStructuralHash,
    depth_state: &DynamicDepthState,
) -> usize {
    let mut node_depths = vec![0usize; expr.nodes.len()];
    for (id, node) in expr.nodes.iter().copied().enumerate() {
        node_depths[id] = factored_node_depth_from_inputs(
            node,
            leaf_ops,
            structural_hash_state,
            depth_state,
            &node_depths,
        );
    }
    node_depths[expr.root.0]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct LargeTruthTable {
    nvars: usize,
    words: [u64; LARGE_TRUTH_TABLE_WORDS],
}

impl LargeTruthTable {
    fn assert_supported_nvars(nvars: usize) {
        assert!(
            nvars <= LARGE_CONE_MAX_LEAVES,
            "large-cone truth table got {nvars} variables, exceeding max {LARGE_CONE_MAX_LEAVES}"
        );
        assert!(
            nvars <= 8,
            "LargeTruthTable is specialized to at most 8 variables"
        );
    }

    pub(super) fn bit_count(nvars: usize) -> usize {
        Self::assert_supported_nvars(nvars);
        1usize << nvars
    }

    fn word_count(nvars: usize) -> usize {
        Self::assert_supported_nvars(nvars);
        ((Self::bit_count(nvars) + 63) / 64).max(1)
    }

    fn unused_high_mask(nvars: usize) -> u64 {
        let used_in_last = Self::bit_count(nvars) % 64;
        if used_in_last == 0 {
            u64::MAX
        } else {
            (1u64 << used_in_last) - 1
        }
    }

    pub(super) fn const0(nvars: usize) -> Self {
        Self::assert_supported_nvars(nvars);
        Self {
            nvars,
            words: [0; LARGE_TRUTH_TABLE_WORDS],
        }
    }

    pub(super) fn const1(nvars: usize) -> Self {
        Self::assert_supported_nvars(nvars);
        let mut out = Self {
            nvars,
            words: [u64::MAX; LARGE_TRUTH_TABLE_WORDS],
        };
        out.mask_unused_bits();
        out
    }

    fn var(nvars: usize, index: usize) -> Self {
        let mut out = Self::const0(nvars);
        for assignment in 0..Self::bit_count(nvars) {
            if ((assignment >> index) & 1) != 0 {
                out.set_bit(assignment, true);
            }
        }
        out
    }

    fn mask_unused_bits(&mut self) {
        let word_count = Self::word_count(self.nvars);
        self.words[word_count - 1] &= Self::unused_high_mask(self.nvars);
        for word in &mut self.words[word_count..] {
            *word = 0;
        }
    }

    fn active_words(&self) -> &[u64] {
        &self.words[..Self::word_count(self.nvars)]
    }

    pub(super) fn get_bit(&self, assignment: usize) -> bool {
        debug_assert!(assignment < Self::bit_count(self.nvars));
        ((self.words[assignment / 64] >> (assignment % 64)) & 1) != 0
    }

    pub(super) fn set_bit(&mut self, assignment: usize, value: bool) {
        debug_assert!(assignment < Self::bit_count(self.nvars));
        let word = &mut self.words[assignment / 64];
        let mask = 1u64 << (assignment % 64);
        if value {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }

    fn not(&self) -> Self {
        let mut out = Self {
            nvars: self.nvars,
            words: [0; LARGE_TRUTH_TABLE_WORDS],
        };
        for (out_word, word) in out.words.iter_mut().zip(self.active_words().iter()) {
            *out_word = !*word;
        }
        out.mask_unused_bits();
        out
    }

    fn and(&self, other: &Self) -> Self {
        debug_assert_eq!(self.nvars, other.nvars);
        let mut out = Self {
            nvars: self.nvars,
            words: [0; LARGE_TRUTH_TABLE_WORDS],
        };
        for i in 0..Self::word_count(self.nvars) {
            out.words[i] = self.words[i] & other.words[i];
        }
        out
    }

    fn or(&self, other: &Self) -> Self {
        debug_assert_eq!(self.nvars, other.nvars);
        let mut out = Self {
            nvars: self.nvars,
            words: [0; LARGE_TRUTH_TABLE_WORDS],
        };
        for i in 0..Self::word_count(self.nvars) {
            out.words[i] = self.words[i] | other.words[i];
        }
        out
    }

    fn sharp(&self, other: &Self) -> Self {
        debug_assert_eq!(self.nvars, other.nvars);
        let mut out = Self {
            nvars: self.nvars,
            words: [0; LARGE_TRUTH_TABLE_WORDS],
        };
        for i in 0..Self::word_count(self.nvars) {
            out.words[i] = self.words[i] & !other.words[i];
        }
        out
    }

    fn cofactor_fixed(&self, var: usize, value: bool) -> Self {
        debug_assert!(var < self.nvars);
        let mut out = Self::const0(self.nvars);
        let bit = 1usize << var;
        for assignment in 0..Self::bit_count(self.nvars) {
            let source = if value {
                assignment | bit
            } else {
                assignment & !bit
            };
            out.set_bit(assignment, self.get_bit(source));
        }
        out
    }

    fn depends_on_var(&self, var: usize) -> bool {
        debug_assert!(var < self.nvars);
        let bit = 1usize << var;
        for assignment in 0..Self::bit_count(self.nvars) {
            if (assignment & bit) != 0 {
                continue;
            }
            if self.get_bit(assignment) != self.get_bit(assignment | bit) {
                return true;
            }
        }
        false
    }

    fn topmost_support_var_with(&self, other: &Self) -> Option<usize> {
        debug_assert_eq!(self.nvars, other.nvars);
        (0..self.nvars)
            .rev()
            .find(|var| self.depends_on_var(*var) || other.depends_on_var(*var))
    }

    fn count_ones(&self) -> usize {
        self.active_words()
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    fn is_const0(&self) -> bool {
        self.active_words().iter().all(|word| *word == 0)
    }

    fn is_const1(&self) -> bool {
        self.count_ones() == Self::bit_count(self.nvars)
    }
}

#[derive(Debug)]
struct CubeCoverCache {
    nvars: usize,
    assignment_count: usize,
    word_count: usize,
    covers: Vec<Option<[u64; LARGE_TRUTH_TABLE_WORDS]>>,
}

impl CubeCoverCache {
    /// Creates a per-SOP-construction cache of cube coverage bitsets.
    fn new(nvars: usize) -> Self {
        Self {
            nvars,
            assignment_count: LargeTruthTable::bit_count(nvars),
            word_count: LargeTruthTable::word_count(nvars),
            covers: vec![None; ternary_cube_space(nvars)],
        }
    }

    fn build_cover(&self, cube: SopCube) -> [u64; LARGE_TRUTH_TABLE_WORDS] {
        let mut words = [0; LARGE_TRUTH_TABLE_WORDS];
        let full_mask = if self.nvars == 16 {
            u16::MAX
        } else {
            (1u16 << self.nvars) - 1
        };
        let fixed = cube.value_mask & cube.care_mask;
        let free_mask = full_mask & !cube.care_mask;
        let mut free_bits = free_mask;
        loop {
            let assignment = (fixed | free_bits) as usize;
            debug_assert!(assignment < self.assignment_count);
            words[assignment / 64] |= 1u64 << (assignment % 64);
            if free_bits == 0 {
                break;
            }
            free_bits = free_bits.wrapping_sub(1) & free_mask;
        }
        words
    }

    fn cover(&mut self, cube: SopCube) -> &[u64] {
        let index = ternary_cube_index(cube, self.nvars);
        if self.covers[index].is_none() {
            let cover = self.build_cover(cube);
            self.covers[index] = Some(cover);
        }
        &self.covers[index]
            .as_ref()
            .expect("cube cover should be cached")[..self.word_count]
    }

    fn is_implicant(&mut self, cube: SopCube, tt: &LargeTruthTable) -> bool {
        debug_assert_eq!(tt.nvars, self.nvars);
        self.cover(cube)
            .iter()
            .zip(tt.active_words().iter())
            .all(|(cover, target)| (*cover & !*target) == 0)
    }

    fn covered_count(&mut self, cube: SopCube, uncovered: &[u64]) -> usize {
        self.cover(cube)
            .iter()
            .zip(uncovered.iter())
            .map(|(cover, uncovered)| (*cover & *uncovered).count_ones() as usize)
            .sum()
    }

    fn clear_covered(&mut self, cube: SopCube, uncovered: &mut [u64]) {
        for (uncovered, cover) in uncovered.iter_mut().zip(self.cover(cube).iter()) {
            *uncovered &= !*cover;
        }
    }

    fn cover_matches_target(&mut self, cubes: &[SopCube], target: &LargeTruthTable) -> bool {
        debug_assert_eq!(target.nvars, self.nvars);
        let mut covered = [0u64; LARGE_TRUTH_TABLE_WORDS];
        for cube in cubes {
            for (covered_word, cube_word) in covered.iter_mut().zip(self.cover(*cube).iter()) {
                *covered_word |= *cube_word;
            }
        }
        &covered[..self.word_count] == target.active_words()
    }
}

fn prune_subsumed_cubes(cubes: &[SopCube]) -> Vec<SopCube> {
    let mut out = Vec::new();
    for cube in cubes {
        let subsumed = cubes
            .iter()
            .any(|other| other != cube && other.covers_cube(*cube));
        if !subsumed {
            out.push(*cube);
        }
    }
    out
}

fn greedy_cover_from_cubes(
    cubes: &[SopCube],
    target: &LargeTruthTable,
    cover_cache: &mut CubeCoverCache,
) -> Vec<SopCube> {
    let mut uncovered = target.words;
    let mut selected = Vec::new();

    while uncovered.iter().any(|word| *word != 0) {
        let mut best: Option<(usize, SopCube)> = None;
        for cube in cubes {
            let covered_count = cover_cache.covered_count(*cube, &uncovered);
            if covered_count == 0 {
                continue;
            }
            match best {
                None => best = Some((covered_count, *cube)),
                Some((best_count, best_cube)) => {
                    if covered_count > best_count
                        || (covered_count == best_count
                            && (cube.literal_count(), *cube)
                                < (best_cube.literal_count(), best_cube))
                    {
                        best = Some((covered_count, *cube));
                    }
                }
            }
        }

        let Some((_, cube)) = best else {
            break;
        };
        cover_cache.clear_covered(cube, &mut uncovered);
        selected.push(cube);
    }

    selected.sort();
    selected
}

fn prune_redundant_cover(
    mut cubes: Vec<SopCube>,
    target: &LargeTruthTable,
    cover_cache: &mut CubeCoverCache,
) -> Vec<SopCube> {
    let mut index = 0;
    while index < cubes.len() {
        let cube = cubes.remove(index);
        if cover_cache.cover_matches_target(&cubes, target) {
            continue;
        }
        cubes.insert(index, cube);
        index += 1;
    }
    cubes
}

fn merge_adjacent_implicant_cubes(
    cubes: &DenseCubeSet,
    target: &LargeTruthTable,
    cover_cache: &mut CubeCoverCache,
) -> DenseCubeSet {
    let mut newly_merged = DenseCubeSet::new(cubes.nvars);
    for cube in cubes.iter() {
        let mut cared_bits = cube.care_mask;
        while cared_bits != 0 {
            let bit = 1u16 << cared_bits.trailing_zeros();
            cared_bits &= !bit;

            let neighbor = SopCube {
                care_mask: cube.care_mask,
                value_mask: (cube.value_mask ^ bit) & cube.care_mask,
            };
            if *cube >= neighbor || !cubes.contains(neighbor) {
                continue;
            }

            let care_mask = cube.care_mask & !bit;
            let merged = SopCube {
                care_mask,
                value_mask: cube.value_mask & care_mask,
            };
            debug_assert!(cover_cache.is_implicant(merged, target));
            newly_merged.insert(merged);
        }
    }
    newly_merged
}

#[derive(Debug, Clone)]
struct IsopResult {
    cubes: Vec<SopCube>,
    realized: LargeTruthTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct IsopKey {
    on: LargeTruthTable,
    on_dc: LargeTruthTable,
}

type IsopMemo = HashMap<IsopKey, Option<IsopResult>>;

fn add_literal_to_cube(cube: SopCube, var: usize, negated: bool) -> SopCube {
    let bit = 1u16 << var;
    let value_mask = if negated {
        cube.value_mask & !bit
    } else {
        cube.value_mask | bit
    };
    SopCube {
        care_mask: cube.care_mask | bit,
        value_mask,
    }
}

fn truth_from_cofactored_results(
    low: &LargeTruthTable,
    high: &LargeTruthTable,
    var: usize,
) -> LargeTruthTable {
    debug_assert_eq!(low.nvars, high.nvars);
    let var_tt = LargeTruthTable::var(low.nvars, var);
    let not_var_tt = var_tt.not();
    low.and(&not_var_tt).or(&high.and(&var_tt))
}

/// Computes an ISOP cover using the same recursive decomposition shape as ABC's
/// `Kit_TruthIsop`, without external don't-cares at the top level.
fn isop_rec(
    on: LargeTruthTable,
    on_dc: LargeTruthTable,
    memo: &mut IsopMemo,
) -> Option<IsopResult> {
    debug_assert_eq!(on.nvars, on_dc.nvars);
    debug_assert!(on.sharp(&on_dc).is_const0());

    let key = IsopKey { on, on_dc };
    if let Some(result) = memo.get(&key) {
        return result.clone();
    }

    let result = isop_rec_uncached(on, on_dc, memo);
    memo.insert(key, result.clone());
    result
}

fn isop_rec_uncached(
    on: LargeTruthTable,
    on_dc: LargeTruthTable,
    memo: &mut IsopMemo,
) -> Option<IsopResult> {
    if on.is_const0() {
        return Some(IsopResult {
            cubes: Vec::new(),
            realized: LargeTruthTable::const0(on.nvars),
        });
    }
    if on_dc.is_const1() {
        return Some(IsopResult {
            cubes: vec![SopCube {
                care_mask: 0,
                value_mask: 0,
            }],
            realized: LargeTruthTable::const1(on.nvars),
        });
    }

    let var = on.topmost_support_var_with(&on_dc)?;
    let on0 = on.cofactor_fixed(var, false);
    let on1 = on.cofactor_fixed(var, true);
    let on_dc0 = on_dc.cofactor_fixed(var, false);
    let on_dc1 = on_dc.cofactor_fixed(var, true);

    let res0 = isop_rec(on0.sharp(&on_dc1), on_dc0, memo)?;
    let res1 = isop_rec(on1.sharp(&on_dc0), on_dc1, memo)?;
    let res2_on = on0.sharp(&res0.realized).or(&on1.sharp(&res1.realized));
    let res2_dc = on_dc0.and(&on_dc1);
    let res2 = isop_rec(res2_on, res2_dc, memo)?;

    let cube_count = res0.cubes.len() + res1.cubes.len() + res2.cubes.len();
    if cube_count > SOP_MAX_CUBES_DURING_MERGE {
        return None;
    }

    let mut cubes = Vec::with_capacity(cube_count);
    cubes.extend(
        res0.cubes
            .iter()
            .copied()
            .map(|cube| add_literal_to_cube(cube, var, true)),
    );
    cubes.extend(
        res1.cubes
            .iter()
            .copied()
            .map(|cube| add_literal_to_cube(cube, var, false)),
    );
    cubes.extend(res2.cubes.iter().copied());

    let low = res0.realized.or(&res2.realized);
    let high = res1.realized.or(&res2.realized);
    Some(IsopResult {
        cubes,
        realized: truth_from_cofactored_results(&low, &high, var),
    })
}

fn derive_isop_cover_for_target(target: &LargeTruthTable) -> Option<Vec<SopCube>> {
    let mut memo = IsopMemo::new();
    let result = isop_rec(*target, *target, &mut memo)?;
    let cubes = normalize_sop_cubes(&result.cubes);
    if cubes.len() > SOP_MAX_CUBES_DURING_MERGE {
        return None;
    }
    Some(cubes)
}

#[derive(Debug)]
struct FactoredExprBuilder {
    nodes: Vec<FactoredExprNode>,
    const_ids: [Option<FactoredExprId>; 2],
    literal_ids: [[Option<FactoredExprId>; 2]; 16],
}

impl Default for FactoredExprBuilder {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            const_ids: [None; 2],
            literal_ids: [[None; 2]; 16],
        }
    }
}

impl FactoredExprBuilder {
    fn add_node(&mut self, node: FactoredExprNode) -> FactoredExprId {
        let id = FactoredExprId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    fn const_expr(&mut self, value: bool) -> FactoredExprId {
        let index = usize::from(value);
        if let Some(id) = self.const_ids[index] {
            return id;
        }
        let id = self.add_node(FactoredExprNode::Const(value));
        self.const_ids[index] = Some(id);
        id
    }

    fn literal_expr(&mut self, var: usize, negated: bool) -> FactoredExprId {
        debug_assert!(var < self.literal_ids.len());
        let negated_index = usize::from(negated);
        if let Some(id) = self.literal_ids[var][negated_index] {
            return id;
        }
        let id = self.add_node(FactoredExprNode::Lit { var, negated });
        self.literal_ids[var][negated_index] = Some(id);
        id
    }

    fn expr_and(&mut self, lhs: FactoredExprId, rhs: FactoredExprId) -> FactoredExprId {
        match (self.nodes[lhs.0], self.nodes[rhs.0]) {
            (FactoredExprNode::Const(false), _) | (_, FactoredExprNode::Const(false)) => {
                self.const_expr(false)
            }
            (FactoredExprNode::Const(true), _) => rhs,
            (_, FactoredExprNode::Const(true)) => lhs,
            _ if lhs == rhs => lhs,
            _ => {
                let (lhs, rhs) = if rhs < lhs { (rhs, lhs) } else { (lhs, rhs) };
                self.add_node(FactoredExprNode::And { lhs, rhs })
            }
        }
    }

    fn expr_or(&mut self, lhs: FactoredExprId, rhs: FactoredExprId) -> FactoredExprId {
        match (self.nodes[lhs.0], self.nodes[rhs.0]) {
            (FactoredExprNode::Const(true), _) | (_, FactoredExprNode::Const(true)) => {
                self.const_expr(true)
            }
            (FactoredExprNode::Const(false), _) => rhs,
            (_, FactoredExprNode::Const(false)) => lhs,
            _ if lhs == rhs => lhs,
            _ => {
                let (lhs, rhs) = if rhs < lhs { (rhs, lhs) } else { (lhs, rhs) };
                self.add_node(FactoredExprNode::Or { lhs, rhs })
            }
        }
    }

    fn finish(self, root: FactoredExprId) -> FactoredExpr {
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if reachable[id.0] {
                continue;
            }
            reachable[id.0] = true;
            if let Some((lhs, rhs)) = self.nodes[id.0].children() {
                stack.push(lhs);
                stack.push(rhs);
            }
        }

        let mut old_to_new = vec![None; self.nodes.len()];
        let mut nodes = Vec::with_capacity(reachable.iter().filter(|live| **live).count());
        for (old_index, node) in self.nodes.iter().copied().enumerate() {
            if !reachable[old_index] {
                continue;
            }
            let remap = |id: FactoredExprId, old_to_new: &[Option<FactoredExprId>]| {
                old_to_new[id.0].expect("child expression node should have been remapped")
            };
            let node = match node {
                FactoredExprNode::Const(value) => FactoredExprNode::Const(value),
                FactoredExprNode::Lit { var, negated } => FactoredExprNode::Lit { var, negated },
                FactoredExprNode::And { lhs, rhs } => FactoredExprNode::And {
                    lhs: remap(lhs, &old_to_new),
                    rhs: remap(rhs, &old_to_new),
                },
                FactoredExprNode::Or { lhs, rhs } => FactoredExprNode::Or {
                    lhs: remap(lhs, &old_to_new),
                    rhs: remap(rhs, &old_to_new),
                },
            };
            let new_id = FactoredExprId(nodes.len());
            nodes.push(node);
            old_to_new[old_index] = Some(new_id);
        }

        FactoredExpr {
            nodes,
            root: old_to_new[root.0].expect("root expression node should have been remapped"),
        }
    }
}

fn expr_and_balanced(builder: &mut FactoredExprBuilder, args: &[FactoredExprId]) -> FactoredExprId {
    match args.len() {
        0 => builder.const_expr(true),
        1 => args[0],
        _ => {
            let (lhs, rhs) = args.split_at(args.len() / 2);
            let lhs = expr_and_balanced(builder, lhs);
            let rhs = expr_and_balanced(builder, rhs);
            builder.expr_and(lhs, rhs)
        }
    }
}

fn expr_or_balanced(builder: &mut FactoredExprBuilder, args: &[FactoredExprId]) -> FactoredExprId {
    match args.len() {
        0 => builder.const_expr(false),
        1 => args[0],
        _ => {
            let (lhs, rhs) = args.split_at(args.len() / 2);
            let lhs = expr_or_balanced(builder, lhs);
            let rhs = expr_or_balanced(builder, rhs);
            builder.expr_or(lhs, rhs)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum FactoredTreeOp {
    And,
    Or,
}

fn expr_combine(
    builder: &mut FactoredExprBuilder,
    op: FactoredTreeOp,
    lhs: FactoredExprId,
    rhs: FactoredExprId,
) -> FactoredExprId {
    match op {
        FactoredTreeOp::And => builder.expr_and(lhs, rhs),
        FactoredTreeOp::Or => builder.expr_or(lhs, rhs),
    }
}

fn expr_identity(builder: &mut FactoredExprBuilder, op: FactoredTreeOp) -> FactoredExprId {
    match op {
        FactoredTreeOp::And => builder.const_expr(true),
        FactoredTreeOp::Or => builder.const_expr(false),
    }
}

fn expr_arrival_balanced(
    builder: &mut FactoredExprBuilder,
    mut args: Vec<(FactoredExprId, usize)>,
    op: FactoredTreeOp,
) -> (FactoredExprId, usize) {
    if args.is_empty() {
        return (expr_identity(builder, op), 0);
    }

    while args.len() > 1 {
        args.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let (lhs, lhs_depth) = args.remove(0);
        let (rhs, rhs_depth) = args.remove(0);
        let combined = expr_combine(builder, op, lhs, rhs);
        let combined_depth = if combined == lhs {
            lhs_depth
        } else if combined == rhs {
            rhs_depth
        } else if matches!(builder.nodes[combined.0], FactoredExprNode::Const(_)) {
            0
        } else {
            lhs_depth.max(rhs_depth) + 1
        };
        args.push((combined, combined_depth));
    }

    args.pop()
        .expect("arrival-balanced expression should have one result")
}

fn literal_in_cube(cube: SopCube, var: usize) -> Option<bool> {
    let bit = 1u16 << var;
    if (cube.care_mask & bit) == 0 {
        None
    } else {
        Some((cube.value_mask & bit) == 0)
    }
}

fn remove_literal_from_cube(cube: SopCube, var: usize) -> SopCube {
    let bit = 1u16 << var;
    SopCube {
        care_mask: cube.care_mask & !bit,
        value_mask: cube.value_mask & !bit,
    }
}

fn remove_literals_from_cube(cube: SopCube, literals: &[(usize, bool)]) -> SopCube {
    literals.iter().fold(cube, |accum, (var, _negated)| {
        remove_literal_from_cube(accum, *var)
    })
}

fn normalize_sop_cubes(cubes: &[SopCube]) -> Vec<SopCube> {
    let set: BTreeSet<SopCube> = cubes.iter().copied().collect();
    let unique: Vec<SopCube> = set.into_iter().collect();
    let mut cubes = prune_subsumed_cubes(&unique);
    cubes.sort();
    cubes
}

fn flat_sop_expr(cubes: &[SopCube], nvars: usize) -> FactoredExpr {
    let mut builder = FactoredExprBuilder::default();
    let mut terms = Vec::with_capacity(cubes.len());
    for cube in cubes {
        let mut literals = Vec::new();
        for var in 0..nvars {
            if let Some(negated) = literal_in_cube(*cube, var) {
                literals.push(builder.literal_expr(var, negated));
            }
        }
        terms.push(expr_and_balanced(&mut builder, &literals));
    }
    let root = expr_or_balanced(&mut builder, &terms);
    builder.finish(root)
}

fn arrival_balanced_sop_expr(
    cubes: &[SopCube],
    nvars: usize,
    leaf_depths: &[usize],
) -> FactoredExpr {
    debug_assert_eq!(leaf_depths.len(), nvars);
    let mut builder = FactoredExprBuilder::default();
    let mut terms = Vec::with_capacity(cubes.len());
    for cube in cubes {
        let mut literals = Vec::new();
        for var in 0..nvars {
            if let Some(negated) = literal_in_cube(*cube, var) {
                literals.push((builder.literal_expr(var, negated), leaf_depths[var]));
            }
        }
        terms.push(expr_arrival_balanced(
            &mut builder,
            literals,
            FactoredTreeOp::And,
        ));
    }
    let root = expr_arrival_balanced(&mut builder, terms, FactoredTreeOp::Or).0;
    builder.finish(root)
}

fn common_literals(cubes: &[SopCube], nvars: usize) -> Vec<(usize, bool)> {
    if cubes.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    'vars: for var in 0..nvars {
        let Some(first) = literal_in_cube(cubes[0], var) else {
            continue;
        };
        for cube in &cubes[1..] {
            if literal_in_cube(*cube, var) != Some(first) {
                continue 'vars;
            }
        }
        out.push((var, first));
    }
    out
}

fn common_cube(lhs: SopCube, rhs: SopCube) -> SopCube {
    let same_value = !(lhs.value_mask ^ rhs.value_mask);
    let care_mask = lhs.care_mask & rhs.care_mask & same_value;
    SopCube {
        care_mask,
        value_mask: lhs.value_mask & care_mask,
    }
}

fn cube_product_expr(
    builder: &mut FactoredExprBuilder,
    cube: SopCube,
    nvars: usize,
) -> FactoredExprId {
    let mut literals = Vec::new();
    for var in 0..nvars {
        if let Some(negated) = literal_in_cube(cube, var) {
            literals.push(builder.literal_expr(var, negated));
        }
    }
    expr_and_balanced(builder, &literals)
}

fn cube_contains_common(cube: SopCube, common: SopCube) -> bool {
    common.covers_cube(cube)
}

fn remove_common_cube(cube: SopCube, common: SopCube) -> SopCube {
    debug_assert!(cube_contains_common(cube, common));
    SopCube {
        care_mask: cube.care_mask & !common.care_mask,
        value_mask: cube.value_mask & !common.care_mask,
    }
}

fn choose_factoring_cube(cubes: &[SopCube]) -> Option<SopCube> {
    let mut best: Option<(usize, usize, SopCube)> = None;
    for i in 0..cubes.len() {
        for j in (i + 1)..cubes.len() {
            let common = common_cube(cubes[i], cubes[j]);
            let literal_count = common.literal_count();
            if literal_count < 2 {
                continue;
            }
            let covered_count = cubes
                .iter()
                .filter(|cube| cube_contains_common(**cube, common))
                .count();
            if covered_count < 2 || covered_count == cubes.len() {
                continue;
            }
            let score = literal_count * (covered_count - 1);
            match best {
                None => best = Some((score, covered_count, common)),
                Some((best_score, best_covered_count, best_common)) => {
                    if (score, covered_count, std::cmp::Reverse(common))
                        > (
                            best_score,
                            best_covered_count,
                            std::cmp::Reverse(best_common),
                        )
                    {
                        best = Some((score, covered_count, common));
                    }
                }
            }
        }
    }
    best.map(|(_, _, cube)| cube)
}

fn choose_factoring_literal(cubes: &[SopCube], nvars: usize) -> Option<(usize, bool)> {
    let mut best: Option<(usize, usize, bool)> = None;
    for var in 0..nvars {
        for negated in [false, true] {
            let count = cubes
                .iter()
                .filter(|cube| literal_in_cube(**cube, var) == Some(negated))
                .count();
            if count < 2 || count == cubes.len() {
                continue;
            }
            match best {
                None => best = Some((count, var, negated)),
                Some((best_count, best_var, best_negated)) => {
                    if (count, usize::MAX - var, !negated)
                        > (best_count, usize::MAX - best_var, !best_negated)
                    {
                        best = Some((count, var, negated));
                    }
                }
            }
        }
    }
    best.map(|(_, var, negated)| (var, negated))
}

fn factor_sop_cubes_inner(
    builder: &mut FactoredExprBuilder,
    cubes: &[SopCube],
    nvars: usize,
) -> FactoredExprId {
    let cubes = normalize_sop_cubes(cubes);
    if cubes.is_empty() {
        return builder.const_expr(false);
    }
    if cubes.iter().any(|cube| cube.care_mask == 0) {
        return builder.const_expr(true);
    }

    let common = common_literals(&cubes, nvars);
    if !common.is_empty() {
        let common_exprs: Vec<FactoredExprId> = common
            .iter()
            .map(|(var, negated)| builder.literal_expr(*var, *negated))
            .collect();
        let residual: Vec<SopCube> = cubes
            .iter()
            .copied()
            .map(|cube| remove_literals_from_cube(cube, &common))
            .collect();
        let common_expr = expr_and_balanced(builder, &common_exprs);
        let residual_expr = factor_sop_cubes_inner(builder, &residual, nvars);
        return builder.expr_and(common_expr, residual_expr);
    }

    if let Some(common) = choose_factoring_cube(&cubes) {
        let mut quotient = Vec::new();
        let mut remainder = Vec::new();
        for cube in &cubes {
            if cube_contains_common(*cube, common) {
                quotient.push(remove_common_cube(*cube, common));
            } else {
                remainder.push(*cube);
            }
        }
        let common_expr = cube_product_expr(builder, common, nvars);
        let quotient_expr = factor_sop_cubes_inner(builder, &quotient, nvars);
        let factored = builder.expr_and(common_expr, quotient_expr);
        if remainder.is_empty() {
            return factored;
        }
        let remainder_expr = factor_sop_cubes_inner(builder, &remainder, nvars);
        return builder.expr_or(factored, remainder_expr);
    }

    if let Some((var, negated)) = choose_factoring_literal(&cubes, nvars) {
        let mut quotient = Vec::new();
        let mut remainder = Vec::new();
        for cube in &cubes {
            if literal_in_cube(*cube, var) == Some(negated) {
                quotient.push(remove_literal_from_cube(*cube, var));
            } else {
                remainder.push(*cube);
            }
        }
        let literal_expr = builder.literal_expr(var, negated);
        let quotient_expr = factor_sop_cubes_inner(builder, &quotient, nvars);
        let factored = builder.expr_and(literal_expr, quotient_expr);
        if remainder.is_empty() {
            return factored;
        }
        let remainder_expr = factor_sop_cubes_inner(builder, &remainder, nvars);
        return builder.expr_or(factored, remainder_expr);
    }

    let mut terms = Vec::with_capacity(cubes.len());
    for cube in &cubes {
        terms.push(cube_product_expr(builder, *cube, nvars));
    }
    expr_or_balanced(builder, &terms)
}

fn factor_sop_cubes(cubes: &[SopCube], nvars: usize) -> FactoredExpr {
    let mut builder = FactoredExprBuilder::default();
    let root = factor_sop_cubes_inner(&mut builder, cubes, nvars);
    builder.finish(root)
}

#[derive(Debug, Default)]
pub(super) struct SopCoverMemo {
    covers: HashMap<LargeTruthTable, Option<Vec<SopCube>>>,
}

impl SopCoverMemo {
    fn cover_for(&mut self, target: &LargeTruthTable) -> Option<Vec<SopCube>> {
        if let Some(cached) = self.covers.get(target) {
            return cached.clone();
        }
        let cover = derive_sop_cover_for_target_uncached(target);
        self.covers.insert(*target, cover.clone());
        cover
    }
}

fn derive_sop_cover_for_target(
    target: &LargeTruthTable,
    memo: &mut SopCoverMemo,
) -> Option<Vec<SopCube>> {
    memo.cover_for(target)
}

fn derive_sop_cover_for_target_uncached(target: &LargeTruthTable) -> Option<Vec<SopCube>> {
    if target.is_const0() {
        return Some(Vec::new());
    }
    if target.is_const1() {
        return Some(vec![SopCube {
            care_mask: 0,
            value_mask: 0,
        }]);
    }

    if let Some(cubes) = derive_isop_cover_for_target(target) {
        return Some(cubes);
    }

    let full_care_mask = if target.nvars == 16 {
        u16::MAX
    } else {
        (1u16 << target.nvars) - 1
    };
    let mut cubes = DenseCubeSet::new(target.nvars);
    let mut cover_cache = CubeCoverCache::new(target.nvars);
    for assignment in 0..LargeTruthTable::bit_count(target.nvars) {
        if target.get_bit(assignment) {
            cubes.insert(SopCube {
                care_mask: full_care_mask,
                value_mask: (assignment as u16) & full_care_mask,
            });
        }
    }

    loop {
        let newly_merged = merge_adjacent_implicant_cubes(&cubes, target, &mut cover_cache);
        let old_len = cubes.len();
        cubes.extend(newly_merged);
        if cubes.len() > SOP_MAX_CUBES_DURING_MERGE {
            return None;
        }
        if cubes.len() == old_len {
            break;
        }
    }

    let sorted_cubes = cubes.into_sorted_vec();
    let prime_cubes = prune_subsumed_cubes(&sorted_cubes);
    let greedy_cover = greedy_cover_from_cubes(&prime_cubes, target, &mut cover_cache);
    Some(prune_redundant_cover(
        greedy_cover,
        target,
        &mut cover_cache,
    ))
}

fn same_sop_replacement_shape(lhs: &SopReplacement, rhs: &SopReplacement) -> bool {
    lhs.nvars == rhs.nvars
        && lhs.cubes == rhs.cubes
        && lhs.factored == rhs.factored
        && lhs.output_negated == rhs.output_negated
}

fn push_sop_variant(
    out: &mut Vec<SopReplacement>,
    nvars: usize,
    cubes: &[SopCube],
    factored: FactoredExpr,
    output_negated: bool,
    kind: SopVariantKind,
) {
    let replacement = SopReplacement {
        nvars,
        cubes: cubes.to_vec(),
        factored,
        output_negated,
        kind,
    };
    if !out
        .iter()
        .any(|existing| same_sop_replacement_shape(existing, &replacement))
    {
        out.push(replacement);
    }
}

fn append_sop_replacements_for_target(
    out: &mut Vec<SopReplacement>,
    cover_memo: &mut SopCoverMemo,
    target: &LargeTruthTable,
    output_negated: bool,
    leaf_depths: &[usize],
) {
    let Some(cubes) = derive_sop_cover_for_target(target, cover_memo) else {
        return;
    };

    let nvars = target.nvars;
    let flat = flat_sop_expr(&cubes, nvars);
    push_sop_variant(
        out,
        nvars,
        &cubes,
        flat,
        output_negated,
        SopVariantKind::Flat,
    );

    let arrival_balanced = arrival_balanced_sop_expr(&cubes, nvars, leaf_depths);
    push_sop_variant(
        out,
        nvars,
        &cubes,
        arrival_balanced,
        output_negated,
        SopVariantKind::ArrivalBalanced,
    );

    let factored = factor_sop_cubes(&cubes, nvars);
    push_sop_variant(
        out,
        nvars,
        &cubes,
        factored,
        output_negated,
        SopVariantKind::Factored,
    );
}

#[cfg(test)]
pub(super) fn derive_sop_replacements(
    tt: &LargeTruthTable,
    leaf_depths: &[usize],
) -> Vec<SopReplacement> {
    let mut cover_memo = SopCoverMemo::default();
    derive_sop_replacements_with_memo(tt, leaf_depths, &mut cover_memo)
}

fn derive_sop_replacements_with_memo(
    tt: &LargeTruthTable,
    leaf_depths: &[usize],
    cover_memo: &mut SopCoverMemo,
) -> Vec<SopReplacement> {
    debug_assert!(tt.nvars <= 16);
    debug_assert_eq!(leaf_depths.len(), tt.nvars);

    let mut out = Vec::new();
    let ones = tt.count_ones();
    let bit_count = LargeTruthTable::bit_count(tt.nvars);

    if ones <= bit_count - ones {
        append_sop_replacements_for_target(&mut out, cover_memo, tt, false, leaf_depths);
        append_sop_replacements_for_target(&mut out, cover_memo, &tt.not(), true, leaf_depths);
    } else {
        append_sop_replacements_for_target(&mut out, cover_memo, &tt.not(), true, leaf_depths);
        append_sop_replacements_for_target(&mut out, cover_memo, tt, false, leaf_depths);
    }

    out
}

fn positive_operand(node: AigRef) -> AigOperand {
    AigOperand {
        node,
        negated: false,
    }
}

fn is_cone_leaf_node(g: &GateFn, node: AigRef) -> bool {
    !matches!(g.gates[node.id], AigNode::Literal { .. })
}

fn find_reconvergent_cone(
    g: &GateFn,
    root: AigRef,
    depths: &[usize],
    use_counts: &[usize],
) -> Option<Vec<AigOperand>> {
    let AigNode::And2 { a, b, .. } = &g.gates[root.id] else {
        return None;
    };

    let mut leaves = BTreeSet::new();
    let mut marked = BTreeSet::new();
    marked.insert(root);
    for child in [a.node, b.node] {
        if is_cone_leaf_node(g, child) && marked.insert(child) {
            leaves.insert(child);
        }
    }

    loop {
        let mut best: Option<(usize, usize, AigRef, Vec<AigRef>)> = None;
        let internal_count = marked.len().saturating_sub(leaves.len());
        if internal_count >= LARGE_CONE_MAX_INTERNAL_NODES {
            break;
        }

        for leaf in &leaves {
            let AigNode::And2 { a, b, .. } = &g.gates[leaf.id] else {
                continue;
            };

            let mut new_children = Vec::new();
            for child in [a.node, b.node] {
                if !is_cone_leaf_node(g, child) || marked.contains(&child) {
                    continue;
                }
                new_children.push(child);
            }

            let cost = new_children.len();
            if cost == 2 && use_counts[leaf.id] > LARGE_CONE_FANOUT_STOP {
                continue;
            }
            if leaves.len().saturating_sub(1) + cost > LARGE_CONE_MAX_LEAVES {
                continue;
            }
            let key = (
                cost,
                usize::MAX - depths[leaf.id],
                *leaf,
                new_children.clone(),
            );
            match &best {
                None => best = Some(key),
                Some((best_cost, best_reverse_depth, best_leaf, best_children)) => {
                    if key
                        < (
                            *best_cost,
                            *best_reverse_depth,
                            *best_leaf,
                            best_children.clone(),
                        )
                    {
                        best = Some(key);
                    }
                }
            }
        }

        let Some((_, _, leaf, new_children)) = best else {
            break;
        };
        leaves.remove(&leaf);
        for child in new_children {
            marked.insert(child);
            leaves.insert(child);
        }
    }

    if leaves.is_empty() || leaves.len() > LARGE_CONE_MAX_LEAVES {
        return None;
    }

    Some(leaves.into_iter().map(positive_operand).collect())
}

fn collapse_cone_truth(g: &GateFn, root: AigRef, leaves: &[AigOperand]) -> LargeTruthTable {
    let nvars = leaves.len();
    assert!(
        nvars <= LARGE_CONE_MAX_LEAVES,
        "large-cone truth collapse got {} leaves for root {:?}, exceeding max {}",
        nvars,
        root,
        LARGE_CONE_MAX_LEAVES
    );

    let mut boundary = BTreeMap::new();
    for (i, leaf) in leaves.iter().enumerate() {
        assert!(
            boundary.insert(leaf.node, i).is_none(),
            "large-cone truth collapse got duplicate boundary leaf {:?} for root {:?}",
            leaf.node,
            root
        );
    }

    fn eval_operand(
        g: &GateFn,
        root: AigRef,
        op: AigOperand,
        nvars: usize,
        boundary: &BTreeMap<AigRef, usize>,
        memo: &mut BTreeMap<AigRef, LargeTruthTable>,
    ) -> LargeTruthTable {
        let mut tt = eval_node(g, root, op.node, nvars, boundary, memo);
        if op.negated {
            tt = tt.not();
        }
        tt
    }

    fn eval_node(
        g: &GateFn,
        root: AigRef,
        node: AigRef,
        nvars: usize,
        boundary: &BTreeMap<AigRef, usize>,
        memo: &mut BTreeMap<AigRef, LargeTruthTable>,
    ) -> LargeTruthTable {
        if let Some(index) = boundary.get(&node) {
            return LargeTruthTable::var(nvars, *index);
        }
        if let Some(tt) = memo.get(&node) {
            return tt.clone();
        }
        let tt = match &g.gates[node.id] {
            AigNode::Input { .. } => panic!(
                "large-cone truth collapse reached unbound input {:?} under root {:?}",
                node, root
            ),
            AigNode::Literal { value, .. } => {
                if *value {
                    LargeTruthTable::const1(nvars)
                } else {
                    LargeTruthTable::const0(nvars)
                }
            }
            AigNode::And2 { a, b, .. } => {
                let a_tt = eval_operand(g, root, *a, nvars, boundary, memo);
                let b_tt = eval_operand(g, root, *b, nvars, boundary, memo);
                a_tt.and(&b_tt)
            }
        };
        memo.insert(node, tt.clone());
        tt
    }

    eval_node(g, root, root, nvars, &boundary, &mut BTreeMap::new())
}

/// Constructs large-cone candidate replacements for `root`.
///
/// This stage only chooses the cone boundary, collapses it to a truth table,
/// and derives replacement SOP variants. Costing and acceptance are handled by
/// the caller so delay and area modes can tune those policies independently.
pub(super) fn construct_large_cone_candidate_replacements_for_root(
    g: &GateFn,
    root: AigRef,
    depths: &[usize],
    use_counts: &[usize],
    candidate_evals: &mut usize,
    stats: &mut LargeConeCandidateStats,
    structural_hash_state: &DynamicStructuralHash,
    sop_cover_memo: &mut SopCoverMemo,
) -> ConstructedLargeConeCandidates {
    let mut cands = Vec::new();
    let mut candidates_considered = 0usize;

    let Some(leaf_ops) = find_reconvergent_cone(g, root, depths, use_counts) else {
        stats.rejected_no_cone += 1;
        return ConstructedLargeConeCandidates {
            candidates: cands,
            candidates_considered,
        };
    };
    stats.cones_built += 1;
    stats.cone_leaves_sum += leaf_ops.len();
    stats.cone_leaves_max = stats.cone_leaves_max.max(leaf_ops.len());
    let internal_node_count = collect_internal_and_nodes_under_cut(g, root, &leaf_ops).len();
    stats.cone_internal_sum += internal_node_count;
    stats.cone_internal_max = stats.cone_internal_max.max(internal_node_count);
    let mffc_nodes = collect_mffc_nodes_under_cut(structural_hash_state, root, &leaf_ops);
    if mffc_nodes.is_empty() {
        stats.rejected_empty_mffc += 1;
        return ConstructedLargeConeCandidates {
            candidates: cands,
            candidates_considered,
        };
    }
    stats.cone_mffc_sum += mffc_nodes.len();
    stats.cone_mffc_max = stats.cone_mffc_max.max(mffc_nodes.len());

    let tt = collapse_cone_truth(g, root, &leaf_ops);

    candidates_considered += 1;
    let leaf_depths: Vec<usize> = leaf_ops.iter().map(|leaf| depths[leaf.node.id]).collect();
    let sops = derive_sop_replacements_with_memo(&tt, &leaf_depths, sop_cover_memo);
    if sops.is_empty() {
        stats.rejected_sop_failed += 1;
        return ConstructedLargeConeCandidates {
            candidates: cands,
            candidates_considered,
        };
    }
    *candidate_evals += sops.len();
    stats.sop_variants_sum += sops.len();
    stats.sop_variants_max = stats.sop_variants_max.max(sops.len());

    for (variant_order, sop) in sops.into_iter().enumerate() {
        let raw_and_count = estimate_factored_and_count(&sop.factored);
        let implementation = ReplacementImpl::Sop(sop);
        let new_root_depth = match &implementation {
            ReplacementImpl::Fragment { .. } => unreachable!("large-cone rewrites use SOPs"),
            ReplacementImpl::Sop(sop) => sop_depth_from_depths(&leaf_ops, sop, depths),
        };
        cands.push(LargeConeCandidate {
            replacement: Replacement {
                root,
                leaf_ops: leaf_ops.clone(),
                score_depth: new_root_depth,
                score_ands: raw_and_count,
                raw_score_ands: raw_and_count,
                structural_hash_only_area_win: false,
                implementation,
            },
            mffc_nodes: mffc_nodes.clone(),
            new_root_depth,
            raw_and_count,
            variant_order,
        });
    }

    cands.sort_by(|a, b| {
        (
            a.new_root_depth,
            a.raw_and_count,
            a.replacement.root.id,
            &a.replacement.leaf_ops,
            a.variant_order,
        )
            .cmp(&(
                b.new_root_depth,
                b.raw_and_count,
                b.replacement.root.id,
                &b.replacement.leaf_ops,
                b.variant_order,
            ))
    });

    ConstructedLargeConeCandidates {
        candidates: cands,
        candidates_considered,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn eval_factored_expr(expr: &FactoredExpr, assignment: usize) -> bool {
        let mut values = Vec::with_capacity(expr.nodes().len());
        for node in expr.nodes().iter().copied() {
            let value = match node {
                FactoredExprNode::Const(value) => value,
                FactoredExprNode::Lit { var, negated } => {
                    (((assignment >> var) & 1) != 0) ^ negated
                }
                FactoredExprNode::And { lhs, rhs } => values[lhs.0] && values[rhs.0],
                FactoredExprNode::Or { lhs, rhs } => values[lhs.0] || values[rhs.0],
            };
            values.push(value);
        }
        values[expr.root().0]
    }

    fn eval_sop_replacement(sop: &SopReplacement, assignment: usize) -> bool {
        let mut value = eval_factored_expr(&sop.factored, assignment);
        if sop.output_negated {
            value = !value;
        }
        value
    }

    fn factored_depth_with_leaf_depths(expr: &FactoredExpr, leaf_depths: &[usize]) -> usize {
        let mut depths: Vec<usize> = Vec::with_capacity(expr.nodes().len());
        for node in expr.nodes().iter().copied() {
            let depth = match node {
                FactoredExprNode::Const(_) => 0,
                FactoredExprNode::Lit { var, .. } => leaf_depths[var],
                FactoredExprNode::And { lhs, rhs } | FactoredExprNode::Or { lhs, rhs } => {
                    depths[lhs.0].max(depths[rhs.0]) + 1
                }
            };
            depths.push(depth);
        }
        depths[expr.root().0]
    }

    #[test]
    fn test_sop_replacement_variants_preserve_truth_table() {
        let mut tt = LargeTruthTable::const0(3);
        for assignment in 0..LargeTruthTable::bit_count(3) {
            let a = (assignment & 0b001) != 0;
            let b = (assignment & 0b010) != 0;
            let c = (assignment & 0b100) != 0;
            tt.set_bit(assignment, a && (b || c));
        }

        let variants = derive_sop_replacements(&tt, &[2, 0, 0]);
        assert!(
            variants.len() > 1,
            "expected multiple SOP construction variants; got {variants:?}"
        );
        assert!(
            variants.iter().any(|sop| sop.output_negated)
                && variants.iter().any(|sop| !sop.output_negated),
            "both output polarities should be considered"
        );

        for sop in &variants {
            for assignment in 0..LargeTruthTable::bit_count(3) {
                assert_eq!(
                    eval_sop_replacement(sop, assignment),
                    tt.get_bit(assignment),
                    "variant {sop:?} mismatch at assignment={assignment:03b}"
                );
            }
        }
    }

    #[test]
    fn test_sop_replacements_preserve_all_3var_truth_tables() {
        for mask in 0u16..=0xff {
            let mut tt = LargeTruthTable::const0(3);
            for assignment in 0..LargeTruthTable::bit_count(3) {
                tt.set_bit(assignment, ((mask >> assignment) & 1) != 0);
            }

            let variants = derive_sop_replacements(&tt, &[0, 0, 0]);
            assert!(
                !variants.is_empty(),
                "expected at least one SOP variant for mask={mask:#04x}"
            );
            for sop in &variants {
                for assignment in 0..LargeTruthTable::bit_count(3) {
                    assert_eq!(
                        eval_sop_replacement(sop, assignment),
                        tt.get_bit(assignment),
                        "variant {sop:?} mismatch for mask={mask:#04x} assignment={assignment:03b}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_arrival_balanced_sop_variant_can_reduce_depth() {
        let mut tt = LargeTruthTable::const0(4);
        for assignment in 0..LargeTruthTable::bit_count(4) {
            tt.set_bit(assignment, assignment == 0b1111);
        }

        let leaf_depths = [10, 9, 0, 0];
        let variants = derive_sop_replacements(&tt, &leaf_depths);
        let depths: BTreeSet<usize> = variants
            .iter()
            .filter(|sop| !sop.output_negated)
            .map(|sop| factored_depth_with_leaf_depths(&sop.factored, &leaf_depths))
            .collect();

        assert!(
            depths.contains(&11) && depths.contains(&12),
            "expected both delay-aware and balanced AND4 shapes; got depths {depths:?}"
        );
    }
}
