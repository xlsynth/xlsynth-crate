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

/// A candidate circuit derived from a Boolean sum of products (an OR of AND
/// terms). For example, factoring `(a & b) | (a & c)` gives `a & (b | c)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::aig) struct SopReplacement {
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
    /// The low six bits of an assignment select a bit within a 64-bit word.
    /// Higher bits select the word, where those input variables are constant.
    const WITHIN_WORD_VARIABLES: usize = u64::BITS.ilog2() as usize;

    // A set bit marks an assignment where the corresponding variable is one.
    const VARIABLE_ONES: [u64; Self::WITHIN_WORD_VARIABLES] = [
        0xaaaa_aaaa_aaaa_aaaa,
        0xcccc_cccc_cccc_cccc,
        0xf0f0_f0f0_f0f0_f0f0,
        0xff00_ff00_ff00_ff00,
        0xffff_0000_ffff_0000,
        0xffff_ffff_0000_0000,
    ];

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
        debug_assert!(index < nvars);
        let mut out = Self::const0(nvars);
        let active = &mut out.words[..Self::word_count(nvars)];
        if index < Self::WITHIN_WORD_VARIABLES {
            // One repeated pattern per word; only narrow tables need trimming.
            active.fill(Self::VARIABLE_ONES[index] & Self::unused_high_mask(nvars));
        } else {
            let stride = 1 << (index - Self::WITHIN_WORD_VARIABLES);
            for (word, value) in active.iter_mut().enumerate() {
                *value = if word & stride == 0 { 0 } else { u64::MAX };
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

    #[cfg(test)]
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

    /// Fix one input while keeping the original row numbering. Fixing input
    /// zero to false, for example, copies each even row to its odd neighbor.
    /// For the low six inputs, operate within a word. Higher inputs select
    /// whole words.
    fn cofactor_fixed(&self, var: usize, value: bool) -> Self {
        debug_assert!(var < self.nvars);
        let mut out = Self::const0(self.nvars);
        let inputs = self.active_words();
        let outputs = &mut out.words[..inputs.len()];
        if var < Self::WITHIN_WORD_VARIABLES {
            let stride = 1 << var;
            let zero_mask = !Self::VARIABLE_ONES[var];
            for (dst, &src) in outputs.iter_mut().zip(inputs) {
                let low = (if value { src >> stride } else { src }) & zero_mask;
                *dst = low | (low << stride);
            }
        } else {
            let stride = 1 << (var - Self::WITHIN_WORD_VARIABLES);
            let offset = if value { stride } else { 0 };
            for (word, dst) in outputs.iter_mut().enumerate() {
                *dst = inputs[(word & !stride) | offset];
            }
        }
        out
    }

    fn depends_on_var(&self, var: usize) -> bool {
        debug_assert!(var < self.nvars);
        let words = self.active_words();
        if var < Self::WITHIN_WORD_VARIABLES {
            let stride = 1 << var;
            let zero_mask = !Self::VARIABLE_ONES[var];
            return words
                .iter()
                .any(|&word| (word ^ (word >> stride)) & zero_mask != 0);
        }
        let stride = 1 << (var - Self::WITHIN_WORD_VARIABLES);
        words
            .chunks_exact(stride * 2)
            .any(|group| group[..stride] != group[stride..])
    }

    fn topmost_support_var_with(&self, other: &Self, upper_var: usize) -> Option<usize> {
        debug_assert_eq!(self.nvars, other.nvars);
        (0..upper_var)
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

/// Appends the cubes for this subproblem, with its ancestor literals attached,
/// and returns the realized truth needed by the sibling subproblems.
fn isop_rec(
    on: LargeTruthTable,
    on_dc: LargeTruthTable,
    upper_var: usize,
    prefix: SopCube,
    cubes: &mut Vec<SopCube>,
) -> Option<LargeTruthTable> {
    debug_assert_eq!(on.nvars, on_dc.nvars);
    debug_assert!(on.sharp(&on_dc).is_const0());
    if on.is_const0() {
        return Some(on);
    }
    if on_dc.is_const1() {
        if cubes.len() == SOP_MAX_CUBES_DURING_MERGE {
            return None;
        }
        cubes.push(prefix);
        return Some(on_dc);
    }

    // Both cofactors and all Boolean combinations below are independent of
    // this split and the higher variables. Keep the original split order.
    let var = on.topmost_support_var_with(&on_dc, upper_var)?;
    let on0 = on.cofactor_fixed(var, false);
    let on1 = on.cofactor_fixed(var, true);
    let on_dc0 = on_dc.cofactor_fixed(var, false);
    let on_dc1 = on_dc.cofactor_fixed(var, true);
    let low = isop_rec(
        on0.sharp(&on_dc1),
        on_dc0,
        var,
        add_literal_to_cube(prefix, var, true),
        cubes,
    )?;
    let high = isop_rec(
        on1.sharp(&on_dc0),
        on_dc1,
        var,
        add_literal_to_cube(prefix, var, false),
        cubes,
    )?;
    let shared = isop_rec(
        on0.sharp(&low).or(&on1.sharp(&high)),
        on_dc0.and(&on_dc1),
        var,
        prefix,
        cubes,
    )?;
    Some(shared.or(&truth_from_cofactored_results(&low, &high, var)))
}

fn derive_isop_cover_for_target(target: &LargeTruthTable) -> Option<Vec<SopCube>> {
    let mut cubes = Vec::new();
    let prefix = SopCube {
        care_mask: 0,
        value_mask: 0,
    };
    isop_rec(*target, *target, target.nvars, prefix, &mut cubes)?;
    Some(normalize_sop_cubes(&cubes))
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
        // A node is appended only after its children. Once we reach an ID in
        // this reverse walk, all parents that could make it live were visited.
        let mut reachable = vec![false; self.nodes.len()];
        reachable[root.0] = true;
        let mut live_count = 0;
        for id in (0..=root.0).rev() {
            if reachable[id] {
                live_count += 1;
                if let Some((lhs, rhs)) = self.nodes[id].children() {
                    reachable[lhs.0] = true;
                    reachable[rhs.0] = true;
                }
            }
        }
        if live_count == self.nodes.len() {
            return FactoredExpr {
                nodes: self.nodes,
                root,
            };
        }

        let mut old_to_new = vec![None; self.nodes.len()];
        let mut nodes = Vec::with_capacity(live_count);
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
    if cubes.is_empty() {
        return builder.const_expr(false);
    }
    if let [cube] = cubes {
        return cube_product_expr(builder, *cube, nvars);
    }

    let common = common_literals(cubes, nvars);
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

    if let Some(common) = choose_factoring_cube(cubes) {
        let mut quotient = Vec::new();
        let mut remainder = Vec::new();
        for cube in cubes {
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

    if let Some((var, negated)) = choose_factoring_literal(cubes, nvars) {
        let mut quotient = Vec::new();
        let mut remainder = Vec::new();
        for cube in cubes {
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
    for cube in cubes {
        terms.push(cube_product_expr(builder, *cube, nvars));
    }
    expr_or_balanced(builder, &terms)
}

fn factor_sop_cubes(cubes: &[SopCube], nvars: usize) -> FactoredExpr {
    let mut builder = FactoredExprBuilder::default();
    // Each recursive branch keeps a subsequence of this sorted cover. A
    // quotient removes exactly the same fixed bits from each of its cubes,
    // preserving their order, distinctness, and subsumption relationships.
    let normalized = normalize_sop_cubes(cubes);
    let root = factor_sop_cubes_inner(&mut builder, &normalized, nvars);
    builder.finish(root)
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

type SopVariantKey = (
    LargeTruthTable,
    smallvec::SmallVec<[usize; LARGE_CONE_MAX_LEAVES]>,
);

/// Reuse covers by truth and completed expressions by truth and ordered input
/// depths across cuts in this rewrite. The caller evaluates each expression
/// against its current graph.
#[derive(Debug, Default)]
pub(super) struct SopMemo {
    covers: HashMap<LargeTruthTable, Option<Vec<SopCube>>>,
    variants: HashMap<SopVariantKey, Vec<SopReplacement>>,
    cached_variant_nodes: usize,
}

impl SopMemo {
    const MAX_ENTRIES: usize = 512;
    // Account for expression nodes across all entries so one larger shape can
    // be reused while the cache stays within the same memory budget.
    const MAX_CACHED_NODES: usize = 128 * 1024;

    fn replacements_for(
        &mut self,
        truth: &LargeTruthTable,
        leaf_depths: &[usize],
    ) -> Vec<SopReplacement> {
        debug_assert_eq!(leaf_depths.len(), truth.nvars);
        let key = (*truth, smallvec::SmallVec::from_slice(leaf_depths));
        if let Some(variants) = self.variants.get(&key) {
            return variants.clone();
        }

        let variants = self.build_replacements(truth, leaf_depths);
        let nodes: usize = variants.iter().map(|sop| sop.factored.nodes.len()).sum();
        if nodes <= Self::MAX_CACHED_NODES {
            if self.variants.len() >= Self::MAX_ENTRIES
                || self.cached_variant_nodes + nodes > Self::MAX_CACHED_NODES
            {
                self.variants.clear();
                self.cached_variant_nodes = 0;
            }
            self.variants.insert(key, variants.clone());
            self.cached_variant_nodes += nodes;
        }
        variants
    }

    fn build_replacements(
        &mut self,
        truth: &LargeTruthTable,
        leaf_depths: &[usize],
    ) -> Vec<SopReplacement> {
        let mut out = Vec::new();
        let complement = truth.not();
        let ones = truth.count_ones();
        let phases = if ones <= LargeTruthTable::bit_count(truth.nvars) - ones {
            [(truth, false), (&complement, true)]
        } else {
            [(&complement, true), (truth, false)]
        };
        for (target, output_negated) in phases {
            self.append_polarity(&mut out, target, output_negated, leaf_depths);
        }
        out
    }

    fn append_polarity(
        &mut self,
        out: &mut Vec<SopReplacement>,
        target: &LargeTruthTable,
        output_negated: bool,
        leaf_depths: &[usize],
    ) {
        let Some(cubes) = self
            .covers
            .entry(*target)
            .or_insert_with_key(derive_sop_cover_for_target_uncached)
            .as_deref()
        else {
            return;
        };
        let nvars = target.nvars;
        // Only this polarity's variants share the same target and cover. Retain
        // the first builder for each distinct expression, in the original
        // order.
        let phase_start = out.len();
        for (kind, factored) in [
            (SopVariantKind::Flat, flat_sop_expr(cubes, nvars)),
            (
                SopVariantKind::ArrivalBalanced,
                arrival_balanced_sop_expr(cubes, nvars, leaf_depths),
            ),
            (SopVariantKind::Factored, factor_sop_cubes(cubes, nvars)),
        ] {
            if out[phase_start..]
                .iter()
                .all(|sop| sop.factored != factored)
            {
                out.push(SopReplacement {
                    factored,
                    output_negated,
                    kind,
                });
            }
        }
    }
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
    sop_memo: &mut SopMemo,
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
    let sops = sop_memo.replacements_for(&tt, &leaf_depths);
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

    #[test]
    fn packed_variable_tables_match_each_assignment_and_padding() {
        for nvars in 1..=LARGE_CONE_MAX_LEAVES {
            for var in 0..nvars {
                let mut expected = LargeTruthTable::const0(nvars);
                for row in 0..LargeTruthTable::bit_count(nvars) {
                    expected.set_bit(row, (row >> var) & 1 != 0);
                }
                assert_eq!(LargeTruthTable::var(nvars, var), expected);
            }
        }
    }

    #[test]
    fn packed_cofactors_and_dependence_match_assignment_pairs() {
        for nvars in 1..=LARGE_CONE_MAX_LEAVES {
            let mut dense = LargeTruthTable::const0(nvars);
            for row in 0..LargeTruthTable::bit_count(nvars) {
                dense.set_bit(row, (row.wrapping_mul(7919) % 149) < 75);
            }
            for truth in [
                dense,
                LargeTruthTable::const1(nvars),
                LargeTruthTable::var(nvars, nvars - 1),
            ] {
                for var in 0..nvars {
                    let mut low = LargeTruthTable::const0(nvars);
                    let mut high = LargeTruthTable::const0(nvars);
                    let bit = 1 << var;
                    for row in 0..LargeTruthTable::bit_count(nvars) {
                        low.set_bit(row, truth.get_bit(row & !bit));
                        high.set_bit(row, truth.get_bit(row | bit));
                    }
                    assert_eq!(truth.cofactor_fixed(var, false), low);
                    assert_eq!(truth.cofactor_fixed(var, true), high);
                    assert_eq!(truth.depends_on_var(var), low != high);
                }
            }
        }
    }

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
    fn isop_emits_low_high_and_shared_cubes_in_order() {
        // The high input is absent and the upper bound includes two don't-care
        // assignments per three-input slice.
        let mut on = LargeTruthTable::const0(4);
        on.words[0] = 0xc9c9;
        let mut on_dc = LargeTruthTable::const0(4);
        on_dc.words[0] = 0xebeb;
        let mut cubes = Vec::new();
        let prefix = SopCube {
            care_mask: 0,
            value_mask: 0,
        };
        let realized = isop_rec(on, on_dc, 4, prefix, &mut cubes).unwrap();
        assert_eq!(realized, on_dc);
        let masks: Vec<_> = cubes.iter().map(|c| (c.care_mask, c.value_mask)).collect();
        assert_eq!(masks, [(0b110, 0), (0b110, 0b110), (1, 1)]);
    }

    #[test]
    fn finish_compacts_unused_nodes_before_and_after_the_root() {
        let mut builder = FactoredExprBuilder::default();
        let a = builder.literal_expr(0, false);
        let unused = builder.literal_expr(1, false);
        builder.expr_or(a, unused);
        let b = builder.literal_expr(2, false);
        let root = builder.expr_and(a, b);
        builder.expr_or(root, unused);

        let mut minimal = FactoredExprBuilder::default();
        let a = minimal.literal_expr(0, false);
        let b = minimal.literal_expr(2, false);
        let and = minimal.expr_and(a, b);
        assert_eq!(builder.finish(root), minimal.finish(and));
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

        let variants = SopMemo::default().replacements_for(&tt, &[2, 0, 0]);
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

            let variants = SopMemo::default().replacements_for(&tt, &[0, 0, 0]);
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
    fn large_variants_share_the_cache_node_budget() {
        let mut parity = LargeTruthTable::const0(8);
        for a in 0..256usize {
            parity.set_bit(a, a.count_ones() % 2 != 0);
        }
        let mut memo = SopMemo::default();
        let expected = memo.replacements_for(&parity, &[0; 8]);
        let nodes = expected
            .iter()
            .map(|sop| sop.factored.nodes.len())
            .sum::<usize>();
        let fit = SopMemo::MAX_CACHED_NODES / nodes;
        assert!((1..SopMemo::MAX_ENTRIES).contains(&fit));
        assert_eq!(memo.cached_variant_nodes, nodes);
        assert_eq!(memo.replacements_for(&parity, &[0; 8]), expected);
        assert_eq!(memo.cached_variant_nodes, nodes);

        // Shifting all input arrivals leaves the balanced shape unchanged
        // but produces a distinct memo key with the same node count.
        for arrival in 1..fit {
            assert_eq!(memo.replacements_for(&parity, &[arrival; 8]), expected);
        }
        assert_eq!(memo.variants.len(), fit);
        assert_eq!(memo.cached_variant_nodes, fit * nodes);
        assert_eq!(memo.replacements_for(&parity, &[fit; 8]), expected);
        assert_eq!(memo.variants.len(), 1);
        assert_eq!(memo.cached_variant_nodes, nodes);
    }

    #[test]
    fn memoized_sop_variants_preserve_order_shape_and_polarity() {
        let mut tt = LargeTruthTable::const0(4);
        tt.set_bit(15, true);
        let left_late = [10, 9, 0, 0];
        let right_late = [0, 0, 9, 10];
        let mut memo = SopMemo::default();

        let expected = |truth: &LargeTruthTable, depths: &[usize]| {
            SopMemo::default().build_replacements(truth, depths)
        };
        assert_ne!(expected(&tt, &left_late), expected(&tt, &right_late));
        for _ in 0..2 {
            for truth in [tt, tt.not()] {
                for depths in [&left_late, &right_late] {
                    assert_eq!(
                        memo.replacements_for(&truth, depths),
                        expected(&truth, depths),
                    );
                }
            }
        }
        assert_eq!(memo.variants.len(), 4);
    }

    #[test]
    fn test_arrival_balanced_sop_variant_can_reduce_depth() {
        let mut tt = LargeTruthTable::const0(4);
        for assignment in 0..LargeTruthTable::bit_count(4) {
            tt.set_bit(assignment, assignment == 0b1111);
        }

        let leaf_depths = [10, 9, 0, 0];
        let variants = SopMemo::default().replacements_for(&tt, &leaf_depths);
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
