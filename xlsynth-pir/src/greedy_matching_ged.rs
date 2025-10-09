// SPDX-License-Identifier: Apache-2.0

//! Greedy edit-distance computation between two XLS IR functions.
//! Contains the matching machinery and conversion of matches into concrete
//! edits.

use crate::matching_ged::{
    DepGraph, MatchAction, MatchSelector, NewNodeRef, NodeSide, OldNodeRef, ReadyNode,
    build_dependency_graph,
};
use crate::node_hashing::FwdHash;
use log::{Level, debug, log_enabled, trace};
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Base of the fixed-point score scale used by greedy matching.
const SCORE_BASE: i32 = 1000;
/// Score for a perfect forward match or reverse match. Bias towards reverse
/// matches because reverse matches are computed once globally, while
/// forward matches are computed as the algorithm progresses so "perfect"
/// forward matches may be due to imprecise heuristic matches done earlier.
const PERFECT_FORWARD_MATCH_SCORE: i32 = SCORE_BASE * 100;
const PERFECT_REVERSE_MATCH_SCORE: i32 = SCORE_BASE * 200;

/// Generic vector indexed by node reference types (e.g. OldNodeRef/NewNodeRef).
/// Because these node ref types are (close to) dense, NodeVector can be used as
/// a fast map/set.
#[derive(Clone)]
struct NodeVector<I, D> {
    data: Vec<D>,
    _marker: PhantomData<I>,
}

impl<I, D: Clone> NodeVector<I, D> {
    fn new(len: usize, value: D) -> Self {
        Self {
            data: vec![value; len],
            _marker: PhantomData,
        }
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<I, D> NodeVector<I, D>
where
    I: Into<usize> + Copy,
{
    fn get(&self, index: I) -> &D {
        &self.data[index.into()]
    }
    fn get_mut(&mut self, index: I) -> &mut D {
        &mut self.data[index.into()]
    }
    fn set(&mut self, index: I, value: D) {
        self.data[index.into()] = value;
    }
}

// A map from (OldNodeRef, NewNodeRef) pairs to values.
struct NodePairMap<D> {
    data: NodeVector<OldNodeRef, NodeVector<NewNodeRef, Option<D>>>,
}

impl<D: Clone> NodePairMap<D> {
    fn new(old_len: usize, new_len: usize, default: Option<D>) -> Self {
        let inner = NodeVector::new(new_len, default);
        Self {
            data: NodeVector::new(old_len, inner),
        }
    }
    fn get(&self, old_index: OldNodeRef, new_index: NewNodeRef) -> Option<&D> {
        self.data.get(old_index).get(new_index).as_ref()
    }
    fn set(&mut self, old_index: OldNodeRef, new_index: NewNodeRef, value: D) {
        *self.data.get_mut(old_index).get_mut(new_index) = Some(value);
    }
}

/// A data structure collecting nodes which are structurally equivalent.
/// Specifically, it maps OldNodeRefs to equivalent NewNodeRefs and vice versa.
struct EquivalentNodeSet {
    // Equivalence class index for each Old/Node node. Indexed by OldNodeRef/NewNodeRef.index.
    old_index_to_equiv_class: NodeVector<OldNodeRef, usize>,
    new_index_to_equiv_class: NodeVector<NewNodeRef, usize>,
    // Vectors of equivalent nodes, indexed by equivalence class index.
    equivalent_new_nodes: Vec<Vec<NewNodeRef>>,
    equivalent_old_nodes: Vec<Vec<OldNodeRef>>,
}

impl EquivalentNodeSet {
    fn new(old_graph: &DepGraph<OldNodeRef>, new_graph: &DepGraph<NewNodeRef>) -> Self {
        // Compute max indices to size class index vectors deterministically
        let max_old_index = old_graph.nodes.len() - 1;
        let max_new_index = new_graph.nodes.len() - 1;
        let mut old_index_to_equiv_class = NodeVector::new(max_old_index + 1, 0usize);
        let mut new_index_to_equiv_class = NodeVector::new(max_new_index + 1, 0usize);

        let mut hash_set: HashSet<FwdHash> = HashSet::new();
        for node in old_graph.nodes.iter() {
            hash_set.insert(node.structural_hash);
        }
        for node in new_graph.nodes.iter() {
            hash_set.insert(node.structural_hash);
        }
        let equivalent_class_count = hash_set.len();
        let mut equivalent_old_nodes: Vec<Vec<OldNodeRef>> =
            vec![Vec::new(); equivalent_class_count];
        let mut equivalent_new_nodes: Vec<Vec<NewNodeRef>> =
            vec![Vec::new(); equivalent_class_count];

        let mut class_by_hash: HashMap<FwdHash, usize> = HashMap::new();
        let mut get_or_create_class = |h: FwdHash| -> usize {
            if let Some(&idx) = class_by_hash.get(&h) {
                idx
            } else {
                let idx = class_by_hash.len();
                class_by_hash.insert(h, idx);
                idx
            }
        };
        for (idx, node) in old_graph.nodes.iter().enumerate() {
            let equiv_class = get_or_create_class(node.structural_hash);
            old_index_to_equiv_class.set(OldNodeRef(idx), equiv_class);
            equivalent_old_nodes[equiv_class].push(OldNodeRef(idx));
        }
        for (idx, node) in new_graph.nodes.iter().enumerate() {
            let equiv_class = get_or_create_class(node.structural_hash);
            new_index_to_equiv_class.set(NewNodeRef(idx), equiv_class);
            equivalent_new_nodes[equiv_class].push(NewNodeRef(idx));
        }
        Self {
            old_index_to_equiv_class,
            new_index_to_equiv_class,
            equivalent_new_nodes,
            equivalent_old_nodes,
        }
    }
    pub fn get_equivalent_old_nodes(&self, new_index: NewNodeRef) -> &Vec<OldNodeRef> {
        let class = self.new_index_to_equiv_class.get(new_index);
        &self.equivalent_old_nodes[*class]
    }
    pub fn get_equivalent_new_nodes(&self, old_index: OldNodeRef) -> &Vec<NewNodeRef> {
        let equiv_class = self.old_index_to_equiv_class.get(old_index);
        &self.equivalent_new_nodes[*equiv_class]
    }
    pub fn drop_old_node(&mut self, old_node: OldNodeRef) {
        let equiv_class = self.old_index_to_equiv_class.get(old_node);
        let members = &mut self.equivalent_old_nodes[*equiv_class];
        let pos = members.iter().position(|x| *x == old_node).unwrap();
        members.swap_remove(pos);
    }
    pub fn drop_new_node(&mut self, new_node: NewNodeRef) {
        let equiv_class = self.new_index_to_equiv_class.get(new_node);
        let members = &mut self.equivalent_new_nodes[*equiv_class];
        let pos = members.iter().position(|x| *x == new_node).unwrap();
        members.swap_remove(pos);
    }
}

/// A cache of foward match scores for (OldNodeRef, NewNodeRef) pairs. The cache
/// is intelligently updated after matches are made to minimize wasted
/// computation.
struct ForwardScoreCache {
    scores: NodePairMap<i32>,
}

impl ForwardScoreCache {
    fn new(old_graph: &DepGraph<OldNodeRef>, new_graph: &DepGraph<NewNodeRef>) -> Self {
        Self {
            scores: NodePairMap::new(old_graph.nodes.len(), new_graph.nodes.len(), None),
        }
    }
    fn get(&self, old_index: OldNodeRef, new_index: NewNodeRef) -> i32 {
        self.scores.get(old_index, new_index).copied().unwrap_or(0)
    }

    fn compute_score(
        &self,
        old_index: OldNodeRef,
        new_index: NewNodeRef,
        matched_nodes: &NodeVector<OldNodeRef, Option<NewNodeRef>>,
        old_graph: &DepGraph<OldNodeRef>,
        new_graph: &DepGraph<NewNodeRef>,
    ) -> i32 {
        let old_operands: &Vec<OldNodeRef> = &old_graph.get_node(old_index).operands;
        let new_operands: &Vec<NewNodeRef> = &new_graph.get_node(new_index).operands;

        assert!(
            old_operands.len() == new_operands.len(),
            "forward_match_score expects equal operand counts for shape-equal nodes"
        );
        if old_operands.is_empty() {
            return SCORE_BASE;
        };
        let mut matches = 0usize;
        for i in 0..old_operands.len() {
            let op_old = old_operands[i];
            let op_new = new_operands[i];
            if *matched_nodes.get(op_old) == Some(op_new) {
                matches += 1;
            }
        }
        ((matches as i32) * SCORE_BASE) / (old_operands.len() as i32)
    }
    fn update_after_match(
        &mut self,
        old_index: OldNodeRef,
        new_index: NewNodeRef,
        equivalents: &EquivalentNodeSet,
        matched_nodes: &NodeVector<OldNodeRef, Option<NewNodeRef>>,
        old_graph: &DepGraph<OldNodeRef>,
        new_graph: &DepGraph<NewNodeRef>,
    ) {
        for old_user in old_graph.get_node(old_index).users.iter() {
            for new_node in equivalents.get_equivalent_new_nodes(old_user.0) {
                let score =
                    self.compute_score(old_user.0, *new_node, matched_nodes, old_graph, new_graph);
                self.scores.set(old_user.0, *new_node, score);
            }
        }
        for new_user in new_graph.get_node(new_index).users.iter() {
            for old_node in equivalents.get_equivalent_old_nodes(new_user.0) {
                let score =
                    self.compute_score(*old_node, new_user.0, matched_nodes, old_graph, new_graph);
                self.scores.set(*old_node, new_user.0, score);
            }
        }
    }
}

/// A cache of unready match scores for (OldNodeRef, NewNodeRef) pairs. The
/// unready match score for a node A is the best score for any node pair (A, B)
/// where B is not ready.The cache is intelligently updated after matches are
/// made to minimize wasted computation.
struct UnreadyMatchScoreCache {
    old_scores: RefCell<NodeVector<OldNodeRef, Option<i32>>>,
    new_scores: RefCell<NodeVector<NewNodeRef, Option<i32>>>,
}

impl UnreadyMatchScoreCache {
    fn new(old_graph: &DepGraph<OldNodeRef>, new_graph: &DepGraph<NewNodeRef>) -> Self {
        Self {
            old_scores: RefCell::new(NodeVector::new(old_graph.nodes.len(), None)),
            new_scores: RefCell::new(NodeVector::new(new_graph.nodes.len(), None)),
        }
    }
    fn get_old_score<IsUnreadyF, GetScoreF>(
        &self,
        old_index: OldNodeRef,
        equivalents: &EquivalentNodeSet,
        is_unready: IsUnreadyF,
        get_match_score: GetScoreF,
    ) -> i32
    where
        IsUnreadyF: Fn(NewNodeRef) -> bool,
        GetScoreF: Fn(OldNodeRef, NewNodeRef) -> i32,
    {
        if let Some(score) = self.old_scores.borrow().get(old_index).clone() {
            // let new_score =
            //     self.compute_old_score(old_index, equivalents, is_unready,
            // get_match_score); assert_eq!(score, new_score);
            score
        } else {
            let new_score =
                self.compute_old_score(old_index, equivalents, is_unready, get_match_score);
            self.old_scores.borrow_mut().set(old_index, Some(new_score));
            new_score
        }
    }
    fn get_new_score<IsUnreadyF, GetScoreF>(
        &self,
        new_index: NewNodeRef,
        equivalents: &EquivalentNodeSet,
        is_unready: IsUnreadyF,
        get_match_score: GetScoreF,
    ) -> i32
    where
        IsUnreadyF: Fn(OldNodeRef) -> bool,
        GetScoreF: Fn(OldNodeRef, NewNodeRef) -> i32,
    {
        if let Some(score) = self.new_scores.borrow().get(new_index).clone() {
            // let new_score =
            //     self.compute_new_score(new_index, equivalents, is_unready,
            // get_match_score); assert_eq!(score, new_score);
            score
        } else {
            let new_score =
                self.compute_new_score(new_index, equivalents, is_unready, get_match_score);
            self.new_scores.borrow_mut().set(new_index, Some(new_score));
            new_score
        }
    }
    fn compute_old_score<IsUnreadyF, GetScoreF>(
        &self,
        old_index: OldNodeRef,
        equivalents: &EquivalentNodeSet,
        is_unready: IsUnreadyF,
        get_match_score: GetScoreF,
    ) -> i32
    where
        IsUnreadyF: Fn(NewNodeRef) -> bool,
        GetScoreF: Fn(OldNodeRef, NewNodeRef) -> i32,
    {
        // Compute the best score matching `old_index` with an equivalent not-ready new
        // node.
        let mut best = 0i32;
        log::trace!("Computing old score for {:?}", old_index);
        for &new_equiv in equivalents
            .get_equivalent_new_nodes(old_index)
            .iter()
            .filter(|&x| is_unready(*x))
        {
            let score = get_match_score(old_index, new_equiv);
            log::trace!(
                "Considering new equivalent {:?}, score={}",
                new_equiv,
                score
            );
            if score > best {
                log::trace!("New best score: {}", score);
                best = score;
            }
        }
        log::trace!("Best old score for {:?}: {}", old_index, best);
        return best;
    }
    fn compute_new_score<IsUnreadyF, GetScoreF>(
        &self,
        new_index: NewNodeRef,
        equivalents: &EquivalentNodeSet,
        is_unready: IsUnreadyF,
        get_match_score: GetScoreF,
    ) -> i32
    where
        IsUnreadyF: Fn(OldNodeRef) -> bool,
        GetScoreF: Fn(OldNodeRef, NewNodeRef) -> i32,
    {
        // Compute the best score matching `new_index` with an equivalent not-ready old
        // node.
        let mut best = 0i32;
        log::trace!("Computing new score for {:?}", new_index);
        for &old_equiv in equivalents
            .get_equivalent_old_nodes(new_index)
            .iter()
            .filter(|&x| is_unready(*x))
        {
            let score = get_match_score(old_equiv, new_index);
            log::trace!(
                "Considering old equivalent {:?}, score={}",
                old_equiv,
                score
            );
            if score > best {
                log::trace!("New best score: {}", score);
                best = score;
            }
        }
        log::trace!("Best new score for {:?}: {}", new_index, best);
        return best;
    }
    fn update_after_match(
        &mut self,
        old_index: OldNodeRef,
        new_index: NewNodeRef,
        equivalents: &EquivalentNodeSet,
        old_graph: &DepGraph<OldNodeRef>,
        new_graph: &DepGraph<NewNodeRef>,
    ) {
        // Invalidate scores of all new (old) equivalent nodes of the users of
        // `old_index` (`new_index`). These are the only nodes which might
        // change by this matching.
        for old_user in old_graph.get_node(old_index).users.iter() {
            for new_equiv in equivalents.get_equivalent_new_nodes(old_user.0) {
                self.new_scores.borrow_mut().set(*new_equiv, None);
            }
        }
        for new_user in new_graph.get_node(new_index).users.iter() {
            for old_equiv in equivalents.get_equivalent_old_nodes(new_user.0) {
                self.old_scores.borrow_mut().set(*old_equiv, None);
            }
        }
    }
    fn update_after_adding_ready_old_node(
        &mut self,
        added_node: OldNodeRef,
        equivalents: &EquivalentNodeSet,
    ) {
        // `added_node` is now ready and thus is no longer in the set of unready nodes.
        // Invalidate all equivalent new nodes.
        for new_equiv in equivalents.get_equivalent_new_nodes(added_node) {
            self.new_scores.borrow_mut().set(*new_equiv, None);
        }
    }
    fn update_after_adding_ready_new_node(
        &mut self,
        added_node: NewNodeRef,
        equivalents: &EquivalentNodeSet,
    ) {
        // `added_node` is now ready and thus is no longer in the set of unready nodes.
        // Invalidate all equivalent old nodes.
        for old_equiv in equivalents.get_equivalent_old_nodes(added_node) {
            self.old_scores.borrow_mut().set(*old_equiv, None);
        }
    }
}

/// Greedy match selector which tries to produce a minimal-sized graph edit.
pub struct GreedyMatchSelector {
    old_graph: DepGraph<OldNodeRef>,
    new_graph: DepGraph<NewNodeRef>,
    /// Per-node equivalence sets derived from structural hashes.
    equivalents: EquivalentNodeSet,
    /// Reverse-direction similarity scores for candidate node pairs.
    reverse_scores: NodePairMap<i32>,
    forward_score_cache: ForwardScoreCache,
    unready_match_score_cache: UnreadyMatchScoreCache,

    ready_old: NodeVector<OldNodeRef, bool>,
    ready_new: NodeVector<NewNodeRef, bool>,
    /// Mapping from matched old node index to new node index.
    matched_old_to_new: NodeVector<OldNodeRef, Option<NewNodeRef>>,
    /// Nodes that have been handled (matched or added/removed) in old/new.
    handled_old: NodeVector<OldNodeRef, bool>,
    handled_new: NodeVector<NewNodeRef, bool>,
}

impl GreedyMatchSelector {
    pub fn new(old: &crate::ir::Fn, new: &crate::ir::Fn) -> Self {
        let old_graph = build_dependency_graph::<OldNodeRef>(old);
        let new_graph = build_dependency_graph::<NewNodeRef>(new);
        let equivalents = EquivalentNodeSet::new(&old_graph, &new_graph);
        let reverse_scores = compute_reverse_match_scores(&old_graph, &new_graph);
        let old_len = old_graph.nodes.len();
        let new_len = new_graph.nodes.len();
        let forward_score_cache = ForwardScoreCache::new(&old_graph, &new_graph);
        let unready_match_score_cache = UnreadyMatchScoreCache::new(&old_graph, &new_graph);
        Self {
            old_graph,
            new_graph,
            equivalents,
            reverse_scores,
            forward_score_cache,
            unready_match_score_cache,
            ready_old: NodeVector::new(old_len, false),
            ready_new: NodeVector::new(new_len, false),
            matched_old_to_new: NodeVector::new(old_len, None),
            handled_old: NodeVector::new(old_len, false),
            handled_new: NodeVector::new(new_len, false),
        }
    }
    /// Returns the best score for any match which includes nodes a or b (which
    /// is not the match(a, b)) and for which the match is not yet ready. This
    /// is some measure of the opportunity cost of matching a with b at the
    /// moment.
    ///
    /// best_unready_match_score(a, None) and best_unready_match_score(None, b)
    /// are the opportunity costs of matching a (b) with any non ready node.
    fn best_unready_match_score(&self, a: Option<OldNodeRef>, b: Option<NewNodeRef>) -> i32 {
        let is_new_unready = |x: NewNodeRef| !*self.ready_new.get(x) && !*self.handled_new.get(x);
        let is_old_unready = |x: OldNodeRef| !*self.ready_old.get(x) && !*self.handled_old.get(x);
        let match_score = |a: OldNodeRef, b: NewNodeRef| self.match_score(&a, &b);
        match (a, b) {
            (Some(a_idx), Some(b_idx)) => {
                assert!(*self.ready_old.get(a_idx));
                assert!(*self.ready_new.get(b_idx));
                let best_old_score = self.unready_match_score_cache.get_old_score(
                    a_idx,
                    &self.equivalents,
                    is_new_unready,
                    match_score,
                );
                let best_new_score = self.unready_match_score_cache.get_new_score(
                    b_idx,
                    &self.equivalents,
                    is_old_unready,
                    match_score,
                );
                std::cmp::max(best_old_score, best_new_score)
            }
            (Some(a_idx), None) => {
                assert!(*self.ready_old.get(a_idx));
                self.unready_match_score_cache.get_old_score(
                    a_idx,
                    &self.equivalents,
                    is_new_unready,
                    match_score,
                )
            }
            (None, Some(b_idx)) => {
                assert!(*self.ready_new.get(b_idx));
                self.unready_match_score_cache.get_new_score(
                    b_idx,
                    &self.equivalents,
                    is_old_unready,
                    match_score,
                )
            }
            (None, None) => unreachable!(
                "best_unready_match_score called with (None, None); at least one side must be Some"
            ),
        }
    }

    /// Score for matching a ready old node `a` with a ready new node `b`.
    fn match_score(&self, a: &OldNodeRef, b: &NewNodeRef) -> i32 {
        let fwd = self.forward_score_cache.get(*a, *b);
        let fwd = if fwd == SCORE_BASE {
            PERFECT_FORWARD_MATCH_SCORE
        } else {
            fwd
        };
        let rev = self.reverse_scores.get(*a, *b).copied().unwrap_or(0);
        let rev = if rev == SCORE_BASE {
            PERFECT_REVERSE_MATCH_SCORE
        } else {
            rev
        };
        fwd + rev
    }
    fn net_match_score(&self, a: &OldNodeRef, b: &NewNodeRef) -> i32 {
        let score = self.match_score(a, b);
        score - self.best_unready_match_score(Some(*a), Some(*b))
    }

    fn delete_node_score(&self, _a: &OldNodeRef) -> i32 {
        0
    }
    fn net_delete_node_score(&self, a: &OldNodeRef) -> i32 {
        self.delete_node_score(a) - self.best_unready_match_score(Some(*a), None)
    }

    fn add_node_score(&self, _b: &NewNodeRef) -> i32 {
        0
    }
    fn net_add_node_score(&self, b: &NewNodeRef) -> i32 {
        self.add_node_score(b) - self.best_unready_match_score(None, Some(*b))
    }

    /// Build a MatchNodes action for (a, b).
    fn build_match_action(&self, a: &OldNodeRef, b: &NewNodeRef) -> MatchAction {
        let new_operands = self
            .new_graph
            .get_node(*b)
            .operands
            .iter()
            .copied()
            .collect::<Vec<NewNodeRef>>();
        MatchAction::MatchNodes {
            old_index: *a,
            new_index: *b,
            new_operands,
            is_new_return: *b == self.new_graph.return_value,
        }
    }

    /// Build a DeleteNode action for old node `a`.
    fn build_delete_node_action(&self, a: &OldNodeRef) -> MatchAction {
        MatchAction::DeleteNode { old_index: *a }
    }

    /// Build an AddNode action for new node `b`.
    fn build_add_node_action(&self, b: &NewNodeRef) -> MatchAction {
        MatchAction::AddNode {
            new_index: *b,
            is_return: *b == self.new_graph.return_value,
        }
    }

    fn select_best_action(&mut self) -> Option<(i32, MatchAction)> {
        // TODO(meheff): There is probably an order of magnitude speedup to be had by
        // keeping an ordered queue of match actions rather than iterating through all
        // possible actions every time.

        if !(0..self.ready_old.len()).any(|i| *self.ready_old.get(OldNodeRef(i)))
            && !(0..self.ready_new.len()).any(|i| *self.ready_new.get(NewNodeRef(i)))
        {
            return None;
        }

        let mut best_action: Option<MatchAction> = None;
        let mut best_score: i32 = i32::MIN;

        // 1) Consider match actions for each ready old against compatible ready new.
        let ready_old_sorted: Vec<OldNodeRef> = (0..self.ready_old.len())
            .filter(|&i| *self.ready_old.get(OldNodeRef(i)))
            .map(OldNodeRef)
            .collect();
        for a in ready_old_sorted.into_iter() {
            let news = self.equivalents.get_equivalent_new_nodes(a);
            for &b in news.iter() {
                if !*self.ready_new.get(b) {
                    continue;
                }
                let score = self.net_match_score(&a, &b);
                trace!(
                    "Considering match action: {} <-> {}, score={}",
                    self.old_graph.get_node(a).name,
                    self.new_graph.get_node(b).name,
                    score
                );
                if score > best_score {
                    trace!("New best score: {}", score);
                    best_score = score;
                    best_action = Some(self.build_match_action(&a, &b));
                }
            }
        }

        // 2) Consider deleting a node from the old graph.
        let ready_old_sorted: Vec<OldNodeRef> = (0..self.ready_old.len())
            .filter(|&i| *self.ready_old.get(OldNodeRef(i)))
            .map(OldNodeRef)
            .collect();
        for a in ready_old_sorted.into_iter() {
            let score = self.net_delete_node_score(&a);
            trace!(
                "Considering delete node action: {}, score={}",
                self.old_graph.get_node(a).name,
                score,
            );
            if score > best_score {
                trace!("New best score: {}", score);
                best_score = score;
                best_action = Some(self.build_delete_node_action(&a));
            }
        }

        // 3) Consider adding a node to the new graph.
        let ready_new_sorted: Vec<NewNodeRef> = (0..self.ready_new.len())
            .filter(|&i| *self.ready_new.get(NewNodeRef(i)))
            .map(NewNodeRef)
            .collect();
        for b in ready_new_sorted.into_iter() {
            let score = self.net_add_node_score(&b);
            trace!(
                "Considering add node action: {}, score={}",
                self.new_graph.get_node(b).name,
                score,
            );
            if score > best_score {
                trace!("New best score: {}", score);
                best_score = score;
                best_action = Some(self.build_add_node_action(&b));
            }
        }
        assert!(best_action.is_some());
        Some((best_score, best_action.unwrap()))
    }
}

impl MatchSelector for GreedyMatchSelector {
    fn add_ready_node(&mut self, node: ReadyNode) {
        trace!(
            "Adding ready node: {:?}",
            match node.side {
                NodeSide::Old => format!("old:{}", self.old_graph.nodes[node.index].name),
                NodeSide::New => format!("new:{}", self.new_graph.nodes[node.index].name),
            }
        );
        match node.side {
            NodeSide::Old => {
                self.ready_old.set(OldNodeRef(node.index), true);
                self.unready_match_score_cache
                    .update_after_adding_ready_old_node(OldNodeRef(node.index), &self.equivalents);
            }
            NodeSide::New => {
                self.ready_new.set(NewNodeRef(node.index), true);
                self.unready_match_score_cache
                    .update_after_adding_ready_new_node(NewNodeRef(node.index), &self.equivalents);
            }
        }
    }

    fn select_next_match(&mut self) -> Option<MatchAction> {
        trace!("Selecting next match");
        if log_enabled!(Level::Debug) {
            let ready_old_names: Vec<String> = (0..self.ready_old.len())
                .filter(|&i| *self.ready_old.get(OldNodeRef(i)))
                .map(|i| self.old_graph.get_node(OldNodeRef(i)).name.clone())
                .collect();
            let ready_new_names: Vec<String> = (0..self.ready_new.len())
                .filter(|&i| *self.ready_new.get(NewNodeRef(i)))
                .map(|i| self.new_graph.get_node(NewNodeRef(i)).name.clone())
                .collect();
            trace!("Ready old: [{}]", ready_old_names.join(", "));
            trace!("Ready new: [{}]", ready_new_names.join(", "));
        }
        match self.select_best_action() {
            Some((score, action)) => {
                debug!(
                    "best_action (score={}): {}",
                    score,
                    crate::matching_ged::format_match_action(&action, |side, idx| match side {
                        NodeSide::Old =>
                            format!("old:{}", self.old_graph.get_node(OldNodeRef(idx)).name),
                        NodeSide::New =>
                            format!("new:{}", self.new_graph.get_node(NewNodeRef(idx)).name),
                    })
                );
                Some(action)
            }
            None => {
                debug!("No best action found");
                None
            }
        }
    }

    fn update_after_match(&mut self, edit: &MatchAction) {
        match edit {
            MatchAction::DeleteNode { old_index } => {
                self.ready_old.set(*old_index, false);
                self.handled_old.set(*old_index, true);
                self.equivalents.drop_old_node(*old_index);
            }
            MatchAction::AddNode { new_index, .. } => {
                self.ready_new.set(*new_index, false);
                self.handled_new.set(*new_index, true);
                self.equivalents.drop_new_node(*new_index);
            }
            MatchAction::MatchNodes {
                old_index,
                new_index,
                ..
            } => {
                self.ready_old.set(*old_index, false);
                self.ready_new.set(*new_index, false);
                self.matched_old_to_new.set(*old_index, Some(*new_index));
                self.handled_old.set(*old_index, true);
                self.handled_new.set(*new_index, true);
                self.forward_score_cache.update_after_match(
                    *old_index,
                    *new_index,
                    &self.equivalents,
                    &self.matched_old_to_new,
                    &self.old_graph,
                    &self.new_graph,
                );
                self.unready_match_score_cache.update_after_match(
                    *old_index,
                    *new_index,
                    &self.equivalents,
                    &self.old_graph,
                    &self.new_graph,
                );
                self.equivalents.drop_old_node(*old_index);
                self.equivalents.drop_new_node(*new_index);
            }
        }
    }
}

/// Computes reverse-direction similarity scores M(A,B) for compatible node
/// pairs.
fn compute_reverse_match_scores(
    old_graph: &DepGraph<OldNodeRef>,
    new_graph: &DepGraph<NewNodeRef>,
) -> NodePairMap<i32> {
    // Helper: local shape compatibility via stored structural hash.
    let shapes_equal = |oi: OldNodeRef, ni: NewNodeRef| {
        old_graph.get_node(oi).structural_hash == new_graph.get_node(ni).structural_hash
    };

    // Scores map: best-known M(a,b) so far; None means 0 (unseen).
    let mut m_scores: NodePairMap<i32> =
        NodePairMap::new(old_graph.nodes.len(), new_graph.nodes.len(), None);

    // Worklist seeded with all compatible sink pairs (nodes with no users).
    let mut worklist: VecDeque<(OldNodeRef, NewNodeRef)> = VecDeque::new();
    let mut on_worklist: HashSet<(OldNodeRef, NewNodeRef)> = HashSet::new();
    // Just seed the worklist with the return values.
    worklist.push_back((old_graph.return_value, new_graph.return_value));
    on_worklist.insert((old_graph.return_value, new_graph.return_value));

    let recompute_score = |a: OldNodeRef, b: NewNodeRef, m: &NodePairMap<i32>| -> i32 {
        if !shapes_equal(a, b) {
            return 0;
        }
        let z = std::cmp::max(
            old_graph.get_node(a).users.len(),
            new_graph.get_node(b).users.len(),
        );
        if z == 0 {
            return SCORE_BASE;
        }
        let mut sum: i32 = 0;
        for &(u_a, slot) in old_graph.get_node(a).users.iter() {
            let mut best: i32 = 0;
            for &(u_b, slot_b) in new_graph.get_node(b).users.iter() {
                if slot_b != slot {
                    continue;
                }
                if !shapes_equal(u_a, u_b) {
                    continue;
                }
                if let Some(&val) = m.get(u_a, u_b) {
                    if val > best {
                        best = val;
                    }
                }
            }
            sum += best;
        }
        sum / (z as i32)
    };

    // Process worklist: on improvement, propagate to operand pairs.
    while let Some((a, b)) = worklist.pop_front() {
        // Mark as no longer on the worklist so it can be re-enqueued upon future
        // improvements.
        on_worklist.remove(&(a, b));
        let new_score = recompute_score(a, b, &m_scores);
        let old_score = m_scores.get(a, b).copied().unwrap_or(0);
        if new_score > old_score {
            m_scores.set(a, b, new_score);

            // Walk operands in lockstep; if counts match, enqueue compatible pairs.
            let a_ops: Vec<OldNodeRef> = old_graph.get_node(a).operands.iter().copied().collect();
            let b_ops: Vec<NewNodeRef> = new_graph.get_node(b).operands.iter().copied().collect();
            if a_ops.len() == b_ops.len() {
                for (op_a, op_b) in a_ops.iter().zip(b_ops.iter()) {
                    if shapes_equal(*op_a, *op_b) && !on_worklist.contains(&(*op_a, *op_b)) {
                        worklist.push_back((*op_a, *op_b));
                        on_worklist.insert((*op_a, *op_b));
                    }
                }
            }
        }
    }

    m_scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodePayload;
    use crate::ir_parser::Parser;
    use crate::matching_ged::{
        IrEdit, MatchAction, NewNodeRef, NodeSide, OldNodeRef, ReadyNode, apply_fn_edits,
        compute_fn_edit,
    };

    fn parse_ir_from_string(s: &str) -> crate::ir::Package {
        let mut parser = Parser::new(s);
        parser.parse_and_validate_package().unwrap()
    }

    #[test]
    fn compute_returns_adds_and_deletes_for_all_nodes() {
        let pkg = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8]) -> bits[8] {
                ret identity.2: bits[8] = identity(x, id=2)
            }
            "#,
        );
        let lhs = pkg.get_top_fn().unwrap();

        let pkg2 = parse_ir_from_string(
            r#"package p2
            top fn f2(x: bits[8]) -> bits[8] {
                ret identity.2: bits[8] = identity(x, id=2)
            }
            "#,
        );
        let rhs = pkg2.get_top_fn().unwrap();

        // Use GreedyMatchSelector-based matcher
        let mut selector = GreedyMatchSelector::new(lhs, rhs);
        let edits = compute_fn_edit(lhs, rhs, &mut selector).unwrap();
        assert!(edits.edits.is_empty());
        let patched = apply_fn_edits(lhs, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, rhs
        ));
    }

    #[test]
    fn compute_errors_on_mismatched_signatures() {
        let pkg = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8]) -> bits[8] {
                ret identity.2: bits[8] = identity(x, id=2)
            }
            "#,
        );
        let lhs = pkg.get_top_fn().unwrap();

        let pkg2 = parse_ir_from_string(
            r#"package p2
            top fn f2(x: bits[9]) -> bits[9] {
                ret identity.2: bits[9] = identity(x, id=2)
            }
            "#,
        );
        let rhs = pkg2.get_top_fn().unwrap();

        let mut selector = GreedyMatchSelector::new(lhs, rhs);
        let result = compute_fn_edit(lhs, rhs, &mut selector);
        assert!(result.is_err());
    }

    #[test]
    fn apply_edits_literal_change_matches_new() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[8] {
                ret literal.1: bits[8] = literal(value=0, id=1)
            }
            "#,
        );
        let old_fn = pkg_old.get_top_fn().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[8] {
                ret literal.101: bits[8] = literal(value=1, id=101)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
    }

    #[test]
    fn apply_edits_add_identity_of_literal() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                ret literal.1: bits[1] = literal(value=0, id=1)
            }
            "#,
        );
        let old_fn = pkg_old.get_top_fn().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                literal.11: bits[1] = literal(value=1, id=11)
                ret identity.22: bits[1] = identity(literal.11, id=22)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
    }

    #[test]
    fn edit_distance_handles_different_return_nodes() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                literal.1: bits[1] = literal(value=1, id=1)
                ret identity.2: bits[1] = identity(literal.1, id=2)
            }
            "#,
        );
        let old_fn = pkg_old.get_top_fn().unwrap();

        // Same nodes, but return points to a different node (the literal).
        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f() -> bits[1] {
                ret literal.1: bits[1] = literal(value=1, id=1)
                identity.2: bits[1] = identity(literal.1, id=2)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
    }

    #[test]
    fn edit_distance_multiple_parameters_identity_wrap() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
                ret add.3: bits[8] = add(a, b, id=3)
            }
            "#,
        );
        let old_fn = pkg_old.get_top_fn().unwrap();

        // Different structure: wrap the add in an identity, keep params identical.
        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
                add.103: bits[8] = add(a, b, id=103)
                ret identity.104: bits[8] = identity(add.103, id=104)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();
        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
    }

    #[test]
    fn single_operand_substitution_between_adds() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8] id=1, y: bits[8] id=2, z: bits[8] id=3) -> bits[8] {
                ret add.10: bits[8] = add(x, y, id=10)
            }
            "#,
        );
        let old_fn = pkg_old.get_top_fn().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8] id=1, y: bits[8] id=2, z: bits[8] id=3) -> bits[8] {
                ret add.20: bits[8] = add(x, z, id=20)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();

        assert_eq!(
            edits.edits.len(),
            1,
            "expected exactly one edit, got {:?}",
            edits.edits
        );
        // Verify the edit substitutes to the parameter named 'z'.
        match edits.edits[0] {
            IrEdit::SubstituteOperand { new_operand, .. } => {
                let target_node = old_fn.get_node(new_operand);
                assert_eq!(target_node.name.as_deref(), Some("z"));
                assert!(matches!(target_node.payload, NodePayload::GetParam(_)));
            }
            ref other => panic!(
                "expected edit to be an operand substitution, got {:?}",
                other
            ),
        }
        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
    }

    #[test]
    fn multiple_operand_substitutions_in_concat_permutation() {
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8] id=1, y: bits[8] id=2, z: bits[8] id=3) -> bits[24] {
                ret concat.10: bits[24] = concat(x, y, z, id=10)
            }
            "#,
        );
        let old_fn = pkg_old.get_top_fn().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p
            top fn f(x: bits[8] id=1, y: bits[8] id=2, z: bits[8] id=3) -> bits[24] {
                ret concat.20: bits[24] = concat(z, x, y, id=20)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
        let edits = compute_fn_edit(old_fn, new_fn, &mut selector).unwrap();

        // Expect exactly three operand substitutions, and patched result is isomorphic.
        assert_eq!(
            edits.edits.len(),
            3,
            "expected exactly three edits, got {:?}",
            edits.edits
        );
        assert!(
            edits
                .edits
                .iter()
                .all(|e| matches!(e, IrEdit::SubstituteOperand { .. })),
            "expected only SubstituteOperand edits, got {:?}",
            edits.edits
        );

        let patched = apply_fn_edits(old_fn, &edits).unwrap();
        assert!(crate::node_hashing::functions_structurally_equivalent(
            &patched, new_fn
        ));
    }

    #[test]
    fn reverse_matches_three_adds_under_or_with_param_permutation() {
        // Old: or(or(A(w,x), B(x,y)), C(y,z))
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(w: bits[8] id=1, x: bits[8] id=2, y: bits[8] id=3, z: bits[8] id=4) -> bits[8] {
                a: bits[8] = add(w, x, id=10)
                b: bits[8] = add(x, y, id=11)
                c: bits[8] = add(y, z, id=12)
                foo: bits[8] = or(a, b, id=13)
                ret bar: bits[8] = or(foo, b, c, id=14)
            }
            "#,
        );
        let old_fn = pkg_old.get_top_fn().unwrap();

        // New: or(or(A'(w,x), C'(y,z)), B'(x,y)) — A stays in inner slot0; B,C swap
        // levels.
        let pkg_new = parse_ir_from_string(
            r#"package p2
            top fn f2(w: bits[8] id=1, x: bits[8] id=2, y: bits[8] id=3, z: bits[8] id=4) -> bits[8] {
                a: bits[8] = add(w, x, id=110)
                b: bits[8] = add(x, y, id=111)
                c: bits[8] = add(y, z, id=112)
                foo: bits[8] = or(a, c, id=113)
                ret bar: bits[8] = or(foo, b, b, id=114)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        // Build dependency graphs and compute reverse matches.
        let old_graph = crate::matching_ged::build_dependency_graph::<OldNodeRef>(old_fn);
        let new_graph = crate::matching_ged::build_dependency_graph::<NewNodeRef>(new_fn);
        let reverse = compute_reverse_match_scores(&old_graph, &new_graph);
        let base = SCORE_BASE;

        // Helper to get node indices by name.
        let find_named = |f: &crate::ir::Fn, name: &str| -> usize {
            f.nodes
                .iter()
                .position(|n| n.name.as_deref() == Some(name))
                .expect("node by name not found")
        };
        // Helper to read reverse score by node names, or 0 if missing.
        // Panics if nodes are not found.
        // Usage: rev("old_name", "new_name")
        let rev = |old_name: &str, new_name: &str| -> i32 {
            let oi = find_named(old_fn, old_name);
            let ni = find_named(new_fn, new_name);
            reverse
                .get(OldNodeRef(oi), NewNodeRef(ni))
                .copied()
                .unwrap_or(0)
        };

        assert_eq!(rev("bar", "bar"), base);
        assert_eq!(rev("foo", "bar"), 0);
        assert_eq!(rev("bar", "foo"), 0);
        assert_eq!(rev("a", "a"), base);
        assert_eq!(rev("b", "b"), base / 2);
        assert_eq!(rev("c", "c"), 0);
        assert_eq!(rev("w", "w"), base);
        assert_eq!(rev("x", "x"), 3 * base / 4);
        assert_eq!(rev("y", "y"), base / 4);
        assert_eq!(rev("z", "z"), 0);
    }

    #[test]
    fn forward_match_and_opportunity_costs_behave_as_expected() {
        // Reuse the same IR structure as the reverse-match test.
        let pkg_old = parse_ir_from_string(
            r#"package p
            top fn f(w: bits[8] id=1, x: bits[8] id=2, y: bits[8] id=3, z: bits[8] id=4) -> bits[8] {
                a: bits[8] = add(w, x, id=10)
                b: bits[8] = add(x, y, id=11)
                c: bits[8] = add(x, z, id=12)
                ret foo: bits[8] = or(a, b, c, id=13)
            }
            "#,
        );
        let old_fn = pkg_old.get_top_fn().unwrap();

        let pkg_new = parse_ir_from_string(
            r#"package p2
            top fn f2(w: bits[8] id=1, x: bits[8] id=2, y: bits[8] id=3, z: bits[8] id=4) -> bits[8] {
                a: bits[8] = add(w, x, id=110)
                not_y: bits[8] = not(y, id=111)
                b: bits[8] = add(x, not_y, id=112)
                c: bits[8] = add(x, z, id=114)
                ret foo: bits[8] = or(a, b, c, id=115)
            }
            "#,
        );
        let new_fn = pkg_new.get_top_fn().unwrap();

        let mut sel = GreedyMatchSelector::new(old_fn, new_fn);
        let base = SCORE_BASE;
        let perf_fwd = PERFECT_FORWARD_MATCH_SCORE;
        let perf_rev = PERFECT_REVERSE_MATCH_SCORE;

        // Helper to get node indices by name.
        let find_named = |f: &crate::ir::Fn, name: &str| -> usize {
            f.nodes
                .iter()
                .position(|n| n.name.as_deref() == Some(name))
                .expect("node by name not found")
        };

        // Explicitly pre-match parameters by name using the MatchSelector API.
        for pname in ["w", "x", "y", "z"].iter() {
            let oi = find_named(old_fn, pname);
            let ni = find_named(new_fn, pname);
            sel.add_ready_node(ReadyNode {
                side: NodeSide::Old,
                index: oi,
            });
            sel.add_ready_node(ReadyNode {
                side: NodeSide::New,
                index: ni,
            });
            let action = MatchAction::MatchNodes {
                old_index: OldNodeRef(oi),
                new_index: NewNodeRef(ni),
                new_operands: Vec::new(),
                is_new_return: false,
            };
            sel.update_after_match(&action);
        }
        for n in ["a", "b", "c"].iter() {
            sel.add_ready_node(ReadyNode {
                side: NodeSide::Old,
                index: find_named(old_fn, n),
            });
        }
        for n in ["a", "c", "not_y"].iter() {
            sel.add_ready_node(ReadyNode {
                side: NodeSide::New,
                index: find_named(new_fn, n),
            });
        }

        // Forward match scores on adds should be perfect under param mapping.
        let match_score = |old_name: &str, new_name: &str| -> i32 {
            let oi = find_named(old_fn, old_name);
            let ni = find_named(new_fn, new_name);
            sel.match_score(&OldNodeRef(oi), &NewNodeRef(ni))
        };
        assert_eq!(match_score("a", "a"), perf_fwd + perf_rev);
        assert_eq!(match_score("b", "b"), perf_rev + base / 2);
        assert_eq!(match_score("c", "c"), perf_fwd + perf_rev);
        assert_eq!(match_score("foo", "foo"), perf_rev);

        // Helper: opportunity cost via names (use None to omit a side).
        let best_unready = |a: Option<&str>, b: Option<&str>| -> i32 {
            sel.best_unready_match_score(
                a.map(|n| OldNodeRef(find_named(old_fn, n))),
                b.map(|n| NewNodeRef(find_named(new_fn, n))),
            )
        };
        assert_eq!(best_unready(Some("a"), Some("a")), 0);
        assert_eq!(best_unready(Some("c"), Some("c")), base / 2);
        assert_eq!(best_unready(Some("a"), Some("c")), 0);
        assert_eq!(best_unready(Some("b"), Some("a")), perf_rev + base / 2);

        // Helpers for name-based selector scores.
        let match_score = |old_name: &str, new_name: &str| -> i32 {
            let oi = OldNodeRef(find_named(old_fn, old_name));
            let ni = NewNodeRef(find_named(new_fn, new_name));
            sel.match_score(&oi, &ni)
        };
        assert_eq!(match_score("a", "a"), perf_fwd + perf_rev);
        assert_eq!(match_score("b", "c"), base / 2);
        assert_eq!(match_score("c", "c"), perf_fwd + perf_rev);
    }
}
