// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering as CmpOrdering;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use blake3::Hasher;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use xlsynth::{IrBits, IrValue};
use xlsynth_pir::corners::{CornerEvent, CornerKind, FailureEvent};
use xlsynth_pir::ir;
use xlsynth_pir::ir_eval::{BoolNodeEvent, EvalObserver, FnEvalResult, SelectEvent, SelectKind};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_value_utils::{ir_bits_from_value_with_type, ir_value_from_bits_with_type};

pub const FEATURE_MAP_SIZE: usize = 65_536;
const FEATURE_MAP_BYTES: usize = FEATURE_MAP_SIZE / 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CornerEventId {
    pub node_text_id: usize,
    pub kind: CornerKind,
    pub tag: u8,
}

pub fn corner_tag_description(kind: CornerKind, tag: u8) -> String {
    match kind {
        CornerKind::Add => match tag {
            0 => "add lhs is zero".to_string(),
            1 => "add rhs is zero".to_string(),
            _ => panic!("Add defines tags {{0,1}}; got {}", tag),
        },
        CornerKind::Neg => match tag {
            0 => "neg operand is min signed (e.g. 0x80..00)".to_string(),
            1 => "neg operand msb is one (negative)".to_string(),
            _ => panic!("Neg defines tags {{0,1}}; got {}", tag),
        },
        CornerKind::SignExt => {
            assert_eq!(tag, 0, "SignExt only defines tag=0");
            "sign_ext: msb is zero (extending with zeros)".to_string()
        }
        CornerKind::DynamicBitSlice => match tag {
            0 => "dynamic_bit_slice in-bounds".to_string(),
            1 => "dynamic_bit_slice out-of-bounds".to_string(),
            _ => panic!("DynamicBitSlice defines tags {{0,1}}; got {}", tag),
        },
        CornerKind::ArrayIndex => match tag {
            0 => "array_index in-bounds".to_string(),
            1 => "array_index clamped (index out-of-bounds)".to_string(),
            _ => panic!("ArrayIndex defines tags {{0,1}}; got {}", tag),
        },
        CornerKind::Shift => match tag {
            0 => "shift amount is zero".to_string(),
            1 => "shift amount < lhs bit width".to_string(),
            2 => "shift amount >= lhs bit width".to_string(),
            _ => panic!("Shift defines tags {{0,1,2}}; got {}", tag),
        },
        CornerKind::Shra => match tag {
            0 => "shra: msb=0 and shift amount < width".to_string(),
            1 => "shra: msb=0 and shift amount >= width".to_string(),
            2 => "shra: msb=1 and shift amount < width".to_string(),
            3 => "shra: msb=1 and shift amount >= width".to_string(),
            _ => panic!("Shra defines tags 0..=3; got {}", tag),
        },
        CornerKind::CompareDistance => match tag {
            0 => "eq/ne compare distance: xor-popcount == 0 (equal)".to_string(),
            1 => "eq/ne compare distance: xor-popcount == 1".to_string(),
            2 => "eq/ne compare distance: xor-popcount == 2".to_string(),
            3 => "eq/ne compare distance: xor-popcount == 3".to_string(),
            4 => "eq/ne compare distance: xor-popcount == 4".to_string(),
            5 => "eq/ne compare distance: xor-popcount in 5..=8".to_string(),
            6 => "eq/ne compare distance: xor-popcount in 9..=16".to_string(),
            7 => "eq/ne compare distance: xor-popcount >= 17".to_string(),
            _ => panic!("CompareDistance defines tags 0..=7; got {}", tag),
        },
    }
}

fn max_value_for_bit_width(w: usize) -> Option<u128> {
    if w >= 128 {
        return None;
    }
    if w == 0 {
        return Some(0);
    }
    Some((1u128 << w) - 1u128)
}

fn bit_width_if_bits(ty: &ir::Type) -> Option<usize> {
    match ty {
        ir::Type::Bits(w) => Some(*w),
        _ => None,
    }
}

impl PartialOrd for CornerEventId {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for CornerEventId {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        (self.node_text_id, self.kind as u8, self.tag).cmp(&(
            other.node_text_id,
            other.kind as u8,
            other.tag,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct AutocovConfig {
    pub seed: u64,
    pub max_iters: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct AutocovReport {
    pub iters: u64,
    pub corpus_len: usize,
    pub mux_features_set: usize,
    pub path_features_set: usize,
    pub bools_features_set: usize,
    pub corner_features_set: usize,
    pub compare_distance_features_set: usize,
    pub failure_features_set: usize,
    pub mux_outcomes_observed: usize,
    pub mux_outcomes_possible: usize,
    pub mux_outcomes_missing: usize,
}

#[derive(Debug, Clone)]
pub struct MuxOutcomeReportEntry {
    pub node_text_id: usize,
    pub kind: MuxNodeKind,
    pub observed_count: usize,
    pub possible_count: usize,
    pub missing: Vec<MuxOutcomeId>,
}

#[derive(Debug, Clone)]
pub struct MuxOutcomeReport {
    pub entries: Vec<MuxOutcomeReportEntry>,
    pub total_missing: usize,
    pub total_possible: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct BitConstant {
    bit_count: usize,
    value_u64: u64,
}

#[derive(Debug, Clone)]
pub struct CandidateObservation {
    pub ok: bool,
    pub mux_indices: Vec<usize>,
    pub path_index: usize,
    pub bools_index: usize,
    pub corner_indices: Vec<usize>,
    pub compare_distance_indices: Vec<usize>,
    pub failure_indices: Vec<usize>,
    pub mux_outcomes: Vec<(usize, MuxOutcomeId)>,
}

pub trait CorpusSink {
    fn on_new_sample(&mut self, tuple_value: &IrValue);
}

#[derive(Debug, Clone, Copy)]
pub struct AutocovProgress {
    pub iters: u64,
    pub corpus_len: usize,
    pub mux_features_set: usize,
    pub path_features_set: usize,
    pub bools_features_set: usize,
    pub corner_features_set: usize,
    pub compare_distance_features_set: usize,
    pub failure_features_set: usize,
    pub mux_outcomes_observed: usize,
    pub mux_outcomes_possible: usize,
    pub mux_outcomes_missing: usize,
    pub last_iter_added: bool,
}

pub trait ProgressSink {
    fn on_progress(&mut self, p: AutocovProgress);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuxNodeKind {
    Sel,
    PrioritySel,
    OneHotSel,
}

#[derive(Debug, Clone)]
pub struct MuxNodeSpace {
    pub node_text_id: usize,
    pub kind: MuxNodeKind,
    pub cases_len: usize,
    pub has_default: bool,
    pub selector_bit_count: Option<usize>,
}

impl MuxNodeSpace {
    pub fn feature_possibilities(&self) -> usize {
        match self.kind {
            MuxNodeKind::Sel => {
                let w = self.selector_bit_count.unwrap_or(0);
                let space: usize = if w >= (usize::BITS as usize) {
                    usize::MAX
                } else {
                    1usize << w
                };
                let reachable_cases = std::cmp::min(self.cases_len, space);
                let out_of_range_possible = space > self.cases_len;
                let default_possible = out_of_range_possible;
                reachable_cases + (default_possible as usize)
            }
            MuxNodeKind::PrioritySel => {
                let w = self.selector_bit_count.unwrap_or(0);
                let reachable_cases = std::cmp::min(self.cases_len, w);
                reachable_cases + 1
            }
            MuxNodeKind::OneHotSel => {
                let w = self.selector_bit_count.unwrap_or(0);
                let reachable_cases = std::cmp::min(self.cases_len, w);
                reachable_cases + 1
            }
        }
    }

    pub fn log10_path_possibilities_upper_bound(&self) -> f64 {
        match self.kind {
            MuxNodeKind::Sel | MuxNodeKind::PrioritySel => {
                (self.feature_possibilities() as f64).log10()
            }
            MuxNodeKind::OneHotSel => {
                let w = self.selector_bit_count.unwrap_or(0);
                let reachable_cases = std::cmp::min(self.cases_len, w);
                (reachable_cases as f64) * std::f64::consts::LOG10_2
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MuxSpaceSummary {
    pub muxes: Vec<MuxNodeSpace>,
    pub total_mux_feature_possibilities: usize,
    pub log10_path_space_upper_bound: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MuxOutcomeId {
    /// Case index selected (for `sel`/`priority_sel`, or per-bit for
    /// `one_hot_sel`).
    CaseIndex(usize),
    /// Default selected (`sel` out-of-range, or `priority_sel` no bits set).
    Default,
    /// `one_hot_sel`: no in-range selector bits set.
    NoBitsSet,
    /// `one_hot_sel`: two or more in-range selector bits set.
    MultiBitsSet,
}

#[derive(Debug, Clone)]
pub struct MuxOutcomeSpace {
    pub node_text_id: usize,
    pub kind: MuxNodeKind,
    pub reachable_case_count: usize,
    pub has_default: bool,
    pub has_no_bits_set: bool,
    pub has_multi_bits_set: bool,
}

impl MuxOutcomeSpace {
    pub fn outcome_count(&self) -> usize {
        self.reachable_case_count
            + (self.has_default as usize)
            + (self.has_no_bits_set as usize)
            + (self.has_multi_bits_set as usize)
    }

    pub fn outcome_to_index(&self, o: MuxOutcomeId) -> Option<usize> {
        match o {
            MuxOutcomeId::CaseIndex(i) => {
                if i < self.reachable_case_count {
                    Some(i)
                } else {
                    None
                }
            }
            MuxOutcomeId::Default => {
                if self.has_default {
                    Some(self.reachable_case_count)
                } else {
                    None
                }
            }
            MuxOutcomeId::NoBitsSet => {
                if self.has_no_bits_set {
                    Some(self.reachable_case_count + (self.has_default as usize))
                } else {
                    None
                }
            }
            MuxOutcomeId::MultiBitsSet => {
                if self.has_multi_bits_set {
                    Some(
                        self.reachable_case_count
                            + (self.has_default as usize)
                            + (self.has_no_bits_set as usize),
                    )
                } else {
                    None
                }
            }
        }
    }

    pub fn index_to_outcome(&self, idx: usize) -> Option<MuxOutcomeId> {
        if idx < self.reachable_case_count {
            return Some(MuxOutcomeId::CaseIndex(idx));
        }
        let mut cur = self.reachable_case_count;
        if self.has_default {
            if idx == cur {
                return Some(MuxOutcomeId::Default);
            }
            cur += 1;
        }
        if self.has_no_bits_set {
            if idx == cur {
                return Some(MuxOutcomeId::NoBitsSet);
            }
            cur += 1;
        }
        if self.has_multi_bits_set && idx == cur {
            return Some(MuxOutcomeId::MultiBitsSet);
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MuxSelectKind {
    CaseIndex,
    Default,
    NoBitsSet,
    MultiBitsSet,
}

impl From<SelectKind> for MuxSelectKind {
    fn from(value: SelectKind) -> Self {
        match value {
            SelectKind::CaseIndex => MuxSelectKind::CaseIndex,
            SelectKind::Default => MuxSelectKind::Default,
            SelectKind::NoBitsSet => MuxSelectKind::NoBitsSet,
            SelectKind::MultiBitsSet => MuxSelectKind::MultiBitsSet,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MuxFeature {
    pub node_id: u32,
    pub select_kind: MuxSelectKind,
    pub selected_index: u16,
}

#[derive(Debug, Clone)]
struct FeatureMap64k {
    bytes: [u8; FEATURE_MAP_BYTES],
    set_count: usize,
}

impl FeatureMap64k {
    fn new() -> Self {
        Self {
            bytes: [0u8; FEATURE_MAP_BYTES],
            set_count: 0,
        }
    }

    fn observe_index(&mut self, idx: usize) -> bool {
        let byte_idx = idx >> 3;
        let bit_idx = idx & 7;
        let mask: u8 = 1u8 << bit_idx;
        let slot = &mut self.bytes[byte_idx];
        if (*slot & mask) == 0 {
            *slot |= mask;
            self.set_count += 1;
            true
        } else {
            false
        }
    }

    fn set_count(&self) -> usize {
        self.set_count
    }
}

#[derive(Debug)]
struct Observations {
    mux_new: bool,
    path_new: bool,
    bools_new: bool,
    corner_new: bool,
    compare_distance_new: bool,
    failure_new: bool,
}

#[derive(Debug, Clone)]
struct CandidateFeatures {
    mux_indices: Vec<usize>,
    path_index: usize,
    bools_index: usize,
    corner_indices: Vec<usize>,
    compare_distance_indices: Vec<usize>,
    failure_indices: Vec<usize>,
    mux_outcomes: Vec<(usize, MuxOutcomeId)>,
}

fn mux_feature_hash(f: &MuxFeature) -> blake3::Hash {
    let mut hasher = Hasher::new();
    hasher.update(b"xlsynth-autocov:mux");
    hasher.update(&f.node_id.to_le_bytes());
    hasher.update(&[match f.select_kind {
        MuxSelectKind::CaseIndex => 0,
        MuxSelectKind::Default => 1,
        MuxSelectKind::NoBitsSet => 2,
        MuxSelectKind::MultiBitsSet => 3,
    }]);
    hasher.update(&f.selected_index.to_le_bytes());
    hasher.finalize()
}

fn hash_mux_feature(f: &MuxFeature) -> usize {
    let h = mux_feature_hash(f);
    u16::from_le_bytes([h.as_bytes()[0], h.as_bytes()[1]]) as usize
}

fn hash_corner_event(ev: &CornerEvent) -> usize {
    let mut hasher = Hasher::new();
    hasher.update(b"xlsynth-autocov:corner");
    let node_id_u32 = u32::try_from(ev.node_text_id).unwrap_or(u32::MAX);
    hasher.update(&node_id_u32.to_le_bytes());
    hasher.update(&[ev.kind as u8]);
    hasher.update(&[ev.tag]);
    let h = hasher.finalize();
    u16::from_le_bytes([h.as_bytes()[0], h.as_bytes()[1]]) as usize
}

fn hash_compare_distance_event(ev: &CornerEvent) -> usize {
    debug_assert!(ev.kind == CornerKind::CompareDistance);
    let mut hasher = Hasher::new();
    hasher.update(b"xlsynth-autocov:compare-distance");
    let node_id_u32 = u32::try_from(ev.node_text_id).unwrap_or(u32::MAX);
    hasher.update(&node_id_u32.to_le_bytes());
    hasher.update(&[ev.tag]);
    let h = hasher.finalize();
    u16::from_le_bytes([h.as_bytes()[0], h.as_bytes()[1]]) as usize
}

fn hash_failure_event(ev: &FailureEvent) -> usize {
    let mut hasher = Hasher::new();
    hasher.update(b"xlsynth-autocov:failure");
    let node_id_u32 = u32::try_from(ev.node_text_id).unwrap_or(u32::MAX);
    hasher.update(&node_id_u32.to_le_bytes());
    hasher.update(&[ev.kind as u8]);
    hasher.update(&[ev.tag]);
    let h = hasher.finalize();
    u16::from_le_bytes([h.as_bytes()[0], h.as_bytes()[1]]) as usize
}

#[derive(Debug)]
struct CollectingMuxObserver {
    mux_indices: Vec<usize>,
    path_hasher: Hasher,
    bools_hasher: Hasher,
    corner_indices: Vec<usize>,
    compare_distance_indices: Vec<usize>,
    failure_indices: Vec<usize>,
    mux_outcomes: Vec<(usize, MuxOutcomeId)>,
}

impl CollectingMuxObserver {
    fn new() -> Self {
        let mut path_hasher = Hasher::new();
        path_hasher.update(b"xlsynth-autocov:path");
        let mut bools_hasher = Hasher::new();
        bools_hasher.update(b"xlsynth-autocov:bools");
        Self {
            mux_indices: Vec::new(),
            path_hasher,
            bools_hasher,
            corner_indices: Vec::new(),
            compare_distance_indices: Vec::new(),
            failure_indices: Vec::new(),
            mux_outcomes: Vec::new(),
        }
    }

    fn finish(self) -> CandidateFeatures {
        let path_hash = self.path_hasher.finalize();
        let path_index =
            u16::from_le_bytes([path_hash.as_bytes()[0], path_hash.as_bytes()[1]]) as usize;
        let bools_hash = self.bools_hasher.finalize();
        let bools_index =
            u16::from_le_bytes([bools_hash.as_bytes()[0], bools_hash.as_bytes()[1]]) as usize;
        CandidateFeatures {
            mux_indices: self.mux_indices,
            path_index,
            bools_index,
            corner_indices: self.corner_indices,
            compare_distance_indices: self.compare_distance_indices,
            failure_indices: self.failure_indices,
            mux_outcomes: self.mux_outcomes,
        }
    }
}

impl EvalObserver for CollectingMuxObserver {
    fn on_select(&mut self, ev: SelectEvent) {
        let selected_index_u16 = if ev.select_kind == SelectKind::CaseIndex {
            u16::try_from(ev.selected_index).unwrap_or(u16::MAX)
        } else {
            u16::MAX
        };

        let feature = MuxFeature {
            node_id: u32::try_from(ev.node_text_id).unwrap_or(u32::MAX),
            select_kind: ev.select_kind.into(),
            selected_index: selected_index_u16,
        };

        let idx = hash_mux_feature(&feature);
        self.mux_indices.push(idx);

        // Map the event into a node-local mux outcome.
        let outcome = match ev.select_kind {
            SelectKind::CaseIndex => MuxOutcomeId::CaseIndex(ev.selected_index),
            SelectKind::Default => MuxOutcomeId::Default,
            SelectKind::NoBitsSet => MuxOutcomeId::NoBitsSet,
            SelectKind::MultiBitsSet => MuxOutcomeId::MultiBitsSet,
        };
        self.mux_outcomes.push((ev.node_text_id, outcome));

        // Path hash is the concatenation of the mux features in observation order.
        self.path_hasher.update(&feature.node_id.to_le_bytes());
        self.path_hasher.update(&[match feature.select_kind {
            MuxSelectKind::CaseIndex => 0,
            MuxSelectKind::Default => 1,
            MuxSelectKind::NoBitsSet => 2,
            MuxSelectKind::MultiBitsSet => 3,
        }]);
        self.path_hasher
            .update(&feature.selected_index.to_le_bytes());
    }

    fn on_bool_node(&mut self, ev: BoolNodeEvent) {
        let node_id_u32 = u32::try_from(ev.node_text_id).unwrap_or(u32::MAX);
        self.bools_hasher.update(&node_id_u32.to_le_bytes());
        self.bools_hasher.update(&[if ev.value { 1 } else { 0 }]);
    }

    fn on_corner_event(&mut self, ev: CornerEvent) {
        if ev.kind == CornerKind::CompareDistance {
            let idx = hash_compare_distance_event(&ev);
            self.compare_distance_indices.push(idx);
        } else {
            let idx = hash_corner_event(&ev);
            self.corner_indices.push(idx);
        }
    }

    fn on_failure_event(&mut self, ev: FailureEvent) {
        let idx = hash_failure_event(&ev);
        self.failure_indices.push(idx);
    }
}

pub struct AutocovEngine {
    f: ir::Fn,
    args_tuple_type: ir::Type,
    args_bit_count: usize,
    rng: StdRng,
    max_iters: Option<u64>,
    stop: Arc<AtomicBool>,

    input_slices_by_width: BTreeMap<usize, Vec<usize>>,
    bit_constant_dict: Vec<BitConstant>,

    mux_map: FeatureMap64k,
    path_map: FeatureMap64k,
    bools_map: FeatureMap64k,
    corner_map: FeatureMap64k,
    compare_distance_map: FeatureMap64k,
    failure_map: FeatureMap64k,

    corpus: Vec<IrBits>,
    corpus_hashes: BTreeSet<[u8; 32]>,

    mux_outcome_spaces: BTreeMap<usize, MuxOutcomeSpace>,
    mux_outcome_observed: BTreeMap<usize, Vec<bool>>,
    mux_outcomes_possible_total: usize,
    mux_outcomes_observed_total: usize,
}

impl AutocovEngine {
    pub fn args_bit_count(&self) -> usize {
        self.args_bit_count
    }

    pub fn bits_from_arg_tuple(&self, tuple_value: &IrValue) -> Result<IrBits, String> {
        let elems = tuple_value
            .get_elements()
            .map_err(|e| format!("corpus value is not a tuple: {}", e))?;
        if elems.len() != self.f.params.len() {
            return Err(format!(
                "corpus tuple has {} elements but function has {} params",
                elems.len(),
                self.f.params.len()
            ));
        }
        Ok(ir_bits_from_value_with_type(
            tuple_value,
            &self.args_tuple_type,
        ))
    }

    pub fn corpus_len(&self) -> usize {
        self.corpus.len()
    }

    pub fn mux_features_set(&self) -> usize {
        self.mux_map.set_count()
    }

    pub fn path_features_set(&self) -> usize {
        self.path_map.set_count()
    }

    pub fn bools_features_set(&self) -> usize {
        self.bools_map.set_count()
    }

    pub fn corner_features_set(&self) -> usize {
        self.corner_map.set_count()
    }

    pub fn compare_distance_features_set(&self) -> usize {
        self.compare_distance_map.set_count()
    }

    pub fn failure_features_set(&self) -> usize {
        self.failure_map.set_count()
    }

    pub fn mux_outcomes_observed(&self) -> usize {
        self.mux_outcomes_observed_total
    }

    pub fn mux_outcomes_possible(&self) -> usize {
        self.mux_outcomes_possible_total
    }

    pub fn mux_outcomes_missing(&self) -> usize {
        self.mux_outcomes_possible_total - self.mux_outcomes_observed_total
    }

    /// Observes a candidate by running the interpreter and updating all feature
    /// maps.
    ///
    /// This does not mutate the corpus (no dedupe/corpus growth logic). Returns
    /// true when the evaluation succeeded, false when it returned
    /// `FnEvalResult::Failure`.
    pub fn observe_candidate(&mut self, cand: &IrBits) -> bool {
        let obs = self.evaluate_observation(cand);
        self.apply_observation(&obs);
        obs.ok
    }

    pub fn evaluate_observation(&self, cand: &IrBits) -> CandidateObservation {
        let args_tuple_value = ir_value_from_bits_with_type(cand, &self.args_tuple_type);
        let args = args_tuple_value.get_elements().unwrap();
        let mut obs = CollectingMuxObserver::new();
        let r = xlsynth_pir::ir_eval::eval_fn_with_observer(&self.f, &args, Some(&mut obs));
        let features = obs.finish();
        CandidateObservation {
            ok: matches!(r, FnEvalResult::Success(_)),
            mux_indices: features.mux_indices,
            path_index: features.path_index,
            bools_index: features.bools_index,
            corner_indices: features.corner_indices,
            compare_distance_indices: features.compare_distance_indices,
            failure_indices: features.failure_indices,
            mux_outcomes: features.mux_outcomes,
        }
    }

    pub fn evaluate_corner_events(&self, cand: &IrBits) -> (bool, BTreeSet<CornerEventId>) {
        #[derive(Debug)]
        struct CornerOnlyObserver {
            events: BTreeSet<CornerEventId>,
        }

        impl EvalObserver for CornerOnlyObserver {
            fn on_select(&mut self, _ev: SelectEvent) {}

            fn on_corner_event(&mut self, ev: CornerEvent) {
                self.events.insert(CornerEventId {
                    node_text_id: ev.node_text_id,
                    kind: ev.kind,
                    tag: ev.tag,
                });
            }

            fn on_failure_event(&mut self, _ev: FailureEvent) {}
        }

        let args_tuple_value = ir_value_from_bits_with_type(cand, &self.args_tuple_type);
        let args = args_tuple_value.get_elements().unwrap();
        let mut obs = CornerOnlyObserver {
            events: BTreeSet::new(),
        };
        let r = xlsynth_pir::ir_eval::eval_fn_with_observer(&self.f, &args, Some(&mut obs));
        let ok = matches!(r, FnEvalResult::Success(_));
        (ok, obs.events)
    }

    pub fn corner_event_domain(&self) -> BTreeSet<CornerEventId> {
        let mut out: BTreeSet<CornerEventId> = BTreeSet::new();
        for nr in self.f.node_refs() {
            let n = self.f.get_node(nr);
            match &n.payload {
                ir::NodePayload::Binop(ir::Binop::Add, lhs, rhs) => {
                    let lhs_is_lit =
                        matches!(self.f.get_node(*lhs).payload, ir::NodePayload::Literal(_));
                    let rhs_is_lit =
                        matches!(self.f.get_node(*rhs).payload, ir::NodePayload::Literal(_));
                    if !lhs_is_lit {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Add,
                            tag: 0, // LhsIsZero
                        });
                    }
                    if !rhs_is_lit {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Add,
                            tag: 1, // RhsIsZero
                        });
                    }
                }
                ir::NodePayload::Unop(ir::Unop::Neg, operand) => {
                    // OperandIsMinSigned only makes sense for non-empty bit vectors.
                    let w = bit_width_if_bits(self.f.get_node_ty(*operand)).unwrap_or(0);
                    let operand_is_lit = matches!(
                        self.f.get_node(*operand).payload,
                        ir::NodePayload::Literal(_)
                    );
                    if w > 0 && !operand_is_lit {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Neg,
                            tag: 1, // OperandMsbIsOne
                        });
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Neg,
                            tag: 0,
                        });
                    }
                }
                ir::NodePayload::SignExt { arg, new_bit_count } => {
                    let old_w = match self.f.get_node_ty(*arg) {
                        ir::Type::Bits(w) => *w,
                        _ => 0,
                    };
                    if *new_bit_count > old_w && old_w > 0 {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::SignExt,
                            tag: 0,
                        });
                    }
                }
                ir::NodePayload::Binop(binop, lhs, rhs)
                    if matches!(binop, ir::Binop::Eq | ir::Binop::Ne) =>
                {
                    // Only include compare-distance buckets reachable for this operand width.
                    let lhs_w = bit_width_if_bits(self.f.get_node_ty(*lhs)).unwrap_or(0);
                    let rhs_w = bit_width_if_bits(self.f.get_node_ty(*rhs)).unwrap_or(0);
                    if lhs_w == 0 || lhs_w != rhs_w {
                        continue;
                    }
                    let mut tags: BTreeSet<u8> = BTreeSet::new();
                    for d in 0..=lhs_w {
                        tags.insert(xlsynth_pir::corners::bucket_xor_popcount(d));
                    }
                    for tag in tags {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::CompareDistance,
                            tag,
                        });
                    }
                }
                ir::NodePayload::Binop(binop, lhs, rhs)
                    if matches!(binop, ir::Binop::Shll | ir::Binop::Shrl) =>
                {
                    let lhs_w = bit_width_if_bits(self.f.get_node_ty(*lhs)).unwrap_or(0);
                    let rhs_w = bit_width_if_bits(self.f.get_node_ty(*rhs)).unwrap_or(0);

                    // AmtIsZero is always reachable (including rhs_w==0 where amt==0).
                    out.insert(CornerEventId {
                        node_text_id: n.text_id,
                        kind: CornerKind::Shift,
                        tag: 0,
                    });
                    if lhs_w > 0 {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Shift,
                            tag: 1, // AmtLtWidth
                        });
                    }
                    // AmtGeWidth is reachable only if the selector can represent a value >= lhs_w.
                    if let Some(max_amt) = max_value_for_bit_width(rhs_w) {
                        if max_amt >= (lhs_w as u128) {
                            out.insert(CornerEventId {
                                node_text_id: n.text_id,
                                kind: CornerKind::Shift,
                                tag: 2, // AmtGeWidth
                            });
                        }
                    } else {
                        // Conservative: huge rhs can represent >= lhs_w.
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Shift,
                            tag: 2,
                        });
                    }
                }
                ir::NodePayload::Binop(ir::Binop::Shra, lhs, rhs) => {
                    let lhs_w = bit_width_if_bits(self.f.get_node_ty(*lhs)).unwrap_or(0);
                    let rhs_w = bit_width_if_bits(self.f.get_node_ty(*rhs)).unwrap_or(0);
                    if lhs_w == 0 {
                        continue;
                    }

                    // msb can be 0 or 1 in general, so include both msb branches.
                    let can_amt_lt = true; // amt==0 exists, and 0 < lhs_w holds since lhs_w>0
                    let can_amt_ge = match max_value_for_bit_width(rhs_w) {
                        Some(max_amt) => max_amt >= (lhs_w as u128),
                        None => true,
                    };

                    if can_amt_lt {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Shra,
                            tag: 0, // Msb0_AmtLt
                        });
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Shra,
                            tag: 2, // Msb1_AmtLt
                        });
                    }
                    if can_amt_ge {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Shra,
                            tag: 1, // Msb0_AmtGe
                        });
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::Shra,
                            tag: 3, // Msb1_AmtGe
                        });
                    }
                }
                ir::NodePayload::DynamicBitSlice {
                    arg: _,
                    start: _,
                    width: _,
                } => {
                    for tag in [0u8, 1u8] {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::DynamicBitSlice,
                            tag,
                        });
                    }
                }
                ir::NodePayload::ArrayIndex {
                    array: _,
                    indices: _,
                    assumed_in_bounds: _,
                } => {
                    for tag in [0u8, 1u8] {
                        out.insert(CornerEventId {
                            node_text_id: n.text_id,
                            kind: CornerKind::ArrayIndex,
                            tag,
                        });
                    }
                }
                _ => {}
            }
        }
        out
    }

    pub fn node_to_string_by_text_id(&self, node_text_id: usize) -> Option<String> {
        for nr in self.f.node_refs() {
            let n = self.f.get_node(nr);
            if n.text_id == node_text_id {
                return n.to_string(&self.f);
            }
        }
        None
    }

    pub fn apply_observation(&mut self, obs: &CandidateObservation) {
        let features = CandidateFeatures {
            mux_indices: obs.mux_indices.clone(),
            path_index: obs.path_index,
            bools_index: obs.bools_index,
            corner_indices: obs.corner_indices.clone(),
            compare_distance_indices: obs.compare_distance_indices.clone(),
            failure_indices: obs.failure_indices.clone(),
            mux_outcomes: obs.mux_outcomes.clone(),
        };
        let _ = self.apply_candidate_features(&features);
    }

    pub fn from_ir_path(
        ir_file: &Path,
        entry_fn: &str,
        cfg: AutocovConfig,
    ) -> Result<Self, String> {
        let ir_text = std::fs::read_to_string(ir_file).map_err(|e| e.to_string())?;
        Self::from_ir_text(&ir_text, Some(ir_file.to_path_buf()), entry_fn, cfg)
    }

    pub fn from_ir_text(
        ir_text: &str,
        filename: Option<PathBuf>,
        entry_fn: &str,
        cfg: AutocovConfig,
    ) -> Result<Self, String> {
        let mut parser = Parser::new(ir_text);
        let _ = filename;
        let pkg = parser
            .parse_and_validate_package()
            .map_err(|e| format!("PIR parse: {}", e))?;
        let f = pkg
            .get_fn(entry_fn)
            .ok_or_else(|| format!("function not found: {}", entry_fn))?
            .clone();

        let args_tuple_type = ir::Type::Tuple(
            f.params
                .iter()
                .map(|p| Box::new(p.ty.clone()))
                .collect::<Vec<_>>(),
        );
        let args_bit_count = args_tuple_type.bit_count();

        let stop = Arc::new(AtomicBool::new(false));
        let mut engine = Self {
            f,
            args_tuple_type,
            args_bit_count,
            rng: StdRng::seed_from_u64(cfg.seed),
            max_iters: cfg.max_iters,
            stop,
            input_slices_by_width: BTreeMap::new(),
            bit_constant_dict: Vec::new(),
            mux_map: FeatureMap64k::new(),
            path_map: FeatureMap64k::new(),
            bools_map: FeatureMap64k::new(),
            corner_map: FeatureMap64k::new(),
            compare_distance_map: FeatureMap64k::new(),
            failure_map: FeatureMap64k::new(),
            corpus: Vec::new(),
            corpus_hashes: BTreeSet::new(),
            mux_outcome_spaces: BTreeMap::new(),
            mux_outcome_observed: BTreeMap::new(),
            mux_outcomes_possible_total: 0,
            mux_outcomes_observed_total: 0,
        };
        engine.mux_outcome_spaces = engine.compute_mux_outcome_spaces();
        engine.mux_outcome_observed = engine
            .mux_outcome_spaces
            .iter()
            .map(|(&node_id, space)| (node_id, vec![false; space.outcome_count()]))
            .collect();
        engine.mux_outcomes_possible_total = engine
            .mux_outcome_spaces
            .values()
            .map(|s| s.outcome_count())
            .sum();
        engine.input_slices_by_width = engine.compute_input_slices_by_width();
        engine.bit_constant_dict = engine.compute_bit_constant_dict();
        Ok(engine)
    }

    fn compute_input_slices_by_width(&self) -> BTreeMap<usize, Vec<usize>> {
        let mut out: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let ir::Type::Tuple(types) = &self.args_tuple_type else {
            return out;
        };
        for (i, ty) in types.iter().enumerate() {
            let ir::Type::Bits(w) = ty.as_ref() else {
                continue;
            };
            if *w == 0 || *w > 64 {
                continue;
            }
            let sl = self
                .args_tuple_type
                .tuple_get_flat_bit_slice_for_index(i)
                .expect("args_tuple_type is a tuple");
            out.entry(*w).or_default().push(sl.start);
        }
        for starts in out.values_mut() {
            starts.sort();
        }
        out
    }

    fn compute_bit_constant_dict(&self) -> Vec<BitConstant> {
        let mut set: BTreeSet<BitConstant> = BTreeSet::new();
        for nr in self.f.node_refs() {
            let n = self.f.get_node(nr);
            let ir::NodePayload::Literal(v) = &n.payload else {
                continue;
            };
            let bits = match v.to_bits() {
                Ok(b) => b,
                Err(_) => continue,
            };
            let w = bits.get_bit_count();
            if w == 0 || w > 64 {
                continue;
            }
            let value_u64 = match bits.to_u64() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if self.input_slices_by_width.contains_key(&w) {
                set.insert(BitConstant {
                    bit_count: w,
                    value_u64,
                });
            }
        }
        set.into_iter().collect()
    }

    fn compute_mux_outcome_spaces(&self) -> BTreeMap<usize, MuxOutcomeSpace> {
        let mut out: BTreeMap<usize, MuxOutcomeSpace> = BTreeMap::new();
        for nr in self.f.node_refs() {
            let n = self.f.get_node(nr);
            match &n.payload {
                ir::NodePayload::Sel {
                    selector,
                    cases,
                    default: _,
                } => {
                    let selector_w = match self.f.get_node_ty(*selector) {
                        ir::Type::Bits(w) => *w,
                        _ => 0,
                    };
                    let space: usize = if selector_w >= (usize::BITS as usize) {
                        usize::MAX
                    } else {
                        1usize << selector_w
                    };
                    let reachable_case_count = std::cmp::min(cases.len(), space);
                    let has_default = space > cases.len();
                    out.insert(
                        n.text_id,
                        MuxOutcomeSpace {
                            node_text_id: n.text_id,
                            kind: MuxNodeKind::Sel,
                            reachable_case_count,
                            has_default,
                            has_no_bits_set: false,
                            has_multi_bits_set: false,
                        },
                    );
                }
                ir::NodePayload::PrioritySel {
                    selector,
                    cases,
                    default: _,
                } => {
                    let selector_w = match self.f.get_node_ty(*selector) {
                        ir::Type::Bits(w) => *w,
                        _ => 0,
                    };
                    let reachable_case_count = std::cmp::min(cases.len(), selector_w);
                    out.insert(
                        n.text_id,
                        MuxOutcomeSpace {
                            node_text_id: n.text_id,
                            kind: MuxNodeKind::PrioritySel,
                            reachable_case_count,
                            has_default: true,
                            has_no_bits_set: false,
                            has_multi_bits_set: false,
                        },
                    );
                }
                ir::NodePayload::OneHotSel { selector, cases } => {
                    let selector_w = match self.f.get_node_ty(*selector) {
                        ir::Type::Bits(w) => *w,
                        _ => 0,
                    };
                    let reachable_case_count = std::cmp::min(cases.len(), selector_w);
                    out.insert(
                        n.text_id,
                        MuxOutcomeSpace {
                            node_text_id: n.text_id,
                            kind: MuxNodeKind::OneHotSel,
                            reachable_case_count,
                            has_default: false,
                            has_no_bits_set: true,
                            has_multi_bits_set: reachable_case_count >= 2,
                        },
                    );
                }
                _ => {}
            }
        }
        out
    }

    pub fn set_stop_flag(&mut self, stop: Arc<AtomicBool>) {
        self.stop = stop;
    }

    pub fn add_corpus_sample_from_arg_tuple(
        &mut self,
        tuple_value: &IrValue,
    ) -> Result<(), String> {
        let elems = tuple_value
            .get_elements()
            .map_err(|e| format!("corpus line is not a tuple: {}", e))?;
        if elems.len() != self.f.params.len() {
            return Err(format!(
                "corpus tuple has {} elements but function has {} params",
                elems.len(),
                self.f.params.len()
            ));
        }
        let bits = ir_bits_from_value_with_type(tuple_value, &self.args_tuple_type);
        if !self.insert_corpus_hash(&bits) {
            return Ok(());
        }
        // Seed feature maps from the existing corpus so subsequent runs don't
        // treat already-covered paths/features as novel.
        let features = self.evaluate_candidate_features(&bits);
        self.apply_candidate_features(&features);
        self.corpus.push(bits);
        Ok(())
    }

    pub fn seed_structured_corpus<'a>(
        &mut self,
        two_hot_max_bits: usize,
        sink: Option<&'a mut (dyn CorpusSink + 'a)>,
    ) -> usize {
        let sink_ptr: Option<*mut (dyn CorpusSink + 'a)> =
            sink.map(|s| s as *mut (dyn CorpusSink + 'a));

        let mut added = 0usize;
        let n = self.args_bit_count;

        // Always seed all-zeros and all-ones.
        added += self.force_add_seed_bits(self.make_all_zeros_bits(), sink_ptr) as usize;
        added += self.force_add_seed_bits(self.make_all_ones_bits(), sink_ptr) as usize;

        // One-hot.
        for i in 0..n {
            added += self.force_add_seed_bits(self.make_one_hot_bits(i), sink_ptr) as usize;
        }

        // Two-hot (can be quadratic).
        if n <= two_hot_max_bits {
            for i in 0..n {
                for j in (i + 1)..n {
                    added +=
                        self.force_add_seed_bits(self.make_two_hot_bits(i, j), sink_ptr) as usize;
                }
            }
        }

        added
    }

    pub fn get_mux_space_summary(&self) -> MuxSpaceSummary {
        let mut muxes: Vec<MuxNodeSpace> = Vec::new();
        for nr in self.f.node_refs() {
            let n = self.f.get_node(nr);
            match &n.payload {
                ir::NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    let selector_bit_count = match self.f.get_node_ty(*selector) {
                        ir::Type::Bits(w) => Some(*w),
                        _ => None,
                    };
                    muxes.push(MuxNodeSpace {
                        node_text_id: n.text_id,
                        kind: MuxNodeKind::Sel,
                        cases_len: cases.len(),
                        has_default: default.is_some(),
                        selector_bit_count,
                    });
                }
                ir::NodePayload::PrioritySel {
                    selector,
                    cases,
                    default,
                } => {
                    let selector_bit_count = match self.f.get_node_ty(*selector) {
                        ir::Type::Bits(w) => Some(*w),
                        _ => None,
                    };
                    muxes.push(MuxNodeSpace {
                        node_text_id: n.text_id,
                        kind: MuxNodeKind::PrioritySel,
                        cases_len: cases.len(),
                        has_default: default.is_some(),
                        selector_bit_count,
                    });
                }
                ir::NodePayload::OneHotSel { selector, cases } => {
                    let selector_bit_count = match self.f.get_node_ty(*selector) {
                        ir::Type::Bits(w) => Some(*w),
                        _ => None,
                    };
                    muxes.push(MuxNodeSpace {
                        node_text_id: n.text_id,
                        kind: MuxNodeKind::OneHotSel,
                        cases_len: cases.len(),
                        has_default: false,
                        selector_bit_count,
                    });
                }
                _ => {}
            }
        }

        muxes.sort_by_key(|m| m.node_text_id);

        let total_mux_feature_possibilities: usize =
            muxes.iter().map(|m| m.feature_possibilities()).sum();
        let log10_path_space_upper_bound: f64 = muxes
            .iter()
            .map(|m| m.log10_path_possibilities_upper_bound())
            .sum();

        MuxSpaceSummary {
            muxes,
            total_mux_feature_possibilities,
            log10_path_space_upper_bound,
        }
    }

    pub fn get_mux_outcome_report(&self) -> MuxOutcomeReport {
        let mut entries: Vec<MuxOutcomeReportEntry> = Vec::new();
        let mut total_missing: usize = 0;
        let mut total_possible: usize = 0;

        for (&node_id, space) in self.mux_outcome_spaces.iter() {
            let possible_count = space.outcome_count();
            total_possible += possible_count;

            let observed_bits = self
                .mux_outcome_observed
                .get(&node_id)
                .cloned()
                .unwrap_or_else(|| vec![false; possible_count]);
            let observed_count = observed_bits.iter().filter(|&&b| b).count();

            let mut missing: Vec<MuxOutcomeId> = Vec::new();
            for i in 0..possible_count {
                if !observed_bits[i] {
                    if let Some(o) = space.index_to_outcome(i) {
                        missing.push(o);
                    }
                }
            }
            total_missing += missing.len();
            entries.push(MuxOutcomeReportEntry {
                node_text_id: node_id,
                kind: space.kind,
                observed_count,
                possible_count,
                missing,
            });
        }

        entries.sort_by_key(|e| e.node_text_id);
        MuxOutcomeReport {
            entries,
            total_missing,
            total_possible,
        }
    }

    pub fn run(&mut self) -> AutocovReport {
        self.run_with_sink(None)
    }

    pub fn run_with_sink<'a>(
        &mut self,
        sink: Option<&'a mut (dyn CorpusSink + 'a)>,
    ) -> AutocovReport {
        self.run_with_sinks(sink, None, None)
    }

    pub fn run_with_sinks<'a>(
        &mut self,
        sink: Option<&'a mut (dyn CorpusSink + 'a)>,
        progress: Option<&'a mut (dyn ProgressSink + 'a)>,
        progress_every_iters: Option<u64>,
    ) -> AutocovReport {
        let sink_ptr: Option<*mut (dyn CorpusSink + 'a)> =
            sink.map(|s| s as *mut (dyn CorpusSink + 'a));
        let progress_ptr: Option<*mut (dyn ProgressSink + 'a)> =
            progress.map(|p| p as *mut (dyn ProgressSink + 'a));
        let mut iters: u64 = 0;
        loop {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            if let Some(max) = self.max_iters {
                if iters >= max {
                    break;
                }
            }

            let cand = self.generate_proposal();
            let features = self.evaluate_candidate_features(&cand);
            let added = self.maybe_add_to_corpus(cand, &features, sink_ptr);

            if let Some(p_ptr) = progress_ptr {
                let should_report = added
                    || progress_every_iters
                        .is_some_and(|every| every > 0 && ((iters + 1) % every == 0));
                if should_report {
                    let mux_outcomes_observed = self.mux_outcomes_observed_total;
                    let mux_outcomes_possible = self.mux_outcomes_possible_total;
                    let mux_outcomes_missing = mux_outcomes_possible - mux_outcomes_observed;
                    let p = AutocovProgress {
                        iters: iters + 1,
                        corpus_len: self.corpus.len(),
                        mux_features_set: self.mux_map.set_count(),
                        path_features_set: self.path_map.set_count(),
                        bools_features_set: self.bools_map.set_count(),
                        corner_features_set: self.corner_map.set_count(),
                        compare_distance_features_set: self.compare_distance_map.set_count(),
                        failure_features_set: self.failure_map.set_count(),
                        mux_outcomes_observed,
                        mux_outcomes_possible,
                        mux_outcomes_missing,
                        last_iter_added: added,
                    };
                    // Safety: caller holds exclusive mutable access for duration of run.
                    unsafe { &mut *p_ptr }.on_progress(p);
                }
            }

            iters += 1;
        }

        let mux_outcomes_observed = self.mux_outcomes_observed_total;
        let mux_outcomes_possible = self.mux_outcomes_possible_total;
        let mux_outcomes_missing = mux_outcomes_possible - mux_outcomes_observed;
        AutocovReport {
            iters,
            corpus_len: self.corpus.len(),
            mux_features_set: self.mux_map.set_count(),
            path_features_set: self.path_map.set_count(),
            bools_features_set: self.bools_map.set_count(),
            corner_features_set: self.corner_map.set_count(),
            compare_distance_features_set: self.compare_distance_map.set_count(),
            failure_features_set: self.failure_map.set_count(),
            mux_outcomes_observed,
            mux_outcomes_possible,
            mux_outcomes_missing,
        }
    }

    pub fn run_parallel_with_sinks<'a>(
        &mut self,
        threads: usize,
        sink: Option<&'a mut (dyn CorpusSink + 'a)>,
        progress: Option<&'a mut (dyn ProgressSink + 'a)>,
        progress_every_iters: Option<u64>,
    ) -> AutocovReport {
        assert!(threads > 0, "threads must be > 0");
        let sink_ptr: Option<*mut (dyn CorpusSink + 'a)> =
            sink.map(|s| s as *mut (dyn CorpusSink + 'a));
        let progress_ptr: Option<*mut (dyn ProgressSink + 'a)> =
            progress.map(|p| p as *mut (dyn ProgressSink + 'a));

        #[derive(Debug)]
        struct WorkItem {
            seq: u64,
            bits: IrBits,
        }

        #[derive(Debug)]
        struct WorkResult {
            seq: u64,
            bits: IrBits,
            features: CandidateFeatures,
        }

        let work_cap = std::cmp::max(threads * 4, 16);
        let (work_tx, work_rx): (SyncSender<WorkItem>, Receiver<WorkItem>) = sync_channel(work_cap);
        let (res_tx, res_rx) = sync_channel::<WorkResult>(work_cap);

        let f = Arc::new(self.f.clone());
        let args_tuple_type = Arc::new(self.args_tuple_type.clone());
        let work_rx = Arc::new(Mutex::new(work_rx));

        fn spawn_worker(
            f: Arc<ir::Fn>,
            args_tuple_type: Arc<ir::Type>,
            work_rx: Arc<Mutex<Receiver<WorkItem>>>,
            res_tx: SyncSender<WorkResult>,
        ) -> JoinHandle<()> {
            std::thread::spawn(move || {
                loop {
                    let item = {
                        let rx = work_rx.lock().unwrap();
                        rx.recv()
                    };
                    let item = match item {
                        Ok(v) => v,
                        Err(_) => break,
                    };
                    let features = {
                        let tuple_value =
                            ir_value_from_bits_with_type(&item.bits, args_tuple_type.as_ref());
                        let args = tuple_value.get_elements().unwrap();
                        let mut obs = CollectingMuxObserver::new();
                        let _ = xlsynth_pir::ir_eval::eval_fn_with_observer(
                            f.as_ref(),
                            &args,
                            Some(&mut obs),
                        );
                        obs.finish()
                    };
                    // Best-effort: ignore send failures (coordinator gone).
                    let _ = res_tx.send(WorkResult {
                        seq: item.seq,
                        bits: item.bits,
                        features,
                    });
                }
            })
        }

        let mut workers: Vec<JoinHandle<()>> = Vec::with_capacity(threads);
        for _ in 0..threads {
            workers.push(spawn_worker(
                f.clone(),
                args_tuple_type.clone(),
                work_rx.clone(),
                res_tx.clone(),
            ));
        }
        drop(res_tx);

        let mut seq_next_send: u64 = 0;
        let mut seq_next_apply: u64 = 0;
        let mut inflight: usize = 0;
        let mut pending: BTreeMap<u64, WorkResult> = BTreeMap::new();

        loop {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            if let Some(max) = self.max_iters {
                if seq_next_send >= max {
                    break;
                }
            }

            while inflight < work_cap {
                if self.stop.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(max) = self.max_iters {
                    if seq_next_send >= max {
                        break;
                    }
                }
                let bits = self.generate_proposal();
                if work_tx
                    .send(WorkItem {
                        seq: seq_next_send,
                        bits,
                    })
                    .is_err()
                {
                    break;
                }
                inflight += 1;
                seq_next_send += 1;
            }

            if inflight == 0 {
                break;
            }

            let r = match res_rx.recv() {
                Ok(r) => r,
                Err(_) => break,
            };
            pending.insert(r.seq, r);

            while let Some(r) = pending.remove(&seq_next_apply) {
                let added = self.maybe_add_to_corpus(r.bits, &r.features, sink_ptr);
                inflight -= 1;

                if let Some(p_ptr) = progress_ptr {
                    let should_report = added
                        || progress_every_iters
                            .is_some_and(|every| every > 0 && ((seq_next_apply + 1) % every == 0));
                    if should_report {
                        let mux_outcomes_observed = self.mux_outcomes_observed_total;
                        let mux_outcomes_possible = self.mux_outcomes_possible_total;
                        let mux_outcomes_missing = mux_outcomes_possible - mux_outcomes_observed;
                        let p = AutocovProgress {
                            iters: seq_next_apply + 1,
                            corpus_len: self.corpus.len(),
                            mux_features_set: self.mux_map.set_count(),
                            path_features_set: self.path_map.set_count(),
                            bools_features_set: self.bools_map.set_count(),
                            corner_features_set: self.corner_map.set_count(),
                            compare_distance_features_set: self.compare_distance_map.set_count(),
                            failure_features_set: self.failure_map.set_count(),
                            mux_outcomes_observed,
                            mux_outcomes_possible,
                            mux_outcomes_missing,
                            last_iter_added: added,
                        };
                        unsafe { &mut *p_ptr }.on_progress(p);
                    }
                }

                seq_next_apply += 1;
            }
        }

        // Stop workers and drain outstanding results for determinism.
        drop(work_tx);
        while inflight > 0 {
            if let Ok(r) = res_rx.recv() {
                pending.insert(r.seq, r);
                while let Some(r) = pending.remove(&seq_next_apply) {
                    let _ = self.maybe_add_to_corpus(r.bits, &r.features, sink_ptr);
                    inflight -= 1;
                    seq_next_apply += 1;
                }
            } else {
                break;
            }
        }

        for h in workers {
            let _ = h.join();
        }

        let mux_outcomes_observed = self.mux_outcomes_observed_total;
        let mux_outcomes_possible = self.mux_outcomes_possible_total;
        let mux_outcomes_missing = mux_outcomes_possible - mux_outcomes_observed;
        AutocovReport {
            iters: seq_next_apply,
            corpus_len: self.corpus.len(),
            mux_features_set: self.mux_map.set_count(),
            path_features_set: self.path_map.set_count(),
            bools_features_set: self.bools_map.set_count(),
            corner_features_set: self.corner_map.set_count(),
            compare_distance_features_set: self.compare_distance_map.set_count(),
            failure_features_set: self.failure_map.set_count(),
            mux_outcomes_observed,
            mux_outcomes_possible,
            mux_outcomes_missing,
        }
    }

    fn generate_proposal(&mut self) -> IrBits {
        if self.corpus.is_empty() {
            return self.random_bits(self.args_bit_count);
        }

        // Mutation/crossover strategy mix (single-threaded, deterministic via PRNG).
        //
        // Rationale:
        // - Single-bit flip is good for local edge conditions.
        // - Multi-bit "havoc" helps jump between regions.
        // - Sub-slice crossover helps preserve "good" chunks.
        // - XOR mixing is often effective when conditions depend on parity-like
        //   structure.
        // - Occasional full-random prevents corpus lock-in.
        let roll = (self.rng.next_u64() % 100) as u8;
        match roll {
            // 10%: fully random resample
            0..=9 => self.random_bits(self.args_bit_count),
            // 10%: dictionary overwrite (IR literal constants -> input slices)
            10..=19 => {
                let parent = self.pick_parent().clone();
                self.mutate_dictionary_overwrite(parent)
            }
            // 25%: multi-bit havoc
            20..=44 => {
                let parent = self.pick_parent().clone();
                self.mutate_havoc(parent)
            }
            // 25%: arbitrary subslice crossover
            45..=69 => {
                let a = self.pick_parent().clone();
                let b = self.pick_parent().clone();
                self.crossover_subslice(a, b)
            }
            // 20%: XOR-mix
            70..=89 => {
                let a = self.pick_parent().clone();
                let b = self.pick_parent().clone();
                self.xor_mix(a, b)
            }
            // 10%: single-bit flip
            _ => {
                let parent = self.pick_parent().clone();
                self.mutate_flip_bit(parent)
            }
        }
    }

    fn pick_parent(&mut self) -> &IrBits {
        let idx = (self.rng.next_u64() as usize) % self.corpus.len();
        &self.corpus[idx]
    }

    fn random_bits(&mut self, bit_count: usize) -> IrBits {
        let mut bits: Vec<bool> = Vec::with_capacity(bit_count);
        let mut remaining = bit_count;
        while remaining > 0 {
            let word = self.rng.next_u64();
            let take = std::cmp::min(64, remaining);
            for i in 0..take {
                bits.push(((word >> i) & 1) != 0);
            }
            remaining -= take;
        }
        IrBits::from_lsb_is_0(&bits)
    }

    fn bits_to_vec_lsb_is_0(&self, bits: &IrBits) -> Vec<bool> {
        let mut v: Vec<bool> = Vec::with_capacity(self.args_bit_count);
        for i in 0..self.args_bit_count {
            v.push(bits.get_bit(i).unwrap());
        }
        v
    }

    fn make_all_zeros_bits(&self) -> IrBits {
        let v: Vec<bool> = vec![false; self.args_bit_count];
        IrBits::from_lsb_is_0(&v)
    }

    fn make_all_ones_bits(&self) -> IrBits {
        let v: Vec<bool> = vec![true; self.args_bit_count];
        IrBits::from_lsb_is_0(&v)
    }

    fn make_one_hot_bits(&self, idx: usize) -> IrBits {
        assert!(idx < self.args_bit_count);
        let mut v: Vec<bool> = vec![false; self.args_bit_count];
        v[idx] = true;
        IrBits::from_lsb_is_0(&v)
    }

    fn make_two_hot_bits(&self, i: usize, j: usize) -> IrBits {
        assert!(i < self.args_bit_count);
        assert!(j < self.args_bit_count);
        assert!(i != j);
        let mut v: Vec<bool> = vec![false; self.args_bit_count];
        v[i] = true;
        v[j] = true;
        IrBits::from_lsb_is_0(&v)
    }

    fn hash_bits(bits: &IrBits) -> [u8; 32] {
        let mut h = Hasher::new();
        h.update(b"xlsynth-autocov:candidate-bits");
        h.update(&(bits.get_bit_count() as u64).to_le_bytes());
        h.update(&bits.to_bytes().unwrap());
        *h.finalize().as_bytes()
    }

    fn insert_corpus_hash(&mut self, bits: &IrBits) -> bool {
        self.corpus_hashes.insert(Self::hash_bits(bits))
    }

    fn force_add_seed_bits(
        &mut self,
        bits: IrBits,
        sink_ptr: Option<*mut (dyn CorpusSink + '_)>,
    ) -> bool {
        if !self.insert_corpus_hash(&bits) {
            return false;
        }
        let features = self.evaluate_candidate_features(&bits);
        let _ = self.apply_candidate_features(&features);

        if let Some(sink_ptr) = sink_ptr {
            let tuple_value = ir_value_from_bits_with_type(&bits, &self.args_tuple_type);
            unsafe { &mut *sink_ptr }.on_new_sample(&tuple_value);
        }

        self.corpus.push(bits);
        true
    }

    fn mutate_flip_bit(&mut self, bits: IrBits) -> IrBits {
        if self.args_bit_count == 0 {
            return bits;
        }
        let i = (self.rng.next_u64() as usize) % self.args_bit_count;
        let mut v = self.bits_to_vec_lsb_is_0(&bits);
        v[i] = !v[i];
        IrBits::from_lsb_is_0(&v)
    }

    fn mutate_havoc(&mut self, bits: IrBits) -> IrBits {
        if self.args_bit_count == 0 {
            return bits;
        }
        let mut v = self.bits_to_vec_lsb_is_0(&bits);
        // Choose a small-ish number of flips; cap keeps it efficient.
        let max_flips = std::cmp::min(self.args_bit_count, 64);
        let flips = 1 + ((self.rng.next_u64() as usize) % max_flips);
        let mut seen: Vec<usize> = Vec::with_capacity(flips);
        // Best-effort distinct indices; duplicates are harmless but less effective.
        for _ in 0..flips {
            let idx = (self.rng.next_u64() as usize) % self.args_bit_count;
            if !seen.contains(&idx) {
                seen.push(idx);
                v[idx] = !v[idx];
            }
        }
        IrBits::from_lsb_is_0(&v)
    }

    fn mutate_dictionary_overwrite(&mut self, bits: IrBits) -> IrBits {
        if self.args_bit_count == 0 {
            return bits;
        }
        if self.bit_constant_dict.is_empty() {
            return self.mutate_havoc(bits);
        }
        let mut v = self.bits_to_vec_lsb_is_0(&bits);

        // Choose a width present in both the input layout and the dictionary.
        let c_idx = (self.rng.next_u64() as usize) % self.bit_constant_dict.len();
        let w = self.bit_constant_dict[c_idx].bit_count;
        let starts = match self.input_slices_by_width.get(&w) {
            Some(s) if !s.is_empty() => s,
            _ => return self.mutate_havoc(bits),
        };

        // Collect all dictionary values of this width.
        let mut values: Vec<u64> = Vec::new();
        for c in &self.bit_constant_dict {
            if c.bit_count == w {
                values.push(c.value_u64);
            }
        }
        if values.is_empty() {
            return self.mutate_havoc(bits);
        }

        // Overwrite up to 4 distinct slices at this width. This makes it plausible to
        // satisfy multiple independent predicates in a single proposal (classic
        // "magic bytes").
        let overwrite_count = std::cmp::min(4, starts.len());
        let mut idxs: Vec<usize> = (0..starts.len()).collect();
        for i in 0..overwrite_count {
            let j = i + ((self.rng.next_u64() as usize) % (starts.len() - i));
            idxs.swap(i, j);
        }
        for i in 0..overwrite_count {
            let start = starts[idxs[i]];
            let value = values[(self.rng.next_u64() as usize) % values.len()];
            for bit in 0..w {
                v[start + bit] = ((value >> bit) & 1) != 0;
            }
        }
        IrBits::from_lsb_is_0(&v)
    }

    fn crossover_subslice(&mut self, a: IrBits, b: IrBits) -> IrBits {
        if self.args_bit_count == 0 {
            return a;
        }
        let mut v = self.bits_to_vec_lsb_is_0(&a);
        let start = (self.rng.next_u64() as usize) % self.args_bit_count;
        let max_len = self.args_bit_count - start;
        let len = 1 + ((self.rng.next_u64() as usize) % max_len);
        for i in start..start + len {
            v[i] = b.get_bit(i).unwrap();
        }
        IrBits::from_lsb_is_0(&v)
    }

    fn xor_mix(&mut self, a: IrBits, b: IrBits) -> IrBits {
        if self.args_bit_count == 0 {
            return a;
        }
        let mut v: Vec<bool> = Vec::with_capacity(self.args_bit_count);
        for i in 0..self.args_bit_count {
            v.push(a.get_bit(i).unwrap() ^ b.get_bit(i).unwrap());
        }
        IrBits::from_lsb_is_0(&v)
    }

    fn evaluate_candidate_features(&self, cand: &IrBits) -> CandidateFeatures {
        let args_tuple_value = ir_value_from_bits_with_type(cand, &self.args_tuple_type);
        let args = args_tuple_value.get_elements().unwrap();
        let mut obs = CollectingMuxObserver::new();
        let _ = xlsynth_pir::ir_eval::eval_fn_with_observer(&self.f, &args, Some(&mut obs));
        obs.finish()
    }

    fn apply_candidate_features(&mut self, features: &CandidateFeatures) -> Observations {
        let mut mux_new = false;
        for &idx in features.mux_indices.iter() {
            mux_new |= self.mux_map.observe_index(idx);
        }
        let path_new = self.path_map.observe_index(features.path_index);
        let bools_new = self.bools_map.observe_index(features.bools_index);
        let mut corner_new = false;
        for &idx in features.corner_indices.iter() {
            corner_new |= self.corner_map.observe_index(idx);
        }
        let mut compare_distance_new = false;
        for &idx in features.compare_distance_indices.iter() {
            compare_distance_new |= self.compare_distance_map.observe_index(idx);
        }
        let mut failure_new = false;
        for &idx in features.failure_indices.iter() {
            failure_new |= self.failure_map.observe_index(idx);
        }

        // Record mux outcome observations (per-node).
        for &(node_id, outcome) in features.mux_outcomes.iter() {
            let space = match self.mux_outcome_spaces.get(&node_id) {
                Some(s) => s,
                None => continue,
            };
            let idx = match space.outcome_to_index(outcome) {
                Some(i) => i,
                None => continue,
            };
            if let Some(bits) = self.mux_outcome_observed.get_mut(&node_id) {
                if idx < bits.len() {
                    if !bits[idx] {
                        bits[idx] = true;
                        self.mux_outcomes_observed_total += 1;
                    }
                }
            }
        }

        Observations {
            mux_new,
            path_new,
            bools_new,
            corner_new,
            compare_distance_new,
            failure_new,
        }
    }

    fn maybe_add_to_corpus(
        &mut self,
        cand: IrBits,
        features: &CandidateFeatures,
        sink: Option<*mut (dyn CorpusSink + '_)>,
    ) -> bool {
        if !self.insert_corpus_hash(&cand) {
            return false;
        }
        let obs = self.apply_candidate_features(features);
        if !obs.mux_new
            && !obs.path_new
            && !obs.bools_new
            && !obs.corner_new
            && !obs.compare_distance_new
            && !obs.failure_new
        {
            return false;
        }
        if let Some(sink_ptr) = sink {
            let tuple_value = ir_value_from_bits_with_type(&cand, &self.args_tuple_type);
            // Safety: `run_with_sink` requires the caller to provide a stable, exclusive
            // sink reference for the duration of the run. We only create a
            // temporary `&mut` for this call site, and we remain
            // single-threaded.
            unsafe { &mut *sink_ptr }.on_new_sample(&tuple_value);
        }
        self.corpus.push(cand);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth::IrValue;

    #[test]
    fn first_candidate_produces_new_features() {
        let ir_text = r#"package test

fn f(selidx: bits[2] id=1, a: bits[8] id=2, b: bits[8] id=3, d: bits[8] id=4) -> bits[8] {
  ret s: bits[8] = sel(selidx, cases=[a, b], default=d, id=10)
}
"#;
        let mut engine = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(1),
            },
        )
        .unwrap();

        let tuple = IrValue::make_tuple(&[
            IrValue::make_ubits(2, 1).unwrap(),
            IrValue::make_ubits(8, 10).unwrap(),
            IrValue::make_ubits(8, 20).unwrap(),
            IrValue::make_ubits(8, 30).unwrap(),
        ]);
        let bits = ir_bits_from_value_with_type(&tuple, &engine.args_tuple_type);

        let f1 = engine.evaluate_candidate_features(&bits);
        assert!(engine.maybe_add_to_corpus(bits.clone(), &f1, None));

        let f2 = engine.evaluate_candidate_features(&bits);
        assert!(!engine.maybe_add_to_corpus(bits, &f2, None));
    }

    #[test]
    fn mux_feature_hash_changes_with_selected_index() {
        let a = MuxFeature {
            node_id: 10,
            select_kind: MuxSelectKind::CaseIndex,
            selected_index: 1,
        };
        let b = MuxFeature {
            node_id: 10,
            select_kind: MuxSelectKind::CaseIndex,
            selected_index: 2,
        };
        assert_ne!(mux_feature_hash(&a), mux_feature_hash(&b));
    }

    #[test]
    fn mutations_preserve_bit_count() {
        let ir_text = r#"package test

fn f(x: bits[16] id=1) -> bits[16] {
  ret y: bits[16] = identity(x, id=2)
}
"#;
        let mut engine = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(1),
            },
        )
        .unwrap();
        let base = engine.random_bits(engine.args_bit_count);
        assert_eq!(base.get_bit_count(), engine.args_bit_count);

        let a = engine.mutate_flip_bit(base.clone());
        assert_eq!(a.get_bit_count(), engine.args_bit_count);
        let b = engine.mutate_havoc(base.clone());
        assert_eq!(b.get_bit_count(), engine.args_bit_count);
        let c = engine.crossover_subslice(base.clone(), base.clone());
        assert_eq!(c.get_bit_count(), engine.args_bit_count);
        let d = engine.xor_mix(base.clone(), base);
        assert_eq!(d.get_bit_count(), engine.args_bit_count);
    }

    #[test]
    fn parallel_matches_single_threaded_report() {
        let ir_text = r#"package test

fn f(selidx: bits[2] id=1, a: bits[8] id=2, b: bits[8] id=3, d: bits[8] id=4) -> bits[8] {
  ret s: bits[8] = sel(selidx, cases=[a, b], default=d, id=10)
}
"#;
        let cfg = AutocovConfig {
            seed: 0,
            max_iters: Some(200),
        };
        let mut s1 = AutocovEngine::from_ir_text(ir_text, None, "f", cfg.clone()).unwrap();
        let mut s2 = AutocovEngine::from_ir_text(ir_text, None, "f", cfg).unwrap();
        let r1 = s1.run();
        let r2 = s2.run_parallel_with_sinks(2, None, None, None);
        assert_eq!(r1.iters, r2.iters);
        assert_eq!(r1.corpus_len, r2.corpus_len);
        assert_eq!(r1.mux_features_set, r2.mux_features_set);
        assert_eq!(r1.path_features_set, r2.path_features_set);
    }

    #[test]
    fn structured_seed_count_matches_closed_form_when_enabled() {
        let ir_text = r#"package test

fn f(x: bits[4] id=1) -> bits[4] {
  ret y: bits[4] = identity(x, id=2)
}
"#;
        let mut engine = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(1),
            },
        )
        .unwrap();

        let n = engine.args_bit_count;
        assert_eq!(n, 4);
        let added = engine.seed_structured_corpus(/* two_hot_max_bits= */ 64, None);
        let expected = 2 + n + (n * (n - 1) / 2);
        assert_eq!(added, expected);
        assert_eq!(engine.corpus.len(), expected);
    }

    #[test]
    fn one_hot_sel_missing_includes_multi_bits_set_until_observed() {
        let ir_text = r#"package test

fn f(oh: bits[3] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[8] {
  ret o: bits[8] = one_hot_sel(oh, cases=[a, b, c], id=10)
}
"#;
        let mut engine = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(0),
            },
        )
        .unwrap();

        // Seed with a single-bit selector (no MultiBitsSet).
        let t1 = IrValue::make_tuple(&[
            IrValue::make_ubits(3, 0b001).unwrap(),
            IrValue::make_ubits(8, 1).unwrap(),
            IrValue::make_ubits(8, 2).unwrap(),
            IrValue::make_ubits(8, 3).unwrap(),
        ]);
        engine.add_corpus_sample_from_arg_tuple(&t1).unwrap();

        let r1 = engine.get_mux_outcome_report();
        let e1 = r1
            .entries
            .iter()
            .find(|e| e.node_text_id == 10)
            .expect("node 10 present");
        assert!(e1.missing.contains(&MuxOutcomeId::MultiBitsSet));

        // Now seed with a multi-bit selector (should observe MultiBitsSet).
        let t2 = IrValue::make_tuple(&[
            IrValue::make_ubits(3, 0b101).unwrap(),
            IrValue::make_ubits(8, 1).unwrap(),
            IrValue::make_ubits(8, 2).unwrap(),
            IrValue::make_ubits(8, 3).unwrap(),
        ]);
        engine.add_corpus_sample_from_arg_tuple(&t2).unwrap();

        let r2 = engine.get_mux_outcome_report();
        let e2 = r2
            .entries
            .iter()
            .find(|e| e.node_text_id == 10)
            .expect("node 10 present");
        assert!(!e2.missing.contains(&MuxOutcomeId::MultiBitsSet));
    }

    #[test]
    fn bools_hash_index_changes_when_a_computed_bool_changes() {
        let ir_text = r#"package test

fn f(x: bits[2] id=1) -> bits[1] {
  c0: bits[2] = literal(value=0, id=9)
  ret z: bits[1] = eq(x, c0, id=10)
}
"#;
        let mut engine = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(0),
            },
        )
        .unwrap();

        let t0 = IrValue::make_tuple(&[IrValue::make_ubits(2, 0).unwrap()]);
        let b0 = ir_bits_from_value_with_type(&t0, &engine.args_tuple_type);
        let f0 = engine.evaluate_candidate_features(&b0);
        let _ = engine.apply_candidate_features(&f0);
        let set_after_first = engine.bools_map.set_count();
        assert_eq!(set_after_first, 1);

        let t1 = IrValue::make_tuple(&[IrValue::make_ubits(2, 1).unwrap()]);
        let b1 = ir_bits_from_value_with_type(&t1, &engine.args_tuple_type);
        let f1 = engine.evaluate_candidate_features(&b1);
        let _ = engine.apply_candidate_features(&f1);
        assert_ne!(f0.bools_index, f1.bools_index);
        assert_eq!(engine.bools_map.set_count(), 2);
    }

    #[test]
    fn corner_and_failure_maps_set_bits_on_new_events() {
        // Corner coverage: add rhs == 0 should set a bit in the corner map.
        let ir_corner = r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret z: bits[8] = add(x, y, id=10)
}
"#;
        let mut corner_engine = AutocovEngine::from_ir_text(
            ir_corner,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(0),
            },
        )
        .unwrap();
        let t_corner = IrValue::make_tuple(&[
            IrValue::make_ubits(8, 7).unwrap(),
            IrValue::make_ubits(8, 0).unwrap(),
        ]);
        let b_corner = ir_bits_from_value_with_type(&t_corner, &corner_engine.args_tuple_type);
        let f_corner = corner_engine.evaluate_candidate_features(&b_corner);
        let _ = corner_engine.apply_candidate_features(&f_corner);
        assert_eq!(corner_engine.corner_map.set_count(), 1);

        // Failure coverage: array_index OOB with assumed_in_bounds=true should set a
        // bit in the failure map.
        let ir_fail = r#"package test

fn f(a: bits[8][4] id=1, i: bits[3] id=2) -> bits[8] {
  ret out: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=11)
}
"#;
        let mut fail_engine = AutocovEngine::from_ir_text(
            ir_fail,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(0),
            },
        )
        .unwrap();
        let arr = IrValue::make_array(&[
            IrValue::make_ubits(8, 10).unwrap(),
            IrValue::make_ubits(8, 11).unwrap(),
            IrValue::make_ubits(8, 12).unwrap(),
            IrValue::make_ubits(8, 13).unwrap(),
        ])
        .unwrap();
        let t_fail = IrValue::make_tuple(&[arr, IrValue::make_ubits(3, 7).unwrap()]);
        let b_fail = ir_bits_from_value_with_type(&t_fail, &fail_engine.args_tuple_type);
        let f_fail = fail_engine.evaluate_candidate_features(&b_fail);
        assert!(fail_engine.maybe_add_to_corpus(b_fail.clone(), &f_fail, None));
        assert_eq!(fail_engine.failure_map.set_count(), 1);
    }

    #[test]
    fn compare_distance_buckets_set_distinct_corner_indices() {
        let ir_text = r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[1] {
  ret t: bits[1] = eq(x, y, id=10)
}
"#;
        let mut engine = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(0),
            },
        )
        .unwrap();

        let t0 = IrValue::make_tuple(&[
            IrValue::make_ubits(8, 0xAA).unwrap(),
            IrValue::make_ubits(8, 0xAA).unwrap(),
        ]);
        let b0 = ir_bits_from_value_with_type(&t0, &engine.args_tuple_type);
        let f0 = engine.evaluate_candidate_features(&b0);
        assert_eq!(f0.compare_distance_indices.len(), 1);
        let _ = engine.apply_candidate_features(&f0);

        let t1 = IrValue::make_tuple(&[
            IrValue::make_ubits(8, 0xAA).unwrap(),
            IrValue::make_ubits(8, 0xAB).unwrap(),
        ]);
        let b1 = ir_bits_from_value_with_type(&t1, &engine.args_tuple_type);
        let f1 = engine.evaluate_candidate_features(&b1);
        assert_eq!(f1.compare_distance_indices.len(), 1);
        assert_ne!(
            f0.compare_distance_indices[0],
            f1.compare_distance_indices[0]
        );
        let _ = engine.apply_candidate_features(&f1);

        assert_eq!(engine.compare_distance_map.set_count(), 2);
    }

    #[test]
    fn corner_event_domain_enumerates_expected_tags() {
        // IR includes one instance of each corner-like node kind.
        let ir_text = r#"package test

fn f(
  x: bits[8] id=1,
  y: bits[8] id=2,
  shamt: bits[8] id=3,
  start: bits[8] id=4,
  arr: bits[8][4] id=5
) -> bits[8] {
  add0: bits[8] = add(x, y, id=10)
  neg0: bits[8] = neg(x, id=11)
  sx0: bits[16] = sign_ext(x, new_bit_count=16, id=12)
  cd0: bits[1] = eq(x, y, id=13)
  sh0: bits[8] = shll(x, shamt, id=14)
  shra0: bits[8] = shra(x, shamt, id=15)
  dbs0: bits[3] = dynamic_bit_slice(x, start, width=3, id=16)
  ai0: bits[8] = array_index(arr, indices=[start], assumed_in_bounds=false, id=17)
  ret r: bits[8] = identity(add0, id=18)
}
"#;
        let engine = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(0),
            },
        )
        .unwrap();

        let domain = engine.corner_event_domain();

        let has = |node_text_id: usize, kind: CornerKind, tag: u8| -> bool {
            domain.contains(&CornerEventId {
                node_text_id,
                kind,
                tag,
            })
        };

        assert!(has(10, CornerKind::Add, 0));
        assert!(has(10, CornerKind::Add, 1));
        assert!(has(11, CornerKind::Neg, 1));
        assert!(has(11, CornerKind::Neg, 0));
        assert!(has(12, CornerKind::SignExt, 0));

        // 8-bit compare can only reach xor-popcount 0..=8, which maps to buckets 0..=5.
        for tag in 0u8..=5u8 {
            assert!(has(13, CornerKind::CompareDistance, tag));
        }
        for tag in [0u8, 1u8, 2u8] {
            assert!(has(14, CornerKind::Shift, tag));
        }
        for tag in 0u8..=3u8 {
            assert!(has(15, CornerKind::Shra, tag));
        }
        for tag in [0u8, 1u8] {
            assert!(has(16, CornerKind::DynamicBitSlice, tag));
            assert!(has(17, CornerKind::ArrayIndex, tag));
        }
    }

    #[test]
    fn compare_distance_domain_does_not_include_unreachable_buckets_for_small_widths() {
        // For an 8-bit compare, xor-popcount is in 0..=8, which only reaches buckets
        // 0..=5.
        let ir_text = r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[1] {
  ret t: bits[1] = eq(x, y, id=10)
}
"#;
        let engine = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(0),
            },
        )
        .unwrap();

        let domain = engine.corner_event_domain();
        for tag in 0u8..=5u8 {
            assert!(
                domain.contains(&CornerEventId {
                    node_text_id: 10,
                    kind: CornerKind::CompareDistance,
                    tag
                }),
                "expected tag {} to be present for 8-bit compare",
                tag
            );
        }
        for tag in [6u8, 7u8] {
            assert!(
                !domain.contains(&CornerEventId {
                    node_text_id: 10,
                    kind: CornerKind::CompareDistance,
                    tag
                }),
                "did not expect tag {} to be present for 8-bit compare",
                tag
            );
        }
    }

    #[test]
    fn exhaustive_style_observation_2bit_space_is_deterministic() {
        let ir_text = r#"package test

fn f(x: bits[2] id=1) -> bits[2] {
  ret z: bits[2] = identity(x, id=2)
}
"#;
        let cfg = AutocovConfig {
            seed: 0,
            max_iters: Some(0),
        };
        let mut a = AutocovEngine::from_ir_text(ir_text, None, "f", cfg.clone()).unwrap();
        let mut b = AutocovEngine::from_ir_text(ir_text, None, "f", cfg).unwrap();

        // Enumerate all 2^2 candidates.
        for ctr in 0u8..4u8 {
            let bits = IrBits::from_lsb_is_0(&[(ctr & 1) != 0, (ctr & 2) != 0]);
            let _ = a.observe_candidate(&bits);
            let _ = b.observe_candidate(&bits);
        }

        assert_eq!(a.mux_features_set(), b.mux_features_set());
        assert_eq!(a.path_features_set(), b.path_features_set());
        assert_eq!(a.bools_features_set(), b.bools_features_set());
        assert_eq!(a.corner_features_set(), b.corner_features_set());
        assert_eq!(
            a.compare_distance_features_set(),
            b.compare_distance_features_set()
        );
        assert_eq!(a.failure_features_set(), b.failure_features_set());
        assert_eq!(a.mux_outcomes_observed(), b.mux_outcomes_observed());
    }

    #[test]
    fn exhaustive_parallel_matches_single_thread_metrics() {
        let ir_text = r#"package test

fn f(x: bits[4] id=1, y: bits[4] id=2) -> bits[1] {
  ret t: bits[1] = eq(x, y, id=10)
}
"#;
        let cfg = AutocovConfig {
            seed: 0,
            max_iters: Some(0),
        };
        let mut single = AutocovEngine::from_ir_text(ir_text, None, "f", cfg.clone()).unwrap();
        let mut parallel = AutocovEngine::from_ir_text(ir_text, None, "f", cfg).unwrap();

        // Single-thread: observe all candidates.
        for ctr in 0u16..(1u16 << 8) {
            let bits = IrBits::from_lsb_is_0(&[
                (ctr & 0x01) != 0,
                (ctr & 0x02) != 0,
                (ctr & 0x04) != 0,
                (ctr & 0x08) != 0,
                (ctr & 0x10) != 0,
                (ctr & 0x20) != 0,
                (ctr & 0x40) != 0,
                (ctr & 0x80) != 0,
            ]);
            let _ = single.observe_candidate(&bits);
        }

        // Parallel-style: evaluate observations on two independent evaluators and apply
        // to a single accumulating engine in arbitrary order.
        let eval_a = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(0),
            },
        )
        .unwrap();
        let eval_b = AutocovEngine::from_ir_text(
            ir_text,
            None,
            "f",
            AutocovConfig {
                seed: 0,
                max_iters: Some(0),
            },
        )
        .unwrap();
        for ctr in 0u16..(1u16 << 8) {
            let bits = IrBits::from_lsb_is_0(&[
                (ctr & 0x01) != 0,
                (ctr & 0x02) != 0,
                (ctr & 0x04) != 0,
                (ctr & 0x08) != 0,
                (ctr & 0x10) != 0,
                (ctr & 0x20) != 0,
                (ctr & 0x40) != 0,
                (ctr & 0x80) != 0,
            ]);
            let obs = if (ctr & 1) == 0 {
                eval_a.evaluate_observation(&bits)
            } else {
                eval_b.evaluate_observation(&bits)
            };
            parallel.apply_observation(&obs);
        }

        assert_eq!(single.mux_features_set(), parallel.mux_features_set());
        assert_eq!(single.path_features_set(), parallel.path_features_set());
        assert_eq!(single.bools_features_set(), parallel.bools_features_set());
        assert_eq!(single.corner_features_set(), parallel.corner_features_set());
        assert_eq!(
            single.compare_distance_features_set(),
            parallel.compare_distance_features_set()
        );
        assert_eq!(
            single.failure_features_set(),
            parallel.failure_features_set()
        );
        assert_eq!(
            single.mux_outcomes_observed(),
            parallel.mux_outcomes_observed()
        );
    }

    #[test]
    fn case_study_finds_xls_bang_tuple_in_corpus() {
        // The intent is to exercise "classic" magic-byte comparisons and ensure the
        // coverage guidance (especially compare-distance + bools hashing) can
        // find the satisfying input.
        let ir_text = r#"package test

fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3, d: bits[8] id=4) -> bits[1] {
  x: bits[8] = literal(value=0x58, id=10)
  l: bits[8] = literal(value=0x4c, id=11)
  s: bits[8] = literal(value=0x53, id=12)
  bang: bits[8] = literal(value=0x21, id=13)
  eq_a: bits[1] = eq(a, x, id=20)
  eq_b: bits[1] = eq(b, l, id=21)
  eq_c: bits[1] = eq(c, s, id=22)
  eq_d: bits[1] = eq(d, bang, id=23)
  ab: bits[1] = and(eq_a, eq_b, id=30)
  cd: bits[1] = and(eq_c, eq_d, id=31)
  ret r: bits[1] = and(ab, cd, id=32)
}
"#;

        let cfg = AutocovConfig {
            seed: 0,
            max_iters: Some(10_000),
        };
        let mut engine = AutocovEngine::from_ir_text(ir_text, None, "f", cfg).unwrap();
        assert_eq!(
            engine.input_slices_by_width.get(&8).map(Vec::len),
            Some(4),
            "expected four 8-bit input slices (one per u8 param)"
        );
        let dict8: Vec<u64> = engine
            .bit_constant_dict
            .iter()
            .filter(|c| c.bit_count == 8)
            .map(|c| c.value_u64)
            .collect();
        assert!(
            dict8.contains(&(b'X' as u64))
                && dict8.contains(&(b'L' as u64))
                && dict8.contains(&(b'S' as u64))
                && dict8.contains(&(b'!' as u64)),
            "expected dictionary to contain X/L/S/! literals; got dict8={:?}",
            dict8
        );
        let _ = engine.seed_structured_corpus(/* two_hot_max_bits= */ 64, None);
        let mut first_hit_iter: Option<u64> = None;
        let max_iters = engine.max_iters.unwrap();
        for iter in 0..max_iters {
            let cand = engine.generate_proposal();
            let features = engine.evaluate_candidate_features(&cand);
            let added = engine.maybe_add_to_corpus(cand, &features, None);
            if !added {
                continue;
            }

            // Check whether the newly-added candidate is the satisfying tuple.
            let last = engine.corpus.last().unwrap();
            let tuple = ir_value_from_bits_with_type(last, &engine.args_tuple_type);
            let elems = tuple.get_elements().unwrap();
            assert_eq!(elems.len(), 4);
            let av = elems[0].to_bits().unwrap().to_u64().unwrap() as u8;
            let bv = elems[1].to_bits().unwrap().to_u64().unwrap() as u8;
            let cv = elems[2].to_bits().unwrap().to_u64().unwrap() as u8;
            let dv = elems[3].to_bits().unwrap().to_u64().unwrap() as u8;
            if (av, bv, cv, dv) == (b'X', b'L', b'S', b'!') {
                first_hit_iter = Some(iter + 1);
                break;
            }
        }

        fn corpus_contains_tuple(engine: &AutocovEngine, a: u8, b: u8, c: u8, d: u8) -> bool {
            for bits in &engine.corpus {
                let tuple = ir_value_from_bits_with_type(bits, &engine.args_tuple_type);
                let elems = tuple.get_elements().unwrap();
                assert_eq!(elems.len(), 4);
                let av = elems[0].to_bits().unwrap().to_u64().unwrap() as u8;
                let bv = elems[1].to_bits().unwrap().to_u64().unwrap() as u8;
                let cv = elems[2].to_bits().unwrap().to_u64().unwrap() as u8;
                let dv = elems[3].to_bits().unwrap().to_u64().unwrap() as u8;
                if (av, bv, cv, dv) == (a, b, c, d) {
                    return true;
                }
            }
            false
        }

        // Note: stdout is captured by default; use `cargo test -- --nocapture` to see
        // this.
        println!(
            "case_study first_hit_iter={:?} corpus_len={}",
            first_hit_iter,
            engine.corpus.len()
        );

        assert!(
            corpus_contains_tuple(&engine, b'X', b'L', b'S', b'!'),
            "did not find expected tuple in corpus after iters={} corpus_len={}",
            engine.max_iters.unwrap(),
            engine.corpus.len()
        );
    }
}
