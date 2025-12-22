// SPDX-License-Identifier: Apache-2.0

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
use xlsynth_pir::ir;
use xlsynth_pir::ir_eval::{EvalObserver, SelectEvent, SelectKind};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_value_utils::{ir_bits_from_value_with_type, ir_value_from_bits_with_type};

pub const FEATURE_MAP_SIZE: usize = 65_536;

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

pub trait CorpusSink {
    fn on_new_sample(&mut self, tuple_value: &IrValue);
}

#[derive(Debug, Clone, Copy)]
pub struct AutocovProgress {
    pub iters: u64,
    pub corpus_len: usize,
    pub mux_features_set: usize,
    pub path_features_set: usize,
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
    bytes: [u8; FEATURE_MAP_SIZE],
    set_count: usize,
}

impl FeatureMap64k {
    fn new() -> Self {
        Self {
            bytes: [0u8; FEATURE_MAP_SIZE],
            set_count: 0,
        }
    }

    fn observe_index(&mut self, idx: usize) -> bool {
        let slot = &mut self.bytes[idx];
        if *slot == 0 {
            *slot = 1;
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
}

#[derive(Debug, Clone)]
struct CandidateFeatures {
    mux_indices: Vec<usize>,
    path_index: usize,
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

#[derive(Debug)]
struct CollectingMuxObserver {
    mux_indices: Vec<usize>,
    path_hasher: Hasher,
    mux_outcomes: Vec<(usize, MuxOutcomeId)>,
}

impl CollectingMuxObserver {
    fn new() -> Self {
        let mut path_hasher = Hasher::new();
        path_hasher.update(b"xlsynth-autocov:path");
        Self {
            mux_indices: Vec::new(),
            path_hasher,
            mux_outcomes: Vec::new(),
        }
    }

    fn finish(self) -> CandidateFeatures {
        let path_hash = self.path_hasher.finalize();
        let path_index =
            u16::from_le_bytes([path_hash.as_bytes()[0], path_hash.as_bytes()[1]]) as usize;
        CandidateFeatures {
            mux_indices: self.mux_indices,
            path_index,
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
}

pub struct AutocovEngine {
    f: ir::Fn,
    args_tuple_type: ir::Type,
    args_bit_count: usize,
    rng: StdRng,
    max_iters: Option<u64>,
    stop: Arc<AtomicBool>,

    mux_map: FeatureMap64k,
    path_map: FeatureMap64k,

    corpus: Vec<IrBits>,
    corpus_hashes: BTreeSet<[u8; 32]>,

    mux_outcome_spaces: BTreeMap<usize, MuxOutcomeSpace>,
    mux_outcome_observed: BTreeMap<usize, Vec<bool>>,
    mux_outcomes_possible_total: usize,
    mux_outcomes_observed_total: usize,
}

impl AutocovEngine {
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
            mux_map: FeatureMap64k::new(),
            path_map: FeatureMap64k::new(),
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
        Ok(engine)
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
            // 30%: multi-bit havoc
            10..=39 => {
                let parent = self.pick_parent().clone();
                self.mutate_havoc(parent)
            }
            // 25%: arbitrary subslice crossover
            40..=64 => {
                let a = self.pick_parent().clone();
                let b = self.pick_parent().clone();
                self.crossover_subslice(a, b)
            }
            // 20%: XOR-mix
            65..=84 => {
                let a = self.pick_parent().clone();
                let b = self.pick_parent().clone();
                self.xor_mix(a, b)
            }
            // 15%: single-bit flip
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

        Observations { mux_new, path_new }
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
        if !obs.mux_new && !obs.path_new {
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
}
