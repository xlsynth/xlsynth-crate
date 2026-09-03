// SPDX-License-Identifier: Apache-2.0

//! Deterministic semantic-feature coverage, separate from libFuzzer edge
//! coverage.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use serde::Serialize;
use xlsynth::external_tool::ToolError;
use xlsynth::{IrBits, IrValue};
use xlsynth_g8r_fuzz::random_block::block_output_refs;
use xlsynth_pir::ir::{Binop, BlockMetadata, Fn, NodePayload, NodeRef, Package, Type};
use xlsynth_pir::ir_random::gather_block_stats;
use xlsynth_pir::ir_utils::operands;
use xlsynth_pir::ir_value_utils::ir_bits_to_usize_in_range;

use crate::{block_options, operations, top_block};

/// Records whether a generated case reached and passed a semantic oracle.
pub enum Outcome<'a> {
    GeneratedOnly,
    Checked,
    Inconclusive(&'a ToolError),
}

#[derive(Default, Serialize)]
pub struct CoverageReport {
    pub generated_cases: u64,
    pub checked_cases: u64,
    pub inconclusive_cases: BTreeMap<String, u64>,
    pub generated_operations: BTreeMap<String, u64>,
    pub checked_graph_operations: BTreeMap<String, u64>,
    pub checked_live_operations: BTreeMap<String, u64>,
    pub bit_widths: BTreeMap<usize, u64>,
    pub array_lengths: BTreeMap<usize, u64>,
    pub tuple_lengths: BTreeMap<usize, u64>,
    pub type_depths: BTreeMap<usize, u64>,
    pub selector_widths: BTreeMap<usize, u64>,
    pub multiply_shapes: BTreeMap<String, u64>,
    pub div_mod_widths: BTreeMap<String, u64>,
    pub attributes: BTreeMap<String, u64>,
    pub register_counts: BTreeMap<usize, u64>,
    pub node_counts: BTreeMap<usize, u64>,
    pub producer_consumer_pairs: BTreeMap<String, u64>,
    pub checked_live_pairs: BTreeMap<String, u64>,
    pub operand_result_widths: BTreeMap<String, u64>,
    pub register_dependency_depths: BTreeMap<String, u64>,
    pub observed_live_behaviors: BTreeMap<String, u64>,
    pub checked_vectors: u64,
    pub rejected_inputs: BTreeMap<String, u64>,
    pub engines: BTreeMap<String, u64>,
    pub generated_data_widths: BTreeMap<String, u64>,
    pub checked_data_widths: BTreeMap<String, u64>,
    pub checked_live_data_widths: BTreeMap<String, u64>,
    pub generated_multiply_categories: BTreeMap<String, u64>,
    pub checked_multiply_categories: BTreeMap<String, u64>,
    pub checked_live_multiply_categories: BTreeMap<String, u64>,
    pub unique_graphs: usize,
    pub unique_graph_stimulus_pairs: usize,
    #[serde(skip)]
    graph_hashes: BTreeSet<[u8; 32]>,
    #[serde(skip)]
    stimulus_hashes: BTreeSet<[u8; 32]>,
}

impl CoverageReport {
    /// Records graph identity separately from independently varied stimuli.
    pub fn record_case(
        &mut self,
        case: &crate::input::FuzzCase,
        outcome: Outcome<'_>,
        trace: Option<&crate::semantics::Trace>,
    ) {
        let checked = matches!(outcome, Outcome::Checked);
        self.record(&case.package, outcome);
        *self.engines.entry(case.engine().into()).or_default() += 1;
        let mut hash = blake3::Hasher::new();
        hash.update(case.package.to_string().as_bytes());
        hash.update(&case.stimulus_seed);
        self.stimulus_hashes.insert(*hash.finalize().as_bytes());
        self.unique_graph_stimulus_pairs = self.stimulus_hashes.len();
        if let Some(trace) = trace.filter(|_| checked) {
            self.record_trace(trace);
        }
    }

    /// Adds a graph without requiring that all its operations be observable.
    pub fn record(&mut self, package: &Package, outcome: Outcome<'_>) {
        self.generated_cases += 1;
        self.graph_hashes
            .insert(*blake3::hash(package.to_string().as_bytes()).as_bytes());
        self.unique_graphs = self.graph_hashes.len();
        let checked = matches!(outcome, Outcome::Checked);
        match outcome {
            Outcome::Checked => self.checked_cases += 1,
            Outcome::Inconclusive(error) => {
                *self
                    .inconclusive_cases
                    .entry(error.reason_key())
                    .or_default() += 1
            }
            Outcome::GeneratedOnly => { /* A corpus census need not execute an oracle. */ }
        }
        let (block, metadata) = top_block(package);
        let live = live_nodes(block, metadata);
        let depth = register_dependency_depth(block, metadata);
        *self.register_dependency_depths.entry(depth).or_default() += 1;
        let stats = gather_block_stats(block);
        *self
            .node_counts
            .entry(stats.emitted_node_count)
            .or_default() += 1;
        *self
            .register_counts
            .entry(metadata.registers.len())
            .or_default() += 1;
        for (op, count) in stats.emitted_operations {
            *self.generated_operations.entry(op.clone()).or_default() += count as u64;
            if checked {
                *self.checked_graph_operations.entry(op).or_default() += count as u64;
            }
        }
        if checked {
            for (op, count) in stats.live_operations {
                *self.checked_live_operations.entry(op).or_default() += count as u64;
            }
        }
        for (index, node) in block.nodes.iter().enumerate().skip(1) {
            for (position, operand) in operands(&node.payload).into_iter().enumerate() {
                let producer = block.get_node(operand);
                if let Type::Bits(width) = &producer.ty
                    && is_data_operand(&node.payload, position)
                {
                    let key = format!("{}:{width}", node.payload.get_operator());
                    *self.generated_data_widths.entry(key.clone()).or_default() += 1;
                    if checked {
                        *self.checked_data_widths.entry(key.clone()).or_default() += 1;
                    }
                    if checked && live[index] {
                        *self.checked_live_data_widths.entry(key).or_default() += 1;
                    }
                }
                let pair = format!(
                    "{} -> {}",
                    producer.payload.get_operator(),
                    node.payload.get_operator()
                );
                *self
                    .producer_consumer_pairs
                    .entry(pair.clone())
                    .or_default() += 1;
                if checked && live[index] {
                    *self.checked_live_pairs.entry(pair).or_default() += 1;
                }
                let relation = match producer.ty.bit_count().cmp(&node.ty.bit_count()) {
                    std::cmp::Ordering::Less => "narrower",
                    std::cmp::Ordering::Equal => "equal",
                    std::cmp::Ordering::Greater => "wider",
                };
                *self
                    .operand_result_widths
                    .entry(format!(
                        "{}:operand-{relation}-than-result",
                        node.payload.get_operator()
                    ))
                    .or_default() += 1;
            }
            self.record_type(&node.ty, 0);
            match &node.payload {
                NodePayload::Sel {
                    selector, default, ..
                }
                | NodePayload::PrioritySel {
                    selector, default, ..
                } => {
                    *self
                        .selector_widths
                        .entry(block.get_node_ty(*selector).bit_count())
                        .or_default() += 1;
                    *self
                        .attributes
                        .entry(format!("select_default={}", default.is_some()))
                        .or_default() += 1;
                }
                NodePayload::OneHotSel { selector, .. } => {
                    *self
                        .selector_widths
                        .entry(block.get_node_ty(*selector).bit_count())
                        .or_default() += 1;
                }
                NodePayload::ArrayIndex {
                    assumed_in_bounds,
                    indices,
                    ..
                }
                | NodePayload::ArrayUpdate {
                    assumed_in_bounds,
                    indices,
                    ..
                } => {
                    *self
                        .attributes
                        .entry(format!("assumed_in_bounds={assumed_in_bounds}"))
                        .or_default() += 1;
                    *self
                        .attributes
                        .entry(format!("array_index_depth={}", indices.len()))
                        .or_default() += 1;
                }
                NodePayload::Binop(Binop::Umul | Binop::Smul, lhs, rhs) => {
                    let l = block.get_node_ty(*lhs).bit_count();
                    let r = block.get_node_ty(*rhs).bit_count();
                    let result = node.ty.bit_count();
                    let inputs = if l == r {
                        "equal"
                    } else if l < r {
                        "lhs-narrower"
                    } else {
                        "lhs-wider"
                    };
                    let output = if result < l + r {
                        "truncated"
                    } else if result == l + r {
                        "full"
                    } else {
                        "extended"
                    };
                    let category = format!("{}:{inputs}:{output}", node.payload.get_operator());
                    *self
                        .generated_multiply_categories
                        .entry(category.clone())
                        .or_default() += 1;
                    if checked {
                        *self
                            .checked_multiply_categories
                            .entry(category.clone())
                            .or_default() += 1;
                    }
                    if checked && live[index] {
                        *self
                            .checked_live_multiply_categories
                            .entry(category)
                            .or_default() += 1;
                    }
                    // Power-of-two buckets bound the histogram size in long
                    // campaigns.
                    let shape = format!(
                        "{}:<= {} x <= {} -> <= {}",
                        node.payload.get_operator(),
                        block.get_node_ty(*lhs).bit_count().next_power_of_two(),
                        block.get_node_ty(*rhs).bit_count().next_power_of_two(),
                        node.ty.bit_count().next_power_of_two()
                    );
                    *self.multiply_shapes.entry(shape).or_default() += 1;
                }
                NodePayload::Binop(Binop::Udiv | Binop::Sdiv | Binop::Umod | Binop::Smod, _, _) => {
                    *self
                        .div_mod_widths
                        .entry(format!(
                            "{}:{}",
                            node.payload.get_operator(),
                            node.ty.bit_count()
                        ))
                        .or_default() += 1;
                }
                NodePayload::RegisterWrite {
                    load_enable, reset, ..
                } => {
                    *self
                        .attributes
                        .entry(format!("load_enable={}", load_enable.is_some()))
                        .or_default() += 1;
                    *self
                        .attributes
                        .entry(format!("register_reset={}", reset.is_some()))
                        .or_default() += 1;
                }
                _ => { /* The opcode/type census covers other operation families. */ }
            }
        }
        if let Some(reset) = &metadata.reset {
            *self
                .attributes
                .entry(format!("reset_active_low={}", reset.active_low))
                .or_default() += 1;
            *self
                .attributes
                .entry(format!("reset_asynchronous={}", reset.asynchronous))
                .or_default() += 1;
        }
    }

    /// Records behaviors from stimuli that passed every required oracle.
    pub fn record_trace(&mut self, trace: &crate::semantics::Trace) {
        self.checked_vectors += trace.samples.len() as u64;
        for (behavior, count) in &trace.observed_live_behaviors {
            *self
                .observed_live_behaviors
                .entry(behavior.clone())
                .or_default() += count;
        }
    }

    fn record_type(&mut self, ty: &Type, depth: usize) {
        *self.type_depths.entry(depth).or_default() += 1;
        match ty {
            Type::Bits(width) => *self.bit_widths.entry(*width).or_default() += 1,
            Type::Array(array) => {
                *self.array_lengths.entry(array.element_count).or_default() += 1;
                self.record_type(&array.element_type, depth + 1);
            }
            Type::Tuple(fields) => {
                *self.tuple_lengths.entry(fields.len()).or_default() += 1;
                for field in fields {
                    self.record_type(field, depth + 1);
                }
            }
            Type::Token => { /* No bit width for event tokens. */ }
        }
    }

    /// Serializes stable maps and explicit missing-op lists, including dead
    /// ops.
    pub fn json(&self) -> String {
        let operations = operations(true);
        let limits = block_options(true, true).function_options;
        let missing_generated: Vec<_> = operations
            .iter()
            .map(|op| op.name())
            .filter(|op| !self.generated_operations.contains_key(*op))
            .collect();
        let missing_checked: Vec<_> = operations
            .iter()
            .map(|op| op.name())
            .filter(|op| !self.checked_graph_operations.contains_key(*op))
            .collect();
        let missing_live: Vec<_> = operations
            .iter()
            .map(|op| op.name())
            .filter(|op| !self.checked_live_operations.contains_key(*op))
            .collect();
        serde_json::to_string(&serde_json::json!({
            "coverage": self,
            "generator_version": crate::input::GENERATOR_VERSION,
            "width_policy": "balanced data widths; budgeted operand adaptation",
            "unique_count_scope": "exact within this reporter; not additive across workers",
            "input_format_version": crate::input::FORMAT_VERSION,
            "configured_generation_width_limits": {
                "general": limits.max_bit_width,
                "multiply_operand": limits.max_multiply_operand_bit_width,
                "div_mod_operand_and_result": limits.max_div_mod_bit_width,
            },
            "observed_behavior_scope": "live-PIR-nodes-on-vectors-passing-all-required-oracles",
            "feature_histogram_scope": "all-generated-cases",
            "excluded_operations": ["invoke", "counted_for", "umulp", "smulp", "after_all", "assert", "cover", "trace"],
            "zero_width_generation": true,
            "public_extensions": true,
            "missing_generated_operations": missing_generated,
            "missing_checked_operations": missing_checked,
            "missing_checked_live_operations": missing_live,
        })).expect("coverage consists of JSON-compatible counters")
    }
}

#[cfg(test)]
mod tests {
    use super::{CoverageReport, Outcome};
    use crate::parse_reference;

    #[test]
    fn data_widths_distinguish_operand_roles_even_when_nodes_are_shared() {
        let package = parse_reference(
            r#"package roles
top block roles(x: bits[8], out: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  shifted: bits[8] = shll(x, x, id=2)
  out: () = output_port(shifted, name=out, id=3)
}
"#,
        );
        let mut report = CoverageReport::default();
        report.record(&package, Outcome::GeneratedOnly);
        assert_eq!(report.generated_data_widths["shll:8"], 1);
        assert!(report.checked_live_data_widths.is_empty());
        report.record(&package, Outcome::Checked);
        assert_eq!(report.checked_data_widths["shll:8"], 1);
        assert_eq!(report.checked_live_data_widths["shll:8"], 1);
        assert_eq!(report.unique_graphs, 1);
    }

    #[test]
    fn unique_graphs_are_separate_from_stimulus_and_presentation_variations() {
        let mut case = crate::input::FuzzCase::random(2);
        let mut report = CoverageReport::default();
        report.record_case(&case, Outcome::GeneratedOnly, None);
        report.record_case(&case, Outcome::GeneratedOnly, None);
        case.options.max_inline_depth += 1;
        report.record_case(&case, Outcome::GeneratedOnly, None);
        assert_eq!(report.unique_graphs, 1);
        assert_eq!(report.unique_graph_stimulus_pairs, 1);
        case.stimulus_seed[0] ^= 1;
        report.record_case(&case, Outcome::GeneratedOnly, None);
        assert_eq!(report.unique_graphs, 1);
        assert_eq!(report.unique_graph_stimulus_pairs, 2);
        assert_eq!(report.engines["random"], 4);
    }

    #[test]
    fn reports_independent_generation_width_limits() {
        let report: serde_json::Value =
            serde_json::from_str(&CoverageReport::default().json()).unwrap();
        assert_eq!(
            report["configured_generation_width_limits"],
            serde_json::json!({
                "general": 256,
                "multiply_operand": 64,
                "div_mod_operand_and_result": 16,
            })
        );
    }

    #[test]
    fn behavioral_census_distinguishes_actual_accesses_arms_and_dead_nodes() {
        let package = parse_reference(
            r#"package public_behaviors
top block behaviors(values: bits[8][3], index: bits[128], data: bits[65], read: bits[8], priority: bits[66], selected: bits[65], updated: bits[8][3]) {
  values: bits[8][3] = input_port(name=values, id=1)
  index: bits[128] = input_port(name=index, id=2)
  data: bits[65] = input_port(name=data, id=3)
  element: bits[8] = array_index(values, indices=[index], id=4)
  encoded: bits[66] = one_hot(data, lsb_prio=true, id=5)
  zero: bits[65] = literal(value=0, id=6)
  choice: bits[65] = sel(index, cases=[data, zero], default=zero, id=7)
  replaced: bits[8][3] = array_update(values, element, indices=[index], id=8)
  dead: bits[65] = udiv(data, zero, id=9)
  read: () = output_port(element, name=read, id=10)
  priority: () = output_port(encoded, name=priority, id=11)
  selected: () = output_port(choice, name=selected, id=12)
  updated: () = output_port(replaced, name=updated, id=13)
}
"#,
        );
        let trace = crate::semantics::Trace::for_package(&package);
        for event in [
            "array_index:in-bounds",
            "array_index:out-of-bounds",
            "array_update:in-bounds",
            "array_update:out-of-bounds",
            "sel:case-0",
            "sel:default",
            "one_hot:zero-input",
            "one_hot:nonzero-input",
        ] {
            assert!(trace.observed_live_behaviors[event] > 0, "{event}");
        }
        assert!(
            !trace
                .observed_live_behaviors
                .contains_key("udiv:zero-divisor")
        );
        let mut report = CoverageReport::default();
        report.record(&package, Outcome::GeneratedOnly);
        assert_eq!(
            report.div_mod_widths,
            std::collections::BTreeMap::from([("udiv:65".into(), 1)])
        );
        assert!(report.observed_live_behaviors.is_empty());
        report.record_trace(&trace);
        assert_eq!(report.checked_vectors, 16);
        assert!(
            report
                .producer_consumer_pairs
                .contains_key("get_param -> array_index")
        );
        assert!(
            report
                .operand_result_widths
                .contains_key("array_index:operand-wider-than-result")
        );
    }

    #[test]
    fn census_distinguishes_generated_checked_and_live_nodes() {
        let package = parse_reference(
            r#"package census
top block census(x: bits[8], y: bits[8], out: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  y: bits[8] = input_port(name=y, id=2)
  sum: bits[8] = add(x, y, id=3)
  dead: bits[8] = xor(x, y, id=4)
  out: () = output_port(sum, name=out, id=5)
}
"#,
        );
        let mut report = CoverageReport::default();
        report.record(&package, Outcome::GeneratedOnly);
        assert!(report.checked_graph_operations.is_empty());
        report.record(&package, Outcome::Checked);
        assert_eq!(report.generated_cases, 2);
        assert_eq!(report.checked_cases, 1);
        assert_eq!(report.generated_operations["add"], 2);
        assert_eq!(report.checked_graph_operations["xor"], 1);
        assert_eq!(report.checked_live_operations["add"], 1);
        assert!(!report.checked_live_operations.contains_key("xor"));
        let json = report.json();
        assert_eq!(json, report.json());
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            parsed["missing_checked_live_operations"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("xor"))
        );
        assert!(
            !parsed["missing_checked_operations"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("xor"))
        );
    }
}

thread_local! {
    static REPORTS: RefCell<BTreeMap<&'static str, (CoverageReport, Instant)>> = const { RefCell::new(BTreeMap::new()) };
}

/// Returns snapshots on inconclusive checks, the first case, every 4096 cases,
/// or after 30 seconds at a completed case. Callers own printing; generation is
/// unaffected.
pub fn record_progress(
    case: &crate::input::FuzzCase,
    outcome: Outcome<'_>,
    trace: Option<&crate::semantics::Trace>,
) -> Option<String> {
    REPORTS.with(|reports| {
        let mut reports = reports.borrow_mut();
        let (report, last) = reports
            .entry(case.engine())
            .or_insert_with(|| (CoverageReport::default(), Instant::now()));
        let inconclusive = matches!(outcome, Outcome::Inconclusive(_));
        report.record_case(case, outcome, trace);
        if inconclusive
            || report.generated_cases == 1
            || report.generated_cases % 4096 == 0
            || last.elapsed() >= Duration::from_secs(30)
        {
            *last = Instant::now();
            Some(report.json())
        } else {
            None
        }
    })
}

/// Separates scalar data widths from selector, index, reset and enable widths.
fn is_data_operand(payload: &NodePayload, position: usize) -> bool {
    match payload {
        NodePayload::Sel { .. }
        | NodePayload::PrioritySel { .. }
        | NodePayload::OneHotSel { .. } => position != 0,
        NodePayload::ArrayIndex { .. } | NodePayload::ArraySlice { .. } => position == 0,
        NodePayload::ArrayUpdate { .. } => position < 2,
        NodePayload::DynamicBitSlice { .. } | NodePayload::BitSliceUpdate { .. } => position != 1,
        NodePayload::Binop(Binop::Shll | Binop::Shrl | Binop::Shra, _, _) => position == 0,
        NodePayload::Binop(Binop::Gate, _, _) => position == 1,
        NodePayload::RegisterWrite { .. } => position == 0,
        NodePayload::ExtCarryOut { .. } => position < 2,
        _ => true,
    }
}

/// Marks dependencies of observed outputs and all checked register updates.
pub(crate) fn live_nodes(block: &Fn, metadata: &BlockMetadata) -> Vec<bool> {
    let mut pending = block_output_refs(block, metadata);
    pending.extend(block.nodes.iter().enumerate().filter_map(|(index, node)| {
        matches!(node.payload, NodePayload::RegisterWrite { .. }).then_some(NodeRef { index })
    }));
    let mut live = vec![false; block.nodes.len()];
    while let Some(node) = pending.pop() {
        if !std::mem::replace(&mut live[node.index], true) {
            pending.extend(operands(&block.get_node(node).payload));
        }
    }
    live
}

/// Reports longest register dependency paths, or feedback including indirect
/// cycles. Implicit load-enable holds do not count as explicit data feedback.
fn register_dependency_depth(block: &Fn, metadata: &BlockMetadata) -> String {
    let names: BTreeMap<_, _> = metadata
        .registers
        .iter()
        .enumerate()
        .map(|(i, r)| (r.name.as_str(), i))
        .collect();
    let mut predecessors = vec![vec![false; names.len()]; names.len()];
    for node in &block.nodes {
        if let NodePayload::RegisterWrite { register, .. } = &node.payload {
            let mut pending = operands(&node.payload);
            let mut visited = vec![false; block.nodes.len()];
            while let Some(operand) = pending.pop() {
                if std::mem::replace(&mut visited[operand.index], true) {
                    continue;
                }
                let producer = &block.get_node(operand).payload;
                if let NodePayload::RegisterRead { register: source } = producer {
                    predecessors[names[register.as_str()]][names[source.as_str()]] = true;
                }
                pending.extend(operands(producer));
            }
        }
    }
    let mut depths = vec![0usize; names.len()];
    for iteration in 0..=names.len() {
        let mut changed = false;
        for (target, sources) in predecessors.iter().enumerate() {
            let next = sources
                .iter()
                .enumerate()
                .filter(|(_, connected)| **connected)
                .map(|(source, _)| depths[source] + 1)
                .max()
                .unwrap_or(1);
            if next > depths[target] {
                depths[target] = next;
                changed = true;
            }
        }
        if !changed {
            return depths.into_iter().max().unwrap_or(0).to_string();
        }
        if iteration == names.len() {
            return "feedback".into();
        }
    }
    unreachable!()
}

/// Classifies actual evaluated operands; static presence is reported
/// separately.
pub(crate) fn record_behaviors(
    block: &Fn,
    metadata: &BlockMetadata,
    values: &[Option<IrValue>],
    live: &[bool],
    counts: &mut BTreeMap<String, u64>,
) {
    let bits =
        |node: NodeRef| -> IrBits { values[node.index].as_ref().unwrap().to_bits().unwrap() };
    for (index, node) in block.nodes.iter().enumerate().skip(1) {
        if !live[index] {
            continue;
        }
        let op = node.payload.get_operator();
        let event = match &node.payload {
            NodePayload::ArrayIndex { array, indices, .. }
            | NodePayload::ArrayUpdate { array, indices, .. } => {
                let mut ty = block.get_node_ty(*array);
                let mut in_bounds = true;
                for index in indices {
                    let Type::Array(array) = ty else {
                        unreachable!()
                    };
                    in_bounds &= ir_bits_to_usize_in_range(
                        &bits(*index),
                        array.element_count.saturating_sub(1),
                    )
                    .is_some();
                    ty = &array.element_type;
                }
                if in_bounds {
                    "in-bounds".to_string()
                } else {
                    "out-of-bounds".to_string()
                }
            }
            NodePayload::Sel {
                selector, cases, ..
            } => match ir_bits_to_usize_in_range(&bits(*selector), cases.len().saturating_sub(1)) {
                Some(index) if index < cases.len() => format!("case-{index}"),
                _ => "default".into(),
            },
            NodePayload::PrioritySel {
                selector, cases, ..
            } => {
                let selector = bits(*selector);
                (0..cases.len())
                    .find(|i| selector.get_bit(*i).unwrap())
                    .map(|index| format!("case-{index}"))
                    .unwrap_or_else(|| "default".into())
            }
            NodePayload::OneHotSel { selector, .. } => {
                let selector = bits(*selector);
                let population = (0..selector.get_bit_count())
                    .filter(|i| selector.get_bit(*i).unwrap())
                    .count();
                match population {
                    0 => "no-bits-set",
                    1 => "single-bit",
                    _ => "multiple-bits",
                }
                .into()
            }
            NodePayload::OneHot { arg, .. }
            | NodePayload::ExtPrioEncode { arg, .. }
            | NodePayload::ExtClz { arg, .. }
            | NodePayload::ExtNormalizeLeft { arg, .. } => if bits(*arg).is_zero() {
                "zero-input"
            } else {
                "nonzero-input"
            }
            .into(),
            NodePayload::Binop(Binop::Udiv | Binop::Sdiv | Binop::Umod | Binop::Smod, _, rhs) => {
                if bits(*rhs).is_zero() {
                    "zero-divisor"
                } else {
                    "nonzero-divisor"
                }
                .into()
            }
            NodePayload::RegisterWrite {
                load_enable, reset, ..
            } => {
                let enabled = load_enable
                    .map(|r| bits(r).get_bit(0).unwrap())
                    .unwrap_or(true);
                let reset = reset
                    .map(|r| {
                        bits(r).get_bit(0).unwrap() ^ metadata.reset.as_ref().unwrap().active_low
                    })
                    .unwrap_or(false);
                format!("enable={enabled},reset={reset}")
            }
            _ => {
                continue;
            } // Other families are covered by static interaction counts.
        };
        *counts.entry(format!("{op}:{event}")).or_default() += 1;
    }
}
