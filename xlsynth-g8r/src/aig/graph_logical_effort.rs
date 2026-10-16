// SPDX-License-Identifier: Apache-2.0

//! Graph-based logical effort worst-case delay estimation.
//!
//! With the NAND2 model, a path of length n and log effort a has delay
//! D(n, a) = n * exp(a / n) + 2*n. This is convex (the perspective of exp)
//! and increasing in a. If B lies on or below the chord AC in (n, a), then
//! after any common suffix s, D(B+s) <= max(D(A+s), D(C+s)). Thus each node
//! needs only the upper hull of its source-to-node prefixes. This set-wise
//! certificate is valid even when neither endpoint alone dominates B.
//!
//! Hull merges take linear work in their input sizes. Uniform-effort
//! reconvergent chains have constant-size hulls, but arbitrary graphs can
//! still require O(nodes * depth) work and traceback storage, excluding the
//! bit cost of exact arithmetic. The proof relies on constant parasitic
//! delay per stage; it does not extend unchanged to variable parasitics.

use crate::aig::topo::topo_sort_refs;
use crate::aig::{AigNode, AigRef, GateFn};
use num_bigint::{BigInt, Sign};
use num_traits::{ToPrimitive, Zero};
use std::collections::HashMap;
use std::num::NonZeroUsize;

/// A binary64 weight represented exactly as mantissa * 2^exponent.
struct Dyadic {
    mantissa: BigInt,
    exponent: i32,
}

impl Dyadic {
    /// Decodes finite binary64 values, including subnormals and signed zero.
    fn from_f64(value: f64) -> Self {
        assert!(value.is_finite(), "graph LE requires finite log weights");
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let mantissa = (bits & ((1u64 << 52) - 1)) | if exponent == 0 { 0 } else { 1u64 << 52 };
        Self {
            mantissa: BigInt::from(mantissa) * if bits >> 63 == 0 { 1 } else { -1 },
            exponent: if exponent == 0 {
                -1074
            } else {
                exponent - 1075
            },
        }
    }
}

/// Rounds an exact sum at the shared exponent to nearest-even binary64.
/// Constructing the result bits avoids overflow/underflow in an intermediate
/// conversion or power of two. The shared exponent is at least -1074.
fn scaled_log_to_f64(value: &BigInt, scale: i32) -> f64 {
    if value.is_zero() {
        return 0.0;
    }
    let sign = if value.sign() == Sign::Minus {
        1u64 << 63
    } else {
        0
    };
    let magnitude = value.magnitude();
    let bits = magnitude.bits();
    let mut exponent = bits as i64 - 1 + i64::from(scale);
    if exponent < -1022 {
        // A sum of binary64 dyadics cannot fall below the subnormal grid.
        debug_assert!(scale >= -1074);
        let fraction = (magnitude << (scale + 1074) as usize).to_u64().unwrap();
        return f64::from_bits(sign | fraction);
    }
    let mut significand = if bits <= 53 {
        magnitude.to_u64().unwrap() << (53 - bits)
    } else {
        let shift = bits - 53;
        let mut top = (magnitude >> shift as usize).to_u64().unwrap();
        let halfway = magnitude.bit(shift - 1);
        let sticky = magnitude.trailing_zeros().unwrap() < shift - 1;
        if halfway && (sticky || top & 1 != 0) {
            top += 1;
        }
        top
    };
    if significand == 1u64 << 53 {
        significand >>= 1;
        exponent += 1;
    }
    if exponent > 1023 {
        return f64::from_bits(sign | (0x7ffu64 << 52));
    }
    f64::from_bits(sign | (((exponent + 1023) as u64) << 52) | (significand & ((1u64 << 52) - 1)))
}

/// Edge log weights at one common binary exponent, with exact prefix sums.
struct LogWeights {
    // None denotes zero effort, whose logarithm is negative infinity.
    weights: Vec<Option<BigInt>>,
    exponent: i32,
}

impl LogWeights {
    /// Uses the public model's binary64 log weights, then sums them exactly.
    fn new(fanouts: &[usize], options: &GraphLogicalEffortOptions) -> Self {
        let raw = fanouts
            .iter()
            .map(|&fanout| {
                if fanout == 0 {
                    return Some(Dyadic::from_f64(0.0));
                }
                let f = fanout as f64;
                let h = options.beta1 * f + options.beta2 * f.powi(2);
                assert!(
                    h >= 0.0 && h.is_finite(),
                    "graph LE requires finite nonnegative pin loads"
                );
                if h == 0.0 {
                    None
                } else {
                    Some(Dyadic::from_f64((4.0_f64 / 3.0).ln() + h.ln()))
                }
            })
            .collect::<Vec<_>>();
        let exponent = raw
            .iter()
            .flatten()
            .filter(|x| !x.mantissa.is_zero())
            .map(|x| x.exponent)
            .min()
            .unwrap_or(0);
        let weights = raw
            .into_iter()
            .map(|x| {
                x.map(|x| {
                    if x.mantissa.is_zero() {
                        x.mantissa
                    } else {
                        x.mantissa << (x.exponent - exponent) as usize
                    }
                })
            })
            .collect();
        Self { weights, exponent }
    }
}

/// A vertex on a node's upper hull, with an optional immutable traceback link.
#[derive(Clone)]
struct State {
    n: usize,
    log_f: BigInt,
    prev: Option<NonZeroUsize>,
}

/// Once any edge has zero load, effort stays zero; only maximum length matters.
#[derive(Clone, Copy)]
struct ZeroEffortState {
    n: usize,
    prev: Option<NonZeroUsize>,
}

/// A finite upper hull and at most one zero-effort prefix, both with witnesses.
#[derive(Clone, Default)]
struct Frontier {
    hull: Vec<State>,
    zero: Option<ZeroEffortState>,
}

impl Frontier {
    /// Counts retained finite and zero-effort states.
    fn len(&self) -> usize {
        self.hull.len() + usize::from(self.zero.is_some())
    }

    /// Extends existing zero effort, or converts all prefixes on a zero edge.
    fn extend_zero(&self, zero_edge: bool) -> Option<ZeroEffortState> {
        let mut longest = self.zero;
        if zero_edge
            && let Some(last) = self.hull.last()
            && longest.is_none_or(|old| last.n > old.n)
        {
            longest = Some(ZeroEffortState {
                n: last.n,
                prev: last.prev,
            });
        }
        longest.map(|state| ZeroEffortState {
            n: state.n + 1,
            prev: state.prev,
        })
    }
}

/// Merges both finite hulls and the longest zero-effort prefix at an AND.
fn merge_frontiers(
    a: &Frontier,
    b: &Frontier,
    wa: Option<&BigInt>,
    wb: Option<&BigInt>,
    stats: &mut LogicalEffortStats,
) -> Frontier {
    stats.candidates += a.len() + b.len();
    let zero_weight = BigInt::zero();
    let hull = merge_hulls(
        if wa.is_some() { &a.hull } else { &[] },
        if wb.is_some() { &b.hull } else { &[] },
        wa.unwrap_or(&zero_weight),
        wb.unwrap_or(&zero_weight),
        stats,
    );
    let za = a.extend_zero(wa.is_none());
    let zb = b.extend_zero(wb.is_none());
    let zero = match (za, zb) {
        (Some(a), Some(b)) => Some(if a.n >= b.n { a } else { b }),
        (a, b) => a.or(b),
    };
    // A finite prefix at least as long also dominates zero effort after every
    // nonnegative-load suffix, including one that makes both efforts zero.
    let zero = zero.filter(|z| hull.last().is_none_or(|last| z.n > last.n));
    Frontier { hull, zero }
}

/// Whether the middle point lies on/below its neighbors' chord, exactly.
fn below_chord(a: &State, b: &State, c: &State) -> bool {
    debug_assert!(a.n < b.n && b.n < c.n);
    // Exact dyadic accumulation is needed as well as an exact determinant:
    // rounding later suffix additions could otherwise reverse collinearity.
    (&c.log_f - &a.log_f) * BigInt::from(b.n - a.n)
        >= (&b.log_f - &a.log_f) * BigInt::from(c.n - a.n)
}

/// Merges sorted predecessor hulls after their edge translations, in O(A+B).
fn merge_hulls(
    a: &[State],
    b: &[State],
    wa: &BigInt,
    wb: &BigInt,
    stats: &mut LogicalEffortStats,
) -> Vec<State> {
    let (mut i, mut j) = (0, 0);
    let mut hull: Vec<State> = Vec::with_capacity(a.len() + b.len());
    while i < a.len() || j < b.len() {
        let candidate = if j == b.len() || (i < a.len() && a[i].n < b[j].n) {
            let state = &a[i];
            i += 1;
            State {
                n: state.n + 1,
                log_f: &state.log_f + wa,
                prev: state.prev,
            }
        } else if i == a.len() || b[j].n < a[i].n {
            let state = &b[j];
            j += 1;
            State {
                n: state.n + 1,
                log_f: &state.log_f + wb,
                prev: state.prev,
            }
        } else {
            stats.dominance_comparisons += 1;
            let la = &a[i].log_f + wa;
            let lb = &b[j].log_f + wb;
            // Prefer the first operand on ties; no hash iteration affects
            // paths.
            let candidate = if la >= lb {
                State {
                    n: a[i].n + 1,
                    log_f: la,
                    prev: a[i].prev,
                }
            } else {
                State {
                    n: b[j].n + 1,
                    log_f: lb,
                    prev: b[j].prev,
                }
            };
            i += 1;
            j += 1;
            candidate
        };
        while hull.len() >= 2 {
            stats.dominance_comparisons += 1;
            if !below_chord(&hull[hull.len() - 2], &hull[hull.len() - 1], &candidate) {
                break;
            }
            hull.pop();
        }
        hull.push(candidate);
    }
    hull
}

/// Immutable traceback records outlive the numeric frontiers that created them.
struct Trace {
    node: AigRef,
    prev: Option<NonZeroUsize>,
}

/// Work and storage counters for profiling graph logical effort hulls.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LogicalEffortStats {
    /// Prefix extensions considered before equal-length and hull pruning.
    pub candidates: usize,
    /// Equal-length comparisons and exact chord predicates.
    pub dominance_comparisons: usize,
    /// Peak retained numeric states, including a merged hull before releasing
    /// its predecessors. Excludes allocator capacity and predicate temporaries.
    pub peak_frontier_slots: usize,
    /// Compact traceback records retained until the analysis finishes.
    pub traceback_records: usize,
    /// Total retained finite/zero-effort states across nodes, including
    /// sources.
    pub total_frontier_states: usize,
    /// Largest retained frontier (finite hull plus zero effort) at any node.
    pub max_frontier_len: usize,
}

/// A maximum score, its optional witness, and the work needed to find it.
struct DelayResult {
    path: Vec<AigRef>,
    delay: f64,
    stats: LogicalEffortStats,
}

/// Propagates hulls in topological order, releasing each after its last use.
fn worst_case_delay(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
    with_traceback: bool,
) -> DelayResult {
    let nodes = &gate_fn.gates;
    let mut fanouts = vec![0usize; nodes.len()];
    for node in nodes {
        for arg in node.args() {
            fanouts[arg.id] += 1;
        }
    }
    let weights = LogWeights::new(&fanouts, options);
    let mut remaining = fanouts.clone();
    let mut frontiers = vec![Frontier::default(); nodes.len()];
    let mut traces = Vec::<Trace>::new();
    let mut stats = LogicalEffortStats::default();
    let mut live = 0usize;
    let mut best_delay = -1.0_f64;
    let mut best_trace = None;
    for u in topo_sort_refs(nodes) {
        let mut current = match &nodes[u.id] {
            AigNode::And2 { a, b, .. } => merge_frontiers(
                &frontiers[a.node.id],
                &frontiers[b.node.id],
                weights.weights[a.node.id].as_ref(),
                weights.weights[b.node.id].as_ref(),
                &mut stats,
            ),
            _ => Frontier {
                hull: vec![State {
                    n: 0,
                    log_f: BigInt::zero(),
                    prev: None,
                }],
                zero: None,
            },
        };
        stats.peak_frontier_slots = stats.peak_frontier_slots.max(live + current.len());
        for arg in nodes[u.id].args() {
            remaining[arg.id] -= 1;
            if remaining[arg.id] == 0 {
                live -= frontiers[arg.id].len();
                frontiers[arg.id] = Frontier::default();
            }
        }
        stats.total_frontier_states += current.len();
        stats.max_frontier_len = stats.max_frontier_len.max(current.len());
        if with_traceback {
            for state in &mut current.hull {
                traces.push(Trace {
                    node: u,
                    prev: state.prev,
                });
                state.prev = NonZeroUsize::new(traces.len());
            }
            if let Some(state) = &mut current.zero {
                traces.push(Trace {
                    node: u,
                    prev: state.prev,
                });
                state.prev = NonZeroUsize::new(traces.len());
            }
        }
        if fanouts[u.id] == 0 {
            for state in &current.hull {
                if state.n == 0 {
                    continue;
                }
                let n = state.n as f64;
                let log_f = scaled_log_to_f64(&state.log_f, weights.exponent);
                let delay = n * (log_f / n).exp() + 2.0 * n;
                if delay > best_delay {
                    best_delay = delay;
                    best_trace = state.prev;
                }
            }
            if let Some(state) = current.zero {
                let delay = 2.0 * state.n as f64;
                if delay > best_delay {
                    best_delay = delay;
                    best_trace = state.prev;
                }
            }
        } else {
            live += current.len();
            frontiers[u.id] = current;
        }
    }
    stats.traceback_records = traces.len();
    let mut path = Vec::new();
    while let Some(index) = best_trace {
        let trace = &traces[index.get() - 1];
        path.push(trace.node);
        best_trace = trace.prev;
    }
    path.reverse();
    DelayResult {
        path,
        delay: best_delay,
        stats,
    }
}

/// Pre-compute an `h` function (effort) using fan-out and a quadratic model:
/// effort(u) = β₁ · f + β₂ · f², where f = fan-out of `u`.
///
/// * `beta1` defaults to 1.0
/// * `beta2` defaults to 0.0
///
/// Assumes all sinks have Cin = 1.0.
/// Returns a closure for querying the model independently of the scorer.
pub fn eff_with_branch(
    dag: &HashMap<AigRef, Vec<(AigRef, f64, f64)>>,
    beta1: f64,
    beta2: f64,
) -> impl Fn(AigRef, AigRef) -> f64 + '_ {
    // 1. pre-compute fan-out for every node
    let mut tot_load: HashMap<AigRef, usize> = HashMap::new();
    for (&u, edges) in dag.iter() {
        tot_load.insert(u, edges.len());
    }

    // 2. capture β₁, β₂ and the table by value
    move |u, _v| {
        let f = *tot_load.get(&u).unwrap_or(&0) as f64;
        beta1 * f + beta2 * f.powi(2)
    }
}

/// Result of logical effort analysis for a GateFn.
pub struct LogicalEffortAnalysis {
    pub dag: HashMap<AigRef, Vec<(AigRef, f64, f64)>>,
    pub path: Vec<AigRef>,
    pub delay: f64,
    pub stats: LogicalEffortStats,
}

/// Load coefficients; every used pin load must be finite and nonnegative.
pub struct GraphLogicalEffortOptions {
    pub beta1: f64,
    pub beta2: f64,
}

/// Worst-case score and profiling counters, without path storage.
pub struct LogicalEffortScore {
    pub delay: f64,
    pub stats: LogicalEffortStats,
}

/// Computes a maximum and a concrete source-to-sink path using NAND2
/// parameters. No-AND graphs return -1 and an empty path. Ties are
/// deterministic, but the selected maximizing path may differ from other
/// algorithms' witnesses.
///
/// Log weights use binary64 evaluation; their sums and hull pruning are exact.
/// Final delay evaluation uses binary64. Summing exactly can change the last
/// bits compared with repeatedly rounded prefix sums. Traceback can still take
/// O(nodes * depth) space on arbitrary graphs; use the score API if unneeded.
///
/// # Panics
/// Panics if a used pin load is not finite and nonnegative.
pub fn analyze_graph_logical_effort(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
) -> LogicalEffortAnalysis {
    let result = worst_case_delay(gate_fn, options, /* with_traceback= */ true);
    let mut dag: HashMap<AigRef, Vec<(AigRef, f64, f64)>> = HashMap::new();
    for (id, node) in gate_fn.gates.iter().enumerate() {
        for arg in node.args() {
            dag.entry(arg)
                .or_default()
                .push((AigRef { id }, 4.0 / 3.0, 2.0));
        }
    }
    LogicalEffortAnalysis {
        dag,
        path: result.path,
        delay: result.delay,
        stats: result.stats,
    }
}

/// Computes the same score as analysis while omitting traceback and DAG output.
/// Numeric hulls are released after their last use; arbitrary graphs can still
/// retain many lengths. No state cap or approximate pruning is used.
///
/// # Panics
/// Panics if a used pin load is not finite and nonnegative.
pub fn score_graph_logical_effort(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
) -> LogicalEffortScore {
    let result = worst_case_delay(gate_fn, options, /* with_traceback= */ false);
    LogicalEffortScore {
        delay: result.delay,
        stats: result.stats,
    }
}

#[cfg(test)]
mod tests {
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    use super::*;

    #[test]
    fn dyadic_conversion_round_trips_binary64_extremes() {
        for exponent in 0..=2046u64 {
            for fraction in [0, 1, (1u64 << 51) - 1, (1u64 << 52) - 1] {
                for sign in [0, 1u64 << 63] {
                    let value = f64::from_bits(sign | (exponent << 52) | fraction);
                    let dyadic = Dyadic::from_f64(value);
                    assert_eq!(scaled_log_to_f64(&dyadic.mantissa, dyadic.exponent), value);
                }
            }
        }
    }

    #[test]
    fn dyadic_conversion_rounds_ties_even_without_intermediate_overflow() {
        let base = BigInt::from(1) << 53usize;
        for sign in [1, -1] {
            assert_eq!(
                scaled_log_to_f64(&((&base + 1) * sign), 0),
                sign as f64 * 2.0_f64.powi(53)
            );
            assert_eq!(
                scaled_log_to_f64(&((&base + 3) * sign), 0),
                sign as f64 * (2.0_f64.powi(53) + 4.0)
            );
            let huge = (BigInt::from(1) << 1100usize) * sign;
            assert_eq!(
                scaled_log_to_f64(&huge, -1074),
                sign as f64 * 2.0_f64.powi(26)
            );
        }
        assert_eq!(
            scaled_log_to_f64(&BigInt::from(1), -1074),
            f64::from_bits(1)
        );
        assert_eq!(
            scaled_log_to_f64(&((BigInt::from(1) << 1024usize) - 1), 0),
            f64::INFINITY
        );
        // Rounding carries out of the significand into the exponent.
        assert_eq!(
            scaled_log_to_f64(&((BigInt::from(1) << 54usize) - 1), 0),
            2.0_f64.powi(54)
        );
    }

    #[test]
    fn chord_predicate_keeps_arbitrarily_small_bulges_and_negative_weights() {
        let huge = BigInt::from(1) << 1100usize;
        for sign in [1, -1] {
            let a = State {
                n: 1,
                log_f: &huge * sign,
                prev: None,
            };
            let mut b = State {
                n: 2,
                log_f: &huge * (2 * sign),
                prev: None,
            };
            let c = State {
                n: 3,
                log_f: &huge * (3 * sign),
                prev: None,
            };
            assert!(below_chord(&a, &b, &c));
            b.log_f += 1;
            assert!(!below_chord(&a, &b, &c));
            b.log_f -= 2;
            assert!(below_chord(&a, &b, &c));
        }
    }

    #[test]
    fn exact_accumulation_preserves_cancelled_low_bits() {
        let a = Dyadic::from_f64(2.0_f64.powi(60));
        let b = Dyadic::from_f64(1.0);
        let scale = b.exponent;
        let mut sum = &a.mantissa << (a.exponent - scale) as usize;
        sum += &b.mantissa;
        sum -= &a.mantissa << (a.exponent - scale) as usize;
        assert_eq!(scaled_log_to_f64(&sum, scale), 1.0);
        assert_eq!((2.0_f64.powi(60) + 1.0) - 2.0_f64.powi(60), 0.0);
    }

    #[test]
    fn hull_merge_keeps_best_equal_length_and_deterministic_ties() {
        let states = |values: &[(usize, i64)]| {
            values
                .iter()
                .map(|&(n, a)| State {
                    n,
                    log_f: BigInt::from(a),
                    prev: NonZeroUsize::new(n),
                })
                .collect::<Vec<_>>()
        };
        let a = states(&[(1, -10), (2, 0), (4, 10)]);
        let mut b = states(&[(1, -9), (2, 0), (3, 5)]);
        b[1].prev = NonZeroUsize::new(100);
        let mut stats = LogicalEffortStats::default();
        let hull = merge_hulls(&a, &b, &BigInt::zero(), &BigInt::zero(), &mut stats);
        assert_eq!(hull.iter().map(|s| s.n).collect::<Vec<_>>(), vec![2, 3, 5]);
        assert_eq!(hull[0].log_f, BigInt::from(-9));
        assert_eq!(hull[1].prev, a[1].prev);
    }

    #[test]
    fn test_nand_fanout_case() {
        let mut gb = GateBuilder::new("nand_fanout_case".to_string(), GateBuilderOptions::no_opt());
        // Inputs
        let a0 = gb.add_input("a0".to_string(), 1);
        let a1 = gb.add_input("a1".to_string(), 1);
        let b0 = gb.add_input("b0".to_string(), 1);
        let b1 = gb.add_input("b1".to_string(), 1);
        // First-level NANDs
        let n1 = gb.add_nand_binary(*a0.get_lsb(0), *a1.get_lsb(0));
        let n2 = gb.add_nand_binary(*b0.get_lsb(0), *b1.get_lsb(0));
        // Each first-level NAND drives four more NANDs
        let mut n1_sinks = vec![];
        let mut n2_sinks = vec![];
        for _i in 0..4 {
            let n1_sink = gb.add_nand_binary(n1, gb.get_true());
            n1_sinks.push(n1_sink);
            let n2_sink = gb.add_nand_binary(n2, gb.get_true());
            n2_sinks.push(n2_sink);
        }
        // Outputs (sinks)
        for (_i, &sink) in n1_sinks.iter().enumerate() {
            gb.add_output(format!("n1_{}", _i + 1), sink.into());
        }
        for (_i, &sink) in n2_sinks.iter().enumerate() {
            gb.add_output(format!("n2_{}", _i + 1), sink.into());
        }
        let gate_fn = gb.build();
        let options = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        };
        let analysis = analyze_graph_logical_effort(&gate_fn, &options);
        log::info!("critical path: {:?}", analysis.path);
        let expected = 12.666666666666663;
        let epsilon = 1e-6;
        assert!(
            (analysis.delay - expected).abs() < epsilon,
            "delay was {}",
            analysis.delay
        );
        let options = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 1.0,
        };
        let analysis = analyze_graph_logical_effort(&gate_fn, &options);
        log::info!("critical path: {:?}", analysis.path);
        let expected = 98.0;
        let epsilon = 1e-6;
        assert!(
            (analysis.delay - expected).abs() < epsilon,
            "delay was {}",
            analysis.delay
        );
    }

    #[test]
    fn test_nand_branch_case() {
        let mut gb = GateBuilder::new("nand_branch_case".to_string(), GateBuilderOptions::no_opt());
        // Inputs
        let a0 = gb.add_input("a0".to_string(), 1);
        let a1 = gb.add_input("a1".to_string(), 1);
        let a2 = gb.add_input("a2".to_string(), 1);
        let a3 = gb.add_input("a3".to_string(), 1);
        let b0 = gb.add_input("b0".to_string(), 1);
        let b1 = gb.add_input("b1".to_string(), 1);
        let b2 = gb.add_input("b2".to_string(), 1);
        let b3 = gb.add_input("b3".to_string(), 1);
        // Level-1 NANDs
        let n1 = gb.add_nand_binary(*a0.get_lsb(0), *a1.get_lsb(0));
        let n2 = gb.add_nand_binary(*a2.get_lsb(0), *a3.get_lsb(0));
        let n3 = gb.add_nand_binary(*b0.get_lsb(0), *b1.get_lsb(0));
        let n4 = gb.add_nand_binary(*b2.get_lsb(0), *b3.get_lsb(0));
        // Level-2 NANDs
        let n5 = gb.add_nand_binary(n1, n2);
        let n6 = gb.add_nand_binary(n3, n4);
        // Root NAND
        let n7 = gb.add_nand_binary(n5, n6);
        // Side buffer/inverter
        let inv1 = gb.add_not(n2);
        let buf1 = gb.add_nand_binary(n4, gb.get_true());
        // Sinks
        let m1 = gb.add_nand_binary(n5, gb.get_true());
        let m2 = gb.add_nand_binary(n5, gb.get_true());
        let m3 = gb.add_nand_binary(n6, gb.get_true());
        let v1 = gb.add_nand_binary(inv1, gb.get_true());
        let v2 = gb.add_nand_binary(buf1, gb.get_true());
        let y1 = gb.add_nand_binary(n7, gb.get_true());
        let y2 = gb.add_nand_binary(n7, gb.get_true());
        let y3 = gb.add_nand_binary(n7, gb.get_true());
        let y4 = gb.add_nand_binary(n7, gb.get_true());
        // Outputs (sinks)
        for (_i, &sink) in [m1, m2, m3, v1, v2, y1, y2, y3, y4].iter().enumerate() {
            gb.add_output(format!("sink_{}", _i + 1), sink.into());
        }
        let gate_fn = gb.build();
        let options = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        };
        let analysis = analyze_graph_logical_effort(&gate_fn, &options);
        log::info!("critical path: {:?}", analysis.path);
        let expected = 19.804607143470097;
        let epsilon = 1e-6;
        assert!(
            (analysis.delay - expected).abs() < epsilon,
            "delay was {}",
            analysis.delay
        );
    }
}
