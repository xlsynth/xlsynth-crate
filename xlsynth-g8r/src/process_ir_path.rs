// SPDX-License-Identifier: Apache-2.0

use std::cmp::min;
use std::collections::BTreeMap;
use std::collections::HashMap;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::aig::fanout::fanout_histogram;
use crate::aig::find_structures;
use crate::aig::fraig;
use crate::aig::fraig::{DidConverge, FraigIterationStat, IterationBounds};
use crate::aig::gate;
use crate::aig::get_summary_stats::get_gate_depth;
use crate::aig::graph_logical_effort::{self, analyze_graph_logical_effort};
use crate::aig::logical_effort;
use crate::aig::logical_effort::compute_logical_effort_min_delay;
use crate::aig_serdes::emit_netlist;
use crate::aig_serdes::ir2gate;
use crate::aig_sim::count_toggles;
use crate::check_equivalence;
use crate::use_count::get_id_to_use_count;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils;

#[derive(Debug, serde::Serialize)]
pub struct Ir2GatesSummaryStats {
    pub live_nodes: usize,
    pub deepest_path: usize,
    pub fanout_histogram: BTreeMap<usize, usize>,
    pub toggle_stats: Option<count_toggles::ToggleStats>,
    pub toggle_transitions: Option<usize>,
    pub logical_effort_deepest_path_min_delay: f64,
    pub graph_logical_effort_worst_case_delay: Option<f64>,
    pub fraig_did_converge: Option<DidConverge>,
    pub fraig_iteration_stats: Option<Vec<FraigIterationStat>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub independent_op_stats: Option<IndependentOpStats>,
}

#[derive(Debug, serde::Serialize)]
pub struct IndependentOpEntry {
    pub node_index: usize,
    pub text_id: usize,
    pub op: String,
    pub live_nodes: usize,
    pub deepest_path: usize,
}

#[derive(Debug, serde::Serialize)]
pub struct IndependentOpStats {
    pub independent_sum_live_nodes: usize,
    pub independent_sum_deepest_path: usize,
    pub independent_max_live_nodes: usize,
    pub independent_max_deepest_path: usize,
    pub independent_critical_path_depth: usize,
    pub independent_included_node_count: usize,
    pub per_node: Vec<IndependentOpEntry>,
}

fn independent_live_nodes_excluding_params(g: &gate::GateFn) -> usize {
    let id_to_use_count = get_id_to_use_count(g);
    id_to_use_count
        .keys()
        .filter(|r| !matches!(g.gates[r.id], gate::AigNode::Input { .. }))
        .count()
}

fn independent_deepest_path_excluding_params(g: &gate::GateFn) -> usize {
    // We want a depth metric that matches the AIG mental model: NOT is free and
    // inputs are not logic levels. We therefore count only AND nodes along the
    // deepest path.
    let id_to_use_count = get_id_to_use_count(g);
    let live_nodes: Vec<gate::AigRef> = id_to_use_count.keys().cloned().collect();
    let depth_stats = get_gate_depth(g, &live_nodes);
    depth_stats
        .deepest_path
        .iter()
        .filter(|r| matches!(g.gates[r.id], gate::AigNode::And2 { .. }))
        .count()
}

fn should_skip_independent_op_stats(payload: &ir::NodePayload) -> bool {
    match payload {
        ir::NodePayload::GetParam(_) => true,
        ir::NodePayload::Literal(_) => true,
        ir::NodePayload::Nil => true,
        ir::NodePayload::Unop(ir::Unop::Not, _) => true,
        ir::NodePayload::Unop(ir::Unop::Identity, _) => true,
        ir::NodePayload::Tuple(_) => true,
        ir::NodePayload::TupleIndex { .. } => true,
        ir::NodePayload::Array(_) => true,
        ir::NodePayload::Nary(ir::NaryOp::Concat, _) => true,
        ir::NodePayload::BitSlice { .. } => true,
        ir::NodePayload::ZeroExt { .. } => true,
        ir::NodePayload::SignExt { .. } => true,
        _ => false,
    }
}

pub struct Options {
    pub check_equivalence: bool,
    pub fold: bool,
    pub hash: bool,
    pub adder_mapping: crate::ir2gate_utils::AdderMapping,
    pub fraig: bool,
    pub emit_independent_op_stats: bool,

    /// If not set, we fraig to convergence.
    pub fraig_max_iterations: Option<usize>,

    /// If not set, we scale the gate count down by some policy (e.g. divide
    /// down by 128 and then round to the nearest 256) and use that many
    /// samples.
    pub fraig_sim_samples: Option<usize>,

    pub quiet: bool,
    pub emit_netlist: bool,
    /// If > 0, generate this many random input samples and print toggle stats.
    pub toggle_sample_count: usize,
    /// Seed for random toggle stimulus (default 0).
    pub toggle_sample_seed: u64,
    /// If true, compute the graph logical effort worst case delay.
    pub compute_graph_logical_effort: bool,
    pub graph_logical_effort_beta1: f64,
    pub graph_logical_effort_beta2: f64,
}

/// Command line entry point (e.g. it exits the process on error).
pub fn process_ir_path_for_cli(
    ir_path: &std::path::Path,
    options: &Options,
) -> Ir2GatesSummaryStats {
    // Read the file into a string.
    let file_content = std::fs::read_to_string(&ir_path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {}", ir_path.display(), err));
    let mut parser = ir_parser::Parser::new(&file_content);
    let ir_package = parser.parse_and_validate_package().unwrap_or_else(|err| {
        eprintln!("Error encountered parsing XLS IR package: {:?}", err);
        std::process::exit(1);
    });

    let ir_top = match ir_package.get_top_fn() {
        Some(ir_top) => ir_top,
        None => {
            eprintln!("No top module found in the IR package");
            std::process::exit(1);
        }
    };

    log::info!("IR top:\n{}", ir_top.to_string());

    if !options.quiet {
        print_op_freqs(&ir_top);
    }

    // Gatify the IR function
    let gatify_output = ir2gate::gatify(
        &ir_top,
        ir2gate::GatifyOptions {
            fold: options.fold,
            hash: options.hash,
            adder_mapping: options.adder_mapping,
            check_equivalence: false, // Check is done below if requested
        },
    )
    .unwrap();
    let mut gate_fn = gatify_output.gate_fn;

    let independent_op_stats = if options.emit_independent_op_stats {
        let gatify_options = ir2gate::GatifyOptions {
            fold: options.fold,
            hash: options.hash,
            adder_mapping: options.adder_mapping,
            check_equivalence: false,
        };
        let mut per_node: Vec<IndependentOpEntry> = Vec::new();
        let mut cost_by_node_index: Vec<usize> = vec![0; ir_top.nodes.len()];
        let mut independent_sum_live_nodes: usize = 0;
        let mut independent_sum_deepest_path: usize = 0;
        let mut independent_max_live_nodes: usize = 0;
        let mut independent_max_deepest_path: usize = 0;
        let mut independent_included_node_count: usize = 0;

        for (node_index, node) in ir_top.nodes.iter().enumerate() {
            if should_skip_independent_op_stats(&node.payload) {
                continue;
            }
            let node_ref = ir::NodeRef { index: node_index };
            let node_gate_fn = ir2gate::gatify_node_as_fn(&ir_top, node_ref, &gatify_options)
                .unwrap_or_else(|e| {
                    eprintln!(
                        "Failed to gatify node {} (text_id={}): {}",
                        node_index, node.text_id, e
                    );
                    std::process::exit(1);
                });
            let live_nodes = independent_live_nodes_excluding_params(&node_gate_fn);
            let deepest_path = independent_deepest_path_excluding_params(&node_gate_fn);
            cost_by_node_index[node_index] = deepest_path;
            independent_sum_live_nodes += live_nodes;
            independent_sum_deepest_path += deepest_path;
            independent_max_live_nodes = std::cmp::max(independent_max_live_nodes, live_nodes);
            independent_max_deepest_path =
                std::cmp::max(independent_max_deepest_path, deepest_path);
            independent_included_node_count += 1;

            per_node.push(IndependentOpEntry {
                node_index,
                text_id: node.text_id,
                op: node.to_signature_string(&ir_top),
                live_nodes,
                deepest_path,
            });
        }

        // Sort most expensive first for readability.
        // Deterministic tie-breaks are important for stable output.
        per_node.sort_by(|a, b| {
            b.live_nodes
                .cmp(&a.live_nodes)
                .then_with(|| b.deepest_path.cmp(&a.deepest_path))
                .then_with(|| a.node_index.cmp(&b.node_index))
        });

        let independent_critical_path_depth = match ir_top.ret_node_ref {
            Some(ret_node_ref) => {
                let mut dp: Vec<usize> = vec![0; ir_top.nodes.len()];
                for nr in ir_utils::get_topological(&ir_top) {
                    let node = ir_top.get_node(nr);
                    let deps = ir_utils::operands(&node.payload);
                    let max_dep = deps.into_iter().map(|d| dp[d.index]).max().unwrap_or(0);
                    dp[nr.index] = cost_by_node_index[nr.index] + max_dep;
                }
                dp[ret_node_ref.index]
            }
            None => 0,
        };

        Some(IndependentOpStats {
            independent_sum_live_nodes,
            independent_sum_deepest_path,
            independent_max_live_nodes,
            independent_max_deepest_path,
            independent_critical_path_depth,
            independent_included_node_count,
            per_node,
        })
    } else {
        None
    };

    // Map each gate reference back to the IR node positions, if available.
    let mut gate_to_sources: HashMap<usize, Vec<String>> = HashMap::new();
    for (node_ref, bit_vec) in gatify_output.lowering_map.iter() {
        if let Some(pos_data) = ir_top.get_node(*node_ref).pos.as_ref() {
            let sources: Vec<String> = pos_data
                .iter()
                .filter_map(|p| p.to_human_string(&ir_package.file_table))
                .collect();
            for operand in bit_vec.iter_lsb_to_msb() {
                gate_to_sources
                    .entry(operand.node.id)
                    .or_default()
                    .extend(sources.clone());
            }
        }
    }
    // Prepare to capture fraig statistics if enabled
    let mut fraig_did_converge: Option<DidConverge> = None;
    let mut fraig_iteration_stats: Option<Vec<FraigIterationStat>> = None;
    if options.fraig {
        log::info!(
            "fraig is enabled for GateFn {:?} with {} nodes",
            gate_fn.name,
            gate_fn.gates.len()
        );
        let iteration_bounds = if let Some(max_iterations) = options.fraig_max_iterations {
            IterationBounds::MaxIterations(max_iterations)
        } else {
            IterationBounds::ToConvergence
        };
        let sim_samples = match options.fraig_sim_samples {
            Some(n) => n,
            None => {
                let gate_count = gate_fn.gates.len();
                let scaled = (gate_count as f64 / 8.0).ceil() as usize;
                // Round the scaled value to the nearest 256, it must be more than zero.
                let result = round_up_to_nearest_multiple(scaled, 256);
                log::info!(
                    "fraig sim samples; gate count: {}, scaled: {}, result: {}",
                    gate_count,
                    scaled,
                    result
                );
                result
            }
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let fraig_result: Result<_, _> =
            fraig::fraig_optimize(&gate_fn, sim_samples, iteration_bounds, &mut rng);
        if !fraig_result.is_ok() {
            eprintln!("Fraig optimization failed");
            std::process::exit(1);
        }
        let (optimized_fn, did_converge, iteration_stats) = fraig_result.unwrap();
        if !options.quiet {
            println!("== Fraig convergence: {:?}", did_converge);
        }
        gate_fn = optimized_fn;
        // Capture fraig results for JSON
        fraig_did_converge = Some(did_converge);
        fraig_iteration_stats = Some(iteration_stats);
    }

    log::info!("gate fn signature: {}", gate_fn.get_signature());

    // Pass the extracted gate_fn reference to subsequent functions
    if options.check_equivalence {
        println!("== Checking equivalence...");
        let start = std::time::Instant::now();
        let eq = check_equivalence::validate_same_fn(&ir_top, &gate_fn);
        let duration = start.elapsed();
        println!("Equivalence check took {:?}.", duration);
        println!("Equivalence: {:?}", eq);
    }

    if options.emit_netlist {
        let netlist =
            match emit_netlist::emit_netlist(&ir_top.name, &gate_fn, false, false, false, None) {
                Ok(netlist) => netlist,
                Err(e) => {
                    eprintln!("Failed to emit netlist: {}", e);
                    std::process::exit(1);
                }
            };
        println!("{}", netlist);
    }

    let id_to_use_count: HashMap<gate::AigRef, usize> = get_id_to_use_count(&gate_fn);
    let live_nodes: Vec<gate::AigRef> = id_to_use_count.keys().cloned().collect();

    let depth_stats = get_gate_depth(&gate_fn, &live_nodes);

    // Compute critical path delay
    let logical_effort_deepest_path_min_delay =
        compute_logical_effort_min_delay(&gate_fn, &logical_effort::Options::default());

    // Compute fanout histogram and include in summary stats
    let hist = fanout_histogram(&gate_fn);
    let hist_sorted: BTreeMap<usize, usize> = hist.clone().into_iter().collect();

    // Compute toggle stats if requested
    let (toggle_stats, toggle_transitions) = if options.toggle_sample_count > 0 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(options.toggle_sample_seed);
        let mut batch_inputs = Vec::with_capacity(options.toggle_sample_count);
        let input_widths: Vec<usize> = gate_fn
            .inputs
            .iter()
            .map(|input| input.bit_vector.get_bit_count())
            .collect();
        for _ in 0..options.toggle_sample_count {
            let mut input_vec = Vec::with_capacity(input_widths.len());
            for &width in &input_widths {
                let bits = xlsynth_pir::fuzz_utils::arbitrary_irbits(&mut rng, width);
                input_vec.push(bits);
            }
            batch_inputs.push(input_vec);
        }
        let stats = count_toggles::count_toggles(&gate_fn, &batch_inputs);
        (
            Some(stats),
            Some(options.toggle_sample_count.saturating_sub(1)),
        )
    } else {
        (None, None)
    };

    let graph_logical_effort_worst_case_delay = if options.compute_graph_logical_effort {
        let graph_logical_effort_analysis = analyze_graph_logical_effort(
            &gate_fn,
            &graph_logical_effort::GraphLogicalEffortOptions {
                beta1: options.graph_logical_effort_beta1,
                beta2: options.graph_logical_effort_beta2,
            },
        );
        Some(graph_logical_effort_analysis.delay)
    } else {
        None
    };

    let summary_stats = Ir2GatesSummaryStats {
        live_nodes: live_nodes.len(),
        deepest_path: depth_stats.deepest_path.len(),
        fanout_histogram: hist_sorted,
        toggle_stats,
        toggle_transitions,
        logical_effort_deepest_path_min_delay,
        graph_logical_effort_worst_case_delay,
        fraig_did_converge,
        fraig_iteration_stats,
        independent_op_stats,
    };

    if options.quiet {
        return summary_stats;
    }

    println!("== Deepest path ({}):", depth_stats.deepest_path.len());
    for gate_ref in depth_stats.deepest_path.iter() {
        let gate: &gate::AigNode = &gate_fn.gates[gate_ref.id];
        println!("  {:4} :: {}", gate_ref.id, format_aig_node(gate));
        if let Some(tags) = match gate {
            gate::AigNode::And2 { tags, .. } => tags.as_ref(),
            _ => None,
        } {
            if !tags.is_empty() {
                println!("          tags: {}", tags.join(", "));
            }
        }
        if let Some(srcs) = gate_to_sources.get(&gate_ref.id) {
            if !srcs.is_empty() {
                let mut unique = srcs.clone();
                unique.sort();
                unique.dedup();
                println!("          source: {}", unique.join(" "));
            }
        }
        println!(
            "          uses: {}",
            id_to_use_count.get(&gate_ref).unwrap_or(&0)
        );
    }

    // Print the critical path delay
    println!(
        "== Logical effort deepest path min delay: {:.4} (FO4 units)",
        logical_effort_deepest_path_min_delay
    );

    // ANSI histogram printing with buckets of width 5:
    let bucket_width = 5;
    let mut bucketed_counts: HashMap<usize, usize> = HashMap::new();

    // Group the depths into buckets.
    for (depth, count) in depth_stats.depth_to_count {
        let bucket = (depth / bucket_width) * bucket_width;
        *bucketed_counts.entry(bucket).or_insert(0) += count;
    }

    let max_bar_width = 50;
    let max_bucket_count = bucketed_counts.values().max().unwrap_or(&1);

    println!("Gate Depth Histogram (Bucketed by {}):", bucket_width);
    let mut sorted_buckets: Vec<_> = bucketed_counts.iter().collect();
    sorted_buckets.sort_by_key(|(bucket, _)| *bucket);

    for (bucket, bucket_count) in sorted_buckets {
        // Compute bar length scaled to max_bar_width:
        let bar_length = ((*bucket_count as f64) / (*max_bucket_count as f64)
            * max_bar_width as f64)
            .round() as usize;
        let bar = "█".repeat(bar_length);
        // Display the bucket as a range [bucket, bucket+bucket_width-1]
        println!(
            "Depth {:3} - {:3} | {:<50} | {:4}",
            bucket,
            bucket + bucket_width - 1,
            bar,
            bucket_count,
        );
    }

    println!("== Live node count: {}", live_nodes.len());

    if let Some(ind_stats) = summary_stats.independent_op_stats.as_ref() {
        println!("== Independent-op aggregates (direct operands treated as inputs):");
        println!(
            "  independent_sum_live_nodes: {}",
            ind_stats.independent_sum_live_nodes
        );
        println!(
            "  independent_sum_deepest_path: {}",
            ind_stats.independent_sum_deepest_path
        );
        println!(
            "  independent_max_live_nodes: {}",
            ind_stats.independent_max_live_nodes
        );
        println!(
            "  independent_max_deepest_path: {}",
            ind_stats.independent_max_deepest_path
        );
        println!(
            "  independent_critical_path_depth: {}",
            ind_stats.independent_critical_path_depth
        );
        println!(
            "  independent_included_node_count: {}",
            ind_stats.independent_included_node_count
        );
        println!("== Independent-op per-node stats:");
        for entry in ind_stats.per_node.iter() {
            println!(
                "  node {} (text_id={}): live_nodes={} deepest_path={} op={}",
                entry.node_index, entry.text_id, entry.live_nodes, entry.deepest_path, entry.op
            );
        }
    }

    let structure_to_count = find_structures::find_structures(&gate_fn);
    let mut sorted_structures = structure_to_count
        .iter()
        .map(|(s, c)| (s.clone(), *c))
        .collect::<Vec<_>>();
    sorted_structures.sort_by(|(s1, c1), (s2, c2)| c2.cmp(c1).then_with(|| s1.cmp(s2)));
    println!("== Structures:");
    for (structure, count) in sorted_structures[0..min(20, sorted_structures.len())].iter() {
        println!("{:3} :: {}", count, structure);
    }

    // Print fanout histogram
    println!("== Fanout histogram:");
    let mut sorted_hist: Vec<_> = hist.iter().collect();
    sorted_hist.sort_by_key(|(fanout, _)| *fanout);
    for (fanout, count) in sorted_hist {
        println!("  {}: {}", fanout, count);
    }

    // Print toggle stats if present
    if let Some(stats) = toggle_stats {
        println!("  Gate output toggles: {}", stats.gate_output_toggles);
        println!("  Gate input toggles: {}", stats.gate_input_toggles);
        println!("  Primary input toggles: {}", stats.primary_input_toggles);
        println!("  Primary output toggles: {}", stats.primary_output_toggles);
    }

    summary_stats
}

fn round_up_to_nearest_multiple<T>(x: T, y: T) -> T
where
    T: std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::One
        + Copy,
{
    ((x + y - T::one()) / y) * y
}

fn format_aig_operand(op: &gate::AigOperand) -> String {
    if op.negated {
        format!("~Ref({})", op.node.id)
    } else {
        format!("Ref({})", op.node.id)
    }
}

fn format_aig_node(node: &gate::AigNode) -> String {
    match node {
        gate::AigNode::Input { name, lsb_index } => {
            format!("Input({}[{}])", name, lsb_index)
        }
        gate::AigNode::Literal(v) => format!("Literal({})", v),
        gate::AigNode::And2 { a, b, .. } => {
            format!("And2({}, {})", format_aig_operand(a), format_aig_operand(b))
        }
    }
}

// Add back print_op_freqs if missing
fn collect_op_frequencies(f: &ir::Fn) -> HashMap<String, usize> {
    let mut op_freqs = HashMap::new();
    for node in f.nodes.iter() {
        let op = node.to_signature_string(f);
        *op_freqs.entry(op).or_insert(0) += 1;
    }
    op_freqs
}

fn print_op_freqs(ir_top: &ir::Fn) {
    let op_freqs = collect_op_frequencies(&ir_top);
    println!("== Op frequencies:");
    let mut op_freqs_vec: Vec<_> = op_freqs.iter().collect();
    op_freqs_vec.sort_by(|(op_a, count_a), (op_b, count_b)| {
        count_b.cmp(count_a).then_with(|| op_a.cmp(op_b))
    });
    for (op, freq) in op_freqs_vec.iter() {
        println!("  {:4} :: {}", freq, op);
    }
}
