// SPDX-License-Identifier: Apache-2.0

use std::cmp::min;
use std::collections::HashMap;

use crate::check_equivalence;
use crate::emit_netlist;
use crate::find_structures;
use crate::gate;
use crate::get_summary_stats::get_gate_depth;
use crate::get_summary_stats::SummaryStats;
use crate::ir2gate;
use crate::use_count::get_id_to_use_count;
use crate::xls_ir::ir;
use crate::xls_ir::ir_parser;

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
    println!("Op frequencies:");
    let mut op_freqs_vec: Vec<_> = op_freqs.iter().collect();
    op_freqs_vec.sort_by(|(op_a, count_a), (op_b, count_b)| {
        count_b.cmp(count_a).then_with(|| op_a.cmp(op_b))
    });
    for (op, freq) in op_freqs_vec.iter() {
        println!("  {:4} :: {}", freq, op);
    }
}

pub struct Options {
    pub check_equivalence: bool,
    pub fold: bool,
    pub quiet: bool,
    pub emit_netlist: bool,
}

/// Command line entry point (e.g. it exits the process on error).
pub fn process_ir_path(ir_path: &std::path::Path, options: &Options) -> SummaryStats {
    // Read the file into a string.
    let file_content = std::fs::read_to_string(&ir_path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {}", ir_path.display(), err));
    let mut parser = ir_parser::Parser::new(&file_content);
    let ir_package = parser.parse_package().unwrap_or_else(|err| {
        eprintln!("Error encountered parsing XLS IR package: {:?}", err);
        std::process::exit(1);
    });

    let ir_top = match ir_package.get_top() {
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
            check_equivalence: false, // Check is done below if requested
        },
    )
    .unwrap();
    let gate_fn = &gatify_output.gate_fn; // Extract GateFn reference

    log::info!("gate fn signature: {}", gate_fn.get_signature());

    // Pass the extracted gate_fn reference to subsequent functions
    if options.check_equivalence {
        println!("== Checking equivalence...");
        let start = std::time::Instant::now();
        let eq = check_equivalence::validate_same_fn(&ir_top, gate_fn);
        let duration = start.elapsed();
        println!("Equivalence check took {:?}.", duration);
        println!("Equivalence: {:?}", eq);
    }

    if options.emit_netlist {
        let netlist = emit_netlist::emit_netlist(&ir_top.name, gate_fn);
        println!("{}", netlist);
    }

    let id_to_use_count: HashMap<gate::AigRef, usize> = get_id_to_use_count(gate_fn);
    let live_nodes: Vec<gate::AigRef> = id_to_use_count.keys().cloned().collect();

    let (depth_to_count, deepest_path) = get_gate_depth(gate_fn, &live_nodes);

    let summary_stats = SummaryStats {
        live_nodes: live_nodes.len(),
        deepest_path: deepest_path.len(),
    };

    if options.quiet {
        return summary_stats;
    }

    println!("== Deepest path ({}):", deepest_path.len());
    for gate_ref in deepest_path.iter() {
        // Access gates via the gate_fn reference
        let gate: &gate::AigNode = &gate_fn.gates[gate_ref.id];
        println!(
            "  {:4} :: {:?} :: uses: {}",
            gate_ref.id,
            gate,
            id_to_use_count.get(&gate_ref).unwrap_or(&0)
        );
    }

    // ANSI histogram printing with buckets of width 5:
    let bucket_width = 5;
    let mut bucketed_counts: HashMap<usize, usize> = HashMap::new();

    // Group the depths into buckets.
    for (depth, count) in depth_to_count {
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

    let structure_to_count = find_structures::find_structures(gate_fn);
    let mut sorted_structures = structure_to_count
        .iter()
        .map(|(s, c)| (s.clone(), *c))
        .collect::<Vec<_>>();
    sorted_structures.sort_by(|(s1, c1), (s2, c2)| c2.cmp(c1).then_with(|| s1.cmp(s2)));
    println!("== Structures:");
    for (structure, count) in sorted_structures[0..min(20, sorted_structures.len())].iter() {
        println!("{:3} :: {}", count, structure);
    }

    summary_stats
}
