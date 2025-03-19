// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use g8r::ir;
use std::cmp::min;
use std::collections::HashMap;

use g8r::find_structures;
use g8r::gate;
use g8r::ir2gate;
use g8r::ir_parser;
use g8r::use_count::get_id_to_use_count;

/// Simple program to parse an XLS IR file and emit a Verilog netlist.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The name of the generated Verilog module.
    #[arg(long, default_value = "default_module")]
    module_name: String,

    /// Whether to perform AIG folding optimizations
    #[arg(long, default_value = "true")]
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    fold: bool,

    /// Whether to check equivalence between the IR and the gate function.
    #[arg(long, default_value = "true")]
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    check_equivalence: bool,

    /// The path to the XLS IR file.
    input: String,
}

/// Returns a mapping that shows {depth: count} where the count is in number of
/// gates.
fn get_gate_depth(
    gate_fn: &gate::GateFn,
    live_nodes: &[gate::AigRef],
) -> (HashMap<usize, usize>, Vec<gate::AigRef>) {
    let mut depths: HashMap<gate::AigRef, usize> = HashMap::new();
    for input in gate_fn.inputs.iter() {
        for operand in input.bit_vector.iter_lsb_to_msb() {
            depths.insert(operand.node, 0);
        }
    }
    for (gate_id, gate) in gate_fn.gates.iter().enumerate() {
        let gate_ref = gate::AigRef { id: gate_id };
        match gate {
            &gate::AigNode::Input { .. } => {
                continue;
            }
            &gate::AigNode::Literal(_) => {
                assert!(gate_ref.id < 2);
                depths.insert(gate_ref, 0);
            }
            &gate::AigNode::And2 { a, b, .. } => {
                depths.insert(
                    gate_ref,
                    1 + std::cmp::max(depths.get(&a.node).unwrap(), depths.get(&b.node).unwrap()),
                );
            }
        }
    }

    // Filter to just the nodes that are outputs to determine the deepest primary
    // output.
    let mut deepest_primary_output: Option<(gate::AigRef, usize)> = None;
    for output in gate_fn.outputs.iter() {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            if let Some(depth) = depths.get(&operand.node) {
                if deepest_primary_output.is_none()
                    || *depth > deepest_primary_output.as_ref().unwrap().1
                {
                    deepest_primary_output = Some((operand.node, *depth));
                }
            }
        }
    }

    log::info!("Deepest primary output: {:?}", deepest_primary_output);

    let mut deepest_path = vec![];
    // Get the GateRef having the largest depth.
    let deepest_gate_ref = deepest_primary_output.unwrap().0;
    let mut current_gate_ref = Some(deepest_gate_ref);
    while let Some(gate_ref) = current_gate_ref {
        deepest_path.push(gate_ref);
        // Get whichever arg of this gate has the largest depth.
        let gate: &gate::AigNode = &gate_fn.gates[gate_ref.id];
        if let gate::AigNode::Input { .. } = gate {
            break;
        }
        let args: Vec<gate::AigRef> = gate.get_args();
        assert!(!args.is_empty(), "gate {:?} should have args", gate);
        let max_arg_depth = args
            .iter()
            .map(|arg| depths.get(arg).unwrap())
            .max()
            .unwrap();
        current_gate_ref = args
            .iter()
            .find(|arg| depths.get(arg).unwrap() == max_arg_depth)
            .map(|arg| *arg);
    }

    let mut depth_to_count: HashMap<usize, usize> = HashMap::new();
    for node in live_nodes {
        if let Some(depth) = depths.get(node) {
            *depth_to_count.entry(*depth).or_insert(0) += 1;
        }
    }
    (depth_to_count, deepest_path)
}

fn collect_op_frequencies(f: &ir::Fn) -> HashMap<String, usize> {
    let mut op_freqs = HashMap::new();
    for node in f.nodes.iter() {
        let op = node.to_signature_string(f);
        *op_freqs.entry(op).or_insert(0) += 1;
    }
    op_freqs
}

fn main() {
    let _ = env_logger::builder().try_init();
    let args = Args::parse();

    // Read the file into a string.
    let file_content = std::fs::read_to_string(&args.input)
        .unwrap_or_else(|err| panic!("Failed to read {}: {}", args.input, err));
    let mut parser = ir_parser::Parser::new(&file_content);
    let ir_package = parser.parse_package().unwrap_or_else(|err| {
        eprintln!("Error parsing package: {:?}", err);
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

    let op_freqs = collect_op_frequencies(&ir_top);
    println!("Op frequencies:");
    let mut op_freqs_vec: Vec<_> = op_freqs.iter().collect();
    op_freqs_vec.sort_by(|(_, a), (_, b)| b.cmp(a));
    for (op, freq) in op_freqs_vec.iter() {
        println!("  {:4} :: {}", freq, op);
    }

    let gate_fn = ir2gate::gatify(
        &ir_top,
        ir2gate::GatifyOptions {
            fold: args.fold,
            check_equivalence: false,
        },
    )
    .unwrap();

    log::info!("gate fn signature: {}", gate_fn.get_signature());

    if args.check_equivalence {
        println!("== Checking equivalence...");
        let start = std::time::Instant::now();
        let eq = g8r::check_equivalence::validate_same_fn(&ir_top, &gate_fn);
        let duration = start.elapsed();
        println!("Equivalence check took {:?}.", duration);
        println!("Equivalence: {:?}", eq);
    }

    //let netlist = emit_netlist::emit_netlist(&args.module_name, &gate_fn);

    let id_to_use_count: HashMap<gate::AigRef, usize> = get_id_to_use_count(&gate_fn);
    let live_nodes: Vec<gate::AigRef> = id_to_use_count.keys().cloned().collect();

    let (depth_to_count, deepest_path) = get_gate_depth(&gate_fn, &live_nodes);

    println!("== Deepest path ({}):", deepest_path.len());
    for gate_ref in deepest_path.iter() {
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

    let structure_to_count = find_structures::find_structures(&gate_fn);
    let mut sorted_structures = structure_to_count
        .iter()
        .map(|(s, c)| (s.clone(), *c))
        .collect::<Vec<_>>();
    sorted_structures.sort_by(|(_s1, c1), (_s2, c2)| c2.cmp(c1));
    println!("== Structures:");
    for (structure, count) in sorted_structures[0..min(20, sorted_structures.len())].iter() {
        println!("{:3} :: {}", count, structure);
    }
}
