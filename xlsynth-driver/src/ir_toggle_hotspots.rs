// SPDX-License-Identifier: Apache-2.0

//! CLI adapter for ordered IR/AIG toggle activity.

use std::path::Path;

use clap::ArgMatches;
use xlsynth_g8r::ir_toggle_hotspots::analyze_toggle_hotspots_from_files;

pub fn handle_ir_toggle_hotspots(matches: &ArgMatches) {
    if let Err(error) = run(matches) {
        eprintln!("ir-toggle-hotspots: {error}");
        std::process::exit(2);
    }
}

fn run(matches: &ArgMatches) -> Result<(), String> {
    let path = matches
        .get_one::<String>("ir_input_file")
        .expect("required IR file");
    let ir_text =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read IR {path:?}: {e}"))?;
    let irvals = matches
        .get_one::<String>("input_irvals")
        .expect("required stimulus file");
    let mut report = analyze_toggle_hotspots_from_files(
        &ir_text,
        matches.get_one::<String>("top").map(String::as_str),
        Path::new(irvals),
    )?;
    let limit = *matches.get_one::<usize>("limit").expect("default limit");
    report.words.truncate(limit);
    report.gates.truncate(limit);

    if matches
        .get_one::<String>("format")
        .is_some_and(|format| format == "json")
    {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serializable activity report")
        );
    } else {
        println!(
            "{}: {} samples, {} transitions",
            report.function, report.sample_count, report.transition_count
        );
        println!("IR words (word changes, bit flips):");
        for word in &report.words {
            println!(
                "  {} id={} op={} word={} bits={}",
                word.ir_node.name,
                word.ir_node.id,
                word.ir_node.op,
                word.word_toggles,
                word.bit_toggles
            );
        }
        println!("AIG AND2 gates (output changes):");
        for gate in &report.gates {
            let sources = gate
                .sources
                .iter()
                .map(|source| source.name.as_str())
                .collect::<Vec<_>>()
                .join(",");
            let bits = gate
                .ir_output_bits
                .iter()
                .map(|bit| {
                    format!(
                        "{}{}[{}]",
                        if bit.inverted { "!" } else { "" },
                        bit.ir_node.name,
                        bit.bit_index
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "  %{} toggles={} sources=[{}] ir_bits=[{}]",
                gate.node_id, gate.toggle_count, sources, bits
            );
        }
    }
    Ok(())
}
