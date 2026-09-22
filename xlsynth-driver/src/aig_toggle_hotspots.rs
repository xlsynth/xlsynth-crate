// SPDX-License-Identifier: Apache-2.0

//! CLI adapter for ordered AIG AND2 toggle activity.

use std::path::Path;

use clap::ArgMatches;
use xlsynth_g8r::aig_toggle_hotspots::analyze_aig_toggle_hotspots_from_files;

use crate::fn_type_arg::parse_function_type_text;

pub fn handle_aig_toggle_hotspots(matches: &ArgMatches) {
    if let Err(error) = run(matches) {
        eprintln!("aig-toggle-hotspots: {error}");
        std::process::exit(2);
    }
}

fn run(matches: &ArgMatches) -> Result<(), String> {
    let artifact = matches.get_one::<String>("aig_file").expect("required AIG");
    let irvals = matches
        .get_one::<String>("input_irvals")
        .expect("required stimulus");
    let source_ir = matches
        .get_one::<String>("source_ir")
        .map(|path| {
            std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read source IR {path:?}: {e}"))
        })
        .transpose()?;
    let fn_type = matches
        .get_one::<String>("fn_type")
        .map(|text| parse_function_type_text(text))
        .transpose()?;
    let mut report = analyze_aig_toggle_hotspots_from_files(
        Path::new(artifact),
        Path::new(irvals),
        fn_type.as_ref(),
        source_ir.as_ref().map(|text| {
            (
                text.as_str(),
                matches.get_one::<String>("top").map(String::as_str),
            )
        }),
    )?;
    report
        .gates
        .truncate(*matches.get_one::<usize>("limit").expect("default limit"));
    if matches
        .get_one::<String>("format")
        .is_some_and(|format| format == "json")
    {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serializable AIG activity report")
        );
    } else {
        println!(
            "{}: {} samples, {} transitions",
            artifact, report.sample_count, report.transition_count
        );
        println!("AIG AND2 gates (output changes):");
        for gate in &report.gates {
            let sources = gate
                .sources
                .iter()
                .map(|source| {
                    source
                        .name
                        .as_ref()
                        .map(|name| {
                            format!(
                                "{}:{}({})",
                                source.id,
                                name,
                                source.op.as_deref().unwrap_or("?")
                            )
                        })
                        .unwrap_or_else(|| source.id.to_string())
                })
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "  %{} toggles={} sources=[{}]",
                gate.node_id, gate.toggle_count, sources
            );
        }
    }
    Ok(())
}
