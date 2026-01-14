// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

pub fn handle_ir_diverse_samples(matches: &ArgMatches) -> Result<(), String> {
    let corpus_dir = matches
        .get_one::<String>("corpus_dir")
        .expect("corpus_dir is required");

    let signature_depth: usize = match matches.get_one::<String>("signature_depth") {
        Some(s) => s
            .parse::<usize>()
            .map_err(|_| format!("Invalid --signature-depth: {s:?}"))?,
        None => 2,
    };

    let log_skipped = match matches.get_one::<String>("log-skipped").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };

    let explain_new_hashes = match matches
        .get_one::<String>("explain-new-hashes")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };

    let selected = xlsynth_g8r::diverse_samples::select_ir_diverse_samples_with_options(
        std::path::Path::new(corpus_dir),
        &xlsynth_g8r::diverse_samples::DiverseSamplesOptions {
            signature_depth,
            log_skipped,
            explain_new_hashes,
        },
    )?;

    for e in selected.into_iter() {
        println!(
            "{} g8r-nodes={} g8r-levels={} new-hashes={}",
            e.ir_file_path.display(),
            e.g8r_nodes,
            e.g8r_levels,
            e.new_hashes
        );
        if explain_new_hashes {
            if let Some(details) = e.new_hash_details.as_ref() {
                for d in details.iter() {
                    println!(
                        "  new-signature node_index={} text_id={} {}",
                        d.node_index, d.text_id, d.signature
                    );
                }
            }
        }
    }
    Ok(())
}
