// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

#[cfg(unix)]
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

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

    let make_symlink_dir: Option<PathBuf> = matches
        .get_one::<String>("make_symlink_dir")
        .map(|s| PathBuf::from(s));

    let selected = xlsynth_g8r::diverse_samples::select_ir_diverse_samples_with_options(
        Path::new(corpus_dir),
        &xlsynth_g8r::diverse_samples::DiverseSamplesOptions {
            signature_depth,
            log_skipped,
            explain_new_hashes,
        },
    )?;

    if let Some(dir) = make_symlink_dir.as_deref() {
        populate_symlink_dir(dir, &selected)?;
    }

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

fn populate_symlink_dir(
    dir: &Path,
    selected: &[xlsynth_g8r::diverse_samples::DiverseSampleSelectionEntry],
) -> Result<(), String> {
    if dir.exists() {
        if !dir.is_dir() {
            return Err(format!(
                "--make-symlink-dir path exists but is not a directory: {}",
                dir.display()
            ));
        }
        let mut it = std::fs::read_dir(dir)
            .map_err(|e| format!("failed to read symlink dir {}: {e}", dir.display()))?;
        if it.next().is_some() {
            return Err(format!(
                "--make-symlink-dir directory must be empty (refusing to overwrite): {}",
                dir.display()
            ));
        }
    } else {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("failed to create symlink dir {}: {e}", dir.display()))?;
    }

    #[cfg(not(unix))]
    {
        let _ = selected;
        return Err("--make-symlink-dir is only supported on unix platforms".to_string());
    }

    #[cfg(unix)]
    {
        for (i, e) in selected.iter().enumerate() {
            let basename = e
                .ir_file_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| {
                    format!(
                        "selected sample has non-utf8 filename: {}",
                        e.ir_file_path.display()
                    )
                })?;
            let link_name = format!("{:05}_{}", i, basename);
            let link_path = dir.join(link_name);
            symlink(&e.ir_file_path, &link_path).map_err(|e2| {
                format!(
                    "failed to create symlink {} -> {}: {e2}",
                    link_path.display(),
                    e.ir_file_path.display()
                )
            })?;
        }
        Ok(())
    }
}
