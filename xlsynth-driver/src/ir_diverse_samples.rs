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
            // Use an absolute target so the symlink remains valid regardless of the
            // symlink's location. This avoids surprising breakage when
            // `ir_file_path` is relative (e.g. when the corpus dir passed to
            // `ir-diverse-samples` was `.`).
            //
            // We first try to interpret relative paths as relative to the symlink
            // directory's parent (common case: `--make-symlink-dir` lives under
            // the corpus dir). Regardless of where we found it, we always
            // canonicalize to ensure an absolute target.
            let candidate: PathBuf = if e.ir_file_path.is_absolute() {
                e.ir_file_path.clone()
            } else if let Some(parent) = dir.parent() {
                parent.join(&e.ir_file_path)
            } else {
                e.ir_file_path.clone()
            };
            let target_path = std::fs::canonicalize(&candidate).map_err(|e2| {
                format!(
                    "failed to canonicalize selected sample path {}: {e2}",
                    candidate.display()
                )
            })?;

            symlink(&target_path, &link_path).map_err(|e2| {
                format!(
                    "failed to create symlink {} -> {}: {e2}",
                    link_path.display(),
                    target_path.display()
                )
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::populate_symlink_dir;
    use std::path::PathBuf;
    use xlsynth_g8r::diverse_samples::DiverseSampleSelectionEntry;

    #[test]
    fn populate_symlink_dir_uses_absolute_targets_for_relative_inputs() {
        // Arrange a corpus-like layout:
        //   <tmp>/samples/foo.ir
        //   <tmp>/diverse/ (symlink dir)
        let tmp = tempfile::tempdir().expect("tempdir");
        let corpus_root = tmp.path();
        let samples_dir = corpus_root.join("samples");
        std::fs::create_dir_all(&samples_dir).expect("create samples dir");
        let sample_path = samples_dir.join("foo.ir");
        std::fs::write(&sample_path, "package p\n").expect("write sample");

        let symlink_dir = corpus_root.join("diverse");
        // Provide a relative path like `samples/foo.ir` (this is what the CLI prints
        // today).
        let selected = vec![DiverseSampleSelectionEntry {
            ir_file_path: PathBuf::from("samples/foo.ir"),
            g8r_nodes: 0,
            g8r_levels: 0,
            new_hashes: 0,
            new_hash_details: None,
        }];

        populate_symlink_dir(&symlink_dir, &selected).expect("populate");

        let link_path = symlink_dir.join("00000_foo.ir");
        let target = std::fs::read_link(&link_path).expect("read_link");
        assert!(
            target.is_absolute(),
            "expected absolute symlink target, got: {}",
            target.display()
        );
        assert_eq!(
            std::fs::canonicalize(target).expect("canonicalize target"),
            std::fs::canonicalize(sample_path).expect("canonicalize sample")
        );

        // Ensure it is a usable link.
        let _ = std::fs::read_to_string(&link_path).expect("read through link");
    }

    #[test]
    fn populate_symlink_dir_uses_absolute_targets_when_symlink_dir_is_relative() {
        // Mirror common CLI usage:
        //   xlsynth-driver ir-diverse-samples . --make-symlink-dir diverse
        let tmp = tempfile::tempdir().expect("tempdir");
        let corpus_root = tmp.path();
        let samples_dir = corpus_root.join("samples");
        std::fs::create_dir_all(&samples_dir).expect("create samples dir");
        let sample_path = samples_dir.join("foo.ir");
        std::fs::write(&sample_path, "package p\n").expect("write sample");

        let old_cwd = std::env::current_dir().expect("current_dir");
        std::env::set_current_dir(corpus_root).expect("set_current_dir");

        let selected = vec![DiverseSampleSelectionEntry {
            ir_file_path: PathBuf::from("samples/foo.ir"),
            g8r_nodes: 0,
            g8r_levels: 0,
            new_hashes: 0,
            new_hash_details: None,
        }];

        let symlink_dir = PathBuf::from("diverse");
        populate_symlink_dir(&symlink_dir, &selected).expect("populate");

        let link_path = symlink_dir.join("00000_foo.ir");
        let target = std::fs::read_link(&link_path).expect("read_link");
        assert!(
            target.is_absolute(),
            "expected absolute symlink target, got: {}",
            target.display()
        );
        assert_eq!(
            std::fs::canonicalize(target).expect("canonicalize target"),
            std::fs::canonicalize(sample_path).expect("canonicalize sample")
        );

        std::env::set_current_dir(old_cwd).expect("restore cwd");
    }
}
