// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use clap::ArgMatches;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_utils::fn_node_count;

use crate::common::parse_bool_flag_or;
use crate::common::write_stdout_line;
use crate::toolchain_config::ToolchainConfig;

fn collect_ir_files_recursively(root: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;
            let is_ir = path.extension().and_then(|s| s.to_str()) == Some("ir");
            if ty.is_dir() {
                stack.push(path);
            } else if ty.is_file() {
                if is_ir {
                    out.push(path);
                }
            } else if ty.is_symlink() {
                if is_ir {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        if meta.is_file() {
                            out.push(path);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn count_nodes_in_file(path: &Path, top: Option<&str>) -> Result<usize, String> {
    let file_content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let mut parser = ir_parser::Parser::new(&file_content);
    let mut pkg = parser
        .parse_and_validate_package()
        .map_err(|e| format!("parse/validate failed for {}: {e}", path.display()))?;

    if let Some(top) = top {
        pkg.set_top_fn(top)
            .map_err(|e| format!("failed to set --top for {}: {e}", path.display()))?;
    }

    let top_fn = pkg
        .get_top_fn()
        .ok_or_else(|| format!("no top function found in {}", path.display()))?;

    Ok(fn_node_count(top_fn))
}

struct WorkResult {
    index: usize,
    path: PathBuf,
    result: Result<usize, String>,
}

pub fn handle_ir_fn_node_count_corpus(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let corpus_dir = matches
        .get_one::<String>("corpus_dir")
        .expect("corpus_dir is required");
    let top = matches.get_one::<String>("ir_top").cloned();
    let ignore_parse_errors = parse_bool_flag_or(matches, "ignore-parse-errors", true);

    let max_files: Option<usize> = matches
        .get_one::<String>("max-files")
        .map(|s| {
            s.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: ir-fn-node-count-corpus: --max-files must be an integer");
                std::process::exit(2);
            })
        })
        .filter(|&n| n != 0);

    let mut files: Vec<PathBuf> = Vec::new();
    let root = Path::new(corpus_dir);
    if let Err(e) = collect_ir_files_recursively(root, &mut files) {
        eprintln!(
            "error: ir-fn-node-count-corpus: failed to read corpus dir {}: {e}",
            root.display()
        );
        std::process::exit(2);
    }
    files.sort();

    if let Some(limit) = max_files {
        files.truncate(limit);
    }
    let job_count = files.len();
    if job_count == 0 {
        return;
    }

    let worker_count = std::cmp::max(1, std::cmp::min(num_cpus::get(), job_count));
    let job_queue: VecDeque<(usize, PathBuf)> = files.into_iter().enumerate().collect();
    let jobs = Arc::new(Mutex::new(job_queue));
    let (tx, rx) = mpsc::channel::<WorkResult>();

    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let jobs = Arc::clone(&jobs);
        let tx = tx.clone();
        let top = top.clone();
        handles.push(thread::spawn(move || loop {
            let next_job = {
                let mut jobs = jobs.lock().expect("job queue mutex poisoned");
                jobs.pop_front()
            };
            let Some((index, path)) = next_job else {
                break;
            };

            let result = count_nodes_in_file(&path, top.as_deref());
            if tx
                .send(WorkResult {
                    index,
                    path,
                    result,
                })
                .is_err()
            {
                break;
            }
        }));
    }
    drop(tx);

    let mut pending = BTreeMap::<usize, WorkResult>::new();
    let mut next_to_emit = 0usize;

    for _ in 0..job_count {
        let work = rx.recv().unwrap_or_else(|_| {
            eprintln!(
                "error: ir-fn-node-count-corpus: internal error: worker channel closed unexpectedly"
            );
            std::process::exit(1);
        });
        pending.insert(work.index, work);

        while let Some(ready) = pending.remove(&next_to_emit) {
            match ready.result {
                Ok(node_count) => {
                    write_stdout_line(&format!("{}: {}", ready.path.display(), node_count));
                }
                Err(err) => {
                    if !ignore_parse_errors {
                        eprintln!("error: ir-fn-node-count-corpus: {err}");
                        std::process::exit(2);
                    }
                }
            }
            next_to_emit += 1;
        }
    }

    for handle in handles {
        let _ = handle.join();
    }
}
