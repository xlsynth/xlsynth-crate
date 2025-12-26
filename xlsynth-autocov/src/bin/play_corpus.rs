// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xlsynth_prover::prover::corner_prover::{
    CornerOrResult, CornerProver, CornerProverOptions, CornerQuery,
};

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "xlsynth-autocov-play-corpus")]
#[command(
    about = "Replays a corpus through an XLS IR function and reports corner events missed by all samples"
)]
struct Args {
    /// Path to an XLS IR text file (package).
    #[arg(long)]
    ir_file: PathBuf,

    /// Name of the function within the IR package to evaluate.
    #[arg(long)]
    entry_fn: String,

    /// Newline-delimited corpus file. Each line is a typed IR tuple value whose
    /// elements are the function arguments.
    #[arg(long)]
    corpus_file: Option<PathBuf>,

    /// Directory containing corpus samples. Each file must contain a single
    /// typed IR tuple value (whitespace allowed).
    #[arg(long)]
    corpus_dir: Option<PathBuf>,

    /// Optional cap on number of samples processed (after sorting).
    #[arg(long)]
    max_samples: Option<usize>,

    /// Emit a human-oriented explanation for each never-observed corner event.
    #[arg(long, default_value_t = true)]
    explain: bool,

    /// Prove never-observed corners reachable/unreachable using an SMT solver.
    ///
    /// This runs after printing `corner_never_observed ...` lines.
    #[arg(long, default_value_t = true)]
    prove: bool,

    /// Show, for each bool node, which corpus samples demonstrate `false` vs
    /// `true`.
    #[arg(long, default_value_t = false)]
    show_bools: bool,
}

fn corner_kind_str(k: xlsynth_pir::corners::CornerKind) -> &'static str {
    match k {
        xlsynth_pir::corners::CornerKind::Add => "Add",
        xlsynth_pir::corners::CornerKind::Neg => "Neg",
        xlsynth_pir::corners::CornerKind::SignExt => "SignExt",
        xlsynth_pir::corners::CornerKind::DynamicBitSlice => "DynamicBitSlice",
        xlsynth_pir::corners::CornerKind::ArrayIndex => "ArrayIndex",
        xlsynth_pir::corners::CornerKind::Shift => "Shift",
        xlsynth_pir::corners::CornerKind::Shra => "Shra",
        xlsynth_pir::corners::CornerKind::CompareDistance => "CompareDistance",
    }
}

fn list_corpus_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = Vec::new();
    for ent in std::fs::read_dir(dir)? {
        let ent = ent?;
        let ty = ent.file_type()?;
        if !ty.is_file() {
            continue;
        }
        out.push(ent.path());
    }
    out.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
    Ok(out)
}

fn first_non_empty_trimmed_line(s: &str) -> Result<&str, String> {
    let mut found: Option<&str> = None;
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if found.is_some() {
            return Err("expected exactly one non-empty line in sample file".to_string());
        }
        found = Some(t);
    }
    found.ok_or_else(|| "sample file had no non-empty lines".to_string())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let corpus_file = args.corpus_file.as_ref();
    let corpus_dir = args.corpus_dir.as_ref();
    if corpus_file.is_some() == corpus_dir.is_some() {
        return Err(anyhow::anyhow!(
            "provide exactly one of --corpus-file or --corpus-dir"
        ));
    }

    let cfg = xlsynth_autocov::AutocovConfig {
        seed: 0,
        max_iters: Some(0),
    };
    let engine = xlsynth_autocov::AutocovEngine::from_ir_path(&args.ir_file, &args.entry_fn, cfg)
        .map_err(|e| anyhow::anyhow!(e))?;

    let domain: BTreeSet<xlsynth_autocov::CornerEventId> = engine.corner_event_domain();
    let bool_domain: BTreeSet<xlsynth_autocov::BoolEventId> = engine.bool_event_domain();
    let bool_nodes: BTreeSet<usize> = bool_domain.iter().map(|e| e.node_text_id).collect();
    if domain.is_empty() {
        eprintln!("corner_domain_empty (no corner-like nodes found)");
    }

    let mut observed_any: BTreeSet<xlsynth_autocov::CornerEventId> = BTreeSet::new();
    let mut bool_observed_any: BTreeSet<xlsynth_autocov::BoolEventId> = BTreeSet::new();
    let mut samples_total: usize = 0;
    let mut samples_ok: usize = 0;
    let mut samples_failed: usize = 0;

    #[derive(Clone)]
    struct BestSample {
        complexity: u64,
        sample_idx: usize,
        value_str: String,
    }

    #[derive(Default, Clone)]
    struct BoolBestSamples {
        false_best: Option<BestSample>,
        true_best: Option<BestSample>,
        false_count: usize,
        true_count: usize,
    }

    let mut bool_best_by_node: BTreeMap<usize, BoolBestSamples> = BTreeMap::new();
    if args.show_bools {
        for &node_text_id in bool_nodes.iter() {
            bool_best_by_node.insert(node_text_id, BoolBestSamples::default());
        }
    }

    if let Some(corpus_file) = corpus_file {
        let f = std::fs::File::open(corpus_file)?;
        let rdr = std::io::BufReader::new(f);
        for (lineno, line) in rdr.lines().enumerate() {
            if args.max_samples.is_some_and(|m| samples_total >= m) {
                break;
            }
            let line = line?;
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            let v = xlsynth::IrValue::parse_typed(t)?;
            let sample_complexity = if args.show_bools {
                Some(xlsynth_autocov::irvalue_complexity_key(&v))
            } else {
                None
            };
            let bits = engine
                .bits_from_arg_tuple(&v)
                .map_err(|e| anyhow::anyhow!("corpus parse error at line {}: {}", lineno + 1, e))?;
            let (ok, events) = engine.evaluate_corner_events(&bits);
            observed_any.extend(events);
            let (_ok2, bool_events) = engine.evaluate_bool_events(&bits);
            bool_observed_any.extend(bool_events.iter().copied());
            if args.show_bools && ok {
                let sample_idx = samples_total;
                let complexity = sample_complexity.expect("show_bools implies complexity computed");
                let value_str = v.to_string();
                for ev in bool_events.iter() {
                    if !bool_nodes.contains(&ev.node_text_id) {
                        continue;
                    }
                    let ent = bool_best_by_node
                        .entry(ev.node_text_id)
                        .or_insert_with(BoolBestSamples::default);
                    if ev.value {
                        ent.true_count += 1;
                        let should_replace = match &ent.true_best {
                            None => true,
                            Some(cur) => {
                                (complexity, sample_idx) < (cur.complexity, cur.sample_idx)
                            }
                        };
                        if should_replace {
                            ent.true_best = Some(BestSample {
                                complexity,
                                sample_idx,
                                value_str: value_str.clone(),
                            });
                        }
                    } else {
                        ent.false_count += 1;
                        let should_replace = match &ent.false_best {
                            None => true,
                            Some(cur) => {
                                (complexity, sample_idx) < (cur.complexity, cur.sample_idx)
                            }
                        };
                        if should_replace {
                            ent.false_best = Some(BestSample {
                                complexity,
                                sample_idx,
                                value_str: value_str.clone(),
                            });
                        }
                    }
                }
            }
            samples_total += 1;
            if ok {
                samples_ok += 1;
            } else {
                samples_failed += 1;
            }
        }
    } else if let Some(corpus_dir) = corpus_dir {
        let mut files = list_corpus_files(corpus_dir)?;
        if let Some(m) = args.max_samples {
            if files.len() > m {
                files.truncate(m);
            }
        }
        for p in files {
            let s = std::fs::read_to_string(&p)?;
            let line = first_non_empty_trimmed_line(&s)
                .map_err(|e| anyhow::anyhow!("{}: {}", p.display(), e))?;
            let v = xlsynth::IrValue::parse_typed(line)?;
            let sample_complexity = if args.show_bools {
                Some(xlsynth_autocov::irvalue_complexity_key(&v))
            } else {
                None
            };
            let bits = engine
                .bits_from_arg_tuple(&v)
                .map_err(|e| anyhow::anyhow!("{}: {}", p.display(), e))?;
            let (ok, events) = engine.evaluate_corner_events(&bits);
            observed_any.extend(events);
            let (_ok2, bool_events) = engine.evaluate_bool_events(&bits);
            bool_observed_any.extend(bool_events.iter().copied());
            if args.show_bools && ok {
                let sample_idx = samples_total;
                let complexity = sample_complexity.expect("show_bools implies complexity computed");
                let value_str = v.to_string();
                for ev in bool_events.iter() {
                    if !bool_nodes.contains(&ev.node_text_id) {
                        continue;
                    }
                    let ent = bool_best_by_node
                        .entry(ev.node_text_id)
                        .or_insert_with(BoolBestSamples::default);
                    if ev.value {
                        ent.true_count += 1;
                        let should_replace = match &ent.true_best {
                            None => true,
                            Some(cur) => {
                                (complexity, sample_idx) < (cur.complexity, cur.sample_idx)
                            }
                        };
                        if should_replace {
                            ent.true_best = Some(BestSample {
                                complexity,
                                sample_idx,
                                value_str: value_str.clone(),
                            });
                        }
                    } else {
                        ent.false_count += 1;
                        let should_replace = match &ent.false_best {
                            None => true,
                            Some(cur) => {
                                (complexity, sample_idx) < (cur.complexity, cur.sample_idx)
                            }
                        };
                        if should_replace {
                            ent.false_best = Some(BestSample {
                                complexity,
                                sample_idx,
                                value_str: value_str.clone(),
                            });
                        }
                    }
                }
            }
            samples_total += 1;
            if ok {
                samples_ok += 1;
            } else {
                samples_failed += 1;
            }
        }
    }

    let never_observed: Vec<xlsynth_autocov::CornerEventId> =
        domain.difference(&observed_any).copied().collect();
    let bool_never_observed: Vec<xlsynth_autocov::BoolEventId> = bool_domain
        .difference(&bool_observed_any)
        .copied()
        .collect();

    println!(
        "samples_total={} samples_ok={} samples_failed={} corner_domain={} corner_observed_any={} corner_never_observed={} bool_domain={} bool_observed_any={} bool_never_observed={}",
        samples_total,
        samples_ok,
        samples_failed,
        domain.len(),
        observed_any.len(),
        never_observed.len(),
        bool_domain.len(),
        bool_observed_any.len(),
        bool_never_observed.len()
    );
    for ev in never_observed.iter().copied() {
        if args.explain {
            let node_str = engine
                .node_to_string_by_text_id(ev.node_text_id)
                .unwrap_or_else(|| "<node not found>".to_string());
            let meaning = xlsynth_autocov::corner_tag_description(ev.kind, ev.tag);
            println!(
                "corner_never_observed node_id={} kind={} tag={} meaning=\"{}\" node=\"{}\"",
                ev.node_text_id,
                corner_kind_str(ev.kind),
                ev.tag,
                meaning,
                node_str
            );
        } else {
            println!(
                "corner_never_observed node_id={} kind={} tag={}",
                ev.node_text_id,
                corner_kind_str(ev.kind),
                ev.tag
            );
        }
    }
    for ev in bool_never_observed {
        let node_str = engine
            .node_to_string_by_text_id(ev.node_text_id)
            .unwrap_or_else(|| "<node not found>".to_string());
        println!(
            "bool_never_observed node_id={} value={} node=\"{}\"",
            ev.node_text_id,
            xlsynth_autocov::bool_value_description(ev.value),
            node_str
        );
    }

    if args.show_bools {
        for node_text_id in bool_nodes.iter().copied() {
            let node_str = engine
                .node_to_string_by_text_id(node_text_id)
                .unwrap_or_else(|| format!("bool_node_id={}", node_text_id));
            let ent = bool_best_by_node
                .get(&node_text_id)
                .cloned()
                .unwrap_or_default();
            println!("{}:", node_str);
            let false_line = match ent.false_best.as_ref() {
                None => "(none)".to_string(),
                Some(b) => {
                    if ent.false_count > 1 {
                        format!("{}, ... {} more", b.value_str, ent.false_count - 1)
                    } else {
                        b.value_str.clone()
                    }
                }
            };
            let true_line = match ent.true_best.as_ref() {
                None => "(none)".to_string(),
                Some(b) => {
                    if ent.true_count > 1 {
                        format!("{}, ... {} more", b.value_str, ent.true_count - 1)
                    } else {
                        b.value_str.clone()
                    }
                }
            };
            println!("  false: {}", false_line);
            println!("  true: {}", true_line);
        }
    }

    if args.prove && !never_observed.is_empty() {
        let ir_text = std::fs::read_to_string(&args.ir_file)?;
        let mut parser = xlsynth_pir::ir_parser::Parser::new(&ir_text);
        let pkg = parser
            .parse_and_validate_package()
            .map_err(|e| anyhow::anyhow!("PIR parse: {}", e))?;
        let f = pkg
            .get_fn(&args.entry_fn)
            .ok_or_else(|| anyhow::anyhow!("function not found: {}", args.entry_fn))?;

        let mut prover = CornerProver::new_auto(&pkg, f, CornerProverOptions::default())
            .map_err(|e| anyhow::anyhow!(e))?;

        let mut remaining: BTreeSet<xlsynth_autocov::CornerEventId> =
            never_observed.iter().copied().collect();

        while !remaining.is_empty() {
            let qs: Vec<CornerQuery> = remaining
                .iter()
                .map(|ev| CornerQuery {
                    node_text_id: ev.node_text_id,
                    kind: ev.kind,
                    tag: ev.tag,
                })
                .collect();

            let prove_start = Instant::now();
            let prove_r = prover.solve_any(&qs).map_err(|e| anyhow::anyhow!(e))?;
            let prove_seconds = prove_start.elapsed().as_secs_f64();
            match prove_r {
                CornerOrResult::Unsat => {
                    for ev in remaining.iter() {
                        println!(
                            "corner_prove node_id={} kind={} tag={} status=unreachable proven_seconds={:.3}",
                            ev.node_text_id,
                            corner_kind_str(ev.kind),
                            ev.tag,
                            prove_seconds
                        );
                    }
                    break;
                }
                CornerOrResult::Unknown { message } => {
                    return Err(anyhow::anyhow!("corner_prove unknown: {}", message));
                }
                CornerOrResult::Sat { witness } => {
                    let bits = engine.bits_from_arg_tuple(&witness).map_err(|e| {
                        anyhow::anyhow!("witness not convertible to arg tuple: {}", e)
                    })?;
                    let (ok, observed) = engine.evaluate_corner_events(&bits);
                    if !ok {
                        return Err(anyhow::anyhow!(
                            "corner_prove: SAT witness caused evaluation failure; witness={}",
                            witness
                        ));
                    }
                    let hit: Vec<xlsynth_autocov::CornerEventId> =
                        observed.intersection(&remaining).copied().collect();
                    if hit.is_empty() {
                        return Err(anyhow::anyhow!(
                            "corner_prove: SAT witness did not satisfy any remaining corner; witness={}",
                            witness
                        ));
                    }
                    for ev in hit.iter() {
                        println!(
                            "corner_prove node_id={} kind={} tag={} status=reachable witness=\"{}\" proven_seconds={:.3}",
                            ev.node_text_id,
                            corner_kind_str(ev.kind),
                            ev.tag,
                            witness,
                            prove_seconds
                        );
                    }
                    for ev in hit {
                        remaining.remove(&ev);
                    }
                }
            }
        }
    }

    Ok(())
}
