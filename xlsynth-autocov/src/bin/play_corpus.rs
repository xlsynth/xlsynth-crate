// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xlsynth_prover::prover::corner_prover::{
    CornerOrResult, CornerProver, CornerProverOptions, CornerQuery,
};

use clap::{Parser, ValueEnum};

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

    /// How much boolean witness information to print.
    #[arg(long, value_enum, default_value_t = ShowBoolsMode::Quiet)]
    show_bools: ShowBoolsMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ShowBoolsMode {
    Quiet,
    Sample,
    All,
}

#[derive(Clone)]
struct BestSample {
    complexity: u64,
    sample_idx: usize,
    value_str: String,
}

#[derive(Default, Clone)]
struct OutcomeBest {
    pass_best: Option<BestSample>,
    fail_best: Option<BestSample>,
    pass_count: usize,
    fail_count: usize,
}

#[derive(Default, Clone)]
struct BoolBestSamples {
    false_outcome: OutcomeBest,
    true_outcome: OutcomeBest,
}

#[derive(Default, Clone)]
struct OutcomeAll {
    pass_samples: Vec<usize>,
    fail_samples: Vec<usize>,
}

#[derive(Default, Clone)]
struct BoolAllSamples {
    false_outcome: OutcomeAll,
    true_outcome: OutcomeAll,
}

enum ShowBoolsAccumulator {
    Quiet,
    Sample {
        best_by_node: BTreeMap<usize, BoolBestSamples>,
    },
    All {
        all_by_node: BTreeMap<usize, BoolAllSamples>,
        corpus_value_strs: Vec<String>,
    },
}

impl ShowBoolsAccumulator {
    fn new(mode: ShowBoolsMode, bool_nodes: &BTreeSet<usize>) -> Self {
        match mode {
            ShowBoolsMode::Quiet => ShowBoolsAccumulator::Quiet,
            ShowBoolsMode::Sample => {
                let mut best_by_node = BTreeMap::new();
                for &node_text_id in bool_nodes.iter() {
                    best_by_node.insert(node_text_id, BoolBestSamples::default());
                }
                ShowBoolsAccumulator::Sample { best_by_node }
            }
            ShowBoolsMode::All => {
                let mut all_by_node = BTreeMap::new();
                for &node_text_id in bool_nodes.iter() {
                    all_by_node.insert(node_text_id, BoolAllSamples::default());
                }
                ShowBoolsAccumulator::All {
                    all_by_node,
                    corpus_value_strs: Vec::new(),
                }
            }
        }
    }

    fn on_sample(
        &mut self,
        sample_idx: usize,
        v: &xlsynth::IrValue,
        ok: bool,
        bool_events: &BTreeSet<xlsynth_autocov::BoolEventId>,
        bool_nodes: &BTreeSet<usize>,
    ) {
        match self {
            ShowBoolsAccumulator::Quiet => {}
            ShowBoolsAccumulator::Sample { best_by_node } => {
                let complexity = xlsynth_autocov::irvalue_complexity_key(v);
                let value_str = v.to_string();
                for ev in bool_events.iter() {
                    if !bool_nodes.contains(&ev.node_text_id) {
                        continue;
                    }
                    let ent = best_by_node
                        .entry(ev.node_text_id)
                        .or_insert_with(BoolBestSamples::default);
                    let outcome = if ev.value {
                        &mut ent.true_outcome
                    } else {
                        &mut ent.false_outcome
                    };

                    let (count_ref, best_ref) = if ok {
                        (&mut outcome.pass_count, &mut outcome.pass_best)
                    } else {
                        (&mut outcome.fail_count, &mut outcome.fail_best)
                    };
                    *count_ref += 1;
                    let should_replace = match best_ref {
                        None => true,
                        Some(cur) => (complexity, sample_idx) < (cur.complexity, cur.sample_idx),
                    };
                    if should_replace {
                        *best_ref = Some(BestSample {
                            complexity,
                            sample_idx,
                            value_str: value_str.clone(),
                        });
                    }
                }
            }
            ShowBoolsAccumulator::All {
                all_by_node,
                corpus_value_strs,
            } => {
                // Always store every sample string so `corpus:N` is a stable index space.
                corpus_value_strs.push(v.to_string());
                for ev in bool_events.iter() {
                    if !bool_nodes.contains(&ev.node_text_id) {
                        continue;
                    }
                    let ent = all_by_node
                        .entry(ev.node_text_id)
                        .or_insert_with(BoolAllSamples::default);
                    let outcome = if ev.value {
                        &mut ent.true_outcome
                    } else {
                        &mut ent.false_outcome
                    };
                    if ok {
                        outcome.pass_samples.push(sample_idx);
                    } else {
                        outcome.fail_samples.push(sample_idx);
                    }
                }
            }
        }
    }

    fn print(&self, engine: &xlsynth_autocov::AutocovEngine, bool_nodes: &BTreeSet<usize>) {
        match self {
            ShowBoolsAccumulator::Quiet => {}
            ShowBoolsAccumulator::Sample { best_by_node } => {
                fn fmt_witness(best: &Option<BestSample>, count: usize) -> String {
                    match best.as_ref() {
                        None => "(none)".to_string(),
                        Some(b) => {
                            if count > 1 {
                                format!(
                                    "{} @ corpus:{} ... {} more",
                                    b.value_str,
                                    b.sample_idx,
                                    count - 1
                                )
                            } else {
                                format!("{} @ corpus:{}", b.value_str, b.sample_idx)
                            }
                        }
                    }
                }

                for node_text_id in bool_nodes.iter().copied() {
                    let node_str = engine
                        .node_to_string_by_text_id(node_text_id)
                        .unwrap_or_else(|| format!("bool_node_id={}", node_text_id));
                    let ent = best_by_node.get(&node_text_id).cloned().unwrap_or_default();
                    println!("{}:", node_str);
                    println!("  false:");
                    println!(
                        "    passes: {}",
                        fmt_witness(&ent.false_outcome.pass_best, ent.false_outcome.pass_count)
                    );
                    if ent.false_outcome.fail_count > 0 {
                        println!(
                            "    failures: {}",
                            fmt_witness(&ent.false_outcome.fail_best, ent.false_outcome.fail_count)
                        );
                    }
                    println!("  true:");
                    println!(
                        "    passes: {}",
                        fmt_witness(&ent.true_outcome.pass_best, ent.true_outcome.pass_count)
                    );
                    if ent.true_outcome.fail_count > 0 {
                        println!(
                            "    failures: {}",
                            fmt_witness(&ent.true_outcome.fail_best, ent.true_outcome.fail_count)
                        );
                    }
                }
            }
            ShowBoolsAccumulator::All {
                all_by_node,
                corpus_value_strs,
            } => {
                fn print_list(label: &str, idxs: &[usize], corpus_value_strs: &[String]) {
                    println!("    {} ({}):", label, idxs.len());
                    for &idx in idxs.iter() {
                        let v = corpus_value_strs
                            .get(idx)
                            .map(|s| s.as_str())
                            .unwrap_or("<missing>");
                        println!("      {} @ corpus:{}", v, idx);
                    }
                }

                for node_text_id in bool_nodes.iter().copied() {
                    let node_str = engine
                        .node_to_string_by_text_id(node_text_id)
                        .unwrap_or_else(|| format!("bool_node_id={}", node_text_id));
                    let ent = all_by_node.get(&node_text_id).cloned().unwrap_or_default();
                    println!("{}:", node_str);
                    println!("  false:");
                    print_list("passes", &ent.false_outcome.pass_samples, corpus_value_strs);
                    if !ent.false_outcome.fail_samples.is_empty() {
                        print_list(
                            "failures",
                            &ent.false_outcome.fail_samples,
                            corpus_value_strs,
                        );
                    }
                    println!("  true:");
                    print_list("passes", &ent.true_outcome.pass_samples, corpus_value_strs);
                    if !ent.true_outcome.fail_samples.is_empty() {
                        print_list(
                            "failures",
                            &ent.true_outcome.fail_samples,
                            corpus_value_strs,
                        );
                    }
                }
            }
        }
    }
}

enum CorpusInput<'a> {
    File(&'a Path),
    Dir(&'a Path),
}

fn corpus_input_from_args(args: &Args) -> anyhow::Result<CorpusInput<'_>> {
    let corpus_file = args.corpus_file.as_deref();
    let corpus_dir = args.corpus_dir.as_deref();
    if corpus_file.is_some() == corpus_dir.is_some() {
        return Err(anyhow::anyhow!(
            "provide exactly one of --corpus-file or --corpus-dir"
        ));
    }
    Ok(if let Some(p) = corpus_file {
        CorpusInput::File(p)
    } else {
        CorpusInput::Dir(corpus_dir.expect("one of file/dir"))
    })
}

struct RunAccumulator {
    observed_any: BTreeSet<xlsynth_autocov::CornerEventId>,
    bool_observed_any: BTreeSet<xlsynth_autocov::BoolEventId>,
    samples_total: usize,
    samples_ok: usize,
    samples_failed: usize,
    show_bools: ShowBoolsAccumulator,
}

impl RunAccumulator {
    fn new(show_bools: ShowBoolsAccumulator) -> Self {
        Self {
            observed_any: BTreeSet::new(),
            bool_observed_any: BTreeSet::new(),
            samples_total: 0,
            samples_ok: 0,
            samples_failed: 0,
            show_bools,
        }
    }

    fn on_sample_result(
        &mut self,
        sample_idx: usize,
        v: &xlsynth::IrValue,
        ok: bool,
        corner_events: BTreeSet<xlsynth_autocov::CornerEventId>,
        bool_events: BTreeSet<xlsynth_autocov::BoolEventId>,
        bool_nodes: &BTreeSet<usize>,
    ) {
        self.observed_any.extend(corner_events);
        self.bool_observed_any.extend(bool_events.iter().copied());
        self.show_bools
            .on_sample(sample_idx, v, ok, &bool_events, bool_nodes);
        self.samples_total += 1;
        if ok {
            self.samples_ok += 1;
        } else {
            self.samples_failed += 1;
        }
    }
}

fn eval_one_sample(
    engine: &xlsynth_autocov::AutocovEngine,
    v: &xlsynth::IrValue,
) -> Result<
    (
        bool,
        BTreeSet<xlsynth_autocov::CornerEventId>,
        BTreeSet<xlsynth_autocov::BoolEventId>,
    ),
    String,
> {
    let bits = engine.bits_from_arg_tuple(v)?;
    let (ok, corners) = engine.evaluate_corner_events(&bits);
    let (_ok2, bools) = engine.evaluate_bool_events(&bits);
    Ok((ok, corners, bools))
}

fn replay_corpus(
    corpus: CorpusInput<'_>,
    max_samples: Option<usize>,
    engine: &xlsynth_autocov::AutocovEngine,
    bool_nodes: &BTreeSet<usize>,
    acc: &mut RunAccumulator,
) -> anyhow::Result<()> {
    match corpus {
        CorpusInput::File(path) => {
            let f = std::fs::File::open(path)?;
            let rdr = std::io::BufReader::new(f);
            for (lineno, line) in rdr.lines().enumerate() {
                if max_samples.is_some_and(|m| acc.samples_total >= m) {
                    break;
                }
                let line = line?;
                let t = line.trim();
                if t.is_empty() {
                    continue;
                }
                let v = xlsynth::IrValue::parse_typed(t)?;
                let (ok, corners, bools) = eval_one_sample(engine, &v).map_err(|e| {
                    anyhow::anyhow!("corpus parse error at line {}: {}", lineno + 1, e)
                })?;
                let sample_idx = acc.samples_total;
                acc.on_sample_result(sample_idx, &v, ok, corners, bools, bool_nodes);
            }
        }
        CorpusInput::Dir(dir) => {
            let mut files = list_corpus_files(dir)?;
            if let Some(m) = max_samples {
                if files.len() > m {
                    files.truncate(m);
                }
            }
            for p in files {
                let s = std::fs::read_to_string(&p)?;
                let line = first_non_empty_trimmed_line(&s)
                    .map_err(|e| anyhow::anyhow!("{}: {}", p.display(), e))?;
                let v = xlsynth::IrValue::parse_typed(line)?;
                let (ok, corners, bools) = eval_one_sample(engine, &v)
                    .map_err(|e| anyhow::anyhow!("{}: {}", p.display(), e))?;
                let sample_idx = acc.samples_total;
                acc.on_sample_result(sample_idx, &v, ok, corners, bools, bool_nodes);
            }
        }
    }
    Ok(())
}

fn prove_never_observed_corners(
    engine: &xlsynth_autocov::AutocovEngine,
    ir_file: &Path,
    entry_fn: &str,
    never_observed: &[xlsynth_autocov::CornerEventId],
) -> anyhow::Result<()> {
    if never_observed.is_empty() {
        return Ok(());
    }

    let ir_text = std::fs::read_to_string(ir_file)?;
    let mut parser = xlsynth_pir::ir_parser::Parser::new(&ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .map_err(|e| anyhow::anyhow!("PIR parse: {}", e))?;
    let f = pkg
        .get_fn(entry_fn)
        .ok_or_else(|| anyhow::anyhow!("function not found: {}", entry_fn))?;

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
                let bits = engine
                    .bits_from_arg_tuple(&witness)
                    .map_err(|e| anyhow::anyhow!("witness not convertible to arg tuple: {}", e))?;
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
    Ok(())
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
    let corpus = corpus_input_from_args(&args)?;

    let cfg = xlsynth_autocov::AutocovConfig {
        seed: 0,
        max_iters: Some(0),
        max_corpus_len: None,
    };
    let engine = xlsynth_autocov::AutocovEngine::from_ir_path(&args.ir_file, &args.entry_fn, cfg)
        .map_err(|e| anyhow::anyhow!(e))?;

    let domain: BTreeSet<xlsynth_autocov::CornerEventId> = engine.corner_event_domain();
    let bool_domain: BTreeSet<xlsynth_autocov::BoolEventId> = engine.bool_event_domain();
    let bool_nodes: BTreeSet<usize> = bool_domain.iter().map(|e| e.node_text_id).collect();
    if domain.is_empty() {
        eprintln!("corner_domain_empty (no corner-like nodes found)");
    }

    let show_bools = ShowBoolsAccumulator::new(args.show_bools, &bool_nodes);
    let mut acc = RunAccumulator::new(show_bools);
    replay_corpus(corpus, args.max_samples, &engine, &bool_nodes, &mut acc)?;

    let never_observed: Vec<xlsynth_autocov::CornerEventId> =
        domain.difference(&acc.observed_any).copied().collect();
    let bool_never_observed: Vec<xlsynth_autocov::BoolEventId> = bool_domain
        .difference(&acc.bool_observed_any)
        .copied()
        .collect();

    println!(
        "samples_total={} samples_ok={} samples_failed={} corner_domain={} corner_observed_any={} corner_never_observed={} bool_domain={} bool_observed_any={} bool_never_observed={}",
        acc.samples_total,
        acc.samples_ok,
        acc.samples_failed,
        domain.len(),
        acc.observed_any.len(),
        never_observed.len(),
        bool_domain.len(),
        acc.bool_observed_any.len(),
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

    acc.show_bools.print(&engine, &bool_nodes);

    if args.prove && !never_observed.is_empty() {
        prove_never_observed_corners(&engine, &args.ir_file, &args.entry_fn, &never_observed)?;
    }

    Ok(())
}
