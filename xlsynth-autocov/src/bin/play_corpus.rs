// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::io::BufRead;
use std::path::{Path, PathBuf};

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
    if domain.is_empty() {
        eprintln!("corner_domain_empty (no corner-like nodes found)");
    }

    let mut observed_any: BTreeSet<xlsynth_autocov::CornerEventId> = BTreeSet::new();
    let mut samples_total: usize = 0;
    let mut samples_ok: usize = 0;
    let mut samples_failed: usize = 0;

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
            let bits = engine
                .bits_from_arg_tuple(&v)
                .map_err(|e| anyhow::anyhow!("corpus parse error at line {}: {}", lineno + 1, e))?;
            let (ok, events) = engine.evaluate_corner_events(&bits);
            observed_any.extend(events);
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
            let bits = engine
                .bits_from_arg_tuple(&v)
                .map_err(|e| anyhow::anyhow!("{}: {}", p.display(), e))?;
            let (ok, events) = engine.evaluate_corner_events(&bits);
            observed_any.extend(events);
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

    println!(
        "samples_total={} samples_ok={} samples_failed={} corner_domain={} corner_observed_any={} corner_never_observed={}",
        samples_total,
        samples_ok,
        samples_failed,
        domain.len(),
        observed_any.len(),
        never_observed.len()
    );
    for ev in never_observed {
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

    Ok(())
}
