// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context;
use clap::Parser;
use xlsynth_prover::prover::SolverChoice;

#[derive(Debug, Parser)]
#[command(name = "xlsynth-autocov-relevant")]
#[command(
    about = "Checks whether a boolean node is relevant by proving stuck-at-0 vs stuck-at-1 are equivalent"
)]
struct Args {
    /// Path to an XLS IR text file (package).
    #[arg(long)]
    ir_file: PathBuf,

    /// Name of the function within the IR package to analyze.
    #[arg(long)]
    entry_fn: String,

    /// IR node text id for the boolean value to check (must be `bits[1]`).
    #[arg(long)]
    node_text_id: usize,

    /// Solver selection for equivalence checking.
    ///
    /// Values match `xlsynth-prover`'s `SolverChoice` strings (e.g. `auto`,
    /// `toolchain`, `z3-binary` when enabled).
    #[arg(long, default_value = "auto")]
    solver: String,

    /// Optional toolchain path (used only with `--solver toolchain`).
    #[arg(long)]
    tool_path: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let ir_text = std::fs::read_to_string(&args.ir_file)
        .with_context(|| format!("reading ir file: {}", args.ir_file.display()))?;

    let solver = SolverChoice::from_str(&args.solver).map_err(|e| anyhow::anyhow!(e))?;

    let r = xlsynth_autocov::relevant_from_ir_text(
        &ir_text,
        &args.entry_fn,
        args.node_text_id,
        xlsynth_autocov::RelevanceCheckMethod::Prove {
            solver,
            tool_path: args.tool_path,
        },
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    match r.detail {
        xlsynth_autocov::RelevanceDetail::ProvedEquivalent
        | xlsynth_autocov::RelevanceDetail::ExhaustiveEquivalent => {
            println!(
                "relevant_result node_text_id={} relevant=false",
                args.node_text_id
            );
            Ok(())
        }
        xlsynth_autocov::RelevanceDetail::DisprovedEquivalent { equiv } => {
            println!(
                "relevant_result node_text_id={} relevant=true equiv={:?}",
                args.node_text_id, equiv
            );
            Ok(())
        }
        xlsynth_autocov::RelevanceDetail::ExhaustiveNotEquivalent {
            counterexample_args,
        } => {
            println!(
                "relevant_result node_text_id={} relevant=true counterexample_args={:?}",
                args.node_text_id, counterexample_args
            );
            Ok(())
        }
    }
}
