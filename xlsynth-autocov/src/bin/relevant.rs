// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use clap::ValueEnum;
use xlsynth_prover::prover::SolverChoice;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum SolverArg {
    /// Let the library select an appropriate prover based on available
    /// features.
    Auto,
    /// Use the external XLS tool-chain binaries (configured via tool_path or
    /// environment).
    Toolchain,
    #[cfg(feature = "has-easy-smt")]
    Z3Binary,
    #[cfg(feature = "has-easy-smt")]
    BitwuzlaBinary,
    #[cfg(feature = "has-easy-smt")]
    BoolectorBinary,
    #[cfg(feature = "has-bitwuzla")]
    Bitwuzla,
    #[cfg(feature = "has-boolector")]
    Boolector,
}

fn solver_choice_from_arg(arg: SolverArg) -> SolverChoice {
    match arg {
        SolverArg::Auto => SolverChoice::Auto,
        SolverArg::Toolchain => SolverChoice::Toolchain,
        #[cfg(feature = "has-easy-smt")]
        SolverArg::Z3Binary => SolverChoice::Z3Binary,
        #[cfg(feature = "has-easy-smt")]
        SolverArg::BitwuzlaBinary => SolverChoice::BitwuzlaBinary,
        #[cfg(feature = "has-easy-smt")]
        SolverArg::BoolectorBinary => SolverChoice::BoolectorBinary,
        #[cfg(feature = "has-bitwuzla")]
        SolverArg::Bitwuzla => SolverChoice::Bitwuzla,
        #[cfg(feature = "has-boolector")]
        SolverArg::Boolector => SolverChoice::Boolector,
    }
}

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
    /// `toolchain`, `bitwuzla` when enabled).
    #[arg(long, value_enum, default_value_t = SolverArg::Auto)]
    solver: SolverArg,

    /// Optional toolchain path (used only with `--solver toolchain`).
    #[arg(long)]
    tool_path: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let ir_text = std::fs::read_to_string(&args.ir_file)
        .with_context(|| format!("reading ir file: {}", args.ir_file.display()))?;

    let solver = solver_choice_from_arg(args.solver);
    if args.tool_path.is_some() && solver != SolverChoice::Toolchain {
        anyhow::bail!("--tool-path is only valid with --solver toolchain");
    }

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

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "has-bitwuzla")]
    fn test_solver_arg_accepts_bitwuzla_when_enabled() {
        let args = Args::try_parse_from([
            "xlsynth-autocov-relevant",
            "--ir-file",
            "p.ir",
            "--entry-fn",
            "f",
            "--node-text-id",
            "0",
            "--solver",
            "bitwuzla",
        ])
        .expect("args should parse");
        assert_eq!(args.solver, SolverArg::Bitwuzla);
    }
}
