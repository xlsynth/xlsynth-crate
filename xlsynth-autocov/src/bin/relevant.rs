// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context;
use clap::Parser;
use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule, run_ir_equiv};
use xlsynth_prover::prover::SolverChoice;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism};

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

    // Parse and validate the package so we can (a) validate node type and (b)
    // keep the IR syntax stable with other tools.
    let mut parser = xlsynth_pir::ir_parser::Parser::new(&ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .map_err(|e| anyhow::anyhow!("PIR parse: {}", e))?;
    let f = pkg
        .get_fn(&args.entry_fn)
        .ok_or_else(|| anyhow::anyhow!("function not found: {}", args.entry_fn))?;

    let node = f
        .nodes
        .iter()
        .find(|n| n.text_id == args.node_text_id)
        .ok_or_else(|| anyhow::anyhow!("node_text_id not found: {}", args.node_text_id))?;
    if node.ty != xlsynth_pir::ir::Type::Bits(1) {
        return Err(anyhow::anyhow!(
            "node_text_id {} is not bits[1]; ty={:?}",
            args.node_text_id,
            node.ty
        ));
    }

    let stuck0 = xlsynth_autocov::clone_fn_with_stuck_at_bool_node(f, args.node_text_id, false)
        .map_err(|e| anyhow::anyhow!(e))?;
    let stuck1 = xlsynth_autocov::clone_fn_with_stuck_at_bool_node(f, args.node_text_id, true)
        .map_err(|e| anyhow::anyhow!(e))?;

    fn replace_fn_by_name(
        pkg: &mut xlsynth_pir::ir::Package,
        fn_name: &str,
        new_fn: xlsynth_pir::ir::Fn,
    ) -> anyhow::Result<()> {
        for member in pkg.members.iter_mut() {
            match member {
                xlsynth_pir::ir::PackageMember::Function(f) if f.name == fn_name => {
                    *f = new_fn;
                    return Ok(());
                }
                xlsynth_pir::ir::PackageMember::Block { func, .. } if func.name == fn_name => {
                    // We currently only analyze functions, but if a block wrapper uses
                    // the same name, replace it too for clarity.
                    *func = new_fn;
                    return Ok(());
                }
                _ => {}
            }
        }
        Err(anyhow::anyhow!(
            "function not found in package members: {}",
            fn_name
        ))
    }

    let mut pkg0 = pkg.clone();
    let mut pkg1 = pkg.clone();
    replace_fn_by_name(&mut pkg0, &args.entry_fn, stuck0)?;
    replace_fn_by_name(&mut pkg1, &args.entry_fn, stuck1)?;

    // Re-serialize each modified package into IR text for the prover.
    //
    // Note: The pretty-printer output should be deterministic.
    let lhs_text = pkg0.to_string();
    let rhs_text = pkg1.to_string();

    let solver = SolverChoice::from_str(&args.solver).map_err(|e| anyhow::anyhow!(e))?;

    let request = IrEquivRequest {
        lhs: IrModule::new(&lhs_text).with_top(Some(&args.entry_fn)),
        rhs: IrModule::new(&rhs_text).with_top(Some(&args.entry_fn)),
        drop_params: &[],
        flatten_aggregates: false,
        parallelism: EquivParallelism::SingleThreaded,
        assertion_semantics: AssertionSemantics::Same,
        assert_label_filter: None,
        solver: Some(solver),
        tool_path: args.tool_path.as_deref(),
    };
    let report = run_ir_equiv(&request).map_err(|e| anyhow::anyhow!(e))?;

    // The node is relevant iff stuck-at-0 and stuck-at-1 are *not* equivalent.
    match report.result {
        xlsynth_prover::prover::types::EquivResult::Proved => {
            println!(
                "relevant_result node_text_id={} relevant=false",
                args.node_text_id
            );
            Ok(())
        }
        xlsynth_prover::prover::types::EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } => {
            println!(
                "relevant_result node_text_id={} relevant=true lhs_inputs={:?} rhs_inputs={:?} lhs_output={:?} rhs_output={:?}",
                args.node_text_id, lhs_inputs, rhs_inputs, lhs_output, rhs_output
            );
            Ok(())
        }
        xlsynth_prover::prover::types::EquivResult::ToolchainDisproved(msg) => {
            println!(
                "relevant_result node_text_id={} relevant=true toolchain_disproved={:?}",
                args.node_text_id, msg
            );
            Ok(())
        }
        xlsynth_prover::prover::types::EquivResult::Error(msg) => Err(anyhow::anyhow!(msg)),
    }
}
