// SPDX-License-Identifier: Apache-2.0

use crate::gatify::ir2gate;
use crate::gatify::prep_for_gatify::PrepForGatifyOptions;
use xlsynth_pir::aug_opt::AugOptOptions;
use xlsynth_pir::desugar_extensions;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_prover::prover::SolverChoice;

pub struct Ir2GatesOptions {
    pub fold: bool,
    pub hash: bool,
    pub check_equivalence: bool,
    pub equivalence_solver: SolverChoice,
    pub enable_rewrite_carry_out: bool,
    pub enable_rewrite_prio_encode: bool,
    pub enable_rewrite_nary_add: bool,
    pub enable_rewrite_mask_low: bool,
    pub enable_rewrite_normalize_left: bool,
    /// Whether to prove and rewrite array reads through update chains before
    /// the lightweight prep-for-gatify rewrites.
    pub enable_formal_array_read_rewrite: bool,
    pub adder_mapping: crate::ir2gate_utils::AdderMapping,
    pub mul_adder_mapping: Option<crate::ir2gate_utils::AdderMapping>,
    pub unsafe_gatify_gate_operation: bool,
    pub aug_opt: AugOptOptions,
}

impl Ir2GatesOptions {
    /// Returns ir2gates options with all prep-for-gatify rewrites enabled.
    pub fn all_opts_enabled() -> Self {
        let prep_defaults = PrepForGatifyOptions::all_opts_enabled();
        Self {
            fold: true,
            hash: true,
            check_equivalence: false,
            equivalence_solver: SolverChoice::Bitwuzla,
            enable_rewrite_carry_out: prep_defaults.enable_rewrite_carry_out,
            enable_rewrite_prio_encode: prep_defaults.enable_rewrite_prio_encode,
            enable_rewrite_nary_add: prep_defaults.enable_rewrite_nary_add,
            enable_rewrite_mask_low: prep_defaults.enable_rewrite_mask_low,
            enable_rewrite_normalize_left: prep_defaults.enable_rewrite_normalize_left,
            enable_formal_array_read_rewrite: false,
            adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            unsafe_gatify_gate_operation: false,
            aug_opt: AugOptOptions::default(),
        }
    }

    /// Returns ir2gates options with all prep-for-gatify rewrites disabled.
    pub fn all_opts_disabled() -> Self {
        let prep_defaults = PrepForGatifyOptions::all_opts_disabled();
        Self {
            fold: true,
            hash: true,
            check_equivalence: false,
            equivalence_solver: SolverChoice::Bitwuzla,
            enable_rewrite_carry_out: prep_defaults.enable_rewrite_carry_out,
            enable_rewrite_prio_encode: prep_defaults.enable_rewrite_prio_encode,
            enable_rewrite_nary_add: prep_defaults.enable_rewrite_nary_add,
            enable_rewrite_mask_low: prep_defaults.enable_rewrite_mask_low,
            enable_rewrite_normalize_left: prep_defaults.enable_rewrite_normalize_left,
            enable_formal_array_read_rewrite: false,
            adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            unsafe_gatify_gate_operation: false,
            aug_opt: AugOptOptions::default(),
        }
    }
}

/// Applies the optional solver-backed portion of prep-for-gatify.
fn apply_formal_array_read_rewrite(
    package: &mut ir::Package,
    top_fn_name: &str,
    enabled: bool,
) -> Result<(), String> {
    if !enabled {
        return Ok(());
    }

    #[cfg(feature = "has-bitwuzla")]
    {
        use xlsynth_prover::array_access::prove_and_rewrite_array_reads;
        use xlsynth_prover::solver::bitwuzla::{Bitwuzla, BitwuzlaOptions, BitwuzlaSatSolver};

        let original = package
            .get_fn(top_fn_name)
            .ok_or_else(|| format!("PIR package has no function named '{top_fn_name}'"))?
            .clone();
        let mut solver_options = BitwuzlaOptions::new();
        solver_options
            .disable_produce_models()
            .set_sat_solver(BitwuzlaSatSolver::CaDiCaL)
            .set_nthreads(1);
        let outcome =
            prove_and_rewrite_array_reads::<Bitwuzla>(&solver_options, package, &original)?;
        log::info!(
            "formal array-read prep analysis: {:?}; rewrite: {:?}",
            outcome.analysis.stats,
            outcome.rewrite.stats
        );
        for member in &mut package.members {
            match member {
                ir::PackageMember::Function(function) if function.name == top_fn_name => {
                    *function = outcome.rewrite.rewritten_fn.clone();
                    return Ok(());
                }
                ir::PackageMember::Block { func, .. } if func.name == top_fn_name => {
                    *func = outcome.rewrite.rewritten_fn.clone();
                    return Ok(());
                }
                _ => {
                    // Only the selected top function participates in
                    // gatification.
                }
            }
        }
        Err(format!(
            "PIR package has no function or block named '{top_fn_name}'"
        ))
    }

    #[cfg(not(feature = "has-bitwuzla"))]
    {
        let _ = package;
        let _ = top_fn_name;
        Err("formal array-read prep requires a build with Bitwuzla enabled".to_string())
    }
}

impl Default for Ir2GatesOptions {
    fn default() -> Self {
        Self::all_opts_enabled()
    }
}

pub struct Ir2GatesOutput {
    pub pir_package: ir::Package,
    pub top_fn_name: String,
    pub gatify_output: ir2gate::GatifyOutput,
    pub range_info: std::sync::Arc<IrRangeInfo>,
}

impl Ir2GatesOutput {
    pub fn pir_top_fn(&self) -> &ir::Fn {
        self.pir_package
            .get_fn(&self.top_fn_name)
            .expect("top_fn_name should be present in pir_package")
    }
}

/// PIR package plus analysis data needed immediately before gatification.
pub struct PreparedIrForGatify {
    pub pir_package: ir::Package,
    pub top_fn_name: String,
    pub range_info: std::sync::Arc<IrRangeInfo>,
}

impl PreparedIrForGatify {
    pub fn pir_top_fn(&self) -> &ir::Fn {
        self.pir_package
            .get_fn(&self.top_fn_name)
            .expect("top_fn_name should be present in pir_package")
    }
}

/// Parses IR and builds the PIR/range-analysis state needed by
/// `prep_for_gatify`.
pub fn prepare_ir_for_gatify_from_ir_text(
    ir_text: &str,
    top: Option<&str>,
    options: &Ir2GatesOptions,
) -> Result<PreparedIrForGatify, String> {
    let mut ir_text_for_processing: String = ir_text.to_string();

    // Parse with PIR for lowering.
    let mut parser = ir_parser::Parser::new(&ir_text_for_processing);
    let mut pir_package = parser
        .parse_and_validate_package()
        .map_err(|e| format!("PIR parse/validate failed: {e}"))?;

    let top_fn_name: String = match top {
        Some(name) => name.to_string(),
        None => {
            let pir_top = pir_package
                .get_top_fn()
                .ok_or_else(|| "PIR package has no top function".to_string())?;
            pir_top.name.clone()
        }
    };

    if options.aug_opt.enable {
        ir_text_for_processing = xlsynth_pir::aug_opt::run_aug_opt_over_ir_text(
            &ir_text_for_processing,
            Some(&top_fn_name),
            options.aug_opt,
        )?;
        let mut parser = ir_parser::Parser::new(&ir_text_for_processing);
        pir_package = parser
            .parse_and_validate_package()
            .map_err(|e| format!("PIR parse/validate failed after aug_opt: {e}"))?;
    }

    apply_formal_array_read_rewrite(
        &mut pir_package,
        &top_fn_name,
        options.enable_formal_array_read_rewrite,
    )?;

    let pir_fn = pir_package
        .get_fn(&top_fn_name)
        .ok_or_else(|| format!("PIR package has no function named '{top_fn_name}'"))?;

    // Parse with xlsynth for libxls analysis.
    let lowered_ir_text = desugar_extensions::emit_package_as_xls_ir_text(&pir_package)
        .map_err(|e| format!("emit_package_as_xls_ir_text failed: {e}"))?;
    let mut xlsynth_package = xlsynth::IrPackage::parse_ir(&lowered_ir_text, None)
        .map_err(|e| format!("xlsynth parse_ir failed: {e}"))?;
    xlsynth_package
        .set_top_by_name(&top_fn_name)
        .map_err(|e| format!("xlsynth set_top_by_name('{top_fn_name}') failed: {e}"))?;
    let analysis = xlsynth_package
        .create_ir_analysis()
        .map_err(|e| format!("xlsynth create_ir_analysis failed: {e}"))?;

    let range_info = IrRangeInfo::build_from_analysis(&analysis, pir_fn)
        .map_err(|e| format!("building IrRangeInfo failed: {e}"))?;

    Ok(PreparedIrForGatify {
        pir_package,
        top_fn_name,
        range_info,
    })
}

pub fn ir2gates_from_ir_text(
    ir_text: &str,
    top: Option<&str>,
    options: Ir2GatesOptions,
) -> Result<Ir2GatesOutput, String> {
    let PreparedIrForGatify {
        pir_package,
        top_fn_name,
        range_info,
    } = prepare_ir_for_gatify_from_ir_text(ir_text, top, &options)?;
    let pir_fn = pir_package
        .get_fn(&top_fn_name)
        .expect("top_fn_name should be present in pir_package");

    let gatify_output = ir2gate::gatify(
        pir_fn,
        ir2gate::GatifyOptions {
            fold: options.fold,
            hash: options.hash,
            check_equivalence: options.check_equivalence,
            equivalence_solver: options.equivalence_solver,
            adder_mapping: options.adder_mapping,
            mul_adder_mapping: options.mul_adder_mapping,
            range_info: Some(range_info.clone()),
            enable_rewrite_carry_out: options.enable_rewrite_carry_out,
            enable_rewrite_prio_encode: options.enable_rewrite_prio_encode,
            enable_rewrite_nary_add: options.enable_rewrite_nary_add,
            enable_rewrite_mask_low: options.enable_rewrite_mask_low,
            enable_rewrite_normalize_left: options.enable_rewrite_normalize_left,
            unsafe_gatify_gate_operation: options.unsafe_gatify_gate_operation,
            ..ir2gate::GatifyOptions::all_opts_disabled()
        },
    )?;

    Ok(Ir2GatesOutput {
        pir_package,
        top_fn_name,
        gatify_output,
        range_info,
    })
}

#[cfg(all(test, feature = "has-bitwuzla"))]
mod tests {
    use super::*;
    use xlsynth_pir::ir::NodePayload;

    #[test]
    fn formal_array_read_rewrite_runs_during_gatify_preparation() {
        let ir_text = r#"package sample

top fn main(a: bits[8][4] id=1, value: bits[8] id=2, read_index: bits[2] id=3) -> bits[8] {
  one: bits[2] = literal(value=1, id=4)
  write_index: bits[2] = xor(read_index, one, id=5)
  updated: bits[8][4] = array_update(a, value, indices=[write_index], id=6)
  ret read: bits[8] = array_index(updated, indices=[read_index], id=7)
}
"#;
        let mut options = Ir2GatesOptions::all_opts_disabled();
        options.enable_formal_array_read_rewrite = true;

        let prepared = prepare_ir_for_gatify_from_ir_text(ir_text, None, &options).unwrap();
        let function = prepared.pir_top_fn();
        let array_param = function
            .nodes
            .iter()
            .position(|node| node.text_id == 1)
            .unwrap();
        let read = function
            .nodes
            .iter()
            .find(|node| node.text_id == 7)
            .unwrap();

        assert!(matches!(
            &read.payload,
            NodePayload::ArrayIndex { array, .. } if array.index == array_param
        ));
    }
}
