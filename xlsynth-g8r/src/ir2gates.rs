// SPDX-License-Identifier: Apache-2.0

use crate::gatify::ir2gate;
use crate::gatify::prep_for_gatify::PrepForGatifyOptions;
use xlsynth_pir::aug_opt::AugOptOptions;
use xlsynth_pir::desugar_extensions;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;
use xlsynth_pir::ir_range_info::IrRangeInfo;

pub struct Ir2GatesOptions {
    pub fold: bool,
    pub hash: bool,
    pub check_equivalence: bool,
    pub enable_rewrite_carry_out: bool,
    pub enable_rewrite_prio_encode: bool,
    pub adder_mapping: crate::ir2gate_utils::AdderMapping,
    pub mul_adder_mapping: Option<crate::ir2gate_utils::AdderMapping>,
    pub aug_opt: AugOptOptions,
}

impl Default for Ir2GatesOptions {
    fn default() -> Self {
        let prep_defaults = PrepForGatifyOptions::default();
        Self {
            fold: true,
            hash: true,
            check_equivalence: false,
            enable_rewrite_carry_out: prep_defaults.enable_rewrite_carry_out,
            enable_rewrite_prio_encode: prep_defaults.enable_rewrite_prio_encode,
            adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: AugOptOptions::default(),
        }
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

pub fn ir2gates_from_ir_text(
    ir_text: &str,
    top: Option<&str>,
    options: Ir2GatesOptions,
) -> Result<Ir2GatesOutput, String> {
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

    let gatify_output = ir2gate::gatify(
        pir_fn,
        ir2gate::GatifyOptions {
            fold: options.fold,
            hash: options.hash,
            check_equivalence: options.check_equivalence,
            adder_mapping: options.adder_mapping,
            mul_adder_mapping: options.mul_adder_mapping,
            range_info: Some(range_info.clone()),
            enable_rewrite_carry_out: options.enable_rewrite_carry_out,
            enable_rewrite_prio_encode: options.enable_rewrite_prio_encode,
        },
    )?;

    Ok(Ir2GatesOutput {
        pir_package,
        top_fn_name,
        gatify_output,
        range_info,
    })
}
