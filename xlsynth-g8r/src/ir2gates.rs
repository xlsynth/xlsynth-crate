// SPDX-License-Identifier: Apache-2.0

use crate::aig_serdes::ir2gate;
use crate::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

pub struct Ir2GatesOptions {
    pub fold: bool,
    pub hash: bool,
    pub check_equivalence: bool,
    pub adder_mapping: crate::ir2gate_utils::AdderMapping,
    pub mul_adder_mapping: Option<crate::ir2gate_utils::AdderMapping>,
}

pub struct Ir2GatesOutput {
    pub pir_package: ir::Package,
    pub top_fn_name: String,
    pub gatify_output: ir2gate::GatifyOutput,
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
    // Parse with PIR for lowering.
    let mut parser = ir_parser::Parser::new(ir_text);
    let pir_package = parser
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

    let pir_fn = pir_package
        .get_fn(&top_fn_name)
        .ok_or_else(|| format!("PIR package has no function named '{top_fn_name}'"))?;

    // Parse with xlsynth for libxls analysis.
    let mut xlsynth_package = xlsynth::IrPackage::parse_ir(ir_text, None)
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
            range_info: Some(range_info),
        },
    )?;

    Ok(Ir2GatesOutput {
        pir_package,
        top_fn_name,
        gatify_output,
    })
}
