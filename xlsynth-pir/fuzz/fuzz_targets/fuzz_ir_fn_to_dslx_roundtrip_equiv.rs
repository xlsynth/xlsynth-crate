// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::Path;
use std::sync::Once;
use xlsynth::DslxConvertOptions;
use xlsynth_pir::ir_fn_to_dslx::IrFnToDslxError;
use xlsynth_pir::ir_fuzz::{FuzzSample, generate_ir_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::prove_equiv_via_toolchain::{
    ToolchainEquivResult, prove_ir_pkg_equiv_with_tool_dir,
};

static INIT_LOGGER: Once = Once::new();

fuzz_target!(|sample: FuzzSample| {
    INIT_LOGGER.call_once(|| {
        let _ = env_logger::builder().is_test(true).try_init();
    });

    let mut pkg =
        xlsynth::IrPackage::new("fuzz_ir_fn_to_dslx_roundtrip").expect("IrPackage::new failed");
    if generate_ir_fn(sample.ops.clone(), &mut pkg, None).is_err() {
        // Early-return rationale: this target checks IR->DSLX->IR roundtrip
        // soundness for samples that successfully build IR. Some arbitrary
        // `FuzzSample` values are rejected by IR construction itself (e.g.
        // invalid selector/case shapes for one_hot_select), which is outside
        // the translator-equivalence property under test here.
        return;
    }
    let ir_in = pkg.to_string();

    let mut pir_parser = Parser::new(&ir_in);
    let pir_pkg = pir_parser
        .parse_and_validate_package()
        .expect("generated IR failed PIR parse/validate");
    let top_fn = pir_pkg
        .get_top_fn()
        .expect("generated package should have a top function");

    let translated =
        match xlsynth_pir::ir_fn_to_dslx::convert_ir_package_fn_to_dslx(&ir_in, Some(&top_fn.name))
        {
            Ok(v) => v,
            Err(IrFnToDslxError::UnsupportedType(_)) | Err(IrFnToDslxError::UnsupportedNode(_)) => {
                // Early-return rationale: this target focuses on the MVP
                // translator surface (bits-only + currently supported ops).
                // Unsupported type/op samples are outside that scoped contract.
                return;
            }
            Err(e) => panic!("IR->DSLX translation failed: {}", e),
        };

    let dslx_roundtrip = xlsynth::convert_dslx_to_ir_text(
        &translated.dslx_text,
        Path::new("fuzz_ir_fn_to_dslx_roundtrip.x"),
        &DslxConvertOptions::default(),
    )
    .expect("DSLX->IR conversion failed");

    let mut lhs_pkg_with_top =
        xlsynth::IrPackage::parse_ir(&ir_in, None).expect("parse generated lhs IR");
    lhs_pkg_with_top
        .set_top_by_name(&top_fn.name)
        .expect("set top on generated lhs package");
    let lhs_ir_with_top = lhs_pkg_with_top.to_string();

    let mut rhs_pkg_with_top =
        xlsynth::IrPackage::parse_ir(&dslx_roundtrip.ir, None).expect("parse roundtrip rhs IR");
    let rhs_top_name =
        xlsynth::mangle_dslx_name("fuzz_ir_fn_to_dslx_roundtrip", &translated.function_name)
            .expect("mangle translated DSLX function name");
    rhs_pkg_with_top
        .set_top_by_name(&rhs_top_name)
        .expect("set top on roundtrip rhs package");
    let rhs_ir_with_top = rhs_pkg_with_top.to_string();

    let tools_dir = std::env::var("XLSYNTH_TOOLS").expect("XLSYNTH_TOOLS must be set");
    match prove_ir_pkg_equiv_with_tool_dir(&lhs_ir_with_top, &rhs_ir_with_top, None, tools_dir) {
        ToolchainEquivResult::Proved => {}
        ToolchainEquivResult::Disproved(msg) => panic!(
            "IR equivalence disproved: {}\n=== INPUT_IR ===\n{}\n=== DSLX ===\n{}\n=== ROUNDTRIP_IR ===\n{}",
            msg, lhs_ir_with_top, translated.dslx_text, rhs_ir_with_top
        ),
        ToolchainEquivResult::Error(msg) => panic!("IR equivalence tooling error: {}", msg),
    }
});
