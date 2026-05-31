// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::Path;
use std::sync::Once;
use xlsynth::DslxConvertOptions;
use xlsynth_pir::ir_fn_to_dslx::IrFnToDslxError;
use xlsynth_pir_fuzz::generate_standard_random_pir_package;
use xlsynth_prover::prover::types::EquivResult;
use xlsynth_prover::prover::{SolverChoice, prover_for_choice};

static INIT_LOGGER: Once = Once::new();

fuzz_target!(|data: &[u8]| {
    INIT_LOGGER.call_once(|| {
        let _ = env_logger::builder().is_test(true).try_init();
    });

    let pir_pkg = generate_standard_random_pir_package(data, "fuzz_ir_fn_to_dslx_roundtrip");
    let ir_in = pir_pkg.to_string();
    let top_fn = pir_pkg
        .get_top_fn()
        .expect("generated package should have a top function");

    let translated =
        match xlsynth_pir::ir_fn_to_dslx::convert_ir_package_fn_to_dslx(&ir_in, Some(&top_fn.name))
        {
            Ok(v) => v,
            Err(IrFnToDslxError::UnsupportedType(_)) | Err(IrFnToDslxError::UnsupportedNode(_)) => {
                // Early-return rationale: this target focuses on the MVP
                // translator surface; generated forms outside that advertised
                // subset are not translation soundness failures.
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
    .unwrap_or_else(|e| {
        panic!(
            "DSLX->IR conversion failed: {}\n=== INPUT_IR ===\n{}\n=== DSLX ===\n{}",
            e, ir_in, translated.dslx_text
        )
    });

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

    match prover_for_choice(SolverChoice::Toolchain, None).prove_ir_pkg_text_equiv(
        &lhs_ir_with_top,
        &rhs_ir_with_top,
        None,
    ) {
        EquivResult::Proved => {}
        EquivResult::ToolchainDisproved(msg) => panic!(
            "IR equivalence disproved: {}\n=== INPUT_IR ===\n{}\n=== DSLX ===\n{}\n=== ROUNDTRIP_IR ===\n{}",
            msg, lhs_ir_with_top, translated.dslx_text, rhs_ir_with_top
        ),
        // Early-return rationale: interruption or timeout of the external oracle
        // is not a translation-soundness failure for this sample.
        EquivResult::Inconclusive(_) => return,
        EquivResult::Error(msg) => panic!("IR equivalence tooling error: {}", msg),
        other => panic!("unexpected toolchain equivalence result: {other:?}"),
    }
});
