// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::Once;
use xlsynth_pir::ir_verify_parity::{categorize_pir_error, categorize_xls_error_text, ErrorCategory};

static INIT_LOGGER: Once = Once::new();

fuzz_target!(|ir_text: String| {
    INIT_LOGGER.call_once(|| {
        let _ = env_logger::builder().is_test(true).try_init();
    });

    // Early returns are expected in fuzzing harnesses; see FUZZ.md.
    if ir_text.len() > 64 * 1024 {
        return;
    }

    // First, require that PIR parses as a package; otherwise, skip the sample.
    let pkg = match xlsynth_pir::ir_parser::Parser::new(&ir_text).parse_package() {
        Ok(pkg) => pkg,
        Err(parse_err) =>{
            log::debug!("PIR parse failed: {:?}", parse_err);
            return;
        }
    };

    // PIR validate
    let pir_result: Result<(), ErrorCategory> = xlsynth_pir::ir_validate::validate_package(&pkg)
        .map_err(|e| categorize_pir_error(&e));

    // XLS must also parse if PIR parses; treat parse failures as a fuzz failure.
    let xls_pkg = match xlsynth::IrPackage::parse_ir(&ir_text, None) {
        Ok(pkg) => pkg,
        Err(e) => panic!("xlsynth parse failed for IR that PIR parses:\n{}\nerror: {}", ir_text, e),
    };
    let xls_result: Result<(), ErrorCategory> = xls_pkg
        .verify()
        .map_err(|e| categorize_xls_error_text(&e.to_string()));

    match (pir_result, xls_result) {
        (Ok(()), Ok(())) => {
            log::info!("IR verified for both PIR and XLS successfully");
        }
        (Err(pir_cat), Err(xls_cat)) => {
            log::info!("PIR category: {:?}, XLS category: {:?}", pir_cat, xls_cat);
            if pir_cat != xls_cat {
                panic!(
                    "category mismatch: PIR={:?} XLS={:?}\nIR=\n{}",
                    pir_cat, xls_cat, ir_text
                );
            }
        }
        (Ok(()), Err(_)) | (Err(_), Ok(())) => {
            panic!("verifier success/failure mismatch on IR:\n{}", ir_text);
        }
    }
});
