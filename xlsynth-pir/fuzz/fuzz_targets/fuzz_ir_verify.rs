// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_pir::ir_verify_parity::{categorize_pir_error, categorize_xls_error_text, ErrorCategory};

fuzz_target!(|data: &[u8]| {
    // Early returns are expected in fuzzing harnesses; see FUZZ.md.
    if data.len() > 64 * 1024 {
        return;
    }
    let Ok(ir_text) = std::str::from_utf8(data) else { return; };

    // PIR parse+validate
    let pir_result: Result<(), ErrorCategory> = (|| {
        let mut p = xlsynth_pir::ir_parser::Parser::new(ir_text);
        let pkg = p.parse_package().map_err(|e| categorize_xls_error_text(&e.to_string()))?;
        xlsynth_pir::ir_validate::validate_package(&pkg)
            .map_err(|e| categorize_pir_error(&e))
    })();

    // XLS parse+verify
    let xls_result: Result<(), ErrorCategory> = match xlsynth::IrPackage::parse_ir(ir_text, None) {
        Ok(pkg) => pkg
            .verify()
            .map_err(|e| categorize_xls_error_text(&e.to_string())),
        Err(e) => Err(categorize_xls_error_text(&e.to_string())),
    };

    match (pir_result, xls_result) {
        (Ok(()), Ok(())) => {}
        (Err(pir_cat), Err(xls_cat)) => {
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
