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
        // Early-return rationale: bound resource usage; this does not suppress
        // interesting bugs because the fuzzer will minimize and explore shorter
        // variants. See FUZZ.md guidance about allowed early-returns.
        return;
    }
    if ir_text.contains('\0') {
        // Early-return rationale: interior NUL causes upstream C API to fail
        // `CString::new` with NulError. Not a property of the IR semantics.
        // We skip to avoid harness-level panics. Documented in FUZZ.md.
        return;
    }
    if ir_text.contains("ext_") {
        // Early-return rationale: this fuzz target checks verification parity
        // between PIR and upstream XLS IR. Upstream does not understand PIR
        // extension ops (e.g. `ext_carry_out`), so such samples are outside the
        // target's scope. Extension semantics are fuzzed/tested separately.
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
    let xls_result: Result<(), ErrorCategory> = match xlsynth::IrPackage::parse_ir(&ir_text, None) {
        Ok(pkg) => match pkg.verify() {
            Ok(()) => Ok(()),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("Expected token of type \"ident\"") && msg.contains("Token(\"keyword\"") {
                    // Early-return rationale: tokenizer differences (ident vs keyword)
                    // that our parser permits contextually. Not a semantic mismatch.
                    return;
                }
                Err(categorize_xls_error_text(&msg))
            }
        },
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("Expected token of type \"ident\"") && msg.contains("Token(\"keyword\"") {
                // Early-return rationale: tokenizer differences (ident vs keyword)
                // that our parser permits contextually. Not a semantic mismatch.
                return;
            }
            if msg.to_lowercase().contains("expected 'ret' in function") {
                // Map upstream parse error for missing return into our MissingReturnNode
                // category so parity can be checked at a coarse level.
                Err(ErrorCategory::MissingReturnNode)
            } else {
                panic!("xlsynth parse failed for IR that PIR parses:\n{}\nerror: {}", ir_text, msg)
            }
        }
    };

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
