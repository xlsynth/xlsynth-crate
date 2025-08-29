// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;

use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::xls_ir::ir::{Fn as IrFn, Type};
use xlsynth_g8r::xls_ir::localized_eco;
use xlsynth_g8r::xls_ir::ir_parser;
use xlsynth_test_helpers::ir_fuzz::{FuzzSample, generate_ir_fn};

fuzz_target!(|orig: FuzzSample| {
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        panic!("XLSYNTH_TOOLS environment variable must be set for fuzzing.");
    }

    // Build an initial XLS IR package from the fuzz sample, then parse to g8r IR.
    if orig.input_bits == 0 || orig.ops.is_empty() {
        return;
    }
    let _ = env_logger::builder().is_test(true).try_init();
    let mut pkg = match xlsynth::IrPackage::new("fuzz_pkg") {
        Ok(p) => p,
        Err(_) => return,
    };
    if let Err(e) = generate_ir_fn(orig.input_bits, orig.ops.clone(), &mut pkg) {
        log::info!("generate_ir_fn failed: {}", e);
        return;
    }
    let pkg_text = pkg.to_string();
    let parsed = match ir_parser::Parser::new(&pkg_text).parse_package() {
        Ok(p) => p,
        Err(e) => {
            log::info!("parse_package failed: {}", e);
            return;
        }
    };
    let Some(old_fn) = parsed.get_top() else { return; };
    // Only operate on bit-returning functions to keep edits simple/valid.
    if !matches!(old_fn.ret_ty, Type::Bits(_)) {
        return;
    }

    let target_sample = FuzzSample::new_with_same_signature(&orig);
    let mut target_pkg = match xlsynth::IrPackage::new("fuzz_pkg") {
        Ok(p) => p,
        Err(_) => return,
    };
    if let Err(e) = generate_ir_fn(target_sample.input_bits, target_sample.ops.clone(), &mut target_pkg) {
        log::info!("generate_ir_fn failed: {}", e);
        return;
    }
    let target_pkg_text = target_pkg.to_string();
    let target_parsed = match ir_parser::Parser::new(&target_pkg_text).parse_package() {
        Ok(p) => p,
        Err(e) => {
            log::info!("parse_package (target) failed: {}", e);
            return;
        }
    };
    let Some(target_fn) = target_parsed.get_top() else { return; };

    // Compute localized ECO, apply to old to produce patched(old).
    let diff = localized_eco::compute_localized_eco(&old_fn, &target_fn);
    let patched = localized_eco::apply_localized_eco(&old_fn, &target_fn, &diff);

    // Prove patched(old) ≡ new using external checker (feature-independent).
    let lhs_pkg_text = format!("package lhs\n\ntop {}", patched.to_string());
    let rhs_pkg_text = format!("package rhs\n\ntop {}", target_fn.to_string());
    if let Err(e) = check_equivalence::check_equivalence_with_top(
        &lhs_pkg_text,
        &rhs_pkg_text,
        Some(target_fn.name.as_str()),
        false,
    ) {
        panic!("external ir-equiv failed: {}", e);
    }
});
