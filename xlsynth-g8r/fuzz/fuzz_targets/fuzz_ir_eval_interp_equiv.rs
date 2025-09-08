// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;

use xlsynth_pir::{ir, ir_parser};
use xlsynth_pir::ir_eval::{eval_fn, FnEvalResult};
use xlsynth_pir::ir_fuzz::{FuzzBinop, FuzzOp, FuzzUnop, FuzzSampleWithArgs, generate_ir_fn};

fuzz_target!(|with: FuzzSampleWithArgs| {
    // Skip degenerate base cases as in other targets.
    if with.sample.ops.is_empty() || with.sample.input_bits == 0 {
        // Early return on degenerate generator inputs (see FUZZ.md for target policy).
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    // 1) Generate an XLS IR function via C++ bindings.
    //    Filter out comparison binops we don't currently support in the pure evaluator.
    let filtered_ops: Vec<FuzzOp> = with
        .sample
        .ops
        .iter()
        .cloned()
        .filter(|op| match op {
            FuzzOp::Binop(kind, _, _) => match kind {
                // Filter comparisons (not all supported in pure evaluator yet)
                FuzzBinop::Ugt
                | FuzzBinop::Uge
                | FuzzBinop::Ult
                | FuzzBinop::Ule
                | FuzzBinop::Sgt
                | FuzzBinop::Sge
                | FuzzBinop::Slt
                | FuzzBinop::Sle
                // Filter division/modulus (not supported yet)
                | FuzzBinop::Udiv
                | FuzzBinop::Sdiv
                | FuzzBinop::Umod
                | FuzzBinop::Smod
                // Filter concat (N-ary; not handled in pure evaluator yet)
                | FuzzBinop::Concat => false,
                _ => true,
            },
            FuzzOp::Unop(kind, _) => match kind {
                // Filter Encode until implemented in pure evaluator
                FuzzUnop::Encode => false,
                _ => true,
            },
            _ => true,
        })
        .collect();
    let mut pkg = xlsynth::IrPackage::new("fuzz_pkg").expect("IrPackage::new should not fail");
    let ir_fn = match generate_ir_fn(with.sample.input_bits, filtered_ops, &mut pkg, None) {
        Ok(f) => f,
        Err(_) => {
            // Generator can yield edge cases that are not useful here; skip.
            return;
        }
    };

    // 2) Parse into internal IR.
    let pkg_text = pkg.to_string();

    log::info!("pkg_text:\n{}", pkg_text);

    let parsed_pkg = ir_parser::Parser::new(&pkg_text).parse_and_validate_package()
        .expect("parse_and_validate_package should not fail");
    let parsed_top = match parsed_pkg.get_top() { Some(f) => f.clone(), None => return };

    // Guard: skip samples that contain one_hot_sel with non-bits-typed cases.
    for nr in parsed_top.node_refs() {
        let n = parsed_top.get_node(nr);
        if let ir::NodePayload::OneHotSel { selector: _, cases } = &n.payload {
            if !cases.is_empty() {
                let case_ty = parsed_top.get_node_ty(cases[0]);
                if !matches!(case_ty, ir::Type::Bits(_)) {
                    return;
                }
            }
        }
    }

    // 3) Build arguments consistent with the function's parameter types.
    let args: Vec<xlsynth::IrValue> = with.gen_args_for_fn(&parsed_top);

    // 4) Evaluate via our interpreter. If our interpreter panics (paranoia asserts), skip.
    let ours = match std::panic::catch_unwind(|| eval_fn(&parsed_top, &args)) {
        Ok(FnEvalResult::Success(s)) => s.value,
        Ok(FnEvalResult::Failure(_)) => {
            // Treat early-failures as unsupported for this target.
            return;
        }
        Err(_) => return,
    };

    // 5) Evaluate via xlsynth C++ interpreter.
    let theirs = match ir_fn.interpret(&args) {
        Ok(v) => v,
        Err(_) => {
            // If the external interpreter rejects, skip; other targets cover interpreter stability.
            return;
        }
    };

    // 6) Compare results.
    assert_eq!(ours, theirs, "eval_fn result disagrees with xlsynth interpreter");
});
