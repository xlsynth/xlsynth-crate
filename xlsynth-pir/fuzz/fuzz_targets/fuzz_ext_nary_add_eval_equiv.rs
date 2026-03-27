// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_package;
use xlsynth_pir::ext_nary_add_fuzz::{
    build_ext_nary_add_eval_corpus, generate_ext_nary_add_fn_sample, render_ext_nary_add_fn_sample,
};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let sample = generate_ext_nary_add_fn_sample(data);
    let ir_text = render_ext_nary_add_fn_sample(&sample);
    log::info!("generated ext_nary_add IR:\n{}", ir_text);
    let mut parser = Parser::new(&ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .unwrap_or_else(|e| panic!("generated ext_nary_add IR should parse:\n{ir_text}\nerror: {e}"));
    let pir_fn = pkg.get_top_fn().expect("generated package should have a top fn");

    let mut desugared_pkg = pkg.clone();
    desugar_extensions_in_package(&mut desugared_pkg)
        .expect("desugaring generated ext_nary_add package should succeed");
    let desugared_fn = desugared_pkg
        .get_top_fn()
        .expect("desugared package should retain the top fn");

    for args in build_ext_nary_add_eval_corpus(&sample, &ir_text) {
        let ext_value = match eval_fn_in_package(&pkg, pir_fn, &args) {
            FnEvalResult::Success(success) => success.value,
            other => panic!(
                "ext_nary_add evaluator failed for IR:\n{ir_text}\nargs={args:?}\nresult={other:?}"
            ),
        };
        let desugared_value = match eval_fn_in_package(&desugared_pkg, desugared_fn, &args) {
            FnEvalResult::Success(success) => success.value,
            other => panic!(
                "desugared evaluator failed for IR:\n{ir_text}\nargs={args:?}\nresult={other:?}"
            ),
        };
        assert_eq!(
            ext_value, desugared_value,
            "ext_nary_add evaluator mismatch\nIR:\n{ir_text}\nargs={args:?}"
        );
    }
});
