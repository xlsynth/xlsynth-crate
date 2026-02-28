// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;

use xlsynth_autocov::{generate_ir_fn_corpus_from_ir_text, IrFnAutocovGenerateConfig};
use xlsynth_pir::ir_eval::{eval_fn_in_package, FnEvalResult};
use xlsynth_pir::ir_fuzz::{generate_ir_fn, FuzzSample};
use xlsynth_pir::ir_parser::Parser;

const AUTOCOV_MAX_ITERS: u64 = 256;
const AUTOCOV_MAX_CORPUS_LEN: usize = 64;
const AUTOCOV_TWO_HOT_MAX_BITS: usize = 64;

fn stable_hash_u64(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fuzz_target!(|sample: FuzzSample| {
    // Empty op lists cannot form a function body, so they do not exercise the
    // interpreter parity property.
    if sample.ops.is_empty() {
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    // Generate an XLS IR function via C++ bindings.
    let mut pkg = xlsynth::IrPackage::new("fuzz_pkg").expect("IrPackage::new should not fail");
    let xls_fn = match generate_ir_fn(sample.ops.clone(), &mut pkg, None) {
        Ok(f) => f,
        Err(_) => {
            // The generator can intentionally reject unsupported op/type
            // combinations; those are not actionable parity failures.
            return;
        }
    };
    let pkg_text = pkg.to_string();
    let fn_name = xls_fn.get_name();

    log::info!("pkg_text:\n{}", pkg_text);

    let parsed_pkg = Parser::new(&pkg_text)
        .parse_and_validate_package()
        .expect("parse_and_validate_package should succeed for generated IR");
    let parsed_fn = match parsed_pkg.get_fn(&fn_name) {
        Some(f) => f,
        None => panic!("missing function {fn_name} in parsed package"),
    };

    // Nullary functions do not benefit from autocov-driven input exploration and
    // are covered by simpler direct tests elsewhere.
    if parsed_fn.params.is_empty() {
        return;
    }

    let corpus_result = generate_ir_fn_corpus_from_ir_text(
        &pkg_text,
        &fn_name,
        IrFnAutocovGenerateConfig {
            seed: stable_hash_u64(&pkg_text),
            max_iters: Some(AUTOCOV_MAX_ITERS),
            max_corpus_len: Some(AUTOCOV_MAX_CORPUS_LEN),
            progress_every: None,
            threads: Some(1),
            seed_structured: true,
            seed_two_hot_max_bits: AUTOCOV_TWO_HOT_MAX_BITS,
        },
    )
    .expect("autocov corpus generation should succeed for generated IR");

    assert!(
        !corpus_result.corpus.is_empty(),
        "autocov should produce at least one corpus sample"
    );

    for tuple_value in &corpus_result.corpus {
        let args = tuple_value
            .get_elements()
            .expect("autocov corpus samples should be tuples");
        let ours = match eval_fn_in_package(&parsed_pkg, parsed_fn, &args) {
            FnEvalResult::Success(success) => success.value,
            other => panic!("expected PIR evaluator success, got {:?}", other),
        };
        let theirs = xls_fn
            .interpret(&args)
            .expect("xlsynth interpreter should succeed on autocov corpus values");
        assert_eq!(
            ours, theirs,
            "eval_fn result disagrees with xlsynth interpreter for corpus sample {tuple_value}"
        );
    }
});
