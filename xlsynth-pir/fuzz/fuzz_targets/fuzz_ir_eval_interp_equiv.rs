// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;

use xlsynth_autocov::{generate_ir_fn_corpus_from_ir_text, IrFnAutocovGenerateConfig};
use xlsynth_pir_fuzz::generate_upstream_eval_random_pir_package;
use xlsynth_pir::ir_eval::{eval_fn_in_package, FnEvalResult};

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

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let parsed_pkg = generate_upstream_eval_random_pir_package(data, "fuzz_pkg");
    let pkg_text = parsed_pkg.to_string();
    let fn_name = parsed_pkg
        .get_top_fn()
        .expect("generated package should have a top function")
        .name
        .clone();
    let xls_pkg = xlsynth::IrPackage::parse_ir(&pkg_text, None)
        .expect("PIR-emitted standard XLS IR should parse in libxls");
    let xls_fn = xls_pkg
        .get_function(&fn_name)
        .expect("libxls package should contain the generated top function");

    log::info!("pkg_text:\n{}", pkg_text);

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
        fn_name.as_str(),
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
