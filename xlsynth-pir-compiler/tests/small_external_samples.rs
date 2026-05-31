// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use xlsynth::IrValue;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::{PirFunctionCompiler, ScalarLayout};

const SAMPLE_COUNT: usize = 10;

fn representative_values(layout: ScalarLayout) -> Vec<u64> {
    let mask = if layout.bit_count == 64 {
        u64::MAX
    } else {
        (1u64 << layout.bit_count) - 1
    };
    if layout.bit_count <= 12 {
        return (0..=mask).collect();
    }
    let mut values = vec![
        0,
        1,
        layout.bit_count.saturating_sub(1) as u64,
        layout.bit_count as u64,
        layout.bit_count as u64 + 1,
        mask >> 1,
        mask,
    ];
    values.retain(|value| *value <= mask);
    values.sort_unstable();
    values.dedup();
    values
}

fn push_argument_sets(
    layouts: &[ScalarLayout],
    current: &mut Vec<u64>,
    output: &mut Vec<Vec<u64>>,
) {
    if current.len() == layouts.len() {
        output.push(current.clone());
        return;
    }
    for value in representative_values(layouts[current.len()]) {
        current.push(value);
        push_argument_sets(layouts, current, output);
        current.pop();
    }
}

fn smallest_ir_samples(sample_dir: &Path) -> Vec<PathBuf> {
    let mut samples = std::fs::read_dir(sample_dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", sample_dir.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| panic!("failed to read sample directory entry: {error}"))
                .path()
        })
        .filter(|path| path.extension().is_some_and(|extension| extension == "ir"))
        .map(|path| {
            let size = std::fs::metadata(&path)
                .unwrap_or_else(|error| panic!("failed to stat {}: {error}", path.display()))
                .len();
            (size, path)
        })
        .collect::<Vec<_>>();
    samples.sort_by(|left, right| left.cmp(right));
    assert!(
        samples.len() >= SAMPLE_COUNT,
        "expected at least {SAMPLE_COUNT} IR samples in {}, got {}",
        sample_dir.display(),
        samples.len()
    );
    samples
        .into_iter()
        .take(SAMPLE_COUNT)
        .map(|(_, path)| path)
        .collect()
}

#[test]
fn compiler_matches_pir_evaluator_on_ten_smallest_external_ir_samples() {
    let Some(sample_dir) = std::env::var_os("XLSYNTH_PIR_COMPILER_SAMPLE_DIR") else {
        return;
    };
    let mut executed_cases = 0usize;

    for path in smallest_ir_samples(Path::new(&sample_dir)) {
        let ir_text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let package = Parser::new(&ir_text)
            .parse_and_validate_package()
            .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
        let function = package
            .get_top_fn()
            .unwrap_or_else(|| panic!("{} has no top function", path.display()));
        let compiler = PirFunctionCompiler::compile(function)
            .unwrap_or_else(|error| panic!("failed to compile {}: {error}", path.display()));
        let scalar_param_layouts = compiler
            .param_layouts()
            .iter()
            .map(|layout| {
                layout
                    .as_scalar()
                    .expect("selected sample function parameter should be scalar")
            })
            .collect::<Vec<_>>();
        let mut argument_sets = Vec::new();
        push_argument_sets(&scalar_param_layouts, &mut Vec::new(), &mut argument_sets);

        for integer_args in argument_sets {
            let ir_args = integer_args
                .iter()
                .zip(&scalar_param_layouts)
                .map(|(value, layout)| IrValue::make_ubits(layout.bit_count, *value).unwrap())
                .collect::<Vec<_>>();
            let expected = match eval_fn_in_package(&package, function, &ir_args) {
                FnEvalResult::Success(success) => success.value,
                FnEvalResult::Failure(_) => panic!(
                    "PIR evaluator unexpectedly failed for {} with args {:?}",
                    path.display(),
                    integer_args
                ),
            };
            let actual = compiler.run_ir_values(&ir_args).unwrap_or_else(|error| {
                panic!(
                    "compiled execution failed for {} with args {:?}: {error}",
                    path.display(),
                    integer_args
                )
            });
            assert_eq!(
                actual,
                expected,
                "compiler mismatch for {} with args {:?}",
                path.display(),
                integer_args
            );
            executed_cases += 1;
        }
    }

    assert!(executed_cases > 0);
}
