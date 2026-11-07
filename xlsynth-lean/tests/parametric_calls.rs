// SPDX-License-Identifier: Apache-2.0

#[path = "common/lean_bundle.rs"]
mod lean_bundle;
#[path = "common/parametric_cases.rs"]
mod parametric_cases;
#[path = "common/parametric_check.rs"]
mod parametric_check;

fn sample(flags: u8, values: [u16; 5]) -> parametric_cases::Case {
    let mut data = [0; 11];
    data[0] = flags;
    for (bytes, value) in data[1..].as_chunks_mut::<2>().0.iter_mut().zip(values) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    parametric_cases::generate(data)
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn concat_permutations_have_distinct_results() {
    let case = sample(16, [2, 5, 7, 11, 13]);
    let parametric_check::CheckedCase {
        bundle,
        expected: results,
    } = parametric_check::checked_bundle(&case).unwrap();
    let mut values = results[0][..6]
        .iter()
        .map(|pair| pair.0)
        .collect::<Vec<_>>();
    values.sort_unstable();
    assert_eq!(values, [0x257, 0x275, 0x527, 0x572, 0x725, 0x752]);
    assert_eq!(results[0][0], (0x257, 0x57));
    assert_eq!(results[0][0], results[0][6]);
    lean_bundle::check_bundle(&bundle).unwrap();
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn wrong_specialization_in_caller_fails_the_oracle() {
    for flags in [16, 9] {
        let case = sample(flags, [2, 5, 7, 11, 13]);
        let mut bundle = parametric_check::checked_bundle(&case).unwrap().bundle;
        lean_bundle::check_bundle(&bundle).unwrap();
        let (_, source) = bundle
            .files
            .iter_mut()
            .find(|(name, _)| name == "ReadableGenerated.lean")
            .unwrap();
        let caller_decl = "def _root_.XLS.«machine».«g»";
        let (helpers, caller) = source.split_once(caller_decl).unwrap();
        let correct = "_root_.XLS.«machine».«outer<2,5>»";
        let wrong = "_root_.XLS.«machine».«outer<5,2>»";
        assert!(helpers.contains(&format!("def {correct}")));
        assert!(helpers.contains(&format!("def {wrong}")));
        assert!(caller.contains(correct));
        // Keep both correct helper definitions and alter only g's first call.
        *source = format!(
            "{helpers}{caller_decl}{}",
            caller.replacen(correct, wrong, 1)
        );
        let error = lean_bundle::check_bundle(&bundle).unwrap_err().to_string();
        assert!(error.contains("CheckParametric.lean"), "{error}");
        assert!(error.contains("did not evaluate to `true`"), "{error}");
    }
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn square_parameter_order_and_body_order() {
    for (flags, expected) in [
        (9, (2, 65534)),
        (11, (20, 65531)),
        (13, (20, 65531)),
        (15, (2, 65534)),
    ] {
        let case = sample(flags, [2, 5, 0, 7, 0]);
        let parametric_check::CheckedCase {
            bundle,
            expected: results,
        } = parametric_check::checked_bundle(&case).unwrap();
        assert_eq!(results[0][0], expected, "flags={flags}");
        assert_eq!(results[0][0], results[0][2]);
        lean_bundle::check_bundle(&bundle).unwrap();
    }
}

#[test]
#[ignore = "requires matching XLS DSO and XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn generated_callers_cover_arities_and_binding_orders() {
    for arity in 0..3 {
        for variant in 0..8 {
            let flags = arity * 8 + variant;
            let case = sample(flags, [0xffff, 0x8000, 0x7fff, 0, 0xffff]);
            parametric_check::check_case(&case)
                .unwrap_or_else(|error| panic!("flags={flags}: {error:#}"));
        }
    }
}
