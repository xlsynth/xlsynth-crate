// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig_sim::{gate_sim, gate_simd};
use xlsynth_g8r::test_utils::{interesting_ir_roundtrip_cases, load_interesting_ir_roundtrip_case};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;
use xlsynth_pir::libxls_bridge::value_from_libxls;
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;
use xlsynth_pir::{IrBits, IrValue, IrValuesFile, NamedIrValueSet, parse_ir_values};

#[test]
fn native_irvals_feed_scalar_and_simd_gate_simulators() {
    for case in interesting_ir_roundtrip_cases() {
        let sample = load_interesting_ir_roundtrip_case(case);
        let f = &sample.g8r_fn;
        let values = generate_argument_sets_from_seed(f, 42, 8)
            .iter()
            .map(|args| IrValue::make_tuple(args))
            .collect();
        let text = IrValuesFile::ValueSequence(values).to_string();
        let names = f.params.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
        let samples = parse_ir_values(&text)
            .unwrap()
            .into_positional_values(&names)
            .unwrap();
        let mut batch = Vec::new();
        let mut expected = Vec::new();
        for sample in samples {
            let args = sample.get_elements().unwrap();
            let inputs = args
                .iter()
                .zip(&f.params)
                .map(|(value, param)| {
                    let mut flat = Vec::new();
                    flatten_ir_value_to_lsb0_bits_for_type(value, &param.ty, &mut flat).unwrap();
                    IrBits::from_lsb_is_0(&flat)
                })
                .collect::<Vec<_>>();
            let FnEvalResult::Success(result) = eval_fn(f, &args) else {
                panic!("native evaluation failed for {}", case.name);
            };
            let mut flat = Vec::new();
            flatten_ir_value_to_lsb0_bits_for_type(&result.value, &f.ret_ty, &mut flat).unwrap();
            expected.push(vec![IrBits::from_lsb_is_0(&flat)]);
            batch.push(inputs);
        }
        let scalar = batch
            .iter()
            .map(|inputs| gate_sim::eval(&sample.gate_fn, inputs, gate_sim::Collect::None).outputs)
            .collect::<Vec<_>>();
        let simd = gate_simd::eval_ordered_batch(&sample.gate_fn, &batch).unwrap();
        assert_eq!(scalar, expected, "scalar case={}", case.name);
        assert_eq!(simd, expected, "SIMD case={}", case.name);
    }
}

#[test]
fn native_irvals_roundtrip_and_match_libxls_for_shared_signatures() {
    for case in interesting_ir_roundtrip_cases() {
        let pkg = Parser::new(case.ir_text)
            .parse_and_validate_package()
            .unwrap();
        let f = pkg.get_top_fn().unwrap();
        let names = f.params.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
        let samples = generate_argument_sets_from_seed(&f, 42, 8)
            .iter()
            .map(|args| IrValue::make_tuple(args))
            .collect::<Vec<_>>();
        let named = samples
            .iter()
            .map(|sample| {
                let mut entries = NamedIrValueSet::from_positional_tuple(&names, sample)
                    .unwrap()
                    .into_entries();
                // Exercise name-based binding independently of source order.
                entries.reverse();
                NamedIrValueSet::new(entries).unwrap()
            })
            .collect();
        for file in [
            IrValuesFile::ValueSequence(samples.clone()),
            IrValuesFile::NamedValueSequence(named),
        ] {
            let text = file.to_string();
            let parsed = parse_ir_values(&text).unwrap();
            assert_eq!(parsed, file, "case={}", case.name);
            assert_eq!(parsed.into_positional_values(&names).unwrap(), samples);
        }
        // XLS parses individual typed values, not the native record container.
        for sample in &samples {
            let reference = xlsynth::XlsIrValue::parse_typed(&sample.to_string()).unwrap();
            assert_eq!(
                reference.to_string(),
                sample.to_string(),
                "case={}",
                case.name
            );
            assert_eq!(
                value_from_libxls(&reference, &sample.type_()).unwrap(),
                *sample
            );
        }
    }
}
