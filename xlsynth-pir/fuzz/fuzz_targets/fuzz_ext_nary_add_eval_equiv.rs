// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use xlsynth::{IrBits, IrValue};
use xlsynth_pir::desugar_extensions::desugar_extensions_in_package;
use xlsynth_pir::ir::{
    self, ExtNaryAddArchitecture, ExtNaryAddTerm, FileTable, MemberType, Node, NodePayload,
    Package, PackageMember, Param, ParamId, Type,
};
use xlsynth_pir::ir_eval::{eval_fn_in_package, FnEvalResult};
use xlsynth_pir::ir_verify;
use xlsynth_pir::random_inputs::{
    BitValuePattern, generate_corner_irbits, generate_flat_bitvector_argument_sets_with_rng,
    generate_pattern_irbits,
};

const PACKAGE_NAME: &str = "sample";
const FUNCTION_NAME: &str = "main";
const MAX_RESULT_WIDTH: u8 = 10;
const MAX_OPERAND_COUNT: u8 = 5;
const MAX_PARAM_COUNT: u8 = 8;
const MAX_OPERAND_WIDTH: u8 = 8;
const RANDOM_TUPLE_COUNT: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExtNaryAddFnSample {
    result_width: usize,
    arch: Option<ExtNaryAddArchitecture>,
    params: Vec<ParamSample>,
    terms: Vec<ExtNaryAddTermSample>,
    value_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParamSample {
    width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExtNaryAddTermSample {
    source: OperandSourceSample,
    signed: bool,
    negated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OperandSourceSample {
    Param { param_index: usize },
    Literal { width: usize, value_bits: u16 },
}

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let sample = generate_ext_nary_add_fn_sample(data);
    let pkg = build_ext_nary_add_package(&sample);
    ir_verify::verify_package(&pkg)
        .unwrap_or_else(|e| panic!("generated ext_nary_add package should validate: {e:?}"));
    let pir_fn = pkg
        .get_top_fn()
        .expect("generated package should have a top fn");
    let ir_text = pkg.to_string();
    log::info!("generated ext_nary_add IR:\n{}", ir_text);

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

/// Generates a typed `ext_nary_add` fuzz sample from deterministic entropy
/// derived from the raw fuzz input bytes.
fn generate_ext_nary_add_fn_sample(data: &[u8]) -> ExtNaryAddFnSample {
    let mut seed = [0u8; 32];
    seed.copy_from_slice(blake3::hash(data).as_bytes());
    let mut rng = StdRng::from_seed(seed);

    let result_width = usize::from(rng.gen_range(0..=MAX_RESULT_WIDTH));
    let operand_count = usize::from(rng.gen_range(0..=MAX_OPERAND_COUNT));
    let arch = match rng.gen_range(0..=3) {
        0 => None,
        1 => Some(ExtNaryAddArchitecture::RippleCarry),
        2 => Some(ExtNaryAddArchitecture::BrentKung),
        _ => Some(ExtNaryAddArchitecture::KoggeStone),
    };

    let mut params = Vec::new();
    let mut terms = Vec::with_capacity(operand_count);
    for _ in 0..operand_count {
        let source_choice = rng.gen_range(0..10);
        let source = if source_choice < 3 {
            make_literal_operand_source(&mut rng)
        } else if source_choice == 3 && !params.is_empty() {
            OperandSourceSample::Param {
                param_index: rng.gen_range(0..params.len()),
            }
        } else if params.len() < usize::from(MAX_PARAM_COUNT) {
            let param_index = params.len();
            params.push(ParamSample {
                width: usize::from(rng.gen_range(0..=MAX_OPERAND_WIDTH)),
            });
            OperandSourceSample::Param { param_index }
        } else if !params.is_empty() {
            OperandSourceSample::Param {
                param_index: rng.gen_range(0..params.len()),
            }
        } else {
            make_literal_operand_source(&mut rng)
        };
        terms.push(ExtNaryAddTermSample {
            source,
            signed: rng.r#gen::<bool>(),
            negated: rng.r#gen::<bool>(),
        });
    }

    ExtNaryAddFnSample {
        result_width,
        arch,
        params,
        terms,
        value_seed: rng.r#gen::<u64>(),
    }
}

fn make_literal_operand_source<R: Rng>(rng: &mut R) -> OperandSourceSample {
    let width = usize::from(rng.gen_range(0..=MAX_OPERAND_WIDTH));
    OperandSourceSample::Literal {
        width,
        value_bits: rng.r#gen::<u16>() & width_mask_u16(width),
    }
}

/// Builds a generated sample as a one-function PIR package.
fn build_ext_nary_add_package(sample: &ExtNaryAddFnSample) -> Package {
    let params = sample
        .params
        .iter()
        .enumerate()
        .map(|(i, param)| Param {
            name: format!("p{i}"),
            ty: Type::Bits(param.width),
            id: ParamId::new(i + 1),
        })
        .collect::<Vec<_>>();

    let mut nodes = Vec::new();
    nodes.push(Node {
        text_id: 0,
        name: None,
        ty: Type::nil(),
        payload: NodePayload::Nil,
        pos: None,
    });
    for param in &params {
        nodes.push(Node {
            text_id: param.id.get_wrapped_id(),
            name: Some(param.name.clone()),
            ty: param.ty.clone(),
            payload: NodePayload::GetParam(param.id),
            pos: None,
        });
    }

    let mut next_text_id = params.len() + 1;
    let mut terms = Vec::with_capacity(sample.terms.len());
    for (term_index, term) in sample.terms.iter().enumerate() {
        let operand = match term.source {
            OperandSourceSample::Param { param_index } => ir::NodeRef {
                index: param_index + 1,
            },
            OperandSourceSample::Literal { width, value_bits } => {
                let literal_ref = ir::NodeRef { index: nodes.len() };
                nodes.push(Node {
                    text_id: next_text_id,
                    name: Some(format!("lit_{term_index}")),
                    ty: Type::Bits(width),
                    payload: NodePayload::Literal(make_bits_value(width, u64::from(value_bits))),
                    pos: None,
                });
                next_text_id += 1;
                literal_ref
            }
        };
        terms.push(ExtNaryAddTerm {
            operand,
            signed: term.signed,
            negated: term.negated,
        });
    }

    let ret_node_ref = ir::NodeRef { index: nodes.len() };
    nodes.push(Node {
        text_id: next_text_id,
        name: Some("r".to_string()),
        ty: Type::Bits(sample.result_width),
        payload: NodePayload::ExtNaryAdd {
            terms,
            arch: sample.arch,
        },
        pos: None,
    });

    let function = ir::Fn {
        name: FUNCTION_NAME.to_string(),
        params,
        ret_ty: Type::Bits(sample.result_width),
        nodes,
        ret_node_ref: Some(ret_node_ref),
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
    };

    Package {
        name: PACKAGE_NAME.to_string(),
        file_table: FileTable::new(),
        members: vec![PackageMember::Function(function)],
        top: Some((FUNCTION_NAME.to_string(), MemberType::Function)),
    }
}

/// Builds a deterministic, non-coverage-guided argument corpus for a sample.
fn build_ext_nary_add_eval_corpus(sample: &ExtNaryAddFnSample, ir_text: &str) -> Vec<Vec<IrValue>> {
    if sample.params.is_empty() {
        return vec![Vec::new()];
    }

    let mut corpus = Vec::new();
    let input_widths = sample
        .params
        .iter()
        .map(|param| param.width)
        .collect::<Vec<_>>();
    let zeros = sample
        .params
        .iter()
        .map(|param| {
            IrValue::from_bits(&generate_pattern_irbits(
                param.width,
                BitValuePattern::Zero,
            ))
        })
        .collect::<Vec<IrValue>>();
    for pattern in [
        BitValuePattern::Zero,
        BitValuePattern::AllOnes,
        BitValuePattern::SignedMin,
        BitValuePattern::SignedMax,
    ] {
        push_unique_tuple(
            &mut corpus,
            sample
                .params
                .iter()
                .map(|param| IrValue::from_bits(&generate_pattern_irbits(param.width, pattern)))
                .collect(),
        );
    }

    for (param_index, param) in sample.params.iter().enumerate() {
        for bits in generate_corner_irbits(param.width) {
            let mut tuple = zeros.clone();
            tuple[param_index] = IrValue::from_bits(&bits);
            push_unique_tuple(&mut corpus, tuple);
        }
    }

    let mut rng = StdRng::seed_from_u64(sample.value_seed ^ stable_hash_u64(ir_text));
    let random_sets = generate_flat_bitvector_argument_sets_with_rng(
        &mut rng,
        &input_widths,
        RANDOM_TUPLE_COUNT + 2,
    );
    for tuple in random_sets.into_iter().skip(2) {
        push_unique_tuple(
            &mut corpus,
            tuple
                .iter()
                .map(IrValue::from_bits)
                .collect::<Vec<IrValue>>(),
        );
    }

    corpus
}

fn push_unique_tuple(corpus: &mut Vec<Vec<IrValue>>, tuple: Vec<IrValue>) {
    if !corpus.contains(&tuple) {
        corpus.push(tuple);
    }
}

fn make_bits_value(width: usize, raw_value: u64) -> IrValue {
    let bits = IrBits::make_ubits(width, raw_value & width_mask_u64(width))
        .expect("masked literal bits must fit the requested width");
    IrValue::from_bits(&bits)
}

fn width_mask_u16(width: usize) -> u16 {
    if width == 0 {
        0
    } else if width >= u16::BITS as usize {
        u16::MAX
    } else {
        (1u16 << width) - 1
    }
}

fn width_mask_u64(width: usize) -> u64 {
    if width == 0 {
        0
    } else if width >= u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn stable_hash_u64(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
