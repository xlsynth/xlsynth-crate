// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for `ext_nary_add` fuzz targets.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use xlsynth::ir_value::IrFormatPreference;
use xlsynth::{IrBits, IrValue};

use crate::fuzz_utils::arbitrary_irbits;
use crate::ir::ExtNaryAddArchitecture;

pub const EXT_NARY_ADD_FUZZ_PACKAGE_NAME: &str = "sample";
pub const EXT_NARY_ADD_FUZZ_FUNCTION_NAME: &str = "main";
const MAX_RESULT_WIDTH: u8 = 10;
const MAX_OPERAND_COUNT: u8 = 5;
const MAX_PARAM_COUNT: u8 = 8;
const MAX_OPERAND_WIDTH: u8 = 8;
const RANDOM_TUPLE_COUNT: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExtNaryAddSampleOptions {
    allow_zero_width: bool,
}

/// Randomly generated `ext_nary_add` function sample used by fuzz targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtNaryAddFnSample {
    pub result_width: usize,
    pub arch: Option<ExtNaryAddArchitecture>,
    pub params: Vec<ParamSample>,
    pub terms: Vec<ExtNaryAddTermSample>,
    pub value_seed: u64,
}

/// Parameter description for a generated sample function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSample {
    pub width: usize,
}

/// One term in a generated `ext_nary_add`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtNaryAddTermSample {
    pub source: OperandSourceSample,
    pub signed: bool,
    pub negated: bool,
}

/// Operand source used by a generated `ext_nary_add` term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandSourceSample {
    Param { param_index: usize },
    Literal { width: usize, value_bits: u16 },
}

/// Generates a typed `ext_nary_add` fuzz sample from deterministic entropy
/// derived from the raw fuzz input bytes.
pub fn generate_ext_nary_add_fn_sample(data: &[u8]) -> ExtNaryAddFnSample {
    generate_ext_nary_add_fn_sample_with_options(
        data,
        ExtNaryAddSampleOptions {
            allow_zero_width: true,
        },
    )
}

/// Generates a typed `ext_nary_add` fuzz sample with nonzero result, operand,
/// literal, and parameter widths.
pub fn generate_ext_nary_add_fn_sample_without_zero_widths(data: &[u8]) -> ExtNaryAddFnSample {
    generate_ext_nary_add_fn_sample_with_options(
        data,
        ExtNaryAddSampleOptions {
            allow_zero_width: false,
        },
    )
}

/// Generates a typed `ext_nary_add` fuzz sample from deterministic entropy
/// derived from the raw fuzz input bytes using the provided width policy.
fn generate_ext_nary_add_fn_sample_with_options(
    data: &[u8],
    options: ExtNaryAddSampleOptions,
) -> ExtNaryAddFnSample {
    let mut seed = [0u8; 32];
    seed.copy_from_slice(blake3::hash(data).as_bytes());
    let mut rng = StdRng::from_seed(seed);

    let result_width = usize::from(generate_width(
        &mut rng,
        MAX_RESULT_WIDTH,
        /* allow_zero_width= */ options.allow_zero_width,
    ));
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
            make_literal_operand_source(&mut rng, options)
        } else if source_choice == 3 && !params.is_empty() {
            OperandSourceSample::Param {
                param_index: rng.gen_range(0..params.len()),
            }
        } else {
            if params.len() < usize::from(MAX_PARAM_COUNT) {
                let param_index = params.len();
                params.push(ParamSample {
                    width: usize::from(generate_width(
                        &mut rng,
                        MAX_OPERAND_WIDTH,
                        /* allow_zero_width= */ options.allow_zero_width,
                    )),
                });
                OperandSourceSample::Param { param_index }
            } else if !params.is_empty() {
                OperandSourceSample::Param {
                    param_index: rng.gen_range(0..params.len()),
                }
            } else {
                make_literal_operand_source(&mut rng, options)
            }
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

fn make_literal_operand_source<R: Rng>(
    rng: &mut R,
    options: ExtNaryAddSampleOptions,
) -> OperandSourceSample {
    let width = usize::from(generate_width(
        rng,
        MAX_OPERAND_WIDTH,
        /* allow_zero_width= */ options.allow_zero_width,
    ));
    OperandSourceSample::Literal {
        width,
        value_bits: rng.r#gen::<u16>() & width_mask_u16(width),
    }
}

fn generate_width<R: Rng>(rng: &mut R, max_width: u8, allow_zero_width: bool) -> u8 {
    if allow_zero_width {
        rng.gen_range(0..=max_width)
    } else {
        rng.gen_range(1..=max_width)
    }
}

/// Renders a generated sample as a one-function PIR package.
pub fn render_ext_nary_add_fn_sample(sample: &ExtNaryAddFnSample) -> String {
    let params_sig = sample
        .params
        .iter()
        .enumerate()
        .map(|(i, param)| format!("p{i}: bits[{}] id={}", param.width, i + 1))
        .collect::<Vec<String>>()
        .join(", ");

    let mut next_id = sample.params.len() + 1;
    let mut body_lines = Vec::new();
    let mut operand_names = Vec::with_capacity(sample.terms.len());
    for (term_index, term) in sample.terms.iter().enumerate() {
        match term.source {
            OperandSourceSample::Param { param_index } => {
                assert!(
                    param_index < sample.params.len(),
                    "param_index {} out of bounds for {} params",
                    param_index,
                    sample.params.len()
                );
                operand_names.push(format!("p{param_index}"));
            }
            OperandSourceSample::Literal { width, value_bits } => {
                let name = format!("lit_{term_index}");
                let literal_text = format_bits_literal(width, value_bits);
                body_lines.push(format!(
                    "  {name}: bits[{width}] = literal(value={literal_text}, id={next_id})"
                ));
                next_id += 1;
                operand_names.push(name);
            }
        }
    }

    let mut attrs = Vec::new();
    if !operand_names.is_empty() {
        attrs.push(operand_names.join(", "));
    }
    attrs.push(format!(
        "signed=[{}]",
        sample
            .terms
            .iter()
            .map(|term| term.signed.to_string())
            .collect::<Vec<String>>()
            .join(", ")
    ));
    attrs.push(format!(
        "negated=[{}]",
        sample
            .terms
            .iter()
            .map(|term| term.negated.to_string())
            .collect::<Vec<String>>()
            .join(", ")
    ));
    if let Some(arch) = sample.arch {
        attrs.push(format!("arch={arch}"));
    }
    attrs.push(format!("id={next_id}"));
    body_lines.push(format!(
        "  ret r: bits[{}] = ext_nary_add({})",
        sample.result_width,
        attrs.join(", ")
    ));

    format!(
        "package {EXT_NARY_ADD_FUZZ_PACKAGE_NAME}\n\n\
top fn {EXT_NARY_ADD_FUZZ_FUNCTION_NAME}({params_sig}) -> bits[{result_width}] {{\n\
{body}\n\
}}\n",
        result_width = sample.result_width,
        body = body_lines.join("\n")
    )
}

/// Builds a deterministic, non-coverage-guided argument corpus for a sample.
pub fn build_ext_nary_add_eval_corpus(
    sample: &ExtNaryAddFnSample,
    ir_text: &str,
) -> Vec<Vec<IrValue>> {
    if sample.params.is_empty() {
        return vec![Vec::new()];
    }

    let mut corpus = Vec::new();
    let zeros = sample
        .params
        .iter()
        .map(|param| make_bits_value(param.width, 0))
        .collect::<Vec<IrValue>>();
    push_unique_tuple(&mut corpus, zeros.clone());
    push_unique_tuple(
        &mut corpus,
        sample
            .params
            .iter()
            .map(|param| make_bits_value(param.width, width_mask_u64(param.width)))
            .collect(),
    );
    push_unique_tuple(
        &mut corpus,
        sample
            .params
            .iter()
            .map(|param| make_bits_value(param.width, signed_min_value(param.width)))
            .collect(),
    );
    push_unique_tuple(
        &mut corpus,
        sample
            .params
            .iter()
            .map(|param| make_bits_value(param.width, signed_max_value(param.width)))
            .collect(),
    );

    for (param_index, param) in sample.params.iter().enumerate() {
        for value in edge_values_for_width(param.width) {
            let mut tuple = zeros.clone();
            tuple[param_index] = make_bits_value(param.width, value);
            push_unique_tuple(&mut corpus, tuple);
        }
    }

    let mut rng = StdRng::seed_from_u64(sample.value_seed ^ stable_hash_u64(ir_text));
    for _ in 0..RANDOM_TUPLE_COUNT {
        let tuple = sample
            .params
            .iter()
            .map(|param| IrValue::from_bits(&arbitrary_irbits(&mut rng, param.width)))
            .collect::<Vec<IrValue>>();
        push_unique_tuple(&mut corpus, tuple);
    }

    corpus
}

fn push_unique_tuple(corpus: &mut Vec<Vec<IrValue>>, tuple: Vec<IrValue>) {
    if !corpus.contains(&tuple) {
        corpus.push(tuple);
    }
}

fn format_bits_literal(width: usize, value_bits: u16) -> String {
    let value = make_bits_value(width, u64::from(value_bits));
    value
        .to_string_fmt_no_prefix(IrFormatPreference::Default)
        .expect("literal formatting should succeed")
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

fn signed_min_value(width: usize) -> u64 {
    if width == 0 { 0 } else { 1u64 << (width - 1) }
}

fn signed_max_value(width: usize) -> u64 {
    if width == 0 {
        0
    } else {
        (1u64 << (width - 1)) - 1
    }
}

fn alternating_pattern(width: usize, lsb_is_one: bool) -> u64 {
    let mut value = 0u64;
    for bit_index in 0..width {
        let bit_is_one = if lsb_is_one {
            bit_index % 2 == 0
        } else {
            bit_index % 2 == 1
        };
        if bit_is_one {
            value |= 1u64 << bit_index;
        }
    }
    value
}

fn edge_values_for_width(width: usize) -> Vec<u64> {
    let mut values = Vec::new();
    for candidate in [
        0,
        width_mask_u64(width),
        if width == 0 { 0 } else { 1 },
        if width == 0 { 0 } else { 1u64 << (width - 1) },
        alternating_pattern(width, /* lsb_is_one= */ true),
        alternating_pattern(width, /* lsb_is_one= */ false),
        signed_min_value(width),
        signed_max_value(width),
    ] {
        if !values.contains(&candidate) {
            values.push(candidate);
        }
    }
    values
}

fn stable_hash_u64(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
