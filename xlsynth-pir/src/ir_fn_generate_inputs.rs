// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::Path;

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use xlsynth::{IrBits, IrValue};

use crate::ir::{self, PackageMember, Type};
use crate::ir_parser::Parser;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatFormat {
    Fp8,
    Bf16,
    Fp32,
    Fp64,
}

impl FloatFormat {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "fp8" => Ok(Self::Fp8),
            "bf16" => Ok(Self::Bf16),
            "fp32" | "f32" => Ok(Self::Fp32),
            "fp64" | "f64" => Ok(Self::Fp64),
            other => Err(format!(
                "unsupported float format '{}'; expected one of fp8, bf16, fp32, fp64",
                other
            )),
        }
    }

    fn exponent_bits(self) -> usize {
        match self {
            Self::Fp8 => 4,
            Self::Bf16 | Self::Fp32 => 8,
            Self::Fp64 => 11,
        }
    }

    fn fraction_bits(self) -> usize {
        match self {
            Self::Fp8 => 3,
            Self::Bf16 => 7,
            Self::Fp32 => 23,
            Self::Fp64 => 52,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FloatDistribution {
    Gaussian { mean: f64, stddev: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FloatParamSpec {
    pub param_name: String,
    pub format: FloatFormat,
    pub distribution: FloatDistribution,
}

impl FloatParamSpec {
    /// Parses `name=format:gaussian(mean=...,stddev=...)`.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let (param_name, rhs) = spec
            .split_once('=')
            .ok_or_else(|| format!("float param '{}' is missing '='", spec))?;
        let param_name = param_name.trim();
        if param_name.is_empty() {
            return Err(format!(
                "float param '{}' has an empty parameter name",
                spec
            ));
        }

        let (format, distribution) = rhs
            .split_once(':')
            .ok_or_else(|| format!("float param '{}' is missing ':'", spec))?;
        let format = FloatFormat::parse(format)?;
        let distribution = parse_float_distribution(distribution)?;

        Ok(Self {
            param_name: param_name.to_string(),
            format,
            distribution,
        })
    }
}

#[derive(Debug, Clone)]
pub struct IrFnGenerateInputsConfig {
    pub count: usize,
    pub seed: u64,
    pub float_params: Vec<FloatParamSpec>,
}

/// Generates newline-ready typed argument tuples for a selected IR function.
pub fn generate_ir_fn_inputs_from_ir_path(
    ir_input_file: &Path,
    top: Option<&str>,
    config: &IrFnGenerateInputsConfig,
) -> Result<Vec<IrValue>, String> {
    let ir_text = std::fs::read_to_string(ir_input_file)
        .map_err(|e| format!("failed to read {}: {}", ir_input_file.display(), e))?;
    generate_ir_fn_inputs_from_ir_text(&ir_text, top, config)
}

/// Generates typed argument tuples for a selected IR function.
pub fn generate_ir_fn_inputs_from_ir_text(
    ir_text: &str,
    top: Option<&str>,
    config: &IrFnGenerateInputsConfig,
) -> Result<Vec<IrValue>, String> {
    let mut parser = Parser::new(ir_text);
    let mut package = parser
        .parse_and_validate_package()
        .map_err(|e| format!("failed to parse/validate IR package: {}", e))?;
    let function = select_function(&mut package, top)?.clone();
    let float_params = validate_float_params(&function, &config.float_params)?;
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut tuples = Vec::with_capacity(config.count);
    for _ in 0..config.count {
        let args = function
            .params
            .iter()
            .map(|param| {
                if let Some(spec) = float_params.get(&param.name) {
                    random_float_value(spec, &mut rng)
                } else {
                    random_value_for_type(&param.ty, &mut rng)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        tuples.push(IrValue::make_tuple(&args));
    }
    Ok(tuples)
}

fn select_function<'a>(
    package: &'a mut ir::Package,
    top: Option<&str>,
) -> Result<&'a ir::Fn, String> {
    if let Some(top) = top {
        package
            .set_top_fn(top)
            .map_err(|e| format!("failed to set --top: {}", e))?;
    }

    if package.top.is_some() {
        return package
            .get_top_fn()
            .ok_or_else(|| "package top is not a function; provide --top".to_string());
    }

    let functions = package
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Function(function) => Some(function),
            PackageMember::Block { .. } => None,
        })
        .collect::<Vec<_>>();
    match functions.as_slice() {
        [function] => Ok(function),
        [] => Err("package has no functions".to_string()),
        _ => Err("package has no top function; provide --top".to_string()),
    }
}

fn validate_float_params(
    function: &ir::Fn,
    float_params: &[FloatParamSpec],
) -> Result<BTreeMap<String, FloatParamSpec>, String> {
    let mut specs = BTreeMap::new();
    for spec in float_params {
        let param = function
            .params
            .iter()
            .find(|param| param.name == spec.param_name)
            .ok_or_else(|| {
                format!(
                    "float param '{}' is not a parameter of function '{}'",
                    spec.param_name, function.name
                )
            })?;
        validate_float_param_type(&param.ty, spec)?;
        if specs
            .insert(spec.param_name.clone(), spec.clone())
            .is_some()
        {
            return Err(format!(
                "float param '{}' was specified more than once",
                spec.param_name
            ));
        }
    }
    Ok(specs)
}

fn validate_float_param_type(ty: &Type, spec: &FloatParamSpec) -> Result<(), String> {
    let expected = Type::Tuple(vec![
        Box::new(Type::Bits(1)),
        Box::new(Type::Bits(spec.format.exponent_bits())),
        Box::new(Type::Bits(spec.format.fraction_bits())),
    ]);
    if *ty != expected {
        return Err(format!(
            "float param '{}' uses {:?}, but parameter type is {}; expected {}",
            spec.param_name, spec.format, ty, expected
        ));
    }
    Ok(())
}

fn parse_float_distribution(s: &str) -> Result<FloatDistribution, String> {
    let s = s.trim();
    let args = s
        .strip_prefix("gaussian(")
        .and_then(|rest| rest.strip_suffix(')'))
        .ok_or_else(|| {
            format!(
                "unsupported float distribution '{}'; expected gaussian(mean=...,stddev=...)",
                s
            )
        })?;

    let mut mean = None;
    let mut stddev = None;
    for arg in args.split(',') {
        let (name, value) = arg
            .split_once('=')
            .ok_or_else(|| format!("invalid gaussian argument '{}'", arg.trim()))?;
        let parsed = value
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("invalid gaussian value '{}': {}", value.trim(), e))?;
        if !parsed.is_finite() {
            return Err(format!(
                "gaussian argument '{}' must be finite",
                name.trim()
            ));
        }
        match name.trim() {
            "mean" => {
                if mean.replace(parsed).is_some() {
                    return Err("gaussian mean specified more than once".to_string());
                }
            }
            "stddev" => {
                if parsed < 0.0 {
                    return Err("gaussian stddev must be non-negative".to_string());
                }
                if stddev.replace(parsed).is_some() {
                    return Err("gaussian stddev specified more than once".to_string());
                }
            }
            other => return Err(format!("unknown gaussian argument '{}'", other)),
        }
    }

    Ok(FloatDistribution::Gaussian {
        mean: mean.ok_or_else(|| "gaussian is missing mean".to_string())?,
        stddev: stddev.ok_or_else(|| "gaussian is missing stddev".to_string())?,
    })
}

fn random_value_for_type(ty: &Type, rng: &mut StdRng) -> Result<IrValue, String> {
    match ty {
        Type::Token => Ok(IrValue::make_token()),
        Type::Bits(width) => Ok(IrValue::from_bits(&random_bits(*width, rng))),
        Type::Tuple(members) => {
            let values = members
                .iter()
                .map(|member| random_value_for_type(member, rng))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(IrValue::make_tuple(&values))
        }
        Type::Array(array) => {
            let values = (0..array.element_count)
                .map(|_| random_value_for_type(&array.element_type, rng))
                .collect::<Result<Vec<_>, _>>()?;
            IrValue::make_array(&values).map_err(|e| e.to_string())
        }
    }
}

fn random_bits(width: usize, rng: &mut StdRng) -> IrBits {
    let mut bits = Vec::with_capacity(width);
    while bits.len() < width {
        let word = rng.next_u64();
        let remaining = width - bits.len();
        let take = remaining.min(u64::BITS as usize);
        for bit_index in 0..take {
            bits.push(((word >> bit_index) & 1) != 0);
        }
    }
    IrBits::from_lsb_is_0(&bits)
}

fn random_float_value(spec: &FloatParamSpec, rng: &mut StdRng) -> Result<IrValue, String> {
    let sample = match spec.distribution {
        FloatDistribution::Gaussian { mean, stddev } => mean + stddev * sample_standard_normal(rng),
    };
    let bits = match spec.format {
        FloatFormat::Fp8 => encode_apfloat_bits(sample, 4, 3),
        FloatFormat::Bf16 => encode_bf16_bits(sample as f32) as u64,
        FloatFormat::Fp32 => (sample as f32).to_bits() as u64,
        FloatFormat::Fp64 => sample.to_bits(),
    };
    float_bits_to_ir_value(spec.format, bits)
}

fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = open_unit_interval(rng);
    let u2 = open_unit_interval(rng);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn open_unit_interval(rng: &mut StdRng) -> f64 {
    let word = rng.next_u64();
    ((word as f64) + 1.0) / ((u64::MAX as f64) + 2.0)
}

fn encode_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn encode_apfloat_bits(value: f64, exponent_bits: usize, fraction_bits: usize) -> u64 {
    debug_assert!(exponent_bits > 0);
    debug_assert!(fraction_bits < 64);
    let sign = if value.is_sign_negative() { 1u64 } else { 0u64 };
    let sign_shift = exponent_bits + fraction_bits;
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let fraction_mask = (1u64 << fraction_bits) - 1;
    let sign_field = sign << sign_shift;

    if value.is_nan() {
        let fraction = if fraction_bits == 0 {
            0
        } else {
            1u64 << (fraction_bits - 1)
        };
        return sign_field | (exponent_mask << fraction_bits) | fraction;
    }
    if value.is_infinite() {
        return sign_field | (exponent_mask << fraction_bits);
    }
    if value == 0.0 {
        return sign_field;
    }

    let abs = value.abs();
    let bias = (1i32 << (exponent_bits - 1)) - 1;
    let min_normal_exponent = 1 - bias;
    let mut exponent = abs.log2().floor() as i32;
    let (biased_exponent, fraction) = if exponent < min_normal_exponent {
        let scaled = abs / 2f64.powi(min_normal_exponent - fraction_bits as i32);
        let fraction = round_ties_even_to_u64(scaled);
        if fraction == 0 {
            return sign_field;
        }
        if fraction > fraction_mask {
            (1, 0)
        } else {
            (0, fraction)
        }
    } else {
        let mantissa = abs / 2f64.powi(exponent);
        let fraction = round_ties_even_to_u64((mantissa - 1.0) * 2f64.powi(fraction_bits as i32));
        if fraction > fraction_mask {
            exponent += 1;
            (exponent + bias, 0)
        } else {
            (exponent + bias, fraction)
        }
    };

    if biased_exponent as u64 >= exponent_mask {
        return sign_field | (exponent_mask << fraction_bits);
    }
    sign_field | ((biased_exponent as u64) << fraction_bits) | fraction
}

fn round_ties_even_to_u64(value: f64) -> u64 {
    let floor = value.floor();
    let fraction = value - floor;
    let floor_u64 = floor as u64;
    if fraction > 0.5 || (fraction == 0.5 && floor_u64 % 2 == 1) {
        floor_u64 + 1
    } else {
        floor_u64
    }
}

fn float_bits_to_ir_value(format: FloatFormat, bits: u64) -> Result<IrValue, String> {
    let exponent_bits = format.exponent_bits();
    let fraction_bits = format.fraction_bits();
    let sign = (bits >> (exponent_bits + fraction_bits)) & 1;
    let exponent = (bits >> fraction_bits) & ((1u64 << exponent_bits) - 1);
    let fraction = bits & ((1u64 << fraction_bits) - 1);
    Ok(IrValue::make_tuple(&[
        IrValue::make_ubits(1, sign).map_err(|e| e.to_string())?,
        IrValue::make_ubits(exponent_bits, exponent).map_err(|e| e.to_string())?,
        IrValue::make_ubits(fraction_bits, fraction).map_err(|e| e.to_string())?,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_IR: &str = r#"package sample

top fn main(x: bits[8] id=1, y: (bits[1], bits[8], bits[23]) id=2) -> bits[8] {
  ret identity.3: bits[8] = identity(x, id=3)
}
"#;
    const FLOAT_FORMATS_IR: &str = r#"package float_formats

top fn main(
    fp8: (bits[1], bits[4], bits[3]) id=1,
    bf16: (bits[1], bits[8], bits[7]) id=2,
    fp32: (bits[1], bits[8], bits[23]) id=3,
    fp64: (bits[1], bits[11], bits[52]) id=4
) -> bits[1] {
  ret literal.5: bits[1] = literal(value=0, id=5)
}
"#;

    #[test]
    fn parse_float_param_spec() {
        let spec = FloatParamSpec::parse("y=fp32:gaussian(mean=3.2,stddev=3.0)").unwrap();
        assert_eq!(spec.param_name, "y");
        assert_eq!(spec.format, FloatFormat::Fp32);
        assert_eq!(
            spec.distribution,
            FloatDistribution::Gaussian {
                mean: 3.2,
                stddev: 3.0
            }
        );
    }

    #[test]
    fn generate_inputs_is_seed_deterministic() {
        let config = IrFnGenerateInputsConfig {
            count: 3,
            seed: 7,
            float_params: vec![],
        };
        let first = generate_ir_fn_inputs_from_ir_text(SAMPLE_IR, None, &config).unwrap();
        let second = generate_ir_fn_inputs_from_ir_text(SAMPLE_IR, None, &config).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn gaussian_fp32_overrides_parameter_shape() {
        let config = IrFnGenerateInputsConfig {
            count: 1,
            seed: 0,
            float_params: vec![
                FloatParamSpec::parse("y=fp32:gaussian(mean=1.0,stddev=0.0)").unwrap(),
            ],
        };
        let tuples = generate_ir_fn_inputs_from_ir_text(SAMPLE_IR, None, &config).unwrap();
        assert_eq!(
            tuples[0].to_string(),
            "(bits[8]:127, (bits[1]:0, bits[8]:127, bits[23]:0))"
        );
    }

    #[test]
    fn float_override_requires_matching_shape() {
        let config = IrFnGenerateInputsConfig {
            count: 1,
            seed: 0,
            float_params: vec![
                FloatParamSpec::parse("x=fp32:gaussian(mean=1.0,stddev=0.0)").unwrap(),
            ],
        };
        let error = generate_ir_fn_inputs_from_ir_text(SAMPLE_IR, None, &config).unwrap_err();
        assert!(error.contains("expected (bits[1], bits[8], bits[23])"));
    }

    #[test]
    fn gaussian_float_overrides_cover_supported_formats() {
        let config = IrFnGenerateInputsConfig {
            count: 1,
            seed: 0,
            float_params: vec![
                FloatParamSpec::parse("fp8=fp8:gaussian(mean=1.0,stddev=0.0)").unwrap(),
                FloatParamSpec::parse("bf16=bf16:gaussian(mean=1.0,stddev=0.0)").unwrap(),
                FloatParamSpec::parse("fp32=fp32:gaussian(mean=1.0,stddev=0.0)").unwrap(),
                FloatParamSpec::parse("fp64=fp64:gaussian(mean=1.0,stddev=0.0)").unwrap(),
            ],
        };
        let tuples = generate_ir_fn_inputs_from_ir_text(FLOAT_FORMATS_IR, None, &config).unwrap();
        assert_eq!(
            tuples[0].to_string(),
            "((bits[1]:0, bits[4]:7, bits[3]:0), (bits[1]:0, bits[8]:127, bits[7]:0), (bits[1]:0, bits[8]:127, bits[23]:0), (bits[1]:0, bits[11]:1023, bits[52]:0))"
        );
    }
}
