// SPDX-License-Identifier: Apache-2.0

//! Map wrappers and source functions can share an IR specialization name.

use super::ContractCase;
use crate::SourceModule;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrValue, ir};

type ScalarOp = fn(&mut FnBuilder, BValue) -> Result<BValue, BuilderError>;

/// A builtin's DSLX spelling, concrete result width and independent scalar PIR
/// operation. The oracle never creates a map or resolves a function by name.
struct Builtin {
    name: &'static str,
    result_width: usize,
    scalar: ScalarOp,
}

/// The source function signature and its concrete specialization arguments.
struct UserSpecialization {
    parameters: &'static str,
    result: &'static str,
    arguments: String,
}

impl Builtin {
    fn choose(selector: u8, width: usize) -> Self {
        let (name, result_width, scalar): (&str, usize, ScalarOp) = match selector % 7 {
            0 => ("rev", width, FnBuilder::rev),
            1 => ("clz", width, FnBuilder::clz),
            2 => ("ctz", width, FnBuilder::ctz),
            3 => ("and_reduce", 1, FnBuilder::and_reduce),
            4 => ("or_reduce", 1, FnBuilder::or_reduce),
            5 => ("xor_reduce", 1, FnBuilder::xor_reduce),
            _ => (
                "encode",
                width.next_power_of_two().ilog2() as usize,
                FnBuilder::encode,
            ),
        };
        Self {
            name,
            result_width,
            scalar,
        }
    }

    /// Makes the source specialization parameters concrete in the caller.
    fn user_specialization(&self, width: usize) -> UserSpecialization {
        if self.name == "encode" {
            // encode specializes on its output and input dimensions, in that
            // order.
            UserSpecialization {
                parameters: "M: u32, N: u32",
                result: "uN[M]",
                arguments: format!("u32:{}, u32:{width}", self.result_width),
            }
        } else {
            UserSpecialization {
                parameters: "N: u32",
                result: if self.result_width == 1 {
                    "u1"
                } else {
                    "uN[N]"
                },
                arguments: format!("u32:{width}"),
            }
        }
    }

    /// Observes the source function and both builtin maps separately so a
    /// common substitution cannot cancel out of the top's return value.
    fn reference(&self, width: usize, helper_value: u64) -> Result<String, BuilderError> {
        let literal = IrValue::make_ubits(self.result_width, helper_value)
            .map_err(|error| BuilderError::InvalidOperation(error.to_string()))?;
        let mut builder = FnBuilder::new("main");
        let x = builder.param("x", ir::Type::Bits(width))?;
        let user = builder.literal(literal)?;
        let complement = builder.not(x)?;
        let direct = (self.scalar)(&mut builder, x)?;
        let inverse = (self.scalar)(&mut builder, complement)?;
        let result = builder.tuple(&[user, direct, inverse])?;
        Ok(builder.build_package(result, "expected")?.to_string())
    }
}

/// Varies source naming and module boundaries around two maps of the same
/// builtin and width. Safe-name and map-only layouts exercise legitimate reuse.
pub(super) fn generate(data: &[u8]) -> Result<ContractCase, String> {
    let byte = |index: usize| data.get(index).copied().unwrap_or(0);
    let width = [3usize, 2, 4, 5, 7, 8, 9, 16][usize::from(byte(2) % 8)];
    let op = Builtin::choose(byte(0), width);
    let layout = data.get(1).copied().unwrap_or(4) % 5;
    let salt = byte(3);
    let helper_value = u64::from(salt & 1);
    let name = op.name;
    let output = op.result_width;
    let tuple_type = format!("(u{output}, u{output}, u{output})");
    let mut modules = Vec::new();
    let mut prefix = String::new();
    let mut entry_name = "main";
    let mut caller_module = "sample".to_string();
    let mut top = None;
    let source_value = match layout {
        0 | 1 => {
            let suffix = if name == "encode" {
                format!("{output}_{width}")
            } else {
                width.to_string()
            };
            let helper = if layout == 0 {
                format!("{name}__{suffix}")
            } else {
                format!("user_{name}__{suffix}")
            };
            prefix =
                format!("fn {helper}(x: u{width}) -> u{output} {{ u{output}:{helper_value} }}\n");
            format!("{helper}(x)")
        }
        2 | 3 => {
            // pkgN.opsK and pkgN_opsK both encode as pkgN_opsK in an IR name.
            let package = format!("pkg{}", salt / 16);
            let leaf = format!("ops{}", salt % 16);
            let mut flat = format!("{package}_{leaf}");
            if layout == 3 {
                flat.push_str("_user");
            }
            let UserSpecialization {
                parameters,
                result,
                arguments,
            } = op.user_specialization(width);
            modules.push(SourceModule {
                name: flat.clone(),
                text: format!(
                    "pub fn {name}<{parameters}>(x: uN[N]) -> {result} {{ u1:{helper_value} as {result} }}\n"
                ),
            });
            prefix = format!("import {flat};\n");
            caller_module = format!("{package}/{leaf}");
            entry_name = "run";
            top = Some(SourceModule {
                name: "sample".into(),
                text: format!(
                    "import {package}.{leaf};\npub fn main(x: u{width}) -> {tuple_type} {{ {leaf}::run(x) }}\n"
                ),
            });
            format!("{flat}::{name}<{arguments}>(x)")
        }
        _ => format!("u{output}:{helper_value}"),
    };
    modules.push(SourceModule {
        name: caller_module,
        text: format!(
            r#"{prefix}pub fn {entry_name}(x: u{width}) -> {tuple_type} {{
    let user = {source_value};
    let direct = map([x], {name})[u32:0];
    let inverse = map([!x], {name})[u32:0];
    (user, direct, inverse)
}}
"#
        ),
    });
    modules.extend(top);
    Ok(ContractCase {
        family: "builtin_map",
        modules,
        expected_ir: op
            .reference(width, helper_value)
            .map_err(|error| error.to_string())?,
        max_and_nodes: None,
    })
}
