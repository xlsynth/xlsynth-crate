// SPDX-License-Identifier: Apache-2.0

//! Constant contexts evaluated independently through native PIR semantics.

use crate::ast::{BinaryOp, BitsType, Body, Fragment, HelperModule, Width};
use crate::compound::Declarations;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_value_utils::ir_bits_to_usize;
use xlsynth_pir::{BValue, FnBuilder, IrBits, IrFormatPreference, IrValue};

/// The lexical scope of a generated constant definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstantScope {
    Module,
    Local,
}

/// An explicit constant argument or the enclosing helper's width parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstInput {
    Literal(IrBits),
    BitWidth,
}

/// A bits expression whose two inputs are known when its context is evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstExpr {
    body: Body,
    inputs: [ConstInput; 2],
}

impl ConstExpr {
    /// Checks a concrete expression template and its constant input types.
    pub fn new(body: Body, inputs: [ConstInput; 2]) -> Result<Self, String> {
        let ty = body.bits_type();
        let Width::Concrete(width) = ty.width else {
            return Err("constant expression body requires a concrete type".to_string());
        };
        if body.computed_width().is_some() {
            return Err("constant expression body cannot declare its own width".to_string());
        }
        for input in &inputs {
            match input {
                ConstInput::Literal(bits) if bits.get_bit_count() != width => {
                    return Err("constant argument width does not match its body".to_string());
                }
                ConstInput::BitWidth
                    if ty
                        != (BitsType {
                            signed: false,
                            width: Width::Concrete(32),
                        }) =>
                {
                    return Err(
                        "width parameter inputs require an unsigned 32-bit body".to_string()
                    );
                }
                _ => {
                    // Matching-width literals and unsigned width inputs are
                    // valid.
                }
            }
        }
        if body.fragments().iter().any(|fragment| {
            matches!(
                fragment,
                Fragment::Constant {
                    scope: ConstantScope::Module,
                    ..
                }
            )
        }) {
            return Err(
                "constant expression blocks cannot contain module declarations".to_string(),
            );
        }
        Ok(Self { body, inputs })
    }

    pub fn bits_type(&self) -> BitsType {
        self.body.bits_type()
    }

    /// Reports whether this expression needs an enclosing generic width.
    pub fn requires_bit_width(&self) -> bool {
        self.inputs
            .iter()
            .any(|input| matches!(input, ConstInput::BitWidth))
            || self.body.fragments().iter().any(|fragment| match fragment {
                Fragment::Constant { expression, .. } => expression.requires_bit_width(),
                _ => false, // Other fragments have no additional constant-input environment.
            })
    }

    pub(crate) fn validate_calls(&self, available: usize) -> Result<(), String> {
        self.body.validate_calls(available)
    }

    pub(crate) fn declarations(&self) -> Declarations {
        self.body.declarations()
    }

    /// Gives ordinary loops a function boundary for constexpr invocation.
    pub(crate) fn render_definition(
        &self,
        prefix: &str,
        modules: &[HelperModule],
    ) -> Result<String, String> {
        let mut text = self.body.render_const_functions(prefix, modules)?;
        if self.body.needs_const_call() {
            let ty = self.bits_type().to_dslx();
            text.push_str(&format!(
                "\nfn const_{prefix}({prefix}v0: {ty}, {prefix}v1: {ty}) -> {ty} {{\n"
            ));
            text.push_str(&self.body.render_bindings(prefix, modules)?);
            text.push_str(&format!("    {}\n}}\n", self.body.result_name(prefix)));
        }
        Ok(text)
    }

    /// Renders a block with literal arguments, retaining imported function
    /// calls.
    pub(crate) fn to_dslx(&self, prefix: &str, modules: &[HelperModule]) -> Result<String, String> {
        if self.body.needs_const_call() {
            let args = self
                .inputs
                .iter()
                .map(|input| match input {
                    ConstInput::Literal(bits) => bits_literal(self.bits_type(), bits),
                    ConstInput::BitWidth => "N".to_string(),
                })
                .collect::<Vec<_>>();
            return Ok(format!("const_{prefix}({})", args.join(", ")));
        }
        let mut result = "{\n".to_string();
        for (index, input) in self.inputs.iter().enumerate() {
            let value = match input {
                ConstInput::Literal(bits) => bits_literal(self.bits_type(), bits),
                ConstInput::BitWidth => "N".to_string(),
            };
            result.push_str(&format!("    let {prefix}v{index} = {value};\n"));
        }
        result.push_str(&self.body.render_bindings(prefix, modules)?);
        result.push_str(&format!("    {}\n}}", self.body.result_name(prefix)));
        Ok(result)
    }

    /// Instantiates the expression using PIR, with no DSLX conversion or
    /// evaluation.
    pub(crate) fn emit_ir(
        &self,
        builder: &mut FnBuilder,
        modules: &[HelperModule],
        bit_width: Option<usize>,
    ) -> Result<BValue, String> {
        let mut args = Vec::new();
        for input in &self.inputs {
            let bits = match input {
                ConstInput::Literal(bits) => bits.clone(),
                ConstInput::BitWidth => {
                    metadata_bits(bit_width.ok_or("constant expression has no enclosing width")?)?
                }
            };
            args.push(
                builder
                    .literal(IrValue::from_bits(&bits))
                    .map_err(|error| error.to_string())?,
            );
        }
        let Width::Concrete(width) = self.bits_type().width else {
            return Err("constant expression has no concrete type".to_string());
        };
        self.body
            .emit_ir(builder, [args[0], args[1]], width, modules)
            .map_err(|error| error.to_string())
    }

    /// Computes the reference value using the native PIR interpreter.
    pub fn evaluate(
        &self,
        modules: &[HelperModule],
        bit_width: Option<usize>,
    ) -> Result<IrBits, String> {
        for (index, module) in modules.iter().enumerate() {
            module.validate_calls(index)?;
        }
        self.validate_calls(modules.len())?;
        let mut builder = FnBuilder::new("constant");
        let value = self.emit_ir(&mut builder, modules, bit_width)?;
        let function = builder.build(value).map_err(|error| error.to_string())?;
        let FnEvalResult::Success(result) = eval_fn(&function, &[]) else {
            return Err("reference constant evaluation failed".to_string());
        };
        result.value.to_bits().map_err(|error| error.to_string())
    }
}

/// A checked expression defining a concrete lane's source-level type width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputedWidth {
    pub expression: Box<ConstExpr>,
    pub assert_value: bool,
}

/// A non-inferred parameter whose default is evaluated from the helper width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParametricDefault {
    pub expression: ConstExpr,
    pub combine: BinaryOp,
}

/// Encodes width/index metadata as a DSLX u32 value without scalar value APIs.
pub(crate) fn metadata_bits(value: usize) -> Result<IrBits, String> {
    let bits = IrBits::from_lsb_fn(32, |index| {
        index < usize::BITS as usize && value & (1usize << index) != 0
    });
    if ir_bits_to_usize(&bits) != Some(value) {
        return Err("metadata value does not fit the DSLX u32 type".to_string());
    }
    Ok(bits)
}

pub(crate) fn bits_literal(ty: BitsType, bits: &IrBits) -> String {
    format!(
        "{}:{}",
        ty.to_dslx(),
        bits.to_string_fmt(IrFormatPreference::Hex, false)
    )
}
