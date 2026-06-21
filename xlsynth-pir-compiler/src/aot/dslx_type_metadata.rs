// SPDX-License-Identifier: Apache-2.0

//! DSLX type naming helpers used while constructing native AOT metadata.

use xlsynth::{IrValue, XlsynthError, dslx, ir_value::IrFormatPreference};

/// Resolves typed DSLX annotations to the canonical generated Rust type path.
pub(crate) struct DslxTypeMetadata;

/// Runtime type vocabulary used by the PIR AOT metadata renderer.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RustTypeTarget {
    PirAot,
}

impl DslxTypeMetadata {
    /// Returns the generated Rust type name for a DSLX type in one module.
    ///
    /// Typed callers use this path so imported type references stay canonical
    /// while bits-like values map to the native runtime's `UBits` and `SBits`
    /// runtime types.
    pub(crate) fn rust_type_name_from_dslx_module(
        current_module_name: &str,
        type_info: &dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        let module_path = rust_module_path_from_dslx_module_name(current_module_name);
        Self::convert_type_with_annotation(
            RustTypeTarget::PirAot,
            &module_path,
            Some(type_info),
            type_annotation,
            ty,
        )
    }

    fn convert_type_with_annotation(
        target: RustTypeTarget,
        current_module_path: &[String],
        type_info: Option<&dslx::TypeInfo>,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        if let Some(type_annotation) = type_annotation {
            if let Some(array_annotation) = type_annotation.to_array_type_annotation()
                && ty.is_array()
            {
                let element_annotation = array_annotation.get_element_type();
                let element_ty = ty.get_array_element_type();
                let rust_ty = Self::convert_type_with_annotation(
                    target,
                    current_module_path,
                    type_info,
                    Some(&element_annotation),
                    &element_ty,
                )?;
                return Ok(format!("[{rust_ty}; {}]", ty.get_array_size()));
            }
            if let Some(type_ref_annotation) = type_annotation.to_type_ref_type_annotation()
                && let Some(rust_ty) = Self::convert_type_ref_annotation(
                    current_module_path,
                    type_info,
                    &type_ref_annotation,
                    ty,
                )?
            {
                return Ok(rust_ty);
            }
        }
        if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            Ok(bits_rust_type(target, is_signed, bit_count))
        } else if ty.is_enum() {
            let enum_def = ty.get_enum_def()?;
            Ok(enum_def.get_identifier().to_string())
        } else if ty.is_struct() {
            let struct_def = ty.get_struct_def()?;
            Ok(struct_def.get_identifier().to_string())
        } else if ty.is_array() {
            let array_ty = ty.get_array_element_type();
            let array_size = ty.get_array_size();
            let rust_ty = Self::convert_type_with_annotation(
                target,
                current_module_path,
                type_info,
                None,
                &array_ty,
            )?;
            Ok(format!("[{rust_ty}; {array_size}]"))
        } else {
            let ty_text = ty.to_string()?;
            if ty_text.trim_start().starts_with('(') {
                return Ok(parse_concrete_dslx_type_shape(&ty_text)?.rust_type(target));
            }
            Err(XlsynthError(format!(
                "Unsupported type for conversion from DSLX to Rust: {:?}",
                ty_text
            )))
        }
    }

    /// Converts a typed DSLX type-reference annotation into a generated Rust
    /// type path.
    ///
    /// Non-reference annotations return `None` so callers can continue with
    /// concrete structural conversion.
    fn convert_type_ref_annotation(
        current_module_path: &[String],
        type_info: Option<&dslx::TypeInfo>,
        type_ref_annotation: &dslx::TypeRefTypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<Option<String>, XlsynthError> {
        let type_ref = type_ref_annotation.get_type_ref();
        let type_definition = type_ref.get_type_definition();
        let suffix = Self::parametric_type_suffix(type_info, type_ref_annotation, ty)?;
        if let Some(colon_ref) = type_definition.to_colon_ref() {
            let attr = format!("{}{}", colon_ref.get_attr(), suffix);
            if let Some(import) = colon_ref.resolve_import_subject() {
                let module_path = rust_module_path_from_import(&import);
                Ok(Some(rust_type_path_between_module_paths(
                    current_module_path,
                    &module_path,
                    &attr,
                )))
            } else {
                Ok(Some(attr))
            }
        } else {
            Ok(type_definition
                .to_type_alias()
                .map(|alias| alias.get_identifier())
                .or_else(|| {
                    if ty.is_struct() {
                        ty.get_struct_def()
                            .ok()
                            .map(|struct_def| format!("{}{}", struct_def.get_identifier(), suffix))
                    } else if ty.is_enum() {
                        ty.get_enum_def()
                            .ok()
                            .map(|enum_def| enum_def.get_identifier())
                    } else {
                        None
                    }
                }))
        }
    }

    fn parametric_type_suffix(
        type_info: Option<&dslx::TypeInfo>,
        type_ref_annotation: &dslx::TypeRefTypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        let parametric_count = type_ref_annotation.get_parametric_count();
        if parametric_count == 0 {
            return Ok(String::new());
        }
        let type_info = type_info.ok_or_else(|| {
            XlsynthError(
                "Generated Rust types cannot name a parametric type without a TypeInfo context"
                    .to_string(),
            )
        })?;
        if !ty.is_struct() {
            return Err(XlsynthError(format!(
                "Generated Rust types only support parametric struct type references, got `{}`",
                ty.to_string()?
            )));
        }
        let struct_def = ty.get_struct_def()?;
        let binding_count = struct_def.get_parametric_binding_count();
        if binding_count != parametric_count {
            return Err(XlsynthError(format!(
                "Generated Rust type parametric mismatch for `{}`: struct has {binding_count} binding(s), annotation has {parametric_count} argument(s)",
                struct_def.get_identifier()
            )));
        }
        let mut parts = Vec::with_capacity(parametric_count);
        for index in 0..parametric_count {
            let binding = struct_def.get_parametric_binding(index);
            let expr = type_ref_annotation.get_parametric_expr(index).ok_or_else(|| {
                XlsynthError(format!(
                    "Generated Rust types do not support type-valued parametric argument {index} for `{}`",
                    struct_def.get_identifier()
                ))
            })?;
            let is_signed = typed_literal_text_is_signed(&expr.to_text());
            let value = const_parametric_expr_value(type_info, &expr)?;
            parts.push(format!(
                "{}_{}",
                sanitize_type_parametric_segment(&binding.get_identifier()),
                rust_type_parametric_value_suffix(&value, is_signed)?
            ));
        }
        Ok(format!("__{}", parts.join("__")))
    }
}

/// Returns the smallest runtime-neutral Rust type that can hold one DSLX bits
/// value.
fn bits_rust_type(target: RustTypeTarget, is_signed: bool, bit_count: usize) -> String {
    match target {
        RustTypeTarget::PirAot => {
            if is_signed {
                format!("SBits<{bit_count}>")
            } else {
                format!("UBits<{bit_count}>")
            }
        }
    }
}

/// Concrete structural DSLX type spellings that currently lack direct FFI
/// accessors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ConcreteDslxTypeShape {
    Bits {
        is_signed: bool,
        bit_count: usize,
    },
    Tuple {
        elements: Vec<ConcreteDslxTypeShape>,
    },
    Array {
        size: usize,
        element: Box<ConcreteDslxTypeShape>,
    },
}

impl ConcreteDslxTypeShape {
    pub(crate) fn rust_type(&self, target: RustTypeTarget) -> String {
        match self {
            ConcreteDslxTypeShape::Bits {
                is_signed,
                bit_count,
            } => bits_rust_type(target, *is_signed, *bit_count),
            ConcreteDslxTypeShape::Tuple { elements } => match elements.as_slice() {
                [] => "()".to_string(),
                [element] => format!("({},)", element.rust_type(target)),
                _ => format!(
                    "({})",
                    elements
                        .iter()
                        .map(|element| element.rust_type(target))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            },
            ConcreteDslxTypeShape::Array { size, element } => {
                format!("[{}; {size}]", element.rust_type(target))
            }
        }
    }
}

/// Parses concrete DSLX type strings for structural tuple lowering.
///
/// The DSLX bindings expose helpers for bits, arrays, structs, and enums but
/// not tuple members. This parser is intentionally narrow: it accepts only the
/// fully concrete bits/array/tuple spellings emitted by
/// `dslx::Type::to_string`.
pub(crate) fn parse_concrete_dslx_type_shape(
    text: &str,
) -> Result<ConcreteDslxTypeShape, XlsynthError> {
    let mut parser = ConcreteDslxTypeShapeParser { text, offset: 0 };
    let shape = parser.parse_type()?;
    parser.skip_ws();
    if parser.is_eof() {
        Ok(shape)
    } else {
        Err(XlsynthError(format!(
            "unsupported concrete DSLX type shape near `{}` in `{text}`",
            &text[parser.offset..]
        )))
    }
}

struct ConcreteDslxTypeShapeParser<'a> {
    text: &'a str,
    offset: usize,
}

impl ConcreteDslxTypeShapeParser<'_> {
    fn is_eof(&self) -> bool {
        self.offset >= self.text.len()
    }

    fn skip_ws(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.offset += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.text.as_bytes().get(self.offset).copied()
    }

    fn consume_char(&mut self, expected: u8) -> bool {
        self.skip_ws();
        if self.text.as_bytes().get(self.offset).copied() == Some(expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn consume_str(&mut self, expected: &str) -> bool {
        self.skip_ws();
        if self.text[self.offset..].starts_with(expected) {
            self.offset += expected.len();
            true
        } else {
            false
        }
    }

    fn expect_char(&mut self, expected: u8) -> Result<(), XlsynthError> {
        if self.consume_char(expected) {
            Ok(())
        } else {
            Err(XlsynthError(format!(
                "expected `{}` while parsing concrete DSLX type shape `{}`",
                expected as char, self.text
            )))
        }
    }

    fn parse_usize(&mut self) -> Result<usize, XlsynthError> {
        self.skip_ws();
        let start = self.offset;
        while self
            .text
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_digit)
        {
            self.offset += 1;
        }
        if self.offset == start {
            return Err(XlsynthError(format!(
                "expected decimal number while parsing concrete DSLX type shape `{}`",
                self.text
            )));
        }
        self.text[start..self.offset]
            .parse::<usize>()
            .map_err(|error| {
                XlsynthError(format!(
                    "invalid decimal number `{}` in concrete DSLX type shape `{}`: {error}",
                    &self.text[start..self.offset],
                    self.text
                ))
            })
    }

    fn parse_type(&mut self) -> Result<ConcreteDslxTypeShape, XlsynthError> {
        let mut shape = self.parse_atom()?;
        while self.consume_char(b'[') {
            let size = self.parse_usize()?;
            self.expect_char(b']')?;
            shape = ConcreteDslxTypeShape::Array {
                size,
                element: Box::new(shape),
            };
        }
        Ok(shape)
    }

    fn parse_atom(&mut self) -> Result<ConcreteDslxTypeShape, XlsynthError> {
        if self.consume_char(b'(') {
            let mut elements = Vec::new();
            if self.consume_char(b')') {
                return Ok(ConcreteDslxTypeShape::Tuple { elements });
            }
            loop {
                elements.push(self.parse_type()?);
                if self.consume_char(b')') {
                    break;
                }
                self.expect_char(b',')?;
                if self.consume_char(b')') {
                    break;
                }
            }
            return Ok(ConcreteDslxTypeShape::Tuple { elements });
        }

        if self.consume_str("uN") {
            return self.parse_bits(/* is_signed= */ false);
        }
        if self.consume_str("sN") {
            return self.parse_bits(/* is_signed= */ true);
        }
        if self.consume_str("bits") {
            return self.parse_bits(/* is_signed= */ false);
        }
        if self.peek() == Some(b'u') || self.peek() == Some(b's') {
            let is_signed = self.consume_char(b's');
            if !is_signed {
                self.expect_char(b'u')?;
            }
            let bit_count = self.parse_usize()?;
            return Ok(ConcreteDslxTypeShape::Bits {
                is_signed,
                bit_count,
            });
        }

        Err(XlsynthError(format!(
            "unsupported concrete DSLX type shape near `{}` in `{}`",
            &self.text[self.offset..],
            self.text
        )))
    }

    fn parse_bits(&mut self, is_signed: bool) -> Result<ConcreteDslxTypeShape, XlsynthError> {
        self.expect_char(b'[')?;
        let bit_count = self.parse_usize()?;
        self.expect_char(b']')?;
        Ok(ConcreteDslxTypeShape::Bits {
            is_signed,
            bit_count,
        })
    }
}

fn const_parametric_expr_value(
    type_info: &dslx::TypeInfo,
    expr: &dslx::Expr,
) -> Result<IrValue, XlsynthError> {
    if let Ok(value) =
        dslx::InterpValue::from_string(&expr.to_text()).and_then(|value| value.convert_to_ir())
    {
        return Ok(value);
    }
    type_info.get_const_expr(expr)?.convert_to_ir()
}

fn typed_literal_text_is_signed(text: &str) -> bool {
    let text = text.trim();
    text.starts_with('s') || text.starts_with("-")
}
/// Converts a dotted DSLX module name into sanitized Rust module path segments.
pub(crate) fn rust_module_path_from_dslx_module_name(module_name: &str) -> Vec<String> {
    module_name
        .split('.')
        .filter(|segment| !segment.is_empty())
        .map(sanitize_module_segment)
        .collect()
}

/// Converts a parsed DSLX import subject into sanitized Rust module segments.
fn rust_module_path_from_import(import: &dslx::Import) -> Vec<String> {
    import
        .get_subject()
        .iter()
        .map(|segment| sanitize_module_segment(segment))
        .collect()
}

/// Builds the Rust path from one DSLX module to a type declared in another.
///
/// The result uses `super` segments when needed, so callers should pass DSLX
/// module names rather than already-qualified Rust paths.
pub(crate) fn rust_type_path_between_dslx_modules(
    current_module_name: &str,
    target_module_name: &str,
    type_name: &str,
) -> String {
    let current_path = rust_module_path_from_dslx_module_name(current_module_name);
    let target_path = rust_module_path_from_dslx_module_name(target_module_name);
    rust_type_path_between_module_paths(&current_path, &target_path, type_name)
}

/// Builds the Rust path from one generated module path to a target type path.
///
/// If both paths are the same module, only `type_name` is returned. Otherwise
/// the result walks back to the generated root with `super` and then descends
/// into the target module path.
pub(crate) fn rust_type_path_between_module_paths(
    current_module_path: &[String],
    target_module_path: &[String],
    type_name: &str,
) -> String {
    if current_module_path == target_module_path {
        type_name.to_string()
    } else {
        std::iter::repeat_n("super".to_string(), current_module_path.len())
            .chain(target_module_path.iter().cloned())
            .chain([type_name.to_string()])
            .collect::<Vec<_>>()
            .join("::")
    }
}

/// Rewrites a DSLX module path segment into a Rust module identifier segment.
fn sanitize_module_segment(segment: &str) -> String {
    segment
        .chars()
        .map(|ch| {
            if ch == '_' || ch.is_ascii_alphanumeric() {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn sanitize_type_parametric_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for (index, ch) in segment.chars().enumerate() {
        let valid = ch == '_' || ch.is_ascii_alphanumeric();
        let ch = if valid { ch } else { '_' };
        if index == 0 && ch.is_ascii_digit() {
            out.push('_');
        }
        out.push(ch);
    }
    if out.is_empty() { "P".to_string() } else { out }
}

fn rust_type_parametric_value_suffix(
    value: &IrValue,
    is_signed: bool,
) -> Result<String, XlsynthError> {
    let format = if is_signed {
        IrFormatPreference::SignedDecimal
    } else {
        IrFormatPreference::UnsignedDecimal
    };
    let value = value.to_string_fmt_no_prefix(format)?;
    Ok(sanitize_type_parametric_value_segment(&value))
}

fn sanitize_type_parametric_value_segment(value: &str) -> String {
    let value = value.trim();
    if let Some(value) = value.strip_prefix('-') {
        format!("m{}", sanitize_type_parametric_value_atom(value))
    } else {
        sanitize_type_parametric_value_atom(value)
    }
}

fn sanitize_type_parametric_value_atom(value: &str) -> String {
    let out = value
        .chars()
        .map(|ch| {
            if ch == '_' || ch.is_ascii_alphanumeric() {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if out.is_empty() { "P".to_string() } else { out }
}
