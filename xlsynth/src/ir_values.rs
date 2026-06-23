// SPDX-License-Identifier: Apache-2.0

//! Parsing and formatting for newline-delimited XLS IR value sequences.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use crate::{IrValue, XlsynthError};

/// One name/value entry in a named evaluator input record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedIrValue {
    pub name: String,
    pub value: IrValue,
}

/// An ordered, duplicate-free collection of named XLS IR values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedIrValueSet {
    entries: Vec<NamedIrValue>,
}

/// Identifies which homogeneous record form an `.irvals` file uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrValuesFileKind {
    Positional,
    Named,
}

impl NamedIrValueSet {
    /// Creates a named value set, rejecting duplicate names.
    pub fn new(entries: Vec<NamedIrValue>) -> Result<Self, XlsynthError> {
        let mut names = BTreeSet::new();
        for entry in &entries {
            if !names.insert(entry.name.as_str()) {
                return Err(XlsynthError(format!(
                    "duplicate name '{}' in named IR value set",
                    entry.name
                )));
            }
        }
        Ok(Self { entries })
    }

    /// Creates a named value set from a positional tuple and argument names.
    pub fn from_positional_tuple(
        argument_names: &[String],
        tuple_value: &IrValue,
    ) -> Result<Self, XlsynthError> {
        let elements = tuple_value.get_elements().map_err(|e| {
            XlsynthError(format!(
                "cannot create named IR value set from a non-tuple value: {}",
                e
            ))
        })?;
        if elements.len() != argument_names.len() {
            return Err(XlsynthError(format!(
                "positional tuple arity mismatch: expected {}, got {}",
                argument_names.len(),
                elements.len()
            )));
        }
        let entries = argument_names
            .iter()
            .cloned()
            .zip(elements)
            .map(|(name, value)| NamedIrValue { name, value })
            .collect();
        Self::new(entries)
    }

    /// Returns entries in source order.
    pub fn entries(&self) -> &[NamedIrValue] {
        &self.entries
    }

    /// Consumes the set and returns entries in source order.
    pub fn into_entries(self) -> Vec<NamedIrValue> {
        self.entries
    }
}

/// The two supported homogeneous forms of an `.irvals` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrValuesFile {
    /// One arbitrary XLS typed value per line.
    ValueSequence(Vec<IrValue>),
    /// One named argument set per line.
    NamedValueSequence(Vec<NamedIrValueSet>),
}

impl IrValuesFile {
    /// Returns the record form used by this value sequence.
    pub fn kind(&self) -> IrValuesFileKind {
        match self {
            Self::ValueSequence(_) => IrValuesFileKind::Positional,
            Self::NamedValueSequence(_) => IrValuesFileKind::Named,
        }
    }

    /// Converts named records to tuples in `argument_names` order.
    ///
    /// Positional records are returned unchanged. Every named record must have
    /// exactly the expected names; entry order in the file is immaterial for
    /// binding.
    pub fn into_positional_values(
        self,
        argument_names: &[String],
    ) -> Result<Vec<IrValue>, XlsynthError> {
        let sets = match self {
            Self::ValueSequence(values) => return Ok(values),
            Self::NamedValueSequence(sets) => sets,
        };

        let expected = argument_names.iter().cloned().collect::<BTreeSet<_>>();
        if expected.len() != argument_names.len() {
            return Err(XlsynthError(
                "evaluator interface contains duplicate argument names".to_string(),
            ));
        }

        sets.into_iter()
            .enumerate()
            .map(|(sample_index, set)| {
                let mut by_name = set
                    .into_entries()
                    .into_iter()
                    .map(|entry| (entry.name, entry.value))
                    .collect::<BTreeMap<_, _>>();
                let observed = by_name.keys().cloned().collect::<BTreeSet<_>>();
                let missing = expected.difference(&observed).cloned().collect::<Vec<_>>();
                let unknown = observed.difference(&expected).cloned().collect::<Vec<_>>();
                if !missing.is_empty() || !unknown.is_empty() {
                    return Err(XlsynthError(format!(
                        "named input sample {} does not match evaluator arguments: missing {:?}, unknown {:?}",
                        sample_index + 1,
                        missing,
                        unknown
                    )));
                }
                let args = argument_names
                    .iter()
                    .map(|name| {
                        by_name
                            .remove(name)
                            .expect("exact name-set validation guarantees an entry")
                    })
                    .collect::<Vec<_>>();
                Ok(IrValue::make_tuple(&args))
            })
            .collect()
    }
}

/// Parses an `.irvals` sequence from text.
pub fn parse_ir_values(text: &str) -> Result<IrValuesFile, XlsynthError> {
    enum FileForm {
        Values,
        NamedValues,
    }

    let mut form: Option<FileForm> = None;
    let mut values = Vec::new();
    let mut named_values = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            return Err(XlsynthError(format!(
                "empty line {} in IR values file is not allowed",
                line_number
            )));
        }
        let line_is_named = line.starts_with('{');
        match &form {
            None => {
                form = Some(if line_is_named {
                    FileForm::NamedValues
                } else {
                    FileForm::Values
                });
            }
            Some(FileForm::Values) if line_is_named => {
                return Err(XlsynthError(format!(
                    "IR values file mixes positional and named records at line {}",
                    line_number
                )));
            }
            Some(FileForm::NamedValues) if !line_is_named => {
                return Err(XlsynthError(format!(
                    "IR values file mixes named and positional records at line {}",
                    line_number
                )));
            }
            Some(FileForm::Values | FileForm::NamedValues) => {
                // The record agrees with the file form selected by the first
                // line.
            }
        }

        if line_is_named {
            named_values.push(parse_named_value_set(line, line_number)?);
        } else {
            values.push(IrValue::parse_typed(line).map_err(|e| {
                XlsynthError(format!(
                    "failed to parse typed IR value at line {}: {}",
                    line_number, e
                ))
            })?);
        }
    }

    Ok(match form {
        Some(FileForm::NamedValues) => IrValuesFile::NamedValueSequence(named_values),
        Some(FileForm::Values) | None => IrValuesFile::ValueSequence(values),
    })
}

/// Reads and parses an `.irvals` file.
pub fn parse_ir_values_file(path: &Path) -> Result<IrValuesFile, XlsynthError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        XlsynthError(format!(
            "failed to read IR values file '{}': {}",
            path.display(),
            e
        ))
    })?;
    parse_ir_values(&text)
}

/// Parses one `{name: value, ...}` record.
fn parse_named_value_set(line: &str, line_number: usize) -> Result<NamedIrValueSet, XlsynthError> {
    let Some(inner) = line
        .strip_prefix('{')
        .and_then(|contents| contents.strip_suffix('}'))
    else {
        return Err(XlsynthError(format!(
            "named IR value record at line {} must end with '}}'",
            line_number
        )));
    };
    let raw_entries = split_named_entries(inner, line_number)?;
    let mut entries = Vec::with_capacity(raw_entries.len());
    for raw_entry in raw_entries {
        let Some(separator) = find_name_separator(raw_entry) else {
            return Err(XlsynthError(format!(
                "named IR value entry '{}' at line {} is missing ':'",
                raw_entry.trim(),
                line_number
            )));
        };
        let raw_name = raw_entry[..separator].trim();
        let raw_value = raw_entry[separator + 1..].trim();
        if raw_value.is_empty() {
            return Err(XlsynthError(format!(
                "named IR value '{}' at line {} has no value",
                raw_name, line_number
            )));
        }
        let name = parse_value_name(raw_name, line_number)?;
        let value = IrValue::parse_typed(raw_value).map_err(|e| {
            XlsynthError(format!(
                "failed to parse value for '{}' at line {}: {}",
                name, line_number, e
            ))
        })?;
        entries.push(NamedIrValue { name, value });
    }
    NamedIrValueSet::new(entries).map_err(|e| {
        XlsynthError(format!(
            "invalid named IR value record at line {}: {}",
            line_number, e.0
        ))
    })
}

/// Splits entries at commas outside strings, tuples, and arrays.
fn split_named_entries(inner: &str, line_number: usize) -> Result<Vec<&str>, XlsynthError> {
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    let mut start = 0;
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in inner.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '(' => paren_depth += 1,
            ')' => {
                paren_depth = paren_depth.checked_sub(1).ok_or_else(|| {
                    XlsynthError(format!(
                        "unmatched ')' in named IR value record at line {}",
                        line_number
                    ))
                })?;
            }
            '[' => bracket_depth += 1,
            ']' => {
                bracket_depth = bracket_depth.checked_sub(1).ok_or_else(|| {
                    XlsynthError(format!(
                        "unmatched ']' in named IR value record at line {}",
                        line_number
                    ))
                })?;
            }
            ',' if paren_depth == 0 && bracket_depth == 0 => {
                let entry = inner[start..index].trim();
                if entry.is_empty() {
                    return Err(XlsynthError(format!(
                        "empty named IR value entry at line {}",
                        line_number
                    )));
                }
                entries.push(entry);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    if in_string || escaped {
        return Err(XlsynthError(format!(
            "unterminated quoted name in named IR value record at line {}",
            line_number
        )));
    }
    if paren_depth != 0 || bracket_depth != 0 {
        return Err(XlsynthError(format!(
            "unclosed aggregate in named IR value record at line {}",
            line_number
        )));
    }
    let final_entry = inner[start..].trim();
    if !final_entry.is_empty() {
        entries.push(final_entry);
    }
    Ok(entries)
}

fn find_name_separator(entry: &str) -> Option<usize> {
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in entry.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            ':' => return Some(index),
            _ => {}
        }
    }
    None
}

fn parse_value_name(raw_name: &str, line_number: usize) -> Result<String, XlsynthError> {
    if raw_name.starts_with('"') {
        return serde_json::from_str(raw_name).map_err(|e| {
            XlsynthError(format!(
                "invalid quoted name '{}' at line {}: {}",
                raw_name, line_number, e
            ))
        });
    }
    if is_bare_name(raw_name) {
        return Ok(raw_name.to_string());
    }
    Err(XlsynthError(format!(
        "invalid name '{}' at line {}; use a JSON-quoted name for punctuation",
        raw_name, line_number
    )))
}

fn is_bare_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch == '$' || ch == '.' || ch.is_ascii_alphanumeric())
}

fn format_value_name(name: &str) -> String {
    if is_bare_name(name) {
        name.to_string()
    } else {
        serde_json::to_string(name).expect("serializing a string to JSON cannot fail")
    }
}

impl fmt::Display for NamedIrValueSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (index, entry) in self.entries.iter().enumerate() {
            if index != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", format_value_name(&entry.name), entry.value)?;
        }
        write!(f, "}}")
    }
}

impl fmt::Display for IrValuesFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ValueSequence(values) => {
                for value in values {
                    writeln!(f, "{value}")?;
                }
            }
            Self::NamedValueSequence(sets) => {
                for set in sets {
                    writeln!(f, "{set}")?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named_value(name: &str, value: &str) -> NamedIrValue {
        NamedIrValue {
            name: name.to_string(),
            value: IrValue::parse_typed(value).unwrap(),
        }
    }

    #[test]
    fn parses_positional_value_sequence() {
        let parsed = parse_ir_values("(bits[8]:1, bits[1]:0)\n(bits[8]:2, bits[1]:1)\n").unwrap();
        let IrValuesFile::ValueSequence(values) = parsed else {
            panic!("expected positional values");
        };
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].to_string(), "(bits[8]:1, bits[1]:0)");
    }

    #[test]
    fn parses_named_value_sequence_and_binds_argument_order() {
        let parsed =
            parse_ir_values("{y: bits[1]:0, x: bits[8]:1}\n{x: bits[8]:2, y: bits[1]:1}\n")
                .unwrap();
        assert_eq!(
            parsed.to_string(),
            "{y: bits[1]:0, x: bits[8]:1}\n{x: bits[8]:2, y: bits[1]:1}\n"
        );
        let values = parsed
            .into_positional_values(&["x".to_string(), "y".to_string()])
            .unwrap();
        assert_eq!(values[0].to_string(), "(bits[8]:1, bits[1]:0)");
        assert_eq!(values[1].to_string(), "(bits[8]:2, bits[1]:1)");
    }

    #[test]
    fn parses_nested_values_and_quoted_names() {
        let parsed = parse_ir_values(
            r#"{"pipeline.reg[0]": (bits[8]:1, bits[8]:2), ys: [bits[4]:3, bits[4]:4]}
"#,
        )
        .unwrap();
        assert_eq!(
            parsed.to_string(),
            "{\"pipeline.reg[0]\": (bits[8]:1, bits[8]:2), ys: [bits[4]:3, bits[4]:4]}\n"
        );
    }

    #[test]
    fn rejects_duplicate_and_mismatched_names() {
        let duplicate = parse_ir_values("{x: bits[1]:0, x: bits[1]:1}\n").unwrap_err();
        assert!(duplicate.0.contains("duplicate name 'x'"));

        let parsed = parse_ir_values("{x: bits[1]:0, z: bits[1]:1}\n").unwrap();
        let mismatch = parsed
            .into_positional_values(&["x".to_string(), "y".to_string()])
            .unwrap_err();
        assert!(mismatch.0.contains("missing [\"y\"]"));
        assert!(mismatch.0.contains("unknown [\"z\"]"));
    }

    #[test]
    fn rejects_mixed_file_forms() {
        let error = parse_ir_values("(bits[1]:0)\n{x: bits[1]:1}\n").unwrap_err();
        assert!(error.0.contains("mixes positional and named records"));
    }

    #[test]
    fn named_value_set_constructor_preserves_order() {
        let set = NamedIrValueSet::new(vec![
            named_value("y", "bits[1]:0"),
            named_value("x", "bits[8]:1"),
        ])
        .unwrap();
        assert_eq!(set.entries()[0].name, "y");
        assert_eq!(set.entries()[1].name, "x");
    }

    #[test]
    fn named_value_set_from_positional_tuple() {
        let tuple = IrValue::parse_typed("(bits[8]:1, bits[1]:0)").unwrap();
        let set =
            NamedIrValueSet::from_positional_tuple(&["x".to_string(), "y".to_string()], &tuple)
                .unwrap();
        assert_eq!(set.to_string(), "{x: bits[8]:1, y: bits[1]:0}");

        let scalar = IrValue::parse_typed("bits[1]:0").unwrap();
        let error =
            NamedIrValueSet::from_positional_tuple(&["x".to_string()], &scalar).unwrap_err();
        assert!(error.0.contains("non-tuple"));

        let error = NamedIrValueSet::from_positional_tuple(&["x".to_string()], &tuple).unwrap_err();
        assert!(error.0.contains("expected 1, got 2"));
    }

    #[test]
    fn reports_file_kind() {
        assert_eq!(
            parse_ir_values("bits[1]:0\n").unwrap().kind(),
            IrValuesFileKind::Positional
        );
        assert_eq!(
            parse_ir_values("{x: bits[1]:0}\n").unwrap().kind(),
            IrValuesFileKind::Named
        );
    }
}
