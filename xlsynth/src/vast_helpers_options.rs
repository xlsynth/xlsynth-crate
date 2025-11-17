// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TemplateVariable {
    Reg,
    Next,
    Clock,
    Reset,
    Enable,
    ResetValue,
    Condition,
    Label,
    Signal,
}

impl std::fmt::Display for TemplateVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateVariable::Reg => write!(f, "reg"),
            TemplateVariable::Next => write!(f, "next"),
            TemplateVariable::Clock => write!(f, "clock"),
            TemplateVariable::Reset => write!(f, "reset"),
            TemplateVariable::Enable => write!(f, "en"),
            TemplateVariable::ResetValue => write!(f, "reset_value"),
            TemplateVariable::Condition => write!(f, "condition"),
            TemplateVariable::Label => write!(f, "label"),
            TemplateVariable::Signal => write!(f, "signal"),
        }
    }
}

fn parse_template_variable(token: &str) -> Result<TemplateVariable, String> {
    match token.trim() {
        "reg" => Ok(TemplateVariable::Reg),
        "next" => Ok(TemplateVariable::Next),
        "clock" => Ok(TemplateVariable::Clock),
        "reset" => Ok(TemplateVariable::Reset),
        "en" => Ok(TemplateVariable::Enable),
        "reset_value" => Ok(TemplateVariable::ResetValue),
        "condition" => Ok(TemplateVariable::Condition),
        "label" => Ok(TemplateVariable::Label),
        "signal" => Ok(TemplateVariable::Signal),
        other => Err(format!("unknown template variable: {other}")),
    }
}

fn extract_variables_and_normalize_template(
    template: &str,
) -> Result<(String, std::collections::BTreeSet<TemplateVariable>), String> {
    use regex::Regex;
    let re = Regex::new(r"\{\{([^{}]*)\}\}").map_err(|e| e.to_string())?;
    let mut out = String::with_capacity(template.len());
    let mut vars: std::collections::BTreeSet<TemplateVariable> = std::collections::BTreeSet::new();
    let mut last_end = 0usize;
    for caps in re.captures_iter(template) {
        let m = caps.get(0).unwrap();
        let inner = caps.get(1).unwrap().as_str().trim();
        if inner.is_empty() {
            return Err("empty template variable name".to_string());
        }
        let var = parse_template_variable(inner)?;
        vars.insert(var.clone());
        // Push literal section before the match
        out.push_str(&template[last_end..m.start()]);
        // Push normalized placeholder
        out.push_str("{{");
        out.push_str(&var.to_string());
        out.push_str("}}");
        last_end = m.end();
    }
    // Tail section after last match
    out.push_str(&template[last_end..]);
    Ok((out, vars))
}

fn validate_template_variables(
    found: &std::collections::BTreeSet<TemplateVariable>,
    expected: &[TemplateVariable],
) -> Result<(), String> {
    let expected_set: std::collections::BTreeSet<TemplateVariable> =
        expected.iter().cloned().collect();
    if &expected_set != found {
        let fmt_set = |s: &std::collections::BTreeSet<TemplateVariable>| {
            let mut v: Vec<String> = s.iter().map(|x| x.to_string()).collect();
            v.sort();
            v.join(", ")
        };
        return Err(format!(
            "template variables mismatch: expected {{{}}} but found {{{}}}",
            fmt_set(&expected_set),
            fmt_set(found)
        ));
    }
    Ok(())
}

fn normalize_and_validate_template(
    template: &str,
    expected: &[TemplateVariable],
) -> Result<String, String> {
    let (normalized, found) = extract_variables_and_normalize_template(template)?;
    validate_template_variables(&found, expected)?;
    Ok(normalized)
}

// TODO(meheff): add a verify method
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodegenOptions {
    pub includes: Option<Vec<String>>,
    #[serde(default, deserialize_with = "deserialize_unused_signal_template")]
    pub unused_signal_template: Option<String>,
    #[serde(default, deserialize_with = "deserialize_static_assert_template")]
    pub static_assert_template: Option<String>,
    #[serde(default, deserialize_with = "deserialize_reg_template")]
    pub reg_template: Option<String>,
    #[serde(default, deserialize_with = "deserialize_reg_with_en_template")]
    pub reg_with_en_template: Option<String>,
    #[serde(default, deserialize_with = "deserialize_reg_with_reset_template")]
    pub reg_with_reset_template: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_reg_with_reset_with_en_template"
    )]
    pub reg_with_reset_with_en_template: Option<String>,
}

#[derive(Debug, Error)]
pub enum OptionsLoadError {
    #[error("failed to read options file {path}: {source}")]
    Io {
        #[source]
        source: std::io::Error,
        path: String,
    },
    #[error("failed to parse TOML options from {path}: {err}")]
    Toml { err: toml::de::Error, path: String },
}

// Field-level deserializers with validation and normalization
fn deserialize_template<'de, D>(
    deserializer: D,
    key_name: &str,
    expected_variables: &[TemplateVariable],
) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    match opt {
        Some(s) => normalize_and_validate_template(&s, expected_variables)
            .map(Some)
            .map_err(|e| {
                serde::de::Error::custom(format!("failed to parse template for `{key_name}`: {e}"))
            }),
        None => Ok(None),
    }
}

fn deserialize_unused_signal_template<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_template(
        deserializer,
        "unused_signal_template",
        &[TemplateVariable::Label, TemplateVariable::Signal],
    )
}

fn deserialize_static_assert_template<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_template(
        deserializer,
        "static_assert_template",
        &[TemplateVariable::Label, TemplateVariable::Condition],
    )
}

fn deserialize_reg_template<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_template(
        deserializer,
        "reg_template",
        &[
            TemplateVariable::Reg,
            TemplateVariable::Next,
            TemplateVariable::Clock,
        ],
    )
}

fn deserialize_reg_with_en_template<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_template(
        deserializer,
        "reg_with_en_template",
        &[
            TemplateVariable::Reg,
            TemplateVariable::Next,
            TemplateVariable::Enable,
            TemplateVariable::Clock,
        ],
    )
}

fn deserialize_reg_with_reset_template<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_template(
        deserializer,
        "reg_with_reset_template",
        &[
            TemplateVariable::Reg,
            TemplateVariable::Next,
            TemplateVariable::ResetValue,
            TemplateVariable::Clock,
            TemplateVariable::Reset,
        ],
    )
}

fn deserialize_reg_with_reset_with_en_template<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_template(
        deserializer,
        "reg_with_reset_with_en_template",
        &[
            TemplateVariable::Reg,
            TemplateVariable::Next,
            TemplateVariable::Enable,
            TemplateVariable::ResetValue,
            TemplateVariable::Clock,
            TemplateVariable::Reset,
        ],
    )
}
