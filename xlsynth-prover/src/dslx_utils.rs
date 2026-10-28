// SPDX-License-Identifier: Apache-2.0
//! Shared helpers for working with DSLX entry points and parametric metadata.

use std::collections::HashMap;
use std::path::Path;

use xlsynth::dslx::{Function, InterpValue, ParametricBinding, ParametricEnv};
use xlsynth::{XlsynthError, mangle_dslx_name};

use crate::prover::uf::validate_uf_symbol;

/// Returns the path text representable by the DSLX C API.
fn dslx_path_text(path: &Path) -> Result<&str, String> {
    let text = path
        .to_str()
        .ok_or_else(|| format!("DSLX path is not UTF-8: {path:?}"))?;
    if text.contains('\0') {
        return Err(format!("DSLX path contains a NUL byte: {path:?}"));
    }
    Ok(text)
}

/// Reject paths the DSLX C API cannot represent before passing them to libxls.
pub(crate) fn validate_dslx_paths<'a>(
    paths: impl IntoIterator<Item = &'a Path>,
) -> Result<(), String> {
    for path in paths {
        dslx_path_text(path)?;
    }
    Ok(())
}

/// Encodes search paths only when the receiving tool can split them back into
/// the same directories, in the same order.
pub(crate) fn join_dslx_search_paths(paths: &[&Path], separator: char) -> Result<String, String> {
    let mut joined = String::new();
    for path in paths {
        let text = dslx_path_text(path)?;
        if text.is_empty() {
            return Err("DSLX search path contains an empty directory name".into());
        }
        if text.contains(separator) {
            return Err(format!(
                "DSLX search path contains the separator {separator:?}: {path:?}"
            ));
        }
        if !joined.is_empty() {
            joined.push(separator);
        }
        joined.push_str(text);
    }
    Ok(joined)
}

/// Checks a DSLX mapping before it crosses a worker or solver boundary.
fn validate_dslx_uf_entry(function: &str, symbol: &str) -> Result<(), String> {
    if function.contains('\0') {
        return Err("xlsynth error: DSLX function name contains a NUL byte".into());
    }
    if function.is_empty() || function.trim() != function || function.contains(':') {
        return Err(format!(
            "Invalid DSLX function name in UF mapping: {function:?}"
        ));
    }
    if symbol.is_empty() || symbol.trim() != symbol {
        return Err(format!("Invalid uninterpreted function symbol: {symbol:?}"));
    }
    validate_uf_symbol(symbol)
}

/// Parses repeated `function:symbol` entries without letting later entries
/// hide malformed or conflicting proof assumptions.
pub fn parse_dslx_uf_specs<'a>(
    specs: impl IntoIterator<Item = &'a str>,
) -> Result<HashMap<String, String>, String> {
    let mut map = HashMap::new();
    for spec in specs {
        let (function, symbol) = spec.split_once(':')
            .filter(|(function, symbol)| !function.is_empty() && !symbol.is_empty())
            .ok_or_else(|| format!(
                "invalid uninterpreted function specification '{spec}'; expected <function-name>:<uf-name>"
            ))?;
        validate_dslx_uf_entry(function, symbol)?;
        if let Some(previous) = map.insert(function.to_owned(), symbol.to_owned()) {
            if previous != symbol {
                return Err(format!(
                    "Conflicting UF mappings for {function:?}: {previous:?} and {symbol:?}"
                ));
            }
        }
    }
    Ok(map)
}

/// Validates an existing typed mapping in deterministic order.
pub(crate) fn validate_dslx_uf_map<'a>(
    entries: impl IntoIterator<Item = (&'a String, &'a String)>,
) -> Result<(), String> {
    let mut entries: Vec<_> = entries.into_iter().collect();
    entries.sort_unstable();
    for (function, symbol) in entries {
        validate_dslx_uf_entry(function, symbol)?;
    }
    Ok(())
}

/// Translates DSLX function names to the IR names the solver sees, retaining
/// shared UF symbols.
pub fn mangle_dslx_uf_map(
    module_name: &str,
    uf_map: &HashMap<String, String>,
) -> Result<HashMap<String, String>, String> {
    validate_dslx_uf_map(uf_map)?;
    uf_map
        .iter()
        .map(|(name, symbol)| {
            Ok((
                mangle_dslx_name(module_name, name).map_err(|e| e.to_string())?,
                symbol.clone(),
            ))
        })
        .collect()
}

/// Parsed description of a DSLX top reference, including optional positional
/// param bindings.
#[derive(Debug)]
pub struct TopFunctionSpec {
    pub name: String,
    pub bindings: Vec<String>,
}

/// Parses a top function specification, optionally carrying param bindings like
/// `foo<u32:32,s8:-1>`.
pub fn parse_top_function_spec(raw: &str) -> Result<TopFunctionSpec, XlsynthError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(XlsynthError("Top function must not be empty".to_string()));
    }
    if !trimmed.contains('<') {
        if trimmed.contains('>') {
            return Err(XlsynthError(
                "Unexpected '>' in top function specification".to_string(),
            ));
        }
        return Ok(TopFunctionSpec {
            name: trimmed.to_string(),
            bindings: Vec::new(),
        });
    }

    let open_idx = trimmed.find('<').expect("guarded by contains('<') earlier");
    let name_part = trimmed[..open_idx].trim();
    if name_part.is_empty() {
        return Err(XlsynthError(
            "Top function name missing before parameter list".to_string(),
        ));
    }

    let params_section = &trimmed[open_idx + 1..];
    let close_idx = params_section
        .rfind('>')
        .ok_or_else(|| XlsynthError("Missing closing '>' in top parameter list".to_string()))?;
    let trailing = params_section[close_idx + 1..].trim();
    if !trailing.is_empty() {
        return Err(XlsynthError(
            "Unexpected trailing characters after top parameter list".to_string(),
        ));
    }

    let bindings_str = params_section[..close_idx].trim();
    if bindings_str.is_empty() {
        return Err(XlsynthError(
            "Top parameter list must not be empty".to_string(),
        ));
    }

    let mut bindings: Vec<String> = Vec::new();

    for raw_binding in bindings_str.split(',') {
        let binding = raw_binding.trim();
        if binding.is_empty() {
            return Err(XlsynthError(
                "Empty entry in top parameter list".to_string(),
            ));
        }
        if binding.contains('=') {
            return Err(XlsynthError(format!(
                "Named bindings are not supported; use positional syntax like '<u32:32>'. Offending binding: '{}'",
                binding
            )));
        }
        bindings.push(binding.to_string());
    }

    Ok(TopFunctionSpec {
        name: name_part.to_string(),
        bindings,
    })
}

/// Resolves the parsed param bindings against a concrete DSLX function and
/// materializes a `ParametricEnv`.
pub fn resolve_parametric_env(
    function: &Function,
    spec: &TopFunctionSpec,
    require_explicit: bool,
) -> Result<Option<ParametricEnv>, XlsynthError> {
    let formals: Vec<ParametricBinding> = function.parametric_bindings();
    let total_parametrics = formals.len();

    if total_parametrics == 0 {
        if spec.bindings.is_empty() {
            return Ok(None);
        }
        return Err(XlsynthError(format!(
            "Function '{}' does not accept parametric bindings but values were provided",
            function.get_identifier()
        )));
    }

    if require_explicit || !spec.bindings.is_empty() {
        if spec.bindings.len() != total_parametrics {
            return Err(XlsynthError(format!(
                "Function '{}' expects {} parametric bindings but {} were provided",
                spec.name,
                total_parametrics,
                spec.bindings.len()
            )));
        }
    } else {
        // The caller did not require explicit bindings and none were provided.
        // Returning `None` signals that defaults (if any) should be used.
        return Ok(None);
    }

    let mut assignments: Vec<(String, InterpValue)> = Vec::with_capacity(total_parametrics);
    for (value, formal) in spec.bindings.iter().zip(formals.iter()) {
        let name = formal.get_identifier();
        let interp = InterpValue::from_string(value).map_err(|err| {
            XlsynthError(format!(
                "Failed to parse value '{}' for parameter '{}': {}",
                value, name, err.0
            ))
        })?;
        assignments.push((name, interp));
    }

    let binding_refs: Vec<(&str, &InterpValue)> = assignments
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect();
    let env = ParametricEnv::new(&binding_refs).map_err(|err| {
        XlsynthError(format!(
            "Failed to build parametric environment for top '{}': {}",
            spec.name, err.0
        ))
    })?;
    Ok(Some(env))
}
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    #[cfg(unix)]
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

    use super::*;

    #[test]
    fn repeated_uf_specs_must_agree_and_each_be_valid() {
        let valid = HashMap::from([("f".to_owned(), "F".to_owned())]);
        assert_eq!(parse_dslx_uf_specs(["f:F", "f:F"]), Ok(valid));
        for specs in [["f:F ", "f:F"], ["f:F", "f:F "], ["f:F", "f:G"]] {
            assert!(parse_dslx_uf_specs(specs).is_err(), "{specs:?}");
        }
    }

    #[test]
    fn search_path_encoding_preserves_directory_names() {
        let paths = [Path::new("second dir"), Path::new("./first;dir")];
        assert_eq!(
            join_dslx_search_paths(&paths, ':').unwrap(),
            "second dir:./first;dir"
        );
        assert_eq!(join_dslx_search_paths(&[], ';').unwrap(), "");
        for path in [paths[1], Path::new(""), Path::new("dir\0name")] {
            assert!(join_dslx_search_paths(&[path], ';').is_err(), "{path:?}");
        }
        #[cfg(unix)]
        assert!(join_dslx_search_paths(&[Path::new(OsStr::from_bytes(b"dir\xff"))], ';').is_err());
    }

    fn parse_function_source(source: &str, name: &str) -> Function {
        use xlsynth::dslx::{ImportData, MatchableModuleMember, parse_and_typecheck};

        let mut import_data = ImportData::default();
        let tm =
            parse_and_typecheck(source, "test.x", "test", &mut import_data).expect("typecheck");
        let module = tm.get_module();
        for idx in 0..module.get_member_count() {
            if let Some(MatchableModuleMember::Function(f)) = module.get_member(idx).to_matchable()
            {
                if f.get_identifier() == name {
                    return f;
                }
            }
        }
        panic!("Function '{}' not found in generated module", name);
    }

    fn fake_function(binding_names: &[&str]) -> Function {
        let bindings = if binding_names.is_empty() {
            String::new()
        } else {
            format!(
                "<{}>",
                binding_names
                    .iter()
                    .map(|n| format!("{n}: u32"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let source = format!("fn wrapper{bindings}() -> u32 {{ u32:0 }}");
        parse_function_source(&source, "wrapper")
    }

    #[test]
    fn parses_plain_name() {
        let spec = parse_top_function_spec("foo").expect("parse plain top");
        assert_eq!(spec.name, "foo");
        assert!(spec.bindings.is_empty());
    }

    #[test]
    fn parses_param_bindings() {
        let spec = parse_top_function_spec("foo<u32:32>").expect("parse param top");
        assert_eq!(spec.name, "foo");
        assert_eq!(spec.bindings, vec!["u32:32"]);
    }

    #[test]
    fn rejects_empty_binding() {
        let err = parse_top_function_spec("foo<>").expect_err("expect parse failure");
        assert!(format!("{}", err).contains("must not be empty"));
    }

    #[test]
    fn resolve_positional_bindings() {
        let spec = parse_top_function_spec("wrapper<u32:32, s32:7>").unwrap();
        let function = fake_function(&["N", "M"]);
        let env = resolve_parametric_env(&function, &spec, false)
            .expect("resolve")
            .expect("env");
        let mut map: BTreeMap<String, String> = BTreeMap::new();
        for (name, value) in env.bindings() {
            map.insert(name, value.to_text());
        }
        assert_eq!(map.get("N").map(String::as_str), Some("u32:32"));
        assert_eq!(map.get("M").map(String::as_str), Some("s32:7"));
    }

    #[test]
    fn too_many_positional_bindings_errors() {
        let spec = parse_top_function_spec("wrapper<u32:1, u32:2>").unwrap();
        let function = fake_function(&["N"]);
        let err = resolve_parametric_env(&function, &spec, false).expect_err("should fail");
        assert!(format!("{}", err).contains("expects 1 parametric bindings but 2 were provided"));
    }

    #[test]
    fn too_few_positional_bindings_errors() {
        let spec = parse_top_function_spec("wrapper<u32:32>").unwrap();
        let function = fake_function(&["N", "M"]);
        let err = resolve_parametric_env(&function, &spec, false).expect_err("should fail");
        assert!(format!("{}", err).contains("expects 2 parametric bindings but 1 were provided"));
    }

    #[test]
    fn require_explicit_only_applies_when_bindings_required() {
        let spec = parse_top_function_spec("wrapper").unwrap();
        let function = parse_function_source("fn wrapper<N: u32>() -> u32 { u32:0 }", "wrapper");
        let err =
            resolve_parametric_env(&function, &spec, true).expect_err("should require bindings");
        let msg = format!("{}", err);
        assert!(
            msg.contains("requires explicit parametric bindings")
                || msg.contains("expects 1 parametric bindings"),
            "unexpected error message: {}",
            msg
        );
    }

    #[test]
    fn non_parametric_function_returns_none() {
        let spec = parse_top_function_spec("plain").unwrap();
        let function = parse_function_source("fn plain() -> u32 { u32:0 }", "plain");
        let env = resolve_parametric_env(&function, &spec, false).expect("resolve");
        assert!(env.is_none());
    }

    #[test]
    fn bindings_optional_when_not_required() {
        let spec = parse_top_function_spec("wrapper").unwrap();
        let function = parse_function_source("fn wrapper<N: u32>() -> u32 { u32:0 }", "wrapper");
        let env =
            resolve_parametric_env(&function, &spec, false).expect("resolve without explicit");
        assert!(
            env.is_none(),
            "expected None when bindings omitted and explicit not required"
        );
    }

    #[test]
    fn defaults_allowed_without_explicit_flag() {
        let spec = parse_top_function_spec("wrapper").unwrap();
        let function =
            parse_function_source("fn wrapper<N: u32 = {u32:4}>() -> u32 { u32:0 }", "wrapper");
        let env = resolve_parametric_env(&function, &spec, false).expect("resolve");
        assert!(env.is_none());
    }

    #[test]
    fn require_explicit_errors_for_defaults_only_formals() {
        let spec = parse_top_function_spec("wrapper").unwrap();
        let function =
            parse_function_source("fn wrapper<N: u32 = {u32:4}>() -> u32 { u32:0 }", "wrapper");
        let err = resolve_parametric_env(&function, &spec, true)
            .expect_err("explicit flag should still fail");
        let msg = format!("{}", err);
        assert!(
            msg.contains("requires explicit parametric bindings")
                || msg.contains("expects 1 parametric bindings"),
            "unexpected error message: {}",
            msg
        );
    }
}
