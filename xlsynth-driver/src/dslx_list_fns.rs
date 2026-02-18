// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::ArgMatches;
use serde::Serialize;

use crate::common::get_dslx_paths;
use crate::toolchain_config::ToolchainConfig;

#[derive(Debug, Serialize)]
struct ParamRecord {
    name: String,
    #[serde(rename = "type")]
    r#type: Option<String>,
}

#[derive(Debug, Serialize)]
struct ParametricBindingRecord {
    name: String,
    #[serde(rename = "type")]
    r#type: Option<String>,
    default_expr: Option<String>,
}

#[derive(Debug, Serialize)]
struct FunctionRecord {
    module_name: String,
    name: String,
    is_public: bool,
    is_parametric: bool,
    params: Vec<ParamRecord>,
    parametric_bindings: Vec<ParametricBindingRecord>,
    return_type: Option<String>,
    function_type: Option<String>,
}

fn resolve_type_annotation_text(
    type_info: &xlsynth::dslx::TypeInfo,
    type_annotation: &xlsynth::dslx::TypeAnnotation,
) -> Result<Option<String>, String> {
    let Some(ty) = type_info.get_type_for_type_annotation(type_annotation) else {
        return Ok(None);
    };
    ty.to_string().map(Some).map_err(|e| e.to_string())
}

fn make_function_record(
    module_name: &str,
    type_info: &xlsynth::dslx::TypeInfo,
    f: &xlsynth::dslx::Function,
) -> Result<FunctionRecord, String> {
    let is_parametric = f.is_parametric();

    let mut params = Vec::with_capacity(f.get_param_count());
    let mut param_type_texts = Vec::with_capacity(f.get_param_count());
    for i in 0..f.get_param_count() {
        let p = f.get_param(i);
        let type_text = resolve_type_annotation_text(type_info, &p.get_type_annotation())?;
        param_type_texts.push(type_text.clone());
        params.push(ParamRecord {
            name: p.get_name(),
            r#type: type_text,
        });
    }

    let mut parametric_bindings = Vec::with_capacity(f.get_parametric_binding_count());
    for i in 0..f.get_parametric_binding_count() {
        let b = f.get_parametric_binding(i);
        let type_text = match b.get_type_annotation() {
            Some(ta) => resolve_type_annotation_text(type_info, &ta)?,
            None => None,
        };
        let default_expr = b.get_expr().map(|expr| expr.to_text());
        parametric_bindings.push(ParametricBindingRecord {
            name: b.get_identifier(),
            r#type: type_text,
            default_expr,
        });
    }

    let return_type = match f.try_get_return_type() {
        Some(return_type) => resolve_type_annotation_text(type_info, &return_type)?,
        None => None,
    };
    let function_type = if !is_parametric
        && return_type.is_some()
        && param_type_texts.iter().all(|p| p.is_some())
    {
        let params_text = param_type_texts
            .iter()
            .map(|p| p.as_ref().expect("all parameter types should be present"))
            .map(std::string::String::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!(
            "({}) -> {}",
            params_text,
            return_type.as_ref().expect("return type should be present")
        ))
    } else {
        None
    };

    Ok(FunctionRecord {
        module_name: module_name.to_string(),
        name: f.get_identifier(),
        is_public: f.is_public(),
        is_parametric,
        params,
        parametric_bindings,
        return_type,
        function_type,
    })
}

fn is_ordinary_function(f: &xlsynth::dslx::Function) -> bool {
    !f.attributes().iter().any(|attr| {
        matches!(
            attr.kind(),
            xlsynth::dslx::AttributeKind::Quickcheck
                | xlsynth::dslx::AttributeKind::Test
                | xlsynth::dslx::AttributeKind::TestProc
        )
    })
}

fn collect_function_records(
    tcm: &xlsynth::dslx::TypecheckedModule,
) -> Result<Vec<FunctionRecord>, String> {
    let module = tcm.get_module();
    let module_name = module.get_name();
    let type_info = tcm.get_type_info();

    let mut out = Vec::new();
    for i in 0..module.get_member_count() {
        let member = module.get_member(i);
        match member.to_matchable() {
            Some(xlsynth::dslx::MatchableModuleMember::Function(f)) => {
                if is_ordinary_function(&f) {
                    out.push(make_function_record(&module_name, &type_info, &f)?);
                }
            }
            _ => {
                // Non-function module members are intentionally excluded from
                // function listing.
            }
        }
    }
    Ok(out)
}

pub fn handle_dslx_list_fns(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_path = PathBuf::from(
        matches
            .get_one::<String>("dslx_input_file")
            .expect("dslx_input_file required"),
    );
    let output_format = matches
        .get_one::<String>("format")
        .map(std::string::String::as_str)
        .unwrap_or("jsonl");

    let dslx_contents = match std::fs::read_to_string(&input_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "failed to read DSLX input file '{}': {}",
                input_path.display(),
                e
            );
            std::process::exit(1);
        }
    };
    let module_name = input_path
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("module")
        .to_string();

    let mut dslx_paths = get_dslx_paths(matches, config);
    if let Some(parent) = input_path.parent() {
        dslx_paths.search_paths.insert(0, parent.to_path_buf());
    }
    let stdlib_path = dslx_paths.stdlib_path.as_deref();
    let search_path_views = dslx_paths.search_path_views();
    let mut import_data = xlsynth::dslx::ImportData::new(stdlib_path, &search_path_views);

    let tcm = match xlsynth::dslx::parse_and_typecheck(
        &dslx_contents,
        input_path.to_str().unwrap_or("<non-utf8-path>"),
        &module_name,
        &mut import_data,
    ) {
        Ok(tcm) => tcm,
        Err(e) => {
            eprintln!("parse_and_typecheck failed: {}", e);
            std::process::exit(1);
        }
    };

    let records = match collect_function_records(&tcm) {
        Ok(records) => records,
        Err(e) => {
            eprintln!("failed to collect function metadata: {e}");
            std::process::exit(1);
        }
    };

    match output_format {
        "jsonl" => {
            for record in &records {
                println!("{}", serde_json::to_string(record).unwrap());
            }
        }
        "json" => {
            println!("{}", serde_json::to_string_pretty(&records).unwrap());
        }
        other => {
            eprintln!("unsupported --format value: {other}");
            std::process::exit(1);
        }
    }
}
