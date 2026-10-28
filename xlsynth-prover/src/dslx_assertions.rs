// SPDX-License-Identifier: Apache-2.0

//! Library entry point for proving assertions reachable from a DSLX function.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Serialize;
use xlsynth::dslx::{self, MatchableModuleMember, TypecheckedModule};
use xlsynth::{DslxCallingConvention, DslxConvertOptions};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;

use crate::dslx_utils::{mangle_dslx_uf_map, validate_dslx_paths};
use crate::proof_outcome::{ProofStatus, ProofSummary};
use crate::prover::property_fn::add_assertion_dependency_property;
use crate::prover::prover_for_choice;
use crate::prover::types::{
    BoolPropertyResult, FnInput, FnOutput, ParamDomains, ProverFn, QuickCheckAssertionSemantics,
};
use crate::prover::{Prover, SolverChoice};

/// Request describing a proof that DSLX assertions reachable from `top` hold.
pub struct DslxAssertionsRequest<'a> {
    pub source: &'a str,
    pub path: &'a Path,
    pub top: &'a str,
    pub solver: Option<SolverChoice>,
    pub dslx_stdlib_path: Option<&'a Path>,
    pub additional_search_paths: Vec<PathBuf>,
    pub assert_label_filter: Option<&'a str>,
    pub assume_enum_in_bound: bool,
    pub uf_map: AssertionUfMappings,
}

/// Names of functions to abstract, either local DSLX names or resolved IR names
/// (which may belong to an imported module).
pub enum AssertionUfMappings {
    LocalDslx(HashMap<String, String>),
    Ir(HashMap<String, String>),
}

/// Result of a DSLX assertion proof.
pub struct DslxAssertionsReport {
    pub duration: Duration,
    pub result: BoolPropertyResult,
}

const IMPLICIT_TOKEN_RUNTIME_PARAM_COUNT: usize = 2;

/// A user-visible DSLX input in the counterexample.
#[derive(Debug, Serialize)]
pub struct AssertionInput {
    pub name: String,
    pub value: String,
}

/// The observed value and any violated assertion.
#[derive(Debug, Serialize)]
pub struct AssertionOutput {
    pub value: String,
    pub assertion_label: Option<String>,
    pub assertion_message: Option<String>,
}

/// Inputs and output witnessing an assertion violation.
#[derive(Debug, Serialize)]
pub struct AssertionCounterexample {
    pub inputs: Vec<AssertionInput>,
    pub output: AssertionOutput,
}

/// An assertion proof and its counterexample with the synthetic ABI inputs
/// removed.
#[derive(Debug, Serialize)]
pub struct AssertionsOutcome {
    #[serde(flatten)]
    pub summary: ProofSummary,
    pub counterexample: Option<AssertionCounterexample>,
}

impl DslxAssertionsRequest<'_> {
    /// Proves this request, also reporting failures before a solver result was
    /// available.
    pub fn prove(&self) -> AssertionsOutcome {
        let start = Instant::now();
        match run_dslx_assertions(self) {
            Ok(report) => report.into(),
            Err(error) => AssertionsOutcome::execution_error(start.elapsed(), error),
        }
    }
}

impl AssertionsOutcome {
    /// Reports an invocation failure without counterexample evidence.
    pub fn execution_error(duration: Duration, error: String) -> Self {
        Self {
            summary: ProofSummary::execution_error(duration, error),
            counterexample: None,
        }
    }
}

/// Drops only the token and activation parameters inserted for the synthetic
/// property.
fn serialize_inputs(inputs: &[FnInput]) -> Vec<AssertionInput> {
    inputs
        .iter()
        // The synthetic property function always uses the standard
        // implicit-token ABI: param 0 is token, param 1 is activation.
        .skip(IMPLICIT_TOKEN_RUNTIME_PARAM_COUNT)
        .map(|input| AssertionInput {
            name: input.name.clone(),
            value: format!("{:?}", input.value),
        })
        .collect()
}

fn serialize_output(output: &FnOutput) -> AssertionOutput {
    let (assertion_label, assertion_message) = output
        .assertion_violation
        .as_ref()
        .map(|vio| (Some(vio.label.clone()), Some(vio.message.clone())))
        .unwrap_or((None, None));
    AssertionOutput {
        value: format!("{:?}", output.value),
        assertion_label,
        assertion_message,
    }
}

/// Keeps execution errors and inconclusive results separate from
/// counterexamples.
impl From<DslxAssertionsReport> for AssertionsOutcome {
    fn from(report: DslxAssertionsReport) -> Self {
        let DslxAssertionsReport { result, duration } = report;
        let status = ProofStatus::from(&result);
        let (counterexample, error_str) = match result {
            BoolPropertyResult::Proved => (None, None),
            BoolPropertyResult::Disproved { inputs, output } => (
                Some(AssertionCounterexample {
                    inputs: serialize_inputs(&inputs),
                    output: serialize_output(&output),
                }),
                None,
            ),
            BoolPropertyResult::Inconclusive(msg)
            | BoolPropertyResult::ToolchainDisproved(msg)
            | BoolPropertyResult::Error(msg) => (None, Some(msg)),
        };
        AssertionsOutcome {
            summary: ProofSummary {
                conclusion: status.into(),
                time_micros: duration.as_micros(),
                error_str,
            },
            counterexample,
        }
    }
}

fn collect_enum_values(tcm: &TypecheckedModule, enum_def: &xlsynth::dslx::EnumDef) -> Vec<IrValue> {
    let mut values = Vec::new();
    for member_index in 0..enum_def.get_member_count() {
        let member = enum_def.get_member(member_index);
        let expr = member.get_value();
        let owner_module = expr.get_owner_module();
        let owner_type_info = tcm
            .get_type_info_for_module(&owner_module)
            .expect("imported enum type info");
        let interp = owner_type_info
            .get_const_expr(&expr)
            .expect("enum member constexpr");
        let ir_value = interp.convert_to_ir().expect("enum member to IR value");
        let bits = ir_value.to_bits().expect("enum values are bits-typed");
        values.push(IrValue::from_bits(
            &xlsynth_pir::libxls_bridge::bits_from_libxls(&bits)
                .expect("enum bits should convert to native storage"),
        ));
    }
    values
}

fn get_function_enum_param_domains(
    tcm: &TypecheckedModule,
    dslx_top: &str,
) -> Result<ParamDomains, String> {
    let module = tcm.get_module();
    let type_info = tcm.get_type_info();
    let mut domains: ParamDomains = HashMap::new();
    let mut found = false;

    for member_index in 0..module.get_member_count() {
        if let Some(MatchableModuleMember::Function(function)) =
            module.get_member(member_index).to_matchable()
        {
            if function.get_identifier() == dslx_top {
                found = true;
                for param_index in 0..function.get_param_count() {
                    let param = function.get_param(param_index);
                    let name = param.get_name();
                    let annotation = param.get_type_annotation();
                    let ty = type_info
                        .get_type_for_type_annotation(&annotation)
                        .expect("param type");
                    if ty.is_enum() {
                        let enum_def = ty.get_enum_def().expect("enum def");
                        domains.insert(name, collect_enum_values(tcm, &enum_def));
                    }
                }
            }
        }
    }

    if found {
        Ok(domains)
    } else {
        let available = (0..module.get_member_count())
            .filter_map(|idx| module.get_member(idx).to_matchable())
            .filter_map(|member| match member {
                MatchableModuleMember::Function(function) => Some(function.get_identifier()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(", ");
        Err(format!(
            "Function '{}' not found in module '{}': available functions: {}",
            dslx_top,
            module.get_name(),
            available
        ))
    }
}

fn typecheck_request(
    request: &DslxAssertionsRequest<'_>,
    additional_search_paths: &[&Path],
) -> Result<TypecheckedModule, String> {
    let module_name = request
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            format!(
                "Failed to derive DSLX module name from path {}",
                request.path.display()
            )
        })?;
    let path_str = request
        .path
        .to_str()
        .ok_or_else(|| format!("DSLX path is not UTF-8: {}", request.path.display()))?;
    let mut import_data = dslx::ImportData::new(request.dslx_stdlib_path, additional_search_paths);
    dslx::parse_and_typecheck(request.source, path_str, module_name, &mut import_data)
        .map_err(|e| format!("DSLX parse/typecheck failed: {}", e))
}

fn prover_for_assertions(choice: SolverChoice) -> Result<Box<dyn Prover>, String> {
    match choice {
        SolverChoice::Toolchain => {
            Err("Solver 'toolchain' is not supported for dslx-fn-prove-assertions".to_string())
        }
        other => Ok(prover_for_choice(other, None)),
    }
}

/// Adds a boolean property function that invokes `top_name` and always returns
/// true, forcing assertion collection through the invoke dependency.
pub fn add_assertions_property_function(
    package: &mut ir::Package,
    top_name: &str,
) -> Result<String, String> {
    add_assertion_dependency_property(package, top_name, "assertions")
}

/// Proves that all selected assertions reachable from a DSLX top hold.
pub fn run_dslx_assertions(
    request: &DslxAssertionsRequest<'_>,
) -> Result<DslxAssertionsReport, String> {
    validate_dslx_paths(
        std::iter::once(request.path)
            .chain(request.dslx_stdlib_path)
            .chain(request.additional_search_paths.iter().map(PathBuf::as_path)),
    )?;
    let additional_search_path_refs: Vec<&Path> = request
        .additional_search_paths
        .iter()
        .map(|p| p.as_path())
        .collect();
    let typechecked = typecheck_request(request, &additional_search_path_refs)?;
    let top_domains = if request.assume_enum_in_bound {
        Some(get_function_enum_param_domains(&typechecked, request.top)?)
    } else {
        get_function_enum_param_domains(&typechecked, request.top)?;
        None
    };

    let module_name = request
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            format!(
                "Failed to derive DSLX module name from path {}",
                request.path.display()
            )
        })?;
    let mangled_top = xlsynth::mangle_dslx_name_with_calling_convention(
        module_name,
        request.top,
        DslxCallingConvention::ImplicitToken,
    )
    .map_err(|e| format!("Failed to mangle top function '{}': {}", request.top, e))?;

    let conversion = xlsynth::convert_dslx_to_ir_text(
        request.source,
        request.path,
        &DslxConvertOptions {
            dslx_stdlib_path: request.dslx_stdlib_path,
            additional_search_paths: additional_search_path_refs,
            enable_warnings: None,
            disable_warnings: None,
            force_implicit_token_calling_convention: true,
        },
    )
    .map_err(|e| format!("DSLX->IR conversion failed: {}", e))?;
    let mut package = Parser::new(&conversion.ir)
        .parse_package()
        .map_err(|e| format!("Failed to parse converted IR package: {}", e))?;
    let property_name = add_assertions_property_function(&mut package, &mangled_top)?;
    let property_fn = package
        .get_fn(&property_name)
        .ok_or_else(|| format!("IR function '{}' not found", property_name))?;
    let prover_fn = ProverFn::new(property_fn, Some(&package))
        .with_fixed_implicit_activation(true)
        .with_domains(top_domains)
        .with_uf_map(match &request.uf_map {
            AssertionUfMappings::LocalDslx(names) => mangle_dslx_uf_map(module_name, names)?,
            AssertionUfMappings::Ir(names) => names.clone(),
        });

    let prover = prover_for_assertions(request.solver.unwrap_or(SolverChoice::Bitwuzla))?;
    let start = Instant::now();
    let result = prover.prove_ir_quickcheck(
        &prover_fn,
        QuickCheckAssertionSemantics::Never,
        request.assert_label_filter,
    );
    Ok(DslxAssertionsReport {
        duration: start.elapsed(),
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::types::AssertionViolation;
    use xlsynth_pir::ir::NodePayload;

    /// A resolved IR name can abstract assertions in an imported helper.
    #[cfg(feature = "has-bitwuzla")]
    #[test]
    fn retains_imported_helper_ir_names_for_assertion_proofs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entry.x");
        let source = "import dep; pub fn main(x: u8) -> u8 { dep::helper(x) }";
        std::fs::write(
            dir.path().join("dep.x"),
            r#"
pub fn helper(x: u8) -> u8 { assert!(x != u8:0, "nonzero"); x }
"#,
        )
        .unwrap();
        for (mappings, expected) in [
            (HashMap::new(), ProofStatus::Disproved),
            (
                HashMap::from([("__dep__helper".to_owned(), "F".to_owned())]),
                ProofStatus::Proved,
            ),
        ] {
            let request = DslxAssertionsRequest {
                source,
                path: &path,
                top: "main",
                solver: Some(SolverChoice::Bitwuzla),
                dslx_stdlib_path: None,
                additional_search_paths: vec![dir.path().to_owned()],
                assert_label_filter: None,
                assume_enum_in_bound: false,
                uf_map: AssertionUfMappings::Ir(mappings),
            };
            let result = request.prove();
            assert_eq!(result.summary.conclusion.status, expected, "{result:?}");
        }
    }

    #[test]
    fn property_function_depends_on_invoke_result() {
        let ir_text = r#"
package p

fn __itok__f(__token: token id=1, __activation: bits[1] id=2, x: bits[8] id=3) -> (token, bits[8]) {
  ret tuple.4: (token, bits[8]) = tuple(__token, x, id=4)
}
"#;
        let mut package = Parser::new(ir_text).parse_package().expect("parse package");
        let property_name =
            add_assertions_property_function(&mut package, "__itok__f").expect("property");
        let property_fn = package.get_fn(&property_name).expect("property fn");
        let ret_ref = property_fn.ret_node_ref.expect("ret ref");
        match &property_fn.nodes[ret_ref.index].payload {
            NodePayload::TupleIndex { tuple, index } => {
                assert_eq!(*index, 1);
                match &property_fn.nodes[tuple.index].payload {
                    NodePayload::Tuple(elements) => {
                        assert_eq!(elements.len(), 2);
                        assert!(matches!(
                            property_fn.nodes[elements[0].index].payload,
                            NodePayload::Invoke { .. }
                        ));
                    }
                    other => panic!("expected tuple node, got {other:?}"),
                }
            }
            other => panic!("expected tuple index return, got {other:?}"),
        }
    }

    #[test]
    fn assertion_json_does_not_conflate_errors_with_counterexamples() {
        for (result, status, error) in [
            (BoolPropertyResult::Proved, "proved", None),
            (
                BoolPropertyResult::Inconclusive("unknown".into()),
                "inconclusive",
                Some("unknown"),
            ),
            (
                BoolPropertyResult::Error("solver failed".into()),
                "execution-error",
                Some("solver failed"),
            ),
            (
                BoolPropertyResult::ToolchainDisproved("tool failed".into()),
                "execution-error",
                Some("tool failed"),
            ),
        ] {
            let outcome = AssertionsOutcome::from(DslxAssertionsReport {
                result,
                duration: Duration::ZERO,
            });
            assert_eq!(
                serde_json::to_value(outcome).unwrap(),
                serde_json::json!({
                    "time_micros": 0,
                    "status": status,
                    "success": status == "proved",
                    "counterexample": null,
                    "error_str": error,
                }),
            );
        }
    }

    #[test]
    fn assertion_counterexample_exposes_user_inputs_including_double_underscore_names() {
        let result = BoolPropertyResult::Disproved {
            inputs: vec![
                FnInput {
                    name: "__token".into(),
                    value: IrValue::make_token(),
                },
                FnInput {
                    name: "__activated".into(),
                    value: IrValue::bool(true),
                },
                FnInput {
                    name: "__user".into(),
                    value: IrValue::make_ubits(8, 7).unwrap(),
                },
            ],
            output: FnOutput {
                value: IrValue::bool(false),
                assertion_violation: Some(AssertionViolation {
                    label: "range".into(),
                    message: "value out of range".into(),
                }),
            },
        };
        let outcome = AssertionsOutcome::from(DslxAssertionsReport {
            result,
            duration: Duration::from_micros(123),
        });
        assert_eq!(
            serde_json::to_value(outcome).unwrap(),
            serde_json::json!({
                "time_micros": 123,
                "status": "disproved",
                "success": false,
                "counterexample": {
                    "inputs": [{ "name": "__user", "value": "bits[8]:7" }],
                    "output": { "value": "bits[1]:0", "assertion_label": "range", "assertion_message": "value out of range" }
                },
                "error_str": null,
            })
        );
    }
}
