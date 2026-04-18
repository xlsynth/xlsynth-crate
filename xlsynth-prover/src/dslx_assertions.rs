// SPDX-License-Identifier: Apache-2.0

//! Library entry point for proving assertions reachable from a DSLX function.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use xlsynth::dslx::{self, MatchableModuleMember, TypecheckedModule};
use xlsynth::{DslxCallingConvention, DslxConvertOptions, IrValue};
use xlsynth_pir::ir::{self, Node, NodePayload, NodeRef, PackageMember, ParamId, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils::next_text_id;

#[cfg(any(
    feature = "has-bitwuzla",
    feature = "has-boolector",
    feature = "has-easy-smt"
))]
use crate::prover::prover_for_choice;
use crate::prover::types::{
    BoolPropertyResult, ParamDomains, ProverFn, QuickCheckAssertionSemantics,
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
    pub uf_map: HashMap<String, String>,
}

/// Result of a DSLX assertion proof.
pub struct DslxAssertionsReport {
    pub duration: Duration,
    pub result: BoolPropertyResult,
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
        values.push(interp.convert_to_ir().expect("enum member to IR value"));
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

fn push_node(
    nodes: &mut Vec<Node>,
    next_id: &mut usize,
    name: Option<String>,
    ty: Type,
    payload: NodePayload,
) -> NodeRef {
    let text_id = *next_id;
    *next_id += 1;
    let index = nodes.len();
    nodes.push(Node {
        text_id,
        name,
        ty,
        payload,
        pos: None,
    });
    NodeRef { index }
}

fn clone_param_with_new_id(param: &ir::Param, id: usize) -> ir::Param {
    ir::Param {
        name: param.name.clone(),
        ty: param.ty.clone(),
        id: ParamId::new(id),
    }
}

fn auto_prover_for_assertions() -> Result<Box<dyn Prover>, String> {
    #[cfg(feature = "has-bitwuzla")]
    {
        use crate::solver::bitwuzla::BitwuzlaOptions;
        return Ok(Box::new(BitwuzlaOptions::new()));
    }
    #[cfg(all(feature = "has-boolector", not(feature = "has-bitwuzla")))]
    {
        use crate::solver::boolector::BoolectorConfig;
        return Ok(Box::new(BoolectorConfig::new()));
    }
    #[cfg(all(
        feature = "has-easy-smt",
        not(feature = "has-bitwuzla"),
        not(feature = "has-boolector")
    ))]
    {
        use crate::solver::{
            Response, Solver,
            easy_smt::{EasySmtConfig, EasySmtSolver},
        };

        fn is_usable(config: &EasySmtConfig) -> bool {
            match EasySmtSolver::new(config) {
                Ok(mut solver) => {
                    if solver.declare("probe_x", 1).is_err() {
                        return false;
                    }
                    let one = solver.one(1);
                    let numerical_one = solver.numerical(1, 1);
                    let eq = solver.eq(&one, &numerical_one);
                    if solver.assert(&eq).is_err() {
                        return false;
                    }
                    matches!(solver.check(), Ok(Response::Sat))
                }
                Err(_) => false,
            }
        }

        for cfg in [
            EasySmtConfig::z3(),
            EasySmtConfig::boolector(),
            EasySmtConfig::bitwuzla(),
        ] {
            if is_usable(&cfg) {
                return Ok(Box::new(cfg));
            }
        }
    }

    Err(
        "No supported in-process SMT backend is available for dslx-fn-prove-assertions; rebuild with an in-process solver feature or pass an explicit supported --solver"
            .to_string(),
    )
}

fn prover_for_assertions(choice: SolverChoice) -> Result<Box<dyn Prover>, String> {
    match choice {
        SolverChoice::Auto => auto_prover_for_assertions(),
        SolverChoice::Toolchain => {
            Err("Solver 'toolchain' is not supported for dslx-fn-prove-assertions".to_string())
        }
        #[cfg(any(
            feature = "has-bitwuzla",
            feature = "has-boolector",
            feature = "has-easy-smt"
        ))]
        other => Ok(prover_for_choice(other, None)),
    }
}

/// Adds a boolean property function that invokes `top_name` and always returns
/// true, forcing assertion collection through the invoke dependency.
pub fn add_assertions_property_function(
    package: &mut ir::Package,
    top_name: &str,
) -> Result<String, String> {
    let top_fn = package
        .get_fn(top_name)
        .ok_or_else(|| format!("IR function '{}' not found", top_name))?;
    let property_name = format!("__assertions_property__{}", top_name);
    if package.get_fn(&property_name).is_some() {
        return Ok(property_name);
    }

    let mut nodes = vec![Node {
        text_id: 0,
        name: Some("reserved_zero_node".to_string()),
        ty: Type::nil(),
        payload: NodePayload::Nil,
        pos: None,
    }];
    let mut params = Vec::with_capacity(top_fn.params.len());
    let mut arg_refs = Vec::with_capacity(top_fn.params.len());
    let mut next_id = next_text_id(package);

    for param in &top_fn.params {
        let new_param = clone_param_with_new_id(param, next_id);
        let param_name = new_param.name.clone();
        let param_ty = new_param.ty.clone();
        let param_id = new_param.id;
        let param_ref = push_node(
            &mut nodes,
            &mut next_id,
            Some(param_name),
            param_ty,
            NodePayload::GetParam(param_id),
        );
        params.push(new_param);
        arg_refs.push(param_ref);
    }

    let invoke_ref = push_node(
        &mut nodes,
        &mut next_id,
        Some(format!("assertions_invoke__{}", top_name)),
        top_fn.ret_ty.clone(),
        NodePayload::Invoke {
            to_apply: top_fn.name.clone(),
            operands: arg_refs,
        },
    );
    let true_ref = push_node(
        &mut nodes,
        &mut next_id,
        Some("assertions_true".to_string()),
        Type::Bits(1),
        NodePayload::Literal(IrValue::make_ubits(1, 1).expect("make true literal")),
    );
    let pair_ty = Type::Tuple(vec![
        Box::new(top_fn.ret_ty.clone()),
        Box::new(Type::Bits(1)),
    ]);
    let pair_ref = push_node(
        &mut nodes,
        &mut next_id,
        Some("assertions_pair".to_string()),
        pair_ty,
        NodePayload::Tuple(vec![invoke_ref, true_ref]),
    );
    let ret_ref = push_node(
        &mut nodes,
        &mut next_id,
        Some("assertions_result".to_string()),
        Type::Bits(1),
        NodePayload::TupleIndex {
            tuple: pair_ref,
            index: 1,
        },
    );

    let property_fn = ir::Fn {
        name: property_name.clone(),
        params,
        ret_ty: Type::Bits(1),
        nodes,
        ret_node_ref: Some(ret_ref),
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
    };
    package.members.push(PackageMember::Function(property_fn));
    Ok(property_name)
}

/// Proves that all selected assertions reachable from a DSLX top hold.
pub fn run_dslx_assertions(
    request: &DslxAssertionsRequest<'_>,
) -> Result<DslxAssertionsReport, String> {
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
        .with_uf_map(request.uf_map.clone());

    let prover = prover_for_assertions(request.solver.unwrap_or(SolverChoice::Auto))?;
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
}
