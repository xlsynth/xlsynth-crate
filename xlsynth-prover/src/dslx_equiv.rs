// SPDX-License-Identifier: Apache-2.0

//! Library entry point for running equivalence checks between two DSLX modules.
//!
//! The driver crate parses CLI arguments and reads DSLX files; this module
//! performs the remaining work: converting source text to IR, preparing UF
//! metadata, selecting a solver, and dispatching the equivalence proof.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xlsynth::IrValue;
use xlsynth::dslx::{self, MatchableModuleMember, TypecheckedModule};
use xlsynth::{DslxConvertOptions, IrPackage, mangle_dslx_name, optimize_ir};

use crate::ir_utils;
use crate::prover::{SolverChoice, prover_for_choice};
use crate::toolchain;
use crate::types::{
    AssertionSemantics, EquivParallelism, EquivReport, EquivResult, IrFn, ParamDomains, ProverFn,
};

/// DSLX side description passed into [`run_dslx_equiv`].
pub struct DslxModule<'a> {
    pub source: &'a str,
    pub path: Option<&'a Path>,
    pub top: &'a str,
    pub uf_map: Cow<'a, HashMap<String, String>>,
    pub fixed_implicit_activation: bool,
}

impl<'a> DslxModule<'a> {
    pub fn new(source: &'a str, top: &'a str) -> Self {
        Self {
            source,
            path: None,
            top,
            uf_map: Cow::Owned(HashMap::new()),
            fixed_implicit_activation: false,
        }
    }

    pub fn with_uf_map(mut self, uf_map: &'a HashMap<String, String>) -> Self {
        self.uf_map = Cow::Borrowed(uf_map);
        self
    }

    pub fn with_path(mut self, path: Option<&'a Path>) -> Self {
        self.path = path;
        self
    }

    pub fn with_owned_uf_map(mut self, uf_map: HashMap<String, String>) -> Self {
        self.uf_map = Cow::Owned(uf_map);
        self
    }

    pub fn with_fixed_implicit_activation(mut self, fixed: bool) -> Self {
        self.fixed_implicit_activation = fixed;
        self
    }
}

#[derive(Clone)]
pub struct DslxOptions<'a> {
    pub tool_path: Option<&'a Path>,
    pub dslx_stdlib_path: Option<&'a Path>,
    pub additional_search_paths: Vec<PathBuf>,
    pub enable_warnings: Option<&'a [String]>,
    pub disable_warnings: Option<&'a [String]>,
    pub type_inference_v2: Option<bool>,
    pub optimize: bool,
}

impl<'a> Default for DslxOptions<'a> {
    fn default() -> Self {
        Self {
            tool_path: None,
            dslx_stdlib_path: None,
            additional_search_paths: Vec::new(),
            enable_warnings: None,
            disable_warnings: None,
            type_inference_v2: None,
            optimize: false,
        }
    }
}

impl<'a> DslxOptions<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tool_path(mut self, tool_path: Option<&'a Path>) -> Self {
        self.tool_path = tool_path;
        self
    }

    pub fn with_dslx_stdlib_path(mut self, path: Option<&'a Path>) -> Self {
        self.dslx_stdlib_path = path;
        self
    }

    pub fn with_additional_search_paths<I, P>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        self.additional_search_paths = paths.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_enable_warnings(mut self, warnings: Option<&'a [String]>) -> Self {
        self.enable_warnings = warnings;
        self
    }

    pub fn with_disable_warnings(mut self, warnings: Option<&'a [String]>) -> Self {
        self.disable_warnings = warnings;
        self
    }

    pub fn with_type_inference_v2(mut self, flag: Option<bool>) -> Self {
        self.type_inference_v2 = flag;
        self
    }

    pub fn with_optimize(mut self, optimize: bool) -> Self {
        self.optimize = optimize;
        self
    }
}

/// Options for a DSLX equivalence proof.
pub struct DslxEquivRequest<'a> {
    pub lhs: DslxModule<'a>,
    pub rhs: DslxModule<'a>,
    pub drop_params: &'a [String],
    pub flatten_aggregates: bool,
    pub parallelism: EquivParallelism,
    pub assertion_semantics: AssertionSemantics,
    pub assert_label_filter: Option<&'a str>,
    pub solver: Option<SolverChoice>,
    pub options: DslxOptions<'a>,
    pub assume_enum_in_bound: bool,
}

impl<'a> DslxEquivRequest<'a> {
    pub fn new(lhs: DslxModule<'a>, rhs: DslxModule<'a>) -> Self {
        Self {
            lhs,
            rhs,
            drop_params: &[],
            flatten_aggregates: false,
            parallelism: EquivParallelism::SingleThreaded,
            assertion_semantics: AssertionSemantics::Same,
            assert_label_filter: None,
            solver: None,
            options: DslxOptions::default(),
            assume_enum_in_bound: false,
        }
    }

    pub fn with_drop_params(mut self, drop_params: &'a [String]) -> Self {
        self.drop_params = drop_params;
        self
    }

    pub fn with_flatten_aggregates(mut self, flatten: bool) -> Self {
        self.flatten_aggregates = flatten;
        self
    }

    pub fn with_parallelism(mut self, parallelism: EquivParallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    pub fn with_assertion_semantics(mut self, semantics: AssertionSemantics) -> Self {
        self.assertion_semantics = semantics;
        self
    }

    pub fn with_assert_label_filter(mut self, filter: Option<&'a str>) -> Self {
        self.assert_label_filter = filter;
        self
    }

    pub fn with_solver(mut self, solver: Option<SolverChoice>) -> Self {
        self.solver = solver;
        self
    }

    pub fn with_assume_enum_in_bound(mut self, assume: bool) -> Self {
        self.assume_enum_in_bound = assume;
        self
    }

    pub fn with_optimize(mut self, optimize: bool) -> Self {
        self.options.optimize = optimize;
        self
    }

    pub fn with_tool_path(mut self, tool_path: Option<&'a Path>) -> Self {
        self.options.tool_path = tool_path;
        self
    }

    pub fn with_dslx_stdlib_path(mut self, path: Option<&'a Path>) -> Self {
        self.options.dslx_stdlib_path = path;
        self
    }

    pub fn with_additional_search_paths<I, P>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        self.options.additional_search_paths = paths.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_enable_warnings(mut self, warnings: Option<&'a [String]>) -> Self {
        self.options.enable_warnings = warnings;
        self
    }

    pub fn with_disable_warnings(mut self, warnings: Option<&'a [String]>) -> Self {
        self.options.disable_warnings = warnings;
        self
    }

    pub fn with_type_inference_v2(mut self, flag: Option<bool>) -> Self {
        self.options.type_inference_v2 = flag;
        self
    }
}

/// Proves equivalence between two DSLX modules described by `request`.
pub fn run_dslx_equiv(request: &DslxEquivRequest<'_>) -> Result<EquivReport, String> {
    validate_tool_options(request)?;

    let start = Instant::now();

    let lhs_ir = prepare_side(request, &request.lhs)?;
    let rhs_ir = prepare_side(request, &request.rhs)?;

    let (lhs_pkg, lhs_fn) = ir_utils::parse_package_and_drop_params(
        &lhs_ir.ir_text,
        Some(&lhs_ir.mangled_top),
        request.drop_params,
    )
    .map_err(|err| map_prepare_error(err, request.lhs.path))?;
    let (rhs_pkg, rhs_fn) = ir_utils::parse_package_and_drop_params(
        &rhs_ir.ir_text,
        Some(&rhs_ir.mangled_top),
        request.drop_params,
    )
    .map_err(|err| map_prepare_error(err, request.rhs.path))?;

    let lhs_ir_fn = IrFn {
        fn_ref: &lhs_fn,
        pkg_ref: Some(&lhs_pkg),
        fixed_implicit_activation: request.lhs.fixed_implicit_activation,
    };
    let rhs_ir_fn = IrFn {
        fn_ref: &rhs_fn,
        pkg_ref: Some(&rhs_pkg),
        fixed_implicit_activation: request.rhs.fixed_implicit_activation,
    };

    let prover = prover_for_choice(
        request.solver.unwrap_or(SolverChoice::Auto),
        request.options.tool_path,
    );
    let assert_label_filter = request.assert_label_filter;

    let lhs_side = ProverFn {
        ir_fn: &lhs_ir_fn,
        domains: lhs_ir.param_domains.clone(),
        uf_map: request.lhs.uf_map.clone().into_owned(),
    };
    let rhs_side = ProverFn {
        ir_fn: &rhs_ir_fn,
        domains: rhs_ir.param_domains.clone(),
        uf_map: request.rhs.uf_map.clone().into_owned(),
    };

    let result = prover.prove_ir_equiv(
        &lhs_side,
        &rhs_side,
        request.parallelism,
        request.assertion_semantics,
        assert_label_filter,
        request.flatten_aggregates,
    );

    let duration = start.elapsed();
    match result {
        EquivResult::Error(msg) => Err(msg),
        other => Ok(EquivReport {
            duration,
            result: other,
        }),
    }
}

struct PreparedIr {
    ir_text: String,
    mangled_top: String,
    param_domains: Option<ParamDomains>,
}

fn prepare_side(
    request: &DslxEquivRequest<'_>,
    module: &DslxModule<'_>,
) -> Result<PreparedIr, String> {
    let module_name = module_name(module)?;
    let mangled_top = mangle_dslx_name(module_name, module.top).map_err(|e| e.to_string())?;

    let options = &request.options;
    let want_enum_domains = request.assume_enum_in_bound;
    let additional_path_refs: Vec<&Path> = options
        .additional_search_paths
        .iter()
        .map(|p| p.as_path())
        .collect();
    let ir_text = if !want_enum_domains && options.tool_path.is_some() {
        let module_path = module.path.ok_or_else(|| {
            "--tool_path requires DSLX modules to carry a filesystem path".to_string()
        })?;
        let mut ir_text = toolchain::run_ir_converter_main(
            options.tool_path.unwrap(),
            module_path,
            Some(module.top),
            options.dslx_stdlib_path,
            &additional_path_refs,
            options.enable_warnings,
            options.disable_warnings,
            options.type_inference_v2,
        )
        .map_err(|e| e)?;
        if options.optimize {
            ir_text = toolchain::run_opt_main(options.tool_path.unwrap(), &ir_text, &mangled_top)
                .map_err(|e| e)?;
        }
        ir_text
    } else {
        if options.type_inference_v2 == Some(true) {
            return Err("--type_inference_v2 requires external toolchain support".to_string());
        }
        let origin_path = module.path.unwrap_or_else(|| Path::new("<inline.dslx>"));
        let result = xlsynth::convert_dslx_to_ir_text(
            module.source,
            origin_path,
            &DslxConvertOptions {
                dslx_stdlib_path: options.dslx_stdlib_path,
                additional_search_paths: additional_path_refs.clone(),
                enable_warnings: options.enable_warnings,
                disable_warnings: options.disable_warnings,
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;
        if options.optimize {
            let pkg = IrPackage::parse_ir(&result.ir, Some(&mangled_top))
                .map_err(|e| format!("Failed to parse IR for optimization: {}", e))?;
            optimize_ir(&pkg, &mangled_top)
                .map_err(|e| e.to_string())?
                .to_string()
        } else {
            result.ir
        }
    };

    let param_domains = if want_enum_domains {
        Some(compute_enum_param_domains(
            module.source,
            module.path,
            module_name,
            module.top,
            options.dslx_stdlib_path,
            &additional_path_refs,
        )?)
    } else {
        None
    };

    Ok(PreparedIr {
        ir_text,
        mangled_top,
        param_domains,
    })
}

fn validate_tool_options(request: &DslxEquivRequest<'_>) -> Result<(), String> {
    if request.options.type_inference_v2.is_some() && request.options.tool_path.is_none() {
        return Err("--type_inference_v2 requires --tool_path to be set".to_string());
    }
    Ok(())
}

fn module_name<'a>(module: &'a DslxModule<'a>) -> Result<&'a str, String> {
    if let Some(path) = module.path {
        path.file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("Failed to derive module name from path: {}", path.display()))
    } else {
        Ok(module.top)
    }
}

fn map_prepare_error(msg: String, origin: Option<&Path>) -> String {
    match origin {
        Some(path) => format!("{} (input: {})", msg, path.display()),
        None => msg,
    }
}

fn compute_enum_param_domains(
    source: &str,
    path: Option<&Path>,
    module_name: &str,
    top: &str,
    dslx_stdlib_path: Option<&Path>,
    additional_search_paths: &[&Path],
) -> Result<ParamDomains, String> {
    let mut import_data = dslx::ImportData::new(dslx_stdlib_path, additional_search_paths);
    let origin = path
        .and_then(|p| p.to_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| "<inline>".to_string());
    let tcm = dslx::parse_and_typecheck(source, &origin, module_name, &mut import_data)
        .map_err(|e| format!("Typecheck failed: {}", e))?;
    get_function_enum_param_domains(&tcm, top)
}

fn get_function_enum_param_domains(
    tcm: &TypecheckedModule,
    dslx_top: &str,
) -> Result<ParamDomains, String> {
    let module = tcm.get_module();
    let type_info = tcm.get_type_info();
    let mut domains: ParamDomains = HashMap::new();
    let mut found = false;

    for idx in 0..module.get_member_count() {
        if let Some(MatchableModuleMember::Function(f)) = module.get_member(idx).to_matchable() {
            if f.get_identifier() == dslx_top {
                found = true;
                for pidx in 0..f.get_param_count() {
                    let param = f.get_param(pidx);
                    let name = param.get_name();
                    let type_annotation = param.get_type_annotation();
                    let ty = type_info.get_type_for_type_annotation(&type_annotation);
                    if ty.is_enum() {
                        let enum_def = ty.get_enum_def().expect("enum def");
                        let values = collect_enum_values(tcm, &enum_def);
                        domains.insert(name, values);
                    }
                }
            }
        }
    }

    if found {
        Ok(domains)
    } else {
        Err(format!(
            "Function '{}' not found in module '{}'",
            dslx_top,
            module.get_name()
        ))
    }
}

fn collect_enum_values(tcm: &TypecheckedModule, enum_def: &xlsynth::dslx::EnumDef) -> Vec<IrValue> {
    let mut values = Vec::new();
    for idx in 0..enum_def.get_member_count() {
        let member = enum_def.get_member(idx);
        let expr = member.get_value();
        let owner_module = expr.get_owner_module();
        let owner_type_info = tcm
            .get_type_info_for_module(&owner_module)
            .expect("imported type info");
        let interp = owner_type_info.get_const_expr(&expr).expect("constexpr");
        let ir_value = interp.convert_to_ir().expect("convert to IR");
        values.push(ir_value);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn parallelism_enum_matches_variants() {
        assert_eq!(EquivParallelism::SingleThreaded as u8, 0);
        assert_eq!(EquivParallelism::OutputBits as u8, 1);
        assert_eq!(EquivParallelism::InputBitSplit as u8, 2);
    }

    #[test]
    fn proves_simple_identity_modules() {
        let tmpdir = tempfile::tempdir().expect("temp dir");
        let lhs_path = tmpdir.path().join("lhs.x");
        let rhs_path = tmpdir.path().join("rhs.x");
        let dslx = "pub fn f(x: u32) -> u32 { x }";
        std::fs::write(&lhs_path, dslx).expect("write lhs");
        std::fs::write(&rhs_path, dslx).expect("write rhs");

        let lhs_source = std::fs::read_to_string(&lhs_path).expect("lhs text");
        let rhs_source = std::fs::read_to_string(&rhs_path).expect("rhs text");
        let lhs_uf_map: HashMap<String, String> = HashMap::new();
        let rhs_uf_map: HashMap<String, String> = HashMap::new();
        let drop_params: Vec<String> = Vec::new();

        let lhs_module = DslxModule::new(&lhs_source, "f")
            .with_path(Some(lhs_path.as_path()))
            .with_uf_map(&lhs_uf_map);
        let rhs_module = DslxModule::new(&rhs_source, "f")
            .with_path(Some(rhs_path.as_path()))
            .with_uf_map(&rhs_uf_map);

        let request = DslxEquivRequest::new(lhs_module, rhs_module)
            .with_drop_params(&drop_params)
            .with_parallelism(EquivParallelism::SingleThreaded)
            .with_assertion_semantics(AssertionSemantics::Same)
            .with_solver(Some(SolverChoice::Auto))
            .with_assume_enum_in_bound(true)
            .with_optimize(true);

        let report = run_dslx_equiv(&request).expect("prove success");
        assert!(matches!(report.result, EquivResult::Proved));
    }
}
