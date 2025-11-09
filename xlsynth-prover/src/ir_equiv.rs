// SPDX-License-Identifier: Apache-2.0

//! Library helpers for IR equivalence flows shared by multiple driver commands.

use crate::ir_utils;
use crate::prover::types::{
    AssertionSemantics, EquivParallelism, EquivReport, EquivResult, ParamDomains, ProverFn,
};
use crate::prover::{SolverChoice, prover_for_choice};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Description of a single IR module participating in equivalence.
#[derive(Clone)]
pub struct IrModule<'a> {
    pub source: &'a str,
    pub path: Option<&'a Path>,
    pub top: Option<&'a str>,
    pub param_domains: Option<&'a ParamDomains>,
    pub uf_map: Cow<'a, HashMap<String, String>>,
    pub fixed_implicit_activation: bool,
}

impl<'a> IrModule<'a> {
    /// Creates a new IR module description with default settings.
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            path: None,
            top: None,
            param_domains: None,
            uf_map: Cow::Owned(HashMap::new()),
            fixed_implicit_activation: false,
        }
    }

    /// Sets the top-level function name (if any) to compare.
    pub fn with_top(mut self, top: Option<&'a str>) -> Self {
        self.top = top;
        self
    }

    /// Supplies optional parameter domains for this module.
    pub fn with_param_domains(mut self, domains: Option<&'a ParamDomains>) -> Self {
        self.param_domains = domains;
        self
    }

    /// Sets the UF mapping to use when proving equivalence.
    pub fn with_uf_map(mut self, uf_map: &'a HashMap<String, String>) -> Self {
        self.uf_map = Cow::Borrowed(uf_map);
        self
    }

    /// Provides an owned UF mapping (used mostly in tests).
    pub fn with_owned_uf_map(mut self, uf_map: HashMap<String, String>) -> Self {
        self.uf_map = Cow::Owned(uf_map);
        self
    }

    /// Records the filesystem path the IR was loaded from (if any).
    pub fn with_path(mut self, path: Option<&'a Path>) -> Self {
        self.path = path;
        self
    }

    /// Specifies whether the implicit activation input should be fixed.
    pub fn with_fixed_implicit_activation(mut self, fixed: bool) -> Self {
        self.fixed_implicit_activation = fixed;
        self
    }
}

/// Request describing an IR equivalence proof.
#[derive(Clone)]
pub struct IrEquivRequest<'a> {
    pub lhs: IrModule<'a>,
    pub rhs: IrModule<'a>,
    pub drop_params: &'a [String],
    pub flatten_aggregates: bool,
    pub parallelism: EquivParallelism,
    pub assertion_semantics: AssertionSemantics,
    pub assert_label_filter: Option<&'a str>,
    pub solver: Option<SolverChoice>,
    pub tool_path: Option<&'a Path>,
}

impl<'a> IrEquivRequest<'a> {
    /// Creates a new request with default comparison settings.
    pub fn new(lhs: IrModule<'a>, rhs: IrModule<'a>) -> Self {
        Self {
            lhs,
            rhs,
            drop_params: &[],
            flatten_aggregates: false,
            parallelism: EquivParallelism::SingleThreaded,
            assertion_semantics: AssertionSemantics::Same,
            assert_label_filter: None,
            solver: None,
            tool_path: None,
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

    pub fn with_tool_path(mut self, tool_path: Option<&'a Path>) -> Self {
        self.tool_path = tool_path;
        self
    }
}

/// Dispatches an IR equivalence comparison using the requested solver.
pub fn run_ir_equiv(request: &IrEquivRequest<'_>) -> Result<EquivReport, String> {
    let choice = request.solver.unwrap_or(SolverChoice::Auto);
    let prover = prover_for_choice(choice, request.tool_path);
    let (lhs_pkg, lhs_fn_dropped) = ir_utils::parse_package_and_drop_params(
        request.lhs.source,
        request.lhs.top,
        request.drop_params,
    )?;
    let (rhs_pkg, rhs_fn_dropped) = ir_utils::parse_package_and_drop_params(
        request.rhs.source,
        request.rhs.top,
        request.drop_params,
    )?;

    let assert_label_filter = request.assert_label_filter;

    let lhs_side = ProverFn::new(&lhs_fn_dropped, Some(&lhs_pkg))
        .with_fixed_implicit_activation(request.lhs.fixed_implicit_activation)
        .with_domains(request.lhs.param_domains.map(|d| d.clone()))
        .with_uf_map(request.lhs.uf_map.clone().into_owned());
    let rhs_side = ProverFn::new(&rhs_fn_dropped, Some(&rhs_pkg))
        .with_fixed_implicit_activation(request.rhs.fixed_implicit_activation)
        .with_domains(request.rhs.param_domains.map(|d| d.clone()))
        .with_uf_map(request.rhs.uf_map.clone().into_owned());

    let start_time = Instant::now();
    let result = prover.prove_ir_equiv(
        &lhs_side,
        &rhs_side,
        request.parallelism,
        request.assertion_semantics,
        assert_label_filter,
        request.flatten_aggregates,
    );
    let duration = start_time.elapsed();

    match result {
        EquivResult::Error(msg) => Err(msg),
        other => Ok(EquivReport {
            duration,
            result: other,
        }),
    }
}
