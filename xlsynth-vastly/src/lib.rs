// SPDX-License-Identifier: Apache-2.0

//! Verilog(-ish) expression evaluator with 4-state values (0/1/X/Z).
//!
//! v1 focuses on a small operator subset centered on the ternary operator.
//! Verilog/SystemVerilog semantics are defined by the language standard;
//! external simulators are used here as differential reference
//! implementations rather than as spec providers.

pub mod ast;
pub mod ast_spanned;
mod combo_compile;
mod combo_eval;
mod combo_harness;
mod coverability;
mod coverage;
mod coverage_render2;
mod eval;
mod irvals;
mod iverilog_combo;
mod lexer;
mod lexer_spanned;
mod module_compile;
mod module_eval;
pub mod parser;
mod parser_spanned;
mod pipeline_compile;
mod pipeline_harness;
mod reference_sim;
mod sim_harness;
mod sim_observer;
mod sv_ast;
mod sv_lexer;
mod sv_parser;
mod value;
mod vcd;
mod vcd_diff;
mod vcd_writer;
mod yosys_cxxrtl_combo;

pub use crate::ast_spanned::SpannedExpr;
pub use crate::ast_spanned::SpannedExprKind;
pub use crate::combo_compile::CompiledComboModule;
pub use crate::combo_compile::Port;
pub use crate::combo_compile::PortDir;
pub use crate::combo_compile::compile_combo_module;
pub use crate::combo_eval::ComboEvalPlan;
pub use crate::combo_eval::eval_combo;
pub use crate::combo_eval::eval_combo_seeded;
pub use crate::combo_eval::eval_combo_seeded_with_coverage;
pub use crate::combo_eval::plan_combo_eval;
pub use crate::combo_harness::run_combo_and_write_vcd;
pub use crate::coverability::CoverabilityMap;
pub use crate::coverability::LineCoverability;
pub use crate::coverability::compute_coverability;
pub use crate::coverability::compute_coverability_or_fallback;
pub use crate::coverability::compute_coverability_or_fallback_with_defines;
pub use crate::coverability::compute_coverability_with_defines;
pub use crate::coverage::CoverageCounters;
pub use crate::coverage::SourceText;
pub use crate::coverage::SpanKey;
pub use crate::coverage_render2::render_annotated_source;
pub use crate::eval::EvalObserver;
pub use crate::eval::eval_expr;
pub use crate::irvals::cycles_from_irvals_file;
pub use crate::iverilog_combo::run_iverilog_combo_and_collect_vcd;
pub use crate::module_compile::CompiledModule;
pub use crate::module_compile::State;
pub use crate::module_compile::compile_module;
pub use crate::module_eval::step_module_with_env;
pub use crate::parser_spanned::parse_expr_spanned;
pub use crate::pipeline_compile::CompiledPipelineModule;
pub use crate::pipeline_compile::FunctionArmMeta;
pub use crate::pipeline_compile::FunctionMeta;
pub use crate::pipeline_compile::compile_pipeline_module;
pub use crate::pipeline_compile::compile_pipeline_module_with_defines;
pub use crate::pipeline_harness::PipelineCycle;
pub use crate::pipeline_harness::PipelineStimulus;
pub use crate::pipeline_harness::run_pipeline_and_collect_coverage;
pub use crate::pipeline_harness::run_pipeline_and_collect_outputs;
pub use crate::pipeline_harness::run_pipeline_and_write_vcd;
pub use crate::pipeline_harness::step_pipeline_state_with_env;
pub use crate::reference_sim::ReferenceSimCapabilities;
pub use crate::reference_sim::ReferenceSimKind;
pub use crate::reference_sim::ValueDomain;
pub use crate::reference_sim::env_is_two_value_safe;
pub use crate::reference_sim::expr_is_two_value_safe;
pub use crate::sim_harness::Cycle;
pub use crate::sim_harness::Stimulus;
pub use crate::sim_harness::run_and_write_vcd;
pub use crate::sim_observer::SimObserver;
pub use crate::value::LogicBit;
pub use crate::value::Signedness;
pub use crate::value::Value4;
pub use crate::vcd::Vcd;
pub use crate::vcd_diff::VcdDiffOptions;
pub use crate::vcd_diff::diff_vcd_exact;
pub use crate::yosys_cxxrtl_combo::eval_yosys_cxxrtl_combo;
pub use crate::yosys_cxxrtl_combo::has_yosys_cxxrtl_toolchain;

use std::collections::BTreeMap;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Lex(String),
    Parse(String),
    UnknownIdentifier(String),
}

/// Evaluation environment: identifier -> value.
#[derive(Debug, Default, Clone)]
pub struct Env {
    vars: BTreeMap<String, Value4>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            vars: BTreeMap::new(),
        }
    }

    pub fn insert<S: Into<String>>(&mut self, name: S, value: Value4) {
        self.vars.insert(name.into(), value);
    }

    pub fn get(&self, name: &str) -> Option<&Value4> {
        self.vars.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value4)> {
        self.vars.iter()
    }
}

/// Result of evaluating an expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalResult {
    pub value: Value4,
}
