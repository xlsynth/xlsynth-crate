// SPDX-License-Identifier: Apache-2.0

//! DSLX programs assembled from AST fragments with independent PIR semantics.

pub mod aggregate_flow;
pub mod aggregate_match;
mod aggregate_method_contracts;
mod array_cast_contracts;
pub mod array_fragment;
pub mod ast;
pub mod audit_contracts;
pub mod bit_projection;
mod bound_parametric_contracts;
mod builtin_contracts;
pub mod compound;
pub mod constexpr;
pub mod converter_contracts;
mod dead_branch_contracts;
mod empty_builtin_contracts;
mod enum_width_contracts;
mod generate;
pub mod imported_const;
pub mod imported_struct;
mod inferred_flow_contracts;
mod lambda_dimension_contracts;
mod literal_order_contracts;
pub mod map_fragment;
mod map_parametric_contracts;
pub mod mission_expr;
mod mixed_literal_contracts;
mod nested_scope_contracts;
mod nominal_contracts;
mod observable_value_contracts;
mod observer_contracts;
mod pattern_acceptance;
pub mod range_match;
mod scalar_update_builtin_contracts;
mod scope_contracts;
pub mod select_fold;
mod self_dimension_contracts;
mod signed_array_cast_contracts;
mod signed_call_contracts;
pub mod sum_pattern;
mod tuple_literal_contracts;
pub mod tuple_pattern;
mod unrolled_loop_contracts;
mod zero_width_cast_contracts;

pub use generate::{
    GenerateOptions, Grammar, generate, generate_with_grammar, generate_with_options,
};

use aggregate_flow::AggregateFlow;
use ast::{Body, HelperModule, ValueRef, Width};
use constexpr::{ConstExpr, metadata_bits};
use std::path::Path;
use xlsynth_pir::{FnBuilder, IrFormatPreference, IrValue, ir};

/// A deterministic source file in the generated translation environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceModule {
    pub name: String,
    pub text: String,
}

/// A checked module DAG and the independently observable lanes of its top.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    helpers: Vec<HelperModule>,
    lanes: Vec<Body>,
    // Mission generation selects one ordinary intermediate per lane.
    observations: Vec<ValueRef>,
    aggregate: Option<AggregateOutput>,
    specialization: Option<ConstExpr>,
}

/// The cross-helper flow and the lane bindings that supply its three words.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AggregateOutput {
    flow: AggregateFlow,
    operands: [ValueRef; 3],
}

/// The two IR programs to compare, together with portable DSLX reproducers.
#[derive(Debug)]
pub struct PreparedSample {
    pub modules: Vec<SourceModule>,
    pub expected_ir: String,
    pub converted_ir: String,
    pub converted_top: String,
}

/// Imports each available module and gives its exported struct a local alias.
fn render_imports(text: &mut String, helpers: &[HelperModule]) {
    for index in 0..helpers.len() {
        text.push_str(&format!("import m{index};\n"));
    }
    for (index, helper) in helpers.iter().enumerate() {
        if helper.struct_export().is_some() {
            text.push_str(&format!(
                "type FuzzImportedRow{index} = m{index}::FuzzRow;\n"
            ));
        }
    }
}

impl Sample {
    /// Validates widths and module dependencies before either rendering path.
    pub fn new(helpers: Vec<HelperModule>, lanes: Vec<Body>) -> Result<Self, String> {
        if lanes.is_empty() {
            return Err("sample must have at least one output lane".to_string());
        }
        for (index, module) in helpers.iter().enumerate() {
            module.validate_imports(&helpers[..index])?;
        }
        for lane in &lanes {
            if lane.bits_type().width == Width::Parametric {
                return Err("top-level lanes must have concrete widths".to_string());
            }
            lane.validate_imports(&helpers)?;
            if let Some(computation) = lane.computed_width() {
                let Width::Concrete(width) = lane.bits_type().width else {
                    return Err("computed width must have a concrete type".to_string());
                };
                if computation.expression.evaluate(&helpers, None)? != metadata_bits(width)? {
                    return Err(
                        "computed dimension does not match its reference bit width".to_string()
                    );
                }
            }
        }
        Ok(Self {
            helpers,
            lanes,
            observations: vec![],
            aggregate: None,
            specialization: None,
        })
    }

    /// Observes an ordinary call and a parametric specialization with the same
    /// constant operands as separate top outputs, each checked against PIR.
    pub fn with_specialization_check(mut self, expression: ConstExpr) -> Result<Self, String> {
        if expression.requires_bit_width() {
            return Err("specialization observation requires closed arguments".into());
        }
        expression.validate_imports(&self.helpers)?;
        self.specialization = Some(expression);
        Ok(self)
    }

    pub fn helpers(&self) -> &[HelperModule] {
        &self.helpers
    }

    pub fn lanes(&self) -> &[Body] {
        &self.lanes
    }

    /// Orders each lane's result and optional intermediate in the top tuple.
    fn lane_outputs(&self, index: usize) -> impl Iterator<Item = ValueRef> {
        std::iter::once(self.lanes[index].result()).chain(self.observations.get(index).copied())
    }

    /// Formats the AST as helper files followed by the importing entry module.
    pub fn sources(&self) -> Result<Vec<SourceModule>, String> {
        let mut modules = Vec::new();
        let mut runtime_assertions = false;
        for (index, module) in self.helpers.iter().enumerate() {
            let mut text = "// SPDX-License-Identifier: Apache-2.0\n\n".to_string();
            render_imports(&mut text, &self.helpers[..index]);
            let declarations = module.declarations();
            runtime_assertions |= declarations.runtime_assertions;
            text.push_str(&declarations.to_dslx());
            if let Some(export) = module.struct_export() {
                text.push_str(&export.declaration());
            }
            if let Some(exports) = module.exports() {
                text.push_str(&exports.declarations());
            }
            for signed in [false, true] {
                text.push_str(
                    &module
                        .body(signed)
                        .render_const_functions("", &self.helpers)?,
                );
            }
            if let Some(default) = module.parametric_default() {
                text.push_str(
                    &default
                        .expression
                        .render_definition("default_", &self.helpers)?,
                );
            }
            for signed in [false, true] {
                let body = module.body(signed);
                let ty = body.bits_type().to_dslx();
                let suffix = if signed { "s" } else { "u" };
                let default = if let Some(default) = module.parametric_default() {
                    format!(
                        ", K: u32 = {{ {} }}",
                        default.expression.to_dslx("default_", &self.helpers)?
                    )
                } else {
                    String::new()
                };
                text.push_str(&format!(
                    "\npub fn apply_{suffix}<N: u32{default}>(v0: {ty}, v1: {ty}) -> {ty} {{\n"
                ));
                text.push_str(&body.render_bindings("", &self.helpers)?);
                let result = if let Some(default) = module.parametric_default() {
                    default.combine.to_dslx(
                        &body.result_name(""),
                        &format!("(K as {ty})"),
                        body.bits_type(),
                    )
                } else {
                    body.result_name("")
                };
                text.push_str(&format!("    {result}\n}}\n"));
            }
            if let Some(aggregate) = &self.aggregate {
                text.push_str(&aggregate.flow.helper_declarations(index));
            }
            modules.push(SourceModule {
                name: format!("m{index}"),
                text,
            });
        }
        let mut text = "// SPDX-License-Identifier: Apache-2.0\n\n".to_string();
        render_imports(&mut text, &self.helpers);
        let mut declarations = compound::Declarations::default();
        for lane in &self.lanes {
            declarations.extend(lane.declarations());
        }
        if let Some(expr) = &self.specialization {
            declarations.extend(expr.declarations());
        }
        text.push_str(&declarations.to_dslx());
        if let Some(expr) = &self.specialization {
            text.push_str(&expr.render_function("observation_", &self.helpers)?);
            let ty = expr.bits_type().to_dslx();
            text.push_str(&format!("\nfn specialized<V: {ty}>() -> {ty} {{ V }}\n"));
        }
        for (index, lane) in self.lanes.iter().enumerate() {
            text.push_str(&lane.render_const_functions(&format!("a{index}_"), &self.helpers)?);
        }
        let mut params = Vec::new();
        let mut return_types = Vec::new();
        let mut results = Vec::new();
        for (index, lane) in self.lanes.iter().enumerate() {
            let prefix = format!("a{index}_");
            text.push_str(&lane.render_module_items(&prefix, &self.helpers)?);
            let ty = lane.render_type(&prefix);
            params.push(format!("a{index}_v0: {ty}, a{index}_v1: {ty}"));
            for output in self.lane_outputs(index) {
                return_types.push(ty.clone());
                results.push(lane.value_name(output, &prefix));
            }
        }
        if let Some(aggregate) = &self.aggregate {
            return_types.push(aggregate.flow.output_type());
        }
        if let Some(expr) = &self.specialization {
            return_types.extend([expr.bits_type().to_dslx(), expr.bits_type().to_dslx()]);
        }
        // Fallible functions use an implicit token. A public entry guarantees
        // that full-module conversion also emits its ordinary calling wrapper.
        let visibility = if runtime_assertions || declarations.runtime_assertions {
            "pub "
        } else {
            ""
        };
        text.push_str(&format!(
            "\n{visibility}fn main({}) -> ({},) {{\n",
            params.join(", "),
            return_types.join(", ")
        ));
        for (index, lane) in self.lanes.iter().enumerate() {
            text.push_str(&lane.render_bindings(&format!("a{index}_"), &self.helpers)?);
        }
        if let Some(aggregate) = &self.aggregate {
            let index = aggregate.flow.lane();
            let prefix = format!("a{index}_");
            let args = aggregate
                .operands
                .map(|v| self.lanes[index].value_name(v, &prefix));
            let expr = aggregate
                .flow
                .caller_expression(args.each_ref().map(String::as_str));
            text.push_str(&format!("    let aggregate_output = {expr};\n"));
            results.push("aggregate_output".to_string());
        }
        if let Some(expr) = &self.specialization {
            let call = expr.call("observation_");
            results.push(call.clone());
            results.push(format!("specialized<{{{call}}}>()"));
        }
        text.push_str(&format!("    ({},)\n}}\n", results.join(", ")));
        modules.push(SourceModule {
            name: "sample".to_string(),
            text,
        });
        Ok(modules)
    }

    /// Adds a nullary entry that interprets every supplied argument vector.
    ///
    /// Returning all results prevents dead calls from disappearing. The wrapper
    /// adds only calls, tuples, and typed literals to the generated
    /// environment; it performs no IR conversion and uses no fixed-width
    /// value extraction.
    pub fn interpreter_sources(
        &self,
        argument_sets: &[Vec<IrValue>],
    ) -> Result<Vec<SourceModule>, String> {
        if argument_sets.is_empty() {
            return Err("interpreter entry requires at least one argument vector".to_string());
        }
        let result_type = format!(
            "({},)",
            self.lanes
                .iter()
                .enumerate()
                .flat_map(|(index, lane)| self
                    .lane_outputs(index)
                    .map(|_| lane.bits_type().to_dslx()))
                .chain(self.aggregate.iter().map(|a| a.flow.output_type()))
                .chain(
                    self.specialization
                        .iter()
                        .flat_map(|expr| [expr.bits_type().to_dslx(), expr.bits_type().to_dslx()])
                )
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut calls = Vec::new();
        for args in argument_sets {
            if args.len() != self.lanes.len() * 2 {
                return Err("interpreter argument count does not match the sample".to_string());
            }
            let mut literals = Vec::new();
            for (lane, args) in self.lanes.iter().zip(args.as_chunks::<2>().0) {
                for value in args {
                    let IrValue::Bits(bits) = value else {
                        return Err("interpreter parameters must be bits values".to_string());
                    };
                    if lane.bits_type().width != Width::Concrete(bits.get_bit_count()) {
                        return Err(
                            "interpreter argument width does not match the sample".to_string()
                        );
                    }
                    literals.push(format!(
                        "{}:{}",
                        lane.bits_type().to_dslx(),
                        bits.to_string_fmt(IrFormatPreference::Hex, false)
                    ));
                }
            }
            calls.push(format!("main({})", literals.join(", ")));
        }
        let mut sources = self.sources()?;
        let entry = sources.last_mut().ok_or("missing entry module")?;
        entry.text.push_str(&format!(
            "\nfn coverage_entry() -> ({},) {{\n    ({},)\n}}\n",
            vec![result_type; calls.len()].join(", "),
            calls.join(", ")
        ));
        Ok(sources)
    }

    /// Constructs the composite reference function without converting DSLX.
    pub fn expected_ir(&self) -> Result<ir::Package, String> {
        let mut builder = FnBuilder::new("main");
        let mut args = Vec::new();
        for (index, lane) in self.lanes.iter().enumerate() {
            let Width::Concrete(width) = lane.bits_type().width else {
                return Err("top-level lane has no concrete width".to_string());
            };
            let x = builder.param(&format!("a{index}_v0"), ir::Type::Bits(width));
            let x = x.map_err(|error| error.to_string())?;
            let y = builder.param(&format!("a{index}_v1"), ir::Type::Bits(width));
            let y = y.map_err(|error| error.to_string())?;
            args.push([x, y]);
        }
        let mut outputs = Vec::new();
        let mut aggregate_output = None;
        for ((index, lane), args) in self.lanes.iter().enumerate().zip(args) {
            let Width::Concrete(width) = lane.bits_type().width else {
                return Err("top-level lane has no concrete width".to_string());
            };
            let values = lane
                .emit_values(&mut builder, args, width, &self.helpers)
                .map_err(|error| error.to_string())?;
            outputs.extend(self.lane_outputs(index).map(|output| values[output.0]));
            if let Some(aggregate) = &self.aggregate
                && aggregate.flow.lane() == index
            {
                let args = aggregate.operands.map(|v| values[v.0]);
                aggregate_output = Some(
                    aggregate
                        .flow
                        .emit_ir(&mut builder, args)
                        .map_err(|e| e.to_string())?,
                );
            }
        }
        outputs.extend(aggregate_output);
        if let Some(expr) = &self.specialization {
            let expected = expr.emit_ir(&mut builder, &self.helpers, None)?;
            outputs.extend([expected, expected]);
        }
        let result = builder.tuple(&outputs).map_err(|error| error.to_string())?;
        builder
            .build_package(result, "expected")
            .map_err(|error| error.to_string())
    }

    /// Materializes the module environment and invokes the real DSLX converter.
    ///
    /// Compiler rejection is an error: generated ASTs are valid by
    /// construction. The returned sources and IR remain available after
    /// temporary files vanish.
    pub fn prepare(&self) -> Result<PreparedSample, String> {
        let modules = self.sources()?;
        let expected_ir = self.expected_ir()?.to_string();
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        for module in &modules {
            std::fs::write(
                directory.path().join(format!("{}.x", module.name)),
                &module.text,
            )
            .map_err(|error| error.to_string())?;
        }
        let entry = modules.last().ok_or("missing entry module")?;
        let options = xlsynth::DslxConvertOptions {
            additional_search_paths: vec![directory.path()],
            ..Default::default()
        };
        // The entry path is portable; imported files are resolved in the
        // temporary directory and their source metadata is remapped below.
        let converted =
            xlsynth::convert_dslx_to_ir_text(&entry.text, Path::new("sample.x"), &options)
                .map_err(|error| error.to_string())?;
        let converted_top =
            xlsynth::mangle_dslx_name("sample", "main").map_err(|error| error.to_string())?;
        let mut converted_package = xlsynth_pir::ir_parser::Parser::new(&converted.ir)
            .parse_package()
            .map_err(|error| error.to_string())?;
        for path in converted_package.file_table.id_to_path.values_mut() {
            if let Ok(relative) = Path::new(path).strip_prefix(directory.path()) {
                *path = relative.to_string_lossy().into_owned();
            }
        }
        Ok(PreparedSample {
            modules,
            expected_ir,
            converted_ir: converted_package.to_string(),
            converted_top,
        })
    }
}
