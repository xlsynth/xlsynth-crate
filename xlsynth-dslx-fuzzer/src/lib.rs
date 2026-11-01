// SPDX-License-Identifier: Apache-2.0

//! DSLX programs assembled from AST fragments with independent PIR semantics.

pub mod ast;
pub mod compound;
pub mod constexpr;
mod generate;
pub mod tuple_pattern;

pub use generate::{GenerateOptions, generate, generate_with_options};

use ast::{Body, HelperModule, Width};
use constexpr::metadata_bits;
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
}

/// The two IR programs to compare, together with portable DSLX reproducers.
#[derive(Debug)]
pub struct PreparedSample {
    pub modules: Vec<SourceModule>,
    pub expected_ir: String,
    pub converted_ir: String,
    pub converted_top: String,
}

impl Sample {
    /// Validates widths and module dependencies before either rendering path.
    pub fn new(helpers: Vec<HelperModule>, lanes: Vec<Body>) -> Result<Self, String> {
        if lanes.is_empty() {
            return Err("sample must have at least one output lane".to_string());
        }
        for (index, module) in helpers.iter().enumerate() {
            module.validate_calls(index)?;
        }
        for lane in &lanes {
            if lane.bits_type().width == Width::Parametric {
                return Err("top-level lanes must have concrete widths".to_string());
            }
            lane.validate_calls(helpers.len())?;
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
        Ok(Self { helpers, lanes })
    }

    pub fn helpers(&self) -> &[HelperModule] {
        &self.helpers
    }

    pub fn lanes(&self) -> &[Body] {
        &self.lanes
    }

    /// Formats the AST as helper files followed by the importing entry module.
    pub fn sources(&self) -> Result<Vec<SourceModule>, String> {
        let mut modules = Vec::new();
        for (index, module) in self.helpers.iter().enumerate() {
            let mut text = "// SPDX-License-Identifier: Apache-2.0\n\n".to_string();
            for dependency in 0..index {
                text.push_str(&format!("import m{dependency};\n"));
            }
            text.push_str(&module.declarations().to_dslx());
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
            modules.push(SourceModule {
                name: format!("m{index}"),
                text,
            });
        }
        let mut text = "// SPDX-License-Identifier: Apache-2.0\n\n".to_string();
        for index in 0..self.helpers.len() {
            text.push_str(&format!("import m{index};\n"));
        }
        let mut declarations = compound::Declarations::default();
        for lane in &self.lanes {
            declarations.extend(lane.declarations());
        }
        text.push_str(&declarations.to_dslx());
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
            return_types.push(ty);
            results.push(lane.result_name(&format!("a{index}_")));
        }
        text.push_str(&format!(
            "\nfn main({}) -> ({},) {{\n",
            params.join(", "),
            return_types.join(", ")
        ));
        for (index, lane) in self.lanes.iter().enumerate() {
            text.push_str(&lane.render_bindings(&format!("a{index}_"), &self.helpers)?);
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
                .map(|lane| lane.bits_type().to_dslx())
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
        for (lane, args) in self.lanes.iter().zip(args) {
            let Width::Concrete(width) = lane.bits_type().width else {
                return Err("top-level lane has no concrete width".to_string());
            };
            outputs.push(
                lane.emit_ir(&mut builder, args, width, &self.helpers)
                    .map_err(|error| error.to_string())?,
            );
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
