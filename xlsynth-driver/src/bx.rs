// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, ArgMatches, Command, value_parser};
use xlsynth::dslx;
use xlsynth_bx::{ImportOptions, ResetBehavior, compile_with_options, empty_package};
use xlsynth_g8r::block2sequential::optimize_and_lower_bx_block;
use xlsynth_g8r::process_ir_path::CanonicalG8rOptions;
use xlsynth_lean::block;
use xlsynth_pir::ir::{MemberType, Package, PackageMember};
use xlsynth_pir::ir_parser::parse_xls_ir_package;
use xlsynth_pir::ir_utils::is_valid_package_name;

use crate::common::write_stdout;
use crate::toolchain_config::{ToolchainConfig, get_dslx_search_paths, get_dslx_stdlib_path};
use crate::{g8r2gv, ir2gates};

/// Adds the source, linking, and reset options shared by the BX commands.
fn add_bx_args(command: Command) -> Command {
    let mut command = command
        .arg(
            Arg::new("input")
                .value_name("BX_FILE")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("top")
                .long("top")
                .value_name("BLOCK")
                .help("Select the top block (defaults to the sole BX block or unique pub block)"),
        )
        .arg(
            Arg::new("link-ir")
                .long("link-ir")
                .value_name("PACKAGE")
                .value_parser(value_parser!(PathBuf))
                .help("IR package providing implementations for BX extern declarations"),
        )
        .arg(
            Arg::new("dslx_path")
                .long("dslx-path")
                .visible_alias("dslx_path")
                .value_name("DIRS")
                .help("Semicolon-separated DSLX import directories (overrides the toolchain path)"),
        )
        .arg(
            Arg::new("dslx_stdlib_path")
                .long("dslx-stdlib-path")
                .visible_alias("dslx_stdlib_path")
                .value_name("DIR")
                .help("Override the DSLX standard library directory"),
        );
    for (name, help) in [
        (
            "reset-active-low",
            "Use active-low reset (default active-high)",
        ),
        (
            "reset-asynchronous",
            "Use asynchronous reset (default synchronous)",
        ),
    ] {
        command = command.arg(
            Arg::new(name)
                .long(name)
                .help(help)
                .num_args(0..=1)
                .require_equals(true)
                .default_missing_value("true")
                .default_value("false")
                .value_parser(value_parser!(bool)),
        );
    }
    command
}

pub fn bx2ir_command() -> Command {
    add_bx_args(Command::new("bx2ir").about("Compiles BX source to an XLS block IR package"))
}

pub fn bx2lean_command() -> Command {
    let command = add_bx_args(Command::new("bx2lean").about("Generates raw and readable Lean for a BX block"))
        .arg(
            Arg::new("link-dslx")
                .long("link-dslx")
                .value_name("MODULE.x")
                .conflicts_with("link-ir")
                .help("Compile a DSLX module to implement BX extern functions and retain their declared types"),
        )
        .arg(
            Arg::new("output-dir")
                .long("output-dir")
                .value_name("NEW_DIR")
                .value_parser(value_parser!(PathBuf))
                .required(true)
                .help("Create a new directory for the five Lean modules"),
        );
    #[cfg(feature = "dslx-loop-bindings")]
    let command = command.arg(
        Arg::new("readable-loop-names")
            .long("readable-loop-names")
            .action(clap::ArgAction::SetTrue)
            .help("Use checked DSLX loop bindings to name locals in the readable model"),
    );
    command
}

pub fn bx2gv_command() -> Command {
    g8r2gv::add_mapping_args(add_bx_args(
        Command::new("bx2gv")
            .about("Compiles BX source and maps it to a Liberty gate-level netlist"),
    ))
}

/// Marks the requested block, honoring the frontend's inferred top by default.
fn select_top(package: &mut Package, requested: Option<&str>) -> Result<()> {
    if let Some(name) = requested {
        return package.set_top_block(name).map_err(anyhow::Error::msg);
    }
    if matches!(package.top, Some((_, MemberType::Block))) {
        return Ok(());
    }
    let blocks: Vec<_> = package
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Block(block) => Some(block.name.as_str()),
            PackageMember::Function(_) => None,
        })
        .collect();
    match blocks.as_slice() {
        [only] => package.top = Some(((*only).to_owned(), MemberType::Block)),
        [] => bail!("package contains no top-level block"),
        _ => bail!(
            "multiple blocks ({}) without a selected top; specify --top",
            blocks.join(", ")
        ),
    }
    Ok(())
}

/// Supplies the chosen BX compiler with the same source, link package and
/// import paths for each command.
fn with_bx_input<T>(
    matches: &ArgMatches,
    config: &Option<ToolchainConfig>,
    compile: impl FnOnce(&str, Package, ResetBehavior, &ImportOptions<'_>) -> Result<T>,
) -> Result<T> {
    let path = matches
        .get_one::<PathBuf>("input")
        .expect("clap requires BX_FILE");
    let source = fs::read_to_string(path)
        .with_context(|| format!("failed to read BX file '{}'", path.display()))?;
    let package = if let Some(link) = matches.get_one::<PathBuf>("link-ir") {
        let ir = fs::read_to_string(link)
            .with_context(|| format!("failed to read linked IR file '{}'", link.display()))?;
        parse_xls_ir_package(&ir).map_err(|error| {
            anyhow!(
                "failed to parse linked IR file '{}': {error}",
                link.display()
            )
        })?
    } else {
        let name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|name| is_valid_package_name(name))
            .unwrap_or("design");
        empty_package(name)
    };
    let paths = get_dslx_search_paths(matches, config);
    let stdlib = get_dslx_stdlib_path(matches, config);
    let imports = ImportOptions {
        dslx_stdlib_path: stdlib.as_deref().map(Path::new),
        additional_search_paths: paths.iter().map(PathBuf::as_path).collect(),
    };
    let reset = ResetBehavior {
        active_low: matches.get_flag("reset-active-low"),
        asynchronous: matches.get_flag("reset-asynchronous"),
    };
    compile(&source, package, reset, &imports).with_context(|| path.display().to_string())
}

fn compile_input(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<Package> {
    let mut package = with_bx_input(matches, config, |source, package, reset, imports| {
        compile_with_options(source, package, reset, imports).map_err(anyhow::Error::msg)
    })?;
    select_top(
        &mut package,
        matches.get_one::<String>("top").map(String::as_str),
    )?;
    Ok(package)
}

pub fn bx2ir(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    write_stdout(&compile_input(matches, config)?.to_string());
    Ok(())
}

/// Generate and validate the complete model before creating its output
/// directory.
pub fn bx2lean(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    let bx_path = matches
        .get_one::<PathBuf>("input")
        .expect("clap requires BX_FILE");
    let bx_name = bx_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .context("BX input requires a UTF-8 module name")?;
    let compiled = with_bx_input(matches, config, |source, package, reset, imports| {
        let Some(path) = matches.get_one::<String>("link-dslx") else {
            #[cfg(feature = "dslx-loop-bindings")]
            if matches.get_flag("readable-loop-names") {
                return block::compile_bx_with_loop_bindings(
                    bx_name, source, package, reset, imports,
                );
            }
            return block::compile_bx(bx_name, source, package, reset, imports);
        };
        let dslx_source = fs::read_to_string(path)
            .with_context(|| format!("reading linked DSLX module '{path}'"))?;
        let name = xlsynth::dslx_path_to_module_name(Path::new(path))?;
        let mut import_data =
            dslx::ImportData::new(imports.dslx_stdlib_path, &imports.additional_search_paths);
        let module = dslx::parse_and_typecheck(&dslx_source, path, name, &mut import_data)
            .with_context(|| format!("checking linked DSLX module '{path}'"))?;
        #[cfg(feature = "dslx-loop-bindings")]
        if matches.get_flag("readable-loop-names") {
            return block::compile_bx_with_dslx_loop_bindings(
                bx_name, source, module, reset, imports,
            );
        }
        block::compile_bx_with_dslx(bx_name, source, module, reset, imports)
    })?;
    let top = matches.get_one::<String>("top").map(String::as_str);
    let bundle = compiled.emit_readable(top)?;
    let directory = matches
        .get_one::<PathBuf>("output-dir")
        .expect("clap requires NEW_DIR");
    fs::create_dir(directory)
        .with_context(|| format!("creating new output directory '{}'", directory.display()))?;
    for (name, text) in bundle.files {
        let path = directory.join(name);
        fs::write(&path, text)
            .with_context(|| format!("writing generated Lean module '{}'", path.display()))?;
    }
    Ok(())
}

/// Optimizes the selected BX block and maps its sequential design to the
/// requested cells.
pub fn bx2gv(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    g8r2gv::map_design(matches, || {
        let package = compile_input(matches, config)?;
        let options = ir2gates::block_gatify_options(&CanonicalG8rOptions::default());
        optimize_and_lower_bx_block(&package, options).map_err(anyhow::Error::msg)
    })
}
