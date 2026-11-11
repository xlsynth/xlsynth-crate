// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, ArgMatches, Command, value_parser};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_source_file_with_options, empty_package, lower_module,
};
use xlsynth_g8r::block2sequential::optimize_and_lower_bx_block;
use xlsynth_g8r::process_ir_path::CanonicalG8rOptions;
use xlsynth_pir::ir::{MemberType, Package, PackageMember};
use xlsynth_pir::ir_parser::parse_xls_ir_package;
use xlsynth_pir::ir_utils::is_valid_package_name;

use crate::common::{DEFAULT_WARNINGS_AS_ERRORS, write_stdout};
use crate::toolchain_config::{ToolchainConfig, get_dslx_search_paths, get_dslx_stdlib_path};
use crate::{g8r2gv, ir2gates};

/// Adds the source, linking, and reset options shared by both BX commands.
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
                .help("Semicolon-separated BX and DSLX import directories (overrides the toolchain path)"),
        )
        .arg(
            Arg::new("warnings-as-errors")
                .long("warnings-as-errors")
                .help("Treat BX source warnings as errors (default true)")
                .num_args(0..=1)
                .require_equals(true)
                .default_missing_value("true")
                .value_parser(value_parser!(bool)),
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

/// Reads the source and optional link package and compiles them with the
/// selected top.
fn compile_input(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<Package> {
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
    let mut paths = get_dslx_search_paths(matches, config);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty())
        && !paths.iter().any(|p| p == parent)
    {
        paths.push(parent.to_path_buf());
    }
    let stdlib = get_dslx_stdlib_path(matches, config);
    let imports = ImportOptions {
        dslx_stdlib_path: stdlib.as_deref().map(Path::new),
        additional_search_paths: paths.iter().map(PathBuf::as_path).collect(),
    };
    let reset = ResetBehavior {
        active_low: matches.get_flag("reset-active-low"),
        asynchronous: matches.get_flag("reset-asynchronous"),
    };
    let checked = check_source_file_with_options(&source, path, &imports)
        .map_err(|error| anyhow!("{}: {error}", path.display()))?;
    let mut package = lower_module(&checked, package, reset)
        .map_err(|error| anyhow!("{}: {error}", path.display()))?;
    select_top(
        &mut package,
        matches.get_one::<String>("top").map(String::as_str),
    )?;
    for warning in checked.warnings() {
        eprintln!("{}", warning.display_with_root(path));
    }
    let warnings_as_errors = matches
        .get_one::<bool>("warnings-as-errors")
        .copied()
        .unwrap_or(DEFAULT_WARNINGS_AS_ERRORS);
    if warnings_as_errors && !checked.warnings().is_empty() {
        bail!("BX warnings treated as errors");
    }
    Ok(package)
}

pub fn bx2ir(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    write_stdout(&compile_input(matches, config)?.to_string());
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
