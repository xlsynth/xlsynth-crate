// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use xlsynth_bx::{
    ImportOptions, QuickCheckOptions, ResetBehavior, check_source_file_with_options, empty_package,
    lower_module, run_tests_with_quickcheck,
};
use xlsynth_g8r::block2sequential::optimize_and_lower_bx_block;
use xlsynth_g8r::process_ir_path::CanonicalG8rOptions;
use xlsynth_pir::ir::{MemberType, Package, PackageMember};
use xlsynth_pir::ir_parser::parse_xls_ir_package;
use xlsynth_pir::ir_utils::is_valid_package_name;

use crate::common::{DEFAULT_WARNINGS_AS_ERRORS, write_stdout, write_stdout_line};
use crate::toolchain_config::{ToolchainConfig, get_dslx_search_paths, get_dslx_stdlib_path};
use crate::{g8r2gv, ir2gates};

/// Adds the source, linking, and reset options shared by the BX commands.
fn add_bx_source_args(command: Command) -> Command {
    command
        .arg(
            Arg::new("input")
                .value_name("BX_FILE")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
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
        )
        .arg(reset_flag("reset-active-low", "Use active-low reset (default active-high)"))
}

fn reset_flag(name: &'static str, help: &'static str) -> Arg {
    Arg::new(name)
        .long(name)
        .help(help)
        .num_args(0..=1)
        .require_equals(true)
        .default_missing_value("true")
        .default_value("false")
        .value_parser(value_parser!(bool))
}

/// Compilation commands emit a design with one selected top block.
fn add_bx_args(command: Command) -> Command {
    add_bx_source_args(command)
        .arg(reset_flag(
            "reset-asynchronous",
            "Use asynchronous reset (default synchronous)",
        ))
        .arg(
            Arg::new("top")
                .long("top")
                .value_name("BLOCK")
                .help("Select the top block (defaults to the sole BX block or unique pub block)"),
        )
}

pub fn bx_test_command() -> Command {
    let command = Command::new("bx-test")
        .about("Runs unit tests and clocked QuickCheck properties in a BX file");
    add_bx_source_args(command)
        .arg(
            Arg::new("quickcheck-trials")
                .long("quickcheck-trials")
                .value_name("COUNT")
                .value_parser(value_parser!(NonZeroUsize))
                .help("Random trials per QuickCheck block (default 100)"),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .value_name("U64")
                .value_parser(value_parser!(u64))
                .help("QuickCheck random seed (default 0)"),
        )
        .arg(
            Arg::new("cross-check-vastly")
                .long("cross-check-vastly")
                .action(ArgAction::SetTrue)
                .help("Compare each QuickCheck sample to RTL and report a separate run from unknown register state"),
        )
}

pub fn bx2ir_command() -> Command {
    add_bx_args(Command::new("bx2ir").about("Compiles BX source to an XLS block IR package"))
}

pub fn bx2gv_command() -> Command {
    g8r2gv::add_mapping_args(
        add_bx_args(
            Command::new("bx2gv")
                .about("Compiles BX source and maps it to a Liberty gate-level netlist"),
        ),
        "area",
        "false",
    )
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

/// Input and frontend options resolved for one BX command invocation.
struct BxInput<'a> {
    path: &'a Path,
    source: String,
    package: Package,
    imports: ImportOptions<'a>,
    active_low: bool,
    warnings_as_errors: bool,
}

/// Reads the BX source and link package and resolves toolchain search paths.
fn with_bx_input<T>(
    matches: &ArgMatches,
    config: &Option<ToolchainConfig>,
    run: impl FnOnce(BxInput<'_>) -> Result<T>,
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
    let warnings_as_errors = matches
        .get_one::<bool>("warnings-as-errors")
        .copied()
        .unwrap_or(DEFAULT_WARNINGS_AS_ERRORS);
    run(BxInput {
        path,
        source,
        package,
        imports,
        active_low: matches.get_flag("reset-active-low"),
        warnings_as_errors,
    })
}

/// Compiles the input and its imports, then selects the requested top block.
fn compile_input(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<Package> {
    with_bx_input(matches, config, |input| {
        let checked = check_source_file_with_options(&input.source, input.path, &input.imports)
            .map_err(|error| anyhow!("{}: {error}", input.path.display()))?;
        let reset = ResetBehavior {
            active_low: input.active_low,
            asynchronous: matches.get_flag("reset-asynchronous"),
        };
        let mut package = lower_module(&checked, input.package, reset)
            .map_err(|error| anyhow!("{}: {error}", input.path.display()))?;
        select_top(
            &mut package,
            matches.get_one::<String>("top").map(String::as_str),
        )?;
        for warning in checked.warnings() {
            eprintln!("{}", warning.display_with_root(input.path));
        }
        if input.warnings_as_errors && !checked.warnings().is_empty() {
            bail!("BX warnings treated as errors");
        }
        Ok(package)
    })
}

pub fn bx2ir(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    write_stdout(&compile_input(matches, config)?.to_string());
    Ok(())
}

/// Runs each unit test in the input file and reports all assertion failures.
pub fn bx_test(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    with_bx_input(matches, config, |input| {
        let reset = ResetBehavior {
            active_low: input.active_low,
            asynchronous: false,
        };
        let defaults = QuickCheckOptions::default();
        let report = run_tests_with_quickcheck(
            &input.source,
            input.path,
            input.package,
            reset,
            &input.imports,
            input.warnings_as_errors,
            QuickCheckOptions {
                trials: matches
                    .get_one::<NonZeroUsize>("quickcheck-trials")
                    .copied()
                    .unwrap_or(defaults.trials),
                seed: matches
                    .get_one::<u64>("seed")
                    .copied()
                    .unwrap_or(defaults.seed),
                cross_check_vastly: matches.get_flag("cross-check-vastly"),
            },
        )
        .map_err(anyhow::Error::msg)?;
        for warning in report.warnings {
            eprintln!("{}", warning.display_with_root(input.path));
        }
        let total = report.tests.len();
        let mut failed = 0;
        for test in report.tests {
            if let Some(qc) = test.quickcheck {
                write_stdout_line(&format!(
                    "quickcheck {}: {} concrete trials (seed {})",
                    test.name, qc.trials, qc.seed
                ));
                if let Some(four) = qc.four_valued {
                    write_stdout_line(&format!(
                        "  Vastly from X: {} true, {} false, {} indeterminate",
                        four.passed, four.failed, four.indeterminate
                    ));
                    if let Some(f) = four.first_false {
                        eprintln!("first four-valued false: {f}");
                    }
                    if let Some(f) = four.first_indeterminate {
                        eprintln!("first four-valued indeterminate: {f}");
                    }
                }
            }
            if let Some(failure) = test.failure {
                failed += 1;
                write_stdout_line(&format!("test {} ... FAILED", test.name));
                eprintln!("{failure}");
            } else {
                write_stdout_line(&format!("test {} ... ok", test.name));
            }
        }
        write_stdout_line(&format!("{} passed; {failed} failed", total - failed));
        if failed != 0 {
            bail!("BX tests failed");
        }
        Ok(())
    })
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
