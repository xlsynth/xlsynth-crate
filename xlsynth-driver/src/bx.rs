// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};
use xlsynth::dslx;
use xlsynth_bx::invariant_prover::{
    InvariantMode, InvariantQueryResult, InvariantResult, check_invariant,
};
use xlsynth_bx::{
    BxTestOptions, ImportOptions, QuickCheckOptions, ResetBehavior, check_source_file_with_options,
    empty_package, lower_module, prepare_invariant_proofs, prepare_quickcheck_proofs,
    run_tests_with_options,
};
use xlsynth_g8r::block2sequential::optimize_and_lower_bx_block;
use xlsynth_g8r::process_ir_path::CanonicalG8rOptions;
use xlsynth_lean::block;
use xlsynth_pir::ir::{MemberType, Package, PackageMember};
use xlsynth_pir::ir_parser::parse_xls_ir_package;
use xlsynth_pir::ir_utils::is_valid_package_name;
use xlsynth_prover::prover::types::{BoolPropertyResult, ProverFn};
use xlsynth_prover::prover::{SolverChoice, SolverLimits, prover_for_choice_with_limits};

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
        .about("Runs unit tests, clocked QuickChecks and invariants in a BX file");
    add_bx_source_args(command)
        .arg(
            Arg::new("quickcheck-trials")
                .long("quickcheck-trials")
                .value_name("COUNT")
                .value_parser(value_parser!(NonZeroUsize))
                .help("Random trials per QuickCheck or invariant block (default 100)"),
        )
        .arg(
            Arg::new("invariant-cycles")
                .long("invariant-cycles")
                .value_name("COUNT")
                .value_parser(value_parser!(NonZeroUsize))
                .help("Sampled cycles per invariant trial (default 16)"),
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
                .help("Compare each QuickCheck or invariant sample to RTL and report a separate run from unknown register state"),
        )
}

pub fn bx2ir_command() -> Command {
    add_bx_args(Command::new("bx2ir").about("Compiles BX source to an XLS block IR package"))
}

pub fn bx_prove_quickcheck_command() -> Command {
    let command = Command::new("bx-prove-quickcheck")
        .about("Proves clocked QuickCheck properties over their bounded tick schedules");
    add_bx_source_args(command)
        .arg(crate::solver_arg_with_toolchain(
            "SMT solver for the bounded proof",
            false,
        ))
        .arg(time_limit_arg())
}

fn time_limit_arg() -> Arg {
    Arg::new("time-limit-ms")
        .long("time-limit-ms")
        .value_name("MS")
        .value_parser(value_parser!(u64).range(1..))
        .help("Positive time limit per solver check in milliseconds (in-process Bitwuzla only)")
}

pub fn bx_prove_invariant_command() -> Command {
    add_bx_source_args(
        Command::new("bx-prove-invariant")
            .about("Checks clocked BX invariants by k-induction or bounded model checking"),
    )
    .arg(
        Arg::new("engine")
            .long("engine")
            .value_parser(["bitwuzla"])
            .default_value("bitwuzla")
            .help("Proof engine"),
    )
    .arg(
        Arg::new("mode")
            .long("mode")
            .value_parser(["induction", "bmc"])
            .default_value("induction")
            .help("Prove an unbounded invariant or check a bounded prefix"),
    )
    .arg(
        Arg::new("max-depth")
            .long("max-depth")
            .value_name("K")
            .default_value("16")
            .value_parser(value_parser!(NonZeroUsize))
            .help("Maximum induction depth or number of sampled cycles in BMC mode"),
    )
    .arg(time_limit_arg())
}

pub fn bx_to_lean_command() -> Command {
    let command = add_bx_args(Command::new("bx-to-lean").about("Generates raw and readable Lean for a BX block"))
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
            .requires("link-dslx")
            .help("Use checked DSLX loop bindings to name locals in the readable model"),
    );
    command
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
        report_warnings(input.path, input.warnings_as_errors, checked.warnings())?;
        Ok(package)
    })
}

fn report_warnings(
    path: &Path,
    warnings_as_errors: bool,
    warnings: &[xlsynth_bx::BxWarning],
) -> Result<()> {
    for warning in warnings {
        eprintln!("{}", warning.display_with_root(path));
    }
    if warnings_as_errors && !warnings.is_empty() {
        bail!("BX warnings treated as errors");
    }
    Ok(())
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
        let defaults = BxTestOptions::default();
        let invariant_cycles = matches
            .get_one::<NonZeroUsize>("invariant-cycles")
            .copied()
            .unwrap_or(defaults.invariant_cycles);
        let report = run_tests_with_options(
            &input.source,
            input.path,
            input.package,
            reset,
            &input.imports,
            input.warnings_as_errors,
            BxTestOptions {
                quickcheck: QuickCheckOptions {
                    trials: matches
                        .get_one::<NonZeroUsize>("quickcheck-trials")
                        .copied()
                        .unwrap_or(defaults.quickcheck.trials),
                    seed: matches
                        .get_one::<u64>("seed")
                        .copied()
                        .unwrap_or(defaults.quickcheck.seed),
                    cross_check_vastly: matches.get_flag("cross-check-vastly"),
                },
                invariant_cycles,
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
                if report.invariant_names.contains(&test.name) {
                    write_stdout_line(&format!(
                        "invariant {}: {} concrete trials of {invariant_cycles} cycles (seed {})",
                        test.name, qc.trials, qc.seed
                    ));
                } else {
                    write_stdout_line(&format!(
                        "quickcheck {}: {} concrete trials (seed {})",
                        test.name, qc.trials, qc.seed
                    ));
                }
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

/// Proves each clocked property and replays any solver counterexample in BX.
pub fn bx_prove_quickcheck(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    let choice = matches
        .get_one::<String>("solver")
        .expect("clap supplies a default solver")
        .parse::<SolverChoice>()
        .map_err(anyhow::Error::msg)?;
    let limit = matches.get_one::<u64>("time-limit-ms").copied();
    if limit.is_some() && choice != SolverChoice::Bitwuzla {
        bail!("--time-limit-ms requires --solver=bitwuzla");
    }
    let prover = prover_for_choice_with_limits(
        choice,
        None,
        SolverLimits {
            time_limit_per_ms: limit,
            ..Default::default()
        },
    );
    with_bx_input(matches, config, |input| {
        let suite = prepare_quickcheck_proofs(
            &input.source,
            input.path,
            input.package,
            ResetBehavior {
                active_low: input.active_low,
                asynchronous: false,
            },
            &input.imports,
            input.warnings_as_errors,
        )
        .map_err(anyhow::Error::msg)?;
        for warning in suite.warnings {
            eprintln!("{}", warning.display_with_root(input.path));
        }
        if suite.properties.is_empty() {
            bail!("no #[quickcheck] blocks in {}", input.path.display());
        }
        let mut proved = 0;
        let mut disproved = 0;
        let mut inconclusive = 0;
        let mut errors = 0;
        for case in suite.properties {
            let property = match case.proof {
                Ok(property) => property,
                Err(diagnostic) => {
                    inconclusive += 1;
                    write_stdout_line(&format!("quickcheck {} ... INCONCLUSIVE", case.name));
                    eprintln!("{diagnostic}");
                    continue;
                }
            };
            let result = match property.package.get_top_fn() {
                Some(top) => {
                    prover.prove_ir_fn_quickcheck(&ProverFn::new(top, Some(&property.package)))
                }
                None => BoolPropertyResult::Error("proof package has no top function".into()),
            };
            let (status, diagnostic) = match result {
                BoolPropertyResult::Proved => {
                    proved += 1;
                    ("proved", None)
                }
                BoolPropertyResult::Disproved { inputs, .. } => {
                    let inputs: Vec<_> = inputs.into_iter().map(|input| input.value).collect();
                    match property.replay(&inputs) {
                        Ok(trace) => {
                            disproved += 1;
                            ("DISPROVED", Some(trace))
                        }
                        Err(error) => {
                            errors += 1;
                            (
                                "ERROR",
                                Some(format!("counterexample replay failed: {error}")),
                            )
                        }
                    }
                }
                BoolPropertyResult::Inconclusive(reason) => {
                    inconclusive += 1;
                    ("INCONCLUSIVE", Some(reason))
                }
                BoolPropertyResult::Error(error)
                | BoolPropertyResult::ToolchainDisproved(error) => {
                    errors += 1;
                    ("ERROR", Some(error))
                }
            };
            write_stdout_line(&format!("quickcheck {} ... {status}", case.name));
            if let Some(message) = diagnostic {
                eprintln!("{message}");
            }
        }
        write_stdout_line(&format!(
            "{proved} proved; {disproved} disproved; {inconclusive} inconclusive; {errors} errors"
        ));
        if disproved + inconclusive + errors != 0 {
            bail!("BX QuickCheck proof suite did not pass");
        }
        Ok(())
    })
}

/// Checks named clocked invariants and reports reachable witnesses separately
/// from induction steps that remain open.
pub fn bx_prove_invariant(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    let mode = match matches.get_one::<String>("mode").map(String::as_str) {
        Some("bmc") => InvariantMode::Bmc,
        Some("induction") => InvariantMode::Induction,
        _ => unreachable!("clap validates the proof mode"),
    };
    let depth = *matches
        .get_one::<NonZeroUsize>("max-depth")
        .expect("clap supplies a default depth");
    let choice = matches
        .get_one::<String>("engine")
        .expect("clap supplies a default engine")
        .parse::<SolverChoice>()
        .map_err(anyhow::Error::msg)?;
    let prover = prover_for_choice_with_limits(
        choice,
        None,
        SolverLimits {
            time_limit_per_ms: matches.get_one::<u64>("time-limit-ms").copied(),
            ..Default::default()
        },
    );
    with_bx_input(matches, config, |input| {
        let suite = prepare_invariant_proofs(
            &input.source,
            input.path,
            input.package,
            ResetBehavior {
                active_low: input.active_low,
                asynchronous: false,
            },
            &input.imports,
            input.warnings_as_errors,
        )
        .map_err(anyhow::Error::msg)?;
        for warning in suite.warnings {
            eprintln!("{}", warning.display_with_root(input.path));
        }
        if suite.properties.is_empty() {
            bail!("no #[invariant] blocks in {}", input.path.display());
        }
        let (mut proved, mut bounded, mut disproved, mut inconclusive, mut errors) =
            (0, 0, 0, 0, 0);
        for case in suite.properties {
            let result = match case.proof {
                Err(reason) => Ok(InvariantResult::Inconclusive { reason }),
                Ok(property) => check_invariant(&property, mode, depth, |package| {
                    let top = package
                        .get_top_fn()
                        .ok_or("invariant query has no top function")?;
                    match prover.prove_ir_fn_quickcheck(&ProverFn::new(top, Some(package))) {
                        BoolPropertyResult::Proved => Ok(InvariantQueryResult::Proved),
                        BoolPropertyResult::Disproved { inputs, .. } => {
                            Ok(InvariantQueryResult::Counterexample(
                                inputs.into_iter().map(|input| input.value).collect(),
                            ))
                        }
                        BoolPropertyResult::Inconclusive(reason) => {
                            Ok(InvariantQueryResult::Inconclusive(reason))
                        }
                        BoolPropertyResult::Error(error)
                        | BoolPropertyResult::ToolchainDisproved(error) => Err(error),
                    }
                }),
            };
            let (status, diagnostic) = match result {
                Ok(InvariantResult::Proved { depth }) => {
                    proved += 1;
                    (format!("proved at k={depth}"), None)
                }
                Ok(InvariantResult::Bounded { cycles }) => {
                    bounded += 1;
                    (format!("passed {cycles} bounded cycles"), None)
                }
                Ok(InvariantResult::Disproved { trace }) => {
                    disproved += 1;
                    ("DISPROVED".into(), Some(trace))
                }
                Ok(InvariantResult::Inconclusive { reason }) => {
                    inconclusive += 1;
                    ("INCONCLUSIVE".into(), Some(reason))
                }
                Err(error) => {
                    errors += 1;
                    ("ERROR".into(), Some(error))
                }
            };
            write_stdout_line(&format!("invariant {} ... {status}", case.name));
            if let Some(message) = diagnostic {
                eprintln!("{message}");
            }
        }
        write_stdout_line(&format!(
            "{proved} proved; {bounded} bounded; {disproved} disproved; {inconclusive} inconclusive; {errors} errors"
        ));
        if disproved + inconclusive + errors != 0 {
            bail!("BX invariant suite did not pass");
        }
        Ok(())
    })
}

/// Generate and validate the complete model before creating its output
/// directory.
pub fn bx_to_lean(matches: &ArgMatches, config: &Option<ToolchainConfig>) -> Result<()> {
    let compiled = with_bx_input(matches, config, |input| {
        let bx_name = input
            .path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .context("BX input requires a UTF-8 module name")?;
        let reset = ResetBehavior {
            active_low: input.active_low,
            asynchronous: matches.get_flag("reset-asynchronous"),
        };
        let compiled = if let Some(path) = matches.get_one::<String>("link-dslx") {
            let dslx_source = fs::read_to_string(path)
                .with_context(|| format!("reading linked DSLX module '{path}'"))?;
            let name = xlsynth::dslx_path_to_module_name(Path::new(path))?;
            let mut import_data = dslx::ImportData::new(
                input.imports.dslx_stdlib_path,
                &input.imports.additional_search_paths,
            );
            let module = dslx::parse_and_typecheck(&dslx_source, path, name, &mut import_data)
                .with_context(|| format!("checking linked DSLX module '{path}'"))?;
            let compile = block::compile_bx_with_dslx;
            #[cfg(feature = "dslx-loop-bindings")]
            let compile = if matches.get_flag("readable-loop-names") {
                block::compile_bx_with_dslx_loop_bindings
            } else {
                compile
            };
            compile(bx_name, &input.source, module, reset, &input.imports)
        } else {
            block::compile_bx(bx_name, &input.source, input.package, reset, &input.imports)
        }
        .with_context(|| input.path.display().to_string())?;
        report_warnings(input.path, input.warnings_as_errors, compiled.warnings())?;
        Ok(compiled)
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
