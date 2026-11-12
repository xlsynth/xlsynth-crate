// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};

use xlsynth_g8r::aig::GateFn;

use crate::gate_fn::export_gate_fn_aag;
use crate::ir_proof::FIXED_IR_PROOF_FILES;
use crate::process::{resolve_lean_executable, run_lean_process};
use crate::support::{AIG_PROOF_SUPPORT, IR_FUNCTION_SUPPORT, write_file};
use crate::{InterfaceDirection, LeanError, LeanModuleName};

const CHECK_SOURCE: &str = r#"-- SPDX-License-Identifier: Apache-2.0

import Target
import UserProof

example : Target.Equivalent := UserProof.equivalent
"#;

const TARGET_ARTIFACTS: &[&str] = &["Lhs.aag", "Rhs.aag", "Target.lean", "Check.lean"];

/// Mechanically generated circuits and root proposition for an equivalence
/// proof.
#[derive(Debug, Clone)]
pub struct LeanEquivalenceTarget {
    lhs_aag: String,
    rhs_aag: String,
    input_count: usize,
    output_count: usize,
}

impl LeanEquivalenceTarget {
    /// Writes the complete prepared target using a fixed portable layout.
    pub fn write_to(&self, root: &Path) -> Result<(), LeanError> {
        AIG_PROOF_SUPPORT.write_to(root)?;
        write_file(root, Path::new("Lhs.aag"), self.lhs_aag.as_bytes())?;
        write_file(root, Path::new("Rhs.aag"), self.rhs_aag.as_bytes())?;
        let target_source = target_source(self.input_count, self.output_count);
        write_file(root, Path::new("Target.lean"), target_source.as_bytes())?;
        write_file(root, Path::new("Check.lean"), CHECK_SOURCE.as_bytes())
    }
}

/// Defines both circuits through the shared AAG loader and the root
/// proposition.
fn target_source(input_count: usize, output_count: usize) -> String {
    format!(
        r#"-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Aag

namespace Generated.Lhs

def circuit : XlsynthLean.Aig.Circuit {input_count} {output_count} :=
  (XlsynthLean.Aag.parse {input_count} {output_count}
    (include_str "Lhs.aag")).toOption.get (by native_decide)

def eval (inputs : BitVec {input_count}) : BitVec {output_count} :=
  circuit.eval inputs

end Generated.Lhs

namespace Generated.Rhs

def circuit : XlsynthLean.Aig.Circuit {input_count} {output_count} :=
  (XlsynthLean.Aag.parse {input_count} {output_count}
    (include_str "Rhs.aag")).toOption.get (by native_decide)

def eval (inputs : BitVec {input_count}) : BitVec {output_count} :=
  circuit.eval inputs

end Generated.Rhs

namespace Target

abbrev Equivalent : Prop :=
  XlsynthLean.Aig.Circuit.Equivalent Generated.Lhs.circuit Generated.Rhs.circuit

end Target
"#
    )
}

/// User-authored proof body, with helper paths relative to `root`.
///
/// The body is a complete Lean term, such as `by ...`. The runner generates
/// `UserProof.lean` with imports and the fixed theorem declaration. Helpers
/// retain their paths relative to `root`, which determine their Lean module
/// names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeanProofSources {
    pub root: PathBuf,
    pub body: PathBuf,
    pub modules: Vec<PathBuf>,
}

/// Fully qualified Lean declaration name selected by an IR equivalence target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeanTheoremName(String);

impl LeanTheoremName {
    /// Accept dotted identifiers (including names declared at the root).
    pub fn new(name: &str) -> Result<Self, LeanError> {
        let valid = name.split('.').all(|part| {
            let mut chars = part.chars();
            chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
                && chars.all(|c| c.is_alphanumeric() || c == '_' || c == '\'')
        });
        if !valid || name.split('.').next() == Some("_root_") {
            return Err(LeanError::InvalidProofSources(format!(
                "invalid Lean proof theorem name {name:?}"
            )));
        }
        Ok(Self(name.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A user module defining the selected IR theorem, and its available imports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeanIrProofSources {
    pub root: PathBuf,
    pub module: PathBuf,
    pub theorem: LeanTheoremName,
    pub modules: Vec<PathBuf>,
}

/// Axiom policy applied to the selected IR theorem and its dependencies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LeanIrAxiomPolicy {
    /// Allow the three standard axioms and replay closed Boolean checks.
    #[default]
    ReplayNativeChecks,
    /// Allow only `propext`, `Classical.choice`, and `Quot.sound`.
    RejectNativeChecks,
}

/// Captured diagnostics from a successful equivalence proof check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeanProofOutput {
    /// Version string reported by the selected Lean executable.
    pub lean_version: String,
    /// Number of distinct native Boolean axioms replayed by the IR audit.
    /// AIG proof checks omit the audit and return `None`.
    pub native_checks_replayed: Option<usize>,
    /// Combined standard output from the compilation steps.
    pub stdout: String,
    /// Combined standard error from the compilation steps.
    pub stderr: String,
}

/// A failed proof invocation with its available version and process
/// diagnostics.
#[derive(Debug, thiserror::Error)]
#[error("{error}")]
pub struct LeanProofFailure {
    /// Underlying staging, process, or proof error.
    #[source]
    pub error: Box<LeanError>,
    /// Version reported by Lean, if the version query succeeded.
    pub lean_version: Option<String>,
    /// Combined standard output from completed compilation steps and the failed
    /// process.
    pub stdout: String,
    /// Combined standard error from completed compilation steps and the failed
    /// process.
    pub stderr: String,
}

impl LeanProofFailure {
    /// Adds diagnostics from completed compilation steps to a failure.
    fn with_output(error: LeanError, output: &LeanProofOutput) -> Self {
        let mut failure = Self::from(error);
        failure.lean_version = Some(output.lean_version.clone());
        failure.stdout.insert_str(0, &output.stdout);
        failure.stderr.insert_str(0, &output.stderr);
        failure
    }
}

impl From<LeanError> for LeanProofFailure {
    fn from(error: LeanError) -> Self {
        let (stdout, stderr) = match &error {
            LeanError::ProcessFailed { stdout, stderr, .. } => (stdout.clone(), stderr.clone()),
            _ => (String::new(), String::new()),
        };
        Self {
            error: Box::new(error),
            lean_version: None,
            stdout,
            stderr,
        }
    }
}

/// Prepares the exact Lean equivalence target for two combinational gate
/// graphs.
///
/// Input and output ports must have matching names and widths in the same
/// order.
pub fn prepare_equivalence_target(
    lhs: &GateFn,
    rhs: &GateFn,
) -> Result<LeanEquivalenceTarget, LeanError> {
    validate_matching_interfaces(lhs, rhs)?;
    let (interface, lhs_aag) = export_gate_fn_aag(lhs)?;
    let (_, rhs_aag) = export_gate_fn_aag(rhs)?;
    Ok(LeanEquivalenceTarget {
        lhs_aag,
        rhs_aag,
        input_count: interface.input_bit_count(),
        output_count: interface.output_bit_count(),
    })
}

/// Runs a supplied proof against a prepared target in an empty work directory.
///
/// The proof body is inserted into a generated declaration of
/// `UserProof.equivalent : Target.Equivalent`. Every supplied helper module is
/// imported mechanically. Dependencies are discovered with Lean's `--src-deps`
/// mode and compiled in dependency order. Solver selection is left to Lean's
/// default lookup.
///
/// Lean's normal acceptance policy applies, including its treatment of axioms
/// and `sorry`. This function does not audit the proof or sandbox Lean.
pub fn run_equivalence_proof(
    lean_executable: &Path,
    target_root: &Path,
    proof_sources: &LeanProofSources,
    work_root: &Path,
) -> Result<LeanProofOutput, LeanProofFailure> {
    let artifacts = AIG_PROOF_SUPPORT
        .paths()
        .chain(TARGET_ARTIFACTS.iter().copied())
        .collect::<Vec<_>>();
    let modules = AIG_PROOF_SUPPORT
        .module_paths()
        .chain(["Target.lean"])
        .collect::<Vec<_>>();
    run_prepared_proof(
        lean_executable,
        target_root,
        ProofInputs::Aig(proof_sources),
        work_root,
        &artifacts,
        &modules,
    )
}

/// Checks the named theorem against the generated IR equality, replaying
/// native Boolean checks in its transitive axiom dependencies.
pub fn run_ir_function_equivalence_proof(
    lean_executable: &Path,
    target_root: &Path,
    proof_sources: &LeanIrProofSources,
    work_root: &Path,
) -> Result<LeanProofOutput, LeanProofFailure> {
    run_ir_function_equivalence_proof_with_policy(
        lean_executable,
        target_root,
        proof_sources,
        work_root,
        LeanIrAxiomPolicy::ReplayNativeChecks,
    )
}

/// Checks the named IR equivalence theorem and enforces the given axiom policy.
pub fn run_ir_function_equivalence_proof_with_policy(
    lean_executable: &Path,
    target_root: &Path,
    proof_sources: &LeanIrProofSources,
    work_root: &Path,
    policy: LeanIrAxiomPolicy,
) -> Result<LeanProofOutput, LeanProofFailure> {
    let proof_modules = [
        "XlsynthLean/IRProof.lean",
        "XlsynthLean/IRDecompose.lean",
        "Generated/Lhs.lean",
        "Generated/Rhs.lean",
        "Target.lean",
    ];
    let modules = IR_FUNCTION_SUPPORT
        .module_paths()
        .chain(proof_modules)
        .collect::<Vec<_>>();
    let artifacts = IR_FUNCTION_SUPPORT
        .paths()
        .chain(proof_modules)
        .collect::<Vec<_>>();
    run_prepared_proof(
        lean_executable,
        target_root,
        ProofInputs::Ir(proof_sources, policy),
        work_root,
        &artifacts,
        &modules,
    )
}

/// Selects the target's support files and the verifier's axiom policy.
#[derive(Clone, Copy)]
enum ProofInputs<'a> {
    Aig(&'a LeanProofSources),
    Ir(&'a LeanIrProofSources, LeanIrAxiomPolicy),
}

/// Stages the fixed target before any user modules and checks the generated
/// root theorem.
fn run_prepared_proof(
    lean_executable: &Path,
    target_root: &Path,
    proof_sources: ProofInputs<'_>,
    work_root: &Path,
    artifacts: &[&str],
    modules: &[&str],
) -> Result<LeanProofOutput, LeanProofFailure> {
    let lean_executable = resolve_lean_executable(lean_executable)?;
    let work_root = absolute_path(work_root)?;
    prepare_empty_work_directory(&work_root)?;
    let (ir_audit, fixed_files) = match proof_sources {
        ProofInputs::Aig(_) => (None, &[][..]),
        ProofInputs::Ir(ir, policy) => (Some((&ir.theorem, policy)), FIXED_IR_PROOF_FILES),
    };
    copy_prepared_target(target_root, &work_root, artifacts, fixed_files)?;
    let module_roots = [work_root.as_path()];
    let version_output = run_lean_process(
        &lean_executable,
        &work_root,
        &module_roots,
        "version query",
        &[OsString::from("--version")],
    )?;
    let mut result = LeanProofOutput {
        lean_version: version_output.stdout.trim().to_string(),
        native_checks_replayed: None,
        stdout: String::new(),
        stderr: String::new(),
    };

    let generated_modules = modules.iter().map(PathBuf::from).collect::<Vec<_>>();
    for relative in &generated_modules {
        compile_module(
            &lean_executable,
            &work_root,
            relative,
            "generated target compilation",
            &mut result,
        )
        .map_err(|error| LeanProofFailure::with_output(error, &result))?;
    }
    // Keep the original generated declarations in the driver before Lean reads
    // any helper, including during dependency discovery.
    let target_objects = read_compiled_modules(&work_root, &generated_modules)
        .map_err(|error| LeanProofFailure::with_output(error, &result))?;
    let proof_modules = stage_proof_sources(proof_sources, &work_root, artifacts)
        .map_err(|error| LeanProofFailure::with_output(error, &result))?;
    let ordered_proof_modules = order_proof_modules(
        &lean_executable,
        &work_root,
        &proof_modules,
        &generated_modules,
    )
    .map_err(|error| LeanProofFailure::with_output(error, &result))?;
    for relative in ordered_proof_modules {
        compile_module(
            &lean_executable,
            &work_root,
            &relative,
            "user proof compilation",
            &mut result,
        )
        .map_err(|error| LeanProofFailure::with_output(error, &result))?;
    }
    verify_compiled_proof(
        &lean_executable,
        &work_root,
        target_objects,
        &proof_modules,
        ir_audit,
        &mut result,
    )
    .map_err(|error| LeanProofFailure::with_output(error, &result))?;
    Ok(result)
}

/// Reads the public .olean and any split module parts that Lean emitted.
fn read_compiled_modules(
    root: &Path,
    modules: &[PathBuf],
) -> Result<BTreeMap<PathBuf, Vec<u8>>, LeanError> {
    let mut files = BTreeMap::new();
    for module in modules {
        for extension in ["olean", "olean.server", "olean.private", "ir.sig", "ir"] {
            let relative = module.with_extension(extension);
            let path = root.join(&relative);
            match fs::read(&path) {
                Ok(bytes) => {
                    files.insert(relative, bytes);
                }
                Err(error)
                    if extension != "olean" && error.kind() == std::io::ErrorKind::NotFound =>
                {
                    // Legacy Lean modules have no separate module parts.
                }
                Err(source) => {
                    return Err(LeanError::Io {
                        operation: "reading compiled Lean module",
                        path,
                        source,
                    });
                }
            }
        }
    }
    Ok(files)
}

/// Starts the crate-owned checker without user sources or frontend extensions
/// on its Lean search path. Only declared compiled modules cross this boundary.
fn verify_compiled_proof(
    lean: &Path,
    work_root: &Path,
    target_objects: BTreeMap<PathBuf, Vec<u8>>,
    proof_modules: &[PathBuf],
    ir_audit: Option<(&LeanTheoremName, LeanIrAxiomPolicy)>,
    result: &mut LeanProofOutput,
) -> Result<(), LeanError> {
    let verification = tempfile::Builder::new()
        .prefix("xlsynth-lean-verify-")
        .tempdir()
        .map_err(LeanError::TemporaryDirectory)?;
    let root = verification.path();
    let objects = root.join("modules");
    let proof_objects = read_compiled_modules(work_root, proof_modules)?;
    for (path, bytes) in target_objects.into_iter().chain(proof_objects) {
        if objects.join(&path).exists() {
            return Err(LeanError::InvalidProofSources(format!(
                "compiled module path {} collides with another artifact",
                path.display()
            )));
        }
        write_file(&objects, &path, &bytes)?;
    }
    let verifier = Path::new("VerifyEquivalence.lean");
    write_file(
        root,
        verifier,
        include_bytes!("../lean/VerifyEquivalence.lean"),
    )?;
    let mut args = vec![
        OsString::from("--run"),
        root.join(verifier).into_os_string(),
        objects.into_os_string(),
    ];
    if let Some((theorem, policy)) = ir_audit {
        let mode = match policy {
            LeanIrAxiomPolicy::ReplayNativeChecks => "ir",
            LeanIrAxiomPolicy::RejectNativeChecks => "ir-strict",
        };
        args.push(mode.into());
        args.push(theorem.as_str().into());
    } else {
        args.push("aig".into());
    }
    for module in proof_modules {
        if module != Path::new("UserProof.lean") {
            args.push(
                proof_module_name(module, ir_audit.is_none())?
                    .as_str()
                    .into(),
            );
        }
    }
    let output = run_lean_process(lean, root, &[], "equivalence check", &args)?;
    result.stdout.push_str(&output.stdout);
    result.stderr.push_str(&output.stderr);
    if output.stdout.lines().last() != Some("xlsynth: verified UserProof.equivalent") {
        return Err(LeanError::Protocol(
            "equivalence verifier exited without confirming the root theorem".into(),
        ));
    }
    if ir_audit.is_some() {
        // Read only the verifier's private output, independently of diagnostics
        // produced while compiling source modules.
        let path = root.join("native-checks-replayed.json");
        let bytes = fs::read(&path).map_err(|source| LeanError::Io {
            operation: "reading IR axiom audit result",
            path,
            source,
        })?;
        result.native_checks_replayed = Some(serde_json::from_slice(&bytes)?);
    }
    Ok(())
}

/// Prepares and checks a proof body in a temporary directory.
pub fn check_equivalence_proof(
    lean_executable: &Path,
    lhs: &GateFn,
    rhs: &GateFn,
    proof_file: &Path,
) -> Result<LeanProofOutput, LeanProofFailure> {
    let target = prepare_equivalence_target(lhs, rhs)?;
    let temporary = tempfile::Builder::new()
        .prefix("xlsynth-lean-proof-")
        .tempdir()
        .map_err(LeanError::TemporaryDirectory)?;
    let target_root = temporary.path().join("target");
    target.write_to(&target_root)?;
    let proof_sources = LeanProofSources {
        root: proof_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        body: proof_file.to_path_buf(),
        modules: Vec::new(),
    };
    run_equivalence_proof(
        lean_executable,
        &target_root,
        &proof_sources,
        &temporary.path().join("work"),
    )
}

/// Resolves a path lexically without changing symlink-based logical paths.
fn absolute_path(path: &Path) -> Result<PathBuf, LeanError> {
    std::path::absolute(path).map_err(|source| LeanError::Io {
        operation: "resolving path",
        path: path.to_path_buf(),
        source,
    })
}

/// Creates the work directory and rejects files from an earlier invocation.
fn prepare_empty_work_directory(root: &Path) -> Result<(), LeanError> {
    fs::create_dir_all(root).map_err(|source| LeanError::Io {
        operation: "creating proof work directory",
        path: root.to_path_buf(),
        source,
    })?;
    let mut entries = fs::read_dir(root).map_err(|source| LeanError::Io {
        operation: "reading proof work directory",
        path: root.to_path_buf(),
        source,
    })?;
    if let Some(entry) = entries.next() {
        entry.map_err(|source| LeanError::Io {
            operation: "reading proof work directory entry",
            path: root.to_path_buf(),
            source,
        })?;
        return Err(LeanError::WorkDirectoryNotEmpty(root.to_path_buf()));
    }
    Ok(())
}

/// Copies only the fixed artifacts for the selected target kind.
fn copy_prepared_target(
    source_root: &Path,
    work_root: &Path,
    artifacts: &[&str],
    fixed_files: &[(&str, &[u8])],
) -> Result<(), LeanError> {
    for relative in artifacts {
        if let Some((_, contents)) = fixed_files.iter().find(|(path, _)| path == relative) {
            write_file(work_root, Path::new(relative), contents)?;
            continue;
        }
        let source_path = source_root.join(relative);
        let contents = fs::read(&source_path).map_err(|source| LeanError::Io {
            operation: "reading prepared Lean target",
            path: source_path,
            source,
        })?;
        write_file(work_root, Path::new(relative), &contents)?;
    }
    Ok(())
}

/// Stages the declared modules and generates the fixed root theorem.
fn stage_proof_sources(
    sources: ProofInputs<'_>,
    work_root: &Path,
    artifacts: &[&str],
) -> Result<Vec<PathBuf>, LeanError> {
    let (root, entry, helpers, theorem) = match sources {
        ProofInputs::Aig(s) => (&s.root, &s.body, &s.modules, None),
        ProofInputs::Ir(s, _) => (&s.root, &s.module, &s.modules, Some(&s.theorem)),
    };
    let root = absolute_path(root)?;
    let entry = absolute_path(entry)?;
    if entry.extension().and_then(|v| v.to_str()) != Some("lean") || !entry.starts_with(&root) {
        return Err(LeanError::InvalidProofSources(format!(
            "proof file {} must be a .lean file under proof root {}",
            entry.display(),
            root.display()
        )));
    }
    let mut staged = BTreeMap::<PathBuf, PathBuf>::new();
    let mut modules = vec![PathBuf::from("UserProof.lean")];
    let mut imports = Vec::new();
    let reserved = artifacts
        .iter()
        .copied()
        .chain(["UserProof.lean"])
        .map(PathBuf::from)
        .collect::<BTreeSet<_>>();
    // The named entry module comes first so the verifier can check the
    // selected declaration's module of origin independently of user syntax.
    let entry_module = theorem.map(|_| entry.as_path());
    for path in entry_module
        .into_iter()
        .chain(helpers.iter().map(PathBuf::as_path))
    {
        let source_path = absolute_path(path)?;
        if theorem.is_none() && source_path == entry {
            return Err(LeanError::InvalidProofSources(
                "the proof body must not also appear in modules".to_string(),
            ));
        }
        let relative = source_path
            .strip_prefix(&root)
            .map_err(|_| {
                LeanError::InvalidProofSources(format!(
                    "{} is outside proof root {}",
                    source_path.display(),
                    root.display()
                ))
            })?
            .to_path_buf();
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(LeanError::InvalidProofSources(format!(
                "invalid proof-relative path {}",
                relative.display()
            )));
        }
        let module_name = proof_module_name(&relative, theorem.is_none())?;
        if theorem.is_none() || source_path == entry {
            imports.push(module_name);
        }
        modules.push(relative.clone());
        if reserved.contains(&relative) || staged.insert(relative.clone(), source_path).is_some() {
            return Err(LeanError::InvalidProofSources(format!(
                "proof path {} collides with another artifact",
                relative.display()
            )));
        }
    }
    for (relative, source_path) in staged {
        let contents = fs::read(&source_path).map_err(|source| LeanError::Io {
            operation: "reading Lean proof input",
            path: source_path,
            source,
        })?;
        write_file(work_root, &relative, &contents)?;
    }
    let body_source = match theorem {
        Some(name) => format!("_root_.{}", crate::gate_fn::quoted_lean_name(name.as_str())),
        None => fs::read_to_string(&entry).map_err(|source| LeanError::Io {
            operation: "reading Lean proof body",
            path: entry,
            source,
        })?,
    };
    let source = user_proof_source(&body_source, &imports);
    write_file(work_root, Path::new("UserProof.lean"), source.as_bytes())?;
    Ok(modules)
}

/// Generates the imports and fixed theorem declaration around a Lean term.
fn user_proof_source(body: &str, imports: &[LeanModuleName]) -> String {
    let mut source = String::from("-- SPDX-License-Identifier: Apache-2.0\n\nimport Target\n");
    for module in imports {
        source.push_str("import ");
        source.push_str(&module.lean_identifier());
        source.push('\n');
    }
    source.push_str("\ntheorem UserProof.equivalent : Target.Equivalent := (\n");
    source.push_str(body);
    if !body.ends_with('\n') {
        source.push('\n');
    }
    source.push_str(")\n");
    source
}

/// Returns the validated module name implied by a helper's relative path.
fn proof_module_name(path: &Path, reserve_gate_checker: bool) -> Result<LeanModuleName, LeanError> {
    if path.extension().and_then(|value| value.to_str()) != Some("lean") {
        return Err(LeanError::InvalidProofSources(format!(
            "proof module {} must have a .lean extension",
            path.display()
        )));
    }
    let without_extension = path.with_extension("");
    let parts = without_extension
        .components()
        .map(|component| {
            component
                .as_os_str()
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| {
                    LeanError::InvalidProofSources(format!(
                        "proof module path {} is not UTF-8",
                        path.display()
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parts.first().is_some_and(|part| {
        (reserve_gate_checker && part.eq_ignore_ascii_case("Check"))
            || ["Generated", "Target", "UserProof"]
                .iter()
                .any(|reserved| part.eq_ignore_ascii_case(reserved))
    }) {
        return Err(LeanError::InvalidProofSources(format!(
            "proof module path {} uses a generated module root",
            path.display()
        )));
    }
    let name = LeanModuleName::new(parts.join("."))?;
    if name.source_path() != path {
        return Err(LeanError::InvalidProofSources(format!(
            "proof module path {} must use a directory per module component",
            path.display()
        )));
    }
    Ok(name)
}

/// Uses Lean's source dependency listing to topologically order proof modules.
fn order_proof_modules(
    lean_executable: &Path,
    work_root: &Path,
    proof_modules: &[PathBuf],
    generated_modules: &[PathBuf],
) -> Result<Vec<PathBuf>, LeanError> {
    let canonical_root = fs::canonicalize(work_root).map_err(|source| LeanError::Io {
        operation: "canonicalizing proof work directory",
        path: work_root.to_path_buf(),
        source,
    })?;
    let proof_set = proof_modules.iter().cloned().collect::<BTreeSet<_>>();
    let generated_set = generated_modules.iter().cloned().collect::<BTreeSet<_>>();
    let mut pending = BTreeMap::<PathBuf, BTreeSet<PathBuf>>::new();
    for relative in proof_modules {
        let output = run_lean_process(
            lean_executable,
            work_root,
            &[work_root],
            "proof dependency discovery",
            &[
                OsString::from("--src-deps"),
                OsString::from("-R"),
                work_root.as_os_str().to_os_string(),
                work_root.join(relative).into_os_string(),
            ],
        )?;
        let mut dependencies = BTreeSet::new();
        for line in output.stdout.lines().filter(|line| !line.trim().is_empty()) {
            let path = PathBuf::from(line.trim());
            let canonical = fs::canonicalize(&path).map_err(|source| LeanError::Io {
                operation: "resolving Lean source dependency",
                path,
                source,
            })?;
            let Ok(dependency) = canonical.strip_prefix(&canonical_root) else {
                continue;
            };
            if proof_set.contains(dependency) {
                dependencies.insert(dependency.to_path_buf());
            } else if !generated_set.contains(dependency) {
                return Err(LeanError::InvalidProofSources(format!(
                    "{} imports undeclared proof module {}",
                    relative.display(),
                    dependency.display()
                )));
            }
        }
        pending.insert(relative.clone(), dependencies);
    }
    let mut ordered = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let ready = pending
            .iter()
            .filter_map(|(module, dependencies)| dependencies.is_empty().then_some(module.clone()))
            .collect::<Vec<_>>();
        if ready.is_empty() {
            return Err(LeanError::InvalidProofSources(format!(
                "proof module import cycle among: {}",
                pending
                    .keys()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        for module in ready {
            pending.remove(&module);
            for dependencies in pending.values_mut() {
                dependencies.remove(&module);
            }
            ordered.push(module);
        }
    }
    Ok(ordered)
}

/// Compiles one module and appends its diagnostics to the successful-run
/// result.
fn compile_module(
    lean_executable: &Path,
    work_root: &Path,
    relative: &Path,
    operation: &'static str,
    result: &mut LeanProofOutput,
) -> Result<(), LeanError> {
    let source = work_root.join(relative);
    let args = vec![
        OsString::from("-R"),
        work_root.as_os_str().to_os_string(),
        OsString::from("-o"),
        source.with_extension("olean").into_os_string(),
        source.into_os_string(),
    ];
    let output = run_lean_process(lean_executable, work_root, &[work_root], operation, &args)?;
    result.stdout.push_str(&output.stdout);
    result.stderr.push_str(&output.stderr);
    Ok(())
}

/// Checks ordered port metadata before performing any proof I/O.
fn validate_matching_interfaces(lhs: &GateFn, rhs: &GateFn) -> Result<(), LeanError> {
    validate_ports(
        InterfaceDirection::Input,
        lhs.inputs
            .iter()
            .map(|port| (port.name.as_str(), port.get_bit_count())),
        rhs.inputs
            .iter()
            .map(|port| (port.name.as_str(), port.get_bit_count())),
    )?;
    validate_ports(
        InterfaceDirection::Output,
        lhs.outputs
            .iter()
            .map(|port| (port.name.as_str(), port.get_bit_count())),
        rhs.outputs
            .iter()
            .map(|port| (port.name.as_str(), port.get_bit_count())),
    )
}

/// Reports the first count, name, or width mismatch for one port direction.
fn validate_ports<'lhs, 'rhs>(
    direction: InterfaceDirection,
    lhs: impl ExactSizeIterator<Item = (&'lhs str, usize)>,
    rhs: impl ExactSizeIterator<Item = (&'rhs str, usize)>,
) -> Result<(), LeanError> {
    if lhs.len() != rhs.len() {
        return Err(LeanError::InterfacePortCountMismatch {
            direction,
            lhs: lhs.len(),
            rhs: rhs.len(),
        });
    }
    for (port_index, ((lhs_name, lhs_width), (rhs_name, rhs_width))) in lhs.zip(rhs).enumerate() {
        if lhs_name != rhs_name {
            return Err(LeanError::InterfacePortNameMismatch {
                direction,
                port_index,
                lhs: lhs_name.to_string(),
                rhs: rhs_name.to_string(),
            });
        }
        if lhs_width != rhs_width {
            return Err(LeanError::InterfacePortWidthMismatch {
                direction,
                port_index,
                lhs: lhs_width,
                rhs: rhs_width,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "lean-tests")]
    use std::path::{Path, PathBuf};

    use super::LeanTheoremName;
    #[cfg(feature = "lean-tests")]
    use super::{
        LeanError, LeanProofOutput, compile_module, read_compiled_modules, verify_compiled_proof,
        write_file,
    };

    #[test]
    fn theorem_selector_accepts_names_and_rejects_lean_expressions() {
        for name in ["equivalent", "Adder.Proofs.eq'", "Adder.eq₁"] {
            assert_eq!(LeanTheoremName::new(name).unwrap().as_str(), name);
        }
        for name in [
            "",
            "Adder..eq",
            "proof.",
            "_root_.foo",
            "by rfl",
            "Foo.(eq)",
            "a\nb",
        ] {
            assert!(LeanTheoremName::new(name).is_err(), "{name:?}");
        }
    }

    #[cfg(feature = "lean-tests")]
    #[test]
    fn rejects_a_precompiled_proof_for_a_different_target() {
        let lean = PathBuf::from(std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN"));
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut output = LeanProofOutput {
            lean_version: String::new(),
            native_checks_replayed: None,
            stdout: String::new(),
            stderr: String::new(),
        };
        let mut compile = |name: &str, source: &str| -> Result<(), LeanError> {
            let path = Path::new(name);
            write_file(root, path, source.as_bytes())?;
            compile_module(&lean, root, path, "compiling test fixture", &mut output)
        };

        compile(
            "Target.lean",
            "abbrev Target.Equivalent : Prop := (0 : Nat) = 1",
        )
        .unwrap();
        let original = read_compiled_modules(root, &["Target.lean".into()]).unwrap();
        compile(
            "Target.lean",
            "abbrev Target.Equivalent : Prop := (0 : Nat) = 0",
        )
        .unwrap();
        compile(
            "UserProof.lean",
            "import Target\ntheorem UserProof.equivalent : Target.Equivalent := rfl",
        )
        .unwrap();

        // Its declared type has the correct name, but its proof body only
        // typechecks against the second definition of Target.Equivalent.
        let Err(failure) = verify_compiled_proof(
            &lean,
            root,
            original,
            &["UserProof.lean".into()],
            None,
            &mut output,
        ) else {
            panic!("accepted a proof compiled against a different target");
        };
        assert!(
            matches!(
                &failure,
                LeanError::ProcessFailed {
                    operation: "equivalence check",
                    stderr,
                    ..
                } if stderr.contains("while replaying declaration 'UserProof.equivalent'")
                    && stderr.contains("declaration type mismatch")
            ),
            "unexpected error: {failure}"
        );
    }
}
