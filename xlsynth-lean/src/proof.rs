// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};

use xlsynth_g8r::aig::GateFn;

use crate::gate_fn::export_gate_fn_aag;
use crate::process::{resolve_lean_executable, run_lean_process};
use crate::support::{
    lean_support_module_paths, lean_support_paths, write_file, write_lean_support,
    write_proof_support,
};
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
        write_lean_support(root)?;
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

/// Captured diagnostics from a successful equivalence proof check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeanProofOutput {
    /// Version string reported by the selected Lean executable.
    pub lean_version: String,
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
    let lean_executable = resolve_lean_executable(lean_executable)?;
    let work_root = absolute_path(work_root)?;
    prepare_empty_work_directory(&work_root)?;
    copy_prepared_target(target_root, &work_root)?;
    let proof_modules = stage_proof_sources(proof_sources, &work_root)?;
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
        stdout: String::new(),
        stderr: String::new(),
    };

    let generated_modules = lean_support_module_paths()
        .map(PathBuf::from)
        .chain([PathBuf::from("Target.lean")])
        .collect::<Vec<_>>();
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
    compile_module(
        &lean_executable,
        &work_root,
        Path::new("Check.lean"),
        "equivalence check",
        &mut result,
    )
    .map_err(|error| LeanProofFailure::with_output(error, &result))?;
    Ok(result)
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

/// Stages the prepared circuits and their support files alongside the checker's
/// proof library.
fn copy_prepared_target(source_root: &Path, work_root: &Path) -> Result<(), LeanError> {
    for relative in lean_support_paths().chain(TARGET_ARTIFACTS.iter().copied()) {
        let source_path = source_root.join(relative);
        let contents = fs::read(&source_path).map_err(|source| LeanError::Io {
            operation: "reading prepared Lean target",
            path: source_path,
            source,
        })?;
        write_file(work_root, Path::new(relative), &contents)?;
    }
    // The checker supplies its tactics even for targets prepared before they
    // existed.
    write_proof_support(work_root)
}

/// Stages helpers and generates the proof module around the body.
fn stage_proof_sources(
    sources: &LeanProofSources,
    work_root: &Path,
) -> Result<Vec<PathBuf>, LeanError> {
    let root = absolute_path(&sources.root)?;
    let body = absolute_path(&sources.body)?;
    if body.extension().and_then(|value| value.to_str()) != Some("lean")
        || body.strip_prefix(&root).is_err()
    {
        return Err(LeanError::InvalidProofSources(format!(
            "proof body {} must be a .lean file under proof root {}",
            body.display(),
            root.display()
        )));
    }
    let mut staged = BTreeMap::<PathBuf, PathBuf>::new();
    let mut modules = vec![PathBuf::from("UserProof.lean")];
    let mut imports = Vec::with_capacity(sources.modules.len());
    let reserved = lean_support_paths()
        .chain(TARGET_ARTIFACTS.iter().copied())
        .chain(["UserProof.lean"])
        .map(PathBuf::from)
        .collect::<BTreeSet<_>>();
    for source_path in &sources.modules {
        let source_path = absolute_path(source_path)?;
        if source_path == body {
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
        imports.push(proof_module_name(&relative)?);
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
    let body_source = fs::read_to_string(&body).map_err(|source| LeanError::Io {
        operation: "reading Lean proof body",
        path: body,
        source,
    })?;
    let source = user_proof_source(&body_source, &imports);
    write_file(work_root, Path::new("UserProof.lean"), source.as_bytes())?;
    Ok(modules)
}

/// Generates the imports and fixed theorem declaration around a Lean term.
fn user_proof_source(body: &str, imports: &[LeanModuleName]) -> String {
    let mut source = String::from(
        "-- SPDX-License-Identifier: Apache-2.0\n\nimport Target\nimport XlsynthLean.Proof\n",
    );
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
fn proof_module_name(path: &Path) -> Result<LeanModuleName, LeanError> {
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
        ["Generated", "Target", "Check", "UserProof"]
            .iter()
            .any(|reserved| part.eq_ignore_ascii_case(reserved))
    }) {
        return Err(LeanError::InvalidProofSources(format!(
            "proof module path {} uses a generated module root",
            path.display()
        )));
    }
    LeanModuleName::new(parts.join("."))
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
