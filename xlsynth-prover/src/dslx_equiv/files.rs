// SPDX-License-Identifier: Apache-2.0

//! File-based DSLX equivalence, either proved directly or decomposed by
//! tactics.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::{DslxEquivRequest, DslxModule, DslxOptions, run_dslx_equiv};
use crate::dslx_specializer::{SpecializedModule, specialize_dslx_module};
use crate::dslx_tactics::obligations::{
    FileWithHistory, LecObligation, LecProofOptions, LecSide, ObligationPayload, ProverObligation,
    SourceFile,
};
use crate::dslx_tactics::script::{OblTree, OblTreeConfig};
use crate::dslx_utils::{mangle_dslx_uf_map, validate_dslx_paths};
use crate::proof_outcome::ProofSummary;
use crate::prover::SolverChoice;
use crate::prover::uf::validate_uf_symbols;

/// One DSLX source, its top (which may have parametric bindings), and its
/// assumptions.
pub struct DslxEquivFile<'a> {
    pub path: &'a Path,
    pub top: &'a str,
    /// DSLX function names mapped to shared uninterpreted symbols.
    pub uf_map: HashMap<String, String>,
    pub fixed_implicit_activation: bool,
}

impl DslxEquivFile<'_> {
    fn source(&self) -> Result<String, String> {
        std::fs::read_to_string(self.path)
            .map_err(|e| format!("Failed to read DSLX file {}: {e}", self.path.display()))
    }

    /// Retains the original top and source so each generated obligation can
    /// select its own top.
    fn into_lec_side(self) -> Result<LecSide, String> {
        validate_uf_symbols(&self.uf_map)?;
        Ok(LecSide {
            top_func: self.top.to_owned(),
            file: FileWithHistory {
                text: self.source()?,
                base_source: SourceFile::Path(self.path.to_owned()),
                edits: Vec::new(),
            },
            fixed_implicit_activation: self.fixed_implicit_activation,
            uf_map: self.uf_map.into_iter().collect(),
        })
    }

    fn specialize(
        &self,
        stdlib: Option<&Path>,
        search: &[PathBuf],
    ) -> Result<SpecializedModule, String> {
        specialize_dslx_module(&self.source()?, self.path, self.top, stdlib, search).map_err(|e| {
            format!(
                "DSLX specialization failed\n  path: {}\n  error: {e}",
                self.path.display()
            )
        })
    }

    fn as_module<'a>(
        &'a self,
        specialized: &'a SpecializedModule,
    ) -> Result<DslxModule<'a>, String> {
        let module_name = self
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                format!(
                    "Failed to derive module name from path: {}",
                    self.path.display()
                )
            })?;
        Ok(DslxModule::new(&specialized.source, &specialized.top_name)
            .with_path(Some(self.path))
            .with_owned_uf_map(mangle_dslx_uf_map(module_name, &self.uf_map)?)
            .with_fixed_implicit_activation(self.fixed_implicit_activation))
    }
}

/// A single equivalence property with the same DSLX names and assumptions in
/// direct and tactic proofs.
pub struct DslxEquivFiles<'a> {
    pub lhs: DslxEquivFile<'a>,
    pub rhs: DslxEquivFile<'a>,
    pub options: LecProofOptions,
    pub solver: Option<SolverChoice>,
    pub dslx_stdlib_path: Option<&'a Path>,
    pub additional_search_paths: Vec<PathBuf>,
}

impl DslxEquivFiles<'_> {
    fn validate_paths(&self) -> Result<(), String> {
        validate_dslx_paths(
            [self.lhs.path, self.rhs.path]
                .into_iter()
                .chain(self.dslx_stdlib_path)
                .chain(self.additional_search_paths.iter().map(PathBuf::as_path)),
        )
    }

    /// Creates the root obligation without pruning helper definitions that
    /// tactics may target.
    pub fn into_tactic_tree(self) -> Result<OblTree, String> {
        self.validate_paths()?;
        let root = LecObligation {
            lhs: self.lhs.into_lec_side()?,
            rhs: self.rhs.into_lec_side()?,
            options: self.options,
        };
        Ok(OblTree::new(
            ProverObligation {
                selector_segment: String::new(),
                description: None,
                payload: ObligationPayload::Lec(root),
            },
            OblTreeConfig {
                solver: self.solver,
                dslx_paths: self.additional_search_paths,
                dslx_stdlib_path: self.dslx_stdlib_path.map(Path::to_owned),
                timeout_ms: None,
            },
        ))
    }

    /// Specializes both tops, preserves calls marked uninterpreted, and proves
    /// the requested equivalence.
    pub fn prove(
        self,
        tool_path: Option<&Path>,
        enable_warnings: Option<&[String]>,
        disable_warnings: Option<&[String]>,
    ) -> ProofSummary {
        ProofSummary::run(|| {
            self.validate_paths()?;
            let lhs = self
                .lhs
                .specialize(self.dslx_stdlib_path, &self.additional_search_paths)?;
            let rhs = self
                .rhs
                .specialize(self.dslx_stdlib_path, &self.additional_search_paths)?;
            // IR optimization can inline calls before the prover substitutes
            // their UF symbols.
            let optimize = self.lhs.uf_map.is_empty() && self.rhs.uf_map.is_empty();
            let request = DslxEquivRequest {
                lhs: self.lhs.as_module(&lhs)?,
                rhs: self.rhs.as_module(&rhs)?,
                solver: self.solver,
                drop_params: &self.options.drop_params,
                flatten_aggregates: self.options.flatten_aggregates,
                parallelism: self.options.parallelism,
                assertion_semantics: self.options.assertion_semantics,
                assert_label_filter: self.options.assert_label_filter.as_deref(),
                assume_enum_in_bound: self.options.assume_enum_in_bound,
                options: DslxOptions {
                    tool_path,
                    enable_warnings,
                    disable_warnings,
                    optimize,
                    dslx_stdlib_path: self.dslx_stdlib_path,
                    additional_search_paths: self.additional_search_paths,
                },
            };
            run_dslx_equiv(&request)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_outcome::ProofStatus;
    #[cfg(unix)]
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

    /// Reject a non-UTF-8 parent before reading the source, even when the
    /// module name itself is valid.
    #[cfg(unix)]
    #[test]
    fn unrepresentable_source_path_does_not_start_a_proof() {
        let path = Path::new(OsStr::from_bytes(b"dir_\xff/entry.x"));
        let side = || DslxEquivFile {
            path,
            top: "main",
            uf_map: HashMap::new(),
            fixed_implicit_activation: false,
        };
        let request = || DslxEquivFiles {
            lhs: side(),
            rhs: side(),
            options: LecProofOptions::default(),
            solver: None,
            dslx_stdlib_path: None,
            additional_search_paths: Vec::new(),
        };
        let message = format!("DSLX path is not UTF-8: {path:?}");
        let outcome = request().prove(None, None, None);
        assert_eq!(outcome.conclusion.status, ProofStatus::ExecutionError);
        assert_eq!(outcome.error_str.as_ref(), Some(&message));
        assert_eq!(request().into_tactic_tree().unwrap_err(), message);
    }

    /// Shared UF symbols may abstract different helpers, but never a difference
    /// in the caller.
    #[cfg(feature = "has-bitwuzla")]
    #[test]
    fn file_equivalence_preserves_top_bindings_and_uninterpreted_calls() {
        let dir = tempfile::tempdir().unwrap();
        let lhs = dir.path().join("lhs.x");
        let rhs = dir.path().join("rhs.x");
        for (path, op, name) in [(&lhs, "+", "g"), (&rhs, "-", "h")] {
            std::fs::write(
                path,
                format!(
                    r#"
fn {name}(x: u8) -> u8 {{ x {op} u8:1 }}
pub fn main<N: u32>(x: u8) -> u8 {{ {name}(x) + (N as u8) }}
"#
                ),
            )
            .unwrap();
        }
        for (symbol, rhs_top, expected) in [
            (None, "main<u32:2>", ProofStatus::Disproved),
            (Some("F"), "main<u32:2>", ProofStatus::Proved),
            (Some("F"), "main<u32:3>", ProofStatus::Disproved),
            (Some("F\0x"), "main<u32:2>", ProofStatus::ExecutionError),
        ] {
            let mappings = |name: &str| {
                symbol
                    .map(|s| HashMap::from([(name.to_owned(), s.to_owned())]))
                    .unwrap_or_default()
            };
            let request = || DslxEquivFiles {
                lhs: DslxEquivFile {
                    path: &lhs,
                    top: "main<u32:2>",
                    uf_map: mappings("g"),
                    fixed_implicit_activation: false,
                },
                rhs: DslxEquivFile {
                    path: &rhs,
                    top: rhs_top,
                    uf_map: mappings("h"),
                    fixed_implicit_activation: false,
                },
                options: LecProofOptions::default(),
                solver: Some(SolverChoice::Bitwuzla),
                dslx_stdlib_path: None,
                additional_search_paths: Vec::new(),
            };
            let result = request().prove(None, None, None);
            assert_eq!(result.conclusion.status, expected, "{result:?}");
            if expected == ProofStatus::ExecutionError {
                assert!(request().into_tactic_tree().is_err());
            }
        }
    }

    /// External conversion must see the specialized top and resolve its
    /// original imports.
    #[cfg(feature = "has-bitwuzla")]
    #[test]
    fn external_converter_uses_specialized_sources_and_search_paths() {
        let dir = tempfile::tempdir().unwrap();
        let tools = std::env::var_os("XLSYNTH_TOOLS")
            .expect("XLSYNTH_TOOLS is required for toolchain tests");
        let deps = dir.path().join("deps");
        std::fs::create_dir(&deps).unwrap();
        std::fs::write(
            deps.join("arithmetic.x"),
            "pub fn bump(x: u32) -> u32 { x + u32:1 }",
        )
        .unwrap();
        let lhs = dir.path().join("parametric.x");
        let rhs = dir.path().join("fixed.x");
        std::fs::write(
            &lhs,
            r#"
import arithmetic;
pub fn main<N: u32>(x: u32) -> u32 { arithmetic::bump(x) + N }
"#,
        )
        .unwrap();
        std::fs::write(&rhs, "pub fn main(x: u32) -> u32 { x + u32:5 }").unwrap();

        let request = DslxEquivFiles {
            lhs: DslxEquivFile {
                path: &lhs,
                top: "main<u32:4>",
                uf_map: HashMap::new(),
                fixed_implicit_activation: false,
            },
            rhs: DslxEquivFile {
                path: &rhs,
                top: "main",
                uf_map: HashMap::new(),
                fixed_implicit_activation: false,
            },
            // Without enum domains, the external tools perform IR conversion.
            options: LecProofOptions::default(),
            solver: Some(SolverChoice::Bitwuzla),
            dslx_stdlib_path: None,
            additional_search_paths: vec![dir.path().to_owned(), deps],
        };
        let result = request.prove(Some(Path::new(&tools)), None, None);
        assert_eq!(result.conclusion.status, ProofStatus::Proved, "{result:?}");
    }
}
