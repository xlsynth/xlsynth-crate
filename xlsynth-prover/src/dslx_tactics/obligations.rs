// SPDX-License-Identifier: Apache-2.0

//! Proof obligation abstraction.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::prover::types::{AssertionSemantics, EquivParallelism};

pub use crate::dslx_tactics::source::{Edit, FileWithHistory, SourceFile};

/// One side of an equivalence obligation, including its activation convention.
#[derive(Clone, Debug)]
pub struct LecSide {
    pub top_func: String,
    /// DSLX function names mapped to shared symbols, before IR name mangling.
    pub uf_map: BTreeMap<String, String>,
    pub file: FileWithHistory,
    pub fixed_implicit_activation: bool,
}

impl LecSide {
    pub fn from_text(top: &str, text: &str) -> Self {
        Self {
            top_func: top.to_string(),
            uf_map: BTreeMap::new(),
            file: FileWithHistory::from_text(text),
            fixed_implicit_activation: false,
        }
    }

    pub fn from_path(top: &str, path: &PathBuf) -> Self {
        Self {
            top_func: top.to_string(),
            uf_map: BTreeMap::new(),
            file: FileWithHistory::from_path(path),
            fixed_implicit_activation: false,
        }
    }

    pub fn from_source(top: &str, source: SourceFile) -> Self {
        let text = source.contents().unwrap_or_default();
        Self {
            top_func: top.to_string(),
            uf_map: BTreeMap::new(),
            file: FileWithHistory {
                base_source: source,
                edits: Vec::new(),
                text,
            },
            fixed_implicit_activation: false,
        }
    }
}

impl LecSide {
    pub fn apply_edit(&mut self, edit: Edit) -> Result<(), String> {
        self.file.apply_edit(edit)
    }
}

/// The property and input interpretation required by an equivalence obligation.
#[derive(Clone, Debug)]
pub struct LecProofOptions {
    pub assertion_semantics: AssertionSemantics,
    pub assume_enum_in_bound: bool,
    pub flatten_aggregates: bool,
    pub drop_params: Vec<String>,
    pub assert_label_filter: Option<String>,
    pub parallelism: EquivParallelism,
}

impl Default for LecProofOptions {
    fn default() -> Self {
        Self {
            assertion_semantics: AssertionSemantics::Ignore,
            assume_enum_in_bound: false,
            flatten_aggregates: false,
            drop_params: Vec::new(),
            assert_label_filter: None,
            parallelism: EquivParallelism::SingleThreaded,
        }
    }
}

/// Two DSLX functions and the precise equivalence property to prove between
/// them.
#[derive(Clone, Debug)]
pub struct LecObligation {
    pub lhs: LecSide,
    pub rhs: LecSide,
    pub options: LecProofOptions,
}

#[derive(Clone, Debug)]
pub struct QcObligation {
    pub file: FileWithHistory,
    pub tests: Vec<String>,
    pub uf_map: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub enum ObligationPayload {
    Lec(LecObligation),
    QuickCheck(QcObligation),
}

#[derive(Clone, Debug)]
pub struct ProverObligation {
    pub selector_segment: String,
    pub description: Option<String>,
    pub payload: ObligationPayload,
}

pub enum Side {
    Lhs,
    Rhs,
}

impl LecObligation {
    /// Checks the preconditions of the current UF-based decomposition rules.
    pub(crate) fn validate_decomposition(&self) -> Result<(), String> {
        if self.options.assertion_semantics != AssertionSemantics::Ignore {
            return Err("decomposition requires assertion-semantics=ignore; UF abstraction does not preserve assertion effects".to_string());
        }
        if !self.options.drop_params.is_empty() {
            return Err(
                "decomposition does not support drop_params; helper signatures may differ"
                    .to_string(),
            );
        }
        if self.options.flatten_aggregates {
            return Err("decomposition does not support flatten_aggregates=true; helper signatures must match".to_string());
        }
        if self.lhs.fixed_implicit_activation || self.rhs.fixed_implicit_activation {
            return Err("decomposition does not support fixed implicit activation; helper activation conventions may differ".to_string());
        }
        if self.options.parallelism != EquivParallelism::SingleThreaded {
            return Err("decomposition requires single-threaded parallelism; UF proofs do not support other strategies".to_string());
        }
        Ok(())
    }

    pub fn clone_lhs_self(&self) -> Self {
        Self {
            lhs: self.lhs.clone(),
            rhs: self.lhs.clone(),
            options: self.options.clone(),
        }
    }

    pub fn clone_rhs_self(&self) -> Self {
        Self {
            lhs: self.rhs.clone(),
            rhs: self.rhs.clone(),
            options: self.options.clone(),
        }
    }

    pub fn apply_edit(&mut self, side: Side, edit: Edit) -> Result<&mut Self, String> {
        match side {
            Side::Lhs => self.lhs.apply_edit(edit)?,
            Side::Rhs => self.rhs.apply_edit(edit)?,
        }
        Ok(self)
    }

    pub fn set_file(&mut self, side: Side, file: FileWithHistory) -> &mut Self {
        match side {
            Side::Lhs => self.lhs.file = file.clone(),
            Side::Rhs => self.rhs.file = file.clone(),
        }
        self
    }

    pub fn clear_uf_map(&mut self, side: Side) -> &mut Self {
        match side {
            Side::Lhs => self.lhs.uf_map.clear(),
            Side::Rhs => self.rhs.uf_map.clear(),
        }
        self
    }

    pub fn set_top(&mut self, side: Side, top: &str) -> &mut Self {
        match side {
            Side::Lhs => self.lhs.top_func = top.to_string(),
            Side::Rhs => self.rhs.top_func = top.to_string(),
        }
        self
    }

    pub fn set_uf_alias(&mut self, left: &str, right: &str, uf: &str) -> &mut Self {
        self.lhs.uf_map.insert(left.to_string(), uf.to_string());
        self.rhs.uf_map.insert(right.to_string(), uf.to_string());
        self
    }
}

impl QcObligation {
    pub fn apply_edit(&mut self, edit: Edit) -> Result<(), String> {
        self.file.apply_edit(edit)
    }

    pub fn set_tests(&mut self, tests: Vec<String>) -> &mut Self {
        self.tests = tests;
        self
    }

    pub fn add_test<S: Into<String>>(&mut self, test: S) -> &mut Self {
        self.tests.push(test.into());
        self
    }

    pub fn clear_uf_map(&mut self) -> &mut Self {
        self.uf_map.clear();
        self
    }

    pub fn set_uf_map(&mut self, map: BTreeMap<String, String>) -> &mut Self {
        self.uf_map = map;
        self
    }
}

impl ProverObligation {
    pub fn set_selector_segment(&mut self, selector: &str) -> &mut Self {
        self.selector_segment = selector.to_string();
        self
    }

    pub fn set_description(&mut self, description: Option<&str>) -> &mut Self {
        self.description = description.map(|s| s.to_string());
        self
    }
}
