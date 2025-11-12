// SPDX-License-Identifier: Apache-2.0

//! Proof obligation abstraction.

use std::collections::BTreeMap;
use std::path::PathBuf;

pub use crate::dslx_tactics::source::{Edit, FileWithHistory, SourceFile};

#[derive(Clone, Debug)]
pub struct LecSide {
    pub top_func: String,
    pub uf_map: BTreeMap<String, String>,
    pub file: FileWithHistory,
}

impl LecSide {
    pub fn from_text(top: &str, text: &str) -> Self {
        Self {
            top_func: top.to_string(),
            uf_map: BTreeMap::new(),
            file: FileWithHistory::from_text(text),
        }
    }

    pub fn from_path(top: &str, path: &PathBuf) -> Self {
        Self {
            top_func: top.to_string(),
            uf_map: BTreeMap::new(),
            file: FileWithHistory::from_path(path),
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
        }
    }
}

impl LecSide {
    pub fn apply_edit(&mut self, edit: Edit) -> Result<(), String> {
        self.file.apply_edit(edit)
    }
}

#[derive(Clone, Debug)]
pub struct LecObligation {
    pub lhs: LecSide,
    pub rhs: LecSide,
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
    pub fn clone_lhs_self(&self) -> Self {
        Self {
            lhs: self.lhs.clone(),
            rhs: self.lhs.clone(),
        }
    }

    pub fn clone_rhs_self(&self) -> Self {
        Self {
            lhs: self.rhs.clone(),
            rhs: self.rhs.clone(),
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
