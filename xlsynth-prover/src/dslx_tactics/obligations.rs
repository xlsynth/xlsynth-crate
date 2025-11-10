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
    pub selector_segment: String,
    pub lhs: LecSide,
    pub rhs: LecSide,
    pub description: Option<String>,
}

pub enum Side {
    Lhs,
    Rhs,
}

impl LecObligation {
    pub fn clone_lhs_self(&self, selector: &str, description: Option<&str>) -> Self {
        Self {
            selector_segment: selector.to_string(),
            lhs: self.lhs.clone(),
            rhs: self.lhs.clone(),
            description: description.map(|s| s.to_string()),
        }
    }

    pub fn clone_rhs_self(&self, selector: &str, description: Option<&str>) -> Self {
        Self {
            selector_segment: selector.to_string(),
            lhs: self.rhs.clone(),
            rhs: self.rhs.clone(),
            description: description.map(|s| s.to_string()),
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

    pub fn set_selector_segment(&mut self, selector: &str) -> &mut Self {
        self.selector_segment = selector.to_string();
        self
    }
}
