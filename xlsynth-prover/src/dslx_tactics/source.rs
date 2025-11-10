// SPDX-License-Identifier: Apache-2.0

//! File content representation and edit operations used by proof obligations.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SourceFile {
    Path(PathBuf),
    Text(String),
}

impl SourceFile {
    pub fn contents(&self) -> Result<String, String> {
        match self {
            SourceFile::Text(t) => Ok(t.clone()),
            SourceFile::Path(p) => {
                std::fs::read_to_string(p).map_err(|e| format!("read {:?}: {}", p, e))
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Edit {
    AppendSource { source: SourceFile },
}

impl Edit {
    pub fn apply(&self, current: &str) -> Result<String, String> {
        match self {
            Edit::AppendSource { source } => {
                let code = source.contents()?;
                let mut out = current.to_owned();
                if !out.is_empty() {
                    out.push_str("\n\n");
                }
                out.push_str(&code);
                Ok(out)
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileWithHistory {
    pub base_source: SourceFile,
    pub edits: Vec<Edit>,
    pub text: String,
}

impl FileWithHistory {
    pub fn apply_edit(&mut self, edit: Edit) -> Result<(), String> {
        self.edits.push(edit.clone());
        self.text = edit.apply(&self.text)?;
        Ok(())
    }

    pub fn from_text(text: &str) -> Self {
        Self {
            base_source: SourceFile::Text(text.to_string()),
            edits: Vec::new(),
            text: text.to_string(),
        }
    }

    pub fn from_path(path: &PathBuf) -> Self {
        let file = SourceFile::Path(path.clone());
        let text = file.contents().unwrap();
        Self {
            base_source: file,
            edits: Vec::new(),
            text,
        }
    }
}
