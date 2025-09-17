// SPDX-License-Identifier: Apache-2.0

//! Focus tactic: prove a set of function pairs equivalent and a skeleton
//! equivalence with those functions abstracted as shared UFs.

use crate::proofs::obligations::{LecObligation, Side};
use crate::proofs::tactics::utils::is_valid_ident;
use crate::proofs::tactics::IsTactic;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FocusPair {
    pub lhs: String,
    pub rhs: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FocusTactic {
    pub pairs: Vec<FocusPair>,
}

impl IsTactic for FocusTactic {
    fn name(&self) -> &'static str {
        "focus"
    }

    fn apply(&self, base: &LecObligation) -> Result<Vec<LecObligation>, String> {
        // Basic validation
        for (idx, p) in self.pairs.iter().enumerate() {
            if !is_valid_ident(&p.lhs) {
                return Err(format!("lhs pair {} invalid function name", idx + 1));
            }
            if !is_valid_ident(&p.rhs) {
                return Err(format!("rhs pair {} invalid function name", idx + 1));
            }
        }

        let mut obligations: Vec<LecObligation> = Vec::new();

        // 1) Per-pair equivalence checks
        for (idx, p) in self.pairs.iter().enumerate() {
            let mut ob = base.clone();
            ob.set_selector_segment(&format!("pair_{}", idx + 1));
            ob.set_top(Side::Lhs, &p.lhs);
            ob.set_top(Side::Rhs, &p.rhs);
            obligations.push(ob);
        }

        // 2) Skeleton with shared UFs for focused functions
        {
            let mut ob = base.clone();
            ob.set_selector_segment("skeleton");
            for (idx, p) in self.pairs.iter().enumerate() {
                let uf = format!("__uf_focus_{}", idx + 1);
                ob.set_uf_alias(&p.lhs, &p.rhs, &uf);
            }
            obligations.push(ob);
        }

        Ok(obligations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::obligations::{FileWithHistory, LecSide, SourceFile};

    fn base_obligation_with(sel: &str) -> LecObligation {
        let lhs = LecSide {
            top_func: "f1".to_string(),
            uf_map: std::collections::BTreeMap::new(),
            file: FileWithHistory {
                base_source: SourceFile::Text(String::new()),
                edits: Vec::new(),
                text: String::new(),
            },
        };
        let rhs = LecSide {
            top_func: "f2".to_string(),
            uf_map: std::collections::BTreeMap::new(),
            file: FileWithHistory {
                base_source: SourceFile::Text(String::new()),
                edits: Vec::new(),
                text: String::new(),
            },
        };
        LecObligation {
            selector_segment: sel.to_string(),
            lhs,
            rhs,
            description: None,
        }
    }

    #[test]
    fn builds_expected_focus_obligations() {
        let base = base_obligation_with("root");
        let t = FocusTactic {
            pairs: vec![
                FocusPair {
                    lhs: "a".to_string(),
                    rhs: "b".to_string(),
                },
                FocusPair {
                    lhs: "c".to_string(),
                    rhs: "d".to_string(),
                },
            ],
        };
        let obs = t.apply(&base).expect("ok");
        assert_eq!(obs.len(), 3);

        let p1 = obs.iter().find(|o| o.selector_segment == "pair_1").unwrap();
        assert_eq!(p1.lhs.top_func, "a");
        assert_eq!(p1.rhs.top_func, "b");
        assert_eq!(p1.lhs.file.edits.len(), 0);
        assert_eq!(p1.rhs.file.edits.len(), 0);

        let p2 = obs.iter().find(|o| o.selector_segment == "pair_2").unwrap();
        assert_eq!(p2.lhs.top_func, "c");
        assert_eq!(p2.rhs.top_func, "d");

        let skel = obs
            .iter()
            .find(|o| o.selector_segment == "skeleton")
            .unwrap();
        assert_eq!(skel.lhs.top_func, "f1");
        assert_eq!(skel.rhs.top_func, "f2");
        assert_eq!(
            skel.lhs.uf_map.get("a").map(|s| s.as_str()),
            Some("__uf_focus_1")
        );
        assert_eq!(
            skel.rhs.uf_map.get("b").map(|s| s.as_str()),
            Some("__uf_focus_1")
        );
        assert_eq!(
            skel.lhs.uf_map.get("c").map(|s| s.as_str()),
            Some("__uf_focus_2")
        );
        assert_eq!(
            skel.rhs.uf_map.get("d").map(|s| s.as_str()),
            Some("__uf_focus_2")
        );
    }
}
