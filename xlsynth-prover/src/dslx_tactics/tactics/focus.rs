// SPDX-License-Identifier: Apache-2.0

//! Focus tactic: prove a set of function pairs equivalent and a skeleton
//! equivalence with those functions abstracted as shared UFs.

use crate::dslx_tactics::obligations::{ObligationPayload, ProverObligation, Side};
use crate::dslx_tactics::tactics::IsTactic;
use crate::dslx_tactics::tactics::utils::is_valid_ident;
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

    fn apply(&self, base: &ProverObligation) -> Result<Vec<ProverObligation>, String> {
        // Basic validation
        for (idx, p) in self.pairs.iter().enumerate() {
            if !is_valid_ident(&p.lhs) {
                return Err(format!("lhs pair {} invalid function name", idx + 1));
            }
            if !is_valid_ident(&p.rhs) {
                return Err(format!("rhs pair {} invalid function name", idx + 1));
            }
        }

        let base_lec = match &base.payload {
            ObligationPayload::Lec(lec) => lec,
            _ => return Err("focus tactic requires a Lec obligation".to_string()),
        };

        let mut obligations: Vec<ProverObligation> = Vec::new();

        // 1) Per-pair equivalence checks
        for (idx, p) in self.pairs.iter().enumerate() {
            let mut ob = ProverObligation {
                selector_segment: format!("pair_{}", idx + 1),
                description: None,
                payload: ObligationPayload::Lec(base_lec.clone()),
            };
            if let ObligationPayload::Lec(lec) = &mut ob.payload {
                lec.set_top(Side::Lhs, &p.lhs);
                lec.set_top(Side::Rhs, &p.rhs);
            }
            obligations.push(ob);
        }

        // 2) Skeleton with shared UFs for focused functions
        {
            let mut ob = ProverObligation {
                selector_segment: "skeleton".to_string(),
                description: None,
                payload: ObligationPayload::Lec(base_lec.clone()),
            };
            if let ObligationPayload::Lec(lec) = &mut ob.payload {
                for (idx, p) in self.pairs.iter().enumerate() {
                    let uf = format!("__uf_focus_{}", idx + 1);
                    lec.set_uf_alias(&p.lhs, &p.rhs, &uf);
                }
            }
            obligations.push(ob);
        }

        Ok(obligations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dslx_tactics::obligations::{
        FileWithHistory, LecObligation, LecSide, ObligationPayload, ProverObligation, SourceFile,
    };

    fn base_obligation_with(sel: &str) -> ProverObligation {
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
        ProverObligation {
            selector_segment: sel.to_string(),
            description: None,
            payload: ObligationPayload::Lec(LecObligation { lhs, rhs }),
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
        let p1_lec = match &p1.payload {
            ObligationPayload::Lec(lec) => lec,
            _ => panic!("expected Lec payload"),
        };
        assert_eq!(p1_lec.lhs.top_func, "a");
        assert_eq!(p1_lec.rhs.top_func, "b");
        assert_eq!(p1_lec.lhs.file.edits.len(), 0);
        assert_eq!(p1_lec.rhs.file.edits.len(), 0);

        let p2 = obs.iter().find(|o| o.selector_segment == "pair_2").unwrap();
        let p2_lec = match &p2.payload {
            ObligationPayload::Lec(lec) => lec,
            _ => panic!("expected Lec payload"),
        };
        assert_eq!(p2_lec.lhs.top_func, "c");
        assert_eq!(p2_lec.rhs.top_func, "d");

        let skel = obs
            .iter()
            .find(|o| o.selector_segment == "skeleton")
            .unwrap();
        let skel_lec = match &skel.payload {
            ObligationPayload::Lec(lec) => lec,
            _ => panic!("expected Lec payload"),
        };
        assert_eq!(skel_lec.lhs.top_func, "f1");
        assert_eq!(skel_lec.rhs.top_func, "f2");
        assert_eq!(
            skel_lec.lhs.uf_map.get("a").map(|s| s.as_str()),
            Some("__uf_focus_1")
        );
        assert_eq!(
            skel_lec.rhs.uf_map.get("b").map(|s| s.as_str()),
            Some("__uf_focus_1")
        );
        assert_eq!(
            skel_lec.lhs.uf_map.get("c").map(|s| s.as_str()),
            Some("__uf_focus_2")
        );
        assert_eq!(
            skel_lec.rhs.uf_map.get("d").map(|s| s.as_str()),
            Some("__uf_focus_2")
        );
    }
}
