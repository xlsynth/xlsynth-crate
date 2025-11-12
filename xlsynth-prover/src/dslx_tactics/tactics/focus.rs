// SPDX-License-Identifier: Apache-2.0

//! Focus tactic: prove a set of function pairs equivalent and a skeleton
//! equivalence with those functions abstracted as shared UFs.

use crate::dslx_tactics::obligations::{
    FileWithHistory, LecObligation, LecSide, ObligationPayload, ProverObligation, QcObligation,
    Side,
};
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

        match &base.payload {
            ObligationPayload::Lec(lec) => self.apply_to_lec(lec),
            ObligationPayload::QuickCheck(qc) => self.apply_to_qc(qc),
        }
    }
}

impl FocusTactic {
    fn apply_to_lec(&self, base: &LecObligation) -> Result<Vec<ProverObligation>, String> {
        let mut obligations: Vec<ProverObligation> = Vec::new();

        for (idx, pair) in self.pairs.iter().enumerate() {
            let mut lec = base.clone();
            lec.set_top(Side::Lhs, &pair.lhs);
            lec.set_top(Side::Rhs, &pair.rhs);
            obligations.push(ProverObligation {
                selector_segment: format!("pair_{}", idx + 1),
                description: None,
                payload: ObligationPayload::Lec(lec),
            });
        }

        let mut skeleton = base.clone();
        for pair in self.pairs.iter() {
            skeleton.set_next_uf_alias(&pair.lhs, &pair.rhs, "__uf_focus");
        }
        obligations.push(ProverObligation {
            selector_segment: "skeleton".to_string(),
            description: None,
            payload: ObligationPayload::Lec(skeleton),
        });

        Ok(obligations)
    }

    fn apply_to_qc(&self, base: &QcObligation) -> Result<Vec<ProverObligation>, String> {
        if self.pairs.is_empty() {
            return Err(
                "focus tactic requires at least one pair for QuickCheck obligations".to_string(),
            );
        }

        let mut obligations: Vec<ProverObligation> = Vec::new();

        for (idx, pair) in self.pairs.iter().enumerate() {
            obligations.push(ProverObligation {
                selector_segment: format!("pair_{}", idx + 1),
                description: None,
                payload: ObligationPayload::Lec(self.qc_pair_to_lec(base, pair)),
            });
        }

        let mut skeleton = base.clone();
        for pair in self.pairs.iter() {
            skeleton.set_next_uf_alias(&pair.lhs, &pair.rhs, "__uf_focus");
        }
        obligations.push(ProverObligation {
            selector_segment: "skeleton".to_string(),
            description: None,
            payload: ObligationPayload::QuickCheck(skeleton),
        });

        Ok(obligations)
    }

    fn qc_pair_to_lec(&self, base: &QcObligation, pair: &FocusPair) -> LecObligation {
        // let stripped_text = Self::strip_quickcheck_functions(&base.file.text);
        let file = FileWithHistory::from_text(&base.file.text);
        LecObligation {
            lhs: LecSide {
                top_func: pair.lhs.clone(),
                uf_map: base.uf_map.clone(),
                file: file.clone(),
            },
            rhs: LecSide {
                top_func: pair.rhs.clone(),
                uf_map: base.uf_map.clone(),
                file,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dslx_tactics::obligations::{
        FileWithHistory, LecObligation, LecSide, ObligationPayload, ProverObligation, QcObligation,
        SourceFile,
    };
    use std::collections::BTreeMap;

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

    #[test]
    fn transforms_quickcheck_into_lec_and_qc_skeleton() {
        let qc = QcObligation {
            file: FileWithHistory::from_text(
                "pub fn lhs(x: u32) -> u32 { x + u32:1 }\n\
                 pub fn rhs(x: u32) -> u32 { x + u32:1 }\n",
            ),
            tests: vec!["prop_equiv".to_string()],
            uf_map: BTreeMap::from([("shared".to_string(), "UF_SHARED".to_string())]),
        };
        let base = ProverObligation {
            selector_segment: "root".to_string(),
            description: None,
            payload: ObligationPayload::QuickCheck(qc),
        };
        let t = FocusTactic {
            pairs: vec![FocusPair {
                lhs: "lhs".to_string(),
                rhs: "rhs".to_string(),
            }],
        };

        let obs = t.apply(&base).expect("quickcheck focus");
        assert_eq!(obs.len(), 2);

        let pair = obs
            .iter()
            .find(|o| o.selector_segment == "pair_1")
            .expect("pair obligation");
        let pair_lec = match &pair.payload {
            ObligationPayload::Lec(lec) => lec,
            _ => panic!("expected Lec payload"),
        };
        assert_eq!(pair_lec.lhs.top_func, "lhs");
        assert_eq!(pair_lec.rhs.top_func, "rhs");
        assert_eq!(
            pair_lec.lhs.uf_map.get("shared").map(|s| s.as_str()),
            Some("UF_SHARED")
        );
        assert_eq!(
            pair_lec.rhs.uf_map.get("shared").map(|s| s.as_str()),
            Some("UF_SHARED")
        );

        let skeleton = obs
            .iter()
            .find(|o| o.selector_segment == "skeleton")
            .expect("skeleton obligation");
        let skeleton_qc = match &skeleton.payload {
            ObligationPayload::QuickCheck(qc) => qc,
            _ => panic!("expected QuickCheck payload"),
        };
        assert_eq!(
            skeleton_qc.uf_map.get("shared").map(|s| s.as_str()),
            Some("UF_SHARED")
        );
        assert_eq!(
            skeleton_qc.uf_map.get("lhs").map(|s| s.as_str()),
            Some("__uf_focus_2")
        );
        assert_eq!(
            skeleton_qc.uf_map.get("rhs").map(|s| s.as_str()),
            Some("__uf_focus_2")
        );
        assert!(skeleton_qc.file.text.contains("pub fn lhs"));
        assert!(skeleton_qc.file.text.contains("pub fn rhs"));
    }
}
