// SPDX-License-Identifier: Apache-2.0

//! Cosliced tactic: transforms a base ProofObligation into sub-obligations.
//!
//! This mirrors the Python variant's obligation construction logic without any
//! model- or bus-related functionality.

use crate::dslx_tactics::obligations::{
    Edit, ObligationPayload, ProverObligation, Side, SourceFile,
};
use crate::dslx_tactics::tactics::IsTactic;
use crate::dslx_tactics::tactics::utils::is_valid_ident;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedSlice {
    pub func_name: String,
    pub code: SourceFile,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoslicedTactic {
    pub lhs_slices: Vec<NamedSlice>,
    pub rhs_slices: Vec<NamedSlice>,
    pub lhs_composed: NamedSlice,
    pub rhs_composed: NamedSlice,
}

impl IsTactic for CoslicedTactic {
    fn name(&self) -> &'static str {
        "cosliced"
    }

    fn apply(&self, base: &ProverObligation) -> Result<Vec<ProverObligation>, String> {
        let file1_slices = &self.lhs_slices;
        let file2_slices = &self.rhs_slices;
        let file1_composed = &self.lhs_composed;
        let file2_composed = &self.rhs_composed;

        let k = validate_and_collect(file1_slices, file2_slices)?;
        if !is_valid_ident(&file1_composed.func_name) {
            return Err("lhs_composed invalid function name".to_string());
        }
        if !is_valid_ident(&file2_composed.func_name) {
            return Err("rhs_composed invalid function name".to_string());
        }

        let base_lec = match &base.payload {
            ObligationPayload::Lec(lec) => lec,
            _ => return Err("cosliced tactic requires a Lec obligation".to_string()),
        };

        let mut obligations: Vec<ProverObligation> = Vec::new();

        // LHS self-equivalence
        {
            let mut lec = base_lec.clone_lhs_self();
            for sl in file1_slices.iter() {
                lec.apply_edit(
                    Side::Rhs,
                    Edit::AppendSource {
                        source: sl.code.clone(),
                    },
                )?;
            }
            lec.apply_edit(
                Side::Rhs,
                Edit::AppendSource {
                    source: file1_composed.code.clone(),
                },
            )?;
            lec.set_top(Side::Rhs, &file1_composed.func_name);

            let ob = ProverObligation {
                selector_segment: "lhs_self".to_string(),
                description: Some("LHS self-equivalence".to_string()),
                payload: ObligationPayload::Lec(lec),
            };
            obligations.push(ob);
        }

        // RHS self-equivalence
        {
            let mut lec = base_lec.clone_rhs_self();
            for sl in file2_slices.iter() {
                lec.apply_edit(
                    Side::Rhs,
                    Edit::AppendSource {
                        source: sl.code.clone(),
                    },
                )?;
            }
            lec.apply_edit(
                Side::Rhs,
                Edit::AppendSource {
                    source: file2_composed.code.clone(),
                },
            )?;
            lec.set_top(Side::Rhs, &file2_composed.func_name);

            let ob = ProverObligation {
                selector_segment: "rhs_self".to_string(),
                description: Some("RHS self-equivalence".to_string()),
                payload: ObligationPayload::Lec(lec),
            };
            obligations.push(ob);
        }

        // Cross-slice per index
        for i in 1..=k {
            let sl1 = &file1_slices[i - 1];
            let sl2 = &file2_slices[i - 1];
            let mut lec = base_lec.clone();
            lec.apply_edit(
                Side::Lhs,
                Edit::AppendSource {
                    source: sl1.code.clone(),
                },
            )?;
            lec.apply_edit(
                Side::Rhs,
                Edit::AppendSource {
                    source: sl2.code.clone(),
                },
            )?;
            lec.set_top(Side::Lhs, &sl1.func_name);
            lec.set_top(Side::Rhs, &sl2.func_name);

            let ob = ProverObligation {
                selector_segment: format!("slice_{}", i),
                description: None,
                payload: ObligationPayload::Lec(lec),
            };
            obligations.push(ob);
        }

        // Skeleton with UF mapping of corresponding slices
        {
            let mut lec = base_lec.clone();
            lec.set_top(Side::Lhs, &file1_composed.func_name);
            lec.set_top(Side::Rhs, &file2_composed.func_name);
            for sl in file1_slices.iter() {
                lec.apply_edit(
                    Side::Lhs,
                    Edit::AppendSource {
                        source: sl.code.clone(),
                    },
                )?;
            }
            for sl in file2_slices.iter() {
                lec.apply_edit(
                    Side::Rhs,
                    Edit::AppendSource {
                        source: sl.code.clone(),
                    },
                )?;
            }
            lec.apply_edit(
                Side::Lhs,
                Edit::AppendSource {
                    source: file1_composed.code.clone(),
                },
            )?;
            lec.apply_edit(
                Side::Rhs,
                Edit::AppendSource {
                    source: file2_composed.code.clone(),
                },
            )?;
            for i in 0..k {
                let name1 = file1_slices[i].func_name.clone();
                let name2 = file2_slices[i].func_name.clone();
                let u = format!("__uf_{}", i);
                lec.set_uf_alias(&name1, &name2, &u);
            }

            let ob = ProverObligation {
                selector_segment: "skeleton".to_string(),
                description: None,
                payload: ObligationPayload::Lec(lec),
            };
            obligations.push(ob);
        }

        Ok(obligations)
    }
}

// is_valid_ident moved to `tactics::utils`

fn validate_and_collect(s1: &Vec<NamedSlice>, s2: &Vec<NamedSlice>) -> Result<usize, String> {
    let k1 = s1.len();
    let k2 = s2.len();
    if k1 == 0 || k2 == 0 {
        return Err("At least one slice on each side is required".to_string());
    }
    if k1 != k2 {
        return Err(format!("Slice count mismatch: lhs has {k1}, rhs has {k2}"));
    }
    for (i, sl) in s1.iter().enumerate() {
        let idx = i + 1;
        if !is_valid_ident(&sl.func_name) {
            return Err(format!("lhs.slice_{} invalid function name", idx));
        }
    }
    for (i, sl) in s2.iter().enumerate() {
        let idx = i + 1;
        if !is_valid_ident(&sl.func_name) {
            return Err(format!("rhs.slice_{} invalid function name", idx));
        }
    }
    Ok(k1)
}

/// Builds sub-obligations for a cosliced proof from a base obligation.
///
/// Behavior:
/// - LHS self: both sides start from LHS base; RHS side appends all file1
///   slices and composed. top1=base.top1, top2=composed1_name
/// - RHS self: both sides start from RHS base; RHS side appends all file2
///   slices and composed. top1=base.top2, top2=composed2_name
/// - Cross-slice i: default sides; append slice_i code on each side; top1/2 are
///   the slice function names
/// - Skeleton: default sides; append all functions on each side; top1/2 are
///   composed1_name/composed2_name; uf maps unify corresponding slice functions
// build_cosliced_obligations inlined into CoslicedTactic::apply

#[cfg(test)]
mod tests {
    use super::{CoslicedTactic, NamedSlice};
    use crate::dslx_tactics::obligations::{
        FileWithHistory, LecObligation, LecSide, ObligationPayload, ProverObligation, SourceFile,
    };
    use crate::dslx_tactics::tactics::IsTactic;

    fn base_obligation_with(sel: Option<&str>) -> ProverObligation {
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
            selector_segment: sel.unwrap_or("root").to_string(),
            description: None,
            payload: ObligationPayload::Lec(LecObligation { lhs, rhs }),
        }
    }

    fn slices_pair_2() -> (Vec<NamedSlice>, Vec<NamedSlice>) {
        let s1 = vec![
            NamedSlice {
                func_name: "f1_s1".to_string(),
                code: SourceFile::Text("pub fn f1_s1(x: u8) -> u8 { x + u8:1 }".to_string()),
            },
            NamedSlice {
                func_name: "f1_s2".to_string(),
                code: SourceFile::Text("pub fn f1_s2(x: u8) -> u8 { x + u8:2 }".to_string()),
            },
        ];
        let s2 = vec![
            NamedSlice {
                func_name: "f2_s1".to_string(),
                code: SourceFile::Text("pub fn f2_s1(x: u8) -> u8 { x + u8:1 }".to_string()),
            },
            NamedSlice {
                func_name: "f2_s2".to_string(),
                code: SourceFile::Text("pub fn f2_s2(x: u8) -> u8 { x + u8:2 }".to_string()),
            },
        ];
        (s1, s2)
    }

    #[test]
    fn errors_on_zero_slices() {
        let base = base_obligation_with(Some("root"));
        let s1: Vec<NamedSlice> = vec![];
        let s2: Vec<NamedSlice> = vec![];
        let c1 = NamedSlice {
            func_name: "ac".to_string(),
            code: SourceFile::Text("pub fn ac(x: u8) -> u8 { x }".to_string()),
        };
        let c2 = NamedSlice {
            func_name: "bc".to_string(),
            code: SourceFile::Text("pub fn bc(x: u8) -> u8 { x }".to_string()),
        };
        let t = CoslicedTactic {
            lhs_slices: s1,
            rhs_slices: s2,
            lhs_composed: c1,
            rhs_composed: c2,
        };
        assert!(t.apply(&base).is_err());
    }

    #[test]
    fn builds_expected_obligations_with_composed() {
        let base = base_obligation_with(Some("cosliced"));
        let (s1, s2) = slices_pair_2();
        let c1 = NamedSlice {
            func_name: "f1_comp".to_string(),
            code: SourceFile::Text("pub fn f1_comp(x: u8) -> u8 { x }".to_string()),
        };
        let c2 = NamedSlice {
            func_name: "f2_comp".to_string(),
            code: SourceFile::Text("pub fn f2_comp(x: u8) -> u8 { x }".to_string()),
        };
        let t = CoslicedTactic {
            lhs_slices: s1,
            rhs_slices: s2,
            lhs_composed: c1,
            rhs_composed: c2,
        };
        let obs = t.apply(&base).expect("ok");

        // 2 self + 2 cross + 1 skeleton = 5
        assert_eq!(obs.len(), 5);

        // LHS self
        let lhs_self = obs
            .iter()
            .find(|o| o.selector_segment == "lhs_self")
            .unwrap();
        let lhs_payload = match &lhs_self.payload {
            ObligationPayload::Lec(lec) => lec,
            _ => panic!("expected Lec payload"),
        };
        assert_eq!(lhs_payload.lhs.top_func, "f1");
        let rhs_appends = lhs_payload.rhs.file.edits.len();
        assert_eq!(rhs_appends, 3); // two slices + composed on RHS
        assert_eq!(
            lhs_self.description.as_ref().map(|s| s.as_str()),
            Some("LHS self-equivalence"),
        );

        // RHS self
        let rhs_self = obs
            .iter()
            .find(|o| o.selector_segment == "rhs_self")
            .unwrap();
        let rhs_payload = match &rhs_self.payload {
            ObligationPayload::Lec(lec) => lec,
            _ => panic!("expected Lec payload"),
        };
        assert_eq!(rhs_payload.lhs.top_func, "f2");
        let rhs_appends = rhs_payload.rhs.file.edits.len();
        assert_eq!(rhs_appends, 3); // two slices + composed on RHS
        assert_eq!(
            rhs_self.description.as_ref().map(|s| s.as_str()),
            Some("RHS self-equivalence"),
        );

        // Cross-slices: top names match slice functions and one append per side
        for i in 1..=2 {
            let cs = obs
                .iter()
                .find(|o| o.selector_segment == format!("slice_{}", i))
                .unwrap();
            let expected_lhs = format!("f1_s{}", i);
            let expected_rhs = format!("f2_s{}", i);
            let cs_payload = match &cs.payload {
                ObligationPayload::Lec(lec) => lec,
                _ => panic!("expected Lec payload"),
            };
            assert_eq!(cs_payload.lhs.top_func, expected_lhs);
            assert_eq!(cs_payload.rhs.top_func, expected_rhs);
            let lhs_appends = cs_payload.lhs.file.edits.len();
            let rhs_appends = cs_payload.rhs.file.edits.len();
            assert_eq!(lhs_appends, 1);
            assert_eq!(rhs_appends, 1);
        }

        // Skeleton (non-slice/non-self entry)
        let skel = obs
            .iter()
            .find(|o| {
                o.selector_segment != "lhs_self"
                    && o.selector_segment != "rhs_self"
                    && !o.selector_segment.starts_with("slice_")
            })
            .unwrap();
        let skel_payload = match &skel.payload {
            ObligationPayload::Lec(lec) => lec,
            _ => panic!("expected Lec payload"),
        };
        assert_eq!(skel_payload.lhs.top_func, "f1_comp");
        assert_eq!(skel_payload.rhs.top_func, "f2_comp");
        // LHS gets all file1 slices + composed; RHS gets all file2 slices + composed
        let lhs_appends = skel_payload.lhs.file.edits.len();
        let rhs_appends = skel_payload.rhs.file.edits.len();
        assert_eq!(lhs_appends, 3);
        assert_eq!(rhs_appends, 3);
        // UF mapping across slice pairs (indices start at 0)
        assert_eq!(
            skel_payload.lhs.uf_map.get("f1_s1").map(|s| s.as_str()),
            Some("__uf_0"),
        );
        assert_eq!(
            skel_payload.lhs.uf_map.get("f1_s2").map(|s| s.as_str()),
            Some("__uf_1"),
        );
        assert_eq!(
            skel_payload.rhs.uf_map.get("f2_s1").map(|s| s.as_str()),
            Some("__uf_0"),
        );
        assert_eq!(
            skel_payload.rhs.uf_map.get("f2_s2").map(|s| s.as_str()),
            Some("__uf_1"),
        );
    }

    #[test]
    fn builder_sets_segment_selectors() {
        let base = base_obligation_with(Some("root"));
        let (s1, s2) = slices_pair_2();
        let c1 = NamedSlice {
            func_name: "f1c".to_string(),
            code: SourceFile::Text("pub fn f1c(x: u8) -> u8 { x }".to_string()),
        };
        let c2 = NamedSlice {
            func_name: "f2c".to_string(),
            code: SourceFile::Text("pub fn f2c(x: u8) -> u8 { x }".to_string()),
        };
        let t = CoslicedTactic {
            lhs_slices: s1,
            rhs_slices: s2,
            lhs_composed: c1,
            rhs_composed: c2,
        };
        let obs = t.apply(&base).expect("ok");
        // Builder should set segment-level selectors
        let sels: Vec<String> = obs.iter().map(|o| o.selector_segment.clone()).collect();
        assert!(sels.contains(&"lhs_self".to_string()));
        assert!(sels.contains(&"rhs_self".to_string()));
        assert!(sels.contains(&"slice_1".to_string()));
        assert!(sels.contains(&"slice_2".to_string()));
        assert!(sels.contains(&"skeleton".to_string()) || sels.contains(&"root".to_string()));
    }
}
