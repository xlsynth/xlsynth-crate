// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use rayon::prelude::*;

use crate::aig::get_summary_stats::get_gate_depth;
use crate::ir2gate_utils::AdderMapping;
use crate::ir2gates;
use crate::use_count::get_id_to_use_count;
use xlsynth_pir::ir;
use xlsynth_pir::ir_utils;

pub struct DiverseSampleSelectionEntry {
    pub ir_file_path: PathBuf,
    pub g8r_nodes: usize,
    pub g8r_levels: usize,
    pub new_hashes: usize,
    pub new_hash_details: Option<Vec<NewHashDetail>>,
}

#[derive(Clone, Debug)]
pub struct NewHashDetail {
    pub hash: [u8; 32],
    pub node_index: usize,
    pub text_id: usize,
    pub signature: String,
}

pub struct DiverseSamplesOptions {
    pub signature_depth: usize,
    pub log_skipped: bool,
    pub explain_new_hashes: bool,
}

impl Default for DiverseSamplesOptions {
    fn default() -> Self {
        Self {
            signature_depth: 2,
            log_skipped: false,
            explain_new_hashes: false,
        }
    }
}

#[derive(Clone, Debug)]
struct HashCause {
    node_index: usize,
    text_id: usize,
    signature: String,
}

fn format_depth_limited_nested_signature(
    f: &ir::Fn,
    node_ref: ir::NodeRef,
    depth: usize,
) -> String {
    let local = xlsynth_pir::node_hashing::node_structural_signature_string(f, node_ref);
    if depth == 0 {
        return local;
    }
    let node = f.get_node(node_ref);
    let child_refs = ir_utils::operands(&node.payload);
    if child_refs.is_empty() {
        return local;
    }
    let children = child_refs
        .into_iter()
        .map(|r| format_depth_limited_nested_signature(f, r, depth - 1))
        .collect::<Vec<String>>()
        .join(", ");
    format!("{local} [{children}]")
}

fn should_exclude_signature_hash_root(f: &ir::Fn, node_ref: ir::NodeRef) -> bool {
    let node = f.get_node(node_ref);
    let payload = &node.payload;
    match payload {
        ir::NodePayload::Nil => true,
        ir::NodePayload::GetParam(_) => true,
        ir::NodePayload::Literal(_) => true,
        ir::NodePayload::Tuple(_) => true,
        ir::NodePayload::Array(_) => true,
        ir::NodePayload::TupleIndex { .. } => true,
        ir::NodePayload::Nary(ir::NaryOp::Concat, _) => true,
        ir::NodePayload::Nary(
            ir::NaryOp::And | ir::NaryOp::Or | ir::NaryOp::Xor | ir::NaryOp::Nand | ir::NaryOp::Nor,
            operands,
        ) => {
            // Exclude “simple gates” when they’re single-bit ops; keep wider
            // bitwise operations as roots (they tend to be more meaningful).
            if node.ty != ir::Type::Bits(1) {
                return false;
            }
            operands
                .iter()
                .all(|r| *f.get_node_ty(*r) == ir::Type::Bits(1))
        }
        ir::NodePayload::Unop(ir::Unop::Not, _) => true,
        ir::NodePayload::Unop(ir::Unop::Identity, _) => true,
        ir::NodePayload::BitSlice { .. } => true,
        ir::NodePayload::ArraySlice { .. } => true,
        ir::NodePayload::ArrayIndex { .. } => true,
        ir::NodePayload::ArrayUpdate { .. } => true,
        ir::NodePayload::BitSliceUpdate { .. } => true,
        ir::NodePayload::ZeroExt { .. } => true,
        ir::NodePayload::SignExt { .. } => true,
        ir::NodePayload::AfterAll(_) => true,
        ir::NodePayload::Assert { .. } => true,
        ir::NodePayload::Trace { .. } => true,
        ir::NodePayload::Cover { .. } => true,
        ir::NodePayload::Invoke { .. } => true,
        ir::NodePayload::CountedFor { .. } => true,
        ir::NodePayload::PrioritySel { .. } => true,
        ir::NodePayload::OneHotSel { .. } => true,
        ir::NodePayload::Sel { .. } => true,
        _ => false,
    }
}

struct SampleInfo {
    ir_file_path: PathBuf,
    ir_file_path_key: String,
    hashes: BTreeSet<[u8; 32]>,
    hash_causes: Option<BTreeMap<[u8; 32], HashCause>>,
    g8r_nodes: usize,
    g8r_levels: usize,
    cost: u64,
}

fn path_key(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn collect_ir_paths(corpus_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![corpus_dir.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .map_err(|e| format!("failed to read dir {}: {e}", dir.display()))?;
        for entry_result in entries {
            let entry = entry_result
                .map_err(|e| format!("failed to read dir entry in {}: {e}", dir.display()))?;
            let path = entry.path();
            let ty = entry
                .file_type()
                .map_err(|e| format!("failed to get file type for {}: {e}", path.display()))?;
            if ty.is_dir() {
                stack.push(path);
            } else if ty.is_file() {
                if matches!(path.extension(), Some(ext) if ext == "ir") {
                    out.push(path);
                }
            }
        }
    }

    out.sort_by(|a, b| path_key(a).cmp(&path_key(b)));
    Ok(out)
}

fn compute_g8r_stats(gate_fn: &crate::aig::gate::GateFn) -> (usize, usize) {
    let id_to_use_count = get_id_to_use_count(gate_fn);
    let live_nodes: Vec<crate::aig::gate::AigRef> = id_to_use_count.keys().cloned().collect();
    let depth_stats = get_gate_depth(gate_fn, &live_nodes);
    let g8r_nodes = live_nodes.len();
    let g8r_levels = depth_stats.deepest_path.len();
    (g8r_nodes, g8r_levels)
}

fn select_diverse_samples(mut samples: Vec<SampleInfo>) -> Vec<DiverseSampleSelectionEntry> {
    samples.sort_by(|a, b| {
        b.cost
            .cmp(&a.cost)
            .then_with(|| a.ir_file_path_key.cmp(&b.ir_file_path_key))
    });

    let mut seen: HashSet<[u8; 32]> = HashSet::new();
    let mut selected: Vec<DiverseSampleSelectionEntry> = Vec::new();
    for s in samples.into_iter() {
        let new_hashes = s.hashes.iter().filter(|h| !seen.contains(*h)).count();
        if new_hashes == 0 {
            continue;
        }
        let new_hash_details = s.hash_causes.as_ref().map(|causes| {
            let mut out: Vec<NewHashDetail> = Vec::new();
            for h in s.hashes.iter().copied() {
                if seen.contains(&h) {
                    continue;
                }
                let cause = causes
                    .get(&h)
                    .expect("hash_causes must contain an entry for every hash");
                out.push(NewHashDetail {
                    hash: h,
                    node_index: cause.node_index,
                    text_id: cause.text_id,
                    signature: cause.signature.clone(),
                });
            }
            out.sort_by(|a, b| {
                a.node_index
                    .cmp(&b.node_index)
                    .then_with(|| a.text_id.cmp(&b.text_id))
                    .then_with(|| a.hash.cmp(&b.hash))
            });
            out
        });
        for h in s.hashes.iter().copied() {
            seen.insert(h);
        }
        selected.push(DiverseSampleSelectionEntry {
            ir_file_path: s.ir_file_path,
            g8r_nodes: s.g8r_nodes,
            g8r_levels: s.g8r_levels,
            new_hashes,
            new_hash_details,
        });
    }
    selected
}

/// Walks `corpus_dir` to find all `.ir` files and selects a diverse subset.
///
/// Diversity criterion: Greedy selection over increasing cost (nodes * levels),
/// including a sample if it introduces any previously unseen depth-limited
/// forward structural hashes for nodes in the package top function.
pub fn select_ir_diverse_samples(
    corpus_dir: &Path,
) -> Result<Vec<DiverseSampleSelectionEntry>, String> {
    select_ir_diverse_samples_with_options(corpus_dir, &DiverseSamplesOptions::default())
}

pub fn select_ir_diverse_samples_with_signature_depth(
    corpus_dir: &Path,
    signature_depth: usize,
) -> Result<Vec<DiverseSampleSelectionEntry>, String> {
    select_ir_diverse_samples_with_options(
        corpus_dir,
        &DiverseSamplesOptions {
            signature_depth,
            log_skipped: false,
            explain_new_hashes: false,
        },
    )
}

pub fn select_ir_diverse_samples_with_options(
    corpus_dir: &Path,
    options: &DiverseSamplesOptions,
) -> Result<Vec<DiverseSampleSelectionEntry>, String> {
    let ir_paths = collect_ir_paths(corpus_dir)?;

    let samples: Vec<SampleInfo> = ir_paths
        .par_iter()
        .filter_map(|ir_path| {
            let ir_text = match std::fs::read_to_string(ir_path) {
                Ok(s) => s,
                Err(e) => {
                    if options.log_skipped {
                        log::error!("Skipping {}: failed to read: {e}", ir_path.display());
                    }
                    return None;
                }
            };

            let output = match ir2gates::ir2gates_from_ir_text(
                &ir_text,
                None,
                ir2gates::Ir2GatesOptions {
                    fold: true,
                    hash: true,
                    check_equivalence: false,
                    enable_rewrite_carry_out: false,
                    adder_mapping: AdderMapping::default(),
                    mul_adder_mapping: None,
                    aug_opt: Default::default(),
                },
            ) {
                Ok(o) => o,
                Err(e) => {
                    if options.log_skipped {
                        log::error!(
                            "Skipping {}: failed to lower via ir2gates: {e}",
                            ir_path.display()
                        );
                    }
                    return None;
                }
            };

            let f = output.pir_top_fn();
            let depth_hashes = xlsynth_pir::node_hashing::compute_depth_limited_forward_hashes(
                f,
                options.signature_depth,
            );
            let mut hashes: BTreeSet<[u8; 32]> = BTreeSet::new();
            for (node_index, h) in depth_hashes.iter().enumerate() {
                let node_ref = ir::NodeRef { index: node_index };
                if should_exclude_signature_hash_root(f, node_ref) {
                    continue;
                }
                hashes.insert(*h.as_bytes());
            }

            let hash_causes = if options.explain_new_hashes {
                let mut out: BTreeMap<[u8; 32], HashCause> = BTreeMap::new();
                for (node_index, node) in f.nodes.iter().enumerate() {
                    let node_ref = ir::NodeRef { index: node_index };
                    if should_exclude_signature_hash_root(f, node_ref) {
                        continue;
                    }
                    let h = *depth_hashes[node_index].as_bytes();
                    out.entry(h).or_insert_with(|| HashCause {
                        node_index,
                        text_id: node.text_id,
                        signature: format_depth_limited_nested_signature(
                            f,
                            node_ref,
                            options.signature_depth,
                        ),
                    });
                }
                Some(out)
            } else {
                None
            };

            let (g8r_nodes, g8r_levels) = compute_g8r_stats(&output.gatify_output.gate_fn);
            let cost = (g8r_nodes as u64).saturating_mul(g8r_levels as u64);
            Some(SampleInfo {
                ir_file_path_key: path_key(ir_path),
                ir_file_path: ir_path.to_path_buf(),
                hashes,
                hash_causes,
                g8r_nodes,
                g8r_levels,
                cost,
            })
        })
        .collect();

    Ok(select_diverse_samples(samples))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format_selected(entries: &[DiverseSampleSelectionEntry]) -> String {
        let mut out = String::new();
        for e in entries.iter() {
            out.push_str(&format!(
                "{} g8r-nodes={} g8r-levels={} new-hashes={}\n",
                e.ir_file_path.display(),
                e.g8r_nodes,
                e.g8r_levels,
                e.new_hashes
            ));
        }
        out
    }

    #[test]
    fn greedy_diversity_selection_is_deterministic_for_fixed_inputs() {
        let samples = vec![
            SampleInfo {
                ir_file_path: PathBuf::from("b.ir"),
                ir_file_path_key: "b.ir".to_string(),
                hashes: BTreeSet::from([[1u8; 32]]),
                hash_causes: None,
                g8r_nodes: 2,
                g8r_levels: 2,
                cost: 4,
            },
            SampleInfo {
                ir_file_path: PathBuf::from("a.ir"),
                ir_file_path_key: "a.ir".to_string(),
                hashes: BTreeSet::from([[2u8; 32]]),
                hash_causes: None,
                g8r_nodes: 1,
                g8r_levels: 3,
                cost: 3,
            },
            SampleInfo {
                ir_file_path: PathBuf::from("c.ir"),
                ir_file_path_key: "c.ir".to_string(),
                hashes: BTreeSet::from([[1u8; 32], [2u8; 32]]),
                hash_causes: None,
                g8r_nodes: 1,
                g8r_levels: 10,
                cost: 10,
            },
        ];
        let selected = select_diverse_samples(samples);

        assert_eq!(
            format_selected(&selected),
            "c.ir g8r-nodes=1 g8r-levels=10 new-hashes=2\n"
        );
    }

    #[test]
    fn collect_ir_paths_is_recursive_and_deterministic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path();
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::create_dir_all(root.join("z")).unwrap();

        std::fs::write(root.join("top.ir"), "package p\n").unwrap();
        std::fs::write(root.join("top.txt"), "nope\n").unwrap();
        std::fs::write(root.join("a").join("b").join("nested.ir"), "package p\n").unwrap();
        std::fs::write(root.join("z").join("zzz.ir"), "package p\n").unwrap();

        let paths = collect_ir_paths(root).unwrap();
        let joined = paths
            .iter()
            .map(|p| path_key(p))
            .collect::<Vec<String>>()
            .join("\n");

        assert_eq!(
            joined,
            format!(
                "{}\n{}\n{}",
                path_key(&root.join("a").join("b").join("nested.ir")),
                path_key(&root.join("top.ir")),
                path_key(&root.join("z").join("zzz.ir"))
            )
        );
    }

    #[test]
    fn signature_depth_changes_selection_for_some_inputs() {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path();
        let a_path = root.join("a.ir");
        let b_path = root.join("b.ir");

        let a_ir = r#"package p
top fn f(a: bits[8], b: bits[8], c: bits[8]) -> bits[8] {
  add.4: bits[8] = add(a, b)
  ret add.5: bits[8] = add(add.4, c)
}
"#;
        let b_ir = r#"package p
top fn f(a: bits[8], b: bits[8], c: bits[8]) -> bits[8] {
  add.4: bits[8] = add(a, c)
  ret add.5: bits[8] = add(add.4, b)
}
"#;

        std::fs::write(&a_path, a_ir).unwrap();
        std::fs::write(&b_path, b_ir).unwrap();

        let selected_d0 =
            select_ir_diverse_samples_with_signature_depth(root, 0).expect("selection should work");
        let selected_d2 =
            select_ir_diverse_samples_with_signature_depth(root, 2).expect("selection should work");

        let mut d0_paths = selected_d0
            .iter()
            .map(|e| path_key(&e.ir_file_path))
            .collect::<Vec<String>>();
        d0_paths.sort();
        let d0_joined = d0_paths.join("\n");

        let mut d2_paths = selected_d2
            .iter()
            .map(|e| path_key(&e.ir_file_path))
            .collect::<Vec<String>>();
        d2_paths.sort();
        let d2_joined = d2_paths.join("\n");

        assert_eq!(d0_joined, path_key(&a_path));
        assert_eq!(
            d2_joined,
            format!("{}\n{}", path_key(&a_path), path_key(&b_path))
        );
    }
}
