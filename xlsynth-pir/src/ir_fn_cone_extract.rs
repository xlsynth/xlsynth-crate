// SPDX-License-Identifier: Apache-2.0

//! Backward cone extraction for PIR functions.
//!
//! This module extracts the (fanin) cone feeding a selected "sink" node, down
//! to the function's primary inputs (`get_param` nodes).
//!
//! The resulting package contains a single function named `cone` whose
//! parameters are the subset of the original function's parameters that affect
//! the sink.

use crate::ir;
use crate::ir_utils::{operands, remap_payload_with};
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkSelector {
    TextId(usize),
    Name(String),
}

pub fn parse_sink_selector(s: &str) -> Result<SinkSelector, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("sink selector must be non-empty".to_string());
    }

    // Prefer the most common case: numeric `id=` / `text_id`.
    if let Ok(id) = s.parse::<usize>() {
        return Ok(SinkSelector::TextId(id));
    }

    // Support `op.id` forms like `and.123` by parsing the numeric suffix.
    if let Some((_prefix, suffix)) = s.rsplit_once('.') {
        if let Ok(id) = suffix.parse::<usize>() {
            return Ok(SinkSelector::TextId(id));
        }
    }

    Ok(SinkSelector::Name(s.to_string()))
}

#[derive(Debug, Clone)]
pub struct ExtractedFnCone {
    pub package: ir::Package,
    pub sink_text_id: usize,
    pub sink_node_index: usize,
    pub used_params: Vec<ir::Param>,
}

fn resolve_sink_node_index(f: &ir::Fn, sink: SinkSelector) -> Result<usize, String> {
    match sink {
        SinkSelector::TextId(id) => {
            if id == 0 {
                return Err("sink text_id must be > 0".to_string());
            }
            let matches: Vec<usize> = f
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(idx, n)| (n.text_id == id).then_some(idx))
                .collect();
            match matches.as_slice() {
                [] => Err(format!(
                    "no node with text_id={} found in function '{}'",
                    id, f.name
                )),
                [idx] => Ok(*idx),
                _ => Err(format!(
                    "multiple nodes with text_id={} found in function '{}' (expected unique ids)",
                    id, f.name
                )),
            }
        }
        SinkSelector::Name(name) => {
            let matches: Vec<usize> = f
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(idx, n)| (n.name.as_deref() == Some(name.as_str())).then_some(idx))
                .collect();
            match matches.as_slice() {
                [] => Err(format!(
                    "no node named '{}' found in function '{}'",
                    name, f.name
                )),
                [idx] => Ok(*idx),
                _ => Err(format!(
                    "multiple nodes named '{}' found in function '{}' (ambiguous sink)",
                    name, f.name
                )),
            }
        }
    }
}

fn compute_get_param_index_map(f: &ir::Fn) -> Result<HashMap<ir::ParamId, usize>, String> {
    let mut out: HashMap<ir::ParamId, usize> = HashMap::new();
    for (idx, node) in f.nodes.iter().enumerate() {
        if let ir::NodePayload::GetParam(pid) = node.payload {
            if out.insert(pid, idx).is_some() {
                return Err(format!(
                    "duplicate get_param node for ParamId={} in function '{}'",
                    pid.get_wrapped_id(),
                    f.name
                ));
            }
        }
    }
    Ok(out)
}

fn collect_cone_nodes_and_params(
    f: &ir::Fn,
    sink_idx: usize,
) -> Result<(BTreeSet<usize>, HashSet<ir::ParamId>), String> {
    if sink_idx >= f.nodes.len() {
        return Err(format!(
            "sink index {} out of bounds (len={}) in function '{}'",
            sink_idx,
            f.nodes.len(),
            f.name
        ));
    }

    let mut visited: HashSet<usize> = HashSet::new();
    let mut included_internal: BTreeSet<usize> = BTreeSet::new();
    let mut used_param_ids: HashSet<ir::ParamId> = HashSet::new();

    let mut stack: Vec<usize> = vec![sink_idx];
    while let Some(idx) = stack.pop() {
        if !visited.insert(idx) {
            continue;
        }
        let node = &f.nodes[idx];
        match node.payload {
            ir::NodePayload::Nil => continue,
            ir::NodePayload::GetParam(pid) => {
                used_param_ids.insert(pid);
                continue;
            }
            _ => {
                included_internal.insert(idx);
            }
        }
        for dep in operands(&node.payload) {
            stack.push(dep.index);
        }
    }

    Ok((included_internal, used_param_ids))
}

pub fn extract_fn_cone_to_params(
    f: &ir::Fn,
    pkg_file_table: Option<&ir::FileTable>,
    sink: SinkSelector,
    emit_pos_data: bool,
) -> Result<ExtractedFnCone, String> {
    let sink_idx = resolve_sink_node_index(f, sink)?;
    let sink_text_id = f.nodes[sink_idx].text_id;

    let get_param_idx_by_id = compute_get_param_index_map(f)?;

    let (included_internal, used_param_ids) = collect_cone_nodes_and_params(f, sink_idx)?;

    let used_params: Vec<ir::Param> = f
        .params
        .iter()
        .cloned()
        .filter(|p| used_param_ids.contains(&p.id))
        .collect();

    // Build a new function named `cone` with only the used params.
    let mut nodes: Vec<ir::Node> = Vec::new();
    nodes.push(ir::Node {
        text_id: 0,
        name: None,
        ty: ir::Type::nil(),
        payload: ir::NodePayload::Nil,
        pos: None,
    });

    // Map old node indices to new node indices for all nodes that may be
    // referenced (used params + included internal nodes).
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();

    for p in used_params.iter() {
        let old_param_idx = *get_param_idx_by_id.get(&p.id).ok_or_else(|| {
            format!(
                "function '{}' signature param '{}' (ParamId={}) is missing a get_param node",
                f.name,
                p.name,
                p.id.get_wrapped_id()
            )
        })?;

        let new_idx = nodes.len();
        nodes.push(ir::Node {
            text_id: p.id.get_wrapped_id(),
            name: Some(p.name.clone()),
            ty: p.ty.clone(),
            payload: ir::NodePayload::GetParam(p.id),
            pos: None,
        });
        old_to_new.insert(old_param_idx, new_idx);
    }

    let internal_indices: Vec<usize> = included_internal.iter().copied().collect();
    let internal_base = nodes.len();
    for (i, old_idx) in internal_indices.iter().copied().enumerate() {
        old_to_new.insert(old_idx, internal_base + i);
    }

    for old_idx in internal_indices.iter().copied() {
        let old_node = &f.nodes[old_idx];
        let new_payload = remap_payload_with(&old_node.payload, |(_slot, r)| {
            let new_index = *old_to_new.get(&r.index).unwrap_or_else(|| {
                panic!(
                    "missing old_to_new mapping for operand index {} (sink_idx={} function='{}')",
                    r.index, sink_idx, f.name
                )
            });
            ir::NodeRef { index: new_index }
        });
        nodes.push(ir::Node {
            text_id: old_node.text_id,
            name: old_node.name.clone(),
            ty: old_node.ty.clone(),
            payload: new_payload,
            pos: if emit_pos_data {
                old_node.pos.clone()
            } else {
                None
            },
        });
    }

    let new_sink_idx = *old_to_new.get(&sink_idx).ok_or_else(|| {
        format!(
            "internal error: sink index {} (text_id={}) was not mapped",
            sink_idx, sink_text_id
        )
    })?;

    let func = ir::Fn {
        name: "cone".to_string(),
        params: used_params.clone(),
        ret_ty: f.nodes[sink_idx].ty.clone(),
        nodes,
        ret_node_ref: Some(ir::NodeRef {
            index: new_sink_idx,
        }),
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
    };

    let package = ir::Package {
        name: "fn_cone".to_string(),
        file_table: if emit_pos_data {
            pkg_file_table.cloned().unwrap_or_else(ir::FileTable::new)
        } else {
            ir::FileTable::new()
        },
        members: vec![ir::PackageMember::Function(func)],
        top: Some(("cone".to_string(), ir::MemberType::Function)),
    };

    Ok(ExtractedFnCone {
        package,
        sink_text_id,
        sink_node_index: sink_idx,
        used_params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser;

    #[test]
    fn parse_sink_selector_accepts_numeric_id() {
        assert_eq!(
            parse_sink_selector("123").unwrap(),
            SinkSelector::TextId(123)
        );
    }

    #[test]
    fn parse_sink_selector_accepts_op_dot_id() {
        assert_eq!(
            parse_sink_selector("and.42").unwrap(),
            SinkSelector::TextId(42)
        );
    }

    #[test]
    fn parse_sink_selector_accepts_name() {
        assert_eq!(
            parse_sink_selector("some_node").unwrap(),
            SinkSelector::Name("some_node".to_string())
        );
    }

    #[test]
    fn extract_by_text_id_uses_only_needed_params() {
        let ir_text = r#"fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  x: bits[8] = add(a, b, id=4)
  y: bits[8] = xor(x, c, id=5)
  ret z: bits[8] = not(y, id=6)
}"#;
        let mut p = ir_parser::Parser::new(ir_text);
        let f = p.parse_fn().unwrap();

        let extracted = extract_fn_cone_to_params(
            &f,
            None,
            SinkSelector::TextId(4),
            /* emit_pos_data= */ false,
        )
        .unwrap();

        let member = match extracted.package.members.as_slice() {
            [m] => m,
            _ => panic!("expected exactly one member"),
        };
        let cone_fn = match member {
            ir::PackageMember::Function(f) => f,
            ir::PackageMember::Block { .. } => panic!("expected function member"),
        };

        assert_eq!(
            cone_fn.params.iter().map(|p| &p.name).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(cone_fn.ret_ty, ir::Type::Bits(8));
        assert_eq!(cone_fn.get_node(cone_fn.ret_node_ref.unwrap()).text_id, 4);
    }

    #[test]
    fn extract_by_name_works() {
        let ir_text = r#"fn f(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  x: bits[8] = add(a, b, id=4)
  y: bits[8] = xor(x, c, id=5)
  ret z: bits[8] = not(y, id=6)
}"#;
        let mut p = ir_parser::Parser::new(ir_text);
        let f = p.parse_fn().unwrap();

        let extracted = extract_fn_cone_to_params(
            &f,
            None,
            SinkSelector::Name("y".to_string()),
            /* emit_pos_data= */ false,
        )
        .unwrap();

        let member = match extracted.package.members.as_slice() {
            [m] => m,
            _ => panic!("expected exactly one member"),
        };
        let cone_fn = match member {
            ir::PackageMember::Function(f) => f,
            ir::PackageMember::Block { .. } => panic!("expected function member"),
        };

        assert_eq!(
            cone_fn.params.iter().map(|p| &p.name).collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert_eq!(cone_fn.get_node(cone_fn.ret_node_ref.unwrap()).text_id, 5);
    }

    #[test]
    fn sink_can_be_param() {
        let ir_text = r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret r: bits[8] = param(name=a, id=1)
}"#;
        let mut p = ir_parser::Parser::new(ir_text);
        let f = p.parse_fn().unwrap();

        let extracted = extract_fn_cone_to_params(
            &f,
            None,
            SinkSelector::Name("a".to_string()),
            /* emit_pos_data= */ false,
        )
        .unwrap();

        let member = match extracted.package.members.as_slice() {
            [m] => m,
            _ => panic!("expected exactly one member"),
        };
        let cone_fn = match member {
            ir::PackageMember::Function(f) => f,
            ir::PackageMember::Block { .. } => panic!("expected function member"),
        };

        assert_eq!(
            cone_fn.params.iter().map(|p| &p.name).collect::<Vec<_>>(),
            ["a"]
        );
        let ret = cone_fn.get_node(cone_fn.ret_node_ref.unwrap());
        assert!(matches!(ret.payload, ir::NodePayload::GetParam(_)));
    }
}
