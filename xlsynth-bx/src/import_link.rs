// SPDX-License-Identifier: Apache-2.0

//! Imports the pure IR functions implementing concrete DSLX calls.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_pir::ir::{NodePayload, Package, PackageMember};
use xlsynth_pir::ir_rebase_ids::{package_max_emitted_node_id, rebase_fn_ids_in_place};
use xlsynth_pir::ir_utils::external_function_references;
use xlsynth_pir::ir_verify::verify_package;

use crate::lower::check_function_purity;

/// Appends a checked import snapshot and returns the destination indices of its
/// roots. All allocation and validation take place before changing the
/// destination.
pub(crate) fn link_imported_functions(
    destination: &mut Package,
    mut imported: Package,
    roots: &[String],
    reserved_blocks: &[String],
) -> Result<Vec<usize>, String> {
    if roots.is_empty() {
        return Ok(Vec::new());
    }
    let symbols = allocate_symbols(destination, &imported, reserved_blocks);
    let files = allocate_files(destination, &imported)?;
    let mut node_base = package_max_emitted_node_id(destination);
    // DSLX assigns IDs starting at one. Also allow an explicit zero on a
    // non-sentinel IR node without reusing the destination's highest ID.
    if !destination.members.is_empty()
        && imported.members.iter().any(|member| {
            member
                .graph()
                .nodes
                .iter()
                .any(|node| node.text_id == 0 && !matches!(node.payload, NodePayload::Nil))
        })
    {
        node_base = node_base
            .checked_add(1)
            .ok_or("imported node IDs exceed the available ID space")?;
    }
    for member in &mut imported.members {
        let PackageMember::Function(function) = member else {
            unreachable!("only reachable functions were retained")
        };
        function.name = symbols[&function.name].name.clone();
        rebase_fn_ids_in_place(function, node_base).map_err(|error| error.to_string())?;
        for node in &mut function.nodes {
            match &mut node.payload {
                NodePayload::Invoke { to_apply, .. } | NodePayload::Map { to_apply, .. } => {
                    *to_apply = symbols[to_apply].name.clone();
                }
                NodePayload::CountedFor { body, .. } => *body = symbols[body].name.clone(),
                _ => {
                    // Other function operations contain no callee symbol.
                }
            }
            for position in node.pos.iter_mut().flatten() {
                position.fileno = files.remap[&position.fileno];
            }
        }
    }
    verify_package(&imported).map_err(|error| format!("invalid imported DSLX IR: {error}"))?;
    destination.file_table.id_to_path.extend(files.additions);
    destination.members.extend(imported.members);
    Ok(roots.iter().map(|root| symbols[root].index).collect())
}

/// Selects and checks the IR bodies reachable from the requested
/// specializations. XLS dumps callees before callers, so a reverse scan
/// discovers each dependency before reaching its definition.
pub(crate) fn retain_reachable_functions(
    package: &mut Package,
    roots: &[String],
) -> Result<(), String> {
    let mut needed: BTreeSet<_> = roots.iter().cloned().collect();
    let mut retained = Vec::new();
    for member in package.members.drain(..).rev() {
        if let PackageMember::Function(function) = &member
            && needed.remove(&function.name)
        {
            check_function_purity(function)?;
            needed.extend(external_function_references(function));
            retained.push(member);
        }
    }
    if let Some(name) = needed.first() {
        return Err(format!(
            "missing imported function '{name}' in callee-before-caller order"
        ));
    }
    retained.reverse();
    package.members = retained;
    package.top = None;
    Ok(())
}

/// The destination identity of an imported function.
struct Symbol {
    name: String,
    index: usize,
}

/// Keeps original function names wherever possible, including when another
/// imported function would otherwise take such a name as its fresh suffix.
fn allocate_symbols(
    destination: &Package,
    imported: &Package,
    reserved_blocks: &[String],
) -> BTreeMap<String, Symbol> {
    let occupied: BTreeSet<_> = destination
        .members
        .iter()
        .map(|member| member.graph().name.as_str())
        .chain(reserved_blocks.iter().map(String::as_str))
        .collect();
    let mut used: BTreeSet<String> = occupied.iter().map(|name| (*name).to_owned()).collect();
    used.extend(
        imported
            .members
            .iter()
            .map(|member| member.graph().name.clone()),
    );
    let mut symbols = BTreeMap::new();
    for (offset, member) in imported.members.iter().enumerate() {
        let original = &member.graph().name;
        let mut name = original.clone();
        if occupied.contains(original.as_str()) {
            let mut suffix = 1usize;
            loop {
                name = format!("{original}__bx_{suffix}");
                if used.insert(name.clone()) {
                    break;
                }
                suffix += 1;
            }
        }
        symbols.insert(
            original.clone(),
            Symbol {
                name,
                index: destination.members.len() + offset,
            },
        );
    }
    symbols
}

/// Imported position IDs and the new file declarations they require.
struct FileImports {
    remap: BTreeMap<usize, usize>,
    additions: Vec<(usize, String)>,
}

/// Collects file numbers from positions, including ones lacking a file
/// declaration.
fn referenced_file_ids(package: &Package) -> impl Iterator<Item = usize> + '_ {
    package
        .members
        .iter()
        .flat_map(|member| &member.graph().nodes)
        .flat_map(|node| node.pos.iter().flatten())
        .map(|pos| pos.fileno)
}

/// Interns only reachable file paths and keeps undeclared file numbers
/// unresolved, including those already used in caller source positions.
fn allocate_files(destination: &Package, imported: &Package) -> Result<FileImports, String> {
    let mut paths = BTreeMap::<&str, usize>::new();
    for (id, path) in &destination.file_table.id_to_path {
        paths
            .entry(path)
            .and_modify(|old| *old = (*old).min(*id))
            .or_insert(*id);
    }
    let occupied: BTreeSet<_> = destination
        .file_table
        .id_to_path
        .keys()
        .copied()
        .chain(referenced_file_ids(destination))
        .collect();
    let referenced: BTreeSet<_> = referenced_file_ids(imported).collect();
    let mut fresh_ids =
        (0..=usize::MAX).filter(|id| !occupied.contains(id) && !referenced.contains(id));
    let mut remap = BTreeMap::new();
    let mut additions = Vec::new();
    for &id in &referenced {
        let path = imported.file_table.id_to_path.get(&id);
        if let Some(existing) = path.and_then(|path| paths.get(path.as_str())) {
            remap.insert(id, *existing);
            continue;
        }
        let assigned = if occupied.contains(&id) {
            fresh_ids
                .next()
                .ok_or("imported file numbers exceed the available ID space")?
        } else {
            id
        };
        if let Some(path) = path {
            paths.insert(path, assigned);
            additions.push((assigned, path.clone()));
        }
        remap.insert(id, assigned);
    }
    Ok(FileImports { remap, additions })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn reachable_functions_keep_their_calls_and_source_files_across_collisions() {
        let mut destination = Parser::new(
            r#"package circuit
file_number 3 "caller.x"
file_number 8 "shared.x"
top fn piece(x: bits[8] id=21) -> bits[8] {
  ret old: bits[8] = not(x, id=22, pos=[(3,0,0), (1,0,0)])
}
block body() {}
"#,
        )
        .parse_package()
        .unwrap();
        let mut imported = Parser::new(
            r#"package wrappers
file_number 1 "other.x"
file_number 2 "shared.x"
file_number 3 "helper.x"
file_number 4 "unused.x"
fn piece(x: bits[8] id=1) -> bits[8] {
  ret n: bits[8] = neg(x, id=2, pos=[(3,5,6), (2,1,2), (1,2,3), (8,4,5)])
}
fn body(i: bits[32] id=3, x: bits[8] id=4) -> bits[8] {
  ret call: bits[8] = invoke(x, to_apply=piece, id=5)
}
fn main(xs: bits[8][2] id=6) -> (bits[8][2], bits[8]) {
  mapped: bits[8][2] = map(xs, to_apply=piece, id=7)
  init: bits[8] = literal(value=0, id=8)
  looped: bits[8] = counted_for(init, trip_count=2, stride=1, body=body, id=9)
  ret pair: (bits[8][2], bits[8]) = tuple(mapped, looped, id=10)
}
fn piece__bx_1(x: bits[8] id=11) -> bits[8] {
  ret answer: bits[8] = not(x, id=12)
}
fn unused() -> bits[8] {
  ret zero: bits[8] = literal(value=0, id=13, pos=[(4,0,0)])
}
"#,
        )
        .parse_package()
        .unwrap();
        let original = destination.get_fn("piece").unwrap().to_string();
        let top = destination.top.clone();
        let roots = ["main".into(), "piece__bx_1".into(), "main".into()];
        retain_reachable_functions(&mut imported, &roots).unwrap();
        let indices =
            link_imported_functions(&mut destination, imported, &roots, &["main".into()]).unwrap();
        assert_eq!(indices[0], indices[2]);
        assert_eq!(destination.members[indices[0]].graph().name, "main__bx_1");
        assert_eq!(destination.members[indices[1]].graph().name, "piece__bx_1");
        assert_eq!(destination.get_fn("piece").unwrap().to_string(), original);
        assert_eq!(destination.top, top);
        assert_eq!(destination.name, "circuit");
        assert_eq!(destination.members.len(), 6);
        let main = destination.get_fn("main__bx_1").unwrap();
        assert_eq!(
            external_function_references(main),
            BTreeSet::from(["piece__bx_2".into(), "body__bx_1".into()])
        );
        let body = destination.get_fn("body__bx_1").unwrap();
        assert_eq!(
            external_function_references(body),
            BTreeSet::from(["piece__bx_2".into()])
        );
        let piece = destination.get_fn("piece__bx_2").unwrap();
        let positions = piece
            .get_node(piece.ret_node_ref.unwrap())
            .pos
            .as_ref()
            .unwrap();
        let files = &destination.file_table;
        assert_eq!(
            positions[0].to_human_string(files).as_deref(),
            Some("helper.x:6:7")
        );
        assert_eq!(positions[1].fileno, 8);
        assert_eq!(
            positions[2].to_human_string(files).as_deref(),
            Some("other.x:3:4")
        );
        assert_eq!(positions[3].to_human_string(files), None);
        assert!(!files.id_to_path.contains_key(&1));
        assert_eq!(files.id_to_path.len(), 4);
        verify_package(&destination).unwrap();
        xlsynth::IrPackage::parse_ir(&destination.to_string(), None).unwrap();
    }

    #[test]
    fn failing_to_rebase_does_not_partially_append_functions_or_files() {
        let mut destination = Parser::new(
            r#"package target
top fn old() -> bits[1] { ret x: bits[1] = literal(value=0, id=1) }"#,
        )
        .parse_package()
        .unwrap();
        let mut imported = Parser::new(
            r#"package source
file_number 0 "new.x"
fn first() -> bits[1] { ret x: bits[1] = literal(value=0, id=1, pos=[(0,0,0)]) }
fn last() -> bits[1] { ret x: bits[1] = literal(value=0, id=2) }"#,
        )
        .parse_package()
        .unwrap();
        let last = imported.get_fn_mut("last").unwrap();
        let result = last.ret_node_ref.unwrap();
        last.get_node_mut(result).text_id = usize::MAX;
        let before = destination.to_string();
        assert!(
            link_imported_functions(
                &mut destination,
                imported,
                &["first".into(), "last".into()],
                &[]
            )
            .is_err()
        );
        assert_eq!(destination.to_string(), before);
    }
}
