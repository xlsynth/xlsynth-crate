// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;

use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::xls_ir::ir;
use xlsynth_g8r::xls_ir::ir::{Fn as IrFn, Node, NodePayload, NodeRef, Type};
use xlsynth_g8r::xls_ir::localized_eco;
use xlsynth_g8r::xls_ir::ir_parser;
use xlsynth_test_helpers::ir_fuzz::{FuzzSample, generate_ir_fn};

fn make_new_with_simple_edit(mut f: IrFn) -> Option<IrFn> {
    // Insert a new node and substitute it as an operand into a suitable target node.
    // Only perform edits that keep the parent operator unchanged (operand rewires/insertions).
    let width = match &f.ret_ty { Type::Bits(w) => *w, _ => 0 };
    if width == 0 {
        return None; // Non-bits return types: skip edits
    }
    let cur_len = f.nodes.len();
    if cur_len == 0 {
        return None;
    }
    // Find a candidate Binop/Nary node to rewrite.
    let target_idx_opt = (0..cur_len).rev().find(|&idx| match f.nodes[idx].payload {
        NodePayload::Binop(_, _, _) => true,
        NodePayload::Nary(_, ref elems) => !elems.is_empty(),
        _ => false,
    });
    let target_idx = if let Some(i) = target_idx_opt { i } else { return None };

    // Build the inserted node from earlier nodes.
    let insert_a = NodeRef { index: cur_len.saturating_sub(1) };
    let insert_b = NodeRef { index: cur_len / 2 };
    let new_payload = if (cur_len % 2) == 0 {
        NodePayload::Unop(ir::Unop::Not, insert_a)
    } else {
        NodePayload::Nary(ir::NaryOp::And, vec![insert_a, insert_b])
    };
    let new_idx = f.nodes.len();
    let new_text_id = f.nodes.last().map(|n| n.text_id + 1).unwrap_or(1);
    f.nodes.push(Node {
        text_id: new_text_id,
        name: None,
        ty: Type::Bits(width),
        payload: new_payload,
        pos: None,
    });

    // Rewrite one operand of the target to point to the newly inserted node.
    match &mut f.nodes[target_idx].payload {
        NodePayload::Binop(_, ref mut a, _) => {
            *a = NodeRef { index: new_idx };
        }
        NodePayload::Nary(_, ref mut elems) if !elems.is_empty() => {
            elems[0] = NodeRef { index: new_idx };
        }
        _ => unreachable!(),
    }
    Some(f)
}

fuzz_target!(|sample: FuzzSample| {
    // Build an initial XLS IR package from the fuzz sample, then parse to g8r IR.
    if sample.input_bits == 0 || sample.ops.is_empty() {
        return;
    }
    let _ = env_logger::builder().is_test(true).try_init();
    let mut pkg = match xlsynth::IrPackage::new("fuzz_pkg") {
        Ok(p) => p,
        Err(_) => return,
    };
    if let Err(e) = generate_ir_fn(sample.input_bits, sample.ops.clone(), &mut pkg) {
        log::info!("generate_ir_fn failed: {}", e);
        return;
    }
    let pkg_text = pkg.to_string();
    let parsed = match ir_parser::Parser::new(&pkg_text).parse_package() {
        Ok(p) => p,
        Err(e) => {
            log::info!("parse_package failed: {}", e);
            return;
        }
    };
    let Some(old_fn) = parsed.get_top() else { return; };
    // Only operate on bit-returning functions to keep edits simple/valid.
    if !matches!(old_fn.ret_ty, Type::Bits(_)) {
        return;
    }
    let mut new_fn = old_fn.clone();
    if let Some(edited) = make_new_with_simple_edit(new_fn) {
        new_fn = edited;
    } else {
        return; // No suitable edit for this sample
    }

    // Compute localized ECO, apply to old to produce patched(old).
    let diff = localized_eco::compute_localized_eco(&old_fn, &new_fn);
    let patched = localized_eco::apply_localized_eco(&old_fn, &new_fn, &diff);

    // Prove patched(old) ≡ new using external checker (feature-independent).
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        panic!("XLSYNTH_TOOLS environment variable must be set for fuzzing.");
    }
    let lhs_pkg_text = format!("package lhs\n\ntop {}", patched.to_string());
    let rhs_pkg_text = format!("package rhs\n\ntop {}", new_fn.to_string());
    if let Err(e) = check_equivalence::check_equivalence_with_top(
        &lhs_pkg_text,
        &rhs_pkg_text,
        Some(new_fn.name.as_str()),
        false,
    ) {
        panic!("external ir-equiv failed: {}", e);
    }
});
