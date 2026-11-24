// SPDX-License-Identifier: Apache-2.0

//! Turn evaluated array bounds contracts into ordinary Boolean proof results.

use std::collections::{HashMap, HashSet};

use xlsynth_pir::IrValue;
use xlsynth_pir::block_invoke_inline::expand_iterations;
use xlsynth_pir::ir::{
    Binop, Fn, NaryOp, Node, NodePayload, NodeRef, Package, PackageMember, Type, Unop,
};
use xlsynth_pir::ir_inline::reachable_functions_in_dependency_order;
use xlsynth_pir::ir_rebase_ids::package_max_emitted_node_id;
use xlsynth_pir::ir_utils::{compact_and_toposort_in_place, get_topological, operands};

use super::proof::MAX_PROOF_WORK;

/// Returns true when it appends one bounds predicate per transition result
/// field. Called functions return (value, all_bounds_ok) because concrete
/// invocation evaluates their entire retained bodies, including operations
/// outside the returned value's cone.
pub(super) fn instrument(package: &mut Package) -> Result<bool, String> {
    if !package
        .members
        .iter()
        .any(|m| m.graph().nodes.iter().any(|n| marked(&n.payload)))
    {
        return Ok(false);
    }
    let top = package.get_top_fn().ok_or("missing proof transition")?;
    let mut allocation = ProofAllocation {
        last_id: package_max_emitted_node_id(package),
        remaining: MAX_PROOF_WORK,
    };
    let ordered = reachable_functions_in_dependency_order(package, &[&top.name])?;
    // A zero-trip body can itself contain a huge loop. Only its caller's
    // operands are evaluated, so exclude this edge before rewriting callees.
    let mut executed = HashSet::from([top.name.as_str()]);
    for source in ordered.iter().rev() {
        if executed.contains(source.name.as_str()) {
            executed.extend(source.nodes.iter().filter_map(executed_callee));
        }
    }
    let mut checked = HashMap::new();
    let mut edits = HashMap::new();
    for source in ordered {
        if !executed.contains(source.name.as_str()) {
            edits.insert(source.name.clone(), None);
        } else if source.nodes.iter().any(|n| {
            marked(&n.payload) || executed_callee(n).is_some_and(|name| checked.contains_key(name))
        }) {
            let rewritten =
                rewrite_function(source, source.name == top.name, &checked, &mut allocation)?;
            checked.insert(source.name.as_str(), source);
            edits.insert(source.name.clone(), Some(rewritten));
        } else if source
            .nodes
            .iter()
            .any(|n| empty_iteration_result(n).is_some())
        {
            // Keep the ordinary return signature when the only marked access
            // is in an unexecuted body.
            let mut f = source.clone();
            for node in &mut f.nodes {
                if let Some(result) = empty_iteration_result(node) {
                    replace_operation(node, result);
                }
            }
            edits.insert(source.name.clone(), Some(f));
        }
    }
    let has_bounds = checked.contains_key(top.name.as_str());
    // Apply every edit together so an unsuccessful rewrite leaves the
    // original package intact.
    package
        .members
        .retain_mut(|member| match edits.remove(&member.graph().name) {
            Some(Some(f)) => {
                *member = PackageMember::Function(f);
                true
            }
            Some(None) => false,
            None => true,
        });
    Ok(has_bounds)
}

fn marked(payload: &NodePayload) -> bool {
    matches!(
        payload,
        NodePayload::ArrayIndex {
            assumed_in_bounds: true,
            ..
        } | NodePayload::ArrayUpdate {
            assumed_in_bounds: true,
            ..
        }
    )
}

/// The value of a statically empty iteration has no callee dependency.
fn empty_iteration_result(node: &Node) -> Option<NodePayload> {
    match &node.payload {
        NodePayload::CountedFor {
            init,
            trip_count: 0,
            ..
        } => Some(NodePayload::Unop(Unop::Identity, *init)),
        NodePayload::Map { .. } if matches!(&node.ty, Type::Array(a) if a.element_count == 0) => {
            Some(NodePayload::Array(vec![]))
        }
        _ => None,
    }
}

/// Identifies calls whose body is evaluated at least once.
pub(super) fn executed_callee(node: &Node) -> Option<&str> {
    if empty_iteration_result(node).is_some() {
        return None;
    }
    match &node.payload {
        NodePayload::Invoke { to_apply, .. } | NodePayload::Map { to_apply, .. } => Some(to_apply),
        NodePayload::CountedFor { body, .. } => Some(body),
        _ => None,
    }
}

/// Allocates unique package IDs and bounds additional IR materialized before
/// the symbolic schedule's own expansion accounting runs.
struct ProofAllocation {
    last_id: usize,
    remaining: usize,
}

impl ProofAllocation {
    fn charge(&mut self, count: usize) -> Result<(), String> {
        self.remaining = self.remaining.checked_sub(count).ok_or_else(|| {
            format!("bounds instrumentation exceeds the symbolic unrolling limit of {MAX_PROOF_WORK} work units")
        })?;
        Ok(())
    }

    fn append(&mut self, f: &mut Fn, ty: Type, payload: NodePayload) -> Result<NodeRef, String> {
        self.charge(1)?;
        self.last_id = self
            .last_id
            .checked_add(1)
            .ok_or("proof node IDs overflow usize")?;
        let index = f.nodes.len();
        f.nodes.push(Node {
            text_id: self.last_id,
            name: None,
            ty,
            payload,
            pos: None,
        });
        Ok(NodeRef { index })
    }

    /// An empty conjunction is represented by None until a return needs a bit.
    fn conjunction(
        &mut self,
        f: &mut Fn,
        mut checks: Vec<NodeRef>,
    ) -> Result<Option<NodeRef>, String> {
        checks.sort_unstable_by_key(|r| r.index);
        checks.dedup();
        match checks.as_slice() {
            [] => Ok(None),
            [only] => Ok(Some(*only)),
            _ => self
                .append(f, Type::Bits(1), NodePayload::Nary(NaryOp::And, checks))
                .map(Some),
        }
    }

    /// Records the unsigned check at each indexed dimension using the full
    /// index width. A dimension of four already covers every two-bit index.
    fn check_indices(
        &mut self,
        f: &mut Fn,
        source: &Fn,
        array: NodeRef,
        indices: &[NodeRef],
        checks: &mut Vec<NodeRef>,
    ) -> Result<(), String> {
        let mut ty = source.get_node_ty(array);
        for &index in indices {
            let Type::Array(dim) = ty else {
                return Err("expected an indexed array dimension".into());
            };
            let Type::Bits(width) = *source.get_node_ty(index) else {
                return Err("expected a bits index".into());
            };
            let count = dim.element_count;
            if count == 0 {
                checks.push(self.append(
                    f,
                    Type::Bits(1),
                    NodePayload::Literal(IrValue::bool(false)),
                )?);
            } else if width >= (usize::BITS - count.leading_zeros()) as usize {
                let limit =
                    IrValue::make_ubits(width, u64::try_from(count).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                let limit = self.append(f, Type::Bits(width), NodePayload::Literal(limit))?;
                checks.push(self.append(
                    f,
                    Type::Bits(1),
                    NodePayload::Binop(Binop::Ult, index, limit),
                )?);
            }
            ty = &dim.element_type;
        }
        Ok(())
    }

    /// Projects the original value and its validity from one checked invoke.
    fn check_invoke(&mut self, f: &mut Fn, site: NodeRef) -> Result<NodeRef, String> {
        let node = f.get_node(site);
        let ty = Type::Tuple(vec![Box::new(node.ty.clone()), Box::new(Type::Bits(1))]);
        let payload = node.payload.clone();
        let tuple = self.append(f, ty, payload)?;
        replace_operation(
            f.get_node_mut(site),
            NodePayload::TupleIndex { tuple, index: 0 },
        );
        self.append(
            f,
            Type::Bits(1),
            NodePayload::TupleIndex { tuple, index: 1 },
        )
    }

    /// Unrolls only iterations whose callee now returns an additional validity
    /// bit. Each original iteration contributes its check, including those
    /// before the final carry of a counted loop.
    fn check_iterations(
        &mut self,
        f: &mut Fn,
        source: &Node,
        site: NodeRef,
        callee: &Fn,
        checks: &mut Vec<NodeRef>,
    ) -> Result<(), String> {
        let unroll_work = match &source.payload {
            NodePayload::Map { .. } => {
                let Type::Array(dim) = &source.ty else {
                    return Err("expected array map result".into());
                };
                dim.element_count.saturating_mul(3)
            }
            NodePayload::CountedFor {
                trip_count,
                invariant_args,
                ..
            } => {
                // Each generated call also copies every loop-invariant operand.
                trip_count.saturating_mul(2usize.saturating_add(invariant_args.len()))
            }
            _ => unreachable!("unrolling only maps and counted loops"),
        };
        self.charge(unroll_work)?;
        let first = f.nodes.len();
        let result = expand_iterations(source, callee, &mut f.nodes, &mut self.last_id)?;
        replace_operation(f.get_node_mut(site), result);
        for index in first..f.nodes.len() {
            let inserted = &mut f.nodes[index].payload;
            match inserted {
                NodePayload::Invoke { .. } => checks.push(self.check_invoke(f, NodeRef { index })?),
                NodePayload::ArrayIndex {
                    assumed_in_bounds, ..
                } => {
                    // The map unroller created this read at a literal lane in
                    // 0..count.
                    *assumed_in_bounds = false;
                }
                _ => { /* Induction and array index literals have no contracts. */ }
            }
        }
        Ok(())
    }
}

/// Dotted default names must follow the new operator on serialization.
fn replace_operation(node: &mut Node, payload: NodePayload) {
    if node.name.as_deref().is_some_and(|name| name.contains('.')) {
        node.name = None;
    }
    node.payload = payload;
}

/// Propagates checks through original operand edges before replacing any calls.
/// This retains eager argument checks even for zero-trip loops and ignored
/// parameters. The DAG shares a single validity expression per transition node.
fn rewrite_function(
    source: &Fn,
    is_transition: bool,
    checked: &HashMap<&str, &Fn>,
    allocation: &mut ProofAllocation,
) -> Result<Fn, String> {
    let mut f = source.clone();
    let ret = source.ret_node_ref.ok_or("missing proof function return")?;
    let mut node_checks = vec![None; source.nodes.len()];
    let mut body_checks = Vec::new();
    for site in get_topological(source) {
        // The transition returns a tuple for independent port/state samples.
        if is_transition && site == ret {
            continue;
        }
        let node = source.get_node(site);
        let mut checks = Vec::new();
        if is_transition {
            checks.extend(
                operands(&node.payload)
                    .into_iter()
                    .filter_map(|r| node_checks[r.index]),
            );
        }
        match &node.payload {
            NodePayload::ArrayIndex {
                array,
                indices,
                assumed_in_bounds: true,
            }
            | NodePayload::ArrayUpdate {
                array,
                indices,
                assumed_in_bounds: true,
                ..
            } => {
                allocation.check_indices(&mut f, source, *array, indices, &mut checks)?;
                match &mut f.get_node_mut(site).payload {
                    NodePayload::ArrayIndex {
                        assumed_in_bounds, ..
                    }
                    | NodePayload::ArrayUpdate {
                        assumed_in_bounds, ..
                    } => *assumed_in_bounds = false,
                    _ => unreachable!("copy of annotated array operation"),
                }
            }
            NodePayload::Invoke { to_apply, .. } if checked.contains_key(to_apply.as_str()) => {
                checks.push(allocation.check_invoke(&mut f, site)?);
            }
            NodePayload::Map { to_apply: body, .. } | NodePayload::CountedFor { body, .. } => {
                if let Some(result) = empty_iteration_result(node) {
                    // Operand contracts were collected above, including
                    // those of a discarded input array or invariant argument.
                    replace_operation(f.get_node_mut(site), result);
                } else if let Some(callee) = checked.get(body.as_str()) {
                    allocation.check_iterations(&mut f, node, site, callee, &mut checks)?;
                }
            }
            _ => { /* Ordinary operations contribute only their operands' contracts. */ }
        }
        if is_transition {
            node_checks[site.index] = allocation.conjunction(&mut f, checks)?;
        } else {
            body_checks.extend(checks);
        }
    }
    let always = allocation.append(
        &mut f,
        Type::Bits(1),
        NodePayload::Literal(IrValue::bool(true)),
    )?;
    let fields = if is_transition {
        let NodePayload::Tuple(values) = &source.get_node(ret).payload else {
            return Err("expected transition to return its output and register tuple".into());
        };
        let mut fields = values.clone();
        fields.extend(
            values
                .iter()
                .map(|r| node_checks[r.index].unwrap_or(always)),
        );
        fields
    } else {
        vec![
            ret,
            allocation
                .conjunction(&mut f, body_checks)?
                .unwrap_or(always),
        ]
    };
    f.ret_ty = Type::Tuple(
        fields
            .iter()
            .map(|&r| Box::new(f.get_node_ty(r).clone()))
            .collect(),
    );
    let ty = f.ret_ty.clone();
    f.ret_node_ref = Some(allocation.append(&mut f, ty, NodePayload::Tuple(fields))?);
    compact_and_toposort_in_place(&mut f)?;
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::instrument;
    use xlsynth_pir::IrValue;
    use xlsynth_pir::ir::Type;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
    use xlsynth_pir::ir_inline::package_with_reachable_callees;
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_pir::ir_verify::verify_package;

    #[test]
    fn failed_transition_rewrite_preserves_prepared_callees() {
        let mut package = Parser::new(
            r#"package overflow
fn read(i: bits[1] id=1) -> bits[1] {
  table: bits[1][2] = literal(value=[0, 1], id=2)
  ret bit: bits[1] = array_index(table, indices=[i], assumed_in_bounds=true, id=3)
}
top fn step(i: bits[1] id=4) -> (bits[1]) {
  bit: bits[1] = invoke(i, to_apply=read, id=5)
  ret result: (bits[1]) = tuple(bit, id=6)
}
"#,
        )
        .parse_and_validate_package()
        .unwrap();
        let top = package.get_top_fn_mut().unwrap();
        let ret = top.ret_node_ref.unwrap();
        // The callee needs two IDs for its literal true and return tuple. One
        // further ID is available for the caller, which needs at least two.
        top.get_node_mut(ret).text_id = usize::MAX - 3;
        let before = package.to_string();
        assert_eq!(
            instrument(&mut package).unwrap_err(),
            "proof node IDs overflow usize"
        );
        assert_eq!(package.to_string(), before);
    }

    #[test]
    fn empty_iterations_keep_operand_checks_without_expanding_their_bodies() {
        let original = Parser::new(r#"package skipped
fn leaf(i: bits[24] id=1, acc: bits[1] id=2) -> bits[1] {
  table: bits[1][1] = literal(value=[0], id=3)
  ret read: bits[1] = array_index(table, indices=[i], assumed_in_bounds=true, id=4)
}
fn outer_body(i: bits[1] id=5, acc: bits[1] id=6, arg: bits[1] id=7) -> bits[1] {
  ret result: bits[1] = counted_for(acc, trip_count=2000000, stride=1, body=leaf, id=8)
}
fn map_body(arg: bits[1] id=9) -> bits[2] {
  result: bits[1] = counted_for(arg, trip_count=2000000, stride=1, body=leaf, id=10)
  ret wide: bits[2] = zero_ext(result, new_bit_count=2, id=11)
}
fn empty_loop(x: bits[1] id=12) -> bits[1] {
  ret result: bits[1] = counted_for(x, trip_count=0, stride=1, body=outer_body, invariant_args=[x], id=13)
}
top fn transition(loop_i: bits[1] id=14, map_i: bits[1] id=15) -> (bits[1], bits[2][0], bits[1]) {
  table: bits[1][1] = literal(value=[1], id=16)
  arg: bits[1] = array_index(table, indices=[loop_i], assumed_in_bounds=true, id=17)
  zero: bits[1] = literal(value=0, id=18)
  result: bits[1] = counted_for(zero, trip_count=0, stride=1, body=outer_body, invariant_args=[arg], id=19)
  empty: bits[1][0] = array(id=20)
  rows: bits[1][0][1] = array(empty, id=21)
  input: bits[1][0] = array_index(rows, indices=[map_i], assumed_in_bounds=true, id=22)
  mapped: bits[2][0] = map(input, to_apply=map_body, id=23)
  x: bits[1] = invoke(zero, to_apply=empty_loop, id=24)
  ret fields: (bits[1], bits[2][0], bits[1]) = tuple(result, mapped, x, id=25)
}
"#).parse_and_validate_package().unwrap();
        let mut bounds_free =
            package_with_reachable_callees(&original, original.get_fn("empty_loop").unwrap())
                .unwrap();
        assert!(!instrument(&mut bounds_free).unwrap());
        verify_package(&bounds_free).unwrap();
        assert_eq!(bounds_free.members.len(), 1);
        assert_eq!(bounds_free.get_top_fn().unwrap().ret_ty, Type::Bits(1));

        let mut checked = original;
        assert!(instrument(&mut checked).unwrap());
        verify_package(&checked).unwrap();
        assert_eq!(checked.members.len(), 2);
        let transition = checked.get_top_fn().unwrap();
        let empty_result = IrValue::make_array_typed(Type::Bits(2), &[]).unwrap();
        for loop_i in [false, true] {
            for map_i in [false, true] {
                let args = [IrValue::bool(loop_i), IrValue::bool(map_i)];
                let FnEvalResult::Success(result) = eval_fn_in_package(&checked, transition, &args)
                else {
                    panic!("expected bounds to be represented by result fields");
                };
                assert_eq!(
                    result.value,
                    IrValue::make_tuple(&[
                        IrValue::bool(false),
                        empty_result.clone(),
                        IrValue::bool(false),
                        IrValue::bool(!loop_i),
                        IrValue::bool(!map_i),
                        IrValue::bool(true),
                    ])
                );
            }
        }
    }

    #[test]
    fn counted_loop_checks_all_evaluated_arguments_and_body_contracts() {
        let mut checked = Parser::new(r#"package bounds
fn body(i: bits[1] id=1, acc: bits[1] id=2, arg: bits[1] id=3, row: bits[2] id=4, col: bits[80] id=5) -> bits[1] {
  table: bits[1][3][2] = literal(value=[[0, 0, 0], [0, 0, 0]], id=6)
  dead: bits[1][3][2] = array_update(table, arg, indices=[row, col], assumed_in_bounds=true, id=7)
  ret acc: bits[1] = param(name=acc, id=2)
}
top fn transition(row: bits[2] id=8, col: bits[80] id=9) -> (bits[1], bits[1], bits[1]) {
  table: bits[1][2] = literal(value=[0, 0], id=10)
  argument: bits[1] = array_index(table, indices=[row], assumed_in_bounds=true, id=11)
  init: bits[1] = literal(value=0, id=12)
  empty_with_arg: bits[1] = counted_for(init, trip_count=0, stride=1, body=body, invariant_args=[argument, row, col], id=13)
  empty: bits[1] = counted_for(init, trip_count=0, stride=1, body=body, invariant_args=[init, row, col], id=14)
  once: bits[1] = counted_for(init, trip_count=1, stride=1, body=body, invariant_args=[init, row, col], id=15)
  ret fields: (bits[1], bits[1], bits[1]) = tuple(empty_with_arg, empty, once, id=16)
}
"#).parse_and_validate_package().unwrap();
        assert!(instrument(&mut checked).unwrap());
        verify_package(&checked).unwrap();
        let transition = checked.get_top_fn().unwrap();
        for (row, col, argument_ok, body_ok) in [
            (1, "bits[80]:2", true, true),
            (2, "bits[80]:2", false, false),
            (1, "bits[80]:0x10000000000000000", true, false),
        ] {
            let args = [
                IrValue::make_ubits(2, row).unwrap(),
                IrValue::parse_typed(col).unwrap(),
            ];
            let FnEvalResult::Success(result) = eval_fn_in_package(&checked, transition, &args)
            else {
                panic!("instrumented transition still has a failing annotation");
            };
            assert_eq!(
                result.value,
                IrValue::make_tuple(&[
                    IrValue::bool(false),
                    IrValue::bool(false),
                    IrValue::bool(false),
                    IrValue::bool(argument_ok),
                    IrValue::bool(true),
                    IrValue::bool(body_ok),
                ])
            );
        }
    }
}
