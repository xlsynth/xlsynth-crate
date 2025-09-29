// SPDX-License-Identifier: Apache-2.0

use crate::prover::Prover;
use crate::types::{
    BoolPropertyResult, IrFn, ParamDomains, ProverFn, QuickCheckAssertionSemantics,
};

use std::collections::HashMap;
use xlsynth::IrValue;

use xlsynth_pir::ir::{self, Binop, NaryOp, Node, NodePayload, NodeRef, ParamId, Type};
use xlsynth_pir::ir_utils::{
    next_text_id, param_node_ref_by_index, param_node_ref_by_name, param_type_by_name,
};

pub const ASSERT_LABEL_PREFIX: &str = "enum-in-bound";

fn push_node_with_offset(
    nodes: &mut Vec<Node>,
    base_index: usize,
    next_id: &mut usize,
    name: Option<String>,
    ty: Type,
    payload: NodePayload,
) -> NodeRef {
    let text_id = *next_id;
    *next_id += 1;
    let node = Node {
        text_id,
        name,
        ty,
        payload,
        pos: None,
    };
    let index = base_index + nodes.len();
    nodes.push(node);
    NodeRef { index }
}

fn find_implicit_token_ref(f: &ir::Fn) -> Result<NodeRef, String> {
    match f.params.first() {
        Some(param) if matches!(param.ty, Type::Token) => {
            Ok(param_node_ref_by_index(f, 0).unwrap())
        }
        _ => Err(format!(
            "Implicit token parameter not found in IR function '{}'",
            f.name
        )),
    }
}

fn find_implicit_activation_ref(f: &ir::Fn) -> Result<NodeRef, String> {
    match f.params.get(1) {
        Some(param) if matches!(param.ty, Type::Bits(1)) => {
            Ok(param_node_ref_by_index(f, 1).unwrap())
        }
        _ => Err(format!(
            "Implicit activation parameter not found in IR function '{}'",
            f.name
        )),
    }
}

fn find_return_token_ref(f: &ir::Fn) -> Result<NodeRef, String> {
    let ret_ref = f
        .ret_node_ref
        .ok_or_else(|| format!("IR function '{}' has no return node", f.name))?;
    let ret_node = &f.nodes[ret_ref.index];
    if let NodePayload::Tuple(elements) = &ret_node.payload {
        let candidate = elements
            .first()
            .ok_or_else(|| format!("Return tuple in '{}' is empty", f.name))?;
        let node = &f.nodes[candidate.index];
        if matches!(node.ty, Type::Token) {
            Ok(*candidate)
        } else {
            Err(format!(
                "First return element in '{}' is not a token",
                f.name
            ))
        }
    } else {
        Err(format!(
            "Return value of '{}' is not a tuple containing a token",
            f.name
        ))
    }
}

fn instrument_function_for_enum_bounds(
    f: &mut ir::Fn,
    domains: &ParamDomains,
    next_id: &mut usize,
) -> Result<(), String> {
    if domains.is_empty() {
        return Ok(());
    }

    let token_param_ref = find_implicit_token_ref(f)?;
    let insertion_index = f
        .ret_node_ref
        .map(|nr| nr.index)
        .unwrap_or_else(|| f.nodes.len());
    let mut pending_nodes: Vec<Node> = Vec::new();

    let original_return_token = find_return_token_ref(f).unwrap_or(token_param_ref);
    let mut current_token_ref = original_return_token;
    let base_activation_ref = find_implicit_activation_ref(f)?;

    // Collect domain entries and sort them by parameter name for determinism.
    let mut domain_entries: Vec<(&String, &Vec<IrValue>)> = domains.iter().collect();
    domain_entries.sort_by(|(a, _), (b, _)| a.cmp(b));

    for (param_name, allowed_values) in domain_entries {
        if allowed_values.is_empty() {
            continue;
        }

        let missing_param_msg = format!(
            "Parameter '{}' not found in IR function '{}'",
            param_name, f.name
        );

        let param_ty = match param_type_by_name(f, param_name.as_str()) {
            Some(ty @ Type::Bits(_)) => ty,
            Some(_) => {
                return Err(format!(
                    "Enum parameter '{}' in function '{}' does not lower to bits",
                    param_name, f.name
                ));
            }
            None => return Err(missing_param_msg.clone()),
        };

        let param_node_ref = param_node_ref_by_name(f, param_name.as_str())
            .ok_or_else(|| missing_param_msg.clone())?;

        let mut eq_refs: Vec<NodeRef> = Vec::with_capacity(allowed_values.len());
        for (idx, value) in allowed_values.iter().enumerate() {
            let literal_ref = push_node_with_offset(
                &mut pending_nodes,
                insertion_index,
                next_id,
                Some(format!(
                    "enum_in_bound_value__{}__{}__{}",
                    f.name, param_name, idx
                )),
                param_ty.clone(),
                NodePayload::Literal(value.clone()),
            );

            let eq_ref = push_node_with_offset(
                &mut pending_nodes,
                insertion_index,
                next_id,
                Some(format!(
                    "enum_in_bound_eq__{}__{}__{}",
                    f.name, param_name, idx
                )),
                Type::Bits(1),
                NodePayload::Binop(Binop::Eq, param_node_ref, literal_ref),
            );
            eq_refs.push(eq_ref);
        }

        let predicate_ref = if eq_refs.len() == 1 {
            eq_refs[0]
        } else {
            push_node_with_offset(
                &mut pending_nodes,
                insertion_index,
                next_id,
                Some(format!("enum_in_bound_any__{}__{}", f.name, param_name)),
                Type::Bits(1),
                NodePayload::Nary(NaryOp::Or, eq_refs),
            )
        };

        let activate_ref = push_node_with_offset(
            &mut pending_nodes,
            insertion_index,
            next_id,
            Some(format!(
                "enum_in_bound_activation__{}__{}",
                f.name, param_name
            )),
            Type::Bits(1),
            NodePayload::Nary(NaryOp::And, vec![base_activation_ref, predicate_ref]),
        );

        let token_input = current_token_ref;
        let message = format!(
            "Enum argument '{}' out of bounds in function '{}'",
            param_name, f.name
        );
        let label = format!("{}::{}::{}", ASSERT_LABEL_PREFIX, f.name, param_name);
        let assert_ref = push_node_with_offset(
            &mut pending_nodes,
            insertion_index,
            next_id,
            Some(format!("enum_in_bound_assert__{}__{}", f.name, param_name)),
            Type::Token,
            NodePayload::Assert {
                token: token_input,
                activate: activate_ref,
                message,
                label,
            },
        );
        current_token_ref = assert_ref;
    }

    let pending_count = pending_nodes.len();
    if pending_count > 0 {
        f.nodes
            .splice(insertion_index..insertion_index, pending_nodes.into_iter());
        if let Some(ret_ref) = f.ret_node_ref.as_mut() {
            ret_ref.index += pending_count;
        }
    }

    if let Some(ret_ref) = f.ret_node_ref.as_ref() {
        if let NodePayload::Tuple(elements) = &mut f.nodes[ret_ref.index].payload {
            if !elements.is_empty() {
                elements[0] = current_token_ref;
            }
        }
    }

    Ok(())
}

fn clone_param_with_new_id(param: &ir::Param, id: usize) -> ir::Param {
    ir::Param {
        name: param.name.clone(),
        ty: param.ty.clone(),
        id: ParamId::new(id),
    }
}

pub fn add_property_function(
    pkg: &mut ir::Package,
    top_name: &str,
    next_id_hint: usize,
) -> Result<String, String> {
    let top_fn = pkg
        .get_fn(top_name)
        .ok_or_else(|| format!("IR function '{}' not found", top_name))?;

    let property_name = format!("__enum_in_bound_property__{}", top_name);
    if pkg.get_fn(&property_name).is_some() {
        return Ok(property_name);
    }

    let mut nodes = vec![Node {
        text_id: 0,
        name: Some("reserved_zero_node".to_string()),
        ty: Type::nil(),
        payload: NodePayload::Nil,
        pos: None,
    }];

    let mut params: Vec<ir::Param> = Vec::with_capacity(top_fn.params.len());
    let mut arg_refs: Vec<NodeRef> = Vec::with_capacity(top_fn.params.len());

    let mut next_id = next_id_hint;

    for param in &top_fn.params {
        let new_param = clone_param_with_new_id(param, next_id);
        let param_name = new_param.name.clone();
        let param_ty = new_param.ty.clone();
        let param_id = new_param.id;
        let node_ref = push_node_with_offset(
            &mut nodes,
            0,
            &mut next_id,
            Some(param_name),
            param_ty,
            NodePayload::GetParam(param_id),
        );
        params.push(new_param);
        arg_refs.push(node_ref);
    }

    let invoke_ref = push_node_with_offset(
        &mut nodes,
        0,
        &mut next_id,
        Some(format!("enum_in_bound_invoke__{}", top_name)),
        top_fn.ret_ty.clone(),
        NodePayload::Invoke {
            to_apply: top_fn.name.clone(),
            operands: arg_refs.clone(),
        },
    );

    let bool_ref = push_node_with_offset(
        &mut nodes,
        0,
        &mut next_id,
        Some("enum_in_bound_true".to_string()),
        Type::Bits(1),
        NodePayload::Literal(IrValue::make_ubits(1, 1).expect("make bool literal")),
    );

    let tuple_ty = Type::Tuple(vec![
        Box::new(top_fn.ret_ty.clone()),
        Box::new(Type::Bits(1)),
    ]);
    let tuple_ref = push_node_with_offset(
        &mut nodes,
        0,
        &mut next_id,
        Some("enum_in_bound_pair".to_string()),
        tuple_ty,
        NodePayload::Tuple(vec![invoke_ref, bool_ref]),
    );

    let ret_ref = push_node_with_offset(
        &mut nodes,
        0,
        &mut next_id,
        Some("enum_in_bound_result".to_string()),
        Type::Bits(1),
        NodePayload::TupleIndex {
            tuple: tuple_ref,
            index: 1,
        },
    );

    let property_fn = ir::Fn {
        name: property_name.clone(),
        params,
        ret_ty: Type::Bits(1),
        nodes,
        ret_node_ref: Some(ret_ref),
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
    };

    pkg.members.push(ir::PackageMember::Function(property_fn));
    Ok(property_name)
}

pub fn prepare_package_for_enum_in_bound(
    pkg: &mut ir::Package,
    top_name: &str,
    targets: &HashMap<String, ParamDomains>,
) -> Result<String, String> {
    pkg.get_fn(top_name)
        .ok_or_else(|| format!("IR function '{}' not found", top_name))?;
    let mut next_id_hint = next_text_id(pkg);

    for (fn_name, domains) in targets {
        let func = pkg
            .get_fn_mut(fn_name)
            .ok_or_else(|| format!("IR function '{}' not found", fn_name))?;
        instrument_function_for_enum_bounds(func, domains, &mut next_id_hint)?;
    }

    add_property_function(pkg, top_name, next_id_hint)
}

pub fn prove_enum_in_bound(
    prover: &dyn Prover,
    pkg: &ir::Package,
    top_name: &str,
    targets: &HashMap<String, ParamDomains>,
    top_domains: Option<&ParamDomains>,
) -> BoolPropertyResult {
    let mut instrumented_pkg = pkg.clone();

    let property_name =
        match prepare_package_for_enum_in_bound(&mut instrumented_pkg, top_name, targets) {
            Ok(name) => name,
            Err(err) => return BoolPropertyResult::Error(err),
        };
    let property_fn = match instrumented_pkg.get_fn(&property_name) {
        Some(func) => func,
        None => {
            return BoolPropertyResult::Error(format!("IR function '{}' not found", property_name));
        }
    };
    let ir_fn = IrFn {
        fn_ref: property_fn,
        pkg_ref: Some(&instrumented_pkg),
        fixed_implicit_activation: true,
    };
    let prover_fn = ProverFn {
        ir_fn: &ir_fn,
        domains: top_domains.cloned(),
        uf_map: HashMap::new(),
    };

    let label_regex = format!("^{}::", ASSERT_LABEL_PREFIX);
    prover.prove_ir_fn_always_true_full(
        &prover_fn,
        QuickCheckAssertionSemantics::Never,
        Some(label_regex.as_str()),
        &HashMap::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use xlsynth_pir::ir_parser::Parser;

    fn parse_fn(ir_text: &str) -> ir::Fn {
        let mut parser = Parser::new(ir_text);
        parser.parse_fn().expect("IR function must parse")
    }

    fn sample_domain() -> ParamDomains {
        HashMap::from([(
            "x".to_string(),
            vec![
                IrValue::make_ubits(2, 0).expect("make ubits"),
                IrValue::make_ubits(2, 1).expect("make ubits"),
            ],
        )])
    }

    #[test]
    fn requires_implicit_token_parameter() {
        let ir_text = r#"
fn foo(x: bits[2] id=1) -> bits[2] {
  ret x: bits[2] = param(name=x, id=1)
}
"#;
        let mut func = parse_fn(ir_text);
        let domains = sample_domain();
        let mut next_id = func.nodes.iter().map(|n| n.text_id).max().unwrap_or(0) + 1;
        let err = instrument_function_for_enum_bounds(&mut func, &domains, &mut next_id)
            .expect_err("expected missing token error");
        assert_eq!(
            err,
            "Implicit token parameter not found in IR function 'foo'"
        );
    }

    #[test]
    fn activation_gate_threads_base_activation() {
        let ir_text = r#"
fn __itok__target(__token: token id=1, __activation: bits[1] id=2, x: bits[2] id=3) -> (token, bits[2]) {
  __token: token = param(name=__token, id=1)
  __activation: bits[1] = param(name=__activation, id=2)
  x: bits[2] = param(name=x, id=3)
  ret tuple.4: (token, bits[2]) = tuple(__token, x, id=4)
}
"#;
        let mut func = parse_fn(ir_text);
        let domains = sample_domain();
        let mut next_id = func.nodes.iter().map(|n| n.text_id).max().unwrap_or(0) + 1;
        instrument_function_for_enum_bounds(&mut func, &domains, &mut next_id)
            .expect("instrumentation succeeds");

        let activation_param_idx = func
            .nodes
            .iter()
            .position(|n| n.name.as_deref() == Some("__activation"))
            .expect("activation param node");

        let predicate_name = format!("enum_in_bound_any__{}__{}", func.name, "x");
        let predicate_idx = func
            .nodes
            .iter()
            .position(|n| n.name.as_deref() == Some(&predicate_name))
            .expect("predicate node");

        let activation_name = format!("enum_in_bound_activation__{}__{}", func.name, "x");
        let activation_idx = func
            .nodes
            .iter()
            .position(|n| n.name.as_deref() == Some(&activation_name))
            .expect("activation gate node");
        let activation_node = &func.nodes[activation_idx];
        match &activation_node.payload {
            NodePayload::Nary(NaryOp::And, operands) => {
                assert!(
                    operands.contains(&NodeRef {
                        index: activation_param_idx
                    }),
                    "activation gate should reference the implicit activation parameter"
                );
                assert!(
                    operands.contains(&NodeRef {
                        index: predicate_idx
                    }),
                    "activation gate should reference the predicate node"
                );
            }
            other => panic!("expected activation gate, found {other:?}"),
        }

        let token_param_idx = func
            .nodes
            .iter()
            .position(|n| n.name.as_deref() == Some("__token"))
            .expect("token param node");
        let assert_name = format!("enum_in_bound_assert__{}__{}", func.name, "x");
        let assert_idx = func
            .nodes
            .iter()
            .position(|n| n.name.as_deref() == Some(&assert_name))
            .expect("assert node");
        match &func.nodes[assert_idx].payload {
            NodePayload::Assert {
                token,
                activate,
                label,
                ..
            } => {
                assert_eq!(
                    *token,
                    NodeRef {
                        index: token_param_idx
                    }
                );
                assert_eq!(
                    *activate,
                    NodeRef {
                        index: activation_idx
                    }
                );
                assert_eq!(
                    label,
                    &format!("{}::{}::{}", ASSERT_LABEL_PREFIX, func.name, "x")
                );
            }
            other => panic!("expected assert payload, got {other:?}"),
        }
    }

    #[test]
    fn predicate_requires_activation_parameter() {
        let ir_text = r#"
fn __itok__target(__token: token id=1, x: bits[2] id=2) -> (token, bits[2]) {
  __token: token = param(name=__token, id=1)
  x: bits[2] = param(name=x, id=2)
  ret tuple.3: (token, bits[2]) = tuple(__token, x, id=3)
}
"#;
        let mut func = parse_fn(ir_text);
        let domains = sample_domain();
        let mut next_id = func.nodes.iter().map(|n| n.text_id).max().unwrap_or(0) + 1;
        let err = instrument_function_for_enum_bounds(&mut func, &domains, &mut next_id)
            .expect_err("missing activation should be an error");
        assert_eq!(
            err,
            "Implicit activation parameter not found in IR function '__itok__target'"
        );
    }

    const EXPECTED_TARGET_IR: &str = r#"fn __itok__target(__token: token id=1, __activation: bits[1] id=2, x: bits[2] id=3, y: bits[2] id=4) -> (token, bits[2], bits[2]) {
  sum: bits[2] = add(x, y, id=5)
  diff: bits[2] = xor(sum, x, id=6)
  user_fail: bits[1] = literal(value=0, id=7)
  user_assert: token = assert(__token, user_fail, message="user-level assert", label="user", id=8)
  enum_in_bound_value____itok__target__x__0: bits[2] = literal(value=0, id=10)
  enum_in_bound_eq____itok__target__x__0: bits[1] = eq(x, enum_in_bound_value____itok__target__x__0, id=11)
  enum_in_bound_value____itok__target__x__1: bits[2] = literal(value=1, id=12)
  enum_in_bound_eq____itok__target__x__1: bits[1] = eq(x, enum_in_bound_value____itok__target__x__1, id=13)
  enum_in_bound_any____itok__target__x: bits[1] = or(enum_in_bound_eq____itok__target__x__0, enum_in_bound_eq____itok__target__x__1, id=14)
  enum_in_bound_activation____itok__target__x: bits[1] = and(__activation, enum_in_bound_any____itok__target__x, id=15)
  enum_in_bound_assert____itok__target__x: token = assert(user_assert, enum_in_bound_activation____itok__target__x, message="Enum argument 'x' out of bounds in function '__itok__target'", label="enum-in-bound::__itok__target::x", id=16)
  enum_in_bound_value____itok__target__y__0: bits[2] = literal(value=2, id=17)
  enum_in_bound_eq____itok__target__y__0: bits[1] = eq(y, enum_in_bound_value____itok__target__y__0, id=18)
  enum_in_bound_value____itok__target__y__1: bits[2] = literal(value=3, id=19)
  enum_in_bound_eq____itok__target__y__1: bits[1] = eq(y, enum_in_bound_value____itok__target__y__1, id=20)
  enum_in_bound_any____itok__target__y: bits[1] = or(enum_in_bound_eq____itok__target__y__0, enum_in_bound_eq____itok__target__y__1, id=21)
  enum_in_bound_activation____itok__target__y: bits[1] = and(__activation, enum_in_bound_any____itok__target__y, id=22)
  enum_in_bound_assert____itok__target__y: token = assert(enum_in_bound_assert____itok__target__x, enum_in_bound_activation____itok__target__y, message="Enum argument 'y' out of bounds in function '__itok__target'", label="enum-in-bound::__itok__target::y", id=23)
  ret tuple.9: (token, bits[2], bits[2]) = tuple(enum_in_bound_assert____itok__target__y, diff, sum, id=9)
}"#;

    const EXPECTED_PROPERTY_IR: &str = r#"fn __enum_in_bound_property____itok__target(__token: token id=24, __activation: bits[1] id=25, x: bits[2] id=26, y: bits[2] id=27) -> bits[1] {
  enum_in_bound_invoke____itok__target: (token, bits[2], bits[2]) = invoke(__token, __activation, x, y, to_apply=__itok__target, id=28)
  enum_in_bound_true: bits[1] = literal(value=1, id=29)
  enum_in_bound_pair: ((token, bits[2], bits[2]), bits[1]) = tuple(enum_in_bound_invoke____itok__target, enum_in_bound_true, id=30)
  ret enum_in_bound_result: bits[1] = tuple_index(enum_in_bound_pair, index=1, id=31)
}"#;

    #[test]
    fn token_chain_updates_all_token_users() {
        let pkg_text = r#"package test

fn __itok__target(
    __token: token id=1,
    __activation: bits[1] id=2,
    x: bits[2] id=3,
    y: bits[2] id=4
) -> (token, bits[2], bits[2]) {
  sum: bits[2] = add(x, y, id=5)
  diff: bits[2] = xor(sum, x, id=6)
  user_fail: bits[1] = literal(value=0, id=7)
  user_assert: token = assert(__token, user_fail, message="user-level assert", label="user", id=8)
  ret tuple.9: (token, bits[2], bits[2]) = tuple(user_assert, diff, sum, id=9)
}
"#;

        let mut parser = Parser::new(pkg_text);
        let mut pkg = parser
            .parse_and_validate_package()
            .expect("package must parse");

        let targets = HashMap::from([(
            "__itok__target".to_string(),
            HashMap::from([
                (
                    "x".to_string(),
                    vec![
                        IrValue::make_ubits(2, 0).expect("ubits"),
                        IrValue::make_ubits(2, 1).expect("ubits"),
                    ],
                ),
                (
                    "y".to_string(),
                    vec![
                        IrValue::make_ubits(2, 2).expect("ubits"),
                        IrValue::make_ubits(2, 3).expect("ubits"),
                    ],
                ),
            ]),
        )]);

        let property_name = prepare_package_for_enum_in_bound(&mut pkg, "__itok__target", &targets)
            .expect("prepare succeeds");
        Parser::new(&pkg.to_string())
            .parse_and_validate_package()
            .expect("package must parse");

        Parser::new(EXPECTED_TARGET_IR)
            .parse_fn()
            .expect("expected target IR to parse");
        Parser::new(EXPECTED_PROPERTY_IR)
            .parse_fn()
            .expect("expected property IR to parse");

        let actual_fn = pkg.get_fn("__itok__target").expect("target function");
        let actual_fn_str = ir::emit_fn_with_node_comments(actual_fn, |_, _| None);
        assert_eq!(actual_fn_str, EXPECTED_TARGET_IR);

        let actual_property = pkg.get_fn(&property_name).expect("property function");
        let actual_property_str = ir::emit_fn_with_node_comments(actual_property, |_, _| None);
        assert_eq!(actual_property_str, EXPECTED_PROPERTY_IR);
    }

    #[cfg(any(
        feature = "with-bitwuzla-built",
        feature = "with-boolector-built",
        feature = "with-bitwuzla-binary-test",
        feature = "with-boolector-binary-test",
        feature = "with-z3-binary-test"
    ))]
    fn nested_targets_fixture() -> (ir::Package, HashMap<String, ParamDomains>, ParamDomains) {
        let pkg_text = r#"package test

fn __itok__inner(
    __token: token id=1,
    __activation: bits[1] id=2,
    a: bits[2] id=3,
    b: bits[2] id=4
) -> (token, bits[2]) {
  sum: bits[2] = add(a, b, id=5)
  inner_fail: bits[1] = literal(value=0, id=6)
  inner_assert: token = assert(__token, inner_fail, message="inner user assert", label="inner-user", id=7)
  ret tuple.8: (token, bits[2]) = tuple(inner_assert, sum, id=8)
}

// Collect domain entries and sort them by parameter name for determinism.    for param in &top_fn.params {
fn __itok__top(
    __token: token id=17,
    __activation: bits[1] id=18,
    a: bits[2] id=19,
    b: bits[2] id=20
) -> (token, bits[2]) {
  call: (token, bits[2]) = invoke(__token, __activation, a, b, to_apply=__itok__inner, id=21)
  call_token: token = tuple_index(call, index=0, id=22)
  call_value: bits[2] = tuple_index(call, index=1, id=23)
  top_fail: bits[1] = literal(value=0, id=24)
  top_assert: token = assert(call_token, top_fail, message="top user assert", label="top-user", id=25)
  ret tuple.26: (token, bits[2]) = tuple(top_assert, call_value, id=26)
}
"#;

        let mut parser = Parser::new(pkg_text);
        let pkg = parser
            .parse_and_validate_package()
            .expect("package must parse");

        let mk_domains = |vals_a: &[u64], vals_b: &[u64]| -> ParamDomains {
            let to_values = |vals: &[u64]| -> Vec<IrValue> {
                vals.iter()
                    .map(|v| IrValue::make_ubits(2, *v).expect("ubits"))
                    .collect()
            };
            HashMap::from([
                ("a".to_string(), to_values(vals_a)),
                ("b".to_string(), to_values(vals_b)),
            ])
        };

        let top_domains = mk_domains(&[0, 1], &[2, 3]);
        let targets = HashMap::from([
            ("__itok__inner".to_string(), mk_domains(&[0, 1], &[2, 3])),
            ("__itok__top".to_string(), top_domains.clone()),
        ]);

        (pkg, targets, top_domains)
    }

    #[cfg(any(
        feature = "with-bitwuzla-built",
        feature = "with-boolector-built",
        feature = "with-bitwuzla-binary-test",
        feature = "with-boolector-binary-test",
        feature = "with-z3-binary-test"
    ))]
    fn check_nested_targets(prover: &dyn crate::prover::Prover) {
        let (pkg, targets, top_domains) = nested_targets_fixture();
        let result = prove_enum_in_bound(prover, &pkg, "__itok__top", &targets, Some(&top_domains));
        assert_eq!(result, BoolPropertyResult::Proved);
        let result = prove_enum_in_bound(prover, &pkg, "__itok__top", &targets, None);
        assert!(matches!(result, BoolPropertyResult::Disproved { .. }));
    }

    #[cfg(feature = "with-bitwuzla-built")]
    #[test]
    fn nested_targets_prove_success_bitwuzla_built() {
        let prover = crate::bitwuzla_backend::BitwuzlaOptions::new();
        check_nested_targets(&prover);
    }

    #[cfg(feature = "with-boolector-built")]
    #[test]
    fn nested_targets_prove_success_boolector_built() {
        let prover = crate::boolector_backend::BoolectorConfig::new();
        check_nested_targets(&prover);
    }

    #[cfg(feature = "with-bitwuzla-binary-test")]
    #[test]
    fn nested_targets_prove_success_bitwuzla_binary() {
        let prover = crate::easy_smt_backend::EasySmtConfig::bitwuzla();
        check_nested_targets(&prover);
    }

    #[cfg(feature = "with-boolector-binary-test")]
    #[test]
    fn nested_targets_prove_success_boolector_binary() {
        let prover = crate::easy_smt_backend::EasySmtConfig::boolector();
        check_nested_targets(&prover);
    }

    #[cfg(feature = "with-z3-binary-test")]
    #[test]
    fn nested_targets_prove_success_z3_binary() {
        let prover = crate::easy_smt_backend::EasySmtConfig::z3();
        check_nested_targets(&prover);
    }
}
