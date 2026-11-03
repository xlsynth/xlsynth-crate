// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::ir_parser::Parser;
use crate::ir_utils::remap_payload_with;

fn setup(text: &str) -> ir::Fn {
    Parser::new(text)
        .parse_and_validate_package()
        .unwrap()
        .get_top_fn()
        .unwrap()
        .clone()
}

fn limits() -> Limits {
    Limits {
        nodes: 48,
        edges: 192,
        value_bits: 256,
        input_bits: 8,
    }
}

#[test]
fn large_unrelated_parameters_and_shared_intermediates_are_outside_the_plan() {
    let extras = (0..40)
        .map(|i| format!(", z{i}: (bits[7000], bits[1])"))
        .collect::<String>();
    let f = setup(&format!(
        "package p
        top fn f(k: bits[3]{extras}) -> bits[3] {{
            negated: bits[3] = not(k, id=100)
            ret doubled: bits[3] = add(negated, negated, id=101)
        }}"
    ));
    let budget = &mut 1000;
    let plan = BitsCone::compile(&f, f.ret_node_ref.unwrap(), limits(), budget).unwrap();
    assert_eq!(plan.input(), Some((f.params[0], 3)));
    assert_eq!(1000 - *budget, 6);
    for k in 0..8 {
        let input = IrValue::make_ubits(3, k).unwrap();
        assert_eq!(
            plan.evaluate(Some(&input))
                .unwrap()
                .as_bits()
                .unwrap()
                .to_u64()
                .unwrap(),
            2 * (!k & 7) & 7
        );
    }
    assert_eq!(
        plan.evaluate(Some(&IrValue::bool(false))).unwrap_err(),
        Error::InputDomain
    );
}

#[test]
fn evaluation_order_follows_edges_even_when_users_precede_definitions() {
    let mut f = setup(
        "package p
        top fn f(k: bits[3]) -> bits[3] {
            first: bits[3] = not(k, id=100)
            ret last: bits[3] = neg(first, id=101)
        }",
    );
    let last = f.ret_node_ref.unwrap();
    let first = operands(&f.get_node(last).payload)[0];
    let map = |r| {
        if r == first {
            last
        } else if r == last {
            first
        } else {
            r
        }
    };
    f.nodes.swap(first.index, last.index);
    for node in &mut f.nodes {
        node.payload = remap_payload_with(&node.payload, |(_, r)| map(r));
    }
    let plan = BitsCone::compile(&f, first, limits(), &mut 1000).unwrap();
    for k in 0..8 {
        let input = IrValue::make_ubits(3, k).unwrap();
        assert_eq!(
            plan.evaluate(Some(&input))
                .unwrap()
                .as_bits()
                .unwrap()
                .to_u64()
                .unwrap(),
            (k + 1) & 7
        );
    }
    f.get_node_mut(last).payload = P::Unop(ir::Unop::Not, first);
    assert!(matches!(
        BitsCone::compile(&f, first, limits(), &mut 1000),
        Err(Error::Cycle)
    ));
}

#[test]
fn rejects_extra_inputs_effects_and_excessive_work() {
    let f = setup(
        "package p
      top fn f(x: bits[2], y: bits[2]) -> bits[2] { ret sum: bits[2] = add(x, y, id=100) }",
    );
    let root = f.ret_node_ref.unwrap();
    assert!(matches!(
        BitsCone::compile(&f, root, limits(), &mut 1000),
        Err(Error::InputDomain)
    ));
    assert!(matches!(
        BitsCone::compile(&f, root, limits(), &mut 1),
        Err(Error::Work)
    ));
    let mut f = setup(
        "package p
      top fn f(x: bits[1]) -> bits[1] { ret joint: bits[1] = and(x, x, x, x, id=100) }",
    );
    let root = f.ret_node_ref.unwrap();
    assert!(matches!(
        BitsCone::compile(
            &f,
            root,
            Limits {
                edges: 3,
                ..limits()
            },
            &mut 1000
        ),
        Err(Error::Budget)
    ));
    assert!(matches!(
        BitsCone::compile(
            &f,
            root,
            Limits {
                nodes: 1,
                ..limits()
            },
            &mut 1000
        ),
        Err(Error::Budget)
    ));
    f.get_node_mut(root).payload = P::RegisterRead {
        register: "r".to_owned(),
    };
    assert!(matches!(
        BitsCone::compile(&f, root, limits(), &mut 1000),
        Err(Error::Effect)
    ));
}

#[test]
fn wide_constants_keep_their_numerical_magnitude() {
    for value in ["1", "0x100000000000000000000000000000000"] {
        let f = setup(&format!(
            "package p
          top fn f() -> bits[129] {{ ret wide: bits[129] = literal(value={value}, id=100) }}"
        ));
        let plan = BitsCone::compile(&f, f.ret_node_ref.unwrap(), limits(), &mut 1000).unwrap();
        assert_eq!(plan.input(), None);
        let actual = plan.evaluate(None).unwrap();
        assert_eq!(
            actual.as_bits().unwrap().to_u64().ok(),
            if value == "1" { Some(1) } else { None }
        );
    }
}

#[test]
fn bounded_products_honor_the_declared_width() {
    for op in ["umul", "smul"] {
        for width in [2, 5, 7] {
            let f = setup(&format!(
                "package p
                top fn f(x: bits[2] id=1) -> bits[{width}] {{
                    three: bits[3] = literal(value=3, id=2)
                    ret product: bits[{width}] = {op}(x, three, id=3)
                }}"
            ));
            let cone = BitsCone::compile(&f, f.ret_node_ref.unwrap(), limits(), &mut 1000).unwrap();
            for x in 0..4 {
                let arg = IrValue::make_ubits(2, x).unwrap();
                let operand = if op == "smul" && x >= 2 {
                    x as i64 - 4
                } else {
                    x as i64
                };
                let expected =
                    IrValue::make_ubits(width, (operand * 3) as u64 & ((1 << width) - 1)).unwrap();
                let actual = cone.evaluate(Some(&arg)).unwrap();
                assert_eq!(actual, expected, "{op}, width={width}, x={x}");
                let crate::ir_eval::FnEvalResult::Success(result) =
                    crate::ir_eval::eval_fn(&f, &[arg])
                else {
                    panic!("failed to evaluate product");
                };
                assert_eq!(result.value, expected);
            }
        }
    }
}
