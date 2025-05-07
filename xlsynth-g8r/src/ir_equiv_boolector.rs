use crate::test_utils::{load_bf16_add_sample, Opt};
use crate::xls_ir::ir::{Fn, NodePayload, NodeRef, Param, ParamId};
use crate::xls_ir::ir_utils::get_topological;
use boolector::option::{BtorOption, ModelGen};
use boolector::{Btor, SolverResult, BV};
use std::collections::HashMap;
use std::rc::Rc;
use xlsynth::IrBits;

/// Converts an XLS IR function to Boolector bitvector logic.
/// Processes all nodes in topological order. Supports Literal and GetParam.
pub fn ir_fn_to_boolector(btor: Rc<Btor>, f: &Fn) -> BV<Rc<Btor>> {
    let topo = get_topological(f);
    let mut env: HashMap<NodeRef, BV<Rc<Btor>>> = HashMap::new();
    for node_ref in topo {
        let node = &f.nodes[node_ref.index];
        let bv = match &node.payload {
            NodePayload::Literal(ir_value) => {
                let bits = ir_value.to_bits().expect("Literal must be bits");
                let width = bits.get_bit_count() as u32;
                let mut value = 0u64;
                for i in 0..width {
                    if bits.get_bit(i as usize).unwrap() {
                        value |= 1 << i;
                    }
                }
                BV::from_u64(btor.clone(), value, width)
            }
            NodePayload::GetParam(param_id) => {
                // Find the parameter by id
                let param = f
                    .params
                    .iter()
                    .find(|p| p.id == *param_id)
                    .expect("Param not found");
                let width = param.ty.bit_count() as u32;
                BV::new(btor.clone(), width, Some(&param.name))
            }
            NodePayload::TupleIndex { tuple, index } => {
                // Get the tuple BV and type
                let tuple_bv = env.get(tuple).expect("Tuple operand must be present");
                let tuple_ty = f.get_node_ty(*tuple);
                // Get the bit range for the requested index
                let slice = tuple_ty
                    .tuple_get_flat_bit_slice_for_index(*index)
                    .expect("TupleIndex: not a tuple type");
                let width = slice.limit - slice.start;
                assert!(width > 0, "TupleIndex: width must be > 0");
                // Boolector's slice is inclusive: slice(high, low)
                let high = (slice.limit - 1) as u32;
                let low = slice.start as u32;
                tuple_bv.slice(high, low)
            }
            NodePayload::Nil => {
                // Do not insert a BV for Nil nodes
                continue;
            }
            NodePayload::Unop(op, arg) => {
                let arg_bv = env.get(arg).expect("Unop argument must be present");
                match op {
                    crate::xls_ir::ir::Unop::Not => arg_bv.not(),
                    crate::xls_ir::ir::Unop::Neg => arg_bv.neg(),
                    _ => panic!("Unop {:?} not yet implemented in Boolector conversion", op),
                }
            }
            NodePayload::Nary(op, elems) => match op {
                crate::xls_ir::ir::NaryOp::Concat => {
                    assert!(elems.len() >= 2, "Concat must have at least two operands");
                    let mut it = elems.iter();
                    let first = env
                        .get(it.next().unwrap())
                        .expect("Concat operand must be present")
                        .clone();
                    it.fold(first, |acc, nref| {
                        let next = env.get(nref).expect("Concat operand must be present");
                        acc.concat(next)
                    })
                }
                crate::xls_ir::ir::NaryOp::And => {
                    assert!(elems.len() >= 2, "And must have at least two operands");
                    let mut it = elems.iter();
                    let first = env
                        .get(it.next().unwrap())
                        .expect("And operand must be present")
                        .clone();
                    it.fold(first, |acc, nref| {
                        let next = env.get(nref).expect("And operand must be present");
                        acc.and(next)
                    })
                }
                crate::xls_ir::ir::NaryOp::Xor => {
                    assert!(elems.len() >= 2, "Xor must have at least two operands");
                    let mut it = elems.iter();
                    let first = env
                        .get(it.next().unwrap())
                        .expect("Xor operand must be present")
                        .clone();
                    it.fold(first, |acc, nref| {
                        let next = env.get(nref).expect("Xor operand must be present");
                        acc.xor(next)
                    })
                }
                _ => panic!(
                    "NaryOp {:?} not yet implemented in Boolector conversion",
                    op
                ),
            },
            NodePayload::Binop(op, a, b) => {
                let a_bv = env.get(a).expect("Binop lhs must be present");
                let b_bv = env.get(b).expect("Binop rhs must be present");
                use crate::xls_ir::ir::Binop;
                match op {
                    Binop::Add => a_bv.add(b_bv),
                    Binop::Sub => a_bv.sub(b_bv),
                    Binop::Eq => a_bv._eq(b_bv),
                    Binop::Ne => a_bv._ne(b_bv),
                    Binop::Uge => a_bv.ugte(b_bv),
                    Binop::Ugt => a_bv.ugt(b_bv),
                    Binop::Ult => a_bv.ult(b_bv),
                    Binop::Ule => a_bv.ulte(b_bv),
                    Binop::Umul | Binop::Smul => a_bv.mul(b_bv),
                    Binop::Shll => a_bv.sll(b_bv),
                    Binop::Shrl => a_bv.srl(b_bv),
                    Binop::Shra => a_bv.sra(b_bv),
                    _ => panic!("Binop {:?} not yet implemented in Boolector conversion", op),
                }
            }
            NodePayload::BitSlice { arg, start, width } => {
                let arg_bv = env.get(arg).expect("BitSlice argument must be present");
                assert!(*width > 0, "BitSlice width must be > 0");
                let high = (*start + *width - 1) as u32;
                let low = *start as u32;
                arg_bv.slice(high, low)
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                // Only support 2-way select for now
                assert!(
                    cases.len() == 2 && default.is_none(),
                    "Only 2-way select without default supported"
                );
                let sel_bv = env.get(selector).expect("Selector BV must be present");
                let case0 = env.get(&cases[0]).expect("Case 0 BV must be present");
                let case1 = env.get(&cases[1]).expect("Case 1 BV must be present");
                assert_eq!(
                    sel_bv.get_width(),
                    1,
                    "Selector BV must be 1 bit for 2-way select"
                );
                assert_eq!(
                    case0.get_width(),
                    case1.get_width(),
                    "Case widths must match"
                );
                sel_bv.cond_bv(case1, case0)
            }
            NodePayload::SignExt { arg, new_bit_count } => {
                let arg_bv = env.get(arg).expect("SignExt argument must be present");
                let from_width = arg_bv.get_width();
                let to_width = *new_bit_count as u32;
                assert!(
                    to_width > from_width,
                    "SignExt: new_bit_count must be greater than argument width"
                );
                arg_bv.sext(to_width - from_width)
            }
            other => todo!("Boolector conversion for {:?} not yet implemented", other),
        };
        // Assert the width matches the node's annotated type
        let expected_width = node.ty.bit_count() as u32;
        let actual_width = bv.get_width();
        assert_eq!(
            actual_width, expected_width,
            "Boolector BV width {} does not match node type width {} for node {:?}",
            actual_width, expected_width, node
        );
        env.insert(node_ref, bv);
    }
    let ret_node_ref = f.ret_node_ref.expect("Function must have a return node");
    if let Some(bv) = env.remove(&ret_node_ref) {
        bv
    } else {
        panic!("Cannot return Nil node from ir_fn_to_boolector: Boolector does not support 0-width bitvectors");
    }
}

/// Result of equivalence checking.
#[derive(Debug, PartialEq, Eq)]
pub enum EquivResult {
    Proved,
    Disproved(Vec<IrBits>), // Counterexample input
}

/// Checks equivalence of two IR functions using Boolector.
/// Only supports literal-only, zero-parameter functions for now.
pub fn check_equiv(lhs: &Fn, rhs: &Fn) -> EquivResult {
    // Only support zero-parameter, literal-only functions for now.
    assert!(
        lhs.params.is_empty() && rhs.params.is_empty(),
        "Only zero-parameter functions supported"
    );
    assert_eq!(lhs.ret_ty, rhs.ret_ty, "Return types must match");
    let btor = Rc::new(Btor::new());
    btor.set_opt(BtorOption::ModelGen(ModelGen::All));
    let lhs_bv = ir_fn_to_boolector(btor.clone(), lhs);
    let rhs_bv = ir_fn_to_boolector(btor.clone(), rhs);
    // Assert that outputs are not equal (look for a counterexample)
    let diff = lhs_bv._ne(&rhs_bv);
    diff.assert();
    match btor.sat() {
        SolverResult::Unsat => EquivResult::Proved,
        SolverResult::Sat => {
            // No inputs, so just return empty Vec for counterexample
            EquivResult::Disproved(vec![])
        }
        SolverResult::Unknown => panic!("Solver returned unknown result"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xls_ir::ir::{Fn, Node, NodePayload, Type};
    use xlsynth::IrValue;

    #[test]
    fn test_check_equiv_literals_proved() {
        // Both functions return literal 42
        let value = IrValue::make_ubits(8, 42).unwrap();
        let node = Node {
            text_id: 1,
            name: Some("literal.1".to_string()),
            ty: Type::Bits(8),
            payload: NodePayload::Literal(value),
        };
        let ir_fn = Fn {
            name: "f".to_string(),
            params: vec![],
            ret_ty: Type::Bits(8),
            nodes: vec![node],
            ret_node_ref: Some(crate::xls_ir::ir::NodeRef { index: 0 }),
        };
        let result = check_equiv(&ir_fn, &ir_fn);
        assert_eq!(result, EquivResult::Proved);
    }

    #[test]
    fn test_check_equiv_literals_disproved() {
        // Functions return different literals
        let value1 = IrValue::make_ubits(8, 42).unwrap();
        let node1 = Node {
            text_id: 1,
            name: Some("literal.1".to_string()),
            ty: Type::Bits(8),
            payload: NodePayload::Literal(value1),
        };
        let ir_fn1 = Fn {
            name: "f1".to_string(),
            params: vec![],
            ret_ty: Type::Bits(8),
            nodes: vec![node1],
            ret_node_ref: Some(crate::xls_ir::ir::NodeRef { index: 0 }),
        };
        let value2 = IrValue::make_ubits(8, 99).unwrap();
        let node2 = Node {
            text_id: 1,
            name: Some("literal.1".to_string()),
            ty: Type::Bits(8),
            payload: NodePayload::Literal(value2),
        };
        let ir_fn2 = Fn {
            name: "f2".to_string(),
            params: vec![],
            ret_ty: Type::Bits(8),
            nodes: vec![node2],
            ret_node_ref: Some(crate::xls_ir::ir::NodeRef { index: 0 }),
        };
        let result = check_equiv(&ir_fn1, &ir_fn2);
        assert_eq!(result, EquivResult::Disproved(vec![]));
    }

    #[test]
    fn test_bf16_add_sample_to_boolector() {
        let sample = load_bf16_add_sample(Opt::No);
        let g8r_fn = sample.g8r_pkg.get_fn(&sample.mangled_fn_name).unwrap();
        let btor = Rc::new(Btor::new());
        let bv = ir_fn_to_boolector(btor, g8r_fn);
        // bf16 is 16 bits
        assert_eq!(bv.get_width(), 16);
    }
}
