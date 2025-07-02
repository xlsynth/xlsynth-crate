use std::collections::HashMap;

use xlsynth::IrValue;

use crate::{
    equiv::solver_interface::{BitVec, Response, Solver},
    xls_ir::{
        ir::{self, NodePayload, NodeRef, Unop},
        ir_utils::get_topological,
    },
};

#[derive(Clone)]
pub struct IRTypedBitVec<'a, S: Solver> {
    pub ir_type: &'a ir::Type,
    pub bitvec: BitVec<S::Rep>,
}

impl<'a, S: Solver> std::fmt::Debug for IRTypedBitVec<'a, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IRTypedBitVec {{ ir_type: {:?}, bitvec: {:?} }}",
            self.ir_type, self.bitvec
        )
    }
}

impl<'a, S: Solver> IRTypedBitVec<'a, S> {
    pub fn ir_value_to_bv(solver: &mut S, ir_value: &IrValue, ir_type: &'a ir::Type) -> Self {
        fn gather_bits(v: &IrValue, ty: &ir::Type, bits: &mut Vec<bool>) {
            match ty {
                ir::Type::Bits(width) => {
                    let b = v.to_bits().expect("Expected bits literal");
                    assert_eq!(b.get_bit_count(), *width);
                    for i in 0..*width {
                        bits.push(b.get_bit(i).unwrap()); // LSB first
                    }
                }
                ir::Type::Array(arr) => {
                    let elems = v.get_elements().expect("Array literal elements");
                    assert_eq!(elems.len(), arr.element_count);
                    for elem in elems.iter() {
                        gather_bits(elem, &arr.element_type, bits);
                    }
                }
                ir::Type::Tuple(members) => {
                    let elems = v.get_elements().expect("Tuple literal elements");
                    assert_eq!(elems.len(), members.len());
                    for (elem, elem_ty) in elems.iter().rev().zip(members.iter().rev()) {
                        gather_bits(elem, elem_ty, bits);
                    }
                }
                ir::Type::Token => {
                    // Tokens are zero-width; nothing to gather.
                }
            }
        }
        let mut bits_vec: Vec<bool> = Vec::new();
        gather_bits(ir_value, ir_type, &mut bits_vec);
        let width = bits_vec.len();
        if width == 0 {
            return IRTypedBitVec {
                ir_type,
                bitvec: BitVec::ZeroWidth,
            };
        }
        let mut s = String::from("#b");
        // SMT bit-vector constant expects MSB first.
        for bit in bits_vec.iter().rev() {
            s.push(if *bit { '1' } else { '0' });
        }
        IRTypedBitVec {
            ir_type,
            bitvec: solver.from_raw_str(width, &s),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FnInputs<'a, S: Solver> {
    pub fn_ref: &'a ir::Fn,
    pub inputs: HashMap<String, IRTypedBitVec<'a, S>>,
}

impl<'a, S: Solver> FnInputs<'a, S> {
    pub fn new(solver: &mut S, fn_ref: &'a ir::Fn, name_prefix: Option<&str>) -> Self {
        let mut inputs = HashMap::new();
        for p in fn_ref.params.iter() {
            let name = match name_prefix {
                Some(prefix) => format!("__{}__{}", prefix, p.name),
                None => p.name.clone(),
            };
            let bv = solver.declare(&name, p.ty.bit_count() as usize).unwrap();
            inputs.insert(
                p.name.clone(),
                IRTypedBitVec {
                    ir_type: &p.ty,
                    bitvec: bv,
                },
            );
        }
        Self { fn_ref, inputs }
    }

    pub fn total_width(&self) -> usize {
        self.inputs.values().map(|b| b.bitvec.get_width()).sum()
    }
}

pub struct AlignedFnInputs<'a, S: Solver> {
    pub lhs: FnInputs<'a, S>,
    pub rhs: FnInputs<'a, S>,
    pub flattened: BitVec<S::Rep>,
}

impl<'a, S: Solver> AlignedFnInputs<'a, S> {
    pub fn align(
        solver: &mut S,
        lhs_inputs: &FnInputs<'a, S>,
        rhs_inputs: &FnInputs<'a, S>,
        allow_flatten: bool,
    ) -> Self {
        let lhs_inputs_total_width = lhs_inputs.total_width();
        let rhs_inputs_total_width = rhs_inputs.total_width();
        assert_eq!(
            lhs_inputs_total_width, rhs_inputs_total_width,
            "LHS and RHS must have the same number of bits"
        );
        if !allow_flatten {
            assert_eq!(
                lhs_inputs.fn_ref.params.len(),
                rhs_inputs.fn_ref.params.len(),
                "LHS and RHS must have the same number of inputs"
            );
            for (l, r) in lhs_inputs
                .fn_ref
                .params
                .iter()
                .zip(rhs_inputs.fn_ref.params.iter())
            {
                assert_eq!(
                    l.ty, r.ty,
                    "Input type mismatch for {} vs {}: {:?} vs {:?}",
                    l.name, r.name, l.ty, r.ty
                );
            }
        }
        if lhs_inputs_total_width == 0 {
            return Self {
                lhs: lhs_inputs.clone(),
                rhs: rhs_inputs.clone(),
                flattened: BitVec::ZeroWidth,
            };
        }
        let params_name = format!(
            "__flattened_params__{}__{}",
            lhs_inputs.fn_ref.name, rhs_inputs.fn_ref.name
        );
        let flattened = solver
            .declare(&params_name, lhs_inputs_total_width)
            .unwrap();
        // Split into individual param symbols
        let mut split_map = |inputs: &FnInputs<'a, S>| -> HashMap<String, IRTypedBitVec<'a, S>> {
            let mut m = HashMap::new();
            let mut offset = 0;
            for n in inputs.fn_ref.params.iter() {
                let existing_bitvec = inputs.inputs.get(&n.name).unwrap();
                let new_bitvec = {
                    let w = existing_bitvec.bitvec.get_width();
                    let h = (offset + w - 1) as i32;
                    let l = offset as i32;
                    offset += w;
                    let extracted = solver.extract(flattened.clone(), h, l);
                    let eq_expr = solver.eq(extracted.clone(), existing_bitvec.bitvec.clone());
                    solver.assert(eq_expr).unwrap();
                    extracted
                };
                m.insert(
                    n.name.clone(),
                    IRTypedBitVec {
                        ir_type: &n.ty,
                        bitvec: new_bitvec,
                    },
                );
            }
            m
        };
        let lhs_params = split_map(&lhs_inputs);
        let rhs_params = split_map(&rhs_inputs);

        Self {
            lhs: FnInputs {
                fn_ref: lhs_inputs.fn_ref,
                inputs: lhs_params,
            },
            rhs: FnInputs {
                fn_ref: rhs_inputs.fn_ref,
                inputs: rhs_params,
            },
            flattened,
        }
    }
}

pub struct SmtFn<'a, S: Solver> {
    pub fn_ref: &'a ir::Fn,
    pub inputs: Vec<IRTypedBitVec<'a, S>>,
    pub output: IRTypedBitVec<'a, S>,
}

impl<'a, S: Solver> SmtFn<'a, S> {
    pub fn ir_to_smt(solver: &mut S, inputs: &'a FnInputs<'a, S>) -> Self {
        let topo = get_topological(inputs.fn_ref);
        let mut env: HashMap<NodeRef, IRTypedBitVec<'a, S>> = HashMap::new();
        for nr in topo {
            let node = &inputs.fn_ref.nodes[nr.index];
            let exp: IRTypedBitVec<'a, S> = match &node.payload {
                NodePayload::Nil => continue,
                NodePayload::GetParam(pid) => {
                    let p = inputs.fn_ref.params.iter().find(|p| p.id == *pid).unwrap();
                    if let Some(sym) = inputs.inputs.get(&p.name) {
                        sym.clone()
                    } else {
                        panic!("Param not found: {}", p.name);
                    }
                }
                NodePayload::Unop(op, arg) => {
                    let a = env.get(&arg).unwrap().clone();
                    let rep = match op {
                        Unop::Not => solver.not(a.bitvec),
                        Unop::Neg => solver.neg(a.bitvec),
                        Unop::OrReduce => solver.or_reduce(a.bitvec),
                        Unop::AndReduce => solver.and_reduce(a.bitvec),
                        Unop::XorReduce => solver.xor_reduce(a.bitvec),
                        Unop::Identity => a.bitvec,
                        Unop::Reverse => solver.reverse(a.bitvec),
                    };
                    IRTypedBitVec {
                        ir_type: &node.ty,
                        bitvec: rep,
                    }
                }
                NodePayload::Literal(v) => IRTypedBitVec::ir_value_to_bv(solver, v, &node.ty),
                _ => panic!("Not implemented for {:?}", node.payload),
            };
            env.insert(nr, exp);
        }
        let ret = inputs.fn_ref.ret_node_ref.unwrap();
        SmtFn {
            fn_ref: inputs.fn_ref,
            inputs: inputs.inputs.values().cloned().collect(),
            output: env.remove(&ret).unwrap(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum EquivResult {
    Proved,
    Disproved {
        inputs: Vec<IrValue>,
        outputs: (IrValue, IrValue),
    },
}

fn check_aligned_fn_equiv_internal<'a, S: Solver>(
    solver: &mut S,
    lhs: &SmtFn<'a, S>,
    rhs: &SmtFn<'a, S>,
) -> EquivResult {
    let diff = solver.ne(lhs.output.bitvec.clone(), rhs.output.bitvec.clone());
    solver.assert(diff).unwrap();

    match solver.check().unwrap() {
        Response::Sat => {
            let mut get_value = |i: &IRTypedBitVec<'a, S>| -> IrValue {
                solver.get_value(i.bitvec.clone(), &i.ir_type).unwrap()
            };
            EquivResult::Disproved {
                inputs: lhs.inputs.iter().map(|i| get_value(i)).collect(),
                outputs: (get_value(&lhs.output), get_value(&rhs.output)),
            }
        }
        Response::Unsat => EquivResult::Proved,
        Response::Unknown => panic!("Solver returned unknown result"),
    }
}

pub fn prove_ir_fn_equiv<'a, S: Solver>(
    solver: &mut S,
    lhs: &ir::Fn,
    rhs: &ir::Fn,
    allow_flatten: bool,
) -> EquivResult {
    let fn_inputs_1 = FnInputs::new(solver, lhs, Some("lhs"));
    let fn_inputs_2 = FnInputs::new(solver, rhs, Some("rhs"));
    let aligned_fn_inputs =
        AlignedFnInputs::align(solver, &fn_inputs_1, &fn_inputs_2, allow_flatten);
    let smt_fn_1 = SmtFn::ir_to_smt(solver, &aligned_fn_inputs.lhs);
    let smt_fn_2 = SmtFn::ir_to_smt(solver, &aligned_fn_inputs.rhs);
    check_aligned_fn_equiv_internal(solver, &smt_fn_1, &smt_fn_2)
}

mod test_utils {
    #[macro_export]
    macro_rules! assert_smt_fn_eq {
        ($fn_name:ident, $solver:expr, $ir_text:expr, $expected:expr) => {
            #[test]
            fn $fn_name() {
                let mut parser = crate::xls_ir::ir_parser::Parser::new(&$ir_text);
                let f = parser.parse_fn().unwrap();
                let mut solver = $solver;
                let fn_inputs = FnInputs::new(&mut solver, &f, None);
                let smt_fn = SmtFn::ir_to_smt(&mut solver, &fn_inputs);
                let expected = $expected(&mut solver, &fn_inputs);
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    smt_fn.output.bitvec,
                    expected,
                );
            }
        };
    }

    #[macro_export]
    macro_rules! assert_ir_value_to_bv_eq {
        ($fn_name:ident, $solver:expr, $ir_value:expr, $ir_type:expr, $expected:expr) => {
            #[test]
            fn $fn_name() {
                let mut solver = $solver;
                let ir_value = $ir_value;
                let ir_type = $ir_type;
                let bv = IRTypedBitVec::ir_value_to_bv(&mut solver, &ir_value, &ir_type);
                assert_eq!(bv.ir_type, &ir_type);
                let expected = $expected(&mut solver);
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    bv.bitvec,
                    expected,
                );
            }
        };
    }

    #[macro_export]
    macro_rules! test_ir_fn_equiv {
        ($fn_name:ident, $solver:expr, $ir_text_1:expr, $ir_text_2:expr, $allow_flatten:expr, $result:pat) => {
            #[test]
            fn $fn_name() {
                let mut parser = crate::xls_ir::ir_parser::Parser::new($ir_text_1);
                let f1 = parser.parse_fn().unwrap();
                let mut parser = crate::xls_ir::ir_parser::Parser::new($ir_text_2);
                let f2 = parser.parse_fn().unwrap();
                let mut solver = $solver;
                let result = prove_ir_fn_equiv(&mut solver, &f1, &f2, $allow_flatten);
                assert!(matches!(result, $result));
            }
        };
    }

    #[macro_export]
    macro_rules! test_assert_fn_equiv_to_self {
        ($fn_name:ident, $solver:expr, $ir_text:expr) => {
            crate::test_ir_fn_equiv!(
                $fn_name,
                $solver,
                $ir_text,
                $ir_text,
                false,
                EquivResult::Proved
            );
        };
    }

    #[macro_export]
    macro_rules! test_assert_fn_inequiv {
        ($fn_name:ident, $solver:expr, $ir_text_1:expr, $ir_text_2:expr, $allow_flatten:expr) => {
            crate::test_ir_fn_equiv!(
                $fn_name,
                $solver,
                $ir_text_1,
                $ir_text_2,
                $allow_flatten,
                EquivResult::Disproved { .. }
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_value_bits {
        ($solver:expr, $solver_type:ty) => {
            crate::assert_ir_value_to_bv_eq!(
                test_ir_value_bits,
                $solver,
                IrValue::u32(0x12345678),
                ir::Type::Bits(32),
                |solver: &mut $solver_type| solver.from_raw_str(32, "#x12345678")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_bits {
        ($solver:expr, $solver_type:ty) => {
            crate::assert_smt_fn_eq!(
                test_ir_bits,
                $solver,
                r#"fn f() -> bits[32] {
                    ret literal.1: bits[32] = literal(value=0x12345678, id=1)
                }"#,
                |solver: &mut $solver_type, _: &FnInputs<$solver_type>| solver
                    .from_raw_str(32, "#x12345678")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_value_array {
        ($solver:expr, $solver_type:ty) => {
            crate::assert_ir_value_to_bv_eq!(
                test_ir_value_array,
                $solver,
                IrValue::make_array(&[
                    IrValue::make_array(&[
                        IrValue::make_ubits(8, 0x12).unwrap(),
                        IrValue::make_ubits(8, 0x34).unwrap(),
                    ])
                    .unwrap(),
                    IrValue::make_array(&[
                        IrValue::make_ubits(8, 0x56).unwrap(),
                        IrValue::make_ubits(8, 0x78).unwrap(),
                    ])
                    .unwrap(),
                    IrValue::make_array(&[
                        IrValue::make_ubits(8, 0x9a).unwrap(),
                        IrValue::make_ubits(8, 0xbc).unwrap(),
                    ])
                    .unwrap(),
                ])
                .unwrap(),
                ir::Type::Array(ir::ArrayTypeData {
                    element_type: Box::new(ir::Type::Array(ir::ArrayTypeData {
                        element_type: Box::new(ir::Type::Bits(8)),
                        element_count: 2,
                    })),
                    element_count: 3,
                }),
                |solver: &mut $solver_type| solver.from_raw_str(48, "#xbc9a78563412")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_array {
        ($solver:expr, $solver_type:ty) => {
            crate::assert_smt_fn_eq!(
                test_ir_array,
                $solver,
                r#"fn f() -> bits[8][2][3] {
                    ret literal.1: bits[8][2][3] = literal(value=[[0x12, 0x34], [0x56, 0x78], [0x9a, 0xbc]], id=1)
                }"#,
                |solver: &mut $solver_type, _: &FnInputs<$solver_type>| solver
                    .from_raw_str(48, "#xbc9a78563412")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_value_tuple {
        ($solver:expr, $solver_type:ty) => {
            crate::assert_ir_value_to_bv_eq!(
                test_ir_value_tuple,
                $solver,
                IrValue::make_tuple(&[
                    IrValue::make_ubits(8, 0x12).unwrap(),
                    IrValue::make_ubits(4, 0x4).unwrap(),
                ]),
                ir::Type::Tuple(vec![
                    Box::new(ir::Type::Bits(8)),
                    Box::new(ir::Type::Bits(4)),
                ]),
                |solver: &mut $solver_type| solver.from_raw_str(12, "#x124")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_tuple {
        ($solver:expr, $solver_type:ty) => {
            crate::assert_smt_fn_eq!(
                test_ir_tuple,
                $solver,
                r#"fn f() -> (bits[8], bits[4]) {
                   ret literal.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
                }"#,
                |solver: &mut $solver_type, _: &FnInputs<$solver_type>| solver
                    .from_raw_str(12, "#x124")
            );
        };
    }

    #[macro_export]
    macro_rules! test_ir_value_token {
        ($solver:expr, $solver_type:ty) => {
            crate::assert_ir_value_to_bv_eq!(
                test_ir_value_token,
                $solver,
                IrValue::make_ubits(0, 0).unwrap(),
                ir::Type::Token,
                |_: &mut $solver_type| BitVec::ZeroWidth
            );
        };
    }

    #[macro_export]
    macro_rules! test_unop_base {
        ($fn_name:ident, $solver:expr, $solver_type:ty, $unop_xls_name:expr, $unop:expr) => {
            crate::assert_smt_fn_eq!(
                $fn_name,
                $solver,
                r#"fn f(x: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = "#
                    .to_string()
                    + $unop_xls_name
                    + r#"(x, id=1)
                }"#,
                |solver: &mut $solver_type, inputs: &FnInputs<$solver_type>| {
                    $unop(solver, inputs.inputs.get("x").unwrap().bitvec.clone())
                }
            );
        };
    }

    #[macro_export]
    macro_rules! test_all_unops {
        ($solver:expr, $solver_type:ty) => {
            crate::test_unop_base!(test_not, $solver, $solver_type, "not", Solver::not);
            crate::test_unop_base!(test_neg, $solver, $solver_type, "neg", Solver::neg);
            crate::test_unop_base!(
                test_or_reduce,
                $solver,
                $solver_type,
                "or_reduce",
                Solver::or_reduce
            );
            crate::test_unop_base!(
                test_and_reduce,
                $solver,
                $solver_type,
                "and_reduce",
                Solver::and_reduce
            );
            crate::test_unop_base!(
                test_xor_reduce,
                $solver,
                $solver_type,
                "xor_reduce",
                Solver::xor_reduce
            );
            crate::test_unop_base!(test_identity, $solver, $solver_type, "identity", |_, a| a);
            crate::test_unop_base!(
                test_reverse,
                $solver,
                $solver_type,
                "reverse",
                Solver::reverse
            );
        };
    }

    #[macro_export]
    macro_rules! test_prove_fn_equiv {
        ($solver:expr) => {
            crate::test_assert_fn_equiv_to_self!(
                test_prove_fn_equiv,
                $solver,
                r#"fn f(x: bits[8], y: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = identity(x, id=1)
                }"#
            );
        };
    }

    #[macro_export]
    macro_rules! test_prove_fn_inequiv {
        ($solver:expr) => {
            crate::test_ir_fn_equiv!(
                test_prove_fn_inequiv,
                $solver,
                r#"fn f(x: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = identity(x, id=1)
                }"#,
                r#"fn g(x: bits[8]) -> bits[8] {
                    ret get_param.1: bits[8] = not(x, id=1)
                }"#,
                false,
                EquivResult::Disproved { .. }
            );
        };
    }
}

macro_rules! test_with_solver {
    ($mod_ident:ident, $solver:expr, $solver_type:ty) => {
        #[cfg(test)]
        mod $mod_ident {
            use super::*;

            crate::test_ir_value_bits!($solver, $solver_type);
            crate::test_ir_bits!($solver, $solver_type);
            crate::test_ir_value_array!($solver, $solver_type);
            crate::test_ir_array!($solver, $solver_type);
            crate::test_ir_value_tuple!($solver, $solver_type);
            crate::test_ir_tuple!($solver, $solver_type);
            crate::test_ir_value_token!($solver, $solver_type);
            crate::test_all_unops!($solver, $solver_type);
            crate::test_prove_fn_equiv!($solver);
            crate::test_prove_fn_inequiv!($solver);
        }
    };
}

#[cfg(feature = "with-bitwuzla-binary-test")]
test_with_solver!(
    bitwuzla_tests,
    crate::equiv::easy_smt_backend::EasySMTSolver::new(
        crate::equiv::easy_smt_backend::EasySMTConfig::bitwuzla()
    )
    .unwrap(),
    crate::equiv::easy_smt_backend::EasySMTSolver
);

#[cfg(feature = "with-boolector-binary-test")]
test_with_solver!(
    boolector_tests,
    crate::equiv::easy_smt_backend::EasySMTSolver::new(
        crate::equiv::easy_smt_backend::EasySMTConfig {
            replay_file: Some("boolector_replay.smt2".into()),
            ..crate::equiv::easy_smt_backend::EasySMTConfig::boolector()
        }
    )
    .unwrap(),
    crate::equiv::easy_smt_backend::EasySMTSolver
);

#[cfg(feature = "with-z3-binary-test")]
test_with_solver!(
    z3_tests,
    crate::equiv::easy_smt_backend::EasySMTSolver::new(
        crate::equiv::easy_smt_backend::EasySMTConfig::z3()
    )
    .unwrap(),
    crate::equiv::easy_smt_backend::EasySMTSolver
);
