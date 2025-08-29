// SPDX-License-Identifier: Apache-2.0

use arbitrary::Arbitrary;
use xlsynth::{BValue, FnBuilder, IrFunction, IrType, XlsynthError};

#[derive(Debug, Arbitrary, Clone)]
pub enum FuzzUnop {
    // bitwise not (one's complement negation)
    Not,
    // two's complement negation
    Neg,

    OrReduce,
    AndReduce,
    XorReduce,

    Identity,
    Reverse,

    Encode,
}

#[derive(Debug, Arbitrary, Clone)]
pub enum FuzzBinop {
    // bitwise
    And,
    Nand,
    Nor,
    Or,
    Xor,

    // comparisons
    Eq,
    Ne,
    Ugt,
    Ule,
    Ult,
    Uge,
    Sge,
    Sgt,
    Slt,
    Sle,

    // shifts
    Shrl,
    Shra,
    Shll,

    // arithmetic
    Add,
    Sub,

    // division / modulus
    Udiv,
    Sdiv,
    Umod,
    Smod,

    Concat,
}

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzOperand {
    index: u8,
}

#[derive(Debug, Arbitrary, Clone)]
pub enum FuzzOp {
    Literal {
        bits: u8,
        value: u64,
    },
    Unop(FuzzUnop, u8),
    Binop(FuzzBinop, u8, u8),
    ZeroExt {
        operand: FuzzOperand,
        new_bit_count: u8,
    },
    SignExt {
        operand: FuzzOperand,
        new_bit_count: u8,
    },
    BitSlice {
        operand: FuzzOperand,
        start: u8,
        width: u8,
    },
    OneHot {
        arg: FuzzOperand,
        lsb_prio: bool,
    },
    Sel {
        selector: FuzzOperand,
        cases: Vec<FuzzOperand>,
        default: FuzzOperand,
    },
    OneHotSel {
        selector: FuzzOperand,
        cases: Vec<FuzzOperand>,
    },
    PrioritySel {
        selector: FuzzOperand,
        cases: Vec<FuzzOperand>,
        default: FuzzOperand,
    },
    Array {
        elements: Vec<FuzzOperand>,
    },
    ArrayIndex {
        array: FuzzOperand,
        index: FuzzOperand,
    },
    Decode {
        arg: FuzzOperand,
        width: u8,
    },
    UMul {
        lhs: FuzzOperand,
        rhs: FuzzOperand,
    },
    SMul {
        lhs: FuzzOperand,
        rhs: FuzzOperand,
    },
    Tuple {
        elements: Vec<FuzzOperand>,
    },
    TupleIndex {
        tuple: FuzzOperand,
        index: u8,
    },
    DynamicBitSlice {
        arg: FuzzOperand,
        start: FuzzOperand,
        width: u8,
    },
    BitSliceUpdate {
        value: FuzzOperand,
        start: FuzzOperand,
        update: FuzzOperand,
    },
}

#[derive(Debug, Clone, Arbitrary)]
pub enum FuzzOpFlat {
    Literal,
    Unop,
    Binop,
    ZeroExt,
    SignExt,
    BitSlice,
    OneHot,
    Sel,
    PrioritySel,
    ArrayIndex,
    Array,
    Decode,
    OneHotSel,
    UMul,
    SMul,
    Tuple,
    TupleIndex,
    DynamicBitSlice,
    BitSliceUpdate,
}

/// Flattened opcode-only version of FuzzOp so we can ensure we select among all
/// available ops when making an arbitrary op.
#[allow(dead_code)]
fn to_flat(op: &FuzzOp) -> FuzzOpFlat {
    match op {
        FuzzOp::Literal { .. } => FuzzOpFlat::Literal,
        FuzzOp::Unop { .. } => FuzzOpFlat::Unop,
        FuzzOp::Binop { .. } => FuzzOpFlat::Binop,
        FuzzOp::ZeroExt { .. } => FuzzOpFlat::ZeroExt,
        FuzzOp::SignExt { .. } => FuzzOpFlat::SignExt,
        FuzzOp::BitSlice { .. } => FuzzOpFlat::BitSlice,
        FuzzOp::OneHot { .. } => FuzzOpFlat::OneHot,
        FuzzOp::Sel { .. } => FuzzOpFlat::Sel,
        FuzzOp::PrioritySel { .. } => FuzzOpFlat::PrioritySel,
        FuzzOp::ArrayIndex { .. } => FuzzOpFlat::ArrayIndex,
        FuzzOp::Array { .. } => FuzzOpFlat::Array,
        FuzzOp::Decode { .. } => FuzzOpFlat::Decode,
        FuzzOp::OneHotSel { .. } => FuzzOpFlat::OneHotSel,
        FuzzOp::UMul { .. } => FuzzOpFlat::UMul,
        FuzzOp::SMul { .. } => FuzzOpFlat::SMul,
        FuzzOp::Tuple { .. } => FuzzOpFlat::Tuple,
        FuzzOp::TupleIndex { .. } => FuzzOpFlat::TupleIndex,
        FuzzOp::DynamicBitSlice { .. } => FuzzOpFlat::DynamicBitSlice,
        FuzzOp::BitSliceUpdate { .. } => FuzzOpFlat::BitSliceUpdate,
    }
}

// Generate a random IR function with only AND/NOT operations
pub fn generate_ir_fn(
    input_bits: u8,
    ops: Vec<FuzzOp>,
    package: &mut xlsynth::IrPackage,
) -> Result<IrFunction, XlsynthError> {
    assert!(input_bits > 0, "input_bits must be greater than 0");

    let mut fn_builder = FnBuilder::new(package, "fuzz_test", true);

    // Add a single input parameter
    let input_type: IrType = package.get_bits_type(input_bits as u64);
    let input_node = fn_builder.param("input", &input_type);

    // Track all available nodes that can be used as operands
    let mut available_nodes: Vec<BValue> = vec![input_node];

    // Process each operation
    for op in ops {
        fn_builder.last_value()?;
        match op {
            FuzzOp::Literal { bits, value } => {
                assert!(bits > 0, "literal op has no bits");
                let ir_value = xlsynth::IrValue::make_ubits(bits as usize, value)?;
                let node = fn_builder.literal(&ir_value, None);
                available_nodes.push(node);
            }
            FuzzOp::Unop(unop, idx) => {
                let idx = (idx as usize) % available_nodes.len();
                let operand = &available_nodes[idx];
                let node = match unop {
                    FuzzUnop::Not => fn_builder.not(operand, None),
                    FuzzUnop::Neg => fn_builder.neg(operand, None),
                    FuzzUnop::OrReduce => fn_builder.or_reduce(operand, None),
                    FuzzUnop::AndReduce => fn_builder.and_reduce(operand, None),
                    FuzzUnop::XorReduce => fn_builder.xor_reduce(operand, None),
                    FuzzUnop::Reverse => fn_builder.rev(operand, None),
                    FuzzUnop::Identity => fn_builder.identity(operand, None),
                    FuzzUnop::Encode => {
                        let bits_type: IrType = fn_builder.get_type(operand).unwrap();
                        let bit_count: u64 = bits_type.get_flat_bit_count();
                        // For one bit input, encode gives back a zero-bit result.
                        if bit_count <= 1 {
                            return Err(XlsynthError(
                                "encode needs more than 1 bit input operand".to_string(),
                            ));
                        }
                        fn_builder.encode(operand, None)
                    }
                };
                available_nodes.push(node);
            }
            FuzzOp::Binop(binop, idx1, idx2) => {
                let idx1 = (idx1 as usize) % available_nodes.len();
                let idx2 = (idx2 as usize) % available_nodes.len();
                let operand1: &BValue = &available_nodes[idx1];
                let operand2: &BValue = &available_nodes[idx2];
                let node = match binop {
                    FuzzBinop::Add => fn_builder.add(operand1, operand2, None),
                    FuzzBinop::Sub => fn_builder.sub(operand1, operand2, None),
                    FuzzBinop::And => fn_builder.and(operand1, operand2, None),
                    FuzzBinop::Nand => fn_builder.nand(operand1, operand2, None),
                    FuzzBinop::Nor => fn_builder.nor(operand1, operand2, None),
                    FuzzBinop::Or => fn_builder.or(operand1, operand2, None),
                    FuzzBinop::Xor => fn_builder.xor(operand1, operand2, None),

                    // comparisons
                    FuzzBinop::Eq => fn_builder.eq(operand1, operand2, None),
                    FuzzBinop::Ne => fn_builder.ne(operand1, operand2, None),
                    FuzzBinop::Ugt => fn_builder.ugt(operand1, operand2, None),
                    FuzzBinop::Ult => fn_builder.ult(operand1, operand2, None),
                    FuzzBinop::Uge => fn_builder.uge(operand1, operand2, None),
                    FuzzBinop::Ule => fn_builder.ule(operand1, operand2, None),
                    FuzzBinop::Sgt => fn_builder.sgt(operand1, operand2, None),
                    FuzzBinop::Sge => fn_builder.sge(operand1, operand2, None),
                    FuzzBinop::Slt => fn_builder.slt(operand1, operand2, None),
                    FuzzBinop::Sle => fn_builder.sle(operand1, operand2, None),

                    // shifts
                    FuzzBinop::Shrl => fn_builder.shrl(operand1, operand2, None),
                    FuzzBinop::Shra => fn_builder.shra(operand1, operand2, None),
                    FuzzBinop::Shll => fn_builder.shll(operand1, operand2, None),

                    // division / modulus
                    FuzzBinop::Udiv => fn_builder.udiv(operand1, operand2, None),
                    FuzzBinop::Sdiv => fn_builder.sdiv(operand1, operand2, None),
                    FuzzBinop::Umod => fn_builder.umod(operand1, operand2, None),
                    FuzzBinop::Smod => fn_builder.smod(operand1, operand2, None),

                    FuzzBinop::Concat => fn_builder.concat(&[operand1, operand2], None),
                };
                available_nodes.push(node);
            }
            FuzzOp::ZeroExt {
                operand,
                new_bit_count,
            } => {
                assert!(new_bit_count > 0, "zero extend has new bit count of 0");
                let operand = &available_nodes[(operand.index as usize) % available_nodes.len()];
                let operand_width =
                    fn_builder.get_type(operand).unwrap().get_flat_bit_count() as u8;
                let clamped_new_bit_count = std::cmp::max(new_bit_count, operand_width);
                let node = fn_builder.zero_extend(operand, clamped_new_bit_count as u64, None);
                available_nodes.push(node);
            }
            FuzzOp::SignExt {
                operand,
                new_bit_count,
            } => {
                assert!(new_bit_count > 0, "sign extend has new bit count of 0");
                let operand = &available_nodes[(operand.index as usize) % available_nodes.len()];
                let operand_width =
                    fn_builder.get_type(operand).unwrap().get_flat_bit_count() as u8;
                let clamped_new_bit_count = std::cmp::max(new_bit_count, operand_width);
                let node = fn_builder.sign_extend(operand, clamped_new_bit_count as u64, None);
                available_nodes.push(node);
            }
            FuzzOp::BitSlice {
                operand,
                start,
                width,
            } => {
                assert!(width > 0, "bit slice has no width");
                let operand_bv = &available_nodes[(operand.index as usize) % available_nodes.len()];
                let operand_width = fn_builder
                    .get_type(operand_bv)
                    .unwrap()
                    .get_flat_bit_count() as u8;
                assert!(
                    start + width <= operand_width,
                    "The bit-width of operand ({}) must be greater than or equal to start ({}) + width ({})",
                    operand_width,
                    start,
                    width
                );
                let operand = &available_nodes[(operand.index as usize) % available_nodes.len()];
                let node = fn_builder.bit_slice(operand, start as u64, width as u64, None);
                available_nodes.push(node);
            }
            FuzzOp::OneHot { arg, lsb_prio } => {
                let arg = &available_nodes[(arg.index as usize) % available_nodes.len()];
                let node = fn_builder.one_hot(arg, lsb_prio, None);
                available_nodes.push(node);
            }
            FuzzOp::Sel {
                selector,
                cases,
                default,
            } => {
                let selector = &available_nodes[(selector.index as usize) % available_nodes.len()];
                let cases = cases
                    .iter()
                    .map(|idx| &available_nodes[(idx.index as usize) % available_nodes.len()])
                    .collect::<Vec<_>>();
                let default = &available_nodes[(default.index as usize) % available_nodes.len()];
                let node = fn_builder.select(selector, cases.as_slice(), default, None);
                available_nodes.push(node);
            }
            FuzzOp::PrioritySel {
                selector,
                cases,
                default,
            } => {
                let selector: &BValue =
                    &available_nodes[(selector.index as usize) % available_nodes.len()];
                let cases: Vec<BValue> = cases
                    .iter()
                    .map(|idx| {
                        available_nodes[(idx.index as usize) % available_nodes.len()].clone()
                    })
                    .collect::<Vec<_>>();
                let default = &available_nodes[(default.index as usize) % available_nodes.len()];
                let node = fn_builder.priority_select(selector, cases.as_slice(), default, None);
                available_nodes.push(node);
            }
            FuzzOp::ArrayIndex { array, index } => {
                let array = &available_nodes[(array.index as usize) % available_nodes.len()];
                let index = &available_nodes[(index.index as usize) % available_nodes.len()];
                let node = fn_builder.array_index(array, index, None);
                available_nodes.push(node);
            }
            FuzzOp::Array { elements } => {
                let elements: Vec<BValue> = elements
                    .iter()
                    .map(|idx| {
                        available_nodes[(idx.index as usize) % available_nodes.len()].clone()
                    })
                    .collect::<Vec<_>>();
                let element_type: Option<IrType> = fn_builder.get_type(&elements[0]);
                if element_type.is_none() {
                    return Err(XlsynthError(
                        "no type available for array element BValue".to_string(),
                    ));
                }
                let element_type_ref: &IrType = element_type.as_ref().unwrap();
                let elements: &[BValue] = elements.as_slice();
                let elements_refs = elements.iter().map(|e: &BValue| e).collect::<Vec<_>>();
                let node = fn_builder.array(element_type_ref, &elements_refs, None);
                available_nodes.push(node);
            }
            FuzzOp::Tuple { elements } => {
                let tuple_elems: Vec<BValue> = elements
                    .iter()
                    .map(|idx| {
                        available_nodes[(idx.index as usize) % available_nodes.len()].clone()
                    })
                    .collect::<Vec<_>>();
                let refs: Vec<&BValue> = tuple_elems.iter().collect();
                let node = fn_builder.tuple(&refs, None);
                available_nodes.push(node);
            }
            FuzzOp::TupleIndex { tuple, index } => {
                let tuple_bv = &available_nodes[(tuple.index as usize) % available_nodes.len()];
                let node = fn_builder.tuple_index(tuple_bv, index as u64, None);
                available_nodes.push(node);
            }
            FuzzOp::Decode { arg, width } => {
                assert!(width > 0, "decode has width of 0");
                let arg = &available_nodes[(arg.index as usize) % available_nodes.len()];
                let node = fn_builder.decode(arg, Some(width as u64), None);
                available_nodes.push(node);
            }
            FuzzOp::OneHotSel { selector, cases } => {
                let selector = &available_nodes[(selector.index as usize) % available_nodes.len()];
                let cases: Vec<BValue> = cases
                    .iter()
                    .map(|idx| {
                        available_nodes[(idx.index as usize) % available_nodes.len()].clone()
                    })
                    .collect::<Vec<_>>();
                let node = fn_builder.one_hot_select(selector, cases.as_slice(), None);
                available_nodes.push(node);
            }
            FuzzOp::UMul { lhs, rhs } => {
                let lhs = &available_nodes[(lhs.index as usize) % available_nodes.len()];
                let rhs = &available_nodes[(rhs.index as usize) % available_nodes.len()];
                let node = fn_builder.umul(lhs, rhs, None);
                available_nodes.push(node);
            }
            FuzzOp::SMul { lhs, rhs } => {
                let lhs = &available_nodes[(lhs.index as usize) % available_nodes.len()];
                let rhs = &available_nodes[(rhs.index as usize) % available_nodes.len()];
                let node = fn_builder.smul(lhs, rhs, None);
                available_nodes.push(node);
            }
            FuzzOp::DynamicBitSlice { arg, start, width } => {
                assert!(width > 0, "dynamic bit slice has no width");
                let arg = &available_nodes[(arg.index as usize) % available_nodes.len()];
                let start = &available_nodes[(start.index as usize) % available_nodes.len()];
                let node = fn_builder.dynamic_bit_slice(arg, start, width as u64, None);
                available_nodes.push(node);
            }
            FuzzOp::BitSliceUpdate {
                value,
                start,
                update,
            } => {
                let value_bv = &available_nodes[(value.index as usize) % available_nodes.len()];
                let start_bv = &available_nodes[(start.index as usize) % available_nodes.len()];
                let update_bv = &available_nodes[(update.index as usize) % available_nodes.len()];
                let node = fn_builder.bit_slice_update(value_bv, start_bv, update_bv, None);
                available_nodes.push(node);
            }
        }
    }
    // Set the last node as the return value
    fn_builder.build_with_return_value(available_nodes.last().unwrap())
}

#[derive(Debug, Clone)]
pub struct FuzzSample {
    pub input_bits: u8,
    pub ops: Vec<FuzzOp>,
}

impl<'a> arbitrary::Arbitrary<'a> for FuzzSample {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let input_bits = u.int_in_range(1..=8)?;
        // Decide how many operations to generate.
        let num_ops = u.int_in_range(0..=20)?;
        // We maintain a parallel vector that records, for each existing IR node,
        // whether it is a tuple and how many elements it contains. `None` means
        // the node is *not* a tuple. This lets us generate in-bounds indices
        // for `TupleIndex` operations.
        let mut available_nodes = 1; // starts with the primary input (not a tuple)
        let mut tuple_sizes: Vec<Option<u8>> = vec![None];
        let mut ops = Vec::with_capacity(num_ops as usize);

        // TODO: In the long term we may want to carry an {IrOp: IrType} mapping so we
        // can ensure that we don't generate invalid IR.
        //
        // There has been multiple pull requests related to temporary fixes due to this
        // issue. See #428 and $432.

        for _ in 0..num_ops {
            // Randomly decide which kind of operation to generate
            let op_type = u.arbitrary::<FuzzOpFlat>()?;
            match op_type {
                FuzzOpFlat::Literal => {
                    // Literal op: nothing to sample, just generate a literal byte value.
                    let literal_bits = u.int_in_range(1..=8)?;
                    let value = u.int_in_range(0..=((1 << literal_bits) - 1))?;
                    ops.push(FuzzOp::Literal {
                        bits: literal_bits,
                        value,
                    });
                }
                FuzzOpFlat::Unop => {
                    // NOT op: sample an index from [0, available_nodes)
                    let idx = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let unop = u.arbitrary::<FuzzUnop>()?;
                    ops.push(FuzzOp::Unop(unop, idx));
                }
                FuzzOpFlat::Binop => {
                    // Binary op: sample two valid indices.
                    let idx1 = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let idx2 = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let binop = u.arbitrary::<FuzzBinop>()?;
                    ops.push(FuzzOp::Binop(binop, idx1, idx2));
                }
                FuzzOpFlat::ZeroExt => {
                    // ZeroExt op: sample an index from [0, available_nodes)
                    let index = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let new_bit_count = u.int_in_range(1..=8)?;
                    ops.push(FuzzOp::ZeroExt {
                        operand: FuzzOperand { index },
                        new_bit_count,
                    });
                }
                FuzzOpFlat::SignExt => {
                    // SignExt op: sample an index from [0, available_nodes)
                    let index = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let new_bit_count = u.int_in_range(1..=8)?;
                    ops.push(FuzzOp::SignExt {
                        operand: FuzzOperand { index },
                        new_bit_count,
                    });
                }
                FuzzOpFlat::BitSlice => {
                    // Generate an in-bounds BitSlice that always refers to the primary input (index
                    // 0) whose width is `input_bits`. This guarantees the slice
                    // is legal without having to track the widths of all
                    // intermediate IR nodes.
                    let index: u8 = 0; // primary input

                    // Choose a valid start such that 0 <= start < input_bits
                    let start = u.int_in_range(0..=((input_bits as u64) - 1))? as u8;

                    // Width must be at least 1 and satisfy start + width <= input_bits
                    let max_width = input_bits - start;
                    let width = u.int_in_range(1..=(max_width as u64))? as u8;

                    ops.push(FuzzOp::BitSlice {
                        operand: FuzzOperand { index },
                        start,
                        width,
                    });
                }
                FuzzOpFlat::OneHot => {
                    // OneHot op: sample an index from [0, available_nodes)
                    let index = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let lsb_prio = u.int_in_range(0..=1)? == 1;
                    ops.push(FuzzOp::OneHot {
                        arg: FuzzOperand { index },
                        lsb_prio,
                    });
                }
                FuzzOpFlat::PrioritySel => {
                    // PrioritySel op: sample an index from [0, available_nodes)
                    let index = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let num_cases = u.int_in_range(1..=8)?;
                    let mut cases: Vec<FuzzOperand> = Vec::with_capacity(num_cases as usize);
                    for _ in 0..num_cases {
                        cases.push(FuzzOperand {
                            index: u.int_in_range(0..=(available_nodes as u64 - 1))? as u8,
                        });
                    }
                    let last_case: FuzzOperand = cases.last().unwrap().clone();
                    ops.push(FuzzOp::PrioritySel {
                        selector: FuzzOperand { index },
                        cases,
                        default: last_case,
                    });
                }
                FuzzOpFlat::OneHotSel => {
                    // OneHotSel op: sample an index from [0, available_nodes)
                    let index = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let num_cases = u.int_in_range(1..=8)?;
                    let mut cases: Vec<FuzzOperand> = Vec::with_capacity(num_cases as usize);
                    for _ in 0..num_cases {
                        cases.push(FuzzOperand {
                            index: u.int_in_range(0..=(available_nodes as u64 - 1))? as u8,
                        });
                    }
                    ops.push(FuzzOp::OneHotSel {
                        selector: FuzzOperand { index },
                        cases,
                    });
                }
                FuzzOpFlat::Decode => {
                    // Decode op: sample an index from [0, available_nodes)
                    let index = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let width = u.int_in_range(1..=8)?;
                    ops.push(FuzzOp::Decode {
                        arg: FuzzOperand { index },
                        width,
                    });
                }
                FuzzOpFlat::ArrayIndex => {
                    // ArrayIndex op: sample an index from [0, available_nodes)
                    let index = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    ops.push(FuzzOp::ArrayIndex {
                        array: FuzzOperand { index },
                        index: FuzzOperand { index },
                    });
                }
                FuzzOpFlat::Array => {
                    // Array op: pick up to 8 elements to put in an array.
                    let num_elements = u.int_in_range(1..=8)?;
                    let mut elements: Vec<FuzzOperand> = Vec::with_capacity(num_elements as usize);
                    for _ in 0..num_elements {
                        elements.push(FuzzOperand {
                            index: u.int_in_range(0..=(available_nodes as u64 - 1))? as u8,
                        });
                    }
                    ops.push(FuzzOp::Array { elements });
                }
                FuzzOpFlat::Tuple => {
                    let num_elements = u.int_in_range(1..=8)?;
                    let mut elements: Vec<FuzzOperand> = Vec::with_capacity(num_elements as usize);
                    for _ in 0..num_elements {
                        elements.push(FuzzOperand {
                            index: u.int_in_range(0..=(available_nodes as u64 - 1))? as u8,
                        });
                    }
                    ops.push(FuzzOp::Tuple { elements });
                    // Record that the new node is a tuple with `num_elements` elements.
                    tuple_sizes.push(Some(num_elements as u8));
                }
                FuzzOpFlat::TupleIndex => {
                    // Choose only from nodes that are tuples so we can stay in-bounds.
                    let tuple_candidates: Vec<(usize, u8)> = tuple_sizes
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, sz)| sz.map(|s| (idx, s)))
                        .collect();

                    if !tuple_candidates.is_empty() {
                        // Pick a random tuple node.
                        let which =
                            u.int_in_range(0..=(tuple_candidates.len() as u64 - 1))? as usize;
                        let (tuple_idx, tuple_len) = tuple_candidates[which];
                        // Now pick an element index that is definitely in-bounds.
                        let index = if tuple_len == 1 {
                            0u8
                        } else {
                            u.int_in_range(0..=((tuple_len - 1) as u64))? as u8
                        };
                        ops.push(FuzzOp::TupleIndex {
                            tuple: FuzzOperand {
                                index: tuple_idx as u8,
                            },
                            index,
                        });
                    } else {
                        // No tuple exists yet – fall back to a no-op literal so we still
                        // generate the expected number of ops without causing an error.
                        let literal_bits = 1u8;
                        ops.push(FuzzOp::Literal {
                            bits: literal_bits,
                            value: 0,
                        });
                    }
                    // The result of TupleIndex is not a tuple.
                    tuple_sizes.push(None);
                }
                FuzzOpFlat::UMul => {
                    // UMul op: sample two valid indices.
                    let idx1 = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let idx2 = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    ops.push(FuzzOp::UMul {
                        lhs: FuzzOperand { index: idx1 },
                        rhs: FuzzOperand { index: idx2 },
                    });
                }
                FuzzOpFlat::SMul => {
                    // SMul op: sample two valid indices.
                    let idx1 = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let idx2 = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    ops.push(FuzzOp::SMul {
                        lhs: FuzzOperand { index: idx1 },
                        rhs: FuzzOperand { index: idx2 },
                    });
                }
                FuzzOpFlat::Sel => {
                    // Sel op: sample an index from [0, available_nodes)
                    let index = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let num_cases = u.int_in_range(1..=8)?;
                    let mut cases: Vec<FuzzOperand> = Vec::with_capacity(num_cases as usize);
                    for _ in 0..num_cases {
                        cases.push(FuzzOperand {
                            index: u.int_in_range(0..=(available_nodes as u64 - 1))? as u8,
                        });
                    }
                    let default = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    ops.push(FuzzOp::Sel {
                        selector: FuzzOperand { index },
                        cases,
                        default: FuzzOperand { index: default },
                    });
                }
                FuzzOpFlat::DynamicBitSlice => {
                    // DynamicBitSlice op: sample an index from [0, available_nodes)
                    let arg = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let start = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let width = u.int_in_range(1..=8)?;
                    ops.push(FuzzOp::DynamicBitSlice {
                        arg: FuzzOperand { index: arg },
                        start: FuzzOperand { index: start },
                        width,
                    });
                }
                FuzzOpFlat::BitSliceUpdate => {
                    // BitSliceUpdate op: sample three valid indices.
                    let value_idx = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let start_idx = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let update_idx = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    ops.push(FuzzOp::BitSliceUpdate {
                        value: FuzzOperand { index: value_idx },
                        start: FuzzOperand { index: start_idx },
                        update: FuzzOperand { index: update_idx },
                    });
                }
            }
            // For non-tuple ops that didn’t manually push into `tuple_sizes`, record None.
            if tuple_sizes.len() < (available_nodes + 1) as usize {
                tuple_sizes.push(None);
            }

            // Each operation produces one new IR node.
            available_nodes += 1;
        }
        Ok(FuzzSample { input_bits, ops })
    }
}

impl FuzzSample {
    /// Constructs a new sample that is otherwise arbitrary, but preserves the
    /// same function signature as `orig` when lowered with `generate_ir_fn`:
    ///
    /// - Same parameter count and names
    /// - Same parameter types
    /// - Same return type
    ///
    /// Note: Currently this function is specialized for bit-typed returns
    /// (which is what the localized ECO fuzzer uses). If the original sample's
    /// return type is non-bits, we conservatively mirror the input width as the
    /// return width so the generated IR remains valid; callers that require
    /// non-bits support can extend this routine.
    pub fn new_with_same_signature(orig: &FuzzSample) -> FuzzSample {
        // Discover the original return type by materializing a tiny package and
        // building the function, then introspecting its function type.
        let mut pkg = match xlsynth::IrPackage::new("sig_probe_pkg") {
            Ok(p) => p,
            Err(_) => {
                // In the unlikely event package construction fails, fall back to a
                // trivial identity with the same input bits; the fuzzer will skip if
                // it's unusable.
                return FuzzSample {
                    input_bits: orig.input_bits,
                    ops: vec![],
                };
            }
        };

        let (ret_is_bits, ret_bit_width) =
            match generate_ir_fn(orig.input_bits, orig.ops.clone(), &mut pkg) {
                Ok(func) => {
                    let fty = func.get_type().expect("get_type must succeed");
                    let ret_ty = fty.return_type();
                    let ret_str = ret_ty.to_string();
                    if let Some(width_str) = ret_str.strip_prefix("bits[") {
                        let width_str = width_str.trim_end_matches(']');
                        let width: u64 = width_str.parse().unwrap_or(orig.input_bits as u64);
                        (true, width as u8)
                    } else {
                        (false, orig.input_bits)
                    }
                }
                Err(_) => {
                    // If generating the original function failed, mirror input bits.
                    (true, orig.input_bits)
                }
            };

        // Build an "otherwise arbitrary" but simple body. To keep generation
        // robust and avoid type tracking, we only apply a few identity-style
        // unary ops to the primary input, then append a final coercion op to
        // ensure the return width matches exactly when returning bits.
        let mut ops: Vec<FuzzOp> = Vec::new();

        // Add a few no-op identities to vary the body deterministically based
        // on original ops count (keeps determinism without RNG deps).
        let noise_ops: usize = (orig.ops.len() % 3) + 1; // in 1..=3
        for _ in 0..noise_ops {
            ops.push(FuzzOp::Unop(FuzzUnop::Identity, 0));
        }

        if ret_is_bits {
            // If the desired return width is smaller than or equal to the input,
            // slice the primary input. Otherwise extend it to the requested width.
            if ret_bit_width <= orig.input_bits {
                ops.push(FuzzOp::BitSlice {
                    operand: FuzzOperand { index: 0 },
                    start: 0,
                    width: ret_bit_width,
                });
            } else {
                ops.push(FuzzOp::ZeroExt {
                    operand: FuzzOperand { index: 0 },
                    new_bit_count: ret_bit_width,
                });
            }
        } else {
            // Non-bits return types are not required by current fuzzers; return a
            // width-preserving identity of the input so the signature remains sane.
            ops.push(FuzzOp::Unop(FuzzUnop::Identity, 0));
        }

        FuzzSample {
            input_bits: orig.input_bits,
            ops,
        }
    }

    /// Constructs a sample that matches the given XLS IR function type.
    ///
    /// Constraints:
    /// - Exactly one parameter, of type bits[N].
    /// - Return type must be bits[M].
    ///
    /// Returns an error if constraints are not met.
    pub fn new_with_ir_function_type(
        fty: &xlsynth::ir_package::IrFunctionType,
    ) -> Result<FuzzSample, XlsynthError> {
        // Extract the single parameter width from the function type.
        if fty.param_count() != 1 {
            return Err(XlsynthError("expected exactly one parameter".to_string()));
        }
        let param_ty = fty.param_type(0)?;
        let param_ty_str = param_ty.to_string();
        let input_bits: u8 = if let Some(s) = param_ty_str.strip_prefix("bits[") {
            let s = s.trim_end_matches(']');
            s.parse::<u64>()
                .map_err(|_| XlsynthError("invalid param bits width".to_string()))?
                as u8
        } else {
            return Err(XlsynthError("parameter type must be bits[N]".to_string()));
        };

        // Extract return type width.
        let ret_ty = fty.return_type();
        let ret_ty_str = ret_ty.to_string();
        let ret_bits: u8 = if let Some(s) = ret_ty_str.strip_prefix("bits[") {
            let s = s.trim_end_matches(']');
            s.parse::<u64>()
                .map_err(|_| XlsynthError("invalid return bits width".to_string()))?
                as u8
        } else {
            return Err(XlsynthError("return type must be bits[M]".to_string()));
        };

        Ok(FuzzSample::new_with_param_and_ret_bits(
            input_bits, ret_bits,
        ))
    }

    /// Convenience: construct a sample from parameter and return bit widths.
    /// This mirrors the shape produced by `generate_ir_fn` (single input named
    /// "input").
    pub fn new_with_param_and_ret_bits(param_bits: u8, ret_bits: u8) -> FuzzSample {
        let mut ops: Vec<FuzzOp> = Vec::new();
        let noise_ops: usize = ((param_bits as usize) % 3) + 1;
        for _ in 0..noise_ops {
            ops.push(FuzzOp::Unop(FuzzUnop::Identity, 0));
        }
        if ret_bits <= param_bits {
            ops.push(FuzzOp::BitSlice {
                operand: FuzzOperand { index: 0 },
                start: 0,
                width: ret_bits,
            });
        } else {
            ops.push(FuzzOp::ZeroExt {
                operand: FuzzOperand { index: 0 },
                new_bit_count: ret_bits,
            });
        }
        FuzzSample {
            input_bits: param_bits,
            ops,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_same_signature_bits_roundtrip() {
        // Original sample: 5-bit input, return lower 3 bits via slice.
        let orig = FuzzSample {
            input_bits: 5,
            ops: vec![FuzzOp::BitSlice {
                operand: FuzzOperand { index: 0 },
                start: 0,
                width: 3,
            }],
        };

        let mut pkg_a = xlsynth::IrPackage::new("a").unwrap();
        let f_a = generate_ir_fn(orig.input_bits, orig.ops.clone(), &mut pkg_a).unwrap();
        let fty_a = f_a.get_type().unwrap();
        let param_count_a = fty_a.param_count();
        let param_ty_a = fty_a.param_type(0).unwrap().to_string();
        let ret_ty_a = fty_a.return_type().to_string();

        let new_sample = FuzzSample::new_with_same_signature(&orig);
        let mut pkg_b = xlsynth::IrPackage::new("b").unwrap();
        let f_b =
            generate_ir_fn(new_sample.input_bits, new_sample.ops.clone(), &mut pkg_b).unwrap();
        let fty_b = f_b.get_type().unwrap();

        // Same parameter count and type.
        assert_eq!(fty_b.param_count(), param_count_a);
        assert_eq!(fty_b.param_type(0).unwrap().to_string(), param_ty_a);

        // Same return type.
        assert_eq!(fty_b.return_type().to_string(), ret_ty_a);

        // Same parameter name.
        assert_eq!(f_a.param_name(0).unwrap(), "input".to_string());
        assert_eq!(f_b.param_name(0).unwrap(), "input".to_string());
    }

    #[test]
    fn test_new_with_ir_function_type_roundtrip() {
        // Build a function type via a real function, then mirror it.
        let mut pkg = xlsynth::IrPackage::new("make_fty").unwrap();
        let u6 = pkg.get_bits_type(6);
        let mut b = xlsynth::FnBuilder::new(&mut pkg, "f", true);
        let x = b.param("input", &u6);
        let sliced = b.bit_slice(&x, 0, 4, None);
        let f = b.build_with_return_value(&sliced).unwrap();
        let fty = f.get_type().unwrap();

        let sample = FuzzSample::new_with_ir_function_type(&fty).unwrap();
        let mut pkg2 = xlsynth::IrPackage::new("mirror").unwrap();
        let f2 = generate_ir_fn(sample.input_bits, sample.ops.clone(), &mut pkg2).unwrap();
        let fty2 = f2.get_type().unwrap();

        assert_eq!(fty2.param_count(), 1);
        assert_eq!(fty2.param_type(0).unwrap().to_string(), "bits[6]");
        assert_eq!(fty2.return_type().to_string(), "bits[4]");
        assert_eq!(f.param_name(0).unwrap(), "input");
        assert_eq!(f2.param_name(0).unwrap(), "input");
    }
}
