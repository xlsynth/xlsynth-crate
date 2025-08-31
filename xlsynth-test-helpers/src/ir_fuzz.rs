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
    pub fn gen_with_same_signature<'a>(
        orig: &FuzzSample,
        u: &mut arbitrary::Unstructured<'a>,
    ) -> arbitrary::Result<Self> {
        let mut pkg =
            xlsynth::IrPackage::new("orig").map_err(|_| arbitrary::Error::IncorrectFormat)?;
        let ret_bits_usize: usize = generate_ir_fn(orig.input_bits, orig.ops.clone(), &mut pkg)
            .ok()
            .and_then(|f| {
                f.get_type()
                    .ok()
                    .map(|t| t.return_type().get_flat_bit_count() as usize)
            })
            .unwrap_or(orig.input_bits as usize);

        let mut sample = FuzzSample::arbitrary(u)?;
        sample.input_bits = orig.input_bits;

        for op in &mut sample.ops {
            if let FuzzOp::BitSlice { operand, start, width } = op {
                // Only clamp bit-slices that explicitly target the primary input (operand 0).
                // Randomly generated BitSlice ops in this fuzzer always slice the primary input.
                // BitSlices we append in gen_with_same_signature may target non-input nodes and
                // should not be clamped against input width.
                if operand.index == 0 {
                    if *start >= sample.input_bits {
                        *start = 0;
                    }
                    let max_width = sample.input_bits - *start;
                    if max_width == 0 {
                        *width = 1;
                    } else if *width > max_width {
                        *width = max_width;
                    }
                }
            }
        }

        let mut pkg_new =
            xlsynth::IrPackage::new("new").map_err(|_| arbitrary::Error::IncorrectFormat)?;
        let new_ret_bits_usize: usize = generate_ir_fn(sample.input_bits, sample.ops.clone(), &mut pkg_new)
            .ok()
            .and_then(|f| {
                f.get_type()
                    .ok()
                    .map(|t| t.return_type().get_flat_bit_count() as usize)
            })
            .unwrap_or(ret_bits_usize);

        if new_ret_bits_usize < ret_bits_usize {
            let idx = sample.ops.len() as u8;
            let ret_bits_u8 = std::convert::TryInto::<u8>::try_into(ret_bits_usize)
                .map_err(|_| arbitrary::Error::IncorrectFormat)?;
            sample.ops.push(FuzzOp::ZeroExt {
                operand: FuzzOperand { index: idx },
                new_bit_count: ret_bits_u8,
            });
        } else if new_ret_bits_usize > ret_bits_usize {
            let idx = sample.ops.len() as u8;
            let ret_bits_u8 = std::convert::TryInto::<u8>::try_into(ret_bits_usize)
                .map_err(|_| arbitrary::Error::IncorrectFormat)?;
            sample.ops.push(FuzzOp::BitSlice {
                operand: FuzzOperand { index: idx },
                start: 0,
                width: ret_bits_u8,
            });
        }

        Ok(sample)
    }
}

#[derive(Debug, Clone)]
pub struct FuzzSampleSameTypedPair {
    pub first: FuzzSample,
    pub second: FuzzSample,
}

impl<'a> arbitrary::Arbitrary<'a> for FuzzSampleSameTypedPair {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let first = FuzzSample::arbitrary(u)?;
        let second = FuzzSample::gen_with_same_signature(&first, u)?;
        Ok(FuzzSampleSameTypedPair { first, second })
    }
}
