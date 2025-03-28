// SPDX-License-Identifier: Apache-2.0

#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use xlsynth::{BValue, FnBuilder, IrFunction, IrType, XlsynthError};
use xlsynth_g8r::ir2gate::gatify;
use xlsynth_g8r::xls_ir::ir_parser;

#[derive(Debug, Arbitrary)]
enum FuzzUnop {
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

#[derive(Debug, Arbitrary)]
enum FuzzBinop {
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

    Concat,
}

#[derive(Debug, Clone, Arbitrary)]
struct FuzzOperand {
    index: u8,
}

#[derive(Debug, Arbitrary)]
enum FuzzOp {
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
}

/// Flattened opcode-only version of FuzzOp so we can ensure we select among all available ops when making an arbitrary op.
#[derive(Debug, Clone, Arbitrary)]
enum FuzzOpFlat {
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
    Decode,
    OneHotSel,
    UMul,
    SMul,
}

/// This function helps assure we have a FuzzOpFlat for each FuzzOp.
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
        FuzzOp::Decode { .. } => FuzzOpFlat::Decode,
        FuzzOp::OneHotSel { .. } => FuzzOpFlat::OneHotSel,
        FuzzOp::UMul { .. } => FuzzOpFlat::UMul,
        FuzzOp::SMul { .. } => FuzzOpFlat::SMul,
    }
}

// Generate a random IR function with only AND/NOT operations
fn generate_ir_fn(
    input_bits: u8,
    ops: Vec<FuzzOp>,
    package: &mut xlsynth::IrPackage,
) -> Result<IrFunction, XlsynthError> {
    let mut fn_builder = FnBuilder::new(package, "fuzz_test", true);

    // Add a single input parameter
    let input_type: IrType = package.get_bits_type(input_bits as u64);
    let input_node = fn_builder.param("input", &input_type);

    // Track all available nodes that can be used as operands
    let mut available_nodes = vec![input_node];

    // Process each operation
    for op in ops {
        match op {
            FuzzOp::Literal { bits, value } => {
                let ir_value = xlsynth::IrValue::make_ubits(bits as usize, value as u64)?;
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
                    FuzzUnop::Encode => fn_builder.encode(operand, None),
                };
                available_nodes.push(node);
            }
            FuzzOp::Binop(binop, idx1, idx2) => {
                let idx1 = (idx1 as usize) % available_nodes.len();
                let idx2 = (idx2 as usize) % available_nodes.len();
                let operand1 = &available_nodes[idx1];
                let operand2 = &available_nodes[idx2];
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

                    FuzzBinop::Concat => fn_builder.concat(&[operand1, operand2], None),
                };
                available_nodes.push(node);
            }
            FuzzOp::ZeroExt {
                operand,
                new_bit_count,
            } => {
                let operand = &available_nodes[(operand.index as usize) % available_nodes.len()];
                let node = fn_builder.zero_extend(operand, new_bit_count as u64, None);
                available_nodes.push(node);
            }
            FuzzOp::SignExt {
                operand,
                new_bit_count,
            } => {
                let operand = &available_nodes[(operand.index as usize) % available_nodes.len()];
                let node = fn_builder.sign_extend(operand, new_bit_count as u64, None);
                available_nodes.push(node);
            }
            FuzzOp::BitSlice {
                operand,
                start,
                width,
            } => {
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
            FuzzOp::Decode { arg, width } => {
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
        }
    }
    // Set the last node as the return value
    fn_builder.build_with_return_value(available_nodes.last().unwrap())
}

#[derive(Debug)]
struct FuzzSample {
    input_bits: u8,
    ops: Vec<FuzzOp>,
}

impl<'a> arbitrary::Arbitrary<'a> for FuzzSample {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let input_bits = u.int_in_range(1..=8)?;
        // Decide how many operations to generate.
        let num_ops = u.int_in_range(0..=20)?;
        // We always start with one available IR node: the input parameter
        let mut available_nodes = 1;
        let mut ops = Vec::with_capacity(num_ops as usize);

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
                    // BitSlice op: sample an index from [0, available_nodes)
                    let index = u.int_in_range(0..=(available_nodes as u64 - 1))? as u8;
                    let start = u.int_in_range(0..=8)?;
                    let width = u.int_in_range(1..=8)?;
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
            }
            // Each operation produces one new IR node
            available_nodes += 1;
        }
        Ok(FuzzSample { input_bits, ops })
    }
}

fuzz_target!(|sample: FuzzSample| {
    // Skip empty operation lists or empty input bits
    if sample.ops.is_empty() || sample.input_bits == 0 {
        return;
    }

    let _ = env_logger::builder().try_init();

    // Generate IR function from fuzz input
    let mut package = xlsynth::IrPackage::new("fuzz_test").unwrap();
    if let Err(e) = generate_ir_fn(sample.input_bits, sample.ops, &mut package) {
        log::info!("Error generating IR function: {}", e);
        return;
    }

    let parsed_package = match ir_parser::Parser::new(&package.to_string()).parse_package() {
        Ok(parsed_package) => parsed_package,
        Err(e) => {
            log::error!(
                "Error parsing IR package: {}\npackage:\n{}",
                e,
                package.to_string()
            );
            return;
        }
    };
    let parsed_fn = parsed_package.get_top().unwrap();

    // Convert to gates with folding disabled to make less machinery under test.
    let _gate_fn_no_fold = gatify(
        &parsed_fn,
        xlsynth_g8r::ir2gate::GatifyOptions {
            fold: false,
            check_equivalence: true,
        },
    );

    log::info!("unfolded conversion succeeded, attempting folded version...");

    // Now check the folded version is also equivalent.
    let _gate_fn_fold = gatify(
        &parsed_fn,
        xlsynth_g8r::ir2gate::GatifyOptions {
            fold: true,
            check_equivalence: true,
        },
    );

    // If we got here the equivalence checks passed.
});
