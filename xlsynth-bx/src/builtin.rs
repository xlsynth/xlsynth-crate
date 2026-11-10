// SPDX-License-Identifier: Apache-2.0

//! Pure DSLX builtin operations shared by IR lowering and constexpr evaluation.

use xlsynth_pir::ir::{self, Type};
use xlsynth_pir::ir_builder::{BValue, Builder, BuilderError, BuilderState, FnBuilder};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_value_utils::ir_bits_to_usize_in_range;
use xlsynth_pir::{IrBits, IrValue};

/// An operation with all type arguments and constexpr controls resolved.
#[derive(Clone, Copy, Debug)]
pub enum BuiltinOp {
    Rev,
    Clz,
    Ctz,
    AndReduce,
    OrReduce,
    XorReduce,
    Encode,
    Decode {
        width: usize,
    },
    OneHot {
        lsb_priority: bool,
    },
    OneHotSel,
    PrioritySel,
    Gate,
    BitSliceUpdate,
    Signex {
        width: usize,
    },
    Umulp {
        width: usize,
    },
    Smulp {
        width: usize,
    },
    CheckedCast {
        source_signed: bool,
        target_signed: bool,
        width: usize,
    },
    ArrayRev,
    ArraySlice {
        width: usize,
    },
    /// Update at the tuple of indices in the second operand.
    ArrayUpdate {
        index_count: usize,
    },
    Zip,
}

impl BuiltinOp {
    /// Emit the hardware operation for a checked builtin in either builder.
    pub(crate) fn emit<S: BuilderState>(
        self,
        builder: &mut Builder<S>,
        args: &[BValue],
    ) -> Result<BValue, BuilderError> {
        match self {
            Self::Rev => builder.rev(args[0]),
            Self::Clz => builder.clz(args[0]),
            Self::Ctz => builder.ctz(args[0]),
            Self::AndReduce => builder.and_reduce(args[0]),
            Self::OrReduce => builder.or_reduce(args[0]),
            Self::XorReduce => builder.xor_reduce(args[0]),
            Self::Encode => builder.encode(args[0]),
            Self::Decode { width } => {
                let input_width = builder.get_type(args[0])?.bit_count();
                let decoded_width = if input_width < usize::BITS as usize {
                    (1usize << input_width).min(width)
                } else {
                    width
                };
                let decoded = builder.decode(args[0], Some(decoded_width))?;
                if decoded_width == width {
                    Ok(decoded)
                } else {
                    builder.zero_extend(decoded, width)
                }
            }
            Self::OneHot { lsb_priority } => builder.one_hot(args[0], lsb_priority),
            Self::OneHotSel | Self::PrioritySel => {
                let cases = array_elements(builder, args[1])?;
                if cases.is_empty() {
                    return if matches!(self, Self::OneHotSel) {
                        let width = array_type(builder, args[1])?.element_type.bit_count();
                        builder.literal(IrValue::Bits(IrBits::zero(width)))
                    } else {
                        Ok(args[2])
                    };
                }
                if matches!(self, Self::OneHotSel) {
                    builder.one_hot_select(args[0], &cases)
                } else {
                    builder.priority_select(args[0], &cases, args[2])
                }
            }
            Self::Gate => builder.gate(args[0], args[1]),
            Self::BitSliceUpdate => builder.bit_slice_update(args[0], args[1], args[2]),
            Self::Signex { width } => resize(builder, args[0], true, width),
            Self::Umulp { width } => builder.umulp(args[0], args[1], width),
            Self::Smulp { width } => builder.smulp(args[0], args[1], width),
            Self::CheckedCast {
                source_signed,
                width,
                ..
            } => resize(builder, args[0], source_signed, width),
            Self::ArrayRev => {
                let element = *array_type(builder, args[0])?.element_type;
                let mut values = array_elements(builder, args[0])?;
                values.reverse();
                builder.array(element, &values)
            }
            Self::ArraySlice { width: 0 } => {
                let element = *array_type(builder, args[0])?.element_type;
                builder.array(element, &[])
            }
            Self::ArraySlice { width } => builder.array_slice(args[0], args[1], width),
            Self::ArrayUpdate { index_count: 0 } => Ok(args[2]),
            Self::ArrayUpdate { index_count } => {
                let indices = (0..index_count)
                    .map(|i| builder.tuple_index(args[1], i))
                    .collect::<Result<Vec<_>, _>>()?;
                builder.array_update(args[0], args[2], &indices)
            }
            Self::Zip => {
                let lhs = array_type(builder, args[0])?;
                let rhs = array_type(builder, args[1])?;
                let element = Type::Tuple(vec![lhs.element_type, rhs.element_type]);
                let mut pairs = Vec::with_capacity(lhs.element_count);
                for i in 0..lhs.element_count {
                    let index = builder.literal(IrValue::u32(i as u32))?;
                    let a = builder.array_index(args[0], index)?;
                    let b = builder.array_index(args[1], index)?;
                    pairs.push(builder.tuple(&[a, b])?);
                }
                builder.array(element, &pairs)
            }
        }
    }

    /// Account for temporary IR used to evaluate builtins that expand arrays.
    pub(crate) fn evaluation_work(self, args: &[IrValue]) -> usize {
        let (array, nodes_per_element) = match self {
            Self::ArrayRev => (&args[0], 2),
            Self::Zip => (&args[0], 4),
            Self::OneHotSel | Self::PrioritySel => (&args[1], 2),
            _ => return 1,
        };
        array
            .get_element_count()
            .unwrap_or_default()
            .saturating_mul(nodes_per_element)
            .saturating_add(1)
    }

    /// Evaluate the same checked operations at compile time. DSLX constexpr
    /// update and cast reject inputs for which hardware still has a value.
    pub(crate) fn evaluate(self, args: &[IrValue]) -> Option<IrValue> {
        match self {
            Self::ArrayUpdate { index_count } => {
                let mut indexed = &args[0];
                for index in args[1].as_elements().ok()?.iter().take(index_count) {
                    let values = indexed.as_elements().ok()?;
                    let i = ir_bits_to_usize_in_range(index.as_bits().ok()?, values.len())?;
                    indexed = &values[i];
                }
            }
            Self::CheckedCast {
                source_signed,
                target_signed,
                width,
            } if !cast_fits(args[0].as_bits().ok()?, source_signed, target_signed, width) => {
                return None;
            }
            _ => { /* The hardware result is also the constexpr result. */ }
        }
        let mut builder = FnBuilder::new("bx_builtin");
        let operands = args
            .iter()
            .enumerate()
            .map(|(i, value)| builder.param(&format!("arg{i}"), value.type_()))
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let result = self.emit(&mut builder, &operands).ok()?;
        let function = builder.build(result).ok()?;
        match eval_fn(&function, args) {
            FnEvalResult::Success(result) => Some(result.value),
            FnEvalResult::Failure(_) => None,
        }
    }
}

/// Resize a bit pattern, choosing the extension independently of its BX type.
fn resize<S: BuilderState>(
    builder: &mut Builder<S>,
    arg: BValue,
    signed: bool,
    width: usize,
) -> Result<BValue, BuilderError> {
    let source_width = builder.get_type(arg)?.bit_count();
    if width == source_width {
        Ok(arg)
    } else if width < source_width {
        builder.bit_slice(arg, 0, width)
    } else if signed && source_width > 0 {
        builder.sign_extend(arg, width)
    } else {
        builder.zero_extend(arg, width)
    }
}

/// Read a checked array's IR layout for builtin expansion.
fn array_type<S: BuilderState>(
    builder: &Builder<S>,
    array: BValue,
) -> Result<ir::ArrayTypeData, BuilderError> {
    let Type::Array(ty) = builder.get_type(array)? else {
        return Err(BuilderError::InvalidOperation(
            "builtin requires an array".into(),
        ));
    };
    Ok(ty.clone())
}

/// Index each element once in declaration order.
fn array_elements<S: BuilderState>(
    builder: &mut Builder<S>,
    array: BValue,
) -> Result<Vec<BValue>, BuilderError> {
    (0..array_type(builder, array)?.element_count)
        .map(|i| {
            let index = builder.literal(IrValue::u32(i as u32))?;
            builder.array_index(array, index)
        })
        .collect()
}

/// Check representability without restricting either operand to machine width.
fn cast_fits(bits: &IrBits, source_signed: bool, target_signed: bool, width: usize) -> bool {
    let source_width = bits.get_bit_count();
    let negative = source_signed
        && source_width > 0
        && bits.get_bit(source_width - 1).expect("source sign bit");
    if negative && (!target_signed || width == 0) {
        return false;
    }
    let payload = width.saturating_sub(usize::from(target_signed));
    (payload..source_width).all(|i| bits.get_bit(i).expect("source bit") == negative)
}
