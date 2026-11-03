// SPDX-License-Identifier: Apache-2.0

//! Aggregate and builtin contracts with a bits-valued composition boundary.

use crate::aggregate_match::{self, AggregateMatch};
use crate::array_fragment::{ArrayCast, MatrixUpdate};
use crate::ast::{BitsType, Width};
use crate::constexpr::metadata_bits;
use crate::map_fragment::MapFragment;
use crate::mission_expr::{CheckedCast, LogicalPredicates, ReceiverMap};
use crate::range_match::{ConstMatch, RangeMatch, RangeValue};
use crate::select_fold::{ArraySelect, NestedFold};
use crate::sum_pattern::{self, SumComparison, SumPattern, SumShape};
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue, ir};

/// A builtin whose result is converted back to the surrounding bits type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinOp {
    Reverse,
    Clz,
    Ctz,
    AndReduce,
    OrReduce,
    XorReduce,
    OneHotLsb,
    OneHotMsb,
    Decode,
    Encode,
    PrioritySelect,
    OneHotSelect,
    Gate,
    BitCount,
    ArraySize,
    Zero,
    AllOnes,
    CheckedCast,
    WideningCast,
    BitSliceUpdate,
    Signex,
    ElementCount,
}

impl BuiltinOp {
    pub const ALL: [Self; 22] = [
        Self::Reverse,
        Self::Clz,
        Self::Ctz,
        Self::AndReduce,
        Self::OrReduce,
        Self::XorReduce,
        Self::OneHotLsb,
        Self::OneHotMsb,
        Self::Decode,
        Self::Encode,
        Self::PrioritySelect,
        Self::OneHotSelect,
        Self::Gate,
        Self::BitCount,
        Self::ArraySize,
        Self::Zero,
        Self::AllOnes,
        Self::CheckedCast,
        Self::WideningCast,
        Self::BitSliceUpdate,
        Self::Signex,
        Self::ElementCount,
    ];
}

/// An array computation with dynamically bounded indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayOp {
    Index,
    Update,
    Slice,
    Concat,
    Reverse,
    Enumerate,
    LocalEnumerate,
    Zip,
    Repeat,
}

impl ArrayOp {
    /// The eight distinct computations drawn by the decoder. Both
    /// enumerate implementations occupy the same production slot.
    pub const ALL: [Self; 8] = [
        Self::Index,
        Self::Update,
        Self::Slice,
        Self::Concat,
        Self::Reverse,
        Self::Enumerate,
        Self::Zip,
        Self::Repeat,
    ];
}

/// Exhaustive patterns over bits, tuples, or a small nominal enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternOp {
    Alternatives,
    Range,
    Tuple,
    Enum,
}

impl PatternOp {
    pub const ALL: [Self; 4] = [Self::Alternatives, Self::Range, Self::Tuple, Self::Enum];
}

/// A bounded loop over a range or an array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    Range,
    Array,
    ConstRange,
}

/// A compound expression consuming three bits values and returning one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundOp {
    Builtin(BuiltinOp),
    Array(ArrayOp),
    ArrayCast(ArrayCast),
    MatrixUpdate(MatrixUpdate),
    RangeMatch(RangeMatch),
    Map(MapFragment),
    Sum(SumPattern),
    SumComparison(SumComparison),
    ArraySelect(ArraySelect),
    AggregateMatch(AggregateMatch),
    NestedFold(NestedFold),
    ReceiverMap(ReceiverMap),
    LogicalPredicates(LogicalPredicates),
    CheckedCast(CheckedCast),
    ConstMatch(ConstMatch),
    RangeValue(RangeValue),
    RangeLoop {
        range: RangeValue,
        constant: bool,
    },
    Pattern(PatternOp),
    Record {
        update: bool,
    },
    Loop {
        kind: LoopKind,
        start: usize,
        trips: usize,
    },
}

/// Module declarations required by the recursively composed expression bodies.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Declarations {
    record: bool,
    choice: bool,
    sum: bool,
    nested_sum: bool,
    aggregate_match: bool,
    receiver_maps: [bool; ReceiverMap::WIDTHS.len()],
    std: bool,
    local_is_even: bool,
    local_enumerate: bool,
}

impl Declarations {
    pub(crate) fn extend(&mut self, other: Self) {
        self.record |= other.record;
        self.choice |= other.choice;
        self.sum |= other.sum;
        self.nested_sum |= other.nested_sum;
        self.aggregate_match |= other.aggregate_match;
        for (used, additional) in self.receiver_maps.iter_mut().zip(other.receiver_maps) {
            *used |= additional;
        }
        self.std |= other.std;
        self.local_is_even |= other.local_is_even;
        self.local_enumerate |= other.local_enumerate;
    }

    /// Prints nominal types only when an expression in this module uses them.
    pub(crate) fn to_dslx(self) -> String {
        let mut text = String::new();
        if self.std {
            text.push_str("import std;\n");
        }
        if self.local_is_even {
            text.push_str(
                "\nfn fuzz_is_even<S: bool, N: u32>(x: xN[S][N]) -> bool { (x as u1) == u1:0 }\n",
            );
        }
        if self.local_enumerate {
            text.push_str(
                r#"
fn fuzz_enumerate<S: bool, W: u32, N: u32>(xs: xN[S][W][N]) -> (u32, xN[S][W])[N] {
    for (i, result) in u32:0..N {
        update(result, i, (i, xs[i]))
    }(zero!<(u32, xN[S][W])[N]>())
}
"#,
            );
        }
        if self.record {
            text.push_str("\npub struct FuzzPair<N: u32> { first: uN[N], second: uN[N] }\n");
        }
        if self.choice {
            text.push_str("\npub enum FuzzChoice: u2 { First = 0, Second = 1, Third = 2 }\n");
        }
        if self.sum {
            text.push_str(sum_pattern::DECLARATION);
        }
        if self.nested_sum {
            text.push_str(sum_pattern::NESTED_DECLARATION);
        }
        if self.aggregate_match {
            text.push_str(aggregate_match::DECLARATION);
        }
        for (used, width) in self.receiver_maps.into_iter().zip(ReceiverMap::WIDTHS) {
            if used {
                text.push_str(&ReceiverMap::declaration(width));
            }
        }
        text
    }
}

impl CompoundOp {
    /// Checks bounds and variant choices before rendering either
    /// representation.
    pub(crate) fn validate(self) -> Result<(), String> {
        if matches!(self, Self::RangeValue(range) if range.len() == 0) {
            return Err("indexed ranges must be nonempty".to_string());
        }
        if matches!(self, Self::RangeLoop { range, .. } if range.len() > 4) {
            return Err("range loops require at most four trips".to_string());
        }
        if let Self::Loop { kind, start, trips } = self
            && (trips > 4 || start > 4 || (kind == LoopKind::Array && trips == 0))
        {
            return Err("loop contracts require at most four trips and a start in 0..=4; array loops must be nonempty".to_string());
        }
        if let Self::Sum(pattern) = self {
            return pattern.validate();
        }
        if let Self::SumComparison(comparison) = self {
            return comparison.validate();
        }
        Ok(())
    }

    pub(crate) fn declarations(self) -> Declarations {
        Declarations {
            record: matches!(self, Self::Record { .. } | Self::AggregateMatch(_)),
            choice: self == Self::Pattern(PatternOp::Enum),
            sum: matches!(
                self,
                Self::Sum(_) | Self::SumComparison(_) | Self::AggregateMatch(_)
            ),
            aggregate_match: matches!(self, Self::AggregateMatch(_)),
            receiver_maps: ReceiverMap::WIDTHS
                .map(|width| matches!(self, Self::ReceiverMap(map) if map.key_width() == width)),
            nested_sum: matches!(self, Self::SumComparison(comparison) if comparison.shape == SumShape::Nested),
            std: self == Self::Array(ArrayOp::Enumerate)
                || matches!(self, Self::Map(map) if map.needs_std()),
            local_is_even: matches!(self, Self::Map(map) if map.needs_local_is_even()),
            local_enumerate: self == Self::Array(ArrayOp::LocalEnumerate),
        }
    }

    /// Formats a scoped expression; local aggregate names cannot escape it.
    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let [x, y, z] = args;
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let result = ty.to_dslx();
        let ux = format!("({x} as {unsigned})");
        let selector = format!("({ux} as u2)");
        match self {
            Self::ArrayCast(cast) => cast.to_dslx(ty, args),
            Self::MatrixUpdate(update) => update.to_dslx(ty, args),
            Self::RangeMatch(pattern) => pattern.to_dslx(args),
            Self::Map(map) => map.to_dslx(ty, args),
            Self::Sum(pattern) => pattern.to_dslx(ty, args),
            Self::SumComparison(comparison) => comparison.to_dslx(ty, args),
            Self::ArraySelect(select) => select.to_dslx(ty, args),
            Self::AggregateMatch(pattern) => pattern.to_dslx(ty, args),
            Self::NestedFold(fold) => fold.to_dslx(ty, args),
            Self::ReceiverMap(map) => map.to_dslx(ty, args),
            Self::LogicalPredicates(predicates) => predicates.to_dslx(args),
            Self::CheckedCast(cast) => cast.to_dslx(ty, args),
            Self::ConstMatch(pattern) => pattern.to_dslx(ty, args),
            Self::RangeValue(range) => range.to_dslx(ty, args),
            Self::RangeLoop { range, constant } => range.loop_dslx(constant, ty, args),
            Self::Builtin(op) => {
                let wide = match ty.width {
                    Width::Concrete(width) => {
                        format!("{}N[{}]", if ty.signed { "s" } else { "u" }, width + 1)
                    }
                    Width::Parametric => {
                        format!("{}N[N + u32:1]", if ty.signed { "s" } else { "u" })
                    }
                };
                let expression = match op {
                    BuiltinOp::Reverse => format!("rev({ux})"),
                    BuiltinOp::Clz => format!("clz({ux})"),
                    BuiltinOp::Ctz => format!("ctz({ux})"),
                    BuiltinOp::AndReduce => format!("and_reduce({ux})"),
                    BuiltinOp::OrReduce => format!("or_reduce({ux})"),
                    BuiltinOp::XorReduce => format!("xor_reduce({ux})"),
                    BuiltinOp::OneHotLsb => format!("one_hot({ux}, true)"),
                    BuiltinOp::OneHotMsb => format!("one_hot({ux}, false)"),
                    BuiltinOp::Decode => format!("decode<{unsigned}>(({ux} as u8))"),
                    BuiltinOp::Encode => format!("encode({ux})"),
                    BuiltinOp::PrioritySelect => {
                        format!("priority_sel({selector}, [{y}, {z}], {x})")
                    }
                    BuiltinOp::OneHotSelect => format!("one_hot_sel({selector}, [{y}, {z}])"),
                    BuiltinOp::Gate => format!("gate!({selector} != u2:0, {y})"),
                    BuiltinOp::BitCount => format!("(bit_count<{result}>() as {result}) ^ {x}"),
                    BuiltinOp::ArraySize => {
                        format!("(array_size([{x}, {y}, {z}]) as {result}) + {x}")
                    }
                    BuiltinOp::Zero => format!("{x} + zero!<{result}>()"),
                    BuiltinOp::AllOnes => format!("{x} ^ all_ones!<{result}>()"),
                    BuiltinOp::CheckedCast => format!("checked_cast<{result}>(({x} as {wide}))"),
                    BuiltinOp::WideningCast => format!("widening_cast<{wide}>({x}) >> u1:1"),
                    BuiltinOp::BitSliceUpdate => format!(
                        "bit_slice_update({ux}, (({y} as {unsigned}) as u2), ({z} as {unsigned}))"
                    ),
                    BuiltinOp::Signex => {
                        let width = match ty.width {
                            Width::Concrete(width) => width.to_string(),
                            Width::Parametric => "N".to_string(),
                        };
                        format!("signex({ux}, zero!<uN[{width} + u32:3]>()) >> u2:2")
                    }
                    BuiltinOp::ElementCount => {
                        format!("(element_count<{result}[3]>() as {result}) ^ {x}")
                    }
                };
                format!("(({expression}) as {result})")
            }
            Self::Array(op) => {
                let index = format!("(({ux} as u32) % u32:3)");
                let expression = match op {
                    ArrayOp::Index => "a[i]".to_string(),
                    ArrayOp::Update => format!("update(a, i, {z})[u32:1]"),
                    ArrayOp::Slice => format!("array_slice(a, i, [{x}, {y}])[u32:1]"),
                    ArrayOp::Concat => format!("([{x}, {y}] ++ [{z}])[i]"),
                    ArrayOp::Reverse => "array_rev(a)[i]".to_string(),
                    ArrayOp::Enumerate | ArrayOp::LocalEnumerate => {
                        let name = if op == ArrayOp::LocalEnumerate {
                            "fuzz_enumerate"
                        } else {
                            "std::enumerate"
                        };
                        format!("{{ let (j, element) = {name}(a)[i]; element ^ (j as {result}) }}")
                    }
                    ArrayOp::Zip => format!(
                        "{{ let (left, right) = zip(a, [{z}, {x}, {y}])[i]; left + right }}"
                    ),
                    ArrayOp::Repeat => {
                        format!("{{ let repeated: {result}[3] = [{x}, {y}, ...]; repeated[i] }}")
                    }
                };
                format!("{{ let a = [{x}, {y}, {z}]; let i = {index}; {expression} }}")
            }
            Self::Pattern(op) => match op {
                PatternOp::Alternatives => {
                    format!("match {selector} {{ u2:0 | u2:2 => {y}, u2:1 => {x}, _ => {z}, }}")
                }
                PatternOp::Range => format!("match {selector} {{ u2:0..u2:2 => {y}, _ => {z}, }}"),
                PatternOp::Tuple => format!(
                    "match ({selector}, ({y}, {z})) {{ (u2:0, (a, b)) => a + b, (_, (_, b)) => b, }}"
                ),
                PatternOp::Enum => format!(
                    "{{ let tag = if {selector} == u2:0 {{ FuzzChoice::First }} else if {selector} == u2:1 {{ FuzzChoice::Second }} else {{ FuzzChoice::Third }}; match tag {{ FuzzChoice::First => {x}, FuzzChoice::Second => {y}, FuzzChoice::Third => {z}, }} }}"
                ),
            },
            Self::Record { update } => {
                let update = if update {
                    format!("let p = FuzzPair {{ second: ({z} as {unsigned}), ..p }};")
                } else {
                    String::new()
                };
                format!(
                    "{{ let p = FuzzPair {{ first: {ux}, second: ({y} as {unsigned}) }}; {update} ((p.first + p.second) as {result}) }}"
                )
            }
            Self::Loop { kind, start, trips } => {
                if kind == LoopKind::Array {
                    let elements = (0..trips)
                        .map(|index| if index % 2 == 0 { y } else { z })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!(
                        "for (element, acc): ({result}, {result}) in [{elements}] {{ (acc + element) ^ {z} }}({x})"
                    )
                } else {
                    RangeValue::unsigned_loop(start, trips).loop_dslx(
                        kind == LoopKind::ConstRange,
                        ty,
                        args,
                    )
                }
            }
        }
    }

    /// Constructs the reference from primitive PIR nodes, unrolling loops.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        width: usize,
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        let [x, y, z] = args;
        match self {
            Self::ArrayCast(cast) => cast.emit_ir(builder, args),
            Self::MatrixUpdate(update) => update.emit_ir(builder, args),
            Self::RangeMatch(pattern) => pattern.emit_ir(builder, args, width, signed),
            Self::Map(map) => map.emit_ir(builder, args, width, signed),
            Self::Sum(pattern) => pattern.emit_ir(builder, args, width),
            Self::SumComparison(comparison) => comparison.emit_ir(builder, args, width),
            Self::ArraySelect(select) => select.emit_ir(builder, args, width),
            Self::AggregateMatch(pattern) => pattern.emit_ir(builder, args, width),
            Self::NestedFold(fold) => fold.emit_ir(builder, args, signed),
            Self::ReceiverMap(map) => map.emit_ir(builder, args, width),
            Self::LogicalPredicates(predicates) => predicates.emit_ir(builder, args, signed),
            Self::CheckedCast(cast) => cast.emit_ir(builder, args, signed),
            Self::ConstMatch(pattern) => pattern.emit_ir(builder, args, width),
            Self::RangeValue(range) => range.emit_ir(builder, args, width),
            Self::RangeLoop { range, .. } => range.emit_loop_ir(builder, args, width),
            Self::Builtin(op) => {
                let value = match op {
                    BuiltinOp::Reverse => builder.rev(x)?,
                    BuiltinOp::Clz => builder.clz(x)?,
                    BuiltinOp::Ctz => builder.ctz(x)?,
                    BuiltinOp::AndReduce => builder.and_reduce(x)?,
                    BuiltinOp::OrReduce => builder.or_reduce(x)?,
                    BuiltinOp::XorReduce => builder.xor_reduce(x)?,
                    BuiltinOp::OneHotLsb => builder.one_hot(x, true)?,
                    BuiltinOp::OneHotMsb => builder.one_hot(x, false)?,
                    BuiltinOp::Decode => {
                        let index = resize_unsigned(builder, x, 8)?;
                        builder.decode(index, Some(width))?
                    }
                    BuiltinOp::Encode => builder.encode(x)?,
                    BuiltinOp::PrioritySelect => {
                        let selector = resize_unsigned(builder, x, 2)?;
                        builder.priority_select(selector, &[y, z], x)?
                    }
                    BuiltinOp::OneHotSelect => {
                        let selector = resize_unsigned(builder, x, 2)?;
                        builder.one_hot_select(selector, &[y, z])?
                    }
                    BuiltinOp::Gate => {
                        let selector = resize_unsigned(builder, x, 2)?;
                        let condition = builder.or_reduce(selector)?;
                        builder.gate(condition, y)?
                    }
                    BuiltinOp::BitCount | BuiltinOp::ArraySize | BuiltinOp::ElementCount => {
                        let amount = metadata_literal(
                            builder,
                            if op == BuiltinOp::BitCount { width } else { 3 },
                        )?;
                        let amount = resize_unsigned(builder, amount, width)?;
                        if op != BuiltinOp::ArraySize {
                            builder.xor(x, amount)?
                        } else {
                            builder.add(x, amount)?
                        }
                    }
                    BuiltinOp::Zero | BuiltinOp::CheckedCast => x,
                    BuiltinOp::AllOnes => builder.not(x)?,
                    BuiltinOp::WideningCast => {
                        let wide = if signed {
                            builder.sign_extend(x, width + 1)?
                        } else {
                            builder.zero_extend(x, width + 1)?
                        };
                        let one = metadata_literal(builder, 1)?;
                        if signed {
                            builder.shra(wide, one)?
                        } else {
                            builder.shrl(wide, one)?
                        }
                    }
                    BuiltinOp::BitSliceUpdate => {
                        let index = resize_unsigned(builder, y, 2)?;
                        builder.bit_slice_update(x, index, z)?
                    }
                    BuiltinOp::Signex => {
                        let extended = builder.sign_extend(x, width + 3)?;
                        let amount = metadata_literal(builder, 2)?;
                        builder.shrl(extended, amount)?
                    }
                };
                resize_unsigned(builder, value, width)
            }
            Self::Array(op) => {
                let array = builder.array(ir::Type::Bits(width), &args)?;
                let index = resize_unsigned(builder, x, 32)?;
                let three = metadata_literal(builder, 3)?;
                let index = builder.umod(index, three)?;
                let one = metadata_literal(builder, 1)?;
                match op {
                    ArrayOp::Index => builder.array_index(array, index),
                    ArrayOp::Update => {
                        let updated = builder.array_update(array, z, &[index])?;
                        builder.array_index(updated, one)
                    }
                    ArrayOp::Slice => {
                        let slice = builder.array_slice(array, index, 2)?;
                        builder.array_index(slice, one)
                    }
                    ArrayOp::Concat => {
                        let left = builder.array(ir::Type::Bits(width), &[x, y])?;
                        let right = builder.array(ir::Type::Bits(width), &[z])?;
                        let array = builder.array_concat(&[left, right])?;
                        builder.array_index(array, index)
                    }
                    ArrayOp::Reverse => {
                        let array = builder.array(ir::Type::Bits(width), &[z, y, x])?;
                        builder.array_index(array, index)
                    }
                    ArrayOp::Enumerate | ArrayOp::LocalEnumerate => {
                        let element = builder.array_index(array, index)?;
                        let i = resize_unsigned(builder, index, width)?;
                        builder.xor(element, i)
                    }
                    ArrayOp::Zip => {
                        let right = builder.array(ir::Type::Bits(width), &[z, x, y])?;
                        let left = builder.array_index(array, index)?;
                        let right = builder.array_index(right, index)?;
                        builder.add(left, right)
                    }
                    ArrayOp::Repeat => {
                        let array = builder.array(ir::Type::Bits(width), &[x, y, y])?;
                        builder.array_index(array, index)
                    }
                }
            }
            Self::Pattern(op) => {
                let selector = resize_unsigned(builder, x, 2)?;
                let zero = builder.literal(IrValue::from_bits(&IrBits::zero(2)))?;
                let is_zero = builder.eq(selector, zero)?;
                match op {
                    PatternOp::Alternatives => {
                        let low_bit = builder.bit_slice(selector, 0, 1)?;
                        let even = builder.not(low_bit)?;
                        let one = builder
                            .literal(IrValue::from_bits(&IrBits::from_lsb_fn(2, |index| {
                                index == 0
                            })))?;
                        let is_one = builder.eq(selector, one)?;
                        let odd = builder.select(is_one, &[z, x], None)?;
                        builder.select(even, &[odd, y], None)
                    }
                    PatternOp::Range => {
                        let low_half = builder.bit_slice(selector, 1, 1)?;
                        builder.select(low_half, &[y, z], None)
                    }
                    PatternOp::Tuple => {
                        let sum = builder.add(y, z)?;
                        builder.select(is_zero, &[z, sum], None)
                    }
                    PatternOp::Enum => {
                        let one = builder
                            .literal(IrValue::from_bits(&IrBits::from_lsb_fn(2, |index| {
                                index == 0
                            })))?;
                        let is_one = builder.eq(selector, one)?;
                        let nonzero = builder.select(is_one, &[z, y], None)?;
                        builder.select(is_zero, &[nonzero, x], None)
                    }
                }
            }
            Self::Record { update } => builder.add(x, if update { z } else { y }),
            Self::Loop { kind, start, trips } => {
                if kind != LoopKind::Array {
                    return RangeValue::unsigned_loop(start, trips)
                        .emit_loop_ir(builder, args, width);
                }
                let mut acc = x;
                for iteration in 0..trips {
                    let element = if iteration % 2 == 0 { y } else { z };
                    let sum = builder.add(acc, element)?;
                    acc = builder.xor(sum, z)?;
                }
                Ok(acc)
            }
        }
    }
}

/// Reinterprets bits as unsigned and changes width without host-value
/// extraction.
pub(crate) fn resize_unsigned(
    builder: &mut FnBuilder,
    value: BValue,
    width: usize,
) -> Result<BValue, BuilderError> {
    let ir::Type::Bits(original) = builder.get_type(value)? else {
        return Err(BuilderError::InvalidOperation(
            "expected a bits value".to_string(),
        ));
    };
    if *original > width {
        builder.bit_slice(value, 0, width)
    } else if *original < width {
        builder.zero_extend(value, width)
    } else {
        Ok(value)
    }
}

/// Encodes loop/index metadata as a checked arbitrary-width literal.
pub(crate) fn metadata_literal(
    builder: &mut FnBuilder,
    value: usize,
) -> Result<BValue, BuilderError> {
    let bits = metadata_bits(value).map_err(BuilderError::InvalidOperation)?;
    builder.literal(IrValue::from_bits(&bits))
}
