// SPDX-License-Identifier: Apache-2.0

//! Array maps with independent elementwise PIR references.

use crate::ast::{BinaryOp, BitsType, Width};
use crate::compound::resize_unsigned;
use crate::constexpr::metadata_bits;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrValue, ir};

/// A unary function supplied to DSLX `map`, possibly capturing earlier values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapBody {
    Reverse,
    Clz,
    Ctz,
    AndReduce,
    OrReduce,
    XorReduce,
    Encode,
    IsEven,
    LocalIsEven,
    Capture(BinaryOp),
    TwoCaptures(BinaryOp, BinaryOp),
}

impl MapBody {
    /// Parametric builtins accepted as a direct function argument to `map`.
    pub const BUILTINS: [Self; 7] = [
        Self::Reverse,
        Self::Clz,
        Self::Ctz,
        Self::AndReduce,
        Self::OrReduce,
        Self::XorReduce,
        Self::Encode,
    ];

    /// Renders the function argument, retaining signedness for lambda
    /// arithmetic.
    fn to_dslx(self, ty: BitsType, y: &str, z: &str) -> String {
        let lane = ty.to_dslx();
        let param = match ty.width {
            Width::Concrete(_) => format!("element: {lane}"),
            Width::Parametric => "element".to_string(),
        };
        match self {
            Self::Reverse => "rev",
            Self::Clz => "clz",
            Self::Ctz => "ctz",
            Self::AndReduce => "and_reduce",
            Self::OrReduce => "or_reduce",
            Self::XorReduce => "xor_reduce",
            Self::Encode => "encode",
            Self::IsEven => "std::is_even",
            Self::LocalIsEven => "fuzz_is_even",
            Self::Capture(op) => {
                let body = op.to_dslx("element", y, ty);
                return format!("|{param}| -> {lane} {{ {body} }}");
            }
            Self::TwoCaptures(first, second) => {
                let first = first.to_dslx("element", y, ty);
                let body = second.to_dslx(&format!("({first})"), z, ty);
                return format!("|{param}| -> {lane} {{ {body} }}");
            }
        }
        .to_string()
    }

    /// Applies a single scalar map body without creating or invoking a
    /// function.
    fn emit_ir(
        self,
        builder: &mut FnBuilder,
        element: BValue,
        y: BValue,
        z: BValue,
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        match self {
            Self::Reverse => builder.rev(element),
            Self::Clz => builder.clz(element),
            Self::Ctz => builder.ctz(element),
            Self::AndReduce => builder.and_reduce(element),
            Self::OrReduce => builder.or_reduce(element),
            Self::XorReduce => builder.xor_reduce(element),
            Self::Encode => builder.encode(element),
            Self::IsEven | Self::LocalIsEven => {
                let lsb = builder.bit_slice(element, 0, 1)?;
                builder.not(lsb)
            }
            Self::Capture(op) => op.emit_ir(builder, element, y, signed),
            Self::TwoCaptures(first, second) => {
                let first = first.emit_ir(builder, element, y, signed)?;
                second.emit_ir(builder, first, z, signed)
            }
        }
    }
}

/// A nonempty array map projected at a dynamic, in-bounds index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapFragment {
    body: MapBody,
    length: usize,
}

impl MapFragment {
    pub const MAX_LENGTH: usize = 5;

    /// Checks the element count before either printing or unrolling the map.
    pub fn new(body: MapBody, length: usize) -> Result<Self, String> {
        if !(1..=Self::MAX_LENGTH).contains(&length) {
            return Err("map length must be in 1..=5".to_string());
        }
        Ok(Self { body, length })
    }

    pub(crate) fn needs_std(self) -> bool {
        self.body == MapBody::IsEven
    }

    pub(crate) fn needs_local_is_even(self) -> bool {
        self.body == MapBody::LocalIsEven
    }

    /// Formats map input and result casts at the lane's composition boundary.
    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let [x, y, z] = args;
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let elements = args
            .iter()
            .cycle()
            .take(self.length)
            .map(|arg| {
                if ty.signed && MapBody::BUILTINS.contains(&self.body) {
                    format!("({arg} as {unsigned})")
                } else {
                    arg.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let index = format!("(({x} as {unsigned}) as u32) % u32:{}", self.length);
        let lane = ty.to_dslx();
        // Normalize concrete parameter and capture types. Otherwise v0.61.0
        // can mangle a computed dimension (e.g. uN[32]:7) into a closure name
        // containing ':', which its own IR parser rejects. Parametric helpers
        // infer the lambda argument so it specializes at each caller's width.
        let (captures, y, z) = match (ty.width, self.body) {
            (Width::Concrete(_), MapBody::Capture(_)) => {
                (format!("let map_y = {y} as {lane}; "), "map_y", z)
            }
            (Width::Concrete(_), MapBody::TwoCaptures(_, _)) => (
                format!("let map_y = {y} as {lane}; let map_z = {z} as {lane}; "),
                "map_y",
                "map_z",
            ),
            _ => (String::new(), y, z),
        };
        let mapper = self.body.to_dslx(ty, y, z);
        format!("{{ {captures}(map([{elements}], {mapper})[{index}] as {lane}) }}")
    }

    /// Constructs each transformed element with PIR operations, then selects
    /// it.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        width: usize,
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        let [x, y, z] = args;
        let mut elements = Vec::with_capacity(self.length);
        for element in args.into_iter().cycle().take(self.length) {
            let mapped = self.body.emit_ir(builder, element, y, z, signed)?;
            elements.push(resize_unsigned(builder, mapped, width)?);
        }
        let mapped = builder.array(ir::Type::Bits(width), &elements)?;
        let index = resize_unsigned(builder, x, 32)?;
        let length = metadata_bits(self.length).map_err(BuilderError::InvalidOperation)?;
        let length = builder.literal(IrValue::from_bits(&length))?;
        let index = builder.umod(index, length)?;
        builder.array_index(mapped, index)
    }
}
