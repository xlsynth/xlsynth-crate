// SPDX-License-Identifier: Apache-2.0

//! Native Rust values for PIR literals and execution.
//!
//! xlsynth::XlsIrBits and xlsynth::XlsIrValue are convenient libxls-backed
//! handles, but they make otherwise-native PIR manipulation depend on the XLS
//! DSO. This module provides the same value-shaped operations needed by PIR
//! while keeping a canonical Rust-owned representation.

use std::fmt;
use std::sync::Arc;

use num_bigint::{BigInt, BigUint, Sign};
use smallvec::SmallVec;

use crate::ir::{ArrayTypeData, Type};

/// Error produced by native PIR value construction or conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueError(pub String);

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ValueError {}

/// Text format used for native PIR values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrFormatPreference {
    Default,
    Binary,
    SignedDecimal,
    UnsignedDecimal,
    Hex,
    PlainBinary,
    ZeroPaddedBinary,
    PlainHex,
    ZeroPaddedHex,
}

/// A canonical arbitrary-width bitvector stored in least-significant-first
/// u64 limbs.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IrBits {
    bit_count: usize,
    limbs: SmallVec<[u64; 1]>,
}

impl IrBits {
    /// Constructs an unsigned bitvector, rejecting values that do not fit.
    pub fn make_ubits(bit_count: usize, value: u64) -> Result<Self, ValueError> {
        if bit_count < 64 && value >> bit_count != 0 {
            return Err(ValueError(format!(
                "value {value} does not fit in bits[{bit_count}]"
            )));
        }
        let mut result = Self::zero(bit_count);
        if let Some(limb) = result.limbs.first_mut() {
            *limb = value;
        }
        Ok(result)
    }

    /// Constructs a signed two's-complement bitvector, rejecting values that
    /// do not fit.
    pub fn make_sbits(bit_count: usize, value: i64) -> Result<Self, ValueError> {
        if bit_count == 0 {
            return if value == 0 {
                Ok(Self::zero(0))
            } else {
                Err(ValueError(format!("value {value} does not fit in bits[0]")))
            };
        }
        if bit_count < 64 {
            let min = -(1i128 << (bit_count - 1));
            let max = (1i128 << (bit_count - 1)) - 1;
            let value_i128 = i128::from(value);
            if value_i128 < min || value_i128 > max {
                return Err(ValueError(format!(
                    "value {value} does not fit in bits[{bit_count}]"
                )));
            }
        }
        let mut result = if value < 0 {
            Self::all_ones(bit_count)
        } else {
            Self::zero(bit_count)
        };
        result.limbs[0] = value as u64;
        result.mask_high_limb();
        Ok(result)
    }

    /// Constructs a bitvector from little-endian payload bytes.
    pub fn from_le_bytes(bit_count: usize, bytes: &[u8]) -> Result<Self, ValueError> {
        let expected = bit_count.div_ceil(8);
        if bytes.len() != expected {
            return Err(ValueError(format!(
                "expected {expected} bytes for bits[{bit_count}], got {}",
                bytes.len()
            )));
        }
        let remainder = bit_count % 8;
        if remainder != 0 && bytes.last().is_some_and(|last| last >> remainder != 0) {
            return Err(ValueError(format!(
                "high bits are set outside bits[{bit_count}] in final byte"
            )));
        }
        let mut result = Self::zero(bit_count);
        for (limb, bytes) in result.limbs.iter_mut().zip(bytes.chunks(8)) {
            let mut padded = [0; 8];
            padded[..bytes.len()].copy_from_slice(bytes);
            *limb = u64::from_le_bytes(padded);
        }
        Ok(result)
    }

    /// Constructs a bitvector from booleans whose index zero is the LSB.
    pub fn from_lsb_is_0(bits: &[bool]) -> Self {
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (index, bit) in bits.iter().copied().enumerate() {
            if bit {
                bytes[index / 8] |= 1u8 << (index % 8);
            }
        }
        Self::from_le_bytes(bits.len(), &bytes).expect("boolean bits are canonical")
    }

    /// Constructs a bitvector from booleans whose index zero is the MSB.
    pub fn from_msb_is_0(bits: &[bool]) -> Self {
        let mut reversed = bits.to_vec();
        reversed.reverse();
        Self::from_lsb_is_0(&reversed)
    }

    /// Returns the all-zero value at this width.
    pub fn zero(bit_count: usize) -> Self {
        Self {
            bit_count,
            limbs: smallvec::smallvec![0; bit_count.div_ceil(64)],
        }
    }

    /// Returns the all-ones value at this width.
    pub fn all_ones(bit_count: usize) -> Self {
        if bit_count == 0 {
            return Self::zero(0);
        }
        let mut result = Self {
            bit_count,
            limbs: smallvec::smallvec![u64::MAX; bit_count.div_ceil(64)],
        };
        result.mask_high_limb();
        result
    }

    /// Returns the maximum signed two's-complement value at this width.
    pub fn signed_max_value(bit_count: usize) -> Self {
        if bit_count == 0 {
            return Self::zero(0);
        }
        let mut result = Self::all_ones(bit_count);
        result.set_bit(bit_count - 1, false);
        result
    }

    /// Returns the minimum signed two's-complement value at this width.
    pub fn signed_min_value(bit_count: usize) -> Self {
        if bit_count == 0 {
            return Self::zero(0);
        }
        let mut result = Self::zero(bit_count);
        result.set_bit(bit_count - 1, true);
        result
    }

    pub fn bool(value: bool) -> Self {
        Self::make_ubits(1, u64::from(value)).expect("bool fits bits[1]")
    }

    pub fn u32(value: u32) -> Self {
        Self::make_ubits(32, u64::from(value)).expect("u32 fits bits[32]")
    }

    pub fn get_bit_count(&self) -> usize {
        self.bit_count
    }

    pub fn get_bit(&self, index: usize) -> Result<bool, ValueError> {
        if index >= self.bit_count {
            return Err(ValueError(format!(
                "bit index {index} out of bounds for bits[{}]",
                self.bit_count
            )));
        }
        Ok(((self.limbs[index / 64] >> (index % 64)) & 1) != 0)
    }

    /// Returns the canonical LSB-first limbs used by the native JIT ABI.
    pub fn limbs(&self) -> &[u64] {
        &self.limbs
    }

    /// Returns the canonical little-endian payload bytes.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.limbs.len() * 8);
        for limb in &self.limbs {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
        bytes.truncate(self.bit_count.div_ceil(8));
        bytes
    }

    /// Returns little-endian payload bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes()
    }

    pub fn to_u64(&self) -> Result<u64, ValueError> {
        if self.bit_count > 64 {
            return Err(ValueError(format!(
                "IrBits::to_u64(): width {} exceeds 64 bits",
                self.bit_count
            )));
        }
        Ok(self.limbs.first().copied().unwrap_or(0))
    }

    pub fn to_i64(&self) -> Result<i64, ValueError> {
        if self.bit_count > 64 {
            return Err(ValueError(format!(
                "IrBits::to_i64(): width {} exceeds 64 bits",
                self.bit_count
            )));
        }
        if self.bit_count == 0 {
            return Ok(0);
        }
        let unsigned = self.to_u64()?;
        let shift = 64 - self.bit_count;
        Ok(((unsigned << shift) as i64) >> shift)
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|limb| *limb == 0)
    }

    /// Returns whether this arbitrary-width value equals the given u64.
    pub fn equals_u64_value(&self, value: u64) -> bool {
        if self.bit_count < 64 && value >> self.bit_count != 0 {
            return false;
        }
        self.limbs.first().copied().unwrap_or(0) == value
            && self.limbs.iter().skip(1).all(|limb| *limb == 0)
    }

    pub fn equals(&self, rhs: &Self) -> bool {
        self == rhs
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.assert_matching_bit_count(rhs);
        Self::from_biguint_truncated(self.bit_count, self.to_biguint() + rhs.to_biguint())
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.assert_matching_bit_count(rhs);
        let modulus = Self::modulus(self.bit_count);
        let lhs = self.to_biguint();
        let rhs = rhs.to_biguint();
        let value = if lhs >= rhs {
            lhs - rhs
        } else {
            modulus + lhs - rhs
        };
        Self::from_biguint_truncated(self.bit_count, value)
    }

    pub fn umul(&self, rhs: &Self) -> Self {
        let result_width = self
            .bit_count
            .checked_add(rhs.bit_count)
            .expect("multiply result width overflow");
        Self::from_biguint_truncated(result_width, self.to_biguint() * rhs.to_biguint())
    }

    pub fn smul(&self, rhs: &Self) -> Self {
        let result_width = self
            .bit_count
            .checked_add(rhs.bit_count)
            .expect("multiply result width overflow");
        Self::from_bigint_truncated(
            result_width,
            self.to_bigint_signed() * rhs.to_bigint_signed(),
        )
    }

    pub fn negate(&self) -> Self {
        Self::from_bigint_truncated(self.bit_count, -self.to_bigint_signed())
    }

    pub fn abs(&self) -> Self {
        let value = self.to_bigint_signed();
        if value.sign() == Sign::Minus {
            Self::from_bigint_truncated(self.bit_count, -value)
        } else {
            self.clone()
        }
    }

    pub fn udiv(&self, rhs: &Self) -> Self {
        self.assert_matching_bit_count(rhs);
        if rhs.is_zero() {
            return Self::all_ones(self.bit_count);
        }
        Self::from_biguint_truncated(self.bit_count, self.to_biguint() / rhs.to_biguint())
    }

    pub fn umod(&self, rhs: &Self) -> Self {
        self.assert_matching_bit_count(rhs);
        if rhs.is_zero() {
            return Self::zero(self.bit_count);
        }
        Self::from_biguint_truncated(self.bit_count, self.to_biguint() % rhs.to_biguint())
    }

    pub fn sdiv(&self, rhs: &Self) -> Self {
        self.assert_matching_bit_count(rhs);
        let lhs = self.to_bigint_signed();
        let rhs = rhs.to_bigint_signed();
        if rhs == BigInt::from(0u8) {
            return if lhs.sign() == Sign::Minus {
                Self::signed_min_value(self.bit_count)
            } else {
                Self::signed_max_value(self.bit_count)
            };
        }
        Self::from_bigint_truncated(self.bit_count, lhs / rhs)
    }

    pub fn smod(&self, rhs: &Self) -> Self {
        self.assert_matching_bit_count(rhs);
        let rhs = rhs.to_bigint_signed();
        if rhs == BigInt::from(0u8) {
            return Self::zero(self.bit_count);
        }
        Self::from_bigint_truncated(self.bit_count, self.to_bigint_signed() % rhs)
    }

    pub fn shll(&self, shift_amount: i64) -> Self {
        let shift = usize::try_from(shift_amount).expect("shift amount must be non-negative");
        if shift >= self.bit_count {
            return Self::zero(self.bit_count);
        }
        Self::from_biguint_truncated(self.bit_count, self.to_biguint() << shift)
    }

    pub fn shrl(&self, shift_amount: i64) -> Self {
        let shift = usize::try_from(shift_amount).expect("shift amount must be non-negative");
        if shift >= self.bit_count {
            return Self::zero(self.bit_count);
        }
        Self::from_biguint_truncated(self.bit_count, self.to_biguint() >> shift)
    }

    pub fn shra(&self, shift_amount: i64) -> Self {
        let shift = usize::try_from(shift_amount).expect("shift amount must be non-negative");
        if shift >= self.bit_count {
            return if self.is_negative() {
                Self::all_ones(self.bit_count)
            } else {
                Self::zero(self.bit_count)
            };
        }
        Self::from_bigint_truncated(self.bit_count, self.to_bigint_signed() >> shift)
    }

    pub fn width_slice(&self, start: i64, width: i64) -> Self {
        let start = usize::try_from(start).expect("slice start must be non-negative");
        let width = usize::try_from(width).expect("slice width must be non-negative");
        let mut result = Self::zero(width);
        for output_index in 0..width {
            if start
                .checked_add(output_index)
                .is_some_and(|input_index| input_index < self.bit_count)
                && self
                    .get_bit(start + output_index)
                    .expect("checked bit index")
            {
                result.set_bit(output_index, true);
            }
        }
        result
    }

    pub fn not(&self) -> Self {
        let mut result = self.clone();
        for limb in &mut result.limbs {
            *limb = !*limb;
        }
        result.mask_high_limb();
        result
    }

    pub fn and(&self, rhs: &Self) -> Self {
        self.bitwise(rhs, |lhs, rhs| lhs & rhs)
    }

    pub fn or(&self, rhs: &Self) -> Self {
        self.bitwise(rhs, |lhs, rhs| lhs | rhs)
    }

    pub fn xor(&self, rhs: &Self) -> Self {
        self.bitwise(rhs, |lhs, rhs| lhs ^ rhs)
    }

    pub fn ult(&self, rhs: &Self) -> bool {
        self.assert_matching_bit_count(rhs);
        self.to_biguint() < rhs.to_biguint()
    }

    pub fn ule(&self, rhs: &Self) -> bool {
        self.assert_matching_bit_count(rhs);
        self.to_biguint() <= rhs.to_biguint()
    }

    pub fn ugt(&self, rhs: &Self) -> bool {
        self.assert_matching_bit_count(rhs);
        self.to_biguint() > rhs.to_biguint()
    }

    pub fn uge(&self, rhs: &Self) -> bool {
        self.assert_matching_bit_count(rhs);
        self.to_biguint() >= rhs.to_biguint()
    }

    pub fn slt(&self, rhs: &Self) -> bool {
        self.assert_matching_bit_count(rhs);
        self.to_bigint_signed() < rhs.to_bigint_signed()
    }

    pub fn sle(&self, rhs: &Self) -> bool {
        self.assert_matching_bit_count(rhs);
        self.to_bigint_signed() <= rhs.to_bigint_signed()
    }

    pub fn sgt(&self, rhs: &Self) -> bool {
        self.assert_matching_bit_count(rhs);
        self.to_bigint_signed() > rhs.to_bigint_signed()
    }

    pub fn sge(&self, rhs: &Self) -> bool {
        self.assert_matching_bit_count(rhs);
        self.to_bigint_signed() >= rhs.to_bigint_signed()
    }

    pub fn is_negative(&self) -> bool {
        self.bit_count != 0
            && self
                .get_bit(self.bit_count - 1)
                .expect("sign bit is in bounds")
    }

    pub fn msb(&self) -> bool {
        self.bit_count != 0
            && self
                .get_bit(self.bit_count - 1)
                .expect("sign bit is in bounds")
    }

    pub fn to_debug_str(&self) -> String {
        format!(
            "bits[{}]:{}",
            self.bit_count,
            self.format_payload(IrFormatPreference::Default)
        )
    }

    pub fn to_string_fmt(&self, format: IrFormatPreference, include_bit_count: bool) -> String {
        let payload = self.format_payload(format);
        if include_bit_count {
            format!("{payload} [{} bits]", self.bit_count)
        } else {
            payload
        }
    }

    fn bitwise(&self, rhs: &Self, op: impl Fn(u64, u64) -> u64) -> Self {
        self.assert_matching_bit_count(rhs);
        let limbs = self
            .limbs
            .iter()
            .zip(rhs.limbs.iter())
            .map(|(lhs, rhs)| op(*lhs, *rhs))
            .collect();
        let mut result = Self {
            bit_count: self.bit_count,
            limbs,
        };
        result.mask_high_limb();
        result
    }

    fn assert_matching_bit_count(&self, rhs: &Self) {
        assert_eq!(
            self.bit_count, rhs.bit_count,
            "bit width mismatch: left bits[{}] vs right bits[{}]",
            self.bit_count, rhs.bit_count
        );
    }

    fn set_bit(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.bit_count);
        let mask = 1u64 << (index % 64);
        let limb = &mut self.limbs[index / 64];
        if value {
            *limb |= mask;
        } else {
            *limb &= !mask;
        }
    }

    fn mask_high_limb(&mut self) {
        if self.bit_count == 0 {
            return;
        }
        let remainder = self.bit_count % 64;
        if remainder != 0 {
            let mask = (1u64 << remainder) - 1;
            *self.limbs.last_mut().expect("nonzero width has limb") &= mask;
        }
    }

    fn modulus(bit_count: usize) -> BigUint {
        BigUint::from(1u8) << bit_count
    }

    fn from_biguint_truncated(bit_count: usize, value: BigUint) -> Self {
        if bit_count == 0 {
            return Self::zero(0);
        }
        let mask = Self::modulus(bit_count) - BigUint::from(1u8);
        let value = value & mask;
        let bytes = value.to_bytes_le();
        let mut limbs = SmallVec::<[u64; 1]>::with_capacity(bit_count.div_ceil(64));
        for chunk in bytes.chunks(8) {
            let mut limb_bytes = [0u8; 8];
            limb_bytes[..chunk.len()].copy_from_slice(chunk);
            limbs.push(u64::from_le_bytes(limb_bytes));
        }
        limbs.resize(bit_count.div_ceil(64), 0);
        let mut result = Self { bit_count, limbs };
        result.mask_high_limb();
        result
    }

    fn from_bigint_truncated(bit_count: usize, value: BigInt) -> Self {
        if bit_count == 0 {
            return Self::zero(0);
        }
        let modulus = BigInt::from(1u8) << bit_count;
        let mut normalized = value % &modulus;
        if normalized.sign() == Sign::Minus {
            normalized += &modulus;
        }
        Self::from_biguint_truncated(
            bit_count,
            normalized
                .to_biguint()
                .expect("normalized two's-complement value is non-negative"),
        )
    }

    fn to_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(&self.to_le_bytes())
    }

    fn to_bigint_signed(&self) -> BigInt {
        let unsigned = self.to_biguint();
        if self.is_negative() {
            BigInt::from_biguint(Sign::Plus, unsigned) - (BigInt::from(1u8) << self.bit_count)
        } else {
            BigInt::from_biguint(Sign::Plus, unsigned)
        }
    }

    fn format_payload(&self, format: IrFormatPreference) -> String {
        let unsigned = self.to_biguint();
        match format {
            IrFormatPreference::Default if self.bit_count > 64 => {
                format!("0x{}", group_from_right(&format!("{unsigned:x}")))
            }
            IrFormatPreference::Default | IrFormatPreference::UnsignedDecimal => {
                unsigned.to_string()
            }
            IrFormatPreference::SignedDecimal => self.to_bigint_signed().to_string(),
            IrFormatPreference::Hex => format!("0x{}", group_from_right(&format!("{unsigned:x}"))),
            IrFormatPreference::PlainHex => format!("{unsigned:x}"),
            IrFormatPreference::ZeroPaddedHex => {
                let digits = self.bit_count.div_ceil(4).max(1);
                group_from_right(&format!("{unsigned:0digits$x}"))
            }
            IrFormatPreference::Binary => {
                format!("0b{}", group_from_right(&format!("{unsigned:b}")))
            }
            IrFormatPreference::PlainBinary => format!("{unsigned:b}"),
            IrFormatPreference::ZeroPaddedBinary => {
                let digits = self.bit_count.max(1);
                group_from_right(&format!("{unsigned:0digits$b}"))
            }
        }
    }
}

impl fmt::Debug for IrBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_debug_str())
    }
}

impl fmt::Display for IrBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bits[{}]:{}",
            self.bit_count,
            self.to_string_fmt(IrFormatPreference::Default, false)
        )
    }
}

/// A homogeneous array, constructed through [`IrValue::make_array_typed`].
///
/// Its element type is retained even for empty arrays. Fields are private so
/// callers cannot bypass type checking or mutate elements independently of
/// their declared type.
///
/// ```compile_fail
/// use std::sync::Arc;
/// use xlsynth_pir::{IrArray, IrValue, ir::Type};
/// let invalid = IrArray {
///     element_type: Type::Bits(8),
///     elements: Arc::from([IrValue::bool(true)]),
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IrArray {
    element_type: Type,
    elements: Arc<[IrValue]>,
}

impl IrArray {
    /// Returns the common element type, including for an empty array.
    pub fn element_type(&self) -> &Type {
        &self.element_type
    }

    /// Borrows the elements without cloning their storage.
    pub fn elements(&self) -> &[IrValue] {
        &self.elements
    }
}

/// A native recursive PIR value.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum IrValue {
    Token,
    Bits(IrBits),
    Tuple(Arc<[IrValue]>),
    Array(IrArray),
}

impl IrValue {
    pub fn make_token() -> Self {
        Self::Token
    }

    pub fn make_tuple(elements: &[IrValue]) -> Self {
        Self::Tuple(Arc::from(elements.to_vec()))
    }

    pub fn make_array(elements: &[IrValue]) -> Result<Self, ValueError> {
        let Some(first) = elements.first() else {
            return Err(ValueError(
                "empty arrays require make_array_typed".to_string(),
            ));
        };
        Self::make_array_typed(first.type_(), elements)
    }

    /// Constructs an array while retaining its element type for the empty case.
    pub fn make_array_typed(element_type: Type, elements: &[IrValue]) -> Result<Self, ValueError> {
        if elements
            .iter()
            .any(|element| element.type_() != element_type)
        {
            return Err(ValueError(format!(
                "array element does not match expected type {element_type}"
            )));
        }
        Ok(Self::Array(IrArray {
            element_type,
            elements: Arc::from(elements.to_vec()),
        }))
    }

    pub fn from_bits(bits: &IrBits) -> Self {
        Self::Bits(bits.clone())
    }

    /// Parses XLS typed-value syntax for bits, tokens, tuples, and nonempty
    /// arrays without consulting libxls.
    pub fn parse_typed(text: &str) -> Result<Self, ValueError> {
        let mut parser = TypedValueParser::new(text);
        let value = parser.parse_value()?;
        parser.skip_whitespace();
        if !parser.is_at_end() {
            return Err(ValueError(format!(
                "unexpected trailing text in typed value: {:?}",
                parser.remaining()
            )));
        }
        Ok(value)
    }

    fn parse_bits_payload(width: usize, payload: &str) -> Result<Self, ValueError> {
        let payload = payload.trim().replace('_', "");
        let negative = payload.starts_with('-');
        let magnitude = payload.strip_prefix('-').unwrap_or(&payload);
        let (digits, radix) = if let Some(hex) = magnitude.strip_prefix("0x") {
            (hex, 16)
        } else if let Some(binary) = magnitude.strip_prefix("0b") {
            (binary, 2)
        } else {
            (magnitude, 10)
        };
        if digits.is_empty() || !digits.chars().all(|digit| digit.is_digit(radix)) {
            return Err(ValueError(format!("invalid bits payload: {payload}")));
        }
        let magnitude = BigUint::parse_bytes(digits.as_bytes(), radix)
            .ok_or_else(|| ValueError(format!("invalid bits payload: {payload}")))?;
        let bits = if negative {
            let value = -BigInt::from(magnitude);
            let minimum = if width == 0 {
                BigInt::from(0u8)
            } else {
                -(BigInt::from(1u8) << (width - 1))
            };
            if value < minimum {
                return Err(ValueError(format!(
                    "value does not fit in signed bits[{width}]"
                )));
            }
            IrBits::from_bigint_truncated(width, value)
        } else {
            Self::bits_from_positive_biguint(width, magnitude)?
        };
        Ok(Self::Bits(bits))
    }

    fn bits_from_positive_biguint(width: usize, value: BigUint) -> Result<IrBits, ValueError> {
        if value >= IrBits::modulus(width) {
            return Err(ValueError(format!("value does not fit in bits[{width}]")));
        }
        Ok(IrBits::from_biguint_truncated(width, value))
    }

    pub fn bool(value: bool) -> Self {
        Self::Bits(IrBits::bool(value))
    }

    pub fn u32(value: u32) -> Self {
        Self::Bits(IrBits::u32(value))
    }

    pub fn u64(value: u64) -> Self {
        Self::Bits(IrBits::make_ubits(64, value).expect("u64 fits bits[64]"))
    }

    pub fn make_ubits(bit_count: usize, value: u64) -> Result<Self, ValueError> {
        Ok(Self::Bits(IrBits::make_ubits(bit_count, value)?))
    }

    pub fn make_sbits(bit_count: usize, value: i64) -> Result<Self, ValueError> {
        Ok(Self::Bits(IrBits::make_sbits(bit_count, value)?))
    }

    pub fn all_ones_bits(bit_count: usize) -> Self {
        Self::Bits(IrBits::all_ones(bit_count))
    }

    pub fn signed_max_bits(bit_count: usize) -> Self {
        Self::Bits(IrBits::signed_max_value(bit_count))
    }

    pub fn signed_min_bits(bit_count: usize) -> Self {
        Self::Bits(IrBits::signed_min_value(bit_count))
    }

    pub fn type_(&self) -> Type {
        match self {
            Self::Token => Type::Token,
            Self::Bits(bits) => Type::Bits(bits.bit_count),
            Self::Tuple(elements) => Type::Tuple(
                elements
                    .iter()
                    .map(|element| Box::new(element.type_()))
                    .collect(),
            ),
            Self::Array(array) => Type::Array(ArrayTypeData {
                element_type: Box::new(array.element_type.clone()),
                element_count: array.elements.len(),
            }),
        }
    }

    pub fn bit_count(&self) -> Result<usize, ValueError> {
        Ok(self.as_bits()?.get_bit_count())
    }

    pub fn to_bits(&self) -> Result<IrBits, ValueError> {
        self.as_bits().cloned()
    }

    /// Borrows bits without cloning their limb storage.
    pub fn as_bits(&self) -> Result<&IrBits, ValueError> {
        match self {
            Self::Bits(bits) => Ok(bits),
            _ => Err(ValueError(format!(
                "value of type {} is not bits-typed",
                self.type_()
            ))),
        }
    }

    pub fn to_bool(&self) -> Result<bool, ValueError> {
        let bits = self.as_bits()?;
        if bits.bit_count != 1 {
            return Err(ValueError(format!(
                "value {self} is not single-bit; must be bits[1] to convert to bool"
            )));
        }
        bits.get_bit(0)
    }

    pub fn to_u64(&self) -> Result<u64, ValueError> {
        self.as_bits()?.to_u64()
    }

    pub fn to_i64(&self) -> Result<i64, ValueError> {
        self.as_bits()?.to_i64()
    }

    pub fn to_u32(&self) -> Result<u32, ValueError> {
        let value = self.to_u64()?;
        u32::try_from(value).map_err(|_| ValueError(format!("value {value} does not fit in u32")))
    }

    pub fn bits_equals_u64_value(&self, value: u64) -> bool {
        self.as_bits()
            .map(|bits| bits.equals_u64_value(value))
            .unwrap_or(false)
    }

    pub fn get_element(&self, index: usize) -> Result<IrValue, ValueError> {
        self.as_elements()?
            .get(index)
            .cloned()
            .ok_or_else(|| ValueError(format!("element index {index} out of bounds")))
    }

    pub fn get_element_count(&self) -> Result<usize, ValueError> {
        Ok(self.as_elements()?.len())
    }

    pub fn get_elements(&self) -> Result<Vec<IrValue>, ValueError> {
        Ok(self.as_elements()?.to_vec())
    }

    /// Borrows tuple or array elements without cloning their storage.
    pub fn as_elements(&self) -> Result<&[IrValue], ValueError> {
        match self {
            Self::Tuple(elements) => Ok(elements),
            Self::Array(array) => Ok(array.elements()),
            _ => Err(ValueError(format!(
                "value of type {} has no elements",
                self.type_()
            ))),
        }
    }

    pub fn to_string_fmt(&self, format: IrFormatPreference) -> String {
        self.format_with_prefix(format, true)
    }

    pub fn to_string_fmt_no_prefix(&self, format: IrFormatPreference) -> String {
        self.format_with_prefix(format, false)
    }

    fn format_with_prefix(&self, format: IrFormatPreference, include_bits_type: bool) -> String {
        match self {
            Self::Token => "token".to_string(),
            Self::Bits(bits) => {
                let payload = bits.to_string_fmt(format, false);
                if include_bits_type {
                    format!("bits[{}]:{payload}", bits.get_bit_count())
                } else {
                    payload
                }
            }
            Self::Tuple(elements) => format!(
                "({})",
                elements
                    .iter()
                    .map(|element| element.format_with_prefix(format, include_bits_type))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Array(array) => format!(
                "[{}]",
                array
                    .elements
                    .iter()
                    .map(|element| element.format_with_prefix(format, include_bits_type))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl fmt::Debug for IrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for IrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_with_prefix(IrFormatPreference::Default, true))
    }
}

fn group_from_right(digits: &str) -> String {
    if digits.len() <= 4 {
        return digits.to_string();
    }
    let mut groups = Vec::new();
    let mut end = digits.len();
    while end > 0 {
        let start = end.saturating_sub(4);
        groups.push(&digits[start..end]);
        end = start;
    }
    groups.reverse();
    groups.join("_")
}

struct TypedValueParser<'a> {
    text: &'a str,
    offset: usize,
}

impl<'a> TypedValueParser<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, offset: 0 }
    }

    fn parse_value(&mut self) -> Result<IrValue, ValueError> {
        self.skip_whitespace();
        match self.peek_char() {
            Some('[') => self.parse_array(),
            Some('(') => self.parse_tuple(),
            Some(_) if self.remaining().starts_with("token") => {
                self.offset += "token".len();
                Ok(IrValue::Token)
            }
            Some(_) if self.remaining().starts_with("bits") => self.parse_bits(),
            _ => Err(ValueError(format!(
                "expected typed value, got {:?}",
                self.remaining()
            ))),
        }
    }

    fn parse_array(&mut self) -> Result<IrValue, ValueError> {
        self.expect_char('[')?;
        self.skip_whitespace();
        if self.consume_char(']') {
            return Err(ValueError(
                "cannot infer the element type of an empty array; use make_array_typed".to_string(),
            ));
        }
        let mut elements = Vec::new();
        loop {
            elements.push(self.parse_value()?);
            self.skip_whitespace();
            if self.consume_char(']') {
                break;
            }
            self.expect_char(',')?;
        }
        IrValue::make_array(&elements)
    }

    fn parse_tuple(&mut self) -> Result<IrValue, ValueError> {
        self.expect_char('(')?;
        self.skip_whitespace();
        if self.consume_char(')') {
            return Ok(IrValue::make_tuple(&[]));
        }
        let mut elements = Vec::new();
        loop {
            elements.push(self.parse_value()?);
            self.skip_whitespace();
            if self.consume_char(')') {
                break;
            }
            self.expect_char(',')?;
        }
        Ok(IrValue::make_tuple(&elements))
    }

    fn parse_bits(&mut self) -> Result<IrValue, ValueError> {
        self.offset += "bits".len();
        self.expect_char('[')?;
        self.skip_whitespace();
        let width_start = self.offset;
        while self.peek_char().is_some_and(|ch| ch.is_ascii_digit()) {
            self.offset += 1;
        }
        let width_text = &self.text[width_start..self.offset];
        let width = width_text
            .parse::<usize>()
            .map_err(|error| ValueError(format!("invalid bits width {width_text:?}: {error}")))?;
        self.expect_char(']')?;
        self.expect_char(':')?;
        self.skip_whitespace();
        let start = self.offset;
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() || matches!(ch, ',' | ')' | ']') {
                break;
            }
            self.offset += ch.len_utf8();
        }
        IrValue::parse_bits_payload(width, &self.text[start..self.offset])
    }

    fn expect_char(&mut self, expected: char) -> Result<(), ValueError> {
        self.skip_whitespace();
        if self.consume_char(expected) {
            Ok(())
        } else {
            Err(ValueError(format!(
                "expected {expected:?}, got {:?}",
                self.remaining()
            )))
        }
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.offset += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn remaining(&self) -> &'a str {
        &self.text[self.offset..]
    }

    fn skip_whitespace(&mut self) {
        while self.peek_char().is_some_and(char::is_whitespace) {
            self.offset += self.peek_char().expect("checked character").len_utf8();
        }
    }

    fn is_at_end(&self) -> bool {
        self.offset == self.text.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_roundtrip_le_bytes_and_format() {
        let bits = IrBits::from_le_bytes(12, &[0xcd, 0x0a]).unwrap();
        assert_eq!(bits.to_le_bytes(), vec![0xcd, 0x0a]);
        assert_eq!(
            bits.to_string_fmt(IrFormatPreference::Hex, true),
            "0xacd [12 bits]"
        );
    }

    #[test]
    fn wide_bits_parse_and_format() {
        let value = IrValue::parse_typed("bits[65]:0x1_ffff_ffff_ffff_fffe").unwrap();
        assert_eq!(
            value.to_string_fmt(IrFormatPreference::Hex),
            "bits[65]:0x1_ffff_ffff_ffff_fffe"
        );
    }

    #[test]
    fn zero_width_signed_conversion_is_zero() {
        let value = IrValue::make_ubits(0, 0).unwrap();
        assert_eq!(value.to_i64().unwrap(), 0);
    }

    #[test]
    fn typed_empty_array_retains_element_type() {
        let value = IrValue::make_array_typed(Type::Bits(8), &[]).unwrap();
        assert_eq!(value.type_(), Type::new_array(Type::Bits(8), 0));
        let IrValue::Array(array) = &value else {
            panic!("expected array payload");
        };
        assert_eq!(array.element_type(), &Type::Bits(8));
        assert!(array.elements().is_empty());
    }

    #[test]
    fn array_construction_checks_full_element_types() {
        let empty_bits8 = IrValue::make_array_typed(Type::Bits(8), &[]).unwrap();
        let empty_bits16 = IrValue::make_array_typed(Type::Bits(16), &[]).unwrap();
        for (first, second) in [
            (IrValue::make_ubits(8, 1).unwrap(), IrValue::bool(true)),
            (IrValue::make_tuple(&[]), IrValue::make_token()),
            (empty_bits8.clone(), empty_bits16),
            (
                IrValue::make_tuple(&[IrValue::bool(true)]),
                IrValue::make_array(&[IrValue::bool(true)]).unwrap(),
            ),
        ] {
            assert!(
                IrValue::make_array_typed(first.type_(), std::slice::from_ref(&second)).is_err()
            );
            assert!(IrValue::make_array(&[first, second]).is_err());
        }
        assert!(IrValue::make_array(&[]).is_err());
        let nested = IrValue::make_array(&[empty_bits8.clone(), empty_bits8]).unwrap();
        assert_eq!(
            nested.type_(),
            Type::new_array(Type::new_array(Type::Bits(8), 0), 2)
        );
    }

    #[test]
    fn aggregate_accessors_borrow_stored_elements() {
        let child = IrValue::make_ubits(129, 7).unwrap();
        for value in [
            IrValue::make_tuple(std::slice::from_ref(&child)),
            IrValue::make_array(std::slice::from_ref(&child)).unwrap(),
        ] {
            let stored = match &value {
                IrValue::Tuple(elements) => elements.as_ref(),
                IrValue::Array(array) => {
                    assert_eq!(array.element_type(), &child.type_());
                    array.elements()
                }
                _ => panic!("expected aggregate"),
            };
            let borrowed = value.as_elements().unwrap();
            assert!(std::ptr::eq(borrowed, stored));
            assert!(std::ptr::eq(
                borrowed[0].as_bits().unwrap(),
                stored[0].as_bits().unwrap()
            ));
            assert_eq!(borrowed, std::slice::from_ref(&child));
            assert_eq!(value.get_element_count().unwrap(), 1);
            assert_eq!(value.get_element(0).unwrap(), child);
            assert!(value.get_element(1).is_err());
            let mut owned = value.get_elements().unwrap();
            owned[0] = IrValue::make_token();
            assert_eq!(value.as_elements().unwrap(), std::slice::from_ref(&child));
        }
        assert!(IrValue::make_tuple(&[]).as_elements().unwrap().is_empty());
        assert!(
            IrValue::make_array_typed(Type::Token, &[])
                .unwrap()
                .as_elements()
                .unwrap()
                .is_empty()
        );
        assert!(child.as_elements().is_err());
        assert!(IrValue::make_token().as_elements().is_err());
    }

    #[test]
    fn scalar_accessors_preserve_width_and_type_checks() {
        let wide = IrValue::make_ubits(129, 7).unwrap();
        assert_eq!(wide.bit_count().unwrap(), 129);
        assert!(wide.bits_equals_u64_value(7));
        assert!(!wide.bits_equals_u64_value(8));
        assert!(wide.to_u64().is_err());
        assert!(wide.to_i64().is_err());
        assert!(wide.to_bool().is_err());
        let negative = IrValue::make_sbits(8, -3).unwrap();
        assert_eq!(negative.to_i64().unwrap(), -3);
        assert_eq!(negative.to_u64().unwrap(), 253);
        assert!(IrValue::bool(true).to_bool().unwrap());
        let aggregate = IrValue::make_tuple(&[wide]);
        assert!(aggregate.bit_count().is_err());
        assert!(aggregate.as_bits().is_err());
        assert!(!aggregate.bits_equals_u64_value(7));
    }

    #[test]
    fn byte_conversion_is_infallible_at_limb_boundaries() {
        for width in [0, 1, 7, 8, 63, 64, 65, 129, 257] {
            let bits = IrBits::all_ones(width);
            let bytes: Vec<u8> = bits.to_le_bytes();
            assert_eq!(bytes.len(), width.div_ceil(8));
            assert_eq!(bits.to_bytes(), bytes);
            assert_eq!(IrBits::from_le_bytes(width, &bytes).unwrap(), bits);
        }
    }

    #[test]
    fn aggregate_typed_values_parse_without_libxls() {
        let text = "([bits[4]:1, bits[4]:2], (token, bits[1]:1))";
        let value = IrValue::parse_typed(text).unwrap();
        assert_eq!(value.to_string(), text);
    }
}
