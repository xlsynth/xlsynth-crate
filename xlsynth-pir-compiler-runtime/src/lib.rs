// SPDX-License-Identifier: Apache-2.0

//! Runtime ABI and observable-event collection for compiled PIR functions.

use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr;

use num_bigint::{BigInt, BigUint, Sign};

/// Native compiled-function entrypoint shared by in-memory and AOT execution.
pub type CompiledEntrypoint = unsafe extern "C" fn(
    inputs: *const *const u8,
    output: *mut u8,
    scratch: *mut u8,
    context: *mut RawExecutionContext,
) -> i32;

/// Opaque execution context forwarded by compiled code to runtime callbacks.
#[repr(C)]
pub struct RawExecutionContext {
    private_state: *mut c_void,
}

/// Error returned by generated native-compiler entrypoint wrappers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunError(pub String);

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RunError {}

/// Constructs a value whose DSLX/PIR data bits are all zero.
///
/// Unlike [`Default`], this trait describes the representation being produced
/// rather than implying a domain-specific default value.
pub trait AllZeros {
    /// Returns a value whose DSLX/PIR data bits are all zero.
    fn all_zeros() -> Self;
}

impl<T: AllZeros, const N: usize> AllZeros for [T; N] {
    fn all_zeros() -> Self {
        std::array::from_fn(|_| T::all_zeros())
    }
}

macro_rules! define_native_bits {
    ($name:ident, $carrier:ty, $carrier_bits:expr) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
        pub struct $name<const BIT_COUNT: usize>($carrier);

        impl<const BIT_COUNT: usize> $name<BIT_COUNT> {
            /// Constructs the canonical all-zero bitvector value.
            pub const fn all_zeros() -> Self {
                Self::from_raw_bits(0)
            }

            fn validate_width() -> Result<(), RunError> {
                if BIT_COUNT == 0 || BIT_COUNT > $carrier_bits {
                    Err(RunError(format!(
                        "bits[{BIT_COUNT}] cannot use a {}-bit native carrier",
                        $carrier_bits
                    )))
                } else {
                    Ok(())
                }
            }

            const fn mask() -> $carrier {
                if BIT_COUNT == $carrier_bits {
                    <$carrier>::MAX
                } else {
                    ((1 as $carrier) << BIT_COUNT) - 1
                }
            }

            fn try_from_u64(value: u64) -> Result<Self, RunError> {
                Self::validate_width()?;
                let carrier = <$carrier>::try_from(value).map_err(|_| {
                    RunError(format!("value {value} does not fit in bits[{BIT_COUNT}]"))
                })?;
                if carrier & !Self::mask() != 0 {
                    Err(RunError(format!(
                        "value {value} does not fit in bits[{BIT_COUNT}]"
                    )))
                } else {
                    Ok(Self(carrier))
                }
            }

            /// Constructs a canonical bitvector, panicking if the value does
            /// not fit the requested width.
            pub fn new(value: u64) -> Self {
                Self::try_from_u64(value).unwrap_or_else(|error| panic!("{error}"))
            }

            /// Constructs a canonical bitvector from raw ABI bits.
            pub const fn from_raw_bits(value: u64) -> Self {
                assert!(
                    BIT_COUNT > 0 && BIT_COUNT <= $carrier_bits,
                    "invalid native bits carrier width"
                );
                assert!(
                    value <= <$carrier>::MAX as u64,
                    "raw bits do not fit native carrier"
                );
                let carrier = value as $carrier;
                assert!(
                    carrier & !Self::mask() == 0,
                    "raw bits do not fit target width"
                );
                Self(carrier)
            }

            /// Returns the native carrier value.
            pub const fn get(self) -> $carrier {
                self.0
            }

            /// Returns the value widened to `u64`.
            pub const fn to_u64(self) -> u64 {
                self.0 as u64
            }
        }

        impl<const BIT_COUNT: usize> AllZeros for $name<BIT_COUNT> {
            fn all_zeros() -> Self {
                Self::all_zeros()
            }
        }

        impl<const BIT_COUNT: usize> TryFrom<u64> for $name<BIT_COUNT> {
            type Error = RunError;

            fn try_from(value: u64) -> Result<Self, Self::Error> {
                Self::try_from_u64(value)
            }
        }
    };
}

define_native_bits!(BitsInU8, u8, 8);
define_native_bits!(BitsInU16, u16, 16);
define_native_bits!(BitsInU32, u32, 32);
define_native_bits!(BitsInU64, u64, 64);

/// Zero-sized native representation of a `bits[0]` value.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bits0;

impl Bits0 {
    /// Constructs the sole `bits[0]` representation.
    pub const fn all_zeros() -> Self {
        Self
    }
}

impl AllZeros for Bits0 {
    fn all_zeros() -> Self {
        Self::all_zeros()
    }
}

/// Public unsigned DSLX-style wrapper for a `bits[0]` value.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnsignedBits0;

impl UnsignedBits0 {
    /// Constructs the sole `bits[0]` representation.
    pub const fn all_zeros() -> Self {
        Self
    }

    /// Constructs the sole unsigned `bits[0]` value, panicking unless the
    /// supplied value is zero.
    pub fn new(value: u64) -> Self {
        Self::try_from(value).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Constructs the sole canonical raw `bits[0]` representation.
    pub const fn from_raw_bits(value: u64) -> Self {
        assert!(value == 0, "raw bits do not fit target width");
        Self
    }

    /// Returns the sole unsigned `bits[0]` value widened to `u64`.
    pub const fn to_u64(self) -> u64 {
        0
    }

    /// Returns the raw ABI bits widened to `u64`.
    pub const fn raw_bits(self) -> u64 {
        0
    }
}

impl AllZeros for UnsignedBits0 {
    fn all_zeros() -> Self {
        Self::all_zeros()
    }
}

impl TryFrom<u64> for UnsignedBits0 {
    type Error = RunError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value == 0 {
            Ok(Self)
        } else {
            Err(RunError(format!("value {value} does not fit in bits[0]")))
        }
    }
}

/// Public signed DSLX-style wrapper for an `sbits[0]` value.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SignedBits0;

impl SignedBits0 {
    /// Constructs the sole `sbits[0]` representation.
    pub const fn all_zeros() -> Self {
        Self
    }

    /// Constructs the sole signed `bits[0]` value, panicking unless the
    /// supplied value is zero.
    pub fn new(value: i64) -> Self {
        Self::try_from(value).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Constructs the sole canonical raw `sbits[0]` representation.
    pub const fn from_raw_bits(value: u64) -> Self {
        assert!(value == 0, "raw bits do not fit target width");
        Self
    }

    /// Returns the sole signed `sbits[0]` value widened to `i64`.
    pub const fn to_i64(self) -> i64 {
        0
    }

    /// Returns the raw ABI bits widened to `u64`.
    pub const fn raw_bits(self) -> u64 {
        0
    }
}

impl AllZeros for SignedBits0 {
    fn all_zeros() -> Self {
        Self::all_zeros()
    }
}

impl TryFrom<i64> for SignedBits0 {
    type Error = RunError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        if value == 0 {
            Ok(Self)
        } else {
            Err(RunError(format!("value {value} does not fit in s0")))
        }
    }
}

macro_rules! define_public_bits_wrappers {
    (
        $unsigned_name:ident,
        $signed_name:ident,
        $raw_name:ident,
        $unsigned_carrier:ty,
        $signed_carrier:ty,
        $to_unsigned:ident,
        $to_signed:ident,
        $carrier_bits:expr
    ) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
        pub struct $unsigned_name<const BIT_COUNT: usize>($raw_name<BIT_COUNT>);

        impl<const BIT_COUNT: usize> $unsigned_name<BIT_COUNT> {
            /// Constructs the canonical all-zero bitvector value.
            pub const fn all_zeros() -> Self {
                Self($raw_name::<BIT_COUNT>::all_zeros())
            }

            const fn mask() -> $unsigned_carrier {
                if BIT_COUNT == $carrier_bits {
                    <$unsigned_carrier>::MAX
                } else {
                    ((1 as $unsigned_carrier) << BIT_COUNT) - 1
                }
            }

            /// Constructs an unsigned DSLX-style bit value, panicking if the
            /// value does not fit the requested width.
            pub fn new(value: $unsigned_carrier) -> Self {
                Self::try_from(u64::from(value)).unwrap_or_else(|error| panic!("{error}"))
            }

            /// Constructs an unsigned DSLX-style bit value from raw ABI bits.
            pub const fn from_raw_bits(value: u64) -> Self {
                assert!(
                    BIT_COUNT > 0 && BIT_COUNT <= $carrier_bits,
                    "invalid raw bit carrier width"
                );
                assert!(
                    value <= <$unsigned_carrier>::MAX as u64,
                    "raw bits do not fit native carrier"
                );
                assert!(
                    (value as $unsigned_carrier) & !Self::mask() == 0,
                    "raw bits do not fit target width"
                );
                Self($raw_name::<BIT_COUNT>::from_raw_bits(value))
            }

            /// Returns the unsigned value in its narrowest native carrier.
            pub const fn $to_unsigned(self) -> $unsigned_carrier {
                self.0.get()
            }

            /// Returns the raw ABI bits widened to `u64`.
            pub const fn raw_bits(self) -> u64 {
                self.0.to_u64()
            }
        }

        impl<const BIT_COUNT: usize> AllZeros for $unsigned_name<BIT_COUNT> {
            fn all_zeros() -> Self {
                Self::all_zeros()
            }
        }

        impl<const BIT_COUNT: usize> TryFrom<u64> for $unsigned_name<BIT_COUNT> {
            type Error = RunError;

            fn try_from(value: u64) -> Result<Self, Self::Error> {
                Ok(Self($raw_name::<BIT_COUNT>::try_from(value)?))
            }
        }

        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
        pub struct $signed_name<const BIT_COUNT: usize>($raw_name<BIT_COUNT>);

        impl<const BIT_COUNT: usize> $signed_name<BIT_COUNT> {
            /// Constructs the canonical all-zero bitvector value.
            pub const fn all_zeros() -> Self {
                Self($raw_name::<BIT_COUNT>::all_zeros())
            }

            fn validate_signed_value(value: i64) -> Result<(), RunError> {
                if BIT_COUNT == 0 || BIT_COUNT > $carrier_bits {
                    return Err(RunError(format!(
                        "s{BIT_COUNT} cannot use a {}-bit native carrier",
                        $carrier_bits
                    )));
                }
                let min = -(1i128 << (BIT_COUNT - 1));
                let max = (1i128 << (BIT_COUNT - 1)) - 1;
                let value = value as i128;
                if value < min || value > max {
                    Err(RunError(format!(
                        "value {value} does not fit in s{BIT_COUNT}"
                    )))
                } else {
                    Ok(())
                }
            }

            const fn mask() -> $unsigned_carrier {
                if BIT_COUNT == $carrier_bits {
                    <$unsigned_carrier>::MAX
                } else {
                    ((1 as $unsigned_carrier) << BIT_COUNT) - 1
                }
            }

            /// Constructs a signed DSLX-style bit value, panicking if the
            /// value does not fit the requested width.
            pub fn new(value: $signed_carrier) -> Self {
                Self::try_from(i64::from(value)).unwrap_or_else(|error| panic!("{error}"))
            }

            /// Constructs a signed DSLX-style bit value from raw ABI bits.
            pub const fn from_raw_bits(value: u64) -> Self {
                assert!(
                    BIT_COUNT > 0 && BIT_COUNT <= $carrier_bits,
                    "invalid raw bit carrier width"
                );
                assert!(
                    value <= <$unsigned_carrier>::MAX as u64,
                    "raw bits do not fit native carrier"
                );
                assert!(
                    (value as $unsigned_carrier) & !Self::mask() == 0,
                    "raw bits do not fit target width"
                );
                Self($raw_name::<BIT_COUNT>::from_raw_bits(value))
            }

            /// Returns the sign-extended native signed carrier value.
            fn to_signed_carrier(self) -> $signed_carrier {
                let raw = self.0.get();
                let sign_bit = 1 as $unsigned_carrier << (BIT_COUNT - 1);
                if raw & sign_bit == 0 {
                    raw as $signed_carrier
                } else {
                    (raw | !Self::mask()) as $signed_carrier
                }
            }

            /// Returns the signed value in its narrowest native carrier.
            pub fn $to_signed(self) -> $signed_carrier {
                self.to_signed_carrier()
            }

            /// Returns the raw ABI bits widened to `u64`.
            pub const fn raw_bits(self) -> u64 {
                self.0.to_u64()
            }
        }

        impl<const BIT_COUNT: usize> AllZeros for $signed_name<BIT_COUNT> {
            fn all_zeros() -> Self {
                Self::all_zeros()
            }
        }

        impl<const BIT_COUNT: usize> TryFrom<i64> for $signed_name<BIT_COUNT> {
            type Error = RunError;

            fn try_from(value: i64) -> Result<Self, Self::Error> {
                Self::validate_signed_value(value)?;
                let raw_bits = (value as u64) & (Self::mask() as u64);
                Ok(Self($raw_name::<BIT_COUNT>::from_raw_bits(raw_bits)))
            }
        }
    };
}

/// Public boolean wrapper for a DSLX-style `u1` value.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bool(BitsInU8<1>);

impl Bool {
    /// Constructs the canonical false value.
    pub const fn all_zeros() -> Self {
        Self::new(false)
    }

    /// Constructs a DSLX-style `u1` value from a Rust boolean.
    pub const fn new(value: bool) -> Self {
        Self(BitsInU8::<1>::from_raw_bits(value as u64))
    }

    /// Constructs a DSLX-style `u1` value from raw ABI bits.
    pub const fn from_raw_bits(value: u64) -> Self {
        Self(BitsInU8::<1>::from_raw_bits(value))
    }

    /// Returns the value as a Rust boolean.
    pub const fn to_bool(self) -> bool {
        self.0.get() != 0
    }

    /// Returns the raw ABI bits widened to `u64`.
    pub const fn raw_bits(self) -> u64 {
        self.0.to_u64()
    }
}

impl AllZeros for Bool {
    fn all_zeros() -> Self {
        Self::all_zeros()
    }
}

impl From<bool> for Bool {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

impl TryFrom<u64> for Bool {
    type Error = RunError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(Self(BitsInU8::<1>::try_from(value)?))
    }
}

define_public_bits_wrappers!(
    UnsignedBitsInU8,
    SignedBitsInU8,
    BitsInU8,
    u8,
    i8,
    to_u8,
    to_i8,
    8
);
define_public_bits_wrappers!(
    UnsignedBitsInU16,
    SignedBitsInU16,
    BitsInU16,
    u16,
    i16,
    to_u16,
    to_i16,
    16
);
define_public_bits_wrappers!(
    UnsignedBitsInU32,
    SignedBitsInU32,
    BitsInU32,
    u32,
    i32,
    to_u32,
    to_i32,
    32
);
define_public_bits_wrappers!(
    UnsignedBitsInU64,
    SignedBitsInU64,
    BitsInU64,
    u64,
    i64,
    to_u64,
    to_i64,
    64
);

/// Native least-significant-first limb storage for a bitvector wider than 64
/// bits.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideBits<const BIT_COUNT: usize, const LIMB_COUNT: usize>([u64; LIMB_COUNT]);

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> WideBits<BIT_COUNT, LIMB_COUNT> {
    /// Constructs the canonical all-zero bitvector value.
    pub fn all_zeros() -> Self {
        Self::from_limbs([0; LIMB_COUNT])
            .unwrap_or_else(|error| panic!("invalid wide bits type: {error}"))
    }

    fn validate_layout() -> Result<(), RunError> {
        if BIT_COUNT <= 64 || LIMB_COUNT != BIT_COUNT.div_ceil(64) {
            Err(RunError(format!(
                "bits[{BIT_COUNT}] cannot use {LIMB_COUNT} native wide limb(s)"
            )))
        } else {
            Ok(())
        }
    }

    fn high_mask() -> u64 {
        let high_width = BIT_COUNT % 64;
        if high_width == 0 {
            u64::MAX
        } else {
            (1u64 << high_width) - 1
        }
    }

    fn logical_byte_count() -> usize {
        BIT_COUNT.div_ceil(8)
    }

    /// Constructs a canonical wide bitvector, rejecting excess high bits.
    pub fn from_limbs(limbs: [u64; LIMB_COUNT]) -> Result<Self, RunError> {
        Self::validate_layout()?;
        if limbs[LIMB_COUNT - 1] & !Self::high_mask() != 0 {
            Err(RunError(format!(
                "high limb does not fit in bits[{BIT_COUNT}]"
            )))
        } else {
            Ok(Self(limbs))
        }
    }

    /// Constructs a canonical wide bitvector from its logical
    /// least-significant-byte-first representation.
    pub fn from_little_endian_bytes(bytes: &[u8]) -> Result<Self, RunError> {
        Self::validate_layout()?;
        let expected_byte_count = Self::logical_byte_count();
        if bytes.len() != expected_byte_count {
            return Err(RunError(format!(
                "bits[{BIT_COUNT}] requires {expected_byte_count} little-endian bytes, got {}",
                bytes.len()
            )));
        }
        let mut limbs = [0; LIMB_COUNT];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks(size_of::<u64>())) {
            let mut limb_bytes = [0; size_of::<u64>()];
            limb_bytes[..chunk.len()].copy_from_slice(chunk);
            *limb = u64::from_le_bytes(limb_bytes);
        }
        Self::from_limbs(limbs)
    }

    /// Returns the logical least-significant-byte-first representation.
    pub fn to_little_endian_bytes<const BYTE_COUNT: usize>(
        &self,
    ) -> Result<[u8; BYTE_COUNT], RunError> {
        Self::validate_layout()?;
        let expected_byte_count = Self::logical_byte_count();
        if BYTE_COUNT != expected_byte_count {
            return Err(RunError(format!(
                "bits[{BIT_COUNT}] requires {expected_byte_count} little-endian bytes, got {BYTE_COUNT}"
            )));
        }
        let mut bytes = [0; BYTE_COUNT];
        for (index, limb) in self.0.iter().enumerate() {
            let start = index * size_of::<u64>();
            if start == BYTE_COUNT {
                break;
            }
            let copy_count = size_of::<u64>().min(BYTE_COUNT - start);
            bytes[start..start + copy_count].copy_from_slice(&limb.to_le_bytes()[..copy_count]);
        }
        Ok(bytes)
    }

    /// Returns the least-significant-first limb representation.
    pub const fn limbs(&self) -> &[u64; LIMB_COUNT] {
        &self.0
    }
}

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> AllZeros for WideBits<BIT_COUNT, LIMB_COUNT> {
    fn all_zeros() -> Self {
        Self::all_zeros()
    }
}

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> Default for WideBits<BIT_COUNT, LIMB_COUNT> {
    fn default() -> Self {
        Self([0; LIMB_COUNT])
    }
}

/// Public unsigned DSLX-style wrapper for a wide native bitvector.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsignedWideBits<const BIT_COUNT: usize, const LIMB_COUNT: usize>(
    WideBits<BIT_COUNT, LIMB_COUNT>,
);

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> UnsignedWideBits<BIT_COUNT, LIMB_COUNT> {
    /// Constructs the canonical all-zero bitvector value.
    pub fn all_zeros() -> Self {
        Self(WideBits::all_zeros())
    }

    /// Constructs a canonical wide unsigned bitvector, rejecting excess high
    /// bits.
    pub fn from_limbs(limbs: [u64; LIMB_COUNT]) -> Result<Self, RunError> {
        Ok(Self(WideBits::<BIT_COUNT, LIMB_COUNT>::from_limbs(limbs)?))
    }

    /// Constructs a canonical wide unsigned bitvector from its logical
    /// least-significant-byte-first representation.
    pub fn from_little_endian_bytes(bytes: &[u8]) -> Result<Self, RunError> {
        Ok(Self(
            WideBits::<BIT_COUNT, LIMB_COUNT>::from_little_endian_bytes(bytes)?,
        ))
    }

    /// Returns the logical least-significant-byte-first representation.
    pub fn to_little_endian_bytes<const BYTE_COUNT: usize>(
        &self,
    ) -> Result<[u8; BYTE_COUNT], RunError> {
        self.0.to_little_endian_bytes()
    }

    /// Returns the least-significant-first raw ABI limb representation.
    pub const fn limbs(&self) -> &[u64; LIMB_COUNT] {
        self.0.limbs()
    }

    /// Returns the unsigned value as a big integer.
    pub fn to_biguint(&self) -> BigUint {
        let mut bytes = Vec::with_capacity(LIMB_COUNT * std::mem::size_of::<u64>());
        for limb in self.limbs() {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }
}

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> AllZeros
    for UnsignedWideBits<BIT_COUNT, LIMB_COUNT>
{
    fn all_zeros() -> Self {
        Self::all_zeros()
    }
}

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> Default
    for UnsignedWideBits<BIT_COUNT, LIMB_COUNT>
{
    fn default() -> Self {
        Self(WideBits::default())
    }
}

/// Public signed DSLX-style wrapper for a wide native bitvector.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedWideBits<const BIT_COUNT: usize, const LIMB_COUNT: usize>(
    WideBits<BIT_COUNT, LIMB_COUNT>,
);

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> SignedWideBits<BIT_COUNT, LIMB_COUNT> {
    /// Constructs the canonical all-zero bitvector value.
    pub fn all_zeros() -> Self {
        Self(WideBits::all_zeros())
    }

    /// Constructs a canonical wide signed bitvector from raw ABI limbs.
    pub fn from_limbs(limbs: [u64; LIMB_COUNT]) -> Result<Self, RunError> {
        Ok(Self(WideBits::<BIT_COUNT, LIMB_COUNT>::from_limbs(limbs)?))
    }

    /// Constructs a canonical wide signed bitvector from its logical raw-bit
    /// least-significant-byte-first representation.
    pub fn from_little_endian_bytes(bytes: &[u8]) -> Result<Self, RunError> {
        Ok(Self(
            WideBits::<BIT_COUNT, LIMB_COUNT>::from_little_endian_bytes(bytes)?,
        ))
    }

    /// Returns the logical raw-bit least-significant-byte-first
    /// representation.
    pub fn to_little_endian_bytes<const BYTE_COUNT: usize>(
        &self,
    ) -> Result<[u8; BYTE_COUNT], RunError> {
        self.0.to_little_endian_bytes()
    }

    /// Returns the least-significant-first raw ABI limb representation.
    pub const fn limbs(&self) -> &[u64; LIMB_COUNT] {
        self.0.limbs()
    }

    /// Returns the sign-extended value as a big integer.
    pub fn to_bigint(&self) -> BigInt {
        let unsigned = UnsignedWideBits::<BIT_COUNT, LIMB_COUNT>(self.0).to_biguint();
        if BIT_COUNT == 0 || !unsigned.bit((BIT_COUNT - 1) as u64) {
            BigInt::from_biguint(Sign::Plus, unsigned)
        } else {
            BigInt::from_biguint(Sign::Plus, unsigned) - (BigInt::from(1u8) << BIT_COUNT)
        }
    }
}

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> AllZeros
    for SignedWideBits<BIT_COUNT, LIMB_COUNT>
{
    fn all_zeros() -> Self {
        Self::all_zeros()
    }
}

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> Default
    for SignedWideBits<BIT_COUNT, LIMB_COUNT>
{
    fn default() -> Self {
        Self(WideBits::default())
    }
}

/// Zero-sized native representation of a PIR token value.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Token;

impl Token {
    /// Constructs the sole token representation.
    pub const fn all_zeros() -> Self {
        Self
    }
}

impl AllZeros for Token {
    fn all_zeros() -> Self {
        Self::all_zeros()
    }
}

/// Kind of observable PIR event described by an event site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Assert,
    Assumption(AssumptionFailureKind),
    Cover,
    Trace,
}

/// Native data description sufficient for immediate trace-value decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceValueLayout {
    Bits {
        bit_count: usize,
        byte_count: usize,
    },
    WideBits {
        bit_count: usize,
        limb_count: usize,
    },
    Array {
        element: Box<TraceValueLayout>,
        element_count: usize,
    },
    Tuple {
        fields: Vec<TraceTupleFieldLayout>,
        byte_count: usize,
    },
    Token,
}

/// Operation implemented by [`xlsynth_pir_runtime_wide_binop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WideBinaryOp {
    Umul = 0,
    Smul = 1,
    Udiv = 2,
    Sdiv = 3,
    Umod = 4,
    Smod = 5,
    Shll = 6,
    Shrl = 7,
    Shra = 8,
}

impl WideBinaryOp {
    fn from_abi(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Umul,
            1 => Self::Smul,
            2 => Self::Udiv,
            3 => Self::Sdiv,
            4 => Self::Umod,
            5 => Self::Smod,
            6 => Self::Shll,
            7 => Self::Shrl,
            8 => Self::Shra,
            _ => return None,
        })
    }
}

/// Operation implemented by [`xlsynth_pir_runtime_wide_unary_op`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WideUnaryOp {
    OneHot = 0,
    Encode = 1,
    Decode = 2,
    ExtPrioEncode = 3,
    ExtClz = 4,
    ExtNormalizeLeft = 5,
    ExtMaskLow = 6,
}

impl WideUnaryOp {
    fn from_abi(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::OneHot,
            1 => Self::Encode,
            2 => Self::Decode,
            3 => Self::ExtPrioEncode,
            4 => Self::ExtClz,
            5 => Self::ExtNormalizeLeft,
            6 => Self::ExtMaskLow,
            _ => return None,
        })
    }
}

/// Description of one tuple field supplied as a trace operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceTupleFieldLayout {
    pub layout: Box<TraceValueLayout>,
    pub offset: usize,
}

/// Static information attached to one observable node in compiled code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventSiteMetadata {
    pub node_text_id: usize,
    pub kind: EventKind,
    pub label: Option<String>,
    pub message: Option<String>,
    pub format: Option<String>,
    pub verbosity: i64,
    pub operand_layouts: Vec<TraceValueLayout>,
}

/// Runtime metadata for all observable sites in one compiled function.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompiledFunctionMetadata {
    pub event_sites: Vec<EventSiteMetadata>,
}

/// A failed compiled assertion, resolved to its source-site metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionFailure {
    pub node_text_id: usize,
    pub message: String,
    pub label: String,
}

/// A failed contract asserted by an `assumed_in_bounds` array operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AssumptionFailureKind {
    ArrayIndexOutOfBounds,
    ArrayUpdateOutOfBounds,
}

/// One failed assumption observed while executing compiled code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssumptionFailure {
    pub node_text_id: usize,
    pub kind: AssumptionFailureKind,
}

/// One emitted compiled trace statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceMessage {
    pub node_text_id: usize,
    pub message: String,
    pub verbosity: i64,
}

/// Accumulated execution count for a compiled `cover` site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverCount {
    pub node_text_id: usize,
    pub label: String,
    pub count: u64,
}

/// Rust-owned observable results recorded while executing compiled code.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionResult {
    pub assertion_failures: Vec<AssertionFailure>,
    pub assumption_failures: Vec<AssumptionFailure>,
    pub trace_messages: Vec<TraceMessage>,
    pub cover_counts: Vec<CoverCount>,
}

/// Runtime options controlling which observable events are collected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionOptions {
    pub trace_verbosity: Option<i64>,
    pub collect_covers: bool,
}

impl ExecutionOptions {
    /// Disables trace and cover collection while still recording failures.
    pub const NO_EVENTS: Self = Self {
        trace_verbosity: None,
        collect_covers: false,
    };

    /// Collects covers and traces whose site verbosity is at most `verbosity`.
    pub const fn new(trace_verbosity: Option<i64>, collect_covers: bool) -> Self {
        Self {
            trace_verbosity,
            collect_covers,
        }
    }

    /// Collects all traces and all covers.
    pub const fn collect_all() -> Self {
        Self {
            trace_verbosity: Some(i64::MAX),
            collect_covers: true,
        }
    }
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self::NO_EVENTS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceFormatPreference {
    Default,
    UnsignedDecimal,
    SignedDecimal,
    PlainHex,
    ZeroPaddedHex,
    Hex,
    PlainBinary,
    ZeroPaddedBinary,
    Binary,
}

const TRACE_FORMAT_SPECIFIERS: [(&str, TraceFormatPreference); 9] = [
    ("{}", TraceFormatPreference::Default),
    ("{:u}", TraceFormatPreference::UnsignedDecimal),
    ("{:d}", TraceFormatPreference::SignedDecimal),
    ("{:x}", TraceFormatPreference::PlainHex),
    ("{:0x}", TraceFormatPreference::ZeroPaddedHex),
    ("{:#x}", TraceFormatPreference::Hex),
    ("{:b}", TraceFormatPreference::PlainBinary),
    ("{:0b}", TraceFormatPreference::ZeroPaddedBinary),
    ("{:#b}", TraceFormatPreference::Binary),
];

struct ContextState {
    metadata: *const CompiledFunctionMetadata,
    options: ExecutionOptions,
    assertion_failures: Vec<AssertionFailure>,
    assumption_failures: Vec<AssumptionFailure>,
    trace_messages: Vec<TraceMessage>,
    event_counts: Option<Vec<u64>>,
}

/// Rust-owned event collector used for one or more compiled executions.
///
/// Cover counts accumulate until [`Self::clear`] is called. Assertion,
/// assumption, and trace results also accumulate, permitting callers to
/// consume a batch of invocations through one context.
pub struct ExecutionContext<'metadata> {
    state: Box<ContextState>,
    marker: PhantomData<&'metadata CompiledFunctionMetadata>,
}

impl<'metadata> ExecutionContext<'metadata> {
    /// Creates an empty collector for the supplied function metadata.
    pub fn new(metadata: &'metadata CompiledFunctionMetadata) -> Self {
        Self::new_with_options(metadata, ExecutionOptions::default())
    }

    /// Creates an empty collector with explicit event collection options.
    pub fn new_with_options(
        metadata: &'metadata CompiledFunctionMetadata,
        options: ExecutionOptions,
    ) -> Self {
        let event_counts = options
            .collect_covers
            .then(|| vec![0; metadata.event_sites.len()]);
        Self {
            state: Box::new(ContextState {
                metadata,
                options,
                assertion_failures: Vec::new(),
                assumption_failures: Vec::new(),
                trace_messages: Vec::new(),
                event_counts,
            }),
            marker: PhantomData,
        }
    }

    /// Returns an opaque ABI object valid while this context is mutably
    /// borrowed.
    pub fn raw_context(&mut self) -> RawExecutionContext {
        RawExecutionContext {
            private_state: ptr::from_mut(self.state.as_mut()).cast(),
        }
    }

    /// Resolves all currently recorded events into ordinary Rust values.
    pub fn result(&self) -> ExecutionResult {
        let metadata = self.metadata();
        let cover_counts = self
            .state
            .event_counts
            .as_ref()
            .map(|event_counts| {
                metadata
                    .event_sites
                    .iter()
                    .zip(event_counts)
                    .filter(|(site, _)| site.kind == EventKind::Cover)
                    .map(|(site, count)| CoverCount {
                        node_text_id: site.node_text_id,
                        label: site.label.clone().unwrap_or_default(),
                        count: *count,
                    })
                    .collect()
            })
            .unwrap_or_default();
        ExecutionResult {
            assertion_failures: self.state.assertion_failures.clone(),
            assumption_failures: self.state.assumption_failures.clone(),
            trace_messages: self.state.trace_messages.clone(),
            cover_counts,
        }
    }

    /// Returns currently recorded assertion failures without cloning.
    pub fn assertion_failures(&self) -> &[AssertionFailure] {
        &self.state.assertion_failures
    }

    /// Returns currently recorded assumption failures without cloning.
    pub fn assumption_failures(&self) -> &[AssumptionFailure] {
        &self.state.assumption_failures
    }

    /// Clears all event records and accumulated cover counters.
    pub fn clear(&mut self) {
        self.clear_with_options(self.state.options);
    }

    /// Clears all event records and switches to the supplied collection
    /// options.
    pub fn clear_with_options(&mut self, options: ExecutionOptions) {
        self.state.assertion_failures.clear();
        self.state.assumption_failures.clear();
        self.state.trace_messages.clear();
        self.state.options = options;
        if options.collect_covers {
            match &mut self.state.event_counts {
                Some(event_counts) => event_counts.fill(0),
                None => {
                    let site_count = self.metadata().event_sites.len();
                    self.state.event_counts = Some(vec![0; site_count]);
                }
            }
        } else {
            self.state.event_counts = None;
        }
    }

    fn metadata(&self) -> &CompiledFunctionMetadata {
        // SAFETY: the context's lifetime guarantees metadata remains alive.
        unsafe { &*self.state.metadata }
    }
}

unsafe fn state_from_raw<'a>(context: *mut RawExecutionContext) -> &'a mut ContextState {
    assert!(
        !context.is_null(),
        "compiled PIR callback requires an execution context"
    );
    // SAFETY: generated entrypoints receive a `RawExecutionContext` produced
    // by `ExecutionContext::raw_context` for the duration of the call.
    unsafe {
        (*context)
            .private_state
            .cast::<ContextState>()
            .as_mut()
            .expect("compiled PIR callback received an invalid execution context")
    }
}

fn site(state: &ContextState, site_id: u32, kind: EventKind) -> Option<&EventSiteMetadata> {
    // SAFETY: the owning `ExecutionContext` keeps metadata alive.
    let metadata = unsafe { state.metadata.as_ref()? };
    let site = metadata.event_sites.get(site_id as usize)?;
    (site.kind == kind).then_some(site)
}

/// Records a failed assertion from generated code.
///
/// # Safety
///
/// `context` must point to an active raw context created by
/// [`ExecutionContext::raw_context`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_record_assert(
    context: *mut RawExecutionContext,
    site_id: u32,
) {
    // SAFETY: forwarded from the caller's ABI contract.
    let state = unsafe { state_from_raw(context) };
    let Some(site) = site(state, site_id, EventKind::Assert).cloned() else {
        return;
    };
    state.assertion_failures.push(AssertionFailure {
        node_text_id: site.node_text_id,
        message: site.message.unwrap_or_default(),
        label: site.label.unwrap_or_default(),
    });
}

/// Records a failed `assumed_in_bounds` contract from generated code.
///
/// # Safety
///
/// `context` must point to an active raw context created by
/// [`ExecutionContext::raw_context`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_record_assumption_failure(
    context: *mut RawExecutionContext,
    site_id: u32,
) {
    // SAFETY: forwarded from the caller's ABI contract.
    let state = unsafe { state_from_raw(context) };
    // SAFETY: the owning `ExecutionContext` keeps metadata alive.
    let Some(site) = (unsafe { state.metadata.as_ref() })
        .and_then(|metadata| metadata.event_sites.get(site_id as usize))
    else {
        return;
    };
    let EventKind::Assumption(kind) = site.kind else {
        return;
    };
    state.assumption_failures.push(AssumptionFailure {
        node_text_id: site.node_text_id,
        kind,
    });
}

/// Records one active cover occurrence from generated code.
///
/// # Safety
///
/// `context` must point to an active raw context created by
/// [`ExecutionContext::raw_context`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_record_cover(context: *mut RawExecutionContext, site_id: u32) {
    // SAFETY: forwarded from the caller's ABI contract.
    let state = unsafe { state_from_raw(context) };
    if state.event_counts.is_none() || site(state, site_id, EventKind::Cover).is_none() {
        return;
    }
    if let Some(count) = state
        .event_counts
        .as_mut()
        .and_then(|event_counts| event_counts.get_mut(site_id as usize))
    {
        *count = count.saturating_add(1);
    }
}

/// Records and formats one active trace occurrence from generated code.
///
/// # Safety
///
/// `context` must point to an active raw context created by
/// [`ExecutionContext::raw_context`]. Each operand pointer must describe
/// readable native storage matching the corresponding site's operand layout
/// for the duration of this callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_record_trace(
    context: *mut RawExecutionContext,
    site_id: u32,
    operand_ptrs: *const *const u8,
) {
    // SAFETY: forwarded from the caller's ABI contract.
    let state = unsafe { state_from_raw(context) };
    let Some(max_verbosity) = state.options.trace_verbosity else {
        return;
    };
    let Some(site) = site(state, site_id, EventKind::Trace) else {
        return;
    };
    if site.verbosity > max_verbosity {
        return;
    }
    if !site.operand_layouts.is_empty() && operand_ptrs.is_null() {
        return;
    }
    state.trace_messages.push(TraceMessage {
        node_text_id: site.node_text_id,
        // SAFETY: the generated caller supplies one pointer per metadata operand.
        message: unsafe {
            format_trace_message(
                site.format.as_deref().unwrap_or(""),
                &site.operand_layouts,
                operand_ptrs,
            )
        },
        verbosity: site.verbosity,
    });
}

fn bit_mask(bit_count: usize) -> BigUint {
    if bit_count == 0 {
        BigUint::from(0u8)
    } else {
        (BigUint::from(1u8) << bit_count) - BigUint::from(1u8)
    }
}

fn truncate_unsigned(value: BigUint, bit_count: usize) -> BigUint {
    value & bit_mask(bit_count)
}

fn as_signed(value: BigUint, bit_count: usize) -> BigInt {
    if bit_count != 0 && (&value & (BigUint::from(1u8) << (bit_count - 1))) != BigUint::from(0u8) {
        BigInt::from_biguint(Sign::Plus, value)
            - BigInt::from_biguint(Sign::Plus, BigUint::from(1u8) << bit_count)
    } else {
        BigInt::from_biguint(Sign::Plus, value)
    }
}

fn truncate_signed(value: BigInt, bit_count: usize) -> BigUint {
    if bit_count == 0 {
        return BigUint::from(0u8);
    }
    let modulus = BigInt::from_biguint(Sign::Plus, BigUint::from(1u8) << bit_count);
    let mut reduced = value % &modulus;
    if reduced.sign() == Sign::Minus {
        reduced += &modulus;
    }
    let (_, bytes) = reduced.to_bytes_le();
    BigUint::from_bytes_le(&bytes)
}

fn bounded_shift_amount(value: &BigUint, bit_count: usize) -> Option<usize> {
    let digits = value.to_u64_digits();
    if digits.len() > 1 {
        return None;
    }
    let amount = digits.first().copied().unwrap_or(0);
    usize::try_from(amount)
        .ok()
        .filter(|amount| *amount < bit_count)
}

fn top_limb_mask(bit_count: usize) -> u64 {
    let used_bits = bit_count % u64::BITS as usize;
    if used_bits == 0 {
        u64::MAX
    } else {
        (1u64 << used_bits) - 1
    }
}

/// Reads one logical limb, returning zero outside the source value.
///
/// # Safety
///
/// `limbs` must be readable for `bit_count.div_ceil(64)` native `u64` values.
unsafe fn read_unsigned_limb(limbs: *const u64, bit_count: usize, index: usize) -> u64 {
    let limb_count = bit_count.div_ceil(u64::BITS as usize);
    if index >= limb_count {
        return 0;
    }
    // SAFETY: the bounds check above enforces the pointer contract.
    let value = unsafe { limbs.add(index).read() };
    if index + 1 == limb_count {
        value & top_limb_mask(bit_count)
    } else {
        value
    }
}

/// Returns the signed extension limb for one logical source limb.
///
/// # Safety
///
/// `limbs` must be readable for `bit_count.div_ceil(64)` native `u64` values.
unsafe fn read_signed_limb(
    limbs: *const u64,
    bit_count: usize,
    index: usize,
    negative: bool,
) -> u64 {
    let limb_count = bit_count.div_ceil(u64::BITS as usize);
    if index >= limb_count {
        return if negative { u64::MAX } else { 0 };
    }
    // SAFETY: the bounds check above enforces the pointer contract.
    let value = unsafe { limbs.add(index).read() };
    if index + 1 != limb_count {
        return value;
    }
    let mask = top_limb_mask(bit_count);
    if negative {
        value | !mask
    } else {
        value & mask
    }
}

/// Reads an unsigned shift amount if it is representable and in range.
///
/// # Safety
///
/// `limbs` must be readable for `bit_count.div_ceil(64)` native `u64` values.
unsafe fn read_bounded_shift_amount(
    limbs: *const u64,
    bit_count: usize,
    shifted_bit_count: usize,
) -> Option<usize> {
    let limb_count = bit_count.div_ceil(u64::BITS as usize);
    let low = if limb_count == 0 {
        0
    } else {
        // SAFETY: a nonzero limb count guarantees logical limb zero exists.
        unsafe { read_unsigned_limb(limbs, bit_count, 0) }
    };
    for index in 1..limb_count {
        // SAFETY: `index` is within the logical limb count.
        if unsafe { read_unsigned_limb(limbs, bit_count, index) } != 0 {
            return None;
        }
    }
    usize::try_from(low)
        .ok()
        .filter(|amount| *amount < shifted_bit_count)
}

/// Executes a logical or arithmetic shift directly over fixed native limbs.
///
/// # Safety
///
/// Pointer requirements match [`xlsynth_pir_runtime_wide_binop`].
unsafe fn shift_wide_limbs(
    dst: *mut u64,
    dst_bit_count: usize,
    lhs: *const u64,
    lhs_bit_count: usize,
    rhs: *const u64,
    rhs_bit_count: usize,
    operation: WideBinaryOp,
) {
    debug_assert!(matches!(
        operation,
        WideBinaryOp::Shll | WideBinaryOp::Shrl | WideBinaryOp::Shra
    ));
    let negative = if operation == WideBinaryOp::Shra && lhs_bit_count != 0 {
        let sign_index = lhs_bit_count - 1;
        // SAFETY: a nonzero width guarantees the sign-bit limb exists.
        (unsafe { lhs.add(sign_index / u64::BITS as usize).read() }
            & (1u64 << (sign_index % u64::BITS as usize)))
            != 0
    } else {
        false
    };
    // SAFETY: forwarded from this helper's pointer contract.
    let amount = unsafe { read_bounded_shift_amount(rhs, rhs_bit_count, lhs_bit_count) };
    let dst_limb_count = dst_bit_count.div_ceil(u64::BITS as usize);
    let Some(amount) = amount else {
        let fill = if operation == WideBinaryOp::Shra && negative {
            u64::MAX
        } else {
            0
        };
        for index in 0..dst_limb_count {
            let value = if index + 1 == dst_limb_count {
                fill & top_limb_mask(dst_bit_count)
            } else {
                fill
            };
            // SAFETY: `index` is within the destination's logical limb count.
            unsafe { dst.add(index).write(value) };
        }
        return;
    };

    let limb_shift = amount / u64::BITS as usize;
    let bit_shift = amount % u64::BITS as usize;
    for dst_index in 0..dst_limb_count {
        let value = if operation == WideBinaryOp::Shll {
            let Some(source_index) = dst_index.checked_sub(limb_shift) else {
                // This destination limb is below the shifted source value.
                unsafe { dst.add(dst_index).write(0) };
                continue;
            };
            // SAFETY: out-of-range source indices are converted to zero.
            let low = unsafe { read_unsigned_limb(lhs, lhs_bit_count, source_index) };
            if bit_shift == 0 {
                low
            } else {
                let high = source_index
                    .checked_sub(1)
                    .map(|index| {
                        // SAFETY: out-of-range source indices are converted to zero.
                        unsafe { read_unsigned_limb(lhs, lhs_bit_count, index) }
                    })
                    .unwrap_or(0);
                (low << bit_shift) | (high >> (u64::BITS as usize - bit_shift))
            }
        } else {
            let source_index = dst_index + limb_shift;
            let read_limb = |index| {
                if operation == WideBinaryOp::Shra {
                    // SAFETY: out-of-range source indices are sign-extended.
                    unsafe { read_signed_limb(lhs, lhs_bit_count, index, negative) }
                } else {
                    // SAFETY: out-of-range source indices are converted to zero.
                    unsafe { read_unsigned_limb(lhs, lhs_bit_count, index) }
                }
            };
            let low = read_limb(source_index);
            if bit_shift == 0 {
                low
            } else {
                let high = read_limb(source_index + 1);
                (low >> bit_shift) | (high << (u64::BITS as usize - bit_shift))
            }
        };
        let value = if dst_index + 1 == dst_limb_count {
            value & top_limb_mask(dst_bit_count)
        } else {
            value
        };
        // SAFETY: `dst_index` is within the destination's logical limb count.
        unsafe { dst.add(dst_index).write(value) };
    }
}

/// Reads a fixed-width value from least-significant-first native `u64` limbs.
///
/// # Safety
///
/// `limbs` must be readable for `bit_count.div_ceil(64)` native `u64` values.
unsafe fn read_wide_bits(limbs: *const u64, bit_count: usize) -> BigUint {
    let limb_count = bit_count.div_ceil(64);
    let mut bytes = Vec::with_capacity(limb_count * std::mem::size_of::<u64>());
    for index in 0..limb_count {
        // SAFETY: forwarded from this function's pointer contract.
        let limb = unsafe { limbs.add(index).read() };
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    truncate_unsigned(BigUint::from_bytes_le(&bytes), bit_count)
}

/// Writes a fixed-width value to least-significant-first native `u64` limbs.
///
/// # Safety
///
/// `limbs` must be writable for `bit_count.div_ceil(64)` native `u64` values.
unsafe fn write_wide_bits(limbs: *mut u64, bit_count: usize, value: BigUint) {
    let limb_count = bit_count.div_ceil(64);
    let bytes = truncate_unsigned(value, bit_count).to_bytes_le();
    for index in 0..limb_count {
        let mut limb_bytes = [0u8; std::mem::size_of::<u64>()];
        let start = index * std::mem::size_of::<u64>();
        if start < bytes.len() {
            let end = bytes.len().min(start + std::mem::size_of::<u64>());
            limb_bytes[..end - start].copy_from_slice(&bytes[start..end]);
        }
        // SAFETY: forwarded from this function's pointer contract.
        unsafe { limbs.add(index).write(u64::from_le_bytes(limb_bytes)) };
    }
}

fn get_bit(value: &BigUint, index: usize) -> bool {
    value
        .to_u64_digits()
        .get(index / u64::BITS as usize)
        .is_some_and(|limb| limb & (1u64 << (index % u64::BITS as usize)) != 0)
}

fn prioritized_set_bit(value: &BigUint, bit_count: usize, lsb_prio: bool) -> Option<usize> {
    if lsb_prio {
        (0..bit_count).find(|index| get_bit(value, *index))
    } else {
        (0..bit_count).rev().find(|index| get_bit(value, *index))
    }
}

fn leading_zero_count(value: &BigUint, bit_count: usize) -> usize {
    prioritized_set_bit(value, bit_count, /* lsb_prio= */ false)
        .map(|index| bit_count - index - 1)
        .unwrap_or(bit_count)
}

fn mulp_offset(result_width: usize) -> BigUint {
    let low_width = result_width.saturating_sub(2);
    let high_width = result_width - low_width;
    let low_shift = low_width.saturating_sub(1).min(3);
    let low = if low_width == 0 {
        BigUint::from(0u8)
    } else {
        bit_mask(low_width) >> low_shift
    };
    let high = if high_width == 0 {
        BigUint::from(0u8)
    } else {
        bit_mask(high_width.saturating_sub(1)) << low_width
    };
    low | high
}

/// Computes a complex arbitrary-width binary operation over native limb
/// storage.
///
/// Limb arrays are ordered from least- to most-significant limb. The result is
/// truncated to `dst_bit_count`. `dst` may not alias either source.
///
/// # Safety
///
/// Each pointer must be valid for the number of `u64` limbs implied by its
/// supplied bit count and obey the non-aliasing rule above.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_runtime_wide_binop(
    dst: *mut u64,
    dst_bit_count: usize,
    lhs: *const u64,
    lhs_bit_count: usize,
    rhs: *const u64,
    rhs_bit_count: usize,
    operation: u32,
) {
    let Some(operation) = WideBinaryOp::from_abi(operation) else {
        return;
    };
    if matches!(
        operation,
        WideBinaryOp::Shll | WideBinaryOp::Shrl | WideBinaryOp::Shra
    ) {
        // SAFETY: forwarded from this callback's pointer contract.
        unsafe {
            shift_wide_limbs(
                dst,
                dst_bit_count,
                lhs,
                lhs_bit_count,
                rhs,
                rhs_bit_count,
                operation,
            )
        };
        return;
    }
    // SAFETY: forwarded from this callback's pointer contract.
    let lhs_unsigned = unsafe { read_wide_bits(lhs, lhs_bit_count) };
    // SAFETY: forwarded from this callback's pointer contract.
    let rhs_unsigned = unsafe { read_wide_bits(rhs, rhs_bit_count) };
    let result = match operation {
        WideBinaryOp::Umul => truncate_unsigned(lhs_unsigned * rhs_unsigned, dst_bit_count),
        WideBinaryOp::Smul => truncate_signed(
            as_signed(lhs_unsigned, lhs_bit_count) * as_signed(rhs_unsigned, rhs_bit_count),
            dst_bit_count,
        ),
        WideBinaryOp::Udiv => {
            if rhs_unsigned == BigUint::from(0u8) {
                bit_mask(dst_bit_count)
            } else {
                truncate_unsigned(lhs_unsigned / rhs_unsigned, dst_bit_count)
            }
        }
        WideBinaryOp::Umod => {
            if rhs_unsigned == BigUint::from(0u8) {
                BigUint::from(0u8)
            } else {
                truncate_unsigned(lhs_unsigned % rhs_unsigned, dst_bit_count)
            }
        }
        WideBinaryOp::Sdiv | WideBinaryOp::Smod => {
            let lhs_signed = as_signed(lhs_unsigned, lhs_bit_count);
            let rhs_signed = as_signed(rhs_unsigned, rhs_bit_count);
            if dst_bit_count == 0 {
                BigUint::from(0u8)
            } else if rhs_signed == BigInt::from(0u8) {
                if operation == WideBinaryOp::Smod {
                    BigUint::from(0u8)
                } else if lhs_signed.sign() == Sign::Minus {
                    BigUint::from(1u8) << (dst_bit_count - 1)
                } else {
                    (BigUint::from(1u8) << (dst_bit_count - 1)) - BigUint::from(1u8)
                }
            } else if operation == WideBinaryOp::Sdiv {
                truncate_signed(lhs_signed / rhs_signed, dst_bit_count)
            } else {
                truncate_signed(lhs_signed % rhs_signed, dst_bit_count)
            }
        }
        WideBinaryOp::Shll | WideBinaryOp::Shrl | WideBinaryOp::Shra => {
            unreachable!("wide shifts return through the allocation-free limb path")
        }
    };
    // SAFETY: forwarded from this callback's pointer contract.
    unsafe { write_wide_bits(dst, dst_bit_count, result) };
}

/// Computes a zero-filled dynamic slice into native limb storage.
///
/// # Safety
///
/// Pointer requirements match [`xlsynth_pir_runtime_wide_binop`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_runtime_wide_dynamic_bit_slice(
    dst: *mut u64,
    dst_bit_count: usize,
    arg: *const u64,
    arg_bit_count: usize,
    start: *const u64,
    start_bit_count: usize,
) {
    // SAFETY: forwarded from this callback's pointer contract.
    let arg = unsafe { read_wide_bits(arg, arg_bit_count) };
    // SAFETY: forwarded from this callback's pointer contract.
    let start = unsafe { read_wide_bits(start, start_bit_count) };
    let result = bounded_shift_amount(&start, arg_bit_count)
        .map(|amount| truncate_unsigned(arg >> amount, dst_bit_count))
        .unwrap_or_else(|| BigUint::from(0u8));
    // SAFETY: forwarded from this callback's pointer contract.
    unsafe { write_wide_bits(dst, dst_bit_count, result) };
}

/// Inserts a dynamically positioned low-to-high slice into native limb storage.
///
/// # Safety
///
/// Pointer requirements match [`xlsynth_pir_runtime_wide_binop`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_runtime_wide_bit_slice_update(
    dst: *mut u64,
    dst_bit_count: usize,
    arg: *const u64,
    arg_bit_count: usize,
    start: *const u64,
    start_bit_count: usize,
    update: *const u64,
    update_bit_count: usize,
) {
    // SAFETY: forwarded from this callback's pointer contract.
    let arg = unsafe { read_wide_bits(arg, arg_bit_count) };
    // SAFETY: forwarded from this callback's pointer contract.
    let start = unsafe { read_wide_bits(start, start_bit_count) };
    // SAFETY: forwarded from this callback's pointer contract.
    let update = unsafe { read_wide_bits(update, update_bit_count) };
    let result = if let Some(start) = bounded_shift_amount(&start, arg_bit_count) {
        let written_width = update_bit_count.min(arg_bit_count - start);
        let written_mask = bit_mask(written_width) << start;
        let retained = &arg & (&bit_mask(arg_bit_count) ^ &written_mask);
        retained | ((update & bit_mask(written_width)) << start)
    } else {
        arg
    };
    // SAFETY: forwarded from this callback's pointer contract.
    unsafe { write_wide_bits(dst, dst_bit_count, result) };
}

/// Computes an arbitrary-width single-operand PIR transform.
///
/// `attribute` is interpreted as `lsb_prio` for `one_hot` and
/// `ext_prio_encode`, and as the static shift/count offset for `ext_clz` and
/// `ext_normalize_left`.
///
/// # Safety
///
/// Pointer requirements match [`xlsynth_pir_runtime_wide_binop`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_runtime_wide_unary_op(
    dst: *mut u64,
    dst_bit_count: usize,
    arg: *const u64,
    arg_bit_count: usize,
    operation: u32,
    attribute: usize,
) {
    let Some(operation) = WideUnaryOp::from_abi(operation) else {
        return;
    };
    // SAFETY: forwarded from this callback's pointer contract.
    let arg = unsafe { read_wide_bits(arg, arg_bit_count) };
    let result = match operation {
        WideUnaryOp::OneHot => {
            let selected =
                prioritized_set_bit(&arg, arg_bit_count, attribute != 0).unwrap_or(arg_bit_count);
            BigUint::from(1u8) << selected
        }
        WideUnaryOp::Encode => {
            let mut result = 0usize;
            for index in 0..arg_bit_count {
                if get_bit(&arg, index) {
                    result |= index;
                }
            }
            BigUint::from(result)
        }
        WideUnaryOp::Decode => bounded_shift_amount(&arg, dst_bit_count)
            .map(|amount| BigUint::from(1u8) << amount)
            .unwrap_or_else(|| BigUint::from(0u8)),
        WideUnaryOp::ExtPrioEncode => BigUint::from(
            prioritized_set_bit(&arg, arg_bit_count, attribute != 0).unwrap_or(arg_bit_count),
        ),
        WideUnaryOp::ExtClz => BigUint::from(leading_zero_count(&arg, arg_bit_count) + attribute),
        WideUnaryOp::ExtNormalizeLeft => {
            let shift = leading_zero_count(&arg, arg_bit_count).saturating_add(attribute);
            if shift >= dst_bit_count {
                BigUint::from(0u8)
            } else {
                truncate_unsigned(arg << shift, dst_bit_count)
            }
        }
        WideUnaryOp::ExtMaskLow => {
            if arg >= BigUint::from(dst_bit_count) {
                bit_mask(dst_bit_count)
            } else {
                let count = arg.to_u64_digits().first().copied().unwrap_or(0) as usize;
                bit_mask(count)
            }
        }
    };
    // SAFETY: forwarded from this callback's pointer contract.
    unsafe { write_wide_bits(dst, dst_bit_count, result) };
}

/// Computes the deterministic pair used for arbitrary-width `umulp`/`smulp`.
///
/// # Safety
///
/// Pointer requirements match [`xlsynth_pir_runtime_wide_binop`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xlsynth_pir_runtime_wide_mulp(
    offset_dst: *mut u64,
    residual_dst: *mut u64,
    dst_bit_count: usize,
    lhs: *const u64,
    lhs_bit_count: usize,
    rhs: *const u64,
    rhs_bit_count: usize,
    signed: u32,
) {
    // SAFETY: forwarded from this callback's pointer contract.
    let lhs = unsafe { read_wide_bits(lhs, lhs_bit_count) };
    // SAFETY: forwarded from this callback's pointer contract.
    let rhs = unsafe { read_wide_bits(rhs, rhs_bit_count) };
    let product = if signed != 0 {
        truncate_signed(
            as_signed(lhs, lhs_bit_count) * as_signed(rhs, rhs_bit_count),
            dst_bit_count,
        )
    } else {
        truncate_unsigned(lhs * rhs, dst_bit_count)
    };
    let offset = mulp_offset(dst_bit_count);
    let residual = truncate_signed(
        BigInt::from_biguint(Sign::Plus, product)
            - BigInt::from_biguint(Sign::Plus, offset.clone()),
        dst_bit_count,
    );
    // SAFETY: forwarded from this callback's pointer contract.
    unsafe { write_wide_bits(offset_dst, dst_bit_count, offset) };
    // SAFETY: forwarded from this callback's pointer contract.
    unsafe { write_wide_bits(residual_dst, dst_bit_count, residual) };
}

/// Formats a trace message according to the XLS trace-format string syntax.
unsafe fn format_trace_message(
    format: &str,
    layouts: &[TraceValueLayout],
    operand_ptrs: *const *const u8,
) -> String {
    let mut output = String::new();
    let mut offset = 0usize;
    let mut operand_index = 0usize;
    while offset < format.len() {
        let remainder = &format[offset..];
        if remainder.starts_with("{{") || remainder.starts_with("}}") {
            // XLS preserves escaped braces in trace output; Verilog emission
            // performs the collapse to single braces separately.
            output.push_str(&remainder[..2]);
            offset += 2;
            continue;
        }
        if let Some((specifier, preference)) = TRACE_FORMAT_SPECIFIERS
            .iter()
            .find(|(specifier, _)| remainder.starts_with(specifier))
        {
            if let Some(layout) = layouts.get(operand_index) {
                // SAFETY: callback ABI provides one matching pointer per layout.
                let pointer = unsafe { *operand_ptrs.add(operand_index) };
                // SAFETY: callback ABI specifies readable storage matching `layout`.
                output.push_str(&unsafe { format_native_value(pointer, layout, *preference) });
            }
            operand_index += 1;
            offset += specifier.len();
            continue;
        }
        let character = remainder
            .chars()
            .next()
            .expect("offset is within trace format string");
        output.push(character);
        offset += character.len_utf8();
    }
    output
}

/// Formats one caller-owned native value without constructing an XLS value.
unsafe fn format_native_value(
    pointer: *const u8,
    layout: &TraceValueLayout,
    preference: TraceFormatPreference,
) -> String {
    match layout {
        TraceValueLayout::Bits {
            bit_count,
            byte_count,
        } => {
            let mut bytes = vec![0u8; *byte_count];
            if *byte_count != 0 {
                // SAFETY: callback ABI provides native scalar storage of this size.
                unsafe { ptr::copy_nonoverlapping(pointer, bytes.as_mut_ptr(), *byte_count) };
            }
            let value = if cfg!(target_endian = "little") {
                BigUint::from_bytes_le(&bytes)
            } else {
                BigUint::from_bytes_be(&bytes)
            };
            format_trace_bits(value, *bit_count, preference)
        }
        TraceValueLayout::WideBits {
            bit_count,
            limb_count: _,
        } => {
            // SAFETY: callback ABI provides the number of native limbs
            // prescribed by this layout.
            let value = unsafe { read_wide_bits(pointer.cast::<u64>(), *bit_count) };
            format_trace_bits(value, *bit_count, preference)
        }
        TraceValueLayout::Array {
            element,
            element_count,
        } => {
            let elements = (0..*element_count)
                .map(|index| {
                    // SAFETY: each element is within the caller-provided array region.
                    unsafe {
                        format_native_value(
                            pointer.wrapping_add(index * element.byte_count()),
                            element,
                            preference,
                        )
                    }
                })
                .collect::<Vec<_>>();
            format!("[{}]", elements.join(", "))
        }
        TraceValueLayout::Tuple { fields, .. } => {
            let fields = fields
                .iter()
                .map(|field| {
                    // SAFETY: each field offset is prescribed by native tuple metadata.
                    unsafe {
                        format_native_value(
                            pointer.wrapping_add(field.offset),
                            &field.layout,
                            preference,
                        )
                    }
                })
                .collect::<Vec<_>>();
            format!("({})", fields.join(", "))
        }
        TraceValueLayout::Token => "token".to_string(),
    }
}

fn format_trace_bits(
    mut value: BigUint,
    bit_count: usize,
    preference: TraceFormatPreference,
) -> String {
    if bit_count == 0 {
        value = BigUint::from(0u8);
    } else {
        value &= (BigUint::from(1u8) << bit_count) - BigUint::from(1u8);
    }
    match preference {
        TraceFormatPreference::Default => {
            if bit_count <= 64 {
                value.to_str_radix(10)
            } else {
                format_trace_bits(value, bit_count, TraceFormatPreference::Hex)
            }
        }
        TraceFormatPreference::UnsignedDecimal => value.to_str_radix(10),
        TraceFormatPreference::SignedDecimal => {
            if bit_count != 0
                && (&value & (BigUint::from(1u8) << (bit_count - 1))) != BigUint::from(0u8)
            {
                (BigInt::from_biguint(Sign::Plus, value)
                    - BigInt::from_biguint(Sign::Plus, BigUint::from(1u8) << bit_count))
                .to_string()
            } else {
                value.to_str_radix(10)
            }
        }
        TraceFormatPreference::PlainHex => value.to_str_radix(16),
        TraceFormatPreference::ZeroPaddedHex => {
            zero_padded_grouped_digits(&value, bit_count, 4, 16)
        }
        TraceFormatPreference::Hex => {
            format!("0x{}", grouped_digits(&value.to_str_radix(16)))
        }
        TraceFormatPreference::PlainBinary => value.to_str_radix(2),
        TraceFormatPreference::ZeroPaddedBinary => {
            zero_padded_grouped_digits(&value, bit_count, 1, 2)
        }
        TraceFormatPreference::Binary => {
            format!("0b{}", grouped_digits(&value.to_str_radix(2)))
        }
    }
}

fn zero_padded_grouped_digits(
    value: &BigUint,
    bit_count: usize,
    bits_per_digit: usize,
    radix: u32,
) -> String {
    let digit_count = bit_count.div_ceil(bits_per_digit).max(1);
    let digits = format!(
        "{:0>width$}",
        value.to_str_radix(radix),
        width = digit_count
    );
    grouped_digits(&digits)
}

fn grouped_digits(digits: &str) -> String {
    let first_group_width = match digits.len() % 4 {
        0 => 4,
        remainder => remainder,
    };
    let mut result = digits[..first_group_width].to_string();
    for group_start in (first_group_width..digits.len()).step_by(4) {
        result.push('_');
        result.push_str(&digits[group_start..group_start + 4]);
    }
    result
}

impl TraceValueLayout {
    fn byte_count(&self) -> usize {
        match self {
            Self::Bits { byte_count, .. } => *byte_count,
            Self::WideBits { limb_count, .. } => limb_count * std::mem::size_of::<u64>(),
            Self::Array {
                element,
                element_count,
            } => element.byte_count() * element_count,
            Self::Tuple { byte_count, .. } => *byte_count,
            Self::Token => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zeros_recursively_initializes_native_values() {
        let scalar = UnsignedBitsInU16::<12>::all_zeros();
        assert_eq!(scalar.to_u16(), 0);

        let wide = SignedWideBits::<129, 3>::all_zeros();
        assert_eq!(wide.limbs(), &[0, 0, 0]);

        let nested: [[UnsignedBitsInU8<4>; 2]; 3] = AllZeros::all_zeros();
        assert!(nested.iter().flatten().all(|value| value.to_u8() == 0));
    }

    #[test]
    fn native_bits_wrappers_enforce_semantic_widths() {
        let value = BitsInU64::<42>::new((1u64 << 41) | 7);
        assert_eq!(value.to_u64(), (1u64 << 41) | 7);
        assert!(BitsInU64::<42>::try_from(1u64 << 42).is_err());
        assert!(std::panic::catch_unwind(|| BitsInU64::<42>::new(1u64 << 42)).is_err());
        assert_eq!(BitsInU16::<9>::new(0x1ff).get(), 0x1ff);
        assert!(BitsInU8::<9>::try_from(0).is_err());
    }

    #[test]
    fn public_signed_and_unsigned_bits_wrappers_preserve_raw_abi_bits() {
        let unsigned = UnsignedBitsInU8::<4>::new(15);
        assert_eq!(unsigned.to_u8(), 15);
        assert_eq!(unsigned.raw_bits(), 15);
        assert!(UnsignedBitsInU8::<4>::try_from(16).is_err());
        assert!(std::panic::catch_unwind(|| UnsignedBitsInU8::<4>::new(16)).is_err());
        assert!(std::panic::catch_unwind(|| UnsignedBitsInU8::<4>::from_raw_bits(16)).is_err());

        let signed = SignedBitsInU8::<4>::new(-1);
        assert_eq!(signed.to_i8(), -1);
        assert_eq!(signed.raw_bits(), 15);
        assert!(SignedBitsInU8::<4>::try_from(8).is_err());
        assert!(SignedBitsInU8::<4>::try_from(-9).is_err());
        assert!(std::panic::catch_unwind(|| SignedBitsInU8::<4>::new(8)).is_err());
        assert!(std::panic::catch_unwind(|| SignedBitsInU8::<4>::from_raw_bits(16)).is_err());
        assert_eq!(SignedBitsInU16::<9>::from_raw_bits(0x101).to_i16(), -255);

        let wide = SignedWideBits::<65, 2>::from_limbs([u64::MAX, 1]).expect("s65 -1");
        assert_eq!(wide.to_bigint(), BigInt::from(-1));
        assert_eq!(wide.limbs(), &[u64::MAX, 1]);
    }

    #[test]
    fn bool_wrapper_uses_rust_bool_values() {
        let value = Bool::new(true);
        assert!(value.to_bool());
        assert_eq!(value.raw_bits(), 1);
        assert_eq!(Bool::from(false), Bool::all_zeros());
        assert_eq!(Bool::try_from(1_u64).expect("1 fits in u1"), value);
        assert!(Bool::try_from(2_u64).is_err());
        assert!(std::panic::catch_unwind(|| Bool::from_raw_bits(2)).is_err());
    }

    #[test]
    fn public_bits_wrappers_use_narrow_rust_carriers() {
        let unsigned8: u8 = UnsignedBitsInU8::<8>::new(255_u8).to_u8();
        let unsigned16: u16 = UnsignedBitsInU16::<9>::new(0x1ff_u16).to_u16();
        let unsigned32: u32 = UnsignedBitsInU32::<17>::new(0x1ffff_u32).to_u32();
        let unsigned64: u64 = UnsignedBitsInU64::<33>::new(0x1ffffffff_u64).to_u64();
        assert_eq!(
            (unsigned8, unsigned16, unsigned32, unsigned64),
            (255, 0x1ff, 0x1ffff, 0x1ffffffff)
        );

        let signed8: i8 = SignedBitsInU8::<4>::new(-8_i8).to_i8();
        let signed16: i16 = SignedBitsInU16::<9>::new(-256_i16).to_i16();
        let signed32: i32 = SignedBitsInU32::<17>::new(-65_536_i32).to_i32();
        let signed64: i64 = SignedBitsInU64::<33>::new(-4_294_967_296_i64).to_i64();
        assert_eq!(
            (signed8, signed16, signed32, signed64),
            (-8, -256, -65_536, -4_294_967_296)
        );
    }

    #[test]
    fn wide_bits_convert_logical_little_endian_bytes() {
        let bytes = [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0x01];
        let value = WideBits::<65, 2>::from_little_endian_bytes(&bytes)
            .expect("nine bytes encode bits[65]");
        assert_eq!(value.limbs(), &[0x0123_4567_89ab_cdef, 1]);
        let round_trip: [u8; 9] = value
            .to_little_endian_bytes()
            .expect("bits[65] uses nine logical bytes");
        assert_eq!(round_trip, bytes);
        assert!(value.to_little_endian_bytes::<8>().is_err());
        assert!(WideBits::<65, 2>::from_little_endian_bytes(&bytes[..8]).is_err());

        let mut excess_high_bits = bytes;
        excess_high_bits[8] = 2;
        assert!(WideBits::<65, 2>::from_little_endian_bytes(&excess_high_bits).is_err());

        let unsigned = UnsignedWideBits::<65, 2>::from_little_endian_bytes(&bytes)
            .expect("unsigned wrapper accepts logical bytes");
        assert_eq!(
            unsigned
                .to_little_endian_bytes::<9>()
                .expect("unsigned wrapper returns logical bytes"),
            bytes
        );
        let signed = SignedWideBits::<65, 2>::from_little_endian_bytes(&bytes)
            .expect("signed wrapper accepts raw logical bytes");
        assert_eq!(
            signed
                .to_little_endian_bytes::<9>()
                .expect("signed wrapper returns raw logical bytes"),
            bytes
        );
    }

    #[test]
    fn public_bits_wrappers_try_from_widened_integers() {
        assert_eq!(std::mem::size_of::<Bits0>(), 0);
        assert_eq!(std::mem::size_of::<UnsignedBits0>(), 0);
        assert_eq!(std::mem::size_of::<SignedBits0>(), 0);

        let unsigned_zero = UnsignedBits0::try_from(0_u64).expect("0 fits in u0");
        assert_eq!(unsigned_zero.to_u64(), 0);
        assert_eq!(unsigned_zero.raw_bits(), 0);
        assert!(UnsignedBits0::try_from(1_u64).is_err());
        assert!(std::panic::catch_unwind(|| UnsignedBits0::from_raw_bits(1)).is_err());

        let signed_zero = SignedBits0::try_from(0_i64).expect("0 fits in s0");
        assert_eq!(signed_zero.to_i64(), 0);
        assert_eq!(signed_zero.raw_bits(), 0);
        assert!(SignedBits0::try_from(-1_i64).is_err());
        assert!(SignedBits0::try_from(1_i64).is_err());
        assert!(std::panic::catch_unwind(|| SignedBits0::from_raw_bits(1)).is_err());

        assert_eq!(
            UnsignedBitsInU8::<4>::try_from(15_u64)
                .expect("15 fits in u4")
                .to_u8(),
            15
        );
        assert!(UnsignedBitsInU8::<4>::try_from(16_u64).is_err());
        assert!(UnsignedBitsInU8::<8>::try_from(256_u64).is_err());
        assert_eq!(
            UnsignedBitsInU16::<9>::try_from(0x1ff_u64)
                .expect("0x1ff fits in u9")
                .to_u16(),
            0x1ff
        );
        assert_eq!(
            UnsignedBitsInU32::<17>::try_from(0x1ffff_u64)
                .expect("0x1ffff fits in u17")
                .to_u32(),
            0x1ffff
        );
        assert_eq!(
            UnsignedBitsInU64::<33>::try_from(0x1ffffffff_u64)
                .expect("0x1ffffffff fits in u33")
                .to_u64(),
            0x1ffffffff
        );

        assert_eq!(
            SignedBitsInU8::<4>::try_from(-8_i64)
                .expect("-8 fits in s4")
                .to_i8(),
            -8
        );
        assert!(SignedBitsInU8::<4>::try_from(-9_i64).is_err());
        assert!(SignedBitsInU8::<8>::try_from(128_i64).is_err());
        assert_eq!(
            SignedBitsInU16::<9>::try_from(-256_i64)
                .expect("-256 fits in s9")
                .to_i16(),
            -256
        );
        assert_eq!(
            SignedBitsInU32::<17>::try_from(-65_536_i64)
                .expect("-65536 fits in s17")
                .to_i32(),
            -65_536
        );
        assert_eq!(
            SignedBitsInU64::<33>::try_from(-4_294_967_296_i64)
                .expect("-4294967296 fits in s33")
                .to_i64(),
            -4_294_967_296
        );
    }

    #[test]
    fn wide_bits_wrappers_use_lsb_first_limbs_and_reject_high_bits() {
        let value =
            WideBits::<65, 2>::from_limbs([0x0123_4567_89ab_cdef, 1]).expect("canonical value");
        assert_eq!(value.limbs(), &[0x0123_4567_89ab_cdef, 1]);
        assert!(WideBits::<65, 2>::from_limbs([0, 2]).is_err());
        assert!(WideBits::<65, 3>::from_limbs([0, 0, 0]).is_err());
        assert_eq!(std::mem::size_of::<Token>(), 0);
    }

    fn metadata() -> CompiledFunctionMetadata {
        CompiledFunctionMetadata {
            event_sites: vec![
                EventSiteMetadata {
                    node_text_id: 10,
                    kind: EventKind::Cover,
                    label: Some("covered".to_string()),
                    message: None,
                    format: None,
                    verbosity: 0,
                    operand_layouts: Vec::new(),
                },
                EventSiteMetadata {
                    node_text_id: 11,
                    kind: EventKind::Assert,
                    label: Some("assert_label".to_string()),
                    message: Some("failed".to_string()),
                    format: None,
                    verbosity: 0,
                    operand_layouts: Vec::new(),
                },
                EventSiteMetadata {
                    node_text_id: 12,
                    kind: EventKind::Trace,
                    label: None,
                    message: None,
                    format: Some("x={} arr={}".to_string()),
                    verbosity: 1,
                    operand_layouts: vec![
                        TraceValueLayout::Bits {
                            bit_count: 8,
                            byte_count: 1,
                        },
                        TraceValueLayout::Array {
                            element: Box::new(TraceValueLayout::Bits {
                                bit_count: 8,
                                byte_count: 1,
                            }),
                            element_count: 2,
                        },
                    ],
                },
                EventSiteMetadata {
                    node_text_id: 13,
                    kind: EventKind::Assumption(AssumptionFailureKind::ArrayIndexOutOfBounds),
                    label: None,
                    message: None,
                    format: None,
                    verbosity: 0,
                    operand_layouts: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn cover_and_assert_callbacks_collect_rust_owned_results() {
        let metadata = metadata();
        let mut context =
            ExecutionContext::new_with_options(&metadata, ExecutionOptions::collect_all());
        let mut raw = context.raw_context();
        // SAFETY: `raw` points into `context` for these immediate calls.
        unsafe {
            xlsynth_pir_record_cover(&mut raw, 0);
            xlsynth_pir_record_cover(&mut raw, 0);
            xlsynth_pir_record_assert(&mut raw, 1);
            xlsynth_pir_record_assumption_failure(&mut raw, 3);
        }
        let result = context.result();
        assert_eq!(result.cover_counts[0].count, 2);
        assert_eq!(result.cover_counts[0].label, "covered");
        assert_eq!(result.assertion_failures[0].message, "failed");
        assert_eq!(result.assertion_failures[0].label, "assert_label");
        assert_eq!(
            result.assumption_failures,
            vec![AssumptionFailure {
                node_text_id: 13,
                kind: AssumptionFailureKind::ArrayIndexOutOfBounds,
            }]
        );
    }

    #[test]
    fn trace_callback_decodes_values_before_native_storage_changes() {
        let metadata = metadata();
        let mut context =
            ExecutionContext::new_with_options(&metadata, ExecutionOptions::collect_all());
        let mut raw = context.raw_context();
        let mut scalar = 7u8;
        let mut array = [2u8, 3u8];
        let operands = [
            ptr::from_ref(&scalar).cast::<u8>(),
            ptr::from_ref(&array).cast::<u8>(),
        ];
        // SAFETY: operands use the native layouts specified by trace metadata.
        unsafe { xlsynth_pir_record_trace(&mut raw, 2, operands.as_ptr()) };
        scalar = 90;
        array[0] = 91;
        assert_eq!(scalar, 90);
        assert_eq!(array[0], 91);
        assert_eq!(context.result().trace_messages[0].message, "x=7 arr=[2, 3]");
    }

    #[test]
    fn trace_callback_formats_all_specifiers_and_wide_decimal_without_xls() {
        let twelve_bits = TraceValueLayout::Bits {
            bit_count: 12,
            byte_count: 2,
        };
        let metadata = CompiledFunctionMetadata {
            event_sites: vec![EventSiteMetadata {
                node_text_id: 20,
                kind: EventKind::Trace,
                label: None,
                message: None,
                format: Some(
                    "literal={{ default={} u={:u} d={:d} x={:x} 0x={:0x} #x={:#x} b={:b} 0b={:0b} #b={:#b} wide={} wide_u={:u}".to_string(),
                    ),
                verbosity: 0,
                operand_layouts: vec![
                    twelve_bits.clone(),
                    twelve_bits.clone(),
                    TraceValueLayout::Bits {
                        bit_count: 8,
                        byte_count: 1,
                    },
                    twelve_bits.clone(),
                    twelve_bits.clone(),
                    twelve_bits.clone(),
                    twelve_bits.clone(),
                    twelve_bits.clone(),
                    twelve_bits,
                    TraceValueLayout::Bits {
                        bit_count: 72,
                        byte_count: 9,
                    },
                    TraceValueLayout::Bits {
                        bit_count: 72,
                        byte_count: 9,
                    },
                ],
            }],
        };
        let mut context =
            ExecutionContext::new_with_options(&metadata, ExecutionOptions::collect_all());
        let mut raw = context.raw_context();
        let twelve = 43u16;
        let negative = 251u8;
        let wide = [1u8, 0, 0, 0, 0, 0, 0, 0, 1];
        let operands = [
            ptr::from_ref(&twelve).cast::<u8>(),
            ptr::from_ref(&twelve).cast::<u8>(),
            ptr::from_ref(&negative).cast::<u8>(),
            ptr::from_ref(&twelve).cast::<u8>(),
            ptr::from_ref(&twelve).cast::<u8>(),
            ptr::from_ref(&twelve).cast::<u8>(),
            ptr::from_ref(&twelve).cast::<u8>(),
            ptr::from_ref(&twelve).cast::<u8>(),
            ptr::from_ref(&twelve).cast::<u8>(),
            ptr::from_ref(&wide).cast::<u8>(),
            ptr::from_ref(&wide).cast::<u8>(),
        ];
        // SAFETY: operands use the native layouts specified by trace metadata.
        unsafe { xlsynth_pir_record_trace(&mut raw, 0, operands.as_ptr()) };
        assert_eq!(
            context.result().trace_messages[0].message,
            "literal={{ default=43 u=43 d=-5 x=2b 0x=02b #x=0x2b b=101011 0b=0000_0010_1011 #b=0b10_1011 wide=0x1_0000_0000_0000_0001 wide_u=18446744073709551617"
        );
    }

    #[test]
    fn clear_resets_accumulated_event_results() {
        let metadata = metadata();
        let mut context =
            ExecutionContext::new_with_options(&metadata, ExecutionOptions::collect_all());
        let mut raw = context.raw_context();
        // SAFETY: `raw` points into `context` for this immediate call.
        unsafe { xlsynth_pir_record_cover(&mut raw, 0) };
        context.clear();
        let result = context.result();
        assert!(result.assertion_failures.is_empty());
        assert!(result.assumption_failures.is_empty());
        assert!(result.trace_messages.is_empty());
        assert_eq!(result.cover_counts[0].count, 0);
    }

    #[test]
    fn default_context_does_not_collect_traces_or_covers() {
        let metadata = metadata();
        let mut context = ExecutionContext::new(&metadata);
        let mut raw = context.raw_context();
        let scalar = 7u8;
        let array = [2u8, 3u8];
        let operands = [
            ptr::from_ref(&scalar).cast::<u8>(),
            ptr::from_ref(&array).cast::<u8>(),
        ];
        // SAFETY: `raw` and operands point to valid test storage.
        unsafe {
            xlsynth_pir_record_cover(&mut raw, 0);
            xlsynth_pir_record_trace(&mut raw, 2, operands.as_ptr());
        }
        let result = context.result();
        assert!(result.cover_counts.is_empty());
        assert!(result.trace_messages.is_empty());
    }

    #[test]
    fn trace_callback_respects_runtime_verbosity() {
        let metadata = metadata();
        let scalar = 7u8;
        let array = [2u8, 3u8];
        let operands = [
            ptr::from_ref(&scalar).cast::<u8>(),
            ptr::from_ref(&array).cast::<u8>(),
        ];
        let mut context = ExecutionContext::new_with_options(
            &metadata,
            ExecutionOptions::new(Some(0), /* collect_covers= */ false),
        );
        let mut raw = context.raw_context();
        // SAFETY: `raw` and operands point to valid test storage.
        unsafe { xlsynth_pir_record_trace(&mut raw, 2, operands.as_ptr()) };
        assert!(context.result().trace_messages.is_empty());

        context.clear_with_options(ExecutionOptions::new(
            Some(1),
            /* collect_covers= */ false,
        ));
        let mut raw = context.raw_context();
        // SAFETY: `raw` and operands point to valid test storage.
        unsafe { xlsynth_pir_record_trace(&mut raw, 2, operands.as_ptr()) };
        assert_eq!(context.result().trace_messages[0].message, "x=7 arr=[2, 3]");
    }

    #[test]
    fn wide_trace_values_use_lsb_first_native_limbs() {
        let value = [1u64, 1u64];
        // SAFETY: `value` contains the two limbs required for bits[72].
        let formatted = unsafe {
            format_native_value(
                value.as_ptr().cast(),
                &TraceValueLayout::WideBits {
                    bit_count: 72,
                    limb_count: 2,
                },
                TraceFormatPreference::Hex,
            )
        };
        assert_eq!(formatted, "0x1_0000_0000_0000_0001");
    }

    #[test]
    fn wide_binary_runtime_helpers_cover_arithmetic_shifts_and_slices() {
        let lhs = [u64::MAX, 1];
        let rhs = [2u64, 0];
        let mut output = [0u64; 2];
        // SAFETY: all arrays contain the required two native limbs.
        unsafe {
            xlsynth_pir_runtime_wide_binop(
                output.as_mut_ptr(),
                65,
                lhs.as_ptr(),
                65,
                rhs.as_ptr(),
                65,
                WideBinaryOp::Umul as u32,
            );
        }
        assert_eq!(output, [u64::MAX - 1, 1]);

        let negative = [0u64, 1];
        let shift = [1u64, 0];
        // SAFETY: all arrays contain the required two native limbs.
        unsafe {
            xlsynth_pir_runtime_wide_binop(
                output.as_mut_ptr(),
                65,
                negative.as_ptr(),
                65,
                shift.as_ptr(),
                65,
                WideBinaryOp::Shra as u32,
            );
        }
        assert_eq!(output, [1u64 << 63, 1]);

        let start = [63u64, 0];
        // SAFETY: all arrays contain the limbs prescribed by their widths.
        unsafe {
            xlsynth_pir_runtime_wide_dynamic_bit_slice(
                output.as_mut_ptr(),
                65,
                lhs.as_ptr(),
                65,
                start.as_ptr(),
                65,
            );
        }
        assert_eq!(output, [3, 0]);

        let zero = [0u64, 0];
        let update = [3u64, 0];
        // SAFETY: all arrays contain the limbs prescribed by their widths.
        unsafe {
            xlsynth_pir_runtime_wide_bit_slice_update(
                output.as_mut_ptr(),
                65,
                zero.as_ptr(),
                65,
                start.as_ptr(),
                65,
                update.as_ptr(),
                65,
            );
        }
        assert_eq!(output, [1u64 << 63, 1]);
    }

    #[test]
    fn allocation_free_wide_shifts_match_bigint_reference() {
        fn reference_shift(
            lhs: &[u64],
            lhs_bit_count: usize,
            rhs: &[u64],
            rhs_bit_count: usize,
            dst_bit_count: usize,
            operation: WideBinaryOp,
        ) -> Vec<u64> {
            // SAFETY: the slices contain the limbs prescribed by their widths.
            let lhs_unsigned = unsafe { read_wide_bits(lhs.as_ptr(), lhs_bit_count) };
            // SAFETY: the slices contain the limbs prescribed by their widths.
            let rhs_unsigned = unsafe { read_wide_bits(rhs.as_ptr(), rhs_bit_count) };
            let result = match bounded_shift_amount(&rhs_unsigned, lhs_bit_count) {
                None if operation == WideBinaryOp::Shra => {
                    if as_signed(lhs_unsigned, lhs_bit_count).sign() == Sign::Minus {
                        bit_mask(dst_bit_count)
                    } else {
                        BigUint::from(0u8)
                    }
                }
                None => BigUint::from(0u8),
                Some(amount) if operation == WideBinaryOp::Shll => {
                    truncate_unsigned(lhs_unsigned << amount, dst_bit_count)
                }
                Some(amount) if operation == WideBinaryOp::Shrl => {
                    truncate_unsigned(lhs_unsigned >> amount, dst_bit_count)
                }
                Some(amount) => truncate_signed(
                    as_signed(lhs_unsigned, lhs_bit_count) >> amount,
                    dst_bit_count,
                ),
            };
            let mut output = vec![0; dst_bit_count.div_ceil(u64::BITS as usize)];
            // SAFETY: `output` contains the limbs prescribed by `dst_bit_count`.
            unsafe { write_wide_bits(output.as_mut_ptr(), dst_bit_count, result) };
            output
        }

        for lhs_bit_count in [1usize, 63, 64, 65, 127, 128, 255, 256, 511, 512] {
            let lhs_limb_count = lhs_bit_count.div_ceil(u64::BITS as usize);
            for negative in [false, true] {
                let mut lhs = (0..lhs_limb_count)
                    .map(|index| {
                        0x0123_4567_89ab_cdefu64.rotate_left((index * 13).try_into().unwrap())
                    })
                    .collect::<Vec<_>>();
                let sign_limb = (lhs_bit_count - 1) / u64::BITS as usize;
                let sign_mask = 1u64 << ((lhs_bit_count - 1) % u64::BITS as usize);
                if negative {
                    lhs[sign_limb] |= sign_mask;
                } else {
                    lhs[sign_limb] &= !sign_mask;
                }
                lhs[lhs_limb_count - 1] &= top_limb_mask(lhs_bit_count);

                let mut amounts = vec![
                    0,
                    1,
                    63,
                    64,
                    65,
                    lhs_bit_count.saturating_sub(1) as u64,
                    lhs_bit_count as u64,
                    lhs_bit_count as u64 + 1,
                ];
                amounts.sort_unstable();
                amounts.dedup();

                let mut destination_widths = vec![
                    lhs_bit_count.saturating_sub(1).max(1),
                    lhs_bit_count,
                    lhs_bit_count + 13,
                ];
                destination_widths.sort_unstable();
                destination_widths.dedup();

                for operation in [WideBinaryOp::Shll, WideBinaryOp::Shrl, WideBinaryOp::Shra] {
                    for rhs in amounts
                        .iter()
                        .map(|amount| [*amount, 0])
                        .chain(std::iter::once([0, 1]))
                    {
                        for dst_bit_count in destination_widths.iter().copied() {
                            let expected = reference_shift(
                                &lhs,
                                lhs_bit_count,
                                &rhs,
                                128,
                                dst_bit_count,
                                operation,
                            );
                            let mut actual = vec![0; dst_bit_count.div_ceil(u64::BITS as usize)];
                            // SAFETY: all slices contain the limbs prescribed by their widths.
                            unsafe {
                                xlsynth_pir_runtime_wide_binop(
                                    actual.as_mut_ptr(),
                                    dst_bit_count,
                                    lhs.as_ptr(),
                                    lhs_bit_count,
                                    rhs.as_ptr(),
                                    128,
                                    operation as u32,
                                );
                            }
                            assert_eq!(
                                actual, expected,
                                "operation={operation:?} lhs_width={lhs_bit_count} dst_width={dst_bit_count} rhs={rhs:?} negative={negative}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn wide_runtime_helpers_accept_zero_width_storage() {
        // SAFETY: zero-width values contain no limbs, so null pointers satisfy
        // the callback storage contract.
        unsafe {
            xlsynth_pir_runtime_wide_binop(
                ptr::null_mut(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
                WideBinaryOp::Sdiv as u32,
            );
            xlsynth_pir_runtime_wide_mulp(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
                0,
            );
        }
    }

    #[test]
    fn wide_unary_runtime_helpers_cover_encoding_and_extensions() {
        let input = [1u64 << 63, 1];
        let mut output = [0u64; 3];
        // SAFETY: all arrays contain sufficient native limbs for their widths.
        unsafe {
            xlsynth_pir_runtime_wide_unary_op(
                output.as_mut_ptr(),
                66,
                input.as_ptr(),
                65,
                WideUnaryOp::OneHot as u32,
                1,
            );
        }
        assert_eq!(output[..2], [1u64 << 63, 0]);

        // SAFETY: all arrays contain sufficient native limbs for their widths.
        unsafe {
            xlsynth_pir_runtime_wide_unary_op(
                output.as_mut_ptr(),
                7,
                input.as_ptr(),
                65,
                WideUnaryOp::ExtPrioEncode as u32,
                0,
            );
        }
        assert_eq!(output[0], 64);

        let zeros = [0u64; 2];
        // SAFETY: all arrays contain sufficient native limbs for their widths.
        unsafe {
            xlsynth_pir_runtime_wide_unary_op(
                output.as_mut_ptr(),
                129,
                zeros.as_ptr(),
                65,
                WideUnaryOp::ExtMaskLow as u32,
                0,
            );
        }
        assert_eq!(output, [0, 0, 0]);

        let count = [80u64, 0];
        // SAFETY: all arrays contain sufficient native limbs for their widths.
        unsafe {
            xlsynth_pir_runtime_wide_unary_op(
                output.as_mut_ptr(),
                129,
                count.as_ptr(),
                65,
                WideUnaryOp::ExtMaskLow as u32,
                0,
            );
        }
        assert_eq!(output, [u64::MAX, 0xffff, 0]);
    }

    #[test]
    fn wide_mulp_runtime_helper_returns_components_summing_to_product() {
        let lhs = [u64::MAX, 1];
        let rhs = [3u64, 0];
        let mut offset = [0u64; 3];
        let mut residual = [0u64; 3];
        // SAFETY: all arrays contain sufficient native limbs for their widths.
        unsafe {
            xlsynth_pir_runtime_wide_mulp(
                offset.as_mut_ptr(),
                residual.as_mut_ptr(),
                129,
                lhs.as_ptr(),
                65,
                rhs.as_ptr(),
                65,
                0,
            );
        }
        let offset = unsafe { read_wide_bits(offset.as_ptr(), 129) };
        let residual = unsafe { read_wide_bits(residual.as_ptr(), 129) };
        assert_eq!(
            truncate_unsigned(offset + residual, 129),
            truncate_unsigned(
                unsafe { read_wide_bits(lhs.as_ptr(), 65) }
                    * unsafe { read_wide_bits(rhs.as_ptr(), 65) },
                129,
            )
        );
    }
}
