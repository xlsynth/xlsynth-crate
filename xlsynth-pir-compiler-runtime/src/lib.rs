// SPDX-License-Identifier: Apache-2.0

//! Runtime ABI and observable-event collection for compiled PIR functions.

use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
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

/// Output and observable events produced by one generated wrapper invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunResult<T> {
    pub output: T,
    pub events: ExecutionResult,
}

macro_rules! define_native_bits {
    ($name:ident, $carrier:ty, $carrier_bits:expr) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
        pub struct $name<const BIT_COUNT: usize>($carrier);

        impl<const BIT_COUNT: usize> $name<BIT_COUNT> {
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

            /// Constructs a canonical bitvector value, rejecting excess high bits.
            pub fn new(value: $carrier) -> Result<Self, RunError> {
                Self::validate_width()?;
                if value & !Self::mask() != 0 {
                    Err(RunError(format!(
                        "value {value} does not fit in bits[{BIT_COUNT}]"
                    )))
                } else {
                    Ok(Self(value))
                }
            }

            /// Constructs a canonical bitvector by truncating high bits.
            pub const fn wrapping(value: $carrier) -> Self {
                assert!(
                    BIT_COUNT > 0 && BIT_COUNT <= $carrier_bits,
                    "invalid native bits carrier width"
                );
                Self(value & Self::mask())
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
    };
}

define_native_bits!(BitsInU8, u8, 8);
define_native_bits!(BitsInU16, u16, 16);
define_native_bits!(BitsInU32, u32, 32);
define_native_bits!(BitsInU64, u64, 64);

/// Native least-significant-first limb storage for a bitvector wider than 64
/// bits.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideBits<const BIT_COUNT: usize, const LIMB_COUNT: usize>([u64; LIMB_COUNT]);

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> WideBits<BIT_COUNT, LIMB_COUNT> {
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

    /// Constructs a canonical wide bitvector by truncating excess high bits.
    pub fn wrapping_limbs(mut limbs: [u64; LIMB_COUNT]) -> Self {
        assert!(
            BIT_COUNT > 64 && LIMB_COUNT == BIT_COUNT.div_ceil(64),
            "invalid native wide bits layout"
        );
        limbs[LIMB_COUNT - 1] &= Self::high_mask();
        Self(limbs)
    }

    /// Returns the least-significant-first limb representation.
    pub const fn limbs(&self) -> &[u64; LIMB_COUNT] {
        &self.0
    }
}

impl<const BIT_COUNT: usize, const LIMB_COUNT: usize> Default for WideBits<BIT_COUNT, LIMB_COUNT> {
    fn default() -> Self {
        Self([0; LIMB_COUNT])
    }
}

/// Zero-sized native representation of a PIR token value.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Token;

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
    assertion_failures: Vec<AssertionFailure>,
    assumption_failures: Vec<AssumptionFailure>,
    trace_messages: Vec<TraceMessage>,
    event_counts: Vec<u64>,
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
        Self {
            state: Box::new(ContextState {
                metadata,
                assertion_failures: Vec::new(),
                assumption_failures: Vec::new(),
                trace_messages: Vec::new(),
                event_counts: vec![0; metadata.event_sites.len()],
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
        let cover_counts = metadata
            .event_sites
            .iter()
            .zip(&self.state.event_counts)
            .filter(|(site, _)| site.kind == EventKind::Cover)
            .map(|(site, count)| CoverCount {
                node_text_id: site.node_text_id,
                label: site.label.clone().unwrap_or_default(),
                count: *count,
            })
            .collect();
        ExecutionResult {
            assertion_failures: self.state.assertion_failures.clone(),
            assumption_failures: self.state.assumption_failures.clone(),
            trace_messages: self.state.trace_messages.clone(),
            cover_counts,
        }
    }

    /// Clears all event records and accumulated cover counters.
    pub fn clear(&mut self) {
        self.state.assertion_failures.clear();
        self.state.assumption_failures.clear();
        self.state.trace_messages.clear();
        self.state.event_counts.fill(0);
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
    if site(state, site_id, EventKind::Cover).is_some() {
        if let Some(count) = state.event_counts.get_mut(site_id as usize) {
            *count = count.saturating_add(1);
        }
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
    let Some(site) = site(state, site_id, EventKind::Trace).cloned() else {
        return;
    };
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
        verbosity: 0,
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
            match bounded_shift_amount(&rhs_unsigned, lhs_bit_count) {
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
            }
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
    fn native_bits_wrappers_enforce_semantic_widths() {
        let value = BitsInU64::<42>::new((1u64 << 41) | 7).expect("value fits in bits[42]");
        assert_eq!(value.to_u64(), (1u64 << 41) | 7);
        assert!(BitsInU64::<42>::new(1u64 << 42).is_err());
        assert_eq!(BitsInU16::<9>::wrapping(0xffff).get(), 0x1ff);
        assert!(BitsInU8::<9>::new(0).is_err());
    }

    #[test]
    fn wide_bits_wrappers_use_lsb_first_limbs_and_mask_high_bits() {
        let value =
            WideBits::<65, 2>::from_limbs([0x0123_4567_89ab_cdef, 1]).expect("canonical value");
        assert_eq!(value.limbs(), &[0x0123_4567_89ab_cdef, 1]);
        assert!(WideBits::<65, 2>::from_limbs([0, 2]).is_err());
        assert_eq!(WideBits::<65, 2>::wrapping_limbs([7, 3]).limbs(), &[7, 1]);
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
                    operand_layouts: Vec::new(),
                },
                EventSiteMetadata {
                    node_text_id: 11,
                    kind: EventKind::Assert,
                    label: Some("assert_label".to_string()),
                    message: Some("failed".to_string()),
                    format: None,
                    operand_layouts: Vec::new(),
                },
                EventSiteMetadata {
                    node_text_id: 12,
                    kind: EventKind::Trace,
                    label: None,
                    message: None,
                    format: Some("x={} arr={}".to_string()),
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
                    operand_layouts: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn cover_and_assert_callbacks_collect_rust_owned_results() {
        let metadata = metadata();
        let mut context = ExecutionContext::new(&metadata);
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
        let mut context = ExecutionContext::new(&metadata);
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
        let mut context = ExecutionContext::new(&metadata);
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
        let mut context = ExecutionContext::new(&metadata);
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
