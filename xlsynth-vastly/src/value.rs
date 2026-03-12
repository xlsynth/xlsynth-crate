// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Signedness {
    Unsigned,
    Signed,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogicBit {
    Zero,
    One,
    X,
    Z,
}

impl LogicBit {
    pub fn is_known_01(self) -> bool {
        matches!(self, LogicBit::Zero | LogicBit::One)
    }

    pub fn as_char(self) -> char {
        match self {
            LogicBit::Zero => '0',
            LogicBit::One => '1',
            LogicBit::X => 'x',
            LogicBit::Z => 'z',
        }
    }
}

impl fmt::Display for LogicBit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

/// A 4-state bitvector value with explicit width and signedness.
///
/// Bits are stored LSB-first: `bits[0]` is bit 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value4 {
    pub width: u32,
    pub signedness: Signedness,
    bits: Vec<LogicBit>,
}

impl Value4 {
    pub fn new(width: u32, signedness: Signedness, bits_lsb_first: Vec<LogicBit>) -> Self {
        assert_eq!(bits_lsb_first.len(), width as usize);
        Self {
            width,
            signedness,
            bits: bits_lsb_first,
        }
    }

    pub fn zeros(width: u32, signedness: Signedness) -> Self {
        Self {
            width,
            signedness,
            bits: vec![LogicBit::Zero; width as usize],
        }
    }

    pub fn from_u64(width: u32, signedness: Signedness, value: u64) -> Self {
        let mut bits = Vec::with_capacity(width as usize);
        for bit in 0..width {
            bits.push(if bit >= 64 {
                LogicBit::Zero
            } else if value & (1u64 << bit) == 0 {
                LogicBit::Zero
            } else {
                LogicBit::One
            });
        }
        Self {
            width,
            signedness,
            bits,
        }
    }

    pub fn from_bits_msb_first(
        width: u32,
        signedness: Signedness,
        bits_msb_first: &[LogicBit],
    ) -> Self {
        assert_eq!(bits_msb_first.len(), width as usize);
        let mut bits = Vec::with_capacity(width as usize);
        for b in bits_msb_first.iter().rev() {
            bits.push(*b);
        }
        Self {
            width,
            signedness,
            bits,
        }
    }

    pub fn bit(&self, idx: u32) -> LogicBit {
        self.bits[idx as usize]
    }

    pub fn msb(&self) -> LogicBit {
        self.bits[(self.width - 1) as usize]
    }

    pub fn bits_lsb_first(&self) -> &[LogicBit] {
        &self.bits
    }

    pub fn to_bit_string_msb_first(&self) -> String {
        let mut s = String::with_capacity(self.width as usize);
        for b in self.bits.iter().rev() {
            s.push(b.as_char());
        }
        s
    }

    pub fn has_unknown(&self) -> bool {
        self.bits
            .iter()
            .any(|b| matches!(b, LogicBit::X | LogicBit::Z))
    }

    pub fn is_all_known_01(&self) -> bool {
        self.bits.iter().all(|b| b.is_known_01())
    }

    pub(crate) fn to_u32_saturating_if_known(&self) -> Option<u32> {
        if !self.is_all_known_01() {
            return None;
        }
        Some(known_u32_saturating(self))
    }

    pub(crate) fn to_u32_if_known(&self) -> Option<u32> {
        if !self.is_all_known_01() {
            return None;
        }
        let mut out = 0u32;
        for i in 0..self.width {
            if self.bit(i) != LogicBit::One {
                continue;
            }
            if i >= 32 {
                return None;
            }
            out |= 1u32 << i;
        }
        Some(out)
    }

    pub fn to_hex_string_if_known(&self) -> Option<String> {
        if !self.is_all_known_01() {
            return None;
        }
        if self.width == 0 {
            return Some("0".to_string());
        }
        let digits = self.width.div_ceil(4) as usize;
        let mut out = String::with_capacity(digits);
        for digit_idx in (0..digits).rev() {
            let base_bit = digit_idx * 4;
            let mut nibble = 0u8;
            for offset in 0..4usize {
                let bit_index = base_bit + offset;
                if bit_index < self.width as usize && self.bits[bit_index] == LogicBit::One {
                    nibble |= 1 << offset;
                }
            }
            out.push(match nibble {
                0..=9 => char::from(b'0' + nibble),
                10..=15 => char::from(b'a' + (nibble - 10)),
                _ => unreachable!("nibble out of range"),
            });
        }
        Some(out)
    }

    pub fn to_decimal_string_if_known(&self) -> Option<String> {
        if !self.is_all_known_01() {
            return None;
        }
        if self.width == 0 {
            return Some("0".to_string());
        }
        let is_negative = self.signedness == Signedness::Signed && self.msb() == LogicBit::One;
        let mut work = if is_negative {
            twos_complement_magnitude_bits_lsb(&self.bits)
        } else {
            self.bits.clone()
        };
        if is_all_zero_known_bits(&work) {
            return Some("0".to_string());
        }
        let mut digits = String::new();
        while !is_all_zero_known_bits(&work) {
            let rem = div_mod_known_bits_by_small_assign_lsb(&mut work, 10);
            digits.push(char::from(b'0' + rem));
        }
        let mut out: String = digits.chars().rev().collect();
        if is_negative {
            out.insert(0, '-');
        }
        Some(out)
    }

    pub fn to_u64_if_known(&self) -> Option<u64> {
        if self.width > 64 || !self.is_all_known_01() {
            return None;
        }
        let mut out = 0u64;
        for bit in 0..self.width {
            if self.bit(bit) == LogicBit::One {
                out |= 1u64 << bit;
            }
        }
        Some(out)
    }

    pub fn slice_lsb_width(&self, lsb: u32, width: u32) -> crate::Result<Self> {
        let end = lsb
            .checked_add(width)
            .ok_or_else(|| crate::Error::Parse("bit slice overflow".to_string()))?;
        if width == 0 {
            return Err(crate::Error::Parse(
                "bit slice width must be > 0".to_string(),
            ));
        }
        if end > self.width {
            return Err(crate::Error::Parse(format!(
                "bit slice {}..{} is out of bounds for width {}",
                lsb, end, self.width
            )));
        }
        Ok(Self {
            width,
            signedness: self.signedness,
            bits: self.bits[lsb as usize..end as usize].to_vec(),
        })
    }

    pub fn replace_slice(&self, lsb: u32, value: &Value4) -> crate::Result<Self> {
        let end = lsb
            .checked_add(value.width)
            .ok_or_else(|| crate::Error::Parse("bit slice overflow".to_string()))?;
        if value.width == 0 {
            return Err(crate::Error::Parse(
                "replacement slice width must be > 0".to_string(),
            ));
        }
        if end > self.width {
            return Err(crate::Error::Parse(format!(
                "replacement slice {}..{} is out of bounds for width {}",
                lsb, end, self.width
            )));
        }
        let mut bits = self.bits.clone();
        bits[lsb as usize..end as usize].copy_from_slice(value.bits_lsb_first());
        Ok(Self {
            width: self.width,
            signedness: self.signedness,
            bits,
        })
    }

    pub fn parse_numeric_token(
        width: u32,
        signedness: Signedness,
        text: &str,
    ) -> crate::Result<Self> {
        let compact: String = text.chars().filter(|c| *c != '_').collect();
        let s = compact.as_str();
        let (base, digits) =
            if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                (16u8, rest)
            } else if let Some(rest) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
                (2u8, rest)
            } else if let Some(rest) = s.strip_prefix("0o").or_else(|| s.strip_prefix("0O")) {
                (8u8, rest)
            } else {
                (10u8, s)
            };
        if digits.is_empty() {
            return Err(crate::Error::Parse(format!(
                "bad integer literal `{text}`: missing digits"
            )));
        }
        let mut bits = vec![LogicBit::Zero; width as usize];
        for c in digits.chars() {
            let digit = c.to_digit(u32::from(base)).ok_or_else(|| {
                crate::Error::Parse(format!("bad integer literal `{text}`: invalid digit `{c}`"))
            })?;
            mul_known_bits_by_small(&mut bits, base);
            add_small_to_known_bits(&mut bits, digit);
        }
        Ok(Value4::new(width, signedness, bits))
    }

    pub fn parse_unsized_decimal_token(signedness: Signedness, text: &str) -> crate::Result<Self> {
        let compact: String = text.chars().filter(|c| *c != '_').collect();
        let digits = compact.as_str();
        if digits.is_empty() {
            return Err(crate::Error::Parse(format!(
                "bad integer literal `{text}`: missing digits"
            )));
        }
        if digits.chars().any(|c| !c.is_ascii_digit()) {
            return Err(crate::Error::Parse(format!(
                "bad integer literal `{text}`: invalid decimal digits"
            )));
        }
        let working_width = digits
            .len()
            .checked_mul(4)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                crate::Error::Parse(format!(
                    "bad integer literal `{text}`: decimal width overflow"
                ))
            })?;
        let working_width = u32::try_from(working_width).map_err(|_| {
            crate::Error::Parse(format!(
                "bad integer literal `{text}`: decimal width overflow"
            ))
        })?;
        let parsed = Self::parse_numeric_token(working_width, signedness, digits)?;
        let magnitude_width = parsed
            .bits
            .iter()
            .rposition(|b| *b == LogicBit::One)
            .map(|i| (i as u32) + 1)
            .unwrap_or(1);
        let minimal_width = match signedness {
            Signedness::Unsigned => magnitude_width,
            Signedness::Signed => magnitude_width.saturating_add(1),
        }
        .max(32);
        Ok(parsed.resize(minimal_width))
    }

    pub fn resize(&self, new_width: u32) -> Self {
        self.clone().into_width(new_width)
    }

    pub fn with_signedness(&self, signedness: Signedness) -> Self {
        self.clone().into_signedness(signedness)
    }

    pub fn into_width(mut self, new_width: u32) -> Self {
        if new_width == self.width {
            return self;
        }
        if new_width < self.width {
            self.bits.truncate(new_width as usize);
            self.width = new_width;
            return self;
        }

        let ext_bit = match self.signedness {
            Signedness::Unsigned => LogicBit::Zero,
            Signedness::Signed if self.width == 0 => LogicBit::Zero,
            Signedness::Signed => self.msb(),
        };
        self.bits.resize(new_width as usize, ext_bit);
        self.width = new_width;
        self
    }

    pub fn into_signedness(mut self, signedness: Signedness) -> Self {
        self.signedness = signedness;
        self
    }

    pub fn into_width_and_signedness(self, new_width: u32, signedness: Signedness) -> Self {
        self.into_signedness(signedness).into_width(new_width)
    }

    pub fn bitwise_not(&self) -> Self {
        let mut bits = Vec::with_capacity(self.bits.len());
        for b in &self.bits {
            let nb = match b {
                LogicBit::Zero => LogicBit::One,
                LogicBit::One => LogicBit::Zero,
                LogicBit::X => LogicBit::X,
                LogicBit::Z => LogicBit::X, // treat Z as unknown for bitwise ops
            };
            bits.push(nb);
        }
        Self {
            width: self.width,
            signedness: self.signedness,
            bits,
        }
    }

    pub fn bitwise_and(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let a = Value4Ref::resize(self, w);
        let b = Value4Ref::resize(rhs, w);
        let mut bits = Vec::with_capacity(w as usize);
        for i in 0..w {
            bits.push(bit_and_4(a.bit(i), b.bit(i)));
        }
        Value4::new(w, merged_signedness(&a, &b), bits)
    }

    pub fn bitwise_or(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let a = Value4Ref::resize(self, w);
        let b = Value4Ref::resize(rhs, w);
        let mut bits = Vec::with_capacity(w as usize);
        for i in 0..w {
            bits.push(bit_or_4(a.bit(i), b.bit(i)));
        }
        Value4::new(w, merged_signedness(&a, &b), bits)
    }

    pub fn bitwise_xor(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let a = Value4Ref::resize(self, w);
        let b = Value4Ref::resize(rhs, w);
        let mut bits = Vec::with_capacity(w as usize);
        for i in 0..w {
            bits.push(bit_xor_4(a.bit(i), b.bit(i)));
        }
        Value4::new(w, merged_signedness(&a, &b), bits)
    }

    pub fn add(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let signedness = merged_signedness(self, rhs);
        if self.has_unknown() || rhs.has_unknown() {
            return Value4::new(w, signedness, vec![LogicBit::X; w as usize]);
        }
        let a = Value4Ref::resize(self, w);
        let b = Value4Ref::resize(rhs, w);
        let bits = add_known_bits_lsb(a.bits_lsb_first(), b.bits_lsb_first());
        Value4::new(w, signedness, bits)
    }

    pub fn sub(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let signedness = merged_signedness(self, rhs);
        if self.has_unknown() || rhs.has_unknown() {
            return Value4::new(w, signedness, vec![LogicBit::X; w as usize]);
        }
        let a = Value4Ref::resize(self, w);
        let b = Value4Ref::resize(rhs, w);
        let bits = sub_known_bits_lsb(a.bits_lsb_first(), b.bits_lsb_first());
        Value4::new(w, signedness, bits)
    }

    pub fn mul(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let signedness = merged_signedness(self, rhs);
        if self.has_unknown() || rhs.has_unknown() {
            return Value4::new(w, signedness, vec![LogicBit::X; w as usize]);
        }
        let a = Value4Ref::resize(self, w);
        let b = Value4Ref::resize(rhs, w);

        if is_all_zero_known_bits(a.bits_lsb_first()) || is_all_zero_known_bits(b.bits_lsb_first())
        {
            return Value4::zeros(w, signedness);
        }
        if is_known_one_bits(a.bits_lsb_first()) {
            return b.with_signedness(signedness);
        }
        if is_known_one_bits(b.bits_lsb_first()) {
            return a.with_signedness(signedness);
        }
        if is_all_one_known_bits(a.bits_lsb_first()) {
            return b.unary_minus().with_signedness(signedness);
        }
        if is_all_one_known_bits(b.bits_lsb_first()) {
            return a.unary_minus().with_signedness(signedness);
        }
        if let Some(shift) = known_single_bit_index(a.bits_lsb_first()) {
            return shl_known_bits(a.width, signedness, b.bits_lsb_first(), shift);
        }
        if let Some(shift) = known_single_bit_index(b.bits_lsb_first()) {
            return shl_known_bits(a.width, signedness, a.bits_lsb_first(), shift);
        }
        if w <= 128 {
            let a_bits = known_bits_to_u128(a.bits_lsb_first());
            let b_bits = known_bits_to_u128(b.bits_lsb_first());
            let product = a_bits.wrapping_mul(b_bits) & mask_for_width(w);
            return Value4::new(w, signedness, u128_to_known_bits_lsb(product, w as usize));
        }

        let bits = mul_known_bits_lsb(a.bits_lsb_first(), b.bits_lsb_first(), w as usize);
        Value4::new(w, signedness, bits)
    }

    pub fn div(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let signedness = merged_signedness(self, rhs);
        if self.has_unknown() || rhs.has_unknown() {
            return Value4::new(w, signedness, vec![LogicBit::X; w as usize]);
        }
        let numer = Value4Ref::resize(self, w);
        let denom = Value4Ref::resize(rhs, w);
        if is_all_zero_known_bits(denom.bits_lsb_first()) {
            return Value4::new(w, signedness, vec![LogicBit::X; w as usize]);
        }
        if signedness == Signedness::Signed {
            let numer_neg = numer.msb() == LogicBit::One;
            let denom_neg = denom.msb() == LogicBit::One;
            let numer_mag = if numer_neg {
                numer.unary_minus().with_signedness(Signedness::Unsigned)
            } else {
                numer.with_signedness(Signedness::Unsigned)
            };
            let denom_mag = if denom_neg {
                denom.unary_minus().with_signedness(Signedness::Unsigned)
            } else {
                denom.with_signedness(Signedness::Unsigned)
            };
            let (q, _) =
                div_mod_known_bits_lsb(numer_mag.bits_lsb_first(), denom_mag.bits_lsb_first());
            let mut quotient = Value4::new(w, Signedness::Unsigned, q);
            if numer_neg ^ denom_neg {
                quotient = quotient.unary_minus();
            }
            return quotient.with_signedness(Signedness::Signed);
        }
        let (q, _) = div_mod_known_bits_lsb(numer.bits_lsb_first(), denom.bits_lsb_first());
        Value4::new(w, signedness, q)
    }

    pub fn modu(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let signedness = merged_signedness(self, rhs);
        if self.has_unknown() || rhs.has_unknown() {
            return Value4::new(w, signedness, vec![LogicBit::X; w as usize]);
        }
        let numer = Value4Ref::resize(self, w);
        let denom = Value4Ref::resize(rhs, w);
        if is_all_zero_known_bits(denom.bits_lsb_first()) {
            return Value4::new(w, signedness, vec![LogicBit::X; w as usize]);
        }
        if signedness == Signedness::Signed {
            let numer_neg = numer.msb() == LogicBit::One;
            let denom_neg = denom.msb() == LogicBit::One;
            let numer_mag = if numer_neg {
                numer.unary_minus().with_signedness(Signedness::Unsigned)
            } else {
                numer.with_signedness(Signedness::Unsigned)
            };
            let denom_mag = if denom_neg {
                denom.unary_minus().with_signedness(Signedness::Unsigned)
            } else {
                denom.with_signedness(Signedness::Unsigned)
            };
            let (_, r) =
                div_mod_known_bits_lsb(numer_mag.bits_lsb_first(), denom_mag.bits_lsb_first());
            let mut rem = Value4::new(w, Signedness::Unsigned, r);
            if numer_neg {
                rem = rem.unary_minus();
            }
            return rem.with_signedness(Signedness::Signed);
        }
        let (_, r) = div_mod_known_bits_lsb(numer.bits_lsb_first(), denom.bits_lsb_first());
        Value4::new(w, signedness, r)
    }

    pub fn shl(&self, rhs: &Value4) -> Value4 {
        let w = self.width;
        if !rhs.is_all_known_01() {
            return Value4::new(w, self.signedness, vec![LogicBit::X; w as usize]);
        }
        let sh = known_u32_saturating(rhs);
        if sh >= w {
            return Value4::new(w, self.signedness, vec![LogicBit::Zero; w as usize]);
        }
        let mut bits = vec![LogicBit::Zero; w as usize];
        for i in 0..(w - sh) {
            bits[(i + sh) as usize] = self.bit(i);
        }
        Value4::new(w, self.signedness, bits)
    }

    pub fn shr(&self, rhs: &Value4) -> Value4 {
        let w = self.width;
        if !rhs.is_all_known_01() {
            return Value4::new(w, self.signedness, vec![LogicBit::X; w as usize]);
        }
        let sh = known_u32_saturating(rhs);
        if sh >= w {
            return Value4::new(w, self.signedness, vec![LogicBit::Zero; w as usize]);
        }
        let mut bits = vec![LogicBit::Zero; w as usize];
        for i in 0..w {
            if i >= sh {
                bits[(i - sh) as usize] = self.bit(i);
            }
        }
        Value4::new(w, self.signedness, bits)
    }

    pub fn sshr(&self, rhs: &Value4) -> Value4 {
        let w = self.width;
        if !rhs.is_all_known_01() {
            return Value4::new(w, self.signedness, vec![LogicBit::X; w as usize]);
        }
        let sh = known_u32_saturating(rhs);
        let fill = match self.signedness {
            Signedness::Signed => self.msb(),
            Signedness::Unsigned => LogicBit::Zero,
        };
        if sh >= w {
            return Value4::new(w, self.signedness, vec![fill; w as usize]);
        }
        let mut bits = vec![fill; w as usize];
        for i in 0..w {
            if i >= sh {
                bits[(i - sh) as usize] = self.bit(i);
            }
        }
        Value4::new(w, self.signedness, bits)
    }

    pub fn unary_minus(&self) -> Value4 {
        let w = self.width;
        if self.has_unknown() {
            return Value4::new(w, self.signedness, vec![LogicBit::X; w as usize]);
        }
        let mut bits = self.bitwise_not().bits;
        let mut carry = true;
        for bit in &mut bits {
            if !carry {
                break;
            }
            match *bit {
                LogicBit::Zero => {
                    *bit = LogicBit::One;
                    carry = false;
                }
                LogicBit::One => {
                    *bit = LogicBit::Zero;
                }
                LogicBit::X | LogicBit::Z => unreachable!("bitwise_not preserves known bits"),
            }
        }
        Value4::new(w, self.signedness, bits)
    }

    pub fn cmp_rel(&self, rhs: &Value4, op: RelOp) -> Value4 {
        if self.has_unknown() || rhs.has_unknown() {
            return Value4::new(1, Signedness::Unsigned, vec![LogicBit::X]);
        }
        let w = self.width.max(rhs.width);
        let use_signed =
            self.signedness == Signedness::Signed && rhs.signedness == Signedness::Signed;
        let (a, b) = if use_signed {
            (Value4Ref::resize(self, w), Value4Ref::resize(rhs, w))
        } else {
            (
                Value4Ref::recontext(self, w, Signedness::Unsigned),
                Value4Ref::recontext(rhs, w, Signedness::Unsigned),
            )
        };
        let ord = if use_signed {
            cmp_known_signed(&a, &b)
        } else {
            cmp_known_unsigned(&a, &b)
        };
        let out = match op {
            RelOp::Lt => ord == Ordering::Less,
            RelOp::Le => ord != Ordering::Greater,
            RelOp::Gt => ord == Ordering::Greater,
            RelOp::Ge => ord != Ordering::Less,
        };
        Value4::new(
            1,
            Signedness::Unsigned,
            vec![if out { LogicBit::One } else { LogicBit::Zero }],
        )
    }

    /// Verilog-ish "truthiness" as a 4-state bit.
    ///
    /// - Returns 1 if any known 1 bit exists.
    /// - Returns 0 if all bits are known 0.
    /// - Otherwise returns X.
    pub fn to_bool4(&self) -> LogicBit {
        let mut saw_unknown = false;
        for b in &self.bits {
            match b {
                LogicBit::One => return LogicBit::One,
                LogicBit::Zero => {}
                LogicBit::X | LogicBit::Z => saw_unknown = true,
            }
        }
        if saw_unknown {
            LogicBit::X
        } else {
            LogicBit::Zero
        }
    }

    pub fn logical_not(&self) -> Value4 {
        let b = match self.to_bool4() {
            LogicBit::Zero => LogicBit::One,
            LogicBit::One => LogicBit::Zero,
            LogicBit::X | LogicBit::Z => LogicBit::X,
        };
        Value4::new(1, Signedness::Unsigned, vec![b])
    }

    pub fn logical_and(&self, rhs: &Value4) -> Value4 {
        let a = self.to_bool4();
        let b = rhs.to_bool4();
        let out = logic_and_bit(a, b);
        Value4::new(1, Signedness::Unsigned, vec![out])
    }

    pub fn logical_or(&self, rhs: &Value4) -> Value4 {
        let a = self.to_bool4();
        let b = rhs.to_bool4();
        let out = logic_or_bit(a, b);
        Value4::new(1, Signedness::Unsigned, vec![out])
    }

    pub fn eq_logical(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let use_signed =
            self.signedness == Signedness::Signed && rhs.signedness == Signedness::Signed;
        let (a, b) = if use_signed {
            (Value4Ref::resize(self, w), Value4Ref::resize(rhs, w))
        } else {
            (
                Value4Ref::recontext(self, w, Signedness::Unsigned),
                Value4Ref::recontext(rhs, w, Signedness::Unsigned),
            )
        };

        let mut saw_unknown = false;
        for i in 0..w {
            let ab = a.bit(i);
            let bb = b.bit(i);
            if !ab.is_known_01() || !bb.is_known_01() {
                saw_unknown = true;
                continue;
            }
            if ab != bb {
                return Value4::new(1, Signedness::Unsigned, vec![LogicBit::Zero]);
            }
        }
        if saw_unknown {
            Value4::new(1, Signedness::Unsigned, vec![LogicBit::X])
        } else {
            Value4::new(1, Signedness::Unsigned, vec![LogicBit::One])
        }
    }

    pub fn neq_logical(&self, rhs: &Value4) -> Value4 {
        let eq = self.eq_logical(rhs);
        match eq.bits[0] {
            LogicBit::Zero => Value4::new(1, Signedness::Unsigned, vec![LogicBit::One]),
            LogicBit::One => Value4::new(1, Signedness::Unsigned, vec![LogicBit::Zero]),
            LogicBit::X | LogicBit::Z => Value4::new(1, Signedness::Unsigned, vec![LogicBit::X]),
        }
    }

    pub fn eq_case(&self, rhs: &Value4) -> Value4 {
        let w = self.width.max(rhs.width);
        let use_signed =
            self.signedness == Signedness::Signed && rhs.signedness == Signedness::Signed;
        let (a, b) = if use_signed {
            (Value4Ref::resize(self, w), Value4Ref::resize(rhs, w))
        } else {
            (
                Value4Ref::recontext(self, w, Signedness::Unsigned),
                Value4Ref::recontext(rhs, w, Signedness::Unsigned),
            )
        };

        for i in 0..w {
            if a.bit(i) != b.bit(i) {
                return Value4::new(1, Signedness::Unsigned, vec![LogicBit::Zero]);
            }
        }
        Value4::new(1, Signedness::Unsigned, vec![LogicBit::One])
    }

    pub fn neq_case(&self, rhs: &Value4) -> Value4 {
        let eq = self.eq_case(rhs);
        match eq.bits[0] {
            LogicBit::Zero => Value4::new(1, Signedness::Unsigned, vec![LogicBit::One]),
            LogicBit::One => Value4::new(1, Signedness::Unsigned, vec![LogicBit::Zero]),
            LogicBit::X | LogicBit::Z => Value4::new(1, Signedness::Unsigned, vec![LogicBit::X]),
        }
    }

    pub fn ternary(cond: &Value4, t: &Value4, f: &Value4) -> Value4 {
        let c = cond.to_bool4();
        let out_width = t.width.max(f.width);
        let out_signedness =
            if t.signedness == Signedness::Signed && f.signedness == Signedness::Signed {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
        let t2 = Value4Ref::recontext(t, out_width, out_signedness);
        let f2 = Value4Ref::recontext(f, out_width, out_signedness);
        match c {
            LogicBit::One => t2.into_owned(),
            LogicBit::Zero => f2.into_owned(),
            LogicBit::X | LogicBit::Z => {
                let mut bits = Vec::with_capacity(out_width as usize);
                for i in 0..out_width {
                    let tb = t2.bit(i);
                    let fb = f2.bit(i);
                    bits.push(if tb == fb { tb } else { LogicBit::X });
                }
                Value4 {
                    width: out_width,
                    signedness: out_signedness,
                    bits,
                }
            }
        }
    }

    pub fn concat(parts: &[Value4]) -> Value4 {
        let mut width: u32 = 0;
        let mut signedness = Signedness::Unsigned;
        for p in parts {
            width = width.saturating_add(p.width);
            if p.signedness == Signedness::Signed {
                // concatenation is unsigned in Verilog, but keep unsigned unless you want to
                // treat differently.
                signedness = Signedness::Unsigned;
            }
        }
        let mut bits: Vec<LogicBit> = Vec::with_capacity(width as usize);
        for p in parts.iter().rev() {
            // MSB..LSB of concatenation => LSB-first storage means append each part's bits.
            bits.extend_from_slice(p.bits_lsb_first());
        }
        Value4::new(width, signedness, bits)
    }

    pub fn replicate(count: u32, v: &Value4) -> Value4 {
        let width = v.width.saturating_mul(count);
        let mut bits: Vec<LogicBit> = Vec::with_capacity(width as usize);
        for _ in 0..count {
            bits.extend_from_slice(v.bits_lsb_first());
        }
        Value4::new(width, Signedness::Unsigned, bits)
    }

    pub fn index(&self, idx: u32) -> Value4 {
        if idx >= self.width {
            return Value4::new(1, Signedness::Unsigned, vec![LogicBit::X]);
        }
        Value4::new(1, Signedness::Unsigned, vec![self.bit(idx)])
    }

    pub fn slice(&self, msb: u32, lsb: u32) -> Value4 {
        if msb < lsb {
            return Value4::new(0, Signedness::Unsigned, Vec::new());
        }
        let w = msb - lsb + 1;
        let mut bits = Vec::with_capacity(w as usize);
        for i in 0..w {
            let src = lsb + i;
            bits.push(if src < self.width {
                self.bit(src)
            } else {
                LogicBit::X
            });
        }
        Value4::new(w, Signedness::Unsigned, bits)
    }

    pub fn indexed_slice(&self, base: u32, width: u32, upward: bool) -> Value4 {
        let mut bits = Vec::with_capacity(width as usize);
        for i in 0..width {
            let bit = if upward {
                let src = base.saturating_add(i);
                if src < self.width {
                    self.bit(src)
                } else {
                    LogicBit::X
                }
            } else if let Some(src) = base
                .checked_add(1)
                .and_then(|v| v.checked_add(i))
                .and_then(|v| v.checked_sub(width))
            {
                if src < self.width {
                    self.bit(src)
                } else {
                    LogicBit::X
                }
            } else {
                LogicBit::X
            };
            bits.push(bit);
        }
        Value4::new(width, Signedness::Unsigned, bits)
    }

    pub fn reduce_and(&self) -> Value4 {
        let mut saw_unknown = false;
        for b in self.bits_lsb_first() {
            match b {
                LogicBit::Zero => {
                    return Value4::new(1, Signedness::Unsigned, vec![LogicBit::Zero]);
                }
                LogicBit::One => {}
                LogicBit::X | LogicBit::Z => saw_unknown = true,
            }
        }
        if saw_unknown {
            Value4::new(1, Signedness::Unsigned, vec![LogicBit::X])
        } else {
            Value4::new(1, Signedness::Unsigned, vec![LogicBit::One])
        }
    }

    pub fn reduce_or(&self) -> Value4 {
        let mut saw_unknown = false;
        for b in self.bits_lsb_first() {
            match b {
                LogicBit::One => return Value4::new(1, Signedness::Unsigned, vec![LogicBit::One]),
                LogicBit::Zero => {}
                LogicBit::X | LogicBit::Z => saw_unknown = true,
            }
        }
        if saw_unknown {
            Value4::new(1, Signedness::Unsigned, vec![LogicBit::X])
        } else {
            Value4::new(1, Signedness::Unsigned, vec![LogicBit::Zero])
        }
    }

    pub fn reduce_xor(&self) -> Value4 {
        let mut saw_unknown = false;
        let mut acc = false;
        for b in self.bits_lsb_first() {
            match b {
                LogicBit::Zero => {}
                LogicBit::One => acc = !acc,
                LogicBit::X | LogicBit::Z => saw_unknown = true,
            }
        }
        if saw_unknown {
            Value4::new(1, Signedness::Unsigned, vec![LogicBit::X])
        } else {
            Value4::new(
                1,
                Signedness::Unsigned,
                vec![if acc { LogicBit::One } else { LogicBit::Zero }],
            )
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RelOp {
    Lt,
    Le,
    Gt,
    Ge,
}

fn merged_signedness(a: &Value4, b: &Value4) -> Signedness {
    if a.signedness == Signedness::Signed && b.signedness == Signedness::Signed {
        Signedness::Signed
    } else {
        Signedness::Unsigned
    }
}

fn known_u32_saturating(v: &Value4) -> u32 {
    debug_assert!(v.is_all_known_01());
    let mut out = 0u32;
    for i in 0..v.width {
        if v.bit(i) != LogicBit::One {
            continue;
        }
        if i >= 32 {
            return u32::MAX;
        }
        out |= 1u32 << i;
    }
    out
}

fn cmp_known_unsigned(a: &Value4, b: &Value4) -> Ordering {
    debug_assert_eq!(a.width, b.width);
    debug_assert!(a.is_all_known_01());
    debug_assert!(b.is_all_known_01());
    cmp_known_bits_unsigned_lsb(a.bits_lsb_first(), b.bits_lsb_first())
}

fn cmp_known_signed(a: &Value4, b: &Value4) -> Ordering {
    debug_assert_eq!(a.width, b.width);
    debug_assert!(a.is_all_known_01());
    debug_assert!(b.is_all_known_01());
    if a.width == 0 {
        return Ordering::Equal;
    }
    match (a.msb(), b.msb()) {
        (LogicBit::One, LogicBit::Zero) => Ordering::Less,
        (LogicBit::Zero, LogicBit::One) => Ordering::Greater,
        _ => cmp_known_unsigned(a, b),
    }
}

fn logic_bit_from_bool(v: bool) -> LogicBit {
    if v { LogicBit::One } else { LogicBit::Zero }
}

enum Value4Ref<'a> {
    Borrowed(&'a Value4),
    Owned(Value4),
}

impl<'a> Value4Ref<'a> {
    fn resize(value: &'a Value4, width: u32) -> Self {
        if value.width == width {
            Self::Borrowed(value)
        } else {
            Self::Owned(value.clone().into_width(width))
        }
    }

    fn recontext(value: &'a Value4, width: u32, signedness: Signedness) -> Self {
        if value.width == width && value.signedness == signedness {
            Self::Borrowed(value)
        } else {
            Self::Owned(value.clone().into_width_and_signedness(width, signedness))
        }
    }

    fn into_owned(self) -> Value4 {
        match self {
            Self::Borrowed(value) => value.clone(),
            Self::Owned(value) => value,
        }
    }
}

impl std::ops::Deref for Value4Ref<'_> {
    type Target = Value4;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

fn cmp_known_bits_unsigned_lsb(a: &[LogicBit], b: &[LogicBit]) -> Ordering {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.iter().all(|bit| bit.is_known_01()));
    debug_assert!(b.iter().all(|bit| bit.is_known_01()));
    for i in (0..a.len()).rev() {
        match (a[i], b[i]) {
            (LogicBit::One, LogicBit::Zero) => return Ordering::Greater,
            (LogicBit::Zero, LogicBit::One) => return Ordering::Less,
            _ => {}
        }
    }
    Ordering::Equal
}

fn add_known_bits_lsb(a: &[LogicBit], b: &[LogicBit]) -> Vec<LogicBit> {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.iter().all(|bit| bit.is_known_01()));
    debug_assert!(b.iter().all(|bit| bit.is_known_01()));
    let mut out = Vec::with_capacity(a.len());
    let mut carry = false;
    for i in 0..a.len() {
        let ones = usize::from(a[i] == LogicBit::One)
            + usize::from(b[i] == LogicBit::One)
            + usize::from(carry);
        out.push(logic_bit_from_bool((ones & 1) != 0));
        carry = ones >= 2;
    }
    out
}

fn sub_known_bits_lsb(a: &[LogicBit], b: &[LogicBit]) -> Vec<LogicBit> {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.iter().all(|bit| bit.is_known_01()));
    debug_assert!(b.iter().all(|bit| bit.is_known_01()));
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = false;
    for i in 0..a.len() {
        let ai = usize::from(a[i] == LogicBit::One);
        let subtrahend = usize::from(b[i] == LogicBit::One) + usize::from(borrow);
        if ai >= subtrahend {
            out.push(logic_bit_from_bool((ai - subtrahend) != 0));
            borrow = false;
        } else {
            out.push(logic_bit_from_bool((ai + 2 - subtrahend) != 0));
            borrow = true;
        }
    }
    out
}

fn sub_known_bits_assign_lsb(a: &mut [LogicBit], b: &[LogicBit]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.iter().all(|bit| bit.is_known_01()));
    debug_assert!(b.iter().all(|bit| bit.is_known_01()));
    let mut borrow = false;
    for i in 0..a.len() {
        let ai = usize::from(a[i] == LogicBit::One);
        let subtrahend = usize::from(b[i] == LogicBit::One) + usize::from(borrow);
        if ai >= subtrahend {
            a[i] = logic_bit_from_bool((ai - subtrahend) != 0);
            borrow = false;
        } else {
            a[i] = logic_bit_from_bool((ai + 2 - subtrahend) != 0);
            borrow = true;
        }
    }
}

fn mul_known_bits_lsb(a: &[LogicBit], b: &[LogicBit], out_width: usize) -> Vec<LogicBit> {
    debug_assert!(a.iter().all(|bit| bit.is_known_01()));
    debug_assert!(b.iter().all(|bit| bit.is_known_01()));
    let mut out = vec![LogicBit::Zero; out_width];
    for (i, &bb) in b.iter().enumerate() {
        if bb != LogicBit::One {
            continue;
        }
        let mut carry = false;
        for (j, &ab) in a.iter().enumerate() {
            let dst = i + j;
            if dst >= out_width {
                break;
            }
            let ones = usize::from(out[dst] == LogicBit::One)
                + usize::from(ab == LogicBit::One)
                + usize::from(carry);
            out[dst] = logic_bit_from_bool((ones & 1) != 0);
            carry = ones >= 2;
        }
        let mut dst = i + a.len();
        while carry && dst < out_width {
            let ones = usize::from(out[dst] == LogicBit::One) + 1;
            out[dst] = logic_bit_from_bool((ones & 1) != 0);
            carry = ones >= 2;
            dst += 1;
        }
    }
    out
}

fn shl1_known_bits_assign_lsb(bits: &mut [LogicBit], lsb_in: LogicBit) {
    debug_assert!(lsb_in.is_known_01());
    if bits.is_empty() {
        return;
    }
    for i in (1..bits.len()).rev() {
        bits[i] = bits[i - 1];
    }
    bits[0] = lsb_in;
}

fn is_all_zero_known_bits(bits: &[LogicBit]) -> bool {
    debug_assert!(bits.iter().all(|bit| bit.is_known_01()));
    bits.iter().all(|bit| *bit == LogicBit::Zero)
}

fn is_all_one_known_bits(bits: &[LogicBit]) -> bool {
    debug_assert!(bits.iter().all(|bit| bit.is_known_01()));
    bits.iter().all(|bit| *bit == LogicBit::One)
}

fn is_known_one_bits(bits: &[LogicBit]) -> bool {
    debug_assert!(bits.iter().all(|bit| bit.is_known_01()));
    bits.first() == Some(&LogicBit::One) && bits[1..].iter().all(|bit| *bit == LogicBit::Zero)
}

fn known_single_bit_index(bits: &[LogicBit]) -> Option<u32> {
    debug_assert!(bits.iter().all(|bit| bit.is_known_01()));
    let mut found: Option<u32> = None;
    for (idx, bit) in bits.iter().enumerate() {
        if *bit != LogicBit::One {
            continue;
        }
        let idx = idx as u32;
        if found.is_some() {
            return None;
        }
        found = Some(idx);
    }
    found
}

fn shl_known_bits(width: u32, signedness: Signedness, bits: &[LogicBit], shift: u32) -> Value4 {
    debug_assert_eq!(bits.len(), width as usize);
    debug_assert!(bits.iter().all(|bit| bit.is_known_01()));
    if shift >= width {
        return Value4::zeros(width, signedness);
    }
    let mut out = vec![LogicBit::Zero; width as usize];
    for src in 0..(width - shift) as usize {
        out[src + shift as usize] = bits[src];
    }
    Value4::new(width, signedness, out)
}

fn known_bits_to_u128(bits: &[LogicBit]) -> u128 {
    debug_assert!(bits.len() <= 128);
    debug_assert!(bits.iter().all(|bit| bit.is_known_01()));
    let mut out = 0u128;
    for (idx, bit) in bits.iter().enumerate() {
        if *bit == LogicBit::One {
            out |= 1u128 << idx;
        }
    }
    out
}

fn u128_to_known_bits_lsb(value: u128, width: usize) -> Vec<LogicBit> {
    let mut out = Vec::with_capacity(width);
    for idx in 0..width {
        out.push(if (value >> idx) & 1 == 0 {
            LogicBit::Zero
        } else {
            LogicBit::One
        });
    }
    out
}

fn mask_for_width(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else if width == 0 {
        0
    } else {
        (1u128 << width) - 1
    }
}

fn div_mod_known_bits_lsb(
    numer: &[LogicBit],
    denom: &[LogicBit],
) -> (Vec<LogicBit>, Vec<LogicBit>) {
    debug_assert_eq!(numer.len(), denom.len());
    debug_assert!(numer.iter().all(|bit| bit.is_known_01()));
    debug_assert!(denom.iter().all(|bit| bit.is_known_01()));
    let mut quot = vec![LogicBit::Zero; numer.len()];
    let mut rem = vec![LogicBit::Zero; numer.len()];
    for i in (0..numer.len()).rev() {
        shl1_known_bits_assign_lsb(&mut rem, numer[i]);
        if cmp_known_bits_unsigned_lsb(&rem, denom) != Ordering::Less {
            sub_known_bits_assign_lsb(&mut rem, denom);
            quot[i] = LogicBit::One;
        }
    }
    (quot, rem)
}

fn div_mod_known_bits_by_small_assign_lsb(bits: &mut [LogicBit], divisor: u8) -> u8 {
    debug_assert!(divisor > 0);
    debug_assert!(bits.iter().all(|bit| bit.is_known_01()));
    let mut rem: u8 = 0;
    for i in (0..bits.len()).rev() {
        let cur = (u16::from(rem) << 1) | u16::from(bits[i] == LogicBit::One);
        if cur >= u16::from(divisor) {
            bits[i] = LogicBit::One;
            rem = (cur - u16::from(divisor)) as u8;
        } else {
            bits[i] = LogicBit::Zero;
            rem = cur as u8;
        }
    }
    rem
}

fn mul_known_bits_by_small(bits: &mut [LogicBit], factor: u8) {
    debug_assert!(bits.iter().all(|bit| bit.is_known_01()));
    let mut carry: u16 = 0;
    for bit in bits {
        let cur = u16::from(*bit == LogicBit::One);
        let prod = cur * u16::from(factor) + carry;
        *bit = logic_bit_from_bool((prod & 1) != 0);
        carry = prod >> 1;
    }
}

fn add_small_to_known_bits(bits: &mut [LogicBit], addend: u32) {
    debug_assert!(bits.iter().all(|bit| bit.is_known_01()));
    let mut carry: u64 = u64::from(addend);
    for bit in bits {
        let cur = u64::from(*bit == LogicBit::One);
        let sum = cur + carry;
        *bit = logic_bit_from_bool((sum & 1) != 0);
        carry = sum >> 1;
        if carry == 0 {
            break;
        }
    }
}

fn twos_complement_magnitude_bits_lsb(bits_lsb: &[LogicBit]) -> Vec<LogicBit> {
    debug_assert!(bits_lsb.iter().all(|bit| bit.is_known_01()));
    let mut out: Vec<LogicBit> = bits_lsb
        .iter()
        .map(|bit| match bit {
            LogicBit::Zero => LogicBit::One,
            LogicBit::One => LogicBit::Zero,
            LogicBit::X | LogicBit::Z => unreachable!("known bits expected"),
        })
        .collect();
    let mut carry = true;
    for bit in &mut out {
        if !carry {
            break;
        }
        match *bit {
            LogicBit::Zero => {
                *bit = LogicBit::One;
                carry = false;
            }
            LogicBit::One => {
                *bit = LogicBit::Zero;
            }
            LogicBit::X | LogicBit::Z => unreachable!("known bits expected"),
        }
    }
    out
}

fn bit_and_4(a: LogicBit, b: LogicBit) -> LogicBit {
    match (a, b) {
        (LogicBit::Zero, _) => LogicBit::Zero,
        (_, LogicBit::Zero) => LogicBit::Zero,
        (LogicBit::One, LogicBit::One) => LogicBit::One,
        (LogicBit::One, LogicBit::X | LogicBit::Z) => LogicBit::X,
        (LogicBit::X | LogicBit::Z, LogicBit::One) => LogicBit::X,
        _ => LogicBit::X,
    }
}

fn bit_or_4(a: LogicBit, b: LogicBit) -> LogicBit {
    match (a, b) {
        (LogicBit::One, _) => LogicBit::One,
        (_, LogicBit::One) => LogicBit::One,
        (LogicBit::Zero, LogicBit::Zero) => LogicBit::Zero,
        (LogicBit::Zero, LogicBit::X | LogicBit::Z) => LogicBit::X,
        (LogicBit::X | LogicBit::Z, LogicBit::Zero) => LogicBit::X,
        _ => LogicBit::X,
    }
}

fn bit_xor_4(a: LogicBit, b: LogicBit) -> LogicBit {
    if !a.is_known_01() || !b.is_known_01() {
        return LogicBit::X;
    }
    match (a, b) {
        (LogicBit::Zero, LogicBit::Zero) => LogicBit::Zero,
        (LogicBit::Zero, LogicBit::One) => LogicBit::One,
        (LogicBit::One, LogicBit::Zero) => LogicBit::One,
        (LogicBit::One, LogicBit::One) => LogicBit::Zero,
        _ => LogicBit::X,
    }
}

fn logic_and_bit(a: LogicBit, b: LogicBit) -> LogicBit {
    match (a, b) {
        (LogicBit::Zero, _) => LogicBit::Zero,
        (_, LogicBit::Zero) => LogicBit::Zero,
        (LogicBit::One, LogicBit::One) => LogicBit::One,
        (LogicBit::One, x) | (x, LogicBit::One) => match x {
            LogicBit::X | LogicBit::Z => LogicBit::X,
            LogicBit::Zero => LogicBit::Zero,
            LogicBit::One => LogicBit::One,
        },
        _ => LogicBit::X,
    }
}

fn logic_or_bit(a: LogicBit, b: LogicBit) -> LogicBit {
    match (a, b) {
        (LogicBit::One, _) => LogicBit::One,
        (_, LogicBit::One) => LogicBit::One,
        (LogicBit::Zero, LogicBit::Zero) => LogicBit::Zero,
        (LogicBit::Zero, x) | (x, LogicBit::Zero) => match x {
            LogicBit::X | LogicBit::Z => LogicBit::X,
            LogicBit::Zero => LogicBit::Zero,
            LogicBit::One => LogicBit::One,
        },
        _ => LogicBit::X,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ubits(width: u32, ones: &[u32]) -> Value4 {
        let mut bits = vec![LogicBit::Zero; width as usize];
        for &idx in ones {
            bits[idx as usize] = LogicBit::One;
        }
        Value4::new(width, Signedness::Unsigned, bits)
    }

    fn sbits(width: u32, msb: &str) -> Value4 {
        assert_eq!(msb.len(), width as usize);
        let mut bits = Vec::with_capacity(width as usize);
        for c in msb.chars().rev() {
            bits.push(match c {
                '0' => LogicBit::Zero,
                '1' => LogicBit::One,
                _ => panic!("bad bit char {c}"),
            });
        }
        Value4::new(width, Signedness::Signed, bits)
    }

    #[test]
    fn wide_known_relational_compare_stays_concrete() {
        let wide_zero = ubits(142, &[]);
        let wide_ge_eight = ubits(142, &[0, 32, 71, 103]);
        let eight = ubits(142, &[3]);

        assert_eq!(
            wide_zero
                .cmp_rel(&eight, RelOp::Ge)
                .to_bit_string_msb_first(),
            "0"
        );
        assert_eq!(
            wide_ge_eight
                .cmp_rel(&eight, RelOp::Ge)
                .to_bit_string_msb_first(),
            "1"
        );
    }

    #[test]
    fn wide_rhs_shift_amount_uses_known_bits_not_u128_fallback() {
        let one8 = ubits(8, &[0]);
        let shift_one_wide = ubits(142, &[0]);
        let shift_huge_wide = ubits(142, &[32]);

        assert_eq!(
            one8.shl(&shift_one_wide).to_bit_string_msb_first(),
            "00000010"
        );
        assert_eq!(
            one8.shl(&shift_huge_wide).to_bit_string_msb_first(),
            "00000000"
        );
    }

    #[test]
    fn wide_decode_shape_from_finding_96_evaluates_concretely() {
        let one8 = ubits(8, &[0]);
        let zero8 = ubits(8, &[]);
        let wide_zero = ubits(142, &[]);
        let wide_ge_eight = ubits(142, &[0, 32, 71, 103]);
        let eight = ubits(142, &[3]);

        let zero_case = Value4::ternary(
            &wide_zero.cmp_rel(&eight, RelOp::Ge),
            &zero8,
            &one8.shl(&wide_zero),
        );
        let ge_case = Value4::ternary(
            &wide_ge_eight.cmp_rel(&eight, RelOp::Ge),
            &zero8,
            &one8.shl(&wide_ge_eight),
        );

        assert_eq!(zero_case.to_bit_string_msb_first(), "00000001");
        assert_eq!(ge_case.to_bit_string_msb_first(), "00000000");
    }

    #[test]
    fn wide_arithmetic_beyond_128_bits_stays_concrete() {
        let a = ubits(130, &[129]);
        let b = ubits(130, &[0]);
        let denom = ubits(130, &[4]);
        let numer = ubits(130, &[129, 4]);

        assert_eq!(a.add(&b), ubits(130, &[129, 0]));
        assert_eq!(a.add(&b).sub(&b), a);
        assert_eq!(a.mul(&ubits(2, &[1])), ubits(130, &[]));
        assert_eq!(numer.div(&denom), ubits(130, &[125, 0]));
        assert_eq!(numer.modu(&denom), ubits(130, &[]));
        assert_eq!(
            b.unary_minus(),
            Value4::new(130, Signedness::Unsigned, vec![LogicBit::One; 130])
        );
    }

    #[test]
    fn wide_unary_minus_beyond_128_bits_matches_twos_complement() {
        let x = ubits(136, &[88, 128]);
        let mut expected_ones: Vec<u32> = (88..128).collect();
        expected_ones.extend(129..136);

        assert_eq!(x.unary_minus(), ubits(136, &expected_ones));
    }

    #[test]
    fn wide_mul_beyond_128_bits_accumulates_partial_products() {
        let a = ubits(136, &[0, 88, 128]);
        let b = ubits(136, &[0, 88]);

        assert_eq!(a.mul(&b), ubits(136, &[0, 89, 128]));
    }

    #[test]
    fn mul_width_128_wraps_with_native_fast_path() {
        let all_ones = Value4::new(128, Signedness::Unsigned, vec![LogicBit::One; 128]);
        let two = ubits(128, &[1]);
        let mut expected_bits = vec![LogicBit::One; 128];
        expected_bits[0] = LogicBit::Zero;

        assert_eq!(
            all_ones.mul(&two),
            Value4::new(128, Signedness::Unsigned, expected_bits)
        );
    }

    #[test]
    fn wide_known_to_u32_conversion_uses_low_bits_and_saturates() {
        assert_eq!(ubits(140, &[5]).to_u32_saturating_if_known(), Some(32));
        assert_eq!(
            ubits(140, &[40]).to_u32_saturating_if_known(),
            Some(u32::MAX)
        );
        assert_eq!(ubits(140, &[5]).to_u32_if_known(), Some(32));
        assert_eq!(ubits(140, &[40]).to_u32_if_known(), None);
        assert_eq!(
            Value4::new(140, Signedness::Unsigned, vec![LogicBit::X; 140])
                .to_u32_saturating_if_known(),
            None
        );
    }

    #[test]
    fn wide_string_helpers_are_unbounded_and_known_only() {
        let v = ubits(200, &[64]);
        assert_eq!(
            v.to_decimal_string_if_known().as_deref(),
            Some("18446744073709551616")
        );
        assert_eq!(
            ubits(200, &[5]).to_hex_string_if_known().unwrap(),
            format!("{}20", "0".repeat(48))
        );
        assert_eq!(
            Value4::new(200, Signedness::Unsigned, vec![LogicBit::X; 200])
                .to_decimal_string_if_known(),
            None
        );
    }

    #[test]
    fn signed_decimal_rendering_respects_negative_values() {
        assert_eq!(
            sbits(4, "1111").to_decimal_string_if_known().as_deref(),
            Some("-1")
        );
        assert_eq!(
            sbits(8, "11111101").to_decimal_string_if_known().as_deref(),
            Some("-3")
        );
        assert_eq!(
            ubits(4, &[0, 1, 2, 3])
                .to_decimal_string_if_known()
                .as_deref(),
            Some("15")
        );
    }

    #[test]
    fn parse_numeric_token_accepts_wide_values() {
        let v =
            Value4::parse_numeric_token(200, Signedness::Unsigned, "18446744073709551616").unwrap();
        assert_eq!(v, ubits(200, &[64]));
        assert_eq!(
            Value4::parse_numeric_token(8, Signedness::Unsigned, "0x1ff")
                .unwrap()
                .to_bit_string_msb_first(),
            "11111111"
        );
    }

    #[test]
    fn indexed_slice_plus_selects_upward_window() {
        let v = Value4::parse_numeric_token(8, Signedness::Unsigned, "0xb3").unwrap();
        assert_eq!(
            v.indexed_slice(0, 4, true).to_bit_string_msb_first(),
            "0011"
        );
        assert_eq!(
            v.indexed_slice(4, 4, true).to_bit_string_msb_first(),
            "1011"
        );
        assert_eq!(
            v.indexed_slice(8, 4, true).to_bit_string_msb_first(),
            "xxxx"
        );
    }

    #[test]
    fn indexed_slice_minus_selects_downward_window() {
        let v = Value4::parse_numeric_token(8, Signedness::Unsigned, "0xb3").unwrap();
        assert_eq!(
            v.indexed_slice(7, 4, false).to_bit_string_msb_first(),
            "1011"
        );
        assert_eq!(
            v.indexed_slice(3, 4, false).to_bit_string_msb_first(),
            "0011"
        );
        assert_eq!(
            v.indexed_slice(2, 4, false).to_bit_string_msb_first(),
            "011x"
        );
    }

    #[test]
    fn signed_division_truncates_toward_zero_for_negative_divisors() {
        assert_eq!(
            sbits(8, "00001101")
                .div(&sbits(8, "11111011"))
                .to_bit_string_msb_first(),
            "11111110"
        );
        assert_eq!(
            sbits(8, "11110011")
                .div(&sbits(8, "00000101"))
                .to_bit_string_msb_first(),
            "11111110"
        );
        assert_eq!(
            sbits(8, "11110011")
                .div(&sbits(8, "11111011"))
                .to_bit_string_msb_first(),
            "00000010"
        );
    }

    #[test]
    fn signed_modulo_keeps_dividend_sign_for_negative_divisors() {
        assert_eq!(
            sbits(8, "00001101")
                .modu(&sbits(8, "11111011"))
                .to_bit_string_msb_first(),
            "00000011"
        );
        assert_eq!(
            sbits(8, "11110011")
                .modu(&sbits(8, "00000101"))
                .to_bit_string_msb_first(),
            "11111101"
        );
        assert_eq!(
            sbits(8, "11111001")
                .modu(&sbits(8, "11111101"))
                .to_bit_string_msb_first(),
            "11111111"
        );
    }

    #[test]
    fn divide_and_mod_by_zero_produce_all_x() {
        let numer_u = ubits(8, &[0, 2, 3]);
        let zero_u = ubits(8, &[]);
        assert_eq!(numer_u.div(&zero_u).to_bit_string_msb_first(), "xxxxxxxx");
        assert_eq!(numer_u.modu(&zero_u).to_bit_string_msb_first(), "xxxxxxxx");

        let numer_s = sbits(8, "11110011");
        let zero_s = sbits(8, "00000000");
        assert_eq!(numer_s.div(&zero_s).to_bit_string_msb_first(), "xxxxxxxx");
        assert_eq!(numer_s.modu(&zero_s).to_bit_string_msb_first(), "xxxxxxxx");
    }

    #[test]
    fn mixed_width_signed_division_and_modulo_use_max_width() {
        assert_eq!(
            sbits(4, "1101")
                .div(&sbits(8, "00000010"))
                .to_bit_string_msb_first(),
            "11111111"
        );
        assert_eq!(
            sbits(4, "1101")
                .modu(&sbits(8, "00000010"))
                .to_bit_string_msb_first(),
            "11111111"
        );
        assert_eq!(
            sbits(8, "11110011")
                .div(&sbits(4, "0101"))
                .to_bit_string_msb_first(),
            "11111110"
        );
        assert_eq!(
            sbits(8, "11110011")
                .modu(&sbits(4, "0101"))
                .to_bit_string_msb_first(),
            "11111101"
        );
    }

    #[test]
    fn mixed_signedness_division_and_modulo_extend_operands_before_unsigned_op() {
        assert_eq!(
            sbits(4, "1101")
                .div(&Value4::new(
                    8,
                    Signedness::Unsigned,
                    sbits(8, "00000010").bits
                ))
                .to_bit_string_msb_first(),
            "01111110"
        );
        assert_eq!(
            sbits(4, "1101")
                .modu(&Value4::new(
                    8,
                    Signedness::Unsigned,
                    sbits(8, "00000010").bits
                ))
                .to_bit_string_msb_first(),
            "00000001"
        );
        assert_eq!(
            ubits(4, &[0, 2, 3])
                .div(&sbits(8, "11111110"))
                .to_bit_string_msb_first(),
            "00000000"
        );
        assert_eq!(
            ubits(4, &[0, 2, 3])
                .modu(&sbits(8, "11111110"))
                .to_bit_string_msb_first(),
            "00001101"
        );
    }

    #[test]
    fn mixed_width_signed_multiply_uses_max_width() {
        assert_eq!(
            sbits(4, "1101")
                .mul(&sbits(8, "00000010"))
                .to_bit_string_msb_first(),
            "11111010"
        );
        assert_eq!(
            sbits(8, "11110011")
                .mul(&sbits(4, "0101"))
                .to_bit_string_msb_first(),
            "10111111"
        );
        assert_eq!(
            sbits(4, "1101")
                .mul(&sbits(8, "11111110"))
                .to_bit_string_msb_first(),
            "00000110"
        );
    }

    #[test]
    fn mixed_signedness_multiply_extends_operands_before_unsigned_op() {
        assert_eq!(
            sbits(4, "1101")
                .mul(&Value4::new(
                    8,
                    Signedness::Unsigned,
                    sbits(8, "00000010").bits
                ))
                .to_bit_string_msb_first(),
            "11111010"
        );
        assert_eq!(
            ubits(4, &[0, 2, 3])
                .mul(&sbits(8, "11111110"))
                .to_bit_string_msb_first(),
            "11100110"
        );
        assert_eq!(
            sbits(8, "11110011")
                .mul(&Value4::new(4, Signedness::Unsigned, sbits(4, "0101").bits))
                .to_bit_string_msb_first(),
            "10111111"
        );
    }
}
