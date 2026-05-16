// SPDX-License-Identifier: Apache-2.0

//! Bit-level helpers for parsed gate-level net references.

use crate::netlist::parse::{AssignExpr, Net, NetIndex, NetRef};
use anyhow::{Result, anyhow};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// One concrete bit of a parsed net, identified by declared bit number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NetBit {
    pub net: NetIndex,
    pub bit_number: u32,
}

/// One bit referenced by an expression that is restricted to wiring and
/// constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NetBitRef {
    Net(NetBit),
    Literal(bool),
    Unknown,
}

/// Returns the width of a packed select range.
pub fn select_width_bits(msb: u32, lsb: u32) -> usize {
    (u32::abs_diff(msb, lsb) as usize) + 1
}

/// Returns the declared bit number at an lsb-based offset within a select.
pub fn select_bit_number(msb: u32, lsb: u32, bit_offset: usize) -> Option<u32> {
    if bit_offset >= select_width_bits(msb, lsb) {
        return None;
    }
    let offset = bit_offset as u32;
    if msb >= lsb {
        Some(lsb + offset)
    } else {
        Some(lsb - offset)
    }
}

/// Resolves a net name for diagnostics.
pub fn net_name(
    idx: NetIndex,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> String {
    nets.get(idx.0)
        .and_then(|net| interner.resolve(net.name))
        .unwrap_or("<unknown>")
        .to_string()
}

/// Renders a concrete net bit using scalar spelling when the net is 1-bit.
pub fn render_net_bit(
    bit: NetBit,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> String {
    if nets[bit.net.0].width_bits() == 1 {
        net_name(bit.net, nets, interner)
    } else {
        format!("{}[{}]", net_name(bit.net, nets, interner), bit.bit_number)
    }
}

/// Renders a parsed net reference for diagnostics.
pub fn render_net_ref(
    net_ref: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> String {
    match net_ref {
        NetRef::Simple(idx) => net_name(*idx, nets, interner),
        NetRef::BitSelect(idx, bit) => format!("{}[{}]", net_name(*idx, nets, interner), bit),
        NetRef::PartSelect(idx, msb, lsb) => {
            format!("{}[{}:{}]", net_name(*idx, nets, interner), msb, lsb)
        }
        NetRef::Literal(bits) => format!("{}", bits),
        NetRef::UnknownLiteral(width) => format!("{}'hx", width),
        NetRef::Unconnected => "<unconnected>".to_string(),
        NetRef::Concat(elems) => {
            let parts: Vec<String> = elems
                .iter()
                .map(|elem| render_net_ref(elem, nets, interner))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Returns a net reference's packed width in bits.
pub fn net_ref_width_bits(
    net_ref: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<usize> {
    match net_ref {
        NetRef::Simple(idx) => {
            let net = nets.get(idx.0).ok_or_else(|| {
                anyhow!(
                    "net reference '{}' uses out-of-range net index {}",
                    render_net_ref(net_ref, nets, interner),
                    idx.0
                )
            })?;
            Ok(net.width_bits())
        }
        NetRef::BitSelect(idx, bit) => {
            let net = nets.get(idx.0).ok_or_else(|| {
                anyhow!(
                    "bit-select '{}' uses out-of-range net index {}",
                    render_net_ref(net_ref, nets, interner),
                    idx.0
                )
            })?;
            if net.bit_offset(*bit).is_none() {
                return Err(anyhow!(
                    "bit {} out of range for net '{}'",
                    bit,
                    net_name(*idx, nets, interner)
                ));
            }
            Ok(1)
        }
        NetRef::PartSelect(idx, msb, lsb) => {
            let net = nets.get(idx.0).ok_or_else(|| {
                anyhow!(
                    "part-select '{}' uses out-of-range net index {}",
                    render_net_ref(net_ref, nets, interner),
                    idx.0
                )
            })?;
            let width = select_width_bits(*msb, *lsb);
            for offset in 0..width {
                let bit_number = select_bit_number(*msb, *lsb, offset).ok_or_else(|| {
                    anyhow!(
                        "invalid part-select [{}:{}] on net '{}'",
                        msb,
                        lsb,
                        net_name(*idx, nets, interner)
                    )
                })?;
                if net.bit_offset(bit_number).is_none() {
                    return Err(anyhow!(
                        "bit {} out of range for net '{}'",
                        bit_number,
                        net_name(*idx, nets, interner)
                    ));
                }
            }
            Ok(width)
        }
        NetRef::Literal(bits) => Ok(bits.get_bit_count()),
        NetRef::UnknownLiteral(width) => Ok(*width),
        NetRef::Unconnected => Ok(0),
        NetRef::Concat(elems) => {
            if elems.is_empty() {
                return Err(anyhow!("empty concatenation is unsupported"));
            }
            elems.iter().try_fold(0usize, |acc, elem| {
                Ok(acc + net_ref_width_bits(elem, nets, interner)?)
            })
        }
    }
}

/// Expands a reference into lsb-first bit sources.
pub fn net_ref_lsb_bit_refs(
    net_ref: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<Vec<NetBitRef>> {
    match net_ref {
        NetRef::Simple(idx) => {
            let net = nets.get(idx.0).ok_or_else(|| {
                anyhow!(
                    "net reference '{}' uses out-of-range net index {}",
                    render_net_ref(net_ref, nets, interner),
                    idx.0
                )
            })?;
            let mut bits = Vec::with_capacity(net.width_bits());
            for offset in 0..net.width_bits() {
                let bit_number = net.bit_number(offset).ok_or_else(|| {
                    anyhow!(
                        "internal error computing bit {} for net '{}'",
                        offset,
                        net_name(*idx, nets, interner)
                    )
                })?;
                bits.push(NetBitRef::Net(NetBit {
                    net: *idx,
                    bit_number,
                }));
            }
            Ok(bits)
        }
        NetRef::BitSelect(idx, bit) => {
            net_ref_width_bits(net_ref, nets, interner)?;
            Ok(vec![NetBitRef::Net(NetBit {
                net: *idx,
                bit_number: *bit,
            })])
        }
        NetRef::PartSelect(idx, msb, lsb) => {
            let width = net_ref_width_bits(net_ref, nets, interner)?;
            let mut bits = Vec::with_capacity(width);
            for offset in 0..width {
                let bit_number = select_bit_number(*msb, *lsb, offset).ok_or_else(|| {
                    anyhow!(
                        "invalid part-select [{}:{}] on net '{}'",
                        msb,
                        lsb,
                        net_name(*idx, nets, interner)
                    )
                })?;
                bits.push(NetBitRef::Net(NetBit {
                    net: *idx,
                    bit_number,
                }));
            }
            Ok(bits)
        }
        NetRef::Literal(bits) => {
            let mut out = Vec::with_capacity(bits.get_bit_count());
            for i in 0..bits.get_bit_count() {
                out.push(NetBitRef::Literal(bits.get_bit(i).unwrap_or(false)));
            }
            Ok(out)
        }
        NetRef::UnknownLiteral(width) => Ok(vec![NetBitRef::Unknown; *width]),
        NetRef::Unconnected => Ok(Vec::new()),
        NetRef::Concat(elems) => {
            if elems.is_empty() {
                return Err(anyhow!("empty concatenation is unsupported"));
            }
            let mut out = Vec::new();
            for elem in elems.iter().rev() {
                out.extend(net_ref_lsb_bit_refs(elem, nets, interner)?);
            }
            Ok(out)
        }
    }
}

/// Expands a left-hand-side reference into lsb-first concrete target bits.
pub fn net_ref_lsb_targets(
    net_ref: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<Vec<NetBit>> {
    let mut out = Vec::new();
    for bit_ref in net_ref_lsb_bit_refs(net_ref, nets, interner)? {
        match bit_ref {
            NetBitRef::Net(bit) => out.push(bit),
            NetBitRef::Literal(_) => {
                return Err(anyhow!(
                    "left-hand side '{}' cannot contain a literal",
                    render_net_ref(net_ref, nets, interner)
                ));
            }
            NetBitRef::Unknown => {
                return Err(anyhow!(
                    "left-hand side '{}' cannot contain an unknown literal",
                    render_net_ref(net_ref, nets, interner)
                ));
            }
        }
    }
    if out.is_empty() {
        return Err(anyhow!(
            "left-hand side '{}' does not target any net bits",
            render_net_ref(net_ref, nets, interner)
        ));
    }
    Ok(out)
}

/// Returns the width of a supported assign expression.
pub fn assign_expr_width_bits(
    expr: &AssignExpr,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<usize> {
    match expr {
        AssignExpr::Leaf(net_ref) => net_ref_width_bits(net_ref, nets, interner),
        AssignExpr::Not(inner) => assign_expr_width_bits(inner, nets, interner),
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            let lhs_width = assign_expr_width_bits(lhs, nets, interner)?;
            let rhs_width = assign_expr_width_bits(rhs, nets, interner)?;
            if lhs_width != rhs_width {
                return Err(anyhow!(
                    "bitwise assign expression width mismatch: lhs {} bits rhs {} bits",
                    lhs_width,
                    rhs_width
                ));
            }
            Ok(lhs_width)
        }
    }
}
