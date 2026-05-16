// SPDX-License-Identifier: Apache-2.0

//! Normalized per-bit connectivity for parsed gate-level netlists.
//!
//! The raw parser preserves Verilog syntax such as concat, selects, continuous
//! assigns, and plain `tran` primitives. This module lowers those syntax forms
//! once into canonical per-bit connectivity so downstream consumers do not each
//! need to rediscover Verilog wiring semantics.

use crate::netlist::bit_ref;
use crate::netlist::parse::{
    AssignExpr, InstIndex, Net, NetIndex, NetRef, NetlistAssignKind, NetlistModule, PortDirection,
    PortId, Span,
};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

pub type BitIndex = usize;

/// One normalized bit source used by pin bindings and assign leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitSource {
    Bit(BitIndex),
    Literal(bool),
    Unknown,
}

/// One normalized per-bit assign expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BitExpr {
    Source(BitSource),
    Not(Box<BitExpr>),
    And(Box<BitExpr>, Box<BitExpr>),
    Or(Box<BitExpr>, Box<BitExpr>),
    Xor(Box<BitExpr>, Box<BitExpr>),
}

impl BitExpr {
    /// Appends every normalized net bit referenced by this expression.
    pub fn collect_source_bits(&self, out: &mut Vec<BitIndex>) {
        match self {
            Self::Source(BitSource::Bit(bit_idx)) => out.push(*bit_idx),
            Self::Source(BitSource::Literal(_) | BitSource::Unknown) => {}
            Self::Not(inner) => inner.collect_source_bits(out),
            Self::And(lhs, rhs) | Self::Or(lhs, rhs) | Self::Xor(lhs, rhs) => {
                lhs.collect_source_bits(out);
                rhs.collect_source_bits(out);
            }
        }
    }
}

/// One normalized continuous assign statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedAssign {
    pub lhs_bits: Vec<BitIndex>,
    pub rhs_bits: Vec<BitExpr>,
    pub rendered_lhs: String,
    pub span: Span,
}

/// One normalized instance pin binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedConnection {
    pub port: PortId,
    pub bits: Vec<BitSource>,
}

/// One normalized instance view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedInstance {
    pub raw_index: InstIndex,
    pub type_name: PortId,
    pub instance_name: PortId,
    pub connections: Vec<NormalizedConnection>,
    pub inst_lineno: u32,
    pub inst_colno: u32,
}

/// One normalized module port view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedPort {
    pub direction: PortDirection,
    pub width: Option<(u32, u32)>,
    pub name: PortId,
    pub bits: Vec<BitIndex>,
}

/// Canonical per-bit connectivity for one parsed module.
pub struct NormalizedNetlistModule<'a> {
    pub raw: &'a NetlistModule,
    bits: Vec<bit_ref::NetBit>,
    index_by_bit: HashMap<bit_ref::NetBit, BitIndex>,
    bits_by_net: Vec<Vec<BitIndex>>,
    canonical_bits: Vec<BitIndex>,
    pub ports: Vec<NormalizedPort>,
    pub instances: Vec<NormalizedInstance>,
    pub assigns: Vec<NormalizedAssign>,
}

impl<'a> NormalizedNetlistModule<'a> {
    /// Builds canonical per-bit connectivity for one parsed module.
    pub fn new(
        module: &'a NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<Self> {
        let (bits, index_by_bit, bits_by_net) = build_bit_index(nets)?;
        let canonical_bits =
            build_tran_aliases(module, nets, interner, bits.as_slice(), &index_by_bit)?;
        let ports = normalize_ports(module, nets, &bits_by_net, canonical_bits.as_slice())?;
        let instances = normalize_instances(
            module,
            nets,
            interner,
            &index_by_bit,
            canonical_bits.as_slice(),
        )?;
        let assigns = normalize_assigns(
            module,
            nets,
            interner,
            &index_by_bit,
            canonical_bits.as_slice(),
        )?;
        Ok(Self {
            raw: module,
            bits,
            index_by_bit,
            bits_by_net,
            canonical_bits,
            ports,
            instances,
            assigns,
        })
    }

    pub fn bit_count(&self) -> usize {
        self.bits.len()
    }

    pub fn bit(&self, bit_idx: BitIndex) -> bit_ref::NetBit {
        self.bits[bit_idx]
    }

    pub fn bit_index(&self, bit: bit_ref::NetBit) -> Result<BitIndex> {
        self.index_by_bit.get(&bit).copied().ok_or_else(|| {
            anyhow!(
                "net bit NetIndex({})[{}] is not present in normalized bit index",
                bit.net.0,
                bit.bit_number
            )
        })
    }

    pub fn net_bits(&self, net: NetIndex) -> &[BitIndex] {
        &self.bits_by_net[net.0]
    }

    pub fn canonical_bit(&self, bit_idx: BitIndex) -> BitIndex {
        self.canonical_bits[bit_idx]
    }

    pub fn canonical_bits(&self) -> &[BitIndex] {
        self.canonical_bits.as_slice()
    }

    pub fn render_bit(
        &self,
        bit_idx: BitIndex,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> String {
        bit_ref::render_net_bit(self.bits[bit_idx], nets, interner)
    }

    pub fn render_source(
        &self,
        source: BitSource,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> String {
        match source {
            BitSource::Bit(bit_idx) => self.render_bit(bit_idx, nets, interner),
            BitSource::Literal(false) => "1'b0".to_string(),
            BitSource::Literal(true) => "1'b1".to_string(),
            BitSource::Unknown => "1'bx".to_string(),
        }
    }

    pub fn render_sources(
        &self,
        sources: &[BitSource],
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> String {
        match sources {
            [] => "<unconnected>".to_string(),
            [source] => self.render_source(*source, nets, interner),
            _ => {
                let mut parts: Vec<String> = sources
                    .iter()
                    .rev()
                    .copied()
                    .map(|source| self.render_source(source, nets, interner))
                    .collect();
                parts.shrink_to_fit();
                format!("{{{}}}", parts.join(", "))
            }
        }
    }
}

fn build_bit_index(
    nets: &[Net],
) -> Result<(
    Vec<bit_ref::NetBit>,
    HashMap<bit_ref::NetBit, BitIndex>,
    Vec<Vec<BitIndex>>,
)> {
    let mut bits = Vec::new();
    let mut index_by_bit = HashMap::new();
    let mut bits_by_net = vec![Vec::new(); nets.len()];
    for (net_raw_idx, net) in nets.iter().enumerate() {
        let net_idx = NetIndex(net_raw_idx);
        for offset in 0..net.width_bits() {
            let bit_number = net.bit_number(offset).ok_or_else(|| {
                anyhow!(
                    "internal error computing bit {} for net index {}",
                    offset,
                    net_raw_idx
                )
            })?;
            let bit = bit_ref::NetBit {
                net: net_idx,
                bit_number,
            };
            let bit_idx = bits.len();
            bits.push(bit);
            index_by_bit.insert(bit, bit_idx);
            bits_by_net[net_raw_idx].push(bit_idx);
        }
    }
    Ok((bits, index_by_bit, bits_by_net))
}

fn build_tran_aliases(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    bits: &[bit_ref::NetBit],
    index_by_bit: &HashMap<bit_ref::NetBit, BitIndex>,
) -> Result<Vec<BitIndex>> {
    let mut parents: Vec<BitIndex> = (0..bits.len()).collect();
    for tran in module
        .assigns
        .iter()
        .filter(|assign| assign.kind == NetlistAssignKind::Tran)
    {
        let AssignExpr::Leaf(rhs) = &tran.rhs else {
            unreachable!("tran parser only constructs leaf net refs")
        };
        let lhs_bits = bit_ref::net_ref_lsb_targets(&tran.lhs, nets, interner)?;
        let rhs_bits = bit_ref::net_ref_lsb_targets(rhs, nets, interner)?;
        if lhs_bits.len() != rhs_bits.len() {
            return Err(anyhow!(
                "plain tran terminals must have equal widths; '{}' has {} bit(s) and '{}' has {} bit(s)",
                bit_ref::render_net_ref(&tran.lhs, nets, interner),
                lhs_bits.len(),
                bit_ref::render_net_ref(rhs, nets, interner),
                rhs_bits.len()
            ));
        }
        for (lhs_bit, rhs_bit) in lhs_bits.into_iter().zip(rhs_bits) {
            let lhs_bit_idx = bit_index(index_by_bit, lhs_bit)?;
            let rhs_bit_idx = bit_index(index_by_bit, rhs_bit)?;
            union_aliases(parents.as_mut_slice(), lhs_bit_idx, rhs_bit_idx);
        }
    }
    Ok((0..bits.len())
        .map(|bit_idx| find_alias_root(parents.as_mut_slice(), bit_idx))
        .collect())
}

fn bit_index(
    index_by_bit: &HashMap<bit_ref::NetBit, BitIndex>,
    bit: bit_ref::NetBit,
) -> Result<BitIndex> {
    index_by_bit.get(&bit).copied().ok_or_else(|| {
        anyhow!(
            "net bit NetIndex({})[{}] is not present in normalized bit index",
            bit.net.0,
            bit.bit_number
        )
    })
}

fn find_alias_root(parents: &mut [BitIndex], bit_idx: BitIndex) -> BitIndex {
    let parent = parents[bit_idx];
    if parent == bit_idx {
        bit_idx
    } else {
        let root = find_alias_root(parents, parent);
        parents[bit_idx] = root;
        root
    }
}

fn union_aliases(parents: &mut [BitIndex], lhs_bit_idx: BitIndex, rhs_bit_idx: BitIndex) {
    let lhs_root = find_alias_root(parents, lhs_bit_idx);
    let rhs_root = find_alias_root(parents, rhs_bit_idx);
    if lhs_root == rhs_root {
        return;
    }
    let (root, child) = if lhs_root < rhs_root {
        (lhs_root, rhs_root)
    } else {
        (rhs_root, lhs_root)
    };
    parents[child] = root;
}

fn normalize_ports(
    module: &NetlistModule,
    nets: &[Net],
    bits_by_net: &[Vec<BitIndex>],
    canonical_bits: &[BitIndex],
) -> Result<Vec<NormalizedPort>> {
    module
        .ports
        .iter()
        .map(|port| {
            let net_idx = module.find_net_index(port.name, nets).ok_or_else(|| {
                anyhow!("module port net for symbol {:?} was not found", port.name)
            })?;
            Ok(NormalizedPort {
                direction: port.direction.clone(),
                width: port.width,
                name: port.name,
                bits: bits_by_net[net_idx.0]
                    .iter()
                    .map(|bit_idx| canonical_bits[*bit_idx])
                    .collect(),
            })
        })
        .collect()
}

fn normalize_instances(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    index_by_bit: &HashMap<bit_ref::NetBit, BitIndex>,
    canonical_bits: &[BitIndex],
) -> Result<Vec<NormalizedInstance>> {
    module
        .instances
        .iter()
        .enumerate()
        .map(|(inst_idx, inst)| {
            let connections = inst
                .connections
                .iter()
                .map(|(port, net_ref)| {
                    Ok(NormalizedConnection {
                        port: *port,
                        bits: normalize_net_ref_bits(
                            net_ref,
                            nets,
                            interner,
                            index_by_bit,
                            canonical_bits,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(NormalizedInstance {
                raw_index: InstIndex(inst_idx),
                type_name: inst.type_name,
                instance_name: inst.instance_name,
                connections,
                inst_lineno: inst.inst_lineno,
                inst_colno: inst.inst_colno,
            })
        })
        .collect()
}

fn normalize_assigns(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    index_by_bit: &HashMap<bit_ref::NetBit, BitIndex>,
    canonical_bits: &[BitIndex],
) -> Result<Vec<NormalizedAssign>> {
    let mut assigns = Vec::new();
    for assign in module
        .assigns
        .iter()
        .filter(|assign| assign.kind == NetlistAssignKind::Continuous)
    {
        let lhs_bits: Vec<BitIndex> = bit_ref::net_ref_lsb_targets(&assign.lhs, nets, interner)?
            .into_iter()
            .map(|bit| bit_index(index_by_bit, bit).map(|bit_idx| canonical_bits[bit_idx]))
            .collect::<Result<Vec<_>>>()?;
        let rhs_width = bit_ref::assign_expr_width_bits(&assign.rhs, nets, interner)?;
        let allow_literal_zero_extend = is_bare_literal_assign_expr(&assign.rhs);
        if lhs_bits.len() != rhs_width && !allow_literal_zero_extend {
            return Err(anyhow!(
                "assign to '{}' has lhs width {} but rhs width {}; normalized netlist requires exact-width assigns except for bare literal tie-offs",
                bit_ref::render_net_ref(&assign.lhs, nets, interner),
                lhs_bits.len(),
                rhs_width
            ));
        }
        let rhs_bits = normalize_assign_expr_bits(
            &assign.rhs,
            lhs_bits.len(),
            allow_literal_zero_extend,
            nets,
            interner,
            index_by_bit,
            canonical_bits,
        )?;
        assigns.push(NormalizedAssign {
            lhs_bits,
            rhs_bits,
            rendered_lhs: bit_ref::render_net_ref(&assign.lhs, nets, interner),
            span: assign.span,
        });
    }
    Ok(assigns)
}

fn normalize_assign_expr_bits(
    expr: &AssignExpr,
    expected_width: usize,
    allow_bare_literal_zero_extend: bool,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    index_by_bit: &HashMap<bit_ref::NetBit, BitIndex>,
    canonical_bits: &[BitIndex],
) -> Result<Vec<BitExpr>> {
    match expr {
        AssignExpr::Leaf(net_ref) => {
            let mut bits =
                normalize_net_ref_bits(net_ref, nets, interner, index_by_bit, canonical_bits)?;
            if bits.len() != expected_width {
                if allow_bare_literal_zero_extend
                    && matches!(net_ref, NetRef::Literal(_))
                    && bits.len() <= expected_width
                {
                    bits.resize(expected_width, BitSource::Literal(false));
                } else {
                    return Err(anyhow!(
                        "assign expression '{}' has width {} but expected {}",
                        bit_ref::render_net_ref(net_ref, nets, interner),
                        bits.len(),
                        expected_width
                    ));
                }
            }
            Ok(bits.into_iter().map(BitExpr::Source).collect())
        }
        AssignExpr::Not(inner) => Ok(normalize_assign_expr_bits(
            inner,
            expected_width,
            false,
            nets,
            interner,
            index_by_bit,
            canonical_bits,
        )?
        .into_iter()
        .map(|bit| BitExpr::Not(Box::new(bit)))
        .collect()),
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            let lhs_bits = normalize_assign_expr_bits(
                lhs,
                expected_width,
                false,
                nets,
                interner,
                index_by_bit,
                canonical_bits,
            )?;
            let rhs_bits = normalize_assign_expr_bits(
                rhs,
                expected_width,
                false,
                nets,
                interner,
                index_by_bit,
                canonical_bits,
            )?;
            Ok(lhs_bits
                .into_iter()
                .zip(rhs_bits)
                .map(|(lhs, rhs)| match expr {
                    AssignExpr::And(_, _) => BitExpr::And(Box::new(lhs), Box::new(rhs)),
                    AssignExpr::Or(_, _) => BitExpr::Or(Box::new(lhs), Box::new(rhs)),
                    AssignExpr::Xor(_, _) => BitExpr::Xor(Box::new(lhs), Box::new(rhs)),
                    _ => unreachable!("binary assign expression matched as non-binary"),
                })
                .collect())
        }
    }
}

fn normalize_net_ref_bits(
    net_ref: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    index_by_bit: &HashMap<bit_ref::NetBit, BitIndex>,
    canonical_bits: &[BitIndex],
) -> Result<Vec<BitSource>> {
    bit_ref::net_ref_lsb_bit_refs(net_ref, nets, interner)?
        .into_iter()
        .map(|bit_ref| match bit_ref {
            bit_ref::NetBitRef::Net(bit) => {
                bit_index(index_by_bit, bit).map(|bit_idx| BitSource::Bit(canonical_bits[bit_idx]))
            }
            bit_ref::NetBitRef::Literal(value) => Ok(BitSource::Literal(value)),
            bit_ref::NetBitRef::Unknown => Ok(BitSource::Unknown),
        })
        .collect()
}

fn is_bare_literal_assign_expr(expr: &AssignExpr) -> bool {
    matches!(expr, AssignExpr::Leaf(NetRef::Literal(_)))
}

#[cfg(test)]
mod tests {
    use super::{BitExpr, BitSource, NormalizedNetlistModule};
    use crate::netlist::parse::{Parser, TokenScanner};

    fn parse_single_module(
        src: &str,
    ) -> (
        crate::netlist::parse::NetlistModule,
        Vec<crate::netlist::parse::Net>,
        string_interner::StringInterner<
            string_interner::backend::StringBackend<string_interner::symbol::SymbolU32>,
        >,
    ) {
        let lines: Vec<String> = src.lines().map(|line| line.to_string()).collect();
        let lookup = move |lineno: u32| lines.get((lineno - 1) as usize).cloned();
        let scanner = TokenScanner::with_line_lookup(
            std::io::Cursor::new(src.as_bytes().to_vec()),
            Box::new(lookup),
        );
        let mut parser = Parser::new(scanner);
        let mut modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        (modules.remove(0), parser.nets, parser.interner)
    }

    #[test]
    fn normalizes_concat_assign_and_tran_aliases() {
        let (module, nets, interner) = parse_single_module(
            r#"
module top(a, y);
  input [1:0] a;
  output y;
  wire [1:0] a;
  wire y;
  wire [1:0] tmp;
  wire y_alias;
  assign tmp = {a[0], a[1]};
  tran(y, y_alias);
  BUF u0 (.A(tmp[1]), .Y(y_alias));
endmodule
"#,
        );
        let normalized =
            NormalizedNetlistModule::new(&module, &nets, &interner).expect("normalize netlist");
        assert_eq!(normalized.assigns.len(), 1);
        assert_eq!(normalized.assigns[0].lhs_bits.len(), 2);
        assert!(matches!(
            normalized.assigns[0].rhs_bits.as_slice(),
            [
                BitExpr::Source(BitSource::Bit(_)),
                BitExpr::Source(BitSource::Bit(_))
            ]
        ));
        let y = normalized
            .ports
            .iter()
            .find(|port| interner.resolve(port.name) == Some("y"))
            .expect("find y port");
        let y_alias = nets
            .iter()
            .position(|net| interner.resolve(net.name) == Some("y_alias"))
            .map(crate::netlist::parse::NetIndex)
            .expect("find y_alias");
        assert_eq!(
            y.bits,
            vec![normalized.canonical_bit(normalized.net_bits(y_alias)[0])]
        );
    }
}
