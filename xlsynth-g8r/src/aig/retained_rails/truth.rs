// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::{Edge, Formals, Key, RailBudget, SignedCone};
use crate::aig::gate::{AigNode, AigOperand, GateFn};

const MAX_BITS: usize = 6;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct CutBit {
    pub port: String,
    pub width: usize,
    pub bit: usize,
}

pub(crate) struct CutSignature<'a> {
    function: &'a str,
    bits: &'a [CutBit],
    table: u64,
}

impl<'a> CutSignature<'a> {
    /// Bit q is the value on assignment q of the ordered coordinates, with
    /// coordinate zero toggling fastest. The caller supplies every row.
    pub(crate) fn from_rows(function: &'a str, bits: &'a [CutBit], rows: u64) -> Option<Self> {
        if bits.len() > MAX_BITS
            || bits.iter().enumerate().any(|(i, b)| {
                b.bit >= b.width
                    || bits[..i]
                        .iter()
                        .any(|p| p.port == b.port && (p.width != b.width || p.bit == b.bit))
            })
        {
            return None;
        }
        let mut n = 1usize << bits.len();
        if n < 64 && rows >> n != 0 {
            return None;
        }
        let mut table = rows;
        while n < 64 {
            table |= table << n;
            n *= 2;
        }
        Some(Self {
            function,
            bits,
            table,
        })
    }

    pub(crate) fn rows(&self) -> u64 {
        let n = 1usize << self.bits.len();
        if n == 64 {
            self.table
        } else {
            self.table & ((1u64 << n) - 1)
        }
    }

    pub(crate) fn coordinates(&self) -> &[CutBit] {
        self.bits
    }
}

pub(crate) struct CutIndex {
    function: String,
    bits: Vec<CutBit>,
    values: Vec<Option<u64>>,
    matches: HashMap<u64, (usize, AigOperand)>,
}

impl CutIndex {
    /// Compute exact values on an ordered, full-care primary-input domain.
    /// A node with an unproven dependency outside the cut is unknown.
    pub(crate) fn build(f: &GateFn, bits: &[CutBit], budget: &mut RailBudget) -> Option<Self> {
        if bits.len() > MAX_BITS || !budget.take_graph(f) || !budget.take(f.inputs.len()) {
            return None;
        }
        let formals = Formals::new(f)?;
        let mut primary = HashMap::new();
        for (i, bit) in bits.iter().enumerate() {
            let port = formals.0.get(bit.port.as_str())?;
            if bit.width != port.get_bit_count() || bit.bit >= bit.width {
                return None;
            }
            let edge = *port.bit_vector.get_lsb(bit.bit);
            match f.gates.get(edge.node.id)? {
                AigNode::Input {
                    name, lsb_index, ..
                } if name == &bit.port && *lsb_index == bit.bit && !edge.negated => {}
                _ => return None,
            }
            if primary.insert(edge.node, column(i)).is_some() {
                return None;
            }
        }
        let mut result = Self {
            function: f.name.clone(),
            bits: bits.to_vec(),
            values: vec![None; f.gates.len()],
            matches: HashMap::new(),
        };
        let mut depth = vec![None; f.gates.len()];
        for edge in f.post_order_operands(false) {
            if depth[edge.node.id].is_some() {
                continue;
            }
            let (value, level) = match &f.gates[edge.node.id] {
                AigNode::Literal { value, .. } => (Some(if *value { u64::MAX } else { 0 }), 0),
                AigNode::Input { .. } => (primary.get(&edge.node).copied(), 0),
                AigNode::And2 { a, b, .. } => {
                    let value = match (result.value(*a), result.value(*b)) {
                        (Some(x), Some(y)) => Some(x & y),
                        (Some(0), _) | (_, Some(0)) => Some(0),
                        _ => None,
                    };
                    (value, 1 + usize::max(depth[a.node.id]?, depth[b.node.id]?))
                }
            };
            depth[edge.node.id] = Some(level);
            if let Some(table) = value {
                result.values[edge.node.id] = Some(table);
                let canonical = table.min(!table);
                let rail = AigOperand {
                    node: edge.node,
                    negated: table != canonical,
                };
                let class = result.matches.entry(canonical).or_insert((level, rail));
                if (level, rail.node.id) < (class.0, class.1.node.id) {
                    *class = (level, rail);
                }
            }
        }
        Some(result)
    }

    pub(crate) fn signature(&self, edge: AigOperand) -> Option<CutSignature<'_>> {
        Some(CutSignature {
            function: &self.function,
            bits: &self.bits,
            table: self.value(edge)?,
        })
    }

    pub(crate) fn resolve(&self, sig: &CutSignature<'_>) -> Option<AigOperand> {
        if self.function != sig.function || self.bits != sig.coordinates() {
            return None;
        }
        self.first(sig.table)
    }

    /// Minimum depth, then lowest existing node index. The returned phase
    /// matches the queried operand even for complemented or duplicate nodes.
    pub(crate) fn best_equivalent(&self, edge: AigOperand) -> Option<AigOperand> {
        self.first(self.value(edge)?)
    }

    pub(super) fn first(&self, table: u64) -> Option<AigOperand> {
        let canonical = table.min(!table);
        let rail = self.matches.get(&canonical)?.1;
        Some(if table == canonical {
            rail
        } else {
            rail.negate()
        })
    }

    fn value(&self, edge: AigOperand) -> Option<u64> {
        let table = self.values.get(edge.node.id).copied().flatten()?;
        Some(if edge.negated { !table } else { table })
    }
}

impl SignedCone {
    pub(super) fn cut_rows(&self) -> Option<(Vec<CutBit>, u64)> {
        let mut bits = Vec::new();
        for key in &self.nodes {
            if let Key::Input(bit) = key {
                if !bits.contains(bit) {
                    if bits.len() == MAX_BITS {
                        return None;
                    }
                    bits.push(bit.clone());
                }
            }
        }
        bits.sort_unstable();
        let mut values = Vec::with_capacity(self.nodes.len());
        for key in &self.nodes {
            let value = match key {
                Key::Literal(value) => {
                    if *value {
                        u64::MAX
                    } else {
                        0
                    }
                }
                Key::Input(bit) => column(bits.binary_search(bit).ok()?),
                Key::And([a, b]) => signed(&values, *a) & signed(&values, *b),
            };
            values.push(value);
        }
        let table = signed(&values, self.root);
        let rows = CutSignature {
            function: &self.function,
            bits: &bits,
            table,
        }
        .rows();
        Some((bits, rows))
    }
}

fn signed(values: &[u64], edge: Edge) -> u64 {
    let value = values[edge.class];
    if edge.inverted { !value } else { value }
}

fn column(i: usize) -> u64 {
    // The 64 rows enumerate assignments with bit zero as the fastest varying
    // input.
    const COLUMNS: [u64; 6] = [
        0xaaaa_aaaa_aaaa_aaaa,
        0xcccc_cccc_cccc_cccc,
        0xf0f0_f0f0_f0f0_f0f0,
        0xff00_ff00_ff00_ff00,
        0xffff_0000_ffff_0000,
        0xffff_ffff_0000_0000,
    ];
    COLUMNS[i]
}
