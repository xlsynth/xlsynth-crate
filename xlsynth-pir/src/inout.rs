// SPDX-License-Identifier: Apache-2.0

//! Passive whole-net, constant-select and concatenated block port bindings.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use crate::ir::{InoutNet, Type};

/// One contiguous piece of an electrical binding. Select bounds are inclusive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InoutPiece<N = String> {
    Whole(N),
    Select { net: N, hi: usize, lo: usize },
}

/// A single piece or an MSB-first concatenation. Whole aggregates retain their
/// PIR type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InoutBinding<N = String> {
    Piece(InoutPiece<N>),
    Concat(Vec<InoutPiece<N>>),
}

impl<N> From<N> for InoutBinding<N> {
    fn from(net: N) -> Self {
        Self::Piece(InoutPiece::Whole(net))
    }
}

impl<N> InoutBinding<N> {
    pub fn pieces(&self) -> &[InoutPiece<N>] {
        match self {
            Self::Piece(piece) => std::slice::from_ref(piece),
            Self::Concat(pieces) => pieces,
        }
    }

    /// Converts builder handles to local names while checking ownership.
    pub(crate) fn try_map<M, E>(
        &self,
        mut f: impl FnMut(&N) -> Result<M, E>,
    ) -> Result<InoutBinding<M>, E> {
        let mut piece = |p: &InoutPiece<N>| {
            Ok(match p {
                InoutPiece::Whole(n) => InoutPiece::Whole(f(n)?),
                InoutPiece::Select { net, hi, lo } => InoutPiece::Select {
                    net: f(net)?,
                    hi: *hi,
                    lo: *lo,
                },
            })
        };
        Ok(match self {
            Self::Piece(p) => InoutBinding::Piece(piece(p)?),
            Self::Concat(ps) => {
                InoutBinding::Concat(ps.iter().map(piece).collect::<Result<_, E>>()?)
            }
        })
    }
}

impl<N: fmt::Display> fmt::Display for InoutBinding<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let concat = matches!(self, Self::Concat(_));
        if concat {
            write!(f, "{{")?;
        }
        for (i, p) in self.pieces().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match p {
                InoutPiece::Whole(n) => write!(f, "{n}")?,
                InoutPiece::Select { net, hi, lo } if hi == lo => write!(f, "{net}[{lo}]")?,
                InoutPiece::Select { net, hi, lo } => write!(f, "{net}[{hi}:{lo}]")?,
            }
        }
        if concat {
            write!(f, "}}")?;
        }
        Ok(())
    }
}

/// Corresponding ascending runs of bits on a port and a local net.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InoutSpan {
    pub net: usize,
    pub net_start: usize,
    pub port_start: usize,
    pub width: usize,
}

/// Resolves local net names once per block for use by verification and RTL
/// emission.
pub struct InoutNetMap<'a> {
    pub nets: &'a [InoutNet],
    indices: HashMap<&'a str, usize>,
}

impl<'a> InoutNetMap<'a> {
    pub fn new(nets: &'a [InoutNet]) -> Result<Self, String> {
        let mut indices = HashMap::with_capacity(nets.len());
        for (idx, net) in nets.iter().enumerate() {
            if net.name.is_empty() || indices.insert(net.name.as_str(), idx).is_some() {
                return Err(format!("empty or duplicate inout net name '{}'", net.name));
            }
        }
        Ok(Self { nets, indices })
    }

    pub fn index(&self, name: &str) -> Result<usize, String> {
        self.indices
            .get(name)
            .copied()
            .ok_or_else(|| format!("undeclared inout net '{name}'"))
    }

    /// Checks the full port type and returns spans in least-significant-first
    /// order.
    pub fn resolve(&self, binding: &InoutBinding, ty: &Type) -> Result<Vec<InoutSpan>, String> {
        resolve_binding(binding, ty, |name| {
            let i = self.index(name)?;
            Ok((i, &self.nets[i]))
        })
    }
}

pub(crate) fn resolve_binding<'a, N>(
    binding: &InoutBinding<N>,
    ty: &Type,
    mut lookup: impl FnMut(&N) -> Result<(usize, &'a InoutNet), String>,
) -> Result<Vec<InoutSpan>, String> {
    let whole = matches!(binding, InoutBinding::Piece(InoutPiece::Whole(_)));
    if !whole && !matches!(ty, Type::Bits(_)) {
        return Err(format!(
            "inout selects and concats require a bits port, got {ty}"
        ));
    }
    if binding.pieces().is_empty() {
        return Err("empty inout concatenation".into());
    }
    let mut port_start = 0usize;
    let mut spans = Vec::with_capacity(binding.pieces().len());
    for p in binding.pieces().iter().rev() {
        let (name, bounds) = match p {
            InoutPiece::Whole(name) => (name, None),
            InoutPiece::Select { net, hi, lo } => (net, Some((*hi, *lo))),
        };
        let (net, def) = lookup(name)?;
        if whole && &def.ty != ty {
            return Err(format!(
                "references net '{}' of type {}, got {ty}",
                def.name, def.ty
            ));
        }
        if !whole && !matches!(def.ty, Type::Bits(_)) {
            return Err(format!(
                "inout selects and concats require a bits net, got '{}' of type {}",
                def.name, def.ty
            ));
        }
        let size = def
            .ty
            .checked_bit_count()
            .ok_or("inout net width overflow")?;
        let (net_start, width) = match bounds {
            Some((hi, lo)) if hi >= lo && hi < size => (lo, hi - lo + 1),
            Some((hi, lo)) => {
                return Err(format!(
                    "invalid select {}[{hi}:{lo}] for {} bits",
                    def.name, size
                ));
            }
            None => (0, size),
        };
        if width == 0 {
            return Err("inout bindings require nonempty nets".into());
        }
        spans.push(InoutSpan {
            net,
            net_start,
            port_start,
            width,
        });
        port_start = port_start
            .checked_add(width)
            .ok_or("inout binding width overflow")?;
    }
    if Some(port_start) != ty.checked_bit_count() {
        return Err(format!(
            "inout binding has width {port_start}, expected {ty}"
        ));
    }
    Ok(spans)
}

/// One public port covering a net segment, at the indicated low port bit.
#[derive(Clone, Copy, Debug)]
pub struct PublicPortSpan {
    pub port: crate::ir::NodeRef,
    pub start: usize,
}

/// A maximal run between public binding boundaries, in local net bit order.
pub struct PublicNetSegment {
    pub start: usize,
    pub width: usize,
    pub ports: Vec<PublicPortSpan>,
}

impl InoutNetMap<'_> {
    /// Partitions nets by the public pins covering them, in interface order.
    pub fn public_segments(
        &self,
        block: &crate::ir::Block,
    ) -> Result<Vec<Vec<PublicNetSegment>>, String> {
        let mut events = vec![BTreeMap::<usize, Vec<(bool, usize)>>::new(); self.nets.len()];
        let mut pieces = Vec::new();
        for port in block.inout_ports() {
            let Some(crate::ir::NodePayload::ExtInoutPort { port_type, net, .. }) =
                block.nodes.get(port.index).map(|n| &n.payload)
            else {
                return Err("invalid public inout port reference".into());
            };
            for span in self.resolve(net, port_type)? {
                let id = pieces.len();
                pieces.push((port, span));
                events[span.net]
                    .entry(span.net_start)
                    .or_default()
                    .push((true, id));
                events[span.net]
                    .entry(span.net_start + span.width)
                    .or_default()
                    .push((false, id));
            }
        }
        let mut result = Vec::with_capacity(self.nets.len());
        for (net, mut ends) in self.nets.iter().zip(events) {
            ends.entry(net.ty.checked_bit_count().ok_or("inout width overflow")?)
                .or_default();
            let mut active = BTreeSet::<usize>::new();
            let mut start = 0;
            let mut segments = Vec::new();
            for (end, changes) in ends {
                if start < end {
                    let ports = active
                        .iter()
                        .map(|&i| {
                            let (port, span) = pieces[i];
                            PublicPortSpan {
                                port,
                                start: span.port_start + (start - span.net_start),
                            }
                        })
                        .collect();
                    segments.push(PublicNetSegment {
                        start,
                        width: end - start,
                        ports,
                    });
                }
                for (add, id) in changes {
                    if add {
                        active.insert(id);
                    } else {
                        active.remove(&id);
                    }
                }
                start = end;
            }
            result.push(segments);
        }
        Ok(result)
    }
}
