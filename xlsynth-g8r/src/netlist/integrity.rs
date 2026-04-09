// SPDX-License-Identifier: Apache-2.0

//! Basic integrity checks for parsed netlists.
//!
//! These checks look for simple wiring issues such as inputs that are never
//! used, outputs that are never driven, and wires that are declared but never
//! connected.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

use crate::liberty_proto::{Library, PinDirection};
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortDirection,
};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// A specific integrity problem found during checking.
#[derive(Debug, PartialEq, Eq)]
pub enum IntegrityFinding {
    /// Input port was declared but never used by any instance.
    UnusedInput(String),
    /// Output port was declared but never driven by any instance.
    UndrivenOutput(String),
    /// A wire was declared but never driven by any instance.
    UndrivenWire(String),
    /// A wire was declared but never used by any instance.
    UnusedWire(String),
}

/// Result of running the integrity checker over a module.
#[derive(Debug, PartialEq, Eq)]
pub enum IntegritySummary {
    /// No issues were found.
    Clean,
    /// One or more problems were detected.
    Findings(Vec<IntegrityFinding>),
}

/// Hard validation failure for Liberty-free structural-assign netlists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralAssignValidationError(pub String);

impl fmt::Display for StructuralAssignValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StructuralAssignValidationError {}

struct BitDriverTracker {
    drivers: HashMap<NetIndex, Vec<bool>>,
}

impl BitDriverTracker {
    fn new() -> Self {
        Self {
            drivers: HashMap::new(),
        }
    }

    fn seed_external_driver(&mut self, idx: NetIndex, width: usize) {
        self.drivers.insert(idx, vec![true; width]);
    }

    fn ensure_entry(&mut self, idx: NetIndex, width: usize) -> &mut Vec<bool> {
        self.drivers
            .entry(idx)
            .or_insert_with(|| vec![false; width])
    }

    fn mark_bit(
        &mut self,
        idx: NetIndex,
        bit: u32,
        _span: crate::netlist::parse::Span,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), StructuralAssignValidationError> {
        let net = &nets[idx.0];
        let width = net.width_bits();
        let Some(offset) = net.bit_offset(bit) else {
            return Err(StructuralAssignValidationError(format!(
                "bit {} out of range for net '{}' (width {}, lsb {})",
                bit,
                net_name(idx, nets, interner),
                width,
                net.declared_lsb_number()
            )));
        };
        let entry = self.ensure_entry(idx, width);
        if entry[offset] {
            return Err(StructuralAssignValidationError(format!(
                "multiple drivers for net '{}' at bit {}",
                net_name(idx, nets, interner),
                bit
            )));
        }
        entry[offset] = true;
        Ok(())
    }

    fn mark_whole(
        &mut self,
        idx: NetIndex,
        span: crate::netlist::parse::Span,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), StructuralAssignValidationError> {
        let net = &nets[idx.0];
        let width = net.width_bits();
        for offset in 0..width {
            let bit_number = net.bit_number(offset).ok_or_else(|| {
                StructuralAssignValidationError(format!(
                    "internal error computing bit {} for net '{}'",
                    offset,
                    net_name(idx, nets, interner)
                ))
            })?;
            self.mark_bit(idx, bit_number, span, nets, interner)?;
        }
        Ok(())
    }

    fn mark_lhs(
        &mut self,
        lhs: &NetRef,
        span: crate::netlist::parse::Span,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), StructuralAssignValidationError> {
        match lhs {
            NetRef::Simple(idx) => self.mark_whole(*idx, span, nets, interner),
            NetRef::BitSelect(idx, bit) => self.mark_bit(*idx, *bit, span, nets, interner),
            NetRef::PartSelect(idx, msb, lsb) => {
                for offset in 0..select_width_bits(*msb, *lsb) {
                    let bit_number = select_bit_number(*msb, *lsb, offset).ok_or_else(|| {
                        StructuralAssignValidationError(format!(
                            "invalid part-select [{}:{}] on net '{}'",
                            msb,
                            lsb,
                            net_name(*idx, nets, interner)
                        ))
                    })?;
                    self.mark_bit(*idx, bit_number, span, nets, interner)?;
                }
                Ok(())
            }
            NetRef::Literal(_) | NetRef::Unconnected | NetRef::Concat(_) => {
                Err(StructuralAssignValidationError(
                    "left-hand side of structural assign must be a net or net select".to_string(),
                ))
            }
        }
    }

    fn is_bit_driven(
        &self,
        idx: NetIndex,
        bit: u32,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<bool, StructuralAssignValidationError> {
        let net = &nets[idx.0];
        let Some(offset) = net.bit_offset(bit) else {
            return Err(StructuralAssignValidationError(format!(
                "bit {} out of range for net '{}'",
                bit,
                net_name(idx, nets, interner)
            )));
        };
        Ok(self
            .drivers
            .get(&idx)
            .and_then(|bits| bits.get(offset))
            .is_some_and(|bit| *bit))
    }

    fn is_netref_driven(
        &self,
        net_ref: &NetRef,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<bool, StructuralAssignValidationError> {
        match net_ref {
            NetRef::Simple(idx) => Ok(self
                .drivers
                .get(idx)
                .is_some_and(|bits| bits.iter().all(|bit| *bit))),
            NetRef::BitSelect(idx, bit) => self.is_bit_driven(*idx, *bit, nets, interner),
            NetRef::PartSelect(idx, msb, lsb) => {
                for offset in 0..select_width_bits(*msb, *lsb) {
                    let bit_number = select_bit_number(*msb, *lsb, offset).ok_or_else(|| {
                        StructuralAssignValidationError(format!(
                            "invalid part-select [{}:{}] on net '{}'",
                            msb,
                            lsb,
                            net_name(*idx, nets, interner)
                        ))
                    })?;
                    if !self.is_bit_driven(*idx, bit_number, nets, interner)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            NetRef::Literal(_) => Ok(true),
            NetRef::Unconnected => Ok(false),
            NetRef::Concat(_) => Err(StructuralAssignValidationError(
                "concatenation is not supported in structural assign expressions".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NetBitTarget {
    idx: NetIndex,
    bit_number: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReferencedNetBit {
    idx: NetIndex,
    bit_number: u32,
    rendered: String,
}

#[derive(Debug, Clone, Copy)]
struct PendingAssignBit {
    assign_index: usize,
    rhs_bit_index: usize,
    target: NetBitTarget,
}

fn select_width_bits(msb: u32, lsb: u32) -> usize {
    (u32::abs_diff(msb, lsb) as usize) + 1
}

fn select_bit_number(msb: u32, lsb: u32, bit_offset: usize) -> Option<u32> {
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

fn net_name(
    idx: NetIndex,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> String {
    interner
        .resolve(nets[idx.0].name)
        .unwrap_or("<unknown>")
        .to_string()
}

fn render_named_bit(
    idx: NetIndex,
    bit_number: u32,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> String {
    if nets[idx.0].width_bits() == 1 {
        net_name(idx, nets, interner)
    } else {
        format!("{}[{}]", net_name(idx, nets, interner), bit_number)
    }
}

fn net_ref_width_bits(
    net_ref: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<usize, StructuralAssignValidationError> {
    match net_ref {
        NetRef::Simple(idx) => Ok(nets[idx.0].width_bits()),
        NetRef::BitSelect(idx, bit) => {
            if nets[idx.0].bit_offset(*bit).is_none() {
                return Err(StructuralAssignValidationError(format!(
                    "bit {} out of range for net '{}'",
                    bit,
                    net_name(*idx, nets, interner)
                )));
            }
            Ok(1)
        }
        NetRef::PartSelect(idx, msb, lsb) => {
            let width = select_width_bits(*msb, *lsb);
            for offset in 0..width {
                let bit_number = select_bit_number(*msb, *lsb, offset).ok_or_else(|| {
                    StructuralAssignValidationError(format!(
                        "invalid part-select [{}:{}] on net '{}'",
                        msb,
                        lsb,
                        net_name(*idx, nets, interner)
                    ))
                })?;
                if nets[idx.0].bit_offset(bit_number).is_none() {
                    return Err(StructuralAssignValidationError(format!(
                        "bit {} out of range for net '{}'",
                        bit_number,
                        net_name(*idx, nets, interner)
                    )));
                }
            }
            Ok(width)
        }
        NetRef::Literal(bits) => Ok(bits.get_bit_count()),
        NetRef::Unconnected => Err(StructuralAssignValidationError(
            "unconnected net reference is not supported in assign expressions".to_string(),
        )),
        NetRef::Concat(_) => Err(StructuralAssignValidationError(
            "concatenation is not supported in structural assign expressions".to_string(),
        )),
    }
}

fn assign_expr_width_bits(
    expr: &AssignExpr,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<usize, StructuralAssignValidationError> {
    match expr {
        AssignExpr::Leaf(net_ref) => net_ref_width_bits(net_ref, nets, interner),
        AssignExpr::Not(inner) => assign_expr_width_bits(inner, nets, interner),
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            let lhs_width = assign_expr_width_bits(lhs, nets, interner)?;
            let rhs_width = assign_expr_width_bits(rhs, nets, interner)?;
            if lhs_width != rhs_width {
                let op = match expr {
                    AssignExpr::And(_, _) => '&',
                    AssignExpr::Or(_, _) => '|',
                    AssignExpr::Xor(_, _) => '^',
                    _ => unreachable!("binary expression matched as non-binary"),
                };
                return Err(StructuralAssignValidationError(format!(
                    "bitwise '{}' width mismatch: lhs {} bits rhs {} bits",
                    op, lhs_width, rhs_width
                )));
            }
            Ok(lhs_width)
        }
    }
}

fn net_ref_bit_is_driven(
    net_ref: &NetRef,
    rhs_bit_index: usize,
    tracker: &BitDriverTracker,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<bool, StructuralAssignValidationError> {
    match net_ref {
        NetRef::Simple(idx) => {
            let bit_number = nets[idx.0].bit_number(rhs_bit_index).ok_or_else(|| {
                StructuralAssignValidationError(format!(
                    "bit {} out of range for net '{}'",
                    rhs_bit_index,
                    net_name(*idx, nets, interner)
                ))
            })?;
            tracker.is_bit_driven(*idx, bit_number, nets, interner)
        }
        NetRef::BitSelect(idx, bit) => {
            if rhs_bit_index != 0 {
                return Err(StructuralAssignValidationError(format!(
                    "bit-select '{}' cannot be indexed at expression bit {}",
                    render_named_bit(*idx, *bit, nets, interner),
                    rhs_bit_index
                )));
            }
            tracker.is_bit_driven(*idx, *bit, nets, interner)
        }
        NetRef::PartSelect(idx, msb, lsb) => {
            let bit_number = select_bit_number(*msb, *lsb, rhs_bit_index).ok_or_else(|| {
                StructuralAssignValidationError(format!(
                    "bit {} out of range for part-select '{}'",
                    rhs_bit_index,
                    render_net_ref(net_ref, nets, interner)
                ))
            })?;
            tracker.is_bit_driven(*idx, bit_number, nets, interner)
        }
        NetRef::Literal(bits) => {
            if rhs_bit_index >= bits.get_bit_count() {
                return Err(StructuralAssignValidationError(format!(
                    "literal bit {} out of range for {}-bit literal",
                    rhs_bit_index,
                    bits.get_bit_count()
                )));
            }
            Ok(true)
        }
        NetRef::Unconnected => Err(StructuralAssignValidationError(
            "unconnected net reference is not supported in assign expressions".to_string(),
        )),
        NetRef::Concat(_) => Err(StructuralAssignValidationError(
            "concatenation is not supported in structural assign expressions".to_string(),
        )),
    }
}

fn validate_assign_expr_bit_refs(
    expr: &AssignExpr,
    rhs_bit_index: usize,
    tracker: &BitDriverTracker,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<bool, StructuralAssignValidationError> {
    match expr {
        AssignExpr::Leaf(net_ref) => {
            net_ref_bit_is_driven(net_ref, rhs_bit_index, tracker, nets, interner)
        }
        AssignExpr::Not(inner) => {
            validate_assign_expr_bit_refs(inner, rhs_bit_index, tracker, nets, interner)
        }
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => Ok(
            validate_assign_expr_bit_refs(lhs, rhs_bit_index, tracker, nets, interner)?
                && validate_assign_expr_bit_refs(rhs, rhs_bit_index, tracker, nets, interner)?,
        ),
    }
}

fn net_ref_unresolved_bits(
    net_ref: &NetRef,
    rhs_bit_index: usize,
    tracker: &BitDriverTracker,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    out: &mut Vec<ReferencedNetBit>,
) -> Result<(), StructuralAssignValidationError> {
    match net_ref {
        NetRef::Simple(idx) => {
            let bit_number = nets[idx.0].bit_number(rhs_bit_index).ok_or_else(|| {
                StructuralAssignValidationError(format!(
                    "bit {} out of range for net '{}'",
                    rhs_bit_index,
                    net_name(*idx, nets, interner)
                ))
            })?;
            if !tracker.is_bit_driven(*idx, bit_number, nets, interner)? {
                out.push(ReferencedNetBit {
                    idx: *idx,
                    bit_number,
                    rendered: render_named_bit(*idx, bit_number, nets, interner),
                });
            }
            Ok(())
        }
        NetRef::BitSelect(idx, bit) => {
            if rhs_bit_index != 0 {
                return Err(StructuralAssignValidationError(format!(
                    "bit-select '{}' cannot be indexed at expression bit {}",
                    render_named_bit(*idx, *bit, nets, interner),
                    rhs_bit_index
                )));
            }
            if !tracker.is_bit_driven(*idx, *bit, nets, interner)? {
                out.push(ReferencedNetBit {
                    idx: *idx,
                    bit_number: *bit,
                    rendered: render_named_bit(*idx, *bit, nets, interner),
                });
            }
            Ok(())
        }
        NetRef::PartSelect(idx, msb, lsb) => {
            let bit_number = select_bit_number(*msb, *lsb, rhs_bit_index).ok_or_else(|| {
                StructuralAssignValidationError(format!(
                    "bit {} out of range for part-select '{}'",
                    rhs_bit_index,
                    render_net_ref(net_ref, nets, interner)
                ))
            })?;
            if !tracker.is_bit_driven(*idx, bit_number, nets, interner)? {
                out.push(ReferencedNetBit {
                    idx: *idx,
                    bit_number,
                    rendered: render_named_bit(*idx, bit_number, nets, interner),
                });
            }
            Ok(())
        }
        NetRef::Literal(bits) => {
            if rhs_bit_index >= bits.get_bit_count() {
                return Err(StructuralAssignValidationError(format!(
                    "literal bit {} out of range for {}-bit literal",
                    rhs_bit_index,
                    bits.get_bit_count()
                )));
            }
            Ok(())
        }
        NetRef::Unconnected => Err(StructuralAssignValidationError(
            "unconnected net reference is not supported in assign expressions".to_string(),
        )),
        NetRef::Concat(_) => Err(StructuralAssignValidationError(
            "concatenation is not supported in structural assign expressions".to_string(),
        )),
    }
}

fn collect_unresolved_assign_refs(
    expr: &AssignExpr,
    rhs_bit_index: usize,
    tracker: &BitDriverTracker,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    out: &mut Vec<ReferencedNetBit>,
) -> Result<(), StructuralAssignValidationError> {
    match expr {
        AssignExpr::Leaf(net_ref) => {
            net_ref_unresolved_bits(net_ref, rhs_bit_index, tracker, nets, interner, out)
        }
        AssignExpr::Not(inner) => {
            collect_unresolved_assign_refs(inner, rhs_bit_index, tracker, nets, interner, out)
        }
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            collect_unresolved_assign_refs(lhs, rhs_bit_index, tracker, nets, interner, out)?;
            collect_unresolved_assign_refs(rhs, rhs_bit_index, tracker, nets, interner, out)
        }
    }
}

fn expand_lhs_bits(
    lhs: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<Vec<NetBitTarget>, StructuralAssignValidationError> {
    match lhs {
        NetRef::Simple(idx) => {
            let net = &nets[idx.0];
            let mut bits = Vec::with_capacity(net.width_bits());
            for offset in 0..net.width_bits() {
                let bit_number = net.bit_number(offset).ok_or_else(|| {
                    StructuralAssignValidationError(format!(
                        "internal error computing bit {} for net '{}'",
                        offset,
                        net_name(*idx, nets, interner)
                    ))
                })?;
                bits.push(NetBitTarget {
                    idx: *idx,
                    bit_number,
                });
            }
            Ok(bits)
        }
        NetRef::BitSelect(idx, bit) => {
            if nets[idx.0].bit_offset(*bit).is_none() {
                return Err(StructuralAssignValidationError(format!(
                    "bit {} out of range for net '{}'",
                    bit,
                    net_name(*idx, nets, interner)
                )));
            }
            Ok(vec![NetBitTarget {
                idx: *idx,
                bit_number: *bit,
            }])
        }
        NetRef::PartSelect(idx, msb, lsb) => {
            let mut bits = Vec::with_capacity(select_width_bits(*msb, *lsb));
            for offset in 0..select_width_bits(*msb, *lsb) {
                let bit_number = select_bit_number(*msb, *lsb, offset).ok_or_else(|| {
                    StructuralAssignValidationError(format!(
                        "invalid part-select [{}:{}] on net '{}'",
                        msb,
                        lsb,
                        net_name(*idx, nets, interner)
                    ))
                })?;
                if nets[idx.0].bit_offset(bit_number).is_none() {
                    return Err(StructuralAssignValidationError(format!(
                        "bit {} out of range for net '{}'",
                        bit_number,
                        net_name(*idx, nets, interner)
                    )));
                }
                bits.push(NetBitTarget {
                    idx: *idx,
                    bit_number,
                });
            }
            Ok(bits)
        }
        NetRef::Literal(_) | NetRef::Unconnected | NetRef::Concat(_) => {
            Err(StructuralAssignValidationError(
                "left-hand side of structural assign must be a net or net select".to_string(),
            ))
        }
    }
}

fn render_net_ref(
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
        NetRef::Unconnected => "<unconnected>".to_string(),
        NetRef::Concat(_) => "<concat>".to_string(),
    }
}

/// Validates the Liberty-free structural-assign netlist subset used by
/// `gv2aig` when no Liberty library is provided.
pub fn validate_structural_assign_module(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<(), StructuralAssignValidationError> {
    if !module.instances.is_empty() {
        return Err(StructuralAssignValidationError(
            "Liberty-free structural mode does not support cell instances".to_string(),
        ));
    }

    if module
        .ports
        .iter()
        .any(|port| port.direction == PortDirection::Inout)
    {
        return Err(StructuralAssignValidationError(
            "Liberty-free structural mode does not support inout ports".to_string(),
        ));
    }

    let mut declared_tracker = BitDriverTracker::new();
    let mut resolved_tracker = BitDriverTracker::new();
    for port in &module.ports {
        if port.direction == PortDirection::Input {
            let Some(net_idx) = module.find_net_index(port.name, nets) else {
                return Err(StructuralAssignValidationError(format!(
                    "input port '{}' net not found",
                    interner.resolve(port.name).unwrap_or("<unknown>")
                )));
            };
            let width = nets[net_idx.0].width_bits();
            declared_tracker.seed_external_driver(net_idx, width);
            resolved_tracker.seed_external_driver(net_idx, width);
        }
    }

    let mut pending = Vec::new();
    for (assign_index, assign) in module.assigns.iter().enumerate() {
        let lhs_bits = expand_lhs_bits(&assign.lhs, nets, interner)?;
        let rhs_width = assign_expr_width_bits(&assign.rhs, nets, interner)?;
        if lhs_bits.len() != rhs_width {
            return Err(StructuralAssignValidationError(format!(
                "assign to '{}' has lhs width {} but rhs width {}",
                render_net_ref(&assign.lhs, nets, interner),
                lhs_bits.len(),
                rhs_width
            )));
        }
        declared_tracker.mark_lhs(&assign.lhs, assign.span, nets, interner)?;
        for (rhs_bit_index, target) in lhs_bits.into_iter().enumerate() {
            pending.push(PendingAssignBit {
                assign_index,
                rhs_bit_index,
                target,
            });
        }
    }

    while !pending.is_empty() {
        let mut next_pending = Vec::new();
        let mut progressed = false;
        for assign in pending {
            let netlist_assign = &module.assigns[assign.assign_index];
            if validate_assign_expr_bit_refs(
                &netlist_assign.rhs,
                assign.rhs_bit_index,
                &resolved_tracker,
                nets,
                interner,
            )? {
                resolved_tracker.mark_bit(
                    assign.target.idx,
                    assign.target.bit_number,
                    netlist_assign.span,
                    nets,
                    interner,
                )?;
                progressed = true;
            } else {
                next_pending.push(assign);
            }
        }
        if !progressed {
            let mut undeclared_refs = BTreeSet::new();
            let mut cyclic_assigns = Vec::new();
            for assign in &next_pending {
                let netlist_assign = &module.assigns[assign.assign_index];
                let mut missing: Vec<ReferencedNetBit> = Vec::new();
                collect_unresolved_assign_refs(
                    &netlist_assign.rhs,
                    assign.rhs_bit_index,
                    &resolved_tracker,
                    nets,
                    interner,
                    &mut missing,
                )?;
                let mut rendered_missing = BTreeSet::new();
                let mut assign_has_undeclared = false;
                for missing_ref in missing {
                    let rendered = missing_ref.rendered.clone();
                    rendered_missing.insert(rendered.clone());
                    if !declared_tracker.is_bit_driven(
                        missing_ref.idx,
                        missing_ref.bit_number,
                        nets,
                        interner,
                    )? {
                        undeclared_refs.insert(rendered);
                        assign_has_undeclared = true;
                    }
                }
                if !assign_has_undeclared && !rendered_missing.is_empty() {
                    cyclic_assigns.push(format!(
                        "{} <- [{}]",
                        render_named_bit(
                            assign.target.idx,
                            assign.target.bit_number,
                            nets,
                            interner,
                        ),
                        rendered_missing.into_iter().collect::<Vec<_>>().join(", ")
                    ));
                }
            }
            if !undeclared_refs.is_empty() {
                return Err(StructuralAssignValidationError(format!(
                    "assign reads undriven net reference(s): [{}]",
                    undeclared_refs.into_iter().collect::<Vec<_>>().join(", ")
                )));
            }
            return Err(StructuralAssignValidationError(format!(
                "structural assigns contain a dependency cycle or unresolved recursion: {}",
                cyclic_assigns.join("; ")
            )));
        }
        pending = next_pending;
    }

    for port in &module.ports {
        if port.direction != PortDirection::Output {
            continue;
        }
        let Some(net_idx) = module.find_net_index(port.name, nets) else {
            return Err(StructuralAssignValidationError(format!(
                "output port '{}' net not found",
                interner.resolve(port.name).unwrap_or("<unknown>")
            )));
        };
        if !resolved_tracker.is_netref_driven(&NetRef::Simple(net_idx), nets, interner)? {
            return Err(StructuralAssignValidationError(format!(
                "output '{}' is not fully driven",
                interner.resolve(port.name).unwrap_or("<unknown>")
            )));
        }
    }

    Ok(())
}

/// Check a parsed module for simple wiring issues.
///
/// `module`    - The module to check.
/// `nets`      - The global list of nets referenced by `module`.
/// `interner`  - The interner used when parsing the netlist.
/// `lib`       - Liberty library providing pin directions for instances.
pub fn check_module(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    lib: &Library,
) -> IntegritySummary {
    // Build mapping of (cell name -> {pin name -> direction})
    let mut dir_map: HashMap<&str, HashMap<&str, i32>> = HashMap::new();
    for cell in &lib.cells {
        let mut pins = HashMap::new();
        for pin in &cell.pins {
            pins.insert(pin.name.as_str(), pin.direction);
        }
        dir_map.insert(cell.name.as_str(), pins);
    }

    let mut used_as_input: HashSet<SymbolU32> = HashSet::new();
    let mut driven: HashSet<SymbolU32> = HashSet::new();

    // Module ports contribute to driving/using sets.
    for NetlistPort {
        direction, name, ..
    } in &module.ports
    {
        match direction {
            PortDirection::Input => {
                driven.insert(*name);
            }
            PortDirection::Output => {
                used_as_input.insert(*name); // environment observes the output
            }
            PortDirection::Inout => {
                driven.insert(*name);
                used_as_input.insert(*name);
            }
        }
    }

    // Walk instances and classify connections according to pin directions.
    for NetlistInstance {
        type_name,
        connections,
        ..
    } in &module.instances
    {
        let Some(type_str) = interner.resolve(*type_name) else {
            continue;
        };
        let pin_dirs = dir_map.get(type_str);
        for (port, netref) in connections {
            let Some(port_str) = interner.resolve(*port) else {
                continue;
            };
            let dir = pin_dirs
                .and_then(|m| m.get(port_str))
                .copied()
                .unwrap_or(PinDirection::Invalid as i32);
            // For concatenations, conservatively mark each element; otherwise classify
            // single net.
            let mut mark = |sym: SymbolU32| {
                if dir == PinDirection::Output as i32 {
                    driven.insert(sym);
                } else if dir == PinDirection::Input as i32 {
                    used_as_input.insert(sym);
                } else {
                    driven.insert(sym);
                    used_as_input.insert(sym);
                }
            };
            match netref {
                NetRef::Simple(idx) | NetRef::BitSelect(idx, _) | NetRef::PartSelect(idx, _, _) => {
                    mark(nets[idx.0].name);
                }
                NetRef::Concat(elems) => {
                    for e in elems {
                        match e {
                            NetRef::Simple(idx)
                            | NetRef::BitSelect(idx, _)
                            | NetRef::PartSelect(idx, _, _) => mark(nets[idx.0].name),
                            NetRef::Literal(_) | NetRef::Unconnected | NetRef::Concat(_) => {}
                        }
                    }
                }
                NetRef::Literal(_) => {}
                NetRef::Unconnected => {}
            }
        }
    }

    let mut findings = Vec::new();

    // Inputs should be used somewhere.
    for port in &module.ports {
        if port.direction == PortDirection::Input && !used_as_input.contains(&port.name) {
            let name = interner
                .resolve(port.name)
                .unwrap_or("<unknown>")
                .to_string();
            findings.push(IntegrityFinding::UnusedInput(name));
        }
    }

    // Outputs must be driven.
    for port in &module.ports {
        if port.direction == PortDirection::Output && !driven.contains(&port.name) {
            let name = interner
                .resolve(port.name)
                .unwrap_or("<unknown>")
                .to_string();
            findings.push(IntegrityFinding::UndrivenOutput(name));
        }
    }

    // Every declared wire should be driven and used.
    for net_idx in &module.wires {
        let sym = nets[net_idx.0].name;
        if !driven.contains(&sym) {
            let name = interner.resolve(sym).unwrap_or("<unknown>").to_string();
            findings.push(IntegrityFinding::UndrivenWire(name.clone()));
        }
        if !used_as_input.contains(&sym) {
            let name = interner.resolve(sym).unwrap_or("<unknown>").to_string();
            findings.push(IntegrityFinding::UnusedWire(name));
        }
    }

    if findings.is_empty() {
        IntegritySummary::Clean
    } else {
        IntegritySummary::Findings(findings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::parse::{Parser, TokenScanner};

    fn parse_single_module(
        src: &str,
    ) -> (
        NetlistModule,
        Vec<Net>,
        StringInterner<StringBackend<SymbolU32>>,
    ) {
        let lines: Vec<String> = src.lines().map(|line| line.to_string()).collect();
        let lookup = move |lineno: u32| lines.get((lineno - 1) as usize).cloned();
        let scanner = TokenScanner::with_line_lookup(
            std::io::Cursor::new(src.as_bytes().to_vec()),
            Box::new(lookup),
        );
        let mut parser = Parser::new(scanner);
        let modules = parser.parse_file().expect("parse ok");
        assert_eq!(modules.len(), 1);
        (
            modules.into_iter().next().unwrap(),
            parser.nets,
            parser.interner,
        )
    }

    fn validate(src: &str) -> Result<(), StructuralAssignValidationError> {
        let (module, nets, interner) = parse_single_module(src);
        validate_structural_assign_module(&module, &nets, &interner)
    }

    fn expect_validation_error(src: &str, needle: &str) {
        let err = validate(src).expect_err("expected validation to fail");
        assert!(
            err.0.contains(needle),
            "expected error containing '{}', got '{}'",
            needle,
            err.0
        );
    }

    #[test]
    fn validate_structural_assign_module_accepts_single_driver_chain() {
        validate(
            r#"
module top(a, b, y);
  input a;
  input b;
  output y;
  wire n;
  assign n = a & b;
  assign y = ~n;
endmodule
"#,
        )
        .expect("single-driver chain should validate");
    }

    #[test]
    fn validate_structural_assign_module_rejects_multiple_scalar_drivers() {
        expect_validation_error(
            r#"
module top(a, b, y);
  input a;
  input b;
  output y;
  assign y = a;
  assign y = b;
endmodule
"#,
            "multiple drivers for net 'y' at bit 0",
        );
    }

    #[test]
    fn validate_structural_assign_module_rejects_overlapping_part_select_drivers() {
        expect_validation_error(
            r#"
module top(a, b, y);
  input [1:0] a;
  input [1:0] b;
  output [3:0] y;
  assign y[2:1] = a;
  assign y[3:2] = b;
endmodule
"#,
            "multiple drivers for net 'y' at bit 2",
        );
    }

    #[test]
    fn validate_structural_assign_module_rejects_whole_net_and_bit_driver_mix() {
        expect_validation_error(
            r#"
module top(a, b, y);
  input [3:0] a;
  input b;
  output [3:0] y;
  assign y = a;
  assign y[0] = b;
endmodule
"#,
            "multiple drivers for net 'y' at bit 0",
        );
    }

    #[test]
    fn validate_structural_assign_module_rejects_undriven_rhs_net() {
        expect_validation_error(
            r#"
module top(a, y);
  input a;
  output y;
  wire n;
  assign y = n;
endmodule
"#,
            "assign reads undriven net reference(s): [n]",
        );
    }

    #[test]
    fn validate_structural_assign_module_rejects_undriven_output() {
        expect_validation_error(
            r#"
module top(a, y);
  input a;
  output y;
endmodule
"#,
            "output 'y' is not fully driven",
        );
    }

    #[test]
    fn validate_structural_assign_module_rejects_out_of_range_selects() {
        expect_validation_error(
            r#"
module top(a, y);
  input a;
  output y;
  assign y = a[1];
endmodule
"#,
            "bit 1 out of range for net 'a'",
        );
    }

    #[test]
    fn validate_structural_assign_module_accepts_disjoint_partial_bus_assembly() {
        validate(
            r#"
module top(lo, hi, y);
  input [1:0] lo;
  input [1:0] hi;
  output [3:0] y;
  assign y[1:0] = lo;
  assign y[3:2] = hi;
endmodule
"#,
        )
        .expect("disjoint partial bus assembly should validate");
    }

    #[test]
    fn validate_structural_assign_module_accepts_acyclic_overlapping_slice_dependencies() {
        validate(
            r#"
module top(a, y);
  input a;
  output [3:0] y;
  assign y[3:1] = y[2:0];
  assign y[0] = a;
endmodule
"#,
        )
        .expect("bit-level overlapping slice dependencies should validate");
    }

    #[test]
    fn validate_structural_assign_module_accepts_ascending_packed_range_selects() {
        validate(
            r#"
module top(a, y);
  input [0:3] a;
  output [0:3] y;
  assign y[0:2] = a[0:2];
  assign y[3] = a[3];
endmodule
"#,
        )
        .expect("ascending packed range selects should validate");
    }

    #[test]
    fn validate_structural_assign_module_rejects_instances() {
        expect_validation_error(
            r#"
module top(a, y);
  input a;
  output y;
  INV u1 (.A(a), .Y(y));
endmodule
"#,
            "does not support cell instances",
        );
    }

    #[test]
    fn validate_structural_assign_module_rejects_cycles() {
        expect_validation_error(
            r#"
module top(a, y);
  input a;
  output y;
  wire n;
  assign n = y;
  assign y = n;
endmodule
"#,
            "dependency cycle",
        );
    }
}
