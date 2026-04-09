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
        bit: i64,
        _span: crate::netlist::parse::Span,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), StructuralAssignValidationError> {
        let (width, lsb) = net_width_bits(&nets[idx.0]);
        let offset_i64 = bit - lsb;
        if offset_i64 < 0 || offset_i64 >= width as i64 {
            return Err(StructuralAssignValidationError(format!(
                "bit {} out of range for net '{}' (width {}, lsb {})",
                bit,
                net_name(idx, nets, interner),
                width,
                lsb
            )));
        }
        let offset = offset_i64 as usize;
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
        let (width, lsb) = net_width_bits(&nets[idx.0]);
        for offset in 0..width {
            self.mark_bit(idx, lsb + offset as i64, span, nets, interner)?;
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
            NetRef::BitSelect(idx, bit) => {
                self.mark_bit(*idx, i64::from(*bit), span, nets, interner)
            }
            NetRef::PartSelect(idx, msb, lsb) => {
                if msb < lsb {
                    return Err(StructuralAssignValidationError(format!(
                        "invalid part-select [{}:{}] on net '{}'",
                        msb,
                        lsb,
                        net_name(*idx, nets, interner)
                    )));
                }
                for bit in *lsb..=*msb {
                    self.mark_bit(*idx, i64::from(bit), span, nets, interner)?;
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
            NetRef::BitSelect(idx, bit) => {
                let (width, lsb) = net_width_bits(&nets[idx.0]);
                let offset_i64 = i64::from(*bit) - lsb;
                if offset_i64 < 0 || offset_i64 >= width as i64 {
                    return Err(StructuralAssignValidationError(format!(
                        "bit {} out of range for net '{}'",
                        bit,
                        net_name(*idx, nets, interner)
                    )));
                }
                Ok(self
                    .drivers
                    .get(idx)
                    .and_then(|bits| bits.get(offset_i64 as usize))
                    .is_some_and(|bit| *bit))
            }
            NetRef::PartSelect(idx, msb, lsb) => {
                if msb < lsb {
                    return Ok(false);
                }
                let (width, net_lsb) = net_width_bits(&nets[idx.0]);
                let Some(bits) = self.drivers.get(idx) else {
                    return Ok(false);
                };
                for bit in *lsb..=*msb {
                    let offset_i64 = i64::from(bit) - net_lsb;
                    if offset_i64 < 0 || offset_i64 >= width as i64 {
                        return Ok(false);
                    }
                    if bits.get(offset_i64 as usize).is_none_or(|entry| !*entry) {
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

fn net_width_bits(net: &Net) -> (usize, i64) {
    if let Some((msb, lsb)) = net.width {
        (((msb as i64 - lsb as i64).abs() as usize) + 1, lsb as i64)
    } else {
        (1, 0)
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

fn validate_assign_expr_refs(
    expr: &AssignExpr,
    tracker: &BitDriverTracker,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<bool, StructuralAssignValidationError> {
    match expr {
        AssignExpr::Leaf(net_ref) => match net_ref {
            NetRef::Simple(idx) | NetRef::BitSelect(idx, _) | NetRef::PartSelect(idx, _, _) => {
                let _ = idx;
                tracker.is_netref_driven(net_ref, nets, interner)
            }
            NetRef::Literal(_) => Ok(true),
            NetRef::Unconnected => Err(StructuralAssignValidationError(
                "unconnected net reference is not supported in assign expressions".to_string(),
            )),
            NetRef::Concat(_) => Err(StructuralAssignValidationError(
                "concatenation is not supported in assign expressions".to_string(),
            )),
        },
        AssignExpr::Not(inner) => validate_assign_expr_refs(inner, tracker, nets, interner),
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            Ok(validate_assign_expr_refs(lhs, tracker, nets, interner)?
                && validate_assign_expr_refs(rhs, tracker, nets, interner)?)
        }
    }
}

fn collect_unresolved_assign_refs(
    expr: &AssignExpr,
    tracker: &BitDriverTracker,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    out: &mut Vec<NetRef>,
) -> Result<(), StructuralAssignValidationError> {
    match expr {
        AssignExpr::Leaf(net_ref) => match net_ref {
            NetRef::Simple(_) | NetRef::BitSelect(_, _) | NetRef::PartSelect(_, _, _) => {
                if !tracker.is_netref_driven(net_ref, nets, interner)? {
                    out.push(net_ref.clone());
                }
                Ok(())
            }
            NetRef::Literal(_) => Ok(()),
            NetRef::Unconnected => Err(StructuralAssignValidationError(
                "unconnected net reference is not supported in assign expressions".to_string(),
            )),
            NetRef::Concat(_) => Err(StructuralAssignValidationError(
                "concatenation is not supported in assign expressions".to_string(),
            )),
        },
        AssignExpr::Not(inner) => {
            collect_unresolved_assign_refs(inner, tracker, nets, interner, out)
        }
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            collect_unresolved_assign_refs(lhs, tracker, nets, interner, out)?;
            collect_unresolved_assign_refs(rhs, tracker, nets, interner, out)
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
            let (width, _) = net_width_bits(&nets[net_idx.0]);
            declared_tracker.seed_external_driver(net_idx, width);
            resolved_tracker.seed_external_driver(net_idx, width);
        }
    }

    for assign in &module.assigns {
        declared_tracker.mark_lhs(&assign.lhs, assign.span, nets, interner)?;
    }

    let mut pending: Vec<&crate::netlist::parse::NetlistAssign> = module.assigns.iter().collect();
    while !pending.is_empty() {
        let mut next_pending = Vec::new();
        let mut progressed = false;
        for assign in pending {
            if validate_assign_expr_refs(&assign.rhs, &resolved_tracker, nets, interner)? {
                resolved_tracker.mark_lhs(&assign.lhs, assign.span, nets, interner)?;
                progressed = true;
            } else {
                next_pending.push(assign);
            }
        }
        if !progressed {
            let mut undeclared_refs = BTreeSet::new();
            let mut cyclic_assigns = Vec::new();
            for assign in &next_pending {
                let mut missing = Vec::new();
                collect_unresolved_assign_refs(
                    &assign.rhs,
                    &resolved_tracker,
                    nets,
                    interner,
                    &mut missing,
                )?;
                let mut rendered_missing = BTreeSet::new();
                let mut assign_has_undeclared = false;
                for missing_ref in missing {
                    let rendered = render_net_ref(&missing_ref, nets, interner);
                    rendered_missing.insert(rendered.clone());
                    if !declared_tracker.is_netref_driven(&missing_ref, nets, interner)? {
                        undeclared_refs.insert(rendered);
                        assign_has_undeclared = true;
                    }
                }
                if !assign_has_undeclared && !rendered_missing.is_empty() {
                    cyclic_assigns.push(format!(
                        "{} <- [{}]",
                        render_net_ref(&assign.lhs, nets, interner),
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
