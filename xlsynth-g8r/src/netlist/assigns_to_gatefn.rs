// SPDX-License-Identifier: Apache-2.0

//! Project a Liberty-free structural assign netlist into a `GateFn`.
//!
//! See `src/netlist/STRUCTURAL_ASSIGNS.md` for the accepted Liberty-free
//! structural subset and its sizing rules.

use crate::aig::{AigBitVector, AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::netlist::integrity::validate_structural_assign_module;
use crate::netlist::parse::{AssignExpr, Net, NetIndex, NetRef, NetlistModule, PortDirection};
use std::collections::HashMap;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

#[derive(Debug, Clone, Copy)]
struct NetBitTarget {
    idx: NetIndex,
    bit_number: u32,
}

#[derive(Debug, Clone, Copy)]
struct PendingAssignBit {
    assign_index: usize,
    rhs_bit_index: usize,
    target: NetBitTarget,
}

struct ResolvedNetValues {
    values: HashMap<NetIndex, Vec<Option<AigOperand>>>,
}

impl ResolvedNetValues {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    fn seed_input(&mut self, idx: NetIndex, bv: &AigBitVector) {
        self.values.insert(
            idx,
            bv.iter_lsb_to_msb()
                .map(|bit| Some(*bit))
                .collect::<Vec<_>>(),
        );
    }

    fn resolve_bit(
        &self,
        idx: NetIndex,
        bit_number: u32,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<Option<AigOperand>, String> {
        let Some(offset) = nets[idx.0].bit_offset(bit_number) else {
            return Err(format!(
                "bit {} out of range for net '{}'",
                bit_number,
                net_name(idx, nets, interner)
            ));
        };
        Ok(self
            .values
            .get(&idx)
            .and_then(|bits| bits.get(offset))
            .copied()
            .flatten())
    }

    fn ensure_entry(&mut self, idx: NetIndex, width: usize) -> &mut Vec<Option<AigOperand>> {
        self.values.entry(idx).or_insert_with(|| vec![None; width])
    }

    fn write_bit(
        &mut self,
        idx: NetIndex,
        bit_number: u32,
        value: AigOperand,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), String> {
        let net = &nets[idx.0];
        let width = net.width_bits();
        let Some(offset) = net.bit_offset(bit_number) else {
            return Err(format!(
                "bit {} out of range for net '{}'",
                bit_number,
                net_name(idx, nets, interner)
            ));
        };
        let entry = self.ensure_entry(idx, width);
        if entry[offset].is_some() {
            return Err(format!(
                "net '{}' bit {} was assigned more than once during projection",
                net_name(idx, nets, interner),
                bit_number
            ));
        }
        entry[offset] = Some(value);
        Ok(())
    }

    fn materialize_ref_bit(
        &self,
        net_ref: &NetRef,
        rhs_bit_index: usize,
        allow_literal_zero_extend: bool,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        gb: &mut GateBuilder,
    ) -> Result<Option<AigOperand>, String> {
        match net_ref {
            NetRef::Simple(idx) => {
                let bit_number = nets[idx.0].bit_number(rhs_bit_index).ok_or_else(|| {
                    format!(
                        "bit {} out of range for net '{}'",
                        rhs_bit_index,
                        net_name(*idx, nets, interner)
                    )
                })?;
                self.resolve_bit(*idx, bit_number, nets, interner)
            }
            NetRef::BitSelect(idx, bit) => {
                if rhs_bit_index != 0 {
                    return Err(format!(
                        "bit-select '{}' cannot be indexed at expression bit {}",
                        render_named_bit(*idx, *bit, nets, interner),
                        rhs_bit_index
                    ));
                }
                self.resolve_bit(*idx, *bit, nets, interner)
            }
            NetRef::PartSelect(idx, msb, lsb_sel) => {
                let bit_number =
                    select_bit_number(*msb, *lsb_sel, rhs_bit_index).ok_or_else(|| {
                        format!(
                            "bit {} out of range for part-select '{}'",
                            rhs_bit_index,
                            render_net_ref(net_ref, nets, interner)
                        )
                    })?;
                self.resolve_bit(*idx, bit_number, nets, interner)
            }
            NetRef::Literal(bits) => {
                let bit_value = if rhs_bit_index >= bits.get_bit_count() {
                    if allow_literal_zero_extend {
                        false
                    } else {
                        return Err(format!(
                            "literal bit {} out of range for {}-bit literal",
                            rhs_bit_index,
                            bits.get_bit_count()
                        ));
                    }
                } else {
                    bits.get_bit(rhs_bit_index).map_err(|_| {
                        format!(
                            "literal bit {} out of range for {}-bit literal",
                            rhs_bit_index,
                            bits.get_bit_count()
                        )
                    })?
                };
                Ok(Some(if bit_value {
                    gb.get_true()
                } else {
                    gb.get_false()
                }))
            }
            NetRef::UnknownLiteral(_) => {
                Err("unknown literal net reference is not supported".to_string())
            }
            NetRef::Unconnected => Err("unconnected net reference is not supported".to_string()),
            NetRef::Concat(_) => {
                Err("concatenation is not supported in structural mode".to_string())
            }
        }
    }

    fn materialize_ref(
        &self,
        net_ref: &NetRef,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        gb: &mut GateBuilder,
    ) -> Result<Option<AigBitVector>, String> {
        let width = net_ref_width_bits(net_ref, nets, interner)?;
        let mut bits = Vec::with_capacity(width);
        for rhs_bit_index in 0..width {
            let Some(bit) =
                self.materialize_ref_bit(net_ref, rhs_bit_index, false, nets, interner, gb)?
            else {
                return Ok(None);
            };
            bits.push(bit);
        }
        Ok(Some(AigBitVector::from_lsb_is_index_0(&bits)))
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

fn net_ref_width_bits(
    net_ref: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<usize, String> {
    match net_ref {
        NetRef::Simple(idx) => Ok(nets[idx.0].width_bits()),
        NetRef::BitSelect(idx, bit) => {
            if nets[idx.0].bit_offset(*bit).is_none() {
                return Err(format!(
                    "bit {} out of range for net '{}'",
                    bit,
                    net_name(*idx, nets, interner)
                ));
            }
            Ok(1)
        }
        NetRef::PartSelect(idx, msb, lsb) => {
            let width = select_width_bits(*msb, *lsb);
            for offset in 0..width {
                let bit_number = select_bit_number(*msb, *lsb, offset).ok_or_else(|| {
                    format!(
                        "invalid part-select [{}:{}] on '{}'",
                        msb,
                        lsb,
                        net_name(*idx, nets, interner)
                    )
                })?;
                if nets[idx.0].bit_offset(bit_number).is_none() {
                    return Err(format!(
                        "bit {} out of range for net '{}'",
                        bit_number,
                        net_name(*idx, nets, interner)
                    ));
                }
            }
            Ok(width)
        }
        NetRef::Literal(bits) => Ok(bits.get_bit_count()),
        NetRef::UnknownLiteral(_) => {
            Err("unknown literal net reference is not supported in structural mode".to_string())
        }
        NetRef::Unconnected => Err("unconnected net reference is not supported".to_string()),
        NetRef::Concat(_) => Err("concatenation is not supported in structural mode".to_string()),
    }
}

fn assign_expr_width_bits(
    expr: &AssignExpr,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<usize, String> {
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
                return Err(format!(
                    "bitwise '{}' width mismatch: lhs {} bits rhs {} bits; Liberty-free structural mode requires exact-width bitwise operands",
                    op, lhs_width, rhs_width
                ));
            }
            Ok(lhs_width)
        }
    }
}

fn is_bare_literal_assign_expr(expr: &AssignExpr) -> bool {
    matches!(expr, AssignExpr::Leaf(NetRef::Literal(_)))
}

fn expand_lhs_bits(
    lhs: &NetRef,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<Vec<NetBitTarget>, String> {
    match lhs {
        NetRef::Simple(idx) => {
            let net = &nets[idx.0];
            let mut bits = Vec::with_capacity(net.width_bits());
            for offset in 0..net.width_bits() {
                let bit_number = net.bit_number(offset).ok_or_else(|| {
                    format!(
                        "internal error computing bit {} for net '{}'",
                        offset,
                        net_name(*idx, nets, interner)
                    )
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
                return Err(format!(
                    "bit {} out of range for net '{}'",
                    bit,
                    net_name(*idx, nets, interner)
                ));
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
                    format!(
                        "invalid part-select [{}:{}] on '{}'",
                        msb,
                        lsb,
                        net_name(*idx, nets, interner)
                    )
                })?;
                if nets[idx.0].bit_offset(bit_number).is_none() {
                    return Err(format!(
                        "bit {} out of range for net '{}'",
                        bit_number,
                        net_name(*idx, nets, interner)
                    ));
                }
                bits.push(NetBitTarget {
                    idx: *idx,
                    bit_number,
                });
            }
            Ok(bits)
        }
        NetRef::Literal(_)
        | NetRef::UnknownLiteral(_)
        | NetRef::Unconnected
        | NetRef::Concat(_) => {
            Err("left-hand side of structural assign must be a net or net select".to_string())
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
        NetRef::UnknownLiteral(width) => format!("{}'hx", width),
        NetRef::Unconnected => "<unconnected>".to_string(),
        NetRef::Concat(_) => "<concat>".to_string(),
    }
}

fn eval_assign_expr_bit(
    expr: &AssignExpr,
    rhs_bit_index: usize,
    allow_literal_zero_extend: bool,
    resolved: &ResolvedNetValues,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
) -> Result<Option<AigOperand>, String> {
    match expr {
        AssignExpr::Leaf(net_ref) => resolved.materialize_ref_bit(
            net_ref,
            rhs_bit_index,
            allow_literal_zero_extend,
            nets,
            interner,
            gb,
        ),
        AssignExpr::Not(inner) => {
            let Some(inner) = eval_assign_expr_bit(
                inner,
                rhs_bit_index,
                allow_literal_zero_extend,
                resolved,
                nets,
                interner,
                gb,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(gb.add_not(inner)))
        }
        AssignExpr::And(lhs, rhs) => {
            let Some(lhs) = eval_assign_expr_bit(
                lhs,
                rhs_bit_index,
                allow_literal_zero_extend,
                resolved,
                nets,
                interner,
                gb,
            )?
            else {
                return Ok(None);
            };
            let Some(rhs) = eval_assign_expr_bit(
                rhs,
                rhs_bit_index,
                allow_literal_zero_extend,
                resolved,
                nets,
                interner,
                gb,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(gb.add_and_binary(lhs, rhs)))
        }
        AssignExpr::Or(lhs, rhs) => {
            let Some(lhs) = eval_assign_expr_bit(
                lhs,
                rhs_bit_index,
                allow_literal_zero_extend,
                resolved,
                nets,
                interner,
                gb,
            )?
            else {
                return Ok(None);
            };
            let Some(rhs) = eval_assign_expr_bit(
                rhs,
                rhs_bit_index,
                allow_literal_zero_extend,
                resolved,
                nets,
                interner,
                gb,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(gb.add_or_binary(lhs, rhs)))
        }
        AssignExpr::Xor(lhs, rhs) => {
            let Some(lhs) = eval_assign_expr_bit(
                lhs,
                rhs_bit_index,
                allow_literal_zero_extend,
                resolved,
                nets,
                interner,
                gb,
            )?
            else {
                return Ok(None);
            };
            let Some(rhs) = eval_assign_expr_bit(
                rhs,
                rhs_bit_index,
                allow_literal_zero_extend,
                resolved,
                nets,
                interner,
                gb,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(gb.add_xor_binary(lhs, rhs)))
        }
    }
}

fn collect_missing_refs(
    expr: &AssignExpr,
    rhs_bit_index: usize,
    allow_literal_zero_extend: bool,
    resolved: &ResolvedNetValues,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    out: &mut Vec<String>,
) {
    match expr {
        AssignExpr::Leaf(net_ref) => match net_ref {
            NetRef::Simple(idx) => {
                if let Some(bit_number) = nets[idx.0].bit_number(rhs_bit_index)
                    && resolved
                        .resolve_bit(*idx, bit_number, nets, interner)
                        .ok()
                        .flatten()
                        .is_none()
                {
                    out.push(render_named_bit(*idx, bit_number, nets, interner));
                }
            }
            NetRef::BitSelect(idx, bit) => {
                if rhs_bit_index == 0
                    && resolved
                        .resolve_bit(*idx, *bit, nets, interner)
                        .ok()
                        .flatten()
                        .is_none()
                {
                    out.push(render_named_bit(*idx, *bit, nets, interner));
                }
            }
            NetRef::PartSelect(idx, msb, lsb) => {
                if let Some(bit_number) = select_bit_number(*msb, *lsb, rhs_bit_index)
                    && resolved
                        .resolve_bit(*idx, bit_number, nets, interner)
                        .ok()
                        .flatten()
                        .is_none()
                {
                    out.push(render_named_bit(*idx, bit_number, nets, interner));
                }
            }
            NetRef::Literal(_)
            | NetRef::UnknownLiteral(_)
            | NetRef::Unconnected
            | NetRef::Concat(_) => {}
        },
        AssignExpr::Not(inner) => collect_missing_refs(
            inner,
            rhs_bit_index,
            allow_literal_zero_extend,
            resolved,
            nets,
            interner,
            out,
        ),
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            collect_missing_refs(
                lhs,
                rhs_bit_index,
                allow_literal_zero_extend,
                resolved,
                nets,
                interner,
                out,
            );
            collect_missing_refs(
                rhs,
                rhs_bit_index,
                allow_literal_zero_extend,
                resolved,
                nets,
                interner,
                out,
            );
        }
    }
}

/// Projects a Liberty-free structural-assign module to a `GateFn`.
pub fn project_gatefn_from_structural_assigns(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<GateFn, String> {
    validate_structural_assign_module(module, nets, interner).map_err(|e| e.to_string())?;

    let module_name = interner.resolve(module.name).unwrap_or("<unknown>");
    let mut gb = GateBuilder::new(module_name.to_string(), GateBuilderOptions::no_opt());
    let mut resolved = ResolvedNetValues::new();

    for port in &module.ports {
        match port.direction {
            PortDirection::Input => {
                let net_idx = module.find_net_index(port.name, nets).ok_or_else(|| {
                    format!(
                        "input port '{}' net not found",
                        interner.resolve(port.name).unwrap_or("<unknown>")
                    )
                })?;
                let width = nets[net_idx.0].width_bits();
                let bv = gb.add_input(
                    interner
                        .resolve(port.name)
                        .unwrap_or("<unknown>")
                        .to_string(),
                    width,
                );
                resolved.seed_input(net_idx, &bv);
            }
            PortDirection::Inout => {
                return Err(
                    "inout ports are not supported in Liberty-free structural mode".to_string(),
                );
            }
            PortDirection::Output => {}
        }
    }

    let mut pending: Vec<PendingAssignBit> = Vec::new();
    for (assign_index, assign) in module.assigns.iter().enumerate() {
        let lhs_bits = expand_lhs_bits(&assign.lhs, nets, interner)?;
        let rhs_width = assign_expr_width_bits(&assign.rhs, nets, interner)?;
        let allow_literal_zero_extend = is_bare_literal_assign_expr(&assign.rhs);
        if lhs_bits.len() != rhs_width && !allow_literal_zero_extend {
            return Err(format!(
                "assign to '{}' has lhs width {} but rhs width {}; Liberty-free structural mode requires exact-width assignments except for bare literal tie-offs",
                render_net_ref(&assign.lhs, nets, interner),
                lhs_bits.len(),
                rhs_width
            ));
        }
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
        let mut processed_any = false;
        for pending_bit in pending {
            let assign = &module.assigns[pending_bit.assign_index];
            let allow_literal_zero_extend = is_bare_literal_assign_expr(&assign.rhs);
            let rhs = eval_assign_expr_bit(
                &assign.rhs,
                pending_bit.rhs_bit_index,
                allow_literal_zero_extend,
                &resolved,
                nets,
                interner,
                &mut gb,
            )?;
            let Some(rhs) = rhs else {
                next_pending.push(pending_bit);
                continue;
            };
            resolved.write_bit(
                pending_bit.target.idx,
                pending_bit.target.bit_number,
                rhs,
                nets,
                interner,
            )?;
            processed_any = true;
        }
        if !processed_any {
            let pending = next_pending;
            let mut lines = Vec::new();
            for pending_bit in pending.iter().take(10) {
                let assign = &module.assigns[pending_bit.assign_index];
                let mut missing = Vec::new();
                collect_missing_refs(
                    &assign.rhs,
                    pending_bit.rhs_bit_index,
                    is_bare_literal_assign_expr(&assign.rhs),
                    &resolved,
                    nets,
                    interner,
                    &mut missing,
                );
                missing.sort();
                missing.dedup();
                let lhs = render_named_bit(
                    pending_bit.target.idx,
                    pending_bit.target.bit_number,
                    nets,
                    interner,
                );
                if missing.is_empty() {
                    lines.push(format!(
                        "- assign to '{}' at {} likely participates in a dependency cycle",
                        lhs,
                        assign.span.to_human_string()
                    ));
                } else {
                    lines.push(format!(
                        "- assign to '{}' at {} is waiting on [{}]",
                        lhs,
                        assign.span.to_human_string(),
                        missing.join(", ")
                    ));
                }
            }
            return Err(format!(
                "could not resolve all structural assigns; {} assign bit(s) remain\n{}",
                pending.len(),
                lines.join("\n")
            ));
        }
        pending = next_pending;
    }

    for port in &module.ports {
        if port.direction != PortDirection::Output {
            continue;
        }
        let net_idx = module.find_net_index(port.name, nets).ok_or_else(|| {
            format!(
                "output port '{}' net not found",
                interner.resolve(port.name).unwrap_or("<unknown>")
            )
        })?;
        let Some(output_bv) =
            resolved.materialize_ref(&NetRef::Simple(net_idx), nets, interner, &mut gb)?
        else {
            return Err(format!(
                "output '{}' was not fully resolved after structural projection",
                interner.resolve(port.name).unwrap_or("<unknown>")
            ));
        };
        gb.add_output(
            interner
                .resolve(port.name)
                .unwrap_or("<unknown>")
                .to_string(),
            output_bv,
        );
    }

    Ok(gb.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_sim::gate_sim::{self, Collect};
    use crate::netlist::parse::{Parser, TokenScanner};
    use xlsynth::IrBits;

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

    #[test]
    fn project_gatefn_from_structural_assigns_supports_acyclic_overlapping_slice_dependencies() {
        let (module, nets, interner) = parse_single_module(
            r#"
module top(a, y);
  input a;
  output [3:0] y;
  assign y[3:1] = y[2:0];
  assign y[0] = a;
endmodule
"#,
        );
        let gate_fn =
            project_gatefn_from_structural_assigns(&module, &nets, &interner).expect("project");
        let sim = gate_sim::eval(
            &gate_fn,
            &[IrBits::make_ubits(1, 1).unwrap()],
            Collect::None,
        );
        assert_eq!(sim.outputs, vec![IrBits::make_ubits(4, 0b1111).unwrap()]);
    }

    #[test]
    fn project_gatefn_from_structural_assigns_supports_bare_literal_tieoff_resize() {
        let (module, nets, interner) = parse_single_module(
            r#"
module top(y);
  output [3:0] y;
  assign y = 1'b0;
endmodule
"#,
        );
        let gate_fn =
            project_gatefn_from_structural_assigns(&module, &nets, &interner).expect("project");
        let sim = gate_sim::eval(&gate_fn, &[], Collect::None);
        assert_eq!(sim.outputs, vec![IrBits::make_ubits(4, 0).unwrap()]);
    }

    #[test]
    fn project_gatefn_from_structural_assigns_supports_ascending_packed_range_selects() {
        let (module, nets, interner) = parse_single_module(
            r#"
module top(a, y);
  input [0:3] a;
  output [0:3] y;
  assign y[0:2] = a[0:2];
  assign y[3] = a[3];
endmodule
"#,
        );
        let gate_fn =
            project_gatefn_from_structural_assigns(&module, &nets, &interner).expect("project");
        let sim = gate_sim::eval(
            &gate_fn,
            &[IrBits::make_ubits(4, 0b1010).unwrap()],
            Collect::None,
        );
        assert_eq!(sim.outputs, vec![IrBits::make_ubits(4, 0b1010).unwrap()]);
    }
}
