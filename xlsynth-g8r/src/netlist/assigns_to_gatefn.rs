// SPDX-License-Identifier: Apache-2.0

//! Project a Liberty-free structural assign netlist into a `GateFn`.

use crate::aig::{AigBitVector, AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::netlist::integrity::validate_structural_assign_module;
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistAssign, NetlistModule, PortDirection,
};
use std::collections::HashMap;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

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

    fn ensure_entry(&mut self, idx: NetIndex, width: usize) -> &mut Vec<Option<AigOperand>> {
        self.values.entry(idx).or_insert_with(|| vec![None; width])
    }

    fn materialize_ref(
        &self,
        net_ref: &NetRef,
        nets: &[Net],
        gb: &mut GateBuilder,
    ) -> Result<Option<AigBitVector>, String> {
        match net_ref {
            NetRef::Simple(idx) => {
                let Some(bits) = self.values.get(idx) else {
                    return Ok(None);
                };
                if bits.iter().any(|bit| bit.is_none()) {
                    return Ok(None);
                }
                Ok(Some(AigBitVector::from_lsb_is_index_0(
                    &bits.iter().map(|bit| bit.unwrap()).collect::<Vec<_>>(),
                )))
            }
            NetRef::BitSelect(idx, bit) => {
                let (width, lsb) = net_width_bits(&nets[idx.0]);
                let offset_i64 = i64::from(*bit) - lsb;
                if offset_i64 < 0 || offset_i64 >= width as i64 {
                    return Err(format!("bit {} out of range for net index {}", bit, idx.0));
                }
                let Some(bits) = self.values.get(idx) else {
                    return Ok(None);
                };
                let Some(Some(bit_value)) = bits.get(offset_i64 as usize) else {
                    return Ok(None);
                };
                Ok(Some(AigBitVector::from_bit(*bit_value)))
            }
            NetRef::PartSelect(idx, msb, lsb_sel) => {
                if msb < lsb_sel {
                    return Err(format!(
                        "invalid part-select [{}:{}] on net index {}",
                        msb, lsb_sel, idx.0
                    ));
                }
                let (width, net_lsb) = net_width_bits(&nets[idx.0]);
                let Some(bits) = self.values.get(idx) else {
                    return Ok(None);
                };
                let mut out = Vec::new();
                for bit in *lsb_sel..=*msb {
                    let offset_i64 = i64::from(bit) - net_lsb;
                    if offset_i64 < 0 || offset_i64 >= width as i64 {
                        return Err(format!("bit {} out of range for net index {}", bit, idx.0));
                    }
                    let Some(Some(bit_value)) = bits.get(offset_i64 as usize) else {
                        return Ok(None);
                    };
                    out.push(*bit_value);
                }
                Ok(Some(AigBitVector::from_lsb_is_index_0(&out)))
            }
            NetRef::Literal(bits) => {
                let mut out = Vec::new();
                for i in 0..bits.get_bit_count() {
                    let operand = if bits.get_bit(i).unwrap_or(false) {
                        gb.get_true()
                    } else {
                        gb.get_false()
                    };
                    out.push(operand);
                }
                Ok(Some(AigBitVector::from_lsb_is_index_0(&out)))
            }
            NetRef::Unconnected => Err("unconnected net reference is not supported".to_string()),
            NetRef::Concat(_) => {
                Err("concatenation is not supported in structural mode".to_string())
            }
        }
    }

    fn is_ref_resolved(&self, net_ref: &NetRef, nets: &[Net]) -> bool {
        match net_ref {
            NetRef::Simple(idx) => self
                .values
                .get(idx)
                .is_some_and(|bits| bits.iter().all(|bit| bit.is_some())),
            NetRef::BitSelect(idx, bit) => {
                let (width, lsb) = net_width_bits(&nets[idx.0]);
                let offset_i64 = i64::from(*bit) - lsb;
                if offset_i64 < 0 || offset_i64 >= width as i64 {
                    return false;
                }
                self.values
                    .get(idx)
                    .and_then(|bits| bits.get(offset_i64 as usize))
                    .is_some_and(|bit| bit.is_some())
            }
            NetRef::PartSelect(idx, msb, lsb_sel) => {
                if msb < lsb_sel {
                    return false;
                }
                let (width, net_lsb) = net_width_bits(&nets[idx.0]);
                let Some(bits) = self.values.get(idx) else {
                    return false;
                };
                for bit in *lsb_sel..=*msb {
                    let offset_i64 = i64::from(bit) - net_lsb;
                    if offset_i64 < 0 || offset_i64 >= width as i64 {
                        return false;
                    }
                    if bits
                        .get(offset_i64 as usize)
                        .is_none_or(|entry| entry.is_none())
                    {
                        return false;
                    }
                }
                true
            }
            NetRef::Literal(_) => true,
            NetRef::Unconnected | NetRef::Concat(_) => false,
        }
    }

    fn write_lhs(
        &mut self,
        lhs: &NetRef,
        rhs: &AigBitVector,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), String> {
        match lhs {
            NetRef::Simple(idx) => {
                let (width, _) = net_width_bits(&nets[idx.0]);
                if rhs.get_bit_count() != width {
                    return Err(format!(
                        "width mismatch assigning '{}' : lhs width {} rhs width {}",
                        net_name(*idx, nets, interner),
                        width,
                        rhs.get_bit_count()
                    ));
                }
                let entry = self.ensure_entry(*idx, width);
                if entry.iter().any(|bit| bit.is_some()) {
                    return Err(format!(
                        "net '{}' was assigned more than once during projection",
                        net_name(*idx, nets, interner)
                    ));
                }
                for i in 0..width {
                    entry[i] = Some(*rhs.get_lsb(i));
                }
                Ok(())
            }
            NetRef::BitSelect(idx, bit) => {
                if rhs.get_bit_count() != 1 {
                    return Err(format!(
                        "bit-select assignment to '{}' requires 1-bit rhs, got {} bits",
                        net_name(*idx, nets, interner),
                        rhs.get_bit_count()
                    ));
                }
                let (width, lsb) = net_width_bits(&nets[idx.0]);
                let offset_i64 = i64::from(*bit) - lsb;
                if offset_i64 < 0 || offset_i64 >= width as i64 {
                    return Err(format!(
                        "bit {} out of range for net '{}'",
                        bit,
                        net_name(*idx, nets, interner)
                    ));
                }
                let offset = offset_i64 as usize;
                let entry = self.ensure_entry(*idx, width);
                if entry[offset].is_some() {
                    return Err(format!(
                        "net '{}' bit {} was assigned more than once during projection",
                        net_name(*idx, nets, interner),
                        bit
                    ));
                }
                entry[offset] = Some(*rhs.get_lsb(0));
                Ok(())
            }
            NetRef::PartSelect(idx, msb, lsb_sel) => {
                if msb < lsb_sel {
                    return Err(format!(
                        "invalid part-select [{}:{}] on '{}'",
                        msb,
                        lsb_sel,
                        net_name(*idx, nets, interner)
                    ));
                }
                let part_width = (*msb as usize) - (*lsb_sel as usize) + 1;
                if rhs.get_bit_count() != part_width {
                    return Err(format!(
                        "part-select assignment to '{}' requires {} rhs bits, got {}",
                        net_name(*idx, nets, interner),
                        part_width,
                        rhs.get_bit_count()
                    ));
                }
                let (width, lsb) = net_width_bits(&nets[idx.0]);
                let entry = self.ensure_entry(*idx, width);
                for (i, bit) in (*lsb_sel..=*msb).enumerate() {
                    let offset_i64 = i64::from(bit) - lsb;
                    if offset_i64 < 0 || offset_i64 >= width as i64 {
                        return Err(format!(
                            "bit {} out of range for net '{}'",
                            bit,
                            net_name(*idx, nets, interner)
                        ));
                    }
                    let offset = offset_i64 as usize;
                    if entry[offset].is_some() {
                        return Err(format!(
                            "net '{}' bit {} was assigned more than once during projection",
                            net_name(*idx, nets, interner),
                            bit
                        ));
                    }
                    entry[offset] = Some(*rhs.get_lsb(i));
                }
                Ok(())
            }
            NetRef::Literal(_) | NetRef::Unconnected | NetRef::Concat(_) => {
                Err("left-hand side of structural assign must be a net or net select".to_string())
            }
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

fn eval_assign_expr(
    expr: &AssignExpr,
    resolved: &ResolvedNetValues,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
) -> Result<Option<AigBitVector>, String> {
    match expr {
        AssignExpr::Leaf(net_ref) => resolved.materialize_ref(net_ref, nets, gb),
        AssignExpr::Not(inner) => {
            let Some(inner) = eval_assign_expr(inner, resolved, nets, interner, gb)? else {
                return Ok(None);
            };
            Ok(Some(gb.add_not_vec(&inner)))
        }
        AssignExpr::And(lhs, rhs) => {
            let Some(lhs) = eval_assign_expr(lhs, resolved, nets, interner, gb)? else {
                return Ok(None);
            };
            let Some(rhs) = eval_assign_expr(rhs, resolved, nets, interner, gb)? else {
                return Ok(None);
            };
            if lhs.get_bit_count() != rhs.get_bit_count() {
                return Err(format!(
                    "bitwise '&' width mismatch: lhs {} bits rhs {} bits",
                    lhs.get_bit_count(),
                    rhs.get_bit_count()
                ));
            }
            Ok(Some(gb.add_and_vec(&lhs, &rhs)))
        }
        AssignExpr::Or(lhs, rhs) => {
            let Some(lhs) = eval_assign_expr(lhs, resolved, nets, interner, gb)? else {
                return Ok(None);
            };
            let Some(rhs) = eval_assign_expr(rhs, resolved, nets, interner, gb)? else {
                return Ok(None);
            };
            if lhs.get_bit_count() != rhs.get_bit_count() {
                return Err(format!(
                    "bitwise '|' width mismatch: lhs {} bits rhs {} bits",
                    lhs.get_bit_count(),
                    rhs.get_bit_count()
                ));
            }
            Ok(Some(gb.add_or_vec(&lhs, &rhs)))
        }
        AssignExpr::Xor(lhs, rhs) => {
            let Some(lhs) = eval_assign_expr(lhs, resolved, nets, interner, gb)? else {
                return Ok(None);
            };
            let Some(rhs) = eval_assign_expr(rhs, resolved, nets, interner, gb)? else {
                return Ok(None);
            };
            if lhs.get_bit_count() != rhs.get_bit_count() {
                return Err(format!(
                    "bitwise '^' width mismatch: lhs {} bits rhs {} bits",
                    lhs.get_bit_count(),
                    rhs.get_bit_count()
                ));
            }
            Ok(Some(gb.add_xor_vec(&lhs, &rhs)))
        }
    }
}

fn collect_missing_refs(
    expr: &AssignExpr,
    resolved: &ResolvedNetValues,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    out: &mut Vec<String>,
) {
    match expr {
        AssignExpr::Leaf(net_ref) => {
            if !resolved.is_ref_resolved(net_ref, nets) {
                out.push(render_net_ref(net_ref, nets, interner));
            }
        }
        AssignExpr::Not(inner) => collect_missing_refs(inner, resolved, nets, interner, out),
        AssignExpr::And(lhs, rhs) | AssignExpr::Or(lhs, rhs) | AssignExpr::Xor(lhs, rhs) => {
            collect_missing_refs(lhs, resolved, nets, interner, out);
            collect_missing_refs(rhs, resolved, nets, interner, out);
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
                let width = net_width_bits(&nets[net_idx.0]).0;
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

    let mut pending: Vec<&NetlistAssign> = module.assigns.iter().collect();
    let mut processed_any = true;
    while !pending.is_empty() && processed_any {
        processed_any = false;
        let mut i = 0;
        while i < pending.len() {
            let assign = pending[i];
            let rhs = eval_assign_expr(&assign.rhs, &resolved, nets, interner, &mut gb)?;
            let Some(rhs) = rhs else {
                i += 1;
                continue;
            };
            resolved.write_lhs(&assign.lhs, &rhs, nets, interner)?;
            pending.remove(i);
            processed_any = true;
        }
    }

    if !pending.is_empty() {
        let mut lines = Vec::new();
        for assign in pending.iter().take(10) {
            let mut missing = Vec::new();
            collect_missing_refs(&assign.rhs, &resolved, nets, interner, &mut missing);
            missing.sort();
            missing.dedup();
            let lhs = render_net_ref(&assign.lhs, nets, interner);
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
            "could not resolve all structural assigns; {} assign(s) remain\n{}",
            pending.len(),
            lines.join("\n")
        ));
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
        let Some(output_bv) = resolved.materialize_ref(&NetRef::Simple(net_idx), nets, &mut gb)?
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
