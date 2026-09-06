// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_pir::IrValue;
use xlsynth_pir::ValueError;
use xlsynth_pir::ir::{Binop, Fn as PirFn, NodePayload, Type as PirType};
use xlsynth_pir::ir_utils::operands;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;

pub const MAX_PIPELINE_MULTIPLY_CHAIN_DEPTH: usize = 8;

/// Returns the largest number of multiplications on any node-dependency path.
pub fn max_multiply_chain_depth(f: &PirFn) -> usize {
    let mut depths = vec![0usize; f.nodes.len()];
    for (index, node) in f.nodes.iter().enumerate() {
        let operand_depth = operands(&node.payload)
            .iter()
            .map(|operand| depths[operand.index])
            .max()
            .unwrap_or(0);
        let is_multiply = matches!(
            node.payload,
            NodePayload::Binop(Binop::Umul | Binop::Smul, _, _)
        );
        depths[index] = operand_depth + usize::from(is_multiply);
    }
    depths.into_iter().max().unwrap_or(0)
}

#[derive(Debug, Clone)]
pub struct TypedPortSig {
    pub name: String,
    pub ty: PirType,
    pub width: u32,
}

#[derive(Debug, Clone)]
pub struct PackedSig {
    pub params: Vec<TypedPortSig>,
    pub ret_ty: PirType,
    pub ret_width: u32,
}

pub fn parse_pir_top_fn(ir_text: &str, top_name: &str) -> Result<xlsynth_pir::ir::Fn, String> {
    let mut parser = xlsynth_pir::ir_parser::Parser::new(ir_text);
    let mut pkg = parser
        .parse_and_validate_package()
        .map_err(|e| format!("PIR parse/validate failed: {e}"))?;
    pkg.set_top_fn(top_name)
        .map_err(|e| format!("failed to set PIR top fn `{top_name}`: {e}"))?;
    pkg.get_top_fn()
        .cloned()
        .ok_or_else(|| format!("PIR top fn `{top_name}` not found after parse"))
}

pub fn packed_signature(f: &xlsynth_pir::ir::Fn) -> Option<PackedSig> {
    let mut params = Vec::with_capacity(f.params.len());
    for p in f.param_nodes() {
        let width = packed_width(&p.ty)?;
        if width == 0 {
            return None;
        }
        params.push(TypedPortSig {
            name: p.param_name().to_string(),
            ty: p.ty.clone(),
            width,
        });
    }
    let ret_width = packed_width(&f.ret_ty)?;
    if ret_width == 0 {
        return None;
    }
    Some(PackedSig {
        params,
        ret_ty: f.ret_ty.clone(),
        ret_width,
    })
}

pub fn packed_width(ty: &PirType) -> Option<u32> {
    let width = ty.bit_count();
    if width == 0 {
        return None;
    }
    if contains_token(ty) {
        return None;
    }
    u32::try_from(width).ok()
}

pub fn contains_token(ty: &PirType) -> bool {
    match ty {
        PirType::Token => true,
        PirType::Bits(_) => false,
        PirType::Tuple(fields) => fields.iter().any(|f| contains_token(f)),
        PirType::Array(data) => contains_token(&data.element_type),
    }
}

fn append_ir_bits_lsb(
    bits: &xlsynth_pir::IrBits,
    out: &mut Vec<LogicBit>,
) -> Result<(), ValueError> {
    for i in 0..bits.get_bit_count() {
        let b = bits.get_bit(i)?;
        out.push(if b { LogicBit::One } else { LogicBit::Zero });
    }
    Ok(())
}

pub fn pack_ir_value_to_value4(ty: &PirType, value: &IrValue) -> Result<Value4, ValueError> {
    let width = packed_width(ty)
        .ok_or_else(|| ValueError(format!("unsupported top-level type for packing: {ty}")))?;
    let mut out = Vec::with_capacity(width as usize);
    append_packed_bits_lsb(ty, value, &mut out)?;
    if out.len() != width as usize {
        return Err(ValueError(format!(
            "packed width mismatch for type {ty}: expected {width}, got {}",
            out.len()
        )));
    }
    Ok(Value4::new(width, Signedness::Unsigned, out))
}

pub fn make_vastly_input_map(
    sig: &PackedSig,
    args: &[IrValue],
) -> Result<BTreeMap<String, Value4>, ValueError> {
    let mut out = BTreeMap::new();
    for (param, arg) in sig.params.iter().zip(args.iter()) {
        let packed = pack_ir_value_to_value4(&param.ty, arg)?;
        if packed.width != param.width {
            return Err(ValueError(format!(
                "packed arg width mismatch for `{}`: expected {}, got {}",
                param.name, param.width, packed.width
            )));
        }
        out.insert(param.name.clone(), packed);
    }
    Ok(out)
}

fn append_packed_bits_lsb(
    ty: &PirType,
    value: &IrValue,
    out: &mut Vec<LogicBit>,
) -> Result<(), ValueError> {
    match ty {
        PirType::Token => Err(ValueError("cannot pack token value".to_string())),
        PirType::Bits(width) => {
            let bits = value.to_bits()?;
            if bits.get_bit_count() != *width {
                return Err(ValueError(format!(
                    "bits width mismatch while packing: type says bits[{width}] but value is bits[{}]",
                    bits.get_bit_count()
                )));
            }
            append_ir_bits_lsb(&bits, out)?;
            Ok(())
        }
        PirType::Tuple(fields) => {
            let elements = value.get_elements()?;
            if elements.len() != fields.len() {
                return Err(ValueError(format!(
                    "tuple arity mismatch while packing: type has {} fields, value has {}",
                    fields.len(),
                    elements.len()
                )));
            }
            for (field_ty, element) in fields.iter().zip(elements.iter()).rev() {
                append_packed_bits_lsb(field_ty, element, out)?;
            }
            Ok(())
        }
        PirType::Array(data) => {
            let elements = value.get_elements()?;
            if elements.len() != data.element_count {
                return Err(ValueError(format!(
                    "array length mismatch while packing: type has {} elements, value has {}",
                    data.element_count,
                    elements.len()
                )));
            }
            for element in &elements {
                append_packed_bits_lsb(&data.element_type, element, out)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::max_multiply_chain_depth;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn multiply_depth_follows_dependencies_through_structural_nodes() {
        let package = Parser::new(
            r#"package test
top fn f(x: bits[8] id=1) -> bits[8] {
  square: bits[8] = umul(x, x, id=2)
  slice: bits[8] = bit_slice(square, start=0, width=8, id=3)
  fourth: bits[8] = umul(slice, slice, id=4)
  independent: bits[8] = smul(x, x, id=5)
  ret eighth: bits[8] = umul(fourth, independent, id=6)
}
"#,
        )
        .parse_and_validate_package()
        .unwrap();
        assert_eq!(max_multiply_chain_depth(package.get_top_fn().unwrap()), 3);
    }
}
