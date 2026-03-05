// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_vastly::LogicBit;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;
use xlsynth::IrValue;
use xlsynth::XlsynthError;
use xlsynth_pir::ir::Type as PirType;

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
    for p in &f.params {
        let width = packed_width(&p.ty)?;
        if width == 0 {
            return None;
        }
        params.push(TypedPortSig {
            name: p.name.clone(),
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
    bits: &xlsynth::IrBits,
    out: &mut Vec<LogicBit>,
) -> Result<(), XlsynthError> {
    for i in 0..bits.get_bit_count() {
        let b = bits.get_bit(i)?;
        out.push(if b { LogicBit::One } else { LogicBit::Zero });
    }
    Ok(())
}

pub fn pack_ir_value_to_value4(ty: &PirType, value: &IrValue) -> Result<Value4, XlsynthError> {
    let width = packed_width(ty)
        .ok_or_else(|| XlsynthError(format!("unsupported top-level type for packing: {ty}")))?;
    let mut out = Vec::with_capacity(width as usize);
    append_packed_bits_lsb(ty, value, &mut out)?;
    if out.len() != width as usize {
        return Err(XlsynthError(format!(
            "packed width mismatch for type {ty}: expected {width}, got {}",
            out.len()
        )));
    }
    Ok(Value4::new(width, Signedness::Unsigned, out))
}

pub fn make_vastly_input_map(
    sig: &PackedSig,
    args: &[IrValue],
) -> Result<BTreeMap<String, Value4>, XlsynthError> {
    let mut out = BTreeMap::new();
    for (param, arg) in sig.params.iter().zip(args.iter()) {
        let packed = pack_ir_value_to_value4(&param.ty, arg)?;
        if packed.width != param.width {
            return Err(XlsynthError(format!(
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
) -> Result<(), XlsynthError> {
    match ty {
        PirType::Token => Err(XlsynthError("cannot pack token value".to_string())),
        PirType::Bits(width) => {
            let bits = value.to_bits()?;
            if bits.get_bit_count() != *width {
                return Err(XlsynthError(format!(
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
                return Err(XlsynthError(format!(
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
                return Err(XlsynthError(format!(
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
