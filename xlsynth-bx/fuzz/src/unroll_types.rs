// SPDX-License-Identifier: Apache-2.0

//! Compare loop acceptance and inferred types before lowering to signless IR.

use xlsynth::dslx::{self, ImportData, MatchableModuleMember};
use xlsynth_bx::ast::Type;
use xlsynth_bx::check::CheckedStmt;
use xlsynth_bx::{check_module, parse_module};

use crate::eval::expr_eval::{bx_source, dslx_source};

/// Compare the inferred signedness, width and dimensions of bits and arrays.
fn same_type(bx: &Type, native: &dslx::Type) -> bool {
    match bx {
        Type::Bool => native.is_bits_like() == Some((false, 1)),
        Type::Bits { signed, width } => native.is_bits_like() == Some((*signed, *width)),
        Type::Array { element, size } => {
            native.is_array()
                && *size == native.get_array_size()
                && same_type(element, &native.get_array_element_type())
        }
        _ => false,
    }
}

/// Generate a typed or inferred loop, optionally replacing one component
/// with a runtime range endpoint or an incompatible type.
pub fn check_unroll_types(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let mode = flags % 8;
    let width = usize::from(bytes.next().unwrap()) % 129 + 1;
    let sign = if flags & 8 == 0 { "u" } else { "s" };
    let element = format!("{sign}N[{width}]");
    let array = flags & 16 != 0;
    let ty = if array {
        format!("{element}[2]")
    } else {
        element.clone()
    };
    let start = bytes.next().unwrap() % 4;
    let count = bytes.next().unwrap() % 4 + u8::from(mode >= 2);
    let end = start + count;
    let range = match mode {
        2 => format!("bound..u32:{end}"),
        3 => format!("u32:{start}..bound"),
        4 => format!("u32:{end}..u32:{start}"),
        _ => format!("u32:{start}..u32:{end}"),
    };
    let annotation = match mode {
        5 => format!(": (u16, {ty})"),
        6 => format!(": (u32, uN[{}])", width + 1),
        1 => String::new(),
        _ => format!(": (u32, {ty})"),
    };
    let step = if mode == 7 {
        format!("i as uN[{}]", width + 1)
    } else if array {
        format!("update(acc, i % u32:2, i as {element})")
    } else {
        format!("acc - (i as {element})")
    };
    let expr = format!("unroll_for! (i, acc){annotation} in {range} {{ {step} }}(zero!<{ty}>())");
    let body = format!("let folded = {expr}; true");
    let block = bx_source("bound: u32", "bool", &body);
    let checked = parse_module(&block).and_then(|module| check_module(&module));
    let mut imports = ImportData::new(None, &[]);
    if mode >= 2 {
        let reference = dslx_source("bound: u32", "bool", &body);
        let native = dslx::parse_and_typecheck(&reference, "oracle.x", "oracle", &mut imports);
        if checked.is_ok() || native.is_ok() {
            return Err(format!(
                "expected rejection, BX error {:?}, DSLX error {:?}\n{block}",
                checked.err(),
                native.err()
            ));
        }
        return Ok(());
    }
    // The native API exposes an inferred constant's type, including
    // signedness. The BX let and the DSLX constant have no result annotation.
    let reference = format!("const FOLDED = {expr};");
    let native = dslx::parse_and_typecheck(&reference, "oracle.x", "oracle", &mut imports)
        .map_err(|e| format!("{e}\n{reference}"))?;
    let Some(MatchableModuleMember::ConstantDef(folded)) =
        native.get_module().get_member(0).to_matchable()
    else {
        return Err(format!("missing DSLX constant\n{reference}"));
    };
    let native_ty = native.get_type_info().get_type_for_constant_def(&folded);
    let checked = checked.map_err(|e| format!("{e}\n{block}"))?;
    let bx_ty = checked.blocks()[0]
        .statements
        .iter()
        .find_map(|s| match s {
            CheckedStmt::Let { name, value } if name == "folded" => Some(&value.ty),
            _ => None,
        })
        .ok_or_else(|| format!("missing BX binding\n{block}"))?;
    if !same_type(bx_ty, &native_ty) {
        return Err(format!(
            "inferred BX {bx_ty}, DSLX {}\n{block}",
            native_ty.to_string().map_err(|e| e.to_string())?
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_unroll_types;

    #[test]
    fn compare_loop_acceptance_and_inferred_types() -> Result<(), String> {
        for mode in 0..32 {
            check_unroll_types(&[mode, 64, 2, 2])?;
        }
        Ok(())
    }
}
