// SPDX-License-Identifier: Apache-2.0

//! Signed array packing in module constants and runtime expressions.

use crate::converter_contracts::ContractCase;

/// Whether the packed word is observed directly or widened first.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cast {
    Pack,
    Widen,
}

/// Observe whether a packed signed word keeps its sign when widened.
pub(crate) fn widen(data: &[u8]) -> ContractCase {
    generate(data, Cast::Widen)
}

/// Observe the packed bits of signed array elements.
pub(crate) fn pack(data: &[u8]) -> ContractCase {
    generate(data, Cast::Pack)
}

/// Varies the array shape and sign bit. The reference directly concatenates
/// lane bits in array order and extends the packed result when requested.
fn generate(data: &[u8], cast: Cast) -> ContractCase {
    let byte = |index: usize| data.get(index).copied().unwrap_or_default();
    let flags = byte(0);
    let runtime = flags & 1 != 0;
    let unsigned_control = flags & 2 != 0;
    let width = [4, 2, 8, 16, 32][usize::from(byte(1)) % 5];
    let length = 2 + usize::from(byte(2) % 2);
    let packed_width = length * width;
    let result_width = match cast {
        Cast::Pack => packed_width,
        Cast::Widen => packed_width + [8, 1, 4, 16][usize::from(byte(3) % 4)],
    };
    let signed_lanes = cast == Cast::Pack && !unsigned_control;
    let signed_word = cast == Cast::Widen && !unsigned_control;
    let lane_type = bits_type(signed_lanes, width);
    let packed_type = bits_type(signed_word, packed_width);
    let result_type = bits_type(cast == Cast::Widen, result_width);
    let mut dslx_params = Vec::new();
    let mut ir_params = Vec::new();
    let mut ir_statements = Vec::new();
    let mut elements = Vec::new();
    let mut names = Vec::new();
    let sign_bit = 1u64 << (width - 1);
    for (index, default) in [0, 3, 1].into_iter().take(length).enumerate() {
        let name = format!("x{index}");
        let id = index + 1;
        if runtime {
            dslx_params.push(format!("{name}: {lane_type}"));
            ir_params.push(format!("{name}: bits[{width}] id={id}"));
            elements.push(name.clone());
        } else {
            let value = u64::from(data.get(index + 4).copied().unwrap_or(default));
            let bits = if index == 0 {
                let high_bit = if flags & 4 == 0 { sign_bit } else { 0 };
                high_bit | (value & (sign_bit - 1))
            } else {
                value & (2 * sign_bit - 1)
            };
            let literal = if signed_lanes && bits & sign_bit != 0 {
                format!("-{}", 2 * sign_bit - bits)
            } else {
                bits.to_string()
            };
            elements.push(format!("{lane_type}:{literal}"));
            ir_statements.push(format!(
                "{name}: bits[{width}] = literal(value={bits}, id={id})"
            ));
        }
        names.push(name);
    }

    let array = elements.join(", ");
    let packed = format!("[{array}] as {packed_type}");
    let expression = match cast {
        Cast::Pack => packed,
        Cast::Widen => format!("({packed}) as {result_type}"),
    };
    let source = if runtime {
        let params = dslx_params.join(", ");
        format!("pub fn main({params}) -> {result_type} {{ {expression} }}\n")
    } else {
        format!("const PACKED = {expression};\npub fn main() -> {result_type} {{ PACKED }}\n")
    };

    let operands = names.join(", ");
    let pack_id = length + 1;
    let family = match cast {
        Cast::Pack => {
            ir_statements.push(format!(
                "ret packed: bits[{packed_width}] = concat({operands}, id={pack_id})"
            ));
            "signed_array_element_pack"
        }
        Cast::Widen => {
            ir_statements.push(format!(
                "packed: bits[{packed_width}] = concat({operands}, id={pack_id})"
            ));
            let op = if signed_word { "sign_ext" } else { "zero_ext" };
            let result_id = pack_id + 1;
            ir_statements.push(format!(
                "ret result: bits[{result_width}] = {op}(packed, new_bit_count={result_width}, id={result_id})"
            ));
            "packed_array_signed_widen"
        }
    };
    let params = ir_params.join(", ");
    let statements = ir_statements.join("\n  ");
    let expected_ir = format!(
        "package expected\ntop fn main({params}) -> bits[{result_width}] {{\n  {statements}\n}}\n"
    );
    ContractCase::single(family, source, expected_ir)
}

/// Formats a concrete DSLX bits type, including widths beyond shorthand.
fn bits_type(signed: bool, width: usize) -> String {
    let prefix = if signed { 's' } else { 'u' };
    if width <= 64 {
        format!("{prefix}{width}")
    } else {
        format!("{prefix}N[{width}]")
    }
}

#[cfg(test)]
mod tests {
    use super::{pack, widen};
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn module_constants_expose_packed_bits_and_the_sign_extension()
    -> Result<(), Box<dyn std::error::Error>> {
        let widen_case = widen(&[]);
        let pack_case = pack(&[]);
        assert_eq!(
            widen_case.modules[0].text,
            "const PACKED = ([u4:8, u4:3] as s8) as s16;\npub fn main() -> s16 { PACKED }\n"
        );
        assert_eq!(
            pack_case.modules[0].text,
            "const PACKED = [s4:-8, s4:3] as u8;\npub fn main() -> u8 { PACKED }\n"
        );
        for (case, width, expected) in [
            (widen_case, 16, 0xff83),
            (widen(&[2]), 16, 0x83),
            (pack_case, 8, 0x83),
            (pack(&[2]), 8, 0x83),
        ] {
            let ir = IrPackage::parse_ir(&case.expected_ir, None)?;
            assert_eq!(
                ir.get_function("main")?.interpret(&[])?,
                XlsIrValue::make_ubits(width, expected)?
            );
        }
        Ok(())
    }

    #[test]
    fn runtime_reference_uses_every_lane_in_array_order() -> Result<(), Box<dyn std::error::Error>>
    {
        let inputs = [8, 3, 7]
            .map(|value| XlsIrValue::make_ubits(4, value))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        for (case, width, expected) in [
            (widen(&[1, 0, 1]), 20, 0xff837),
            (widen(&[3, 0, 1]), 20, 0x837),
            (pack(&[1, 0, 1]), 12, 0x837),
        ] {
            let ir = IrPackage::parse_ir(&case.expected_ir, None)?;
            assert_eq!(
                ir.get_function("main")?.interpret(&inputs)?,
                XlsIrValue::make_ubits(width, expected)?
            );
        }
        Ok(())
    }
}
