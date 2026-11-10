// SPDX-License-Identifier: Apache-2.0

//! Accepted DSLX builtins, patterns, and casts with independently derived IR.

use super::ContractCase;
use xlsynth_pir::{BuilderError, FnBuilder, ir};

/// Isolates the lowest bit with two's-complement arithmetic. Reversing the
/// operand and result gives the highest bit. Zero sets the extra output bit.
fn one_hot_oracle(mode: u8, width: usize) -> Result<String, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(width))?;
    let priority = builder.param("priority", ir::Type::Bits(1))?;
    let neg = builder.neg(x)?;
    let low = builder.and(x, neg)?;
    let reversed = builder.rev(x)?;
    let neg_reversed = builder.neg(reversed)?;
    let low_reversed = builder.and(reversed, neg_reversed)?;
    let high = builder.rev(low_reversed)?;
    let chosen = match mode {
        0 | 2 => high,
        1 | 3 => low,
        4 => builder.select(priority, &[high, low], None)?,
        _ => builder.select(priority, &[low, high], None)?,
    };
    let any = builder.or_reduce(x)?;
    let none = builder.not(any)?;
    let result = builder.concat(&[none, chosen])?;
    Ok(builder.build_package(result, "expected")?.to_string())
}

/// Varies the priority expression independently from the one-hot input width.
pub(super) fn one_hot(data: &[u8]) -> Result<ContractCase, String> {
    let mode = data.first().copied().unwrap_or_default() % 6;
    let width = usize::from(data.get(1).copied().unwrap_or_default() % 15) + 2;
    let (family, priority) = match mode {
        0 => ("one_hot_literal", "false"),
        1 => ("one_hot_literal", "true"),
        2 => ("one_hot_computed", "!true"),
        3 => ("one_hot_computed", "!false"),
        4 => ("one_hot_runtime", "priority"),
        _ => ("one_hot_runtime", "!priority"),
    };
    let text = format!(
        "pub fn main(x: u{width}, priority: bool) -> u{} {{ one_hot(x, {priority}) }}\n",
        width + 1
    );
    Ok(ContractCase::single(
        family,
        text,
        one_hot_oracle(mode, width).map_err(|error| error.to_string())?,
    ))
}

/// Permutes two to four named payload fields. Two occurrences of the same
/// name in source order bind the first value and compare the second to it.
pub(super) fn payload_pattern(data: &[u8]) -> Result<ContractCase, String> {
    let count = usize::from(data.get(1).copied().unwrap_or_default() % 3) + 2;
    let width = usize::from(data.get(2).copied().unwrap_or_default() % 16) + 1;
    let fields = &["a", "b", "c", "d"][..count];
    let pairs: Vec<_> = (0..count)
        .flat_map(|lhs| (lhs + 1..count).map(move |rhs| (lhs, rhs)))
        .collect();
    let (lhs, rhs) = pairs[usize::from(data.get(3).copied().unwrap_or_default()) % pairs.len()];
    let mut order = usize::from(data.first().copied().unwrap_or_default());
    let mut remaining: Vec<_> = (0..count).collect();
    let mut pattern = Vec::new();
    while !remaining.is_empty() {
        let radix = remaining.len();
        let index = remaining.remove(order % radix);
        order /= radix;
        let name = if index == lhs || index == rhs {
            "v"
        } else {
            "_"
        };
        pattern.push(format!("{}: {name}", fields[index]));
    }
    let declarations = fields
        .iter()
        .map(|field| format!("{field}: u{width}"))
        .collect::<Vec<_>>()
        .join(", ");
    let arguments = fields.join(", ");
    let pattern = pattern.join(", ");
    let text = format!(
        r#"enum E {{ P {{ {declarations} }}, End }}
pub fn main({declarations}) -> bool {{
    let packet = E::P {{ {arguments} }};
    match packet {{ E::P {{ {pattern} }} => true, _ => false }}
}}
"#
    );
    let params = fields
        .iter()
        .enumerate()
        .map(|(index, field)| format!("{field}: bits[{width}] id={}", index + 1))
        .collect::<Vec<_>>()
        .join(", ");
    let expected_ir = format!(
        "package expected\n\ntop fn main({params}) -> bits[1] {{\n  ret same: bits[1] = eq({}, {}, id={})\n}}\n",
        fields[lhs],
        fields[rhs],
        count + 1
    );
    Ok(ContractCase::single("payload_pattern", text, expected_ir))
}

/// Equivalent spellings of a runtime array identity.
pub(crate) enum ArrayIdentityForm {
    Uncast,
    DirectCast,
    AliasCast,
    Rebuild,
}

/// Preserves the original array-identity byte mapping for focused replays.
pub(super) fn array_cast(data: &[u8]) -> ContractCase {
    let form = match data.first().copied().unwrap_or_default() % 4 {
        0 => ArrayIdentityForm::Uncast,
        1 => ArrayIdentityForm::DirectCast,
        2 => ArrayIdentityForm::AliasCast,
        _ => ArrayIdentityForm::Rebuild,
    };
    let width = usize::from(data.get(1).copied().unwrap_or_default() % 16) + 1;
    let length = usize::from(data.get(2).copied().unwrap_or_default() % 4) + 2;
    let sign = if data.get(3).copied().unwrap_or_default() & 1 == 0 {
        'u'
    } else {
        's'
    };
    array_identity("array_cast", form, width, length, sign)
}

/// Checks that a runtime array retains its full shape and contents.
pub(crate) fn array_identity(
    family: &'static str,
    form: ArrayIdentityForm,
    width: usize,
    length: usize,
    sign: char,
) -> ContractCase {
    let ty = if width <= 64 {
        format!("{sign}{width}[{length}]")
    } else {
        format!("{sign}N[{width}][{length}]")
    };
    let (declaration, expression) = match form {
        ArrayIdentityForm::Uncast => (String::new(), "x".into()),
        ArrayIdentityForm::DirectCast => (String::new(), format!("x as {ty}")),
        ArrayIdentityForm::AliasCast => (format!("type A = {ty};\n"), "x as A".into()),
        ArrayIdentityForm::Rebuild => {
            let elements = (0..length)
                .map(|index| format!("x[u32:{index}]"))
                .collect::<Vec<_>>()
                .join(", ");
            (String::new(), format!("[{elements}]"))
        }
    };
    let text = format!("{declaration}pub fn main(x: {ty}) -> {ty} {{ {expression} }}\n");
    let ir_ty = format!("bits[{width}][{length}]");
    let expected_ir = format!(
        "package expected\n\ntop fn main(x: {ir_ty} id=1) -> {ir_ty} {{\n  ret x: {ir_ty} = param(name=x, id=1)\n}}\n"
    );
    ContractCase::single(family, text, expected_ir)
}

#[cfg(test)]
mod tests {
    use super::one_hot;
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn all_priority_forms_agree_with_two_bit_truth_tables() -> Result<(), Box<dyn std::error::Error>>
    {
        let msb = [4, 1, 2, 2];
        let lsb = [4, 1, 2, 1];
        for mode in 0..6 {
            let case = one_hot(&[mode])?;
            let package = IrPackage::parse_ir(&case.expected_ir, None)?;
            let function = package.get_function("main")?;
            for (priority, table) in match mode {
                0 | 2 => [(false, msb), (true, msb)],
                1 | 3 => [(false, lsb), (true, lsb)],
                4 => [(false, msb), (true, lsb)],
                _ => [(false, lsb), (true, msb)],
            } {
                for (x, expected) in table.into_iter().enumerate() {
                    let inputs = [
                        XlsIrValue::make_ubits(2, x as u64)?,
                        XlsIrValue::make_ubits(1, u64::from(priority))?,
                    ];
                    assert_eq!(
                        function.interpret(&inputs)?,
                        XlsIrValue::make_ubits(3, expected)?
                    );
                }
            }
        }
        Ok(())
    }
}
