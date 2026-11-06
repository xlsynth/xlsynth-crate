// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, ensure};
use xlsynth::dslx::{ImportData, MatchableModuleMember, parse_and_typecheck};
use xlsynth_lean::block::{LeanBlockBundle, compile_bx_with_dslx};

use super::{lean_bundle, parametric_cases::Case};

/// Executable Lean and native results for each generated runtime input set.
pub struct CheckedCase {
    pub bundle: LeanBlockBundle,
    pub expected: Vec<Vec<(u16, u16)>>,
}

/// Evaluate concrete calls in native DSLX and check the emitted runtime caller.
pub fn check_case(case: &Case) -> Result<()> {
    let checked = checked_bundle(case)?;
    assert_eq!(checked.expected.len(), case.inputs.len());
    lean_bundle::check_bundle(&checked.bundle).with_context(|| case.source.clone())
}

/// Build guards from AST interpreter results, independently of IR lowering.
pub fn checked_bundle(case: &Case) -> Result<CheckedCase> {
    let calls = case
        .inputs
        .iter()
        .enumerate()
        .map(|(index, inputs)| {
            let args = inputs
                .iter()
                .map(|value| format!("u16:{value}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("const RESULT_{index} = g({args});")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let source = format!("{}\n{calls}", case.source);
    let mut imports = ImportData::new(None, &[]);
    let checked = parse_and_typecheck(&source, "machine.x", "machine", &mut imports)
        .with_context(|| format!("typecheck generated DSLX:\n{source}"))?;
    let module = checked.get_module();
    let type_info = checked.get_type_info();
    let mut expected = Vec::new();
    for index in 0..case.inputs.len() {
        let name = format!("RESULT_{index}");
        let constant = (0..module.get_member_count())
            .find_map(|member| match module.get_member(member).to_matchable() {
                Some(MatchableModuleMember::ConstantDef(constant))
                    if constant.get_name() == name =>
                {
                    Some(constant)
                }
                _ => None,
            })
            .with_context(|| format!("missing oracle constant {name}"))?;
        let value = type_info
            .get_const_expr(&constant.get_value())?
            .convert_to_ir()?;
        let values = value
            .get_elements()?
            .iter()
            .map(|pair| {
                Ok((
                    u16::try_from(pair.get_element(0)?.to_u64()?)?,
                    u16::try_from(pair.get_element(1)?.to_u64()?)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            values.len() == case.result_count,
            "wrong native result count"
        );
        expected.push(values);
    }
    let parameters = case
        .parameters
        .iter()
        .map(|(name, width)| format!("{name}: u{width}"))
        .collect::<Vec<_>>()
        .join(", ");
    let arguments = case
        .parameters
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let result_type = format!("({})", vec!["(u16, u16)"; case.result_count].join(", "));
    let bx = format!(
        "extern fn machine::g({parameters}) -> {result_type};\nblock sample(clk, {parameters}) -> {result_type} {{ machine::g({arguments}) }}"
    );
    let mut bundle = compile_bx_with_dslx(
        "design",
        &bx,
        checked,
        Default::default(),
        &Default::default(),
    )?
    .emit_readable(None)
    .with_context(|| format!("emit generated caller:\n{source}"))?;
    // This differential check executes the model. Correspondence proofs have
    // separate tests and can exhaust Lean's recursion budget on large literals.
    bundle.files.retain(|(name, _)| {
        matches!(
            name.as_str(),
            "Generated.lean" | "ReadableTypes.lean" | "ReadableGenerated.lean"
        )
    });
    let mut guards = String::from("import ReadableGenerated\nopen XlsReadable\n");
    for (inputs, outputs) in case.inputs.iter().zip(&expected) {
        let fields = case
            .parameters
            .iter()
            .zip(inputs)
            .map(|((name, _), value)| format!("{name} := {value}"))
            .collect::<Vec<_>>()
            .join(", ");
        let result = outputs
            .iter()
            .map(|(final_value, mid)| format!("({final_value}, {mid})"))
            .collect::<Vec<_>>()
            .join(", ");
        guards.push_str(&format!("#guard observe {{}} {{{fields}}} == ({result})\n"));
    }
    bundle.files.push(("CheckParametric.lean".into(), guards));
    Ok(CheckedCase { bundle, expected })
}
