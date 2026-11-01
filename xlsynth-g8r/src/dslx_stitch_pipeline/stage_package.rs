// SPDX-License-Identifier: Apache-2.0

//! Extracts the XLS functions that implement a stitched pipeline.

use std::collections::HashSet;

use xlsynth::ir_package::IrPackage;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils::external_function_references;

/// A depth-first traversal emits each declaration after all its callees.
enum Visit {
    Load(String),
    Emit(ir::Fn),
}

/// Extracts the selected functions and their transitive callees, retaining
/// their original XLS declarations and source locations.
pub(crate) fn load_stage_package(
    source: &IrPackage,
    stages: &[(String, String)],
) -> Result<ir::Package, String> {
    // libxls prints its file_number declarations before all package members.
    // The package can also contain unrelated procs which PIR cannot parse.
    let mut package = Parser::new(&source.to_string())
        .parse_xls_package_header()
        .map_err(|e| format!("could not read XLS package header: {e}"))?;
    let mut visited = HashSet::new();
    for (stage, root) in stages {
        let mut pending = vec![Visit::Load(root.clone())];
        while let Some(visit) = pending.pop() {
            match visit {
                Visit::Load(name) => {
                    if !visited.insert(name.clone()) {
                        continue;
                    }
                    let declaration = source
                        .get_function(&name)
                        .and_then(|f| f.to_ir_string())
                        .map_err(|e| {
                            format!("stage '{stage}': could not read XLS function '{name}': {e}")
                        })?;
                    // Preserve FFI attributes and their behavioral bodies
                    // without replacing calls with native PIR extension ops.
                    let function = Parser::new(&declaration)
                        .parse_function_declaration()
                        .map_err(|e| {
                            format!("stage '{stage}': could not read XLS function '{name}': {e}")
                        })?;
                    let callees = external_function_references(&function);
                    pending.push(Visit::Emit(function));
                    pending.extend(callees.into_iter().rev().map(Visit::Load));
                }
                Visit::Emit(function) => {
                    package.members.push(ir::PackageMember::Function(function))
                }
            }
        }
    }
    Ok(package)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_stages_round_trip_through_xls_with_ffi_and_source_positions() {
        let expected = r#"package original

file_number 1 "pipeline.x"
file_number 2 "helper.x"

#[ffi_proto("""code_template: "external {fn} (.x({x}), .result({return}));"
""")]
fn helper(x: bits[8] id=1) -> bits[8] {
  ret y: bits[8] = not(x, id=2, pos=[(2,4,7)])
}

fn middle(x: bits[8] id=3) -> bits[8] {
  ret y: bits[8] = invoke(x, to_apply=helper, id=4)
}

fn pipeline_stage(x: bits[8] id=5) -> bits[8] {
  ret y: bits[8] = invoke(x, to_apply=middle, id=6, pos=[(1,12,3)])
}
"#;
        let input = format!(
            "{expected}\ntop fn unused(x: bits[8] id=7) -> bits[8] {{ ret y: bits[8] = not(x, id=8) }}"
        );
        let source = IrPackage::parse_ir(&input, None).unwrap();
        // middle is both an explicit root and a callee of pipeline_stage.
        let stages = [
            ("pipeline_stage".into(), "pipeline_stage".into()),
            ("middle".into(), "middle".into()),
        ];
        let package = load_stage_package(&source, &stages).unwrap();
        let actual = IrPackage::parse_ir(&package.to_string(), None).unwrap();
        let expected = IrPackage::parse_ir(expected, None).unwrap();
        assert_eq!(actual.to_string(), expected.to_string());
    }
}
