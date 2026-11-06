// SPDX-License-Identifier: Apache-2.0

use xlsynth::dslx::{ImportData, parse_and_typecheck};
use xlsynth_lean::block::compile_bx_with_dslx;

#[test]
#[ignore = "requires a matching native XLS library and Lean 4.34.1"]
fn same_layout_extern_uses_the_selected_modules_declarations() {
    let directory = tempfile::tempdir().unwrap();
    for (module, fields) in [
        ("left_side", "tag: u8, data: u8"),
        ("right_side", "data: u8, tag: u8"),
    ] {
        let source = format!(
            "pub struct Pair {{ {fields} }}
             pub type Request = Pair;
             pub fn step(x: Request) -> u8 {{ x.tag }}"
        );
        std::fs::write(directory.path().join(format!("{module}.x")), source).unwrap();
    }
    let mut imports = ImportData::new(None, &[directory.path()]);
    let checked = parse_and_typecheck(
        "import left_side;
         import right_side;
         fn both(l: left_side::Request, r: right_side::Request) -> (u8, u8) {
             (left_side::step(l), right_side::step(r))
         }",
        "entry.x",
        "entry",
        &mut imports,
    )
    .unwrap();

    // Both native functions exist in the linked package and accept the same
    // pair of bytes. Change only the BX callee to select its source signature.
    for (owner, other, expected) in [
        ("left_side", "right_side", "tag"),
        ("right_side", "left_side", "data"),
    ] {
        let bx = format!(
            "struct Pair {{ tag: u8, data: u8 }}
             extern fn {owner}::step(x: Pair) -> u8;
             block circuit(clk, x: Pair) -> u8 {{ {owner}::step(x) }}"
        );
        let compiled = compile_bx_with_dslx(
            "design",
            &bx,
            checked.clone(),
            Default::default(),
            &Default::default(),
        )
        .unwrap();
        let mut bundle = compiled.emit_readable(Some("circuit")).unwrap();
        let (_, executable) = bundle
            .files
            .iter()
            .find(|(name, _)| name == "ReadableGenerated.lean")
            .unwrap();
        let signature = executable
            .lines()
            .find(|line| line.starts_with(&format!("def Helpers.«{owner}::step»")))
            .unwrap();
        assert!(
            signature.contains(&format!("«{owner}».«Request»")),
            "{signature}"
        );
        assert!(!executable.contains(&format!("Helpers.«{other}::step»")));
        bundle.files.push(("Selected.lean".into(), format!(
            "import ReadableCorrespondence\nopen XlsReadable\nexample (q : State) (input : Input) : observe q input = input.x.{expected} := rfl\n"
        )));
        super::check_bundle(bundle, owner);
    }
}
