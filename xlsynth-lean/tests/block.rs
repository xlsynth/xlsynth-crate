// SPDX-License-Identifier: Apache-2.0

//! Public block emission contracts, including port and sequential semantics.

use xlsynth_lean::block::{emit, select_block};
use xlsynth_pir::ir_parser::Parser;

const BLOCK: &str = include_str!("fixtures/block_step.ir");

// Verifies: Raw emission preserves old outputs, simultaneous writes, and
// assertions. Catches: Port adaptation that changes clock-step semantics or
// drops assertions.
#[test]
fn raw_step_preserves_old_outputs_simultaneous_writes_and_disconnected_assertions() {
    let package = Parser::new(BLOCK).parse_and_validate_package().unwrap();
    let source = emit(&package, package.get_top_block().unwrap()).unwrap();
    // The golden reads output/a and next/b from old_a, next/a from old_b,
    // gives reset precedence over enable, and retains the unused assertion.
    // The checked-in fixture's license is metadata, not generated user text.
    let golden = include_str!("fixtures/block_step.lean")
        .strip_prefix("-- SPDX-License-Identifier: Apache-2.0\n\n")
        .unwrap();
    pretty_assertions::assert_eq!(source, golden);
}

// Each field becomes a projection in the Lean structure's namespace.
// Report the owning field when Lean would generate the same declaration.
#[test]
fn raw_emission_rejects_field_names_used_by_lean() {
    let check = |ir: String, owner: &str, field: &str| {
        let package = Parser::new(&ir).parse_and_validate_package().unwrap();
        let error = emit(&package, package.get_top_block().unwrap()).unwrap_err();
        assert_eq!(
            error.root_cause().to_string(),
            format!("cannot emit {owner}.{field}: Lean generates a declaration with this name")
        );
    };
    for field in [
        "rec",
        "recOn",
        "casesOn",
        "noConfusion",
        "noConfusionType",
        "ctorIdx",
        "_sizeOf_1",
        "_sizeOf_inst",
    ] {
        let ir = format!(
            "package ports\n\ntop block ports({field}: bits[1], out: bits[1]) {{
  {field}: bits[1] = input_port(name={field}, id=1)
  out: () = output_port({field}, name=out, id=2)
}}"
        );
        check(ir, "Input", field);
    }
    check(
        BLOCK
            .replace("reg b(", "reg rec(")
            .replace("register=b", "register=rec"),
        "State",
        "rec",
    );
    check(
        BLOCK
            .replace("out:", "rec:")
            .replace("name=out", "name=rec"),
        "Output",
        "rec",
    );
    check(BLOCK.replace("checked:", "rec:"), "Assertions", "rec");
}

// A port named mk must remain distinct from the constructor and from ports
// with trailing underscores when proofs construct and observe an input.
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn qualified_constructors_preserve_port_projections() {
    let package = Parser::new(
        r#"package ports

top block ports(mk: bits[1], mk_: bits[1], mk__: bits[1], rec_: bits[1], result: bits[1]) {
  mk: bits[1] = input_port(name=mk, id=1)
  mk_: bits[1] = input_port(name=mk_, id=2)
  mk__: bits[1] = input_port(name=mk__, id=3)
  rec_: bits[1] = input_port(name=rec_, id=4)
  result: () = output_port(mk_, name=result, id=5)
}"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let mut source = emit(&package, package.get_top_block().unwrap()).unwrap();
    source.push_str(
        r#"
open XlsBlock
example (a b c d : BitVec 1) :
    let x := Input.Constructors.mk a b c d
    (x.mk, x.mk_, x.mk__, x.rec_) = (a, b, c, d) := rfl
example (q : State) (x : Input) : (observe q x).result = x.mk_ := rfl
#check State.Constructors.mk
#check Output.Constructors.mk
#check Assertions.Constructors.mk
#check Evaluation.Constructors.mk
"#,
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("Ports.lean");
    std::fs::write(&path, source).unwrap();
    let lean = std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN to the Lean executable");
    let output = std::process::Command::new(lean)
        .current_dir(directory.path())
        .env_remove("LEAN_SYSROOT")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Lean failed to check the emitted port projections:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// Verifies: Named, top, and sole-block selection, including invalid choices.
#[test]
fn selects_top_or_explicit_block_and_reports_ambiguity() {
    let another = BLOCK
        .split_once("top block")
        .unwrap()
        .1
        .replace("swap(", "other(")
        .replace("id=", "id=10");
    let multiple = format!("{BLOCK}\nblock{another}");
    let with_top = Parser::new(&multiple).parse_and_validate_package().unwrap();
    assert_eq!(select_block(&with_top, None).unwrap().name, "swap");
    assert_eq!(
        select_block(&with_top, Some("other")).unwrap().name,
        "other"
    );
    assert!(select_block(&with_top, Some("missing")).is_err());
    let ambiguous = Parser::new(&multiple.replace("top block", "block"))
        .parse_and_validate_package()
        .unwrap();
    assert!(select_block(&ambiguous, None).is_err());
    let sole = Parser::new(&BLOCK.replace("top block", "block"))
        .parse_and_validate_package()
        .unwrap();
    assert_eq!(select_block(&sole, None).unwrap().name, "swap");
}

// A same-named block from a different package would use the wrong file table.
#[test]
fn emission_rejects_block_from_another_package() {
    let original = Parser::new(BLOCK).parse_and_validate_package().unwrap();
    let copy = original.clone();
    let error = emit(&original, copy.get_top_block().unwrap()).unwrap_err();
    assert!(error.to_string().contains("does not belong to package"));
}

// Negative test: Unsupported resets and operations return contextual emission
// errors.
#[test]
fn emission_rejects_asynchronous_reset_and_unsupported_operations() {
    let asynchronous = Parser::new(&BLOCK.replace("asynchronous=false", "asynchronous=true"))
        .parse_and_validate_package()
        .unwrap();
    let error = emit(&asynchronous, asynchronous.get_top_block().unwrap()).unwrap_err();
    assert!(format!("{error:#}").contains("asynchronous reset"));
    let unsupported = BLOCK.replace("package swap", "package swap\n\nfile_number 0 \"counter.x\"").replace(
        "  out: () = output_port(old_a,",
        "  reversed: bits[2] = reverse(old_a, id=10, pos=[(0,1,2)])\n  out: () = output_port(reversed,",
    );
    let unsupported = Parser::new(&unsupported)
        .parse_and_validate_package()
        .unwrap();
    let error = emit(&unsupported, unsupported.get_top_block().unwrap()).unwrap_err();
    assert!(format!("{error:#}").contains("unsupported Lean operation reverse"));
    assert!(format!("{error:#}").contains("counter.x:2:3"));
}
