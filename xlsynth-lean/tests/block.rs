// SPDX-License-Identifier: Apache-2.0

//! Public block emission contracts, including port and sequential semantics.

use xlsynth_lean::block::{emit, select_block};
use xlsynth_pir::{ir::Package, ir_parser::Parser};

const BLOCK: &str = include_str!("fixtures/block_step.ir");

fn package(text: &str) -> Package {
    Parser::new(text).parse_and_validate_package().unwrap()
}

// Verifies: Each indexed dimension contributes one Lean vector operation.
// Catches: Repeating earlier array reads in a fallback or nested update.
#[test]
fn multidimensional_array_operations_emit_each_level_once() {
    let rank = 10;
    let array_type = format!("bits[8]{}", "[1]".repeat(rank));
    let indices = vec!["i"; rank].join(", ");
    let package = package(&format!(
        "package arrays
top block arrays(a: {array_type}, i: bits[1], v: bits[8], read: bits[8], written: {array_type}) {{
  a: {array_type} = input_port(name=a, id=1)
  i: bits[1] = input_port(name=i, id=2)
  v: bits[8] = input_port(name=v, id=3)
  element: bits[8] = array_index(a, indices=[{indices}], id=4)
  updated: {array_type} = array_update(a, v, indices=[{indices}], id=5)
  read: () = output_port(element, name=read, id=6)
  written: () = output_port(updated, name=written, id=7)
}}"
    ));
    let source = emit(&package, package.get_top_block().unwrap()).unwrap();
    let read = source
        .lines()
        .find(|line| line.contains("Vector.getD") && !line.contains("Vector.setIfInBounds"))
        .unwrap();
    let write = source
        .lines()
        .find(|line| line.contains("Vector.setIfInBounds"))
        .unwrap();
    assert_eq!(read.matches("Vector.getD").count(), rank);
    assert_eq!(write.matches("Vector.getD").count(), rank - 1);
    assert_eq!(write.matches("Vector.setIfInBounds").count(), rank);
}

// Verifies: Raw emission preserves old outputs, simultaneous writes, and
// assertions. Catches: Port adaptation that changes clock-step semantics or
// drops assertions.
#[test]
fn raw_step_preserves_old_outputs_simultaneous_writes_and_disconnected_assertions() {
    let package = package(BLOCK);
    let source = emit(&package, package.get_top_block().unwrap()).unwrap();
    // The golden reads output/a and next/b from old_a, next/a from old_b,
    // gives reset precedence over enable, and retains the unused assertion.
    // The checked-in fixture's license is metadata, not generated user text.
    let golden = include_str!("fixtures/block_step.lean")
        .strip_prefix("-- SPDX-License-Identifier: Apache-2.0\n\n")
        .unwrap();
    pretty_assertions::assert_eq!(source, golden);
}

// All four physical records can have fields whose names Lean also generates.
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn raw_projections_keep_lean_generated_member_names() {
    let package = package(
        r#"package ports
top block ports(clk: clock, recOn: bits[1], casesOn: bits[1], rec: bits[1]) {
  #![reset(port="recOn", asynchronous=false, active_low=false)]
  reg noConfusion(bits[1], reset_value=1)
  reg b(bits[1])
  recOn: bits[1] = input_port(name=recOn, id=1)
  casesOn: bits[1] = input_port(name=casesOn, id=2)
  current: bits[1] = register_read(register=noConfusion, id=3)
  old_b: bits[1] = register_read(register=b, id=9)
  tok: token = literal(value=token, id=4)
  ctorIdx: token = assert(tok, casesOn, message="enabled", label="enabled", id=5)
  set_current: () = register_write(casesOn, register=noConfusion, reset=recOn, id=6)
  save_old: () = register_write(current, register=b, id=7)
  rec: () = output_port(current, name=rec, id=8)
}"#,
    );
    let mut source = emit(&package, package.get_top_block().unwrap()).unwrap();
    source.push_str(
        r#"
open XlsBlock
example (rst en : BitVec 1) :
    let x : Input := ⟨rst, en⟩
    (x.recOn, x.casesOn) = (rst, en) := rfl
example (q : State) (x : Input) : (observe q x).rec = q.noConfusion := rfl
example (q : State) (x : Input) : (step q x).b = q.noConfusion := rfl
example (q : State) (x : Input) : (eval q x).assertions.ctorIdx = (x.casesOn == 1) := rfl
example (q : State) (x : Input) : (reset q x).next.noConfusion = 1 := rfl
"#,
    );
    check_lean(&source);
}

// A port named mk must remain distinct from the constructor and from ports
// with trailing underscores when proofs construct and observe an input.
#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn qualified_constructors_preserve_port_projections() {
    let package = package(
        r#"package ports

top block ports(mk: bits[1], mk_: bits[1], mk__: bits[1], rec_: bits[1], result: bits[1]) {
  mk: bits[1] = input_port(name=mk, id=1)
  mk_: bits[1] = input_port(name=mk_, id=2)
  mk__: bits[1] = input_port(name=mk__, id=3)
  rec_: bits[1] = input_port(name=rec_, id=4)
  result: () = output_port(mk_, name=result, id=5)
}"#,
    );
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
    check_lean(&source);
}

fn check_lean(source: &str) {
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
    let with_top = package(&multiple);
    assert_eq!(select_block(&with_top, None).unwrap().name, "swap");
    assert_eq!(
        select_block(&with_top, Some("other")).unwrap().name,
        "other"
    );
    assert!(select_block(&with_top, Some("missing")).is_err());
    let ambiguous = package(&multiple.replace("top block", "block"));
    assert!(select_block(&ambiguous, None).is_err());
    let sole = package(&BLOCK.replace("top block", "block"));
    assert_eq!(select_block(&sole, None).unwrap().name, "swap");
}

// A same-named block from a different package would use the wrong file table.
#[test]
fn emission_rejects_block_from_another_package() {
    let original = package(BLOCK);
    let copy = original.clone();
    let error = emit(&original, copy.get_top_block().unwrap()).unwrap_err();
    assert!(error.to_string().contains("does not belong to package"));
}

// Negative test: Unsupported resets and operations return contextual emission
// errors.
#[test]
fn emission_rejects_asynchronous_reset_and_unsupported_operations() {
    let asynchronous = package(&BLOCK.replace("asynchronous=false", "asynchronous=true"));
    let error = emit(&asynchronous, asynchronous.get_top_block().unwrap()).unwrap_err();
    assert!(format!("{error:#}").contains("asynchronous reset"));
    let unsupported = BLOCK.replace("package swap", "package swap\n\nfile_number 0 \"counter.x\"").replace(
        "  out: () = output_port(old_a,",
        "  reversed: bits[2] = reverse(old_a, id=10, pos=[(0,1,2)])\n  out: () = output_port(reversed,",
    );
    let unsupported = package(&unsupported);
    let error = emit(&unsupported, unsupported.get_top_block().unwrap()).unwrap_err();
    assert!(format!("{error:#}").contains("unsupported Lean operation reverse"));
    assert!(format!("{error:#}").contains("counter.x:2:3"));
}
