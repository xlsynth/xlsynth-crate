// SPDX-License-Identifier: Apache-2.0

//! Public block emission contracts, including port and sequential semantics.

use std::path::{Path, PathBuf};

use xlsynth_lean::LeanModuleName;
use xlsynth_lean::block::{emit, select_block, write_to};
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_parser::Parser;

const BLOCK: &str = include_str!("fixtures/block_step.ir");

fn compile_lean(root: &Path, imports: &[&Path], source: &Path, object: &Path) {
    let lean = std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN to the Lean executable");
    let output = std::process::Command::new(lean)
        .current_dir(root)
        .env_remove("LEAN_SYSROOT")
        .env("LEAN_PATH", std::env::join_paths(imports).unwrap())
        .arg("-o")
        .arg(object)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Lean failed to compile {}:\n{}\n{}",
        source.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn compile_project(root: &Path, sources: impl IntoIterator<Item = PathBuf>) {
    // Compile the repository module once, outside the model output directory.
    let support = tempfile::tempdir().unwrap();
    let library = Path::new(env!("CARGO_MANIFEST_DIR")).join("lean");
    let object = support.path().join("XlsynthLean/IR.olean");
    std::fs::create_dir(object.parent().unwrap()).unwrap();
    compile_lean(&library, &[], &library.join("XlsynthLean/IR.lean"), &object);
    for source in sources {
        compile_lean(
            root,
            &[root, support.path()],
            &source,
            &source.with_extension("olean"),
        );
    }
}

fn check_block(package: &Package, client: &str) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let name = LeanModuleName::new("XlsBlock").unwrap();
    let block = write_to(package, package.get_top_block().unwrap(), &name, root).unwrap();
    let proof = root.join("Client.lean");
    std::fs::write(&proof, format!("import XlsBlock\n{client}")).unwrap();
    compile_project(root, [block, proof]);
}

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
    let directory = tempfile::tempdir().unwrap();
    let name = LeanModuleName::new("Swap").unwrap();
    let written = write_to(
        &package,
        package.get_top_block().unwrap(),
        &name,
        directory.path(),
    )
    .unwrap();
    assert_eq!(written, directory.path().join("Swap.lean"));
    assert_eq!(
        std::fs::read_to_string(written).unwrap(),
        golden.replace("XlsBlock", "«Swap»")
    );
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[cfg(feature = "lean-tests")]
#[test]
fn evaluates_swap_reset_and_disconnected_assertion_in_lean() {
    let package = Parser::new(BLOCK).parse_and_validate_package().unwrap();
    check_block(
        &package,
        r#"
open XlsBlock
def sample (rst en : BitVec 1) :=
  let v := eval { a := 2, b := 3 } { rst := rst, en := en }
  (v.output.out, v.next.a, v.next.b, v.assertions.checked)
example : sample 1 1 = (2, 3, 2, true) := by decide
example : sample 0 1 = (2, 1, 2, true) := by decide
example : sample 1 0 = (2, 2, 2, false) := by decide
"#,
    );
}

#[cfg(feature = "lean-tests")]
#[test]
fn lowering_carry_preserves_source_identity_and_port_order() {
    let package = Parser::new(
        r#"package adder
file_number 0 "adder.x"
top block adder(rhs: bits[4], cin: bits[1], lhs: bits[4], copied: bits[4], cout: bits[1]) {
  lhs: bits[4] = input_port(name=lhs, id=1)
  rhs: bits[4] = input_port(name=rhs, id=2)
  cin: bits[1] = input_port(name=cin, id=3)
  carry: bits[1] = ext_carry_out(lhs, rhs, cin, id=40, pos=[(0,2,3)])
  cout: () = output_port(carry, name=cout, id=41)
  copied: () = output_port(rhs, name=copied, id=42)
}"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let source = emit(&package, package.get_top_block().unwrap()).unwrap();
    assert!(
        source
            .lines()
            .any(|line| line.contains("node carry [id=40,") && line.contains("adder.x:3:4")),
        "missing source identity for the lowered carry:\n{source}"
    );

    // Positional construction and matching use the port list in the block
    // header. Both inputs and outputs appear in a different order in the body.
    check_block(
        &package,
        r#"
open XlsBlock
def outputs (cin : BitVec 1) :=
  let ⟨copied, cout⟩ := observe {} (Input.Constructors.mk 3 cin 12)
  (copied, cout)
example : outputs 0 = (3, 0) := by decide
example : outputs 1 = (3, 1) := by decide
"#,
    );
}

#[cfg(feature = "lean-tests")]
#[test]
fn indexes_array_literal_of_tuples_with_zero_width_field() {
    let package = Parser::new(
        r#"package lookup
top block lookup(i: bits[2], y: (bits[0], bits[4])) {
  i: bits[2] = input_port(name=i, id=1)
  values: (bits[0], bits[4])[2] = literal(value=[(0, 2), (0, 9)], id=2)
  selected: (bits[0], bits[4]) = array_index(values, indices=[i], id=3)
  y: () = output_port(selected, name=y, id=4)
}"#,
    )
    .parse_and_validate_package()
    .unwrap();
    check_block(
        &package,
        r#"
open XlsBlock
example : (observe {} { i := 0 }).y = (0#0, 2#4, ()) := by decide
example : (observe {} { i := 3 }).y = (0#0, 9#4, ()) := by decide
"#,
    );
}

#[cfg(feature = "lean-tests")]
#[test]
fn shifts_and_gates_use_qualified_library_names() {
    let package = Parser::new(
        r#"package offsets
top block offsets(amount: bits[64], enable: bits[1], result: (bits[5], bits[3])) {
  amount: bits[64] = input_port(name=amount, id=1)
  enable: bits[1] = input_port(name=enable, id=2)
  input: bits[5] = literal(value=13, id=3)
  shifted: bits[5] = shll(input, amount, id=4)
  slice: bits[3] = dynamic_bit_slice(input, amount, width=3, id=5)
  pair: (bits[5], bits[3]) = tuple(shifted, slice, id=6)
  gated: (bits[5], bits[3]) = gate(enable, pair, id=7)
  result: () = output_port(gated, name=result, id=8)
}"#,
    )
    .parse_and_validate_package()
    .unwrap();
    check_block(
        &package,
        r#"
open XlsBlock
#guard (observe {} { amount := 1, enable := 1 }).result == (26#5, 6#3, ())
#guard (observe {} { amount := 1, enable := 0 }).result == (0#5, 0#3, ())
#guard (observe {} { amount := 1152921504606846976, enable := 1 }).result == (0#5, 0#3, ())
"#,
    );
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
    check_block(
        &package,
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
}

#[cfg(feature = "lean-tests")]
#[test]
fn tuple_one_hot_select_agrees_with_pir_for_functions_and_blocks() {
    use xlsynth_lean::export_ir_function;
    use xlsynth_pir::{
        IrValue,
        block_eval::{BlockEvaluator, RegisterInit},
        ir_eval::eval_fn_in_package_value,
    };

    let package = Parser::new(
        r#"package selection
fn select_pair(s: bits[2] id=1, a: (bits[4], bits[4]) id=2, b: (bits[4], bits[4]) id=3) -> (bits[4], bits[4]) {
  ret chosen: (bits[4], bits[4]) = one_hot_sel(s, cases=[a, b], id=4)
}
top block select_ports(s: bits[2], a: (bits[4], bits[4]), b: (bits[4], bits[4]), y: (bits[4], bits[4])) {
  s: bits[2] = input_port(name=s, id=5)
  a: (bits[4], bits[4]) = input_port(name=a, id=6)
  b: (bits[4], bits[4]) = input_port(name=b, id=7)
  chosen: (bits[4], bits[4]) = one_hot_sel(s, cases=[a, b], id=8)
  y: () = output_port(chosen, name=y, id=9)
}
block swap_cases(s: bits[2], a: (bits[4], bits[4]), b: (bits[4], bits[4]), y: (bits[4], bits[4])) {
  s: bits[2] = input_port(name=s, id=10)
  a: (bits[4], bits[4]) = input_port(name=a, id=11)
  b: (bits[4], bits[4]) = input_port(name=b, id=12)
  chosen: (bits[4], bits[4]) = one_hot_sel(s, cases=[b, a], id=13)
  y: () = output_port(chosen, name=y, id=14)
}"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let args = [
        "bits[2]:3",
        "(bits[4]:1, bits[4]:2)",
        "(bits[4]:4, bits[4]:8)",
    ]
    .map(|s| IrValue::parse_typed(s).unwrap())
    .to_vec();
    // Both selector bits are set, so the result ORs the cases field by field.
    let expected = IrValue::parse_typed("(bits[4]:5, bits[4]:10)").unwrap();
    let pir_function = package.get_fn("select_pair").unwrap();
    assert_eq!(
        eval_fn_in_package_value(&package, pir_function, &args),
        expected
    );
    let pir_block = BlockEvaluator::prepare(&package, "select_ports").unwrap();
    let state = pir_block.initial_state(RegisterInit::AllZeros).unwrap();
    assert_eq!(
        pir_block.step(&state, &args).unwrap().outputs.as_slice(),
        std::slice::from_ref(&expected)
    );

    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    for (block, module) in [
        ("select_ports", "Generated.Select"),
        ("swap_cases", "Generated.Swap"),
    ] {
        write_to(
            &package,
            package.get_block(block).unwrap(),
            &LeanModuleName::new(module).unwrap(),
            root,
        )
        .unwrap();
    }
    let module = LeanModuleName::new("Generated.Selection").unwrap();
    let function = export_ir_function(&package, Some("select_pair"), &module).unwrap();
    let function = function.write_to(root).unwrap();
    let client = root.join("Client.lean");
    std::fs::write(
        &client,
        r#"import XlsynthLean.IR
import Generated.Selection
import Generated.Select
import Generated.Swap

abbrev Pair4 := BitVec 4 × BitVec 4 × Unit

-- A proof can use library operations absent from every generated model.
example : XlsynthLean.IR.arrayIndex (#[2#4, 9#4].toVector) 3 = 9#4 := by decide

def selected (s : BitVec 2) (a b : Pair4) := XlsynthLean.IR.oneHotSel s #[a, b]
example (s : BitVec 2) (a b : Pair4) :
    Generated.Selection.eval s a b = selected s a b := rfl
example (s : BitVec 2) (a b : Pair4) :
    (Generated.Select.observe {} ⟨s, a, b⟩).y = selected s a b := rfl
example (s : BitVec 2) (a b : Pair4) :
    (Generated.Swap.observe {} ⟨s, a, b⟩).y = selected s b a := rfl
example : selected 3 (1, 2, ()) (4, 8, ()) = (5, 10, ()) := by decide
"#,
    )
    .unwrap();
    compile_project(
        root,
        [
            function,
            root.join("Generated/Select.lean"),
            root.join("Generated/Swap.lean"),
            client,
        ],
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
    let unsupported = BLOCK
        .replace(
            "package swap",
            "package swap\n\nfile_number 0 \"counter.x\"",
        )
        .replace(
            "  out: ()",
            "  hit: () = cover(en, label=\"hit\", id=10, pos=[(0,1,2)])\n  out: ()",
        );
    let unsupported = Parser::new(&unsupported)
        .parse_and_validate_package()
        .unwrap();
    let error = emit(&unsupported, unsupported.get_top_block().unwrap()).unwrap_err();
    assert!(format!("{error:#}").contains("unsupported Lean operation cover"));
    assert!(format!("{error:#}").contains("counter.x:2:3"));
}
