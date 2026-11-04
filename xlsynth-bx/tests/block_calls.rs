// SPDX-License-Identifier: Apache-2.0

use xlsynth_bx::{ResetBehavior, check_module, compile, empty_package, parse_module};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn two_call_bindings_create_the_same_hardware_as_two_explicit_instances() {
    let child = r#"
block delay(clk, rst, x: u8, en: bool) -> u8 {
    reg q: u8 = 0;
    if en { q := x; }
    q
}"#;
    let signature = "pub block lanes(clk, rst, x: u8, en: bool) -> (u8, u8)";
    let calls = r#"{
    let a = delay(x, en);
    let b = delay(x + u8:1, !en);
    (a, b)
}"#;
    let explicit = r#"{
    inst a: delay;
    inst b: delay;
    a.x := x;
    a.en := en;
    b.x := x + u8:1;
    b.en := !en;
    (a.out, b.out)
}"#;
    let actual = compile(
        &format!("{signature} {calls}\n{child}"),
        empty_package("design"),
        ResetBehavior::default(),
    )
    .unwrap();
    let expected = compile(
        &format!("{signature} {explicit}\n{child}"),
        empty_package("design"),
        ResetBehavior::default(),
    )
    .unwrap();
    assert_eq!(actual.to_string(), expected.to_string());
    let parent = actual.get_block("lanes").unwrap();
    assert_eq!(
        parent
            .instantiations
            .iter()
            .map(|i| (i.name.as_str(), i.block.as_str()))
            .collect::<Vec<_>>(),
        [("a", "delay"), ("b", "delay")]
    );
    xlsynth::IrPackage::parse_ir(&actual.to_string(), None).unwrap();
    emit_system_verilog(&actual, &BlockCodegenOptions::default()).unwrap();
}

#[test]
fn block_call_results_retain_their_declared_shapes() {
    let source = r#"
struct Bundle { data: u8, flags: (bool, bool) }
pub block parent(clk, x: u8) -> Bundle {
    let pair: (u8, bool) = split(x);
    let array = samples(pair.0);
    let empty: () = sink(array[0]);
    let outputs = bundle(array[1], pair.1);
    Bundle { data: outputs.data, flags: outputs.flags }
}
block split(clk, x: u8) -> (u8, bool) { (x, x == u8:0) }
block samples(clk, x: u8) -> u8[2] { [x, x + u8:1] }
block sink(clk, x: u8) -> () { reg saved: u8; saved := x; () }
block bundle(clk, x: u8, flag: bool) -> Bundle { Bundle { data: x, flags: (flag, !flag) } }
"#;
    let pkg = compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    let parent = pkg.get_block("parent").unwrap();
    assert_eq!(parent.instantiations.len(), 4);
    xlsynth::IrPackage::parse_ir(&pkg.to_string(), None).unwrap();
}

// Data-port order differs from the BX declaration. Clock and reset use their
// physical IR names, while BX identifies them by role.
const HISTORY: &str = r#"package imported
block history(en: bits[1], previous: bits[8], fast_clock: clock, din: bits[8], clear: bits[1], observed: bits[8]) {
  #![reset(port="clear", asynchronous=false, active_low=false)]
  reg saved(bits[8], reset_value=0)
  en: bits[1] = input_port(name=en, id=1)
  din: bits[8] = input_port(name=din, id=2)
  clear: bits[1] = input_port(name=clear, id=3)
  register_read.4: bits[8] = register_read(register=saved, id=4)
  register_write.5: () = register_write(din, register=saved, load_enable=en, reset=clear, id=5)
  observed: () = output_port(din, name=observed, id=6)
  previous: () = output_port(register_read.4, name=previous, id=7)
}"#;

const IMPORTED_CALL: &str = r#"
struct Samples { observed: u8, previous: u8 }
extern block history(clk, rst, din: u8, en: bool) -> Samples;
block parent(clk, rst, x: u8, valid: bool) -> u8 {
    let outputs = history(x, valid);
    outputs.previous
}"#;

#[test]
fn imported_block_checks_by_name_and_wires_real_clock_and_reset_ports() {
    check_module(&parse_module(IMPORTED_CALL).unwrap()).unwrap();
    for active_low in [false, true] {
        let source = HISTORY.replace("active_low=false", &format!("active_low={active_low}"));
        let ir = Parser::new(&source).parse_and_validate_package().unwrap();
        let pkg = compile(
            IMPORTED_CALL,
            ir,
            ResetBehavior {
                active_low,
                asynchronous: false,
            },
        )
        .unwrap();
        let parent = pkg.get_block("parent").unwrap();
        assert_eq!(parent.instantiations[0].block, "history");
        let mut connections = parent
            .nodes
            .iter()
            .filter_map(|node| match &node.payload {
                NodePayload::InstantiationInput {
                    instantiation,
                    port_name,
                    arg,
                } => Some((
                    instantiation.as_str(),
                    port_name.as_str(),
                    parent.port_name(*arg),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        connections.sort_unstable();
        assert_eq!(
            connections,
            [
                ("outputs", "clear", "rst"),
                ("outputs", "din", "x"),
                ("outputs", "en", "valid")
            ]
        );
        xlsynth::IrPackage::parse_ir(&pkg.to_string(), None).unwrap();
        emit_system_verilog(&pkg, &BlockCodegenOptions::default()).unwrap();
    }
}

#[test]
fn linking_rejects_incompatible_block_interfaces_and_reset_policies() {
    let ir = Parser::new(HISTORY).parse_and_validate_package().unwrap();
    for source in [
        IMPORTED_CALL.replace("history", "unknown"),
        IMPORTED_CALL.replace("din: u8", "data: u8"),
        IMPORTED_CALL.replace("u8", "u16"),
        IMPORTED_CALL.replace("observed: u8, ", ""),
        IMPORTED_CALL.replace("history(clk, rst,", "history(clk,"),
    ] {
        check_module(&parse_module(&source).unwrap()).unwrap();
        assert!(
            compile(&source, ir.clone(), ResetBehavior::default()).is_err(),
            "{source}"
        );
    }
    for reset in [
        ResetBehavior {
            asynchronous: false,
            active_low: true,
        },
        ResetBehavior {
            asynchronous: true,
            active_low: false,
        },
    ] {
        assert!(compile(IMPORTED_CALL, ir.clone(), reset).is_err());
    }
}

#[test]
fn block_calls_require_an_unconditional_binding_and_an_acyclic_hierarchy() {
    let delay = "block delay(clk, rst, d: u8) -> u8 { reg q: u8 = 0; q := d; q }";
    for parent in [
        "block top(clk, rst, x: u8) -> u8 { if true { let q = delay(x); } x }",
        "block top(clk, rst, x: u8) -> u8 { delay(x) }",
        "block top(clk, rst, x: u8) -> u8 { let q = delay(delay(x)); q }",
        "block top(clk, rst, x: u8) -> u8 { let q = delay(x, x); q }",
        "block top(clk, rst, x: u8) -> u8 { let q: u16 = delay(x); x }",
        "block top(clk, x: u8) -> u8 { let q = delay(x); q }",
        "block top(clk, x: u8) -> u8 { let q = top(x); q }",
        "block top(clk, rst, x: u8) -> u8 { let q = delay(x); q.d := x; q }",
    ] {
        let parsed = parse_module(&format!("{parent}\n{delay}")).unwrap();
        assert!(check_module(&parsed).is_err(), "{parent}");
    }
}
