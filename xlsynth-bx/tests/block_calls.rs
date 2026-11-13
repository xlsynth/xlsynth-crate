// SPDX-License-Identifier: Apache-2.0

use xlsynth_bx::{ResetBehavior, check_module, compile, empty_package, parse_module};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::block_invoke_inline::inline_block_invokes_in_package;
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn nested_let_patterns_require_matching_tuple_members() -> Result<(), String> {
    for (ty, pattern, result) in [
        ("((u8,), u8)", "((a), b)", "a + b"),
        ("(u8, u8)", "(a!, b')", "a! + b'"),
        ("((), u8)", "((), a)", "a"),
        ("(u8, u8)", "(a, .., b)", "a + b"),
    ] {
        let source = format!("block sample(clk, x: {ty}) -> u8 {{ let {pattern} = x; {result} }}");
        check_module(&parse_module(&source)?)?;
    }
    for pattern in ["((a), b)", "((..), b)", "((), b)"] {
        let source = format!("block sample(clk, x: (u8, u8)) -> u8 {{ let {pattern} = x; b }}");
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(
            error.contains("tuple pattern requires a tuple"),
            "{pattern}: {error}"
        );
    }
    let nominal =
        "struct Pair { a: u8, b: u8 } block sample(clk, x: Pair) -> u8 { let (a,b) = x; a+b }";
    let error = check_module(&parse_module(nominal)?).unwrap_err();
    assert!(error.contains("tuple pattern requires a tuple"), "{error}");
    Ok(())
}

#[test]
fn destructuring_a_clocked_callback_block_reads_one_instance() {
    let source = r#"
fn widen(x: u8) -> u16 { x as u16 }
block stage<In, Out, F: fn(In) -> Out>(clk, en: bool, x: In) -> (bool, (In, Out)) {
    reg valid: bool = false;
    reg data: Out = zero!<Out>();
    valid := en;
    if en { data := F(x); }
    (valid, (x, data))
}
pub block pipeline(clk, en: bool, x: u8) -> (bool, u16) {
    let (valid, (.., data)): (bool, (u8, u16)) = stage<widen>(en, x);
    (valid, data)
}
"#;
    let mut package = compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    let parent = package.get_block("pipeline").unwrap();
    assert_eq!(parent.instantiations.len(), 1);
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
    inline_all_blocks_in_package(&mut package).unwrap();
    let package = inline_block_invokes_in_package(package).unwrap();
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap();
    let module = compile_pipeline_module(&rtl.system_verilog).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let reset = ports.input("rst").unwrap();
    let enable = ports.input("en").unwrap();
    let input = ports.input("x").unwrap();
    for (rst, en, x, expected_valid, expected_data) in [
        (true, false, 0, 0, 0),
        (false, true, 37, 1, 37),
        (false, false, 99, 0, 37),
        (false, true, 255, 1, 255),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_bool(&reset, rst)?;
                drive.set_bool(&enable, en)?;
                drive.set_u64(&input, x)
            })
            .unwrap();
        assert_eq!(
            edge.outputs["out_0"].to_u64_if_known(),
            Some(expected_valid)
        );
        assert_eq!(edge.outputs["out_1"].to_u64_if_known(), Some(expected_data));
    }
}

#[test]
fn two_call_bindings_create_two_named_instances() {
    let source = r#"
pub block lanes(clk, x: u8, en: bool) -> (u8, u8) {
    let a = delay(x, en);
    let b = delay(x + u8:1, !en);
    (a, b)
}
block delay(clk, x: u8, en: bool) -> u8 {
    reg q: u8 = 0;
    if en { q := x; }
    q
}"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    let parent = package.get_block("lanes").unwrap();
    assert_eq!(
        parent
            .instantiations
            .iter()
            .map(|i| (i.name.as_str(), i.block.as_str()))
            .collect::<Vec<_>>(),
        [("a", "delay"), ("b", "delay")]
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
    emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap();
}

#[test]
fn unrolled_and_explicit_delay_arrays_agree_across_cycles() {
    let source = r#"
pub block delay_array(clk, data: u8[4], enable: bool[4]) -> u8[4] {
    unroll_for! (i, values): (u32, u8[4]) in u32:0..u32:4 {
        let element = delay(data[i], enable[i]);
        update(values, i, element)
    }(zero!<u8[4]>())
}

block explicit_array(clk, data: u8[4], enable: bool[4]) -> u8[4] {
    let a = delay(data[0], enable[0]);
    let b = delay(data[1], enable[1]);
    let c = delay(data[2], enable[2]);
    let d = delay(data[3], enable[3]);
    [a, b, c, d]
}
block delay(clk, data: u8, enable: bool) -> u8 {
    reg saved: u8 = 0;
    if enable { saved := data; }
    saved
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    let parent = package.get_block("delay_array").unwrap();
    assert_eq!(
        parent
            .instantiations
            .iter()
            .map(|inst| inst.name.as_str())
            .collect::<Vec<_>>(),
        ["element_0", "element_1", "element_2", "element_3"]
    );
    let [unrolled, explicit] = ["delay_array", "explicit_array"].map(|top| {
        // Vastly simulates one sequential module, so inline the clocked
        // children.
        let mut flat = package.clone();
        flat.set_top_block(top).unwrap();
        inline_all_blocks_in_package(&mut flat).unwrap();
        let rtl = emit_system_verilog(&flat, &BlockCodegenOptions::default()).unwrap();
        let module = compile_pipeline_module(&rtl.system_verilog).unwrap();
        let mut runner = FixtureRunner::new(&module).unwrap();
        let ports = runner.port_bindings();
        let data = ports.input("data").unwrap();
        let enable = ports.input("enable").unwrap();
        let rst = ports.input("rst").unwrap();
        [
            (0x4433_2211, 0b1111, 1),
            (0x4433_2211, 0b1111, 0),
            (0x8877_6655, 0b0101, 0),
            (0xccbb_aa99, 0b1010, 0),
            (0xffff_ffff, 0b0000, 0),
            (0x1234_5678, 0b1111, 1),
            (0xefcd_ab89, 0b0010, 0),
        ]
        .map(|(input, enables, reset)| {
            let edge = runner
                .step_cycle_with_drive(&mut [], |drive| {
                    drive.set_u64(&data, input)?;
                    drive.set_u64(&enable, enables)?;
                    drive.set_u64(&rst, reset)
                })
                .unwrap();
            edge.outputs["out"]
                .to_u64_if_known()
                .expect("known output after reset")
        })
    });
    assert_eq!(unrolled, explicit);
    assert_eq!(
        explicit,
        [
            0,
            0x4433_2211,
            0x4477_2255,
            0xcc77_aa55,
            0xcc77_aa55,
            0,
            0x0000_ab00
        ]
    );
}

#[test]
fn nested_unroll_instance_names_avoid_outer_bindings_and_skip_empty_ranges() {
    let source = r#"
pub block parent(clk, element_1_0: u8) -> u8 {
    let result = unroll_for! (_, a) in u32:2..u32:4 {
        unroll_for! (_, b) in u32:5..u32:7 {
            let element = echo(b);
            element
        }(a)
    }(element_1_0);
    let element_0_0 = echo(result);
    unroll_for! (_, a) in u32:3..u32:3 {
        let skipped = echo(a);
        skipped
    }(element_0_0)
}
#[allow(combinational_block)]
block echo(clk, x: u8) -> u8 { x }
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    let parent = package.get_block("parent").unwrap();
    assert_eq!(
        parent
            .instantiations
            .iter()
            .map(|inst| inst.name.as_str())
            .collect::<Vec<_>>(),
        [
            "element_0_0_1",
            "element_0_1",
            "element_1_0_1",
            "element_1_1",
            "element_0_0"
        ]
    );
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
}

#[test]
fn unrolled_block_calls_reject_conditional_hardware_and_cycles() {
    for (body, reason) in [
        (
            r#"if enable {
            unroll_for! (_, a) in u32:0..u32:1 { let element = echo(a); element }(x)
        } else { x }"#,
            "unconditional",
        ),
        (
            r#"unroll_for! (_, a) in u32:0..u32:1 {
            let element = parent(a, enable);
            element
        }(x)"#,
            "recursive block dependency",
        ),
        (
            r#"let next_value: u8;
        let loopback = echo(next_value);
        let result = unroll_for! (_, a) in u32:0..u32:1 {
            let element = echo(a);
            element
        }(loopback);
        next_value = result;
        result"#,
            "combinational cycle",
        ),
    ] {
        let source = format!(
            "pub block parent(clk, x: u8, enable: bool) -> u8 {{ {body} }}\nblock echo(clk, x: u8) -> u8 {{ x }}"
        );
        let error =
            compile(&source, empty_package("design"), ResetBehavior::default()).unwrap_err();
        assert!(error.contains(reason), "{body}: {error}");
    }
}

#[test]
fn unrolled_tuple_bindings_instantiate_each_child_once() -> Result<(), String> {
    let source = r#"
pub block parent(clk, x:u8) -> u8 {
    let result = unroll_for! (_, a) in u8:0..u8:2 {
        let (left, right) = split(a);
        let (_, tail) = split(right);
        left + tail
    }(x);
    let bx_inst_0 = split(result);
    bx_inst_0.0
}
#[allow(combinational_block)]
block split(clk, x:u8) -> (u8, u8) { (x, x + u8:1) }
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    let parent = package.get_block("parent").unwrap();
    assert_eq!(parent.instantiations.len(), 5);
    assert!(
        parent
            .instantiations
            .iter()
            .all(|instance| instance.block == "split")
    );
    assert_eq!(parent.instantiations.last().unwrap().name, "bx_inst_0");
    xlsynth::IrPackage::parse_ir(&package.to_string(), None).map_err(|e| e.to_string())?;
    Ok(())
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
#[allow(combinational_block)]
block split(clk, x: u8) -> (u8, bool) { (x, x == u8:0) }
#[allow(combinational_block)]
block samples(clk, x: u8) -> u8[2] { [x, x + u8:1] }
block sink(clk, x: u8) -> () { reg saved: u8; saved := x; () }
#[allow(combinational_block)]
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
block history(din: bits[8], previous: bits[8], fast_clock: clock, en: bits[1], clear: bits[1], observed: bits[8]) {
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
extern block history(clk, rst, en: bool, din: u8) -> Samples;
block parent(clk, valid: bool, x: u8) -> u8 {
    let outputs = history(valid, x);
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
fn inferred_reset_wires_by_role_when_a_child_has_an_output_named_rst() -> Result<(), String> {
    let outputs = "struct Outputs { rst: u8 }";
    let child = r#"block child(clk, x: u8) -> Outputs {
    reg q: u8 = 42;
    q := x;
    Outputs { rst: q }
}"#;
    let parent = r#"pub block parent(clk, x: u8) -> u8 {
    let sample = child(x);
    sample.rst
}"#;
    let reset = ResetBehavior::default();
    let local = compile(
        &format!("{outputs}\n{child}\n{parent}"),
        empty_package("design"),
        reset,
    )?;
    let child_package = compile(
        &format!("{outputs}\n{child}"),
        empty_package("design"),
        reset,
    )?;
    let external = "extern block child(clk, rst, x: u8) -> Outputs;";
    let linked = compile(
        &format!("{outputs}\n{external}\n{parent}"),
        child_package,
        reset,
    )?;
    for package in [local, linked] {
        let child = package.get_block("child").unwrap();
        let child_reset = child.port_name(child.reset.as_ref().unwrap().port);
        assert_ne!(child_reset, "rst");
        assert!(child.get_output_port("rst").is_some());
        let parent = package.get_block("parent").unwrap();
        let parent_reset = parent.reset.as_ref().unwrap().port;
        assert_eq!(parent.port_name(parent_reset), "rst");
        assert!(parent.nodes.iter().any(|node| matches!(
            &node.payload,
            NodePayload::InstantiationInput { instantiation, port_name, arg }
                if instantiation == "sample" && port_name == child_reset && *arg == parent_reset
        )));
        xlsynth::IrPackage::parse_ir(&package.to_string(), None).map_err(|e| e.to_string())?;
    }
    Ok(())
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
    let delay = "block delay(clk, d: u8) -> u8 { reg q: u8 = 0; q := d; q }";
    for parent in [
        "block top(clk, x: u8) -> u8 { if true { let q = delay(x); } x }",
        "block top(clk, x: u8) -> u8 { delay(x) }",
        "block top(clk, x: u8) -> u8 { let q = delay(delay(x)); q }",
        "block top(clk, x: u8) -> u8 { let q = delay(x, x); q }",
        "block top(clk, x: u8) -> u8 { let q: u16 = delay(x); x }",
        "block top(clk, x: u8) -> u8 { let q = top(x); q }",
    ] {
        let parsed = parse_module(&format!("{parent}\n{delay}")).unwrap();
        assert!(check_module(&parsed).is_err(), "{parent}");
    }
}
