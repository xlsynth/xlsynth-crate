// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::ir::{InstantiationKind, MemberType, NodePayload, Package, PackageMember, Type};
use xlsynth_pir::ir_value_utils::{
    flatten_ir_value_to_lsb0_bits_for_type, ir_value_from_lsb0_bits_with_layout,
};
use xlsynth_vastly::{
    DriveContext, FixtureRunner, InputPortHandle, LogicBit, Signedness, Value4,
    compile_pipeline_module_without_spans,
};

/// One typed stimulus input and its current drive. Zero-width inputs have no
/// corresponding RTL port.
struct Input {
    ty: Type,
    drive: Option<(InputPortHandle, Value4)>,
}

/// Simulates the test's connected hardware, preserving both input drives and
/// register state between statements.
pub(super) struct Simulation {
    runner: FixtureRunner,
    inputs: BTreeMap<String, Input>,
    outputs: Vec<(String, Type)>,
    settled: Option<BTreeMap<String, Value4>>,
}

impl Simulation {
    pub(super) fn new(package: &Package, top: &str) -> Result<Self, String> {
        let mut package = test_hierarchy(package, top)?;
        inline_all_blocks_in_package(&mut package)?;
        let block = package.get_block(top).ok_or("missing inlined test block")?;
        let reset = block.reset.as_ref();
        if reset.is_some_and(|r| r.asynchronous)
            && block.nodes.iter().any(|node| {
                matches!(node.payload, NodePayload::RegisterWrite { arg, reset: Some(_), .. }
                    if block.get_node_ty(arg).bit_count() != 0)
            })
        {
            return Err("BX test simulation does not support asynchronous reset".to_owned());
        }
        let rtl = emit_system_verilog(
            &package,
            &BlockCodegenOptions {
                emit_sv_types: false,
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;
        let module = compile_pipeline_module_without_spans(&rtl.system_verilog)
            .map_err(|e| format!("could not compile test RTL: {e:?}"))?;
        let runner = FixtureRunner::new(&module).map_err(|e| format!("{e:?}"))?;
        let mut inputs = BTreeMap::new();
        for port in block.input_ports() {
            let name = block.port_name(port);
            let ty = block.port_type(port);
            let width = u32::try_from(ty.bit_count()).map_err(|_| "RTL input is too wide")?;
            let drive = if width == 0 {
                None
            } else {
                let handle = runner
                    .port_bindings()
                    .input_with_width(name, width)
                    .map_err(|e| format!("{e:?}"))?;
                let value = if let Some(reset) = reset.filter(|r| r.port == port) {
                    Value4::from_u64(1, Signedness::Unsigned, u64::from(reset.active_low))
                } else {
                    Value4::new(
                        width,
                        Signedness::Unsigned,
                        vec![LogicBit::X; width as usize],
                    )
                };
                Some((handle, value))
            };
            inputs.insert(
                name.to_owned(),
                Input {
                    ty: ty.clone(),
                    drive,
                },
            );
        }
        let outputs = block
            .output_ports()
            .map(|port| {
                (
                    block.port_name(port).to_owned(),
                    block.port_type(port).clone(),
                )
            })
            .collect();
        Ok(Self {
            runner,
            inputs,
            outputs,
            settled: None,
        })
    }

    /// Updates one drive immediately; register values change at the next edge.
    pub(super) fn drive(&mut self, port: &str, value: &IrValue) -> Result<(), String> {
        let input = self
            .inputs
            .get_mut(port)
            .ok_or_else(|| format!("unknown test input '{port}'"))?;
        let ty = value.type_();
        if input.ty != ty {
            return Err(format!(
                "test input '{port}' expects {}, got {ty}",
                input.ty
            ));
        }
        if let Some((handle, drive)) = &mut input.drive {
            let mut bits = Vec::with_capacity(handle.width() as usize);
            flatten_ir_value_to_lsb0_bits_for_type(value, &input.ty, &mut bits)?;
            let bits = bits
                .into_iter()
                .map(|bit| if bit { LogicBit::One } else { LogicBit::Zero })
                .collect();
            *drive = Value4::new(handle.width(), handle.signedness(), bits);
            self.settled = None;
        }
        Ok(())
    }

    /// Advances every connected instance on the same rising edge.
    pub(super) fn tick(&mut self) -> Result<(), String> {
        let edge = self
            .runner
            .step_cycle_with_drive(&mut [], |ctx| apply_drives(&self.inputs, ctx))
            .map_err(|e| format!("test clock edge failed: {e:?}"))?;
        self.settled = Some(edge.outputs);
        Ok(())
    }

    /// Returns the typed output after the most recent stimulus or clock edge.
    pub(super) fn output(&mut self, index: usize) -> Result<IrValue, String> {
        let (name, ty) = self
            .outputs
            .get(index)
            .ok_or_else(|| format!("no test output at index {index}"))?;
        if ty.bit_count() == 0 {
            return ir_value_from_lsb0_bits_with_layout(ty, &[]);
        }
        if self.settled.is_none() {
            self.settled = Some(
                self.runner
                    .sample_with_drive(|ctx| apply_drives(&self.inputs, ctx))
                    .map_err(|e| format!("could not sample test output: {e:?}"))?,
            );
        }
        let value = self
            .settled
            .as_ref()
            .and_then(|values| values.get(name))
            .ok_or_else(|| format!("test RTL has no output '{name}'"))?;
        if value.has_unknown() {
            return Err(format!(
                "output '{name}' contains unknown bits: {}'b{}",
                value.width,
                value.to_bit_string_msb_first()
            ));
        }
        let bits: Vec<bool> = value
            .bits_lsb_first()
            .iter()
            .map(|&b| b == LogicBit::One)
            .collect();
        ir_value_from_lsb0_bits_with_layout(ty, &bits)
    }
}

/// Copy only the hardware connected by this test and its transitive pure
/// callees.
fn test_hierarchy(package: &Package, top: &str) -> Result<Package, String> {
    let members: BTreeMap<_, _> = package
        .members
        .iter()
        .map(|member| (member.graph().name.as_str(), member))
        .collect();
    let mut reachable = BTreeSet::new();
    let mut pending = vec![top];
    while let Some(name) = pending.pop() {
        if !reachable.insert(name) {
            continue;
        }
        let member = members
            .get(name)
            .ok_or_else(|| format!("missing test dependency '{name}'"))?;
        if let PackageMember::Block(block) = member {
            for instance in &block.instantiations {
                if instance.kind != InstantiationKind::Block {
                    return Err(format!(
                        "cannot simulate external instance '{}'",
                        instance.name
                    ));
                }
                pending.push(&instance.block);
            }
        }
        pending.extend(
            member
                .graph()
                .nodes
                .iter()
                .filter_map(|node| match &node.payload {
                    NodePayload::Invoke { to_apply, .. } | NodePayload::Map { to_apply, .. } => {
                        Some(to_apply.as_str())
                    }
                    NodePayload::CountedFor { body, .. } => Some(body.as_str()),
                    _ => None,
                }),
        );
    }
    Ok(Package {
        name: package.name.clone(),
        file_table: package.file_table.clone(),
        members: package
            .members
            .iter()
            .filter(|member| reachable.contains(member.graph().name.as_str()))
            .cloned()
            .collect(),
        top: Some((top.to_owned(), MemberType::Block)),
    })
}

/// Applies persistent stimulus to a fresh sampling or ticking context.
fn apply_drives(
    inputs: &BTreeMap<String, Input>,
    ctx: &mut DriveContext<'_>,
) -> xlsynth_vastly::Result<()> {
    for (handle, value) in inputs.values().filter_map(|input| input.drive.as_ref()) {
        ctx.set(handle, value.clone())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use xlsynth_pir::ir_parser::parse_xls_ir_package;

    use super::*;
    use crate::{ImportOptions, ResetBehavior, compile, empty_package, run_tests};

    const SOURCE: &str = r#"
type Payload = (uN[80], u4[2], ());
block delay(clk, rst, data: Payload) -> Payload {
    reg q: Payload = zero!<Payload>();
    q := data;
    q
}
pub block sample(clk, rst, data: Payload) -> (Payload, Payload, bool, ()) {
    let q = delay(data);
    (data, q, rst, ())
}
"#;

    #[test]
    fn drive_settles_aggregates_without_clocking_the_child() -> Result<(), String> {
        let zero = IrValue::parse_typed("(bits[80]:0, [bits[4]:0, bits[4]:0], ())")
            .map_err(|e| e.to_string())?;
        let first =
            IrValue::parse_typed("(bits[80]:0xabcdef12345678012345, [bits[4]:2, bits[4]:7], ())")
                .map_err(|e| e.to_string())?;
        let second =
            IrValue::parse_typed("(bits[80]:0xf0000000000000000002, [bits[4]:10, bits[4]:3], ())")
                .map_err(|e| e.to_string())?;
        let asserted = IrValue::make_ubits(1, 1).map_err(|e| e.to_string())?;
        let inactive = IrValue::make_ubits(1, 0).map_err(|e| e.to_string())?;
        for active_low in [false, true] {
            let package = compile(
                SOURCE,
                empty_package("sample"),
                ResetBehavior {
                    asynchronous: false,
                    active_low,
                },
            )?;
            let mut sim = Simulation::new(&package, "sample")?;
            sim.drive("data", &first)?;
            assert_eq!(sim.output(0)?, first);
            assert_eq!(
                sim.output(1).unwrap_err(),
                format!(
                    "output 'out_1' contains unknown bits: 88'b{}",
                    "x".repeat(88)
                )
            );
            assert_eq!(sim.output(2)?, inactive);
            assert_eq!(sim.output(3)?, IrValue::make_tuple(&[]));
            sim.tick()?;
            assert_eq!(sim.output(1)?, first);
            sim.drive("data", &second)?;
            assert_eq!(sim.output(0)?, second);
            assert_eq!(sim.output(1)?, first);
            sim.tick()?;
            assert_eq!(sim.output(1)?, second);
            sim.drive("rst", if active_low { &inactive } else { &asserted })?;
            assert_eq!(sim.output(2)?, asserted);
            assert_eq!(sim.output(1)?, second);
            sim.tick()?;
            assert_eq!(sim.output(1)?, zero);
        }
        Ok(())
    }

    #[test]
    fn asynchronous_reset_has_an_explicit_simulation_diagnostic() -> Result<(), String> {
        let package = compile(
            SOURCE,
            empty_package("sample"),
            ResetBehavior {
                asynchronous: true,
                active_low: false,
            },
        )?;
        assert_eq!(
            Simulation::new(&package, "sample").err().as_deref(),
            Some("BX test simulation does not support asynchronous reset")
        );
        Ok(())
    }

    #[test]
    fn tests_without_dut_or_observed_outputs_can_advance_the_clock() -> Result<(), String> {
        let source = r#"
#[test] block empty() {}
#[test] block clocks_only() { tick!(); tick!(); }
"#;
        let report = run_tests(
            source,
            Path::new("empty.bx"),
            empty_package("empty"),
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
        )?;
        assert_eq!(report.tests.len(), 2);
        for test in report.tests {
            assert_eq!(test.failure, None, "{}", test.name);
        }
        Ok(())
    }

    #[test]
    fn shared_helpers_evaluate_discarded_fields_only_in_pure_tests() -> Result<(), String> {
        let source = r#"extern fn lookup(xs: u8[2], i: u8) -> u8;
fn keep(x: u8) -> u8 {
    let (_, y) = (for (i, a) in u8:1..=u8:2 { lookup([a, x], i) }(x), x);
    y
}
block delay(clk, x: u8) -> u8 {
    reg q: u8;
    q := keep(x);
    q
}
#[test] block clocked() {
    reg x: u8 = u8:10;
    let q = delay(x);
    tick!();
    assert_eq(q, u8:10);
}
#[test] fn pure_helper() { assert_eq(keep(u8:10), u8:10); }
#[test] fn pure_scope() {
    assert_eq({ let (_, y) = (lookup([u8:1, u8:2], u8:2), u8:10); y }, u8:10);
}"#;
        let package = parse_xls_ir_package(
            r#"package mixed
fn lookup(xs: bits[8][2] id=1, i: bits[8] id=2) -> bits[8] {
    ret selected: bits[8] = array_index(xs, indices=[i], assumed_in_bounds=true, id=3)
}"#,
        )
        .map_err(|e| e.to_string())?;
        let report = run_tests(
            source,
            Path::new("mixed.bx"),
            package,
            ResetBehavior::default(),
            &ImportOptions::default(),
            true,
        )?;
        assert_eq!(report.tests.len(), 3);
        assert_eq!(report.tests[0].failure, None);
        let failure = "IR array_index at node 3 was out of bounds despite assumed_in_bounds=true";
        assert_eq!(
            report.tests[1].failure,
            Some(format!("mixed.bx:17:28: test 'pure_helper': {failure}"))
        );
        assert_eq!(
            report.tests[2].failure,
            Some(format!("mixed.bx:19:5: test 'pure_scope': {failure}"))
        );
        Ok(())
    }

    #[test]
    fn counted_loops_in_dut_helpers_report_the_codegen_limitation() -> Result<(), String> {
        let source = r#"
fn sum(words: u8[2]) -> u8 {
    for (word, acc) in words { word + acc }(u8:0)
}
block sample(clk, words: u8[2]) -> u8 {
    reg total: u8;
    total := sum(words);
    total
}
"#;
        let package = compile(source, empty_package("design"), ResetBehavior::default())?;
        assert_eq!(
            Simulation::new(&package, "sample").err().as_deref(),
            Some(
                "counted_for is not supported in `sample`; unroll loops before block2sv code generation"
            ),
        );
        Ok(())
    }
}
