// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::collections::BTreeMap;

use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog_with_register_names};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_inline::flatten_block_hierarchy;
use xlsynth_pir::block_invoke_inline::BlockInvokeInliner;
use xlsynth_pir::ir::{NodePayload, Package, Type};
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
    /// Physical level of a deasserted reset for this port, if it is reset.
    inactive_reset: Option<bool>,
}

/// Connects a flattened PIR register to its packed RTL signal. A zero-width
/// register participates in the IR state without requiring an RTL signal.
struct Register {
    ty: Type,
    rtl_name: Option<String>,
}

/// A settled output or the four-state diagnostic for an unknown observation.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Observed {
    Known(IrValue),
    Unknown(String),
}

/// Simulates the test's connected hardware, preserving both input drives and
/// register state between statements.
pub(super) struct Simulation {
    runner: FixtureRunner,
    inputs: BTreeMap<String, Input>,
    registers: BTreeMap<String, Register>,
    outputs: Vec<(String, Type)>,
    settled: Option<BTreeMap<String, Value4>>,
}

impl Simulation {
    pub(super) fn new(package: &Package, top: &str) -> Result<Self, String> {
        let package = flatten_block_hierarchy(package, top)?;
        Self::compile_rtl(Cow::Owned(package), top, None)
    }

    /// Prepares RTL from the same flattened hierarchy used for PIR evaluation.
    pub(super) fn from_flattened(package: &Package, top: &str) -> Result<Self, String> {
        // Only the RTL cross-check unrolls loops and maps. PIR evaluates them
        // directly. Ordinary function inlining does not need this budget.
        const MAX_UNROLLED_NODES: usize = 65_536;
        let has_loops = package.members.iter().any(|member| {
            member.graph().nodes.iter().any(|node| {
                matches!(
                    node.payload,
                    NodePayload::Map { .. } | NodePayload::CountedFor { .. }
                )
            })
        });
        Self::compile_rtl(
            Cow::Borrowed(package),
            top,
            has_loops.then_some(MAX_UNROLLED_NODES),
        )
    }

    /// Inlines functions in the flattened test block, then compiles its RTL
    /// and binds its data ports and state.
    fn compile_rtl(
        mut package: Cow<'_, Package>,
        top: &str,
        unroll_budget: Option<usize>,
    ) -> Result<Self, String> {
        let mut inliner = BlockInvokeInliner::new(&package)?;
        let block = match unroll_budget {
            Some(limit) => inliner.inline_block_with_unrolling(top, limit)?,
            None => inliner.inline_block(top)?,
        };
        if let Cow::Owned(expanded) = block {
            package.to_mut().replace_block(top, expanded)?;
        }
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
        let rtl = emit_system_verilog_with_register_names(
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
            let inactive_reset = reset.filter(|r| r.port == port).map(|r| r.active_low);
            let drive = if width == 0 {
                None
            } else {
                let handle = runner
                    .port_bindings()
                    .input_with_width(name, width)
                    .map_err(|e| format!("{e:?}"))?;
                let value = if let Some(level) = inactive_reset {
                    Value4::from_u64(1, Signedness::Unsigned, u64::from(level))
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
                    inactive_reset,
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
        let mut rtl_names = rtl.top_register_names;
        let registers = block
            .registers
            .iter()
            .map(|reg| {
                let rtl_name = rtl_names.remove(&reg.name);
                if reg.ty.bit_count() != 0 && rtl_name.is_none() {
                    return Err(format!(
                        "test RTL has no signal for register '{}'",
                        reg.name
                    ));
                }
                Ok((
                    reg.name.clone(),
                    Register {
                        ty: reg.ty.clone(),
                        rtl_name,
                    },
                ))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            runner,
            inputs,
            registers,
            outputs,
            settled: None,
        })
    }

    /// Clears test drives, sampled outputs and clock history, then installs the
    /// trial's concrete power-up values, or all-X when running an X-state
    /// check.
    pub(super) fn reset_trial(
        &mut self,
        initial: Option<&BTreeMap<String, IrValue>>,
    ) -> Result<(), String> {
        if let Some(initial) = initial {
            if self.registers.keys().ne(initial.keys()) {
                return Err("initial state must supply exactly the flattened PIR registers".into());
            }
            let mut state = BTreeMap::new();
            for (name, register) in &self.registers {
                let value = &initial[name];
                if value.type_() != register.ty {
                    return Err(format!(
                        "initial register '{name}' expects {}, got {}",
                        register.ty,
                        value.type_()
                    ));
                }
                if let Some(rtl_name) = &register.rtl_name {
                    state.insert(
                        rtl_name.clone(),
                        packed_value(value, &register.ty, Signedness::Unsigned)?,
                    );
                }
            }
            self.runner
                .reset_with_state(state)
                .map_err(|e| format!("{e:?}"))?;
        } else {
            self.runner.reset_state_x();
        }
        // Both QuickCheck executions begin with identical testbench drives.
        // Only the DUT register state differs between concrete and X trials.
        for input in self.inputs.values_mut() {
            if let Some((handle, drive)) = &mut input.drive {
                *drive = Value4::from_u64(
                    handle.width(),
                    handle.signedness(),
                    u64::from(input.inactive_reset.unwrap_or(false)),
                );
            }
        }
        self.settled = None;
        Ok(())
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
            *drive = packed_value(value, &input.ty, handle.signedness())?;
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
    #[cfg(test)]
    fn output(&mut self, index: usize) -> Result<IrValue, String> {
        match self.output_observed(index)? {
            Observed::Known(value) => Ok(value),
            Observed::Unknown(diagnostic) => Err(diagnostic),
        }
    }

    /// Keeps X/Z observations distinct from failures to simulate or sample.
    pub(super) fn output_observed(&mut self, index: usize) -> Result<Observed, String> {
        let (name, ty) = self
            .outputs
            .get(index)
            .ok_or_else(|| format!("no test output at index {index}"))?;
        if ty.bit_count() == 0 {
            return ir_value_from_lsb0_bits_with_layout(ty, &[]).map(Observed::Known);
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
            return Ok(Observed::Unknown(format!(
                "output '{name}' contains unknown bits: {}'b{}",
                value.width,
                value.to_bit_string_msb_first()
            )));
        }
        let bits: Vec<bool> = value
            .bits_lsb_first()
            .iter()
            .map(|&b| b == LogicBit::One)
            .collect();
        ir_value_from_lsb0_bits_with_layout(ty, &bits).map(Observed::Known)
    }
}

/// Packs tuple tails and array heads into the least significant RTL bits.
fn packed_value(value: &IrValue, ty: &Type, signedness: Signedness) -> Result<Value4, String> {
    let width = u32::try_from(ty.bit_count()).map_err(|_| "RTL value is too wide")?;
    let mut bits = Vec::with_capacity(width as usize);
    flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut bits)?;
    let bits = bits
        .into_iter()
        .map(|bit| if bit { LogicBit::One } else { LogicBit::Zero })
        .collect();
    Ok(Value4::new(width, signedness, bits))
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
#![allow(block_reset_signal)]
type Payload = (uN[80], u4[2], ());
block delay(clk, data: Payload) -> Payload {
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
    fn trials_restart_aggregate_registers_drives_and_observations() -> Result<(), String> {
        let package = compile(SOURCE, empty_package("sample"), ResetBehavior::default())?;
        let flat = flatten_block_hierarchy(&package, "sample")?;
        let register = &flat.get_block("sample").unwrap().registers[0];
        let value =
            IrValue::parse_typed("(bits[80]:0xabcdef12345678012345, [bits[4]:2, bits[4]:7], ())")
                .map_err(|e| e.to_string())?;
        let state = BTreeMap::from([(register.name.clone(), value.clone())]);
        let mut sim = Simulation::from_flattened(&flat, "sample")?;
        sim.reset_trial(Some(&state))?;
        assert_eq!(sim.output(1)?, value);
        sim.drive("data", &value)?;
        sim.drive("rst", &IrValue::bool(true))?;
        sim.tick()?;
        assert_ne!(sim.output(1)?, value);
        assert_eq!(sim.runner.cycle(), 1);

        sim.reset_trial(None)?;
        assert_eq!(sim.runner.cycle(), 0);
        assert!(sim.runner.last_inputs().is_empty());
        assert!(sim.runner.last_outputs().is_empty());
        assert_eq!(
            sim.output_observed(1)?,
            Observed::Unknown(format!(
                "output 'out_1' contains unknown bits: 88'b{}",
                "x".repeat(88)
            ))
        );
        let zero = IrValue::parse_typed("(bits[80]:0, [bits[4]:0, bits[4]:0], ())")
            .map_err(|e| e.to_string())?;
        assert_eq!(sim.output(0)?, zero);
        assert_eq!(sim.output(2)?, IrValue::bool(false));
        assert_eq!(sim.output(3)?, IrValue::make_tuple(&[]));

        sim.reset_trial(Some(&state))?;
        assert_eq!(sim.output(1)?, value);
        Ok(())
    }

    #[test]
    fn concrete_state_uses_register_names_after_both_ir_and_rtl_collisions() -> Result<(), String> {
        let package = parse_xls_ir_package(
            r#"package design
block delay(clk: clock, data: bits[8], q: bits[8]) {
  reg r(bits[8])
  data: bits[8] = input_port(name=data, id=1)
  read: bits[8] = register_read(register=r, id=2)
  write: () = register_write(data, register=r, id=3)
  q: () = output_port(read, name=q, id=4)
}
top block sample(clk: clock, data: bits[8], u0__r: bits[8], parent_q: bits[8]) {
  reg u0__r(bits[8])
  reg unit(())
  instantiation u0(block=delay, kind=block)
  data: bits[8] = input_port(name=data, id=11)
  child_drive: () = instantiation_input(data, instantiation=u0, port_name=data, id=12)
  child_q: bits[8] = instantiation_output(instantiation=u0, port_name=q, id=13)
  u0__r: () = output_port(child_q, name=u0__r, id=14)
  parent_read: bits[8] = register_read(register=u0__r, id=15)
  parent_write: () = register_write(data, register=u0__r, id=16)
  parent_q: () = output_port(parent_read, name=parent_q, id=17)
  unit_value: () = register_read(register=unit, id=18)
  unit_write: () = register_write(unit_value, register=unit, id=19)
}"#,
        )
        .map_err(|e| e.to_string())?;
        let child = IrValue::make_ubits(8, 0xa1).map_err(|e| e.to_string())?;
        let parent = IrValue::make_ubits(8, 0x3d).map_err(|e| e.to_string())?;
        let state = BTreeMap::from([
            ("u0__r".into(), parent.clone()),
            ("u0__r__1".into(), child.clone()),
            ("unit".into(), IrValue::make_tuple(&[])),
        ]);
        let mut sim = Simulation::new(&package, "sample")?;
        sim.reset_trial(Some(&state))?;
        assert_eq!(sim.output(0)?, child);
        assert_eq!(sim.output(1)?, parent);
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

    #[test]
    fn quickcheck_rtl_unrolls_counted_loops_inside_a_mapped_helper() -> Result<(), String> {
        let source = r#"
fn advance(x: u8) -> u8 {
    for (i, a) in u8:0..u8:3 { a + x + i }(u8:1)
}
block sample(clk, words: u8[2]) -> u8[2] {
    reg q: u8[2];
    q := map(words, advance);
    q
}
"#;
        let package = compile(source, empty_package("design"), ResetBehavior::default())?;
        let flat = flatten_block_hierarchy(&package, "sample")?;
        let mut inliner = BlockInvokeInliner::new(&flat)?;
        // The top-level map adds six nodes. Its copied callbacks and counted
        // loops must share that same allowance.
        assert_eq!(
            inliner
                .inline_block_with_unrolling("sample", 6)
                .unwrap_err(),
            "expanding calls in block 'sample' exceeds the budget of 6 additional IR nodes"
        );
        let words = IrValue::parse_typed("[bits[8]:2, bits[8]:10]").map_err(|e| e.to_string())?;
        let expected =
            IrValue::parse_typed("[bits[8]:10, bits[8]:34]").map_err(|e| e.to_string())?;
        let mut sim = Simulation::from_flattened(&flat, "sample")?;
        sim.reset_trial(Some(&BTreeMap::from([("q".into(), words.clone())])))?;
        assert_eq!(sim.output(0)?, words);
        sim.drive("words", &words)?;
        sim.tick()?;
        assert_eq!(sim.output(0)?, expected);
        Ok(())
    }
}
