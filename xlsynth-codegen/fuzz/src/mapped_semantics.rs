// SPDX-License-Identifier: Apache-2.0

//! Evaluates the actual emitted RTL after Yosys/ABC Liberty technology mapping.

use std::collections::{BTreeMap, HashMap};
use std::process::Command;
use std::time::Duration;

use xlsynth::IrBits;
use xlsynth::external_tool::{ToolError, run_checked_detailed};
use xlsynth_g8r::aig_sim::sequential::SequentialState;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, LabeledSequentialNetlistAig, SequentialClockEdge,
    load_labeled_sequential_netlist_aig_with_liberty,
};
use xlsynth_g8r_fuzz::external_yosys::ExternalYosysContext;
use xlsynth_pir::ir::Package;
use xlsynth_test_helpers::rtl_sim::{Bindings, LogicValue};

use crate::semantics::Trace;
use crate::top_block;

/// Checks mapped-cell behavior, including arbitrary initial and next state.
pub fn assert_mapped_trace(
    package: &Package,
    rtl: &str,
    context: &ExternalYosysContext,
    trace: &Trace,
) -> Result<(), ToolError> {
    let model = MappedModel::build(package, rtl, context)
        .map_err(|error| error.with_context(format!("IR:\n{package}\nRTL:\n{rtl}")))?;
    model.check_trace(trace).unwrap_or_else(|error| {
        panic!("Yosys/Liberty gate evaluation mismatch: {error}\nIR:\n{package}\nRTL:\n{rtl}\nMapped netlist:\n{}", model.netlist)
    });
    Ok(())
}

/// An original register bit's correspondence to a mapped FF state variable.
#[derive(Clone, Copy)]
struct StateBit {
    register_index: usize,
    inverted: bool,
}

struct MappedModel {
    mapped: LabeledSequentialNetlistAig,
    state_bits: BTreeMap<String, Vec<StateBit>>,
    netlist: String,
}

impl MappedModel {
    /// Preserves observable register Q bits without rewriting or parsing RTL.
    fn build(
        package: &Package,
        rtl: &str,
        context: &ExternalYosysContext,
    ) -> Result<Self, ToolError> {
        let (block, metadata) = top_block(package);
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        std::fs::write(directory.path().join("dut.sv"), rtl).map_err(|error| error.to_string())?;
        let mut program = String::new();
        let mut libraries = String::new();
        for path in context.yosys.liberty_files().paths() {
            let path = path
                .display()
                .to_string()
                .replace('\\', "\\\\")
                .replace('"', "\\\"");
            program.push_str(&format!("read_liberty -lib \"{path}\"\n"));
            libraries.push_str(&format!(" -liberty \"{path}\""));
        }
        // Generated fuzz names are plain identifiers, not Yosys selection syntax.
        for name in std::iter::once(&block.name).chain(metadata.registers.iter().map(|r| &r.name)) {
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                return Err(
                    format!("mapped fuzz oracle requires a plain identifier: {name:?}").into(),
                );
            }
        }
        program.push_str(&format!(
            "read_verilog -sv dut.sv\nhierarchy -check -top {}\nproc -noopt\nflatten\n",
            block.name
        ));
        for register in &metadata.registers {
            if register.ty.bit_count() != 0 {
                program.push_str(&format!("expose {}/w:{}\n", block.name, register.name));
            }
        }
        // Do not run opt_dff: constant-data and equivalent registers still have
        // independent arbitrary initial values in the PIR/Icarus trace. Exposing
        // Q also preserves registers that do not feed an ordinary output.
        // Prune unused logic and consolidate repeated mux inputs before
        // constant folding or bit-blasting large aggregate priority selects.
        // Restrict opt_merge to combinational cells so independently initialized
        // registers cannot merge. Sharing repeated one-hot masks before reducing
        // their OR tree avoids expanding duplicate logic during bit-blasting.
        program.push_str("opt_clean\nopt_merge t:$and t:$or t:$xor t:$not t:$mux t:$pmux\nopt_reduce\nopt_expr -keepdc\nopt_clean\ntechmap\nopt_expr -keepdc\nopt_clean\n");
        // Bitwise AND/OR/XOR chains become reduction cells whose repeated
        // operands can be eliminated before technology mapping. These passes
        // preserve all FF cells and their independent initial state.
        // Do not flatten the resulting shared reduction prefixes with opt_reduce:
        // remapping expanded prefixes duplicates their common input cones.
        program.push_str(
            "extract_reduce\nopt_expr -keepdc\nopt_clean\ntechmap\nopt_expr -keepdc\nopt_clean\n",
        );
        // Direct mapping avoids SAT sweeping wide arithmetic; synthesis QoR is
        // not part of this cycle-by-cycle simulation oracle.
        std::fs::write(directory.path().join("map.abc"), "strash\ndretime\nmap\n")
            .map_err(|error| error.to_string())?;
        program.push_str(&format!("dfflibmap{libraries}\nabc -script map.abc{libraries}\nclean -purge\ncheck -assert\nwrite_verilog -noattr mapped.gv\n"));
        run_checked_detailed(
            Command::new(context.yosys.yosys_path())
                .current_dir(directory.path())
                .args(["-Q", "-T", "-p", &program]),
            directory.path(),
            "yosys-liberty",
            Duration::from_secs(120),
        )
        .map_err(|error| error.with_context(format!("Yosys program:\n{program}")))?;
        let path = directory.path().join("mapped.gv");
        let netlist = std::fs::read_to_string(&path).map_err(|error| error.to_string())?;
        let mapped = load_labeled_sequential_netlist_aig_with_liberty(
            &path,
            &context.liberty,
            &GvEvalOptions {
                module_name: Some(block.name.clone()),
                clock_port_name: metadata.clock_port_name.clone(),
            },
        )
        .map_err(|error| format!("load mapped netlist: {error}\n{netlist}"))?;
        if let Some(clock) = &mapped.clock {
            if clock
                .active_edge
                .is_some_and(|edge| edge != SequentialClockEdge::Rising)
            {
                return Err("mapped clock does not use the expected rising edge".into());
            }
        }
        let design = &mapped.sequential_gate_fn;
        let mut q_bits = HashMap::new();
        for (index, register) in design.registers.iter().enumerate() {
            let q = &design.transition.inputs[register.q.index()].bit_vector;
            if q.get_bit_count() != 1 {
                return Err("Liberty FF importer must produce scalar state variables".into());
            }
            q_bits.insert(q.get_lsb(0).node, (index, q.get_lsb(0).negated));
        }
        let mut state_bits = BTreeMap::new();
        for register in &metadata.registers {
            if register.ty.bit_count() == 0 {
                // Zero-bit state has no physical register or observation port.
                continue;
            }
            let output = design
                .outputs
                .iter()
                .map(|id| &design.transition.outputs[id.index()])
                .find(|output| output.name == register.name)
                .ok_or_else(|| format!("missing exposed register {}", register.name))?;
            if output.get_bit_count() != register.ty.bit_count() {
                return Err(
                    format!("exposed register {} has incorrect width", register.name).into(),
                );
            }
            let bits = output
                .bit_vector
                .iter_lsb_to_msb()
                .map(|bit| {
                    let (register_index, q_inverted) = *q_bits.get(&bit.node).ok_or_else(|| {
                        format!("register {} does not map directly to FF Q", register.name)
                    })?;
                    Ok(StateBit {
                        register_index,
                        inverted: bit.negated ^ q_inverted,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            state_bits.insert(register.name.clone(), bits);
        }
        Ok(Self {
            mapped,
            state_bits,
            netlist,
        })
    }

    /// Runs the mapped state machine independently through the entire trace.
    fn check_trace(&self, trace: &Trace) -> Result<(), String> {
        let design = &self.mapped.sequential_gate_fn;
        let mut initial_bits = vec![None; design.registers.len()];
        if trace.initial_state.len() != self.state_bits.len() {
            return Err("initial state names do not match the exposed registers".to_string());
        }
        for (name, locations) in &self.state_bits {
            let bits = trace
                .initial_state
                .get(name)
                .ok_or_else(|| format!("missing initial state: {name}"))?
                .to_bits()?;
            if bits.get_bit_count() != locations.len() {
                return Err(format!("initial register {name} width mismatch"));
            }
            for (index, location) in locations.iter().enumerate() {
                let bit =
                    bits.get_bit(index).map_err(|error| error.to_string())? ^ location.inverted;
                if initial_bits[location.register_index].replace(bit).is_some() {
                    return Err("mapped FF aliases independent original state bits".to_string());
                }
            }
        }
        let values = initial_bits
            .into_iter()
            .enumerate()
            .map(|(index, bit)| {
                bit.map(|bit| IrBits::from_lsb_is_0(&[bit])).ok_or_else(|| {
                    format!("mapped FF {index} has no original state correspondence")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let state = SequentialState::from_register_values(design, values)?;
        let mut all_inputs = Vec::new();
        for (cycle, sample) in trace.samples.iter().enumerate() {
            let inputs = design
                .inputs
                .iter()
                .map(|id| {
                    let name = &design.transition.inputs[id.index()].name;
                    sample
                        .inputs
                        .get(name)
                        .ok_or_else(|| format!("missing mapped input {name}"))?
                        .to_bits()
                })
                .collect::<Result<Vec<_>, _>>()?;
            if inputs.len() != sample.inputs.len() {
                return Err(format!("cycle {cycle}: mapped input set mismatch"));
            }
            all_inputs.push(inputs);
        }
        let result = self.mapped.simulate_bits(&all_inputs, state)?;
        let mut expected_state = &trace.initial_state;
        for (cycle, sample) in trace.samples.iter().enumerate() {
            let outputs: Bindings = design
                .outputs
                .iter()
                .zip(&result.external_outputs()[cycle])
                .map(|(id, bits)| {
                    (
                        design.transition.outputs[id.index()].name.clone(),
                        LogicValue::from_bits(bits),
                    )
                })
                .collect();
            let mut expected = sample.outputs.clone();
            for (name, value) in expected_state {
                if expected.insert(name.clone(), value.clone()).is_some() {
                    return Err(format!("state/output observation name collision: {name}"));
                }
            }
            if outputs != expected {
                return Err(format!(
                    "cycle {cycle} outputs/state differ\ninputs: {:?}\nexpected: {expected:?}\nactual: {outputs:?}",
                    sample.inputs
                ));
            }
            if let Some(next) = &sample.next_state {
                expected_state = next;
            }
        }
        // Exposed Q at the next sample checks each preceding edge. Check the
        // final edge explicitly too, without rebuilding the gate evaluator.
        if let Some(next) = trace
            .samples
            .last()
            .and_then(|sample| sample.next_state.as_ref())
        {
            let state = result.final_state();
            let actual: Bindings = self
                .state_bits
                .iter()
                .map(|(name, locations)| {
                    let bits = locations
                        .iter()
                        .map(|location| {
                            state.values()[location.register_index]
                                .get_bit(0)
                                .expect("scalar mapped state")
                                ^ location.inverted
                        })
                        .collect::<Vec<_>>();
                    (
                        name.clone(),
                        LogicValue::from_bits(&IrBits::from_lsb_is_0(&bits)),
                    )
                })
                .collect();
            if actual != *next {
                return Err(format!(
                    "final next state differs\nexpected: {next:?}\nactual: {actual:?}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use xlsynth_codegen::BlockCodegenOptions;
    use xlsynth_g8r_fuzz::external_yosys::required_external_yosys_context;
    use xlsynth_test_helpers::rtl_sim::LogicValue;

    use super::MappedModel;
    use crate::semantics::Trace;
    use crate::{Profile, emit, parse_reference, references};

    #[test]
    fn wide_multiply_cones_match_all_four_oracles() {
        for ir in [
            include_str!("../testdata/mapped_product_enable.ir"),
            include_str!("../testdata/mapped_chained_squares.ir"),
        ] {
            let package = parse_reference(ir);
            let trace = Trace::for_package(&package);
            assert_eq!(
                crate::check_profile_trace(
                    &package,
                    Profile::StockXls,
                    &BlockCodegenOptions::default(),
                    &trace,
                )
                .unwrap(),
                None
            );
        }
    }

    #[test]
    fn wide_repeated_priority_cases_preserve_independent_state() {
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let cases = (0..129)
            .map(|index| ["data", "qa", "qb"][index % 3])
            .collect::<Vec<_>>()
            .join(", ");
        let package = parse_reference(&format!(
            r#"package public_wide_priority
top block wide_priority(clk: clock, data: bits[64][32], selector: bits[129], result: bits[64][32]) {{
  reg a(bits[64][32])
  reg b(bits[64][32])
  data: bits[64][32] = input_port(name=data, id=1)
  selector: bits[129] = input_port(name=selector, id=2)
  qa: bits[64][32] = register_read(register=a, id=3)
  qb: bits[64][32] = register_read(register=b, id=4)
  selected: bits[64][32] = priority_sel(selector, cases=[{cases}], default=qa, id=5)
  unused: bits[64][32] = priority_sel(selector, cases=[{cases}], default=qb, id=6)
  wa: () = register_write(selected, register=a, id=7)
  wb: () = register_write(selected, register=b, id=8)
  result: () = output_port(qa, name=result, id=9)
}}
"#,
        ));
        let rtl = emit(&package, &BlockCodegenOptions::default());
        let model = MappedModel::build(&package, &rtl, context).unwrap();
        assert_eq!(model.mapped.sequential_gate_fn.registers.len(), 4096);
        let trace = Trace::for_package(&package);
        assert_ne!(trace.initial_state["a"], trace.initial_state["b"]);
        model.check_trace(&trace).unwrap();
    }

    #[test]
    fn repeated_one_hot_masks_preserve_independent_state() {
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let cases = vec!["qa"; 233].join(", ");
        let package = parse_reference(&format!(
            r#"package public_repeated_one_hot
top block repeated_one_hot(clk: clock, selector: bits[4], result: bits[32][32]) {{
  reg a(bits[32][32])
  reg b(bits[32][32])
  selector: bits[4] = input_port(name=selector, id=1)
  qa: bits[32][32] = register_read(register=a, id=2)
  qb: bits[32][32] = register_read(register=b, id=3)
  extended: bits[233] = sign_ext(selector, new_bit_count=233, id=4)
  selected: bits[32][32] = one_hot_sel(extended, cases=[{cases}], id=5)
  wa: () = register_write(selected, register=a, id=6)
  wb: () = register_write(selected, register=b, id=7)
  result: () = output_port(selected, name=result, id=8)
}}
"#,
        ));
        let rtl = emit(&package, &BlockCodegenOptions::default());
        let model = MappedModel::build(&package, &rtl, context).unwrap();
        assert_eq!(model.mapped.sequential_gate_fn.registers.len(), 2048);
        let trace = Trace::for_package(&package);
        assert_ne!(trace.initial_state["a"], trace.initial_state["b"]);
        model.check_trace(&trace).unwrap();
    }

    #[test]
    fn shared_reduction_prefixes_preserve_independent_state() {
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let terms = (0..64)
            .map(|i| ["lhs", "rhs"][i % 2])
            .collect::<Vec<_>>()
            .join(", ");
        let cases = (0..64)
            .map(|i| ["qa", "qb"][i % 2])
            .collect::<Vec<_>>()
            .join(", ");
        let package = parse_reference(&format!(
            r#"package public_shared_prefix
top block shared_prefix(clk: clock, lhs: bits[64], rhs: bits[64], result: bits[9]) {{
  reg a(bits[9])
  reg b(bits[9])
  lhs: bits[64] = input_port(name=lhs, id=1)
  rhs: bits[64] = input_port(name=rhs, id=2)
  qa: bits[9] = register_read(register=a, id=3)
  qb: bits[9] = register_read(register=b, id=4)
  terms: bits[64] = one_hot_sel(lhs, cases=[{terms}], id=5)
  selected: bits[9] = priority_sel(terms, cases=[{cases}], default=qa, id=6)
  wa: () = register_write(selected, register=a, id=7)
  wb: () = register_write(selected, register=b, id=8)
  result: () = output_port(selected, name=result, id=9)
}}
"#,
        ));
        let rtl = emit(&package, &BlockCodegenOptions::default());
        let model = MappedModel::build(&package, &rtl, context).unwrap();
        assert_eq!(model.mapped.sequential_gate_fn.registers.len(), 18);
        let trace = Trace::for_package(&package);
        assert_ne!(trace.initial_state["a"], trace.initial_state["b"]);
        model.check_trace(&trace).unwrap();
    }

    #[test]
    fn mapped_state_preserves_resetless_constant_dead_and_wide_registers() {
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let package = parse_reference(
            r#"package public_mapped_state

top block mapped_state(clk: clock, data: bits[65][2], enable: bits[1], result: bits[65][2]) {
  reg a(bits[65][2])
  reg b(bits[65][2])
  reg unused(bits[8])
  data: bits[65][2] = input_port(name=data, id=1)
  enable: bits[1] = input_port(name=enable, id=2)
  current_a: bits[65][2] = register_read(register=a, id=3)
  current_b: bits[65][2] = register_read(register=b, id=4)
  current_unused: bits[8] = register_read(register=unused, id=5)
  zero: bits[8] = literal(value=0, id=6)
  wa: () = register_write(data, register=a, load_enable=enable, id=7)
  wb: () = register_write(data, register=b, load_enable=enable, id=8)
  wu: () = register_write(zero, register=unused, id=9)
  result: () = output_port(current_a, name=result, id=10)
}
"#,
        );
        let rtl = emit(&package, &BlockCodegenOptions::default());
        let model = MappedModel::build(&package, &rtl, context).unwrap();
        assert_eq!(model.mapped.sequential_gate_fn.registers.len(), 268);
        let mut trace = Trace::for_package(&package);
        assert_ne!(trace.initial_state["a"], trace.initial_state["b"]);
        model.check_trace(&trace).unwrap();
        // An unobservable register after the final edge must still be checked.
        trace
            .samples
            .last_mut()
            .unwrap()
            .next_state
            .as_mut()
            .unwrap()
            .insert("unused".to_string(), LogicValue::from_u64(8, 1));
        assert!(model.check_trace(&trace).is_err());
    }

    #[test]
    fn mapped_oracle_detects_corrupt_outputs_and_intermediate_state() {
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let package = parse_reference(references::SEQUENTIAL);
        let rtl = emit(&package, &BlockCodegenOptions::default());
        let model = MappedModel::build(&package, &rtl, context).unwrap();
        let mut trace = Trace::for_package(&package);
        model.check_trace(&trace).unwrap();
        trace.samples[2].next_state.as_mut().unwrap().clear();
        assert!(model.check_trace(&trace).is_err());
        let mut trace = Trace::for_package(&package);
        trace.samples[2].outputs.clear();
        assert!(model.check_trace(&trace).is_err());
    }
}
