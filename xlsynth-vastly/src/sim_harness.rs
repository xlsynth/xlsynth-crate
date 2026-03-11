// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::compiled_module::CompiledSeqBlock;
use crate::compiled_module::State;
use crate::vcd_writer::VcdWriter;
use crate::Env;
use crate::Error;
use crate::Result;

#[derive(Debug, Clone)]
pub struct Stimulus {
    pub half_period: u64,
    pub cycles: Vec<Cycle>,
}

#[derive(Debug, Clone)]
pub struct Cycle {
    pub inputs: BTreeMap<String, crate::Value4>,
}

pub fn run_and_write_vcd(
    m: &CompiledSeqBlock,
    stimulus: &Stimulus,
    initial_state: &State,
    out_path: &std::path::Path,
) -> Result<()> {
    let mut writer = VcdWriter::new("1ns");
    for (name, info) in &m.decls {
        // Mirror iverilog `$dumpvars(0, tb)` naming:
        // - tb.<signal> for testbench-level nets/regs
        // - tb.dut.<signal> for DUT ports/regs
        writer.add_var(&format!("tb.{name}"), info.width)?;
        writer.add_var(&format!("tb.dut.{name}"), info.width)?;
    }

    let mut f =
        std::fs::File::create(out_path).map_err(|e| Error::Parse(format!("io error: {e}")))?;
    writer.write_header(&mut f)?;

    let mut time: u64 = 0;
    writer.change(&mut f, time, &format!("tb.{}", m.clk_name), "0")?;
    writer.change(&mut f, time, &format!("tb.dut.{}", m.clk_name), "0")?;

    // Initialize declared signals from initial_state if present; otherwise leave X.
    for (name, v) in initial_state {
        writer.change(
            &mut f,
            time,
            &format!("tb.{name}"),
            &v.to_bit_string_msb_first(),
        )?;
        writer.change(
            &mut f,
            time,
            &format!("tb.dut.{name}"),
            &v.to_bit_string_msb_first(),
        )?;
    }

    // Match the iverilog TB behavior: drive cycle-0 inputs at time 0.
    if let Some(c0) = stimulus.cycles.first() {
        for (k, v) in &c0.inputs {
            writer.change(
                &mut f,
                time,
                &format!("tb.{k}"),
                &v.to_bit_string_msb_first(),
            )?;
            writer.change(
                &mut f,
                time,
                &format!("tb.dut.{k}"),
                &v.to_bit_string_msb_first(),
            )?;
        }
    }

    let mut state = initial_state.clone();

    for (cyc_idx, cyc) in stimulus.cycles.iter().enumerate() {
        let base_t = (cyc_idx as u64) * stimulus.half_period * 2;
        time = base_t;
        // Ensure clock low at start of cycle.
        writer.change(&mut f, time, &format!("tb.{}", m.clk_name), "0")?;
        writer.change(&mut f, time, &format!("tb.dut.{}", m.clk_name), "0")?;

        // Apply inputs at +1 (stable before posedge).
        let input_t = time + 1;
        let mut inputs = Env::new();
        for (k, v) in &cyc.inputs {
            inputs.insert(k.clone(), v.clone());
            writer.change(
                &mut f,
                input_t,
                &format!("tb.{k}"),
                &v.to_bit_string_msb_first(),
            )?;
            writer.change(
                &mut f,
                input_t,
                &format!("tb.dut.{k}"),
                &v.to_bit_string_msb_first(),
            )?;
        }

        // posedge at +half_period
        let pos_t = time + stimulus.half_period;
        writer.change(&mut f, pos_t, &format!("tb.{}", m.clk_name), "1")?;
        writer.change(&mut f, pos_t, &format!("tb.dut.{}", m.clk_name), "1")?;

        // Compute next state and update outputs (state regs are outputs in v1 tests).
        let next = m.step(&inputs, &state)?;
        for (name, v) in &next {
            writer.change(
                &mut f,
                pos_t,
                &format!("tb.{name}"),
                &v.to_bit_string_msb_first(),
            )?;
            writer.change(
                &mut f,
                pos_t,
                &format!("tb.dut.{name}"),
                &v.to_bit_string_msb_first(),
            )?;
        }
        state = next;

        // negedge at +2*half_period
        let neg_t = time + stimulus.half_period * 2;
        writer.change(&mut f, neg_t, &format!("tb.{}", m.clk_name), "0")?;
        writer.change(&mut f, neg_t, &format!("tb.dut.{}", m.clk_name), "0")?;
    }
    Ok(())
}
