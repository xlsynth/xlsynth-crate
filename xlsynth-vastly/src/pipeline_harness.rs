// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::CoverageCounters;
use crate::Env;
use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::SourceText;
use crate::Value4;
use crate::combo_eval::eval_combo_seeded;
use crate::combo_eval::eval_combo_seeded_with_coverage;
use crate::combo_eval::plan_combo_eval;
use crate::compiled_module::State;
use crate::vcd_writer::VcdWriter;
use std::io::Write;

#[derive(Debug, Clone)]
pub struct PipelineStimulus {
    pub half_period: u64,
    pub cycles: Vec<PipelineCycle>,
}

#[derive(Debug, Clone)]
pub struct PipelineCycle {
    pub inputs: BTreeMap<String, Value4>,
}

pub fn step_pipeline_state_with_env(
    m: &crate::pipeline_compile::CompiledPipelineModule,
    env: &Env,
    state: &State,
) -> Result<State> {
    if m.seqs.is_empty() {
        return Ok(state.clone());
    }
    let mut next = state.clone();
    for seq in &m.seqs {
        let seq_next = crate::module_eval::step_module_with_env(seq, env, state)?;
        for reg in &seq.state_regs {
            if let Some(v) = seq_next.get(reg) {
                next.insert(reg.clone(), v.clone());
            }
        }
    }
    Ok(next)
}

pub fn run_pipeline_and_write_vcd(
    m: &crate::pipeline_compile::CompiledPipelineModule,
    stimulus: &PipelineStimulus,
    initial_state: &State,
    out_path: &std::path::Path,
) -> Result<()> {
    if stimulus.cycles.is_empty() {
        return Err(Error::Parse("no cycles provided".to_string()));
    }
    if stimulus.half_period == 0 {
        return Err(Error::Parse("half_period must be > 0".to_string()));
    }

    let plan = plan_combo_eval(&m.combo)?;

    let mut writer = VcdWriter::new("1ns");
    let port_names = collect_port_names(m);
    for (name, info) in &m.combo.decls {
        if port_names.contains(name) {
            writer.add_var(&format!("tb.{name}"), info.width)?;
            writer.add_var(&format!("tb.dut.{name}"), info.width)?;
        } else {
            writer.add_var(&format!("tb.dut.{name}"), info.width)?;
        }
    }

    let mut f =
        std::fs::File::create(out_path).map_err(|e| Error::Parse(format!("io error: {e}")))?;
    writer.write_header(&mut f)?;

    let mut state = initial_state.clone();

    // Time 0: clk low, drive cycle0 inputs, then compute and dump.
    write_clk(&mut writer, &mut f, 0, &m.clk_name, "0")?;
    if let Some(c0) = stimulus.cycles.first() {
        let seed = build_seed_env(m, &state, &c0.inputs, LogicLevel::Low)?;
        let values = eval_combo_seeded(&m.combo, &plan, &seed)?;
        write_values(&mut writer, &mut f, 0, &values, &port_names)?;
    }

    for (cyc_idx, cyc) in stimulus.cycles.iter().enumerate() {
        let base_t = (cyc_idx as u64) * stimulus.half_period * 2;
        // Keep clock low at the start of cycle.
        write_clk(&mut writer, &mut f, base_t, &m.clk_name, "0")?;

        if cyc_idx != 0 {
            // Apply inputs at base+1 (stable before posedge).
            let input_t = base_t + 1;
            let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::Low)?;
            let values = eval_combo_seeded(&m.combo, &plan, &seed)?;
            write_values(&mut writer, &mut f, input_t, &values, &port_names)?;
        }

        // posedge at base+half_period
        let pos_t = base_t + stimulus.half_period;
        write_clk(&mut writer, &mut f, pos_t, &m.clk_name, "1")?;

        // Evaluate combo with clk=1 and current state before state update.
        {
            let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::High)?;
            let values = eval_combo_seeded(&m.combo, &plan, &seed)?;
            write_values(&mut writer, &mut f, pos_t, &values, &port_names)?;
            // SIMULATION observers fire on posedge (pre-state-update).
            exec_observers(
                m,
                cyc_idx as u64,
                &values,
                None,
                &mut std::io::stdout().lock(),
            )?;
        }

        // Apply state update (if any) and then recompute combo with new state.
        if !m.seqs.is_empty() {
            let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::High)?;
            let values = eval_combo_seeded(&m.combo, &plan, &seed)?;
            let env = env_from_values(&values);
            state = step_pipeline_state_with_env(m, &env, &state)?;
            let seed2 = build_seed_env(m, &state, &cyc.inputs, LogicLevel::High)?;
            let values2 = eval_combo_seeded(&m.combo, &plan, &seed2)?;
            write_values(&mut writer, &mut f, pos_t, &values2, &port_names)?;
        }

        // negedge at base+2*half_period
        let neg_t = base_t + stimulus.half_period * 2;
        write_clk(&mut writer, &mut f, neg_t, &m.clk_name, "0")?;
        let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::Low)?;
        let values = eval_combo_seeded(&m.combo, &plan, &seed)?;
        write_values(&mut writer, &mut f, neg_t, &values, &port_names)?;
    }

    Ok(())
}

pub fn run_pipeline_and_collect_coverage(
    m: &crate::pipeline_compile::CompiledPipelineModule,
    stimulus: &PipelineStimulus,
    initial_state: &State,
    src: &SourceText,
    cov: &mut CoverageCounters,
) -> Result<()> {
    if stimulus.cycles.is_empty() {
        return Err(Error::Parse("no cycles provided".to_string()));
    }
    let plan = plan_combo_eval(&m.combo)?;
    let skip_toggle_names = literal_assigned_names(m);
    let mut state = initial_state.clone();
    let mut last_values: BTreeMap<String, Value4> = BTreeMap::new();

    // Time 0.
    if let Some(c0) = stimulus.cycles.first() {
        let seed = build_seed_env(m, &state, &c0.inputs, LogicLevel::Low)?;
        let values = eval_combo_seeded_with_coverage(&m.combo, &plan, &seed, src, cov, &m.fn_meta)?;
        update_toggles(cov, &last_values, &values, &skip_toggle_names);
        last_values = values;
    }

    for (cyc_idx, cyc) in stimulus.cycles.iter().enumerate() {
        let base_t = (cyc_idx as u64) * stimulus.half_period * 2;
        let _ = base_t;

        if cyc_idx != 0 {
            let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::Low)?;
            let values =
                eval_combo_seeded_with_coverage(&m.combo, &plan, &seed, src, cov, &m.fn_meta)?;
            update_toggles(cov, &last_values, &values, &skip_toggle_names);
            last_values = values;
        }

        // Posedge evaluation (clk high).
        {
            let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::High)?;
            let values =
                eval_combo_seeded_with_coverage(&m.combo, &plan, &seed, src, cov, &m.fn_meta)?;
            update_toggles(cov, &last_values, &values, &skip_toggle_names);
            last_values = values;
            exec_observers(
                m,
                cyc_idx as u64,
                &last_values,
                Some((src, cov)),
                &mut std::io::stdout().lock(),
            )?;
        }

        if !m.seqs.is_empty() {
            for s in &m.seq_spans {
                cov.hit_span(src, *s);
            }
            let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::High)?;
            let values = eval_combo_seeded(&m.combo, &plan, &seed)?;
            let env = env_from_values(&values);
            state = step_pipeline_state_with_env(m, &env, &state)?;

            // Recompute with updated state.
            let seed2 = build_seed_env(m, &state, &cyc.inputs, LogicLevel::High)?;
            let values2 =
                eval_combo_seeded_with_coverage(&m.combo, &plan, &seed2, src, cov, &m.fn_meta)?;
            update_toggles(cov, &last_values, &values2, &skip_toggle_names);
            last_values = values2;
        }

        // Negedge evaluation (clk low).
        {
            let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::Low)?;
            let values =
                eval_combo_seeded_with_coverage(&m.combo, &plan, &seed, src, cov, &m.fn_meta)?;
            update_toggles(cov, &last_values, &values, &skip_toggle_names);
            last_values = values;
        }
    }
    Ok(())
}

pub fn run_pipeline_and_collect_outputs(
    m: &crate::pipeline_compile::CompiledPipelineModule,
    stimulus: &PipelineStimulus,
    initial_state: &State,
) -> Result<Vec<BTreeMap<String, Value4>>> {
    if stimulus.cycles.is_empty() {
        return Err(Error::Parse("no cycles provided".to_string()));
    }
    let plan = plan_combo_eval(&m.combo)?;
    let mut state = initial_state.clone();
    let mut out = Vec::with_capacity(stimulus.cycles.len());

    for cyc in &stimulus.cycles {
        if !m.seqs.is_empty() {
            let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::High)?;
            let values = eval_combo_seeded(&m.combo, &plan, &seed)?;
            let env = env_from_values(&values);
            state = step_pipeline_state_with_env(m, &env, &state)?;
        }

        let seed = build_seed_env(m, &state, &cyc.inputs, LogicLevel::Low)?;
        let values = eval_combo_seeded(&m.combo, &plan, &seed)?;
        let mut cycle_out = BTreeMap::new();
        for p in &m.combo.output_ports {
            if let Some(v) = values.get(&p.name) {
                cycle_out.insert(p.name.clone(), v.clone());
            }
        }
        out.push(cycle_out);
    }

    Ok(out)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum LogicLevel {
    Low,
    High,
}

fn collect_port_names(m: &crate::pipeline_compile::CompiledPipelineModule) -> BTreeSet<String> {
    let mut s: BTreeSet<String> = BTreeSet::new();
    s.insert(m.clk_name.clone());
    for p in &m.combo.input_ports {
        s.insert(p.name.clone());
    }
    for p in &m.combo.output_ports {
        s.insert(p.name.clone());
    }
    s
}

fn build_seed_env(
    m: &crate::pipeline_compile::CompiledPipelineModule,
    state: &State,
    inputs: &BTreeMap<String, Value4>,
    clk_level: LogicLevel,
) -> Result<Env> {
    let mut seed = Env::new();
    for (k, v) in state {
        seed.insert(k.clone(), v.clone());
    }
    for (k, v) in inputs {
        seed.insert(k.clone(), v.clone());
    }

    let clk_v = match clk_level {
        LogicLevel::Low => Value4::new(1, Signedness::Unsigned, vec![LogicBit::Zero]),
        LogicLevel::High => Value4::new(1, Signedness::Unsigned, vec![LogicBit::One]),
    };
    seed.insert(m.clk_name.clone(), clk_v);
    Ok(seed)
}

fn env_from_values(values: &BTreeMap<String, Value4>) -> Env {
    let mut env = Env::new();
    for (k, v) in values {
        env.insert(k.clone(), v.clone());
    }
    env
}

fn exec_observers<W: Write>(
    m: &crate::pipeline_compile::CompiledPipelineModule,
    cycle_idx: u64,
    values: &BTreeMap<String, Value4>,
    cov_ctx: Option<(&SourceText, &mut CoverageCounters)>,
    w: &mut W,
) -> Result<()> {
    if m.observers.is_empty() {
        return Ok(());
    }
    let mut env = Env::new();
    for (k, v) in &m.combo.consts {
        env.insert(k.clone(), v.clone());
    }
    for (k, v) in values {
        env.insert(k.clone(), v.clone());
    }
    for obs in &m.observers {
        if obs.clk_name != m.clk_name {
            continue;
        }
        if let Some(line) = obs.eval_and_format(&env)? {
            writeln!(w, "[cycle {cycle_idx}] {line}").ok();
        }
    }

    // Line coverage: if SIMULATION observers are present, the always_ff blocks are
    // being evaluated each posedge. Mark their spans as hit (even when the
    // condition is false).
    if let Some((src, cov)) = cov_ctx {
        for s in &m.observer_spans {
            cov.hit_span(src, *s);
        }
    }
    Ok(())
}

fn write_clk<W: std::io::Write>(
    writer: &mut VcdWriter,
    w: &mut W,
    time: u64,
    clk_name: &str,
    value: &str,
) -> Result<()> {
    writer.change(&mut *w, time, &format!("tb.{clk_name}"), value)?;
    writer.change(&mut *w, time, &format!("tb.dut.{clk_name}"), value)?;
    Ok(())
}

fn write_values<W: std::io::Write>(
    writer: &mut VcdWriter,
    w: &mut W,
    time: u64,
    values: &BTreeMap<String, Value4>,
    port_names: &BTreeSet<String>,
) -> Result<()> {
    for (name, v) in values {
        if port_names.contains(name) {
            writer.change(
                &mut *w,
                time,
                &format!("tb.{name}"),
                &v.to_bit_string_msb_first(),
            )?;
            writer.change(
                &mut *w,
                time,
                &format!("tb.dut.{name}"),
                &v.to_bit_string_msb_first(),
            )?;
        } else {
            writer.change(
                &mut *w,
                time,
                &format!("tb.dut.{name}"),
                &v.to_bit_string_msb_first(),
            )?;
        }
    }
    Ok(())
}

fn update_toggles(
    cov: &mut CoverageCounters,
    last: &BTreeMap<String, Value4>,
    cur: &BTreeMap<String, Value4>,
    skip_names: &BTreeSet<String>,
) {
    for (name, v) in cur {
        if skip_names.contains(name) {
            continue;
        }
        cov.observe_toggles(last.get(name), v, name);
    }
}

fn literal_assigned_names(m: &crate::pipeline_compile::CompiledPipelineModule) -> BTreeSet<String> {
    let mut s: BTreeSet<String> = BTreeSet::new();
    for a in &m.combo.assigns {
        let rhs_spanned = a
            .rhs_spanned
            .as_ref()
            .expect("coverage registration requires spanned assign expressions");
        match &rhs_spanned.kind {
            crate::ast_spanned::SpannedExprKind::Literal(_) => {
                s.insert(a.lhs_base().to_string());
            }
            crate::ast_spanned::SpannedExprKind::UnsizedNumber(_) => {
                s.insert(a.lhs_base().to_string());
            }
            crate::ast_spanned::SpannedExprKind::UnbasedUnsized(_) => {
                s.insert(a.lhs_base().to_string());
            }
            _ => {}
        }
    }
    s
}
