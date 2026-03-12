// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::ArgAction;
use clap::Parser;

use std::collections::BTreeSet;
use xlsynth_vastly::CoverageCounters;
use xlsynth_vastly::LogicBit;
use xlsynth_vastly::PipelineCycle;
use xlsynth_vastly::PipelineStimulus;
use xlsynth_vastly::Signedness;
use xlsynth_vastly::SourceText;
use xlsynth_vastly::Value4;
use xlsynth_vastly::compile_pipeline_module_with_defines;
use xlsynth_vastly::compute_coverability_or_fallback_with_defines;
use xlsynth_vastly::cycles_from_irvals_file;
use xlsynth_vastly::eval_combo_seeded;
use xlsynth_vastly::plan_combo_eval;
use xlsynth_vastly::render_annotated_source;
use xlsynth_vastly::run_pipeline_and_collect_coverage;
use xlsynth_vastly::run_pipeline_and_write_vcd;
use xlsynth_vastly::step_pipeline_state_with_env;

#[derive(Parser, Debug)]
#[command(name = "vastly-sim-pipeline")]
#[command(about = "Run a clocked SV module over scheduled input vectors and dump a VCD", long_about = None)]
struct Args {
    /// Path to the .sv file.
    sv: PathBuf,

    /// Scheduled input vectors: `<vals>@<cycle>`, semicolon-separated entries.
    /// Example: "0xf00,0xba5@2;0x0000,0xf800@3"
    #[arg(long)]
    inputs: Option<String>,

    /// Read cycle-by-cycle inputs from an XLS `.irvals` file (one value per
    /// line).
    #[arg(long = "input-irvals")]
    input_irvals: Option<PathBuf>,

    /// Clock half-period in ns for the generated VCD.
    #[arg(long, default_value_t = 5)]
    half_period: u64,

    /// Total number of cycles to run. If omitted, inferred from max scheduled
    /// tick + 1.
    #[arg(long)]
    cycles: Option<u64>,

    /// Output VCD path.
    #[arg(long, default_value = "pipeline.vcd")]
    vcd_out: PathBuf,

    /// Collect and print coverage (line, ternary decision, toggle) instead of
    /// emitting a VCD.
    #[arg(long)]
    coverage: bool,

    /// Print the original SV source annotated with coverage (per-line + ternary
    /// branch segments).
    #[arg(long)]
    coverage_annotate: bool,

    /// Print top-level output port values at the start of each cycle (clk low).
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    print_outputs: bool,

    /// Define a preprocessor symbol (repeatable), used for `` `ifdef NAME``
    /// blocks.
    #[arg(long = "define")]
    defines: Vec<String>,

    /// Convenience for `--define SIMULATION`.
    #[arg(long)]
    simulation: bool,
}

fn main() {
    if let Err(e) = main_inner() {
        eprintln!("error: {e:?}");
        std::process::exit(1);
    }
}

fn main_inner() -> xlsynth_vastly::Result<()> {
    let args = Args::parse();
    let src = std::fs::read_to_string(&args.sv)
        .map_err(|e| xlsynth_vastly::Error::Parse(format!("io error: {e}")))?;

    let mut defines: BTreeSet<String> = args.defines.iter().cloned().collect();
    if args.simulation {
        defines.insert("SIMULATION".to_string());
    }
    let m = compile_pipeline_module_with_defines(&src, &defines)?;
    let src_text = SourceText::new(src.clone());

    let vectors = match (&args.inputs, &args.input_irvals) {
        (Some(_), Some(_)) => {
            return Err(xlsynth_vastly::Error::Parse(
                "provide exactly one of --inputs or --input-irvals".to_string(),
            ));
        }
        (None, None) => {
            return Err(xlsynth_vastly::Error::Parse(
                "missing required input: --inputs or --input-irvals".to_string(),
            ));
        }
        (Some(inputs), None) => {
            let (cycles, schedule) =
                parse_scheduled_inputs(inputs, m.combo.input_ports.len(), args.cycles)?;
            build_cycle_vectors(&m, cycles as usize, schedule)?
        }
        (None, Some(p)) => cycles_from_irvals_file(&m, p, args.cycles)?,
    };

    if args.print_outputs {
        let plan = plan_combo_eval(&m.combo)?;
        let mut state = m.initial_state_x();
        for (cyc_idx, cyc) in vectors.iter().enumerate() {
            let mut seed = xlsynth_vastly::Env::new();
            for (k, v) in state.iter() {
                seed.insert(k.clone(), v.clone());
            }
            for (k, v) in cyc.inputs.iter() {
                seed.insert(k.clone(), v.clone());
            }
            seed.insert(
                m.clk_name.clone(),
                Value4::new(1, Signedness::Unsigned, vec![LogicBit::Zero]),
            );
            let values = eval_combo_seeded(&m.combo, &plan, &seed)?;
            print!("cycle[{cyc_idx}]");
            for p in &m.combo.output_ports {
                let v = values.get(&p.name).ok_or_else(|| {
                    xlsynth_vastly::Error::Parse(format!(
                        "no value computed for output `{}`",
                        p.name
                    ))
                })?;
                print!(" {}={}", p.name, format_value(v));
            }
            println!();

            if !m.seqs.is_empty() {
                let mut seed2 = seed.clone();
                seed2.insert(
                    m.clk_name.clone(),
                    Value4::new(1, Signedness::Unsigned, vec![LogicBit::One]),
                );
                let values2 = eval_combo_seeded(&m.combo, &plan, &seed2)?;
                let mut env = xlsynth_vastly::Env::new();
                for (k, v) in values2.iter() {
                    env.insert(k.clone(), v.clone());
                }
                state = step_pipeline_state_with_env(&m, &env, &state)?;
            }
        }
    }

    let stimulus = PipelineStimulus {
        half_period: args.half_period,
        cycles: vectors,
    };
    let init = m.initial_state_x();

    if args.coverage {
        let mut cov = CoverageCounters::default();
        cov.defines = defines.clone();
        for a in &m.combo.assigns {
            cov.register_ternaries_from_spanned_expr(
                a.rhs_spanned
                    .as_ref()
                    .expect("coverage registration requires spanned assign expressions"),
            );
        }
        cov.register_functions(&m.fn_meta);
        let cover = compute_coverability_or_fallback_with_defines(&src_text, &defines);
        run_pipeline_and_collect_coverage(&m, &stimulus, &init, &src_text, &mut cov)?;
        print_coverage_report(&cover, &src_text, &cov);
        if args.coverage_annotate {
            let ansi = std::env::var_os("NO_COLOR").is_none();
            println!("annotated_source:");
            print!("{}", render_annotated_source(&src_text, &cov, ansi));
        }
    } else {
        run_pipeline_and_write_vcd(&m, &stimulus, &init, &args.vcd_out)?;
    }
    Ok(())
}

fn format_value(v: &Value4) -> String {
    if let Some(hex) = v.to_hex_string_if_known() {
        if let Some(dec) = v.to_decimal_string_if_known() {
            return format!(
                "0x{hex} (dec={dec}, {}'b{})",
                v.width,
                v.to_bit_string_msb_first()
            );
        }
        return format!("0x{hex} ({}'b{})", v.width, v.to_bit_string_msb_first());
    }
    format!("{}'b{}", v.width, v.to_bit_string_msb_first())
}

fn parse_scheduled_inputs(
    s: &str,
    expected_arity: usize,
    cycles_override: Option<u64>,
) -> xlsynth_vastly::Result<(u64, BTreeMap<u64, Vec<String>>)> {
    let mut out: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    let mut max_tick: Option<u64> = None;

    let entries = s.split(';').map(|x| x.trim()).filter(|x| !x.is_empty());
    for (entry_idx, e) in entries.enumerate() {
        let (vals_s, tick_s) = e.split_once('@').ok_or_else(|| {
            xlsynth_vastly::Error::Parse(format!("entry {entry_idx}: missing `@tick`"))
        })?;
        let tick: u64 = tick_s.trim().parse().map_err(|e| {
            xlsynth_vastly::Error::Parse(format!("entry {entry_idx}: bad tick `{tick_s}`: {e}"))
        })?;
        let vals: Vec<String> = vals_s
            .split(',')
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .map(|x| x.to_string())
            .collect();
        if vals.len() != expected_arity {
            return Err(xlsynth_vastly::Error::Parse(format!(
                "entry {entry_idx} @ {tick}: got {} values, expected {}",
                vals.len(),
                expected_arity
            )));
        }
        if out.insert(tick, vals).is_some() {
            return Err(xlsynth_vastly::Error::Parse(format!(
                "duplicate schedule for cycle {tick}"
            )));
        }
        max_tick = Some(max_tick.map(|m| m.max(tick)).unwrap_or(tick));
    }
    let max_tick = max_tick.ok_or_else(|| {
        xlsynth_vastly::Error::Parse("no entries parsed from --inputs".to_string())
    })?;
    let cycles = cycles_override.unwrap_or(max_tick + 1);
    if let Some(co) = cycles_override {
        if co <= max_tick {
            return Err(xlsynth_vastly::Error::Parse(format!(
                "--cycles={co} is too small for max scheduled tick {max_tick}"
            )));
        }
    }
    if cycles == 0 {
        return Err(xlsynth_vastly::Error::Parse(
            "cycles must be > 0".to_string(),
        ));
    }
    Ok((cycles, out))
}

fn build_cycle_vectors(
    m: &xlsynth_vastly::CompiledPipelineModule,
    cycles: usize,
    schedule: BTreeMap<u64, Vec<String>>,
) -> xlsynth_vastly::Result<Vec<PipelineCycle>> {
    let mut out: Vec<PipelineCycle> = Vec::with_capacity(cycles);
    for cyc in 0..cycles {
        let mut map: BTreeMap<String, Value4> = BTreeMap::new();
        // Default: drive all inputs to zero.
        for p in &m.combo.input_ports {
            let info = m.combo.decls.get(&p.name).ok_or_else(|| {
                xlsynth_vastly::Error::Parse(format!("no decl for input `{}`", p.name))
            })?;
            map.insert(p.name.clone(), Value4::zeros(info.width, info.signedness));
        }
        if let Some(vals) = schedule.get(&(cyc as u64)) {
            for (p, n) in m.combo.input_ports.iter().zip(vals.iter()) {
                let info = m.combo.decls.get(&p.name).ok_or_else(|| {
                    xlsynth_vastly::Error::Parse(format!("no decl for input `{}`", p.name))
                })?;
                let v = Value4::parse_numeric_token(info.width, info.signedness, n)?;
                map.insert(p.name.clone(), v);
            }
        }
        out.push(PipelineCycle { inputs: map });
    }
    Ok(out)
}

fn print_coverage_report(
    cover: &xlsynth_vastly::CoverabilityMap,
    src: &SourceText,
    cov: &CoverageCounters,
) {
    let mut coverable_total: u64 = 0;
    let mut coverable_hit: u64 = 0;
    for l in 1..=src.line_count() {
        if cover.is_coverable(l) {
            coverable_total += 1;
            if cov.line_hits.contains_key(&l) {
                coverable_hit += 1;
            }
        }
    }
    let pct = if coverable_total == 0 {
        100.0
    } else {
        (coverable_hit as f64) * 100.0 / (coverable_total as f64)
    };
    println!("coverage:");
    println!("  line: {coverable_hit}/{coverable_total} ({pct:.2}%)");

    let mut ternary_total: u64 = 0;
    let mut ternary_hit: u64 = 0;
    for (_k, c) in &cov.ternary_branches {
        ternary_total += 1;
        if c.t_taken != 0 || c.f_taken != 0 || c.cond_unknown != 0 {
            ternary_hit += 1;
        }
    }
    println!("  ternary: {ternary_hit}/{ternary_total} hit");

    let mut toggle_signals_nonzero: Vec<(String, u64)> = Vec::new();
    let mut toggle_signals_zero_names: Vec<String> = Vec::new();
    for (name, per_bit) in &cov.toggle_counts {
        let mut sum: u64 = 0;
        for c in per_bit {
            sum += *c;
        }
        if sum == 0 {
            toggle_signals_zero_names.push(name.clone());
        } else {
            toggle_signals_nonzero.push((name.clone(), sum));
        }
    }
    // Least toggles are typically most actionable (stuck/rarely-changing nets).
    toggle_signals_nonzero.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let n = 10usize.min(toggle_signals_nonzero.len());
    toggle_signals_zero_names.sort();
    println!(
        "  toggles: {} signals toggled ({} signals never toggled; showing bottom {n} non-zero)",
        toggle_signals_nonzero.len(),
        toggle_signals_zero_names.len()
    );
    if !toggle_signals_zero_names.is_empty() {
        println!("    never_toggled:");
        for name in &toggle_signals_zero_names {
            println!("      {name}");
        }
    }
    for (name, sum) in toggle_signals_nonzero.into_iter().take(n) {
        println!("    {name}: {sum}");
    }
}
