// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::ArgAction;
use clap::Parser;

use xlsynth_vastly::Signedness;
use xlsynth_vastly::Value4;
use xlsynth_vastly::Vcd;
use xlsynth_vastly::VcdDiffOptions;
use xlsynth_vastly::compile_combo_module;
use xlsynth_vastly::diff_vcd_exact;
use xlsynth_vastly::eval_combo;
use xlsynth_vastly::plan_combo_eval;
use xlsynth_vastly::run_combo_and_write_vcd;
use xlsynth_vastly::run_iverilog_combo_and_collect_vcd;

#[derive(Parser, Debug)]
#[command(name = "vastly-sim-combo")]
#[command(about = "Run a *.combo.v over input vectors and dump a VCD", long_about = None)]
struct Args {
    /// Path to the *.combo.v file.
    combo_v: PathBuf,

    /// Input vectors: CSV values, semicolon-separated batches. Example:
    /// "0xf00,0xba5;0x000,0x001"
    #[arg(long)]
    inputs: String,

    /// Output VCD path.
    #[arg(long, default_value = "combo.vcd")]
    vcd_out: PathBuf,

    /// Compare our VCD to Icarus Verilog (iverilog/vvp) by semantic VCD diff.
    #[arg(long)]
    compare_to_iverilog: bool,

    /// Print top-level output port values for each input vector to stdout.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    print_outputs: bool,
}

fn main() {
    if let Err(e) = main_inner() {
        eprintln!("error: {e:?}");
        std::process::exit(1);
    }
}

fn main_inner() -> xlsynth_vastly::Result<()> {
    let args = Args::parse();
    let src = std::fs::read_to_string(&args.combo_v)
        .map_err(|e| xlsynth_vastly::Error::Parse(format!("io error: {e}")))?;

    let m = compile_combo_module(&src)?;
    let plan = plan_combo_eval(&m)?;

    let vectors = parse_vectors(&args.inputs, &m)?;

    if args.print_outputs {
        for (vec_idx, vec_inputs) in vectors.iter().enumerate() {
            let values = eval_combo(&m, &plan, vec_inputs)?;
            print!("vec[{vec_idx}]");
            for p in &m.output_ports {
                let v = values.get(&p.name).ok_or_else(|| {
                    xlsynth_vastly::Error::Parse(format!(
                        "no value computed for output `{}`",
                        p.name
                    ))
                })?;
                print!(" {}={}", p.name, format_value(v));
            }
            println!();
        }
    }

    run_combo_and_write_vcd(&m, &plan, &vectors, &args.vcd_out)?;

    if args.compare_to_iverilog {
        let td = mk_temp_dir()?;
        let iv_vcd = td.join("iverilog.vcd");
        run_iverilog_combo_and_collect_vcd(&args.combo_v, &m, &vectors, &iv_vcd)?;

        let ours_text = std::fs::read_to_string(&args.vcd_out)
            .map_err(|e| xlsynth_vastly::Error::Parse(format!("io error: {e}")))?;
        let iv_text = std::fs::read_to_string(&iv_vcd)
            .map_err(|e| xlsynth_vastly::Error::Parse(format!("io error: {e}")))?;
        let ours = Vcd::parse(&ours_text)?;
        let iv = Vcd::parse(&iv_text)?;
        diff_vcd_exact(&ours, &iv, &VcdDiffOptions::default())?;
    }

    Ok(())
}

fn format_value(v: &Value4) -> String {
    if let Some(hex) = v.to_hex_string_if_known() {
        return format!("0x{hex} ({}'b{})", v.width, v.to_bit_string_msb_first());
    }
    format!("{}'b{}", v.width, v.to_bit_string_msb_first())
}

fn parse_vectors(
    inputs: &str,
    m: &xlsynth_vastly::CompiledComboModule,
) -> xlsynth_vastly::Result<Vec<BTreeMap<String, Value4>>> {
    let mut out: Vec<BTreeMap<String, Value4>> = Vec::new();
    let vec_strs = inputs
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty());
    for (vec_idx, vs) in vec_strs.enumerate() {
        let items: Vec<&str> = vs
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if items.len() != m.input_ports.len() {
            return Err(xlsynth_vastly::Error::Parse(format!(
                "vector {vec_idx}: got {} inputs, expected {}",
                items.len(),
                m.input_ports.len()
            )));
        }
        let mut map: BTreeMap<String, Value4> = BTreeMap::new();
        for (tok, port) in items.iter().zip(m.input_ports.iter()) {
            let v = Value4::parse_numeric_token(port.width, Signedness::Unsigned, tok)?;
            map.insert(port.name.clone(), v);
        }
        out.push(map);
    }
    if out.is_empty() {
        return Err(xlsynth_vastly::Error::Parse(
            "no vectors parsed from --inputs".to_string(),
        ));
    }
    Ok(out)
}

fn mk_temp_dir() -> xlsynth_vastly::Result<std::path::PathBuf> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0u32..1000u32 {
        let p = base.join(format!("vastly_sim_combo_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return Ok(p),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(xlsynth_vastly::Error::Parse(format!(
                    "create temp dir: {e:?}"
                )));
            }
        }
    }
    Err(xlsynth_vastly::Error::Parse(
        "failed to create temp dir".to_string(),
    ))
}
