// SPDX-License-Identifier: Apache-2.0

//! Register-boundary adapter for Yosys-free, external-mapper experiments.

use anyhow::{Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use prost::Message;
use std::path::PathBuf;
use xlsynth_g8r::aig::ChoiceAig;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_serdes::g8r::load_sequential_gate_fn_from_path;
use xlsynth_g8r::liberty::library_to_proto;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::{load_liberty_with_timing_data_from_path, parse_netlist_from_path};
use xlsynth_g8r::netlist::sta::StaOptions;
use xlsynth_g8r::techmap::{
    MappedNetlist, SequentialTechMapConstraints, TechMapStats, prepare_fixed_flip_flop_transition,
    restore_fixed_flip_flop_boundary, restrict_mapping_flip_flop,
};

#[derive(Clone, Copy, ValueEnum)]
enum Operation {
    Prepare,
    Restore,
    Inspect,
    FilterLibrary,
}

#[derive(Parser)]
struct Args {
    #[arg(value_enum)]
    operation: Operation,
    #[arg(long)]
    design: PathBuf,
    #[arg(long)]
    liberty_proto: PathBuf,
    #[arg(long)]
    cell: String,
    #[arg(long)]
    netlist: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0.01)]
    input_transition: f64,
    #[arg(long, default_value_t = 2.308)]
    output_load: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let design = load_sequential_gate_fn_from_path(&args.design).map_err(|e| anyhow!(e))?;
    let mut library = load_liberty_with_timing_data_from_path(&args.liberty_proto)?;
    let options = StaOptions {
        primary_input_transition: args.input_transition,
        module_output_load: args.output_load,
        ..StaOptions::default()
    };
    match args.operation {
        Operation::Inspect => {
            let inputs = design
                .inputs
                .iter()
                .map(|id| {
                    let port = &design.transition.inputs[id.index()];
                    serde_json::json!({"name": port.name, "width": port.get_bit_count()})
                })
                .collect::<Vec<_>>();
            let cell = library
                .cells
                .iter()
                .find(|c| c.name == args.cell)
                .ok_or_else(|| anyhow!("unknown cell"))?;
            let pins = cell
                .pins
                .iter()
                .map(|pin| {
                    serde_json::json!({
                        "name": library.resolve_string(&pin.name),
                        "capacitance": pin.capacitance,
                    })
                })
                .collect::<Vec<_>>();
            std::fs::write(
                &args.output,
                serde_json::to_vec_pretty(&serde_json::json!({
                    "inputs": inputs, "cell": args.cell, "pins": pins,
                    "register_bits": design.registers.iter().map(|r| design.transition.inputs[r.q.index()].get_bit_count()).sum::<usize>(),
                }))?,
            )?;
        }
        Operation::FilterLibrary => {
            restrict_mapping_flip_flop(&mut library, &args.cell)?;
            std::fs::write(&args.output, library_to_proto(library)?.encode_to_vec())?;
        }
        Operation::Prepare => {
            let choices = ChoiceAig::without_choices(design.transition.clone());
            let phased = prepare_fixed_flip_flop_transition(
                &design, &choices, &library, &args.cell, options,
            )?;
            std::fs::write(
                &args.output,
                emit_aiger_binary(phased.graph(), true).map_err(|e| anyhow!(e))?,
            )?;
        }
        Operation::Restore => {
            let path = args
                .netlist
                .ok_or_else(|| anyhow!("restore requires --netlist"))?;
            let mut parsed = parse_netlist_from_path(&path)?;
            if parsed.modules.len() != 1 {
                bail!("expected one mapped transition module");
            }
            let mut mapped = MappedNetlist {
                module: parsed.modules.remove(0),
                nets: parsed.nets,
                interner: parsed.interner,
                stats: TechMapStats::default(),
            };
            mapped.module.name = mapped.interner.get_or_intern(&design.name);
            restore_fixed_flip_flop_boundary(
                &mut mapped,
                &design,
                &library,
                &args.cell,
                &SequentialTechMapConstraints::default(),
                options,
            )?;
            std::fs::write(
                &args.output,
                emit_module_as_netlist_text(&mapped.module, &mapped.nets, &mapped.interner)?,
            )?;
        }
    }
    Ok(())
}
