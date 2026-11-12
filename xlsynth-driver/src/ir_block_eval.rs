// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::io::{self, Write};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, ArgAction, ArgGroup, ArgMatches, Command, value_parser};
use xlsynth_pir::block_eval::{BlockEvalError, BlockEvalField, BlockEvaluator, RegisterInit};
use xlsynth_pir::ir::{MemberType, Package, PackageMember, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_values::parse_ir_values_with_fields;
use xlsynth_pir::{IrValue, NamedIrValueSet};

/// Defines a cycle-based PIR evaluator with explicit register initialization.
pub fn command() -> Command {
    Command::new("ir-block-eval")
        .about("Evaluate a PIR block for one or more clock cycles")
        .arg(
            Arg::new("ir_file")
                .value_name("BLOCK_IR")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("top")
                .long("top")
                .value_name("BLOCK")
                .help("Override the package top"),
        )
        .arg(
            Arg::new("arg_tuple")
                .value_name("ARG_TUPLE")
                .index(2)
                .help("Typed tuple of external inputs for one cycle"),
        )
        .arg(
            Arg::new("input_irvals")
                .long("input-irvals")
                .value_name("PATH")
                .help("Read a positional tuple or named input record per cycle"),
        )
        .arg(
            Arg::new("zeros")
                .long("initial-state-all-zeros")
                .action(ArgAction::SetTrue)
                .help("Initialize every register to its type's zero"),
        )
        .arg(
            Arg::new("seed")
                .long("initial-state-random-seed")
                .value_name("U64")
                .value_parser(value_parser!(u64))
                .help("Initialize every register reproducibly from the given seed"),
        )
        .arg(
            Arg::new("initial_file")
                .long("initial-state-file")
                .value_name("PATH")
                .help("Read initial registers from one named or positional .irvals record"),
        )
        .arg(
            Arg::new("final_file")
                .long("final-state-irvals")
                .value_name("PATH")
                .help("Save registers after the last cycle as one named .irvals record"),
        )
        .group(
            ArgGroup::new("inputs")
                .args(["arg_tuple", "input_irvals"])
                .required(true),
        )
        .group(ArgGroup::new("init").args(["zeros", "seed", "initial_file"]))
}

/// Selects the declared block top or the only block when there is no block top.
fn select_top<'a>(package: &'a Package, explicit: Option<&'a String>) -> Result<&'a str> {
    if let Some(name) = explicit {
        return Ok(name);
    }
    if let Some((name, MemberType::Block)) = &package.top {
        return Ok(name);
    }
    let mut blocks = package.members.iter().filter_map(|member| match member {
        PackageMember::Block(block) => Some(block.name.as_str()),
        PackageMember::Function(_) => None,
    });
    let first = blocks.next().context("package contains no blocks")?;
    if blocks.next().is_some() {
        bail!("package has multiple blocks and no block top; specify --top");
    }
    Ok(first)
}

/// Binds named records to the evaluator's ordered interface and preserves
/// aggregate values.
fn read_records(path: &str, fields: &[BlockEvalField]) -> Result<Vec<IrValue>> {
    let names = fields
        .iter()
        .map(|field| field.name.clone())
        .collect::<Vec<_>>();
    let text = fs::read_to_string(path).with_context(|| format!("read IR values file {path}"))?;
    parse_ir_values_with_fields(
        &text,
        fields.iter().map(|field| (field.name.as_str(), &field.ty)),
    )
    .and_then(|records| records.into_positional_values(&names))
    .map_err(|e| anyhow!("{path}: {e}"))
}

/// Resolves the command-line startup policy after preparing the selected block.
fn register_init(matches: &ArgMatches, registers: &[BlockEvalField]) -> Result<RegisterInit> {
    let has_policy = matches.get_flag("zeros")
        || matches.contains_id("seed")
        || matches.contains_id("initial_file");
    if registers.is_empty() {
        if has_policy || matches.contains_id("final_file") {
            bail!("register state options are invalid for a block without registers");
        }
        return Ok(RegisterInit::Supplied(Vec::new()));
    }
    if matches.get_flag("zeros") {
        return Ok(RegisterInit::AllZeros);
    }
    if let Some(&seed) = matches.get_one::<u64>("seed") {
        return Ok(RegisterInit::Random { seed });
    }
    if let Some(path) = matches.get_one::<String>("initial_file") {
        let mut records = read_records(path, registers)?;
        if records.len() != 1 {
            bail!(
                "--initial-state-file requires exactly one record, found {}",
                records.len()
            );
        }
        let Some(IrValue::Tuple(values)) = records.pop() else {
            bail!("--initial-state-file requires a tuple or named register record");
        };
        return Ok(RegisterInit::Supplied(values.to_vec()));
    }
    bail!(
        "a block with registers requires --initial-state-all-zeros, --initial-state-random-seed, or --initial-state-file"
    )
}

/// Prints pre-edge outputs for each cycle and optionally saves the last
/// committed state.
pub fn run(matches: &ArgMatches) -> Result<()> {
    let path = matches
        .get_one::<String>("ir_file")
        .expect("clap requires the IR file");
    let ir = fs::read_to_string(path).with_context(|| format!("read block IR from {path}"))?;
    let package = Parser::new(&ir)
        .parse_package()
        .map_err(|e| anyhow!("parse block IR: {e}"))?;
    let top = select_top(&package, matches.get_one::<String>("top"))?;
    let evaluator = BlockEvaluator::prepare(&package, top)?;
    let init = register_init(matches, evaluator.registers())?;
    let mut state = evaluator.initial_state(init)?;
    let records = if let Some(path) = matches.get_one::<String>("input_irvals") {
        read_records(path, evaluator.inputs())?
    } else {
        let tuple = matches
            .get_one::<String>("arg_tuple")
            .expect("clap requires one input source");
        let input_type = Type::Tuple(
            evaluator
                .inputs()
                .iter()
                .map(|input| Box::new(input.ty.clone()))
                .collect(),
        );
        vec![
            IrValue::parse_typed_with_hint(tuple, Some(&input_type))
                .map_err(|e| anyhow!("invalid input tuple: {e}"))?,
        ]
    };
    let mut stdout = io::stdout().lock();
    for (index, record) in records.into_iter().enumerate() {
        let cycle = index + 1;
        let IrValue::Tuple(inputs) = record else {
            bail!("input cycle {cycle} requires a tuple or named port record");
        };
        let result = evaluator.step(&state, &inputs);
        let traces = match &result {
            Ok(success) => success.trace_messages.as_slice(),
            Err(BlockEvalError::Runtime(failure)) => failure.trace_messages.as_slice(),
            Err(_) => &[],
        };
        for trace in traces {
            eprintln!("block '{top}', cycle {cycle}: {}", trace.message);
        }
        let result = result.with_context(|| format!("block '{top}', cycle {cycle}"))?;
        if let [only] = result.outputs.as_slice() {
            writeln!(stdout, "{only}")?;
        } else {
            writeln!(stdout, "{}", IrValue::Tuple(result.outputs.into()))?;
        }
        state = result.next_state;
    }
    if let Some(path) = matches.get_one::<String>("final_file") {
        let names = evaluator
            .registers()
            .iter()
            .map(|reg| reg.name.clone())
            .collect::<Vec<_>>();
        let tuple = IrValue::make_tuple(state.values());
        let record =
            NamedIrValueSet::from_positional_tuple(&names, &tuple).map_err(|e| anyhow!("{e}"))?;
        fs::write(path, format!("{record}\n"))
            .with_context(|| format!("write final state to {path}"))?;
    }
    Ok(())
}
