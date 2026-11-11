// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::io;
use std::path::Path;
use std::process::{Command, Output};

use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{BlockEvaluator, RegisterInit};
use xlsynth_pir::ir_parser::Parser;

const PIPELINE: &str = r#"package p

block child(clk: clock, rst: bits[1], d: bits[8], out: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg r(bits[8], reset_value=13)
  rst: bits[1] = input_port(name=rst, id=11)
  d: bits[8] = input_port(name=d, id=12)
  q: bits[8] = register_read(register=r, id=13)
  wr: () = register_write(d, register=r, reset=rst, id=14)
  out: () = output_port(q, name=out, id=15)
}

top block pipeline(clk: clock, rst: bits[1], data: bits[8], out: bits[8], delayed: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg r(bits[8], reset_value=7)
  instantiation stage(block=child, kind=block)
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[8] = input_port(name=data, id=2)
  q: bits[8] = register_read(register=r, id=3)
  wr: () = register_write(data, register=r, reset=rst, id=4)
  out: () = output_port(q, name=out, id=5)
  cd: () = instantiation_input(q, instantiation=stage, port_name=d, id=6)
  cr: () = instantiation_input(rst, instantiation=stage, port_name=rst, id=7)
  cq: bits[8] = instantiation_output(instantiation=stage, port_name=out, id=8)
  delayed: () = output_port(cq, name=delayed, id=9)
}
"#;

const MULTI: &str = r#"package p

block empty() {
}

block pair(x: (bits[8], bits[1][2]), y: (bits[8], bits[1][2])) {
  x: (bits[8], bits[1][2]) = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}
"#;

fn run(directory: &Path, args: &[&str]) -> io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .env_remove("RUST_LOG")
        .args(["ir-block-eval", "block.ir"])
        .args(args)
        .output()
}

/// The first cycle reads supplied state even when reset is active. The state
/// file saved at that edge must resume with the same second-cycle outputs.
#[test]
fn reset_and_resume_from_checkpoint() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = temp.path();
    fs::write(dir.join("block.ir"), PIPELINE)?;
    fs::write(
        dir.join("initial.irvals"),
        "{stage.r: bits[8]:32, r: bits[8]:19}\n",
    )?;
    fs::write(
        dir.join("inputs.irvals"),
        "{data: bits[8]:11, rst: bits[1]:1}\n{data: bits[8]:23, rst: bits[1]:0}\n",
    )?;
    let full = run(
        dir,
        &[
            "--initial-state-file",
            "initial.irvals",
            "--input-irvals",
            "inputs.irvals",
            "--final-state-irvals",
            "full.irvals",
        ],
    )?;
    assert!(
        full.status.success(),
        "{}",
        String::from_utf8_lossy(&full.stderr)
    );
    assert_eq!(
        full.stdout,
        include_bytes!("goldens/ir_block_eval_reset_outputs.golden.txt")
    );
    assert_eq!(
        fs::read_to_string(dir.join("full.irvals"))?,
        "{r: bits[8]:23, stage.r: bits[8]:7}\n"
    );
    let first = run(
        dir,
        &[
            "(bits[1]:1, bits[8]:11)",
            "--initial-state-file",
            "initial.irvals",
            "--final-state-irvals",
            "checkpoint.irvals",
        ],
    )?;
    let second = run(
        dir,
        &[
            "(bits[1]:0, bits[8]:23)",
            "--initial-state-file",
            "checkpoint.irvals",
            "--final-state-irvals",
            "resumed.irvals",
        ],
    )?;
    for result in [&first, &second] {
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!([first.stdout, second.stdout].concat(), full.stdout);
    assert_eq!(
        fs::read(dir.join("resumed.irvals"))?,
        fs::read(dir.join("full.irvals"))?
    );
    Ok(())
}

#[test]
fn random_start_can_be_replayed_and_zero_start_is_explicit() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = temp.path();
    fs::write(dir.join("block.ir"), PIPELINE)?;
    let zeros = run(
        dir,
        &["(bits[1]:0, bits[8]:2)", "--initial-state-all-zeros"],
    )?;
    assert!(
        zeros.status.success(),
        "{}",
        String::from_utf8_lossy(&zeros.stderr)
    );
    assert_eq!(zeros.stdout, b"(bits[8]:0, bits[8]:0)\n");
    let args = [
        "(bits[1]:0, bits[8]:2)",
        "--initial-state-random-seed",
        "923",
    ];
    let first = run(dir, &args)?;
    let replay = run(dir, &args)?;
    for result in [&first, &replay] {
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!(first.stdout, replay.stdout);
    let package = Parser::new(PIPELINE)
        .parse_package()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let evaluator = BlockEvaluator::prepare(&package, "pipeline")?;
    let initial = evaluator.initial_state(RegisterInit::Random { seed: 923 })?;
    let cycle = evaluator.step(
        &initial,
        &[IrValue::bool(false), IrValue::make_ubits(8, 2)?],
    )?;
    assert_eq!(
        String::from_utf8(first.stdout)?,
        format!("{}\n", IrValue::make_tuple(&cycle.outputs))
    );
    for invalid in [
        vec!["(bits[1]:0, bits[8]:2)"],
        vec![
            "(bits[1]:0, bits[8]:2)",
            "--initial-state-all-zeros",
            "--initial-state-random-seed",
            "3",
        ],
        vec!["(bits[1]:0, bits[3]:2)", "--initial-state-all-zeros"],
    ] {
        let output = run(dir, &invalid)?;
        assert!(!output.status.success(), "{invalid:?}");
        assert!(output.stdout.is_empty(), "{invalid:?}");
    }
    Ok(())
}

#[test]
fn rejects_incomplete_mistyped_or_multiple_startup_records() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = temp.path();
    fs::write(dir.join("block.ir"), PIPELINE)?;
    for invalid in [
        "{r: bits[8]:1}\n",
        "{r: bits[8]:1, stage.r: bits[8]:2, extra: bits[8]:3}\n",
        "{r: bits[8]:1, r: bits[8]:2}\n",
        "(bits[8]:1, bits[7]:2)\n",
        "(bits[8]:1, bits[8]:2)\n(bits[8]:3, bits[8]:4)\n",
    ] {
        fs::write(dir.join("initial.irvals"), invalid)?;
        let result = run(
            dir,
            &[
                "(bits[1]:0, bits[8]:3)",
                "--initial-state-file",
                "initial.irvals",
            ],
        )?;
        assert!(!result.status.success(), "{invalid}");
        assert!(result.stdout.is_empty(), "{invalid}");
    }
    Ok(())
}

#[test]
fn nested_empty_arrays_roundtrip_through_inputs_and_checkpoint() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = temp.path();
    fs::write(
        dir.join("block.ir"),
        r#"package p

top block delay(clk: clock, x: (bits[8][0], bits[3]), y: (bits[8][0], bits[3])) {
  reg r((bits[8][0], bits[3]))
  x: (bits[8][0], bits[3]) = input_port(name=x, id=1)
  q: (bits[8][0], bits[3]) = register_read(register=r, id=2)
  wr: () = register_write(x, register=r, id=3)
  y: () = output_port(q, name=y, id=4)
}
"#,
    )?;
    let first = run(
        dir,
        &[
            "(([], bits[3]:5))",
            "--initial-state-all-zeros",
            "--final-state-irvals",
            "state.irvals",
        ],
    )?;
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(first.stdout, b"([], bits[3]:0)\n");
    assert_eq!(
        fs::read(dir.join("state.irvals"))?,
        b"{r: ([], bits[3]:5)}\n"
    );
    fs::write(dir.join("inputs.irvals"), "{x: ([], bits[3]:3)}\n")?;
    let resumed = run(
        dir,
        &[
            "--input-irvals",
            "inputs.irvals",
            "--initial-state-file",
            "state.irvals",
            "--final-state-irvals",
            "state.irvals",
        ],
    )?;
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert_eq!(resumed.stdout, b"([], bits[3]:5)\n");
    assert_eq!(
        fs::read(dir.join("state.irvals"))?,
        b"{r: ([], bits[3]:3)}\n"
    );
    Ok(())
}

#[test]
fn selects_block_by_name_and_preserves_aggregate_port_types() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = temp.path();
    fs::write(dir.join("block.ir"), MULTI)?;
    let explicit = run(
        dir,
        &["--top", "pair", "((bits[8]:42, [bits[1]:1, bits[1]:0]))"],
    )?;
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    assert_eq!(explicit.stdout, b"(bits[8]:42, [bits[1]:1, bits[1]:0])\n");
    let no_ports = run(dir, &["--top", "empty", "()"])?;
    assert!(
        no_ports.status.success(),
        "{}",
        String::from_utf8_lossy(&no_ports.stderr)
    );
    assert_eq!(no_ports.stdout, b"()\n");
    for args in [
        vec!["()"],
        vec!["--top", "empty", "()", "--initial-state-all-zeros"],
    ] {
        assert!(!run(dir, &args)?.status.success(), "{args:?}");
    }
    Ok(())
}

#[test]
fn traces_and_failed_cycle_diagnostics_go_to_stderr() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let dir = temp.path();
    fs::write(
        dir.join("block.ir"),
        r#"package p

top block checked(ok: bits[1], out: bits[1]) {
  ok: bits[1] = input_port(name=ok, id=1)
  token_in: token = after_all(id=2)
  enabled: bits[1] = literal(value=1, id=3)
  watched: token = trace(token_in, enabled, format="sample", data_operands=[], verbosity=0, id=4)
  tested: token = assert(watched, ok, message="expected ok", label="check_ok", id=5)
  out: () = output_port(ok, name=out, id=6)
}
"#,
    )?;
    fs::write(dir.join("inputs.irvals"), "(bits[1]:1)\n(bits[1]:0)\n")?;
    let output = run(dir, &["--input-irvals", "inputs.irvals"])?;
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stdout, b"bits[1]:1\n");
    assert_eq!(
        output.stderr,
        include_bytes!("goldens/ir_block_eval_trace_failure.golden.txt")
    );
    Ok(())
}
