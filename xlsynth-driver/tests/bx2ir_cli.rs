// SPDX-License-Identifier: Apache-2.0

//! Exercise BX source, linked IR, RTL, and mapped netlists at the driver
//! boundary.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use anyhow::{Context, Result};
use prost::Message;
use xlsynth_g8r::netlist::proto::pb;
use xlsynth_pir::ir::MemberType;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

fn driver(directory: &Path, arguments: &[&str]) -> std::io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .env_remove("RUST_LOG")
        .env("NO_COLOR", "1")
        .args(arguments)
        .output()
}

/// Returns successful stdout and includes the command's diagnostics on failure.
fn stdout(directory: &Path, arguments: &[&str]) -> Result<String> {
    let output = driver(directory, arguments)?;
    anyhow::ensure!(
        output.status.success(),
        "{arguments:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?)
}

#[test]
fn imported_dslx_logic_runs_at_the_bx_register_boundary() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::create_dir(root.join("empty"))?;
    fs::create_dir(root.join("library"))?;
    fs::write(
        root.join("library/arithmetic.x"),
        "pub fn increment(x: u8) -> u8 { x + u8:1 }",
    )?;
    fs::write(
        root.join("design.bx"),
        r#"use arithmetic::increment;
block registered_sum(clk, x: u8) -> u8 {
    reg q: u8;
    q := increment(x);
    q
}"#,
    )?;
    let ir = stdout(
        root,
        &["bx2ir", "design.bx", "--dslx-path", "empty;library"],
    )?;
    let package = Parser::new(&ir).parse_and_verify_package()?;
    assert_eq!(package.name, "design");
    assert_eq!(
        package.top,
        Some(("registered_sum".into(), MemberType::Block))
    );

    fs::write(root.join("design.ir"), ir)?;
    let rtl = stdout(root, &["block2sv", "design.ir"])?;
    let module = compile_pipeline_module(&rtl).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let x = runner.port_bindings().input_with_width("x", 8).unwrap();
    let mut previous = None;
    for (input, expected) in [(7, 8), (255, 0)] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| drive.set_u64(&x, input))
            .unwrap();
        assert_eq!(edge.pre_outputs["out"].to_u64_if_known(), previous);
        assert_eq!(edge.outputs["out"].to_u64_if_known(), Some(expected));
        previous = Some(expected);
    }
    Ok(())
}

#[test]
fn inferred_dslx_struct_can_share_its_name_with_functions_and_constants() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(root.join("wire.x"), "pub struct Leaf { z: u8 }")?;
    fs::write(
        root.join("factory.x"),
        "import wire; pub fn make(x: u8) -> (wire::Leaf,) { (wire::Leaf { z: x },) }",
    )?;
    fs::write(
        root.join("design.bx"),
        "use decoy; use factory; block b(clk, x: u8) -> u8 { factory::make(x).0.z }",
    )?;
    // The generated type probe for decoy::Leaf fails. The subsequent probe
    // for wire::Leaf must still see an intact DSLX inference graph.
    for decoy in ["pub fn Leaf(x: u8) -> u8 { x }", "pub const Leaf = u32:2;"] {
        fs::write(root.join("decoy.x"), decoy)?;
        let ir = stdout(root, &["bx2ir", "design.bx"])?;
        let package = Parser::new(&ir).parse_and_verify_package()?;
        assert!(package.get_fn("__factory__make").is_some());
    }
    Ok(())
}

#[test]
fn dslx_path_replaces_the_configured_import_directories() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::create_dir(root.join("configured"))?;
    fs::create_dir(root.join("empty"))?;
    fs::write(
        root.join("tools.toml"),
        r#"[toolchain.dslx]
dslx_path = ["configured"]
"#,
    )?;
    fs::write(
        root.join("configured/custom_math.x"),
        "pub fn twice(x: u8) -> u8 { x + x }",
    )?;
    fs::write(
        root.join("math_pipe.bx"),
        r#"use custom_math;
block math_pipe(clk, x: u8) -> u8 { custom_math::twice(x) }"#,
    )?;

    let mut args = vec!["--toolchain=tools.toml", "bx2ir", "math_pipe.bx"];
    let ir = stdout(root, &args)?;
    Parser::new(&ir).parse_and_verify_package()?;

    args.push("--dslx-path=empty");
    let output = driver(root, &args)?;
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("custom_math"), "{stderr}");
    Ok(())
}

#[test]
fn extern_function_links_into_the_dslx_package() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("arithmetic.x"),
        "pub fn twice(x: u8) -> u8 { x + x }",
    )?;
    fs::write(
        root.join("arithmetic.ir"),
        stdout(root, &["dslx2ir", "--dslx_input_file=arithmetic.x"])?,
    )?;
    fs::write(
        root.join("doubler.bx"),
        r#"extern fn arithmetic::twice(x: u8) -> u8;
block doubler(clk, x: u8) -> u8 { arithmetic::twice(x) }"#,
    )?;
    let actual = stdout(root, &["bx2ir", "doubler.bx", "--link-ir=arithmetic.ir"])?;
    assert_eq!(actual, include_str!("bx2ir_linked.golden.ir"));

    // Handwritten XLS source can leave the parameter and add IDs implicit.
    fs::write(
        root.join("arithmetic.ir"),
        r#"package arithmetic
file_number 0 "arithmetic.x"
fn __arithmetic__twice(x: bits[8]) -> bits[8] {
  ret sum: bits[8] = add(x, x, pos=[(0,0,28)])
}"#,
    )?;
    let from_source = stdout(root, &["bx2ir", "doubler.bx", "--link-ir=arithmetic.ir"])?;
    let package = xlsynth::IrPackage::parse_ir(&from_source, None)?;
    let twice = package.get_function("__arithmetic__twice")?;
    assert_eq!(
        twice.interpret(&[xlsynth::XlsIrValue::make_ubits(8, 21)?])?,
        xlsynth::XlsIrValue::make_ubits(8, 42)?
    );
    Ok(())
}

#[test]
fn top_override_and_reset_flags_apply_to_a_standalone_module() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("blocks.bx"),
        r#"pub block passthrough(clk, x: u8) -> u8 { x }
block delay(clk, rst, x: u8) -> u8 {
    reg q: u8 = 0;
    q := x;
    q
}"#,
    )?;
    let default_ir = stdout(root, &["bx2ir", "blocks.bx"])?;
    let package = Parser::new(&default_ir).parse_and_verify_package()?;
    assert_eq!(package.top, Some(("passthrough".into(), MemberType::Block)));
    let default_reset = package
        .get_block("delay")
        .context("missing delay")?
        .reset
        .as_ref()
        .context("missing reset")?;
    assert!(!default_reset.active_low);
    assert!(!default_reset.asynchronous);

    let ir = stdout(
        root,
        &[
            "bx2ir",
            "blocks.bx",
            "--top=delay",
            "--reset-active-low",
            "--reset-asynchronous=true",
        ],
    )?;
    let package = Parser::new(&ir).parse_and_verify_package()?;
    assert_eq!(package.top, Some(("delay".into(), MemberType::Block)));
    let top = package.get_top_block().context("missing top")?;
    let reset = top.reset.as_ref().context("missing reset")?;
    assert!(reset.active_low);
    assert!(reset.asynchronous);
    Ok(())
}

#[test]
fn bx2lean_emits_the_selected_block_and_preserves_existing_output() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("design.bx"),
        r#"pub block passthrough(clk, x: u8) -> u8 { x }
struct Outputs { seen: u8, valid: bool }
block counter(clk, rst, x: u8) -> Outputs {
    reg count: u8 = 0;
    count := x;
    Outputs { seen: count, valid: true }
}"#,
    )?;
    let options = ["design.bx", "--top=counter", "--reset-active-low"];
    let args = [&["bx2lean"][..], &options, &["--output-dir=lean"]].concat();
    assert!(stdout(root, &args)?.is_empty());
    let mut files = fs::read_dir(root.join("lean"))?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    assert_eq!(
        files,
        [
            "Generated.lean",
            "ReadableCorrespondence.lean",
            "ReadableGenerated.lean",
            "ReadableRepresentation.lean",
            "ReadableTypes.lean",
        ]
    );
    fs::write(
        root.join("saved.ir"),
        stdout(root, &[&["bx2ir"][..], &options].concat())?,
    )?;
    assert_eq!(
        fs::read_to_string(root.join("lean/Generated.lean"))?,
        stdout(root, &["block2lean", "saved.ir"])?
    );
    fs::write(root.join("lean/Generated.lean"), "keep")?;
    assert!(!driver(root, &args)?.status.success());
    assert_eq!(
        fs::read_to_string(root.join("lean/Generated.lean"))?,
        "keep"
    );
    let missing = driver(
        root,
        &[
            "bx2lean",
            "design.bx",
            "--top=unknown",
            "--output-dir=missing",
        ],
    )?;
    assert!(!missing.status.success());
    assert!(String::from_utf8(missing.stderr)?.contains("unknown"));
    assert!(!root.join("missing").try_exists()?);
    Ok(())
}

#[test]
fn ambiguous_and_missing_top_fail_without_emitting_ir() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("choices.bx"),
        r#"block a(clk, x: u8) -> u8 { x }
block b(clk, x: u8) -> u8 { x + x }"#,
    )?;
    for (arguments, diagnostic) in [
        (vec!["bx2ir", "choices.bx"], "--top"),
        (vec!["bx2ir", "choices.bx", "--top=absent"], "absent"),
    ] {
        let output = driver(root, &arguments)?;
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr)?;
        assert!(stderr.contains(diagnostic), "{stderr}");
    }
    Ok(())
}

#[test]
fn bx2lean_links_externs_from_a_checked_dslx_module() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::create_dir(root.join("modules"))?;
    fs::write(
        root.join("modules/packet.x"),
        "pub struct Packet { octet: u8, tag: u8 }",
    )?;
    fs::write(
        root.join("math.x"),
        "import packet; pub fn second(request: packet::Packet) -> u8 { request.tag }",
    )?;
    fs::write(
        root.join("design.bx"),
        r#"extern fn math::second(request: (u8, u8)) -> u8;
block delay(clk, x: u8, y: u8) -> u8 {
    reg q: u8;
    q := math::second((x, y));
    q
}"#,
    )?;
    let args = [
        "bx2lean",
        "design.bx",
        "--link-dslx=math.x",
        "--dslx-path=modules",
        "--output-dir=model",
    ];
    assert!(stdout(root, &args)?.is_empty());
    let types = fs::read_to_string(root.join("model/ReadableTypes.lean"))?;
    for field in ["octet", "tag"] {
        assert!(types.contains(field), "missing DSLX field {field}: {types}");
    }
    let both = driver(root, &[&args[..], &["--link-ir=saved.ir"]].concat())?;
    assert_eq!(both.status.code(), Some(2));
    let error = String::from_utf8(both.stderr)?;
    assert!(
        error.contains("--link-ir") && error.contains("--link-dslx"),
        "{error}"
    );
    Ok(())
}

#[test]
fn bx2lean_offers_loop_names_only_with_compiler_support() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    let help = stdout(root, &["bx2lean", "--help"])?;
    assert_eq!(
        help.contains("--readable-loop-names"),
        cfg!(feature = "dslx-loop-bindings")
    );
    #[cfg(feature = "dslx-loop-bindings")]
    {
        fs::write(
            root.join("sum.x"),
            r#"pub fn sum_lanes(xs: u8[3]) -> u8 {
    for (lane, acc) in u32:1..u32:3 { acc + xs[lane] }(u8:0)
}"#,
        )?;
        fs::write(
            root.join("lanes.bx"),
            r#"extern fn sum::sum_lanes(xs: u8[3]) -> u8;
block delay(clk, xs: u8[3]) -> u8 {
    reg q: u8;
    q := sum::sum_lanes(xs);
    q
}"#,
        )?;
        let args = [
            "bx2lean",
            "lanes.bx",
            "--link-dslx=sum.x",
            "--readable-loop-names",
            "--output-dir=model",
        ];
        assert!(stdout(root, &args)?.is_empty());
        let source = fs::read_to_string(root.join("model/ReadableGenerated.lean"))?;
        assert!(
            source
                .lines()
                .any(|line| line.trim_start().starts_with("let «lane» : ")),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn mapped_protobuf_matches_the_staged_pipeline_with_synchronous_reset_and_enable() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::create_dir(root.join("modules"))?;
    fs::write(
        root.join("modules/math.x"),
        "pub fn invert(x: u2) -> u2 { !x }",
    )?;
    fs::write(
        root.join("pipeline.bx"),
        r#"use math;
block pipeline(clk, rst, x: u2, en: bool) -> u2 {
    reg q: u2 = 0;
    if en { q := math::invert(x); }
    q
}"#,
    )?;
    fs::write(root.join("cells.lib"), MAPPING_CELLS)?;
    stdout(root, &["lib2proto", "cells.lib", "--output=cells.proto"])?;
    let bx_options = ["pipeline.bx", "--dslx-path=modules", "--reset-active-low"];
    let map_options = ["--liberty-proto=cells.proto", "--module-name=mapped"];

    fs::write(
        root.join("pipeline.ir"),
        stdout(root, &[&["bx2ir"][..], &bx_options].concat())?,
    )?;
    fs::write(
        root.join("pipeline.opt.ir"),
        stdout(root, &["ir2opt", "pipeline.ir", "--top=pipeline"])?,
    )?;
    fs::write(
        root.join("pipeline.g8r"),
        stdout(
            root,
            &[
                "ir2g8r",
                "pipeline.opt.ir",
                "--enable-formal-array-alias-analysis=false",
            ],
        )?,
    )?;
    stdout(
        root,
        &[
            &["g8r2gv", "pipeline.g8r"][..],
            &map_options,
            &["-o", "staged.pb"],
        ]
        .concat(),
    )?;
    let staged = fs::read(root.join("staged.pb"))?;

    let direct = [&["bx2gv"][..], &bx_options, &map_options].concat();
    let mut file_args = direct.clone();
    file_args.extend(["-o", "direct.pb"]);
    assert!(stdout(root, &file_args)?.is_empty());
    assert_eq!(fs::read(root.join("direct.pb"))?, staged);
    let mut stdout_args = direct.clone();
    stdout_args.push("--netlist-format=pb");
    let output = driver(root, &stdout_args)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, staged);
    let netlist = pb::Netlist::decode(staged.as_slice())?;
    let [module] = netlist.modules.as_slice() else {
        anyhow::bail!("expected one mapped module")
    };
    assert_eq!(module.name, "mapped");
    assert_eq!(
        module
            .instances
            .iter()
            .filter(|instance| netlist.interfaces[instance.interface as usize].cell == "DFF")
            .count(),
        2
    );

    let mut async_args = direct;
    async_args.extend(["--reset-asynchronous", "-o", "async.pb"]);
    let output = driver(root, &async_args)?;
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!root.join("async.pb").try_exists()?);
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("asynchronous reset"), "{stderr}");
    Ok(())
}

const MAPPING_CELLS: &str = r#"library(test) {
  cell(BUF) { area: 1; pin(A) { direction: input; }
    pin(Y) { direction: output; function: "A"; } }
  cell(INV) { area: 1; pin(A) { direction: input; }
    pin(Y) { direction: output; function: "!A"; } }
  cell(NAND2) { area: 2; pin(A) { direction: input; } pin(B) { direction: input; }
    pin(Y) { direction: output; function: "!(A & B)"; } }
  cell(DFF) { area: 3; pin(D) { direction: input; } pin(CLK) { direction: input; clock: true; }
    pin(Q) { direction: output; function: "IQ"; }
    ff(IQ,IQN) { clocked_on: "CLK"; next_state: "D"; } }
}"#;

#[test]
fn bx_child_register_uses_parent_clock_despite_linked_port_name() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("delay.ir"),
        r#"package cells
block delay(CK: clock, d: bits[1], out: bits[1]) {
  reg q(bits[1])
  d: bits[1] = input_port(name=d, id=1)
  register_write.2: () = register_write(d, register=q, id=2)
  register_read.3: bits[1] = register_read(register=q, id=3)
  out: () = output_port(register_read.3, name=out, id=4)
}"#,
    )?;
    fs::write(
        root.join("parent.bx"),
        r#"extern block delay(clk, d: bool) -> bool;
block parent(clk, x: bool) -> bool {
    let delayed = delay(x);
    delayed
}"#,
    )?;
    fs::write(root.join("cells.lib"), MAPPING_CELLS)?;
    stdout(root, &["lib2proto", "cells.lib", "--output=cells.proto"])?;
    stdout(
        root,
        &[
            "bx2gv",
            "parent.bx",
            "--link-ir=delay.ir",
            "--liberty-proto=cells.proto",
            "-o",
            "parent.pb",
        ],
    )?;
    let netlist = pb::Netlist::decode(fs::read(root.join("parent.pb"))?.as_slice())?;
    let [module] = netlist.modules.as_slice() else {
        anyhow::bail!("expected one flattened module")
    };
    let [flop] = module.instances.as_slice() else {
        anyhow::bail!("expected one mapped register")
    };
    let interface = &netlist.interfaces[flop.interface as usize];
    assert_eq!(interface.cell, "DFF");
    let clock_pin = interface
        .pins
        .iter()
        .position(|pin| pin == "CLK")
        .context("DFF is missing CLK")?;
    let clock_net = flop.nets[clock_pin];
    assert_eq!(module.nets[clock_net as usize].name, "clk");
    assert!(module.ports.iter().any(|port| port.net == clock_net));
    Ok(())
}
