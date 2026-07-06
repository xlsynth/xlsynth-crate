// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::process::Command;

use xlsynth_pir::ir::{NaryOp, NodePayload, NodeRef, PackageMember, Unop};
use xlsynth_pir::ir_parser::Parser;

fn run_driver(driver: &str, args: &[&str]) -> std::process::Output {
    Command::new(driver).args(args).output().unwrap()
}

fn command_is_available(command: &str) -> bool {
    Command::new(command)
        .arg("-V")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn slang_path() -> Option<String> {
    let command = std::env::var("SLANG_PATH").unwrap_or_else(|_| "slang".to_string());
    Command::new(&command)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
        .then_some(command)
}

fn e2e_required() -> bool {
    std::env::var("XLSYNTH_REQUIRE_BLOCK_E2E").is_ok_and(|value| value == "1")
}

fn tool_path_or_skip() -> Option<String> {
    let required = e2e_required();
    std::env::var("XLSYNTH_TEST_TOOL_PATH")
        .or_else(|_| std::env::var("XLSYNTH_TOOLS"))
        .map(Some)
        .unwrap_or_else(|_| {
            assert!(
                !required,
                "XLSYNTH_REQUIRE_BLOCK_E2E=1 but no XLSYNTH_TEST_TOOL_PATH or XLSYNTH_TOOLS was provided"
            );
            eprintln!("skipping block E2E: XLS tool path is unavailable");
            None
        })
}

fn simulation_tools_or_skip() -> Option<(String, String)> {
    let iverilog = std::env::var("IVERILOG").unwrap_or_else(|_| "iverilog".to_string());
    let vvp = std::env::var("VVP").unwrap_or_else(|_| "vvp".to_string());
    if command_is_available(&iverilog) && command_is_available(&vvp) {
        Some((iverilog, vvp))
    } else {
        assert!(
            !e2e_required(),
            "XLSYNTH_REQUIRE_BLOCK_E2E=1 but Icarus or VVP is unavailable"
        );
        eprintln!("skipping block E2E: Icarus or VVP is unavailable");
        None
    }
}

fn write_toolchain(path: &Path, tool_path: &str) {
    std::fs::write(
        path,
        format!(
            "[toolchain]\ntool_path = {tool_path:?}\n\n[toolchain.dslx]\ndslx_stdlib_path = {:?}\n",
            Path::new(tool_path).join("xls/dslx/stdlib")
        ),
    )
    .unwrap();
}

fn assert_reset_masked_property_ir(ir_text: &str, reset_name: &str, active_low: bool) {
    let package = Parser::new(ir_text)
        .parse_and_validate_package()
        .expect("generated property Block IR should verify");
    let PackageMember::Block { func, .. } = package
        .get_top_block()
        .expect("generated property package should have a top block")
    else {
        panic!("property top should be a block");
    };
    let named_ref = |name: &str| {
        func.nodes
            .iter()
            .position(|node| node.name.as_deref() == Some(name))
            .map(|index| NodeRef { index })
            .unwrap_or_else(|| panic!("generated block has no node named '{name}'"))
    };
    let reset = named_ref(reset_name);
    let predicate = named_ref("predicate");
    let reset_active = if active_low {
        func.nodes
            .iter()
            .enumerate()
            .find_map(|(index, node)| {
                matches!(node.payload, NodePayload::Unop(Unop::Not, operand) if operand == reset)
                    .then_some(NodeRef { index })
            })
            .expect("active-low property block should invert reset")
    } else {
        reset
    };
    let assert_activate = func
        .nodes
        .iter()
        .find_map(|node| match node.payload {
            NodePayload::Assert { activate, .. } => Some(activate),
            _ => None,
        })
        .expect("property block should contain an assertion");
    assert!(matches!(
        &func.get_node(assert_activate).payload,
        NodePayload::Nary(NaryOp::Or, operands)
            if operands.contains(&reset_active) && operands.contains(&predicate)
    ));
    let cover_predicate = func
        .nodes
        .iter()
        .find_map(|node| match node.payload {
            NodePayload::Cover { predicate, .. } => Some(predicate),
            _ => None,
        })
        .expect("property block should contain a cover");
    let NodePayload::Nary(NaryOp::And, cover_operands) = &func.get_node(cover_predicate).payload
    else {
        panic!("cover should be gated by an and node");
    };
    assert!(cover_operands.contains(&predicate));
    assert!(cover_operands.iter().any(|operand| {
        matches!(
            func.get_node(*operand).payload,
            NodePayload::Unop(Unop::Not, reset_operand) if reset_operand == reset_active
        )
    }));
}

#[test]
fn composed_blocks_proc_and_extern_verilog_roundtrip_codegen_and_simulate() {
    let Some(tool_path) = tool_path_or_skip() else {
        return;
    };
    let Some((iverilog, vvp)) = simulation_tools_or_skip() else {
        return;
    };

    let temporary = tempfile::tempdir().unwrap();
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/dslx_block_e2e");
    let source_path = fixture_dir.join("composed.x");
    let testbench_path = fixture_dir.join("tb.sv");
    let toolchain_path = temporary.path().join("xlsynth-toolchain.toml");
    let ir_path = temporary.path().join("composed.ir");
    let sv_path = temporary.path().join("composed.sv");
    let simulation_path = temporary.path().join("simv");
    write_toolchain(&toolchain_path, &tool_path);

    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let ir = run_driver(
        driver,
        &[
            "--toolchain",
            toolchain_path.to_str().unwrap(),
            "dslx-block2ir",
            "--dslx_input_file",
            source_path.to_str().unwrap(),
        ],
    );
    assert!(
        ir.status.success(),
        "{}",
        String::from_utf8_lossy(&ir.stderr)
    );
    let ir_text = String::from_utf8(ir.stdout).unwrap();
    assert!(ir_text.contains("kind=extern"));
    assert!(ir_text.contains("block custom_logic"));
    assert!(ir_text.contains("block proc_wrapper"));
    assert!(ir_text.contains("top block composed_top"));
    assert!(ir_text.contains("__xlsynth_proc_composed_Producer_active_high"));
    assert!(!ir_text.contains("invoke("));
    std::fs::write(&ir_path, &ir_text).unwrap();

    let roundtrip = run_driver(
        driver,
        &[
            "ir-round-trip",
            ir_path.to_str().unwrap(),
            "--preserve-block-port-order=true",
        ],
    );
    assert!(
        roundtrip.status.success(),
        "{}",
        String::from_utf8_lossy(&roundtrip.stderr)
    );
    assert_eq!(String::from_utf8(roundtrip.stdout).unwrap(), ir_text);

    let sv = run_driver(
        driver,
        &[
            "--toolchain",
            toolchain_path.to_str().unwrap(),
            "dslx-block2sv",
            "--dslx_input_file",
            source_path.to_str().unwrap(),
        ],
    );
    assert!(
        sv.status.success(),
        "{}",
        String::from_utf8_lossy(&sv.stderr)
    );
    let sv_text = String::from_utf8(sv.stdout).unwrap();
    assert!(sv_text.contains("module custom_logic("));
    assert!(sv_text.contains("module proc_wrapper("));
    assert!(sv_text.contains("module composed_top("));
    assert_eq!(sv_text.matches("^ 8'hA5").count(), 1);
    assert!(sv_text.contains(
        "module custom_logic(\n  output wire [7:0] y,\n  input wire [7:0] x,\n  input wire clk,\n  input wire rst\n);"
    ));
    std::fs::write(&sv_path, sv_text).unwrap();

    let compile = Command::new(&iverilog)
        .arg("-g2012")
        .arg("-s")
        .arg("tb")
        .arg("-o")
        .arg(&simulation_path)
        .arg(&sv_path)
        .arg(&testbench_path)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let simulation = Command::new(&vvp).arg(&simulation_path).output().unwrap();
    assert!(
        simulation.status.success(),
        "{}",
        String::from_utf8_lossy(&simulation.stderr)
    );
    assert!(String::from_utf8_lossy(&simulation.stdout).contains("BLOCK_E2E_PASS"));

    let renamed = run_driver(
        driver,
        &[
            "--toolchain",
            toolchain_path.to_str().unwrap(),
            "dslx-block2sv",
            "--dslx_input_file",
            source_path.to_str().unwrap(),
            "--module_name",
            "renamed_composed_top",
        ],
    );
    assert!(
        renamed.status.success(),
        "{}",
        String::from_utf8_lossy(&renamed.stderr)
    );
    let renamed_text = String::from_utf8(renamed.stdout).unwrap();
    assert!(renamed_text.contains("module renamed_composed_top("));
    assert!(!renamed_text.contains("module composed_top("));
}

#[test]
fn reset_masked_assertions_and_covers_survive_official_codegen() {
    let Some(tool_path) = tool_path_or_skip() else {
        return;
    };
    let Some((iverilog, _)) = simulation_tools_or_skip() else {
        return;
    };
    let slang = slang_path();
    assert!(
        !e2e_required() || slang.is_some(),
        "XLSYNTH_REQUIRE_BLOCK_E2E=1 but Slang is unavailable; required property proofs may not disable assertions"
    );
    let temporary = tempfile::tempdir().unwrap();
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/dslx_block_e2e");
    let source_path = fixture_dir.join("properties.x");
    let toolchain_path = temporary.path().join("xlsynth-toolchain.toml");
    write_toolchain(&toolchain_path, &tool_path);
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    for (top, reset_signal, active_low) in [
        ("active_high_props", "rst", false),
        ("active_low_props", "rst_n", true),
    ] {
        let ir = run_driver(
            driver,
            &[
                "--toolchain",
                toolchain_path.to_str().unwrap(),
                "dslx-block2ir",
                "--dslx_input_file",
                source_path.to_str().unwrap(),
                "--dslx_top",
                top,
            ],
        );
        assert!(
            ir.status.success(),
            "{}",
            String::from_utf8_lossy(&ir.stderr)
        );
        assert_reset_masked_property_ir(
            &String::from_utf8(ir.stdout).unwrap(),
            reset_signal,
            active_low,
        );
        let output = run_driver(
            driver,
            &[
                "--toolchain",
                toolchain_path.to_str().unwrap(),
                "dslx-block2sv",
                "--dslx_input_file",
                source_path.to_str().unwrap(),
                "--dslx_top",
                top,
            ],
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let system_verilog = String::from_utf8(output.stdout).unwrap();
        assert!(system_verilog.contains("assert property (@(posedge clk)"));
        assert!(system_verilog.contains("cover property (@(posedge clk)"));
        assert!(system_verilog.contains("__xlsynth_assert_0_"));
        assert!(system_verilog.contains("__xlsynth_cover_0_"));
        if active_low {
            assert!(system_verilog.contains("__xlsynth_reset_active = ~rst_n"));
        } else {
            assert!(system_verilog.contains(" = rst | predicate"));
        }
        assert!(system_verilog.contains(reset_signal));

        let sv_path = temporary.path().join(format!("{top}.sv"));
        let simulation_path = temporary.path().join(format!("{top}.simv"));
        std::fs::write(&sv_path, system_verilog).unwrap();
        let compile = if let Some(slang) = &slang {
            Command::new(slang)
                .arg("--std")
                .arg("1800-2023")
                .arg("--lint-only")
                .arg("--top")
                .arg(top)
                .arg(&sv_path)
                .output()
                .unwrap()
        } else {
            Command::new(&iverilog)
                .arg("-g2012")
                .arg("-gno-assertions")
                .arg("-s")
                .arg(top)
                .arg("-o")
                .arg(&simulation_path)
                .arg(&sv_path)
                .output()
                .unwrap()
        };
        assert!(
            compile.status.success(),
            "{}",
            String::from_utf8_lossy(&compile.stderr)
        );
    }
}

#[test]
fn active_low_proc_wrapper_holds_state_under_backpressure() {
    let Some(tool_path) = tool_path_or_skip() else {
        return;
    };
    let Some((iverilog, vvp)) = simulation_tools_or_skip() else {
        return;
    };
    let temporary = tempfile::tempdir().unwrap();
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/dslx_block_e2e");
    let source_path = fixture_dir.join("proc_active_low.x");
    let testbench_path = fixture_dir.join("proc_active_low_tb.sv");
    let toolchain_path = temporary.path().join("xlsynth-toolchain.toml");
    let sv_path = temporary.path().join("proc_active_low.sv");
    let simulation_path = temporary.path().join("proc_active_low.simv");
    write_toolchain(&toolchain_path, &tool_path);

    let output = run_driver(
        env!("CARGO_BIN_EXE_xlsynth-driver"),
        &[
            "--toolchain",
            toolchain_path.to_str().unwrap(),
            "dslx-block2sv",
            "--dslx_input_file",
            source_path.to_str().unwrap(),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let system_verilog = String::from_utf8(output.stdout).unwrap();
    assert!(system_verilog.contains("module proc_active_low("));
    assert!(system_verilog.contains("rst_n"));
    std::fs::write(&sv_path, system_verilog).unwrap();

    let compile = Command::new(&iverilog)
        .arg("-g2012")
        .arg("-s")
        .arg("proc_active_low_tb")
        .arg("-o")
        .arg(&simulation_path)
        .arg(&sv_path)
        .arg(&testbench_path)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let simulation = Command::new(&vvp).arg(&simulation_path).output().unwrap();
    assert!(
        simulation.status.success(),
        "{}",
        String::from_utf8_lossy(&simulation.stderr)
    );
    assert!(String::from_utf8_lossy(&simulation.stdout).contains("BLOCK_PROC_ACTIVE_LOW_PASS"));
}
