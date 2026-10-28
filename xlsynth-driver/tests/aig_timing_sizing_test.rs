// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use prost::Message;
use xlsynth_g8r::aig_sim::gate_sim::{Collect, eval};
use xlsynth_g8r::liberty::model::library_to_proto;
use xlsynth_g8r::liberty::parser::parse_liberty_files;
use xlsynth_g8r::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
use xlsynth_g8r::netlist::io::{load_liberty_from_path, read_gv_from_str};
use xlsynth_pir::IrBits;

/// Size-only equivalents independently exercise timing and recovery, with
/// load/slew-sensitive delay so a mismatched model affects measured results.
fn write_fixture(dir: &Path) -> anyhow::Result<()> {
    std::fs::write(dir.join("input.aag"), "aag 3 2 0 1 1\n2\n4\n6\n6 2 4\n")?;
    let mut text = String::from(
        r#"library(test) {
        time_unit: "1ps"; capacitive_load_unit(1,pf);
        lu_table_template(delay) { variable_1: input_net_transition;
            variable_2: total_output_net_capacitance;
            index_1("0,10"); index_2("0,10"); }
    "#,
    );
    for (name, area, delay, size_only, pins, function) in [
        ("INV", 1.0, 1.0, false, "A", "!A"),
        ("INV_SMALL", 0.9, 1.0, true, "A", "!A"),
        ("NAND2", 2.0, 2.0, false, "A B", "!(A*B)"),
        ("NAND2_FAST", 2.04, 1.0, true, "A B", "!(A*B)"),
    ] {
        text.push_str(&format!(
            "cell({name}) {{ area: {area}; use_for_size_only: {size_only};"
        ));
        for pin in pins.split_whitespace() {
            text.push_str(&format!(
                "pin({pin}) {{ direction: input; capacitance: 0.01; }}"
            ));
        }
        text.push_str(&format!(
            r#"pin(Y) {{ direction: output; function: "{function}";"#
        ));
        for pin in pins.split_whitespace() {
            text.push_str(&format!(
                r#"timing() {{ related_pin: "{pin}";
                timing_sense: negative_unate; timing_type: combinational;"#
            ));
            for kind in ["cell_rise", "cell_fall"] {
                text.push_str(&format!(
                    r#"{kind}(delay) {{ values("{delay},{}", "{},{}"); }}"#,
                    delay + 10.0,
                    delay + 1.0,
                    delay + 11.0
                ));
            }
            text.push_str(
                r#"rise_transition(scalar) { values("0.2"); }
                fall_transition(scalar) { values("0.2"); } }"#,
            );
        }
        text.push_str("}}");
    }
    text.push('}');
    let raw = dir.join("library.lib");
    std::fs::write(&raw, text)?;
    let mut library = parse_liberty_files(&[&raw]).map_err(anyhow::Error::msg)?;
    library.default_output_load = Some(0.02);
    std::fs::write(
        dir.join("library.proto"),
        library_to_proto(library)?.encode_to_vec(),
    )?;
    Ok(())
}

/// Runs the integrated flow, returning the emitted netlist and structured
/// report.
fn map(
    dir: &Path,
    name: &str,
    repair: bool,
    extra: &[&str],
) -> anyhow::Result<(String, serde_json::Value)> {
    let netlist = dir.join(format!("{name}.gv"));
    let report = dir.join(format!("{name}.json"));
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("aig-tech-map")
        .arg(dir.join("input.aag"))
        .args(["--strategy", "area", "--liberty_proto"])
        .arg(dir.join("library.proto"))
        .arg(format!("--fanout-opt={repair}"))
        .arg("--netlist_out")
        .arg(&netlist)
        .arg("--json_out")
        .arg(&report)
        .args(extra)
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok((
        std::fs::read_to_string(netlist)?,
        serde_json::from_slice(&std::fs::read(report)?)?,
    ))
}

/// Checks all input assignments independently of the sizing decision.
fn check_and(dir: &Path, text: &str) -> anyhow::Result<()> {
    let library = load_liberty_from_path(&dir.join("library.proto"))?;
    let parsed = read_gv_from_str(text)?;
    let projected = project_gatefn_from_netlist_and_liberty(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        &library,
        &HashSet::new(),
        &HashSet::new(),
    )
    .map_err(anyhow::Error::msg)?;
    for a in [false, true] {
        for b in [false, true] {
            let output = eval(
                &projected,
                &[IrBits::bool(a), IrBits::bool(b)],
                Collect::None,
            )
            .outputs;
            assert_eq!(output, vec![IrBits::bool(a && b)]);
        }
    }
    Ok(())
}

#[test]
fn timing_sizing_defaults_caps_budgets_and_phase_optouts() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    write_fixture(dir.path())?;
    let (raw, raw_report) = map(dir.path(), "raw", false, &["--timing-sizing=false"])?;
    assert!(raw_report.get("timing_sizing").is_none());
    assert!(raw_report.get("area_recovery").is_none());
    let (sized, report) = map(dir.path(), "default", false, &[])?;
    let (explicit, explicit_report) = map(dir.path(), "explicit", false, &["--timing-sizing"])?;
    assert_eq!(sized, explicit);
    assert_eq!(report, explicit_report);
    let r = &report["timing_sizing"];
    assert_eq!(report["timing_sizing_max_area_growth"], 0.02);
    assert_eq!(report["timing_sizing_rounds"], 4);
    assert_eq!(report["area_recovery_rounds"], 2);
    assert_eq!(r["work_limits"]["timing_evaluations"], 50000);
    assert_eq!(r["work_limits"]["recovery_evaluations"], 100000);
    assert_eq!(r["upsize"]["accepted"], 1);
    assert_eq!(r["downsize"]["accepted"], 1);
    assert!(r["upsized_area"].as_f64().unwrap() <= 3.0 * 1.02);
    assert!(r["final_area"].as_f64().unwrap() < r["upsized_area"].as_f64().unwrap());
    assert!(
        r["upsized_delay"]["input_to_output"].as_f64().unwrap()
            < r["initial_delay"]["input_to_output"].as_f64().unwrap()
    );
    check_and(dir.path(), &sized)?;
    let (_, cap) = map(
        dir.path(),
        "cap",
        false,
        &["--timing-sizing-max-area-growth", "0"],
    )?;
    assert_eq!(cap["timing_sizing"]["upsize"]["accepted"], 0);
    assert!(cap["timing_sizing"]["final_area"].as_f64().unwrap() <= 3.0);
    let (_, no_recovery) = map(
        dir.path(),
        "no-recovery",
        false,
        &["--timing-sizing", "--area-recovery=false"],
    )?;
    assert_eq!(
        no_recovery["timing_sizing"]["downsize"]["stop_reason"],
        "disabled"
    );
    assert_eq!(
        no_recovery["timing_sizing"]["final_area"],
        r["upsized_area"]
    );
    for (name, option) in [
        ("zero-budget", "--timing-sizing-evaluation-budget"),
        ("zero-rounds", "--timing-sizing-rounds"),
    ] {
        let (result, report) = map(
            dir.path(),
            name,
            false,
            &[
                "--timing-sizing",
                option,
                "0",
                "--area-recovery-evaluation-budget",
                "0",
            ],
        )?;
        assert_eq!(result, raw);
        assert_eq!(report["timing_sizing"]["upsize"]["propagated_instances"], 0);
        assert_eq!(
            report["timing_sizing"]["downsize"]["propagated_instances"],
            0
        );
    }
    let (_, separate_budget) = map(
        dir.path(),
        "separate-budget",
        false,
        &["--timing-sizing", "--timing-sizing-evaluation-budget", "0"],
    )?;
    assert_eq!(separate_budget["timing_sizing"]["upsize"]["accepted"], 0);
    assert_eq!(separate_budget["timing_sizing"]["downsize"]["accepted"], 1);
    let (_, repeated) = map(dir.path(), "repeat", false, &["--timing-sizing"])?;
    assert_eq!(report, repeated);
    Ok(())
}

#[test]
fn timing_sizing_matches_explicit_resize_after_optional_repair_with_shared_model()
-> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    write_fixture(dir.path())?;
    for repair in [false, true] {
        let (before, _) = map(dir.path(), "before", repair, &["--timing-sizing=false"])?;
        std::fs::write(dir.path().join("before.gv"), before)?;
        let mut previous_delay = 0.0;
        for wire in ["0", "500"] {
            let flags = [
                "--timing-sizing",
                "--primary-input-transition",
                "0.03",
                "--module-output-load",
                "0.02",
                "--wire-capacitance-constant-ff",
                wire,
                "--wire-capacitance-per-sink-ff",
                wire,
                "--timing-sizing-max-area-growth",
                "0.02",
                "--timing-sizing-rounds",
                "3",
                "--timing-sizing-evaluation-budget",
                "4000",
                "--area-recovery-rounds",
                "1",
                "--area-recovery-evaluation-budget",
                "3000",
            ];
            let (after, report) = map(dir.path(), "after", repair, &flags)?;
            // Repair and sizing must use the same model; obtain the actual
            // repaired input under that model before checking composition.
            if repair {
                let mut repair_flags = vec!["--timing-sizing=false"];
                repair_flags.extend_from_slice(&flags[1..9]);
                let (before, _) = map(dir.path(), "before", true, &repair_flags)?;
                std::fs::write(dir.path().join("before.gv"), before)?;
            }
            let explicit = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
                .args(["gv-resize", "--netlist"])
                .arg(dir.path().join("before.gv"))
                .arg("--liberty_proto")
                .arg(dir.path().join("library.proto"))
                .args([
                    "--primary_input_transition",
                    "0.03",
                    "--module_output_load",
                    "0.02",
                    "--wire-capacitance-constant-ff",
                    wire,
                    "--wire-capacitance-per-sink-ff",
                    wire,
                    "--max-area-growth",
                    "0.02",
                    "--upsize-rounds",
                    "3",
                    "--timing-evaluation-budget",
                    "4000",
                    "--downsize-rounds",
                    "1",
                    "--recovery-evaluation-budget",
                    "3000",
                    "--report-json",
                ])
                .arg(dir.path().join("explicit.json"))
                .output()?;
            anyhow::ensure!(
                explicit.status.success(),
                "{}",
                String::from_utf8_lossy(&explicit.stderr)
            );
            assert_eq!(after.as_bytes(), explicit.stdout);
            let standalone: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.path().join("explicit.json"))?)?;
            assert_eq!(report["timing_sizing"], standalone);
            if repair {
                assert_eq!(
                    report["fanout_optimization"]["options"]["wire_load"],
                    report["post_mapping_timing"]["wire_load"]
                );
                assert_eq!(
                    report["fanout_optimization"]["after"]["area"],
                    report["timing_sizing"]["initial_area"]
                );
            }
            let delay = standalone["final_delay"]["input_to_output"]
                .as_f64()
                .unwrap();
            assert!(delay > previous_delay);
            previous_delay = delay;
            check_and(dir.path(), &after)?;
        }
    }
    Ok(())
}

#[test]
fn timing_sizing_rejects_inapplicable_or_invalid_options_before_writing() -> anyhow::Result<()> {
    for (flags, error) in [
        (
            vec!["--timing-sizing"],
            "--timing-sizing requires --strategy area",
        ),
        (
            vec!["--strategy", "delay", "--timing-sizing"],
            "--timing-sizing requires --strategy area",
        ),
        (
            vec!["--timing-sizing-rounds", "0"],
            "timing sizing tuning options require --timing-sizing=true",
        ),
        (
            vec![
                "--strategy",
                "area",
                "--timing-sizing=false",
                "--timing-sizing-evaluation-budget",
                "1",
            ],
            "timing sizing tuning options require --timing-sizing=true",
        ),
        (
            vec![
                "--strategy",
                "area",
                "--timing-sizing",
                "--timing-sizing-max-area-growth=NaN",
            ],
            "--timing-sizing-max-area-growth must be finite and nonnegative",
        ),
        (
            vec![
                "--strategy",
                "area",
                "--timing-sizing",
                "--timing-sizing-max-area-growth=-0.01",
            ],
            "--timing-sizing-max-area-growth must be finite and nonnegative",
        ),
        (
            vec![
                "--strategy",
                "area",
                "--timing-sizing",
                "--timing-sizing-max-area-growth=inf",
            ],
            "--timing-sizing-max-area-growth must be finite and nonnegative",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
            .args([
                "aig-tech-map",
                "unused.aag",
                "--liberty_proto",
                "unused.proto",
                "--netlist_out",
                "-",
            ])
            .args(flags)
            .output()?;
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8(output.stderr)?,
            format!("aig-tech-map error: {error}\n")
        );
        assert!(output.stdout.is_empty());
    }
    Ok(())
}
