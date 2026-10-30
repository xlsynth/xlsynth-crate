// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use xlsynth_g8r::aig_serdes::g8r::parse_g8r;
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};
use xlsynth_g8r::prove_gate_fn_equiv_sat::CADICAL_CONFIG_ENV;
use xlsynth_pir::IrBits;

const SPEC: &str = r#"
dslx_input_file = "pipe.x"
dslx_top = "pipe"
flop_inputs = false
flop_outputs = true
"#;

const LIBERTY: &str = r#"
format_magic: 5496997758228255311
cells: {
  name: "DFF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  sequential: { state_var: "IQ" next_state: "D" clock_expr: "CLK" kind: SEQUENTIAL_KIND_FF }
}
cells: {
  name: "INV"
  pins: { name_string_id: 5 direction: INPUT }
  pins: { name_string_id: 6 direction: OUTPUT function_string_id: 7 }
}
interned_strings: ["D", "CLK", "Q", "IQ", "A", "Y", "!A"]
"#;

const NETLIST: &str = r#"
module mapped(out, clk, x);
  input x, clk;
  output out;
  wire middle;
  DFF r0 (.D(x), .CLK(clk), .Q(middle));
  DFF r1 (.D(middle), .CLK(clk), .Q(out));
endmodule
"#;

fn driver(directory: &Path, args: &[&str]) -> std::io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .current_dir(directory)
        .args(args)
        .output()
}

/// Writes a stitch spec with unregistered external inputs and outputs.
fn stage_fixture(directory: &Path, dslx: &str, gv: &str) -> std::io::Result<()> {
    fs::write(directory.join("pipe.x"), dslx)?;
    fs::write(directory.join("pipe.gv"), gv)?;
    fs::write(
        directory.join("pipe.toml"),
        SPEC.replace("flop_outputs = true", "flop_outputs = false"),
    )?;
    fs::write(directory.join("cells.textproto"), LIBERTY)
}

/// Builds a stitch-equiv command that reports the proof method.
fn stage_command(directory: &Path, options: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
    command
        .current_dir(directory)
        .args([
            "stitch-equiv",
            "--spec",
            "pipe.toml",
            "--netlist",
            "pipe.gv",
            "--liberty-proto",
            "cells.textproto",
            "--show-proof-plan",
        ])
        .args(options);
    command
}

const SPLIT: &str = r#"[
  {"selector":["root"],"command":{"Apply":"SplitOutputs"}},
  {"selector":["root","low"],"command":"Solve"},
  {"selector":["root","high"],"command":"Solve"}
]"#;

const BITWUZLA_PLAN: &str = "solver bitwuzla: behavioral XLS IR vs. mapped cells\n";
const CADICAL_PLAN: &str = "solver cadical: gatified reference vs. mapped cells\n";
const DEFAULT_SOLVER_PLAN: &str = if cfg!(feature = "has-bitwuzla") {
    BITWUZLA_PLAN
} else {
    CADICAL_PLAN
};

#[cfg(feature = "has-bitwuzla")]
#[test]
fn behavioral_stitch_bounds_helper_depth_before_parallel_translation() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("split.json"), SPLIT).unwrap();
    for depth in [64, 65, 256] {
        let mut source = "fn h0(x: u2) -> u2 { x }\n".to_owned();
        let last = depth - 2;
        for i in 1..=last {
            source.push_str(&format!("fn h{i}(x: u2) -> u2 {{ h{}(x) }}\n", i - 1));
        }
        // Visit h31 first. Its cached height still contributes to the full
        // path through h{last}, even when all of h31's callees are known.
        source.push_str(&format!(
            "fn pipe_cycle0(x: bool) -> u2 {{ h{last}(h31(x as u2)) }}"
        ));
        stage_fixture(
            dir.path(),
            &source,
            "module pipe(input x, output [1:0] out); assign out = {1'b0, x}; endmodule",
        )
        .unwrap();
        let proof = stage_command(
            dir.path(),
            &[
                "--solver",
                "bitwuzla",
                "--proof-mode",
                "stages",
                "--jobs",
                "2",
                "--stage-script",
                "pipe_cycle0=split.json",
            ],
        )
        .output()
        .unwrap();
        if depth == 64 {
            assert_eq!(
                proof.status.code(),
                Some(0),
                "{}",
                String::from_utf8_lossy(&proof.stderr)
            );
            assert_eq!(
                String::from_utf8(proof.stderr).unwrap(),
                format!("stage pipe_cycle0: 2 proved leaves\n{BITWUZLA_PLAN}")
            );
            // The maximum supported depth must also replay a concrete
            // mismatch on the main thread, including the full helper chain.
            fs::write(
                dir.path().join("pipe.gv"),
                "module pipe(input x, output [1:0] out); assign out = 2'b00; endmodule",
            )
            .unwrap();
            let replay = stage_command(
                dir.path(),
                &["--solver", "bitwuzla", "--proof-mode", "whole-pipeline"],
            )
            .output()
            .unwrap();
            assert_eq!(replay.status.code(), Some(1));
            assert!(replay.stdout.is_empty());
            assert_eq!(
                String::from_utf8(replay.stderr).unwrap(),
                format!(
                    r#"whole-pipeline proof
{BITWUZLA_PLAN}xlsynth-driver: stitch-equiv: post-fill output mismatch
Reference initial state {{}}
Mapped initial state {{}}
Frame 0 inputs {{"x": bits[1]:1}}
Reference outputs {{"out": bits[2]:1}}
Mapped outputs {{"out": bits[2]:0}}
"#
                )
            );
        } else {
            assert_eq!(proof.status.code(), Some(1));
            assert_eq!(
                String::from_utf8(proof.stderr).unwrap(),
                "xlsynth-driver: stitch-equiv: stage 'pipe_cycle0': Bitwuzla stitch proofs support at most 64 nested XLS functions including the stage. Inline helpers to shorten the call chain\n"
            );
        }
    }
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn behavioral_stitch_zero_extends_wide_loop_indices_for_proof_and_replay() {
    let dir = tempfile::tempdir().unwrap();
    // The three iterations set both low bits. Repeating either nonzero
    // index in the upper 128 bits would also set bit two of the result.
    stage_fixture(
        dir.path(),
        r#"fn pipe_cycle0() -> u3 {
    for (i, carry): (uN[256], u3) in uN[256]:0..uN[256]:3 {
        let high = or_reduce(i[128+:uN[128]]);
        carry | (high ++ i[0+:u2])
    }(u3:0)
}"#,
        "module pipe(output [2:0] out); assign out = 3'b011; endmodule",
    )
    .unwrap();
    let options = ["--solver", "bitwuzla", "--proof-mode", "whole-pipeline"];
    let proved = stage_command(dir.path(), &options).output().unwrap();
    assert_eq!(
        proved.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&proved.stderr)
    );
    assert_eq!(
        proved.stdout,
        b"Proved pipe equivalent to pipe after 0 rising clock edges.\n"
    );

    fs::write(
        dir.path().join("pipe.gv"),
        "module pipe(output [2:0] out); assign out = 3'b111; endmodule",
    )
    .unwrap();
    let disproved = stage_command(dir.path(), &options).output().unwrap();
    assert_eq!(disproved.status.code(), Some(1));
    assert!(disproved.stdout.is_empty());
    assert_eq!(
        String::from_utf8(disproved.stderr).unwrap(),
        format!(
            r#"whole-pipeline proof
{BITWUZLA_PLAN}xlsynth-driver: stitch-equiv: post-fill output mismatch
Reference initial state {{}}
Mapped initial state {{}}
Frame 0 inputs {{}}
Reference outputs {{"out": bits[3]:3}}
Mapped outputs {{"out": bits[3]:7}}
"#
        )
    );
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn behavioral_stitch_proves_and_replays_shared_calls_through_distinct_wrappers() {
    let dir = tempfile::tempdir().unwrap();
    let mut source = String::from("fn f0(a: bool, b: bool) -> bool { a & !b }\n");
    for level in 1..=24 {
        let prev = level - 1;
        // Both wrappers invoke the same child on the same operands. Their
        // different results make the callee part of the cache key observable.
        source.push_str(&format!(
            r#"fn left{level}(a: bool, b: bool) -> bool {{ f{prev}(a, b) }}
fn right{level}(a: bool, b: bool) -> bool {{ !f{prev}(a, b) }}
fn f{level}(a: bool, b: bool) -> bool {{ left{level}(a, b) | !right{level}(a, b) }}
"#
        ));
    }
    // The two argument orders give x and !x respectively at every level.
    source.push_str("fn pipe_cycle0(x: bool) -> u2 { f24(x, !x) ++ f24(!x, x) }\n");
    let mapped = r#"module pipe(input x, output [1:0] out);
  INV inv (.A(x), .Y(out[0]));
  assign out[1] = x;
endmodule"#;
    stage_fixture(dir.path(), &source, mapped).unwrap();
    let options = ["--solver", "bitwuzla", "--proof-mode", "whole-pipeline"];
    let proved = stage_command(dir.path(), &options).output().unwrap();
    assert_eq!(
        proved.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&proved.stderr)
    );
    assert_eq!(
        proved.stdout,
        b"Proved pipe equivalent to pipe after 0 rising clock edges.\n"
    );

    // Tying the inverter's input high gives a unique witness at x=0.
    fs::write(
        dir.path().join("pipe.gv"),
        mapped.replace(".A(x)", ".A(1'b1)"),
    )
    .unwrap();
    let disproved = stage_command(dir.path(), &options).output().unwrap();
    assert_eq!(disproved.status.code(), Some(1));
    assert!(disproved.stdout.is_empty());
    assert_eq!(
        String::from_utf8(disproved.stderr).unwrap(),
        format!(
            r#"whole-pipeline proof
{BITWUZLA_PLAN}xlsynth-driver: stitch-equiv: post-fill output mismatch
Reference initial state {{}}
Mapped initial state {{}}
Frame 0 inputs {{"x": bits[1]:0}}
Reference outputs {{"out": bits[2]:1}}
Mapped outputs {{"out": bits[2]:0}}
"#
        )
    );
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn behavioral_stitch_reads_stage_callees_in_a_source_with_an_unrelated_proc() {
    let dir = tempfile::tempdir().unwrap();
    stage_fixture(
        dir.path(),
        r#"fn invert(x: bool) -> bool { !x }
fn transform(x: bool) -> bool { map([x], invert)[u32:0] }
fn pipe_cycle0(x: bool) -> bool { transform(x) }

proc Counter {
    output: chan<u8> out;
    config(output: chan<u8> out) { (output,) }
    init { u8:0 }
    next(state: u8) {
        let tok = send(join(), output, state);
        state + u8:1
    }
}
"#,
        "module pipe(input x, output out); INV inv (.A(x), .Y(out)); endmodule",
    )
    .unwrap();
    let result = stage_command(
        dir.path(),
        &["--solver", "bitwuzla", "--proof-mode", "whole-pipeline"],
    )
    .output()
    .unwrap();
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"Proved pipe equivalent to pipe after 0 rising clock edges.\n"
    );
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        format!("whole-pipeline proof\n{BITWUZLA_PLAN}")
    );
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn behavioral_stitch_checks_xls_function_body_with_contradictory_ffi_metadata() {
    let dir = tempfile::tempdir().unwrap();
    stage_fixture(
        dir.path(),
        r#"#[extern_verilog("assign {return} = {x}; /* xlsynth_pir_ext=ext_mask_low;out_width=1;count_width=1 */")]
fn spec(x: bool) -> bool { !x }
fn pipe_cycle0(x: bool) -> bool { spec(x) }
"#,
        "module pipe(input x, output out); assign out = x; endmodule",
    ).unwrap();
    for (mode, diagnostic) in [
        (
            "stages",
            "xlsynth-driver: stitch-equiv: the proposed correspondence fails the one-step obligation for 'pipe_cycle0'. Use whole-pipeline mode to check the external outputs after fill",
        ),
        (
            "whole-pipeline",
            "xlsynth-driver: stitch-equiv: post-fill output mismatch",
        ),
    ] {
        let result = driver(
            dir.path(),
            &[
                "stitch-equiv",
                "--spec",
                "pipe.toml",
                "--netlist",
                "pipe.gv",
                "--liberty-proto",
                "cells.textproto",
                "--solver",
                "bitwuzla",
                "--proof-mode",
                mode,
            ],
        )
        .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
        assert_eq!(
            String::from_utf8(result.stderr).unwrap().lines().next(),
            Some(diagnostic)
        );
    }
}

#[test]
fn stitch_equiv_selects_the_solver_for_stages_and_whole_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    // The physical flop captures !x. Invert its Q to recover the DSLX stage
    // value.
    stage_fixture(
        dir.path(),
        "fn pipe_cycle0(x: bool) -> bool { x }\nfn pipe_cycle1(y: bool) -> bool { y }",
        r#"module pipe(input x, input clk, output out);
  wire nx, nq;
  INV invert_d (.A(x), .Y(nx));
  DFF r (.D(nx), .CLK(clk), .Q(nq));
  INV invert_q (.A(nq), .Y(out));
endmodule"#,
    )
    .unwrap();
    for (mode, method, error_context) in [
        (
            "stages",
            "stage pipe_cycle0: 1 proved leaf\nstage pipe_cycle1: 1 proved leaf\n",
            "stage 'pipe_cycle0': ",
        ),
        ("whole-pipeline", "whole-pipeline proof\n", ""),
    ] {
        let mut cadical = stage_command(dir.path(), &["--solver", "cadical", "--proof-mode", mode]);
        let proved = cadical.output().unwrap();
        assert_eq!(
            proved.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&proved.stderr)
        );
        assert_eq!(
            String::from_utf8(proved.stderr).unwrap(),
            format!("{method}{CADICAL_PLAN}")
        );

        // An invalid CaDiCaL configuration distinguishes the actual backend
        // from a command that only changes the reported solver name.
        let rejected = cadical.env(CADICAL_CONFIG_ENV, "invalid").output().unwrap();
        assert_eq!(rejected.status.code(), Some(1));
        assert_eq!(
            String::from_utf8(rejected.stderr).unwrap(),
            format!(
                "xlsynth-driver: stitch-equiv: {error_context}CadicalConfigError(\"invalid: invalid config\")\n"
            )
        );
        let bitwuzla = stage_command(dir.path(), &["--solver", "bitwuzla", "--proof-mode", mode])
            .env(CADICAL_CONFIG_ENV, "invalid")
            .output()
            .unwrap();
        if cfg!(feature = "has-bitwuzla") {
            assert_eq!(
                bitwuzla.status.code(),
                Some(0),
                "{}",
                String::from_utf8_lossy(&bitwuzla.stderr)
            );
            assert_eq!(
                String::from_utf8(bitwuzla.stderr).unwrap(),
                format!("{method}{BITWUZLA_PLAN}")
            );
        } else {
            assert_eq!(bitwuzla.status.code(), Some(1));
            assert_eq!(
                String::from_utf8(bitwuzla.stderr).unwrap(),
                "xlsynth-driver: stitch-equiv: Bitwuzla behavioral stitch proofs require a build with with-bitwuzla-system or with-bitwuzla-built. Use --solver cadical for a gate-level proof\n"
            );
        }
    }
}

#[test]
fn stitch_equiv_defaults_to_stages_and_accepts_one_scripted_override() {
    let dir = tempfile::tempdir().unwrap();
    let dslx = r#"fn pipe_cycle0(x: bool, y: bool) -> u2 { x ++ y }
fn pipe_cycle1(word: u2) -> u2 { word }"#;
    // Port and flop order are both reversed relative to the reference.
    let gv = r#"module pipe(output [1:0] out, input y, input clk, input x);
  DFF alpha (.D(x), .CLK(clk), .Q(out[1]));
  DFF zebra (.D(y), .CLK(clk), .Q(out[0]));
endmodule"#;
    stage_fixture(dir.path(), dslx, gv).unwrap();
    fs::write(dir.path().join("split.json"), SPLIT).unwrap();
    let default_plan = "stage pipe_cycle0: 1 proved leaf\nstage pipe_cycle1: 1 proved leaf\n";
    let split_plan = "stage pipe_cycle0: 2 proved leaves\nstage pipe_cycle1: 1 proved leaf\n";
    for (options, expected) in [
        (vec![], default_plan),
        (vec!["--proof-mode", "stages", "--jobs", "2"], default_plan),
        (
            vec!["--proof-mode", "whole-pipeline"],
            "whole-pipeline proof\n",
        ),
        (
            vec!["--jobs", "1", "--stage-script", "pipe_cycle0=split.json"],
            split_plan,
        ),
        (
            vec!["--jobs", "4", "--stage-script", "pipe_cycle0=split.json"],
            split_plan,
        ),
    ] {
        let result = stage_command(dir.path(), &options).output().unwrap();
        assert_eq!(
            result.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            result.stdout,
            b"Proved pipe equivalent to pipe after 1 rising clock edge.\n"
        );
        assert_eq!(
            String::from_utf8(result.stderr).unwrap(),
            format!("{expected}{DEFAULT_SOLVER_PLAN}")
        );
    }
    fs::write(dir.path().join("unresolved.json"),
        r#"[{"selector":["root"],"command":{"Apply":"SplitOutputs"}},{"selector":["root","low"],"command":"Solve"}]"#
    ).unwrap();
    fs::write(
        dir.path().join("wrong.json"),
        r#"[{"selector":["other"],"command":"Solve"}]"#,
    )
    .unwrap();
    for (options, message) in [
        (
            vec!["--stage-script", "pipe_cycle1=wrong.json"],
            r#"stage 'pipe_cycle1' script 'wrong.json': selector ["other"] does not name an open leaf"#,
        ),
        (
            vec!["--stage-script", "pipe_cycle1=unresolved.json"],
            r#"stage 'pipe_cycle1' script 'unresolved.json': stage proof script leaves ["root", "high"] unsolved"#,
        ),
        (
            vec![
                "--stage-script",
                "pipe_cycle1=split.json",
                "--stage-script",
                "pipe_cycle1=split.json",
            ],
            "more than one script supplied for stage 'pipe_cycle1'",
        ),
        (
            vec!["--stage-script", "unknown=split.json"],
            "unknown scripted stage 'unknown'. Available stages: pipe_cycle0, pipe_cycle1",
        ),
        (
            vec![
                "--proof-mode",
                "whole-pipeline",
                "--stage-script",
                "pipe_cycle1=split.json",
            ],
            "whole-pipeline proof mode cannot run per-stage scripts",
        ),
    ] {
        let result = stage_command(dir.path(), &options).output().unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
        assert_eq!(
            String::from_utf8(result.stderr).unwrap(),
            format!("xlsynth-driver: stitch-equiv: {message}\n")
        );
    }
    fs::write(dir.path().join("pipe.gv"),
        "module pipe(output [1:0] out, input y, input clk, input x); assign out = {x, y}; endmodule"
    ).unwrap();
    let missing = stage_command(dir.path(), &["--stage-script", "pipe_cycle0=split.json"])
        .output()
        .unwrap();
    assert_eq!(missing.status.code(), Some(1));
    assert!(missing.stdout.is_empty());
    assert_eq!(
        String::from_utf8(missing.stderr).unwrap(),
        "xlsynth-driver: stitch-equiv: cannot run per-stage scripts without a direct register correspondence: no direct mapped partner for reference boundary register 'p0_word[0]' (2 reference bits, 0 mapped bits)\n"
    );
}

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
#[test]
fn stitch_equiv_runs_stages_serially_when_workers_cannot_start() {
    let dir = tempfile::tempdir().unwrap();
    stage_fixture(
        dir.path(),
        "fn pipe_cycle0(x: bool) -> bool { x }\nfn pipe_cycle1(y: bool) -> bool { y }",
        "module pipe(input x, input clk, output out); DFF r (.D(x), .CLK(clk), .Q(out)); endmodule",
    )
    .unwrap();
    // A 2^62-byte stack exceeds the Linux user address space, so the child
    // cannot spawn Rust worker threads, even when running as root in CI.
    let result = stage_command(dir.path(), &["--jobs", "2", "--proof-mode", "stages"])
        .env("RUST_MIN_STACK", (1_u64 << 62).to_string())
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        format!(
            "stage pipe_cycle0: 1 proved leaf\nstage pipe_cycle1: 1 proved leaf\n{DEFAULT_SOLVER_PLAN}"
        )
    );
}

#[test]
fn stitch_equiv_distinguishes_failed_stage_relations_from_post_fill_mismatches() {
    let dir = tempfile::tempdir().unwrap();
    // Both stored bits always capture x. The stage's OR and the mapped wire
    // disagree for an arbitrary cutpoint assignment, but agree after one edge.
    let dslx = r#"fn pipe_cycle0(x: bool) -> u2 { x ++ x }
fn pipe_cycle1(pair: u2) -> bool { or_reduce(pair) }"#;
    let gv = r#"module pipe(input x, input clk, output out);
  wire a, b;
  DFF ra (.D(x), .CLK(clk), .Q(a));
  DFF rb (.D(x), .CLK(clk), .Q(b));
  assign out = b;
endmodule"#;
    stage_fixture(dir.path(), dslx, gv).unwrap();
    let reason = "the proposed correspondence fails the one-step obligation for 'pipe_cycle1'";
    let strict = stage_command(dir.path(), &["--proof-mode", "stages"])
        .output()
        .unwrap();
    assert_eq!(strict.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(strict.stderr).unwrap(),
        format!(
            "xlsynth-driver: stitch-equiv: {reason}. Use whole-pipeline mode to check the external outputs after fill\n"
        )
    );
    let auto = stage_command(dir.path(), &[]).output().unwrap();
    assert_eq!(auto.status.code(), Some(0));
    assert_eq!(
        auto.stdout,
        b"Proved pipe equivalent to pipe after 1 rising clock edge.\n"
    );
    assert_eq!(
        String::from_utf8(auto.stderr).unwrap(),
        format!("whole-pipeline fallback: {reason}\n{DEFAULT_SOLVER_PLAN}")
    );

    fs::write(
        dir.path().join("pipe.gv"),
        gv.replace("assign out = b;", "assign out = x;"),
    )
    .unwrap();
    let corrupt = stage_command(dir.path(), &[]).output().unwrap();
    assert_eq!(corrupt.status.code(), Some(1));
    assert!(corrupt.stdout.is_empty());
    let stderr = String::from_utf8(corrupt.stderr).unwrap();
    // The solver may choose either direction of the transition in its witness.
    assert_eq!(
        stderr.lines().take(3).collect::<Vec<_>>(),
        [
            format!("whole-pipeline fallback: {reason}"),
            DEFAULT_SOLVER_PLAN.trim_end().into(),
            "xlsynth-driver: stitch-equiv: post-fill output mismatch".into()
        ]
    );
    assert!(stderr.contains("Frame 0 inputs") && stderr.contains("Frame 1 inputs"));

    // Synthesis can merge the identical flops and still implement the delay.
    fs::write(
        dir.path().join("pipe.gv"),
        "module pipe(input x, input clk, output out); DFF r (.D(x), .CLK(clk), .Q(out)); endmodule",
    )
    .unwrap();
    let merged = stage_command(dir.path(), &[]).output().unwrap();
    assert_eq!(merged.status.code(), Some(0));
    assert_eq!(
        String::from_utf8(merged.stderr).unwrap(),
        format!(
            "whole-pipeline fallback: no direct register correspondence: register bit counts differ (reference 2, mapped 1)\n{DEFAULT_SOLVER_PLAN}"
        )
    );
}

/// Checks the product and addend at the register separating the example's two
/// operations.
#[test]
fn stitch_muladd_registers_the_product_before_adding() {
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/stitch-equiv");
    let generated = driver(
        &example,
        &["dslx-stitch-g8r-pipeline", "--spec", "muladd.stitch.toml"],
    )
    .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let design = parse_g8r(std::str::from_utf8(&generated.stdout).unwrap()).unwrap();
    let mut state = SequentialState::all_zeros(&design);
    // Each row gives this edge's operands, its registered {product, z}, and
    // the registered sum from the preceding row. Both operations wrap to u8.
    for (operands, partials, sum) in [
        ([7, 9, 5], 0x3f05, 0),
        ([255, 2, 3], 0xfe03, 68),
        ([0, 0, 0], 0, 1),
    ] {
        let inputs = operands.map(|v| IrBits::make_ubits(8, v).unwrap());
        let edge = simulate(&design, &[inputs.to_vec()], state).unwrap();
        assert_eq!(
            edge.final_state().values(),
            [
                IrBits::make_ubits(16, partials).unwrap(),
                IrBits::make_ubits(8, sum).unwrap()
            ],
            "register values after applying {operands:?}"
        );
        state = edge.final_state().clone();
    }
}

/// Both emitters consume the same relative-path spec and preserve their
/// flag-based output.
#[test]
fn stitch_generators_load_spec_from_another_directory() {
    let temp = tempfile::tempdir().unwrap();
    let design = temp.path().join("design");
    fs::create_dir(&design).unwrap();
    fs::write(design.join("pipe.stitch.toml"), SPEC).unwrap();
    fs::write(design.join("pipe.x"), "fn pipe_cycle0(x: u2) -> u2 { x }").unwrap();
    for command in ["dslx-stitch-pipeline", "dslx-stitch-g8r-pipeline"] {
        let from_spec =
            driver(temp.path(), &[command, "--spec", "design/pipe.stitch.toml"]).unwrap();
        let from_flags = driver(
            temp.path(),
            &[
                command,
                "--dslx_input_file",
                "design/pipe.x",
                "--dslx_top",
                "pipe",
                "--flop_inputs=false",
                "--flop_outputs=true",
            ],
        )
        .unwrap();
        for output in [&from_spec, &from_flags] {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        assert_eq!(from_spec.stdout, from_flags.stdout);
        let conflicting = driver(
            temp.path(),
            &[
                command,
                "--spec",
                "design/pipe.stitch.toml",
                "--flop_inputs=true",
            ],
        )
        .unwrap();
        assert_eq!(conflicting.status.code(), Some(2));
    }
}

/// A working-directory module can change the generated logic even when the
/// shadowed transitive import contributes only a constant to the stage.
#[test]
fn stitch_spec_imports_are_independent_of_the_working_directory() {
    let temp = tempfile::tempdir().unwrap();
    let design = temp.path().join("design");
    let imports = design.join("imports");
    let callers = temp.path().join("callers");
    let clean = callers.join("clean");
    let direct = callers.join("direct");
    let transitive = callers.join("transitive");
    fs::create_dir_all(&imports).unwrap();
    fs::create_dir_all(&clean).unwrap();
    fs::create_dir(&direct).unwrap();
    fs::create_dir(&transitive).unwrap();
    fs::write(
        design.join("pipe.toml"),
        format!("{SPEC}\ndslx_path = [\"imports\"]\n"),
    )
    .unwrap();
    fs::write(
        design.join("pipe.x"),
        r#"import stitch_helper;
fn pipe_cycle0(x: bool) -> bool { stitch_helper::apply(x) }
"#,
    )
    .unwrap();
    fs::write(
        imports.join("stitch_helper.x"),
        r#"import stitch_constant;
pub fn apply(x: bool) -> bool { x ^ stitch_constant::INVERT }
"#,
    )
    .unwrap();
    fs::write(
        imports.join("stitch_constant.x"),
        "pub const INVERT = false;",
    )
    .unwrap();
    fs::write(
        direct.join("stitch_helper.x"),
        "pub fn apply(x: bool) -> bool { !x }",
    )
    .unwrap();
    fs::write(
        transitive.join("stitch_constant.x"),
        "pub const INVERT = true;",
    )
    .unwrap();

    fs::write(
        callers.join("pipe.gv"),
        r#"module pipe(input clk, input x, output out);
  DFF r0 (.D(x), .CLK(clk), .Q(out));
endmodule"#,
    )
    .unwrap();
    fs::write(callers.join("cells.textproto"), LIBERTY).unwrap();
    for command in [
        "dslx-stitch-pipeline",
        "dslx-stitch-g8r-pipeline",
        "stitch-equiv",
    ] {
        let mut args = vec![command, "--spec", "../../design/pipe.toml"];
        if command == "dslx-stitch-g8r-pipeline" {
            args.extend(["--bin-out", "pipe.g8rbin"]);
        }
        if command == "stitch-equiv" {
            args.extend([
                "--netlist",
                "../pipe.gv",
                "--liberty-proto",
                "../cells.textproto",
            ]);
        }
        let expected = driver(&clean, &args).unwrap();
        assert!(
            expected.status.success(),
            "{command}: {}",
            String::from_utf8_lossy(&expected.stderr)
        );
        for cwd in [&direct, &transitive] {
            let shadowed = driver(cwd, &args).unwrap();
            assert!(
                shadowed.status.success(),
                "{command}: {}",
                String::from_utf8_lossy(&shadowed.stderr)
            );
            assert_eq!(shadowed.stdout, expected.stdout, "{command} from {cwd:?}");
            if command == "dslx-stitch-g8r-pipeline" {
                assert_eq!(
                    fs::read(cwd.join("pipe.g8rbin")).unwrap(),
                    fs::read(clean.join("pipe.g8rbin")).unwrap()
                );
            }
        }
    }
}

/// A valid-enabled design needs a different observation contract even when
/// the mapped implementation uses the same control names.
#[test]
fn stitch_equiv_reports_configured_valid_and_reset_ports() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(x: bool) -> bool { x }",
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        format!(
            "{SPEC}\n{}",
            r#"input_valid_signal = "vin"
output_valid_signal = "vout"
reset = "rst_n"
reset_active_low = true"#
        ),
    )
    .unwrap();
    let result = driver(
        temp.path(),
        &[
            "stitch-equiv",
            "--spec",
            "pipe.toml",
            "--netlist",
            "unused.gv",
            "--liberty-proto",
            "unused.proto",
        ],
    )
    .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        r#"xlsynth-driver: stitch-equiv: unsupported stitch controls for 'pipe': input_valid_signal = "vin", output_valid_signal = "vout", reset = "rst_n". stitch-equiv checks free-running pipelines and compares data on every cycle after fill. Valid-qualified observations and reset sequences are not yet supported
"#
    );
}

/// Every source stage must resolve before proving the latency of the selected
/// pipeline, including stages lowered with the implicit-token convention.
#[test]
fn stitch_equiv_requires_a_data_entry_point_for_every_stage() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("pipe.toml"), SPEC).unwrap();
    let source = temp.path().join("pipe.x");
    let private_stage = r#"fn pipe_cycle0(x: bool) -> bool { x }
fn pipe_cycle1(x: bool) -> bool { assert!(x, "OK"); x }
"#;
    fs::write(&source, private_stage).unwrap();
    let args = [
        "stitch-equiv",
        "--spec",
        "pipe.toml",
        "--netlist",
        "mapped.gv",
        "--liberty-proto",
        "cells.textproto",
        "--module-name",
        "mapped",
    ];
    // One flop exactly implements the silently truncated first stage.
    fs::write(
        temp.path().join("mapped.gv"),
        r#"module mapped(input x, input clk, output out);
  DFF r0 (.D(x), .CLK(clk), .Q(out));
endmodule"#,
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    let result = driver(temp.path(), &args).unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        "xlsynth-driver: stitch-equiv: xlsynth error: stage 'pipe_cycle1' uses the implicit-token calling convention and has no data-only IR entry point. Declare the stage pub to emit a wrapper for its packed data ports\n"
    );

    // XLS emits a data-only wrapper for the public stage. Both stages must
    // contribute to the proved data path and its latency.
    fs::write(
        &source,
        private_stage.replace("fn pipe_cycle1", "pub fn pipe_cycle1"),
    )
    .unwrap();
    fs::write(temp.path().join("mapped.gv"), NETLIST).unwrap();
    let result = driver(temp.path(), &args).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"Proved pipe equivalent to mapped after 2 rising clock edges.\n"
    );
}

/// Reports the selected DSLX stage and IR operation when gatification fails.
#[test]
fn stitch_equiv_reports_unsupported_stage_operation() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(x: u4, y: u4) -> (u4, u4) { umulp(x, y) }",
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        SPEC.replace("flop_outputs = true", "flop_outputs = false"),
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.gv"),
        "module pipe(input [3:0] x, input [3:0] y, output [7:0] out); assign out = {x, y}; endmodule",
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    let result = driver(
        temp.path(),
        &[
            "stitch-equiv",
            "--solver",
            "cadical",
            "--spec",
            "pipe.toml",
            "--netlist",
            "pipe.gv",
            "--liberty-proto",
            "cells.textproto",
        ],
    )
    .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        concat!(
            "xlsynth-driver: stitch-equiv: xlsynth error: failed to lower pipeline stage ",
            "'pipe_cycle0' to g8r: Unsupported node payload Binop(Umulp, ",
            "NodeRef { index: 1 }, NodeRef { index: 2 })\n",
        )
    );
}

/// The post-fill proof detects a bypassed register even when the arithmetic is
/// identical.
#[test]
fn stitch_equiv_checks_clocked_netlist() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("pipe.stitch.toml"), SPEC).unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        r#"
fn pipe_cycle0(x: u1) -> u1 { x }
fn pipe_cycle1(x: u1) -> u1 { x }
"#,
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    fs::write(temp.path().join("mapped.gv"), NETLIST).unwrap();
    let args = [
        "stitch-equiv",
        "--spec",
        "pipe.stitch.toml",
        "--netlist",
        "mapped.gv",
        "--liberty-proto",
        "cells.textproto",
        "--module-name",
        "mapped",
    ];
    let result = driver(temp.path(), &args).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"Proved pipe equivalent to mapped after 2 rising clock edges.\n"
    );

    fs::write(
        temp.path().join("mapped.gv"),
        NETLIST.replace(".D(middle)", ".D(x)"),
    )
    .unwrap();
    let wrong_latency = driver(temp.path(), &args).unwrap();
    assert_eq!(wrong_latency.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&wrong_latency.stderr).contains("post-fill output mismatch"));
}

/// The stage boundary gives the first parameter the low bit and the second
/// parameter the next two bits of the registered tuple.
#[test]
fn stitch_equiv_checks_unequal_width_stage_parameter_slices() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        r#"fn pipe_cycle0(x: u3) -> (u1, u2) { (x[2+:u1], x[0+:u2]) }
fn pipe_cycle1(a: u1, b: u2) -> u3 { a ++ b }
"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        SPEC.replace("flop_outputs = true", "flop_outputs = false"),
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    let mapped = temp.path().join("pipe.gv");
    // The first stage packs {x[2], x[1:0]}, so its registered result is q.
    // The second stage concatenates its two parameter slices in order.
    let gv = r#"module pipe(input clk, input [2:0] x, output [2:0] out);
  wire [2:0] q;
  DFF r0 (.CLK(clk), .D(x[0]), .Q(q[0]));
  DFF r1 (.CLK(clk), .D(x[1]), .Q(q[1]));
  DFF r2 (.CLK(clk), .D(x[2]), .Q(q[2]));
  assign out = {q[0], q[2:1]};
endmodule"#;
    let args = [
        "stitch-equiv",
        "--spec",
        "pipe.toml",
        "--netlist",
        "pipe.gv",
        "--liberty-proto",
        "cells.textproto",
    ];
    fs::write(&mapped, gv).unwrap();
    let result = driver(temp.path(), &args).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"Proved pipe equivalent to pipe after 1 rising clock edge.\n"
    );

    fs::write(&mapped, gv.replace("{q[0], q[2:1]}", "{q[2], q[1:0]}")).unwrap();
    let reversed = driver(temp.path(), &args).unwrap();
    assert_eq!(reversed.status.code(), Some(1));
    assert!(reversed.stdout.is_empty());
    // The solver can choose any input whose rotated value differs from x.
    assert_eq!(
        String::from_utf8(reversed.stderr).unwrap().lines().next(),
        Some("xlsynth-driver: stitch-equiv: post-fill output mismatch")
    );
}

/// Native clockless mapping and an RTL wrapper's unused clock have the same
/// zero-edge contract.
#[test]
fn stitch_equiv_checks_gate_semantics_with_and_without_a_clock_port() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(x: bool, en: bool) -> bool { gate!(en, x) }",
    )
    .unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        SPEC.replace("flop_outputs = true", "flop_outputs = false"),
    )
    .unwrap();
    fs::write(
        temp.path().join("cells.textproto"),
        r#"
format_magic: 5496997758228255311
cells: {
  name: "AND2"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
}

interned_strings: ["A", "B", "Y", "A & B"]
"#,
    )
    .unwrap();
    for (clock, out) in [("", "output_value"), ("input clk, ", "out")] {
        fs::write(temp.path().join("pipe.gv"), format!(
            "module pipe({clock}input en, input x, output {out}); AND2 g(.A(en), .B(x), .Y({out})); endmodule"
        )).unwrap();
        let result = driver(
            temp.path(),
            &[
                "stitch-equiv",
                "--spec",
                "pipe.toml",
                "--netlist",
                "pipe.gv",
                "--liberty-proto",
                "cells.textproto",
            ],
        )
        .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            result.stdout,
            b"Proved pipe equivalent to pipe after 0 rising clock edges.\n"
        );
    }
}

/// The CLI preserves the Liberty decoder's remediation for an obsolete proto.
#[test]
fn stitch_equiv_reports_unsupported_liberty_header() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("pipe.toml"), SPEC).unwrap();
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(x: u1) -> u1 { x }",
    )
    .unwrap();
    fs::write(temp.path().join("mapped.gv"), NETLIST).unwrap();
    fs::write(temp.path().join("old.proto"), []).unwrap();
    let result = driver(
        temp.path(),
        &[
            "stitch-equiv",
            "--spec",
            "pipe.toml",
            "--netlist",
            "mapped.gv",
            "--liberty-proto",
            "old.proto",
            "--module-name",
            "mapped",
        ],
    )
    .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        concat!(
            "xlsynth-driver: stitch-equiv: decoding Liberty proto 'old.proto': ",
            "invalid Liberty proto header: magic=0x0000000000000000; ",
            "regenerate from source Liberty with the current importer (older protos may omit functional semantics)\n",
        )
    );
}

/// An input boundary without data or valid adds no fill edge, including in a
/// pipeline that still has registers between stages.
#[test]
fn stitch_equiv_observes_nullary_stages_at_their_actual_latency() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("pipe.x");
    let mapped = temp.path().join("pipe.gv");
    let nullary_stage = "fn pipe_cycle0() -> bool { true }";
    fs::write(&source, nullary_stage).unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        r#"dslx_input_file = "pipe.x"
dslx_top = "pipe"
flop_inputs = true
flop_outputs = false"#,
    )
    .unwrap();
    fs::write(temp.path().join("cells.textproto"), LIBERTY).unwrap();
    let args = [
        "stitch-equiv",
        "--spec",
        "pipe.toml",
        "--netlist",
        "pipe.gv",
        "--liberty-proto",
        "cells.textproto",
    ];
    for clock in ["", "input clk, "] {
        fs::write(
            &mapped,
            format!("module pipe({clock}output out); assign out = 1'b1; endmodule"),
        )
        .unwrap();
        let result = driver(temp.path(), &args).unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            result.stdout,
            b"Proved pipe equivalent to pipe after 0 rising clock edges.\n"
        );
    }

    fs::write(
        &mapped,
        r#"module pipe(input clk, output out);
  DFF r0 (.D(1'b1), .CLK(clk), .Q(out));
endmodule"#,
    )
    .unwrap();
    let spurious_flop = driver(temp.path(), &args).unwrap();
    assert_eq!(spurious_flop.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(spurious_flop.stderr).unwrap(),
        r#"xlsynth-driver: stitch-equiv: post-fill output mismatch
Reference initial state {}
Mapped initial state {"r0__IQ": bits[1]:0}
Frame 0 inputs {}
Reference outputs {"out": bits[1]:1}
Mapped outputs {"out": bits[1]:0}
"#
    );

    fs::write(
        &source,
        format!("{nullary_stage}\nfn pipe_cycle1(x: bool) -> bool {{ x }}"),
    )
    .unwrap();
    let one_boundary = driver(temp.path(), &args).unwrap();
    assert!(
        one_boundary.status.success(),
        "{}",
        String::from_utf8_lossy(&one_boundary.stderr)
    );
    assert_eq!(
        one_boundary.stdout,
        b"Proved pipe equivalent to pipe after 1 rising clock edge.\n"
    );

    fs::write(
        &mapped,
        r#"module pipe(input clk, output out);
  wire m;
  DFF r0 (.D(1'b1), .CLK(clk), .Q(m));
  DFF r1 (.D(m), .CLK(clk), .Q(out));
endmodule"#,
    )
    .unwrap();
    let extra_edge = driver(temp.path(), &args).unwrap();
    assert_eq!(extra_edge.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&extra_edge.stderr).contains("post-fill output mismatch"));
}
