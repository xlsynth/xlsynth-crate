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
cells: {
  name: "AND2"
  pins: { name_string_id: 5 direction: INPUT }
  pins: { name_string_id: 8 direction: INPUT }
  pins: { name_string_id: 6 direction: OUTPUT function_string_id: 9 }
}
cells: {
  name: "OR2"
  pins: { name_string_id: 5 direction: INPUT }
  pins: { name_string_id: 8 direction: INPUT }
  pins: { name_string_id: 6 direction: OUTPUT function_string_id: 10 }
}
interned_strings: ["D", "CLK", "Q", "IQ", "A", "Y", "!A", "B", "A & B", "A | B"]
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
const STITCH_SOLVERS: &[(&str, &str)] = &[
    ("cadical", CADICAL_PLAN),
    #[cfg(feature = "has-bitwuzla")]
    ("bitwuzla", BITWUZLA_PLAN),
];
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
                "xlsynth-driver: stitch-equiv: stage 'pipe_cycle0': Stitch stage checks support at most 64 nested XLS functions including the stage. Inline helpers to shorten the call chain\n"
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
    fs::write(
        callers.join("inputs.irvals"),
        "{x: bits[1]:0}\n{x: bits[1]:1}\n",
    )
    .unwrap();
    for command in [
        "dslx-stitch-pipeline",
        "dslx-stitch-g8r-pipeline",
        "stitch-equiv",
        "stitch-check-vectors",
    ] {
        let mut args = vec![command, "--spec", "../../design/pipe.toml"];
        if command == "dslx-stitch-g8r-pipeline" {
            args.extend(["--bin-out", "pipe.g8rbin"]);
        }
        if command == "stitch-check-vectors" {
            args.extend(["--input-irvals", "../inputs.irvals"]);
        }
        if matches!(command, "stitch-equiv" | "stitch-check-vectors") {
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

/// Two-stage identity pipeline with synchronous active-low reset and a mapped
/// datapath that continues to capture data during bubbles.
fn valid_reset_fixture(directory: &Path) -> std::io::Result<&'static str> {
    let source = r#"fn pipe_cycle0(x: bool) -> bool { x }
fn pipe_cycle1(y: bool) -> bool { y }"#;
    let gv = r#"module pipe(input x, input clk, input vin, input rst_n, output vout, output out);
  wire data0, valid0, d0, d1;
  DFF data_ff0 (.D(x), .CLK(clk), .Q(data0));
  DFF data_ff1 (.D(data0), .CLK(clk), .Q(out));
  AND2 flush0 (.A(vin), .B(rst_n), .Y(d0));
  DFF valid_ff0 (.D(d0), .CLK(clk), .Q(valid0));
  AND2 flush1 (.A(valid0), .B(rst_n), .Y(d1));
  DFF valid_ff1 (.D(d1), .CLK(clk), .Q(vout));
endmodule"#;
    stage_fixture(directory, source, gv)?;
    fs::write(
        directory.join("pipe.toml"),
        format!(
            r#"{SPEC}
input_valid_signal = "vin"
output_valid_signal = "vout"
reset = "rst_n"
reset_active_low = true"#
        ),
    )?;
    Ok(gv)
}

#[test]
fn stitch_equiv_checks_valid_data_and_synchronous_reset() {
    let temp = tempfile::tempdir().unwrap();
    let gv = valid_reset_fixture(temp.path()).unwrap();
    let mapped = temp.path().join("pipe.gv");
    for &(solver, _) in STITCH_SOLVERS {
        fs::write(&mapped, gv).unwrap();
        let mut command = stage_command(temp.path(), &["--solver", solver]);
        let proved = command.output().unwrap();
        assert!(
            proved.status.success(),
            "{}",
            String::from_utf8_lossy(&proved.stderr)
        );
        assert_eq!(
            proved.stdout,
            b"Proved pipe equivalent to pipe under the stitch valid protocol (2 rising clock edges of latency).\n"
        );

        // The extra flops are zero after two edges, so their initial bits
        // cannot disturb a post-fill observation. They can still raise vout
        // immediately after a reset edge, when it must be known low.
        let stale_valid = gv.replace(".Q(vout)", ".Q(clean_valid)").replace(
            "  DFF valid_ff1",
            r#"  wire clean_valid, stale0, stale1;
  DFF extra0 (.D(1'b0), .CLK(clk), .Q(stale0));
  DFF extra1 (.D(stale0), .CLK(clk), .Q(stale1));
  OR2 leak (.A(clean_valid), .B(stale1), .Y(vout));
  DFF valid_ff1"#,
        );
        command.args(["--proof-mode", "whole-pipeline"]);
        for (defect, broken) in [
            ("payload bypass", gv.replace(".D(data0)", ".D(x)")),
            ("early valid", gv.replace(".D(d1)", ".D(d0)")),
            (
                "reset leaks an in-flight transaction",
                gv.replace(".D(d1)", ".D(valid0)"),
            ),
            ("pre-fill stale valid", stale_valid),
        ] {
            fs::write(&mapped, broken).unwrap();
            let result = command.output().unwrap();
            let stderr = String::from_utf8(result.stderr).unwrap();
            assert_eq!(
                result.status.code(),
                Some(1),
                "{solver}, {defect}: {stderr}"
            );
            assert!(
                stderr
                    .lines()
                    .any(|line| line
                        == "xlsynth-driver: stitch-equiv: valid-qualified output mismatch"),
                "{solver}, {defect}: {stderr}"
            );
        }
    }
}

/// A clockless, valid-enabled pipeline compares data on the current valid
/// input. Synchronous reset has no edge on which to act.
#[test]
fn stitch_equiv_qualifies_data_without_an_output_valid_port() {
    let temp = tempfile::tempdir().unwrap();
    let gv = r#"module pipe(input x, input vin, input rst, output out);
  AND2 mask (.A(x), .B(vin), .Y(out));
endmodule"#;
    stage_fixture(temp.path(), "fn pipe_cycle0(x: bool) -> bool { x }", gv).unwrap();
    let spec = SPEC.replace("flop_outputs = true", "flop_outputs = false");
    fs::write(
        temp.path().join("pipe.toml"),
        format!(
            r#"{spec}
input_valid_signal = "vin"
reset = "rst""#
        ),
    )
    .unwrap();
    let mapped = temp.path().join("pipe.gv");
    for &(solver, _) in STITCH_SOLVERS {
        fs::write(&mapped, gv).unwrap();
        let mut command = stage_command(
            temp.path(),
            &["--solver", solver, "--proof-mode", "whole-pipeline"],
        );
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        // This incorrectly masks a valid payload during an asserted reset.
        let gated_on_reset = gv
            .replace(
                "  AND2 mask",
                "  wire running;\n  INV inv (.A(rst), .Y(running));\n  AND2 mask",
            )
            .replace(".B(vin)", ".B(running)");
        fs::write(&mapped, gated_on_reset).unwrap();
        let result = command.output().unwrap();
        assert_eq!(result.status.code(), Some(1));
        let stderr = String::from_utf8(result.stderr).unwrap();
        assert!(
            stderr.lines().any(|line| line == "xlsynth-driver: stitch-equiv: valid-qualified output mismatch"),
            "{stderr}"
        );
    }
}

/// Both stage solvers discharge the valid-register reset and the data
/// register's hold-on-bubble obligation.
#[test]
fn stitch_equiv_proves_valid_and_reset_in_stages_mode() {
    let temp = tempfile::tempdir().unwrap();
    let gv = r#"module pipe(input clk, input x, input vin, input rst, output out, output vout);
  wire sample, bubble, held, d, running, next_valid;
  AND2 capture (.A(x), .B(vin), .Y(sample));
  INV invert_valid (.A(vin), .Y(bubble));
  AND2 hold (.A(out), .B(bubble), .Y(held));
  OR2 mux (.A(sample), .B(held), .Y(d));
  DFF data_ff (.D(d), .CLK(clk), .Q(out));
  INV invert_reset (.A(rst), .Y(running));
  AND2 flush (.A(vin), .B(running), .Y(next_valid));
  DFF valid_ff (.D(next_valid), .CLK(clk), .Q(vout));
endmodule"#;
    stage_fixture(temp.path(), "fn pipe_cycle0(x: bool) -> bool { x }", gv).unwrap();
    let input_flop = SPEC
        .replace("flop_inputs = false", "flop_inputs = true")
        .replace("flop_outputs = true", "flop_outputs = false");
    // For an identity stage the same two physical flops can implement either
    // wrapper boundary. The stage proof has different D roots in each case.
    for boundary in [SPEC, &input_flop] {
        fs::write(
            temp.path().join("pipe.toml"),
            format!(
                r#"{boundary}
input_valid_signal = "vin"
output_valid_signal = "vout"
reset = "rst""#
            ),
        )
        .unwrap();
        for &(solver, plan) in STITCH_SOLVERS {
            let result =
                stage_command(temp.path(), &["--solver", solver, "--proof-mode", "stages"])
                    .output()
                    .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            assert_eq!(
                String::from_utf8(result.stderr).unwrap(),
                format!("stage pipe_cycle0: 1 proved leaf\n{plan}")
            );
        }
    }
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

    // With input valid, a nullary stage still has a one-bit input register
    // layer. The constant payload requires no flops.
    fs::write(&source, nullary_stage).unwrap();
    fs::write(
        temp.path().join("pipe.toml"),
        r#"dslx_input_file = "pipe.x"
dslx_top = "pipe"
flop_inputs = true
flop_outputs = false
input_valid_signal = "vin"
output_valid_signal = "vout""#,
    )
    .unwrap();
    fs::write(
        &mapped,
        r#"module pipe(input clk, input vin, output out, output vout);
  DFF valid_ff (.D(vin), .CLK(clk), .Q(vout));
  assign out = 1'b1;
endmodule"#,
    )
    .unwrap();
    for &(solver, _) in STITCH_SOLVERS {
        let result = stage_command(temp.path(), &["--solver", solver, "--proof-mode", "stages"])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(result.stdout, b"Proved pipe equivalent to pipe under the stitch valid protocol (1 rising clock edge of latency).\n");
    }
}

/// Runs the concrete checker on the same DSLX, netlist, and cell fixtures as
/// the stitch proof tests.
fn check_vectors(directory: &Path, options: &[&str]) -> std::io::Result<Output> {
    let mut args = vec![
        "stitch-check-vectors",
        "--spec",
        "pipe.toml",
        "--netlist",
        "pipe.gv",
        "--liberty-proto",
        "cells.textproto",
    ];
    args.extend_from_slice(options);
    driver(directory, &args)
}

#[test]
fn stitch_vectors_reject_ascending_packed_ranges() {
    let temp = tempfile::tempdir().unwrap();
    stage_fixture(
        temp.path(),
        "fn pipe_cycle0(x: u4) -> u4 { x }",
        "module pipe(input [3:0] x, output [0:3] out); assign out = x; endmodule",
    )
    .unwrap();
    let result = check_vectors(temp.path(), &["--random-samples", "1"]).unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert_eq!(
        result.stderr,
        b"xlsynth-driver: stitch-check-vectors: mapped port 'out' has Verilog index 3 at bit offset 0, expected packed range [3:0] with index 0 as the LSB\n"
    );
}

#[test]
fn stitch_vectors_flush_in_flight_data_and_drain_the_last_transaction() {
    let temp = tempfile::tempdir().unwrap();
    let gv = valid_reset_fixture(temp.path()).unwrap();
    let vectors = temp.path().join("vectors.irvals");
    // Row zero resets the pipeline. The transaction launched in row one is
    // flushed by row two. After a bubble, rows four and five produce 0 and 1.
    fs::write(
        &vectors,
        r#"{vin: bits[1]:1, rst_n: bits[1]:0, x: bits[1]:0}
{x: bits[1]:1, vin: bits[1]:1, rst_n: bits[1]:1}
{rst_n: bits[1]:0, x: bits[1]:1, vin: bits[1]:1}
{vin: bits[1]:0, x: bits[1]:1, rst_n: bits[1]:1}
{rst_n: bits[1]:1, vin: bits[1]:1, x: bits[1]:0}
{vin: bits[1]:1, x: bits[1]:1, rst_n: bits[1]:1}
"#,
    )
    .unwrap();
    let args = ["--input-irvals", "vectors.irvals"];
    let result = check_vectors(temp.path(), &args).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"Concrete check passed for pipe (input rows 6, data comparisons 2, output-valid comparisons 7).\n"
    );
    let netlist = temp.path().join("pipe.gv");
    fs::write(&netlist, gv.replace(".D(data0)", ".D(x)")).unwrap();
    let bypass = check_vectors(temp.path(), &args).unwrap();
    assert_eq!(bypass.status.code(), Some(1));
    assert!(bypass.stdout.is_empty());
    assert_eq!(
        String::from_utf8(bypass.stderr).unwrap(),
        r#"xlsynth-driver: stitch-check-vectors: mismatch at cycle 6 on 'out': expected bits[1]:0, got bits[1]:1
Source row 4 inputs {"rst_n": bits[1]:1, "vin": bits[1]:1, "x": bits[1]:0}
Observation inputs {"rst_n": bits[1]:1, "vin": bits[1]:0, "x": bits[1]:0}
"#
    );

    // A reset in row zero defines output valid immediately, before any
    // source transaction could emerge from this two-edge pipeline.
    fs::write(&netlist, gv.replace(".D(d1)", ".D(vin)")).unwrap();
    let early = check_vectors(temp.path(), &args).unwrap();
    assert_eq!(early.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(early.stderr).unwrap(),
        r#"xlsynth-driver: stitch-check-vectors: mismatch at cycle 1 on 'vout': expected bits[1]:0, got bits[1]:1
Before pipeline fill
Observation inputs {"rst_n": bits[1]:1, "vin": bits[1]:1, "x": bits[1]:1}
"#
    );

    // The output of a single valid row is checked on the final drain cycle.
    fs::write(&netlist, gv).unwrap();
    fs::write(
        &vectors,
        "{x: bits[1]:1, rst_n: bits[1]:1, vin: bits[1]:1}\n",
    )
    .unwrap();
    let single = check_vectors(temp.path(), &args).unwrap();
    assert!(
        single.status.success(),
        "{}",
        String::from_utf8_lossy(&single.stderr)
    );
    assert_eq!(
        single.stdout,
        b"Concrete check passed for pipe (input rows 1, data comparisons 1, output-valid comparisons 1).\n"
    );
    fs::write(&vectors, "").unwrap();
    let empty = check_vectors(temp.path(), &args).unwrap();
    assert_eq!(empty.status.code(), Some(1));
    assert!(empty.stdout.is_empty());
    assert_eq!(
        empty.stderr,
        b"xlsynth-driver: stitch-check-vectors: no input rows\n"
    );

    // With no output-valid port, a stream of bubbles gives the checker no
    // defined external data to compare.
    let spec = temp.path().join("pipe.toml");
    let input_valid_only = fs::read_to_string(&spec)
        .unwrap()
        .replace("output_valid_signal = \"vout\"\n", "");
    fs::write(&spec, input_valid_only).unwrap();
    fs::write(
        &netlist,
        gv.replace("output vout, ", "")
            .replace("wire data0", "wire vout, data0"),
    )
    .unwrap();
    fs::write(
        &vectors,
        "{x: bits[1]:1, rst_n: bits[1]:1, vin: bits[1]:0}\n",
    )
    .unwrap();
    let bubbles = check_vectors(temp.path(), &args).unwrap();
    assert_eq!(bubbles.status.code(), Some(1));
    assert_eq!(bubbles.stderr, b"xlsynth-driver: stitch-check-vectors: the input stream produced no defined output observations\n");
}

#[test]
fn stitch_vectors_evaluate_current_dslx_and_bind_ports_by_name() {
    let temp = tempfile::tempdir().unwrap();
    let gv =
        "module pipe(input [1:0] b, output [1:0] out, input [1:0] a); assign out = a; endmodule";
    stage_fixture(temp.path(), "fn pipe_cycle0(a: u2, b: u2) -> u2 { a }", gv).unwrap();
    fs::write(
        temp.path().join("vectors.irvals"),
        "{b: bits[2]:1, a: bits[2]:2}\n",
    )
    .unwrap();
    let args = ["--input-irvals", "vectors.irvals"];
    let result = check_vectors(temp.path(), &args).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        result.stdout,
        b"Concrete check passed for pipe (input rows 1, data comparisons 1, output-valid comparisons 0).\n"
    );
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(a: u2, b: u2) -> u2 { b }",
    )
    .unwrap();
    let stale = check_vectors(temp.path(), &args).unwrap();
    assert_eq!(stale.status.code(), Some(1));
    assert!(stale.stdout.is_empty());
    assert_eq!(
        String::from_utf8(stale.stderr).unwrap(),
        r#"xlsynth-driver: stitch-check-vectors: mismatch at cycle 0 on 'out': expected bits[2]:1, got bits[2]:2
Source row 0 inputs {"a": bits[2]:2, "b": bits[2]:1}
Observation inputs {"a": bits[2]:2, "b": bits[2]:1}
"#
    );
}

#[test]
fn stitch_random_vectors_reproduce_a_failure_with_the_same_seed() {
    let temp = tempfile::tempdir().unwrap();
    let gv = "module pipe(input [128:0] x, output [128:0] out); assign out = x; endmodule";
    stage_fixture(
        temp.path(),
        "fn pipe_cycle0(x: uN[129]) -> uN[129] { x }",
        gv,
    )
    .unwrap();
    let args = ["--random-samples", "4", "--seed", "42"];
    let matched = check_vectors(temp.path(), &args).unwrap();
    assert!(
        matched.status.success(),
        "{}",
        String::from_utf8_lossy(&matched.stderr)
    );
    assert_eq!(
        matched.stdout,
        b"Concrete check passed for pipe (input rows 4, data comparisons 4, output-valid comparisons 0). Random seed 42.\n"
    );
    // Every random input is a failing vector. The report includes the full
    // arbitrary-width input and observation, so matching reports replay it.
    fs::write(
        temp.path().join("pipe.x"),
        "fn pipe_cycle0(x: uN[129]) -> uN[129] { !x }",
    )
    .unwrap();
    let first = check_vectors(temp.path(), &args).unwrap();
    let second = check_vectors(temp.path(), &args).unwrap();
    assert_eq!(first.status.code(), Some(1));
    assert_eq!(second.status.code(), Some(1));
    assert_eq!(first.stderr, second.stderr);
    assert_eq!(
        String::from_utf8(first.stderr).unwrap().lines().last(),
        Some("Replay with --random-samples 4 --seed 42")
    );
}
