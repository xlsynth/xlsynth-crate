// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, encode_g8r_binary};
use xlsynth_g8r::block2sequential::block_ir_to_sequential_gate_fn_with_controls;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;

#[test]
fn evaluates_serialized_explicit_controls_and_rejects_physical_toggle_accounting() {
    let block = r#"package controls
top block top(clk: clock, rst: bits[1], data: bits[8], en: bits[1], out: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg state(bits[8], reset_value=3)
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[8] = input_port(name=data, id=2)
  en: bits[1] = input_port(name=en, id=3)
  q: bits[8] = register_read(register=state, id=4)
  d: () = register_write(data, register=state, load_enable=en, reset=rst, id=5)
  out: () = output_port(q, name=out, id=6)
}
"#;
    let design =
        block_ir_to_sequential_gate_fn_with_controls(block, GatifyOptions::all_opts_disabled())
            .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let stimulus = dir.path().join("inputs.irvals");
    std::fs::write(
        &stimulus,
        r#"(bits[1]:0, bits[8]:5, bits[1]:1)
(bits[1]:1, bits[8]:9, bits[1]:0)
(bits[1]:0, bits[8]:1, bits[1]:0)
"#,
    )
    .unwrap();
    for (extension, contents) in [
        ("g8r", emit_g8r(&design).unwrap().into_bytes()),
        ("g8rbin", encode_g8r_binary(&design).unwrap()),
    ] {
        let path = dir.path().join(format!("controls.{extension}"));
        std::fs::write(&path, contents).unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        command
            .arg("g8r-eval")
            .arg(&path)
            .arg("--input-irvals")
            .arg(&stimulus)
            .arg("--initial-state-all-zeros");
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "bits[8]:0\nbits[8]:5\nbits[8]:3\n"
        );
        let toggle_path = dir.path().join("toggles.json");
        let rejected = command
            .arg("--toggle-output-json")
            .arg(&toggle_path)
            .output()
            .unwrap();
        assert!(!rejected.status.success());
        assert!(rejected.stdout.is_empty());
        assert!(!toggle_path.exists());
    }
}
