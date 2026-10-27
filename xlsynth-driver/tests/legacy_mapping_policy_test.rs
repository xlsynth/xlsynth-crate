// SPDX-License-Identifier: Apache-2.0

use std::process::Command;
use xlsynth_g8r::liberty::model::LIBERTY_FORMAT_MAGIC_V5;

#[test]
fn mapping_old_proto_reports_how_to_restore_sizing_policy_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let aig = dir.path().join("input.aag");
    let library = dir.path().join("legacy.textproto");
    let netlist = dir.path().join("mapped.v");
    std::fs::write(&aig, "aag 3 2 0 1 1\n2\n4\n6\n6 4 2\ni0 a\ni1 b\no0 y\nc\n").unwrap();
    std::fs::write(
        &library,
        format!(
            r#"
format_magic: {LIBERTY_FORMAT_MAGIC_V5}
cells: {{
  name: "INV" area: 1.0
  pins: {{ name_string_id: 1 direction: INPUT }}
  pins: {{ name_string_id: 2 direction: OUTPUT function_string_id: 3 }}
}}
cells: {{
  name: "NAND2" area: 2.0
  pins: {{ name_string_id: 1 direction: INPUT }}
  pins: {{ name_string_id: 4 direction: INPUT }}
  pins: {{ name_string_id: 2 direction: OUTPUT function_string_id: 5 }}
}}
interned_strings: ["A", "Y", "!A", "B", "!(A*B)"]
"#
        ),
    )
    .unwrap();

    for (strategy, policy, expected_error) in [
        (
            "nand2",
            Some("small-normal-vt"),
            "aig-tech-map error: cell 'INV' has unknown use_for_size_only source coverage; regenerate from source Liberty before mapping\n",
        ),
        (
            "nand2",
            Some("max-speed"),
            "aig-tech-map error: cell 'INV' has unknown use_for_size_only source coverage; regenerate from source Liberty before mapping\n",
        ),
        (
            "area",
            None,
            "aig-tech-map error: area mapping failed: cell 'INV' has unknown use_for_size_only source coverage; regenerate from source Liberty before mapping: no cover for demanded node 3 polarity 0; library may lack a required cell\n",
        ),
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        command
            .arg("aig-tech-map")
            .arg(&aig)
            .args(["--strategy", strategy])
            .arg("--liberty_proto")
            .arg(&library)
            .arg("--netlist_out")
            .arg(&netlist);
        if strategy == "area" {
            // Exercise legacy mapping policy independently of timing
            // requirements.
            command.args(["--fanout-opt=false", "--timing-sizing=false"]);
        }
        if let Some(policy) = policy {
            command.args(["--cell_policy", policy]);
        }
        let output = command.output().unwrap();
        assert!(!output.status.success());
        assert_eq!(String::from_utf8(output.stderr).unwrap(), expected_error);
        assert!(!netlist.exists());
    }
}
