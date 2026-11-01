// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use xlsynth_g8r::aig_serdes::load_aiger_binary::load_aiger_binary;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_pir::IrBits;

const PAIR: &str = r#"package pair_select

top fn f(a: bits[7] id=1, b: bits[7] id=2, x: bits[1] id=3, y: bits[1] id=4, t: bits[1] id=5) -> (bits[7], bits[7]) {
  fallback: bits[7] = literal(value=82, id=6)
  xy: bits[7] = sel(x, cases=[b, a], id=7)
  bypass: bits[7] = sel(y, cases=[xy, a], id=8)
  left: bits[7] = sel(t, cases=[bypass, fallback], id=9)
  controls: bits[3] = concat(x, y, t, id=10)
  right: bits[7] = priority_sel(controls, cases=[fallback, a, b], default=a, id=11)
  ret outputs: (bits[7], bits[7]) = tuple(left, right, id=12)
}"#;

fn lower(dir: &Path, choices: Option<&str>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
    cmd.current_dir(dir).args([
        "ir2g8r",
        "input.ir",
        "--fold=true",
        "--hash=true",
        "--fraig=false",
        "--enable-formal-array-alias-analysis=false",
        "--enable-rewrite-carry-out=false",
        "--enable-rewrite-prio-encode=false",
        "--aiger-out",
        "incumbent.aig",
        "--stats-out",
        "stats.json",
    ]);
    if let Some(name) = choices {
        cmd.args(["--word-choice-dir", name]);
    }
    cmd.output().unwrap()
}

#[test]
fn word_choices_keep_incumbent_and_export_equivalent_named_outputs() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fs::write(dir.join("input.ir"), PAIR).unwrap();
    let ordinary = lower(dir, None);
    assert!(
        ordinary.status.success(),
        "{}",
        String::from_utf8_lossy(&ordinary.stderr)
    );
    let incumbent = fs::read(dir.join("incumbent.aig")).unwrap();
    let stats = fs::read(dir.join("stats.json")).unwrap();
    assert!(!dir.join("choices").exists());
    let with = lower(dir, Some("choices"));
    assert!(
        with.status.success(),
        "{}",
        String::from_utf8_lossy(&with.stderr)
    );
    assert_eq!(with.stdout, ordinary.stdout);
    assert_eq!(fs::read(dir.join("incumbent.aig")).unwrap(), incumbent);
    assert_eq!(fs::read(dir.join("stats.json")).unwrap(), stats);

    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("choices/manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["status"], "offered");
    assert_eq!(manifest["stage"], "native-terminal");
    let rows = manifest["candidates"].as_array().unwrap();
    assert!(!rows.is_empty());
    let stock = load_aiger_binary(&incumbent, GateBuilderOptions::no_opt())
        .unwrap()
        .gate_fn;
    let mut stock_sim = PreparedGateSim::new(&stock);
    let mut distinct = false;
    for row in rows {
        assert!(!row["output_bits"].as_array().unwrap().is_empty());
        let name = row["file"].as_str().unwrap();
        assert_eq!(name, format!("{}.aig", row["key"].as_str().unwrap()));
        let bytes = fs::read(dir.join("choices").join(name)).unwrap();
        distinct |= bytes != incumbent;
        let graph = load_aiger_binary(&bytes, GateBuilderOptions::no_opt())
            .unwrap()
            .gate_fn;
        assert_eq!(graph.get_signature(), stock.get_signature());
        let mut sim = PreparedGateSim::new(&graph);
        for (a, b) in [(0, 127), (127, 0), (82, 45), (35, 89)] {
            for controls in 0..8 {
                let bits: u32 = a | (b << 7) | (controls << 14);
                let inputs: Vec<_> = (0..17)
                    .map(|i| IrBits::make_ubits(1, u64::from(bits >> i & 1)).unwrap())
                    .collect();
                assert_eq!(sim.eval_outputs(&inputs), stock_sim.eval_outputs(&inputs));
            }
        }
    }
    assert!(distinct);
    let before = fs::read(dir.join("choices/manifest.json")).unwrap();
    let repeat = lower(dir, Some("choices"));
    assert!(!repeat.status.success());
    assert_eq!(fs::read(dir.join("choices/manifest.json")).unwrap(), before);
}

#[test]
fn word_choices_preserve_no_offer_and_reject_an_existing_empty_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fs::write(dir.join("input.ir"), "package p\n\ntop fn f(x: bits[1] id=1) -> bits[1] {\n  ret value: bits[1] = not(x, id=2)\n}").unwrap();
    let ok = lower(dir, Some("empty-choices"));
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let m: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("empty-choices/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(m["status"], "no-offers");
    assert_eq!(m["candidates"], serde_json::json!([]));
    let occupied = dir.join("occupied");
    fs::create_dir(&occupied).unwrap();
    assert!(!lower(dir, Some("occupied")).status.success());
    assert_eq!(fs::read_dir(occupied).unwrap().count(), 0);
}
