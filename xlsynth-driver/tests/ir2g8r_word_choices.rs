// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use xlsynth_g8r::aig::AigNode;
use xlsynth_g8r::aig::table::build_area_table_report;
use xlsynth_g8r::aig_serdes::load_aiger_binary::load_aiger_binary;
use xlsynth_g8r::aig_sim::gate_sim::{Collect, PreparedGateSim};
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::mcmc_logic::oracle_equiv_sat;
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_with_word_choices,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_parser::Parser;

const PAIR: &str = include_str!("testdata/pair_select.ir");

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

#[test]
fn word_choices_price_independent_bits_of_padded_selects() {
    // In each word bit 8 is zero and bits 6 and 7 share a nonconstant gate.
    let ir = r#"package padded

top fn f(a: bits[7] id=1, b: bits[7] id=2, x: bits[1] id=3, y: bits[1] id=4, t: bits[1] id=5) -> (bits[9], bits[9]) {
  sa: bits[8] = sign_ext(a, new_bit_count=8, id=6)
  sb: bits[8] = sign_ext(b, new_bit_count=8, id=7)
  aa: bits[9] = zero_ext(sa, new_bit_count=9, id=8)
  bb: bits[9] = zero_ext(sb, new_bit_count=9, id=9)
  fallback: bits[9] = literal(value=210, id=10)
  xy: bits[9] = sel(x, cases=[bb, aa], id=11)
  bypass: bits[9] = sel(y, cases=[xy, aa], id=12)
  left: bits[9] = sel(t, cases=[bypass, fallback], id=13)
  controls: bits[3] = concat(x, y, t, id=14)
  right: bits[9] = priority_sel(controls, cases=[fallback, aa, bb], default=aa, id=15)
  ret outputs: (bits[9], bits[9]) = tuple(left, right, id=16)
}"#;
    let options = CanonicalG8rOptions {
        fraig: false,
        reassociation: false,
        cut_db_rewrite: false,
        enable_formal_array_alias_analysis: false,
        ..Default::default()
    };
    let (incumbent, choices) =
        canonical_ir_text_to_g8r_with_word_choices(ir, None, &options).unwrap();
    let stock = &incumbent.gate_fn;
    let roots: Vec<_> = stock
        .outputs
        .iter()
        .flat_map(|o| o.bit_vector.iter_lsb_to_msb())
        .collect();
    assert_eq!(roots.len(), 18);
    for word in roots.chunks_exact(9) {
        assert_eq!(word[6], word[7]);
        assert!(matches!(stock.gates[word[6].node.id], AigNode::And2 { .. }));
        assert!(matches!(
            stock.gates[word[8].node.id],
            AigNode::Literal { .. }
        ));
    }
    assert_eq!(choices.status, "offered");
    assert!(!choices.candidates.is_empty());
    for offer in choices.candidates {
        assert!(!offer.output_bits.is_empty());
        assert!(offer.output_bits.iter().all(|bit| bit % 9 < 6));
        assert!(oracle_equiv_sat(stock, &offer.graph).unwrap());
    }
}

#[test]
fn word_choice_area_reports_attribute_every_live_and() {
    let options = CanonicalG8rOptions {
        track_pir_node_ids: true,
        fraig: false,
        reassociation: false,
        cut_db_rewrite: false,
        enable_formal_array_alias_analysis: false,
        ..Default::default()
    };
    let (_, choices) = canonical_ir_text_to_g8r_with_word_choices(PAIR, None, &options).unwrap();
    let package = Parser::new(PAIR).parse_and_validate_package().unwrap();
    assert!(!choices.candidates.is_empty());
    for choice in choices.candidates {
        let report = build_area_table_report(&choice.graph, package.get_top_fn().unwrap()).unwrap();
        assert!(report.total_aig_node_count > 0);
        assert_eq!(report.unattributed, None);
    }
}

#[test]
fn word_choices_preserve_returned_select_bit_tags() {
    let options = CanonicalG8rOptions {
        fraig: false,
        reassociation: false,
        cut_db_rewrite: false,
        enable_formal_array_alias_analysis: false,
        ..Default::default()
    };
    let (stock, choices) =
        canonical_ir_text_to_g8r_with_word_choices(PAIR, None, &options).unwrap();
    let mut stock_sim = PreparedGateSim::new(&stock.gate_fn);
    assert!(!choices.candidates.is_empty());
    for choice in choices.candidates {
        let mut sim = PreparedGateSim::new(&choice.graph);
        for controls in 0..8 {
            let inputs = [
                IrBits::make_ubits(7, 0b1010101).unwrap(),
                IrBits::make_ubits(7, 0b0101010).unwrap(),
                IrBits::make_ubits(1, controls & 1).unwrap(),
                IrBits::make_ubits(1, controls >> 1 & 1).unwrap(),
                IrBits::make_ubits(1, controls >> 2).unwrap(),
            ];
            let expected = stock_sim
                .eval(&inputs, Collect::Tagged)
                .tagged_values
                .unwrap();
            let actual = sim.eval(&inputs, Collect::Tagged).tagged_values.unwrap();
            for prefix in ["sel_9", "priority_sel_11"] {
                for bit in 0..7 {
                    let tag = format!("{prefix}_output_bit_{bit}");
                    assert_eq!(actual.get(&tag), Some(&expected[&tag]), "{tag}");
                }
            }
        }
    }
}
