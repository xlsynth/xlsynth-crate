// SPDX-License-Identifier: Apache-2.0

use std::sync::{Mutex, Once};
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::gate_fn_optimize::{GateFnOptimizeOptions, optimize_gate_fn};
use xlsynth_g8r::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils;

const GATE_TARGET: &str = "xlsynth_g8r::gate_fn_optimize";
const PREP_TARGET: &str = "xlsynth_g8r::gatify::prep_for_gatify";

/// Captures records submitted to the logger, including any graph built before
/// the logger gets a chance to filter it out.
struct SnapshotLog(Mutex<Vec<String>>);

impl log::Log for SnapshotLog {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        (metadata.target() == PREP_TARGET && metadata.level() == log::Level::Trace)
            || (metadata.target() == GATE_TARGET && metadata.level() <= log::Level::Debug)
    }

    fn log(&self, record: &log::Record) {
        if matches!(record.target(), GATE_TARGET | PREP_TARGET) {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

static LOG: SnapshotLog = SnapshotLog(Mutex::new(Vec::new()));

fn init_logging() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        log::set_logger(&LOG).unwrap();
        log::set_max_level(log::LevelFilter::Trace);
    });
}

#[test]
fn gate_debug_reports_a_high_id_literal_without_submitting_the_trace_dump() {
    init_logging();
    let gates = GateFn::try_from(
        r#"fn f(x: bits[1]=[%1]) -> (o: bits[1]=[%2]) {
  %2 = literal(true)
}"#,
    )
    .unwrap();
    let outcome = optimize_gate_fn(gates, &GateFnOptimizeOptions::all_disabled()).unwrap();
    assert_eq!(get_aig_stats(&outcome.gate_fn).max_depth, 0);
    let records = LOG.0.lock().unwrap();
    assert!(
        records
            .iter()
            .any(|r| r.starts_with("gate_pass function=f stage=mapped "))
    );
    assert!(!records.iter().any(|r| r.starts_with("gate_ir ")));
}

#[test]
fn trace_without_debug_replays_shifted_nonzero_before_dead_code_removal() {
    init_logging();
    let pkg = Parser::new(include_str!("data/shift_window_threshold.ir"))
        .parse_and_validate_package()
        .unwrap();
    prep_for_gatify(
        pkg.get_top_fn().unwrap(),
        None,
        PrepForGatifyOptions::default(),
    );

    let records = LOG.0.lock().unwrap();
    assert!(!records.iter().any(|r| r.starts_with("prep_pass ")));
    let body = records
        .iter()
        .find_map(|r| r.strip_prefix("prep_ir function=f pass=shifted_nonzero round=0\n"))
        .expect("shifted_nonzero trace");
    let replay = Parser::new(body).parse_function_declaration().unwrap();
    let root = replay.get_node(replay.ret_node_ref.unwrap());
    // The rewritten result uses new helpers, and the original dead shift is
    // still in storage at this checkpoint.
    assert!(
        ir_utils::operands(&root.payload)
            .iter()
            .any(|r| replay.get_node(*r).text_id > root.text_id)
    );
    assert_eq!(ir_utils::op_histogram(&replay)["shll"], 1);
}
