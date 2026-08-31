// SPDX-License-Identifier: Apache-2.0

use std::process::Command;
use std::time::{Duration, Instant};
use xlsynth::IrBits;
use xlsynth_test_helpers::rtl_sim::{LogicValue, run_checked};

#[test]
fn four_state_transport_preserves_unknowns_and_arbitrary_widths() {
    let value = LogicValue::parse_binary("10xZ", 4).unwrap();
    assert_eq!(value.to_bit_string_msb_first(), "10xz");
    assert!(value.to_bits().is_err());
    assert_eq!(value.to_u64_if_known(), None);
    assert!(LogicValue::parse_binary("1", 2).is_err());
    assert!(LogicValue::parse_binary("?", 1).is_err());
    let bits = IrBits::from_lsb_is_0(&(0..257).map(|i| i % 3 == 0).collect::<Vec<_>>());
    let value = LogicValue::from_bits(&bits);
    assert_eq!(value.to_bits().unwrap(), bits);
    assert_eq!(value.to_u64_if_known(), None);
    assert_eq!(LogicValue::from_u64(0, 0).to_u64_if_known(), Some(0));
}

#[cfg(unix)]
#[test]
fn external_tool_timeout_terminates_child_process_group() {
    let directory = tempfile::tempdir().unwrap();
    let start = Instant::now();
    let error = run_checked(
        Command::new("sh").args(["-c", "sleep 30 & wait"]),
        directory.path(),
        "timeout-test",
        Duration::from_millis(50),
    )
    .unwrap_err();
    assert!(error.contains("exceeded"), "{error}");
    assert!(start.elapsed() < Duration::from_secs(5));
}
