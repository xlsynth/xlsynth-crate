// SPDX-License-Identifier: Apache-2.0

mod assert_valid_sv;
mod simulate_sv;

pub use assert_valid_sv::{assert_valid_sv, assert_valid_sv_flist, FlistEntry};
pub use simulate_sv::{
    simulate_pipeline_single_pulse, simulate_pipeline_single_pulse_custom, simulate_sv_flist,
};

pub mod ir_fuzz;

/// Compare arbitrary text against a golden file on disk, with an opt-in
/// update mechanism controlled by the XLSYNTH_UPDATE_GOLDEN environment
/// variable. Uses full-string equality (no trimming) for exactness.
pub fn compare_golden_text(got: &str, relpath: &str) {
    let golden_path = std::path::Path::new(relpath);
    if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok()
        || !golden_path.exists()
        || golden_path.metadata().map(|m| m.len()).unwrap_or(0) == 0
    {
        log::info!(
            "compare_golden_text; writing golden file to {}",
            golden_path.display()
        );
        std::fs::write(golden_path, got).expect("write golden");
    } else {
        log::info!(
            "compare_golden_text; reading golden file from {}",
            golden_path.display()
        );
        let want = std::fs::read_to_string(golden_path).expect("read golden");
        assert_eq!(
            got, want,
            "Golden mismatch; run with XLSYNTH_UPDATE_GOLDEN=1 to update."
        );
    }
}

pub fn compare_golden_sv(got: &str, relpath: &str) {
    let golden_path = std::path::Path::new(relpath);
    if std::env::var("XLSYNTH_UPDATE_GOLDEN").is_ok()
        || !golden_path.exists()
        || golden_path.metadata().map(|m| m.len()).unwrap_or(0) == 0
    {
        log::info!(
            "compare_golden_sv; writing golden file to {}",
            golden_path.display()
        );
        std::fs::write(golden_path, got).expect("write golden");
    } else {
        log::info!(
            "compare_golden_sv; reading golden file from {}",
            golden_path.display()
        );
        let want = std::fs::read_to_string(golden_path).expect("read golden");
        assert_eq!(
            got.trim(),
            want.trim(),
            "Golden mismatch; run with XLSYNTH_UPDATE_GOLDEN=1 to update."
        );
    }

    // Validate generated Verilog is syntactically correct after golden check.
    assert_valid_sv(&got);
}

/// Creates a unique temporary directory for tests under the system temp dir,
/// using the provided base prefix combined with the process id and a nanosecond
/// timestamp.
///
/// The directory is cleaned up automatically when the returned `TempDir` is
/// dropped.
pub fn make_test_tmpdir(base_prefix: &str) -> tempfile::TempDir {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let prefix = format!("{}_{}_{}", base_prefix, pid, nanos);
    tempfile::Builder::new()
        .prefix(&prefix)
        .tempdir_in(std::env::temp_dir())
        .expect("tempdir create")
}
