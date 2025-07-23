// SPDX-License-Identifier: Apache-2.0

mod assert_valid_sv;
mod simulate_sv;

pub use assert_valid_sv::{assert_valid_sv, assert_valid_sv_flist, FlistEntry};
pub use simulate_sv::{
    simulate_pipeline_single_pulse, simulate_pipeline_single_pulse_custom, simulate_sv_flist,
};

pub mod ir_fuzz;

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
