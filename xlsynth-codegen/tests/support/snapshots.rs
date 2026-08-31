// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

const LICENSE: &str = "// SPDX-License-Identifier: Apache-2.0\n\n";

/// Updates or compares a standalone SystemVerilog golden, preserving its
/// license.
pub(super) fn assert_golden_sv(source: &str, relative_path: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative_path);
    let expected = format!("{LICENSE}{source}");
    if std::env::var_os("XLSYNTH_UPDATE_GOLDEN").is_some() {
        std::fs::write(&path, expected).expect("update SystemVerilog golden");
    } else {
        let contents = std::fs::read_to_string(&path).expect("read SystemVerilog golden");
        pretty_assertions::assert_eq!(
            expected,
            contents,
            "golden mismatch for {}; run with XLSYNTH_UPDATE_GOLDEN=1 to update",
            path.display()
        );
    }
}
