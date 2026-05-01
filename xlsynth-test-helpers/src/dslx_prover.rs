// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use xlsynth_prover::prover::types::{BoolPropertyResult, QuickCheckAssertionSemantics};
use xlsynth_prover::prover::{prover_for_choice, SolverChoice};

/// Returns `XLSYNTH_TOOLS`, or the local checkout convention under `$HOME`.
pub fn default_xlsynth_tools_path() -> PathBuf {
    if let Ok(path) = std::env::var("XLSYNTH_TOOLS") {
        return PathBuf::from(path);
    }
    let home = std::env::var("HOME").expect("HOME should be set");
    let path = PathBuf::from(home).join("opt/xlsynth/latest");
    assert!(
        path.join("prove_quickcheck_main").exists(),
        "XLSYNTH_TOOLS is not set and default tool path does not contain prove_quickcheck_main: {}",
        path.display()
    );
    path
}

/// Proves every DSLX quickcheck in `path` with the selected prover backend.
pub fn assert_dslx_quickchecks_prove(path: &Path, additional_search_paths: &[&Path]) {
    let tool_path = default_xlsynth_tools_path();
    let stdlib_path = tool_path.join("xls/dslx/stdlib");
    let prover = prover_for_choice(SolverChoice::Auto, Some(&tool_path));
    let results = prover.prove_dslx_quickcheck(
        path,
        Some(&stdlib_path),
        additional_search_paths,
        None,
        QuickCheckAssertionSemantics::Never,
        None,
        &HashMap::new(),
    );

    assert!(
        !results.is_empty(),
        "expected quickcheck obligations in {}",
        path.display()
    );
    for result in results {
        assert!(
            matches!(result.result, BoolPropertyResult::Proved),
            "quickcheck {} in {} did not prove: {:?}",
            result.name,
            path.display(),
            result.result
        );
    }
}
