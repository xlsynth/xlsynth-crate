// SPDX-License-Identifier: Apache-2.0

use xlsynth_spdx_test::{assert_workspace_spdx_clean, find_missing_spdx_files, SpdxTestConfig};

#[test]
fn test_finds_rust_file_missing_spdx_in_tempdir() {
    let temp_dir = tempfile::tempdir().unwrap();
    let temp_dir_path = temp_dir.path();

    // Minimal config to use C++-style comments by default.
    let cfg = SpdxTestConfig {
        license: "Apache-2.0".to_string(),
        hash_comment_extensions: vec!["py".to_string(), "yml".to_string(), "yaml".to_string()],
        hash_comment_filenames: vec!["requirements.txt".to_string()],
        expect_shebang_extensions: vec!["py".to_string()],
        exclude_dir_names: vec![],
        exclude_path_suffixes: vec![],
        exclude_exact_filenames: vec![],
        exclude_extensions: vec![],
        exclude_prefix_with_extension: vec![],
    };

    let has_spdx_file = temp_dir_path.join("has_spdx.rs");
    std::fs::write(&has_spdx_file, "// SPDX-License-Identifier: Apache-2.0\n").unwrap();

    let missing_spdx_file = temp_dir_path.join("missing_spdx.rs");
    std::fs::write(&missing_spdx_file, "").unwrap();

    let missing_spdx_files = find_missing_spdx_files(temp_dir_path, &cfg);
    assert_eq!(missing_spdx_files.len(), 1);
    assert!(missing_spdx_files.contains(&missing_spdx_file));
}

#[test]
fn check_all_files_for_spdx() {
    // Uses repo root config.
    assert_workspace_spdx_clean();
}
