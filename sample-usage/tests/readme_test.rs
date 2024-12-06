// SPDX-License-Identifier: Apache-2.0

//! Tests that the README.md file has a properly executable and up to date
//! sample usage displayed in it.

// On OS X we sometimes see multiple artifacts in the target directory, so we
// restrict to Linux only for now.
#[cfg(target_os = "linux")]
#[test]
fn test_readme() {
    let workspace_dir = cargo_metadata::MetadataCommand::new()
        .exec()
        .unwrap()
        .workspace_root;

    docmatic::Assert::default()
        .library_path(xlsynth_sys::XLS_DSO_PATH)
        .test_file(format!("{}/README.md", workspace_dir));
}
