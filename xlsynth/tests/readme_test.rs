// SPDX-License-Identifier: Apache-2.0

extern crate docmatic;

// On OS X we sometimes see multiple artifacts in the target directory, so we
// restrict to Linux only for now.
#[cfg(target_os = "linux")]
#[test]
fn test_readme() {
    docmatic::Assert::default()
        .library_path(xlsynth_sys::XLS_DSO_PATH)
        .test_file(format!("../README.md"));
}
