// SPDX-License-Identifier: Apache-2.0

extern crate docmatic;

#[test]
fn test_readme() {
    docmatic::Assert::default()
        .library_path(xlsynth_sys::XLS_DSO_PATH)
        .test_file(format!("../README.md"));
}
