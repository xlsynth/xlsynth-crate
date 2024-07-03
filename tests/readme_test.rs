// SPDX-License-Identifier: Apache-2.0

extern crate docmatic;

#[test]
fn test_readme() {
    let readme_path = "README.md";
    docmatic::Assert::default()
        .library_path(env!("XLS_DSO_PATH"))
        .test_file(readme_path);
}
