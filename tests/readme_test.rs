// SPDX-License-Identifier: Apache-2.0

extern crate docmatic;

#[test]
fn test_readme() {
    let readme_path = "README.md";
    docmatic::assert_file(readme_path);
}