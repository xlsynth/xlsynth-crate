// SPDX-License-Identifier: Apache-2.0

//! Tests that the README.md file has a properly executable and up to date
//! sample usage displayed in it.

#[cfg(test)]
mod readme_integration_test {
    // This "includes" the file we generated in build.rs
    include!(concat!(env!("OUT_DIR"), "/readme_snippet_test.rs"));
}
