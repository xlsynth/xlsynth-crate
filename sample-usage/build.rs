// SPDX-License-Identifier: Apache-2.0

/// A very naive fence extractor
fn extract_rust_fence(content: &str) -> Option<String> {
    let fence_start = "```rust";
    let fence_end = "```";

    let start_idx = content.find(fence_start)?;
    // skip the fence_start itself
    let after_start = &content[start_idx + fence_start.len()..];

    // skip any newline after "```rust"
    let after_newline = after_start.trim_start_matches(|c| c == '\r' || c == '\n');

    let end_idx = after_newline.find(fence_end)?;
    Some(after_newline[..end_idx].to_string())
}

fn make_readme_snippet_file() {
    // So Cargo rebuilds if README changes
    println!("cargo:rerun-if-changed=README.md");

    // Read entire README
    let readme_contents = std::fs::read_to_string("README.md").expect("Failed to read README.md");

    // Extract the snippet between ```rust and ``` lines
    let snippet = extract_rust_fence(&readme_contents)
        .expect("Could not find a valid ```rust fenced block in README.md");

    // We'll write a new .rs file into OUT_DIR that becomes part of our test module.
    // We'll wrap the snippet in something that makes it testable.
    // For example: we define a `mod readme_snippet { ... }` with #[test] functions.
    //
    // If your snippet has `fn main() {}`, we can call that from the test function,
    // so that `cargo test` effectively runs what your snippet does.
    let test_code = format!(
        r#"
mod readme_snippet {{
    {code}

    #[test]
    fn test_readme_snippet_main() {{
        // If the snippet has a main(), call it:
        main();
    }}
}}
"#,
        code = snippet
    );

    // Now write that out:
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let test_file = std::path::Path::new(&out_dir).join("readme_snippet_test.rs");
    std::fs::write(&test_file, test_code).expect("Failed to write readme_snippet_test.rs");
}

fn main() {
    make_readme_snippet_file();

    let sample_with_enum_def_rs_path =
        xlsynth::x_path_to_rs_bridge_via_env("src/sample_with_enum_def.x");
    println!(
        "cargo:rustc-env=DSLX_SAMPLE_WITH_ENUM_DEF_PATH={}",
        sample_with_enum_def_rs_path.display()
    );

    // TODO(cdleary): 2024-12-02: Broken for the moment due to the type that has an
    // extern-type member. let rs_path =
    // xlsynth::x_path_to_rs_bridge_via_env("../xlsynth/tests/structure_zoo.x");
    // println!("cargo:rustc-env=DSLX_STRUCTURE_ZOO={}", rs_path.display());

    let rs_path = xlsynth::x_path_to_rs_bridge_via_env("src/sample_with_struct_def.x");
    println!(
        "cargo:rustc-env=DSLX_SAMPLE_WITH_STRUCT_DEF={}",
        rs_path.display()
    );

    println!("cargo:rerun-if-changed=build.rs");
}
