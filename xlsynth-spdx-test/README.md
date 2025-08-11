SPDX-License-Identifier: Apache-2.0

## xlsynth-spdx-test

Config-driven SPDX header checker for workspaces. This crate provides a simple API to assert that all source files in the repo have the expected SPDX header, with exclusions and exceptions specified in a repository-level `spdx_test.toml`.

### Configuration

Place a `spdx_test.toml` at your workspace root. Example:

```toml
license = "Apache-2.0"

hash_comment_extensions = ["yml", "yaml", "py"]
hash_comment_filenames = ["requirements.txt"]
expect_shebang_extensions = ["py"]

exclude_dir_names = [
  "target", ".git", ".venv", "xlsynth_tools", "__pycache__",
  ".pytest_cache", ".mypy_cache", ".ruff_cache", ".vscode",
]
exclude_path_suffixes = [
  ".golden.sv", ".golden.v", ".golden.txt", ".so", ".bin",
]
exclude_exact_filenames = ["estimator_model.proto"]
exclude_extensions = ["md", "lock", "toml", "supp"]

[[exclude_prefix_with_extension]]
prefix = "generated_"
extension = "json"
```

### Usage

Add a test that delegates to the helper:

```rust
#[test]
fn check_all_files_for_spdx() {
    xlsynth_spdx_test::assert_workspace_spdx_clean();
}
```

Alternatively, call the lower-level API to get the missing paths and format your own assertion:

```rust
let cfg = xlsynth_spdx_test::load_config_from_repo_root();
let root = cargo_metadata::MetadataCommand::new().exec().unwrap().workspace_root;
let missing = xlsynth_spdx_test::find_missing_spdx_files(root.as_std_path(), &cfg);
assert!(missing.is_empty(), "missing SPDX: {:?}", missing);
```

### FAQ

- **Will my `cargo test` automatically run this crate's tests when used as a dependency?** No. Cargo does not run tests of dependencies when you run `cargo test` in your own crate. You must add a small test in your workspace (as shown above) that calls `assert_workspace_spdx_clean()`.
- **Which repo root does it use?** The functions call `cargo_metadata` at runtime to determine your workspace root, so when invoked from your tests they will check your repository and read your `spdx_test.toml` at the top of your repo.
- **Where do I put `spdx_test.toml`?** At the root of your Cargo workspace (same directory as the top-level `Cargo.toml`).
