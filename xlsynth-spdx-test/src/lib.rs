// SPDX-License-Identifier: Apache-2.0

use cargo_metadata::MetadataCommand;
use serde::Deserialize;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct PrefixWithExtension {
    pub prefix: String,
    pub extension: String,
}

#[derive(Debug, Deserialize)]
pub struct SpdxTestConfig {
    pub license: String,

    // Determines comment prefix selection.
    pub hash_comment_extensions: Vec<String>,
    pub hash_comment_filenames: Vec<String>,
    pub expect_shebang_extensions: Vec<String>,

    // Exclusions when crawling files.
    pub exclude_dir_names: Vec<String>,
    pub exclude_path_suffixes: Vec<String>,
    pub exclude_exact_filenames: Vec<String>,
    pub exclude_extensions: Vec<String>,
    pub exclude_prefix_with_extension: Vec<PrefixWithExtension>,
}

fn choose_comment_prefix(file_path: &Path, cfg: &SpdxTestConfig) -> (&'static str, bool) {
    let filename = file_path.file_name().unwrap().to_str().unwrap();
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let comment_prefix = if cfg
        .hash_comment_extensions
        .iter()
        .any(|e| e.as_str() == ext)
        || cfg
            .hash_comment_filenames
            .iter()
            .any(|n| n.as_str() == filename)
    {
        "#"
    } else {
        "//"
    };

    let expect_shebang = cfg
        .expect_shebang_extensions
        .iter()
        .any(|e| e.as_str() == ext);

    (comment_prefix, expect_shebang)
}

fn check_spdx_identifier(file_path: &Path, cfg: &SpdxTestConfig) -> bool {
    let (comment_prefix, expect_shebang) = choose_comment_prefix(file_path, cfg);
    let expected_spdx_identifier =
        format!("{comment_prefix} SPDX-License-Identifier: {}", cfg.license);

    let file = match std::fs::File::open(file_path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();
    let ok = if expect_shebang {
        let first_line = match lines.next() {
            Some(Ok(l)) => l,
            _ => String::new(),
        };
        if first_line.starts_with("#!") {
            match lines.next() {
                Some(Ok(l)) => l.starts_with(&expected_spdx_identifier),
                _ => false,
            }
        } else {
            first_line.starts_with(&expected_spdx_identifier)
        }
    } else {
        match lines.next() {
            Some(Ok(line)) => line.starts_with(&expected_spdx_identifier),
            _ => false,
        }
    };
    if ok {
        println!("Found SPDX identifier in file: {:?}", file_path);
    } else {
        eprintln!("Missing SPDX identifier in file: {:?}", file_path);
    }
    ok
}

fn should_exclude_path(path: &Path, cfg: &SpdxTestConfig) -> bool {
    let filename = path.file_name().unwrap().to_str().unwrap();
    if cfg
        .exclude_exact_filenames
        .iter()
        .any(|n| n.as_str() == filename)
    {
        return true;
    }

    let path_str = path.as_os_str().to_str().unwrap();
    if cfg
        .exclude_path_suffixes
        .iter()
        .any(|s| path_str.ends_with(s))
    {
        return true;
    }

    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        if cfg
            .exclude_extensions
            .iter()
            .any(|e| e.as_str() == extension)
        {
            return true;
        }

        if cfg
            .exclude_prefix_with_extension
            .iter()
            .any(|pe| pe.extension.as_str() == extension && filename.starts_with(&pe.prefix))
        {
            return true;
        }
    }

    false
}

fn is_excluded_dir(entry_name: &str, cfg: &SpdxTestConfig) -> bool {
    cfg.exclude_dir_names
        .iter()
        .any(|d| d.as_str() == entry_name)
}

pub fn find_missing_spdx_files(root: &Path, cfg: &SpdxTestConfig) -> Vec<PathBuf> {
    let mut missing_spdx_files = Vec::new();
    let mut dir_worklist: Vec<PathBuf> = vec![root.into()];

    loop {
        let dir = match dir_worklist.pop() {
            Some(dir) => dir,
            None => break,
        };
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.is_dir() {
                let entry_name = entry.file_name();
                let entry_name = entry_name.to_str().unwrap();
                if !is_excluded_dir(entry_name, cfg) {
                    println!("Adding to directory worklist: {:?}", path);
                    dir_worklist.push(path.clone());
                }
                continue;
            }

            if should_exclude_path(&path, cfg) {
                continue;
            }

            if let Some(_extension) = path.extension() {
                if !check_spdx_identifier(&path, cfg) {
                    missing_spdx_files.push(path);
                }
            }
        }
    }
    missing_spdx_files
}

pub fn load_config_from_repo_root() -> SpdxTestConfig {
    let metadata = MetadataCommand::new().exec().unwrap();
    let workspace_dir = metadata.workspace_root;
    let config_path = workspace_dir.as_std_path().join("spdx_test.toml");
    let raw = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("Failed to read config at {}: {}", config_path.display(), e));
    toml::from_str(&raw).unwrap()
}

pub fn find_missing_spdx_files_from_workspace_config() -> Vec<PathBuf> {
    let cfg = load_config_from_repo_root();
    let workspace_dir = MetadataCommand::new().exec().unwrap().workspace_root;
    find_missing_spdx_files(workspace_dir.as_std_path(), &cfg)
}

pub fn assert_workspace_spdx_clean() {
    let missing_spdx_files = find_missing_spdx_files_from_workspace_config();
    if !missing_spdx_files.is_empty() {
        eprintln!(
            "\nSummary of files missing SPDX identifiers ({}):",
            missing_spdx_files.len()
        );
        for path in &missing_spdx_files {
            eprintln!("  - {}", path.display());
        }
        eprintln!("\n");
    }
    assert!(
        missing_spdx_files.is_empty(),
        "The following files are missing SPDX identifiers: {:?}",
        missing_spdx_files
    );
}
