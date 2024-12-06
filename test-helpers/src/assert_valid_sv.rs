// SPDX-License-Identifier: Apache-2.0

use tempfile::NamedTempFile;

fn can_use_slang() -> bool {
    let maybe_slang_path = std::env::var("SLANG_PATH").unwrap_or_default();
    !maybe_slang_path.is_empty()
}

/// Helper that asserts that the given SV string is valid.
pub fn assert_valid_sv(sv: &str) {
    if !can_use_slang() {
        log::warn!("Skipping SV validation because Slang is not available.");
        return;
    }

    // Write out a named temporary file with `sv` as the contents.
    let temp_file = NamedTempFile::new().unwrap();
    std::fs::write(temp_file.path(), sv).unwrap();

    // Now parse those contents we wrote out with slang.
    let cfg = slang_rs::SlangConfig {
        sources: &[temp_file.path().to_str().unwrap()],
        ..Default::default()
    };
    slang_rs::run_slang(&cfg).expect("expect single SystemVerilog file contents is valid");
}

pub struct FlistEntry {
    pub filename: String,
    pub contents: String,
}

pub fn assert_valid_sv_flist(files: &[FlistEntry]) {
    if !can_use_slang() {
        log::warn!("Skipping SV validation because Slang is not available.");
        return;
    }

    // Make a temporary directory and write out all the files to it.
    let temp_dir = tempfile::tempdir().unwrap();

    // Create all the temporary files based on the file list entries given.
    let mut sources = vec![];
    for entry in files {
        let path = temp_dir.path().join(&entry.filename);
        sources.push(path.to_str().unwrap().to_string());
        std::fs::write(path, &entry.contents).unwrap();
    }

    // Now run Slang on those files.
    let sources_strs = sources.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    let cfg = slang_rs::SlangConfig {
        sources: &sources_strs,
        ..Default::default()
    };
    let result = slang_rs::run_slang(&cfg);
    if result.is_err() {
        panic!(
            "expect we can parse valid SystemVerilog via whole file list; error:\n{}",
            result.err().unwrap().to_string().replace("\\n", "\n")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observe_error_if_can_use_slang() {
        let _ = env_logger::builder().is_test(true).try_init();
        if !can_use_slang() {
            log::warn!("Skipping SV validation because Slang is not available.");
            return;
        }
        let invalid_sv = "module foo; garbage; endmodule";
        // This call should panic on the invalid SV.
        let result = std::panic::catch_unwind(|| assert_valid_sv(&invalid_sv));
        assert!(result.is_err());
        let error = result.err().unwrap();
        let error_message = error.downcast_ref::<String>().unwrap();
        assert!(error_message.contains("expected a declaration name"));
    }
}
