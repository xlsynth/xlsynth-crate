// SPDX-License-Identifier: Apache-2.0

//! Locate DSLX source candidates in Bazel runfiles for the OSS XLS repository.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const XLS_REPOSITORY: &str = "com_google_xls";

/// The portion of a Bazel runfiles tree or manifest that belongs to XLS.
pub(super) struct XlsRunfiles {
    directory: Option<PathBuf>,
    manifest: BTreeMap<PathBuf, PathBuf>,
    repository: String,
}

impl XlsRunfiles {
    pub(super) fn from_env() -> Self {
        Self::from_locations(
            env::var_os("RUNFILES_MANIFEST_FILE").map(PathBuf::from),
            env::var_os("RUNFILES_DIR").map(PathBuf::from),
            env::current_exe().ok(),
        )
    }

    /// Bazel can supply the manifest, its sibling tree, or both. With no
    /// environment variables it locates them alongside the running executable.
    fn from_locations(
        manifest: Option<PathBuf>,
        directory: Option<PathBuf>,
        executable: Option<PathBuf>,
    ) -> Self {
        let mut manifest = manifest.filter(|path| path.is_file());
        let mut directory = directory.filter(|path| path.is_dir());
        if manifest.is_none()
            && let Some(tree) = directory
                .clone()
                .or_else(|| executable.map(|p| append(&p, ".runfiles")))
        {
            manifest = [tree.join("MANIFEST"), append(&tree, "_manifest")]
                .into_iter()
                .find(|path| path.is_file());
            directory = tree.is_dir().then_some(tree);
        }
        if directory.is_none()
            && let Some(path) = manifest.as_deref()
        {
            directory = if path.file_name().is_some_and(|s| s == "MANIFEST") {
                path.parent().map(Path::to_path_buf)
            } else {
                path.to_str()
                    .and_then(|s| s.strip_suffix(".runfiles_manifest"))
                    .map(|prefix| append(Path::new(prefix), ".runfiles"))
            }
            .filter(|path| path.is_dir());
        }

        let mut result = Self {
            directory,
            manifest: BTreeMap::new(),
            repository: XLS_REPOSITORY.into(),
        };
        if let Some(bytes) = manifest.and_then(|path| fs::read(path).ok()) {
            // DSLX requires UTF-8 import paths. Unrelated runfile entries can
            // still contain arbitrary OS path bytes.
            for line in bytes
                .split(|&byte| byte == b'\n')
                .take_while(|line| !line.is_empty())
                .filter_map(|line| std::str::from_utf8(line).ok())
            {
                let line = line.strip_suffix('\r').unwrap_or(line);
                let entry = if let Some(escaped) = line.strip_prefix(' ') {
                    escaped
                        .split_once(' ')
                        .map(|(key, value)| (unescape(key), unescape(value)))
                } else {
                    line.split_once(' ')
                        .map(|(key, value)| (key.into(), value.into()))
                };
                if let Some((key, value)) = entry {
                    result
                        .manifest
                        .insert(PathBuf::from(key), PathBuf::from(value));
                }
            }
        }
        // XLS calls Rlocation from the main repository (empty source name).
        if let Some(mapping) = result
            .resolve(Path::new("_repo_mapping"))
            .and_then(|path| fs::read_to_string(path).ok())
        {
            let mut exact = None;
            let mut wildcard = None;
            for line in mapping.lines() {
                let mut columns = line.splitn(3, ',');
                if let (Some(source), Some(XLS_REPOSITORY), Some(target)) =
                    (columns.next(), columns.next(), columns.next())
                {
                    match source {
                        "" => exact = Some(target),
                        "*" => wildcard = Some(target),
                        _ => { /* Mapping for a different source repository. */ }
                    }
                }
            }
            if let Some(name) = exact.or(wildcard) {
                result.repository = name.to_owned();
            }
        }
        result
            .manifest
            .retain(|key, _| key.starts_with(&result.repository));
        result
    }

    /// Resolve the longest manifest prefix, falling back to the runfiles tree.
    fn resolve(&self, key: &Path) -> Option<PathBuf> {
        for prefix in key.ancestors() {
            if let Some(target) = self.manifest.get(prefix) {
                let suffix = key.strip_prefix(prefix).ok()?;
                return Some(if suffix.as_os_str().is_empty() {
                    target.clone()
                } else {
                    target.join(suffix)
                });
            }
        }
        self.directory.as_ref().map(|dir| dir.join(key))
    }

    pub(super) fn lookup(&self, suffix: &Path) -> Option<PathBuf> {
        let path = if suffix.is_absolute() {
            suffix.to_owned()
        } else {
            self.resolve(&Path::new(&self.repository).join(suffix))?
        };
        path.is_file().then_some(path)
    }
}

fn append(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(suffix);
    value.into()
}

/// Decode Bazel's space, newline and backslash escapes in a manifest record.
fn unescape(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('s') => result.push(' '),
                Some('n') => result.push('\n'),
                Some('b') => result.push('\\'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn discovers_the_runfiles_tree_beside_the_executable() -> std::io::Result<()> {
        let temp = tempdir()?;
        let tree = temp.path().join("driver.runfiles").join(XLS_REPOSITORY);
        fs::create_dir_all(&tree)?;
        let module = tree.join("types.x");
        fs::write(&module, "")?;
        let runfiles = XlsRunfiles::from_locations(None, None, Some(temp.path().join("driver")));
        assert_eq!(runfiles.lookup(Path::new("types.x")), Some(module));
        assert_eq!(runfiles.lookup(Path::new("missing.x")), None);
        Ok(())
    }

    #[test]
    fn remapped_repository_uses_exact_and_directory_manifest_entries() -> std::io::Result<()> {
        let temp = tempdir()?;
        let tree = temp.path().join("driver.runfiles");
        fs::create_dir(&tree)?;
        let prefix = temp.path().join("source directory");
        fs::create_dir(&prefix)?;
        let nested = prefix.join("nested.x");
        let exact = temp.path().join("override.x");
        let mapping = temp.path().join("mapping");
        fs::write(&nested, "")?;
        fs::write(&exact, "")?;
        fs::write(&mapping, ",com_google_xls,_main\n")?;
        fs::write(
            tree.join("MANIFEST"),
            format!(
                "_repo_mapping {}\n _main/pkg {}\n_main/pkg/direct.x {}\n",
                mapping.display(),
                prefix
                    .display()
                    .to_string()
                    .replace('\\', "\\b")
                    .replace(' ', "\\s"),
                exact.display()
            ),
        )?;
        let runfiles = XlsRunfiles::from_locations(None, Some(tree), None);
        assert_eq!(runfiles.lookup(Path::new("pkg/nested.x")), Some(nested));
        assert_eq!(runfiles.lookup(Path::new("pkg/direct.x")), Some(exact));
        Ok(())
    }
}
