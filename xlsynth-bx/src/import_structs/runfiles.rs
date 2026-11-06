// SPDX-License-Identifier: Apache-2.0

//! Locate DSLX source candidates in Bazel runfiles for the OSS XLS repository.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::{ffi::OsString, os::unix::ffi::OsStringExt};

const XLS_REPOSITORY: &str = "com_google_xls";

/// The portion of a Bazel runfiles tree or manifest that belongs to XLS.
#[derive(Default)]
pub(super) struct XlsRunfiles {
    directory: Option<PathBuf>,
    manifest: BTreeMap<PathBuf, PathBuf>,
    repository: PathBuf,
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
        // Native PathsFrom tries opening each candidate, so an unreadable
        // explicit manifest can fall through to a sibling or the tree.
        let open_manifest = |path: PathBuf| fs::File::open(&path).ok().map(|file| (path, file));
        let mut manifest = manifest.and_then(open_manifest);
        let mut directory = directory.filter(|path| path.is_dir());
        if manifest.is_none()
            && let Some(tree) = directory
                .clone()
                .or_else(|| executable.map(|p| append(&p, ".runfiles")))
        {
            manifest = [tree.join("MANIFEST"), append(&tree, "_manifest")]
                .into_iter()
                .find_map(open_manifest);
            directory = tree.is_dir().then_some(tree);
        }
        if directory.is_none()
            && let Some((path, _)) = &manifest
        {
            directory = manifest_tree_path(path).filter(|path| path.is_dir());
        }

        // A malformed manifest or repository mapping prevents native Bazel
        // runfiles initialization, including fallback to its sibling tree.
        Self::load_manifest(manifest.map(|(_, file)| file), directory).unwrap_or_default()
    }

    /// Load the same manifest entries and repository mapping as native XLS.
    fn load_manifest(manifest: Option<fs::File>, directory: Option<PathBuf>) -> Option<Self> {
        let mut result = Self {
            directory,
            manifest: BTreeMap::new(),
            repository: XLS_REPOSITORY.into(),
        };
        if let Some(mut file) = manifest {
            let mut bytes = Vec::new();
            // C++ getline stops on a read error. In particular, on Unix a
            // directory can open successfully and supply an empty manifest.
            let _ = file.read_to_end(&mut bytes);
            for line in runfiles_lines(&bytes) {
                let escaped = line.starts_with(b" ");
                let entry = line.strip_prefix(b" ").unwrap_or(line);
                let split = entry.iter().position(|&b| b == b' ')?;
                let key = manifest_path(&entry[..split], escaped)?;
                let value = manifest_path(&entry[split + 1..], escaped)?;
                result.manifest.insert(key, value);
            }
        }
        if let Some(mapping) = result
            .resolve(Path::new("_repo_mapping"))
            .and_then(|path| fs::read(path).ok())
        {
            let mut entries = BTreeMap::new();
            for line in runfiles_lines(&mapping) {
                let mut columns = line.splitn(3, |&b| b == b',');
                let source = columns.next()?;
                let apparent = columns.next()?;
                entries.insert((apparent, source), columns.next()?);
            }
            // Bazel's C++ Rlocation selects the floor across all apparent
            // repositories, or the first entry if the key precedes the map.
            // XLS calls it with an empty source repository.
            let key = (XLS_REPOSITORY.as_bytes(), b"".as_slice());
            if let Some((&(apparent, source), &canonical)) = entries
                .range(..=key)
                .next_back()
                .or_else(|| entries.first_key_value())
                && apparent == key.0
                && (source.is_empty() || source == b"*")
            {
                result.repository = manifest_path(canonical, false)?;
            }
        }
        result
            .manifest
            .retain(|key, _| key.starts_with(&result.repository));
        Some(result)
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
            self.resolve(&self.repository.join(suffix))?
        };
        // Native DSLX selects the first existing path. Reading it can then
        // consume a FIFO or report an error for a directory.
        path.exists().then_some(path)
    }

    /// A private module prefix must be absent from the tree and all manifest
    /// entries below that prefix.
    pub(super) fn contains_prefix(&self, prefix: &str) -> bool {
        let key = self.repository.join(prefix);
        self.resolve(&key).is_some_and(|p| p.exists())
            || self
                .manifest
                .range(key.clone()..)
                .next()
                .is_some_and(|(candidate, _)| candidate.starts_with(&key))
    }
}

/// Derive the sibling tree without decoding the manifest path's OS bytes.
fn manifest_tree_path(path: &Path) -> Option<PathBuf> {
    if path.ends_with("MANIFEST") {
        path.parent().map(Path::to_path_buf)
    } else if path.ends_with(".runfiles_manifest") {
        Some(path.with_file_name(".runfiles"))
    } else if path
        .extension()
        .is_some_and(|ext| ext == "runfiles_manifest")
    {
        Some(path.with_extension("runfiles"))
    } else {
        None
    }
}

/// Bazel reads text lines until the first empty line. Its Unix streams
/// preserve a trailing carriage return as part of the physical filename.
fn runfiles_lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    bytes
        .split(|&b| b == b'\n')
        .map(|line| {
            if cfg!(windows) {
                line.strip_suffix(b"\r").unwrap_or(line)
            } else {
                line
            }
        })
        .take_while(|line| !line.is_empty())
}

fn append(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(suffix);
    value.into()
}

/// Decode Bazel's space, newline and backslash escapes without imposing a
/// text encoding on a Unix filename.
fn manifest_path(text: &[u8], escaped: bool) -> Option<PathBuf> {
    let mut bytes = Vec::with_capacity(text.len());
    let mut input = text.iter().copied();
    while let Some(b) = input.next() {
        if escaped && b == b'\\' {
            match input.next() {
                Some(b's') => bytes.push(b' '),
                Some(b'n') => bytes.push(b'\n'),
                Some(b'b') => bytes.push(b'\\'),
                Some(other) => bytes.extend([b'\\', other]),
                None => bytes.push(b'\\'),
            }
        } else {
            bytes.push(b);
        }
    }
    #[cfg(unix)]
    {
        Some(OsString::from_vec(bytes).into())
    }
    #[cfg(not(unix))]
    {
        String::from_utf8(bytes).ok().map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::{ffi::OsStrExt, fs::PermissionsExt};
    use std::process::Command;
    use tempfile::tempdir;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};

    use crate::{ImportOptions, check_module_with_options, parse_module};

    #[test]
    fn runfile_precedence_matches_native_imports() -> Result<(), Box<dyn std::error::Error>> {
        // libxls caches its runfiles. Each child observes exactly one manifest
        // and mapping without changing the environment of other tests.
        if let Ok(expected) = env::var("BX_RUNFILES_PROBE_TYPE") {
            // The fallback search module returns u32. If BX skips a runfile
            // directory, the fallback would compile and fail this error case.
            let ty = if expected == "error" {
                "u32"
            } else {
                &expected
            };
            let search = Path::new("search");
            let mut data = ImportData::new(None, &[search]);
            let native = parse_and_typecheck(
                &format!("import wire; fn run() -> {ty} {{ wire::value() }}"),
                "run.x",
                "run",
                &mut data,
            );
            let module = parse_module(&format!(
                "use wire; block sample(clk) -> {ty} {{ wire::value() }}"
            ))?;
            let options = ImportOptions {
                additional_search_paths: vec![search],
                ..Default::default()
            };
            let bx = check_module_with_options(&module, &options);
            assert_eq!(native.is_ok(), expected != "error", "{:?}", native.err());
            assert_eq!(bx.is_ok(), expected != "error", "{:?}", bx.err());
            return Ok(());
        }

        let temp = tempdir()?;
        let tree = temp.path().join("test.runfiles");
        let mapped = tree.join("mapped");
        for (root, ty) in [
            (tree.join(XLS_REPOSITORY), "u8"),
            (mapped.clone(), "u16"),
            (temp.path().join("search"), "u32"),
        ] {
            fs::create_dir_all(&root)?;
            fs::write(
                root.join("wire.x"),
                format!("pub fn value() -> {ty} {{ {ty}:42 }}"),
            )?;
        }
        let directory_entry = format!("{XLS_REPOSITORY}/wire.x {}\n", mapped.display());
        let file_entry = format!(
            "{XLS_REPOSITORY}/wire.x {}",
            mapped.join("wire.x").display()
        );
        let probe =
            |expected: &str, manifest_file: Option<&Path>, case: &str| -> std::io::Result<()> {
                let mut command = Command::new(env::current_exe()?);
                command
                    .arg("runfile_precedence_matches_native_imports")
                    .current_dir(temp.path())
                    .env("BX_RUNFILES_PROBE_TYPE", expected)
                    .env("RUNFILES_DIR", &tree)
                    .env_remove("RUNFILES_MANIFEST_FILE");
                if let Some(path) = manifest_file {
                    command.env("RUNFILES_MANIFEST_FILE", path);
                }
                let output = command.output()?;
                assert!(
                    output.status.success(),
                    "{case}: {}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                Ok(())
            };
        let crlf_type = if cfg!(windows) { "u16" } else { "u32" };
        for (manifest, mapping, expected) in [
            (directory_entry.as_str(), "", "error"),
            ("", "*,com_google_xls,mapped\n", "u16"),
            ("", ",aaa,aaa\n*,com_google_xls,mapped\n", "u8"),
            ("", "\n,com_google_xls,mapped\n", "u8"),
            ("", ",com_google_xls,mapped\ninvalid\n", "u32"),
            (&format!("{file_entry}\ninvalid\n"), "", "u32"),
            (&format!("{file_entry}\r\n"), "", crlf_type),
            ("", ",com_google_xls,mapped\r\n", crlf_type),
        ] {
            fs::write(tree.join("MANIFEST"), manifest)?;
            fs::write(tree.join("_repo_mapping"), mapping)?;
            probe(
                expected,
                None,
                &format!("manifest={manifest:?} mapping={mapping:?}"),
            )?;
        }

        fs::write(tree.join("_repo_mapping"), "")?;
        let sibling = tree.join("MANIFEST");
        let readable_entry = format!("{file_entry}\n");
        fs::write(&sibling, &readable_entry)?;
        if fs::File::open(&mapped).is_ok() {
            // An open directory supplies an empty manifest to C++ getline.
            probe("u8", Some(&mapped), "explicit directory as manifest")?;
        }
        #[cfg(unix)]
        {
            let unreadable = temp.path().join("unreadable");
            fs::write(&unreadable, &readable_entry)?;
            fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o000))?;
            // Privileged users and some filesystems can still open mode 000.
            if fs::File::open(&unreadable).is_err() {
                probe("u16", Some(&unreadable), "unreadable explicit manifest")?;
                fs::set_permissions(&sibling, fs::Permissions::from_mode(0o000))?;
                probe("u8", None, "unreadable sibling with usable tree")?;
                fs::write(append(&tree, "_manifest"), &readable_entry)?;
                probe("u16", None, "readable alternative sibling manifest")?;
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn manifest_tree_paths_preserve_non_utf8_components() {
        for (manifest, tree) in [
            (
                &b"dir_\xff/tool.runfiles_manifest"[..],
                &b"dir_\xff/tool.runfiles"[..],
            ),
            (b"tool_\xff.runfiles_manifest", b"tool_\xff.runfiles"),
            (b"dir_\xff/.runfiles_manifest", b"dir_\xff/.runfiles"),
        ] {
            let path = Path::new(std::ffi::OsStr::from_bytes(manifest));
            assert_eq!(
                manifest_tree_path(path),
                Some(std::ffi::OsStr::from_bytes(tree).into())
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn manifest_preserves_non_utf8_target_bytes() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let target = dir
            .path()
            .join(OsString::from_vec(b"source_\xff.x".to_vec()));
        let manifest = dir.path().join("manifest");
        let mut entry = b"com_google_xls/example.x ".to_vec();
        entry.extend(target.as_os_str().as_bytes());
        entry.push(b'\n');
        fs::write(&manifest, entry)?;
        let runfiles = XlsRunfiles::from_locations(Some(manifest), None, None);
        assert_eq!(
            runfiles.resolve(Path::new("com_google_xls/example.x")),
            Some(target)
        );
        Ok(())
    }

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
