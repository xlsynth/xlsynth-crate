// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::{Arg, ArgMatches, ValueEnum};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use xlsynth_g8r::netlist::emit::emit_netlist_as_text;
use xlsynth_g8r::netlist::io::ParsedNetlist;
use xlsynth_g8r::netlist::proto::encode_netlist;

/// Output formats accepted by the technology mapping commands.
#[derive(Clone, Copy, ValueEnum)]
pub enum Format {
    Gv,
    Pb,
}

pub fn output_arg() -> Arg {
    Arg::new("output")
        .long("output")
        .short('o')
        .value_name("PATH")
        .default_value("-")
        .help("Mapped netlist output path ('-' for stdout). Defaults to protobuf for .pb or .proto and Verilog otherwise")
}

pub fn format_arg() -> Arg {
    Arg::new("netlist-format")
        .long("netlist-format")
        .value_name("FORMAT")
        .value_parser(clap::value_parser!(Format))
        .help("Mapped netlist format: gate-level Verilog or binary netlist protobuf")
}

impl Format {
    /// An explicit format overrides the output extension. Stdout defaults to
    /// GV.
    pub fn from_matches(matches: &ArgMatches, output: &Path) -> Self {
        if let Some(format) = matches.get_one::<Self>("netlist-format") {
            return *format;
        }
        match output.extension().and_then(|ext| ext.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("pb") || ext.eq_ignore_ascii_case("proto") => {
                Self::Pb
            }
            _ => Self::Gv,
        }
    }

    /// Serializes the mapper's final netlist directly in the selected format.
    pub fn encode(self, netlist: &ParsedNetlist) -> Result<Vec<u8>> {
        match self {
            Self::Gv => Ok(emit_netlist_as_text(netlist)
                .context("failed to emit mapped netlist text")?
                .into_bytes()),
            Self::Pb => encode_netlist(netlist).context("failed to encode mapped netlist protobuf"),
        }
    }
}

/// Flushes the entire netlist before continuing to sidecar output or exiting.
pub fn write_stdout(bytes: &[u8]) -> Result<()> {
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(bytes)
        .and_then(|()| stdout.flush())
        .context("writing netlist to stdout")
}

/// Resolves existing output targets or their parents before either file is
/// written.
fn resolved_output_path(path: &Path) -> Result<PathBuf> {
    match path.canonicalize() {
        Ok(resolved) => Ok(resolved),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            // A dangling file symlink still identifies its eventual write
            // target.
            if path
                .symlink_metadata()
                .is_ok_and(|m| m.file_type().is_symlink())
            {
                let target = std::fs::read_link(path)?;
                return resolved_output_path(&parent.join(target));
            }
            let name = path.file_name().context("output path has no file name")?;
            // file_name() strips trailing separators and "/.". Those spellings
            // require a directory and cannot designate a new regular file.
            anyhow::ensure!(
                path.as_os_str()
                    .as_encoded_bytes()
                    .ends_with(name.as_encoded_bytes()),
                "output path '{}' requires a directory",
                path.display()
            );
            Ok(parent.canonicalize()?.join(name))
        }
        Err(error) => Err(error.into()),
    }
}

/// Detects equal destinations, including symlinks, hard links, and
/// not-yet-created files.
pub(super) fn output_paths_alias(lhs: &Path, rhs: &Path) -> Result<bool> {
    let lhs_metadata = metadata_if_exists(lhs)?;
    let rhs_metadata = metadata_if_exists(rhs)?;
    if let (Some(lhs_meta), Some(rhs_meta)) = (&lhs_metadata, &rhs_metadata) {
        // Distinct open files can have the same /proc/self/fd link text after
        // unlinking. Prefer the identity of the files those paths actually
        // open.
        #[cfg(unix)]
        return Ok(same_unix_file(lhs_meta, rhs_meta));
        #[cfg(not(unix))]
        if lhs_meta.is_file() && rhs_meta.is_file() {
            if let (Ok(lhs_handle), Ok(rhs_handle)) = (identity_handle(lhs), identity_handle(rhs)) {
                return Ok(lhs_handle == rhs_handle);
            }
        }
    }
    if lhs_metadata.is_some() != rhs_metadata.is_some() {
        // Check the missing destination's parent before either output is
        // written.
        resolved_output_path(if lhs_metadata.is_none() { lhs } else { rhs })?;
        return Ok(false);
    }
    Ok(resolved_output_path(lhs)? == resolved_output_path(rhs)?)
}

/// Reads metadata without treating a missing output as an error.
pub(super) fn metadata_if_exists(path: &Path) -> std::io::Result<Option<std::fs::Metadata>> {
    match std::fs::metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Identifies existing stream-like outputs, excluding invalid directories.
pub(super) fn is_special_output(path: &Path) -> std::io::Result<bool> {
    Ok(metadata_if_exists(path)?.is_some_and(|metadata| !metadata.is_file() && !metadata.is_dir()))
}

/// Compares Unix file identities without opening either file's data.
#[cfg(unix)]
pub(super) fn same_unix_file(lhs: &std::fs::Metadata, rhs: &std::fs::Metadata) -> bool {
    (lhs.dev(), lhs.ino()) == (rhs.dev(), rhs.ino())
}

/// Opens a file for identity queries without requesting read or write access.
#[cfg(windows)]
fn identity_handle(path: &Path) -> std::io::Result<same_file::Handle> {
    let file = OpenOptions::new().access_mode(0).open(path)?;
    same_file::Handle::from_file(file)
}

/// Opens a file for identity queries on other supported platforms.
#[cfg(not(any(unix, windows)))]
fn identity_handle(path: &Path) -> std::io::Result<same_file::Handle> {
    same_file::Handle::from_path(path)
}

/// Checks that a placeholder still names the file that was created.
fn path_matches_handle(path: &Path, handle: &same_file::Handle) -> bool {
    #[cfg(unix)]
    {
        match (std::fs::metadata(path), handle.as_file().metadata()) {
            (Ok(path_metadata), Ok(handle_metadata)) => {
                same_unix_file(&path_metadata, &handle_metadata)
            }
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file())
            && identity_handle(path).is_ok_and(|path_handle| path_handle == *handle)
    }
}

/// An output opened without truncation, with cleanup for a new placeholder.
pub(super) struct PendingOutput {
    handle: Option<same_file::Handle>,
    created_path: Option<PathBuf>,
}

impl PendingOutput {
    /// Opens a destination without destroying existing contents.
    pub(super) fn open(path: &Path) -> Result<Self> {
        // Open existing paths as given, including descriptor paths for unlinked
        // files. Resolve a missing target for cleanup without deleting its
        // symlink.
        let resolved = if metadata_if_exists(path)?.is_some() {
            path.to_owned()
        } else {
            resolved_output_path(path)?
        };
        let (file, created) = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&resolved)
        {
            Ok(file) => (file, true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                (OpenOptions::new().write(true).open(&resolved)?, false)
            }
            Err(error) => return Err(error.into()),
        };
        let handle = match same_file::Handle::from_file(file) {
            Ok(handle) => handle,
            Err(error) => {
                if created {
                    let _ = std::fs::remove_file(&resolved);
                }
                return Err(error.into());
            }
        };
        Ok(Self {
            handle: Some(handle),
            created_path: created.then_some(resolved),
        })
    }

    pub(super) fn handle(&self) -> &same_file::Handle {
        self.handle.as_ref().unwrap()
    }

    /// Keeps the destination and truncates it immediately before writing.
    pub(super) fn prepare_write(&mut self) -> Result<&mut File> {
        self.created_path = None;
        let file = self.handle.as_mut().unwrap().as_file_mut();
        if file.metadata()?.is_file() {
            file.set_len(0)?;
        }
        Ok(file)
    }
}

impl Drop for PendingOutput {
    fn drop(&mut self) {
        if let Some(path) = self.created_path.take() {
            let handle = self.handle.take().unwrap();
            let still_same_file = path_matches_handle(&path, &handle);
            drop(handle);
            if still_same_file {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

/// Detects an existing path that aliases a standard output stream.
pub(super) fn path_aliases_stream(
    path: &Path,
    stream: &same_file::Handle,
    #[cfg_attr(not(unix), allow(unused_variables))] stream_paths: &[&str],
) -> Result<bool> {
    let Some(metadata) = metadata_if_exists(path)? else {
        return Ok(false);
    };
    #[cfg(unix)]
    let aliases = {
        // On some Unix systems, stat of a /dev/fd entry reports the identity
        // of its descriptor vnode rather than the identity returned by fstat
        // on the descriptor itself. Compare both forms without opening a
        // potentially blocking FIFO or another special output.
        same_unix_file(&metadata, &stream.as_file().metadata()?)
            || stream_paths.iter().any(|stream_path| {
                std::fs::metadata(stream_path)
                    .is_ok_and(|stream_metadata| same_unix_file(&metadata, &stream_metadata))
            })
    };
    #[cfg(not(unix))]
    let aliases = metadata.is_file() && identity_handle(path).is_ok_and(|handle| handle == *stream);
    Ok(aliases)
}

#[cfg(all(test, unix))]
mod tests {
    use super::PendingOutput;
    #[cfg(target_os = "linux")]
    use super::output_paths_alias;
    use std::fs;
    #[cfg(target_os = "linux")]
    use std::io::Write;
    #[cfg(target_os = "linux")]
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;

    #[test]
    fn missing_outputs_that_require_a_directory_are_rejected() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let link = dir.path().join("link");
        symlink("missing/", &link)?;
        for path in [
            dir.path().join("missing/"),
            dir.path().join("missing/."),
            link.clone(),
        ] {
            assert!(PendingOutput::open(&path).is_err(), "{}", path.display());
            assert!(!dir.path().join("missing").exists());
        }
        assert_eq!(fs::read_link(link)?, PathBuf::from("missing/"));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn descriptor_paths_retain_the_identity_of_unlinked_files() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("output");
        let first = fs::File::create(&path)?;
        fs::remove_file(&path)?;
        let second = fs::File::create(&path)?;
        fs::remove_file(&path)?;
        let first_path = PathBuf::from(format!("/proc/self/fd/{}", first.as_raw_fd()));
        let second_path = PathBuf::from(format!("/proc/self/fd/{}", second.as_raw_fd()));
        assert_eq!(fs::read_link(&first_path)?, fs::read_link(&second_path)?);
        assert!(!output_paths_alias(&first_path, &second_path)?);

        let mut output = PendingOutput::open(&first_path)?;
        output.prepare_write()?.write_all(b"netlist")?;
        assert_eq!(fs::read(&first_path)?, b"netlist");
        assert!(fs::read(&second_path)?.is_empty());
        assert!(fs::read_dir(dir.path())?.next().is_none());
        Ok(())
    }
}
