// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

pub fn collect_ir_files_sorted(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;
            let is_ir = path.extension().and_then(|s| s.to_str()) == Some("ir");
            if ty.is_dir() {
                stack.push(path);
            } else if ty.is_file() {
                if is_ir {
                    files.push(path);
                }
            } else if ty.is_symlink() {
                if is_ir {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        if meta.is_file() {
                            files.push(path);
                        }
                    }
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

pub fn walk_ir_files_sorted<F>(
    root: &Path,
    max_files: Option<usize>,
    mut visitor: F,
) -> std::io::Result<()>
where
    F: FnMut(&Path) -> bool,
{
    let mut files = collect_ir_files_sorted(root)?;
    if let Some(limit) = max_files {
        files.truncate(limit);
    }

    for path in files {
        if !visitor(&path) {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::os::unix::fs::symlink;

    #[test]
    fn walk_ir_files_sorted_stable_order_and_max_files() {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path().join("corpus");
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();

        std::fs::write(root.join("z.ir"), "package p\n").unwrap();
        std::fs::write(root.join("a.ir"), "package p\n").unwrap();
        std::fs::write(nested.join("b.ir"), "package p\n").unwrap();
        std::fs::write(root.join("ignore.txt"), "not ir").unwrap();

        let mut seen = Vec::<String>::new();
        walk_ir_files_sorted(&root, Some(2), |path| {
            seen.push(path.file_name().unwrap().to_str().unwrap().to_string());
            true
        })
        .unwrap();

        assert_eq!(seen, vec!["a.ir", "b.ir"]);
    }

    #[cfg(unix)]
    #[test]
    fn collect_ir_files_sorted_includes_symlinked_ir_files() {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path().join("corpus");
        let real = temp_dir.path().join("real.ir");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(&real, "package p\n").unwrap();

        let link = root.join("link.ir");
        symlink(&real, &link).unwrap();

        let files = collect_ir_files_sorted(&root).unwrap();
        assert_eq!(files, vec![link]);
    }
}
