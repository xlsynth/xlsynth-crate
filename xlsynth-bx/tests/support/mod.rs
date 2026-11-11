// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_bx::ImportOptions;

/// Build default BX import options with the given ordered search roots.
pub(crate) fn search_paths<'a>(roots: &[&'a Path]) -> ImportOptions<'a> {
    ImportOptions {
        additional_search_paths: roots.to_vec(),
        ..Default::default()
    }
}
