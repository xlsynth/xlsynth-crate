// SPDX-License-Identifier: Apache-2.0

use xlsynth_lean::block::LeanBlockBundle;

/// Look up a generated module independently of its position in import order.
pub fn bundle_file<'a>(bundle: &'a LeanBlockBundle, name: &str) -> &'a str {
    bundle
        .files
        .iter()
        .find(|(file, _)| file == name)
        .unwrap_or_else(|| panic!("missing generated file {name}"))
        .1
        .as_str()
}
