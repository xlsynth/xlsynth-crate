// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

pub const BIT_SCANS_DSLX: &str = include_str!("dslx/bit_scans.x");
pub const CLZ_VARIANTS_DSLX: &str = include_str!("dslx/clz_variants.x");
pub const ADD_VARIANTS_DSLX: &str = include_str!("dslx/add_variants.x");
pub const MUL_VARIANTS_DSLX: &str = include_str!("dslx/mul_variants.x");
pub const ADD_SEQ_VARIANTS_DSLX: &str = include_str!("dslx/add_seq_variants.x");
pub const MUL_SEQ_VARIANTS_DSLX: &str = include_str!("dslx/mul_seq_variants.x");

pub const DSLX_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/dslx");

pub fn dslx_dir() -> &'static Path {
    Path::new(DSLX_DIR)
}

pub fn bit_scans_path() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/dslx/bit_scans.x"))
}

pub fn clz_variants_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/dslx/clz_variants.x"
    ))
}

pub fn add_variants_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/dslx/add_variants.x"
    ))
}

pub fn mul_variants_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/dslx/mul_variants.x"
    ))
}

pub fn add_seq_variants_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/dslx/add_seq_variants.x"
    ))
}

pub fn mul_seq_variants_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/dslx/mul_seq_variants.x"
    ))
}
