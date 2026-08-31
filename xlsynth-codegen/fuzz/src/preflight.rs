// SPDX-License-Identifier: Apache-2.0

//! Authoritative external-oracle setup for fuzz initialization and examples.

use xlsynth_g8r_fuzz::external_yosys::preflight_mapping;
use xlsynth_test_helpers::iverilog::required_iverilog_toolchain;

use crate::{Profile, xls};

/// The external checks selected by a fuzz target or evaluation profile.
#[derive(Clone, Copy, Debug)]
pub enum Oracles {
    Icarus,
    YosysCombinational,
    YosysSequential,
    StockXls,
}

impl From<Profile> for Oracles {
    fn from(profile: Profile) -> Self {
        match profile {
            Profile::NativeSemantics => Self::Icarus,
            Profile::StockXls => Self::StockXls,
        }
    }
}

/// Validates only the selected oracles before processing any fuzz samples.
pub fn validate(oracles: Oracles) -> Result<(), String> {
    match oracles {
        Oracles::Icarus => required_iverilog_toolchain()
            .map(|_| ())
            .map_err(str::to_owned),
        Oracles::YosysCombinational => preflight_mapping(false),
        Oracles::YosysSequential => preflight_mapping(true),
        Oracles::StockXls => {
            required_iverilog_toolchain().map_err(str::to_owned)?;
            xls::preflight_stock_xls()?;
            preflight_mapping(true)
        }
    }
}
