// SPDX-License-Identifier: Apache-2.0

//! Compatibility exports for the structural technology mapper.

pub use crate::netlist::mapped::MappedNetlist as StructuralMappedNetlist;
pub use crate::techmap::nand2::{StructuralTechMapOptions, map_gatefn_to_structural_netlist};
