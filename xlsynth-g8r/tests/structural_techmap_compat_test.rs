// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::netlist::techmap::{
    StructuralMappedNetlist, StructuralTechMapOptions, map_gatefn_to_structural_netlist,
};

#[test]
fn structural_techmap_remains_available_at_its_original_public_path() {
    let _: fn(&GateFn, &StructuralTechMapOptions) -> anyhow::Result<StructuralMappedNetlist> =
        map_gatefn_to_structural_netlist;
}
