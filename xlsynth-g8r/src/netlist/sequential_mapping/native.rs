// SPDX-License-Identifier: Apache-2.0

use anyhow::{Error, Result};

use super::SequentialMappingOptions;
use super::assembly::SequentialMappingCatalog;
use crate::aig::SequentialGateFn;
use crate::liberty_model::Library;
use crate::netlist::io::ParsedNetlist;
use crate::techmap::area::{AreaMappingLibrary, map_gatefn_to_area_netlist};

/// Maps combinational logic with g8r's area mapper and assembles the selected
/// flops and clock gates into the same flat netlist.
pub fn map_sequential_to_area_netlist(
    design: &SequentialGateFn,
    library: &Library,
    options: &SequentialMappingOptions,
) -> Result<ParsedNetlist> {
    let catalog = SequentialMappingCatalog::new(library).map_err(Error::msg)?;
    let prepared = catalog
        .prepare(library, design, options)
        .map_err(Error::msg)?;
    let cells = AreaMappingLibrary::new(library)?;
    let mut netlist = map_gatefn_to_area_netlist(prepared.graph(), &cells)?
        .netlist
        .into();
    prepared.assemble(&mut netlist, "g8r").map_err(Error::msg)?;
    Ok(netlist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{
        AigBitVector, ClockPort, GateBuilder, GateBuilderOptions, add_output_registers,
    };
    use crate::aig_sim::sequential::{SequentialState, simulate};
    use crate::liberty::parser::parse_liberty_files;
    use crate::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
    use std::path::Path;
    use xlsynth_pir::IrBits;

    #[test]
    fn wire_and_constant_register_data_need_only_a_flop_library() -> Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/clock_gate.lib");
        let mut library = parse_liberty_files(&[fixture]).map_err(Error::msg)?;
        library.cells.retain(|cell| cell.name == "DFF");
        for constant_msb in [false, true] {
            let mut builder = GateBuilder::new("bank".into(), GateBuilderOptions::no_opt());
            let x = builder.add_input("x".into(), 1);
            let x = *x.get_lsb(0);
            let msb = if constant_msb { builder.get_true() } else { x };
            builder.add_output("y".into(), AigBitVector::from_lsb_is_index_0(&[x, msb]));
            let design = add_output_registers(
                &SequentialGateFn::from_gate_fn(builder.build()),
                ClockPort { name: "clk".into() },
            )
            .map_err(Error::msg)?;
            let mapped = map_sequential_to_area_netlist(
                &design,
                &library,
                &SequentialMappingOptions::default(),
            )?;
            let module = &mapped.modules[0];
            assert_eq!(module.instances.len(), 2);
            let recovered = project_labeled_sequential_netlist_aig(
                module,
                &mapped.nets,
                &mapped.interner,
                &library,
                Some("clk"),
            )
            .map_err(Error::msg)?
            .sequential_gate_fn;
            let stimulus = [false, true, false].map(|b| vec![IrBits::bool(b)]);
            let original = simulate(&design, &stimulus, SequentialState::all_zeros(&design))
                .map_err(Error::msg)?;
            let actual = simulate(
                &recovered,
                &stimulus,
                SequentialState::all_zeros(&recovered),
            )
            .map_err(Error::msg)?;
            assert_eq!(original.external_outputs(), actual.external_outputs());
        }
        Ok(())
    }
}
