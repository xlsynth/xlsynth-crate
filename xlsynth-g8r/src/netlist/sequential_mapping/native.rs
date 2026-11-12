// SPDX-License-Identifier: Apache-2.0

//! The combinational cover proposes cells. The assembled stage determines how
//! they time. In particular, capture D capacitance loads the final logic gate.
//!
//! ```text
//! input -> [launch FF] --Q--> combinational cover --D--> [capture FF] -> output
//!            |<-------- clock-to-Q + logic + setup -------->|
//! ```
use anyhow::{Error, Result};

use super::SequentialMappingOptions;
use super::assembly::SequentialMappingCatalog;
use crate::aig::SequentialGateFn;
use crate::aig_serdes::canonical_mapping_graph;
use crate::liberty_model::Library;
use crate::netlist::io::ParsedNetlist;
use crate::netlist::register_mapping::RegisterCellSelection;
use crate::netlist::registered_opt::{RegisteredOptimizationOptions, optimize_registered_netlist};
use crate::netlist::resize::{ResizeOptions, resize_netlist};
use crate::netlist::sta::StaOptions;
use crate::techmap::area::AreaMappedNetlist;
use crate::techmap::delay::{DelayMappingLibrary, map_gatefn_to_delay_candidates};
use crate::techmap::postprocess::{CoverStrategy, PostMappingOptions, map_and_optimize_cover};

/// Maps combinational logic with g8r's area mapper and assembles the selected
/// flops and clock gates into the same flat netlist.
///
/// Flop selection precedes combinational mapping. With no library inverter,
/// mapping can fail even when another choice of flop polarities would work.
pub fn map_sequential_to_area_netlist(
    design: &SequentialGateFn,
    library: &Library,
    options: &SequentialMappingOptions,
) -> Result<ParsedNetlist> {
    map_sequential_to_native_netlist(
        design,
        library,
        options,
        CoverStrategy::Area,
        &PostMappingOptions::default(),
    )
}

/// Covers the transition graph, then reconnects its original ports and physical
/// registers.
pub fn map_sequential_to_native_netlist(
    design: &SequentialGateFn,
    library: &Library,
    options: &SequentialMappingOptions,
    strategy: CoverStrategy,
    post: &PostMappingOptions,
) -> Result<ParsedNetlist> {
    map_sequential_with_policy(design, library, options, strategy, post, None)
}

/// Maps with an optional final sizing policy. Registered candidates are
/// evaluated with their actual launch drivers and capture pin loads.
pub fn map_sequential_with_policy(
    design: &SequentialGateFn,
    library: &Library,
    options: &SequentialMappingOptions,
    strategy: CoverStrategy,
    post: &PostMappingOptions,
    final_resize: Option<&ResizeOptions>,
) -> Result<ParsedNetlist> {
    let catalog = SequentialMappingCatalog::new(library).map_err(Error::msg)?;
    let prepared = catalog
        .prepare(library, design, options)
        .map_err(Error::msg)?;
    let canonical;
    let graph = if matches!(strategy, CoverStrategy::Delay(_)) {
        // Match the node and operand ordering used by aig-tech-map's binary AIG
        // input.
        canonical = canonical_mapping_graph(prepared.graph()).map_err(Error::msg)?;
        &canonical
    } else {
        prepared.graph()
    };
    let assemble = |cover: AreaMappedNetlist| -> Result<ParsedNetlist> {
        let mut netlist = cover.netlist.into();
        prepared.assemble(&mut netlist, "g8r").map_err(Error::msg)?;
        Ok(netlist)
    };
    if prepared.design.registers.is_empty() {
        let (cover, _) = map_and_optimize_cover(graph, library, strategy, post)?;
        let mut mapped = assemble(cover)?;
        if let Some(resize) = final_resize {
            let [module] = mapped.modules.as_mut_slice() else {
                anyhow::bail!("expected one assembled module for resizing");
            };
            resize_netlist(module, &mapped.nets, &mut mapped.interner, library, resize)?;
        }
        return Ok(mapped);
    }

    // Full sizing includes area recovery. A fixed register-cell choice also
    // constrains sizing on the assembled circuit.
    let mut resize = final_resize.or(post.recovery.as_ref()).cloned();
    if let Some(policy) = &mut resize {
        policy.resize_registers &= final_resize.is_some()
            && !matches!(
                options.register_selection,
                RegisterCellSelection::FixedCell(_)
            );
    }
    let timing = match (&resize, &post.fanout) {
        (Some(policy), _) => policy.timing,
        (None, Some(fanout)) => StaOptions {
            primary_input_transition: fanout.primary_input_transition,
            module_output_load: fanout.module_output_load,
            wire_load: fanout.wire_load,
        },
        (None, None) => return assemble(map_and_optimize_cover(graph, library, strategy, post)?.0),
    };
    let mut policy = RegisteredOptimizationOptions {
        timing,
        fanout: post.fanout.clone(),
        resize,
    };
    let mut finish = |cover| -> Result<_> {
        let mut netlist = assemble(cover)?;
        let report = optimize_registered_netlist(&mut netlist, library, &policy)?;
        if let Some(sizing) = &mut policy.resize {
            if let Some(left) = &mut sizing.timing_evaluation_budget {
                *left = left.saturating_sub(report.timing_evaluations);
            }
            if let Some(left) = &mut sizing.recovery_evaluation_budget {
                *left = left.saturating_sub(report.recovery_evaluations);
            }
        }
        log::debug!(target: "xlsynth_g8r::netlist::registered_opt", "{report:?}");
        Ok(((report.objective_delay, report.final_area), netlist))
    };
    match strategy {
        CoverStrategy::Area => {
            let (cover, _) =
                map_and_optimize_cover(graph, library, strategy, &PostMappingOptions::default())?;
            Ok(finish(cover)?.1)
        }
        CoverStrategy::Delay(delay) => {
            let cells = DelayMappingLibrary::new(library, delay)?;
            let candidates = map_gatefn_to_delay_candidates(graph, &cells)?;
            let Some(local) = candidates.local else {
                return Ok(finish(candidates.demanded)?.1);
            };
            let local = finish(local);
            match (finish(candidates.demanded), local) {
                (Ok(demanded), Ok(local)) => Ok(if local.0 < demanded.0 {
                    local.1
                } else {
                    demanded.1
                }),
                (Ok(best), Err(_)) | (Err(_), Ok(best)) => Ok(best.1),
                (Err(demanded), Err(local)) => Err(demanded.context(format!(
                    "local-polarity registered cover also failed: {local:#}"
                ))),
            }
        }
    }
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

    // Exercise final sizing with an explicitly selected register cell.
    // Combinational optimization may proceed while the register master stays
    // fixed.
    #[test]
    fn full_sizing_preserves_an_explicit_register_master() -> Result<()> {
        let library = crate::netlist::resize::tests::library()?;
        let mut builder = GateBuilder::new("bank".into(), GateBuilderOptions::no_opt());
        let data = builder.add_input("data".into(), 3);
        builder.add_output("result".into(), data);
        let clock = ClockPort { name: "clk".into() };
        let launched = crate::aig::add_input_registers(
            &SequentialGateFn::from_gate_fn(builder.build()),
            clock.clone(),
        )
        .map_err(Error::msg)?;
        let design = add_output_registers(&launched, clock).map_err(Error::msg)?;
        let mapped = map_sequential_with_policy(
            &design,
            &library,
            &SequentialMappingOptions {
                register_selection: RegisterCellSelection::FixedCell("DFF4".into()),
                ..Default::default()
            },
            CoverStrategy::Area,
            &PostMappingOptions::default(),
            Some(&ResizeOptions::default()),
        )?;
        let cells = mapped.modules[0]
            .instances
            .iter()
            .map(|instance| mapped.interner.resolve(instance.type_name))
            .collect::<Vec<_>>();
        assert_eq!(cells, vec![Some("DFF4"); 6]);
        Ok(())
    }

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
