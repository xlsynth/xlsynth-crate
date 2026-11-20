// SPDX-License-Identifier: Apache-2.0

//! Generate buffer-tree alternatives without deciding which circuit wins.
//! The caller evaluates each complete netlist against its timing and area
//! policy.

use super::{buffers::Buffer, edit, plan};
use crate::liberty_model::Library;
use crate::netlist::editability::EditableOutput as Root;
use crate::netlist::{mapped::MappedNetlist, sta::StaOptions};
use anyhow::{Context, Result};

pub(super) struct TopologyTrial<'a> {
    pub buffer: &'a Buffer,
    pub width: usize,
    pub plans: &'a [(usize, Vec<edit::Branch>)],
}

/// Visit the most stressed launch alone and all fixable launches together.
/// Each alternative starts from the same circuit; duplicate forests are
/// skipped. This keeps the number of STA trials independent of the register
/// count.
pub(super) fn visit_topology_candidates(
    original: &MappedNetlist,
    library: &Library,
    timing: StaOptions,
    roots: &[Root],
    ranked: &[(usize, f64)],
    choices: &[&Buffer],
    mut visit: impl FnMut(MappedNetlist, TopologyTrial<'_>) -> Result<()>,
) -> Result<()> {
    for &buffer in choices {
        let mut previous = Vec::new();
        for count in [1, ranked.len()]
            .into_iter()
            .take(if ranked.len() == 1 { 1 } else { 2 })
        {
            for width in [16, 64] {
                let plans: Vec<_> = ranked[..count]
                    .iter()
                    .filter_map(|&(id, _)| {
                        plan::tree(&roots[id], buffer, width, timing).map(|p| (id, p))
                    })
                    .collect();
                if plans.is_empty() || previous.contains(&plans) {
                    continue;
                }
                let mut trial = original.clone();
                edit::apply_forest(&mut trial, library, buffer, roots, &plans)
                    .context("applying validated register buffer forest")?;
                visit(
                    trial,
                    TopologyTrial {
                        buffer,
                        width,
                        plans: &plans,
                    },
                )?;
                previous.push(plans);
            }
        }
    }
    Ok(())
}
