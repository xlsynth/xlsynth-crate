// SPDX-License-Identifier: Apache-2.0

//! Reconsider the buffer trees inserted by registered optimization. Evaluate
//! each replacement in its own copy of the current circuit and preserve
//! wiring that does not belong to the mapper's trees.

use super::{OwnedBuffers, buffer_choices, edit, graph, plan};
use crate::liberty_model::Library;
use crate::netlist::io::ParsedNetlist;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::parse::PortId;
use crate::netlist::registered_opt::{FanoutAreaPolicy, Score, preserves_endpoints};
use crate::netlist::resize::timing::ElectricalGuard;
use crate::netlist::resize::verify_timing;
use crate::netlist::sta::incremental::{
    IncrementalSta, TIMING_COMPARISON_TOLERANCE, measure_boundary_delays_with_library,
};
use crate::netlist::sta::{PreparedStaLibrary, StaOptions};
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Default, Serialize)]
pub struct TreeRefinementReport {
    pub initial_delay: f64,
    pub final_delay: f64,
    pub final_area: f64,
    pub rounds: Vec<TreeRound>,
}

#[derive(Default, Serialize)]
pub struct TreeRound {
    pub eligible_roots: usize,
    pub ordinary_trials: usize,
    pub groups_with_protected_or_missing_roots: usize,
    pub infeasible_geometries: usize,
    pub rejected_electrical: usize,
    pub rejected_endpoints: usize,
    pub unsupported_timing: usize,
    pub rejected_score: usize,
    pub accepted: Option<TreeEdit>,
}

#[derive(Serialize)]
pub struct TreeEdit {
    pub roots: Vec<String>,
    pub geometry: &'static str,
    pub buffer: Option<String>,
    pub max_leaf_fanout: Option<usize>,
    pub cells_removed: usize,
    pub cells_inserted: usize,
    pub delay: f64,
}

type Pin = (PortId, PortId);

/// Refine one assembled module using its original fanout area policy.
/// Buffer ownership must accompany the exact in-memory netlist returned by
/// the optimizer. An empty ledger still permits new trees on critical
/// unbuffered drivers. The original netlist is retained on errors.
pub(crate) fn refine_registered_trees(
    netlist: &mut ParsedNetlist,
    library: &Library,
    timing: StaOptions,
    ownership: &mut OwnedBuffers,
    policy: FanoutAreaPolicy,
) -> Result<TreeRefinementReport> {
    let [module] = netlist.modules.as_slice() else {
        anyhow::bail!("tree refinement requires one assembled module");
    };
    let mut current = MappedNetlist {
        module: module.clone(),
        nets: netlist.nets.clone(),
        interner: netlist.interner.clone(),
    };
    let mut owned = ownership.clone();
    let prepared = Arc::new(PreparedStaLibrary::new(library)?);
    let analyze = |net: &MappedNetlist| {
        IncrementalSta::with_registered_library(
            &net.module,
            &net.nets,
            &net.interner,
            Arc::clone(&prepared),
            timing,
        )
    };
    let mut sta = analyze(&current)?;
    let initial_delay = sta.score().values()[sta.objective()];
    let mut report = TreeRefinementReport {
        initial_delay,
        final_delay: initial_delay,
        final_area: 0.0,
        rounds: vec![],
    };
    let buffers = graph::buffers(library, &prepared);
    let choices = buffer_choices(&buffers, library);
    // Limit expensive whole-stage evaluations independently of the number of
    // endpoints in a tied critical region. Refresh required times after a win.
    for _ in 0..2 {
        let mut round = TreeRound::default();
        let live = graph::collect_roots(&current, library, timing.module_output_load, Some(1))?;
        let objective = sta.objective();
        let sites = critical_trees(&current, &sta, &owned, &live, objective)?;
        round.eligible_roots = sites.len();
        let base_score = Score::measure(&sta, library, objective)?;
        report.final_area = base_score.area;
        let guard = ElectricalGuard::Relative;
        let mut electrical = HashMap::new();
        for (i, inst) in current.module.instances.iter().enumerate() {
            for (pin, &violation) in sta.cached_electrical_violations(i).iter().enumerate() {
                let violation = guard.excess(&sta, sta.cell_indices()[i], pin, violation);
                if guard.worsens(violation, Default::default()) {
                    electrical.insert((inst.instance_name, pin), violation);
                }
            }
        }
        let mut winner = None;
        let mut best = None;
        // If a tree was removed, compare the direct connection with replacement
        // trees built from the same flattened circuit. Keep whichever candidate
        // wins after timing and electrical checks.
        let mut evaluate_site = |site: &Site, round: &mut TreeRound| -> Result<()> {
            let mut flat = current.clone();
            let mut flat_owned = owned.clone();
            let removed = if owned.contains_root(site.root) {
                let selected = HashSet::from([site.root]);
                let count = flat_owned.flatten_with_graph(&mut flat, library, &live, &selected)?;
                if flat_owned.contains_root(site.root) {
                    round.groups_with_protected_or_missing_roots += 1;
                    return Ok(());
                }
                count
            } else {
                0
            };
            // Compaction can change instance indices. Search by stable pin
            // identity.
            let flat_graph;
            let graph = if removed > 0 {
                flat_graph =
                    graph::collect_roots(&flat, library, timing.module_output_load, Some(1))?;
                &flat_graph
            } else {
                &live
            };
            let Some(root) = graph.iter().find(|root| root_pin(&flat, root) == site.root) else {
                round.groups_with_protected_or_missing_roots += 1;
                return Ok(());
            };
            let mut price =
                |trial: MappedNetlist, ledger: OwnedBuffers, mut edit: TreeEdit| -> Result<()> {
                    round.ordinary_trials += 1;
                    let Ok(trial_sta) = analyze(&trial) else {
                        round.unsupported_timing += 1;
                        return Ok(());
                    };
                    if !preserves_endpoints(&sta, &trial_sta) {
                        round.rejected_endpoints += 1;
                        return Ok(());
                    }
                    let mut unchanged_violations = 0;
                    for (i, inst) in trial.module.instances.iter().enumerate() {
                        let violations = trial_sta.cached_electrical_violations(i);
                        let cell = trial_sta.cell_indices()[i];
                        for (pin, &raw) in violations.iter().enumerate() {
                            let previous = electrical.get(&(inst.instance_name, pin)).copied();
                            let next = guard.excess(&trial_sta, cell, pin, raw);
                            if guard.worsens(next, previous.unwrap_or_default()) {
                                round.rejected_electrical += 1;
                                return Ok(());
                            }
                            if previous.is_some_and(|old| !guard.worsens(old, next)) {
                                unchanged_violations += 1;
                            }
                        }
                    }
                    let score = Score::measure(&trial_sta, library, objective)?;
                    // Removing a buffer also eliminates any violation on its
                    // pins.
                    let repaired = unchanged_violations < electrical.len();
                    if score.area > policy.limit.max(base_score.area) && !repaired {
                        round.rejected_score += 1;
                        return Ok(());
                    }
                    let Some(rank) = score.rank(base_score, policy.weight) else {
                        round.rejected_score += 1;
                        return Ok(());
                    };
                    if best.is_none_or(|old| rank < old) {
                        best = Some(rank);
                        edit.delay = score.delay;
                        winner = Some((trial, ledger, trial_sta, edit, score.area));
                    }
                    Ok(())
                };
            let named = || {
                let (inst, pin) = site.root;
                vec![format!(
                    "{}.{}",
                    flat.interner.resolve(inst).unwrap(),
                    flat.interner.resolve(pin).unwrap()
                )]
            };
            if removed > 0 {
                price(
                    flat.clone(),
                    flat_owned.clone(),
                    TreeEdit {
                        roots: named(),
                        geometry: "direct",
                        buffer: None,
                        max_leaf_fanout: None,
                        cells_removed: removed,
                        cells_inserted: 0,
                        delay: 0.0,
                    },
                )?;
            }
            for buffer in &choices {
                let mut seen = Vec::new();
                for width in [16, 64] {
                    let Some(branches) = plan::tree(root, buffer, width, timing) else {
                        round.infeasible_geometries += 1;
                        continue;
                    };
                    if seen.contains(&branches) {
                        continue;
                    }
                    let mut trial = flat.clone();
                    let mut ledger = flat_owned.clone();
                    edit::apply_tree_tracked(
                        &mut trial,
                        library,
                        root,
                        buffer,
                        &branches,
                        &mut ledger,
                    )?;
                    let cells_inserted = trial.module.instances.len() - flat.module.instances.len();
                    price(
                        trial,
                        ledger,
                        TreeEdit {
                            roots: named(),
                            geometry: "cap_balanced",
                            buffer: Some(library.cells[buffer.cell].name.clone()),
                            max_leaf_fanout: Some(width),
                            cells_removed: removed,
                            cells_inserted,
                            delay: 0.0,
                        },
                    )?;
                    seen.push(branches);
                }
            }
            Ok(())
        };
        for site in sites.iter().take(2) {
            evaluate_site(site, &mut round)?;
        }
        let Some((next, next_owned, next_sta, edit, area)) = winner else {
            report.rounds.push(round);
            break;
        };
        report.final_delay = edit.delay;
        report.final_area = area;
        current = next;
        owned = next_owned;
        sta = next_sta;
        round.accepted = Some(edit);
        report.rounds.push(round);
    }
    if report.rounds.iter().any(|r| r.accepted.is_some()) {
        let full = measure_boundary_delays_with_library(
            &current.module,
            &current.nets,
            &current.interner,
            &prepared,
            timing,
        )
        .context("verifying registered boundaries after whole-tree replacement")?;
        verify_timing(sta.score(), full)?;
        *netlist = current.into();
        *ownership = owned;
    }
    Ok(report)
}

struct Site {
    root: Pin,
    slack: f64,
    sink_count: usize,
}

fn root_pin(net: &MappedNetlist, root: &graph::Root) -> Pin {
    let inst = &net.module.instances[root.instance];
    (inst.instance_name, inst.connections[root.connection].0)
}

fn critical_trees(
    net: &MappedNetlist,
    sta: &IncrementalSta<'_>,
    owned: &OwnedBuffers,
    roots: &[graph::Root],
    objective: usize,
) -> Result<Vec<Site>> {
    let urgency = sta.pin_timing_urgency(objective)?;
    let mut groups = HashMap::new();
    let mut sites = Vec::<Site>::new();
    for root in roots {
        let pin = root_pin(net, root);
        let owner = owned.root_of(pin.0).unwrap_or(pin);
        if !owned.contains_root(owner) && root.sinks.len() < 2 {
            continue;
        }
        let i = *groups.entry(owner).or_insert_with(|| {
            sites.push(Site {
                root: owner,
                slack: f64::INFINITY,
                sink_count: 0,
            });
            sites.len() - 1
        });
        for sink in &root.sinks {
            let inst = &net.module.instances[sink.instance];
            if owned.root_of(inst.instance_name) == Some(owner) {
                continue;
            }
            if urgency
                .required_arrival(sink.instance, sink.connection)
                .is_none()
            {
                continue;
            }
            let Some(slack) = urgency.sink_slack(sink.instance, sink.connection) else {
                continue;
            };
            sites[i].slack = sites[i].slack.min(slack);
            sites[i].sink_count += 1;
        }
    }
    // Rank an entire owned tree by its most urgent external sink. Counting its
    // own buffers as sinks would let tree depth influence its search priority.
    let delay = sta.score().values()[objective];
    let threshold = (delay * 0.05).max(TIMING_COMPARISON_TOLERANCE);
    sites.retain(|s| s.slack <= threshold && s.sink_count != 0);
    // Backward subtraction leaves roundoff on the critical path. Scale the
    // zero-slack band with the path delay so fanout breaks ties in any unit.
    for site in &mut sites {
        if site.slack.abs() <= delay.abs() * TIMING_COMPARISON_TOLERANCE {
            site.slack = 0.0;
        }
    }
    sites.sort_by(|a, b| {
        a.slack
            .total_cmp(&b.slack)
            .then(b.sink_count.cmp(&a.sink_count))
    });
    Ok(sites)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::io::read_gv_from_str;

    // All three drivers lie on the same critical path. The first two each
    // feed two loads. The last feeds 32 and needs one of the two trial slots.
    #[test]
    fn refinement_considers_the_largest_fanout_on_a_critical_path() -> Result<()> {
        let lib = super::super::tests::library()?;
        let mut gv = String::from(
            "module t(input clk, input d, output [33:0] y);
             wire q, u, v;
             DFF1 launch(.CLK(clk), .D(d), .Q(q));
             INV1 first(.A(q), .Y(u));
             INV1 last(.A(u), .Y(v));
             DFF1 tap_q(.CLK(clk), .D(q), .Q(y[32]));
             DFF1 tap_u(.CLK(clk), .D(u), .Q(y[33]));",
        );
        for i in 0..32 {
            gv.push_str(&format!("DFF1 cap{i}(.CLK(clk), .D(v), .Q(y[{i}]));"));
        }
        gv.push_str("endmodule");
        let report = refine_registered_trees(
            &mut read_gv_from_str(&gv)?,
            &lib,
            StaOptions::default(),
            &mut OwnedBuffers::default(),
            FanoutAreaPolicy::UNBOUNDED,
        )?;
        assert!(report.final_delay < report.initial_delay);
        Ok(())
    }

    // The launch Q initially drives a chain of three buffers, whose last
    // buffer drives two data pins. Removing the entire chain restores two
    // legal direct connections and improves both delay and area.
    #[test]
    fn remove_an_entire_legal_tree_at_a_one_sink_launch() -> Result<()> {
        let lib = super::super::tests::library()?;
        let parsed = read_gv_from_str(
            "module t(input clk,input d,output [1:0] y); wire q,u,v; DFF1 launch(.CLK(clk),.D(d),.Q(q)); INV1 a(.A(q),.Y(u)); INV1 b(.A(q),.Y(v)); DFF1 ca(.CLK(clk),.D(u),.Q(y[0])); DFF1 cb(.CLK(clk),.D(v),.Q(y[1])); endmodule",
        )?;
        let mut net = MappedNetlist {
            module: parsed.modules[0].clone(),
            nets: parsed.nets,
            interner: parsed.interner,
        };
        let roots = graph::collect_roots(&net, &lib, 0.0, Some(1))?;
        let root = roots.iter().find(|r| r.instance_name == "launch").unwrap();
        let driver_wire = root.source.clone();
        let buf = graph::buffers(&lib, &PreparedStaLibrary::new(&lib)?)
            .into_iter()
            .find(|b| lib.cells[b.cell].name == "BUF4")
            .unwrap();
        let mut owned = OwnedBuffers::default();
        edit::apply_tree_tracked(
            &mut net,
            &lib,
            root,
            &buf,
            &[
                edit::Branch {
                    parent: None,
                    sinks: vec![],
                },
                edit::Branch {
                    parent: Some(0),
                    sinks: vec![],
                },
                edit::Branch {
                    parent: Some(1),
                    sinks: vec![0, 1],
                },
            ],
            &mut owned,
        )?;
        let mut parsed = ParsedNetlist {
            modules: vec![net.module],
            nets: net.nets,
            interner: net.interner,
        };
        let r = refine_registered_trees(
            &mut parsed,
            &lib,
            StaOptions::default(),
            &mut owned,
            FanoutAreaPolicy::UNBOUNDED,
        )?;
        let selected = r
            .rounds
            .iter()
            .find_map(|r| r.accepted.as_ref())
            .expect("redundant three-relay chain adds timing and area cost");
        assert_eq!(selected.geometry, "direct");
        assert_eq!(selected.cells_removed, 3);
        assert!(owned.is_empty());
        assert!(r.final_delay < r.initial_delay);
        for inst in parsed.modules[0]
            .instances
            .iter()
            .filter(|i| parsed.interner.resolve(i.type_name) == Some("INV1"))
        {
            let pin = inst
                .connections
                .iter()
                .find(|(pin, _)| parsed.interner.resolve(*pin) == Some("A"))
                .unwrap();
            assert_eq!(pin.1, driver_wire);
        }
        Ok(())
    }
}
