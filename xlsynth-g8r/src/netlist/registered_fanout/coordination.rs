// SPDX-License-Identifier: Apache-2.0

//! Find editable drivers upstream of critical timing endpoints.

use crate::netlist::sta::incremental::{IncrementalSta, TimingEndpoint};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Map a timed sink pin to its unique editable driver on this snapshot.
pub(super) fn editable_sites<'a>(
    sites: impl IntoIterator<Item = (usize, usize, &'a str)>,
) -> Option<HashMap<(usize, &'a str), usize>> {
    let mut owners = HashMap::new();
    for (root, sink, pin) in sites {
        if owners
            .insert((sink, pin), root)
            .is_some_and(|old| old != root)
        {
            // Overlapping edits cannot be composed on this graph.
            return None;
        }
    }
    let mut ids = owners.values();
    let first = ids.next()?;
    ids.any(|i| i != first).then_some(owners)
}

/// Walk backward from each critical endpoint. On each timing path, stop at
/// the nearest driver for which the requested buffer tree can be built.
/// Visit shared logic once and return the selected drivers as one proposal.
/// Improving only one of two equally slow paths may leave stage delay flat.
///
/// ```text
/// launch Q1 ---> AND/A
/// launch Q2 ---> AND/B     AND/Y ---> capture D
/// ```
pub(super) fn select_endpoint_roots(
    sta: &IncrementalSta<'_>,
    boundary: usize,
    owners: &HashMap<(usize, &str), usize>,
    window: f64,
    follows: impl Fn(usize, &str) -> bool,
    mut feasible: impl FnMut(usize) -> bool,
) -> Vec<usize> {
    let Some((_, worst)) = sta.endpoint_arrivals(boundary).next_back() else {
        return Vec::new();
    };
    let mut pending: Vec<_> = sta
        .endpoint_arrivals(boundary)
        .rev()
        .take_while(|(_, delay)| worst - delay <= window)
        .map(|(endpoint, _)| endpoint)
        .collect();
    // A given register can be both a capture (follow its D pin) and a launch
    // (stop at its Q). Bit and capture IDs belong to disjoint ranges.
    let key = |endpoint| match endpoint {
        TimingEndpoint::Output(bit) => bit,
        TimingEndpoint::Capture(instance) => sta.bit_count() + instance,
    };
    let mut visited = HashSet::new();
    let mut selected = BTreeSet::new();
    while let Some(node) = pending.pop() {
        let id = key(node);
        if !visited.insert(id) {
            continue;
        }
        sta.visit_timed_fanins(node, boundary / 2, |sink, pin, bit| {
            if !follows(sink, pin) {
                return;
            }
            if let Some(&tree) = owners.get(&(sink, pin))
                && feasible(tree)
            {
                selected.insert(tree);
            } else {
                pending.push(TimingEndpoint::Output(bit));
            }
        });
    }
    if selected.len() < 2 {
        return Vec::new();
    }
    selected.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::io::read_gv_from_str;
    use crate::netlist::resize::tests::library;
    use crate::netlist::sta::{PreparedStaLibrary, StaOptions};
    use anyhow::Result;
    use std::fmt::Write;
    use std::sync::Arc;

    use crate::netlist::sta::incremental::TIMING_COMPARISON_TOLERANCE;

    fn eligible_endpoint_roots<'a>(
        sta: &IncrementalSta<'_>,
        boundary: usize,
        sites: impl IntoIterator<Item = (usize, usize, &'a str)>,
    ) -> Vec<usize> {
        let Some(owners) = editable_sites(sites) else {
            return Vec::new();
        };
        select_endpoint_roots(
            sta,
            boundary,
            &owners,
            TIMING_COMPARISON_TOLERANCE,
            |_, _| true,
            |_| true,
        )
    }

    #[test]
    fn one_capture_can_need_two_independent_edits() -> Result<()> {
        let lib = library()?;
        let prepared = Arc::new(PreparedStaLibrary::new(&lib)?);
        let source = r#"module top(input clk, input d, output y);
            wire qa, qb, both;
            DFF1 la(.CLK(clk),.D(d),.Q(qa)); DFF1 lb(.CLK(clk),.D(d),.Q(qb));
            AND1 gate(.A(qa),.B(qb),.Y(both)); DFF1 capture(.CLK(clk),.D(both),.Q(y));
            endmodule"#;
        let net = read_gv_from_str(source)?;
        let sta = IncrementalSta::with_registered_library(
            &net.modules[0],
            &net.nets,
            &net.interner,
            prepared,
            StaOptions::default(),
        )?;
        let id = |name: &str| {
            (0..sta.instance_count())
                .find(|&i| sta.instance(i).unwrap().name == name)
                .unwrap()
        };
        let gate = id("gate");
        let candidates = [(13, gate, "A"), (2, gate, "B"), (77, id("la"), "D")];
        assert_eq!(eligible_endpoint_roots(&sta, 3, candidates), [2, 13]);
        // A replacement directly at the capture takes priority over both
        // deeper edits on this cone, leaving only a single-root proposal.
        assert!(
            eligible_endpoint_roots(
                &sta,
                3,
                candidates.into_iter().chain([(5, id("capture"), "D")]),
            )
            .is_empty()
        );
        assert!(
            eligible_endpoint_roots(&sta, 3, candidates.into_iter().chain([(34, gate, "A")]),)
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn independent_tied_captures_only_improve_together() -> Result<()> {
        // Forty identical captures exceed the endpoint sampling used by the
        // cell resizer. Omitting even one keeps the stage maximum unchanged.
        const LANES: usize = 40;
        let lib = library()?;
        let prepared = Arc::new(PreparedStaLibrary::new(&lib)?);
        let make = |strong: usize| -> Result<_> {
            let mut src = format!(
                "module top(input clk, input d, output [{}:0] y);\n",
                LANES - 1
            );
            for i in 0..LANES {
                let driver = if i < strong { "DFF4" } else { "DFF1" };
                writeln!(
                    src,
                    "wire q{i}; {driver} l{i}(.CLK(clk),.D(d),.Q(q{i})); DFF1 c{i}(.CLK(clk),.D(q{i}),.Q(y[{i}]));"
                )?;
            }
            src.push_str("endmodule");
            Ok(read_gv_from_str(&src)?)
        };
        let net = make(0)?;
        let sta = IncrementalSta::with_registered_library(
            &net.modules[0],
            &net.nets,
            &net.interner,
            prepared.clone(),
            StaOptions::default(),
        )?;
        assert_eq!(sta.endpoint_maximum_count(3), LANES);
        let candidates: Vec<_> = (0..sta.instance_count())
            .filter_map(|i| {
                let lane: usize = sta.instance(i)?.name.strip_prefix('c')?.parse().ok()?;
                Some((lane, i, "D"))
            })
            .collect();
        assert_eq!(
            eligible_endpoint_roots(&sta, 3, candidates.iter().copied()),
            (0..LANES).collect::<Vec<_>>()
        );
        assert_eq!(
            eligible_endpoint_roots(&sta, 3, candidates.iter().skip(1).copied()),
            (1..LANES).collect::<Vec<_>>()
        );
        let worst = sta.endpoint_arrivals(3).next_back().unwrap().1;
        for (strong, expected_improvement) in [(1, false), (LANES - 1, false), (LANES, true)] {
            let variant = make(strong)?;
            let trial = IncrementalSta::with_registered_library(
                &variant.modules[0],
                &variant.nets,
                &variant.interner,
                prepared.clone(),
                StaOptions::default(),
            )?;
            let new_worst = trial.endpoint_arrivals(3).next_back().unwrap().1;
            if expected_improvement {
                assert!(new_worst < worst);
            } else {
                assert_eq!(new_worst, worst);
            }
        }
        Ok(())
    }
}
