// SPDX-License-Identifier: Apache-2.0

use super::TimingCones;
use crate::netlist::registered_fanout::graph::collect_roots;
use crate::netlist::registered_fanout::tests::read_mapped_netlist;
use crate::netlist::resize::tests::library;
use crate::netlist::sta::incremental::IncrementalSta;
use crate::netlist::sta::{PreparedStaLibrary, StaOptions};
use anyhow::Result;
use std::fmt::Write;
use std::sync::Arc;

/// Time the fixture and nominate the named candidates in both electrical
/// ranking orders. Return driver names so assertions describe the circuit.
fn nominated_with_unavailable_plans(
    source: &str,
    unavailable: &[&str],
) -> Result<[Vec<String>; 2]> {
    let library = library()?;
    let netlist = read_mapped_netlist(source)?;
    let options = StaOptions::default();
    let sta = IncrementalSta::with_registered_library(
        &netlist.module,
        &netlist.nets,
        &netlist.interner,
        Arc::new(PreparedStaLibrary::new(&library)?),
        options,
    )?;
    let roots = collect_roots(&netlist, &library, options.module_output_load, Some(2))?;
    let mut ranked: Vec<_> = roots
        .iter()
        .enumerate()
        .filter(|(_, root)| root.instance_name.starts_with("candidate_"))
        .map(|(i, _)| i)
        .collect();
    let selected = |ranked: &[usize]| -> Result<_> {
        Ok(TimingCones::prepare(&sta, &roots, ranked, 3)?
            .expect("eligible roots")
            .groups(|id| !unavailable.contains(&roots[id].instance_name.as_str())))
    };
    let groups = selected(&ranked)?;
    ranked.reverse();
    assert_eq!(groups, selected(&ranked)?);
    for group in &groups {
        assert!(group.windows(2).all(|pair| pair[0] < pair[1]));
    }
    Ok(groups.map(|group| {
        let mut names: Vec<_> = group
            .into_iter()
            .map(|i| roots[i].instance_name.clone())
            .collect();
        names.sort();
        names
    }))
}

#[test]
fn one_capture_can_require_more_than_sixty_four_launch_drivers() -> Result<()> {
    const LANES: usize = 128;
    let mut source = String::from("module top(input clk, input d, output y, output early);\n");
    // Every lane crosses seven identical logic levels into the same capture.
    // The additional launch supplies a disjoint, faster register-to-register
    // path.
    for i in 0..LANES {
        let leaf = LANES + i;
        writeln!(
            source,
            "wire n{leaf}, tap_q{i}; DFF1 candidate_lane_{i}(.CLK(clk),.D(d),.Q(n{leaf}));
             DFF1 tap_{i}(.CLK(clk),.D(n{leaf}),.Q(tap_q{i}));"
        )?;
    }
    for i in (1..LANES).rev() {
        writeln!(
            source,
            "wire n{i}; AND1 merge_{i}(.A(n{}),.B(n{}),.Y(n{i}));",
            2 * i,
            2 * i + 1
        )?;
    }
    source.push_str(
        r#"DFF1 capture(.CLK(clk),.D(n1),.Q(y));
        wire fast,fast_copy;
        DFF1 candidate_early(.CLK(clk),.D(d),.Q(fast));
        DFF1 early_capture(.CLK(clk),.D(fast),.Q(early));
        DFF1 early_copy(.CLK(clk),.D(fast),.Q(fast_copy));
        endmodule"#,
    );
    let [tight, wider] = nominated_with_unavailable_plans(&source, &[])?;
    let mut expected: Vec<_> = (0..LANES).map(|i| format!("candidate_lane_{i}")).collect();
    expected.sort();
    assert_eq!(tight, expected);
    assert!(wider.is_empty() || wider == expected);
    Ok(())
}

#[test]
fn widening_includes_an_internal_driver_on_a_competing_arc_to_the_same_capture() -> Result<()> {
    let mut source = String::from("module top(input clk, input d, output y);\n");
    // The first two paths tie. The third is one gate shorter and has a
    // candidate on a combinational output. The fourth is far from critical.
    for (lane, depth) in [32, 32, 31, 1].into_iter().enumerate() {
        let name = ["candidate_a", "candidate_b", "launch_c", "candidate_far"][lane];
        writeln!(
            source,
            "wire n{lane}_0; DFF1 {name}(.CLK(clk),.D(d),.Q(n{lane}_0));"
        )?;
        for stage in 1..=depth {
            let buffer = if lane == 2 && stage == 1 {
                String::from("candidate_near")
            } else {
                format!("buf_{lane}_{stage}")
            };
            writeln!(
                source,
                "wire n{lane}_{stage}; BUF1 {buffer}(.A(n{lane}_{}),.Y(n{lane}_{stage}));",
                stage - 1
            )?;
        }
        let tap_stage = usize::from(lane == 2);
        writeln!(
            source,
            "wire t{lane}; DFF1 tap_{lane}(.CLK(clk),.D(n{lane}_{tap_stage}),.Q(t{lane}));"
        )?;
    }
    source.push_str(
        r#"wire ml,mr,z;
        AND1 left(.A(n0_32),.B(n1_32),.Y(ml));
        AND1 right(.A(n2_31),.B(n3_1),.Y(mr));
        AND1 finish(.A(ml),.B(mr),.Y(z));
        DFF1 capture(.CLK(clk),.D(z),.Q(y));
        endmodule"#,
    );
    let [tight, wider] = nominated_with_unavailable_plans(&source, &[])?;
    assert_eq!(tight, ["candidate_a", "candidate_b"]);
    assert_eq!(wider, ["candidate_a", "candidate_b", "candidate_near"]);
    Ok(())
}

#[test]
fn an_unplannable_downstream_driver_does_not_hide_a_launch_repair() -> Result<()> {
    // Two identical paths tie at the same capture. Both a launch and a
    // downstream driver have two movable sinks on each path.
    let source = r#"module top(input clk, input d, output y);
        wire qa,qb,da,db,z,aq,bq,ad,bd;
        DFF1 candidate_launch_a(.CLK(clk),.D(d),.Q(qa));
        DFF1 candidate_launch_b(.CLK(clk),.D(d),.Q(qb));
        DFF1 tap_a(.CLK(clk),.D(qa),.Q(aq)); DFF1 tap_b(.CLK(clk),.D(qb),.Q(bq));
        BUF1 candidate_downstream_a(.A(qa),.Y(da));
        BUF1 candidate_downstream_b(.A(qb),.Y(db));
        DFF1 tap_da(.CLK(clk),.D(da),.Q(ad)); DFF1 tap_db(.CLK(clk),.D(db),.Q(bd));
        AND1 merge(.A(da),.B(db),.Y(z)); DFF1 capture(.CLK(clk),.D(z),.Q(y));
        endmodule"#;
    for (unavailable, expected) in [
        (vec![], ["candidate_downstream_a", "candidate_downstream_b"]),
        (
            vec!["candidate_downstream_a"],
            ["candidate_downstream_b", "candidate_launch_a"],
        ),
        (
            vec!["candidate_downstream_a", "candidate_downstream_b"],
            ["candidate_launch_a", "candidate_launch_b"],
        ),
    ] {
        let [tight, _] = nominated_with_unavailable_plans(source, &unavailable)?;
        assert_eq!(tight, expected);
    }
    Ok(())
}
