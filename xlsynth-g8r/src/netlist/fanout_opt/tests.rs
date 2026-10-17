// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::aig_sim::gate_sim::{Collect, eval};
#[cfg(feature = "has-bitwuzla")]
use crate::check_equivalence::{IrCheckResult, prove_same_gate_fn_via_ir_status_with_solver};
use crate::liberty_model::{Cell, LibraryBuilder, LuTableTemplate};
use crate::liberty_proto::TimingTableKind;
use crate::netlist::emit::emit_module_as_netlist_text;
use crate::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
use crate::netlist::io::read_gv_from_str;
use std::collections::HashSet;
use xlsynth_pir::IrBits;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::SolverChoice;

/// Compact test cell with independently controlled input load and output
/// resistance.
struct Spec<'a> {
    name: &'a str,
    function: &'a str,
    area: f64,
    cap: f64,
    base: f64,
    resistance: f64,
    output: &'a str,
}
impl<'a> Spec<'a> {
    /// Provides explicit electrical parameters without drive-strength name
    /// assumptions.
    fn new(name: &'a str, function: &'a str, cap: f64, resistance: f64) -> Self {
        Self {
            name,
            function,
            area: 1.0,
            cap,
            base: 1.0,
            resistance,
            output: "Y",
        }
    }
}

/// Builds 2D Liberty tables whose output delay includes load and incoming slew.
fn library(specs: &[Spec]) -> Library {
    let mut b = LibraryBuilder::new();
    b.lu_table_templates = vec![LuTableTemplate {
        name: "lut".into(),
        kind: "lu_table_template".into(),
        variable_1: "input_net_transition".into(),
        variable_2: "total_output_net_capacitance".into(),
        index_1: vec![0.01, 1000.0],
        index_2: vec![0.01, 1000.0],
        ..Default::default()
    }];
    for spec in specs {
        let mut inputs = parse_formula(spec.function).unwrap().inputs();
        inputs.sort();
        inputs.dedup();
        let mut pins = Vec::new();
        let mut arcs = Vec::new();
        for input in &inputs {
            pins.push(Pin {
                name: b.intern_string(input).unwrap(),
                direction: PinDirection::Input as i32,
                capacitance: Some(spec.cap),
                ..Default::default()
            });
            let mut tables = Vec::new();
            for kind in [
                TimingTableKind::CellRise,
                TimingTableKind::CellFall,
                TimingTableKind::RiseTransition,
                TimingTableKind::FallTransition,
            ] {
                let mut values = Vec::new();
                for slew in [0.01, 1000.0] {
                    for load in [0.01, 1000.0] {
                        values.push(
                            if matches!(kind, TimingTableKind::CellRise | TimingTableKind::CellFall)
                            {
                                spec.base + spec.resistance * load + 0.05 * slew
                            } else {
                                1.0 + 0.01 * load + 0.01 * slew
                            },
                        );
                    }
                }
                tables.push(
                    b.add_timing_table_f64(kind, 1, vec![], vec![], vec![], values, vec![2, 2], "")
                        .unwrap(),
                );
            }
            arcs.push(
                b.add_timing_arc(
                    input,
                    if spec.function.starts_with('!') && inputs.len() == 1 {
                        "negative_unate"
                    } else if inputs.len() == 1 {
                        "positive_unate"
                    } else {
                        "non_unate"
                    },
                    "combinational",
                    "",
                    tables,
                )
                .unwrap(),
            );
        }
        pins.push(Pin {
            name: b.intern_string(spec.output).unwrap(),
            direction: PinDirection::Output as i32,
            function: b.intern_string(spec.function).unwrap(),
            timing_arcs: arcs,
            ..Default::default()
        });
        b.cells.push(Cell {
            name: spec.name.into(),
            area: spec.area,
            pins,
            ..Default::default()
        });
    }
    b.finish()
}

/// Parses an owned scalar module for trial edits.
fn parsed(text: &str) -> MappedNetlist {
    let mut p = read_gv_from_str(text).unwrap();
    MappedNetlist {
        module: p.modules.remove(0),
        nets: p.nets,
        interner: p.interner,
    }
}

/// Creates a loaded inverter, optionally behind a weak upstream driver.
fn circuit(count: usize, upstream: bool, aliases: bool) -> MappedNetlist {
    let outputs = (0..count).map(|i| format!("o{i}")).collect::<Vec<_>>();
    let mut s = format!(
        "module top(a, {}{}); input a; output {}; wire n;\n",
        outputs.join(","),
        if aliases {
            ",alias1,alias2,constant"
        } else {
            ""
        },
        outputs.join(",")
    );
    if upstream {
        s.push_str("wire p; PRE pre(.A(a),.Y(p)); DRIVE drv(.A(p),.Y(n));\n");
    } else {
        s.push_str("DRIVE drv(.A(a),.Y(n));\n");
    }
    for i in 0..count {
        s.push_str(&format!("LOAD sink{i}(.A(n),.Y(o{i}));\n"));
    }
    if aliases {
        s.push_str("output alias1,alias2,constant; assign alias1=n; assign alias2=alias1; assign constant=1'b1;\n");
    }
    s.push_str("endmodule");
    parsed(&s)
}

/// Projects serialized edits and checks all inputs, with Bitwuzla when enabled.
fn equivalent(a: &MappedNetlist, b: &MappedNetlist, lib: &Library) {
    let to_aig = |n: &MappedNetlist| {
        let text = emit_module_as_netlist_text(&n.module, &n.nets, &n.interner).unwrap();
        let p = read_gv_from_str(&text).unwrap();
        project_gatefn_from_netlist_and_liberty(
            &p.modules[0],
            &p.nets,
            &p.interner,
            lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap()
    };
    let x = to_aig(a);
    let y = to_aig(b);
    for v in 0..(1usize << x.inputs.len()) {
        let input = (0..x.inputs.len())
            .map(|i| IrBits::make_ubits(1, ((v >> i) & 1) as u64).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            eval(&x, &input, Collect::None).outputs,
            eval(&y, &input, Collect::None).outputs
        );
    }
    #[cfg(feature = "has-bitwuzla")]
    assert_eq!(
        prove_same_gate_fn_via_ir_status_with_solver(&x, &y, SolverChoice::Bitwuzla, None),
        IrCheckResult::Equivalent
    );
}

/// Gives synthetic cases enough area headroom to exercise nontrivial
/// candidates.
fn options() -> FanoutOptimizationOptions {
    FanoutOptimizationOptions {
        full_sta: false,
        verify_incremental: true,
        max_rounds: 2,
        drivers_per_round: 1,
        fanout_trigger: 4,
        area_growth_limit: 2.0,
        area_weight: 0.0,
        primary_input_transition: 1.0,
        module_output_load: 1.0,
    }
}

#[test]
fn buffer_and_two_level_trees_preserve_polarity_constants_and_shared_outputs() {
    let lib = library(&[
        Spec::new("DRIVE", "!A", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
        Spec::new("BUF", "A", 0.25, 0.5),
    ]);
    let original = circuit(16, false, true);
    let opts = options();
    let (before, _, g) = measure(&original, &lib, &opts).unwrap();
    let d = g.drivers.iter().find(|d| d.sinks.len() == 16).unwrap();
    assert_eq!(d.output_ports, 2);
    assert_eq!(d.fixed_output_load, 2.0);
    let cells = choices(&lib).unwrap();
    for levels in [1, 2] {
        let mut changed = original.clone();
        apply(
            &mut changed,
            d,
            &FanoutEdit::Buffer {
                cell: "BUF".into(),
                groups: 4,
                levels,
            },
            &cells,
        )
        .unwrap();
        let (after, _, after_g) = measure(&changed, &lib, &opts).unwrap();
        assert!(after.worst_delay < before.worst_delay);
        assert_eq!(
            after_g
                .drivers
                .iter()
                .find(|x| x.instance == d.instance)
                .unwrap()
                .output_ports,
            2
        );
        equivalent(&original, &changed, &lib);
    }
}

#[test]
fn replication_balances_capacitance_and_keeps_shared_outputs_on_original() {
    let lib = library(&[
        Spec::new("DRIVE", "!A", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
    ]);
    let original = circuit(16, false, true);
    let (before, _, g) = measure(&original, &lib, &options()).unwrap();
    let d = g.drivers.iter().find(|d| d.sinks.len() == 16).unwrap();
    let mut changed = original.clone();
    apply(
        &mut changed,
        d,
        &FanoutEdit::Replicate { groups: 4 },
        &choices(&lib).unwrap(),
    )
    .unwrap();
    let (after, _, _) = measure(&changed, &lib, &options()).unwrap();
    assert_eq!(after.cell_count, before.cell_count + 3);
    assert!(after.worst_delay < before.worst_delay);
    equivalent(&original, &changed, &lib);
}

#[test]
fn rejection_accounts_for_replication_loading_upstream() {
    let mut load = Spec::new("LOAD", "A", 1.0, 0.1);
    load.base = 1000.0;
    let lib = library(&[
        Spec::new("PRE", "A", 0.1, 100.0),
        Spec::new("DRIVE", "!A", 1.0, 2.0),
        load,
    ]);
    let original = circuit(16, true, false);
    let (before, sta, g) = measure(&original, &lib, &options()).unwrap();
    let d = g.drivers.iter().find(|d| d.sinks.len() == 16).unwrap();
    let mut changed = original.clone();
    apply(
        &mut changed,
        d,
        &FanoutEdit::Replicate { groups: 4 },
        &choices(&lib).unwrap(),
    )
    .unwrap();
    let (after, after_sta, _) = measure(&changed, &lib, &options()).unwrap();
    let upstream = g.drivers.iter().find(|d| d.instance == 0).unwrap().net;
    assert!(
        after_sta
            .timing_for_net(NetIndex(upstream))
            .unwrap()
            .rise
            .arrival
            > sta.timing_for_net(NetIndex(upstream)).unwrap().rise.arrival + 200.0
    );
    assert!(after.worst_delay > before.worst_delay);
    let mut repaired = original.clone();
    let report = optimize_fanout(&mut repaired, &lib, options()).unwrap();
    assert!(report.trials.iter().any(|t| {
        matches!(t.kind, FanoutEdit::Replicate { .. })
            && t.metrics
                .as_ref()
                .is_some_and(|m| m.worst_delay > before.worst_delay)
    }));
    assert!(report.decisions.is_empty());
    equivalent(&original, &repaired, &lib);
}

#[test]
fn rejection_accounts_for_upsizing_loading_upstream_and_output_renaming() {
    let mut strong = Spec::new("STRONG", "!A", 20.0, 0.1);
    strong.area = 2.0;
    strong.output = "Z";
    let mut load = Spec::new("LOAD", "A", 1.0, 0.1);
    load.base = 1000.0;
    let lib = library(&[
        Spec::new("PRE", "A", 0.1, 100.0),
        Spec::new("DRIVE", "!A", 1.0, 2.0),
        strong,
        load,
    ]);
    let original = circuit(16, true, false);
    let (before, _, g) = measure(&original, &lib, &options()).unwrap();
    let d = g.drivers.iter().find(|d| d.sinks.len() == 16).unwrap();
    let mut changed = original.clone();
    apply(
        &mut changed,
        d,
        &FanoutEdit::Upsize {
            cell: "STRONG".into(),
        },
        &choices(&lib).unwrap(),
    )
    .unwrap();
    let (after, _, _) = measure(&changed, &lib, &options()).unwrap();
    assert!(after.worst_delay > before.worst_delay + 1000.0);
    equivalent(&original, &changed, &lib);
    let mut repaired = original.clone();
    let r = optimize_fanout(&mut repaired, &lib, options()).unwrap();
    assert!(r.decisions.is_empty());
    assert!(
        r.trials
            .iter()
            .any(|t| matches!(t.kind, FanoutEdit::Upsize { .. })
                && t.reason == "no global delay/area-score improvement")
    );
}

#[test]
fn asymmetric_function_matching_is_by_named_pin_truth_not_cell_name() {
    let mut good = Spec::new("GOOD", "A & !B", 1.0, 0.1);
    good.area = 2.0;
    let mut wrong = Spec::new("WRONG", "!A & B", 1.0, 0.1);
    wrong.area = 2.0;
    let lib = library(&[Spec::new("DRIVE", "A & !B", 1.0, 10.0), good, wrong]);
    let cells = choices(&lib).unwrap();
    let old = cells.iter().find(|c| c.name == "DRIVE").unwrap();
    let d = Driver {
        instance: 0,
        output_connection: 2,
        net: 0,
        sinks: vec![],
        load: [2.0; 2],
        output_ports: 1,
        fixed_output_load: 1.0,
        slew_bound: 1000.0,
    };
    let c = candidates(&d, old, &cells);
    assert_eq!(c.len(), 1);
    assert!(matches!(&c[0],FanoutEdit::Upsize {cell} if cell=="GOOD"));
}

#[test]
fn no_op_area_rejection_and_determinism() {
    let lib = library(&[
        Spec::new("DRIVE", "!A", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
    ]);
    let original = circuit(16, false, false);
    let mut blocked = original.clone();
    let mut opts = options();
    opts.area_growth_limit = 0.0;
    let r = optimize_fanout(&mut blocked, &lib, opts).unwrap();
    assert!(r.decisions.is_empty());
    assert_eq!(r.before.worst_delay, r.after.worst_delay);
    assert!(r.trials.iter().all(|t| t.reason == "area budget"));
    let mut a = original.clone();
    let mut b = original.clone();
    let ra = optimize_fanout(&mut a, &lib, options()).unwrap();
    let rb = optimize_fanout(&mut b, &lib, options()).unwrap();
    assert!(!ra.decisions.is_empty());
    assert!(ra.after.worst_delay < ra.before.worst_delay);
    assert_eq!(
        serde_json::to_value(&ra.decisions).unwrap(),
        serde_json::to_value(&rb.decisions).unwrap()
    );
    assert_eq!(
        emit_module_as_netlist_text(&a.module, &a.nets, &a.interner).unwrap(),
        emit_module_as_netlist_text(&b.module, &b.nets, &b.interner).unwrap()
    );
    equivalent(&original, &a, &lib);
}

#[test]
fn scalar_scope_and_alias_cycles_fail_closed() {
    let lib = library(&[Spec::new("LOAD", "A", 1.0, 1.0)]);
    let mut vector =
        parsed("module top(a,y); input [1:0] a; output [1:0] y; assign y=a; endmodule");
    assert!(optimize_fanout(&mut vector, &lib, options()).is_err());
    let mut cyclic =
        parsed("module top(a,y); input a; output y; wire n; assign y=n; assign n=y; endmodule");
    assert!(optimize_fanout(&mut cyclic, &lib, options()).is_err());
}

#[test]
fn multiple_output_cell_is_never_replicated() {
    let base = library(&[
        Spec::new("DUAL", "A", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
    ]);
    let mut b = LibraryBuilder::from_library(base);
    let z = b.intern_string("Z").unwrap();
    let mut extra = b.cells[0].pins[1].clone();
    extra.name = z;
    b.cells[0].pins.push(extra);
    let lib = b.finish();
    assert!(choices(&lib).unwrap().iter().all(|c| c.name != "DUAL"));
    let mut s = String::from(
        "module top(a,z,o0,o1,o2,o3); input a; output z,o0,o1,o2,o3; wire n; DUAL d(.A(a),.Y(n),.Z(z));",
    );
    for i in 0..4 {
        s.push_str(&format!("LOAD s{i}(.A(n),.Y(o{i}));"));
    }
    s.push_str("endmodule");
    let mut n = parsed(&s);
    let original = n.clone();
    let r = optimize_fanout(&mut n, &lib, options()).unwrap();
    assert!(r.decisions.is_empty());
    equivalent(&original, &n, &lib);
}

#[test]
fn constant_input_and_name_collisions_survive_replication() {
    let lib = library(&[
        Spec::new("DRIVE", "A & !B", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
    ]);
    let mut s = String::from(
        "module top(a,o0,o1,o2,o3); input a; output o0,o1,o2,o3; wire n,fanout_opt_15; DRIVE drv(.A(a),.B(1'b0),.Y(n));",
    );
    for i in 0..4 {
        s.push_str(&format!("LOAD s{i}(.A(n),.Y(o{i}));"));
    }
    s.push_str("endmodule");
    let original = parsed(&s);
    let (_, _, g) = measure(&original, &lib, &options()).unwrap();
    let mut n = original.clone();
    apply(
        &mut n,
        &g.drivers[0],
        &FanoutEdit::Replicate { groups: 2 },
        &choices(&lib).unwrap(),
    )
    .unwrap();
    equivalent(&original, &n, &lib);
}

#[test]
fn conditional_arcs_remain_pruned_after_replication_with_constant_input() {
    let base = library(&[
        Spec::new("DRIVE", "A & !B", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
    ]);
    let mut b = LibraryBuilder::from_library(base);
    let active = b.intern_string("!B").unwrap();
    let inactive = b.intern_string("B").unwrap();
    b.cells[0].pins[2].timing_arcs[0].when = active;
    let mut impossible = b.cells[0].pins[2].timing_arcs[0].clone();
    impossible.when = inactive;
    impossible.tables = [
        TimingTableKind::CellRise,
        TimingTableKind::CellFall,
        TimingTableKind::RiseTransition,
        TimingTableKind::FallTransition,
    ]
    .into_iter()
    .map(|kind| {
        b.add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![10000.0], vec![], "")
            .unwrap()
    })
    .collect();
    b.cells[0].pins[2].timing_arcs.push(impossible);
    let lib = b.finish();
    let original = parsed(
        "module top(a,o0,o1,o2,o3); input a; output o0,o1,o2,o3; wire n; DRIVE d(.A(a),.B(1'b0),.Y(n)); LOAD s0(.A(n),.Y(o0)); LOAD s1(.A(n),.Y(o1)); LOAD s2(.A(n),.Y(o2)); LOAD s3(.A(n),.Y(o3)); endmodule",
    );
    check_incremental_candidates(&original, &lib);
    let (before, _, g) = measure(&original, &lib, &options()).unwrap();
    assert!(before.worst_delay < 100.0);
    let mut changed = original.clone();
    apply(
        &mut changed,
        &g.drivers[0],
        &FanoutEdit::Replicate { groups: 2 },
        &choices(&lib).unwrap(),
    )
    .unwrap();
    let (after, _, _) = measure(&changed, &lib, &options()).unwrap();
    assert!(after.worst_delay < before.worst_delay);
    equivalent(&original, &changed, &lib);
}

/// Exercises every enumerated topology, regardless of area/objective rejection,
/// and proves rollback leaves the initial graph usable for the next trial.
fn check_incremental_candidates(original: &MappedNetlist, lib: &Library) {
    let opts = options();
    let cells = choices(lib).unwrap();
    let (baseline, sta, g) = measure(original, lib, &opts).unwrap();
    let mut engine = incremental::Engine::new(original, lib, &opts, &cells, &sta).unwrap();
    let mut tried = 0;
    for d in &g.drivers {
        let name = resolve_symbol(
            &original.interner,
            original.module.instances[d.instance].type_name,
            "cell",
        )
        .unwrap();
        let Some(old) = cells.iter().find(|c| c.name == name) else {
            continue;
        };
        for kind in candidates(d, old, &cells) {
            let edit = engine.edit(d, &kind, &cells).unwrap();
            let mut changed = original.clone();
            apply(&mut changed, d, &kind, &cells).unwrap();
            let (m, full, full_g) = measure(&changed, lib, &opts).unwrap();
            let (actual, _) = engine
                .trial(edit, |e, actual| {
                    e.check(&full)?;
                    e.check_loads(&full_g)?;
                    ensure!(*actual == m, "trial metrics");
                    Ok(())
                })
                .unwrap();
            assert_eq!(actual, m);
            engine.check(&sta).unwrap();
            assert_eq!(engine.metrics().unwrap(), baseline);
            // Error exits roll back just like unselected candidates.
            let edit = engine.edit(d, &kind, &cells).unwrap();
            assert!(
                engine
                    .trial(edit, |_, _| Err(anyhow!("intentional oracle failure")))
                    .is_err()
            );
            engine.check(&sta).unwrap();
            tried += 1;
        }
    }
    assert!(tried > 0);
}

#[test]
fn incremental_oracle_covers_all_topologies_aliases_edge_loads_and_siblings() {
    let mut strong = Spec::new("STRONG", "!A", 20.0, 0.1);
    strong.area = 2.0;
    strong.output = "Z";
    let base = library(&[
        Spec::new("PRE", "A", 0.1, 100.0),
        Spec::new("DRIVE", "!A", 1.0, 2.0),
        strong,
        Spec::new("LOAD", "A", 1.0, 0.1),
    ]);
    let mut b = LibraryBuilder::from_library(base);
    for (i, c) in b.cells.iter_mut().enumerate() {
        c.pins[0].rise_capacitance = Some(0.3 + i as f64 * 0.37);
        c.pins[0].fall_capacitance = Some(0.7 + i as f64 * 0.23);
    }
    let lib = b.finish();
    let original = circuit(16, true, true);
    let text =
        emit_module_as_netlist_text(&original.module, &original.nets, &original.interner).unwrap();
    let text = text.replace(
        "endmodule",
        "wire sibling; LOAD sibling_cell(.A(p),.Y(sibling)); endmodule",
    );
    let original = parsed(&text);
    check_incremental_candidates(&original, &lib);
    let mut incremental = original.clone();
    let mut full = original.clone();
    let ir = optimize_fanout(&mut incremental, &lib, options()).unwrap();
    let mut opts = options();
    opts.full_sta = true;
    opts.verify_incremental = false;
    let fr = optimize_fanout(&mut full, &lib, opts).unwrap();
    assert_eq!(
        serde_json::to_value(&ir.trials).unwrap(),
        serde_json::to_value(&fr.trials).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&ir.decisions).unwrap(),
        serde_json::to_value(&fr.decisions).unwrap()
    );
    equivalent(&original, &incremental, &lib);
}

#[test]
fn incremental_stops_only_after_both_edge_arrivals_and_slews_match() {
    // Two equal inputs reconverge. Changing the weaker side can leave the
    // arrival envelope unchanged while changing slew; downstream still matters.
    let lib = library(&[
        Spec::new("PRE", "A", 0.1, 4.0),
        Spec::new("DRIVE", "!A", 1.0, 2.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
        Spec::new("JOIN", "A & B", 0.3, 0.2),
    ]);
    let original = parsed(
        "module top(a,b,y,z); input a,b; output y,z; wire p,q,r,s,t; PRE p0(.A(a),.Y(p)); PRE p1(.A(b),.Y(q)); DRIVE d(.A(p),.Y(r)); LOAD l0(.A(r),.Y(s)); LOAD l1(.A(r),.Y(t)); JOIN j(.A(s),.B(q),.Y(y)); JOIN k(.A(t),.B(q),.Y(z)); endmodule",
    );
    check_incremental_candidates(&original, &lib);
}

#[test]
fn unusable_replacement_timing_is_excluded_without_hiding_invalid_instances() {
    let mut lib = library(&[
        Spec::new("DRIVE", "!A", 1.0, 20.0),
        Spec::new("LOAD", "!A", 1.0, 1.0),
        Spec::new("BROKEN", "A", 0.1, 0.1),
    ]);
    let broken = lib.cells.iter_mut().find(|c| c.name == "BROKEN").unwrap();
    broken.pins.last_mut().unwrap().timing_arcs[0]
        .tables
        .clear();
    let selected = choices(&lib).unwrap();
    assert!(!selected.iter().any(|c| c.name == "BROKEN"));
    let mut original = circuit(32, false, false);
    let mut opts = options();
    opts.verify_incremental = true;
    optimize_fanout(&mut original, &lib, opts).unwrap();
    let mut invalid =
        parsed("module top(a,y); input a; output y; BROKEN b(.A(a),.Y(y)); endmodule");
    assert!(optimize_fanout(&mut invalid, &lib, options()).is_err());
}

#[test]
fn contextual_missing_arc_rejects_candidate_and_continues_search() {
    let mut strong = Spec::new("STRONG", "A & !B", 1.0, 0.1);
    strong.area = 2.0;
    let base = library(&[
        Spec::new("DRIVE", "A & !B", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
        strong,
    ]);
    let mut builder = LibraryBuilder::from_library(base);
    let when = builder.intern_string("B").unwrap();
    builder.cells[2].pins.last_mut().unwrap().timing_arcs[0].when = when;
    let lib = builder.finish();
    assert!(choices(&lib).unwrap().iter().any(|c| c.name == "STRONG"));
    let original = parsed(
        "module top(a,o0,o1,o2,o3); input a; output o0,o1,o2,o3; wire n; DRIVE d(.A(a),.B(1'b0),.Y(n)); LOAD s0(.A(n),.Y(o0)); LOAD s1(.A(n),.Y(o1)); LOAD s2(.A(n),.Y(o2)); LOAD s3(.A(n),.Y(o3)); endmodule",
    );
    for full_sta in [false, true] {
        let mut changed = original.clone();
        let mut opts = options();
        opts.full_sta = full_sta;
        opts.verify_incremental = !full_sta;
        let report = optimize_fanout(&mut changed, &lib, opts).unwrap();
        assert!(report.trials.iter().any(
            |trial| matches!(&trial.kind, FanoutEdit::Upsize { cell } if cell == "STRONG")
                && trial.reason.starts_with("STA rejected:")
        ));
        assert!(!report.decisions.is_empty());
        equivalent(&original, &changed, &lib);
    }
}

#[test]
fn size_only_cells_can_resize_but_cannot_buffer_or_replicate() {
    let mut larger = Spec::new("LARGER", "A", 1.0, 0.1);
    larger.area = 2.0;
    let mut lib = library(&[
        Spec::new("DRIVE", "A", 1.0, 10.0),
        Spec::new("LOAD", "!A", 1.0, 1.0),
        larger,
    ]);
    lib.cells[0].use_for_size_only = Some(true);
    lib.cells[2].use_for_size_only = Some(true);
    let original = circuit(32, false, false);
    let cells = choices(&lib).unwrap();
    let (_, _, graph) = measure(&original, &lib, &options()).unwrap();
    let old = cells.iter().find(|c| c.name == "DRIVE").unwrap();
    let edits = candidates(&graph.drivers[0], old, &cells);
    assert!(
        edits
            .iter()
            .any(|e| matches!(e, FanoutEdit::Upsize { cell } if cell == "LARGER"))
    );
    assert!(edits.iter().all(|e| matches!(e, FanoutEdit::Upsize { .. })));
}

#[test]
fn unused_output_arcs_do_not_invalidate_other_fanout_candidates() {
    let base = library(&[
        Spec::new("DRIVE", "!A", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
        Spec::new("DUAL", "A", 1.0, 1.0),
    ]);
    let mut builder = LibraryBuilder::from_library(base);
    let mut unused = builder.cells[2].pins[1].clone();
    unused.name = builder.intern_string("Z").unwrap();
    unused.timing_arcs[0] = builder
        .add_timing_arc(
            "A",
            "positive_unate",
            "rising_edge",
            "",
            unused.timing_arcs[0].tables.clone(),
        )
        .unwrap();
    builder.cells[2].pins.push(unused);
    let lib = builder.finish();
    let mut original = circuit(8, false, false);
    let text = emit_module_as_netlist_text(&original.module, &original.nets, &original.interner)
        .unwrap()
        .replace(
            "endmodule",
            "wire dummy; DUAL d(.A(a),.Y(dummy)); endmodule",
        );
    original = parsed(&text);
    check_incremental_candidates(&original, &lib);
    let mut builder = LibraryBuilder::from_library(lib);
    let malformed = builder.intern_string("A +").unwrap();
    builder.cells[2].pins[2].function = malformed;
    builder.cells[2].pins[2].timing_arcs.clear();
    check_incremental_candidates(&original, &builder.finish());
}

#[test]
fn unused_unbound_formula_is_not_a_replacement_candidate() {
    let base = library(&[
        Spec::new("DRIVE", "!A", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
        Spec::new("BROKEN", "A", 1.0, 0.1),
    ]);
    let mut builder = LibraryBuilder::from_library(base);
    let function = builder.intern_string("A & MISSING").unwrap();
    builder.cells[2].pins[1].function = function;
    let lib = builder.finish();
    assert!(!choices(&lib).unwrap().iter().any(|c| c.name == "BROKEN"));
    let mut netlist = circuit(8, false, false);
    optimize_fanout(&mut netlist, &lib, options()).unwrap();
}

#[test]
fn unknown_size_policy_cannot_create_new_instances() {
    let mut lib = library(&[
        Spec::new("DRIVE", "A", 1.0, 10.0),
        Spec::new("LOAD", "!A", 1.0, 1.0),
    ]);
    lib.cells[0]
        .unsupported_semantics
        .push(crate::liberty_proto::UnsupportedSemantics {
            feature: crate::liberty_proto::UnsupportedFeature::IncompleteSizeOnlyCoverage as i32,
            ..Default::default()
        });
    let original = circuit(8, false, false);
    let cells = choices(&lib).unwrap();
    let (_, _, graph) = measure(&original, &lib, &options()).unwrap();
    let old = cells.iter().find(|c| c.name == "DRIVE").unwrap();
    assert!(candidates(&graph.drivers[0], old, &cells).is_empty());
}

#[test]
fn same_instance_boolean_constant_feedback_matches_full_sta() {
    let mut dual = Spec::new("DUAL", "B & C", 1.0, 0.1);
    dual.output = "Z";
    let base = library(&[
        Spec::new("DRIVE", "!A", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
        dual,
    ]);
    let mut builder = LibraryBuilder::from_library(base);
    builder.cells[2].pins[2].timing_arcs.remove(0); // B is tied to a proven zero.
    let mut input = builder.cells[2].pins[0].clone();
    input.name = builder.intern_string("A").unwrap();
    let mut output = builder.cells[2].pins[2].clone();
    output.name = builder.intern_string("Y").unwrap();
    output.function = input.name;
    output.timing_arcs[0] = builder
        .add_timing_arc(
            "A",
            "positive_unate",
            "combinational",
            "",
            output.timing_arcs[0].tables.clone(),
        )
        .unwrap();
    builder.cells[2].pins.extend([input, output]);
    let lib = builder.finish();
    let netlist = circuit(8, false, false);
    let text = emit_module_as_netlist_text(&netlist.module, &netlist.nets, &netlist.interner)
        .unwrap()
        .replace(
            "endmodule",
            "wire tied, extra; DUAL dual(.A(1'b0),.B(tied),.C(a),.Y(tied),.Z(extra)); endmodule",
        );
    check_incremental_candidates(&parsed(&text), &lib);
}

#[test]
fn boolean_only_connectivity_cycles_fall_back_to_full_sta() {
    let base = library(&[
        Spec::new("DRIVE", "!A", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
        Spec::new("EXTRA", "A", 1.0, 0.1),
    ]);
    let mut builder = LibraryBuilder::from_library(base);
    let mut ignored = builder.cells[2].pins[0].clone();
    ignored.name = builder.intern_string("B").unwrap();
    builder.cells[2].pins.push(ignored);
    let lib = builder.finish();
    let original = circuit(8, false, false);
    let text = emit_module_as_netlist_text(&original.module, &original.nets, &original.interner)
        .unwrap()
        .replace(
            "endmodule",
            "wire p,q; EXTRA e0(.A(a),.B(q),.Y(p)); EXTRA e1(.A(a),.B(p),.Y(q)); endmodule",
        );
    let mut incremental = parsed(&text);
    let mut full = incremental.clone();
    let report = optimize_fanout(&mut incremental, &lib, options()).unwrap();
    assert!(report.incremental.is_none());
    assert!(report.incremental_fallback_reason.is_some());
    let mut opts = options();
    opts.full_sta = true;
    opts.verify_incremental = false;
    let oracle = optimize_fanout(&mut full, &lib, opts).unwrap();
    assert_eq!(report.after, oracle.after);
    assert_eq!(
        serde_json::to_value(report.decisions).unwrap(),
        serde_json::to_value(oracle.decisions).unwrap()
    );
}

#[test]
fn clock_marked_cells_do_not_supply_boolean_coverage_constants() {
    let mut strong = Spec::new("STRONG", "A & !B", 1.0, 0.1);
    strong.area = 2.0;
    let base = library(&[
        Spec::new("DRIVE", "A & !B", 1.0, 10.0),
        Spec::new("LOAD", "A", 1.0, 0.1),
        strong,
        Spec::new("CLOCKBUF", "A", 1.0, 0.1),
    ]);
    let mut builder = LibraryBuilder::from_library(base);
    let when = builder.intern_string("B").unwrap();
    builder.cells[2].pins.last_mut().unwrap().timing_arcs[0].when = when;
    builder.cells[3].pins[0].is_clocking_pin = true;
    let lib = builder.finish();
    let original = parsed(
        "module top(a,o0,o1,o2,o3); input a; output o0,o1,o2,o3; wire n,fixed; CLOCKBUF clk(.A(1'b0),.Y(fixed)); DRIVE d(.A(fixed),.B(1'b0),.Y(n)); LOAD s0(.A(n),.Y(o0)); LOAD s1(.A(n),.Y(o1)); LOAD s2(.A(n),.Y(o2)); LOAD s3(.A(n),.Y(o3)); endmodule",
    );
    for full_sta in [false, true] {
        let mut changed = original.clone();
        let mut opts = options();
        opts.full_sta = full_sta;
        opts.verify_incremental = !full_sta;
        let report = optimize_fanout(&mut changed, &lib, opts).unwrap();
        assert!(report.trials.iter().any(
            |trial| matches!(&trial.kind, FanoutEdit::Upsize { cell } if cell == "STRONG")
                && trial.reason.starts_with("STA rejected:")
        ));
    }
}
