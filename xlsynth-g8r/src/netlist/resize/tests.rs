// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::liberty::parser::parse_liberty_files;
use crate::netlist::io::{ParsedNetlist, read_gv_from_str};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::io::Write;
use tempfile::NamedTempFile;

/// A small load- and slew-sensitive library with explicitly sized equivalents.
fn library() -> Result<Library> {
    let mut text = String::from(
        r#"library(test) {
        lu_table_template(delay) { variable_1: input_net_transition; variable_2: total_output_net_capacitance; index_1("0,10"); index_2("0,10"); }
        lu_table_template(setup) { variable_1: constrained_pin_transition; index_1("0,10"); }
    "#,
    );
    for kind in ["BUF", "INV", "DFF", "AND", "DUAL"] {
        for size in [1, 2, 4] {
            let d = 1.0 / f64::from(size);
            let cap = f64::from(size) * 0.2;
            let (input, output, function, related, sense, ty) = match kind {
                "DFF" => ("D", "Q", "IQ", "CLK", "non_unate", "rising_edge"),
                "INV" => ("A", "Y", "!A", "A", "negative_unate", "combinational"),
                "AND" => ("A", "Y", "A & B", "A B", "positive_unate", "combinational"),
                _ => ("A", "Y", "A", "A", "positive_unate", "combinational"),
            };
            text.push_str(&format!(r#"cell({kind}{size}) {{ area: {size}; pin({input}) {{ direction: input; capacitance: {cap}; "#));
            if kind == "DFF" {
                text.push_str(&format!(r#"timing() {{ related_pin: "CLK"; timing_type: setup_rising;
                    rise_constraint(setup) {{ values("{},{}"); }} fall_constraint(setup) {{ values("{},{}"); }} }}"#, d, d + 1.0, d, d + 1.0));
            }
            text.push('}');
            if kind == "AND" {
                text.push_str(&format!(r#"pin(B) {{ direction: input; rise_capacitance: {cap}; fall_capacitance: {}; }}"#, cap * 0.5));
            }
            if kind == "DFF" {
                text.push_str(r#"pin(CLK) { direction: input; clock: true; capacitance: 0.1; } ff(IQ, IQN) { clocked_on: "CLK"; next_state: "D"; }"#);
            }
            text.push_str(&format!(
                r#"pin({output}) {{ direction: output; function: "{function}";"#
            ));
            for related_pin in related.split_ascii_whitespace() {
                text.push_str(&format!(r#"timing() {{ related_pin: "{related_pin}"; timing_sense: {sense}; timing_type: {ty};"#));
                for name in [
                    "cell_rise",
                    "cell_fall",
                    "rise_transition",
                    "fall_transition",
                ] {
                    text.push_str(&format!(
                        r#"{name}(delay) {{ values("{},{}", "{},{}"); }}"#,
                        d,
                        d * 21.0,
                        d + 2.0,
                        d * 21.0 + 2.0
                    ));
                }
                text.push('}');
            }
            text.push('}');
            if kind == "DUAL" {
                text.push_str(
                    r#"pin(YN) { direction: output; function: "!A";
                    timing() { related_pin: "A"; timing_sense: negative_unate;"#,
                );
                for name in [
                    "cell_rise",
                    "cell_fall",
                    "rise_transition",
                    "fall_transition",
                ] {
                    text.push_str(&format!(
                        r#"{name}(delay) {{ values("{},{}", "{},{}"); }}"#,
                        d * 2.0,
                        d * 42.0,
                        d * 2.0 + 2.0,
                        d * 42.0 + 2.0
                    ));
                }
                text.push_str("}}");
            }
            text.push('}');
        }
    }
    text.push('}');
    let mut file = NamedTempFile::new()?;
    file.write_all(text.as_bytes())?;
    parse_liberty_files(&[file.path()]).map_err(|e| anyhow!(e))
}

fn options() -> ResizeOptions {
    ResizeOptions {
        max_area_growth: 10.0,
        timing: StaOptions {
            primary_input_transition: 0.01,
            module_output_load: 10.0,
        },
        ..ResizeOptions::default()
    }
}

fn resize(
    parsed: &mut ParsedNetlist,
    library: &Library,
    options: &ResizeOptions,
) -> Result<ResizeReport> {
    resize_netlist(
        &mut parsed.modules[0],
        &parsed.nets,
        &mut parsed.interner,
        library,
        options,
    )
}

#[test]
fn timing_then_area_preserves_wires_and_polarity() -> Result<()> {
    let library = library()?;
    let mut parsed = read_gv_from_str(
        "module m(input a, output y); wire n; INV1 i(.A(a), .Y(n)); BUF1 b(.A(n), .Y(y)); endmodule",
    )?;
    let original = parsed.modules[0].clone();
    let report = resize(&mut parsed, &library, &options())?;
    assert!(report.final_delay.input_to_output < report.initial_delay.input_to_output);
    assert!(report.final_delay.within(report.upsized_delay));
    assert!(report.upsize.accepted > 0);
    for (old, new) in original.instances.iter().zip(&parsed.modules[0].instances) {
        assert_eq!(old.connections, new.connections);
        assert_eq!(old.instance_name, new.instance_name);
    }
    assert!(
        parsed
            .interner
            .resolve(parsed.modules[0].instances[0].type_name)
            .unwrap()
            .starts_with("INV")
    );
    Ok(())
}

#[test]
fn negative_arrival_paths_remain_eligible_for_area_neutral_timing_improvement() -> Result<()> {
    let mut text = String::from("library(test) {");
    for (name, delay) in [("BUF1", -1), ("BUF2", -2)] {
        text.push_str(&format!(
            r#"cell({name}) {{ area: 1;
            pin(A) {{ direction: input; capacitance: 1; }}
            pin(Y) {{ direction: output; function: "A";
                timing() {{ related_pin: "A"; timing_sense: positive_unate;
                    cell_rise(scalar) {{ values("{delay}"); }}
                    cell_fall(scalar) {{ values("{delay}"); }}
                    rise_transition(scalar) {{ values("1"); }}
                    fall_transition(scalar) {{ values("1"); }}
                }} }} }}"#
        ));
    }
    text.push('}');
    let mut file = NamedTempFile::new()?;
    file.write_all(text.as_bytes())?;
    let library = parse_liberty_files(&[file.path()]).map_err(anyhow::Error::msg)?;
    let mut parsed =
        read_gv_from_str("module m(input a, output y); BUF1 b(.A(a),.Y(y)); endmodule")?;
    let report = resize(&mut parsed, &library, &options())?;
    assert_eq!(report.initial_delay.input_to_output, -1.0);
    assert_eq!(report.final_delay.input_to_output, -2.0);
    assert_eq!(report.final_area, report.initial_area);
    Ok(())
}

#[test]
fn downsize_uses_fixed_worst_endpoint_limit() -> Result<()> {
    let library = library()?;
    let mut parsed = read_gv_from_str(
        "module m(input a, output y, output z); BUF1 slow(.A(a),.Y(y)); BUF4 fast(.A(a),.Y(z)); endmodule",
    )?;
    let options = ResizeOptions {
        max_upsize_rounds: 0,
        ..options()
    };
    let report = resize(&mut parsed, &library, &options)?;
    assert!(report.final_area < report.initial_area);
    assert!(report.final_delay.within(report.initial_delay));
    assert!(report.downsize.accepted > 0);
    Ok(())
}

#[test]
fn recovery_headroom_distinguishes_independent_critical_and_noncritical_cones() -> Result<()> {
    let library = library()?;
    let parsed = read_gv_from_str(
        "module m(input a, output y, output z); BUF1 slow(.A(a),.Y(y)); BUF4 fast(.A(a),.Y(z)); endmodule",
    )?;
    let sta = IncrementalSta::new(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        &library,
        options().timing,
    )?;
    let margins = sta.recovery_headroom(sta.score());
    assert_eq!(margins[0], 0.0);
    assert!(margins[1] > 0.0);
    Ok(())
}

#[test]
fn improves_more_than_thirty_two_equally_critical_outputs() -> Result<()> {
    let library = library()?;
    let mut text = String::from("module m(input a, output [63:0] y);\n");
    for i in 0..64 {
        text.push_str(&format!("BUF1 b{i}(.A(a), .Y(y[{i}]));\n"));
    }
    text.push_str("endmodule\n");
    let mut parsed = read_gv_from_str(&text)?;
    let report = resize(&mut parsed, &library, &options())?;
    assert!(report.upsize.accepted >= 64);
    assert!(report.final_delay.input_to_output < report.initial_delay.input_to_output);
    Ok(())
}

#[test]
fn fixed_capture_registers_do_not_consume_timing_candidate_slots() -> Result<()> {
    let mut library = library()?;
    // Isolate candidate coverage from the area/delay merit guard.
    for cell in &mut library.cells {
        cell.area = 1.0;
    }
    let mut parsed = read_gv_from_str(
        r#"module m(input clk, input d, output q);
        wire a,b;
        DFF1 capture(.CLK(clk),.D(b),.Q(q));
        DFF1 launch(.CLK(clk),.D(d),.Q(a));
        BUF1 comb(.A(a),.Y(b));
        endmodule"#,
    )?;
    let report = resize(
        &mut parsed,
        &library,
        &ResizeOptions {
            resize_registers: false,
            max_candidate_sites: 1,
            max_upsize_rounds: 1,
            max_downsize_rounds: 0,
            ..options()
        },
    )?;
    assert_eq!(report.upsize.register_changes, 0);
    assert!(report.final_delay.register_to_register < report.initial_delay.register_to_register);
    Ok(())
}

#[test]
fn unaffordable_cells_do_not_hide_area_neutral_timing_candidates() -> Result<()> {
    let mut library = library()?;
    for cell in &mut library.cells {
        if cell.name.starts_with("BUF") {
            cell.area = 1.0;
        }
    }
    let mut parsed = read_gv_from_str(
        r#"module m(input clk, input d, output q);
        wire a,b;
        DFF1 capture(.CLK(clk),.D(b),.Q(q));
        DFF1 launch(.CLK(clk),.D(d),.Q(a));
        BUF1 comb(.A(a),.Y(b));
        endmodule"#,
    )?;
    let report = resize(
        &mut parsed,
        &library,
        &ResizeOptions {
            max_area_growth: 0.0,
            max_candidate_sites: 1,
            max_upsize_rounds: 1,
            max_downsize_rounds: 0,
            ..options()
        },
    )?;
    assert_eq!(report.upsize.register_changes, 0);
    assert_eq!(report.final_area, report.initial_area);
    assert!(report.final_delay.register_to_register < report.initial_delay.register_to_register);
    Ok(())
}

#[test]
fn cells_without_alternatives_do_not_hide_resizable_fanin() -> Result<()> {
    let mut library = library()?;
    for cell in &mut library.cells {
        cell.area = 1.0;
        if cell.name == "BUF2" || cell.name == "BUF4" {
            cell.dont_use = Some(true);
        }
    }
    let mut parsed = read_gv_from_str(
        r#"module m(input a, output y);
        wire b;
        BUF1 fixed(.A(b),.Y(y));
        INV1 resizable(.A(a),.Y(b));
        endmodule"#,
    )?;
    let report = resize(
        &mut parsed,
        &library,
        &ResizeOptions {
            max_candidate_sites: 1,
            max_upsize_rounds: 1,
            max_downsize_rounds: 0,
            ..options()
        },
    )?;
    assert!(report.final_delay.input_to_output < report.initial_delay.input_to_output);
    Ok(())
}

#[test]
fn resizes_launch_and_capture_without_boundary_inverters() -> Result<()> {
    let library = library()?;
    let mut parsed = read_gv_from_str(
        "module m(input clk, input d, output q); wire a,b; DFF1 launch(.CLK(clk),.D(d),.Q(a)); BUF1 comb(.A(a),.Y(b)); DFF1 capture(.CLK(clk),.D(b),.Q(q)); endmodule",
    )?;
    let wires: Vec<_> = parsed.modules[0]
        .instances
        .iter()
        .map(|i| i.connections.clone())
        .collect();
    let report = resize(&mut parsed, &library, &options())?;
    assert!(report.upsize.register_changes > 0);
    assert!(report.final_delay.register_to_register < report.initial_delay.register_to_register);
    assert!(report.final_delay.within(report.initial_delay));
    assert_eq!(
        wires,
        parsed.modules[0]
            .instances
            .iter()
            .map(|i| i.connections.clone())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn critical_candidates_do_not_cross_launch_register_data_boundaries() -> Result<()> {
    let library = library()?;
    let parsed = read_gv_from_str(
        r#"module m(input clk, input d, output q);
        wire a,b,c;
        BUF1 before_launch(.A(d),.Y(a));
        DFF1 launch(.CLK(clk),.D(a),.Q(b));
        BUF1 comb(.A(b),.Y(c));
        DFF1 capture(.CLK(clk),.D(c),.Q(q));
        endmodule"#,
    )?;
    let sta = IncrementalSta::new(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        &library,
        options().timing,
    )?;
    assert_eq!(sta.objective(), 3);
    let mut candidates = sta.critical_instances(3, usize::MAX, |_| true);
    candidates.sort_unstable();
    assert_eq!(candidates, vec![1, 2, 3]);
    assert_eq!(sta.critical_instances(2, usize::MAX, |_| true), vec![3]);
    Ok(())
}

#[test]
fn zero_budget_is_identity_and_area_limit_does_not_compound() -> Result<()> {
    let library = library()?;
    let mut parsed =
        read_gv_from_str("module m(input a, output y); BUF1 b(.A(a),.Y(y)); endmodule")?;
    let before = parsed.modules[0].clone();
    let report = resize(
        &mut parsed,
        &library,
        &ResizeOptions {
            max_propagated_instances: 0,
            ..options()
        },
    )?;
    assert_eq!(before, parsed.modules[0]);
    assert!(report.budget_exhausted);
    let report = resize(
        &mut parsed,
        &library,
        &ResizeOptions {
            max_area_growth: 0.02,
            ..options()
        },
    )?;
    assert!(report.final_area <= report.initial_area * 1.02);
    assert_eq!(before, parsed.modules[0]);
    Ok(())
}

#[test]
fn dont_use_and_semantic_mismatches_are_not_substituted() -> Result<()> {
    let mut library = library()?;
    for cell in &mut library.cells {
        if cell.name == "BUF2" || cell.name == "BUF4" {
            cell.dont_use = Some(true);
        }
    }
    let families = CellFamilies::new(&library);
    let idx = library.cells.iter().position(|c| c.name == "BUF1").unwrap();
    assert_eq!(families.alternatives[idx], vec![idx]);
    let dff = library.cells.iter().position(|c| c.name == "DFF2").unwrap();
    library.cells[dff].sequential[0].clock_expr = "!CLK".to_string();
    let families = CellFamilies::new(&library);
    assert_eq!(families.alternatives[dff], vec![dff]);
    Ok(())
}

#[test]
fn random_substitution_sequences_match_full_timing_and_rollback() -> Result<()> {
    let mut library = library()?;
    // Equivalent cells need not use the same Liberty pin storage order.
    for cell in library.cells.iter_mut().step_by(2) {
        cell.pins.reverse();
    }
    let mut parsed = read_gv_from_str(
        r#"module m(input clk, input [1:0] d, output [1:0] y);
        wire a,b,c,e,f,g;
        DFF1 ff(.CLK(clk),.D(d[0]),.Q(a));
        BUF1 p(.A(a),.Y(b)); INV2 n(.A(b),.Y(c));
        BUF4 side(.A(b),.Y(e)); BUF1 other(.A(d[1]),.Y(f));
        AND1 reconverge(.A(c),.B(f),.Y(g));
        DFF2 capture(.CLK(clk),.D(g),.Q(y[0])); assign y[1] = e;
        endmodule"#,
    )?;
    let original = parsed.modules[0].clone();
    let families = CellFamilies::new(&library);
    for cell in &library.cells {
        parsed.interner.get_or_intern(&cell.name);
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x98765);
    let mut sta = IncrementalSta::new(
        &original,
        &parsed.nets,
        &parsed.interner,
        &library,
        options().timing,
    )?;
    let mut full_propagation = IncrementalSta::new(
        &original,
        &parsed.nets,
        &parsed.interner,
        &library,
        options().timing,
    )?;
    for iteration in 0..250 {
        let i = rng.gen_range(0..sta.cells.len());
        let alternatives = &families.alternatives[sta.cells[i]];
        let replacement = alternatives[rng.gen_range(0..alternatives.len())];
        let accept = iteration % 3 != 0;
        let old = sta.score();
        let old_cells = sta.cells.clone();
        let limits = (iteration % 2 == 0).then_some(old);
        let acceptable = |timing: &IncrementalSta<'_>| {
            accept && limits.is_none_or(|limits| timing.score().within(limits))
        };
        let expected = full_propagation.trial(i, replacement, usize::MAX, None, acceptable)?;
        let committed = sta.trial(i, replacement, usize::MAX, limits, acceptable)?;
        assert_eq!(committed, expected);
        assert_eq!(sta.score(), full_propagation.score());
        assert_eq!(sta.cells, full_propagation.cells);
        if !committed {
            assert_eq!(sta.score(), old);
            assert_eq!(sta.cells, old_cells);
        }
        let mut trial_module = original.clone();
        // All cell names are preinterned by fixture setup above rather
        // than modifying the interner while the timing engine borrows it.
        for (instance, cell) in trial_module.instances.iter_mut().zip(&sta.cells) {
            instance.type_name = parsed.interner.get(&library.cells[*cell].name).unwrap();
        }
        let full = measure_boundary_delays(
            &trial_module,
            &parsed.nets,
            &parsed.interner,
            &library,
            options().timing,
        )?;
        verify_timing(sta.score(), full)?;
    }
    Ok(())
}

#[test]
fn multi_output_resizing_updates_both_cones_and_restores_rejected_trials() -> Result<()> {
    let mut library = library()?;
    // Opposite storage orders must not swap the positive and negative output.
    let larger = library
        .cells
        .iter()
        .position(|c| c.name == "DUAL4")
        .unwrap();
    library.cells[larger].pins.reverse();
    // Different downstream strengths make each output cone critical in turn.
    for inverter in ["INV1", "INV2"] {
        let parsed = read_gv_from_str(&format!(
            r#"module m(input a, output y, output z);
        wire p,n;
        DUAL1 dual(.A(a),.Y(p),.YN(n));
        BUF1 positive(.A(p),.Y(y));
        {inverter} negative(.A(n),.Y(z));
        endmodule"#,
        ))?;
        let mut sta = IncrementalSta::new(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            &library,
            options().timing,
        )?;
        let before = sta.score();
        assert!(!sta.trial(0, larger, usize::MAX, None, |_| false)?);
        assert_eq!(sta.score(), before);
        assert!(sta.trial(0, larger, usize::MAX, None, |_| true)?);
        let mut changed = parsed.modules[0].clone();
        let mut interner = parsed.interner.clone();
        changed.instances[0].type_name = interner.get_or_intern("DUAL4");
        verify_timing(
            sta.score(),
            measure_boundary_delays(
                &changed,
                &parsed.nets,
                &interner,
                &library,
                options().timing,
            )?,
        )?;
        assert!(sta.score().input_to_output < before.input_to_output);
    }
    Ok(())
}

#[test]
fn multi_output_families_compare_every_output_function() -> Result<()> {
    let mut library = library()?;
    let small = library
        .cells
        .iter()
        .position(|c| c.name == "DUAL1")
        .unwrap();
    let medium = library
        .cells
        .iter()
        .position(|c| c.name == "DUAL2")
        .unwrap();
    let large = library
        .cells
        .iter()
        .position(|c| c.name == "DUAL4")
        .unwrap();
    assert_eq!(
        CellFamilies::new(&library).alternatives[small],
        vec![small, medium, large]
    );
    let positive = library.cells[large]
        .pins
        .iter()
        .find(|pin| library.resolve_string(&pin.name) == "Y")
        .unwrap()
        .function;
    let negative = library.cells[large]
        .pins
        .iter()
        .position(|pin| library.resolve_string(&pin.name) == "YN")
        .unwrap();
    library.cells[large].pins[negative].function = positive;
    assert_eq!(
        CellFamilies::new(&library).alternatives[small],
        vec![small, medium]
    );
    assert_eq!(CellFamilies::new(&library).alternatives[large], vec![large]);
    Ok(())
}

#[test]
fn disabling_register_resizing_preserves_all_register_cells() -> Result<()> {
    let library = library()?;
    let mut parsed = read_gv_from_str(
        "module m(input clk, input d, output q); wire a; DFF1 launch(.CLK(clk),.D(d),.Q(a)); DFF1 capture(.CLK(clk),.D(a),.Q(q)); endmodule",
    )?;
    let before = parsed.modules[0].clone();
    let report = resize(
        &mut parsed,
        &library,
        &ResizeOptions {
            resize_registers: false,
            ..options()
        },
    )?;
    assert_eq!(before, parsed.modules[0]);
    assert_eq!(report.upsize.register_changes, 0);
    assert_eq!(report.downsize.register_changes, 0);
    Ok(())
}

#[test]
fn invalid_options_leave_the_netlist_unchanged() -> Result<()> {
    let library = library()?;
    let mut parsed =
        read_gv_from_str("module m(input a, output y); BUF4 b(.A(a),.Y(y)); endmodule")?;
    let before = parsed.modules[0].clone();
    for options in [
        ResizeOptions {
            max_area_growth: f64::INFINITY,
            ..options()
        },
        ResizeOptions {
            max_area_growth: f64::MAX,
            ..options()
        },
        ResizeOptions {
            max_area_growth: -0.01,
            ..options()
        },
        ResizeOptions {
            max_candidate_sites: 0,
            ..options()
        },
        ResizeOptions {
            max_cell_candidates: 0,
            ..options()
        },
    ] {
        assert!(resize(&mut parsed, &library, &options).is_err());
        assert_eq!(before, parsed.modules[0]);
    }
    Ok(())
}

#[test]
fn rejection_and_budget_rollback_restore_exact_timing() -> Result<()> {
    let library = library()?;
    let parsed = read_gv_from_str(
        "module m(input a, output y); wire b,c; BUF1 p(.A(a),.Y(b)); BUF1 q(.A(b),.Y(c)); BUF1 r(.A(c),.Y(y)); endmodule",
    )?;
    let mut sta = IncrementalSta::new(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        &library,
        options().timing,
    )?;
    let original_score = sta.score();
    let original_cells = sta.cells.clone();
    let larger = library.cells.iter().position(|c| c.name == "BUF4").unwrap();
    assert!(!sta.trial(1, larger, usize::MAX, None, |_| false)?);
    assert_eq!(sta.score(), original_score);
    assert_eq!(sta.cells, original_cells);
    let budget = sta.propagated + 1;
    assert!(!sta.trial(1, larger, budget, None, |_| true)?);
    assert!(sta.budget_exhausted);
    assert_eq!(sta.score(), original_score);
    assert_eq!(sta.cells, original_cells);
    // A subsequent trial must not inherit any rejected or aborted work.
    assert!(sta.trial(1, larger, usize::MAX, None, |_| true)?);
    let mut module = parsed.modules[0].clone();
    let mut interner = parsed.interner.clone();
    module.instances[1].type_name = interner.get_or_intern("BUF4");
    verify_timing(
        sta.score(),
        measure_boundary_delays(&module, &parsed.nets, &interner, &library, options().timing)?,
    )?;
    Ok(())
}

#[test]
fn settled_output_violation_rejects_before_propagating_unrelated_logic() -> Result<()> {
    let library = library()?;
    for register_launch in [false, true] {
        let mut text =
            String::from("module m(input clk, input d, output tap, output y); wire [16:0] n;");
        text.push_str(if register_launch {
            "DFF4 root(.CLK(clk),.D(d),.Q(n[0]));"
        } else {
            "BUF4 root(.A(d),.Y(n[0]));"
        });
        for i in 0..16 {
            text.push_str(&format!("BUF4 b{i}(.A(n[{i}]),.Y(n[{}]));", i + 1));
        }
        text.push_str("assign tap=n[0]; assign y=n[16]; endmodule");
        let parsed = read_gv_from_str(&text)?;
        let mut sta = IncrementalSta::new(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            &library,
            StaOptions {
                module_output_load: 1000.0,
                ..options().timing
            },
        )?;
        let before = sta.score();
        let original = sta.cells.clone();
        let weak = library
            .cells
            .iter()
            .position(|cell| cell.name == if register_launch { "DFF1" } else { "BUF1" })
            .unwrap();
        assert!(!sta.trial(0, weak, usize::MAX, None, |timing| {
            timing.score().within(before)
        })?);
        let full_work = sta.propagated;
        assert!(full_work > 1);
        assert!(!sta.trial(0, weak, usize::MAX, Some(before), |timing| {
            timing.score().within(before)
        })?);
        assert!(sta.propagated - full_work < full_work);
        assert_eq!(sta.score(), before);
        assert_eq!(sta.cells, original);
        assert!(!sta.budget_exhausted);
    }
    Ok(())
}

#[test]
fn capture_limit_rejection_precedes_final_acceptance_and_restores_timing() -> Result<()> {
    let library = library()?;
    for register_launch in [false, true] {
        let root = if register_launch {
            "DFF4 launch(.CLK(clk),.D(d),.Q(n));"
        } else {
            "BUF4 launch(.A(d),.Y(n));"
        };
        let parsed = read_gv_from_str(&format!(
            "module m(input clk, input d, output q); wire n; {root}
             DFF1 capture(.CLK(clk),.D(n),.Q(q)); endmodule"
        ))?;
        let mut sta = IncrementalSta::new(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            &library,
            options().timing,
        )?;
        let before = sta.score();
        let cells = sta.cells.clone();
        let weak = library
            .cells
            .iter()
            .position(|cell| cell.name == if register_launch { "DFF1" } else { "BUF1" })
            .unwrap();
        assert!(!sta.trial(0, weak, usize::MAX, None, |timing| {
            timing.score().within(before)
        })?);
        let mut accepted_called = false;
        assert!(!sta.trial(0, weak, usize::MAX, Some(before), |_| {
            accepted_called = true;
            true
        })?);
        assert!(!accepted_called);
        assert_eq!(sta.score(), before);
        assert_eq!(sta.cells, cells);
        assert!(!sta.budget_exhausted);
    }
    Ok(())
}

#[test]
fn failed_candidate_restores_upstream_load_and_timing() -> Result<()> {
    let mut library = library()?;
    let bad = library.cells.iter().position(|c| c.name == "BUF4").unwrap();
    library.cells[bad]
        .pins
        .iter_mut()
        .find(|pin| pin.direction == PinDirection::Output as i32)
        .unwrap()
        .timing_arcs
        .clear();
    let parsed = read_gv_from_str(
        "module m(input a, output y); wire b; BUF1 p(.A(a),.Y(b)); BUF1 q(.A(b),.Y(y)); endmodule",
    )?;
    let mut sta = IncrementalSta::new(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        &library,
        options().timing,
    )?;
    let before = sta.score();
    assert!(sta.trial(1, bad, usize::MAX, None, |_| true).is_err());
    assert_eq!(sta.score(), before);
    let good = library.cells.iter().position(|c| c.name == "BUF2").unwrap();
    assert!(sta.trial(1, good, usize::MAX, None, |_| true)?);
    let mut module = parsed.modules[0].clone();
    let mut interner = parsed.interner.clone();
    module.instances[1].type_name = interner.get_or_intern("BUF2");
    verify_timing(
        sta.score(),
        measure_boundary_delays(&module, &parsed.nets, &interner, &library, options().timing)?,
    )?;
    Ok(())
}

#[test]
fn electrical_limit_cannot_be_exchanged_for_timing_gain() -> Result<()> {
    for limit in ["capacitance", "transition", "fanout"] {
        let mut library = library()?;
        let larger = library.cells.iter().position(|c| c.name == "BUF4").unwrap();
        let output = library.cells[larger]
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Output as i32)
            .unwrap();
        match limit {
            "capacitance" => output.max_capacitance = Some(0.1),
            "transition" => output.max_transition = Some(0.1),
            "fanout" => output.max_fanout = Some(0.5),
            _ => unreachable!(),
        }
        let parsed = read_gv_from_str(
            "module m(input a, output y); wire b; BUF1 p(.A(a),.Y(b)); BUF1 q(.A(b),.Y(y)); endmodule",
        )?;
        let mut sta = IncrementalSta::new(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            &library,
            options().timing,
        )?;
        let before = sta.score();
        assert!(
            !sta.trial(0, larger, usize::MAX, None, |_| true)?,
            "{limit}"
        );
        assert_eq!(sta.score(), before);
    }
    Ok(())
}

#[test]
fn resizing_clock_drivers_preserves_clock_pin_transition_limits() -> Result<()> {
    let mut library = library()?;
    let ff = library.cells.iter().position(|c| c.name == "DFF1").unwrap();
    library.cells[ff]
        .pins
        .iter_mut()
        .find(|pin| pin.is_clocking_pin)
        .unwrap()
        .max_transition = Some(0.5);
    let parsed = read_gv_from_str(
        r#"module m(input clk, input d, output q); wire c;
        BUF4 driver(.A(clk),.Y(c));
        DFF1 ff(.CLK(c),.D(d),.Q(q)); endmodule"#,
    )?;
    let mut sta = IncrementalSta::new(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        &library,
        options().timing,
    )?;
    let before = sta.score();
    let cells = sta.cells.clone();
    let weak = library.cells.iter().position(|c| c.name == "BUF1").unwrap();
    assert!(!sta.trial(0, weak, usize::MAX, None, |_| true)?);
    assert_eq!(sta.score(), before);
    assert_eq!(sta.cells, cells);
    Ok(())
}

#[test]
fn clock_annotations_do_not_cut_combinational_timing_dependencies() -> Result<()> {
    let mut library = library()?;
    for cell in &mut library.cells {
        if cell.name.starts_with("BUF") {
            cell.pins
                .iter_mut()
                .find(|pin| pin.direction == PinDirection::Input as i32)
                .unwrap()
                .is_clocking_pin = true;
        }
    }
    // Deliberately store the consumer before its driver: a pin annotation
    // must not replace the cell's actual combinational timing dependencies.
    let mut parsed = read_gv_from_str(
        r#"module m(input a, output y); wire n;
        BUF1 consumer(.A(n),.Y(y)); BUF1 driver(.A(a),.Y(n)); endmodule"#,
    )?;
    let report = resize(&mut parsed, &library, &options())?;
    assert!(report.final_delay.input_to_output < report.initial_delay.input_to_output);
    assert!(report.upsize.accepted > 0);
    Ok(())
}

#[test]
fn resizing_a_consumer_preserves_its_drivers_electrical_limits() -> Result<()> {
    for limit in ["capacitance", "fanout"] {
        let mut library = library()?;
        let small = library.cells.iter().position(|c| c.name == "BUF1").unwrap();
        let large = library.cells.iter().position(|c| c.name == "BUF4").unwrap();
        let output = library.cells[small]
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Output as i32)
            .unwrap();
        if limit == "capacitance" {
            output.max_capacitance = Some(0.3);
        } else {
            output.max_fanout = Some(1.5);
            let input = library.cells[large]
                .pins
                .iter_mut()
                .find(|pin| pin.direction == PinDirection::Input as i32)
                .unwrap();
            // A weighted fanout change must be checked even when the load
            // capacitance is unchanged and no driver retiming is needed.
            input.capacitance = Some(0.2);
            input.fanout_load = Some(2.0);
        }
        let parsed = read_gv_from_str(
            "module m(input a, output y); wire b; BUF1 p(.A(a),.Y(b)); BUF1 q(.A(b),.Y(y)); endmodule",
        )?;
        let mut sta = IncrementalSta::new(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            &library,
            options().timing,
        )?;
        let before = sta.score();
        assert!(!sta.trial(1, large, usize::MAX, None, |_| true)?, "{limit}");
        assert_eq!(sta.score(), before);
        assert_eq!(sta.cells, vec![small, small]);
        assert!(sta.trial(0, large, usize::MAX, None, |_| true)?);
    }
    Ok(())
}

#[test]
fn resizing_a_driver_preserves_combinational_and_capture_pin_slew_limits() -> Result<()> {
    for capture in [false, true] {
        let mut library = library()?;
        let consumer_name = if capture { "DFF1" } else { "BUF1" };
        let consumer = library
            .cells
            .iter()
            .position(|cell| cell.name == consumer_name)
            .unwrap();
        library.cells[consumer]
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Input as i32 && !pin.is_clocking_pin)
            .unwrap()
            .max_transition = Some(0.5);
        let text = if capture {
            "module m(input clk, input a, output y); wire b; BUF4 p(.A(a),.Y(b)); DFF1 q(.CLK(clk),.D(b),.Q(y)); endmodule"
        } else {
            "module m(input a, output y); wire b; BUF4 p(.A(a),.Y(b)); BUF1 q(.A(b),.Y(y)); endmodule"
        };
        let parsed = read_gv_from_str(text)?;
        let mut sta = IncrementalSta::new(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            &library,
            options().timing,
        )?;
        let before = sta.score();
        let cells_before = sta.cells.clone();
        let replacement = library.cells.iter().position(|c| c.name == "BUF1").unwrap();
        assert!(!sta.trial(0, replacement, usize::MAX, None, |_| true)?);
        assert_eq!(sta.score(), before);
        assert_eq!(sta.cells, cells_before);
    }
    Ok(())
}

#[test]
fn budget_sweeps_preserve_timing_and_area_guards() -> Result<()> {
    let library = library()?;
    for max_propagated_instances in [0, 1, 10, 100, 1000] {
        let mut parsed = read_gv_from_str(
            "module m(input clk, input d, output q); wire a,b; DFF1 launch(.CLK(clk),.D(d),.Q(a)); BUF1 comb(.A(a),.Y(b)); DFF1 capture(.CLK(clk),.D(b),.Q(q)); endmodule",
        )?;
        let options = ResizeOptions {
            max_propagated_instances,
            ..options()
        };
        let report = resize(&mut parsed, &library, &options)?;
        assert!(report.final_delay.within(report.initial_delay));
        assert!(report.final_delay.within(report.upsized_delay));
        assert!(report.final_area <= report.initial_area * (1.0 + options.max_area_growth));
    }
    Ok(())
}
