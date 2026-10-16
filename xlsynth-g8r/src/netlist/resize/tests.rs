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
    for kind in ["BUF", "INV", "DFF", "AND"] {
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
            text.push_str("}}");
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
    let mut candidates = sta.critical_instances(3, usize::MAX);
    candidates.sort_unstable();
    assert_eq!(candidates, vec![1, 2, 3]);
    assert_eq!(sta.critical_instances(2, usize::MAX), vec![3]);
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
    for iteration in 0..250 {
        let i = rng.gen_range(0..sta.cells.len());
        let alternatives = &families.alternatives[sta.cells[i]];
        let replacement = alternatives[rng.gen_range(0..alternatives.len())];
        let accept = iteration % 3 != 0;
        let old = sta.score();
        let old_cells = sta.cells.clone();
        assert_eq!(sta.trial(i, replacement, usize::MAX, |_| accept)?, accept);
        if !accept {
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
    assert!(!sta.trial(1, larger, usize::MAX, |_| false)?);
    assert_eq!(sta.score(), original_score);
    assert_eq!(sta.cells, original_cells);
    let budget = sta.propagated + 1;
    assert!(!sta.trial(1, larger, budget, |_| true)?);
    assert!(sta.budget_exhausted);
    assert_eq!(sta.score(), original_score);
    assert_eq!(sta.cells, original_cells);
    // A subsequent trial must not inherit any rejected or aborted work.
    assert!(sta.trial(1, larger, usize::MAX, |_| true)?);
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
    assert!(sta.trial(1, bad, usize::MAX, |_| true).is_err());
    assert_eq!(sta.score(), before);
    let good = library.cells.iter().position(|c| c.name == "BUF2").unwrap();
    assert!(sta.trial(1, good, usize::MAX, |_| true)?);
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
        assert!(!sta.trial(0, larger, usize::MAX, |_| true)?, "{limit}");
        assert_eq!(sta.score(), before);
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
