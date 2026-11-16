// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::liberty::parser::parse_liberty_files;
use crate::liberty_model::SequentialKind;
use crate::netlist::io::{ParsedNetlist, read_gv_from_str};
use crate::netlist::sta::analyze_combinational_max_arrival;
use std::io::Write;
use std::time::Instant;
use tempfile::NamedTempFile;

/// Small exact-binary capacitances make load comparisons independent of
/// rounding.
fn library() -> Result<Library> {
    let mut text = String::from(
        r#"library(test) {
lu_table_template(delay) { variable_1: input_net_transition; variable_2: total_output_net_capacitance; index_1("0,16"); index_2("0,16"); }
"#,
    );
    for (name, cap, resistance, slew) in [
        ("BUF", 0.25, 1.0, None),
        ("INV", 0.25, 2.0, None),
        ("STRONG", 1.0, 0.25, None),
        ("SLEW1", 0.25, 1.0, Some(1.0)),
        ("SLEW2", 0.25, 1.0, Some(2.0)),
        ("GATE", 0.25, 1.0, None),
    ] {
        let function = if name == "INV" || name == "STRONG" {
            "!A"
        } else if name == "GATE" {
            "A & B"
        } else {
            "A"
        };
        let sense = if function == "!A" {
            "negative_unate"
        } else {
            "positive_unate"
        };
        text.push_str(&format!(r#"cell({name}) {{ area: 1; pin(A) {{ direction: input; rise_capacitance: {cap}; fall_capacitance: {}; }} "#, cap * 2.0));
        if name == "GATE" {
            text.push_str("pin(B) { direction: input; capacitance: 0.25; }");
        }
        text.push_str(&format!(
            r#"pin(Y) {{ direction: output; function: "{function}"; "#
        ));
        for related in if name == "GATE" {
            vec!["A", "B"]
        } else {
            vec!["A"]
        } {
            text.push_str(&format!(r#"timing() {{ related_pin: "{related}"; timing_sense: "{sense}"; timing_type: combinational; "#));
            if name == "GATE" {
                text.push_str(&format!(
                    r#"when: "{}"; "#,
                    if related == "A" { "B" } else { "A" }
                ));
            }
            for kind in [
                "cell_rise",
                "cell_fall",
                "rise_transition",
                "fall_transition",
            ] {
                let values = if kind.ends_with("transition") && slew.is_some() {
                    [slew.unwrap(); 4]
                } else {
                    [1.0, 1.0 + 16.0 * resistance, 9.0, 9.0 + 16.0 * resistance]
                };
                text.push_str(&format!(
                    r#"{kind}(delay) {{ values("{},{}", "{},{}"); }}"#,
                    values[0], values[1], values[2], values[3]
                ));
            }
            text.push('}');
        }
        text.push('}');
        text.push('}');
    }
    text.push('}');
    let mut file = NamedTempFile::new()?;
    file.write_all(text.as_bytes())?;
    parse_liberty_files(&[file.path()]).map_err(anyhow::Error::msg)
}

fn options() -> StaOptions {
    StaOptions {
        primary_input_transition: 1.0,
        module_output_load: 0.5,
        wire_load: Default::default(),
    }
}

fn cell(lib: &Library, name: &str) -> Result<usize> {
    lib.cells
        .iter()
        .position(|c| c.name == name)
        .ok_or_else(|| anyhow!("missing test cell {name}"))
}

fn bit(p: &ParsedNetlist, name: &str) -> Result<usize> {
    p.nets
        .iter()
        .position(|n| p.interner.resolve(n.name) == Some(name))
        .ok_or_else(|| anyhow!("missing test bit {name}"))
}

/// Checks every signal, rather than only the worst endpoint.
fn compare(sta: &IncrementalSta<'_>, p: &ParsedNetlist, lib: &Library) -> Result<()> {
    compare_options(sta, p, lib, options())
}

/// Checks full/incremental agreement with a caller-selected load policy.
fn compare_options(
    sta: &IncrementalSta<'_>,
    p: &ParsedNetlist,
    lib: &Library,
    options: StaOptions,
) -> Result<()> {
    let full =
        analyze_combinational_max_arrival(&p.modules[0], &p.nets, &p.interner, lib, options)?;
    ensure!(sta.bit_count() == p.nets.len(), "bit count");
    let order = sta.topological_instances();
    ensure!(
        order.len() == sta.instance_count(),
        "traversal includes new instances"
    );
    let mut positions = vec![usize::MAX; sta.instance_count()];
    for (position, &i) in order.iter().enumerate() {
        ensure!(positions[i] == usize::MAX, "unique traversal");
        positions[i] = position;
    }
    for i in 0..sta.instance_count() {
        let inst = sta.instance(i).unwrap();
        for &bit in inst.outputs {
            let pin = sta.output_pin_index(i, bit).expect("connected output");
            let name = lib.resolve_string(&lib.cells[inst.cell].pins[pin].name);
            ensure!(
                inst.pins[name].as_slice() == [BitSource::Bit(bit)],
                "output pin identity"
            );
        }
        for (pin, sources) in inst.pins {
            if sta.lib.pin(inst.cell, pin).unwrap().direction != PinDirection::Input as i32 {
                continue;
            }
            for source in sources {
                if let BitSource::Bit(bit) = source {
                    if let Some(driver) = sta.driver_for_bit(*bit) {
                        ensure!(
                            driver == i || positions[driver] < positions[i],
                            "current traversal orders driver {driver} before consumer {i}"
                        );
                    }
                }
            }
        }
    }

    for (i, expected) in full.net_timing.iter().enumerate() {
        let actual = sta.timing_for_bit(i);
        match (actual, expected) {
            (Some(a), Some(b)) => {
                for (x, y) in [
                    (a.rise.arrival, b.rise.arrival),
                    (a.fall.arrival, b.fall.arrival),
                    (a.rise.transition, b.rise.transition),
                    (a.fall.transition, b.fall.transition),
                ] {
                    ensure!((x - y).abs() < 1e-9, "bit {i}: {a:?} != {b:?}");
                }
            }
            (None, None) => { /* Both analyses report no timing for this bit. */ }
            _ => anyhow::bail!("bit {i}: {actual:?} != {expected:?}"),
        }
    }
    ensure!(
        sta.combinational_query_diagnostics() == Some(full.timing_query_diagnostic_counts),
        "query diagnostics"
    );
    ensure!(
        (sta.score().input_to_output - full.worst_output_arrival).abs() < 1e-9,
        "endpoint"
    );
    Ok(())
}

fn network(unrelated: usize) -> String {
    let mut s = String::from(
        "module m(a,y,z); input a; output y,z; wire p,n; BUF pre(.A(a),.Y(p)); INV drive(.A(p),.Y(n)); BUF sink(.A(n),.Y(y)); BUF sibling(.A(p),.Y(z));",
    );
    for i in 0..unrelated {
        s.push_str(&format!("wire u{i}; BUF extra{i}(.A(a),.Y(u{i}));"));
    }
    s.push_str("endmodule");
    s
}

fn inserted(name: &str, lib: &Library, a: BitSource, y: usize) -> Result<TimingMutation> {
    Ok(TimingMutation::Insert(TimingInstance {
        name: name.into(),
        cell: cell(lib, "BUF")?,
        connections: vec![("A".into(), a), ("Y".into(), BitSource::Bit(y))],
    }))
}

/// Checks each launch class and individual capture against the full boundary
/// analysis.
fn compare_registered(
    sta: &IncrementalSta<'_>,
    parsed: &ParsedNetlist,
    registers: &[usize],
) -> Result<()> {
    for mode in 0..2 {
        let full = sta.lib.analyze_register_boundary_max_arrival(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            options(),
            mode == 0,
            if mode == 0 { &[] } else { registers },
        )?;
        assert_eq!(sta.bit_count(), full.net_timing.len());
        for (bit, expected) in full.net_timing.iter().enumerate() {
            let actual = sta.timing[mode][bit]
                .as_ref()
                .and_then(SignalTimingSet::as_report_signal_timing);
            assert_eq!(&actual, expected, "mode {mode}, bit {bit}");
        }
        for (endpoint, arrivals) in sta.endpoint_values() {
            if let TimingEndpoint::Capture(i) = endpoint {
                assert_eq!(
                    arrivals[mode], full.register_input_arrivals[i],
                    "mode {mode}, capture {i}"
                );
            }
        }
        let scores = sta.score().values();
        assert_eq!(scores[mode * 2], full.worst_output_arrival);
        assert_eq!(scores[mode * 2 + 1], full.worst_register_input_arrival);
    }
    Ok(())
}

// The capture changes its data source. The changed launch Q load also
// affects the sibling timing through its original wire.
//
//     before  Q +--> capture D
//               +--> sibling
//     trial   Q +--> BUF --> capture D
//               +--> sibling
#[test]
fn registered_buffer_trial_retimes_launch_siblings_and_setup_capture() -> Result<()> {
    for next_state in ["D", "!D"] {
        check_registered_buffer_trial(next_state)?;
    }
    Ok(())
}

fn check_registered_buffer_trial(next_state: &str) -> Result<()> {
    let mut lib = crate::netlist::resize::tests::library()?;
    let ff = cell(&lib, "DFF1")?;
    lib.cells[ff].sequential[0].next_state = next_state.into();
    let source = r#"module m(input clk, input d, output captured, output sibling);
      wire q;
      DFF1 capture(.CLK(clk), .D(q), .Q(captured));
      DFF1 launch(.CLK(clk), .D(d), .Q(q));
      BUF1 receiver(.A(q), .Y(sibling));
    endmodule"#;
    let original = read_gv_from_str(source)?;
    let buffered = read_gv_from_str(&source.replace(".D(q)", ".D(added)").replace(
        "endmodule",
        "wire added; BUF2 inserted(.A(q), .Y(added)); endmodule",
    ))?;
    let registers = [0, 1];
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let q = bit(&original, "q")?;
    let sibling = bit(&original, "sibling")?;
    let added = sta.bit_count();
    let before = sta.score();
    let old_load = sta.load_for_bit(q);
    let old_sibling = sta.timing[1][sibling]
        .as_ref()
        .and_then(SignalTimingSet::as_report_signal_timing)
        .expect("register-launched sibling timing");
    assert!(before.register_to_register > 0.0);
    let cycle = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Rewire {
            instance: 2,
            pin: "A".into(),
            source: BitSource::Bit(sibling),
        }],
    };
    let error = sta
        .trial_edit(&cycle, TrialOptions::default(), |s, _| {
            panic!(
                "cycle reached acceptance with register-to-output delay {} (was {})",
                s.score().register_to_output,
                before.register_to_output
            )
        })
        .unwrap_err();
    assert!(error.to_string().contains("cycle"), "{error}");
    assert_eq!(sta.instance(2).unwrap().pins["A"], [BitSource::Bit(q)]);
    compare_registered(&sta, &original, &registers)?;
    let edit = TimingEdit {
        new_bits: 1,
        mutations: vec![
            TimingMutation::Insert(TimingInstance {
                name: "inserted".into(),
                cell: cell(&lib, "BUF2")?,
                connections: vec![
                    ("A".into(), BitSource::Bit(q)),
                    ("Y".into(), BitSource::Bit(added)),
                ],
            }),
            TimingMutation::Rewire {
                instance: 0,
                pin: "D".into(),
                source: BitSource::Bit(added),
            },
        ],
    };
    // The extra buffer slows the setup path. Hitting either bound must roll
    // back the appended instance, the capture binding and all launch timing.
    for (opts, outcome) in [
        (
            TrialOptions {
                cell_budget: 0,
                ..Default::default()
            },
            TrialOutcome::PropagationLimit,
        ),
        (
            TrialOptions {
                limits: Some(BoundaryDelays {
                    register_to_output: f64::INFINITY,
                    ..before
                }),
                ..Default::default()
            },
            TrialOutcome::Rejected,
        ),
    ] {
        let result = sta.trial_edit(&edit, opts, |_, _| panic!("trial exceeded its bound"))?;
        assert_eq!(result.outcome, outcome);
        compare_registered(&sta, &original, &registers)?;
    }
    for accept in [false, true] {
        let result = sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
            assert_ne!(s.load_for_bit(q), old_load);
            let sibling = s.timing[1][sibling]
                .as_ref()
                .and_then(SignalTimingSet::as_report_signal_timing)
                .expect("register-launched sibling timing");
            assert!(sibling.rise.arrival > old_sibling.rise.arrival);
            assert!(sibling.rise.transition > old_sibling.rise.transition);
            assert!(s.score().register_to_register > before.register_to_register);
            compare_registered(s, &buffered, &registers)?;
            Ok(accept)
        })?;
        assert_eq!(
            result.outcome,
            if accept {
                TrialOutcome::Accepted
            } else {
                TrialOutcome::Rejected
            }
        );
        compare_registered(&sta, if accept { &buffered } else { &original }, &registers)?;
    }
    // Both buffers become ready when the launch register is visited. The
    // sparse insertion can place the new buffer immediately after the launch,
    // while a fresh traversal visits the lower-indexed receiver first.
    let fresh = IncrementalSta::new(
        &buffered.modules[0],
        &buffered.nets,
        &buffered.interner,
        &lib,
        options(),
    )?;
    sta.canonicalize_registered_order()?;
    assert_eq!(sta.topological_instances(), fresh.topological_instances());
    assert_eq!(sta.timing, fresh.timing);
    assert_eq!(sta.endpoint_values, fresh.endpoint_values);
    assert_eq!(sta.score(), fresh.score());
    let clock_edit = TimingEdit {
        new_bits: 0,
        mutations: vec![
            TimingMutation::Rewire {
                instance: 0,
                pin: "D".into(),
                source: BitSource::Bit(q),
            },
            TimingMutation::Rewire {
                instance: 0,
                pin: "CLK".into(),
                source: BitSource::Bit(q),
            },
        ],
    };
    let error = sta
        .trial_edit(&clock_edit, TrialOptions::default(), |_, _| Ok(true))
        .unwrap_err();
    assert!(error.is::<UnsupportedRegisteredEdit>());
    assert_eq!(sta.instance(0).unwrap().pins["D"], [BitSource::Bit(added)]);
    compare_registered(&sta, &buffered, &registers)
}

#[test]
fn capture_rewire_uses_each_instances_ties() -> Result<()> {
    let source = r#"module m(input clk, input d, input en, output q, output r);
        wire n;
        BUF1 driver(.A(d), .Y(n));
        DFF1 tied(.CLK(clk), .A(1'b0), .D(n), .Q(q));
        DFF1 live(.CLK(clk), .A(en), .D(n), .Q(r));
    endmodule"#;
    let parsed = read_gv_from_str(source)?;
    let edited = read_gv_from_str(&source.replacen(".D(n)", ".D(d)", 1))?;
    let mut lib = crate::netlist::resize::tests::library()?;
    let ff = cell(&lib, "DFF1")?;
    let select = lib.cells[cell(&lib, "BUF1")?]
        .pins
        .iter()
        .find(|pin| lib.resolve_string(&pin.name) == "A")
        .unwrap()
        .clone();
    lib.cells[ff].pins.push(select);
    lib.cells[ff].sequential[0].next_state = "!A & !D".into();
    let mut sta = IncrementalSta::new(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        &lib,
        options(),
    )?;
    let d = BitSource::Bit(bit(&parsed, "d")?);
    let edit = |instance| TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Rewire {
            instance,
            pin: "D".into(),
            source: d,
        }],
    };
    for accept in [false, true] {
        sta.trial_edit(&edit(1), TrialOptions::default(), |trial, _| {
            compare_registered(trial, &edited, &[1, 2])?;
            Ok(accept)
        })?;
        let error = sta
            .trial_edit(&edit(2), TrialOptions::default(), |_, _| Ok(true))
            .unwrap_err();
        assert!(error.is::<UnsupportedRegisteredEdit>());
        compare_registered(&sta, if accept { &edited } else { &parsed }, &[1, 2])?;
    }
    Ok(())
}

// Unsupported sequential controls must leave the session reusable.
#[test]
fn capture_rewire_requires_an_unambiguous_flip_flop_data_pin() -> Result<()> {
    let parsed = read_gv_from_str(
        r#"module m(input clk, input d, output q);
        wire n;
        BUF1 driver(.A(d), .Y(n));
        DFF1 capture(.CLK(clk), .A(clk), .D(n), .Q(q));
    endmodule"#,
    )?;
    let d = BitSource::Bit(bit(&parsed, "d")?);
    let clk = BitSource::Bit(bit(&parsed, "clk")?);
    let n = BitSource::Bit(bit(&parsed, "n")?);
    for (kind, clock_expr, clk_marked, a_marked) in [
        (SequentialKind::Ff, "CLK", true, true),
        (SequentialKind::Ff, "D", false, false),
        (SequentialKind::Latch, "CLK", true, false),
    ] {
        let mut lib = crate::netlist::resize::tests::library()?;
        let ff = cell(&lib, "DFF1")?;
        let clock = lib.cells[ff]
            .pins
            .iter()
            .position(|pin| lib.resolve_string(&pin.name) == "CLK")
            .unwrap();
        let mut aux = lib.cells[cell(&lib, "BUF1")?]
            .pins
            .iter()
            .find(|pin| lib.resolve_string(&pin.name) == "A")
            .unwrap()
            .clone();
        aux.is_clocking_pin = a_marked;
        lib.cells[ff].pins.push(aux);
        lib.cells[ff].pins[clock].is_clocking_pin = clk_marked;
        lib.cells[ff].sequential[0].kind = kind as i32;
        lib.cells[ff].sequential[0].clock_expr = clock_expr.into();
        let mut sta = IncrementalSta::new(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            &lib,
            options(),
        )?;
        let edit = TimingEdit {
            new_bits: 0,
            mutations: vec![
                TimingMutation::Rewire {
                    instance: 0,
                    pin: "A".into(),
                    source: clk,
                },
                TimingMutation::Rewire {
                    instance: 1,
                    pin: "D".into(),
                    source: d,
                },
            ],
        };
        let error = sta
            .trial_edit(&edit, TrialOptions::default(), |_, _| {
                panic!("unsupported capture rewire reached acceptance")
            })
            .unwrap_err();
        assert!(
            error.is::<UnsupportedRegisteredEdit>(),
            "{kind:?} {clock_expr}: {error}"
        );
        assert_eq!(sta.instance(0).unwrap().pins["A"], [d]);
        assert_eq!(sta.instance(1).unwrap().pins["D"], [n]);
        compare_registered(&sta, &parsed, &[1])?;
    }
    Ok(())
}

// Swapping equal-time data nets through a tran alias leaves endpoint scores
// unchanged. Resize only the left branch afterward to reveal which capture it
// actually feeds. Test both the restored and the committed graph.
#[test]
fn equal_arrival_capture_swap_tracks_the_next_edit_after_rollback_and_commit() -> Result<()> {
    let lib = crate::netlist::resize::tests::library()?;
    let source = r#"module m(input clk, input d, output x, output y);
      wire qa, qb, a, b, b_alias;
      tran(b, b_alias);
      DFF1 launch_a(.CLK(clk), .D(x), .Q(qa));
      DFF1 launch_b(.CLK(clk), .D(d), .Q(qb));
      BUF1 left(.A(qa), .Y(a));
      BUF1 right(.A(qb), .Y(b));
      DFF1 capture_x(.CLK(clk), .D(a), .Q(x));
      DFF1 capture_y(.CLK(clk), .D(b), .Q(y));
    endmodule"#;
    let swapped_text = source
        .replace(".D(a), .Q(x)", ".D(b_alias), .Q(x)")
        .replace(".D(b), .Q(y)", ".D(a), .Q(y)");
    let original = read_gv_from_str(source)?;
    let swapped = read_gv_from_str(&swapped_text)?;
    let registers = [0, 1, 4, 5];
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let a = bit(&original, "a")?;
    let b = bit(&original, "b")?;
    let b_alias = bit(&original, "b_alias")?;
    let connect = |x, y| TimingEdit {
        new_bits: 0,
        mutations: vec![
            TimingMutation::Rewire {
                instance: 4,
                pin: "D".into(),
                source: BitSource::Bit(x),
            },
            TimingMutation::Rewire {
                instance: 5,
                pin: "D".into(),
                source: BitSource::Bit(y),
            },
        ],
    };
    let resize_left = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Resize {
            instance: 2,
            cell: cell(&lib, "BUF2")?,
        }],
    };
    let before: Vec<_> = sta.endpoint_values().collect();
    for accept in [false, true] {
        sta.trial_edit(&connect(b_alias, a), TrialOptions::default(), |s, _| {
            // Both branches carry the same timing and retain equal loads.
            assert_eq!(s.endpoint_values().collect::<Vec<_>>(), before);
            compare_registered(s, &swapped, &registers)?;
            Ok(accept)
        })?;
        let (current, parsed) = if accept {
            (swapped_text.as_str(), &swapped)
        } else {
            (source, &original)
        };
        compare_registered(&sta, parsed, &registers)?;
        let resized = read_gv_from_str(&current.replace("BUF1 left", "BUF2 left"))?;
        sta.trial_edit(&resize_left, TrialOptions::default(), |s, _| {
            // Only the capture still attached to the left branch must change.
            let changed: Vec<_> = s
                .endpoint_values()
                .zip(&before)
                .filter_map(|((endpoint, values), (_, previous))| {
                    (values != *previous).then_some(endpoint)
                })
                .collect();
            assert_eq!(
                changed,
                [TimingEndpoint::Capture(if accept { 5 } else { 4 })]
            );
            compare_registered(s, &resized, &registers)?;
            Ok(accept)
        })?;
        compare_registered(&sta, if accept { &resized } else { parsed }, &registers)?;
    }
    // Swap the now unequal sources back. The data nets keep their timing and
    // loads, but both setup captures must update in this transaction.
    let resized_original = read_gv_from_str(&source.replace("BUF1 left", "BUF2 left"))?;
    sta.trial_edit(&connect(a, b), TrialOptions::default(), |s, _| {
        compare_registered(s, &resized_original, &registers)?;
        Ok(true)
    })?;
    compare_registered(&sta, &resized_original, &registers)
}

// Two sinks share an arrival but have different paths to the endpoint.
// A single slack per wire would hide which branch is urgent.
#[test]
fn tree_sink_urgency_distinguishes_paths_from_the_same_net() -> Result<()> {
    let lib = library()?;
    let p = read_gv_from_str(&network(1))?;
    let sta = IncrementalSta::new(&p.modules[0], &p.nets, &p.interner, &lib, options())?;
    let urgency = sta.pin_timing_urgency(0)?;
    // drive/A and sibling/A see the very same net arrival. The former goes
    // through a further inverter and buffer; the latter goes straight to z.
    let slow = urgency.sink_slack(1, 0).unwrap();
    let fast = urgency.sink_slack(3, 0).unwrap();
    assert!(slow.abs() < 1e-9 && fast > slow);
    assert!(urgency.required_arrival(1, 0) < urgency.required_arrival(3, 0));
    assert_eq!(urgency.sink_slack(4, 0), None); // Unused buffer output.
    assert!(sta.pin_timing_urgency(3)?.sink_slack(1, 0).is_none());
    Ok(())
}

#[test]
fn generic_insert_rewire_preview_commit_and_following_resize_match_full_sta() -> Result<()> {
    let lib = library()?;
    let text = network(0);
    let original = read_gv_from_str(&text)?;
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let n = bit(&original, "n")?;
    let added = sta.bit_count();
    let edit = TimingEdit {
        new_bits: 1,
        mutations: vec![
            inserted("inserted", &lib, BitSource::Bit(n), added)?,
            TimingMutation::Rewire {
                instance: 2,
                pin: "A".into(),
                source: BitSource::Bit(added),
            },
        ],
    };
    let expected = read_gv_from_str(
        &text
            .replace("BUF sink(.A(n)", "BUF sink(.A(added)")
            .replace(
                "endmodule",
                "wire added; BUF inserted(.A(n),.Y(added)); endmodule",
            ),
    )?;
    let preview = sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
        compare(s, &expected, &lib)?;
        Ok(false)
    })?;
    assert_eq!(preview.outcome, TrialOutcome::Rejected);
    compare(&sta, &original, &lib)?;
    let commit = sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
        compare(s, &expected, &lib)?;
        Ok(true)
    })?;
    assert_eq!(commit.outcome, TrialOutcome::Accepted);
    compare(&sta, &expected, &lib)?;
    let resized = read_gv_from_str(
        &text
            .replace("INV drive", "STRONG drive")
            .replace("BUF sink(.A(n)", "BUF sink(.A(added)")
            .replace(
                "endmodule",
                "wire added; BUF inserted(.A(n),.Y(added)); endmodule",
            ),
    )?;
    let edit = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Resize {
            instance: 1,
            cell: cell(&lib, "STRONG")?,
        }],
    };
    sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
        compare(s, &resized, &lib)?;
        Ok(true)
    })?;
    compare(&sta, &resized, &lib)?;
    Ok(())
}

#[test]
fn resizing_updates_upstream_loading_and_sibling_slew() -> Result<()> {
    let lib = library()?;
    let text = network(0);
    let original = read_gv_from_str(&text)?;
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let p = bit(&original, "p")?;
    let z = bit(&original, "z")?;
    let old_load = sta.load_for_bit(p);
    let old_sibling = sta.timing_for_bit(z);
    let expected = read_gv_from_str(&text.replace("INV drive", "STRONG drive"))?;
    let edit = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Resize {
            instance: 1,
            cell: cell(&lib, "STRONG")?,
        }],
    };
    sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
        assert_ne!(s.load_for_bit(p), old_load);
        assert_ne!(s.timing_for_bit(z), old_sibling);
        compare(s, &expected, &lib)?;
        Ok(false)
    })?;
    assert_eq!(sta.load_for_bit(p), old_load);
    assert_eq!(sta.timing_for_bit(z), old_sibling);
    Ok(())
}

#[test]
fn unchanged_arrival_with_changed_slew_still_propagates() -> Result<()> {
    let lib = library()?;
    let text = "module m(a,y); input a; output y; wire n; SLEW1 d(.A(a),.Y(n)); BUF s(.A(n),.Y(y)); endmodule";
    let original = read_gv_from_str(text)?;
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let n = bit(&original, "n")?;
    let y = bit(&original, "y")?;
    let old = sta.timing_for_bit(n).unwrap();
    let endpoint = sta.timing_for_bit(y).unwrap();
    let expected = read_gv_from_str(&text.replace("SLEW1", "SLEW2"))?;
    let edit = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Resize {
            instance: 0,
            cell: cell(&lib, "SLEW2")?,
        }],
    };
    sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
        let new = s.timing_for_bit(n).unwrap();
        assert_eq!(new.rise.arrival, old.rise.arrival);
        assert_ne!(new.rise.transition, old.rise.transition);
        assert_ne!(
            s.timing_for_bit(y).unwrap().rise.arrival,
            endpoint.rise.arrival
        );
        compare(s, &expected, &lib)?;
        Ok(true)
    })?;
    Ok(())
}

#[test]
fn errors_cycles_and_work_limits_restore_graph_and_timing() -> Result<()> {
    let lib = library()?;
    let original = read_gv_from_str(&network(0))?;
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let edit = TimingEdit {
        new_bits: 1,
        mutations: vec![TimingMutation::Resize {
            instance: 1,
            cell: cell(&lib, "STRONG")?,
        }],
    };
    assert_eq!(
        sta.trial_edit(
            &edit,
            TrialOptions {
                cell_budget: 0,
                ..Default::default()
            },
            |_, _| Ok(true)
        )?
        .outcome,
        TrialOutcome::PropagationLimit
    );
    compare(&sta, &original, &lib)?;
    assert!(
        sta.trial_edit(&edit, TrialOptions::default(), |_, _| Err(anyhow!(
            "oracle failed"
        )))
        .is_err()
    );
    compare(&sta, &original, &lib)?;
    for mutation in [
        TimingMutation::Rewire {
            instance: 1,
            pin: "MISSING".into(),
            source: BitSource::Literal(false),
        },
        TimingMutation::Resize {
            instance: 1,
            cell: usize::MAX,
        },
        TimingMutation::Rewire {
            instance: 0,
            pin: "A".into(),
            source: BitSource::Bit(bit(&original, "n")?),
        },
        TimingMutation::Rewire {
            instance: 0,
            pin: "Y".into(),
            source: BitSource::Bit(bit(&original, "a")?),
        },
    ] {
        assert!(
            sta.trial_edit(
                &TimingEdit {
                    new_bits: 0,
                    mutations: vec![mutation]
                },
                TrialOptions::default(),
                |_, _| Ok(true)
            )
            .is_err()
        );
        compare(&sta, &original, &lib)?;
    }
    Ok(())
}

#[test]
fn rewired_literals_aliases_and_conditional_coverage_match_full_sta() -> Result<()> {
    let lib = library()?;
    let text = "module m(a,y); input a; output y; wire select,alias; assign select=1'b0; assign alias=a; GATE g(.A(alias),.B(select),.Y(y)); endmodule";
    let original = read_gv_from_str(text)?;
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let expected = read_gv_from_str(&text.replace(".B(select)", ".B(1'b1)"))?;
    let edit = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Rewire {
            instance: 0,
            pin: "B".into(),
            source: BitSource::Literal(true),
        }],
    };
    sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
        compare(s, &expected, &lib)?;
        Ok(true)
    })?;
    compare(&sta, &expected, &lib)?;
    Ok(())
}

#[test]
fn replacement_can_rename_output_and_insert_replication_topology() -> Result<()> {
    let mut lib = library()?;
    let mut builder = crate::liberty_model::LibraryBuilder::from_library(lib);
    let z = builder.intern_string("Z")?;
    let strong = builder
        .cells
        .iter_mut()
        .find(|c| c.name == "STRONG")
        .unwrap();
    strong
        .pins
        .iter_mut()
        .find(|p| p.direction == PinDirection::Output as i32)
        .unwrap()
        .name = z;
    lib = builder.finish();
    let text = network(0);
    let original = read_gv_from_str(&text)?;
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let expected =
        read_gv_from_str(&text.replace("INV drive(.A(p),.Y(n))", "STRONG drive(.A(p),.Z(n))"))?;
    let edit = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Replace {
            instance: 1,
            cell: cell(&lib, "STRONG")?,
            connections: vec![
                ("A".into(), BitSource::Bit(bit(&original, "p")?)),
                ("Z".into(), BitSource::Bit(bit(&original, "n")?)),
            ],
        }],
    };
    sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
        compare(s, &expected, &lib)?;
        Ok(true)
    })?;
    let added = sta.bit_count();
    let expected = read_gv_from_str(
        &text
            .replace("INV drive(.A(p),.Y(n))", "STRONG drive(.A(p),.Z(n))")
            .replace("BUF sink(.A(n)", "BUF sink(.A(copy)")
            .replace(
                "endmodule",
                "wire copy; INV replica(.A(p),.Y(copy)); endmodule",
            ),
    )?;
    let edit = TimingEdit {
        new_bits: 1,
        mutations: vec![
            TimingMutation::Insert(TimingInstance {
                name: "replica".into(),
                cell: cell(&lib, "INV")?,
                connections: vec![
                    ("A".into(), BitSource::Bit(bit(&original, "p")?)),
                    ("Y".into(), BitSource::Bit(added)),
                ],
            }),
            TimingMutation::Rewire {
                instance: 2,
                pin: "A".into(),
                source: BitSource::Bit(added),
            },
        ],
    };
    sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
        compare(s, &expected, &lib)?;
        Ok(true)
    })?;
    Ok(())
}

#[test]
fn affected_timing_work_stays_bounded_with_unrelated_logic() -> Result<()> {
    let lib = library()?;
    let mut expected_work = None;
    for unrelated in [0, 1000, 10000, 100000] {
        let text = network(unrelated);
        let original = read_gv_from_str(&text)?;
        let mut sta = IncrementalSta::new(
            &original.modules[0],
            &original.nets,
            &original.interner,
            &lib,
            options(),
        )?;
        let expected = read_gv_from_str(&text.replace("INV drive", "STRONG drive"))?;
        let edit = TimingEdit {
            new_bits: 0,
            mutations: vec![TimingMutation::Resize {
                instance: 1,
                cell: cell(&lib, "STRONG")?,
            }],
        };
        let prepared = PreparedStaLibrary::new(&lib)?;
        prepared.analyze_combinational_max_arrival(
            &expected.modules[0],
            &expected.nets,
            &expected.interner,
            options(),
        )?;
        let mut incremental_us = Vec::new();
        let mut prepared_full_us = Vec::new();
        let work = sta.trial_edit(&edit, TrialOptions::default(), |_, _| Ok(false))?;
        for _ in 0..5 {
            let start = Instant::now();
            std::hint::black_box(sta.trial_edit(&edit, TrialOptions::default(), |_, _| Ok(false))?);
            incremental_us.push(start.elapsed().as_micros());
            let start = Instant::now();
            std::hint::black_box(prepared.analyze_combinational_max_arrival(
                &expected.modules[0],
                &expected.nets,
                &expected.interner,
                options(),
            )?);
            prepared_full_us.push(start.elapsed().as_micros());
        }
        incremental_us.sort();
        prepared_full_us.sort();
        if let Some(expected) = expected_work {
            assert_eq!(work.propagated_instances, expected);
        }
        expected_work = Some(work.propagated_instances);
        assert_eq!(work.rebuilt_instances, 1);
        assert_eq!(work.coverage_instances, 0);
        assert!(work.touched_nets <= 2);
        assert_eq!(work.reordered_instances, 0);
        sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
            compare(s, &expected, &lib)?;
            Ok(false)
        })?;
        compare(&sta, &original, &lib)?;
        eprintln!(
            "topology scaling: cells={} propagated={} rebuilt={} coverage={} incremental_us={} prepared_full_us={}",
            unrelated + 4,
            work.propagated_instances,
            work.rebuilt_instances,
            work.coverage_instances,
            incremental_us[2],
            prepared_full_us[2]
        );
    }
    Ok(())
}

#[test]
fn backward_acyclic_rewire_uses_ordering_fallback_and_rolls_back_on_callback_error() -> Result<()> {
    let lib = library()?;
    let text = network(0);
    let original = read_gv_from_str(&text)?;
    let expected = read_gv_from_str(&text.replace("INV drive(.A(p)", "INV drive(.A(z)"))?;
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let edit = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Rewire {
            instance: 1,
            pin: "A".into(),
            source: BitSource::Bit(bit(&original, "z")?),
        }],
    };
    assert!(
        sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
            compare(s, &expected, &lib)?;
            anyhow::bail!("caller rejects with an error")
        })
        .is_err()
    );
    compare(&sta, &original, &lib)?;
    let work = sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
        compare(s, &expected, &lib)?;
        Ok(true)
    })?;
    assert_eq!(work.reordered_instances, 4);
    compare(&sta, &expected, &lib)?;
    Ok(())
}

#[test]
fn unused_malformed_arc_is_rejected_like_full_sta() -> Result<()> {
    let lib = library()?;
    let index = cell(&lib, "BUF")?;
    let mut builder = crate::liberty_model::LibraryBuilder::from_library(lib);
    let mut bad = builder.cells[index].clone();
    bad.name = "BADREF".into();
    let mut unused = bad
        .pins
        .iter()
        .find(|p| p.direction == PinDirection::Output as i32)
        .ok_or_else(|| anyhow!("missing output"))?
        .clone();
    unused.name = builder.intern_string("Z").map_err(anyhow::Error::msg)?;
    unused.timing_arcs[0].related_pin = builder
        .intern_string("MISSING")
        .map_err(anyhow::Error::msg)?;
    bad.pins.push(unused);
    builder.cells.push(bad);
    let lib = builder.finish();
    let original = read_gv_from_str(&network(0))?;
    let invalid = read_gv_from_str(&network(0).replace("BUF pre", "BADREF pre"))?;
    assert!(
        analyze_combinational_max_arrival(
            &invalid.modules[0],
            &invalid.nets,
            &invalid.interner,
            &lib,
            options()
        )
        .is_err()
    );
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let edit = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Resize {
            instance: 0,
            cell: cell(&lib, "BADREF")?,
        }],
    };
    assert!(
        sta.trial_edit(&edit, TrialOptions::default(), |_, _| Ok(true))
            .is_err()
    );
    compare(&sta, &original, &lib)?;
    Ok(())
}

#[test]
fn empty_binding_stays_unconnected_when_conditional_arc_becomes_active() -> Result<()> {
    let lib = library()?;
    let text = "module m(y); output y; GATE g(.A(1'b0),.B(),.Y(y)); endmodule";
    let original = read_gv_from_str(text)?;
    let invalid = read_gv_from_str(&text.replace("1'b0", "1'b1"))?;
    assert!(
        analyze_combinational_max_arrival(
            &invalid.modules[0],
            &invalid.nets,
            &invalid.interner,
            &lib,
            options()
        )
        .is_err()
    );
    let mut sta = IncrementalSta::new(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let resize = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Resize {
            instance: 0,
            cell: cell(&lib, "GATE")?,
        }],
    };
    sta.trial_edit(&resize, TrialOptions::default(), |s, _| {
        compare(s, &original, &lib)?;
        Ok(true)
    })?;
    let active = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Rewire {
            instance: 0,
            pin: "A".into(),
            source: BitSource::Literal(true),
        }],
    };
    assert!(
        sta.trial_edit(&active, TrialOptions::default(), |_, _| Ok(true))
            .is_err()
    );
    compare(&sta, &original, &lib)?;
    // A later edit can connect the previously empty pin.
    let connect = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Rewire {
            instance: 0,
            pin: "B".into(),
            source: BitSource::Literal(true),
        }],
    };
    let connected = read_gv_from_str(&text.replace(".B()", ".B(1'b1)"))?;
    sta.trial_edit(&connect, TrialOptions::default(), |s, _| {
        compare(s, &connected, &lib)?;
        Ok(true)
    })?;
    Ok(())
}

#[test]
fn combinational_session_can_edit_a_cell_with_unused_sequential_output_arc() -> Result<()> {
    let lib = library()?;
    let index = cell(&lib, "BUF")?;
    let mut builder = crate::liberty_model::LibraryBuilder::from_library(lib);
    let mut unused = builder.cells[index]
        .pins
        .iter()
        .find(|p| p.direction == PinDirection::Output as i32)
        .ok_or_else(|| anyhow!("missing output"))?
        .clone();
    unused.name = builder.intern_string("Z").map_err(anyhow::Error::msg)?;
    unused.timing_arcs[0] = builder
        .add_timing_arc(
            "A",
            "positive_unate",
            "rising_edge",
            "",
            unused.timing_arcs[0].tables.clone(),
        )
        .map_err(anyhow::Error::msg)?;
    builder.cells[index].pins.push(unused);
    let lib = builder.finish();
    let original =
        read_gv_from_str("module m(a,b,y); input a,b; output y; BUF g(.A(a),.Y(y)); endmodule")?;
    let expected =
        read_gv_from_str("module m(a,b,y); input a,b; output y; BUF g(.A(b),.Y(y)); endmodule")?;
    let mut sta = IncrementalSta::new_combinational(
        &original.modules[0],
        &original.nets,
        &original.interner,
        &lib,
        options(),
    )?;
    let resize = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Resize {
            instance: 0,
            cell: index,
        }],
    };
    sta.trial_edit(&resize, TrialOptions::default(), |s, _| {
        compare(s, &original, &lib)?;
        Ok(true)
    })?;
    let rewire = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Rewire {
            instance: 0,
            pin: "A".into(),
            source: BitSource::Bit(bit(&original, "b")?),
        }],
    };
    sta.trial_edit(&rewire, TrialOptions::default(), |s, _| {
        compare(s, &expected, &lib)?;
        Ok(true)
    })?;
    let active = TimingEdit {
        new_bits: 0,
        mutations: vec![TimingMutation::Replace {
            instance: 0,
            cell: index,
            connections: vec![
                ("A".into(), BitSource::Bit(bit(&original, "b")?)),
                ("Z".into(), BitSource::Bit(bit(&original, "y")?)),
            ],
        }],
    };
    assert!(
        sta.trial_edit(&active, TrialOptions::default(), |_, _| Ok(true))
            .is_err()
    );
    compare(&sta, &expected, &lib)?;
    Ok(())
}

#[test]
fn affine_wire_load_counts_pins_and_tracks_fanout_edits() -> Result<()> {
    let lib = library()?;
    let text = "module m(a,n,y,z); input a; output n,y,z; wire dead; BUF drive(.A(a),.Y(n)); GATE sink(.A(n),.B(n),.Y(y)); BUF dangling(.A(a),.Y(dead)); assign z=n; endmodule";
    let p = read_gv_from_str(text)?;
    let n = bit(&p, "n")?;
    for (constant, per_sink) in [(0.0, 0.0), (0.75, 0.0), (0.0, 0.125), (0.75, 0.125)] {
        let opts = StaOptions {
            wire_load: crate::netlist::sta::WireLoadModel {
                constant_capacitance: constant,
                capacitance_per_sink: per_sink,
            },
            ..options()
        };
        let mut sta = IncrementalSta::new(&p.modules[0], &p.nets, &p.interner, &lib, opts)?;
        compare_options(&sta, &p, &lib, opts)?;
        // Two input pins on one cell and two aliased external output loads.
        assert_eq!(sta.capacitances[n].rise, 1.5 + constant + 2.0 * per_sink);
        assert_eq!(sta.capacitances[n].fall, 1.75 + constant + 2.0 * per_sink);
        assert_eq!(sta.capacitances[bit(&p, "dead")?].rise, 0.0);
        assert_eq!(sta.capacitances[bit(&p, "a")?].rise, 0.5);
        let expected = read_gv_from_str(&text.replace(".B(n)", ".B(a)"))?;
        let edit = TimingEdit {
            new_bits: 0,
            mutations: vec![TimingMutation::Rewire {
                instance: 1,
                pin: "B".into(),
                source: BitSource::Bit(bit(&p, "a")?),
            }],
        };
        for accept in [false, true] {
            sta.trial_edit(&edit, TrialOptions::default(), |s, _| {
                compare_options(s, &expected, &lib, opts)?;
                assert_eq!(s.capacitances[n].rise, 1.25 + constant + per_sink);
                Ok(accept)
            })?;
            compare_options(&sta, if accept { &expected } else { &p }, &lib, opts)?;
        }
    }
    Ok(())
}
