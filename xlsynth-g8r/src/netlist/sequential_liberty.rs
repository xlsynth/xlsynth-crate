// SPDX-License-Identifier: Apache-2.0

//! Shared Liberty interpretation for synchronous netlist projection.

use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, SequentialKind};

/// One supported Liberty FF clock, including its active-edge polarity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SequentialClockSpec {
    pub pin_name: String,
    pub is_negated: bool,
}

/// Liberty metadata needed to project one mapped FF into a transition AIG.
#[derive(Debug, Clone)]
pub(crate) struct GvEvalSequentialCellSpec {
    pub state_var: String,
    pub complementary_state_var: Option<String>,
    pub next_state: Term,
    pub next_state_text: String,
    pub clock: SequentialClockSpec,
}

/// Resolves the one supported FF clock expression for a Liberty cell.
///
/// One sequential transition represents one active clock edge. Positive-edge
/// and negative-edge FFs are both representable, but callers must reject a
/// design that mixes active-edge polarities.
pub(crate) fn get_sequential_ff_clock_spec(
    cell: &Cell,
    library: &Library,
) -> Result<Option<SequentialClockSpec>, String> {
    cell.validate_functional_semantics()?;
    let Some(seq) = cell.sequential.first() else {
        return Ok(None);
    };
    if cell.sequential.len() != 1 {
        return Err(format!(
            "cell '{}' has {} sequential entries; sequential netlist projection supports exactly one",
            cell.name,
            cell.sequential.len()
        ));
    }
    if seq.kind != SequentialKind::Ff as i32 {
        if seq.kind == SequentialKind::Latch as i32 {
            return Err(format!(
                "cell '{}' is a level-sensitive latch. Sequential netlist projection requires edge-triggered FFs",
                cell.name
            ));
        }
        return Err(format!(
            "cell '{}' uses unsupported sequential kind {:?}",
            cell.name, seq.kind
        ));
    }

    let clock_from_seq = if seq.clock_expr.is_empty() {
        None
    } else {
        let (pin_name, is_negated) = parse_simple_clock_expr(&seq.clock_expr)
            .map_err(|error| format!("cell '{}': {error}", cell.name))?;
        Some(SequentialClockSpec {
            pin_name,
            is_negated,
        })
    };
    let clock_pin_candidates = cell
        .pins
        .iter()
        .filter(|pin| pin.is_clocking_pin)
        .map(|pin| library.resolve_string(&pin.name).to_string())
        .collect::<Vec<_>>();
    let clock_from_pin = match clock_pin_candidates.as_slice() {
        [] => None,
        [pin_name] => Some(pin_name.clone()),
        _ => {
            return Err(format!(
                "cell '{}' has ambiguous clocking pins {:?}",
                cell.name, clock_pin_candidates
            ));
        }
    };

    if let (Some(from_seq), Some(from_pin)) = (&clock_from_seq, &clock_from_pin)
        && from_seq.pin_name.as_str() != from_pin.as_str()
    {
        return Err(format!(
            "cell '{}' clock pin mismatch: seq '{}' vs pin '{}'",
            cell.name, from_seq.pin_name, from_pin
        ));
    }

    match (clock_from_seq, clock_from_pin) {
        (Some(spec), _) => Ok(Some(spec)),
        (None, Some(pin_name)) => Ok(Some(SequentialClockSpec {
            pin_name,
            is_negated: false,
        })),
        (None, None) => Err(format!(
            "cell '{}' sequential entry missing clock expression",
            cell.name
        )),
    }
}

/// Resolves the stricter synchronous FF subset accepted by sequential gv-eval.
pub(crate) fn get_gv_eval_sequential_cell_spec(
    cell: &Cell,
    library: &Library,
) -> Result<Option<GvEvalSequentialCellSpec>, String> {
    let Some(clock) = get_sequential_ff_clock_spec(cell, library)? else {
        return Ok(None);
    };
    let seq = cell
        .sequential
        .first()
        .expect("clock resolution found one sequential entry");
    if seq.clock_expr.is_empty() {
        return Err(format!(
            "cell '{}' identifies clock pin '{}' but its Liberty FF entry has no clock_expr. Specify the active clock edge to project this FF",
            cell.name, clock.pin_name
        ));
    }
    for (control, expression) in [("clear", &seq.clear_expr), ("preset", &seq.preset_expr)] {
        if !expression.is_empty() {
            return Err(format!(
                "cell '{}' uses asynchronous {control} '{expression}'. Sequential netlist projection requires synchronous FF behavior",
                cell.name
            ));
        }
    }
    if seq.state_var.is_empty() {
        return Err(format!(
            "cell '{}' FF entry is missing state_var",
            cell.name
        ));
    }
    if seq.next_state.is_empty() {
        return Err(format!(
            "cell '{}' FF entry is missing next_state",
            cell.name
        ));
    }
    let next_state = parse_formula(&seq.next_state).map_err(|error| {
        format!(
            r#"failed to parse next_state for cell '{}' (next_state: "{}"): {}"#,
            cell.name, seq.next_state, error
        )
    })?;
    Ok(Some(GvEvalSequentialCellSpec {
        state_var: seq.state_var.clone(),
        complementary_state_var: seq
            .complementary_state_var
            .as_ref()
            .filter(|name| !name.is_empty())
            .cloned(),
        next_state,
        next_state_text: seq.next_state.clone(),
        clock,
    }))
}

fn parse_simple_clock_expr(expr: &str) -> Result<(String, bool), String> {
    let term = parse_formula(expr).map_err(|error| error.to_string())?;
    match term {
        Term::Input(name) => Ok((name, false)),
        Term::Negate(inner) => match *inner {
            Term::Input(name) => Ok((name, true)),
            _ => Err(format!(
                "clock expression '{}' is not a simple input or negated input",
                expr
            )),
        },
        _ => Err(format!(
            "clock expression '{}' is not a simple input or negated input",
            expr
        )),
    }
}

#[cfg(test)]
mod import_tests {
    use super::*;
    use crate::liberty::model::{library_from_proto, library_to_proto};
    use crate::liberty::parser::parse_liberty_files;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Exercise real importing rather than hand-constructing incomplete cell
    /// metadata.
    fn imported(extra_pin: &str, seq_extra: &str, output_extra: &str) -> Library {
        let mut file = NamedTempFile::new().unwrap();
        write!(
            file,
            r#"library(test) {{ cell(DFF) {{
            pin(D) {{ direction: input; }} pin(CLK) {{ clock: true; }}
            pin(Q) {{ direction: output; function: "IQ"; {output_extra} }}
            {extra_pin}
            ff(IQ,IQN) {{ clocked_on: "CLK"; next_state: "D"; {seq_extra} }}
        }} }}"#
        )
        .unwrap();
        library_from_proto(library_to_proto(parse_liberty_files(&[file.path()]).unwrap()).unwrap())
            .unwrap()
    }

    #[test]
    fn ordinary_ff_survives_import_and_proto_conversion() {
        let library = imported("", "", "");
        assert_eq!(
            get_gv_eval_sequential_cell_spec(&library.cells[0], &library)
                .unwrap()
                .unwrap()
                .clock
                .pin_name,
            "CLK"
        );
    }

    #[test]
    fn imported_secondary_clock_and_tristate_cells_are_rejected() {
        for (extra_pin, seq_extra, output_extra) in [
            (
                "pin(CLK2) { direction: input; }",
                "clocked_on_also: \"CLK2\";",
                "",
            ),
            ("pin(CLK2) { direction: input; clock: true; }", "", ""),
            ("pin(OE) { direction: input; }", "", "three_state: \"!OE\";"),
        ] {
            let library = imported(extra_pin, seq_extra, output_extra);
            assert!(get_gv_eval_sequential_cell_spec(&library.cells[0], &library).is_err());
        }
    }
}
