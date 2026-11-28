// SPDX-License-Identifier: Apache-2.0

//! Compare structural const-match elaboration with a const-if chain.

use std::error::Error;

use xlsynth_bx::{ResetBehavior, compile, empty_package};
use xlsynth_pir::block_eval::{BlockEvaluator, RegisterInit};
use xlsynth_pir::ir::Package;
use xlsynth_pir::random_inputs::generate_flat_bitvector_argument_sets_from_seed;
use xlsynth_pir::{IrBits, IrValue};

use crate::input_seed;

/// Pattern partitions exercised against both a const-if chain and a host model.
#[derive(Clone, Copy)]
enum Patterns {
    OverlappingRanges,
    Alternatives,
    TupleBindings,
}

/// The selected hardware path for one lane.
#[derive(Clone, Copy)]
enum Lane {
    Capture,
    Invert,
    Bypass,
}

impl Patterns {
    fn expressions(self) -> (&'static str, &'static str) {
        match self {
            Self::OverlappingRanges => (
                r#"const match tag {
                    u32:0..u32:2 => capture(xs[index]),
                    u32:1..u32:4 => invert(xs[index]),
                    _ => (xs[index], !xs[index])
                }"#,
                r#"const if tag < u32:2 { capture(xs[index]) }
                   else if tag < u32:4 { invert(xs[index]) }
                   else { (xs[index], !xs[index]) }"#,
            ),
            Self::Alternatives => (
                r#"const match tag {
                    0 | 3 => capture(xs[index]),
                    1 | 4 => invert(xs[index]),
                    _ => (xs[index], !xs[index])
                }"#,
                r#"const if tag == u32:0 || tag == u32:3 { capture(xs[index]) }
                   else if tag == u32:1 || tag == u32:4 { invert(xs[index]) }
                   else { (xs[index], !xs[index]) }"#,
            ),
            Self::TupleBindings => (
                r#"const match (tag, index) {
                    (0, lane) => capture(xs[lane]),
                    (1, lane) => invert(xs[lane]),
                    _ => (xs[index], !xs[index])
                }"#,
                r#"const if tag == u32:0 { capture(xs[index]) }
                   else if tag == u32:1 { invert(xs[index]) }
                   else { (xs[index], !xs[index]) }"#,
            ),
        }
    }

    fn lane(self, tag: usize) -> Lane {
        match (self, tag) {
            (Self::OverlappingRanges, 0 | 1)
            | (Self::Alternatives, 0 | 3)
            | (Self::TupleBindings, 0) => Lane::Capture,
            (Self::OverlappingRanges, 2 | 3)
            | (Self::Alternatives, 1 | 4)
            | (Self::TupleBindings, 1) => Lane::Invert,
            _ => Lane::Bypass,
        }
    }
}

/// Check each lowered lane's child target and compare the instance names of
/// the two elaborations, including skipped iterations.
fn check_instances(package: &Package, selected: &[Lane]) -> Result<(), String> {
    let top = package.get_block("compare").ok_or("missing compare")?;
    let [matched, chained] = top.instantiations.as_slice() else {
        return Err(format!(
            "expected two comparison instances, got {:?}",
            top.instantiations
        ));
    };
    let instances = |name: &str| -> Result<Vec<(&str, &str)>, String> {
        let block = package
            .get_block(name)
            .ok_or_else(|| format!("missing {name}"))?;
        Ok(block
            .instantiations
            .iter()
            .map(|i| (i.name.as_str(), i.block.as_str()))
            .collect())
    };
    let matched = instances(&matched.block)?;
    let chained = instances(&chained.block)?;
    let targets: Vec<_> = matched.iter().map(|(_, target)| *target).collect();
    let expected: Vec<_> = selected
        .iter()
        .filter_map(|lane| match lane {
            Lane::Capture => Some("capture"),
            Lane::Invert => Some("invert"),
            Lane::Bypass => None,
        })
        .collect();
    if matched != chained || targets != expected {
        return Err(format!(
            "expected child targets {expected:?}, got match {matched:?}, if {chained:?}"
        ));
    }
    Ok(())
}

/// Model the bypass and the two registers independently, checking all lane
/// outputs on each cycle before advancing their state.
fn check_cycles(
    package: &Package,
    selected: &[Lane],
    width: usize,
    active_low: bool,
    seed: u64,
) -> Result<(), Box<dyn Error>> {
    let eval = BlockEvaluator::prepare(package, "compare")?;
    let mut state = eval.initial_state(RegisterInit::AllZeros)?;
    let zero = IrBits::make_ubits(width, 0)?;
    let one = IrBits::make_ubits(width, 1)?;
    let mut registers = vec![zero.clone(); selected.len()];
    let cycles =
        generate_flat_bitvector_argument_sets_from_seed(&vec![width; selected.len()], seed, 6);
    for (cycle, words) in cycles.iter().enumerate() {
        let reset = cycle == 0 || cycle == 3;
        let xs = IrValue::make_array(&words.iter().map(IrValue::from_bits).collect::<Vec<_>>())?;
        let inputs = [IrValue::bool(reset ^ active_low), xs];
        let step = eval.step(&state, &inputs)?;
        let mut expected = Vec::with_capacity(selected.len());
        for (i, lane) in selected.iter().enumerate() {
            let word = &words[(i + 1) % selected.len()];
            let out = match lane {
                Lane::Capture => std::mem::replace(
                    &mut registers[i],
                    if reset { zero.clone() } else { word.clone() },
                ),
                Lane::Invert => std::mem::replace(
                    &mut registers[i],
                    if reset { one.clone() } else { word.not() },
                ),
                Lane::Bypass => word.clone(),
            };
            expected.push(IrValue::make_tuple(&[
                IrValue::from_bits(&out),
                IrValue::from_bits(&out.not()),
            ]));
        }
        let expected = IrValue::make_array(&expected)?;
        let [matched, chained] = step.outputs.as_slice() else {
            return Err(format!("expected two output arrays, got {:?}", step.outputs).into());
        };
        if matched != chained || matched != &expected {
            return Err(format!("cycle {cycle}, reset {reset}, inputs {inputs:?}: expected {expected}, match {matched}, if {chained}").into());
        }
        state = step.next_state;
    }
    Ok(())
}

/// Generate one pair of parametric blocks that select children in a block
/// let, an indexed fold, or a structural map.
pub fn check_structural_const_match(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let placement = bytes.next().unwrap() % 4;
    let patterns = match bytes.next().unwrap() % 3 {
        0 => Patterns::OverlappingRanges,
        1 => Patterns::Alternatives,
        _ => Patterns::TupleBindings,
    };
    let width = 1 + usize::from(bytes.next().unwrap() % 129);
    let flags = usize::from(bytes.next().unwrap());
    let shift = flags % 5;
    let lanes = if placement == 0 {
        1
    } else {
        1 + (flags / 5) % 5
    };
    let active_low = flags & 128 != 0;
    let (matched, chained) = patterns.expressions();
    let mut source = format!(
        r#"#![allow(block_reset_signal)]
type Word = uN[{width}];
type Pair = (Word, Word);
type Lanes = Word[{lanes}];
type Pairs = Pair[{lanes}];
block capture(clk, x: Word) -> Pair {{ reg q: Word = Word:0; q := x; (q, !q) }}
block invert(clk, x: Word) -> Pair {{ reg q: Word = Word:1; q := !x; (q, !q) }}
"#
    );
    for (name, expr) in [("matched", matched), ("chained", chained)] {
        let bindings =
            format!("let tag = (i + SHIFT) % u32:5; let index = (i + u32:1) % u32:{lanes};");
        let body = match placement {
            0 => format!("let i = u32:0; {bindings} let pair: Pair = {expr}; [pair]"),
            1 => format!(
                "unroll_for! (i, acc) in u32:0..u32:{lanes} {{ {bindings} let pair = {expr}; update(acc, i, pair) }}(zero!<Pairs>())"
            ),
            2 => format!(
                "unroll_map! i in u32:0..u32:{lanes} {{ {bindings} let (a, b) = {expr}; (a, b) }}"
            ),
            _ => format!("unroll_map! i in u32:0..u32:{lanes} {{ {bindings} {expr} }}"),
        };
        source.push_str(&format!("#[allow(combinational_block)]\nblock {name}<SHIFT: u32>(clk, xs: Lanes) -> Pairs {{ {body} }}\n"));
    }
    source.push_str(&format!(
        r#"
pub block compare(clk, rst, xs: Lanes) -> (Pairs, Pairs) {{
    let m = matched<u32:{shift}>(xs);
    let c = chained<u32:{shift}>(xs);
    (m, c)
}}"#
    ));
    let context = |e: String| format!("{e}\nBX:\n{source}");
    let reset = ResetBehavior {
        asynchronous: false,
        active_low,
    };
    let package =
        compile(&source, empty_package("structural_const_match"), reset).map_err(&context)?;
    let selected = (0..lanes)
        .map(|i| patterns.lane((i + shift) % 5))
        .collect::<Vec<_>>();
    check_instances(&package, &selected).map_err(&context)?;
    check_cycles(&package, &selected, width, active_low, input_seed(data))
        .map_err(|e| context(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::check_structural_const_match;

    #[test]
    fn structural_matches_agree_with_const_if_over_reset_and_data_edges() -> Result<(), String> {
        for placement in 0..4 {
            for family in 0..3 {
                // The one-lane block visits a register and the bypass. Five
                // unrolled lanes expose every arm and repeated child targets.
                for flags in [20, 22, 24, 148] {
                    check_structural_const_match(&[
                        placement,
                        family,
                        32 * placement + family,
                        flags,
                        47,
                        198,
                    ])?;
                }
            }
        }
        Ok(())
    }
}
