// SPDX-License-Identifier: Apache-2.0

//! Concrete transaction checking against the behavioral stages and mapped
//! gates.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::num::NonZeroUsize;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use xlsynth::ir_package::IrPackage;
use xlsynth_pir::ir::{Binop, NodePayload, Package, PackageMember};
use xlsynth_pir::random_inputs::generate_uniform_irbits_with_rng;
use xlsynth_pir::{IrBits, NamedIrValueSet};

use super::stage_functions::{
    compose_concrete, interface_layout, resolve_functions, validate_stage_package,
};
use super::{StitchObservation, output_name, validate_interfaces};
use crate::aig::SequentialPipelineOptions;
use crate::aig_sim::sequential::{SequentialSimulator, SequentialState};
use crate::dslx_stitch_pipeline::stage_package::load_stage_package;
use crate::netlist::gv_eval::LabeledSequentialNetlistAig;

/// One named record per cycle, or a deterministic generator of such records.
pub enum StitchVectors {
    Named(Vec<NamedIrValueSet>),
    Random { count: NonZeroUsize, seed: u64 },
}

/// Number of input rows and defined output observations in a completed check.
#[derive(Debug, PartialEq, Eq)]
pub struct StitchVectorCounts {
    pub rows: usize,
    pub data: usize,
    pub valid: usize,
}

/// Originating input record whose transaction reaches the observation cycle.
#[derive(Debug, PartialEq, Eq)]
pub struct StitchInputRow {
    pub index: usize,
    pub inputs: BTreeMap<String, IrBits>,
}

/// The first disagreement, with the observation inputs and originating
/// transaction, if any.
#[derive(Debug, PartialEq, Eq)]
pub struct StitchVectorMismatch {
    pub cycle: usize,
    pub output: String,
    pub expected: IrBits,
    pub actual: IrBits,
    pub source_row: Option<StitchInputRow>,
    pub observation_inputs: BTreeMap<String, IrBits>,
}

impl fmt::Display for StitchVectorMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "mismatch at cycle {} on '{}': expected {}, got {}",
            self.cycle, self.output, self.expected, self.actual
        )?;
        if let Some(row) = &self.source_row {
            writeln!(f, "Source row {} inputs {:?}", row.index, row.inputs)?;
        } else {
            writeln!(f, "Before pipeline fill")?;
        }
        write!(f, "Observation inputs {:?}", self.observation_inputs)
    }
}

/// Outcome of comparing the supplied stream of concrete inputs.
#[derive(Debug, PartialEq, Eq)]
pub enum StitchVectorResult {
    Checked(StitchVectorCounts),
    Mismatch(StitchVectorMismatch),
}

type Frame = BTreeMap<String, IrBits>;

impl StitchVectors {
    /// Binds or generates frames on demand so large random runs need only a
    /// latency window.
    fn into_frames<'a>(
        self,
        widths: &'a BTreeMap<&str, usize>,
        controls: StitchObservation<'a>,
        latency: usize,
    ) -> Box<dyn ExactSizeIterator<Item = Result<Frame, String>> + 'a> {
        match self {
            Self::Named(rows) => Box::new(
                rows.into_iter()
                    .enumerate()
                    .map(move |(cycle, row)| named_frame(row, widths, cycle)),
            ),
            Self::Random { count, seed } => {
                let count = count.get();
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                Box::new((0..count).map(move |i| {
                    Ok(random_frame(
                        widths,
                        controls,
                        latency,
                        i + 1 == count,
                        &mut rng,
                    ))
                }))
            }
        }
    }
}

/// Checks that the supplied row binds exactly the packed ports in the stitch
/// ABI.
fn named_frame(
    row: NamedIrValueSet,
    widths: &BTreeMap<&str, usize>,
    cycle: usize,
) -> Result<Frame, String> {
    let mut frame = BTreeMap::new();
    for entry in row.into_entries() {
        let width = widths
            .get(entry.name.as_str())
            .ok_or_else(|| format!("input row {cycle}: unknown port '{}'", entry.name))?;
        let bits = entry.value.to_bits().map_err(|e| {
            format!(
                "input row {cycle}, '{}': expected packed bits: {e}",
                entry.name
            )
        })?;
        if bits.get_bit_count() != *width {
            return Err(format!(
                "input row {cycle}, '{}': expected {width} bits, got {}",
                entry.name,
                bits.get_bit_count()
            ));
        }
        frame.insert(entry.name, bits);
    }
    if let Some(name) = widths.keys().find(|name| !frame.contains_key(**name)) {
        return Err(format!("input row {cycle}: missing port '{name}'"));
    }
    Ok(frame)
}

/// Generates reproducible data and bubbles. Reset frequency scales with latency
/// so transactions can retire, and the last row is always accepted to guarantee
/// a data comparison after drain.
fn random_frame(
    widths: &BTreeMap<&str, usize>,
    controls: StitchObservation<'_>,
    latency: usize,
    last: bool,
    rng: &mut Xoshiro256PlusPlus,
) -> Frame {
    widths
        .iter()
        .map(|(&name, &width)| {
            let value = if Some(name) == controls.reset {
                let asserted =
                    !last && rng.gen_range(0..latency.saturating_add(1).max(8) as u64) == 0;
                IrBits::bool(asserted != controls.reset_active_low)
            } else if last && Some(name) == controls.input_valid {
                IrBits::bool(true)
            } else {
                generate_uniform_irbits_with_rng(rng, width)
            };
            (name.to_owned(), value)
        })
        .collect()
}

/// Rejects partial products because the XLS value contract only specifies their
/// sum.
fn check_concrete_support(package: &Package) -> Result<(), String> {
    for member in &package.members {
        let PackageMember::Function(func) = member else {
            // Stage calls only evaluate functions in the behavioral package.
            continue;
        };
        for node in &func.nodes {
            if matches!(
                node.payload,
                NodePayload::Binop(Binop::Smulp | Binop::Umulp, _, _)
            ) {
                return Err(format!(
                    "stitch-check-vectors does not support {} in function '{}': XLS specifies only the sum of the partial products. The interpreter and mapped logic can produce different component bits",
                    node.payload.get_operator(),
                    func.name
                ));
            }
        }
    }
    Ok(())
}

/// Interprets the DSLX data transaction that should retire at each observation,
/// including drain.
pub(crate) fn check_stitch_vectors(
    source: &IrPackage,
    stages: &[(String, String)],
    boundary: &SequentialPipelineOptions,
    latency: usize,
    mapped: &LabeledSequentialNetlistAig,
    vectors: StitchVectors,
) -> Result<StitchVectorResult, String> {
    let package = load_stage_package(source, stages)?;
    validate_stage_package(&package, stages)?;
    check_concrete_support(&package)?;
    let functions = resolve_functions(
        &package,
        stages
            .iter()
            .map(|(name, ir_name)| (name.as_str(), ir_name.as_str())),
    )?;
    let layout = interface_layout(&functions, boundary)?;
    let widths = validate_interfaces(&layout.design, mapped)?.inputs;
    let controls = StitchObservation::from(boundary);
    let mut rows = vectors.into_frames(&widths, controls, latency);
    let count = rows.len();
    if count == 0 {
        return Err("no input rows".into());
    }
    let cycle_count = count
        .checked_add(latency)
        .ok_or("input count plus pipeline latency overflows")?;
    let drain: Frame = widths
        .iter()
        .map(|(&name, &width)| {
            let value = if Some(name) == controls.reset {
                IrBits::bool(controls.reset_active_low)
            } else {
                IrBits::zero(width)
            };
            (name.to_owned(), value)
        })
        .collect();
    let design = &mapped.sequential_gate_fn;
    let mut sim = SequentialSimulator::new(design, SequentialState::all_zeros(design))?;
    let outputs = design
        .outputs
        .iter()
        .enumerate()
        .map(|(i, id)| (output_name(&design.transition.outputs[id.index()].name), i))
        .collect::<BTreeMap<_, _>>();
    let mut counts = StitchVectorCounts {
        rows: count,
        data: 0,
        valid: 0,
    };
    let mut window = VecDeque::new();
    let mut last_reset_edge = None;
    for cycle in 0..cycle_count {
        let frame = rows.next().transpose()?.unwrap_or_else(|| drain.clone());
        let inputs = design
            .inputs
            .iter()
            .map(|id| frame[&design.transition.inputs[id.index()].name].clone())
            .collect::<Vec<_>>();
        let actual = sim.step(&inputs)?;
        let reset_now = controls.reset_asserted(&frame);
        window.push_back(frame);
        let source_index = cycle.checked_sub(latency);
        let source_frame = source_index.map(|_| &window[0]);
        // The current reset level takes effect at the edge after this
        // observation.
        let reset_since_source =
            last_reset_edge.is_some_and(|edge| edge >= cycle.saturating_sub(latency));
        if let Some(valid) = controls.valid_for_source(source_frame, reset_since_source) {
            let mismatch = |name: &str, expected: IrBits| {
                StitchVectorResult::Mismatch(StitchVectorMismatch {
                    cycle,
                    output: name.to_owned(),
                    expected,
                    actual: actual[outputs[name]].clone(),
                    source_row: source_index.map(|index| StitchInputRow {
                        index,
                        inputs: window[0].clone(),
                    }),
                    observation_inputs: window.back().expect("current frame is present").clone(),
                })
            };
            if let Some(name) = controls.output_valid {
                counts.valid += 1;
                let expected = IrBits::bool(valid);
                if actual[outputs[name]] != expected {
                    return Ok(mismatch(name, expected));
                }
            }
            if valid {
                counts.data += 1;
                let expected = compose_concrete(&package, &functions, &window[0])?;
                if actual[outputs["out"]] != expected {
                    return Ok(mismatch("out", expected));
                }
            }
        }
        if reset_now {
            last_reset_edge = Some(cycle);
        }
        if window.len() > latency {
            window.pop_front();
        }
    }
    if counts.data == 0 && counts.valid == 0 {
        return Err("the input stream produced no defined output observations".into());
    }
    Ok(StitchVectorResult::Checked(counts))
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_pir::{IrValue, NamedIrValue};

    use super::*;

    #[test]
    fn rejects_unspecified_partial_product_components_in_helpers() {
        for op in ["umulp", "smulp"] {
            let source = format!(
                r#"package arithmetic
fn helper(x: bits[4] id=1) -> (bits[4], bits[4]) {{
    ret pieces: (bits[4], bits[4]) = {op}(x, x, id=2)
}}
fn stage(x: bits[4] id=3) -> (bits[4], bits[4]) {{
    ret pieces: (bits[4], bits[4]) = invoke(x, to_apply=helper, id=4)
}}"#
            );
            let package = Parser::new(&source).parse_package().unwrap();
            assert_eq!(
                check_concrete_support(&package).unwrap_err(),
                format!(
                    "stitch-check-vectors does not support {op} in function 'helper': XLS specifies only the sum of the partial products. The interpreter and mapped logic can produce different component bits"
                )
            );
        }
    }

    #[test]
    fn named_rows_require_exact_packed_port_bindings() {
        let widths = BTreeMap::from([("x", 3)]);
        for (values, expected) in [
            (vec![], "input row 7: missing port 'x'"),
            (vec![("y", 3)], "input row 7: unknown port 'y'"),
            (vec![("x", 2)], "input row 7, 'x': expected 3 bits, got 2"),
        ] {
            let row = NamedIrValueSet::new(
                values
                    .into_iter()
                    .map(|(name, width)| NamedIrValue {
                        name: name.to_owned(),
                        value: IrValue::Bits(IrBits::zero(width)),
                    })
                    .collect(),
            )
            .unwrap();
            assert_eq!(named_frame(row, &widths, 7).unwrap_err(), expected);
        }
    }

    #[test]
    fn random_stream_guarantees_a_surviving_last_transaction_for_both_reset_polarities() {
        let widths = BTreeMap::from([("x", 129), ("vin", 1), ("reset", 1)]);
        for active_low in [false, true] {
            let controls = StitchObservation {
                input_valid: Some("vin"),
                reset: Some("reset"),
                reset_active_low: active_low,
                ..StitchObservation::default()
            };
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(31415);
            let mut accepted = 0;
            let mut resets = 0;
            for _ in 0..128 {
                let frame = random_frame(&widths, controls, 4, false, &mut rng);
                if controls.reset_asserted(&frame) {
                    resets += 1;
                }
                if !frame["vin"].is_zero() {
                    accepted += 1;
                }
            }
            assert!(accepted > 0 && accepted < 128);
            assert!(resets > 0 && resets < 128);
            let final_row = random_frame(&widths, controls, 4, true, &mut rng);
            assert!(!controls.reset_asserted(&final_row));
            assert!(!final_row["vin"].is_zero());
        }
    }
}
