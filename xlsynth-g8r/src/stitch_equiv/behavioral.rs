// SPDX-License-Identifier: Apache-2.0

//! Data-port behavior of the resolved XLS functions and stitch boundaries.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::Hash;
use std::ops::Range;

use xlsynth::ir_package::IrPackage;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{self, Binop, NodePayload, Unop};
use xlsynth_prover::prover::translate::ir_to_smt_data;
use xlsynth_prover::prover::types::{FnInputs, IrTypedBitVec, ProverFn};
use xlsynth_prover::solver::{BitVec, Solver};

use super::stage_functions;
use super::stage_functions::named_inputs;
use crate::aig::SequentialPipelineOptions;
use crate::aig::sequential_pipeline::StitchedGatePipeline;

/// Data Q words in parameter order followed by the optional valid Q.
struct RegisterLayer {
    data: Range<usize>,
    valid: Option<usize>,
}

impl RegisterLayer {
    /// Allocates the layer in the same order as the stitch wrapper.
    fn allocate(next: &mut usize, data_words: usize, has_valid: bool) -> Self {
        let data = *next..*next + data_words;
        let valid = has_valid.then_some(data.end);
        *next = data.end + usize::from(has_valid);
        Self { data, valid }
    }

    /// Packs effective D in logical register order. Data holds during a bubble
    /// while the valid flop loads every cycle, with synchronous reset priority.
    fn capture<S: Solver>(
        &self,
        solver: &mut S,
        state: &[BitVec<S::Term>],
        data: &BitVec<S::Term>,
        valid: Option<&BitVec<S::Term>>,
        reset_inactive: Option<&BitVec<S::Term>>,
    ) -> BitVec<S::Term> {
        let Some(valid) = valid else {
            return data.clone();
        };
        let held = solver.concat_many(state[self.data.clone()].iter().rev().collect());
        let next_data = solver.ite(valid, data, &held);
        let next_valid = match reset_inactive {
            Some(inactive) => solver.and(valid, inactive),
            None => valid.clone(),
        };
        solver.concat(&next_valid, &next_data)
    }
}

/// One DSLX stage and the logical Q words that supply its parameters.
struct Stage {
    name: String,
    ir_name: String,
    input_layer: Option<RegisterLayer>,
    obligation_width: usize,
}

/// Original stage functions and the register layout of their stitch wrapper.
pub(crate) struct BehavioralPipeline {
    package: ir::Package,
    stages: Vec<Stage>,
    output_layer: Option<RegisterLayer>,
    boundary: SequentialPipelineOptions,
    register_count: usize,
}

/// Rejects operations without a value encoding in the function SMT translator.
fn check_smt_support<'a>(
    package: &'a ir::Package,
    func: &'a ir::Fn,
    checked: &mut HashSet<&'a str>,
) -> Result<(), String> {
    if !checked.insert(&func.name) {
        return Ok(());
    }
    for node in &func.nodes {
        let zero_width = |r| func.get_node_ty(r).bit_count() == 0;
        let unsupported = |detail: &str| {
            format!(
                "SMT translation does not support {}{detail} in function '{}'",
                node.payload.get_operator(),
                func.name
            )
        };
        match &node.payload {
            NodePayload::Binop(Binop::Smulp | Binop::Umulp, _, _) => {
                return Err(unsupported(""));
            }
            NodePayload::DynamicBitSlice { width: 0, .. } => {
                return Err(unsupported(" with a zero-width result"));
            }
            NodePayload::Unop(Unop::OrReduce | Unop::AndReduce | Unop::XorReduce, arg)
                if zero_width(*arg) =>
            {
                return Err(unsupported(" with a zero-width operand"));
            }
            NodePayload::Sel { cases, .. } if cases.is_empty() => {
                return Err(unsupported(" with an empty case list"));
            }
            NodePayload::Sel { selector, .. } if zero_width(*selector) => {
                return Err(unsupported(" with a zero-width selector"));
            }
            NodePayload::ArrayIndex { array, indices, .. }
            | NodePayload::ArrayUpdate { array, indices, .. } => {
                let mut ty = func.get_node_ty(*array);
                let is_update = matches!(&node.payload, NodePayload::ArrayUpdate { .. });
                for (offset, index) in indices.iter().enumerate() {
                    let dimension = offset + 1;
                    let ir::Type::Array(arr) = ty else {
                        return Err(unsupported(" with more indices than array dimensions"));
                    };
                    if arr.element_count == 0 {
                        return Err(unsupported(&format!(
                            " when dimension {dimension} is empty"
                        )));
                    }
                    if is_update && zero_width(*index) {
                        return Err(unsupported(&format!(
                            " with a zero-width index for dimension {dimension}"
                        )));
                    }
                    ty = &arr.element_type;
                }
            }
            NodePayload::ArraySlice { array, start, .. }
                if zero_width(*start)
                    && matches!(func.get_node_ty(*array), ir::Type::Array(arr) if arr.element_type.bit_count() <= 1) =>
            {
                return Err(unsupported(
                    " with a zero-width start and elements of at most one bit",
                ));
            }
            NodePayload::Invoke { to_apply, .. }
            | NodePayload::CountedFor { body: to_apply, .. } => {
                let callee = package.get_fn(to_apply).ok_or_else(|| {
                    format!("missing IR callee '{to_apply}' in function '{}'", func.name)
                })?;
                check_smt_support(package, callee, checked)?;
            }
            _ => {
                // Other nodes need no additional stitch-specific preflight.
            }
        }
    }
    Ok(())
}

impl BehavioralPipeline {
    /// Loads the original stage declarations and their transitive callees.
    /// Unrelated functions, procs and channels have no stage data behavior.
    pub(crate) fn from_xls(
        source: &IrPackage,
        names: &[(String, String)],
        boundary: &SequentialPipelineOptions,
    ) -> Result<Self, String> {
        let package = stage_functions::load_stage_package(source, names)?;
        Self::new(package, names, boundary)
    }

    /// Resolves all stage interfaces before constructing any solver terms.
    pub(crate) fn new(
        package: ir::Package,
        names: &[(String, String)],
        boundary: &SequentialPipelineOptions,
    ) -> Result<Self, String> {
        stage_functions::validate_stage_package(&package, names)?;
        let mut stages = Vec::with_capacity(names.len());
        let mut register_count = 0;
        let has_valid = boundary.input_valid_signal.is_some();
        let mut previous_width = None;
        let mut checked = HashSet::new();
        for (index, (name, ir_name)) in names.iter().enumerate() {
            let func = package
                .get_fn(ir_name)
                .ok_or_else(|| format!("stage '{name}': no IR function '{ir_name}'"))?;
            check_smt_support(&package, func, &mut checked)
                .map_err(|e| format!("stage '{name}': {e}"))?;
            let input_width: usize = func.param_nodes().map(|p| p.ty.bit_count()).sum();
            if let Some(width) = previous_width
                && width != input_width
            {
                return Err(format!(
                    "stage '{name}': expected {input_width} input bits, previous stage returns {width}"
                ));
            }
            let input_layer = (index != 0 || boundary.flop_inputs).then(|| {
                RegisterLayer::allocate(&mut register_count, func.params.len(), has_valid)
            });
            let output_width = func.ret_ty.bit_count();
            previous_width = Some(output_width);
            // Every stage has either D bits or an unregistered data output.
            // The last stage with output flops checks both D and visible Q.
            let mut obligation_width = output_width;
            if index == 0 && boundary.flop_inputs {
                obligation_width += input_width + usize::from(has_valid);
            }
            if index + 1 < names.len() || boundary.flop_outputs {
                obligation_width += usize::from(has_valid);
            }
            if index + 1 == names.len() {
                if boundary.flop_outputs {
                    obligation_width += output_width;
                }
                obligation_width += usize::from(boundary.output_valid_signal.is_some());
            }
            stages.push(Stage {
                name: name.clone(),
                ir_name: ir_name.clone(),
                input_layer,
                obligation_width,
            });
        }
        let output_layer = boundary
            .flop_outputs
            .then(|| RegisterLayer::allocate(&mut register_count, 1, has_valid));
        Ok(Self {
            package,
            stages,
            output_layer,
            boundary: boundary.clone(),
            register_count,
        })
    }

    /// Builds the ABI and state layout from IR signatures alone. Data outputs
    /// are placeholders and must never be used to evaluate or prove behavior.
    pub(crate) fn interface_layout(&self) -> Result<StitchedGatePipeline, String> {
        stage_functions::interface_layout(&self.resolve_functions()?, &self.boundary)
    }

    /// Resolves the proof stages for interface construction and interpreter
    /// replay.
    fn resolve_functions(&self) -> Result<Vec<stage_functions::StageFunction<'_>>, String> {
        stage_functions::resolve_functions(
            &self.package,
            self.stages
                .iter()
                .map(|stage| (stage.name.as_str(), stage.ir_name.as_str())),
        )
    }

    /// Returns the ordered D and visible output width of a stage obligation.
    pub(super) fn stage_width(&self, index: usize) -> Result<usize, String> {
        self.stages
            .get(index)
            .map(|s| s.obligation_width)
            .ok_or_else(|| format!("missing behavioral stage index {index}"))
    }

    /// Gets a stage's verified IR function, retaining its package for calls.
    fn function(&self, stage: &Stage) -> Result<&ir::Fn, String> {
        self.package.get_fn(&stage.ir_name).ok_or_else(|| {
            format!(
                "stage '{}': missing IR function '{}'",
                stage.name, stage.ir_name
            )
        })
    }

    /// Translates a stage directly against its caller's ordered parameter
    /// terms.
    fn translate<S: Solver>(
        &self,
        solver: &mut S,
        stage: &Stage,
        func: &ir::Fn,
        args: &[BitVec<S::Term>],
    ) -> Result<BitVec<S::Term>, String>
    where
        S::Term: Eq + Hash,
    {
        if args.len() != func.params.len() {
            return Err(format!(
                "stage '{}': expected {} parameters, got {}",
                stage.name,
                func.params.len(),
                args.len()
            ));
        }
        let mut inputs = HashMap::new();
        for (param, arg) in func.param_nodes().zip(args) {
            if arg.get_width() != param.ty.bit_count() {
                return Err(format!(
                    "stage '{}': parameter '{}' expects {} bits, got {}",
                    stage.name,
                    param.param_name(),
                    param.ty.bit_count(),
                    arg.get_width()
                ));
            }
            inputs.insert(
                param.param_name().to_string(),
                IrTypedBitVec {
                    ir_type: &param.ty,
                    bitvec: arg.clone(),
                },
            );
        }
        let bound = FnInputs {
            prover_fn: ProverFn::new(func, Some(&self.package)),
            inputs,
        };
        Ok(ir_to_smt_data(solver, &bound))
    }

    /// Computes the expected D words in logical register order, followed by
    /// the external data output on the last stage. Word zero is at the LSB.
    pub(super) fn stage_expected<S: Solver>(
        &self,
        solver: &mut S,
        index: usize,
        inputs: &BTreeMap<String, BitVec<S::Term>>,
        state: &[BitVec<S::Term>],
    ) -> Result<BitVec<S::Term>, String>
    where
        S::Term: Eq + Hash,
    {
        let stage = self
            .stages
            .get(index)
            .ok_or_else(|| format!("missing behavioral stage index {index}"))?;
        let func = self.function(stage)?;
        if state.len() != self.register_count {
            return Err(format!(
                "stage '{}': expected {} logical registers, got {}",
                stage.name,
                self.register_count,
                state.len()
            ));
        }
        let external = if index == 0 {
            named_inputs(&stage.name, func, inputs, BitVec::get_width)?
        } else {
            Vec::new()
        };
        let external_valid = self
            .boundary
            .input_valid_signal
            .as_ref()
            .map(|name| &inputs[name]);
        let reset_inactive = self.boundary.reset_signal.as_ref().map(|name| {
            if self.boundary.reset_active_low {
                inputs[name].clone()
            } else {
                solver.not(&inputs[name])
            }
        });

        // Each stage owns the layer following it. Stage zero additionally
        // owns any wrapper input flops, including a nullary stage's valid flop.
        let mut expected = BitVec::ZeroWidth;
        let (args, valid) = if let Some(layer) = &stage.input_layer {
            if index == 0 {
                let data = solver.concat_many(external.iter().rev().collect());
                expected = layer.capture(
                    solver,
                    state,
                    &data,
                    external_valid,
                    reset_inactive.as_ref(),
                );
            }
            (&state[layer.data.clone()], layer.valid.map(|q| &state[q]))
        } else {
            (&external[..], external_valid)
        };
        let result = self.translate(solver, stage, func, args)?;
        let next = self.stages.get(index + 1);
        let capture = next
            .and_then(|s| s.input_layer.as_ref())
            .or(self.output_layer.as_ref());
        if let Some(layer) = capture {
            let d = layer.capture(solver, state, &result, valid, reset_inactive.as_ref());
            expected = solver.concat(&d, &expected);
        }
        if next.is_none() {
            let (output, output_valid) = match &self.output_layer {
                Some(layer) => (&state[layer.data.start], layer.valid.map(|q| &state[q])),
                None => (&result, valid),
            };
            expected = solver.concat(output, &expected);
            if self.boundary.output_valid_signal.is_some() {
                let output_valid = output_valid.ok_or("output valid requires input valid")?;
                expected = solver.concat(output_valid, &expected);
            }
        }
        Ok(expected)
    }

    /// Composes the behavioral stage functions on the data accepted in frame
    /// zero.
    pub(super) fn compose_smt<S: Solver>(
        &self,
        solver: &mut S,
        inputs: &BTreeMap<String, BitVec<S::Term>>,
    ) -> Result<BitVec<S::Term>, String>
    where
        S::Term: Eq + Hash,
    {
        let mut previous = None;
        for stage in &self.stages {
            let func = self.function(stage)?;
            let args = if let Some(value) = previous.as_ref() {
                // The RTL wrapper slices the previous result from the LSB up
                // in destination parameter order, including multi-arg stages.
                let mut offset = 0;
                func.param_nodes()
                    .map(|p| {
                        let width = p.ty.bit_count();
                        let arg = solver.slice(value, offset, width);
                        offset += width;
                        arg
                    })
                    .collect()
            } else {
                named_inputs(&stage.name, func, inputs, BitVec::get_width)?
            };
            previous = Some(self.translate(solver, stage, func, &args)?);
        }
        previous.ok_or_else(|| "behavioral pipeline needs at least one stage".into())
    }

    /// Replays the post-fill data value using the PIR interpreter and packed
    /// port layout.
    pub(super) fn compose_concrete(
        &self,
        inputs: &BTreeMap<String, IrBits>,
    ) -> Result<IrBits, String> {
        stage_functions::compose_concrete(&self.package, &self.resolve_functions()?, inputs)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;
    use std::path::Path;

    use super::*;
    use crate::aig::ClockPort;
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_prover::solver::{
        Response,
        bitwuzla::{Bitwuzla, BitwuzlaOptions},
    };

    fn free_running(flop_inputs: bool, flop_outputs: bool) -> SequentialPipelineOptions {
        SequentialPipelineOptions {
            name: "pipe".into(),
            clock: ClockPort { name: "clk".into() },
            flop_inputs,
            flop_outputs,
            input_valid_signal: None,
            output_valid_signal: None,
            reset_signal: None,
            reset_active_low: false,
        }
    }

    const AGGREGATE_PIPELINE: &str = r#"package aggregate

fn pack(x: (bits[2][2], bits[1]) id=1, en: bits[1] id=2) -> (bits[1], bits[2][2]) {
    lanes: bits[2][2] = tuple_index(x, index=0, id=3)
    flag: bits[1] = tuple_index(x, index=1, id=4)
    active: bits[2][2] = gate(en, lanes, id=5)
    ret output: (bits[1], bits[2][2]) = tuple(flag, active, id=6)
}

fn read(a: bits[1] id=7, b: bits[2][2] id=8) -> bits[3] {
    index: bits[1] = literal(value=0, id=9)
    low_lane: bits[2] = array_index(b, indices=[index], id=10)
    ret output: bits[3] = concat(a, low_lane, id=11)
}
"#;

    #[test]
    fn interpreter_replays_nested_aggregate_and_multi_parameter_layout() {
        let package = Parser::new(AGGREGATE_PIPELINE).parse_package().unwrap();
        let names = [
            ("pack".into(), "pack".into()),
            ("read".into(), "read".into()),
        ];
        let pipeline = BehavioralPipeline::new(package, &names, &free_running(true, true)).unwrap();
        let mut inputs = BTreeMap::from([
            ("x".into(), IrBits::make_ubits(5, 0b10011).unwrap()),
            ("en".into(), IrBits::bool(true)),
        ]);
        // Tuple tails and array element zero occupy the low bits. pack
        // returns 1_1001. read receives a=1 and b=[0, 3].
        assert_eq!(
            pipeline.compose_concrete(&inputs).unwrap(),
            IrBits::make_ubits(3, 0b100).unwrap()
        );
        inputs.insert("en".into(), IrBits::bool(false));
        assert_eq!(pipeline.compose_concrete(&inputs).unwrap(), IrBits::zero(3));
    }

    #[test]
    fn staged_terms_use_free_boundary_q_and_compose_uses_frame_zero_inputs() {
        let package = Parser::new(AGGREGATE_PIPELINE).parse_package().unwrap();
        let names = [
            ("pack".into(), "pack".into()),
            ("read".into(), "read".into()),
        ];
        let pipeline = BehavioralPipeline::new(package, &names, &free_running(true, true)).unwrap();
        let mut solver = Bitwuzla::new(&BitwuzlaOptions::new()).unwrap();
        let inputs = BTreeMap::from([
            ("x".into(), solver.numerical(5, 0b10011)),
            ("en".into(), solver.numerical(1, 0)),
        ]);
        let state = [
            solver.numerical(5, 0b10011),
            solver.numerical(1, 1),
            solver.numerical(1, 0),
            solver.numerical(4, 0b1011),
            solver.numerical(3, 6),
        ];
        let first = pipeline
            .stage_expected(&mut solver, 0, &inputs, &state)
            .unwrap();
        let last = pipeline
            .stage_expected(&mut solver, 1, &inputs, &state)
            .unwrap();
        let whole = pipeline.compose_smt(&mut solver, &inputs).unwrap();
        let mut differences = Vec::new();
        for (actual, width, expected) in [
            (first, 11, 0b11001_0_10011),
            (last, 6, 0b110_011),
            (whole, 3, 0),
        ] {
            let expected = solver.numerical(width, expected);
            differences.push(solver.ne(&actual, &expected));
        }
        let any_difference = solver.or_many(differences.iter().collect());
        solver.assert(&any_difference).unwrap();
        assert_eq!(solver.check().unwrap(), Response::Unsat);
    }

    #[test]
    fn repeated_helper_calls_share_one_evaluation_per_argument_list() {
        const DEPTH: usize = 25;
        let mut dslx = String::from("fn f0(x: bool) -> bool { !x }\n");
        for level in 1..=DEPTH {
            let prev = level - 1;
            writeln!(
                dslx,
                "fn f{level}(x: bool) -> bool {{ f{prev}(x) | f{prev}(x) }}"
            )
            .unwrap();
        }
        writeln!(
            dslx,
            "fn stage(x: bool) -> u2 {{ f{DEPTH}(x) ++ f{DEPTH}(!x) }}"
        )
        .unwrap();
        let source =
            xlsynth::convert_dslx_to_ir(&dslx, Path::new("calls.x"), &Default::default()).unwrap();
        let pipeline = BehavioralPipeline::from_xls(
            &source.ir,
            &[("stage".into(), "__calls__stage".into())],
            &free_running(false, false),
        )
        .unwrap();
        for (correct, expected) in [(true, Response::Unsat), (false, Response::Sat)] {
            let mut solver = Bitwuzla::new(&BitwuzlaOptions::new()).unwrap();
            let x = solver.declare("x", 1).unwrap();
            let nx = solver.not(&x);
            let candidate = if correct {
                solver.concat(&nx, &x)
            } else {
                solver.concat(&x, &nx)
            };
            let result = pipeline
                .compose_smt(&mut solver, &BTreeMap::from([("x".into(), x)]))
                .unwrap();
            let mismatch = solver.ne(&candidate, &result);
            solver.assert(&mismatch).unwrap();
            assert_eq!(solver.check().unwrap(), expected);
        }
        for (x, output) in [(false, 0b10), (true, 0b01)] {
            assert_eq!(
                pipeline
                    .compose_concrete(&BTreeMap::from([("x".into(), IrBits::bool(x))]))
                    .unwrap(),
                IrBits::make_ubits(2, output).unwrap()
            );
        }
    }

    #[test]
    fn reports_zero_width_selector_with_stage_and_function() {
        let ir = r#"package selectors
fn choose(x: bits[4] id=1, which: bits[0] id=2) -> bits[4] {
    ret value: bits[4] = sel(which, cases=[x], id=3)
}
"#;
        let package = Parser::new(ir).parse_package().unwrap();
        let error = BehavioralPipeline::new(
            package,
            &[("pipe_cycle0".into(), "choose".into())],
            &free_running(false, false),
        )
        .err()
        .unwrap();
        assert_eq!(
            error,
            "stage 'pipe_cycle0': SMT translation does not support sel with a zero-width selector in function 'choose'"
        );
    }

    #[test]
    fn rejects_unsupported_callee_with_dslx_stage_name() {
        let ir = r#"package bad

fn helper(x: bits[4] id=1) -> (bits[4], bits[4]) {
    ret product: (bits[4], bits[4]) = umulp(x, x, id=2)
}
fn __pkg__entry(x: bits[4] id=3) -> (bits[4], bits[4]) {
    ret result: (bits[4], bits[4]) = invoke(x, to_apply=helper, id=4)
}
"#;
        let package = Parser::new(ir).parse_package().unwrap();
        let error = BehavioralPipeline::new(
            package,
            &[("entry".into(), "__pkg__entry".into())],
            &free_running(false, false),
        )
        .err()
        .unwrap();
        assert_eq!(
            error,
            "stage 'entry': SMT translation does not support umulp in function 'helper'"
        );
    }
}
