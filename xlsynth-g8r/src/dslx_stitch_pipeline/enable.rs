// SPDX-License-Identifier: Apache-2.0

//! Nominal DSLX register banks at internal stitch boundaries.

use super::layout::{EnabledBank, PipelineLayout};
use crate::aig::import::import_gate_fn;
use crate::aig::topo::extract_cone;
use crate::aig::{AigBitVector, AigRef, GateFn};
use crate::aig_sim::gate_sim::PreparedGateSim;
use crate::gate_builder::{GateBuilder, GateBuilderOptions, ReductionKind};
use crate::gatify::ir2gate::{GatifyOptions, gatify};
use crate::prove_gate_fn_equiv_sat::{GateFormalOptions, find_cone_witness};
use std::collections::BTreeMap;
use xlsynth::{IrPackage, XlsynthError, dslx, optimize_ir};
use xlsynth_pir::ir::{self, NodePayload};
use xlsynth_pir::ir_fn_rm_asserts::remove_asserts_from_fn;
use xlsynth_pir::ir_parser::Parser;

pub(super) const STITCH_MODULE: &str = r#"#![feature(generics)]
pub struct Enable<T: type> { enabled: bool, data: T }
"#;

/// A stage's combinational data graph and extra roots used by diagnostics.
struct ProofStage {
    graph: GateFn,
    effect_roots: Vec<AigRef>,
}

/// Exposes the predicates and operands of diagnostics to bit-level liveness.
/// The ordinary result stays in the low bits, in its original packed order.
fn expose_effect_operands(function: &mut ir::Fn) {
    let mut fields = Vec::new();
    for node in &function.nodes {
        match &node.payload {
            NodePayload::Assert { activate, .. } => fields.push(*activate),
            NodePayload::Trace {
                activated,
                operands,
                ..
            } => {
                fields.push(*activated);
                fields.extend_from_slice(operands);
            }
            NodePayload::Cover { predicate, .. } => fields.push(*predicate),
            _ => { /* Ordinary operations contribute through their consumers. */ }
        }
    }
    if fields.is_empty() {
        return;
    }
    fields.push(
        function
            .ret_node_ref
            .expect("parsed stage has a return value"),
    );
    function.ret_ty = ir::Type::Tuple(
        fields
            .iter()
            .map(|&r| Box::new(function.get_node_ty(r).clone()))
            .collect(),
    );
    function.ret_node_ref = Some(ir::NodeRef {
        index: function.nodes.len(),
    });
    let result = ir::Node {
        text_id: function.nodes.iter().map(|n| n.text_id).max().unwrap_or(0) + 1,
        ty: function.ret_ty.clone(),
        payload: NodePayload::Tuple(fields),
        pos: None,
        name: None,
    };
    function.nodes.push(result);
}

/// Checks one transaction symbolically, sharing its inputs between the ordinary
/// pipeline and a copy with fresh independent values in *all* disabled banks.
pub(super) fn validate_banks(
    ir: &IrPackage,
    stages: &[(String, String)],
    layout: &mut PipelineLayout,
) -> Result<(), XlsynthError> {
    if layout.stage_inputs.iter().all(|b| b.banks.is_empty()) {
        return Ok(());
    }
    let mut graphs = Vec::with_capacity(stages.len());
    for (name, mangled) in stages {
        let opt = optimize_ir(ir, mangled)?;
        let text = opt.get_function(mangled)?.to_ir_string()?;
        let mut function = Parser::new(&text)
            .parse_function_declaration()
            .map_err(|e| XlsynthError(format!("enable proof for {name}: {e}")))?;
        let data_width = function.ret_ty.bit_count();
        expose_effect_operands(&mut function);
        let data = remove_asserts_from_fn(&function).rewritten_fn;
        let mut graph = gatify(
            &data,
            GatifyOptions {
                unsafe_gatify_gate_operation: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .map_err(|e| XlsynthError(format!("enable proof could not lower {name}: {e}")))?
        .gate_fn;
        let inputs = &layout.stage_inputs[graphs.len()];
        if graph
            .inputs
            .iter()
            .map(|p| p.get_bit_count())
            .ne(inputs.ports.iter().map(|p| p.bits.len()))
            || graph.outputs.len() != 1
            || graph.outputs[0].get_bit_count() != function.ret_ty.bit_count()
            || data_width != layout.after_stage(graphs.len()).width()
        {
            return Err(XlsynthError(format!(
                "enable proof: unexpected packed output for {name}"
            )));
        }
        let result = &mut graph.outputs[0].bit_vector;
        let effect_roots = result
            .iter_lsb_to_msb()
            .skip(data_width)
            .map(|op| op.node)
            .collect();
        *result = result.get_lsb_slice(0, data_width);
        graphs.push(ProofStage {
            graph,
            effect_roots,
        });
    }
    mark_live_enables(&graphs, layout);
    prove_insensitivity(&graphs, layout)
}

/// Propagates packed-bit demand from the output and every bank write predicate.
fn mark_live_enables(stages: &[ProofStage], layout: &mut PipelineLayout) {
    let mut live = vec![true; layout.output.width()];
    for (stage, inputs) in stages.iter().zip(&mut layout.stage_inputs).rev() {
        let graph = &stage.graph;
        let roots = graph.outputs[0]
            .bit_vector
            .iter_lsb_to_msb()
            .zip(&live)
            .filter(|(_, needed)| **needed)
            .map(|(bit, _)| bit.node)
            .chain(stage.effect_roots.iter().copied())
            .collect::<Vec<_>>();
        let (_, cone_inputs) = extract_cone(&roots, &graph.gates);
        live = graph
            .inputs
            .iter()
            .flat_map(|p| p.bit_vector.iter_lsb_to_msb())
            .map(|bit| cone_inputs.contains(&bit.node))
            .collect();
        for bank in &mut inputs.banks {
            bank.carry_enable = live[bank.data.end];
            // The preceding stage must still compute the combinational write
            // enable.
            live[bank.data.end] |= !bank.data.is_empty();
        }
    }
}

/// Imports a stage with the same LSB-first port binding as the emitted wrapper.
fn evaluate(
    stage: &GateFn,
    packed: &AigBitVector,
    builder: &mut GateBuilder,
) -> Result<AigBitVector, XlsynthError> {
    let mut start = 0;
    let args = stage
        .inputs
        .iter()
        .map(|input| {
            let bits = packed.get_lsb_slice(start, input.get_bit_count());
            start += input.get_bit_count();
            bits
        })
        .collect::<Vec<_>>();
    Ok(import_gate_fn(stage, &args, builder)
        .map_err(XlsynthError)?
        .remove(0))
}

/// Proves the final packed outputs equal, or decodes and replays a SAT witness.
fn prove_insensitivity(stages: &[ProofStage], layout: &PipelineLayout) -> Result<(), XlsynthError> {
    let mut builder = GateBuilder::new("enable_insensitivity".into(), GateBuilderOptions::opt());
    let mut bits = Vec::new();
    for port in &stages[0].graph.inputs {
        let input = builder.add_input(port.name.clone(), port.get_bit_count());
        bits.extend(input.iter_lsb_to_msb().copied());
    }
    let mut reference = AigBitVector::from_lsb_is_index_0(&bits);
    let mut candidate = reference.clone();
    for (i, stage) in stages.iter().enumerate() {
        reference = evaluate(&stage.graph, &reference, &mut builder)?;
        candidate = evaluate(&stage.graph, &candidate, &mut builder)?;
        let mut bits: Vec<_> = candidate.iter_lsb_to_msb().copied().collect();
        for bank in &layout.after_stage(i).banks {
            let enabled = bits[bank.data.end];
            let arbitrary = builder.add_input(
                format!("stage[{i}].{}.disabled_data", bank.path),
                bank.data.len(),
            );
            for (dst, fresh) in bits[bank.data.clone()]
                .iter_mut()
                .zip(arbitrary.iter_lsb_to_msb())
            {
                *dst = builder.add_mux2(enabled, *dst, *fresh);
            }
            builder.add_output(
                format!("stage[{i}].{}.enabled", bank.path),
                AigBitVector::from_bit(enabled),
            );
            if !bank.carry_enable {
                bits[bank.data.end] = builder.get_false();
            }
        }
        candidate = AigBitVector::from_lsb_is_index_0(&bits);
    }
    let different_bits = builder.add_xor_vec(&reference, &candidate);
    let differs = builder.add_or_reduce(&different_bits, ReductionKind::Tree);
    builder.add_output("reference.out".into(), reference);
    builder.add_output("candidate.out".into(), candidate);
    let graph = builder.build();
    let Some(witness) = find_cone_witness(&graph, differs, GateFormalOptions::default())
        .map_err(|error| XlsynthError(format!("enable insensitivity proof: {error}")))?
    else {
        return Ok(());
    };
    let values: Vec<_> = graph
        .inputs
        .iter()
        .map(|port| witness.decode_inputs(&port.bit_vector))
        .collect();
    let outputs = PreparedGateSim::new(&graph).eval_outputs(&values);
    if outputs[outputs.len() - 2] == outputs[outputs.len() - 1] {
        return Err(XlsynthError(
            "enable insensitivity counterexample did not reproduce the output mismatch".into(),
        ));
    }
    let details: Vec<_> = graph
        .inputs
        .iter()
        .map(|p| &p.name)
        .zip(&values)
        .chain(graph.outputs.iter().map(|p| &p.name).zip(&outputs))
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    Err(XlsynthError(format!(
        "disabled stitch::Enable payload changes the final output: {}",
        details.join(", ")
    )))
}

/// Records nominal annotations before struct names are erased by IR conversion.
pub(super) fn collect_banks(
    module: &dslx::TypecheckedModule,
    marker: &dslx::StructDef,
    stages: &[(String, String)],
    layout: &mut PipelineLayout,
) -> Result<(), XlsynthError> {
    let info = module.get_type_info();
    let ast = module.get_module();
    let functions: BTreeMap<_, _> = (0..ast.get_member_count())
        .filter_map(|i| {
            if let Some(dslx::MatchableModuleMember::Function(f)) = ast.get_member(i).to_matchable()
            {
                Some((f.get_identifier(), f))
            } else {
                None
            }
        })
        .collect();
    let boundaries = layout
        .stage_inputs
        .iter_mut()
        .skip(1)
        .chain([&mut layout.output]);
    for (i, ((name, _), boundary)) in stages.iter().zip(boundaries).enumerate() {
        let func = &functions[name];
        let Some(annotation) = func.get_return_type() else {
            // A DSLX function without a return annotation returns unit.
            continue;
        };
        let ty = info
            .get_type_for_type_annotation(&annotation)
            .ok_or_else(|| {
                XlsynthError(format!("missing concrete return type for stage '{name}'"))
            })?;
        if ty.get_total_bit_count()? != boundary.width() {
            return Err(XlsynthError(format!(
                "DSLX and packed IR boundary widths disagree for stage '{name}'"
            )));
        }
        let mut banks = Vec::new();
        let mut visitor = BankVisitor {
            info: &info,
            marker,
            banks: &mut banks,
        };
        visitor.visit(&ty, Some(annotation), &format!("{name}.out"), 0, 0)?;
        if i + 1 < stages.len() {
            // Only interstage registers may hold stale data. External results
            // obey the same placement rule but expose their complete value.
            banks.sort_by_key(|bank| bank.data.start);
            boundary.banks = banks;
        }
    }
    Ok(())
}

/// Nominal annotations in one producer result, in packed-bit coordinates.
struct BankVisitor<'a> {
    info: &'a dslx::TypeInfo,
    marker: &'a dslx::StructDef,
    banks: &'a mut Vec<EnabledBank>,
}

/// Follows written aliases to a tuple or array annotation.
fn unalias(mut annotation: dslx::TypeAnnotation) -> dslx::TypeAnnotation {
    while let Some(alias) = annotation
        .to_type_ref_type_annotation()
        .and_then(|r| r.get_type_ref().get_type_definition().to_type_alias())
    {
        annotation = alias.get_type_annotation();
    }
    annotation
}

impl BankVisitor<'_> {
    /// Walks children from the LSB. An annotation in an array is always nested.
    fn visit(
        &mut self,
        ty: &dslx::Type,
        annotation: Option<dslx::TypeAnnotation>,
        path: &str,
        depth: usize,
        offset: usize,
    ) -> Result<(), XlsynthError> {
        if ty.is_bits_like().is_some() || ty.is_enum() {
            return Ok(());
        }
        if ty.is_struct() {
            let def = ty.get_struct_def()?;
            if def.is_same_definition(self.marker) {
                if depth > 1 {
                    return Err(XlsynthError(format!(
                        "nested stitch::Enable at {path}; use a whole stage result or a direct struct field or tuple element"
                    )));
                }
                let payload = ty.get_struct_member_type(1);
                self.visit(&payload, None, &format!("{path}.data"), 2, offset)?;
                self.banks.push(EnabledBank {
                    path: path.into(),
                    data: offset..offset + payload.get_total_bit_count()?,
                    carry_enable: true,
                });
            } else {
                let mut start = offset;
                for i in (0..ty.get_struct_member_count()).rev() {
                    let field = ty.get_struct_member_type(i);
                    let member = def.get_member(i);
                    // The definition of a generic field can describe a
                    // different type than its concrete
                    // instantiation. Use concrete accessors
                    // for tuples in that case.
                    let written =
                        (def.get_parametric_binding_count() == 0).then(|| member.get_type());
                    self.visit(
                        &field,
                        written,
                        &format!("{path}.{}", member.get_name()),
                        depth + 1,
                        start,
                    )?;
                    start += field.get_total_bit_count()?;
                }
            }
        } else if ty.is_array() {
            let written = annotation
                .map(unalias)
                .and_then(|a| a.to_array_type_annotation())
                .map(|a| a.get_element_type());
            self.visit(
                &ty.get_array_element_type(),
                written,
                &format!("{path}[]"),
                depth.max(1) + 1,
                offset,
            )?;
        } else {
            // The published DSLX C API has no token predicate. Its canonical
            // spelling distinguishes this leaf from zero-width aggregates.
            if ty.get_total_bit_count()? == 0 && ty.to_string()? == "token" {
                return Ok(());
            }
            let missing = || {
                XlsynthError(format!(
                    "cannot inspect the concrete aggregate at {path}; opaque tuples require libxls concrete tuple-member accessors, and DSLX semantic sums are unsupported"
                ))
            };
            let mut start = offset;
            if let Some(members) = ty.concrete_tuple_members() {
                for (i, member) in members.iter().enumerate().rev() {
                    self.visit(member, None, &format!("{path}.{i}"), depth + 1, start)?;
                    start += member.get_total_bit_count()?;
                }
            } else {
                let members = annotation
                    .map(unalias)
                    .and_then(|a| a.tuple_members())
                    .ok_or_else(missing)?;
                for (i, written) in members.into_iter().enumerate().rev() {
                    let member = self
                        .info
                        .get_type_for_type_annotation(&written)
                        .ok_or_else(missing)?;
                    self.visit(
                        &member,
                        Some(written),
                        &format!("{path}.{i}"),
                        depth + 1,
                        start,
                    )?;
                    start += member.get_total_bit_count()?;
                }
            }
            if start - offset != ty.get_total_bit_count()? {
                return Err(XlsynthError(format!(
                    "tuple member widths disagree at {path}"
                )));
            }
        }
        Ok(())
    }
}
