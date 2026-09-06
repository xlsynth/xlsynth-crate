// SPDX-License-Identifier: Apache-2.0

//! Functionality for converting a gate function into an IR function.
//!
//! This is useful for getting it "back into" XLS IR form after transforms so we
//! can test equivalence.

use std::collections::{HashMap, HashSet};
use std::iter::zip;

use crate::aig::gate::{self, AigBitVector, AigNode, AigOperand, Input, Output};
use xlsynth_pir::ir::{self, ArrayTypeData};
use xlsynth_pir::ir_builder::is_valid_identifier;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateFnInterfacePort {
    pub name: String,
    pub ty: ir::Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateFnInterfaceSchema {
    pub input_ports: Vec<GateFnInterfacePort>,
    pub output_ports: Vec<GateFnInterfacePort>,
    pub return_type: ir::Type,
}

fn type_contains_token(ty: &ir::Type) -> bool {
    match ty {
        ir::Type::Token => true,
        ir::Type::Bits(_) => false,
        ir::Type::Tuple(members) => members.iter().any(|member| type_contains_token(member)),
        ir::Type::Array(ArrayTypeData { element_type, .. }) => type_contains_token(element_type),
    }
}

fn validate_schema_port(kind: &str, port: &GateFnInterfacePort) -> Result<(), String> {
    if type_contains_token(&port.ty) {
        return Err(format!(
            "{kind} port `{}` uses token-typed schema, which is unsupported for AIGER regrouping",
            port.name
        ));
    }
    Ok(())
}

impl GateFnInterfaceSchema {
    fn validate(&self) -> Result<(), String> {
        if type_contains_token(&self.return_type) {
            return Err(
                "return type uses token-typed schema, which is unsupported for AIGER regrouping"
                    .to_string(),
            );
        }
        for port in &self.input_ports {
            validate_schema_port("input", port)?;
        }
        for port in &self.output_ports {
            validate_schema_port("output", port)?;
        }
        let output_bits: usize = self
            .output_ports
            .iter()
            .map(|port| port.ty.bit_count())
            .sum();
        if output_bits != self.return_type.bit_count() {
            return Err(format!(
                "output schema width mismatch: output ports total bits[{}] but return type is bits[{}]",
                output_bits,
                self.return_type.bit_count()
            ));
        }
        Ok(())
    }

    /// Builds a schema from an explicit function type using synthesized names.
    ///
    /// The parameter list is grouped per declared parameter. The return value
    /// is represented as a single top-level port carrying the flattened
    /// return value bitstream.
    pub fn from_function_type(function_type: &ir::FunctionType) -> Result<Self, String> {
        let input_ports = function_type
            .param_types
            .iter()
            .enumerate()
            .map(|(index, ty)| GateFnInterfacePort {
                name: format!("arg{}", index),
                ty: ty.clone(),
            })
            .collect::<Vec<GateFnInterfacePort>>();
        let output_ports = if function_type.return_type.bit_count() == 0 {
            vec![]
        } else {
            vec![GateFnInterfacePort {
                name: "ret".to_string(),
                ty: function_type.return_type.clone(),
            }]
        };
        let schema = Self {
            input_ports,
            output_ports,
            return_type: function_type.return_type.clone(),
        };
        schema.validate()?;
        Ok(schema)
    }

    /// Builds a schema from a PIR function, preserving declared parameter
    /// names while treating the return value as a single flattened top-level
    /// port.
    pub fn from_pir_fn(pir_fn: &ir::Fn) -> Result<Self, String> {
        let input_ports = pir_fn
            .param_nodes()
            .map(|param| GateFnInterfacePort {
                name: param.param_name().to_string(),
                ty: param.ty.clone(),
            })
            .collect::<Vec<GateFnInterfacePort>>();
        let output_ports = if pir_fn.ret_ty.bit_count() == 0 {
            vec![]
        } else {
            vec![GateFnInterfacePort {
                name: "ret".to_string(),
                ty: pir_fn.ret_ty.clone(),
            }]
        };
        let schema = Self {
            input_ports,
            output_ports,
            return_type: pir_fn.ret_ty.clone(),
        };
        schema.validate()?;
        Ok(schema)
    }

    /// Builds a schema from the current GateFn interface itself.
    pub fn from_gate_fn(gate_fn: &gate::GateFn) -> Result<Self, String> {
        let input_ports = gate_fn
            .inputs
            .iter()
            .map(|input| GateFnInterfacePort {
                name: input.name.clone(),
                ty: ir::Type::Bits(input.get_bit_count()),
            })
            .collect::<Vec<GateFnInterfacePort>>();
        let output_ports = gate_fn
            .outputs
            .iter()
            .map(|output| GateFnInterfacePort {
                name: output.name.clone(),
                ty: ir::Type::Bits(output.get_bit_count()),
            })
            .collect::<Vec<GateFnInterfacePort>>();
        let schema = Self {
            input_ports,
            output_ports,
            return_type: gate_fn.get_flat_type().return_type,
        };
        schema.validate()?;
        Ok(schema)
    }

    pub fn function_type(&self) -> ir::FunctionType {
        ir::FunctionType {
            param_types: self
                .input_ports
                .iter()
                .map(|port| port.ty.clone())
                .collect::<Vec<ir::Type>>(),
            return_type: self.return_type.clone(),
        }
    }
}

fn flatten_gate_port_bit_vectors_lsb_is_0(bit_vectors: &[AigBitVector]) -> Vec<AigOperand> {
    let total_bits: usize = bit_vectors.iter().map(AigBitVector::get_bit_count).sum();
    let mut flat = Vec::with_capacity(total_bits);
    for bit_vector in bit_vectors {
        for bit in bit_vector.iter_lsb_to_msb() {
            flat.push(*bit);
        }
    }
    flat
}

fn repack_inputs_from_flat_schema(
    flat_inputs_lsb_is_0: &[AigOperand],
    schema: &GateFnInterfaceSchema,
) -> Result<Vec<Input>, String> {
    let expected_bits: usize = schema
        .input_ports
        .iter()
        .map(|port| port.ty.bit_count())
        .sum::<usize>();
    if flat_inputs_lsb_is_0.len() != expected_bits {
        return Err(format!(
            "input width mismatch: raw interface has bits[{}] but schema expects bits[{}]",
            flat_inputs_lsb_is_0.len(),
            expected_bits
        ));
    }

    let mut new_inputs = Vec::with_capacity(schema.input_ports.len());
    let mut offset = 0usize;
    for port in &schema.input_ports {
        let width = port.ty.bit_count();
        let slice = &flat_inputs_lsb_is_0[offset..offset + width];
        new_inputs.push(Input {
            name: port.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(slice),
        });
        offset += width;
    }
    Ok(new_inputs)
}

fn repack_outputs_from_flat_schema(
    flat_outputs_lsb_is_0: &[AigOperand],
    schema: &GateFnInterfaceSchema,
) -> Result<Vec<Output>, String> {
    let expected_bits: usize = schema
        .output_ports
        .iter()
        .map(|port| port.ty.bit_count())
        .sum::<usize>();
    if flat_outputs_lsb_is_0.len() != expected_bits {
        return Err(format!(
            "output width mismatch: raw interface has bits[{}] but schema expects bits[{}]",
            flat_outputs_lsb_is_0.len(),
            expected_bits
        ));
    }

    let mut new_outputs = Vec::with_capacity(schema.output_ports.len());
    let mut offset = 0usize;
    for port in &schema.output_ports {
        let width = port.ty.bit_count();
        let slice = &flat_outputs_lsb_is_0[offset..offset + width];
        new_outputs.push(Output {
            name: port.name.clone(),
            bit_vector: AigBitVector::from_lsb_is_index_0(slice),
        });
        offset += width;
    }
    Ok(new_outputs)
}

/// Updates underlying input leaf names and indices to match regrouped inputs.
fn retag_input_leaves(gate_fn: &mut gate::GateFn) -> Result<(), String> {
    let mut assignments = Vec::new();
    for input in &gate_fn.inputs {
        for (lsb_index, operand) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            if operand.negated {
                return Err(format!(
                    "input `{}` bit {} unexpectedly references a negated operand",
                    input.name, lsb_index
                ));
            }
            assignments.push((operand.node, input.name.clone(), lsb_index));
        }
    }

    let mut seen = HashSet::new();
    for (node_ref, input_name, lsb_index) in assignments {
        if !seen.insert(node_ref) {
            return Err(format!(
                "input leaf %{} was reused across regrouped inputs; expected unique flat input leaves",
                node_ref.id
            ));
        }
        let node = gate_fn.gates.get_mut(node_ref.id).ok_or_else(|| {
            format!(
                "input `{}` bit {} references missing node %{}",
                input_name, lsb_index, node_ref.id
            )
        })?;
        match node {
            AigNode::Input {
                name,
                lsb_index: node_lsb_index,
                ..
            } => {
                *name = input_name;
                *node_lsb_index = lsb_index;
            }
            other => {
                return Err(format!(
                    "input `{}` bit {} expected input leaf at node %{}, got {:?}",
                    input_name, lsb_index, node_ref.id, other
                ));
            }
        }
    }
    Ok(())
}

/// Rebuilds the GateFn inputs and outputs strictly according to the provided
/// explicit schema.
pub fn repack_gate_fn_interface_with_schema(
    mut gate_fn: gate::GateFn,
    schema: &GateFnInterfaceSchema,
) -> Result<gate::GateFn, String> {
    schema.validate()?;
    let flat_inputs_lsb_is_0 = flatten_gate_port_bit_vectors_lsb_is_0(
        &gate_fn
            .inputs
            .iter()
            .map(|i| i.bit_vector.clone())
            .collect::<Vec<AigBitVector>>(),
    );
    gate_fn.inputs = repack_inputs_from_flat_schema(&flat_inputs_lsb_is_0, schema)?;
    retag_input_leaves(&mut gate_fn)?;

    let flat_outputs_lsb_is_0 = flatten_gate_port_bit_vectors_lsb_is_0(
        &gate_fn
            .outputs
            .iter()
            .map(|output| output.bit_vector.clone())
            .collect::<Vec<AigBitVector>>(),
    );
    gate_fn.outputs = repack_outputs_from_flat_schema(&flat_outputs_lsb_is_0, schema)?;
    Ok(gate_fn)
}

/// Rebuilds only the GateFn inputs according to the provided explicit schema.
pub fn repack_gate_fn_inputs_with_schema(
    mut gate_fn: gate::GateFn,
    schema: &GateFnInterfaceSchema,
) -> Result<gate::GateFn, String> {
    schema.validate()?;
    let flat_inputs_lsb_is_0 = flatten_gate_port_bit_vectors_lsb_is_0(
        &gate_fn
            .inputs
            .iter()
            .map(|i| i.bit_vector.clone())
            .collect::<Vec<AigBitVector>>(),
    );
    gate_fn.inputs = repack_inputs_from_flat_schema(&flat_inputs_lsb_is_0, schema)?;
    retag_input_leaves(&mut gate_fn)?;
    Ok(gate_fn)
}

/// Flattens a typed parameter into the bit ordering used by the gate interface.
fn flatten(param: BValue, ty: &ir::Type, fb: &mut FnBuilder) -> Result<BValue, BuilderError> {
    match ty {
        ir::Type::Bits(_width) => Ok(param),
        ir::Type::Tuple(types) => {
            let mut elements = Vec::new();
            for (i, t) in types.iter().enumerate() {
                let element = fb.tuple_index(param, i)?;
                elements.push(flatten(element, t, fb)?);
            }
            fb.concat(&elements)
        }
        ir::Type::Array(ArrayTypeData {
            element_type,
            element_count,
        }) => {
            let mut elements = Vec::new();
            // `concat()` places its first operand in the highest bits, so
            // reverse array iteration here to keep element 0 at the
            // least-significant bits.
            let index_width =
                (usize::BITS - element_count.saturating_sub(1).leading_zeros()).max(32) as usize;
            for i in (0..*element_count).rev() {
                let index = fb.literal(
                    IrValue::make_ubits(index_width, i as u64)
                        .expect("array index fits the width computed from its element count"),
                )?;
                let element = fb.array_index(param, index)?;
                elements.push(flatten(element, element_type, fb)?);
            }
            fb.concat(&elements)
        }
        ir::Type::Token => {
            // Tokens are zero bits so the nearest definition that makes sense
            // for flattening is that we make a zero-bit literal
            // value.
            fb.literal(IrValue::make_ubits(0, 0).unwrap())
        }
    }
}

/// Legalizes external pin names using the XLS identifier spelling convention.
fn sanitize_identifier(raw: &str) -> String {
    let mut result: String = raw
        .bytes()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == b'_' {
                c as char
            } else {
                '_'
            }
        })
        .collect();
    if result.is_empty() {
        result.push_str("name");
    }
    if !is_valid_identifier(&result) {
        result.insert(0, '_');
    }
    result
}

/// Keeps every external pin and generated label distinct after legalization.
#[derive(Default)]
struct GateIrNames {
    used: HashSet<String>,
}

impl GateIrNames {
    fn unique(&mut self, raw: &str) -> String {
        let name = sanitize_identifier(raw);
        let (root, requested_suffix) = name
            .rsplit_once("__")
            .and_then(|(root, suffix)| suffix.parse::<i64>().ok().map(|n| (root, n)))
            .map_or((name.as_str(), None), |(root, n)| (root, Some(n)));
        let root = if root.is_empty() { "name" } else { root };
        let candidate = requested_suffix
            .map(|n| format!("{root}__{n}"))
            .unwrap_or_else(|| root.to_string());
        if self.used.insert(candidate.clone()) {
            return candidate;
        }
        let mut suffix = 1usize;
        loop {
            let candidate = format!("{root}__{suffix}");
            if self.used.insert(candidate.clone()) {
                return candidate;
            }
            suffix += 1;
        }
    }
}

// Since the node env tracks individual bits we decompose inputs (which are
// vectors) into bits to place them into the map.
//
// We also need to match the given param_type in the function signature, so we
// may need to grab things out like tuple members at the very start of the
// function.
fn bind_param_bits_for_gate_fn_input(
    fb: &mut FnBuilder,
    node_env: &mut HashMap<AigOperand, BValue>,
    names: &mut GateIrNames,
    param: BValue,
    input: &gate::Input,
    param_type: &ir::Type,
) -> Result<(), BuilderError> {
    log::trace!("Processing input {:?}", input);
    // Note that inputs are bitvectors, so we add them as a single parameter and
    // then slice out all the bits for the environment.
    let param_bit_count = input.get_bit_count();
    if param_bit_count == 0 {
        // Zero-bit parameters still belong in the lifted function signature,
        // but they contribute no bits to the AIG node environment.
        return Ok(());
    }
    let flat_param = flatten(param, param_type, fb)?;
    if param_bit_count == 1 {
        log::trace!("Mapping single-bit input {:?}", input);
        node_env.insert(*input.bit_vector.get_lsb(0), flat_param);
    } else {
        log::trace!("Processing multi-bit input {:?}", input);
        for (i, g) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            log::trace!("Processing bit {} of multi-bit input {:?}", i, input);
            let bit = fb.bit_slice(flat_param, i, 1)?;
            fb.set_name(bit, &names.unique(&format!("{}[{}]", input.name, i)))?;
            node_env.insert(*g, bit);
        }
    }
    Ok(())
}

fn unflatten(
    bits_msb_is_0: &[BValue],
    ty: &ir::Type,
    fb: &mut FnBuilder,
) -> Result<BValue, BuilderError> {
    assert_eq!(
        bits_msb_is_0.len(),
        ty.bit_count(),
        "attempting to unflatten {} bits with associated type {:?}",
        bits_msb_is_0.len(),
        ty
    );
    match ty {
        ir::Type::Bits(width) => {
            assert_eq!(bits_msb_is_0.len(), *width);
            if *width == 1 {
                Ok(bits_msb_is_0[0])
            } else {
                fb.concat(bits_msb_is_0)
            }
        }
        ir::Type::Tuple(types) => {
            let mut elements: Vec<BValue> = Vec::new();
            let mut offset = 0;
            for t in types {
                let t_bit_count = t.bit_count();
                elements.push(unflatten(
                    &bits_msb_is_0[offset..offset + t_bit_count],
                    t,
                    fb,
                )?);
                offset += t_bit_count;
            }
            fb.tuple(&elements)
        }
        ir::Type::Token => fb.literal(IrValue::Token),
        ir::Type::Array(ArrayTypeData {
            element_type,
            element_count,
        }) => {
            let mut elements_rev: Vec<BValue> = Vec::new();
            let element_bit_count = element_type.bit_count();
            for i in 0..*element_count {
                let element_bits =
                    &bits_msb_is_0[i * element_bit_count..(i + 1) * element_bit_count];
                elements_rev.push(unflatten(element_bits, element_type, fb)?);
            }
            elements_rev.reverse();
            let elements = elements_rev;
            fb.array(element_type.as_ref().clone(), &elements)
        }
    }
}

fn make_return_value_from_outputs(
    fb: &mut FnBuilder,
    node_env: &HashMap<AigOperand, BValue>,
    outputs: &[gate::Output],
    ret_type: &ir::Type,
) -> Result<BValue, BuilderError> {
    let mut output_bits_msb_is_0 = Vec::new();
    for output in outputs {
        for g in output.bit_vector.iter_msb_to_lsb() {
            let bit = node_env.get(g).unwrap();
            output_bits_msb_is_0.push(*bit);
        }
    }

    // Now unflatten the bit vector according to the return type.
    unflatten(&output_bits_msb_is_0, ret_type, fb)
}

/// Returns an IR package with a single function as the top, which is the given
/// "gate function".
///
/// External function, package, and pin names are legalized to XLS identifiers;
/// colliding pin names are uniquified without changing the positional
/// interface.
pub fn gate_fn_to_pir(
    gate_fn: &gate::GateFn,
    package_name: &str,
    function_type: &ir::FunctionType,
) -> Result<ir::Package, BuilderError> {
    if gate_fn.inputs.len() != function_type.param_types.len() {
        return Err(BuilderError::InvalidOperation(format!(
            "gate interface has {} inputs, but the requested function has {} parameters",
            gate_fn.inputs.len(),
            function_type.param_types.len()
        )));
    }
    for (index, (input, ty)) in zip(&gate_fn.inputs, &function_type.param_types).enumerate() {
        let expected_width = ty.checked_bit_count().ok_or(BuilderError::WidthOverflow)?;
        if input.get_bit_count() != expected_width {
            return Err(BuilderError::InvalidOperation(format!(
                "gate input {index} '{}' has {} bits, but requested type {ty} has {expected_width} bits",
                input.name,
                input.get_bit_count()
            )));
        }
    }
    let output_width = gate_fn.outputs.iter().try_fold(0usize, |width, output| {
        width
            .checked_add(output.bit_vector.get_bit_count())
            .ok_or(BuilderError::WidthOverflow)
    })?;
    let expected_width = function_type
        .return_type
        .checked_bit_count()
        .ok_or(BuilderError::WidthOverflow)?;
    if output_width != expected_width {
        return Err(BuilderError::InvalidOperation(format!(
            "gate interface has {output_width} output bits, but requested result type {} has {expected_width} bits",
            function_type.return_type
        )));
    }
    log::trace!(
        "Converting gate function `{}` to IR:\n{}",
        gate_fn.name,
        gate_fn.to_string()
    );
    let mut fb = FnBuilder::new(&sanitize_identifier(&gate_fn.name));
    let mut node_env: HashMap<AigOperand, BValue> = HashMap::new();
    let mut names = GateIrNames::default();

    // We'll process from the input pins to the output pins in dependency order.
    let params = zip(gate_fn.inputs.iter(), function_type.param_types.iter())
        .map(|(input, ty)| fb.param(&names.unique(&input.name), ty.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    for ((input, param_type), param) in zip(
        zip(gate_fn.inputs.iter(), function_type.param_types.iter()),
        params,
    ) {
        bind_param_bits_for_gate_fn_input(
            &mut fb,
            &mut node_env,
            &mut names,
            param,
            input,
            param_type,
        )?;
    }

    for aig_operand in gate_fn.post_order_operands(true) {
        let aig_ref = aig_operand.node;
        let aig_node: &AigNode = gate_fn.get(aig_ref);
        match (aig_operand.negated, aig_node) {
            (true, AigNode::Input { .. }) => {
                // First we retrieve the non-inverted (positive) input.
                let pos = node_env
                    .get(&AigOperand {
                        node: aig_ref,
                        negated: false,
                    })
                    .unwrap();
                // Then we invert it and place it in the map.
                let neg = fb.not(*pos)?;
                node_env.insert(aig_operand, neg);
            }
            (false, AigNode::Input { .. }) => {
                panic!(
                    "Inputs should have been discarded; got: {:?} => {:?}",
                    aig_operand, aig_node
                );
            }
            (negated, AigNode::Literal { value, .. }) => {
                let result = fb.literal(IrValue::bool(*value))?;
                let result = if negated { fb.not(result)? } else { result };
                node_env.insert(aig_operand, result);
            }
            (negated, &AigNode::And2 { a, b, .. }) => {
                let lhs = node_env
                    .get(&a)
                    .expect(&format!("lhs of AND2 `{:?}` not found", a));
                let rhs = node_env
                    .get(&b)
                    .expect(&format!("rhs of AND2 `{:?}` not found", b));
                let result = fb.and(*lhs, *rhs)?;
                let result = if negated { fb.not(result)? } else { result };
                node_env.insert(aig_operand, result);
            }
        }
    }

    let return_value = make_return_value_from_outputs(
        &mut fb,
        &node_env,
        &gate_fn.outputs,
        &function_type.return_type,
    )?;

    fb.build_package(return_value, &sanitize_identifier(package_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::gate::GateFn;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::gatify::ir2gate::{GatifyOptions, gatify};
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth_pir::ir_parser;

    #[test]
    fn gate_lifting_rejects_incompatible_signatures_without_panicking() {
        let mut gb = GateBuilder::new("identity".to_string(), GateBuilderOptions::opt());
        let input = gb.add_input("x".to_string(), 2);
        gb.add_output("result".to_string(), input);
        let gates = gb.build();
        let signature = gates.get_flat_type();
        let mut wrong_count = gates.get_flat_type();
        wrong_count.param_types.clear();
        assert!(gate_fn_to_pir(&gates, "sample", &wrong_count).is_err());
        for width in [0, 1, 3, 65] {
            let mut wrong_input = gates.get_flat_type();
            wrong_input.param_types[0] = ir::Type::Bits(width);
            assert!(gate_fn_to_pir(&gates, "sample", &wrong_input).is_err());
            let mut wrong_output = gates.get_flat_type();
            wrong_output.return_type = ir::Type::Bits(width);
            assert!(gate_fn_to_pir(&gates, "sample", &wrong_output).is_err());
        }
        let mut overflowing = gates.get_flat_type();
        overflowing.return_type = ir::Type::new_array(ir::Type::Bits(usize::MAX), 2);
        assert!(matches!(
            gate_fn_to_pir(&gates, "sample", &overflowing),
            Err(BuilderError::WidthOverflow)
        ));
        gate_fn_to_pir(&gates, "sample", &signature).unwrap();
    }

    #[test]
    fn gate_lifting_checks_zero_bit_input_widths() {
        let mut gb = GateBuilder::new("empty".to_string(), GateBuilderOptions::opt());
        let input = gb.add_input("x".to_string(), 0);
        gb.add_output("result".to_string(), input);
        let gates = gb.build();
        let mut signature = gates.get_flat_type();
        gate_fn_to_pir(&gates, "sample", &signature).unwrap();
        signature.param_types[0] = ir::Type::Bits(1);
        assert!(gate_fn_to_pir(&gates, "sample", &signature).is_err());
    }

    /// Checks pin legalization preserves the complete positional interface.
    fn assert_echo_values(package: &ir::Package, args: &[IrValue]) {
        let f = package.get_top_fn().unwrap();
        match eval_fn(f, args) {
            FnEvalResult::Success(success) => {
                assert_eq!(success.value, IrValue::make_tuple(args));
            }
            FnEvalResult::Failure(failure) => panic!("echo evaluation failed: {failure:?}"),
        }
        let text = package.to_string();
        ir_parser::Parser::new(&text)
            .parse_and_validate_package()
            .unwrap();
        xlsynth::IrPackage::parse_ir(&text, None).expect("lifted IR must also be accepted by XLS");
    }

    #[test]
    fn test_gate_fn_to_pir_legalizes_external_names_without_losing_pins() {
        let mut gb = GateBuilder::new("9.external/top".to_string(), GateBuilderOptions::opt());
        let raw_names = [
            "p[0]", "p_0_", "p_0___1", "fn", "true", "", "9input", "p[0]",
        ];
        for (i, name) in raw_names.iter().enumerate() {
            let bits = gb.add_input((*name).to_string(), 1);
            gb.add_output(format!("out{i}"), bits);
        }
        let gates = gb.build();
        let package = gate_fn_to_pir(&gates, "external-package", &gates.get_flat_type()).unwrap();
        let f = package.get_top_fn().unwrap();
        assert_eq!(f.name, "_9_external_top");
        assert_eq!(
            f.param_nodes().map(|p| p.param_name()).collect::<Vec<_>>(),
            [
                "p_0_", "p_0___1", "p_0___2", "_fn", "_true", "name", "_9input", "p_0___3"
            ]
        );
        for enabled in 0..raw_names.len() {
            let args = (0..raw_names.len())
                .map(|i| IrValue::make_ubits(1, u64::from(i == enabled)).unwrap())
                .collect::<Vec<_>>();
            assert_echo_values(&package, &args);
        }
        assert_eq!(
            package.to_string(),
            gate_fn_to_pir(&gates, "external-package", &gates.get_flat_type())
                .unwrap()
                .to_string()
        );
    }

    #[test]
    fn test_gate_fn_to_pir_generated_slice_names_do_not_collide_with_params() {
        let mut gb = GateBuilder::new("slice_names".to_string(), GateBuilderOptions::opt());
        let wide = gb.add_input("foo".to_string(), 2);
        let collision = gb.add_input("foo_0_".to_string(), 1);
        gb.add_output("wide".to_string(), wide);
        gb.add_output("collision".to_string(), collision);
        let gates = gb.build();
        let package = gate_fn_to_pir(&gates, "sample", &gates.get_flat_type()).unwrap();
        let f = package.get_top_fn().unwrap();
        assert_eq!(f.get_param(1).param_name(), "foo_0_");
        let slice_names = f
            .nodes
            .iter()
            .filter_map(|node| {
                matches!(node.payload, ir::NodePayload::BitSlice { .. })
                    .then_some(node.name.as_deref())
                    .flatten()
            })
            .collect::<Vec<_>>();
        assert_eq!(slice_names, ["foo_0___1", "foo_1_"]);
        for wide in 0..4 {
            for collision in 0..2 {
                assert_echo_values(
                    &package,
                    &[
                        IrValue::make_ubits(2, wide).unwrap(),
                        IrValue::make_ubits(1, collision).unwrap(),
                    ],
                );
            }
        }
    }

    #[test]
    fn test_gate_fn_to_pir_preserves_token_and_zero_width_interface() {
        let mut gb = GateBuilder::new("token_echo".to_string(), GateBuilderOptions::opt());
        let token = gb.add_input("tok".to_string(), 0);
        let empty = gb.add_input("empty".to_string(), 0);
        let bit = gb.add_input("bit".to_string(), 1);
        gb.add_output("tok_out".to_string(), token);
        gb.add_output("empty_out".to_string(), empty);
        gb.add_output("bit_out".to_string(), bit);
        let gates = gb.build();
        let ty = ir::FunctionType {
            param_types: vec![ir::Type::Token, ir::Type::Bits(0), ir::Type::Bits(1)],
            return_type: ir::Type::Tuple(vec![
                Box::new(ir::Type::Token),
                Box::new(ir::Type::Bits(0)),
                Box::new(ir::Type::Bits(1)),
            ]),
        };
        let package = gate_fn_to_pir(&gates, "sample", &ty).unwrap();
        assert_eq!(package.get_top_fn().unwrap().get_type(), ty);
        for bit in 0..2 {
            assert_echo_values(
                &package,
                &[
                    IrValue::Token,
                    IrValue::make_ubits(0, 0).unwrap(),
                    IrValue::make_ubits(1, bit).unwrap(),
                ],
            );
        }
    }

    #[test]
    fn test_gate_fn_to_ir_one_and_gate() {
        let input_ir_text = "package sample

top fn do_and(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}
";
        let mut parser = ir_parser::Parser::new(input_ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_top = ir_package.get_top_fn().unwrap();
        let gatify_output = gatify(
            &ir_top,
            GatifyOptions {
                check_equivalence: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let package = gate_fn_to_pir(&gatify_output.gate_fn, "sample", &ir_top.get_type()).unwrap();
        assert_eq!(package.to_string(), input_ir_text);
    }

    #[test]
    fn test_gate_fn_to_ir_inverter() {
        let input_ir_text = "package sample

top fn do_not(a: bits[1] id=1) -> bits[1] {
  ret not.2: bits[1] = not(a, id=2)
}
";
        let mut parser = ir_parser::Parser::new(input_ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_top = ir_package.get_top_fn().unwrap();
        let gatify_output = gatify(
            &ir_top,
            GatifyOptions {
                check_equivalence: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let package = gate_fn_to_pir(&gatify_output.gate_fn, "sample", &ir_top.get_type()).unwrap();
        assert_eq!(package.to_string(), input_ir_text);
    }

    #[test]
    fn test_gate_fn_to_ir_nand() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input_ir_text = "package sample

top fn do_nand(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  and.3: bits[1] = and(a, b, id=3)
  ret not.4: bits[1] = not(and.3, id=4)
}
";
        let mut parser = ir_parser::Parser::new(input_ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_top = ir_package.get_top_fn().unwrap();
        let gatify_output = gatify(
            &ir_top,
            GatifyOptions {
                check_equivalence: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        let package = gate_fn_to_pir(&gatify_output.gate_fn, "sample", &ir_top.get_type()).unwrap();
        let gate_fn_as_xls_ir = package.to_string();
        log::trace!("gate_fn_as_xls_ir:\n{}", gate_fn_as_xls_ir);
        assert_eq!(gate_fn_as_xls_ir, input_ir_text);
    }

    #[test]
    fn test_schema_from_function_type_allows_unit_return() {
        let function_type = ir::FunctionType {
            param_types: vec![],
            return_type: ir::Type::Tuple(vec![]),
        };
        let schema = GateFnInterfaceSchema::from_function_type(&function_type).unwrap();
        assert!(schema.output_ports.is_empty());
        assert_eq!(schema.return_type.bit_count(), 0);
    }

    #[test]
    fn test_schema_from_function_type_allows_zero_width_input_port() {
        let function_type = ir::FunctionType {
            param_types: vec![ir::Type::Tuple(vec![]), ir::Type::Bits(1)],
            return_type: ir::Type::Bits(1),
        };
        let schema = GateFnInterfaceSchema::from_function_type(&function_type).unwrap();
        assert_eq!(schema.input_ports.len(), 2);
        assert_eq!(schema.input_ports[0].ty.bit_count(), 0);
        assert_eq!(schema.input_ports[1].ty.bit_count(), 1);
    }

    #[test]
    fn test_repack_gate_fn_inputs_with_schema_retags_input_leaves() {
        let gate_fn = GateFn::try_from(
            r#"fn sample(x_0: bits[1] = [%1], x_1: bits[1] = [%2]) -> (out: bits[1] = [%3]) {
  %3 = and(x_0[0], x_1[0])
  out[0] = %3
}"#,
        )
        .unwrap();
        let schema = GateFnInterfaceSchema::from_function_type(&ir::FunctionType {
            param_types: vec![ir::Type::Bits(2)],
            return_type: ir::Type::Bits(1),
        })
        .unwrap();

        let repacked = repack_gate_fn_inputs_with_schema(gate_fn, &schema).unwrap();
        let reparsed = GateFn::try_from(repacked.to_string().as_str()).unwrap();

        assert_eq!(repacked.inputs.len(), 1);
        assert_eq!(repacked.inputs[0].name, "arg0");
        assert_eq!(repacked.inputs[0].get_bit_count(), 2);
        assert_eq!(repacked.to_string(), reparsed.to_string());
    }
}
