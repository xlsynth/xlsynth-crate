// SPDX-License-Identifier: Apache-2.0

//! Functionality for converting a gate function into an IR function.
//!
//! This is useful for getting it "back into" XLS IR form after transforms so we
//! can test equivalence.

use std::collections::{HashMap, HashSet};
use std::iter::zip;

use crate::aig::gate::{self, AigBitVector, AigNode, AigOperand, Input, Output};
use xlsynth;
use xlsynth::{BValue, FnBuilder, IrType, IrValue, XlsynthError};
use xlsynth_pir::ir::{self, ArrayTypeData};

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
            .params
            .iter()
            .map(|param| GateFnInterfacePort {
                name: param.name.clone(),
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

fn make_xlsynth_type(param_type: &ir::Type, package: &mut xlsynth::IrPackage) -> IrType {
    match param_type {
        ir::Type::Bits(width) => package.get_bits_type(*width as u64),
        ir::Type::Tuple(types) => {
            let members = types
                .iter()
                .map(|t| make_xlsynth_type(t, package))
                .collect::<Vec<_>>();
            package.get_tuple_type(&members)
        }
        ir::Type::Token => package.get_token_type(),
        ir::Type::Array(ArrayTypeData {
            element_type,
            element_count,
        }) => {
            let element_type = make_xlsynth_type(element_type, package);
            package.get_array_type(&element_type, *element_count as i64)
        }
    }
}

fn flatten(param: &BValue, ty: &ir::Type, fb: &mut FnBuilder) -> BValue {
    match ty {
        ir::Type::Bits(_width) => param.clone(),
        ir::Type::Tuple(types) => {
            let mut elements = Vec::new();
            for (i, t) in types.iter().enumerate() {
                let element = fb.tuple_index(param, i as u64, None);
                elements.push(flatten(&element, t, fb));
            }
            let elements_refs = elements.iter().map(|e| e as &BValue).collect::<Vec<_>>();
            fb.concat(&elements_refs, None)
        }
        ir::Type::Array(ArrayTypeData {
            element_type,
            element_count,
        }) => {
            let mut elements = Vec::new();
            // `concat()` places its first operand in the highest bits, so reverse
            // array iteration here to keep element 0 at the least-significant bits.
            for i in (0..*element_count).rev() {
                let index = fb.literal(&IrValue::u32(i as u32), None);
                let element = fb.array_index(param, &index, None);
                elements.push(flatten(&element, element_type, fb));
            }
            let elements_refs = elements.iter().map(|e| e as &BValue).collect::<Vec<_>>();
            fb.concat(&elements_refs, None)
        }
        ir::Type::Token => {
            // Tokens are zero bits so the nearest definition that makes sense for
            // flattening is that we make a zero-bit literal value.
            fb.literal(&IrValue::make_ubits(0, 0).unwrap(), None)
        }
    }
}

// Since the node env tracks individual bits we decompose inputs (which are
// vectors) into bits to place them into the map.
//
// We also need to match the given param_type in the function signature, so we
// may need to grab things out like tuple members at the very start of the
// function.
fn add_param_for_gate_fn_input(
    fb: &mut FnBuilder,
    node_env: &mut HashMap<AigOperand, BValue>,
    package: &mut xlsynth::IrPackage,
    input: &gate::Input,
    param_type: &ir::Type,
) -> Result<(), XlsynthError> {
    log::trace!("Processing input {:?}", input);
    // Note that inputs are bitvectors, so we add them as a single parameter and
    // then slice out all the bits for the environment.
    let param_bit_count = input.get_bit_count() as u64;
    let ty: IrType = make_xlsynth_type(param_type, package);
    if param_bit_count == 0 {
        // Zero-bit parameters still belong in the lifted function signature,
        // but they contribute no bits to the AIG node environment.
        let _ = fb.param(input.name.as_str(), &ty);
        return Ok(());
    }
    let param = fb.param(input.name.as_str(), &ty);
    let flat_param = flatten(&param, param_type, fb);
    if param_bit_count == 1 {
        log::trace!("Mapping single-bit input {:?}", input);
        node_env.insert(*input.bit_vector.get_lsb(0), flat_param);
    } else {
        log::trace!("Processing multi-bit input {:?}", input);
        for (i, g) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            log::trace!("Processing bit {} of multi-bit input {:?}", i, input);
            let bit = fb.bit_slice(
                &flat_param,
                i as u64,
                1,
                Some(&format!("{}[{}]", input.name, i)),
            );
            node_env.insert(*g, bit);
        }
    }
    Ok(())
}

fn unflatten(
    bits_msb_is_0: &[&BValue],
    ty: &ir::Type,
    fb: &mut FnBuilder,
    package: &mut xlsynth::IrPackage,
) -> BValue {
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
            let elements_refs = bits_msb_is_0
                .iter()
                .map(|e| e as &BValue)
                .collect::<Vec<_>>();
            if *width == 1 {
                elements_refs[0].clone()
            } else {
                fb.concat(&elements_refs, None)
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
                    package,
                ));
                offset += t_bit_count;
            }
            let elements_refs = elements.iter().map(|e| e as &BValue).collect::<Vec<_>>();
            fb.tuple(&elements_refs, None)
        }
        ir::Type::Token => todo!(),
        ir::Type::Array(ArrayTypeData {
            element_type,
            element_count,
        }) => {
            let mut elements_rev: Vec<BValue> = Vec::new();
            let element_bit_count = element_type.bit_count();
            for i in 0..*element_count {
                let element_bits =
                    &bits_msb_is_0[i * element_bit_count..(i + 1) * element_bit_count];
                elements_rev.push(unflatten(element_bits, element_type, fb, package));
            }
            elements_rev.reverse();
            let elements = elements_rev;
            let elements_refs = elements.iter().map(|e| e as &BValue).collect::<Vec<_>>();
            fb.array(
                &make_xlsynth_type(element_type, package),
                &elements_refs,
                None,
            )
        }
    }
}

fn make_return_value_from_outputs(
    fb: &mut FnBuilder,
    node_env: &mut HashMap<AigOperand, BValue>,
    outputs: &[gate::Output],
    ret_type: &ir::Type,
    package: &mut xlsynth::IrPackage,
) -> BValue {
    let mut output_bits_msb_is_0 = Vec::new();
    for output in outputs {
        for g in output.bit_vector.iter_msb_to_lsb() {
            let bit = node_env.get(g).unwrap();
            output_bits_msb_is_0.push(bit);
        }
    }

    // Now unflatten the bit vector according to the return type.
    unflatten(&output_bits_msb_is_0, ret_type, fb, package)
}

/// Returns an IR package with a single function as the top, which is the given
/// "gate function".
pub fn gate_fn_to_xlsynth_ir(
    gate_fn: &gate::GateFn,
    package_name: &str,
    function_type: &ir::FunctionType,
) -> Result<xlsynth::IrPackage, XlsynthError> {
    assert_eq!(gate_fn.inputs.len(), function_type.param_types.len());
    let gate_fn_type = gate_fn.get_flat_type();
    assert_eq!(
        gate_fn_type.return_type.bit_count(),
        function_type.return_type.bit_count()
    );
    log::trace!(
        "Converting gate function `{}` to IR:\n{}",
        gate_fn.name,
        gate_fn.to_string()
    );
    let mut package = xlsynth::IrPackage::new(package_name).unwrap();
    let mut fb = FnBuilder::new(&mut package, gate_fn.name.as_str(), true);
    let mut node_env: HashMap<AigOperand, BValue> = HashMap::new();

    // We'll process from the input pins to the output pins in dependency order.
    for (input, param_type) in zip(gate_fn.inputs.iter(), function_type.param_types.iter()) {
        add_param_for_gate_fn_input(&mut fb, &mut node_env, &mut package, input, param_type)?;
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
                let neg = fb.not(pos, None);
                node_env.insert(aig_operand, neg);
            }
            (false, AigNode::Input { .. }) => {
                panic!(
                    "Inputs should have been discarded; got: {:?} => {:?}",
                    aig_operand, aig_node
                );
            }
            (negated, AigNode::Literal { value, .. }) => {
                let result = fb.literal(&IrValue::make_ubits(1, *value as u64).unwrap(), None);
                let result = if negated {
                    fb.not(&result, None)
                } else {
                    result
                };
                node_env.insert(aig_operand, result);
            }
            (negated, &AigNode::And2 { a, b, .. }) => {
                let lhs = node_env
                    .get(&a)
                    .expect(&format!("lhs of AND2 `{:?}` not found", a));
                let rhs = node_env
                    .get(&b)
                    .expect(&format!("rhs of AND2 `{:?}` not found", b));
                let result = fb.and(lhs, rhs, None);
                let result = if negated {
                    fb.not(&result, None)
                } else {
                    result
                };
                node_env.insert(aig_operand, result);
            }
        }
    }

    let return_value = make_return_value_from_outputs(
        &mut fb,
        &mut node_env,
        &gate_fn.outputs,
        &function_type.return_type,
        &mut package,
    );

    let f = fb.build_with_return_value(&return_value)?;
    package
        .set_top_by_name(f.get_name().as_str())
        .expect("Failed to set top function");
    Ok(package)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::gate::GateFn;
    use crate::gatify::ir2gate::{GatifyOptions, gatify};
    use xlsynth_pir::ir_parser;

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
                fold: true,
                check_equivalence: true,
                hash: true,
                adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .unwrap();
        let package =
            gate_fn_to_xlsynth_ir(&gatify_output.gate_fn, "sample", &ir_top.get_type()).unwrap();
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
                fold: true,
                check_equivalence: true,
                hash: true,
                adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .unwrap();
        let package =
            gate_fn_to_xlsynth_ir(&gatify_output.gate_fn, "sample", &ir_top.get_type()).unwrap();
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
                fold: true,
                check_equivalence: true,
                hash: true,
                adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .unwrap();
        let package =
            gate_fn_to_xlsynth_ir(&gatify_output.gate_fn, "sample", &ir_top.get_type()).unwrap();
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
