// SPDX-License-Identifier: Apache-2.0

//! Functionality for converting a gate function into an IR function.
//!
//! This is useful for getting it "back into" XLS IR form after transforms so we
//! can test equivalence.

use std::collections::HashMap;
use std::iter::zip;

use crate::gate::{self, AigNode, AigOperand};
use crate::xls_ir::ir::{self, ArrayTypeData};
use xlsynth;
use xlsynth::{BValue, FnBuilder, IrType, IrValue, XlsynthError};

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
            for i in 0..*element_count {
                let index = fb.literal(&IrValue::make_ubits(32, i as u64).unwrap(), None);
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
    log::debug!("Processing input {:?}", input);
    // Note that inputs are bitvectors, so we add them as a single parameter and
    // then slice out all the bits for the environment.
    let param_bit_count = input.get_bit_count() as u64;
    let ty: IrType = make_xlsynth_type(param_type, package);
    let param = fb.param(input.name.as_str(), &ty);
    let flat_param = flatten(&param, param_type, fb);
    if param_bit_count == 1 {
        log::debug!("Mapping single-bit input {:?}", input);
        node_env.insert(*input.bit_vector.get_lsb(0), flat_param);
    } else {
        log::debug!("Processing multi-bit input {:?}", input);
        for (i, g) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            log::debug!("Processing bit {} of multi-bit input {:?}", i, input);
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
            let mut elements: Vec<BValue> = Vec::new();
            let element_bit_count = element_type.bit_count();
            for i in 0..*element_count {
                let element_bits =
                    &bits_msb_is_0[i * element_bit_count..(i + 1) * element_bit_count];
                elements.push(unflatten(element_bits, element_type, fb, package));
            }
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
    // Flatten all the output gates into a single bit vector.
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
    log::info!(
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

    log::debug!("RPO: {:?}", gate_fn.post_order(true));
    for aig_operand in gate_fn.post_order(true) {
        log::debug!("Processing {:?}", aig_operand);
        let aig_ref = aig_operand.node;
        let aig_node: &AigNode = gate_fn.get(aig_ref);
        log::debug!(" aig_node: {:?}", aig_node);
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
            (negated, AigNode::Literal(value)) => {
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
    use crate::{ir2gate::gatify, ir2gate::GatifyOptions, xls_ir::ir_parser};

    #[test]
    fn test_gate_fn_to_ir_one_and_gate() {
        let input_ir_text = "package sample

top fn do_and(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}
";
        let mut parser = ir_parser::Parser::new(input_ir_text);
        let ir_package = parser.parse_package().unwrap();
        let ir_top = ir_package.get_top().unwrap();
        let gate_fn = gatify(
            &ir_top,
            GatifyOptions {
                fold: true,
                check_equivalence: true,
            },
        )
        .unwrap();
        let package = gate_fn_to_xlsynth_ir(&gate_fn, "sample", &ir_top.get_type()).unwrap();
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
        let ir_package = parser.parse_package().unwrap();
        let ir_top = ir_package.get_top().unwrap();
        let gate_fn = gatify(
            &ir_top,
            GatifyOptions {
                fold: true,
                check_equivalence: true,
            },
        )
        .unwrap();
        let package = gate_fn_to_xlsynth_ir(&gate_fn, "sample", &ir_top.get_type()).unwrap();
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
        let ir_package = parser.parse_package().unwrap();
        let ir_top = ir_package.get_top().unwrap();
        let gate_fn = gatify(
            &ir_top,
            GatifyOptions {
                fold: true,
                check_equivalence: true,
            },
        )
        .unwrap();
        let package = gate_fn_to_xlsynth_ir(&gate_fn, "sample", &ir_top.get_type()).unwrap();
        let gate_fn_as_xls_ir = package.to_string();
        log::info!("gate_fn_as_xls_ir:\n{}", gate_fn_as_xls_ir);
        assert_eq!(gate_fn_as_xls_ir, input_ir_text);
    }
}
