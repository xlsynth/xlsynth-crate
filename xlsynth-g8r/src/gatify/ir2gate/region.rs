// SPDX-License-Identifier: Apache-2.0

//! Bounded IR region lowering through the ordinary g8r node implementation.

use super::{GateEnv, GateOrVec, GatifyOptions, gatify_concat, gatify_node};
use crate::aig::dce::dce;
use crate::aig::gate::{AigBitVector, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use std::collections::HashSet;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils;

/// Borrows a region of bits-typed shifts, selections, and wiring.
///
/// Interior operations may be `literal`, `identity`, `not`, `bit_slice`,
/// `zero_ext`, `sign_ext`, `and`, `concat`, `sel`, `priority_sel`, `shll`, or
/// `shrl`. Boundary inputs may have any payload: they are independent values,
/// even when the containing function defines them as literals or operations.
///
/// The caller must supply well-typed IR nodes. Inputs must be distinct, and
/// interior nodes must be distinct, exclude inputs, and appear after all their
/// operands. Outputs must refer to inputs or interior nodes. Unused inputs and
/// repeated outputs are allowed. Only this boundary and the listed nodes are
/// checked; the containing function is not traversed or validated.
#[derive(Clone, Copy, Debug)]
pub struct IrRegion<'a> {
    /// Contains the node payloads and types referenced by this region.
    pub function: &'a ir::Fn,
    /// Lists interior nodes in dependency order, excluding boundary inputs.
    pub nodes: &'a [NodeRef],
    /// Lists independent bits-typed inputs in the desired interface order.
    pub inputs: &'a [NodeRef],
    /// Lists output values in concatenation order, most significant first.
    pub outputs: &'a [NodeRef],
}

impl IrRegion<'_> {
    /// Checks region membership, ordering, and supported operation kinds.
    fn validate(&self) -> Result<(), String> {
        let mut available = HashSet::new();
        for &input in self.inputs {
            let node = self.function.nodes.get(input.index).ok_or_else(|| {
                format!("gatify_region: input node {} does not exist", input.index)
            })?;
            if !matches!(node.ty, Type::Bits(_)) {
                return Err(format!(
                    "gatify_region: input node {} is not bits-typed",
                    input.index
                ));
            }
            if !available.insert(input) {
                return Err(format!(
                    "gatify_region: duplicate input node {}",
                    input.index
                ));
            }
        }
        for &nr in self.nodes {
            let node = self.function.nodes.get(nr.index).ok_or_else(|| {
                format!("gatify_region: interior node {} does not exist", nr.index)
            })?;
            if available.contains(&nr) {
                return Err(format!(
                    "gatify_region: node {} is repeated or also listed as an input",
                    nr.index
                ));
            }
            if !matches!(node.ty, Type::Bits(_)) {
                return Err(format!(
                    "gatify_region: interior node {} is not bits-typed",
                    nr.index
                ));
            }
            if !matches!(
                node.payload,
                NodePayload::Literal(_)
                    | NodePayload::Unop(Unop::Identity | Unop::Not, _)
                    | NodePayload::BitSlice { .. }
                    | NodePayload::ZeroExt { .. }
                    | NodePayload::SignExt { .. }
                    | NodePayload::Nary(NaryOp::And | NaryOp::Concat, _)
                    | NodePayload::Sel { .. }
                    | NodePayload::PrioritySel { .. }
                    | NodePayload::Binop(Binop::Shll | Binop::Shrl, _, _)
            ) {
                return Err(format!(
                    "gatify_region: unsupported interior operation '{}' at node {}",
                    node.payload.get_operator(),
                    nr.index
                ));
            }
            for operand in ir_utils::operands(&node.payload) {
                if !available.contains(&operand) {
                    return Err(format!(
                        "gatify_region: operand {} of node {} must be an input or an earlier interior node",
                        operand.index, nr.index
                    ));
                }
            }
            available.insert(nr);
        }
        let mut output_width = 0usize;
        for &output in self.outputs {
            if !available.contains(&output) {
                return Err(format!(
                    "gatify_region: output node {} is not an input or interior node",
                    output.index
                ));
            }
            output_width = output_width
                .checked_add(self.function.get_node(output).ty.bit_count())
                .ok_or_else(|| "gatify_region: concatenated output width overflows".to_string())?;
        }
        Ok(())
    }
}

/// Lowers an explicit region with independent inputs and one concatenated
/// output.
///
/// Uses the caller's exact gate folding and hashing settings. Preparation,
/// range analysis, and structural peepholes are disabled so they cannot look
/// through the boundary. Dead gates are removed while all declared inputs are
/// preserved. Invalid boundaries and unsupported interior operations return an
/// error; see [`IrRegion`] for the node and ordering contract.
pub fn gatify_region(
    region: IrRegion<'_>,
    gate_builder_options: GateBuilderOptions,
) -> Result<GateFn, String> {
    region.validate()?;
    let f = region.function;
    let mut builder = GateBuilder::new("ir_region".to_string(), gate_builder_options);
    let mut env = GateEnv::for_region(f, region.nodes, region.outputs);
    let options = GatifyOptions {
        fold: gate_builder_options.fold,
        hash: gate_builder_options.hash,
        ..GatifyOptions::all_opts_disabled()
    };
    for (index, &input) in region.inputs.iter().enumerate() {
        let node = f.get_node(input);
        let bits = builder.add_input(format!("input_{index}"), node.ty.bit_count());
        env.add(input, GateOrVec::BitVector(bits));
    }
    for &nr in region.nodes {
        let node = f.get_node(nr);
        let direct = if node.ty.bit_count() == 0 {
            Some(AigBitVector::zeros(0))
        } else {
            match &node.payload {
                NodePayload::Sel { cases, default, .. } if cases.is_empty() => {
                    let default = default.ok_or_else(|| {
                        format!(
                            "gatify_region: select at node {} has neither cases nor a default",
                            nr.index
                        )
                    })?;
                    Some(env.get_bit_vector(default)?)
                }
                NodePayload::Sel {
                    selector, cases, ..
                } if f.get_node(*selector).ty.bit_count() == 0 => {
                    Some(env.get_bit_vector(cases[0])?)
                }
                _ => None, // Ordinary nodes use the shared lowering.
            }
        };
        if let Some(bits) = direct {
            env.add(nr, GateOrVec::BitVector(bits));
        } else {
            gatify_node(f, nr, node, &mut builder, &mut env, &options)?;
        }
    }
    let outputs = region
        .outputs
        .iter()
        .map(|&output| env.get_bit_vector(output))
        .collect::<Result<Vec<_>, _>>()?;
    builder.add_output("out".to_string(), gatify_concat(&outputs));
    Ok(dce(&builder.build()))
}

#[cfg(test)]
mod tests {
    use super::{IrRegion, gatify_region};
    use crate::aig_sim::gate_sim;
    use crate::gate_builder::GateBuilderOptions;
    use xlsynth_pir::IrBits;
    use xlsynth_pir::ir::{Binop, NodePayload};
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn boundary_payloads_are_opaque_and_arithmetic_interiors_are_rejected() {
        let package = Parser::new(
            r#"package sample

top fn main(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  sum: bits[8] = add(x, y, id=3)
  one: bits[2] = literal(value=1, id=4)
  ret shifted: bits[8] = shrl(sum, one, id=5)
}
"#,
        )
        .parse_and_validate_package()
        .unwrap();
        let f = package.get_top_fn().unwrap();
        let shifted = f.ret_node_ref.unwrap();
        let NodePayload::Binop(Binop::Shrl, sum, one) = f.get_node(shifted).payload else {
            panic!("test result must be a logical right shift");
        };
        let region = IrRegion {
            function: f,
            nodes: &[shifted],
            inputs: &[sum, one],
            outputs: &[shifted],
        };
        for fold in [false, true] {
            for hash in [false, true] {
                let gate_fn = gatify_region(region, GateBuilderOptions { fold, hash }).unwrap();
                // Neither the add's operands nor the literal's value constrain
                // the independently supplied boundary values.
                let result = gate_sim::eval(
                    &gate_fn,
                    &[
                        IrBits::make_ubits(8, 0xf0).unwrap(),
                        IrBits::make_ubits(2, 2).unwrap(),
                    ],
                    gate_sim::Collect::None,
                );
                assert_eq!(result.outputs, vec![IrBits::make_ubits(8, 0x3c).unwrap()]);
            }
        }
        let arithmetic_region = IrRegion {
            nodes: &[sum, shifted],
            inputs: &[f.params[0], f.params[1], one],
            ..region
        };
        assert_eq!(
            gatify_region(arithmetic_region, GateBuilderOptions::opt()).unwrap_err(),
            format!(
                "gatify_region: unsupported interior operation 'add' at node {}",
                sum.index
            )
        );
    }

    #[test]
    fn missing_boundary_operand_returns_an_error() {
        let package = Parser::new(
            r#"package sample

top fn main(x: bits[8] id=1) -> bits[8] {
  ret result: bits[8] = not(x, id=2)
}
"#,
        )
        .parse_and_validate_package()
        .unwrap();
        let f = package.get_top_fn().unwrap();
        let result = f.ret_node_ref.unwrap();
        assert_eq!(
            gatify_region(
                IrRegion {
                    function: f,
                    nodes: &[result],
                    inputs: &[],
                    outputs: &[result],
                },
                GateBuilderOptions::opt(),
            )
            .unwrap_err(),
            format!(
                "gatify_region: operand {} of node {} must be an input or an earlier interior node",
                f.params[0].index, result.index
            )
        );
    }
}
