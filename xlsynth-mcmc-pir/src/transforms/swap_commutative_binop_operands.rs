// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A simple transform that swaps the operands of commutative binary operators.
///
/// Currently this is restricted to `add` nodes, which are clearly commutative.
#[derive(Debug)]
pub struct SwapCommutativeBinopOperandsTransform;

impl PirTransform for SwapCommutativeBinopOperandsTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SwapCommutativeBinopOperands
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        f.node_refs()
            .into_iter()
            .filter(|nr| {
                matches!(
                    f.get_node(*nr).payload,
                    NodePayload::Binop(Binop::Add, _, _)
                )
            })
            .map(TransformLocation::Node)
            .collect()
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        match loc {
            TransformLocation::Node(nr) => {
                let node = f.get_node_mut(*nr);
                match &mut node.payload {
                    NodePayload::Binop(Binop::Add, lhs, rhs) => {
                        mem::swap(lhs, rhs);
                        Ok(())
                    }
                    other => Err(format!(
                        "SwapCommutativeBinopOperandsTransform: expected add binop, found {:?}",
                        other
                    )),
                }
            }
            TransformLocation::RewireOperand { .. } => Err(
                "SwapCommutativeBinopOperandsTransform: expected TransformLocation::Node, got RewireOperand"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
