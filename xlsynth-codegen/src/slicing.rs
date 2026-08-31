// SPDX-License-Identifier: Apache-2.0

//! Width-specialized dynamic slicing with explicit range and sizing boundaries.

use xlsynth::{IrBits, IrValue};
use xlsynth_pir::ir::{Fn, Node, NodePayload, NodeRef, Type};
use xlsynth_vast::Expr;

use crate::BlockCodegenError;
use crate::block::{BlockEmitter, Value};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum SliceHelper {
    Dynamic {
        operand_width: usize,
        start_width: usize,
        width: usize,
    },
    Update {
        width: usize,
        start_width: usize,
        update_width: usize,
    },
}

impl BlockEmitter<'_, '_> {
    /// Emits one typed function per specialization before its callers.
    pub(crate) fn emit_slice_helpers(&mut self) -> Result<(), BlockCodegenError> {
        for node in &self.func.nodes {
            let Some(helper) = SliceHelper::for_node(self.func, node) else {
                continue;
            };
            if self.slice_helpers.contains_key(&helper) {
                continue;
            }
            let name = self.unique_name(&helper.name());
            match helper {
                SliceHelper::Dynamic {
                    operand_width,
                    start_width,
                    width,
                } => self.emit_dynamic_slice_helper(&name, operand_width, start_width, width)?,
                SliceHelper::Update {
                    width,
                    start_width,
                    update_width,
                } => self.emit_slice_update_helper(&name, width, start_width, update_width)?,
            }
            self.slice_helpers.insert(helper, name);
        }
        Ok(())
    }

    /// Calls a typed helper, folding empty operands without zero-width SV
    /// types.
    pub(crate) fn emit_slice_call(&mut self, node_ref: NodeRef) -> Result<Expr, BlockCodegenError> {
        let node = self.func.get_node(node_ref);
        let operands = match node.payload {
            NodePayload::DynamicBitSlice { arg, start, width } => {
                if self.func.get_node_ty(arg).bit_count() == 0 {
                    return self.zero(width);
                }
                vec![arg, start]
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                if self.func.get_node_ty(update_value).bit_count() == 0 {
                    return Ok(self.required_value(arg)?.expr);
                }
                vec![arg, start, update_value]
            }
            _ => unreachable!("slice call requires a dynamic slice or update"),
        };
        let helper = SliceHelper::for_node(self.func, node).expect("nonempty slice helper");
        let arguments = operands
            .into_iter()
            .map(|operand| self.numeric_value(operand).map(|value| value.expr))
            .collect::<Result<Vec<_>, _>>()?;
        let name = self.slice_helpers.get(&helper).expect("predeclared helper");
        Ok(self
            .file
            .make_function_call(name, &arguments.iter().collect::<Vec<_>>()))
    }

    /// Pads partial overlaps with zero and guards completely out-of-range
    /// starts.
    fn emit_dynamic_slice_helper(
        &mut self,
        name: &str,
        operand_width: usize,
        start_width: usize,
        width: usize,
    ) -> Result<(), BlockCodegenError> {
        let result_type = self.bits_type(width);
        let handle = self.file.add_function(self.module, name, &result_type)?;
        let operand_type = self.bits_type(operand_width);
        let operand = self
            .file
            .function_add_logic_input(handle, "operand", &operand_type)?;
        let start_type = self.bits_type(start_width.max(1));
        let start = self
            .file
            .function_add_logic_input(handle, "start", &start_type)?;
        let extended_type = self.bits_type(operand_width + width);
        let extended = self
            .file
            .function_add_logic(handle, "extended_operand", &extended_type)?;
        let zeros = self.zero(width)?;
        let padded = self.file.make_concat(&[&zeros, &operand.to_expr()]);
        let body = self.file.function_body(handle);
        self.file
            .block_add_blocking_assignment(body, &extended.to_expr(), &padded);
        let subject = self.file.make_indexable_expression(&extended.to_expr());
        let selected = self
            .file
            .make_indexed_part_select(&subject, &start.to_expr(), width as i64);
        let value = if start_width >= (usize::BITS - operand_width.leading_zeros()) as usize {
            let bound = self.sized_usize(start_width, operand_width)?;
            let out_of_bounds = self.file.make_ge(&start.to_expr(), &bound);
            self.file.make_ternary(&out_of_bounds, &zeros, &selected)
        } else {
            selected
        };
        let result = self.file.function_result(handle).to_expr();
        self.file
            .block_add_blocking_assignment(body, &result, &value);
        Ok(())
    }

    /// Resizes update data before shifting and preserves the input on
    /// overshift.
    fn emit_slice_update_helper(
        &mut self,
        name: &str,
        width: usize,
        start_width: usize,
        update_width: usize,
    ) -> Result<(), BlockCodegenError> {
        let result_type = self.bits_type(width);
        let handle = self.file.add_function(self.module, name, &result_type)?;
        let original = self
            .file
            .function_add_logic_input(handle, "to_update", &result_type)?;
        let start_type = self.bits_type(start_width.max(1));
        let start = self
            .file
            .function_add_logic_input(handle, "start", &start_type)?;
        let update_type = self.bits_type(update_width);
        let update = self
            .file
            .function_add_logic_input(handle, "update_value", &update_type)?;
        let adjusted = self.resize_unsigned(
            Value::signal(update).with_width(update_width),
            update_width,
            width,
        )?;
        let mask_bits = IrBits::all_ones(width).shrl((width - update_width.min(width)) as i64);
        let mask = self
            .literal(&IrValue::from_bits(&mask_bits), &Type::Bits(width))?
            .expect("nonempty mask");
        let shifted_mask = self.file.make_shll(&mask, &start.to_expr());
        let inverse_mask = self.file.make_bitwise_not(&shifted_mask);
        let preserved = self
            .file
            .make_bitwise_and(&inverse_mask, &original.to_expr());
        let inserted = self.file.make_shll(&adjusted, &start.to_expr());
        let updated = self.file.make_bitwise_or(&inserted, &preserved);
        let value = if start_width >= (usize::BITS - width.leading_zeros()) as usize {
            let bound = self.sized_usize(start_width, width)?;
            let out_of_bounds = self.file.make_ge(&start.to_expr(), &bound);
            self.file
                .make_ternary(&out_of_bounds, &original.to_expr(), &updated)
        } else {
            updated
        };
        let result = self.file.function_result(handle).to_expr();
        let body = self.file.function_body(handle);
        self.file
            .block_add_blocking_assignment(body, &result, &value);
        Ok(())
    }
}

impl SliceHelper {
    /// Omits operations whose empty result or operand requires no helper.
    fn for_node(function: &Fn, node: &Node) -> Option<Self> {
        if node.ty.bit_count() == 0 {
            return None;
        }
        match node.payload {
            NodePayload::DynamicBitSlice { arg, start, width }
                if function.get_node_ty(arg).bit_count() != 0 =>
            {
                Some(Self::Dynamic {
                    operand_width: function.get_node_ty(arg).bit_count(),
                    start_width: function.get_node_ty(start).bit_count(),
                    width,
                })
            }
            NodePayload::BitSliceUpdate {
                start,
                update_value,
                ..
            } if function.get_node_ty(update_value).bit_count() != 0 => Some(Self::Update {
                width: node.ty.bit_count(),
                start_width: function.get_node_ty(start).bit_count(),
                update_width: function.get_node_ty(update_value).bit_count(),
            }),
            _ => None,
        }
    }

    fn name(self) -> String {
        match self {
            Self::Dynamic {
                operand_width,
                start_width,
                width,
            } => {
                format!("dynamic_bit_slice_w{width}_{operand_width}b_{start_width}b")
            }
            Self::Update {
                width,
                start_width,
                update_width,
            } => {
                format!("bit_slice_update_w{width}_{start_width}b_{update_width}b")
            }
        }
    }
}
