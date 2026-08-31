// SPDX-License-Identifier: Apache-2.0

//! Width-specialized arithmetic helpers with XLS-compatible sizing boundaries.

use xlsynth_pir::ir::{Binop, NodePayload, Type};
use xlsynth_vast::Expr;

use crate::BlockCodegenError;
use crate::block::{BlockEmitter, Value};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct ArithmeticHelper {
    kind: ArithmeticKind,
    lhs_width: usize,
    rhs_width: usize,
    width: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ArithmeticKind {
    Multiply { signed: bool, partial: bool },
    Divide { signed: bool, remainder: bool },
}

impl BlockEmitter<'_, '_> {
    /// Declares each arithmetic specialization before module logic or stages.
    pub(crate) fn emit_arithmetic_helpers(&mut self) -> Result<(), BlockCodegenError> {
        for node in &self.func.nodes {
            let NodePayload::Binop(op, lhs, rhs) = node.payload else {
                continue;
            };
            let Some(helper) = ArithmeticHelper::for_operation(
                op,
                self.func.get_node_ty(lhs).bit_count(),
                self.func.get_node_ty(rhs).bit_count(),
                &node.ty,
            ) else {
                continue;
            };
            if !self.arithmetic_helpers.contains_key(&helper) {
                let name = self.unique_name(&helper.name());
                self.emit_arithmetic_helper(helper, &name)?;
                self.arithmetic_helpers.insert(helper, name);
            }
        }
        Ok(())
    }

    /// Calls a helper without widening, truncating, or signing its operands.
    pub(crate) fn emit_arithmetic_call(
        &mut self,
        op: Binop,
        lhs: Value,
        rhs: Value,
        lhs_width: usize,
        rhs_width: usize,
        result_ty: &Type,
    ) -> Result<Expr, BlockCodegenError> {
        let helper = ArithmeticHelper::for_operation(op, lhs_width, rhs_width, result_ty)
            .expect("arithmetic node has a nonzero result");
        let name = self
            .arithmetic_helpers
            .get(&helper)
            .expect("arithmetic helper was predeclared");
        Ok(self.file.make_function_call(name, &[&lhs.expr, &rhs.expr]))
    }

    /// Uses explicit signed temporaries and result widths to contain SV sizing.
    fn emit_arithmetic_helper(
        &mut self,
        helper: ArithmeticHelper,
        name: &str,
    ) -> Result<(), BlockCodegenError> {
        let width = helper.width;
        let partial = matches!(helper.kind, ArithmeticKind::Multiply { partial: true, .. });
        let result_type = self.bits_type(if partial { 2 * width } else { width });
        let handle = self.file.add_function(self.module, name, &result_type)?;
        // Zero-width numeric operands are represented by a constant scalar
        // zero.
        let lhs_type = self.bits_type(helper.lhs_width.max(1));
        let rhs_type = self.bits_type(helper.rhs_width.max(1));
        let lhs = self
            .file
            .function_add_logic_input(handle, "lhs", &lhs_type)?;
        let rhs = self
            .file
            .function_add_logic_input(handle, "rhs", &rhs_type)?;
        let body = self.file.function_body(handle);
        let result = self.file.function_result(handle).to_expr();
        let value = match helper.kind {
            ArithmeticKind::Divide { signed, remainder } => {
                let op = match (signed, remainder) {
                    (false, false) => Binop::Udiv,
                    (false, true) => Binop::Umod,
                    (true, false) => Binop::Sdiv,
                    (true, true) => Binop::Smod,
                };
                self.emit_division(
                    op,
                    Value::signal(lhs).with_width(width),
                    Value::signal(rhs).with_width(width),
                    width,
                    width,
                    width,
                )?
            }
            ArithmeticKind::Multiply { signed, partial } => {
                let product = if signed {
                    let signed_lhs_type = self
                        .file
                        .make_bit_vector_type(helper.lhs_width.max(1) as i64, true);
                    let signed_rhs_type = self
                        .file
                        .make_bit_vector_type(helper.rhs_width.max(1) as i64, true);
                    let signed_result_type = self.file.make_bit_vector_type(width as i64, true);
                    let signed_lhs =
                        self.file
                            .function_add_logic(handle, "signed_lhs", &signed_lhs_type)?;
                    let signed_rhs =
                        self.file
                            .function_add_logic(handle, "signed_rhs", &signed_rhs_type)?;
                    let signed_result = self.file.function_add_logic(
                        handle,
                        "signed_result",
                        &signed_result_type,
                    )?;
                    let left = self.file.make_function_call("$signed", &[&lhs.to_expr()]);
                    let right = self.file.make_function_call("$signed", &[&rhs.to_expr()]);
                    self.file
                        .block_add_blocking_assignment(body, &signed_lhs.to_expr(), &left);
                    self.file
                        .block_add_blocking_assignment(body, &signed_rhs.to_expr(), &right);
                    let multiplied = self
                        .file
                        .make_mul(&signed_lhs.to_expr(), &signed_rhs.to_expr());
                    self.file.block_add_blocking_assignment(
                        body,
                        &signed_result.to_expr(),
                        &multiplied,
                    );
                    self.file
                        .make_function_call("$unsigned", &[&signed_result.to_expr()])
                } else {
                    self.file.make_mul(&lhs.to_expr(), &rhs.to_expr())
                };
                if partial {
                    let part_type = self.bits_type(width);
                    let product_name = if signed { "unsigned_result" } else { "result" };
                    let product_ref =
                        self.file
                            .function_add_logic(handle, product_name, &part_type)?;
                    self.file
                        .block_add_blocking_assignment(body, &product_ref.to_expr(), &product);
                    let offset = self.mulp_offset(width)?;
                    let remainder = self.file.make_sub(&product_ref.to_expr(), &offset);
                    let remainder = if signed {
                        let difference =
                            self.file
                                .function_add_logic(handle, "offset_result", &part_type)?;
                        self.file.block_add_blocking_assignment(
                            body,
                            &difference.to_expr(),
                            &remainder,
                        );
                        difference.to_expr()
                    } else {
                        remainder
                    };
                    self.file.make_concat(&[&offset, &remainder])
                } else {
                    product
                }
            }
        };
        self.file
            .block_add_blocking_assignment(body, &result, &value);
        Ok(())
    }
}

impl ArithmeticHelper {
    /// Records the original operand widths and independently sized result.
    fn for_operation(
        op: Binop,
        lhs_width: usize,
        rhs_width: usize,
        result_ty: &Type,
    ) -> Option<Self> {
        if result_ty.bit_count() == 0 {
            return None;
        }
        let kind = match op {
            Binop::Umul | Binop::Smul | Binop::Umulp | Binop::Smulp => ArithmeticKind::Multiply {
                signed: matches!(op, Binop::Smul | Binop::Smulp),
                partial: matches!(op, Binop::Umulp | Binop::Smulp),
            },
            Binop::Udiv | Binop::Sdiv | Binop::Umod | Binop::Smod => ArithmeticKind::Divide {
                signed: matches!(op, Binop::Sdiv | Binop::Smod),
                remainder: matches!(op, Binop::Umod | Binop::Smod),
            },
            _ => return None,
        };
        let width = if matches!(kind, ArithmeticKind::Multiply { partial: true, .. }) {
            result_ty.bit_count() / 2
        } else {
            result_ty.bit_count()
        };
        Some(Self {
            kind,
            lhs_width,
            rhs_width,
            width,
        })
    }

    fn name(self) -> String {
        match self.kind {
            ArithmeticKind::Multiply { signed, partial } => {
                let sign = if signed { "s" } else { "u" };
                let suffix = if partial { "p" } else { "" };
                format!(
                    "{sign}mul{suffix}{}b_{}b_x_{}b",
                    self.width, self.lhs_width, self.rhs_width
                )
            }
            ArithmeticKind::Divide { signed, remainder } => {
                let sign = if signed { "s" } else { "u" };
                let op = if remainder { "mod" } else { "div" };
                format!("{sign}{op}_{}b", self.width)
            }
        }
    }
}
