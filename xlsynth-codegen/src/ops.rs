// SPDX-License-Identifier: Apache-2.0

//! Faithful lowering of standard XLS scalar and aggregate operations.

use xlsynth_pir::ir::{Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::math::ceil_log2;
use xlsynth_pir::{IrBits, IrValue};
use xlsynth_vast::{Expr, GenerateLoop, LiteralFormat, LogicRef};

use crate::BlockCodegenError;
use crate::block::{BlockEmitter, Value, validate_external_identifier};

impl BlockEmitter<'_, '_> {
    /// Lowers one representable combinational or connectivity operation.
    pub(crate) fn emit_node(&mut self, node_ref: NodeRef) -> Result<(), BlockCodegenError> {
        let node = self.func.get_node(node_ref);
        if node.ty.bit_count() != 0 {
            match &node.payload {
                NodePayload::ArrayUpdate {
                    array,
                    value,
                    indices,
                    ..
                } if !indices.is_empty() => {
                    self.emit_generated_array_update(node_ref, *array, *value, indices)?;
                    return Ok(());
                }
                NodePayload::ArraySlice {
                    array,
                    start,
                    width,
                } if *width > 1 => {
                    self.emit_generated_array_slice(node_ref, *array, *start, *width)?;
                    return Ok(());
                }
                _ => {
                    // Other nodes use ordinary expression emission.
                }
            }
        }
        let result = match &node.payload {
            NodePayload::InstantiationInput {
                instantiation,
                port_name,
                arg,
            } => {
                if let Some(value) = self.values[arg.index] {
                    self.instance_inputs
                        .insert((instantiation.clone(), port_name.clone()), value.expr);
                }
                None
            }
            NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } if node.ty.bit_count() != 0 => {
                let requested = node
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("{instantiation}_{port_name}"));
                let name = self.unique_name(&requested);
                let ty = self.value_type(&node.ty);
                let signal = self.file.add_logic(self.module, &name, &ty)?;
                self.instance_outputs
                    .insert((instantiation.clone(), port_name.clone()), signal.to_expr());
                self.values[node_ref.index] = Some(Value::signal(signal).with_type(&node.ty));
                return Ok(());
            }
            NodePayload::Assert {
                activate,
                message,
                label,
                ..
            } => {
                if self.options.emit_asserts {
                    self.emit_assertion(*activate, message, label)?;
                }
                None
            }
            NodePayload::Cover { predicate, label } => {
                self.emit_cover(*predicate, label)?;
                None
            }
            NodePayload::Trace {
                activated,
                format,
                operands,
                ..
            } => {
                self.emit_trace(*activated, format, operands)?;
                None
            }
            NodePayload::Nil
            | NodePayload::AfterAll(_)
            | NodePayload::RegisterWrite { .. }
            | NodePayload::InstantiationOutput { .. } => None,
            NodePayload::Literal(value) => self
                .literal(value, &node.ty)?
                .map(|expression| Value::expression(expression, 0)),
            _ if node.ty.bit_count() == 0 => None,
            NodePayload::Tuple(elements) => {
                let expressions = self.represented_operands(elements)?;
                self.concat_or_only(&expressions)
                    .map(|expression| Value::expression(expression, self.operand_depth(elements)))
            }
            NodePayload::Array(elements) => {
                let mut expressions = self.represented_operands(elements)?;
                expressions.reverse();
                self.concat_or_only(&expressions)
                    .map(|expression| Value::expression(expression, self.operand_depth(elements)))
            }
            NodePayload::ArrayConcat(arrays) => {
                let mut expressions = self.represented_operands(arrays)?;
                expressions.reverse();
                self.concat_or_only(&expressions)
                    .map(|expression| Value::expression(expression, self.operand_depth(arrays)))
            }
            NodePayload::TupleIndex { tuple, index } => {
                let source = self.required_value(*tuple)?;
                let slice = self
                    .func
                    .get_node_ty(*tuple)
                    .tuple_get_flat_bit_slice_for_index(*index)
                    .map_err(BlockCodegenError::InvalidBlock)?;
                let expression = self.static_slice(source, slice.start, node.ty.bit_count());
                Some(Value::static_slice(
                    expression,
                    source,
                    slice.start,
                    node.ty.bit_count(),
                ))
            }
            NodePayload::Binop(op, lhs, rhs) => {
                let expression = self.emit_binop(*op, *lhs, *rhs, &node.ty)?;
                Some(Value::expression(
                    expression,
                    self.operand_depth(&[*lhs, *rhs]),
                ))
            }
            NodePayload::Unop(op, arg) => {
                if self.func.get_node_ty(*arg).bit_count() == 0 {
                    let value = usize::from(matches!(op, Unop::AndReduce));
                    let expression = self.sized_usize(node.ty.bit_count(), value)?;
                    Some(Value::expression(expression, 0))
                } else {
                    let value = self.required_value(*arg)?;
                    let expression = self.emit_unop(*op, value, node.ty.bit_count())?;
                    Some(Value::expression(expression, value.depth + 1))
                }
            }
            NodePayload::ZeroExt { arg, new_bit_count } => {
                let value = self.numeric_value(*arg)?;
                let expression = self.resize_unsigned(
                    value,
                    self.func.get_node_ty(*arg).bit_count(),
                    *new_bit_count,
                )?;
                Some(Value::expression(expression, value.depth + 1))
            }
            NodePayload::SignExt { arg, new_bit_count } => {
                let value = self.numeric_value(*arg)?;
                let expression = self.resize_signed(
                    value,
                    self.func.get_node_ty(*arg).bit_count(),
                    *new_bit_count,
                )?;
                Some(Value::expression(expression, value.depth + 1))
            }
            NodePayload::BitSlice { arg, start, width } => {
                let value = self.required_value(*arg)?;
                let expression = self.static_slice(value, *start, *width);
                Some(Value::static_slice(expression, value, *start, *width))
            }
            NodePayload::DynamicBitSlice { arg, start, .. } => {
                let expression = self.emit_slice_call(node_ref)?;
                Some(Value::expression(
                    expression,
                    self.operand_depth(&[*arg, *start]),
                ))
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                let expression = self.emit_slice_call(node_ref)?;
                Some(Value::expression(
                    expression,
                    self.operand_depth(&[*arg, *start, *update_value]),
                ))
            }
            NodePayload::ArrayIndex {
                array,
                indices,
                assumed_in_bounds,
            } => Some(self.emit_array_index(node_ref, *array, indices, *assumed_in_bounds)?),
            NodePayload::ArrayUpdate { value, .. } => {
                // Indexed updates are generated above; an empty index list
                // replaces the entire array.
                Some(self.required_value(*value)?)
            }
            NodePayload::ArraySlice {
                array,
                start,
                width,
            } => {
                let expression = self.emit_array_slice(*array, *start, *width)?;
                Some(Value::expression(
                    expression,
                    self.operand_depth(&[*array, *start]),
                ))
            }
            NodePayload::Nary(op, operands) => {
                let expression = self.emit_nary(*op, operands, node.ty.bit_count())?;
                expression
                    .map(|expression| Value::expression(expression, self.operand_depth(operands)))
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                let expression = self.emit_sel(*selector, cases, *default)?;
                let mut refs = vec![*selector];
                refs.extend(cases);
                refs.extend(default);
                Some(Value::expression(expression, self.operand_depth(&refs)))
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                let expression = self.emit_priority_call(node_ref)?;
                let mut refs = vec![*selector];
                refs.extend(cases);
                refs.extend(default);
                Some(Value::expression(expression, self.operand_depth(&refs)))
            }
            NodePayload::OneHotSel { selector, cases } => {
                let expression = self.emit_one_hot_sel(*selector, cases, node.ty.bit_count())?;
                let mut refs = vec![*selector];
                refs.extend(cases);
                Some(Value::expression(expression, self.operand_depth(&refs)))
            }
            NodePayload::OneHot { arg, .. } => {
                let expression = self.emit_priority_call(node_ref)?;
                Some(Value::expression(
                    expression,
                    self.numeric_value(*arg)?.depth + 1,
                ))
            }
            NodePayload::Encode { arg } => {
                let expression = self.emit_encode(*arg, node.ty.bit_count())?;
                Some(Value::expression(
                    expression,
                    self.required_value(*arg)?.depth + 1,
                ))
            }
            NodePayload::Decode { arg, width } => {
                let argument = self.numeric_value(*arg)?;
                let one = self.sized_usize(*width, 1)?;
                let expression = self.file.make_shll(&one, &argument.expr);
                Some(Value::expression(expression, argument.depth + 1))
            }
            NodePayload::Invoke { .. } => {
                unreachable!("invoke nodes are rejected before emission")
            }
            NodePayload::CountedFor { .. } => {
                unreachable!("counted_for nodes are rejected before emission")
            }
            NodePayload::ExtCarryOut { lhs, rhs, c_in } => {
                let operand_depth = self.operand_depth(&[*lhs, *rhs, *c_in]);
                let width = self.func.get_node_ty(*lhs).bit_count() + 1;
                let lhs_value = self.required_value(*lhs)?;
                let rhs_value = self.required_value(*rhs)?;
                let carry_value = self.required_value(*c_in)?;
                let lhs = self.resize_unsigned(
                    lhs_value,
                    self.func.get_node_ty(*lhs).bit_count(),
                    width,
                )?;
                let rhs = self.resize_unsigned(
                    rhs_value,
                    self.func.get_node_ty(*rhs).bit_count(),
                    width,
                )?;
                let carry = self.resize_unsigned(carry_value, 1, width)?;
                let partial = self.file.make_add(&lhs, &rhs);
                let sum = self.file.make_add(&partial, &carry);
                let expression = self.static_slice(
                    Value::expression(sum, operand_depth).with_width(width),
                    width - 1,
                    1,
                );
                Some(Value::expression(expression, operand_depth))
            }
            NodePayload::ExtPrioEncode { arg, .. }
            | NodePayload::ExtClz { arg, .. }
            | NodePayload::ExtNormalizeLeft { arg, .. } => {
                let expression = self.emit_priority_call(node_ref)?;
                Some(Value::expression(expression, self.operand_depth(&[*arg])))
            }
            NodePayload::ExtMaskLow { count } => {
                let expression = self.emit_mask_low(*count, node.ty.bit_count())?;
                Some(Value::expression(expression, self.operand_depth(&[*count])))
            }
            NodePayload::ExtNaryAdd { terms, .. } => {
                let expression = self.emit_extended_nary_add(terms, node.ty.bit_count())?;
                let refs = terms.iter().map(|term| term.operand).collect::<Vec<_>>();
                Some(Value::expression(expression, self.operand_depth(&refs)))
            }
            NodePayload::GetParam(_) | NodePayload::RegisterRead { .. } => {
                return Err(BlockCodegenError::InvalidBlock(format!(
                    "node `{}` should have been represented by an existing signal",
                    node.payload.get_operator()
                )));
            }
        };
        if let Some(value) = result {
            let value = if self.should_assign(node_ref, value.depth) {
                self.assign_node(node_ref, value.expr)?
            } else {
                value
            };
            self.values[node_ref.index] = Some(value.with_width(node.ty.bit_count()));
        }
        Ok(())
    }

    /// Returns one greater than the maximum represented operand depth.
    fn operand_depth(&self, nodes: &[NodeRef]) -> usize {
        nodes
            .iter()
            .filter_map(|node| self.values[node.index])
            .map(|value| value.depth)
            .max()
            .unwrap_or(0)
            + 1
    }

    /// Retrieves operands with nonzero packed widths in their source order.
    fn represented_operands(&self, nodes: &[NodeRef]) -> Result<Vec<Expr>, BlockCodegenError> {
        let mut values = Vec::with_capacity(nodes.len());
        for &node in nodes {
            if self.func.get_node_ty(node).bit_count() != 0 {
                values.push(self.required_value(node)?.expr);
            }
        }
        Ok(values)
    }

    /// Represents a zero-width numeric operand by its unique constant value.
    pub(crate) fn numeric_value(&mut self, node: NodeRef) -> Result<Value, BlockCodegenError> {
        if self.func.get_node_ty(node).bit_count() == 0 {
            Ok(Value::expression(self.zero(1)?, 0).with_width(1))
        } else {
            self.required_value(node)
        }
    }

    /// Builds a fixed-width, arbitrary-host-size unsigned integer literal.
    pub(crate) fn sized_usize(
        &mut self,
        width: usize,
        value: usize,
    ) -> Result<Expr, BlockCodegenError> {
        let needed = (usize::BITS - value.leading_zeros()) as usize;
        let width = width.max(needed).max(1);
        Ok(self
            .file
            .make_literal(&format!("bits[{width}]:{value}"), &LiteralFormat::Hex)?)
    }

    /// Returns a known zero at the required nonzero width.
    pub(crate) fn zero(&mut self, width: usize) -> Result<Expr, BlockCodegenError> {
        self.sized_usize(width, 0)
    }

    /// Casts an expression to its precisely required packed width.
    fn width_cast(&mut self, width: usize, expression: Expr) -> Expr {
        let size = self.file.make_unsized_decimal_literal(width as i32);
        self.file.make_width_cast(&size, &expression)
    }

    /// Extracts one statically located packed slice from a value expression.
    pub(crate) fn static_slice(&mut self, value: Value, start: usize, width: usize) -> Expr {
        if start == 0 && value.bit_width == Some(width) {
            return value.expr;
        }
        let (indexable, start) = if let Some((origin, offset)) = value.static_origin {
            (origin, offset + start)
        } else if let Some(indexable) = value.indexable.filter(|_| value.array_rank == 0) {
            (indexable, start)
        } else {
            let shifted = if start == 0 {
                value.expr
            } else {
                let amount = self.file.make_unsized_decimal_literal(start as i32);
                self.file.make_shrl(&value.expr, &amount)
            };
            return self.width_cast(width, shifted);
        };
        if width == 1 {
            self.file.make_index(&indexable, start as i64).to_expr()
        } else {
            self.file
                .make_slice(&indexable, (start + width - 1) as i64, start as i64)
                .to_expr()
        }
    }

    /// Zero-extends or truncates a packed expression without sign ambiguity.
    pub(crate) fn resize_unsigned(
        &mut self,
        value: Value,
        current: usize,
        target: usize,
    ) -> Result<Expr, BlockCodegenError> {
        if current == 0 {
            return self.zero(target);
        }
        if current == target {
            return Ok(value.expr);
        }
        if target < current {
            return Ok(self.static_slice(value, 0, target));
        }
        let zero = self.zero(target - current)?;
        Ok(self.file.make_concat(&[&zero, &value.expr]))
    }

    /// Sign-extends or truncates a packed expression without host-width limits.
    fn resize_signed(
        &mut self,
        value: Value,
        current: usize,
        target: usize,
    ) -> Result<Expr, BlockCodegenError> {
        if current == 0 {
            return self.zero(target);
        }
        if current >= target {
            return self.resize_unsigned(value, current, target);
        }
        let sign = self.static_slice(value, current - 1, 1);
        let repeats = self
            .file
            .make_replicated_concat_i64((target - current) as i64, &[&sign]);
        Ok(self.file.make_concat(&[&repeats, &value.expr]))
    }

    /// Returns a signed interpretation preserving the original packed width.
    fn signed(&mut self, expression: Expr) -> Expr {
        self.file.make_function_call("$signed", &[&expression])
    }

    /// Lowers ordinary arithmetic, comparison, shift, division, and gate nodes.
    fn emit_binop(
        &mut self,
        op: Binop,
        lhs_ref: NodeRef,
        rhs_ref: NodeRef,
        result_ty: &Type,
    ) -> Result<Expr, BlockCodegenError> {
        let lhs = self.numeric_value(lhs_ref)?;
        let rhs = self.numeric_value(rhs_ref)?;
        let lhs_width = self.func.get_node_ty(lhs_ref).bit_count();
        let rhs_width = self.func.get_node_ty(rhs_ref).bit_count();
        let result_width = result_ty.bit_count();
        let expression = match op {
            Binop::Add => self.file.make_add(&lhs.expr, &rhs.expr),
            Binop::Sub => self.file.make_sub(&lhs.expr, &rhs.expr),
            Binop::Shll => self.file.make_shll(&lhs.expr, &rhs.expr),
            Binop::Shrl => self.file.make_shrl(&lhs.expr, &rhs.expr),
            Binop::Shra => {
                let signed_lhs = self.signed(lhs.expr);
                let shifted = self.file.make_shra(&signed_lhs, &rhs.expr);
                self.file.make_function_call("$unsigned", &[&shifted])
            }
            Binop::Eq => self.file.make_eq(&lhs.expr, &rhs.expr),
            Binop::Ne => self.file.make_ne(&lhs.expr, &rhs.expr),
            Binop::Uge => self.file.make_ge(&lhs.expr, &rhs.expr),
            Binop::Ugt => self.file.make_gt(&lhs.expr, &rhs.expr),
            Binop::Ult => self.file.make_lt(&lhs.expr, &rhs.expr),
            Binop::Ule => self.file.make_le(&lhs.expr, &rhs.expr),
            Binop::Sgt | Binop::Sge | Binop::Slt | Binop::Sle => {
                let left = self.signed(lhs.expr);
                let right = self.signed(rhs.expr);
                match op {
                    Binop::Sgt => self.file.make_gt(&left, &right),
                    Binop::Sge => self.file.make_ge(&left, &right),
                    Binop::Slt => self.file.make_lt(&left, &right),
                    Binop::Sle => self.file.make_le(&left, &right),
                    _ => unreachable!("only signed comparisons enter this branch"),
                }
            }
            Binop::Umul
            | Binop::Smul
            | Binop::Umulp
            | Binop::Smulp
            | Binop::Udiv
            | Binop::Umod
            | Binop::Sdiv
            | Binop::Smod => {
                self.emit_arithmetic_call(op, lhs, rhs, lhs_width, rhs_width, result_ty)?
            }
            Binop::Gate => {
                let zero = self.zero(result_width)?;
                self.file.make_ternary(&lhs.expr, &rhs.expr, &zero)
            }
        };
        Ok(expression)
    }

    /// Preserves XLS division-by-zero semantics for signed and unsigned values.
    pub(crate) fn emit_division(
        &mut self,
        op: Binop,
        lhs: Value,
        rhs: Value,
        lhs_width: usize,
        rhs_width: usize,
        result_width: usize,
    ) -> Result<Expr, BlockCodegenError> {
        let rhs_zero = self.zero(rhs_width)?;
        let is_zero = self.file.make_eq(&rhs.expr, &rhs_zero);
        let left = if matches!(op, Binop::Sdiv | Binop::Smod) {
            let resized = self.resize_signed(lhs, lhs_width, result_width)?;
            self.signed(resized)
        } else {
            self.resize_unsigned(lhs, lhs_width, result_width)?
        };
        let right = if matches!(op, Binop::Sdiv | Binop::Smod) {
            let resized = self.resize_signed(rhs, rhs_width, result_width)?;
            self.signed(resized)
        } else {
            self.resize_unsigned(rhs, rhs_width, result_width)?
        };
        let mut arithmetic = if matches!(op, Binop::Umod | Binop::Smod) {
            self.file.make_mod(&left, &right)
        } else {
            self.file.make_div(&left, &right)
        };
        if matches!(op, Binop::Sdiv | Binop::Smod) {
            arithmetic = self.file.make_function_call("$unsigned", &[&arithmetic]);
        }
        if op == Binop::Sdiv {
            let minimum = self
                .literal(
                    &IrValue::signed_min_bits(result_width),
                    &Type::Bits(result_width),
                )?
                .expect("signed division has nonzero result width");
            let negative_one = self
                .literal(
                    &IrValue::from_bits(&IrBits::all_ones(result_width)),
                    &Type::Bits(result_width),
                )?
                .expect("signed division has nonzero result width");
            let numerator_is_minimum = self.file.make_eq(&lhs.expr, &minimum);
            let denominator_is_negative_one = self.file.make_eq(&rhs.expr, &negative_one);
            let overflow = self
                .file
                .make_logical_and(&numerator_is_minimum, &denominator_is_negative_one);
            arithmetic = self.file.make_ternary(&overflow, &minimum, &arithmetic);
        }
        let on_zero = match op {
            Binop::Udiv => self.file.make_unsized_one_literal(),
            Binop::Umod | Binop::Smod => self.zero(result_width)?,
            Binop::Sdiv => {
                let sign = self.static_slice(lhs, lhs_width - 1, 1);
                let min = self
                    .literal(
                        &IrValue::signed_min_bits(result_width),
                        &Type::Bits(result_width),
                    )?
                    .expect("signed minimum has nonzero width");
                let max = self
                    .literal(
                        &IrValue::signed_max_bits(result_width),
                        &Type::Bits(result_width),
                    )?
                    .expect("signed maximum has nonzero width");
                self.file.make_ternary(&sign, &min, &max)
            }
            _ => unreachable!("division helper receives only division or modulo"),
        };
        Ok(self.file.make_ternary(&is_zero, &on_zero, &arithmetic))
    }

    /// Constructs the public XLS simulation offset at the partial-product
    /// width.
    pub(crate) fn mulp_offset(&mut self, width: usize) -> Result<Expr, BlockCodegenError> {
        let low_width = width.saturating_sub(2);
        let high_width = width - low_width;
        let low_shift = low_width.saturating_sub(1).min(4);
        let mut bits = vec![false; width];
        for bit in bits.iter_mut().take(low_width.saturating_sub(low_shift)) {
            *bit = true;
        }
        for bit in bits
            .iter_mut()
            .skip(low_width)
            .take(high_width.saturating_sub(1))
        {
            *bit = true;
        }
        let offset_value = IrValue::from_bits(&IrBits::from_lsb_is_0(&bits));
        let offset = self
            .literal(&offset_value, &Type::Bits(width))?
            .expect("partial-product width is nonzero");
        Ok(offset)
    }

    /// Lowers unary bit operations and explicit bit reversal.
    fn emit_unop(
        &mut self,
        op: Unop,
        value: Value,
        width: usize,
    ) -> Result<Expr, BlockCodegenError> {
        Ok(match op {
            Unop::Neg => self.file.make_negate(&value.expr),
            Unop::Not => self.file.make_not(&value.expr),
            Unop::Identity => value.expr,
            Unop::OrReduce => self.file.make_or_reduce(&value.expr),
            Unop::AndReduce => self.file.make_and_reduce(&value.expr),
            Unop::XorReduce => self.file.make_xor_reduce(&value.expr),
            Unop::Reverse => {
                let bits = (0..width)
                    .map(|index| self.static_slice(value, index, 1))
                    .collect::<Vec<_>>();
                self.concat_or_only(&bits)
                    .expect("nonzero reverse has at least one bit")
            }
        })
    }

    /// Combines variadic bitwise and concatenation operations
    /// deterministically.
    fn emit_nary(
        &mut self,
        op: NaryOp,
        operands: &[NodeRef],
        width: usize,
    ) -> Result<Option<Expr>, BlockCodegenError> {
        let expressions = self.represented_operands(operands)?;
        if op == NaryOp::Concat {
            return Ok(self.concat_or_only(&expressions));
        }
        let Some(mut result) = expressions.first().copied() else {
            return Ok(Some(self.zero(width)?));
        };
        for expression in expressions.iter().skip(1) {
            result = match op {
                NaryOp::And | NaryOp::Nand => self.file.make_bitwise_and(&result, expression),
                NaryOp::Or | NaryOp::Nor => self.file.make_bitwise_or(&result, expression),
                NaryOp::Xor => self.file.make_bitwise_xor(&result, expression),
                NaryOp::Concat => unreachable!("concatenation returns before reduction"),
            };
        }
        if matches!(op, NaryOp::Nand | NaryOp::Nor) {
            result = self.file.make_bitwise_not(&result);
        }
        Ok(Some(result))
    }

    /// Selects nested packed-array elements with XLS out-of-bounds semantics.
    fn emit_array_index(
        &mut self,
        node_ref: NodeRef,
        array: NodeRef,
        indices: &[NodeRef],
        assumed_in_bounds: bool,
    ) -> Result<Value, BlockCodegenError> {
        let mut value = self.required_value(array)?;
        let mut ty = self.func.get_node_ty(array);
        for (dimension, &index) in indices.iter().enumerate() {
            let Type::Array(array_ty) = ty else {
                return Err(BlockCodegenError::InvalidBlock(
                    "array_index has too many indices".to_owned(),
                ));
            };
            let index_value = self.numeric_value(index)?;
            value = self.dynamic_array_element(
                value,
                index_value,
                self.func.get_node_ty(index).bit_count(),
                array_ty.element_count,
                &array_ty.element_type,
                assumed_in_bounds,
                false,
            )?;
            ty = &array_ty.element_type;
            if dimension + 1 < indices.len() {
                // Materialize subarrays so each dynamic packed selection has
                // a named subject, including in simulators that restrict
                // chained dynamic indices.
                let name = self.node_name(node_ref);
                let name = self.unique_name(&format!("{name}__dim{dimension}"));
                let data_type = self.value_type(ty);
                let signal = self.file.add_logic(self.module, &name, &data_type)?;
                let assignment = self
                    .file
                    .make_continuous_assignment(&signal.to_expr(), &value.expr);
                self.file
                    .add_member_continuous_assignment(self.module, assignment);
                value = Value::signal(signal).with_type(ty);
            }
        }
        Ok(value)
    }

    /// Retrieves one packed-array element, optionally clamping invalid indices.
    fn dynamic_array_element(
        &mut self,
        value: Value,
        index: Value,
        index_width: usize,
        count: usize,
        element_type: &Type,
        assumed_in_bounds: bool,
        force_bounds_checking: bool,
    ) -> Result<Value, BlockCodegenError> {
        if count == 0 || element_type.bit_count() == 0 {
            return Err(BlockCodegenError::InvalidBlock(
                "nonempty array index requires representable array elements".to_owned(),
            ));
        }
        let bound_width = (usize::BITS - count.leading_zeros()) as usize;
        let checked_index = if index_width < bound_width {
            // Every unsigned index fits when 2^index_width <= count.
            index.expr
        } else {
            let bounds_checked = self.options.array_index_bounds_checking || force_bounds_checking;
            if bounds_checked && !assumed_in_bounds {
                let bound = self.sized_usize(index_width, count)?;
                let in_bounds = self.file.make_lt(&index.expr, &bound);
                let last = self.sized_usize(index_width, count - 1)?;
                self.file.make_ternary(&in_bounds, &index.expr, &last)
            } else {
                let address_width = ceil_log2(count).max(1);
                let truncated = self.resize_unsigned(index, index_width, address_width)?;
                self.resize_unsigned(
                    Value::expression(truncated, index.depth),
                    address_width,
                    index_width,
                )?
            }
        };
        self.array_element(value, checked_index, element_type)
            .map(|mut selected| {
                selected.depth = value.depth.max(index.depth) + 1;
                selected
            })
    }

    /// Selects one native packed dimension and preserves the remaining shape.
    fn array_element(
        &mut self,
        value: Value,
        index: Expr,
        element_type: &Type,
    ) -> Result<Value, BlockCodegenError> {
        let source = value
            .indexable
            .filter(|_| value.array_rank != 0)
            .ok_or_else(|| {
                BlockCodegenError::InvalidBlock(
                    "array elements require a materialized packed-array signal".to_owned(),
                )
            })?;
        let selected = self.file.make_index_expr(&source, &index);
        let indexable = selected.to_indexable_expr();
        Ok(Value {
            expr: selected.to_expr(),
            indexable: Some(indexable),
            static_origin: Some((indexable, 0)),
            ..Value::expression(selected.to_expr(), value.depth + 1)
        }
        .with_type(element_type))
    }

    /// Emits nested generate loops for one- or multi-dimensional array updates.
    fn emit_generated_array_update(
        &mut self,
        node_ref: NodeRef,
        array: NodeRef,
        replacement: NodeRef,
        indices: &[NodeRef],
    ) -> Result<(), BlockCodegenError> {
        let mut source = self.required_value(array)?;
        let replacement = self.required_value(replacement)?;
        let mut dimensions = Vec::with_capacity(indices.len());
        let mut current = self.func.get_node_ty(array);
        for &index in indices {
            let Type::Array(array_type) = current else {
                return Err(BlockCodegenError::InvalidBlock(
                    "array_update has too many indices".to_owned(),
                ));
            };
            dimensions.push((
                array_type.element_count,
                array_type.element_type.as_ref(),
                index,
            ));
            current = &array_type.element_type;
        }
        let signal = self.declare_node(node_ref)?;
        let target = Value::signal(signal).with_type(self.func.get_node_ty(node_ref));
        let mut destination = target;
        let mut parent = None;
        let mut index_matches = None;

        for (dimension, (count, element_type, index_ref)) in dimensions.into_iter().enumerate() {
            let generated = self.make_array_generate_loop(signal, parent, dimension, count)?;
            let variable = self.file.generate_genvar(generated).to_expr();
            destination = self.array_element(destination, variable, element_type)?;
            source = self.array_element(source, variable, element_type)?;
            let index = self.numeric_value(index_ref)?;
            let matched = self.file.make_eq(&index.expr, &variable);
            index_matches = Some(if let Some(previous) = index_matches {
                self.file.make_logical_and(&previous, &matched)
            } else {
                matched
            });
            parent = Some(generated);
        }

        let generated = parent.expect("indexed array update has at least one generate loop");
        let matched = index_matches.expect("indexed array update has an index comparison");
        let selected = self
            .file
            .make_ternary(&matched, &replacement.expr, &source.expr);
        self.file
            .generate_add_continuous_assignment(generated, &destination.expr, &selected);
        self.values[node_ref.index] = Some(target);
        Ok(())
    }

    /// Emits a generated array slice while preserving saturating XLS indexing.
    fn emit_generated_array_slice(
        &mut self,
        node_ref: NodeRef,
        array: NodeRef,
        start: NodeRef,
        width: usize,
    ) -> Result<(), BlockCodegenError> {
        let source = self.required_value(array)?;
        let start_value = self.numeric_value(start)?;
        let start_width = self.func.get_node_ty(start).bit_count();
        let Type::Array(array_type) = self.func.get_node_ty(array) else {
            return Err(BlockCodegenError::InvalidBlock(
                "array_slice source is not an array".to_owned(),
            ));
        };
        let count = array_type.element_count;
        let index_width = start_width
            .saturating_add(ceil_log2(width.max(1)))
            .max(ceil_log2(count.saturating_add(width).saturating_add(1)));
        let widened_start = self.resize_unsigned(start_value, start_width, index_width)?;
        let signal = self.declare_node(node_ref)?;
        let target = Value::signal(signal).with_type(self.func.get_node_ty(node_ref));
        let generated = self.make_array_generate_loop(signal, None, 0, width)?;
        let variable = self.file.generate_genvar(generated).to_expr();
        let bounded_variable = self.width_cast(index_width, variable);
        let index = self.file.make_add(&widened_start, &bounded_variable);
        let index = self.width_cast(index_width, index);
        let selected = self.dynamic_array_element(
            source,
            Value::expression(index, start_value.depth + 1).with_width(index_width),
            index_width,
            count,
            &array_type.element_type,
            false,
            true,
        )?;
        let destination = self.array_element(target, variable, &array_type.element_type)?;
        self.file
            .generate_add_continuous_assignment(generated, &destination.expr, &selected.expr);
        self.values[node_ref.index] = Some(target);
        Ok(())
    }

    /// Creates a labeled array loop with a short genvar reusable by siblings.
    fn make_array_generate_loop(
        &mut self,
        signal: LogicRef,
        parent: Option<GenerateLoop>,
        dimension: usize,
        count: usize,
    ) -> Result<GenerateLoop, BlockCodegenError> {
        let signal_name = self.file.logic_ref_name(signal);
        let variable = if let Some(name) = self.genvar_names.get(&dimension) {
            name.clone()
        } else {
            // Reserve the local name against module signals, including those
            // emitted later, without forcing sibling loops to use new names.
            let name = self.unique_name(&format!("__i{dimension}"));
            self.genvar_names.insert(dimension, name.clone());
            name
        };
        let label = self.unique_name(&format!("gen__{signal_name}_{dimension}"));
        let zero = self.file.make_unsized_decimal_literal(0);
        let limit = self.generate_index_literal(count)?;
        Ok(if let Some(parent) = parent {
            self.file
                .generate_add_generate_loop(parent, &variable, &zero, &limit, Some(&label))
        } else {
            self.file
                .add_generate_loop(self.module, &variable, &zero, &limit, Some(&label))
        })
    }

    /// Produces a portable nonnegative literal for a generated packed index.
    fn generate_index_literal(&mut self, value: usize) -> Result<Expr, BlockCodegenError> {
        let value = i32::try_from(value).map_err(|_| {
            BlockCodegenError::Unsupported(format!(
                "array generate dimension or packed width exceeds the supported limit: {value}"
            ))
        })?;
        Ok(self.file.make_unsized_decimal_literal(value))
    }

    /// Slices packed arrays and repeats the last element after the upper bound.
    fn emit_array_slice(
        &mut self,
        array: NodeRef,
        start: NodeRef,
        width: usize,
    ) -> Result<Expr, BlockCodegenError> {
        let source = self.required_value(array)?;
        let start_value = self.numeric_value(start)?;
        let start_width = self.func.get_node_ty(start).bit_count();
        let Type::Array(array_ty) = self.func.get_node_ty(array) else {
            return Err(BlockCodegenError::InvalidBlock(
                "array_slice source is not an array".to_owned(),
            ));
        };
        let index_width = start_width
            .saturating_add(ceil_log2(width.max(1)))
            .max(ceil_log2(
                array_ty
                    .element_count
                    .saturating_add(width)
                    .saturating_add(1),
            ));
        let widened_start = self.resize_unsigned(start_value, start_width, index_width)?;
        let mut elements = Vec::with_capacity(width);
        for offset in 0..width {
            let offset_value = self.sized_usize(index_width, offset)?;
            let index = self.file.make_add(&widened_start, &offset_value);
            let index = Value::expression(index, start_value.depth + 1);
            let element = self.dynamic_array_element(
                source,
                index,
                index_width,
                array_ty.element_count,
                &array_ty.element_type,
                false,
                true,
            )?;
            elements.push(element.expr);
        }
        elements.reverse();
        Ok(self
            .concat_or_only(&elements)
            .expect("nonzero-width array_slice has representable elements"))
    }

    /// Lowers a complete or defaulted selector into an exact equality mux.
    fn emit_sel(
        &mut self,
        selector: NodeRef,
        cases: &[NodeRef],
        default: Option<NodeRef>,
    ) -> Result<Expr, BlockCodegenError> {
        let selected = self.numeric_value(selector)?;
        let selector_width = self.func.get_node_ty(selector).bit_count();
        let mut result = if let Some(default) = default {
            self.required_value(default)?.expr
        } else {
            self.required_value(*cases.last().ok_or_else(|| {
                BlockCodegenError::InvalidBlock("sel requires a case or default".to_owned())
            })?)?
            .expr
        };
        for (index, &case) in cases.iter().enumerate().rev() {
            let index_value = self.sized_usize(selector_width, index)?;
            let condition = self.file.make_eq(&selected.expr, &index_value);
            let case_value = self.required_value(case)?;
            result = self
                .file
                .make_ternary(&condition, &case_value.expr, &result);
        }
        Ok(result)
    }

    /// OR-combines every selected one-hot case in its packed representation.
    fn emit_one_hot_sel(
        &mut self,
        selector: NodeRef,
        cases: &[NodeRef],
        width: usize,
    ) -> Result<Expr, BlockCodegenError> {
        let selected = self.numeric_value(selector)?;
        let zero = self.zero(width)?;
        let mut result = zero;
        for (index, &case) in cases.iter().enumerate() {
            let condition = self.static_slice(selected, index, 1);
            let value = self.required_value(case)?;
            let contribution = self.file.make_ternary(&condition, &value.expr, &zero);
            result = self.file.make_bitwise_or(&result, &contribution);
        }
        Ok(result)
    }

    /// Forms each encoded bit from input positions containing that index bit.
    fn emit_encode(&mut self, arg: NodeRef, width: usize) -> Result<Expr, BlockCodegenError> {
        let argument = self.required_value(arg)?;
        let arg_width = self.func.get_node_ty(arg).bit_count();
        let mut output_bits = Vec::with_capacity(width);
        for bit in (0..width).rev() {
            let mut result = None;
            for index in 0..arg_width {
                if index.checked_shr(bit as u32).unwrap_or(0) & 1 != 0 {
                    let selected = self.static_slice(argument, index, 1);
                    result = Some(match result {
                        Some(previous) => self.file.make_bitwise_or(&previous, &selected),
                        None => selected,
                    });
                }
            }
            output_bits.push(match result {
                Some(result) => result,
                None => self.zero(1)?,
            });
        }
        Ok(self
            .concat_or_only(&output_bits)
            .expect("encode result is nonzero width"))
    }

    /// Builds a dynamic low-bit mask with saturation at the result width.
    fn emit_mask_low(&mut self, count: NodeRef, width: usize) -> Result<Expr, BlockCodegenError> {
        let count_value = self.required_value(count)?;
        let count_width = self.func.get_node_ty(count).bit_count();
        let comparison_width = count_width.max(ceil_log2(width + 1));
        let expanded_count = self.resize_unsigned(count_value, count_width, comparison_width)?;
        let bound = self.sized_usize(comparison_width, width)?;
        let saturated = self.file.make_ge(&expanded_count, &bound);
        let one = self.sized_usize(width, 1)?;
        let shifted = self.file.make_shll(&one, &expanded_count);
        let mask = self.file.make_sub(&shifted, &one);
        let all = self.file.make_unsized_one_literal();
        Ok(self.file.make_ternary(&saturated, &all, &mask))
    }

    /// Resizes and combines extension adder terms using their declared signs.
    fn emit_extended_nary_add(
        &mut self,
        terms: &[xlsynth_pir::ir::ExtNaryAddTerm],
        width: usize,
    ) -> Result<Expr, BlockCodegenError> {
        let mut result = self.zero(width)?;
        for term in terms {
            let operand = self.required_value(term.operand)?;
            let operand_width = self.func.get_node_ty(term.operand).bit_count();
            let mut value = if term.signed {
                self.resize_signed(operand, operand_width, width)?
            } else {
                self.resize_unsigned(operand, operand_width, width)?
            };
            if term.negated {
                value = self.file.make_negate(&value);
            }
            result = self.file.make_add(&result, &value);
        }
        Ok(result)
    }

    /// Emits a clocked assertion without altering synthesizable datapath logic.
    fn emit_assertion(
        &mut self,
        predicate: NodeRef,
        message: &str,
        label: &str,
    ) -> Result<(), BlockCodegenError> {
        let condition = self.required_value(predicate)?.expr;
        let predicate = self.file.emit_expression(&condition);
        let escaped = escape_sv_string(message);
        if !label.is_empty() {
            validate_external_identifier(label, "assertion label")?;
        }
        let label_prefix = if label.is_empty() {
            String::new()
        } else {
            format!("{label}: ")
        };
        let unknown = format!("$isunknown({predicate})");
        let disabled = if let Some(reset) = &self.metadata.reset {
            let inactive = if reset.active_low { "1'b1" } else { "1'b0" };
            format!("{} !== {inactive} || {unknown}", reset.port_name)
        } else {
            unknown
        };
        let statement = if let Some(clock) = &self.metadata.clock_port_name {
            let disabled = if self
                .metadata
                .reset
                .as_ref()
                .is_some_and(|reset| reset.asynchronous)
            {
                disabled
            } else {
                format!("$sampled({disabled})")
            };
            format!(
                "`ifndef SYNTHESIS\n{label_prefix}assert property (@(posedge {clock}) \
                 disable iff ({disabled}) {predicate}) else $error(\"{escaped}\");\n`endif"
            )
        } else {
            let label_suffix = if label.is_empty() {
                String::new()
            } else {
                format!(" : {label}")
            };
            format!(
                "`ifndef SYNTHESIS\nalways_comb begin{label_suffix}\n  \
                 if (!({disabled})) assert ({predicate}) \
                 else $error(\"{escaped}\");\nend\n`endif"
            )
        };
        let inline = self.file.make_inline_verilog_statement(&statement);
        self.file.add_member_inline_statement(self.module, inline);
        Ok(())
    }

    /// Emits a clocked coverage property when a source clock is available.
    fn emit_cover(&mut self, predicate: NodeRef, label: &str) -> Result<(), BlockCodegenError> {
        validate_external_identifier(label, "coverage label")?;
        let condition = self.required_value(predicate)?.expr;
        let predicate = self.file.emit_expression(&condition);
        let statement = if let Some(clock) = &self.metadata.clock_port_name {
            format!(
                "`ifndef SYNTHESIS\n{label}: cover property (@(posedge {clock}) {predicate});\n`endif"
            )
        } else {
            format!(
                "`ifndef SYNTHESIS\nalways_comb begin : {label}\n  cover ({predicate});\nend\n`endif"
            )
        };
        let inline = self.file.make_inline_verilog_statement(&statement);
        self.file.add_member_inline_statement(self.module, inline);
        Ok(())
    }

    /// Emits an activation-gated display statement for source trace operations.
    fn emit_trace(
        &mut self,
        activated: NodeRef,
        format: &str,
        operands: &[NodeRef],
    ) -> Result<(), BlockCodegenError> {
        let condition = self.required_value(activated)?.expr;
        let condition = self.file.emit_expression(&condition);
        let (format, signed_operands) = verilog_trace_format(format, operands.len())?;
        let format = escape_sv_string(&format);
        let mut values = Vec::with_capacity(operands.len());
        for (&operand, &signed) in operands.iter().zip(&signed_operands) {
            let value = self.numeric_value(operand)?.expr;
            let value = if signed { self.signed(value) } else { value };
            values.push(self.file.emit_expression(&value));
        }
        let arguments = if values.is_empty() {
            String::new()
        } else {
            format!(", {}", values.join(", "))
        };
        let trigger = self
            .metadata
            .clock_port_name
            .as_ref()
            .map(|clock| format!("@(posedge {clock})"))
            .unwrap_or_else(|| "@(*)".to_owned());
        let statement = format!(
            "`ifndef SYNTHESIS\nalways {trigger} begin\n  if ({condition}) \
             $display(\"{format}\"{arguments});\nend\n`endif"
        );
        let inline = self.file.make_inline_verilog_statement(&statement);
        self.file.add_member_inline_statement(self.module, inline);
        Ok(())
    }
}

/// Converts XLS trace directives into their corresponding display directives.
fn verilog_trace_format(
    format: &str,
    operand_count: usize,
) -> Result<(String, Vec<bool>), BlockCodegenError> {
    let mut converted = String::with_capacity(format.len());
    let mut signed_operands = Vec::with_capacity(operand_count);
    let mut remaining = format;
    while !remaining.is_empty() {
        if let Some(rest) = remaining.strip_prefix("{{") {
            converted.push('{');
            remaining = rest;
        } else if let Some(rest) = remaining.strip_prefix("}}") {
            converted.push('}');
            remaining = rest;
        } else if let Some(rest) = remaining.strip_prefix('{') {
            let end = rest.find('}').ok_or_else(|| {
                BlockCodegenError::InvalidBlock(
                    "trace format contains an unterminated format directive".to_owned(),
                )
            })?;
            let (directive, rest) = rest.split_at(end);
            let (specifier, signed) = match directive {
                "" | ":u" => ("%0d", false),
                ":d" => ("%0d", true),
                ":x" => ("%0h", false),
                ":0x" => ("%h", false),
                ":#x" => ("0x%h", false),
                ":b" => ("%0b", false),
                ":0b" => ("%b", false),
                ":#b" => ("0b%b", false),
                _ => {
                    return Err(BlockCodegenError::InvalidBlock(format!(
                        "trace format contains unsupported directive `{{{directive}}}`"
                    )));
                }
            };
            converted.push_str(specifier);
            signed_operands.push(signed);
            remaining = &rest[1..];
        } else if remaining.starts_with('}') {
            return Err(BlockCodegenError::InvalidBlock(
                "trace format contains an unmatched closing brace".to_owned(),
            ));
        } else if let Some(rest) = remaining.strip_prefix('%') {
            converted.push_str("%%");
            remaining = rest;
        } else {
            let character = remaining
                .chars()
                .next()
                .expect("nonempty trace format has a leading character");
            converted.push(character);
            remaining = &remaining[character.len_utf8()..];
        }
    }
    if signed_operands.len() != operand_count {
        return Err(BlockCodegenError::InvalidBlock(format!(
            "trace format has {} directives for {operand_count} data operands",
            signed_operands.len()
        )));
    }
    Ok((converted, signed_operands))
}

/// Escapes a public diagnostic string for use in a SystemVerilog literal.
fn escape_sv_string(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}
