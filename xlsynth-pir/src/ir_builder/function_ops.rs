// SPDX-License-Identifier: Apache-2.0

//! Function operations with side effects, multioperand forms, and extensions.

use super::{BValue, BuilderError, FnBuilder};
use crate::ir::{self, Binop, NaryOp, NodePayload, Type};

/// One signed or unsigned contribution to an extended modular sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NaryAddTerm {
    pub operand: BValue,
    pub signed: bool,
    pub negated: bool,
}

impl NaryAddTerm {
    /// Adds an unsigned operand without negating it.
    pub fn unsigned(operand: BValue) -> Self {
        Self {
            operand,
            signed: false,
            negated: false,
        }
    }

    /// Adds a sign-extended operand without negating it.
    pub fn signed(operand: BValue) -> Self {
        Self {
            operand,
            signed: true,
            negated: false,
        }
    }

    /// Toggles subtraction of this term after extending it to the result width.
    pub fn negate(mut self) -> Self {
        self.negated = !self.negated;
        self
    }
}

/// Result width and optional lowering architecture for an extended sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NaryAddOptions {
    pub bit_count: usize,
    pub architecture: Option<ir::ExtNaryAddArchitecture>,
}

impl NaryAddOptions {
    /// Selects a result width without prescribing an adder architecture.
    pub fn new(bit_count: usize) -> Self {
        Self {
            bit_count,
            architecture: None,
        }
    }
}

/// Controls normalization width, additional left shift, and optional CLZ
/// output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizeLeftOptions {
    pub normalized_bit_count: usize,
    pub shift_offset: usize,
    pub clz_bit_count: Option<usize>,
}

impl NormalizeLeftOptions {
    /// Normalizes to the requested width without an extra shift or CLZ output.
    pub fn new(normalized_bit_count: usize) -> Self {
        Self {
            normalized_bit_count,
            shift_offset: 0,
            clz_bit_count: None,
        }
    }
}

macro_rules! bitwise_all {
    ($name:ident, $op:ident) => {
        #[doc = concat!("Adds a `", stringify!($op), "` operation on one or more equal-width operands.")]
        pub fn $name(&mut self, args: &[BValue]) -> Result<BValue, BuilderError> {
            self.add_node(NodePayload::Nary(NaryOp::$op, self.nodes(args)?), None)
        }
    };
}

impl FnBuilder {
    bitwise_all!(and_all, And);
    bitwise_all!(nand_all, Nand);
    bitwise_all!(or_all, Or);
    bitwise_all!(nor_all, Nor);
    bitwise_all!(xor_all, Xor);

    /// Returns the data value when the one-bit condition is true, otherwise
    /// zero.
    pub fn gate(&mut self, condition: BValue, value: BValue) -> Result<BValue, BuilderError> {
        let ty = self.get_type(value)?.clone();
        self.add_node(
            NodePayload::Binop(Binop::Gate, self.node(condition)?, self.node(value)?),
            Some(ty),
        )
    }

    /// Produces two unsigned partial products whose modular sum is the product.
    pub fn umulp(
        &mut self,
        lhs: BValue,
        rhs: BValue,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        self.partial_product(Binop::Umulp, lhs, rhs, width)
    }

    /// Produces two signed partial products whose modular sum is the product.
    pub fn smulp(
        &mut self,
        lhs: BValue,
        rhs: BValue,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        self.partial_product(Binop::Smulp, lhs, rhs, width)
    }

    fn partial_product(
        &mut self,
        operation: Binop,
        lhs: BValue,
        rhs: BValue,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::Binop(operation, self.node(lhs)?, self.node(rhs)?),
            Some(Type::Tuple(vec![
                Box::new(Type::Bits(width)),
                Box::new(Type::Bits(width)),
            ])),
        )
    }

    /// Joins token dependencies; an empty list creates an initial token.
    pub fn after_all(&mut self, tokens: &[BValue]) -> Result<BValue, BuilderError> {
        self.add_node(NodePayload::AfterAll(self.nodes(tokens)?), None)
    }

    /// Checks a one-bit condition, reporting a message when it is false.
    pub fn assert(
        &mut self,
        token: BValue,
        condition: BValue,
        message: &str,
        label: &str,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::Assert {
                token: self.node(token)?,
                activate: self.node(condition)?,
                message: message.to_string(),
                label: label.to_string(),
            },
            None,
        )
    }

    /// Emits a conditional trace using checked XLS format directives and arity.
    ///
    /// Data operands may be bits, aggregates, or tokens. Every directive
    /// formats one operand, recursively applying its preference to
    /// aggregate elements.
    pub fn trace(
        &mut self,
        token: BValue,
        condition: BValue,
        format: &str,
        operands: &[BValue],
        verbosity: i64,
    ) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::Trace {
                token: self.node(token)?,
                activated: self.node(condition)?,
                format: format.to_string(),
                verbosity,
                operands: self.nodes(operands)?,
            },
            None,
        )
    }

    /// Records a coverage event for a one-bit predicate, returning an empty
    /// tuple.
    pub fn cover(&mut self, predicate: BValue, label: &str) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::Cover {
                predicate: self.node(predicate)?,
                label: label.to_string(),
            },
            None,
        )
    }

    /// Computes the carry bit of equal-width operands plus a one-bit carry-in.
    pub fn ext_carry_out(
        &mut self,
        lhs: BValue,
        rhs: BValue,
        carry_in: BValue,
    ) -> Result<BValue, BuilderError> {
        self.bits_width(lhs)?
            .checked_add(1)
            .ok_or(BuilderError::WidthOverflow)?;
        self.add_node(
            NodePayload::ExtCarryOut {
                lhs: self.node(lhs)?,
                rhs: self.node(rhs)?,
                c_in: self.node(carry_in)?,
            },
            None,
        )
    }

    /// Encodes the priority set bit, using the input width as the zero
    /// sentinel.
    pub fn ext_prio_encode(
        &mut self,
        arg: BValue,
        lsb_is_priority: bool,
    ) -> Result<BValue, BuilderError> {
        self.bits_width(arg)?
            .checked_add(1)
            .ok_or(BuilderError::WidthOverflow)?;
        self.add_node(
            NodePayload::ExtPrioEncode {
                arg: self.node(arg)?,
                lsb_prio: lsb_is_priority,
            },
            None,
        )
    }

    /// Counts leading zeros plus an offset, modulo the explicit result width.
    pub fn ext_clz(
        &mut self,
        arg: BValue,
        offset: usize,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        self.bits_width(arg)?
            .checked_add(1)
            .ok_or(BuilderError::WidthOverflow)?;
        self.add_node(
            NodePayload::ExtClz {
                arg: self.node(arg)?,
                offset,
                new_bit_count: width,
            },
            None,
        )
    }

    /// Zero-extends and left-normalizes bits, optionally returning the CLZ too.
    pub fn ext_normalize_left(
        &mut self,
        arg: BValue,
        options: NormalizeLeftOptions,
    ) -> Result<BValue, BuilderError> {
        let input_width = self.bits_width(arg)?;
        if options.normalized_bit_count < input_width {
            return Err(BuilderError::InvalidOperation(format!(
                "ext_normalize_left normalized width {} must be at least input width {}",
                options.normalized_bit_count, input_width
            )));
        }
        input_width
            .checked_add(options.shift_offset)
            .and_then(|width| width.checked_add(1))
            .ok_or(BuilderError::WidthOverflow)?;
        self.add_node(
            NodePayload::ExtNormalizeLeft {
                arg: self.node(arg)?,
                shift_offset: options.shift_offset,
                normalized_bit_count: options.normalized_bit_count,
                clz_bit_count: options.clz_bit_count,
            },
            None,
        )
    }

    /// Produces a low-bit mask, saturating to all ones when count reaches
    /// width.
    pub fn ext_mask_low(&mut self, count: BValue, width: usize) -> Result<BValue, BuilderError> {
        self.add_node(
            NodePayload::ExtMaskLow {
                count: self.node(count)?,
            },
            Some(Type::Bits(width)),
        )
    }

    /// Sums resized signed/unsigned terms modulo the result width; empty is
    /// zero.
    pub fn ext_nary_add(
        &mut self,
        terms: &[NaryAddTerm],
        options: NaryAddOptions,
    ) -> Result<BValue, BuilderError> {
        let terms = terms
            .iter()
            .map(|term| {
                Ok(ir::ExtNaryAddTerm {
                    operand: self.node(term.operand)?,
                    signed: term.signed,
                    negated: term.negated,
                })
            })
            .collect::<Result<Vec<_>, BuilderError>>()?;
        self.add_node(
            NodePayload::ExtNaryAdd {
                terms,
                arch: options.architecture,
            },
            Some(Type::Bits(options.bit_count)),
        )
    }

    /// Indexes nested arrays while promising downstream passes the indices fit.
    ///
    /// Operand types and indexed dimensions are still validated. The caller is
    /// responsible for ensuring every runtime index is in bounds.
    pub fn array_index_assuming_in_bounds(
        &mut self,
        array: BValue,
        indices: &[BValue],
    ) -> Result<BValue, BuilderError> {
        let value = self.array_index_multi(array, indices)?;
        let NodePayload::ArrayIndex {
            assumed_in_bounds, ..
        } = &mut self.function.get_node_mut(value.node).payload
        else {
            unreachable!("array_index_multi builds an ArrayIndex node");
        };
        *assumed_in_bounds = true;
        Ok(value)
    }

    /// Updates an array while promising downstream passes the indices fit.
    ///
    /// Operand types and indexed dimensions are still validated. The caller is
    /// responsible for ensuring every runtime index is in bounds.
    pub fn array_update_assuming_in_bounds(
        &mut self,
        array: BValue,
        update: BValue,
        indices: &[BValue],
    ) -> Result<BValue, BuilderError> {
        let value = self.array_update(array, update, indices)?;
        let NodePayload::ArrayUpdate {
            assumed_in_bounds, ..
        } = &mut self.function.get_node_mut(value.node).payload
        else {
            unreachable!("array_update builds an ArrayUpdate node");
        };
        *assumed_in_bounds = true;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{NaryAddOptions, NaryAddTerm, NormalizeLeftOptions};
    use crate::IrValue;
    use crate::ir::{self, ExtNaryAddArchitecture, NodePayload, Type};
    use crate::ir_builder::{BValue, BuilderError, FnBuilder};
    use crate::ir_eval::{FnEvalResult, eval_fn};
    use crate::ir_parser::Parser;

    fn bits(width: usize, value: u64) -> IrValue {
        IrValue::make_ubits(width, value).unwrap()
    }

    /// Evaluates a pure constructed function and checks its textual roundtrip.
    fn evaluate(function: &ir::Fn, args: &[IrValue]) -> IrValue {
        let text = function.to_string();
        let parsed = Parser::new(&text).parse_fn().unwrap();
        assert_eq!(parsed.to_string(), text);
        let FnEvalResult::Success(result) = eval_fn(function, args) else {
            panic!("unexpected evaluation failure");
        };
        result.value
    }

    /// Checks that failed construction preserves the complete node sequence.
    fn rejects_without_changes(
        builder: &mut FnBuilder,
        operation: impl FnOnce(&mut FnBuilder) -> Result<BValue, BuilderError>,
    ) {
        let nodes = builder.function.nodes.clone();
        let last = builder.last_value();
        assert!(operation(builder).is_err());
        assert_eq!(builder.last_value(), last);
        assert_eq!(builder.function.nodes.len(), nodes.len());
        for (before, after) in nodes.iter().zip(&builder.function.nodes) {
            assert_eq!(before.text_id, after.text_id);
            assert_eq!(before.name, after.name);
            assert_eq!(before.ty, after.ty);
            assert_eq!(before.payload, after.payload);
        }
    }

    #[test]
    fn bitwise_multioperand_and_gate_semantics() {
        let mut b = FnBuilder::new("bitwise");
        let x = b.param("x", Type::Bits(4)).unwrap();
        let y = b.param("y", Type::Bits(4)).unwrap();
        let z = b.param("z", Type::Bits(4)).unwrap();
        let enabled = b.param("enabled", Type::Bits(1)).unwrap();
        let values = [
            b.and_all(&[x, y, z]).unwrap(),
            b.nand_all(&[x, y, z]).unwrap(),
            b.or_all(&[x, y, z]).unwrap(),
            b.nor_all(&[x, y, z]).unwrap(),
            b.xor_all(&[x, y, z]).unwrap(),
            b.gate(enabled, x).unwrap(),
        ];
        let result = b.tuple(&values).unwrap();
        let function = b.build(result).unwrap();
        for enabled in [false, true] {
            assert_eq!(
                evaluate(
                    &function,
                    &[bits(4, 3), bits(4, 5), bits(4, 9), IrValue::bool(enabled)],
                ),
                IrValue::make_tuple(&[
                    bits(4, 1),
                    bits(4, 14),
                    bits(4, 15),
                    bits(4, 0),
                    bits(4, 15),
                    bits(4, if enabled { 3 } else { 0 }),
                ])
            );
        }
    }

    #[test]
    fn partial_products_sum_to_signed_and_unsigned_products() {
        let mut b = FnBuilder::new("partial_products");
        let x = b.param("x", Type::Bits(4)).unwrap();
        let y = b.param("y", Type::Bits(5)).unwrap();
        let unsigned = b.umulp(x, y, 7).unwrap();
        let signed = b.smulp(x, y, 7).unwrap();
        let mut sums = Vec::new();
        for partial in [unsigned, signed] {
            let lhs = b.tuple_index(partial, 0).unwrap();
            let rhs = b.tuple_index(partial, 1).unwrap();
            sums.push(b.add(lhs, rhs).unwrap());
        }
        let result = b.tuple(&sums).unwrap();
        let function = b.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(4, 14), bits(5, 3)]),
            IrValue::make_tuple(&[bits(7, 42), bits(7, 122)])
        );
    }

    #[test]
    fn token_contracts_trace_formats_and_coverage() {
        let mut b = FnBuilder::new("contracts");
        let x = b.param("x", Type::Bits(4)).unwrap();
        let enabled = b.param("enabled", Type::Bits(1)).unwrap();
        let token = b.after_all(&[]).unwrap();
        let checked = b
            .assert(token, enabled, "must be enabled\n", "condition")
            .unwrap();
        let aggregate = b.tuple(&[x, enabled]).unwrap();
        let traced = b
            .trace(checked, enabled, "{{x}}={:x}; {}", &[x, aggregate], 3)
            .unwrap();
        let joined = b.after_all(&[token, traced]).unwrap();
        let covered = b.cover(enabled, "enabled_case").unwrap();
        let result = b.tuple(&[joined, covered]).unwrap();
        let function = b.build(result).unwrap();
        let text = function.to_string();
        let parsed = Parser::new(&text).parse_fn().unwrap();
        assert_eq!(parsed.to_string(), text);
        let FnEvalResult::Success(result) = eval_fn(&function, &[bits(4, 10), IrValue::bool(true)])
        else {
            panic!("enabled function should succeed");
        };
        assert_eq!(result.trace_messages.len(), 1);
        // XLS IR trace fragments preserve doubled braces; Verilog rendering
        // performs the separate unescaping step.
        assert_eq!(result.trace_messages[0].message, "{{x}}=a; (10, 1)");
        assert_eq!(result.trace_messages[0].verbosity, 3);
        assert_eq!(result.cover_counts.len(), 1);
        assert_eq!(result.cover_counts[0].label, "enabled_case");
        assert_eq!(result.cover_counts[0].count, 1);
        assert!(matches!(
            eval_fn(&function, &[bits(4, 10), IrValue::bool(false)]),
            FnEvalResult::Failure(_)
        ));
    }

    #[test]
    fn every_extension_builds_and_preserves_attributes() {
        let mut b = FnBuilder::new("extensions");
        let x = b.param("x", Type::Bits(4)).unwrap();
        let y = b.param("y", Type::Bits(4)).unwrap();
        let carry_in = b.param("carry_in", Type::Bits(1)).unwrap();
        let values = [
            b.ext_carry_out(x, y, carry_in).unwrap(),
            b.ext_prio_encode(x, true).unwrap(),
            b.ext_prio_encode(x, false).unwrap(),
            b.ext_clz(x, 5, 3).unwrap(),
            b.ext_normalize_left(
                x,
                NormalizeLeftOptions {
                    normalized_bit_count: 8,
                    shift_offset: 1,
                    clz_bit_count: Some(3),
                },
            )
            .unwrap(),
            b.ext_mask_low(y, 5).unwrap(),
            b.ext_nary_add(
                &[NaryAddTerm::signed(y), NaryAddTerm::unsigned(x).negate()],
                NaryAddOptions {
                    bit_count: 6,
                    architecture: Some(ExtNaryAddArchitecture::BrentKung),
                },
            )
            .unwrap(),
            b.ext_nary_add(&[], NaryAddOptions::new(7)).unwrap(),
        ];
        let result = b.tuple(&values).unwrap();
        let function = b.build(result).unwrap();
        assert_eq!(
            function
                .nodes
                .iter()
                .filter(|node| node.payload.is_extension_op())
                .count(),
            8
        );
        assert_eq!(
            evaluate(&function, &[bits(4, 3), bits(4, 14), IrValue::bool(true)]),
            IrValue::make_tuple(&[
                IrValue::bool(true),
                bits(3, 0),
                bits(3, 1),
                bits(3, 7),
                IrValue::make_tuple(&[bits(8, 24), bits(3, 2)]),
                bits(5, 31),
                bits(6, 59),
                bits(7, 0),
            ])
        );
    }

    #[test]
    fn extension_zero_width_results_and_modular_clz_offset() {
        let mut b = FnBuilder::new("zero_width_extensions");
        let zero = b.param("zero", Type::Bits(0)).unwrap();
        let carry = b.param("carry", Type::Bits(1)).unwrap();
        let values = [
            b.ext_carry_out(zero, zero, carry).unwrap(),
            b.ext_prio_encode(zero, true).unwrap(),
            b.ext_clz(zero, usize::MAX, 7).unwrap(),
            b.ext_normalize_left(zero, NormalizeLeftOptions::new(0))
                .unwrap(),
            b.ext_mask_low(carry, 0).unwrap(),
            b.ext_nary_add(&[NaryAddTerm::signed(carry)], NaryAddOptions::new(0))
                .unwrap(),
        ];
        let result = b.tuple(&values).unwrap();
        let function = b.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(0, 0), IrValue::bool(true)]),
            IrValue::make_tuple(&[
                IrValue::bool(true),
                bits(0, 0),
                bits(7, 127),
                bits(0, 0),
                bits(0, 0),
                bits(0, 0),
            ])
        );
    }

    #[test]
    fn bounds_hints_are_checked_and_roundtrip() {
        let mut b = FnBuilder::new("bounds_hints");
        let array = b.param("array", Type::new_array(Type::Bits(4), 2)).unwrap();
        let index = b.param("index", Type::Bits(1)).unwrap();
        let element = b.array_index_assuming_in_bounds(array, &[index]).unwrap();
        let updated = b
            .array_update_assuming_in_bounds(array, element, &[index])
            .unwrap();
        let function = b.build(updated).unwrap();
        assert!(matches!(
            function.get_node(element.node).payload,
            NodePayload::ArrayIndex {
                assumed_in_bounds: true,
                ..
            }
        ));
        assert!(matches!(
            function.get_node(updated.node).payload,
            NodePayload::ArrayUpdate {
                assumed_in_bounds: true,
                ..
            }
        ));
        let array = IrValue::make_array(&[bits(4, 3), bits(4, 7)]).unwrap();
        assert_eq!(
            evaluate(&function, &[array.clone(), IrValue::bool(true)]),
            array
        );
    }

    #[test]
    fn invalid_function_operations_are_transactional() {
        let mut b = FnBuilder::new("invalid_operations");
        let x = b.param("x", Type::Bits(4)).unwrap();
        let y = b.param("y", Type::Bits(5)).unwrap();
        let condition = b.param("condition", Type::Bits(1)).unwrap();
        let token = b.after_all(&[]).unwrap();
        let tuple = b.tuple(&[x]).unwrap();
        rejects_without_changes(&mut b, |b| b.and_all(&[]));
        rejects_without_changes(&mut b, |b| b.nand_all(&[x, y]));
        rejects_without_changes(&mut b, |b| b.or_all(&[tuple]));
        rejects_without_changes(&mut b, |b| b.gate(x, y));
        rejects_without_changes(&mut b, |b| b.umulp(tuple, x, 4));
        rejects_without_changes(&mut b, |b| b.smulp(x, y, usize::MAX));
        rejects_without_changes(&mut b, |b| b.after_all(&[x]));
        rejects_without_changes(&mut b, |b| b.assert(x, condition, "bad token", ""));
        rejects_without_changes(&mut b, |b| b.assert(token, x, "bad condition", ""));
        rejects_without_changes(&mut b, |b| b.cover(x, "bad condition"));
        for format in ["{", "}", "{:q}", "{:08x}", "{0}"] {
            rejects_without_changes(&mut b, |b| b.trace(token, condition, format, &[x], 0));
        }
        rejects_without_changes(&mut b, |b| b.trace(token, condition, "{}", &[], 0));
        rejects_without_changes(&mut b, |b| b.trace(token, condition, "{{}}", &[x], 0));
        rejects_without_changes(&mut b, |b| b.trace(x, condition, "{}", &[x], 0));
        rejects_without_changes(&mut b, |b| b.trace(token, x, "{}", &[x], 0));
        rejects_without_changes(&mut b, |b| b.ext_carry_out(x, y, condition));
        rejects_without_changes(&mut b, |b| b.ext_carry_out(x, x, x));
        rejects_without_changes(&mut b, |b| b.ext_prio_encode(tuple, true));
        rejects_without_changes(&mut b, |b| b.ext_clz(tuple, 0, 3));
        rejects_without_changes(&mut b, |b| {
            b.ext_normalize_left(x, NormalizeLeftOptions::new(3))
        });
        rejects_without_changes(&mut b, |b| b.ext_mask_low(tuple, 4));
        rejects_without_changes(&mut b, |b| {
            b.ext_nary_add(&[NaryAddTerm::unsigned(tuple)], NaryAddOptions::new(4))
        });
        rejects_without_changes(&mut b, |b| {
            b.array_index_assuming_in_bounds(x, &[condition])
        });
        rejects_without_changes(&mut b, |b| {
            b.array_update_assuming_in_bounds(x, x, &[condition])
        });
        b.build(x).unwrap();
    }

    #[test]
    fn trace_verbosity_and_cover_labels_are_checked_transactionally() {
        let mut b = FnBuilder::new("contract_attributes");
        let condition = b.param("condition", Type::Bits(1)).unwrap();
        let token = b.after_all(&[]).unwrap();
        for verbosity in [-1, i64::MIN] {
            rejects_without_changes(&mut b, |b| b.trace(token, condition, "", &[], verbosity));
        }
        rejects_without_changes(&mut b, |b| b.cover(condition, ""));

        // Empty trace text and an absent assertion label remain legal; only
        // coverage events require a nonempty label.
        let trace = b.trace(token, condition, "", &[], 0).unwrap();
        let assertion = b.assert(trace, condition, "", "").unwrap();
        let cover = b.cover(condition, " ").unwrap();
        let result = b.tuple(&[assertion, cover]).unwrap();
        let function = b.build(result).unwrap();
        assert_eq!(
            evaluate(&function, &[bits(1, 1)]),
            IrValue::make_tuple(&[IrValue::make_token(), IrValue::make_tuple(&[])])
        );
    }

    #[test]
    fn extensions_reject_internal_width_overflow() {
        let mut b = FnBuilder::new("width_overflow");
        let huge = b.param("huge", Type::Bits(usize::MAX)).unwrap();
        let small = b.param("small", Type::Bits(1)).unwrap();
        rejects_without_changes(&mut b, |b| b.ext_carry_out(huge, huge, small));
        rejects_without_changes(&mut b, |b| b.ext_prio_encode(huge, true));
        rejects_without_changes(&mut b, |b| b.ext_clz(huge, 0, 1));
        rejects_without_changes(&mut b, |b| {
            b.ext_normalize_left(
                small,
                NormalizeLeftOptions {
                    normalized_bit_count: 1,
                    shift_offset: usize::MAX,
                    clz_bit_count: None,
                },
            )
        });
        rejects_without_changes(&mut b, |b| {
            b.ext_normalize_left(
                small,
                NormalizeLeftOptions {
                    normalized_bit_count: usize::MAX,
                    shift_offset: 0,
                    clz_bit_count: Some(1),
                },
            )
        });
        b.build(small).unwrap();
    }

    #[test]
    fn every_function_operation_rejects_foreign_handles() {
        let mut foreign_builder = FnBuilder::new("foreign");
        let foreign = foreign_builder.param("foreign", Type::Bits(4)).unwrap();
        let mut b = FnBuilder::new("checked_handles");
        let x = b.param("x", Type::Bits(4)).unwrap();
        let condition = b.param("condition", Type::Bits(1)).unwrap();
        let token = b.after_all(&[]).unwrap();
        let array = b.array(Type::Bits(4), &[x]).unwrap();
        rejects_without_changes(&mut b, |b| b.and_all(&[foreign]));
        rejects_without_changes(&mut b, |b| b.nand_all(&[foreign]));
        rejects_without_changes(&mut b, |b| b.or_all(&[foreign]));
        rejects_without_changes(&mut b, |b| b.nor_all(&[foreign]));
        rejects_without_changes(&mut b, |b| b.xor_all(&[foreign]));
        rejects_without_changes(&mut b, |b| b.gate(condition, foreign));
        rejects_without_changes(&mut b, |b| b.umulp(x, foreign, 4));
        rejects_without_changes(&mut b, |b| b.smulp(foreign, x, 4));
        rejects_without_changes(&mut b, |b| b.after_all(&[foreign]));
        rejects_without_changes(&mut b, |b| b.assert(token, foreign, "message", "label"));
        rejects_without_changes(&mut b, |b| b.trace(token, condition, "{}", &[foreign], 0));
        rejects_without_changes(&mut b, |b| b.cover(foreign, "label"));
        rejects_without_changes(&mut b, |b| b.ext_carry_out(x, foreign, condition));
        rejects_without_changes(&mut b, |b| b.ext_prio_encode(foreign, true));
        rejects_without_changes(&mut b, |b| b.ext_clz(foreign, 0, 3));
        rejects_without_changes(&mut b, |b| {
            b.ext_normalize_left(foreign, NormalizeLeftOptions::new(8))
        });
        rejects_without_changes(&mut b, |b| b.ext_mask_low(foreign, 4));
        rejects_without_changes(&mut b, |b| {
            b.ext_nary_add(&[NaryAddTerm::unsigned(foreign)], NaryAddOptions::new(4))
        });
        rejects_without_changes(&mut b, |b| {
            b.array_index_assuming_in_bounds(array, &[foreign])
        });
        rejects_without_changes(&mut b, |b| {
            b.array_update_assuming_in_bounds(array, foreign, &[condition])
        });
        b.build(x).unwrap();
    }
}
