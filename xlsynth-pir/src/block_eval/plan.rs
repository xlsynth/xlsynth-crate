// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use super::prepare::{self, Prepared};
use super::{
    BlockAssumptionFailure, BlockCoverCount, BlockCycleResult, BlockEvalError, BlockEvalField,
    BlockEvalSite, BlockRuntimeFailure, BlockState, BlockValueKind, RegisterInit,
};
use crate::IrValue;
use crate::block_inline::reject_reachable_opaque_extern_blocks;
use crate::ir::{Fn, NodePayload, Package, Type};
use crate::ir_eval::{AssertionFailure, AssumptionFailure, CoverCount, TraceMessage};
use crate::ir_value_utils::zero_ir_value_for_type;
use crate::ir_verify::verify_package;
use crate::random_inputs::generate_uniform_value_with_rng;

/// Backend-independent cycle function, ordered interface and update rules for a
/// block. States created by separate preparations are distinct even for
/// identical source blocks.
pub struct PreparedBlock {
    pub(super) prepared: Prepared,
    owner: Arc<()>,
    origin_by_id: HashMap<usize, usize>,
}

/// Observations collected during one successful clock evaluation.
#[derive(Debug, Clone, Default)]
pub struct BlockCycleEvents {
    pub trace_messages: Vec<TraceMessage>,
    pub cover_counts: Vec<BlockCoverCount>,
}

/// Source of a register's next value after applying reset and checking write
/// enables.
#[derive(Debug, Clone, Copy)]
pub enum BlockRegisterSource<'a> {
    /// Retain the old register.
    Hold,
    /// Use the indicated field of the evaluated function's result tuple.
    Sample(usize),
    /// Use the register's declared reset constant.
    Reset(&'a IrValue),
}

impl PreparedBlock {
    /// Validate the package and prepare the selected block and its block
    /// instances.
    pub fn prepare(package: &Package, top: &str) -> Result<Self, BlockEvalError> {
        reject_reachable_opaque_extern_blocks(package, Some(top))
            .map_err(BlockEvalError::Preparation)?;
        verify_package(package).map_err(|err| BlockEvalError::InvalidPackage(Box::new(err)))?;
        let prepared = prepare::prepare(package, top).map_err(BlockEvalError::Preparation)?;
        let origin_by_id = prepared
            .function
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                matches!(
                    node.payload,
                    NodePayload::Cover { .. }
                        | NodePayload::ArrayIndex {
                            assumed_in_bounds: true,
                            ..
                        }
                        | NodePayload::ArrayUpdate {
                            assumed_in_bounds: true,
                            ..
                        }
                )
            })
            .map(|(i, node)| (node.text_id, i))
            .collect();
        Ok(Self {
            prepared,
            owner: Arc::new(()),
            origin_by_id,
        })
    }

    /// Data inputs in port order, including reset pins. The clock is implicit.
    pub fn inputs(&self) -> &[BlockEvalField] {
        &self.prepared.inputs
    }

    /// Outputs in port order.
    pub fn outputs(&self) -> &[BlockEvalField] {
        &self.prepared.outputs
    }

    /// Register layout in declaration order, followed by child instances in
    /// preorder.
    pub fn registers(&self) -> &[BlockEvalField] {
        &self.prepared.registers
    }

    /// Create register state independent of each register's synchronous reset
    /// value.
    pub fn initial_state(&self, init: RegisterInit) -> Result<BlockState, BlockEvalError> {
        let values = match init {
            RegisterInit::AllZeros => self
                .registers()
                .iter()
                .map(|r| zero_ir_value_for_type(&r.ty))
                .collect(),
            RegisterInit::Random { seed } => {
                let mut rng = Pcg64Mcg::seed_from_u64(seed);
                self.registers()
                    .iter()
                    .map(|r| generate_uniform_value_with_rng(&mut rng, &r.ty))
                    .collect()
            }
            RegisterInit::Supplied(values) => {
                validate_values(BlockValueKind::Register, self.registers(), &values)?;
                values
            }
        };
        Ok(BlockState {
            owner: self.owner.clone(),
            values,
        })
    }

    /// Projected function with data inputs followed by old registers as
    /// arguments. Its result begins with pre-edge outputs, followed by
    /// sampled register-write operands.
    pub fn function(&self) -> &Fn {
        &self.prepared.function
    }

    /// Reject register state originating from another preparation.
    pub fn validate_state(&self, state: &BlockState) -> Result<(), BlockEvalError> {
        if Arc::ptr_eq(&self.owner, &state.owner) {
            Ok(())
        } else {
            Err(BlockEvalError::ForeignState)
        }
    }

    /// Check input count, widths and aggregate shapes in port order.
    pub fn validate_inputs(&self, inputs: &[IrValue]) -> Result<(), BlockEvalError> {
        validate_values(BlockValueKind::Input, self.inputs(), inputs)
    }

    /// Wired reset constants in register order, suitable for preparing native
    /// reset images.
    pub fn reset_values(&self) -> impl Iterator<Item = Option<&IrValue>> {
        self.prepared
            .updates
            .iter()
            .map(|update| update.reset.as_ref().map(|(_, _, value)| value))
    }

    /// Map cycle events to source sites and report any runtime failures before
    /// writes.
    pub fn check_events(
        &self,
        assertion_failures: Vec<AssertionFailure>,
        assumption_failures: Vec<AssumptionFailure>,
        trace_messages: Vec<TraceMessage>,
        cover_counts: Vec<CoverCount>,
    ) -> Result<BlockCycleEvents, BlockEvalError> {
        let cover_counts = self.cover_counts(cover_counts);
        if assertion_failures.is_empty() && assumption_failures.is_empty() {
            Ok(BlockCycleEvents {
                trace_messages,
                cover_counts,
            })
        } else {
            Err(BlockEvalError::Runtime(Box::new(BlockRuntimeFailure {
                assertion_failures,
                assumption_failures: assumption_failures
                    .into_iter()
                    .map(|a| BlockAssumptionFailure {
                        site: self.site(a.node_text_id),
                        kind: a.kind,
                    })
                    .collect(),
                trace_messages,
                cover_counts,
            })))
        }
    }

    /// Resolve all writes using reset and enable bits from the projected
    /// result.
    ///
    /// The selection callback may run for early registers before a later
    /// register reports a conflict. Write only into staged state and commit
    /// it on success.
    pub fn resolve_writes<'a>(
        &'a self,
        mut sample_bit: impl FnMut(usize) -> bool,
        mut select: impl FnMut(usize, BlockRegisterSource<'a>),
    ) -> Result<(), BlockEvalError> {
        for (index, update) in self.prepared.updates.iter().enumerate() {
            if let Some((sample, active_low, value)) = &update.reset
                && sample_bit(*sample) != *active_low
            {
                select(index, BlockRegisterSource::Reset(value));
                continue;
            }
            let mut enabled: Option<&prepare::Write> = None;
            for write in &update.writes {
                if write.enable.is_none_or(&mut sample_bit) {
                    if let Some(first) = enabled {
                        return Err(BlockEvalError::ConflictingWrites {
                            register: self.registers()[index].name.clone(),
                            first: first.node_id,
                            second: write.node_id,
                        });
                    }
                    enabled = Some(write);
                }
            }
            let source = enabled.map_or(BlockRegisterSource::Hold, |write| {
                BlockRegisterSource::Sample(write.data)
            });
            select(index, source);
        }
        Ok(())
    }

    /// Resolve a typed result of the projected function against the original
    /// state.
    pub fn finish_cycle(
        &self,
        state: &BlockState,
        samples: IrValue,
        events: BlockCycleEvents,
    ) -> Result<BlockCycleResult, BlockEvalError> {
        self.validate_state(state)?;
        let expected = &self.function().ret_ty;
        let results = match &samples {
            IrValue::Tuple(results) if value_matches_type(&samples, expected) => results,
            _ => {
                return Err(BlockEvalError::CycleResultType {
                    expected: expected.clone(),
                    actual: samples.type_(),
                });
            }
        };
        let mut values = state.values.clone();
        self.resolve_writes(
            |sample| {
                results[sample]
                    .to_bool()
                    .expect("checked one-bit cycle sample")
            },
            |register, source| match source {
                BlockRegisterSource::Hold => { /* The staged value already holds the old state. */ }
                BlockRegisterSource::Reset(value) => values[register].clone_from(value),
                BlockRegisterSource::Sample(sample) => {
                    values[register].clone_from(&results[sample])
                }
            },
        )?;
        Ok(BlockCycleResult {
            outputs: results[..self.outputs().len()].to_vec(),
            next_state: BlockState {
                owner: self.owner.clone(),
                values,
            },
            trace_messages: events.trace_messages,
            cover_counts: events.cover_counts,
        })
    }

    /// Resolve a projected event back to its instance, retaining package IDs of
    /// called functions.
    fn site(&self, id: usize) -> BlockEvalSite {
        match self
            .origin_by_id
            .get(&id)
            .and_then(|i| self.prepared.origins[*i].as_ref())
        {
            Some(origin) => BlockEvalSite {
                instance: origin.instance.to_string(),
                node_text_id: origin.text_id,
            },
            None => BlockEvalSite {
                instance: String::new(),
                node_text_id: id,
            },
        }
    }

    fn cover_counts(&self, counts: Vec<CoverCount>) -> Vec<BlockCoverCount> {
        counts
            .into_iter()
            .map(|c| BlockCoverCount {
                site: self.site(c.node_text_id),
                label: c.label,
                count: c.count,
            })
            .collect()
    }
}

/// Check both arity and exact structural type at the public boundary.
fn validate_values(
    kind: BlockValueKind,
    fields: &[BlockEvalField],
    values: &[IrValue],
) -> Result<(), BlockEvalError> {
    if fields.len() != values.len() {
        return Err(BlockEvalError::ValueCount {
            kind,
            expected: fields.len(),
            actual: values.len(),
        });
    }
    for (field, value) in fields.iter().zip(values) {
        field.validate(kind, value)?;
    }
    Ok(())
}

impl BlockEvalField {
    /// Validate one bound port or register without allocating its type on
    /// success.
    pub fn validate(&self, kind: BlockValueKind, value: &IrValue) -> Result<(), BlockEvalError> {
        if value_matches_type(value, &self.ty) {
            Ok(())
        } else {
            Err(BlockEvalError::ValueType {
                kind,
                name: self.name.clone(),
                expected: self.ty.clone(),
                actual: value.type_(),
            })
        }
    }
}

/// Compare the structural type without allocating a temporary type tree.
fn value_matches_type(value: &IrValue, ty: &Type) -> bool {
    match (value, ty) {
        (IrValue::Bits(bits), Type::Bits(width)) => bits.get_bit_count() == *width,
        (IrValue::Tuple(values), Type::Tuple(fields)) => {
            values.len() == fields.len()
                && values
                    .iter()
                    .zip(fields)
                    .all(|(v, ty)| value_matches_type(v, ty))
        }
        (IrValue::Array(values), Type::Array(array)) => {
            // IrArray enforces this element type at construction, including for
            // empty arrays.
            values.elements().len() == array.element_count
                && values.element_type() == array.element_type.as_ref()
        }
        (IrValue::Token, Type::Token) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    #[test]
    fn finish_cycle_checks_state_and_backend_result_shape_before_selecting_writes() {
        let ir = r#"package test
top block sink(clk: clock, x: bits[1]) {
  reg q(bits[1])
  x: bits[1] = input_port(name=x, id=1)
  old: bits[1] = register_read(register=q, id=3)
  w: () = register_write(x, register=q, id=2)
}"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        let plan = PreparedBlock::prepare(&pkg, "sink").unwrap();
        let state = plan.initial_state(RegisterInit::AllZeros).unwrap();
        let other = PreparedBlock::prepare(&pkg, "sink").unwrap();
        let valid = IrValue::make_tuple(&[IrValue::bool(true)]);
        assert!(matches!(
            other.finish_cycle(&state, valid.clone(), BlockCycleEvents::default()),
            Err(BlockEvalError::ForeignState)
        ));

        let expected = Type::Tuple(vec![Box::new(Type::Bits(1))]);
        for malformed in [
            IrValue::bool(true),
            IrValue::make_tuple(&[]),
            IrValue::make_tuple(&[IrValue::make_ubits(2, 1).unwrap()]),
        ] {
            let malformed_ty = malformed.type_();
            let error = plan
                .finish_cycle(&state, malformed, BlockCycleEvents::default())
                .unwrap_err();
            assert!(
                matches!(error, BlockEvalError::CycleResultType { expected: e, actual } if e == expected && actual == malformed_ty)
            );
        }
        let next = plan
            .finish_cycle(&state, valid, BlockCycleEvents::default())
            .unwrap()
            .next_state;
        assert_eq!(state.values(), &[IrValue::bool(false)]);
        assert_eq!(next.values(), &[IrValue::bool(true)]);
    }

    #[test]
    fn bound_field_checks_element_type_even_in_nested_empty_arrays() {
        let expected = Type::Tuple(vec![Box::new(Type::new_array(Type::Bits(65), 0))]);
        let field = BlockEvalField {
            name: "a".to_owned(),
            ty: expected.clone(),
        };
        let value = IrValue::make_tuple(&[IrValue::make_array_typed(Type::Bits(65), &[]).unwrap()]);
        field.validate(BlockValueKind::Input, &value).unwrap();
        let invalid =
            IrValue::make_tuple(&[IrValue::make_array_typed(Type::Bits(64), &[]).unwrap()]);
        let error = field.validate(BlockValueKind::Input, &invalid).unwrap_err();
        assert!(
            matches!(error, BlockEvalError::ValueType { kind: BlockValueKind::Input, name, expected: e, actual }
            if name == "a" && e == expected && actual == invalid.type_())
        );
    }
}
