// SPDX-License-Identifier: Apache-2.0

//! Shared casez helpers for leading-bit classification and normalization.

use xlsynth_pir::ir::{Fn, Node, NodePayload, NodeRef, Type};
use xlsynth_pir::{IrBits, IrValue};
use xlsynth_vast::{Expr, LogicRef};

use crate::BlockCodegenError;
use crate::block::{BlockEmitter, Value};

/// Identifies a reusable helper by every parameter affecting its behavior.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct PriorityHelper {
    input_width: usize,
    kind: PriorityKind,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PriorityKind {
    Select {
        cases: usize,
        width: usize,
    },
    Encode {
        lsb_prio: bool,
        width: usize,
    },
    OneHot {
        lsb_prio: bool,
    },
    Clz {
        offset: usize,
        width: usize,
    },
    Normalize {
        shift_offset: usize,
        normalized_width: usize,
        clz_width: Option<usize>,
    },
}

impl BlockEmitter<'_, '_> {
    /// Emits each needed helper once, before its callers and pipeline sections.
    pub(crate) fn emit_priority_helpers(&mut self, function: &Fn) -> Result<(), BlockCodegenError> {
        for node in &function.nodes {
            let Some((helper, _)) = PriorityHelper::for_node(function, node) else {
                continue;
            };
            if helper.input_width == 0 || self.priority_helpers.contains_key(&helper) {
                continue;
            }
            let name = self.unique_name(&helper.name());
            let result_type = self.bits_type(helper.result_width());
            let handle = self.file.add_function(self.module, &name, &result_type)?;
            let input_type = self.bits_type(helper.input_width);
            let is_select = matches!(helper.kind, PriorityKind::Select { .. });
            let input = self.file.function_add_logic_input(
                handle,
                if is_select { "sel" } else { "value" },
                &input_type,
            )?;
            let mut case_values = Vec::new();
            let fallback = if let PriorityKind::Select { cases, .. } = helper.kind {
                for index in 0..cases {
                    case_values.push(
                        self.file
                            .function_add_logic_input(
                                handle,
                                &format!("case{index}"),
                                &result_type,
                            )?
                            .to_expr(),
                    );
                }
                Some(
                    self.file
                        .function_add_logic_input(handle, "default_value", &result_type)?
                        .to_expr(),
                )
            } else if matches!(helper.kind, PriorityKind::OneHot { .. }) {
                Some(self.priority_result(helper, None, None)?)
            } else {
                None
            };
            let result = self.file.function_result(handle).to_expr();
            let body = self.file.function_body(handle);
            let case = self.file.block_add_casez(body, &input.to_expr());
            // Every leading/trailing-one pattern fixes all higher-priority bits
            // to zero, so the arms are disjoint for two-state inputs.
            self.file.case_set_unique(case, true)?;
            let lsb_prio = is_select
                || matches!(
                    helper.kind,
                    PriorityKind::Encode { lsb_prio: true, .. }
                        | PriorityKind::OneHot { lsb_prio: true }
                );
            for priority in 0..helper.input_width {
                let index = if lsb_prio {
                    priority
                } else {
                    helper.input_width - 1 - priority
                };
                let pattern = if lsb_prio {
                    format!(
                        "{}1{}",
                        "?".repeat(helper.input_width - priority - 1),
                        "0".repeat(priority)
                    )
                } else {
                    format!(
                        "{}1{}",
                        "0".repeat(priority),
                        "?".repeat(helper.input_width - priority - 1)
                    )
                };
                let pattern = self.file.make_binary_pattern(&pattern)?;
                let arm = self.file.case_add_item(case, &pattern);
                let value = if is_select {
                    case_values[index]
                } else {
                    self.priority_result(helper, Some(input), Some(index))?
                };
                self.file
                    .block_add_blocking_assignment(arm, &result, &value);
            }
            let value = if let Some(fallback) = fallback {
                let zeros = self
                    .file
                    .make_binary_pattern(&"0".repeat(helper.input_width))?;
                let arm = self.file.case_add_item(case, &zeros);
                self.file
                    .block_add_blocking_assignment(arm, &result, &fallback);
                self.file.make_unsized_x_literal()
            } else {
                self.priority_result(helper, Some(input), None)?
            };
            let default = self.file.case_add_default(case);
            self.file
                .block_add_blocking_assignment(default, &result, &value);
            self.priority_helpers.insert(helper, name);
        }
        Ok(())
    }

    /// Calls the predeclared helper, or folds its unique zero-width input.
    pub(crate) fn emit_priority_call(
        &mut self,
        node_ref: NodeRef,
    ) -> Result<Expr, BlockCodegenError> {
        let (helper, arg) = PriorityHelper::for_node(self.func, self.func.get_node(node_ref))
            .expect("priority operation has a representable result");
        if helper.input_width == 0 {
            return self.priority_result(helper, None, None);
        }
        let mut arguments = vec![self.required_value(arg)?.expr];
        if let NodePayload::PrioritySel { cases, default, .. } =
            &self.func.get_node(node_ref).payload
        {
            for operand in cases.iter().chain(default.iter()) {
                arguments.push(self.required_value(*operand)?.expr);
            }
        }
        let name = self
            .priority_helpers
            .get(&helper)
            .expect("priority helper was predeclared");
        Ok(self
            .file
            .make_function_call(name, &arguments.iter().collect::<Vec<_>>()))
    }

    /// Forms one branch result, using static wiring for normalization shifts.
    fn priority_result(
        &mut self,
        helper: PriorityHelper,
        input: Option<LogicRef>,
        index: Option<usize>,
    ) -> Result<Expr, BlockCodegenError> {
        let leading = index.map_or(helper.input_width, |index| helper.input_width - 1 - index);
        match helper.kind {
            PriorityKind::Select { .. } => {
                unreachable!("priority select returns its selected argument")
            }
            PriorityKind::Encode { width, .. } => {
                self.count_literal(width, index.unwrap_or(helper.input_width))
            }
            PriorityKind::OneHot { .. } => {
                let selected = index.unwrap_or(helper.input_width);
                let bits = IrBits::from_lsb_is_0(
                    &(0..helper.result_width())
                        .map(|bit| bit == selected)
                        .collect::<Vec<_>>(),
                );
                Ok(self
                    .literal(
                        &IrValue::from_bits(&bits),
                        &Type::Bits(helper.result_width()),
                    )?
                    .unwrap())
            }
            PriorityKind::Clz { offset, width } => {
                let adjusted = count_bits(width, leading).add(&count_bits(width, offset));
                Ok(self
                    .literal(&IrValue::from_bits(&adjusted), &Type::Bits(width))?
                    .unwrap())
            }
            PriorityKind::Normalize {
                shift_offset,
                normalized_width,
                clz_width,
            } => {
                let mut parts = Vec::new();
                if normalized_width != 0 {
                    let shift = leading.saturating_add(shift_offset);
                    if index.is_none() || shift >= normalized_width {
                        parts.push(self.zero(normalized_width)?);
                    } else {
                        let source_width = helper.input_width.min(normalized_width - shift);
                        let padding = normalized_width - source_width - shift;
                        if padding != 0 {
                            parts.push(self.zero(padding)?);
                        }
                        let input = Value::signal(input.expect("nonzero input has a signal"))
                            .with_width(helper.input_width);
                        parts.push(self.static_slice(input, 0, source_width));
                        if shift != 0 {
                            parts.push(self.zero(shift)?);
                        }
                    }
                }
                if let Some(width) = clz_width.filter(|width| *width != 0) {
                    parts.push(self.count_literal(width, leading)?);
                }
                Ok(self
                    .concat_or_only(&parts)
                    .expect("helper result is nonzero width"))
            }
        }
    }

    /// Truncates or extends a host-sized count to its exact IR result width.
    fn count_literal(&mut self, width: usize, value: usize) -> Result<Expr, BlockCodegenError> {
        let bits = count_bits(width, value);
        Ok(self
            .literal(&IrValue::from_bits(&bits), &Type::Bits(width))?
            .unwrap())
    }
}

/// Represents a count modulo the requested width, without host shift overflow.
fn count_bits(width: usize, value: usize) -> IrBits {
    IrBits::from_lsb_is_0(
        &(0..width)
            .map(|bit| bit < usize::BITS as usize && value & (1usize << bit) != 0)
            .collect::<Vec<_>>(),
    )
}

impl PriorityHelper {
    /// Extracts the specialization parameters and operand of a priority node.
    fn for_node(function: &Fn, node: &Node) -> Option<(Self, NodeRef)> {
        if node.ty.bit_count() == 0 {
            return None;
        }
        let (arg, kind) = match node.payload {
            NodePayload::PrioritySel {
                selector,
                ref cases,
                ..
            } => (
                selector,
                PriorityKind::Select {
                    cases: cases.len(),
                    width: node.ty.bit_count(),
                },
            ),
            NodePayload::ExtPrioEncode { arg, lsb_prio } => (
                arg,
                PriorityKind::Encode {
                    lsb_prio,
                    width: node.ty.bit_count(),
                },
            ),
            NodePayload::OneHot { arg, lsb_prio } => (arg, PriorityKind::OneHot { lsb_prio }),
            NodePayload::ExtClz {
                arg,
                offset,
                new_bit_count,
            } => (
                arg,
                PriorityKind::Clz {
                    offset,
                    width: new_bit_count,
                },
            ),
            NodePayload::ExtNormalizeLeft {
                arg,
                shift_offset,
                normalized_bit_count,
                clz_bit_count,
            } => (
                arg,
                PriorityKind::Normalize {
                    shift_offset,
                    normalized_width: normalized_bit_count,
                    clz_width: clz_bit_count,
                },
            ),
            _ => return None,
        };
        Some((
            Self {
                input_width: function.get_node_ty(arg).bit_count(),
                kind,
            },
            arg,
        ))
    }

    fn result_width(self) -> usize {
        match self.kind {
            PriorityKind::Select { width, .. }
            | PriorityKind::Encode { width, .. }
            | PriorityKind::Clz { width, .. } => width,
            PriorityKind::OneHot { .. } => self.input_width + 1,
            PriorityKind::Normalize {
                normalized_width,
                clz_width,
                ..
            } => normalized_width + clz_width.unwrap_or(0),
        }
    }

    fn name(self) -> String {
        let input = self.input_width;
        match self.kind {
            PriorityKind::Select { cases, width } => format!("priority_sel_{width}b_{cases}way"),
            PriorityKind::Encode { lsb_prio, width } => {
                let direction = if lsb_prio { "lsb" } else { "msb" };
                format!("priority_encode_{direction}_{input}_{width}")
            }
            PriorityKind::OneHot { lsb_prio } => {
                let direction = if lsb_prio { "lsb" } else { "msb" };
                format!("one_hot_{direction}_{input}")
            }
            PriorityKind::Clz { offset, width } => format!("clz_{input}_{width}_offset_{offset}"),
            PriorityKind::Normalize {
                shift_offset,
                normalized_width,
                clz_width,
            } => {
                let count = clz_width.map_or_else(|| "none".to_owned(), |width| width.to_string());
                format!(
                    "normalize_left_{input}_{normalized_width}_offset_{shift_offset}_clz_{count}"
                )
            }
        }
    }
}
