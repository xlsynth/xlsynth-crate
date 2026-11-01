// SPDX-License-Identifier: Apache-2.0

//! Bounded, deterministic generation of well-typed module environments.

use crate::Sample;
use crate::ast::{
    BinaryOp, BitsType, Body, Comparison, Fragment, HelperModule, SliceKind, UnaryOp, ValueRef,
    Width,
};
use crate::compound::{ArrayOp, BuiltinOp, CompoundOp, LoopKind, PatternOp};
use crate::constexpr::{ComputedWidth, ConstExpr, ConstInput, ConstantScope, ParametricDefault};
use crate::tuple_pattern::{BindingContext, TupleElement, TuplePattern};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_value_utils::ir_bits_to_usize;

/// Bounds for generated entry widths; runtime values remain IrBits.
#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
    pub max_width: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self { max_width: 129 }
    }
}

/// Consumes mutation bytes directly; exhausted inputs choose the smallest form.
struct Entropy<'a> {
    data: &'a [u8],
    offset: usize,
}

impl Entropy<'_> {
    fn byte(&mut self) -> u8 {
        let byte = self.data.get(self.offset).copied().unwrap_or(0);
        self.offset += 1;
        byte
    }

    fn choose(&mut self, choices: usize) -> usize {
        usize::from(self.byte()) % choices
    }

    /// Draws all literal bits, including those beyond a machine word.
    fn bits(&mut self, width: usize) -> IrBits {
        let bytes: Vec<u8> = (0..width.div_ceil(8)).map(|_| self.byte()).collect();
        IrBits::from_lsb_fn(width, |index| bytes[index / 8] & (1 << (index % 8)) != 0)
    }

    /// Draws tuple values and compatible binding trees within a 40-node budget.
    fn tuple_pattern(&mut self, depth: usize) -> Result<TuplePattern, String> {
        let count = 1 + self.choose(3);
        let mut elements = Vec::new();
        for _ in 0..count {
            if depth < 2 && self.choose(4) == 1 {
                elements.push(TupleElement::Tuple(self.tuple_pattern(depth + 1)?));
            } else {
                let input = self.choose(3);
                elements.push(if self.choose(4) == 1 {
                    TupleElement::Wildcard(input)
                } else {
                    TupleElement::Bind(input)
                });
            }
        }
        let rest = if self.choose(2) == 1 {
            let start = self.choose(count + 1);
            Some(start..start + self.choose(count - start + 1))
        } else {
            None
        };
        TuplePattern::new(elements, rest)
    }

    /// Composes only already-typed values, so no rejection sampling is needed.
    fn body(
        &mut self,
        ty: BitsType,
        available_modules: usize,
        max_fragments: usize,
        constants: bool,
    ) -> Result<Body, String> {
        let mut body = Body::new(ty)?;
        let mut values = vec![ValueRef::X, ValueRef::Y];
        if let Width::Concrete(width) = ty.width {
            values.push(body.push(Fragment::Literal(self.bits(width)))?);
        }
        let count = 1 + self.choose(max_fragments);
        for _ in 0..count {
            let choice = self.choose(if available_modules == 0 { 21 } else { 22 });
            let lhs = values[self.choose(values.len())];
            let rhs = values[self.choose(values.len())];
            let fragment = match choice {
                0..=6 => {
                    let op = [
                        BinaryOp::Add,
                        BinaryOp::Sub,
                        BinaryOp::And,
                        BinaryOp::Or,
                        BinaryOp::Xor,
                        BinaryOp::ShiftLeft,
                        BinaryOp::ShiftRight,
                    ][choice];
                    Fragment::Binary { op, lhs, rhs }
                }
                7 => Fragment::Unary {
                    op: UnaryOp::Not,
                    arg: lhs,
                },
                8 => Fragment::Unary {
                    op: UnaryOp::Neg,
                    arg: lhs,
                },
                9 => Fragment::IfCompare {
                    op: [
                        Comparison::Eq,
                        Comparison::Ne,
                        Comparison::Lt,
                        Comparison::Le,
                        Comparison::Gt,
                        Comparison::Ge,
                    ][self.choose(6)],
                    lhs,
                    rhs,
                    on_true: values[self.choose(values.len())],
                    on_false: values[self.choose(values.len())],
                },
                10 => Fragment::TupleIndex {
                    elements: [lhs, rhs],
                    index: self.choose(2),
                },
                11 => Fragment::ConcatSlice {
                    lhs,
                    rhs,
                    kind: if self.choose(2) == 0 {
                        SliceKind::Fixed
                    } else {
                        SliceKind::Width
                    },
                },
                12 => Fragment::WidenShift {
                    arg: lhs,
                    extra: 1 + self.choose(65),
                    shift: self.choose(194),
                },
                13..=17 => {
                    let op = match choice {
                        13 => {
                            CompoundOp::Builtin(BuiltinOp::ALL[self.choose(BuiltinOp::ALL.len())])
                        }
                        14 => CompoundOp::Array(ArrayOp::ALL[self.choose(ArrayOp::ALL.len())]),
                        15 => {
                            CompoundOp::Pattern(PatternOp::ALL[self.choose(PatternOp::ALL.len())])
                        }
                        16 => CompoundOp::Record {
                            update: self.choose(2) != 0,
                        },
                        _ => {
                            let kind = [LoopKind::Range, LoopKind::Array, LoopKind::ConstRange]
                                [self.choose(3)];
                            let trips = if kind == LoopKind::Array {
                                1 + self.choose(4)
                            } else {
                                self.choose(5)
                            };
                            CompoundOp::Loop {
                                kind,
                                start: self.choose(5),
                                trips,
                            }
                        }
                    };
                    Fragment::Compound {
                        op,
                        args: [lhs, rhs, values[self.choose(values.len())]],
                    }
                }
                18 => Fragment::Binary {
                    op: [BinaryOp::Mul, BinaryOp::Div, BinaryOp::Mod][self.choose(3)],
                    lhs,
                    rhs,
                },
                19 => Fragment::StringLookup {
                    bytes: [self.bits(8), self.bits(8), self.bits(8)],
                    index: lhs,
                },
                20 => Fragment::TuplePattern {
                    pattern: self.tuple_pattern(0)?,
                    context: BindingContext::ALL[self.choose(BindingContext::ALL.len())],
                    args: [lhs, rhs, values[self.choose(values.len())]],
                },
                _ => Fragment::Call {
                    module: self.choose(available_modules),
                    args: [lhs, rhs],
                    explicit_parametric: self.choose(2) != 0,
                },
            };
            values.push(body.push(fragment)?);
            if constants && matches!(ty.width, Width::Concrete(_)) && self.choose(4) == 1 {
                let expression = self.constant_expression(ty, available_modules, false)?;
                let scope = if self.choose(2) == 0 {
                    ConstantScope::Module
                } else {
                    ConstantScope::Local
                };
                values.push(body.push(Fragment::Constant {
                    expression: Box::new(expression),
                    scope,
                    assert_value: self.choose(2) != 0,
                })?);
            }
        }
        // Every concrete lane crosses a module boundary; random helper calls
        // above additionally exercise nested specialization through the DAG.
        if matches!(ty.width, Width::Concrete(_)) && available_modules != 0 {
            let result = *values.last().ok_or("body has no values")?;
            body.push(Fragment::Call {
                module: self.choose(available_modules),
                args: [result, values[self.choose(values.len())]],
                explicit_parametric: self.choose(2) != 0,
            })?;
        }
        Ok(body)
    }

    /// Draws a bounded constexpr body without recursively adding constants.
    fn constant_expression(
        &mut self,
        ty: BitsType,
        available_modules: usize,
        use_width: bool,
    ) -> Result<ConstExpr, String> {
        let body = self.body(ty, available_modules, 3, false)?;
        let Width::Concrete(width) = ty.width else {
            return Err("constant has no concrete width".to_string());
        };
        let x = if use_width {
            ConstInput::BitWidth
        } else {
            ConstInput::Literal(self.bits(width))
        };
        ConstExpr::new(body, [x, ConstInput::Literal(self.bits(width))])
    }

    /// Keeps a random dimension expression in the positive 1..=128 range.
    fn computed_width(
        &mut self,
        available_modules: usize,
        max_width: usize,
    ) -> Result<ConstExpr, String> {
        let ty = BitsType {
            signed: false,
            width: Width::Concrete(32),
        };
        let mut body = self.body(ty, available_modules, 3, false)?;
        let result = body.result();
        let dimension_bits = (usize::BITS - 1 - max_width.leading_zeros()) as usize;
        let mask = body.push(Fragment::Literal(IrBits::from_lsb_fn(32, |index| {
            index < dimension_bits
        })))?;
        let masked = body.push(Fragment::Binary {
            op: BinaryOp::And,
            lhs: result,
            rhs: mask,
        })?;
        let one = body.push(Fragment::Literal(IrBits::from_lsb_fn(32, |index| {
            index == 0
        })))?;
        body.push(Fragment::Binary {
            op: BinaryOp::Add,
            lhs: masked,
            rhs: one,
        })?;
        ConstExpr::new(
            body,
            [
                ConstInput::Literal(self.bits(32)),
                ConstInput::Literal(self.bits(32)),
            ],
        )
    }
}

/// Generates one valid sample, including for empty or exhausted input bytes.
///
/// At most three helper modules, three output lanes and six random fragments
/// per body bound inlined reference size. Widths range from 1 through 129, with
/// extra coverage at machine-word boundaries. No DSLX value uses host integers.
pub fn generate(data: &[u8]) -> Result<Sample, String> {
    generate_with_options(data, GenerateOptions::default())
}

/// Generates the same fragment families with a configurable width bound.
pub fn generate_with_options(data: &[u8], options: GenerateOptions) -> Result<Sample, String> {
    if !(1..=129).contains(&options.max_width) {
        return Err("maximum width must be in 1..=129".to_string());
    }
    let mut entropy = Entropy { data, offset: 0 };
    let module_count = 1 + entropy.choose(3);
    let lane_count = 1 + entropy.choose(3);
    let mut helpers = Vec::new();
    for index in 0..module_count {
        let unsigned = entropy.body(
            BitsType {
                signed: false,
                width: Width::Parametric,
            },
            index,
            6,
            false,
        )?;
        let signed = entropy.body(
            BitsType {
                signed: true,
                width: Width::Parametric,
            },
            index,
            6,
            false,
        )?;
        let mut helper = HelperModule::new(unsigned, signed)?;
        if entropy.choose(3) == 1 {
            let expression = entropy.constant_expression(
                BitsType {
                    signed: false,
                    width: Width::Concrete(32),
                },
                index,
                true,
            )?;
            helper = helper.with_default(ParametricDefault {
                expression,
                combine: [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Xor][entropy.choose(3)],
            })?;
        }
        helpers.push(helper);
    }
    let mut lanes = Vec::new();
    for _ in 0..lane_count {
        let signed = entropy.choose(2) != 0;
        let computation = if entropy.choose(3) == 1 {
            Some(entropy.computed_width(module_count, options.max_width)?)
        } else {
            None
        };
        let width = if let Some(expression) = &computation {
            ir_bits_to_usize(&expression.evaluate(&helpers, None)?)
                .ok_or("computed width does not fit metadata")?
        } else if entropy.choose(2) == 0 {
            let boundaries: Vec<_> = [1, 2, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129]
                .into_iter()
                .filter(|width| *width <= options.max_width)
                .collect();
            boundaries[entropy.choose(boundaries.len())]
        } else {
            1 + entropy.choose(options.max_width)
        };
        let mut body = entropy.body(
            BitsType {
                signed,
                width: Width::Concrete(width),
            },
            module_count,
            6,
            true,
        )?;
        if let Some(expression) = computation {
            body.set_computed_width(ComputedWidth {
                expression: Box::new(expression),
                assert_value: entropy.choose(2) != 0,
            })?;
        }
        lanes.push(body);
    }
    Sample::new(helpers, lanes)
}
