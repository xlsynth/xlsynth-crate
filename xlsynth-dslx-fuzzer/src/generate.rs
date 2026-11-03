// SPDX-License-Identifier: Apache-2.0

//! Bounded, deterministic generation of well-typed module environments.

use crate::Sample;
use crate::aggregate_match::AggregateMatch;
use crate::array_fragment::{ArrayCast, CastDirection, MatrixUpdate};
use crate::ast::{
    BinaryOp, BitsType, Body, Comparison, Fragment, HelperModule, SliceKind, UnaryOp, ValueRef,
    Width,
};
use crate::compound::{ArrayOp, BuiltinOp, CompoundOp, LoopKind, PatternOp};
use crate::constexpr::{ComputedWidth, ConstExpr, ConstInput, ConstantScope, ParametricDefault};
use crate::map_fragment::{MapBody, MapFragment};
use crate::range_match::{ConstMatch, RangeMatch, RangeValue};
use crate::select_fold::{ArraySelect, CaseArray, NestedFold, SelectKind};
use crate::sum_pattern::{SumComparison, SumPattern, SumShape, SumVariant};
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

/// Selects the DSLX constructs available in every generated body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grammar {
    /// Uses named functions and builtins for maps and ordinary match
    /// expressions.
    Mission,
    /// Uses the same mission productions with small module-local functions
    /// in place of standard library imports.
    MissionLocal,
    /// Composes deeper runtime lanes, aggregate selections and nested folds.
    MissionV4,
    /// Uses v4 productions with module-local helpers.
    MissionLocalV4,
    /// Replays the signed-range and observation decoder with stdlib imports.
    MissionV2,
    /// Replays the signed-range and observation decoder with local helpers.
    MissionLocalV2,
    /// Replays inputs from the mission decoder before signed ranges and extra
    /// observable values were introduced.
    MissionV1,
    /// Also permits lambda map bodies and compile-time match expressions.
    Experimental,
}

impl Grammar {
    /// Selects the decoder with signed ranges and per-lane observations.
    pub fn uses_mission_extensions(self) -> bool {
        matches!(
            self,
            Self::Mission
                | Self::MissionLocal
                | Self::MissionV2
                | Self::MissionLocalV2
                | Self::MissionV4
                | Self::MissionLocalV4
        )
    }

    fn local_helpers(self) -> bool {
        matches!(
            self,
            Self::MissionLocal | Self::MissionLocalV2 | Self::MissionLocalV4
        )
    }

    fn is_v4(self) -> bool {
        matches!(self, Self::MissionV4 | Self::MissionLocalV4)
    }
}

impl std::str::FromStr for Grammar {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "mission" | "mission-v3" => Ok(Self::Mission),
            "mission-local" | "mission-local-v3" => Ok(Self::MissionLocal),
            "mission-v4" => Ok(Self::MissionV4),
            "mission-local-v4" => Ok(Self::MissionLocalV4),
            "mission-v2" => Ok(Self::MissionV2),
            "mission-local-v2" => Ok(Self::MissionLocalV2),
            "mission-v1" => Ok(Self::MissionV1),
            "experimental" => Ok(Self::Experimental),
            _ => Err(format!(
                "unknown DSLX grammar {name:?}; expected mission[-v3], mission-local[-v3], mission-v4, mission-local-v4, mission-v2, mission-local-v2, mission-v1, or experimental"
            )),
        }
    }
}

/// Consumes mutation bytes directly; exhausted inputs choose the smallest form.
struct Entropy<'a> {
    data: &'a [u8],
    offset: usize,
    grammar: Grammar,
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

    /// Draws a bounded array and the scalar function to apply to each element.
    fn map_fragment(&mut self) -> Result<MapFragment, String> {
        const OPS: [BinaryOp; 5] = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Xor,
            BinaryOp::And,
            BinaryOp::Or,
        ];
        let choices = if self.grammar == Grammar::Experimental {
            4
        } else {
            2
        };
        let body = match self.choose(choices) {
            0 => MapBody::BUILTINS[self.choose(MapBody::BUILTINS.len())],
            1 if self.grammar.local_helpers() => MapBody::LocalIsEven,
            1 => MapBody::IsEven,
            2 => MapBody::Capture(OPS[self.choose(OPS.len())]),
            _ => MapBody::TwoCaptures(OPS[self.choose(OPS.len())], OPS[self.choose(OPS.len())]),
        };
        MapFragment::new(body, 1 + self.choose(MapFragment::MAX_LENGTH))
    }

    /// Chooses distinct runtime variants and independent payload widths.
    fn sum_variants_and_widths(&mut self) -> ([SumVariant; 2], [usize; 2]) {
        let count = SumVariant::ALL.len();
        let first = self.choose(count);
        let second = (first + 1 + self.choose(count - 1)) % count;
        (
            [SumVariant::ALL[first], SumVariant::ALL[second]],
            [1 + self.choose(129), 1 + self.choose(129)],
        )
    }

    /// Draws two overlapping match arms with a nonempty tail unique to the
    /// second arm.
    fn range_match(&mut self) -> Result<RangeMatch, String> {
        let signed = self.choose(2) != 0;
        let widths = [5, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129];
        let width = widths[self.choose(widths.len())];
        let start = if signed {
            self.choose(16) as i16 - 8
        } else {
            self.choose(8) as i16
        };
        let length = 1 + self.choose(4) as u8;
        let overlap = 1 + self.choose(usize::from(length)) as u8;
        let tail = 1 + self.choose(4) as u8;
        RangeMatch::new([
            RangeValue::typed(signed, width, start, length, self.choose(2) != 0)?,
            RangeValue::typed(
                signed,
                width,
                start + i16::from(length - overlap),
                overlap + tail,
                self.choose(2) != 0,
            )?,
        ])
    }

    /// Draws representable endpoints on either side of zero while keeping
    /// ranges small enough to materialize or unroll in every context.
    fn bounded_range(
        &mut self,
        max_length: usize,
        allow_empty: bool,
    ) -> Result<RangeValue, String> {
        let signed = self.choose(2) != 0;
        let width = if self.choose(2) == 0 {
            let boundaries = [1, 2, 3, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129];
            boundaries[self.choose(boundaries.len())]
        } else {
            1 + self.choose(129)
        };
        let inclusive = self.choose(2) != 0;
        // Intersect the endpoint sampling window [-8, 8] with the induction
        // type's domain before choosing a length and start.
        let magnitude = 1i16 << (width - usize::from(signed)).min(4);
        let lower = if signed { -magnitude.min(8) } else { 0 };
        let upper = (magnitude - 1).min(8);
        let available = (upper - lower) as usize + usize::from(inclusive);
        let minimum = usize::from(inclusive || !allow_empty);
        let length = minimum + self.choose(max_length.min(available) - minimum + 1);
        let last_start = upper - length as i16 + i16::from(inclusive);
        let start = lower + self.choose((last_start - lower + 1) as usize) as i16;
        RangeValue::typed(signed, width, start, length as u8, inclusive)
    }

    /// Selects a fragment while keeping the historical byte decoders
    /// replayable.
    fn fragment_choice(&mut self, available_modules: usize) -> usize {
        const CONST_MATCH: usize = 23;
        let call = usize::from(available_modules != 0);
        let draw = usize::from(self.byte());
        if self.grammar.is_v4() && (200..224).contains(&draw) {
            return 30 + (draw - 200) % 3;
        }
        if (matches!(self.grammar, Grammar::Mission | Grammar::MissionLocal)
            || self.grammar.is_v4())
            && draw >= 224
        {
            // Reserve the high eighth for the four new productions. Lower
            // selector bytes retain their meanings in the inherited corpus.
            return 26 + draw % 4;
        }
        if self.grammar == Grammar::Experimental {
            return draw % (25 + call);
        }
        let slot = draw % (24 + call);
        if slot >= CONST_MATCH { slot + 1 } else { slot }
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
        // Only top lanes admit constants. Keep helper and constexpr operand
        // draws independent while chaining ordinary results in v4 top lanes.
        let chain = constants && self.grammar.is_v4();
        let mut previous_runtime = None;
        for _ in 0..count {
            let choice = self.fragment_choice(available_modules);
            let draw = usize::from(self.byte());
            let lhs = if chain && let Some(previous) = previous_runtime {
                if draw % 4 == 0 {
                    // The low two bits selected a free draw. Use the remaining
                    // bits so every operand is reachable even at lengths 4 or
                    // 8.
                    values[(draw / 4) % values.len()]
                } else {
                    previous
                }
            } else {
                values[draw % values.len()]
            };
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
                13..=17 | 21..=24 | 26..=32 => {
                    let op = match choice {
                        13 => {
                            CompoundOp::Builtin(BuiltinOp::ALL[self.choose(BuiltinOp::ALL.len())])
                        }
                        14 => {
                            let op = ArrayOp::ALL[self.choose(ArrayOp::ALL.len())];
                            CompoundOp::Array(
                                if self.grammar.local_helpers() && op == ArrayOp::Enumerate {
                                    ArrayOp::LocalEnumerate
                                } else {
                                    op
                                },
                            )
                        }
                        15 => {
                            CompoundOp::Pattern(PatternOp::ALL[self.choose(PatternOp::ALL.len())])
                        }
                        16 => CompoundOp::Record {
                            update: self.choose(2) != 0,
                        },
                        17 => {
                            let kind = [LoopKind::Range, LoopKind::Array, LoopKind::ConstRange]
                                [self.choose(3)];
                            if kind != LoopKind::Array && self.grammar.uses_mission_extensions() {
                                CompoundOp::RangeLoop {
                                    range: self.bounded_range(4, true)?,
                                    constant: kind == LoopKind::ConstRange,
                                }
                            } else {
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
                        }
                        21 => {
                            let (variants, widths) = self.sum_variants_and_widths();
                            CompoundOp::Sum(SumPattern {
                                variants,
                                widths,
                                tail_when_marked: self.choose(2) != 0,
                            })
                        }
                        22 => CompoundOp::Map(self.map_fragment()?),
                        23 => CompoundOp::ConstMatch(ConstMatch::new(self.choose(4) as u8)?),
                        26 => {
                            let (variants, widths) = self.sum_variants_and_widths();
                            CompoundOp::SumComparison(SumComparison {
                                variants,
                                widths,
                                shape: SumShape::ALL[self.choose(SumShape::ALL.len())],
                                unequal: self.choose(2) != 0,
                            })
                        }
                        27 => {
                            let direction = if self.choose(2) == 0 {
                                CastDirection::BitsToArray
                            } else {
                                CastDirection::ArrayToBits
                            };
                            CompoundOp::ArrayCast(ArrayCast::new(direction, 2 + self.choose(2))?)
                        }
                        28 => CompoundOp::RangeMatch(self.range_match()?),
                        29 => {
                            let rows = 2 + self.choose(3);
                            let cols = 2 + self.choose(3);
                            CompoundOp::MatrixUpdate(MatrixUpdate::new(
                                rows,
                                cols,
                                self.choose(rows),
                                self.choose(cols),
                            )?)
                        }
                        30 => CompoundOp::ArraySelect(ArraySelect::new(
                            SelectKind::ALL[self.choose(2)],
                            CaseArray::ALL[self.choose(2)],
                            3 + self.choose(3),
                        )?),
                        31 => CompoundOp::AggregateMatch(AggregateMatch::new(
                            [1 + self.choose(129), 1 + self.choose(129)],
                            self.choose(2) != 0,
                        )?),
                        32 => CompoundOp::NestedFold(NestedFold::new(
                            2 + self.choose(3),
                            1 + self.choose(3),
                            NestedFold::COMBINERS[self.choose(NestedFold::COMBINERS.len())],
                            self.choose(3),
                        )?),
                        _ if self.grammar.uses_mission_extensions() => {
                            CompoundOp::RangeValue(self.bounded_range(8, false)?)
                        }
                        _ => {
                            let length = 1 + self.choose(8) as u8;
                            let start = self.choose(256 - usize::from(length)) as u8;
                            CompoundOp::RangeValue(RangeValue::new(start, length)?)
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
            let ordinary = body.push(fragment)?;
            values.push(ordinary);
            previous_runtime = Some(ordinary);
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
            let result = if chain {
                previous_runtime
            } else {
                values.last().copied()
            }
            .ok_or("body has no values")?;
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

/// Generates one mission-grammar sample, including for empty or exhausted input
/// bytes.
///
/// At most three helper modules, three output lanes and six random fragments
/// per body bound inlined reference size. Widths range from 1 through 129, with
/// extra coverage at machine-word boundaries. No DSLX value uses host integers.
pub fn generate(data: &[u8]) -> Result<Sample, String> {
    generate_with_options(data, GenerateOptions::default())
}

/// Generates mission-grammar samples with a configurable width bound.
pub fn generate_with_options(data: &[u8], options: GenerateOptions) -> Result<Sample, String> {
    generate_with_grammar(data, options, Grammar::Mission)
}

/// Decodes mutation bytes deterministically with the given grammar and width
/// bound.
pub fn generate_with_grammar(
    data: &[u8],
    options: GenerateOptions,
    grammar: Grammar,
) -> Result<Sample, String> {
    if !(1..=129).contains(&options.max_width) {
        return Err("maximum width must be in 1..=129".to_string());
    }
    let mut entropy = Entropy {
        data,
        offset: 0,
        grammar,
    };
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
    // Reserve independent mutation bytes for observing ordinary runtime
    // fragments. Earlier grammars retain their original input layout and
    // top signatures.
    let observation_draws: Vec<_> = if grammar.uses_mission_extensions() {
        (0..lane_count).map(|_| entropy.byte()).collect()
    } else {
        vec![]
    };
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
    let mut sample = Sample::new(helpers, lanes)?;
    sample.observations = sample
        .lanes
        .iter()
        .zip(observation_draws)
        .map(|(lane, draw)| lane.runtime_intermediate(draw, grammar.is_v4()))
        .collect::<Result<_, _>>()?;
    Ok(sample)
}
