// SPDX-License-Identifier: Apache-2.0

//! Bounded, deterministic generation of well-typed module environments.

use crate::aggregate_flow::{AggregateFlow, FlowKind, FlowObservation, FlowStep};
use crate::aggregate_match::AggregateMatch;
use crate::array_fragment::{ArrayCast, CastDirection, MatrixUpdate, UpdateIndexForm, UpdateShape};
use crate::ast::{
    BinaryOp, BitsType, Body, Comparison, Fragment, HelperModule, SliceKind, UnaryOp, ValueRef,
    Width,
};
use crate::bit_projection::{BitProjection, ProjectionKind};
use crate::compound::{ArrayOp, BuiltinOp, CompoundOp, LoopKind, PatternOp};
use crate::constexpr::{ComputedWidth, ConstExpr, ConstInput, ConstantScope, ParametricDefault};
use crate::imported_const::ExportedConstants;
use crate::imported_struct::StructExport;
use crate::map_fragment::{MapBody, MapFragment};
use crate::mission_expr::{
    CheckedCast, LogicalPredicates, NamedAggregateMatch, NamedPatternShape, ReceiverMap,
    WideningCast,
};
use crate::range_match::{ConstMatch, GuardedMatch, NestedCapture, RangeMatch, RangeValue};
use crate::select_fold::{ArrayFold, ArraySelect, CaseArray, NestedFold, SelectKind};
use crate::sum_pattern::{SumComparison, SumPattern, SumShape, SumVariant};
use crate::tuple_pattern::{BindingContext, TupleElement, TuplePattern};
use crate::{AggregateOutput, Sample};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_value_utils::ir_bits_to_usize;

const COMPARISONS: [Comparison; 6] = [
    Comparison::Eq,
    Comparison::Ne,
    Comparison::Lt,
    Comparison::Le,
    Comparison::Gt,
    Comparison::Ge,
];

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
    /// Adds bound-method maps, qualified constants, logical predicates and
    /// checked casts.
    MissionV5,
    /// Uses v5 productions with module-local helpers.
    MissionLocalV5,
    /// Adds named aggregate match patterns, widening casts and imported struct
    /// aliases.
    MissionV6,
    /// Uses v6 productions with module-local helpers.
    MissionLocalV6,
    /// Transports runtime aggregates through producer and updater functions.
    MissionV7,
    /// Uses v7 productions with module-local helpers.
    MissionLocalV7,
    /// Varies how multidimensional updates receive their tuple-valued indices.
    MissionV8,
    /// Uses v8 productions with module-local helpers.
    MissionLocalV8,
    /// Adds update index shapes, loop captures and selectively deeper aggregate
    /// flows.
    MissionV9,
    /// Uses v9 productions with module-local helpers.
    MissionLocalV9,
    /// Checks match-arm activation and varies slice addresses and aggregate
    /// dependencies.
    MissionV10,
    /// Uses v10 productions with module-local helpers.
    MissionLocalV10,
    /// Selects in-scope loop results and observes typed bit projections at
    /// runtime and during parametric specialization.
    MissionV11,
    /// Uses v11 productions with module-local helpers.
    MissionLocalV11,
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
                | Self::MissionV5
                | Self::MissionLocalV5
                | Self::MissionV6
                | Self::MissionV7
                | Self::MissionLocalV6
                | Self::MissionLocalV7
                | Self::MissionV8
                | Self::MissionLocalV8
                | Self::MissionV9
                | Self::MissionLocalV9
                | Self::MissionV10
                | Self::MissionV11
                | Self::MissionLocalV10
                | Self::MissionLocalV11
        )
    }

    fn local_helpers(self) -> bool {
        matches!(
            self,
            Self::MissionLocal
                | Self::MissionLocalV2
                | Self::MissionLocalV4
                | Self::MissionLocalV5
                | Self::MissionLocalV6
                | Self::MissionLocalV7
                | Self::MissionLocalV8
                | Self::MissionLocalV9
                | Self::MissionLocalV10
                | Self::MissionLocalV11
        )
    }

    fn uses_deep_lanes(self) -> bool {
        matches!(self, Self::MissionV4 | Self::MissionLocalV4) || self.uses_v5_productions()
    }

    fn uses_v5_productions(self) -> bool {
        matches!(self, Self::MissionV5 | Self::MissionLocalV5) || self.uses_v6_productions()
    }

    fn uses_v6_productions(self) -> bool {
        matches!(self, Self::MissionV6 | Self::MissionLocalV6) || self.uses_aggregate_flow()
    }

    fn uses_aggregate_flow(self) -> bool {
        matches!(self, Self::MissionV7 | Self::MissionLocalV7) || self.uses_tuple_coordinates()
    }

    fn uses_tuple_coordinates(self) -> bool {
        matches!(self, Self::MissionV8 | Self::MissionLocalV8) || self.uses_v9_productions()
    }

    fn uses_v9_productions(self) -> bool {
        matches!(self, Self::MissionV9 | Self::MissionLocalV9) || self.uses_v10_productions()
    }

    /// Includes the match-guard proof contract introduced by v10.
    pub fn uses_v10_productions(self) -> bool {
        self.is_v10() || self.uses_v11_productions()
    }

    /// Identifies the original v10 decoder.
    pub fn is_v10(self) -> bool {
        matches!(self, Self::MissionV10 | Self::MissionLocalV10)
    }

    fn uses_v11_productions(self) -> bool {
        matches!(self, Self::MissionV11 | Self::MissionLocalV11)
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
            "mission-v5" => Ok(Self::MissionV5),
            "mission-local-v5" => Ok(Self::MissionLocalV5),
            "mission-v7" => Ok(Self::MissionV7),
            "mission-local-v7" => Ok(Self::MissionLocalV7),
            "mission-v8" => Ok(Self::MissionV8),
            "mission-local-v8" => Ok(Self::MissionLocalV8),
            "mission-v9" => Ok(Self::MissionV9),
            "mission-local-v9" => Ok(Self::MissionLocalV9),
            "mission-v11" => Ok(Self::MissionV11),
            "mission-local-v11" => Ok(Self::MissionLocalV11),
            "mission-v10" => Ok(Self::MissionV10),
            "mission-local-v10" => Ok(Self::MissionLocalV10),
            "mission-v6" => Ok(Self::MissionV6),
            "mission-local-v6" => Ok(Self::MissionLocalV6),
            "mission-v2" => Ok(Self::MissionV2),
            "mission-local-v2" => Ok(Self::MissionLocalV2),
            "mission-v1" => Ok(Self::MissionV1),
            "experimental" => Ok(Self::Experimental),
            _ => Err(format!(
                "unknown DSLX grammar {name:?}; expected mission[-v3], mission-local[-v3], mission-v11, mission-local-v11, mission-v10, mission-local-v10, mission-v9, mission-local-v9, mission-v8, mission-local-v8, mission-v7, mission-local-v7, mission-v6, mission-local-v6, mission-v5, mission-local-v5, mission-v4, mission-local-v4, mission-v2, mission-local-v2, mission-v1, or experimental"
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

    /// Reuses the range decoder's eight draws for both pure and asserted
    /// matches.
    fn range_match(&mut self, allow_assertions: bool) -> Result<CompoundOp, String> {
        let [
            sign,
            width,
            start,
            length,
            overlap,
            tail,
            first_bound,
            second_bound,
        ] = std::array::from_fn(|_| self.byte());
        if allow_assertions && sign & 2 != 0 {
            return Ok(CompoundOp::GuardedMatch(GuardedMatch::from_draws(
                usize::from(length % 4),
                start,
            )?));
        }
        let signed = sign % 2 != 0;
        let widths = [5, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129];
        let width = widths[usize::from(width) % widths.len()];
        let start = if signed {
            i16::from(start % 16) - 8
        } else {
            i16::from(start % 8)
        };
        let length = 1 + length % 4;
        let overlap = 1 + overlap % length;
        let tail = 1 + tail % 4;
        Ok(CompoundOp::RangeMatch(RangeMatch::new([
            RangeValue::typed(signed, width, start, length, first_bound % 2 != 0)?,
            RangeValue::typed(
                signed,
                width,
                start + i16::from(length - overlap),
                overlap + tail,
                second_bound % 2 != 0,
            )?,
        ])?))
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
        if self.grammar.uses_v11_productions() && (112..136).contains(&draw) {
            return 41 + draw % 2;
        }
        if self.grammar.uses_v10_productions() && (136..152).contains(&draw) {
            return 40;
        }
        // In the first helper, selectors for imported types and constants
        // retain their ordinary meaning because no earlier module exists.
        if self.grammar.uses_v6_productions() && (152..176).contains(&draw) {
            let production = 37 + (draw - 152) % 3;
            if production != 39 || call != 0 {
                return production;
            }
        }
        if self.grammar.uses_v5_productions()
            && (176..200).contains(&draw)
            && (draw % 4 != 3 || call != 0)
        {
            return 33 + draw % 4;
        }
        if self.grammar.uses_deep_lanes() && (200..224).contains(&draw) {
            return 30 + (draw - 200) % 3;
        }
        if (matches!(self.grammar, Grammar::Mission | Grammar::MissionLocal)
            || self.grammar.uses_deep_lanes())
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
        // draws independent while chaining ordinary results in deeper top
        // lanes.
        let chain = constants && self.grammar.uses_deep_lanes();
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
            let rhs_slot = self.choose(values.len());
            let rhs = values[rhs_slot];
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
                    op: COMPARISONS[self.choose(COMPARISONS.len())],
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
                40 => {
                    // A runtime parameter supplies the address independently of
                    // the selected data whenever the data itself is a
                    // parameter.
                    let start = if lhs == ValueRef::X {
                        ValueRef::Y
                    } else if lhs == ValueRef::Y || draw % 2 == 0 {
                        ValueRef::X
                    } else {
                        ValueRef::Y
                    };
                    let widths = [32, 64, 65, 96, 129];
                    Fragment::RuntimeWidthSlice {
                        subject: lhs,
                        start,
                        start_width: widths[self.choose(widths.len())],
                    }
                }
                13..=17 | 21..=24 | 26..=35 | 37..=38 | 41..=42 => {
                    let mut diversify_cube = false;
                    let op = match choice {
                        41 => {
                            let length_draw = usize::from(self.byte());
                            let length = 1 + length_draw % 4;
                            let bindings = self.choose(3);
                            let result = self.choose(3 + bindings);
                            let fold = ArrayFold::new(length, bindings, result)?;
                            // Exercise guarded effects only in ordinary runtime
                            // lanes. Other folds remain pure, including calls
                            // used for parametric specialization.
                            CompoundOp::ArrayFold(if constants && (length_draw / 4) % 4 == 1 {
                                fold.with_guarded_assertion()
                            } else {
                                fold
                            })
                        }
                        42 => {
                            let widths = [1, 2, 3, 4, 7, 8, 16, 31, 32, 64, 65];
                            CompoundOp::BitProjection(BitProjection::new(
                                ProjectionKind::ALL[self.choose(3)],
                                widths[self.choose(widths.len())],
                                self.choose(2) != 0,
                                [None, Some(UnaryOp::Not), Some(UnaryOp::Neg)][self.choose(3)],
                            )?)
                        }
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
                            let kind_draw = usize::from(self.byte());
                            let kind = [LoopKind::Range, LoopKind::Array, LoopKind::ConstRange]
                                [kind_draw % 3];
                            if kind != LoopKind::Array && self.grammar.uses_mission_extensions() {
                                let range = self.bounded_range(4, true)?;
                                if self.grammar.uses_v9_productions()
                                    && kind == LoopKind::Range
                                    && (kind_draw / 3) % 2 != 0
                                {
                                    CompoundOp::IgnoredCarryLoop(range)
                                } else {
                                    CompoundOp::RangeLoop {
                                        range,
                                        constant: kind == LoopKind::ConstRange,
                                    }
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
                        28 => self.range_match(constants && self.grammar.uses_v10_productions())?,
                        29 => {
                            let rows = 2 + self.choose(3);
                            let cols = 2 + self.choose(3);
                            let row_draw = usize::from(self.byte());
                            let col_draw = usize::from(self.byte());
                            let mut update =
                                MatrixUpdate::new(rows, cols, row_draw % rows, col_draw % cols)?;
                            if self.grammar.uses_tuple_coordinates() {
                                let forms = UpdateIndexForm::ALL;
                                update =
                                    update.with_index_form(forms[col_draw / cols % forms.len()]);
                            }
                            if self.grammar.uses_v9_productions() {
                                let shape_draw = row_draw / rows;
                                diversify_cube = self.grammar.uses_v10_productions()
                                    && shape_draw % 4 == 3
                                    && (col_draw / cols) % 4 != 0;
                                update = update.with_shape(match shape_draw % 4 {
                                    0 => UpdateShape::MatrixCell,
                                    1 => UpdateShape::WholeMatrix,
                                    2 => UpdateShape::Row,
                                    _ => UpdateShape::CubeCell {
                                        observed_plane: (shape_draw / 4) % 2 != 0,
                                    },
                                });
                            }
                            CompoundOp::MatrixUpdate(update)
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
                        32 => {
                            let [outer, inner, combiner, member] =
                                std::array::from_fn(|_| usize::from(self.byte()));
                            if self.grammar.uses_v9_productions() && (outer / 3) % 2 != 0 {
                                CompoundOp::NestedCapture(NestedCapture::new(
                                    nested_capture_range(outer, combiner)?,
                                    nested_capture_range(inner, member)?,
                                )?)
                            } else {
                                CompoundOp::NestedFold(NestedFold::new(
                                    2 + outer % 3,
                                    1 + inner % 3,
                                    NestedFold::COMBINERS[combiner % NestedFold::COMBINERS.len()],
                                    member % 3,
                                )?)
                            }
                        }
                        33 => CompoundOp::ReceiverMap(ReceiverMap::new(
                            ReceiverMap::WIDTHS[self.choose(ReceiverMap::WIDTHS.len())],
                            2 + self.choose(3),
                        )?),
                        34 => CompoundOp::LogicalPredicates(LogicalPredicates::new(
                            [
                                COMPARISONS[self.choose(COMPARISONS.len())],
                                COMPARISONS[self.choose(COMPARISONS.len())],
                            ],
                            self.choose(2) != 0,
                        )),
                        35 => CompoundOp::CheckedCast(CheckedCast::new(self.choose(5))?),
                        37 => CompoundOp::NamedAggregateMatch(NamedAggregateMatch::new(
                            NamedPatternShape::ALL[self.choose(NamedPatternShape::ALL.len())],
                            std::array::from_fn(|_| 1 + self.choose(8)),
                            [self.byte(), self.byte()],
                        )?),
                        38 => {
                            let extensions = [0, 1, 4, 8, 32, 64, 128];
                            CompoundOp::WideningCast(WideningCast::new(
                                extensions[self.choose(extensions.len())],
                            )?)
                        }
                        _ if self.grammar.uses_mission_extensions() => {
                            CompoundOp::RangeValue(self.bounded_range(8, false)?)
                        }
                        _ => {
                            let length = 1 + self.choose(8) as u8;
                            let start = self.choose(256 - usize::from(length)) as u8;
                            CompoundOp::RangeValue(RangeValue::new(start, length)?)
                        }
                    };
                    let mut third_slot = self.choose(values.len());
                    let mut right = rhs;
                    if diversify_cube {
                        if right == lhs {
                            right = values[(rhs_slot + 1) % values.len()];
                        }
                        // Each SSA value appears once. With three available,
                        // at most two skips are needed to keep the write data
                        // distinct from the two coordinate operands.
                        while values.len() >= 3
                            && (values[third_slot] == lhs || values[third_slot] == right)
                        {
                            third_slot = (third_slot + 1) % values.len();
                        }
                    }
                    Fragment::Compound {
                        op,
                        args: [lhs, right, values[third_slot]],
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
                36 => Fragment::ImportedConstant {
                    module: self.choose(available_modules),
                    arg: lhs,
                    combine: self.export_combiners(),
                },
                39 => Fragment::ImportedStructUpdate {
                    module: self.choose(available_modules),
                    args: [lhs, rhs, values[self.choose(values.len())]],
                    update_hi: self.choose(2) != 0,
                    combine: ExportedConstants::COMBINERS
                        [self.choose(ExportedConstants::COMBINERS.len())],
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
        if matches!(ty.width, Width::Concrete(_))
            && available_modules != 0
            && (constants || !self.grammar.uses_v11_productions())
        {
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

    /// Draws the two operations in a constant definition or its consumer.
    fn export_combiners(&mut self) -> [BinaryOp; 2] {
        std::array::from_fn(|_| {
            ExportedConstants::COMBINERS[self.choose(ExportedConstants::COMBINERS.len())]
        })
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

const AGGREGATE_OPS: [BinaryOp; 5] = [
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Xor,
    BinaryOp::Or,
    BinaryOp::And,
];

/// Draws a nonempty loop range using the two bytes already assigned to this
/// axis in the nested-fold production.
fn nested_capture_range(length: usize, bounds: usize) -> Result<RangeValue, String> {
    let widths = [5, 8, 31, 32, 63, 64, 65, 129];
    let width = widths[bounds % widths.len()];
    let signed = bounds & 8 != 0;
    let inclusive = bounds & 16 != 0;
    let start = (bounds >> 5) as i16 - if signed { 4 } else { 0 };
    RangeValue::typed(signed, width, start, (1 + length % 3) as u8, inclusive)
}

/// Decodes a helper's operation, runtime write index and scalar input.
fn draw_flow_step(choose: &mut impl FnMut(usize) -> usize) -> FlowStep {
    FlowStep {
        op: AGGREGATE_OPS[choose(AGGREGATE_OPS.len())],
        invert_index: choose(2) != 0,
        delta: choose(3),
    }
}

/// Draws aggregate operations from the end without advancing the scalar
/// decoder.
fn aggregate_output(
    data: &[u8],
    sample: &Sample,
    grammar: Grammar,
) -> Result<AggregateOutput, String> {
    let mut bytes = data.iter().rev().copied();
    let mut choose = |count: usize| usize::from(bytes.next().unwrap_or(0)) % count;
    let index = choose(sample.lanes.len());
    let lane = &sample.lanes[index];
    let Width::Concrete(width) = lane.bits_type().width else {
        return Err("aggregate flow requires a concrete top lane".to_string());
    };
    let modules = sample.helpers.len();
    let producer = choose(modules.saturating_sub(1).max(1));
    let updater = if modules == 1 {
        producer
    } else {
        producer + 1 + choose(modules - producer - 1)
    };
    let kind = [FlowKind::Array, FlowKind::Struct, FlowKind::Enum][choose(3)];
    let seed_op = AGGREGATE_OPS[choose(AGGREGATE_OPS.len())];
    let step_draw = choose(256);
    let mut steps: Vec<_> = (0..1 + step_draw % 3)
        .map(|_| draw_flow_step(&mut choose))
        .collect();
    let observation = [FlowObservation::Whole, FlowObservation::Select][choose(2)];
    let [left, right] =
        lane.aggregate_operands(sample.observations[index], [choose(256), choose(256)]);
    let parameter = [ValueRef::X, ValueRef::Y][choose(2)];
    // Decode extra updates after the original operands and observation so
    // increasing the depth preserves the prefix of the aggregate computation.
    if grammar.uses_v9_productions() && (step_draw / 3) % 4 == 1 {
        steps.extend((0..3).map(|_| draw_flow_step(&mut choose)));
    }
    // Reuse the length byte to select a later write that reads the other
    // field as well as its destination, preserving the original write indices.
    let cross_field_step =
        (grammar.uses_v10_productions() && steps.len() > 1 && (step_draw / 12) % 4 == 2)
            .then(|| 1 + (step_draw / 48) % (steps.len() - 1));
    let mut flow = AggregateFlow::new(
        index,
        width,
        [producer, updater],
        kind,
        seed_op,
        steps,
        observation,
    )?;
    if let Some(step) = cross_field_step {
        flow = flow.with_cross_field_update(step)?;
    }
    Ok(AggregateOutput {
        flow,
        operands: [left, right, parameter],
    })
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
        if grammar.uses_v5_productions() {
            let width = 1 + entropy.choose(options.max_width);
            helper = helper.with_exports(ExportedConstants::new(
                entropy.bits(width),
                entropy.bits(width),
                entropy.export_combiners(),
            )?);
        }
        if grammar.uses_v6_productions() {
            let export = if index > 0 && entropy.choose(3) != 0 {
                StructExport::Alias {
                    module: entropy.choose(index),
                }
            } else {
                StructExport::Definition {
                    widths: std::array::from_fn(|_| 1 + entropy.choose(options.max_width)),
                }
            };
            helper = helper.with_struct_export(export);
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
        .map(|(lane, draw)| lane.runtime_intermediate(draw, grammar.uses_deep_lanes()))
        .collect::<Result<_, _>>()?;
    if grammar.uses_aggregate_flow() {
        sample.aggregate = Some(aggregate_output(data, &sample, grammar)?);
    }
    if grammar.uses_v11_productions() {
        // Every program exposes an ordinary call and a specialization of the
        // same closed expression. Width, operations and literals come from the
        // same grammar as other constexpr consumers.
        // Read from the middle independently so short inputs can still mutate
        // the observation after the main decoder has exhausted its bytes.
        let mut observation = Entropy {
            data,
            offset: data.len() / 2,
            grammar,
        };
        let lane = observation.choose(sample.lanes.len());
        let expression =
            observation.constant_expression(sample.lanes[lane].bits_type(), module_count, false)?;
        sample = sample.with_specialization_check(expression)?;
    }
    Ok(sample)
}
