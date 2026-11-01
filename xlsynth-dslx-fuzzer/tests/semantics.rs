// SPDX-License-Identifier: Apache-2.0

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use xlsynth_dslx_fuzzer::ast::{
    BinaryOp, BitsType, Body, Comparison, Fragment, HelperModule, SliceKind, UnaryOp, ValueRef,
    Width,
};
use xlsynth_dslx_fuzzer::compound::{ArrayOp, BuiltinOp, CompoundOp, LoopKind, PatternOp};
use xlsynth_dslx_fuzzer::constexpr::{
    ComputedWidth, ConstExpr, ConstInput, ConstantScope, ParametricDefault,
};
use xlsynth_dslx_fuzzer::tuple_pattern::{BindingContext, TupleElement, TuplePattern};
use xlsynth_dslx_fuzzer::{GenerateOptions, Sample, generate, generate_with_options};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::libxls_bridge::{value_from_libxls, value_to_libxls};
use xlsynth_pir::random_inputs::generate_mixed_argument_sets_with_rng;
use xlsynth_pir::{IrBits, IrValue};

/// Checks converter semantics with the native PIR and libxls interpreters.
fn check(sample: &Sample, inputs: Option<Vec<Vec<IrValue>>>) {
    let prepared = sample
        .prepare()
        .unwrap_or_else(|error| panic!("{error}\n{:#?}", sample.sources()));
    let expected = sample.expected_ir().unwrap();
    let expected_fn = expected.get_top_fn().unwrap();
    let converted = xlsynth::IrPackage::parse_ir(&prepared.converted_ir, None).unwrap();
    let actual_fn = converted.get_function(&prepared.converted_top).unwrap();
    let mut rng = StdRng::from_seed([42; 32]);
    let inputs =
        inputs.unwrap_or_else(|| generate_mixed_argument_sets_with_rng(expected_fn, &mut rng, 24));
    for args in inputs {
        let FnEvalResult::Success(expected_value) = eval_fn(expected_fn, &args) else {
            panic!("reference evaluation failed");
        };
        let libxls_args = args
            .iter()
            .map(value_to_libxls)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let actual = actual_fn.interpret(&libxls_args).unwrap();
        let actual = value_from_libxls(&actual, &expected_fn.ret_ty).unwrap();
        assert_eq!(
            expected_value.value, actual,
            "args: {args:?}\nsources: {:?}",
            prepared.modules
        );
    }
}

/// Places every primitive on a separate output so none can be masked by
/// composition.
fn primitive_lanes(signed: bool, width: usize) -> Result<Vec<Body>, String> {
    let ty = BitsType {
        signed,
        width: Width::Concrete(width),
    };
    let mut fragments = Vec::new();
    for op in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::And,
        BinaryOp::Or,
        BinaryOp::Xor,
        BinaryOp::ShiftLeft,
        BinaryOp::ShiftRight,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Mod,
    ] {
        fragments.push(Fragment::Binary {
            op,
            lhs: ValueRef::X,
            rhs: ValueRef::Y,
        });
    }
    for op in [UnaryOp::Not, UnaryOp::Neg] {
        fragments.push(Fragment::Unary {
            op,
            arg: ValueRef::X,
        });
    }
    for op in [
        Comparison::Eq,
        Comparison::Ne,
        Comparison::Lt,
        Comparison::Le,
        Comparison::Gt,
        Comparison::Ge,
    ] {
        fragments.push(Fragment::IfCompare {
            op,
            lhs: ValueRef::X,
            rhs: ValueRef::Y,
            on_true: ValueRef::X,
            on_false: ValueRef::Y,
        });
    }
    for kind in [SliceKind::Fixed, SliceKind::Width] {
        fragments.push(Fragment::ConcatSlice {
            lhs: ValueRef::X,
            rhs: ValueRef::Y,
            kind,
        });
    }
    for shift in [0, 1, width, width + 7] {
        fragments.push(Fragment::WidenShift {
            arg: ValueRef::X,
            extra: 7,
            shift,
        });
    }
    for index in 0..2 {
        fragments.push(Fragment::TupleIndex {
            elements: [ValueRef::X, ValueRef::Y],
            index,
        });
    }
    fragments.push(Fragment::Literal(IrBits::all_ones(width)));
    fragments.push(Fragment::StringLookup {
        bytes: [
            IrBits::zero(8),
            IrBits::from_lsb_fn(8, |index| index < 7),
            IrBits::all_ones(8),
        ],
        index: ValueRef::X,
    });
    for op in BuiltinOp::ALL
        .into_iter()
        .map(CompoundOp::Builtin)
        .chain(ArrayOp::ALL.into_iter().map(CompoundOp::Array))
        .chain(PatternOp::ALL.into_iter().map(CompoundOp::Pattern))
        .chain([
            CompoundOp::Record { update: false },
            CompoundOp::Record { update: true },
        ])
    {
        fragments.push(Fragment::Compound {
            op,
            args: [ValueRef::X, ValueRef::Y, ValueRef::X],
        });
    }
    for kind in [LoopKind::Range, LoopKind::Array, LoopKind::ConstRange] {
        for trips in 0..=4 {
            if kind == LoopKind::Array && trips == 0 {
                continue; /* Array loop contracts require nonempty arrays. */
            }
            fragments.push(Fragment::Compound {
                op: CompoundOp::Loop {
                    kind,
                    start: 1,
                    trips,
                },
                args: [ValueRef::X, ValueRef::Y, ValueRef::X],
            });
        }
    }
    let mut lanes = fragments
        .into_iter()
        .map(|fragment| {
            let mut body = Body::new(ty)?;
            body.push(fragment)?;
            Ok(body)
        })
        .collect::<Result<Vec<_>, String>>()?;
    lanes.extend(tuple_pattern_lanes(signed, width)?);
    Ok(lanes)
}

/// Keeps distinct operands in retained, wildcarded, and rest-skipped positions.
fn tuple_patterns() -> Result<Vec<TuplePattern>, String> {
    let elements = vec![
        TupleElement::Bind(0),
        TupleElement::Bind(1),
        TupleElement::Wildcard(2),
        TupleElement::Bind(2),
    ];
    let mut patterns = vec![TuplePattern::new(elements.clone(), None)?];
    for rest in [0..2, 1..3, 2..4, 1..1, 0..0, 4..4, 0..4] {
        patterns.push(TuplePattern::new(elements.clone(), Some(rest))?);
    }
    let inner = TuplePattern::new(
        vec![
            TupleElement::Bind(2),
            TupleElement::Bind(0),
            TupleElement::Bind(1),
        ],
        Some(1..2),
    )?;
    patterns.push(TuplePattern::new(
        vec![
            TupleElement::Wildcard(0),
            TupleElement::Tuple(inner.clone()),
            TupleElement::Bind(0),
        ],
        Some(0..1),
    )?);
    patterns.push(TuplePattern::new(
        vec![TupleElement::Tuple(TuplePattern::new(
            vec![TupleElement::Bind(1), TupleElement::Tuple(inner)],
            Some(1..1),
        )?)],
        None,
    )?);
    patterns.push(TuplePattern::new(
        vec![TupleElement::Wildcard(0), TupleElement::Wildcard(1)],
        None,
    )?);
    for rest in [0..2, 0..0] {
        let left = TuplePattern::new(
            vec![
                TupleElement::Bind(0),
                TupleElement::Bind(0),
                TupleElement::Bind(1),
            ],
            Some(rest),
        )?;
        let right = TuplePattern::new(vec![TupleElement::Bind(0), TupleElement::Bind(1)], None)?;
        patterns.push(TuplePattern::new(
            vec![TupleElement::Tuple(left), TupleElement::Tuple(right)],
            None,
        )?);
    }
    Ok(patterns)
}

/// Exercises every binding context with a third value distinct from both
/// inputs.
fn tuple_pattern_lanes(signed: bool, width: usize) -> Result<Vec<Body>, String> {
    let mut lanes = Vec::new();
    for pattern in tuple_patterns()? {
        for context in BindingContext::ALL {
            let mut body = Body::new(BitsType {
                signed,
                width: Width::Concrete(width),
            })?;
            let third = body.push(Fragment::Unary {
                op: UnaryOp::Not,
                arg: ValueRef::X,
            })?;
            body.push(Fragment::TuplePattern {
                pattern: pattern.clone(),
                context,
                args: [ValueRef::X, ValueRef::Y, third],
            })?;
            lanes.push(body);
        }
    }
    Ok(lanes)
}

#[test]
fn tuple_bindings_preserve_order_and_skip_whole_subtrees() {
    let bindings = tuple_patterns()
        .unwrap()
        .iter()
        .map(TuplePattern::bound_inputs)
        .collect::<Vec<_>>();
    assert_eq!(
        bindings,
        vec![
            vec![0, 1, 2],
            vec![2],
            vec![0, 2],
            vec![0, 1],
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![],
            vec![2, 1, 0],
            vec![1, 2, 1],
            vec![],
            vec![1, 0, 1],
            vec![0, 0, 1, 0, 1],
        ]
    );
}

#[test]
fn tuple_patterns_reject_invalid_references_ranges_and_tree_budgets() {
    assert!(TuplePattern::new(vec![], None).is_err());
    assert!(TuplePattern::new(vec![TupleElement::Bind(3)], None).is_err());
    assert!(TuplePattern::new(vec![TupleElement::Wildcard(3)], Some(0..1)).is_err());
    let reversed_start = 2;
    for rest in [0..2, reversed_start..1] {
        assert!(TuplePattern::new(vec![TupleElement::Bind(0)], Some(rest)).is_err());
    }
    let mut pattern = TuplePattern::new(vec![TupleElement::Bind(0)], None).unwrap();
    for _ in 0..2 {
        pattern = TuplePattern::new(vec![TupleElement::Tuple(pattern)], None).unwrap();
    }
    assert!(TuplePattern::new(vec![TupleElement::Tuple(pattern)], None).is_err());
    assert!(TuplePattern::new(vec![TupleElement::Bind(0); 64], None).is_err());
}

/// Enumerates small vectors with arbitrary-width arithmetic throughout.
fn all_bits(width: usize) -> Vec<IrBits> {
    let one = IrBits::from_lsb_fn(width, |index| index == 0);
    let mut current = IrBits::zero(width);
    let mut values = Vec::new();
    loop {
        values.push(current.clone());
        current = current.add(&one);
        if current.is_zero() {
            break;
        }
    }
    values
}

#[test]
fn primitives_match_exhaustively_at_small_widths() {
    for signed in [false, true] {
        for width in [1, 2] {
            let lanes = primitive_lanes(signed, width).unwrap();
            let mut inputs = Vec::new();
            for x in all_bits(width) {
                for y in all_bits(width) {
                    let mut args = Vec::new();
                    for _ in &lanes {
                        args.extend([IrValue::from_bits(&x), IrValue::from_bits(&y)]);
                    }
                    inputs.push(args);
                }
            }
            check(&Sample::new(vec![], lanes).unwrap(), Some(inputs));
        }
    }
}

#[test]
fn primitives_match_at_wide_boundaries() {
    for signed in [false, true] {
        for width in [7, 32, 63, 64, 65, 127, 128, 129] {
            check(
                &Sample::new(vec![], primitive_lanes(signed, width).unwrap()).unwrap(),
                None,
            );
        }
    }
}

#[test]
fn generated_module_environments_match() {
    check(&generate(&[]).unwrap(), None);
    let mut rng = StdRng::from_seed([7; 32]);
    for _ in 0..48 {
        let mut data = vec![0; 768];
        rng.fill_bytes(&mut data);
        let sample = generate(&data).unwrap();
        assert_eq!(sample, generate(&data).unwrap());
        check(&sample, None);
    }
}

#[test]
fn configurable_width_bounds_preserve_valid_composition() {
    let mut rng = StdRng::from_seed([63; 32]);
    for max_width in [1, 8, 65, 129] {
        for _ in 0..8 {
            let mut data = vec![0; 1024];
            rng.fill_bytes(&mut data);
            let sample = generate_with_options(&data, GenerateOptions { max_width }).unwrap();
            for lane in sample.lanes() {
                let Width::Concrete(width) = lane.bits_type().width else {
                    panic!("entry width must be concrete");
                };
                assert!((1..=max_width).contains(&width));
            }
            check(&sample, None);
        }
    }
    for max_width in [0, 130] {
        assert!(generate_with_options(&[], GenerateOptions { max_width }).is_err());
    }
}

#[test]
fn primitives_evaluate_in_explicit_constant_contexts() {
    for signed in [false, true] {
        let mut lanes = Vec::new();
        for (index, body) in primitive_lanes(signed, 65).unwrap().into_iter().enumerate() {
            let ty = body.bits_type();
            let expression = ConstExpr::new(
                body,
                [
                    ConstInput::Literal(IrBits::all_ones(65)),
                    ConstInput::Literal(IrBits::from_lsb_fn(65, |bit| bit == 0)),
                ],
            )
            .unwrap();
            let mut lane = Body::new(ty).unwrap();
            lane.push(Fragment::Constant {
                expression: Box::new(expression),
                scope: if index % 2 == 0 {
                    ConstantScope::Module
                } else {
                    ConstantScope::Local
                },
                assert_value: true,
            })
            .unwrap();
            lanes.push(lane);
        }
        check(&Sample::new(vec![], lanes).unwrap(), None);
    }
}

/// Builds a module chain instantiated repeatedly at distinct signednesses and
/// widths.
fn nested_sample() -> Result<Sample, String> {
    let mut modules = Vec::new();
    for module in 0..2 {
        let mut bodies = Vec::new();
        for signed in [false, true] {
            let mut body = Body::new(BitsType {
                signed,
                width: Width::Parametric,
            })?;
            let sum = body.push(Fragment::Binary {
                op: BinaryOp::Sub,
                lhs: ValueRef::X,
                rhs: ValueRef::Y,
            })?;
            if module == 0 {
                body.push(Fragment::IfCompare {
                    op: Comparison::Lt,
                    lhs: sum,
                    rhs: ValueRef::Y,
                    on_true: ValueRef::X,
                    on_false: sum,
                })?;
            } else {
                body.push(Fragment::Call {
                    module: 0,
                    args: [ValueRef::Y, sum],
                    explicit_parametric: true,
                })?;
            }
            bodies.push(body);
        }
        let signed = bodies.pop().ok_or("missing signed body")?;
        let unsigned = bodies.pop().ok_or("missing unsigned body")?;
        modules.push(HelperModule::new(unsigned, signed)?);
    }
    let mut lanes = Vec::new();
    for signed in [false, true] {
        for width in [2, 65, 129] {
            let mut body = Body::new(BitsType {
                signed,
                width: Width::Concrete(width),
            })?;
            let first = body.push(Fragment::Call {
                module: 1,
                args: [ValueRef::X, ValueRef::Y],
                explicit_parametric: false,
            })?;
            let second = body.push(Fragment::Call {
                module: 0,
                args: [ValueRef::Y, first],
                explicit_parametric: true,
            })?;
            body.push(Fragment::Binary {
                op: BinaryOp::Xor,
                lhs: first,
                rhs: second,
            })?;
            lanes.push(body);
        }
    }
    Sample::new(modules, lanes)
}

#[test]
fn nested_modules_specialize_at_multiple_widths() {
    let sample = nested_sample().unwrap();
    check(&sample, None);
    let first = sample.prepare().unwrap();
    let second = sample.prepare().unwrap();
    assert_eq!(first.converted_ir, second.converted_ir);
    assert_eq!(first.expected_ir, second.expected_ir);
}

/// Builds observable constant contexts using the same independent fragments.
fn constexpr_sample() -> Result<Sample, String> {
    let u32_type = BitsType {
        signed: false,
        width: Width::Concrete(32),
    };
    let mut modules = Vec::new();
    for index in 0..2 {
        let mut unsigned = Body::new(BitsType {
            signed: false,
            width: Width::Parametric,
        })?;
        unsigned.push(Fragment::Binary {
            op: BinaryOp::Xor,
            lhs: ValueRef::X,
            rhs: ValueRef::Y,
        })?;
        let mut signed = Body::new(BitsType {
            signed: true,
            width: Width::Parametric,
        })?;
        signed.push(Fragment::Binary {
            op: BinaryOp::Sub,
            lhs: ValueRef::X,
            rhs: ValueRef::Y,
        })?;
        let mut default = Body::new(u32_type)?;
        default.push(if index == 0 {
            Fragment::Binary {
                op: BinaryOp::Add,
                lhs: ValueRef::X,
                rhs: ValueRef::Y,
            }
        } else {
            Fragment::Call {
                module: 0,
                args: [ValueRef::X, ValueRef::Y],
                explicit_parametric: false,
            }
        })?;
        let expression = ConstExpr::new(
            default,
            [
                ConstInput::BitWidth,
                ConstInput::Literal(IrBits::from_lsb_fn(32, |bit| bit == 0)),
            ],
        )?;
        modules.push(
            HelperModule::new(unsigned, signed)?.with_default(ParametricDefault {
                expression,
                combine: BinaryOp::Add,
            })?,
        );
    }
    let mut lanes = Vec::new();
    for signed in [false, true] {
        for width in [2, 65, 129] {
            let ty = BitsType {
                signed,
                width: Width::Concrete(width),
            };
            let mut lane = Body::new(ty)?;
            for scope in [ConstantScope::Module, ConstantScope::Local] {
                let mut constant = Body::new(ty)?;
                constant.push(Fragment::Call {
                    module: 1,
                    args: [ValueRef::X, ValueRef::Y],
                    explicit_parametric: true,
                })?;
                let expression = ConstExpr::new(
                    constant,
                    [
                        ConstInput::Literal(IrBits::all_ones(width)),
                        ConstInput::Literal(IrBits::from_lsb_fn(width, |bit| bit == 0)),
                    ],
                )?;
                let value = lane.push(Fragment::Constant {
                    expression: Box::new(expression),
                    scope,
                    assert_value: true,
                })?;
                lane.push(Fragment::Binary {
                    op: BinaryOp::Xor,
                    lhs: lane.result(),
                    rhs: ValueRef::X,
                })?;
                lane.push(Fragment::Call {
                    module: 0,
                    args: [value, ValueRef::Y],
                    explicit_parametric: false,
                })?;
            }
            let mut dimension = Body::new(u32_type)?;
            dimension.push(Fragment::Binary {
                op: BinaryOp::Add,
                lhs: ValueRef::X,
                rhs: ValueRef::Y,
            })?;
            let expression = ConstExpr::new(
                dimension,
                [
                    ConstInput::Literal(IrBits::from_lsb_fn(32, |bit| {
                        bit < usize::BITS as usize && (width - 1) & (1usize << bit) != 0
                    })),
                    ConstInput::Literal(IrBits::from_lsb_fn(32, |bit| bit == 0)),
                ],
            )?;
            lane.set_computed_width(ComputedWidth {
                expression: Box::new(expression),
                assert_value: true,
            })?;
            lanes.push(lane);
        }
    }
    Sample::new(modules, lanes)
}

#[test]
fn constexpr_contexts_match_at_multiple_widths() {
    let sample = constexpr_sample().unwrap();
    check(&sample, None);
    assert_eq!(sample.sources().unwrap(), sample.sources().unwrap());
}

#[test]
fn rejects_invalid_constant_contexts() {
    let ty = BitsType {
        signed: false,
        width: Width::Concrete(32),
    };
    assert!(
        ConstExpr::new(
            Body::new(ty).unwrap(),
            [
                ConstInput::Literal(IrBits::zero(65)),
                ConstInput::Literal(IrBits::zero(32))
            ]
        )
        .is_err()
    );
    let dependent = ConstExpr::new(
        Body::new(ty).unwrap(),
        [ConstInput::BitWidth, ConstInput::Literal(IrBits::zero(32))],
    )
    .unwrap();
    assert!(dependent.evaluate(&[], None).is_err());
    let mut lane = Body::new(ty).unwrap();
    assert!(
        lane.set_computed_width(ComputedWidth {
            expression: Box::new(dependent.clone()),
            assert_value: true
        })
        .is_err()
    );
    assert!(
        lane.push(Fragment::Constant {
            expression: Box::new(dependent),
            scope: ConstantScope::Local,
            assert_value: false
        })
        .is_err()
    );
    let wrong_width = ConstExpr::new(
        Body::new(ty).unwrap(),
        [
            ConstInput::Literal(IrBits::zero(32)),
            ConstInput::Literal(IrBits::zero(32)),
        ],
    )
    .unwrap();
    lane.set_computed_width(ComputedWidth {
        expression: Box::new(wrong_width),
        assert_value: true,
    })
    .unwrap();
    assert!(Sample::new(vec![], vec![lane]).is_err());
}

#[test]
fn interpreter_entry_preserves_every_argument_vector() {
    let sample = constexpr_sample().unwrap();
    let reference = sample.expected_ir().unwrap();
    let top = reference.get_top_fn().unwrap();
    let mut rng = StdRng::from_seed([53; 32]);
    let inputs = generate_mixed_argument_sets_with_rng(top, &mut rng, 4);
    let modules = sample.interpreter_sources(&inputs).unwrap();
    let directory = tempfile::tempdir().unwrap();
    for module in &modules {
        std::fs::write(
            directory.path().join(format!("{}.x", module.name)),
            &module.text,
        )
        .unwrap();
    }
    let converted = xlsynth::convert_dslx_to_ir_text(
        &modules.last().unwrap().text,
        &directory.path().join("sample.x"),
        &xlsynth::DslxConvertOptions {
            additional_search_paths: vec![directory.path()],
            ..Default::default()
        },
    )
    .unwrap();
    let package = xlsynth::IrPackage::parse_ir(&converted.ir, None).unwrap();
    let entry_name = xlsynth::mangle_dslx_name("sample", "coverage_entry").unwrap();
    let entry = package.get_function(&entry_name).unwrap();
    let actual = entry.interpret(&[]).unwrap();
    let mut expected = Vec::new();
    for args in &inputs {
        let FnEvalResult::Success(result) = eval_fn(top, args) else {
            panic!("reference evaluation failed");
        };
        expected.push(result.value);
    }
    let result_type =
        xlsynth_pir::ir::Type::Tuple(vec![Box::new(top.ret_ty.clone()); inputs.len()]);
    assert_eq!(
        value_from_libxls(&actual, &result_type).unwrap(),
        IrValue::make_tuple(&expected)
    );
}

#[test]
fn interpreter_entry_rejects_invalid_argument_vectors() {
    let sample = generate(&[]).unwrap();
    assert!(sample.interpreter_sources(&[]).is_err());
    assert!(sample.interpreter_sources(&[vec![]]).is_err());
    assert!(
        sample
            .interpreter_sources(&[vec![IrValue::make_token(); 2]])
            .is_err()
    );
    assert!(
        sample
            .interpreter_sources(&[vec![IrValue::from_bits(&IrBits::zero(65)); 2]])
            .is_err()
    );
}

#[test]
fn rejects_invalid_fragment_and_module_compositions() {
    let ty = BitsType {
        signed: false,
        width: Width::Concrete(65),
    };
    assert!(
        Body::new(BitsType {
            width: Width::Concrete(0),
            ..ty
        })
        .is_err()
    );
    let mut body = Body::new(ty).unwrap();
    assert!(body.push(Fragment::Literal(IrBits::all_ones(64))).is_err());
    assert!(
        body.push(Fragment::TupleIndex {
            elements: [ValueRef::X, ValueRef::Y],
            index: 2
        })
        .is_err()
    );
    let mut other = Body::new(ty).unwrap();
    let future = other
        .push(Fragment::Unary {
            op: UnaryOp::Not,
            arg: ValueRef::X,
        })
        .unwrap();
    assert!(
        body.push(Fragment::Unary {
            op: UnaryOp::Neg,
            arg: future
        })
        .is_err()
    );
    body.push(Fragment::Call {
        module: 0,
        args: [ValueRef::X, ValueRef::Y],
        explicit_parametric: false,
    })
    .unwrap();
    assert!(Sample::new(vec![], vec![body]).is_err());
    for signed in [false, true] {
        let mut recursive = Body::new(BitsType {
            signed,
            width: Width::Parametric,
        })
        .unwrap();
        recursive
            .push(Fragment::Call {
                module: 0,
                args: [ValueRef::X, ValueRef::Y],
                explicit_parametric: false,
            })
            .unwrap();
        let identity = Body::new(BitsType {
            signed: !signed,
            width: Width::Parametric,
        })
        .unwrap();
        let helper = if signed {
            HelperModule::new(identity, recursive)
        } else {
            HelperModule::new(recursive, identity)
        }
        .unwrap();
        assert!(Sample::new(vec![helper], vec![Body::new(ty).unwrap()]).is_err());
    }
}

#[test]
fn empty_seed_source_goldens() {
    let modules = generate(&[]).unwrap().sources().unwrap();
    assert_eq!(modules.len(), 2);
    assert_eq!(modules[0].name, "m0");
    assert_eq!(modules[0].text, include_str!("goldens/empty_m0.x"));
    assert_eq!(modules[1].name, "sample");
    assert_eq!(modules[1].text, include_str!("goldens/empty_sample.x"));
}
