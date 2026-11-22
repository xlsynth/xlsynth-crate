// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{
    BlockAssumptionFailure, BlockEvalError, BlockEvalSite, BlockEvaluator, RegisterInit,
    test_fixtures,
};
use xlsynth_pir::block_inline::{flatten_block_hierarchy, inline_all_blocks_in_package};
use xlsynth_pir::block_transition::lower_block_transition;
use xlsynth_pir::ir::NodeRef;
use xlsynth_pir::ir_eval::{AssumptionFailureKind, EvalObserver, SelectEvent, SelectKind};
use xlsynth_pir::ir_parser::Parser;

#[test]
fn mixed_reset_is_sampled_on_the_edge_and_writes_read_old_state() {
    let pkg = Parser::new(test_fixtures::MIXED_RESET)
        .parse_and_validate_package()
        .unwrap();
    let evaluator = BlockEvaluator::prepare(&pkg, "pair").unwrap();
    let low = IrValue::make_ubits(1, 0).unwrap();
    let high = IrValue::make_ubits(1, 1).unwrap();
    let zero = IrValue::make_ubits(8, 0).unwrap();
    let reset = IrValue::make_ubits(8, 7).unwrap();
    let a = IrValue::make_ubits(8, 18).unwrap();
    let b = IrValue::make_ubits(8, 42).unwrap();

    let zeros = evaluator.initial_state(RegisterInit::AllZeros).unwrap();
    assert_eq!(zeros.values(), [zero.clone(), zero]);
    let old = evaluator
        .initial_state(RegisterInit::Supplied(vec![a.clone(), b.clone()]))
        .unwrap();

    let swapped = evaluator
        .step(&old, &[high.clone(), high.clone(), high.clone()])
        .unwrap();
    assert_eq!(swapped.outputs, old.values());
    assert_eq!(swapped.next_state.values(), [b.clone(), a.clone()]);

    let held = evaluator
        .step(&old, &[high.clone(), low.clone(), low.clone()])
        .unwrap();
    assert_eq!(held.next_state.values(), old.values());

    // With a disabled and b enabled, reset takes precedence for a and b samples
    // old a.
    let enabled = evaluator
        .step(&old, &[low.clone(), low.clone(), high.clone()])
        .unwrap();
    assert_eq!(enabled.outputs, old.values());
    assert_eq!(enabled.next_state.values(), [reset.clone(), a.clone()]);

    // With reset still asserted, disabling b now holds its last sampled value.
    let held_on_reset = evaluator
        .step(&enabled.next_state, &[low.clone(), high, low])
        .unwrap();
    assert_eq!(held_on_reset.outputs, enabled.next_state.values());
    assert_eq!(held_on_reset.next_state.values(), [reset, a]);
}

#[test]
fn exclusive_writes_reject_overlap_only_when_reset_is_inactive() {
    let source = test_fixtures::EXCLUSIVE_WRITES;
    let pkg = Parser::new(source).parse_and_validate_package().unwrap();
    let eval = BlockEvaluator::prepare(&pkg, "dual").unwrap();
    let start = eval.initial_state(RegisterInit::AllZeros).unwrap();
    let mut inputs = vec![IrValue::make_ubits(1, 1).unwrap(); 3];
    let reset = eval.step(&start, &inputs).unwrap();
    assert!(reset.outputs.is_empty());
    assert_eq!(
        reset.next_state.values(),
        [IrValue::make_ubits(8, 9).unwrap()]
    );
    inputs[0] = IrValue::make_ubits(1, 0).unwrap();
    assert!(matches!(
        eval.step(&reset.next_state, &inputs),
        Err(BlockEvalError::ConflictingWrites { register, first: 7, second: 8 }) if register == "r"
    ));
    for (x, y, expected) in [(0, 0, 9), (0, 1, 2), (1, 0, 1)] {
        inputs[1] = IrValue::make_ubits(1, x).unwrap();
        inputs[2] = IrValue::make_ubits(1, y).unwrap();
        let outcome = eval.step(&reset.next_state, &inputs).unwrap();
        assert_eq!(
            outcome.next_state.values(),
            [IrValue::make_ubits(8, expected).unwrap()]
        );
    }
}

#[test]
fn aggregate_state_preserves_width_shape_and_random_seed() {
    let source = test_fixtures::AGGREGATE_STATE;
    let pkg = Parser::new(source).parse_and_validate_package().unwrap();
    let eval = BlockEvaluator::prepare(&pkg, "store").unwrap();
    let zeros = eval.initial_state(RegisterInit::AllZeros).unwrap();
    let zero = IrValue::parse_typed("(bits[129]:0, [bits[3]:0, bits[3]:0], bits[0]:0)").unwrap();
    assert_eq!(zeros.values(), [zero.clone()]);
    let seeded = eval
        .initial_state(RegisterInit::Random { seed: 7 })
        .unwrap();
    let replay = eval
        .initial_state(RegisterInit::Random { seed: 7 })
        .unwrap();
    let other = eval
        .initial_state(RegisterInit::Random { seed: 42 })
        .unwrap();
    assert_eq!(seeded.values(), replay.values());
    assert_ne!(seeded.values(), other.values());
    assert_eq!(seeded.values()[0].type_(), zero.type_());

    assert!(eval.initial_state(RegisterInit::Supplied(vec![])).is_err());
    assert!(
        eval.initial_state(RegisterInit::Supplied(vec![
            IrValue::make_ubits(135, 0).unwrap()
        ]))
        .is_err()
    );
    assert!(eval.step(&seeded, &[]).is_err());
    assert!(
        eval.step(&seeded, &[IrValue::make_ubits(135, 0).unwrap()])
            .is_err()
    );
    let step = eval.step(&seeded, std::slice::from_ref(&zero)).unwrap();
    assert_eq!(step.outputs, seeded.values());
    assert_eq!(step.next_state.values(), [zero]);
}

#[test]
fn sibling_instances_use_independent_state_and_commit_together() {
    let source = test_fixtures::SIBLING_STATE;
    let pkg = Parser::new(source).parse_and_validate_package().unwrap();
    let eval = BlockEvaluator::prepare(&pkg, "pair").unwrap();
    assert_eq!(
        eval.registers()
            .iter()
            .map(|reg| reg.name.as_str())
            .collect::<Vec<_>>(),
        ["left.r", "right.r"]
    );
    let left = IrValue::make_ubits(32, 123).unwrap();
    let right = IrValue::make_ubits(32, 456).unwrap();
    let supplied = eval
        .initial_state(RegisterInit::Supplied(vec![left.clone(), right.clone()]))
        .unwrap();
    let mut observed = Observed::default();
    eval.step_with_instance_observer(&supplied, &[], "right", &mut observed)
        .unwrap();
    // The sibling instances share source IDs. Only the right instance's Q
    // and D (the left instance's Q) should appear in this observer.
    assert_eq!(observed.values[&1].1, left);
    assert_eq!(observed.values[&2].1, right);
    assert!(
        eval.step_with_instance_observer(&supplied, &[], "missing", &mut observed)
            .is_err()
    );
    let independent = BlockEvaluator::prepare(&pkg, "pair").unwrap();
    assert!(independent.step(&supplied, &[]).is_err());
    let seeded = eval
        .initial_state(RegisterInit::Random { seed: 42 })
        .unwrap();
    let replay = independent
        .initial_state(RegisterInit::Random { seed: 42 })
        .unwrap();
    assert_eq!(seeded.values(), replay.values());
    let restored_seed = eval
        .initial_state(RegisterInit::Supplied(replay.values().to_vec()))
        .unwrap();
    assert_eq!(
        eval.step(&seeded, &[]).unwrap().outputs,
        eval.step(&restored_seed, &[]).unwrap().outputs
    );
    let exchanged = eval.step(&supplied, &[]).unwrap();
    assert_eq!(exchanged.outputs, supplied.values());
    assert_eq!(exchanged.next_state.values(), [right, left]);
    let restored = eval.step(&exchanged.next_state, &[]).unwrap();
    assert_eq!(restored.next_state.values(), supplied.values());
}

#[test]
fn external_block_evaluates_its_model_with_local_clock_and_reset_ports() {
    let ir = test_fixtures::MODELED_EXTERNAL;
    let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
    let evaluator = BlockEvaluator::prepare(&pkg, "top").unwrap();
    assert_eq!(evaluator.registers()[0].name, "delay.saved");

    let zero = IrValue::make_ubits(8, 0).unwrap();
    let reset = IrValue::make_ubits(8, 42).unwrap();
    let data = IrValue::make_ubits(8, 19).unwrap();
    let start = evaluator.initial_state(RegisterInit::AllZeros).unwrap();
    let asserted = evaluator
        .step(&start, &[IrValue::bool(false), data.clone()])
        .unwrap();
    assert_eq!(asserted.outputs, [zero]);
    assert_eq!(asserted.next_state.values(), [reset.clone()]);

    let released = evaluator
        .step(&asserted.next_state, &[IrValue::bool(true), data.clone()])
        .unwrap();
    assert_eq!(released.outputs, [reset]);
    assert_eq!(released.next_state.values(), [data]);
}

#[test]
fn reachable_opaque_instance_reports_qualified_path_before_inout_rejection() {
    let ir = test_fixtures::OPAQUE_EXTERNAL;
    let mut package = Parser::new(ir).parse_and_validate_package().unwrap();
    BlockEvaluator::prepare(&package, "independent").unwrap();
    flatten_block_hierarchy(&package, "independent").unwrap();
    lower_block_transition(&package, "independent").unwrap();

    let expected = "opaque external block instance 'wrapper.leaf.black_box' of module 'vendor_io' has no behavioral model";
    let Err(BlockEvalError::Preparation(actual)) = BlockEvaluator::prepare(&package, "wrapper")
    else {
        panic!("expected an opaque instance preparation error");
    };
    assert_eq!(actual, expected);
    assert_eq!(
        flatten_block_hierarchy(&package, "wrapper").unwrap_err(),
        expected
    );
    assert_eq!(
        lower_block_transition(&package, "wrapper").unwrap_err(),
        expected
    );

    package.set_top_block("wrapper").unwrap();
    let before = package.to_string();
    assert_eq!(
        inline_all_blocks_in_package(&mut package).unwrap_err(),
        expected
    );
    assert_eq!(package.to_string(), before);
}

/// Original block node identities reported by a prepared evaluator.
#[derive(Default)]
struct Observed {
    values: BTreeMap<usize, (NodeRef, IrValue)>,
    selections: Vec<SelectEvent>,
}

impl EvalObserver for Observed {
    fn on_node_value(&mut self, node: NodeRef, id: usize, value: &IrValue) {
        assert!(self.values.insert(id, (node, value.clone())).is_none());
    }
    fn on_select(&mut self, event: SelectEvent) {
        self.selections.push(event);
    }
}

#[test]
fn observed_nodes_keep_the_original_ids_and_header_order() {
    let source = test_fixtures::CALLED_FUNCTION;
    let pkg = Parser::new(source).parse_and_validate_package().unwrap();
    let block = pkg.get_top_block().unwrap();
    let eval = BlockEvaluator::prepare(&pkg, &block.name).unwrap();
    let state = eval.initial_state(RegisterInit::AllZeros).unwrap();
    let mut seen = Observed::default();
    let inputs = [
        IrValue::make_ubits(8, 9).unwrap(),
        IrValue::make_ubits(1, 1).unwrap(),
        IrValue::make_ubits(8, 3).unwrap(),
    ];
    let cycle = eval.step_with_observer(&state, &inputs, &mut seen).unwrap();
    let expected_output = IrValue::make_ubits(8, 10).unwrap();
    assert_eq!(cycle.outputs, [expected_output.clone()]);
    let by_id: BTreeMap<_, _> = block
        .node_refs()
        .into_iter()
        .map(|r| (block.get_node(r).text_id, r))
        .collect();
    assert_eq!(seen.values.len(), by_id.len());
    for (id, expected) in [
        (1, inputs[2].clone()),
        (2, inputs[0].clone()),
        (3, inputs[1].clone()),
        (4, expected_output.clone()),
        (5, expected_output),
        (6, IrValue::make_ubits(8, 6).unwrap()),
        (7, IrValue::make_tuple(&[])),
    ] {
        assert_eq!(seen.values.get(&id), Some(&(by_id[&id], expected)));
    }
    assert_eq!(
        seen.selections,
        [SelectEvent {
            node_ref: by_id[&5],
            node_text_id: 5,
            select_kind: SelectKind::CaseIndex,
            selected_index: 1,
        }]
    );
}

#[test]
fn rejects_asynchronous_reset_during_preparation() {
    let source = test_fixtures::MIXED_RESET.replace("asynchronous=false", "asynchronous=true");
    let pkg = Parser::new(&source).parse_and_validate_package().unwrap();
    assert!(BlockEvaluator::prepare(&pkg, "pair").is_err());
}

#[test]
fn failed_child_assumption_reports_its_source_site_and_state_can_be_retried() {
    let source = test_fixtures::CHILD_BOUNDS;
    let pkg = Parser::new(source).parse_and_validate_package().unwrap();
    let eval = BlockEvaluator::prepare(&pkg, "parent").unwrap();
    let state = eval.initial_state(RegisterInit::AllZeros).unwrap();
    let data = IrValue::parse_typed("[bits[8]:11, bits[8]:22]").unwrap();
    let Err(BlockEvalError::Runtime(failure)) =
        eval.step(&state, &[data.clone(), IrValue::make_ubits(2, 2).unwrap()])
    else {
        panic!("out-of-bounds array index must fail its bounds assumption")
    };
    assert_eq!(
        failure.assumption_failures,
        [BlockAssumptionFailure {
            site: BlockEvalSite {
                instance: "child".to_string(),
                node_text_id: 3
            },
            kind: AssumptionFailureKind::ArrayIndexOutOfBounds,
        }]
    );
    let retry = eval
        .step(&state, &[data, IrValue::make_ubits(2, 1).unwrap()])
        .unwrap();
    assert_eq!(retry.outputs, [IrValue::make_ubits(8, 0).unwrap()]);
    assert_eq!(
        retry.next_state.values(),
        [IrValue::make_ubits(8, 22).unwrap()]
    );
}

#[test]
fn outputless_children_execute_each_cover_and_aggregate_function_calls_once() {
    let source = test_fixtures::CALLED_COVER;
    let pkg = Parser::new(source).parse_package().unwrap();
    let eval = BlockEvaluator::prepare(&pkg, "pair").unwrap();
    let state = eval.initial_state(RegisterInit::AllZeros).unwrap();
    for (a, b) in [(true, true), (false, true), (false, false)] {
        let result = eval
            .step(&state, &[IrValue::bool(a), IrValue::bool(b)])
            .unwrap();
        assert!(result.outputs.is_empty());
        let counts: BTreeMap<_, _> = result
            .cover_counts
            .iter()
            .map(|c| {
                (
                    (
                        c.site.instance.as_str(),
                        c.site.node_text_id,
                        c.label.as_str(),
                    ),
                    c.count,
                )
            })
            .collect();
        assert_eq!(
            counts,
            BTreeMap::from([
                (("", 2, "call"), u64::from(a) + u64::from(b)),
                (("left", 6, "child"), u64::from(a)),
                (("right", 6, "child"), u64::from(b)),
            ])
        );
    }
}
