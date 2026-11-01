// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::aig::AigRef;
use crate::liberty_model::{InternalPower, LibraryBuilder, TimingArc};

/// Two buffers with opposite sample values and shared condition bindings for
/// testing cache isolation and reusable sample storage.
struct ObservationFixture {
    library: Library,
    gate_fn: GateFn,
    instances: Vec<PowerInstance>,
    topology: PowerTopology,
    previous: SampleState,
    current: SampleState,
}

impl ObservationFixture {
    /// Builds timing and power selectors that share a condition expression.
    fn new(power_groups: usize) -> anyhow::Result<Self> {
        let mut builder = LibraryBuilder::new();
        let a = builder.intern_string("A")?;
        let c = builder.intern_string("C")?;
        let y = builder.intern_string("Y")?;
        let condition = builder.intern_string("!C")?;
        let mut groups = Vec::new();
        for _ in 0..power_groups {
            groups.push(InternalPower {
                when: condition,
                related_pins: vec![a],
                ..InternalPower::default()
            });
        }
        // Observation only checks table kinds; it does not evaluate LUTs.
        let tables = [
            TimingTableKind::RiseTransition,
            TimingTableKind::FallTransition,
        ]
        .into_iter()
        .map(|kind| {
            let mut table = TimingTable::default();
            table.kind = kind;
            table
        })
        .collect();
        builder.library_mut().cells.push(Cell {
            name: "BUF".to_string(),
            pins: vec![
                Pin {
                    name: a,
                    direction: PinDirection::Input as i32,
                    ..Pin::default()
                },
                Pin {
                    name: c,
                    direction: PinDirection::Input as i32,
                    ..Pin::default()
                },
                Pin {
                    name: y,
                    direction: PinDirection::Output as i32,
                    timing_arcs: vec![TimingArc {
                        related_pin: a,
                        when: condition,
                        tables,
                        ..TimingArc::default()
                    }],
                    internal_power: groups,
                    ..Pin::default()
                },
            ],
            ..Cell::default()
        });
        let library = builder.finish();
        let instances: Vec<_> = (0..2)
            .map(|index| PowerInstance {
                instance_name: format!("buf_{index}"),
                cell_type: "BUF".to_string(),
                pins: [
                    ("A", PinDirection::Input),
                    ("C", PinDirection::Input),
                    ("Y", PinDirection::Output),
                ]
                .into_iter()
                .map(|(name, direction)| PowerPin {
                    pin_name: name.to_string(),
                    direction,
                    // Observation consumes the explicitly supplied samples.
                    // Clock sources also let this fixture exercise fallible
                    // sample filling without constructing an AIG.
                    signal: PowerSignal::Clock,
                    connection: PinConnection::Unconnected,
                })
                .collect(),
                is_sequential: false,
                state_signal: None,
            })
            .collect();
        let state_aliases = vec![StateAlias {
            name: "IQN".to_string(),
            signal: PowerSignal::Clock,
            complemented: true,
        }];
        let mut budget = CONDITION_TABLE_BUDGET;
        let bindings = Arc::new(prepare_instance_bindings(
            &library,
            &library.cells[0],
            &instances[0],
            &[0, 1, 2],
            &state_aliases,
            &mut budget,
        ));
        let topology = PowerTopology {
            signals: (0..6)
                .map(|index| Signal {
                    source: PowerSignal::Clock,
                    owner: if index % 3 == 2 {
                        SignalOwner::CellOutput {
                            instance: index / 3,
                        }
                    } else {
                        SignalOwner::ModuleInput
                    },
                    load: EdgeLoadCapacitance::default(),
                    edges: EdgeValue::default(),
                    histogram: EdgeValue::default(),
                })
                .collect(),
            instances: (0..2)
                .map(|index| InstanceTopology {
                    cell: 0,
                    liberty_pins: vec![0, 1, 2],
                    bindings: Arc::clone(&bindings),
                    pin_sources: (0..3)
                        .map(|pin| ResolvedSource::Signal(SignalId(index * 3 + pin)))
                        .collect(),
                    output_signals: vec![None, None, Some(SignalId(index * 3 + 2))],
                    is_sequential: false,
                    state_aliases: state_aliases.clone(),
                })
                .collect(),
            instance_order: vec![0, 1],
        };
        Ok(Self {
            library,
            gate_fn: GateFn {
                name: "test".to_string(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                gates: Vec::new(),
            },
            instances,
            topology,
            previous: SampleState {
                signal_values: vec![false, false, false, true, true, true],
                pin_values: vec![vec![false; 3], vec![true; 3]],
                state_alias_values: vec![vec![true], vec![false]],
            },
            current: SampleState {
                signal_values: vec![true, true, true, false, false, false],
                pin_values: vec![vec![true; 3], vec![false; 3]],
                state_alias_values: vec![vec![false], vec![true]],
            },
        })
    }
}

#[test]
fn cache_reuses_selectors_across_timing_and_power_but_not_instances_or_transitions() {
    let mut fixture = ObservationFixture::new(2).unwrap();
    let model = PowerModelView {
        module_name: "test",
        gate_fn: &fixture.gate_fn,
        module_ports: Vec::new(),
        instances: fixture.instances,
    };
    let mut scratch = TransitionScratch::new(&fixture.topology);
    let mut observed = ObservedActivity::default();
    assert!(Arc::ptr_eq(
        &fixture.topology.instances[0].bindings,
        &fixture.topology.instances[1].bindings
    ));
    for _ in 0..2 {
        observe_transition(
            &model,
            &fixture.library,
            &mut fixture.topology,
            &fixture.previous,
            &fixture.current,
            PowerTransitionKind::Combinational,
            &mut observed,
            &mut scratch,
        )
        .unwrap();
        std::mem::swap(&mut fixture.previous, &mut fixture.current);
    }
    // An unchanged transition must not reuse cached truth to emit extra events.
    observe_transition(
        &model,
        &fixture.library,
        &mut fixture.topology,
        &fixture.previous,
        &fixture.previous,
        PowerTransitionKind::Combinational,
        &mut observed,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(scratch.conditions.evaluated_pairs, 4);
    assert_eq!(scratch.conditions.hits, 8);
    assert_eq!(
        observed.diagnostics,
        GvDynamicPowerDiagnostics {
            when_evaluation_count: 12,
            when_changed_during_transition_count: 12,
            unattributed_output_transition_count: 2,
            multiply_attributed_internal_event_count: 2,
            ..GvDynamicPowerDiagnostics::default()
        }
    );
    let expected_timing: BTreeMap<_, _> = (0..2)
        .map(|instance| {
            (
                TimingCauseKey {
                    output: SignalId(instance * 3 + 2),
                    source: SignalId(instance * 3),
                    source_rise: true,
                    output_rise: true,
                    instance,
                    output_pin: 2,
                    arc: 0,
                },
                1.0,
            )
        })
        .collect();
    let expected_power: BTreeMap<_, _> = (0..2)
        .flat_map(|instance| {
            (0..2).map(move |group| {
                (
                    InternalEventKey {
                        instance,
                        owner_pin: 2,
                        group,
                        source: SignalId(instance * 3),
                        source_rise: true,
                        owner_output_rise: Some(true),
                    },
                    0.5,
                )
            })
        })
        .collect();
    assert_eq!(observed.timing_causes, expected_timing);
    assert_eq!(observed.internal_events, expected_power);
    assert_eq!(
        observed.unattributed,
        BTreeMap::from([((SignalId(2), false), 1), ((SignalId(5), false), 1)])
    );
}

#[test]
fn condition_tables_respect_slot_count_shape_and_shared_budget() {
    let conditions = [PreparedCondition {
        expression: Default::default(),
        term: Ok(BoundConditionTerm::Input(Some(
            ConditionValueSlot::StateAlias(0),
        ))),
    }];
    let mut bytes = 1024;
    assert!(ConditionTable::prepare(&conditions, 10, 1, &mut bytes).is_none());
    let table = ConditionTable::prepare(&conditions, 9, 1, &mut bytes).unwrap();
    assert_eq!(bytes, 0);
    assert!(ConditionTable::prepare(&conditions, 0, 1, &mut bytes).is_none());
    assert_eq!(
        table.row(ConditionValues {
            pins: &[false; 9],
            state_aliases: &[true]
        }),
        Some([true].as_slice())
    );
    assert!(
        table
            .row(ConditionValues {
                pins: &[false; 10],
                state_aliases: &[]
            })
            .is_none()
    );
}

#[test]
fn cache_preserves_lazy_errors_and_counts_only_successful_requests() {
    let mut builder = LibraryBuilder::new();
    let expression = builder.intern_string("C & MISSING").unwrap();
    let invalid_expression = builder.intern_string("C &").unwrap();
    let library = builder.finish();
    let conditions = vec![
        PreparedCondition {
            expression,
            term: Ok(BoundConditionTerm::And(
                Box::new(BoundConditionTerm::Input(Some(
                    ConditionValueSlot::StateAlias(0),
                ))),
                Box::new(BoundConditionTerm::Input(None)),
            )),
        },
        PreparedCondition {
            expression: invalid_expression,
            term: Err("deferred parse error".to_string()),
        },
    ];
    let mut cache = ConditionCache::new(2);
    let mut budget = 1024;
    assert!(ConditionTable::prepare(&conditions[..1], 0, 1, &mut budget).is_none());
    assert!(ConditionTable::prepare(&conditions[1..], 0, 1, &mut budget).is_none());
    let masked_unknown = PreparedCondition {
        expression,
        term: Ok(BoundConditionTerm::And(
            Box::new(BoundConditionTerm::Constant(false)),
            Box::new(BoundConditionTerm::Input(None)),
        )),
    };
    assert!(ConditionTable::prepare(&[masked_unknown], 0, 0, &mut budget).is_none());
    assert_eq!(budget, 1024);
    let mut diagnostics = GvDynamicPowerDiagnostics::default();
    let absent = ConditionValues {
        pins: &[],
        state_aliases: &[],
    };
    assert!(
        when_is_true(
            &library,
            None,
            &conditions,
            ConditionStates::Slots(absent, absent),
            &mut diagnostics,
            &mut cache
        )
        .unwrap()
    );
    assert_eq!(diagnostics, GvDynamicPowerDiagnostics::default());
    let false_values = ConditionValues {
        pins: &[],
        state_aliases: &[false],
    };
    let true_values = ConditionValues {
        pins: &[],
        state_aliases: &[true],
    };
    for _ in 0..2 {
        assert!(
            !when_is_true(
                &library,
                Some(0),
                &conditions,
                ConditionStates::Slots(false_values, false_values),
                &mut diagnostics,
                &mut cache
            )
            .unwrap()
        );
    }
    cache.reset(2);
    assert_eq!(
        when_is_true(
            &library,
            Some(0),
            &conditions,
            ConditionStates::Slots(false_values, true_values),
            &mut diagnostics,
            &mut cache
        )
        .unwrap_err(),
        "Liberty when 'C & MISSING' could not be evaluated from cell pin values"
    );
    assert!(cache.outcomes[0].is_none());
    assert_eq!(
        when_is_true(
            &library,
            Some(1),
            &conditions,
            ConditionStates::Slots(absent, absent),
            &mut diagnostics,
            &mut cache
        )
        .unwrap_err(),
        "deferred parse error"
    );
    assert!(cache.outcomes[1].is_none());
    assert_eq!(cache.evaluated_pairs, 1);
    assert_eq!(cache.hits, 1);
    assert_eq!(
        diagnostics,
        GvDynamicPowerDiagnostics {
            when_evaluation_count: 2,
            ..GvDynamicPowerDiagnostics::default()
        }
    );
}

#[test]
fn sample_fill_overwrites_values_and_reuses_nested_storage() {
    let fixture = ObservationFixture::new(1).unwrap();
    let model = PowerModelView {
        module_name: "test",
        gate_fn: &fixture.gate_fn,
        module_ports: Vec::new(),
        instances: fixture.instances,
    };
    let mut sample = SampleState::new(&model, &fixture.topology);
    let signal_pointer = sample.signal_values.as_ptr();
    let pin_pointers: Vec<_> = sample.pin_values.iter().map(Vec::as_ptr).collect();
    let alias_pointers: Vec<_> = sample.state_alias_values.iter().map(Vec::as_ptr).collect();
    for value in [true, false, true] {
        sample_state_into(&model, &fixture.topology, &[], 0, Some(value), &mut sample).unwrap();
        assert_eq!(sample.signal_values, vec![value; 6]);
        assert_eq!(sample.pin_values, vec![vec![value; 3]; 2]);
        assert_eq!(sample.state_alias_values, vec![vec![!value]; 2]);
        assert_eq!(sample.signal_values.as_ptr(), signal_pointer);
        assert_eq!(
            sample
                .pin_values
                .iter()
                .map(Vec::as_ptr)
                .collect::<Vec<_>>(),
            pin_pointers
        );
        assert_eq!(
            sample
                .state_alias_values
                .iter()
                .map(Vec::as_ptr)
                .collect::<Vec<_>>(),
            alias_pointers
        );
    }
}

#[test]
fn failed_sample_fill_stops_before_transition_accounting() {
    let mut fixture = ObservationFixture::new(1).unwrap();
    fixture.gate_fn.gates.push(AigNode::Literal {
        value: false,
        pir_node_ids: Default::default(),
    });
    fixture.topology.signals[0].source = PowerSignal::Operand(AigRef { id: 0 }.into());
    let model = PowerModelView {
        module_name: "test",
        gate_fn: &fixture.gate_fn,
        module_ports: Vec::new(),
        instances: fixture.instances,
    };
    // The second fill writes its first signal, then fails. Its remaining slots
    // still contain false, unlike the true predecessor, and must not add edges.
    assert_eq!(
        observe_ordered_activity(
            &model,
            &fixture.library,
            &[vec![], vec![]],
            &[Some(true), None],
            &[PowerTransitionKind::Combinational],
            &mut fixture.topology
        )
        .unwrap_err(),
        "power sample omitted selected clock value"
    );
    for signal in &fixture.topology.signals {
        assert_eq!((signal.edges.rise, signal.edges.fall), (0, 0));
    }
}
