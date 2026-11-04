// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::aig::{ClockPort, GateBuilder, GateBuilderOptions, RegisterBinding, TransitionOutputId};
use crate::aig_sim::gate_sim;
use crate::liberty::parser::parse_liberty_files;
use crate::netlist::register_mapping::RegisterCellSelection;

pub(in crate::netlist) const CLOCK_GATE_LIB: &str =
    include_str!("../../../../tests/data/clock_gate.lib");

pub(in crate::netlist) fn library() -> Library {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), r#"library(test) {
        capacitive_load_unit(1, pf);
        cell(BUF) { area: 1; pin(A) { direction: input; capacitance: 0.0005; }
            pin(Y) { direction: output; function: "A"; } }
        cell(INV) { area: 1; pin(A) { direction: input; } pin(Y) { direction: output; function: "!A"; } }
        cell(DFFN) { area: 1; pin(D) { direction: input; } pin(CLK) { clock: true; }
            pin(QN) { direction: output; function: "IQN"; }
            ff(IQ,IQN) { clocked_on: "CLK"; next_state: "D"; } }
        cell(DFF) { area: 2; pin(D) { direction: input; } pin(CLK) { clock: true; }
            pin(Q) { direction: output; function: "IQ"; }
            ff(IQ,IQN) { clocked_on: "CLK"; next_state: "D"; } }
    }"#).unwrap();
    let mut library = parse_liberty_files(&[file.path()]).unwrap();
    library.reference_buffer_cell = Some("BUF".into());
    library.default_output_load = Some(0.004);
    library
}

/// Includes a shared D/external-output endpoint and an internal-name collision.
pub(in crate::netlist) fn design() -> SequentialGateFn {
    let mut b = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
    let a = b.add_input("__abc_n0".into(), 2);
    let q = b.add_input("q".into(), 2);
    b.add_output("data".into(), a);
    b.add_output("wire".into(), q);
    SequentialGateFn::new(
        "top".into(),
        b.build(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(0), TransitionOutputId::new(1)],
        Some(ClockPort { name: "clk".into() }),
        vec![RegisterBinding {
            name: "r".into(),
            q: TransitionInputId::new(1),
            d: TransitionOutputId::new(0),
            enable: None,
            reset: None,
        }],
    )
    .unwrap()
}

/// Keeps actual logic in the transition graph for external-process tests.
pub(in crate::netlist) fn design_with_logic() -> SequentialGateFn {
    let mut design = design();
    design.transition.outputs[0].bit_vector = AigBitVector::from_lsb_is_index_0(
        &design.transition.outputs[0]
            .bit_vector
            .iter_lsb_to_msb()
            .map(|b| b.negate())
            .collect::<Vec<_>>(),
    );
    design
}

#[test]
fn clock_gate_selection_requires_certified_mappable_cells_and_orders_by_area_then_name()
-> anyhow::Result<()> {
    let raw = tempfile::NamedTempFile::new()?;
    std::fs::write(raw.path(), CLOCK_GATE_LIB)?;
    let mut library = parse_liberty_files(&[raw.path()]).map_err(anyhow::Error::msg)?;

    let legacy = library
        .cells
        .iter_mut()
        .find(|cell| cell.name == "ICG")
        .unwrap();
    let certified = legacy.clone();
    // Older protobufs described the pin roles without establishing latch
    // behavior.
    legacy.clock_gate.as_mut().unwrap().behavior =
        crate::liberty_proto::ClockGateBehavior::Unknown as i32;
    for (name, dont_use, size_only) in [("blocked", true, false), ("sizing", false, true)] {
        let mut cell = certified.clone();
        cell.name = name.into();
        cell.dont_use = Some(dont_use);
        cell.use_for_size_only = Some(size_only);
        library.cells.push(cell);
    }
    let catalog = SequentialMappingCatalog::new(&library).map_err(anyhow::Error::msg)?;
    assert_eq!(catalog.clock_gate_cell, None);
    for (name, area) in [("A_large", 2.0), ("C_small", 1.0), ("B_small", 1.0)] {
        let mut cell = certified.clone();
        cell.name = name.into();
        cell.area = area;
        library.cells.push(cell);
    }
    for _ in 0..2 {
        let catalog = SequentialMappingCatalog::new(&library).map_err(anyhow::Error::msg)?;
        assert_eq!(
            library.cells[catalog.clock_gate_cell.unwrap()].name,
            "B_small"
        );
        library.cells.reverse();
    }
    Ok(())
}

#[test]
fn physical_d_polarity_does_not_invert_shared_external_output() {
    let design = design_with_logic();
    let plan = RegisterMappingLibrary::new(&library())
        .unwrap()
        .map_registers(&design, &RegisterCellSelection::SmallestArea)
        .unwrap();
    assert!(plan.cells[0].inverts_data_to_output);
    let boundary = Boundary::new(&design, &plan, None);
    let values = [true, false, false, true].map(|v| IrBits::from_lsb_is_0(&[v]));
    let outputs = gate_sim::eval(&boundary.graph, &values, gate_sim::Collect::None).outputs;
    let expected = [false, true].map(|v| IrBits::from_lsb_is_0(&[v]));
    assert_eq!(outputs, expected);
    assert_eq!(
        boundary.register_data,
        [
            BoundarySignal::Direct("i0".into()),
            BoundarySignal::Direct("i1".into())
        ]
    );
}

#[test]
fn mixed_boundaries_choose_per_bit_cells_and_rephase_every_state_use() {
    let mut builder = GateBuilder::new("mixed".into(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".into(), 2);
    let q = builder.add_input("q".into(), 4);
    let shared = builder.add_and_binary(*q.get_lsb(0), *q.get_lsb(1));
    let feedback = builder.add_and_binary(*q.get_lsb(3), *q.get_lsb(1));
    builder.add_output(
        "d".into(),
        AigBitVector::from_lsb_is_index_0(&[*a.get_lsb(0), *a.get_lsb(1), shared, feedback]),
    );
    builder.add_output(
        "y".into(),
        AigBitVector::from_lsb_is_index_0(&[
            *q.get_lsb(0),
            *q.get_lsb(2),
            *q.get_lsb(0),
            q.get_lsb(1).negate(),
            feedback,
        ]),
    );
    let design = SequentialGateFn::new(
        "mixed".into(),
        builder.build(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(1)],
        Some(ClockPort { name: "clk".into() }),
        vec![RegisterBinding {
            name: "r".into(),
            q: TransitionInputId::new(1),
            d: TransitionOutputId::new(0),
            enable: None,
            reset: None,
        }],
    )
    .unwrap();
    let plan = RegisterMappingLibrary::new(&library())
        .unwrap()
        .map_registers(&design, &RegisterCellSelection::SmallestArea)
        .unwrap();
    assert_eq!(
        plan.bits
            .iter()
            .map(|bit| (
                plan.cells[bit.cell_index].cell_name.as_str(),
                bit.output_inverted,
            ))
            .collect::<Vec<_>>(),
        [
            ("DFF", false),
            ("DFFN", true),
            ("DFFN", false),
            ("DFFN", false)
        ]
    );
    let boundary = Boundary::new(&design, &plan, None);
    assert_eq!(
        &boundary.register_data[..2],
        &[
            BoundarySignal::Direct("i0".into()),
            BoundarySignal::Direct("i1".into())
        ]
    );
    assert_eq!(
        boundary.external_outputs[0][0],
        BoundarySignal::Direct("i2".into())
    );
    assert_eq!(
        boundary.external_outputs[0][1],
        BoundarySignal::Direct("i4".into())
    );
    assert_eq!(
        boundary.external_outputs[0][2],
        BoundarySignal::Direct("i2".into())
    );
    // Exhaust all states/inputs and compare transition and external outputs.
    // This catches missing phase changes on shared fanout and feedback edges.
    for assignment in 0..64 {
        let source = [
            IrBits::make_ubits(2, assignment & 3).unwrap(),
            IrBits::make_ubits(4, assignment >> 2).unwrap(),
        ];
        let expected = gate_sim::eval(&design.transition, &source, gate_sim::Collect::None).outputs;
        let physical = (0..6)
            .map(|index| IrBits::from_lsb_is_0(&[(assignment & (1 << index) != 0) ^ (index == 3)]))
            .collect::<Vec<_>>();
        let actual = gate_sim::eval(&boundary.graph, &physical, gate_sim::Collect::None).outputs;
        let value = |signal: &BoundarySignal| {
            let values = match signal {
                BoundarySignal::Direct(name) => {
                    &physical[boundary
                        .graph
                        .inputs
                        .iter()
                        .position(|p| &p.name == name)
                        .unwrap()]
                }
                BoundarySignal::Mapped(name) => {
                    &actual[boundary
                        .graph
                        .outputs
                        .iter()
                        .position(|p| &p.name == name)
                        .unwrap()]
                }
            };
            values.get_bit(0).unwrap()
        };
        for (index, bit) in plan.bits.iter().enumerate() {
            assert_eq!(
                value(&boundary.register_data[index])
                    ^ plan.cells[bit.cell_index].inverts_data_to_output
                    ^ bit.output_inverted,
                expected[0].get_bit(index).unwrap()
            );
        }
        for (index, signal) in boundary.external_outputs[0].iter().enumerate() {
            assert_eq!(value(signal), expected[1].get_bit(index).unwrap());
        }
    }
}

#[test]
fn name_validation_ignores_nonphysical_names_for_uniqueness() {
    let mut b = GateBuilder::new("a".into(), GateBuilderOptions::no_opt());
    let zero = b.add_input("a".into(), 0);
    let a = b.add_input("a".into(), 2);
    b.add_output("a".into(), zero);
    b.add_output("y".into(), a);
    let combo = SequentialGateFn::from_gate_fn(b.build());
    combo.validate_legalized().unwrap();
    validate_names(&combo).unwrap();

    let mut registered = design();
    registered.transition.inputs[1].name = registered.transition.inputs[0].name.clone();
    registered.name = registered.transition.inputs[0].name.clone();
    registered.validate_legalized().unwrap();
    validate_names(&registered).unwrap();
}
