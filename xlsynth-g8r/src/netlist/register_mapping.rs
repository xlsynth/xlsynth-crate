// SPDX-License-Identifier: Apache-2.0

//! Deterministic, area-first mapping of logical state to Liberty flip-flops.
//!
//! This produces register binding plans, not a complete netlist. Combinational
//! mapping, materializing data inversions, and final module assembly belong to
//! the caller. No register cleanup or timing-driven sizing is performed here.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::aig::sequential_gate::{
    ClockPort, SequentialGateFn, TransitionInputId, TransitionOutputId,
};
use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{Cell, Library, PinDirection};
use crate::netlist::sequential_liberty::{
    GvEvalSequentialCellSpec, get_gv_eval_sequential_cell_spec,
};

// Exhaustive tie-off checking is intentionally bounded: at most 9 candidate
// data pins, 256 control assignments each, and four state/data valuations.
const MAX_CONTROL_PINS: usize = 8;

/// Selects the smallest supported physical FF, or fixes its cell name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum RegisterCellSelection {
    /// Minimize FF area, using stable tie-breaks and no timing assumptions.
    #[default]
    SmallestArea,
    /// Require this cell; unsupported requests fail rather than falling back.
    FixedCell(String),
}

/// A verified single-bit FF implementation of logical `Q_next = D`.
#[derive(Clone, Debug, PartialEq)]
pub struct RegisterCellMapping {
    pub cell_name: String,
    /// Physical FF area only; excludes any required data inverter.
    pub area: f64,
    pub clock_pin: String,
    pub data_pin: String,
    /// Connect this output directly to logical Q, regardless of its pin name.
    pub output_pin: String,
    /// Physical data receives `!D` when true, otherwise `D`. This accounts for
    /// both next-state and output polarity; no restoring Q inverter is needed.
    pub invert_data: bool,
    /// Every other non-clock input is driven to this verified constant value.
    pub tied_inputs: BTreeMap<String, bool>,
    /// Other outputs are left unconnected, not tied to constants.
    pub unused_outputs: Vec<String>,
}

/// One physical FF instance's logical source bit, before net names are
/// assigned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterBitMapping {
    pub register_name: String,
    /// Original LSB-based bit position within the packed logical register.
    pub bit_index: usize,
    pub q: TransitionInputId,
    pub d: TransitionOutputId,
}

/// Register-only mapping plan tied to the supplied sequential interface.
#[derive(Clone, Debug, PartialEq)]
pub struct RegisterMappingPlan {
    pub clock: Option<ClockPort>,
    /// All physical bits use the selected cell. None for a state-free design.
    pub cell: Option<RegisterCellMapping>,
    /// Original register order, then LSB-first bit order. No state is removed.
    pub bits: Vec<RegisterBitMapping>,
}

/// An unsupported sequential cell and the reason it cannot be selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RejectedRegisterCell {
    pub cell_name: String,
    pub reason: String,
}

/// Reusable FF catalog; independent of timing tables and the source library's
/// lifetime.
#[derive(Clone, Debug)]
pub struct RegisterMappingLibrary {
    cells: Vec<RegisterCellMapping>,
    rejected: Vec<RejectedRegisterCell>,
}

impl RegisterMappingLibrary {
    /// Indexes supported cells, retaining diagnostics for unsupported FFs.
    /// Duplicate cell names are rejected because emitted cell names must be
    /// unambiguous. Other unsupported cells do not poison valid alternatives.
    pub fn new(library: &Library) -> Result<Self, String> {
        let mut names = BTreeSet::new();
        let mut cells = Vec::new();
        let mut rejected = Vec::new();
        for cell in &library.cells {
            if !names.insert(cell.name.as_str()) {
                return Err(format!("duplicate Liberty cell name '{}'", cell.name));
            }
            if cell.sequential.is_empty() {
                continue;
            }
            match map_cell(cell, library) {
                Ok(binding) => cells.push(binding),
                Err(reason) => rejected.push(RejectedRegisterCell {
                    cell_name: cell.name.clone(),
                    reason,
                }),
            }
        }
        cells.sort_by(compare_mappings);
        rejected.sort_by(|a, b| a.cell_name.cmp(&b.cell_name));
        Ok(Self { cells, rejected })
    }

    /// Returns usable mappings, ordered by area, no required data inversion,
    /// cell name, output pin, data pin, and finally tie-off values.
    pub fn cells(&self) -> &[RegisterCellMapping] {
        &self.cells
    }

    /// Returns unsupported sequential cells in deterministic cell-name order.
    pub fn rejected_cells(&self) -> &[RejectedRegisterCell] {
        &self.rejected
    }

    /// Selects a supported FF; an invalid fixed-cell request never falls back.
    pub fn select(
        &self,
        selection: &RegisterCellSelection,
    ) -> Result<&RegisterCellMapping, String> {
        match selection {
            RegisterCellSelection::SmallestArea => self.cells.first().ok_or_else(|| {
                let reasons = self
                    .rejected
                    .iter()
                    .take(6)
                    .map(|cell| format!("{}: {}", cell.cell_name, cell.reason))
                    .collect::<Vec<_>>()
                    .join("; ");
                if reasons.is_empty() {
                    "Liberty has no supported positive-edge flip-flop".to_string()
                } else {
                    format!("Liberty has no supported positive-edge flip-flop: {reasons}")
                }
            }),
            RegisterCellSelection::FixedCell(name) => self
                .cells
                .iter()
                .find(|cell| cell.cell_name == *name)
                .ok_or_else(|| {
                    let reason = self
                        .rejected
                        .iter()
                        .find(|cell| cell.cell_name == *name)
                        .map(|cell| cell.reason.as_str())
                        .unwrap_or("not a known sequential cell");
                    format!("cannot use register cell '{name}': {reason}")
                }),
        }
    }

    /// Maps every logical register bit without modifying or simplifying the
    /// design.
    ///
    /// Requires legalized controls: synchronous reset/enable behavior must
    /// already be expressed in D. Power-up state remains unspecified.
    /// A state-free design needs no FF cell. Plans refer to these exact packed
    /// port IDs; cleanup or interface changes require a new plan.
    pub fn map_registers(
        &self,
        design: &SequentialGateFn,
        selection: &RegisterCellSelection,
    ) -> Result<RegisterMappingPlan, String> {
        design.validate_legalized()?;
        let mut bits = Vec::new();
        for register in &design.registers {
            let width = design.transition.inputs[register.q.index()].get_bit_count();
            for bit_index in 0..width {
                bits.push(RegisterBitMapping {
                    register_name: register.name.clone(),
                    bit_index,
                    q: register.q,
                    d: register.d,
                });
            }
        }
        let cell = if bits.is_empty() {
            None
        } else {
            Some(self.select(selection)?.clone())
        };
        Ok(RegisterMappingPlan {
            clock: design.clock.clone(),
            cell,
            bits,
        })
    }
}

/// Stable selection independent of Liberty cell/pin declaration order.
fn compare_mappings(a: &RegisterCellMapping, b: &RegisterCellMapping) -> Ordering {
    a.area
        .total_cmp(&b.area)
        .then_with(|| a.invert_data.cmp(&b.invert_data))
        .then_with(|| a.cell_name.cmp(&b.cell_name))
        .then_with(|| a.output_pin.cmp(&b.output_pin))
        .then_with(|| a.data_pin.cmp(&b.data_pin))
        .then_with(|| a.tied_inputs.cmp(&b.tied_inputs))
}

/// Finds the deterministic legal data/output/control assignment for one cell.
fn map_cell(cell: &Cell, library: &Library) -> Result<RegisterCellMapping, String> {
    cell.validate_functional_semantics()?;
    if cell.name.is_empty() {
        return Err("cell name is empty".into());
    }
    if cell.dont_use == Some(true) {
        return Err("cell is marked dont_use".into());
    }
    if !cell.area.is_finite() || cell.area < 0.0 {
        return Err("cell area must be finite and non-negative".into());
    }
    if cell.clock_gate.is_some() {
        return Err("clock-gating cells are not supported".into());
    }
    // A clock flag alone does not specify an active edge for physical mapping.
    if cell
        .sequential
        .first()
        .is_none_or(|seq| seq.clock_expr.is_empty())
    {
        return Err("FF clock expression is missing".into());
    }
    let spec = get_gv_eval_sequential_cell_spec(cell, library)?
        .ok_or_else(|| "cell is not sequential".to_string())?;
    if spec.clock.is_negated {
        return Err("negative-edge flip-flops are not supported".into());
    }
    let mut pins = BTreeMap::new();
    for pin in &cell.pins {
        let name = library.resolve_string(&pin.name);
        if name.is_empty() || pins.insert(name, pin).is_some() {
            return Err(format!("empty or duplicate pin name '{name}'"));
        }
        if pin.direction != PinDirection::Input as i32
            && pin.direction != PinDirection::Output as i32
        {
            return Err(format!("unsupported direction for pin '{name}'"));
        }
    }
    if pins
        .get(spec.clock.pin_name.as_str())
        .is_none_or(|pin| pin.direction != PinDirection::Input as i32)
    {
        return Err("clock expression must reference a physical input pin".into());
    }
    if pins.contains_key(spec.state_var.as_str())
        || spec
            .complementary_state_var
            .as_ref()
            .is_some_and(|name| name == &spec.state_var || pins.contains_key(name.as_str()))
    {
        return Err("FF state names must be distinct from each other and physical pins".into());
    }
    let inputs = pins
        .iter()
        .filter(|(name, pin)| {
            pin.direction == PinDirection::Input as i32 && **name != spec.clock.pin_name
        })
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    if inputs.is_empty() || inputs.len() > MAX_CONTROL_PINS + 1 {
        return Err(format!(
            "FF requires a data input and at most {MAX_CONTROL_PINS} tied control inputs"
        ));
    }
    let allowed = inputs
        .iter()
        .chain(std::iter::once(&spec.state_var))
        .chain(spec.complementary_state_var.iter())
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for name in spec.next_state.inputs() {
        if !allowed.contains(name.as_str()) {
            return Err(format!("next_state references unsupported signal '{name}'"));
        }
    }
    let outputs = pins
        .iter()
        .filter(|(_, pin)| pin.direction == PinDirection::Output as i32)
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    let mut state_outputs = Vec::new();
    for name in &outputs {
        let formula = library.resolve_string(&pins[name.as_str()].function);
        let Ok(term) = parse_formula(formula) else {
            // An unused output need not have a supported state-only function.
            continue;
        };
        if term.inputs().iter().any(|name| {
            name != &spec.state_var && Some(name) != spec.complementary_state_var.as_ref()
        }) {
            // This first mapper only connects a direct state polarity; other
            // output functions may remain unused.
            continue;
        }
        if let Some(inverted) = state_output_polarity(&term, &spec) {
            state_outputs.push((name, inverted));
        }
    }
    let mut best: Option<RegisterCellMapping> = None;
    for data_pin in &inputs {
        let controls = inputs
            .iter()
            .filter(|name| *name != data_pin)
            .collect::<Vec<_>>();
        for assignment in 0..(1usize << controls.len()) {
            let ties = controls
                .iter()
                .enumerate()
                .map(|(index, name)| ((*name).clone(), assignment & (1 << index) != 0))
                .collect::<BTreeMap<_, _>>();
            let Some(data_inverted) = data_polarity(&spec, data_pin, &ties) else {
                continue;
            };
            for (output_pin, output_inverted) in &state_outputs {
                let candidate = RegisterCellMapping {
                    cell_name: cell.name.clone(),
                    area: cell.area,
                    clock_pin: spec.clock.pin_name.clone(),
                    data_pin: data_pin.clone(),
                    output_pin: (*output_pin).clone(),
                    invert_data: data_inverted ^ output_inverted,
                    tied_inputs: ties.clone(),
                    unused_outputs: outputs
                        .iter()
                        .filter(|name| *name != *output_pin)
                        .cloned()
                        .collect(),
                };
                if best
                    .as_ref()
                    .is_none_or(|current| compare_mappings(&candidate, current).is_lt())
                {
                    best = Some(candidate);
                }
            }
        }
    }
    best.ok_or_else(|| {
        "no state output and constant-control assignment implements Q_next = D".into()
    })
}

/// Binds physical internal state and its optional complementary state variable.
fn bind_state(values: &mut HashMap<String, bool>, spec: &GvEvalSequentialCellSpec, state: bool) {
    values.insert(spec.state_var.clone(), state);
    if let Some(complement) = &spec.complementary_state_var {
        values.insert(complement.clone(), !state);
    }
}

/// Recognizes Q/QN behavior by function, never by an output pin's spelling.
fn state_output_polarity(term: &Term, spec: &GvEvalSequentialCellSpec) -> Option<bool> {
    let mut values = HashMap::new();
    bind_state(&mut values, spec, false);
    let at_zero = term.evaluate_partial(&values)?;
    bind_state(&mut values, spec, true);
    let at_one = term.evaluate_partial(&values)?;
    (at_zero != at_one).then_some(at_zero)
}

/// Exhaustively proves that tied controls reduce next_state to D or !D,
/// independently of the previous physical state.
fn data_polarity(
    spec: &GvEvalSequentialCellSpec,
    data: &str,
    ties: &BTreeMap<String, bool>,
) -> Option<bool> {
    let mut values = ties
        .iter()
        .map(|(name, value)| (name.clone(), *value))
        .collect::<HashMap<_, _>>();
    let mut polarity = None;
    for old_state in [false, true] {
        bind_state(&mut values, spec, old_state);
        values.insert(data.to_string(), false);
        let at_zero = spec.next_state.evaluate_partial(&values)?;
        values.insert(data.to_string(), true);
        let at_one = spec.next_state.evaluate_partial(&values)?;
        if at_zero == at_one || polarity.is_some_and(|previous| previous != at_zero) {
            return None;
        }
        polarity = Some(at_zero);
    }
    polarity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{
        AigBitVector, AigOperand, AigRef, GateBuilder, GateBuilderOptions, Output, RegisterBinding,
        RegisterControl, SynchronousReset, add_output_registers,
    };
    use crate::aig_sim::{
        gate_sim,
        sequential::{SequentialState, simulate},
    };
    use crate::liberty::model::{LibraryBuilder, Pin, library_from_proto, library_to_proto};
    use crate::liberty::parser::parse_liberty_files;
    use crate::liberty_proto::{Sequential, SequentialKind};
    use prost::Message;
    use rand::{SeedableRng, rngs::StdRng};
    use xlsynth_pir::IrBits;
    use xlsynth_pir::random_inputs::generate_mixed_irbits_with_rng;

    /// Builds an untimed cell with explicit complementary state and arbitrary
    /// pin names.
    fn add_ff(
        builder: &mut LibraryBuilder,
        name: &str,
        area: f64,
        next: &str,
        inputs: &[&str],
        outputs: &[(&str, &str)],
    ) -> anyhow::Result<()> {
        let mut pins = vec![Pin {
            name: builder.intern_string("CLK")?,
            direction: PinDirection::Input as i32,
            is_clocking_pin: true,
            ..Default::default()
        }];
        for name in inputs {
            pins.push(Pin {
                name: builder.intern_string(name)?,
                direction: PinDirection::Input as i32,
                ..Default::default()
            });
        }
        for (name, function) in outputs {
            pins.push(Pin {
                name: builder.intern_string(name)?,
                direction: PinDirection::Output as i32,
                function: builder.intern_string(function)?,
                ..Default::default()
            });
        }
        builder.cells.push(Cell {
            name: name.into(),
            area,
            pins,
            sequential: vec![Sequential {
                state_var: "IQ".into(),
                complementary_state_var: Some("IQN".into()),
                next_state: next.into(),
                clock_expr: "CLK".into(),
                kind: SequentialKind::Ff as i32,
                ..Default::default()
            }],
            ..Default::default()
        });
        Ok(())
    }

    /// Evaluates the original Liberty equations under the emitted binding for
    /// every physical old-state/logical-data pair, independent of the
    /// recognizer.
    fn assert_binding_semantics(library: &Library, binding: &RegisterCellMapping) {
        let cell = library
            .cells
            .iter()
            .find(|c| c.name == binding.cell_name)
            .unwrap();
        let seq = &cell.sequential[0];
        let next = parse_formula(&seq.next_state).unwrap();
        let output = cell
            .pins
            .iter()
            .find(|pin| library.resolve_string(&pin.name) == binding.output_pin)
            .unwrap();
        let output_function = parse_formula(library.resolve_string(&output.function)).unwrap();
        let bound_inputs = binding
            .tied_inputs
            .keys()
            .chain([&binding.clock_pin, &binding.data_pin])
            .cloned()
            .collect::<BTreeSet<_>>();
        let actual_inputs = cell
            .pins
            .iter()
            .filter(|p| p.direction == PinDirection::Input as i32)
            .map(|p| library.resolve_string(&p.name).to_string())
            .collect::<BTreeSet<_>>();
        assert_eq!(bound_inputs, actual_inputs);
        for old_state in [false, true] {
            for data in [false, true] {
                let mut values = binding
                    .tied_inputs
                    .iter()
                    .map(|(k, v)| (k.clone(), *v))
                    .collect::<HashMap<_, _>>();
                values.insert(binding.clock_pin.clone(), true);
                values.insert(binding.data_pin.clone(), data ^ binding.invert_data);
                values.insert(seq.state_var.clone(), old_state);
                if let Some(complement) = &seq.complementary_state_var {
                    values.insert(complement.clone(), !old_state);
                }
                let new_state = next.evaluate_partial(&values).unwrap();
                values.insert(seq.state_var.clone(), new_state);
                if let Some(complement) = &seq.complementary_state_var {
                    values.insert(complement.clone(), !new_state);
                }
                assert_eq!(
                    output_function.evaluate_partial(&values),
                    Some(data),
                    "{binding:?}"
                );
            }
        }
    }

    #[test]
    fn handles_all_data_and_output_phases_without_pin_name_assumptions() {
        for (next, data_inverted) in [("payload", false), ("!payload", true)] {
            for (output, output_inverted) in [
                ("IQ", false),
                ("IQN", true),
                ("!IQ", true),
                ("IQ ^ 1", true),
                ("!IQN", false),
            ] {
                let mut b = LibraryBuilder::new();
                add_ff(
                    &mut b,
                    "DFF",
                    1.0,
                    next,
                    &["payload"],
                    &[("result", output)],
                )
                .unwrap();
                let library = b.finish();
                let catalog = RegisterMappingLibrary::new(&library).unwrap();
                let cell = catalog
                    .select(&RegisterCellSelection::SmallestArea)
                    .unwrap();
                assert_eq!(cell.data_pin, "payload");
                assert_eq!(cell.output_pin, "result");
                assert_eq!(cell.invert_data, data_inverted ^ output_inverted);
                assert!(cell.tied_inputs.is_empty());
                assert_binding_semantics(&library, cell);
            }
        }
    }

    #[test]
    fn chooses_minimum_area_before_polarity_and_honors_fixed_override() {
        let mut b = LibraryBuilder::new();
        // Drive-strength naming is intentional: this test exercises area
        // selection.
        add_ff(&mut b, "DFF_X2", 2.0, "D", &["D"], &[("Q", "IQ")]).unwrap();
        add_ff(&mut b, "DFFN_X1", 1.0, "D", &["D"], &[("QN", "IQN")]).unwrap();
        let library = b.finish();
        let catalog = RegisterMappingLibrary::new(&library).unwrap();
        let default = catalog.select(&RegisterCellSelection::default()).unwrap();
        assert_eq!(default.cell_name, "DFFN_X1");
        assert!(default.invert_data);
        let fixed = catalog
            .select(&RegisterCellSelection::FixedCell("DFF_X2".into()))
            .unwrap();
        assert_eq!(fixed.cell_name, "DFF_X2");
        assert!(!fixed.invert_data);
        assert_eq!(
            catalog
                .select(&RegisterCellSelection::FixedCell("missing".into()))
                .unwrap_err(),
            "cannot use register cell 'missing': not a known sequential cell"
        );
    }

    #[test]
    fn deterministic_ties_and_dual_outputs_prefer_no_data_inversion() {
        let mut b = LibraryBuilder::new();
        add_ff(
            &mut b,
            "DFF_B",
            1.0,
            "!D",
            &["D"],
            &[("Q", "IQ"), ("QN", "IQN")],
        )
        .unwrap();
        add_ff(&mut b, "DFF_A", 1.0, "D", &["D"], &[("Q", "IQ")]).unwrap();
        add_ff(&mut b, "DFFN", 1.0, "D", &["D"], &[("QN", "IQN")]).unwrap();
        let mut library = b.finish();
        let expected = RegisterMappingLibrary::new(&library).unwrap();
        assert_eq!(expected.cells()[0].cell_name, "DFF_A");
        let dual = expected
            .select(&RegisterCellSelection::FixedCell("DFF_B".into()))
            .unwrap();
        assert_eq!(dual.output_pin, "QN");
        assert!(!dual.invert_data);
        assert_eq!(dual.unused_outputs, ["Q"]);
        for cell in expected.cells() {
            assert_binding_semantics(&library, cell);
        }
        library.cells.reverse();
        for cell in &mut library.cells {
            cell.pins.reverse();
        }
        let reordered = RegisterMappingLibrary::new(&library).unwrap();
        assert_eq!(expected.cells(), reordered.cells());
    }

    #[test]
    fn establishes_safe_scan_enable_and_unused_input_ties_by_truth_table() {
        let mut b = LibraryBuilder::new();
        add_ff(
            &mut b,
            "SDFF",
            1.0,
            "(!SE * D) + (SE * SI)",
            &["D", "SI", "SE", "UNUSED"],
            &[("Q", "IQ")],
        )
        .unwrap();
        add_ff(
            &mut b,
            "EDFF",
            1.0,
            "(!EN * IQ) + (EN * D)",
            &["D", "EN"],
            &[("Q", "IQ")],
        )
        .unwrap();
        let library = b.finish();
        let catalog = RegisterMappingLibrary::new(&library).unwrap();
        let scan = catalog
            .select(&RegisterCellSelection::FixedCell("SDFF".into()))
            .unwrap();
        assert_eq!(scan.data_pin, "D");
        assert_eq!(
            scan.tied_inputs,
            BTreeMap::from([
                ("SE".into(), false),
                ("SI".into(), false),
                ("UNUSED".into(), false)
            ])
        );
        let enabled = catalog
            .select(&RegisterCellSelection::FixedCell("EDFF".into()))
            .unwrap();
        assert_eq!(enabled.tied_inputs, BTreeMap::from([("EN".into(), true)]));
        for cell in catalog.cells() {
            assert_binding_semantics(&library, cell);
        }
    }

    /// Reads one row of a three-input Boolean next-state truth table.
    fn next_state_bit(table: u8, data: bool, control: bool, old_state: bool) -> bool {
        let index = usize::from(data) | (usize::from(control) << 1) | (usize::from(old_state) << 2);
        table & (1 << index) != 0
    }

    /// Emits all asserted minterms without using the mapping recognizer.
    fn next_state_formula(table: u8) -> String {
        let terms = (0..8)
            .filter(|index| table & (1 << index) != 0)
            .map(|index| {
                let terms = ["D", "EN", "IQ"]
                    .into_iter()
                    .enumerate()
                    .map(|(bit, name)| {
                        if index & (1 << bit) != 0 {
                            name.to_string()
                        } else {
                            format!("!{name}")
                        }
                    })
                    .collect::<Vec<_>>();
                format!("({})", terms.join(" * "))
            })
            .collect::<Vec<_>>();
        if terms.is_empty() {
            "0".into()
        } else {
            terms.join(" + ")
        }
    }

    #[test]
    fn exhaustively_checks_all_three_input_functions_and_output_polarities() {
        for table in 0..=u8::MAX {
            for output_inverted in [false, true] {
                let mut builder = LibraryBuilder::new();
                add_ff(
                    &mut builder,
                    "DFF",
                    1.0,
                    &next_state_formula(table),
                    &["D", "EN"],
                    &[("OUT", if output_inverted { "IQN" } else { "IQ" })],
                )
                .unwrap();
                let library = builder.finish();
                let catalog = RegisterMappingLibrary::new(&library).unwrap();
                // Either physical input may carry logical D. Compare against
                // the original truth bits, not the Liberty expression
                // evaluator.
                let supported = [false, true].into_iter().any(|use_control_as_data| {
                    [false, true].into_iter().any(|tie| {
                        [false, true].into_iter().any(|invert_data| {
                            [false, true].into_iter().all(|old_state| {
                                [false, true].into_iter().all(|data| {
                                    let physical_data = data ^ invert_data;
                                    let (d, en) = if use_control_as_data {
                                        (tie, physical_data)
                                    } else {
                                        (physical_data, tie)
                                    };
                                    next_state_bit(table, d, en, old_state) ^ output_inverted
                                        == data
                                })
                            })
                        })
                    })
                });
                assert_eq!(
                    !catalog.cells().is_empty(),
                    supported,
                    "table={table}, output_inverted={output_inverted}"
                );
                if let Some(binding) = catalog.cells().first() {
                    for old_state in [false, true] {
                        for data in [false, true] {
                            let pin_value = |name| {
                                if binding.data_pin == name {
                                    data ^ binding.invert_data
                                } else {
                                    binding.tied_inputs[name]
                                }
                            };
                            assert_eq!(
                                next_state_bit(table, pin_value("D"), pin_value("EN"), old_state)
                                    ^ output_inverted,
                                data,
                                "table={table}, binding={binding:?}"
                            );
                        }
                    }
                    assert_binding_semantics(&library, binding);
                }
            }
        }
    }

    #[test]
    fn supports_the_maximum_control_assignment_and_undeclared_complement() {
        let mut builder = LibraryBuilder::new();
        let controls = "A * B * C * E * F * G * H * J";
        add_ff(
            &mut builder,
            "DFF",
            1.0,
            &format!("(!({controls}) * IQ) + (({controls}) * D)"),
            &["D", "A", "B", "C", "E", "F", "G", "H", "J"],
            &[("Q", "!IQ")],
        )
        .unwrap();
        let mut library = builder.finish();
        library.cells[0].sequential[0].complementary_state_var = None;
        let catalog = RegisterMappingLibrary::new(&library).unwrap();
        let binding = catalog
            .select(&RegisterCellSelection::SmallestArea)
            .unwrap();
        assert_eq!(binding.data_pin, "D");
        assert_eq!(binding.tied_inputs.len(), MAX_CONTROL_PINS);
        assert!(binding.tied_inputs.values().all(|value| *value));
        assert!(binding.invert_data);
        assert_binding_semantics(&library, binding);
    }

    #[test]
    fn parser_to_mapping_supports_untimed_liberty() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            r#"library(test) {
            cell(DFFN) {
                area : 1;
                ff(IQ, IQN) { clocked_on : "CLK"; next_state : "D"; }
                pin(CLK) { direction : input; clock : true; }
                pin(D) { direction : input; }
                pin(QN) { direction : output; function : "IQN"; }
            }
        }"#,
        )
        .unwrap();
        let library = parse_liberty_files(&[file.path()]).unwrap();
        let catalog = RegisterMappingLibrary::new(&library).unwrap();
        let cell = catalog
            .select(&RegisterCellSelection::SmallestArea)
            .unwrap();
        assert!(cell.invert_data);
        assert_binding_semantics(&library, cell);
    }

    /// Tests the full ingestion path, including serialization of rejection
    /// metadata.
    fn import_roundtrip(text: &str) -> Library {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), text).unwrap();
        let wire = library_to_proto(parse_liberty_files(&[file.path()]).unwrap()).unwrap();
        library_from_proto(
            crate::liberty_proto::Library::decode(wire.encode_to_vec().as_slice()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn raw_imported_scan_flop_keeps_verified_tieoffs_after_proto_roundtrip() {
        let library = import_roundtrip(
            r#"library(test) { cell(SDFF) {
            area: 1;
            pin(D,SI,SE) { direction: input; }
            pin(CLK) { clock: true; }
            pin(QN) { direction: output; function: "IQN"; }
            ff(IQ,IQN) { clocked_on: "CLK"; next_state: "(D * SE') + (SI * SE)"; }
        } }"#,
        );
        let catalog = RegisterMappingLibrary::new(&library).unwrap();
        let mapping = catalog
            .select(&RegisterCellSelection::SmallestArea)
            .unwrap();
        assert_eq!(mapping.cell_name, "SDFF");
        assert_binding_semantics(&library, mapping);
    }

    #[test]
    fn raw_imported_unsupported_behavior_cannot_win_by_area_or_fixed_override() {
        for (extra_pin, seq_extra, output_extra) in [
            ("pin(OE) { direction: input; }", "", "three_state: \"!OE\";"),
            (
                "pin(CLK2) { direction: input; }",
                "clocked_on_also: \"CLK2\";",
                "",
            ),
            ("pin(CLK2) { clock: true; }", "", ""),
            ("", "", "x_function: \"D\";"),
        ] {
            let library = import_roundtrip(&format!(
                r#"library(test) {{
                cell(DFF) {{ area: 2;
                    pin(D) {{}} pin(CLK) {{ clock: true; }}
                    pin(Q) {{ direction: output; function: "IQ"; }}
                    ff(IQ,IQN) {{ clocked_on: "CLK"; next_state: "D"; }}
                }}
                cell(SPECIAL) {{ area: 0.1;
                    pin(D) {{}} pin(CLK) {{ clock: true; }}
                    pin(Q) {{ direction: output; function: "IQ"; {output_extra} }}
                    {extra_pin}
                    ff(IQ,IQN) {{ clocked_on: "CLK"; next_state: "D"; {seq_extra} }}
                }}
            }}"#
            ));
            let catalog = RegisterMappingLibrary::new(&library).unwrap();
            assert_eq!(
                catalog
                    .select(&RegisterCellSelection::SmallestArea)
                    .unwrap()
                    .cell_name,
                "DFF"
            );
            assert_eq!(catalog.rejected_cells().len(), 1);
            assert_eq!(catalog.rejected_cells()[0].cell_name, "SPECIAL");
            assert!(
                catalog
                    .select(&RegisterCellSelection::FixedCell("SPECIAL".into()))
                    .is_err()
            );
        }
    }

    #[test]
    fn imported_bank_is_preserved_but_not_mapped_as_one_flop() {
        let library = import_roundtrip(
            r#"library(test) { cell(DFF_BANK) {
            area: 1;
            bundle(D) { members(D1,D0); direction: input; }
            bundle(Q) { members(Q1,Q0); direction: output; function: "IQ"; }
            pin(CLK) { clock: true; }
            ff_bank(IQ,IQN,2) { clocked_on: "CLK"; next_state: "D"; }
        } }"#,
        );
        assert_eq!(library.cells[0].sequential.len(), 2);
        let catalog = RegisterMappingLibrary::new(&library).unwrap();
        assert!(catalog.cells().is_empty());
        assert_eq!(catalog.rejected_cells().len(), 1);
        assert!(
            catalog
                .select(&RegisterCellSelection::FixedCell("DFF_BANK".into()))
                .is_err()
        );
    }

    #[test]
    fn unsupported_cells_are_diagnosed_and_never_used_as_fixed_overrides() {
        let mut b = LibraryBuilder::new();
        for name in [
            "DFF",
            "DISABLED",
            "NEG",
            "LATCH",
            "ASYNC",
            "BANK",
            "BAD_AREA",
            "NO_CLOCK",
            "GATED",
            "MISSING_PIN",
            "BAD_NEXT",
            "BAD_STATE",
            "BAD_OUTPUT",
            "STATEFUL",
        ] {
            add_ff(&mut b, name, 1.0, "D", &["D"], &[("Q", "IQ")]).unwrap();
        }
        let mut library = b.finish();
        library.cells[1].dont_use = Some(true);
        library.cells[2].sequential[0].clock_expr = "!CLK".into();
        library.cells[3].sequential[0].kind = SequentialKind::Latch as i32;
        library.cells[4].sequential[0].clear_expr = "D".into();
        let second = library.cells[5].sequential[0].clone();
        library.cells[5].sequential.push(second);
        library.cells[6].area = f64::NAN;
        library.cells[7].sequential[0].clock_expr.clear();
        library.cells[8].sequential[0].clock_expr = "CLK * D".into();
        library.cells[9].pins.remove(0);
        library.cells[10].sequential[0].next_state = "missing".into();
        library.cells[11].sequential[0].state_var = "D".into();
        library.cells[12]
            .pins
            .retain(|pin| pin.direction != PinDirection::Output as i32);
        library.cells[13].sequential[0].next_state = "D ^ IQ".into();
        let catalog = RegisterMappingLibrary::new(&library).unwrap();
        assert_eq!(catalog.cells().len(), 1);
        assert_eq!(catalog.cells()[0].cell_name, "DFF");
        assert_eq!(catalog.rejected_cells().len(), library.cells.len() - 1);
        for rejected in catalog.rejected_cells() {
            assert_eq!(
                catalog
                    .select(&RegisterCellSelection::FixedCell(
                        rejected.cell_name.clone()
                    ))
                    .unwrap_err(),
                format!(
                    "cannot use register cell '{}': {}",
                    rejected.cell_name, rejected.reason
                )
            );
        }
        // A disabled cell is not made available just because it was requested.
        assert_eq!(
            catalog
                .select(&RegisterCellSelection::FixedCell("DISABLED".into()))
                .unwrap_err(),
            "cannot use register cell 'DISABLED': cell is marked dont_use"
        );
    }

    #[test]
    fn bounds_control_search_and_rejects_duplicate_or_invalid_pins() {
        let mut b = LibraryBuilder::new();
        add_ff(
            &mut b,
            "TOO_MANY",
            1.0,
            "D",
            &["D", "A", "B", "C", "E", "F", "G", "H", "I", "J"],
            &[("Q", "IQ")],
        )
        .unwrap();
        add_ff(&mut b, "DUP", 1.0, "D", &["D", "D"], &[("Q", "IQ")]).unwrap();
        add_ff(&mut b, "INVALID", 1.0, "D", &["D"], &[("Q", "IQ")]).unwrap();
        let mut library = b.finish();
        library.cells[2].pins[1].direction = PinDirection::Invalid as i32;
        let catalog = RegisterMappingLibrary::new(&library).unwrap();
        assert!(catalog.cells().is_empty());
        assert_eq!(catalog.rejected_cells().len(), 3);
        assert!(
            catalog
                .select(&RegisterCellSelection::SmallestArea)
                .is_err()
        );
        library.cells.push(library.cells[0].clone());
        assert_eq!(
            RegisterMappingLibrary::new(&library).unwrap_err(),
            "duplicate Liberty cell name 'TOO_MANY'"
        );
    }

    /// Builds a packed register from an identity function without assigning
    /// power-up state.
    fn packed_design(width: usize) -> Result<SequentialGateFn, String> {
        let mut b = GateBuilder::new("packed".into(), GateBuilderOptions::no_opt());
        let data = b.add_input("data".into(), width);
        b.add_output("out".into(), data);
        add_output_registers(
            &SequentialGateFn::from_gate_fn(b.build()),
            ClockPort { name: "clk".into() },
        )
    }

    #[test]
    fn maps_every_packed_bit_without_cleanup_or_mutation() {
        let mut b = LibraryBuilder::new();
        add_ff(&mut b, "DFF", 1.0, "D", &["D"], &[("Q", "IQ")]).unwrap();
        let catalog = RegisterMappingLibrary::new(&b.finish()).unwrap();
        for width in [0, 1, 8, 129] {
            let design = packed_design(width).unwrap();
            let original_text = crate::aig_serdes::g8r::emit_g8r(&design).unwrap();
            let plan = catalog
                .map_registers(&design, &RegisterCellSelection::default())
                .unwrap();
            assert_eq!(plan.clock, design.clock);
            assert_eq!(plan.cell.is_some(), width != 0);
            assert_eq!(plan.bits.len(), width);
            for (bit, mapping) in plan.bits.iter().enumerate() {
                assert_eq!(mapping.bit_index, bit);
                assert_eq!(mapping.register_name, design.registers[0].name);
                assert_eq!(mapping.q, design.registers[0].q);
                assert_eq!(mapping.d, design.registers[0].d);
            }
            assert_eq!(
                crate::aig_serdes::g8r::emit_g8r(&design).unwrap(),
                original_text
            );
        }
    }

    #[test]
    fn maps_shared_data_in_register_order_and_skips_zero_width_state() {
        let mut builder = GateBuilder::new("shared".into(), GateBuilderOptions::no_opt());
        let data = builder.add_input("data".into(), 3);
        let qa = builder.add_input("qa".into(), 3);
        let qb = builder.add_input("qb".into(), 3);
        let empty = builder.add_input("empty_q".into(), 0);
        builder.add_output("shared_d".into(), data);
        builder.add_output("a_out".into(), qa);
        builder.add_output("b_out".into(), qb);
        builder.add_output("empty_d".into(), empty);
        let design = SequentialGateFn::new(
            "shared".into(),
            builder.build(),
            vec![TransitionInputId::new(0)],
            vec![
                TransitionOutputId::new(0),
                TransitionOutputId::new(1),
                TransitionOutputId::new(2),
            ],
            Some(ClockPort { name: "clk".into() }),
            [("b", 2, 0), ("empty", 3, 3), ("a", 1, 0)]
                .into_iter()
                .map(|(name, q, d)| RegisterBinding {
                    name: name.into(),
                    q: TransitionInputId::new(q),
                    d: TransitionOutputId::new(d),
                    enable: None,
                    reset: None,
                })
                .collect(),
        )
        .unwrap();
        let mut library = LibraryBuilder::new();
        add_ff(&mut library, "DFF", 1.0, "D", &["D"], &[("Q", "IQ")]).unwrap();
        let catalog = RegisterMappingLibrary::new(&library.finish()).unwrap();
        let plan = catalog
            .map_registers(&design, &RegisterCellSelection::SmallestArea)
            .unwrap();
        let expected = [("b", 2), ("a", 1)]
            .into_iter()
            .flat_map(|(name, q)| {
                (0..3).map(move |bit_index| RegisterBitMapping {
                    register_name: name.into(),
                    bit_index,
                    q: TransitionInputId::new(q),
                    d: TransitionOutputId::new(0),
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(plan.bits, expected);
    }

    #[test]
    fn requires_reset_and_enable_legalization_before_mapping() {
        let mut b = LibraryBuilder::new();
        add_ff(&mut b, "DFF", 1.0, "D", &["D"], &[("Q", "IQ")]).unwrap();
        let catalog = RegisterMappingLibrary::new(&b.finish()).unwrap();
        for width in [0, 1, 129] {
            for active_low in [false, true] {
                for (enable, reset) in [(true, false), (false, true), (true, true)] {
                    let mut design = packed_design(width).unwrap();
                    let signal = TransitionOutputId::new(design.transition.outputs.len());
                    let bit = AigOperand {
                        node: AigRef { id: 0 },
                        negated: true,
                    };
                    design.transition.outputs.push(Output {
                        name: "control".into(),
                        bit_vector: bit.into(),
                    });
                    let control = RegisterControl { signal, active_low };
                    design.registers[0].enable = enable.then_some(control);
                    design.registers[0].reset = reset.then(|| SynchronousReset {
                        control,
                        value: IrBits::from_lsb_fn(width, |bit| bit % 3 == 0),
                    });
                    design.validate().unwrap();
                    assert_eq!(
                        catalog
                            .map_registers(&design, &RegisterCellSelection::default())
                            .unwrap_err(),
                        format!(
                            "register '{}' has explicit reset/enable controls; call legalize_register_controls() first",
                            design.registers[0].name
                        )
                    );
                    let legalized = design.legalize_register_controls().unwrap();
                    let plan = catalog
                        .map_registers(&legalized, &RegisterCellSelection::default())
                        .unwrap();
                    assert_eq!(plan.bits.len(), width);
                    assert_eq!(plan.cell.is_some(), width != 0);
                }
            }
        }
    }

    #[test]
    fn state_free_design_needs_no_ff_library_but_stateful_design_does() {
        let catalog = RegisterMappingLibrary::new(&Library::default()).unwrap();
        let mut b = GateBuilder::new("combo".into(), GateBuilderOptions::no_opt());
        let x = b.add_input("x".into(), 1);
        b.add_output("y".into(), x);
        let design = SequentialGateFn::from_gate_fn(b.build());
        let plan = catalog
            .map_registers(&design, &RegisterCellSelection::default())
            .unwrap();
        assert_eq!(
            plan,
            RegisterMappingPlan {
                clock: None,
                cell: None,
                bits: vec![]
            }
        );
        assert_eq!(
            catalog
                .map_registers(
                    &packed_design(1).unwrap(),
                    &RegisterCellSelection::default()
                )
                .unwrap_err(),
            "Liberty has no supported positive-edge flip-flop"
        );
    }

    #[test]
    fn leaves_dead_and_constant_registers_in_the_plan() {
        let mut b = LibraryBuilder::new();
        add_ff(&mut b, "DFF", 1.0, "D", &["D"], &[("Q", "IQ")]).unwrap();
        let catalog = RegisterMappingLibrary::new(&b.finish()).unwrap();
        let mut design = packed_design(8).unwrap();
        let zero = AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        };
        for output in &mut design.transition.outputs {
            output.bit_vector = AigBitVector::from_lsb_is_index_0(&[zero; 8]);
        }
        let plan = catalog
            .map_registers(&design, &RegisterCellSelection::SmallestArea)
            .unwrap();
        assert_eq!(plan.bits.len(), 8);
    }

    #[test]
    fn mapped_feedback_with_synchronous_controls_matches_sequential_simulation() {
        let mut b = GateBuilder::new("feedback".into(), GateBuilderOptions::opt());
        let data = b.add_input("data".into(), 2);
        let controls = b.add_input("controls".into(), 2);
        let q = b.add_input("state".into(), 3);
        let input = b.add_xor_binary(*data.get_lsb(0), *q.get_lsb(2));
        let enabled = b.add_mux2(*controls.get_lsb(0), *data.get_lsb(1), *q.get_lsb(1));
        let reset = b.add_and_binary(enabled, controls.get_lsb(1).negate());
        b.add_output("visible".into(), q.clone());
        b.add_output(
            "next".into(),
            AigBitVector::from_lsb_is_index_0(&[input, *q.get_lsb(0), reset]),
        );
        let design = SequentialGateFn::new(
            "feedback".into(),
            b.build(),
            vec![TransitionInputId::new(0), TransitionInputId::new(1)],
            vec![TransitionOutputId::new(0)],
            Some(ClockPort { name: "clk".into() }),
            vec![RegisterBinding {
                name: "state".into(),
                q: TransitionInputId::new(2),
                d: TransitionOutputId::new(1),
                enable: None,
                reset: None,
            }],
        )
        .unwrap();

        let mut b = LibraryBuilder::new();
        add_ff(&mut b, "DFF", 1.0, "D", &["D"], &[("Q", "IQ")]).unwrap();
        add_ff(&mut b, "DFFN", 1.0, "D", &["D"], &[("QN", "IQN")]).unwrap();
        add_ff(
            &mut b,
            "SDFF",
            1.0,
            "(!SE * !D) + (SE * SI)",
            &["D", "SE", "SI"],
            &[("QN", "IQN")],
        )
        .unwrap();
        let library = b.finish();
        let catalog = RegisterMappingLibrary::new(&library).unwrap();
        let mut rng = StdRng::seed_from_u64(20261010);
        let inputs = (0..128)
            .map(|_| generate_mixed_irbits_with_rng(&mut rng, &[2, 2]))
            .collect::<Vec<_>>();
        for cell in &library.cells {
            let plan = catalog
                .map_registers(
                    &design,
                    &RegisterCellSelection::FixedCell(cell.name.clone()),
                )
                .unwrap();
            let binding = plan.cell.as_ref().unwrap();
            let seq = &cell.sequential[0];
            let next = parse_formula(&seq.next_state).unwrap();
            let output_pin = cell
                .pins
                .iter()
                .find(|p| library.resolve_string(&p.name) == binding.output_pin)
                .unwrap();
            let output = parse_formula(library.resolve_string(&output_pin.function)).unwrap();
            let cell_inputs = |state: bool, data: bool| {
                let mut values = binding
                    .tied_inputs
                    .iter()
                    .map(|(k, v)| (k.clone(), *v))
                    .collect::<HashMap<_, _>>();
                values.insert(seq.state_var.clone(), state);
                values.insert(seq.complementary_state_var.clone().unwrap(), !state);
                values.insert(binding.clock_pin.clone(), true);
                values.insert(binding.data_pin.clone(), data ^ binding.invert_data);
                values
            };
            for initial in 0..8 {
                let mut physical = (0..3)
                    .map(|bit| initial & (1 << bit) != 0)
                    .collect::<Vec<_>>();
                let initial_q = physical
                    .iter()
                    .map(|&state| output.evaluate_partial(&cell_inputs(state, false)).unwrap())
                    .collect::<Vec<_>>();
                let reference = simulate(
                    &design,
                    &inputs,
                    SequentialState::from_register_values(
                        &design,
                        vec![IrBits::from_lsb_is_0(&initial_q)],
                    )
                    .unwrap(),
                )
                .unwrap();
                for (cycle, external) in inputs.iter().enumerate() {
                    let mut port_bits = design
                        .transition
                        .inputs
                        .iter()
                        .map(|p| vec![false; p.get_bit_count()])
                        .collect::<Vec<_>>();
                    for (id, values) in design.inputs.iter().zip(external) {
                        port_bits[id.index()] = (0..values.get_bit_count())
                            .map(|bit| values.get_bit(bit).unwrap())
                            .collect();
                    }
                    for (index, bit) in plan.bits.iter().enumerate() {
                        port_bits[bit.q.index()][bit.bit_index] = output
                            .evaluate_partial(&cell_inputs(physical[index], false))
                            .unwrap();
                    }
                    let ports = port_bits
                        .iter()
                        .map(|bits| IrBits::from_lsb_is_0(bits))
                        .collect::<Vec<_>>();
                    let transition =
                        gate_sim::eval(&design.transition, &ports, gate_sim::Collect::None);
                    let visible = design
                        .outputs
                        .iter()
                        .map(|id| transition.outputs[id.index()].clone())
                        .collect::<Vec<_>>();
                    assert_eq!(
                        visible,
                        reference.external_outputs()[cycle],
                        "{} cycle {cycle}",
                        cell.name
                    );
                    physical = plan
                        .bits
                        .iter()
                        .enumerate()
                        .map(|(index, bit)| {
                            let d = transition.outputs[bit.d.index()]
                                .get_bit(bit.bit_index)
                                .unwrap();
                            next.evaluate_partial(&cell_inputs(physical[index], d))
                                .unwrap()
                        })
                        .collect();
                }
                let final_q = physical
                    .iter()
                    .map(|&state| output.evaluate_partial(&cell_inputs(state, false)).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    IrBits::from_lsb_is_0(&final_q),
                    reference.final_state().values()[0]
                );
            }
        }
    }
}
