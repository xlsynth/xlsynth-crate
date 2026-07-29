// SPDX-License-Identifier: Apache-2.0

//! Functionally classified combinational Liberty cells for netlist transforms.

use crate::liberty::cell_formula::parse_formula;
use crate::liberty_model::{Cell, Library, PinDirection};
use crate::liberty_proto::TimingTableKind;
use crate::netlist::sta::{
    CombinationalOutputLoad, effective_input_capacitance_for_mapping,
    validate_output_pin_for_basic_sta,
};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, HashMap};

const MAX_CATALOG_INPUTS: usize = 6;

/// Pin-compatible Boolean identity shared by a standard-cell size family.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct CellFamilyKey {
    pub input_names: Vec<String>,
    pub output_name: String,
    pub truth: u64,
}

/// One usable, timing-complete single-output combinational cell.
#[derive(Clone, Debug)]
pub(crate) struct CatalogCell {
    pub cell_index: usize,
    pub name: String,
    pub family: CellFamilyKey,
    pub input_pin_indices: Vec<usize>,
    pub input_capacitances: Vec<CombinationalOutputLoad>,
    pub output_pin_index: usize,
    pub output_max_capacitance: Option<f64>,
    pub area: f64,
    pub nominal_delay: f64,
}

impl CatalogCell {
    /// Returns true for an actual noninverting one-input Liberty buffer.
    pub(crate) fn is_buffer(&self) -> bool {
        self.family.input_names.len() == 1 && self.family.truth == 0b10
    }
}

/// Reusable deterministic catalog of timing-complete combinational cells.
pub(crate) struct CellCatalog {
    cells: Vec<CatalogCell>,
    by_name: HashMap<String, usize>,
    families: BTreeMap<CellFamilyKey, Vec<usize>>,
    buffers: Vec<usize>,
}

impl CellCatalog {
    /// Classifies usable Liberty cells by their exact named-pin truth tables.
    pub(crate) fn new(library: &Library) -> Result<Self> {
        let mut cells = Vec::new();
        let mut by_name = HashMap::new();
        let mut families: BTreeMap<CellFamilyKey, Vec<usize>> = BTreeMap::new();
        let mut buffers = Vec::new();

        for (cell_index, cell) in library.cells.iter().enumerate() {
            let Some(catalog_cell) = classify_cell(library, cell_index, cell)? else {
                continue;
            };
            let index = cells.len();
            if by_name.insert(catalog_cell.name.clone(), index).is_some() {
                return Err(anyhow!(
                    "Liberty defines usable cell '{}' more than once",
                    catalog_cell.name
                ));
            }
            if catalog_cell.is_buffer() {
                buffers.push(index);
            }
            families
                .entry(catalog_cell.family.clone())
                .or_default()
                .push(index);
            cells.push(catalog_cell);
        }

        for members in families.values_mut() {
            members.sort_by(|lhs, rhs| catalog_cell_order(&cells[*lhs], &cells[*rhs]));
        }
        buffers.sort_by(|lhs, rhs| catalog_cell_order(&cells[*lhs], &cells[*rhs]));

        Ok(Self {
            cells,
            by_name,
            families,
            buffers,
        })
    }

    /// Returns the classified representation of one named Liberty cell.
    pub(crate) fn by_name(&self, name: &str) -> Option<&CatalogCell> {
        self.by_name.get(name).map(|index| &self.cells[*index])
    }

    /// Returns exact same-interface, same-function size alternatives.
    pub(crate) fn family(&self, cell: &CatalogCell) -> impl Iterator<Item = &CatalogCell> {
        self.families
            .get(&cell.family)
            .into_iter()
            .flatten()
            .map(|index| &self.cells[*index])
    }

    /// Returns noninverting buffers ordered by increasing area and delay.
    pub(crate) fn buffers(&self) -> impl Iterator<Item = &CatalogCell> {
        self.buffers.iter().map(|index| &self.cells[*index])
    }
}

/// Orders cell choices independently of Liberty source ordering.
fn catalog_cell_order(lhs: &CatalogCell, rhs: &CatalogCell) -> std::cmp::Ordering {
    lhs.area
        .total_cmp(&rhs.area)
        .then_with(|| lhs.nominal_delay.total_cmp(&rhs.nominal_delay))
        .then_with(|| lhs.name.cmp(&rhs.name))
}

/// Extracts a timing-complete native combinational cell without name
/// heuristics.
fn classify_cell(library: &Library, cell_index: usize, cell: &Cell) -> Result<Option<CatalogCell>> {
    if cell.dont_use == Some(true)
        || !cell.sequential.is_empty()
        || cell.clock_gate.is_some()
        || !cell.area.is_finite()
        || cell.area < 0.0
    {
        return Ok(None);
    }

    let mut input_pins: Vec<(String, usize)> = cell
        .pins
        .iter()
        .enumerate()
        .filter(|(_, pin)| pin.direction == PinDirection::Input as i32)
        .map(|(index, pin)| (library.resolve_string(&pin.name).to_string(), index))
        .collect();
    if input_pins.len() > MAX_CATALOG_INPUTS
        || input_pins
            .iter()
            .any(|(_, index)| cell.pins[*index].is_clocking_pin)
    {
        return Ok(None);
    }
    input_pins.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    if input_pins.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Ok(None);
    }

    let outputs: Vec<usize> = cell
        .pins
        .iter()
        .enumerate()
        .filter(|(_, pin)| pin.direction == PinDirection::Output as i32)
        .map(|(index, _)| index)
        .collect();
    if outputs.len() != 1 {
        return Ok(None);
    }
    let output_pin_index = outputs[0];
    let output_pin = &cell.pins[output_pin_index];
    let function = library.resolve_string(&output_pin.function);
    if function.is_empty() {
        return Ok(None);
    }
    let Ok(term) = parse_formula(function) else {
        return Ok(None);
    };

    let input_names: Vec<String> = input_pins.iter().map(|(name, _)| name.clone()).collect();
    let mut formula_inputs = term.inputs();
    formula_inputs.sort();
    formula_inputs.dedup();
    if formula_inputs != input_names {
        return Ok(None);
    }
    if validate_output_pin_for_basic_sta(
        library,
        cell.name.as_str(),
        output_pin,
        input_names.as_slice(),
    )
    .is_err()
    {
        return Ok(None);
    }

    let mut truth = 0_u64;
    let mut assignment_values = HashMap::with_capacity(input_names.len());
    for assignment in 0..(1_usize << input_names.len()) {
        assignment_values.clear();
        for (index, name) in input_names.iter().enumerate() {
            assignment_values.insert(name.clone(), (assignment >> index) & 1 != 0);
        }
        let value = term.evaluate_partial(&assignment_values).ok_or_else(|| {
            anyhow!(
                "Liberty cell '{}' has an unevaluable output function",
                cell.name
            )
        })?;
        if value {
            truth |= 1_u64 << assignment;
        }
    }

    let input_capacitances = input_pins
        .iter()
        .map(|(name, index)| {
            effective_input_capacitance_for_mapping(
                &cell.pins[*index],
                format!("standard-cell input '{}.{}'", cell.name, name).as_str(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let nominal_delay = output_pin
        .timing_arcs
        .iter()
        .flat_map(|arc| &arc.tables)
        .filter(|table| {
            matches!(
                table.kind,
                TimingTableKind::CellRise | TimingTableKind::CellFall
            )
        })
        .filter_map(|table| library.timing_table_values(table).first().copied())
        .map(f64::from)
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::max);

    Ok(Some(CatalogCell {
        cell_index,
        name: cell.name.clone(),
        family: CellFamilyKey {
            input_names,
            output_name: library.resolve_string(&output_pin.name).to_string(),
            truth,
        },
        input_pin_indices: input_pins.into_iter().map(|(_, index)| index).collect(),
        input_capacitances,
        output_pin_index,
        output_max_capacitance: output_pin.max_capacitance,
        area: cell.area,
        nominal_delay,
    }))
}

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::liberty_model::{Cell, Library, LibraryBuilder, Pin, PinDirection, TimingTable};
    use crate::liberty_proto::TimingTableKind;
    use crate::netlist::parse::{Net, NetlistModule, Parser, TokenScanner};
    use std::io::Cursor;
    use string_interner::symbol::SymbolU32;
    use string_interner::{StringInterner, backend::StringBackend};

    /// Builds a small timing-complete library with equivalent drive variants.
    pub(crate) fn sizing_library() -> Library {
        let mut builder = LibraryBuilder::new();
        let buffer = timed_cell(&mut builder, "BUF", &["A"], "A", 1.0, 4.0, 0.1, 0.8);
        let fast_buffer = timed_cell(&mut builder, "BUF_FAST", &["A"], "A", 2.0, 1.0, 0.2, 1.6);
        let and = timed_cell(
            &mut builder,
            "AND2",
            &["A", "B"],
            "A * B",
            1.0,
            5.0,
            0.1,
            0.8,
        );
        let fast_and = timed_cell(
            &mut builder,
            "AND2_FAST",
            &["A", "B"],
            "A * B",
            3.0,
            1.0,
            0.2,
            1.6,
        );
        builder.cells = vec![buffer, fast_buffer, and, fast_and];
        builder.finish()
    }

    /// Parses a single small scalar module for netlist-transform tests.
    pub(crate) fn parse_module(
        source: &str,
    ) -> (
        NetlistModule,
        Vec<Net>,
        StringInterner<StringBackend<SymbolU32>>,
    ) {
        let bytes = source.as_bytes().to_vec();
        let lines: Vec<String> = source.lines().map(ToString::to_string).collect();
        let lookup = move |line: u32| lines.get((line - 1) as usize).cloned();
        let scanner = TokenScanner::with_line_lookup(Cursor::new(bytes), Box::new(lookup));
        let mut parser = Parser::new(scanner);
        let mut modules = parser.parse_file().expect("test netlist should parse");
        assert_eq!(modules.len(), 1);
        (modules.remove(0), parser.nets, parser.interner)
    }

    /// Creates one combinational cell with complete scalar rise/fall NLDM data.
    fn timed_cell(
        builder: &mut LibraryBuilder,
        name: &str,
        inputs: &[&str],
        function: &str,
        area: f64,
        delay: f64,
        input_capacitance: f64,
        max_capacitance: f64,
    ) -> Cell {
        let mut pins: Vec<Pin> = inputs
            .iter()
            .map(|input| Pin {
                direction: PinDirection::Input as i32,
                name: builder.intern_string(input).unwrap(),
                capacitance: Some(input_capacitance),
                ..Pin::default()
            })
            .collect();
        let arcs = inputs
            .iter()
            .map(|input| {
                let tables: Vec<TimingTable> = [
                    (TimingTableKind::CellRise, delay),
                    (TimingTableKind::CellFall, delay),
                    (TimingTableKind::RiseTransition, 0.1),
                    (TimingTableKind::FallTransition, 0.1),
                ]
                .into_iter()
                .map(|(kind, value)| {
                    builder
                        .add_timing_table_f64(
                            kind,
                            0,
                            vec![],
                            vec![],
                            vec![],
                            vec![value],
                            vec![],
                            "",
                        )
                        .unwrap()
                })
                .collect();
                builder
                    .add_timing_arc(input, "positive_unate", "combinational", "", tables)
                    .unwrap()
            })
            .collect();
        pins.push(Pin {
            direction: PinDirection::Output as i32,
            name: builder.intern_string("Y").unwrap(),
            function: builder.intern_string(function).unwrap(),
            max_capacitance: Some(max_capacitance),
            timing_arcs: arcs,
            ..Pin::default()
        });
        Cell {
            name: name.to_string(),
            pins,
            area,
            ..Cell::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CellCatalog;
    use super::test_utils::sizing_library;

    #[test]
    fn groups_cells_by_function_instead_of_name() {
        let library = sizing_library();
        let catalog = CellCatalog::new(&library).unwrap();
        let and = catalog.by_name("AND2").unwrap();
        let names: Vec<&str> = catalog
            .family(and)
            .map(|candidate| candidate.name.as_str())
            .collect();
        assert_eq!(names, ["AND2", "AND2_FAST"]);
    }

    #[test]
    fn discovers_all_actual_identity_buffer_strengths() {
        let library = sizing_library();
        let catalog = CellCatalog::new(&library).unwrap();
        let names: Vec<&str> = catalog
            .buffers()
            .map(|candidate| candidate.name.as_str())
            .collect();
        assert_eq!(names, ["BUF", "BUF_FAST"]);
    }
}
