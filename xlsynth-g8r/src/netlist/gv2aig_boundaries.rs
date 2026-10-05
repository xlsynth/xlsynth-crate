// SPDX-License-Identifier: Apache-2.0

//! Projects gate-level netlists to AIGs while collapsing pipeline flops.

use std::collections::{HashMap, HashSet};

use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

use crate::aig::{AigBitVector, AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::liberty::cell_formula::{EmitContext, Term};
use crate::liberty_model::{Library, PinDirection};
use crate::netlist::gatefn_from_netlist::build_cell_formula_map;
use crate::netlist::hierarchy::ElaboratedNetlist;
use crate::netlist::normalized::{BitExpr, BitIndex, BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{Net, PortDirection};

mod feedback;

/// Extracts all top-module inputs and outputs, optionally collapsing reached
/// flops. Recognized direct hold-feedback flops use zero for their prior Q
/// value when load-enable feedback collapsing is enabled.
pub(super) fn extract_gatefn_all_top_ports(
    elaborated: &ElaboratedNetlist,
    liberty: Option<&Library>,
    collapse_sequential: bool,
    collapse_load_enable_feedback: bool,
) -> Result<GateFn, String> {
    BoundaryExtractor::new(
        elaborated,
        liberty,
        collapse_sequential,
        collapse_load_enable_feedback,
    )?
    .extract()
}

#[derive(Debug, Clone)]
enum BitDriver {
    ModuleInput {
        port_name: String,
    },
    Assign {
        assign_index: usize,
        rhs_bit_index: usize,
    },
    CellOutput {
        instance_index: usize,
        connection_index: usize,
    },
}

#[derive(Debug, Clone)]
enum PreparedDriver {
    Assign {
        assign_index: usize,
        rhs_bit_index: usize,
    },
    CellOutput {
        instance_index: usize,
        formula_key: (String, String),
        inputs: Vec<(String, BitSource)>,
    },
}

struct BoundaryExtractor<'a> {
    normalized: NormalizedNetlistModule<'a>,
    nets: &'a [Net],
    interner: &'a StringInterner<StringBackend<SymbolU32>>,
    liberty: Option<&'a Library>,
    collapse_sequential: bool,
    collapse_load_enable_feedback: bool,
    cell_index_by_name: HashMap<String, usize>,
    drivers: Vec<Vec<BitDriver>>,
    prepared: Vec<Option<PreparedDriver>>,
    values: Vec<Option<AigOperand>>,
    source_owners: Vec<Option<String>>,
    visit_state: Vec<u8>,
    prepared_cell_types: HashSet<String>,
    cell_formulas: HashMap<(String, String), (Term, String)>,
    builder: GateBuilder,
}

impl<'a> BoundaryExtractor<'a> {
    /// Indexes signal drivers and Liberty cell directions for the elaborated
    /// netlist.
    fn new(
        elaborated: &'a ElaboratedNetlist,
        liberty: Option<&'a Library>,
        collapse_sequential: bool,
        collapse_load_enable_feedback: bool,
    ) -> Result<Self, String> {
        let normalized = NormalizedNetlistModule::new(
            &elaborated.module,
            &elaborated.nets,
            &elaborated.interner,
        )
        .map_err(|error| error.to_string())?;
        let module_name = elaborated
            .interner
            .resolve(elaborated.module.name)
            .ok_or_else(|| "could not resolve selected module name".to_string())?;
        if normalized
            .ports
            .iter()
            .any(|port| port.direction == PortDirection::Inout)
        {
            return Err(format!(
                "module '{}' has an inout port; gv2aig supports only input and output module ports",
                module_name
            ));
        }
        let mut cell_index_by_name = HashMap::new();
        if let Some(library) = liberty {
            for (index, cell) in library.cells.iter().enumerate() {
                if cell_index_by_name
                    .insert(cell.name.clone(), index)
                    .is_some()
                {
                    return Err(format!("Liberty contains duplicate cell '{}'", cell.name));
                }
            }
        }
        let mut instance_names = HashSet::new();
        let mut drivers = vec![Vec::new(); normalized.bit_count()];
        for port in &normalized.ports {
            if port.direction == PortDirection::Input {
                let port_name = elaborated.interner.resolve(port.name).unwrap().to_string();
                for &bit in &port.bits {
                    drivers[bit].push(BitDriver::ModuleInput {
                        port_name: port_name.clone(),
                    });
                }
            }
        }
        for (assign_index, assign) in normalized.assigns.iter().enumerate() {
            for (rhs_bit_index, &bit) in assign.lhs_bits.iter().enumerate() {
                drivers[bit].push(BitDriver::Assign {
                    assign_index,
                    rhs_bit_index,
                });
            }
        }
        for (instance_index, instance) in normalized.instances.iter().enumerate() {
            let instance_name = elaborated
                .interner
                .resolve(instance.instance_name)
                .unwrap()
                .to_string();
            if !instance_names.insert(instance_name.clone()) {
                return Err(format!(
                    "duplicate flattened instance name '{}'",
                    instance_name
                ));
            }
            let type_name = elaborated.interner.resolve(instance.type_name).unwrap();
            let library = liberty.ok_or_else(|| {
                format!(
                    "boundary extraction requires Liberty for cell '{}' instance '{}'",
                    type_name, instance_name
                )
            })?;
            let cell_index = cell_index_by_name.get(type_name).ok_or_else(|| {
                format!(
                    "cell '{}' instance '{}' not found in Liberty",
                    type_name, instance_name
                )
            })?;
            let cell = &library.cells[*cell_index];
            for (connection_index, connection) in instance.connections.iter().enumerate() {
                let pin_name = elaborated.interner.resolve(connection.port).unwrap();
                let Some(pin) = cell
                    .pins
                    .iter()
                    .find(|pin| library.resolve_string(&pin.name) == pin_name)
                else {
                    return Err(format!(
                        "cell '{}' instance '{}' has unknown pin '{}'",
                        type_name, instance_name, pin_name
                    ));
                };
                if pin.direction != PinDirection::Output as i32 || connection.bits.is_empty() {
                    continue;
                }
                let [BitSource::Bit(bit)] = connection.bits.as_slice() else {
                    return Err(format!(
                        "cell '{}' instance '{}' output pin '{}' must connect to one net bit",
                        type_name, instance_name, pin_name
                    ));
                };
                drivers[*bit].push(BitDriver::CellOutput {
                    instance_index,
                    connection_index,
                });
            }
        }
        let bit_count = normalized.bit_count();
        Ok(Self {
            normalized,
            nets: &elaborated.nets,
            interner: &elaborated.interner,
            liberty,
            collapse_sequential,
            collapse_load_enable_feedback,
            cell_index_by_name,
            drivers,
            prepared: vec![None; bit_count],
            values: vec![None; bit_count],
            source_owners: vec![None; bit_count],
            visit_state: vec![0; bit_count],
            prepared_cell_types: HashSet::new(),
            cell_formulas: HashMap::new(),
            builder: GateBuilder::new(module_name.to_string(), GateBuilderOptions::no_opt()),
        })
    }

    /// Creates the top-module interface and lowers logic reached from its
    /// outputs.
    fn extract(mut self) -> Result<GateFn, String> {
        let inputs = self
            .normalized
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Input)
            .map(|port| {
                (
                    self.interner.resolve(port.name).unwrap().to_string(),
                    port.bits.clone(),
                )
            })
            .collect::<Vec<_>>();
        let outputs = self
            .normalized
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
            .map(|port| {
                (
                    self.interner.resolve(port.name).unwrap().to_string(),
                    port.bits.clone(),
                )
            })
            .collect::<Vec<_>>();
        if outputs.is_empty() {
            return Err("gv2aig requires at least one top-module output".to_string());
        }
        validate_names(inputs.iter().map(|(name, _)| name.as_str()), "source")?;
        validate_names(outputs.iter().map(|(name, _)| name.as_str()), "sink")?;
        for (name, bits) in inputs {
            let input = self.builder.add_input(name.clone(), bits.len());
            for (offset, &bit) in bits.iter().enumerate() {
                let owner = format!("{}[{}]", name, offset);
                if let Some(previous) = &self.source_owners[bit] {
                    return Err(format!(
                        "source '{}' selects net bit '{}' already selected by source '{}'",
                        owner,
                        self.render_bit(bit),
                        previous
                    ));
                }
                self.source_owners[bit] = Some(owner);
                self.values[bit] = Some(*input.get_lsb(offset));
            }
        }
        for (name, bits) in outputs {
            let mut output_bits = Vec::with_capacity(bits.len());
            for bit in bits {
                output_bits.push(
                    self.resolve_bit(bit).map_err(|error| {
                        format!("while extracting output '{}': {}", name, error)
                    })?,
                );
            }
            self.builder
                .add_output(name, AigBitVector::from_lsb_is_index_0(&output_bits));
        }
        Ok(self.builder.build())
    }

    /// Resolves one sink bit with iterative DFS so long netlist chains do not
    /// recurse.
    fn resolve_bit(&mut self, root: BitIndex) -> Result<AigOperand, String> {
        let mut stack = vec![(root, false)];
        while let Some((bit, exiting)) = stack.pop() {
            if self.values[bit].is_some() {
                continue;
            }
            if exiting {
                self.emit_prepared(bit)?;
                self.visit_state[bit] = 2;
                continue;
            }
            if self.visit_state[bit] == 1 {
                return Err(format!(
                    "dependency cycle reaches net bit '{}' in the selected cone",
                    self.render_bit(bit)
                ));
            }
            self.visit_state[bit] = 1;
            let dependencies = self.prepare_bit(bit)?;
            stack.push((bit, true));
            for dependency in dependencies.into_iter().rev() {
                if self.values[dependency].is_none() {
                    stack.push((dependency, false));
                }
            }
        }
        self.values[root].ok_or_else(|| {
            format!(
                "internal error: net bit '{}' was not resolved",
                self.render_bit(root)
            )
        })
    }

    /// Prepares the driver of one reached bit and returns its net-bit
    /// dependencies.
    fn prepare_bit(&mut self, bit: BitIndex) -> Result<Vec<BitIndex>, String> {
        let driver = match self.drivers[bit].as_slice() {
            [] => return Err(format!("net bit '{}' has no driver", self.render_bit(bit))),
            [driver] => driver.clone(),
            drivers => {
                return Err(format!(
                    "net bit '{}' has {} drivers in the selected cone",
                    self.render_bit(bit),
                    drivers.len()
                ));
            }
        };
        let prepared = match driver {
            BitDriver::ModuleInput { port_name } => {
                return Err(format!(
                    "reached unselected top module input '{}' at net bit '{}'; add it as a source",
                    port_name,
                    self.render_bit(bit)
                ));
            }
            BitDriver::Assign {
                assign_index,
                rhs_bit_index,
            } => PreparedDriver::Assign {
                assign_index,
                rhs_bit_index,
            },
            BitDriver::CellOutput {
                instance_index,
                connection_index,
            } => self.prepare_cell_output(instance_index, connection_index)?,
        };
        let mut dependencies = Vec::new();
        match &prepared {
            PreparedDriver::Assign {
                assign_index,
                rhs_bit_index,
            } => {
                self.normalized.assigns[*assign_index].rhs_bits[*rhs_bit_index]
                    .collect_source_bits(&mut dependencies);
            }
            PreparedDriver::CellOutput { inputs, .. } => {
                dependencies.extend(inputs.iter().filter_map(|(_, source)| match source {
                    BitSource::Bit(bit) => Some(*bit),
                    BitSource::Literal(_) | BitSource::Unknown => None,
                }));
            }
        }
        dependencies.sort_unstable();
        dependencies.dedup();
        self.prepared[bit] = Some(prepared);
        Ok(dependencies)
    }

    /// Resolves only the Liberty inputs used by one reached output formula.
    fn prepare_cell_output(
        &mut self,
        instance_index: usize,
        connection_index: usize,
    ) -> Result<PreparedDriver, String> {
        let instance = &self.normalized.instances[instance_index];
        let type_name = self
            .interner
            .resolve(instance.type_name)
            .unwrap()
            .to_string();
        let instance_name = self
            .interner
            .resolve(instance.instance_name)
            .unwrap()
            .to_string();
        let pin_name = self
            .interner
            .resolve(instance.connections[connection_index].port)
            .unwrap()
            .to_string();
        let library = self.liberty.expect("cell drivers require Liberty");
        let cell = &library.cells[self.cell_index_by_name[&type_name]];
        if !self.collapse_sequential && !cell.sequential.is_empty() {
            return Err(format!(
                "reached sequential cell '{}' instance '{}' output '{}'; enable collapse_sequential",
                type_name, instance_name, pin_name
            ));
        }
        if self.prepared_cell_types.insert(type_name.clone()) {
            let used_cells = HashSet::from([type_name.clone()]);
            let empty = HashSet::new();
            self.cell_formulas.extend(build_cell_formula_map(
                library,
                self.collapse_sequential,
                &used_cells,
                &empty,
                &empty,
            )?);
        }
        let formula_key = (type_name.clone(), pin_name.clone());
        let (formula, _) = self.cell_formulas.get(&formula_key).ok_or_else(|| {
            format!(
                "cell '{}' instance '{}' output '{}' has no supported Liberty function",
                type_name, instance_name, pin_name
            )
        })?;
        let mut input_names = formula.inputs();
        input_names.sort();
        input_names.dedup();
        let mut inputs = Vec::with_capacity(input_names.len());
        for input_name in input_names {
            let connection = instance
                .connections
                .iter()
                .find(|connection| {
                    self.interner.resolve(connection.port) == Some(input_name.as_str())
                })
                .ok_or_else(|| {
                    format!(
                        "cell '{}' instance '{}' output '{}' requires unconnected pin '{}'",
                        type_name, instance_name, pin_name, input_name
                    )
                })?;
            let [source] = connection.bits.as_slice() else {
                return Err(format!(
                    "cell '{}' instance '{}' input '{}' must have one connected bit",
                    type_name, instance_name, input_name
                ));
            };
            if *source == BitSource::Unknown {
                return Err(format!(
                    "cell '{}' instance '{}' input '{}' is unknown in the selected cone",
                    type_name, instance_name, input_name
                ));
            }
            inputs.push((input_name, *source));
        }
        let hold_feedback = if self.collapse_sequential && !cell.sequential.is_empty() {
            feedback::recognize_direct_hold_mux(
                self,
                instance_index,
                connection_index,
                formula,
                &inputs,
            )?
        } else {
            None
        };
        if let Some(output_bit) = hold_feedback {
            if !self.collapse_load_enable_feedback {
                return Err(format!(
                    "load-enable feedback encountered in flop '{}' (cell '{}', output '{}'); set collapse_load_enable_feedback=true (CLI: --collapse_load_enable_feedback) to collapse it assuming prior Q is zero",
                    instance_name, type_name, pin_name
                ));
            }
            for (_, source) in &mut inputs {
                if *source == BitSource::Bit(output_bit) {
                    *source = BitSource::Literal(false);
                }
            }
            log::info!("assuming zero prior state for hold-feedback flop '{instance_name}'");
        }
        Ok(PreparedDriver::CellOutput {
            instance_index,
            formula_key,
            inputs,
        })
    }

    /// Emits a prepared bit after all of its dependencies have been resolved.
    fn emit_prepared(&mut self, bit: BitIndex) -> Result<(), String> {
        let prepared = self.prepared[bit]
            .as_ref()
            .ok_or_else(|| "internal error: missing prepared driver".to_string())?;
        let value = match prepared {
            PreparedDriver::Assign {
                assign_index,
                rhs_bit_index,
            } => eval_bit_expr(
                &self.normalized.assigns[*assign_index].rhs_bits[*rhs_bit_index],
                &self.values,
                &mut self.builder,
            )?,
            PreparedDriver::CellOutput {
                instance_index,
                formula_key,
                inputs,
            } => {
                let mut input_map = HashMap::new();
                for (name, source) in inputs {
                    input_map.insert(
                        name.clone(),
                        resolved_source(*source, &self.values, &mut self.builder)?,
                    );
                }
                let (formula, original_formula) = &self.cell_formulas[formula_key];
                let instance = &self.normalized.instances[*instance_index];
                let context = EmitContext {
                    cell_name: &formula_key.0,
                    original_formula,
                    instance_name: self.interner.resolve(instance.instance_name),
                    port_map: None,
                };
                formula.emit_formula_term(&mut self.builder, &input_map, &context)?
            }
        };
        self.values[bit] = Some(value);
        Ok(())
    }

    fn render_bit(&self, bit: BitIndex) -> String {
        self.normalized.render_bit(bit, self.nets, self.interner)
    }
}

/// Validates names used for one side of the extracted AIG interface.
fn validate_names<'a>(names: impl Iterator<Item = &'a str>, kind: &str) -> Result<(), String> {
    let mut seen = HashSet::new();
    for name in names {
        if name.is_empty() || name.contains(['\n', '\r']) {
            return Err(format!(
                "{} name must be nonempty and contain no newlines",
                kind
            ));
        }
        if !seen.insert(name) {
            return Err(format!("duplicate {} name '{}'", kind, name));
        }
    }
    Ok(())
}

/// Resolves a normalized source after its net-bit dependencies have been
/// emitted.
fn resolved_source(
    source: BitSource,
    values: &[Option<AigOperand>],
    builder: &mut GateBuilder,
) -> Result<AigOperand, String> {
    match source {
        BitSource::Bit(bit) => values[bit]
            .ok_or_else(|| format!("internal error: dependency bit {} is unresolved", bit)),
        BitSource::Literal(true) => Ok(builder.get_true()),
        BitSource::Literal(false) => Ok(builder.get_false()),
        BitSource::Unknown => Err("unknown literal in selected cone".to_string()),
    }
}

/// Emits a normalized continuous-assignment expression into the AIG.
fn eval_bit_expr(
    expression: &BitExpr,
    values: &[Option<AigOperand>],
    builder: &mut GateBuilder,
) -> Result<AigOperand, String> {
    match expression {
        BitExpr::Source(source) => resolved_source(*source, values, builder),
        BitExpr::Not(inner) => {
            let value = eval_bit_expr(inner, values, builder)?;
            Ok(builder.add_not(value))
        }
        BitExpr::And(lhs, rhs) | BitExpr::Or(lhs, rhs) | BitExpr::Xor(lhs, rhs) => {
            let lhs = eval_bit_expr(lhs, values, builder)?;
            let rhs = eval_bit_expr(rhs, values, builder)?;
            Ok(match expression {
                BitExpr::And(_, _) => builder.add_and_binary(lhs, rhs),
                BitExpr::Or(_, _) => builder.add_or_binary(lhs, rhs),
                BitExpr::Xor(_, _) => builder.add_xor_binary(lhs, rhs),
                BitExpr::Source(_) | BitExpr::Not(_) => unreachable!(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::IrBits;

    use super::*;
    use crate::aig_sim::gate_sim::PreparedGateSim;
    use crate::netlist::hierarchy::elaborate_hierarchy;
    use crate::netlist::io::{ParsedNetlist, load_liberty_from_path, select_module};
    use crate::netlist::parse::{Parser, TokenScanner};

    const LIBERTY: &str = r#"
format_magic: 5496997758177923663
cells: {
  name: "DFF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
cells: {
  name: "DFFR"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  pins: { name_string_id: 5 direction: INPUT }
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "CLK"
    clear_expr: "!RSTN"
    kind: SEQUENTIAL_KIND_FF
  }
}
cells: {
  name: "DFFAO22"
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  pins: { name_string_id: 6 direction: INPUT }
  pins: { name_string_id: 7 direction: INPUT }
  pins: { name_string_id: 8 direction: INPUT }
  pins: { name_string_id: 9 direction: INPUT }
  pins: { name_string_id: 10 direction: INPUT }
  pins: { name_string_id: 11 direction: INPUT }
  sequential: {
    state_var: "IQ"
    next_state: "(((d0_0 & d0_1) | (d1_1 & d1_0)) & !te) | (ti & te)"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
cells: {
  name: "INV"
  pins: { name_string_id: 12 direction: INPUT }
  pins: { name_string_id: 13 direction: OUTPUT function_string_id: 14 }
}
interned_strings: ["D", "CLK", "Q", "IQ", "RSTN", "d0_0", "d0_1", "d1_0", "d1_1", "ti", "te", "I", "O", "!I"]
"#;

    /// Parses a small netlist and provides its elaboration and test Liberty.
    fn with_fixture<T>(
        netlist: &'static str,
        f: impl FnOnce(&ElaboratedNetlist, &Library) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut parser = Parser::new(TokenScanner::from_str(netlist));
        let modules = parser.parse_file().unwrap();
        let parsed = ParsedNetlist {
            modules,
            nets: parser.nets,
            interner: parser.interner,
        };
        let module = select_module(&parsed, Some("top")).unwrap();
        let elaborated = elaborate_hierarchy(&parsed, module).unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        let liberty_path = temp_dir.path().join("cells.textproto");
        std::fs::write(&liberty_path, LIBERTY).unwrap();
        let liberty = load_liberty_from_path(&liberty_path).unwrap();
        f(&elaborated, &liberty)
    }

    #[test]
    fn collapses_direct_hold_feedback_with_zero_prior_state() {
        let netlist = r#"
module top(data, valid, clk, y);
  input [1:0] data;
  input valid;
  input clk;
  output [1:0] y;
  wire hold;
  INV u_inv (.I(valid), .O(hold));
  DFFAO22 z_pipe (.d0_0(y[0]), .d0_1(hold), .d1_0(data[0]), .d1_1(valid),
                   .ti(1'b0), .te(1'b0), .CLK(clk), .Q(y[0]));
  DFFAO22 a_pipe (.d0_0(y[1]), .d0_1(hold), .d1_0(data[1]), .d1_1(valid),
                   .ti(1'b0), .te(1'b0), .CLK(clk), .Q(y[1]));
endmodule
"#;
        let gate_fn = with_fixture(netlist, |elaborated, liberty| {
            extract_gatefn_all_top_ports(
                elaborated,
                Some(liberty),
                /* collapse_sequential= */ true,
                /* collapse_load_enable_feedback= */ true,
            )
        })
        .unwrap();
        assert_eq!(
            gate_fn
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            vec!["data", "valid", "clk"]
        );

        // Both feedback loops use zero as their prior state. Updating loads
        // data, while holding produces zero without adding interface inputs.
        let mut sim = PreparedGateSim::new(&gate_fn);
        for data in 0..=3 {
            for valid in 0..=1 {
                let expected = if valid == 1 { data } else { 0 };
                let outputs = sim.eval_outputs(&[
                    IrBits::make_ubits(2, data).unwrap(),
                    IrBits::make_ubits(1, valid).unwrap(),
                    IrBits::make_ubits(1, 0).unwrap(),
                ]);
                assert_eq!(outputs, vec![IrBits::make_ubits(2, expected).unwrap()]);
            }
        }
    }

    #[test]
    fn rejects_direct_feedback_without_complementary_enable_guards() {
        let netlist = r#"
module top(data, enable, clk, y);
  input data;
  input enable;
  input clk;
  output y;
  DFFAO22 u_pipe (.d0_0(y), .d0_1(enable), .d1_0(data), .d1_1(enable),
                   .ti(1'b0), .te(1'b0), .CLK(clk), .Q(y));
endmodule
"#;
        let error = with_fixture(netlist, |elaborated, liberty| {
            extract_gatefn_all_top_ports(
                elaborated,
                Some(liberty),
                /* collapse_sequential= */ true,
                /* collapse_load_enable_feedback= */ false,
            )
        })
        .unwrap_err();
        assert!(error.contains("dependency cycle"));
    }
}
