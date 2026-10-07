// SPDX-License-Identifier: Apache-2.0

//! Named source/sink extraction for gate-level netlists projected to AIGs.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

use crate::aig::{AigBitVector, AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::liberty::cell_formula::{EmitContext, Term};
use crate::liberty_model::{Library, PinDirection};
use crate::netlist::gatefn_from_netlist::build_cell_formula_map;
use crate::netlist::hierarchy::{ElaboratedModuleBoundary, ElaboratedNetlist};
use crate::netlist::normalized::{BitExpr, BitIndex, BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{Net, PortDirection};

mod feedback;
mod selectors;

pub use selectors::{
    Gv2AigBoundaryEndpoint, Gv2AigBoundaryKind, Gv2AigBoundaryMatcher, Gv2AigBoundaryRequest,
    Gv2AigBoundarySelection, load_gv2aig_boundary_request,
};

/// Ordered source and sink boundaries for one extracted AIG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gv2AigBoundarySpec {
    pub sources: Vec<Gv2AigSourceBoundary>,
    pub sinks: Vec<Gv2AigSinkBoundary>,
}

/// One named input of the extracted AIG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gv2AigSourceBoundary {
    pub name: String,
    pub selector: Gv2AigSourceSelector,
}

/// A module input port or mapped cell output pin at which traversal stops.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Gv2AigSourceSelector {
    ModuleInput {
        /// Exact elaborated module instance name; empty selects the top.
        #[serde(default)]
        instance_name: String,
        port: String,
    },
    /// Cuts one connected scalar output pin of a combinational or sequential
    /// leaf cell, making its signal an independent AIG input.
    CellOutput {
        /// Exact elaborated name of the mapped leaf instance.
        instance_name: String,
        pin: String,
    },
}

/// One named output of the extracted AIG, selected from a module output port.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gv2AigSinkBoundary {
    pub name: String,
    /// Exact elaborated module instance name; empty selects the top.
    #[serde(default)]
    pub instance_name: String,
    pub port: String,
}

/// An extracted AIG together with the exact boundaries resolved from selectors.
pub struct Gv2AigBoundaryExtraction {
    pub gate_fn: GateFn,
    pub boundaries: Gv2AigBoundarySpec,
}

/// Loads a JSON boundary specification from a file.
pub fn load_gv2aig_boundary_spec(path: &Path) -> anyhow::Result<Gv2AigBoundarySpec> {
    let contents = std::fs::read(path)
        .with_context(|| format!("failed to read boundary specification {}", path.display()))?;
    serde_json::from_slice(&contents)
        .with_context(|| format!("failed to parse boundary specification {}", path.display()))
}

/// Extracts the selected cones, stopping at sources and optionally crossing
/// other flops. When load-enable feedback collapsing is enabled, crossing a
/// recognized direct hold-feedback flop assumes its prior Q value is zero.
/// Explicit source boundaries remain arbitrary inputs.
pub fn extract_gatefn_with_boundaries(
    elaborated: &ElaboratedNetlist,
    liberty: &Library,
    collapse_sequential: bool,
    collapse_load_enable_feedback: bool,
    boundaries: &Gv2AigBoundarySpec,
) -> Result<GateFn, String> {
    BoundaryExtractor::new(
        elaborated,
        liberty,
        collapse_sequential,
        collapse_load_enable_feedback,
    )?
    .extract(boundaries)
}

/// Resolves compact selectors, or selects all top-module ports, and extracts
/// their source/sink cones through the same traversal.
pub fn extract_gatefn_with_optional_boundary_request(
    elaborated: &ElaboratedNetlist,
    liberty: &Library,
    collapse_sequential: bool,
    collapse_load_enable_feedback: bool,
    request: Option<&Gv2AigBoundaryRequest>,
) -> Result<Gv2AigBoundaryExtraction, String> {
    let extractor = BoundaryExtractor::new(
        elaborated,
        liberty,
        collapse_sequential,
        collapse_load_enable_feedback,
    )?;
    let boundaries = match request {
        Some(request) => extractor.resolve_request(request)?,
        None => extractor.top_port_boundaries(),
    };
    let gate_fn = extractor.extract(&boundaries)?;
    Ok(Gv2AigBoundaryExtraction {
        gate_fn,
        boundaries,
    })
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

#[derive(Debug)]
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

/// One pending action in the iterative dependency traversal.
enum Frame {
    Enter(BitIndex),
    Emit {
        bit: BitIndex,
        driver: PreparedDriver,
    },
}

struct BoundaryExtractor<'a> {
    normalized: NormalizedNetlistModule<'a>,
    nets: &'a [Net],
    interner: &'a StringInterner<StringBackend<SymbolU32>>,
    module_boundaries: &'a [ElaboratedModuleBoundary],
    liberty: &'a Library,
    collapse_sequential: bool,
    collapse_load_enable_feedback: bool,
    cell_index_by_name: HashMap<String, usize>,
    instance_index_by_name: HashMap<String, usize>,
    drivers: Vec<Vec<BitDriver>>,
    values: Vec<Option<AigOperand>>,
    source_owners: Vec<Option<String>>,
    active: Vec<bool>,
    prepared_cell_types: HashSet<String>,
    cell_formulas: HashMap<(String, String), (Term, String)>,
    builder: GateBuilder,
}

impl<'a> BoundaryExtractor<'a> {
    /// Indexes signal drivers and Liberty cell directions for the elaborated
    /// netlist.
    fn new(
        elaborated: &'a ElaboratedNetlist,
        liberty: &'a Library,
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
        for (index, cell) in liberty.cells.iter().enumerate() {
            if cell_index_by_name
                .insert(cell.name.clone(), index)
                .is_some()
            {
                return Err(format!("Liberty contains duplicate cell '{}'", cell.name));
            }
        }
        let mut instance_index_by_name = HashMap::new();
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
            if instance_index_by_name
                .insert(instance_name.clone(), instance_index)
                .is_some()
            {
                return Err(format!(
                    "duplicate flattened instance name '{}'",
                    instance_name
                ));
            }
            let type_name = elaborated.interner.resolve(instance.type_name).unwrap();
            let cell_index = cell_index_by_name.get(type_name).ok_or_else(|| {
                format!(
                    "cell '{}' instance '{}' not found in Liberty",
                    type_name, instance_name
                )
            })?;
            let cell = &liberty.cells[*cell_index];
            for (connection_index, connection) in instance.connections.iter().enumerate() {
                let pin_name = elaborated.interner.resolve(connection.port).unwrap();
                let Some(pin) = cell
                    .pins
                    .iter()
                    .find(|pin| liberty.resolve_string(&pin.name) == pin_name)
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
            module_boundaries: &elaborated.module_boundaries,
            liberty,
            collapse_sequential,
            collapse_load_enable_feedback,
            cell_index_by_name,
            instance_index_by_name,
            drivers,
            values: vec![None; bit_count],
            source_owners: vec![None; bit_count],
            active: vec![false; bit_count],
            prepared_cell_types: HashSet::new(),
            cell_formulas: HashMap::new(),
            builder: GateBuilder::new(module_name.to_string(), GateBuilderOptions::no_opt()),
        })
    }

    /// Selects every directed top-module port in declaration order.
    fn top_port_boundaries(&self) -> Gv2AigBoundarySpec {
        let mut boundaries = Gv2AigBoundarySpec {
            sources: Vec::new(),
            sinks: Vec::new(),
        };
        for port in &self.normalized.ports {
            let name = self.interner.resolve(port.name).unwrap().to_string();
            match &port.direction {
                PortDirection::Input => boundaries.sources.push(Gv2AigSourceBoundary {
                    name: name.clone(),
                    selector: Gv2AigSourceSelector::ModuleInput {
                        instance_name: String::new(),
                        port: name,
                    },
                }),
                PortDirection::Output => boundaries.sinks.push(Gv2AigSinkBoundary {
                    name: name.clone(),
                    instance_name: String::new(),
                    port: name,
                }),
                PortDirection::Inout => unreachable!("inout ports were rejected"),
            }
        }
        boundaries
    }

    /// Resolves selectors, creates the ordered interface, and lowers only sink
    /// cones.
    fn extract(mut self, boundaries: &Gv2AigBoundarySpec) -> Result<GateFn, String> {
        if boundaries.sinks.is_empty() {
            return Err("boundary specification must contain at least one sink".to_string());
        }
        validate_names(
            boundaries.sources.iter().map(|source| source.name.as_str()),
            "source",
        )?;
        validate_names(
            boundaries.sinks.iter().map(|sink| sink.name.as_str()),
            "sink",
        )?;
        for source in &boundaries.sources {
            let bits = self
                .resolve_source(&source.selector)
                .map_err(|error| format!("while resolving source '{}': {}", source.name, error))?;
            let input = self.builder.add_input(source.name.clone(), bits.len());
            for (offset, &bit) in bits.iter().enumerate() {
                let owner = format!("{}[{}]", source.name, offset);
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
        for sink in &boundaries.sinks {
            let bits = self
                .module_port_bits(&sink.instance_name, &sink.port, PortDirection::Output)
                .map_err(|error| format!("while resolving sink '{}': {}", sink.name, error))?;
            let mut output_bits = Vec::with_capacity(bits.len());
            for bit in bits {
                output_bits.push(self.resolve_bit(bit).map_err(|error| {
                    format!("while extracting sink '{}': {}", sink.name, error)
                })?);
            }
            self.builder.add_output(
                sink.name.clone(),
                AigBitVector::from_lsb_is_index_0(&output_bits),
            );
        }
        Ok(self.builder.build())
    }

    /// Resolves a source selector to canonical signal bits.
    fn resolve_source(&self, selector: &Gv2AigSourceSelector) -> Result<Vec<BitIndex>, String> {
        match selector {
            Gv2AigSourceSelector::ModuleInput {
                instance_name,
                port,
            } => self.module_port_bits(instance_name, port, PortDirection::Input),
            Gv2AigSourceSelector::CellOutput { instance_name, pin } => self
                .cell_output_bit(instance_name, pin)
                .map(|bit| vec![bit]),
        }
    }

    /// Resolves a top or child module port to canonical bits in LSB-first
    /// order.
    fn module_port_bits(
        &self,
        instance_name: &str,
        port_name: &str,
        expected_direction: PortDirection,
    ) -> Result<Vec<BitIndex>, String> {
        if instance_name.is_empty() {
            let port = self
                .normalized
                .ports
                .iter()
                .find(|port| self.interner.resolve(port.name) == Some(port_name))
                .ok_or_else(|| format!("top module has no port '{}'", port_name))?;
            if port.direction != expected_direction {
                return Err(format!(
                    "top module port '{}' has direction {:?}; expected {:?}",
                    port_name, port.direction, expected_direction
                ));
            }
            return Ok(port.bits.clone());
        }
        let boundary = self
            .module_boundaries
            .iter()
            .find(|boundary| boundary.instance_path == instance_name)
            .ok_or_else(|| format!("module instance '{}' was not found", instance_name))?;
        let port = boundary
            .ports
            .iter()
            .find(|port| self.interner.resolve(port.name) == Some(port_name))
            .ok_or_else(|| {
                format!(
                    "module instance '{}' has no port '{}'",
                    instance_name, port_name
                )
            })?;
        if port.direction != expected_direction {
            return Err(format!(
                "module instance '{}' port '{}' has direction {:?}; expected {:?}",
                instance_name, port_name, port.direction, expected_direction
            ));
        }
        Ok(self
            .normalized
            .net_bits(port.net)
            .iter()
            .map(|&bit| self.normalized.canonical_bit(bit))
            .collect())
    }

    /// Resolves and validates one mapped cell output pin selected as a source.
    fn cell_output_bit(&self, instance_name: &str, pin_name: &str) -> Result<BitIndex, String> {
        let instance_index = self
            .instance_index_by_name
            .get(instance_name)
            .ok_or_else(|| format!("cell instance '{}' was not found", instance_name))?;
        let instance = &self.normalized.instances[*instance_index];
        let type_name = self.interner.resolve(instance.type_name).unwrap();
        let library = self.liberty;
        let cell = &library.cells[self.cell_index_by_name[type_name]];
        let pin = cell
            .pins
            .iter()
            .find(|pin| library.resolve_string(&pin.name) == pin_name)
            .ok_or_else(|| format!("cell '{}' has no pin '{}'", type_name, pin_name))?;
        if pin.direction != PinDirection::Output as i32 {
            return Err(format!(
                "cell instance '{}' pin '{}' is not an output",
                instance_name, pin_name
            ));
        }
        let connection = instance
            .connections
            .iter()
            .find(|connection| self.interner.resolve(connection.port) == Some(pin_name))
            .ok_or_else(|| {
                format!(
                    "cell instance '{}' pin '{}' is unconnected",
                    instance_name, pin_name
                )
            })?;
        let [BitSource::Bit(bit)] = connection.bits.as_slice() else {
            return Err(format!(
                "cell instance '{}' pin '{}' must connect to one net bit",
                instance_name, pin_name
            ));
        };
        Ok(*bit)
    }

    /// Resolves one sink bit with iterative DFS so long netlist chains do not
    /// recurse.
    fn resolve_bit(&mut self, root: BitIndex) -> Result<AigOperand, String> {
        let mut stack = vec![Frame::Enter(root)];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(bit) => {
                    if self.values[bit].is_some() {
                        continue;
                    }
                    if self.active[bit] {
                        return Err(format!(
                            "dependency cycle reaches net bit '{}' in the selected cone",
                            self.render_bit(bit)
                        ));
                    }
                    self.active[bit] = true;
                    let (driver, dependencies) = self.prepare_bit(bit)?;
                    stack.push(Frame::Emit { bit, driver });
                    for dependency in dependencies.into_iter().rev() {
                        if self.values[dependency].is_none() {
                            stack.push(Frame::Enter(dependency));
                        }
                    }
                }
                Frame::Emit { bit, driver } => {
                    self.emit_prepared(bit, driver)?;
                    self.active[bit] = false;
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

    /// Prepares the driver of one reached bit and returns it with its net-bit
    /// dependencies.
    fn prepare_bit(&mut self, bit: BitIndex) -> Result<(PreparedDriver, Vec<BitIndex>), String> {
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
        Ok((prepared, dependencies))
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
        let library = self.liberty;
        let cell = &library.cells[self.cell_index_by_name[&type_name]];
        if !self.collapse_sequential && !cell.sequential.is_empty() {
            return Err(format!(
                "reached sequential cell '{}' instance '{}' output '{}'; select it as a source or enable collapse_sequential",
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
    fn emit_prepared(&mut self, bit: BitIndex, prepared: PreparedDriver) -> Result<(), String> {
        let value = match prepared {
            PreparedDriver::Assign {
                assign_index,
                rhs_bit_index,
            } => eval_bit_expr(
                &self.normalized.assigns[assign_index].rhs_bits[rhs_bit_index],
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
                        name,
                        resolved_source(source, &self.values, &mut self.builder)?,
                    );
                }
                let (formula, original_formula) = &self.cell_formulas[&formula_key];
                let instance = &self.normalized.instances[instance_index];
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
    use crate::aig_serdes::emit_aiger::emit_aiger;
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
cells: {
  name: "BUF_INV"
  pins: { name_string_id: 12 direction: INPUT }
  pins: { name_string_id: 13 direction: OUTPUT function_string_id: 12 }
  pins: { name_string_id: 15 direction: OUTPUT function_string_id: 14 }
}
interned_strings: ["D", "CLK", "Q", "IQ", "RSTN", "d0_0", "d0_1", "d1_0", "d1_1", "ti", "te", "I", "O", "!I", "ON"]
"#;

    const MULTI_OUTPUT_NETLIST: &str = r#"
module top(data, y, z);
  input data;
  output y;
  output z;
  BUF_INV u_pair (.I(data), .O(y), .ON(z));
endmodule
"#;

    const NETLIST: &str = r#"
module stage(data, spare, clk, rstn, result, complement, unused);
  input [1:0] data;
  input spare;
  input clk;
  input rstn;
  output [1:0] result;
  output complement;
  output unused;
  wire state;
  wire state_next;
  wire pipe_next;
  wire pipe_q;
  wire result_hi;
  assign state_next = state ^ spare;
  DFFR u_state (.D(state_next), .CLK(clk), .RSTN(rstn), .Q(state));
  assign pipe_next = data[0] ^ state;
  DFF u_pipe (.D(pipe_next), .CLK(clk), .Q(pipe_q));
  assign result_hi = data[1] & state;
  assign result = {result_hi, pipe_q};
  assign complement = ~pipe_q;
  assign unused = spare;
endmodule

module top(ext_data, spare, clk, rstn, top_result, unused_top);
  input [1:0] ext_data;
  input spare;
  input clk;
  input rstn;
  output [1:0] top_result;
  output unused_top;
  wire [1:0] pre_data;
  wire complement;
  assign pre_data = ~ext_data;
  stage u_stage (.data(pre_data), .spare(spare), .clk(clk), .rstn(rstn),
                   .result(top_result), .complement(complement), .unused(unused_top));
endmodule
"#;

    const BOUNDARIES: &str = r#"
{
  "sources": [
    {"name": "state", "selector": {"kind": "cell_output", "instance_name": "u_stage/u_state", "pin": "Q"}},
    {"name": "data", "selector": {"kind": "module_input", "instance_name": "u_stage", "port": "data"}}
  ],
  "sinks": [
    {"name": "not_result_bit", "instance_name": "u_stage", "port": "complement"},
    {"name": "result", "instance_name": "u_stage", "port": "result"}
  ]
}
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

    /// Parses a small netlist and extracts its selected top-module cones.
    fn extract_fixture(
        netlist: &'static str,
        boundaries: &Gv2AigBoundarySpec,
        collapse_sequential: bool,
    ) -> Result<GateFn, String> {
        with_fixture(netlist, |elaborated, liberty| {
            extract_gatefn_with_boundaries(
                elaborated,
                liberty,
                collapse_sequential,
                /* collapse_load_enable_feedback= */ false,
                boundaries,
            )
        })
    }

    fn boundary_spec() -> Gv2AigBoundarySpec {
        serde_json::from_str(BOUNDARIES).unwrap()
    }

    #[test]
    fn resolves_regexp_boundaries_across_hierarchy_in_lexical_order() {
        let request = Gv2AigBoundaryRequest::from_selectors(
            &[
                "cell_output_regex:u_stage/u_(?:pipe|state):Q".to_string(),
                "input_port_regex:u_stage:data".to_string(),
            ],
            &["output_port_regex:u_stage:(?:complement|result)".to_string()],
        )
        .unwrap();
        let extraction = with_fixture(NETLIST, |elaborated, liberty| {
            extract_gatefn_with_optional_boundary_request(
                elaborated,
                liberty,
                /* collapse_sequential= */ false,
                /* collapse_load_enable_feedback= */ false,
                Some(&request),
            )
        })
        .unwrap();
        let expected: Gv2AigBoundarySpec = serde_json::from_str(
            r#"{
              "sources": [
                {"name": "u_stage/u_pipe:Q", "selector": {"kind": "cell_output", "instance_name": "u_stage/u_pipe", "pin": "Q"}},
                {"name": "u_stage/u_state:Q", "selector": {"kind": "cell_output", "instance_name": "u_stage/u_state", "pin": "Q"}},
                {"name": "u_stage:data", "selector": {"kind": "module_input", "instance_name": "u_stage", "port": "data"}}
              ],
              "sinks": [
                {"name": "u_stage:complement", "instance_name": "u_stage", "port": "complement"},
                {"name": "u_stage:result", "instance_name": "u_stage", "port": "result"}
              ]
            }"#,
        )
        .unwrap();
        assert_eq!(extraction.boundaries, expected);
        let exact = extract_fixture(NETLIST, &expected, false).unwrap();
        assert_eq!(
            emit_aiger(&extraction.gate_fn, true).unwrap(),
            emit_aiger(&exact, true).unwrap()
        );
        assert_eq!(
            extraction
                .gate_fn
                .inputs
                .iter()
                .map(|input| (input.name.as_str(), input.bit_vector.get_bit_count()))
                .collect::<Vec<_>>(),
            vec![
                ("u_stage/u_pipe:Q", 1),
                ("u_stage/u_state:Q", 1),
                ("u_stage:data", 2),
            ]
        );
    }

    #[test]
    fn extracts_hierarchical_cones_across_pipeline_flops() {
        let gate_fn = extract_fixture(NETLIST, &boundary_spec(), true).unwrap();
        assert_eq!(
            gate_fn
                .inputs
                .iter()
                .map(|input| (input.name.as_str(), input.bit_vector.get_bit_count()))
                .collect::<Vec<_>>(),
            vec![("state", 1), ("data", 2)]
        );
        assert_eq!(
            gate_fn
                .outputs
                .iter()
                .map(|output| (output.name.as_str(), output.bit_vector.get_bit_count()))
                .collect::<Vec<_>>(),
            vec![("not_result_bit", 1), ("result", 2)]
        );

        // The selected state output hides its feedback and asynchronous reset.
        // The selected child input hides the parent inverter. The pipeline FF
        // contributes its D expression, while unused outputs add no inputs.
        let mut sim = PreparedGateSim::new(&gate_fn);
        for state in 0..=1 {
            for data in 0..=3 {
                let pipe = (data & 1) ^ state;
                let result = (((data >> 1) & state) << 1) | pipe;
                let outputs = sim.eval_outputs(&[
                    IrBits::make_ubits(1, state).unwrap(),
                    IrBits::make_ubits(2, data).unwrap(),
                ]);
                assert_eq!(
                    outputs,
                    vec![
                        IrBits::make_ubits(1, pipe ^ 1).unwrap(),
                        IrBits::make_ubits(2, result).unwrap(),
                    ]
                );
            }
        }
    }

    #[test]
    fn rejects_reached_unselected_flop_when_collapse_is_disabled() {
        let error = extract_fixture(NETLIST, &boundary_spec(), false).unwrap_err();
        assert!(error.contains("reached sequential cell 'DFF' instance 'u_stage/u_pipe'"));
    }

    #[test]
    fn selected_cell_outputs_stop_at_flops_with_collapse_disabled() {
        let mut boundaries = boundary_spec();
        boundaries.sources.push(Gv2AigSourceBoundary {
            name: "pipe".to_string(),
            selector: Gv2AigSourceSelector::CellOutput {
                instance_name: "u_stage/u_pipe".to_string(),
                pin: "Q".to_string(),
            },
        });
        let gate_fn = extract_fixture(NETLIST, &boundaries, false).unwrap();
        let mut sim = PreparedGateSim::new(&gate_fn);
        for state in 0..=1 {
            for data in 0..=3 {
                for pipe in 0..=1 {
                    let outputs = sim.eval_outputs(&[
                        IrBits::make_ubits(1, state).unwrap(),
                        IrBits::make_ubits(2, data).unwrap(),
                        IrBits::make_ubits(1, pipe).unwrap(),
                    ]);
                    let result = (((data >> 1) & state) << 1) | pipe;
                    assert_eq!(
                        outputs,
                        vec![
                            IrBits::make_ubits(1, pipe ^ 1).unwrap(),
                            IrBits::make_ubits(2, result).unwrap(),
                        ]
                    );
                }
            }
        }
    }

    #[test]
    fn selected_combinational_cell_output_does_not_cut_other_outputs() {
        let boundaries = serde_json::from_str(
            r#"{
              "sources": [
                {"name": "cut", "selector": {"kind": "cell_output", "instance_name": "u_pair", "pin": "O"}},
                {"name": "data", "selector": {"kind": "module_input", "port": "data"}}
              ],
              "sinks": [{"name": "y", "port": "y"}, {"name": "z", "port": "z"}]
            }"#,
        )
        .unwrap();
        let gate_fn = extract_fixture(MULTI_OUTPUT_NETLIST, &boundaries, false).unwrap();
        let mut sim = PreparedGateSim::new(&gate_fn);
        for cut in 0..=1 {
            for data in 0..=1 {
                let outputs = sim.eval_outputs(&[
                    IrBits::make_ubits(1, cut).unwrap(),
                    IrBits::make_ubits(1, data).unwrap(),
                ]);
                assert_eq!(
                    outputs,
                    vec![
                        IrBits::make_ubits(1, cut).unwrap(),
                        IrBits::make_ubits(1, data ^ 1).unwrap(),
                    ]
                );
            }
        }
    }

    #[test]
    fn selected_cell_outputs_are_independent_inputs() {
        let request = Gv2AigBoundaryRequest::from_selectors(
            &["cell_output_regex:u_pair:(?:O|ON)".to_string()],
            &["output_port:y".to_string(), "output_port:z".to_string()],
        )
        .unwrap();
        let extraction = with_fixture(MULTI_OUTPUT_NETLIST, |elaborated, liberty| {
            extract_gatefn_with_optional_boundary_request(
                elaborated,
                liberty,
                /* collapse_sequential= */ false,
                /* collapse_load_enable_feedback= */ false,
                Some(&request),
            )
        })
        .unwrap();
        assert_eq!(
            extraction
                .gate_fn
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            vec!["u_pair:O", "u_pair:ON"]
        );
        let mut sim = PreparedGateSim::new(&extraction.gate_fn);
        for y in 0..=1 {
            for z in 0..=1 {
                let inputs = [
                    IrBits::make_ubits(1, y).unwrap(),
                    IrBits::make_ubits(1, z).unwrap(),
                ];
                assert_eq!(sim.eval_outputs(&inputs), inputs.to_vec());
            }
        }
    }

    #[test]
    fn rejects_cell_input_pin_as_source() {
        let mut boundaries = boundary_spec();
        boundaries.sources[0].selector = Gv2AigSourceSelector::CellOutput {
            instance_name: "u_stage/u_pipe".to_string(),
            pin: "D".to_string(),
        };
        let error = extract_fixture(NETLIST, &boundaries, true).unwrap_err();
        assert_eq!(
            error,
            "while resolving source 'state': cell instance 'u_stage/u_pipe' pin 'D' is not an output"
        );
    }

    #[test]
    fn rejects_unselected_terminal_input() {
        let mut boundaries = boundary_spec();
        boundaries.sources.pop();
        let error = extract_fixture(NETLIST, &boundaries, true).unwrap_err();
        assert!(error.contains("reached unselected top module input 'ext_data'"));
    }

    #[test]
    fn rejects_duplicate_source_signal_through_hierarchy_alias() {
        let mut boundaries = boundary_spec();
        for (name, instance_name) in [("top_spare", ""), ("child_spare", "u_stage")] {
            boundaries.sources.push(Gv2AigSourceBoundary {
                name: name.to_string(),
                selector: Gv2AigSourceSelector::ModuleInput {
                    instance_name: instance_name.to_string(),
                    port: "spare".to_string(),
                },
            });
        }
        let error = extract_fixture(NETLIST, &boundaries, true).unwrap_err();
        assert!(error.contains("already selected by source 'top_spare[0]'"));
    }

    #[test]
    fn rejects_invalid_selector_direction() {
        let mut boundaries = boundary_spec();
        boundaries.sinks[0].port = "data".to_string();
        let error = extract_fixture(NETLIST, &boundaries, true).unwrap_err();
        assert!(error.contains("port 'data' has direction Input; expected Output"));
    }

    #[test]
    fn rejects_feedback_cycle_when_crossing_an_unselected_flop() {
        let netlist = r#"
module top(a, clk, y);
  input a;
  input clk;
  output y;
  wire next;
  assign next = y ^ a;
  DFF u_feedback (.D(next), .CLK(clk), .Q(y));
endmodule
"#;
        let boundaries = serde_json::from_str(
            r#"{
              "sources": [{"name": "a", "selector": {"kind": "module_input", "port": "a"}}],
              "sinks": [{"name": "y", "port": "y"}]
            }"#,
        )
        .unwrap();
        let error = extract_fixture(netlist, &boundaries, true).unwrap_err();
        assert!(error.contains("dependency cycle"));
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
            extract_gatefn_with_optional_boundary_request(
                elaborated,
                liberty,
                /* collapse_sequential= */ true,
                /* collapse_load_enable_feedback= */ true,
                None,
            )
            .map(|extraction| extraction.gate_fn)
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
        let boundaries = serde_json::from_str(
            r#"{
              "sources": [
                {"name": "data", "selector": {"kind": "module_input", "port": "data"}},
                {"name": "enable", "selector": {"kind": "module_input", "port": "enable"}}
              ],
              "sinks": [{"name": "y", "port": "y"}]
            }"#,
        )
        .unwrap();
        let error = extract_fixture(netlist, &boundaries, true).unwrap_err();
        assert!(error.contains("dependency cycle"));
    }
}
