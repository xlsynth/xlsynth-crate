// SPDX-License-Identifier: Apache-2.0

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::Serialize;
use xlsynth_g8r::aig::topo::topo_order_and_cycle_check;
use xlsynth_g8r::aig::{AigNode, AigOperand, GateFn};

use crate::LeanError;
use crate::support::write_file;

/// A validated dotted Lean module name with portable path components.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LeanModuleName(String);

impl LeanModuleName {
    /// Validates an ASCII dotted module name such as `Generated.Adder`.
    /// Lean's standard module roots and the companion support roots are
    /// reserved.
    pub fn new(name: impl Into<String>) -> Result<Self, LeanError> {
        let name = name.into();
        if name.is_empty()
            || name.split('.').any(|part| {
                let mut chars = part.chars();
                !chars
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    || chars.any(|c| !c.is_ascii_alphanumeric() && c != '_')
            })
        {
            return Err(LeanError::InvalidModuleName(name));
        }
        if name.split('.').next().is_some_and(|part| {
            ["Init", "Lean", "Std", "Lake", "XlsynthLean", "EvalMain"]
                .iter()
                .any(|reserved| part.eq_ignore_ascii_case(reserved))
        }) {
            return Err(LeanError::InvalidModuleName(format!(
                "{name} is reserved for Lean or the companion support modules"
            )));
        }
        Ok(Self(name))
    }

    /// Returns the dotted module name.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the module's portable source path.
    fn source_path(&self) -> PathBuf {
        let mut path = self.0.split('.').collect::<PathBuf>();
        path.set_extension("lean");
        path
    }

    /// Renders every component as a quoted Lean identifier.
    pub(crate) fn lean_identifier(&self) -> String {
        self.0
            .split('.')
            .map(|part| format!("«{part}»"))
            .collect::<Vec<_>>()
            .join(".")
    }
}

/// Options controlling the generated Lean module name.
#[derive(Debug, Clone)]
pub struct LeanExportOptions {
    pub module_name: LeanModuleName,
}

impl Default for LeanExportOptions {
    fn default() -> Self {
        Self {
            module_name: LeanModuleName("Generated.Circuit".to_string()),
        }
    }
}

/// One named packed port in the original `GateFn` interface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CircuitPort {
    pub name: String,
    pub bit_count: usize,
}

/// Ordered vector interface used to flatten inputs and regroup outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CircuitInterface {
    inputs: Vec<CircuitPort>,
    outputs: Vec<CircuitPort>,
    input_bit_count: usize,
    output_bit_count: usize,
}

impl CircuitInterface {
    /// Returns input ports in `GateFn.inputs` order.
    pub fn inputs(&self) -> &[CircuitPort] {
        &self.inputs
    }

    /// Returns output ports in `GateFn.outputs` order.
    pub fn outputs(&self) -> &[CircuitPort] {
        &self.outputs
    }

    /// Returns the total flattened input width.
    pub fn input_bit_count(&self) -> usize {
        self.input_bit_count
    }

    /// Returns the total flattened output width.
    pub fn output_bit_count(&self) -> usize {
        self.output_bit_count
    }
}

/// Generated Lean source, ASCII AIGER data, and interface for one circuit.
#[derive(Debug, Clone)]
pub struct LeanCircuitBundle {
    module_name: LeanModuleName,
    interface: CircuitInterface,
    lean_source: String,
    aag_source: String,
}

impl LeanCircuitBundle {
    /// Returns the generated module name.
    pub fn module_name(&self) -> &LeanModuleName {
        &self.module_name
    }

    /// Returns the original ordered vector interface.
    pub fn interface(&self) -> &CircuitInterface {
        &self.interface
    }

    /// Returns the generated Lean module source.
    pub fn lean_source(&self) -> &str {
        &self.lean_source
    }

    /// Returns the generated ASCII AIGER source.
    pub fn aag_source(&self) -> &str {
        &self.aag_source
    }

    /// Returns the generated source path relative to a bundle root.
    pub fn source_relative_path(&self) -> PathBuf {
        self.module_name.source_path()
    }

    /// Returns the ASCII AIGER path relative to a bundle root.
    pub fn aag_relative_path(&self) -> PathBuf {
        self.module_name.source_path().with_extension("aag")
    }

    /// Writes the circuit source and ASCII AIGER data into `root`.
    ///
    /// Use [`crate::write_lean_support`] separately to write the companion Lean
    /// support sources and toolchain pin.
    pub fn write_to(&self, root: &Path) -> Result<WrittenLeanCircuit, LeanError> {
        let source = self.source_relative_path();
        let aag = self.aag_relative_path();
        write_file(root, &source, self.lean_source.as_bytes())?;
        write_file(root, &aag, self.aag_source.as_bytes())?;
        Ok(WrittenLeanCircuit {
            source: root.join(source),
            aag: root.join(aag),
        })
    }
}

/// Paths written by [`LeanCircuitBundle::write_to`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenLeanCircuit {
    pub source: PathBuf,
    pub aag: PathBuf,
}

/// Canonical graph records used to generate ASCII AIGER.
struct GraphData {
    input_count: usize,
    ands: Vec<(u64, u64)>,
    outputs: Vec<u64>,
}

/// Exports a `GateFn` directly to a Lean circuit bundle.
///
/// Inputs must be unique, non-negated references to input nodes. Every stored
/// input node must be bound by the interface. All references must be in bounds
/// and the complete graph must be acyclic. Input and output vectors are
/// flattened in port order, with each vector's least-significant bit first.
pub fn export_gate_fn(
    graph: &GateFn,
    options: &LeanExportOptions,
) -> Result<LeanCircuitBundle, LeanError> {
    let interface = make_interface(graph)?;
    let data = canonical_graph_data(graph, &interface)?;
    let aag_source = emit_aag(&data)?;
    let aag_filename = options
        .module_name
        .source_path()
        .with_extension("aag")
        .file_name()
        .expect("validated module name has a filename")
        .to_string_lossy()
        .into_owned();
    let namespace = options.module_name.lean_identifier();
    let input_count = interface.input_bit_count;
    let output_count = interface.output_bit_count;
    let lean_source = format!(
        r#"-- SPDX-License-Identifier: Apache-2.0

import XlsynthLean.Aag

namespace {namespace}

def circuit : XlsynthLean.Aig.Circuit {input_count} {output_count} :=
  (XlsynthLean.Aag.parse {input_count} {output_count}
    (include_str "{aag_filename}")).toOption.get (by native_decide)

def eval (inputs : BitVec {input_count}) : BitVec {output_count} :=
  circuit.eval inputs

end {namespace}
"#
    );
    Ok(LeanCircuitBundle {
        module_name: options.module_name.clone(),
        interface,
        lean_source,
        aag_source,
    })
}

/// Builds the vector interface while checking total-width arithmetic.
fn make_interface(graph: &GateFn) -> Result<CircuitInterface, LeanError> {
    let inputs = graph
        .inputs
        .iter()
        .map(|port| CircuitPort {
            name: port.name.clone(),
            bit_count: port.get_bit_count(),
        })
        .collect::<Vec<_>>();
    let outputs = graph
        .outputs
        .iter()
        .map(|port| CircuitPort {
            name: port.name.clone(),
            bit_count: port.get_bit_count(),
        })
        .collect::<Vec<_>>();
    let total = |ports: &[CircuitPort], kind: &str| {
        ports.iter().try_fold(0usize, |sum, port| {
            sum.checked_add(port.bit_count).ok_or_else(|| {
                LeanError::InvalidGraph(format!("total {kind} width overflows usize"))
            })
        })
    };
    Ok(CircuitInterface {
        input_bit_count: total(&inputs, "input")?,
        output_bit_count: total(&outputs, "output")?,
        inputs,
        outputs,
    })
}

/// Checks source references and produces a topologically ordered graph.
fn canonical_graph_data(
    graph: &GateFn,
    interface: &CircuitInterface,
) -> Result<GraphData, LeanError> {
    let node_count = graph.gates.len();
    let check_ref = |operand: AigOperand, context: String| {
        if operand.node.id >= node_count {
            Err(LeanError::InvalidGraph(format!(
                "{context} references node {}, but the graph has {node_count} nodes",
                operand.node.id
            )))
        } else {
            Ok(())
        }
    };
    for (node_id, node) in graph.gates.iter().enumerate() {
        if let AigNode::And2 { a, b, .. } = node {
            check_ref(*a, format!("node {node_id} first operand"))?;
            check_ref(*b, format!("node {node_id} second operand"))?;
        }
    }
    for (port_index, output) in graph.outputs.iter().enumerate() {
        for (bit_index, operand) in output.bit_vector.iter_lsb_to_msb().enumerate() {
            check_ref(*operand, format!("output {port_index} bit {bit_index}"))?;
        }
    }

    let mut mapped = vec![None; node_count];
    let mut flat_input = 0usize;
    for (port_index, input) in graph.inputs.iter().enumerate() {
        for (bit_index, operand) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            check_ref(*operand, format!("input {port_index} bit {bit_index}"))?;
            if operand.negated {
                return Err(LeanError::InvalidGraph(format!(
                    "input {port_index} bit {bit_index} is negated"
                )));
            }
            if !matches!(graph.gates[operand.node.id], AigNode::Input { .. }) {
                return Err(LeanError::InvalidGraph(format!(
                    "input {port_index} bit {bit_index} does not reference an input node"
                )));
            }
            if mapped[operand.node.id].is_some() {
                return Err(LeanError::InvalidGraph(format!(
                    "input node {} is bound more than once",
                    operand.node.id
                )));
            }
            let canonical_id = flat_input.checked_add(1).ok_or_else(|| {
                LeanError::InvalidGraph("canonical input index overflows usize".to_string())
            })?;
            mapped[operand.node.id] = Some(raw_literal(canonical_id, false)?);
            flat_input += 1;
        }
    }
    for (node_id, node) in graph.gates.iter().enumerate() {
        if matches!(node, AigNode::Input { .. }) && mapped[node_id].is_none() {
            return Err(LeanError::InvalidGraph(format!(
                "input node {node_id} is not bound by the GateFn interface"
            )));
        }
    }

    let (order, cycle) = topo_order_and_cycle_check(&graph.gates);
    if let Some(nodes) = cycle {
        return Err(LeanError::InvalidGraph(format!(
            "graph contains a cycle involving nodes {nodes:?}"
        )));
    }
    let mut ands = Vec::new();
    for node_ref in order {
        match &graph.gates[node_ref.id] {
            AigNode::Input { .. } => {
                // Interface validation assigned the canonical input reference.
            }
            AigNode::Literal { value, .. } => {
                mapped[node_ref.id] = Some(u64::from(*value));
            }
            AigNode::And2 { a, b, .. } => {
                let first = mapped_operand(*a, &mapped)?;
                let second = mapped_operand(*b, &mapped)?;
                let canonical_id = interface
                    .input_bit_count
                    .checked_add(ands.len())
                    .and_then(|id| id.checked_add(1))
                    .ok_or_else(|| {
                        LeanError::InvalidGraph("canonical node index overflows usize".to_string())
                    })?;
                mapped[node_ref.id] = Some(raw_literal(canonical_id, false)?);
                ands.push((first, second));
            }
        }
    }
    let outputs = graph
        .outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb())
        .map(|operand| mapped_operand(*operand, &mapped))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GraphData {
        input_count: interface.input_bit_count,
        ands,
        outputs,
    })
}

/// Maps a source operand to its canonical raw literal.
fn mapped_operand(operand: AigOperand, mapped: &[Option<u64>]) -> Result<u64, LeanError> {
    mapped
        .get(operand.node.id)
        .and_then(|value| *value)
        .map(|literal| literal ^ u64::from(operand.negated))
        .ok_or_else(|| {
            LeanError::InvalidGraph(format!("node {} has no canonical mapping", operand.node.id))
        })
}

/// Encodes a canonical node index and edge polarity without overflow.
fn raw_literal(node_id: usize, negated: bool) -> Result<u64, LeanError> {
    u64::try_from(node_id)
        .ok()
        .and_then(|id| id.checked_mul(2))
        .and_then(|literal| literal.checked_add(u64::from(negated)))
        .ok_or_else(|| {
            LeanError::InvalidGraph(
                "node index exceeds supported range for AIG references".to_string(),
            )
        })
}

/// Emits the canonical combinational subset of standard ASCII AIGER.
fn emit_aag(data: &GraphData) -> Result<String, LeanError> {
    let max_var = data
        .input_count
        .checked_add(data.ands.len())
        .ok_or_else(|| {
            LeanError::InvalidGraph("maximum AIGER variable index overflows usize".to_string())
        })?;
    let mut source = format!(
        "aag {max_var} {} 0 {} {}\n",
        data.input_count,
        data.outputs.len(),
        data.ands.len()
    );
    for input in 0..data.input_count {
        let node_id = input.checked_add(1).ok_or_else(|| {
            LeanError::InvalidGraph("AIGER input index overflows usize".to_string())
        })?;
        writeln!(&mut source, "{}", raw_literal(node_id, false)?)
            .expect("writing to String cannot fail");
    }
    for output in &data.outputs {
        writeln!(&mut source, "{output}").expect("writing to String cannot fail");
    }
    for (index, (first, second)) in data.ands.iter().enumerate() {
        let node_id = data
            .input_count
            .checked_add(index)
            .and_then(|id| id.checked_add(1))
            .ok_or_else(|| {
                LeanError::InvalidGraph("AIGER AND index overflows usize".to_string())
            })?;
        writeln!(
            &mut source,
            "{} {first} {second}",
            raw_literal(node_id, false)?
        )
        .expect("writing to String cannot fail");
    }
    Ok(source)
}
