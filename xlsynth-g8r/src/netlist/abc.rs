// SPDX-License-Identifier: Apache-2.0

//! Direct ABC combinational mapping with native sequential netlist assembly.
//!
//! Raw Liberty files are passed to ABC; the separately supplied normalized
//! library is used for FF selection and checking emitted cells. Matching names
//! and pins do not prove that the two library sources have identical semantics
//! or timing. No Yosys invocation, retiming, or register sizing is performed.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use xlsynth::external_tool::{ToolError, resolve_executable, run_with_timeout_detailed};
use xlsynth_pir::IrBits;

use crate::aig::gate::{Input, Output};
use crate::aig::sequential_cleanup::{SequentialCleanupStats, cleanup_sequential_gate_fn};
use crate::aig::{AigBitVector, AigNode, AigOperand, GateFn, SequentialGateFn};
use crate::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use crate::liberty::cell_formula::abc_source_formula_issues;
use crate::liberty::io_defaults::{
    buffer_input_pin, default_output_load_ff, resolve_reference_buffer,
};
use crate::liberty_model::{Library, PinDirection};
use crate::netlist::emit::emit_module_as_netlist_text;
use crate::netlist::io::{ParsedNetlist, read_gv_from_str};
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistAssign, NetlistAssignKind, NetlistInstance,
    NetlistPort, PortDirection, Pos, Span,
};
use crate::netlist::register_mapping::{
    RegisterCellSelection, RegisterMappingLibrary, RegisterMappingPlan,
};

/// Explicit standalone ABC executable, independent of a Liberty library.
#[derive(Clone, Debug)]
pub struct AbcToolchain {
    path: PathBuf,
    version: String,
}

impl AbcToolchain {
    /// Resolves the executable and probes it without reading startup scripts.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, ToolError> {
        let path = resolve_executable(path.as_ref()).map_err(ToolError::failure)?;
        let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
        let result = run_with_timeout_detailed(
            Command::new(&path).args(["-s", "-c", "version"]),
            directory.path(),
            "abc-version",
            Duration::from_secs(10),
        )?;
        if !result.status.success() {
            return Err(ToolError::from_exit_status(
                "ABC",
                result.status,
                &String::from_utf8_lossy(&result.stdout),
                &String::from_utf8_lossy(&result.stderr),
            ));
        }
        Ok(Self {
            path,
            version: String::from_utf8_lossy(&result.stdout).trim().into(),
        })
    }

    /// Uses XLSYNTH_ABC_PATH, or the standalone `abc` executable on PATH.
    pub fn from_env() -> Result<Self, ToolError> {
        Self::new(std::env::var_os("XLSYNTH_ABC_PATH").unwrap_or_else(|| "abc".into()))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn version(&self) -> &str {
        &self.version
    }
}

/// ABC's legacy global drive/load constraints, not per-register STA bounds.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct AbcIoConstraints {
    /// A usable ordinary non-inverting buffer in both raw and normalized
    /// Liberty.
    pub driving_cell: String,
    /// Output capacitance in ABC's `set_load` units (femtofarads).
    pub output_load_ff: f64,
}

impl AbcIoConstraints {
    /// Uses the library's independent driver and output-capacitance defaults.
    pub fn from_library(library: &Library) -> Result<Self, String> {
        let buffer = resolve_reference_buffer(library)?;
        let constraints = Self {
            driving_cell: buffer.to_owned(),
            output_load_ff: default_output_load_ff(library)?,
        };
        constraints.validate_abc_values()?;
        Ok(constraints)
    }

    /// Rejects constraints ABC cannot safely interpret, including zero load.
    fn validate(&self, library: &Library) -> Result<(), String> {
        self.validate_abc_values()?;
        buffer_input_pin(library, &self.driving_cell)?;
        Ok(())
    }

    /// Checks ABC's textual/numeric limits without repeating library lookup.
    fn validate_abc_values(&self) -> Result<(), String> {
        // ABC stores load as a float; underflow to zero disables its override.
        let load = self.output_load_ff as f32;
        if !simple_identifier(&self.driving_cell) || !load.is_finite() || load <= 0.0 {
            return Err(
                "ABC needs a simple driving-cell name and a positive finite output load".into(),
            );
        }
        Ok(())
    }

    fn to_abc_text(&self) -> String {
        format!(
            "set_driving_cell {}\nset_load {}\n",
            self.driving_cell, self.output_load_ff
        )
    }
}

/// Bounded, deterministic mapping policy. Cleanup is opt-in.
#[derive(Clone, Debug)]
pub struct AbcMappingOptions {
    pub register_selection: RegisterCellSelection,
    pub cleanup_registers: bool,
    pub timeout: Duration,
    /// Required electrical boundary conditions, validated against the mapping
    /// library before ABC runs. There is no unconstrained mapping mode.
    pub io_constraints: AbcIoConstraints,
    /// Optimization/mapping passes, defaulting to `ABC_SPEED_SCRIPT`. Starts
    /// with a strashed network and must leave a mapped network in ABC's main
    /// workspace (`&put` after GIA commands). The wrapper supplies library,
    /// input, and constraint loading and final Verilog export; do not include
    /// those commands here. This is trusted ABC code, not a sandboxed DSL.
    pub script: String,
}

impl AbcMappingOptions {
    /// Uses explicit constraints with area-first FF selection, no register
    /// cleanup, the default speed script, and a 180-second timeout. Mapping
    /// validates the constraints.
    pub fn new(io_constraints: AbcIoConstraints) -> Self {
        Self {
            register_selection: RegisterCellSelection::SmallestArea,
            cleanup_registers: false,
            timeout: Duration::from_secs(180),
            io_constraints,
            script: ABC_SPEED_SCRIPT.to_owned(),
        }
    }

    /// Resolves the library's reference buffer and output load into explicit
    /// constraints, rejecting missing or invalid defaults.
    pub fn from_library(library: &Library) -> Result<Self, String> {
        Ok(Self::new(AbcIoConstraints::from_library(library)?))
    }
}

/// Captured invocation details. Serialization includes only stable metadata;
/// volatile raw logs remain available to callers and error reporting.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct AbcRunDiagnostics {
    pub executable: PathBuf,
    pub version: String,
    pub raw_liberty_files: Vec<PathBuf>,
    pub script: String,
    /// Resolved constraints, including library-derived defaults, for
    /// reproducibility.
    pub io_constraints: Option<AbcIoConstraints>,
    /// Raw tool output, intentionally excluded from deterministic metadata.
    #[serde(skip_serializing)]
    pub stdout: String,
    /// Raw tool errors, intentionally excluded from deterministic metadata.
    #[serde(skip_serializing)]
    pub stderr: String,
}

impl AbcRunDiagnostics {
    /// Keeps invocation details when a successful process yields unusable GV.
    fn failure(&self, error: impl std::fmt::Display) -> ToolError {
        ToolError::failure(error.to_string()).with_context(format!(
            r#"ABC executable: {}
ABC version: {}
ABC Liberty files: {:?}
ABC IO constraints: {:?}
ABC program:
{}
ABC stdout:
{}
ABC stderr:
{}"#,
            self.executable.display(),
            self.version,
            self.raw_liberty_files,
            self.io_constraints,
            self.script,
            self.stdout,
            self.stderr,
        ))
    }
}

/// Complete, flat gate-level netlist and the native metadata used to assemble
/// it.
pub struct AbcMappedDesign {
    pub netlist: ParsedNetlist,
    /// Legalized, optionally cleaned design. Zero-width external ports remain
    /// here but are omitted from Verilog, which has no zero-width port syntax.
    pub design: SequentialGateFn,
    pub registers: RegisterMappingPlan,
    pub cleanup_stats: Option<SequentialCleanupStats>,
    pub diagnostics: AbcRunDiagnostics,
}

impl AbcMappedDesign {
    /// Emits deterministic GV without ABC's timestamp comments.
    pub fn emit_verilog(&self) -> Result<String, String> {
        emit_module_as_netlist_text(
            &self.netlist.modules[0],
            &self.netlist.nets,
            &self.netlist.interner,
        )
        .map_err(|e| e.to_string())
    }
}

/// Reusable native library/catalog and separately specified ABC source files.
pub struct AbcMappingContext {
    toolchain: AbcToolchain,
    raw_liberty_files: Vec<PathBuf>,
    library: Library,
    registers: RegisterMappingLibrary,
}

impl AbcMappingContext {
    /// Keeps source files unchanged. The first one or two files must together
    /// form a usable ABC library; remaining files are merged with `read_lib
    /// -m`.
    pub fn new<P: AsRef<Path>>(
        toolchain: AbcToolchain,
        raw_liberty_files: &[P],
        library: Library,
    ) -> Result<Self, ToolError> {
        if raw_liberty_files.is_empty() {
            return Err("ABC requires raw Liberty files".into());
        }
        let files = raw_liberty_files
            .iter()
            .map(|p| {
                let path = p.as_ref();
                if !path.is_file() {
                    return Err(format!(
                        "ABC Liberty file does not exist: {}",
                        path.display()
                    ));
                }
                if path.extension().is_some_and(|e| e == "gz" || e == "7z") {
                    return Err("ABC integration requires uncompressed raw .lib files".into());
                }
                path.canonicalize().map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        let registers = RegisterMappingLibrary::new(&library)?;
        Ok(Self {
            toolchain,
            raw_liberty_files: files,
            library,
            registers,
        })
    }

    pub fn library(&self) -> &Library {
        &self.library
    }

    /// Maps logic with ABC and assembles physical FFs without passing state to
    /// ABC. Explicit register controls are legalized before optional cleanup.
    pub fn map_sequential(
        &self,
        source: &SequentialGateFn,
        options: &AbcMappingOptions,
    ) -> Result<AbcMappedDesign, ToolError> {
        let mut design = source.legalize_register_controls()?;
        let cleanup_stats = if options.cleanup_registers {
            let result = cleanup_sequential_gate_fn(&design)?;
            design = result.design;
            Some(result.stats)
        } else {
            None
        };
        validate_names(&design)?;
        if self.library.cells.iter().any(|c| c.name == design.name) {
            return Err("top module name conflicts with a Liberty cell name".into());
        }
        if options.timeout.is_zero() {
            return Err("ABC timeout must be positive".into());
        }
        options.io_constraints.validate(&self.library)?;
        let registers = self
            .registers
            .map_registers(&design, &options.register_selection)?;
        let boundary = Boundary::new(&design, &registers);
        let (mut netlist, diagnostics) = self.map_transition(&boundary.graph, options)?;
        validate_mapped_cells(&netlist, &self.library).map_err(|e| diagnostics.failure(e))?;
        assemble(&mut netlist, &design, &registers, &boundary)
            .map_err(|e| diagnostics.failure(e))?;
        // Reparse emitted GV to exercise its syntax/form contract before
        // return.
        let text =
            emit_module_as_netlist_text(&netlist.modules[0], &netlist.nets, &netlist.interner)
                .map_err(|e| diagnostics.failure(e))?;
        let netlist = read_gv_from_str(&text).map_err(|e| diagnostics.failure(e))?;
        Ok(AbcMappedDesign {
            netlist,
            design,
            registers,
            cleanup_stats,
            diagnostics,
        })
    }

    fn map_transition(
        &self,
        graph: &GateFn,
        options: &AbcMappingOptions,
    ) -> Result<(ParsedNetlist, AbcRunDiagnostics), ToolError> {
        let constraints = &options.io_constraints;
        let mut diagnostics = AbcRunDiagnostics {
            executable: self.toolchain.path.clone(),
            version: self.toolchain.version.clone(),
            raw_liberty_files: self.raw_liberty_files.clone(),
            io_constraints: Some(constraints.clone()),
            ..Default::default()
        };
        if graph.outputs.is_empty() {
            // There is no observable transition logic for ABC to optimize.
            let inputs = graph
                .inputs
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>();
            let mut text = format!("module mapped({});\n", inputs.join(", "));
            for input in inputs {
                writeln!(text, "input {input};").unwrap();
            }
            text.push_str("endmodule\n");
            return Ok((
                read_gv_from_str(&text).map_err(|e| e.to_string())?,
                diagnostics,
            ));
        }
        let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
        let mut staged_libraries = Vec::with_capacity(self.raw_liberty_files.len());
        for (index, source) in self.raw_liberty_files.iter().enumerate() {
            let target = directory.path().join(format!("library_{index}.lib"));
            stage_library(source, &target)?;
            staged_libraries.push(target);
        }
        std::fs::write(
            directory.path().join("transition.aig"),
            emit_aiger_binary(graph, true)?,
        )
        .map_err(|e| e.to_string())?;
        std::fs::write(
            directory.path().join("timing.constr"),
            constraints.to_abc_text(),
        )
        .map_err(|e| e.to_string())?;
        diagnostics.script = mapping_script(self.raw_liberty_files.len(), &options.script);
        std::fs::write(directory.path().join("map.abc"), &diagnostics.script)
            .map_err(|e| e.to_string())?;
        let result = run_with_timeout_detailed(
            Command::new(self.toolchain.path())
                .current_dir(directory.path())
                .args(["-s", "-f", "map.abc"]),
            directory.path(),
            "abc",
            options.timeout,
        )
        .map_err(|e| e.with_context(format!("ABC program:\n{}", diagnostics.script)))?;
        diagnostics.stdout = String::from_utf8_lossy(&result.stdout).into_owned();
        diagnostics.stderr = String::from_utf8_lossy(&result.stderr).into_owned();
        if !result.status.success() {
            return Err(ToolError::from_exit_status(
                "ABC",
                result.status,
                &diagnostics.stdout,
                &diagnostics.stderr,
            ));
        }
        // ABC can warn and silently continue with a different driver model.
        // Treat that raw/proto mismatch as an error even when the process exits
        // 0.
        for output in [&diagnostics.stdout, &diagnostics.stderr] {
            if output.contains("Cannot find the default PI driving cell") {
                return Err(diagnostics.failure(format!(
                    "ABC could not use driving cell '{}'",
                    constraints.driving_cell
                )));
            }
        }
        let mapped = std::fs::read_to_string(directory.path().join("mapped.v"))
            .map_err(|e| diagnostics.failure(format!("ABC did not produce mapped.v: {e}")))?;
        let parsed = read_gv_from_str(&mapped)
            .map_err(|e| diagnostics.failure(format!("invalid ABC netlist: {e}")))?;
        validate_abc_source_formulas(&parsed, &staged_libraries)
            .map_err(|e| diagnostics.failure(e))?;
        Ok((parsed, diagnostics))
    }
}

/// Snapshots source libraries while avoiding interpolation of user paths.
fn stage_library(source: &Path, target: &Path) -> Result<(), String> {
    std::fs::copy(source, target)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Checks every source definition for each cell ABC actually emitted.
fn validate_abc_source_formulas(parsed: &ParsedNetlist, paths: &[PathBuf]) -> Result<(), String> {
    let cells = parsed
        .modules
        .iter()
        .flat_map(|module| &module.instances)
        .map(|instance| parsed.interner.resolve(instance.type_name).unwrap())
        .collect::<BTreeSet<_>>();
    if cells.is_empty() {
        return Ok(());
    }
    for path in paths {
        let issues = abc_source_formula_issues(path, &cells)?;
        if let Some(error) = issues.into_values().next() {
            return Err(error);
        }
    }
    Ok(())
}

/// Default optimization/mapping passes, including intermediate mapping and
/// final buffering/sizing. The wrapper loads libraries, a strashed input
/// network, and constraints before these commands and exports Verilog after.
pub const ABC_SPEED_SCRIPT: &str = r#"&get -n
&st
&dch
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&st
&syn2
&if -g -K 6
&synch2
&nf
&put
buffer -c
topo
stime -c
upsize -c
dnsize -c
"#;

/// Wraps caller-supplied passes with fixed library, input, constraint, and
/// output handling. The complete script is retained in run diagnostics.
fn mapping_script(libraries: usize, passes: &str) -> String {
    let first = libraries.min(2);
    let mut script = format!(
        "read_lib {}\n",
        (0..first)
            .map(|i| format!("library_{i}.lib"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for i in first..libraries {
        writeln!(script, "read_lib -m library_{i}.lib").unwrap();
    }
    script.push_str("read_aiger transition.aig\nstrash\nread_constr timing.constr\n");
    script.push_str(passes);
    // A final comment or command without a newline must not absorb the export.
    if !passes.ends_with('\n') {
        script.push('\n');
    }
    script.push_str("write_verilog mapped.v\n");
    script
}

/// A boundary endpoint either aliases an input net or consumes mapped logic.
#[derive(Clone, Debug, PartialEq, Eq)]
enum BoundarySignal {
    Direct(String),
    Mapped(String),
}

impl BoundarySignal {
    fn port_name(&self) -> &str {
        match self {
            Self::Direct(name) | Self::Mapped(name) => name,
        }
    }
}

/// Scalar ABC interface with direct wires kept outside ABC's output buffering.
struct Boundary {
    graph: GateFn,
    inputs: BTreeMap<(usize, usize), String>,
    external_outputs: Vec<Vec<BoundarySignal>>,
    register_data: Vec<BoundarySignal>,
}

impl Boundary {
    fn new(design: &SequentialGateFn, plan: &RegisterMappingPlan) -> Self {
        let mut graph = design.transition.clone();
        graph.name = "mapped".into();
        graph.inputs.clear();
        graph.outputs.clear();
        let mut phases = vec![false; graph.gates.len()];
        for bit in &plan.bits {
            let q = design.transition.inputs[bit.q.index()]
                .bit_vector
                .get_lsb(bit.bit_index);
            phases[q.node.id] = bit.output_inverted;
        }
        let rephase = |mut operand: AigOperand| {
            operand.negated ^= phases[operand.node.id];
            operand
        };
        // Rewrite every use, including feedback, while the input nodes now
        // denote physical FF outputs. No explicit inverter nodes are needed.
        for gate in &mut graph.gates {
            if let AigNode::And2 { a, b, .. } = gate {
                *a = rephase(*a);
                *b = rephase(*b);
            }
        }
        let mut inputs = BTreeMap::new();
        let mut input_nodes = BTreeMap::new();
        for (id, input) in design.transition.inputs.iter().enumerate() {
            for (bit, operand) in input.bit_vector.iter_lsb_to_msb().enumerate() {
                let name = format!("i{}", graph.inputs.len());
                graph.inputs.push(Input {
                    name: name.clone(),
                    bit_vector: AigBitVector::from_bit(*operand),
                });
                input_nodes.insert(operand.node.id, name.clone());
                inputs.insert((id, bit), name);
            }
        }
        let mut external_outputs = Vec::new();
        for id in &design.outputs {
            let mut names = Vec::new();
            for operand in design.transition.outputs[id.index()]
                .bit_vector
                .iter_lsb_to_msb()
            {
                names.push(add_boundary_signal(
                    &mut graph,
                    rephase(*operand),
                    &input_nodes,
                ));
            }
            external_outputs.push(names);
        }
        let mut register_data = Vec::new();
        for bit in &plan.bits {
            let mut operand = rephase(
                *design.transition.outputs[bit.d.index()]
                    .bit_vector
                    .get_lsb(bit.bit_index),
            );
            // Physical D = logical D XOR cell polarity XOR state phase.
            // `rephase` above expresses logical D using physical state inputs.
            operand.negated ^=
                plan.cells[bit.cell_index].inverts_data_to_output ^ bit.output_inverted;
            register_data.push(add_boundary_signal(&mut graph, operand, &input_nodes));
        }
        Self {
            graph,
            inputs,
            external_outputs,
            register_data,
        }
    }
}

fn add_boundary_signal(
    graph: &mut GateFn,
    operand: AigOperand,
    input_nodes: &BTreeMap<usize, String>,
) -> BoundarySignal {
    if !operand.negated {
        if let Some(input) = input_nodes.get(&operand.node.id) {
            // This connection needs no mapped cell. Exporting it as an ABC PO
            // would let the global output load introduce an unnecessary buffer.
            return BoundarySignal::Direct(input.clone());
        }
    }
    let name = format!("o{}", graph.outputs.len());
    graph.outputs.push(Output {
        name: name.clone(),
        bit_vector: AigBitVector::from_bit(operand),
    });
    BoundarySignal::Mapped(name)
}

fn simple_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Rejects unrepresentable names and collisions in the emitted port namespace.
fn validate_names(design: &SequentialGateFn) -> Result<(), String> {
    // Module names have their own namespace, and zero-width logical ports are
    // not emitted. Validate their spelling without reserving a physical port.
    let names = std::iter::once((design.name.as_str(), false))
        .chain(design.inputs.iter().map(|id| {
            let port = &design.transition.inputs[id.index()];
            (port.name.as_str(), port.get_bit_count() != 0)
        }))
        .chain(design.outputs.iter().map(|id| {
            let port = &design.transition.outputs[id.index()];
            (port.name.as_str(), port.get_bit_count() != 0)
        }))
        .chain(design.clock.iter().map(|c| (c.name.as_str(), true)));
    let mut physical_ports = BTreeSet::new();
    for (name, physical_port) in names {
        if name.is_empty() || name.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(format!(
                "name cannot be represented as a Verilog identifier: {name:?}"
            ));
        }
        if physical_port && !physical_ports.insert(name) {
            return Err(format!("duplicate external port or clock name: {name:?}"));
        }
    }
    Ok(())
}

/// Checks obvious raw/proto divergence, without asserting semantic identity.
fn validate_mapped_cells(parsed: &ParsedNetlist, library: &Library) -> Result<(), String> {
    if parsed.modules.len() != 1 {
        return Err("ABC must emit exactly one combinational module".into());
    }
    let cells = library
        .cells
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect::<BTreeMap<_, _>>();
    for instance in &parsed.modules[0].instances {
        let name = parsed.interner.resolve(instance.type_name).unwrap();
        let cell = cells
            .get(name)
            .ok_or_else(|| format!("ABC emitted cell absent from native Liberty: {name}"))?;
        cell.validate_functional_semantics()?;
        cell.validate_mapping_policy_coverage()?;
        // ABC does not report whether a cell was introduced by sizing, so a
        // sizing-only cell could have been selected during initial mapping.
        if cell.use_for_size_only == Some(true) {
            return Err(format!("ABC emitted cell marked use_for_size_only: {name}"));
        }
        if !cell.sequential.is_empty() || cell.dont_use == Some(true) || cell.clock_gate.is_some() {
            return Err(format!(
                "ABC emitted unavailable/non-combinational cell: {name}"
            ));
        }
        let mut connected = BTreeSet::new();
        for (pin, _) in &instance.connections {
            let pin = parsed.interner.resolve(*pin).unwrap();
            if !connected.insert(pin)
                || !cell
                    .pins
                    .iter()
                    .any(|p| library.resolve_string(&p.name) == pin)
            {
                return Err(format!("ABC emitted unknown/duplicate pin {name}.{pin}"));
            }
        }
        for pin in &cell.pins {
            let pin_name = library.resolve_string(&pin.name);
            if pin.direction == PinDirection::Input as i32 && !connected.contains(pin_name) {
                return Err(format!("ABC omitted input pin on {name}"));
            }
        }
    }
    Ok(())
}

/// Reuses ABC's flat netlist, replacing its interface with native ports and
/// FFs.
fn assemble(
    parsed: &mut ParsedNetlist,
    design: &SequentialGateFn,
    plan: &RegisterMappingPlan,
    boundary: &Boundary,
) -> Result<(), String> {
    let module = &parsed.modules[0];
    let mut ports = BTreeMap::new();
    for port in &module.ports {
        let name = parsed.interner.resolve(port.name).unwrap().to_string();
        if port.width.is_some() {
            return Err("ABC boundary ports must be scalar".into());
        }
        let net = module
            .find_net_index(port.name, &parsed.nets)
            .ok_or("ABC port net missing")?;
        ports.insert(name, (port.direction.clone(), net));
    }
    let expected = boundary
        .graph
        .inputs
        .iter()
        .map(|p| (p.name.clone(), PortDirection::Input))
        .chain(
            boundary
                .graph
                .outputs
                .iter()
                .map(|p| (p.name.clone(), PortDirection::Output)),
        )
        .collect::<BTreeMap<_, _>>();
    if ports
        .iter()
        .map(|(n, (d, _))| (n.clone(), d.clone()))
        .collect::<BTreeMap<_, _>>()
        != expected
    {
        return Err("ABC changed the scalar boundary names, directions, or counts".into());
    }
    let mut used = design
        .inputs
        .iter()
        .map(|id| design.transition.inputs[id.index()].name.clone())
        .chain(
            design
                .outputs
                .iter()
                .map(|id| design.transition.outputs[id.index()].name.clone()),
        )
        .chain(design.clock.iter().map(|c| c.name.clone()))
        .collect::<BTreeSet<_>>();
    for (i, net) in parsed.nets.iter_mut().enumerate() {
        net.name = parsed
            .interner
            .get_or_intern(fresh_name(&mut used, format!("__abc_n{i}")));
    }
    let module = &mut parsed.modules[0];
    module.name = parsed.interner.get_or_intern(&design.name);
    module.ports.clear();
    module.wires = (0..parsed.nets.len()).map(NetIndex).collect();
    for (i, instance) in module.instances.iter_mut().enumerate() {
        instance.instance_name = parsed
            .interner
            .get_or_intern(fresh_name(&mut used, format!("__abc_c{i}")));
    }
    for id in &design.inputs {
        let input = &design.transition.inputs[id.index()];
        if let Some(net) = add_port(
            parsed,
            &input.name,
            input.get_bit_count(),
            PortDirection::Input,
        )? {
            for bit in 0..input.get_bit_count() {
                add_alias(
                    parsed,
                    NetRef::Simple(ports[&boundary.inputs[&(id.index(), bit)]].1),
                    bit_ref(net, bit, input.get_bit_count()),
                );
            }
        }
    }
    for (index, id) in design.outputs.iter().enumerate() {
        let output = &design.transition.outputs[id.index()];
        if let Some(net) = add_port(
            parsed,
            &output.name,
            output.get_bit_count(),
            PortDirection::Output,
        )? {
            for bit in 0..output.get_bit_count() {
                add_alias(
                    parsed,
                    bit_ref(net, bit, output.get_bit_count()),
                    NetRef::Simple(ports[boundary.external_outputs[index][bit].port_name()].1),
                );
            }
        }
    }
    let clock = design
        .clock
        .as_ref()
        .map(|clock| add_port(parsed, &clock.name, 1, PortDirection::Input))
        .transpose()?
        .flatten();
    for (index, bit) in plan.bits.iter().enumerate() {
        let cell = &plan.cells[bit.cell_index];
        let mut connections = vec![
            (
                parsed.interner.get_or_intern(&cell.clock_pin),
                NetRef::Simple(clock.ok_or("mapped FF has no clock")?),
            ),
            (
                parsed.interner.get_or_intern(&cell.data_pin),
                NetRef::Simple(ports[boundary.register_data[index].port_name()].1),
            ),
            (
                parsed.interner.get_or_intern(&cell.output_pin),
                NetRef::Simple(ports[&boundary.inputs[&(bit.q.index(), bit.bit_index)]].1),
            ),
        ];
        for (pin, value) in &cell.tied_inputs {
            connections.push((
                parsed.interner.get_or_intern(pin),
                NetRef::Literal(IrBits::from_lsb_is_0(&[*value])),
            ));
        }
        for pin in &cell.unused_outputs {
            connections.push((parsed.interner.get_or_intern(pin), NetRef::Unconnected));
        }
        parsed.modules[0].instances.push(NetlistInstance {
            type_name: parsed.interner.get_or_intern(&cell.cell_name),
            instance_name: parsed
                .interner
                .get_or_intern(fresh_name(&mut used, format!("__reg_{index}"))),
            connections,
            inst_lineno: 0,
            inst_colno: 0,
        });
    }
    parsed.modules[0].net_index_range = 0..parsed.nets.len();
    Ok(())
}

fn fresh_name(used: &mut BTreeSet<String>, base: String) -> String {
    let mut name = base.clone();
    let mut suffix = 0;
    while !used.insert(name.clone()) {
        suffix += 1;
        name = format!("{base}_{suffix}");
    }
    name
}

fn add_port(
    parsed: &mut ParsedNetlist,
    name: &str,
    width: usize,
    direction: PortDirection,
) -> Result<Option<NetIndex>, String> {
    if width == 0 {
        return Ok(None);
    }
    let width = if width == 1 {
        None
    } else {
        Some((
            u32::try_from(width - 1).map_err(|_| "port width exceeds Verilog index range")?,
            0,
        ))
    };
    let symbol = parsed.interner.get_or_intern(name);
    let index = NetIndex(parsed.nets.len());
    parsed.nets.push(Net {
        name: symbol,
        width,
    });
    parsed.modules[0].ports.push(NetlistPort {
        direction,
        width,
        name: symbol,
    });
    Ok(Some(index))
}

fn bit_ref(net: NetIndex, bit: usize, width: usize) -> NetRef {
    if width == 1 {
        NetRef::Simple(net)
    } else {
        NetRef::BitSelect(net, bit as u32)
    }
}

fn add_alias(parsed: &mut ParsedNetlist, lhs: NetRef, rhs: NetRef) {
    parsed.modules[0].assigns.push(NetlistAssign {
        kind: NetlistAssignKind::Continuous,
        lhs,
        rhs: AssignExpr::Leaf(rhs),
        span: Span {
            start: Pos {
                lineno: 0,
                colno: 0,
            },
            limit: Pos {
                lineno: 0,
                colno: 0,
            },
        },
    });
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod mapping_policy_tests {
    use super::*;
    use crate::aig::{GateBuilder, GateBuilderOptions};
    use crate::liberty::model::{library_from_proto, library_to_proto};
    use crate::liberty::parser::parse_liberty_files;
    use prost::Message;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn rejects_sizing_only_abc_cells_after_proto_roundtrip() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            r#"library(test) {
                cell(SIZE_ONLY) { area: 1; use_for_size_only: true;
                    pin(A) { direction: input; }
                    pin(Y) { direction: output; function: "A"; }
                }
                cell(ALLOWED) { area: 2; use_for_size_only: false;
                    pin(A) { direction: input; }
                    pin(Y) { direction: output; function: "A"; }
                }
                cell(UNSPECIFIED) { area: 3;
                    pin(A) { direction: input; }
                    pin(Y) { direction: output; function: "A"; }
                }
            }"#,
        )
        .unwrap();
        let wire = library_to_proto(parse_liberty_files(&[file.path()]).unwrap()).unwrap();
        let library = library_from_proto(
            crate::liberty_proto::Library::decode(wire.encode_to_vec().as_slice()).unwrap(),
        )
        .unwrap();
        for cell in ["ALLOWED", "UNSPECIFIED"] {
            let parsed = read_gv_from_str(&format!(
                "module m(a,y); input a; output y; {cell} c(.A(a),.Y(y)); endmodule"
            ))
            .unwrap();
            validate_mapped_cells(&parsed, &library).unwrap();
        }
        let parsed = read_gv_from_str(
            "module m(a,y); input a; output y; SIZE_ONLY c(.A(a),.Y(y)); endmodule",
        )
        .unwrap();
        assert_eq!(
            validate_mapped_cells(&parsed, &library).unwrap_err(),
            "ABC emitted cell marked use_for_size_only: SIZE_ONLY"
        );
    }

    #[cfg(unix)]
    #[test]
    fn map_sequential_rejects_sizing_only_cells_emitted_by_abc() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("cells.lib");
        std::fs::write(
            &raw,
            r#"library(test) {
                cell(BUF) { pin(A) { direction: input; }
                    pin(Y) { direction: output; function: "A"; } }
                cell(SIZE_ONLY) { use_for_size_only: true;
                    pin(A) { direction: input; }
                    pin(Y) { direction: output; function: "!A"; } }
            }"#,
        )
        .unwrap();
        let library = parse_liberty_files(&[&raw]).unwrap();
        let executable = dir.path().join("fake-abc");
        std::fs::write(
            &executable,
            r#"#!/bin/sh
if [ "$2" = "-c" ]; then exit 0; fi
cat > mapped.v <<'EOF'
module mapped(i0,o0);
input i0; output o0;
SIZE_ONLY inst(.A(i0),.Y(o0));
endmodule
EOF
"#,
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let context =
            AbcMappingContext::new(AbcToolchain::new(&executable).unwrap(), &[raw], library)
                .unwrap();
        let mut builder = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".into(), 1);
        builder.add_output("y".into(), AigBitVector::from_bit(a.get_lsb(0).negate()));
        let design = SequentialGateFn::from_gate_fn(builder.build());
        let options = AbcMappingOptions::new(AbcIoConstraints {
            driving_cell: "BUF".into(),
            output_load_ff: 20.0,
        });
        let error = context
            .map_sequential(&design, &options)
            .err()
            .expect("ABC's sizing-only cell must be rejected");
        assert_eq!(
            error.to_string().lines().next().unwrap(),
            "ABC emitted cell marked use_for_size_only: SIZE_ONLY"
        );
    }
}

#[cfg(test)]
mod formula_precedence_tests {
    use super::*;
    use crate::aig::{GateBuilder, GateBuilderOptions};
    use crate::liberty::parser::parse_liberty_files;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn mapped_output_rejects_abc_precedence_differences() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let mapped = read_gv_from_str(
            "module m(a,b,c,y); input a,b,c; output y; MIX inst(.A(a),.B(b),.C(c),.Y(y)); endmodule",
        )
        .unwrap();
        for (formula, reject) in [
            ("A*B^C", true),
            ("A^B*C", true),
            ("A B^C", true),
            ("A^B C", true),
            ("A*(B^C)", false),
            ("(A*B)^C", false),
            ("A^(B*C)", false),
            ("(A^B)*C", false),
            ("A+B^C", false),
        ] {
            std::fs::write(
                file.path(),
                format!(
                    r#"library(test) {{
                        cell(MIX) {{
                            pin(A, B, C) {{ direction: input; }}
                            pin(Y) {{ direction: output; function: "{formula}"; }}
                        }}
                        cell(UNUSED) {{ pin(Y) {{ direction: output; function: "@"; }} }}
                    }}"#
                ),
            )
            .unwrap();
            // Grouped input pins cause the importer to parenthesize all cell
            // formulas, so this must examine the source, not the model.
            let library = parse_liberty_files(&[file.path()]).unwrap();
            let normalized = library
                .cells
                .iter()
                .find(|cell| cell.name == "MIX")
                .unwrap();
            let output = normalized
                .pins
                .iter()
                .find(|pin| library.resolve_string(&pin.name) == "Y")
                .unwrap();
            assert!(
                !crate::liberty::cell_formula::abc_formula_has_different_parse(
                    library.resolve_string(&output.function)
                )
                .unwrap()
            );
            let result = validate_abc_source_formulas(&mapped, &[file.path().to_path_buf()]);
            if reject {
                assert_eq!(
                    result.unwrap_err(),
                    format!(
                        "ABC may interpret AND/XOR precedence differently for MIX.Y: {formula:?}"
                    ),
                );
            } else {
                result.unwrap();
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn map_sequential_rejects_abc_mixed_precedence_cell() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("cells.lib");
        std::fs::write(
            &raw,
            r#"library(test) {
                cell(BUF) { pin(A) { direction: input; }
                    pin(Y) { direction: output; function: "A"; } }
                cell(MIX) { pin(A, B, C) { direction: input; }
                    pin(Y) { direction: output; function: "A*B^C"; } }
            }"#,
        )
        .unwrap();
        let library = parse_liberty_files(&[&raw]).unwrap();
        let executable = dir.path().join("fake-abc");
        std::fs::write(
            &executable,
            r#"#!/bin/sh
if [ "$2" = "-c" ]; then exit 0; fi
cat > mapped.v <<'EOF'
module mapped(i0,i1,i2,o0);
input i0,i1,i2; output o0;
MIX inst(.A(i0),.B(i1),.C(i2),.Y(o0));
endmodule
EOF
"#,
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let context =
            AbcMappingContext::new(AbcToolchain::new(&executable).unwrap(), &[raw], library)
                .unwrap();
        let mut builder = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".into(), 1);
        let b = builder.add_input("b".into(), 1);
        let c = builder.add_input("c".into(), 1);
        let ab = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        let output = builder.add_xor_binary(ab, *c.get_lsb(0));
        builder.add_output("y".into(), AigBitVector::from_bit(output));
        let design = SequentialGateFn::from_gate_fn(builder.build());
        let options = AbcMappingOptions::new(AbcIoConstraints {
            driving_cell: "BUF".into(),
            output_load_ff: 20.0,
        });
        let error = context
            .map_sequential(&design, &options)
            .err()
            .expect("ambiguous ABC formula must be rejected");
        assert_eq!(
            error.to_string().lines().next().unwrap(),
            "ABC may interpret AND/XOR precedence differently for MIX.Y: \"A*B^C\""
        );
    }

    #[test]
    fn source_check_includes_duplicate_cells_in_every_library_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.lib");
        let bad = dir.path().join("bad.lib");
        let snapshot = dir.path().join("snapshot.lib");
        let source = |formula: &str| {
            format!(
                r#"library(test) {{ cell(MIX) {{
                    pin(Y) {{ direction: output; function: "{formula}"; }}
                }} }}"#
            )
        };
        std::fs::write(&good, source("(A*B)^C")).unwrap();
        std::fs::write(&bad, source("A*B^C")).unwrap();
        stage_library(&bad, &snapshot).unwrap();
        std::fs::write(&bad, source("(A*B)^C")).unwrap();
        let mapped = read_gv_from_str("module m(y); output y; MIX inst(.Y(y)); endmodule").unwrap();
        assert_eq!(
            validate_abc_source_formulas(&mapped, &[good, snapshot]).unwrap_err(),
            "ABC may interpret AND/XOR precedence differently for MIX.Y: \"A*B^C\""
        );
    }
}
