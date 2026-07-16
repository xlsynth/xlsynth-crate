// SPDX-License-Identifier: Apache-2.0

//! Project a parsed netlist and Liberty proto into a GateFn.

use crate::aig::gate::{AigBitVector, AigOperand, GateFn};
use crate::aig::sequential_gate::{canonical_register_d_name, canonical_register_q_name};
use crate::aig::{
    ClockPort, RegisterBinding, SequentialGateFn, TransitionInputId, TransitionOutputId,
};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::liberty::cell_formula::Term;
use crate::liberty_model::{Cell, Library, PinDirection, SequentialKind};
use crate::netlist::normalized::{
    BitExpr, BitIndex, BitSource, NormalizedConnection, NormalizedInstance, NormalizedNetlistModule,
};
use crate::netlist::parse::{Net, NetlistModule, PortDirection};
use crate::netlist::sequential_liberty::{
    GvEvalSequentialCellSpec, get_gv_eval_sequential_cell_spec,
};
use std::collections::HashMap;
use std::collections::HashSet;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// A combinational netlist projected into one AIG with source-level labels.
#[derive(Debug, Clone)]
pub struct LabeledNetlistAig {
    pub module_name: String,
    pub gate_fn: GateFn,
    pub module_ports: Vec<ModulePortAigBinding>,
    pub instances: Vec<InstanceAigBinding>,
}

/// One module-port bundle bound to AIG operands in LSb-to-MSb order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePortAigBinding {
    pub name: String,
    pub direction: PortDirection,
    pub bits_lsb_to_msb: Vec<LabeledAigBit>,
}

/// One named Verilog bit bound to an AIG operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabeledAigBit {
    pub bit_number: u32,
    pub operand: AigOperand,
}

/// One standard-cell instance and its explicitly connected external pins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceAigBinding {
    pub instance_name: String,
    pub cell_type: String,
    pub pins: Vec<InstancePinAigBinding>,
}

/// One scalar standard-cell pin bound to its signal in the global AIG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstancePinAigBinding {
    pub pin_name: String,
    pub direction: PinDirection,
    pub operand: AigOperand,
    pub connection: PinConnection,
}

/// The source-level connection attached to one scalar standard-cell pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinConnection {
    Net { net_name: String, bit_number: u32 },
    Literal(bool),
    Unconnected,
}

/// Active edge observed by a labeled sequential netlist clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequentialClockEdge {
    Rising,
    Falling,
}

/// One top-level clock omitted from the transition function's data inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledSequentialClock {
    pub port_name: String,
    /// Unknown when a clock-port hint preserves an otherwise optimized-away
    /// clock and no FF remains from which to infer polarity.
    pub active_edge: Option<SequentialClockEdge>,
}

/// A sequential pin or port signal in the transition graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequentialAigSignal {
    Operand(AigOperand),
    Clock,
}

/// One named Verilog bit bound to a sequential transition signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabeledSequentialAigBit {
    pub bit_number: u32,
    pub signal: SequentialAigSignal,
}

/// One module-port bundle bound to sequential transition signals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialModulePortAigBinding {
    pub name: String,
    pub direction: PortDirection,
    pub bits_lsb_to_msb: Vec<LabeledSequentialAigBit>,
}

/// One scalar standard-cell pin in a labeled sequential transition graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialInstancePinAigBinding {
    pub pin_name: String,
    pub direction: PinDirection,
    pub signal: SequentialAigSignal,
    pub connection: PinConnection,
}

/// One mapped standard-cell instance in a labeled sequential transition graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialInstanceAigBinding {
    pub instance_name: String,
    pub cell_type: String,
    pub pins: Vec<SequentialInstancePinAigBinding>,
    /// Index in [`SequentialGateFn::registers`] for an FF instance.
    pub state_register_index: Option<usize>,
}

/// A stateful netlist projected into a transition AIG while preserving mapped
/// module, instance, pin, and connection labels.
#[derive(Debug, Clone)]
pub struct LabeledSequentialNetlistAig {
    pub module_name: String,
    pub sequential_gate_fn: SequentialGateFn,
    pub clock: Option<LabeledSequentialClock>,
    pub module_ports: Vec<SequentialModulePortAigBinding>,
    pub instances: Vec<SequentialInstanceAigBinding>,
}

fn substitute_state_vars_in_term(
    term: &Term,
    sequential_terms: &HashMap<String, (Term, String)>,
) -> Term {
    match term {
        Term::Input(name) => sequential_terms
            .get(name)
            .map(|(replacement, _)| replacement.clone())
            .unwrap_or_else(|| Term::Input(name.clone())),
        Term::And(lhs, rhs) => Term::And(
            Box::new(substitute_state_vars_in_term(lhs, sequential_terms)),
            Box::new(substitute_state_vars_in_term(rhs, sequential_terms)),
        ),
        Term::Or(lhs, rhs) => Term::Or(
            Box::new(substitute_state_vars_in_term(lhs, sequential_terms)),
            Box::new(substitute_state_vars_in_term(rhs, sequential_terms)),
        ),
        Term::Xor(lhs, rhs) => Term::Xor(
            Box::new(substitute_state_vars_in_term(lhs, sequential_terms)),
            Box::new(substitute_state_vars_in_term(rhs, sequential_terms)),
        ),
        Term::Negate(inner) => Term::Negate(Box::new(substitute_state_vars_in_term(
            inner,
            sequential_terms,
        ))),
        Term::Constant(value) => Term::Constant(*value),
    }
}

fn build_cell_formula_map(
    liberty_lib: &Library,
    collapse_sequential: bool,
    used_cells: &HashSet<String>,
    dff_cells_identity: &HashSet<String>,
    dff_cells_inverted: &HashSet<String>,
) -> Result<HashMap<(String, String), (Term, String)>, String> {
    let mut cell_formula_map = HashMap::new();
    let override_cells: HashSet<String> = dff_cells_identity
        .iter()
        .cloned()
        .chain(dff_cells_inverted.iter().cloned())
        .collect();
    for cell in &liberty_lib.cells {
        if !used_cells.contains(&cell.name) {
            continue;
        }
        let collapse_for_cell = collapse_sequential && !override_cells.contains(&cell.name);
        let pin_names: HashSet<String> = cell
            .pins
            .iter()
            .map(|pin| liberty_lib.resolve_string(&pin.name).to_string())
            .collect();
        let state_vars: HashSet<String> = cell
            .sequential
            .iter()
            .flat_map(|seq| {
                std::iter::once(seq.state_var.clone())
                    .chain(seq.complementary_state_var.iter().cloned())
            })
            .collect();
        let mut sequential_terms: HashMap<String, (crate::liberty::cell_formula::Term, String)> =
            HashMap::new();
        if collapse_for_cell {
            for seq in &cell.sequential {
                if seq.kind == crate::liberty_model::SequentialKind::Latch as i32
                    && !seq.clock_expr.is_empty()
                {
                    return Err(format!(
                        "collapse_sequential does not support latches with enable; cell '{}' state '{}' enable '{}'",
                        cell.name, seq.state_var, seq.clock_expr
                    ));
                }
                if seq.kind == crate::liberty_model::SequentialKind::Ff as i32
                    && (!seq.clear_expr.is_empty() || !seq.preset_expr.is_empty())
                {
                    return Err(format!(
                        "collapse_sequential does not support flops with async clear/preset; cell '{}' state '{}' clear '{}' preset '{}'",
                        cell.name, seq.state_var, seq.clear_expr, seq.preset_expr
                    ));
                }
                if seq.state_var.is_empty() || seq.next_state.is_empty() {
                    continue;
                }
                let term = crate::liberty::cell_formula::parse_formula(&seq.next_state).map_err(
                    |e| {
                        format!(
                            "Failed to parse next_state for cell '{}' state '{}' (next_state: \"{}\"): {}",
                            cell.name, seq.state_var, seq.next_state, e
                        )
                    },
                )?;
                let inputs = term.inputs();
                let mut state_refs: Vec<String> = inputs
                    .iter()
                    .filter(|name| state_vars.contains((*name).as_str()))
                    .cloned()
                    .collect();
                state_refs.sort();
                state_refs.dedup();
                if !state_refs.is_empty() {
                    return Err(format!(
                        "collapse_sequential does not support next_state formulas that reference state variables; cell '{}' state '{}' references [{}] in next_state \"{}\"",
                        cell.name,
                        seq.state_var,
                        state_refs.join(", "),
                        seq.next_state
                    ));
                }
                let mut non_pin_inputs: Vec<String> = inputs
                    .iter()
                    .filter(|name| !pin_names.contains((*name).as_str()))
                    .cloned()
                    .collect();
                non_pin_inputs.sort();
                non_pin_inputs.dedup();
                if !non_pin_inputs.is_empty() {
                    return Err(format!(
                        "collapse_sequential requires next_state inputs to be cell pins; cell '{}' state '{}' references non-pin inputs [{}] in next_state \"{}\"",
                        cell.name,
                        seq.state_var,
                        non_pin_inputs.join(", "),
                        seq.next_state
                    ));
                }
                sequential_terms.insert(
                    seq.state_var.clone(),
                    (term.clone(), seq.next_state.clone()),
                );
                if let Some(complementary_state_var) = seq
                    .complementary_state_var
                    .as_ref()
                    .filter(|name| !name.is_empty())
                {
                    sequential_terms.insert(
                        complementary_state_var.clone(),
                        (
                            crate::liberty::cell_formula::Term::Negate(Box::new(term)),
                            format!("!({})", seq.next_state),
                        ),
                    );
                }
            }
        }
        for pin in &cell.pins {
            let pin_name = liberty_lib.resolve_string(&pin.name);
            let function = liberty_lib.resolve_string(&pin.function);
            if pin.direction == 1 && !function.is_empty() {
                let original_formula_string = function.to_string();
                match crate::liberty::cell_formula::parse_formula(function) {
                    Ok(term) => {
                        let term = if collapse_for_cell {
                            let replaced = substitute_state_vars_in_term(&term, &sequential_terms);
                            let mut remaining_state_refs: Vec<String> = replaced
                                .inputs()
                                .into_iter()
                                .filter(|name| state_vars.contains(name))
                                .collect();
                            remaining_state_refs.sort();
                            remaining_state_refs.dedup();
                            if !remaining_state_refs.is_empty() {
                                return Err(format!(
                                    "collapse_sequential could not safely collapse state variables [{}] for cell '{}' output pin '{}' (function \"{}\")",
                                    remaining_state_refs.join(", "),
                                    cell.name,
                                    pin_name,
                                    function
                                ));
                            }
                            replaced
                        } else {
                            term
                        };
                        cell_formula_map.insert(
                            (cell.name.clone(), pin_name.to_string()),
                            (term, original_formula_string),
                        );
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to parse formula for cell '{}', pin '{}' (formula: \"{}\"): {}",
                            cell.name,
                            pin_name,
                            original_formula_string,
                            e
                        );
                    }
                }
            }
        }
    }
    Ok(cell_formula_map)
}

fn check_undriven_bits(
    used_as_input: &HashSet<BitIndex>,
    driven: &HashSet<BitIndex>,
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<(), String> {
    for bit_idx in used_as_input {
        if !driven.contains(bit_idx) {
            return Err(format!(
                "net bit '{}' is used as an input but is never driven by any instance, continuous assign, or module input",
                normalized.render_bit(*bit_idx, nets, interner)
            ));
        }
    }
    Ok(())
}

struct ResolvedBitValues {
    values: Vec<Option<AigOperand>>,
}

#[derive(Clone, Copy, Debug)]
struct PendingAssignBit {
    assign_index: usize,
    rhs_bit_index: usize,
    target: BitIndex,
}

impl ResolvedBitValues {
    fn new(bit_count: usize) -> Self {
        Self {
            values: vec![None; bit_count],
        }
    }

    fn resolve_bit(&self, bit_idx: BitIndex) -> Option<AigOperand> {
        self.values[bit_idx]
    }

    fn resolve_source(
        &self,
        source: BitSource,
        gb: &mut GateBuilder,
    ) -> Result<Option<AigOperand>, String> {
        match source {
            BitSource::Bit(bit_idx) => Ok(self.resolve_bit(bit_idx)),
            BitSource::Literal(value) => {
                Ok(Some(if value { gb.get_true() } else { gb.get_false() }))
            }
            BitSource::Unknown => Err("unknown literal net reference is not supported".to_string()),
        }
    }

    fn materialize_sources(
        &self,
        sources: &[BitSource],
        gb: &mut GateBuilder,
    ) -> Result<Option<AigBitVector>, String> {
        if sources.is_empty() {
            return Ok(None);
        }
        let mut bits = Vec::with_capacity(sources.len());
        for source in sources {
            let Some(bit) = self.resolve_source(*source, gb)? else {
                return Ok(None);
            };
            bits.push(bit);
        }
        Ok(Some(AigBitVector::from_lsb_is_index_0(&bits)))
    }

    fn write_bit(
        &mut self,
        bit_idx: BitIndex,
        value: AigOperand,
        normalized: &NormalizedNetlistModule<'_>,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), String> {
        if self.values[bit_idx].is_some() {
            let bit = normalized.bit(bit_idx);
            return Err(format!(
                "net '{}' bit {} was assigned more than once during projection",
                crate::netlist::bit_ref::net_name(bit.net, nets, interner),
                bit.bit_number
            ));
        }
        self.values[bit_idx] = Some(value);
        Ok(())
    }

    fn write_bits(
        &mut self,
        targets: &[BitIndex],
        src_bv: &AigBitVector,
        normalized: &NormalizedNetlistModule<'_>,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
    ) -> Result<(), String> {
        if targets.is_empty() {
            return Ok(());
        }
        if src_bv.get_bit_count() != targets.len() {
            return Err(format!(
                "width mismatch assigning to '{}': expected {} bits but got {}",
                normalized.render_bit(targets[0], nets, interner),
                targets.len(),
                src_bv.get_bit_count()
            ));
        }
        for (offset, target) in targets.iter().copied().enumerate() {
            self.write_bit(target, *src_bv.get_lsb(offset), normalized, nets, interner)?;
        }
        Ok(())
    }

    fn materialize_output_or_false(
        &self,
        output_bits: &[BitIndex],
        gb: &mut GateBuilder,
    ) -> AigBitVector {
        let bits = output_bits
            .iter()
            .map(|bit_idx| self.resolve_bit(*bit_idx).unwrap_or_else(|| gb.get_false()))
            .collect::<Vec<_>>();
        AigBitVector::from_lsb_is_index_0(&bits)
    }
}

fn collect_expr_source_bits(expr: &BitExpr, out: &mut HashSet<BitIndex>) {
    let mut bits = Vec::new();
    expr.collect_source_bits(&mut bits);
    out.extend(bits);
}

fn output_target_bits(connection: &NormalizedConnection) -> Result<Vec<BitIndex>, String> {
    connection
        .bits
        .iter()
        .map(|source| match source {
            BitSource::Bit(bit_idx) => Ok(*bit_idx),
            BitSource::Literal(_) | BitSource::Unknown => {
                Err("output destination cannot be a literal or unknown".to_string())
            }
        })
        .collect()
}

fn eval_bit_expr(
    expr: &BitExpr,
    resolved: &ResolvedBitValues,
    gb: &mut GateBuilder,
) -> Result<Option<AigOperand>, String> {
    match expr {
        BitExpr::Source(source) => resolved.resolve_source(*source, gb),
        BitExpr::Not(inner) => {
            let Some(value) = eval_bit_expr(inner, resolved, gb)? else {
                return Ok(None);
            };
            Ok(Some(gb.add_not(value)))
        }
        BitExpr::And(lhs, rhs) | BitExpr::Or(lhs, rhs) | BitExpr::Xor(lhs, rhs) => {
            let Some(lhs_value) = eval_bit_expr(lhs, resolved, gb)? else {
                return Ok(None);
            };
            let Some(rhs_value) = eval_bit_expr(rhs, resolved, gb)? else {
                return Ok(None);
            };
            Ok(Some(match expr {
                BitExpr::And(_, _) => gb.add_and_binary(lhs_value, rhs_value),
                BitExpr::Or(_, _) => gb.add_or_binary(lhs_value, rhs_value),
                BitExpr::Xor(_, _) => gb.add_xor_binary(lhs_value, rhs_value),
                BitExpr::Source(_) | BitExpr::Not(_) => unreachable!(),
            }))
        }
    }
}

fn collect_missing_expr_sources(
    expr: &BitExpr,
    resolved: &ResolvedBitValues,
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    out: &mut Vec<String>,
) {
    match expr {
        BitExpr::Source(BitSource::Bit(bit_idx)) => {
            if resolved.resolve_bit(*bit_idx).is_none() {
                out.push(normalized.render_bit(*bit_idx, nets, interner));
            }
        }
        BitExpr::Source(BitSource::Literal(_) | BitSource::Unknown) => {}
        BitExpr::Not(inner) => {
            collect_missing_expr_sources(inner, resolved, normalized, nets, interner, out)
        }
        BitExpr::And(lhs, rhs) | BitExpr::Or(lhs, rhs) | BitExpr::Xor(lhs, rhs) => {
            collect_missing_expr_sources(lhs, resolved, normalized, nets, interner, out);
            collect_missing_expr_sources(rhs, resolved, normalized, nets, interner, out);
        }
    }
}

fn process_instance_outputs(
    inst: &NormalizedInstance,
    type_name: &str,
    inst_name: &str,
    pin_directions: &HashMap<&str, i32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    gb: &mut GateBuilder,
    resolved: &mut ResolvedBitValues,
    normalized: &NormalizedNetlistModule<'_>,
    dff_cells_identity: &std::collections::HashSet<String>,
    dff_cells_inverted: &std::collections::HashSet<String>,
    cell_formula_map: &HashMap<(String, String), (crate::liberty::cell_formula::Term, String)>,
    input_map: &HashMap<String, AigOperand>,
    port_map: &HashMap<String, String>,
) -> Result<ProcessedInstanceOutputs, String> {
    let mut processed_any_output = false;
    let mut output_operands = HashMap::new();
    for connection in &inst.connections {
        let port_name = interner.resolve(connection.port).unwrap();
        let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
        if pin_dir == 1 {
            if handle_dff_identity_override(
                type_name,
                inst_name,
                port_name,
                inst,
                interner,
                gb,
                resolved,
                normalized,
                dff_cells_identity,
                dff_cells_inverted,
                nets,
            )? {
                processed_any_output = true;
                continue;
            }
            let key = (type_name.to_string(), port_name.to_string());
            let (formula_ast, original_formula_str) = match cell_formula_map.get(&key) {
                Some(x) => x,
                None => continue,
            };
            let context = crate::liberty::cell_formula::EmitContext {
                cell_name: type_name,
                original_formula: original_formula_str.as_str(),
                instance_name: Some(inst_name),
                port_map: Some(port_map),
            };
            let out_op = formula_ast
                .emit_formula_term(gb, input_map, &context)
                .map_err(|e| {
                    format!(
                        "Failed to emit formula for cell '{}' instance '{}' pin '{}': {}",
                        type_name, inst_name, port_name, e
                    )
                })?;
            output_operands.insert(port_name.to_string(), out_op);
            let src_bv = AigBitVector::from_bit(out_op);
            let output_bits = output_target_bits(connection)?;
            resolved.write_bits(output_bits.as_slice(), &src_bv, normalized, nets, interner)?;
            processed_any_output = true;
        }
    }
    Ok(ProcessedInstanceOutputs {
        processed_any_output,
        output_operands,
    })
}

struct ProcessedInstanceOutputs {
    processed_any_output: bool,
    output_operands: HashMap<String, AigOperand>,
}

pub fn project_gatefn_from_netlist_and_liberty(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
    dff_cells_identity: &std::collections::HashSet<String>,
    dff_cells_inverted: &std::collections::HashSet<String>,
) -> Result<GateFn, String> {
    let options = GateFnProjectOptions {
        collapse_sequential: false,
    };
    project_gatefn_from_netlist_and_liberty_with_options(
        module,
        nets,
        interner,
        liberty_lib,
        dff_cells_identity,
        dff_cells_inverted,
        &options,
    )
}

#[derive(Debug, Clone)]
pub struct GateFnProjectOptions {
    /// If true, collapse sequential state variables by substituting next_state.
    pub collapse_sequential: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionMode {
    GateFnOnly,
    LabeledEval,
}

struct GateFnProjection {
    gate_fn: GateFn,
    module_ports: Vec<ModulePortAigBinding>,
    instances: Vec<InstanceAigBinding>,
}

pub fn project_gatefn_from_netlist_and_liberty_with_options(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
    dff_cells_identity: &std::collections::HashSet<String>,
    dff_cells_inverted: &std::collections::HashSet<String>,
    options: &GateFnProjectOptions,
) -> Result<GateFn, String> {
    project_gatefn_from_netlist_and_liberty_internal(
        module,
        nets,
        interner,
        liberty_lib,
        dff_cells_identity,
        dff_cells_inverted,
        options,
        ProjectionMode::GateFnOnly,
    )
    .map(|projection| projection.gate_fn)
}

/// Projects a combinational Liberty-backed netlist into one labeled AIG.
pub fn project_labeled_netlist_aig(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
) -> Result<LabeledNetlistAig, String> {
    let empty_dff_cells = HashSet::new();
    let projection = project_gatefn_from_netlist_and_liberty_internal(
        module,
        nets,
        interner,
        liberty_lib,
        &empty_dff_cells,
        &empty_dff_cells,
        &GateFnProjectOptions {
            collapse_sequential: false,
        },
        ProjectionMode::LabeledEval,
    )?;
    let module_name = interner
        .resolve(module.name)
        .ok_or_else(|| "could not resolve module name".to_string())?
        .to_string();
    Ok(LabeledNetlistAig {
        module_name,
        gate_fn: projection.gate_fn,
        module_ports: projection.module_ports,
        instances: projection.instances,
    })
}

#[derive(Debug, Clone)]
struct SelectedSequentialClock {
    bit: BitIndex,
    port_name: String,
    active_edge: Option<SequentialClockEdge>,
}

struct PendingSequentialInstance {
    normalized_index: usize,
    spec: GvEvalSequentialCellSpec,
    register_name: String,
    q: TransitionInputId,
    state_operands: HashMap<String, AigOperand>,
    output_operands: HashMap<String, AigOperand>,
    d_operand: Option<AigOperand>,
}

/// Projects an FF-only Liberty-backed netlist into one labeled sequential AIG.
///
/// The transition function contains ordinary data inputs plus one synthetic Q
/// input per mapped FF. The selected top-level clock remains metadata rather
/// than a transition input, because one simulation step represents one active
/// clock edge.
pub fn project_labeled_sequential_netlist_aig(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
    clock_port_name_hint: Option<&str>,
) -> Result<LabeledSequentialNetlistAig, String> {
    let normalized =
        NormalizedNetlistModule::new(module, nets, interner).map_err(|e| e.to_string())?;
    let sequential_specs = collect_gv_eval_sequential_specs(&normalized, interner, liberty_lib)?;
    validate_labeled_sequential_eval_module(
        &normalized,
        nets,
        interner,
        liberty_lib,
        &sequential_specs,
    )?;
    let selected_clock = select_sequential_clock(
        &normalized,
        nets,
        interner,
        &sequential_specs,
        clock_port_name_hint,
    )?;

    let used_cell_names: HashSet<String> = module
        .instances
        .iter()
        .filter_map(|inst| interner.resolve(inst.type_name).map(str::to_string))
        .collect();
    let empty_dff_cells = HashSet::new();
    let cell_formula_map = build_cell_formula_map(
        liberty_lib,
        /* collapse_sequential= */ false,
        &used_cell_names,
        &empty_dff_cells,
        &empty_dff_cells,
    )?;
    let module_name = interner
        .resolve(module.name)
        .ok_or_else(|| "could not resolve module name".to_string())?
        .to_string();
    let mut gb = GateBuilder::new(module_name.clone(), GateBuilderOptions::no_opt());
    let mut resolved = ResolvedBitValues::new(normalized.bit_count());
    let external_inputs = collect_sequential_module_input_bits(
        &normalized,
        nets,
        interner,
        selected_clock.as_ref(),
        &mut gb,
        &mut resolved,
    )?;

    let mut pending_sequential = Vec::new();
    for (normalized_index, spec) in sequential_specs.iter().enumerate() {
        let Some(spec) = spec else {
            continue;
        };
        let inst = &normalized.instances[normalized_index];
        let instance_name = interner
            .resolve(inst.instance_name)
            .ok_or_else(|| "could not resolve instance name".to_string())?;
        let type_name = interner
            .resolve(inst.type_name)
            .ok_or_else(|| "could not resolve cell type name".to_string())?;
        let cell = liberty_lib
            .cells
            .iter()
            .find(|cell| cell.name == type_name)
            .ok_or_else(|| format!("Cell '{}' not found in Liberty", type_name))?;
        let register_name = format!("{}__{}", instance_name, spec.state_var);
        let q = TransitionInputId::new(external_inputs.len() + pending_sequential.len());
        let q_bv = gb.add_input(canonical_register_q_name(&register_name), 1);
        let q_operand = *q_bv.get_lsb(0);
        let state_operands = build_sequential_state_operand_map(spec, q_operand, &mut gb);
        let output_operands = seed_sequential_instance_outputs(
            inst,
            cell,
            type_name,
            instance_name,
            interner,
            nets,
            &normalized,
            &state_operands,
            &mut gb,
            &mut resolved,
            liberty_lib,
        )?;
        pending_sequential.push(PendingSequentialInstance {
            normalized_index,
            spec: spec.clone(),
            register_name,
            q,
            state_operands,
            output_operands,
            d_operand: None,
        });
    }

    let (used_as_input, driven) = collect_netlist_bit_usage(
        &normalized,
        interner,
        liberty_lib,
        |instance_index, pin_name| {
            sequential_specs[instance_index]
                .as_ref()
                .is_some_and(|spec| spec.clock.pin_name == pin_name)
        },
    )?;
    check_undriven_bits(&used_as_input, &driven, &normalized, nets, interner)?;

    let module_output_bits = collect_module_output_bits(&normalized);
    let mut pending_assigns = build_pending_assign_bits(&normalized);
    let mut labeled_instances = vec![None; normalized.instances.len()];
    let mut unprocessed = normalized
        .instances
        .iter()
        .enumerate()
        .filter_map(|(index, inst)| sequential_specs[index].is_none().then_some(inst))
        .collect::<Vec<_>>();
    let mut processed_any = true;
    while (!unprocessed.is_empty() || !pending_assigns.is_empty()) && processed_any {
        processed_any = false;
        let (next_pending_assigns, processed_assign) = process_pending_assign_bits(
            pending_assigns,
            &normalized,
            &mut resolved,
            nets,
            interner,
            &mut gb,
        )?;
        pending_assigns = next_pending_assigns;
        processed_any |= processed_assign;

        let mut index = 0;
        while index < unprocessed.len() {
            let inst = unprocessed[index];
            let type_name = interner.resolve(inst.type_name).unwrap();
            let instance_name = interner.resolve(inst.instance_name).unwrap();
            let cell = liberty_lib
                .cells
                .iter()
                .find(|cell| cell.name == type_name)
                .ok_or_else(|| format!("Cell '{}' not found in Liberty", type_name))?;
            let pin_directions = build_pin_direction_map(cell, liberty_lib);
            let (input_map, missing_inputs, port_map) = build_instance_input_map(
                inst,
                &pin_directions,
                interner,
                nets,
                &resolved,
                &mut gb,
                &normalized,
            );
            if !missing_inputs.is_empty() {
                index += 1;
                continue;
            }
            let processed = process_instance_outputs(
                inst,
                type_name,
                instance_name,
                &pin_directions,
                interner,
                nets,
                &mut gb,
                &mut resolved,
                &normalized,
                &empty_dff_cells,
                &empty_dff_cells,
                &cell_formula_map,
                &input_map,
                &port_map,
            )?;
            if processed.processed_any_output {
                let binding = build_sequential_instance_aig_binding(
                    inst,
                    cell,
                    liberty_lib,
                    type_name,
                    instance_name,
                    interner,
                    nets,
                    &normalized,
                    &input_map,
                    &processed.output_operands,
                    None,
                    None,
                )?;
                labeled_instances[inst.raw_index.0] = Some(binding);
                unprocessed.remove(index);
                processed_any = true;
            } else {
                index += 1;
            }
        }
        if !processed_any && unprocessed.is_empty() && !pending_assigns.is_empty() {
            // Clock-alias and dead wiring assigns do not affect visible
            // outputs once the selected clock has been recorded as metadata.
            if pending_assigns
                .iter()
                .all(|pending_bit| !module_output_bits.contains(&pending_bit.target))
            {
                break;
            }
        }
    }
    let unresolved_output_assigns = pending_assigns
        .iter()
        .filter(|pending_bit| module_output_bits.contains(&pending_bit.target))
        .count();
    if !unprocessed.is_empty() || unresolved_output_assigns != 0 {
        return Err(format!(
            "Could not resolve sequential netlist combinational dependencies: {} instances and {} output-relevant continuous assign bits remain",
            unprocessed.len(),
            unresolved_output_assigns
        ));
    }

    for (register_index, pending) in pending_sequential.iter_mut().enumerate() {
        let inst = &normalized.instances[pending.normalized_index];
        let type_name = interner.resolve(inst.type_name).unwrap();
        let instance_name = interner.resolve(inst.instance_name).unwrap();
        let cell = liberty_lib
            .cells
            .iter()
            .find(|cell| cell.name == type_name)
            .ok_or_else(|| format!("Cell '{}' not found in Liberty", type_name))?;
        let pin_directions = build_pin_direction_map(cell, liberty_lib);
        let (mut input_map, missing_inputs, port_map) = build_instance_input_map_skipping_pin(
            inst,
            &pin_directions,
            interner,
            nets,
            &resolved,
            &mut gb,
            &normalized,
            Some(pending.spec.clock.pin_name.as_str()),
        );
        if !missing_inputs.is_empty() {
            return Err(format!(
                "sequential cell '{}' instance '{}' has unresolved inputs: [{}]",
                type_name,
                instance_name,
                missing_inputs.join(", ")
            ));
        }
        input_map.extend(
            pending
                .state_operands
                .iter()
                .map(|(name, operand)| (name.clone(), *operand)),
        );
        let context = crate::liberty::cell_formula::EmitContext {
            cell_name: type_name,
            original_formula: pending.spec.next_state_text.as_str(),
            instance_name: Some(instance_name),
            port_map: Some(&port_map),
        };
        let d_operand = pending
            .spec
            .next_state
            .emit_formula_term(&mut gb, &input_map, &context)
            .map_err(|error| {
                format!(
                    "Failed to emit next_state for cell '{}' instance '{}': {}",
                    type_name, instance_name, error
                )
            })?;
        pending.d_operand = Some(d_operand);
        let binding = build_sequential_instance_aig_binding(
            inst,
            cell,
            liberty_lib,
            type_name,
            instance_name,
            interner,
            nets,
            &normalized,
            &input_map,
            &pending.output_operands,
            Some(pending.spec.clock.pin_name.as_str()),
            Some(register_index),
        )?;
        labeled_instances[inst.raw_index.0] = Some(binding);
    }

    let module_ports = build_sequential_module_port_aig_bindings(
        &normalized,
        nets,
        interner,
        &resolved,
        selected_clock.as_ref(),
    )?;
    let mut external_outputs = Vec::new();
    for port in &normalized.ports {
        if port.direction != PortDirection::Output {
            continue;
        }
        let net_name = interner.resolve(port.name).unwrap();
        let bv = resolved.materialize_output_or_false(port.bits.as_slice(), &mut gb);
        let output_id = TransitionOutputId::new(external_outputs.len());
        gb.add_output(net_name.to_string(), bv);
        external_outputs.push(output_id);
    }
    let mut registers = Vec::with_capacity(pending_sequential.len());
    for pending in pending_sequential {
        let d_id = TransitionOutputId::new(external_outputs.len() + registers.len());
        let d_operand = pending
            .d_operand
            .expect("every pending sequential instance has a D operand");
        gb.add_output(
            canonical_register_d_name(&pending.register_name),
            AigBitVector::from_bit(d_operand),
        );
        registers.push(RegisterBinding {
            name: pending.register_name,
            q: pending.q,
            d: d_id,
            initial_value: None,
        });
    }
    let added_synthetic_output = external_outputs.is_empty() && registers.is_empty();
    if added_synthetic_output {
        // GateBuilder needs an endpoint while it constructs an otherwise
        // empty transition graph.
        gb.add_output("__gv_eval_empty".to_string(), AigBitVector::zeros(0));
    }
    let mut transition = gb.build();
    if added_synthetic_output {
        transition.outputs.clear();
    }
    let instances = labeled_instances
        .into_iter()
        .enumerate()
        .map(|(instance_index, binding)| {
            binding.ok_or_else(|| {
                format!(
                    "internal error: no labeled sequential AIG binding was produced for instance index {}",
                    instance_index
                )
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let labeled_clock = selected_clock.map(|clock| LabeledSequentialClock {
        port_name: clock.port_name,
        active_edge: clock.active_edge,
    });
    let clock = labeled_clock.as_ref().map(|clock| ClockPort {
        name: clock.port_name.clone(),
    });
    let sequential_gate_fn = SequentialGateFn::new(
        module_name.clone(),
        transition,
        external_inputs,
        external_outputs,
        clock,
        registers,
    )
    .map_err(|error| format!("invalid sequential netlist transition function: {}", error))?;
    Ok(LabeledSequentialNetlistAig {
        module_name,
        sequential_gate_fn,
        clock: labeled_clock,
        module_ports,
        instances,
    })
}

fn collect_gv_eval_sequential_specs(
    normalized: &NormalizedNetlistModule<'_>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
) -> Result<Vec<Option<GvEvalSequentialCellSpec>>, String> {
    normalized
        .instances
        .iter()
        .map(|inst| {
            let type_name = interner
                .resolve(inst.type_name)
                .ok_or_else(|| "could not resolve cell type name".to_string())?;
            let instance_name = interner
                .resolve(inst.instance_name)
                .ok_or_else(|| "could not resolve instance name".to_string())?;
            let cell = liberty_lib
                .cells
                .iter()
                .find(|cell| cell.name == type_name)
                .ok_or_else(|| {
                    format!(
                        "cell '{}' used by instance '{}' was not found in Liberty data",
                        type_name, instance_name
                    )
                })?;
            get_gv_eval_sequential_cell_spec(cell, liberty_lib)
        })
        .collect()
}

/// Validates the standard-cell subset used by labeled sequential gv-eval.
fn validate_labeled_sequential_eval_module(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
    sequential_specs: &[Option<GvEvalSequentialCellSpec>],
) -> Result<(), String> {
    let module_name = interner.resolve(normalized.raw.name).unwrap_or("<unknown>");
    if normalized
        .ports
        .iter()
        .any(|port| port.direction == PortDirection::Inout)
    {
        return Err(format!(
            "module '{}' has an inout port; sequential gv-eval supports only input and output module ports",
            module_name
        ));
    }
    for (instance_index, inst) in normalized.instances.iter().enumerate() {
        let type_name = interner.resolve(inst.type_name).unwrap_or("<unknown>");
        let instance_name = interner.resolve(inst.instance_name).unwrap_or("<unknown>");
        let cell = liberty_lib
            .cells
            .iter()
            .find(|cell| cell.name == type_name)
            .ok_or_else(|| {
                format!(
                    "cell '{}' used by instance '{}' was not found in Liberty data",
                    type_name, instance_name
                )
            })?;
        let sequential_spec = sequential_specs[instance_index].as_ref();
        let mut connected_pin_names = HashSet::new();
        let mut connected_output_count = 0usize;
        for connection in &inst.connections {
            let pin_name = interner.resolve(connection.port).unwrap_or("<unknown>");
            if !connected_pin_names.insert(pin_name.to_string()) {
                return Err(format!(
                    "cell '{}' instance '{}' connects pin '{}' more than once",
                    type_name, instance_name, pin_name
                ));
            }
            let pin = cell
                .pins
                .iter()
                .find(|pin| liberty_lib.resolve_string(&pin.name) == pin_name)
                .ok_or_else(|| {
                    format!(
                        "cell '{}' instance '{}' connects unknown Liberty pin '{}'",
                        type_name, instance_name, pin_name
                    )
                })?;
            let direction = checked_pin_direction(type_name, pin_name, pin.direction)?;
            match direction {
                PinDirection::Input => {
                    if connection.bits.len() != 1 {
                        return Err(format!(
                            "cell '{}' instance '{}' input pin '{}' must have exactly one connected bit; got {} ({})",
                            type_name,
                            instance_name,
                            pin_name,
                            connection.bits.len(),
                            normalized.render_sources(connection.bits.as_slice(), nets, interner)
                        ));
                    }
                    if matches!(connection.bits[0], BitSource::Unknown) {
                        return Err(format!(
                            "cell '{}' instance '{}' input pin '{}' is connected to an unknown literal",
                            type_name, instance_name, pin_name
                        ));
                    }
                }
                PinDirection::Output => {
                    connected_output_count += 1;
                    if connection.bits.len() > 1 {
                        return Err(format!(
                            "cell '{}' instance '{}' output pin '{}' must connect to at most one bit; got {} ({})",
                            type_name,
                            instance_name,
                            pin_name,
                            connection.bits.len(),
                            normalized.render_sources(connection.bits.as_slice(), nets, interner)
                        ));
                    }
                    if matches!(
                        connection.bits.first(),
                        Some(BitSource::Literal(_) | BitSource::Unknown)
                    ) {
                        return Err(format!(
                            "cell '{}' instance '{}' output pin '{}' must drive a net bit or be unconnected",
                            type_name, instance_name, pin_name
                        ));
                    }
                    let function = liberty_lib.resolve_string(&pin.function);
                    if function.is_empty() {
                        let has_state_fallback = sequential_spec.is_some_and(|spec| {
                            pin_name == spec.state_var
                                || spec
                                    .complementary_state_var
                                    .as_deref()
                                    .is_some_and(|name| pin_name == name)
                        });
                        if !has_state_fallback {
                            return Err(format!(
                                "cell '{}' instance '{}' output pin '{}' has no Liberty function",
                                type_name, instance_name, pin_name
                            ));
                        }
                    } else {
                        crate::liberty::cell_formula::parse_formula(function).map_err(|error| {
                            format!(
                                r#"failed to parse Liberty function for cell '{}' instance '{}' output pin '{}' (function "{}"): {}"#,
                                type_name, instance_name, pin_name, function, error
                            )
                        })?;
                    }
                }
                PinDirection::Invalid => unreachable!("checked_pin_direction rejects invalid"),
            }
        }
        if connected_output_count == 0 && sequential_spec.is_none() {
            return Err(format!(
                "cell '{}' instance '{}' has no explicitly connected output pins",
                type_name, instance_name
            ));
        }
    }
    Ok(())
}

fn build_pin_direction_map<'a>(cell: &'a Cell, liberty_lib: &'a Library) -> HashMap<&'a str, i32> {
    cell.pins
        .iter()
        .map(|pin| (liberty_lib.resolve_string(&pin.name), pin.direction))
        .collect()
}

fn select_sequential_clock(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    sequential_specs: &[Option<GvEvalSequentialCellSpec>],
    clock_port_name_hint: Option<&str>,
) -> Result<Option<SelectedSequentialClock>, String> {
    let mut inferred_clock_bit = None;
    let mut inferred_edge = None;
    for (instance_index, spec) in sequential_specs.iter().enumerate() {
        let Some(spec) = spec else {
            continue;
        };
        let inst = &normalized.instances[instance_index];
        let instance_name = interner.resolve(inst.instance_name).unwrap_or("<unknown>");
        let connection = inst
            .connections
            .iter()
            .find(|connection| {
                interner.resolve(connection.port) == Some(spec.clock.pin_name.as_str())
            })
            .ok_or_else(|| {
                format!(
                    "clock pin '{}' on instance '{}' is unconnected",
                    spec.clock.pin_name, instance_name
                )
            })?;
        let [source] = connection.bits.as_slice() else {
            return Err(format!(
                "clock pin '{}' on instance '{}' must connect exactly one bit",
                spec.clock.pin_name, instance_name
            ));
        };
        let clock_bit = resolve_simple_clock_source(*source, normalized, nets, interner)?;
        match inferred_clock_bit {
            None => inferred_clock_bit = Some(clock_bit),
            Some(existing) if existing != clock_bit => {
                return Err(format!(
                    "multiple clock nets detected: '{}' vs '{}'",
                    normalized.render_bit(existing, nets, interner),
                    normalized.render_bit(clock_bit, nets, interner)
                ));
            }
            _ => {}
        }
        let edge = if spec.clock.is_negated {
            SequentialClockEdge::Falling
        } else {
            SequentialClockEdge::Rising
        };
        match inferred_edge {
            None => inferred_edge = Some(edge),
            Some(existing) if existing != edge => {
                return Err(
                    "mixed positive-edge and negative-edge FF cells are not supported".to_string(),
                );
            }
            _ => {}
        }
    }

    let inferred_port = inferred_clock_bit
        .map(|clock_bit| top_input_clock_port(normalized, nets, interner, clock_bit))
        .transpose()?;
    if let Some(clock_port_name_hint) = clock_port_name_hint {
        let hinted_port = normalized
            .ports
            .iter()
            .find(|port| {
                port.direction == PortDirection::Input
                    && interner.resolve(port.name) == Some(clock_port_name_hint)
            })
            .ok_or_else(|| {
                format!(
                    "clock-port hint '{}' is not a top-level input port",
                    clock_port_name_hint
                )
            })?;
        if hinted_port.bits.len() != 1 {
            return Err(format!(
                "clock-port hint '{}' has width {}; expected one bit",
                clock_port_name_hint,
                hinted_port.bits.len()
            ));
        }
        if let Some((inferred_name, inferred_bit)) = inferred_port.as_ref() {
            if inferred_name != clock_port_name_hint || *inferred_bit != hinted_port.bits[0] {
                return Err(format!(
                    "clock-port hint '{}' disagrees with inferred clock port '{}'",
                    clock_port_name_hint, inferred_name
                ));
            }
        }
        return Ok(Some(SelectedSequentialClock {
            bit: hinted_port.bits[0],
            port_name: clock_port_name_hint.to_string(),
            active_edge: inferred_edge,
        }));
    }
    Ok(
        inferred_port.map(|(port_name, bit)| SelectedSequentialClock {
            bit,
            port_name,
            active_edge: inferred_edge,
        }),
    )
}

fn top_input_clock_port(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    clock_bit: BitIndex,
) -> Result<(String, BitIndex), String> {
    let port = normalized
        .ports
        .iter()
        .find(|port| port.direction == PortDirection::Input && port.bits.contains(&clock_bit))
        .ok_or_else(|| {
            format!(
                "derived clock '{}' is not a top-level input port",
                normalized.render_bit(clock_bit, nets, interner)
            )
        })?;
    let port_name = interner
        .resolve(port.name)
        .ok_or_else(|| "could not resolve clock port name".to_string())?;
    if port.bits.len() != 1 {
        return Err(format!(
            "clock port '{}' has width {}; expected one bit",
            port_name,
            port.bits.len()
        ));
    }
    Ok((port_name.to_string(), clock_bit))
}

fn resolve_simple_clock_source(
    source: BitSource,
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<BitIndex, String> {
    let mut current = source;
    let mut seen = HashSet::new();
    loop {
        let bit_idx = match current {
            BitSource::Bit(bit_idx) => bit_idx,
            BitSource::Literal(_) => {
                return Err("clock pin is driven by a literal".to_string());
            }
            BitSource::Unknown => {
                return Err("clock pin is driven by an unknown literal".to_string());
            }
        };
        if !seen.insert(bit_idx) {
            return Err(format!(
                "clock wiring contains a cycle at '{}'",
                normalized.render_bit(bit_idx, nets, interner)
            ));
        }
        let mut rhs = None;
        for assign in &normalized.assigns {
            for (lhs_bit, rhs_expr) in assign.lhs_bits.iter().copied().zip(&assign.rhs_bits) {
                if lhs_bit != bit_idx {
                    continue;
                }
                if rhs.replace(rhs_expr).is_some() {
                    return Err(format!(
                        "clock bit '{}' has multiple preserved wiring assigns",
                        normalized.render_bit(bit_idx, nets, interner)
                    ));
                }
            }
        }
        let Some(rhs) = rhs else {
            return Ok(bit_idx);
        };
        match rhs {
            BitExpr::Source(next) => current = *next,
            BitExpr::Not(_) | BitExpr::And(_, _) | BitExpr::Or(_, _) | BitExpr::Xor(_, _) => {
                return Err(format!(
                    "derived clock '{}' contains combinational logic",
                    normalized.render_bit(bit_idx, nets, interner)
                ));
            }
        }
    }
}

fn collect_sequential_module_input_bits(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    selected_clock: Option<&SelectedSequentialClock>,
    gb: &mut GateBuilder,
    resolved: &mut ResolvedBitValues,
) -> Result<Vec<TransitionInputId>, String> {
    let mut external_inputs = Vec::new();
    for port in &normalized.ports {
        if port.direction != PortDirection::Input {
            continue;
        }
        if selected_clock.is_some_and(|clock| {
            port.bits.len() == 1 && port.bits.first().copied() == Some(clock.bit)
        }) {
            continue;
        }
        let port_name = interner
            .resolve(port.name)
            .ok_or_else(|| "could not resolve module input port name".to_string())?;
        let input_id = TransitionInputId::new(external_inputs.len());
        let bv = gb.add_input(port_name.to_string(), port.bits.len());
        for (bit_idx, bit) in port.bits.iter().copied().zip(bv.iter_lsb_to_msb()) {
            resolved.write_bit(bit_idx, *bit, normalized, nets, interner)?;
        }
        external_inputs.push(input_id);
    }
    Ok(external_inputs)
}

fn build_sequential_module_port_aig_bindings(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    resolved: &ResolvedBitValues,
    selected_clock: Option<&SelectedSequentialClock>,
) -> Result<Vec<SequentialModulePortAigBinding>, String> {
    normalized
        .ports
        .iter()
        .map(|port| {
            let name = interner
                .resolve(port.name)
                .ok_or_else(|| "could not resolve module port name".to_string())?
                .to_string();
            let bits_lsb_to_msb = port
                .bits
                .iter()
                .map(|bit_idx| {
                    let bit = normalized.bit(*bit_idx);
                    let signal = if selected_clock
                        .is_some_and(|clock| clock.bit == *bit_idx)
                    {
                        if port.direction != PortDirection::Input {
                            return Err(format!(
                                "selected clock '{}' is connected to a non-input module port",
                                name
                            ));
                        }
                        SequentialAigSignal::Clock
                    } else {
                        let operand = resolved.resolve_bit(*bit_idx).ok_or_else(|| {
                            format!(
                                "module port bit '{}' is unresolved after sequential AIG projection",
                                normalized.render_bit(*bit_idx, nets, interner)
                            )
                        })?;
                        SequentialAigSignal::Operand(operand)
                    };
                    Ok(LabeledSequentialAigBit {
                        bit_number: bit.bit_number,
                        signal,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(SequentialModulePortAigBinding {
                name,
                direction: port.direction.clone(),
                bits_lsb_to_msb,
            })
        })
        .collect()
}

fn build_sequential_state_operand_map(
    spec: &GvEvalSequentialCellSpec,
    q_operand: AigOperand,
    gb: &mut GateBuilder,
) -> HashMap<String, AigOperand> {
    let mut state_operands = HashMap::new();
    state_operands.insert(spec.state_var.clone(), q_operand);
    if let Some(complementary_state_var) = &spec.complementary_state_var {
        state_operands.insert(complementary_state_var.clone(), gb.add_not(q_operand));
    }
    state_operands
}

#[allow(clippy::too_many_arguments)]
fn seed_sequential_instance_outputs(
    inst: &NormalizedInstance,
    cell: &Cell,
    type_name: &str,
    instance_name: &str,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    normalized: &NormalizedNetlistModule<'_>,
    state_operands: &HashMap<String, AigOperand>,
    gb: &mut GateBuilder,
    resolved: &mut ResolvedBitValues,
    liberty_lib: &Library,
) -> Result<HashMap<String, AigOperand>, String> {
    let pin_directions = build_pin_direction_map(cell, liberty_lib);
    let mut output_operands = HashMap::new();
    for connection in &inst.connections {
        let pin_name = interner.resolve(connection.port).unwrap();
        if *pin_directions.get(pin_name).unwrap_or(&0) != PinDirection::Output as i32 {
            continue;
        }
        let pin = cell
            .pins
            .iter()
            .find(|pin| liberty_lib.resolve_string(&pin.name) == pin_name)
            .ok_or_else(|| {
                format!(
                    "cell '{}' instance '{}' connects unknown Liberty pin '{}'",
                    type_name, instance_name, pin_name
                )
            })?;
        let operand = sequential_output_operand(
            cell,
            pin,
            type_name,
            instance_name,
            pin_name,
            state_operands,
            gb,
            liberty_lib,
        )?;
        output_operands.insert(pin_name.to_string(), operand);
        let output_bits = output_target_bits(connection)?;
        resolved.write_bits(
            output_bits.as_slice(),
            &AigBitVector::from_bit(operand),
            normalized,
            nets,
            interner,
        )?;
    }
    Ok(output_operands)
}

#[allow(clippy::too_many_arguments)]
fn sequential_output_operand(
    cell: &Cell,
    pin: &crate::liberty_model::Pin,
    type_name: &str,
    instance_name: &str,
    pin_name: &str,
    state_operands: &HashMap<String, AigOperand>,
    gb: &mut GateBuilder,
    liberty_lib: &Library,
) -> Result<AigOperand, String> {
    let function = liberty_lib.resolve_string(&pin.function);
    if function.is_empty() {
        if let Some(operand) = state_operands.get(pin_name) {
            return Ok(*operand);
        }
        return Err(format!(
            "cell '{}' instance '{}' output pin '{}' has no Liberty function",
            type_name, instance_name, pin_name
        ));
    }
    let term = crate::liberty::cell_formula::parse_formula(function).map_err(|error| {
        format!(
            r#"failed to parse Liberty function for cell '{}' instance '{}' output pin '{}' (function "{}"): {}"#,
            type_name, instance_name, pin_name, function, error
        )
    })?;
    let mut non_state_inputs = term
        .inputs()
        .into_iter()
        .filter(|name| !state_operands.contains_key(name))
        .collect::<Vec<_>>();
    non_state_inputs.sort();
    non_state_inputs.dedup();
    if !non_state_inputs.is_empty() {
        return Err(format!(
            "sequential output function for cell '{}' instance '{}' pin '{}' references non-state inputs [{}]",
            type_name,
            instance_name,
            pin_name,
            non_state_inputs.join(", ")
        ));
    }
    let context = crate::liberty::cell_formula::EmitContext {
        cell_name: &cell.name,
        original_formula: function,
        instance_name: Some(instance_name),
        port_map: None,
    };
    term.emit_formula_term(gb, state_operands, &context)
        .map_err(|error| {
            format!(
                "Failed to emit sequential output function for cell '{}' instance '{}' pin '{}': {}",
                type_name, instance_name, pin_name, error
            )
        })
}

/// Collects net bits used as cell inputs and driven by ports, assigns, or
/// cells.
fn collect_netlist_bit_usage<F>(
    normalized: &NormalizedNetlistModule<'_>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
    mut should_skip_input_pin: F,
) -> Result<(HashSet<BitIndex>, HashSet<BitIndex>), String>
where
    F: FnMut(usize, &str) -> bool,
{
    let mut used_as_input = HashSet::new();
    let mut driven = HashSet::new();
    for port in &normalized.ports {
        if port.direction == PortDirection::Input {
            driven.extend(port.bits.iter().copied());
        }
    }
    for assign in &normalized.assigns {
        driven.extend(assign.lhs_bits.iter().copied());
        for rhs in &assign.rhs_bits {
            collect_expr_source_bits(rhs, &mut used_as_input);
        }
    }
    for (instance_index, inst) in normalized.instances.iter().enumerate() {
        let type_name = interner.resolve(inst.type_name).unwrap();
        let cell = liberty_lib
            .cells
            .iter()
            .find(|cell| cell.name == type_name)
            .ok_or_else(|| format!("Cell '{}' not found in Liberty", type_name))?;
        let pin_directions = build_pin_direction_map(cell, liberty_lib);
        for connection in &inst.connections {
            let port_name = interner.resolve(connection.port).unwrap();
            let pin_dir = *pin_directions.get(port_name).unwrap_or(&0);
            if pin_dir == PinDirection::Output as i32 {
                driven.extend(output_target_bits(connection)?);
            } else if pin_dir == PinDirection::Input as i32
                && !should_skip_input_pin(instance_index, port_name)
            {
                used_as_input.extend(connection.bits.iter().filter_map(|source| match source {
                    BitSource::Bit(bit_idx) => Some(*bit_idx),
                    BitSource::Literal(_) | BitSource::Unknown => None,
                }));
            }
        }
    }
    Ok((used_as_input, driven))
}

#[allow(clippy::too_many_arguments)]
fn project_gatefn_from_netlist_and_liberty_internal(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
    dff_cells_identity: &HashSet<String>,
    dff_cells_inverted: &HashSet<String>,
    options: &GateFnProjectOptions,
    projection_mode: ProjectionMode,
) -> Result<GateFnProjection, String> {
    let normalized =
        NormalizedNetlistModule::new(module, nets, interner).map_err(|e| e.to_string())?;
    if projection_mode == ProjectionMode::LabeledEval {
        validate_labeled_eval_module(&normalized, nets, interner, liberty_lib)?;
    }
    let used_cell_names: HashSet<String> = module
        .instances
        .iter()
        .filter_map(|inst| interner.resolve(inst.type_name).map(str::to_string))
        .collect();
    let cell_formula_map = build_cell_formula_map(
        liberty_lib,
        options.collapse_sequential,
        &used_cell_names,
        dff_cells_identity,
        dff_cells_inverted,
    )?;
    let module_name = interner.resolve(module.name).unwrap();
    let mut gb = GateBuilder::new(module_name.to_string(), GateBuilderOptions::no_opt());
    let mut resolved = ResolvedBitValues::new(normalized.bit_count());
    collect_module_io_bits(&normalized, nets, interner, &mut gb, &mut resolved)?;
    let module_output_bits = collect_module_output_bits(&normalized);
    let mut pending_assigns = build_pending_assign_bits(&normalized);
    let mut labeled_instances = if projection_mode == ProjectionMode::LabeledEval {
        vec![None; normalized.instances.len()]
    } else {
        Vec::new()
    };
    let (used_as_input, driven) =
        collect_netlist_bit_usage(&normalized, interner, liberty_lib, |_, _| false)?;
    check_undriven_bits(&used_as_input, &driven, &normalized, nets, interner)?;
    let mut unprocessed: Vec<_> = normalized.instances.iter().collect();
    let mut processed_any = true;
    while (!unprocessed.is_empty() || !pending_assigns.is_empty()) && processed_any {
        processed_any = false;
        let (next_pending_assigns, processed_assign) = process_pending_assign_bits(
            pending_assigns,
            &normalized,
            &mut resolved,
            nets,
            interner,
            &mut gb,
        )?;
        pending_assigns = next_pending_assigns;
        processed_any |= processed_assign;
        let mut i = 0;
        while i < unprocessed.len() {
            let inst = unprocessed[i];
            let type_name = interner.resolve(inst.type_name).unwrap();
            let inst_name = interner.resolve(inst.instance_name).unwrap();
            let cell = liberty_lib
                .cells
                .iter()
                .find(|c| c.name == type_name)
                .ok_or_else(|| format!("Cell '{}' not found in Liberty", type_name))?;
            let mut pin_directions = HashMap::new();
            for pin in &cell.pins {
                pin_directions.insert(liberty_lib.resolve_string(&pin.name), pin.direction);
            }
            let (input_map, missing_inputs, port_map) = build_instance_input_map(
                inst,
                &pin_directions,
                interner,
                nets,
                &resolved,
                &mut gb,
                &normalized,
            );
            if !missing_inputs.is_empty() {
                log::trace!(
                    "Skipping instance '{}' (cell '{}') due to missing input nets: {:?}",
                    inst_name,
                    type_name,
                    missing_inputs
                );
                i += 1;
                continue;
            }
            let processed = process_instance_outputs(
                inst,
                type_name,
                inst_name,
                &pin_directions,
                interner,
                nets,
                &mut gb,
                &mut resolved,
                &normalized,
                dff_cells_identity,
                dff_cells_inverted,
                &cell_formula_map,
                &input_map,
                &port_map,
            )?;
            if processed.processed_any_output {
                if projection_mode == ProjectionMode::LabeledEval {
                    let binding = build_instance_aig_binding(
                        inst,
                        cell,
                        liberty_lib,
                        type_name,
                        inst_name,
                        interner,
                        nets,
                        &normalized,
                        &input_map,
                        &processed.output_operands,
                    )?;
                    labeled_instances[inst.raw_index.0] = Some(binding);
                }
                unprocessed.remove(i);
                processed_any = true;
            } else {
                i += 1;
            }
        }
        if !processed_any && unprocessed.is_empty() && !pending_assigns.is_empty() {
            if pending_assigns
                .iter()
                .all(|pending_bit| !module_output_bits.contains(&pending_bit.target))
            {
                break;
            }
        }
        if !processed_any && (!unprocessed.is_empty() || !pending_assigns.is_empty()) {
            // Build a short, actionable diagnostic to help pinpoint why instances
            // could not be resolved. We show up to a bounded number of examples
            // with their missing inputs.
            let mut diag_lines: Vec<String> = Vec::new();
            let example_limit = 10usize.min(unprocessed.len());
            for inst in unprocessed.iter().take(example_limit) {
                let type_name = interner.resolve(inst.type_name).unwrap_or("<unknown>");
                let inst_name = interner.resolve(inst.instance_name).unwrap_or("<unknown>");
                // Determine pin directions for this cell type (to identify inputs).
                let mut pin_directions = HashMap::new();
                if let Some(cell) = liberty_lib.cells.iter().find(|c| c.name == type_name) {
                    for pin in &cell.pins {
                        pin_directions.insert(liberty_lib.resolve_string(&pin.name), pin.direction);
                    }
                }
                // Recompute missing inputs for this instance w.r.t. current resolved nets.
                let (_input_map, missing_inputs, _port_map) = build_instance_input_map(
                    inst,
                    &pin_directions,
                    interner,
                    nets,
                    &resolved,
                    &mut gb,
                    &normalized,
                );
                if missing_inputs.is_empty() {
                    diag_lines.push(format!(
                        "- cell '{}' instance '{}' at {}:{} has no missing inputs detected; unresolved due to dependency cycle or unknown output formula",
                        type_name, inst_name, inst.inst_lineno, inst.inst_colno
                    ));
                } else {
                    diag_lines.push(format!(
                        "- cell '{}' instance '{}' at {}:{} missing inputs: [{}]",
                        type_name,
                        inst_name,
                        inst.inst_lineno,
                        inst.inst_colno,
                        missing_inputs.join(", ")
                    ));
                }
            }

            let mut msg = String::new();
            msg.push_str(&format!(
                "Could not resolve all instance dependencies (possible cycle or missing driver). Remaining instances: {}\n",
                unprocessed.len()
            ));
            msg.push_str(&format!(
                "Examples of unresolved instances (showing up to {}):\n",
                example_limit
            ));
            for line in diag_lines {
                msg.push_str(&line);
                msg.push('\n');
            }
            if !pending_assigns.is_empty() {
                msg.push_str(&format!(
                    "Unresolved continuous assign bits: {}\n",
                    pending_assigns.len()
                ));
                for pending_bit in pending_assigns.iter().take(10) {
                    let assign = &normalized.assigns[pending_bit.assign_index];
                    let mut missing = Vec::new();
                    collect_missing_expr_sources(
                        &assign.rhs_bits[pending_bit.rhs_bit_index],
                        &resolved,
                        &normalized,
                        nets,
                        interner,
                        &mut missing,
                    );
                    missing.sort();
                    missing.dedup();
                    if missing.is_empty() {
                        msg.push_str(&format!(
                            "- assign to '{}' at {} remains unresolved\n",
                            normalized.render_bit(pending_bit.target, nets, interner),
                            assign.span.to_human_string()
                        ));
                    } else {
                        msg.push_str(&format!(
                            "- assign to '{}' at {} is waiting on [{}]\n",
                            normalized.render_bit(pending_bit.target, nets, interner),
                            assign.span.to_human_string(),
                            missing.join(", ")
                        ));
                    }
                }
            }
            msg.push_str(r#"Hint: re-run with RUST_LOG=trace to log skipped instances and their missing nets during processing."#);
            return Err(msg);
        }
    }
    let module_ports = if projection_mode == ProjectionMode::LabeledEval {
        build_module_port_aig_bindings(&normalized, nets, interner, &resolved)?
    } else {
        Vec::new()
    };
    for port in &normalized.ports {
        if port.direction == PortDirection::Output {
            let net_name = interner.resolve(port.name).unwrap();
            let bv = resolved.materialize_output_or_false(port.bits.as_slice(), &mut gb);
            assert_eq!(
                bv.get_bit_count(),
                port.bits.len(),
                "Output net '{}' width mismatch",
                net_name
            );
            gb.add_output(net_name.to_string(), bv);
        }
    }
    let has_visible_outputs = normalized
        .ports
        .iter()
        .any(|port| port.direction == PortDirection::Output);
    if projection_mode == ProjectionMode::LabeledEval && !has_visible_outputs {
        // GateBuilder requires an endpoint while it constructs the graph, but
        // an outputless gv-eval module has no visible values to expose.
        gb.add_output("__gv_eval_empty".to_string(), AigBitVector::zeros(0));
    }
    let instances = if projection_mode == ProjectionMode::LabeledEval {
        labeled_instances
            .into_iter()
            .enumerate()
            .map(|(instance_index, binding)| {
                binding.ok_or_else(|| {
                    format!(
                        "internal error: no labeled AIG binding was produced for instance index {}",
                        instance_index
                    )
                })
            })
            .collect::<Result<Vec<InstanceAigBinding>, String>>()?
    } else {
        Vec::new()
    };
    let mut gate_fn = gb.build();
    if projection_mode == ProjectionMode::LabeledEval && !has_visible_outputs {
        gate_fn.outputs.clear();
    }
    Ok(GateFnProjection {
        gate_fn,
        module_ports,
        instances,
    })
}

/// Validates the narrower combinational standard-cell subset used by gv-eval.
fn validate_labeled_eval_module(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    liberty_lib: &Library,
) -> Result<(), String> {
    let module_name = interner.resolve(normalized.raw.name).unwrap_or("<unknown>");
    if normalized
        .ports
        .iter()
        .any(|port| port.direction == PortDirection::Inout)
    {
        return Err(format!(
            "module '{}' has an inout port; gv-eval supports only input and output module ports",
            module_name
        ));
    }
    for inst in &normalized.instances {
        let type_name = interner.resolve(inst.type_name).unwrap_or("<unknown>");
        let instance_name = interner.resolve(inst.instance_name).unwrap_or("<unknown>");
        let cell = liberty_lib
            .cells
            .iter()
            .find(|cell| cell.name == type_name)
            .ok_or_else(|| {
                format!(
                    "cell '{}' used by instance '{}' was not found in Liberty data",
                    type_name, instance_name
                )
            })?;
        if let Some(sequential) = cell.sequential.first() {
            let kind = match sequential.kind {
                value if value == SequentialKind::Ff as i32 => "ff",
                value if value == SequentialKind::Latch as i32 => "latch",
                _ => "unknown",
            };
            return Err(format!(
                "sequential cell '{}' instance '{}' at {}:{} is not supported by gv-eval (kind: {})",
                type_name, instance_name, inst.inst_lineno, inst.inst_colno, kind
            ));
        }

        let mut connected_pin_names = HashSet::new();
        let mut connected_output_count = 0usize;
        for connection in &inst.connections {
            let pin_name = interner.resolve(connection.port).unwrap_or("<unknown>");
            if !connected_pin_names.insert(pin_name.to_string()) {
                return Err(format!(
                    "cell '{}' instance '{}' connects pin '{}' more than once",
                    type_name, instance_name, pin_name
                ));
            }
            let pin = cell
                .pins
                .iter()
                .find(|pin| liberty_lib.resolve_string(&pin.name) == pin_name)
                .ok_or_else(|| {
                    format!(
                        "cell '{}' instance '{}' connects unknown Liberty pin '{}'",
                        type_name, instance_name, pin_name
                    )
                })?;
            let direction = checked_pin_direction(type_name, pin_name, pin.direction)?;
            match direction {
                PinDirection::Input => {
                    if connection.bits.len() != 1 {
                        return Err(format!(
                            "cell '{}' instance '{}' input pin '{}' must have exactly one connected bit; got {} ({})",
                            type_name,
                            instance_name,
                            pin_name,
                            connection.bits.len(),
                            normalized.render_sources(connection.bits.as_slice(), nets, interner)
                        ));
                    }
                    if matches!(connection.bits[0], BitSource::Unknown) {
                        return Err(format!(
                            "cell '{}' instance '{}' input pin '{}' is connected to an unknown literal",
                            type_name, instance_name, pin_name
                        ));
                    }
                }
                PinDirection::Output => {
                    connected_output_count += 1;
                    if connection.bits.len() > 1 {
                        return Err(format!(
                            "cell '{}' instance '{}' output pin '{}' must connect to at most one bit; got {} ({})",
                            type_name,
                            instance_name,
                            pin_name,
                            connection.bits.len(),
                            normalized.render_sources(connection.bits.as_slice(), nets, interner)
                        ));
                    }
                    if matches!(
                        connection.bits.first(),
                        Some(BitSource::Literal(_) | BitSource::Unknown)
                    ) {
                        return Err(format!(
                            "cell '{}' instance '{}' output pin '{}' must drive a net bit or be unconnected",
                            type_name, instance_name, pin_name
                        ));
                    }
                    let function = liberty_lib.resolve_string(&pin.function);
                    if function.is_empty() {
                        return Err(format!(
                            "cell '{}' instance '{}' output pin '{}' has no Liberty function",
                            type_name, instance_name, pin_name
                        ));
                    }
                    crate::liberty::cell_formula::parse_formula(function).map_err(|e| {
                        format!(
                            "failed to parse Liberty function for cell '{}' instance '{}' output pin '{}' (function \"{}\"): {}",
                            type_name, instance_name, pin_name, function, e
                        )
                    })?;
                }
                PinDirection::Invalid => unreachable!("checked_pin_direction rejects invalid"),
            }
        }
        if connected_output_count == 0 {
            return Err(format!(
                "cell '{}' instance '{}' has no explicitly connected output pins",
                type_name, instance_name
            ));
        }
    }
    Ok(())
}

fn checked_pin_direction(
    cell_name: &str,
    pin_name: &str,
    raw_direction: i32,
) -> Result<PinDirection, String> {
    match raw_direction {
        value if value == PinDirection::Input as i32 => Ok(PinDirection::Input),
        value if value == PinDirection::Output as i32 => Ok(PinDirection::Output),
        _ => Err(format!(
            "cell '{}' pin '{}' has invalid Liberty direction value {}",
            cell_name, pin_name, raw_direction
        )),
    }
}

fn build_module_port_aig_bindings(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    resolved: &ResolvedBitValues,
) -> Result<Vec<ModulePortAigBinding>, String> {
    normalized
        .ports
        .iter()
        .map(|port| {
            let name = interner
                .resolve(port.name)
                .ok_or_else(|| "could not resolve module port name".to_string())?
                .to_string();
            let bits_lsb_to_msb = port
                .bits
                .iter()
                .map(|bit_idx| {
                    let bit = normalized.bit(*bit_idx);
                    let operand = resolved.resolve_bit(*bit_idx).ok_or_else(|| {
                        format!(
                            "module port bit '{}' is unresolved after AIG projection",
                            normalized.render_bit(*bit_idx, nets, interner)
                        )
                    })?;
                    Ok(LabeledAigBit {
                        bit_number: bit.bit_number,
                        operand,
                    })
                })
                .collect::<Result<Vec<LabeledAigBit>, String>>()?;
            Ok(ModulePortAigBinding {
                name,
                direction: port.direction.clone(),
                bits_lsb_to_msb,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_instance_aig_binding(
    inst: &NormalizedInstance,
    cell: &Cell,
    liberty_lib: &Library,
    type_name: &str,
    instance_name: &str,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    normalized: &NormalizedNetlistModule<'_>,
    input_operands: &HashMap<String, AigOperand>,
    output_operands: &HashMap<String, AigOperand>,
) -> Result<InstanceAigBinding, String> {
    let mut pins = Vec::new();
    for pin in &cell.pins {
        let pin_name = liberty_lib.resolve_string(&pin.name);
        let Some(connection) = inst
            .connections
            .iter()
            .find(|connection| interner.resolve(connection.port) == Some(pin_name))
        else {
            continue;
        };
        let direction = checked_pin_direction(type_name, pin_name, pin.direction)?;
        let operand = match direction {
            PinDirection::Input => input_operands.get(pin_name),
            PinDirection::Output => output_operands.get(pin_name),
            PinDirection::Invalid => unreachable!("checked_pin_direction rejects invalid"),
        }
        .copied()
        .ok_or_else(|| {
            format!(
                "internal error: no AIG operand was produced for cell '{}' instance '{}' pin '{}'",
                type_name, instance_name, pin_name
            )
        })?;
        pins.push(InstancePinAigBinding {
            pin_name: pin_name.to_string(),
            direction,
            operand,
            connection: build_pin_connection(connection, normalized, nets, interner)?,
        });
    }
    Ok(InstanceAigBinding {
        instance_name: instance_name.to_string(),
        cell_type: type_name.to_string(),
        pins,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_sequential_instance_aig_binding(
    inst: &NormalizedInstance,
    cell: &Cell,
    liberty_lib: &Library,
    type_name: &str,
    instance_name: &str,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    normalized: &NormalizedNetlistModule<'_>,
    input_operands: &HashMap<String, AigOperand>,
    output_operands: &HashMap<String, AigOperand>,
    clock_pin: Option<&str>,
    state_register_index: Option<usize>,
) -> Result<SequentialInstanceAigBinding, String> {
    let mut pins = Vec::new();
    for pin in &cell.pins {
        let pin_name = liberty_lib.resolve_string(&pin.name);
        let Some(connection) = inst
            .connections
            .iter()
            .find(|connection| interner.resolve(connection.port) == Some(pin_name))
        else {
            continue;
        };
        let direction = checked_pin_direction(type_name, pin_name, pin.direction)?;
        let signal = if clock_pin == Some(pin_name) {
            SequentialAigSignal::Clock
        } else {
            let operand = match direction {
                PinDirection::Input => input_operands.get(pin_name),
                PinDirection::Output => output_operands.get(pin_name),
                PinDirection::Invalid => unreachable!("checked_pin_direction rejects invalid"),
            }
            .copied()
            .ok_or_else(|| {
                format!(
                    "internal error: no sequential AIG signal was produced for cell '{}' instance '{}' pin '{}'",
                    type_name, instance_name, pin_name
                )
            })?;
            SequentialAigSignal::Operand(operand)
        };
        pins.push(SequentialInstancePinAigBinding {
            pin_name: pin_name.to_string(),
            direction,
            signal,
            connection: build_pin_connection(connection, normalized, nets, interner)?,
        });
    }
    Ok(SequentialInstanceAigBinding {
        instance_name: instance_name.to_string(),
        cell_type: type_name.to_string(),
        pins,
        state_register_index,
    })
}

fn build_pin_connection(
    connection: &NormalizedConnection,
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<PinConnection, String> {
    match connection.bits.as_slice() {
        [] => Ok(PinConnection::Unconnected),
        [BitSource::Bit(bit_idx)] => {
            let bit = normalized.bit(*bit_idx);
            Ok(PinConnection::Net {
                net_name: crate::netlist::bit_ref::net_name(bit.net, nets, interner),
                bit_number: bit.bit_number,
            })
        }
        [BitSource::Literal(value)] => Ok(PinConnection::Literal(*value)),
        [BitSource::Unknown] => Err("unknown literal pin connection is not supported".to_string()),
        _ => Err("standard-cell pin connections must be scalar".to_string()),
    }
}

fn collect_module_io_bits(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
    resolved: &mut ResolvedBitValues,
) -> Result<(), String> {
    for port in &normalized.ports {
        if port.direction == PortDirection::Input {
            let port_name = interner.resolve(port.name).unwrap();
            let bv = gb.add_input(port_name.to_string(), port.bits.len());
            for (bit_idx, bit) in port.bits.iter().copied().zip(bv.iter_lsb_to_msb()) {
                resolved.write_bit(bit_idx, *bit, normalized, nets, interner)?;
            }
        }
    }
    Ok(())
}

fn collect_module_output_bits(normalized: &NormalizedNetlistModule<'_>) -> HashSet<BitIndex> {
    let mut output_bits = HashSet::new();
    for port in &normalized.ports {
        if port.direction != PortDirection::Output {
            continue;
        }
        output_bits.extend(port.bits.iter().copied());
    }
    output_bits
}

fn build_instance_input_map(
    inst: &NormalizedInstance,
    pin_directions: &HashMap<&str, i32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    resolved: &ResolvedBitValues,
    gb: &mut GateBuilder,
    normalized: &NormalizedNetlistModule<'_>,
) -> (
    HashMap<String, AigOperand>,
    Vec<String>,
    HashMap<String, String>,
) {
    build_instance_input_map_skipping_pin(
        inst,
        pin_directions,
        interner,
        nets,
        resolved,
        gb,
        normalized,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_instance_input_map_skipping_pin(
    inst: &NormalizedInstance,
    pin_directions: &HashMap<&str, i32>,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    nets: &[Net],
    resolved: &ResolvedBitValues,
    gb: &mut GateBuilder,
    normalized: &NormalizedNetlistModule<'_>,
    skipped_input_pin: Option<&str>,
) -> (
    HashMap<String, AigOperand>,
    Vec<String>,
    HashMap<String, String>,
) {
    let mut input_map = HashMap::new();
    let mut missing_inputs = Vec::new();
    let mut port_map = HashMap::new();
    for connection in &inst.connections {
        let port_name = interner.resolve(connection.port).unwrap();
        let pin_dir = *pin_directions.get(port_name).unwrap_or(&0); // 1=OUTPUT, 2=INPUT
        port_map.insert(
            port_name.to_string(),
            normalized.render_sources(connection.bits.as_slice(), nets, interner),
        );
        if pin_dir == 2 {
            if skipped_input_pin == Some(port_name) {
                continue;
            }
            if connection.bits.is_empty() {
                missing_inputs.push(format!("{} (<unconnected>)", port_name));
                continue;
            }
            match resolved.materialize_sources(connection.bits.as_slice(), gb) {
                Ok(Some(bv)) => {
                    if bv.get_bit_count() == 1 {
                        input_map.insert(port_name.to_string(), *bv.get_lsb(0));
                    }
                }
                Ok(None) => missing_inputs.push(format!(
                    "{} ({})",
                    port_name,
                    normalized.render_sources(connection.bits.as_slice(), nets, interner)
                )),
                Err(e) => missing_inputs.push(format!("{} ({})", port_name, e)),
            }
        }
    }
    (input_map, missing_inputs, port_map)
}
// Helpers for DFF identity/inverted overrides.
fn invert_bv(gb: &mut GateBuilder, src: &AigBitVector) -> AigBitVector {
    let width = src.get_bit_count();
    let mut out = AigBitVector::from_lsb_is_index_0(&vec![gb.get_false(); width]);
    for i in 0..width {
        out.set_lsb(i, gb.add_not(*src.get_lsb(i)));
    }
    out
}

fn build_d_bv(
    d_connection: &NormalizedConnection,
    gb: &mut GateBuilder,
    resolved: &ResolvedBitValues,
    invert: bool,
) -> Result<Option<AigBitVector>, String> {
    let Some(bv) = resolved.materialize_sources(d_connection.bits.as_slice(), gb)? else {
        return Ok(None);
    };
    Ok(Some(if invert { invert_bv(gb, &bv) } else { bv }))
}

fn write_bv_to_port_destination(
    inst: &NormalizedInstance,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    resolved: &mut ResolvedBitValues,
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    target_port_ci: &str,
    src_bv: &AigBitVector,
) -> Result<(), String> {
    for connection in &inst.connections {
        let pname = interner.resolve(connection.port).unwrap();
        if !pname.eq_ignore_ascii_case(target_port_ci) {
            continue;
        }
        let output_bits = output_target_bits(connection)?;
        resolved.write_bits(output_bits.as_slice(), src_bv, normalized, nets, interner)?;
    }
    Ok(())
}

fn build_pending_assign_bits(normalized: &NormalizedNetlistModule<'_>) -> Vec<PendingAssignBit> {
    let mut pending = Vec::new();
    for (assign_index, assign) in normalized.assigns.iter().enumerate() {
        for (rhs_bit_index, target) in assign.lhs_bits.iter().copied().enumerate() {
            pending.push(PendingAssignBit {
                assign_index,
                rhs_bit_index,
                target,
            });
        }
    }
    pending
}

fn process_pending_assign_bits(
    pending: Vec<PendingAssignBit>,
    normalized: &NormalizedNetlistModule<'_>,
    resolved: &mut ResolvedBitValues,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
) -> Result<(Vec<PendingAssignBit>, bool), String> {
    let mut next_pending = Vec::new();
    let mut processed_any = false;
    for pending_bit in pending {
        let assign = &normalized.assigns[pending_bit.assign_index];
        let Some(value) = eval_bit_expr(&assign.rhs_bits[pending_bit.rhs_bit_index], resolved, gb)?
        else {
            next_pending.push(pending_bit);
            continue;
        };
        resolved.write_bit(pending_bit.target, value, normalized, nets, interner)?;
        processed_any = true;
    }
    Ok((next_pending, processed_any))
}

/// Implements DFF output overrides for identity (Q=D) and inverted (QN=NOT(D)).
///
/// This bypasses formula emission by directly wiring from the connected `D`
/// bits into the `Q`/`QN` destination bits after normalized bit expansion.
///
/// Returns `true` when this function fully handled this output port; `false`
/// when the instance is not a recognized DFF and normal handling should
/// proceed.
fn handle_dff_identity_override(
    type_name: &str,
    inst_name: &str,
    port_name: &str,
    inst: &NormalizedInstance,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    gb: &mut GateBuilder,
    resolved: &mut ResolvedBitValues,
    normalized: &NormalizedNetlistModule<'_>,
    dff_cells_identity: &std::collections::HashSet<String>,
    dff_cells_inverted: &std::collections::HashSet<String>,
    nets: &[Net],
) -> Result<bool, String> {
    if !dff_cells_identity.contains(type_name) && !dff_cells_inverted.contains(type_name) {
        return Ok(false);
    }
    // Decide identity vs inverted based on the current output port name and
    // membership.
    let (target_port, invert) = if port_name.eq_ignore_ascii_case("q")
        && dff_cells_identity.contains(type_name)
    {
        ("q", false)
    } else if port_name.eq_ignore_ascii_case("qn") && dff_cells_inverted.contains(type_name) {
        ("qn", true)
    } else {
        log::warn!(
            "DFF identity override: unexpected D/Q pin mapping for cell '{}' instance '{}' output '{}'",
            type_name,
            inst_name,
            port_name
        );
        return Ok(true);
    };
    // Resolve D input
    let d_input = inst.connections.iter().find(|connection| {
        let pname = interner.resolve(connection.port).unwrap();
        if pname.eq_ignore_ascii_case("d") {
            true
        } else {
            false
        }
    });
    if d_input.is_none() {
        return Err(format!(
            "DFF identity override: D input not found for cell '{}' instance '{}' (output '{}'). \
This cell was classified as DFF-like but does not expose a 'd' pin. \
Ensure the cell exposes a 'd' pin or do not classify it as DFF-like.",
            type_name, inst_name, target_port
        ));
    }
    // Build D (optionally inverted) and write to destination port.
    if let Some(d_bv) = build_d_bv(d_input.unwrap(), gb, resolved, invert)? {
        write_bv_to_port_destination(
            inst,
            interner,
            resolved,
            normalized,
            nets,
            target_port,
            &d_bv,
        )?;
    } else {
        return Err(format!(
            "DFF override: D net not available for cell '{}' instance '{}' (output '{}'). \
This indicates missing drivers or a DFF classification mismatch; re-run with RUST_LOG=trace to diagnose.",
            type_name, inst_name, target_port
        ));
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_sim::gate_sim::{self, Collect};
    use crate::liberty_model::{Cell, Library, LibraryBuilder, Pin, PinDirection};
    use crate::netlist::parse::{
        Net, NetIndex, NetRef, NetlistInstance, NetlistModule, NetlistPort, Parser, PortDirection,
        TokenScanner,
    };
    use string_interner::{StringInterner, backend::StringBackend};
    use xlsynth::IrBits;

    fn input_pin(builder: &mut LibraryBuilder, name: &str) -> Pin {
        Pin {
            name: builder.intern_string(name).unwrap(),
            direction: PinDirection::Input as i32,
            is_clocking_pin: false,
            ..Default::default()
        }
    }

    fn output_pin(builder: &mut LibraryBuilder, name: &str, function: &str) -> Pin {
        Pin {
            name: builder.intern_string(name).unwrap(),
            direction: PinDirection::Output as i32,
            function: builder.intern_string(function).unwrap(),
            is_clocking_pin: false,
            ..Default::default()
        }
    }

    fn test_cell(
        builder: &mut LibraryBuilder,
        cell_name: &str,
        pin_specs: &[(&str, i32, &str, bool)],
        sequential: Vec<crate::liberty_model::Sequential>,
    ) -> Cell {
        let pins = pin_specs
            .iter()
            .map(|(name, direction, function, is_clocking_pin)| Pin {
                name: builder.intern_string(name).unwrap(),
                direction: *direction,
                function: builder.intern_string(function).unwrap(),
                is_clocking_pin: *is_clocking_pin,
                ..Default::default()
            })
            .collect();
        Cell {
            name: cell_name.to_string(),
            pins,
            area: 1.0,
            sequential,
            ..Default::default()
        }
    }

    fn test_library(
        cell_name: &str,
        pin_specs: &[(&str, i32, &str, bool)],
        sequential: Vec<crate::liberty_model::Sequential>,
    ) -> Library {
        let mut builder = LibraryBuilder::new();
        let cell = test_cell(&mut builder, cell_name, pin_specs, sequential);
        builder.cells = vec![cell];
        builder.finish()
    }

    fn projection_order_test_liberty() -> Library {
        let mut builder = LibraryBuilder::new();
        let cells = vec![
            Cell {
                name: "BUF".to_string().into(),
                pins: vec![
                    input_pin(&mut builder, "A"),
                    output_pin(&mut builder, "Y", "A"),
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            },
            Cell {
                name: "INV".to_string().into(),
                pins: vec![
                    input_pin(&mut builder, "A"),
                    output_pin(&mut builder, "Y", "(!A)"),
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            },
            Cell {
                name: "AO21".to_string().into(),
                pins: vec![
                    input_pin(&mut builder, "A1"),
                    input_pin(&mut builder, "A2"),
                    input_pin(&mut builder, "B"),
                    output_pin(&mut builder, "Y", "((A1 & A2) | B)"),
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            },
            // Similar to ASAP7 `DFFHQx1_ASAP7_75t_R`: a D flip-flop
            // whose Q output is modeled by the override path as Q = D.
            Cell {
                name: "DFFHQ".to_string().into(),
                pins: vec![
                    input_pin(&mut builder, "D"),
                    input_pin(&mut builder, "CLK"),
                    output_pin(&mut builder, "Q", "IQ"),
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            },
            // Similar to ASAP7 `DFFHQNx1_ASAP7_75t_R`: a D flip-flop
            // whose QN output is modeled by the override path as QN = !D.
            Cell {
                name: "DFFHQN".to_string().into(),
                pins: vec![
                    input_pin(&mut builder, "D"),
                    input_pin(&mut builder, "CLK"),
                    output_pin(&mut builder, "QN", "IQN"),
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            },
        ];
        builder.cells = cells;
        builder.finish()
    }

    fn eval_single_output_bit(gate_fn: &GateFn, inputs: &[bool]) -> bool {
        let input_bits = inputs.iter().copied().map(IrBits::bool).collect::<Vec<_>>();
        let got = gate_sim::eval(gate_fn, &input_bits, Collect::None);
        got.outputs[0].get_bit(0).unwrap()
    }

    fn eval_output_by_name(gate_fn: &GateFn, inputs: Vec<IrBits>, output_name: &str) -> IrBits {
        let output_index = gate_fn
            .outputs
            .iter()
            .position(|output| output.name == output_name)
            .unwrap_or_else(|| panic!("output '{output_name}' not found"));
        let got = gate_sim::eval(gate_fn, &inputs, Collect::None);
        got.outputs.into_iter().nth(output_index).unwrap()
    }

    fn project_parsed_netlist(src: &'static str) -> GateFn {
        let mut parser = Parser::new(TokenScanner::from_str(src));
        let modules = parser.parse_file().expect("parse netlist");
        assert_eq!(modules.len(), 1);
        project_gatefn_from_netlist_and_liberty(
            &modules[0],
            &parser.nets,
            &parser.interner,
            &projection_order_test_liberty(),
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect("project parsed netlist")
    }

    fn dffhq_identity_cells() -> HashSet<String> {
        let mut dff_cells_identity = HashSet::new();
        dff_cells_identity.insert("DFFHQ".to_string());
        dff_cells_identity
    }

    fn dffhqn_inverted_cells() -> HashSet<String> {
        let mut dff_cells_inverted = HashSet::new();
        dff_cells_inverted.insert("DFFHQN".to_string());
        dff_cells_inverted
    }

    fn simple_ref(idx: NetIndex) -> NetRef {
        NetRef::Simple(idx)
    }

    fn bit_ref(idx: NetIndex, bit: u32) -> NetRef {
        NetRef::BitSelect(idx, bit)
    }

    fn part_ref(idx: NetIndex, msb: u32, lsb: u32) -> NetRef {
        NetRef::PartSelect(idx, msb, lsb)
    }

    fn bool_ir(values: &[bool]) -> Vec<IrBits> {
        values.iter().copied().map(IrBits::bool).collect()
    }

    #[test]
    fn test_liberty_projection_supports_preserved_concat_assigns() {
        let gate_fn = project_parsed_netlist(
            r#"
module top (a, y);
  input [1:0] a;
  output y;
  wire [1:0] a;
  wire y;
  wire [1:0] tmp;
  assign tmp = {a[0], a[1]};
  BUF u0 (.A(tmp[1]), .Y(y));
endmodule
"#,
        );

        let y_from_bit0_one =
            eval_output_by_name(&gate_fn, vec![IrBits::make_ubits(2, 0b01).unwrap()], "y");
        assert_bits(&y_from_bit0_one, &[true]);
        let y_from_bit0_zero =
            eval_output_by_name(&gate_fn, vec![IrBits::make_ubits(2, 0b10).unwrap()], "y");
        assert_bits(&y_from_bit0_zero, &[false]);
    }

    #[test]
    fn test_liberty_projection_supports_preserved_tran_aliases() {
        let gate_fn = project_parsed_netlist(
            r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire y_alias;
  tran(y, y_alias);
  BUF u0 (.A(a), .Y(y_alias));
endmodule
"#,
        );

        assert!(eval_single_output_bit(&gate_fn, &[true]));
        assert!(!eval_single_output_bit(&gate_fn, &[false]));
    }

    fn assert_bits(bits: &IrBits, expected_lsb_to_msb: &[bool]) {
        assert_eq!(bits.get_bit_count(), expected_lsb_to_msb.len());
        for (i, expected) in expected_lsb_to_msb.iter().enumerate() {
            assert_eq!(
                bits.get_bit(i).unwrap(),
                *expected,
                "bit {i} mismatch in {bits}"
            );
        }
    }

    struct NetlistFixture {
        interner: StringInterner<StringBackend<SymbolU32>>,
        nets: Vec<Net>,
        ports: Vec<NetlistPort>,
        wires: Vec<NetIndex>,
        instances: Vec<NetlistInstance>,
    }

    impl NetlistFixture {
        fn new() -> Self {
            Self {
                interner: StringInterner::new(),
                nets: Vec::new(),
                ports: Vec::new(),
                wires: Vec::new(),
                instances: Vec::new(),
            }
        }

        fn add_net(&mut self, name: &str, width: Option<(u32, u32)>) -> NetIndex {
            let idx = NetIndex(self.nets.len());
            let name = self.interner.get_or_intern(name);
            self.nets.push(Net { name, width });
            idx
        }

        fn input(&mut self, name: &str, width: Option<(u32, u32)>) -> NetIndex {
            let idx = self.add_net(name, width);
            self.ports.push(NetlistPort {
                direction: PortDirection::Input,
                width,
                name: self.nets[idx.0].name,
            });
            idx
        }

        fn output(&mut self, name: &str, width: Option<(u32, u32)>) -> NetIndex {
            let idx = self.add_net(name, width);
            self.ports.push(NetlistPort {
                direction: PortDirection::Output,
                width,
                name: self.nets[idx.0].name,
            });
            idx
        }

        fn wire(&mut self, name: &str, width: Option<(u32, u32)>) -> NetIndex {
            let idx = self.add_net(name, width);
            self.wires.push(idx);
            idx
        }

        fn inst(&mut self, type_name: &str, instance_name: &str, connections: Vec<(&str, NetRef)>) {
            let connections = connections
                .into_iter()
                .map(|(port, net_ref)| (self.interner.get_or_intern(port), net_ref))
                .collect();
            self.instances.push(NetlistInstance {
                type_name: self.interner.get_or_intern(type_name),
                instance_name: self.interner.get_or_intern(instance_name),
                connections,
                inst_lineno: 0,
                inst_colno: 0,
            });
        }

        fn module(&mut self) -> NetlistModule {
            NetlistModule {
                name: self.interner.get_or_intern("top"),
                net_index_range: 0..self.nets.len(),
                ports: self.ports.clone(),
                wires: self.wires.clone(),
                assigns: vec![],
                instances: self.instances.clone(),
            }
        }

        fn project(
            &mut self,
            dff_cells_identity: &HashSet<String>,
            dff_cells_inverted: &HashSet<String>,
        ) -> Result<GateFn, String> {
            let module = self.module();
            project_gatefn_from_netlist_and_liberty(
                &module,
                &self.nets,
                &self.interner,
                &projection_order_test_liberty(),
                dff_cells_identity,
                dff_cells_inverted,
            )
        }

        fn project_plain(&mut self) -> Result<GateFn, String> {
            self.project(&HashSet::new(), &HashSet::new())
        }
    }

    #[test]
    fn test_inverter_projection() {
        // Build a minimal netlist and Liberty proto for an inverter
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let invx1 = interner.get_or_intern("INV");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: invx1,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("A"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(0)),
                ),
                (
                    interner.get_or_intern("Y"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(1)),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "INV",
            &[("A", 2, "", false), ("Y", 1, "(!A)", false)],
            vec![],
        );
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap();
        let s = gate_fn.to_string();
        assert!(s.contains("not("), "GateFn output: {}", s);
    }

    #[test]
    fn test_unconnected_output_is_ignored() {
        // Cell with output Y unconnected should not error.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let invx1 = interner.get_or_intern("INV");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: invx1,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("A"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("Y"), NetRef::Unconnected),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "INV",
            &[("A", 2, "", false), ("Y", 1, "(!A)", false)],
            vec![],
        );
        let res = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        );
        assert!(
            res.is_ok(),
            "projection should tolerate unconnected output: {:?}",
            res
        );
    }

    #[test]
    fn test_unconnected_input_errors_clearly() {
        // Cell with input A unconnected should surface a clear error.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let y = interner.get_or_intern("y");
        let and2 = interner.get_or_intern("AND2");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![Net {
            name: y,
            width: None,
        }];
        let ports = vec![NetlistPort {
            direction: PortDirection::Output,
            width: None,
            name: y,
        }];
        let instances = vec![NetlistInstance {
            type_name: and2,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("A"), NetRef::Unconnected),
                (interner.get_or_intern("B"), NetRef::Unconnected),
                (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(0))),
            ],
            inst_lineno: 10,
            inst_colno: 5,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "AND2",
            &[
                ("A", 2, "", false),
                ("B", 2, "", false),
                ("Y", 1, "(A & B)", false),
            ],
            vec![],
        );
        let err = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect_err("should error on unconnected input");
        assert!(err.contains("missing inputs"), "{}", err);
        assert!(err.contains("<unconnected>"), "{}", err);
    }

    #[test]
    fn test_bitselect_output_projection() {
        // Build a netlist and Liberty proto for a buffer with bit-select output
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let n = interner.get_or_intern("n");
        let buf = interner.get_or_intern("BUF");
        let u1 = interner.get_or_intern("u1");
        // n is a 4-bit net
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: n,
                width: Some((3, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((3, 0)),
                name: n,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: buf,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("A"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(0)),
                ),
                (
                    interner.get_or_intern("Y"),
                    NetRef::BitSelect(crate::netlist::parse::NetIndex(1), 3),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library("BUF", &[("A", 2, "", false), ("Y", 1, "A", false)], vec![]);
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect("BitSelect output should be supported");
        let s = gate_fn.to_string();
        // n[3] should be assigned to a nonzero node (not literal false)
        assert!(
            s.contains("n[3] = %1") || s.contains("n[3] = %"),
            "GateFn output: {}",
            s
        );
    }

    #[test]
    fn test_liberty_projection_waits_for_selected_vector_bit_before_using_it() {
        let mut nl = NetlistFixture::new();
        nl.input("n0", None);
        let n1 = nl.input("n1", None);
        let n2 = nl.input("n2", None);
        let ctrl = nl.input("ctrl", None);
        let other = nl.input("other", None);
        let clk = nl.input("clk", None);
        let y = nl.output("y", None);
        let exp_bus = nl.wire("exp_bus", Some((7, 0)));

        nl.inst(
            "DFFHQ",
            "exp_reg_2",
            vec![
                ("D", simple_ref(n2)),
                ("CLK", simple_ref(clk)),
                ("Q", bit_ref(exp_bus, 2)),
            ],
        );
        nl.inst(
            "AO21",
            "use_bit_1",
            vec![
                ("A1", bit_ref(exp_bus, 1)),
                ("A2", simple_ref(ctrl)),
                ("B", simple_ref(other)),
                ("Y", simple_ref(y)),
            ],
        );
        nl.inst(
            "DFFHQN",
            "exp_reg_1",
            vec![
                ("D", simple_ref(n1)),
                ("CLK", simple_ref(clk)),
                ("QN", bit_ref(exp_bus, 1)),
            ],
        );
        let gate_fn = nl
            .project(&dffhq_identity_cells(), &dffhqn_inverted_cells())
            .expect("projection should wait for exp_bus[1] before AO21");

        // Inputs are n0, n1, n2, ctrl, other, clk. With n1=0, ctrl=1,
        // other=0, the AO21 should see exp_bus[1]=~n1 and drive y=1.
        assert!(eval_single_output_bit(
            &gate_fn,
            &[false, false, false, true, false, false]
        ));
    }

    #[test]
    fn test_liberty_projection_waits_for_combinational_vector_bit_before_using_it() {
        let mut nl = NetlistFixture::new();
        let n1 = nl.input("n1", None);
        let n2 = nl.input("n2", None);
        let ctrl = nl.input("ctrl", None);
        let other = nl.input("other", None);
        let y = nl.output("y", None);
        let exp_bus = nl.wire("exp_bus", Some((7, 0)));

        nl.inst(
            "BUF",
            "drive_bit_2",
            vec![("A", simple_ref(n2)), ("Y", bit_ref(exp_bus, 2))],
        );
        nl.inst(
            "AO21",
            "use_bit_1",
            vec![
                ("A1", bit_ref(exp_bus, 1)),
                ("A2", simple_ref(ctrl)),
                ("B", simple_ref(other)),
                ("Y", simple_ref(y)),
            ],
        );
        nl.inst(
            "INV",
            "drive_bit_1",
            vec![("A", simple_ref(n1)), ("Y", bit_ref(exp_bus, 1))],
        );
        let gate_fn = nl
            .project_plain()
            .expect("projection should wait for combinational exp_bus[1] driver");

        // Inputs are n1, n2, ctrl, other. With n1=0, ctrl=1, other=0,
        // the AO21 should see exp_bus[1]=~n1 and drive y=1.
        assert!(eval_single_output_bit(
            &gate_fn,
            &[false, false, true, false]
        ));
    }

    #[test]
    fn test_liberty_projection_preflight_counts_partselect_output_driver() {
        let mut nl = NetlistFixture::new();
        let a = nl.input("a", None);
        let ctrl = nl.input("ctrl", None);
        let other = nl.input("other", None);
        let bus = nl.output("bus", Some((3, 0)));
        let y = nl.output("y", None);

        nl.inst(
            "BUF",
            "drive_bus_bit_3",
            vec![("A", simple_ref(a)), ("Y", part_ref(bus, 3, 3))],
        );
        nl.inst(
            "AO21",
            "use_bus_bit_3",
            vec![
                ("A1", bit_ref(bus, 3)),
                ("A2", simple_ref(ctrl)),
                ("B", simple_ref(other)),
                ("Y", simple_ref(y)),
            ],
        );
        let gate_fn = nl
            .project_plain()
            .expect("part-select output should count as a bus driver during preflight");

        let inputs = bool_ir(&[true, true, false]);
        let bus_bits = eval_output_by_name(&gate_fn, inputs.clone(), "bus");
        let y_bits = eval_output_by_name(&gate_fn, inputs, "y");
        assert_bits(&bus_bits, &[false, false, false, true]);
        assert_bits(&y_bits, &[true]);
    }

    #[test]
    fn test_liberty_projection_rejects_duplicate_bit_drivers() {
        let mut nl = NetlistFixture::new();
        let a = nl.input("a", None);
        let b = nl.input("b", None);
        let y = nl.output("y", Some((3, 0)));

        nl.inst(
            "BUF",
            "u0",
            vec![("A", simple_ref(a)), ("Y", bit_ref(y, 1))],
        );
        nl.inst(
            "BUF",
            "u1",
            vec![("A", simple_ref(b)), ("Y", bit_ref(y, 1))],
        );
        let err = nl
            .project_plain()
            .expect_err("duplicate bit driver should be rejected");
        assert!(err.contains("y"), "unexpected error: {err}");
        assert!(err.contains("bit 1"), "unexpected error: {err}");
    }

    #[test]
    fn test_liberty_projection_waits_for_partselect_bits_before_dff_override() {
        let mut nl = NetlistFixture::new();
        let b1 = nl.input("b1", None);
        let b2 = nl.input("b2", None);
        let y = nl.output("y", Some((1, 0)));
        let bus = nl.wire("bus", Some((2, 0)));

        nl.inst(
            "BUF",
            "drive_bit_2",
            vec![("A", simple_ref(b2)), ("Y", bit_ref(bus, 2))],
        );
        nl.inst(
            "DFFHQ",
            "capture_slice",
            vec![("D", part_ref(bus, 2, 1)), ("Q", simple_ref(y))],
        );
        nl.inst(
            "BUF",
            "drive_bit_1",
            vec![("A", simple_ref(b1)), ("Y", bit_ref(bus, 1))],
        );
        let gate_fn = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect("DFF override should wait for every selected D bit");

        let y_bits = eval_output_by_name(&gate_fn, bool_ir(&[true, false]), "y");
        assert_bits(&y_bits, &[true, false]);
    }

    #[test]
    fn test_liberty_projection_defaults_only_final_output_missing_bits() {
        let mut nl = NetlistFixture::new();
        let a = nl.input("a", None);
        let b = nl.input("b", None);
        let ctrl = nl.input("ctrl", None);
        let other = nl.input("other", None);
        let out_bus = nl.output("out_bus", Some((3, 0)));
        let y = nl.output("y", None);

        nl.inst(
            "BUF",
            "drive_output_bit_2",
            vec![("A", simple_ref(a)), ("Y", bit_ref(out_bus, 2))],
        );
        nl.inst(
            "AO21",
            "use_output_bit_1",
            vec![
                ("A1", bit_ref(out_bus, 1)),
                ("A2", simple_ref(ctrl)),
                ("B", simple_ref(other)),
                ("Y", simple_ref(y)),
            ],
        );
        nl.inst(
            "INV",
            "drive_output_bit_1",
            vec![("A", simple_ref(b)), ("Y", bit_ref(out_bus, 1))],
        );
        let gate_fn = nl
            .project_plain()
            .expect("internal output-net consumer should wait for the real selected bit");

        let inputs = bool_ir(&[true, false, true, false]);
        let out_bits = eval_output_by_name(&gate_fn, inputs.clone(), "out_bus");
        let y_bits = eval_output_by_name(&gate_fn, inputs, "y");
        assert_bits(&out_bits, &[false, true, true, false]);
        assert_bits(&y_bits, &[true]);
    }

    #[test]
    fn test_liberty_projection_honors_ascending_packed_range_bit_numbers() {
        let mut nl = NetlistFixture::new();
        let a = nl.input("a", None);
        let b = nl.input("b", None);
        let ctrl = nl.input("ctrl", None);
        let other = nl.input("other", None);
        let bus = nl.output("bus", Some((0, 3)));
        let y = nl.output("y", None);

        nl.inst(
            "BUF",
            "drive_bit_2",
            vec![("A", simple_ref(b)), ("Y", bit_ref(bus, 2))],
        );
        nl.inst(
            "AO21",
            "use_bit_1",
            vec![
                ("A1", bit_ref(bus, 1)),
                ("A2", simple_ref(ctrl)),
                ("B", simple_ref(other)),
                ("Y", simple_ref(y)),
            ],
        );
        nl.inst(
            "INV",
            "drive_bit_1",
            vec![("A", simple_ref(a)), ("Y", bit_ref(bus, 1))],
        );
        let gate_fn = nl
            .project_plain()
            .expect("ascending packed ranges should use declared bit numbers");

        let inputs = bool_ir(&[false, true, true, false]);
        let bus_bits = eval_output_by_name(&gate_fn, inputs.clone(), "bus");
        let y_bits = eval_output_by_name(&gate_fn, inputs, "y");
        assert_bits(&bus_bits, &[false, true, true, false]);
        assert_bits(&y_bits, &[true]);
    }

    #[test]
    fn test_liberty_projection_rejects_simple_output_overlap_with_bit_driver() {
        let mut nl = NetlistFixture::new();
        let x = nl.input("x", Some((3, 0)));
        let a = nl.input("a", None);
        let bus = nl.output("bus", Some((3, 0)));

        nl.inst(
            "DFFHQ",
            "drive_whole_bus",
            vec![("D", simple_ref(x)), ("Q", simple_ref(bus))],
        );
        nl.inst(
            "BUF",
            "drive_bit_1",
            vec![("A", simple_ref(a)), ("Y", bit_ref(bus, 1))],
        );
        let err = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect_err("whole-vector write should conflict with later bit write");
        assert!(err.contains("bus"), "unexpected error: {err}");
        assert!(err.contains("bit 1"), "unexpected error: {err}");
    }

    #[test]
    fn test_liberty_projection_rejects_partselect_output_overlap_with_bit_driver() {
        let mut nl = NetlistFixture::new();
        let x = nl.input("x", Some((1, 0)));
        let a = nl.input("a", None);
        let bus = nl.output("bus", Some((3, 0)));

        nl.inst(
            "DFFHQ",
            "drive_bus_slice",
            vec![("D", simple_ref(x)), ("Q", part_ref(bus, 2, 1))],
        );
        nl.inst(
            "BUF",
            "drive_bit_1",
            vec![("A", simple_ref(a)), ("Y", bit_ref(bus, 1))],
        );
        let err = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect_err("slice write should conflict with later overlapping bit write");
        assert!(err.contains("bus"), "unexpected error: {err}");
        assert!(err.contains("bit 1"), "unexpected error: {err}");
    }

    #[test]
    fn test_dff_override_reads_partselect_and_writes_bitselect() {
        let mut nl = NetlistFixture::new();
        let x = nl.input("x", Some((3, 0)));
        let y = nl.output("y", Some((3, 0)));

        nl.inst(
            "DFFHQ",
            "capture_single_bit_slice",
            vec![("D", part_ref(x, 1, 1)), ("Q", bit_ref(y, 2))],
        );
        let gate_fn = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect("DFF override should read a one-bit part-select and write a bit-select");

        let y_bits =
            eval_output_by_name(&gate_fn, vec![IrBits::make_ubits(4, 0b0010).unwrap()], "y");
        assert_bits(&y_bits, &[false, false, true, false]);
    }

    #[test]
    fn test_liberty_projection_waits_for_full_vector_before_dff_override() {
        let mut nl = NetlistFixture::new();
        let low = nl.input("low", None);
        let high = nl.input("high", None);
        let y = nl.output("y", Some((1, 0)));
        let bus = nl.wire("bus", Some((1, 0)));

        nl.inst(
            "BUF",
            "drive_high",
            vec![("A", simple_ref(high)), ("Y", bit_ref(bus, 1))],
        );
        nl.inst(
            "DFFHQ",
            "capture_bus",
            vec![("D", simple_ref(bus)), ("Q", simple_ref(y))],
        );
        nl.inst(
            "BUF",
            "drive_low",
            vec![("A", simple_ref(low)), ("Y", bit_ref(bus, 0))],
        );
        let gate_fn = nl
            .project(&dffhq_identity_cells(), &HashSet::new())
            .expect("DFF override should wait for every bit of a full-vector D input");

        let y_bits = eval_output_by_name(&gate_fn, bool_ir(&[true, false]), "y");
        assert_bits(&y_bits, &[true, false]);
    }

    #[test]
    fn test_dff_identity_q_bitselect_expands_to_full_width() {
        // Build a netlist with a DFF cell where Q drives a single bit of a 4-bit output
        // net. Regression: ensure we size the destination vector to the full
        // net width (4), not just the bit index slice width (1), and that no
        // width-assertion fires.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let dff = interner.get_or_intern("DFF");
        let u1 = interner.get_or_intern("u1");

        // a: 1-bit input, y: 4-bit output
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: Some((3, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((3, 0)),
                name: y,
            },
        ];
        // Connect D = a, Q = y[3]
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("D"),
                    NetRef::Simple(crate::netlist::parse::NetIndex(0)),
                ),
                (
                    interner.get_or_intern("Q"),
                    NetRef::BitSelect(crate::netlist::parse::NetIndex(1), 3),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        // Liberty with a DFF cell having pins D (input) and Q (output)
        let liberty_lib = test_library("DFF", &[("D", 2, "", false), ("Q", 1, "Q", false)], vec![]);
        let mut dff_cells = std::collections::HashSet::new();
        dff_cells.insert("DFF".to_string());

        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells,
            &HashSet::new(),
        )
        .expect("projection should succeed without width mismatch");

        // Signature must show full 4-bit output width.
        assert_eq!(gate_fn.get_signature(), "fn top(a: bits[1]) -> bits[4]");
        // And we should be writing to y[3] (the selected bit) in the output mapping.
        let s = gate_fn.to_string();
        assert!(s.contains("  y[3] = "), "GateFn output: {}", s);
    }

    #[test]
    fn test_dff_identity_q_partselect_writes_at_offset() {
        // DFF cell where Q drives a 4-bit slice y[7:4] of an 8-bit output net.
        // Regression: ensure we size to full width (8) and write at lsb offset 4.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let dff = interner.get_or_intern("DFF");
        let u1 = interner.get_or_intern("u1");

        // x: 8-bit input, y: 8-bit output
        let nets = vec![
            Net {
                name: x,
                width: Some((7, 0)),
            },
            Net {
                name: y,
                width: Some((7, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: Some((7, 0)),
                name: x,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((7, 0)),
                name: y,
            },
        ];
        // Connect D = x[7:4], Q = y[7:4]
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("D"),
                    NetRef::PartSelect(crate::netlist::parse::NetIndex(0), 7, 4),
                ),
                (
                    interner.get_or_intern("Q"),
                    NetRef::PartSelect(crate::netlist::parse::NetIndex(1), 7, 4),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library("DFF", &[("D", 2, "", false), ("Q", 1, "Q", false)], vec![]);
        let mut dff_cells = std::collections::HashSet::new();
        dff_cells.insert("DFF".to_string());

        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells,
            &HashSet::new(),
        )
        .expect("projection should succeed without width mismatch");

        // Signature must show full 8-bit output width.
        assert_eq!(gate_fn.get_signature(), "fn top(x: bits[8]) -> bits[8]");
        // And we should be writing to y[4]..y[7] in the output mapping.
        let s = gate_fn.to_string();
        assert!(s.contains("  y[4] = "), "GateFn output: {}", s);
        assert!(s.contains("  y[7] = "), "GateFn output: {}", s);
    }

    #[test]
    fn test_dff_identity_override_simple() {
        // D and Q are both Simple netrefs
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let q = interner.get_or_intern("q");
        let dff = interner.get_or_intern("DFF");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("d"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("q"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library("DFF", &[("d", 2, "", false), ("q", 1, "d", false)], vec![]);
        let mut dff_cells = std::collections::HashSet::new();
        dff_cells.insert("DFF".to_string());
        // Set up the input value for D
        let mut gb = GateBuilder::new("top".to_string(), GateBuilderOptions::no_opt());
        let d_bv = gb.add_input("d".to_string(), 1);
        let mut net_to_bv = HashMap::new();
        net_to_bv.insert(NetIndex(0), d_bv.clone());
        // Run the projection
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells,
            &HashSet::new(),
        )
        .unwrap();
        // The output should be wired directly from D
        let q_input = gate_fn.inputs.iter().find(|i| i.name == "d").unwrap();
        let q_output = gate_fn.outputs.iter().find(|o| o.name == "q").unwrap();
        assert_eq!(
            q_input.bit_vector.get_lsb(0),
            q_output.bit_vector.get_lsb(0)
        );
    }

    #[test]
    fn test_dff_inverted_qn_simple() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let qn = interner.get_or_intern("qn");
        let dffn = interner.get_or_intern("DFFN");
        let u1 = interner.get_or_intern("u1");

        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: qn,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: qn,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dffn,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("QN"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "DFFN",
            &[("D", 2, "", false), ("QN", 1, "IQN", false)],
            vec![],
        );
        let dff_cells_identity = HashSet::new();
        let mut dff_cells_inverted = HashSet::new();
        dff_cells_inverted.insert("DFFN".to_string());
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells_identity,
            &dff_cells_inverted,
        )
        .unwrap();
        let s = gate_fn.to_string();
        assert!(s.contains("qn[0] = not("), "GateFn output: {}", s);
    }

    #[test]
    fn test_dff_inverted_qn_bitselect() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let y = interner.get_or_intern("y");
        let dffn = interner.get_or_intern("DFFN");
        let u1 = interner.get_or_intern("u1");

        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: y,
                width: Some((3, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((3, 0)),
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dffn,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (
                    interner.get_or_intern("QN"),
                    NetRef::BitSelect(NetIndex(1), 2),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "DFFN",
            &[("D", 2, "", false), ("QN", 1, "IQN", false)],
            vec![],
        );
        let dff_cells_identity = HashSet::new();
        let mut dff_cells_inverted = HashSet::new();
        dff_cells_inverted.insert("DFFN".to_string());
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells_identity,
            &dff_cells_inverted,
        )
        .unwrap();
        let s = gate_fn.to_string();
        assert!(s.contains("  y[2] = not("), "GateFn output: {}", s);
    }

    #[test]
    fn test_dff_inverted_qn_partselect() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let x = interner.get_or_intern("x");
        let y = interner.get_or_intern("y");
        let dffn = interner.get_or_intern("DFFN");
        let u1 = interner.get_or_intern("u1");

        let nets = vec![
            Net {
                name: x,
                width: Some((7, 0)),
            },
            Net {
                name: y,
                width: Some((7, 0)),
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: Some((7, 0)),
                name: x,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: Some((7, 0)),
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dffn,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("D"),
                    NetRef::PartSelect(NetIndex(0), 7, 4),
                ),
                (
                    interner.get_or_intern("QN"),
                    NetRef::PartSelect(NetIndex(1), 7, 4),
                ),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "DFFN",
            &[("D", 2, "", false), ("QN", 1, "IQN", false)],
            vec![],
        );
        let dff_cells_identity = HashSet::new();
        let mut dff_cells_inverted = HashSet::new();
        dff_cells_inverted.insert("DFFN".to_string());
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells_identity,
            &dff_cells_inverted,
        )
        .unwrap();
        let s = gate_fn.to_string();
        assert!(s.contains("  y[4] = not("), "GateFn output: {}", s);
        assert!(s.contains("  y[7] = not("), "GateFn output: {}", s);
    }

    #[test]
    fn test_sequential_iqn_inverts_next_state_when_collapsing() {
        // Collapsing sequential state variables should treat IQN as the complement of
        // IQ.
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let q = interner.get_or_intern("q");
        let qn = interner.get_or_intern("qn");
        // Synthetic cell name, not an ASAP7-specific alias.
        let dff = interner.get_or_intern("DFFX");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
            Net {
                name: qn,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: qn,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(1))),
                (interner.get_or_intern("QN"), NetRef::Simple(NetIndex(2))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "DFFX",
            &[
                ("D", 2, "", false),
                ("Q", 1, "IQ", false),
                ("QN", 1, "IQN", false),
            ],
            vec![crate::liberty_model::Sequential {
                state_var: "IQ".to_string(),
                next_state: "D".to_string(),
                clock_expr: "CLK".to_string(),
                kind: crate::liberty_model::SequentialKind::Ff as i32,
                clear_expr: String::new(),
                preset_expr: String::new(),
                complementary_state_var: Some("IQN".to_string()),
            }],
        );
        let gate_fn = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .unwrap();
        let d_input = gate_fn
            .inputs
            .iter()
            .find(|input| input.name == "d")
            .unwrap();
        let q_output = gate_fn
            .outputs
            .iter()
            .find(|output| output.name == "q")
            .unwrap();
        let qn_output = gate_fn
            .outputs
            .iter()
            .find(|output| output.name == "qn")
            .unwrap();
        let d_bit = *d_input.bit_vector.get_lsb(0);
        let q_bit = *q_output.bit_vector.get_lsb(0);
        let qn_bit = *qn_output.bit_vector.get_lsb(0);
        assert_eq!(q_bit, d_bit);
        assert_eq!(qn_bit, d_bit.negate());
    }

    #[test]
    fn test_collapse_ignores_unsupported_unused_cells() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let inv = interner.get_or_intern("INV");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: inv,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("A"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        // Include an unsupported sequential cell in the library, but do not instantiate
        // it.
        let mut builder = LibraryBuilder::new();
        let inv = test_cell(
            &mut builder,
            "INV",
            &[("A", 2, "", false), ("Y", 1, "!A", false)],
            vec![],
        );
        let dffen = test_cell(
            &mut builder,
            "DFFEN",
            &[
                ("D", 2, "", false),
                ("EN", 2, "", false),
                ("CLK", 2, "", true),
                ("Q", 1, "IQ", false),
            ],
            vec![crate::liberty_model::Sequential {
                state_var: "IQ".to_string(),
                next_state: "(!EN * IQ) + (EN * D)".to_string(),
                clock_expr: "CLK".to_string(),
                kind: crate::liberty_model::SequentialKind::Ff as i32,
                clear_expr: String::new(),
                preset_expr: String::new(),
                complementary_state_var: None,
            }],
        );
        builder.cells = vec![inv, dffen];
        let liberty_lib = builder.finish();
        let gate_fn = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect("unused unsupported sequential cells should not cause collapse to fail");
        let s = gate_fn.to_string();
        assert!(s.contains("y[0] = not("), "GateFn output: {s}");
    }

    #[test]
    fn test_collapse_errors_on_async_flop_controls() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let rn = interner.get_or_intern("rn");
        let q = interner.get_or_intern("q");
        // Approximate ASAP7 async-reset flavor (e.g. DFFASRHQNx1_ASAP7_75t_R).
        let dff = interner.get_or_intern("DFFAR");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: rn,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: rn,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("RN"), NetRef::Simple(NetIndex(1))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(2))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "DFFAR",
            &[
                ("D", 2, "", false),
                ("RN", 2, "", false),
                ("CLK", 2, "", true),
                ("Q", 1, "IQ", false),
            ],
            vec![crate::liberty_model::Sequential {
                state_var: "IQ".to_string(),
                next_state: "D".to_string(),
                clock_expr: "CLK".to_string(),
                kind: crate::liberty_model::SequentialKind::Ff as i32,
                clear_expr: "!RN".to_string(),
                preset_expr: String::new(),
                complementary_state_var: None,
            }],
        );
        let err = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect_err("async flop controls should be rejected when collapsing");
        assert!(
            err.contains("async clear/preset"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_collapse_errors_on_latch_enable() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let clk = interner.get_or_intern("clk");
        let q = interner.get_or_intern("q");
        let lat = interner.get_or_intern("LATCH");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: clk,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: clk,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: lat,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("CLK"), NetRef::Simple(NetIndex(1))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(2))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "LATCH",
            &[
                ("D", 2, "", false),
                ("CLK", 2, "", true),
                ("Q", 1, "IQ", false),
            ],
            vec![crate::liberty_model::Sequential {
                state_var: "IQ".to_string(),
                next_state: "D".to_string(),
                clock_expr: "CLK".to_string(),
                kind: crate::liberty_model::SequentialKind::Latch as i32,
                clear_expr: String::new(),
                preset_expr: String::new(),
                complementary_state_var: None,
            }],
        );
        let err = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect_err("latch enable should be rejected when collapsing");
        assert!(
            err.contains("latches with enable"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_collapse_errors_on_state_var_references_in_next_state() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let en = interner.get_or_intern("en");
        let q = interner.get_or_intern("q");
        let dff = interner.get_or_intern("DFFEN");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: en,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: en,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("EN"), NetRef::Simple(NetIndex(1))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(2))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "DFFEN",
            &[
                ("D", 2, "", false),
                ("EN", 2, "", false),
                ("CLK", 2, "", true),
                ("Q", 1, "IQ", false),
            ],
            vec![crate::liberty_model::Sequential {
                state_var: "IQ".to_string(),
                next_state: "(!EN * IQ) + (EN * D)".to_string(),
                clock_expr: "CLK".to_string(),
                kind: crate::liberty_model::SequentialKind::Ff as i32,
                clear_expr: String::new(),
                preset_expr: String::new(),
                complementary_state_var: None,
            }],
        );
        let err = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect_err("state-var references should be rejected when collapsing");
        assert!(
            err.contains("reference state variables"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_collapse_errors_on_non_pin_inputs_in_next_state() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let d = interner.get_or_intern("d");
        let q = interner.get_or_intern("q");
        // Approximate ASAP7 QN-only flop naming (e.g. DFFLQNx1_ASAP7_75t_R).
        let dff = interner.get_or_intern("DFFNP");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: d,
                width: None,
            },
            Net {
                name: q,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: d,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: q,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (interner.get_or_intern("D"), NetRef::Simple(NetIndex(0))),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "DFFNP",
            &[
                ("D", 2, "", false),
                ("CLK", 2, "", true),
                ("Q", 1, "IQ", false),
            ],
            vec![crate::liberty_model::Sequential {
                state_var: "IQ".to_string(),
                next_state: "D & FOO".to_string(),
                clock_expr: "CLK".to_string(),
                kind: crate::liberty_model::SequentialKind::Ff as i32,
                clear_expr: String::new(),
                preset_expr: String::new(),
                complementary_state_var: None,
            }],
        );
        let err = project_gatefn_from_netlist_and_liberty_with_options(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
            &GateFnProjectOptions {
                collapse_sequential: true,
            },
        )
        .expect_err("non-pin inputs should be rejected when collapsing");
        assert!(err.contains("non-pin inputs"), "unexpected error: {err}");
    }

    #[test]
    fn test_dff_override_allows_literal_d_input() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let q = interner.get_or_intern("q");
        let dff = interner.get_or_intern("DFF");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![Net {
            name: q,
            width: None,
        }];
        let ports = vec![NetlistPort {
            direction: PortDirection::Output,
            width: None,
            name: q,
        }];
        let instances = vec![NetlistInstance {
            type_name: dff,
            instance_name: u1,
            connections: vec![
                (
                    interner.get_or_intern("D"),
                    NetRef::Literal(xlsynth::IrBits::make_ubits(1, 1).unwrap()),
                ),
                (interner.get_or_intern("Q"), NetRef::Simple(NetIndex(0))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib =
            test_library("DFF", &[("D", 2, "", false), ("Q", 1, "IQ", false)], vec![]);
        let mut dff_cells_identity = HashSet::new();
        dff_cells_identity.insert("DFF".to_string());
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &dff_cells_identity,
            &HashSet::new(),
        )
        .expect("DFF override should accept literal D inputs");
        let s = gate_fn.to_string();
        assert!(
            s.contains("literal(true)") || (s.contains("literal(false)") && s.contains("not(")),
            "expected constant-true behavior in gatefn, got: {s}"
        );
    }

    #[test]
    fn test_instance_input_tied_to_literal() {
        // Build a netlist and Liberty proto for a 2-input AND gate, with one input tied
        // to 1'b0
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let a = interner.get_or_intern("a");
        let y = interner.get_or_intern("y");
        let and2 = interner.get_or_intern("AND2");
        let u1 = interner.get_or_intern("u1");
        let nets = vec![
            Net {
                name: a,
                width: None,
            },
            Net {
                name: y,
                width: None,
            },
        ];
        let ports = vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: a,
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: y,
            },
        ];
        let instances = vec![NetlistInstance {
            type_name: and2,
            instance_name: u1,
            connections: vec![
                // .A(a), .B(1'b0), .Y(y)
                (interner.get_or_intern("A"), NetRef::Simple(NetIndex(0))),
                (
                    interner.get_or_intern("B"),
                    NetRef::Literal(xlsynth::IrBits::make_ubits(1, 0).unwrap()),
                ),
                (interner.get_or_intern("Y"), NetRef::Simple(NetIndex(1))),
            ],
            inst_lineno: 0,
            inst_colno: 0,
        }];
        let module = NetlistModule {
            name: interner.get_or_intern("top"),
            net_index_range: 0..nets.len(),
            ports,
            wires: vec![],
            assigns: vec![],
            instances,
        };
        let liberty_lib = test_library(
            "AND2",
            &[
                ("A", 2, "", false),
                ("B", 2, "", false),
                ("Y", 1, "(A & B)", false),
            ],
            vec![],
        );
        let gate_fn = project_gatefn_from_netlist_and_liberty(
            &module,
            &nets,
            &interner,
            &liberty_lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap();
        let s = gate_fn.to_string();
        // The output should be always false (since B is tied to 0)
        assert!(
            s.contains("literal(false)") || s.contains("= %0"),
            "GateFn output: {}",
            s
        );
    }
}
