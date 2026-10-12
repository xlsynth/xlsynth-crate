// SPDX-License-Identifier: Apache-2.0

//! Bounded direct generation of combinational and sequential AIG designs.
//!
//! The same construction engine accepts coverage-guided bytes or seeded RNG
//! entropy. Registers are sources in the combinational DAG; feedback exists
//! only across their Q/D boundary. Explicit controls are not legalized here.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::RangeInclusive;

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_pir::ir_random::{EntropySource, RngEntropy};
use xlsynth_pir::random_inputs::generate_uniform_irbits;

use crate::aig::{
    AigBitVector, AigOperand, ClockPort, GateFn, RegisterBinding, RegisterControl,
    SequentialGateFn, SynchronousReset, TransitionInputId, TransitionOutputId,
};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

/// Invalid options or a violation of the direct generator's construction
/// contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateGenerationError {
    InvalidOptions(String),
    Construction(String),
}

impl fmt::Display for GateGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(message) => write!(f, "invalid random G8R options: {message}"),
            Self::Construction(message) => write!(f, "random G8R construction failed: {message}"),
        }
    }
}

impl std::error::Error for GateGenerationError {}

/// Boolean construction operations; all lower to AND nodes and complemented
/// edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RandomGateOperation {
    And,
    Or,
    Xor,
    Xnor,
    Mux,
}

impl RandomGateOperation {
    /// Maximum new AIG nodes needed, including when folding/hashing is
    /// disabled.
    fn node_cost(self) -> usize {
        match self {
            Self::And | Self::Or => 1,
            Self::Xor | Self::Xnor | Self::Mux => 3,
        }
    }
}

/// Selects whether graph construction folds and structurally hashes
/// expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomGateBuilderMode {
    Optimized,
    Unoptimized,
    Either,
}

/// Permitted asserted level of an explicit synchronous register control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomControlPolarity {
    ActiveHigh,
    ActiveLow,
    Either,
}

/// Optional structural stress distributions, disabled in generic generation.
///
/// These caps preserve the bulk-substitution campaign's chain, duplicate, and
/// fanout cases without injecting those structures into unrelated campaigns.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RandomGateShapes {
    pub max_chain_length: usize,
    pub max_redundant_pairs: usize,
    pub max_fanout_branches: usize,
}

/// Shape and capability constraints for the shared direct-G8R generator.
///
/// Widths vary independently per port/register; ranges starting at zero permit
/// zero-width values. Setting registers to 0..=0 yields combinational designs.
/// Enable/reset options allow each register to independently omit that control.
/// Node budgets include the constant and input/Q nodes, not just body
/// operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomGateOptions {
    pub input_ports: RangeInclusive<usize>,
    pub input_width: RangeInclusive<usize>,
    pub output_ports: RangeInclusive<usize>,
    pub output_width: RangeInclusive<usize>,
    pub registers: RangeInclusive<usize>,
    pub register_width: RangeInclusive<usize>,
    pub max_operations: usize,
    pub max_aig_nodes: usize,
    pub operations: Vec<RandomGateOperation>,
    pub builder_mode: RandomGateBuilderMode,
    /// Include constants as candidate operands even when input bits exist.
    /// Input-free designs always have constant false available.
    pub allow_constants: bool,
    pub enable: Option<RandomControlPolarity>,
    pub reset: Option<RandomControlPolarity>,
    /// Permit D/control endpoints to reuse matching external or register
    /// outputs.
    pub reuse_output_endpoints: bool,
    /// Make the first nonempty external output observe a generated result.
    pub observe_generated_logic: bool,
    pub shapes: RandomGateShapes,
}

impl Default for RandomGateOptions {
    fn default() -> Self {
        Self {
            input_ports: 0..=4,
            input_width: 1..=4,
            output_ports: 1..=4,
            output_width: 1..=1,
            registers: 0..=0,
            register_width: 1..=8,
            max_operations: 32,
            max_aig_nodes: 256,
            operations: vec![
                RandomGateOperation::And,
                RandomGateOperation::Or,
                RandomGateOperation::Xor,
                RandomGateOperation::Xnor,
                RandomGateOperation::Mux,
            ],
            builder_mode: RandomGateBuilderMode::Either,
            allow_constants: true,
            enable: Some(RandomControlPolarity::Either),
            reset: Some(RandomControlPolarity::Either),
            reuse_output_endpoints: true,
            observe_generated_logic: true,
            shapes: RandomGateShapes::default(),
        }
    }
}

impl RandomGateOptions {
    /// Rejects inconsistent bounds before reading entropy or constructing
    /// nodes.
    pub fn validate(&self) -> Result<(), GateGenerationError> {
        for (name, range) in [
            ("input_ports", &self.input_ports),
            ("input_width", &self.input_width),
            ("output_ports", &self.output_ports),
            ("output_width", &self.output_width),
            ("registers", &self.registers),
            ("register_width", &self.register_width),
        ] {
            if range.is_empty() || *range.end() == usize::MAX {
                return Err(GateGenerationError::InvalidOptions(format!(
                    "invalid {name} range"
                )));
            }
        }
        if self
            .input_ports
            .end()
            .checked_add(*self.registers.end())
            .is_none()
            || self
                .output_ports
                .end()
                .checked_mul(*self.output_width.end())
                .is_none()
        {
            return Err(GateGenerationError::InvalidOptions(
                "interface bounds overflow usize".into(),
            ));
        }
        if self.max_aig_nodes == 0 {
            return Err(GateGenerationError::InvalidOptions(
                "max_aig_nodes must include constant false".into(),
            ));
        }
        let minimum = self
            .input_ports
            .start()
            .checked_mul(*self.input_width.start())
            .and_then(|n| {
                self.registers
                    .start()
                    .checked_mul(*self.register_width.start())
                    .and_then(|r| n.checked_add(r))
            })
            .and_then(|n| n.checked_add(1));
        if minimum.is_none_or(|n| n > self.max_aig_nodes) {
            return Err(GateGenerationError::InvalidOptions(
                "minimum interface exceeds max_aig_nodes".into(),
            ));
        }
        if self.max_operations != 0 && self.operations.is_empty() {
            return Err(GateGenerationError::InvalidOptions(
                "body generation requires at least one operation".into(),
            ));
        }
        for bound in [
            self.shapes.max_chain_length,
            self.shapes.max_redundant_pairs,
            self.shapes.max_fanout_branches,
        ] {
            if bound == usize::MAX {
                return Err(GateGenerationError::InvalidOptions(
                    "structural shape cap is too large".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Coverage of random body operations, excluding opt-in stress construction.
#[derive(Debug, Clone, Default)]
pub struct GeneratedGateStats {
    /// Completed operation requests before folding or structural hashing.
    pub emitted_operations: BTreeMap<RandomGateOperation, usize>,
    /// Requests that created a new root reachable from an external, D, or
    /// control output. Folded-away requests are deliberately not counted.
    pub live_operations: BTreeMap<RandomGateOperation, usize>,
}

/// Generated graph plus the builder policy needed by reconstruction tests.
#[derive(Debug, Clone)]
pub struct GeneratedGateDesign {
    pub design: SequentialGateFn,
    pub builder_options: GateBuilderOptions,
    pub stats: GeneratedGateStats,
}

/// Register interface allocated before any body logic, avoiding combinational
/// cycles.
struct PendingRegister {
    name: String,
    q: TransitionInputId,
    width: usize,
}

/// Generates one bounded valid design, preserving optional explicit controls.
pub fn generate_gate_design<S: EntropySource>(
    source: &mut S,
    options: &RandomGateOptions,
) -> Result<GeneratedGateDesign, GateGenerationError> {
    options.validate()?;
    // Reserve independent wiring/shape entropy before the variable-length body,
    // so depleted body bytes cannot collapse all outputs onto the first input.
    let wiring_seed = source.take_u64();
    let shape_seed = source.take_u64();
    let optimized = source.take_u64() & 1 != 0;
    let builder_options = match options.builder_mode {
        RandomGateBuilderMode::Optimized => GateBuilderOptions::opt(),
        RandomGateBuilderMode::Unoptimized => GateBuilderOptions::no_opt(),
        RandomGateBuilderMode::Either if optimized => GateBuilderOptions::opt(),
        RandomGateBuilderMode::Either => GateBuilderOptions::no_opt(),
    };
    let mut builder = GateBuilder::new("random_g8r".into(), builder_options);
    let budget = options.max_aig_nodes - 1;
    let min_input_width = *options.input_width.start();
    let min_register_width = *options.register_width.start();
    let reserved_register_bits = options.registers.start() * min_register_width;
    let input_cap = affordable_count(
        *options.input_ports.end(),
        budget - reserved_register_bits,
        min_input_width,
    );
    let input_count = choose(source, *options.input_ports.start()..=input_cap);
    let register_cap = affordable_count(
        *options.registers.end(),
        budget - input_count * min_input_width,
        min_register_width,
    );
    let register_count = choose(source, *options.registers.start()..=register_cap);
    let output_count = choose(source, options.output_ports.clone());
    let mut reserved_bits = input_count * min_input_width + register_count * min_register_width;
    let mut pool = Vec::new();
    let mut pending = Vec::new();
    for index in 0..input_count + register_count {
        let is_register = index >= input_count;
        let range = if is_register {
            &options.register_width
        } else {
            &options.input_width
        };
        reserved_bits -= range.start();
        let max_width =
            (*range.end()).min(options.max_aig_nodes - builder.gates.len() - reserved_bits);
        let width = choose(source, *range.start()..=max_width);
        let name = if is_register {
            format!("r{}__q", index - input_count)
        } else {
            format!("in{index}")
        };
        let value = builder.add_input(name, width);
        pool.extend(value.iter_lsb_to_msb().copied());
        if is_register {
            pending.push(PendingRegister {
                name: format!("r{}", index - input_count),
                q: TransitionInputId::new(index),
                width,
            });
        }
    }
    if options.allow_constants {
        pool.extend([builder.get_false(), builder.get_true()]);
    } else if pool.is_empty() {
        pool.push(builder.get_false());
    }
    let mut generated = Vec::new();
    let mut stats = GeneratedGateStats::default();
    let mut operation_roots = Vec::new();
    for _ in 0..options.max_operations {
        if source.is_depleted() {
            break;
        }
        let available = options
            .operations
            .iter()
            .copied()
            .filter(|op| op.node_cost() <= options.max_aig_nodes - builder.gates.len())
            .collect::<Vec<_>>();
        if available.is_empty() {
            break;
        }
        let incomplete = source.incomplete_reads();
        let op = available[choose(source, 0..=available.len() - 1)];
        let a = pick_operand(source, &pool);
        let b = pick_operand(source, &pool);
        let c = if op == RandomGateOperation::Mux {
            pick_operand(source, &pool)
        } else {
            a
        };
        // A truncated body instruction is not partially committed to the graph.
        if source.incomplete_reads() != incomplete {
            break;
        }
        let first_new_node = builder.gates.len();
        let result = emit_operation(&mut builder, op, a, b, c);
        *stats.emitted_operations.entry(op).or_default() += 1;
        if result.node.id >= first_new_node {
            operation_roots.push((op, result.node.id));
        }
        pool.push(result);
        generated.push(result);
    }
    add_shapes(
        &mut builder,
        &mut pool,
        &mut generated,
        &mut RngEntropy::new(StdRng::seed_from_u64(shape_seed)),
        options,
    );
    let mut wiring = RngEntropy::new(StdRng::seed_from_u64(wiring_seed));
    let mut endpoints = BTreeMap::new();
    let mut outputs = Vec::new();
    let mut observed = false;
    for index in 0..output_count {
        let width = choose(&mut wiring, options.output_width.clone());
        let mut bits = (0..width)
            .map(|_| pick_operand(&mut wiring, &pool))
            .collect::<Vec<_>>();
        if !observed && !bits.is_empty() && options.observe_generated_logic && !generated.is_empty()
        {
            bits[0] = pick_operand(&mut wiring, &generated);
            observed = true;
        }
        outputs.push(add_endpoint(
            &mut builder,
            &mut endpoints,
            format!("out{index}"),
            AigBitVector::from_lsb_is_index_0(&bits),
            false,
        ));
    }
    let mut registers = Vec::new();
    for register in pending {
        // Sample both whole-bus reuse and independently wired bits. Pure
        // per-bit draws almost never alias wider ports, even when reuse is
        // enabled, leaving a substantial valid interface shape unexplored.
        let reusable = endpoints
            .iter()
            .filter(|(bits, _)| bits.len() == register.width)
            .map(|(_, id)| *id)
            .collect::<Vec<_>>();
        let d =
            if options.reuse_output_endpoints && !reusable.is_empty() && wiring.take_u64() & 1 != 0
            {
                reusable[choose(&mut wiring, 0..=reusable.len() - 1)]
            } else {
                let data = (0..register.width)
                    .map(|_| pick_operand(&mut wiring, &pool))
                    .collect::<Vec<_>>();
                add_endpoint(
                    &mut builder,
                    &mut endpoints,
                    format!("{}__d", register.name),
                    AigBitVector::from_lsb_is_index_0(&data),
                    options.reuse_output_endpoints,
                )
            };
        let enable = add_control(
            &mut builder,
            &mut endpoints,
            &mut wiring,
            &pool,
            format!("{}__enable", register.name),
            options.enable,
            options.reuse_output_endpoints,
        );
        let reset = add_control(
            &mut builder,
            &mut endpoints,
            &mut wiring,
            &pool,
            format!("{}__reset", register.name),
            options.reset,
            options.reuse_output_endpoints,
        )
        .map(|control| SynchronousReset {
            control,
            value: generate_uniform_irbits(&mut wiring, register.width),
        });
        registers.push(RegisterBinding {
            name: register.name,
            q: register.q,
            d,
            enable,
            reset,
        });
    }
    // Empty output interfaces are valid when explicitly requested by the
    // options.
    let transition = GateFn {
        name: builder.name,
        gates: builder.gates,
        inputs: builder.inputs,
        outputs: builder.outputs,
    };
    let clock = (register_count != 0).then(|| ClockPort { name: "clk".into() });
    let design = SequentialGateFn::new(
        "random_g8r".into(),
        transition,
        (0..input_count).map(TransitionInputId::new).collect(),
        outputs,
        clock,
        registers,
    )
    .map_err(GateGenerationError::Construction)?;
    let mut live = vec![false; design.transition.gates.len()];
    for operand in design.transition.post_order_operands(false) {
        live[operand.node.id] = true;
    }
    for (operation, root) in operation_roots {
        if live[root] {
            *stats.live_operations.entry(operation).or_default() += 1;
        }
    }
    Ok(GeneratedGateDesign {
        stats,
        design,
        builder_options,
    })
}

/// Bounds interface counts while reserving enough nodes for every required bit.
fn affordable_count(cap: usize, budget: usize, minimum_width: usize) -> usize {
    if minimum_width == 0 {
        cap
    } else {
        cap.min(budget / minimum_width)
    }
}

/// Chooses a bounded value; option validation guarantees a nonempty finite
/// range.
fn choose<S: EntropySource>(source: &mut S, range: RangeInclusive<usize>) -> usize {
    *range.start() + (source.take_u64() % (*range.end() - *range.start() + 1) as u64) as usize
}

/// Selects an existing node and an independent edge polarity.
fn pick_operand<S: EntropySource>(source: &mut S, pool: &[AigOperand]) -> AigOperand {
    let word = source.take_u64();
    let operand = pool[((word >> 1) % pool.len() as u64) as usize];
    if word & 1 != 0 {
        operand.negate()
    } else {
        operand
    }
}

/// Emits a Boolean operation through the same builder policy for every profile.
fn emit_operation(
    builder: &mut GateBuilder,
    op: RandomGateOperation,
    a: AigOperand,
    b: AigOperand,
    c: AigOperand,
) -> AigOperand {
    match op {
        RandomGateOperation::And => builder.add_and_binary(a, b),
        RandomGateOperation::Or => builder.add_or_binary(a, b),
        RandomGateOperation::Xor => builder.add_xor_binary(a, b),
        RandomGateOperation::Xnor => builder.add_xnor(a, b),
        RandomGateOperation::Mux => builder.add_mux2(a, b, c),
    }
}

/// Applies only the explicitly enabled structural stress distributions.
fn add_shapes<S: EntropySource>(
    builder: &mut GateBuilder,
    pool: &mut Vec<AigOperand>,
    generated: &mut Vec<AigOperand>,
    source: &mut S,
    options: &RandomGateOptions,
) {
    let mut chain = pick_operand(source, pool);
    for _ in 0..choose(source, 0..=options.shapes.max_chain_length) {
        if builder.gates.len() == options.max_aig_nodes {
            break;
        }
        let rhs = pick_operand(source, pool);
        chain = builder.add_and_binary(chain, rhs);
        pool.push(chain);
        generated.push(chain);
    }
    for _ in 0..choose(source, 0..=options.shapes.max_redundant_pairs) {
        if options.max_aig_nodes - builder.gates.len() < 2 {
            break;
        }
        let a = pick_operand(source, pool);
        let b = pick_operand(source, pool);
        for _ in 0..2 {
            let value = builder.add_and_binary(a, b);
            pool.push(value);
            generated.push(value);
        }
    }
    let root = pick_operand(source, pool);
    for _ in 0..choose(source, 0..=options.shapes.max_fanout_branches) {
        if builder.gates.len() == options.max_aig_nodes {
            break;
        }
        let rhs = pick_operand(source, pool);
        let value = builder.add_and_binary(root, rhs);
        pool.push(value);
        generated.push(value);
    }
}

/// Keys endpoints by their packed Boolean operands, preserving external
/// identities.
type EndpointMap = BTreeMap<Vec<(usize, bool)>, TransitionOutputId>;

/// Adds an endpoint or reuses an existing role-compatible packed output.
fn add_endpoint(
    builder: &mut GateBuilder,
    endpoints: &mut EndpointMap,
    name: String,
    bits: AigBitVector,
    reuse: bool,
) -> TransitionOutputId {
    let key = bits
        .iter_lsb_to_msb()
        .map(|bit| (bit.node.id, bit.negated))
        .collect::<Vec<_>>();
    if reuse {
        if let Some(id) = endpoints.get(&key) {
            return *id;
        }
    }
    let id = TransitionOutputId::new(builder.outputs.len());
    builder.add_output(name, bits);
    endpoints.entry(key).or_insert(id);
    id
}

/// Samples an optional control without adding any combinational feedback edge.
fn add_control<S: EntropySource>(
    builder: &mut GateBuilder,
    endpoints: &mut EndpointMap,
    source: &mut S,
    pool: &[AigOperand],
    name: String,
    polarity: Option<RandomControlPolarity>,
    reuse: bool,
) -> Option<RegisterControl> {
    let polarity = polarity?;
    if source.take_u64() & 1 == 0 {
        return None;
    }
    let signal = add_endpoint(
        builder,
        endpoints,
        name,
        pick_operand(source, pool).into(),
        reuse,
    );
    let active_low = match polarity {
        RandomControlPolarity::ActiveHigh => false,
        RandomControlPolarity::ActiveLow => true,
        RandomControlPolarity::Either => source.take_u64() & 1 != 0,
    };
    Some(RegisterControl { signal, active_low })
}
