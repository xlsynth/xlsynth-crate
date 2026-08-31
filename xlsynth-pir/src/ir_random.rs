// SPDX-License-Identifier: Apache-2.0

//! Direct construction of random, typed PIR functions.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use rand::RngCore;
use xlsynth::{IrBits, IrValue};

use crate::ir::{
    Binop, BlockMetadata, BlockResetMetadata, ExtNaryAddArchitecture, ExtNaryAddTerm, FileTable,
    Fn, MemberType, NaryOp, Node, NodePayload, NodeRef, Package, PackageMember, Param, ParamId,
    Register, Type, Unop,
};
use crate::ir_rebase_ids::rebase_fn_ids;
use crate::ir_utils::{is_observable_effect_root, operands};
use crate::math::ceil_log2;
use crate::random_inputs::generate_uniform_value;

const TRACE_FORMAT_SPECIFIERS: [&str; 9] = [
    "{}", "{:u}", "{:d}", "{:x}", "{:0x}", "{:#x}", "{:b}", "{:0b}", "{:#b}",
];
const MAX_RANDOM_SEL_CASES: usize = 16;
const MAX_RANDOM_PRIORITY_SEL_CASES: usize = 256;

/// Operations that the random generator can introduce into a function body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RandomOperation {
    Literal,
    Identity,
    Not,
    Neg,
    Reverse,
    OrReduce,
    AndReduce,
    XorReduce,
    And,
    Nand,
    Nor,
    Or,
    Xor,
    Add,
    Sub,
    Umul,
    Smul,
    Udiv,
    Sdiv,
    Umod,
    Smod,
    Umulp,
    Smulp,
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,
    Shll,
    Shrl,
    Shra,
    Gate,
    ZeroExt,
    SignExt,
    BitSlice,
    Concat,
    Array,
    ArrayIndex,
    ArrayConcat,
    ArraySlice,
    ArrayUpdate,
    Tuple,
    TupleIndex,
    DynamicBitSlice,
    BitSliceUpdate,
    Sel,
    PrioritySel,
    OneHotSel,
    OneHot,
    Encode,
    Decode,
    ExtCarryOut,
    ExtPrioEncode,
    ExtClz,
    ExtNormalizeLeft,
    ExtMaskLow,
    ExtNaryAdd,
    AfterAll,
    Cover,
    Assert,
    Trace,
    Invoke,
    CountedFor,
}

impl RandomOperation {
    fn all_supported() -> &'static [Self] {
        &[
            Self::Literal,
            Self::Identity,
            Self::Not,
            Self::Neg,
            Self::Reverse,
            Self::OrReduce,
            Self::AndReduce,
            Self::XorReduce,
            Self::And,
            Self::Nand,
            Self::Nor,
            Self::Or,
            Self::Xor,
            Self::Add,
            Self::Sub,
            Self::Umul,
            Self::Smul,
            Self::Udiv,
            Self::Sdiv,
            Self::Umod,
            Self::Smod,
            Self::Umulp,
            Self::Smulp,
            Self::Eq,
            Self::Ne,
            Self::Ugt,
            Self::Uge,
            Self::Ult,
            Self::Ule,
            Self::Sgt,
            Self::Sge,
            Self::Slt,
            Self::Sle,
            Self::Shll,
            Self::Shrl,
            Self::Shra,
            Self::Gate,
            Self::ZeroExt,
            Self::SignExt,
            Self::BitSlice,
            Self::Concat,
            Self::Array,
            Self::ArrayIndex,
            Self::ArrayConcat,
            Self::ArraySlice,
            Self::ArrayUpdate,
            Self::Tuple,
            Self::TupleIndex,
            Self::DynamicBitSlice,
            Self::BitSliceUpdate,
            Self::Sel,
            Self::PrioritySel,
            Self::OneHotSel,
            Self::OneHot,
            Self::Encode,
            Self::Decode,
            Self::ExtCarryOut,
            Self::ExtPrioEncode,
            Self::ExtClz,
            Self::ExtNormalizeLeft,
            Self::ExtMaskLow,
            Self::ExtNaryAdd,
            Self::AfterAll,
            Self::Cover,
            Self::Assert,
            Self::Trace,
            Self::Invoke,
            Self::CountedFor,
        ]
    }

    /// Returns the PIR operator spelling for this operation.
    pub fn name(self) -> &'static str {
        match self {
            Self::Literal => "literal",
            Self::Identity => "identity",
            Self::Not => "not",
            Self::Neg => "neg",
            Self::Reverse => "reverse",
            Self::OrReduce => "or_reduce",
            Self::AndReduce => "and_reduce",
            Self::XorReduce => "xor_reduce",
            Self::And => "and",
            Self::Nand => "nand",
            Self::Nor => "nor",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Umul => "umul",
            Self::Smul => "smul",
            Self::Udiv => "udiv",
            Self::Sdiv => "sdiv",
            Self::Umod => "umod",
            Self::Smod => "smod",
            Self::Umulp => "umulp",
            Self::Smulp => "smulp",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Ugt => "ugt",
            Self::Uge => "uge",
            Self::Ult => "ult",
            Self::Ule => "ule",
            Self::Sgt => "sgt",
            Self::Sge => "sge",
            Self::Slt => "slt",
            Self::Sle => "sle",
            Self::Shll => "shll",
            Self::Shrl => "shrl",
            Self::Shra => "shra",
            Self::Gate => "gate",
            Self::ZeroExt => "zero_ext",
            Self::SignExt => "sign_ext",
            Self::BitSlice => "bit_slice",
            Self::Concat => "concat",
            Self::Array => "array",
            Self::ArrayIndex => "array_index",
            Self::ArrayConcat => "array_concat",
            Self::ArraySlice => "array_slice",
            Self::ArrayUpdate => "array_update",
            Self::Tuple => "tuple",
            Self::TupleIndex => "tuple_index",
            Self::DynamicBitSlice => "dynamic_bit_slice",
            Self::BitSliceUpdate => "bit_slice_update",
            Self::Sel => "sel",
            Self::PrioritySel => "priority_sel",
            Self::OneHotSel => "one_hot_sel",
            Self::OneHot => "one_hot",
            Self::Encode => "encode",
            Self::Decode => "decode",
            Self::ExtCarryOut => "ext_carry_out",
            Self::ExtPrioEncode => "ext_prio_encode",
            Self::ExtClz => "ext_clz",
            Self::ExtNormalizeLeft => "ext_normalize_left",
            Self::ExtMaskLow => "ext_mask_low",
            Self::ExtNaryAdd => "ext_nary_add",
            Self::AfterAll => "after_all",
            Self::Cover => "cover",
            Self::Assert => "assert",
            Self::Trace => "trace",
            Self::Invoke => "invoke",
            Self::CountedFor => "counted_for",
        }
    }
}

/// Set of operations permitted when generating a function body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationSet {
    enabled: BTreeSet<RandomOperation>,
}

impl OperationSet {
    /// Creates a set from the provided operations.
    pub fn new(operations: impl IntoIterator<Item = RandomOperation>) -> Self {
        Self {
            enabled: operations.into_iter().collect(),
        }
    }

    /// Creates a set containing every operation implemented by this generator.
    pub fn all_supported() -> Self {
        Self::new(RandomOperation::all_supported().iter().copied())
    }

    /// Returns whether an operation is permitted.
    pub fn contains(&self, operation: RandomOperation) -> bool {
        self.enabled.contains(&operation)
    }

    /// Iterates over permitted operations in stable order.
    pub fn iter(&self) -> impl Iterator<Item = RandomOperation> + '_ {
        self.enabled.iter().copied()
    }
}

impl Default for OperationSet {
    fn default() -> Self {
        Self::all_supported()
    }
}

/// Controls generation of array operations with `assumed_in_bounds=true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayAssumptionMode {
    /// Never emits an asserted in-bounds assumption.
    Disabled,
    /// May assert in-bounds for any generated index.
    Unrestricted,
    /// May assert in-bounds only for indices proven safe by width or literal.
    ProvenSafe,
}

/// Configures shape and type limits for random PIR functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomFnOptions {
    pub max_params: usize,
    pub max_nodes: usize,
    pub max_bit_width: usize,
    /// Optional operand/result width cap for signed and unsigned div/mod.
    /// This cap does not affect other operations.
    pub max_div_mod_bit_width: Option<usize>,
    /// Optional operand width cap for signed/unsigned multiplies and partial
    /// products. Result widths continue to use `max_bit_width`.
    pub max_multiply_operand_bit_width: Option<usize>,
    pub max_type_depth: usize,
    pub max_aggregate_leaves: usize,
    pub max_array_length: usize,
    pub max_tuple_length: usize,
    /// Maximum number of operands generated for a standard n-ary operation.
    ///
    /// XLS requires at least one operand for bitwise n-ary operations. A
    /// zero-operand `concat` is generated only when `allow_zero_width_bits`
    /// is true.
    pub max_nary_operands: usize,
    pub allow_arrays: bool,
    pub allow_tuples: bool,
    /// Permits operations whose result can be `bits[0]`, such as an empty
    /// concat or a zero-width slice. This is off by default because not every
    /// downstream consumer supports zero-width values yet.
    pub allow_zero_width_bits: bool,
    /// Permits multiply operands/results, including `mulp` result fields, to
    /// have independently selected widths, as XLS allows.
    pub allow_arbitrary_width_multiply: bool,
    /// Permits `sel(selector, cases=[], default=value)`.
    ///
    /// XLS evaluates this node form, but its text IR parser currently rejects
    /// it, so generators whose output is parsed by libxls should leave this
    /// disabled.
    pub allow_empty_case_sel: bool,
    /// Permits the XLS `gate` operation when it is included in
    /// `enabled_operations`.
    pub allow_gate: bool,
    /// Permits non-upstream `ext_*` operations when they are included in
    /// `enabled_operations`.
    pub allow_extension_ops: bool,
    /// Permits token values and the effect operations `after_all`, `cover`,
    /// `assert`, and `trace` when they are included in `enabled_operations`.
    ///
    /// `cover` additionally requires `allow_tuples` because its result type is
    /// the empty tuple.
    pub allow_events: bool,
    /// Controls whether generated array operations may assert
    /// `assumed_in_bounds=true`, and whether such assertions require indices
    /// proven safe by width or literal value.
    pub array_assumption_mode: ArrayAssumptionMode,
    /// Maximum total number of generated functions, including the top, when
    /// constructing a package.
    pub max_functions: usize,
    /// Maximum number of invoke nodes emitted in any one generated function.
    pub max_invokes_per_function: usize,
    /// Maximum number of `counted_for` nodes emitted in any one generated
    /// function.
    pub max_counted_fors_per_function: usize,
    /// Maximum static trip count generated for a `counted_for` node.
    pub max_counted_for_trip_count: usize,
    /// Maximum non-negative static stride generated for a `counted_for` node.
    pub max_counted_for_stride: usize,
    /// Maximum product of `counted_for` trip counts along any nested call path.
    ///
    /// This bounds execution expansion while still permitting multiple
    /// sequential loops in one function.
    pub max_nested_counted_for_iterations: usize,
    pub enabled_operations: OperationSet,
}

impl Default for RandomFnOptions {
    fn default() -> Self {
        Self {
            max_params: 5,
            max_nodes: 32,
            max_bit_width: 64,
            max_div_mod_bit_width: None,
            max_multiply_operand_bit_width: None,
            max_type_depth: 2,
            max_aggregate_leaves: 32,
            max_array_length: 4,
            max_tuple_length: 4,
            max_nary_operands: 4,
            allow_arrays: true,
            allow_tuples: true,
            allow_zero_width_bits: false,
            allow_arbitrary_width_multiply: false,
            allow_empty_case_sel: false,
            allow_gate: false,
            allow_extension_ops: false,
            allow_events: false,
            array_assumption_mode: ArrayAssumptionMode::Disabled,
            max_functions: 5,
            max_invokes_per_function: 3,
            max_counted_fors_per_function: 3,
            max_counted_for_trip_count: 8,
            max_counted_for_stride: 8,
            max_nested_counted_for_iterations: 64,
            enabled_operations: OperationSet::default(),
        }
    }
}

impl RandomFnOptions {
    fn validate(&self) -> Result<(), GenerationError> {
        if self.max_nodes == 0 {
            return Err(GenerationError::InvalidOptions(
                "max_nodes must permit at least one return-producing node".to_string(),
            ));
        }
        if self.max_bit_width == 0 {
            return Err(GenerationError::InvalidOptions(
                "max_bit_width must be nonzero".to_string(),
            ));
        }
        if self.max_div_mod_bit_width == Some(0) {
            return Err(GenerationError::InvalidOptions(
                "max_div_mod_bit_width must be nonzero when specified".to_string(),
            ));
        }
        if self.max_multiply_operand_bit_width == Some(0) {
            return Err(GenerationError::InvalidOptions(
                "max_multiply_operand_bit_width must be nonzero when specified".to_string(),
            ));
        }
        if self.max_aggregate_leaves == 0 {
            return Err(GenerationError::InvalidOptions(
                "max_aggregate_leaves must be nonzero".to_string(),
            ));
        }
        if self.max_nary_operands == 0 {
            return Err(GenerationError::InvalidOptions(
                "max_nary_operands must be nonzero".to_string(),
            ));
        }
        if self.allow_arrays && self.max_array_length == 0 {
            return Err(GenerationError::InvalidOptions(
                "max_array_length must be nonzero when arrays are allowed".to_string(),
            ));
        }
        if !self.enabled_operations.contains(RandomOperation::Literal) {
            return Err(GenerationError::InvalidOptions(
                "literal must be enabled so zero-parameter functions can be constructed"
                    .to_string(),
            ));
        }
        if self.max_functions == 0 {
            return Err(GenerationError::InvalidOptions(
                "max_functions must be nonzero".to_string(),
            ));
        }
        if self.max_counted_for_trip_count > i64::MAX as usize {
            return Err(GenerationError::InvalidOptions(
                "max_counted_for_trip_count must fit in an XLS int64 attribute".to_string(),
            ));
        }
        if self.max_counted_for_stride > i64::MAX as usize {
            return Err(GenerationError::InvalidOptions(
                "max_counted_for_stride must fit in an XLS int64 attribute".to_string(),
            ));
        }
        if self.max_nested_counted_for_iterations == 0 {
            return Err(GenerationError::InvalidOptions(
                "max_nested_counted_for_iterations must be nonzero".to_string(),
            ));
        }
        Ok(())
    }
}

/// Controls when node-by-node function body construction stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopPolicy {
    /// Continue until the bounded byte stream has supplied all available data.
    WhenEntropyDepleted,
    /// Emit this many body nodes, subject to `max_nodes`. Unconstrained
    /// generation always emits at least one body node to provide its result.
    ExactBodyNodes(usize),
    /// After the minimum, stop probabilistically until reaching the maximum.
    Geometric {
        min_body_nodes: usize,
        max_body_nodes: usize,
        stop_numerator: u64,
        stop_denominator: u64,
    },
}

/// Provides choices to the random generator.
pub trait EntropySource {
    /// Reports whether a finite source has no remaining entropy.
    fn is_depleted(&self) -> bool;

    /// Returns the next random word, using zero when a finite source is spent.
    fn take_u64(&mut self) -> u64;

    /// Counts incomplete reads so a body operation can be discarded atomically.
    fn incomplete_reads(&self) -> usize {
        0
    }
}

/// Finite little-endian entropy suitable for coverage-guided fuzzing inputs.
#[derive(Debug, Clone)]
pub struct DepletableBytes<'a> {
    data: &'a [u8],
    offset: usize,
    incomplete_reads: usize,
}

impl<'a> DepletableBytes<'a> {
    /// Creates a finite entropy stream backed by fuzzer input bytes.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            offset: 0,
            incomplete_reads: 0,
        }
    }

    /// Splits fuzzer input into independent streams for paired generation.
    pub fn split(data: &'a [u8]) -> (Self, Self) {
        let midpoint = data.len() / 2 + data.len() % 2;
        let (first, second) = data.split_at(midpoint);
        (Self::new(first), Self::new(second))
    }
}

impl EntropySource for DepletableBytes<'_> {
    fn is_depleted(&self) -> bool {
        self.offset >= self.data.len()
    }

    fn take_u64(&mut self) -> u64 {
        let mut bytes = [0_u8; 8];
        let remaining = self.data.len().saturating_sub(self.offset);
        let count = remaining.min(bytes.len());
        self.incomplete_reads += usize::from(count < bytes.len());
        bytes[..count].copy_from_slice(&self.data[self.offset..self.offset + count]);
        self.offset += count;
        u64::from_le_bytes(bytes)
    }

    fn incomplete_reads(&self) -> usize {
        self.incomplete_reads
    }
}

/// Non-depleting entropy backed by a `rand` random-number generator.
#[derive(Debug, Clone)]
pub struct RngEntropy<R> {
    rng: R,
}

impl<R> RngEntropy<R> {
    /// Creates entropy backed by the supplied RNG.
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: RngCore> EntropySource for RngEntropy<R> {
    fn is_depleted(&self) -> bool {
        false
    }

    fn take_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }
}

/// Errors caused by invalid generator configuration or an internal construction
/// failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationError {
    InvalidOptions(String),
    InvalidSignature(String),
    Construction(String),
}

impl Display for GenerationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions(message) => {
                write!(formatter, "invalid random PIR options: {message}")
            }
            Self::InvalidSignature(message) => {
                write!(
                    formatter,
                    "invalid random PIR function signature: {message}"
                )
            }
            Self::Construction(message) => {
                write!(formatter, "random PIR construction failed: {message}")
            }
        }
    }
}

impl Error for GenerationError {}

/// Coverage-relevant measurements for one generated function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedFnStats {
    pub emitted_node_count: usize,
    pub live_node_count: usize,
    pub emitted_operations: BTreeMap<String, usize>,
    pub live_operations: BTreeMap<String, usize>,
    pub emitted_bits_widths: BTreeSet<usize>,
    pub live_bits_widths: BTreeSet<usize>,
}

/// A directly constructed PIR function and its generation statistics.
#[derive(Debug, Clone)]
pub struct GeneratedFn {
    pub function: Fn,
    pub stats: GeneratedFnStats,
}

impl GeneratedFn {
    /// Moves the generated function into a package with that function marked
    /// top.
    pub fn into_top_package(self, package_name: impl Into<String>) -> Package {
        let function_name = self.function.name.clone();
        Package {
            name: package_name.into(),
            file_table: FileTable::new(),
            members: vec![PackageMember::Function(self.function)],
            top: Some((function_name, MemberType::Function)),
        }
    }
}

/// A directly constructed acyclic PIR package and per-function statistics.
#[derive(Debug, Clone)]
pub struct GeneratedPackage {
    pub package: Package,
    pub function_stats: Vec<GeneratedFnStats>,
}

/// Controls reset timing metadata for generated random PIR blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomBlockResetTiming {
    /// Generates only synchronous block reset metadata.
    Synchronous,
    /// Generates only asynchronous block reset metadata.
    Asynchronous,
    /// Chooses synchronous or asynchronous block reset metadata with equal
    /// probability.
    Either,
}

/// Selects the register topology of a generated block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockTopology {
    /// Generates no registers.
    Combinational,
    /// Generates synchronous registers with unrestricted type-compatible
    /// next-state, enable, output, and feedback wiring.
    GeneralSequential,
    /// Restricts each stage to its preceding register, producing a layered
    /// feed-forward pipeline with one register per boundary.
    FeedForwardPipeline,
}

/// Configures shape and state limits for random PIR blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomBlockOptions {
    /// Options used for random data types and combinational body operations.
    pub function_options: RandomFnOptions,
    /// Minimum number of data input ports, excluding any generated reset port.
    pub min_input_ports: usize,
    /// Maximum number of data input ports, excluding any generated reset port.
    pub max_input_ports: usize,
    /// Minimum number of output ports; zero-output blocks are supported.
    pub min_output_ports: usize,
    pub max_output_ports: usize,
    /// Selects whether the block is combinational, generally sequential, or a
    /// layered feed-forward pipeline.
    pub topology: BlockTopology,
    /// Maximum generated register count for sequential topologies.
    pub max_registers: usize,
    /// Permits port and register types whose flattened bit count is zero,
    /// including zero-width aggregates such as arrays of empty tuples.
    pub allow_zero_width_ports_and_registers: bool,
    /// Permits generated register writes to use an available `bits[1]` value as
    /// a load-enable.
    pub allow_load_enable: bool,
    /// Permits generated blocks with registers to add a reset port and use it
    /// on some register writes.
    pub allow_reset: bool,
    /// Requires every generated register to use a generated reset port and
    /// carry a reset value. Blocks without registers remain reset-free.
    pub require_reset_on_all_registers: bool,
    /// Controls the timing kind when a reset port is generated.
    pub reset_timing: RandomBlockResetTiming,
}

impl Default for RandomBlockOptions {
    fn default() -> Self {
        Self {
            function_options: RandomFnOptions::default(),
            min_input_ports: 0,
            max_input_ports: 5,
            min_output_ports: 0,
            max_output_ports: 3,
            topology: BlockTopology::GeneralSequential,
            max_registers: 3,
            allow_zero_width_ports_and_registers: false,
            allow_load_enable: true,
            allow_reset: true,
            require_reset_on_all_registers: false,
            reset_timing: RandomBlockResetTiming::Either,
        }
    }
}

impl RandomBlockOptions {
    fn minimum_registers(&self) -> usize {
        match self.topology {
            BlockTopology::Combinational => 0,
            BlockTopology::GeneralSequential => 1,
            BlockTopology::FeedForwardPipeline => 2,
        }
    }

    /// Bytes reserved before body entropy: one u64 per output, then next-state
    /// and load-enable u64s for each register, including unused maximum slots.
    pub fn wiring_header_byte_count(&self) -> Result<usize, GenerationError> {
        self.max_registers
            .checked_mul(2)
            .and_then(|count| count.checked_add(self.max_output_ports))
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| {
                GenerationError::InvalidOptions("block wiring header is too large".into())
            })
    }

    fn validate(&self) -> Result<(), GenerationError> {
        self.function_options.validate()?;
        self.wiring_header_byte_count()?;
        let minimum_registers = self.minimum_registers();
        if self.topology == BlockTopology::FeedForwardPipeline && self.min_input_ports == 0 {
            return Err(GenerationError::InvalidOptions(
                "feed-forward pipeline topology requires min_input_ports >= 1".into(),
            ));
        }
        if self.min_input_ports > self.max_input_ports {
            return Err(GenerationError::InvalidOptions(format!(
                "min_input_ports {} exceeds max_input_ports {}",
                self.min_input_ports, self.max_input_ports
            )));
        }
        if self.min_output_ports > self.max_output_ports {
            return Err(GenerationError::InvalidOptions(format!(
                "min_output_ports {} exceeds max_output_ports {}",
                self.min_output_ports, self.max_output_ports
            )));
        }
        if minimum_registers > self.max_registers {
            return Err(GenerationError::InvalidOptions(format!(
                "topology requires at least {minimum_registers} registers but max_registers is {}",
                self.max_registers
            )));
        }
        if self.require_reset_on_all_registers && !self.allow_reset && self.max_registers > 0 {
            return Err(GenerationError::InvalidOptions(
                "require_reset_on_all_registers requires allow_reset".to_string(),
            ));
        }
        if self.min_input_ports > self.function_options.max_params {
            return Err(GenerationError::InvalidOptions(format!(
                "min_input_ports {} exceeds function_options.max_params {}",
                self.min_input_ports, self.function_options.max_params
            )));
        }
        if self.require_reset_on_all_registers
            && minimum_registers > 0
            && self.min_input_ports.saturating_add(1) > self.function_options.max_params
        {
            return Err(GenerationError::InvalidOptions(format!(
                "minimum block shape needs {} parameters including reset but function_options.max_params is {}",
                self.min_input_ports.saturating_add(1),
                self.function_options.max_params
            )));
        }

        let min_output_tuple_node = usize::from(self.min_output_ports != 1);
        let min_seed_node = usize::from(
            self.min_output_ports > 0 && self.min_input_ports == 0 && minimum_registers == 0,
        );
        let min_reset_port_node =
            usize::from(self.require_reset_on_all_registers && minimum_registers > 0);
        let required_nodes = self
            .min_input_ports
            .saturating_add(minimum_registers.saturating_mul(2))
            .saturating_add(min_reset_port_node)
            .saturating_add(min_output_tuple_node)
            .saturating_add(min_seed_node);
        if required_nodes > self.function_options.max_nodes {
            return Err(GenerationError::InvalidOptions(format!(
                "minimum block shape requires {required_nodes} nodes but max_nodes is {}",
                self.function_options.max_nodes
            )));
        }
        Ok(())
    }
}

/// A directly constructed PIR block and its generation statistics.
#[derive(Debug, Clone)]
pub struct GeneratedBlock {
    pub function: Fn,
    pub metadata: BlockMetadata,
    pub stats: GeneratedFnStats,
}

impl GeneratedBlock {
    /// Moves the generated block into a package with that block marked top.
    pub fn into_top_package(self, package_name: impl Into<String>) -> Package {
        let block_name = self.function.name.clone();
        Package {
            name: package_name.into(),
            file_table: FileTable::new(),
            members: vec![PackageMember::Block {
                func: self.function,
                metadata: self.metadata,
            }],
            top: Some((block_name, MemberType::Block)),
        }
    }
}

/// A directly constructed PIR block package and per-block statistics.
#[derive(Debug, Clone)]
pub struct GeneratedBlockPackage {
    pub package: Package,
    pub block_stats: Vec<GeneratedFnStats>,
}

/// Parameter and return types required for constrained random function
/// generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSignature {
    pub params: Vec<Type>,
    pub return_type: Type,
}

impl FunctionSignature {
    /// Captures the type signature of an existing PIR function.
    pub fn from_fn(function: &Fn) -> Self {
        Self {
            params: function
                .params
                .iter()
                .map(|param| param.ty.clone())
                .collect(),
            return_type: function.ret_ty.clone(),
        }
    }
}

/// Generates a typed PIR function directly from an entropy source.
pub fn generate_fn<S: EntropySource>(
    source: &mut S,
    options: &RandomFnOptions,
    stop_policy: StopPolicy,
) -> Result<GeneratedFn, GenerationError> {
    options.validate()?;
    validate_stop_policy(stop_policy)?;

    let mut generator = FunctionGenerator::new(options);
    let max_params = options.max_params.min(options.max_nodes.saturating_sub(1));
    let param_count = choose_count(source, max_params + 1);
    for _ in 0..param_count {
        let ty = random_type(source, options, 0);
        generator.add_param(ty);
    }

    let body_capacity = options.max_nodes - generator.params.len();
    let mut body_nodes = 0;
    while body_nodes < body_capacity
        && (body_nodes == 0 || should_add_node(source, stop_policy, body_nodes))
    {
        let added = generator.add_random_body_node(
            source,
            operation_budget(stop_policy, body_capacity - body_nodes, body_nodes),
        )?;
        if added == 0 {
            break;
        }
        body_nodes += added;
    }

    if generator.nodes.len() == 1 {
        generator.emit_operation(source, RandomOperation::Literal)?;
    }

    let function = generator.finish()?;
    let stats = gather_stats(&function);
    Ok(GeneratedFn { function, stats })
}

/// Generates an acyclic package with functions created on demand by calls.
///
/// Each function body is generated node by node. An ordinary node request
/// occasionally emits an `invoke` or `counted_for` to a completed function or
/// recursively creates a new callee with an independently generated signature.
/// Functions under construction are never eligible callees, so the resulting
/// call graph is acyclic by construction.
pub fn generate_package<S: EntropySource>(
    source: &mut S,
    options: &RandomFnOptions,
    stop_policy: StopPolicy,
) -> Result<GeneratedPackage, GenerationError> {
    options.validate()?;
    validate_stop_policy(stop_policy)?;
    PackageGenerator::new(source, options, stop_policy).generate()
}

/// Generates a typed PIR block directly from an entropy source.
///
/// Reserves all output/next-state/load-enable words before generating any body
/// nodes. With `DepletableBytes`, this is a fixed, zero-padded wiring header of
/// `options.wiring_header_byte_count()` bytes, independent of the body stream.
pub fn generate_block<S: EntropySource>(
    source: &mut S,
    options: &RandomBlockOptions,
    stop_policy: StopPolicy,
) -> Result<GeneratedBlock, GenerationError> {
    options.validate()?;
    validate_stop_policy(stop_policy)?;
    BlockGenerator::new(source, options, stop_policy).generate_block("random_block_0".to_string())
}

/// Generates a package containing one top block directly from an entropy
/// source.
pub fn generate_block_package<S: EntropySource>(
    source: &mut S,
    options: &RandomBlockOptions,
    stop_policy: StopPolicy,
) -> Result<GeneratedBlockPackage, GenerationError> {
    let generated = generate_block(source, options, stop_policy)?;
    let stats = generated.stats.clone();
    let package = generated.into_top_package("random_block_package");
    crate::ir_verify::verify_package(&package)
        .map_err(|error| GenerationError::Construction(error.to_string()))?;
    Ok(GeneratedBlockPackage {
        package,
        block_stats: vec![stats],
    })
}

#[derive(Debug, Clone)]
struct GeneratedRegisterState {
    name: String,
    ty: Type,
    reset_ref: Option<NodeRef>,
}

struct GeneratedRegisterWrite {
    next_ref: NodeRef,
    load_enable: Option<NodeRef>,
}

struct BlockGenerator<'a, S> {
    source: &'a mut S,
    options: &'a RandomBlockOptions,
    stop_policy: StopPolicy,
    output_wiring: Vec<u64>,
    register_wiring: Vec<(u64, u64)>,
}

impl<'a, S: EntropySource> BlockGenerator<'a, S> {
    fn new(source: &'a mut S, options: &'a RandomBlockOptions, stop_policy: StopPolicy) -> Self {
        let output_wiring = (0..options.max_output_ports)
            .map(|_| source.take_u64())
            .collect();
        let register_wiring = (0..options.max_registers)
            .map(|_| (source.take_u64(), source.take_u64()))
            .collect();
        Self {
            source,
            options,
            stop_policy,
            output_wiring,
            register_wiring,
        }
    }

    fn generate_block(&mut self, name: String) -> Result<GeneratedBlock, GenerationError> {
        let mut generator = FunctionGenerator::new(&self.options.function_options);
        let mut metadata = BlockMetadata {
            clock_port_name: None,
            port_order: Vec::new(),
            port_sv_types: std::collections::BTreeMap::new(),
            input_port_ids: HashMap::new(),
            output_port_ids: HashMap::new(),
            output_names: Vec::new(),
            reset: None,
            registers: Vec::new(),
            instantiations: Vec::new(),
        };

        let input_count = self.choose_input_count()?;
        for index in 0..input_count {
            let ty = self.choose_interface_type(&generator);
            self.add_block_param(&mut generator, &mut metadata, format!("in{index}"), ty);
        }

        let register_count = self.choose_register_count(generator.params.len())?;
        let output_tuple_reserve = usize::from(self.options.min_output_ports != 1);
        let reset_ref = self.maybe_add_reset_port(
            &mut generator,
            &mut metadata,
            register_count,
            output_tuple_reserve,
        )?;
        let registers =
            self.add_registers(&mut generator, &mut metadata, register_count, reset_ref);

        let staged_writes = if self.options.topology == BlockTopology::FeedForwardPipeline
            && register_count != 0
        {
            Some(self.generate_pipeline_body(&mut generator, &registers, output_tuple_reserve)?)
        } else {
            self.generate_body(&mut generator, register_count, output_tuple_reserve)?;
            None
        };

        let output_count = self.choose_output_count(&generator, register_count)?;
        let output_refs = self.choose_output_refs(&generator, output_count)?;
        let register_writes = if let Some(writes) = staged_writes {
            writes
        } else {
            let mut writes = Vec::with_capacity(registers.len());
            for (index, register) in registers.iter().enumerate() {
                let (next_choice, enable_choice) = self.register_wiring[index];
                let next_ref = self.choose_register_next_ref(&generator, register, next_choice)?;
                let load_enable = self.maybe_choose_load_enable_ref(&generator, enable_choice);
                writes.push(GeneratedRegisterWrite {
                    next_ref,
                    load_enable,
                });
            }
            writes
        };

        // For zero or multiple outputs, the synthetic tuple return is omitted
        // when the function is emitted as a block. Keep it out of the register
        // D-value candidate set so register writes cannot reference that
        // suppressed internal node.
        let ret_node_ref = self.materialize_block_return(&mut generator, &output_refs)?;

        for (register, write) in registers.iter().zip(register_writes) {
            generator.add_node(
                Type::nil(),
                NodePayload::RegisterWrite {
                    arg: write.next_ref,
                    register: register.name.clone(),
                    load_enable: write.load_enable,
                    reset: register.reset_ref,
                },
                Some(format!("{}_d", register.name)),
            );
        }

        self.populate_output_metadata(&generator, &mut metadata, output_count);
        let mut function = generator.finish_with_return(ret_node_ref)?;
        function.name = name;
        let stats = gather_block_stats(&function);

        Ok(GeneratedBlock {
            function,
            metadata,
            stats,
        })
    }

    fn choose_input_count(&mut self) -> Result<usize, GenerationError> {
        let minimum_registers = self.options.minimum_registers();
        let output_tuple_reserve = usize::from(self.options.min_output_ports != 1);
        let reset_port_reserve =
            usize::from(self.options.require_reset_on_all_registers && minimum_registers > 0);
        let reserved_nodes = self
            .options
            .minimum_registers()
            .saturating_mul(2)
            .saturating_add(reset_port_reserve)
            .saturating_add(output_tuple_reserve);
        let max_by_nodes = self
            .options
            .function_options
            .max_nodes
            .saturating_sub(reserved_nodes);
        let max_by_params = self
            .options
            .function_options
            .max_params
            .saturating_sub(reset_port_reserve);
        let max_input_count = self
            .options
            .max_input_ports
            .min(max_by_params)
            .min(max_by_nodes);
        if max_input_count < self.options.min_input_ports {
            return Err(GenerationError::InvalidOptions(format!(
                "minimum block input count {} cannot fit with max_nodes {}",
                self.options.min_input_ports, self.options.function_options.max_nodes
            )));
        }
        Ok(choose_between(
            self.source,
            self.options.min_input_ports,
            max_input_count,
        ))
    }

    fn choose_register_count(&mut self, input_count: usize) -> Result<usize, GenerationError> {
        let minimum_registers = self.options.minimum_registers();
        if self.options.topology == BlockTopology::Combinational {
            return Ok(0);
        }
        let output_tuple_reserve = usize::from(self.options.min_output_ports != 1);
        let reset_port_reserve = usize::from(self.options.require_reset_on_all_registers);
        let max_by_nodes = self.options.function_options.max_nodes.saturating_sub(
            input_count
                .saturating_add(output_tuple_reserve)
                .saturating_add(reset_port_reserve),
        ) / 2;
        let max_register_count = if self.options.require_reset_on_all_registers
            && input_count >= self.options.function_options.max_params
        {
            0
        } else {
            self.options.max_registers.min(max_by_nodes)
        };
        if max_register_count < minimum_registers {
            return Err(GenerationError::InvalidOptions(format!(
                "minimum register count {} cannot fit with {} input nodes and max_nodes {}",
                minimum_registers, input_count, self.options.function_options.max_nodes
            )));
        }
        Ok(choose_between(
            self.source,
            minimum_registers,
            max_register_count,
        ))
    }

    fn add_block_param(
        &mut self,
        generator: &mut FunctionGenerator<'_>,
        metadata: &mut BlockMetadata,
        name: String,
        ty: Type,
    ) -> NodeRef {
        let node_ref = generator.add_named_param(name.clone(), ty);
        metadata.input_port_ids.insert(
            name.clone(),
            generator.params.last().unwrap().id.get_wrapped_id(),
        );
        metadata.port_order.push(name);
        node_ref
    }

    fn maybe_add_reset_port(
        &mut self,
        generator: &mut FunctionGenerator<'_>,
        metadata: &mut BlockMetadata,
        register_count: usize,
        output_tuple_reserve: usize,
    ) -> Result<Option<NodeRef>, GenerationError> {
        let reset_required = self.options.require_reset_on_all_registers && register_count > 0;
        if register_count == 0
            || !self.options.allow_reset
            || (!reset_required && self.source.take_u64() & 1 == 0)
        {
            return Ok(None);
        }
        let used_nodes = generator.nodes.len().saturating_sub(1);
        let reserved_nodes = register_count
            .saturating_mul(2)
            .saturating_add(output_tuple_reserve);
        if generator.params.len() >= self.options.function_options.max_params
            || used_nodes.saturating_add(1).saturating_add(reserved_nodes)
                > self.options.function_options.max_nodes
        {
            if reset_required {
                return Err(GenerationError::Construction(
                    "required register reset port does not fit configured block limits".to_string(),
                ));
            }
            return Ok(None);
        }

        let reset_ref = self.add_block_param(generator, metadata, "rst".to_string(), Type::Bits(1));
        let asynchronous = match self.options.reset_timing {
            RandomBlockResetTiming::Synchronous => false,
            RandomBlockResetTiming::Asynchronous => true,
            RandomBlockResetTiming::Either => self.source.take_u64() & 1 != 0,
        };
        metadata.reset = Some(BlockResetMetadata {
            port_name: "rst".to_string(),
            asynchronous,
            active_low: self.source.take_u64() & 1 != 0,
        });
        Ok(Some(reset_ref))
    }

    fn add_registers(
        &mut self,
        generator: &mut FunctionGenerator<'_>,
        metadata: &mut BlockMetadata,
        register_count: usize,
        reset_ref: Option<NodeRef>,
    ) -> Vec<GeneratedRegisterState> {
        if register_count != 0 {
            metadata.clock_port_name = Some("clk".to_string());
            metadata.port_order.insert(0, "clk".to_string());
        }

        let reset_enabled = self.choose_register_reset_enabled(register_count, reset_ref);
        let mut registers = Vec::with_capacity(register_count);
        for index in 0..register_count {
            let name = format!("r{index}");
            let ty = if self.options.topology == BlockTopology::FeedForwardPipeline {
                generator.params[0].ty.clone()
            } else {
                self.choose_interface_type(generator)
            };
            let register_reset_ref = if reset_enabled[index] {
                reset_ref
            } else {
                None
            };
            let reset_value = register_reset_ref.map(|_| generate_uniform_value(self.source, &ty));
            metadata.registers.push(Register {
                name: name.clone(),
                ty: ty.clone(),
                reset_value,
            });
            generator.add_node(
                ty.clone(),
                NodePayload::RegisterRead {
                    register: name.clone(),
                },
                Some(format!("{name}_q")),
            );
            registers.push(GeneratedRegisterState {
                name,
                ty,
                reset_ref: register_reset_ref,
            });
        }
        registers
    }

    /// Reuses compatible types without coupling final wiring to body entropy.
    fn choose_interface_type(&mut self, generator: &FunctionGenerator<'_>) -> Type {
        let existing =
            block_data_types(generator, self.options.allow_zero_width_ports_and_registers);
        if !existing.is_empty() && self.source.take_u64() % 3 == 0 {
            existing[choose_count(self.source, existing.len())].clone()
        } else {
            random_block_interface_type(
                self.source,
                &self.options.function_options,
                self.options.allow_zero_width_ports_and_registers,
            )
        }
    }

    /// Generates ordinary random logic in stage-local pools, retaining all
    /// nodes and selecting register D/enables from the reserved wiring header.
    fn generate_pipeline_body(
        &mut self,
        generator: &mut FunctionGenerator<'_>,
        registers: &[GeneratedRegisterState],
        output_tuple_reserve: usize,
    ) -> Result<Vec<GeneratedRegisterWrite>, GenerationError> {
        let capacity = self.options.function_options.max_nodes.saturating_sub(
            generator.nodes.len().saturating_sub(1) + registers.len() + output_tuple_reserve,
        );
        let reads: Vec<_> = generator
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                matches!(node.payload, NodePayload::RegisterRead { .. })
                    .then_some(NodeRef { index })
            })
            .collect();
        let params: Vec<_> = generator
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                matches!(node.payload, NodePayload::GetParam(_)).then_some(NodeRef { index })
            })
            .collect();
        let mut writes = Vec::new();
        let mut body_nodes = 0;
        let mut stopped = false;
        for stage in 0..=registers.len() {
            generator.nodes_by_type.clear();
            let available = if stage == 0 {
                params.as_slice()
            } else {
                std::slice::from_ref(&reads[stage - 1])
            };
            for node_ref in available {
                generator
                    .nodes_by_type
                    .entry(generator.get_node(*node_ref).ty.clone())
                    .or_default()
                    .push(*node_ref);
            }
            let end = capacity * (stage + 1) / (registers.len() + 1);
            while body_nodes < end && !stopped {
                if !should_add_node(self.source, self.stop_policy, body_nodes) {
                    stopped = true;
                    break;
                }
                let added = generator.add_random_body_node(
                    self.source,
                    operation_budget(self.stop_policy, end - body_nodes, body_nodes),
                )?;
                if added == 0 {
                    stopped = true;
                    break;
                }
                body_nodes += added;
            }
            if let Some(register) = registers.get(stage) {
                let (next, enable) = self.register_wiring[stage];
                writes.push(GeneratedRegisterWrite {
                    next_ref: self.choose_register_next_ref(generator, register, next)?,
                    load_enable: self.maybe_choose_load_enable_ref(generator, enable),
                });
            }
        }
        // Output wiring may select any stage, including dead nodes. Restoring
        // the full pool also preserves the ordinary return-selection contract.
        generator.nodes_by_type.clear();
        for (index, node) in generator.nodes.iter().enumerate().skip(1) {
            generator
                .nodes_by_type
                .entry(node.ty.clone())
                .or_default()
                .push(NodeRef { index });
        }
        Ok(writes)
    }

    fn choose_register_reset_enabled(
        &mut self,
        register_count: usize,
        reset_ref: Option<NodeRef>,
    ) -> Vec<bool> {
        if reset_ref.is_none() {
            return vec![false; register_count];
        }
        if self.options.require_reset_on_all_registers {
            return vec![true; register_count];
        }
        let mut enabled: Vec<bool> = (0..register_count)
            .map(|_| self.source.take_u64() & 1 != 0)
            .collect();
        if register_count > 1 {
            if enabled.iter().all(|value| *value) {
                enabled[register_count - 1] = false;
            } else if enabled.iter().all(|value| !*value) {
                enabled[0] = true;
            }
        }
        enabled
    }

    fn generate_body(
        &mut self,
        generator: &mut FunctionGenerator<'_>,
        register_count: usize,
        output_tuple_reserve: usize,
    ) -> Result<(), GenerationError> {
        let used_nodes = generator.nodes.len().saturating_sub(1);
        let reserved_nodes = register_count.saturating_add(output_tuple_reserve);
        let body_capacity = self
            .options
            .function_options
            .max_nodes
            .saturating_sub(used_nodes.saturating_add(reserved_nodes));
        let mut body_nodes = 0;
        if block_data_types(generator, self.options.allow_zero_width_ports_and_registers).is_empty()
            && self.options.max_output_ports > 0
        {
            if body_capacity == 0 {
                if self.options.min_output_ports == 0 {
                    return Ok(());
                }
                return Err(GenerationError::Construction(
                    "block has no data value available for an output".to_string(),
                ));
            }
            self.add_random_data_literal(generator);
            body_nodes += 1;
        }

        while body_nodes < body_capacity
            && should_add_node(self.source, self.stop_policy, body_nodes)
        {
            let added = generator.add_random_body_node(
                self.source,
                operation_budget(self.stop_policy, body_capacity - body_nodes, body_nodes),
            )?;
            if added == 0 {
                break;
            }
            body_nodes += added;
        }
        Ok(())
    }

    fn add_random_data_literal(&mut self, generator: &mut FunctionGenerator<'_>) -> NodeRef {
        let ty = random_block_interface_type(
            self.source,
            &self.options.function_options,
            self.options.allow_zero_width_ports_and_registers,
        );
        generator.add_node(
            ty.clone(),
            NodePayload::Literal(generate_uniform_value(self.source, &ty)),
            None,
        )
    }

    fn choose_output_count(
        &mut self,
        generator: &FunctionGenerator<'_>,
        register_count: usize,
    ) -> Result<usize, GenerationError> {
        let used_nodes = generator.nodes.len().saturating_sub(1);
        let remaining_after_register_writes = self
            .options
            .function_options
            .max_nodes
            .saturating_sub(used_nodes.saturating_add(register_count));
        let data_available =
            !block_data_types(generator, self.options.allow_zero_width_ports_and_registers)
                .is_empty();
        let (min_output_count, max_output_count) = if !data_available {
            (self.options.min_output_ports, 0)
        } else if remaining_after_register_writes == 0 {
            (self.options.min_output_ports.max(1), 1)
        } else {
            (self.options.min_output_ports, self.options.max_output_ports)
        };
        if max_output_count < min_output_count {
            return Err(GenerationError::Construction(format!(
                "minimum output count {} cannot fit after generated body",
                self.options.min_output_ports
            )));
        }
        Ok(choose_between(
            self.source,
            min_output_count,
            max_output_count,
        ))
    }

    fn choose_output_refs(
        &mut self,
        generator: &FunctionGenerator<'_>,
        output_count: usize,
    ) -> Result<Vec<NodeRef>, GenerationError> {
        if output_count == 0 {
            return Ok(Vec::new());
        }
        if block_data_types(generator, self.options.allow_zero_width_ports_and_registers).is_empty()
        {
            return Err(GenerationError::Construction(
                "block has no data value available for an output".to_string(),
            ));
        }
        let candidates: Vec<NodeRef> = generator
            .nodes
            .iter()
            .enumerate()
            // The dummy node has type `()`, as do legitimate zero-bit values.
            .skip(1)
            .filter(|(_, node)| {
                type_is_block_data(&node.ty)
                    && (self.options.allow_zero_width_ports_and_registers
                        || node.ty.bit_count() != 0)
            })
            .map(|(index, _)| NodeRef { index })
            .collect();
        Ok(self.output_wiring[..output_count]
            .iter()
            .map(|choice| candidates[(*choice % candidates.len() as u64) as usize])
            .collect())
    }

    fn materialize_block_return(
        &mut self,
        generator: &mut FunctionGenerator<'_>,
        output_refs: &[NodeRef],
    ) -> Result<NodeRef, GenerationError> {
        if output_refs.len() == 1 {
            return Ok(output_refs[0]);
        }
        let fields = output_refs
            .iter()
            .map(|node_ref| Box::new(generator.get_node(*node_ref).ty.clone()))
            .collect();
        Ok(generator.add_node(
            Type::Tuple(fields),
            NodePayload::Tuple(output_refs.to_vec()),
            Some("outputs".to_string()),
        ))
    }

    fn choose_register_next_ref(
        &mut self,
        generator: &FunctionGenerator<'_>,
        register: &GeneratedRegisterState,
        choice: u64,
    ) -> Result<NodeRef, GenerationError> {
        let refs = generator.nodes_by_type.get(&register.ty).ok_or_else(|| {
            GenerationError::Construction(format!(
                "no value available for register write '{}'",
                register.name
            ))
        })?;
        Ok(refs[(choice % refs.len() as u64) as usize])
    }

    fn maybe_choose_load_enable_ref(
        &mut self,
        generator: &FunctionGenerator<'_>,
        choice: u64,
    ) -> Option<NodeRef> {
        if !self.options.allow_load_enable {
            return None;
        }
        let refs = generator.nodes_by_type.get(&Type::Bits(1))?;
        let index = (choice % (refs.len() as u64 + 1)) as usize;
        index.checked_sub(1).map(|index| refs[index])
    }

    fn populate_output_metadata(
        &self,
        generator: &FunctionGenerator<'_>,
        metadata: &mut BlockMetadata,
        output_count: usize,
    ) {
        metadata.output_names = if output_count == 1 {
            vec!["out".to_string()]
        } else {
            (0..output_count)
                .map(|index| format!("out{index}"))
                .collect()
        };
        let mut next_id = generator
            .nodes
            .iter()
            .map(|node| node.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        for name in &metadata.output_names {
            metadata.output_port_ids.insert(name.clone(), next_id);
            metadata.port_order.push(name.clone());
            next_id = next_id.saturating_add(1);
        }
    }
}

struct PackageGenerator<'a, S> {
    source: &'a mut S,
    options: &'a RandomFnOptions,
    stop_policy: StopPolicy,
    completed_functions: Vec<CompletedFunction>,
    function_stats: Vec<GeneratedFnStats>,
    next_function_index: usize,
    next_text_id: usize,
}

impl<'a, S: EntropySource> PackageGenerator<'a, S> {
    fn new(source: &'a mut S, options: &'a RandomFnOptions, stop_policy: StopPolicy) -> Self {
        Self {
            source,
            options,
            stop_policy,
            completed_functions: Vec::new(),
            function_stats: Vec::new(),
            next_function_index: 0,
            next_text_id: 1,
        }
    }

    fn generate(mut self) -> Result<GeneratedPackage, GenerationError> {
        let top_name = self.allocate_function_name();
        self.generate_unconstrained_function(
            top_name.clone(),
            self.options.max_nested_counted_for_iterations,
        )?;
        let members = self
            .completed_functions
            .into_iter()
            .map(|completed| PackageMember::Function(completed.function))
            .collect();
        let package = Package {
            name: "random_package".to_string(),
            file_table: FileTable::new(),
            members,
            top: Some((top_name, MemberType::Function)),
        };
        crate::ir_verify::verify_package(&package)
            .map_err(|error| GenerationError::Construction(error.to_string()))?;
        Ok(GeneratedPackage {
            package,
            function_stats: self.function_stats,
        })
    }

    fn allocate_function_name(&mut self) -> String {
        let name = format!("random_fn_{}", self.next_function_index);
        self.next_function_index += 1;
        name
    }

    fn generate_unconstrained_function(
        &mut self,
        name: String,
        nested_iteration_budget: usize,
    ) -> Result<(), GenerationError> {
        let mut generator = FunctionGenerator::new(self.options);
        let max_params = self
            .options
            .max_params
            .min(self.options.max_nodes.saturating_sub(1));
        let param_count = choose_count(self.source, max_params + 1);
        for _ in 0..param_count {
            let ty = random_type(self.source, self.options, 0);
            generator.add_param(ty);
        }

        let body_capacity = self.options.max_nodes - generator.params.len();
        let mut body_nodes = 0;
        let mut invoke_count = 0;
        let mut counted_for_count = 0;
        while body_nodes < body_capacity
            && (body_nodes == 0 || should_add_node(self.source, self.stop_policy, body_nodes))
        {
            let added = self.add_random_body_node(
                &mut generator,
                &mut invoke_count,
                &mut counted_for_count,
                nested_iteration_budget,
                operation_budget(self.stop_policy, body_capacity - body_nodes, body_nodes),
            )?;
            if added == 0 {
                break;
            }
            body_nodes += added;
        }
        if generator.nodes.len() == 1 {
            generator.emit_operation(self.source, RandomOperation::Literal)?;
        }
        let function = generator.finish()?;
        self.finish_function(name, function);
        Ok(())
    }

    fn generate_function_with_signature(
        &mut self,
        name: String,
        signature: &FunctionSignature,
        nested_iteration_budget: usize,
    ) -> Result<(), GenerationError> {
        validate_signature(signature, self.options)?;
        let mut generator = FunctionGenerator::new(self.options);
        for ty in &signature.params {
            generator.add_param(ty.clone());
        }

        let terminal_node_budget =
            generator.minimum_materialization_nodes(&signature.return_type)?;
        let required_node_count = generator.params.len() + terminal_node_budget;
        if required_node_count > self.options.max_nodes {
            return Err(GenerationError::InvalidSignature(format!(
                "signature requires at least {required_node_count} nodes but max_nodes is {}",
                self.options.max_nodes
            )));
        }

        let body_capacity = self.options.max_nodes - required_node_count;
        let mut body_nodes = 0;
        let mut invoke_count = 0;
        let mut counted_for_count = 0;
        while body_nodes < body_capacity
            && should_add_node(self.source, self.stop_policy, body_nodes)
        {
            let added = self.add_random_body_node(
                &mut generator,
                &mut invoke_count,
                &mut counted_for_count,
                nested_iteration_budget,
                operation_budget(self.stop_policy, body_capacity - body_nodes, body_nodes),
            )?;
            if added == 0 {
                break;
            }
            body_nodes += added;
        }
        let ret_node_ref =
            generator.pick_or_generate_value_of_type(self.source, &signature.return_type)?;
        let function = generator.finish_with_return(ret_node_ref)?;
        self.finish_function(name, function);
        Ok(())
    }

    fn finish_function(&mut self, name: String, mut function: Fn) {
        function.name = name;
        let function = rebase_fn_ids(&function, self.next_text_id);
        self.next_text_id = function
            .nodes
            .iter()
            .map(|node| node.text_id)
            .max()
            .unwrap_or(self.next_text_id)
            .saturating_add(1);
        self.function_stats.push(gather_stats(&function));
        let max_nested_counted_for_iterations =
            self.function_max_nested_counted_for_iterations(&function);
        self.completed_functions.push(CompletedFunction {
            function,
            max_nested_counted_for_iterations,
        });
    }

    fn add_random_body_node(
        &mut self,
        generator: &mut FunctionGenerator<'_>,
        invoke_count: &mut usize,
        counted_for_count: &mut usize,
        nested_iteration_budget: usize,
        budget: usize,
    ) -> Result<usize, GenerationError> {
        let state = generator.begin_budgeted_operation(self.source, budget);
        let mut applicable = generator.applicable_operations();
        if *invoke_count < self.options.max_invokes_per_function
            && self.can_emit_invoke(generator, nested_iteration_budget)
        {
            applicable.push(RandomOperation::Invoke);
        }
        if *counted_for_count < self.options.max_counted_fors_per_function
            && self.can_emit_counted_for(generator)
        {
            applicable.push(RandomOperation::CountedFor);
        }
        if applicable.is_empty() {
            generator.discard_budgeted_operation();
            return Err(GenerationError::Construction(
                "no operation can be emitted from the available values".to_string(),
            ));
        }
        let operation = applicable[choose_count(self.source, applicable.len())];
        if operation == RandomOperation::Invoke {
            generator.discard_budgeted_operation();
            *invoke_count += 1;
            let before = generator.nodes.len();
            self.try_emit_invoke(generator, nested_iteration_budget)?
                .expect("invoke was selected only after checking applicability");
            debug_assert!(generator.nodes.len() - before <= budget);
            return Ok(generator.nodes.len() - before);
        }
        if operation == RandomOperation::CountedFor {
            generator.discard_budgeted_operation();
            *counted_for_count += 1;
            let before = generator.nodes.len();
            self.try_emit_counted_for(generator, nested_iteration_budget)?
                .expect("counted_for was selected only after checking applicability");
            debug_assert!(generator.nodes.len() - before <= budget);
            return Ok(generator.nodes.len() - before);
        }
        generator.emit_budgeted_operation(self.source, operation, state, budget)
    }

    fn can_emit_invoke(
        &self,
        generator: &FunctionGenerator<'_>,
        nested_iteration_budget: usize,
    ) -> bool {
        self.next_function_index < self.options.max_functions
            || !self
                .callable_completed_functions(generator, nested_iteration_budget)
                .is_empty()
    }

    fn try_emit_invoke(
        &mut self,
        generator: &mut FunctionGenerator<'_>,
        nested_iteration_budget: usize,
    ) -> Result<Option<NodeRef>, GenerationError> {
        let existing = self.callable_completed_functions(generator, nested_iteration_budget);
        let can_create = self.next_function_index < self.options.max_functions;
        if existing.is_empty() && !can_create {
            return Ok(None);
        }
        let create_new = can_create && (existing.is_empty() || self.source.take_u64() & 1 == 0);
        let signature = if create_new {
            let signature = self.random_callable_signature(generator);
            let name = self.allocate_function_name();
            self.generate_function_with_signature(
                name.clone(),
                &signature,
                nested_iteration_budget,
            )?;
            let completed = self
                .completed_function(&name)
                .expect("newly generated callee is complete");
            CallableFunction::from_completed(completed)
        } else {
            existing[choose_count(self.source, existing.len())].clone()
        };
        let operands = signature
            .signature
            .params
            .iter()
            .map(|ty| generator.choose_ref_for_type(self.source, ty))
            .collect();
        Ok(Some(generator.add_node(
            signature.signature.return_type,
            NodePayload::Invoke {
                to_apply: signature.name,
                operands,
            },
            None,
        )))
    }

    fn can_emit_counted_for(&self, generator: &FunctionGenerator<'_>) -> bool {
        if !self
            .options
            .enabled_operations
            .contains(RandomOperation::CountedFor)
            || generator.nodes_by_type.is_empty()
        {
            return false;
        }
        self.can_create_counted_for_body()
            || !self.reusable_counted_for_bodies(generator).is_empty()
    }

    fn can_create_counted_for_body(&self) -> bool {
        self.next_function_index < self.options.max_functions
            && self.options.max_params >= 2
            && self.options.max_nodes >= 2
    }

    fn try_emit_counted_for(
        &mut self,
        generator: &mut FunctionGenerator<'_>,
        nested_iteration_budget: usize,
    ) -> Result<Option<NodeRef>, GenerationError> {
        let existing = self.reusable_counted_for_bodies(generator);
        let can_create = self.can_create_counted_for_body();
        if existing.is_empty() && !can_create {
            return Ok(None);
        }
        let create_new = can_create && (existing.is_empty() || self.source.take_u64() & 1 == 0);
        let emission = if create_new {
            self.create_counted_for_body(generator, nested_iteration_budget)?
        } else {
            let body_index = choose_count(self.source, existing.len());
            self.reuse_counted_for_body(generator, &existing[body_index], nested_iteration_budget)
        };
        let ty = generator.get_node(emission.init).ty.clone();
        Ok(Some(generator.add_node(
            ty,
            NodePayload::CountedFor {
                init: emission.init,
                trip_count: emission.trip_count,
                stride: emission.stride,
                body: emission.body,
                invariant_args: emission.invariant_args,
            },
            None,
        )))
    }

    fn create_counted_for_body(
        &mut self,
        generator: &FunctionGenerator<'_>,
        nested_iteration_budget: usize,
    ) -> Result<CountedForEmission, GenerationError> {
        let attributes = self.random_new_counted_for_attributes(nested_iteration_budget);
        let init = generator.choose_any_ref(self.source);
        let max_invariant_args = self
            .options
            .max_params
            .min(self.options.max_nodes)
            .saturating_sub(2);
        let invariant_args: Vec<NodeRef> = (0..choose_count(self.source, max_invariant_args + 1))
            .map(|_| generator.choose_any_ref(self.source))
            .collect();
        let mut params = vec![
            Type::Bits(attributes.induction_width),
            generator.get_node(init).ty.clone(),
        ];
        params.extend(
            invariant_args
                .iter()
                .map(|node_ref| generator.get_node(*node_ref).ty.clone()),
        );
        let signature = FunctionSignature {
            return_type: generator.get_node(init).ty.clone(),
            params,
        };
        let body = self.allocate_function_name();
        let body_budget = nested_iteration_budget / attributes.trip_count.max(1);
        self.generate_function_with_signature(body.clone(), &signature, body_budget)?;
        Ok(CountedForEmission {
            body,
            init,
            invariant_args,
            trip_count: attributes.trip_count,
            stride: attributes.stride,
        })
    }

    fn reuse_counted_for_body(
        &mut self,
        generator: &FunctionGenerator<'_>,
        body: &CallableFunction,
        nested_iteration_budget: usize,
    ) -> CountedForEmission {
        let Type::Bits(induction_width) = &body.signature.params[0] else {
            unreachable!("reusable counted_for bodies have bits-typed induction parameters")
        };
        let max_trip_count = if body.max_nested_counted_for_iterations > nested_iteration_budget {
            0
        } else {
            self.options
                .max_counted_for_trip_count
                .min(nested_iteration_budget / body.max_nested_counted_for_iterations)
        };
        let trip_count = random_biased_bounded(self.source, max_trip_count);
        let stride = self.random_counted_for_stride(trip_count, *induction_width);
        let init = generator.choose_ref_for_type(self.source, &body.signature.params[1]);
        let invariant_args = body.signature.params[2..]
            .iter()
            .map(|ty| generator.choose_ref_for_type(self.source, ty))
            .collect();
        CountedForEmission {
            body: body.name.clone(),
            init,
            invariant_args,
            trip_count,
            stride,
        }
    }

    fn random_new_counted_for_attributes(
        &mut self,
        nested_iteration_budget: usize,
    ) -> CountedForAttributes {
        let max_trip_count = self
            .options
            .max_counted_for_trip_count
            .min(nested_iteration_budget);
        let trip_count = random_biased_bounded(self.source, max_trip_count);
        let mut stride = random_biased_bounded(self.source, self.options.max_counted_for_stride);
        let mut minimum_width = counted_for_minimum_induction_width(trip_count, stride);
        if minimum_width.is_none_or(|width| width > self.options.max_bit_width) {
            stride = 0;
            minimum_width = counted_for_minimum_induction_width(trip_count, stride);
        }
        let minimum_width = minimum_width
            .expect("zero stride always has a representable induction range")
            .max(usize::from(!self.options.allow_zero_width_bits));
        let induction_width = if self.source.take_u64() & 1 == 0 {
            minimum_width
        } else {
            choose_between(self.source, minimum_width, self.options.max_bit_width)
        };
        CountedForAttributes {
            trip_count,
            stride,
            induction_width,
        }
    }

    fn random_counted_for_stride(&mut self, trip_count: usize, induction_width: usize) -> usize {
        let stride = random_biased_bounded(self.source, self.options.max_counted_for_stride);
        if counted_for_minimum_induction_width(trip_count, stride)
            .is_some_and(|width| width <= induction_width)
        {
            stride
        } else {
            0
        }
    }

    fn random_callable_signature(
        &mut self,
        generator: &FunctionGenerator<'_>,
    ) -> FunctionSignature {
        let available_types: Vec<Type> = generator.nodes_by_type.keys().cloned().collect();
        let max_params = self
            .options
            .max_params
            .min(self.options.max_nodes.saturating_sub(1));
        let param_count = if available_types.is_empty() {
            0
        } else {
            choose_count(self.source, max_params + 1)
        };
        let params: Vec<Type> = (0..param_count)
            .map(|_| available_types[choose_count(self.source, available_types.len())].clone())
            .collect();
        let return_type = self.random_materializable_return_type(&params);
        FunctionSignature {
            params,
            return_type,
        }
    }

    fn random_materializable_return_type(&mut self, params: &[Type]) -> Type {
        let available_node_budget = self.options.max_nodes - params.len();
        for _ in 0..4 {
            let ty = random_type(self.source, self.options, 0);
            let mut available_types: BTreeSet<Type> = params.iter().cloned().collect();
            if required_materialization_nodes(&mut available_types, &ty)
                .is_ok_and(|count| count <= available_node_budget)
            {
                return ty;
            }
        }
        Type::Bits(random_width(self.source, self.options.max_bit_width))
    }

    fn callable_completed_functions(
        &self,
        generator: &FunctionGenerator<'_>,
        nested_iteration_budget: usize,
    ) -> Vec<CallableFunction> {
        self.completed_functions
            .iter()
            .filter(|completed| {
                completed.max_nested_counted_for_iterations <= nested_iteration_budget
                    && completed
                        .function
                        .params
                        .iter()
                        .all(|param| generator.nodes_by_type.contains_key(&param.ty))
            })
            .map(CallableFunction::from_completed)
            .collect()
    }

    fn reusable_counted_for_bodies(
        &self,
        generator: &FunctionGenerator<'_>,
    ) -> Vec<CallableFunction> {
        self.completed_functions
            .iter()
            .filter(|completed| {
                let function = &completed.function;
                matches!(function.params.first(), Some(Param { ty: Type::Bits(width), .. }) if *width > 0)
                    && function.params.len() >= 2
                    && function.ret_ty == function.params[1].ty
                    && function.params[1..]
                        .iter()
                        .all(|param| generator.nodes_by_type.contains_key(&param.ty))
            })
            .map(CallableFunction::from_completed)
            .collect()
    }

    fn completed_function(&self, name: &str) -> Option<&CompletedFunction> {
        self.completed_functions
            .iter()
            .find(|completed| completed.function.name == name)
    }

    fn function_max_nested_counted_for_iterations(&self, function: &Fn) -> usize {
        function.nodes.iter().fold(1, |maximum, node| {
            let expansion = match &node.payload {
                NodePayload::Invoke { to_apply, .. } => {
                    self.completed_function(to_apply)
                        .expect("invoke callee is complete before caller")
                        .max_nested_counted_for_iterations
                }
                NodePayload::CountedFor {
                    trip_count, body, ..
                } => trip_count.saturating_mul(
                    self.completed_function(body)
                        .expect("counted_for body is complete before caller")
                        .max_nested_counted_for_iterations,
                ),
                _ => 1,
            };
            maximum.max(expansion)
        })
    }
}

#[derive(Debug, Clone)]
struct CompletedFunction {
    function: Fn,
    max_nested_counted_for_iterations: usize,
}

#[derive(Debug, Clone)]
struct CallableFunction {
    name: String,
    signature: FunctionSignature,
    max_nested_counted_for_iterations: usize,
}

struct CountedForEmission {
    body: String,
    init: NodeRef,
    invariant_args: Vec<NodeRef>,
    trip_count: usize,
    stride: usize,
}

struct CountedForAttributes {
    trip_count: usize,
    stride: usize,
    induction_width: usize,
}

impl CallableFunction {
    fn from_completed(completed: &CompletedFunction) -> Self {
        Self {
            name: completed.function.name.clone(),
            signature: FunctionSignature::from_fn(&completed.function),
            max_nested_counted_for_iterations: completed.max_nested_counted_for_iterations,
        }
    }
}

/// Generates a typed PIR function with exactly the requested parameter and
/// return types.
///
/// `max_nodes` includes any nodes inserted to materialize the requested return
/// type. Unlike unconstrained generation, `ExactBodyNodes(0)` is meaningful:
/// the terminal materialization itself provides the return value.
pub fn generate_fn_with_signature<S: EntropySource>(
    source: &mut S,
    options: &RandomFnOptions,
    stop_policy: StopPolicy,
    signature: &FunctionSignature,
) -> Result<GeneratedFn, GenerationError> {
    options.validate()?;
    validate_stop_policy(stop_policy)?;
    validate_signature(signature, options)?;

    let mut generator = FunctionGenerator::new(options);
    for ty in &signature.params {
        generator.add_param(ty.clone());
    }

    let terminal_node_budget = generator.minimum_materialization_nodes(&signature.return_type)?;
    let required_node_count = generator.params.len() + terminal_node_budget;
    if required_node_count > options.max_nodes {
        return Err(GenerationError::InvalidSignature(format!(
            "signature requires at least {required_node_count} nodes but max_nodes is {}",
            options.max_nodes
        )));
    }

    let body_capacity = options.max_nodes - required_node_count;
    let mut body_nodes = 0;
    while body_nodes < body_capacity && should_add_node(source, stop_policy, body_nodes) {
        let added = generator.add_random_body_node(
            source,
            operation_budget(stop_policy, body_capacity - body_nodes, body_nodes),
        )?;
        if added == 0 {
            break;
        }
        body_nodes += added;
    }

    let ret_node_ref = generator.pick_or_generate_value_of_type(source, &signature.return_type)?;
    let function = generator.finish_with_return(ret_node_ref)?;
    let stats = gather_stats(&function);
    debug_assert!(stats.emitted_node_count <= options.max_nodes);
    Ok(GeneratedFn { function, stats })
}

/// Generates two independently randomized functions with an identical
/// signature, using separate entropy sources for their bodies.
pub fn generate_same_signature_pair<S1: EntropySource, S2: EntropySource>(
    first_source: &mut S1,
    second_source: &mut S2,
    options: &RandomFnOptions,
    stop_policy: StopPolicy,
) -> Result<(GeneratedFn, GeneratedFn), GenerationError> {
    let first = generate_fn(first_source, options, stop_policy)?;
    let signature = FunctionSignature::from_fn(&first.function);
    let second = generate_fn_with_signature(second_source, options, stop_policy, &signature)?;
    Ok((first, second))
}

fn validate_stop_policy(stop_policy: StopPolicy) -> Result<(), GenerationError> {
    match stop_policy {
        StopPolicy::WhenEntropyDepleted | StopPolicy::ExactBodyNodes(_) => Ok(()),
        StopPolicy::Geometric {
            min_body_nodes,
            max_body_nodes,
            stop_numerator,
            stop_denominator,
        } => {
            if max_body_nodes < min_body_nodes {
                return Err(GenerationError::InvalidOptions(
                    "geometric maximum must be at least its minimum".to_string(),
                ));
            }
            if stop_denominator == 0 || stop_numerator > stop_denominator {
                return Err(GenerationError::InvalidOptions(
                    "geometric stop probability must be a valid fraction".to_string(),
                ));
            }
            Ok(())
        }
    }
}

fn validate_signature(
    signature: &FunctionSignature,
    options: &RandomFnOptions,
) -> Result<(), GenerationError> {
    if signature.params.len() > options.max_params {
        return Err(GenerationError::InvalidSignature(format!(
            "signature has {} parameters but max_params is {}",
            signature.params.len(),
            options.max_params
        )));
    }
    for ty in signature.params.iter().chain([&signature.return_type]) {
        validate_signature_type(ty, options)?;
    }
    Ok(())
}

fn validate_signature_type(ty: &Type, options: &RandomFnOptions) -> Result<(), GenerationError> {
    if type_depth(ty) > options.max_type_depth {
        return Err(GenerationError::InvalidSignature(format!(
            "type {ty} exceeds max_type_depth {}",
            options.max_type_depth
        )));
    }
    if type_leaf_count(ty) > options.max_aggregate_leaves {
        return Err(GenerationError::InvalidSignature(format!(
            "type {ty} exceeds max_aggregate_leaves {}",
            options.max_aggregate_leaves
        )));
    }
    match ty {
        Type::Token if options.allow_events => Ok(()),
        Type::Token => Err(GenerationError::InvalidSignature(
            "token signature type requires allow_events".to_string(),
        )),
        Type::Bits(width)
            if *width > options.max_bit_width
                || (*width == 0 && !options.allow_zero_width_bits) =>
        {
            let minimum_width = usize::from(!options.allow_zero_width_bits);
            Err(GenerationError::InvalidSignature(format!(
                "bits width {width} is outside the supported range {minimum_width}..={}",
                options.max_bit_width
            )))
        }
        Type::Bits(_) => Ok(()),
        Type::Tuple(fields) => {
            if !options.allow_tuples {
                return Err(GenerationError::InvalidSignature(
                    "tuple signature type requires allow_tuples".to_string(),
                ));
            }
            if fields.len() > options.max_tuple_length {
                return Err(GenerationError::InvalidSignature(format!(
                    "tuple field count {} is outside the supported range 0..={}",
                    fields.len(),
                    options.max_tuple_length
                )));
            }
            for field in fields {
                validate_signature_type(field, options)?;
            }
            Ok(())
        }
        Type::Array(array) => {
            if !options.allow_arrays {
                return Err(GenerationError::InvalidSignature(
                    "array signature type requires allow_arrays".to_string(),
                ));
            }
            if array.element_count == 0 || array.element_count > options.max_array_length {
                return Err(GenerationError::InvalidSignature(format!(
                    "array length {} is outside the supported range 1..={}",
                    array.element_count, options.max_array_length
                )));
            }
            validate_signature_type(&array.element_type, options)
        }
    }
}

fn should_add_node<S: EntropySource>(
    source: &mut S,
    stop_policy: StopPolicy,
    body_nodes: usize,
) -> bool {
    match stop_policy {
        StopPolicy::WhenEntropyDepleted => !source.is_depleted(),
        StopPolicy::ExactBodyNodes(count) => body_nodes < count,
        StopPolicy::Geometric {
            min_body_nodes,
            max_body_nodes,
            stop_numerator,
            stop_denominator,
        } => {
            if body_nodes < min_body_nodes.max(1) {
                return true;
            }
            if body_nodes >= max_body_nodes.max(1) {
                return false;
            }
            source.take_u64() % stop_denominator >= stop_numerator
        }
    }
}

/// Caps one operation transaction by both capacity and the exact-body target.
fn operation_budget(
    stop_policy: StopPolicy,
    capacity_remaining: usize,
    body_nodes: usize,
) -> usize {
    match stop_policy {
        StopPolicy::ExactBodyNodes(target) => {
            capacity_remaining.min(target.max(1).saturating_sub(body_nodes))
        }
        StopPolicy::WhenEntropyDepleted | StopPolicy::Geometric { .. } => capacity_remaining,
    }
}

fn choose_count<S: EntropySource>(source: &mut S, exclusive_limit: usize) -> usize {
    if exclusive_limit <= 1 {
        0
    } else {
        (source.take_u64() as usize) % exclusive_limit
    }
}

fn choose_between<S: EntropySource>(source: &mut S, minimum: usize, maximum: usize) -> usize {
    debug_assert!(minimum <= maximum);
    minimum + choose_count(source, maximum - minimum + 1)
}

fn random_biased_bounded<S: EntropySource>(source: &mut S, maximum: usize) -> usize {
    match source.take_u64() & 7 {
        0 => 0,
        1 => 1.min(maximum),
        2 => 2.min(maximum),
        3 => maximum,
        _ => choose_between(source, 0, maximum),
    }
}

fn counted_for_minimum_induction_width(trip_count: usize, stride: usize) -> Option<usize> {
    if trip_count <= 1 {
        return Some(1);
    }
    let max_induction_value = stride.checked_mul(trip_count - 1)?;
    if max_induction_value > i64::MAX as usize {
        return None;
    }
    if max_induction_value == 0 {
        Some(0)
    } else {
        Some(usize::BITS as usize - max_induction_value.leading_zeros() as usize)
    }
}

fn random_width<S: EntropySource>(source: &mut S, max_bit_width: usize) -> usize {
    let choice = source.take_u64();
    if max_bit_width > 64 && choice & 7 == 7 {
        let preferred: Vec<usize> = [65, 127, 128, 129, 255, 256, 257, max_bit_width]
            .into_iter()
            .filter(|width| *width <= max_bit_width)
            .collect();
        return if choice & 8 == 0 {
            preferred[((choice >> 4) as usize) % preferred.len()]
        } else {
            65 + (((choice >> 4) as usize) % (max_bit_width - 64))
        };
    }
    let narrow_max = max_bit_width.min(64);
    let preferred: Vec<usize> = [
        1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, narrow_max,
    ]
    .into_iter()
    .filter(|width| *width <= narrow_max)
    .collect();
    if choice & 1 == 0 {
        preferred[((choice >> 1) as usize) % preferred.len()]
    } else {
        1 + (((choice >> 1) as usize) % narrow_max)
    }
}

/// Gives one-bit data 10% probability and shares the rest across width bands.
fn balanced_width<S: EntropySource>(source: &mut S, maximum: usize) -> usize {
    if maximum == 1 || choose_count(source, 10) == 0 {
        return 1;
    }
    let ranges: Vec<_> = [(2, 8), (9, 16), (17, 32), (33, 64), (65, maximum)]
        .into_iter()
        .filter(|(lo, _)| *lo <= maximum)
        .map(|(lo, hi)| (lo, hi.min(maximum)))
        .collect();
    let (lo, hi) = ranges[choose_count(source, ranges.len())];
    let mut boundaries = vec![lo, hi];
    for power in [2usize, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        for value in [power - 1, power, power + 1] {
            if (lo..=hi).contains(&value) {
                boundaries.push(value);
            }
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    if choose_count(source, 2) == 0 {
        boundaries[choose_count(source, boundaries.len())]
    } else {
        choose_between(source, lo, hi)
    }
}

fn configured_width<S: EntropySource>(
    source: &mut S,
    _options: &RandomFnOptions,
    maximum: usize,
) -> usize {
    balanced_width(source, maximum)
}

fn max_array_length_for_element(options: &RandomFnOptions, element_ty: &Type) -> usize {
    let element_leaves = type_leaf_count(element_ty);
    if element_leaves == 0 {
        options.max_array_length
    } else {
        options
            .max_array_length
            .min(options.max_aggregate_leaves / element_leaves)
    }
}

fn random_type<S: EntropySource>(source: &mut S, options: &RandomFnOptions, depth: usize) -> Type {
    if depth == 0 && options.allow_events && source.take_u64() % 16 == 0 {
        return Type::Token;
    }
    let may_aggregate = depth < options.max_type_depth;
    let family_count = 1
        + usize::from(may_aggregate && options.allow_arrays)
        + usize::from(may_aggregate && options.allow_tuples);
    let family = choose_count(source, family_count);
    if family == 0 {
        if options.allow_zero_width_bits && source.take_u64() & 15 == 0 {
            return Type::Bits(0);
        }
        return Type::Bits(configured_width(source, options, options.max_bit_width));
    }

    let arrays_are_second = options.allow_arrays;
    if arrays_are_second && family == 1 {
        let element_type = random_type(source, options, depth + 1);
        let max_length = max_array_length_for_element(options, &element_type).max(1);
        return Type::new_array(element_type, choose_between(source, 1, max_length));
    }

    let desired_length = choose_between(source, 0, options.max_tuple_length);
    let mut fields = Vec::with_capacity(desired_length);
    let mut leaves = 0;
    for _ in 0..desired_length {
        let field = random_type(source, options, depth + 1);
        let field_leaves = type_leaf_count(&field);
        if leaves + field_leaves <= options.max_aggregate_leaves {
            fields.push(Box::new(field));
            leaves += field_leaves;
        }
    }
    Type::Tuple(fields)
}

fn random_data_type<S: EntropySource>(
    source: &mut S,
    options: &RandomFnOptions,
    depth: usize,
) -> Type {
    let mut data_options = options.clone();
    data_options.allow_events = false;
    random_type(source, &data_options, depth)
}

/// Chooses a block port/register type, falling back after bounded rejection
/// sampling so pathological entropy cannot loop forever.
fn random_block_interface_type<S: EntropySource>(
    source: &mut S,
    options: &RandomFnOptions,
    allow_zero_width: bool,
) -> Type {
    const MAX_ATTEMPTS: usize = 16;
    for _ in 0..MAX_ATTEMPTS {
        let ty = random_data_type(source, options, 0);
        if allow_zero_width || ty.bit_count() != 0 {
            return ty;
        }
    }
    Type::Bits(1)
}

fn type_is_block_data(ty: &Type) -> bool {
    !type_contains_token(ty)
}

fn block_data_types(generator: &FunctionGenerator<'_>, allow_zero_width: bool) -> Vec<Type> {
    generator
        .nodes_by_type
        .keys()
        .filter(|ty| type_is_block_data(ty) && (allow_zero_width || ty.bit_count() != 0))
        .cloned()
        .collect()
}

fn type_leaf_count(ty: &Type) -> usize {
    match ty {
        Type::Token | Type::Bits(_) => 1,
        Type::Tuple(fields) => fields.iter().map(|field| type_leaf_count(field)).sum(),
        Type::Array(array) => type_leaf_count(&array.element_type) * array.element_count,
    }
}

fn type_contains_token(ty: &Type) -> bool {
    match ty {
        Type::Token => true,
        Type::Bits(_) => false,
        Type::Tuple(fields) => fields.iter().any(|field| type_contains_token(field)),
        Type::Array(array) => type_contains_token(&array.element_type),
    }
}

fn type_depth(ty: &Type) -> usize {
    match ty {
        Type::Token | Type::Bits(_) => 0,
        Type::Tuple(fields) => {
            1 + fields
                .iter()
                .map(|field| type_depth(field))
                .max()
                .unwrap_or(0)
        }
        Type::Array(array) => 1 + type_depth(&array.element_type),
    }
}

fn required_materialization_nodes(
    available_types: &mut BTreeSet<Type>,
    ty: &Type,
) -> Result<usize, GenerationError> {
    if available_types.contains(ty) {
        return Ok(0);
    }
    let nodes = match ty {
        Type::Token => 1,
        Type::Bits(_) => 1,
        Type::Tuple(fields) => {
            let mut nodes = 1;
            for field in fields {
                nodes += required_materialization_nodes(available_types, field)?;
            }
            nodes
        }
        Type::Array(array) => {
            let mut nodes = 1;
            for _ in 0..array.element_count {
                nodes += required_materialization_nodes(available_types, &array.element_type)?;
            }
            nodes
        }
    };
    available_types.insert(ty.clone());
    Ok(nodes)
}

struct FunctionGenerator<'a> {
    options: &'a RandomFnOptions,
    params: Vec<Param>,
    nodes: Vec<Node>,
    nodes_by_type: BTreeMap<Type, Vec<NodeRef>>,
    preparation_remaining: usize,
}

struct BudgetedOperationState {
    before: usize,
    incomplete_reads: usize,
    saved_pool: Option<BTreeMap<Type, Vec<NodeRef>>>,
}

impl<'a> FunctionGenerator<'a> {
    fn new(options: &'a RandomFnOptions) -> Self {
        Self {
            options,
            params: Vec::new(),
            nodes: vec![Node {
                text_id: 0,
                name: None,
                ty: Type::nil(),
                payload: NodePayload::Nil,
                pos: None,
            }],
            nodes_by_type: BTreeMap::new(),
            preparation_remaining: 0,
        }
    }

    fn add_param(&mut self, ty: Type) {
        let name = format!("p{}", self.params.len());
        self.add_named_param(name, ty);
    }

    fn add_named_param(&mut self, name: String, ty: Type) -> NodeRef {
        let id = ParamId::new(self.params.len() + 1);
        let node_ref = self.add_node(
            ty.clone(),
            NodePayload::GetParam(id.clone()),
            Some(name.clone()),
        );
        self.params.push(Param { name, ty, id });
        debug_assert_eq!(node_ref.index, self.params.len());
        node_ref
    }

    fn add_node(&mut self, ty: Type, payload: NodePayload, name: Option<String>) -> NodeRef {
        let node_ref = NodeRef {
            index: self.nodes.len(),
        };
        self.nodes.push(Node {
            text_id: node_ref.index,
            name,
            ty: ty.clone(),
            payload,
            pos: None,
        });
        self.nodes_by_type.entry(ty).or_default().push(node_ref);
        node_ref
    }

    /// Starts one operation transaction before applicability and entropy
    /// selection so width preparation can be rolled back atomically.
    fn begin_budgeted_operation<S: EntropySource>(
        &mut self,
        source: &S,
        budget: usize,
    ) -> BudgetedOperationState {
        let state = BudgetedOperationState {
            before: self.nodes.len(),
            incomplete_reads: source.incomplete_reads(),
            saved_pool: Some(self.nodes_by_type.clone()),
        };
        self.preparation_remaining = budget.saturating_sub(1);
        state
    }

    /// Ends one operation transaction and discards incomplete finite-entropy
    /// attempts together with any preparation nodes they emitted.
    fn finish_budgeted_operation<S: EntropySource>(
        &mut self,
        source: &S,
        state: BudgetedOperationState,
        budget: usize,
    ) -> usize {
        self.preparation_remaining = 0;
        if source.incomplete_reads() != state.incomplete_reads {
            self.nodes.truncate(state.before);
            self.nodes_by_type = state.saved_pool.unwrap();
            return 0;
        }
        debug_assert!(self.nodes.len() - state.before <= budget);
        self.nodes.len() - state.before
    }

    fn discard_budgeted_operation(&mut self) {
        self.preparation_remaining = 0;
    }

    fn add_random_body_node<S: EntropySource>(
        &mut self,
        source: &mut S,
        budget: usize,
    ) -> Result<usize, GenerationError> {
        let state = self.begin_budgeted_operation(source, budget);
        let applicable = self.applicable_operations();
        if applicable.is_empty() {
            self.discard_budgeted_operation();
            return Err(GenerationError::Construction(
                "no operation can be emitted from the available values".to_string(),
            ));
        }
        let operation = applicable[choose_count(source, applicable.len())];
        self.emit_operation(source, operation)?;
        Ok(self.finish_budgeted_operation(source, state, budget))
    }

    /// Emits one already-selected ordinary operation inside a transaction
    /// started before applicability was computed.
    fn emit_budgeted_operation<S: EntropySource>(
        &mut self,
        source: &mut S,
        operation: RandomOperation,
        state: BudgetedOperationState,
        budget: usize,
    ) -> Result<usize, GenerationError> {
        self.emit_operation(source, operation)?;
        Ok(self.finish_budgeted_operation(source, state, budget))
    }

    fn applicable_operations(&self) -> Vec<RandomOperation> {
        self.options
            .enabled_operations
            .iter()
            .filter(|operation| self.operation_is_applicable(*operation))
            .collect()
    }

    fn operation_is_applicable(&self, operation: RandomOperation) -> bool {
        match operation {
            RandomOperation::Literal => true,
            RandomOperation::Identity => !self.selectable_types().is_empty(),
            RandomOperation::Not
            | RandomOperation::Neg
            | RandomOperation::Reverse
            | RandomOperation::OrReduce
            | RandomOperation::AndReduce
            | RandomOperation::XorReduce
            | RandomOperation::And
            | RandomOperation::Nand
            | RandomOperation::Nor
            | RandomOperation::Or
            | RandomOperation::Xor
            | RandomOperation::Add
            | RandomOperation::Sub
            | RandomOperation::Ugt
            | RandomOperation::Uge
            | RandomOperation::Ult
            | RandomOperation::Ule
            | RandomOperation::Sgt
            | RandomOperation::Sge
            | RandomOperation::Slt
            | RandomOperation::Sle
            | RandomOperation::Shll
            | RandomOperation::Shrl
            | RandomOperation::Shra
            | RandomOperation::BitSlice
            | RandomOperation::DynamicBitSlice
            | RandomOperation::BitSliceUpdate
            | RandomOperation::Decode => !self.bits_types().is_empty(),
            RandomOperation::Umul | RandomOperation::Smul => {
                !self.multiply_bits_types().is_empty()
                    || (self.preparation_remaining > 0 && !self.bits_types().is_empty())
            }
            RandomOperation::Udiv
            | RandomOperation::Sdiv
            | RandomOperation::Umod
            | RandomOperation::Smod => {
                !self.div_mod_bits_types().is_empty()
                    || (self.preparation_remaining > 0 && !self.bits_types().is_empty())
            }
            RandomOperation::Eq | RandomOperation::Ne => !self.selectable_types().is_empty(),
            RandomOperation::Umulp | RandomOperation::Smulp => {
                self.options.allow_tuples
                    && self.options.max_type_depth >= 1
                    && self.options.max_aggregate_leaves >= 2
                    && if self.options.allow_arbitrary_width_multiply {
                        !self.multiply_bits_types().is_empty()
                            || (self.preparation_remaining > 0 && !self.bits_types().is_empty())
                    } else {
                        self.has_mulp_pair()
                            || (self.preparation_remaining > 0
                                && self.mulp_preparation_limit() > 0
                                && !self.bits_types().is_empty())
                    }
            }
            RandomOperation::Gate => {
                self.options.allow_gate
                    && self.nodes_by_type.contains_key(&Type::Bits(1))
                    && !self.selectable_types().is_empty()
            }
            RandomOperation::ZeroExt => !self.all_bits_types().is_empty(),
            RandomOperation::SignExt => !self.bits_types().is_empty(),
            RandomOperation::Concat => {
                self.options.allow_zero_width_bits || !self.bits_types().is_empty()
            }
            RandomOperation::Array => {
                self.options.allow_arrays && !self.array_element_types().is_empty()
            }
            RandomOperation::ArrayIndex => {
                self.options.allow_arrays && !self.array_index_shapes().is_empty()
            }
            RandomOperation::ArrayConcat => {
                self.options.allow_arrays && !self.array_types().is_empty()
            }
            RandomOperation::ArraySlice => {
                self.options.allow_arrays
                    && !self.array_types().is_empty()
                    && !self.bits_types().is_empty()
            }
            RandomOperation::ArrayUpdate => {
                self.options.allow_arrays && !self.array_update_shapes().is_empty()
            }
            RandomOperation::Tuple => self.options.allow_tuples,
            RandomOperation::TupleIndex => {
                self.options.allow_tuples && !self.tuple_types().is_empty()
            }
            RandomOperation::Sel | RandomOperation::PrioritySel | RandomOperation::OneHotSel => {
                !self.select_selector_widths(operation).is_empty()
                    && !self.selectable_types().is_empty()
            }
            RandomOperation::OneHot => self
                .bits_types()
                .iter()
                .any(|width| *width < self.options.max_bit_width),
            RandomOperation::Encode => self.bits_types().iter().any(|width| {
                *width
                    >= if self.options.allow_zero_width_bits {
                        1
                    } else {
                        2
                    }
            }),
            RandomOperation::ExtCarryOut => {
                self.options.allow_extension_ops
                    && self.nodes_by_type.contains_key(&Type::Bits(1))
                    && !self.bits_types().is_empty()
            }
            RandomOperation::ExtPrioEncode
            | RandomOperation::ExtClz
            | RandomOperation::ExtNormalizeLeft
            | RandomOperation::ExtMaskLow
            | RandomOperation::ExtNaryAdd => {
                self.options.allow_extension_ops && !self.bits_types().is_empty()
            }
            RandomOperation::AfterAll => self.options.allow_events,
            RandomOperation::Cover => {
                self.options.allow_events
                    && self.options.allow_tuples
                    && self.nodes_by_type.contains_key(&Type::Bits(1))
            }
            RandomOperation::Assert | RandomOperation::Trace => {
                self.options.allow_events
                    && self.nodes_by_type.contains_key(&Type::Token)
                    && self.nodes_by_type.contains_key(&Type::Bits(1))
            }
            RandomOperation::Invoke | RandomOperation::CountedFor => false,
        }
    }

    fn emit_operation<S: EntropySource>(
        &mut self,
        source: &mut S,
        operation: RandomOperation,
    ) -> Result<NodeRef, GenerationError> {
        match operation {
            RandomOperation::Literal => {
                let ty = random_type(source, self.options, 0);
                Ok(self.add_node(
                    ty.clone(),
                    NodePayload::Literal(generate_uniform_value(source, &ty)),
                    None,
                ))
            }
            RandomOperation::Identity => {
                let ty = self.choose_selectable_type(source);
                let arg = self.choose_ref_for_type(source, &ty);
                Ok(self.add_node(ty, NodePayload::Unop(Unop::Identity, arg), None))
            }
            RandomOperation::Not
            | RandomOperation::Neg
            | RandomOperation::Reverse
            | RandomOperation::OrReduce
            | RandomOperation::AndReduce
            | RandomOperation::XorReduce => {
                let (ty, arg) = self.choose_data_bits_ref(source, self.options.max_bit_width);
                let op = match operation {
                    RandomOperation::Not => Unop::Not,
                    RandomOperation::Neg => Unop::Neg,
                    RandomOperation::Reverse => Unop::Reverse,
                    RandomOperation::OrReduce => Unop::OrReduce,
                    RandomOperation::AndReduce => Unop::AndReduce,
                    RandomOperation::XorReduce => Unop::XorReduce,
                    _ => unreachable!("matched unary operation"),
                };
                let result_ty = match operation {
                    RandomOperation::OrReduce
                    | RandomOperation::AndReduce
                    | RandomOperation::XorReduce => Type::Bits(1),
                    _ => ty,
                };
                Ok(self.add_node(result_ty, NodePayload::Unop(op, arg), None))
            }
            RandomOperation::And
            | RandomOperation::Nand
            | RandomOperation::Nor
            | RandomOperation::Or
            | RandomOperation::Xor => {
                let (ty, _) = self.choose_data_bits_ref(source, self.options.max_bit_width);
                let arg_count = choose_between(source, 1, self.options.max_nary_operands);
                let args = (0..arg_count)
                    .map(|_| self.choose_ref_for_type(source, &ty))
                    .collect();
                let op = match operation {
                    RandomOperation::And => NaryOp::And,
                    RandomOperation::Nand => NaryOp::Nand,
                    RandomOperation::Nor => NaryOp::Nor,
                    RandomOperation::Or => NaryOp::Or,
                    RandomOperation::Xor => NaryOp::Xor,
                    _ => unreachable!("matched nary bitwise operation"),
                };
                Ok(self.add_node(ty, NodePayload::Nary(op, args), None))
            }
            RandomOperation::Add
            | RandomOperation::Sub
            | RandomOperation::Udiv
            | RandomOperation::Sdiv
            | RandomOperation::Umod
            | RandomOperation::Smod
            | RandomOperation::Ugt
            | RandomOperation::Uge
            | RandomOperation::Ult
            | RandomOperation::Ule
            | RandomOperation::Sgt
            | RandomOperation::Sge
            | RandomOperation::Slt
            | RandomOperation::Sle => {
                let (operand_ty, lhs, rhs) = if matches!(
                    operation,
                    RandomOperation::Udiv
                        | RandomOperation::Sdiv
                        | RandomOperation::Umod
                        | RandomOperation::Smod
                ) {
                    let (ty, lhs) = self.choose_data_bits_ref(
                        source,
                        self.options
                            .max_div_mod_bit_width
                            .unwrap_or(self.options.max_bit_width)
                            .min(self.options.max_bit_width),
                    );
                    let rhs = self.choose_matching_data_ref(source, &ty);
                    (ty, lhs, rhs)
                } else {
                    let (ty, lhs) = self.choose_data_bits_ref(source, self.options.max_bit_width);
                    let rhs = self.choose_matching_data_ref(source, &ty);
                    (ty, lhs, rhs)
                };
                let op = match operation {
                    RandomOperation::Add => Binop::Add,
                    RandomOperation::Sub => Binop::Sub,
                    RandomOperation::Udiv => Binop::Udiv,
                    RandomOperation::Sdiv => Binop::Sdiv,
                    RandomOperation::Umod => Binop::Umod,
                    RandomOperation::Smod => Binop::Smod,
                    RandomOperation::Ugt => Binop::Ugt,
                    RandomOperation::Uge => Binop::Uge,
                    RandomOperation::Ult => Binop::Ult,
                    RandomOperation::Ule => Binop::Ule,
                    RandomOperation::Sgt => Binop::Sgt,
                    RandomOperation::Sge => Binop::Sge,
                    RandomOperation::Slt => Binop::Slt,
                    RandomOperation::Sle => Binop::Sle,
                    _ => unreachable!("matched binary operation"),
                };
                let result_ty = match operation {
                    RandomOperation::Ugt
                    | RandomOperation::Uge
                    | RandomOperation::Ult
                    | RandomOperation::Ule
                    | RandomOperation::Sgt
                    | RandomOperation::Sge
                    | RandomOperation::Slt
                    | RandomOperation::Sle => Type::Bits(1),
                    _ => operand_ty,
                };
                Ok(self.add_node(result_ty, NodePayload::Binop(op, lhs, rhs), None))
            }
            RandomOperation::Umul | RandomOperation::Smul => {
                let limit = self
                    .options
                    .max_multiply_operand_bit_width
                    .unwrap_or(self.options.max_bit_width)
                    .min(self.options.max_bit_width);
                let (operand_ty, lhs) = self.choose_data_bits_ref(source, limit);
                let rhs = if self.options.allow_arbitrary_width_multiply
                    && choose_count(source, 2) != 0
                {
                    self.choose_data_bits_ref(source, limit).1
                } else {
                    self.choose_matching_data_ref(source, &operand_ty)
                };
                let result_ty = if self.options.allow_arbitrary_width_multiply {
                    let full =
                        self.get_node(lhs).ty.bit_count() + self.get_node(rhs).ty.bit_count();
                    let maximum = self.options.max_bit_width;
                    let categories = 1 + usize::from(full <= maximum) + usize::from(full < maximum);
                    let width = match choose_count(source, categories) {
                        0 => balanced_width(source, (full - 1).min(maximum)),
                        1 => full,
                        _ => choose_between(source, full + 1, maximum),
                    };
                    Type::Bits(width)
                } else {
                    operand_ty
                };
                let op = if operation == RandomOperation::Umul {
                    Binop::Umul
                } else {
                    Binop::Smul
                };
                Ok(self.add_node(result_ty, NodePayload::Binop(op, lhs, rhs), None))
            }
            RandomOperation::Eq | RandomOperation::Ne => {
                let ty = self.choose_selectable_type(source);
                let lhs = self.choose_ref_for_type(source, &ty);
                let rhs = self.choose_ref_for_type(source, &ty);
                let op = if operation == RandomOperation::Eq {
                    Binop::Eq
                } else {
                    Binop::Ne
                };
                Ok(self.add_node(Type::Bits(1), NodePayload::Binop(op, lhs, rhs), None))
            }
            RandomOperation::Umulp | RandomOperation::Smulp => {
                let (lhs, rhs, result_width) = if self.options.allow_arbitrary_width_multiply {
                    let limit = self
                        .options
                        .max_multiply_operand_bit_width
                        .unwrap_or(self.options.max_bit_width)
                        .min(self.options.max_bit_width);
                    let (_, lhs) = self.choose_data_bits_ref(source, limit);
                    let (_, rhs) = self.choose_data_bits_ref(source, limit);
                    (lhs, rhs, random_width(source, self.options.max_bit_width))
                } else if self.has_mulp_pair() {
                    let (lhs_width, lhs, rhs_width, rhs) = self.choose_mulp_pair(source);
                    (lhs, rhs, lhs_width + rhs_width)
                } else {
                    let (operand_ty, lhs) =
                        self.choose_data_bits_ref(source, self.mulp_preparation_limit());
                    let Type::Bits(width) = operand_ty else {
                        unreachable!("partial-product operand is bits")
                    };
                    let rhs = self.choose_matching_data_ref(source, &Type::Bits(width));
                    (lhs, rhs, width + width)
                };
                let op = if operation == RandomOperation::Umulp {
                    Binop::Umulp
                } else {
                    Binop::Smulp
                };
                Ok(self.add_node(
                    Type::Tuple(vec![
                        Box::new(Type::Bits(result_width)),
                        Box::new(Type::Bits(result_width)),
                    ]),
                    NodePayload::Binop(op, lhs, rhs),
                    None,
                ))
            }
            RandomOperation::Shll | RandomOperation::Shrl | RandomOperation::Shra => {
                let (ty, lhs) = self.choose_data_bits_ref(source, self.options.max_bit_width);
                let (_, rhs) = self.choose_data_bits_ref(source, self.options.max_bit_width);
                let op = match operation {
                    RandomOperation::Shll => Binop::Shll,
                    RandomOperation::Shrl => Binop::Shrl,
                    RandomOperation::Shra => Binop::Shra,
                    _ => unreachable!("matched shift operation"),
                };
                Ok(self.add_node(ty, NodePayload::Binop(op, lhs, rhs), None))
            }
            RandomOperation::Gate => {
                let predicate = self.choose_ref_for_type(source, &Type::Bits(1));
                let ty = self.choose_selectable_type(source);
                let value = self.choose_ref_for_type(source, &ty);
                Ok(self.add_node(ty, NodePayload::Binop(Binop::Gate, predicate, value), None))
            }
            RandomOperation::ZeroExt => {
                let (old_width, arg) = self.choose_any_bits_width_ref(source);
                let width = choose_between(source, old_width, self.options.max_bit_width);
                Ok(self.add_node(
                    Type::Bits(width),
                    NodePayload::ZeroExt {
                        arg,
                        new_bit_count: width,
                    },
                    None,
                ))
            }
            RandomOperation::SignExt => {
                let (old_width, arg) = self.choose_bits_width_ref(source);
                let width = choose_between(source, old_width, self.options.max_bit_width);
                Ok(self.add_node(
                    Type::Bits(width),
                    NodePayload::SignExt {
                        arg,
                        new_bit_count: width,
                    },
                    None,
                ))
            }
            RandomOperation::BitSlice => {
                let (ty, arg) = self.choose_bits_ref(source);
                let Type::Bits(arg_width) = ty else {
                    unreachable!("chosen bits reference has a bits type")
                };
                let generate_zero_width =
                    self.options.allow_zero_width_bits && source.take_u64() & 3 == 0;
                let (start, width) = if generate_zero_width {
                    (choose_between(source, 0, arg_width), 0)
                } else {
                    let start = choose_count(source, arg_width);
                    (start, choose_between(source, 1, arg_width - start))
                };
                Ok(self.add_node(
                    Type::Bits(width),
                    NodePayload::BitSlice { arg, start, width },
                    None,
                ))
            }
            RandomOperation::DynamicBitSlice => {
                let (ty, arg) = self.choose_bits_ref(source);
                let (_, start) = self.choose_bits_ref(source);
                let Type::Bits(arg_width) = ty else {
                    unreachable!("chosen dynamic slice argument has a bits type")
                };
                let width = if self.options.allow_zero_width_bits && source.take_u64() & 3 == 0 {
                    0
                } else {
                    random_width(source, arg_width)
                };
                Ok(self.add_node(
                    Type::Bits(width),
                    NodePayload::DynamicBitSlice { arg, start, width },
                    None,
                ))
            }
            RandomOperation::BitSliceUpdate => {
                let (ty, arg) = self.choose_bits_ref(source);
                let (_, start) = self.choose_bits_ref(source);
                let (_, update_value) = self.choose_bits_ref(source);
                Ok(self.add_node(
                    ty,
                    NodePayload::BitSliceUpdate {
                        arg,
                        start,
                        update_value,
                    },
                    None,
                ))
            }
            RandomOperation::Concat => {
                let (width, args) = self.choose_concat_operands(source);
                Ok(self.add_node(
                    Type::Bits(width),
                    NodePayload::Nary(NaryOp::Concat, args),
                    None,
                ))
            }
            RandomOperation::Array => {
                let element_ty = self.choose_array_element_type(source);
                let max_length = self.max_array_length_for_element(&element_ty);
                let length = choose_between(source, 1, max_length);
                let elements = (0..length)
                    .map(|_| self.choose_ref_for_type(source, &element_ty))
                    .collect();
                Ok(self.add_node(
                    Type::new_array(element_ty, length),
                    NodePayload::Array(elements),
                    None,
                ))
            }
            RandomOperation::ArrayIndex => {
                let (array_ty, index_count, result_ty) = self.choose_array_index_shape(source);
                let array_ref = self.choose_ref_for_type(source, &array_ty);
                let (indices, assumed_in_bounds) =
                    self.choose_array_indices(source, &array_ty, index_count);
                Ok(self.add_node(
                    result_ty,
                    NodePayload::ArrayIndex {
                        array: array_ref,
                        indices,
                        assumed_in_bounds,
                    },
                    None,
                ))
            }
            RandomOperation::ArrayConcat => {
                let (result_ty, operands) = self.choose_array_concat_operands(source);
                Ok(self.add_node(result_ty, NodePayload::ArrayConcat(operands), None))
            }
            RandomOperation::ArraySlice => {
                let array_ty = self.choose_array_type(source);
                let array = self.choose_ref_for_type(source, &array_ty);
                let (_, start) = self.choose_bits_ref(source);
                let Type::Array(array_data) = array_ty else {
                    unreachable!("selected array-slice operand has array type")
                };
                let width = choose_between(
                    source,
                    1,
                    self.max_array_length_for_element(&array_data.element_type),
                );
                Ok(self.add_node(
                    Type::new_array((*array_data.element_type).clone(), width),
                    NodePayload::ArraySlice {
                        array,
                        start,
                        width,
                    },
                    None,
                ))
            }
            RandomOperation::ArrayUpdate => {
                let (array_ty, index_count, update_ty) = self.choose_array_update_shape(source);
                let array = self.choose_ref_for_type(source, &array_ty);
                let value = self.choose_ref_for_type(source, &update_ty);
                let (indices, assumed_in_bounds) =
                    self.choose_array_indices(source, &array_ty, index_count);
                Ok(self.add_node(
                    array_ty,
                    NodePayload::ArrayUpdate {
                        array,
                        value,
                        indices,
                        assumed_in_bounds,
                    },
                    None,
                ))
            }
            RandomOperation::Tuple => {
                let max_length = self.options.max_tuple_length;
                let desired_length = choose_between(source, 0, max_length);
                let candidates = self.tuple_field_refs();
                let mut elements = Vec::new();
                let mut fields = Vec::new();
                for _ in 0..desired_length {
                    if candidates.is_empty() {
                        break;
                    }
                    let (ty, node_ref) = &candidates[choose_count(source, candidates.len())];
                    let mut proposed = fields.clone();
                    proposed.push(Box::new(ty.clone()));
                    let proposed_ty = Type::Tuple(proposed);
                    if type_depth(&proposed_ty) <= self.options.max_type_depth
                        && type_leaf_count(&proposed_ty) <= self.options.max_aggregate_leaves
                    {
                        fields.push(Box::new(ty.clone()));
                        elements.push(*node_ref);
                    }
                }
                Ok(self.add_node(Type::Tuple(fields), NodePayload::Tuple(elements), None))
            }
            RandomOperation::TupleIndex => {
                let tuple_ty = self.choose_tuple_type(source);
                let tuple_ref = self.choose_ref_for_type(source, &tuple_ty);
                let Type::Tuple(fields) = tuple_ty else {
                    unreachable!("chosen tuple reference has tuple type")
                };
                let index = choose_count(source, fields.len());
                Ok(self.add_node(
                    (*fields[index]).clone(),
                    NodePayload::TupleIndex {
                        tuple: tuple_ref,
                        index,
                    },
                    None,
                ))
            }
            RandomOperation::Sel | RandomOperation::PrioritySel | RandomOperation::OneHotSel => {
                let result_ty = self.choose_selectable_type(source);
                let (selector_width, selector) = self.choose_select_selector(source, operation);
                let complete_case_count = u32::try_from(selector_width)
                    .ok()
                    .and_then(|width| 1usize.checked_shl(width));
                let case_count = match operation {
                    RandomOperation::Sel => {
                        let maximum = complete_case_count
                            .unwrap_or(MAX_RANDOM_SEL_CASES)
                            .min(MAX_RANDOM_SEL_CASES);
                        let choice = source.take_u64();
                        if self.options.allow_empty_case_sel && choice % 3 == 2 {
                            0
                        } else if choice & 1 == 0 && complete_case_count == Some(maximum) {
                            maximum
                        } else {
                            choose_between(source, 1, maximum)
                        }
                    }
                    RandomOperation::PrioritySel | RandomOperation::OneHotSel => selector_width,
                    _ => unreachable!("matched selection operation"),
                };
                let cases: Vec<NodeRef> = (0..case_count)
                    .map(|_| self.choose_ref_for_type(source, &result_ty))
                    .collect();
                let payload = match operation {
                    RandomOperation::Sel => NodePayload::Sel {
                        selector,
                        cases,
                        default: (complete_case_count != Some(case_count))
                            .then(|| self.choose_ref_for_type(source, &result_ty)),
                    },
                    RandomOperation::PrioritySel => NodePayload::PrioritySel {
                        selector,
                        cases,
                        default: Some(self.choose_ref_for_type(source, &result_ty)),
                    },
                    RandomOperation::OneHotSel => NodePayload::OneHotSel { selector, cases },
                    _ => unreachable!("matched selection operation"),
                };
                Ok(self.add_node(result_ty, payload, None))
            }
            RandomOperation::OneHot => {
                let widths: Vec<usize> = self
                    .bits_types()
                    .into_iter()
                    .filter(|width| *width < self.options.max_bit_width)
                    .collect();
                let width = widths[choose_count(source, widths.len())];
                let arg = self.choose_ref_for_type(source, &Type::Bits(width));
                Ok(self.add_node(
                    Type::Bits(width + 1),
                    NodePayload::OneHot {
                        arg,
                        lsb_prio: source.take_u64() & 1 != 0,
                    },
                    None,
                ))
            }
            RandomOperation::Encode => {
                let widths: Vec<usize> = self
                    .bits_types()
                    .into_iter()
                    .filter(|width| {
                        *width
                            >= if self.options.allow_zero_width_bits {
                                1
                            } else {
                                2
                            }
                    })
                    .collect();
                let width = widths[choose_count(source, widths.len())];
                let arg = self.choose_ref_for_type(source, &Type::Bits(width));
                Ok(self.add_node(
                    Type::Bits(ceil_log2(width)),
                    NodePayload::Encode { arg },
                    None,
                ))
            }
            RandomOperation::Decode => {
                let (ty, arg) = self.choose_bits_ref(source);
                let Type::Bits(arg_width) = ty else {
                    unreachable!("chosen decode argument has a bits type")
                };
                let max_decode_width = if arg_width >= usize::BITS as usize {
                    self.options.max_bit_width
                } else {
                    self.options.max_bit_width.min(1usize << arg_width)
                };
                let width = if self.options.allow_zero_width_bits && source.take_u64() & 3 == 0 {
                    0
                } else {
                    random_width(source, max_decode_width)
                };
                Ok(self.add_node(Type::Bits(width), NodePayload::Decode { arg, width }, None))
            }
            RandomOperation::ExtCarryOut => {
                let (_, lhs, rhs) = self.choose_same_bits_refs(source);
                let c_in = self.choose_ref_for_type(source, &Type::Bits(1));
                Ok(self.add_node(
                    Type::Bits(1),
                    NodePayload::ExtCarryOut { lhs, rhs, c_in },
                    None,
                ))
            }
            RandomOperation::ExtPrioEncode => {
                let (ty, arg) = self.choose_bits_ref(source);
                let Type::Bits(width) = ty else {
                    unreachable!("selected priority-encode operand has bits type")
                };
                Ok(self.add_node(
                    Type::Bits(ceil_log2(width.saturating_add(1))),
                    NodePayload::ExtPrioEncode {
                        arg,
                        lsb_prio: source.take_u64() & 1 != 0,
                    },
                    None,
                ))
            }
            RandomOperation::ExtClz => {
                let (_, arg) = self.choose_bits_ref(source);
                let width = random_width(source, self.options.max_bit_width);
                Ok(self.add_node(
                    Type::Bits(width),
                    NodePayload::ExtClz {
                        arg,
                        offset: choose_count(source, self.options.max_bit_width.saturating_add(1)),
                        new_bit_count: width,
                    },
                    None,
                ))
            }
            RandomOperation::ExtNormalizeLeft => {
                let (arg_ty, arg) = self.choose_bits_ref(source);
                let Type::Bits(arg_width) = arg_ty else {
                    unreachable!("selected normalize-left operand has bits type")
                };
                let normalized_width =
                    random_width(source, self.options.max_bit_width).max(arg_width);
                let has_clz_result = self.options.allow_tuples
                    && self.options.max_type_depth >= 1
                    && self.options.max_aggregate_leaves >= 2
                    && source.take_u64() & 1 != 0;
                let clz_bit_count =
                    has_clz_result.then(|| random_width(source, self.options.max_bit_width));
                let result_ty = if let Some(clz_width) = clz_bit_count {
                    Type::Tuple(vec![
                        Box::new(Type::Bits(normalized_width)),
                        Box::new(Type::Bits(clz_width)),
                    ])
                } else {
                    Type::Bits(normalized_width)
                };
                Ok(self.add_node(
                    result_ty,
                    NodePayload::ExtNormalizeLeft {
                        arg,
                        shift_offset: choose_count(source, normalized_width.saturating_add(1)),
                        normalized_bit_count: normalized_width,
                        clz_bit_count,
                    },
                    None,
                ))
            }
            RandomOperation::ExtMaskLow => {
                let (_, count) = self.choose_bits_ref(source);
                let width = random_width(source, self.options.max_bit_width);
                Ok(self.add_node(Type::Bits(width), NodePayload::ExtMaskLow { count }, None))
            }
            RandomOperation::ExtNaryAdd => {
                let term_count = choose_between(source, 0, self.options.max_nary_operands);
                let terms = (0..term_count)
                    .map(|_| {
                        let (_, operand) =
                            self.choose_data_bits_ref(source, self.options.max_bit_width);
                        ExtNaryAddTerm {
                            operand,
                            signed: source.take_u64() & 1 != 0,
                            negated: source.take_u64() & 1 != 0,
                        }
                    })
                    .collect();
                let arch = match choose_count(source, 4) {
                    0 => None,
                    1 => Some(ExtNaryAddArchitecture::RippleCarry),
                    2 => Some(ExtNaryAddArchitecture::KoggeStone),
                    _ => Some(ExtNaryAddArchitecture::BrentKung),
                };
                let width = random_width(source, self.options.max_bit_width);
                Ok(self.add_node(
                    Type::Bits(width),
                    NodePayload::ExtNaryAdd { terms, arch },
                    None,
                ))
            }
            RandomOperation::AfterAll => {
                let token_refs = self.token_refs();
                let operand_count = choose_count(
                    source,
                    self.options.max_nary_operands.min(token_refs.len()) + 1,
                );
                let operands = (0..operand_count)
                    .map(|_| token_refs[choose_count(source, token_refs.len())])
                    .collect();
                Ok(self.add_node(Type::Token, NodePayload::AfterAll(operands), None))
            }
            RandomOperation::Cover => {
                let predicate = self.choose_ref_for_type(source, &Type::Bits(1));
                Ok(self.add_node(
                    Type::nil(),
                    NodePayload::Cover {
                        predicate,
                        label: format!("random_cover_{}", self.nodes.len()),
                    },
                    None,
                ))
            }
            RandomOperation::Assert => {
                let token = self.choose_token_ref(source);
                let activate = self.choose_ref_for_type(source, &Type::Bits(1));
                let site = self.nodes.len();
                Ok(self.add_node(
                    Type::Token,
                    NodePayload::Assert {
                        token,
                        activate,
                        message: format!("random assertion at site {site}"),
                        label: format!("random_assert_{site}"),
                    },
                    None,
                ))
            }
            RandomOperation::Trace => {
                let token = self.choose_token_ref(source);
                let activated = self.choose_ref_for_type(source, &Type::Bits(1));
                let operand_count = choose_between(source, 0, 3);
                let operands = (0..operand_count)
                    .map(|_| {
                        let operand_ty = self.choose_selectable_type(source);
                        self.choose_ref_for_type(source, &operand_ty)
                    })
                    .collect();
                let mut format = if choose_count(source, 4) == 0 {
                    "random_trace={{".to_string()
                } else {
                    "random_trace".to_string()
                };
                if operand_count != 0 {
                    let specifiers = (0..operand_count)
                        .map(|_| {
                            TRACE_FORMAT_SPECIFIERS
                                [choose_count(source, TRACE_FORMAT_SPECIFIERS.len())]
                        })
                        .collect::<Vec<_>>();
                    format.push('=');
                    format.push_str(&specifiers.join(","));
                }
                Ok(self.add_node(
                    Type::Token,
                    NodePayload::Trace {
                        token,
                        activated,
                        format,
                        verbosity: 0,
                        operands,
                    },
                    None,
                ))
            }
            RandomOperation::Invoke | RandomOperation::CountedFor => {
                unreachable!("package generation emits calls with package-level call-graph state")
            }
        }
    }

    fn div_mod_bits_types(&self) -> Vec<usize> {
        self.bits_types()
            .into_iter()
            .filter(|width| {
                self.options
                    .max_div_mod_bit_width
                    .is_none_or(|limit| *width <= limit)
            })
            .collect()
    }

    /// Reuses an exact-width value, or adapts an existing computation with one
    /// budgeted slice/extension. Wider sources retain full-width input entropy.
    fn choose_data_bits_ref<S: EntropySource>(
        &mut self,
        source: &mut S,
        maximum: usize,
    ) -> (Type, NodeRef) {
        let widths: Vec<_> = self
            .bits_types()
            .into_iter()
            .filter(|w| *w <= maximum)
            .collect();
        if self.preparation_remaining == 0 {
            let ty = Type::Bits(widths[choose_count(source, widths.len())]);
            return (ty.clone(), self.choose_ref_for_type(source, &ty));
        }
        let width = balanced_width(source, maximum);
        let ty = Type::Bits(width);
        if self.nodes_by_type.contains_key(&ty) {
            return (ty.clone(), self.choose_ref_for_type(source, &ty));
        }
        self.prepare_data_width(source, width)
    }

    /// Avoids always reusing a lone newly adapted operand on both input pins.
    fn choose_matching_data_ref<S: EntropySource>(&mut self, source: &mut S, ty: &Type) -> NodeRef {
        let Type::Bits(width) = ty else {
            unreachable!("data operand is bits")
        };
        if self.preparation_remaining > 0
            && self.nodes_by_type[ty].len() == 1
            && choose_count(source, 2) == 0
            && self.bits_types().iter().any(|w| w != width)
        {
            self.prepare_data_width(source, *width).1
        } else {
            self.choose_ref_for_type(source, ty)
        }
    }

    fn prepare_data_width<S: EntropySource>(
        &mut self,
        source: &mut S,
        width: usize,
    ) -> (Type, NodeRef) {
        let ty = Type::Bits(width);
        let all: Vec<_> = self
            .bits_types()
            .into_iter()
            .filter(|w| *w != width)
            .collect();
        let wider: Vec<_> = all.iter().copied().filter(|w| *w > width).collect();
        let choices = if wider.is_empty() { &all } else { &wider };
        let source_width = choices[choose_count(source, choices.len())];
        let arg = self.choose_ref_for_type(source, &Type::Bits(source_width));
        let payload = if source_width > width {
            NodePayload::BitSlice {
                arg,
                start: choose_between(source, 0, source_width - width),
                width,
            }
        } else if choose_count(source, 2) == 0 {
            NodePayload::ZeroExt {
                arg,
                new_bit_count: width,
            }
        } else {
            NodePayload::SignExt {
                arg,
                new_bit_count: width,
            }
        };
        self.preparation_remaining -= 1;
        let node = self.add_node(ty.clone(), payload, None);
        (ty, node)
    }

    fn multiply_bits_types(&self) -> Vec<usize> {
        self.bits_types()
            .into_iter()
            .filter(|width| {
                self.options
                    .max_multiply_operand_bit_width
                    .is_none_or(|limit| *width <= limit)
            })
            .collect()
    }

    /// Caps a prepared equal-width partial-product pair so its full product
    /// still fits within the configured result width.
    fn mulp_preparation_limit(&self) -> usize {
        self.options
            .max_multiply_operand_bit_width
            .unwrap_or(self.options.max_bit_width)
            .min(self.options.max_bit_width / 2)
    }

    fn bits_types(&self) -> Vec<usize> {
        self.nodes_by_type
            .keys()
            .filter_map(|ty| match ty {
                Type::Bits(width) if *width > 0 => Some(*width),
                _ => None,
            })
            .collect()
    }

    fn all_bits_types(&self) -> Vec<usize> {
        self.nodes_by_type
            .keys()
            .filter_map(|ty| match ty {
                Type::Bits(width) => Some(*width),
                _ => None,
            })
            .collect()
    }

    fn choose_bits_ref<S: EntropySource>(&self, source: &mut S) -> (Type, NodeRef) {
        let widths = self.bits_types();
        let width = widths[choose_count(source, widths.len())];
        let ty = Type::Bits(width);
        let node_ref = self.choose_ref_for_type(source, &ty);
        (ty, node_ref)
    }

    fn select_selector_widths(&self, operation: RandomOperation) -> Vec<usize> {
        self.bits_types()
            .into_iter()
            .filter(|width| {
                operation == RandomOperation::Sel
                    || *width
                        <= self
                            .options
                            .max_bit_width
                            .min(MAX_RANDOM_PRIORITY_SEL_CASES)
            })
            .collect()
    }

    fn choose_select_selector<S: EntropySource>(
        &self,
        source: &mut S,
        operation: RandomOperation,
    ) -> (usize, NodeRef) {
        let widths = self.select_selector_widths(operation);
        let width = widths[choose_count(source, widths.len())];
        (width, self.choose_ref_for_type(source, &Type::Bits(width)))
    }

    fn choose_same_bits_refs<S: EntropySource>(&self, source: &mut S) -> (Type, NodeRef, NodeRef) {
        let (ty, lhs) = self.choose_bits_ref(source);
        let rhs = self.choose_ref_for_type(source, &ty);
        (ty, lhs, rhs)
    }

    fn choose_bits_width_ref<S: EntropySource>(&self, source: &mut S) -> (usize, NodeRef) {
        let (ty, node_ref) = self.choose_bits_ref(source);
        let Type::Bits(width) = ty else {
            unreachable!("selected bits reference has bits type")
        };
        (width, node_ref)
    }

    fn choose_any_bits_width_ref<S: EntropySource>(&self, source: &mut S) -> (usize, NodeRef) {
        let widths = self.all_bits_types();
        let width = widths[choose_count(source, widths.len())];
        (width, self.choose_ref_for_type(source, &Type::Bits(width)))
    }

    fn has_mulp_pair(&self) -> bool {
        let widths = self.multiply_bits_types();
        self.options.max_type_depth >= 1
            && self.options.max_aggregate_leaves >= 2
            && widths.iter().any(|lhs| {
                widths
                    .iter()
                    .any(|rhs| lhs + rhs <= self.options.max_bit_width)
            })
    }

    fn choose_mulp_pair<S: EntropySource>(
        &self,
        source: &mut S,
    ) -> (usize, NodeRef, usize, NodeRef) {
        let widths = self.multiply_bits_types();
        let pairs: Vec<_> = widths
            .iter()
            .flat_map(|&lhs| {
                widths
                    .iter()
                    .copied()
                    .filter(move |rhs| lhs + rhs <= self.options.max_bit_width)
                    .map(move |rhs| (lhs, rhs))
            })
            .collect();
        let (lhs_width, rhs_width) = pairs[choose_count(source, pairs.len())];
        (
            lhs_width,
            self.choose_ref_for_type(source, &Type::Bits(lhs_width)),
            rhs_width,
            self.choose_ref_for_type(source, &Type::Bits(rhs_width)),
        )
    }

    fn choose_ref_for_type<S: EntropySource>(&self, source: &mut S, ty: &Type) -> NodeRef {
        let refs = self
            .nodes_by_type
            .get(ty)
            .expect("selected generated type has values");
        let choice = source.take_u64();
        let window = refs.len().min(4);
        let index = match choice % 4 {
            0 => (choice / 4 % refs.len() as u64) as usize,
            1 => (choice / 4 % window as u64) as usize,
            _ => refs.len() - window + (choice / 4 % window as u64) as usize,
        };
        refs[index]
    }

    fn choose_any_ref<S: EntropySource>(&self, source: &mut S) -> NodeRef {
        let types: Vec<&Type> = self.nodes_by_type.keys().collect();
        let ty = types[choose_count(source, types.len())];
        self.choose_ref_for_type(source, ty)
    }

    fn get_node(&self, node_ref: NodeRef) -> &Node {
        &self.nodes[node_ref.index]
    }

    fn token_refs(&self) -> Vec<NodeRef> {
        self.nodes_by_type
            .get(&Type::Token)
            .cloned()
            .unwrap_or_default()
    }

    fn choose_token_ref<S: EntropySource>(&self, source: &mut S) -> NodeRef {
        self.choose_ref_for_type(source, &Type::Token)
    }

    fn choose_concat_operands<S: EntropySource>(&self, source: &mut S) -> (usize, Vec<NodeRef>) {
        let available_widths = if self.options.allow_zero_width_bits {
            self.all_bits_types()
        } else {
            self.bits_types()
        };
        if self.options.allow_zero_width_bits
            && (available_widths.is_empty() || source.take_u64() & 3 == 0)
        {
            return (0, Vec::new());
        }

        let desired_count = choose_between(source, 1, self.options.max_nary_operands);
        let mut total_width = 0;
        let mut operands = Vec::new();
        for _ in 0..desired_count {
            let eligible_widths: Vec<usize> = available_widths
                .iter()
                .copied()
                .filter(|width| total_width + width <= self.options.max_bit_width)
                .collect();
            if eligible_widths.is_empty() {
                break;
            }
            let width = eligible_widths[choose_count(source, eligible_widths.len())];
            operands.push(self.choose_ref_for_type(source, &Type::Bits(width)));
            total_width += width;
        }
        debug_assert!(!operands.is_empty());
        (total_width, operands)
    }

    fn array_element_types(&self) -> Vec<Type> {
        self.nodes_by_type
            .keys()
            .filter(|ty| {
                !type_contains_token(ty)
                    && type_depth(ty) < self.options.max_type_depth
                    && type_leaf_count(ty) <= self.options.max_aggregate_leaves
            })
            .cloned()
            .collect()
    }

    fn choose_array_element_type<S: EntropySource>(&self, source: &mut S) -> Type {
        let types = self.array_element_types();
        types[choose_count(source, types.len())].clone()
    }

    fn array_types(&self) -> Vec<Type> {
        self.nodes_by_type
            .keys()
            .filter(|ty| matches!(ty, Type::Array(_)) && !type_contains_token(ty))
            .cloned()
            .collect()
    }

    fn choose_array_type<S: EntropySource>(&self, source: &mut S) -> Type {
        let types = self.array_types();
        types[choose_count(source, types.len())].clone()
    }

    fn max_array_length_for_element(&self, element_ty: &Type) -> usize {
        max_array_length_for_element(self.options, element_ty)
    }

    fn choose_array_concat_operands<S: EntropySource>(
        &self,
        source: &mut S,
    ) -> (Type, Vec<NodeRef>) {
        let first_ty = self.choose_array_type(source);
        let Type::Array(first) = &first_ty else {
            unreachable!("selected array concat operand is an array")
        };
        let element_ty = (*first.element_type).clone();
        let max_length = self.max_array_length_for_element(&element_ty);
        let desired_count = choose_between(source, 1, self.options.max_nary_operands);
        let mut result_count = first.element_count;
        let mut operands = vec![self.choose_ref_for_type(source, &first_ty)];
        for _ in 1..desired_count {
            let candidates: Vec<Type> = self
                .array_types()
                .into_iter()
                .filter(|candidate| {
                    let Type::Array(array) = candidate else {
                        unreachable!("array type list contains arrays")
                    };
                    array.element_type.as_ref() == &element_ty
                        && result_count + array.element_count <= max_length
                })
                .collect();
            if candidates.is_empty() {
                break;
            }
            let candidate = &candidates[choose_count(source, candidates.len())];
            let Type::Array(array) = candidate else {
                unreachable!("array concat candidate is an array")
            };
            result_count += array.element_count;
            operands.push(self.choose_ref_for_type(source, candidate));
        }
        (Type::new_array(element_ty, result_count), operands)
    }

    fn indexed_type(ty: &Type, index_count: usize) -> Option<Type> {
        let mut result = ty.clone();
        for _ in 0..index_count {
            let Type::Array(array) = result else {
                return None;
            };
            result = *array.element_type;
        }
        Some(result)
    }

    fn array_index_shapes(&self) -> Vec<(Type, usize, Type)> {
        let has_index = !self.bits_types().is_empty();
        self.array_types()
            .into_iter()
            .flat_map(|array_ty| {
                let mut shapes = vec![(array_ty.clone(), 0, array_ty.clone())];
                if has_index {
                    let mut index_count = 1;
                    while let Some(result_ty) = Self::indexed_type(&array_ty, index_count) {
                        shapes.push((array_ty.clone(), index_count, result_ty.clone()));
                        if !matches!(result_ty, Type::Array(_)) {
                            break;
                        }
                        index_count += 1;
                    }
                }
                shapes
            })
            .collect()
    }

    fn choose_array_index_shape<S: EntropySource>(&self, source: &mut S) -> (Type, usize, Type) {
        let shapes = self.array_index_shapes();
        if choose_count(source, 10) != 0 {
            let indexed: Vec<_> = shapes.iter().filter(|(_, depth, _)| *depth != 0).collect();
            if !indexed.is_empty() {
                return indexed[choose_count(source, indexed.len())].clone();
            }
        }
        shapes[choose_count(source, shapes.len())].clone()
    }

    /// Chooses independent indices, optionally restricting promises to
    /// width-safe or literal-safe nodes without inserting extra body logic.
    fn choose_array_indices<S: EntropySource>(
        &self,
        source: &mut S,
        array_ty: &Type,
        index_count: usize,
    ) -> (Vec<NodeRef>, bool) {
        let require_safe = self.options.array_assumption_mode == ArrayAssumptionMode::ProvenSafe;
        let mut assumed = self.options.array_assumption_mode != ArrayAssumptionMode::Disabled
            && source.take_u64() & 1 != 0;
        let mut ty = array_ty;
        let mut indices = Vec::with_capacity(index_count);
        for _ in 0..index_count {
            let Type::Array(array) = ty else {
                unreachable!("valid array index depth")
            };
            let mut safe: Vec<NodeRef> = if assumed && require_safe {
                self.nodes_by_type
                    .values()
                    .flatten()
                    .copied()
                    .filter(|&node_ref| {
                        let node = self.get_node(node_ref);
                        let Type::Bits(width) = node.ty else {
                            return false;
                        };
                        let bound_width =
                            (usize::BITS - array.element_count.leading_zeros()) as usize;
                        let in_bounds = width < bound_width
                            || match &node.payload {
                                NodePayload::Literal(value) => {
                                    value.to_bits().expect("bits literal").ult(
                                        &IrBits::make_ubits(width, array.element_count as u64)
                                            .expect("array bound fits index width"),
                                    )
                                }
                                _ => false,
                            };
                        in_bounds
                    })
                    .collect()
            } else {
                Vec::new()
            };
            // Respect the active stage-local pool while preserving node-order
            // sampling when the entire graph is available.
            safe.sort_unstable_by_key(|node_ref| node_ref.index);
            let index = if !safe.is_empty() {
                safe[choose_count(source, safe.len())]
            } else {
                if require_safe {
                    assumed = false;
                }
                self.choose_bits_ref(source).1
            };
            indices.push(index);
            ty = &array.element_type;
        }
        (indices, assumed)
    }

    fn array_update_shapes(&self) -> Vec<(Type, usize, Type)> {
        self.array_index_shapes()
            .into_iter()
            .filter(|(_, _, update_ty)| self.nodes_by_type.contains_key(update_ty))
            .collect()
    }

    fn choose_array_update_shape<S: EntropySource>(&self, source: &mut S) -> (Type, usize, Type) {
        let shapes = self.array_update_shapes();
        if choose_count(source, 10) != 0 {
            let indexed: Vec<_> = shapes.iter().filter(|(_, depth, _)| *depth != 0).collect();
            if !indexed.is_empty() {
                return indexed[choose_count(source, indexed.len())].clone();
            }
        }
        shapes[choose_count(source, shapes.len())].clone()
    }

    fn tuple_field_refs(&self) -> Vec<(Type, NodeRef)> {
        self.nodes_by_type
            .iter()
            .filter(|(ty, _)| {
                !type_contains_token(ty)
                    && type_depth(ty) < self.options.max_type_depth
                    && type_leaf_count(ty) <= self.options.max_aggregate_leaves
            })
            .flat_map(|(ty, refs)| refs.iter().map(|node_ref| (ty.clone(), *node_ref)))
            .collect()
    }

    fn tuple_types(&self) -> Vec<Type> {
        self.nodes_by_type
            .keys()
            .filter(|ty| {
                matches!(ty, Type::Tuple(fields) if !fields.is_empty()) && !type_contains_token(ty)
            })
            .cloned()
            .collect()
    }

    fn choose_tuple_type<S: EntropySource>(&self, source: &mut S) -> Type {
        let types = self.tuple_types();
        types[choose_count(source, types.len())].clone()
    }

    fn selectable_types(&self) -> Vec<Type> {
        self.nodes_by_type
            .keys()
            .filter(|ty| !type_contains_token(ty))
            .cloned()
            .collect()
    }

    fn choose_selectable_type<S: EntropySource>(&self, source: &mut S) -> Type {
        let types = self.selectable_types();
        types[choose_count(source, types.len())].clone()
    }

    fn minimum_materialization_nodes(&self, ty: &Type) -> Result<usize, GenerationError> {
        let mut available_types: BTreeSet<Type> = self.nodes_by_type.keys().cloned().collect();
        required_materialization_nodes(&mut available_types, ty)
    }

    /// Selects an existing value of `ty`, or builds one recursively when no
    /// exact typed value is available.
    fn pick_or_generate_value_of_type<S: EntropySource>(
        &mut self,
        source: &mut S,
        ty: &Type,
    ) -> Result<NodeRef, GenerationError> {
        if self.nodes_by_type.contains_key(ty) {
            return Ok(self.choose_ref_for_type(source, ty));
        }
        match ty {
            Type::Token => Ok(self.add_node(
                Type::Token,
                NodePayload::Literal(IrValue::make_token()),
                None,
            )),
            Type::Bits(width) => Ok(self.pick_or_generate_bits_value(source, *width)),
            Type::Tuple(fields) => {
                let elements: Result<Vec<NodeRef>, GenerationError> = fields
                    .iter()
                    .map(|field| self.pick_or_generate_value_of_type(source, field))
                    .collect();
                Ok(self.add_node(ty.clone(), NodePayload::Tuple(elements?), None))
            }
            Type::Array(array) => {
                let elements: Result<Vec<NodeRef>, GenerationError> = (0..array.element_count)
                    .map(|_| self.pick_or_generate_value_of_type(source, &array.element_type))
                    .collect();
                Ok(self.add_node(ty.clone(), NodePayload::Array(elements?), None))
            }
        }
    }

    /// Builds a missing bits-typed terminal value from a related existing value
    /// when possible, with a random literal as the always-available fallback.
    fn pick_or_generate_bits_value<S: EntropySource>(
        &mut self,
        source: &mut S,
        width: usize,
    ) -> NodeRef {
        debug_assert!(!self.nodes_by_type.contains_key(&Type::Bits(width)));
        let wider: Vec<(usize, NodeRef)> = self
            .nodes_by_type
            .iter()
            .filter_map(|(ty, refs)| match ty {
                Type::Bits(existing_width) if *existing_width > width => refs
                    .iter()
                    .map(|node_ref| (*existing_width, *node_ref))
                    .next(),
                _ => None,
            })
            .collect();
        let narrower: Vec<NodeRef> = self
            .nodes_by_type
            .iter()
            .filter_map(|(ty, refs)| match ty {
                Type::Bits(existing_width) if *existing_width < width => refs.first().copied(),
                _ => None,
            })
            .collect();
        let strategy_count = 1 + usize::from(!wider.is_empty()) + usize::from(!narrower.is_empty());
        let strategy = choose_count(source, strategy_count);
        if !wider.is_empty() && strategy == 0 {
            let (source_width, arg) = wider[choose_count(source, wider.len())];
            let start = choose_between(source, 0, source_width - width);
            return self.add_node(
                Type::Bits(width),
                NodePayload::BitSlice { arg, start, width },
                None,
            );
        }
        let zero_ext_strategy = usize::from(!wider.is_empty());
        if !narrower.is_empty() && strategy == zero_ext_strategy {
            let arg = narrower[choose_count(source, narrower.len())];
            return self.add_node(
                Type::Bits(width),
                NodePayload::ZeroExt {
                    arg,
                    new_bit_count: width,
                },
                None,
            );
        }
        self.add_node(
            Type::Bits(width),
            NodePayload::Literal(generate_uniform_value(source, &Type::Bits(width))),
            None,
        )
    }

    fn finish(self) -> Result<Fn, GenerationError> {
        let ret_node_ref = NodeRef {
            index: self.nodes.len() - 1,
        };
        self.finish_with_return(ret_node_ref)
    }

    fn finish_with_return(self, ret_node_ref: NodeRef) -> Result<Fn, GenerationError> {
        let ret_ty = self.nodes[ret_node_ref.index].ty.clone();
        let function = Fn {
            name: "random_fn".to_string(),
            params: self.params,
            ret_ty,
            nodes: self.nodes,
            ret_node_ref: Some(ret_node_ref),
            outer_attrs: Vec::new(),
            inner_attrs: Vec::new(),
        };
        function
            .check_pir_layout_invariants()
            .map_err(|error| GenerationError::Construction(error.to_string()))?;
        Ok(function)
    }
}

fn gather_stats(function: &Fn) -> GeneratedFnStats {
    gather_stats_with_roots(function, |payload| is_observable_effect_root(payload))
}

/// Measures generated and live operations, treating register writes as roots.
pub fn gather_block_stats(function: &Fn) -> GeneratedFnStats {
    gather_stats_with_roots(function, |payload| {
        is_observable_effect_root(payload) || matches!(payload, NodePayload::RegisterWrite { .. })
    })
}

fn gather_stats_with_roots(
    function: &Fn,
    is_extra_root: impl std::ops::Fn(&NodePayload) -> bool,
) -> GeneratedFnStats {
    let mut live_indices = HashSet::new();
    let mut pending = vec![
        function
            .ret_node_ref
            .expect("generated function always has a return node"),
    ];
    pending.extend(
        function
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| is_extra_root(&node.payload))
            .map(|(index, _)| NodeRef { index }),
    );
    while let Some(node_ref) = pending.pop() {
        if live_indices.insert(node_ref.index) {
            pending.extend(operands(&function.nodes[node_ref.index].payload));
        }
    }

    let mut stats = GeneratedFnStats {
        emitted_node_count: 0,
        live_node_count: 0,
        emitted_operations: BTreeMap::new(),
        live_operations: BTreeMap::new(),
        emitted_bits_widths: BTreeSet::new(),
        live_bits_widths: BTreeSet::new(),
    };
    for (index, node) in function.nodes.iter().enumerate().skip(1) {
        stats.emitted_node_count += 1;
        record_bits_widths(&node.ty, &mut stats.emitted_bits_widths);
        let is_live = live_indices.contains(&index);
        if is_live {
            stats.live_node_count += 1;
            record_bits_widths(&node.ty, &mut stats.live_bits_widths);
        }
        if matches!(node.payload, NodePayload::GetParam(_)) {
            continue;
        }
        let operator = node.payload.get_operator().to_string();
        *stats
            .emitted_operations
            .entry(operator.clone())
            .or_default() += 1;
        if is_live {
            *stats.live_operations.entry(operator).or_default() += 1;
        }
    }
    stats
}

fn record_bits_widths(ty: &Type, widths: &mut BTreeSet<usize>) {
    match ty {
        Type::Token => {}
        Type::Bits(width) => {
            widths.insert(*width);
        }
        Type::Tuple(fields) => {
            for field in fields {
                record_bits_widths(field, widths);
            }
        }
        Type::Array(array) => record_bits_widths(&array.element_type, widths),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn balanced_widths_cover_every_width_and_respect_limits() {
        for maximum in [1, 7, 16, 64, 256] {
            let mut entropy = RngEntropy::new(StdRng::seed_from_u64(42));
            let mut counts = vec![0usize; maximum + 1];
            for _ in 0..32768 {
                counts[balanced_width(&mut entropy, maximum)] += 1;
            }
            assert!(counts[1..].iter().all(|count| *count > 0));
            if maximum > 1 {
                assert!((2500..4000).contains(&counts[1]));
            }
        }
    }

    #[test]
    fn balanced_division_prepares_widths_without_literal_fallback() {
        let options = RandomFnOptions {
            max_bit_width: 256,
            max_div_mod_bit_width: Some(16),
            ..Default::default()
        };
        let mut entropy = RngEntropy::new(StdRng::seed_from_u64(99));
        let mut counts = [0usize; 17];
        for _ in 0..4096 {
            let mut generator = FunctionGenerator::new(&options);
            generator.add_named_param("predicate".into(), Type::Bits(1));
            generator.add_named_param("a".into(), Type::Bits(32));
            generator.add_named_param("b".into(), Type::Bits(64));
            generator.preparation_remaining = 2;
            let result = generator
                .emit_operation(&mut entropy, RandomOperation::Udiv)
                .unwrap();
            counts[generator.get_node(result).ty.bit_count()] += 1;
            assert!(generator.nodes.len() <= 7);
            assert!(
                !generator
                    .nodes
                    .iter()
                    .any(|n| matches!(n.payload, NodePayload::Literal(_)))
            );
        }
        assert!(counts[1..].iter().all(|count| *count > 0));
        assert!(counts[1] < 650);
    }

    #[test]
    fn package_operations_use_balanced_width_preparation() {
        let options = RandomFnOptions {
            max_bit_width: 128,
            max_div_mod_bit_width: Some(16),
            max_invokes_per_function: 0,
            max_counted_fors_per_function: 0,
            enabled_operations: OperationSet::new([RandomOperation::Udiv]),
            ..Default::default()
        };
        let mut entropy = RngEntropy::new(StdRng::seed_from_u64(100));
        let mut package_generator =
            PackageGenerator::new(&mut entropy, &options, StopPolicy::ExactBodyNodes(1));
        let mut generator = FunctionGenerator::new(&options);
        generator.add_named_param("wide".into(), Type::Bits(32));
        let mut invoke_count = 0;
        let mut counted_for_count = 0;

        let added = package_generator
            .add_random_body_node(
                &mut generator,
                &mut invoke_count,
                &mut counted_for_count,
                options.max_nested_counted_for_iterations,
                3,
            )
            .unwrap();

        assert!(added >= 2);
        assert!(
            generator.nodes.iter().any(|node| {
                matches!(node.payload, NodePayload::BitSlice { width: 1..=16, .. })
            })
        );
        assert!(matches!(
            generator.nodes.last().unwrap().payload,
            NodePayload::Binop(Binop::Udiv, _, _)
        ));
    }

    #[test]
    fn balanced_partial_products_prepare_capped_operands() {
        for arbitrary_widths in [false, true] {
            for operation in [RandomOperation::Umulp, RandomOperation::Smulp] {
                let options = RandomFnOptions {
                    max_bit_width: 128,
                    max_multiply_operand_bit_width: Some(16),
                    allow_arbitrary_width_multiply: arbitrary_widths,
                    ..Default::default()
                };
                let mut generator = FunctionGenerator::new(&options);
                generator.add_named_param("wide".into(), Type::Bits(32));
                generator.preparation_remaining = 2;

                assert!(generator.operation_is_applicable(operation));
                let node_ref = generator
                    .emit_operation(&mut RngEntropy::new(StdRng::seed_from_u64(101)), operation)
                    .unwrap();
                let NodePayload::Binop(_, lhs, rhs) = generator.get_node(node_ref).payload else {
                    panic!("partial product must have two operands");
                };
                assert!(generator.get_node(lhs).ty.bit_count() <= 16);
                assert!(generator.get_node(rhs).ty.bit_count() <= 16);
                assert!(generator.nodes.iter().any(|node| {
                    matches!(node.payload, NodePayload::BitSlice { width: 1..=16, .. })
                }));
            }
        }
    }

    #[test]
    fn exhausted_balanced_body_rolls_back_all_preparation_nodes() {
        let options = RandomFnOptions::default();
        let mut generator = FunctionGenerator::new(&options);
        generator.add_named_param("x".into(), Type::Bits(32));
        let pool = generator.nodes_by_type.clone();
        for bytes in [&[][..], &[1u8][..]] {
            assert_eq!(
                generator
                    .add_random_body_node(&mut DepletableBytes::new(bytes), 8)
                    .unwrap(),
                0
            );
            assert_eq!(generator.nodes.len(), 2);
            assert_eq!(generator.nodes_by_type, pool);
        }
    }

    #[test]
    fn multiply_operand_limit_excludes_operations_without_eligible_values() {
        for arbitrary_widths in [false, true] {
            let options = RandomFnOptions {
                max_bit_width: 128,
                max_multiply_operand_bit_width: Some(64),
                allow_arbitrary_width_multiply: arbitrary_widths,
                ..Default::default()
            };
            let mut generator = FunctionGenerator::new(&options);
            generator.add_named_param("wide".into(), Type::Bits(128));
            let operations = [
                RandomOperation::Umul,
                RandomOperation::Smul,
                RandomOperation::Umulp,
                RandomOperation::Smulp,
            ];
            for operation in operations {
                assert!(!generator.operation_is_applicable(operation));
            }
            generator.add_named_param("narrow".into(), Type::Bits(64));
            for operation in operations {
                assert!(generator.operation_is_applicable(operation));
                let node_ref = generator
                    .emit_operation(&mut DepletableBytes::new(&[]), operation)
                    .unwrap();
                let NodePayload::Binop(_, lhs, rhs) = generator.get_node(node_ref).payload else {
                    panic!("multiply must have two operands");
                };
                assert!(generator.get_node(lhs).ty.bit_count() <= 64);
                assert!(generator.get_node(rhs).ty.bit_count() <= 64);
            }
        }
    }

    #[test]
    fn assumed_array_indices_only_use_the_active_value_pool() {
        let options = RandomFnOptions {
            array_assumption_mode: ArrayAssumptionMode::ProvenSafe,
            ..Default::default()
        };
        for width in [0, 1, 2, 8, 129] {
            let mut generator = FunctionGenerator::new(&options);
            generator.add_node(
                Type::Bits(width),
                NodePayload::Literal(IrValue::make_ubits(width, 0).unwrap()),
                None,
            );
            generator.add_node(
                Type::Bits(width),
                NodePayload::RegisterRead {
                    register: "future".into(),
                },
                None,
            );
            let available = generator.add_node(
                Type::Bits(width),
                NodePayload::Literal(IrValue::make_ubits(width, u64::from(width != 0)).unwrap()),
                None,
            );
            generator.nodes_by_type.clear();
            generator
                .nodes_by_type
                .insert(Type::Bits(width), vec![available]);
            let array = Type::new_array(Type::new_array(Type::Bits(8), 4), 4);
            for choice in [0u64, 1, 2, u64::MAX] {
                let bytes: Vec<u8> = [1u64, choice, choice]
                    .into_iter()
                    .flat_map(u64::to_le_bytes)
                    .collect();
                let (indices, assumed) =
                    generator.choose_array_indices(&mut DepletableBytes::new(&bytes), &array, 2);
                assert!(assumed);
                assert_eq!(indices, vec![available, available]);
            }
        }
    }
}
