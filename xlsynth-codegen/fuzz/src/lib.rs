// SPDX-License-Identifier: Apache-2.0

//! Shared public block generators and independent codegen reference oracles.

pub mod coverage;
pub mod input;
pub mod iverilog;
pub mod mapped_semantics;
pub mod preflight;
pub mod references;
pub mod semantics;
pub mod stimulus;
pub mod tool_failure;
pub mod xls;
#[cfg(feature = "external-yosys")]
pub mod yosys;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use xlsynth::IrValue;
use xlsynth::external_tool::ToolError;
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::ir::{BlockMetadata, Fn, Package, PackageMember};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_random::{
    ArrayAssumptionMode, BlockTopology, DepletableBytes, OperationSet, RandomBlockOptions,
    RandomBlockResetTiming, RandomFnOptions, RandomOperation, StopPolicy, generate_block_package,
};
use xlsynth_pir::random_inputs::generate_uniform_value_with_rng;

/// Number of independent value samples used for each generated graph.
pub const INPUT_SAMPLE_COUNT: usize = 16;

/// Number of clock cycles checked for each generated sequential graph.
pub const CYCLE_COUNT: usize = 24;

/// Parses and validates a synthetic public block reference.
pub fn parse_reference(ir: &str) -> Package {
    Parser::new(ir)
        .parse_and_validate_package()
        .unwrap_or_else(|error| panic!("public block reference should validate:\n{ir}\n{error}"))
}

/// Emits a generated package and reports its complete input on failure.
pub fn emit(package: &Package, options: &BlockCodegenOptions) -> String {
    emit_system_verilog(package, options)
        .unwrap_or_else(|error| {
            panic!(
                "valid block SystemVerilog emission failed:\nIR:\n{}\noptions: {options:?}\nerror: {error}",
                package
            )
        })
        .system_verilog
}

/// Selects native datapath semantics or the stock-XLS-compatible subset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Profile {
    NativeSemantics,
    StockXls,
}

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::NativeSemantics => "native",
            Self::StockXls => "stock-xls",
        }
    }

    /// Uses one fixed wiring-header size for mixed combinational/sequential
    /// cases.
    pub fn mixed_block_options(self) -> RandomBlockOptions {
        self.block_options(true, true)
    }

    /// Bounds cost without tying native coverage to another frontend's limits.
    pub fn block_options(self, aggregates: bool, sequential: bool) -> RandomBlockOptions {
        RandomBlockOptions {
            min_input_ports: 1,
            max_input_ports: 4,
            min_output_ports: 1,
            max_output_ports: 3,
            topology: if sequential {
                BlockTopology::GeneralSequential
            } else {
                BlockTopology::Combinational
            },
            max_registers: if sequential { 3 } else { 0 },
            allow_zero_width_ports_and_registers: self == Self::NativeSemantics,
            allow_load_enable: sequential,
            allow_reset: sequential,
            reset_timing: RandomBlockResetTiming::Synchronous,
            function_options: RandomFnOptions {
                max_params: 6,
                max_nodes: 48,
                max_bit_width: 256,
                // Bound simulator multiword arithmetic and synthesis expansion.
                max_div_mod_bit_width: Some(16),
                max_multiply_operand_bit_width: Some(64),
                max_type_depth: if aggregates { 3 } else { 0 },
                max_aggregate_leaves: 32,
                max_array_length: 8,
                max_tuple_length: 6,
                max_nary_operands: 8,
                allow_arrays: aggregates,
                allow_tuples: aggregates,
                allow_zero_width_bits: self == Self::NativeSemantics,
                allow_arbitrary_width_multiply: true,
                array_assumption_mode: ArrayAssumptionMode::ProvenSafe,
                allow_extension_ops: self == Self::NativeSemantics,
                allow_gate: true,
                enabled_operations: self.operations(aggregates),
                ..RandomFnOptions::default()
            },
            ..RandomBlockOptions::default()
        }
    }

    /// Calls, partial products, and debug events are outside these campaigns.
    pub fn operations(self, aggregates: bool) -> OperationSet {
        OperationSet::new(OperationSet::all_supported().iter().filter(|operation| {
            if matches!(
                operation,
                RandomOperation::Invoke
                    | RandomOperation::CountedFor
                    | RandomOperation::Umulp
                    | RandomOperation::Smulp
                    | RandomOperation::AfterAll
                    | RandomOperation::Cover
                    | RandomOperation::Assert
                    | RandomOperation::Trace
            ) {
                return false;
            }
            if self == Self::StockXls
                && matches!(
                    operation,
                    RandomOperation::ExtCarryOut
                        | RandomOperation::ExtPrioEncode
                        | RandomOperation::ExtClz
                        | RandomOperation::ExtNormalizeLeft
                        | RandomOperation::ExtMaskLow
                        | RandomOperation::ExtNaryAdd
                )
            {
                return false;
            }
            if !aggregates
                && matches!(
                    operation,
                    RandomOperation::Array
                        | RandomOperation::ArrayIndex
                        | RandomOperation::ArrayConcat
                        | RandomOperation::ArraySlice
                        | RandomOperation::ArrayUpdate
                        | RandomOperation::Tuple
                        | RandomOperation::TupleIndex
                )
            {
                return false;
            }
            true
        }))
    }
}

/// Uses the native profile for existing focused targets.
pub fn block_options(aggregates: bool, sequential: bool) -> RandomBlockOptions {
    Profile::NativeSemantics.block_options(aggregates, sequential)
}

/// Builds a valid block package directly from coverage-guided structure bytes.
pub fn generate(data: &[u8], options: &RandomBlockOptions) -> Package {
    let mut entropy = DepletableBytes::new(data);
    generate_block_package(&mut entropy, options, StopPolicy::WhenEntropyDepleted)
        .expect("fixed public random-block options should construct a valid package")
        .package
}

/// Checks a mixed profile case and returns only documented oracle exclusions.
pub fn check_profile_case(package: &Package, profile: Profile) -> Option<&'static str> {
    check_profile_trace(
        package,
        profile,
        &BlockCodegenOptions::default(),
        &semantics::Trace::for_package(package),
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

/// Checks all selected oracles against identical stimuli and codegen options.
pub fn check_profile_trace(
    package: &Package,
    profile: Profile,
    options: &BlockCodegenOptions,
    trace: &semantics::Trace,
) -> Result<Option<&'static str>, ToolError> {
    if profile == Profile::StockXls {
        let executable = xls::stock_xls_path().expect("stock-XLS profile requires block_to_verilog_main; configure XLSYNTH_TOOLS or XLSYNTH_BLOCK_TO_VERILOG_PATH");
        let context = xlsynth_g8r_fuzz::external_yosys::required_external_yosys_context()
            .unwrap_or_else(|error| {
                panic!("stock-XLS profile requires Yosys/Liberty netlist evaluation: {error}")
            });
        if let Some(reason) = xls::stock_xls_skip_reason(package) {
            return Ok(Some(reason));
        }
        xls::assert_stock_xls_semantics(package, executable, context, options, trace)?;
    } else {
        let rtl = emit(package, options);
        assert_eq!(
            rtl,
            emit(package, options),
            "codegen options produced nondeterministic RTL"
        );
        iverilog::assert_rtl_trace(
            package,
            &rtl,
            options.module_name.as_deref(),
            iverilog::StateLayout::Packed,
            trace,
        )?;
    }
    Ok(None)
}

/// Returns the selected generated block and its structural metadata.
pub fn top_block(package: &Package) -> (&Fn, &BlockMetadata) {
    let Some(PackageMember::Block { func, metadata }) = package.get_top_block() else {
        panic!("generated package should have a top block:\n{package}");
    };
    (func, metadata)
}

/// Derives deterministic value stimuli from the graph, not fuzzer entropy.
pub fn deterministic_rng(ir: &str) -> StdRng {
    StdRng::from_seed(*blake3::hash(ir.as_bytes()).as_bytes())
}

/// Draws one correctly typed value for each visible generated input port.
pub fn generate_inputs(block: &Fn, rng: &mut StdRng) -> Vec<IrValue> {
    block
        .params
        .iter()
        .map(|param| generate_uniform_value_with_rng(rng, &param.ty))
        .collect()
}

/// Draws a reset-aware deterministic input vector for one clock cycle.
pub fn generate_cycle_inputs(
    block: &Fn,
    metadata: &BlockMetadata,
    rng: &mut StdRng,
    cycle: usize,
    require_initial_reset: bool,
) -> Vec<IrValue> {
    block
        .params
        .iter()
        .map(|param| {
            if let Some(reset) = &metadata.reset
                && param.name == reset.port_name
            {
                let asserted = if require_initial_reset {
                    cycle == 0
                } else if cycle == 0 {
                    rng.gen_bool(0.5)
                } else {
                    rng.gen_ratio(1, 8)
                };
                let high = if reset.active_low {
                    !asserted
                } else {
                    asserted
                };
                return IrValue::make_ubits(1, u64::from(high))
                    .expect("reset signal should fit in bits[1]");
            }
            generate_uniform_value_with_rng(rng, &param.ty)
        })
        .collect()
}

/// Compares emitted outputs with PIR using Icarus.
pub fn assert_combinational_semantics(
    package: &Package,
    options: &BlockCodegenOptions,
) -> Result<(), ToolError> {
    let rtl = emit(package, options);
    iverilog::assert_rtl_semantics(
        package,
        &rtl,
        options.module_name.as_deref(),
        iverilog::StateLayout::Packed,
    )
}

/// Compares clocked RTL outputs and next state with PIR using Icarus.
pub fn assert_sequential_semantics(
    package: &Package,
    options: &BlockCodegenOptions,
) -> Result<(), ToolError> {
    let rtl = emit(package, options);
    iverilog::assert_rtl_semantics(
        package,
        &rtl,
        options.module_name.as_deref(),
        iverilog::StateLayout::Packed,
    )
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::coverage::{CoverageReport, Outcome};
    use crate::{Profile, check_profile_case};
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use xlsynth_codegen::BlockCodegenOptions;
    use xlsynth_g8r::liberty::cell_formula::parse_formula;
    use xlsynth_g8r::liberty::parser::{
        LibertyPayloadOptions, parse_liberty_files_with_payload_options,
    };
    use xlsynth_pir::ir_random::RandomOperation;

    use super::{
        assert_combinational_semantics, assert_sequential_semantics, block_options, emit, generate,
        parse_reference, references,
    };

    #[test]
    fn profiles_reach_all_enabled_operations_and_expanded_shapes() {
        for profile in [Profile::NativeSemantics, Profile::StockXls] {
            let options = profile.mixed_block_options();
            assert_eq!(options.wiring_header_byte_count().unwrap(), 72);
            assert_eq!(options.function_options.max_div_mod_bit_width, Some(16));
            assert_eq!(
                options.function_options.max_multiply_operand_bit_width,
                Some(64)
            );
            for operation in [
                RandomOperation::Invoke,
                RandomOperation::CountedFor,
                RandomOperation::Umulp,
                RandomOperation::Smulp,
                RandomOperation::Assert,
                RandomOperation::Cover,
                RandomOperation::Trace,
                RandomOperation::AfterAll,
            ] {
                assert!(
                    !options
                        .function_options
                        .enabled_operations
                        .contains(operation)
                );
            }
            let mut rng = StdRng::seed_from_u64(1);
            let mut report = CoverageReport::default();
            for _ in 0..2000 {
                let mut data = [0; 2048];
                rng.fill_bytes(&mut data);
                report.record(&generate(&data, &options), Outcome::GeneratedOnly);
            }
            for operation in profile.operations(true).iter() {
                assert!(
                    report.generated_operations.contains_key(operation.name()),
                    "{profile:?}: {} missing",
                    operation.name()
                );
            }
            assert!(report.checked_graph_operations.is_empty());
            assert!(
                report
                    .div_mod_widths
                    .keys()
                    .all(|key| { key.split_once(':').unwrap().1.parse::<usize>().unwrap() <= 16 })
            );
            for operation in ["udiv", "sdiv", "umod", "smod"] {
                assert!(
                    report
                        .div_mod_widths
                        .contains_key(&format!("{operation}:16"))
                );
            }
            assert!(report.bit_widths.contains_key(&256));
            assert!(report.selector_widths.keys().any(|width| *width > 64));
            assert!(report.array_lengths.contains_key(&8));
            assert!(report.tuple_lengths.contains_key(&6));
            assert!(report.type_depths.contains_key(&3));
            assert!(report.attributes.contains_key("assumed_in_bounds=true"));
            assert!(report.attributes.contains_key("assumed_in_bounds=false"));
            assert!(report.attributes.contains_key("load_enable=true"));
            assert!(report.attributes.contains_key("load_enable=false"));
            for count in 1..=3 {
                assert!(report.register_counts.contains_key(&count));
            }
            assert_eq!(
                report.bit_widths.contains_key(&0),
                profile == Profile::NativeSemantics
            );
            assert_eq!(
                report.generated_operations.contains_key("ext_clz"),
                profile == Profile::NativeSemantics
            );
        }
    }

    #[test]
    fn native_profile_checks_wide_priority_expressions_with_icarus() {
        let package = parse_reference(
            r#"package wide
top block wide(x: bits[255], out: bits[256]) {
  x: bits[255] = input_port(name=x, id=1)
  hot: bits[256] = one_hot(x, lsb_prio=true, id=2)
  out: () = output_port(hot, name=out, id=3)
}
"#,
        );
        assert_eq!(check_profile_case(&package, Profile::NativeSemantics), None);
    }

    #[test]
    fn native_profile_checks_omitted_zero_bit_ports_and_registers() {
        let package = parse_reference(
            r#"package zero
top block zero(clk: clock, x: bits[0], enable: bits[1], out: bits[0]) {
  reg r(bits[0])
  x: bits[0] = input_port(name=x, id=1)
  enable: bits[1] = input_port(name=enable, id=2)
  q: bits[0] = register_read(register=r, id=3)
  d: () = register_write(x, register=r, load_enable=enable, id=4)
  out: () = output_port(q, name=out, id=5)
}
"#,
        );
        assert_eq!(check_profile_case(&package, Profile::NativeSemantics), None);
    }

    #[test]
    fn synthetic_public_references_are_valid_block_packages() {
        for reference in [
            references::COMBINATIONAL,
            references::SEQUENTIAL,
            references::AGGREGATE,
            references::HIERARCHY,
            references::EXTERN,
        ] {
            assert!(parse_reference(reference).get_top_block().is_some());
        }
        for signed in [false, true] {
            let source = references::partial_product(signed, 5, 7, 11);
            assert!(parse_reference(&source).get_top_block().is_some());
        }
    }

    #[test]
    fn public_scalar_reference_preserves_combinational_semantics() {
        let package = parse_reference(references::COMBINATIONAL);
        assert_combinational_semantics(&package, &BlockCodegenOptions::default()).unwrap();
    }

    #[test]
    fn public_sequential_reference_preserves_feedback_and_reset() {
        let package = parse_reference(references::SEQUENTIAL);
        assert_sequential_semantics(&package, &BlockCodegenOptions::default()).unwrap();
    }

    #[test]
    fn public_aggregate_reference_preserves_packed_port_semantics() {
        let package = parse_reference(references::AGGREGATE);
        assert_combinational_semantics(&package, &BlockCodegenOptions::default()).unwrap();
    }

    #[test]
    fn public_hierarchy_and_external_references_emit_deterministically() {
        let options = BlockCodegenOptions::default();
        for reference in [references::HIERARCHY, references::EXTERN] {
            let package = parse_reference(reference);
            assert_eq!(emit(&package, &options), emit(&package, &options));
        }
    }

    #[test]
    fn deterministic_random_scalar_blocks_preserve_combinational_semantics() {
        let mut rng = StdRng::seed_from_u64(0x5ca1_a000);
        let generation = block_options(false, false);
        for _ in 0..24 {
            let mut entropy = [0u8; 192];
            rng.fill_bytes(&mut entropy);
            let package = generate(&entropy, &generation);
            assert_combinational_semantics(&package, &BlockCodegenOptions::default()).unwrap();
        }
    }

    #[test]
    fn deterministic_random_aggregate_blocks_preserve_combinational_semantics() {
        let mut rng = StdRng::seed_from_u64(0xa663_e6a7);
        let generation = block_options(true, false);
        for _ in 0..24 {
            let mut entropy = [0u8; 192];
            rng.fill_bytes(&mut entropy);
            let package = generate(&entropy, &generation);
            assert_combinational_semantics(&package, &BlockCodegenOptions::default()).unwrap();
        }
    }

    #[test]
    fn deterministic_random_sequential_blocks_preserve_cycle_semantics() {
        let mut rng = StdRng::seed_from_u64(0x5e90_e071);
        let generation = block_options(false, true);
        for _ in 0..12 {
            let mut entropy = [0u8; 192];
            rng.fill_bytes(&mut entropy);
            let package = generate(&entropy, &generation);
            assert_sequential_semantics(&package, &BlockCodegenOptions::default()).unwrap();
        }
    }

    #[test]
    fn public_cell_library_supports_combinational_and_sequential_mapping() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/public_cells.lib");
        let library = parse_liberty_files_with_payload_options(
            &[path],
            LibertyPayloadOptions {
                include_timing: false,
                include_power: false,
            },
        )
        .expect("independently authored public cell library should parse");

        for name in [
            "INV", "BUF", "AND2", "NAND2", "OR2", "NOR2", "XOR2", "XNOR2", "MUX2", "DFF",
        ] {
            let cell = library
                .cells
                .iter()
                .find(|cell| cell.name == name)
                .unwrap_or_else(|| panic!("public library is missing required cell `{name}`"));
            if name == "DFF" {
                assert_eq!(
                    cell.sequential.len(),
                    1,
                    "public DFF must have one sequential model"
                );
            }
            for pin in &cell.pins {
                let function = library.resolve_string(&pin.function);
                if !function.is_empty() {
                    parse_formula(function).unwrap_or_else(|error| {
                        panic!(
                            "public cell `{name}` has an unsupported formula `{function}`: {error}"
                        )
                    });
                }
            }
        }
    }
}
