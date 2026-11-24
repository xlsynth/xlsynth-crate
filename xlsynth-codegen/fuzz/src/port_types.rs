// SPDX-License-Identifier: Apache-2.0

//! Differential checks for DSLX-derived SystemVerilog port annotations.

use xlsynth::external_tool::ToolError;
use xlsynth_codegen::{BlockCodegenOptions, Layout};
use xlsynth_pir::ir::{NodePayload, Package};
use xlsynth_test_helpers::rtl_sim::identifier;

use crate::semantics::Trace;
use crate::{block_options, emit, generate, iverilog};

const MAX_INPUT_PORTS: usize = 3;
const MAX_OUTPUT_PORTS: usize = 2;
const TYPE_CHOICE_BYTES: usize = 4 * (MAX_INPUT_PORTS + MAX_OUTPUT_PORTS);
const STIMULUS_OFFSET: usize = 1 + TYPE_CHOICE_BYTES;
const HEADER_BYTES: usize = STIMULUS_OFFSET + 8;
const ADAPTER_MODULE: &str = "xlsynth_fuzz_flat_adapter";

/// Compares annotated RTL against PIR with independently mutable type, graph,
/// and value inputs.
pub fn check(data: &[u8]) -> Result<(), ToolError> {
    check_with_progress(data).0
}

/// Checks a case and returns any cumulative semantic coverage snapshot to the
/// caller.
pub fn check_with_progress(data: &[u8]) -> (Result<(), ToolError>, Option<String>) {
    let mut header = [0; HEADER_BYTES];
    let count = data.len().min(HEADER_BYTES);
    header[..count].copy_from_slice(&data[..count]);

    let mut generator = block_options(header[0] & 1 != 0, false);
    generator.max_input_ports = MAX_INPUT_PORTS;
    generator.max_output_ports = MAX_OUTPUT_PORTS;
    generator.allow_zero_width_ports_and_registers = false;
    generator.function_options.max_nodes = 24;
    generator.function_options.max_bit_width = 32;
    generator.function_options.max_type_depth = if header[0] & 1 != 0 { 2 } else { 0 };
    generator.function_options.max_aggregate_leaves = 8;
    generator.function_options.max_array_length = 4;
    generator.function_options.max_tuple_length = 4;
    generator.function_options.allow_zero_width_bits = false;
    let package = generate(data.get(HEADER_BYTES..).unwrap_or_default(), &generator);

    let codegen = BlockCodegenOptions {
        layout: if header[0] & 2 != 0 {
            Layout::Pipeline
        } else {
            Layout::None
        },
        ..BlockCodegenOptions::default()
    };
    check_package(
        package,
        &header[1..STIMULUS_OFFSET],
        &codegen,
        *blake3::hash(&header[STIMULUS_OFFSET..]).as_bytes(),
    )
}

/// Applies the selected DSLX port types and compares emitted RTL with PIR.
pub(crate) fn check_package(
    mut package: Package,
    choices: &[u8],
    codegen: &BlockCodegenOptions,
    stimulus_seed: [u8; 32],
) -> (Result<(), ToolError>, Option<String>) {
    let declarations = crate::dslx_types::apply(&mut package, choices);
    let rtl = format!(
        "{declarations}\n{}\n{}",
        emit(&package, codegen),
        flat_boundary_adapter(&package)
    );
    let trace = Trace::with_seed(&package, stimulus_seed);
    let result = iverilog::assert_rtl_trace(&package, &rtl, Some(ADAPTER_MODULE), &trace)
        .map_err(|error| error.with_context(format!("annotated RTL:\n{rtl}")));
    let progress = crate::port_type_coverage::record_progress(&package, choices, &trace, &result);
    (result, progress)
}

/// Adapts the flat oracle interface to nominal SV input types with explicit
/// casts; vector-to-enum input connections do not permit implicit conversion.
fn flat_boundary_adapter(package: &Package) -> String {
    let block = crate::top_block(package);
    assert!(
        block.clock_port_name().is_none(),
        "port-type fuzzing expects a combinational block"
    );
    let mut ports = Vec::new();
    let mut connections = Vec::new();
    for node_ref in block.input_ports().chain(block.output_ports()) {
        let width = block.port_type(node_ref).bit_count();
        if width == 0 {
            continue;
        }
        let node = block.get_node(node_ref);
        let name = identifier(block.port_name(node_ref)).expect("generated port name is valid");
        match &node.payload {
            NodePayload::InputPort { sv_type, .. } => {
                let ty = sv_type
                    .as_deref()
                    .expect("positive-width fuzz input is annotated");
                ports.push(format!("  input logic [{}:0] {name}", width - 1));
                connections.push(format!("    .{name}({ty}'({name}))"));
            }
            NodePayload::OutputPort { .. } => {
                ports.push(format!("  output wire [{}:0] {name}", width - 1));
                connections.push(format!("    .{name}({name})"));
            }
            _ => unreachable!("block data port has an input/output payload"),
        }
    }
    let module = identifier(&block.name).expect("generated module name is valid");
    format!(
        "module {ADAPTER_MODULE}(\n{}\n);\n  {module} inner (\n{}\n  );\nendmodule\n",
        ports.join(",\n"),
        connections.join(",\n")
    )
}

#[cfg(test)]
mod tests {
    use rand::{RngCore, SeedableRng, rngs::StdRng};

    use super::{check, check_package, flat_boundary_adapter};

    /// Exercises scalar and aggregate graphs in both output layouts.
    #[test]
    fn seeded_port_type_cases_match_pir() {
        let mut rng = StdRng::seed_from_u64(0x706f_7274_7479_7065);
        for kind in 0..crate::dslx_types::TYPE_KIND_COUNT {
            let mut input = [0; 2048];
            rng.fill_bytes(&mut input);
            let flags = kind % 4;
            input[0] = flags;
            input[1] = kind;
            check(&input).unwrap_or_else(|error| panic!("kind={kind}, flags={flags}: {error}"));
        }
    }

    /// Replays a packed-array input forwarded directly to an enum output.
    #[test]
    fn packed_array_to_enum_passthrough_matches_pir() {
        // Select the input and output types and leave the graph body empty.
        check(&[0, 2, 0, 0, 0, 4]).unwrap();
    }

    /// Checks packing and signedness through a live multi-operation chain.
    #[test]
    fn chained_operations_on_annotated_ports_match_pir() {
        let source = r#"package chained

top block chained(data: bits[8], signed_data: bits[8], result: bits[8]) {
  data: bits[8] = input_port(name=data, id=1)
  signed_data: bits[8] = input_port(name=signed_data, id=2)
  low: bits[4] = bit_slice(data, start=1, width=4, id=3)
  high: bits[4] = bit_slice(signed_data, start=3, width=4, id=4)
  joined: bits[8] = concat(high, low, id=5)
  amount: bits[3] = literal(value=2, id=6)
  shifted: bits[8] = shrl(joined, amount, id=7)
  comparison: bits[1] = ult(shifted, signed_data, id=8)
  selected: bits[8] = sel(comparison, cases=[shifted, joined], id=9)
  result: () = output_port(selected, name=result, id=10)
}
"#;
        let package = crate::parse_reference(source);
        // Packed-array input, signed input, and enum output, respectively.
        let choices = [2, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0];
        check_package(
            package,
            &choices,
            &xlsynth_codegen::BlockCodegenOptions::default(),
            [0x71; 32],
        )
        .0
        .unwrap();
    }

    /// Checks that an enum input is connected through a legal typed cast.
    #[test]
    fn enum_input_adapter_is_valid_sv() {
        let source = r#"package typed

top block typed(data: bits[2], result: bits[2]) {
  data: bits[2] = input_port(name=data, id=1)
  result: () = output_port(data, name=result, id=2)
}
"#;
        let mut package = crate::parse_reference(source);
        let declarations = crate::dslx_types::apply(&mut package, &[4, 0, 0, 0]);
        let rtl = format!(
            "{declarations}\n{}\n{}",
            crate::emit(&package, &xlsynth_codegen::BlockCodegenOptions::default()),
            flat_boundary_adapter(&package)
        );
        xlsynth_test_helpers::assert_valid_sv(&rtl);
    }

    /// libFuzzer begins with empty input when no corpus is supplied.
    #[test]
    fn empty_port_type_case_matches_pir() {
        check(&[]).unwrap();
    }
}
