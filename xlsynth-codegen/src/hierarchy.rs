// SPDX-License-Identifier: Apache-2.0

//! Combinational-cycle validation for the emission hierarchy.

use xlsynth_pir::BuilderError;
use xlsynth_pir::ir::{Block, NodeRef, Package, node_textual_id};
use xlsynth_pir::ir_builder::{BlockWiringCycleOptions, verify_block_wiring_cycles_with_options};

use crate::BlockCodegenError;

/// Reports a PIR cycle witness using the emitted block's node names.
pub(crate) fn verify_hierarchy(
    top: &Block,
    package: &Package,
    array_index_bounds_checking: bool,
) -> Result<(), BlockCodegenError> {
    let options = BlockWiringCycleOptions {
        array_index_bounds_checking,
    };
    verify_block_wiring_cycles_with_options(top, package, &options).map_err(|error| {
        if let BuilderError::CombinationalCycle { block, steps } = &error
            && let Some(source) = package.get_block(block)
            && let Some(first) = steps.first()
            && let Some(index) = source.nodes.iter().position(|n| n.text_id == first.node_id)
        {
            let name = node_textual_id(source, NodeRef { index });
            return BlockCodegenError::InvalidBlock(format!(
                "block `{block}` contains a combinational cycle through its block hierarchy at node `{name}`"
            ));
        }
        BlockCodegenError::InvalidBlock(error.to_string())
    })
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::ir::Package;
    use xlsynth_pir::ir_parser::Parser;

    use crate::BlockCodegenError;

    /// Parses and verifies the complete hierarchy used by a test.
    fn parse_package(source: &str) -> Package {
        Parser::new(source)
            .parse_and_verify_package()
            .unwrap_or_else(|error| panic!("invalid public test package:\n{source}\n{error}"))
    }

    fn verify_hierarchy(package: &Package) -> Result<(), BlockCodegenError> {
        super::verify_hierarchy(package.get_top_block().unwrap(), package, true)
    }

    #[test]
    fn nested_child_feedback_is_rejected_transitively() {
        let package = parse_package(
            r#"package public_nested_cycle

block inner(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

block middle(value: bits[8], result: bits[8]) {
  instantiation nested(block=inner, kind=block)
  value: bits[8] = input_port(name=value, id=3)
  connected: () = instantiation_input(value, instantiation=nested, port_name=value, id=4)
  received: bits[8] = instantiation_output(instantiation=nested, port_name=result, id=5)
  result: () = output_port(received, name=result, id=6)
}

top block feedback(result: bits[8]) {
  instantiation component(block=middle, kind=block)
  fed: bits[8] = instantiation_output(instantiation=component, port_name=result, id=7)
  connected: () = instantiation_input(fed, instantiation=component, port_name=value, id=8)
  result: () = output_port(fed, name=result, id=9)
}
"#,
        );

        assert!(matches!(
            verify_hierarchy(&package),
            Err(BlockCodegenError::InvalidBlock(message))
                if message.contains("block `feedback` contains a combinational cycle")
        ));
    }

    #[test]
    fn registered_child_legally_breaks_the_feedback_path() {
        let package = parse_package(
            r#"package public_registered_feedback

block delayed(clk: clock, value: bits[8], result: bits[8]) {
  reg state(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  stored: bits[8] = register_read(register=state, id=2)
  update: () = register_write(value, register=state, id=3)
  result: () = output_port(stored, name=result, id=4)
}

top block feedback(clk: clock, result: bits[8]) {
  instantiation component(block=delayed, kind=block)
  fed: bits[8] = instantiation_output(instantiation=component, port_name=result, id=5)
  connected: () = instantiation_input(fed, instantiation=component, port_name=value, id=6)
  result: () = output_port(fed, name=result, id=7)
}
"#,
        );

        assert_eq!(verify_hierarchy(&package), Ok(()));
    }

    #[test]
    fn unrelated_child_input_and_output_do_not_create_false_feedback() {
        let package = parse_package(
            r#"package public_independent_ports

block independent(left: bits[8], right: bits[8], first: bits[8], second: bits[8]) {
  left: bits[8] = input_port(name=left, id=1)
  right: bits[8] = input_port(name=right, id=2)
  first: () = output_port(left, name=first, id=3)
  second: () = output_port(right, name=second, id=4)
}

top block connected(value: bits[8], result: bits[8]) {
  instantiation component(block=independent, kind=block)
  value: bits[8] = input_port(name=value, id=5)
  first_value: bits[8] = instantiation_output(instantiation=component, port_name=first, id=6)
  second_value: bits[8] = instantiation_output(instantiation=component, port_name=second, id=7)
  connected_left: () = instantiation_input(second_value, instantiation=component, port_name=left, id=8)
  connected_right: () = instantiation_input(value, instantiation=component, port_name=right, id=9)
  result: () = output_port(first_value, name=result, id=10)
}
"#,
        );

        assert_eq!(verify_hierarchy(&package), Ok(()));
    }

    #[test]
    fn registered_output_breaks_feedback_even_with_a_combinational_sibling() {
        let package = parse_package(
            r#"package public_mixed_output_feedback

block mixed(clk: clock, value: bits[8], direct: bits[8], delayed: bits[8]) {
  reg state(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  stored: bits[8] = register_read(register=state, id=2)
  update: () = register_write(value, register=state, id=3)
  direct: () = output_port(value, name=direct, id=4)
  delayed: () = output_port(stored, name=delayed, id=5)
}

top block feedback(clk: clock, direct: bits[8], delayed: bits[8]) {
  instantiation component(block=mixed, kind=block)
  direct_value: bits[8] = instantiation_output(instantiation=component, port_name=direct, id=6)
  delayed_value: bits[8] = instantiation_output(instantiation=component, port_name=delayed, id=7)
  connected: () = instantiation_input(delayed_value, instantiation=component, port_name=value, id=8)
  direct: () = output_port(direct_value, name=direct, id=9)
  delayed: () = output_port(delayed_value, name=delayed, id=10)
}
"#,
        );

        assert_eq!(verify_hierarchy(&package), Ok(()));
    }

    #[test]
    fn combinational_output_remains_cyclic_with_a_registered_sibling() {
        let package = parse_package(
            r#"package public_mixed_output_cycle

block mixed(clk: clock, value: bits[8], direct: bits[8], delayed: bits[8]) {
  reg state(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  stored: bits[8] = register_read(register=state, id=2)
  update: () = register_write(value, register=state, id=3)
  direct: () = output_port(value, name=direct, id=4)
  delayed: () = output_port(stored, name=delayed, id=5)
}

top block feedback(clk: clock, direct: bits[8], delayed: bits[8]) {
  instantiation component(block=mixed, kind=block)
  direct_value: bits[8] = instantiation_output(instantiation=component, port_name=direct, id=6)
  delayed_value: bits[8] = instantiation_output(instantiation=component, port_name=delayed, id=7)
  connected: () = instantiation_input(direct_value, instantiation=component, port_name=value, id=8)
  direct: () = output_port(direct_value, name=direct, id=9)
  delayed: () = output_port(delayed_value, name=delayed, id=10)
}
"#,
        );

        assert!(matches!(
            verify_hierarchy(&package),
            Err(BlockCodegenError::InvalidBlock(message))
                if message.contains("block `feedback` contains a combinational cycle")
        ));
    }

    #[test]
    fn forwarded_clock_reset_and_zero_width_ports_preserve_registered_feedback() {
        let package = parse_package(
            r#"package public_empty_reset_feedback

block delayed(clk: clock, rst: bits[1], empty: bits[0], value: bits[8], empty_result: bits[0], result: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg state(bits[8], reset_value=0)
  rst: bits[1] = input_port(name=rst, id=1)
  empty: bits[0] = input_port(name=empty, id=2)
  value: bits[8] = input_port(name=value, id=3)
  stored: bits[8] = register_read(register=state, id=4)
  update: () = register_write(value, register=state, reset=rst, id=5)
  empty_result: () = output_port(empty, name=empty_result, id=6)
  result: () = output_port(stored, name=result, id=7)
}

top block feedback(clk: clock, rst: bits[1], empty: bits[0], empty_result: bits[0], result: bits[8]) {
  instantiation component(block=delayed, kind=block)
  rst: bits[1] = input_port(name=rst, id=8)
  empty: bits[0] = input_port(name=empty, id=9)
  fed: bits[8] = instantiation_output(instantiation=component, port_name=result, id=10)
  empty_value: bits[0] = instantiation_output(instantiation=component, port_name=empty_result, id=11)
  reset_connected: () = instantiation_input(rst, instantiation=component, port_name=rst, id=12)
  empty_connected: () = instantiation_input(empty, instantiation=component, port_name=empty, id=13)
  value_connected: () = instantiation_input(fed, instantiation=component, port_name=value, id=14)
  empty_result: () = output_port(empty_value, name=empty_result, id=15)
  result: () = output_port(fed, name=result, id=16)
}
"#,
        );

        assert_eq!(verify_hierarchy(&package), Ok(()));
    }

    #[test]
    fn foreign_instances_remain_opaque_to_cycle_verification() {
        let package = parse_package(
            r#"package public_external_feedback

#[ffi_proto("""code_template: "external_cell {fn} (.value({value}), .result({return}));"
""")]
fn external_identity(value: bits[8] id=1) -> bits[8] {
  ret value: bits[8] = param(name=value, id=1)
}

top block feedback(result: bits[8]) {
  instantiation component(foreign_function=external_identity, kind=extern)
  fed: bits[8] = instantiation_output(instantiation=component, port_name=return, id=2)
  connected: () = instantiation_input(fed, instantiation=component, port_name=value, id=3)
  result: () = output_port(fed, name=result, id=4)
}
"#,
        );

        assert_eq!(verify_hierarchy(&package), Ok(()));
    }
}
