// SPDX-License-Identifier: Apache-2.0

//! Generated names and namespace validation for DSLX pipeline wrappers.

use std::collections::BTreeMap;

use xlsynth::XlsynthError;
use xlsynth_vast::is_system_verilog_keyword;

use super::layout::{BoundaryPort, PipelineLayout};
use crate::aig::SequentialPipelineOptions;

const NAME_VALIDATION_PREFIX: &str = "name validation failed";

pub(super) fn stage_instance_name(stage_index: usize) -> String {
    format!("stage_{stage_index}")
}

pub(super) fn stage_output_wire_name(stage_index: usize) -> String {
    format!("stage_{stage_index}_out_comb")
}

pub(super) fn flop_reg_name(pipe_stage_number: usize, name: &str) -> String {
    format!("p{pipe_stage_number}_{name}")
}

pub(super) fn slice_wire_name(pipe_stage_number: usize, dest_name: &str) -> String {
    format!("p{pipe_stage_number}_{dest_name}_comb")
}

pub(super) fn valid_reg_name(pipe_stage_number: usize) -> String {
    format!("p{pipe_stage_number}_valid")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NameRole<'a> {
    WrapperModule,
    StageModule { stage: &'a str },
    StageParameter { stage: &'a str, parameter: &'a str },
    ControlPort { role: &'static str },
    WrapperOutputPort,
}

pub(crate) fn validate_simple_verilog_identifier(
    name: &str,
    role: NameRole<'_>,
) -> Result<(), XlsynthError> {
    let label = describe_role(role, name);
    if name.is_empty() {
        return name_error(format!("{label} cannot be emitted as-is: it is empty"));
    }

    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !is_simple_identifier_start(first) || !chars.all(is_simple_identifier_body) {
        return name_error(format!(
            "{label} cannot be emitted as-is: it must match [A-Za-z_][A-Za-z0-9_]*"
        ));
    }

    // Intentionally use the SystemVerilog keyword superset for every output
    // dialect so accepted stitch-pipeline names remain valid if the caller
    // switches between Verilog and SystemVerilog emission.
    if is_system_verilog_keyword(name) {
        return name_error(format!(
            "{label} cannot be emitted as-is: it is a SystemVerilog keyword"
        ));
    }

    Ok(())
}

/// Checks module, port and generated names for the resolved wrapper layout.
pub(crate) fn validate_stitch_pipeline_names(
    stages: &[(String, String)],
    layout: &PipelineLayout,
    opts: &SequentialPipelineOptions,
) -> Result<(), XlsynthError> {
    if stages.is_empty() {
        return name_error("no pipeline stages found during name validation");
    }
    if layout.stage_inputs.len() != stages.len() {
        return name_error(format!(
            "internal error: name validation saw {} stage signatures for {} stages",
            layout.stage_inputs.len(),
            stages.len()
        ));
    }

    validate_simple_verilog_identifier(&opts.name, NameRole::WrapperModule)?;

    let mut module_namespace = BTreeMap::new();
    record_emitted_name(
        &mut module_namespace,
        &opts.name,
        format!("wrapper module `{}`", opts.name),
    )?;

    for (stage_name, _) in stages {
        validate_simple_verilog_identifier(
            stage_name,
            NameRole::StageModule { stage: stage_name },
        )?;
        record_emitted_name(
            &mut module_namespace,
            stage_name,
            format!("stage module `{stage_name}`"),
        )?;
    }

    validate_simple_verilog_identifier(
        &opts.clock.name,
        NameRole::ControlPort {
            role: "clock control",
        },
    )?;
    if let Some(reset_signal) = opts.reset_signal.as_deref() {
        validate_simple_verilog_identifier(
            reset_signal,
            NameRole::ControlPort {
                role: "reset control",
            },
        )?;
    }
    if let Some(input_valid_signal) = opts.input_valid_signal.as_deref() {
        validate_simple_verilog_identifier(
            input_valid_signal,
            NameRole::ControlPort {
                role: "input-valid control",
            },
        )?;
    }
    if let Some(output_valid_signal) = opts.output_valid_signal.as_deref() {
        validate_simple_verilog_identifier(
            output_valid_signal,
            NameRole::ControlPort {
                role: "output-valid control",
            },
        )?;
    }

    let final_output_name = layout.output.ports[0].name.as_str();
    validate_simple_verilog_identifier(final_output_name, NameRole::WrapperOutputPort)?;

    let mut reserved = BTreeMap::new();
    insert_reserved(
        &mut reserved,
        &opts.clock.name,
        format!("reserved wrapper/control port `{}`", opts.clock.name),
    );
    if let Some(reset_signal) = opts.reset_signal.as_deref() {
        insert_reserved(
            &mut reserved,
            reset_signal,
            format!("reserved wrapper/control port `{reset_signal}`"),
        );
    }
    if let Some(input_valid_signal) = opts.input_valid_signal.as_deref() {
        insert_reserved(
            &mut reserved,
            input_valid_signal,
            format!("reserved wrapper/control port `{input_valid_signal}`"),
        );
    }
    if let Some(output_valid_signal) = opts.output_valid_signal.as_deref() {
        insert_reserved(
            &mut reserved,
            output_valid_signal,
            format!("reserved wrapper/control port `{output_valid_signal}`"),
        );
    }
    insert_reserved(
        &mut reserved,
        final_output_name,
        format!("reserved wrapper/output port `{final_output_name}`"),
    );

    for (stage_index, (stage_name, _)) in stages.iter().enumerate() {
        for port in &layout.stage_inputs[stage_index].ports {
            let param = port.name.as_str();
            validate_simple_verilog_identifier(
                param,
                NameRole::StageParameter {
                    stage: stage_name,
                    parameter: param,
                },
            )?;
            if stage_index == 0 {
                if let Some(reserved_owner) = reserved.get(param) {
                    return name_error(format!(
                        "stage parameter `{stage_name}.{param}` cannot be emitted as-is: it collides with {reserved_owner}"
                    ));
                }
            }
        }
    }

    let mut wrapper_namespace = BTreeMap::new();
    record_emitted_name(
        &mut wrapper_namespace,
        &opts.clock.name,
        format!("wrapper/control port `{}` (clock)", opts.clock.name),
    )?;
    if let Some(reset_signal) = opts.reset_signal.as_deref() {
        record_emitted_name(
            &mut wrapper_namespace,
            reset_signal,
            format!("wrapper/control port `{reset_signal}` (reset)"),
        )?;
    }
    if let Some(input_valid_signal) = opts.input_valid_signal.as_deref() {
        record_emitted_name(
            &mut wrapper_namespace,
            input_valid_signal,
            format!("wrapper/control port `{input_valid_signal}` (input valid)"),
        )?;
    }
    if let Some(output_valid_signal) = opts.output_valid_signal.as_deref() {
        record_emitted_name(
            &mut wrapper_namespace,
            output_valid_signal,
            format!("wrapper/control port `{output_valid_signal}` (output valid)"),
        )?;
    }
    for port in &layout.stage_inputs[0].ports {
        record_emitted_name(
            &mut wrapper_namespace,
            &port.name,
            format!(
                "first-stage data input port `{}.{}`",
                stages[0].0, port.name
            ),
        )?;
    }
    record_emitted_name(
        &mut wrapper_namespace,
        final_output_name,
        format!("wrapper output port `{final_output_name}`"),
    )?;

    validate_generated_wrapper_names(
        stages,
        layout,
        opts.input_valid_signal.is_some(),
        &mut wrapper_namespace,
    )
}

/// Records each instance, wire and register in the order the wrapper emits
/// them.
fn validate_generated_wrapper_names(
    stages: &[(String, String)],
    layout: &PipelineLayout,
    has_valid_signal: bool,
    wrapper_namespace: &mut BTreeMap<String, String>,
) -> Result<(), XlsynthError> {
    let input = &layout.stage_inputs[0];
    if let Some(layer) = input.register_layer {
        record_flop_layer_names(wrapper_namespace, layer, &input.ports, has_valid_signal)?;
    }

    for (stage_index, (stage_name, _)) in stages.iter().enumerate() {
        record_emitted_name(
            wrapper_namespace,
            &stage_output_wire_name(stage_index),
            format!("generated stage output wire for `{stage_name}`"),
        )?;
        record_emitted_name(
            wrapper_namespace,
            &stage_instance_name(stage_index),
            format!("generated instance name for `{stage_name}`"),
        )?;

        let boundary = layout.after_stage(stage_index);
        let Some(layer) = boundary.register_layer else {
            // The unregistered wrapper output needs no slices or registers.
            continue;
        };
        if boundary.ports.len() > 1 {
            for port in &boundary.ports {
                let dest_name = &port.name;
                record_emitted_name(
                    wrapper_namespace,
                    &slice_wire_name(layer, dest_name),
                    format!("generated slice wire for `{dest_name}` at pipeline boundary {layer}"),
                )?;
            }
        }
        record_flop_layer_names(wrapper_namespace, layer, &boundary.ports, has_valid_signal)?;
    }

    Ok(())
}

fn record_flop_layer_names(
    namespace: &mut BTreeMap<String, String>,
    pipe_stage_number: usize,
    ports: &[BoundaryPort],
    has_valid_signal: bool,
) -> Result<(), XlsynthError> {
    let mut sorted_names: Vec<&str> = ports.iter().map(|port| port.name.as_str()).collect();
    sorted_names.sort_unstable();
    for name in sorted_names {
        record_emitted_name(
            namespace,
            &flop_reg_name(pipe_stage_number, name),
            format!(
                "generated data flop register for `{name}` at pipeline boundary {pipe_stage_number}"
            ),
        )?;
    }
    if has_valid_signal {
        record_emitted_name(
            namespace,
            &valid_reg_name(pipe_stage_number),
            format!("generated valid flop register at pipeline boundary {pipe_stage_number}"),
        )?;
    }
    Ok(())
}

fn insert_reserved(reserved: &mut BTreeMap<String, String>, name: &str, owner: impl Into<String>) {
    reserved
        .entry(name.to_string())
        .or_insert_with(|| owner.into());
}

fn record_emitted_name(
    namespace: &mut BTreeMap<String, String>,
    name: &str,
    owner: String,
) -> Result<(), XlsynthError> {
    if let Some(existing_owner) = namespace.get(name) {
        return name_error(format!(
            "`{name}` cannot be emitted as-is: {owner} collides with {existing_owner}"
        ));
    }
    namespace.insert(name.to_string(), owner);
    Ok(())
}

fn name_error<T>(message: impl Into<String>) -> Result<T, XlsynthError> {
    Err(XlsynthError(format!(
        "{NAME_VALIDATION_PREFIX}: {}",
        message.into()
    )))
}

fn describe_role(role: NameRole<'_>, name: &str) -> String {
    match role {
        NameRole::WrapperModule => format!("wrapper module name `{name}`"),
        NameRole::StageModule { stage } => format!("stage module `{stage}`"),
        NameRole::StageParameter { stage, parameter } => {
            format!("stage parameter `{stage}.{parameter}`")
        }
        NameRole::ControlPort { role } => format!("{role} port `{name}`"),
        NameRole::WrapperOutputPort => format!("wrapper output port `{name}`"),
    }
}

fn is_simple_identifier_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_simple_identifier_body(c: char) -> bool {
    // This intentionally excludes `$`: stitch-pipeline keeps emitted names in
    // the simple DSLX/Verilog/SystemVerilog identifier intersection.
    c.is_ascii_alphanumeric() || c == '_'
}
