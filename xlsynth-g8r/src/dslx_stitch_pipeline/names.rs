// SPDX-License-Identifier: Apache-2.0

//! Name validation for DSLX stitch-pipeline wrapper generation.

use std::collections::BTreeMap;

use xlsynth::XlsynthError;

use super::StitchPipelineOptions;
use super::build_pipeline::{
    flop_reg_name, slice_wire_name, stage_instance_name, stage_output_wire_name, valid_reg_name,
};

const NAME_VALIDATION_PREFIX: &str = "name validation failed";

const SYSTEM_VERILOG_KEYWORDS: &[&str] = &[
    "accept_on",
    "alias",
    "always",
    "always_comb",
    "always_ff",
    "always_latch",
    "and",
    "assert",
    "assign",
    "assume",
    "automatic",
    "begin",
    "bind",
    "bins",
    "bit",
    "break",
    "buf",
    "bufif0",
    "bufif1",
    "byte",
    "case",
    "casex",
    "casez",
    "cell",
    "chandle",
    "class",
    "clocking",
    "cmos",
    "config",
    "const",
    "constraint",
    "context",
    "continue",
    "cover",
    "covergroup",
    "coverpoint",
    "cross",
    "deassign",
    "default",
    "defparam",
    "design",
    "disable",
    "dist",
    "do",
    "edge",
    "else",
    "end",
    "endcase",
    "endclass",
    "endclocking",
    "endconfig",
    "endfunction",
    "endgenerate",
    "endgroup",
    "endmodule",
    "endpackage",
    "endprimitive",
    "endprogram",
    "endproperty",
    "endspecify",
    "endsequence",
    "endtable",
    "endtask",
    "enum",
    "event",
    "eventually",
    "expect",
    "export",
    "extends",
    "extern",
    "final",
    "first_match",
    "for",
    "force",
    "foreach",
    "forever",
    "fork",
    "forkjoin",
    "function",
    "generate",
    "genvar",
    "global",
    "highz0",
    "highz1",
    "if",
    "iff",
    "ifnone",
    "ignore_bins",
    "illegal_bins",
    "implements",
    "implies",
    "import",
    "incdir",
    "include",
    "initial",
    "inout",
    "input",
    "inside",
    "instance",
    "int",
    "integer",
    "interconnect",
    "interface",
    "intersect",
    "join",
    "join_any",
    "join_none",
    "large",
    "let",
    "liblist",
    "library",
    "local",
    "localparam",
    "logic",
    "longint",
    "macromodule",
    "matches",
    "medium",
    "modport",
    "module",
    "nand",
    "negedge",
    "nettype",
    "new",
    "nexttime",
    "nmos",
    "nor",
    "noshowcancelled",
    "not",
    "notif0",
    "notif1",
    "null",
    "or",
    "output",
    "package",
    "packed",
    "parameter",
    "pmos",
    "posedge",
    "primitive",
    "priority",
    "program",
    "property",
    "protected",
    "pull0",
    "pull1",
    "pulldown",
    "pullup",
    "pulsestyle_ondetect",
    "pulsestyle_onevent",
    "pure",
    "rand",
    "randc",
    "randcase",
    "randsequence",
    "rcmos",
    "real",
    "realtime",
    "ref",
    "reg",
    "reject_on",
    "release",
    "repeat",
    "restrict",
    "return",
    "rnmos",
    "rpmos",
    "rtran",
    "rtranif0",
    "rtranif1",
    "s_always",
    "s_eventually",
    "s_nexttime",
    "s_until",
    "s_until_with",
    "scalared",
    "sequence",
    "shortint",
    "shortreal",
    "showcancelled",
    "signed",
    "small",
    "soft",
    "solve",
    "specify",
    "specparam",
    "static",
    "string",
    "strong",
    "strong0",
    "strong1",
    "struct",
    "super",
    "supply0",
    "supply1",
    "sync_accept_on",
    "sync_reject_on",
    "table",
    "tagged",
    "task",
    "this",
    "throughout",
    "time",
    "timeprecision",
    "timeunit",
    "tran",
    "tranif0",
    "tranif1",
    "tri",
    "tri0",
    "tri1",
    "triand",
    "trior",
    "trireg",
    "type",
    "typedef",
    "union",
    "unique",
    "unique0",
    "unsigned",
    "until",
    "until_with",
    "untyped",
    "use",
    "uwire",
    "var",
    "vectored",
    "virtual",
    "void",
    "wait",
    "wait_order",
    "wand",
    "weak",
    "weak0",
    "weak1",
    "while",
    "wildcard",
    "wire",
    "with",
    "within",
    "wor",
    "xnor",
    "xor",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StageNameInfo {
    pub(crate) stage_name: String,
    pub(crate) params: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StageSignature {
    pub(crate) output_port_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NameRole<'a> {
    WrapperModule,
    StageModule { stage: &'a str },
    StageParameter { stage: &'a str, parameter: &'a str },
    ControlPort { role: &'static str },
    WrapperOutputPort,
}

pub(crate) fn is_system_verilog_keyword(name: &str) -> bool {
    SYSTEM_VERILOG_KEYWORDS.contains(&name)
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

pub(crate) fn collect_stage_name_infos(
    ir: &xlsynth::ir_package::IrPackage,
    stages: &[(String, String)],
) -> Result<(Vec<StageNameInfo>, Vec<StageSignature>), XlsynthError> {
    let mut infos = Vec::with_capacity(stages.len());
    let mut signatures = Vec::with_capacity(stages.len());
    for (stage_name, stage_mangled) in stages {
        let func = ir.get_function(stage_mangled)?;
        let fty = func.get_type()?;
        let mut params = Vec::with_capacity(fty.param_count() as usize);
        for i in 0..fty.param_count() {
            params.push(func.param_name(i)?);
        }
        infos.push(StageNameInfo {
            stage_name: stage_name.clone(),
            params,
        });
        signatures.push(StageSignature {
            output_port_name: "out".to_string(),
        });
    }
    Ok((infos, signatures))
}

pub(crate) fn validate_stitch_pipeline_names(
    stages: &[StageNameInfo],
    signatures: &[StageSignature],
    opts: &StitchPipelineOptions,
) -> Result<(), XlsynthError> {
    if stages.is_empty() {
        return name_error("no pipeline stages found during name validation");
    }
    if signatures.len() != stages.len() {
        return name_error(format!(
            "internal error: name validation saw {} stage signatures for {} stages",
            signatures.len(),
            stages.len()
        ));
    }

    validate_simple_verilog_identifier(opts.output_module_name, NameRole::WrapperModule)?;

    let mut module_namespace = BTreeMap::new();
    record_emitted_name(
        &mut module_namespace,
        opts.output_module_name,
        format!("wrapper module `{}`", opts.output_module_name),
    )?;

    for stage in stages {
        validate_simple_verilog_identifier(
            &stage.stage_name,
            NameRole::StageModule {
                stage: &stage.stage_name,
            },
        )?;
        record_emitted_name(
            &mut module_namespace,
            &stage.stage_name,
            format!("stage module `{}`", stage.stage_name),
        )?;
    }

    validate_simple_verilog_identifier(
        "clk",
        NameRole::ControlPort {
            role: "clock control",
        },
    )?;
    if let Some(reset_signal) = opts.reset_signal {
        validate_simple_verilog_identifier(
            reset_signal,
            NameRole::ControlPort {
                role: "reset control",
            },
        )?;
    }
    if let Some(input_valid_signal) = opts.input_valid_signal {
        validate_simple_verilog_identifier(
            input_valid_signal,
            NameRole::ControlPort {
                role: "input-valid control",
            },
        )?;
    }
    if let Some(output_valid_signal) = opts.output_valid_signal {
        validate_simple_verilog_identifier(
            output_valid_signal,
            NameRole::ControlPort {
                role: "output-valid control",
            },
        )?;
    }

    let final_output_name = signatures
        .last()
        .map(|signature| signature.output_port_name.as_str())
        .unwrap_or("out");
    validate_simple_verilog_identifier(final_output_name, NameRole::WrapperOutputPort)?;

    let mut reserved = BTreeMap::new();
    insert_reserved(&mut reserved, "clk", "reserved wrapper/control port `clk`");
    if let Some(reset_signal) = opts.reset_signal {
        insert_reserved(
            &mut reserved,
            reset_signal,
            format!("reserved wrapper/control port `{reset_signal}`"),
        );
    }
    if let Some(input_valid_signal) = opts.input_valid_signal {
        insert_reserved(
            &mut reserved,
            input_valid_signal,
            format!("reserved wrapper/control port `{input_valid_signal}`"),
        );
    }
    if let Some(output_valid_signal) = opts.output_valid_signal {
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

    for (stage_index, stage) in stages.iter().enumerate() {
        for param in &stage.params {
            validate_simple_verilog_identifier(
                param,
                NameRole::StageParameter {
                    stage: &stage.stage_name,
                    parameter: param,
                },
            )?;
            if stage_index == 0 {
                if let Some(reserved_owner) = reserved.get(param) {
                    return name_error(format!(
                        "stage parameter `{}.{param}` cannot be emitted as-is: it collides with {reserved_owner}",
                        stage.stage_name
                    ));
                }
            }
        }
    }

    let mut wrapper_namespace = BTreeMap::new();
    record_emitted_name(
        &mut wrapper_namespace,
        "clk",
        "wrapper/control port `clk` (clock)".to_string(),
    )?;
    if let Some(reset_signal) = opts.reset_signal {
        record_emitted_name(
            &mut wrapper_namespace,
            reset_signal,
            format!("wrapper/control port `{reset_signal}` (reset)"),
        )?;
    }
    if let Some(input_valid_signal) = opts.input_valid_signal {
        record_emitted_name(
            &mut wrapper_namespace,
            input_valid_signal,
            format!("wrapper/control port `{input_valid_signal}` (input valid)"),
        )?;
    }
    if let Some(output_valid_signal) = opts.output_valid_signal {
        record_emitted_name(
            &mut wrapper_namespace,
            output_valid_signal,
            format!("wrapper/control port `{output_valid_signal}` (output valid)"),
        )?;
    }
    for param in &stages[0].params {
        record_emitted_name(
            &mut wrapper_namespace,
            param,
            format!(
                "first-stage data input port `{}.{param}`",
                stages[0].stage_name
            ),
        )?;
    }
    record_emitted_name(
        &mut wrapper_namespace,
        final_output_name,
        format!("wrapper output port `{final_output_name}`"),
    )?;

    validate_generated_wrapper_names(stages, final_output_name, opts, &mut wrapper_namespace)
}

fn validate_generated_wrapper_names(
    stages: &[StageNameInfo],
    final_output_name: &str,
    opts: &StitchPipelineOptions,
    wrapper_namespace: &mut BTreeMap<String, String>,
) -> Result<(), XlsynthError> {
    let mut next_pipe_stage_number = 0;
    let mut current_names = stages[0].params.clone();
    let has_valid_signal = opts.input_valid_signal.is_some();

    if opts.flop_inputs {
        record_flop_layer_names(
            wrapper_namespace,
            next_pipe_stage_number,
            &current_names,
            has_valid_signal,
        )?;
        next_pipe_stage_number += 1;
    }

    for (stage_index, stage) in stages.iter().enumerate() {
        record_emitted_name(
            wrapper_namespace,
            &stage_output_wire_name(stage_index),
            format!("generated stage output wire for `{}`", stage.stage_name),
        )?;
        record_emitted_name(
            wrapper_namespace,
            &stage_instance_name(stage_index),
            format!("generated instance name for `{}`", stage.stage_name),
        )?;

        let last_stage = stage_index + 1 == stages.len();
        let dest_names = if last_stage {
            vec![final_output_name.to_string()]
        } else {
            stages[stage_index + 1].params.clone()
        };

        if dest_names.len() > 1 {
            for dest_name in &dest_names {
                record_emitted_name(
                    wrapper_namespace,
                    &slice_wire_name(next_pipe_stage_number, dest_name),
                    format!(
                        "generated slice wire for `{dest_name}` at pipeline boundary {next_pipe_stage_number}"
                    ),
                )?;
            }
        }

        current_names = dest_names;

        if !last_stage {
            record_flop_layer_names(
                wrapper_namespace,
                next_pipe_stage_number,
                &current_names,
                has_valid_signal,
            )?;
            next_pipe_stage_number += 1;
        }
    }

    if opts.flop_outputs {
        record_flop_layer_names(
            wrapper_namespace,
            next_pipe_stage_number,
            &current_names,
            has_valid_signal,
        )?;
    }

    Ok(())
}

fn record_flop_layer_names(
    namespace: &mut BTreeMap<String, String>,
    pipe_stage_number: u32,
    names: &[String],
    has_valid_signal: bool,
) -> Result<(), XlsynthError> {
    let mut sorted_names = names.to_vec();
    sorted_names.sort();
    for name in &sorted_names {
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
    c.is_ascii_alphanumeric() || c == '_'
}
