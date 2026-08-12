// SPDX-License-Identifier: Apache-2.0

//! Code-generation-option adapters for the native VAST helper library.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_vast::{Expr, VastError, VastFile};

use crate::XlsynthError;
use crate::vast_helpers_options::{CodegenOptions, TemplateVariable};

pub use xlsynth_vast::helpers::{RegisterDefinition, RegisterScope, Reset};

/// Replaces normalized template placeholders with their rendered expressions.
fn emit_template(template: &str, keys: &BTreeMap<TemplateVariable, String>) -> String {
    let mut template = template.to_string();
    for (key, value) in keys {
        let placeholder = format!("{{{{{key}}}}}");
        template = template.replace(&placeholder, value);
    }
    template
}

/// Validates and normalizes one template, including directly constructed
/// options.
fn normalize_register_template(
    name: &str,
    template: &str,
    expected: &[TemplateVariable],
) -> Result<String, XlsynthError> {
    let expected = expected
        .iter()
        .map(ToString::to_string)
        .collect::<BTreeSet<_>>();
    let mut found = BTreeSet::new();
    let mut normalized = String::with_capacity(template.len());
    let mut remaining = template;

    while let Some(start) = remaining.find("{{") {
        let prefix = &remaining[..start];
        if prefix.contains("}}") {
            return Err(XlsynthError(format!(
                "register template `{name}` contains an unmatched closing placeholder"
            )));
        }
        normalized.push_str(prefix);
        let placeholder = &remaining[start + 2..];
        let end = placeholder.find("}}").ok_or_else(|| {
            XlsynthError(format!(
                "register template `{name}` contains an unterminated placeholder"
            ))
        })?;
        let variable = placeholder[..end].trim();
        if variable.is_empty() || variable.contains('{') || variable.contains('}') {
            return Err(XlsynthError(format!(
                "register template `{name}` contains an invalid placeholder"
            )));
        }

        found.insert(variable.to_owned());
        normalized.push_str("{{");
        normalized.push_str(variable);
        normalized.push_str("}}");
        remaining = &placeholder[end + 2..];
    }
    if remaining.contains("}}") {
        return Err(XlsynthError(format!(
            "register template `{name}` contains an unmatched closing placeholder"
        )));
    }
    normalized.push_str(remaining);

    if found != expected {
        let expected = expected.into_iter().collect::<Vec<_>>().join(", ");
        let found = found.into_iter().collect::<Vec<_>>().join(", ");
        return Err(XlsynthError(format!(
            "register template `{name}` variables mismatch: expected {{{expected}}} \
             but found {{{found}}}"
        )));
    }

    Ok(normalized)
}

/// Resolves every required register template before mutating the output file.
fn prepare_register_templates(
    registers: &[RegisterDefinition],
    options: &CodegenOptions,
) -> Result<Vec<String>, XlsynthError> {
    registers
        .iter()
        .enumerate()
        .map(|(index, register)| {
            let (name, template, expected): (&str, Option<&String>, &[TemplateVariable]) =
                match (register.reset_value.is_some(), register.enable.is_some()) {
                    (true, true) => (
                        "reg_with_reset_with_en_template",
                        options.reg_with_reset_with_en_template.as_ref(),
                        &[
                            TemplateVariable::Reg,
                            TemplateVariable::Next,
                            TemplateVariable::Clock,
                            TemplateVariable::Reset,
                            TemplateVariable::ResetValue,
                            TemplateVariable::Enable,
                        ],
                    ),
                    (true, false) => (
                        "reg_with_reset_template",
                        options.reg_with_reset_template.as_ref(),
                        &[
                            TemplateVariable::Reg,
                            TemplateVariable::Next,
                            TemplateVariable::Clock,
                            TemplateVariable::Reset,
                            TemplateVariable::ResetValue,
                        ],
                    ),
                    (false, true) => (
                        "reg_with_en_template",
                        options.reg_with_en_template.as_ref(),
                        &[
                            TemplateVariable::Reg,
                            TemplateVariable::Next,
                            TemplateVariable::Clock,
                            TemplateVariable::Enable,
                        ],
                    ),
                    (false, false) => (
                        "reg_template",
                        options.reg_template.as_ref(),
                        &[
                            TemplateVariable::Reg,
                            TemplateVariable::Next,
                            TemplateVariable::Clock,
                        ],
                    ),
                };

            let template = template.ok_or_else(|| {
                XlsynthError(format!(
                    "missing register template `{name}` required by register {index}"
                ))
            })?;
            normalize_register_template(name, template, expected)
        })
        .collect()
}

/// Checks every input handle before configured templates modify the output.
fn validate_register_handles(
    clk: &Expr,
    reset: Option<Reset>,
    registers: &[RegisterDefinition],
    scope: RegisterScope,
    file: &VastFile,
) {
    file.emit_expression(clk);
    match scope {
        RegisterScope::Module(module) => {
            file.module_name(module);
        }
        RegisterScope::GenerateLoop(generate_loop) => {
            file.generate_genvar(generate_loop);
        }
    }
    if let Some(reset) = reset {
        file.emit_expression(&reset.signal);
    }
    for register in registers {
        file.emit_expression(&register.reg);
        file.emit_expression(&register.next);
        if let Some(reset_value) = register.reset_value {
            file.emit_expression(&reset_value);
        }
        if let Some(enable) = register.enable {
            file.emit_expression(&enable);
        }
    }
}

/// Emits one configured inline-Verilog template for each register definition.
fn emit_registers_with_templates(
    clk: &Expr,
    reset: Option<Reset>,
    registers: &[RegisterDefinition],
    file: &mut VastFile,
    scope: RegisterScope,
    templates: &[String],
) {
    for (register, template) in registers.iter().zip(templates) {
        let mut keys: BTreeMap<TemplateVariable, String> = BTreeMap::new();
        keys.insert(TemplateVariable::Clock, file.emit_expression(clk));
        if let Some(reset) = reset {
            keys.insert(TemplateVariable::Reset, file.emit_expression(&reset.signal));
        }
        keys.insert(TemplateVariable::Reg, file.emit_expression(&register.reg));
        keys.insert(TemplateVariable::Next, file.emit_expression(&register.next));
        if let Some(reset_value) = register.reset_value {
            keys.insert(
                TemplateVariable::ResetValue,
                file.emit_expression(&reset_value),
            );
        }
        if let Some(enable) = register.enable {
            keys.insert(TemplateVariable::Enable, file.emit_expression(&enable));
        }

        let inline_string = emit_template(template, &keys);
        let inline_statement = file.make_inline_verilog_statement(&format!("{inline_string};"));
        match scope {
            RegisterScope::Module(module) => {
                file.add_member_inline_statement(module, inline_statement);
            }
            RegisterScope::GenerateLoop(generate_loop) => {
                file.generate_add_inline_statement(generate_loop, &inline_statement);
            }
        }
    }
}

/// Adds registers through configured templates or the native VAST helper.
pub fn add_registers(
    clk: &Expr,
    reset: Option<Reset>,
    registers: &[RegisterDefinition],
    scope: RegisterScope,
    file: &mut VastFile,
    options: Option<&CodegenOptions>,
) -> Result<(), XlsynthError> {
    if registers.is_empty() {
        return Ok(());
    }

    if reset.is_none()
        && registers
            .iter()
            .any(|register| register.reset_value.is_some())
    {
        return Err(
            VastError("reset signal is required when a register has a reset value".into()).into(),
        );
    }

    if let Some(opts) = options
        && [
            &opts.reg_template,
            &opts.reg_with_en_template,
            &opts.reg_with_reset_template,
            &opts.reg_with_reset_with_en_template,
        ]
        .iter()
        .any(|template| template.is_some())
    {
        let templates = prepare_register_templates(registers, opts)?;
        validate_register_handles(clk, reset, registers, scope, file);
        emit_registers_with_templates(clk, reset, registers, file, scope, &templates);
        return Ok(());
    }

    xlsynth_vast::helpers::add_registers(clk, reset, registers, scope, file).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use xlsynth_vast::{LiteralFormat, VastFileType};

    use super::*;

    #[test]
    fn options_free_register_generation_delegates_to_native_helpers() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("delegated_register");
        let scalar = file.make_scalar_type();
        let clock = file.add_input(module, "clock", &scalar);
        let reset = file.add_input(module, "reset", &scalar);
        let enable = file.add_input(module, "enable", &scalar);
        let data = file.add_input(module, "data", &scalar);
        let register = file
            .add_logic(module, "state", &scalar)
            .expect("register name is unique");
        let reset_value = file
            .make_literal("bits[1]:0", &LiteralFormat::Hex)
            .expect("valid reset literal");

        add_registers(
            &clock.to_expr(),
            Some(Reset {
                signal: reset.to_expr(),
                active_low: false,
            }),
            &[RegisterDefinition {
                reg: register.to_expr(),
                next: data.to_expr(),
                reset_value: Some(reset_value),
                enable: Some(enable.to_expr()),
            }],
            RegisterScope::Module(module),
            &mut file,
            None,
        )
        .expect("native register helper succeeds");

        let expected = r#"module delegated_register(
  input wire clock,
  input wire reset,
  input wire enable,
  input wire data
);
  logic state;
  always_ff @ (posedge clock) begin
    if (reset) begin
      state <= 1'h0;
    end else begin
      state <= enable ? data : state;
    end
  end
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }

    #[test]
    fn configured_templates_cover_plain_enabled_reset_and_reset_enabled_registers() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("templated_registers");
        let scalar = file.make_scalar_type();
        let clock = file.add_input(module, "clock", &scalar);
        let reset = file.add_input(module, "reset", &scalar);
        let enable = file.add_input(module, "enable", &scalar);
        let data = file.add_input(module, "data", &scalar);
        let plain = file.add_logic(module, "plain", &scalar).unwrap();
        let enabled = file.add_logic(module, "enabled", &scalar).unwrap();
        let resetting = file.add_logic(module, "resetting", &scalar).unwrap();
        let resetting_enabled = file
            .add_logic(module, "resetting_enabled", &scalar)
            .unwrap();
        let zero = file.make_literal("bits[1]:0", &LiteralFormat::Hex).unwrap();

        let options = CodegenOptions {
            reg_template: Some("PLAIN({{clock}}, {{reg}}, {{next}})".into()),
            reg_with_en_template: Some("EN({{clock}}, {{reg}}, {{next}}, {{en}})".into()),
            reg_with_reset_template: Some(
                "RESET({{clock}}, {{reset}}, {{reg}}, {{next}}, {{reset_value}})".into(),
            ),
            reg_with_reset_with_en_template: Some(
                "RESET_EN({{clock}}, {{reset}}, {{reg}}, {{next}}, {{reset_value}}, {{en}})".into(),
            ),
            ..CodegenOptions::default()
        };
        let registers = [
            RegisterDefinition {
                reg: plain.to_expr(),
                next: data.to_expr(),
                reset_value: None,
                enable: None,
            },
            RegisterDefinition {
                reg: enabled.to_expr(),
                next: data.to_expr(),
                reset_value: None,
                enable: Some(enable.to_expr()),
            },
            RegisterDefinition {
                reg: resetting.to_expr(),
                next: data.to_expr(),
                reset_value: Some(zero),
                enable: None,
            },
            RegisterDefinition {
                reg: resetting_enabled.to_expr(),
                next: data.to_expr(),
                reset_value: Some(zero),
                enable: Some(enable.to_expr()),
            },
        ];

        add_registers(
            &clock.to_expr(),
            Some(Reset {
                signal: reset.to_expr(),
                active_low: false,
            }),
            &registers,
            RegisterScope::Module(module),
            &mut file,
            Some(&options),
        )
        .expect("all configured register templates render");

        let expected = r#"module templated_registers(
  input wire clock,
  input wire reset,
  input wire enable,
  input wire data
);
  logic plain;
  logic enabled;
  logic resetting;
  logic resetting_enabled;
  PLAIN(clock, plain, data);
  EN(clock, enabled, data, enable);
  RESET(clock, reset, resetting, data, 1'h0);
  RESET_EN(clock, reset, resetting_enabled, data, 1'h0, enable);
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }

    #[test]
    fn configured_register_templates_can_be_emitted_inside_generate_loops() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("templated_generate");
        let scalar = file.make_scalar_type();
        let clock = file.add_input(module, "clock", &scalar);
        let data = file.add_input(module, "data", &scalar);
        let register = file.add_logic(module, "state", &scalar).unwrap();
        let zero = file.make_unsized_decimal_literal(0);
        let one = file.make_unsized_decimal_literal(1);
        let generate = file.add_generate_loop(module, "index", &zero, &one, Some("lanes"));
        let options = CodegenOptions {
            reg_template: Some("REGISTER({{clock}}, {{reg}}, {{next}})".into()),
            ..CodegenOptions::default()
        };

        add_registers(
            &clock.to_expr(),
            None,
            &[RegisterDefinition {
                reg: register.to_expr(),
                next: data.to_expr(),
                reset_value: None,
                enable: None,
            }],
            RegisterScope::GenerateLoop(generate),
            &mut file,
            Some(&options),
        )
        .expect("generate-scope register template renders");

        let expected = r#"module templated_generate(
  input wire clock,
  input wire data
);
  logic state;
  for (genvar index = 0; index < 1; index = index + 1) begin : lanes
    REGISTER(clock, state, data);
  end
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }

    #[test]
    fn missing_template_variant_is_rejected_before_any_register_is_emitted() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("partial_templates");
        let scalar = file.make_scalar_type();
        let clock = file.add_input(module, "clock", &scalar);
        let enable = file.add_input(module, "enable", &scalar);
        let data = file.add_input(module, "data", &scalar);
        let plain = file.add_logic(module, "plain", &scalar).unwrap();
        let enabled = file.add_logic(module, "enabled", &scalar).unwrap();
        let options = CodegenOptions {
            reg_template: Some("REGISTER({{clock}}, {{reg}}, {{next}})".into()),
            ..CodegenOptions::default()
        };
        let registers = [
            RegisterDefinition {
                reg: plain.to_expr(),
                next: data.to_expr(),
                reset_value: None,
                enable: None,
            },
            RegisterDefinition {
                reg: enabled.to_expr(),
                next: data.to_expr(),
                reset_value: None,
                enable: Some(enable.to_expr()),
            },
        ];
        let before = file.emit();

        let error = add_registers(
            &clock.to_expr(),
            None,
            &registers,
            RegisterScope::Module(module),
            &mut file,
            Some(&options),
        )
        .expect_err("each register shape requires a matching configured template");

        assert_eq!(
            error.0,
            "missing register template `reg_with_en_template` required by register 1"
        );
        assert_eq!(file.emit(), before);
    }

    #[test]
    fn directly_constructed_malformed_templates_are_rejected_without_mutation() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("malformed_templates");
        let scalar = file.make_scalar_type();
        let clock = file.add_input(module, "clock", &scalar);
        let data = file.add_input(module, "data", &scalar);
        let register = file.add_logic(module, "state", &scalar).unwrap();
        let definition = RegisterDefinition {
            reg: register.to_expr(),
            next: data.to_expr(),
            reset_value: None,
            enable: None,
        };
        let cases = [
            (
                "REGISTER({{clock}}, {{reg}})",
                "register template `reg_template` variables mismatch: expected \
                 {clock, next, reg} but found {clock, reg}",
            ),
            (
                "REGISTER({{clock}}, {{reg}}, {{unknown}})",
                "register template `reg_template` variables mismatch: expected \
                 {clock, next, reg} but found {clock, reg, unknown}",
            ),
            (
                "REGISTER({{clock}}, {{reg}}, {{next)",
                "register template `reg_template` contains an unterminated placeholder",
            ),
            (
                "REGISTER({{clock}}, {{reg}}, {{next}}}})",
                "register template `reg_template` contains an unmatched closing placeholder",
            ),
        ];
        let before = file.emit();

        for (template, expected) in cases {
            let options = CodegenOptions {
                reg_template: Some(template.into()),
                ..CodegenOptions::default()
            };
            let error = add_registers(
                &clock.to_expr(),
                None,
                &[definition],
                RegisterScope::Module(module),
                &mut file,
                Some(&options),
            )
            .expect_err("invalid register templates are rejected");

            assert_eq!(error.0, expected);
            assert_eq!(file.emit(), before);
        }
    }

    #[test]
    fn directly_constructed_templates_normalize_placeholder_whitespace() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("normalized_template");
        let scalar = file.make_scalar_type();
        let clock = file.add_input(module, "clock", &scalar);
        let data = file.add_input(module, "data", &scalar);
        let register = file.add_logic(module, "state", &scalar).unwrap();
        let options = CodegenOptions {
            reg_template: Some("REGISTER({{ clock }}, {{ reg }}, {{ next }})".into()),
            ..CodegenOptions::default()
        };

        add_registers(
            &clock.to_expr(),
            None,
            &[RegisterDefinition {
                reg: register.to_expr(),
                next: data.to_expr(),
                reset_value: None,
                enable: None,
            }],
            RegisterScope::Module(module),
            &mut file,
            Some(&options),
        )
        .expect("directly constructed placeholders are normalized");

        let expected = r#"module normalized_template(
  input wire clock,
  input wire data
);
  logic state;
  REGISTER(clock, state, data);
endmodule
"#;
        assert_eq!(file.emit(), expected);
    }

    #[test]
    fn missing_reset_returns_an_error_in_native_and_template_adapter_paths() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("missing_reset");
        let scalar = file.make_scalar_type();
        let clock = file.add_input(module, "clock", &scalar);
        let data = file.add_input(module, "data", &scalar);
        let register = file.add_logic(module, "state", &scalar).unwrap();
        let reset_value = file.make_literal("bits[1]:0", &LiteralFormat::Hex).unwrap();
        let definition = RegisterDefinition {
            reg: register.to_expr(),
            next: data.to_expr(),
            reset_value: Some(reset_value),
            enable: None,
        };
        let options = CodegenOptions {
            reg_with_reset_template: Some(
                "RESET({{clock}}, {{reset}}, {{reg}}, {{next}}, {{reset_value}})".into(),
            ),
            ..CodegenOptions::default()
        };
        let before = file.emit();

        for options in [None, Some(&options)] {
            let error = add_registers(
                &clock.to_expr(),
                None,
                &[definition],
                RegisterScope::Module(module),
                &mut file,
                options,
            )
            .expect_err("register reset values require a reset signal");

            assert_eq!(
                error.0,
                "reset signal is required when a register has a reset value"
            );
            assert_eq!(file.emit(), before);
        }
    }

    #[test]
    fn foreign_later_register_is_rejected_before_templates_mutate_the_module() {
        let mut foreign_file = VastFile::new(VastFileType::SystemVerilog);
        let foreign_value = foreign_file.make_unsized_decimal_literal(1);
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("foreign_template");
        let scalar = file.make_scalar_type();
        let clock = file.add_input(module, "clock", &scalar);
        let data = file.add_input(module, "data", &scalar);
        let first = file.add_logic(module, "first", &scalar).unwrap();
        let second = file.add_logic(module, "second", &scalar).unwrap();
        let options = CodegenOptions {
            reg_template: Some("REGISTER({{clock}}, {{reg}}, {{next}})".into()),
            ..CodegenOptions::default()
        };
        let registers = [
            RegisterDefinition {
                reg: first.to_expr(),
                next: data.to_expr(),
                reset_value: None,
                enable: None,
            },
            RegisterDefinition {
                reg: second.to_expr(),
                next: foreign_value,
                reset_value: None,
                enable: None,
            },
        ];
        let before = file.emit();

        let result = catch_unwind(AssertUnwindSafe(|| {
            add_registers(
                &clock.to_expr(),
                None,
                &registers,
                RegisterScope::Module(module),
                &mut file,
                Some(&options),
            )
        }));

        assert!(result.is_err(), "foreign handles are rejected");
        assert_eq!(file.emit(), before);
    }
}
