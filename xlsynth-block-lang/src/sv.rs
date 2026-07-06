// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_pir::ir::{InstantiationKind, MemberType, Node, NodePayload, Package, PackageMember};
use xlsynth_pir::ir_value_utils::zero_ir_value_for_type;
use xlsynth_pir::ir_verify::verify_package;

/// Returns whether `name` can be emitted verbatim as an unescaped
/// SystemVerilog identifier.
///
/// DSLX-authored block and instance names are deliberately preserved rather
/// than escaped or mangled, so callers must reject backend keywords before
/// code generation.
pub fn is_valid_system_verilog_identifier(name: &str) -> bool {
    if !xlsynth_pir::ir_utils::is_valid_identifier_name(name) {
        return false;
    }
    !matches!(
        name,
        "accept_on"
            | "alias"
            | "always"
            | "always_comb"
            | "always_ff"
            | "always_latch"
            | "and"
            | "assert"
            | "assign"
            | "assume"
            | "automatic"
            | "before"
            | "begin"
            | "bind"
            | "bins"
            | "binsof"
            | "bit"
            | "break"
            | "buf"
            | "bufif0"
            | "bufif1"
            | "byte"
            | "case"
            | "casex"
            | "casez"
            | "cell"
            | "chandle"
            | "checker"
            | "class"
            | "clocking"
            | "cmos"
            | "config"
            | "const"
            | "constraint"
            | "context"
            | "continue"
            | "cover"
            | "covergroup"
            | "coverpoint"
            | "cross"
            | "deassign"
            | "default"
            | "defparam"
            | "design"
            | "disable"
            | "dist"
            | "do"
            | "edge"
            | "else"
            | "end"
            | "endchecker"
            | "endclass"
            | "endclocking"
            | "endconfig"
            | "endfunction"
            | "endgenerate"
            | "endgroup"
            | "endinterface"
            | "endmodule"
            | "endpackage"
            | "endprimitive"
            | "endprogram"
            | "endproperty"
            | "endspecify"
            | "endsequence"
            | "endtable"
            | "endtask"
            | "enum"
            | "event"
            | "eventually"
            | "expect"
            | "export"
            | "extends"
            | "extern"
            | "final"
            | "first_match"
            | "for"
            | "force"
            | "foreach"
            | "forever"
            | "fork"
            | "forkjoin"
            | "function"
            | "generate"
            | "genvar"
            | "global"
            | "highz0"
            | "highz1"
            | "if"
            | "iff"
            | "ifnone"
            | "ignore_bins"
            | "illegal_bins"
            | "implements"
            | "implies"
            | "import"
            | "incdir"
            | "include"
            | "initial"
            | "inout"
            | "input"
            | "inside"
            | "instance"
            | "int"
            | "integer"
            | "interconnect"
            | "interface"
            | "intersect"
            | "join"
            | "join_any"
            | "join_none"
            | "large"
            | "let"
            | "liblist"
            | "library"
            | "local"
            | "localparam"
            | "logic"
            | "longint"
            | "macromodule"
            | "matches"
            | "medium"
            | "modport"
            | "module"
            | "nand"
            | "negedge"
            | "nettype"
            | "new"
            | "nexttime"
            | "nmos"
            | "nor"
            | "noshowcancelled"
            | "not"
            | "notif0"
            | "notif1"
            | "null"
            | "or"
            | "output"
            | "package"
            | "packed"
            | "parameter"
            | "pmos"
            | "posedge"
            | "primitive"
            | "priority"
            | "program"
            | "property"
            | "protected"
            | "pull0"
            | "pull1"
            | "pulldown"
            | "pullup"
            | "pure"
            | "rand"
            | "randc"
            | "randcase"
            | "randsequence"
            | "rcmos"
            | "real"
            | "realtime"
            | "ref"
            | "reg"
            | "reject_on"
            | "release"
            | "repeat"
            | "restrict"
            | "return"
            | "rnmos"
            | "rpmos"
            | "rtran"
            | "rtranif0"
            | "rtranif1"
            | "s_always"
            | "s_eventually"
            | "s_nexttime"
            | "s_until"
            | "s_until_with"
            | "scalared"
            | "sequence"
            | "shortint"
            | "shortreal"
            | "showcancelled"
            | "signed"
            | "small"
            | "soft"
            | "solve"
            | "specify"
            | "specparam"
            | "static"
            | "string"
            | "strong"
            | "strong0"
            | "strong1"
            | "struct"
            | "super"
            | "supply0"
            | "supply1"
            | "sync_accept_on"
            | "sync_reject_on"
            | "table"
            | "tagged"
            | "task"
            | "this"
            | "throughout"
            | "time"
            | "timeprecision"
            | "timeunit"
            | "tran"
            | "tranif0"
            | "tranif1"
            | "tri"
            | "tri0"
            | "tri1"
            | "triand"
            | "trior"
            | "trireg"
            | "type"
            | "typedef"
            | "union"
            | "unique"
            | "unique0"
            | "unsigned"
            | "until"
            | "until_with"
            | "untyped"
            | "use"
            | "uwire"
            | "var"
            | "vectored"
            | "virtual"
            | "void"
            | "wait"
            | "wait_order"
            | "wand"
            | "weak"
            | "weak0"
            | "weak1"
            | "while"
            | "wildcard"
            | "wire"
            | "with"
            | "within"
            | "wor"
            | "xnor"
            | "xor"
    )
}

pub(crate) fn is_valid_xls_ir_block_identifier(name: &str) -> bool {
    let probe = format!("package __xlsynth_name_probe\n\nblock {name}() {{\n}}\n");
    xlsynth::IrPackage::parse_ir(&probe, None).is_ok()
}

/// Uses the authoritative XLS IR parser to determine whether `name` is legal
/// in the distinct instantiation-name grammar position.
pub(crate) fn is_valid_xls_ir_instantiation_identifier(name: &str) -> bool {
    let probe = format!(
        "package __xlsynth_name_probe\n\nblock child() {{\n}}\n\nblock parent() {{\n  instantiation {name}(block=child, kind=block)\n}}\n"
    );
    xlsynth::IrPackage::parse_ir(&probe, None).is_ok()
}

/// Codegen-only representation used to bridge XLS 0.53's inability to parse
/// its own `kind=extern` Block IR output.
#[derive(Debug, Clone)]
pub struct Xls53ExternCodegenPlan {
    pub ir_text: String,
    patches: Vec<ExternVerilogPatch>,
}

#[derive(Debug, Clone)]
struct ExternVerilogPatch {
    module_name: String,
    result_signal: String,
    rendered_template: String,
}

#[derive(Debug, Clone)]
struct FfiTemplate {
    parameter_names: Vec<String>,
    code_template: String,
}

/// Replaces extern instantiations with named zero-valued codegen placeholders.
/// The authoritative package remains unchanged and roundtrippable; this clone
/// exists only because XLS 0.53 `block_to_verilog_main` rejects the
/// `foreign_function=` syntax emitted by `codegen_main` in the same release.
pub fn prepare_xls53_extern_codegen(package: &Package) -> Result<Xls53ExternCodegenPlan, String> {
    let reachable_blocks = reachable_block_names(package)?;
    let mut ffi_templates = BTreeMap::new();
    for member in &package.members {
        let PackageMember::Function(function) = member else {
            continue;
        };
        let Some(attribute) = function
            .outer_attrs
            .iter()
            .find(|attribute| attribute.trim_start().starts_with("#[ffi_proto("))
        else {
            continue;
        };
        ffi_templates.insert(
            function.name.clone(),
            FfiTemplate {
                parameter_names: function
                    .params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect(),
                code_template: extract_ffi_code_template(attribute).map_err(|error| {
                    format!("invalid FFI metadata on '{}': {error}", function.name)
                })?,
            },
        );
    }

    let mut codegen_package = package.clone();
    codegen_package.members.retain(|member| match member {
        PackageMember::Function(_) => true,
        PackageMember::Block { func, .. } => reachable_blocks.contains(&func.name),
    });
    let mut next_id = next_package_text_id(&codegen_package);
    let mut patches = Vec::new();
    for member in &mut codegen_package.members {
        let PackageMember::Block { func, metadata } = member else {
            continue;
        };
        if !reachable_blocks.contains(&func.name) {
            continue;
        }
        let externs = metadata
            .instantiations
            .iter()
            .filter(|instantiation| instantiation.kind == InstantiationKind::Extern)
            .cloned()
            .collect::<Vec<_>>();
        let mut occupied_names = func
            .nodes
            .iter()
            .filter_map(|node| node.name.clone())
            .collect::<BTreeSet<_>>();
        for instantiation in externs {
            let ffi = ffi_templates.get(&instantiation.block).ok_or_else(|| {
                format!(
                    "extern instantiation '{}' references missing FFI function '{}'",
                    instantiation.name, instantiation.block
                )
            })?;
            let input_bindings = func
                .nodes
                .iter()
                .filter_map(|node| match &node.payload {
                    NodePayload::InstantiationInput {
                        instantiation: node_instantiation,
                        port_name,
                        arg,
                    } if node_instantiation == &instantiation.name => {
                        Some((port_name.clone(), *arg))
                    }
                    _ => None,
                })
                .collect::<BTreeMap<_, _>>();
            let outputs = func
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(index, node)| match &node.payload {
                    NodePayload::InstantiationOutput {
                        instantiation: node_instantiation,
                        port_name,
                    } if node_instantiation == &instantiation.name => {
                        Some((index, port_name.clone()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            let [(output_index, output_port)] = outputs.as_slice() else {
                return Err(format!(
                    "extern instantiation '{}' must have exactly one output",
                    instantiation.name
                ));
            };
            if output_port != "return" {
                return Err(format!(
                    "extern instantiation '{}' output must be named 'return'",
                    instantiation.name
                ));
            }

            let base = format!("__xlsynth_ffi_result_{}", patches.len());
            let mut result_signal = base.clone();
            let mut suffix = 2usize;
            while !occupied_names.insert(result_signal.clone()) {
                result_signal = format!("{base}_{suffix}");
                suffix += 1;
            }
            let mut substitutions = BTreeMap::from([
                ("return".to_string(), result_signal.clone()),
                ("fn".to_string(), instantiation.name.clone()),
            ]);
            for (parameter_index, parameter_name) in ffi.parameter_names.iter().enumerate() {
                let arg = input_bindings.get(parameter_name).ok_or_else(|| {
                    format!(
                        "extern instantiation '{}' is missing input '{}'",
                        instantiation.name, parameter_name
                    )
                })?;
                let base = format!("__xlsynth_ffi_arg_{}_{}", patches.len(), parameter_index);
                let mut anchor_name = base.clone();
                let mut suffix = 2usize;
                while !occupied_names.insert(anchor_name.clone()) {
                    anchor_name = format!("{base}_{suffix}");
                    suffix += 1;
                }
                let anchor_type = func.get_node(*arg).ty.clone();
                func.nodes.push(Node {
                    text_id: next_id,
                    name: Some(anchor_name.clone()),
                    ty: anchor_type,
                    payload: NodePayload::Unop(xlsynth_pir::ir::Unop::Identity, *arg),
                    pos: None,
                });
                next_id += 1;
                substitutions.insert(parameter_name.clone(), anchor_name);
            }
            let rendered_template = render_ffi_template(&ffi.code_template, &substitutions)?;
            let output_node = &mut func.nodes[*output_index];
            output_node.name = Some(result_signal.clone());
            output_node.payload = NodePayload::Literal(zero_ir_value_for_type(&output_node.ty));
            for node in &mut func.nodes {
                if matches!(
                    &node.payload,
                    NodePayload::InstantiationInput { instantiation: node_instantiation, .. }
                        if node_instantiation == &instantiation.name
                ) {
                    node.payload = NodePayload::Nil;
                }
            }
            patches.push(ExternVerilogPatch {
                module_name: func.name.clone(),
                result_signal,
                rendered_template,
            });
        }
        metadata
            .instantiations
            .retain(|instantiation| instantiation.kind != InstantiationKind::Extern);
    }
    verify_package(&codegen_package)
        .map_err(|error| format!("FFI codegen placeholder package failed verification: {error}"))?;
    Ok(Xls53ExternCodegenPlan {
        ir_text: codegen_package.to_string(),
        patches,
    })
}

fn reachable_block_names(package: &Package) -> Result<BTreeSet<String>, String> {
    let top = match package.top.as_ref() {
        Some((name, MemberType::Block)) => name.clone(),
        Some((_name, MemberType::Function)) => {
            return Err("extern block codegen requires a block top".to_string());
        }
        None => return Err("extern block codegen requires a selected top".to_string()),
    };
    let mut reachable = BTreeSet::new();
    let mut worklist = vec![top];
    while let Some(name) = worklist.pop() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        let PackageMember::Block { metadata, .. } = package
            .get_block(&name)
            .ok_or_else(|| format!("reachable block '{name}' is missing"))?
        else {
            return Err(format!("reachable member '{name}' is not a block"));
        };
        let mut children = metadata
            .instantiations
            .iter()
            .filter(|instantiation| instantiation.kind == InstantiationKind::Block)
            .map(|instantiation| instantiation.block.clone())
            .collect::<Vec<_>>();
        children.sort();
        worklist.extend(children.into_iter().rev());
    }
    Ok(reachable)
}

fn next_package_text_id(package: &Package) -> usize {
    package
        .members
        .iter()
        .flat_map(|member| match member {
            PackageMember::Function(function) => function
                .nodes
                .iter()
                .map(|node| node.text_id)
                .collect::<Vec<_>>(),
            PackageMember::Block { func, metadata } => func
                .nodes
                .iter()
                .map(|node| node.text_id)
                .chain(metadata.output_port_ids.values().copied())
                .collect::<Vec<_>>(),
        })
        .max()
        .unwrap_or(0)
        + 1
}

/// Restores the authored Verilog FFI templates into code emitted from a
/// placeholder package returned by `prepare_xls53_extern_codegen`.
pub fn apply_xls53_extern_codegen(
    system_verilog: &str,
    plan: &Xls53ExternCodegenPlan,
) -> Result<String, String> {
    let identifiers = sv_identifiers(system_verilog);
    let punctuation = sv_punctuation(system_verilog);
    let mut replacements = Vec::new();
    for patch in &plan.patches {
        let (module_start, module_end) = find_module_bounds(
            &identifiers,
            &patch.module_name,
            system_verilog.len(),
        )
        .ok_or_else(|| {
            format!(
                "generated SystemVerilog has no complete module named '{}' for extern patch",
                patch.module_name
            )
        })?;
        let assignments = identifiers
            .windows(2)
            .filter(|pair| {
                pair[0].0 >= module_start
                    && pair[1].1 <= module_end
                    && pair[0].2 == "assign"
                    && pair[1].2 == patch.result_signal
            })
            .collect::<Vec<_>>();
        let [assignment] = assignments.as_slice() else {
            return Err(format!(
                "module '{}' must contain exactly one placeholder assignment for '{}'",
                patch.module_name, patch.result_signal
            ));
        };
        let equals = punctuation
            .iter()
            .find(|(offset, punctuation)| {
                *offset >= assignment[1].1 && *offset < module_end && *punctuation == b'='
            })
            .map(|(offset, _)| *offset)
            .ok_or_else(|| {
                format!(
                    "placeholder assignment for '{}' has no equals sign",
                    patch.result_signal
                )
            })?;
        let assignment_end = punctuation
            .iter()
            .find(|(offset, punctuation)| {
                *offset > equals && *offset < module_end && *punctuation == b';'
            })
            .map(|(offset, _)| *offset + 1)
            .ok_or_else(|| {
                format!(
                    "placeholder assignment for '{}' is unterminated",
                    patch.result_signal
                )
            })?;
        replacements.push((
            assignment[0].0,
            assignment_end,
            patch.rendered_template.clone(),
        ));
    }
    replacements.sort_by_key(|(start, _, _)| std::cmp::Reverse(*start));
    let mut result = system_verilog.to_string();
    for (start, end, replacement) in replacements {
        result.replace_range(start..end, &replacement);
    }
    Ok(result)
}

fn find_module_bounds(
    identifiers: &[(usize, usize, String)],
    module_name: &str,
    text_len: usize,
) -> Option<(usize, usize)> {
    for (index, pair) in identifiers.windows(2).enumerate() {
        if pair[0].2 != "module" || pair[1].2 != module_name {
            continue;
        }
        let end = identifiers[index + 2..]
            .iter()
            .find(|identifier| identifier.2 == "endmodule")
            .map(|identifier| identifier.0)
            .unwrap_or(text_len);
        return (end != text_len).then_some((pair[0].0, end));
    }
    None
}

fn extract_ffi_code_template(attribute: &str) -> Result<String, String> {
    let marker = "code_template:";
    let marker_start = attribute
        .find(marker)
        .ok_or_else(|| "ffi_proto attribute has no code_template".to_string())?;
    let quote = attribute[marker_start + marker.len()..]
        .find('"')
        .map(|offset| marker_start + marker.len() + offset)
        .ok_or_else(|| "ffi_proto code_template has no opening quote".to_string())?;
    let mut output = String::new();
    let mut characters = attribute[quote + 1..].chars();
    while let Some(character) = characters.next() {
        match character {
            '"' => return Ok(output),
            '\\' => {
                let escaped = characters
                    .next()
                    .ok_or_else(|| "unterminated ffi_proto escape".to_string())?;
                output.push(match escaped {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    '\'' => '\'',
                    other => {
                        return Err(format!("unsupported ffi_proto escape sequence '\\{other}'"));
                    }
                });
            }
            other => output.push(other),
        }
    }
    Err("ffi_proto code_template has no closing quote".to_string())
}

fn render_ffi_template(
    template: &str,
    substitutions: &BTreeMap<String, String>,
) -> Result<String, String> {
    let mut output = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        output.push_str(&rest[..open]);
        let after_open = &rest[open + 1..];
        let close = after_open
            .find('}')
            .ok_or_else(|| "unterminated placeholder in Verilog FFI template".to_string())?;
        let placeholder = &after_open[..close];
        let replacement = substitutions.get(placeholder).ok_or_else(|| {
            format!("unsupported Verilog FFI template placeholder '{{{placeholder}}}'")
        })?;
        output.push_str(replacement);
        rest = &after_open[close + 1..];
    }
    output.push_str(rest);
    Ok(output)
}

/// Restores the source-level order of one generated SystemVerilog module's
/// ANSI-style port declarations without changing their generated types.
pub fn reorder_system_verilog_module_ports(
    system_verilog: &str,
    module_name: &str,
    ordered_port_names: &[String],
) -> Result<String, String> {
    let module_start = find_module_declaration(system_verilog, module_name)
        .ok_or_else(|| format!("generated SystemVerilog has no module named '{module_name}'"))?;
    let open = system_verilog[module_start..]
        .find('(')
        .map(|offset| module_start + offset)
        .ok_or_else(|| format!("module '{module_name}' has no port list"))?;
    let close = matching_paren(system_verilog, open)
        .ok_or_else(|| format!("module '{module_name}' has an unterminated port list"))?;
    let declarations = split_port_declarations(&system_verilog[open + 1..close]);
    let mut by_name = BTreeMap::new();
    let mut generated_extras = Vec::new();
    for declaration in declarations {
        let Some(name) = port_declarator_name(&declaration) else {
            return Err(format!(
                "could not identify the declarator in generated port declaration '{declaration}'"
            ));
        };
        if !ordered_port_names.contains(&name) {
            generated_extras.push(declaration);
            continue;
        }
        if by_name.insert(name.clone(), declaration).is_some() {
            return Err(format!(
                "generated module '{module_name}' declares port '{name}' more than once"
            ));
        }
    }

    let mut reordered = String::new();
    reordered.push('\n');
    let declaration_count = ordered_port_names.len() + generated_extras.len();
    for (index, name) in ordered_port_names.iter().enumerate() {
        let declaration = by_name
            .remove(name)
            .ok_or_else(|| format!("generated module '{module_name}' omits port '{name}'"))?;
        reordered.push_str("  ");
        push_port_declaration(&mut reordered, &declaration, index + 1 != declaration_count);
        reordered.push('\n');
    }
    for (extra_index, declaration) in generated_extras.iter().enumerate() {
        reordered.push_str("  ");
        push_port_declaration(
            &mut reordered,
            declaration,
            ordered_port_names.len() + extra_index + 1 != declaration_count,
        );
        reordered.push('\n');
    }

    let mut result = String::with_capacity(system_verilog.len());
    result.push_str(&system_verilog[..open + 1]);
    result.push_str(&reordered);
    result.push_str(&system_verilog[close..]);
    Ok(result)
}

fn push_port_declaration(output: &mut String, declaration: &str, trailing_comma: bool) {
    let declaration = declaration.trim();
    output.push_str(declaration);
    let escaped_declarator = sv_identifiers(declaration)
        .last()
        .is_some_and(|(start, _, _)| declaration.as_bytes()[*start] == b'\\');
    if escaped_declarator {
        output.push(' ');
    }
    if trailing_comma {
        output.push(',');
    }
}

/// Restores ordered headers for every block represented in `package`.
pub fn reorder_system_verilog_package_ports(
    system_verilog: &str,
    package: &Package,
) -> Result<String, String> {
    let mut result = system_verilog.to_string();
    for member in &package.members {
        let PackageMember::Block { func, metadata } = member else {
            continue;
        };
        if metadata.ports.is_empty() {
            continue;
        }
        // Official codegen emits only the selected top's reachable hierarchy.
        // Package members outside that closure have no generated header.
        if find_module_declaration(&result, &func.name).is_none() {
            continue;
        }
        let ordered_ports = metadata
            .ports
            .iter()
            .map(|port| port.name.clone())
            .collect::<Vec<_>>();
        result = reorder_system_verilog_module_ports(&result, &func.name, &ordered_ports)?;
    }
    Ok(result)
}

/// Renames one block in a codegen-only package, including every block-instance
/// reference and the selected package top.
pub fn rename_package_block(
    package: &mut Package,
    old_name: &str,
    new_name: &str,
) -> Result<(), String> {
    if !is_valid_system_verilog_identifier(new_name) {
        return Err(format!(
            "block name '{new_name}' is not an unreserved SystemVerilog identifier"
        ));
    }
    if !is_valid_xls_ir_block_identifier(new_name) {
        return Err(format!(
            "block name '{new_name}' is reserved by the XLS IR backend"
        ));
    }
    if old_name == new_name {
        return Ok(());
    }
    if package.members.iter().any(|member| match member {
        PackageMember::Function(function) => function.name == new_name,
        PackageMember::Block { func, .. } => func.name == new_name,
    }) {
        return Err(format!(
            "cannot rename block '{old_name}' to existing package member '{new_name}'"
        ));
    }
    let mut found = false;
    for member in &mut package.members {
        let PackageMember::Block { func, metadata } = member else {
            continue;
        };
        if func.name == old_name {
            func.name = new_name.to_string();
            found = true;
        }
        for instantiation in &mut metadata.instantiations {
            if instantiation.kind == InstantiationKind::Block && instantiation.block == old_name {
                instantiation.block = new_name.to_string();
            }
        }
    }
    if !found {
        return Err(format!("package has no block named '{old_name}'"));
    }
    if matches!(
        package.top.as_ref(),
        Some((name, MemberType::Block)) if name == old_name
    ) {
        package.top = Some((new_name.to_string(), MemberType::Block));
    }
    verify_package(package).map_err(|error| error.to_string())?;
    Ok(())
}

fn find_module_declaration(system_verilog: &str, module_name: &str) -> Option<usize> {
    let identifiers = sv_identifiers(system_verilog);
    for pair in identifiers.windows(2) {
        if pair[0].2 == "module" && pair[1].2 == module_name {
            return Some(pair[0].0);
        }
    }
    None
}

fn matching_paren(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, character) in text[open..].char_indices() {
        match character {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_port_declarations(text: &str) -> Vec<String> {
    let mut declarations = Vec::new();
    let mut start = 0usize;
    let mut bracket_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut paren_depth = 0usize;
    for (index, character) in text.char_indices() {
        match character {
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            '{' => brace_depth += 1,
            '}' => brace_depth = brace_depth.saturating_sub(1),
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            ',' if bracket_depth == 0 && brace_depth == 0 && paren_depth == 0 => {
                declarations.push(text[start..index].trim().to_string());
                start = index + 1;
            }
            _ => {}
        }
    }
    if !text[start..].trim().is_empty() {
        declarations.push(text[start..].trim().to_string());
    }
    declarations
}

fn port_declarator_name(text: &str) -> Option<String> {
    let limit = text.find('=').unwrap_or(text.len());
    sv_identifiers(&text[..limit])
        .into_iter()
        .filter(|(start, _, _)| bracket_depth_at(text, *start) == 0)
        .map(|(_, _, identifier)| identifier)
        .next_back()
}

fn bracket_depth_at(text: &str, limit: usize) -> usize {
    text[..limit].chars().fold(0usize, |depth, character| {
        if character == '[' {
            depth + 1
        } else if character == ']' {
            depth.saturating_sub(1)
        } else {
            depth
        }
    })
}

/// Returns SystemVerilog identifier spans outside comments and string
/// literals. Escaped identifiers are normalized without their leading slash.
fn sv_identifiers(text: &str) -> Vec<(usize, usize, String)> {
    let bytes = text.as_bytes();
    let mut identifiers = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
        } else if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index += 2;
            while index + 1 < bytes.len() && &bytes[index..index + 2] != b"*/" {
                index += 1;
            }
            index = (index + 2).min(bytes.len());
        } else if bytes[index] == b'"' {
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                } else if bytes[index] == b'"' {
                    index += 1;
                    break;
                } else {
                    index += 1;
                }
            }
        } else if bytes[index] == b'\\' {
            let start = index;
            index += 1;
            let name_start = index;
            while index < bytes.len() && !bytes[index].is_ascii_whitespace() {
                index += 1;
            }
            identifiers.push((start, index, text[name_start..index].to_string()));
        } else if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len() && is_identifier_byte(bytes[index]) {
                index += 1;
            }
            identifiers.push((start, index, text[start..index].to_string()));
        } else {
            index += 1;
        }
    }
    identifiers
}

/// Returns assignment punctuation outside identifiers, comments, and string
/// literals. This is sufficient for locating codegen-owned continuous
/// assignments without treating authored messages or comments as HDL syntax.
fn sv_punctuation(text: &str) -> Vec<(usize, u8)> {
    let bytes = text.as_bytes();
    let mut punctuation = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
        } else if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index += 2;
            while index + 1 < bytes.len() && &bytes[index..index + 2] != b"*/" {
                index += 1;
            }
            index = (index + 2).min(bytes.len());
        } else if bytes[index] == b'"' {
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                } else if bytes[index] == b'"' {
                    index += 1;
                    break;
                } else {
                    index += 1;
                }
            }
        } else if bytes[index] == b'\\' {
            index += 1;
            while index < bytes.len() && !bytes[index].is_ascii_whitespace() {
                index += 1;
            }
        } else if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            index += 1;
            while index < bytes.len() && is_identifier_byte(bytes[index]) {
                index += 1;
            }
        } else {
            if matches!(bytes[index], b'=' | b';') {
                punctuation.push((index, bytes[index]));
            }
            index += 1;
        }
    }
    punctuation
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::ir::{MemberType, PackageMember};
    use xlsynth_pir::ir_parser::Parser;

    use super::{
        ExternVerilogPatch, Xls53ExternCodegenPlan, apply_xls53_extern_codegen,
        extract_ffi_code_template, prepare_xls53_extern_codegen, rename_package_block,
        reorder_system_verilog_module_ports,
    };

    #[test]
    fn reorders_only_the_selected_module() {
        let text = r#"module child(
  input wire child_in
);
endmodule
module top(input wire clk, input wire rst, input wire [7:0] x, output wire [7:0] y, output wire idle);
endmodule
"#;
        let reordered = reorder_system_verilog_module_ports(
            text,
            "top",
            &["x".into(), "y".into(), "clk".into(), "rst".into()],
        )
        .unwrap();
        assert!(reordered.starts_with("module child(\n  input wire child_in\n);"));
        assert!(reordered.contains(
            "module top(\n  input wire [7:0] x,\n  output wire [7:0] y,\n  input wire clk,\n  input wire rst,\n  output wire idle\n);"
        ));
    }

    #[test]
    fn reorders_by_declarator_with_keyword_named_port_and_ignores_comments() {
        let text = concat!(
            "// module top(input wire fake);\n",
            "module top(\n",
            "  input wire x,\n",
            "  output wire \\wire \n",
            ");\n",
            "endmodule\n",
        );
        let reordered = reorder_system_verilog_module_ports(
            text,
            "top",
            &["wire".to_string(), "x".to_string()],
        )
        .unwrap();
        assert!(reordered.contains("module top(\n  output wire \\wire ,\n  input wire x\n);"));
    }

    #[test]
    fn renames_package_block_top_and_instantiation_targets() {
        let text = r#"package test

top block child(x: bits[1], y: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}

block parent(x: bits[1], y: bits[1]) {
  instantiation child_i(block=child, kind=block)
  x: bits[1] = input_port(name=x, id=3)
  child_y: bits[1] = instantiation_output(instantiation=child_i, port_name=y, id=4)
  child_x: () = instantiation_input(x, instantiation=child_i, port_name=x, id=5)
  y: () = output_port(child_y, name=y, id=6)
}
"#;
        let mut package = Parser::new_preserving_block_port_order(text)
            .parse_package()
            .unwrap();
        rename_package_block(&mut package, "child", "renamed_child").unwrap();
        assert_eq!(
            package.top,
            Some(("renamed_child".to_string(), MemberType::Block))
        );
        let PackageMember::Block { metadata, .. } = package.get_block("parent").unwrap() else {
            panic!("expected parent block");
        };
        assert_eq!(metadata.instantiations[0].block, "renamed_child");
    }

    #[test]
    fn rejects_reserved_system_verilog_block_rename() {
        let text = r#"package test

top block child(x: bits[1], y: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}
"#;
        let mut package = Parser::new_preserving_block_port_order(text)
            .parse_package()
            .unwrap();
        let error = rename_package_block(&mut package, "child", "module").unwrap_err();
        assert!(error.contains("not an unreserved SystemVerilog identifier"));
        assert!(package.get_block("child").is_some());

        let error = rename_package_block(&mut package, "child", "top").unwrap_err();
        assert!(error.contains("reserved by the XLS IR backend"));
        assert!(package.get_block("child").is_some());
    }

    #[test]
    fn ffi_template_decoder_preserves_utf8_and_rejects_unknown_escapes() {
        let attribute = r#"#[ffi_proto("""code_template: "assign {return} = {x}; // café\n"
""")]"#;
        assert_eq!(
            extract_ffi_code_template(attribute).unwrap(),
            "assign {return} = {x}; // café\n"
        );
        assert!(
            extract_ffi_code_template(
                r#"#[ffi_proto("""code_template: "assign {return} = \q;"
""")]"#
            )
            .unwrap_err()
            .contains("unsupported")
        );
    }

    #[test]
    fn ffi_patches_use_lexical_module_and_assignment_spans() {
        let system_verilog = r#"module top;
  initial $display("endmodule; assign __xlsynth_ffi_result_0 = 1'b1;");
  // assign __xlsynth_ffi_result_1 = 1'b1;
  assign __xlsynth_ffi_result_0 = 1'b0;
  assign __xlsynth_ffi_result_1 = 1'b0;
endmodule
"#;
        let plan = Xls53ExternCodegenPlan {
            ir_text: String::new(),
            patches: vec![
                ExternVerilogPatch {
                    module_name: "top".to_string(),
                    result_signal: "__xlsynth_ffi_result_0".to_string(),
                    rendered_template: "assign __xlsynth_ffi_result_0 = 1'b1; // endmodule"
                        .to_string(),
                },
                ExternVerilogPatch {
                    module_name: "top".to_string(),
                    result_signal: "__xlsynth_ffi_result_1".to_string(),
                    rendered_template: "assign __xlsynth_ffi_result_1 = 1'b1;".to_string(),
                },
            ],
        };
        let patched = apply_xls53_extern_codegen(system_verilog, &plan).unwrap();
        assert_eq!(patched.matches("= 1'b1;").count(), 4);
        assert!(!patched.contains("assign __xlsynth_ffi_result_0 = 1'b0"));
        assert!(!patched.contains("assign __xlsynth_ffi_result_1 = 1'b0"));
    }

    #[test]
    fn extern_codegen_ignores_unreachable_ffi_blocks() {
        let ir = r#"package ffi_reachability

#[ffi_proto("""code_template: "assign {return} = ~{x};"
""")]
fn ffi_not(x: bits[1] id=1) -> bits[1] {
  ret x: bits[1] = param(name=x, id=1)
}

top block top(x: bits[1], y: bits[1]) {
  x: bits[1] = input_port(name=x, id=2)
  y: () = output_port(x, name=y, id=3)
}

block unused(x: bits[1], y: bits[1]) {
  instantiation ffi_call(foreign_function=ffi_not, kind=extern)
  x: bits[1] = input_port(name=x, id=4)
  ffi_y: bits[1] = instantiation_output(instantiation=ffi_call, port_name=return, id=5)
  ffi_x: () = instantiation_input(x, instantiation=ffi_call, port_name=x, id=6)
  y: () = output_port(ffi_y, name=y, id=7)
}
"#;
        let package = Parser::new(ir).parse_and_validate_package().unwrap();
        let plan = prepare_xls53_extern_codegen(&package).unwrap();
        assert!(plan.patches.is_empty());
        assert!(!plan.ir_text.contains("block unused"));
        assert!(!plan.ir_text.contains("kind=extern"));
    }
}
