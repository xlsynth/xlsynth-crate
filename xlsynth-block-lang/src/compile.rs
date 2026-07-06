// SPDX-License-Identifier: Apache-2.0

//! Semantic analysis and direct lowering to XLS Block IR.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Read;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use xlsynth_pir::ir::{
    self, BlockMetadata, BlockPort, BlockPortKind, BlockResetMetadata, InstantiationKind,
    MemberType, Node, NodePayload, NodeRef, PackageMember, Register, Type,
};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_verify::verify_package;

use crate::parse::{self, BlockDecl, BlockItem, InstanceDecl, PortDirection, RegisterDecl};
use crate::{
    BlockCompileOptions, BlockCompileOutput, BlockDiagnostic, CombinationalOptimization,
    ParametricBinding,
};

#[derive(Debug, Clone)]
struct ResolvedInterface {
    name: String,
    ports: Vec<ResolvedPort>,
}

#[derive(Debug, Clone)]
struct ResolvedPort {
    direction: PortDirection,
    name: String,
    source_ty: String,
    role: PortRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortRole {
    Clock,
    Reset { active_low: bool },
    Data,
}

#[derive(Debug, Clone)]
struct RegisterLayout {
    name: String,
    next_result: usize,
    enable_result: usize,
    init_value_result: Option<usize>,
}

#[derive(Debug, Clone)]
struct InstanceLayout {
    name: String,
    target: String,
    reset_port_name: String,
    input_results: Vec<(String, usize)>,
    output_params: Vec<(String, String, usize)>,
}

#[derive(Debug, Clone)]
struct FfiCallLayout {
    name: String,
    target: String,
    output_param_name: String,
    input_results: Vec<(String, usize)>,
}

#[derive(Debug, Clone)]
struct PredicateLayout {
    result: usize,
    label: String,
    emitted_label: String,
}

#[derive(Debug, Clone)]
struct BlockLayout {
    source_name: String,
    helper_name: String,
    interface: ResolvedInterface,
    input_param_count: usize,
    registers: Vec<RegisterLayout>,
    instances: Vec<InstanceLayout>,
    ffi_calls: Vec<FfiCallLayout>,
    output_results: Vec<(String, usize)>,
    preserved_let_bindings: Vec<(String, Option<String>)>,
    assertions: Vec<PredicateLayout>,
    covers: Vec<PredicateLayout>,
    reset_name: String,
    reset_active_low: bool,
}

#[derive(Debug, Clone)]
struct RegisterSemantic {
    name: String,
    source_ty: String,
    init_value: Option<String>,
    enable: String,
    next: String,
}

#[derive(Debug, Clone)]
struct DeclarationAnalysis {
    all_symbols: BTreeSet<String>,
}

struct ReachableBlock {
    name: String,
    overrides: BTreeMap<String, String>,
    materialize_base: bool,
}

#[derive(Debug, Clone)]
struct ResolvedInstance {
    declaration: InstanceDecl,
    target: ResolvedInterface,
}

struct LoweredProc {
    interface: ResolvedInterface,
    members: Vec<PackageMember>,
}

struct ProcResolutionContext<'a> {
    proc_names: &'a BTreeSet<String>,
    proc_source: &'a str,
    source_path: &'a Path,
    options: &'a BlockCompileOptions,
    convert_options: &'a xlsynth::DslxConvertOptions<'a>,
    occupied_member_names: &'a mut BTreeSet<String>,
    proc_specializations: &'a mut BTreeMap<(String, bool), (String, ResolvedInterface)>,
    imported_proc_members: &'a mut Vec<PackageMember>,
    interfaces: &'a mut BTreeMap<String, ResolvedInterface>,
}

#[derive(Debug, Clone)]
struct FfiInterface {
    parameter_names: Vec<String>,
}

pub struct ToolRunOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub timed_out: bool,
    pub output_truncated: bool,
    pub artifact_too_large: bool,
}

struct StreamCapture {
    bytes: Vec<u8>,
    truncated: bool,
}

/// Compiles DSLX block source directly into a verified XLS Block IR package.
pub fn compile_block_module(
    source: &str,
    source_path: &Path,
    options: &BlockCompileOptions,
) -> Result<BlockCompileOutput, BlockDiagnostic> {
    let parsed = parse::parse_module(source, source_path)?;
    let top = select_top(&parsed.blocks, options, source_path)?;
    let module_name = xlsynth::dslx_path_to_module_name(source_path).map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("invalid DSLX source path: {error}"),
        )
    })?;
    if !xlsynth_pir::ir_utils::is_valid_identifier_name(module_name) {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "DSLX source filename must produce a valid module identifier; got '{module_name}'"
            ),
        ));
    }

    let search_paths = options
        .additional_search_paths
        .iter()
        .map(|path| path.as_path())
        .collect::<Vec<_>>();
    let convert_options = xlsynth::DslxConvertOptions {
        dslx_stdlib_path: options.dslx_stdlib_path.as_deref(),
        additional_search_paths: search_paths,
        enable_warnings: options.enable_warnings.as_deref(),
        disable_warnings: options.disable_warnings.as_deref(),
        ..xlsynth::DslxConvertOptions::default()
    };
    let reachable_blocks =
        collect_reachable_blocks(&parsed, &top, options, &convert_options, source_path)?;
    validate_backend_block_names(
        &parsed
            .blocks
            .iter()
            .filter(|block| reachable_blocks.contains(&block.name))
            .cloned()
            .collect::<Vec<_>>(),
        source_path,
    )?;

    let mut generated_dslx = parsed.prelude.clone();
    if !generated_dslx.ends_with('\n') {
        generated_dslx.push('\n');
    }

    let mut interfaces = BTreeMap::<String, ResolvedInterface>::new();
    let mut imported_proc_members = Vec::new();
    let mut layouts = Vec::new();
    let mut block_specializations =
        BTreeMap::<(String, Vec<String>), (String, ResolvedInterface)>::new();
    let mut proc_specializations = BTreeMap::<(String, bool), (String, ResolvedInterface)>::new();
    let mut next_specialization = 0usize;
    // Internal package members must not capture any valid authored identifier,
    // including ordinary functions that are retained in `parsed.prelude`.
    let mut occupied_member_names = parse::lex(source, source_path)?
        .into_iter()
        .filter(|token| {
            token
                .text
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        })
        .map(|token| token.text)
        .collect::<BTreeSet<_>>();
    for (block_index, block) in parsed.blocks.iter().enumerate() {
        if !reachable_blocks.contains(&block.name) {
            continue;
        }
        let is_unmaterialized_template =
            block.name != top && block.params.iter().any(|param| param.default.is_none());
        if is_unmaterialized_template {
            continue;
        }
        let overrides = if block.name == top {
            collect_parametric_overrides(&options.parametric_bindings, source_path)?
        } else {
            BTreeMap::new()
        };
        let values = resolve_parametrics(block, &overrides, source_path)?;
        validate_parametrics_with_dslx(
            &parsed.prelude,
            block,
            &values,
            source_path,
            &convert_options,
        )?;
        let mut elaborated_block = block.clone();
        let mut constexpr_values = values.clone();
        let mut structural_scope = StructuralDeclarationScope::new(block);
        elaborated_block.items = elaborate_structural_items(
            &block.items,
            &mut constexpr_values,
            &mut structural_scope,
            &parsed.prelude,
            &convert_options,
            source_path,
        )?;
        let declaration_analysis = analyze_block_declarations(&elaborated_block, source_path)?;
        apply_local_const_substitutions(&mut elaborated_block.items, &values);
        let interface = resolve_interface(&elaborated_block, &values, source_path)?;
        for item in &mut elaborated_block.items {
            let BlockItem::Instance(instance) = item else {
                continue;
            };
            let Some(arguments) = instance.parametrics.as_deref() else {
                continue;
            };
            if parsed.proc_names.contains(&instance.target) {
                continue;
            }
            let (target_index, target_decl) = parsed
                .blocks
                .iter()
                .enumerate()
                .find(|(_, candidate)| candidate.name == instance.target)
                .ok_or_else(|| {
                    BlockDiagnostic::new(
                        source_path,
                        Some(instance.offset),
                        format!(
                            "parametric instance target '{}' was not found",
                            instance.target
                        ),
                    )
                })?;
            if target_index >= block_index {
                return Err(BlockDiagnostic::new(
                    source_path,
                    Some(instance.offset),
                    format!(
                        "parametric block instance target '{}' must be declared before use",
                        instance.target
                    ),
                ));
            }
            let concrete_arguments = split_parametric_arguments(arguments, source_path)?
                .iter()
                .map(|argument| substitute_identifiers(argument.trim(), &values))
                .collect::<Vec<_>>();
            if concrete_arguments.len() > target_decl.params.len() {
                return Err(BlockDiagnostic::new(
                    source_path,
                    Some(instance.offset),
                    format!(
                        "instance '{}' supplies {} parametric arguments, but '{}' declares {}",
                        instance.name,
                        concrete_arguments.len(),
                        target_decl.name,
                        target_decl.params.len()
                    ),
                ));
            }
            let key = (target_decl.name.clone(), concrete_arguments.clone());
            let (specialized_name, specialized_interface) = if let Some(existing) =
                block_specializations.get(&key)
            {
                existing.clone()
            } else {
                let overrides = target_decl
                    .params
                    .iter()
                    .zip(&concrete_arguments)
                    .map(|(param, argument)| (param.name.clone(), argument.clone()))
                    .collect::<BTreeMap<_, _>>();
                let specialized_values = resolve_parametrics(target_decl, &overrides, source_path)?;
                validate_parametrics_with_dslx(
                    &parsed.prelude,
                    target_decl,
                    &specialized_values,
                    source_path,
                    &convert_options,
                )?;
                let specialized_name = fresh_identifier(
                    &format!(
                        "__xlsynth_spec_{}_{}",
                        target_decl.name, next_specialization
                    ),
                    &mut occupied_member_names,
                );
                next_specialization += 1;
                let mut specialized_decl = target_decl.clone();
                specialized_decl.name = specialized_name.clone();
                let mut specialized_constexpr = specialized_values.clone();
                let mut structural_scope = StructuralDeclarationScope::new(target_decl);
                specialized_decl.items = elaborate_structural_items(
                    &target_decl.items,
                    &mut specialized_constexpr,
                    &mut structural_scope,
                    &parsed.prelude,
                    &convert_options,
                    source_path,
                )?;
                let specialized_declaration_analysis =
                    analyze_block_declarations(&specialized_decl, source_path)?;
                apply_local_const_substitutions(&mut specialized_decl.items, &specialized_values);
                if specialized_decl.items.iter().any(|item| {
                        matches!(item, BlockItem::Instance(nested) if nested.parametrics.is_some())
                    }) {
                        return Err(BlockDiagnostic::new(
                            source_path,
                            Some(instance.offset),
                            "nested parametric block specialization is not supported",
                        ));
                    }
                let specialized_interface =
                    resolve_interface(&specialized_decl, &specialized_values, source_path)?;
                resolve_proc_instances(
                    &mut specialized_decl,
                    &specialized_interface,
                    &mut ProcResolutionContext {
                        proc_names: &parsed.proc_names,
                        proc_source: &parsed.proc_prelude,
                        source_path,
                        options,
                        convert_options: &convert_options,
                        occupied_member_names: &mut occupied_member_names,
                        proc_specializations: &mut proc_specializations,
                        imported_proc_members: &mut imported_proc_members,
                        interfaces: &mut interfaces,
                    },
                )?;
                let helper_name = fresh_identifier(
                    &format!("__xlsynth_block_{specialized_name}"),
                    &mut occupied_member_names,
                );
                let (helper, layout) = build_helper(
                    &specialized_decl,
                    specialized_interface.clone(),
                    &specialized_values,
                    &interfaces,
                    helper_name,
                    &parsed.prelude,
                    &convert_options,
                    source_path,
                    &specialized_declaration_analysis,
                )?;
                generated_dslx.push_str(&helper);
                generated_dslx.push('\n');
                interfaces.insert(specialized_name.clone(), specialized_interface.clone());
                layouts.push(layout);
                block_specializations.insert(
                    key,
                    (specialized_name.clone(), specialized_interface.clone()),
                );
                (specialized_name, specialized_interface)
            };
            interfaces.insert(specialized_name.clone(), specialized_interface);
            instance.target = specialized_name;
            instance.parametrics = None;
        }
        resolve_proc_instances(
            &mut elaborated_block,
            &interface,
            &mut ProcResolutionContext {
                proc_names: &parsed.proc_names,
                proc_source: &parsed.proc_prelude,
                source_path,
                options,
                convert_options: &convert_options,
                occupied_member_names: &mut occupied_member_names,
                proc_specializations: &mut proc_specializations,
                imported_proc_members: &mut imported_proc_members,
                interfaces: &mut interfaces,
            },
        )?;
        let helper_name = fresh_identifier(
            &format!("__xlsynth_block_{}", elaborated_block.name),
            &mut occupied_member_names,
        );
        let (helper, layout) = build_helper(
            &elaborated_block,
            interface.clone(),
            &values,
            &interfaces,
            helper_name,
            &parsed.prelude,
            &convert_options,
            source_path,
            &declaration_analysis,
        )?;
        generated_dslx.push_str(&helper);
        generated_dslx.push('\n');
        interfaces.insert(block.name.clone(), interface);
        layouts.push(layout);
    }

    let converted =
        xlsynth::convert_dslx_to_ir_text(&generated_dslx, source_path, &convert_options).map_err(
            |error| {
                BlockDiagnostic::new(
                    source_path,
                    None,
                    format!("DSLX expression conversion failed: {error}"),
                )
            },
        )?;
    let mut warnings = converted.warnings.clone();

    let mut parser = Parser::new(&converted.ir);
    let mut package = parser.parse_package().map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("could not parse converted expression IR: {error}"),
        )
    })?;

    let ffi_interfaces = collect_verilog_ffi_interfaces(&package);
    let mut blocks = Vec::new();
    for layout in &mut layouts {
        let mangled =
            xlsynth::mangle_dslx_name(module_name, &layout.helper_name).map_err(|error| {
                BlockDiagnostic::new(
                    source_path,
                    None,
                    format!("could not mangle helper function name: {error}"),
                )
            })?;
        let member_index = package
            .members
            .iter()
            .position(|member| {
                matches!(member, PackageMember::Function(function) if function.name == mangled)
            })
            .ok_or_else(|| {
                BlockDiagnostic::new(
                    source_path,
                    None,
                    format!("converted IR did not contain helper function '{mangled}'"),
                )
            })?;
        if options.combinational_optimization != CombinationalOptimization::Free {
            let PackageMember::Function(function) = &mut package.members[member_index] else {
                unreachable!("member search only selected functions");
            };
            anchor_named_helper_nodes(function, &layout.preserved_let_bindings, source_path)?;
        }
        outline_direct_ffi_invokes(
            &mut package,
            &mangled,
            layout,
            &ffi_interfaces,
            options.combinational_optimization != CombinationalOptimization::Free,
            source_path,
        )?;
        let helper_package_ir = package.to_string();
        let PackageMember::Function(_) = package.members.remove(member_index) else {
            unreachable!("member search only selected functions");
        };
        // XLS 0.53's Block IR emitter cannot lower invoke nodes. Inline calls
        // in every mode; the preserving modes still skip the later whole-block
        // optimization pass so authored block structure remains intact.
        let function = optimize_helper_function(&helper_package_ir, &mangled, source_path)?;
        blocks.push(lower_helper_to_block(
            function,
            layout,
            options.combinational_optimization != CombinationalOptimization::Free,
            source_path,
        )?);
    }

    package.members.extend(imported_proc_members);
    package.members.extend(blocks);
    package.top = Some((top.clone(), MemberType::Block));
    renumber_package_ids(&mut package);
    verify_package(&package).map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "generated Block IR failed verification: {error}\n{}",
                package
            ),
        )
    })?;
    if options.combinational_optimization == CombinationalOptimization::Free {
        let has_runtime_properties = package.members.iter().any(|member| {
            let PackageMember::Block { func, .. } = member else {
                return false;
            };
            func.nodes.iter().any(|node| {
                matches!(
                    node.payload,
                    NodePayload::Assert { .. } | NodePayload::Cover { .. }
                )
            })
        });
        let has_extern_instantiations = package.members.iter().any(|member| {
            matches!(
                member,
                PackageMember::Block { metadata, .. }
                    if metadata
                        .instantiations
                        .iter()
                        .any(|instantiation| instantiation.kind == InstantiationKind::Extern)
            )
        });
        if has_runtime_properties || has_extern_instantiations {
            warnings.push(
                "post-lowering Block IR optimization was skipped because XLS 0.53 cannot optimize blocks containing runtime properties or extern instantiations; synthesized combinational helpers were already optimized before those constructs were attached"
                    .to_string(),
            );
        } else {
            let unoptimized_text = package.to_string();
            let ordered_ports = package
                .members
                .iter()
                .filter_map(|member| match member {
                    PackageMember::Block { func, metadata } => {
                        Some((func.name.clone(), metadata.ports.clone()))
                    }
                    _ => None,
                })
                .collect::<BTreeMap<_, _>>();
            let filename = source_path.file_name().and_then(|name| name.to_str());
            let xls_package =
                xlsynth::IrPackage::parse_ir(&unoptimized_text, filename).map_err(|error| {
                    BlockDiagnostic::new(
                        source_path,
                        None,
                        format!("could not import generated Block IR for optimization: {error}"),
                    )
                })?;
            match xlsynth::optimize_ir(&xls_package, &top) {
                Ok(optimized) => {
                    let mut parser = Parser::new(&optimized.to_string());
                    package = parser.parse_package().map_err(|error| {
                        BlockDiagnostic::new(
                            source_path,
                            None,
                            format!("could not parse optimized Block IR: {error}"),
                        )
                    })?;
                    for member in &mut package.members {
                        let PackageMember::Block { func, metadata } = member else {
                            continue;
                        };
                        if let Some(ports) = ordered_ports.get(&func.name) {
                            metadata.ports = ports.clone();
                        }
                    }
                    verify_package(&package).map_err(|error| {
                        BlockDiagnostic::new(
                            source_path,
                            None,
                            format!("optimized Block IR failed verification: {error}"),
                        )
                    })?;
                }
                Err(error) => {
                    return Err(BlockDiagnostic::new(
                        source_path,
                        None,
                        format!("combinational Block IR optimization failed: {error}"),
                    ));
                }
            }
        }
    }
    let ir_text = package.to_string();

    if options.combinational_optimization == CombinationalOptimization::PreserveNamesAndFunctions {
        warnings.push(
            "preserve-names-and-functions currently inlines calls for XLS 0.53 codegen compatibility; materializing calls as combinational child blocks is not implemented yet"
                .to_string(),
        );
    }

    Ok(BlockCompileOutput {
        package,
        ir_text,
        warnings,
    })
}

/// Finds the source blocks that can contribute to the selected top before any
/// lowering. Structural conditions are elaborated so discarded branches do not
/// spuriously pull in blocks or procs.
fn collect_reachable_blocks(
    parsed: &parse::Module,
    top: &str,
    options: &BlockCompileOptions,
    convert_options: &xlsynth::DslxConvertOptions<'_>,
    path: &Path,
) -> Result<BTreeSet<String>, BlockDiagnostic> {
    let mut reachable_base_blocks = BTreeSet::new();
    let mut visited_specializations = BTreeSet::new();
    let mut pending = vec![ReachableBlock {
        name: top.to_string(),
        overrides: collect_parametric_overrides(&options.parametric_bindings, path)?,
        materialize_base: true,
    }];
    while let Some(reachable) = pending.pop() {
        if reachable.materialize_base {
            reachable_base_blocks.insert(reachable.name.clone());
        }
        let block = parsed
            .blocks
            .iter()
            .find(|block| block.name == reachable.name)
            .expect("selected and referenced blocks were parsed");
        let values = resolve_parametrics(block, &reachable.overrides, path)?;
        let specialization_key = (
            block.name.clone(),
            block
                .params
                .iter()
                .map(|param| values[&param.name].clone())
                .collect::<Vec<_>>(),
        );
        if !visited_specializations.insert(specialization_key) {
            continue;
        }
        let mut constexpr_values = values.clone();
        let mut structural_scope = StructuralDeclarationScope::new(block);
        let mut items = elaborate_structural_items(
            &block.items,
            &mut constexpr_values,
            &mut structural_scope,
            &parsed.prelude,
            convert_options,
            path,
        )?;
        apply_local_const_substitutions(&mut items, &values);
        for instance in items.iter().filter_map(|item| match item {
            BlockItem::Instance(instance) => Some(instance),
            _ => None,
        }) {
            let Some(target) = parsed
                .blocks
                .iter()
                .find(|candidate| candidate.name == instance.target)
            else {
                // Proc and unknown targets are diagnosed by normal instance
                // resolution after source-block reachability is established.
                continue;
            };
            let (overrides, materialize_base) = if let Some(arguments) =
                instance.parametrics.as_deref()
            {
                let arguments = split_parametric_arguments(arguments, path)?
                    .iter()
                    .map(|argument| substitute_identifiers(argument.trim(), &values))
                    .collect::<Vec<_>>();
                if arguments.len() > target.params.len() {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(instance.offset),
                        format!(
                            "instance '{}' supplies {} parametric arguments, but '{}' declares {}",
                            instance.name,
                            arguments.len(),
                            target.name,
                            target.params.len()
                        ),
                    ));
                }
                (
                    target
                        .params
                        .iter()
                        .zip(arguments)
                        .map(|(param, argument)| (param.name.clone(), argument))
                        .collect(),
                    false,
                )
            } else {
                (BTreeMap::new(), true)
            };
            pending.push(ReachableBlock {
                name: target.name.clone(),
                overrides,
                materialize_base,
            });
        }
    }
    Ok(reachable_base_blocks)
}

fn collect_parametric_overrides(
    bindings: &[ParametricBinding],
    source_path: &Path,
) -> Result<BTreeMap<String, String>, BlockDiagnostic> {
    let mut overrides = BTreeMap::new();
    for binding in bindings {
        if overrides
            .insert(binding.name.clone(), binding.value.clone())
            .is_some()
        {
            return Err(BlockDiagnostic::new(
                source_path,
                None,
                format!(
                    "parametric binding '{}' is supplied more than once",
                    binding.name
                ),
            ));
        }
    }
    Ok(overrides)
}

fn collect_verilog_ffi_interfaces(package: &ir::Package) -> BTreeMap<String, FfiInterface> {
    package
        .members
        .iter()
        .filter_map(|member| {
            let PackageMember::Function(function) = member else {
                return None;
            };
            function
                .outer_attrs
                .iter()
                .any(|attribute| attribute.trim_start().starts_with("#[ffi_proto("))
                .then(|| {
                    (
                        function.name.clone(),
                        FfiInterface {
                            parameter_names: function
                                .params
                                .iter()
                                .map(|param| param.name.clone())
                                .collect(),
                        },
                    )
                })
        })
        .collect()
}

fn property_identifier(kind: &str, index: usize, label: &str) -> String {
    let mut fragment = String::with_capacity(label.len());
    let mut previous_was_separator = false;
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            fragment.push(ch);
            previous_was_separator = false;
        } else if !previous_was_separator {
            fragment.push('_');
            previous_was_separator = true;
        }
    }
    let fragment = fragment.trim_matches('_');
    let fragment = if fragment.is_empty() {
        "property"
    } else {
        fragment
    };
    format!("__xlsynth_{kind}_{index}_{fragment}")
}

/// Runs one external tool with bounded captured streams and a wall timeout.
/// On Unix, the child starts a fresh process group so timeout termination also
/// closes pipes inherited by descendants.
pub fn run_tool_with_limits(
    command: &mut Command,
    timeout: Duration,
    max_output_bytes: usize,
) -> std::io::Result<ToolRunOutput> {
    run_tool_with_optional_artifact_limit(command, timeout, max_output_bytes, None)
}

fn run_tool_with_artifact_limit(
    command: &mut Command,
    timeout: Duration,
    max_output_bytes: usize,
    artifact_path: &Path,
    max_artifact_bytes: usize,
) -> std::io::Result<ToolRunOutput> {
    run_tool_with_optional_artifact_limit(
        command,
        timeout,
        max_output_bytes,
        Some((artifact_path, max_artifact_bytes)),
    )
}

fn run_tool_with_optional_artifact_limit(
    command: &mut Command,
    timeout: Duration,
    max_output_bytes: usize,
    artifact: Option<(&Path, usize)>,
) -> std::io::Result<ToolRunOutput> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    configure_process_group(command);
    let mut child = command.spawn()?;
    let stdout = child.stdout.take().expect("stdout was configured as piped");
    let stderr = child.stderr.take().expect("stderr was configured as piped");
    let stdout_thread = thread::spawn(move || drain_tool_stream(stdout, max_output_bytes));
    let stderr_thread = thread::spawn(move || drain_tool_stream(stderr, max_output_bytes));

    let start = Instant::now();
    let mut timed_out = false;
    let mut artifact_too_large = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            // A successful leader must not leave descendants holding capture
            // pipes or continuing to write a watched artifact.
            terminate_process_group(&mut child);
            break status;
        }
        if artifact.is_some_and(|(path, limit)| artifact_exceeds_limit(path, limit)) {
            artifact_too_large = true;
            terminate_process_group(&mut child);
            break child.wait()?;
        }
        if start.elapsed() >= timeout {
            timed_out = true;
            terminate_process_group(&mut child);
            break child.wait()?;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout_thread.join().unwrap_or_else(|_| StreamCapture {
        bytes: b"external tool stdout reader panicked\n".to_vec(),
        truncated: true,
    });
    let stderr = stderr_thread.join().unwrap_or_else(|_| StreamCapture {
        bytes: b"external tool stderr reader panicked\n".to_vec(),
        truncated: true,
    });
    artifact_too_large |= artifact.is_some_and(|(path, limit)| artifact_exceeds_limit(path, limit));
    Ok(ToolRunOutput {
        status,
        output_truncated: stdout.truncated || stderr.truncated,
        stdout: stdout.bytes,
        stderr: stderr.bytes,
        timed_out,
        artifact_too_large,
    })
}

/// Runs a tool with lossless stdout redirected to `stdout_path` while keeping
/// only a bounded stderr excerpt for diagnostics.
fn run_external_tool_to_file(
    command: &mut Command,
    stdout_path: &Path,
    timeout: Duration,
    max_stderr_bytes: usize,
    max_artifact_bytes: usize,
) -> std::io::Result<ToolRunOutput> {
    let stdout = std::fs::File::create(stdout_path)?;
    command.stdout(Stdio::from(stdout)).stderr(Stdio::piped());
    configure_process_group(command);
    let mut child = command.spawn()?;
    let stderr = child.stderr.take().expect("stderr was configured as piped");
    let stderr_thread = thread::spawn(move || drain_tool_stream(stderr, max_stderr_bytes));

    let start = Instant::now();
    let mut timed_out = false;
    let mut artifact_too_large = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            terminate_process_group(&mut child);
            break status;
        }
        if artifact_exceeds_limit(stdout_path, max_artifact_bytes) {
            artifact_too_large = true;
            terminate_process_group(&mut child);
            break child.wait()?;
        }
        if start.elapsed() >= timeout {
            timed_out = true;
            terminate_process_group(&mut child);
            break child.wait()?;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stderr = stderr_thread.join().unwrap_or_else(|_| StreamCapture {
        bytes: b"external tool stderr reader panicked\n".to_vec(),
        truncated: true,
    });
    artifact_too_large |= artifact_exceeds_limit(stdout_path, max_artifact_bytes);
    Ok(ToolRunOutput {
        status,
        stdout: Vec::new(),
        output_truncated: stderr.truncated,
        stderr: stderr.bytes,
        timed_out,
        artifact_too_large,
    })
}

fn artifact_exceeds_limit(path: &Path, max_artifact_bytes: usize) -> bool {
    std::fs::metadata(path)
        .map(|metadata| metadata.len() > max_artifact_bytes as u64)
        .unwrap_or(false)
}

fn drain_tool_stream(mut stream: impl Read, max_output_bytes: usize) -> StreamCapture {
    let mut retained = Vec::new();
    let mut buffer = [0u8; 8192];
    let mut truncated = false;
    loop {
        match stream.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(count) => {
                let available = max_output_bytes.saturating_sub(retained.len());
                let keep = available.min(count);
                retained.extend_from_slice(&buffer[..keep]);
                truncated |= keep != count;
            }
        }
    }
    if truncated {
        retained.extend_from_slice(b"\n...[external tool output truncated]\n");
    }
    StreamCapture {
        bytes: retained,
        truncated,
    }
}

fn configure_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        command.process_group(0);
    }
}

fn terminate_process_group(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        // SAFETY: the child pid is converted to the negative process-group id
        // created immediately before spawn; SIGKILL requires no shared memory.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

fn find_top_proc_name(ir_text: &str) -> Option<String> {
    ir_text.lines().find_map(|line| {
        let rest = line.trim_start().strip_prefix("top proc ")?;
        let name = rest
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .collect::<String>();
        (!name.is_empty()).then_some(name)
    })
}

fn unique_identifier(text: &str, base: &str) -> String {
    let identifiers = source_identifiers(text);
    if !identifiers.contains(base) {
        return base.to_string();
    }
    let mut suffix = 2usize;
    loop {
        let candidate = format!("{base}_{suffix}");
        if !identifiers.contains(candidate.as_str()) {
            return candidate;
        }
        suffix += 1;
    }
}

fn source_identifiers(text: &str) -> BTreeSet<&str> {
    let mut identifiers = BTreeSet::new();
    let bytes = text.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            identifiers.insert(&text[start..index]);
        } else {
            index += 1;
        }
    }
    identifiers
}

/// Adds source lets to the helper return tuple so the optimizer cannot remove
/// their value-producing nodes in the name-preserving modes.
fn anchor_named_helper_nodes(
    function: &mut ir::Fn,
    bindings: &[(String, Option<String>)],
    source_path: &Path,
) -> Result<(), BlockDiagnostic> {
    if bindings.is_empty() {
        return Ok(());
    }
    let return_ref = function
        .ret_node_ref
        .ok_or_else(|| BlockDiagnostic::new(source_path, None, "generated helper has no return"))?;
    let NodePayload::Tuple(mut elements) = function.get_node(return_ref).payload.clone() else {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            "generated helper return is not a tuple",
        ));
    };
    let Type::Tuple(mut types) = function.ret_ty.clone() else {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            "generated helper return type is not a tuple",
        ));
    };

    for (name, simple_alias) in bindings {
        let direct = function
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| node.name.as_deref() == Some(name))
            .map(|(index, _)| NodeRef { index });
        let alias = simple_alias.as_ref().and_then(|alias| {
            function
                .nodes
                .iter()
                .enumerate()
                .find(|(_, node)| node.name.as_deref() == Some(alias))
                .map(|(index, _)| NodeRef { index })
                .or_else(|| {
                    function
                        .params
                        .iter()
                        .enumerate()
                        .find(|(_, param)| &param.name == alias)
                        .map(|(index, _)| NodeRef { index: index + 1 })
                })
        });
        let node_ref = direct.or(alias).ok_or_else(|| {
            BlockDiagnostic::new(
                source_path,
                None,
                format!(
                    "could not retain source let '{name}' in name-preserving mode; its DSLX lowering did not produce an identifiable value"
                ),
            )
        })?;
        elements.push(node_ref);
        types.push(Box::new(function.get_node(node_ref).ty.clone()));
    }
    function.get_node_mut(return_ref).payload = NodePayload::Tuple(elements);
    let return_type = Type::Tuple(types);
    function.get_node_mut(return_ref).ty = return_type.clone();
    function.ret_ty = return_type;
    Ok(())
}

/// Replaces direct Verilog-FFI invokes with opaque helper parameters before
/// ordinary XLS optimization. Operand values are appended to the helper return
/// tuple so they survive optimization and can drive the eventual extern
/// instantiation inputs.
fn outline_direct_ffi_invokes(
    package: &mut ir::Package,
    helper_name: &str,
    layout: &mut BlockLayout,
    ffi_interfaces: &BTreeMap<String, FfiInterface>,
    preserved_lets_are_anchored: bool,
    source_path: &Path,
) -> Result<(), BlockDiagnostic> {
    let ffi_names = ffi_interfaces.keys().cloned().collect::<BTreeSet<_>>();
    if let Some(target) = nested_ffi_target(package, helper_name, &ffi_names) {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "Verilog FFI function '{target}' is called through an ordinary helper function; call it directly from the block in the MVP"
            ),
        ));
    }

    let function = package.get_fn_mut(helper_name).ok_or_else(|| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("generated helper '{helper_name}' was not found"),
        )
    })?;
    let direct_invokes = function
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| match &node.payload {
            NodePayload::Invoke { to_apply, operands } if ffi_names.contains(to_apply) => {
                Some((index, to_apply.clone(), operands.clone()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if direct_invokes.is_empty() {
        return Ok(());
    }

    let return_ref = function.ret_node_ref.ok_or_else(|| {
        BlockDiagnostic::new(source_path, None, "generated helper has no return value")
    })?;
    let NodePayload::Tuple(mut return_elements) = function.get_node(return_ref).payload.clone()
    else {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            "generated helper return is not a tuple",
        ));
    };
    let Type::Tuple(mut return_types) = function.ret_ty.clone() else {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            "generated helper return type is not a tuple",
        ));
    };
    let preserved_result_count = if preserved_lets_are_anchored {
        layout.preserved_let_bindings.len()
    } else {
        0
    };
    let operand_insert_index = return_elements
        .len()
        .checked_sub(preserved_result_count)
        .ok_or_else(|| {
            BlockDiagnostic::new(
                source_path,
                None,
                "preserved-let metadata exceeds the generated helper result count",
            )
        })?;
    let mut occupied_symbols = function
        .params
        .iter()
        .map(|param| param.name.clone())
        .chain(function.nodes.iter().filter_map(|node| node.name.clone()))
        .chain(
            layout
                .instances
                .iter()
                .map(|instance| instance.name.clone()),
        )
        .collect::<BTreeSet<_>>();
    let mut operand_elements = Vec::new();
    let mut operand_types = Vec::new();

    for (invoke_index, target, operands) in direct_invokes {
        let interface = ffi_interfaces
            .get(&target)
            .expect("direct FFI target came from the interface map");
        if interface.parameter_names.len() != operands.len() {
            return Err(BlockDiagnostic::new(
                source_path,
                None,
                format!(
                    "Verilog FFI call '{target}' has {} operands but {} parameters",
                    operands.len(),
                    interface.parameter_names.len()
                ),
            ));
        }
        let instantiation_name = fresh_identifier(
            &format!("__xlsynth_ffi_{invoke_index}"),
            &mut occupied_symbols,
        );
        let output_param_name = fresh_identifier(
            &format!("__xlsynth_ffi_result_{invoke_index}"),
            &mut occupied_symbols,
        );
        let param_id = ir::ParamId::new(function.nodes[invoke_index].text_id);
        let result_ty = function.nodes[invoke_index].ty.clone();
        function.params.push(ir::Param {
            name: output_param_name.clone(),
            ty: result_ty,
            id: param_id,
        });
        function.nodes[invoke_index].name = Some(output_param_name.clone());
        function.nodes[invoke_index].payload = NodePayload::GetParam(param_id);

        let mut input_results = Vec::new();
        for (port_name, operand) in interface.parameter_names.iter().zip(operands) {
            let result_index = operand_insert_index + operand_elements.len();
            operand_types.push(Box::new(function.get_node(operand).ty.clone()));
            operand_elements.push(operand);
            input_results.push((port_name.clone(), result_index));
        }
        layout.ffi_calls.push(FfiCallLayout {
            name: instantiation_name,
            target,
            output_param_name,
            input_results,
        });
    }

    return_elements.splice(operand_insert_index..operand_insert_index, operand_elements);
    return_types.splice(operand_insert_index..operand_insert_index, operand_types);
    function.get_node_mut(return_ref).payload = NodePayload::Tuple(return_elements);
    let return_type = Type::Tuple(return_types);
    function.get_node_mut(return_ref).ty = return_type.clone();
    function.ret_ty = return_type;
    Ok(())
}

/// Finds FFI calls reached through an ordinary callee. Direct invokes in the
/// synthesized block helper are handled by `outline_direct_ffi_invokes`.
fn nested_ffi_target(
    package: &ir::Package,
    helper_name: &str,
    ffi_names: &BTreeSet<String>,
) -> Option<String> {
    let helper = package.get_fn(helper_name)?;
    let mut worklist = helper
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::Invoke { to_apply, .. } if !ffi_names.contains(to_apply) => {
                Some(to_apply.clone())
            }
            NodePayload::CountedFor { body, .. } => Some(body.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut visited = BTreeSet::new();
    while let Some(name) = worklist.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        if ffi_names.contains(&name) {
            return Some(name);
        }
        let function = package.get_fn(&name)?;
        for node in &function.nodes {
            match &node.payload {
                NodePayload::Invoke { to_apply, .. } => worklist.push(to_apply.clone()),
                NodePayload::CountedFor { body, .. } => worklist.push(body.clone()),
                _ => {}
            }
        }
    }
    None
}

/// Returns the first FFI function reachable from `start`, including a direct
/// call. Stable sets make the diagnostic deterministic across runs.
fn reachable_ffi_target(
    package: &ir::Package,
    start: &str,
    ffi_names: &BTreeSet<String>,
) -> Option<String> {
    let mut worklist = vec![start.to_string()];
    let mut visited = BTreeSet::new();
    while let Some(name) = worklist.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        if name != start && ffi_names.contains(&name) {
            return Some(name);
        }
        let function = package.get_fn(&name)?;
        let mut callees = function
            .nodes
            .iter()
            .filter_map(|node| match &node.payload {
                NodePayload::Invoke { to_apply, .. } => Some(to_apply.clone()),
                NodePayload::CountedFor { body, .. } => Some(body.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        callees.sort();
        worklist.extend(callees.into_iter().rev());
    }
    None
}

fn fresh_identifier(base: &str, occupied: &mut BTreeSet<String>) -> String {
    if occupied.insert(base.to_string()) {
        return base.to_string();
    }
    let mut suffix = 2usize;
    loop {
        let candidate = format!("{base}_{suffix}");
        if occupied.insert(candidate.clone()) {
            return candidate;
        }
        suffix += 1;
    }
}

/// Optimizes a synthesized combinational helper before it becomes Block IR.
///
/// At this point runtime block assertions and covers have not been attached,
/// so ordinary XLS inlining and DCE can run without the optimizer's
/// non-synthesizable-separation pass rejecting the package.
fn optimize_helper_function(
    ir_text: &str,
    top: &str,
    source_path: &Path,
) -> Result<ir::Fn, BlockDiagnostic> {
    let filename = source_path.file_name().and_then(|name| name.to_str());
    let package = xlsynth::IrPackage::parse_ir(ir_text, filename).map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("could not import synthesized DSLX helper IR: {error}"),
        )
    })?;
    let optimized = xlsynth::optimize_ir(&package, top).map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("combinational helper optimization failed for '{top}': {error}"),
        )
    })?;
    let optimized_text = optimized.to_string();
    let mut parser = Parser::new(&optimized_text);
    let optimized_package = parser.parse_package().map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("could not parse optimized helper IR for '{top}': {error}"),
        )
    })?;
    let function = optimized_package.get_fn(top).cloned().ok_or_else(|| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("optimized helper package is missing top function '{top}'"),
        )
    })?;
    if let Some(target) = function.nodes.iter().find_map(|node| {
        let NodePayload::Invoke { to_apply, .. } = &node.payload else {
            return None;
        };
        Some(to_apply)
    }) {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!("helper optimization left an unsupported call to '{target}' in '{top}'"),
        ));
    }
    Ok(function)
}

/// DSLX conversion assigns package-global node ids, but lowering adds nodes to
/// individual functions. Reestablish the package-wide uniqueness invariant
/// after every helper has been converted into a block.
fn renumber_package_ids(package: &mut ir::Package) {
    let mut next_id = 1usize;
    for member in &mut package.members {
        let function = match member {
            PackageMember::Function(function) => function,
            PackageMember::Block { func, .. } => func,
        };
        for node in function.nodes.iter_mut().skip(1) {
            node.text_id = next_id;
            next_id += 1;
        }
        for (index, param) in function.params.iter_mut().enumerate() {
            let node = &mut function.nodes[index + 1];
            let id = ir::ParamId::new(node.text_id);
            param.id = id;
            node.payload = NodePayload::GetParam(id);
        }
    }
    for member in &mut package.members {
        let PackageMember::Block { func, metadata } = member else {
            continue;
        };
        metadata.input_port_ids = func
            .params
            .iter()
            .map(|param| (param.name.clone(), param.id.get_wrapped_id()))
            .collect();
        for name in &metadata.output_names {
            metadata.output_port_ids.insert(name.clone(), next_id);
            next_id += 1;
        }
    }
}

fn select_top(
    blocks: &[BlockDecl],
    options: &BlockCompileOptions,
    path: &Path,
) -> Result<String, BlockDiagnostic> {
    if let Some(top) = &options.top {
        if blocks.iter().any(|block| &block.name == top) {
            return Ok(top.clone());
        }
        return Err(BlockDiagnostic::new(
            path,
            None,
            format!("requested top block '{top}' was not found"),
        ));
    }
    let public = blocks
        .iter()
        .filter(|block| block.is_public)
        .collect::<Vec<_>>();
    match public.as_slice() {
        [only] => Ok(only.name.clone()),
        [] if blocks.len() == 1 => Ok(blocks[0].name.clone()),
        [] => Err(BlockDiagnostic::new(
            path,
            None,
            "multiple private blocks require an explicit top selection",
        )),
        _ => Err(BlockDiagnostic::new(
            path,
            None,
            "multiple public blocks require an explicit top selection",
        )),
    }
}

/// Rejects source block names that the active XLS IR lexer reserves. Source
/// names are otherwise preserved verbatim through PIR and SystemVerilog.
fn validate_backend_block_names(blocks: &[BlockDecl], path: &Path) -> Result<(), BlockDiagnostic> {
    for block in blocks {
        if !crate::sv::is_valid_system_verilog_identifier(&block.name) {
            return Err(BlockDiagnostic::new(
                path,
                Some(block.offset),
                format!(
                    "block name '{}' is reserved by the SystemVerilog backend; choose a different name",
                    block.name
                ),
            ));
        }
        if !crate::sv::is_valid_xls_ir_block_identifier(&block.name) {
            return Err(BlockDiagnostic::new(
                path,
                Some(block.offset),
                format!(
                    "block name '{}' is reserved by the XLS IR backend; choose a different name",
                    block.name
                ),
            ));
        }
    }
    Ok(())
}

/// Resolves every direct proc instance in one already-elaborated block. This
/// is shared by ordinary blocks and concretely specialized parametric blocks
/// so specialization does not bypass the proc adapter.
fn resolve_proc_instances(
    block: &mut BlockDecl,
    interface: &ResolvedInterface,
    context: &mut ProcResolutionContext<'_>,
) -> Result<(), BlockDiagnostic> {
    for item in &mut block.items {
        let BlockItem::Instance(instance) = item else {
            continue;
        };
        if context.interfaces.contains_key(&instance.target)
            || !context.proc_names.contains(&instance.target)
        {
            continue;
        }
        let proc_name = instance.target.clone();
        if instance
            .parametrics
            .as_deref()
            .is_some_and(|arguments| !arguments.trim().is_empty())
        {
            return Err(BlockDiagnostic::new(
                context.source_path,
                Some(instance.offset),
                "parametric proc instances are not supported by the fixed-schedule MVP",
            ));
        }
        let reset_active_low = interface
            .ports
            .iter()
            .find_map(|port| match port.role {
                PortRole::Reset { active_low } => Some(active_low),
                _ => None,
            })
            .expect("validated block interface has a reset");
        let key = (proc_name.clone(), reset_active_low);
        let (target_name, target_interface) =
            if let Some(existing) = context.proc_specializations.get(&key) {
                existing.clone()
            } else {
                let lowered = lower_proc_target(
                    context.proc_source,
                    &proc_name,
                    interface,
                    context.source_path,
                    context.options,
                    context.convert_options,
                    context.occupied_member_names,
                )?;
                let target_name = lowered.interface.name.clone();
                context
                    .interfaces
                    .insert(target_name.clone(), lowered.interface.clone());
                context.imported_proc_members.extend(lowered.members);
                context
                    .proc_specializations
                    .insert(key, (target_name.clone(), lowered.interface.clone()));
                (target_name, lowered.interface)
            };
        context
            .interfaces
            .insert(target_name.clone(), target_interface);
        instance.target = target_name;
    }
    Ok(())
}

fn lower_proc_target(
    proc_source: &str,
    proc_name: &str,
    parent: &ResolvedInterface,
    source_path: &Path,
    options: &BlockCompileOptions,
    convert_options: &xlsynth::DslxConvertOptions,
    occupied_member_names: &mut BTreeSet<String>,
) -> Result<LoweredProc, BlockDiagnostic> {
    let tool_path = options.tool_path.as_deref().ok_or_else(|| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("proc instance '{proc_name}' requires an XLS toolchain path"),
        )
    })?;
    let codegen = tool_path.join("codegen_main");
    let converter = tool_path.join("ir_converter_main");
    if !codegen.is_file() {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!("proc instance '{proc_name}' requires {}", codegen.display()),
        ));
    }
    if !converter.is_file() {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "proc instance '{proc_name}' requires {}",
                converter.display()
            ),
        ));
    }
    let module_name = xlsynth::dslx_path_to_module_name(source_path).map_err(|error| {
        BlockDiagnostic::new(source_path, None, format!("invalid source path: {error}"))
    })?;
    let temporary = tempfile::tempdir().map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("could not create proc codegen temporary directory: {error}"),
        )
    })?;
    let dslx_path = temporary.path().join(
        source_path
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("proc_source.x")),
    );
    let ir_path = temporary.path().join("proc.ir");
    let block_path = temporary.path().join("proc.block.ir");
    std::fs::write(&dslx_path, proc_source).map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("could not write temporary proc DSLX: {error}"),
        )
    })?;
    let mut convert_command = Command::new(&converter);
    convert_command
        .arg(&dslx_path)
        .arg("--top")
        .arg(proc_name)
        .arg("--convert_tests=false");
    if let Some(stdlib) = convert_options.dslx_stdlib_path {
        convert_command.arg("--dslx_stdlib_path").arg(stdlib);
    }
    if !convert_options.additional_search_paths.is_empty() {
        let joined =
            std::env::join_paths(&convert_options.additional_search_paths).map_err(|error| {
                BlockDiagnostic::new(
                    source_path,
                    None,
                    format!("could not join DSLX search paths: {error}"),
                )
            })?;
        convert_command.arg("--dslx_path").arg(joined);
    }
    if let Some(warnings) = convert_options.enable_warnings {
        convert_command
            .arg("--enable_warnings")
            .arg(warnings.join(","));
    }
    if let Some(warnings) = convert_options.disable_warnings {
        convert_command
            .arg("--disable_warnings")
            .arg(warnings.join(","));
    }
    let converted = run_external_tool_to_file(
        &mut convert_command,
        &ir_path,
        options.external_tool_timeout,
        options.max_tool_output_bytes,
        options.max_tool_artifact_bytes,
    )
    .map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("failed to run {}: {error}", converter.display()),
        )
    })?;
    if converted.timed_out {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "proc '{proc_name}' IR conversion exceeded {:?}",
                options.external_tool_timeout
            ),
        ));
    }
    if converted.artifact_too_large {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "proc '{proc_name}' IR conversion exceeded the {}-byte artifact limit",
                options.max_tool_artifact_bytes
            ),
        ));
    }
    if !converted.status.success() {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "could not convert proc '{proc_name}' to XLS IR: {}",
                String::from_utf8_lossy(&converted.stderr)
            ),
        ));
    }
    let converted_ir = std::fs::read_to_string(&ir_path).map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("could not read converted proc IR: {error}"),
        )
    })?;
    let ir_proc_name = find_top_proc_name(&converted_ir).ok_or_else(|| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("converted IR for proc '{proc_name}' has no declared top proc"),
        )
    })?;
    let reset = parent
        .ports
        .iter()
        .find_map(|port| match port.role {
            PortRole::Reset { active_low } => Some((port, active_low)),
            _ => None,
        })
        .expect("validated parent interface has a reset");
    let child_reset_name = unique_identifier(&converted_ir, "__xlsynth_block_reset");
    let mut command = Command::new(&codegen);
    command
        .arg(&ir_path)
        .arg("--top")
        .arg(&ir_proc_name)
        .arg("--generator=pipeline")
        .arg("--delay_model=unit")
        .arg("--pipeline_stages=1")
        .arg("--reset")
        .arg(&child_reset_name)
        .arg(format!("--reset_active_low={}", reset.1))
        .arg("--reset_asynchronous=false")
        .arg("--use_system_verilog=true")
        .arg("--output_block_ir_path")
        .arg(&block_path);
    let output = run_tool_with_artifact_limit(
        &mut command,
        options.external_tool_timeout,
        options.max_tool_output_bytes,
        &block_path,
        options.max_tool_artifact_bytes,
    )
    .map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("failed to run {}: {error}", codegen.display()),
        )
    })?;
    if output.timed_out {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "proc '{proc_name}' codegen exceeded {:?}",
                options.external_tool_timeout
            ),
        ));
    }
    if output.artifact_too_large {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "proc '{proc_name}' Block IR exceeded the {}-byte artifact limit",
                options.max_tool_artifact_bytes
            ),
        ));
    }
    if !output.status.success() {
        return Err(BlockDiagnostic::new(
            source_path,
            None,
            format!(
                "proc '{proc_name}' failed fixed one-stage codegen: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }
    let block_ir = std::fs::read_to_string(&block_path).map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("proc codegen did not produce readable Block IR: {error}"),
        )
    })?;
    let mut parser = Parser::new_preserving_block_port_order(&block_ir);
    let package = parser.parse_package().map_err(|error| {
        BlockDiagnostic::new(
            source_path,
            None,
            format!("could not parse proc-generated Block IR: {error}"),
        )
    })?;
    let top_name = match &package.top {
        Some((name, MemberType::Block)) => name.clone(),
        _ => {
            return Err(BlockDiagnostic::new(
                source_path,
                None,
                "proc-generated package did not declare a top block",
            ));
        }
    };
    let top_metadata = package
        .members
        .iter()
        .find_map(|member| match member {
            PackageMember::Block { func, metadata } if func.name == top_name => Some(metadata),
            _ => None,
        })
        .ok_or_else(|| {
            BlockDiagnostic::new(
                source_path,
                None,
                format!("proc-generated top block '{top_name}' was not found"),
            )
        })?;
    let ports = top_metadata
        .ports
        .iter()
        .map(|port| {
            let role = if port.kind == BlockPortKind::Clock {
                PortRole::Clock
            } else if top_metadata
                .reset
                .as_ref()
                .is_some_and(|reset| reset.port_name == port.name)
            {
                PortRole::Reset {
                    active_low: top_metadata
                        .reset
                        .as_ref()
                        .expect("reset was just matched")
                        .active_low,
                }
            } else {
                PortRole::Data
            };
            ResolvedPort {
                direction: if port.kind == BlockPortKind::Output {
                    PortDirection::Output
                } else {
                    PortDirection::Input
                },
                name: port.name.clone(),
                source_ty: port
                    .ty
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "clock".to_string()),
                role,
            }
        })
        .collect();
    let reset_tag = if reset.1 { "active_low" } else { "active_high" };
    let prefix = format!("__xlsynth_proc_{module_name}_{proc_name}_{reset_tag}");
    let (members, renamed_top) =
        rename_proc_members(package.members, &top_name, &prefix, occupied_member_names);
    Ok(LoweredProc {
        interface: ResolvedInterface {
            name: renamed_top,
            ports,
        },
        members,
    })
}

/// Applies a deterministic namespace to every member in an imported proc
/// closure and rewrites all references within that closure.
fn rename_proc_members(
    mut members: Vec<PackageMember>,
    top_name: &str,
    prefix: &str,
    occupied: &mut BTreeSet<String>,
) -> (Vec<PackageMember>, String) {
    let mut renames = BTreeMap::new();
    for member in &members {
        let original = match member {
            PackageMember::Function(function) => &function.name,
            PackageMember::Block { func, .. } => &func.name,
        };
        let base = format!("{prefix}__{original}");
        let mut candidate = base.clone();
        let mut suffix = 2usize;
        while occupied.contains(&candidate) {
            candidate = format!("{base}_{suffix}");
            suffix += 1;
        }
        occupied.insert(candidate.clone());
        renames.insert(original.clone(), candidate);
    }

    for member in &mut members {
        let (function, metadata) = match member {
            PackageMember::Function(function) => (function, None),
            PackageMember::Block { func, metadata } => (func, Some(metadata)),
        };
        function.name = renames
            .get(&function.name)
            .expect("every imported proc member was assigned a name")
            .clone();
        for node in &mut function.nodes {
            match &mut node.payload {
                NodePayload::Invoke { to_apply, .. } => {
                    if let Some(replacement) = renames.get(to_apply) {
                        *to_apply = replacement.clone();
                    }
                }
                NodePayload::CountedFor { body, .. } => {
                    if let Some(replacement) = renames.get(body) {
                        *body = replacement.clone();
                    }
                }
                _ => {}
            }
        }
        if let Some(metadata) = metadata {
            for instantiation in &mut metadata.instantiations {
                if let Some(replacement) = renames.get(&instantiation.block) {
                    instantiation.block = replacement.clone();
                }
            }
        }
    }
    let renamed_top = renames
        .get(top_name)
        .expect("proc-generated top is part of its package")
        .clone();
    (members, renamed_top)
}

fn resolve_parametrics(
    block: &BlockDecl,
    overrides: &BTreeMap<String, String>,
    path: &Path,
) -> Result<BTreeMap<String, String>, BlockDiagnostic> {
    let mut seen = BTreeSet::new();
    for param in &block.params {
        if !seen.insert(param.name.clone()) {
            return Err(BlockDiagnostic::new(
                path,
                Some(param.offset),
                format!(
                    "parametric binding '{}' is declared more than once",
                    param.name
                ),
            ));
        }
    }
    for name in overrides.keys() {
        if !block.params.iter().any(|param| &param.name == name) {
            return Err(BlockDiagnostic::new(
                path,
                Some(block.offset),
                format!(
                    "unknown parametric binding '{name}' for block '{}'",
                    block.name
                ),
            ));
        }
    }
    let mut values = BTreeMap::new();
    for param in &block.params {
        if param.ty.trim().is_empty() {
            return Err(BlockDiagnostic::new(
                path,
                Some(block.offset),
                format!("parametric binding '{}' has no type", param.name),
            ));
        }
        let value = if let Some(override_value) = overrides.get(&param.name) {
            override_value.clone()
        } else if let Some(default) = &param.default {
            substitute_identifiers(default, &values)
        } else {
            return Err(BlockDiagnostic::new(
                path,
                Some(block.offset),
                format!(
                    "block '{}' requires a value for parametric binding '{}'",
                    block.name, param.name
                ),
            ));
        };
        values.insert(param.name.clone(), value);
    }
    Ok(values)
}

fn validate_parametrics_with_dslx(
    prelude: &str,
    block: &BlockDecl,
    values: &BTreeMap<String, String>,
    path: &Path,
    convert_options: &xlsynth::DslxConvertOptions<'_>,
) -> Result<(), BlockDiagnostic> {
    if block.params.is_empty() {
        return Ok(());
    }
    let helper_name = unique_identifier(
        prelude,
        &format!("__xlsynth_validate_{}_parametrics", block.name),
    );
    let mut prior_values = BTreeMap::new();
    let mut types = Vec::new();
    let mut expressions = Vec::new();
    for param in &block.params {
        types.push(substitute_identifiers(&param.ty, &prior_values));
        let value = values
            .get(&param.name)
            .expect("every resolved parametric has a value")
            .clone();
        expressions.push(brace_dslx_generic_args(&value));
        prior_values.insert(param.name.clone(), value);
    }
    let mut source = prelude.to_string();
    source.push_str(&format!(
        "\nfn {helper_name}() -> {} {{ {} }}\n",
        tuple_text(types.iter().map(String::as_str)),
        tuple_text(expressions.iter().map(String::as_str)),
    ));
    let converted =
        xlsynth::convert_dslx_to_ir_text(&source, path, convert_options).map_err(|error| {
            BlockDiagnostic::new(
                path,
                Some(block.offset),
                format!(
                    "invalid parametric values for block '{}': {error}",
                    block.name
                ),
            )
        })?;
    reject_reachable_ffi_in_constexpr(
        &converted.ir,
        path,
        &helper_name,
        block.offset,
        "parametric value",
    )?;
    Ok(())
}

/// Uses the authoritative DSLX typechecker to compare a `declreg` type with
/// the optional type repeated on its completing `reg` contract. Keeping this
/// as a dedicated probe lets aliases compare by type identity while mapping a
/// mismatch back to the authored inline type instead of a generated helper.
fn validate_repeated_register_type(
    prelude: &str,
    block_name: &str,
    register: &RegisterDecl,
    forward_ty: &str,
    inline_ty: &str,
    path: &Path,
    convert_options: &xlsynth::DslxConvertOptions<'_>,
) -> Result<(), BlockDiagnostic> {
    let helper_name = unique_identifier(
        prelude,
        &format!("__xlsynth_validate_{block_name}_{}_type", register.name),
    );
    let source =
        format!("{prelude}\nfn {helper_name}(value: {forward_ty}) -> {inline_ty} {{ value }}\n");
    xlsynth::convert_dslx_to_ir_text(&source, path, convert_options)
        .map(|_| ())
        .map_err(|error| {
            BlockDiagnostic::new(
                path,
                register.ty_offset.or(Some(register.offset)),
                format!(
                    "register '{}' repeats type '{inline_ty}', which does not match declreg type '{forward_ty}': {error}",
                    register.name
                ),
            )
        })
}

/// Elaborates block-level constexpr conditionals before namespace checking or
/// Block IR construction. Selected declarations occupy the surrounding block
/// namespace, matching hardware elaboration rather than runtime control flow.
fn elaborate_structural_items(
    items: &[BlockItem],
    constexpr_values: &mut BTreeMap<String, String>,
    scope: &mut StructuralDeclarationScope,
    dslx_prelude: &str,
    convert_options: &xlsynth::DslxConvertOptions<'_>,
    path: &Path,
) -> Result<Vec<BlockItem>, BlockDiagnostic> {
    let mut elaborated = Vec::new();
    for item in items {
        match item {
            BlockItem::Conditional {
                condition,
                condition_offset,
                then_items,
                else_items,
                ..
            } => {
                validate_expression_order(
                    condition,
                    &scope.declared,
                    &scope.all_local_symbols,
                    &scope.outputs,
                    *condition_offset,
                    path,
                )?;
                let condition =
                    brace_dslx_generic_args(&substitute_identifiers(condition, constexpr_values));
                let selected = if eval_const_bool_with_dslx(
                    dslx_prelude,
                    &condition,
                    path,
                    convert_options,
                    *condition_offset,
                )? {
                    then_items
                } else {
                    else_items
                };
                elaborated.extend(elaborate_structural_items(
                    selected,
                    constexpr_values,
                    scope,
                    dslx_prelude,
                    convert_options,
                    path,
                )?);
            }
            BlockItem::Const {
                name, expression, ..
            } => {
                constexpr_values.insert(
                    name.clone(),
                    substitute_identifiers(expression, constexpr_values),
                );
                scope.declared.insert(name.clone());
                elaborated.push(item.clone());
            }
            BlockItem::Let { names, .. } => {
                scope.declared.extend(names.iter().cloned());
                elaborated.push(item.clone());
            }
            BlockItem::ForwardRegister { name, .. } => {
                scope.declared.insert(name.clone());
                elaborated.push(item.clone());
            }
            BlockItem::Register(register) => {
                scope.declared.insert(register.name.clone());
                elaborated.push(item.clone());
            }
            BlockItem::Instance(instance) => {
                scope.declared.insert(instance.name.clone());
                elaborated.push(item.clone());
            }
            BlockItem::Assign { .. }
            | BlockItem::Assert { .. }
            | BlockItem::Cover { .. }
            | BlockItem::ConstAssert { .. } => elaborated.push(item.clone()),
        }
    }
    Ok(elaborated)
}

/// Tracks authored declaration order while structural conditionals are
/// selected, before substitution can obscure source offsets.
struct StructuralDeclarationScope {
    declared: BTreeSet<String>,
    all_local_symbols: BTreeSet<String>,
    outputs: BTreeMap<String, String>,
}

impl StructuralDeclarationScope {
    fn new(block: &BlockDecl) -> Self {
        let declared = block
            .params
            .iter()
            .map(|param| param.name.clone())
            .chain(block.ports.iter().map(|port| port.name.clone()))
            .collect::<BTreeSet<_>>();
        let mut all_local_symbols = declared.clone();
        collect_item_symbols(&block.items, &mut all_local_symbols);
        let outputs = block
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
            .map(|port| (port.name.clone(), port.ty.clone()))
            .collect();
        Self {
            declared,
            all_local_symbols,
            outputs,
        }
    }
}

/// Collects authored symbols recursively so later declarations are detectable.
fn collect_item_symbols(items: &[BlockItem], symbols: &mut BTreeSet<String>) {
    for item in items {
        match item {
            BlockItem::Let { names, .. } => symbols.extend(names.iter().cloned()),
            BlockItem::Const { name, .. } | BlockItem::ForwardRegister { name, .. } => {
                symbols.insert(name.clone());
            }
            BlockItem::Register(register) => {
                symbols.insert(register.name.clone());
            }
            BlockItem::Instance(instance) => {
                symbols.insert(instance.name.clone());
            }
            BlockItem::Conditional {
                then_items,
                else_items,
                ..
            } => {
                collect_item_symbols(then_items, symbols);
                collect_item_symbols(else_items, symbols);
            }
            BlockItem::Assign { .. }
            | BlockItem::Assert { .. }
            | BlockItem::Cover { .. }
            | BlockItem::ConstAssert { .. } => {}
        }
    }
}

/// Substitutes declaration-ordered block localparams into every later item.
/// This lets local `const` values participate in types and child parametrics
/// while preserving the source rule that later declarations are unavailable.
fn apply_local_const_substitutions(
    items: &mut [BlockItem],
    parametric_values: &BTreeMap<String, String>,
) {
    let mut values = parametric_values.clone();
    for item in items {
        match item {
            BlockItem::Let {
                names,
                statement,
                expression,
                ..
            } => {
                let mut visible = values.clone();
                for name in names {
                    visible.remove(name);
                }
                *statement = substitute_identifiers(statement, &visible);
                *expression = substitute_identifiers(expression, &visible);
            }
            BlockItem::Const {
                name,
                statement,
                expression,
                ..
            } => {
                let mut visible = values.clone();
                visible.remove(name);
                *statement = substitute_identifiers(statement, &visible);
                *expression = substitute_identifiers(expression, &visible);
                values.insert(name.clone(), expression.clone());
            }
            BlockItem::ForwardRegister { ty, .. } => {
                *ty = substitute_identifiers(ty, &values);
            }
            BlockItem::Register(register) => {
                if let Some(ty) = &mut register.ty {
                    *ty = substitute_identifiers(ty, &values);
                }
                if let Some(expression) = &mut register.init_value {
                    *expression = substitute_identifiers(expression, &values);
                }
                if let Some(enable) = &mut register.enable {
                    *enable = substitute_identifiers(enable, &values);
                }
                register.next = substitute_identifiers(&register.next, &values);
            }
            BlockItem::Assign { expression, .. } => {
                *expression = substitute_identifiers(expression, &values);
            }
            BlockItem::Instance(instance) => {
                if let Some(parametrics) = &mut instance.parametrics {
                    *parametrics = substitute_identifiers(parametrics, &values);
                }
                for binding in &mut instance.bindings {
                    binding.expression = substitute_identifiers(&binding.expression, &values);
                }
            }
            BlockItem::Assert { predicate, .. }
            | BlockItem::Cover { predicate, .. }
            | BlockItem::ConstAssert { predicate, .. } => {
                *predicate = substitute_identifiers(predicate, &values);
            }
            BlockItem::Conditional { .. } => {
                // Structural conditionals are removed before this pass.
            }
        }
    }
}

fn eval_const_bool_with_dslx(
    prelude: &str,
    expression: &str,
    path: &Path,
    convert_options: &xlsynth::DslxConvertOptions<'_>,
    offset: usize,
) -> Result<bool, BlockDiagnostic> {
    let helper_name = unique_identifier(prelude, "__xlsynth_structural_condition");
    let mut source = prelude.to_string();
    source.push_str(&format!(
        "\nfn {helper_name}() -> bool {{ {expression} }}\n"
    ));
    let converted =
        xlsynth::convert_dslx_to_ir_text(&source, path, convert_options).map_err(|error| {
            BlockDiagnostic::new(
                path,
                Some(offset),
                format!("invalid structural constexpr condition '{expression}': {error}"),
            )
        })?;
    reject_reachable_ffi_in_constexpr(
        &converted.ir,
        path,
        &helper_name,
        offset,
        "structural constexpr condition",
    )?;
    let filename = path.file_name().and_then(|name| name.to_str());
    let package = xlsynth::IrPackage::parse_ir(&converted.ir, filename).map_err(|error| {
        BlockDiagnostic::new(
            path,
            Some(offset),
            format!("could not evaluate structural constexpr condition: {error}"),
        )
    })?;
    let module_name = xlsynth::dslx_path_to_module_name(path).map_err(|error| {
        BlockDiagnostic::new(path, Some(offset), format!("invalid source path: {error}"))
    })?;
    let mangled = xlsynth::mangle_dslx_name(module_name, &helper_name).map_err(|error| {
        BlockDiagnostic::new(
            path,
            Some(offset),
            format!("could not identify constexpr helper: {error}"),
        )
    })?;
    let function = package.get_function(&mangled).map_err(|error| {
        BlockDiagnostic::new(
            path,
            Some(offset),
            format!("could not load constexpr helper: {error}"),
        )
    })?;
    let value = function.interpret(&[]).map_err(|error| {
        BlockDiagnostic::new(
            path,
            Some(offset),
            format!("could not interpret structural constexpr condition: {error}"),
        )
    })?;
    let bits = value.to_bits().map_err(|error| {
        BlockDiagnostic::new(
            path,
            Some(offset),
            format!("structural constexpr condition is not boolean: {error}"),
        )
    })?;
    if bits.get_bit_count() != 1 {
        return Err(BlockDiagnostic::new(
            path,
            Some(offset),
            "structural constexpr condition did not produce one bit",
        ));
    }
    bits.get_bit(0).map_err(|error| {
        BlockDiagnostic::new(
            path,
            Some(offset),
            format!("could not read structural constexpr result: {error}"),
        )
    })
}

/// Rejects FFI calls in contexts evaluated by the DSLX interpreter. An FFI
/// function's DSLX body is only a conversion fallback and need not match its
/// Verilog implementation.
fn reject_reachable_ffi_in_constexpr(
    converted_ir: &str,
    path: &Path,
    helper_name: &str,
    offset: usize,
    context: &str,
) -> Result<(), BlockDiagnostic> {
    let module_name = xlsynth::dslx_path_to_module_name(path).map_err(|error| {
        BlockDiagnostic::new(path, Some(offset), format!("invalid source path: {error}"))
    })?;
    let mangled = xlsynth::mangle_dslx_name(module_name, helper_name).map_err(|error| {
        BlockDiagnostic::new(
            path,
            Some(offset),
            format!("could not identify constexpr helper: {error}"),
        )
    })?;
    let package = Parser::new(converted_ir).parse_package().map_err(|error| {
        BlockDiagnostic::new(
            path,
            Some(offset),
            format!("could not inspect {context}: {error}"),
        )
    })?;
    let ffi_names = collect_verilog_ffi_interfaces(&package)
        .into_keys()
        .collect::<BTreeSet<_>>();
    if let Some(target) = reachable_ffi_target(&package, &mangled, &ffi_names) {
        return Err(BlockDiagnostic::new(
            path,
            Some(offset),
            format!(
                "{context} reaches Verilog FFI function '{target}', whose DSLX fallback body cannot be used for elaboration"
            ),
        ));
    }
    Ok(())
}

fn resolve_interface(
    block: &BlockDecl,
    values: &BTreeMap<String, String>,
    path: &Path,
) -> Result<ResolvedInterface, BlockDiagnostic> {
    let mut names = block
        .params
        .iter()
        .map(|param| param.name.clone())
        .collect::<BTreeSet<_>>();
    let mut clock_count = 0usize;
    let mut reset_count = 0usize;
    let mut ports = Vec::new();
    for port in &block.ports {
        if !names.insert(port.name.clone()) {
            return Err(BlockDiagnostic::new(
                path,
                Some(port.offset),
                format!("symbol '{}' is declared more than once", port.name),
            ));
        }
        let normalized = normalize_type_syntax(&port.ty, path, port.offset)?;
        let role = if normalized == "clock" {
            if port.direction != PortDirection::Input {
                return Err(BlockDiagnostic::new(
                    path,
                    Some(port.offset),
                    "clock ports must be inputs",
                ));
            }
            clock_count += 1;
            PortRole::Clock
        } else if normalized == "reset<active_high,sync>" {
            if port.direction != PortDirection::Input {
                return Err(BlockDiagnostic::new(
                    path,
                    Some(port.offset),
                    "reset ports must be inputs",
                ));
            }
            reset_count += 1;
            PortRole::Reset { active_low: false }
        } else if normalized == "reset<active_low,sync>" {
            if port.direction != PortDirection::Input {
                return Err(BlockDiagnostic::new(
                    path,
                    Some(port.offset),
                    "reset ports must be inputs",
                ));
            }
            reset_count += 1;
            PortRole::Reset { active_low: true }
        } else if normalized.starts_with("reset<") {
            return Err(BlockDiagnostic::new(
                path,
                Some(port.offset),
                "the MVP supports only synchronous active_high/active_low reset types",
            ));
        } else {
            PortRole::Data
        };
        let source_ty = match role {
            PortRole::Clock => "clock".to_string(),
            PortRole::Reset { .. } => "bool".to_string(),
            PortRole::Data => substitute_identifiers(&port.ty, values),
        };
        ports.push(ResolvedPort {
            direction: port.direction,
            name: port.name.clone(),
            source_ty,
            role,
        });
    }
    if clock_count != 1 || reset_count != 1 {
        return Err(BlockDiagnostic::new(
            path,
            Some(block.offset),
            format!(
                "block '{}' must have exactly one input clock and one input synchronous reset",
                block.name
            ),
        ));
    }
    Ok(ResolvedInterface {
        name: block.name.clone(),
        ports,
    })
}

/// Enforces block-local lexical ordering against the original authored
/// expressions. This must run before parametric/local-constant substitution,
/// because transformed expression lengths cannot be mapped back to exact
/// source byte offsets without retaining a full rewrite source map.
fn analyze_block_declarations(
    block: &BlockDecl,
    path: &Path,
) -> Result<DeclarationAnalysis, BlockDiagnostic> {
    let mut declared = block
        .params
        .iter()
        .map(|param| param.name.clone())
        .chain(block.ports.iter().map(|port| port.name.clone()))
        .collect::<BTreeSet<_>>();
    let outputs = block
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
        .map(|port| (port.name.clone(), port.ty.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut all_local_symbols = declared.clone();
    for item in &block.items {
        match item {
            BlockItem::Let { names, .. } => all_local_symbols.extend(names.iter().cloned()),
            BlockItem::Const { name, .. } | BlockItem::ForwardRegister { name, .. } => {
                all_local_symbols.insert(name.clone());
            }
            BlockItem::Register(register) => {
                all_local_symbols.insert(register.name.clone());
            }
            BlockItem::Instance(instance) => {
                all_local_symbols.insert(instance.name.clone());
            }
            BlockItem::Assign { .. }
            | BlockItem::Assert { .. }
            | BlockItem::Cover { .. }
            | BlockItem::ConstAssert { .. }
            | BlockItem::Conditional { .. } => {}
        }
    }
    let mut forward_registers = BTreeSet::new();
    for item in &block.items {
        match item {
            BlockItem::Let {
                names,
                expression,
                expression_offset,
                offset,
                ..
            } => {
                validate_expression_order(
                    expression,
                    &declared,
                    &all_local_symbols,
                    &outputs,
                    *expression_offset,
                    path,
                )?;
                for name in names {
                    declare_symbol(&mut declared, name, *offset, path)?;
                }
            }
            BlockItem::Const {
                name,
                expression,
                expression_offset,
                offset,
                ..
            } => {
                validate_expression_order(
                    expression,
                    &declared,
                    &all_local_symbols,
                    &outputs,
                    *expression_offset,
                    path,
                )?;
                declare_symbol(&mut declared, name, *offset, path)?;
            }
            BlockItem::ForwardRegister { name, offset, .. } => {
                declare_symbol(&mut declared, name, *offset, path)?;
                forward_registers.insert(name.clone());
            }
            BlockItem::Register(register) => {
                if !forward_registers.contains(&register.name) {
                    declare_symbol(&mut declared, &register.name, register.offset, path)?;
                }
                if let Some(expression) = &register.init_value {
                    validate_expression_order(
                        expression,
                        &declared,
                        &all_local_symbols,
                        &outputs,
                        register.init_value_offset.unwrap_or(register.offset),
                        path,
                    )?;
                }
                if let Some(enable) = &register.enable {
                    validate_expression_order(
                        enable,
                        &declared,
                        &all_local_symbols,
                        &outputs,
                        register.enable_offset.unwrap_or(register.offset),
                        path,
                    )?;
                }
                validate_expression_order(
                    &register.next,
                    &declared,
                    &all_local_symbols,
                    &outputs,
                    register.next_offset,
                    path,
                )?;
            }
            BlockItem::Assign {
                expression,
                expression_offset,
                ..
            } => validate_expression_order(
                expression,
                &declared,
                &all_local_symbols,
                &outputs,
                *expression_offset,
                path,
            )?,
            BlockItem::Instance(instance) => {
                for binding in &instance.bindings {
                    validate_expression_order(
                        &binding.expression,
                        &declared,
                        &all_local_symbols,
                        &outputs,
                        binding.expression_offset,
                        path,
                    )?;
                }
                declare_symbol(&mut declared, &instance.name, instance.offset, path)?;
            }
            BlockItem::Assert {
                predicate,
                predicate_offset,
                ..
            }
            | BlockItem::Cover {
                predicate,
                predicate_offset,
                ..
            }
            | BlockItem::ConstAssert {
                predicate,
                predicate_offset,
            } => validate_expression_order(
                predicate,
                &declared,
                &all_local_symbols,
                &outputs,
                *predicate_offset,
                path,
            )?,
            BlockItem::Conditional { offset, .. } => {
                return Err(BlockDiagnostic::new(
                    path,
                    Some(*offset),
                    "internal error: structural if was not elaborated",
                ));
            }
        }
    }
    Ok(DeclarationAnalysis {
        all_symbols: all_local_symbols,
    })
}

fn build_helper(
    block: &BlockDecl,
    interface: ResolvedInterface,
    values: &BTreeMap<String, String>,
    prior_interfaces: &BTreeMap<String, ResolvedInterface>,
    helper_name: String,
    dslx_prelude: &str,
    convert_options: &xlsynth::DslxConvertOptions<'_>,
    path: &Path,
    declaration_analysis: &DeclarationAnalysis,
) -> Result<(String, BlockLayout), BlockDiagnostic> {
    let outputs = interface
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
        .map(|port| (port.name.clone(), port.source_ty.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut output_expressions = BTreeMap::<String, String>::new();
    let mut forward_registers = BTreeMap::<String, String>::new();
    let mut registers = Vec::<RegisterSemantic>::new();
    let mut completed_registers = BTreeSet::new();
    let mut instances = Vec::<ResolvedInstance>::new();
    let mut let_statements = Vec::new();
    let mut assertions = Vec::new();
    let mut covers = Vec::new();
    let mut occupied_helper_names = declaration_analysis.all_symbols.clone();

    for item in &block.items {
        match item {
            BlockItem::Let { statement, .. } => {
                let_statements.push(substitute_identifiers(statement, values));
            }
            BlockItem::Const { statement, .. } => {
                let_statements.push(substitute_identifiers(statement, values));
            }
            BlockItem::ForwardRegister { name, ty, .. } => {
                forward_registers.insert(name.clone(), substitute_identifiers(ty, values));
            }
            BlockItem::Register(register) => {
                let source_ty = resolve_register_type(register, &forward_registers, path, values)?;
                if let (Some(forward_ty), Some(inline_ty)) = (
                    forward_registers.get(&register.name),
                    register.ty.as_deref(),
                ) {
                    validate_repeated_register_type(
                        dslx_prelude,
                        &block.name,
                        register,
                        forward_ty,
                        inline_ty,
                        path,
                        convert_options,
                    )?;
                }
                if !completed_registers.insert(register.name.clone()) {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(register.offset),
                        format!("register '{}' has more than one contract", register.name),
                    ));
                }
                registers.push(RegisterSemantic {
                    name: register.name.clone(),
                    source_ty,
                    init_value: register
                        .init_value
                        .as_deref()
                        .map(|expression| substitute_identifiers(expression, values)),
                    enable: substitute_identifiers(
                        register.enable.as_deref().unwrap_or("true"),
                        values,
                    ),
                    next: substitute_identifiers(&register.next, values),
                });
            }
            BlockItem::Assign {
                target,
                expression,
                offset,
                ..
            } => {
                if !outputs.contains_key(target) {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(*offset),
                        format!("assign target '{target}' is not an output port"),
                    ));
                }
                if output_expressions
                    .insert(target.clone(), substitute_identifiers(expression, values))
                    .is_some()
                {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(*offset),
                        format!("output '{target}' is assigned more than once"),
                    ));
                }
            }
            BlockItem::Instance(instance) => {
                if !crate::sv::is_valid_system_verilog_identifier(&instance.name) {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(instance.offset),
                        format!(
                            "instance name '{}' is reserved by the SystemVerilog backend; choose a different name",
                            instance.name
                        ),
                    ));
                }
                if !crate::sv::is_valid_xls_ir_instantiation_identifier(&instance.name) {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(instance.offset),
                        format!(
                            "instance name '{}' is reserved by the XLS IR backend; choose a different name",
                            instance.name
                        ),
                    ));
                }
                if instance
                    .parametrics
                    .as_deref()
                    .is_some_and(|text| !text.is_empty())
                {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(instance.offset),
                        "parametric block instances are deferred in the MVP; instantiate a concretely declared helper block",
                    ));
                }
                let target = prior_interfaces.get(&instance.target).cloned().ok_or_else(|| {
                    BlockDiagnostic::new(
                        path,
                        Some(instance.offset),
                        format!(
                            "instance target '{}' was not found as a previously declared block or proc",
                            instance.target
                        ),
                    )
                })?;
                let parent_reset = interface
                    .ports
                    .iter()
                    .find_map(|port| match port.role {
                        PortRole::Reset { active_low } => Some(active_low),
                        _ => None,
                    })
                    .expect("interface validation requires reset");
                let child_reset = target
                    .ports
                    .iter()
                    .find_map(|port| match port.role {
                        PortRole::Reset { active_low } => Some(active_low),
                        _ => None,
                    })
                    .expect("resolved child interface has a reset");
                if parent_reset != child_reset {
                    return Err(BlockDiagnostic::new(
                        path,
                        Some(instance.offset),
                        format!(
                            "instance '{}' reset polarity does not match its parent block",
                            instance.name
                        ),
                    ));
                }
                validate_instance_bindings(instance, &target, path)?;
                instances.push(ResolvedInstance {
                    declaration: instance.clone(),
                    target,
                });
            }
            BlockItem::Assert {
                predicate, label, ..
            } => assertions.push((substitute_identifiers(predicate, values), label.clone())),
            BlockItem::Cover {
                predicate, label, ..
            } => covers.push((substitute_identifiers(predicate, values), label.clone())),
            BlockItem::ConstAssert { predicate, .. } => {
                let_statements.push(format!(
                    "const_assert!({});",
                    substitute_identifiers(predicate, values)
                ));
            }
            BlockItem::Conditional { offset, .. } => {
                return Err(BlockDiagnostic::new(
                    path,
                    Some(*offset),
                    "internal error: structural if was not elaborated",
                ));
            }
        }
    }

    for name in forward_registers.keys() {
        if !completed_registers.contains(name) {
            return Err(BlockDiagnostic::new(
                path,
                Some(block.offset),
                format!("forward-declared register '{name}' has no reg contract"),
            ));
        }
    }
    for name in outputs.keys() {
        if !output_expressions.contains_key(name) {
            return Err(BlockDiagnostic::new(
                path,
                Some(block.offset),
                format!("output '{name}' has no assign driver"),
            ));
        }
    }

    let mut instance_output_names = BTreeMap::new();
    for instance in &instances {
        for port in instance
            .target
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
        {
            let synthetic_name = fresh_identifier(
                &format!("__xlsynth_{}_{}", instance.declaration.name, port.name),
                &mut occupied_helper_names,
            );
            instance_output_names.insert(
                (instance.declaration.name.clone(), port.name.clone()),
                synthetic_name,
            );
        }
    }
    let rewrite = |expression: &str| {
        let expression = rewrite_instance_outputs(expression, &instance_output_names);
        brace_dslx_generic_args(&substitute_identifiers(&expression, values))
    };

    let input_ports = interface
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Input && port.role != PortRole::Clock)
        .collect::<Vec<_>>();
    let input_param_count = input_ports.len();
    let mut params = input_ports
        .iter()
        .map(|port| (port.name.clone(), port.source_ty.clone()))
        .collect::<Vec<_>>();
    params.extend(
        registers
            .iter()
            .map(|register| (register.name.clone(), register.source_ty.clone())),
    );

    let mut instance_layouts = Vec::new();
    for instance in &instances {
        let mut output_params = Vec::new();
        for port in instance
            .target
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
        {
            let synthetic = instance_output_names
                .get(&(instance.declaration.name.clone(), port.name.clone()))
                .expect("instance output name was created")
                .clone();
            let param_index = params.len();
            params.push((synthetic.clone(), port.source_ty.clone()));
            output_params.push((port.name.clone(), synthetic, param_index));
        }
        instance_layouts.push(InstanceLayout {
            name: instance.declaration.name.clone(),
            target: instance.target.name.clone(),
            reset_port_name: instance
                .target
                .ports
                .iter()
                .find_map(|port| match port.role {
                    PortRole::Reset { .. } => Some(port.name.clone()),
                    _ => None,
                })
                .expect("resolved child interface has a reset"),
            input_results: Vec::new(),
            output_params,
        });
    }

    let mut results = Vec::<(String, String)>::new();
    let mut output_results = Vec::new();
    for port in interface
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
    {
        let result = results.len();
        results.push((
            port.source_ty.clone(),
            rewrite(
                output_expressions
                    .get(&port.name)
                    .expect("missing outputs were diagnosed"),
            ),
        ));
        output_results.push((port.name.clone(), result));
    }

    for (layout, instance) in instance_layouts.iter_mut().zip(&instances) {
        for port in instance.target.ports.iter().filter(|port| {
            port.direction == PortDirection::Input
                && !matches!(port.role, PortRole::Clock | PortRole::Reset { .. })
        }) {
            let expression = instance
                .declaration
                .bindings
                .iter()
                .find(|binding| binding.port == port.name)
                .expect("instance bindings were validated")
                .expression
                .as_str();
            let result = results.len();
            results.push((port.source_ty.clone(), rewrite(expression)));
            layout.input_results.push((port.name.clone(), result));
        }
    }

    let mut register_layouts = Vec::new();
    for register in &registers {
        let next_result = results.len();
        results.push((register.source_ty.clone(), rewrite(&register.next)));
        let enable_result = results.len();
        results.push(("bool".to_string(), rewrite(&register.enable)));
        let init_value_result = match &register.init_value {
            Some(expression) => {
                let result = results.len();
                results.push((register.source_ty.clone(), rewrite(expression)));
                Some(result)
            }
            None => None,
        };
        register_layouts.push(RegisterLayout {
            name: register.name.clone(),
            next_result,
            enable_result,
            init_value_result,
        });
    }

    let mut assertion_layouts = Vec::new();
    for (index, (predicate, label)) in assertions.into_iter().enumerate() {
        let result = results.len();
        results.push(("bool".to_string(), rewrite(&predicate)));
        assertion_layouts.push(PredicateLayout {
            result,
            emitted_label: property_identifier("assert", index, &label),
            label,
        });
    }
    let mut cover_layouts = Vec::new();
    for (index, (predicate, label)) in covers.into_iter().enumerate() {
        let result = results.len();
        results.push(("bool".to_string(), rewrite(&predicate)));
        cover_layouts.push(PredicateLayout {
            result,
            emitted_label: property_identifier("cover", index, &label),
            label,
        });
    }

    let concrete_lets = let_statements
        .iter()
        .map(|statement| rewrite(statement))
        .collect::<Vec<_>>();
    let preserved_let_bindings = block
        .items
        .iter()
        .filter_map(|item| {
            let BlockItem::Let {
                names, expression, ..
            } = item
            else {
                return None;
            };
            let simple_alias = (names.len() == 1)
                .then(|| expression.trim())
                .filter(|rhs| xlsynth_pir::ir_utils::is_valid_identifier_name(rhs))
                .map(str::to_string);
            Some(
                names
                    .iter()
                    .map(|name| (name.clone(), simple_alias.clone()))
                    .collect::<Vec<_>>(),
            )
        })
        .flatten()
        .collect::<Vec<_>>();
    let helper = emit_helper(&helper_name, &params, &results, &concrete_lets);
    let reset = interface
        .ports
        .iter()
        .find_map(|port| match port.role {
            PortRole::Reset { active_low } => Some((port.name.clone(), active_low)),
            _ => None,
        })
        .expect("interface validation requires reset");
    Ok((
        helper,
        BlockLayout {
            source_name: block.name.clone(),
            helper_name,
            interface,
            input_param_count,
            registers: register_layouts,
            instances: instance_layouts,
            ffi_calls: Vec::new(),
            output_results,
            preserved_let_bindings,
            assertions: assertion_layouts,
            covers: cover_layouts,
            reset_name: reset.0,
            reset_active_low: reset.1,
        },
    ))
}

/// Checks the block-local declaration-order rules without attempting to
/// duplicate ordinary DSLX name resolution.
fn validate_expression_order(
    expression: &str,
    declared: &BTreeSet<String>,
    all_local_symbols: &BTreeSet<String>,
    outputs: &BTreeMap<String, String>,
    offset: usize,
    path: &Path,
) -> Result<(), BlockDiagnostic> {
    let nested_bindings = nested_expression_bindings(expression, path, offset)?;
    let tokens = lex_expression(expression, path, offset)?;
    for (index, token) in tokens.iter().enumerate() {
        if !is_identifier_token(token) {
            continue;
        }
        let identifier = token.text.as_str();
        // Comments and whitespace are not tokens, matching ordinary DSLX
        // lexical behavior around member and qualified-name operators.
        let is_member = tokens
            .get(index.wrapping_sub(1))
            .is_some_and(|previous| previous.text == ".");
        let is_qualified_tail = tokens
            .get(index.wrapping_sub(1))
            .is_some_and(|previous| previous.text == "::");
        let is_field_label = tokens.get(index + 1).is_some_and(|next| next.text == ":");
        let is_nested_binding = nested_bindings.iter().any(|binding| {
            binding.name == identifier
                && ((token.start >= binding.declaration_start
                    && token.start < binding.declaration_end)
                    || (token.start >= binding.scope_start && token.start < binding.scope_end))
        });
        if is_member || is_qualified_tail || is_field_label || is_nested_binding {
            continue;
        }
        if outputs.contains_key(identifier) {
            return Err(BlockDiagnostic::new(
                path,
                Some(offset + token.start),
                format!("output '{identifier}' is write-only inside a block"),
            ));
        }
        if all_local_symbols.contains(identifier) && !declared.contains(identifier) {
            return Err(BlockDiagnostic::new(
                path,
                Some(offset + token.start),
                format!("symbol '{identifier}' is used before its declaration"),
            ));
        }
    }
    Ok(())
}

fn lex_expression(
    expression: &str,
    path: &Path,
    offset: usize,
) -> Result<Vec<parse::Token>, BlockDiagnostic> {
    parse::lex(expression, path).map_err(|mut diagnostic| {
        diagnostic.offset = diagnostic
            .offset
            .and_then(|relative| offset.checked_add(relative));
        diagnostic
    })
}

fn is_identifier_token(token: &parse::Token) -> bool {
    token
        .text
        .bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && token
            .text
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

#[derive(Debug)]
struct NestedExpressionBinding {
    name: String,
    declaration_start: usize,
    declaration_end: usize,
    scope_start: usize,
    scope_end: usize,
}

/// Returns whether a token is a nested binding declaration or scoped use.
fn token_is_nested_binding(token: &parse::Token, bindings: &[NestedExpressionBinding]) -> bool {
    bindings.iter().any(|binding| {
        binding.name == token.text
            && ((token.start >= binding.declaration_start && token.start < binding.declaration_end)
                || (token.start >= binding.scope_start && token.start < binding.scope_end))
    })
}

/// Finds DSLX `for` and `match` pattern bindings with their lexical scopes so
/// the block-level use-before-declaration check considers only free names.
fn nested_expression_bindings(
    expression: &str,
    path: &Path,
    offset: usize,
) -> Result<Vec<NestedExpressionBinding>, BlockDiagnostic> {
    let tokens = lex_expression(expression, path, offset)?;
    let mut bindings = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.text == "for" {
            let Some(in_index) = tokens[index + 1..]
                .iter()
                .position(|candidate| candidate.text == "in")
                .map(|relative| index + 1 + relative)
            else {
                continue;
            };
            let pattern_end = tokens[index + 1..in_index]
                .iter()
                .position(|candidate| candidate.text == ":")
                .map(|relative| index + 1 + relative)
                .unwrap_or(in_index);
            let Some(open) = tokens[in_index + 1..]
                .iter()
                .position(|candidate| candidate.text == "{")
                .map(|relative| in_index + 1 + relative)
            else {
                continue;
            };
            let Some(close) = matching_token_delimiter(&tokens, open, "{", "}") else {
                continue;
            };
            let declaration_start = tokens[index + 1].start;
            let declaration_end = tokens[pattern_end.saturating_sub(1)].end;
            for name in pattern_binding_names(&tokens[index + 1..pattern_end]) {
                bindings.push(NestedExpressionBinding {
                    name,
                    declaration_start,
                    declaration_end,
                    scope_start: tokens[open].start,
                    scope_end: tokens[close].end,
                });
            }
        } else if token.text == "match" {
            let Some(open) = tokens[index + 1..]
                .iter()
                .position(|candidate| candidate.text == "{")
                .map(|relative| index + 1 + relative)
            else {
                continue;
            };
            let Some(close) = matching_token_delimiter(&tokens, open, "{", "}") else {
                continue;
            };
            let mut arm_start = open + 1;
            let mut cursor = arm_start;
            let mut depth = 0usize;
            while cursor <= close {
                let is_arm_end = cursor == close || (tokens[cursor].text == "," && depth == 0);
                if is_arm_end {
                    if let Some(arrow) = (arm_start..cursor.saturating_sub(1)).find(|candidate| {
                        tokens[*candidate].text == "=" && tokens[*candidate + 1].text == ">"
                    }) {
                        let pattern = &tokens[arm_start..arrow];
                        let declaration_start = pattern
                            .first()
                            .map(|token| token.start)
                            .unwrap_or(tokens[arrow].start);
                        let declaration_end = pattern
                            .last()
                            .map(|token| token.end)
                            .unwrap_or(tokens[arrow].start);
                        let scope_start = tokens[arrow + 1].end;
                        let scope_end = if cursor == close {
                            tokens[close].start
                        } else {
                            tokens[cursor].start
                        };
                        for name in pattern_binding_names(pattern) {
                            bindings.push(NestedExpressionBinding {
                                name,
                                declaration_start,
                                declaration_end,
                                scope_start,
                                scope_end,
                            });
                        }
                    }
                    arm_start = cursor + 1;
                }
                if cursor == close {
                    break;
                }
                match tokens[cursor].text.as_str() {
                    "(" | "[" | "{" => depth += 1,
                    ")" | "]" | "}" => depth = depth.saturating_sub(1),
                    _ => {}
                }
                cursor += 1;
            }
        }
    }
    Ok(bindings)
}

fn matching_token_delimiter(
    tokens: &[parse::Token],
    open: usize,
    open_text: &str,
    close_text: &str,
) -> Option<usize> {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        if token.text == open_text {
            depth += 1;
        } else if token.text == close_text {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn pattern_binding_names(tokens: &[parse::Token]) -> Vec<String> {
    tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| {
            let first = token.text.as_bytes().first().copied()?;
            if !(first.is_ascii_alphabetic() || first == b'_') || token.text == "_" {
                return None;
            }
            let qualified = tokens
                .get(index.wrapping_sub(1))
                .is_some_and(|previous| previous.text == "::")
                || tokens.get(index + 1).is_some_and(|next| next.text == "::");
            (!qualified && !matches!(token.text.as_str(), "true" | "false"))
                .then(|| token.text.clone())
        })
        .collect()
}

fn resolve_register_type(
    register: &RegisterDecl,
    forward: &BTreeMap<String, String>,
    path: &Path,
    values: &BTreeMap<String, String>,
) -> Result<String, BlockDiagnostic> {
    if let Some(forward_ty) = forward.get(&register.name) {
        Ok(forward_ty.clone())
    } else {
        register
            .ty
            .as_ref()
            .map(|ty| substitute_identifiers(ty, values))
            .ok_or_else(|| {
                BlockDiagnostic::new(
                    path,
                    Some(register.offset),
                    format!(
                        "inline register '{}' needs a type because it has no declreg",
                        register.name
                    ),
                )
            })
    }
}

fn validate_instance_bindings(
    instance: &InstanceDecl,
    target: &ResolvedInterface,
    path: &Path,
) -> Result<(), BlockDiagnostic> {
    let required = target
        .ports
        .iter()
        .filter(|port| {
            port.direction == PortDirection::Input
                && !matches!(port.role, PortRole::Clock | PortRole::Reset { .. })
        })
        .map(|port| port.name.as_str())
        .collect::<BTreeSet<_>>();
    let provided = instance
        .bindings
        .iter()
        .map(|binding| binding.port.as_str())
        .collect::<BTreeSet<_>>();
    if required != provided {
        let missing = required.difference(&provided).copied().collect::<Vec<_>>();
        let unknown = provided.difference(&required).copied().collect::<Vec<_>>();
        return Err(BlockDiagnostic::new(
            path,
            Some(instance.offset),
            format!(
                "instance '{}' port mismatch; missing {:?}, unknown {:?}",
                instance.name, missing, unknown
            ),
        ));
    }
    Ok(())
}

fn declare_symbol(
    declared: &mut BTreeSet<String>,
    name: &str,
    offset: usize,
    path: &Path,
) -> Result<(), BlockDiagnostic> {
    if declared.insert(name.to_string()) {
        Ok(())
    } else {
        Err(BlockDiagnostic::new(
            path,
            Some(offset),
            format!("symbol '{name}' is already declared; shadowing is not allowed"),
        ))
    }
}

fn emit_helper(
    helper_name: &str,
    params: &[(String, String)],
    results: &[(String, String)],
    lets: &[String],
) -> String {
    let params = params
        .iter()
        .map(|(name, ty)| format!("{name}: {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    let result_type = tuple_text(results.iter().map(|(ty, _)| ty.as_str()));
    let result_value = tuple_text(results.iter().map(|(_, expression)| expression.as_str()));
    let mut out = format!("pub fn {helper_name}({params}) -> {result_type} {{\n");
    for statement in lets {
        out.push_str("  ");
        out.push_str(statement.trim());
        out.push('\n');
    }
    out.push_str("  ");
    out.push_str(&result_value);
    out.push_str("\n}\n");
    out
}

fn tuple_text<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let items = items.collect::<Vec<_>>();
    match items.as_slice() {
        [] => "()".to_string(),
        [only] => format!("({only},)"),
        _ => format!("({})", items.join(", ")),
    }
}

fn lower_helper_to_block(
    mut function: ir::Fn,
    layout: &BlockLayout,
    preserve_names: bool,
    path: &Path,
) -> Result<PackageMember, BlockDiagnostic> {
    let expected_params = layout.input_param_count
        + layout.registers.len()
        + layout
            .instances
            .iter()
            .map(|instance| instance.output_params.len())
            .sum::<usize>()
        + layout.ffi_calls.len();
    if function.params.len() != expected_params {
        return Err(BlockDiagnostic::new(
            path,
            None,
            format!(
                "helper '{}' has {} IR parameters, expected {expected_params}",
                layout.helper_name,
                function.params.len()
            ),
        ));
    }

    for (index, register) in layout.registers.iter().enumerate() {
        let node_index = 1 + layout.input_param_count + index;
        let node = &mut function.nodes[node_index];
        node.name = Some(register.name.clone());
        node.payload = NodePayload::RegisterRead {
            register: register.name.clone(),
        };
    }
    let mut output_param_base = 1 + layout.input_param_count + layout.registers.len();
    for instance in &layout.instances {
        for (port_name, synthetic_name, _) in &instance.output_params {
            let node = &mut function.nodes[output_param_base];
            node.name = Some(synthetic_name.clone());
            node.payload = NodePayload::InstantiationOutput {
                instantiation: instance.name.clone(),
                port_name: port_name.clone(),
            };
            output_param_base += 1;
        }
    }
    for ffi_call in &layout.ffi_calls {
        let node = &mut function.nodes[output_param_base];
        if node.name.as_deref() != Some(&ffi_call.output_param_name) {
            return Err(BlockDiagnostic::new(
                path,
                None,
                format!(
                    "optimized helper parameter '{}' was not retained in order",
                    ffi_call.output_param_name
                ),
            ));
        }
        node.payload = NodePayload::InstantiationOutput {
            instantiation: ffi_call.name.clone(),
            port_name: "return".to_string(),
        };
        output_param_base += 1;
    }
    function.params.truncate(layout.input_param_count);

    let mut occupied_node_names = function
        .nodes
        .iter()
        .filter_map(|node| node.name.clone())
        .chain(layout.interface.ports.iter().map(|port| port.name.clone()))
        .chain(
            layout
                .registers
                .iter()
                .map(|register| register.name.clone()),
        )
        .chain(
            layout
                .instances
                .iter()
                .map(|instance| instance.name.clone()),
        )
        .chain(layout.ffi_calls.iter().map(|call| call.name.clone()))
        .chain(
            layout
                .preserved_let_bindings
                .iter()
                .map(|(name, _)| name.clone()),
        )
        .collect::<BTreeSet<_>>();

    let original_ret = function
        .ret_node_ref
        .ok_or_else(|| BlockDiagnostic::new(path, None, "generated helper has no return value"))?;
    let result_types = match &function.ret_ty {
        Type::Tuple(types) => types.iter().map(|ty| (**ty).clone()).collect::<Vec<_>>(),
        other => {
            return Err(BlockDiagnostic::new(
                path,
                None,
                format!("generated helper must return a tuple, got {other}"),
            ));
        }
    };
    let result_refs = match &function.get_node(original_ret).payload {
        NodePayload::Tuple(elements) => elements.clone(),
        _ => {
            let mut results = Vec::with_capacity(result_types.len());
            for (index, ty) in result_types.iter().enumerate() {
                let name = fresh_identifier(
                    &format!("__xlsynth_result_{index}"),
                    &mut occupied_node_names,
                );
                results.push(push_node(
                    &mut function,
                    Some(name),
                    ty.clone(),
                    NodePayload::TupleIndex {
                        tuple: original_ret,
                        index,
                    },
                ));
            }
            results
        }
    };
    if preserve_names {
        let preserved_base = result_refs
            .len()
            .checked_sub(layout.preserved_let_bindings.len())
            .expect("preserved let results were appended to the helper tuple");
        for ((name, _), value) in layout
            .preserved_let_bindings
            .iter()
            .zip(result_refs[preserved_base..].iter().copied())
        {
            if function.get_node(value).name.as_deref() != Some(name) {
                let ty = function.get_node(value).ty.clone();
                push_node(
                    &mut function,
                    Some(name.clone()),
                    ty,
                    NodePayload::Unop(ir::Unop::Identity, value),
                );
            }
        }
    }

    let reset_ref = function
        .params
        .iter()
        .enumerate()
        .find(|(_, param)| param.name == layout.reset_name)
        .map(|(index, _)| NodeRef { index: index + 1 })
        .ok_or_else(|| {
            BlockDiagnostic::new(
                path,
                None,
                format!("reset input '{}' is missing from helper", layout.reset_name),
            )
        })?;
    let reset_active = if layout.reset_active_low {
        let name = fresh_identifier("__xlsynth_reset_active", &mut occupied_node_names);
        push_node(
            &mut function,
            Some(name),
            Type::Bits(1),
            NodePayload::Unop(ir::Unop::Not, reset_ref),
        )
    } else {
        reset_ref
    };
    let not_reset_name = fresh_identifier("__xlsynth_not_reset", &mut occupied_node_names);
    let not_reset = push_node(
        &mut function,
        Some(not_reset_name),
        Type::Bits(1),
        NodePayload::Unop(ir::Unop::Not, reset_active),
    );

    for instance in &layout.instances {
        for (port_name, result) in &instance.input_results {
            push_node(
                &mut function,
                None,
                Type::nil(),
                NodePayload::InstantiationInput {
                    instantiation: instance.name.clone(),
                    port_name: port_name.clone(),
                    arg: result_refs[*result],
                },
            );
        }
        push_node(
            &mut function,
            None,
            Type::nil(),
            NodePayload::InstantiationInput {
                instantiation: instance.name.clone(),
                port_name: instance.reset_port_name.clone(),
                arg: reset_ref,
            },
        );
    }

    for ffi_call in &layout.ffi_calls {
        for (port_name, result) in &ffi_call.input_results {
            push_node(
                &mut function,
                None,
                Type::nil(),
                NodePayload::InstantiationInput {
                    instantiation: ffi_call.name.clone(),
                    port_name: port_name.clone(),
                    arg: result_refs[*result],
                },
            );
        }
    }

    for register in &layout.registers {
        let next = result_refs[register.next_result];
        let enable = result_refs[register.enable_result];
        let (write_data, write_enable) = if let Some(init_value_result) = register.init_value_result
        {
            let init_value = result_refs[init_value_result];
            let next_ty = function.get_node(next).ty.clone();
            let reset_mux_name = fresh_identifier(
                &format!("{}_reset_mux", register.name),
                &mut occupied_node_names,
            );
            let data = push_node(
                &mut function,
                Some(reset_mux_name),
                next_ty,
                NodePayload::Sel {
                    selector: reset_active,
                    cases: vec![next, init_value],
                    default: None,
                },
            );
            let load_enable_name = fresh_identifier(
                &format!("{}_load_enable", register.name),
                &mut occupied_node_names,
            );
            let load_enable = push_node(
                &mut function,
                Some(load_enable_name),
                Type::Bits(1),
                NodePayload::Nary(ir::NaryOp::Or, vec![reset_active, enable]),
            );
            (data, load_enable)
        } else {
            (next, enable)
        };
        let write_name = fresh_identifier(
            &format!("{}_write", register.name),
            &mut occupied_node_names,
        );
        push_node(
            &mut function,
            Some(write_name),
            Type::nil(),
            NodePayload::RegisterWrite {
                arg: write_data,
                register: register.name.clone(),
                load_enable: Some(write_enable),
                reset: None,
            },
        );
    }

    let mut token = push_node(
        &mut function,
        None,
        Type::Token,
        NodePayload::AfterAll(Vec::new()),
    );
    for assertion in &layout.assertions {
        let emitted_label = fresh_identifier(&assertion.emitted_label, &mut occupied_node_names);
        let enabled_name = fresh_identifier(
            &format!("{emitted_label}_enabled"),
            &mut occupied_node_names,
        );
        let gated = push_node(
            &mut function,
            Some(enabled_name),
            Type::Bits(1),
            NodePayload::Nary(
                ir::NaryOp::Or,
                vec![reset_active, result_refs[assertion.result]],
            ),
        );
        token = push_node(
            &mut function,
            None,
            Type::Token,
            NodePayload::Assert {
                token,
                activate: gated,
                message: assertion.label.clone(),
                label: emitted_label,
            },
        );
    }
    for cover in &layout.covers {
        let emitted_label = fresh_identifier(&cover.emitted_label, &mut occupied_node_names);
        let enabled_name = fresh_identifier(
            &format!("{emitted_label}_enabled"),
            &mut occupied_node_names,
        );
        let gated = push_node(
            &mut function,
            Some(enabled_name),
            Type::Bits(1),
            NodePayload::Nary(ir::NaryOp::And, vec![not_reset, result_refs[cover.result]]),
        );
        push_node(
            &mut function,
            None,
            Type::nil(),
            NodePayload::Cover {
                predicate: gated,
                label: emitted_label,
            },
        );
    }

    let output_refs = layout
        .output_results
        .iter()
        .map(|(_, result)| result_refs[*result])
        .collect::<Vec<_>>();
    let (ret_ty, ret_node_ref) = match output_refs.as_slice() {
        [] => {
            let ty = Type::nil();
            let name = fresh_identifier("__xlsynth_no_outputs", &mut occupied_node_names);
            let node = push_node(
                &mut function,
                Some(name),
                ty.clone(),
                NodePayload::Tuple(Vec::new()),
            );
            (ty, Some(node))
        }
        [only] => (function.get_node(*only).ty.clone(), Some(*only)),
        _ => {
            let ty = Type::Tuple(
                output_refs
                    .iter()
                    .map(|node| Box::new(function.get_node(*node).ty.clone()))
                    .collect(),
            );
            let name = fresh_identifier("__xlsynth_outputs", &mut occupied_node_names);
            let node = push_node(
                &mut function,
                Some(name),
                ty.clone(),
                NodePayload::Tuple(output_refs),
            );
            (ty, Some(node))
        }
    };
    function.name = layout.source_name.clone();
    function.ret_ty = ret_ty;
    function.ret_node_ref = ret_node_ref;
    function.outer_attrs.clear();
    function.inner_attrs.clear();

    let output_type_by_name = layout
        .output_results
        .iter()
        .map(|(name, result)| {
            (
                name.clone(),
                function.get_node(result_refs[*result]).ty.clone(),
            )
        })
        .collect::<HashMap<_, _>>();
    let input_type_by_name = function
        .params
        .iter()
        .map(|param| (param.name.clone(), param.ty.clone()))
        .collect::<HashMap<_, _>>();
    let ports = layout
        .interface
        .ports
        .iter()
        .map(|port| match port.role {
            PortRole::Clock => BlockPort {
                name: port.name.clone(),
                kind: BlockPortKind::Clock,
                ty: None,
            },
            _ if port.direction == PortDirection::Input => BlockPort {
                name: port.name.clone(),
                kind: BlockPortKind::Input,
                ty: input_type_by_name.get(&port.name).cloned(),
            },
            _ => BlockPort {
                name: port.name.clone(),
                kind: BlockPortKind::Output,
                ty: output_type_by_name.get(&port.name).cloned(),
            },
        })
        .collect::<Vec<_>>();
    let output_names = layout
        .output_results
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let mut next_id = function
        .nodes
        .iter()
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0)
        + 1;
    let output_port_ids = output_names
        .iter()
        .map(|name| {
            let id = next_id;
            next_id += 1;
            (name.clone(), id)
        })
        .collect();
    let metadata = BlockMetadata {
        ports,
        clock_port_name: layout
            .interface
            .ports
            .iter()
            .find(|port| port.role == PortRole::Clock)
            .map(|port| port.name.clone()),
        input_port_ids: function
            .params
            .iter()
            .map(|param| (param.name.clone(), param.id.get_wrapped_id()))
            .collect(),
        output_port_ids,
        output_names,
        reset: Some(BlockResetMetadata {
            port_name: layout.reset_name.clone(),
            asynchronous: false,
            active_low: layout.reset_active_low,
        }),
        registers: layout
            .registers
            .iter()
            .enumerate()
            .map(|(index, register)| Register {
                name: register.name.clone(),
                ty: function.nodes[1 + layout.input_param_count + index]
                    .ty
                    .clone(),
                reset_value: None,
            })
            .collect(),
        instantiations: layout
            .instances
            .iter()
            .map(|instance| ir::Instantiation::block(&instance.name, &instance.target))
            .chain(layout.ffi_calls.iter().map(|ffi_call| {
                ir::Instantiation::extern_function(&ffi_call.name, &ffi_call.target)
            }))
            .collect(),
    };
    Ok(PackageMember::Block {
        func: function,
        metadata,
    })
}

fn push_node(
    function: &mut ir::Fn,
    name: Option<String>,
    ty: Type,
    payload: NodePayload,
) -> NodeRef {
    // This temporary id is replaced package-wide by renumber_package_ids()
    // before verification or emission. Using the append index avoids a full
    // node scan for every synthesized sink.
    let text_id = function.nodes.len();
    let node_ref = NodeRef {
        index: function.nodes.len(),
    };
    function.nodes.push(Node {
        text_id,
        name,
        ty,
        payload,
        pos: None,
    });
    node_ref
}

fn rewrite_instance_outputs(text: &str, names: &BTreeMap<(String, String), String>) -> String {
    let Ok(tokens) = parse::lex(text, Path::new("<instance-output-rewrite>")) else {
        // The source parser already lexed these fragments. Leave any
        // unexpected generated fragment intact for the official DSLX parser
        // to diagnose rather than rewriting it partially.
        return text.to_string();
    };
    let mut out = String::with_capacity(text.len());
    let nested_bindings =
        nested_expression_bindings(text, Path::new("<instance-output-rewrite>"), 0)
            .unwrap_or_default();
    let mut copied_through = 0usize;
    let mut index = 0usize;
    while index + 2 < tokens.len() {
        let first = &tokens[index];
        let dot = &tokens[index + 1];
        let second = &tokens[index + 2];
        let is_access_root = !tokens
            .get(index.wrapping_sub(1))
            .is_some_and(|previous| matches!(previous.text.as_str(), "." | "::"));
        if is_access_root
            && !token_is_nested_binding(first, &nested_bindings)
            && is_identifier_token(first)
            && dot.text == "."
            && is_identifier_token(second)
            && let Some(replacement) = names.get(&(first.text.clone(), second.text.clone()))
        {
            out.push_str(&text[copied_through..first.start]);
            out.push_str(replacement);
            copied_through = second.end;
            index += 3;
        } else {
            index += 1;
        }
    }
    out.push_str(&text[copied_through..]);
    out
}

fn substitute_identifiers(text: &str, values: &BTreeMap<String, String>) -> String {
    let Ok(tokens) = parse::lex(text, Path::new("<identifier-substitution>")) else {
        // See `rewrite_instance_outputs`: this fallback preserves the source
        // for the authoritative DSLX parser to diagnose.
        return text.to_string();
    };
    let mut out = String::with_capacity(text.len());
    let nested_bindings =
        nested_expression_bindings(text, Path::new("<identifier-substitution>"), 0)
            .unwrap_or_default();
    let mut copied_through = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        if is_identifier_token(token) {
            let qualified = tokens
                .get(index.wrapping_sub(1))
                .is_some_and(|previous| matches!(previous.text.as_str(), "::" | "."))
                || tokens.get(index + 1).is_some_and(|next| next.text == "::");
            let field_label = tokens.get(index + 1).is_some_and(|next| next.text == ":");
            let nested_binding = token_is_nested_binding(token, &nested_bindings);
            if let Some(value) = values
                .get(&token.text)
                .filter(|_| !qualified && !field_label && !nested_binding)
            {
                out.push_str(&text[copied_through..token.start]);
                // Parametric values are expressions, not tokens. Preserve
                // their precedence; complete DSLX generic arguments are
                // braced after substitution by `brace_dslx_generic_args`.
                out.push('(');
                out.push_str(value);
                out.push(')');
                copied_through = token.end;
            }
        }
    }
    out.push_str(&text[copied_through..]);
    out
}

/// Wraps explicit DSLX function generic arguments in const-argument braces.
/// This keeps substituted expressions grouped without relying on the DSLX
/// parser's comparison-vs-generic ambiguity after `<`.
fn brace_dslx_generic_args(text: &str) -> String {
    let Ok(tokens) = parse::lex(text, Path::new("<generated-expression>")) else {
        return text.to_string();
    };
    let mut insertions = Vec::<(usize, char)>::new();
    for open in 0..tokens.len() {
        if tokens[open].text != "<" || !parse::is_generic_open(&tokens, open) {
            continue;
        }
        let mut angle = 1usize;
        let mut paren = 0usize;
        let mut brace = 0usize;
        let mut bracket = 0usize;
        let mut close = None;
        for index in open + 1..tokens.len() {
            match tokens[index].text.as_str() {
                "(" => paren += 1,
                ")" => paren = paren.saturating_sub(1),
                "{" => brace += 1,
                "}" => brace = brace.saturating_sub(1),
                "[" => bracket += 1,
                "]" => bracket = bracket.saturating_sub(1),
                "<" if paren == 0
                    && brace == 0
                    && bracket == 0
                    && parse::is_generic_open(&tokens, index) =>
                {
                    angle += 1;
                }
                ">" if paren == 0 && brace == 0 && bracket == 0 => {
                    angle -= 1;
                    if angle == 0 {
                        if tokens.get(index + 1).is_some_and(|next| next.text == "(") {
                            close = Some(index);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else {
            continue;
        };
        for (start, end) in split_token_ranges(&tokens, open + 1, close, ",") {
            if start == end
                || (tokens[start].text == "{"
                    && matching_token_delimiter(&tokens, start, "{", "}") == Some(end - 1))
            {
                continue;
            }
            insertions.push((tokens[start].start, '{'));
            insertions.push((tokens[end - 1].end, '}'));
        }
    }
    insertions.sort_by_key(|(offset, character)| {
        (
            std::cmp::Reverse(*offset),
            std::cmp::Reverse(*character == '{'),
        )
    });
    let mut result = text.to_string();
    for (offset, character) in insertions {
        result.insert(offset, character);
    }
    result
}

fn split_token_ranges(
    tokens: &[parse::Token],
    start: usize,
    end: usize,
    separator: &str,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut item_start = start;
    let mut paren = 0usize;
    let mut brace = 0usize;
    let mut bracket = 0usize;
    for index in start..end {
        match tokens[index].text.as_str() {
            "(" => paren += 1,
            ")" => paren = paren.saturating_sub(1),
            "{" => brace += 1,
            "}" => brace = brace.saturating_sub(1),
            "[" => bracket += 1,
            "]" => bracket = bracket.saturating_sub(1),
            token if token == separator && paren == 0 && brace == 0 && bracket == 0 => {
                ranges.push((item_start, index));
                item_start = index + 1;
            }
            _ => {}
        }
    }
    ranges.push((item_start, end));
    ranges
}

/// Removes lexical trivia from a structural type without altering its tokens.
fn normalize_type_syntax(
    text: &str,
    path: &Path,
    offset: usize,
) -> Result<String, BlockDiagnostic> {
    lex_expression(text, path, offset).map(|tokens| {
        tokens
            .into_iter()
            .map(|token| token.text)
            .collect::<String>()
    })
}

fn split_parametric_arguments(text: &str, path: &Path) -> Result<Vec<String>, BlockDiagnostic> {
    let tokens = parse::lex(text, path)?;
    let mut arguments = Vec::new();
    let mut start = 0usize;
    let mut paren = 0usize;
    let mut brace = 0usize;
    let mut bracket = 0usize;
    let mut angle = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        match token.text.as_str() {
            "(" => paren += 1,
            ")" => paren = paren.saturating_sub(1),
            "{" => brace += 1,
            "}" => brace = brace.saturating_sub(1),
            "[" => bracket += 1,
            "]" => bracket = bracket.saturating_sub(1),
            "<" if paren == 0
                && brace == 0
                && bracket == 0
                && parse::is_generic_open(&tokens, index) =>
            {
                angle += 1;
            }
            ">" if paren == 0 && brace == 0 && bracket == 0 && angle != 0 => angle -= 1,
            "," if paren == 0 && brace == 0 && bracket == 0 && angle == 0 => {
                arguments.push(text[start..token.start].trim().to_string());
                start = token.end;
            }
            _ => {}
        }
    }
    if !text[start..].trim().is_empty() {
        arguments.push(text[start..].trim().to_string());
    }
    Ok(arguments)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use pretty_assertions::assert_eq;
    use xlsynth_pir::ir::{BlockPortKind, PackageMember};

    use super::*;

    fn compile(source: &str) -> BlockCompileOutput {
        compile_block_module(
            source,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn compiles_flow_register_to_native_block_ir() {
        let source = r#"
pub block br_flow_reg_fwd<WIDTH: u32 = {u32:8}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output push_ready: bool,
  input push_valid: bool,
  input push_data: uN[WIDTH],
  input pop_ready: bool,
  output pop_valid: bool,
  output pop_data: uN[WIDTH],
) {
  reg pop_valid_r: bool {
    init_value: false,
    next: push_valid || !(pop_ready || !pop_valid_r),
  }
  reg pop_data_r: uN[WIDTH] {
    en: (pop_ready || !pop_valid_r) && push_valid,
    next: push_data,
  }
  assign push_ready = pop_ready || !pop_valid_r;
  assign pop_valid = pop_valid_r;
  assign pop_data = pop_data_r;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("top block br_flow_reg_fwd"));
        assert!(output.ir_text.contains("reg pop_valid_r(bits[1])"));
        assert!(output.ir_text.contains("reg pop_data_r(bits[8])"));
        assert!(
            output
                .ir_text
                .contains("register_read(register=pop_valid_r")
        );
        assert!(output.ir_text.contains("register_write("));
        let top = output.package.get_top_block().unwrap();
        let PackageMember::Block { metadata, .. } = top else {
            unreachable!();
        };
        assert_eq!(
            metadata
                .ports
                .iter()
                .map(|port| (port.name.as_str(), port.kind))
                .collect::<Vec<_>>(),
            vec![
                ("clk", BlockPortKind::Clock),
                ("rst", BlockPortKind::Input),
                ("push_ready", BlockPortKind::Output),
                ("push_valid", BlockPortKind::Input),
                ("push_data", BlockPortKind::Input),
                ("pop_ready", BlockPortKind::Input),
                ("pop_valid", BlockPortKind::Output),
                ("pop_data", BlockPortKind::Output),
            ]
        );
    }

    #[test]
    fn applies_function_call_optimization_modes_without_scheduling() {
        let source = r#"
fn add_one(x: u8) -> u8 { x + u8:1 }

pub block wraps_function(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  assign y = add_one(x);
}
"#;
        let output = compile(source);
        assert!(!output.ir_text.contains("invoke("));
        assert!(output.ir_text.contains("add("));

        let options = BlockCompileOptions {
            combinational_optimization: CombinationalOptimization::PreserveNamesAndFunctions,
            ..BlockCompileOptions::default()
        };
        let output = compile_block_module(source, Path::new("block_test.x"), &options).unwrap();
        assert!(!output.ir_text.contains("invoke("));
        assert!(output.ir_text.contains("add("));
        assert!(
            output
                .warnings
                .iter()
                .any(|warning| warning.contains("XLS 0.53 codegen compatibility"))
        );
    }

    #[test]
    fn preserve_names_anchors_used_unused_and_alias_lets() {
        let source = r#"
pub block named_lets(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  let used = x + u8:1;
  let unused = x + u8:2;
  let alias = x;
  assign y = used;
}
"#;
        let free = compile(source);
        assert!(!free.ir_text.contains("unused:"));
        assert!(!free.ir_text.contains("alias:"));

        let options = BlockCompileOptions {
            combinational_optimization: CombinationalOptimization::PreserveNames,
            ..BlockCompileOptions::default()
        };
        let preserved = compile_block_module(source, Path::new("block_test.x"), &options).unwrap();
        assert!(preserved.ir_text.contains("used:"));
        assert!(preserved.ir_text.contains("unused:"));
        assert!(preserved.ir_text.contains("alias: bits[8] = identity(x"));
        assert!(!preserved.ir_text.contains("invoke("));
    }

    #[test]
    fn inlines_function_calls_before_attaching_runtime_properties() {
        let source = r#"
fn add_one(x: u8) -> u8 { x + u8:1 }

pub block property_wraps_function(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  let result = add_one(x);
  assert!(result != u8:0, "nonzero_result");
  cover!(result == u8:7, "result_seven");
  assign y = result;
}
"#;
        let output = compile(source);
        assert!(!output.ir_text.contains("invoke("));
        assert!(output.ir_text.contains("assert("));
        assert!(output.ir_text.contains("cover("));
    }

    #[test]
    fn preserves_explicit_generic_call_syntax_during_parametric_elaboration() {
        let source = r#"
fn identity<N: u32>(x: uN[N]) -> uN[N] { x }

pub block wraps_parametric<WIDTH: u32 = {u32:8}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: uN[WIDTH],
  output y: uN[WIDTH],
) {
  assign y = identity<WIDTH>(x);
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains(
            "top block wraps_parametric(clk: clock, rst: bits[1], x: bits[8], y: bits[8])"
        ));
        assert!(output.ir_text.contains("x: bits[8] = input_port"));
        assert!(output.ir_text.contains("y: () = output_port"));
    }

    #[test]
    fn supports_tuple_let_patterns_and_const_assertions() {
        let source = r#"
pub block tuple_binding(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  const OFFSET = u8:1;
  const_assert!(OFFSET == u8:1);
  let (first, second) = (x, x + OFFSET);
  assign y = first + second;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("add("));
    }

    #[test]
    fn elaborates_structural_constexpr_if_before_namespace_checks() {
        let source = r#"
pub block conditional<ENABLE: bool = {true}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  const LOCAL_ENABLE = ENABLE;
  if LOCAL_ENABLE {
    let selected = x + u8:1;
    assign y = selected;
  } else {
    assign y = x;
  }
}
"#;
        let enabled = compile(source);
        assert!(enabled.ir_text.contains("add("));

        let disabled_options = BlockCompileOptions {
            parametric_bindings: vec![ParametricBinding {
                name: "ENABLE".into(),
                value: "false".into(),
            }],
            ..BlockCompileOptions::default()
        };
        let disabled =
            compile_block_module(source, Path::new("block_test.x"), &disabled_options).unwrap();
        assert!(!disabled.ir_text.contains("add("));

        let function_condition = r#"
fn selected(value: u32) -> bool { value == u32:3 }

pub block conditional<N: u32 = {u32:3}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  if selected(N) {
    assign y = x + u8:1;
  } else {
    assign y = x;
  }
}
"#;
        let selected = compile(function_condition);
        assert!(selected.ir_text.contains("add("));
    }

    #[test]
    fn rejects_runtime_local_consts_and_runtime_const_assertions() {
        let dynamic_const = r#"
pub block bad_const(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  const VALUE = x;
  assign y = VALUE;
}
"#;
        let error = compile_block_module(
            dynamic_const,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("NotConstant"));

        let dynamic_assert = r#"
pub block bad_const_assert(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  const_assert!(x);
  assign y = x;
}
"#;
        let error = compile_block_module(
            dynamic_assert,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("NotConstant"));
    }

    #[test]
    fn rejects_duplicate_parametrics_and_param_port_collisions() {
        let duplicate = r#"
pub block duplicate<N: u32 = {u32:1}, N: u32 = {u32:2}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assign y = true;
}
"#;
        let error = compile_block_module(
            duplicate,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(
            error
                .message
                .contains("parametric binding 'N' is declared more than once")
        );

        let collision = r#"
pub block collision<N: u32 = {u32:1}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input N: u32,
  output y: u32,
) {
  assign y = N;
}
"#;
        let error = compile_block_module(
            collision,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(
            error
                .message
                .contains("symbol 'N' is declared more than once")
        );

        let duplicate_override = BlockCompileOptions {
            parametric_bindings: vec![
                ParametricBinding {
                    name: "N".into(),
                    value: "u32:1".into(),
                },
                ParametricBinding {
                    name: "N".into(),
                    value: "u32:2".into(),
                },
            ],
            ..BlockCompileOptions::default()
        };
        let error = compile_block_module(collision, Path::new("block_test.x"), &duplicate_override)
            .unwrap_err();
        assert!(
            error
                .message
                .contains("parametric binding 'N' is supplied more than once")
        );
    }

    #[test]
    fn typechecks_parametrics_and_materializes_nondefaulted_templates() {
        let invalid = r#"
pub block invalid<N: bool = {u32:7}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assign y = true;
}
"#;
        let error = compile_block_module(
            invalid,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("invalid parametric values"));

        let specialized = r#"
block child<N: u32>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: uN[N],
  output y: uN[N],
) {
  assign y = x;
}

pub block parent(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  inst child_i: child < u32:8 > { x: x, }
  assign y = child_i.y;
}
"#;
        let output = compile(specialized);
        assert!(output.ir_text.contains("block __xlsynth_spec_child_0"));
    }

    #[test]
    fn declaration_order_ignores_identifiers_in_block_comments() {
        let source = r#"
pub block comments(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  assign y = x /* later is intentionally mentioned before declaration */;
  let later = !x;
}
"#;
        compile(source);
    }

    #[test]
    fn keeps_qualified_names_intact_during_parametric_substitution() {
        let source = r#"
enum Choice : u2 { VALUE = 1 }

pub block qualified<VALUE: u32 = {u32:8}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: u2,
) {
  assign y = Choice::VALUE as u2;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("value=1"));
    }

    #[test]
    fn reset_is_readable_and_runtime_properties_are_reset_masked() {
        let source = r#"
pub block reset_reader(
  input data: bool,
  output in_reset: bool,
  input clk: clock,
  input rst: reset<active_low, sync>,
) {
  assert!(data, "data valid while active");
  cover!(data, "data seen while active");
  assign in_reset = rst;
}
"#;
        let output = compile(source);
        let top = output.package.get_top_block().unwrap();
        let PackageMember::Block { func, metadata } = top else {
            unreachable!();
        };
        assert!(metadata.reset.as_ref().unwrap().active_low);
        assert_eq!(
            metadata
                .ports
                .iter()
                .map(|port| port.name.as_str())
                .collect::<Vec<_>>(),
            vec!["data", "in_reset", "clk", "rst"]
        );
        assert!(func.nodes.iter().any(|node| {
            node.name.as_deref() == Some("__xlsynth_reset_active")
                && matches!(node.payload, NodePayload::Unop(ir::Unop::Not, _))
        }));
        assert!(func.nodes.iter().any(|node| {
            node.name
                .as_deref()
                .is_some_and(|name| name.starts_with("__xlsynth_assert_0_"))
                && matches!(node.payload, NodePayload::Nary(ir::NaryOp::Or, _))
        }));
        assert!(func.nodes.iter().any(|node| {
            node.name
                .as_deref()
                .is_some_and(|name| name.starts_with("__xlsynth_cover_0_"))
                && matches!(node.payload, NodePayload::Nary(ir::NaryOp::And, _))
        }));
    }

    #[test]
    fn property_messages_get_unique_ir_safe_labels() {
        let source = r#"
pub block labels(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  assert!(x, "same label!");
  assert!(x, "same label!");
  cover!(x, "spaces and punctuation: !?");
  assign y = x;
}
"#;
        let output = compile(source);
        assert!(
            output
                .ir_text
                .contains("label=\"__xlsynth_assert_0_same_label\"")
        );
        assert!(
            output
                .ir_text
                .contains("label=\"__xlsynth_assert_1_same_label\"")
        );
        assert!(
            output
                .ir_text
                .contains("label=\"__xlsynth_cover_0_spaces_and_punctuation\"")
        );
        assert!(output.ir_text.contains("message=\"same label!\""));
    }

    #[test]
    fn rejects_non_utf8_property_label_byte_escape() {
        let source = r#"
pub block property_byte(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assert!(true, "byte\x80");
  assign y = true;
}
"#;
        let error = compile_block_module(
            source,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(
            error
                .message
                .contains("do not support non-UTF-8 byte escapes")
        );
    }

    #[test]
    fn rejects_invalid_source_module_names_before_ir_emission() {
        let source = r#"
pub block valid(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) { assign y = true; }
"#;
        let error = compile_block_module(
            source,
            Path::new("not-valid.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("valid module identifier"));
    }

    #[test]
    fn lowers_reachable_verilog_ffi_functions_to_extern_instantiations() {
        let reachable = r#"
#[extern_verilog("assign {return} = ~{x};")]
fn ffi(x: u8) -> u8 { x }

pub block wraps_ffi(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  let ffi_input = x;
  assign y = ffi(ffi_input);
}
"#;
        let output = compile(reachable);
        assert!(output.ir_text.contains("#[ffi_proto("));
        assert!(output.ir_text.contains("kind=extern"));
        assert!(
            output
                .ir_text
                .contains("foreign_function=__block_test__ffi")
        );
        assert!(!output.ir_text.contains("invoke("));
        let PackageMember::Block { func, .. } = output.package.get_top_block().unwrap() else {
            panic!("expected top block");
        };
        assert!(matches!(
            func.get_node(func.ret_node_ref.unwrap()).payload,
            NodePayload::InstantiationOutput { .. }
        ));
        let preserved = compile_block_module(
            reachable,
            Path::new("block_test.x"),
            &BlockCompileOptions {
                combinational_optimization: CombinationalOptimization::PreserveNames,
                ..BlockCompileOptions::default()
            },
        )
        .unwrap();
        let PackageMember::Block { func, .. } = preserved.package.get_top_block().unwrap() else {
            panic!("expected top block");
        };
        assert!(matches!(
            func.get_node(func.ret_node_ref.unwrap()).payload,
            NodePayload::InstantiationOutput { .. }
        ));

        let nested = r#"
#[extern_verilog("assign {return} = ~{x};")]
fn ffi(x: u8) -> u8 { x }
fn wrapper(x: u8) -> u8 { ffi(x) }

pub block nested_ffi(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  assign y = wrapper(x);
}
"#;
        let error = compile_block_module(
            nested,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("called through an ordinary helper"));

        let unreachable = r#"
// Mentioning extern_verilog in a comment must not trigger the policy.
#[extern_verilog("assign {return} = {x};")]
fn ffi(x: u8) -> u8 { x }

pub block no_ffi(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  assign y = x;
}
"#;
        let output = compile(unreachable);
        assert!(output.ir_text.contains("top block no_ffi"));
    }

    #[test]
    fn generated_instance_output_names_are_hygienic() {
        let source = r#"
block child(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  assign y = x;
}

pub block parent(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  let __xlsynth_c_y = x + u8:1;
  inst c: child { x: x, }
  assign y = c.y;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("__xlsynth_c_y_2"));
        assert!(
            output
                .ir_text
                .contains("output_port(__xlsynth_c_y_2, name=y")
        );
    }

    #[test]
    fn local_consts_drive_types_and_child_parametrics_in_declaration_order() {
        let source = r#"
block child<N: u32>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: uN[N],
  output y: uN[N],
) {
  assign y = x;
}

pub block parent(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  const W: u32 = u32:8;
  let typed: uN[W] = x;
  inst c: child<W> { x: typed, }
  reg q: uN[W] {
    init_value: uN[W]:0,
    next: c.y,
  }
  assign y = q;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("block __xlsynth_spec_child_0"));
        assert!(output.ir_text.contains("reg q(bits[8])"));
    }

    #[test]
    fn rejects_xls_ir_keyword_block_names_without_mangling() {
        let source = r#"
pub block top(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assign y = true;
}
"#;
        let error = compile_block_module(
            source,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("reserved by the XLS IR backend"));
    }

    #[test]
    fn rejects_system_verilog_keyword_block_and_instance_names_without_mangling() {
        let keyword_block = r#"
pub block module(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assign y = true;
}
"#;
        let error = compile_block_module(
            keyword_block,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(
            error
                .message
                .contains("reserved by the SystemVerilog backend")
        );

        let keyword_instance = r#"
block child(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  assign y = x;
}

pub block parent(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  inst wire: child { x: x, }
  assign y = wire.y;
}
"#;
        let error = compile_block_module(
            keyword_instance,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(
            error
                .message
                .contains("instance name 'wire' is reserved by the SystemVerilog backend")
        );
    }

    #[test]
    fn rejects_xls_ir_keyword_instance_names_in_every_optimization_mode() {
        let source = r#"
block child(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  assign y = x;
}

pub block parent(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  inst top: child { x: x, }
  assign y = top.y;
}
"#;
        for combinational_optimization in [
            CombinationalOptimization::Free,
            CombinationalOptimization::PreserveNames,
            CombinationalOptimization::PreserveNamesAndFunctions,
        ] {
            let options = BlockCompileOptions {
                combinational_optimization,
                ..BlockCompileOptions::default()
            };
            let error = compile_block_module(source, Path::new("block_test.x"), &options)
                .expect_err("XLS IR keyword must be rejected as an instance name");
            assert_eq!(error.offset, source.find("inst top"));
            assert!(
                error
                    .message
                    .contains("instance name 'top' is reserved by the XLS IR backend")
            );
        }
    }

    #[test]
    fn rewrites_only_instance_root_output_accesses() {
        let names = BTreeMap::from([(
            ("child_i".to_string(), "y".to_string()),
            "__xlsynth_child_i_y".to_string(),
        )]);
        assert_eq!(
            rewrite_instance_outputs(
                "s.child_i.y + namespace::child_i.y + child_i.y.field",
                &names,
            ),
            "s.child_i.y + namespace::child_i.y + __xlsynth_child_i_y.field"
        );
    }

    #[test]
    fn instance_output_rewrite_respects_nested_for_and_match_binders() {
        let names = BTreeMap::from([(
            ("child_i".to_string(), "y".to_string()),
            "__xlsynth_child_i_y".to_string(),
        )]);
        let for_expression = "for (child_i, acc) in values { acc + child_i.y }(u8:0) + child_i.y";
        assert_eq!(
            rewrite_instance_outputs(for_expression, &names),
            "for (child_i, acc) in values { acc + child_i.y }(u8:0) + __xlsynth_child_i_y"
        );
        let match_expression = "match value { child_i => child_i.y, } + child_i.y";
        assert_eq!(
            rewrite_instance_outputs(match_expression, &names),
            "match value { child_i => child_i.y, } + __xlsynth_child_i_y"
        );
    }

    #[test]
    fn identifier_substitution_respects_nested_for_and_match_binders() {
        let values = BTreeMap::from([("N".to_string(), "u32:8".to_string())]);
        assert_eq!(
            substitute_identifiers("for (N, acc) in u32:0..N { acc + N }(N as u32)", &values),
            "for (N, acc) in u32:0..(u32:8) { acc + N }((u32:8) as u32)"
        );
        assert_eq!(
            substitute_identifiers("match x { N => N, } + N", &values),
            "match x { N => N, } + (u32:8)"
        );
    }

    #[test]
    fn nested_member_matching_an_instance_name_compiles_without_rewrite() {
        let source = r#"
struct Inner { y: u8 }
struct Outer { child_i: Inner }

block child(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  assign y = x;
}

pub block parent(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input s: Outer,
  output y: u8,
) {
  inst child_i: child { x: s.child_i.y, }
  assign y = s.child_i.y;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("instantiation child_i"));
    }

    #[test]
    fn accepts_comments_and_whitespace_around_member_and_qualified_operators() {
        let source = r#"
enum Choice : u2 { VALUE = 1 }

block child(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: Choice,
  output child_y: Choice,
) {
  assign child_y = x;
}

pub block parent<VALUE: u32 = {u32:7}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: Choice,
) {
  inst child_i: child {
    x: Choice // qualified comment
       :: // tail comment
       VALUE,
  }
  assign y = child_i // member comment
             . // output comment
             child_y;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("instantiation_output("));
        assert!(output.ir_text.contains("value=1"));
    }

    #[test]
    fn uses_dslx_type_identity_for_repeated_register_types() {
        let equivalent = r#"
type Byte = u8;

pub block alias_reg(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  declreg q: Byte;
  reg q: u8 { next: x, }
  assign y = q;
}
"#;
        let output = compile(equivalent);
        assert!(output.ir_text.contains("reg q(bits[8])"));

        let different = equivalent.replace("reg q: u8", "reg q: u16");
        let error = compile_block_module(
            &different,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert_eq!(error.offset, different.find("u16 {"));
        assert!(
            error.message.contains(
                "register 'q' repeats type 'u16', which does not match declreg type 'Byte'"
            )
        );
    }

    #[test]
    fn declaration_order_diagnostics_use_absolute_expression_offsets() {
        let cases = [
            (
                r#"
pub block bad(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  assign y = later;
  let later = true;
}
"#,
                "later;",
            ),
            (
                r#"
pub block bad(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  let early = later;
  let later = true;
  assign y = early;
}
"#,
                "later;",
            ),
            (
                r#"
pub block bad(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  reg q: bool { next: later, }
  let later = true;
  assign y = q;
}
"#,
                "later,",
            ),
            (
                r#"
block child(input clk: clock, input rst: reset<active_high, sync>, input x: bool, output y: bool) {
  assign y = x;
}
pub block bad(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  inst c: child { x: later, }
  let later = true;
  assign y = c.y;
}
"#,
                "later,",
            ),
            (
                r#"
pub block bad(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  assert!(later, "late");
  let later = true;
  assign y = later;
}
"#,
                "later,",
            ),
            (
                r#"
pub block bad<N: u32 = {u32:8}>(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  let early = (N == u32:8) && later;
  let later = true;
  assign y = early;
}
"#,
                "later;",
            ),
        ];
        for (source, marker) in cases {
            let error = compile_block_module(
                source,
                Path::new("block_test.x"),
                &BlockCompileOptions::default(),
            )
            .expect_err("use-before-declaration must fail");
            assert_eq!(error.offset, source.find(marker), "source:\n{source}");
            assert!(error.message.contains("used before its declaration"));
        }
    }

    #[test]
    fn comparisons_inside_parametric_groups_do_not_close_generic_lists() {
        let source = r#"
fn less_than<N: u32>(lhs: uN[N], rhs: uN[N]) -> bool { lhs < rhs }

block child<SELECT: bool, N: u32>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: uN[N],
  output y: uN[N],
) {
  assign y = if SELECT { x } else { uN[N]:0 };
}

pub block parent<
  ENABLED: bool = {
    (u32:3 > u32:2) &&
    (u32:1 < u32:2) &&
    ((u32:8 >> u32:1) == u32:4)
  },
>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  inst c: child<
    (less_than<u32:8>(u8:1, u8:2) && ((u32:8 >> u32:1) == u32:4)),
    u32:8,
  > { x: x, }
  assign y = c.y;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("block __xlsynth_spec_child_0"));
    }

    #[test]
    fn bare_comparison_and_shift_parametric_arguments_parse_without_formatting_rules() {
        let source = r#"
block child<SELECT: bool, SHIFT: u32>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  assign y = SELECT && (SHIFT == u32:4);
}

pub block parent<A: u32 = {u32:1}, B: u32 = {u32:2}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  inst c: child<A < B, u32:8 >> u32:1> {}
  assign y = c.y;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("block __xlsynth_spec_child_0"));
    }

    #[test]
    fn parametric_substitution_preserves_first_argument_precedence() {
        let source = r#"
fn identity<N: u32>(x: uN[N]) -> uN[N] { x }

pub block parent<N: u32 = {u32:8 - u32:1}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u14,
  output y: u14,
) {
  assign y = identity<N * u32:2>(x);
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("x: bits[14] = input_port"));
    }

    #[test]
    fn declaration_order_understands_nested_for_bindings() {
        let source = r#"
pub block nested_binder(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  let folded = for (i, acc): (u32, u8) in u32:0..u32:8 {
    acc + (i as u8) + x
  }(u8:0);
  let matched = match x { u8:0 => u8:1, other => other };
  let i = u32:0;
  let other = u8:0;
  assign y = folded + matched + (i as u8) + other;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("y: () = output_port"));
    }

    #[test]
    fn typed_local_const_type_can_contain_equality() {
        let source = r#"
pub block typed_const(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: u1,
) {
  const VALUE: uN[(u32:1 == u32:1) as u32] = u1:1;
  assign y = VALUE;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("literal(value=1"));
    }

    #[test]
    fn clock_and_reset_types_treat_comments_as_whitespace() {
        let source = r#"
pub block commented_ports(
  input clk: /* clock marker */ clock,
  input rst: reset<active_high, /* synchronous */ sync>,
  output y: bool,
) {
  assign y = rst;
}
"#;
        let output = compile(source);
        let PackageMember::Block { metadata, .. } = output.package.get_top_block().unwrap() else {
            unreachable!();
        };
        assert_eq!(metadata.clock_port_name.as_deref(), Some("clk"));
        assert_eq!(
            metadata
                .reset
                .as_ref()
                .map(|reset| reset.port_name.as_str()),
            Some("rst")
        );
    }

    #[test]
    fn structural_condition_reports_exact_use_before_declaration_offset() {
        let source = r#"
pub block bad(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  if later { assign y = true; } else { assign y = false; }
  const later = true;
}
"#;
        let error =
            compile_block_module(source, Path::new("bad.x"), &BlockCompileOptions::default())
                .unwrap_err();
        assert_eq!(error.offset, source.find("later"));
        assert!(error.message.contains("used before its declaration"));
    }

    #[test]
    fn structural_constexpr_rejects_verilog_ffi_fallback_bodies() {
        let source = r#"
#[extern_verilog("assign {return} = {x};")]
fn ffi(x: bool) -> bool { false }

pub block bad(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output y: bool,
) {
  if ffi(true) { assign y = true; } else { assign y = false; }
}
"#;
        let error = compile_block_module(
            source,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("cannot be used for elaboration"));
    }

    #[test]
    fn compiler_owned_member_and_node_names_are_hygienic() {
        let source = r#"
fn __xlsynth_block_collision(x: bool) -> bool { x }

block child<N: u32>(
  input clk: clock,
  input rst_n: reset<active_low, sync>,
  input x: uN[N],
  output y: uN[N],
) { assign y = x; }

block __xlsynth_spec_child_0(
  input clk: clock,
  input rst_n: reset<active_low, sync>,
  input x: u1,
  output y: u1,
) { assign y = !x; }

pub block collision(
  input clk: clock,
  input rst_n: reset<active_low, sync>,
  input x: bool,
  output y: bool,
) {
  let __xlsynth_reset_active = x;
  inst c: child<u32:1> { x: __xlsynth_reset_active, }
  assert!(__xlsynth_reset_active, "active");
  assign y = c.y;
}
"#;
        let output = compile_block_module(
            source,
            Path::new("block_test.x"),
            &BlockCompileOptions {
                combinational_optimization: CombinationalOptimization::PreserveNames,
                ..BlockCompileOptions::default()
            },
        )
        .unwrap();
        xlsynth::IrPackage::parse_ir(&output.ir_text, Some("block_test.x"))
            .expect("hygienic generated names should parse in official XLS");
        assert!(output.ir_text.contains("block __xlsynth_spec_child_0_2"));
        assert!(output.ir_text.contains("__xlsynth_reset_active_2"));
    }

    #[test]
    fn compiles_block_instantiation() {
        let source = r#"
block passthrough(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  assign y = x;
}

pub block parent(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  output y: u8,
) {
  inst child: passthrough { x: x, }
  assign y = child.y;
}
"#;
        let output = compile(source);
        assert!(
            output
                .ir_text
                .contains("instantiation child(block=passthrough, kind=block)")
        );
        assert!(output.ir_text.contains("instantiation_input("));
        assert!(output.ir_text.contains("instantiation_output("));
    }

    #[test]
    fn materializes_and_reuses_parametric_child_blocks() {
        let source = r#"
block passthrough<WIDTH: u32 = {u32:8}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: uN[WIDTH],
  output y: uN[WIDTH],
) {
  assign y = x;
}

pub block parent<WIDTH: u32 = {u32:4}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: uN[WIDTH],
  output y0: uN[WIDTH],
  output y1: uN[WIDTH],
) {
  inst first: passthrough<WIDTH> { x: x, }
  inst second: passthrough<WIDTH> { x: x, }
  assign y0 = first.y;
  assign y1 = second.y;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains(
            "block __xlsynth_spec_passthrough_0(clk: clock, rst: bits[1], x: bits[4], y: bits[4])"
        ));
        assert_eq!(
            output
                .ir_text
                .matches("block __xlsynth_spec_passthrough_0(")
                .count(),
            1
        );
        assert!(
            output
                .ir_text
                .contains("instantiation first(block=__xlsynth_spec_passthrough_0, kind=block)")
        );
        assert!(
            output
                .ir_text
                .contains("instantiation second(block=__xlsynth_spec_passthrough_0, kind=block)")
        );
    }

    #[test]
    fn rejects_output_read_and_shadowing() {
        let output_read = r#"
pub block bad(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  assign y = !y;
}
"#;
        let error = compile_block_module(
            output_read,
            Path::new("bad.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("write-only"));

        let shadow = r#"
pub block bad(input clk: clock, input rst: reset<active_high, sync>, input x: bool, output y: bool) {
  let x = !x;
  assign y = x;
}
"#;
        let error =
            compile_block_module(shadow, Path::new("bad.x"), &BlockCompileOptions::default())
                .unwrap_err();
        assert!(error.message.contains("shadowing is not allowed"));
    }

    #[test]
    fn enforces_declaration_order_and_allows_declreg() {
        let use_before_declaration = r#"
pub block bad(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  assign y = later;
  let later = true;
}
"#;
        let error = compile_block_module(
            use_before_declaration,
            Path::new("bad.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("used before its declaration"));

        let forward = r#"
pub block good(input clk: clock, input rst: reset<active_high, sync>, output y: bool) {
  declreg q: bool;
  let inverted = !q;
  reg q { next: inverted, }
  assign y = q;
}
"#;
        let output = compile(forward);
        assert!(output.ir_text.contains("reg q(bits[1])"));
    }

    #[test]
    fn lowers_dynamic_and_none_init_values_with_assert_and_cover() {
        let source = r#"
pub block properties(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input seed: u8,
  input en: bool,
  output dynamic_q: u8,
  output none_q: u8,
) {
  reg dynamic_q_r: u8 {
    init_value: seed,
    en: en,
    next: dynamic_q_r + u8:1,
  }
  reg none_q_r: u8 {
    init_value: none,
    next: none_q_r + u8:1,
  }
  assert!(dynamic_q_r >= seed, "dynamic_monotonic");
  cover!(dynamic_q_r == u8:7, "dynamic_seven");
  assign dynamic_q = dynamic_q_r;
  assign none_q = none_q_r;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("dynamic_q_r_reset_mux"));
        assert!(output.ir_text.contains("dynamic_q_r_load_enable"));
        assert!(!output.ir_text.contains("none_q_r_reset_mux"));
        assert!(output.ir_text.contains("assert("));
        assert!(output.ir_text.contains("cover("));
        assert!(output.ir_text.contains(
            "message=\"dynamic_monotonic\", label=\"__xlsynth_assert_0_dynamic_monotonic\""
        ));
        assert!(
            output
                .ir_text
                .contains("label=\"__xlsynth_cover_0_dynamic_seven\"")
        );
    }

    #[test]
    fn proc_instance_requires_toolchain() {
        let source = r#"
proc Worker {
  output: chan<u8> out;
  config(output: chan<u8> out) { (output,) }
  init { u8:0 }
  next(state: u8) {
    let tok = send(join(), output, state);
    state + u8:1
  }
}

pub block wrapper(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input ready: bool,
  output valid: bool,
) {
  inst worker: Worker { block_test__output_rdy: ready, }
  assign valid = worker.block_test__output_vld;
}
"#;
        let error = compile_block_module(
            source,
            Path::new("block_test.x"),
            &BlockCompileOptions::default(),
        )
        .unwrap_err();
        assert!(error.message.contains("requires an XLS toolchain path"));
    }

    #[test]
    fn unreachable_proc_block_does_not_require_toolchain() {
        let source = r#"
proc Worker {
  output: chan<u8> out;
  config(output: chan<u8> out) { (output,) }
  init { u8:0 }
  next(state: u8) {
    let tok = send(join(), output, state);
    state + u8:1
  }
}

block unused_wrapper(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input ready: bool,
  output valid: bool,
) {
  inst worker: Worker { block_test__output_rdy: ready, }
  assign valid = worker.block_test__output_vld;
}

pub block selected(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  assign y = x;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("top block selected"));
        assert!(!output.ir_text.contains("unused_wrapper"));
        assert!(!output.ir_text.contains("Worker"));
    }

    #[test]
    fn unreachable_invalid_block_is_not_lowered() {
        let source = r#"
block unused(
  input clk: clock,
  input rst: reset<active_high, sync>,
  output missing_driver: bool,
) {
}

pub block selected(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  assign y = x;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("top block selected"));
        assert!(!output.ir_text.contains("missing_driver"));
    }

    #[test]
    fn reachable_required_parametric_specialization_includes_ordinary_child() {
        let source = r#"
block leaf(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  assign y = !x;
}

block wrapper<N: u32>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  inst child: leaf { x: x, }
  assign y = child.y;
}

pub block selected(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  inst wrapped: wrapper<u32:1> { x: x, }
  assign y = wrapped.y;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("block leaf"));
        assert!(output.ir_text.contains("top block selected"));
    }

    #[test]
    fn reachability_uses_concrete_parametric_structural_branch() {
        let source = r#"
block left(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  assign y = x;
}

block right(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  assign y = !x;
}

block chooser<USE_RIGHT: bool = {false}>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  if USE_RIGHT {
    inst chosen: right { x: x, }
    assign y = chosen.y;
  } else {
    inst chosen: left { x: x, }
    assign y = chosen.y;
  }
}

pub block selected(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: bool,
  output y: bool,
) {
  inst wrapped: chooser<true> { x: x, }
  assign y = wrapped.y;
}
"#;
        let output = compile(source);
        assert!(output.ir_text.contains("block right"));
        assert!(!output.ir_text.contains("block left"));
        assert!(output.ir_text.contains("top block selected"));
    }

    #[test]
    fn imports_stateful_proc_when_toolchain_is_available() {
        let Ok(tool_path) = std::env::var("XLSYNTH_TEST_TOOL_PATH") else {
            return;
        };
        let source = r#"
proc Worker {
  output: chan<u8> out;
  config(output: chan<u8> out) { (output,) }
  init { u8:0 }
  next(state: u8) {
    let tok = send(join(), output, state);
    state + u8:1
  }
}

pub block wrapper(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input ready: bool,
  output valid: bool,
  output data: u8,
) {
  inst worker: Worker { _output_rdy: ready, }
  assign valid = worker._output_vld;
  assign data = worker._output;
}
"#;
        let options = BlockCompileOptions {
            tool_path: Some(tool_path.into()),
            ..BlockCompileOptions::default()
        };
        let output =
            compile_block_module(source, Path::new("proc_block_test.x"), &options).unwrap();
        assert!(output.ir_text.contains(
            "block __xlsynth_proc_proc_block_test_Worker_active_high____proc_block_test__Worker_0_next"
        ));
        assert!(
            output.ir_text.contains(
                "instantiation worker(block=__xlsynth_proc_proc_block_test_Worker_active_high____proc_block_test__Worker_0_next, kind=block)"
            )
        );
    }

    #[test]
    fn specialized_parametric_block_can_wrap_a_proc() {
        let Ok(tool_path) = std::env::var("XLSYNTH_TEST_TOOL_PATH") else {
            return;
        };
        let source = r#"
proc Worker {
  output: chan<u8> out;
  config(output: chan<u8> out) { (output,) }
  init { u8:0 }
  next(state: u8) {
    let tok = send(join(), output, state);
    state + u8:1
  }
}

block proc_wrapper<N: u32>(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input ready: bool,
  output data: u8,
) {
  const_assert!(N > u32:0);
  inst worker: Worker { _output_rdy: ready, }
  assign data = worker._output;
}

pub block parent(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input ready: bool,
  output data: u8,
) {
  inst wrapped: proc_wrapper<u32:8> { ready: ready, }
  assign data = wrapped.data;
}
"#;
        let output = compile_block_module(
            source,
            Path::new("proc_parametric_wrapper_test.x"),
            &BlockCompileOptions {
                tool_path: Some(tool_path.into()),
                ..BlockCompileOptions::default()
            },
        )
        .unwrap();
        assert!(
            output
                .ir_text
                .contains("block __xlsynth_spec_proc_wrapper_0")
        );
        assert!(
            output
                .ir_text
                .contains("instantiation worker(block=__xlsynth_proc_")
        );
    }

    #[test]
    fn specializes_proc_codegen_by_reset_polarity_and_avoids_port_collisions() {
        let Ok(tool_path) = std::env::var("XLSYNTH_TEST_TOOL_PATH") else {
            return;
        };
        let source = r#"
proc Worker {
  output: chan<u8> out;
  config(output: chan<u8> out) { (output,) }
  init { u8:0 }
  next(state: u8) {
    let tok = send(join(), output, state);
    state + u8:1
  }
}

block high_wrapper(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input ready: bool,
  output data: u8,
) {
  inst worker: Worker { _output_rdy: ready, }
  assign data = worker._output;
}

block low_wrapper(
  input clk: clock,
  input rst_n: reset<active_low, sync>,
  input ready: bool,
  output data: u8,
) {
  inst worker: Worker { _output_rdy: ready, }
  assign data = worker._output;
}

pub block combined(
  input clk: clock,
  input _output: reset<active_high, sync>,
  input ready: bool,
  output data: u8,
) {
  inst wrapper: high_wrapper { ready: ready, }
  assign data = wrapper.data;
}
"#;
        let options = BlockCompileOptions {
            tool_path: Some(tool_path.into()),
            combinational_optimization: CombinationalOptimization::PreserveNames,
            ..BlockCompileOptions::default()
        };
        let output = compile_block_module(source, Path::new("proc_resets.x"), &options).unwrap();
        assert!(output.ir_text.contains("Worker_active_high"));
        assert!(
            output.ir_text.contains("Worker_active_low"),
            "{}",
            output.ir_text
        );

        let collision_source = r#"
proc Worker {
  output: chan<u8> out;
  config(output: chan<u8> out) { (output,) }
  init { u8:0 }
  next(state: u8) {
    let tok = send(join(), output, state);
    state + u8:1
  }
}

pub block wrapper(
  input clk: clock,
  input _output: reset<active_high, sync>,
  input ready: bool,
  output data: u8,
) {
  inst worker: Worker { _output_rdy: ready, }
  assign data = worker._output;
}
"#;
        compile_block_module(
            collision_source,
            Path::new("proc_reset_collision.x"),
            &options,
        )
        .unwrap();
    }

    #[test]
    fn proc_closure_renaming_updates_extern_targets() {
        let ir = r#"package proc_ffi

#[ffi_proto("""code_template: "assign {return} = {x};"
""")]
fn ffi(x: bits[8] id=1) -> bits[8] {
  ret x: bits[8] = param(name=x, id=1)
}

top block wrapper(x: bits[8], y: bits[8]) {
  instantiation call(foreign_function=ffi, kind=extern)
  x: bits[8] = input_port(name=x, id=2)
  result: bits[8] = instantiation_output(instantiation=call, port_name=return, id=3)
  instantiation_input.4: () = instantiation_input(x, instantiation=call, port_name=x, id=4)
  y: () = output_port(result, name=y, id=5)
}
"#;
        let package = Parser::new(ir).parse_package().unwrap();
        let mut occupied = BTreeSet::new();
        let (members, renamed_top) =
            rename_proc_members(package.members, "wrapper", "proc_scope", &mut occupied);
        assert_eq!(renamed_top, "proc_scope__wrapper");
        let target = members.iter().find_map(|member| match member {
            PackageMember::Block { metadata, .. } => {
                metadata.instantiations.first().map(|inst| &inst.block)
            }
            _ => None,
        });
        assert_eq!(target.map(String::as_str), Some("proc_scope__ffi"));
    }

    #[test]
    fn imports_and_renames_an_entire_proc_network_closure() {
        let Ok(tool_path) = std::env::var("XLSYNTH_TEST_TOOL_PATH") else {
            return;
        };
        let source = r#"
proc Child {
  output: chan<u8> out;
  config(output: chan<u8> out) { (output,) }
  init { u8:0 }
  next(state: u8) {
    let tok = send(join(), output, state);
    state + u8:1
  }
}

proc Network {
  config(sink: chan<u8> out) {
    spawn Child(sink);
    ()
  }
  init { () }
  next(state: ()) { () }
}

pub block wrapper(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input ready: bool,
  output valid: bool,
  output data: u8,
) {
  inst network: Network { _sink_rdy: ready, }
  assign valid = network._sink_vld;
  assign data = network._sink;
}
"#;
        let options = BlockCompileOptions {
            tool_path: Some(tool_path.clone().into()),
            ..BlockCompileOptions::default()
        };
        let output =
            compile_block_module(source, Path::new("proc_network_test.x"), &options).unwrap();
        let child = "__xlsynth_proc_proc_network_test_Network_active_high____proc_network_test__Child_0_next";
        let network = "__xlsynth_proc_proc_network_test_Network_active_high____proc_network_test__Network_0_next";
        assert!(output.ir_text.contains(&format!("block {child}")));
        assert!(output.ir_text.contains(&format!("block {network}")));
        assert!(output.ir_text.contains(&format!(
            "instantiation network(block={network}, kind=block)"
        )));
        assert!(
            output
                .ir_text
                .contains(&format!("block={child}, kind=block"))
        );

        let temporary = tempfile::NamedTempFile::new().unwrap();
        let generated_sv = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temporary.path(), &output.ir_text).unwrap();
        let mut command = Command::new(Path::new(&tool_path).join("block_to_verilog_main"));
        command
            .arg(temporary.path())
            .arg("--top")
            .arg("wrapper")
            .arg("--generator=combinational");
        let generated = run_external_tool_to_file(
            &mut command,
            generated_sv.path(),
            options.external_tool_timeout,
            options.max_tool_output_bytes,
            options.max_tool_artifact_bytes,
        )
        .unwrap();
        assert!(
            generated.status.success(),
            "{}",
            String::from_utf8_lossy(&generated.stderr)
        );
        assert!(
            std::fs::read_to_string(generated_sv.path())
                .unwrap()
                .contains(&format!("module {child}"))
        );
    }

    #[test]
    #[cfg(unix)]
    fn external_tool_runner_bounds_time_and_captured_output() {
        let mut slow = Command::new("sh");
        slow.arg("-c").arg("sleep 10 & wait");
        let start = Instant::now();
        let result = run_tool_with_limits(&mut slow, Duration::from_millis(20), 1024).unwrap();
        assert!(result.timed_out);
        assert!(start.elapsed() < Duration::from_secs(2));

        let mut detached = Command::new("sh");
        detached.arg("-c").arg("sleep 10 &");
        let start = Instant::now();
        let result = run_tool_with_limits(&mut detached, Duration::from_millis(100), 1024).unwrap();
        assert!(result.status.success());
        assert!(!result.timed_out);
        assert!(start.elapsed() < Duration::from_secs(2));

        let mut noisy = Command::new("sh");
        noisy.arg("-c").arg("printf 123456789 >&2");
        let result = run_tool_with_limits(&mut noisy, Duration::from_secs(1), 4).unwrap();
        assert!(result.status.success());
        assert!(result.output_truncated);
        assert!(String::from_utf8_lossy(&result.stderr).contains("output truncated"));

        let output = tempfile::NamedTempFile::new().unwrap();
        let mut artifact = Command::new("sh");
        artifact.arg("-c").arg("printf 123456789");
        let result =
            run_external_tool_to_file(&mut artifact, output.path(), Duration::from_secs(1), 4, 100)
                .unwrap();
        assert!(result.status.success());
        assert_eq!(std::fs::read(output.path()).unwrap(), b"123456789");

        let oversized = tempfile::NamedTempFile::new().unwrap();
        let mut artifact = Command::new("sh");
        artifact.arg("-c").arg("printf 123456789");
        let result = run_external_tool_to_file(
            &mut artifact,
            oversized.path(),
            Duration::from_secs(1),
            1024,
            4,
        )
        .unwrap();
        assert!(result.artifact_too_large);

        let watched = tempfile::NamedTempFile::new().unwrap();
        let mut artifact = Command::new("sh");
        artifact
            .arg("-c")
            .arg("printf 123456789 > \"$1\"")
            .arg("sh")
            .arg(watched.path());
        let result = run_tool_with_artifact_limit(
            &mut artifact,
            Duration::from_secs(1),
            1024,
            watched.path(),
            4,
        )
        .unwrap();
        assert!(result.artifact_too_large);
    }
}
