// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use xlsynth::vast_helpers_options::CodegenOptions;

/// Controls whether a block is presented in register-delimited stage sections.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// Emit declarations and logic in ordinary dependency order.
    #[default]
    None,
    /// Partition feed-forward logic into explicitly labeled pipeline stages.
    Pipeline,
}

/// Configures faithful SystemVerilog emission of an existing block.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BlockCodegenOptions {
    /// Overrides package top selection when present.
    pub top: Option<String>,
    /// Controls declaration and pipeline-stage layout.
    pub layout: Layout,
    /// Overrides the generated top module name when present.
    pub module_name: Option<String>,
    /// Clamps dynamic array reads to the last valid element when enabled.
    pub array_index_bounds_checking: bool,
    /// Emits every representable operation as its own named assignment.
    pub separate_lines: bool,
    /// Bounds expression nesting before introducing a named intermediate.
    pub max_inline_depth: usize,
    /// Preserves SystemVerilog type annotations on block ports.
    pub emit_sv_types: bool,
    /// Emits invariant assertions when present in the input block.
    pub add_invariant_assertions: bool,
    /// Supplies optional register-lowering templates and related options.
    pub register_codegen_options: Option<CodegenOptions>,
}

impl Default for BlockCodegenOptions {
    fn default() -> Self {
        Self {
            top: None,
            layout: Layout::None,
            module_name: None,
            array_index_bounds_checking: true,
            separate_lines: false,
            max_inline_depth: 5,
            emit_sv_types: true,
            add_invariant_assertions: true,
            register_codegen_options: None,
        }
    }
}

/// Contains the emitted SystemVerilog source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockCodegenOutput {
    /// Complete, deterministically ordered SystemVerilog source.
    pub system_verilog: String,
}
