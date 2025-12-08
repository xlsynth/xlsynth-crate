// SPDX-License-Identifier: Apache-2.0

//! Utility routines for netlist analysis and reporting.

use crate::netlist::parse::NetlistModule;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Returns a Vec of (instance_name, cell_type) for all instances in the
/// modules, using the interner for resolution.
pub fn instance_names_and_types(
    modules: &[NetlistModule],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for module in modules {
        for inst in &module.instances {
            let inst_str = interner
                .resolve(inst.instance_name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_owned());
            let type_str = interner
                .resolve(inst.type_name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_owned());
            pairs.push((inst_str, type_str));
        }
    }
    pairs
}

/// Returns a Vec of (module_name, instance_name, cell_type) for all instances
/// in the modules, using the interner for resolution.
pub fn module_instance_names_and_types(
    modules: &[NetlistModule],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Vec<(String, String, String)> {
    let mut triples: Vec<(String, String, String)> = Vec::new();
    for module in modules {
        let module_str = interner
            .resolve(module.name)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "<unknown>".to_owned());
        for inst in &module.instances {
            let inst_str = interner
                .resolve(inst.instance_name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_owned());
            let type_str = interner
                .resolve(inst.type_name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_owned());
            triples.push((module_str.clone(), inst_str, type_str));
        }
    }
    triples
}
