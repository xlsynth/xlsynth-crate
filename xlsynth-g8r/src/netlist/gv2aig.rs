// SPDX-License-Identifier: Apache-2.0

//! Convert a gate-level netlist + Liberty proto into a `GateFn` (AIG form).

use crate::aig::GateFn;
use crate::netlist::gv2aig_boundaries::{
    Gv2AigBoundaryExtraction, Gv2AigBoundaryRequest, extract_gatefn_with_optional_boundary_request,
};
use crate::netlist::hierarchy::elaborate_hierarchy;
use crate::netlist::io::{load_liberty_from_path, read_gv_from_path, select_module};
use anyhow::{Result, anyhow};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Gv2AigOptions {
    pub module_name: Option<String>,

    /// If true, collapse sequential state variables by substituting next_state.
    pub collapse_sequential: bool,

    /// If true, collapse recognized load-enable feedback while assuming prior Q
    /// is zero. Applies only when `collapse_sequential` is true; defaults to
    /// false.
    pub collapse_load_enable_feedback: bool,
}

impl Default for Gv2AigOptions {
    fn default() -> Self {
        Self {
            module_name: None,
            collapse_sequential: true,
            collapse_load_enable_feedback: false,
        }
    }
}

/// Converts strict GV and its Liberty library to a combinational AIG.
pub fn convert_gv2aig_paths(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    opts: &Gv2AigOptions,
) -> Result<GateFn> {
    convert_gv2aig_paths_with_optional_boundary_request(
        netlist_path,
        liberty_proto_path,
        opts,
        None,
    )
    .map(|extraction| extraction.gate_fn)
}

/// Converts compact source/sink selectors, or all top-module ports when no
/// request is supplied, using one hierarchy-aware extraction path. Request
/// validation happens before opening the netlist or Liberty files.
pub fn convert_gv2aig_paths_with_optional_boundary_request(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    opts: &Gv2AigOptions,
    request: Option<&Gv2AigBoundaryRequest>,
) -> Result<Gv2AigBoundaryExtraction> {
    if let Some(request) = request {
        request.validate().map_err(anyhow::Error::msg)?;
    }
    let parsed = read_gv_from_path(netlist_path)?;
    let module = select_module(&parsed, opts.module_name.as_deref())?;
    let elaborated = elaborate_hierarchy(&parsed, module)?;
    let liberty = load_liberty_from_path(liberty_proto_path)?;
    extract_gatefn_with_optional_boundary_request(
        &elaborated,
        &liberty,
        opts.collapse_sequential,
        opts.collapse_load_enable_feedback,
        request,
    )
    .map_err(|error| anyhow!(error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::gv2aig_boundaries::{
        Gv2AigBoundaryKind, Gv2AigBoundaryMatcher, Gv2AigBoundarySelection,
    };

    #[test]
    fn validates_typed_request_before_opening_input_paths() {
        let request = Gv2AigBoundaryRequest {
            sources: Vec::new(),
            sinks: vec![Gv2AigBoundarySelection {
                name: None,
                kind: Gv2AigBoundaryKind::OutputPort,
                matcher: Gv2AigBoundaryMatcher::Regex("[".to_string()),
            }],
        };
        let directory = tempfile::tempdir().unwrap();
        let error = convert_gv2aig_paths_with_optional_boundary_request(
            &directory.path().join("missing.gv"),
            &directory.path().join("missing.proto"),
            &Gv2AigOptions::default(),
            Some(&request),
        )
        .err()
        .expect("invalid request should fail before input paths are opened");
        assert!(
            error
                .to_string()
                .contains("invalid regexp in selector 'output_port_regex:['")
        );
    }
}
