// SPDX-License-Identifier: Apache-2.0

//! Compact exact and regexp selectors for source/sink cone boundaries.

use std::path::Path;
use std::str::FromStr;

use anyhow::Context;
use regex::Regex;
use serde::Deserialize;

use super::{
    BoundaryExtractor, Gv2AigBoundarySpec, Gv2AigSinkBoundary, Gv2AigSourceBoundary,
    Gv2AigSourceSelector, validate_names,
};
use crate::liberty_model::{PinDirection, SequentialKind};
use crate::netlist::parse::PortDirection;

/// The kind of netlist endpoint selected by a compact boundary selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gv2AigBoundaryKind {
    InputPort,
    FlopOutput,
    OutputPort,
}

/// One exact endpoint, with an empty instance name denoting the selected top.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gv2AigBoundaryEndpoint {
    pub instance_name: String,
    /// A module port name or mapped flop output pin name.
    pub terminal_name: String,
}

/// An exact endpoint or a regexp matched against canonical endpoint text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gv2AigBoundaryMatcher {
    Exact(Gv2AigBoundaryEndpoint),
    Regex(String),
}

/// One parsed compact selector, optionally overriding the extracted AIG name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gv2AigBoundarySelection {
    pub name: Option<String>,
    pub kind: Gv2AigBoundaryKind,
    pub matcher: Gv2AigBoundaryMatcher,
}

/// Ordered compact selectors to resolve before extracting source/sink cones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gv2AigBoundaryRequest {
    pub sources: Vec<Gv2AigBoundarySelection>,
    pub sinks: Vec<Gv2AigBoundarySelection>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundaryFile {
    sources: Vec<String>,
    sinks: Vec<String>,
}

impl Gv2AigBoundaryRequest {
    /// Parses the selector lists accepted by the CLI and boundary JSON file.
    pub fn from_selectors(sources: &[String], sinks: &[String]) -> Result<Self, String> {
        let parse_list = |selectors: &[String], role: &str| {
            selectors
                .iter()
                .enumerate()
                .map(|(index, selector)| {
                    selector.parse().map_err(|error| {
                        format!("invalid {role} selector {index} '{selector}': {error}")
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        };
        let request = Self {
            sources: parse_list(sources, "source")?,
            sinks: parse_list(sinks, "sink")?,
        };
        request.validate_roles()?;
        Ok(request)
    }

    /// Checks that source and sink lists contain selectors of the right kind.
    fn validate_roles(&self) -> Result<(), String> {
        if self.sinks.is_empty() {
            return Err("boundary specification must contain at least one sink".to_string());
        }
        for selection in &self.sources {
            if selection.kind == Gv2AigBoundaryKind::OutputPort {
                return Err(format!(
                    "source selector '{}' must select an input port or flop output",
                    selection.to_compact_string()
                ));
            }
        }
        for selection in &self.sinks {
            if selection.kind != Gv2AigBoundaryKind::OutputPort {
                return Err(format!(
                    "sink selector '{}' must select an output port",
                    selection.to_compact_string()
                ));
            }
        }
        Ok(())
    }
}

impl Gv2AigBoundarySelection {
    /// Renders a selector using the compact CLI and JSON string grammar.
    pub fn to_compact_string(&self) -> String {
        let is_regex = matches!(self.matcher, Gv2AigBoundaryMatcher::Regex(_));
        let prefix = match (self.kind, is_regex) {
            (Gv2AigBoundaryKind::InputPort, false) => "input_port",
            (Gv2AigBoundaryKind::InputPort, true) => "input_port_regex",
            (Gv2AigBoundaryKind::FlopOutput, false) => "flop_output",
            (Gv2AigBoundaryKind::FlopOutput, true) => "flop_output_regex",
            (Gv2AigBoundaryKind::OutputPort, false) => "output_port",
            (Gv2AigBoundaryKind::OutputPort, true) => "output_port_regex",
        };
        let payload = match &self.matcher {
            Gv2AigBoundaryMatcher::Exact(endpoint) => endpoint.render(),
            Gv2AigBoundaryMatcher::Regex(pattern) => pattern.clone(),
        };
        let alias = self
            .name
            .as_ref()
            .map(|name| format!("{}=", escape_component(name, &['=', ':'])))
            .unwrap_or_default();
        format!("{alias}{prefix}:{payload}")
    }
}

impl FromStr for Gv2AigBoundarySelection {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let starts_with_kind = text
            .split_once(':')
            .and_then(|(prefix, _)| parse_kind(prefix))
            .is_some();
        let (name, selector) = if starts_with_kind {
            (None, text)
        } else if let Some(index) = find_unescaped(text, '=') {
            let name = unescape_component(&text[..index], &['=', ':'])?;
            if name.is_empty() || name.contains(['\n', '\r']) {
                return Err("AIG name must be nonempty and contain no newlines".to_string());
            }
            (Some(name), &text[index + 1..])
        } else {
            (None, text)
        };
        let (prefix, payload) = selector
            .split_once(':')
            .ok_or_else(|| "expected '<kind>:<endpoint-or-regexp>'".to_string())?;
        let (kind, is_regex) = parse_kind(prefix)
            .ok_or_else(|| format!("unknown boundary selector kind '{prefix}'"))?;
        let matcher = if is_regex {
            Gv2AigBoundaryMatcher::Regex(payload.to_string())
        } else {
            Gv2AigBoundaryMatcher::Exact(parse_endpoint(payload, kind)?)
        };
        Ok(Self {
            name,
            kind,
            matcher,
        })
    }
}

impl Gv2AigBoundaryEndpoint {
    /// Renders unambiguous endpoint text for exact lookup and regexp matching.
    fn render(&self) -> String {
        let terminal = escape_component(&self.terminal_name, &[':']);
        if self.instance_name.is_empty() {
            terminal
        } else {
            format!(
                "{}:{terminal}",
                escape_component(&self.instance_name, &[':'])
            )
        }
    }
}

/// Loads compact selector strings from a JSON boundary file.
pub fn load_gv2aig_boundary_request(path: &Path) -> anyhow::Result<Gv2AigBoundaryRequest> {
    let contents = std::fs::read(path)
        .with_context(|| format!("failed to read cone boundary file {}", path.display()))?;
    let file: BoundaryFile = serde_json::from_slice(&contents)
        .with_context(|| format!("failed to parse cone boundary file {}", path.display()))?;
    Gv2AigBoundaryRequest::from_selectors(&file.sources, &file.sinks)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("invalid cone boundary file {}", path.display()))
}

/// Recognizes a selector prefix without consuming any regexp punctuation.
fn parse_kind(prefix: &str) -> Option<(Gv2AigBoundaryKind, bool)> {
    match prefix {
        "input_port" => Some((Gv2AigBoundaryKind::InputPort, false)),
        "input_port_regex" => Some((Gv2AigBoundaryKind::InputPort, true)),
        "flop_output" => Some((Gv2AigBoundaryKind::FlopOutput, false)),
        "flop_output_regex" => Some((Gv2AigBoundaryKind::FlopOutput, true)),
        "output_port" => Some((Gv2AigBoundaryKind::OutputPort, false)),
        "output_port_regex" => Some((Gv2AigBoundaryKind::OutputPort, true)),
        _ => None,
    }
}

/// Finds a separator not protected by a preceding backslash escape.
fn find_unescaped(text: &str, separator: char) -> Option<usize> {
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == separator {
            return Some(index);
        }
    }
    None
}

/// Escapes a separator and backslashes in one exact name component.
fn escape_component(text: &str, separators: &[char]) -> String {
    let mut result = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch == '\\' || separators.contains(&ch) {
            result.push('\\');
        }
        result.push(ch);
    }
    result
}

/// Decodes separator and backslash escapes in one exact name component.
fn unescape_component(text: &str, separators: &[char]) -> Result<String, String> {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        let escaped = chars
            .next()
            .ok_or_else(|| "name ends with an incomplete backslash escape".to_string())?;
        if escaped != '\\' && !separators.contains(&escaped) {
            return Err(format!("unsupported escape '\\{escaped}' in exact name"));
        }
        result.push(escaped);
    }
    Ok(result)
}

/// Parses the one- or two-component exact endpoint payload.
fn parse_endpoint(
    payload: &str,
    kind: Gv2AigBoundaryKind,
) -> Result<Gv2AigBoundaryEndpoint, String> {
    let (instance_name, terminal_name) = if let Some(index) = find_unescaped(payload, ':') {
        if find_unescaped(&payload[index + 1..], ':').is_some() {
            return Err("exact endpoint has more than one unescaped ':' separator".to_string());
        }
        (
            unescape_component(&payload[..index], &[':'])?,
            unescape_component(&payload[index + 1..], &[':'])?,
        )
    } else {
        (String::new(), unescape_component(payload, &[':'])?)
    };
    if terminal_name.is_empty() {
        return Err("endpoint port or pin name must be nonempty".to_string());
    }
    if kind == Gv2AigBoundaryKind::FlopOutput && instance_name.is_empty() {
        return Err("flop output requires '<instance_name>:<pin>'".to_string());
    }
    Ok(Gv2AigBoundaryEndpoint {
        instance_name,
        terminal_name,
    })
}

#[derive(Debug, Clone)]
struct BoundaryCandidate {
    endpoint: Gv2AigBoundaryEndpoint,
    text: String,
}

impl BoundaryCandidate {
    fn new(instance_name: String, terminal_name: String) -> Self {
        let endpoint = Gv2AigBoundaryEndpoint {
            instance_name,
            terminal_name,
        };
        let text = endpoint.render();
        Self { endpoint, text }
    }
}

#[derive(Default)]
struct BoundaryInventory {
    inputs: Vec<BoundaryCandidate>,
    flops: Vec<BoundaryCandidate>,
    outputs: Vec<BoundaryCandidate>,
}

impl BoundaryInventory {
    fn candidates(&self, kind: Gv2AigBoundaryKind) -> &[BoundaryCandidate] {
        match kind {
            Gv2AigBoundaryKind::InputPort => &self.inputs,
            Gv2AigBoundaryKind::FlopOutput => &self.flops,
            Gv2AigBoundaryKind::OutputPort => &self.outputs,
        }
    }
}

impl BoundaryExtractor<'_> {
    /// Expands compact selectors into the exact specification used by
    /// extraction.
    pub(super) fn resolve_request(
        &self,
        request: &Gv2AigBoundaryRequest,
    ) -> Result<Gv2AigBoundarySpec, String> {
        request.validate_roles()?;
        let inventory = self.boundary_inventory();
        let mut boundaries = Gv2AigBoundarySpec {
            sources: Vec::new(),
            sinks: Vec::new(),
        };
        for selection in &request.sources {
            for candidate in resolve_selection(selection, &inventory)? {
                let selector = match selection.kind {
                    Gv2AigBoundaryKind::InputPort => Gv2AigSourceSelector::ModuleInput {
                        instance_name: candidate.endpoint.instance_name.clone(),
                        port: candidate.endpoint.terminal_name.clone(),
                    },
                    Gv2AigBoundaryKind::FlopOutput => Gv2AigSourceSelector::FlopOutput {
                        instance_name: candidate.endpoint.instance_name.clone(),
                        pin: candidate.endpoint.terminal_name.clone(),
                    },
                    Gv2AigBoundaryKind::OutputPort => unreachable!("roles were validated"),
                };
                boundaries.sources.push(Gv2AigSourceBoundary {
                    name: selection
                        .name
                        .clone()
                        .unwrap_or_else(|| candidate.text.clone()),
                    selector,
                });
            }
        }
        for selection in &request.sinks {
            for candidate in resolve_selection(selection, &inventory)? {
                boundaries.sinks.push(Gv2AigSinkBoundary {
                    name: selection
                        .name
                        .clone()
                        .unwrap_or_else(|| candidate.text.clone()),
                    instance_name: candidate.endpoint.instance_name.clone(),
                    port: candidate.endpoint.terminal_name.clone(),
                });
            }
        }
        validate_names(
            boundaries.sources.iter().map(|source| source.name.as_str()),
            "source",
        )?;
        validate_names(
            boundaries.sinks.iter().map(|sink| sink.name.as_str()),
            "sink",
        )?;
        Ok(boundaries)
    }

    /// Collects eligible endpoints once and sorts each kind by canonical text.
    fn boundary_inventory(&self) -> BoundaryInventory {
        let mut inventory = BoundaryInventory::default();
        let mut add_port = |instance_name: &str, port_name: &str, direction: &PortDirection| {
            let candidate =
                BoundaryCandidate::new(instance_name.to_string(), port_name.to_string());
            match direction {
                PortDirection::Input => inventory.inputs.push(candidate),
                PortDirection::Output => inventory.outputs.push(candidate),
                PortDirection::Inout => {
                    // Cone boundaries only support directed module ports.
                }
            }
        };
        for port in &self.normalized.ports {
            add_port(
                "",
                self.interner.resolve(port.name).unwrap(),
                &port.direction,
            );
        }
        for boundary in self.module_boundaries {
            for port in &boundary.ports {
                add_port(
                    &boundary.instance_path,
                    self.interner.resolve(port.name).unwrap(),
                    &port.direction,
                );
            }
        }
        {
            let library = self.liberty;
            for instance in &self.normalized.instances {
                let instance_name = self.interner.resolve(instance.instance_name).unwrap();
                let type_name = self.interner.resolve(instance.type_name).unwrap();
                let cell = &library.cells[self.cell_index_by_name[type_name]];
                if !cell
                    .sequential
                    .iter()
                    .any(|sequential| sequential.kind == SequentialKind::Ff as i32)
                {
                    continue;
                }
                for pin in &cell.pins {
                    if pin.direction != PinDirection::Output as i32 {
                        continue;
                    }
                    let pin_name = library.resolve_string(&pin.name);
                    if self.flop_output_bit(instance_name, pin_name).is_ok() {
                        inventory.flops.push(BoundaryCandidate::new(
                            instance_name.to_string(),
                            pin_name.to_string(),
                        ));
                    }
                }
            }
        }
        inventory.inputs.sort_by(|a, b| a.text.cmp(&b.text));
        inventory.flops.sort_by(|a, b| a.text.cmp(&b.text));
        inventory.outputs.sort_by(|a, b| a.text.cmp(&b.text));
        inventory
    }
}

/// Resolves one selection while preserving the inventory's lexical ordering.
fn resolve_selection<'a>(
    selection: &Gv2AigBoundarySelection,
    inventory: &'a BoundaryInventory,
) -> Result<Vec<&'a BoundaryCandidate>, String> {
    let candidates = inventory.candidates(selection.kind);
    let matched = match &selection.matcher {
        Gv2AigBoundaryMatcher::Exact(endpoint) => candidates
            .iter()
            .filter(|candidate| candidate.endpoint == *endpoint)
            .collect::<Vec<_>>(),
        Gv2AigBoundaryMatcher::Regex(pattern) => {
            let regex = compile_full_regex(pattern).map_err(|error| {
                format!(
                    "invalid regexp in selector '{}': {error}",
                    selection.to_compact_string()
                )
            })?;
            candidates
                .iter()
                .filter(|candidate| regex.is_match(&candidate.text))
                .collect::<Vec<_>>()
        }
    };
    if matched.is_empty() {
        return Err(format!(
            "selector '{}' matched no eligible endpoints",
            selection.to_compact_string()
        ));
    }
    if selection.name.is_some() && matched.len() != 1 {
        return Err(format!(
            "selector '{}' has an AIG name but matched {} endpoints; an explicit name requires exactly one match",
            selection.to_compact_string(),
            matched.len()
        ));
    }
    Ok(matched)
}

/// Anchors a regexp while preserving a trailing comment in verbose mode.
fn compile_full_regex(pattern: &str) -> Result<Regex, regex::Error> {
    match Regex::new(&format!(r"\A(?:{pattern})\z")) {
        Ok(regex) => Ok(regex),
        Err(anchored_error) => {
            Regex::new(pattern)?;
            // A valid verbose-mode regexp can end in a comment that consumes
            // the added closing group. In that case the newline ends the
            // comment and is ignored by verbose mode.
            Regex::new(&format!("\\A(?:{pattern}\n)\\z")).map_err(|_| anchored_error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_selectors_round_trip_exact_names_and_regexp_punctuation() {
        for text in [
            "input_port:data",
            "input_port:u_stage:data",
            "state=flop_output:u_stage/u_reg:Q",
            r"a\=b=output_port:u\:stage:out\\name",
            r"input_port\:alias=output_port:result",
            r"input_port_regex:(?:data|value)_[0-9]{1,2}=x",
            r"flop_output_regex:u_reg\[[0-9]+\]:q",
        ] {
            let selection: Gv2AigBoundarySelection = text.parse().unwrap();
            assert_eq!(selection.to_compact_string(), text);
        }
    }

    #[test]
    fn rejects_invalid_compact_selectors_and_roles() {
        for text in [
            "input_port:",
            "flop_output:q",
            "flop_output:u_reg:q:extra",
            r"input_port:bad\name",
            "unknown:data",
            "=input_port:data",
        ] {
            assert!(text.parse::<Gv2AigBoundarySelection>().is_err(), "{text}");
        }
        assert!(
            Gv2AigBoundaryRequest::from_selectors(
                &["output_port:result".to_string()],
                &["output_port:result".to_string()]
            )
            .is_err()
        );
        assert!(
            Gv2AigBoundaryRequest::from_selectors(&[], &["input_port:data".to_string()]).is_err()
        );
    }

    #[test]
    fn regexp_resolution_is_full_match_sorted_and_checks_alias_cardinality() {
        let mut inventory = BoundaryInventory::default();
        for name in ["data_2", "prefix_data_1", "data_10", "data_1"] {
            inventory
                .inputs
                .push(BoundaryCandidate::new(String::new(), name.to_string()));
        }
        inventory.inputs.sort_by(|a, b| a.text.cmp(&b.text));
        let selection = "input_port_regex:data_[0-9]+".parse().unwrap();
        let matched = resolve_selection(&selection, &inventory).unwrap();
        assert_eq!(
            matched
                .iter()
                .map(|candidate| candidate.text.as_str())
                .collect::<Vec<_>>(),
            vec!["data_1", "data_10", "data_2"]
        );
        let named = "data=input_port_regex:data_[0-9]+".parse().unwrap();
        assert!(resolve_selection(&named, &inventory).is_err());
        let missing = "input_port_regex:missing.*".parse().unwrap();
        assert!(resolve_selection(&missing, &inventory).is_err());
        let invalid = "input_port_regex:[".parse().unwrap();
        assert!(resolve_selection(&invalid, &inventory).is_err());
        let verbose = "input_port_regex:(?x)data_[0-9]+ # selected data ports"
            .parse()
            .unwrap();
        assert_eq!(resolve_selection(&verbose, &inventory).unwrap().len(), 3);
    }
}
