// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use prost_reflect::{
    DescriptorPool, DynamicMessage, MessageDescriptor, ReflectMessage, SerializeOptions, Value,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use xlsynth_g8r::process_ir_path::CanonicalG8rOptions;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;

const PROFILE_FILE_NAME: &str = "profile.textproto";

/// Shares one mapping profile across validated, name-ordered regression cases.
#[derive(Debug)]
pub struct Corpus {
    pub profile: MappingProfile,
    pub cases: Vec<Case>,
}

/// Pins the gate mapping configuration and numerical comparison tolerance.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MappingProfile {
    pub canonical_options: CanonicalG8rOptions,
    pub cut_db_rewrite_max_iterations: usize,
    pub cut_db_rewrite_max_cuts_per_node: usize,
    pub graph_le_tolerance: f64,
}

/// Owns the exact fixture text, its declared top function, and its
/// expectations.
#[derive(Debug)]
pub struct Case {
    pub name: String,
    pub text: String,
    pub original: ir::Fn,
    pub expectations: Expectations,
}

/// Selects the comparisons required by the fixture's optimizer family.
#[derive(Debug)]
pub enum Expectations {
    Shift {
        require_improvement: bool,
        limits: Limits,
    },
    SplitAdder {
        disabled_limits: Limits,
        enabled_limits: Limits,
        same_as: Option<String>,
    },
    PriorityResult {
        expected_fusions: usize,
        disabled_limits: Limits,
        enabled_limits: Limits,
    },
}

/// Allows improvements while bounding mapped area, logical effort, and depth.
#[derive(Debug)]
pub struct Limits {
    pub and_nodes_max: usize,
    pub graph_le_max: f64,
    pub depth_max: usize,
}

/// Loads the checked-in corpus shared by the characterization tests.
pub fn load() -> Result<Corpus, String> {
    load_from_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mffc_regressions"))
}

/// Discovers paired fixtures and validates all metadata before any measurement.
pub fn load_from_dir(root: &Path) -> Result<Corpus, String> {
    let pool = DescriptorPool::decode(include_bytes!("corpus.bin").as_slice())
        .map_err(|error| format!("invalid corpus descriptor: {error}"))?;
    let profile_path = root.join(PROFILE_FILE_NAME);
    let profile = load_profile(&profile_path, &pool)
        .map_err(|error| format!("{}: {error}", profile_path.display()))?;
    let descriptor = pool
        .get_message_by_name("xlsynth.mffc_regressions.Case")
        .ok_or_else(|| "corpus descriptor has no Case message".to_string())?;
    let mut cases = Vec::new();
    for path in discover_pairs(root)? {
        let case = load_case(&path, &descriptor, profile.graph_le_tolerance)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        cases.push(case);
    }
    validate_references(&cases)?;
    Ok(Corpus { profile, cases })
}

/// Reuses the Rust options' enum parsing after validating the textproto schema.
fn load_profile(path: &Path, pool: &DescriptorPool) -> Result<MappingProfile, String> {
    let descriptor = pool
        .get_message_by_name("xlsynth.mffc_regressions.MappingProfile")
        .ok_or_else(|| "corpus descriptor has no MappingProfile message".to_string())?;
    let message = read_textproto(path, &descriptor)?;
    // Preserve Rust field names and numeric u64 values through the serde
    // bridge. Absent optional mul_adder_mapping remains absent, so serde
    // produces None.
    let options = SerializeOptions::new()
        .use_proto_field_name(true)
        .stringify_64_bit_integers(false);
    let value = message
        .serialize_with_options(serde_json::value::Serializer, &options)
        .map_err(|error| error.to_string())?;
    let profile: MappingProfile =
        serde_json::from_value(value).map_err(|error| error.to_string())?;
    if !profile.graph_le_tolerance.is_finite() || profile.graph_le_tolerance < 0.0 {
        return Err("graph_le_tolerance must be finite and nonnegative".to_string());
    }
    Ok(profile)
}

/// Parses typed fixture metadata and rejects omissions before serde can
/// default.
fn read_textproto(path: &Path, descriptor: &MessageDescriptor) -> Result<DynamicMessage, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let message = DynamicMessage::parse_text_format(descriptor.clone(), &text)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    validate_required_fields(&message).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(message)
}

/// Checks required fields in this schema's scalar and nested-message records.
fn validate_required_fields(message: &DynamicMessage) -> Result<(), String> {
    for field in message.descriptor().fields() {
        if message.has_field(&field) {
            if let Some(nested) = message.get_field(&field).as_message() {
                validate_required_fields(nested)?;
            }
        } else if field.is_required() {
            return Err(format!("missing required field {}", field.full_name()));
        }
    }
    Ok(())
}

/// Finds IR inputs in deterministic order and rejects unpaired sidecars.
fn discover_pairs(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut inputs = Vec::new();
    let mut sidecars = BTreeSet::new();
    let profile_path = root.join(PROFILE_FILE_NAME);
    for entry in std::fs::read_dir(root).map_err(|error| format!("{}: {error}", root.display()))? {
        let path = entry
            .map_err(|error| format!("{}: {error}", root.display()))?
            .path();
        if path == profile_path {
            // The shared profile is not a per-fixture expectation sidecar.
            continue;
        }
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("ir") => inputs.push(path),
            Some("textproto") => {
                sidecars.insert(path);
            }
            _ => {
                // The profile and documentation are not fixture inputs.
            }
        }
    }
    inputs.sort();
    for input in &inputs {
        let sidecar = input.with_extension("textproto");
        if !sidecars.remove(&sidecar) {
            return Err(format!("{}: missing paired sidecar", sidecar.display()));
        }
    }
    if let Some(sidecar) = sidecars.first() {
        return Err(format!(
            "{}: orphan sidecar has no paired IR",
            sidecar.display()
        ));
    }
    if inputs.is_empty() {
        return Err(format!("{}: corpus has no IR fixtures", root.display()));
    }
    Ok(inputs)
}

/// Reads one sidecar and verifies that its expectations apply to the exact IR.
fn load_case(path: &Path, descriptor: &MessageDescriptor, tolerance: f64) -> Result<Case, String> {
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| "fixture filename must have a UTF-8 stem".to_string())?
        .to_string();
    let sidecar_path = path.with_extension("textproto");
    let metadata = read_textproto(&sidecar_path, descriptor)?;
    let sha256 = required_field(&metadata, "sha256")?
        .as_str()
        .ok_or_else(|| "sha256 must be a string".to_string())?;
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let actual_hash = format!("{:x}", Sha256::digest(text.as_bytes()));
    if sha256 != actual_hash {
        return Err(format!(
            "sha256 mismatch: sidecar records {sha256}, IR hashes to {actual_hash}"
        ));
    }
    let expectations = parse_expectations(&metadata, tolerance)?;
    let package = Parser::new(&text)
        .parse_and_verify_package()
        .map_err(|error| format!("invalid IR: {error}"))?;
    if package.top.is_none() {
        return Err("IR fixture must declare a top function".to_string());
    }
    let original = package
        .get_top_fn()
        .cloned()
        .ok_or_else(|| "IR fixture top must be a function".to_string())?;
    Ok(Case {
        name,
        text,
        original,
        expectations,
    })
}

/// Requires explicit presence, including for fields whose scalar value is zero.
fn required_field<'a>(message: &'a DynamicMessage, name: &str) -> Result<&'a Value, String> {
    // fields() visits populated fields only; get_field_by_name() would also
    // return protobuf defaults for omitted required fields.
    message
        .fields()
        .find_map(|(field, value)| (field.name() == name).then_some(value))
        .ok_or_else(|| format!("missing required field {name}"))
}

/// Extracts a required nested message without accepting an omitted default.
fn required_message<'a>(
    message: &'a DynamicMessage,
    name: &str,
) -> Result<&'a DynamicMessage, String> {
    required_field(message, name)?
        .as_message()
        .ok_or_else(|| format!("{name} must be a message"))
}

/// Converts an explicit protobuf bound without truncating it on narrower hosts.
fn required_usize(message: &DynamicMessage, name: &str) -> Result<usize, String> {
    let value = required_field(message, name)?
        .as_u64()
        .ok_or_else(|| format!("{name} must be an unsigned integer"))?;
    usize::try_from(value).map_err(|_| format!("{name} is too large for this host"))
}

/// Rejects nonfinite bounds so a malformed fixture cannot disable comparison.
fn parse_limits(message: &DynamicMessage, tolerance: f64) -> Result<Limits, String> {
    let graph_le_max = required_field(message, "graph_le_max")?
        .as_f64()
        .ok_or_else(|| "graph_le_max must be a double".to_string())?;
    if !graph_le_max.is_finite() || graph_le_max < 0.0 {
        return Err("graph_le_max must be finite and nonnegative".to_string());
    }
    if !(graph_le_max + tolerance).is_finite() {
        return Err("graph_le_max plus graph_le_tolerance must be finite".to_string());
    }
    Ok(Limits {
        and_nodes_max: required_usize(message, "and_nodes_max")?,
        graph_le_max,
        depth_max: required_usize(message, "depth_max")?,
    })
}

/// Decodes the single expectation family selected by the sidecar's oneof.
fn parse_expectations(message: &DynamicMessage, tolerance: f64) -> Result<Expectations, String> {
    if message.has_field_by_name("shift") {
        let shift = required_message(message, "shift")?;
        let require_improvement = required_field(shift, "require_improvement")?
            .as_bool()
            .ok_or_else(|| "require_improvement must be a boolean".to_string())?;
        let limits = parse_limits(required_message(shift, "limits")?, tolerance)?;
        Ok(Expectations::Shift {
            require_improvement,
            limits,
        })
    } else if message.has_field_by_name("split_adder") {
        let split_adder = required_message(message, "split_adder")?;
        let disabled_limits =
            parse_limits(required_message(split_adder, "disabled_limits")?, tolerance)?;
        let enabled_limits =
            parse_limits(required_message(split_adder, "enabled_limits")?, tolerance)?;
        let same_as = if split_adder.has_field_by_name("same_as") {
            Some(
                required_field(split_adder, "same_as")?
                    .as_str()
                    .ok_or_else(|| "same_as must be a string".to_string())?
                    .to_string(),
            )
        } else {
            None
        };
        Ok(Expectations::SplitAdder {
            disabled_limits,
            enabled_limits,
            same_as,
        })
    } else if message.has_field_by_name("priority_result") {
        let priority = required_message(message, "priority_result")?;
        Ok(Expectations::PriorityResult {
            expected_fusions: required_usize(priority, "expected_fusions")?,
            disabled_limits: parse_limits(
                required_message(priority, "disabled_limits")?,
                tolerance,
            )?,
            enabled_limits: parse_limits(required_message(priority, "enabled_limits")?, tolerance)?,
        })
    } else {
        Err("missing expectation family: set shift, split_adder, or priority_result".to_string())
    }
}

/// Checks ordering peers against the whole corpus, independent of file order.
fn validate_references(cases: &[Case]) -> Result<(), String> {
    let by_name: BTreeMap<_, _> = cases
        .iter()
        .map(|case| (case.name.as_str(), case))
        .collect();
    for case in cases {
        let Expectations::SplitAdder {
            same_as: Some(peer),
            ..
        } = &case.expectations
        else {
            // Cases without an ordering peer need no cross-fixture validation.
            continue;
        };
        if peer == &case.name {
            return Err(format!("{}: same_as must name another fixture", case.name));
        }
        let target = by_name
            .get(peer.as_str())
            .ok_or_else(|| format!("{}: same_as refers to missing fixture {peer}", case.name))?;
        if !matches!(target.expectations, Expectations::SplitAdder { .. }) {
            return Err(format!(
                "{}: same_as target {peer} must be a split-adder fixture",
                case.name
            ));
        }
    }
    Ok(())
}
