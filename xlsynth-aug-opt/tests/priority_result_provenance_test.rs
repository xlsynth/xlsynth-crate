// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use prost_reflect::{DescriptorPool, DynamicMessage, ReflectMessage, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const METADATA: &str = include_str!("testdata/priority_result/provenance.textproto");

/// Parses the fixture schema without accepting omitted proto2 required fields.
fn parse_provenance(text: &str) -> Result<DynamicMessage, String> {
    let pool = DescriptorPool::decode(
        include_bytes!("testdata/priority_result/provenance.bin").as_slice(),
    )
    .map_err(|error| format!("invalid provenance descriptor: {error}"))?;
    let descriptor = pool
        .get_message_by_name("xlsynth.priority_result.Provenance")
        .ok_or_else(|| "provenance descriptor has no Provenance message".to_string())?;
    let metadata = DynamicMessage::parse_text_format(descriptor, text)
        .map_err(|error| format!("invalid provenance: {error}"))?;
    validate_required_fields(&metadata)?;
    Ok(metadata)
}

/// Checks nested repeated records as well as the manifest's required scalars.
fn validate_required_fields(message: &DynamicMessage) -> Result<(), String> {
    for field in message.descriptor().fields() {
        if field.is_required() && !message.has_field(&field) {
            return Err(format!("missing required field {}", field.full_name()));
        }
        match message.get_field(&field).as_ref() {
            Value::Message(nested) => validate_required_fields(nested)?,
            Value::List(values) => {
                for value in values {
                    if let Value::Message(nested) = value {
                        validate_required_fields(nested)?;
                    }
                }
            }
            _ => {
                // Scalar fields have no nested presence requirements.
            }
        }
    }
    Ok(())
}

/// Requires explicit presence instead of returning a protobuf scalar default.
fn required_field<'a>(message: &'a DynamicMessage, name: &str) -> Result<&'a Value, String> {
    message
        .fields()
        .find_map(|(field, value)| (field.name() == name).then_some(value))
        .ok_or_else(|| format!("missing field {name}"))
}

/// Rejects blank provenance values as well as omitted fields.
fn required_string<'a>(message: &'a DynamicMessage, name: &str) -> Result<&'a str, String> {
    required_field(message, name)?
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{name} must be a nonempty string"))
}

/// Validates archived hash and action identifiers without changing their
/// values.
fn required_digest<'a>(message: &'a DynamicMessage, name: &str) -> Result<&'a str, String> {
    let value = required_string(message, name)?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{name} must contain 64 lowercase hexadecimal digits"
        ));
    }
    Ok(value)
}

/// Verifies exact fixture coverage, portable filenames, and checksums.
fn validate_fixture_metadata(root: &Path, text: &str) -> Result<(), Box<dyn Error>> {
    let metadata = parse_provenance(text)?;
    if required_field(&metadata, "schema_version")?.as_u32() != Some(1) {
        return Err("unsupported provenance schema_version".into());
    }
    for field in ["crate_version", "dso_version", "site"] {
        required_string(&metadata, field)?;
    }

    let mut recorded_files = BTreeSet::new();
    for category in ["samples", "synthetic_controls"] {
        let records = required_field(&metadata, category)?
            .as_list()
            .ok_or_else(|| format!("{category} must be a nonempty list"))?;
        for value in records {
            let record = value
                .as_message()
                .ok_or_else(|| format!("{category} entry must be a message"))?;
            let file = required_string(record, "file")?;
            let path = Path::new(file);
            if path.file_name().and_then(|name| name.to_str()) != Some(file)
                || file.contains('\\')
                || path.extension().and_then(|extension| extension.to_str()) != Some("ir")
            {
                return Err(format!("{file}: expected a fixture IR filename").into());
            }
            if !recorded_files.insert(file.to_string()) {
                return Err(format!("{file}: duplicate fixture metadata").into());
            }
            let hash_field = if category == "samples" {
                required_string(record, "source")?;
                for field in [
                    "source_action_id",
                    "extracted_action_id",
                    "computed_structural_hash",
                ] {
                    required_digest(record, field)?;
                }
                "fixture_sha256"
            } else {
                "sha256"
            };
            let expected = required_digest(record, hash_field)?;
            let actual = format!("{:x}", Sha256::digest(fs::read(root.join(path))?));
            if expected != actual {
                return Err(
                    format!("{file}: sha256 mismatch: expected {expected}, got {actual}").into(),
                );
            }
        }
    }

    let mut fixture_files = BTreeSet::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("ir") {
            let name = entry
                .file_name()
                .into_string()
                .map_err(|name| format!("fixture filename is not UTF-8: {name:?}"))?;
            fixture_files.insert(name);
        }
    }
    if recorded_files != fixture_files {
        return Err(format!(
            "fixture/metadata mismatch: recorded {recorded_files:?}, found {fixture_files:?}"
        )
        .into());
    }
    Ok(())
}

/// Locates the checked-in IR inputs and their shared provenance manifest.
fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/priority_result")
}

/// Copies only IR fixtures so rejection tests can mutate isolated inputs.
fn fixture_directory() -> Result<TempDir, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    for entry in fs::read_dir(fixture_root())? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("ir") {
            fs::copy(&path, directory.path().join(path.file_name().unwrap()))?;
        }
    }
    Ok(directory)
}

#[test]
fn checked_in_provenance_matches_every_fixture() -> Result<(), Box<dyn Error>> {
    validate_fixture_metadata(&fixture_root(), METADATA)
}

#[test]
fn rejects_omitted_required_fields_including_repeated_records() {
    let lines: Vec<_> = METADATA.lines().collect();
    for (omitted, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with('#') || !line.contains(':') {
            continue;
        }
        let text = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| (index != omitted).then_some(*line))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            parse_provenance(&text).is_err(),
            "accepted omitted required field: {line}"
        );
    }
}

#[test]
fn rejects_unknown_fields_and_invalid_provenance() {
    for text in [
        format!("{METADATA}\nunknown: true\n"),
        METADATA.replace("samples {", "samples { unknown: true"),
        METADATA.replace("schema_version: 1", "schema_version: 0"),
        METADATA.replace("source: \"std.x::next_pow2\"", "source: \"\""),
        METADATA.replace("crate_version: \"0.73.0\"", "crate_version: \"\""),
        METADATA.replace(
            "d5ea1dffddf3a34454f8c4fbc5fd90cdd12f2dd9f4f0f16acd6b2965fc866b65",
            "invalid-action-id",
        ),
        METADATA.replace(
            "2e2f5734f10c5f591111d0294601894704f10ac4aa29fe8211c9a94e022c79e7",
            "invalid-structural-hash",
        ),
    ] {
        assert!(
            validate_fixture_metadata(&fixture_root(), &text).is_err(),
            "accepted invalid provenance: {text}"
        );
    }
}

#[test]
fn rejects_duplicate_missing_or_nonlocal_fixture_records() {
    let duplicate = METADATA.replace("file: \"shared-amount63.ir\"", "file: \"priority8.ir\"");
    assert_eq!(
        validate_fixture_metadata(&fixture_root(), &duplicate)
            .unwrap_err()
            .to_string(),
        "priority8.ir: duplicate fixture metadata"
    );

    for file in [
        "missing.ir",
        "../priority8.ir",
        "/priority8.ir",
        "./priority8.ir",
        "subdir/priority8.ir",
        "subdir\\priority8.ir",
        "priority8.txt",
    ] {
        let text = METADATA.replace("file: \"priority8.ir\"", &format!("file: {file:?}"));
        assert!(
            validate_fixture_metadata(&fixture_root(), &text).is_err(),
            "accepted invalid fixture filename {file:?}"
        );
    }
}

#[test]
fn rejects_changed_missing_and_unrecorded_ir() -> Result<(), Box<dyn Error>> {
    let directory = fixture_directory()?;
    let path = directory.path().join("priority8.ir");
    let original = fs::read(&path)?;

    fs::write(&path, b"changed after recording provenance\n")?;
    assert!(validate_fixture_metadata(directory.path(), METADATA).is_err());

    fs::remove_file(&path)?;
    assert!(validate_fixture_metadata(directory.path(), METADATA).is_err());

    fs::write(&path, original)?;
    fs::copy(&path, directory.path().join("unrecorded.ir"))?;
    assert!(validate_fixture_metadata(directory.path(), METADATA).is_err());
    Ok(())
}
