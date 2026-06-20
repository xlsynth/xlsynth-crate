// SPDX-License-Identifier: Apache-2.0

use prost::Message;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use xlsynth_g8r::liberty::CellFilterPolicy;
use xlsynth_g8r::liberty::descriptor::liberty_proto_bytes_to_pretty_textproto;
use xlsynth_g8r::liberty::model::{library_to_proto, strip_power_data, strip_timing_data};
use xlsynth_g8r::liberty::parser::{
    ThresholdVoltageGroupRule, parse_liberty_files_with_vt_rules_without_payload_validation,
    validate_library_consistency,
};

fn parse_vt_group_rule(raw: &str) -> Result<ThresholdVoltageGroupRule, String> {
    let mut parts = raw.splitn(3, ':');
    let Some(name) = parts.next() else {
        return Err(format!(
            "invalid --vt-group '{}'; expected NAME:CLASS_INDEX:REGEX",
            raw
        ));
    };
    let Some(class_index) = parts.next() else {
        return Err(format!(
            "invalid --vt-group '{}'; expected NAME:CLASS_INDEX:REGEX",
            raw
        ));
    };
    let Some(cell_name_regex) = parts.next() else {
        return Err(format!(
            "invalid --vt-group '{}'; expected NAME:CLASS_INDEX:REGEX",
            raw
        ));
    };
    if name.is_empty() {
        return Err(format!(
            "invalid --vt-group '{}'; VT group name must be non-empty",
            raw
        ));
    }
    if cell_name_regex.is_empty() {
        return Err(format!(
            "invalid --vt-group '{}'; regex must be non-empty",
            raw
        ));
    }
    let class_index = class_index.parse::<i32>().map_err(|e| {
        format!(
            "invalid --vt-group '{}'; VT class index must be a signed integer: {}",
            raw, e
        )
    })?;
    Ok(ThresholdVoltageGroupRule {
        name: name.to_string(),
        class_index,
        cell_name_regex: cell_name_regex.to_string(),
    })
}

pub fn handle_lib2proto(matches: &clap::ArgMatches) {
    let liberty_files: Vec<_> = matches
        .get_many::<String>("liberty_files")
        .unwrap()
        .map(|s| s.as_str())
        .collect();
    let output = matches.get_one::<String>("output").unwrap();
    let vt_group_rules = matches
        .get_many::<String>("vt_group")
        .map(|values| {
            values
                .map(|value| parse_vt_group_rule(value))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .unwrap_or_else(|e| panic!("{e}"))
        .unwrap_or_default();

    let mut proto = parse_liberty_files_with_vt_rules_without_payload_validation(
        &liberty_files,
        &vt_group_rules,
    )
    .expect("Failed to parse Liberty files");
    proto.provenance = matches
        .get_one::<String>("provenance")
        .cloned()
        .unwrap_or_default();
    proto.source_files = liberty_files
        .iter()
        .map(|path| {
            Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(path)
                .to_string()
        })
        .collect();
    if let Some(policy_path) = matches.get_one::<String>("cell_filter_policy") {
        let policy = CellFilterPolicy::from_path(Path::new(policy_path))
            .unwrap_or_else(|e| panic!("Failed to load cell-filter policy: {e:#}"));
        let stats = policy.apply(&mut proto);
        log::info!(
            "Applied {} cell-filter rules: input={} native_dont_use={} retained={} removed={}",
            policy.rules().len(),
            stats.input_cells,
            stats.native_dont_use_cells,
            stats.retained_cells,
            stats.removed_cells
        );
    }
    if !matches.get_flag("include_timing") {
        strip_timing_data(&mut proto);
    }
    if !matches.get_flag("include_power") {
        strip_power_data(&mut proto);
    }
    validate_library_consistency(&proto)
        .unwrap_or_else(|e| panic!("Failed to validate retained Liberty data: {e}"));
    let proto = library_to_proto(proto).expect("Failed to encode Liberty LUT data");

    if output.ends_with(".proto") {
        let mut f = File::create(output).expect("Failed to create output file");
        log::info!("Encoding proto...");
        let bytes = proto.encode_to_vec();
        log::info!("Writing proto...");
        f.write_all(&bytes).expect("Failed to write proto");
    } else if output.ends_with(".textproto") {
        log::info!("Encoding proto...");
        let encoded = proto.encode_to_vec();
        let text = liberty_proto_bytes_to_pretty_textproto(&encoded)
            .expect("Failed to format Liberty textproto");
        log::info!("Writing textproto...");
        let mut f = File::create(output).expect("Failed to create output file");
        f.write_all(text.as_bytes())
            .expect("Failed to write textproto");
    } else {
        panic!(
            "Output must end with .proto or .textproto; got {:?}",
            output
        );
    }
}
