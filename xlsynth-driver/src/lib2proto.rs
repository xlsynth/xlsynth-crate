// SPDX-License-Identifier: Apache-2.0

use prost::Message;
use prost_reflect::DynamicMessage;
use std::fs::File;
use std::io::Write;
use xlsynth_g8r::liberty::descriptor::liberty_descriptor_pool;
use xlsynth_g8r::liberty::liberty_to_proto::{
    parse_liberty_files_to_proto_with_vt_rules, ThresholdVoltageGroupRule,
};

fn parse_vt_group_rule(raw: &str) -> Result<ThresholdVoltageGroupRule, String> {
    let Some((name, cell_name_regex)) = raw.split_once(':') else {
        return Err(format!("invalid --vt-group '{}'; expected NAME:REGEX", raw));
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
    Ok(ThresholdVoltageGroupRule {
        name: name.to_string(),
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

    let proto = parse_liberty_files_to_proto_with_vt_rules(&liberty_files, &vt_group_rules)
        .expect("Failed to parse Liberty files");

    if output.ends_with(".proto") {
        let mut f = File::create(output).expect("Failed to create output file");
        log::info!("Encoding proto...");
        let bytes = proto.encode_to_vec();
        log::info!("Writing proto...");
        f.write_all(&bytes).expect("Failed to write proto");
    } else if output.ends_with(".textproto") {
        // Use prost-reflect to emit textproto
        let descriptor_pool = liberty_descriptor_pool();
        let msg_desc = descriptor_pool
            .get_message_by_name("liberty.Library")
            .unwrap();
        let mut dyn_msg = DynamicMessage::new(msg_desc);
        log::info!("Encoding proto...");
        let encoded = proto.encode_to_vec();
        log::info!("Merging proto...");
        dyn_msg.merge(&encoded[..]).unwrap();
        log::info!("Writing textproto...");
        let text = dyn_msg.to_text_format();
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
