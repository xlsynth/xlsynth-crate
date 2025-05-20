// SPDX-License-Identifier: Apache-2.0

use prost::Message;
use prost_reflect::DynamicMessage;
use std::fs::File;
use std::io::Write;
use xlsynth_g8r::liberty::descriptor::liberty_descriptor_pool;
use xlsynth_g8r::liberty::liberty_to_proto::parse_liberty_files_to_proto;

pub fn handle_lib2proto(matches: &clap::ArgMatches) {
    let liberty_files: Vec<_> = matches
        .get_many::<String>("liberty_files")
        .unwrap()
        .map(|s| s.as_str())
        .collect();
    let output = matches.get_one::<String>("output").unwrap();

    let proto =
        parse_liberty_files_to_proto(&liberty_files).expect("Failed to parse Liberty files");

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
