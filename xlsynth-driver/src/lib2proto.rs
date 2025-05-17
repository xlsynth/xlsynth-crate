// SPDX-License-Identifier: Apache-2.0

use prost::Message;
use prost_reflect::{DescriptorPool, DynamicMessage};
use std::fs::File;
use std::io::Write;

pub fn handle_lib2proto(matches: &clap::ArgMatches) {
    use xlsynth_g8r::liberty::liberty_to_proto::parse_liberty_files_to_proto;
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
        let bytes = proto.encode_to_vec();
        f.write_all(&bytes).expect("Failed to write proto");
    } else if output.ends_with(".textproto") {
        // Use prost-reflect to emit textproto
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let descriptor_path = std::path::Path::new(&out_dir).join("liberty.bin");
        let descriptor_bytes = std::fs::read(descriptor_path).expect("Failed to read liberty.bin");
        let descriptor_pool = DescriptorPool::decode(&descriptor_bytes[..]).unwrap();
        let msg_desc = descriptor_pool
            .get_message_by_name("liberty.Library")
            .unwrap();
        let mut dyn_msg = DynamicMessage::new(msg_desc);
        let encoded = proto.encode_to_vec();
        dyn_msg.merge(&encoded[..]).unwrap();
        let text = dyn_msg.to_text_format();
        let mut f = File::create(output).expect("Failed to create output file");
        f.write_all(text.as_bytes())
            .expect("Failed to write textproto");
    } else {
        panic!("Output must end with .proto or .textproto");
    }
}
