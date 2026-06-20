// SPDX-License-Identifier: Apache-2.0

use prost_reflect::{DescriptorPool, DynamicMessage, text_format::FormatOptions};

/// Embedded compiled descriptor set for the Liberty wire schema.
///
/// This descriptor is checked in under `proto/liberty.bin` so that crates
/// depending on xlsynth-g8r only need prost/prost-reflect at build time,
/// not protoc/prost-build.
pub const LIBERTY_DESCRIPTOR: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/proto/liberty.bin"));

/// Returns a prost_reflect::DescriptorPool for the embedded liberty descriptor.
pub fn liberty_descriptor_pool() -> DescriptorPool {
    DescriptorPool::decode(LIBERTY_DESCRIPTOR)
        .expect("Failed to decode embedded liberty descriptor")
}

/// Formats encoded `liberty.Library` bytes as readable multiline textproto.
pub fn liberty_proto_bytes_to_pretty_textproto(bytes: &[u8]) -> Result<String, prost::DecodeError> {
    let message_descriptor = liberty_descriptor_pool()
        .get_message_by_name("liberty.Library")
        .expect("embedded descriptor is missing liberty.Library");
    let message = DynamicMessage::decode(message_descriptor, bytes)?;
    let body = message.to_text_format_with_options(&FormatOptions::new().pretty(true));
    Ok(format!(
        "# SPDX-License-Identifier: Apache-2.0\n\
         # proto-file: xlsynth-g8r/proto/liberty.proto\n\
         # proto-message: liberty.Library\n\n\
         {body}\n"
    ))
}
