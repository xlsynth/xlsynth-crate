// SPDX-License-Identifier: Apache-2.0

use prost_reflect::DescriptorPool;

/// Embedded compiled descriptor set for liberty.proto
pub const LIBERTY_DESCRIPTOR: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/proto/liberty.bin"));

/// Returns a prost_reflect::DescriptorPool for the embedded liberty descriptor.
pub fn liberty_descriptor_pool() -> DescriptorPool {
    DescriptorPool::decode(LIBERTY_DESCRIPTOR)
        .expect("Failed to decode embedded liberty descriptor")
}
