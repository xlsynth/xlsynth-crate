// SPDX-License-Identifier: Apache-2.0

struct RoutePacket {
    tag: u8,
}

pub fn route(packet: RoutePacket) -> RoutePacket {
    packet
}
