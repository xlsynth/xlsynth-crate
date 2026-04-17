// SPDX-License-Identifier: Apache-2.0

enum RouteKind : u2 {
    Local = 0,
    Remote = 1,
    Drop = 2,
}

type Nibbles = u4[3];

struct Packet {
    tag: u8,
    kind: RouteKind,
    lanes: Nibbles,
}

struct RouteResult {
    next_tag: u8,
    selected_lane: u4,
    kind: RouteKind,
}

pub fn pretty_route(packet: Packet, salt: Nibbles) -> RouteResult {
    RouteResult {
        next_tag: packet.tag + u8:1,
        selected_lane: packet.lanes[u32:1] + salt[u32:2],
        kind: packet.kind,
    }
}
