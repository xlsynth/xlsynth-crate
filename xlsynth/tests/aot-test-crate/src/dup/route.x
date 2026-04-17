// SPDX-License-Identifier: Apache-2.0

import foo.packet as foo_packet;
import bar.packet as bar_packet;

pub fn route(packet: foo_packet::Packet) -> bar_packet::Packet {
    bar_packet::Packet { tag: packet.tag + u8:1 }
}
