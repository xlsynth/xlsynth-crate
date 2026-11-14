// SPDX-License-Identifier: Apache-2.0

type Tag = u8;
pub enum Mode: u2 { Idle = 0, Run = 2 }
struct Write { valid: bool, data: u8 }
struct Slot { write: Write }
pub struct Pool<N: u32> { slots: Slot[3], status: (bits[N], Tag) }
