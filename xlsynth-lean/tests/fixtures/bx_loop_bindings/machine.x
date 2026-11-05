// SPDX-License-Identifier: Apache-2.0

import types;
fn fold(pool: types::Pool, mode: types::Mode, count: u8) -> u8 {
    for (bank, acc) in u32:1..u32:3 {
        let slot = pool.slots[bank];
        if slot.write.valid && mode as u2 == u2:2 { acc + slot.write.data } else { acc }
    }(count)
}
fn machine(pool: types::Pool, mode: u2, count: u8) -> u8 {
    fold(pool, mode as types::Mode, count)
}
