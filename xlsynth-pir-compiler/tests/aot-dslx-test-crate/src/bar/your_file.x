// SPDX-License-Identifier: Apache-2.0

import types.shared_types as shared_doodle;

pub fn bump_doodle(doodle: shared_doodle::Doodle) -> shared_doodle::Doodle {
    shared_doodle::Doodle { doodle_id: doodle.doodle_id + u8:1 }
}
