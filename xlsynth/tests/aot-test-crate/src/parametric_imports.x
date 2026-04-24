// SPDX-License-Identifier: Apache-2.0

import parametric_lib;

enum LocalMode : u2 {
    Idle = 0,
    Busy = 1,
}

struct Box<N: u32> {
    value: bits[N],
}

struct Mixed<N: u32> {
    mode: LocalMode,
    remote: parametric_lib::RemotePlain,
    nested: u8[N][2],
    boxes: Box<u32:8>[2],
}

type Mixed4 = Mixed<u32:4>;
type ImportedAlias = parametric_lib::RemoteBox8;

struct ParametricImportsResult {
    mixed: Mixed4,
    imported_alias: ImportedAlias,
}

pub fn exercise_parametric_imports(
    mixed: Mixed4,
    imported_alias: ImportedAlias,
) -> ParametricImportsResult {
    ParametricImportsResult {
        mixed,
        imported_alias,
    }
}
