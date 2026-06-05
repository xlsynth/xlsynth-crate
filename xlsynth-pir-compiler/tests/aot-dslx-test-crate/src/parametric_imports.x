// SPDX-License-Identifier: Apache-2.0

import parametric_lib;

struct ParametricImportsResult {
    remote: parametric_lib::RemotePlain,
    imported_direct: parametric_lib::RemoteBox<u32:8>,
    imported_pair: parametric_lib::RemotePair<u32:8, u32:23>,
}

pub fn exercise_parametric_imports(
    remote: parametric_lib::RemotePlain,
    imported_direct: parametric_lib::RemoteBox<u32:8>,
    imported_pair: parametric_lib::RemotePair<u32:8, u32:23>,
) -> ParametricImportsResult {
    ParametricImportsResult {
        remote,
        imported_direct,
        imported_pair,
    }
}
