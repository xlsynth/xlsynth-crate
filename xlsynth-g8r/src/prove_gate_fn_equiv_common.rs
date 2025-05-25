// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrBits;

#[derive(Debug, PartialEq, Eq)]
pub enum EquivResult {
    Proved,
    Disproved(Vec<IrBits>),
}
