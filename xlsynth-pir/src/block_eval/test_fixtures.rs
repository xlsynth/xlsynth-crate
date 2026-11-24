// SPDX-License-Identifier: Apache-2.0

//! Synthetic PIR blocks shared by the interpreted and compiled execution tests.

pub const AGGREGATE_STATE: &str =
    include_str!("../../tests/fixtures/block_eval/aggregate_state.ir");
pub const CALLED_COVER: &str = include_str!("../../tests/fixtures/block_eval/called_cover.ir");
pub const CALLED_FUNCTION: &str =
    include_str!("../../tests/fixtures/block_eval/called_function.ir");
pub const CHILD_BOUNDS: &str = include_str!("../../tests/fixtures/block_eval/child_bounds.ir");
pub const EXCLUSIVE_WRITES: &str =
    include_str!("../../tests/fixtures/block_eval/exclusive_writes.ir");
pub const MIXED_RESET: &str = include_str!("../../tests/fixtures/block_eval/mixed_reset.ir");
pub const MODELED_EXTERNAL: &str =
    include_str!("../../tests/fixtures/block_eval/modeled_external.ir");
pub const OPAQUE_EXTERNAL: &str =
    include_str!("../../tests/fixtures/block_eval/opaque_external.ir");
pub const SIBLING_STATE: &str = include_str!("../../tests/fixtures/block_eval/sibling_state.ir");
