// SPDX-License-Identifier: Apache-2.0

// Expose the modules needed by the fuzzer and external users.
// Keep this facade minimal to avoid pulling in the whole CLI surface.
pub mod parallelism;
pub mod prover;
pub mod prover_config;
pub mod report_cli_error;
pub mod solver_choice;
pub mod toolchain_config;
