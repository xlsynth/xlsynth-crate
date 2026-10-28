// SPDX-License-Identifier: Apache-2.0

//! Typed proof plans, execution reports, and subprocess scheduling.

pub mod config;
pub mod report;
#[cfg(unix)]
pub mod scheduler;
