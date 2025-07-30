// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ParallelismStrategy {
    SingleThreaded,
    OutputBits,
    InputBitSplit,
}

impl std::str::FromStr for ParallelismStrategy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "single-threaded" => Ok(Self::SingleThreaded),
            "output-bits" => Ok(Self::OutputBits),
            "input-bit-split" => Ok(Self::InputBitSplit),
            _ => Err(format!("invalid parallelism strategy: {}", s)),
        }
    }
}

impl fmt::Display for ParallelismStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ParallelismStrategy::SingleThreaded => "single-threaded",
            ParallelismStrategy::OutputBits => "output-bits",
            ParallelismStrategy::InputBitSplit => "input-bit-split",
        };
        write!(f, "{}", s)
    }
}
