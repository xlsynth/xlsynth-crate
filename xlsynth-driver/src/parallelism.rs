// SPDX-License-Identifier: Apache-2.0

#[derive(Clone, Copy, Debug)]
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
