// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrBits;

/// User-facing gate-level formal proof backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateFormalBackend {
    Cadical,
    Varisat,
    Z3,
    /// Lower both sides through XLS IR and use the IR equivalence checker.
    Ir,
}

impl GateFormalBackend {
    pub const CLI_VALUES: [&'static str; 4] = ["cadical", "varisat", "z3", "ir"];
    pub const DEFAULT_CLI_VALUE: &'static str = "cadical";

    /// Parses a user-facing backend name.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "cadical" => Ok(Self::Cadical),
            "varisat" => Ok(Self::Varisat),
            "z3" => Ok(Self::Z3),
            "ir" => Ok(Self::Ir),
            _ => Err(value.to_string()),
        }
    }

    /// Returns the canonical lower-case backend name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cadical => "cadical",
            Self::Varisat => "varisat",
            Self::Z3 => "z3",
            Self::Ir => "ir",
        }
    }
}

impl Default for GateFormalBackend {
    fn default() -> Self {
        Self::Cadical
    }
}

impl std::fmt::Display for GateFormalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum EquivResult {
    Proved,
    Disproved(Vec<IrBits>),
}
