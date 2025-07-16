// SPDX-License-Identifier: Apache-2.0

//! Shared enumeration for selecting an SMT solver backend on the command line.
//!
//! This lives in its own module so that different sub-commands (e.g. `ir-equiv`
//! and `prove-quickcheck`) can share a single definition and avoid code
//! duplication.

#[derive(Debug, Clone, Copy)]
pub enum SolverChoice {
    #[cfg(feature = "has-easy-smt")]
    Z3Binary,
    #[cfg(feature = "has-easy-smt")]
    BitwuzlaBinary,
    #[cfg(feature = "has-easy-smt")]
    BoolectorBinary,

    #[cfg(feature = "has-bitwuzla")]
    Bitwuzla,

    #[cfg(feature = "has-boolector")]
    Boolector,
    #[cfg(feature = "has-boolector")]
    BoolectorLegacy,

    /// Use the external XLS tool-chain binaries (whatever is configured via
    /// `xlsynth-toolchain.toml`).  Currently only used by the `ir-equiv`
    /// sub-command – other commands may ignore this variant.
    Toolchain,
}

impl std::str::FromStr for SolverChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            #[cfg(feature = "has-easy-smt")]
            "z3-binary" => Ok(Self::Z3Binary),
            #[cfg(feature = "has-easy-smt")]
            "bitwuzla-binary" => Ok(Self::BitwuzlaBinary),
            #[cfg(feature = "has-easy-smt")]
            "boolector-binary" => Ok(Self::BoolectorBinary),

            #[cfg(feature = "has-bitwuzla")]
            "bitwuzla" => Ok(Self::Bitwuzla),

            #[cfg(feature = "has-boolector")]
            "boolector" => Ok(Self::Boolector),
            #[cfg(feature = "has-boolector")]
            "boolector-legacy" => Ok(Self::BoolectorLegacy),

            "toolchain" => Ok(Self::Toolchain),
            _ => Err(format!("invalid solver: {}", s)),
        }
    }
}
