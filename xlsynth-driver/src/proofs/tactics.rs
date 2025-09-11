// SPDX-License-Identifier: Apache-2.0

pub mod cosliced;
pub mod focus;
pub mod utils;

use self::cosliced::CoslicedTactic;
use self::focus::FocusTactic;
use crate::proofs::obligations::LecObligation;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Object-safe tactic interface. Tactics transform a base obligation into
/// zero or more sub-obligations.
pub trait IsTactic: Send + Sync + std::fmt::Debug + Serialize + DeserializeOwned {
    /// Human-readable tactic name.
    fn name(&self) -> &'static str;

    /// Applies the tactic to the given base obligation.
    fn apply(&self, base: &LecObligation) -> Result<Vec<LecObligation>, String>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Tactic {
    Cosliced(CoslicedTactic),
    Focus(FocusTactic),
}

impl IsTactic for Tactic {
    fn name(&self) -> &'static str {
        match self {
            Tactic::Cosliced(t) => t.name(),
            Tactic::Focus(t) => t.name(),
        }
    }

    fn apply(&self, base: &LecObligation) -> Result<Vec<LecObligation>, String> {
        match self {
            Tactic::Cosliced(t) => t.apply(base),
            Tactic::Focus(t) => t.apply(base),
        }
    }
}
