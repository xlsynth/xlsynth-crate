// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use anyhow::Result;

use super::ResolvedStitchPipeline;
use crate::stitch_equiv::{StitchVectorResult, StitchVectors, check_stitch_vectors};

impl ResolvedStitchPipeline {
    /// Checks one continuous stream of named inputs against the mapped module,
    /// including pipeline drain.
    pub fn check_vectors_against_gv(
        &self,
        netlist_path: &Path,
        liberty_path: &Path,
        module_name: Option<&str>,
        vectors: StitchVectors,
    ) -> Result<StitchVectorResult> {
        let mapped = self.load_mapped_for_check(netlist_path, liberty_path, module_name)?;
        check_stitch_vectors(
            &self.ir,
            &self.stages,
            &self.boundary,
            self.latency(),
            &mapped,
            vectors,
        )
        .map_err(anyhow::Error::msg)
    }
}
