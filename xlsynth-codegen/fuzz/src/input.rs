// SPDX-License-Identifier: Apache-2.0

//! Versioned fuzz input regions for stimulus, options, wiring, and body.

use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use xlsynth::external_tool::ToolError;
use xlsynth_codegen::{BlockCodegenOptions, Layout};
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_random::{
    BlockTopology, RandomBlockOptions, RngEntropy, StopPolicy, generate_block_package,
};

use crate::semantics::Trace;
use crate::{block_options, generate};

pub const FORMAT_VERSION: u8 = 1;
pub const GENERATOR_VERSION: u32 = 3;
pub const HEADER_BYTES: usize = 48;
const MAGIC: &[u8; 4] = b"XBCF";

pub struct FuzzCase {
    pub package: Package,
    pub stimulus_seed: [u8; 32],
    pub options: BlockCodegenOptions,
    pub origin: CaseOrigin,
}

pub enum CaseOrigin {
    Guided(Vec<u8>),
    Random(u64),
}

pub struct CheckResult {
    pub trace: Trace,
    pub outcome: CheckOutcome,
}

pub enum CheckOutcome {
    Checked,
    Inconclusive(ToolError),
}

impl CheckResult {
    pub fn coverage_outcome(&self) -> crate::coverage::Outcome<'_> {
        match &self.outcome {
            CheckOutcome::Checked => crate::coverage::Outcome::Checked,
            CheckOutcome::Inconclusive(error) => crate::coverage::Outcome::Inconclusive(error),
        }
    }

    pub fn checked_trace(&self) -> Option<&Trace> {
        matches!(self.outcome, CheckOutcome::Checked).then_some(&self.trace)
    }
}

impl FuzzCase {
    /// Decodes v1 or legacy graph-only inputs. Short recognized headers are
    /// zero-padded, and unknown explicit versions are rejected.
    pub fn decode(data: &[u8]) -> Result<Self, String> {
        let (graph, seed, flags) = if data.starts_with(MAGIC) {
            if data.get(4).copied().unwrap_or_default() != FORMAT_VERSION {
                return Err("unsupported block fuzz input format version".into());
            }
            let mut header = [0; HEADER_BYTES];
            let count = data.len().min(HEADER_BYTES);
            header[..count].copy_from_slice(&data[..count]);
            let seed = header[8..40].try_into().unwrap();
            let flags: [u8; 8] = header[40..48].try_into().unwrap();
            (
                data.get(HEADER_BYTES..).unwrap_or_default(),
                Some(seed),
                flags,
            )
        } else {
            (data, None, [0; 8])
        };
        let (options, codegen_options) = options_for_flags(flags);
        let package = generate(graph, &options);
        let stimulus_seed =
            seed.unwrap_or_else(|| *blake3::hash(package.to_string().as_bytes()).as_bytes());
        Ok(Self {
            package,
            stimulus_seed,
            options: codegen_options,
            origin: CaseOrigin::Guided(data.to_vec()),
        })
    }

    /// Generates a fresh graph from non-depleting entropy with an explicit
    /// size budget; the seed reproduces graph, presentation, and stimuli.
    pub fn random(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut stimulus_seed = [0; 32];
        rng.fill_bytes(&mut stimulus_seed);
        let mut flags = [0; 8];
        rng.fill_bytes(&mut flags);
        let (options, codegen_options) = options_for_flags(flags);
        let body_nodes = rng.gen_range(4..=options.function_options.max_nodes);
        let package = generate_block_package(
            &mut RngEntropy::new(rng),
            &options,
            StopPolicy::ExactBodyNodes(body_nodes),
        )
        .expect("valid random block configuration")
        .package;
        Self {
            package,
            stimulus_seed,
            options: codegen_options,
            origin: CaseOrigin::Random(seed),
        }
    }

    pub fn engine(&self) -> &'static str {
        match self.origin {
            CaseOrigin::Guided(_) => "guided",
            CaseOrigin::Random(_) => "random",
        }
    }

    pub fn trace(&self) -> Trace {
        Trace::with_seed(&self.package, self.stimulus_seed)
    }

    /// Computes one reference trace and checks the generated RTL in iverilog.
    pub fn check(&self) -> CheckResult {
        self.check_with_artifacts(None)
    }

    /// Persists the in-flight reproducer before invoking external tools.
    /// A worker failure leaves these files intact, including on watchdog kill.
    pub fn check_with_artifacts(&self, directory: Option<&Path>) -> CheckResult {
        if let Some(dir) = directory {
            std::fs::create_dir_all(dir).expect("create reproducer directory");
            clear_case_artifacts(dir).expect("clear previous in-flight case");
            let origin = match &self.origin {
                CaseOrigin::Guided(bytes) => {
                    std::fs::write(dir.join("input.bin"), bytes).expect("save guided input");
                    serde_json::json!({"engine":"guided", "input_file":"input.bin"})
                }
                CaseOrigin::Random(seed) => serde_json::json!({"engine":"random", "seed":seed}),
            };
            std::fs::write(dir.join("case.ir"), self.package.to_string()).expect("save case IR");
            let manifest = serde_json::json!({"origin":origin,
                "generator_version":GENERATOR_VERSION, "input_format_version":FORMAT_VERSION,
                "stimulus_seed":self.stimulus_seed, "codegen_options":format!("{:?}", self.options),
                "started_unix":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()});
            std::fs::write(dir.join("case.json"), manifest.to_string())
                .expect("save case manifest");
        }
        let trace = self.trace();
        if let Some(dir) = directory {
            let bindings = |b: &xlsynth_test_helpers::rtl_sim::Bindings| {
                b.iter()
                    .map(|(k, v)| (k.clone(), v.to_bit_string_msb_first()))
                    .collect::<std::collections::BTreeMap<_, _>>()
            };
            let samples: Vec<_> = trace
                .samples
                .iter()
                .map(|s| {
                    serde_json::json!({
                "inputs":bindings(&s.inputs), "outputs":bindings(&s.outputs),
                "next_state":s.next_state.as_ref().map(&bindings)})
                })
                .collect();
            std::fs::write(
                dir.join("trace.json"),
                serde_json::json!({
                "initial_state":bindings(&trace.initial_state),"samples":samples})
                .to_string(),
            )
            .expect("save case trace");
        }
        let outcome = match crate::tool_failure::recover(crate::check_trace(
            &self.package,
            &self.options,
            &trace,
        )) {
            Ok(()) => CheckOutcome::Checked,
            Err(error) => {
                if let Some(dir) = directory {
                    save_inconclusive(dir, &error).expect("save latest inconclusive check");
                }
                CheckOutcome::Inconclusive(error)
            }
        };
        CheckResult { trace, outcome }
    }
}

/// Retains one bounded reproducer snapshot instead of accumulating crash files.
fn save_inconclusive(directory: &Path, error: &ToolError) -> std::io::Result<()> {
    let latest = directory.join("last-inconclusive");
    std::fs::create_dir_all(&latest)?;
    clear_case_artifacts(&latest)?;
    for name in ["case.ir", "case.json", "trace.json", "input.bin"] {
        let source = directory.join(name);
        if source.is_file() {
            std::fs::copy(source, latest.join(name))?;
        }
    }
    std::fs::write(
        latest.join("error.txt"),
        format!("{}\n{error}\n", error.reason_key()),
    )
}

/// Removes stale payloads before preparing a new case; progress/seed records
/// remain available if generation or reference evaluation fails.
pub fn clear_case_artifacts(directory: &Path) -> std::io::Result<()> {
    for name in ["case.ir", "case.json", "trace.json", "input.bin"] {
        match std::fs::remove_file(directory.join(name)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {} // No previous payload.
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn options_for_flags(flags: [u8; 8]) -> (RandomBlockOptions, BlockCodegenOptions) {
    let mut options = block_options(true, true);
    options.topology = if flags[0] & 1 != 0 {
        BlockTopology::FeedForwardPipeline
    } else if flags[2] & 1 != 0 {
        BlockTopology::GeneralSequential
    } else {
        BlockTopology::Combinational
    };
    let codegen = BlockCodegenOptions {
        layout: if options.topology == BlockTopology::FeedForwardPipeline && flags[0] & 2 != 0 {
            Layout::Pipeline
        } else {
            Layout::None
        },
        separate_lines: flags[0] & 4 != 0,
        max_inline_depth: usize::from(flags[1] % 16),
        emit_sv_types: flags[0] & 8 != 0,
        ..BlockCodegenOptions::default()
    };
    (options, codegen)
}

/// Marks generic seed bytes as v1 without injecting any directed circuit.
pub fn mark_versioned(data: &mut Vec<u8>) {
    data.resize(data.len().max(HEADER_BYTES), 0);
    data[..4].copy_from_slice(MAGIC);
    data[4] = FORMAT_VERSION;
    data[5..8].fill(0);
}

#[cfg(test)]
mod tests {
    use super::{FuzzCase, HEADER_BYTES, mark_versioned};
    use crate::iverilog::assert_rtl_trace;
    use crate::{emit, top_block};
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use xlsynth_codegen::{BlockCodegenOptions, Layout};

    #[test]
    fn starting_a_case_clears_stale_payloads_but_keeps_progress() {
        let directory = tempfile::tempdir().unwrap();
        for name in [
            "case.ir",
            "case.json",
            "trace.json",
            "input.bin",
            "progress.json",
        ] {
            std::fs::write(directory.path().join(name), "previous").unwrap();
        }
        super::clear_case_artifacts(directory.path()).unwrap();
        super::clear_case_artifacts(directory.path()).unwrap();
        assert!(directory.path().join("progress.json").is_file());
        assert!(!directory.path().join("trace.json").exists());
        assert!(!directory.path().join("case.json").exists());
    }

    #[test]
    fn random_cases_are_replayable_and_bounded() {
        for seed in 0..256 {
            let a = FuzzCase::random(seed);
            let b = FuzzCase::random(seed);
            assert_eq!(a.package.to_string(), b.package.to_string());
            assert_eq!(a.stimulus_seed, b.stimulus_seed);
            assert_eq!(format!("{:?}", a.options), format!("{:?}", b.options));
            xlsynth_pir::ir_verify::verify_package(&a.package).unwrap();
            let (block, _) = top_block(&a.package);
            assert!(block.nodes.len() - 1 <= 48);
            emit(&a.package, &a.options);
        }
    }

    #[test]
    fn random_cases_match_icarus_and_save_complete_reproducers() {
        let directory = tempfile::tempdir().unwrap();
        for seed in 0..24 {
            let case = FuzzCase::random(seed);
            let checked = case.check_with_artifacts(Some(directory.path()));
            assert!(matches!(checked.outcome, super::CheckOutcome::Checked));
            let manifest: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(directory.path().join("case.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(manifest["origin"]["seed"], seed);
            let trace: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(directory.path().join("trace.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(
                trace["samples"].as_array().unwrap().len(),
                checked.trace.samples.len()
            );
            assert_eq!(
                std::fs::read_to_string(directory.path().join("case.ir")).unwrap(),
                case.package.to_string()
            );
        }
    }

    #[test]
    fn stimulus_mutations_do_not_change_graph_or_codegen_options() {
        let mut data = vec![0; 2048];
        StdRng::seed_from_u64(4).fill_bytes(&mut data);
        mark_versioned(&mut data);
        let a = FuzzCase::decode(&data).unwrap();
        data[8..40].fill(0xfe);
        let b = FuzzCase::decode(&data).unwrap();
        assert_eq!(a.package.to_string(), b.package.to_string());
        assert_eq!(format!("{:?}", a.options), format!("{:?}", b.options));
        assert_ne!(a.trace().samples[15].inputs, b.trace().samples[15].inputs);
    }

    #[test]
    fn short_headers_are_deterministic_and_unknown_versions_are_rejected() {
        let mut short = b"XBCF\x01".to_vec();
        let a = FuzzCase::decode(&short).unwrap();
        short.resize(HEADER_BYTES, 0);
        let b = FuzzCase::decode(&short).unwrap();
        assert_eq!(a.package.to_string(), b.package.to_string());
        assert_eq!(a.stimulus_seed, b.stimulus_seed);
        short[4] = 2;
        assert!(FuzzCase::decode(&short).is_err());
    }

    #[test]
    fn staged_scalar_and_aggregate_graphs_match_both_layouts() {
        let mut saw_aggregate = false;
        for seed in 0..16 {
            let mut bytes = vec![0; 1024];
            StdRng::seed_from_u64(seed).fill_bytes(&mut bytes);
            mark_versioned(&mut bytes);
            bytes[40] |= 3;
            let case = FuzzCase::decode(&bytes).unwrap();
            let (_, metadata) = top_block(&case.package);
            assert!(metadata.registers.len() >= 2);
            saw_aggregate |= metadata
                .registers
                .iter()
                .any(|r| !matches!(r.ty, xlsynth_pir::ir::Type::Bits(_)));
            assert_eq!(case.options.layout, Layout::Pipeline);
            let trace = case.trace();
            for options in [&BlockCodegenOptions::default(), &case.options] {
                let rtl = emit(&case.package, options);
                assert_rtl_trace(&case.package, &rtl, None, &trace).unwrap();
            }
            // Presentation mutations must not perturb graph or stimulus
            // entropy.
            bytes[41] ^= 0xff;
            bytes[40] ^= 12;
            let other = FuzzCase::decode(&bytes).unwrap();
            assert_eq!(other.package.to_string(), case.package.to_string());
            assert_eq!(other.stimulus_seed, case.stimulus_seed);
        }
        assert!(saw_aggregate);
    }
}
