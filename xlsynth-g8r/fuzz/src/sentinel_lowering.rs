// SPDX-License-Identifier: Apache-2.0

//! Focused sentinel-lowering fuzz checks, independent of gate cost estimates.

use std::collections::BTreeMap;

use xlsynth_g8r::aig_serdes::gate2ir;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::IrValue;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir_eval::eval_fn;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::math::ceil_log2;
use xlsynth_prover::prover::types::EquivResult;
use xlsynth_prover::prover::{SolverChoice, prover_for_choice_with_limits};

use crate::fuzz_solver_limits;

pub const FAMILIES: [&str; 6] = [
    "clz_canonical",
    "priority_msb",
    "priority_lsb",
    "clz_offset",
    "clz_widened",
    "clz_truncated",
];
const WIDTHS: [usize; 19] = [
    1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 161,
];

/// Counts actual completed lowerings and proofs, not merely generator labels.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub samples: u64,
    pub canonical_lowerings: u64,
    pub direct_count_bits: u64,
    pub proofs: u64,
    pub inconclusive: u64,
    pub zero_input_checks: u64,
}

impl Counts {
    fn accumulate(&mut self, other: Self) {
        self.samples += other.samples;
        self.canonical_lowerings += other.canonical_lowerings;
        self.direct_count_bits += other.direct_count_bits;
        self.proofs += other.proofs;
        self.inconclusive += other.inconclusive;
        self.zero_input_checks += other.zero_input_checks;
    }
}

/// Preserves family, width, sharing and builder-option coverage independently.
#[derive(Debug, Default)]
pub struct Coverage {
    pub total: Counts,
    pub families: BTreeMap<String, Counts>,
    pub features: BTreeMap<String, Counts>,
}

impl Coverage {
    pub fn accumulate(&mut self, other: Self) {
        self.total.accumulate(other.total);
        for (name, counts) in other.families {
            self.families.entry(name).or_default().accumulate(counts);
        }
        for (name, counts) in other.features {
            self.features.entry(name).or_default().accumulate(counts);
        }
    }

    /// Rejects empty, unproved, or missing canonical/fallback coverage.
    pub fn validate(&self) -> Result<(), String> {
        for (index, name) in FAMILIES.iter().enumerate() {
            let c = self
                .families
                .get(*name)
                .ok_or_else(|| format!("missing {name}"))?;
            if c.proofs == 0
                || c.zero_input_checks == 0
                || (index < 3 && (c.canonical_lowerings == 0 || c.direct_count_bits == 0))
            {
                return Err(format!("unexercised or unproved lowering {name}"));
            }
        }
        for name in [
            "shared_input",
            "shared_result",
            "unshared",
            "fold_hash_on",
            "fold_hash_off",
            "post_count_wrap",
            "wide_output",
        ] {
            if self.features.get(name).is_none_or(|c| c.proofs == 0) {
                return Err(format!("missing proved feature {name}"));
            }
        }
        for n in WIDTHS {
            if self
                .features
                .get(&format!("width_{n}"))
                .is_none_or(|c| c.proofs == 0)
            {
                return Err(format!("missing proved input width {n}"));
            }
        }
        if self.total.inconclusive != 0 {
            return Err("inconclusive sentinel proofs in audited corpus".to_string());
        }
        Ok(())
    }
}

/// Lowers a live extension, then proves its gates against desugared basis IR.
pub fn check_input(data: &[u8]) -> Result<Coverage, String> {
    let byte = |i: usize| data.get(i).copied().unwrap_or(0);
    let family = usize::from(byte(0)) % FAMILIES.len();
    let n = WIDTHS[usize::from(byte(1)) % WIDTHS.len()];
    let flags = byte(2);
    let cw = ceil_log2(n + 1);
    let w = match family {
        4 => {
            if flags & 8 == 0 {
                cw + 1
            } else {
                80
            }
        }
        5 => cw.saturating_sub(1).max(1),
        _ => cw,
    };
    let offset = if matches!(family, 3 | 5) {
        1 + usize::from(byte(3))
    } else {
        0
    };
    let operation = if matches!(family, 1 | 2) {
        format!("ext_prio_encode(input, lsb_prio={}, id=3)", family == 2)
    } else {
        format!("ext_clz(input, offset={offset}, new_bit_count={w}, id=3)")
    };
    let input_op = if flags & 4 == 0 {
        "identity"
    } else {
        "reverse"
    };
    let mut body =
        format!("  input: bits[{n}] = {input_op}(x, id=2)\n  count: bits[{w}] = {operation}\n");
    let value = if flags & 16 != 0 {
        body += &format!(
            "  one: bits[{w}] = literal(value=1, id=4)\n  adjusted: bits[{w}] = add(count, one, id=5)\n"
        );
        "adjusted"
    } else {
        "count"
    };
    let sharing = flags % 3;
    let (ty, result) = match sharing {
        1 => (
            format!("(bits[{w}], bits[{n}])"),
            format!("tuple({value}, x, id=6)"),
        ),
        2 => (
            format!("(bits[{w}], bits[{w}])"),
            format!("tuple({value}, {value}, id=6)"),
        ),
        _ => (format!("bits[{w}]"), format!("identity({value}, id=6)")),
    };
    let text = format!(
        "package sentinel\n\ntop fn main(x: bits[{n}] id=1) -> {ty} {{\n{body}  ret out: {ty} = {result}\n}}\n"
    );
    let package = Parser::new(&text)
        .parse_and_validate_package()
        .map_err(|e| e.to_string())?;
    let source = package.get_top_fn().ok_or("missing generated top")?;
    let mut basis = source.clone();
    desugar_extensions_in_fn(&mut basis).map_err(|e| e.to_string())?;
    let fold_hash = flags & 32 == 0;
    let gates = gatify(
        source,
        GatifyOptions {
            fold: fold_hash,
            hash: fold_hash,
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .map_err(|e| format!("sentinel lowering failed: {e}\n{text}"))?
    .gate_fn;
    let lifted =
        gate2ir::gate_fn_to_pir(&gates, "mapped", &source.get_type()).map_err(|e| e.to_string())?;
    let mapped = lifted.get_top_fn().ok_or("missing mapped top")?;
    let canonical = family < 3;
    let mut counts = Counts {
        samples: 1,
        canonical_lowerings: u64::from(canonical),
        direct_count_bits: if canonical {
            (0..cw).filter(|i| n & (1 << i) == 0).count() as u64
        } else {
            0
        },
        ..Default::default()
    };
    match prover_for_choice_with_limits(SolverChoice::Bitwuzla, None, fuzz_solver_limits())
        .prove_ir_fn_equiv(&basis, mapped)
    {
        EquivResult::Proved => counts.proofs += 1,
        EquivResult::Inconclusive(_) => {
            // A resource limit is not a proof or counterexample. The replay
            // audit rejects campaigns with any unresolved samples.
            counts.inconclusive += 1;
        }
        result => {
            return Err(format!(
                "sentinel proof failed: {result:?}\n{text}\nMapped:\n{mapped}"
            ));
        }
    }
    let zero = IrValue::make_ubits(n, 0).map_err(|e| e.to_string())?;
    if eval_fn(&basis, &[zero.clone()]) != eval_fn(mapped, &[zero]) {
        return Err(format!("zero-input sentinel mismatch\n{text}"));
    }
    counts.zero_input_checks += 1;
    let mut features = vec![
        format!("width_{n}"),
        ["unshared", "shared_input", "shared_result"][usize::from(sharing)].to_string(),
        if fold_hash {
            "fold_hash_on"
        } else {
            "fold_hash_off"
        }
        .to_string(),
    ];
    if flags & 16 != 0
        && w < usize::BITS as usize
        && (offset + n + 1) / (1usize << w) > offset / (1usize << w)
    {
        features.push("post_count_wrap".to_string());
    }
    if w > 64 {
        features.push("wide_output".to_string());
    }
    Ok(Coverage {
        total: counts,
        families: BTreeMap::from([(FAMILIES[family].to_string(), counts)]),
        features: features.into_iter().map(|f| (f, counts)).collect(),
    })
}

/// Reproducible branch/width matrix used by replay to prevent vacuity.
pub fn validation_inputs() -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    for family in 0..FAMILIES.len() {
        for width in 0..WIDTHS.len() {
            for flags in [0, 1, 2, 24, 33, 54] {
                result.push(vec![family as u8, width as u8, flags, width as u8]);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinel_matrix_requires_proved_canonical_and_fallback_lowerings() {
        assert!(Coverage::default().validate().is_err());
        let mut coverage = Coverage::default();
        for data in validation_inputs() {
            coverage
                .accumulate(check_input(&data).unwrap_or_else(|e| panic!("bytes={data:?}: {e}")));
        }
        coverage
            .validate()
            .unwrap_or_else(|e| panic!("{e}\n{coverage:#?}"));
        for c in coverage.families.values_mut() {
            c.proofs = 0;
        }
        assert!(
            coverage.validate().is_err(),
            "visiting labels alone must not pass"
        );
    }
}
