// SPDX-License-Identifier: Apache-2.0

//! Validate scalar match patterns against the complete concrete subject type.

use std::collections::{BTreeMap, HashSet};

use num_bigint::BigUint;
use xlsynth_pir::range_analysis::IntervalSet;
use xlsynth_pir::{IrBits, IrValue};

use crate::ast::{Expr, MatchArm, MatchPattern, Type};
use crate::infer_literal::check_inferred_signed_width;
use crate::typed::{CheckedEnum, TypedExpr, TypedExprKind, TypedMatchPattern, bits};

/// Disjoint, half-open gaps in the unsigned bit-pattern space, plus the
/// nonempty source intervals needed to distinguish exact duplicates.
struct Coverage {
    remaining: BTreeMap<BigUint, BigUint>,
    seen: HashSet<(IrBits, IrBits)>,
    saw_wildcard: bool,
}

impl Coverage {
    fn new(width: usize) -> Self {
        Self {
            remaining: BTreeMap::from([(BigUint::from(0u8), BigUint::from(1u8) << width)]),
            seen: HashSet::new(),
            saw_wildcard: false,
        }
    }

    /// Remove [lower, limit) from the gaps, visiting only intersecting entries.
    /// A list of disjoint literal patterns thus costs O(n log n) in total.
    fn remove(&mut self, lower: BigUint, limit: BigUint) -> bool {
        let first = self
            .remaining
            .range(..=lower.clone())
            .next_back()
            .filter(|(_, end)| **end > lower)
            .map(|(start, _)| start.clone())
            .unwrap_or_else(|| lower.clone());
        let starts: Vec<_> = self
            .remaining
            .range(first..limit.clone())
            .map(|(start, _)| start.clone())
            .collect();
        let adds_coverage = !starts.is_empty();
        for start in starts {
            let end = self.remaining.remove(&start).expect("existing gap");
            if start < lower {
                self.remaining.insert(start, lower.clone());
            }
            if end > limit {
                self.remaining.insert(limit.clone(), end);
            }
        }
        adds_coverage
    }

    /// Add an inclusive interval. A signed range crossing zero becomes two
    /// unsigned bit-pattern intervals; neither endpoints nor cardinalities
    /// need to fit a host integer.
    fn add_interval(&mut self, lower: &IrBits, upper: &IrBits) -> Result<(), String> {
        if !self.seen.insert((lower.clone(), upper.clone())) {
            return Err("exact-duplicate match pattern".into());
        }
        if self.remaining.is_empty() {
            // DSLX only warns for distinct patterns after exhaustiveness.
            return Ok(());
        }
        let set =
            IntervalSet::from_intervals(lower.get_bit_count(), [(lower.clone(), upper.clone())])
                .map_err(|error| error.to_string())?;
        let mut adds_coverage = false;
        for interval in set.intervals() {
            let lower = BigUint::from_bytes_le(&interval.lower().to_le_bytes());
            let limit = BigUint::from_bytes_le(&interval.upper().to_le_bytes()) + 1u8;
            adds_coverage |= self.remove(lower, limit);
        }
        if !adds_coverage {
            return Err("match pattern is fully covered by previous patterns".into());
        }
        Ok(())
    }
}

/// Consume a literal checked in the same type group as the subject.
fn next_literal(
    source: &Expr,
    literals: &mut impl Iterator<Item = TypedExpr>,
    subject_ty: &Type,
) -> Result<IrBits, String> {
    check_inferred_signed_width(&[source], subject_ty, None)?;
    let value = literals.next().ok_or("missing checked pattern literal")?;
    pattern_bits(value, subject_ty)
}

/// Extract a checked literal at the type of the value it will match.
pub(super) fn pattern_bits(value: TypedExpr, subject_ty: &Type) -> Result<IrBits, String> {
    if !super::compatible(&value.ty, subject_ty) {
        return Err(format!(
            "match pattern has type {}, expected {subject_ty}",
            value.ty
        ));
    }
    match value.kind {
        TypedExprKind::Literal(IrValue::Bits(value)) => Ok(value),
        _ => Err("scalar match patterns require literal values".into()),
    }
}

/// Empty ranges never select their arm or create a dependency on its result.
pub(super) fn nonempty_range(start: &IrBits, end: &IrBits, inclusive: bool, signed: bool) -> bool {
    (inclusive && start == end)
        || if signed {
            start.slt(end)
        } else {
            start.ult(end)
        }
}

/// Check nominal alternatives against the declared encodings. XLS IR uses
/// the last source arm as its default even if an earlier arm exhausts the
/// enum; this is observable when hardware supplies an undeclared encoding.
pub(super) fn check_enum_patterns(
    def: &CheckedEnum,
    subject_ty: &Type,
    arms: &[MatchArm],
    mut literals: impl Iterator<Item = TypedExpr>,
) -> Result<Vec<Vec<TypedMatchPattern>>, String> {
    let mut seen = HashSet::new();
    let mut saw_wildcard = false;
    let mut checked = Vec::with_capacity(arms.len());
    for (index, arm) in arms.iter().enumerate() {
        let mut patterns = Vec::with_capacity(arm.patterns.len());
        let mut unconditional = index + 1 == arms.len();
        for pattern in &arm.patterns {
            match pattern {
                MatchPattern::Wildcard => {
                    if saw_wildcard {
                        return Err("exact-duplicate wildcard match pattern".into());
                    }
                    saw_wildcard = true;
                    unconditional = true;
                }
                MatchPattern::Literal(source @ Expr::Name(_)) => {
                    let value = next_literal(source, &mut literals, subject_ty)?;
                    if !def.contains(&value) {
                        return Err(format!(
                            "enum match pattern is not a declared value of {subject_ty}"
                        ));
                    }
                    if !seen.insert(value.clone()) {
                        return Err("exact-duplicate match pattern".into());
                    }
                    patterns.push(TypedMatchPattern::Literal(value));
                }
                MatchPattern::Literal(_) | MatchPattern::Range { .. } | MatchPattern::Tuple(_) => {
                    return Err(format!(
                        "enum match pattern must name a value of {subject_ty}"
                    ));
                }
            }
        }
        checked.push(if unconditional {
            vec![TypedMatchPattern::Wildcard]
        } else {
            patterns
        });
    }
    if !saw_wildcard && seen.len() != def.value_count() {
        return Err("match patterns are not exhaustive".into());
    }
    Ok(checked)
}

/// Check all source alternatives, including ones after coverage becomes
/// exhaustive. The first arm to complete coverage is a valid default whenever
/// earlier arms did not match, so lower it as an unconditional arm.
pub(super) fn check_patterns(
    subject_ty: &Type,
    arms: &[MatchArm],
    mut literals: impl Iterator<Item = TypedExpr>,
) -> Result<Vec<Vec<TypedMatchPattern>>, String> {
    let (signed, width) = bits(subject_ty)
        .ok_or_else(|| format!("scalar match requires a bits or bool subject, got {subject_ty}"))?;
    let mut coverage = Coverage::new(width);
    let mut checked = Vec::with_capacity(arms.len());
    for arm in arms {
        let was_exhaustive = coverage.remaining.is_empty();
        let mut patterns = Vec::with_capacity(arm.patterns.len());
        for pattern in &arm.patterns {
            patterns.push(match pattern {
                MatchPattern::Wildcard => {
                    if coverage.saw_wildcard {
                        return Err("exact-duplicate wildcard match pattern".into());
                    }
                    coverage.saw_wildcard = true;
                    coverage.remaining.clear();
                    TypedMatchPattern::Wildcard
                }
                MatchPattern::Literal(source) => {
                    let value = next_literal(source, &mut literals, subject_ty)?;
                    coverage.add_interval(&value, &value)?;
                    TypedMatchPattern::Literal(value)
                }
                MatchPattern::Range {
                    start,
                    end,
                    inclusive,
                } => {
                    let start = next_literal(start, &mut literals, subject_ty)?;
                    let end = next_literal(end, &mut literals, subject_ty)?;
                    if !nonempty_range(&start, &end, *inclusive, signed) {
                        continue;
                    }
                    if *inclusive {
                        coverage.add_interval(&start, &end)?;
                    } else {
                        let one = IrBits::make_ubits(width, 1).map_err(|e| e.to_string())?;
                        coverage.add_interval(&start, &end.sub(&one))?;
                    }
                    TypedMatchPattern::Range {
                        start,
                        end,
                        inclusive: *inclusive,
                    }
                }
                MatchPattern::Tuple(_) => {
                    return Err("tuple pattern requires a tuple subject".into());
                }
            });
        }
        if !was_exhaustive && coverage.remaining.is_empty() {
            patterns = vec![TypedMatchPattern::Wildcard];
        }
        checked.push(patterns);
    }
    if !coverage.remaining.is_empty() {
        return Err("match patterns are not exhaustive".into());
    }
    Ok(checked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_coverage_matches_an_enumerated_domain() {
        // Split gaps, span several earlier intervals, wrap through zero, and
        // revisit ranges both before and after reaching exhaustiveness.
        let ranges = [
            (0, 0),
            (3, 3),
            (5, 5),
            (2, 4),
            (1, 3),
            (3, 4),
            (0, 2),
            (13, 1),
            (5, 14),
            (15, 15),
            (0, 15),
            (7, 10),
            (5, 14),
            (0, 0),
        ];
        for offset in 0..16 {
            let mut coverage = Coverage::new(4);
            let mut seen = HashSet::new();
            let mut covered = 0u16;
            for (start, end) in ranges {
                let start = (start + offset) % 16;
                let end = (end + offset) % 16;
                let mut mask = 0u16;
                for value in 0..16 {
                    let matched = if start <= end {
                        (start..=end).contains(&value)
                    } else {
                        value >= start || value <= end
                    };
                    if matched {
                        mask |= 1 << value;
                    }
                }
                let accepted =
                    seen.insert((start, end)) && (covered == u16::MAX || mask & !covered != 0);
                let result = coverage.add_interval(
                    &IrBits::make_ubits(4, start).unwrap(),
                    &IrBits::make_ubits(4, end).unwrap(),
                );
                assert_eq!(result.is_ok(), accepted, "{start}..={end}: {result:?}");
                covered |= mask;
                assert_eq!(coverage.remaining.is_empty(), covered == u16::MAX);
            }
        }
    }
}
