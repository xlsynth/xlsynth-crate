// SPDX-License-Identifier: Apache-2.0

//! Check the field types and Cartesian coverage of tuple match patterns.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use num_bigint::BigUint;
use xlsynth_pir::IrBits;

use super::match_patterns::{nonempty_range, pattern_bits};
use super::{BodyChecker, MAX_TYPE_SIZE, MatchSelector, StructTable};
use crate::ast::{Expr, MatchArm, MatchPattern, Type};
use crate::infer_literal::{check_inferred_signed_width, has_concrete_element_type};
use crate::typed::{TypedExpr, TypedMatchPattern, bits};

type Region = Vec<Range<BigUint>>;
type EnumOrdinals = HashMap<String, Option<HashMap<IrBits, usize>>>;

/// The still-unmatched regions in the product of the subject's field domains.
struct Coverage {
    domain: Region,
    remaining: Vec<Region>,
    seen: HashSet<Region>,
    words: usize,
    budget: usize,
}

impl Coverage {
    fn new(domain: Region, budget: usize) -> Self {
        let words = 1 + domain
            .iter()
            .map(|axis| 1 + axis.end.bits().div_ceil(64) as usize)
            .sum::<usize>();
        let remaining = if domain.iter().any(Range::is_empty) {
            Vec::new()
        } else {
            vec![domain.clone()]
        };
        Self {
            remaining,
            domain,
            seen: HashSet::new(),
            words,
            budget,
        }
    }

    fn charge(&mut self) -> Result<(), String> {
        self.budget = self
            .budget
            .checked_sub(self.words)
            .ok_or("tuple match coverage exceeds the elaboration limit")?;
        Ok(())
    }

    /// Subtract a pattern from each intersecting gap. On each axis, save the
    /// slabs outside the pattern, then continue splitting only the overlap.
    fn add(&mut self, region: Region) -> Result<(), String> {
        if !self.seen.insert(region.clone()) {
            return Err("exact-duplicate match pattern".into());
        }
        if region == self.domain {
            self.remaining.clear();
        }
        if self.remaining.is_empty() {
            // DSLX only warns about distinct patterns after exhaustiveness.
            return Ok(());
        }
        let mut adds_coverage = false;
        for mut gap in std::mem::take(&mut self.remaining) {
            self.charge()?;
            if gap
                .iter()
                .zip(&region)
                .any(|(a, b)| a.end <= b.start || b.end <= a.start)
            {
                self.remaining.push(gap);
                continue;
            }
            adds_coverage = true;
            for (i, axis) in region.iter().enumerate() {
                if gap[i].start < axis.start {
                    self.charge()?;
                    let mut slab = gap.clone();
                    slab[i].end.clone_from(&axis.start);
                    self.remaining.push(slab);
                    gap[i].start.clone_from(&axis.start);
                }
                if gap[i].end > axis.end {
                    self.charge()?;
                    let mut slab = gap.clone();
                    slab[i].start.clone_from(&axis.end);
                    self.remaining.push(slab);
                    gap[i].end.clone_from(&axis.end);
                }
            }
        }
        if !adds_coverage {
            return Err("match pattern is fully covered by previous patterns".into());
        }
        Ok(())
    }
}

/// Check whether a wildcard-only field has any values. Visit each array's
/// element type once and bound the expanded walk through named struct fields.
fn opaque_domain_is_empty(
    ty: &Type,
    structs: &StructTable<'_>,
    budget: &mut usize,
) -> Result<bool, String> {
    let mut pending = vec![ty];
    while let Some(ty) = pending.pop() {
        *budget = budget
            .checked_sub(1)
            .ok_or("tuple match coverage exceeds the elaboration limit")?;
        match ty {
            Type::Tuple(fields) => pending.extend(fields.iter().rev()),
            Type::Named(name) => pending.extend(
                structs
                    .definition(name)?
                    .fields
                    .iter()
                    .rev()
                    .map(|field| &field.ty),
            ),
            Type::Array { size: 1.., element } => pending.push(element),
            Type::Enum { name, .. } => {
                let def = structs
                    .enums
                    .get(name)
                    .ok_or_else(|| format!("unknown enum '{name}'"))?;
                if def.value_count() == 0 {
                    return Ok(true);
                }
            }
            _ => { /* Bits and zero-length arrays each have at least one value. */ }
        }
    }
    Ok(false)
}

/// Flatten nested tuple fields in type order. Signed fields use their signed
/// order so each range stays contiguous. Enum fields use dense ordinals of
/// their declared values. When constructing the subject domain (`pattern` is
/// `None`), an empty wildcard-only field contributes an empty axis.
fn region_for(
    ty: &Type,
    pattern: Option<&TypedMatchPattern>,
    structs: &StructTable<'_>,
    enums: &mut EnumOrdinals,
    region: &mut Region,
    budget: &mut usize,
) -> Result<(), String> {
    *budget = budget
        .checked_sub(1)
        .ok_or("tuple match coverage exceeds the elaboration limit")?;
    if let Type::Tuple(types) = ty {
        for (i, ty) in types.iter().enumerate() {
            let field = pattern.map(|pattern| match pattern {
                TypedMatchPattern::Tuple(fields) => &fields[i],
                _ => &TypedMatchPattern::Wildcard,
            });
            region_for(ty, field, structs, enums, region, budget)?;
        }
    } else if let Type::Enum { name, .. } = ty {
        let def = structs
            .enums
            .get(name)
            .ok_or_else(|| format!("unknown enum '{name}'"))?;
        let ordinals = enums.entry(name.clone()).or_default();
        region.push(match pattern {
            Some(TypedMatchPattern::Literal(value)) => {
                if ordinals.is_none() {
                    let cost = def
                        .width
                        .div_ceil(64)
                        .max(1)
                        .saturating_mul(def.members.len());
                    *budget = budget
                        .checked_sub(cost)
                        .ok_or("tuple match coverage exceeds the elaboration limit")?;
                }
                let ordinals = ordinals.get_or_insert_with(|| {
                    let mut ordinals = HashMap::new();
                    for value in def.members.values() {
                        let index = ordinals.len();
                        ordinals.entry(value.clone()).or_insert(index);
                    }
                    ordinals
                });
                let index = ordinals[value];
                BigUint::from(index)..BigUint::from(index + 1)
            }
            _ => BigUint::ZERO..BigUint::from(def.value_count()),
        });
    } else if let Some((signed, width)) = bits(ty) {
        let ordinal = |value: &IrBits| {
            let encoded = BigUint::from_bytes_le(&value.to_le_bytes());
            if signed && width > 0 {
                encoded ^ (BigUint::from(1u8) << (width - 1))
            } else {
                encoded
            }
        };
        region.push(match pattern {
            Some(TypedMatchPattern::Literal(value)) => {
                let start = ordinal(value);
                let end = &start + 1u8;
                start..end
            }
            Some(TypedMatchPattern::Range {
                start,
                end,
                inclusive,
            }) => ordinal(start)..ordinal(end) + u8::from(*inclusive),
            _ => BigUint::ZERO..(BigUint::from(1u8) << width),
        });
    } else if pattern.is_none() && opaque_domain_is_empty(ty, structs, budget)? {
        region.push(BigUint::ZERO..BigUint::ZERO);
    }
    Ok(())
}

impl BodyChecker<'_, '_> {
    pub(super) fn tuple_match_selector(
        &mut self,
        subject: &Expr,
        arms: &[MatchArm],
    ) -> Result<MatchSelector, String> {
        let patterns: Vec<_> = arms.iter().flat_map(|arm| &arm.patterns).collect();
        let subject = self.tuple_match_subjects(&[subject], &patterns)?.remove(0);
        self.structs.validate_type(&subject.ty)?;
        self.finish_tuple_match(subject, arms)
    }

    /// Share each pattern's field hints across the subject's tuple constructors
    /// and branch results.
    fn tuple_match_subjects(
        &mut self,
        items: &[&Expr],
        patterns: &[&MatchPattern],
    ) -> Result<Vec<TypedExpr>, String> {
        let tuples: Vec<_> = patterns
            .iter()
            .filter_map(|pattern| match pattern {
                MatchPattern::Tuple(fields) => Some(fields),
                _ => None,
            })
            .collect();
        if !tuples.is_empty() {
            // Preserve the ordinary result group when an operand already
            // determines its type, including the boundaries of nested matches.
            for item in items {
                if !self.contains_inferred_aggregate(item)
                    && has_concrete_element_type(item, self.common_widths())?
                {
                    return self.check_results(items, None);
                }
            }
            if items
                .iter()
                .any(|item| matches!(item, Expr::If { .. } | Expr::Match { .. }))
            {
                return self.check_result_branches(items, |checker, branches| {
                    checker.tuple_match_subjects(branches, patterns)
                });
            }
            if let Some(subjects) = items
                .iter()
                .map(|item| match item {
                    Expr::Tuple(fields) => Some(fields.as_slice()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
            {
                let arity = subjects[0].len();
                for fields in &tuples {
                    check_arity(arity, fields.len())?;
                }
                for fields in &subjects[1..] {
                    check_arity(fields.len(), arity)?;
                }
                return self.check_tuple_results(&subjects, |checker, index, values| {
                    let fields: Vec<_> = tuples.iter().map(|fields| &fields[index]).collect();
                    checker.tuple_match_subjects(values, &fields)
                });
            }
            return self.check_results(items, None);
        }

        for item in items {
            if has_concrete_element_type(item, self.common_widths())? {
                return if let [subject] = items {
                    Ok(vec![self.expr(subject, None)?])
                } else {
                    self.check_results(items, None)
                };
            }
        }
        let mut operands = items.to_vec();
        operands.extend(patterns.iter().flat_map(|pattern| pattern.literals()));
        let mut values = self.check_results(&operands, None)?;
        values.truncate(items.len());
        Ok(values)
    }

    pub(super) fn finish_tuple_match(
        &mut self,
        subject: TypedExpr,
        arms: &[MatchArm],
    ) -> Result<MatchSelector, String> {
        if !matches!(subject.ty, Type::Tuple(_)) {
            return Err(format!(
                "tuple pattern requires a tuple subject, got {}",
                subject.ty
            ));
        }
        let mut enums = HashMap::new();
        let mut domain = Vec::new();
        let mut budget = MAX_TYPE_SIZE;
        region_for(
            &subject.ty,
            None,
            self.structs,
            &mut enums,
            &mut domain,
            &mut budget,
        )?;
        let mut coverage = Coverage::new(domain, budget);
        let empty_domain = coverage.remaining.is_empty();
        let mut seen_irrefutable = HashSet::new();
        let mut patterns = Vec::with_capacity(arms.len());
        for (index, arm) in arms.iter().enumerate() {
            let was_exhaustive = coverage.remaining.is_empty();
            let mut checked = Vec::new();
            for source in &arm.patterns {
                let Some(pattern) = self.check_tuple_pattern(source, &subject.ty)? else {
                    // Empty patterns have no region to subtract.
                    continue;
                };
                if pattern.is_irrefutable() {
                    if !seen_irrefutable.insert(source) {
                        return Err("exact-duplicate match pattern".into());
                    }
                    coverage.remaining.clear();
                } else if !empty_domain {
                    // In an empty domain, duplicate checking uses source
                    // spelling, which the parser has already compared.
                    coverage.charge()?;
                    let mut region = Vec::new();
                    region_for(
                        &subject.ty,
                        Some(&pattern),
                        self.structs,
                        &mut enums,
                        &mut region,
                        &mut coverage.budget,
                    )?;
                    coverage.add(region)?;
                }
                checked.push(pattern);
            }
            // Preserve DSLX's final-arm default for undeclared enum encodings.
            let is_default = if empty_domain || !enums.is_empty() {
                index + 1 == arms.len()
            } else {
                !was_exhaustive && coverage.remaining.is_empty()
            };
            if is_default {
                checked = vec![TypedMatchPattern::Wildcard];
            }
            patterns.push(checked);
        }
        if !coverage.remaining.is_empty() {
            return Err("match patterns are not exhaustive".into());
        }
        Ok(MatchSelector { subject, patterns })
    }

    fn check_tuple_pattern(
        &mut self,
        pattern: &MatchPattern,
        ty: &Type,
    ) -> Result<Option<TypedMatchPattern>, String> {
        Ok(Some(match pattern {
            MatchPattern::Wildcard => TypedMatchPattern::Wildcard,
            MatchPattern::Tuple(fields) => {
                let Type::Tuple(types) = ty else {
                    return Err(format!("tuple pattern requires a tuple subject, got {ty}"));
                };
                check_arity(types.len(), fields.len())?;
                let fields = fields
                    .iter()
                    .zip(types)
                    .map(|(field, ty)| self.check_tuple_pattern(field, ty))
                    .collect::<Result<Vec<_>, _>>()?;
                let Some(fields) = fields.into_iter().collect() else {
                    // A tuple with an empty range in any member never matches.
                    return Ok(None);
                };
                TypedMatchPattern::Tuple(fields)
            }
            MatchPattern::Literal(expr) => {
                if matches!(ty, Type::Enum { .. }) && !matches!(expr, Expr::Name(_)) {
                    return Err(format!("enum match pattern must name a value of {ty}"));
                }
                let value = self.tuple_pattern_literal(expr, ty)?;
                if let Type::Enum { name, .. } = ty
                    && !self.structs.enums[name].contains(&value)
                {
                    return Err(format!(
                        "enum match pattern is not a declared value of {ty}"
                    ));
                }
                TypedMatchPattern::Literal(value)
            }
            MatchPattern::Range {
                start,
                end,
                inclusive,
            } => {
                let (signed, _) = bits(ty).ok_or_else(|| {
                    format!("match range requires a bits or bool field, got {ty}")
                })?;
                let start = self.tuple_pattern_literal(start, ty)?;
                let end = self.tuple_pattern_literal(end, ty)?;
                if !nonempty_range(&start, &end, *inclusive, signed) {
                    return Ok(None);
                }
                TypedMatchPattern::Range {
                    start,
                    end,
                    inclusive: *inclusive,
                }
            }
        }))
    }

    fn tuple_pattern_literal(&mut self, expr: &Expr, ty: &Type) -> Result<IrBits, String> {
        check_inferred_signed_width(&[expr], ty, None)?;
        pattern_bits(self.expr(expr, Some(ty))?, ty)
    }
}

fn check_arity(expected: usize, actual: usize) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "tuple match pattern has {actual} fields, expected {expected}"
        ));
    }
    Ok(())
}
