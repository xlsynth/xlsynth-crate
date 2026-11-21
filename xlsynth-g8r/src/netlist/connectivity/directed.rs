// SPDX-License-Identifier: Apache-2.0

//! Resolve zero-delay directed assignments without merging electrical groups.
//!
//! An assignment transfers the value of its source to its destination. The
//! reverse direction is not implied; literals and unknown values remain
//! distinguishable from an unassigned bit. Clients supply their own diagnostics
//! for a cycle so their validation contracts stay unchanged.

/// One assignment to a bit; `None` in the input means the bit is unassigned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AssignedSource {
    Literal(bool),
    Unknown,
    Alias(usize),
    /// A terminal bit outside a compact assignment index.
    Root(usize),
}

/// Terminal source after following directed assignments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedSource {
    Literal(bool),
    Unknown,
    Bit(usize),
}

#[derive(Clone, Copy)]
enum Visit {
    Unvisited,
    Visiting,
    Resolved(ResolvedSource),
}

/// Resolve every bit in linear time, returning a bit on the first cycle.
/// Reuse one path buffer and handle unassigned roots without allocating a path.
pub(crate) fn resolve(sources: &[Option<AssignedSource>]) -> Result<Vec<ResolvedSource>, usize> {
    if sources.iter().all(Option::is_none) {
        return Ok((0..sources.len()).map(ResolvedSource::Bit).collect());
    }
    let mut states = vec![Visit::Unvisited; sources.len()];
    let mut path = Vec::new();
    for start in 0..sources.len() {
        if matches!(states[start], Visit::Resolved(_)) {
            continue;
        }
        if sources[start].is_none() {
            states[start] = Visit::Resolved(ResolvedSource::Bit(start));
            continue;
        }
        path.clear();
        let mut current = start;
        let terminal = loop {
            match states[current] {
                Visit::Resolved(source) => break source,
                Visit::Visiting => return Err(current),
                Visit::Unvisited => {
                    states[current] = Visit::Visiting;
                    path.push(current);
                    match sources[current] {
                        Some(AssignedSource::Literal(value)) => {
                            break ResolvedSource::Literal(value);
                        }
                        Some(AssignedSource::Unknown) => break ResolvedSource::Unknown,
                        Some(AssignedSource::Alias(next)) => current = next,
                        Some(AssignedSource::Root(bit)) => break ResolvedSource::Bit(bit),
                        None => break ResolvedSource::Bit(current),
                    }
                }
            }
        };
        for &bit in &path {
            states[bit] = Visit::Resolved(terminal);
        }
    }
    Ok(states
        .into_iter()
        .map(|state| match state {
            Visit::Resolved(source) => source,
            Visit::Unvisited | Visit::Visiting => unreachable!("all assignment paths are resolved"),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Directed chains terminate at a bit, literal, or unknown without
    /// turning the destination into a driver of its source.
    #[test]
    fn chains_keep_direction_and_terminal_kind() {
        use AssignedSource::{Alias, Literal, Unknown};
        assert_eq!(
            resolve(&[
                None,
                Some(Alias(0)),
                Some(Alias(1)),
                Some(Literal(true)),
                Some(Alias(3)),
                Some(Unknown)
            ]),
            Ok(vec![
                ResolvedSource::Bit(0),
                ResolvedSource::Bit(0),
                ResolvedSource::Bit(0),
                ResolvedSource::Literal(true),
                ResolvedSource::Literal(true),
                ResolvedSource::Unknown
            ])
        );
        assert_eq!(resolve(&[Some(Alias(1)), Some(Alias(0))]), Err(0));
        assert_eq!(
            resolve(&[Some(Alias(1)), Some(AssignedSource::Root(9000))]),
            Ok(vec![ResolvedSource::Bit(9000); 2])
        );
    }

    /// A long forward chain must be resolved iteratively, including a terminal
    /// outside the compact index. An unrelated cycle must still be diagnosed.
    #[test]
    fn deep_forward_chain_and_disconnected_cycle() {
        const BITS: usize = 32_768;
        let mut sources: Vec<_> = (1..BITS).map(|i| Some(AssignedSource::Alias(i))).collect();
        sources.push(Some(AssignedSource::Root(1_000_000)));
        assert!(
            resolve(&sources)
                .unwrap()
                .iter()
                .all(|&v| v == ResolvedSource::Bit(1_000_000))
        );
        sources.extend([
            Some(AssignedSource::Alias(BITS + 1)),
            Some(AssignedSource::Alias(BITS)),
        ]);
        assert_eq!(resolve(&sources), Err(BITS));
    }
}
