// SPDX-License-Identifier: Apache-2.0

//! Check temporary wiring edits against full STA on the independently
//! rebuilt candidate circuit.

use anyhow::{Result, ensure};

use super::Score;
use crate::liberty_model::Library;
use crate::netlist::sta::incremental::{IncrementalSta, TimingEndpoint};

/// Stable physical identity used by both the tentative and rebuilt graphs.
fn endpoint_id(endpoint: TimingEndpoint) -> (bool, usize) {
    match endpoint {
        TimingEndpoint::Output(bit) => (false, bit),
        TimingEndpoint::Capture(instance) => (true, instance),
    }
}

/// Compare the values visible to registered proposal ranking and electrical
/// guards, including endpoints which lost or gained a launch class. Recompute
/// trial pin violations from live timing; the ordinary cache holds the
/// incumbent's values until the transaction commits.
pub(super) fn compare_state(
    trial: &IncrementalSta<'_>,
    full: &IncrementalSta<'_>,
    library: &Library,
    objective: usize,
) -> Result<()> {
    ensure!(
        trial.cell_indices() == full.cell_indices(),
        "trial and full STA disagree on instance order or cell selection"
    );
    ensure!(
        trial.bit_count() == full.bit_count(),
        "normalized bit count: incremental {}, full {}",
        trial.bit_count(),
        full.bit_count()
    );
    for bit in 0..full.bit_count() {
        let found = trial.load_for_bit(bit).map(|edges| edges.map(f64::to_bits));
        let reference = full.load_for_bit(bit).map(|edges| edges.map(f64::to_bits));
        ensure!(
            found == reference,
            "bit {bit} capacitance: incremental {:?}, full {:?}; drivers {:?}/{:?}, sink counts {}/{}",
            trial.load_for_bit(bit),
            full.load_for_bit(bit),
            trial.driver_for_bit(bit),
            full.driver_for_bit(bit),
            trial.loads_for_bit(bit).len(),
            full.loads_for_bit(bit).len()
        );
    }
    let actual = Score::measure(trial, library, objective)?;
    let expected = Score::measure(full, library, objective)?;
    for (name, found, reference) in [
        ("area", actual.area, expected.area),
        ("objective delay", actual.delay, expected.delay),
        (
            "near-critical sum",
            actual.secondary.near_sum,
            expected.secondary.near_sum,
        ),
    ] {
        ensure!(
            found.to_bits() == reference.to_bits(),
            "{name}: incremental {found:?}, full {reference:?}"
        );
    }
    let delays = actual.boundaries.values();
    let reference_delays = expected.boundaries.values();
    for boundary in 0..delays.len() {
        let found = delays[boundary];
        let reference = reference_delays[boundary];
        ensure!(
            found.to_bits() == reference.to_bits(),
            "boundary {boundary} delay: incremental {found:?}, full {reference:?}"
        );
        ensure!(
            trial.endpoint_maximum_count(boundary) == full.endpoint_maximum_count(boundary),
            "boundary {boundary} has a different number of endpoints at the worst delay"
        );
    }

    let mut arrivals = trial.endpoint_values();
    for (reference_endpoint, reference_arrivals) in full.endpoint_values() {
        let reference_id = endpoint_id(reference_endpoint);
        let Some((endpoint, found)) = arrivals.next() else {
            anyhow::bail!("incremental STA is missing endpoint {reference_id:?}");
        };
        ensure!(
            endpoint_id(endpoint) == reference_id,
            "endpoint identity differs at full STA endpoint {reference_id:?}"
        );
        ensure!(
            found.map(|v| v.map(f64::to_bits)) == reference_arrivals.map(|v| v.map(f64::to_bits)),
            "endpoint {reference_id:?} arrivals: incremental {found:?}, full {reference_arrivals:?}; full register {:?}",
            match reference_endpoint {
                TimingEndpoint::Capture(instance) => full.instance(instance).map(|view| view.name),
                TimingEndpoint::Output(_) => None,
            }
        );
    }
    ensure!(
        arrivals.next().is_none(),
        "incremental STA has extra endpoints"
    );

    for instance in 0..full.instance_count() {
        let found = trial.electrical_violations(instance);
        let reference = full.cached_electrical_violations(instance);
        ensure!(
            found.len() == reference.len(),
            "instance {instance} has a different number of electrical pins"
        );
        for (ordinal, (a, b)) in found.iter().zip(reference).enumerate() {
            ensure!(
                a.capacitance.to_bits() == b.capacitance.to_bits()
                    && a.transition.to_bits() == b.transition.to_bits()
                    && a.fanout.to_bits() == b.fanout.to_bits(),
                "instance {instance} pin {ordinal} violations: incremental {a:?}, full {b:?}"
            );
        }
    }
    Ok(())
}

/// Verify that the next proposal starts from the original connectivity,
/// electrical cache and timing after a tentative edit was rolled back.
pub(super) fn compare_rollback(
    restored: &IncrementalSta<'_>,
    incumbent: &IncrementalSta<'_>,
    library: &Library,
    objective: usize,
) -> Result<()> {
    compare_state(restored, incumbent, library, objective)?;
    for bit in 0..incumbent.bit_count() {
        ensure!(
            restored.canonical_source(bit) == incumbent.canonical_source(bit)
                && restored.driver_for_bit(bit) == incumbent.driver_for_bit(bit)
                && restored.loads_for_bit(bit) == incumbent.loads_for_bit(bit)
                && restored.output_load_count(bit) == incumbent.output_load_count(bit),
            "bit {bit} connectivity differs after rollback"
        );
    }
    for instance in 0..incumbent.instance_count() {
        ensure!(
            restored.cached_electrical_violations(instance)
                == incumbent.cached_electrical_violations(instance),
            "instance {instance} retains stale electrical violations after rollback"
        );
    }
    Ok(())
}
