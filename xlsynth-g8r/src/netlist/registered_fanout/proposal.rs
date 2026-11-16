// SPDX-License-Identifier: Apache-2.0

//! One registered wiring proposal interpreted by the timing engine or gate
//! emitter.

use super::OwnedBuffers;
use super::edit::{self, Branch};
use super::graph::{Buffer, Root};
use crate::liberty_model::Library;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::normalized::BitSource;
use crate::netlist::sta::incremental::{
    IncrementalSta, TimingEdit, TimingInstance, TimingMutation, UnsupportedRegisteredEdit,
};
use anyhow::{Result, anyhow};

/// Borrow the generator's exact plan until the mapper selects or rejects it.
/// The same plan drives speculative STA and Verilog emission. A winning
/// edit must create exactly the cells and pin bindings that were timed.
pub(crate) struct TopologyProposal<'a> {
    roots: &'a [Root],
    buffer: &'a Buffer,
    trees: &'a [(usize, Vec<Branch>)],
}

impl<'a> TopologyProposal<'a> {
    pub(super) fn forest(
        roots: &'a [Root],
        buffer: &'a Buffer,
        trees: &'a [(usize, Vec<Branch>)],
    ) -> Self {
        Self {
            roots,
            buffer,
            trees,
        }
    }

    /// Apply the original gate-level editor only when a lasting netlist or an
    /// independent timing calculation is needed.
    pub(crate) fn materialize(
        &self,
        original: &MappedNetlist,
        library: &Library,
    ) -> Result<MappedNetlist> {
        let mut candidate = original.clone();
        edit::apply_forest(&mut candidate, library, self.buffer, self.roots, self.trees)?;
        Ok(candidate)
    }

    /// Translate exactly the inserted gates and sink pins. Existing bits use
    /// STA's canonical sources, including vector selects and aliases. Every
    /// appended wire is scalar and follows the existing normalized bits.
    pub(crate) fn timing_edit(
        &self,
        original: &MappedNetlist,
        library: &Library,
        sta: &IncrementalSta<'_>,
    ) -> Result<TimingEdit> {
        // The physical editor inserts nets at the end of the module range.
        // A later module's declarations would shift existing normalized bit
        // IDs.
        if original.module.net_index_range.end != original.nets.len() {
            return Err(UnsupportedRegisteredEdit.into());
        }
        let mut edit = TimingEdit::default();
        let mut ordinal = original.module.instances.len();
        let first_bit = sta.bit_count();
        let Self {
            roots,
            buffer,
            trees,
        } = *self;
        let master = &library.cells[buffer.cell].name;
        for &(i, ref branches) in trees {
            let root = &roots[i];
            let driver = sta
                .instance(root.instance)
                .ok_or_else(|| anyhow!("unknown forest driver"))?;
            let Some(&[source]) = driver.pins.get(&root.pin_name).map(Vec::as_slice) else {
                anyhow::bail!("forest driver requires one connected output bit");
            };
            let tree_first = first_bit + edit.new_bits;
            for branch in branches {
                // The materializer interns the master and its pin names
                // before requesting the first generated wire and gate.
                let (_, name) = edit::next_names(&mut ordinal, |name| {
                    original.interner.get(name).is_some()
                        || name == master
                        || name == buffer.input
                        || name == buffer.output
                });
                let output = BitSource::Bit(first_bit + edit.new_bits);
                let input = branch
                    .parent
                    .map_or(source, |parent| BitSource::Bit(tree_first + parent));
                edit.mutations.push(TimingMutation::Insert(TimingInstance {
                    name,
                    cell: buffer.cell,
                    connections: vec![
                        (buffer.input.clone(), input),
                        (buffer.output.clone(), output),
                    ],
                }));
                edit.new_bits += 1;
                rewire_sinks(&mut edit, original, root, &branch.sinks, output)?;
            }
        }
        Ok(edit)
    }

    /// Record ownership of the exact appended instances in a selected netlist.
    pub(crate) fn record(&self, trial: &MappedNetlist, first: usize, owned: &mut OwnedBuffers) {
        let mut start = first;
        for (i, branches) in self.trees {
            let end = start + branches.len();
            owned.record(trial, &self.roots[*i], start..end);
            start = end;
        }
        debug_assert_eq!(start, trial.module.instances.len());
    }
}

/// Retain the emitter's connection identities even when two taps belong to
/// separate pins of the same receiving gate.
fn rewire_sinks(
    edit: &mut TimingEdit,
    original: &MappedNetlist,
    root: &Root,
    sink_ids: &[usize],
    source: BitSource,
) -> Result<()> {
    for &id in sink_ids {
        let sink = &root.sinks[id];
        let (name, _) = &original.module.instances[sink.instance].connections[sink.connection];
        let pin = original
            .interner
            .resolve(*name)
            .ok_or_else(|| anyhow!("unresolved receiver pin"))?;
        edit.mutations.push(TimingMutation::Rewire {
            instance: sink.instance,
            pin: pin.to_owned(),
            source,
        });
    }
    Ok(())
}
