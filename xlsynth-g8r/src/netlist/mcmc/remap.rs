// SPDX-License-Identifier: Apache-2.0

//! Exact-truth, fanout-safe mapped-cone replacement proposals for MCMC.

use super::{Connectivity, NetlistMcmcObjective, SearchState, bit_reference, instance_name};
use crate::liberty_model::Library;
use crate::netlist::cell_catalog::{CatalogCell, CellCatalog};
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistInstance};
use crate::netlist::timing_resize::SearchIncrementalSta;
use anyhow::{Result, anyhow};
use rand::seq::SliceRandom;
use rand::{Rng, RngCore};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

/// Bounds equivalent physical pin assignments retained for one cell/function.
const MAX_BINDINGS_PER_CELL_FUNCTION: usize = 6;
/// Keeps local decomposition enumeration bounded on large Liberty libraries.
const MAX_DECOMPOSITIONS_PER_FUNCTION: usize = 96;
/// Avoids replacing an unexpectedly large externally visible logic region.
const MAX_WINDOW_INSTANCES: usize = 8;
/// Retries biased root selection before declaring a remap move unavailable.
const MAX_ROOT_SELECTION_ATTEMPTS: usize = 16;

/// One legal Liberty implementation over ordered external Boolean variables.
#[derive(Clone, Debug)]
struct RemapBinding {
    cell_index: usize,
    input_to_leaf: Vec<usize>,
    area: f64,
    nominal_delay: f64,
}

/// One exact two-cell functional decomposition of a bounded root truth table.
#[derive(Clone, Debug)]
struct TwoCellDecomposition {
    child_leaves: Vec<usize>,
    direct_leaves: Vec<usize>,
    child_truth: u64,
    root_truth: u64,
    estimated_area: f64,
    estimated_delay: f64,
}

/// Function-complete cell index shared by every parallel mapped search chain.
pub(super) struct RemapLibrary {
    bindings: BTreeMap<(usize, u64), Vec<RemapBinding>>,
    signatures: Vec<Vec<u64>>,
    decompositions: Mutex<BTreeMap<(usize, u64), Arc<Vec<TwoCellDecomposition>>>>,
}

impl RemapLibrary {
    /// Indexes all characterized cells while preserving useful pin assignments.
    pub(super) fn new(catalog: &CellCatalog, max_leaves: usize) -> Self {
        let permutations = (0..=max_leaves).map(permutations).collect::<Vec<_>>();
        let mut bindings = BTreeMap::<(usize, u64), Vec<RemapBinding>>::new();
        for cell in catalog.cells() {
            let input_count = cell.family.input_names.len();
            if input_count == 0 || input_count > max_leaves {
                continue;
            }
            let mut retained_by_truth = BTreeMap::<u64, usize>::new();
            for permutation in &permutations[input_count] {
                let truth = permuted_truth(cell.family.truth, permutation.as_slice());
                let retained = retained_by_truth.entry(truth).or_default();
                if *retained >= MAX_BINDINGS_PER_CELL_FUNCTION {
                    continue;
                }
                *retained += 1;
                bindings
                    .entry((input_count, truth))
                    .or_default()
                    .push(RemapBinding {
                        cell_index: cell.cell_index,
                        input_to_leaf: permutation.clone(),
                        area: cell.area,
                        nominal_delay: cell.nominal_delay,
                    });
            }
        }
        for matches in bindings.values_mut() {
            matches.sort_by(|lhs, rhs| {
                lhs.area
                    .total_cmp(&rhs.area)
                    .then_with(|| lhs.nominal_delay.total_cmp(&rhs.nominal_delay))
                    .then_with(|| lhs.cell_index.cmp(&rhs.cell_index))
                    .then_with(|| lhs.input_to_leaf.cmp(&rhs.input_to_leaf))
            });
        }
        let mut signatures = vec![Vec::new(); max_leaves + 1];
        for &(input_count, truth) in bindings.keys() {
            signatures[input_count].push(truth);
        }
        Self {
            bindings,
            signatures,
            decompositions: Mutex::new(BTreeMap::new()),
        }
    }

    /// Returns every safe physical realization of one exact Boolean function.
    fn matches(&self, input_count: usize, truth: u64) -> &[RemapBinding] {
        self.bindings
            .get(&(input_count, truth))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Lazily derives exact nontrivial two-cell alternatives for one function.
    fn decompositions(
        &self,
        input_count: usize,
        truth: u64,
    ) -> Result<Arc<Vec<TwoCellDecomposition>>> {
        if input_count < 2 || input_count >= self.signatures.len() {
            return Ok(Arc::new(Vec::new()));
        }
        if let Some(existing) = self
            .decompositions
            .lock()
            .map_err(|_| anyhow!("mapped remap decomposition cache was poisoned"))?
            .get(&(input_count, truth))
            .cloned()
        {
            return Ok(existing);
        }

        let mut found = Vec::new();
        for mask in 1usize..(1usize << input_count) {
            let child_leaves = (0..input_count)
                .filter(|leaf| (mask >> leaf) & 1 != 0)
                .collect::<Vec<_>>();
            let direct_leaves = (0..input_count)
                .filter(|leaf| (mask >> leaf) & 1 == 0)
                .collect::<Vec<_>>();
            let root_input_count = direct_leaves.len() + 1;
            if root_input_count >= self.signatures.len() {
                continue;
            }
            for &child_truth in &self.signatures[child_leaves.len()] {
                if child_leaves.len() == 1 && child_truth == 0b10 {
                    // A transparent unary stage is already explored by buffer moves.
                    continue;
                }
                let Some(root_truth) = derive_root_truth(
                    input_count,
                    truth,
                    child_leaves.as_slice(),
                    direct_leaves.as_slice(),
                    child_truth,
                ) else {
                    continue;
                };
                if root_input_count == 1 && root_truth == 0b10 {
                    // Wrapping an unchanged implementation in a buffer adds no cover.
                    continue;
                }
                let root_matches = self.matches(root_input_count, root_truth);
                let child_matches = self.matches(child_leaves.len(), child_truth);
                let (Some(root), Some(child)) = (root_matches.first(), child_matches.first())
                else {
                    continue;
                };
                found.push(TwoCellDecomposition {
                    child_leaves: child_leaves.clone(),
                    direct_leaves: direct_leaves.clone(),
                    child_truth,
                    root_truth,
                    estimated_area: root.area + child.area,
                    estimated_delay: root.nominal_delay + child.nominal_delay,
                });
            }
        }
        found.sort_by(|lhs, rhs| {
            lhs.estimated_area
                .total_cmp(&rhs.estimated_area)
                .then_with(|| lhs.estimated_delay.total_cmp(&rhs.estimated_delay))
                .then_with(|| lhs.child_leaves.cmp(&rhs.child_leaves))
                .then_with(|| lhs.child_truth.cmp(&rhs.child_truth))
                .then_with(|| lhs.root_truth.cmp(&rhs.root_truth))
        });
        found.truncate(MAX_DECOMPOSITIONS_PER_FUNCTION);
        let found = Arc::new(found);
        let mut cache = self
            .decompositions
            .lock()
            .map_err(|_| anyhow!("mapped remap decomposition cache was poisoned"))?;
        Ok(cache
            .entry((input_count, truth))
            .or_insert_with(|| found.clone())
            .clone())
    }
}

/// Requested topology category for one exact mapped-cover replacement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RemapShape {
    Collapse,
    Expand,
    Recover,
}

/// Objective and bounded selection policy for one mapped-cover proposal.
#[derive(Clone, Copy, Debug)]
pub(super) struct RemapRequest<'a> {
    pub shape: RemapShape,
    pub objective: NetlistMcmcObjective,
    pub max_leaves: usize,
    pub critical: &'a [usize],
}

/// One fanout-closed combinational region and its exact boundary function.
#[derive(Clone, Debug)]
struct RemapWindow {
    root: usize,
    internal: BTreeSet<usize>,
    leaves: Vec<usize>,
    truth: u64,
}

/// Replaces one fanout-safe region with an exactly equivalent cell covering.
#[allow(clippy::too_many_arguments)]
pub(super) fn propose_equivalent_remap<R: RngCore + ?Sized>(
    state: &mut SearchState,
    connectivity: &Connectivity,
    catalog: &CellCatalog,
    library: &Library,
    index: &RemapLibrary,
    timing: &SearchIncrementalSta<'_>,
    request: RemapRequest<'_>,
    rng: &mut R,
) -> Result<Option<Vec<String>>> {
    let Some(root) = choose_remap_root(state, connectivity, timing, request, rng) else {
        return Ok(None);
    };
    match request.shape {
        RemapShape::Expand => expand_root(
            state,
            connectivity,
            catalog,
            library,
            index,
            root,
            request,
            rng,
        ),
        RemapShape::Collapse | RemapShape::Recover => replace_window(
            state,
            connectivity,
            catalog,
            library,
            index,
            root,
            request,
            rng,
        ),
    }
}

/// Biases remapping toward physical critical roots while excluding registers.
fn choose_remap_root<R: RngCore + ?Sized>(
    state: &SearchState,
    connectivity: &Connectivity,
    timing: &SearchIncrementalSta<'_>,
    request: RemapRequest<'_>,
    rng: &mut R,
) -> Option<usize> {
    if state.module.instances.is_empty() {
        return None;
    }
    for _ in 0..MAX_ROOT_SELECTION_ATTEMPTS {
        let selected = if !request.critical.is_empty() && rng.gen_bool(0.7) {
            request.critical.choose(rng).copied()
        } else {
            Some(rng.gen_range(0..state.module.instances.len()))
        }?;
        let Some(node) = connectivity.logic.get(selected).and_then(Option::as_ref) else {
            continue;
        };
        if connectivity.fanouts[node.output_bit].protected_clock
            || connectivity.fanouts[node.output_bit].protected_assign
            || timing.current_cell_index(selected).is_none()
        {
            continue;
        }
        if request.shape == RemapShape::Expand && node.input_bits.len() < 2 {
            continue;
        }
        return Some(selected);
    }
    None
}

/// Chooses an exact one-cell covering for a bounded fanout-free logic window.
#[allow(clippy::too_many_arguments)]
fn replace_window<R: RngCore + ?Sized>(
    state: &mut SearchState,
    connectivity: &Connectivity,
    catalog: &CellCatalog,
    library: &Library,
    index: &RemapLibrary,
    root: usize,
    request: RemapRequest<'_>,
    rng: &mut R,
) -> Result<Option<Vec<String>>> {
    let windows = enumerate_windows(
        connectivity,
        catalog,
        library,
        root,
        request.max_leaves,
        rng,
    )?;
    let mut viable = windows
        .into_iter()
        .filter(|window| {
            (request.shape != RemapShape::Collapse || window.internal.len() > 1)
                && index
                    .matches(window.leaves.len(), window.truth)
                    .iter()
                    .any(|binding| {
                        binding_changes_window(window, binding, connectivity, catalog, library)
                    })
        })
        .collect::<Vec<_>>();
    if viable.is_empty() {
        return Ok(None);
    }
    viable.sort_by_key(|window| std::cmp::Reverse(window.internal.len()));
    let window = if rng.gen_bool(0.8) {
        &viable[0]
    } else {
        viable
            .choose(rng)
            .expect("viable remap windows are nonempty")
    };
    let options = index
        .matches(window.leaves.len(), window.truth)
        .iter()
        .filter(|binding| binding_changes_window(window, binding, connectivity, catalog, library))
        .collect::<Vec<_>>();
    let Some(binding) = choose_binding(options, request.objective, rng) else {
        return Ok(None);
    };
    let root_node = connectivity.logic[root]
        .as_ref()
        .ok_or_else(|| anyhow!("mapped cone root lost its combinational implementation"))?;
    let original = catalog
        .by_name(library.cells[root_node.cell_index].name.as_str())
        .ok_or_else(|| anyhow!("mapped cone root is missing from the Liberty catalog"))?;
    let output = output_reference(state, root, original)?;
    let leaves = window
        .leaves
        .iter()
        .map(|bit| bit_reference(state, connectivity, *bit))
        .collect::<Result<Vec<_>>>()?;
    let replacement = catalog
        .by_name(library.cells[binding.cell_index].name.as_str())
        .ok_or_else(|| anyhow!("replacement cell is missing from the Liberty catalog"))?;
    let root_name = instance_name(state, root)?;
    assign_binding(state, root, replacement, binding, leaves.as_slice(), output)?;
    for internal in window.internal.iter().rev().copied() {
        if internal != root {
            state.module.instances.remove(internal);
        }
    }
    Ok(Some(vec![root_name]))
}

/// Rejects unchanged coverings and ordinary pin-compatible size substitutions.
fn binding_changes_window(
    window: &RemapWindow,
    binding: &RemapBinding,
    connectivity: &Connectivity,
    catalog: &CellCatalog,
    library: &Library,
) -> bool {
    if window.internal.len() > 1 {
        return true;
    }
    let Some(current) = connectivity.logic[window.root].as_ref() else {
        return false;
    };
    if current.cell_index == binding.cell_index {
        return false;
    }
    let Some(current_cell) = catalog.by_name(library.cells[current.cell_index].name.as_str())
    else {
        return false;
    };
    let Some(replacement) = catalog.by_name(library.cells[binding.cell_index].name.as_str()) else {
        return false;
    };
    current_cell.family != replacement.family
}

/// Builds progressively larger windows without crossing shared or visible nets.
fn enumerate_windows<R: RngCore + ?Sized>(
    connectivity: &Connectivity,
    catalog: &CellCatalog,
    library: &Library,
    root: usize,
    max_leaves: usize,
    rng: &mut R,
) -> Result<Vec<RemapWindow>> {
    let mut internal = BTreeSet::from([root]);
    let mut windows = Vec::new();
    loop {
        let leaves = boundary_leaves(connectivity, &internal);
        if leaves.is_empty() || leaves.len() > max_leaves {
            break;
        }
        let truth = window_truth(
            connectivity,
            catalog,
            library,
            root,
            &internal,
            leaves.as_slice(),
        )?;
        windows.push(RemapWindow {
            root,
            internal: internal.clone(),
            leaves,
            truth,
        });
        if internal.len() >= MAX_WINDOW_INSTANCES {
            break;
        }
        let Some(current) = windows.last() else {
            break;
        };
        let mut expandable = current
            .leaves
            .iter()
            .filter_map(|bit| {
                let fanout = connectivity.fanouts.get(*bit)?;
                let driver = fanout.driver?;
                (!fanout.primary_output
                    && !fanout.protected_assign
                    && !fanout.protected_clock
                    && fanout.sinks.len() == 1
                    && connectivity.logic.get(driver.instance_index)?.is_some())
                .then_some(driver.instance_index)
            })
            .filter(|instance| !internal.contains(instance))
            .collect::<Vec<_>>();
        expandable.sort_unstable();
        expandable.dedup();
        expandable.shuffle(rng);
        let next = expandable.into_iter().find(|instance| {
            let mut trial = internal.clone();
            trial.insert(*instance);
            boundary_leaves(connectivity, &trial).len() <= max_leaves
        });
        let Some(next) = next else {
            break;
        };
        internal.insert(next);
    }
    Ok(windows)
}

/// Returns every distinct external bit feeding a closed mapped logic region.
fn boundary_leaves(connectivity: &Connectivity, internal: &BTreeSet<usize>) -> Vec<usize> {
    let mut leaves = BTreeSet::new();
    for &instance in internal {
        let Some(node) = connectivity.logic[instance].as_ref() else {
            continue;
        };
        for &bit in &node.input_bits {
            if connectivity.fanouts[bit]
                .driver
                .is_none_or(|driver| !internal.contains(&driver.instance_index))
            {
                leaves.insert(bit);
            }
        }
    }
    leaves.into_iter().collect()
}

/// Exhaustively computes one cone truth table over its ordered boundary bits.
fn window_truth(
    connectivity: &Connectivity,
    catalog: &CellCatalog,
    library: &Library,
    root: usize,
    internal: &BTreeSet<usize>,
    leaves: &[usize],
) -> Result<u64> {
    let root_bit = connectivity.logic[root]
        .as_ref()
        .ok_or_else(|| anyhow!("mapped cone has no classified root"))?
        .output_bit;
    let leaf_positions = leaves
        .iter()
        .copied()
        .enumerate()
        .map(|(index, bit)| (bit, index))
        .collect::<HashMap<_, _>>();
    let mut truth = 0u64;
    for assignment in 0..(1usize << leaves.len()) {
        let mut memo = HashMap::new();
        if evaluate_bit(
            root_bit,
            assignment,
            connectivity,
            catalog,
            library,
            internal,
            &leaf_positions,
            &mut memo,
        )? {
            truth |= 1u64 << assignment;
        }
    }
    Ok(truth)
}

/// Evaluates one internal combinational signal without crossing cone
/// boundaries.
#[allow(clippy::too_many_arguments)]
fn evaluate_bit(
    bit: usize,
    assignment: usize,
    connectivity: &Connectivity,
    catalog: &CellCatalog,
    library: &Library,
    internal: &BTreeSet<usize>,
    leaves: &HashMap<usize, usize>,
    memo: &mut HashMap<usize, bool>,
) -> Result<bool> {
    if let Some(value) = memo.get(&bit).copied() {
        return Ok(value);
    }
    if let Some(position) = leaves.get(&bit).copied() {
        return Ok((assignment >> position) & 1 != 0);
    }
    let driver = connectivity.fanouts[bit]
        .driver
        .ok_or_else(|| anyhow!("mapped cone contains an undriven internal signal"))?;
    if !internal.contains(&driver.instance_index) {
        return Err(anyhow!("mapped cone crossed its declared Boolean boundary"));
    }
    let node = connectivity.logic[driver.instance_index]
        .as_ref()
        .ok_or_else(|| anyhow!("mapped cone reached a non-combinational instance"))?;
    let cell = catalog
        .by_name(library.cells[node.cell_index].name.as_str())
        .ok_or_else(|| anyhow!("mapped cone cell is missing from its Liberty catalog"))?;
    let mut cell_assignment = 0usize;
    for (pin, input) in node.input_bits.iter().copied().enumerate() {
        if evaluate_bit(
            input,
            assignment,
            connectivity,
            catalog,
            library,
            internal,
            leaves,
            memo,
        )? {
            cell_assignment |= 1usize << pin;
        }
    }
    let value = ((cell.family.truth >> cell_assignment) & 1) != 0;
    memo.insert(bit, value);
    Ok(value)
}

/// Expands one complex gate into an exhaustively verified two-cell network.
#[allow(clippy::too_many_arguments)]
fn expand_root<R: RngCore + ?Sized>(
    state: &mut SearchState,
    connectivity: &Connectivity,
    catalog: &CellCatalog,
    library: &Library,
    index: &RemapLibrary,
    root: usize,
    request: RemapRequest<'_>,
    rng: &mut R,
) -> Result<Option<Vec<String>>> {
    let internal = BTreeSet::from([root]);
    let leaves = boundary_leaves(connectivity, &internal);
    if !(2..=request.max_leaves).contains(&leaves.len()) {
        return Ok(None);
    }
    let truth = window_truth(
        connectivity,
        catalog,
        library,
        root,
        &internal,
        leaves.as_slice(),
    )?;
    let decompositions = index.decompositions(leaves.len(), truth)?;
    let Some(template) = choose_decomposition(decompositions.as_slice(), request.objective, rng)
    else {
        return Ok(None);
    };
    let child = choose_binding(
        index
            .matches(template.child_leaves.len(), template.child_truth)
            .iter()
            .collect(),
        request.objective,
        rng,
    )
    .ok_or_else(|| anyhow!("mapped child decomposition lost its exact cell binding"))?;
    let replacement = choose_binding(
        index
            .matches(template.direct_leaves.len() + 1, template.root_truth)
            .iter()
            .collect(),
        request.objective,
        rng,
    )
    .ok_or_else(|| anyhow!("mapped root decomposition lost its exact cell binding"))?;

    let root_node = connectivity.logic[root]
        .as_ref()
        .ok_or_else(|| anyhow!("expanded root has no combinational implementation"))?;
    let original = catalog
        .by_name(library.cells[root_node.cell_index].name.as_str())
        .ok_or_else(|| anyhow!("expanded root is missing from the Liberty catalog"))?;
    let output = output_reference(state, root, original)?;
    let references = leaves
        .iter()
        .map(|bit| bit_reference(state, connectivity, *bit))
        .collect::<Result<Vec<_>>>()?;
    let intermediate = append_remap_wire(state);
    let child_inputs = template
        .child_leaves
        .iter()
        .map(|leaf| references[*leaf].clone())
        .collect::<Vec<_>>();
    let root_inputs = std::iter::once(NetRef::Simple(intermediate))
        .chain(
            template
                .direct_leaves
                .iter()
                .map(|leaf| references[*leaf].clone()),
        )
        .collect::<Vec<_>>();
    let root_cell = catalog
        .by_name(library.cells[replacement.cell_index].name.as_str())
        .ok_or_else(|| anyhow!("expanded root replacement is missing from the Liberty catalog"))?;
    let child_cell = catalog
        .by_name(library.cells[child.cell_index].name.as_str())
        .ok_or_else(|| anyhow!("expanded child replacement is missing from the Liberty catalog"))?;
    let root_name = instance_name(state, root)?;
    assign_binding(
        state,
        root,
        root_cell,
        replacement,
        root_inputs.as_slice(),
        output,
    )?;
    let child_name = fresh_instance_name(state);
    let child_index = state.module.instances.len();
    state.module.instances.push(NetlistInstance {
        type_name: state.interner.get_or_intern(child_cell.name.as_str()),
        instance_name: state.interner.get_or_intern(child_name.as_str()),
        connections: Vec::new(),
        inst_lineno: 1,
        inst_colno: 1,
    });
    assign_binding(
        state,
        child_index,
        child_cell,
        child,
        child_inputs.as_slice(),
        NetRef::Simple(intermediate),
    )?;
    Ok(Some(vec![root_name, child_name]))
}

/// Chooses promising physical bindings without removing exploratory variation.
fn choose_binding<'a, R: RngCore + ?Sized>(
    mut bindings: Vec<&'a RemapBinding>,
    objective: NetlistMcmcObjective,
    rng: &mut R,
) -> Option<&'a RemapBinding> {
    if bindings.is_empty() {
        return None;
    }
    bindings.sort_by(|lhs, rhs| {
        let primary = if objective == NetlistMcmcObjective::Area {
            lhs.area.total_cmp(&rhs.area)
        } else {
            lhs.nominal_delay.total_cmp(&rhs.nominal_delay)
        };
        primary
            .then_with(|| lhs.area.total_cmp(&rhs.area))
            .then_with(|| lhs.cell_index.cmp(&rhs.cell_index))
            .then_with(|| lhs.input_to_leaf.cmp(&rhs.input_to_leaf))
    });
    let width = bindings.len().min(8);
    if rng.gen_bool(0.75) {
        bindings[..width].choose(rng).copied()
    } else {
        bindings.choose(rng).copied()
    }
}

/// Selects an area- or delay-oriented Boolean decomposition deterministically.
fn choose_decomposition<'a, R: RngCore + ?Sized>(
    templates: &'a [TwoCellDecomposition],
    objective: NetlistMcmcObjective,
    rng: &mut R,
) -> Option<&'a TwoCellDecomposition> {
    if templates.is_empty() {
        return None;
    }
    if objective == NetlistMcmcObjective::Area {
        return templates[..templates.len().min(12)].choose(rng);
    }
    let mut fastest = templates.iter().collect::<Vec<_>>();
    fastest.sort_by(|lhs, rhs| {
        lhs.estimated_delay
            .total_cmp(&rhs.estimated_delay)
            .then_with(|| lhs.estimated_area.total_cmp(&rhs.estimated_area))
    });
    fastest[..fastest.len().min(12)].choose(rng).copied()
}

/// Preserves the original logical output while reconnecting named Liberty pins.
fn assign_binding(
    state: &mut SearchState,
    instance_index: usize,
    cell: &CatalogCell,
    binding: &RemapBinding,
    leaves: &[NetRef],
    output: NetRef,
) -> Result<()> {
    if cell.family.input_names.len() != binding.input_to_leaf.len() {
        return Err(anyhow!(
            "mapped replacement input arity disagrees with its truth binding"
        ));
    }
    let mut connections =
        cell.family
            .input_names
            .iter()
            .enumerate()
            .map(|(pin, name)| {
                let leaf = *binding
                    .input_to_leaf
                    .get(pin)
                    .ok_or_else(|| anyhow!("mapped replacement pin has no boundary assignment"))?;
                let source = leaves.get(leaf).cloned().ok_or_else(|| {
                    anyhow!("mapped replacement pin references an invalid boundary")
                })?;
                Ok((state.interner.get_or_intern(name.as_str()), source))
            })
            .collect::<Result<Vec<_>>>()?;
    connections.push((
        state
            .interner
            .get_or_intern(cell.family.output_name.as_str()),
        output,
    ));
    connections.sort_by_key(|(pin, _)| state.interner.resolve(*pin).unwrap_or("").to_string());
    let instance = state
        .module
        .instances
        .get_mut(instance_index)
        .ok_or_else(|| anyhow!("mapped replacement references a missing physical instance"))?;
    instance.type_name = state.interner.get_or_intern(cell.name.as_str());
    instance.connections = connections;
    Ok(())
}

/// Finds the raw reference that must remain driven by a replacement root.
fn output_reference(state: &SearchState, instance: usize, cell: &CatalogCell) -> Result<NetRef> {
    state.module.instances[instance]
        .connections
        .iter()
        .find(|(pin, _)| state.interner.resolve(*pin) == Some(cell.family.output_name.as_str()))
        .map(|(_, reference)| reference.clone())
        .ok_or_else(|| anyhow!("mapped cone root has no connected combinational output"))
}

/// Allocates a deterministic internal scalar wire for a decomposed cover.
fn append_remap_wire(state: &mut SearchState) -> NetIndex {
    for suffix in 0usize.. {
        let name = format!("n_mcmc_remap_{suffix}");
        if state.interner.get(name.as_str()).is_some() {
            continue;
        }
        let index = NetIndex(state.nets.len());
        state.nets.push(Net {
            name: state.interner.get_or_intern(name.as_str()),
            width: None,
        });
        state.module.wires.push(index);
        state.module.net_index_range.end = state.nets.len();
        return index;
    }
    unreachable!("a collision-free mapped remap wire must exist")
}

/// Produces a collision-free physical name for a newly introduced logic gate.
fn fresh_instance_name(state: &SearchState) -> String {
    (0usize..)
        .map(|suffix| format!("u_mcmc_remap_{suffix}"))
        .find(|name| state.interner.get(name.as_str()).is_none())
        .expect("a collision-free mapped remap instance name must exist")
}

/// Reexpresses a native cell truth table over externally ordered input leaves.
fn permuted_truth(truth: u64, input_to_leaf: &[usize]) -> u64 {
    let mut result = 0u64;
    for assignment in 0..(1usize << input_to_leaf.len()) {
        let mut cell_assignment = 0usize;
        for (pin, leaf) in input_to_leaf.iter().copied().enumerate() {
            if (assignment >> leaf) & 1 != 0 {
                cell_assignment |= 1usize << pin;
            }
        }
        if (truth >> cell_assignment) & 1 != 0 {
            result |= 1u64 << assignment;
        }
    }
    result
}

/// Derives the only possible root function for a candidate child partition.
fn derive_root_truth(
    input_count: usize,
    target_truth: u64,
    child_leaves: &[usize],
    direct_leaves: &[usize],
    child_truth: u64,
) -> Option<u64> {
    let root_inputs = direct_leaves.len() + 1;
    let mut seen = vec![None; 1usize << root_inputs];
    for assignment in 0..(1usize << input_count) {
        let mut child_assignment = 0usize;
        for (index, leaf) in child_leaves.iter().copied().enumerate() {
            if (assignment >> leaf) & 1 != 0 {
                child_assignment |= 1usize << index;
            }
        }
        let child_value = ((child_truth >> child_assignment) & 1) != 0;
        let mut root_assignment = usize::from(child_value);
        for (index, leaf) in direct_leaves.iter().copied().enumerate() {
            if (assignment >> leaf) & 1 != 0 {
                root_assignment |= 1usize << (index + 1);
            }
        }
        let value = ((target_truth >> assignment) & 1) != 0;
        match seen[root_assignment] {
            Some(previous) if previous != value => return None,
            Some(_) => {
                // Repeated assignments with the same value are consistent.
            }
            None => seen[root_assignment] = Some(value),
        }
    }
    let mut truth = 0u64;
    for (assignment, value) in seen.into_iter().enumerate() {
        if value? {
            truth |= 1u64 << assignment;
        }
    }
    Some(truth)
}

/// Lists stable bounded permutations once rather than regenerating per cell.
fn permutations(input_count: usize) -> Vec<Vec<usize>> {
    fn visit(position: usize, current: &mut [usize], output: &mut Vec<Vec<usize>>) {
        if position == current.len() {
            output.push(current.to_vec());
            return;
        }
        for next in position..current.len() {
            current.swap(position, next);
            visit(position + 1, current, output);
            current.swap(position, next);
        }
    }
    let mut current = (0..input_count).collect::<Vec<_>>();
    let mut output = Vec::new();
    visit(0, current.as_mut_slice(), &mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::{derive_root_truth, permuted_truth};

    #[test]
    fn input_permutations_preserve_exact_truth_assignments() {
        let a_and_not_b = 0b0010;
        assert_eq!(permuted_truth(a_and_not_b, &[1, 0]), 0b0100);
    }

    #[test]
    fn derives_an_exact_two_stage_and_or_decomposition() {
        // (a & b) | c in assignment order a, b, c.
        let target = 0b1111_1000;
        let and = 0b1000;
        let or = 0b1110;
        assert_eq!(derive_root_truth(3, target, &[0, 1], &[2], and), Some(or));
    }

    #[test]
    fn rejects_a_child_that_cannot_preserve_the_target_function() {
        // XOR(a, b) cannot be reconstructed from AND(a, b) alone.
        let xor = 0b0110;
        let and = 0b1000;
        assert_eq!(derive_root_truth(2, xor, &[0, 1], &[], and), None);
    }
}
