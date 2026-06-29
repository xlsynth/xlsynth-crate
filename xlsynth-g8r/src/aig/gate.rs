// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use bitvec::vec::BitVec;
use xlsynth::IrBits;
use xlsynth_pir::ir_value_utils::ir_bits_from_bitvec_lsb_is_0;

use crate::aig::topo::post_order_operands;
use xlsynth_pir::ir;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct AigRef {
    pub id: usize,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct AigOperand {
    pub node: AigRef,
    pub negated: bool,
}

impl AigOperand {
    #[must_use]
    pub fn negate(&self) -> Self {
        Self {
            node: self.node,
            negated: !self.negated,
        }
    }

    pub fn non_negated(&self) -> Option<AigRef> {
        if self.negated { None } else { Some(self.node) }
    }
}

impl From<AigRef> for AigOperand {
    fn from(node: AigRef) -> Self {
        AigOperand {
            node,
            negated: false,
        }
    }
}

impl From<&AigRef> for AigOperand {
    fn from(node: &AigRef) -> Self {
        AigOperand {
            node: *node,
            negated: false,
        }
    }
}

#[derive(Clone)]
enum PirNodeIdsRepr {
    Inline(SmallVec<[u32; 2]>),
    Shared(Arc<[u32]>),
}

/// A sorted, deduplicated set of PIR provenance IDs.
///
/// Small sets remain inline. Larger sets can be interned and shared immutably
/// across AIG nodes, which avoids repeatedly copying large provenance unions
/// during graph rewrites.
#[derive(Clone)]
pub struct PirNodeIds {
    repr: PirNodeIdsRepr,
}

impl PirNodeIds {
    pub fn new() -> Self {
        Self {
            repr: PirNodeIdsRepr::Inline(SmallVec::new()),
        }
    }

    fn from_normalized_vec(ids: Vec<u32>) -> Self {
        debug_assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
        if ids.len() <= 2 {
            Self {
                repr: PirNodeIdsRepr::Inline(SmallVec::from_vec(ids)),
            }
        } else {
            Self {
                repr: PirNodeIdsRepr::Shared(Arc::from(ids)),
            }
        }
    }

    fn singleton(id: u32) -> Self {
        Self {
            repr: PirNodeIdsRepr::Inline(SmallVec::from_slice(&[id])),
        }
    }

    pub fn as_slice(&self) -> &[u32] {
        match &self.repr {
            PirNodeIdsRepr::Inline(ids) => ids.as_slice(),
            PirNodeIdsRepr::Shared(ids) => ids.as_ref(),
        }
    }

    /// Unions `ids` into this set using a linear merge of sorted inputs.
    pub fn union_with_slice(&mut self, ids: &[u32]) {
        let normalized = normalize_ids(ids);
        if let Some(merged) = merge_sorted_unique(self.as_slice(), normalized.as_ref()) {
            *self = Self::from_normalized_vec(merged);
        }
    }

    pub fn insert(&mut self, index: usize, id: u32) {
        let mut ids = self.as_slice().to_vec();
        ids.insert(index, id);
        *self = Self::from_normalized_vec(ids);
    }

    pub fn extend(&mut self, ids: impl IntoIterator<Item = u32>) {
        let ids = ids.into_iter().collect::<Vec<_>>();
        self.union_with_slice(&ids);
    }

    #[cfg(test)]
    pub(crate) fn shares_storage_with(&self, other: &Self) -> bool {
        matches!(
            (&self.repr, &other.repr),
            (PirNodeIdsRepr::Shared(lhs), PirNodeIdsRepr::Shared(rhs))
                if Arc::ptr_eq(lhs, rhs)
        )
    }
}

impl Default for PirNodeIds {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<u32> for PirNodeIds {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let mut ids = iter.into_iter().collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        Self::from_normalized_vec(ids)
    }
}

impl Deref for PirNodeIds {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl fmt::Debug for PirNodeIds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl PartialEq for PirNodeIds {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for PirNodeIds {}

impl Hash for PirNodeIds {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl Serialize for PirNodeIds {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PirNodeIds {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut ids = Vec::<u32>::deserialize(deserializer)?;
        ids.sort_unstable();
        ids.dedup();
        Ok(Self::from_normalized_vec(ids))
    }
}

/// Interns immutable provenance sets for a single graph construction.
#[derive(Default)]
pub(crate) struct PirNodeIdsInterner {
    shared: HashSet<Arc<[u32]>>,
}

impl PirNodeIdsInterner {
    pub(crate) fn intern_slice(&mut self, ids: &[u32]) -> PirNodeIds {
        let normalized = normalize_ids(ids);
        self.intern_normalized_slice(normalized.as_ref())
    }

    pub(crate) fn union(&mut self, lhs: &PirNodeIds, rhs: &[u32]) -> PirNodeIds {
        let normalized_rhs = normalize_ids(rhs);
        if normalized_rhs.is_empty() || lhs.as_slice() == normalized_rhs.as_ref() {
            return lhs.clone();
        }
        if lhs.is_empty() {
            return self.intern_normalized_slice(normalized_rhs.as_ref());
        }
        match merge_sorted_unique(lhs.as_slice(), normalized_rhs.as_ref()) {
            None => lhs.clone(),
            Some(merged) => self.intern_normalized_vec(merged),
        }
    }

    pub(crate) fn union_sets(&mut self, lhs: &PirNodeIds, rhs: &PirNodeIds) -> PirNodeIds {
        if rhs.is_empty() || lhs == rhs {
            return lhs.clone();
        }
        if lhs.is_empty() {
            return rhs.clone();
        }
        match merge_sorted_unique(lhs.as_slice(), rhs.as_slice()) {
            None => lhs.clone(),
            Some(merged) => self.intern_normalized_vec(merged),
        }
    }

    fn intern_normalized_slice(&mut self, ids: &[u32]) -> PirNodeIds {
        debug_assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
        if ids.len() <= 2 {
            return PirNodeIds::from_normalized_vec(ids.to_vec());
        }
        if let Some(existing) = self.shared.get(ids) {
            return PirNodeIds {
                repr: PirNodeIdsRepr::Shared(existing.clone()),
            };
        }
        let shared: Arc<[u32]> = Arc::from(ids);
        self.shared.insert(shared.clone());
        PirNodeIds {
            repr: PirNodeIdsRepr::Shared(shared),
        }
    }

    fn intern_normalized_vec(&mut self, ids: Vec<u32>) -> PirNodeIds {
        debug_assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
        if ids.len() <= 2 {
            return PirNodeIds::from_normalized_vec(ids);
        }
        if let Some(existing) = self.shared.get(ids.as_slice()) {
            return PirNodeIds {
                repr: PirNodeIdsRepr::Shared(existing.clone()),
            };
        }
        let shared: Arc<[u32]> = Arc::from(ids);
        self.shared.insert(shared.clone());
        PirNodeIds {
            repr: PirNodeIdsRepr::Shared(shared),
        }
    }
}

fn normalize_ids(ids: &[u32]) -> Cow<'_, [u32]> {
    if ids.windows(2).all(|pair| pair[0] < pair[1]) {
        return Cow::Borrowed(ids);
    }
    let mut normalized = ids.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    Cow::Owned(normalized)
}

/// Returns `None` when `rhs` is already a subset of `lhs`.
fn merge_sorted_unique(lhs: &[u32], rhs: &[u32]) -> Option<Vec<u32>> {
    if rhs.is_empty() {
        return None;
    }
    let mut merged = Vec::with_capacity(lhs.len() + rhs.len());
    let (mut lhs_index, mut rhs_index) = (0, 0);
    while lhs_index < lhs.len() && rhs_index < rhs.len() {
        match lhs[lhs_index].cmp(&rhs[rhs_index]) {
            std::cmp::Ordering::Less => {
                merged.push(lhs[lhs_index]);
                lhs_index += 1;
            }
            std::cmp::Ordering::Equal => {
                merged.push(lhs[lhs_index]);
                lhs_index += 1;
                rhs_index += 1;
            }
            std::cmp::Ordering::Greater => {
                merged.push(rhs[rhs_index]);
                rhs_index += 1;
            }
        }
    }
    merged.extend_from_slice(&lhs[lhs_index..]);
    merged.extend_from_slice(&rhs[rhs_index..]);
    (merged.as_slice() != lhs).then_some(merged)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AigNode {
    Input {
        name: String,
        /// Index where 0 is the least significant bit of the input.
        lsb_index: usize,
        pir_node_ids: PirNodeIds,
    },
    Literal {
        value: bool,
        pir_node_ids: PirNodeIds,
    },
    And2 {
        a: AigOperand,
        b: AigOperand,
        tags: Option<Vec<String>>,
        pir_node_ids: PirNodeIds,
    },
}

impl AigNode {
    pub fn get_operands(&self) -> Vec<AigOperand> {
        match self {
            AigNode::Input { .. } => vec![],
            AigNode::Literal { .. } => vec![],
            AigNode::And2 { a, b, .. } => vec![a.clone(), b.clone()],
        }
    }

    pub fn get_args(&self) -> Vec<AigRef> {
        match self {
            AigNode::Input { .. } => vec![],
            AigNode::Literal { .. } => vec![],
            AigNode::And2 { a, b, .. } => vec![a.node, b.node],
        }
    }

    pub fn add_tag(&mut self, tag: String) {
        match self {
            AigNode::And2 { tags, .. } => {
                if let Some(tags) = tags {
                    tags.push(tag);
                } else {
                    *tags = Some(vec![tag]);
                }
            }
            _ => {}
        }
    }

    fn pir_node_ids_mut(&mut self) -> &mut PirNodeIds {
        match self {
            AigNode::Input { pir_node_ids, .. }
            | AigNode::Literal { pir_node_ids, .. }
            | AigNode::And2 { pir_node_ids, .. } => pir_node_ids,
        }
    }

    pub fn add_pir_node_id(&mut self, pir_node_id: u32) {
        let pir_node_ids = self.pir_node_ids_mut();
        match pir_node_ids.binary_search(&pir_node_id) {
            Ok(_) => {}
            Err(index) => pir_node_ids.insert(index, pir_node_id),
        }
    }

    pub fn try_add_pir_node_ids(&mut self, pir_node_ids_to_add: &[u32]) {
        self.pir_node_ids_mut()
            .union_with_slice(pir_node_ids_to_add);
    }

    pub(crate) fn try_add_interned_pir_node_ids(
        &mut self,
        pir_node_ids_to_add: &[u32],
        interner: &mut PirNodeIdsInterner,
    ) {
        let merged = interner.union(
            match self {
                AigNode::Input { pir_node_ids, .. }
                | AigNode::Literal { pir_node_ids, .. }
                | AigNode::And2 { pir_node_ids, .. } => pir_node_ids,
            },
            pir_node_ids_to_add,
        );
        *self.pir_node_ids_mut() = merged;
    }

    pub(crate) fn try_add_interned_pir_node_id_set(
        &mut self,
        pir_node_ids_to_add: &PirNodeIds,
        interner: &mut PirNodeIdsInterner,
    ) {
        let merged = interner.union_sets(
            match self {
                AigNode::Input { pir_node_ids, .. }
                | AigNode::Literal { pir_node_ids, .. }
                | AigNode::And2 { pir_node_ids, .. } => pir_node_ids,
            },
            pir_node_ids_to_add,
        );
        *self.pir_node_ids_mut() = merged;
    }

    pub fn get_pir_node_ids(&self) -> &[u32] {
        match self {
            AigNode::Input { pir_node_ids, .. }
            | AigNode::Literal { pir_node_ids, .. }
            | AigNode::And2 { pir_node_ids, .. } => pir_node_ids.as_slice(),
        }
    }

    #[cfg(test)]
    pub(crate) fn shares_pir_node_id_storage_with(&self, other: &AigNode) -> bool {
        self.pir_node_id_set()
            .shares_storage_with(other.pir_node_id_set())
    }

    #[cfg(test)]
    fn pir_node_id_set(&self) -> &PirNodeIds {
        match self {
            AigNode::Input { pir_node_ids, .. }
            | AigNode::Literal { pir_node_ids, .. }
            | AigNode::And2 { pir_node_ids, .. } => pir_node_ids,
        }
    }

    pub fn with_pir_node_id(pir_node_id: Option<u32>) -> PirNodeIds {
        match pir_node_id {
            Some(pir_node_id) => PirNodeIds::singleton(pir_node_id),
            None => PirNodeIds::new(),
        }
    }

    pub fn union_pir_node_ids_from_node(&mut self, other: &AigNode) {
        self.try_add_pir_node_ids(other.get_pir_node_ids());
    }

    /// Attempts to add multiple tags to the node.
    /// Tags can only be added to `AigNode::And2` nodes.
    ///
    /// # Arguments
    /// * `tags_to_add`: A slice of strings representing the tags to add.
    ///
    /// # Returns
    /// * `true` if the tags were successfully added (i.e., the node was an
    ///   `And2`).
    /// * `false` otherwise.
    pub fn try_add_tags(&mut self, tags_to_add: &[String]) -> bool {
        match self {
            AigNode::And2 { tags, .. } => {
                if tags_to_add.is_empty() {
                    // No tags to add
                    return true; // Considered success, as no action was needed
                }
                if let Some(existing_tags) = tags {
                    existing_tags.extend_from_slice(tags_to_add);
                } else {
                    // If no tags exist yet, create a new Vec
                    *tags = Some(tags_to_add.to_vec());
                }
                true // Indicate success
            }
            _ => false, // Not an And2 node, cannot add tags
        }
    }

    /// Returns a slice of the tags associated with the node, if any.
    /// Only `AigNode::And2` can have tags.
    pub fn get_tags(&self) -> Option<&[String]> {
        match self {
            AigNode::And2 {
                tags: Some(tags), ..
            } => Some(tags.as_slice()),
            _ => None, // Not an And2 or has no tags
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AigBitVector {
    /// In this representation index 0 is the LSb, the last index is the MSb.
    operands: Vec<AigOperand>,
}

impl Into<AigBitVector> for AigOperand {
    fn into(self) -> AigBitVector {
        AigBitVector {
            operands: vec![self],
        }
    }
}

impl TryInto<AigOperand> for AigBitVector {
    type Error = String;

    fn try_into(self) -> Result<AigOperand, Self::Error> {
        if self.operands.len() != 1 {
            Err(format!(
                "expected a single operand for AigBitVector::try_into<AigOperand>(), but got {} operands",
                self.operands.len()
            ))
        } else {
            Ok(self.operands[0])
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    pub msbs: AigBitVector,
    pub lsbs: AigBitVector,
}

impl AigBitVector {
    pub fn zeros(bit_count: usize) -> Self {
        Self {
            operands: vec![
                AigOperand {
                    node: AigRef { id: 0 },
                    negated: false
                };
                bit_count
            ],
        }
    }

    pub fn concat(msbs: Self, lsbs: Self) -> Self {
        let mut operands = lsbs.operands;
        operands.extend(msbs.operands);
        Self { operands }
    }

    pub fn from_bit(bit: AigOperand) -> Self {
        Self {
            operands: vec![bit],
        }
    }

    pub fn get_msbs(&self, bit_count: usize) -> Self {
        let mut operands = Vec::with_capacity(bit_count);
        for bit in self.iter_msb_to_lsb().take(bit_count) {
            operands.push(*bit);
        }
        operands.reverse();
        Self { operands }
    }

    pub fn get_lsb_slice(&self, start: usize, bit_width: usize) -> Self {
        AigBitVector {
            operands: self
                .operands
                .iter()
                .skip(start)
                .take(bit_width)
                .cloned()
                .collect(),
        }
    }

    pub fn get_lsb_partition(&self, bit_width: usize) -> Split {
        let (low_bits, high_bits) = self.operands.split_at(bit_width);
        Split {
            msbs: Self::from_lsb_is_index_0(high_bits),
            lsbs: Self::from_lsb_is_index_0(low_bits),
        }
    }

    /// Creates a bit vector from a slice where index 0 of the slice is the
    /// least significant bit.
    pub fn from_lsb_is_index_0(operands: &[AigOperand]) -> Self {
        Self {
            operands: operands.to_vec(),
        }
    }

    pub fn iter_lsb_to_msb(&self) -> impl DoubleEndedIterator<Item = &AigOperand> {
        self.operands.iter()
    }

    pub fn iter_msb_to_lsb(&self) -> impl DoubleEndedIterator<Item = &AigOperand> {
        self.operands.iter().rev()
    }

    pub fn get_lsb(&self, index: usize) -> &AigOperand {
        assert!(
            index < self.operands.len(),
            "index {} is out of bounds for bit vector of length {}",
            index,
            self.operands.len()
        );
        &self.operands[index]
    }

    pub fn get_bit_count(&self) -> usize {
        self.operands.len()
    }

    pub fn get_msb(&self, index: usize) -> &AigOperand {
        assert!(
            index < self.operands.len(),
            "index {} is out of bounds for bit vector of length {}",
            index,
            self.operands.len()
        );
        &self.operands[self.operands.len() - index - 1]
    }

    pub fn is_empty(&self) -> bool {
        self.operands.is_empty()
    }

    pub fn set_lsb(&mut self, index: usize, value: AigOperand) {
        assert!(
            index < self.operands.len(),
            "set_lsb: index {} out of bounds for bit vector of length {}",
            index,
            self.operands.len()
        );
        self.operands[index] = value;
    }
}

fn io_to_string(name: &str, bit_vector: &AigBitVector) -> String {
    let array_str = bit_vector
        .iter_lsb_to_msb()
        .map(|bit| {
            if bit.negated {
                format!("not(%{})", bit.node.id)
            } else {
                format!("%{}", bit.node.id)
            }
        })
        .collect::<Vec<String>>()
        .join(", ");
    format!(
        "{}: bits[{}] = [{}]",
        name,
        bit_vector.get_bit_count(),
        array_str
    )
}

/// An input has a name (which should be unique among inputs/outputs) and a
/// vector of gate references that make up this named entity; i.e. we have bit
/// vectors for named inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Input {
    pub name: String,
    pub bit_vector: AigBitVector,
}

impl Input {
    pub fn get_bit_count(&self) -> usize {
        self.bit_vector.get_bit_count()
    }

    fn to_string(&self) -> String {
        io_to_string(&self.name, &self.bit_vector)
    }
}

/// Similar to inputs, but references from the AIG can be negated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Output {
    pub name: String,
    pub bit_vector: AigBitVector,
}

impl Output {
    pub fn get_bit_count(&self) -> usize {
        self.bit_vector.get_bit_count()
    }

    fn to_string(&self) -> String {
        io_to_string(&self.name, &self.bit_vector)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateFn {
    pub name: String,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub gates: Vec<AigNode>,
}

impl GateFn {
    /// Collapses a sparse mapping of AigRefs to boolean values into a vector of
    /// IrBits that can be fed to gate / IR simulation as input stimulus.
    pub fn map_to_inputs(&self, map: HashMap<AigRef, bool>) -> Vec<IrBits> {
        let mut results: Vec<IrBits> = Vec::new();
        for input in self.inputs.iter() {
            let mut bitvec = BitVec::new();
            for bit in input.bit_vector.iter_lsb_to_msb() {
                let aig_ref = bit.non_negated().unwrap();
                let bit_value = map.get(&aig_ref).expect(&format!(
                    "all input gates should be present in provided sparse map; missing: {:?}",
                    aig_ref
                ));
                bitvec.push(*bit_value);
            }
            results.push(ir_bits_from_bitvec_lsb_is_0(&bitvec));
        }
        results
    }

    pub fn get_flat_type(&self) -> ir::FunctionType {
        let params = self
            .inputs
            .iter()
            .map(|input| ir::Type::Bits(input.get_bit_count()))
            .collect();
        let ret = if self.outputs.len() == 1 {
            ir::Type::Bits(self.outputs[0].get_bit_count())
        } else {
            let members = self
                .outputs
                .iter()
                .map(|output| Box::new(ir::Type::Bits(output.get_bit_count())))
                .collect::<Vec<Box<ir::Type>>>();
            ir::Type::Tuple(members)
        };
        ir::FunctionType {
            param_types: params,
            return_type: ret,
        }
    }

    /// Implementation note: we emit nodes here and the negation is folded into
    /// the node emission process, which means we need a sweep over the
    /// outputs to negate those explicitly.
    pub fn to_string(&self) -> String {
        self.check_invariants_with_debug_assert();
        let mut s = String::new();
        let input_str = self
            .inputs
            .iter()
            .map(|input| input.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        let output_str = self
            .outputs
            .iter()
            .map(|output| output.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        let output_str = format!("({})", output_str);

        let get_node_str = |id: usize| {
            // If it's an input, we use the input name.
            match &self.gates[id] {
                AigNode::Input {
                    name, lsb_index, ..
                } => format!("{}[{}]", name, lsb_index),
                _ => format!("%{}", id),
            }
        };

        s.push_str(&format!(
            "fn {}({input_str}) -> {output_str} {{\n",
            self.name
        ));
        for operand in self.post_order_operands(true) {
            let aig_ref = operand.node;
            let this_node = self.get(aig_ref);
            match this_node {
                AigNode::And2 { a, b, tags, .. } => {
                    let a_node_str = get_node_str(a.node.id);
                    let b_node_str = get_node_str(b.node.id);
                    let a_str = if a.negated {
                        format!("not({})", a_node_str)
                    } else {
                        a_node_str
                    };
                    let b_str = if b.negated {
                        format!("not({})", b_node_str)
                    } else {
                        b_node_str
                    };
                    let tags_str = match tags {
                        Some(tags) => format!(", tags=[{}]", tags.join(", ")),
                        None => "".to_string(),
                    };
                    s.push_str(&format!(
                        "  %{} = and({}, {}{})\n",
                        aig_ref.id, a_str, b_str, tags_str
                    ));
                }
                AigNode::Input { .. } => {
                    continue;
                }
                AigNode::Literal { value, .. } => {
                    s.push_str(&format!("  %{} = literal({})\n", aig_ref.id, value));
                }
            }
        }

        let mut visible_node_ids = BTreeSet::from([0usize]);
        for input in &self.inputs {
            for bit in input.bit_vector.iter_lsb_to_msb() {
                visible_node_ids.insert(bit.node.id);
            }
        }
        for operand in self.post_order_operands(true) {
            visible_node_ids.insert(operand.node.id);
        }
        for node_id in visible_node_ids {
            let pir_node_ids = self.gates[node_id].get_pir_node_ids();
            if pir_node_ids.is_empty() {
                continue;
            }
            let pir_node_ids_str = pir_node_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<String>>()
                .join(", ");
            s.push_str(&format!(
                "  %{} provenance=[{}]\n",
                node_id, pir_node_ids_str
            ));
        }

        for output in &self.outputs {
            for (i, output_bit) in output.bit_vector.iter_lsb_to_msb().enumerate() {
                if output_bit.negated {
                    s.push_str(&format!(
                        "  {}[{}] = not(%{})\n",
                        output.name, i, output_bit.node.id
                    ));
                } else {
                    s.push_str(&format!(
                        "  {}[{}] = %{}\n",
                        output.name, i, output_bit.node.id
                    ));
                }
            }
        }

        s.push_str("}");
        s
    }

    pub fn get(&self, aig_ref: AigRef) -> &AigNode {
        &self.gates[aig_ref.id]
    }

    fn output_operands(&self) -> Vec<AigOperand> {
        let mut result = Vec::new();
        for output in &self.outputs {
            for bit in output.bit_vector.iter_lsb_to_msb() {
                result.push(*bit);
            }
        }
        result
    }

    /// Worklist-based postorder traversal from all outputs, returns
    /// Vec<AigOperand> (with negation).
    pub fn post_order_operands(&self, discard_inputs: bool) -> Vec<AigOperand> {
        let starts = self.output_operands();
        post_order_operands(&starts, &self.gates, discard_inputs)
    }

    pub fn post_order_refs(&self) -> Vec<AigRef> {
        let output_operands = self.output_operands();
        if output_operands.is_empty() {
            // If there are no output operands, the post-order list is empty.
            return Vec::new();
        }
        post_order_operands(&output_operands, &self.gates, true)
            .iter()
            .map(|op| op.node)
            .collect()
    }

    pub fn get_signature(&self) -> String {
        let params_str = self
            .inputs
            .iter()
            .map(|input| format!("{}: bits[{}]", input.name, input.get_bit_count()))
            .collect::<Vec<String>>()
            .join(", ");
        let outputs_str = if self.outputs.len() == 1 {
            format!("bits[{}]", self.outputs[0].get_bit_count())
        } else {
            let guts = self
                .outputs
                .iter()
                .map(|output| format!("bits[{}]", output.get_bit_count()))
                .collect::<Vec<String>>()
                .join(", ");
            format!("({})", guts)
        };
        format!("fn {}({}) -> {}", self.name, params_str, outputs_str)
    }

    /// Checks internal invariants of the GateFn, panicking if any are violated.
    /// - All AigRef indices in inputs, outputs, and gates must be in-bounds for
    ///   self.gates.
    pub fn check_invariants_with_debug_assert(&self) {
        // If we're not in debug-assert build just return.
        if !cfg!(debug_assertions) {
            return;
        }

        let gate_count = self.gates.len();
        // Check all input bit vectors
        for input in &self.inputs {
            for bit in input.bit_vector.iter_lsb_to_msb() {
                assert!(
                    bit.node.id < gate_count,
                    "Input AigRef out of bounds: {:?} (gates.len() = {})",
                    bit.node,
                    gate_count
                );
            }
        }
        // Check all output bit vectors
        for output in &self.outputs {
            for bit in output.bit_vector.iter_lsb_to_msb() {
                assert!(
                    bit.node.id < gate_count,
                    "Output AigRef out of bounds: {:?} (gates.len() = {})",
                    bit.node,
                    gate_count
                );
            }
        }
        // Check all gate operands
        for (i, node) in self.gates.iter().enumerate() {
            match node {
                AigNode::And2 { a, b, .. } => {
                    assert!(
                        a.node.id < gate_count,
                        "Gate %{}: 'a' operand AigRef out of bounds: {:?} (gates.len() = {})",
                        i,
                        a.node,
                        gate_count
                    );
                    assert!(
                        b.node.id < gate_count,
                        "Gate %{}: 'b' operand AigRef out of bounds: {:?} (gates.len() = {})",
                        i,
                        b.node,
                        gate_count
                    );
                }
                AigNode::Input { .. } | AigNode::Literal { .. } => {}
            }
        }
    }

    /// Checks that the given AigRef is in-bounds for this GateFn.
    pub fn validate_ref(&self, aig_ref: AigRef) {
        let gate_count = self.gates.len();
        assert!(
            aig_ref.id < gate_count,
            "AigRef out of bounds: {:?} (gates.len() = {})",
            aig_ref,
            gate_count
        );
    }

    /// Checks that the given AigOperand's node is in-bounds for this GateFn.
    pub fn validate_operand(&self, operand: AigOperand) {
        self.validate_ref(operand.node);
    }
}

#[cfg(test)]
mod tests {
    use super::{AigNode, PirNodeIds, PirNodeIdsInterner};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use std::collections::HashMap;
    use xlsynth::IrBits;

    #[test]
    fn test_pir_node_ids_remain_sorted_with_fast_and_fallback_insertions() {
        let mut node = AigNode::Literal {
            value: false,
            pir_node_ids: PirNodeIds::new(),
        };
        for pir_node_id in [3, 4, 4, 2, 5, 1, 3] {
            node.add_pir_node_id(pir_node_id);
        }
        assert_eq!(node.get_pir_node_ids(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_pir_node_ids_bulk_union_sorts_and_deduplicates() {
        let mut ids = PirNodeIds::from_iter([1, 3, 5]);
        ids.union_with_slice(&[6, 3, 2, 2, 4]);
        assert_eq!(ids.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_pir_node_ids_interner_shares_large_sets() {
        let mut interner = PirNodeIdsInterner::default();
        let first = interner.intern_slice(&[1, 2, 3, 4]);
        let second = interner.intern_slice(&[1, 2, 3, 4]);
        assert!(first.shares_storage_with(&second));
    }

    #[test]
    fn test_map_to_inputs_single_input() {
        // Use GateBuilder to create a single 4-bit input
        let mut gb = GateBuilder::new("test_fn".to_string(), GateBuilderOptions::opt());
        let input_vec = gb.add_input("in".to_string(), 4);
        // Add a dummy output (required by GateBuilder)
        gb.add_output("out".to_string(), input_vec.clone());
        let gate_fn = gb.build();

        // lsb 0: true
        // lsb 1: false
        // lsb 2: true
        // lsb 3: false
        let mut map = HashMap::new();
        for i in 0..4 {
            map.insert(input_vec.get_lsb(i).node, i % 2 == 0);
        }

        let inputs = gate_fn.map_to_inputs(map);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0], IrBits::make_ubits(4, 0b0101).unwrap());
    }

    #[test]
    fn test_map_to_inputs_multiple_inputs() {
        // Use GateBuilder to create two 3-bit inputs
        let mut gb = GateBuilder::new("test_fn_multi".to_string(), GateBuilderOptions::opt());
        let input0 = gb.add_input("in0".to_string(), 3);
        let input1 = gb.add_input("in1".to_string(), 3);
        // Add a dummy output (required by GateBuilder)
        gb.add_output("out0".to_string(), input0.clone());
        gb.add_output("out1".to_string(), input1.clone());
        let gate_fn = gb.build();

        // Set up a map for 6 bits: [1,0,1] for in0, [0,1,0] for in1
        let mut map = HashMap::new();
        for i in 0..3 {
            map.insert(input0.get_lsb(i).node, i % 2 == 0);
            map.insert(input1.get_lsb(i).node, i % 2 == 1);
        }
        let inputs = gate_fn.map_to_inputs(map);
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], IrBits::make_ubits(3, 0b101).unwrap());
        assert_eq!(inputs[1], IrBits::make_ubits(3, 0b010).unwrap());
    }
}
