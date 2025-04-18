// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use bitvec::vec::BitVec;
use xlsynth::{IrBits, IrValue};

use crate::xls_ir::ir;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct AigRef {
    pub id: usize,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
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
        if self.negated {
            None
        } else {
            Some(self.node)
        }
    }

    pub fn to_formula_string(&self, nodes: &[AigNode]) -> String {
        if self.negated {
            format!("not({})", nodes[self.node.id].to_formula_string(nodes))
        } else {
            nodes[self.node.id].to_formula_string(nodes)
        }
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

#[derive(Debug, Clone)]
pub enum AigNode {
    Input {
        name: String,
        /// Index where 0 is the least significant bit of the input.
        lsb_index: usize,
    },
    Literal(bool),
    And2 {
        a: AigOperand,
        b: AigOperand,
        tags: Option<Vec<String>>,
    },
}

impl AigNode {
    pub fn to_formula_string(&self, nodes: &[AigNode]) -> String {
        match self {
            AigNode::Input { name, lsb_index } => format!("{}[{}]", name, lsb_index),
            AigNode::Literal(value) => format!("{}", value),
            AigNode::And2 { a, b, .. } => format!(
                "and({},{})",
                a.to_formula_string(nodes),
                b.to_formula_string(nodes)
            ),
        }
    }

    pub fn get_operands(&self) -> Vec<AigOperand> {
        match self {
            AigNode::Input { .. } => vec![],
            AigNode::Literal(_) => vec![],
            AigNode::And2 { a, b, .. } => vec![a.clone(), b.clone()],
        }
    }

    pub fn get_args(&self) -> Vec<AigRef> {
        match self {
            AigNode::Input { .. } => vec![],
            AigNode::Literal(_) => vec![],
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

#[derive(Debug, Clone)]
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
            Err(format!("expected a single operand for AigBitVector::try_into<AigOperand>(), but got {} operands", self.operands.len()))
        } else {
            Ok(self.operands[0])
        }
    }
}

#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct GateFn {
    pub name: String,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub gates: Vec<AigNode>,
}

fn ir_bits_from_bitvec(bv: BitVec) -> IrBits {
    let mut bv_str: String = format!("bits[{}]:0b", bv.len());
    for b in bv {
        bv_str.push(if b { '1' } else { '0' });
    }
    IrValue::parse_typed(&bv_str).unwrap().to_bits().unwrap()
}

impl GateFn {
    /// Collapses a sparse mapping of AigRefs to boolean values into a vector of
    /// IrBits that can be fed to gate / IR simulation as input stimulus.
    pub fn map_to_inputs(&self, map: HashMap<AigRef, bool>) -> Vec<IrBits> {
        let mut bitvecs: Vec<BitVec> = self
            .inputs
            .iter()
            .map(|i| BitVec::repeat(false, i.get_bit_count()))
            .collect();
        for (input_num, input) in self.inputs.iter().enumerate() {
            for bit_index in 0..input.get_bit_count() {
                let aig_ref = input.bit_vector.get_lsb(bit_index).non_negated().unwrap();
                let bit_value = map.get(&aig_ref).expect(&format!(
                    "all input gates should be present in provided sparse map; missing: {:?}",
                    aig_ref
                ));
                bitvecs[input_num].set(bit_index, *bit_value);
            }
        }
        bitvecs
            .into_iter()
            .map(|bv| ir_bits_from_bitvec(bv))
            .collect()
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
                AigNode::Input { name, lsb_index } => format!("{}[{}]", name, lsb_index),
                _ => format!("%{}", id),
            }
        };

        s.push_str(&format!(
            "fn {}({input_str}) -> {output_str} {{\n",
            self.name
        ));
        for operand in self.post_order(true) {
            let this_node_id = operand.node.id;
            let this_node = self.get(operand.node);
            match this_node {
                AigNode::And2 { a, b, tags } => {
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
                        this_node_id, a_str, b_str, tags_str
                    ));
                }
                AigNode::Input { .. } => {
                    continue;
                }
                AigNode::Literal(value) => {
                    s.push_str(&format!("  %{} = literal({})\n", this_node_id, value));
                }
            }
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

    // In post-order a node comes after all of its dependencies, i.e. return values
    // are last.
    pub fn post_order(&self, discard_inputs: bool) -> Vec<AigOperand> {
        let mut seen = HashSet::new();
        let mut order = Vec::new();
        for output in &self.outputs {
            for bit in output.bit_vector.iter_lsb_to_msb() {
                post_order(bit, self, discard_inputs, &mut seen, &mut order);
            }
        }

        order
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    Linear,
    Tree,
}

/// Helper for getting the post-order of the nodes in the AIG.
///
/// If operand is already seen, immediately returns.
/// Otherwise, does any required traversal on operands and then adds the operand
/// to the order.
///
/// Note that this is a recursive implementation for simplicity though a
/// worklist oriented algorithm will scale better in the future.
fn post_order(
    operand: &AigOperand,
    f: &GateFn,
    discard_inputs: bool,
    seen: &mut HashSet<AigOperand>,
    order: &mut Vec<AigOperand>,
) {
    if !seen.insert(*operand) {
        return;
    }
    let gate = f.get(operand.node);
    match gate {
        AigNode::Input { .. } => {
            let should_push = operand.negated || !discard_inputs;
            if should_push {
                order.push(*operand);
            }
        }
        AigNode::Literal(_) => {
            order.push(*operand);
        }
        AigNode::And2 { a, b, .. } => {
            post_order(a, f, discard_inputs, seen, order);
            post_order(b, f, discard_inputs, seen, order);
            order.push(*operand);
        }
    }
}

/// Extracts the combined transitive fan-in cone for a set of nodes.
///
/// Returns:
/// * the set of all gates within the cones (this is given in a deterministic
///   topological ordering, as that can often be more useful than getting the
///   set and the caller needing to reconstruct the ordering).
/// * the set of primary inputs feeding the cones.
pub fn extract_cone(start_nodes: &[AigRef], gates: &[AigNode]) -> (Vec<AigRef>, HashSet<AigRef>) {
    let mut cone_gates_set = HashSet::new();
    let mut cone_gates = Vec::new();
    let mut cone_inputs = HashSet::new();
    let mut visited = HashSet::new();
    let mut worklist: Vec<AigRef> = start_nodes.to_vec();

    let mut add_cone_gate = |aig_ref: AigRef| {
        if cone_gates_set.insert(aig_ref) {
            cone_gates.push(aig_ref);
        }
    };

    while let Some(current_ref) = worklist.pop() {
        if !visited.insert(current_ref) {
            // Already visited or WIP
            continue;
        }

        let node = &gates[current_ref.id];

        match node {
            AigNode::Input { .. } => {
                cone_inputs.insert(current_ref);
            }
            AigNode::Literal(_) => {
                add_cone_gate(current_ref);
            }
            AigNode::And2 { a, b, .. } => {
                add_cone_gate(current_ref);
                worklist.push(a.node);
                worklist.push(b.node);
            }
        }
    }

    (cone_gates, cone_inputs)
}

#[cfg(test)]
mod tests {
    use super::extract_cone;
    use crate::test_utils::setup_simple_graph;
    use std::collections::HashSet;

    /// Tests extracting a single cone.
    #[test]
    fn test_extract_single_cone() {
        let test_data = setup_simple_graph();
        let (cone_gates, cone_inputs) = extract_cone(&[test_data.o.node], &test_data.g.gates);
        // The order depends on the DFS traversal (LIFO worklist)
        assert_eq!(
            cone_gates,
            vec![test_data.o.node, test_data.b.node, test_data.a.node]
        );
        assert_eq!(
            cone_inputs,
            HashSet::from([test_data.i0.node, test_data.i1.node, test_data.i2.node])
        );
    }

    /// Tests extracting the union of two overlapping cones.
    #[test]
    fn test_extract_overlapping_cones() {
        let test_data = setup_simple_graph();
        let start_nodes = vec![test_data.o.node, test_data.c.node];
        let (cone_gates, cone_inputs) = extract_cone(&start_nodes, &test_data.g.gates);

        let expected_gates = vec![
            test_data.c.node,
            test_data.o.node,
            test_data.b.node,
            test_data.a.node,
        ];
        assert_eq!(cone_gates, expected_gates);

        let expected_inputs = HashSet::from([
            test_data.i0.node,
            test_data.i1.node,
            test_data.i2.node,
            test_data.i3.node,
        ]);
        assert_eq!(cone_inputs, expected_inputs);
    }
}
