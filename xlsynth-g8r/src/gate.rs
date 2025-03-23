// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, iter::zip};

use xlsynth::IrBits;

use crate::{aig_simplify, xls_ir::ir};

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
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

pub struct GateBuilder {
    pub name: String,
    pub gates: Vec<AigNode>,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub fold: bool,
}

pub struct GateFn {
    pub name: String,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub gates: Vec<AigNode>,
}

impl GateFn {
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

impl GateBuilder {
    pub fn new(name: String, fold: bool) -> Self {
        Self {
            name,
            gates: vec![AigNode::Literal(false)],
            inputs: Vec::new(),
            outputs: Vec::new(),
            fold,
        }
    }

    pub fn build(self) -> GateFn {
        GateFn {
            name: self.name,
            inputs: self.inputs,
            outputs: self.outputs,
            gates: self.gates,
        }
    }

    pub fn add_tag(&mut self, aig_ref: AigRef, tag: String) {
        let gate = &mut self.gates[aig_ref.id];
        gate.add_tag(tag)
    }

    pub fn to_operand(&self, aig_ref: AigRef) -> AigOperand {
        AigOperand {
            node: aig_ref,
            negated: false,
        }
    }

    pub fn get_false(&self) -> AigOperand {
        AigOperand {
            node: AigRef { id: 0 },
            negated: false,
        }
    }

    pub fn is_known_false(&self, operand: AigOperand) -> bool {
        operand.node.id == 0 && !operand.negated
    }

    pub fn get_true(&self) -> AigOperand {
        AigOperand {
            node: AigRef { id: 0 },
            negated: true,
        }
    }

    pub fn is_known_true(&self, operand: AigOperand) -> bool {
        operand.node.id == 0 && operand.negated
    }

    pub fn add_input(&mut self, name: String, bit_count: usize) -> AigBitVector {
        let mut bits: Vec<AigOperand> = Vec::new();
        for lsb_i in 0..bit_count {
            let gate_ref = AigRef {
                id: self.gates.len(),
            };
            self.gates.push(AigNode::Input {
                name: name.clone(),
                lsb_index: lsb_i,
            });
            bits.push(gate_ref.into());
        }
        let bit_vector = AigBitVector::from_lsb_is_index_0(&bits);
        self.inputs.push(Input {
            name,
            bit_vector: bit_vector.clone(),
        });
        bit_vector
    }

    pub fn add_output(&mut self, name: String, bit_vector: AigBitVector) {
        self.outputs.push(Output { name, bit_vector });
    }

    pub fn replicate(&self, arg: AigOperand, bit_count: usize) -> AigBitVector {
        AigBitVector::from_lsb_is_index_0(&vec![arg; bit_count])
    }

    pub fn add_and_binary(&mut self, lhs: AigOperand, rhs: AigOperand) -> AigOperand {
        if self.fold {
            // If either side is known false, the result is false.
            if self.is_known_false(lhs) || self.is_known_false(rhs) {
                return self.get_false();
            }
            // If both sides are known true, the result is true.
            if self.is_known_true(lhs) && self.is_known_true(rhs) {
                return self.get_true();
            }
            // If one side is known true, the result is the other side.
            if self.is_known_true(lhs) {
                return rhs;
            }
            if self.is_known_true(rhs) {
                return lhs;
            }
        }
        let gate = AigNode::And2 {
            a: lhs,
            b: rhs,
            tags: None,
        };
        let gate_ref = AigRef {
            id: self.gates.len(),
        };
        self.gates.push(gate);
        if self.fold {
            if let Some(simplified) = aig_simplify::operand_simplify(gate_ref, self) {
                return simplified;
            }
        }
        AigOperand {
            node: gate_ref,
            negated: false,
        }
    }

    pub fn add_and_binary_nn(&mut self, lhs: AigOperand, rhs: AigOperand) -> AigOperand {
        let lhs_n = self.add_not(lhs);
        let rhs_n = self.add_not(rhs);
        self.add_and_binary(lhs_n, rhs_n)
    }

    pub fn add_and_nary(
        &mut self,
        args: &[AigOperand],
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        if args.len() == 2 {
            return self.add_and_binary(args[0], args[1]);
        }
        if self.fold {
            if args.iter().any(|arg| self.is_known_false(*arg)) {
                return self.get_false();
            }
            if args.iter().all(|arg| self.is_known_true(*arg)) {
                return self.get_true();
            }
            if let &[lhs, rhs] = args {
                if self.is_known_true(lhs) {
                    return rhs;
                }
                if self.is_known_true(rhs) {
                    return lhs;
                }
            }
        }
        // Do a reduction of ands across the args.
        self.reduce(args, &GateBuilder::add_and_binary, reduction_kind)
    }

    pub fn add_and_reduce(
        &mut self,
        bit_vector: &AigBitVector,
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        self.add_and_nary(
            bit_vector
                .iter_lsb_to_msb()
                .cloned()
                .collect::<Vec<AigOperand>>()
                .as_slice(),
            reduction_kind,
        )
    }

    // Ands together bit i across all the args. That is, given:
    //
    //  a_2 :: a_1 :: a_0
    //  b_2 :: b_1 :: b_0
    //  c_2 :: c_1 :: c_0
    //
    //  add_and_vec_nary(&[a, b, c]) produces:
    //
    //  a_2 & b_2 & c_2 :: a_1 & b_1 & c_1 :: a_0 & b_0 & c_0
    pub fn add_and_vec_nary(
        &mut self,
        args: &[AigBitVector],
        reduction_kind: ReductionKind,
    ) -> AigBitVector {
        // Assert all args are the same bit count.
        for arg in args {
            assert_eq!(arg.get_bit_count(), args[0].get_bit_count());
        }
        let mut operands = Vec::new();
        for i in 0..args[0].get_bit_count() {
            let bit_i_gates: Vec<AigOperand> =
                args.iter().map(|arg| arg.get_lsb(i)).cloned().collect();
            operands.push(self.add_and_nary(&bit_i_gates, reduction_kind));
        }
        AigBitVector::from_lsb_is_index_0(&operands)
    }

    pub fn add_and_vec(&mut self, a: &AigBitVector, b: &AigBitVector) -> AigBitVector {
        assert_eq!(a.get_bit_count(), b.get_bit_count());
        let results = zip(a.iter_lsb_to_msb(), b.iter_lsb_to_msb())
            .map(|(a, b)| self.add_and_binary(*a, *b))
            .collect::<Vec<_>>();
        AigBitVector::from_lsb_is_index_0(&results)
    }

    pub fn add_literal(&mut self, value: &IrBits) -> AigBitVector {
        let mut operands = Vec::new();
        for i in 0..value.get_bit_count() {
            if value.get_bit(i).unwrap() {
                operands.push(self.get_true());
            } else {
                operands.push(self.get_false());
            }
        }
        AigBitVector::from_lsb_is_index_0(&operands)
    }

    pub fn add_not(&mut self, arg: AigOperand) -> AigOperand {
        if self.fold {
            if self.is_known_false(arg) {
                return self.get_true();
            }
            if self.is_known_true(arg) {
                return self.get_false();
            }
            if arg.negated {
                return AigOperand {
                    node: arg.node,
                    negated: false,
                };
            }
        }
        AigOperand {
            node: arg.node,
            negated: !arg.negated,
        }
    }

    pub fn add_xnor(&mut self, a: AigOperand, b: AigOperand) -> AigOperand {
        let xor_gate_ref = self.add_xor_binary(a, b);
        self.add_not(xor_gate_ref)
    }

    pub fn add_xor_binary(&mut self, lhs: AigOperand, rhs: AigOperand) -> AigOperand {
        if self.fold {
            // both sides known
            if self.is_known_true(lhs) && self.is_known_true(rhs) {
                return self.get_false();
            }
            if self.is_known_true(lhs) && self.is_known_false(rhs) {
                return self.get_true();
            }
            if self.is_known_false(lhs) && self.is_known_true(rhs) {
                return self.get_true();
            }
            if self.is_known_false(lhs) && self.is_known_false(rhs) {
                return self.get_false();
            }
            // one side known
            if self.is_known_false(lhs) {
                return rhs;
            }
            if self.is_known_false(rhs) {
                return lhs;
            }
            if self.is_known_true(lhs) {
                return self.add_not(rhs);
            }
            if self.is_known_true(rhs) {
                return self.add_not(lhs);
            }
        }
        // the formula for xor is (~a & b) | (a & ~b)
        // so in terms of only and gates it's:
        // ~(~(~a & b) & ~(a & ~b))
        let a = lhs;
        let b = rhs;
        let not_a = self.add_not(a);
        let not_b = self.add_not(b);
        let and_lhs = self.add_and_binary(not_a, b);
        let and_rhs = self.add_and_binary(a, not_b);
        let outer_and = self.add_and_binary_nn(and_lhs, and_rhs);
        self.add_not(outer_and)
    }

    pub fn add_xor_nary(
        &mut self,
        args: &[AigOperand],
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        self.reduce(args, &GateBuilder::add_xor_binary, reduction_kind)
    }

    pub fn linear_reduce<F>(&mut self, args: &[AigOperand], f: &F) -> AigOperand
    where
        F: Fn(&mut Self, AigOperand, AigOperand) -> AigOperand,
    {
        assert!(
            args.len() > 0,
            "attempted to reduce an empty list of operands"
        );
        let mut accum = args[0];
        for i in 1..args.len() {
            accum = f(self, accum, args[i]);
        }
        accum
    }

    pub fn tree_reduce<F>(&mut self, args: &[AigOperand], f: &F) -> AigOperand
    where
        F: Fn(&mut Self, AigOperand, AigOperand) -> AigOperand,
    {
        assert!(
            args.len() > 0,
            "attempted to reduce an empty list of operands"
        );
        if args.len() == 1 {
            return args[0];
        }
        let halves = args.split_at(args.len() / 2);
        log::info!(
            "tree_reduce; orig: {} lhs: {} rhs: {}",
            args.len(),
            halves.0.len(),
            halves.1.len()
        );
        let first_half = self.tree_reduce(halves.0, f);
        let second_half = self.tree_reduce(halves.1, f);
        f(self, first_half, second_half)
    }

    pub fn reduce<F>(
        &mut self,
        args: &[AigOperand],
        f: &F,
        reduction_kind: ReductionKind,
    ) -> AigOperand
    where
        F: Fn(&mut Self, AigOperand, AigOperand) -> AigOperand,
    {
        match reduction_kind {
            ReductionKind::Linear => self.linear_reduce(args, f),
            ReductionKind::Tree => self.tree_reduce(args, f),
        }
    }

    pub fn add_not_vec(&mut self, args: &AigBitVector) -> AigBitVector {
        AigBitVector::from_lsb_is_index_0(
            &args
                .iter_lsb_to_msb()
                .map(|arg| self.add_not(*arg))
                .collect::<Vec<_>>(),
        )
    }

    pub fn add_xnor_vec(&mut self, a: &AigBitVector, b: &AigBitVector) -> AigBitVector {
        assert_eq!(a.get_bit_count(), b.get_bit_count());
        let mut operands = Vec::new();
        for i in 0..a.get_bit_count() {
            let operand = self.add_xnor(*a.get_lsb(i), *b.get_lsb(i));
            operands.push(operand);
        }
        AigBitVector::from_lsb_is_index_0(&operands)
    }

    pub fn add_or_binary(&mut self, lhs: AigOperand, rhs: AigOperand) -> AigOperand {
        if self.fold {
            if self.is_known_true(lhs) || self.is_known_true(rhs) {
                return self.get_true();
            }
            if self.is_known_false(lhs) && self.is_known_false(rhs) {
                return self.get_false();
            }
            if self.is_known_false(lhs) {
                return rhs;
            }
            if self.is_known_false(rhs) {
                return lhs;
            }
        }
        let not_lhs = self.add_not(lhs);
        let not_rhs = self.add_not(rhs);
        let and = self.add_and_binary(not_lhs, not_rhs);
        self.add_not(and)
    }

    // Performs an `or` across all the gates given in `args` to produce a single bit
    // output.
    pub fn add_or_nary(
        &mut self,
        args: &[AigOperand],
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        if self.fold {
            if args.iter().any(|arg| self.is_known_true(*arg)) {
                return self.get_true();
            }
            // Drop any args that are known zero.
            let nonzero: Vec<AigOperand> = args
                .iter()
                .filter(|arg| !self.is_known_false(**arg))
                .map(|arg| *arg)
                .collect();
            if nonzero.is_empty() {
                return self.get_false();
            }
            if nonzero.len() == 1 {
                return nonzero[0];
            }
            if nonzero.len() < args.len() {
                return self.add_or_nary(&nonzero, reduction_kind);
            }
        }
        if let &[lhs, rhs] = args {
            return self.add_or_binary(lhs, rhs);
        }
        // Do a reduction of ors across the args.
        self.reduce(args, &GateBuilder::add_or_binary, reduction_kind)
    }

    pub fn add_or_vec(&mut self, a: &AigBitVector, b: &AigBitVector) -> AigBitVector {
        assert_eq!(a.get_bit_count(), b.get_bit_count());
        let results = zip(a.iter_lsb_to_msb(), b.iter_lsb_to_msb())
            .map(|(a, b)| self.add_or_binary(*a, *b))
            .collect::<Vec<_>>();
        AigBitVector::from_lsb_is_index_0(&results)
    }

    pub fn add_or_reduce(
        &mut self,
        bit_vector: &AigBitVector,
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        self.add_or_nary(
            bit_vector
                .iter_lsb_to_msb()
                .cloned()
                .collect::<Vec<AigOperand>>()
                .as_slice(),
            reduction_kind,
        )
    }

    /// Returns a bit that indicates if any of the bits in `bit_vector` are
    /// non-zero.
    pub fn add_nez(
        &mut self,
        bit_vector: &AigBitVector,
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        assert!(
            bit_vector.get_bit_count() > 0,
            "attempted to determine if empty bit-vector was != 0"
        );
        self.add_or_reduce(bit_vector, reduction_kind)
    }

    /// Returns a bit that indicates if all the bits in `bit_vector` are zero.
    pub fn add_ez(
        &mut self,
        bit_vector: &AigBitVector,
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        let or_reduced = self.add_or_reduce(bit_vector, reduction_kind);
        self.add_not(or_reduced)
    }

    pub fn add_ez_slice(
        &mut self,
        slice: &[AigOperand],
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        let or_reduced = self.add_or_nary(slice, reduction_kind);
        self.add_not(or_reduced)
    }

    pub fn add_or_vec_nary(
        &mut self,
        args: &[AigBitVector],
        reduction_kind: ReductionKind,
    ) -> AigBitVector {
        let bit_count = args[0].get_bit_count();
        // Assert all vectors are the same length -- we're going to or-reduce the bit
        // positions.
        for arg in args {
            assert_eq!(arg.get_bit_count(), bit_count);
        }
        let mut operands = Vec::new();
        for i in 0..args[0].get_bit_count() {
            // Get the ith bit from each arg.
            let bit_i_gates: Vec<AigOperand> = args
                .iter()
                .map(|arg| arg.get_lsb(i))
                .cloned()
                .collect::<Vec<AigOperand>>();
            // Perform an `or` across those bits.
            operands.push(self.add_or_nary(bit_i_gates.as_slice(), reduction_kind));
        }
        AigBitVector::from_lsb_is_index_0(&operands)
    }

    // Bitwise xor between two bit vectors that must be the same bit count.
    pub fn add_xor_vec(&mut self, a: &AigBitVector, b: &AigBitVector) -> AigBitVector {
        assert_eq!(a.get_bit_count(), b.get_bit_count());
        AigBitVector::from_lsb_is_index_0(
            zip(a.iter_lsb_to_msb(), b.iter_lsb_to_msb())
                .map(|(a, b)| self.add_xor_binary(*a, *b))
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    pub fn add_xor_vec_nary(
        &mut self,
        args: &[AigBitVector],
        reduction_kind: ReductionKind,
    ) -> AigBitVector {
        // Assert all vectors are the same length -- we're going to xor-reduce the bit
        // positions.
        for arg in args {
            assert_eq!(arg.get_bit_count(), args[0].get_bit_count());
        }
        let mut gates = Vec::new();
        for i in 0..args[0].get_bit_count() {
            let bit_i_gates: Vec<AigOperand> = args
                .iter()
                .map(|arg| arg.get_lsb(i))
                .cloned()
                .collect::<Vec<AigOperand>>();
            gates.push(self.add_xor_nary(&bit_i_gates, reduction_kind));
        }
        AigBitVector::from_lsb_is_index_0(&gates)
    }

    pub fn add_xor_reduce(
        &mut self,
        bit_vector: &AigBitVector,
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        self.add_xor_nary(
            bit_vector
                .iter_lsb_to_msb()
                .cloned()
                .collect::<Vec<AigOperand>>()
                .as_slice(),
            reduction_kind,
        )
    }

    pub fn add_mux2(
        &mut self,
        selector: AigOperand,
        on_true: AigOperand,
        on_false: AigOperand,
    ) -> AigOperand {
        if self.fold {
            if self.is_known_false(selector) {
                return on_false;
            }
            if self.is_known_true(selector) {
                return on_true;
            }
            if self.is_known_true(on_true) && self.is_known_false(on_false) {
                return selector;
            }
            if self.is_known_false(on_true) && self.is_known_true(on_false) {
                return self.add_not(selector);
            }
        }
        let not_selector = self.add_not(selector);
        let selector_off_result = self.add_and_binary(not_selector, on_false);
        let selector_on_result = self.add_and_binary(selector, on_true);
        self.add_or_binary(selector_off_result, selector_on_result)
    }

    pub fn add_mux2_vec(
        &mut self,
        selector: &AigOperand,
        on_true: &AigBitVector,
        on_false: &AigBitVector,
    ) -> AigBitVector {
        // Assert cases are equal length vectors.
        assert_eq!(on_true.get_bit_count(), on_false.get_bit_count());

        let mut bits = Vec::new();
        for (on_true_bit, on_false_bit) in
            zip(on_true.iter_lsb_to_msb(), on_false.iter_lsb_to_msb())
        {
            let bit = self.add_mux2(*selector, *on_true_bit, *on_false_bit);
            bits.push(bit);
        }
        AigBitVector::from_lsb_is_index_0(&bits)
    }

    // Returns whether the two bit-vectors are equal (eq) as a single bit value.
    pub fn add_eq_vec(
        &mut self,
        a: &AigBitVector,
        b: &AigBitVector,
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        assert_eq!(a.get_bit_count(), b.get_bit_count());
        let xnors = self.add_xnor_vec(a, b);
        self.add_and_reduce(&xnors, reduction_kind)
    }

    // Returns whether the two bit-vectors are not equal (ne) as a single bit value.
    pub fn add_ne_vec(
        &mut self,
        a: &AigBitVector,
        b: &AigBitVector,
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        assert_eq!(a.get_bit_count(), b.get_bit_count());
        let xors = self.add_xor_vec(a, b);
        assert_eq!(xors.get_bit_count(), a.get_bit_count());
        self.add_or_reduce(&xors, reduction_kind) // or-reduce to see if any bit
                                                  // was different
    }
}

#[cfg(test)]
mod tests {
    use crate::check_equivalence;

    use super::*;

    use pretty_assertions::assert_eq;
    use test_case::test_case;

    #[test]
    fn test_simple_and_to_string() {
        let mut builder = GateBuilder::new("my_and".to_string(), false);
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);
        let a0 = a.get_lsb(0);
        let b0 = b.get_lsb(0);
        let and = builder.add_and_binary(*a0, *b0);
        builder.add_output("o".to_string(), AigBitVector::from_bit(and));
        let gate_fn = builder.build();
        assert_eq!(
            gate_fn.to_string(),
            "fn my_and(a: bits[1] = [%1], b: bits[1] = [%2]) -> (o: bits[1] = [%3]) {
  %3 = and(a[0], b[0])
  o[0] = %3
}"
        );
    }

    #[test]
    fn test_one_bit_mux_vec() {
        let mut builder = GateBuilder::new("my_mux".to_string(), false);
        let selector = builder.add_input("selector".to_string(), 1);
        let on_true = builder.add_input("on_true".to_string(), 1);
        let on_false = builder.add_input("on_false".to_string(), 1);

        let selector0 = selector.get_lsb(0);

        let muxes = builder.add_mux2_vec(selector0, &on_true, &on_false);
        assert_eq!(muxes.get_bit_count(), 1);
        assert_eq!(muxes.get_lsb(0).node.id, 6);

        builder.add_output("o".to_string(), muxes);

        let gate_fn = builder.build();
        assert_eq!(
            gate_fn.to_string(),
            "fn my_mux(selector: bits[1] = [%1], on_true: bits[1] = [%2], on_false: bits[1] = [%3]) -> (o: bits[1] = [not(%6)]) {
  %4 = and(not(selector[0]), on_false[0])
  %5 = and(selector[0], on_true[0])
  %6 = and(not(%4), not(%5))
  o[0] = not(%6)
}"
        );
    }

    // Builds a diamong out of simple and gates and ensures that:
    // - the nodes show up in topological order
    // - the nodes show up at most once
    #[test]
    fn test_topo_simple_diamond() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut builder = GateBuilder::new("my_diamond".to_string(), false);
        let a = builder.add_input("a".to_string(), 1);
        let a0 = a.get_lsb(0);
        let b = builder.add_input("b".to_string(), 1);
        let b0 = b.get_lsb(0);
        let c = builder.add_input("c".to_string(), 1);
        let c0 = c.get_lsb(0);
        let d = builder.add_input("d".to_string(), 1);
        let d0 = d.get_lsb(0);
        let e = builder.add_input("e".to_string(), 1);
        let e0 = e.get_lsb(0);
        let not_e0 = AigOperand {
            node: e0.node,
            negated: true,
        };

        let ab = builder.add_and_binary(*a0, *b0);
        let cd = builder.add_and_binary(*c0, *d0);
        let ab_or_cd = builder.add_and_binary(ab, cd);
        let e_and_ab_or_cd = builder.add_and_binary(not_e0, ab_or_cd);
        builder.add_output("o".to_string(), AigBitVector::from_bit(e_and_ab_or_cd));
        let gate_fn = builder.build();

        let topo = gate_fn.post_order(true);

        log::info!("gate_fn:\n{}", gate_fn.to_string());
        log::info!("topo: {:?}", topo);

        assert_eq!(topo.len(), 5);
        assert_eq!(topo[0], not_e0);
        assert_eq!(topo[1], ab);
        assert_eq!(topo[2], cd);
        assert_eq!(topo[3], ab_or_cd);
        assert_eq!(topo[4], e_and_ab_or_cd);

        assert_eq!(gate_fn.to_string(), "fn my_diamond(a: bits[1] = [%1], b: bits[1] = [%2], c: bits[1] = [%3], d: bits[1] = [%4], e: bits[1] = [%5]) -> (o: bits[1] = [%9]) {
  %6 = and(a[0], b[0])
  %7 = and(c[0], d[0])
  %8 = and(%6, %7)
  %9 = and(not(e[0]), %8)
  o[0] = %9
}");
    }

    #[test]
    fn test_topo_with_negated_input() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut builder = GateBuilder::new("test_negated".to_string(), false);

        // Create a single input 'a'
        let a = builder.add_input("a".to_string(), 1);
        let a0 = a.get_lsb(0);

        // Create two gates:
        // 1. One that uses 'a' directly
        // 2. One that uses NOT(a)
        let not_a = builder.add_not(*a0);
        let and_gate = builder.add_and_binary(*a0, not_a);

        builder.add_output("o".to_string(), AigBitVector::from_bit(and_gate));
        let gate_fn = builder.build();

        // The resulting gate function should look like:
        assert_eq!(
            gate_fn.to_string(),
            "fn test_negated(a: bits[1] = [%1]) -> (o: bits[1] = [%2]) {
  %2 = and(a[0], not(a[0]))
  o[0] = %2
}"
        );

        // Get the topological order
        let topo = gate_fn.post_order(true);

        log::info!("topo: {:?}", topo);
        assert_eq!(topo.len(), 2);
        assert_eq!(topo[0], not_a);
        assert_eq!(topo[1], and_gate);
    }

    #[test]
    fn test_tree_reduce_and_3_wide() {
        let mut builder = GateBuilder::new("test_tree_reduce".to_string(), false);
        let input = builder.add_input("input".to_string(), 3);
        let input_bits_vec: Vec<AigOperand> = input.iter_lsb_to_msb().map(|bit| *bit).collect();
        let result = builder.tree_reduce(&input_bits_vec, &GateBuilder::add_and_binary);
        builder.add_output("o".to_string(), AigBitVector::from_bit(result));
        let gate_fn = builder.build();
        assert_eq!(
            gate_fn.to_string(),
            "fn test_tree_reduce(input: bits[3] = [%1, %2, %3]) -> (o: bits[1] = [%5]) {
  %4 = and(input[1], input[2])
  %5 = and(input[0], %4)
  o[0] = %5
}"
        );
    }

    #[test]
    fn test_tree_reduce_and_4_wide() {
        let mut builder = GateBuilder::new("test_tree_reduce".to_string(), false);
        let input = builder.add_input("input".to_string(), 4);
        let input_bits_vec: Vec<AigOperand> = input.iter_lsb_to_msb().map(|bit| *bit).collect();
        let result = builder.tree_reduce(&input_bits_vec, &GateBuilder::add_and_binary);
        builder.add_output("o".to_string(), AigBitVector::from_bit(result));
        let gate_fn = builder.build();
        assert_eq!(
            gate_fn.to_string(),
            "fn test_tree_reduce(input: bits[4] = [%1, %2, %3, %4]) -> (o: bits[1] = [%7]) {
  %5 = and(input[0], input[1])
  %6 = and(input[2], input[3])
  %7 = and(%5, %6)
  o[0] = %7
}"
        );
    }

    #[test_case(1)]
    #[test_case(2)]
    #[test_case(3)]
    #[test_case(4)]
    #[test_case(5)]
    #[test_case(6)]
    #[test_case(7)]
    #[test_case(8)]
    fn test_tree_reduce_eq_linear_reduce(bits: usize) {
        let _ = env_logger::builder().is_test(true).try_init();
        let gate_fn_tree = {
            let mut builder = GateBuilder::new("tree_reduce".to_string(), false);
            let input = builder.add_input("input".to_string(), bits);
            let input_bits_vec: Vec<AigOperand> = input.iter_lsb_to_msb().map(|bit| *bit).collect();
            let result_tree = builder.tree_reduce(&input_bits_vec, &GateBuilder::add_and_binary);
            builder.add_output("o".to_string(), AigBitVector::from_bit(result_tree));
            builder.build()
        };
        let gate_fn_linear = {
            let mut builder = GateBuilder::new("linear_reduce".to_string(), false);
            let input = builder.add_input("input".to_string(), bits);
            let input_bits_vec: Vec<AigOperand> = input.iter_lsb_to_msb().map(|bit| *bit).collect();
            let result_linear =
                builder.linear_reduce(&input_bits_vec, &GateBuilder::add_and_binary);
            builder.add_output("o".to_string(), AigBitVector::from_bit(result_linear));
            builder.build()
        };
        check_equivalence::validate_same_gate_fn(&gate_fn_tree, &gate_fn_linear)
            .expect("tree and linear reduce should be equivalent");
    }

    #[test]
    fn test_get_msbs() {
        let mut builder = GateBuilder::new("test_get_msbs".to_string(), false);
        let input = builder.add_input("input".to_string(), 4);
        let msb_slice = input.get_msbs(3);
        assert_eq!(msb_slice.get_bit_count(), 3);
        assert_eq!(msb_slice.get_lsb(0), input.get_lsb(1));
        assert_eq!(msb_slice.get_lsb(1), input.get_lsb(2));
        assert_eq!(msb_slice.get_lsb(2), input.get_lsb(3));
    }
}
