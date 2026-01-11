// SPDX-License-Identifier: Apache-2.0

//! The `GateBuilder` is a builder for a `GateFn` -- it builds up the underlying
//! (AIG) data structure as operations are added.
//!
//! It tracks `Input` and `Output` nodes which are bundles of "primary input" /
//! "primary output" values.
//!
//! It can be created with "folding" (opportunistic simplification) on or off --
//! "off" is generally useful for testing there are no issues in the
//! simplification logic.
//!
//! Basic example usage:
//! ```
//! use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
//! use xlsynth_g8r::aig::{GateFn, AigBitVector, AigOperand};
//!
//! let mut builder = GateBuilder::new("my_and_gate".to_string(), GateBuilderOptions::opt());
//! let a: AigBitVector = builder.add_input("a".to_string(), 1);
//! let a0: &AigOperand = a.get_lsb(0);
//! let b: AigBitVector = builder.add_input("b".to_string(), 1);
//! let b0: &AigOperand = b.get_lsb(0);
//! let o0: AigOperand = builder.add_and_binary(*a0, *b0);
//! builder.add_output("o".to_string(), o0.into());
//! let gate_fn: GateFn = builder.build();
//! ```

use std::iter::zip;

use xlsynth::IrBits;

use crate::{
    aig::aig_hasher::AigHasher,
    aig::aig_simplify,
    aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn, Input, Output},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    Linear,
    Tree,
}

pub struct GateBuilder {
    pub name: String,
    pub gates: Vec<AigNode>,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub options: GateBuilderOptions,
    pub hasher: Option<AigHasher>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalfAdderOutput {
    pub sum: AigOperand,
    pub carry: AigOperand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullAdderOutput {
    pub sum: AigOperand,
    pub carry: AigOperand,
}

#[derive(Debug, Clone, Copy)]
pub struct GateBuilderOptions {
    pub fold: bool,
    pub hash: bool,
}

impl GateBuilderOptions {
    /// Returns a default "optimizing" `GateBuilderOptions` with folding and
    /// hashing enabled.
    pub fn opt() -> Self {
        Self {
            fold: true,
            hash: true,
        }
    }

    pub fn no_opt() -> Self {
        Self {
            fold: false,
            hash: false,
        }
    }
}

impl GateBuilder {
    pub fn new(name: String, options: GateBuilderOptions) -> Self {
        Self {
            name,
            gates: vec![AigNode::Literal(false)],
            inputs: Vec::new(),
            outputs: Vec::new(),
            options,
            hasher: if options.hash {
                Some(AigHasher::new())
            } else {
                None
            },
        }
    }

    pub fn build(self) -> GateFn {
        debug_assert!(
            !self.outputs.is_empty(),
            "GateBuilder::build: graph must have at least one output (degenerate/empty graph)"
        );
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
        // Debug assertion: all node indices in bit_vector must be in range
        for bit in bit_vector.iter_lsb_to_msb() {
            debug_assert!(
                bit.node.id < self.gates.len(),
                "add_output: Output node index out of bounds: {} (gates.len() = {})",
                bit.node.id,
                self.gates.len()
            );
        }
        self.outputs.push(Output { name, bit_vector });
    }

    pub fn replicate(&self, arg: AigOperand, bit_count: usize) -> AigBitVector {
        AigBitVector::from_lsb_is_index_0(&vec![arg; bit_count])
    }

    pub fn add_and_binary(&mut self, lhs: AigOperand, rhs: AigOperand) -> AigOperand {
        if self.options.fold {
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
        debug_assert!(
            self.gates.capacity() < 1024 * 1024,
            "gates capacity grew unexpectedly large: {}",
            self.gates.capacity()
        );
        self.gates.push(gate);
        if self.options.fold {
            if let Some(simplified) = aig_simplify::operand_simplify(gate_ref, self) {
                return simplified;
            }
        }
        if let Some(hasher) = &mut self.hasher {
            let existing = hasher.feed_ref(&gate_ref, &self.gates);
            if let Some(existing) = existing {
                return existing.into();
            }
        }
        AigOperand {
            node: gate_ref,
            negated: false,
        }
    }

    /// Returns the 3-input majority function (aka the full-adder carry):
    ///
    /// \(maj(a, b, c) = (a \& b) | (a \& c) | (b \& c)\).
    pub fn add_maj3(&mut self, a: AigOperand, b: AigOperand, c: AigOperand) -> AigOperand {
        let ab = self.add_and_binary(a, b);
        let ac = self.add_and_binary(a, c);
        let bc = self.add_and_binary(b, c);
        self.add_or_nary(&[ab, ac, bc], ReductionKind::Linear)
    }

    /// Emits a 1-bit half-adder.
    pub fn add_half_adder(&mut self, a: AigOperand, b: AigOperand) -> HalfAdderOutput {
        let sum = self.add_xor_binary(a, b);
        let carry = self.add_and_binary(a, b);
        HalfAdderOutput { sum, carry }
    }

    /// Emits a 1-bit full-adder.
    pub fn add_full_adder(
        &mut self,
        a: AigOperand,
        b: AigOperand,
        c: AigOperand,
    ) -> FullAdderOutput {
        let sum = self.add_xor_nary(&[a, b, c], ReductionKind::Linear);
        let carry = self.add_maj3(a, b, c);
        FullAdderOutput { sum, carry }
    }

    pub fn add_and_binary_nn(&mut self, lhs: AigOperand, rhs: AigOperand) -> AigOperand {
        let lhs_n = self.add_not(lhs);
        let rhs_n = self.add_not(rhs);
        self.add_and_binary(lhs_n, rhs_n)
    }

    pub fn add_nand_binary(&mut self, lhs: AigOperand, rhs: AigOperand) -> AigOperand {
        let and_result = self.add_and_binary(lhs, rhs);
        self.add_not(and_result)
    }

    pub fn add_and_nary(
        &mut self,
        args: &[AigOperand],
        reduction_kind: ReductionKind,
    ) -> AigOperand {
        if args.len() == 2 {
            return self.add_and_binary(args[0], args[1]);
        }
        if self.options.fold {
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
        if self.options.fold {
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

    /// Debug helper: checks that the given AigRef is in-bounds for this
    /// builder.
    pub fn validate_ref(&self, aig_ref: AigRef) {
        let count = self.gates.len();
        assert!(
            aig_ref.id < count,
            "AigRef out of bounds: {:?} (gates.len() = {})",
            aig_ref,
            count
        );
    }

    /// Debug helper: checks that the given AigOperand's node is in-bounds.
    pub fn validate_operand(&self, operand: AigOperand) {
        self.validate_ref(operand.node);
    }

    pub fn add_xnor(&mut self, a: AigOperand, b: AigOperand) -> AigOperand {
        let xor_gate_ref = self.add_xor_binary(a, b);
        self.add_not(xor_gate_ref)
    }

    pub fn add_xor_binary(&mut self, lhs: AigOperand, rhs: AigOperand) -> AigOperand {
        if self.options.fold {
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
        log::trace!(
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
        if self.options.fold {
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
        assert!(
            args.len() > 0,
            "add_or_nary; attempted to reduce an empty list of operands; reduction_kind: {:?}",
            reduction_kind
        );
        if args.len() == 1 {
            return args[0];
        }
        if self.options.fold {
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
        if self.options.fold {
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

    /// Returns true if the given AigRef is in-bounds for this builder.
    pub fn is_valid_ref(&self, aig_ref: AigRef) -> bool {
        aig_ref.id < self.gates.len()
    }

    /// Returns true if the given AigOperand's node is in-bounds.
    pub fn is_valid_operand(&self, operand: AigOperand) -> bool {
        self.is_valid_ref(operand.node)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        aig::get_summary_stats::{SummaryStats, get_summary_stats},
        aig_sim::gate_sim,
        check_equivalence,
    };

    use super::*;

    use pretty_assertions::assert_eq;
    use test_case::test_case;

    fn eval_1bit_output(gate_fn: &GateFn, a: bool, b: bool, c: Option<bool>) -> bool {
        let mut inputs = vec![IrBits::bool(a), IrBits::bool(b)];
        if let Some(c) = c {
            inputs.push(IrBits::bool(c));
        }
        let got = gate_sim::eval(gate_fn, &inputs, gate_sim::Collect::None);
        assert_eq!(got.outputs.len(), 1);
        got.outputs[0].get_bit(0).unwrap()
    }

    fn rust_maj3(a: bool, b: bool, c: bool) -> bool {
        (a && b) || (a && c) || (b && c)
    }

    #[test]
    fn test_add_maj3_truth_table() {
        let mut builder = GateBuilder::new("maj3".to_string(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);
        let c = builder.add_input("c".to_string(), 1);
        let maj = builder.add_maj3(*a.get_lsb(0), *b.get_lsb(0), *c.get_lsb(0));
        builder.add_output("maj".to_string(), AigBitVector::from_bit(maj));
        let gate_fn = builder.build();

        for aa in [false, true] {
            for bb in [false, true] {
                for cc in [false, true] {
                    let got = eval_1bit_output(&gate_fn, aa, bb, Some(cc));
                    let want = rust_maj3(aa, bb, cc);
                    assert_eq!(got, want, "a={} b={} c={}", aa, bb, cc);
                }
            }
        }
    }

    #[test]
    fn test_add_half_adder_truth_table() {
        let mut builder = GateBuilder::new("ha".to_string(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);
        let out = builder.add_half_adder(*a.get_lsb(0), *b.get_lsb(0));
        builder.add_output("sum".to_string(), AigBitVector::from_bit(out.sum));
        builder.add_output("carry".to_string(), AigBitVector::from_bit(out.carry));
        let gate_fn = builder.build();

        for aa in [false, true] {
            for bb in [false, true] {
                let inputs = vec![IrBits::bool(aa), IrBits::bool(bb)];
                let got = gate_sim::eval(&gate_fn, &inputs, gate_sim::Collect::None);
                assert_eq!(got.outputs.len(), 2);
                let got_sum = got.outputs[0].get_bit(0).unwrap();
                let got_carry = got.outputs[1].get_bit(0).unwrap();
                let want_sum = aa ^ bb;
                let want_carry = aa && bb;
                assert_eq!(
                    (got_sum, got_carry),
                    (want_sum, want_carry),
                    "a={} b={}",
                    aa,
                    bb
                );
            }
        }
    }

    #[test]
    fn test_add_full_adder_truth_table() {
        let mut builder = GateBuilder::new("fa".to_string(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);
        let c = builder.add_input("c".to_string(), 1);
        let out = builder.add_full_adder(*a.get_lsb(0), *b.get_lsb(0), *c.get_lsb(0));
        builder.add_output("sum".to_string(), AigBitVector::from_bit(out.sum));
        builder.add_output("carry".to_string(), AigBitVector::from_bit(out.carry));
        let gate_fn = builder.build();

        for aa in [false, true] {
            for bb in [false, true] {
                for cc in [false, true] {
                    let inputs = vec![IrBits::bool(aa), IrBits::bool(bb), IrBits::bool(cc)];
                    let got = gate_sim::eval(&gate_fn, &inputs, gate_sim::Collect::None);
                    assert_eq!(got.outputs.len(), 2);
                    let got_sum = got.outputs[0].get_bit(0).unwrap();
                    let got_carry = got.outputs[1].get_bit(0).unwrap();
                    let want_sum = aa ^ bb ^ cc;
                    let want_carry = rust_maj3(aa, bb, cc);
                    assert_eq!(
                        (got_sum, got_carry),
                        (want_sum, want_carry),
                        "a={} b={} c={}",
                        aa,
                        bb,
                        cc
                    );
                }
            }
        }
    }

    #[test]
    fn test_simple_and_to_string() {
        let mut builder = GateBuilder::new("my_and".to_string(), GateBuilderOptions::no_opt());
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
        let mut builder = GateBuilder::new("my_mux".to_string(), GateBuilderOptions::no_opt());
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
        let mut builder = GateBuilder::new("my_diamond".to_string(), GateBuilderOptions::no_opt());
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

        let topo = gate_fn.post_order_operands(true);

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
        let mut builder =
            GateBuilder::new("test_negated".to_string(), GateBuilderOptions::no_opt());

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
        let topo = gate_fn.post_order_operands(true);

        log::info!("topo: {:?}", topo);
        assert_eq!(topo.len(), 2);
        assert_eq!(topo[0], not_a);
        assert_eq!(topo[1], and_gate);
    }

    #[test]
    fn test_tree_reduce_and_3_wide() {
        let mut builder =
            GateBuilder::new("test_tree_reduce".to_string(), GateBuilderOptions::no_opt());
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
        let mut builder =
            GateBuilder::new("test_tree_reduce".to_string(), GateBuilderOptions::no_opt());
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
            let mut builder =
                GateBuilder::new("tree_reduce".to_string(), GateBuilderOptions::no_opt());
            let input = builder.add_input("input".to_string(), bits);
            let input_bits_vec: Vec<AigOperand> = input.iter_lsb_to_msb().map(|bit| *bit).collect();
            let result_tree = builder.tree_reduce(&input_bits_vec, &GateBuilder::add_and_binary);
            builder.add_output("o".to_string(), AigBitVector::from_bit(result_tree));
            builder.build()
        };
        let gate_fn_linear = {
            let mut builder =
                GateBuilder::new("linear_reduce".to_string(), GateBuilderOptions::no_opt());
            let input = builder.add_input("input".to_string(), bits);
            let input_bits_vec: Vec<AigOperand> = input.iter_lsb_to_msb().map(|bit| *bit).collect();
            let result_linear =
                builder.linear_reduce(&input_bits_vec, &GateBuilder::add_and_binary);
            builder.add_output("o".to_string(), AigBitVector::from_bit(result_linear));
            builder.build()
        };
        check_equivalence::prove_same_gate_fn_via_ir(&gate_fn_tree, &gate_fn_linear)
            .expect("tree and linear reduce should be equivalent");
    }

    #[test]
    fn test_get_msbs() {
        let mut builder =
            GateBuilder::new("test_get_msbs".to_string(), GateBuilderOptions::no_opt());
        let input = builder.add_input("input".to_string(), 4);
        let msb_slice = input.get_msbs(3);
        assert_eq!(msb_slice.get_bit_count(), 3);
        assert_eq!(msb_slice.get_lsb(0), input.get_lsb(1));
        assert_eq!(msb_slice.get_lsb(1), input.get_lsb(2));
        assert_eq!(msb_slice.get_lsb(2), input.get_lsb(3));
    }

    #[test]
    fn test_simple_and_dedupe() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut builder = GateBuilder::new(
            "test_simple_and_dedupe".to_string(),
            GateBuilderOptions {
                fold: false, // Keep folding off to isolate hashing effect
                hash: true,
            },
        );

        let a_vec = builder.add_input("a".to_string(), 1);
        let b_vec = builder.add_input("b".to_string(), 1);
        let a = *a_vec.get_lsb(0);
        let b = *b_vec.get_lsb(0);
        let not_a = builder.add_not(a);
        let not_b = builder.add_not(b);

        // Case 1: AND(a, b) vs AND(b, a)
        let ab = builder.add_and_binary(a, b);
        let ba = builder.add_and_binary(b, a);
        assert_eq!(ab.node.id, ba.node.id, "AND(a, b) vs AND(b, a)");
        assert_eq!(ab, ba);

        // Case 2: AND(a, !b) vs AND(!b, a)
        let a_notb = builder.add_and_binary(a, not_b);
        let notb_a = builder.add_and_binary(not_b, a);
        assert_eq!(a_notb.node.id, notb_a.node.id, "AND(a, !b) vs AND(!b, a)");
        assert_eq!(a_notb, notb_a);

        // Case 3: AND(!a, b) vs AND(b, !a)
        let nota_b = builder.add_and_binary(not_a, b);
        let b_nota = builder.add_and_binary(b, not_a);
        assert_eq!(nota_b.node.id, b_nota.node.id, "AND(!a, b) vs AND(b, !a)");
        assert_eq!(nota_b, b_nota);

        // Case 4: AND(!a, !b) vs AND(!b, !a)
        let nota_notb = builder.add_and_binary(not_a, not_b);
        let notb_nota = builder.add_and_binary(not_b, not_a);
        assert_eq!(
            nota_notb.node.id, notb_nota.node.id,
            "AND(!a, !b) vs AND(!b, !a)"
        );
        assert_eq!(nota_notb, notb_nota);

        // Let's also build the GateFn and check the number of AND gates.
        // Should be 1 literal (false), 2 inputs, and *4* distinct AND gates
        // (ab/ba, a_notb/notb_a, nota_b/b_nota, nota_notb/notb_nota).
        builder.add_output("ab".to_string(), ab.into());
        builder.add_output("a_notb".to_string(), a_notb.into());
        builder.add_output("nota_b".to_string(), nota_b.into());
        builder.add_output("nota_notb".to_string(), nota_notb.into());
        let gate_fn = builder.build();

        // Expected gates: Literal(false), Input(a), Input(b), And(a,b), And(a,!b),
        // And(!a,b), And(!a,!b)
        let stats: SummaryStats = get_summary_stats(&gate_fn);
        assert_eq!(stats.live_nodes, 6);
        assert_eq!(stats.deepest_path, 2);

        // Note: The specific node IDs (%3, %4, %5, %6) might vary depending on hash
        // implementation details, but the key is that the deduplicated outputs
        // refer to the same underlying AND gate IDs.
        let expected_str = "fn test_simple_and_dedupe(a: bits[1] = [%1], b: bits[1] = [%2]) -> (ab: bits[1] = [%3], a_notb: bits[1] = [%5], nota_b: bits[1] = [%7], nota_notb: bits[1] = [%9]) {
  %9 = and(not(a[0]), not(b[0]))
  %7 = and(not(a[0]), b[0])
  %5 = and(a[0], not(b[0]))
  %3 = and(a[0], b[0])
  ab[0] = %3
  a_notb[0] = %5
  nota_b[0] = %7
  nota_notb[0] = %9
}";
        assert_eq!(gate_fn.to_string(), expected_str);
    }
}
