// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::mem;

use xlsynth::{IrBits, IrValue};
use xlsynth_pir::ir::{Binop, Fn as IrFn, NaryOp, Node, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils::{compute_users, remap_payload_with};

/// Kinds of PIR transforms used in PIR MCMC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum PirTransformKind {
    /// Swap the operands of a commutative binary operator (currently `add`).
    SwapCommutativeBinopOperands,
    /// Clone a node that has multiple users so that each user gets its own
    /// replica node.
    CloneMultiUserNode,
    /// Distributes equality/inequality over a select (and the reverse folding
    /// form):
    ///
    /// - `eq(sel(p,a,b),c) ↔ sel(p,eq(a,c),eq(b,c))`
    /// - `ne(sel(p,a,b),c) ↔ sel(p,ne(a,c),ne(b,c))`
    EqSelDistribute,
    /// Constant-shift equality through add-with-literal (mod 2^w):
    ///   `eq(add(x, k), c) ↔ eq(x, c - k)` (and same for `ne`)
    EqNeAddLiteralShift,
    /// Distribute NOT over select (and reverse folding form):
    /// `not(sel(p, cases=[a, b])) ↔ sel(p, cases=[not(a), not(b)])`
    NotSelDistribute,
    /// Distribute NEG over select (and reverse folding form):
    /// `neg(sel(p, cases=[a, b])) ↔ sel(p, cases=[neg(a), neg(b)])`
    NegSelDistribute,
    /// Distribute bit_slice over select (and reverse folding form):
    /// `bit_slice(sel(p, cases=[a, b]), start=s, width=w)
    ///    ↔ sel(p, cases=[bit_slice(a,s,w), bit_slice(b,s,w)])`
    BitSliceSelDistribute,
    /// Distribute sign_ext over select (and reverse folding form):
    /// `sign_ext(sel(p, cases=[a, b]), new_bit_count=n)
    ///    ↔ sel(p, cases=[sign_ext(a,n), sign_ext(b,n)])`
    SignExtSelDistribute,
    /// Convert 1-bit priority_sel to sel (and reverse):
    /// `priority_sel(p:bits[1], cases=[a], default=b) ↔ sel(p, cases=[b, a])`
    PrioritySel1ToSel,
    /// Treat `sign_ext(b)` (b:bits[1]) as an all-ones/zeros mask and convert
    /// to/from sel: `and(x, sign_ext(b,w)) ↔ sel(b, cases=[0_w, x])`
    AndMaskSignExtToSel,
    /// `xor(x, sign_ext(b,w))` is a conditional invert; convert to/from sel:
    /// `xor(x, sign_ext(b,w)) ↔ sel(b, cases=[x, not(x)])`
    XorMaskSignExtToSelNot,
    /// Fold `sel` when both cases are identical:
    /// `sel(p, cases=[a, a]) ↔ a`
    SelSameArmsFold,
    /// Swap `sel` arms by negating predicate (2-case `sel` only):
    /// `sel(not(p), cases=[a, b]) ↔ sel(p, cases=[b, a])`
    SelSwapArmsByNotPred,
    /// Cancel double NOT:
    /// `not(not(x)) ↔ x`
    NotNotCancel,
    /// Cancel double NEG (two's complement):
    /// `neg(neg(x)) ↔ x`
    NegNegCancel,
    /// Flip eq/ne through NOT:
    /// `not(eq(a,b)) ↔ ne(a,b)`
    /// `not(ne(a,b)) ↔ eq(a,b)`
    NotEqNeFlip,
    /// Normalize NOR via NOT(OR) (variadic-friendly) and reverse:
    /// `nor(xs...) ↔ not(or(xs...))`
    NorNotOrFold,
    /// Normalize NAND via NOT(AND) (variadic-friendly) and reverse:
    /// `nand(xs...) ↔ not(and(xs...))`
    NandNotAndFold,
    /// Zero-test via OR-reduce and reverse:
    /// `eq(x, 0_w) ↔ not(or_reduce(x))`
    EqZeroOrReduce,
    /// Nonzero-test via OR-reduce and reverse:
    /// `ne(x, 0_w) ↔ or_reduce(x)`
    NeZeroOrReduce,
    /// Fold nested bit_slices:
    /// `bit_slice(bit_slice(x, s1, w1), s2, w2) ↔ bit_slice(x, s1+s2, w2)`
    BitSliceBitSliceFold,
    /// Distribute bit_slice over concat (and reverse folding form):
    /// `bit_slice(concat(a,b), start=s, width=w) ↔ ...`
    /// (handle in-a / in-b / straddle cases)
    BitSliceConcatDistribute,
    /// Lower small priority_sel to a sel-chain (reverse optional):
    /// for selector bits[M], cases len=M:
    /// `priority_sel(sel, cases=[c0..cM-1], default=d)`
    ///   ↔ nested sel(bit_i, [acc, c_i]) with i from M-1 down to 0
    PrioritySelToSelChain,
    /// Rewire one operand of a node to any other node in the function with the
    /// same type.
    ///
    /// This is *not* guaranteed to preserve semantics and therefore requires an
    /// equivalence oracle check before accepting.
    RewireOperandToSameType,
}

impl fmt::Display for PirTransformKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PirTransformKind::SwapCommutativeBinopOperands => {
                write!(f, "SwapCommutativeBinopOperands")
            }
            PirTransformKind::CloneMultiUserNode => write!(f, "CloneMultiUserNode"),
            PirTransformKind::EqSelDistribute => write!(f, "EqSelDistribute"),
            PirTransformKind::EqNeAddLiteralShift => write!(f, "EqNeAddLiteralShift"),
            PirTransformKind::NotSelDistribute => write!(f, "NotSelDistribute"),
            PirTransformKind::NegSelDistribute => write!(f, "NegSelDistribute"),
            PirTransformKind::BitSliceSelDistribute => write!(f, "BitSliceSelDistribute"),
            PirTransformKind::SignExtSelDistribute => write!(f, "SignExtSelDistribute"),
            PirTransformKind::PrioritySel1ToSel => write!(f, "PrioritySel1ToSel"),
            PirTransformKind::AndMaskSignExtToSel => write!(f, "AndMaskSignExtToSel"),
            PirTransformKind::XorMaskSignExtToSelNot => write!(f, "XorMaskSignExtToSelNot"),
            PirTransformKind::SelSameArmsFold => write!(f, "SelSameArmsFold"),
            PirTransformKind::SelSwapArmsByNotPred => write!(f, "SelSwapArmsByNotPred"),
            PirTransformKind::NotNotCancel => write!(f, "NotNotCancel"),
            PirTransformKind::NegNegCancel => write!(f, "NegNegCancel"),
            PirTransformKind::NotEqNeFlip => write!(f, "NotEqNeFlip"),
            PirTransformKind::NorNotOrFold => write!(f, "NorNotOrFold"),
            PirTransformKind::NandNotAndFold => write!(f, "NandNotAndFold"),
            PirTransformKind::EqZeroOrReduce => write!(f, "EqZeroOrReduce"),
            PirTransformKind::NeZeroOrReduce => write!(f, "NeZeroOrReduce"),
            PirTransformKind::BitSliceBitSliceFold => write!(f, "BitSliceBitSliceFold"),
            PirTransformKind::BitSliceConcatDistribute => write!(f, "BitSliceConcatDistribute"),
            PirTransformKind::PrioritySelToSelChain => write!(f, "PrioritySelToSelChain"),
            PirTransformKind::RewireOperandToSameType => write!(f, "RewireOperandToSameType"),
        }
    }
}

/// A specific location in a PIR function where a transform can be applied.
#[derive(Debug, Clone)]
pub enum TransformLocation {
    Node(NodeRef),
    RewireOperand {
        node: NodeRef,
        operand_slot: usize,
        new_operand: NodeRef,
    },
}

/// Defines a reversible transformation that can be applied to a PIR function.
pub trait PirTransform: fmt::Debug + Send + Sync {
    /// Returns the specific `PirTransformKind` that this trait object
    /// represents.
    fn kind(&self) -> PirTransformKind;

    /// Finds all possible application sites for this transform in the given
    /// function.
    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation>;

    /// Applies the transform to the function at the given candidate site.
    ///
    /// The MCMC loop is expected to pass a clone of the function if rejection
    /// implies reverting to the prior state.
    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String>;

    /// Indicates whether this transform is always semantics preserving.
    ///
    /// When `true`, applying the transform cannot change the functional
    /// behaviour of the function, so equivalence checks can be skipped.
    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A transform that clones a node with multiple users so that afterwards each
/// user refers to a distinct replica of the node.
///
/// This is semantics-preserving: all replicas compute the same value as the
/// original node, they are simply duplicated to reduce fanout.
#[derive(Debug)]
pub struct CloneMultiUserNodeTransform;

impl PirTransform for CloneMultiUserNodeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CloneMultiUserNode
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let users_map = compute_users(f);
        users_map
            .iter()
            .filter_map(|(nr, users)| {
                // Do not clone parameter nodes; they are expected to always
                // have names, and cloning them can violate invariants used by
                // pretty-printing and other utilities.
                let node = f.get_node(*nr);
                match &node.payload {
                    NodePayload::GetParam(_) => None,
                    NodePayload::Nil => None,
                    _ if users.len() > 1 => Some(TransformLocation::Node(*nr)),
                    _ => None,
                }
            })
            .collect()
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "CloneMultiUserNode: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let users_map = compute_users(f);
        let users = users_map.get(&target_ref).ok_or_else(|| {
            format!(
                "CloneMultiUserNode: NodeRef {:?} not found in users map",
                target_ref
            )
        })?;

        if users.len() <= 1 {
            return Err("CloneMultiUserNode: target node does not have multiple users".to_string());
        }

        // Stable ordering of users by index for determinism.
        let mut users_vec: Vec<NodeRef> = users.iter().copied().collect();
        users_vec.sort_by_key(|nr| nr.index);

        let original_node = f.get_node(target_ref).clone();
        let mut next_text_id = f
            .nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        for (i, user_nr) in users_vec.into_iter().enumerate() {
            let replacement_ref = if i == 0 {
                // The first user keeps using the original node.
                target_ref
            } else {
                // Subsequent users each get their own cloned node appended to `f.nodes`.
                let mut cloned = original_node.clone();
                cloned.text_id = next_text_id;
                next_text_id = next_text_id.saturating_add(1);
                // Clear the name so that textual IR emission does not produce
                // duplicate node identifiers. The text_id still uniquely
                // identifies the clone, and semantics are unchanged.
                cloned.name = None;
                let new_index = f.nodes.len();
                f.nodes.push(cloned);
                NodeRef { index: new_index }
            };

            let user_node = f.get_node_mut(user_nr);
            user_node.payload = remap_payload_with(&user_node.payload, |(_slot, dep)| {
                if dep == target_ref {
                    replacement_ref
                } else {
                    dep
                }
            });
        }

        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A simple transform that swaps the operands of commutative binary operators.
///
/// Currently this is restricted to `add` nodes, which are clearly commutative.
#[derive(Debug)]
pub struct SwapCommutativeBinopOperandsTransform;

impl PirTransform for SwapCommutativeBinopOperandsTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SwapCommutativeBinopOperands
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        f.node_refs()
            .into_iter()
            .filter(|nr| {
                matches!(
                    f.get_node(*nr).payload,
                    NodePayload::Binop(Binop::Add, _, _)
                )
            })
            .map(TransformLocation::Node)
            .collect()
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        match loc {
            TransformLocation::Node(nr) => {
                let node = f.get_node_mut(*nr);
                match &mut node.payload {
                    NodePayload::Binop(Binop::Add, lhs, rhs) => {
                        mem::swap(lhs, rhs);
                        Ok(())
                    }
                    other => Err(format!(
                        "SwapCommutativeBinopOperandsTransform: expected add binop, found {:?}",
                        other
                    )),
                }
            }
            TransformLocation::RewireOperand { .. } => Err(
                "SwapCommutativeBinopOperandsTransform: expected TransformLocation::Node, got RewireOperand"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing the equivalence:
///
/// - `eq(sel(p,a,b),c) ↔ sel(p,eq(a,c),eq(b,c))`
/// - `ne(sel(p,a,b),c) ↔ sel(p,ne(a,c),ne(b,c))`
///
/// This works for any value type `T` where `eq(T,T) -> bits[1]` is defined.
#[derive(Debug)]
pub struct EqSelDistributeTransform;

impl EqSelDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn sel2_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } if cases.len() == 2 && default.is_none() => Some((*selector, cases[0], cases[1])),
            _ => None,
        }
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn eq_operands(payload: &NodePayload) -> Option<(NodeRef, NodeRef)> {
        match payload {
            NodePayload::Binop(Binop::Eq, lhs, rhs) => Some((*lhs, *rhs)),
            _ => None,
        }
    }

    fn ne_operands(payload: &NodePayload) -> Option<(NodeRef, NodeRef)> {
        match payload {
            NodePayload::Binop(Binop::Ne, lhs, rhs) => Some((*lhs, *rhs)),
            _ => None,
        }
    }

    fn cmp_operands(payload: &NodePayload, op: Binop) -> Option<(NodeRef, NodeRef)> {
        match op {
            Binop::Eq => Self::eq_operands(payload),
            Binop::Ne => Self::ne_operands(payload),
            _ => None,
        }
    }

    fn type_of(f: &IrFn, r: NodeRef) -> Type {
        f.get_node(r).ty.clone()
    }

    fn choose_fold_candidate(
        f: &IrFn,
        op: Binop,
        c1: NodeRef,
        c2: NodeRef,
    ) -> Option<(NodeRef, NodeRef, NodeRef)> {
        let (u, v) = Self::cmp_operands(&f.get_node(c1).payload, op)?;
        let (s, t) = Self::cmp_operands(&f.get_node(c2).payload, op)?;

        let mut candidates: Vec<(NodeRef, NodeRef, NodeRef)> = Vec::new();
        // If u is common
        if u == s {
            candidates.push((v, t, u));
        }
        if u == t {
            candidates.push((v, s, u));
        }
        // If v is common
        if v == s {
            candidates.push((u, t, v));
        }
        if v == t {
            candidates.push((u, s, v));
        }

        candidates.retain(|(a, b, c)| {
            let ta = Self::type_of(f, *a);
            ta == Self::type_of(f, *b) && ta == Self::type_of(f, *c)
        });

        candidates.sort_by_key(|(a, b, c)| (c.index, a.index, b.index));
        candidates.into_iter().next()
    }
}

impl PirTransform for EqSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::EqSelDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                NodePayload::Binop(Binop::Eq, lhs, rhs)
                | NodePayload::Binop(Binop::Ne, lhs, rhs) => {
                    // eq/ne(sel(p,a,b), c) or eq/ne(c, sel(p,a,b))
                    let lhs_node = f.get_node(*lhs);
                    let rhs_node = f.get_node(*rhs);

                    if let Some((p, a, b)) = Self::sel2_parts(&lhs_node.payload) {
                        if Self::is_u1_selector(f, p)
                            && Self::type_of(f, a) == Self::type_of(f, *rhs)
                            && Self::type_of(f, b) == Self::type_of(f, *rhs)
                        {
                            out.push(TransformLocation::Node(nr));
                            continue;
                        }
                    }
                    if let Some((p, a, b)) = Self::sel2_parts(&rhs_node.payload) {
                        if Self::is_u1_selector(f, p)
                            && Self::type_of(f, a) == Self::type_of(f, *lhs)
                            && Self::type_of(f, b) == Self::type_of(f, *lhs)
                        {
                            out.push(TransformLocation::Node(nr));
                        }
                    }
                }
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    // sel(p, eq(a,c), eq(b,c)) or sel(p, ne(a,c), ne(b,c))
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    if Self::choose_fold_candidate(f, Binop::Eq, cases[0], cases[1]).is_some()
                        || Self::choose_fold_candidate(f, Binop::Ne, cases[0], cases[1]).is_some()
                    {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "EqSelDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Binop(Binop::Eq, lhs, rhs) | NodePayload::Binop(Binop::Ne, lhs, rhs) => {
                let op = match &target_payload {
                    NodePayload::Binop(op, _, _) => *op,
                    _ => unreachable!(),
                };
                let (sel_ref, c_ref) = match Self::sel2_parts(&f.get_node(lhs).payload) {
                    Some(_) => (lhs, rhs),
                    None => match Self::sel2_parts(&f.get_node(rhs).payload) {
                        Some(_) => (rhs, lhs),
                        None => {
                            return Err(
                                "EqSelDistributeTransform: cmp node did not match {eq,ne}(sel(...), c)"
                                    .to_string(),
                            );
                        }
                    },
                };

                let (p, a, b) = Self::sel2_parts(&f.get_node(sel_ref).payload)
                    .expect("sel_ref should refer to sel with 2 cases");
                if !Self::is_u1_selector(f, p) {
                    return Err("EqSelDistributeTransform: sel selector is not bits[1]".to_string());
                }
                let c_ty = Self::type_of(f, c_ref);
                if Self::type_of(f, a) != c_ty || Self::type_of(f, b) != c_ty {
                    return Err(
                        "EqSelDistributeTransform: sel cases and c must have the same type"
                            .to_string(),
                    );
                }

                let mut next_text_id = Self::next_text_id(f);
                let eq_ty = Type::Bits(1);

                let eq_a_c_ref = {
                    let new_index = f.nodes.len();
                    f.nodes.push(xlsynth_pir::ir::Node {
                        text_id: next_text_id,
                        name: None,
                        ty: eq_ty.clone(),
                        payload: NodePayload::Binop(op, a, c_ref),
                        pos: None,
                    });
                    next_text_id = next_text_id.saturating_add(1);
                    NodeRef { index: new_index }
                };

                let eq_b_c_ref = {
                    let new_index = f.nodes.len();
                    f.nodes.push(xlsynth_pir::ir::Node {
                        text_id: next_text_id,
                        name: None,
                        ty: eq_ty,
                        payload: NodePayload::Binop(op, b, c_ref),
                        pos: None,
                    });
                    NodeRef { index: new_index }
                };

                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![eq_a_c_ref, eq_b_c_ref],
                    default: None,
                };
                Ok(())
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "EqSelDistributeTransform: sel node did not match 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("EqSelDistributeTransform: sel selector is not bits[1]".to_string());
                }

                let (op, a, b, c) = if let Some((a, b, c)) =
                    Self::choose_fold_candidate(f, Binop::Eq, cases[0], cases[1])
                {
                    (Binop::Eq, a, b, c)
                } else if let Some((a, b, c)) =
                    Self::choose_fold_candidate(f, Binop::Ne, cases[0], cases[1])
                {
                    (Binop::Ne, a, b, c)
                } else {
                    return Err(
                        "EqSelDistributeTransform: sel cases did not match sel(p, {eq,ne}(a,c), {eq,ne}(b,c))"
                            .to_string(),
                    );
                };

                let next_text_id = Self::next_text_id(f);
                let sel_ty = Self::type_of(f, a);

                let sel_ab_ref = {
                    let new_index = f.nodes.len();
                    f.nodes.push(xlsynth_pir::ir::Node {
                        text_id: next_text_id,
                        name: None,
                        ty: sel_ty,
                        payload: NodePayload::Sel {
                            selector,
                            cases: vec![a, b],
                            default: None,
                        },
                        pos: None,
                    });
                    NodeRef { index: new_index }
                };

                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Eq, sel_ab_ref, c);
                if op == Binop::Ne {
                    f.get_node_mut(target_ref).payload =
                        NodePayload::Binop(Binop::Ne, sel_ab_ref, c);
                }
                Ok(())
            }
            _ => Err(
                "EqSelDistributeTransform: expected eq or sel payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// - `eq(add(x, k_lit), c) ↔ eq(x, sub(c, k_lit))`
/// - `ne(add(x, k_lit), c) ↔ ne(x, sub(c, k_lit))`
///
/// This relies on the standard XLS bit-vector semantics where `add`/`sub` on
/// `bits[w]` are performed modulo \(2^w\).
#[derive(Debug)]
pub struct EqNeAddLiteralShiftTransform;

impl EqNeAddLiteralShiftTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_bits_type(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn is_cmp_op(op: Binop) -> bool {
        matches!(op, Binop::Eq | Binop::Ne)
    }

    fn is_literal(f: &IrFn, r: NodeRef) -> bool {
        matches!(f.get_node(r).payload, NodePayload::Literal(_))
    }

    fn add_with_literal_parts(f: &IrFn, add_ref: NodeRef) -> Option<(NodeRef, NodeRef)> {
        match &f.get_node(add_ref).payload {
            NodePayload::Binop(Binop::Add, lhs, rhs) => {
                if Self::is_literal(f, *lhs) {
                    Some((*rhs, *lhs))
                } else if Self::is_literal(f, *rhs) {
                    Some((*lhs, *rhs))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn sub_with_literal_parts(f: &IrFn, sub_ref: NodeRef) -> Option<(NodeRef, NodeRef)> {
        match &f.get_node(sub_ref).payload {
            NodePayload::Binop(Binop::Sub, lhs, rhs) => {
                if Self::is_literal(f, *rhs) {
                    Some((*lhs, *rhs))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn mk_binop_node(f: &mut IrFn, op: Binop, ty: Type, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(xlsynth_pir::ir::Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Binop(op, a, b),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn apply_cmp_shift(
        f: &mut IrFn,
        target_ref: NodeRef,
        op: Binop,
        lhs: NodeRef,
        rhs: NodeRef,
    ) -> Result<(), String> {
        // Direction A: cmp(add(x,k), c) -> cmp(x, sub(c,k))
        if let Some((x, k_lit)) = Self::add_with_literal_parts(f, lhs) {
            let w = Self::is_bits_type(f, x).ok_or_else(|| {
                "EqNeAddLiteralShiftTransform: x must have bits[w] type".to_string()
            })?;
            if Self::is_bits_type(f, rhs) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: c must have the same bits[w] type as x"
                        .to_string(),
                );
            }
            if Self::is_bits_type(f, k_lit) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: k literal must have the same bits[w] type as x"
                        .to_string(),
                );
            }

            let sub_ref = Self::mk_binop_node(f, Binop::Sub, Type::Bits(w), rhs, k_lit);
            f.get_node_mut(target_ref).payload = NodePayload::Binop(op, x, sub_ref);
            return Ok(());
        }
        if let Some((x, k_lit)) = Self::add_with_literal_parts(f, rhs) {
            let w = Self::is_bits_type(f, x).ok_or_else(|| {
                "EqNeAddLiteralShiftTransform: x must have bits[w] type".to_string()
            })?;
            if Self::is_bits_type(f, lhs) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: c must have the same bits[w] type as x"
                        .to_string(),
                );
            }
            if Self::is_bits_type(f, k_lit) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: k literal must have the same bits[w] type as x"
                        .to_string(),
                );
            }

            let sub_ref = Self::mk_binop_node(f, Binop::Sub, Type::Bits(w), lhs, k_lit);
            f.get_node_mut(target_ref).payload = NodePayload::Binop(op, x, sub_ref);
            return Ok(());
        }

        // Direction B (fold): cmp(x, sub(c,k)) -> cmp(add(x,k), c)
        if let Some((c, k_lit)) = Self::sub_with_literal_parts(f, rhs) {
            let w = Self::is_bits_type(f, lhs).ok_or_else(|| {
                "EqNeAddLiteralShiftTransform: x must have bits[w] type".to_string()
            })?;
            if Self::is_bits_type(f, c) != Some(w) || Self::is_bits_type(f, k_lit) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: c and k literal must have the same bits[w] type as x"
                        .to_string(),
                );
            }
            let add_ref = Self::mk_binop_node(f, Binop::Add, Type::Bits(w), lhs, k_lit);
            f.get_node_mut(target_ref).payload = NodePayload::Binop(op, add_ref, c);
            return Ok(());
        }
        if let Some((c, k_lit)) = Self::sub_with_literal_parts(f, lhs) {
            let w = Self::is_bits_type(f, rhs).ok_or_else(|| {
                "EqNeAddLiteralShiftTransform: x must have bits[w] type".to_string()
            })?;
            if Self::is_bits_type(f, c) != Some(w) || Self::is_bits_type(f, k_lit) != Some(w) {
                return Err(
                    "EqNeAddLiteralShiftTransform: c and k literal must have the same bits[w] type as x"
                        .to_string(),
                );
            }
            let add_ref = Self::mk_binop_node(f, Binop::Add, Type::Bits(w), rhs, k_lit);
            f.get_node_mut(target_ref).payload = NodePayload::Binop(op, add_ref, c);
            return Ok(());
        }

        Err(
            "EqNeAddLiteralShiftTransform: target did not match expected cmp/add/sub-with-literal patterns"
                .to_string(),
        )
    }
}

impl PirTransform for EqNeAddLiteralShiftTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::EqNeAddLiteralShift
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match node.payload {
                NodePayload::Binop(op, lhs, rhs) if Self::is_cmp_op(op) => {
                    // Expand direction:
                    //   cmp(add(x,k_lit), c) or cmp(c, add(x,k_lit))
                    if Self::add_with_literal_parts(f, lhs).is_some()
                        || Self::add_with_literal_parts(f, rhs).is_some()
                    {
                        out.push(TransformLocation::Node(nr));
                        continue;
                    }
                    // Fold direction:
                    //   cmp(x, sub(c,k_lit)) or cmp(sub(c,k_lit), x)
                    if Self::sub_with_literal_parts(f, lhs).is_some()
                        || Self::sub_with_literal_parts(f, rhs).is_some()
                    {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "EqNeAddLiteralShiftTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Binop(op, lhs, rhs) if Self::is_cmp_op(op) => {
                Self::apply_cmp_shift(f, target_ref, op, lhs, rhs)
            }
            _ => Err(
                "EqNeAddLiteralShiftTransform: expected eq/ne binop payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `not(sel(p, cases=[a, b])) ↔ sel(p, cases=[not(a), not(b)])`
///
/// This is valid for `bits[w]` values, where `not` is the bitwise complement.
#[derive(Debug)]
pub struct NotSelDistributeTransform;

impl NotSelDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn sel2_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } if cases.len() == 2 && default.is_none() => Some((*selector, cases[0], cases[1])),
            _ => None,
        }
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn not_arg(f: &IrFn, r: NodeRef) -> Option<NodeRef> {
        match &f.get_node(r).payload {
            NodePayload::Unop(Unop::Not, arg) => Some(*arg),
            _ => None,
        }
    }

    fn mk_not_node(f: &mut IrFn, w: usize, arg: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Unop(Unop::Not, arg),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2_node(f: &mut IrFn, w: usize, selector: NodeRef, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Sel {
                selector,
                cases: vec![a, b],
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for NotSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NotSelDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                // Expand: not(sel(p, [a,b]))
                NodePayload::Unop(Unop::Not, arg) => {
                    if let Some((p, a, b)) = Self::sel2_parts(&f.get_node(*arg).payload) {
                        if !Self::is_u1_selector(f, p) {
                            continue;
                        }
                        let wa = Self::bits_width(f, a);
                        let wb = Self::bits_width(f, b);
                        let wout = Self::bits_width(f, nr);
                        if wa.is_some() && wa == wb && wa == wout {
                            out.push(TransformLocation::Node(nr));
                        }
                    }
                }
                // Fold: sel(p, [not(a), not(b)])
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let a = match Self::not_arg(f, cases[0]) {
                        Some(v) => v,
                        None => continue,
                    };
                    let b = match Self::not_arg(f, cases[1]) {
                        Some(v) => v,
                        None => continue,
                    };
                    let wa = Self::bits_width(f, a);
                    let wb = Self::bits_width(f, b);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wa == wout {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "NotSelDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // Expand: not(sel(p, [a,b])) -> sel(p, [not(a), not(b)])
            NodePayload::Unop(Unop::Not, arg_sel_ref) => {
                let (p, a, b) =
                    Self::sel2_parts(&f.get_node(arg_sel_ref).payload).ok_or_else(|| {
                        "NotSelDistributeTransform: expected not(sel(p, cases=[a,b]))".to_string()
                    })?;
                if !Self::is_u1_selector(f, p) {
                    return Err("NotSelDistributeTransform: selector must be bits[1]".to_string());
                }
                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NotSelDistributeTransform: sel cases and output must be bits[w] with same width"
                            .to_string()
                    })?;

                let not_a = Self::mk_not_node(f, w, a);
                let not_b = Self::mk_not_node(f, w, b);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![not_a, not_b],
                    default: None,
                };
                Ok(())
            }

            // Fold: sel(p, [not(a), not(b)]) -> not(sel(p, [a,b]))
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "NotSelDistributeTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("NotSelDistributeTransform: selector must be bits[1]".to_string());
                }
                let a = Self::not_arg(f, cases[0]).ok_or_else(|| {
                    "NotSelDistributeTransform: expected sel case 0 to be not(a)".to_string()
                })?;
                let b = Self::not_arg(f, cases[1]).ok_or_else(|| {
                    "NotSelDistributeTransform: expected sel case 1 to be not(b)".to_string()
                })?;

                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NotSelDistributeTransform: sel cases and output must be bits[w] with same width"
                            .to_string()
                    })?;

                let sel_ab = Self::mk_sel2_node(f, w, selector, a, b);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, sel_ab);
                Ok(())
            }
            _ => Err(
                "NotSelDistributeTransform: expected not(sel(...)) or sel(not(...),not(...))"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `neg(sel(p, cases=[a, b])) ↔ sel(p, cases=[neg(a), neg(b)])`
///
/// This is valid for `bits[w]` values, where `neg` is two's-complement
/// arithmetic negation modulo \(2^w\).
#[derive(Debug)]
pub struct NegSelDistributeTransform;

impl NegSelDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn sel2_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } if cases.len() == 2 && default.is_none() => Some((*selector, cases[0], cases[1])),
            _ => None,
        }
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn neg_arg(f: &IrFn, r: NodeRef) -> Option<NodeRef> {
        match &f.get_node(r).payload {
            NodePayload::Unop(Unop::Neg, arg) => Some(*arg),
            _ => None,
        }
    }

    fn mk_neg_node(f: &mut IrFn, w: usize, arg: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Unop(Unop::Neg, arg),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2_node(f: &mut IrFn, w: usize, selector: NodeRef, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Sel {
                selector,
                cases: vec![a, b],
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for NegSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NegSelDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                // Expand: neg(sel(p, [a,b]))
                NodePayload::Unop(Unop::Neg, arg) => {
                    if let Some((p, a, b)) = Self::sel2_parts(&f.get_node(*arg).payload) {
                        if !Self::is_u1_selector(f, p) {
                            continue;
                        }
                        let wa = Self::bits_width(f, a);
                        let wb = Self::bits_width(f, b);
                        let wout = Self::bits_width(f, nr);
                        if wa.is_some() && wa == wb && wa == wout {
                            out.push(TransformLocation::Node(nr));
                        }
                    }
                }
                // Fold: sel(p, [neg(a), neg(b)])
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let a = match Self::neg_arg(f, cases[0]) {
                        Some(v) => v,
                        None => continue,
                    };
                    let b = match Self::neg_arg(f, cases[1]) {
                        Some(v) => v,
                        None => continue,
                    };
                    let wa = Self::bits_width(f, a);
                    let wb = Self::bits_width(f, b);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wa == wout {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "NegSelDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // Expand: neg(sel(p,[a,b])) -> sel(p,[neg(a),neg(b)])
            NodePayload::Unop(Unop::Neg, arg_sel_ref) => {
                let (p, a, b) =
                    Self::sel2_parts(&f.get_node(arg_sel_ref).payload).ok_or_else(|| {
                        "NegSelDistributeTransform: expected neg(sel(p, cases=[a,b]))".to_string()
                    })?;
                if !Self::is_u1_selector(f, p) {
                    return Err("NegSelDistributeTransform: selector must be bits[1]".to_string());
                }
                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NegSelDistributeTransform: sel cases and output must be bits[w] with same width"
                            .to_string()
                    })?;

                let neg_a = Self::mk_neg_node(f, w, a);
                let neg_b = Self::mk_neg_node(f, w, b);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![neg_a, neg_b],
                    default: None,
                };
                Ok(())
            }

            // Fold: sel(p,[neg(a),neg(b)]) -> neg(sel(p,[a,b]))
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "NegSelDistributeTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("NegSelDistributeTransform: selector must be bits[1]".to_string());
                }
                let a = Self::neg_arg(f, cases[0]).ok_or_else(|| {
                    "NegSelDistributeTransform: expected sel case 0 to be neg(a)".to_string()
                })?;
                let b = Self::neg_arg(f, cases[1]).ok_or_else(|| {
                    "NegSelDistributeTransform: expected sel case 1 to be neg(b)".to_string()
                })?;

                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NegSelDistributeTransform: sel cases and output must be bits[w] with same width"
                            .to_string()
                    })?;

                let sel_ab = Self::mk_sel2_node(f, w, selector, a, b);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Neg, sel_ab);
                Ok(())
            }
            _ => Err(
                "NegSelDistributeTransform: expected neg(sel(...)) or sel(neg(...),neg(...))"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `bit_slice(sel(p, cases=[a, b]), start=s, width=w)
///    ↔ sel(p, cases=[bit_slice(a,s,w), bit_slice(b,s,w)])`
#[derive(Debug)]
pub struct BitSliceSelDistributeTransform;

impl BitSliceSelDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn sel2_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } if cases.len() == 2 && default.is_none() => Some((*selector, cases[0], cases[1])),
            _ => None,
        }
    }

    fn bit_slice_parts(payload: &NodePayload) -> Option<(NodeRef, usize, usize)> {
        match payload {
            NodePayload::BitSlice { arg, start, width } => Some((*arg, *start, *width)),
            _ => None,
        }
    }

    fn mk_bit_slice_node(
        f: &mut IrFn,
        out_w: usize,
        arg: NodeRef,
        start: usize,
        width: usize,
    ) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::BitSlice { arg, start, width },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2_node(
        f: &mut IrFn,
        out_w: usize,
        selector: NodeRef,
        a: NodeRef,
        b: NodeRef,
    ) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::Sel {
                selector,
                cases: vec![a, b],
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for BitSliceSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BitSliceSelDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                // Expand: bit_slice(sel(...), s, w)
                NodePayload::BitSlice { arg, start, width } => {
                    if let Some((p, a, b)) = Self::sel2_parts(&f.get_node(*arg).payload) {
                        if !Self::is_u1_selector(f, p) {
                            continue;
                        }
                        let wa = Self::bits_width(f, a);
                        let wb = Self::bits_width(f, b);
                        let wout = Self::bits_width(f, nr);
                        if wa.is_some() && wa == wb && wout == Some(*width) {
                            // Also require slice to be in-bounds to avoid constructing invalid IR.
                            if start.saturating_add(*width) <= wa.unwrap() {
                                out.push(TransformLocation::Node(nr));
                            }
                        }
                    }
                }
                // Fold: sel(p, [bit_slice(a,s,w), bit_slice(b,s,w)])
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let (a_arg, a_start, a_width) =
                        match Self::bit_slice_parts(&f.get_node(cases[0]).payload) {
                            Some(v) => v,
                            None => continue,
                        };
                    let (b_arg, b_start, b_width) =
                        match Self::bit_slice_parts(&f.get_node(cases[1]).payload) {
                            Some(v) => v,
                            None => continue,
                        };
                    if a_start != b_start || a_width != b_width {
                        continue;
                    }
                    let wa = Self::bits_width(f, a_arg);
                    let wb = Self::bits_width(f, b_arg);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wout == Some(a_width) {
                        if a_start.saturating_add(a_width) <= wa.unwrap() {
                            out.push(TransformLocation::Node(nr));
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "BitSliceSelDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // Expand: bit_slice(sel(...), s, w) -> sel(...bit_slice...)
            NodePayload::BitSlice { arg, start, width } => {
                let (p, a, b) =
                    Self::sel2_parts(&f.get_node(arg).payload).ok_or_else(|| {
                        "BitSliceSelDistributeTransform: expected bit_slice(sel(p, cases=[a,b]), ...)"
                            .to_string()
                    })?;
                if !Self::is_u1_selector(f, p) {
                    return Err(
                        "BitSliceSelDistributeTransform: selector must be bits[1]".to_string(),
                    );
                }
                let in_w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .ok_or_else(|| {
                        "BitSliceSelDistributeTransform: sel cases must be bits[w] with same width"
                            .to_string()
                    })?;
                if start.saturating_add(width) > in_w {
                    return Err(
                        "BitSliceSelDistributeTransform: bit_slice start+width out of bounds"
                            .to_string(),
                    );
                }
                let out_w = width;
                if Self::bits_width(f, target_ref) != Some(out_w) {
                    return Err(
                        "BitSliceSelDistributeTransform: output type must be bits[width]".to_string(),
                    );
                }

                let bs_a = Self::mk_bit_slice_node(f, out_w, a, start, width);
                let bs_b = Self::mk_bit_slice_node(f, out_w, b, start, width);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![bs_a, bs_b],
                    default: None,
                };
                Ok(())
            }

            // Fold: sel(p, [bit_slice(a,s,w), bit_slice(b,s,w)]) -> bit_slice(sel(p,[a,b]), s, w)
            NodePayload::Sel { selector, cases, default } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "BitSliceSelDistributeTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "BitSliceSelDistributeTransform: selector must be bits[1]".to_string(),
                    );
                }
                let (a_arg, a_start, a_width) =
                    Self::bit_slice_parts(&f.get_node(cases[0]).payload).ok_or_else(|| {
                        "BitSliceSelDistributeTransform: expected sel case 0 to be bit_slice(a,...)"
                            .to_string()
                    })?;
                let (b_arg, b_start, b_width) =
                    Self::bit_slice_parts(&f.get_node(cases[1]).payload).ok_or_else(|| {
                        "BitSliceSelDistributeTransform: expected sel case 1 to be bit_slice(b,...)"
                            .to_string()
                    })?;
                if a_start != b_start || a_width != b_width {
                    return Err(
                        "BitSliceSelDistributeTransform: bit_slice params must match across cases"
                            .to_string(),
                    );
                }
                let in_w = Self::bits_width(f, a_arg)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b_arg))
                    .ok_or_else(|| {
                        "BitSliceSelDistributeTransform: bit_slice args must be bits[w] with same width"
                            .to_string()
                    })?;
                if a_start.saturating_add(a_width) > in_w {
                    return Err(
                        "BitSliceSelDistributeTransform: bit_slice start+width out of bounds"
                            .to_string(),
                    );
                }
                if Self::bits_width(f, target_ref) != Some(a_width) {
                    return Err(
                        "BitSliceSelDistributeTransform: output type must be bits[width]".to_string(),
                    );
                }

                let sel_ab = Self::mk_sel2_node(f, in_w, selector, a_arg, b_arg);
                f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                    arg: sel_ab,
                    start: a_start,
                    width: a_width,
                };
                Ok(())
            }
            _ => Err(
                "BitSliceSelDistributeTransform: expected bit_slice(sel(...)) or sel(bit_slice(...),bit_slice(...))"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `sign_ext(sel(p, cases=[a, b]), new_bit_count=n)
///    ↔ sel(p, cases=[sign_ext(a,n), sign_ext(b,n)])`
#[derive(Debug)]
pub struct SignExtSelDistributeTransform;

impl SignExtSelDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn sel2_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } if cases.len() == 2 && default.is_none() => Some((*selector, cases[0], cases[1])),
            _ => None,
        }
    }

    fn sign_ext_parts(payload: &NodePayload) -> Option<(NodeRef, usize)> {
        match payload {
            NodePayload::SignExt { arg, new_bit_count } => Some((*arg, *new_bit_count)),
            _ => None,
        }
    }

    fn mk_sign_ext_node(f: &mut IrFn, out_w: usize, arg: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::SignExt {
                arg,
                new_bit_count: out_w,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2_node(
        f: &mut IrFn,
        out_w: usize,
        selector: NodeRef,
        a: NodeRef,
        b: NodeRef,
    ) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::Sel {
                selector,
                cases: vec![a, b],
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for SignExtSelDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SignExtSelDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                // Expand: sign_ext(sel(...), n)
                NodePayload::SignExt { arg, new_bit_count } => {
                    if let Some((p, a, b)) = Self::sel2_parts(&f.get_node(*arg).payload) {
                        if !Self::is_u1_selector(f, p) {
                            continue;
                        }
                        let wa = Self::bits_width(f, a);
                        let wb = Self::bits_width(f, b);
                        let wout = Self::bits_width(f, nr);
                        if wa.is_some() && wa == wb && wout == Some(*new_bit_count) {
                            if *new_bit_count >= wa.unwrap() {
                                out.push(TransformLocation::Node(nr));
                            }
                        }
                    }
                }
                // Fold: sel(p, [sign_ext(a,n), sign_ext(b,n)])
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let (a_arg, a_n) = match Self::sign_ext_parts(&f.get_node(cases[0]).payload) {
                        Some(v) => v,
                        None => continue,
                    };
                    let (b_arg, b_n) = match Self::sign_ext_parts(&f.get_node(cases[1]).payload) {
                        Some(v) => v,
                        None => continue,
                    };
                    if a_n != b_n {
                        continue;
                    }
                    let wa = Self::bits_width(f, a_arg);
                    let wb = Self::bits_width(f, b_arg);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wout == Some(a_n) {
                        if a_n >= wa.unwrap() {
                            out.push(TransformLocation::Node(nr));
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "SignExtSelDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // Expand: sign_ext(sel(...), n) -> sel(...sign_ext...)
            NodePayload::SignExt { arg, new_bit_count } => {
                let (p, a, b) =
                    Self::sel2_parts(&f.get_node(arg).payload).ok_or_else(|| {
                        "SignExtSelDistributeTransform: expected sign_ext(sel(p, cases=[a,b]), ...)"
                            .to_string()
                    })?;
                if !Self::is_u1_selector(f, p) {
                    return Err(
                        "SignExtSelDistributeTransform: selector must be bits[1]".to_string(),
                    );
                }
                let in_w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .ok_or_else(|| {
                        "SignExtSelDistributeTransform: sel cases must be bits[w] with same width"
                            .to_string()
                    })?;
                if new_bit_count < in_w {
                    return Err(
                        "SignExtSelDistributeTransform: new_bit_count must be >= input width"
                            .to_string(),
                    );
                }
                let out_w = new_bit_count;
                if Self::bits_width(f, target_ref) != Some(out_w) {
                    return Err(
                        "SignExtSelDistributeTransform: output type must be bits[new_bit_count]"
                            .to_string(),
                    );
                }

                let se_a = Self::mk_sign_ext_node(f, out_w, a);
                let se_b = Self::mk_sign_ext_node(f, out_w, b);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![se_a, se_b],
                    default: None,
                };
                Ok(())
            }

            // Fold: sel(p,[sign_ext(a,n),sign_ext(b,n)]) -> sign_ext(sel(p,[a,b]), n)
            NodePayload::Sel { selector, cases, default } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "SignExtSelDistributeTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "SignExtSelDistributeTransform: selector must be bits[1]".to_string(),
                    );
                }
                let (a_arg, a_n) =
                    Self::sign_ext_parts(&f.get_node(cases[0]).payload).ok_or_else(|| {
                        "SignExtSelDistributeTransform: expected sel case 0 to be sign_ext(a,...)"
                            .to_string()
                    })?;
                let (b_arg, b_n) =
                    Self::sign_ext_parts(&f.get_node(cases[1]).payload).ok_or_else(|| {
                        "SignExtSelDistributeTransform: expected sel case 1 to be sign_ext(b,...)"
                            .to_string()
                    })?;
                if a_n != b_n {
                    return Err(
                        "SignExtSelDistributeTransform: new_bit_count must match across cases"
                            .to_string(),
                    );
                }
                let in_w = Self::bits_width(f, a_arg)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b_arg))
                    .ok_or_else(|| {
                        "SignExtSelDistributeTransform: sign_ext args must be bits[w] with same width"
                            .to_string()
                    })?;
                if a_n < in_w {
                    return Err(
                        "SignExtSelDistributeTransform: new_bit_count must be >= input width"
                            .to_string(),
                    );
                }
                if Self::bits_width(f, target_ref) != Some(a_n) {
                    return Err(
                        "SignExtSelDistributeTransform: output type must be bits[new_bit_count]"
                            .to_string(),
                    );
                }

                let sel_ab = Self::mk_sel2_node(f, in_w, selector, a_arg, b_arg);
                f.get_node_mut(target_ref).payload = NodePayload::SignExt {
                    arg: sel_ab,
                    new_bit_count: a_n,
                };
                Ok(())
            }
            _ => Err(
                "SignExtSelDistributeTransform: expected sign_ext(sel(...)) or sel(sign_ext(...),sign_ext(...))"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `priority_sel(p:bits[1], cases=[a], default=b) ↔ sel(p, cases=[b, a])`
#[derive(Debug)]
pub struct PrioritySel1ToSelTransform;

impl PrioritySel1ToSelTransform {
    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }
}

impl PirTransform for PrioritySel1ToSelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::PrioritySel1ToSel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // priority_sel(p, cases=[a], default=b) -> sel(p, cases=[b,a])
                NodePayload::PrioritySel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 1 {
                        continue;
                    }
                    let Some(default_ref) = default else {
                        continue;
                    };
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let a = cases[0];
                    let b = *default_ref;
                    let wa = Self::bits_width(f, a);
                    let wb = Self::bits_width(f, b);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wa == wout {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                // sel(p, cases=[b,a]) -> priority_sel(p, cases=[a], default=b)
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let b = cases[0];
                    let a = cases[1];
                    let wa = Self::bits_width(f, a);
                    let wb = Self::bits_width(f, b);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wa == wout {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "PrioritySel1ToSelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 1 {
                    return Err(
                        "PrioritySel1ToSelTransform: expected priority_sel with 1 case"
                            .to_string(),
                    );
                }
                let Some(b) = default else {
                    return Err(
                        "PrioritySel1ToSelTransform: expected priority_sel to have a default"
                            .to_string(),
                    );
                };
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "PrioritySel1ToSelTransform: selector must be bits[1]".to_string(),
                    );
                }
                let a = cases[0];
                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "PrioritySel1ToSelTransform: cases/default/output must be bits[w] with same width"
                            .to_string()
                    })?;
                let _ = w;

                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector,
                    cases: vec![b, a],
                    default: None,
                };
                Ok(())
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "PrioritySel1ToSelTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "PrioritySel1ToSelTransform: selector must be bits[1]".to_string(),
                    );
                }
                let b = cases[0];
                let a = cases[1];
                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "PrioritySel1ToSelTransform: cases/output must be bits[w] with same width"
                            .to_string()
                    })?;
                let _ = w;

                f.get_node_mut(target_ref).payload = NodePayload::PrioritySel {
                    selector,
                    cases: vec![a],
                    default: Some(b),
                };
                Ok(())
            }
            _ => Err(
                "PrioritySel1ToSelTransform: expected priority_sel or sel payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `and(x, sign_ext(b,w)) ↔ sel(b, cases=[0_w, x])`
#[derive(Debug)]
pub struct AndMaskSignExtToSelTransform;

impl AndMaskSignExtToSelTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn sign_ext_mask_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        match &f.get_node(r).payload {
            NodePayload::SignExt { arg, new_bit_count } => {
                if !Self::is_u1_selector(f, *arg) {
                    return None;
                }
                if Self::bits_width(f, r) != Some(*new_bit_count) {
                    return None;
                }
                Some((*arg, *new_bit_count))
            }
            _ => None,
        }
    }

    fn mk_zero_literal_node(f: &mut IrFn, w: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let value = IrValue::from_bits(&bits);
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Literal(value),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        *v == expected
    }
}

impl PirTransform for AndMaskSignExtToSelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AndMaskSignExtToSel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // Expand: and(x, sign_ext(b,w)) (nary)
                NodePayload::Nary(NaryOp::And, ops) if ops.len() == 2 => {
                    let (a, b) = (ops[0], ops[1]);
                    let mut found = None;
                    for (x, mask) in [(a, b), (b, a)] {
                        if let Some((sel_b, w)) = Self::sign_ext_mask_parts(f, mask) {
                            if Self::bits_width(f, x) == Some(w)
                                && Self::bits_width(f, nr) == Some(w)
                            {
                                found = Some((sel_b, w));
                            }
                        }
                    }
                    if found.is_some() {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                // Fold: sel(b, cases=[0_w, x])
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let w = match Self::bits_width(f, nr) {
                        Some(w) => w,
                        None => continue,
                    };
                    if !Self::is_zero_literal_node(f, cases[0], w) {
                        continue;
                    }
                    if Self::bits_width(f, cases[1]) != Some(w) {
                        continue;
                    }
                    out.push(TransformLocation::Node(nr));
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "AndMaskSignExtToSelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Nary(NaryOp::And, ops) => {
                if ops.len() != 2 {
                    return Err("AndMaskSignExtToSelTransform: expected 2-operand and".to_string());
                }
                let (a, b) = (ops[0], ops[1]);
                // Identify x and sign_ext(b,w)
                let mut matched: Option<(NodeRef, NodeRef, usize)> = None;
                for (x, mask) in [(a, b), (b, a)] {
                    if let Some((sel_b, w)) = Self::sign_ext_mask_parts(f, mask) {
                        if Self::bits_width(f, x) == Some(w) && Self::bits_width(f, target_ref) == Some(w) {
                            matched = Some((x, sel_b, w));
                            break;
                        }
                    }
                }
                let Some((x, sel_b, w)) = matched else {
                    return Err(
                        "AndMaskSignExtToSelTransform: expected and(x, sign_ext(b,w)) pattern"
                            .to_string(),
                    );
                };

                let zero = Self::mk_zero_literal_node(f, w);
                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: sel_b,
                    cases: vec![zero, x],
                    default: None,
                };
                Ok(())
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "AndMaskSignExtToSelTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("AndMaskSignExtToSelTransform: selector must be bits[1]".to_string());
                }
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "AndMaskSignExtToSelTransform: output must be bits[w]".to_string()
                })?;
                let zero_case = cases[0];
                let x = cases[1];
                if !Self::is_zero_literal_node(f, zero_case, w) {
                    return Err(
                        "AndMaskSignExtToSelTransform: expected sel case0 to be 0_w literal"
                            .to_string(),
                    );
                }
                if Self::bits_width(f, x) != Some(w) {
                    return Err(
                        "AndMaskSignExtToSelTransform: expected sel case1 to be bits[w]".to_string(),
                    );
                }

                // Create sign_ext(b,w) node.
                let text_id = Self::next_text_id(f);
                let se_index = f.nodes.len();
                f.nodes.push(Node {
                    text_id,
                    name: None,
                    ty: Type::Bits(w),
                    payload: NodePayload::SignExt {
                        arg: selector,
                        new_bit_count: w,
                    },
                    pos: None,
                });
                let se_ref = NodeRef { index: se_index };

                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::And, vec![x, se_ref]);
                Ok(())
            }
            _ => Err(
                "AndMaskSignExtToSelTransform: expected and(...) or sel(...) payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `xor(x, sign_ext(b,w)) ↔ sel(b, cases=[x, not(x)])`
#[derive(Debug)]
pub struct XorMaskSignExtToSelNotTransform;

impl XorMaskSignExtToSelNotTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn sign_ext_mask_parts(f: &IrFn, r: NodeRef) -> Option<(NodeRef, usize)> {
        match &f.get_node(r).payload {
            NodePayload::SignExt { arg, new_bit_count } => {
                if !Self::is_u1_selector(f, *arg) {
                    return None;
                }
                if Self::bits_width(f, r) != Some(*new_bit_count) {
                    return None;
                }
                Some((*arg, *new_bit_count))
            }
            _ => None,
        }
    }

    fn not_arg(f: &IrFn, r: NodeRef) -> Option<NodeRef> {
        match &f.get_node(r).payload {
            NodePayload::Unop(Unop::Not, arg) => Some(*arg),
            _ => None,
        }
    }
}

impl PirTransform for XorMaskSignExtToSelNotTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::XorMaskSignExtToSelNot
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // Expand: xor(x, sign_ext(b,w))
                NodePayload::Nary(NaryOp::Xor, ops) if ops.len() == 2 => {
                    let (a, b) = (ops[0], ops[1]);
                    let mut ok = false;
                    for (x, mask) in [(a, b), (b, a)] {
                        if let Some((sel_b, w)) = Self::sign_ext_mask_parts(f, mask) {
                            if Self::bits_width(f, x) == Some(w)
                                && Self::bits_width(f, nr) == Some(w)
                            {
                                ok = true;
                                let _ = sel_b;
                            }
                        }
                    }
                    if ok {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                // Fold: sel(b, cases=[x, not(x)])
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let w = match Self::bits_width(f, nr) {
                        Some(w) => w,
                        None => continue,
                    };
                    let x = cases[0];
                    let not_x = cases[1];
                    if Self::bits_width(f, x) != Some(w) {
                        continue;
                    }
                    if Self::bits_width(f, not_x) != Some(w) {
                        continue;
                    }
                    if Self::not_arg(f, not_x) != Some(x) {
                        continue;
                    }
                    out.push(TransformLocation::Node(nr));
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "XorMaskSignExtToSelNotTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::Nary(NaryOp::Xor, ops) => {
                if ops.len() != 2 {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: expected 2-operand xor".to_string(),
                    );
                }
                let (a, b) = (ops[0], ops[1]);
                let mut matched: Option<(NodeRef, NodeRef, usize)> = None;
                for (x, mask) in [(a, b), (b, a)] {
                    if let Some((sel_b, w)) = Self::sign_ext_mask_parts(f, mask) {
                        if Self::bits_width(f, x) == Some(w) && Self::bits_width(f, target_ref) == Some(w) {
                            matched = Some((x, sel_b, w));
                            break;
                        }
                    }
                }
                let Some((x, sel_b, w)) = matched else {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: expected xor(x, sign_ext(b,w)) pattern"
                            .to_string(),
                    );
                };

                // Create not(x) node.
                let text_id = Self::next_text_id(f);
                let not_index = f.nodes.len();
                f.nodes.push(Node {
                    text_id,
                    name: None,
                    ty: Type::Bits(w),
                    payload: NodePayload::Unop(Unop::Not, x),
                    pos: None,
                });
                let not_x = NodeRef { index: not_index };

                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: sel_b,
                    cases: vec![x, not_x],
                    default: None,
                };
                Ok(())
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: selector must be bits[1]".to_string(),
                    );
                }
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "XorMaskSignExtToSelNotTransform: output must be bits[w]".to_string()
                })?;
                let x = cases[0];
                let not_x = cases[1];
                if Self::bits_width(f, x) != Some(w) || Self::bits_width(f, not_x) != Some(w) {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: sel cases must be bits[w]".to_string(),
                    );
                }
                if Self::not_arg(f, not_x) != Some(x) {
                    return Err(
                        "XorMaskSignExtToSelNotTransform: expected sel case1 to be not(case0)"
                            .to_string(),
                    );
                }

                // Create sign_ext(b,w) node.
                let text_id = Self::next_text_id(f);
                let se_index = f.nodes.len();
                f.nodes.push(Node {
                    text_id,
                    name: None,
                    ty: Type::Bits(w),
                    payload: NodePayload::SignExt {
                        arg: selector,
                        new_bit_count: w,
                    },
                    pos: None,
                });
                let se_ref = NodeRef { index: se_index };

                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::Xor, vec![x, se_ref]);
                Ok(())
            }
            _ => Err(
                "XorMaskSignExtToSelNotTransform: expected xor(...) or sel(...) payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `sel(p, cases=[a, a]) ↔ a`
///
/// Since there is no "alias" node payload in PIR, the fold direction rewrites
/// to `identity(a)`, which is semantics-preserving and still removes the `sel`.
#[derive(Debug)]
pub struct SelSameArmsFoldTransform;

impl SelSameArmsFoldTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn compute_fanout_cone(f: &IrFn, root: NodeRef) -> HashSet<NodeRef> {
        let users_map = compute_users(f);
        let mut visited: HashSet<NodeRef> = HashSet::new();
        let mut work: VecDeque<NodeRef> = VecDeque::new();
        visited.insert(root);
        work.push_back(root);
        while let Some(cur) = work.pop_front() {
            if let Some(users) = users_map.get(&cur) {
                for u in users {
                    if visited.insert(*u) {
                        work.push_back(*u);
                    }
                }
            }
        }
        visited
    }
}

impl PirTransform for SelSameArmsFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SelSameArmsFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();

        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                // Fold: sel(p, [a,a]) -> identity(a)
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if cases[0] != cases[1] {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    // Must be able to represent result as identity(bits[w]).
                    let w = Self::bits_width(f, cases[0]);
                    if w.is_some() && w == Self::bits_width(f, nr) {
                        out.push(TransformLocation::Node(nr));
                    }
                }

                // Expand: a -> sel(p, [a,a]) for some safe predicate p (bits[1]).
                //
                // We avoid GetParam/Nil targets (they have structural invariants) and
                // avoid choosing a selector that depends on the target (cycle risk).
                NodePayload::GetParam(_) | NodePayload::Nil => {}
                _ => {
                    let Some(w) = Self::bits_width(f, nr) else {
                        continue;
                    };

                    let fanout_cone = Self::compute_fanout_cone(f, nr);
                    let mut chosen_p: Option<NodeRef> = None;
                    for cand in f.node_refs() {
                        if cand == nr {
                            continue;
                        }
                        if !Self::is_u1_selector(f, cand) {
                            continue;
                        }
                        if fanout_cone.contains(&cand) {
                            // Would introduce a dependency cycle: nr -> cand and cand ->* nr.
                            continue;
                        }
                        chosen_p = Some(cand);
                        break;
                    }
                    if chosen_p.is_some() && w > 0 {
                        out.push(TransformLocation::Node(nr));
                    }
                }
            }
        }

        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "SelSameArmsFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // Fold: sel(p, [a,a]) -> identity(a)
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() || cases[0] != cases[1] {
                    return Err(
                        "SelSameArmsFoldTransform: expected 2-case sel with identical arms"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("SelSameArmsFoldTransform: selector must be bits[1]".to_string());
                }
                let a = cases[0];
                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "SelSameArmsFoldTransform: sel arms/output must be bits[w] with same width"
                            .to_string()
                    })?;
                let _ = w;

                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, a);
                Ok(())
            }

            // Expand: a -> sel(p, [a,a]) by cloning the current node into a new node,
            // then selecting between that new node twice.
            NodePayload::GetParam(_) | NodePayload::Nil => {
                Err("SelSameArmsFoldTransform: refusing to wrap GetParam/Nil nodes".to_string())
            }
            _ => {
                let Some(w) = Self::bits_width(f, target_ref) else {
                    return Err("SelSameArmsFoldTransform: only supports bits[w] nodes".to_string());
                };

                let fanout_cone = Self::compute_fanout_cone(f, target_ref);
                let mut chosen_p: Option<NodeRef> = None;
                for cand in f.node_refs() {
                    if cand == target_ref {
                        continue;
                    }
                    if !Self::is_u1_selector(f, cand) {
                        continue;
                    }
                    if fanout_cone.contains(&cand) {
                        continue;
                    }
                    chosen_p = Some(cand);
                    break;
                }
                let p = chosen_p.ok_or_else(|| {
                    "SelSameArmsFoldTransform: no safe bits[1] selector available".to_string()
                })?;

                // Clone the current node as a new node `a_clone`.
                let mut cloned = f.get_node(target_ref).clone();
                cloned.text_id = Self::next_text_id(f);
                cloned.name = None;
                let a_clone = NodeRef {
                    index: f.nodes.len(),
                };
                f.nodes.push(cloned);

                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![a_clone, a_clone],
                    default: None,
                };

                // Ensure the output type remains bits[w] (it does, since we didn't change ty).
                let _ = w;
                Ok(())
            }
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `sel(not(p), cases=[a, b]) ↔ sel(p, cases=[b, a])`
#[derive(Debug)]
pub struct SelSwapArmsByNotPredTransform;

impl SelSwapArmsByNotPredTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn not_arg(f: &IrFn, r: NodeRef) -> Option<NodeRef> {
        match &f.get_node(r).payload {
            NodePayload::Unop(Unop::Not, arg) => Some(*arg),
            _ => None,
        }
    }

    fn sel2_parts(payload: &NodePayload) -> Option<(NodeRef, NodeRef, NodeRef)> {
        match payload {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } if cases.len() == 2 && default.is_none() => Some((*selector, cases[0], cases[1])),
            _ => None,
        }
    }

    fn mk_not_node(f: &mut IrFn, p: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::Unop(Unop::Not, p),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for SelSwapArmsByNotPredTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SelSwapArmsByNotPred
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            let Some((sel, a, b)) = Self::sel2_parts(&f.get_node(nr).payload) else {
                continue;
            };
            let _ = (a, b);
            // We allow both directions:
            // - selector == not(p)  => remove not, swap arms
            // - selector == p       => add not, swap arms
            if Self::is_u1_selector(f, sel) {
                out.push(TransformLocation::Node(nr));
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "SelSwapArmsByNotPredTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::Sel {
            selector,
            cases,
            default,
        } = f.get_node(target_ref).payload.clone()
        else {
            return Err("SelSwapArmsByNotPredTransform: expected sel payload".to_string());
        };
        if cases.len() != 2 || default.is_some() {
            return Err(
                "SelSwapArmsByNotPredTransform: expected 2-case sel without default".to_string(),
            );
        }

        // swap arms
        let a = cases[0];
        let b = cases[1];

        let new_selector = if let Some(p) = Self::not_arg(f, selector) {
            if !Self::is_u1_selector(f, p) {
                return Err("SelSwapArmsByNotPredTransform: not(p) arg must be bits[1]".to_string());
            }
            p
        } else {
            if !Self::is_u1_selector(f, selector) {
                return Err("SelSwapArmsByNotPredTransform: selector must be bits[1]".to_string());
            }
            Self::mk_not_node(f, selector)
        };

        f.get_node_mut(target_ref).payload = NodePayload::Sel {
            selector: new_selector,
            cases: vec![b, a],
            default: None,
        };
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `not(not(x)) ↔ x`
///
/// Since there is no "alias" node payload in PIR, we represent `x` as
/// `identity(x)` at the IR node level.
#[derive(Debug)]
pub struct NotNotCancelTransform;

impl NotNotCancelTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_not_node(f: &mut IrFn, w: usize, x: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Unop(Unop::Not, x),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for NotNotCancelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NotNotCancel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Unop(Unop::Not, inner) => {
                    if matches!(f.get_node(*inner).payload, NodePayload::Unop(Unop::Not, _)) {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Unop(Unop::Identity, _) => {
                    out.push(TransformLocation::Node(nr));
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "NotNotCancelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // not(not(x)) -> identity(x)
            NodePayload::Unop(Unop::Not, inner) => {
                let NodePayload::Unop(Unop::Not, x) = f.get_node(inner).payload.clone() else {
                    return Err("NotNotCancelTransform: expected not(not(x))".to_string());
                };
                let w = Self::bits_width(f, x)
                    .filter(|wx| Some(*wx) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NotNotCancelTransform: expected bits[w] types for x and output".to_string()
                    })?;
                let _ = w;
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, x);
                Ok(())
            }

            // identity(x) -> not(not(x))
            NodePayload::Unop(Unop::Identity, x) => {
                let w = Self::bits_width(f, x)
                    .filter(|wx| Some(*wx) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NotNotCancelTransform: expected bits[w] types for x and output".to_string()
                    })?;
                let not1 = Self::mk_not_node(f, w, x);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, not1);
                Ok(())
            }
            _ => Err("NotNotCancelTransform: expected not(not(x)) or identity(x)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `neg(neg(x)) ↔ x`
///
/// Since there is no "alias" node payload in PIR, we represent `x` as
/// `identity(x)` at the IR node level.
#[derive(Debug)]
pub struct NegNegCancelTransform;

impl NegNegCancelTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_neg_node(f: &mut IrFn, w: usize, x: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Unop(Unop::Neg, x),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for NegNegCancelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NegNegCancel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Unop(Unop::Neg, inner) => {
                    if matches!(f.get_node(*inner).payload, NodePayload::Unop(Unop::Neg, _)) {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Unop(Unop::Identity, _) => out.push(TransformLocation::Node(nr)),
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "NegNegCancelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // neg(neg(x)) -> identity(x)
            NodePayload::Unop(Unop::Neg, inner) => {
                let NodePayload::Unop(Unop::Neg, x) = f.get_node(inner).payload.clone() else {
                    return Err("NegNegCancelTransform: expected neg(neg(x))".to_string());
                };
                let w = Self::bits_width(f, x)
                    .filter(|wx| Some(*wx) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NegNegCancelTransform: expected bits[w] types for x and output".to_string()
                    })?;
                let _ = w;
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, x);
                Ok(())
            }

            // identity(x) -> neg(neg(x))
            NodePayload::Unop(Unop::Identity, x) => {
                let w = Self::bits_width(f, x)
                    .filter(|wx| Some(*wx) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "NegNegCancelTransform: expected bits[w] types for x and output".to_string()
                    })?;
                let neg1 = Self::mk_neg_node(f, w, x);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Neg, neg1);
                Ok(())
            }
            _ => Err("NegNegCancelTransform: expected neg(neg(x)) or identity(x)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `not(eq(a,b)) ↔ ne(a,b)` and `not(ne(a,b)) ↔ eq(a,b)`
#[derive(Debug)]
pub struct NotEqNeFlipTransform;

impl NotEqNeFlipTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1(f: &IrFn, r: NodeRef) -> bool {
        matches!(f.get_node(r).ty, Type::Bits(1))
    }

    fn mk_cmp_node(f: &mut IrFn, op: Binop, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::Binop(op, lhs, rhs),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    // (no helper for building `not` nodes needed here)
}

impl PirTransform for NotEqNeFlipTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NotEqNeFlip
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Unop(Unop::Not, arg) => {
                    if matches!(
                        f.get_node(*arg).payload,
                        NodePayload::Binop(Binop::Eq, _, _) | NodePayload::Binop(Binop::Ne, _, _)
                    ) {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Binop(Binop::Eq, _, _) | NodePayload::Binop(Binop::Ne, _, _) => {
                    out.push(TransformLocation::Node(nr));
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "NotEqNeFlipTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // not(eq(..)) -> ne(..) and not(ne(..)) -> eq(..)
            NodePayload::Unop(Unop::Not, arg) => {
                if !Self::is_u1(f, target_ref) {
                    return Err("NotEqNeFlipTransform: output must be bits[1]".to_string());
                }
                let arg_payload = f.get_node(arg).payload.clone();
                match arg_payload {
                    NodePayload::Binop(Binop::Eq, lhs, rhs) => {
                        f.get_node_mut(target_ref).payload =
                            NodePayload::Binop(Binop::Ne, lhs, rhs);
                        Ok(())
                    }
                    NodePayload::Binop(Binop::Ne, lhs, rhs) => {
                        f.get_node_mut(target_ref).payload =
                            NodePayload::Binop(Binop::Eq, lhs, rhs);
                        Ok(())
                    }
                    _ => Err("NotEqNeFlipTransform: expected not(eq/ne(...))".to_string()),
                }
            }

            // eq/ne(a,b) -> not(ne/eq(a,b))
            NodePayload::Binop(op, lhs, rhs) if matches!(op, Binop::Eq | Binop::Ne) => {
                if !Self::is_u1(f, target_ref) {
                    return Err("NotEqNeFlipTransform: output must be bits[1]".to_string());
                }
                let flipped = if op == Binop::Eq {
                    Binop::Ne
                } else {
                    Binop::Eq
                };
                let cmp_ref = Self::mk_cmp_node(f, flipped, lhs, rhs);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, cmp_ref);
                Ok(())
            }
            _ => Err("NotEqNeFlipTransform: expected not(eq/ne(..)) or eq/ne(..)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `nor(xs...) ↔ not(or(xs...))`
#[derive(Debug)]
pub struct NorNotOrFoldTransform;

impl NorNotOrFoldTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_nary_bits_node(f: &mut IrFn, op: NaryOp, w: usize, ops: Vec<NodeRef>) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Nary(op, ops),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    // no dedicated `not` node builder needed; the transform rewrites in-place
}

impl PirTransform for NorNotOrFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NorNotOrFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Nary(NaryOp::Nor, ops) if !ops.is_empty() => {
                    out.push(TransformLocation::Node(nr));
                }
                NodePayload::Unop(Unop::Not, arg) => {
                    if matches!(f.get_node(*arg).payload, NodePayload::Nary(NaryOp::Or, _)) {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "NorNotOrFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // nor(xs...) -> not(or(xs...))
            NodePayload::Nary(NaryOp::Nor, ops) => {
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "NorNotOrFoldTransform: expected bits[w] output for nor".to_string()
                })?;
                for o in &ops {
                    if Self::bits_width(f, *o) != Some(w) {
                        return Err(
                            "NorNotOrFoldTransform: all operands must be bits[w] matching output"
                                .to_string(),
                        );
                    }
                }
                let or_ref = Self::mk_nary_bits_node(f, NaryOp::Or, w, ops);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, or_ref);
                Ok(())
            }

            // not(or(xs...)) -> nor(xs...)
            NodePayload::Unop(Unop::Not, arg) => {
                let NodePayload::Nary(NaryOp::Or, ops) = f.get_node(arg).payload.clone() else {
                    return Err("NorNotOrFoldTransform: expected not(or(...))".to_string());
                };
                let w = Self::bits_width(f, target_ref)
                    .ok_or_else(|| "NorNotOrFoldTransform: expected bits[w] output".to_string())?;
                for o in &ops {
                    if Self::bits_width(f, *o) != Some(w) {
                        return Err(
                            "NorNotOrFoldTransform: all operands must be bits[w] matching output"
                                .to_string(),
                        );
                    }
                }
                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::Nor, ops);
                Ok(())
            }
            _ => Err("NorNotOrFoldTransform: expected nor(...) or not(or(...))".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `nand(xs...) ↔ not(and(xs...))`
#[derive(Debug)]
pub struct NandNotAndFoldTransform;

impl NandNotAndFoldTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_nary_bits_node(f: &mut IrFn, op: NaryOp, w: usize, ops: Vec<NodeRef>) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Nary(op, ops),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    // no dedicated `not` node builder needed; the transform rewrites in-place
}

impl PirTransform for NandNotAndFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NandNotAndFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Nary(NaryOp::Nand, ops) if !ops.is_empty() => {
                    out.push(TransformLocation::Node(nr));
                }
                NodePayload::Unop(Unop::Not, arg) => {
                    if matches!(f.get_node(*arg).payload, NodePayload::Nary(NaryOp::And, _)) {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "NandNotAndFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // nand(xs...) -> not(and(xs...))
            NodePayload::Nary(NaryOp::Nand, ops) => {
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "NandNotAndFoldTransform: expected bits[w] output for nand".to_string()
                })?;
                for o in &ops {
                    if Self::bits_width(f, *o) != Some(w) {
                        return Err(
                            "NandNotAndFoldTransform: all operands must be bits[w] matching output"
                                .to_string(),
                        );
                    }
                }
                let and_ref = Self::mk_nary_bits_node(f, NaryOp::And, w, ops);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, and_ref);
                Ok(())
            }

            // not(and(xs...)) -> nand(xs...)
            NodePayload::Unop(Unop::Not, arg) => {
                let NodePayload::Nary(NaryOp::And, ops) = f.get_node(arg).payload.clone() else {
                    return Err("NandNotAndFoldTransform: expected not(and(...))".to_string());
                };
                let w = Self::bits_width(f, target_ref).ok_or_else(|| {
                    "NandNotAndFoldTransform: expected bits[w] output".to_string()
                })?;
                for o in &ops {
                    if Self::bits_width(f, *o) != Some(w) {
                        return Err(
                            "NandNotAndFoldTransform: all operands must be bits[w] matching output"
                                .to_string(),
                        );
                    }
                }
                f.get_node_mut(target_ref).payload = NodePayload::Nary(NaryOp::Nand, ops);
                Ok(())
            }
            _ => Err("NandNotAndFoldTransform: expected nand(...) or not(and(...))".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `eq(x, 0_w) ↔ not(or_reduce(x))`
#[derive(Debug)]
pub struct EqZeroOrReduceTransform;

impl EqZeroOrReduceTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        *v == expected
    }

    fn mk_zero_literal_node(f: &mut IrFn, w: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let value = IrValue::from_bits(&bits);
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Literal(value),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_or_reduce_node(f: &mut IrFn, x: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::Unop(Unop::OrReduce, x),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for EqZeroOrReduceTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::EqZeroOrReduce
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Binop(Binop::Eq, lhs, rhs) => {
                    let Some(w) = Self::bits_width(f, *lhs) else {
                        continue;
                    };
                    if Self::bits_width(f, *rhs) != Some(w) {
                        continue;
                    }
                    if Self::is_zero_literal_node(f, *lhs, w)
                        || Self::is_zero_literal_node(f, *rhs, w)
                    {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Unop(Unop::Not, arg) => {
                    if matches!(
                        f.get_node(*arg).payload,
                        NodePayload::Unop(Unop::OrReduce, _)
                    ) {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "EqZeroOrReduceTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        if !matches!(f.get_node(target_ref).ty, Type::Bits(1)) {
            return Err("EqZeroOrReduceTransform: output must be bits[1]".to_string());
        }

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // eq(x,0) -> not(or_reduce(x))
            NodePayload::Binop(Binop::Eq, lhs, rhs) => {
                let Some(w) = Self::bits_width(f, lhs) else {
                    return Err("EqZeroOrReduceTransform: x must be bits[w]".to_string());
                };
                if Self::bits_width(f, rhs) != Some(w) {
                    return Err(
                        "EqZeroOrReduceTransform: operands must have matching bits[w] types"
                            .to_string(),
                    );
                }
                let x = if Self::is_zero_literal_node(f, lhs, w) {
                    rhs
                } else if Self::is_zero_literal_node(f, rhs, w) {
                    lhs
                } else {
                    return Err("EqZeroOrReduceTransform: expected eq(x,0_w)".to_string());
                };
                let orr = Self::mk_or_reduce_node(f, x);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Not, orr);
                Ok(())
            }

            // not(or_reduce(x)) -> eq(x,0)
            NodePayload::Unop(Unop::Not, arg) => {
                let NodePayload::Unop(Unop::OrReduce, x) = f.get_node(arg).payload.clone() else {
                    return Err("EqZeroOrReduceTransform: expected not(or_reduce(x))".to_string());
                };
                let w = Self::bits_width(f, x).ok_or_else(|| {
                    "EqZeroOrReduceTransform: or_reduce arg must be bits[w]".to_string()
                })?;
                let zero = Self::mk_zero_literal_node(f, w);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Eq, x, zero);
                Ok(())
            }
            _ => Err("EqZeroOrReduceTransform: expected eq(x,0) or not(or_reduce(x))".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `ne(x, 0_w) ↔ or_reduce(x)`
#[derive(Debug)]
pub struct NeZeroOrReduceTransform;

impl NeZeroOrReduceTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn is_zero_literal_node(f: &IrFn, r: NodeRef, w: usize) -> bool {
        let NodePayload::Literal(v) = &f.get_node(r).payload else {
            return false;
        };
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let expected = IrValue::from_bits(&bits);
        *v == expected
    }

    fn mk_zero_literal_node(f: &mut IrFn, w: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        let bits = IrBits::make_ubits(w, 0).expect("make_ubits");
        let value = IrValue::from_bits(&bits);
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(w),
            payload: NodePayload::Literal(value),
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_or_reduce_node(f: &mut IrFn, x: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::Unop(Unop::OrReduce, x),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for NeZeroOrReduceTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::NeZeroOrReduce
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::Binop(Binop::Ne, lhs, rhs) => {
                    let Some(w) = Self::bits_width(f, *lhs) else {
                        continue;
                    };
                    if Self::bits_width(f, *rhs) != Some(w) {
                        continue;
                    }
                    if Self::is_zero_literal_node(f, *lhs, w)
                        || Self::is_zero_literal_node(f, *rhs, w)
                    {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                NodePayload::Unop(Unop::OrReduce, _) => out.push(TransformLocation::Node(nr)),
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "NeZeroOrReduceTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        if !matches!(f.get_node(target_ref).ty, Type::Bits(1)) {
            return Err("NeZeroOrReduceTransform: output must be bits[1]".to_string());
        }

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // ne(x,0) -> or_reduce(x)
            NodePayload::Binop(Binop::Ne, lhs, rhs) => {
                let Some(w) = Self::bits_width(f, lhs) else {
                    return Err("NeZeroOrReduceTransform: x must be bits[w]".to_string());
                };
                if Self::bits_width(f, rhs) != Some(w) {
                    return Err(
                        "NeZeroOrReduceTransform: operands must have matching bits[w] types"
                            .to_string(),
                    );
                }
                let x = if Self::is_zero_literal_node(f, lhs, w) {
                    rhs
                } else if Self::is_zero_literal_node(f, rhs, w) {
                    lhs
                } else {
                    return Err("NeZeroOrReduceTransform: expected ne(x,0_w)".to_string());
                };
                let orr = Self::mk_or_reduce_node(f, x);
                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, orr);
                Ok(())
            }

            // or_reduce(x) -> ne(x,0)
            NodePayload::Unop(Unop::OrReduce, x) => {
                let w = Self::bits_width(f, x).ok_or_else(|| {
                    "NeZeroOrReduceTransform: or_reduce arg must be bits[w]".to_string()
                })?;
                let zero = Self::mk_zero_literal_node(f, w);
                f.get_node_mut(target_ref).payload = NodePayload::Binop(Binop::Ne, x, zero);
                Ok(())
            }
            _ => Err("NeZeroOrReduceTransform: expected ne(x,0) or or_reduce(x)".to_string()),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform implementing:
///
/// `bit_slice(bit_slice(x, s1, w1), s2, w2) ↔ bit_slice(x, s1+s2, w2)`
#[derive(Debug)]
pub struct BitSliceBitSliceFoldTransform;

impl BitSliceBitSliceFoldTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_bit_slice_node(
        f: &mut IrFn,
        out_w: usize,
        arg: NodeRef,
        start: usize,
        width: usize,
    ) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::BitSlice { arg, start, width },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for BitSliceBitSliceFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BitSliceBitSliceFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                NodePayload::BitSlice { arg, .. } => {
                    if matches!(f.get_node(*arg).payload, NodePayload::BitSlice { .. }) {
                        out.push(TransformLocation::Node(nr));
                    } else {
                        // allow expansion too
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "BitSliceBitSliceFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::BitSlice {
            arg,
            start: s2,
            width: w2,
        } = f.get_node(target_ref).payload.clone()
        else {
            return Err("BitSliceBitSliceFoldTransform: expected bit_slice payload".to_string());
        };
        if Self::bits_width(f, target_ref) != Some(w2) {
            return Err(
                "BitSliceBitSliceFoldTransform: output type must be bits[width]".to_string(),
            );
        }

        // Fold if arg is also a bit_slice.
        if let NodePayload::BitSlice {
            arg: x,
            start: s1,
            width: w1,
        } = f.get_node(arg).payload.clone()
        {
            let in_w = Self::bits_width(f, x).ok_or_else(|| {
                "BitSliceBitSliceFoldTransform: input must be bits[w]".to_string()
            })?;
            if s1.saturating_add(w1) > in_w {
                return Err("BitSliceBitSliceFoldTransform: inner slice out of bounds".to_string());
            }
            if s2.saturating_add(w2) > w1 {
                return Err("BitSliceBitSliceFoldTransform: outer slice out of bounds".to_string());
            }
            let s = s1.saturating_add(s2);
            f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                arg: x,
                start: s,
                width: w2,
            };
            return Ok(());
        }

        // Expand: bit_slice(x,s,w) -> bit_slice(bit_slice(x,0,w1),s,w) with w1=s+w.
        let x = arg;
        let in_w = Self::bits_width(f, x)
            .ok_or_else(|| "BitSliceBitSliceFoldTransform: input must be bits[w]".to_string())?;
        let w1 = s2.saturating_add(w2);
        if w1 > in_w {
            return Err("BitSliceBitSliceFoldTransform: slice out of bounds".to_string());
        }
        let inner = Self::mk_bit_slice_node(f, w1, x, 0, w1);
        f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
            arg: inner,
            start: s2,
            width: w2,
        };
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform distributing `bit_slice` over 2-operand
/// concat:
///
/// `bit_slice(concat(a,b), start=s, width=w) ↔ ...` (in-a / in-b / straddle)
///
/// Note: concat is interpreted as `concat(msb=a, lsb=b)` (so `b` occupies the
/// low bits).
#[derive(Debug)]
pub struct BitSliceConcatDistributeTransform;

impl BitSliceConcatDistributeTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_bit_slice_node(
        f: &mut IrFn,
        out_w: usize,
        arg: NodeRef,
        start: usize,
        width: usize,
    ) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::BitSlice { arg, start, width },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_concat_node(f: &mut IrFn, out_w: usize, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(out_w),
            payload: NodePayload::Nary(NaryOp::Concat, vec![a, b]),
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for BitSliceConcatDistributeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BitSliceConcatDistribute
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            if let NodePayload::BitSlice { arg, .. } = &f.get_node(nr).payload {
                if matches!(
                    f.get_node(*arg).payload,
                    NodePayload::Nary(NaryOp::Concat, _)
                ) {
                    out.push(TransformLocation::Node(nr));
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "BitSliceConcatDistributeTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::BitSlice { arg, start, width } = f.get_node(target_ref).payload.clone()
        else {
            return Err(
                "BitSliceConcatDistributeTransform: expected bit_slice payload".to_string(),
            );
        };
        if Self::bits_width(f, target_ref) != Some(width) {
            return Err(
                "BitSliceConcatDistributeTransform: output must be bits[width]".to_string(),
            );
        }

        let NodePayload::Nary(NaryOp::Concat, ops) = f.get_node(arg).payload.clone() else {
            return Err("BitSliceConcatDistributeTransform: expected concat arg".to_string());
        };
        if ops.len() != 2 {
            return Err(
                "BitSliceConcatDistributeTransform: only supports 2-operand concat".to_string(),
            );
        }
        let a = ops[0];
        let b = ops[1];
        let wa = Self::bits_width(f, a)
            .ok_or_else(|| "BitSliceConcatDistributeTransform: a must be bits[wa]".to_string())?;
        let wb = Self::bits_width(f, b)
            .ok_or_else(|| "BitSliceConcatDistributeTransform: b must be bits[wb]".to_string())?;
        let total = wa.saturating_add(wb);
        if start.saturating_add(width) > total {
            return Err("BitSliceConcatDistributeTransform: slice out of bounds".to_string());
        }

        // b occupies low bits [0..wb), a occupies high bits [wb..wb+wa).
        if start.saturating_add(width) <= wb {
            // entirely within b
            f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                arg: b,
                start,
                width,
            };
            return Ok(());
        }
        if start >= wb {
            // entirely within a
            f.get_node_mut(target_ref).payload = NodePayload::BitSlice {
                arg: a,
                start: start - wb,
                width,
            };
            return Ok(());
        }

        // Straddle: low part from b[start..wb), high part from a[0..(start+width-wb)).
        let width_b = wb - start;
        let width_a = width - width_b;
        let bs_b = Self::mk_bit_slice_node(f, width_b, b, start, width_b);
        let bs_a = Self::mk_bit_slice_node(f, width_a, a, 0, width_a);
        let cat = Self::mk_concat_node(f, width, bs_a, bs_b);
        f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, cat);
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A semantics-preserving transform lowering a `priority_sel(bits[M])` into a
/// sel-chain.
///
/// Reverse direction is intentionally not implemented.
#[derive(Debug)]
pub struct PrioritySelToSelChainTransform;

impl PrioritySelToSelChainTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn mk_bit_i(f: &mut IrFn, selector: NodeRef, i: usize) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty: Type::Bits(1),
            payload: NodePayload::BitSlice {
                arg: selector,
                start: i,
                width: 1,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }

    fn mk_sel2(f: &mut IrFn, ty: Type, p: NodeRef, a: NodeRef, b: NodeRef) -> NodeRef {
        let text_id = Self::next_text_id(f);
        let new_index = f.nodes.len();
        f.nodes.push(Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Sel {
                selector: p,
                cases: vec![a, b],
                default: None,
            },
            pos: None,
        });
        NodeRef { index: new_index }
    }
}

impl PirTransform for PrioritySelToSelChainTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::PrioritySelToSelChain
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            if let NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } = &f.get_node(nr).payload
            {
                if default.is_none() {
                    continue;
                }
                let Some(m) = Self::bits_width(f, *selector) else {
                    continue;
                };
                if cases.len() == m && m > 0 {
                    out.push(TransformLocation::Node(nr));
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "PrioritySelToSelChainTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } = f.get_node(target_ref).payload.clone()
        else {
            return Err(
                "PrioritySelToSelChainTransform: expected priority_sel payload".to_string(),
            );
        };
        let Some(d) = default else {
            return Err("PrioritySelToSelChainTransform: expected default".to_string());
        };
        let Some(m) = Self::bits_width(f, selector) else {
            return Err("PrioritySelToSelChainTransform: selector must be bits[M]".to_string());
        };
        if cases.len() != m || m == 0 {
            return Err("PrioritySelToSelChainTransform: cases.len() must equal M".to_string());
        }

        // Output type must be bits[w] for this lowering (keeps it simple).
        let out_ty = f.get_node(target_ref).ty.clone();
        if !matches!(out_ty, Type::Bits(_)) {
            return Err("PrioritySelToSelChainTransform: only supports bits outputs".to_string());
        }

        let mut acc = d;
        for i in (0..m).rev() {
            let bit_i = Self::mk_bit_i(f, selector, i);
            // sel(bit_i, cases=[acc, cases[i]])
            acc = Self::mk_sel2(f, out_ty.clone(), bit_i, acc, cases[i]);
        }
        f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, acc);
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}

/// A transform that rewires one operand of a node to a different node with the
/// same type.
///
/// This is not semantics-preserving in general. It is intended to be used with
/// an equivalence oracle in the MCMC loop.
#[derive(Debug)]
pub struct RewireOperandToSameTypeTransform;

impl RewireOperandToSameTypeTransform {
    const MAX_CANDIDATES: usize = 2000;

    fn operand_pairs(payload: &NodePayload) -> Vec<(usize, NodeRef)> {
        let mut pairs: Vec<(usize, NodeRef)> = Vec::new();
        let _ = remap_payload_with(payload, |(slot, dep)| {
            pairs.push((slot, dep));
            dep
        });
        pairs
    }

    fn node_type(f: &IrFn, nr: NodeRef) -> Type {
        f.get_node(nr).ty.clone()
    }

    fn compute_fanout_cone(
        users_map: &std::collections::HashMap<NodeRef, HashSet<NodeRef>>,
        root: NodeRef,
    ) -> HashSet<NodeRef> {
        let mut visited: HashSet<NodeRef> = HashSet::new();
        let mut work: VecDeque<NodeRef> = VecDeque::new();
        visited.insert(root);
        work.push_back(root);

        while let Some(cur) = work.pop_front() {
            if let Some(users) = users_map.get(&cur) {
                for u in users {
                    if visited.insert(*u) {
                        work.push_back(*u);
                    }
                }
            }
        }
        visited
    }
}

impl PirTransform for RewireOperandToSameTypeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::RewireOperandToSameType
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let users_map = compute_users(f);

        // Deterministic enumeration order:
        //   - node index ascending
        //   - operand slot ascending (as reported by remap_payload_with)
        //   - replacement node index ascending
        let mut out: Vec<TransformLocation> = Vec::new();

        for i in 0..f.nodes.len() {
            if out.len() >= Self::MAX_CANDIDATES {
                break;
            }

            let node_ref = NodeRef { index: i };
            let node = f.get_node(node_ref);
            let pairs = Self::operand_pairs(&node.payload);
            if pairs.is_empty() {
                continue;
            }

            // To avoid introducing cycles, we disallow rewiring a node's operand to any
            // node in the node's fanout cone (i.e., any node that depends on
            // `node_ref`).
            //
            // If `new_operand` depends on `node_ref`, then adding an edge `node_ref ->
            // new_operand` would create a cycle.
            let fanout_cone = Self::compute_fanout_cone(&users_map, node_ref);

            for (slot, old_dep) in pairs {
                if out.len() >= Self::MAX_CANDIDATES {
                    break;
                }
                let old_ty = Self::node_type(f, old_dep);
                for repl_i in 0..f.nodes.len() {
                    if out.len() >= Self::MAX_CANDIDATES {
                        break;
                    }
                    let new_operand = NodeRef { index: repl_i };
                    if new_operand == node_ref {
                        // Self-dependency is always a cycle.
                        continue;
                    }
                    if new_operand == old_dep {
                        continue;
                    }
                    if fanout_cone.contains(&new_operand) {
                        // Would introduce a cycle.
                        continue;
                    }
                    if Self::node_type(f, new_operand) != old_ty {
                        continue;
                    }
                    out.push(TransformLocation::RewireOperand {
                        node: node_ref,
                        operand_slot: slot,
                        new_operand,
                    });
                }
            }
        }

        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (node_ref, operand_slot, new_operand) = match loc {
            TransformLocation::RewireOperand {
                node,
                operand_slot,
                new_operand,
            } => (*node, *operand_slot, *new_operand),
            TransformLocation::Node(_) => {
                return Err(
                    "RewireOperandToSameType: expected TransformLocation::RewireOperand, got Node"
                        .to_string(),
                );
            }
        };

        if node_ref.index >= f.nodes.len() || new_operand.index >= f.nodes.len() {
            return Err("RewireOperandToSameType: node ref out of bounds".to_string());
        }
        if node_ref == new_operand {
            return Err(
                "RewireOperandToSameType: cannot rewire an operand to the node itself".to_string(),
            );
        }

        // Safety check: reject rewires that would introduce cycles.
        let users_map = compute_users(f);
        let fanout_cone = Self::compute_fanout_cone(&users_map, node_ref);
        if fanout_cone.contains(&new_operand) {
            return Err("RewireOperandToSameType: rewire would introduce a cycle".to_string());
        }

        let old_payload = f.get_node(node_ref).payload.clone();
        let new_payload = remap_payload_with(&old_payload, |(slot, dep)| {
            if slot == operand_slot {
                new_operand
            } else {
                dep
            }
        });

        f.get_node_mut(node_ref).payload = new_payload;
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}

/// Returns the default set of PIR transforms used by the PIR MCMC engine.
pub fn get_all_pir_transforms() -> Vec<Box<dyn PirTransform>> {
    vec![
        Box::new(SwapCommutativeBinopOperandsTransform),
        Box::new(CloneMultiUserNodeTransform),
        Box::new(EqSelDistributeTransform),
        Box::new(EqNeAddLiteralShiftTransform),
        Box::new(NotSelDistributeTransform),
        Box::new(NegSelDistributeTransform),
        Box::new(BitSliceSelDistributeTransform),
        Box::new(SignExtSelDistributeTransform),
        Box::new(PrioritySel1ToSelTransform),
        Box::new(AndMaskSignExtToSelTransform),
        Box::new(XorMaskSignExtToSelNotTransform),
        Box::new(SelSameArmsFoldTransform),
        Box::new(SelSwapArmsByNotPredTransform),
        Box::new(NotNotCancelTransform),
        Box::new(NegNegCancelTransform),
        Box::new(NotEqNeFlipTransform),
        Box::new(NorNotOrFoldTransform),
        Box::new(NandNotAndFoldTransform),
        Box::new(EqZeroOrReduceTransform),
        Box::new(NeZeroOrReduceTransform),
        Box::new(BitSliceBitSliceFoldTransform),
        Box::new(BitSliceConcatDistributeTransform),
        Box::new(PrioritySelToSelChainTransform),
        Box::new(RewireOperandToSameTypeTransform),
    ]
}

/// Returns a vector of weights for the given transforms.
///
/// For now all PIR transforms are given equal weight.
pub fn build_transform_weights<T: AsRef<[Box<dyn PirTransform>]>>(transforms: T) -> Vec<f64> {
    transforms.as_ref().iter().map(|_| 1.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    #[test]
    fn eq_sel_distribute_expands_eq_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[1] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret eq.20: bits[1] = eq(sel.10, c, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut eq_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Eq, _, _)) {
                eq_ref = Some(nr);
            }
        }
        let eq_ref = eq_ref.expect("expected eq node");

        let t = EqSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(eq_ref))
            .expect("apply");

        match &f.get_node(eq_ref).payload {
            NodePayload::Sel {
                selector: _,
                cases,
                default,
            } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::Binop(Binop::Eq, _, _)
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_sel_distribute_folds_sel_of_eqs() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[1] {
  eq.11: bits[1] = eq(a, c, id=11)
  eq.12: bits[1] = eq(b, c, id=12)
  ret sel.20: bits[1] = sel(p, cases=[eq.11, eq.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = EqSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Binop(Binop::Eq, _, _)
        ));
    }

    #[test]
    fn eq_sel_distribute_expands_ne_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[1] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret ne.20: bits[1] = ne(sel.10, c, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ne_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Ne, _, _)) {
                ne_ref = Some(nr);
            }
        }
        let ne_ref = ne_ref.expect("expected ne node");

        let t = EqSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(ne_ref))
            .expect("apply");

        match &f.get_node(ne_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::Binop(Binop::Ne, _, _)
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_sel_distribute_folds_sel_of_nes() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3, c: bits[8] id=4) -> bits[1] {
  ne.11: bits[1] = ne(a, c, id=11)
  ne.12: bits[1] = ne(b, c, id=12)
  ret sel.20: bits[1] = sel(p, cases=[ne.11, ne.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = EqSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Binop(Binop::Ne, _, _)
        ));
    }

    #[test]
    fn eq_ne_add_literal_shift_expands_eq_add_lit() {
        let ir_text = r#"fn t(x: bits[8] id=1, c: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  add.11: bits[8] = add(x, literal.10, id=11)
  ret eq.20: bits[1] = eq(add.11, c, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut eq_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Eq, _, _)) {
                eq_ref = Some(nr);
            }
        }
        let eq_ref = eq_ref.expect("expected eq node");

        let t = EqNeAddLiteralShiftTransform;
        t.apply(&mut f, &TransformLocation::Node(eq_ref))
            .expect("apply");

        match &f.get_node(eq_ref).payload {
            NodePayload::Binop(Binop::Eq, _x, rhs) => {
                assert!(matches!(
                    f.get_node(*rhs).payload,
                    NodePayload::Binop(Binop::Sub, _, _)
                ));
            }
            other => panic!("expected eq(x, sub(...)) after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_ne_add_literal_shift_expands_ne_add_lit() {
        let ir_text = r#"fn t(x: bits[8] id=1, c: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  add.11: bits[8] = add(literal.10, x, id=11)
  ret ne.20: bits[1] = ne(add.11, c, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ne_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Ne, _, _)) {
                ne_ref = Some(nr);
            }
        }
        let ne_ref = ne_ref.expect("expected ne node");

        let t = EqNeAddLiteralShiftTransform;
        t.apply(&mut f, &TransformLocation::Node(ne_ref))
            .expect("apply");

        match &f.get_node(ne_ref).payload {
            NodePayload::Binop(Binop::Ne, _x, rhs) => {
                assert!(matches!(
                    f.get_node(*rhs).payload,
                    NodePayload::Binop(Binop::Sub, _, _)
                ));
            }
            other => panic!("expected ne(x, sub(...)) after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_ne_add_literal_shift_folds_eq_sub_lit() {
        let ir_text = r#"fn t(x: bits[8] id=1, c: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  sub.11: bits[8] = sub(c, literal.10, id=11)
  ret eq.20: bits[1] = eq(x, sub.11, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut eq_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Eq, _, _)) {
                eq_ref = Some(nr);
            }
        }
        let eq_ref = eq_ref.expect("expected eq node");

        let t = EqNeAddLiteralShiftTransform;
        t.apply(&mut f, &TransformLocation::Node(eq_ref))
            .expect("apply");

        match &f.get_node(eq_ref).payload {
            NodePayload::Binop(Binop::Eq, lhs, _c) => {
                assert!(matches!(
                    f.get_node(*lhs).payload,
                    NodePayload::Binop(Binop::Add, _, _)
                ));
            }
            other => panic!("expected eq(add(...), c) after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn eq_ne_add_literal_shift_folds_ne_sub_lit() {
        let ir_text = r#"fn t(x: bits[8] id=1, c: bits[8] id=2) -> bits[1] {
  literal.10: bits[8] = literal(value=7, id=10)
  sub.11: bits[8] = sub(c, literal.10, id=11)
  ret ne.20: bits[1] = ne(sub.11, x, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ne_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Ne, _, _)) {
                ne_ref = Some(nr);
            }
        }
        let ne_ref = ne_ref.expect("expected ne node");

        let t = EqNeAddLiteralShiftTransform;
        t.apply(&mut f, &TransformLocation::Node(ne_ref))
            .expect("apply");

        match &f.get_node(ne_ref).payload {
            NodePayload::Binop(Binop::Ne, lhs, _c) => {
                assert!(matches!(
                    f.get_node(*lhs).payload,
                    NodePayload::Binop(Binop::Add, _, _)
                ));
            }
            other => panic!("expected ne(add(...), c) after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn not_sel_distribute_expands_not_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret not.20: bits[8] = not(sel.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut not_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Not, _)) {
                not_ref = Some(nr);
            }
        }
        let not_ref = not_ref.expect("expected not node");

        let t = NotSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(not_ref))
            .expect("apply");

        match &f.get_node(not_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::Unop(Unop::Not, _)
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn not_sel_distribute_folds_sel_of_nots() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  not.11: bits[8] = not(a, id=11)
  not.12: bits[8] = not(b, id=12)
  ret sel.20: bits[8] = sel(p, cases=[not.11, not.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = NotSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
    }

    #[test]
    fn neg_sel_distribute_expands_neg_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret neg.20: bits[8] = neg(sel.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut neg_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Neg, _)) {
                neg_ref = Some(nr);
            }
        }
        let neg_ref = neg_ref.expect("expected neg node");

        let t = NegSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(neg_ref))
            .expect("apply");

        match &f.get_node(neg_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::Unop(Unop::Neg, _)
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn neg_sel_distribute_folds_sel_of_negs() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  neg.11: bits[8] = neg(a, id=11)
  neg.12: bits[8] = neg(b, id=12)
  ret sel.20: bits[8] = sel(p, cases=[neg.11, neg.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = NegSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Unop(Unop::Neg, _)
        ));
    }

    #[test]
    fn bit_slice_sel_distribute_expands_bit_slice_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[3] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret bit_slice.20: bits[3] = bit_slice(sel.10, start=1, width=3, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut bs_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::BitSlice { .. }) {
                bs_ref = Some(nr);
            }
        }
        let bs_ref = bs_ref.expect("expected bit_slice node");

        let t = BitSliceSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(bs_ref))
            .expect("apply");

        match &f.get_node(bs_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::BitSlice {
                            start: 1,
                            width: 3,
                            ..
                        }
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn bit_slice_sel_distribute_folds_sel_of_bit_slices() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[3] {
  bit_slice.11: bits[3] = bit_slice(a, start=1, width=3, id=11)
  bit_slice.12: bits[3] = bit_slice(b, start=1, width=3, id=12)
  ret sel.20: bits[3] = sel(p, cases=[bit_slice.11, bit_slice.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = BitSliceSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::BitSlice {
                start: 1,
                width: 3,
                ..
            }
        ));
    }

    #[test]
    fn sign_ext_sel_distribute_expands_sign_ext_of_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[16] {
  sel.10: bits[8] = sel(p, cases=[a, b], id=10)
  ret sign_ext.20: bits[16] = sign_ext(sel.10, new_bit_count=16, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut se_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::SignExt { .. }) {
                se_ref = Some(nr);
            }
        }
        let se_ref = se_ref.expect("expected sign_ext node");

        let t = SignExtSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(se_ref))
            .expect("apply");

        match &f.get_node(se_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
                for case in cases {
                    assert!(matches!(
                        f.get_node(*case).payload,
                        NodePayload::SignExt {
                            new_bit_count: 16,
                            ..
                        }
                    ));
                }
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn sign_ext_sel_distribute_folds_sel_of_sign_exts() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[16] {
  sign_ext.11: bits[16] = sign_ext(a, new_bit_count=16, id=11)
  sign_ext.12: bits[16] = sign_ext(b, new_bit_count=16, id=12)
  ret sel.20: bits[16] = sel(p, cases=[sign_ext.11, sign_ext.12], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = SignExtSelDistributeTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::SignExt {
                new_bit_count: 16,
                ..
            }
        ));
    }

    #[test]
    fn priority_sel1_to_sel_expands_priority_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  ret priority_sel.10: bits[8] = priority_sel(p, cases=[a], default=b, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ps_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::PrioritySel { .. }) {
                ps_ref = Some(nr);
            }
        }
        let ps_ref = ps_ref.expect("expected priority_sel node");

        let t = PrioritySel1ToSelTransform;
        t.apply(&mut f, &TransformLocation::Node(ps_ref))
            .expect("apply");

        match &f.get_node(ps_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert!(default.is_none());
                assert_eq!(cases.len(), 2);
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn priority_sel1_to_sel_folds_sel() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  ret sel.10: bits[8] = sel(p, cases=[b, a], id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = PrioritySel1ToSelTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::PrioritySel { .. }
        ));
    }

    #[test]
    fn and_mask_sign_ext_to_sel_expands_and() {
        let ir_text = r#"fn t(b: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  sign_ext.10: bits[8] = sign_ext(b, new_bit_count=8, id=10)
  ret and.20: bits[8] = and(x, sign_ext.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut and_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Nary(NaryOp::And, _)) {
                and_ref = Some(nr);
            }
        }
        let and_ref = and_ref.expect("expected and node");

        let t = AndMaskSignExtToSelTransform;
        t.apply(&mut f, &TransformLocation::Node(and_ref))
            .expect("apply");

        match &f.get_node(and_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert!(default.is_none());
                assert_eq!(cases.len(), 2);
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn and_mask_sign_ext_to_sel_folds_sel() {
        let ir_text = r#"fn t(b: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  literal.10: bits[8] = literal(value=0, id=10)
  ret sel.20: bits[8] = sel(b, cases=[literal.10, x], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = AndMaskSignExtToSelTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Nary(NaryOp::And, _)
        ));
    }

    #[test]
    fn xor_mask_sign_ext_to_sel_not_expands_xor() {
        let ir_text = r#"fn t(b: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  sign_ext.10: bits[8] = sign_ext(b, new_bit_count=8, id=10)
  ret xor.20: bits[8] = xor(x, sign_ext.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut xor_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Nary(NaryOp::Xor, _)) {
                xor_ref = Some(nr);
            }
        }
        let xor_ref = xor_ref.expect("expected xor node");

        let t = XorMaskSignExtToSelNotTransform;
        t.apply(&mut f, &TransformLocation::Node(xor_ref))
            .expect("apply");

        match &f.get_node(xor_ref).payload {
            NodePayload::Sel { cases, default, .. } => {
                assert!(default.is_none());
                assert_eq!(cases.len(), 2);
                assert!(matches!(
                    f.get_node(cases[1]).payload,
                    NodePayload::Unop(Unop::Not, _)
                ));
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn xor_mask_sign_ext_to_sel_not_folds_sel() {
        let ir_text = r#"fn t(b: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  not.10: bits[8] = not(x, id=10)
  ret sel.20: bits[8] = sel(b, cases=[x, not.10], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = XorMaskSignExtToSelNotTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Nary(NaryOp::Xor, _)
        ));
    }

    #[test]
    fn sel_same_arms_fold_folds_sel_to_identity() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2) -> bits[8] {
  ret sel.10: bits[8] = sel(p, cases=[a, a], id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = SelSameArmsFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(sel_ref).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }

    #[test]
    fn sel_swap_arms_by_not_pred_removes_not_and_swaps() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  not.10: bits[1] = not(p, id=10)
  ret sel.20: bits[8] = sel(not.10, cases=[a, b], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = SelSwapArmsByNotPredTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        match &f.get_node(sel_ref).payload {
            NodePayload::Sel {
                selector, cases, ..
            } => {
                assert_eq!(cases.len(), 2);
                // selector should now be `p`, not a `not(...)` node.
                assert!(matches!(
                    f.get_node(*selector).payload,
                    NodePayload::GetParam(_)
                ));
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn sel_swap_arms_by_not_pred_adds_not_and_swaps() {
        let ir_text = r#"fn t(p: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {
  ret sel.20: bits[8] = sel(p, cases=[b, a], id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut sel_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Sel { .. }) {
                sel_ref = Some(nr);
            }
        }
        let sel_ref = sel_ref.expect("expected sel node");

        let t = SelSwapArmsByNotPredTransform;
        t.apply(&mut f, &TransformLocation::Node(sel_ref))
            .expect("apply");

        match &f.get_node(sel_ref).payload {
            NodePayload::Sel {
                selector, cases, ..
            } => {
                assert_eq!(cases.len(), 2);
                assert!(matches!(
                    f.get_node(*selector).payload,
                    NodePayload::Unop(Unop::Not, _)
                ));
            }
            other => panic!("expected sel after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn not_not_cancel_folds_double_not() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[8] {
  not.10: bits[8] = not(x, id=10)
  ret not.11: bits[8] = not(not.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut out_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Not, _)) {
                out_ref = Some(nr);
            }
        }
        let out_ref = out_ref.expect("expected outer not node");

        let t = NotNotCancelTransform;
        t.apply(&mut f, &TransformLocation::Node(out_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(out_ref).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }

    #[test]
    fn neg_neg_cancel_folds_double_neg() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[8] {
  neg.10: bits[8] = neg(x, id=10)
  ret neg.11: bits[8] = neg(neg.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut out_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Neg, _)) {
                out_ref = Some(nr);
            }
        }
        let out_ref = out_ref.expect("expected outer neg node");

        let t = NegNegCancelTransform;
        t.apply(&mut f, &TransformLocation::Node(out_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(out_ref).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }

    #[test]
    fn not_eq_ne_flip_turns_not_eq_into_ne() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[1] {
  eq.10: bits[1] = eq(a, b, id=10)
  ret not.11: bits[1] = not(eq.10, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut not_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Not, _)) {
                not_ref = Some(nr);
            }
        }
        let not_ref = not_ref.expect("expected not node");

        let t = NotEqNeFlipTransform;
        t.apply(&mut f, &TransformLocation::Node(not_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(not_ref).payload,
            NodePayload::Binop(Binop::Ne, _, _)
        ));
    }

    #[test]
    fn nor_not_or_fold_expands_nor() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret nor.10: bits[8] = nor(a, b, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut nor_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Nary(NaryOp::Nor, _)) {
                nor_ref = Some(nr);
            }
        }
        let nor_ref = nor_ref.expect("expected nor node");

        let t = NorNotOrFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(nor_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(nor_ref).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
    }

    #[test]
    fn eq_zero_or_reduce_expands_eq_zero() {
        let ir_text = r#"fn t(x: bits[8] id=1) -> bits[1] {
  literal.10: bits[8] = literal(value=0, id=10)
  ret eq.20: bits[1] = eq(x, literal.10, id=20)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut eq_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Binop(Binop::Eq, _, _)) {
                eq_ref = Some(nr);
            }
        }
        let eq_ref = eq_ref.expect("expected eq node");

        let t = EqZeroOrReduceTransform;
        t.apply(&mut f, &TransformLocation::Node(eq_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(eq_ref).payload,
            NodePayload::Unop(Unop::Not, _)
        ));
    }

    #[test]
    fn bit_slice_bit_slice_fold_folds_nested_slices() {
        let ir_text = r#"fn t(x: bits[16] id=1) -> bits[4] {
  bit_slice.10: bits[8] = bit_slice(x, start=4, width=8, id=10)
  ret bit_slice.11: bits[4] = bit_slice(bit_slice.10, start=2, width=4, id=11)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut out_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::BitSlice { .. }) {
                out_ref = Some(nr);
            }
        }
        let out_ref = out_ref.expect("expected outer bit_slice node");

        let t = BitSliceBitSliceFoldTransform;
        t.apply(&mut f, &TransformLocation::Node(out_ref))
            .expect("apply");

        match &f.get_node(out_ref).payload {
            NodePayload::BitSlice { start, width, .. } => {
                assert_eq!(*start, 6);
                assert_eq!(*width, 4);
            }
            other => panic!("expected bit_slice after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn priority_sel_to_sel_chain_lowers() {
        let ir_text = r#"fn t(p: bits[2] id=1, c0: bits[8] id=2, c1: bits[8] id=3, d: bits[8] id=4) -> bits[8] {
  ret priority_sel.10: bits[8] = priority_sel(p, cases=[c0, c1], default=d, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();

        let mut ps_ref: Option<NodeRef> = None;
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::PrioritySel { .. }) {
                ps_ref = Some(nr);
            }
        }
        let ps_ref = ps_ref.expect("expected priority_sel node");

        let t = PrioritySelToSelChainTransform;
        t.apply(&mut f, &TransformLocation::Node(ps_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(ps_ref).payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
    }
}
