// SPDX-License-Identifier: Apache-2.0

use super::*;
use xlsynth_pir::ir::{ExtNaryAddArchitecture, ExtNaryAddTerm};

/// Rewrites `add(a, b)` into
/// `ext_nary_add(a, b, signed=[false, false], negated=[false, false],
/// arch=brent_kung)`.
#[derive(Debug)]
pub struct AddToExtNaryAddTransform;

/// Rewrites `sub(a, b)` into
/// `ext_nary_add(a, b, signed=[false, false], negated=[false, true],
/// arch=brent_kung)`.
#[derive(Debug)]
pub struct SubToExtNaryAddTransform;

/// Rewrites an explicit `ext_nary_add` architecture to `ripple_carry`.
#[derive(Debug)]
pub struct ChangeExtNaryAddToRippleCarryTransform;

/// Rewrites an explicit `ext_nary_add` architecture to `brent_kung`.
#[derive(Debug)]
pub struct ChangeExtNaryAddToBrentKungTransform;

/// Rewrites an explicit `ext_nary_add` architecture to `kogge_stone`.
#[derive(Debug)]
pub struct ChangeExtNaryAddToKoggeStoneTransform;

/// Rewrites a width-preserving 2-input `ext_nary_add` back to plain `add`.
#[derive(Debug)]
pub struct BinaryExtNaryAddToAddTransform;

/// Absorbs a `zero_ext`, `sign_ext`, or `concat(0..., x)` term operand into an
/// nary add.
#[derive(Debug)]
pub struct AbsorbExtendIntoExtNaryAddTermTransform;

/// Absorbs an explicit `neg(x)` term operand into an nary add.
#[derive(Debug)]
pub struct AbsorbNegIntoExtNaryAddTermTransform;

/// Absorbs a term operand that is itself an `add(x, y)` into an nary add.
#[derive(Debug)]
pub struct AbsorbAddOperandIntoExtNaryAddTransform;

/// Absorbs a term operand that is itself a `sub(x, y)` into an nary add.
#[derive(Debug)]
pub struct AbsorbSubOperandIntoExtNaryAddTransform;

/// Extracts an nary add's operand `negated` bit into an explicit `neg(x)`.
#[derive(Debug)]
pub struct ExtractNegateFromExtNaryAddTermTransform;

/// Extracts two adjacent operands from an nary add into a separate add
/// operation.
#[derive(Debug)]
pub struct ExtractAddFromNaryAddTermsTransform;

/// Combines two nary add operations.
#[derive(Debug)]
pub struct CombineNaryAddsTransform;

fn next_text_id(f: &IrFn) -> usize {
    f.nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .expect("text_id overflow")
}

fn push_bits_node(f: &mut IrFn, w: usize, payload: NodePayload) -> NodeRef {
    let text_id = next_text_id(f);
    let new_index = f.nodes.len();
    f.nodes.push(Node {
        text_id,
        name: None,
        ty: Type::Bits(w),
        payload,
        pos: None,
    });
    NodeRef { index: new_index }
}

fn make_zero_bits_literal(f: &mut IrFn, w: usize) -> NodeRef {
    push_bits_node(
        f,
        w,
        NodePayload::Literal(IrValue::make_ubits(w, 0).expect("zero bits literal")),
    )
}

fn is_bits_w(f: &IrFn, nr: NodeRef, w: usize) -> bool {
    matches!(&f.get_node(nr).ty, Type::Bits(ow) if *ow == w)
}

fn bits_width(ty: &Type) -> Option<usize> {
    match ty {
        Type::Bits(w) => Some(*w),
        _ => None,
    }
}

fn ext_nary_add_result_width(f: &IrFn, nr: NodeRef) -> Option<usize> {
    let Type::Bits(w) = f.get_node(nr).ty else {
        return None;
    };
    matches!(f.get_node(nr).payload, NodePayload::ExtNaryAdd { .. }).then_some(w)
}

fn term_operand_width(f: &IrFn, term: &ExtNaryAddTerm) -> Result<usize, String> {
    bits_width(&f.get_node(term.operand).ty)
        .ok_or_else(|| "ext_nary_add term operand must be bits-typed".to_string())
}

fn ext_nary_add_terms_arch_and_width(
    f: &IrFn,
    nr: NodeRef,
    transform_name: &str,
) -> Result<(Vec<ExtNaryAddTerm>, Option<ExtNaryAddArchitecture>, usize), String> {
    let out_w = ext_nary_add_result_width(f, nr)
        .ok_or_else(|| format!("{transform_name}: expected bits-typed ext_nary_add"))?;
    match &f.get_node(nr).payload {
        NodePayload::ExtNaryAdd { terms, arch } => Ok((terms.clone(), *arch, out_w)),
        _ => Err(format!("{transform_name}: expected ext_nary_add")),
    }
}

/// Reuses `RewireOperand` as a compact per-term location carrier for
/// ext_nary_add-specific rewrites. `operand_slot` is the term index, while
/// `new_operand` is only a stable debug hint and is otherwise ignored.
fn make_term_location(
    node: NodeRef,
    term_index: usize,
    hint_operand: NodeRef,
) -> TransformLocation {
    TransformLocation::RewireOperand {
        node,
        operand_slot: term_index,
        new_operand: hint_operand,
    }
}

fn expect_term_location(
    loc: &TransformLocation,
    transform_name: &str,
) -> Result<(NodeRef, usize), String> {
    match loc {
        TransformLocation::RewireOperand {
            node, operand_slot, ..
        } => Ok((*node, *operand_slot)),
        TransformLocation::Node(_) => Err(format!(
            "{transform_name}: expected term-indexed TransformLocation::RewireOperand, got Node"
        )),
    }
}

fn binary_ext_nary_add_operands_matching_result(
    f: &IrFn,
    nr: NodeRef,
) -> Option<(NodeRef, NodeRef, usize)> {
    let Type::Bits(w) = f.get_node(nr).ty else {
        return None;
    };
    let NodePayload::ExtNaryAdd { terms, .. } = &f.get_node(nr).payload else {
        return None;
    };
    if terms.len() != 2 || !terms.iter().all(|term| !term.signed && !term.negated) {
        return None;
    }
    let lhs = terms[0].operand;
    let rhs = terms[1].operand;
    if !is_bits_w(f, lhs, w) || !is_bits_w(f, rhs, w) {
        return None;
    }
    Some((lhs, rhs, w))
}

fn materialize_term_to_width(
    f: &mut IrFn,
    out_w: usize,
    term: &ExtNaryAddTerm,
) -> Result<NodeRef, String> {
    if out_w == 0 {
        return Ok(make_zero_bits_literal(f, 0));
    }

    let operand_w = term_operand_width(f, term)?;
    let mut value = if operand_w == out_w {
        term.operand
    } else if operand_w < out_w {
        if term.signed && operand_w == 0 {
            make_zero_bits_literal(f, out_w)
        } else {
            push_bits_node(
                f,
                out_w,
                if term.signed {
                    NodePayload::SignExt {
                        arg: term.operand,
                        new_bit_count: out_w,
                    }
                } else {
                    NodePayload::ZeroExt {
                        arg: term.operand,
                        new_bit_count: out_w,
                    }
                },
            )
        }
    } else {
        push_bits_node(
            f,
            out_w,
            NodePayload::BitSlice {
                arg: term.operand,
                start: 0,
                width: out_w,
            },
        )
    };

    if term.negated {
        value = push_bits_node(f, out_w, NodePayload::Unop(Unop::Neg, value));
    }
    Ok(value)
}

fn term_payload_matches_resize(f: &IrFn, term: &ExtNaryAddTerm) -> Option<(bool, NodeRef)> {
    match &f.get_node(term.operand).payload {
        NodePayload::SignExt { arg, .. } => Some((true, *arg)),
        NodePayload::ZeroExt { arg, .. } => Some((false, *arg)),
        NodePayload::Nary(NaryOp::Concat, ops) if ops.len() == 2 => {
            let hi = ops[0];
            let lo = ops[1];
            let NodePayload::Literal(v) = &f.get_node(hi).payload else {
                return None;
            };
            let bits = v.to_bits().ok()?;
            if !bits.is_zero() {
                return None;
            }
            Some((false, lo))
        }
        _ => None,
    }
}

fn term_payload_matches_neg(f: &IrFn, term: &ExtNaryAddTerm) -> Option<NodeRef> {
    match f.get_node(term.operand).payload {
        NodePayload::Unop(Unop::Neg, arg) => Some(arg),
        _ => None,
    }
}

fn term_payload_matches_add(f: &IrFn, term: &ExtNaryAddTerm) -> Option<(NodeRef, NodeRef)> {
    match f.get_node(term.operand).payload {
        NodePayload::Binop(Binop::Add, lhs, rhs) => Some((lhs, rhs)),
        _ => None,
    }
}

fn term_payload_matches_sub(f: &IrFn, term: &ExtNaryAddTerm) -> Option<(NodeRef, NodeRef)> {
    match f.get_node(term.operand).payload {
        NodePayload::Binop(Binop::Sub, lhs, rhs) => Some((lhs, rhs)),
        _ => None,
    }
}

fn term_payload_matches_nested_ext_nary_add(
    f: &IrFn,
    term: &ExtNaryAddTerm,
) -> Option<Vec<ExtNaryAddTerm>> {
    match &f.get_node(term.operand).payload {
        NodePayload::ExtNaryAdd { terms, .. } => Some(terms.clone()),
        _ => None,
    }
}

/// Returns whether absorbing a resize op into a term keeps the same value for
/// all inputs without needing an oracle.
fn absorb_extend_candidate_is_always_equivalent(
    f: &IrFn,
    outer_term: &ExtNaryAddTerm,
    inner_signed: bool,
    inner_arg: NodeRef,
    out_w: usize,
) -> bool {
    let Some(inner_resize_w) = bits_width(&f.get_node(outer_term.operand).ty) else {
        return false;
    };
    if inner_resize_w >= out_w {
        return true;
    }

    let Some(inner_arg_w) = bits_width(&f.get_node(inner_arg).ty) else {
        return false;
    };
    inner_arg_w == 0
        || (inner_arg_w <= inner_resize_w
            && (outer_term.signed == inner_signed
                || (!inner_signed && inner_resize_w > inner_arg_w)))
}

fn absorb_neg_candidate_is_always_equivalent(
    f: &IrFn,
    term: &ExtNaryAddTerm,
    out_w: usize,
) -> bool {
    bits_width(&f.get_node(term.operand).ty)
        .is_some_and(|operand_w| operand_w == 0 || operand_w >= out_w)
}

fn absorb_binop_candidate_is_always_equivalent(
    f: &IrFn,
    term: &ExtNaryAddTerm,
    out_w: usize,
) -> bool {
    bits_width(&f.get_node(term.operand).ty)
        .is_some_and(|operand_w| operand_w == 0 || operand_w >= out_w)
}

fn combine_nary_add_candidate_is_always_equivalent(
    f: &IrFn,
    term: &ExtNaryAddTerm,
    out_w: usize,
) -> bool {
    ext_nary_add_result_width(f, term.operand).is_some_and(|inner_w| inner_w >= out_w)
}

fn change_ext_nary_add_candidates(
    f: &IrFn,
    target_arch: ExtNaryAddArchitecture,
    always_equivalent: bool,
) -> Vec<TransformCandidate> {
    f.node_refs()
        .into_iter()
        .filter(|nr| {
            matches!(
                f.get_node(*nr).payload,
                NodePayload::ExtNaryAdd {
                    arch: Some(current_arch),
                    ..
                } if current_arch != target_arch
            )
        })
        .map(|nr| TransformCandidate {
            location: TransformLocation::Node(nr),
            always_equivalent,
        })
        .collect()
}

fn apply_change_ext_nary_add_architecture(
    f: &mut IrFn,
    loc: &TransformLocation,
    target_arch: ExtNaryAddArchitecture,
    transform_name: &str,
) -> Result<(), String> {
    let nr = match loc {
        TransformLocation::Node(nr) => *nr,
        TransformLocation::RewireOperand { .. } => {
            return Err(format!(
                "{transform_name}: expected TransformLocation::Node, got RewireOperand"
            ));
        }
    };
    let node = f.get_node_mut(nr);
    let NodePayload::ExtNaryAdd { arch, .. } = &mut node.payload else {
        return Err(format!("{transform_name}: expected ext_nary_add"));
    };
    let current_arch =
        arch.ok_or_else(|| format!("{transform_name}: ext_nary_add must have an explicit arch"))?;
    if current_arch == target_arch {
        return Err(format!(
            "{transform_name}: target arch must differ from current arch"
        ));
    }
    *arch = Some(target_arch);
    Ok(())
}

impl PirTransform for AddToExtNaryAddTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AddToExtNaryAdd
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        f.node_refs()
            .into_iter()
            .filter(|nr| {
                let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(*nr).payload else {
                    return false;
                };
                let Type::Bits(w) = f.get_node(*nr).ty else {
                    return false;
                };
                is_bits_w(f, lhs, w) && is_bits_w(f, rhs, w)
            })
            .map(|nr| TransformCandidate {
                location: TransformLocation::Node(nr),
                always_equivalent,
            })
            .collect()
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let nr = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "AddToExtNaryAddTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };
        let Type::Bits(w) = f.get_node(nr).ty else {
            return Err("AddToExtNaryAddTransform: add result must be bits-typed".to_string());
        };
        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(nr).payload else {
            return Err("AddToExtNaryAddTransform: expected add".to_string());
        };
        if !is_bits_w(f, lhs, w) || !is_bits_w(f, rhs, w) {
            return Err(
                "AddToExtNaryAddTransform: add operands must match the result bit width"
                    .to_string(),
            );
        }
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd {
            terms: vec![
                ExtNaryAddTerm {
                    operand: lhs,
                    signed: false,
                    negated: false,
                },
                ExtNaryAddTerm {
                    operand: rhs,
                    signed: false,
                    negated: false,
                },
            ],
            arch: Some(ExtNaryAddArchitecture::BrentKung),
        };
        Ok(())
    }
}

impl PirTransform for SubToExtNaryAddTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SubToExtNaryAdd
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        f.node_refs()
            .into_iter()
            .filter(|nr| {
                let NodePayload::Binop(Binop::Sub, lhs, rhs) = f.get_node(*nr).payload else {
                    return false;
                };
                let Type::Bits(w) = f.get_node(*nr).ty else {
                    return false;
                };
                is_bits_w(f, lhs, w) && is_bits_w(f, rhs, w)
            })
            .map(|nr| TransformCandidate {
                location: TransformLocation::Node(nr),
                always_equivalent,
            })
            .collect()
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let nr = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "SubToExtNaryAddTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };
        let Type::Bits(w) = f.get_node(nr).ty else {
            return Err("SubToExtNaryAddTransform: sub result must be bits-typed".to_string());
        };
        let NodePayload::Binop(Binop::Sub, lhs, rhs) = f.get_node(nr).payload else {
            return Err("SubToExtNaryAddTransform: expected sub".to_string());
        };
        if !is_bits_w(f, lhs, w) || !is_bits_w(f, rhs, w) {
            return Err(
                "SubToExtNaryAddTransform: sub operands must match the result bit width"
                    .to_string(),
            );
        }
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd {
            terms: vec![
                ExtNaryAddTerm {
                    operand: lhs,
                    signed: false,
                    negated: false,
                },
                ExtNaryAddTerm {
                    operand: rhs,
                    signed: false,
                    negated: true,
                },
            ],
            arch: Some(ExtNaryAddArchitecture::BrentKung),
        };
        Ok(())
    }
}

impl PirTransform for ChangeExtNaryAddToRippleCarryTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ChangeExtNaryAddToRippleCarry
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        change_ext_nary_add_candidates(f, ExtNaryAddArchitecture::RippleCarry, always_equivalent)
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        apply_change_ext_nary_add_architecture(
            f,
            loc,
            ExtNaryAddArchitecture::RippleCarry,
            "ChangeExtNaryAddToRippleCarryTransform",
        )
    }
}

impl PirTransform for ChangeExtNaryAddToBrentKungTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ChangeExtNaryAddToBrentKung
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        change_ext_nary_add_candidates(f, ExtNaryAddArchitecture::BrentKung, always_equivalent)
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        apply_change_ext_nary_add_architecture(
            f,
            loc,
            ExtNaryAddArchitecture::BrentKung,
            "ChangeExtNaryAddToBrentKungTransform",
        )
    }
}

impl PirTransform for ChangeExtNaryAddToKoggeStoneTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ChangeExtNaryAddToKoggeStone
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        change_ext_nary_add_candidates(f, ExtNaryAddArchitecture::KoggeStone, always_equivalent)
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        apply_change_ext_nary_add_architecture(
            f,
            loc,
            ExtNaryAddArchitecture::KoggeStone,
            "ChangeExtNaryAddToKoggeStoneTransform",
        )
    }
}

impl PirTransform for BinaryExtNaryAddToAddTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BinaryExtNaryAddToAdd
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        f.node_refs()
            .into_iter()
            .filter(|nr| binary_ext_nary_add_operands_matching_result(f, *nr).is_some())
            .map(|nr| TransformCandidate {
                location: TransformLocation::Node(nr),
                always_equivalent,
            })
            .collect()
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let nr = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "BinaryExtNaryAddToAddTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };
        let (lhs, rhs, _w) =
            binary_ext_nary_add_operands_matching_result(f, nr).ok_or_else(|| {
                "BinaryExtNaryAddToAddTransform: expected a width-preserving binary ext_nary_add"
                    .to_string()
            })?;
        f.get_node_mut(nr).payload = NodePayload::Binop(Binop::Add, lhs, rhs);
        Ok(())
    }
}

impl PirTransform for AbsorbExtendIntoExtNaryAddTermTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AbsorbExtendIntoExtNaryAddTerm
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            let NodePayload::ExtNaryAdd { terms, .. } = &f.get_node(nr).payload else {
                continue;
            };
            let Some(out_w) = ext_nary_add_result_width(f, nr) else {
                continue;
            };
            for (term_index, term) in terms.iter().enumerate() {
                if let Some((inner_signed, inner_arg)) = term_payload_matches_resize(f, term) {
                    out.push(TransformCandidate {
                        location: make_term_location(nr, term_index, term.operand),
                        always_equivalent: absorb_extend_candidate_is_always_equivalent(
                            f,
                            term,
                            inner_signed,
                            inner_arg,
                            out_w,
                        ),
                    });
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (nr, term_index) =
            expect_term_location(loc, "AbsorbExtendIntoExtNaryAddTermTransform")?;
        let (mut terms, arch, _out_w) =
            ext_nary_add_terms_arch_and_width(f, nr, "AbsorbExtendIntoExtNaryAddTermTransform")?;
        let term = terms.get_mut(term_index).ok_or_else(|| {
            "AbsorbExtendIntoExtNaryAddTermTransform: invalid term index".to_string()
        })?;
        let (signed, arg) = term_payload_matches_resize(f, term).ok_or_else(|| {
            "AbsorbExtendIntoExtNaryAddTermTransform: expected term operand to be zero_ext/sign_ext/concat-zero".to_string()
        })?;
        term.operand = arg;
        term.signed = signed;
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd { terms, arch };
        Ok(())
    }
}

impl PirTransform for AbsorbNegIntoExtNaryAddTermTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AbsorbNegIntoExtNaryAddTerm
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            let NodePayload::ExtNaryAdd { terms, .. } = &f.get_node(nr).payload else {
                continue;
            };
            let Some(out_w) = ext_nary_add_result_width(f, nr) else {
                continue;
            };
            for (term_index, term) in terms.iter().enumerate() {
                if term_payload_matches_neg(f, term).is_some() {
                    out.push(TransformCandidate {
                        location: make_term_location(nr, term_index, term.operand),
                        always_equivalent: absorb_neg_candidate_is_always_equivalent(
                            f, term, out_w,
                        ),
                    });
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (nr, term_index) = expect_term_location(loc, "AbsorbNegIntoExtNaryAddTermTransform")?;
        let (mut terms, arch, _out_w) =
            ext_nary_add_terms_arch_and_width(f, nr, "AbsorbNegIntoExtNaryAddTermTransform")?;
        let term = terms.get_mut(term_index).ok_or_else(|| {
            "AbsorbNegIntoExtNaryAddTermTransform: invalid term index".to_string()
        })?;
        let arg = term_payload_matches_neg(f, term).ok_or_else(|| {
            "AbsorbNegIntoExtNaryAddTermTransform: expected term operand to be neg(x)".to_string()
        })?;
        term.operand = arg;
        term.negated = !term.negated;
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd { terms, arch };
        Ok(())
    }
}

impl PirTransform for AbsorbAddOperandIntoExtNaryAddTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AbsorbAddOperandIntoExtNaryAdd
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            let NodePayload::ExtNaryAdd { terms, .. } = &f.get_node(nr).payload else {
                continue;
            };
            let Some(out_w) = ext_nary_add_result_width(f, nr) else {
                continue;
            };
            for (term_index, term) in terms.iter().enumerate() {
                if term_payload_matches_add(f, term).is_some() {
                    out.push(TransformCandidate {
                        location: make_term_location(nr, term_index, term.operand),
                        always_equivalent: absorb_binop_candidate_is_always_equivalent(
                            f, term, out_w,
                        ),
                    });
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (nr, term_index) =
            expect_term_location(loc, "AbsorbAddOperandIntoExtNaryAddTransform")?;
        let (mut terms, arch, _out_w) =
            ext_nary_add_terms_arch_and_width(f, nr, "AbsorbAddOperandIntoExtNaryAddTransform")?;
        let original_term = terms.get(term_index).cloned().ok_or_else(|| {
            "AbsorbAddOperandIntoExtNaryAddTransform: invalid term index".to_string()
        })?;
        let (lhs, rhs) = term_payload_matches_add(f, &original_term).ok_or_else(|| {
            "AbsorbAddOperandIntoExtNaryAddTransform: expected term operand to be add(x, y)"
                .to_string()
        })?;
        terms.splice(
            term_index..term_index + 1,
            [
                ExtNaryAddTerm {
                    operand: lhs,
                    signed: original_term.signed,
                    negated: original_term.negated,
                },
                ExtNaryAddTerm {
                    operand: rhs,
                    signed: original_term.signed,
                    negated: original_term.negated,
                },
            ],
        );
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd { terms, arch };
        Ok(())
    }
}

impl PirTransform for AbsorbSubOperandIntoExtNaryAddTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AbsorbSubOperandIntoExtNaryAdd
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            let NodePayload::ExtNaryAdd { terms, .. } = &f.get_node(nr).payload else {
                continue;
            };
            let Some(out_w) = ext_nary_add_result_width(f, nr) else {
                continue;
            };
            for (term_index, term) in terms.iter().enumerate() {
                if term_payload_matches_sub(f, term).is_some() {
                    out.push(TransformCandidate {
                        location: make_term_location(nr, term_index, term.operand),
                        always_equivalent: absorb_binop_candidate_is_always_equivalent(
                            f, term, out_w,
                        ),
                    });
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (nr, term_index) =
            expect_term_location(loc, "AbsorbSubOperandIntoExtNaryAddTransform")?;
        let (mut terms, arch, _out_w) =
            ext_nary_add_terms_arch_and_width(f, nr, "AbsorbSubOperandIntoExtNaryAddTransform")?;
        let original_term = terms.get(term_index).cloned().ok_or_else(|| {
            "AbsorbSubOperandIntoExtNaryAddTransform: invalid term index".to_string()
        })?;
        let (lhs, rhs) = term_payload_matches_sub(f, &original_term).ok_or_else(|| {
            "AbsorbSubOperandIntoExtNaryAddTransform: expected term operand to be sub(x, y)"
                .to_string()
        })?;
        terms.splice(
            term_index..term_index + 1,
            [
                ExtNaryAddTerm {
                    operand: lhs,
                    signed: original_term.signed,
                    negated: original_term.negated,
                },
                ExtNaryAddTerm {
                    operand: rhs,
                    signed: original_term.signed,
                    negated: !original_term.negated,
                },
            ],
        );
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd { terms, arch };
        Ok(())
    }
}

impl PirTransform for ExtractNegateFromExtNaryAddTermTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ExtractNegateFromExtNaryAddTerm
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            let NodePayload::ExtNaryAdd { terms, .. } = &f.get_node(nr).payload else {
                continue;
            };
            let Some(out_w) = ext_nary_add_result_width(f, nr) else {
                continue;
            };
            for (term_index, term) in terms.iter().enumerate() {
                if term.negated {
                    out.push(TransformCandidate {
                        location: make_term_location(nr, term_index, term.operand),
                        always_equivalent: absorb_neg_candidate_is_always_equivalent(
                            f, term, out_w,
                        ),
                    });
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (nr, term_index) =
            expect_term_location(loc, "ExtractNegateFromExtNaryAddTermTransform")?;
        let (mut terms, arch, _out_w) =
            ext_nary_add_terms_arch_and_width(f, nr, "ExtractNegateFromExtNaryAddTermTransform")?;
        let term = terms.get_mut(term_index).ok_or_else(|| {
            "ExtractNegateFromExtNaryAddTermTransform: invalid term index".to_string()
        })?;
        if !term.negated {
            return Err(
                "ExtractNegateFromExtNaryAddTermTransform: expected negated term".to_string(),
            );
        }
        let operand_w = term_operand_width(f, term)?;
        let neg_operand = push_bits_node(f, operand_w, NodePayload::Unop(Unop::Neg, term.operand));
        term.operand = neg_operand;
        term.negated = false;
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd { terms, arch };
        Ok(())
    }
}

impl PirTransform for ExtractAddFromNaryAddTermsTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ExtractAddFromNaryAddTerms
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let always_equivalent = true;
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            let NodePayload::ExtNaryAdd { terms, .. } = &f.get_node(nr).payload else {
                continue;
            };
            for left_index in 0..terms.len().saturating_sub(1) {
                out.push(TransformCandidate {
                    location: make_term_location(nr, left_index, terms[left_index].operand),
                    always_equivalent,
                });
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (nr, left_index) = expect_term_location(loc, "ExtractAddFromNaryAddTermsTransform")?;
        let (mut terms, arch, out_w) =
            ext_nary_add_terms_arch_and_width(f, nr, "ExtractAddFromNaryAddTermsTransform")?;
        if left_index + 1 >= terms.len() {
            return Err(
                "ExtractAddFromNaryAddTermsTransform: invalid adjacent term pair".to_string(),
            );
        }
        let lhs = materialize_term_to_width(f, out_w, &terms[left_index])?;
        let rhs = materialize_term_to_width(f, out_w, &terms[left_index + 1])?;
        let merged_add = push_bits_node(f, out_w, NodePayload::Binop(Binop::Add, lhs, rhs));
        terms.splice(
            left_index..left_index + 2,
            [ExtNaryAddTerm {
                operand: merged_add,
                signed: false,
                negated: false,
            }],
        );
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd { terms, arch };
        Ok(())
    }
}

impl PirTransform for CombineNaryAddsTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CombineNaryAdds
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let mut out = Vec::<TransformCandidate>::new();
        for nr in f.node_refs() {
            let NodePayload::ExtNaryAdd { terms, .. } = &f.get_node(nr).payload else {
                continue;
            };
            let Some(out_w) = ext_nary_add_result_width(f, nr) else {
                continue;
            };
            for (term_index, term) in terms.iter().enumerate() {
                if term_payload_matches_nested_ext_nary_add(f, term).is_some() {
                    out.push(TransformCandidate {
                        location: make_term_location(nr, term_index, term.operand),
                        always_equivalent: combine_nary_add_candidate_is_always_equivalent(
                            f, term, out_w,
                        ),
                    });
                }
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (nr, term_index) = expect_term_location(loc, "CombineNaryAddsTransform")?;
        let (mut outer_terms, arch, _out_w) =
            ext_nary_add_terms_arch_and_width(f, nr, "CombineNaryAddsTransform")?;
        let outer_term = outer_terms
            .get(term_index)
            .cloned()
            .ok_or_else(|| "CombineNaryAddsTransform: invalid term index".to_string())?;
        let nested_terms =
            term_payload_matches_nested_ext_nary_add(f, &outer_term).ok_or_else(|| {
                "CombineNaryAddsTransform: expected nested ext_nary_add operand".to_string()
            })?;
        let replacement_terms = nested_terms.into_iter().map(|term| ExtNaryAddTerm {
            operand: term.operand,
            signed: term.signed,
            negated: term.negated ^ outer_term.negated,
        });
        outer_terms.splice(term_index..term_index + 1, replacement_terms);
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd {
            terms: outer_terms,
            arch,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth_pir::ir_parser;

    fn find_node_ref<F>(f: &IrFn, pred: F) -> NodeRef
    where
        F: Fn(&NodePayload) -> bool,
    {
        f.node_refs()
            .into_iter()
            .find(|nr| pred(&f.get_node(*nr).payload))
            .expect("expected matching node")
    }

    fn find_term_candidate(
        candidates: &[TransformCandidate],
        node: NodeRef,
        term_index: usize,
    ) -> TransformCandidate {
        candidates
            .iter()
            .find(|candidate| {
                matches!(
                    &candidate.location,
                    TransformLocation::RewireOperand {
                        node: loc_node,
                        operand_slot,
                        ..
                    } if *loc_node == node && *operand_slot == term_index
                )
            })
            .cloned()
            .expect("expected matching term candidate")
    }

    fn expect_success_value(result: FnEvalResult) -> IrValue {
        match result {
            FnEvalResult::Success(success) => success.value,
            FnEvalResult::Failure(failure) => {
                panic!("expected eval success, got failure: {:?}", failure)
            }
        }
    }

    fn param_ref(f: &IrFn, param_index: usize) -> NodeRef {
        let param_id = f.params[param_index].id;
        find_node_ref(
            f,
            |payload| matches!(payload, NodePayload::GetParam(index) if *index == param_id),
        )
    }

    #[test]
    fn add_to_ext_nary_add_rewrites_binary_add() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(a, b, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let add_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::Binop(Binop::Add, _, _))
        });

        let t = AddToExtNaryAddTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        match &f.get_node(add_ref).payload {
            NodePayload::ExtNaryAdd { terms, arch } => {
                assert_eq!(terms.len(), 2);
                assert!(terms.iter().all(|term| !term.signed && !term.negated));
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::BrentKung));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn sub_to_ext_nary_add_rewrites_binary_sub() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret sub.3: bits[8] = sub(a, b, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let a_ref = param_ref(&f, 0);
        let b_ref = param_ref(&f, 1);
        let sub_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::Binop(Binop::Sub, _, _))
        });
        let mut t = SubToExtNaryAddTransform;

        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(
            candidates[0].location,
            TransformLocation::Node(nr) if nr == sub_ref
        ));
        assert!(candidates[0].always_equivalent);
        t.apply(&mut f, &TransformLocation::Node(sub_ref))
            .expect("apply");

        match &f.get_node(sub_ref).payload {
            NodePayload::ExtNaryAdd { terms, arch } => {
                assert_eq!(terms.len(), 2);
                assert_eq!(terms[0].operand, a_ref);
                assert_eq!(terms[1].operand, b_ref);
                assert!(!terms[0].signed && !terms[0].negated);
                assert!(!terms[1].signed && terms[1].negated);
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::BrentKung));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn change_ext_nary_add_to_ripple_carry_rewrites_other_architecture() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, signed=[false, false], negated=[false, false], arch=brent_kung, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });

        let mut t = ChangeExtNaryAddToRippleCarryTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(
            candidates[0].location,
            TransformLocation::Node(nr) if nr == nary_ref
        ));
        t.apply(&mut f, &TransformLocation::Node(nary_ref))
            .expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { arch, .. } => {
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::RippleCarry));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn change_ext_nary_add_to_brent_kung_rewrites_other_architecture() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, signed=[false, false], negated=[false, false], arch=kogge_stone, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });

        let mut t = ChangeExtNaryAddToBrentKungTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(
            candidates[0].location,
            TransformLocation::Node(nr) if nr == nary_ref
        ));
        t.apply(&mut f, &TransformLocation::Node(nary_ref))
            .expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { arch, .. } => {
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::BrentKung));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn change_ext_nary_add_to_kogge_stone_rewrites_other_architecture() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, signed=[false, false], negated=[false, false], arch=ripple_carry, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });

        let mut t = ChangeExtNaryAddToKoggeStoneTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(
            candidates[0].location,
            TransformLocation::Node(nr) if nr == nary_ref
        ));
        t.apply(&mut f, &TransformLocation::Node(nary_ref))
            .expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { arch, .. } => {
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::KoggeStone));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn change_ext_nary_add_to_target_skips_same_architecture() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, signed=[false, false], negated=[false, false], arch=brent_kung, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut to_ripple = ChangeExtNaryAddToRippleCarryTransform;
        let mut to_brent = ChangeExtNaryAddToBrentKungTransform;
        let mut to_kogge = ChangeExtNaryAddToKoggeStoneTransform;

        assert_eq!(to_ripple.find_candidates(&f).len(), 1);
        assert!(to_brent.find_candidates(&f).is_empty());
        assert_eq!(to_kogge.find_candidates(&f).len(), 1);
    }

    #[test]
    fn binary_ext_nary_add_to_add_rewrites_width_preserving_case() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, signed=[false, false], negated=[false, false], arch=ripple_carry, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });

        let t = BinaryExtNaryAddToAddTransform;
        t.apply(&mut f, &TransformLocation::Node(nary_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(nary_ref).payload,
            NodePayload::Binop(Binop::Add, _, _)
        ));
    }

    #[test]
    fn binary_ext_nary_add_to_add_skips_resizing_case() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[6] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, signed=[false, false], negated=[false, false], arch=ripple_carry, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let mut t = BinaryExtNaryAddToAddTransform;

        assert!(
            t.find_candidates(&f).is_empty(),
            "expected resizing ext_nary_add to be excluded from add fallback"
        );
    }

    #[test]
    fn absorb_extend_into_ext_nary_add_term_rewrites_term_attrs() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[8] id=2) -> bits[8] {
  sign_ext.3: bits[8] = sign_ext(a, new_bit_count=8, id=3)
  ret ext_nary_add.4: bits[8] = ext_nary_add(sign_ext.3, b, signed=[false, false], negated=[false, false], arch=brent_kung, id=4)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let a_ref = param_ref(&f, 0);
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbExtendIntoExtNaryAddTermTransform;
        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);
        assert!(candidate.always_equivalent);

        t.apply(&mut f, &candidate.location).expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { terms, .. } => {
                assert_eq!(terms[0].operand, a_ref);
                assert!(terms[0].signed);
                assert!(!terms[0].negated);
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn absorb_extend_into_ext_nary_add_term_recognizes_concat_zero_extend() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[8] id=2) -> bits[8] {
  literal.3: bits[4] = literal(value=0, id=3)
  concat.4: bits[8] = concat(literal.3, a, id=4)
  ret ext_nary_add.5: bits[8] = ext_nary_add(concat.4, b, signed=[false, false], negated=[false, false], arch=brent_kung, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let a_ref = param_ref(&f, 0);
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbExtendIntoExtNaryAddTermTransform;
        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);
        assert!(candidate.always_equivalent);

        t.apply(&mut f, &candidate.location).expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { terms, .. } => {
                assert_eq!(terms[0].operand, a_ref);
                assert!(!terms[0].signed);
                assert!(!terms[0].negated);
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn absorb_extend_into_ext_nary_add_term_marks_signed_mismatch_growth_unsafe() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[8] id=2) -> bits[8] {
  sign_ext.3: bits[6] = sign_ext(a, new_bit_count=6, id=3)
  ret ext_nary_add.4: bits[8] = ext_nary_add(sign_ext.3, b, signed=[false, false], negated=[false, false], arch=brent_kung, id=4)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbExtendIntoExtNaryAddTermTransform;

        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);

        assert!(!candidate.always_equivalent);
    }

    #[test]
    fn absorb_neg_into_ext_nary_add_term_toggles_negated() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  neg.3: bits[8] = neg(a, id=3)
  ret ext_nary_add.4: bits[8] = ext_nary_add(neg.3, b, signed=[false, false], negated=[false, false], arch=brent_kung, id=4)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let a_ref = param_ref(&f, 0);
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbNegIntoExtNaryAddTermTransform;
        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);
        assert!(candidate.always_equivalent);

        t.apply(&mut f, &candidate.location).expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { terms, .. } => {
                assert_eq!(terms[0].operand, a_ref);
                assert!(terms[0].negated);
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn absorb_neg_into_ext_nary_add_term_marks_narrow_neg_unsafe() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[8] id=2) -> bits[8] {
  neg.3: bits[4] = neg(a, id=3)
  ret ext_nary_add.4: bits[8] = ext_nary_add(neg.3, b, signed=[true, false], negated=[false, false], arch=brent_kung, id=4)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbNegIntoExtNaryAddTermTransform;

        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);

        assert!(!candidate.always_equivalent);
    }

    #[test]
    fn absorb_add_operand_into_ext_nary_add_preserves_term_attrs() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  add.4: bits[8] = add(a, b, id=4)
  ret ext_nary_add.5: bits[8] = ext_nary_add(add.4, c, signed=[true, false], negated=[true, false], arch=brent_kung, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let a_ref = param_ref(&f, 0);
        let b_ref = param_ref(&f, 1);
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbAddOperandIntoExtNaryAddTransform;
        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);
        assert!(candidate.always_equivalent);

        t.apply(&mut f, &candidate.location).expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { terms, .. } => {
                assert_eq!(terms.len(), 3);
                assert_eq!(terms[0].operand, a_ref);
                assert_eq!(terms[1].operand, b_ref);
                assert!(terms[0].signed && terms[1].signed);
                assert!(terms[0].negated && terms[1].negated);
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn absorb_add_operand_into_ext_nary_add_marks_narrow_add_unsafe() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[4] id=2, c: bits[8] id=3) -> bits[8] {
  add.4: bits[4] = add(a, b, id=4)
  ret ext_nary_add.5: bits[8] = ext_nary_add(add.4, c, signed=[false, false], negated=[false, false], arch=brent_kung, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbAddOperandIntoExtNaryAddTransform;

        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);

        assert!(!candidate.always_equivalent);
    }

    #[test]
    fn absorb_sub_operand_into_ext_nary_add_flips_rhs_negated_bit() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  sub.4: bits[8] = sub(a, b, id=4)
  ret ext_nary_add.5: bits[8] = ext_nary_add(sub.4, c, signed=[true, false], negated=[false, false], arch=brent_kung, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let a_ref = param_ref(&f, 0);
        let b_ref = param_ref(&f, 1);
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbSubOperandIntoExtNaryAddTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        let candidate = find_term_candidate(&candidates, nary_ref, 0);
        assert!(candidate.always_equivalent);

        t.apply(&mut f, &candidate.location).expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { terms, .. } => {
                assert_eq!(terms.len(), 3);
                assert_eq!(terms[0].operand, a_ref);
                assert_eq!(terms[1].operand, b_ref);
                assert!(terms[0].signed && terms[1].signed);
                assert!(!terms[0].negated);
                assert!(terms[1].negated);
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn absorb_sub_operand_into_negated_ext_nary_add_term_unflips_rhs_negated_bit() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  sub.4: bits[8] = sub(a, b, id=4)
  ret ext_nary_add.5: bits[8] = ext_nary_add(sub.4, c, signed=[false, false], negated=[true, false], arch=brent_kung, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let a_ref = param_ref(&f, 0);
        let b_ref = param_ref(&f, 1);
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbSubOperandIntoExtNaryAddTransform;
        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);
        assert!(candidate.always_equivalent);

        t.apply(&mut f, &candidate.location).expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { terms, .. } => {
                assert_eq!(terms.len(), 3);
                assert_eq!(terms[0].operand, a_ref);
                assert_eq!(terms[1].operand, b_ref);
                assert!(terms[0].negated);
                assert!(!terms[1].negated);
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn absorb_sub_operand_into_ext_nary_add_marks_narrow_sub_unsafe() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[4] id=2, c: bits[8] id=3) -> bits[8] {
  sub.4: bits[4] = sub(a, b, id=4)
  ret ext_nary_add.5: bits[8] = ext_nary_add(sub.4, c, signed=[false, false], negated=[false, false], arch=brent_kung, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = AbsorbSubOperandIntoExtNaryAddTransform;

        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);

        assert!(!candidate.always_equivalent);
    }

    #[test]
    fn extract_negate_from_ext_nary_add_term_creates_neg_node() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, signed=[false, false], negated=[true, false], arch=brent_kung, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let a_ref = param_ref(&f, 0);
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = ExtractNegateFromExtNaryAddTermTransform;
        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);
        assert!(candidate.always_equivalent);

        t.apply(&mut f, &candidate.location).expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { terms, .. } => {
                assert!(!terms[0].negated);
                assert!(matches!(
                    f.get_node(terms[0].operand).payload,
                    NodePayload::Unop(Unop::Neg, arg) if arg == a_ref
                ));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn extract_negate_from_ext_nary_add_term_marks_narrow_term_unsafe() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, signed=[true, false], negated=[true, false], arch=brent_kung, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = ExtractNegateFromExtNaryAddTermTransform;

        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);

        assert!(!candidate.always_equivalent);
    }

    #[test]
    fn extract_add_from_nary_add_terms_materializes_term_semantics() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[4] id=2, c: bits[8] id=3) -> bits[8] {
  ret ext_nary_add.10: bits[8] = ext_nary_add(a, b, c, signed=[false, true, false], negated=[false, true, false], arch=brent_kung, id=10)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let original = f.clone();
        let c_ref = param_ref(&f, 2);
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });
        let mut t = ExtractAddFromNaryAddTermsTransform;
        let candidate = find_term_candidate(&t.find_candidates(&f), nary_ref, 0);
        assert!(candidate.always_equivalent);

        t.apply(&mut f, &candidate.location).expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { terms, arch } => {
                assert_eq!(terms.len(), 2);
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::BrentKung));
                assert!(!terms[0].signed && !terms[0].negated);
                assert!(matches!(
                    f.get_node(terms[0].operand).payload,
                    NodePayload::Binop(Binop::Add, _, _)
                ));
                assert_eq!(terms[1].operand, c_ref);
                assert!(!terms[1].signed && !terms[1].negated);
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }

        let args = [
            vec![
                IrValue::make_ubits(4, 0x0).unwrap(),
                IrValue::make_ubits(4, 0x0).unwrap(),
                IrValue::make_ubits(8, 0x00).unwrap(),
            ],
            vec![
                IrValue::make_ubits(4, 0xf).unwrap(),
                IrValue::make_ubits(4, 0xf).unwrap(),
                IrValue::make_ubits(8, 0x03).unwrap(),
            ],
            vec![
                IrValue::make_ubits(4, 0x7).unwrap(),
                IrValue::make_ubits(4, 0x8).unwrap(),
                IrValue::make_ubits(8, 0xa5).unwrap(),
            ],
        ];
        for sample in args.iter() {
            let got_original = expect_success_value(eval_fn(&original, sample));
            let got_rewritten = expect_success_value(eval_fn(&f, sample));
            assert_eq!(got_original, got_rewritten);
        }
    }

    #[test]
    fn combine_nary_adds_splices_inner_terms() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  ext_nary_add.4: bits[8] = ext_nary_add(a, b, signed=[false, true], negated=[false, false], arch=ripple_carry, id=4)
  ret ext_nary_add.5: bits[8] = ext_nary_add(ext_nary_add.4, c, signed=[false, false], negated=[true, false], arch=brent_kung, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let a_ref = param_ref(&f, 0);
        let b_ref = param_ref(&f, 1);
        let c_ref = param_ref(&f, 2);
        let outer_ref = find_node_ref(&f, |payload| {
            matches!(
                payload,
                NodePayload::ExtNaryAdd {
                    arch: Some(ExtNaryAddArchitecture::BrentKung),
                    ..
                }
            )
        });
        let mut t = CombineNaryAddsTransform;
        let candidate = find_term_candidate(&t.find_candidates(&f), outer_ref, 0);
        assert!(candidate.always_equivalent);

        t.apply(&mut f, &candidate.location).expect("apply");

        match &f.get_node(outer_ref).payload {
            NodePayload::ExtNaryAdd { terms, arch } => {
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::BrentKung));
                assert_eq!(terms.len(), 3);
                assert_eq!(terms[0].operand, a_ref);
                assert_eq!(terms[1].operand, b_ref);
                assert_eq!(terms[2].operand, c_ref);
                assert!(terms[0].negated);
                assert!(terms[1].signed);
                assert!(terms[1].negated);
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn combine_nary_adds_marks_narrow_inner_nary_add_unsafe() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[4] id=2, c: bits[8] id=3) -> bits[8] {
  ext_nary_add.4: bits[4] = ext_nary_add(a, b, signed=[false, false], negated=[false, false], arch=ripple_carry, id=4)
  ret ext_nary_add.5: bits[8] = ext_nary_add(ext_nary_add.4, c, signed=[false, false], negated=[false, false], arch=brent_kung, id=5)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let outer_ref = find_node_ref(&f, |payload| {
            matches!(
                payload,
                NodePayload::ExtNaryAdd {
                    arch: Some(ExtNaryAddArchitecture::BrentKung),
                    ..
                }
            )
        });
        let mut t = CombineNaryAddsTransform;

        let candidate = find_term_candidate(&t.find_candidates(&f), outer_ref, 0);

        assert!(!candidate.always_equivalent);
    }
}
