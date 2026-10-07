// SPDX-License-Identifier: Apache-2.0

//! Replaces decoded priority results and scalar predicates with one-hot wiring.
//!
//! Work out where each one-hot bit would land in the output, including the
//! extra bit that represents an all-zero input. Use each operation's original
//! bit width so wrapping behaves the same. Then wire the bits directly to their
//! output positions; out-of-range positions contribute zero.
//!
//! For example, let `h = one_hot(x)` for `x: bits[3]`, so `h` has four bits.
//! With a two-bit encoded index, two-bit addition, and a four-bit output:
//!
//! ```text
//! Before: y = decode(encode(h) + u2:1)
//! After:  y = h[0:3] ++ h[3:4]
//!
//! h[0] ----> y[1]
//! h[1] ----> y[2]
//! h[2] ----> y[3]
//! h[3] ----> y[0]   (all-zero input; 3 + 1 wraps to 0)
//! ```
//!
//! An optional index mask `index & sign_ext(p)` selects the wired result when
//! `p` is true and the value 1 (only bit zero set) otherwise. The pass is
//! independent of the source of the one-hot value and the meaning of `p`.
//!
//! Candidates contain only basis IR. The caller supplies a whole-function cost
//! comparison so external sharing and loads remain part of profitability. Both
//! area and delay must not worsen and at least one must improve.
//!
//! A scalar predicate can also be evaluated for each reachable encoded ordinal.
//! For example, `clz(x: u161) < 2` selects the first two one-hot positions
//! after reversing x, so only x[159:161] is relevant. This produces one boolean
//! bit; it does not produce a two-bit count. The zero-input ordinal is
//! evaluated too.

use crate::ir_cost::IrCost;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_match::{self, MatchCtx};
use xlsynth_pir::ir_utils;
use xlsynth_pir::ir_value_utils::ir_bits_to_usize;
use xlsynth_pir::math::ceil_log2;
use xlsynth_pir::{IrBits, IrValue};

fn bits_width(ty: &Type) -> Option<usize> {
    match ty {
        Type::Bits(width) => Some(*width),
        _ => None,
    }
}

fn literal_usize_value(f: &ir::Fn, node: NodeRef) -> Option<usize> {
    ir_bits_to_usize(&literal_bits(f, node)?)
}

/// A decoder's index and, for an expanded tree, the bits it actually tests.
struct DecodeAmount {
    amount: NodeRef,
    tested_bits: Option<usize>,
}

/// An exact-width operation in the affine path from an encoded one-hot index.
enum IndexStep {
    Widen(usize),
    Add(IrBits),
    Subtract(IrBits),
    Reflect(IrBits),
}

/// Retains each arithmetic width instead of collapsing modular operations.
struct PriorityIndex {
    hot: NodeRef,
    encoded_width: usize,
    steps: Vec<IndexStep>,
}

impl PriorityIndex {
    /// Evaluates one reachable ordinal, including modular overflow/underflow.
    fn at(&self, ordinal: usize) -> IrBits {
        let mut bits = IrBits::make_ubits(self.encoded_width, ordinal as u64)
            .expect("lossless encoded ordinal");
        for step in &self.steps {
            bits = match step {
                IndexStep::Widen(width) => {
                    let mut bytes = bits.to_bytes();
                    bytes.resize(width.div_ceil(8), 0);
                    IrBits::from_le_bytes(*width, &bytes).expect("nontruncating zero extension")
                }
                IndexStep::Add(value) => bits.add(value),
                IndexStep::Subtract(value) => bits.sub(value),
                IndexStep::Reflect(value) => value.sub(&bits),
            };
        }
        bits
    }
}

/// Reads a width-independent constant for bit-vector arithmetic.
fn literal_bits(f: &ir::Fn, node: NodeRef) -> Option<IrBits> {
    match &f.get_node(node).payload {
        NodePayload::Literal(value) => value.to_bits().ok(),
        _ => None,
    }
}

/// Matches encode(one_hot(...)) through constant +/- and lossless extensions.
fn priority_index(f: &ir::Fn, node: NodeRef) -> Option<PriorityIndex> {
    let width = bits_width(&f.get_node(node).ty)?;
    let (arg, step) = match &f.get_node(node).payload {
        NodePayload::Encode { arg: hot } => {
            if !matches!(f.get_node(*hot).payload, NodePayload::OneHot { .. })
                || width < ceil_log2(bits_width(&f.get_node(*hot).ty)?)
            {
                return None;
            }
            return Some(PriorityIndex {
                hot: *hot,
                encoded_width: width,
                steps: vec![],
            });
        }
        NodePayload::ZeroExt { arg, .. } => (*arg, IndexStep::Widen(width)),
        NodePayload::Nary(NaryOp::Concat, parts) => {
            let (arg, prefix) = parts.split_last()?;
            if !prefix
                .iter()
                .all(|&n| literal_bits(f, n).is_some_and(|b| b.is_zero()))
            {
                return None;
            }
            (*arg, IndexStep::Widen(width))
        }
        NodePayload::Binop(Binop::Add, ..) => {
            let bindings = MatchCtx::new(f).matches(
                node,
                ir_match::commutative_binop(Binop::Add, ir_match::any("a"), ir_match::any("b")),
            )?;
            let a = bindings.get_node("a")?;
            let b = bindings.get_node("b")?;
            if let Some(value) = literal_bits(f, a) {
                (b, IndexStep::Add(value))
            } else {
                (a, IndexStep::Add(literal_bits(f, b)?))
            }
        }
        NodePayload::Binop(Binop::Sub, a, b) => {
            if let Some(value) = literal_bits(f, *a) {
                (*b, IndexStep::Reflect(value))
            } else {
                (*a, IndexStep::Subtract(literal_bits(f, *b)?))
            }
        }
        _ => return None,
    };
    let mut index = priority_index(f, arg)?;
    index.steps.push(step);
    Some(index)
}

/// Separates a one-bit predicate mask from the affine index, in either order.
fn masked_priority_index(f: &ir::Fn, amount: NodeRef) -> Option<(PriorityIndex, Option<NodeRef>)> {
    if let Some(index) = priority_index(f, amount) {
        return Some((index, None));
    }
    let parts = MatchCtx::new(f).flattened_nary_operands(amount, NaryOp::And)?;
    let [a, b] = parts.as_slice() else {
        return None;
    };
    for (expression, mask) in [(*a, *b), (*b, *a)] {
        let predicate = match f.get_node(mask).payload {
            NodePayload::SignExt { arg, .. } if bits_width(&f.get_node(arg).ty) == Some(1) => arg,
            _ => continue,
        };
        if let Some(index) = priority_index(f, expression) {
            return Some((index, Some(predicate)));
        }
    }
    None
}

/// Removes an exact zero prefix or suffix from ordered concat segments.
fn without_zero_padding(
    f: &ir::Fn,
    node: NodeRef,
    padding_width: usize,
    high_padding: bool,
) -> Option<Vec<NodeRef>> {
    let mut parts = MatchCtx::new(f).flattened_nary_operands(node, NaryOp::Concat)?;
    let mut padding = 0usize;
    let mut consumed = 0usize;
    while padding < padding_width {
        let index = if high_padding {
            consumed
        } else {
            parts.len().checked_sub(consumed + 1)?
        };
        let part = *parts.get(index)?;
        if !literal_bits(f, part)?.is_zero() {
            return None;
        }
        padding = padding.checked_add(bits_width(&f.get_node(part).ty)?)?;
        if padding > padding_width {
            return None;
        }
        consumed += 1;
    }
    if high_padding {
        parts.drain(..consumed);
    } else {
        parts.truncate(parts.len() - consumed);
    }
    Some(parts)
}

/// Matches a decoder that may have been flattened into its parent's concat.
fn decoded_shift_low_segments(f: &ir::Fn, parts: &[NodeRef]) -> Option<(NodeRef, usize)> {
    if let [node] = parts {
        return decoded_shift_low_bits(f, *node);
    }
    let [bit, complement] = parts else {
        return None;
    };
    if !matches!(f.get_node(*complement).payload, NodePayload::Unop(Unop::Not, arg) if arg == *bit)
        || bits_width(&f.get_node(*complement).ty) != Some(1)
    {
        return None;
    }
    let NodePayload::BitSlice {
        arg: amount,
        start: 0,
        width: 1,
    } = f.get_node(*bit).payload
    else {
        return None;
    };
    // In MSB-first concat order, [amount[0], !amount[0]] is bits[2]:1
    // shifted by amount[0]. Reversing these segments changes the decoder.
    Some((amount, 1))
}

/// Recovers the shared amount from a binary select tree decoding its low bits.
fn decoded_shift_low_bits(f: &ir::Fn, node: NodeRef) -> Option<(NodeRef, usize)> {
    let width = bits_width(&f.get_node(node).ty)?;
    let ctx = MatchCtx::new(f);
    if let Some(parts) = ctx.flattened_nary_operands(node, NaryOp::Concat) {
        return decoded_shift_low_segments(f, &parts);
    }
    if width == 2 {
        let bindings = ctx.matches(
            node,
            ir_match::commutative(
                NaryOp::Xor,
                vec![ir_match::any("one"), ir_match::any("mask")],
            ),
        )?;
        let a = bindings.get_node("one")?;
        let b = bindings.get_node("mask")?;
        for (one, mut mask) in [(a, b), (b, a)] {
            if literal_usize_value(f, one) != Some(1) {
                continue;
            }
            if let Some(parts) = ctx.flattened_nary_operands(mask, NaryOp::And) {
                let mut nonliteral = parts
                    .into_iter()
                    .filter(|&n| literal_usize_value(f, n) != Some(3));
                mask = nonliteral.next()?;
                if nonliteral.next().is_some() {
                    continue;
                }
            }
            let NodePayload::SignExt { arg, .. } = f.get_node(mask).payload else {
                continue;
            };
            let NodePayload::BitSlice {
                arg: amount,
                start: 0,
                width: 1,
            } = f.get_node(arg).payload
            else {
                continue;
            };
            return Some((amount, 1));
        }
        return None;
    }
    let NodePayload::Sel {
        selector,
        cases,
        default: None,
    } = &f.get_node(node).payload
    else {
        return None;
    };
    let [a, b] = cases.as_slice() else {
        return None;
    };
    let half = width / 2;
    if half.checked_mul(2)? != width || half == 0 {
        return None;
    }
    let previous = without_zero_padding(f, *a, half, true)?;
    let same = without_zero_padding(f, *b, half, false)?;
    if previous != same {
        return None;
    }
    let (amount, low_bits) = decoded_shift_low_segments(f, &previous)?;
    if !MatchCtx::new(f).bit_slice_of(*selector, amount, low_bits, 1) {
        return None;
    }
    Some((amount, low_bits + 1))
}

/// Finds the amount used by a one-shift or its already expanded decode tree.
fn priority_decode_amount(f: &ir::Fn, root: NodeRef) -> Option<DecodeAmount> {
    if let NodePayload::Decode { arg, .. } = f.get_node(root).payload {
        return Some(DecodeAmount {
            amount: arg,
            tested_bits: None,
        });
    }
    if let NodePayload::Binop(Binop::Shll, one, amount) = f.get_node(root).payload
        && literal_usize_value(f, one) == Some(1)
    {
        return Some(DecodeAmount {
            amount,
            tested_bits: None,
        });
    }
    let ctx = MatchCtx::new(f);
    let parts = ctx.flattened_nary_operands(root, NaryOp::And)?;
    let [a, b] = parts.as_slice() else {
        return None;
    };
    for (tree, mask) in [(*a, *b), (*b, *a)] {
        let Some((amount, count)) = decoded_shift_low_bits(f, tree) else {
            continue;
        };
        let NodePayload::SignExt { arg, .. } = f.get_node(mask).payload else {
            continue;
        };
        let NodePayload::Unop(Unop::Not, overflow) = f.get_node(arg).payload else {
            continue;
        };
        if ctx.bit_slice_of(overflow, amount, count, 1) {
            // Matching the tree is only structural. The caller checks every
            // reachable amount for untested high bits before rewriting.
            return Some(DecodeAmount {
                amount,
                tested_bits: Some(count + 1),
            });
        }
    }
    None
}

/// Requires the count/decode intermediates to have no users outside this cone.
fn priority_result_cone_is_exclusive(
    f: &ir::Fn,
    root: NodeRef,
    predicate: Option<NodeRef>,
    hot: NodeRef,
) -> bool {
    let mut cone = vec![false; f.nodes.len()];
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if Some(node) == predicate || node == hot || cone[node.index] {
            continue;
        }
        if matches!(f.get_node(node).payload, NodePayload::Literal(_)) {
            continue;
        }
        cone[node.index] = true;
        stack.extend(ir_utils::operands(&f.get_node(node).payload));
    }
    // The root survives with a different payload, and the predicate/one-hot
    // subgraph survives intact. Only eliminated intermediate work must be
    // exclusive; multiple users inside an expanded decode tree are allowed.
    for (index, node) in f.nodes.iter().enumerate() {
        if cone[index] {
            continue;
        }
        if ir_utils::operands(&node.payload)
            .iter()
            .any(|operand| *operand != root && cone[operand.index])
        {
            return false;
        }
    }
    !f.ret_node_ref
        .is_some_and(|ret| ret != root && cone[ret.index])
}

/// Allocates helper text IDs once so wide wiring maps remain linear in size.
struct NodeAppender {
    next_text_id: Option<usize>,
}

impl NodeAppender {
    fn new(f: &ir::Fn) -> Self {
        Self {
            next_text_id: f
                .nodes
                .iter()
                .map(|n| n.text_id)
                .max()
                .unwrap_or(0)
                .checked_add(1),
        }
    }

    fn push(&mut self, f: &mut ir::Fn, ty: Type, payload: NodePayload) -> Option<NodeRef> {
        let text_id = self.next_text_id?;
        let result = NodeRef {
            index: f.nodes.len(),
        };
        f.nodes.push(ir::Node {
            text_id,
            name: None,
            ty,
            payload,
            pos: None,
        });
        self.next_text_id = text_id.checked_add(1);
        Some(result)
    }
}

/// Builds the statically wired decoder, preserving the one-hot sentinel.
fn wired_result(
    f: &mut ir::Fn,
    builder: &mut NodeAppender,
    hot: NodeRef,
    output_sources: &[Vec<usize>],
) -> Option<NodeRef> {
    let zero = builder.push(f, Type::Bits(1), NodePayload::Literal(IrValue::bool(false)))?;
    let mut bits = Vec::with_capacity(output_sources.len());
    for sources in output_sources.iter().rev() {
        let inputs: Vec<_> = sources
            .iter()
            .map(|&start| {
                builder.push(
                    f,
                    Type::Bits(1),
                    NodePayload::BitSlice {
                        arg: hot,
                        start,
                        width: 1,
                    },
                )
            })
            .collect::<Option<Vec<_>>>()?;
        bits.push(match inputs.as_slice() {
            [] => zero,
            [bit] => *bit,
            _ => builder.push(f, Type::Bits(1), NodePayload::Nary(NaryOp::Or, inputs))?,
        });
    }
    builder.push(
        f,
        Type::Bits(bits.len()),
        NodePayload::Nary(NaryOp::Concat, bits),
    )
}

/// A one-bit reduction or unsigned comparison of an exact-width index.
enum IndexPredicate {
    Reduction {
        op: Unop,
        start: usize,
        width: usize,
    },
    Comparison {
        op: Binop,
        constant: IrBits,
    },
}

impl IndexPredicate {
    /// Evaluates a reachable ordinal without converting wide values to
    /// integers.
    fn at(&self, amount: &IrBits) -> bool {
        match self {
            Self::Reduction { op, start, width } => {
                let mut bits = (*start..start + width).map(|i| amount.get_bit(i).unwrap());
                match op {
                    Unop::OrReduce => bits.any(|bit| bit),
                    Unop::AndReduce => bits.all(|bit| bit),
                    Unop::XorReduce => bits.fold(false, |parity, bit| parity ^ bit),
                    _ => unreachable!("matched bit reduction"),
                }
            }
            Self::Comparison { op, constant } => {
                let order = amount
                    .to_bytes()
                    .iter()
                    .rev()
                    .cmp(constant.to_bytes().iter().rev());
                match op {
                    Binop::Eq => order.is_eq(),
                    Binop::Ne => !order.is_eq(),
                    Binop::Ult => order.is_lt(),
                    Binop::Ule => !order.is_gt(),
                    Binop::Ugt => order.is_gt(),
                    Binop::Uge => !order.is_lt(),
                    _ => unreachable!("matched unsigned comparison"),
                }
            }
        }
    }
}

/// Matches reductions of index bits, or comparisons with a literal on either
/// side.
fn priority_index_predicate(f: &ir::Fn, root: NodeRef) -> Option<(PriorityIndex, IndexPredicate)> {
    if bits_width(&f.get_node(root).ty) != Some(1) {
        return None;
    }
    let (amount, predicate) = match &f.get_node(root).payload {
        NodePayload::Unop(op @ (Unop::OrReduce | Unop::AndReduce | Unop::XorReduce), arg) => {
            let (amount, start, width) = match f.get_node(*arg).payload {
                NodePayload::BitSlice { arg, start, width } => (arg, start, width),
                _ => (*arg, 0, bits_width(&f.get_node(*arg).ty)?),
            };
            if width == 0 {
                // Empty reductions are constants; leave them to ordinary
                // folding.
                return None;
            }
            (
                amount,
                IndexPredicate::Reduction {
                    op: *op,
                    start,
                    width,
                },
            )
        }
        NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), ..) => {
            let bindings = MatchCtx::new(f).matches(
                root,
                ir_match::commutative_binop(*op, ir_match::any("a"), ir_match::any("b")),
            )?;
            let a = bindings.get_node("a")?;
            let b = bindings.get_node("b")?;
            let (amount, constant) = if let Some(value) = literal_bits(f, a) {
                (b, value)
            } else {
                (a, literal_bits(f, b)?)
            };
            (amount, IndexPredicate::Comparison { op: *op, constant })
        }
        NodePayload::Binop(op @ (Binop::Ult | Binop::Ule | Binop::Ugt | Binop::Uge), a, b) => {
            let (amount, op, constant) = if let Some(value) = literal_bits(f, *b) {
                (*a, *op, value)
            } else {
                // Reversing a noncommutative comparison reverses its direction.
                let reversed = match op {
                    Binop::Ult => Binop::Ugt,
                    Binop::Ule => Binop::Uge,
                    Binop::Ugt => Binop::Ult,
                    Binop::Uge => Binop::Ule,
                    _ => unreachable!("matched unsigned comparison"),
                };
                (*b, reversed, literal_bits(f, *a)?)
            };
            (amount, IndexPredicate::Comparison { op, constant })
        }
        _ => return None,
    };
    Some((priority_index(f, amount)?, predicate))
}

/// Selects the smaller truth set, using the one-hot invariant to complement it.
fn wired_predicate(
    f: &mut ir::Fn,
    builder: &mut NodeAppender,
    index: &PriorityIndex,
    predicate: &IndexPredicate,
) -> Option<NodeRef> {
    let hot_width = bits_width(&f.get_node(index.hot).ty)?;
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();
    for ordinal in 0..hot_width {
        if predicate.at(&index.at(ordinal)) {
            accepted.push(ordinal);
        } else {
            rejected.push(ordinal);
        }
    }
    // one_hot sets exactly one of these bits, including its all-zero sentinel.
    // Complementing an arbitrary at-most-one value would be incorrect here.
    let complement = rejected.len() < accepted.len();
    let sources = if complement { rejected } else { accepted };
    let wired = wired_result(f, builder, index.hot, &[sources])?;
    if complement {
        builder.push(f, Type::Bits(1), NodePayload::Unop(Unop::Not, wired))
    } else {
        Some(wired)
    }
}

/// Replaces exclusive count predicates while retaining shared inputs/one-hot
/// values.
fn rewrite_priority_predicates(f: &mut ir::Fn) -> Option<usize> {
    let mut rewrites = 0;
    let mut builder = NodeAppender::new(f);
    for index in 0..f.nodes.len() {
        let root = NodeRef { index };
        let Some((index_expr, predicate)) = priority_index_predicate(f, root) else {
            continue;
        };
        if !priority_result_cone_is_exclusive(f, root, None, index_expr.hot) {
            continue;
        }
        let wired = wired_predicate(f, &mut builder, &index_expr, &predicate)?;
        f.nodes[root.index].payload = f.get_node(wired).payload.clone();
        rewrites += 1;
    }
    Some(rewrites)
}

/// Rewrites exclusive affine-index decoders, returning None if text IDs run
/// out. The caller must discard the partially rewritten function on exhaustion.
fn rewrite_priority_results(f: &mut ir::Fn) -> Option<usize> {
    let mut rewrites = 0;
    let mut builder = NodeAppender::new(f);
    for index in 0..f.nodes.len() {
        let root = NodeRef { index };
        let Some(width) = bits_width(&f.get_node(root).ty).filter(|&w| w != 0) else {
            continue;
        };
        let Some(decoded) = priority_decode_amount(f, root) else {
            continue;
        };
        let Some((index_expr, predicate)) = masked_priority_index(f, decoded.amount) else {
            continue;
        };
        if !priority_result_cone_is_exclusive(f, root, predicate, index_expr.hot) {
            continue;
        }
        let hot_width = bits_width(&f.get_node(index_expr.hot).ty).expect("matched one-hot bits");
        let mut sources = vec![Vec::new(); width];
        let mut complete = true;
        for ordinal in 0..hot_width {
            let amount = index_expr.at(ordinal);
            if decoded.tested_bits.is_some_and(|tested| {
                (tested..amount.get_bit_count()).any(|bit| amount.get_bit(bit).unwrap())
            }) {
                complete = false;
                break;
            }
            if let Some(destination) = ir_bits_to_usize(&amount).filter(|&i| i < width) {
                sources[destination].push(ordinal);
            }
        }
        if !complete {
            continue;
        }
        let wired = wired_result(f, &mut builder, index_expr.hot, &sources)?;
        f.nodes[root.index].payload = if let Some(predicate) = predicate {
            let one = builder.push(
                f,
                Type::Bits(width),
                NodePayload::Literal(IrValue::make_ubits(width, 1).expect("nonzero width")),
            )?;
            NodePayload::Sel {
                selector: predicate,
                cases: vec![one, wired],
                default: None,
            }
        } else {
            f.get_node(wired).payload.clone()
        };
        rewrites += 1;
    }
    Some(rewrites)
}

/// A basis-IR alternative and the number of decoded priority results it
/// replaces.
struct PriorityResultCandidate {
    function: ir::Fn,
    rewrites: usize,
}

fn candidate(f: &ir::Fn) -> Option<PriorityResultCandidate> {
    let mut function = f.clone();
    let rewrites =
        rewrite_priority_results(&mut function)? + rewrite_priority_predicates(&mut function)?;
    if rewrites == 0 {
        return None;
    }
    ir_utils::compact_and_toposort_in_place(&mut function).ok()?;
    Some(PriorityResultCandidate { function, rewrites })
}

/// Installs a strictly profitable basis-IR candidate.
///
/// Unsupported shapes, shared intermediate indices, exhausted IDs, invalid
/// costs, and cost errors preserve the input. The caller may cost prepared
/// clones, but preparation never escapes into the returned function.
pub fn rewrite_with_evaluator(
    f: &mut ir::Fn,
    evaluator: &mut impl FnMut(&ir::Fn) -> Result<IrCost, String>,
) -> usize {
    let Some(candidate) = candidate(f) else {
        return 0;
    };
    let (Ok(old_cost), Ok(new_cost)) = (evaluator(f), evaluator(&candidate.function)) else {
        return 0;
    };
    if !new_cost.is_pareto_improvement_on(old_cost) {
        return 0;
    }
    *f = candidate.function;
    candidate.rewrites
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_g8r::check_equivalence::check_equivalence_via_toolchain;
    use xlsynth_pir::ir_parser::Parser;

    /// Constructs unrelated priority encoders with exact-width affine indices.
    fn affine_case(n: usize, low: bool, form: &str, masked: bool) -> String {
        let cw = ceil_log2(n + 1);
        let width = n + 3;
        let (aw, expression) = match form {
            "identity" => (cw, "  amount: bits[C] = identity(count, id=5)".to_string()),
            "add" => (cw, "  one: bits[C] = literal(value=1, id=4)\n  amount: bits[C] = add(one, count, id=5)".to_string()),
            "subtract" => (cw, "  one: bits[C] = literal(value=1, id=4)\n  amount: bits[C] = sub(count, one, id=5)".to_string()),
            "reflect" => (cw, "  zero: bits[C] = literal(value=0, id=4)\n  amount: bits[C] = sub(zero, count, id=5)".to_string()),
            "widen_after" => (cw + 1, "  one: bits[C] = literal(value=1, id=4)\n  added: bits[C] = add(count, one, id=5)\n  amount: bits[A] = zero_ext(added, new_bit_count=A, id=6)".to_string()),
            "widen_before" => (cw + 1, "  wide: bits[A] = zero_ext(count, new_bit_count=A, id=4)\n  one: bits[A] = literal(value=1, id=5)\n  amount: bits[A] = add(wide, one, id=6)".to_string()),
            "wide_constant" => (80, "  wide: bits[80] = zero_ext(count, new_bit_count=80, id=4)\n  constant: bits[80] = literal(value=1208925819614629174706175, id=5)\n  amount: bits[80] = add(wide, constant, id=6)".to_string()),
            _ => unreachable!(),
        };
        let expression = expression
            .replace("C", &cw.to_string())
            .replace("A", &aw.to_string());
        let (mask, amount) = if masked {
            (
                format!(
                    "  mask: bits[{aw}] = sign_ext(p, new_bit_count={aw}, id=7)\n  gated: bits[{aw}] = and(mask, amount, id=8)\n"
                ),
                "gated",
            )
        } else {
            (String::new(), "amount")
        };
        format!(
            r#"package affine
top fn main(x: bits[{n}] id=1, p: bits[1] id=2) -> bits[{width}] {{
  hot: bits[{hot_width}] = one_hot(x, lsb_prio={low}, id=3)
  count: bits[{cw}] = encode(hot, id=9)
{expression}
{mask}  one_out: bits[{width}] = literal(value=1, id=10)
  ret out: bits[{width}] = shll(one_out, {amount}, id=11)
}}
"#,
            hot_width = n + 1
        )
    }

    #[test]
    fn affine_wiring_proves_both_priorities_and_modular_boundaries() {
        for n in [3, 7, 8, 65] {
            for low in [false, true] {
                for form in [
                    "add",
                    "subtract",
                    "reflect",
                    "widen_after",
                    "widen_before",
                    "wide_constant",
                ] {
                    for masked in [false, true] {
                        let source = affine_case(n, low, form, masked);
                        let package = Parser::new(&source).parse_and_validate_package().unwrap();
                        let mut rewritten = package.get_top_fn().unwrap().clone();
                        assert_eq!(
                            rewrite_priority_results(&mut rewritten),
                            Some(1),
                            "{n} {low} {form} {masked}"
                        );
                        ir_utils::compact_and_toposort_in_place(&mut rewritten).unwrap();
                        let result = rewritten;
                        check_equivalence_via_toolchain(
                            &source,
                            &format!("package result\n\ntop {result}"),
                        )
                        .unwrap_or_else(|e| panic!("{n} {low} {form} {masked}: {e}"));
                    }
                }
            }
        }
    }

    #[test]
    fn expanded_decoder_accepts_ordered_flattened_concat_leaves() {
        let source = r#"package expanded
top fn main(x: bits[4] id=1) -> bits[4] {
  hot: bits[5] = one_hot(x, lsb_prio=true, id=2)
  amount: bits[3] = encode(hot, id=3)
  bit0: bits[1] = bit_slice(amount, start=0, width=1, id=4)
  complement: bits[1] = not(bit0, id=5)
  bit1: bits[1] = bit_slice(amount, start=1, width=1, id=6)
  zero: bits[2] = literal(value=0, id=7)
  low: bits[4] = concat(zero, bit0, complement, id=8)
  high: bits[4] = concat(bit0, complement, zero, id=9)
  tree: bits[4] = sel(bit1, cases=[low, high], id=10)
  overflow: bits[1] = bit_slice(amount, start=2, width=1, id=11)
  valid: bits[1] = not(overflow, id=12)
  mask: bits[4] = sign_ext(valid, new_bit_count=4, id=13)
  ret out: bits[4] = and(mask, tree, id=14)
}
"#;
        let nested = source
            .replace(
                "  low:",
                "  leaf: bits[2] = concat(bit0, complement, id=15)\n  low:",
            )
            .replace("concat(zero, bit0, complement,", "concat(zero, leaf,")
            .replace("concat(bit0, complement, zero,", "concat(leaf, zero,");
        let split_padding = source
            .replace("zero: bits[2]", "zero: bits[1]")
            .replace("concat(zero, bit0,", "concat(zero, zero, bit0,")
            .replace(
                "concat(bit0, complement, zero,",
                "concat(bit0, complement, zero, zero,",
            );
        for source in [source.to_string(), nested, split_padding] {
            for low in [false, true] {
                let source = source.replace("lsb_prio=true", &format!("lsb_prio={low}"));
                let package = Parser::new(&source).parse_and_validate_package().unwrap();
                let mut rewritten = package.get_top_fn().unwrap().clone();
                assert_eq!(rewrite_priority_results(&mut rewritten), Some(1));
                ir_utils::compact_and_toposort_in_place(&mut rewritten).unwrap();
                let result = rewritten;
                check_equivalence_via_toolchain(
                    &source,
                    &format!("package result\n\ntop {result}"),
                )
                .unwrap();

                let reversed = source.replace("bit0, complement", "complement, bit0");
                let package = Parser::new(&reversed).parse_and_validate_package().unwrap();
                let mut rewritten = package.get_top_fn().unwrap().clone();
                assert_eq!(rewrite_priority_results(&mut rewritten), Some(0));
            }
        }
    }

    #[test]
    fn expanded_decoder_requires_all_reachable_overflow_bits() {
        let fixture = include_str!("../tests/fixtures/mffc_regressions/priority_historical.ir");
        // This tree tests amount[5] but ignores amount[6]. Changing 32-count
        // to 96-count makes the ignored bit reachable; a true shift would
        // then be zero while this partially guarded tree can produce a bit.
        let source = fixture.replace("literal(value=32, id=67)", "literal(value=96, id=67)");
        assert_ne!(source, fixture);
        let package = Parser::new(&source).parse_and_validate_package().unwrap();
        let mut f = package.get_top_fn().unwrap().clone();
        assert_eq!(rewrite_priority_results(&mut f), Some(0));
    }

    #[test]
    fn retained_predicate_is_a_boundary_but_shared_count_is_not() {
        let source = affine_case(8, true, "widen_before", true)
            .replace("-> bits[11]", "-> (bits[11], bits[1])")
            .replace("ret out:", "decoded:")
            .replace(
                "\n}",
                "\n  ret result: (bits[11], bits[1]) = tuple(decoded, p, id=12)\n}",
            );
        let package = Parser::new(&source).parse_and_validate_package().unwrap();
        let mut f = package.get_top_fn().unwrap().clone();
        assert_eq!(rewrite_priority_results(&mut f), Some(1));
        let shared_count = source
            .replace("(bits[11], bits[1])", "(bits[11], bits[4])")
            .replace("tuple(decoded, p,", "tuple(decoded, count,");
        let package = Parser::new(&shared_count)
            .parse_and_validate_package()
            .unwrap();
        let mut f = package.get_top_fn().unwrap().clone();
        assert_eq!(rewrite_priority_results(&mut f), Some(0));
    }

    #[test]
    fn exhausted_text_ids_preserve_input() {
        let cases = [
            (
                r#"package exhausted
top fn main(x: bits[32] id=1) -> bits[32] {
  ret out: bits[32] = identity(x, id=2)
}
"#,
                false,
            ),
            (
                r#"package exhausted
top fn main(x: bits[32] id=1) -> bits[33] {
  hot: bits[33] = one_hot(x, lsb_prio=false, id=2)
  count: bits[6] = encode(hot, id=3)
  ret out: bits[33] = decode(count, width=33, id=4)
}
"#,
                true,
            ),
        ];
        for (source, matches) in cases {
            for largest_id in [usize::MAX - 1, usize::MAX] {
                let package = Parser::new(source).parse_and_validate_package().unwrap();
                let mut f = package.get_top_fn().unwrap().clone();
                let param_index = f.params[0].index;
                f.nodes[param_index].text_id = largest_id;
                let original = f.to_string();
                let mut partial = f.clone();
                assert_eq!(
                    rewrite_priority_results(&mut partial),
                    if matches { None } else { Some(0) },
                    "matches={matches}, largest_id={largest_id}"
                );
                // With one ID left, allocation succeeds once before exhaustion;
                // run must discard that partially emitted candidate as well.
                if matches && largest_id == usize::MAX - 1 {
                    assert_eq!(partial.nodes.len(), f.nodes.len() + 1);
                    assert_eq!(partial.nodes.last().unwrap().text_id, usize::MAX);
                }
                assert_eq!(
                    rewrite_with_evaluator(&mut f, &mut |_| panic!("no candidate to cost")),
                    0
                );
                assert_eq!(f.to_string(), original);
            }
        }
    }

    #[test]
    fn direct_decode_is_generic_and_emits_basis_ir() {
        let source = r#"package decoder
top fn main(x: bits[32] id=1, p: bits[1] id=2) -> bits[33] {
  hot: bits[33] = one_hot(x, lsb_prio=false, id=3)
  count: bits[6] = encode(hot, id=4)
  ret out: bits[33] = decode(count, width=33, id=5)
}
"#;
        let package = Parser::new(source).parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();
        let mut result = f.clone();
        let mut evaluator = crate::cost::G8rFunctionCostEvaluator::default();
        assert_eq!(
            rewrite_with_evaluator(&mut result, &mut |f| evaluator.estimate(f)),
            1
        );
        assert_ne!(f.to_string(), result.to_string());
        xlsynth::IrPackage::parse_ir(&format!("package result\n\ntop {result}"), None)
            .expect("candidate contains only ordinary XLS IR");
        check_equivalence_via_toolchain(source, &format!("package result\n\ntop {result}"))
            .unwrap();
    }
    #[test]
    fn cost_errors_ties_and_either_regression_preserve_input() {
        let source = affine_case(8, true, "widen_before", true);
        let package = Parser::new(&source).parse_and_validate_package().unwrap();
        let original = package.get_top_fn().unwrap();
        assert!(candidate(original).is_some());
        let old = IrCost {
            area: 100,
            delay: 10.0,
        };
        for new in [
            Ok(old),
            Ok(IrCost {
                area: 101,
                delay: 9.0,
            }),
            Ok(IrCost {
                area: 99,
                delay: 11.0,
            }),
            Ok(IrCost {
                area: 99,
                delay: f64::NAN,
            }),
            Err("uncostable function".to_string()),
        ] {
            let mut f = original.clone();
            let mut first = true;
            let rewrites = rewrite_with_evaluator(&mut f, &mut |_| {
                if std::mem::take(&mut first) {
                    Ok(old)
                } else {
                    new.clone()
                }
            });
            assert_eq!(rewrites, 0);
            assert_eq!(f.to_string(), original.to_string());
        }
        let mut f = original.clone();
        assert_eq!(
            rewrite_with_evaluator(&mut f, &mut |_| Err("unavailable baseline".to_string())),
            0
        );
        assert_eq!(f.to_string(), original.to_string());
    }

    /// Builds a count predicate with exact-width arithmetic before its
    /// consumer.
    fn scalar_case(n: usize, low: bool, form: &str, predicate: &str) -> String {
        let cw = ceil_log2(n + 1);
        let (aw, steps) = match form {
            "identity" => (cw, String::new()),
            "wrap_add" => (
                cw,
                format!(
                    "  k: bits[{cw}] = literal(value=1, id=4)\n  amount: bits[{cw}] = add(k, count, id=5)\n"
                ),
            ),
            "wrap_sub" => (
                cw,
                format!(
                    "  k: bits[{cw}] = literal(value=1, id=4)\n  amount: bits[{cw}] = sub(count, k, id=5)\n"
                ),
            ),
            "reflect" => (
                cw,
                format!(
                    "  k: bits[{cw}] = literal(value=0, id=4)\n  amount: bits[{cw}] = sub(k, count, id=5)\n"
                ),
            ),
            "wide_add" => (
                80,
                format!(
                    "  wide: bits[80] = zero_ext(count, new_bit_count=80, id=4)\n  k: bits[80] = literal(value=1208925819614629174706175, id=5)\n  amount: bits[80] = add(wide, k, id=6)\n"
                ),
            ),
            _ => unreachable!("test arithmetic"),
        };
        let amount = if form == "identity" {
            "count"
        } else {
            "amount"
        };
        let consumer = match predicate {
            "or_slice" => {
                let start = usize::from(aw > 1);
                format!(
                    "  view: bits[{}] = bit_slice({amount}, start={start}, width={}, id=8)\n  ret out: bits[1] = or_reduce(view, id=9)",
                    aw - start,
                    aw - start
                )
            }
            "and" | "xor" => format!("  ret out: bits[1] = {predicate}_reduce({amount}, id=9)"),
            "left_ult" => format!(
                "  limit: bits[{aw}] = literal(value=1, id=8)\n  ret out: bits[1] = ult(limit, {amount}, id=9)"
            ),
            op => format!(
                "  limit: bits[{aw}] = literal(value=1, id=8)\n  ret out: bits[1] = {op}({amount}, limit, id=9)"
            ),
        };
        format!(
            "package scalar\ntop fn main(x: bits[{n}] id=1) -> bits[1] {{\n  hot: bits[{}] = one_hot(x, lsb_prio={low}, id=2)\n  count: bits[{cw}] = encode(hot, id=3)\n{steps}{consumer}\n}}\n",
            n + 1
        )
    }

    #[test]
    fn scalar_predicates_preserve_sentinels_arithmetic_wrap_and_wide_values() {
        for (n, form) in [
            (1, "identity"),
            (3, "wrap_add"),
            (8, "wrap_sub"),
            (65, "reflect"),
            (7, "wide_add"),
        ] {
            for low in [false, true] {
                for predicate in ["or_slice", "eq", "left_ult"] {
                    let source = scalar_case(n, low, form, predicate);
                    let package = Parser::new(&source).parse_and_validate_package().unwrap();
                    let mut rewritten = package.get_top_fn().unwrap().clone();
                    assert_eq!(
                        rewrite_priority_predicates(&mut rewritten),
                        Some(1),
                        "{n} {low} {form} {predicate}"
                    );
                    ir_utils::compact_and_toposort_in_place(&mut rewritten).unwrap();
                    check_equivalence_via_toolchain(
                        &source,
                        &format!("package result\n\ntop {rewritten}"),
                    )
                    .unwrap();
                }
            }
        }
        for predicate in ["and", "xor", "ne", "ult", "ule", "ugt", "uge"] {
            let source = scalar_case(3, true, "wrap_add", predicate);
            let package = Parser::new(&source).parse_and_validate_package().unwrap();
            let mut rewritten = package.get_top_fn().unwrap().clone();
            assert_eq!(rewrite_priority_predicates(&mut rewritten), Some(1));
            ir_utils::compact_and_toposort_in_place(&mut rewritten).unwrap();
            check_equivalence_via_toolchain(&source, &format!("package result\n\ntop {rewritten}"))
                .unwrap();
        }
    }

    #[test]
    fn scalar_predicates_reject_shared_counts_and_unsupported_paths() {
        let source = scalar_case(8, true, "identity", "or_slice");
        let shared = source
            .replace("-> bits[1]", "-> (bits[1], bits[4])")
            .replace("ret out:", "out:")
            .replace(
                "\n}",
                "\n  ret both: (bits[1], bits[4]) = tuple(out, count, id=10)\n}",
            );
        let truncated_arithmetic = source.replace(
            "  view: bits[3] = bit_slice(count, start=1, width=3, id=8)",
            "  trunc: bits[3] = bit_slice(count, start=0, width=3, id=11)\n  k: bits[3] = literal(value=1, id=12)\n  view: bits[3] = add(trunc, k, id=8)");
        let unknown_hot = source
            .replace("x: bits[8] id=1", "x: bits[8] id=1, unknown: bits[9] id=11")
            .replace("encode(hot,", "encode(unknown,");
        let empty_reduction = source.replace(
            "view: bits[3] = bit_slice(count, start=1, width=3",
            "view: bits[0] = bit_slice(count, start=1, width=0",
        );
        for input in [
            shared,
            truncated_arithmetic,
            unknown_hot,
            empty_reduction,
            scalar_case(8, false, "identity", "slt"),
        ] {
            let package = Parser::new(&input).parse_and_validate_package().unwrap();
            let mut rewritten = package.get_top_fn().unwrap().clone();
            let original = rewritten.to_string();
            assert_eq!(
                rewrite_priority_predicates(&mut rewritten),
                Some(0),
                "{input}"
            );
            assert_eq!(rewritten.to_string(), original);
        }
    }

    #[test]
    fn scalar_predicates_allow_shared_onehot_but_preserve_input_on_id_exhaustion() {
        let source = scalar_case(8, false, "identity", "or_slice")
            .replace("-> bits[1]", "-> (bits[1], bits[9])")
            .replace("ret out:", "out:")
            .replace(
                "\n}",
                "\n  ret both: (bits[1], bits[9]) = tuple(out, hot, id=10)\n}",
            );
        let package = Parser::new(&source).parse_and_validate_package().unwrap();
        let original = package.get_top_fn().unwrap();
        let mut rewritten = original.clone();
        assert_eq!(rewrite_priority_predicates(&mut rewritten), Some(1));
        ir_utils::compact_and_toposort_in_place(&mut rewritten).unwrap();
        check_equivalence_via_toolchain(&source, &format!("package result\n\ntop {rewritten}"))
            .unwrap();
        for largest_id in [usize::MAX - 1, usize::MAX] {
            let mut rewritten = original.clone();
            rewritten.nodes[original.params[0].index].text_id = largest_id;
            let before = rewritten.to_string();
            assert_eq!(
                rewrite_with_evaluator(&mut rewritten, &mut |_| panic!(
                    "exhausted candidate must not be costed"
                )),
                0
            );
            assert_eq!(rewritten.to_string(), before);
        }
    }
}
