// SPDX-License-Identifier: Apache-2.0

//! Express a rounded affine table using its upper address and a low-address
//! carry.

use super::RewriteIds;
use crate::gatify::ir2gate::demanded_bits;
use std::collections::{HashMap, HashSet};
use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::{IrBits, IrValue, ir_utils, math::ceil_log2};

/// Lowers live rounded affine reads to address arithmetic.
pub(super) fn rewrite(f: &mut ir::Fn, mut origins: Option<&mut HashMap<usize, usize>>) -> usize {
    let mut models = HashMap::new();
    let mut demand = None;
    let mut ids = None;
    let mut changed = 0;
    for i in 0..f.nodes.len() {
        let NodePayload::ArrayIndex { array, indices, .. } = &f.nodes[i].payload else {
            continue;
        };
        let [addr] = indices[..] else { continue };
        let Type::Bits(n) = *f.get_node_ty(addr) else {
            continue;
        };
        let NodePayload::Literal(IrValue::Array(table)) = &f.get_node(*array).payload else {
            continue;
        };
        let entries = table.elements();
        // Every possible address names a real row, under either IR bounds
        // convention.
        if entries.len() < 4
            || !entries.len().is_power_of_two()
            || entries.len().ilog2() as usize != n
            || matches!(f.get_node(addr).payload, NodePayload::Literal(_))
        {
            continue;
        }
        let plan = models.entry(*array).or_insert_with(|| {
            let words = entries
                .iter()
                .map(|e| e.as_bits().ok())
                .collect::<Option<Vec<_>>>()?;
            let page = find_model(&words)?;
            let (bias, residue) = quantized_bias(&page)?;
            Some((page, bias, residue))
        });
        let Some((page, bias, residue)) = plan.as_ref() else {
            continue;
        };
        // Every exception emits an n-bit address comparison, even when only
        // one result bit is used. Compare just the exception address storage
        // with the entire truth table for the observed bits.
        let address_bits = n.saturating_mul(page.exceptions.len());
        if address_bits >= entries.len() {
            // No demand analysis is needed if even one column costs more.
            // Replacing a read only adds uses of its original index, which
            // already demands every bit. The masks for pending reads remain
            // conservative until the final compaction.
            let masks = demand.get_or_insert_with(|| {
                demanded_bits::analyze(f, &ir_utils::get_topological(f), f.ret_node_ref.as_slice())
            });
            let observed = masks
                .get(&NodeRef { index: i })
                .map_or(0, |bits| bits.iter().filter(|&&b| b).count());
            if address_bits >= entries.len().saturating_mul(observed) {
                continue;
            }
        }
        let mut builder = Emitter {
            first_helper: f.nodes.len(),
            helpers: Vec::new(),
            literals: HashMap::new(),
        };
        let result = builder.emit(addr, n, page, bias.clone(), residue.clone());
        let ids = ids.get_or_insert_with(|| RewriteIds::new(f, origins.take()));
        if !ids.has_room(builder.helpers.len()) {
            continue;
        }
        let read_id = f.nodes[i].text_id;
        log::debug!(
            "quantized_affine_read read={} index={} bias={} k={} shift={} subtract={} exceptions={}",
            read_id,
            f.get_node(addr).text_id,
            bias,
            page.base.len().ilog2(),
            page.shift,
            page.subtract,
            page.exceptions.len()
        );
        for (width, payload) in builder.helpers {
            f.nodes.push(ir::Node {
                text_id: ids.fresh(read_id),
                name: None,
                ty: Type::Bits(width),
                payload,
                pos: None,
            });
        }
        let read = &mut f.nodes[i];
        super::clear_opcode_name(read);
        read.payload = NodePayload::Unop(Unop::Identity, result);
        changed += 1;
    }
    if changed != 0 {
        // Drop table literals whose last read was replaced.
        super::mark_dead_nodes_as_nil(f);
    }
    changed
}

/// The low-address pattern can cross precisely one carry (or borrow) boundary.
fn quantized_bias(page: &PageModel) -> Option<(IrBits, IrBits)> {
    let level = &page.base[0];
    let width = level.get_bit_count();
    let lanes = page.base.len();
    let k = lanes.ilog2() as usize;
    // A constant within-page value translates only the coarse address.
    // Exceptions can still distinguish low bits, but the arithmetic itself
    // cannot.
    let boundary = page.base.iter().position(|v| v != level)?;
    let step = IrBits::from_lsb_fn(width, |i| i == page.shift);
    let adjacent = if page.subtract {
        level.sub(&step)
    } else {
        level.add(&step)
    };
    if !page.base[boundary..].iter().all(|v| v == &adjacent) {
        return None;
    }
    let phase = if page.subtract {
        boundary - 1
    } else {
        lanes - boundary
    };
    let bias = IrBits::from_lsb_fn(width - page.shift + k, |i| {
        if i < k {
            phase >> i & 1 != 0
        } else {
            level.get_bit(i - k + page.shift).unwrap()
        }
    });
    Some((bias, level.width_slice(0, page.shift as i64)))
}

/// Stages one read's helpers until their complete text-ID range is available.
struct Emitter {
    first_helper: usize,
    helpers: Vec<(usize, NodePayload)>,
    literals: HashMap<IrBits, NodeRef>,
}

impl Emitter {
    fn emit(
        &mut self,
        addr: NodeRef,
        n: usize,
        model: &PageModel,
        bias: IrBits,
        residue: IrBits,
    ) -> NodeRef {
        let k = model.base.len().ilog2() as usize;
        let w = model.base[0].get_bit_count();
        let upper_width = w - model.shift;
        let low = self.push(
            k,
            NodePayload::BitSlice {
                arg: addr,
                start: 0,
                width: k,
            },
        );
        let high_width = (n - k).min(upper_width);
        let mut high = self.push(
            high_width,
            NodePayload::BitSlice {
                arg: addr,
                start: k,
                width: high_width,
            },
        );
        if high_width < upper_width {
            high = self.push(
                upper_width,
                NodePayload::ZeroExt {
                    arg: high,
                    new_bit_count: upper_width,
                },
            );
        }
        // For addition, low + phase overflows exactly when low > !phase.
        // For subtraction, phase - low borrows exactly when low > phase.
        let phase = bias.width_slice(0, k as i64);
        let boundary = self.literal(if model.subtract { phase } else { phase.not() });
        let carry = self.push(1, NodePayload::Binop(Binop::Ugt, low, boundary));
        let constant = self.literal(bias.width_slice(k as i64, upper_width as i64));
        let terms = [
            (constant, false),
            (high, model.subtract),
            (carry, model.subtract),
        ]
        .into_iter()
        .map(|(operand, negated)| ExtNaryAddTerm {
            operand,
            signed: false,
            negated,
        })
        .collect();
        let mut value = self.push(upper_width, NodePayload::ExtNaryAdd { terms, arch: None });
        if model.shift != 0 {
            let low = self.literal(residue);
            value = self.push(w, NodePayload::Nary(NaryOp::Concat, vec![value, low]));
        }
        if model.exceptions.is_empty() {
            return value;
        }
        // Exceptional addresses are disjoint. Share each replacement word once.
        let mut groups = HashMap::<&IrBits, Vec<NodeRef>>::new();
        for (i, replacement) in &model.exceptions {
            let index = self.literal(IrBits::make_ubits(n, *i as u64).unwrap());
            let is_index = self.push(1, NodePayload::Binop(Binop::Eq, addr, index));
            groups.entry(replacement).or_default().push(is_index);
        }
        let mut groups: Vec<_> = groups.into_iter().collect();
        groups.sort_by(|(x, _), (y, _)| x.limbs().iter().rev().cmp(y.limbs().iter().rev()));
        let zero = self.literal(IrBits::zero(w));
        let mut flags = Vec::new();
        let mut overrides = Vec::new();
        for (replacement, matches) in groups {
            let flag = self.or(1, matches);
            flags.push(flag);
            if !replacement.is_zero() {
                let word = self.literal(replacement.clone());
                overrides.push(self.push(
                    w,
                    NodePayload::Sel {
                        selector: flag,
                        cases: vec![zero, word],
                        default: None,
                    },
                ));
            }
        }
        let any = self.or(1, flags);
        let normal = self.push(
            w,
            NodePayload::Sel {
                selector: any,
                cases: vec![value, zero],
                default: None,
            },
        );
        overrides.insert(0, normal);
        self.or(w, overrides)
    }

    fn or(&mut self, width: usize, args: Vec<NodeRef>) -> NodeRef {
        if args.len() == 1 {
            args[0]
        } else {
            self.push(width, NodePayload::Nary(NaryOp::Or, args))
        }
    }

    fn literal(&mut self, bits: IrBits) -> NodeRef {
        if let Some(reference) = self.literals.get(&bits) {
            return *reference;
        }
        let node = self.push(
            bits.get_bit_count(),
            NodePayload::Literal(IrValue::Bits(bits.clone())),
        );
        self.literals.insert(bits, node);
        node
    }

    fn push(&mut self, width: usize, payload: NodePayload) -> NodeRef {
        let nr = NodeRef {
            index: self.first_helper + self.helpers.len(),
        };
        self.helpers.push((width, payload));
        nr
    }
}

struct PageModel {
    base: Vec<IrBits>,
    shift: usize,
    subtract: bool,
    exceptions: Vec<(usize, IrBits)>,
    encoded_bits: usize,
}

fn find_model(words: &[&IrBits]) -> Option<PageModel> {
    let width = words[0].get_bit_count();
    let size = words.len();
    let n = size.ilog2() as usize;
    let original_classes = words.iter().copied().collect::<HashSet<_>>().len();
    if original_classes < 2 {
        return None;
    }
    let ordinary_bits = width
        .saturating_mul(original_classes)
        .saturating_add(size.saturating_mul(ceil_log2(original_classes)));
    let mut best: Option<PageModel> = None;
    // A full verification per address bit bounds the search by O(size *
    // log(size)) word operations, even for a table containing many unrelated
    // short runs.
    let mut verifications_left = n;
    for k in 0..n {
        let lanes = 1usize << k;
        let mut checked = HashSet::new();
        let mut prev_step = None;
        for (p, (a, b)) in words
            .chunks_exact(lanes)
            .zip(words[lanes..].chunks_exact(lanes))
            .enumerate()
        {
            let step = b[0].sub(a[0]);
            let signed_step = one_hot(&step)
                .map(|s| (s, false))
                .or_else(|| one_hot(&step.negate()).map(|s| (s, true)));
            if signed_step.is_none() || !a.iter().zip(b).skip(1).all(|(x, y)| x.add(&step) == **y) {
                prev_step = None;
                continue;
            }
            let Some((shift, subtract)) = signed_step else {
                unreachable!()
            };
            if prev_step.replace((shift, subtract)) != signed_step {
                continue;
            }
            // Consecutive equal page differences give three points on this
            // line. Adjacent equal differences necessarily have the same
            // intercept.
            let page_offset = IrBits::from_lsb_fn(width, |bit| {
                bit >= shift && bit - shift < usize::BITS as usize && (p >> (bit - shift)) & 1 != 0
            });
            let base: Vec<_> = a
                .iter()
                .map(|v| {
                    if subtract {
                        v.add(&page_offset)
                    } else {
                        v.sub(&page_offset)
                    }
                })
                .collect();
            if !checked.insert((shift, subtract, base.clone())) {
                continue;
            }
            let mut expected = base.clone();
            let mut exceptions = Vec::new();
            for (i, &actual) in words.iter().enumerate() {
                let v = &mut expected[i & (lanes - 1)];
                if v != actual {
                    exceptions.push((i, actual.clone()));
                }
                *v = v.add(&step);
            }
            let bases = base.iter().collect::<HashSet<_>>().len();
            let replacements = exceptions
                .iter()
                .map(|(_, v)| v)
                .collect::<HashSet<_>>()
                .len();
            let encoded_bits = width
                .saturating_mul(bases + replacements)
                .saturating_add(lanes.saturating_mul(ceil_log2(bases)))
                .saturating_add(exceptions.len().saturating_mul(n + ceil_log2(replacements)))
                .saturating_add((1 + n.ilog2() as usize) + (1 + width.ilog2() as usize) + 1);
            let model = PageModel {
                base,
                shift,
                subtract,
                exceptions,
                encoded_bits,
            };
            // Each exceptional row needs a full address comparison, even for
            // a masked result. Only model a pattern that describes a majority
            // of the table directly.
            if model.exceptions.len() < size / 2
                && encoded_bits < ordinary_bits
                && best
                    .as_ref()
                    .is_none_or(|old| model_order(&model, old).is_lt())
            {
                best = Some(model);
            }
            verifications_left -= 1;
            if verifications_left == 0 {
                return best;
            }
        }
    }
    best
}

fn model_order(a: &PageModel, b: &PageModel) -> std::cmp::Ordering {
    let key = |p: &PageModel| {
        (
            p.encoded_bits,
            p.exceptions.len(),
            p.base.len(),
            p.shift,
            !p.subtract,
        )
    };
    key(a).cmp(&key(b)).then_with(|| {
        a.base
            .iter()
            .zip(&b.base)
            .map(|(x, y)| x.limbs().iter().rev().cmp(y.limbs().iter().rev()))
            .find(|ord| !ord.is_eq())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn one_hot(bits: &IrBits) -> Option<usize> {
    let mut limbs = bits
        .limbs()
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, x)| *x != 0);
    let (i, x) = limbs.next()?;
    (x.is_power_of_two() && limbs.next().is_none()).then(|| 64 * i + x.trailing_zeros() as usize)
}
