// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir::{
    Binop, Fn, FunctionType, NaryOp, Node, NodePayload as P, NodeRef, Package, PackageMember, Type,
    Unop,
};
use xlsynth_pir::ir_inline::package_with_reachable_callees;
use xlsynth_pir::ir_utils::{compact_and_toposort_in_place, external_function_references};
use xlsynth_pir::ir_verify::verify_package;
use xlsynth_pir::{IrFormatPreference, IrValue};

use crate::support::{IR_FUNCTION_SUPPORT, write_file};
use crate::{LeanError, LeanModuleName};

const MODULE_HEADER: &str = "-- SPDX-License-Identifier: Apache-2.0\nimport XlsynthLean.IR\n\n";

/// An executable word-level Lean module and the XLS signature of its public
/// `eval`.
#[derive(Debug)]
pub struct LeanFunction {
    module_name: LeanModuleName,
    signature: FunctionType,
    source: String,
}

impl LeanFunction {
    pub fn lean_source(&self) -> &str {
        &self.source
    }

    /// Declarations that can be nested in a batch driver after its imports.
    pub(crate) fn declarations(&self) -> &str {
        &self.source[MODULE_HEADER.len()..]
    }

    pub fn module_name(&self) -> &LeanModuleName {
        &self.module_name
    }

    pub fn signature(&self) -> &FunctionType {
        &self.signature
    }

    /// Writes the importable module, its Lean support source, and the toolchain
    /// pin.
    pub fn write_to(&self, root: &Path) -> Result<PathBuf, LeanError> {
        IR_FUNCTION_SUPPORT.write_to(root)?;
        self.write_module(root)
    }

    pub(crate) fn write_module(&self, root: &Path) -> Result<PathBuf, LeanError> {
        let path = self.module_name.source_path();
        write_file(root, &path, self.source.as_bytes())?;
        Ok(root.join(path))
    }
}

/// Converts a selected pure function and its callees to executable Lean
/// definitions.
pub fn export_ir_function(
    package: &Package,
    top: Option<&str>,
    module_name: &LeanModuleName,
) -> Result<LeanFunction, LeanError> {
    let entry = match top {
        Some(name) => package.get_named_or_top_or_sole_fn(name),
        None => package.get_top_or_sole_fn(),
    }
    .map_err(LeanError::InvalidIr)?;
    let mut reachable =
        package_with_reachable_callees(package, entry).map_err(LeanError::InvalidIr)?;
    for member in &mut reachable.members {
        if let PackageMember::Function(f) = member {
            for node in &f.nodes {
                check_supported(f, node)?;
            }
            compact_and_toposort_in_place(f).map_err(LeanError::InvalidIr)?;
        }
    }
    verify_package(&reachable).map_err(|e| LeanError::InvalidIr(e.to_string()))?;
    for member in &mut reachable.members {
        if let PackageMember::Function(f) = member
            && f.nodes.iter().any(|n| n.payload.is_extension_op())
        {
            desugar_extensions_in_fn(f).map_err(|e| LeanError::InvalidIr(e.to_string()))?;
        }
    }

    // Emit callees in postorder. Package verification has ruled out recursion.
    let mut work = vec![(entry.name.clone(), false)];
    let mut emitted = HashSet::new();
    let ns = module_name.lean_identifier();
    let mut source = format!(
        "{MODULE_HEADER}namespace {ns}\nopen XlsynthLean.IR\nset_option linter.unusedVariables false\n\n"
    );
    let functions: HashMap<_, _> = reachable
        .members
        .iter()
        .filter_map(|member| match member {
            PackageMember::Function(f) => Some((f.name.as_str(), f)),
            PackageMember::Block(_) => None,
        })
        .collect();
    // Typeclass search needs an instance per product constructor, field, and
    // terminal Unit.
    let instance_size = functions
        .values()
        .flat_map(|f| &f.nodes)
        .map(|n| packed_instance_size(&n.ty))
        .max()
        .unwrap_or(1);
    // Each let nests the remainder of the body. The default elaborator
    // recursion limit is 512, which even a modest word-level data path can
    // exceed.
    let depth = functions
        .values()
        .map(|f| f.nodes.len())
        .max()
        .unwrap_or(0)
        .max(instance_size);
    if depth >= 256 {
        writeln!(
            source,
            "set_option maxRecDepth {}\n",
            depth.saturating_mul(4).saturating_add(512)
        )
        .unwrap();
    }
    if instance_size >= 128 {
        writeln!(
            source,
            "set_option synthInstance.maxSize {}\n",
            instance_size.saturating_mul(2)
        )
        .unwrap();
    }
    while let Some((name, ready)) = work.pop() {
        if emitted.contains(&name) {
            continue;
        }
        let f = functions[name.as_str()];
        if ready {
            emit_function(&mut source, f)?;
            emitted.insert(name);
        } else {
            work.push((name, true));
            work.extend(
                external_function_references(f)
                    .into_iter()
                    .rev()
                    .map(|n| (n, false)),
            );
        }
    }
    let signature = entry.get_type();
    let mut input_width = 0usize;
    let mut call = "eval".to_owned();
    for ty in &signature.param_types {
        write!(call, " (Packed.unpack (inputs.toNat >>> {input_width}))").unwrap();
        input_width = input_width
            .checked_add(width(ty)?)
            .ok_or_else(|| LeanError::InvalidIr("combined input width overflow".into()))?;
    }
    let result_width = width(&signature.return_type)?;
    writeln!(source, "abbrev eval := {}\n", function_name(&entry.name)).unwrap();
    writeln!(source, "def evalPacked (inputs : BitVec {input_width}) : BitVec {result_width} :=\n  BitVec.ofNat _ (Packed.pack ({call}))\n\nend {ns}").unwrap();
    Ok(LeanFunction {
        module_name: module_name.clone(),
        signature,
        source,
    })
}

fn width(ty: &Type) -> Result<usize, LeanError> {
    ty.checked_bit_count()
        .ok_or_else(|| LeanError::InvalidIr(format!("bit width overflows for {ty}")))
}

fn packed_instance_size(ty: &Type) -> usize {
    match ty {
        Type::Token | Type::Bits(_) => 1,
        Type::Tuple(fields) => {
            1 + fields
                .iter()
                .map(|t| 1 + packed_instance_size(t))
                .sum::<usize>()
        }
        Type::Array(a) => 1 + packed_instance_size(&a.element_type),
    }
}

/// Uses an injective ASCII spelling even for names that contain Lean syntax.
fn function_name(name: &str) -> String {
    let mut result = "ir_".to_owned();
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() {
            result.push(char::from(b));
        } else {
            write!(result, "_{b:02x}").unwrap();
        }
    }
    result
}

fn has_token(ty: &Type) -> bool {
    match ty {
        Type::Token => true,
        Type::Bits(_) => false,
        Type::Tuple(fields) => fields.iter().any(|f| has_token(f)),
        Type::Array(a) => has_token(&a.element_type),
    }
}

/// Check even unused nodes so an unsupported observation cannot silently
/// disappear.
fn check_supported(f: &Fn, node: &Node) -> Result<(), LeanError> {
    let reason = match &node.payload {
        P::Binop(Binop::Smulp | Binop::Umulp, ..) => "partial-product multiply",
        P::CountedFor { .. } => "loop",
        P::AfterAll(_) | P::Assert { .. } | P::Trace { .. } | P::Cover { .. } => "token or event",
        P::InputPort { .. }
        | P::OutputPort { .. }
        | P::InstantiationInput { .. }
        | P::InstantiationOutput { .. }
        | P::RegisterRead { .. }
        | P::RegisterWrite { .. } => "block operation",
        _ if has_token(&node.ty) => "token-valued type",
        _ => return Ok(()),
    };
    Err(LeanError::UnsupportedIr {
        function: f.name.clone(),
        op: node.payload.get_operator().to_owned(),
        id: node.text_id,
        reason,
    })
}

fn lean_type(ty: &Type) -> String {
    match ty {
        Type::Bits(w) => format!("BitVec {w}"),
        Type::Array(a) => format!(
            "Vector ({}) {}",
            lean_type(&a.element_type),
            a.element_count
        ),
        Type::Tuple(fields) if fields.is_empty() => "Unit".to_string(),
        Type::Tuple(fields) => format!(
            "({} × Unit)",
            fields
                .iter()
                .map(|t| lean_type(t))
                .collect::<Vec<_>>()
                .join(" × ")
        ),
        Type::Token => unreachable!("token types were rejected before emission"),
    }
}

/// Preserve operand order while keeping the Lean expression depth logarithmic.
fn balanced_infix(operands: &[String], separator: &str) -> String {
    fn emit(out: &mut String, operands: &[String], separator: &str) {
        match operands {
            [] => {}
            [operand] => out.push_str(operand),
            _ => {
                let (left, right) = operands.split_at(operands.len() / 2);
                out.push('(');
                emit(out, left, separator);
                out.push_str(separator);
                emit(out, right, separator);
                out.push(')');
            }
        }
    }

    let mut expression = String::new();
    emit(&mut expression, operands, separator);
    expression
}

fn tuple(fields: &[String]) -> String {
    if fields.is_empty() {
        "()".into()
    } else {
        format!("({}, ())", fields.join(", "))
    }
}

/// `Array.toVector` avoids unfolding computed elements in a `#v` size proof.
/// Explicit append lengths keep size checks local to each leaf.
fn vector_literal(fields: &[String]) -> String {
    if fields.len() <= 64 {
        return format!("#[{}].toVector", fields.join(", "));
    }
    let (left, right) = fields.split_at(fields.len() / 2);
    format!(
        "(@Vector.append _ {} {} {} {})",
        left.len(),
        right.len(),
        vector_literal(left),
        vector_literal(right)
    )
}

fn literal(value: &IrValue) -> Result<String, LeanError> {
    Ok(match value.type_() {
        Type::Bits(w) => format!(
            "BitVec.ofNat {w} 0x{}",
            value.to_string_fmt_no_prefix(IrFormatPreference::PlainHex)
        ),
        Type::Tuple(_) | Type::Array(_) => {
            let fields = value
                .as_elements()?
                .iter()
                .map(literal)
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(value.type_(), Type::Tuple(_)) {
                tuple(&fields)
            } else {
                vector_literal(&fields)
            }
        }
        Type::Token => unreachable!("token literals were rejected before emission"),
    })
}

/// The declared result width is significant for ordinary multiply and shifts.
fn binary(op: Binop, a: &str, b: &str, w: usize) -> String {
    match op {
        Binop::Add => format!("{a} + {b}"),
        Binop::Sub => format!("{a} - {b}"),
        Binop::Umul => format!("{a}.setWidth {w} * {b}.setWidth {w}"),
        Binop::Smul => format!("{a}.signExtend {w} * {b}.signExtend {w}"),
        Binop::Udiv => format!("BitVec.smtUDiv {a} {b}"),
        Binop::Sdiv => format!("sdiv {a} {b}"),
        Binop::Umod => format!("if {b} == 0 then 0 else {a} % {b}"),
        Binop::Smod => format!("if {b} == 0 then 0 else {a}.srem {b}"),
        Binop::Shll => format!("{a} <<< (min {b}.toNat {w})"),
        Binop::Shrl => format!("{a} >>> (min {b}.toNat {w})"),
        Binop::Shra => format!("{a}.sshiftRight (min {b}.toNat {w})"),
        Binop::Eq => format!("BitVec.ofBool ({a} == {b})"),
        Binop::Ne => format!("BitVec.ofBool ({a} != {b})"),
        Binop::Ult => format!("BitVec.ofBool ({a}.ult {b})"),
        Binop::Ule => format!("BitVec.ofBool ({a}.ule {b})"),
        Binop::Ugt => format!("BitVec.ofBool ({b}.ult {a})"),
        Binop::Uge => format!("BitVec.ofBool ({b}.ule {a})"),
        Binop::Slt => format!("BitVec.ofBool ({a}.slt {b})"),
        Binop::Sle => format!("BitVec.ofBool ({a}.sle {b})"),
        Binop::Sgt => format!("BitVec.ofBool ({b}.slt {a})"),
        Binop::Sge => format!("BitVec.ofBool ({b}.sle {a})"),
        Binop::Gate => format!("if {a} == 1 then {b} else default"),
        Binop::Smulp | Binop::Umulp => unreachable!("unsupported partial product"),
    }
}

/// Emits one typed binding per IR node, retaining shared operands and calls.
fn emit_function(out: &mut String, f: &Fn) -> Result<(), LeanError> {
    let name = |r: NodeRef| format!("n{}", f.get_node(r).text_id);
    let names = |rs: &[NodeRef]| rs.iter().copied().map(name).collect::<Vec<_>>();
    write!(out, "def {}", function_name(&f.name)).unwrap();
    for p in f.param_nodes() {
        write!(out, " (n{} : {})", p.text_id, lean_type(&p.ty)).unwrap();
    }
    writeln!(out, " : {} :=", lean_type(&f.ret_ty)).unwrap();
    for n in &f.nodes {
        let e = match &n.payload {
            P::Nil | P::Param => continue,
            P::Literal(v) => literal(v)?,
            P::Tuple(rs) => tuple(&names(rs)),
            P::Array(rs) => vector_literal(&names(rs)),
            P::ArrayConcat(rs) => balanced_infix(&names(rs), " ++ "),
            P::TupleIndex { tuple, index } => format!("{}{}.1", name(*tuple), ".2".repeat(*index)),
            P::ArrayIndex { array, indices, .. } => indices.iter().fold(name(*array), |base, i| {
                format!("(arrayIndex {base} {}.toNat)", name(*i))
            }),
            P::ArrayUpdate {
                array,
                value,
                indices,
                ..
            } => {
                let mut expr = name(*value);
                for (depth, idx) in indices.iter().enumerate().rev() {
                    let base = if depth == 0 {
                        name(*array)
                    } else {
                        format!("v{}", depth - 1)
                    };
                    let binder = if depth + 1 == indices.len() {
                        "_".to_owned()
                    } else {
                        format!("v{depth}")
                    };
                    expr = format!(
                        "arrayModify {base} {}.toNat (fun {binder} => {expr})",
                        name(*idx)
                    );
                }
                expr
            }
            P::ArraySlice {
                array,
                start,
                width,
            } => format!("arraySlice {width} {} {}.toNat", name(*array), name(*start)),
            P::Binop(op, a, b) => binary(*op, &name(*a), &name(*b), width(&n.ty)?),
            P::Unop(op, a) => {
                let a = name(*a);
                match op {
                    Unop::Identity => a,
                    Unop::Neg => format!("-{a}"),
                    Unop::Not => format!("~~~{a}"),
                    Unop::Reverse => format!("{a}.reverse"),
                    Unop::OrReduce => format!("BitVec.ofBool ({a} != 0)"),
                    Unop::AndReduce => format!("BitVec.ofBool ({a} == BitVec.allOnes _)"),
                    Unop::XorReduce => format!("{a}.cpop.setWidth 1"),
                }
            }
            P::Nary(NaryOp::Concat, rs) if rs.is_empty() => "0".to_owned(),
            P::Nary(op, rs) => {
                let sep = match op {
                    NaryOp::And | NaryOp::Nand => " &&& ",
                    NaryOp::Or | NaryOp::Nor => " ||| ",
                    NaryOp::Xor => " ^^^ ",
                    NaryOp::Concat => " ++ ",
                };
                let args = balanced_infix(&names(rs), sep);
                if matches!(op, NaryOp::Nand | NaryOp::Nor) {
                    format!("~~~({args})")
                } else {
                    args
                }
            }
            P::SignExt { arg, new_bit_count } => {
                format!("{}.signExtend {new_bit_count}", name(*arg))
            }
            P::ZeroExt { arg, new_bit_count } => format!("{}.setWidth {new_bit_count}", name(*arg)),
            P::BitSlice { arg, start, width } => {
                format!("BitVec.extractLsb' {start} {width} {}", name(*arg))
            }
            P::DynamicBitSlice {
                arg,
                start,
                width: w,
            } => format!(
                "BitVec.extractLsb' (min {}.toNat {}) {w} {}",
                name(*start),
                width(f.get_node_ty(*arg))?,
                name(*arg)
            ),
            P::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => format!(
                "bitSliceUpdate {} {} {}",
                name(*arg),
                name(*start),
                name(*update_value)
            ),
            P::Decode { arg, width } => {
                format!("BitVec.twoPow {width} (min {}.toNat {width})", name(*arg))
            }
            P::Encode { arg } => format!("encode {} {}", width(&n.ty)?, name(*arg)),
            P::OneHot { arg, lsb_prio } => format!("oneHot {lsb_prio} {}", name(*arg)),
            P::OneHotSel { selector, cases } => format!(
                "oneHotSel {} #[{}]",
                name(*selector),
                names(cases).join(", ")
            ),
            P::PrioritySel {
                selector,
                cases,
                default,
            } => format!(
                "prioritySel {} #[{}] {}",
                name(*selector),
                names(cases).join(", "),
                name(default.expect("verified priority default"))
            ),
            P::Sel {
                selector,
                cases,
                default,
            } => {
                let fallback = default
                    .or_else(|| cases.last().copied())
                    .expect("verified selector has an arm");
                format!(
                    "#[{}].getD {}.toNat {}",
                    names(cases).join(", "),
                    name(*selector),
                    name(fallback)
                )
            }
            P::Map { array, to_apply } => {
                format!("Vector.map {} {}", function_name(to_apply), name(*array))
            }
            P::Invoke { to_apply, operands } => std::iter::once(function_name(to_apply))
                .chain(names(operands))
                .collect::<Vec<_>>()
                .join(" "),
            other => {
                return Err(LeanError::UnsupportedIr {
                    function: f.name.clone(),
                    op: other.get_operator().to_owned(),
                    id: n.text_id,
                    reason: "no word-level translation",
                });
            }
        };
        writeln!(out, "  let n{} : {} := {e}", n.text_id, lean_type(&n.ty)).unwrap();
    }
    writeln!(
        out,
        "  {}\n",
        name(f.ret_node_ref.expect("verified return"))
    )
    .unwrap();
    Ok(())
}
