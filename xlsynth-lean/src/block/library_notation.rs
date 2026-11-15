// SPDX-License-Identifier: Apache-2.0

//! File-local names for Lean library declarations used by block models.
//!
//! A block may have an input named `Vector` or a field named `Repr`. Generated
//! references such as `L.Vector` still resolve to the root library. Each file
//! gets its own names, including the modules that contain proof scripts.

use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Library terms that can be referenced through local notation.
///
/// Keep this list explicit because the scan also sees source identifiers,
/// comments and assertion messages, any of which can mention `L.something`.
const TERM_NAMES: &[&str] = &[
    "And",
    "Array.getD",
    "Array.toVector",
    "BitVec",
    "BitVec.add",
    "BitVec.allOnes",
    "BitVec.and",
    "BitVec.append",
    "BitVec.cpop",
    "BitVec.eq_of_toNat_eq",
    "BitVec.extractLsb'",
    "BitVec.getLsbD",
    "BitVec.isLt",
    "BitVec.mul",
    "BitVec.neg",
    "BitVec.not",
    "BitVec.ofBool",
    "BitVec.ofNat",
    "BitVec.or",
    "BitVec.reverse",
    "BitVec.setWidth",
    "BitVec.shiftLeft",
    "BitVec.signExtend",
    "BitVec.sle",
    "BitVec.slt",
    "BitVec.smtUDiv",
    "BitVec.srem",
    "BitVec.sshiftRight",
    "BitVec.sub",
    "BitVec.toNat",
    "BitVec.twoPow",
    "BitVec.ule",
    "BitVec.ult",
    "BitVec.umod",
    "BitVec.ushiftRight",
    "BitVec.xor",
    "Bool",
    "Decidable",
    "Eq.symm",
    "Eq.trans",
    "Inhabited.default",
    "List.foldl",
    "List.range",
    "Nat",
    "Nat.fold",
    "Nat.min",
    "NeZero",
    "Or.inl",
    "Or.inr",
    "Prod",
    "True",
    "Unit",
    "Vector",
    "Vector.back",
    "Vector.getD",
    "Vector.map",
    "Vector.setIfInBounds",
    "congrArg",
];

/// Library names that must also work in positions requiring an identifier.
///
/// `deriving` needs the class name. Giving `simp` a theorem name preserves
/// its registered rewrite behavior, for example when simplifying with
/// `Vector.map_mk` on an unknown vector.
const DECLARATION_NAMES: &[&str] = &[
    "DecidableEq",
    "Function.comp_def",
    "List.map_cons",
    "List.map_nil",
    "List.map_toArray",
    "Nat.fold_zero",
    "Repr",
    "Vector.append",
    "Vector.back_eq_getElem",
    "Vector.map_append",
    "Vector.map_id'",
    "Vector.map_map",
    "Vector.map_mk",
    "Vector.map_setIfInBounds",
    "apply_ite",
    "ite_self",
];

/// Insert the aliases mentioned by the completed module after its imports.
///
/// Call once, after rendering all declarations. The scan treats leading blank
/// lines, line comments and imports as the header. A mention in later prose
/// can request an existing alias, but only listed library names can add one.
pub(super) fn add_to_module(module: &mut String) {
    let insert_at = module
        .split_inclusive('\n')
        .take_while(|line| {
            line.trim().is_empty() || line.starts_with("--") || line.starts_with("import ")
        })
        .map(str::len)
        .sum();
    let body = &module[insert_at..];
    let names: BTreeSet<_> = body
        .split(|c: char| !c.is_ascii_alphanumeric() && !matches!(c, '.' | '_' | '\''))
        .filter_map(|word| word.strip_prefix("L."))
        .collect();

    let mut prelude = String::new();
    for &name in DECLARATION_NAMES {
        if names.contains(name) {
            let (namespace, member) = name.rsplit_once('.').unwrap_or(("", name));
            let dot = if namespace.is_empty() { "" } else { "." };
            writeln!(
                prelude,
                "open _root_{dot}{namespace} renaming {member} -> L.{name}"
            )
            .unwrap();
        }
    }
    for &name in TERM_NAMES {
        if names.contains(name) {
            writeln!(prelude, "local notation \"L.{name}\" => _root_.{name}").unwrap();
        }
    }
    if !prelude.is_empty() {
        prelude.push('\n');
        module.insert_str(insert_at, &prelude);
    }
}
