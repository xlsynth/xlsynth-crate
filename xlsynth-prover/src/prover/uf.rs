// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for inferring and merging uninterpreted-function signatures.

use std::cell::RefCell;
use std::collections::HashMap;
use xlsynth_pir::ir;

use super::types::UfSignature;

/// Rejects solver symbols that cannot cross a C API or process argument
/// boundary.
pub(crate) fn validate_uf_symbol(symbol: &str) -> Result<(), String> {
    if symbol.contains('\0') {
        return Err("Uninterpreted function symbol contains a NUL byte".into());
    }
    Ok(())
}

/// Checks the implicit-token ABI and, when present, its data-only counterpart.
fn is_implicit_token_variant(function: &ir::Fn, plain: Option<&ir::Fn>) -> bool {
    #[cfg(all(test, feature = "has-bitwuzla"))]
    tests::WRAPPER_CHECKS.set(tests::WRAPPER_CHECKS.get() + 1);
    let mut params = function.param_nodes().map(|param| &param.ty);
    if params.next() != Some(&ir::Type::Token) || params.next() != Some(&ir::Type::Bits(1)) {
        return false;
    }
    let ir::Type::Tuple(fields) = &function.ret_ty else {
        return false;
    };
    let [token, result] = fields.as_slice() else {
        return false;
    };
    **token == ir::Type::Token
        && plain.is_none_or(|f| {
            f.ret_ty == **result && params.eq(f.param_nodes().map(|param| &param.ty))
        })
}

/// Bindings for one package. Cache absent functions and rejected wrappers too,
/// since a loop may dispatch the same concrete or abstract callee many times.
pub(super) struct UfBindings<'a> {
    pkg: Option<&'a ir::Package>,
    names: &'a HashMap<String, String>,
    plain: RefCell<HashMap<&'a str, Option<&'a ir::Fn>>>,
    wrappers: RefCell<HashMap<&'a str, bool>>,
}

impl<'a> UfBindings<'a> {
    pub(super) fn new(pkg: Option<&'a ir::Package>, names: &'a HashMap<String, String>) -> Self {
        Self {
            pkg,
            names,
            plain: Default::default(),
            wrappers: Default::default(),
        }
    }

    fn plain_function(&self, name: &'a str) -> Option<&'a ir::Fn> {
        *self
            .plain
            .borrow_mut()
            .entry(name)
            .or_insert_with(|| self.pkg?.get_fn(name))
    }

    fn matches_wrapper(&self, name: &'a str, wrapper: &ir::Fn) -> bool {
        *self
            .wrappers
            .borrow_mut()
            .entry(name)
            .or_insert_with(|| is_implicit_token_variant(wrapper, self.plain_function(name)))
    }

    /// An exact mapping keeps every argument, even for a prefixed name.
    pub(super) fn lookup(&self, callee: &ir::Fn) -> Option<(&str, bool)> {
        if let Some(symbol) = self.names.get(&callee.name) {
            return Some((symbol, false));
        }
        let (name, symbol) = self
            .names
            .get_key_value(callee.name.strip_prefix("__itok")?)?;
        self.matches_wrapper(name, callee).then_some((symbol, true))
    }

    /// Infers solver widths using the same bindings as call translation.
    pub(super) fn infer_signatures(&self) -> Result<HashMap<String, UfSignature>, String> {
        for symbol in self.names.values() {
            validate_uf_symbol(symbol)?;
        }
        let mut uf_sigs: HashMap<String, UfSignature> = HashMap::new();
        for (fn_name, uf_sym) in self.names {
            let (ir_fn, skip_implicit) = match self.plain_function(fn_name) {
                Some(f) => (f, false),
                None => {
                    let itok_name = format!("__itok{}", fn_name);
                    match self.pkg.and_then(|pkg| pkg.get_fn(&itok_name)) {
                        Some(f) if self.matches_wrapper(fn_name, f) => (f, true),
                        Some(_) => {
                            return Err(format!(
                                "Function '{itok_name}' lacks the implicit-token signature for '{fn_name}'"
                            ));
                        }
                        None => {
                            return Err(format!(
                                "Unknown function '{}' when inferring UF signature for symbol '{}'",
                                fn_name, uf_sym
                            ));
                        }
                    }
                }
            };

            let arg_widths = ir_fn
                .param_nodes()
                .skip(if skip_implicit { 2 } else { 0 })
                .map(|p| p.ty.bit_count())
                .collect();
            let ret_width = ir_fn.ret_ty.bit_count();
            let sig = UfSignature {
                arg_widths,
                ret_width,
            };

            if let Some(prev) = uf_sigs.get(uf_sym) {
                if prev != &sig {
                    return Err(format!(
                        "Conflicting UF signature for symbol '{}': {:?} vs {:?}",
                        uf_sym, prev, sig
                    ));
                }
            } else {
                uf_sigs.insert(uf_sym.clone(), sig);
            }
        }
        Ok(uf_sigs)
    }

    pub(super) fn merged_signatures(
        &self,
        rhs: &Self,
    ) -> Result<HashMap<String, UfSignature>, String> {
        merge_uf_signatures(self.infer_signatures()?, &rhs.infer_signatures()?)
    }
}

/// Infers UF signatures for the provided package and map from functions to UF
/// names.
pub fn infer_uf_signatures(
    pkg: &xlsynth_pir::ir::Package,
    uf_map: &HashMap<String, String>,
) -> Result<HashMap<String, UfSignature>, String> {
    UfBindings::new(Some(pkg), uf_map).infer_signatures()
}

/// Merges UF signature maps, ensuring there are no conflicts.
pub fn merge_uf_signatures(
    mut lhs: HashMap<String, UfSignature>,
    rhs: &HashMap<String, UfSignature>,
) -> Result<HashMap<String, UfSignature>, String> {
    for (name, sig) in rhs {
        if let Some(prev) = lhs.get(name) {
            if prev != sig {
                return Err(format!(
                    "Conflicting UF signature for symbol '{}': {:?} vs {:?}",
                    name, prev, sig
                ));
            }
        } else {
            lhs.insert(name.clone(), sig.clone());
        }
    }
    Ok(lhs)
}

/// Convenience helper that infers UF signatures for both sides of an
/// equivalence proof and merges them, ensuring there are no conflicts.
pub fn infer_merged_uf_signatures(
    lhs_pkg: &xlsynth_pir::ir::Package,
    lhs_map: &HashMap<String, String>,
    rhs_pkg: &xlsynth_pir::ir::Package,
    rhs_map: &HashMap<String, String>,
) -> Result<HashMap<String, UfSignature>, String> {
    UfBindings::new(Some(lhs_pkg), lhs_map)
        .merged_signatures(&UfBindings::new(Some(rhs_pkg), rhs_map))
}

#[cfg(all(test, feature = "has-bitwuzla"))]
mod tests {
    use crate::ir_equiv::{IrEquivRequest, IrModule, run_ir_equiv};
    use crate::prover::types::EquivResult;
    use std::cell::Cell;
    use std::collections::HashMap;
    use xlsynth_pir::{ir_parser::Parser, ir_verify::verify_package};

    thread_local! {
        pub(super) static WRAPPER_CHECKS: Cell<usize> = const { Cell::new(0) };
    }

    #[test]
    fn repeated_implicit_token_uf_in_counted_body() {
        let package = |result: &str| {
            let mut text = format!(
                r#"package counted_wrapper

fn __itokg(t: token id=1, act: bits[1] id=2, x: bits[1] id=3) -> (token, bits[1]) {{
  value: bits[1] = {result}(x, id=4)
  ret pair: (token, bits[1]) = tuple(t, value, id=5)
}}

fn body(i: bits[14] id=6, carry: bits[1] id=7, t: token id=8, act: bits[1] id=9, x: bits[1] id=10) -> bits[1] {{
  pair: (token, bits[1]) = invoke(t, act, x, to_apply=__itokg, id=11)
  ret value: bits[1] = tuple_index(pair, index=1, id=12)
}}

top fn main(t: token id=13, act: bits[1] id=14, x: bits[1] id=15) -> bits[1] {{
  ret value: bits[1] = counted_for(x, trip_count=10000, stride=1, body=body, invariant_args=[t, act, x], id=16)
}}
"#
            );
            // __itokg is the first member, but a lookup for g visits all 1,000.
            for i in 0..997 {
                let id = 17 + i;
                text.push_str(&format!(
                    r#"fn filler{i}() -> bits[1] {{
  ret zero: bits[1] = literal(value=0, id={id})
}}
"#
                ));
            }
            verify_package(&Parser::new(&text).parse_package().unwrap()).unwrap();
            text
        };
        let lhs = package("identity");
        let rhs = package("not");
        let uf_map = HashMap::from([("g".to_string(), "F".to_string())]);
        let request = IrEquivRequest::new(
            IrModule::new(&lhs).with_uf_map(&uf_map),
            IrModule::new(&rhs).with_uf_map(&uf_map),
        );
        WRAPPER_CHECKS.set(0);
        assert_eq!(run_ir_equiv(&request).unwrap().result, EquivResult::Proved);
        assert_eq!(WRAPPER_CHECKS.get(), 2, "one wrapper check per package");
    }
}
