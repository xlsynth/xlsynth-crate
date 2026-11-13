// SPDX-License-Identifier: Apache-2.0

//! Compare compile-time assertions in modules and pure function
//! specializations.

use std::fs;

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program, compare_program};
use crate::input_seed;

/// The declaration being checked and the calls that instantiate its body.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Module,
    Function,
    UncalledFunction,
    ExplicitParametric,
    InferredParametric,
    UncalledParametric,
    CountedLoop,
    UnrolledLoop,
}

/// Predicate shape, source placement, and values of successive specializations.
struct Sample {
    scope: Scope,
    predicate: u8,
    flags: u8,
    first: u8,
    second: u8,
    floor: u8,
}

impl Sample {
    /// Render the predicate and predict whether it passes for one binding of N.
    fn predicate(&self, n: u8) -> (String, bool) {
        let succeeds = n >= self.floor;
        let literal = self.first >= self.floor;
        let module = self.scope == Scope::Module;
        let input = if module { "VALUE" } else { "x" };
        let variable = match self.scope {
            Scope::Module | Scope::Function | Scope::UncalledFunction => "VALUE",
            _ => "N",
        };
        match self.predicate {
            0 => (literal.to_string(), literal),
            1 => (format!("u1:{}", u8::from(literal)), literal),
            2 => (u8::from(literal).to_string(), literal),
            3 => (format!("{variable} >= FLOOR"), succeeds),
            4 => (format!("meets({variable})"), succeeds),
            5 => (
                format!("{{ let pair = (FLOOR, {variable}); pair.1 >= pair.0 }}"),
                succeeds,
            ),
            6 => {
                let ty = ["s1:-1", "u8:1", "(true, true)", "[true]"];
                (ty[usize::from(self.flags >> 5) % ty.len()].into(), false)
            }
            7 if module => ("LATER".into(), false),
            7 if self.scope == Scope::UncalledParametric => ("MISSING".into(), false),
            7 => ("x == x".into(), false),
            8 if module => (format!("if {literal} {{ true }} else {{ false }}"), literal),
            // DSLX constexpr if follows the selected arm, even if the other arm
            // references a runtime input.
            8 => (
                format!("if {literal} {{ true }} else {{ x == x }}"),
                literal,
            ),
            9 if self.flags & 32 != 0 => (
                format!(
                    "match {{ let _ = if true {{ true }} else {{ {input} == {input} }}; true }} {{ true => true, _ => false }}"
                ),
                module,
            ),
            9 if module => (
                format!("match {variable} {{ u32:{n} => meets({variable}), _ => false }}"),
                succeeds,
            ),
            // Pinned DSLX requires constexpr operands in every match arm.
            9 => ("match true { true => true, _ => x == x }".into(), false),
            10 if module => (format!("meets({variable}) || false"), succeeds),
            10 => ("true || (x == x)".into(), false),
            11 if module => ("ignores_arg(VALUE)".into(), true),
            11 => ("ignores_arg(x as u32)".into(), false),
            12 => ("2".into(), false),
            13 => (format!("!(FLOOR > {variable})"), succeeds),
            14 if self.flags & 32 != 0 => (
                format!("{{ let k = {{ {input}; {variable} }}; verified<k>({variable}) }}"),
                module && succeeds,
            ),
            14 => (format!("verified<{variable}>({variable})"), succeeds),
            15..=17 => ("true".into(), false),
            18 => (format!("{{ {input}; true }}"), module),
            19 => (format!("{{ let _ = {input}; true }}"), true),
            20 if self.flags & 32 != 0 => (
                format!(
                    "{{ let pair = (true, u8:1); let (t, _) = (pair, {input}); let (a, _) = t; a }}"
                ),
                module,
            ),
            20 => (format!("{{ let (a, _) = (true, {input}); a }}"), module),
            21 => (format!("{{ let t = ({input}, true); t.1 }}"), module),
            22 => (format!("if true {{ true }} else {{ {input}; true }}"), true),
            _ => {
                let read = format!("read_at({variable} % u32:3)");
                let predicate = if self.flags & 64 == 0 {
                    read
                } else {
                    format!("ceillog2({read} as u32) == u32:0")
                };
                (predicate, n % 3 < 2)
            }
        }
    }

    /// Render valid assertions or syntax errors that precede specialization.
    fn assertion(&self, pred: &str) -> String {
        match self.predicate {
            15 => format!("const_assert!({pred}, true);"),
            16 => format!("const_assert!({pred})"),
            17 if self.scope == Scope::Module => format!("const ATTEMPT = const_assert!({pred});"),
            17 => format!("let attempt = const_assert!({pred});"),
            _ if self.scope == Scope::Module || self.flags & 4 == 0 => {
                format!("const_assert!({pred});")
            }
            // v0.62.0 infers bits[0] for an unannotated let bound to 0 and
            // accepts it as a true assertion. Supply the predicate's bool type.
            2 => format!("let condition: bool = {pred}; const_assert!(condition);"),
            _ => format!("let condition = {pred}; const_assert!(condition);"),
        }
    }

    /// Place the assertion in a statement sequence, including arms that will
    /// never execute, and return a runtime value on every path.
    fn function_body(&self, pred: &str, result: &str) -> String {
        let assertion = if self.predicate == 20 && self.flags & 4 != 0 && self.flags & 32 == 0 {
            "let (a, _) = (true, x); const_assert!(a);".to_owned()
        } else {
            self.assertion(pred)
        };
        match self.flags & 3 {
            0 => format!("{assertion} {result}"),
            1 => format!("if false {{ {assertion} !({result}) }} else {{ {result} }}"),
            2 => format!("match false {{ true => {{ {assertion} !({result}) }}, _ => {result} }}"),
            _ => format!(
                "if x == x {{ if true {{ {result} }} else {{ {assertion} x as u8 }} }} else {{ u8:7 }}"
            ),
        }
    }

    /// Check ordinary loop bodies once and structural unrolls for each bound
    /// iteration. A runtime carry remains nonconstant in both forms.
    fn loop_body(&self) -> (String, bool) {
        let unrolled = self.scope == Scope::UnrolledLoop;
        let trips = self.second % 4;
        let limit = self.first % 4;
        let (pred, passes) = match self.predicate % 5 {
            0 => ("true".to_owned(), true),
            1 => ("false".to_owned(), false),
            2 => (format!("i < u32:{limit}"), unrolled && trips <= limit),
            3 => ("carry == carry".to_owned(), false),
            _ => (
                "meets(i + VALUE)".to_owned(),
                unrolled && self.first >= self.floor,
            ),
        };
        let operator = if unrolled { "unroll_for!" } else { "for" };
        let assertion = self.assertion(&pred);
        let body = format!(
            "{operator} (i, carry) in u32:0..u32:{trips} {{ {assertion} carry + (i as u8) }}(x)"
        );
        (body, passes || (unrolled && trips == 0))
    }

    /// Construct the common declarations and establish the expected outcome
    /// independently of either frontend.
    fn declarations(&self, imported_floor: bool) -> (String, bool) {
        let floor = if imported_floor {
            "settings::FLOOR".to_owned()
        } else {
            format!("u32:{}", self.floor)
        };
        let mut declarations = format!(
            r#"const FLOOR = {floor};
const VALUE = u32:{first};
fn meets(value: u32) -> bool {{ value >= FLOOR }}
fn ignores_arg(value: u32) -> bool {{ true }}
"#,
            first = self.first
        );
        if self.predicate == 14 && !matches!(self.scope, Scope::CountedLoop | Scope::UnrolledLoop) {
            declarations.push_str(
                "fn verified<K: u32>(v: u32) -> bool { const_assert!(K >= FLOOR); v == K }\n",
            );
        }
        if self.predicate == 23 {
            let body = if self.flags & 32 == 0 {
                "[u8:1, u8:2][i]; true"
            } else {
                // The last iteration succeeds independently of the first.
                "unroll_for! (j, a) in u32:0..u32:2 { let _ = [u8:1, u8:2][if j == u32:0 { i } else { u32:0 }]; true }(true)"
            };
            declarations.push_str(&format!("fn read_at(i: u32) -> bool {{ {body} }}\n"));
        }
        let (pred, first_passes) = self.predicate(self.first);
        let (call, valid) = match self.scope {
            Scope::Module => {
                let assertion = self.assertion(&pred);
                declarations.push_str(&format!("{assertion}\nconst LATER = true;\n"));
                ("x ^ (VALUE as u8)".to_owned(), first_passes)
            }
            Scope::Function | Scope::UncalledFunction => {
                let body = self.function_body(&pred, "x + (FLOOR as u8)");
                declarations.push_str(&format!("fn guard(x: u8) -> u8 {{ {body} }}\n"));
                let call = if self.scope == Scope::Function {
                    "guard(x)"
                } else {
                    "x"
                };
                (call.to_owned(), first_passes)
            }
            Scope::ExplicitParametric | Scope::InferredParametric | Scope::UncalledParametric => {
                let inferred = self.scope == Scope::InferredParametric;
                let ty = if inferred { "uN[N]" } else { "u8" };
                let body = self.function_body(&pred, "(x as u8) ^ (N as u8)");
                declarations.push_str(&format!("fn guard<N: u32>(x: {ty}) -> u8 {{ {body} }}\n"));
                if self.scope == Scope::UncalledParametric {
                    // Only lexical lookup is required before specialization.
                    ("x".to_owned(), self.predicate != 7)
                } else {
                    let call = |n| {
                        if inferred {
                            format!("guard(x as uN[{n}])")
                        } else {
                            format!("guard<{n}>(x)")
                        }
                    };
                    let second_passes = self.predicate(self.second).1;
                    (
                        format!("{} + {}", call(self.first), call(self.second)),
                        first_passes && second_passes,
                    )
                }
            }
            Scope::CountedLoop | Scope::UnrolledLoop => self.loop_body(),
        };
        declarations.push_str(&format!("pub fn apply(x: u8) -> u8 {{ {call} }}"));
        (declarations, valid && !(15..=17).contains(&self.predicate))
    }
}

/// Compare source acceptance, compiled IR execution, and concrete DSLX calls
/// for the same assertions and runtime inputs.
pub fn check_const_assert(data: &[u8]) -> Result<(), String> {
    let [scope, predicate, flags, first, second, floor] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let sample = Sample {
        scope: match scope % 8 {
            0 => Scope::Module,
            1 => Scope::Function,
            2 => Scope::UncalledFunction,
            3 => Scope::ExplicitParametric,
            4 => Scope::InferredParametric,
            5 => Scope::UncalledParametric,
            6 => Scope::CountedLoop,
            _ => Scope::UnrolledLoop,
        },
        predicate: predicate % 24,
        flags,
        first: first % 129 + 1,
        second: second % 129 + 1,
        floor: floor % 130,
    };
    let imported_floor = flags & 8 != 0;
    let imported_guard = flags & 16 != 0;
    let (declarations, valid) = sample.declarations(imported_floor);
    let mut program = FunctionProgram {
        functions: declarations,
        params: "x: u8".into(),
        result: "u8".into(),
        body: "apply(x)".into(),
        valid: Some(valid),
        // v0.62.0 cannot lower some let-bound ceillog2 invocations to IR.
        // Keep comparing their typechecking and concrete execution to BX IR.
        dslx_ir: !(sample.predicate == 23
            && flags & 64 != 0
            && flags & 4 != 0
            && matches!(
                sample.scope,
                Scope::Function
                    | Scope::UncalledFunction
                    | Scope::ExplicitParametric
                    | Scope::InferredParametric
                    | Scope::UncalledParametric
            )),
        dynamic_index: None,
    };
    let seed = input_seed(data);
    if !imported_floor && !imported_guard {
        return compare_program(program, seed, DslxExecution::ConcreteConstants, vec![]);
    }
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let mut bx_decls = program.functions.clone();
    if imported_floor {
        let settings = format!("pub const FLOOR = u32:{};", sample.floor);
        fs::write(dir.path().join("settings.x"), settings).map_err(|e| e.to_string())?;
        bx_decls.insert_str(0, "use settings;\n");
        program.functions.insert_str(0, "import settings;\n");
    }
    let library = program.functions.clone();
    if imported_guard {
        fs::write(dir.path().join("bx_guards.bx"), &bx_decls).map_err(|e| e.to_string())?;
        fs::write(dir.path().join("guards.x"), &program.functions).map_err(|e| e.to_string())?;
        bx_decls = "use bx_guards::apply;".into();
        program.functions = "import guards;".into();
        program.body = "guards::apply(x)".into();
    }
    let bx = format!("{bx_decls}\n{}", bx_source("x: u8", "u8", "apply(x)"));
    compare_bx_program(
        &bx,
        program,
        seed,
        DslxExecution::ConcreteConstants,
        vec![],
        &[dir.path()],
    )
    .map_err(|e| format!("{e}\nassertion declarations:\n{library}"))
}

#[cfg(test)]
mod tests {
    use super::check_const_assert;

    #[test]
    fn compare_const_assert_matrix() -> Result<(), String> {
        for scope in 0..8 {
            for predicate in 0..24 {
                // Loop scopes have five predicate shapes and three malformed
                // statements.
                if scope >= 6 && !matches!(predicate, 0..=4 | 15..=17) {
                    continue;
                }
                for (flags, first, second, floor) in [
                    (0, 7, 7, 8),
                    (5, 7, 6, 8),
                    (10, 64, 128, 66),
                    (19, 128, 64, 65),
                    (252, 6, 7, 8),
                ] {
                    check_const_assert(&[scope, predicate, flags, first, second, floor])?;
                }
            }
        }
        Ok(())
    }
}
