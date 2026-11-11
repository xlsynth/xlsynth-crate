// SPDX-License-Identifier: Apache-2.0

//! Parse source tests and their procedural control flow.

use std::collections::BTreeSet;

use super::{Braces, Parser, ValueContext, decode_byte_string};
use crate::ast::{TestDecl, TestKind, TestStmt, TestStmtKind, Type};

impl Parser<'_> {
    /// Parse a parameterless test using its declaration's normal expression
    /// dialect.
    pub(super) fn test_declaration(&mut self) -> Result<TestDecl, String> {
        let kind = if self.eat("fn") {
            self.value_context = ValueContext::DslxItem;
            TestKind::Function
        } else if self.eat("block") {
            TestKind::Block
        } else {
            return Err(self.error("expected private fn or block after #[test]"));
        };
        let position = self.position_at(self.tokens[self.pos].start);
        let name = self.ident()?;
        if self.peek(0) == "<" {
            return Err(self.error("BX tests cannot have parametric bindings"));
        }
        self.expect("(")?;
        if !self.eat(")") {
            return Err(self.error("BX tests take no arguments"));
        }
        if self.eat("->") && self.ty()? != Type::Tuple(Vec::new()) {
            return Err(self.error("BX tests must return unit"));
        }
        self.expect("{")?;
        let statements = self.with_name_scope(|p| p.test_body(kind))?;
        self.value_context = ValueContext::Block;
        Ok(TestDecl {
            kind,
            name,
            position,
            statements,
            overwritten_literal_types: std::mem::take(&mut self.overwritten_literal_types),
        })
    }

    /// Parse the statements after an opening brace, allowing a trailing unit
    /// value.
    fn test_body(&mut self, test: TestKind) -> Result<Vec<TestStmt>, String> {
        let mut body = Vec::new();
        while !self.eat("}") {
            if self.peek(0) == "(" && self.peek(1) == ")" {
                self.pos += 2;
                self.eat(";");
                self.expect("}")?;
                break;
            }
            let position = self.position_at(self.tokens[self.pos].start);
            let kind = self.test_statement(test)?;
            match kind {
                TestStmtKind::If { .. } | TestStmtKind::For { .. } => {
                    self.eat(";");
                }
                TestStmtKind::Tick { .. }
                | TestStmtKind::Assert { .. }
                | TestStmtKind::AssertEq { .. }
                    if self.peek(0) == "}" =>
                {
                    // The final unit-valued call can serve as the body's
                    // result.
                }
                _ => self.expect(";")?,
            }
            body.push(TestStmt { position, kind });
        }
        Ok(body)
    }

    /// Parse one test action, keeping loop binders and branch lets in their own
    /// scopes.
    fn test_statement(&mut self, test: TestKind) -> Result<TestStmtKind, String> {
        if test == TestKind::Function
            && (self.peek(0) == "tick!"
                || self.peek(1) == ":="
                || self.peek(0) == "reg" && self.peek(1) != "(")
        {
            return Err(self.error("clocked stimulus requires a #[test] block"));
        }
        if self.eat("reg") {
            let param = self.param()?;
            let init = if self.eat("=") {
                Some(self.expr()?.value)
            } else {
                None
            };
            self.shadow_callable(&param.name);
            return Ok(TestStmtKind::Reg {
                name: param.name,
                ty: param.ty,
                init,
            });
        }
        if self.eat("let") {
            return Ok(TestStmtKind::Let(self.let_binding(true)?.0));
        }
        if self.peek(1) == ":=" {
            let name = self.ident()?;
            self.expect(":=")?;
            return Ok(TestStmtKind::Assign {
                name,
                value: self.expr()?.value,
            });
        }
        if self.eat("tick!") {
            self.expect("(")?;
            let reset = if self.eat("rst") {
                self.expect(":")?;
                Some(self.expr()?.value)
            } else {
                None
            };
            self.expect(")")?;
            return Ok(TestStmtKind::Tick { reset });
        }
        if self.eat("assert!") {
            self.expect("(")?;
            let predicate = self.expr()?.value;
            let label = if self.eat(",") {
                let literal = self.peek(0);
                if !literal.starts_with('"') {
                    return Err(self.error("expected a string literal for the assertion label"));
                }
                let bytes = decode_byte_string(literal).map_err(|e| self.error(e))?;
                // An escaped byte outside UTF-8 retains its written spelling in
                // diagnostics.
                let label = String::from_utf8(bytes)
                    .unwrap_or_else(|_| literal[1..literal.len() - 1].to_owned());
                self.pos += 1;
                Some(label)
            } else {
                None
            };
            self.expect(")")?;
            return Ok(TestStmtKind::Assert { predicate, label });
        }
        if self.eat("assert_eq") {
            self.expect("(")?;
            let actual = self.expr()?.value;
            self.expect(",")?;
            let expected = self.expr()?.value;
            self.expect(")")?;
            return Ok(TestStmtKind::AssertEq { actual, expected });
        }
        if self.eat("for") {
            let mut names = BTreeSet::new();
            let pattern = self.binding_pattern(&mut names, self.identifier_predicate())?;
            self.expect("in")?;
            let iterable = self.for_iterable(Braces::Body)?.into_ast().0;
            self.expect("{")?;
            let body = self.nested(|p| {
                p.with_name_scope(|p| {
                    for name in names {
                        p.shadow_callable(&name);
                    }
                    p.test_body(test)
                })
            })?;
            return Ok(TestStmtKind::For {
                pattern,
                iterable,
                body,
            });
        }
        if self.eat("if") {
            let condition = self.condition()?.value;
            let then_body = self.nested(|p| p.with_name_scope(|p| p.test_body(test)))?;
            let else_body = if !self.eat("else") {
                Vec::new()
            } else if self.peek(0) == "if" {
                let position = self.position_at(self.tokens[self.pos].start);
                let kind = self.nested(|p| p.test_statement(test))?;
                vec![TestStmt { position, kind }]
            } else {
                self.expect("{")?;
                self.nested(|p| p.with_name_scope(|p| p.test_body(test)))?
            };
            return Ok(TestStmtKind::If {
                condition,
                then_body,
                else_body,
            });
        }
        Err(self.error("expected a BX test statement"))
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{BindingPattern, Expr, ForIterable, SourcePosition, TestKind, TestStmtKind};
    use crate::parse_module;

    #[test]
    fn tests_keep_procedural_order_and_original_locations() -> Result<(), String> {
        let module = parse_module(
            r#"#![require_define_before_use]
fn twice(x: u8) -> u8 { x + x }
#[test] fn pure() -> () {
    assert!(twice(u8:2) == u8:4, "two\x20twice")
}
#[test] block table() {
    reg drive: u8;
    let observed = dut(drive);
    for (stimulus, expected) in [(u8:1, u8:3)] {
        drive := stimulus;
        tick!(rst: false);
        if true { assert_eq(observed, expected); } else { tick!(); }
    }
    for _ in u8:1..=u8:2 { tick!(); }
    ()
}"#,
        )?;
        assert_eq!(module.declaration_order, ["twice", "pure", "table"]);
        assert_eq!(module.functions.len(), 1);
        assert!(module.blocks.is_empty());
        assert_eq!(module.tests[0].kind, TestKind::Function);
        assert!(
            matches!(&module.tests[0].statements[0].kind, TestStmtKind::Assert { label: Some(label), .. } if label == "two twice")
        );
        let block = &module.tests[1];
        assert_eq!(
            block.position,
            SourcePosition {
                line: 6,
                column: 15
            }
        );
        assert_eq!(block.kind, TestKind::Block);
        assert!(
            matches!(&block.statements[0].kind, TestStmtKind::Reg { name, init: None, .. } if name == "drive")
        );
        let TestStmtKind::For {
            pattern: BindingPattern::Tuple(fields),
            iterable: ForIterable::Array(_),
            body,
        } = &block.statements[2].kind
        else {
            panic!("expected a loop over table rows");
        };
        assert_eq!(fields.len(), 2);
        assert!(matches!(&body[0].kind, TestStmtKind::Assign { name, .. } if name == "drive"));
        assert!(matches!(
            &body[1].kind,
            TestStmtKind::Tick {
                reset: Some(Expr::Bool(false))
            }
        ));
        let TestStmtKind::If {
            then_body,
            else_body,
            ..
        } = &body[2].kind
        else {
            panic!("expected a conditional");
        };
        assert_eq!(
            then_body[0].position,
            SourcePosition {
                line: 12,
                column: 19
            }
        );
        assert!(matches!(then_body[0].kind, TestStmtKind::AssertEq { .. }));
        assert!(matches!(
            else_body[0].kind,
            TestStmtKind::Tick { reset: None }
        ));
        assert!(matches!(
            block.statements[3].kind,
            TestStmtKind::For {
                iterable: ForIterable::Range {
                    inclusive: true,
                    ..
                },
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn tests_require_private_parameterless_unit_declarations() {
        for source in [
            "#[test]",
            "#[test] const flag = true;",
            "#[test] pub fn t() {}",
            "#[test] fn t<N: u32>() {}",
            "#[test] fn t(x: u8) {}",
            "#[test] block t(clk) {}",
            "#[test] fn t() -> u8 {}",
            "#[test] fn t() { reg x: u8; }",
            "#[test] fn t() { tick!(); }",
            "#[test] fn t() { x := u8:3; }",
            "#[test] fn t() { u8:1 }",
            "#[test] block t() { tick!(true); }",
            "#[test] fn t() {} #[test] block t() {}",
            "#[test] fn t() {} fn t() {}",
        ] {
            assert!(parse_module(source).is_err(), "{source}");
        }
    }

    #[test]
    fn test_names_do_not_shadow_hardware_builtins() -> Result<(), String> {
        let module = parse_module(
            "#[test] fn map() {} block demo(clk, xs: u8[2]) -> u8[2] { map(xs, |x| x + x) }",
        )?;
        assert!(matches!(module.blocks[0].output, Expr::Map(_)));
        Ok(())
    }

    #[test]
    fn counted_for_in_assertions_retains_its_accumulator_and_seed() -> Result<(), String> {
        for kind in ["fn", "block"] {
            let module = parse_module(&format!(
                "#[test] {kind} fold_test() {{ assert_eq(for (i, acc) in u8:1..u8:4 {{ acc + i }}(u8:0), u8:6); }}"
            ))?;
            assert!(matches!(
                &module.tests[0].statements[0].kind,
                TestStmtKind::AssertEq { actual: Expr::For(fold), .. }
                    if fold.accumulator == BindingPattern::Name("acc".into())
                    && matches!(&fold.seed, Expr::Literal { text, .. } if text == "0")
            ));
        }
        Ok(())
    }
}
