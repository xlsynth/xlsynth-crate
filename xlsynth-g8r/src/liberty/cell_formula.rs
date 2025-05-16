// SPDX-License-Identifier: Apache-2.0

//! Boolean formula AST and parser for Liberty cell functions.

use crate::gate::AigOperand;
use crate::gate_builder::GateBuilder;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    Input(String),
    And(Box<Term>, Box<Term>),
    Or(Box<Term>, Box<Term>),
    Negate(Box<Term>),
    Constant(bool),
}

impl Term {
    /// Recursively collect all input names in the formula.
    pub fn inputs(&self) -> Vec<String> {
        let mut v = Vec::new();
        self.collect_inputs(&mut v);
        v
    }
    fn collect_inputs(&self, v: &mut Vec<String>) {
        match self {
            Term::Input(s) => v.push(s.clone()),
            Term::And(a, b) | Term::Or(a, b) => {
                a.collect_inputs(v);
                b.collect_inputs(v);
            }
            Term::Negate(t) => t.collect_inputs(v),
            Term::Constant(_) => {}
        }
    }

    /// Emits the logic for this formula into the given GateBuilder, using the
    /// provided input mapping. Returns the output AigOperand.
    pub fn emit_formula_term(
        &self,
        gb: &mut GateBuilder,
        input_map: &HashMap<String, AigOperand>,
    ) -> AigOperand {
        match self {
            Term::Input(name) => input_map
                .get(name)
                .cloned()
                .expect("input not found in map"),
            Term::And(lhs, rhs) => {
                let l = lhs.emit_formula_term(gb, input_map);
                let r = rhs.emit_formula_term(gb, input_map);
                gb.add_and_binary(l, r)
            }
            Term::Or(lhs, rhs) => {
                let l = lhs.emit_formula_term(gb, input_map);
                let r = rhs.emit_formula_term(gb, input_map);
                gb.add_or_binary(l, r)
            }
            Term::Negate(inner) => {
                let x = inner.emit_formula_term(gb, input_map);
                gb.add_not(x)
            }
            Term::Constant(true) => gb.get_true(),
            Term::Constant(false) => gb.get_false(),
        }
    }
}

/// Parse a Liberty boolean formula string into a Term AST.
pub fn parse_formula(s: &str) -> Result<Term, String> {
    let tokens = tokenize(s)?;
    let (term, rest) = parse_expr(&tokens)?;
    if !rest.is_empty() {
        return Err(format!("Unexpected tokens at end: {:?}", rest));
    }
    Ok(term)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Ident(String),
    LParen,
    RParen,
    And,
    Or,
    Not,
    Const(bool),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                tokens.push(Tok::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Tok::RParen);
                chars.next();
            }
            '*' => {
                tokens.push(Tok::And);
                chars.next();
            }
            '+' => {
                tokens.push(Tok::Or);
                chars.next();
            }
            '!' => {
                tokens.push(Tok::Not);
                chars.next();
            }
            '1' => {
                tokens.push(Tok::Const(true));
                chars.next();
            }
            '0' => {
                tokens.push(Tok::Const(false));
                chars.next();
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut ident = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2.is_ascii_alphanumeric() || c2 == '_' {
                        ident.push(c2);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Tok::Ident(ident));
            }
            _ => return Err(format!("Unexpected character in formula: '{}'", c)),
        }
    }
    Ok(tokens)
}

// Recursive descent parser for formula expressions
fn parse_expr(tokens: &[Tok]) -> Result<(Term, &[Tok]), String> {
    parse_or(tokens)
}

fn parse_or(tokens: &[Tok]) -> Result<(Term, &[Tok]), String> {
    let (mut lhs, mut rest) = parse_and(tokens)?;
    while let Some(Tok::Or) = rest.first() {
        rest = &rest[1..];
        let (rhs, rest2) = parse_and(rest)?;
        lhs = Term::Or(Box::new(lhs), Box::new(rhs));
        rest = rest2;
    }
    Ok((lhs, rest))
}

fn parse_and(tokens: &[Tok]) -> Result<(Term, &[Tok]), String> {
    let (mut lhs, mut rest) = parse_not(tokens)?;
    while let Some(Tok::And) = rest.first() {
        rest = &rest[1..];
        let (rhs, rest2) = parse_not(rest)?;
        lhs = Term::And(Box::new(lhs), Box::new(rhs));
        rest = rest2;
    }
    Ok((lhs, rest))
}

fn parse_not(tokens: &[Tok]) -> Result<(Term, &[Tok]), String> {
    if let Some(Tok::Not) = tokens.first() {
        let (expr, rest) = parse_not(&tokens[1..])?;
        Ok((Term::Negate(Box::new(expr)), rest))
    } else {
        parse_atom(tokens)
    }
}

fn parse_atom(tokens: &[Tok]) -> Result<(Term, &[Tok]), String> {
    match tokens.first() {
        Some(Tok::Ident(s)) => Ok((Term::Input(s.clone()), &tokens[1..])),
        Some(Tok::Const(b)) => Ok((Term::Constant(*b), &tokens[1..])),
        Some(Tok::LParen) => {
            let (expr, rest) = parse_expr(&tokens[1..])?;
            match rest.first() {
                Some(Tok::RParen) => Ok((expr, &rest[1..])),
                _ => Err("Expected ')'".to_string()),
            }
        }
        Some(tok) => Err(format!("Unexpected token: {:?}", tok)),
        None => Err("Unexpected end of input".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn test_parse_simple_and() {
        let t = parse_formula("(A * B)").unwrap();
        assert_eq!(
            t,
            Term::And(
                Box::new(Term::Input("A".to_string())),
                Box::new(Term::Input("B".to_string()))
            )
        );
    }
    #[test]
    fn test_parse_simple_or() {
        let t = parse_formula("(A + B)").unwrap();
        assert_eq!(
            t,
            Term::Or(
                Box::new(Term::Input("A".to_string())),
                Box::new(Term::Input("B".to_string()))
            )
        );
    }
    #[test]
    fn test_parse_negate() {
        let t = parse_formula("(!A)").unwrap();
        assert_eq!(t, Term::Negate(Box::new(Term::Input("A".to_string()))));
    }
    #[test]
    fn test_parse_nested() {
        let t = parse_formula("(!(A * (B + C)))").unwrap();
        assert_eq!(
            t,
            Term::Negate(Box::new(Term::And(
                Box::new(Term::Input("A".to_string())),
                Box::new(Term::Or(
                    Box::new(Term::Input("B".to_string())),
                    Box::new(Term::Input("C".to_string()))
                ))
            )))
        );
    }
    #[test]
    fn test_parse_input() {
        let t = parse_formula("A").unwrap();
        assert_eq!(t, Term::Input("A".to_string()));
    }
    #[test]
    fn test_parse_error() {
        assert!(parse_formula("(A * )").is_err());
        assert!(parse_formula("(A + )").is_err());
        assert!(parse_formula("(!)").is_err());
        assert!(parse_formula("").is_err());
    }
    #[test]
    fn test_parse_real_world_formula_1() {
        // (!A1 * !B) + (!A2 * !B) + (!C)
        let t = parse_formula("(!A1 * !B) + (!A2 * !B) + (!C)").unwrap();
        use Term::*;
        assert_eq!(
            t,
            Or(
                Box::new(Or(
                    Box::new(And(
                        Box::new(Negate(Box::new(Input("A1".to_string())))),
                        Box::new(Negate(Box::new(Input("B".to_string())))),
                    )),
                    Box::new(And(
                        Box::new(Negate(Box::new(Input("A2".to_string())))),
                        Box::new(Negate(Box::new(Input("B".to_string())))),
                    )),
                )),
                Box::new(Negate(Box::new(Input("C".to_string())))),
            )
        );
    }
    #[test]
    fn test_parse_real_world_formula_2() {
        // (!A1 * !B * !D) + (!A2 * !B * !D) + (!C * !D)
        let t = parse_formula("(!A1 * !B * !D) + (!A2 * !B * !D) + (!C * !D)").unwrap();
        use Term::*;
        assert_eq!(
            t,
            Or(
                Box::new(Or(
                    Box::new(And(
                        Box::new(And(
                            Box::new(Negate(Box::new(Input("A1".to_string())))),
                            Box::new(Negate(Box::new(Input("B".to_string())))),
                        )),
                        Box::new(Negate(Box::new(Input("D".to_string())))),
                    )),
                    Box::new(And(
                        Box::new(And(
                            Box::new(Negate(Box::new(Input("A2".to_string())))),
                            Box::new(Negate(Box::new(Input("B".to_string())))),
                        )),
                        Box::new(Negate(Box::new(Input("D".to_string())))),
                    )),
                )),
                Box::new(And(
                    Box::new(Negate(Box::new(Input("C".to_string())))),
                    Box::new(Negate(Box::new(Input("D".to_string())))),
                )),
            )
        );
    }
    #[test]
    fn test_parse_real_world_formula_3() {
        // (A1 * A2) + (B)
        let t = parse_formula("(A1 * A2) + (B)").unwrap();
        use Term::*;
        assert_eq!(
            t,
            Or(
                Box::new(And(
                    Box::new(Input("A1".to_string())),
                    Box::new(Input("A2".to_string())),
                )),
                Box::new(Input("B".to_string())),
            )
        );
    }
    #[test]
    fn test_parse_real_world_formula_4() {
        // (A1 * A2) + (B1 * B2) + (C)
        let t = parse_formula("(A1 * A2) + (B1 * B2) + (C)").unwrap();
        use Term::*;
        assert_eq!(
            t,
            Or(
                Box::new(Or(
                    Box::new(And(
                        Box::new(Input("A1".to_string())),
                        Box::new(Input("A2".to_string())),
                    )),
                    Box::new(And(
                        Box::new(Input("B1".to_string())),
                        Box::new(Input("B2".to_string())),
                    )),
                )),
                Box::new(Input("C".to_string())),
            )
        );
    }
    #[test]
    fn test_parse_constant_true() {
        let t = parse_formula("1").unwrap();
        assert_eq!(t, Term::Constant(true));
    }
    #[test]
    fn test_parse_constant_false() {
        let t = parse_formula("0").unwrap();
        assert_eq!(t, Term::Constant(false));
    }
    #[test]
    fn test_emit_formula_term_and() {
        let mut gb = GateBuilder::new("test_and".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let mut input_map = HashMap::new();
        input_map.insert("A".to_string(), *a.get_lsb(0));
        input_map.insert("B".to_string(), *b.get_lsb(0));
        let term = Term::And(
            Box::new(Term::Input("A".to_string())),
            Box::new(Term::Input("B".to_string())),
        );
        let out = term.emit_formula_term(&mut gb, &input_map);
        gb.add_output("out".to_string(), crate::gate::AigBitVector::from_bit(out));
        let gate_fn = gb.build();
        let s = gate_fn.to_string();
        assert!(s.contains("and(a[0], b[0])"));
    }
    #[test]
    fn test_emit_formula_term_not_or() {
        let mut gb = GateBuilder::new("test_not_or".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let mut input_map = HashMap::new();
        input_map.insert("A".to_string(), *a.get_lsb(0));
        input_map.insert("B".to_string(), *b.get_lsb(0));
        let term = Term::Negate(Box::new(Term::Or(
            Box::new(Term::Input("A".to_string())),
            Box::new(Term::Input("B".to_string())),
        )));
        let out = term.emit_formula_term(&mut gb, &input_map);
        gb.add_output("out".to_string(), crate::gate::AigBitVector::from_bit(out));
        let gate_fn = gb.build();
        let s = gate_fn.to_string();
        log::info!("GateFn output: {}", s);
        assert!(
            s.contains("and(not(a[0]), not(b[0]))"),
            "Expected De Morgan's form"
        );
        assert!(s.contains("out[0] ="), "Expected output assignment");
    }
    #[test]
    fn test_emit_formula_term_constant_true() {
        let mut gb = GateBuilder::new("test_true".to_string(), GateBuilderOptions::no_opt());
        let input_map = HashMap::new();
        let term = Term::Constant(true);
        let out = term.emit_formula_term(&mut gb, &input_map);
        assert!(gb.is_known_true(out));
    }
    #[test]
    fn test_emit_formula_term_constant_false() {
        let mut gb = GateBuilder::new("test_false".to_string(), GateBuilderOptions::no_opt());
        let input_map = HashMap::new();
        let term = Term::Constant(false);
        let out = term.emit_formula_term(&mut gb, &input_map);
        assert!(gb.is_known_false(out));
    }
}
