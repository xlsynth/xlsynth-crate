// SPDX-License-Identifier: Apache-2.0

//! Boolean formula AST and parser for Liberty cell functions.

use crate::aig::gate::AigOperand;
use crate::gate_builder::GateBuilder;
use crate::liberty::liberty_parser::{Block, BlockMember, Value};
use crate::liberty::{CharReader, LibertyParser};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    Input(String),
    And(Box<Term>, Box<Term>),
    Or(Box<Term>, Box<Term>),
    Xor(Box<Term>, Box<Term>),
    Negate(Box<Term>),
    Constant(bool),
}

/// Context for emitting a formula term, used for error reporting.
pub struct EmitContext<'a> {
    pub cell_name: &'a str,
    pub original_formula: &'a str,
    pub instance_name: Option<&'a str>,
    pub port_map: Option<&'a std::collections::HashMap<String, String>>, // port_name -> net_name
}

impl Term {
    /// Recursively collect all input names in the formula.
    pub fn inputs(&self) -> Vec<String> {
        let mut v = Vec::new();
        self.collect_inputs(&mut v);
        v
    }

    /// Evaluates the formula when enough input bindings are known.
    pub fn evaluate_partial(&self, input_values: &HashMap<String, bool>) -> Option<bool> {
        match self {
            Term::Input(name) => input_values.get(name).copied(),
            Term::And(lhs, rhs) => match (
                lhs.evaluate_partial(input_values),
                rhs.evaluate_partial(input_values),
            ) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            Term::Or(lhs, rhs) => match (
                lhs.evaluate_partial(input_values),
                rhs.evaluate_partial(input_values),
            ) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            Term::Xor(lhs, rhs) => match (
                lhs.evaluate_partial(input_values),
                rhs.evaluate_partial(input_values),
            ) {
                (Some(lhs), Some(rhs)) => Some(lhs ^ rhs),
                _ => None,
            },
            Term::Negate(inner) => inner.evaluate_partial(input_values).map(|value| !value),
            Term::Constant(value) => Some(*value),
        }
    }

    fn collect_inputs(&self, v: &mut Vec<String>) {
        match self {
            Term::Input(s) => v.push(s.clone()),
            Term::And(a, b) | Term::Or(a, b) | Term::Xor(a, b) => {
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
        context: &EmitContext,
    ) -> Result<AigOperand, String> {
        match self {
            Term::Input(name) => input_map.get(name).cloned().ok_or_else(|| {
                let available: Vec<_> = input_map.keys().cloned().collect();
                let required: Vec<_> = self.inputs();
                let mut msg = format!(
                    "Input '{}' not found in map for cell '{}' (formula: \"{}\")",
                    name, context.cell_name, context.original_formula
                );
                if let Some(inst) = context.instance_name {
                    msg.push_str(&format!("\n  Instance name: {}", inst));
                }
                if let Some(port_map) = context.port_map {
                    msg.push_str("\n  Port mapping:");
                    for (port, net) in port_map {
                        msg.push_str(&format!("\n    .{}({})", port, net));
                    }
                }
                msg.push_str(&format!("\n  Available inputs: {:?}", available));
                msg.push_str(&format!("\n  Required by formula: {:?}", required));
                msg
            }),
            Term::And(lhs, rhs) => {
                let l = lhs.emit_formula_term(gb, input_map, context)?;
                let r = rhs.emit_formula_term(gb, input_map, context)?;
                Ok(gb.add_and_binary(l, r))
            }
            Term::Or(lhs, rhs) => {
                let l = lhs.emit_formula_term(gb, input_map, context)?;
                let r = rhs.emit_formula_term(gb, input_map, context)?;
                Ok(gb.add_or_binary(l, r))
            }
            Term::Xor(lhs, rhs) => {
                let l = lhs.emit_formula_term(gb, input_map, context)?;
                let r = rhs.emit_formula_term(gb, input_map, context)?;
                Ok(gb.add_xor_binary(l, r))
            }
            Term::Negate(inner) => {
                let x = inner.emit_formula_term(gb, input_map, context)?;
                Ok(gb.add_not(x))
            }
            Term::Constant(true) => Ok(gb.get_true()),
            Term::Constant(false) => Ok(gb.get_false()),
        }
    }
}

/// Parse a Liberty boolean formula string into a Term AST.
pub fn parse_formula(s: &str) -> Result<Term, String> {
    parse_formula_with_precedence(s, FormulaPrecedence::Liberty)
}

/// Detects a structural parse difference between Liberty and ABC precedence.
pub(crate) fn abc_formula_has_different_parse(s: &str) -> Result<bool, String> {
    let liberty = parse_formula_with_precedence(s, FormulaPrecedence::Liberty)?;
    let abc = parse_formula_with_precedence(s, FormulaPrecedence::Abc)?;
    Ok(liberty != abc)
}

/// Finds mapped cells whose source formulas cannot be interpreted safely by
/// ABC.
pub(crate) fn abc_source_formula_issues(
    path: &Path,
    mapped_cells: &BTreeSet<&str>,
) -> Result<BTreeMap<String, String>, String> {
    let file = File::open(path).map_err(|error| format!("opening {}: {error}", path.display()))?;
    let library = LibertyParser::new_from_iter(CharReader::new(file))
        .parse()
        .map_err(|error| format!("parsing {}: {error}", path.display()))?;
    let mut issues = BTreeMap::new();
    for member in &library.members {
        let BlockMember::SubBlock(cell) = member else {
            continue;
        };
        if cell.block_type != "cell" {
            continue;
        }
        let Some(Value::String(name) | Value::Identifier(name)) = cell.qualifiers.first() else {
            continue;
        };
        if mapped_cells.contains(name.as_str()) {
            collect_abc_formula_issues(cell, name, &mut issues);
        }
    }
    Ok(issues)
}

/// Checks pin and bundle formulas before source normalization can rewrite them.
fn collect_abc_formula_issues(
    block: &Block,
    cell_name: &str,
    issues: &mut BTreeMap<String, String>,
) {
    for member in &block.members {
        match member {
            BlockMember::BlockAttr(attr)
                if matches!(block.block_type.as_str(), "pin" | "bundle" | "bus")
                    && attr.attr_name == "function" =>
            {
                let pin = match block.qualifiers.first() {
                    Some(Value::String(name) | Value::Identifier(name)) => name.as_str(),
                    _ => "<unknown>",
                };
                let error = match &attr.value {
                    Value::String(formula) | Value::Identifier(formula) => {
                        match abc_formula_has_different_parse(formula) {
                            Ok(false) => None,
                            Ok(true) => Some(format!(
                                "ABC may interpret AND/XOR precedence differently for {cell_name}.{pin}: {formula:?}"
                            )),
                            Err(error) => Some(format!(
                                "cannot check ABC formula for {cell_name}.{pin}: {error}"
                            )),
                        }
                    }
                    _ => Some(format!("invalid ABC formula for {cell_name}.{pin}")),
                };
                if let Some(error) = error {
                    issues.entry(cell_name.to_string()).or_insert(error);
                }
            }
            BlockMember::SubBlock(child)
                if matches!(child.block_type.as_str(), "pin" | "bundle" | "bus") =>
            {
                collect_abc_formula_issues(child, cell_name, issues);
            }
            _ => {
                // Only ordinary pin and bundle function attributes affect
                // mapping.
            }
        }
    }
}

#[derive(Clone, Copy)]
enum FormulaPrecedence {
    Liberty,
    Abc,
}

fn parse_formula_with_precedence(s: &str, precedence: FormulaPrecedence) -> Result<Term, String> {
    let tokens = tokenize(s)?;
    let (term, rest) = parse_expr(&tokens, precedence)?;
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
    Xor,
    Not,
    PostfixNot,
    Const(bool),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();
    let mut whitespace = false;
    while let Some(&c) = chars.peek() {
        let token = match c {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
                whitespace = true;
                continue;
            }
            '(' => {
                chars.next();
                Tok::LParen
            }
            ')' => {
                chars.next();
                Tok::RParen
            }
            '*' | '&' => {
                chars.next();
                Tok::And
            }
            '+' | '|' => {
                chars.next();
                Tok::Or
            }
            '^' => {
                chars.next();
                Tok::Xor
            }
            '!' => {
                chars.next();
                Tok::Not
            }
            '\'' => {
                chars.next();
                Tok::PostfixNot
            }
            '1' => {
                chars.next();
                Tok::Const(true)
            }
            '0' => {
                chars.next();
                Tok::Const(false)
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut ident = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2.is_ascii_alphanumeric() || matches!(c2, '_' | '[' | ']') {
                        ident.push(c2);
                        chars.next();
                    } else {
                        break;
                    }
                }
                Tok::Ident(ident)
            }
            _ => return Err(format!("Unexpected character in formula: '{}'", c)),
        };
        // Liberty uses whitespace as AND only between complete operands; it
        // is otherwise insignificant around operators, parentheses, and NOT.
        if whitespace
            && matches!(
                tokens.last(),
                Some(Tok::Ident(_) | Tok::Const(_) | Tok::RParen | Tok::PostfixNot)
            )
            && matches!(
                token,
                Tok::Ident(_) | Tok::Const(_) | Tok::LParen | Tok::Not
            )
        {
            tokens.push(Tok::And);
        }
        tokens.push(token);
        whitespace = false;
    }
    Ok(tokens)
}

// ABC gives AND higher precedence than XOR, unlike Liberty. Use the same
// tokenizer and unary parser for both so explicit parentheses remain binding.
fn parse_expr(tokens: &[Tok], precedence: FormulaPrecedence) -> Result<(Term, &[Tok]), String> {
    parse_binary(tokens, precedence, 1)
}

/// Parses left-associative binary operators in the selected precedence order.
fn parse_binary(
    tokens: &[Tok],
    precedence: FormulaPrecedence,
    min_precedence: u8,
) -> Result<(Term, &[Tok]), String> {
    let (mut lhs, mut rest) = parse_not(tokens, precedence)?;
    while let Some(operator) = rest.first() {
        let binding = match (operator, precedence) {
            (Tok::Or, _) => 1,
            (Tok::And, FormulaPrecedence::Liberty) | (Tok::Xor, FormulaPrecedence::Abc) => 2,
            (Tok::Xor, FormulaPrecedence::Liberty) | (Tok::And, FormulaPrecedence::Abc) => 3,
            _ => break,
        };
        if binding < min_precedence {
            break;
        }
        let (rhs, next) = parse_binary(&rest[1..], precedence, binding + 1)?;
        lhs = match operator {
            Tok::And => Term::And(Box::new(lhs), Box::new(rhs)),
            Tok::Or => Term::Or(Box::new(lhs), Box::new(rhs)),
            Tok::Xor => Term::Xor(Box::new(lhs), Box::new(rhs)),
            _ => unreachable!("only binary operators have a binding"),
        };
        rest = next;
    }
    Ok((lhs, rest))
}

fn parse_not(tokens: &[Tok], precedence: FormulaPrecedence) -> Result<(Term, &[Tok]), String> {
    if let Some(Tok::Not) = tokens.first() {
        let (expr, rest) = parse_not(&tokens[1..], precedence)?;
        Ok((Term::Negate(Box::new(expr)), rest))
    } else {
        let (mut term, mut rest) = parse_atom(tokens, precedence)?;
        while let Some(Tok::PostfixNot) = rest.first() {
            term = Term::Negate(Box::new(term));
            rest = &rest[1..];
        }
        Ok((term, rest))
    }
}

fn parse_atom(tokens: &[Tok], precedence: FormulaPrecedence) -> Result<(Term, &[Tok]), String> {
    match tokens.first() {
        Some(Tok::Ident(s)) => Ok((Term::Input(s.clone()), &tokens[1..])),
        Some(Tok::Const(b)) => Ok((Term::Constant(*b), &tokens[1..])),
        Some(Tok::LParen) => {
            let (expr, rest) = parse_expr(&tokens[1..], precedence)?;
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
    use crate::{
        aig::gate::AigBitVector,
        gate_builder::{GateBuilder, GateBuilderOptions},
    };

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
    fn test_whitespace_and_respects_operands_and_explicit_operators() {
        for (implicit, explicit) in [
            (" A \t B ", "A*B"),
            ("A B C", "A*B*C"),
            ("A (B + C)", "A*(B+C)"),
            ("(A + B) C", "(A+B)*C"),
            ("(A + B) (C + D)", "(A+B)*(C+D)"),
            ("A ! B", "A*!B"),
            ("A ' B", "A'*B"),
            ("! A B", "!A*B"),
            ("A 1 + 0 B", "A*1+0*B"),
            (" A  *  B +  ! ( C ) ", "A*B+!C"),
        ] {
            assert_eq!(
                parse_formula(implicit).unwrap(),
                parse_formula(explicit).unwrap(),
                "{implicit}"
            );
        }
        for invalid in ["A(B)", "(A)B", "A!B", "(A)(B)", "1A"] {
            assert!(parse_formula(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn test_mixed_operator_precedence_against_truth_tables() {
        let formulas = [
            "A * B ^ C + D",
            "A B ^ C + D",
            "A & B ^ C | D",
            " A \t B ^ C + D ",
        ];
        for formula in formulas {
            let term = parse_formula(formula).unwrap();
            for assignment in 0..16 {
                let a = assignment & 1 != 0;
                let b = assignment & 2 != 0;
                let c = assignment & 4 != 0;
                let d = assignment & 8 != 0;
                let values = HashMap::from([
                    ("A".to_string(), a),
                    ("B".to_string(), b),
                    ("C".to_string(), c),
                    ("D".to_string(), d),
                ]);
                assert_eq!(
                    term.evaluate_partial(&values),
                    Some((a && (b ^ c)) || d),
                    "{formula}, assignment {assignment}"
                );
            }
        }
        let term = parse_formula("!A ^ B C + D'").unwrap();
        for assignment in 0..16 {
            let a = assignment & 1 != 0;
            let b = assignment & 2 != 0;
            let c = assignment & 4 != 0;
            let d = assignment & 8 != 0;
            let values = HashMap::from([
                ("A".to_string(), a),
                ("B".to_string(), b),
                ("C".to_string(), c),
                ("D".to_string(), d),
            ]);
            assert_eq!(
                term.evaluate_partial(&values),
                Some(((!a ^ b) && c) || !d),
                "assignment {assignment}"
            );
        }
    }

    #[test]
    fn abc_precedence_guard_detects_only_unparenthesized_differences() {
        for formula in ["A*B^C", "A^B*C", "A B^C", "A^B C", "!A^B*C"] {
            assert!(
                abc_formula_has_different_parse(formula).unwrap(),
                "{formula}"
            );
        }
        for formula in [
            "A*(B^C)", "(A*B)^C", "A^(B*C)", "(A^B)*C", "A (B^C)", "(A B)^C", "A^B+C", "A+B*C",
            "A' B+C",
        ] {
            assert!(
                !abc_formula_has_different_parse(formula).unwrap(),
                "{formula}"
            );
        }
        assert!(abc_formula_has_different_parse("A*").is_err());

        let abc = parse_formula_with_precedence("A*B^C", FormulaPrecedence::Abc).unwrap();
        let liberty = parse_formula("A*B^C").unwrap();
        for assignment in 0..8 {
            let a = assignment & 1 != 0;
            let b = assignment & 2 != 0;
            let c = assignment & 4 != 0;
            let values = HashMap::from([
                ("A".to_string(), a),
                ("B".to_string(), b),
                ("C".to_string(), c),
            ]);
            assert_eq!(abc.evaluate_partial(&values), Some((a && b) ^ c));
            assert_eq!(liberty.evaluate_partial(&values), Some(a && (b ^ c)));
        }
    }

    #[test]
    fn test_evaluate_partial_uses_known_constants() {
        let t = parse_formula("A * !B + C").unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("B".to_string(), true);
        assert_eq!(t.evaluate_partial(&inputs), None);

        inputs.insert("C".to_string(), false);
        assert_eq!(t.evaluate_partial(&inputs), Some(false));

        inputs.insert("C".to_string(), true);
        assert_eq!(t.evaluate_partial(&inputs), Some(true));
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
        let context = EmitContext {
            cell_name: "test_cell_and",
            original_formula: "A * B",
            instance_name: None,
            port_map: None,
        };
        let out = term
            .emit_formula_term(&mut gb, &input_map, &context)
            .unwrap();
        gb.add_output("out".to_string(), AigBitVector::from_bit(out));
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
        let context = EmitContext {
            cell_name: "test_cell_not_or",
            original_formula: "!(A + B)",
            instance_name: None,
            port_map: None,
        };
        let out = term
            .emit_formula_term(&mut gb, &input_map, &context)
            .unwrap();
        gb.add_output("out".to_string(), AigBitVector::from_bit(out));
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
        let context = EmitContext {
            cell_name: "test_cell_true",
            original_formula: "1",
            instance_name: None,
            port_map: None,
        };
        let out = term
            .emit_formula_term(&mut gb, &input_map, &context)
            .unwrap();
        assert!(gb.is_known_true(out));
    }
    #[test]
    fn test_emit_formula_term_constant_false() {
        let mut gb = GateBuilder::new("test_false".to_string(), GateBuilderOptions::no_opt());
        let input_map = HashMap::new();
        let term = Term::Constant(false);
        let context = EmitContext {
            cell_name: "test_cell_false",
            original_formula: "0",
            instance_name: None,
            port_map: None,
        };
        let out = term
            .emit_formula_term(&mut gb, &input_map, &context)
            .unwrap();
        assert!(gb.is_known_false(out));
    }
    #[test]
    fn test_emit_formula_term_missing_input() {
        let mut gb = GateBuilder::new("test_missing".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let mut input_map = HashMap::new();
        input_map.insert("A".to_string(), *a.get_lsb(0));
        // "B" is intentionally missing from input_map

        let term = Term::And(
            Box::new(Term::Input("A".to_string())),
            Box::new(Term::Input("B".to_string())),
        );
        let context = EmitContext {
            cell_name: "test_cell_missing_input",
            original_formula: "A * B",
            instance_name: None,
            port_map: None,
        };
        let result = term.emit_formula_term(&mut gb, &input_map, &context);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains(
                "Input 'B' not found in map for cell 'test_cell_missing_input' (formula: \"A * B\")"
            ));
            assert!(e.contains("Available inputs: [\"A\"]"));
            assert!(e.contains("Required by formula: [\"B\"]"));
        }
    }
    #[test]
    fn test_parse_and_with_star_and_ampersand() {
        let t1 = parse_formula("(A * B)").unwrap();
        let t2 = parse_formula("(A & B)").unwrap();
        assert_eq!(t1, t2);
        let t3 = parse_formula("(A & B * C)").unwrap();
        let t4 = parse_formula("(A * B & C)").unwrap();
        assert_eq!(t3, t4);
    }
    #[test]
    fn test_parse_or_with_plus_and_pipe() {
        let t1 = parse_formula("(A + B)").unwrap();
        let t2 = parse_formula("(A | B)").unwrap();
        assert_eq!(t1, t2);
        let t3 = parse_formula("(A | B + C)").unwrap();
        let t4 = parse_formula("(A + B | C)").unwrap();
        assert_eq!(t3, t4);
    }
    #[test]
    fn test_parse_xor_with_caret() {
        let t1 = parse_formula("(A ^ B)").unwrap();
        let t2 = parse_formula("(A ^ B ^ C)").unwrap();
        use Term::*;
        assert_eq!(
            t1,
            Xor(
                Box::new(Input("A".to_string())),
                Box::new(Input("B".to_string()))
            )
        );
        assert_eq!(
            t2,
            Xor(
                Box::new(Xor(
                    Box::new(Input("A".to_string())),
                    Box::new(Input("B".to_string()))
                )),
                Box::new(Input("C".to_string()))
            )
        );
    }
}
