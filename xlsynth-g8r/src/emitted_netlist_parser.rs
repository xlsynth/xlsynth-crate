// SPDX-License-Identifier: Apache-2.0

//! Simple parser for the minimal netlist format emitted by `emit_netlist`.
//!
//! This parser only understands a very small subset of Verilog consisting of
//! a single module with one-bit inputs and outputs, wire declarations, and
//! simple `assign` statements with `&` and `~` operators.

#[derive(Debug, PartialEq, Eq)]
pub struct Module {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub wires: Vec<String>,
    pub assigns: Vec<(String, Expr)>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Expr {
    Var(String),
    Literal(bool),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
}

#[derive(Debug)]
pub struct ParseError(String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParseError: {}", self.0)
    }
}

pub struct Parser {
    chars: Vec<char>,
    offset: usize,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            offset: 0,
        }
    }

    fn rest(&self) -> String {
        self.chars[self.offset..].iter().collect()
    }

    fn drop_ws(&mut self) {
        while let Some(c) = self.chars.get(self.offset).copied() {
            if c.is_whitespace() {
                self.offset += 1;
            } else {
                break;
            }
        }
    }

    fn try_drop(&mut self, s: &str) -> bool {
        self.drop_ws();
        if self.rest().starts_with(s) {
            self.offset += s.len();
            true
        } else {
            false
        }
    }

    fn drop_or_error(&mut self, s: &str) -> Result<(), ParseError> {
        if self.try_drop(s) {
            Ok(())
        } else {
            Err(ParseError(format!(
                "expected `{}` near `{}`",
                s,
                &self.rest()[..self.rest().len().min(10)]
            )))
        }
    }

    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        self.drop_ws();
        let rest = self.rest();
        let mut chars = rest.chars();
        let mut id = String::new();
        if let Some(c) = chars.next() {
            if c.is_alphabetic() || c == '_' {
                id.push(c);
                self.offset += c.len_utf8();
            } else {
                return Err(ParseError(format!(
                    "expected identifier start, got `{}`",
                    c
                )));
            }
        } else {
            return Err(ParseError("unexpected eof".into()));
        }
        while let Some(c) = self.chars.get(self.offset).copied() {
            if c.is_alphanumeric() || c == '_' {
                id.push(c);
                self.offset += 1;
            } else {
                break;
            }
        }
        Ok(id)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_and_expr()
    }

    fn parse_and_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_unary_expr()?;
        loop {
            self.drop_ws();
            if self.try_drop("&") {
                let rhs = self.parse_unary_expr()?;
                expr = Expr::And(Box::new(expr), Box::new(rhs));
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_unary_expr(&mut self) -> Result<Expr, ParseError> {
        self.drop_ws();
        if self.try_drop("~") {
            let sub = self.parse_unary_expr()?;
            return Ok(Expr::Not(Box::new(sub)));
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        self.drop_ws();
        if self.try_drop("1'b0") {
            return Ok(Expr::Literal(false));
        }
        if self.try_drop("1'b1") {
            return Ok(Expr::Literal(true));
        }
        let id = self.parse_identifier()?;
        Ok(Expr::Var(id))
    }

    fn at_eof(&mut self) -> bool {
        self.drop_ws();
        self.offset >= self.chars.len()
    }

    pub fn parse_module(&mut self) -> Result<Module, ParseError> {
        self.drop_ws();
        self.drop_or_error("module")?;
        let name = self.parse_identifier()?;
        self.drop_or_error("(")?;
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        loop {
            self.drop_ws();
            if self.try_drop(")") {
                break;
            }
            if self.try_drop("input") {
                self.drop_ws();
                self.drop_or_error("wire")?;
                let id = self.parse_identifier()?;
                inputs.push(id);
            } else if self.try_drop("output") {
                self.drop_ws();
                self.drop_or_error("wire")?;
                let id = self.parse_identifier()?;
                outputs.push(id);
            } else {
                return Err(ParseError("expected input or output".into()));
            }
            self.drop_ws();
            self.try_drop(",");
        }
        self.drop_ws();
        self.drop_or_error(";")?;
        let mut wires = Vec::new();
        let mut assigns = Vec::new();
        while !self.at_eof() {
            if self.try_drop("endmodule") {
                break;
            } else if self.try_drop("wire") {
                let id = self.parse_identifier()?;
                self.drop_or_error(";")?;
                wires.push(id);
            } else if self.try_drop("assign") {
                let lhs = self.parse_identifier()?;
                self.drop_or_error("=")?;
                let rhs = self.parse_expr()?;
                self.drop_or_error(";")?;
                assigns.push((lhs, rhs));
            } else {
                return Err(ParseError(format!(
                    "unexpected text near `{}`",
                    self.rest()
                )));
            }
        }
        Ok(Module {
            name,
            inputs,
            outputs,
            wires,
            assigns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_netlist() {
        let text = "module m(\n  input wire i,\n  output wire o\n);\n  wire G0;\n  assign G0 = 1'b0;\n  assign o = ~i;\nendmodule";
        let mut p = Parser::new(text);
        let m = p.parse_module().unwrap();
        assert_eq!(m.name, "m");
        assert_eq!(m.inputs, vec!["i"]);
        assert_eq!(m.outputs, vec!["o"]);
        assert_eq!(m.wires, vec!["G0"]);
        assert_eq!(m.assigns.len(), 2);
    }
}
