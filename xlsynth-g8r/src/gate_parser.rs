// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn, Input, Output};

#[derive(Debug)]
pub struct ParseError {
    msg: String,
}

impl ParseError {
    fn new(msg: String) -> Self {
        Self { msg }
    }

    fn new_with_pos(msg: String, input: &str, pos: usize) -> Self {
        let mut line = 1usize;
        let mut col = 1usize;
        for ch in input[..pos].chars() {
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        Self {
            msg: format!("{} at line {}, column {}", msg, line, col),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParseError: {}", self.msg)
    }
}

use std::collections::HashMap;

struct Parser<'a> {
    input: &'a str,
    pos: usize,
    input_lookup: HashMap<(String, usize), usize>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            input_lookup: HashMap::new(),
        }
    }

    fn rest(&self) -> &'a str {
        &self.input[self.pos..]
    }

    fn drop_ws(&mut self) {
        while let Some(c) = self.rest().chars().next() {
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn try_drop(&mut self, tok: &str) -> bool {
        self.drop_ws();
        if self.rest().starts_with(tok) {
            self.pos += tok.len();
            true
        } else {
            false
        }
    }

    fn err(&self, msg: &str) -> ParseError {
        ParseError::new_with_pos(msg.to_string(), self.input, self.pos)
    }

    fn drop_or_error(&mut self, tok: &str) -> Result<(), ParseError> {
        if self.try_drop(tok) {
            Ok(())
        } else {
            Err(self.err(&format!(
                "expected '{}' got '{}...'",
                tok,
                &self.rest()[..self.rest().len().min(tok.len())]
            )))
        }
    }

    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        self.drop_ws();
        let rest = self.rest();
        let mut chars = rest.chars();
        let mut result = String::new();
        if let Some(c) = chars.next() {
            if c.is_alphabetic() || c == '_' {
                result.push(c);
                self.pos += c.len_utf8();
            } else {
                return Err(self.err(&format!("expected identifier start, got '{}'", c)));
            }
        } else {
            return Err(self.err("unexpected eof"));
        }
        while let Some(c) = self.rest().chars().next() {
            if c.is_alphanumeric() || c == '_' {
                result.push(c);
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        Ok(result)
    }

    fn parse_usize(&mut self) -> Result<usize, ParseError> {
        self.drop_ws();
        let rest = self.rest();
        let mut len = 0;
        for c in rest.chars() {
            if c.is_ascii_digit() {
                len += c.len_utf8();
            } else {
                break;
            }
        }
        if len == 0 {
            return Err(self.err("expected number"));
        }
        let num: usize = rest[..len].parse().unwrap();
        self.pos += len;
        Ok(num)
    }

    fn parse_operand(&mut self) -> Result<AigOperand, ParseError> {
        self.drop_ws();
        let neg = if self.try_drop("not(") { true } else { false };
        self.drop_ws();
        let id = if self.try_drop("%") {
            self.parse_usize()?
        } else {
            let name = self.parse_identifier()?;
            self.drop_or_error("[")?;
            let idx = self.parse_usize()?;
            self.drop_or_error("]")?;
            *self
                .input_lookup
                .get(&(name, idx))
                .ok_or_else(|| ParseError::new("unknown input reference".into()))?
        };
        if neg {
            self.drop_or_error(")")?;
        }
        Ok(AigOperand {
            node: AigRef { id },
            negated: neg,
        })
    }

    fn parse_bit_vector(&mut self) -> Result<AigBitVector, ParseError> {
        self.drop_ws();
        self.drop_or_error("[")?;
        let mut ops = Vec::new();
        if self.try_drop("]") {
            return Ok(AigBitVector::from_lsb_is_index_0(&ops));
        }
        loop {
            let op = self.parse_operand()?;
            ops.push(op);
            if self.try_drop("]") {
                break;
            }
            self.drop_or_error(",")?;
        }
        Ok(AigBitVector::from_lsb_is_index_0(&ops))
    }

    fn parse_io_list(&mut self) -> Result<Vec<(String, AigBitVector)>, ParseError> {
        self.drop_ws();
        self.drop_or_error("(")?;
        if self.try_drop(")") {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        loop {
            let name = self.parse_identifier()?;
            self.drop_or_error(":")?;
            self.drop_or_error("bits[")?;
            let width = self.parse_usize()?;
            self.drop_or_error("]")?;
            self.drop_or_error("=")?;
            let bv = self.parse_bit_vector()?;
            if bv.get_bit_count() != width {
                return Err(ParseError::new("bit count mismatch".to_string()));
            }
            entries.push((name, bv));
            if self.try_drop(")") {
                break;
            }
            self.drop_or_error(",")?;
        }
        Ok(entries)
    }

    fn at_eof(&mut self) -> bool {
        self.drop_ws();
        self.pos >= self.input.len()
    }
}

pub fn parse_gate_fn(text: &str) -> Result<GateFn, ParseError> {
    let mut p = Parser::new(text);
    p.drop_or_error("fn")?;
    let name = p.parse_identifier()?;
    let inputs_pairs = p.parse_io_list()?;
    p.drop_or_error("->")?;
    let outputs_pairs = p.parse_io_list()?;
    p.drop_or_error("{")?;

    use std::collections::HashMap;
    let mut nodes: HashMap<usize, AigNode> = HashMap::new();
    nodes.insert(0, AigNode::Literal(false));

    let mut inputs = Vec::new();
    for (inp_name, bv) in inputs_pairs {
        for (i, op) in bv.iter_lsb_to_msb().enumerate() {
            nodes.insert(
                op.node.id,
                AigNode::Input {
                    name: inp_name.clone(),
                    lsb_index: i,
                },
            );
            p.input_lookup.insert((inp_name.clone(), i), op.node.id);
        }
        inputs.push(Input {
            name: inp_name,
            bit_vector: bv,
        });
    }

    while !p.at_eof() {
        if p.try_drop("}") {
            break;
        }
        p.drop_ws();
        if p.rest().starts_with('%') {
            p.drop_or_error("%")?;
            let id = p.parse_usize()?;
            p.drop_or_error("=")?;
            if p.try_drop("and(") {
                let a = p.parse_operand()?;
                p.drop_or_error(",")?;
                let b = p.parse_operand()?;
                let tags = if p.try_drop(",") {
                    p.drop_or_error("tags=[")?;
                    let mut tag_vec = Vec::new();
                    if !p.try_drop("]") {
                        loop {
                            let tag = p.parse_identifier()?;
                            tag_vec.push(tag);
                            if p.try_drop("]") {
                                break;
                            }
                            p.drop_or_error(",")?;
                        }
                    }
                    p.drop_or_error(")")?;
                    Some(tag_vec)
                } else {
                    p.drop_or_error(")")?;
                    None
                };
                nodes.insert(id, AigNode::And2 { a, b, tags });
                continue;
            } else if p.try_drop("literal(") {
                p.drop_ws();
                let lit_val = if p.try_drop("true") {
                    true
                } else if p.try_drop("false") {
                    false
                } else {
                    let value = p.parse_usize()?;
                    value != 0
                };
                p.drop_or_error(")")?;
                nodes.insert(id, AigNode::Literal(lit_val));
                continue;
            } else {
                return Err(p.err("unknown node kind"));
            }
        } else {
            let _name = p.parse_identifier()?;
            p.drop_or_error("[")?;
            p.parse_usize()?;
            p.drop_or_error("]")?;
            p.drop_or_error("=")?;
            let _ = p.parse_operand()?;
            continue;
        }
    }

    let mut outputs = Vec::new();
    for (name, bv) in outputs_pairs {
        outputs.push(Output {
            name,
            bit_vector: bv,
        });
    }

    let max_id = nodes.keys().copied().max().unwrap_or(0);
    let mut gates = Vec::new();
    for id in 0..=max_id {
        if let Some(node) = nodes.remove(&id) {
            gates.push(node);
        } else {
            return Err(ParseError::new(format!("missing node id {}", id)));
        }
    }

    Ok(GateFn {
        name,
        inputs,
        outputs,
        gates,
    })
}

impl GateFn {
    pub fn from_str(text: &str) -> Result<Self, ParseError> {
        parse_gate_fn(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{setup_simple_graph, structurally_equivalent};

    #[test]
    fn test_round_trip_simple() {
        let g = setup_simple_graph().g;
        let text = g.to_string();
        let parsed = GateFn::from_str(&text).unwrap();
        assert!(structurally_equivalent(&g, &parsed));
    }

    #[test]
    fn test_parse_bool_literal() {
        let src = "fn t() -> (o: bits[1]=[%0]) { %0 = literal(true) }";
        let _g = GateFn::from_str(src).unwrap();
        let src2 = "fn t() -> (o: bits[1]=[%0]) { %0 = literal(false) }";
        let _ = GateFn::from_str(src2).unwrap();
    }

    #[test]
    fn test_round_trip_constant_replace_sample() {
        use crate::test_utils::setup_graph_for_constant_replace;

        let g = setup_graph_for_constant_replace().g;
        let text = g.to_string();
        let parsed = GateFn::from_str(&text).unwrap();
        assert!(structurally_equivalent(&g, &parsed));
    }
}
