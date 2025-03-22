// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, PartialEq)]
enum Value {
    String(String),
    Number(f64),
    Identifier(String),
    Tuple(Vec<Box<Value>>),
}

#[derive(Debug, PartialEq)]
struct BlockAttr {
    attr_name: String,
    value: Value,
}

#[derive(Debug, PartialEq)]
enum BlockMember {
    BlockAttr(BlockAttr),
    SubBlock(Box<Block>),
}

#[derive(Debug, PartialEq)]
struct Block {
    block_type: String,
    name: String,
    members: Vec<BlockMember>,
}

struct LibertyParser {
    chars: Vec<char>,
    pos: usize,
}

impl LibertyParser {
    pub fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
            pos: 0,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn skip_comment(&mut self) -> Result<(), String> {
        if self.peek_is_noskip("/*")? {
            self.pos += 2;
            while !self.peek_is_noskip("*/")? {
                self.pos += 1;
                if self.pos >= self.chars.len() {
                    return Err("Unterminated comment".to_string());
                }
            }
            self.pos += 2;
            Ok(())
        } else {
            Ok(())
        }
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<(), String> {
        loop {
            let start_pos = self.pos;
            self.skip_whitespace();
            self.skip_comment()?;
            if self.pos == start_pos {
                return Ok(());
            }
        }
    }

    /// Note: skips whitespace before attempting to pop the value, as whitespace
    /// insensitivity is the most common case in liberty file reading.
    fn pop_or_error(&mut self, expected: &str, context: &str) -> Result<(), String> {
        if self.peek_is(expected)? {
            self.pos += expected.len();
            Ok(())
        } else {
            Err(format!(
                "Expected: {:?} at {}, rest: {:?}",
                expected,
                context,
                self.peek_line()
            ))
        }
    }

    fn peek_is_noskip(&mut self, expected: &str) -> Result<bool, String> {
        for (i, c) in expected.chars().enumerate() {
            if self.chars[self.pos + i] != c {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn peek_is(&mut self, expected: &str) -> Result<bool, String> {
        self.skip_whitespace_and_comments()?;
        self.peek_is_noskip(expected)
    }

    fn try_pop(&mut self, expected: &str) -> Result<bool, String> {
        if self.peek_is(expected)? {
            self.pos += expected.len();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn peek_line(&mut self) -> String {
        let mut line = String::new();
        let mut pos = self.pos;
        while pos < self.chars.len() && self.chars[pos] != '\n' {
            line.push(self.chars[pos]);
            pos += 1;
        }
        line
    }

    fn pop_identifier_or_error(&mut self, context: &str) -> Result<String, String> {
        self.skip_whitespace_and_comments()?;
        let mut chars = String::new();
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c.is_alphanumeric() || c == '_' {
                chars.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }
        if chars.is_empty() {
            Err(format!(
                "Expected identifier in {}; rest: {:?}",
                context,
                self.peek_line()
            ))
        } else {
            Ok(chars)
        }
    }

    fn pop_number(&mut self) -> Result<f64, String> {
        self.skip_whitespace_and_comments()?;
        let mut chars = String::new();
        let mut saw_dot = false;
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c.is_digit(10) {
                chars.push(c);
                self.pos += 1;
            } else if c == '.' {
                if saw_dot {
                    return Err("Multiple dots in number".to_string());
                }
                saw_dot = true;
                chars.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(chars.parse::<f64>().unwrap())
    }

    fn pop_string(&mut self) -> Result<String, String> {
        self.skip_whitespace_and_comments()?;
        self.pop_or_error("\"", "string value start")?;
        let mut chars = String::new();
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c == '"' {
                self.pos += 1;
                break;
            } else {
                chars.push(c);
                self.pos += 1;
            }
        }
        Ok(chars)
    }

    fn peek_is_numeric(&mut self) -> Result<bool, String> {
        self.skip_whitespace_and_comments()?;
        Ok(self.chars[self.pos].is_digit(10))
    }

    fn pop_value(&mut self) -> Result<Value, String> {
        if self.peek_is("\"")? {
            Ok(Value::String(self.pop_string()?))
        } else if self.peek_is_numeric()? {
            Ok(Value::Number(self.pop_number()?))
        } else {
            Ok(Value::Identifier(self.pop_identifier_or_error("value")?))
        }
    }

    fn pop_tuple_rest(&mut self, leading_value: Value) -> Result<Value, String> {
        let mut values = Vec::new();
        values.push(Box::new(leading_value));
        while !self.peek_is(")")? {
            let value = self.pop_value()?;
            values.push(Box::new(value));
        }
        self.pop_or_error(")", "tuple value end")?;
        Ok(Value::Tuple(values))
    }

    fn parse_block_member(&mut self) -> Result<BlockMember, String> {
        let attr_name = self.pop_identifier_or_error("attribute name")?;
        log::info!("attr_name: {:?}", attr_name);
        if self.try_pop(":")? {
            let value = self.pop_value()?;
            self.pop_or_error(";", "attribute value end")?;
            Ok(BlockMember::BlockAttr(BlockAttr { attr_name, value }))
        } else {
            log::info!(
                "no colon implies block- or tuple-like attribute; rest: {:?}",
                self.peek_line()
            );
            self.pop_or_error("(", "block-like attribute start")?;
            let leading_value = self.pop_value()?;
            if self.try_pop(",")? {
                // Tuple-like attribute.
                let value = self.pop_tuple_rest(leading_value)?;
                self.pop_or_error(";", "tuple-like attribute value end")?;
                Ok(BlockMember::BlockAttr(BlockAttr { attr_name, value }))
            } else {
                self.pop_or_error(")", "block-like attribute end")?;
                let Value::Identifier(name) = leading_value else {
                    return Err(format!(
                        "Expected identifier in block-looking block member; got {:?}; rest: {:?}",
                        leading_value,
                        self.peek_line()
                    ));
                };
                let block = self.parse_block_with_type_and_name(attr_name, name)?;
                Ok(BlockMember::SubBlock(Box::new(block)))
            }
        }
    }

    fn parse_block_with_type_and_name(
        &mut self,
        block_type: String,
        name: String,
    ) -> Result<Block, String> {
        self.pop_or_error("{", "block body start")?;
        let mut members = Vec::new();
        while !self.peek_is("}")? {
            let member = self.parse_block_member()?;
            members.push(member);
        }
        self.pop_or_error("}", "block body end")?;
        Ok(Block {
            block_type,
            name,
            members,
        })
    }

    fn parse_block_with_type(&mut self, block_type: String) -> Result<Block, String> {
        self.pop_or_error("(", "block name start")?;
        let name = self.pop_identifier_or_error("block name")?;
        self.pop_or_error(")", "block name end")?;
        self.parse_block_with_type_and_name(block_type, name)
    }

    fn parse_block(&mut self) -> Result<Block, String> {
        let block_type = self.pop_identifier_or_error("block type")?;
        self.parse_block_with_type(block_type)
    }

    pub fn parse(&mut self) -> Result<Block, String> {
        self.parse_block()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fake_liberty_file() {
        let _ = env_logger::builder().is_test(true).try_init();
        let text = r#"
        library (my_library) {
            cell (my_and) {
                area: 1.0;
                pin (Y) {
                    direction: output;
                    function: "(A * B)";
                    max_capacitance: 12.34;
                }
                pin (A) {
                    direction: input;
                    capacitance: 0.1;
                    rise_capacitance_range (0.1, 0.2);
                }
                pin (B) {
                    direction: input;
                    capacitance: 0.2;
                    rise_capacitance_range (0.2, 0.3);
                }
            }
        }
        "#;
        let mut parser = LibertyParser::new(text);
        let library = parser.parse().unwrap();
        assert_eq!(library.block_type, "library");
        assert_eq!(library.name, "my_library");
        assert_eq!(library.members.len(), 1);
        let cell = match &library.members[0] {
            BlockMember::SubBlock(sub_block) => sub_block,
            _ => panic!("Expected sub_block"),
        };
        assert_eq!(cell.block_type, "cell");
        assert_eq!(cell.name, "my_and");
        assert_eq!(cell.members.len(), 4);
        let area = match &cell.members[0] {
            BlockMember::BlockAttr(block_attr) => block_attr,
            _ => panic!("Expected block_attr"),
        };
        assert_eq!(area.attr_name, "area");
        assert_eq!(area.value, Value::Number(1.0));
        let pin_y = match &cell.members[1] {
            BlockMember::SubBlock(sub_block) => sub_block,
            _ => panic!("Expected sub_block"),
        };
        assert_eq!(pin_y.block_type, "pin");
        assert_eq!(pin_y.name, "Y");
        assert_eq!(pin_y.members.len(), 3);
        let direction = match &pin_y.members[0] {
            BlockMember::BlockAttr(block_attr) => block_attr,
            _ => panic!("Expected block_attr"),
        };
        assert_eq!(direction.attr_name, "direction");
        assert_eq!(direction.value, Value::Identifier("output".to_string()));
        let function = match &pin_y.members[1] {
            BlockMember::BlockAttr(block_attr) => block_attr,
            _ => panic!("Expected block_attr"),
        };
        assert_eq!(function.attr_name, "function");
        assert_eq!(function.value, Value::String("(A * B)".to_string()));
        let max_capacitance = match &pin_y.members[2] {
            BlockMember::BlockAttr(block_attr) => block_attr,
            _ => panic!("Expected block_attr"),
        };
        assert_eq!(max_capacitance.attr_name, "max_capacitance");
        assert_eq!(max_capacitance.value, Value::Number(12.34));
    }

    #[test]
    fn test_parse_real_liberty_file() {
        let _ = env_logger::builder().is_test(true).try_init();
        let path = "/home/cdleary/src/asap7/asap7sc7p5t_SIMPLE_RVT_SS_nldm_211120.lib";
        let text = std::fs::read_to_string(path).unwrap();
        let mut parser = LibertyParser::new(&text);
        let library = parser.parse().unwrap();
        println!("{:?}", library);
    }
}
