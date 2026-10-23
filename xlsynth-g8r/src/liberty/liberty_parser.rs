// SPDX-License-Identifier: Apache-2.0

use std::io::BufReader;
use std::io::Read;
use std::iter::Iterator;

use super::ascii_stream::AsciiStream;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    String(String),
    Number(f64),
    Identifier(String),
    Tuple(Vec<Box<Value>>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockAttr {
    pub attr_name: String,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BlockMember {
    BlockAttr(BlockAttr),
    SubBlock(Box<Block>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub block_type: String,
    // Note: blocks can have no qualifiers; e.g. `leakage_power() { ... }`
    // In that case, it has a `when` attribute that indicates the boolean-valued situation when the
    // information is applicable.
    //
    // Many blocks in practice have a single qualifier that is an identifier, like `cell(BLAH) {
    // ... }`.
    pub qualifiers: Vec<Value>,
    pub members: Vec<BlockMember>,
}

/// Distinguishes syntax failures from errors returned by a root-member visitor.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ParseWithVisitorError {
    Syntax(String),
    Visitor(String),
}

impl From<String> for ParseWithVisitorError {
    fn from(error: String) -> Self {
        Self::Syntax(error)
    }
}

// LibertyParser owns an AsciiStream (instead of managing byte operations
// itself).
pub struct LibertyParser<I: Iterator<Item = u8>> {
    stream: AsciiStream<I>,
}

impl<I: Iterator<Item = u8>> LibertyParser<I> {
    pub fn new_from_iter(iter: I) -> Self {
        Self {
            stream: AsciiStream::new(iter),
        }
    }

    fn pop_value(&mut self, context: &str) -> Result<Value, String> {
        if self.stream.peek_is(b"\"")? {
            Ok(Value::String(self.stream.pop_string()?))
        } else if self.stream.peek_is_numeric()? || self.stream.peek_is(b"-")? {
            Ok(Value::Number(self.stream.pop_number()?))
        } else {
            Ok(Value::Identifier(
                self.stream.pop_identifier_or_error(context)?,
            ))
        }
    }

    fn parse_block_member<F>(
        &mut self,
        parent_type: &str,
        retain: bool,
        retain_subblock: &F,
    ) -> Result<Option<BlockMember>, String>
    where
        F: Fn(&str, &str) -> bool,
    {
        let attr_name = self.stream.pop_identifier_or_error("attribute name")?;
        log::trace!("attr_name: {:?}", attr_name);
        if self.stream.try_pop(b":")? {
            let value = self.pop_value("attribute value")?;
            self.stream.pop_semi_or_newline("attribute value end")?;
            Ok(retain.then_some(BlockMember::BlockAttr(BlockAttr { attr_name, value })))
        } else {
            log::trace!(
                "no colon implies block- or tuple-like attribute; name: {:?} rest: {:?}",
                attr_name,
                self.stream.peek_line()
            );
            self.stream
                .pop_or_error(b"(", "block-like attribute start")?;
            let mut qualifiers = Vec::new();
            while !self.stream.peek_is(b")")? {
                let value = self.pop_value("block-like attribute qualifier")?;
                qualifiers.push(value);
                if !self.stream.try_pop(b",")? {
                    break;
                }
            }
            self.stream.pop_or_error(b")", "block-like attribute end")?;

            // There are a few cases we're interested in:
            // * We see a `{` on this line, in which case we want to start
            //   parsing a sub-block.
            // * We see a `;` on this line, in which case we're done parsing
            //   this member.
            // * We see nothing else meaningful (just whitespace and comments)
            //   until `\n` (EOL), in which case we do automatic semicolon
            //   insertion and consider this member done.
            let next: Option<u8> = self.stream.peek_char_or_eol()?;
            match next {
                None => Err(format!(
                    "Unexpected end of file parsing block member @ {}",
                    self.stream.human_pos()
                )),
                Some(b'{') => {
                    let retain_child = retain && retain_subblock(parent_type, &attr_name);
                    let block = self.parse_block_with_type_and_qualifiers(
                        attr_name,
                        qualifiers,
                        retain_child,
                        retain_subblock,
                    )?;
                    Ok(block.map(|block| BlockMember::SubBlock(Box::new(block))))
                }
                Some(b';') => {
                    assert!(self.stream.try_pop(b";")?);
                    Ok(retain.then(|| {
                        BlockMember::BlockAttr(BlockAttr {
                            attr_name,
                            value: Value::Tuple(qualifiers.into_iter().map(Box::new).collect()),
                        })
                    }))
                }
                Some(b'\n') => {
                    // Automatic semicolon insertion.
                    assert!(self.stream.try_consume(1));
                    Ok(retain.then(|| {
                        BlockMember::BlockAttr(BlockAttr {
                            attr_name,
                            value: Value::Tuple(qualifiers.into_iter().map(Box::new).collect()),
                        })
                    }))
                }
                Some(c) => Err(format!(
                    "Unexpected character: {:?} @ {}",
                    c as char,
                    self.stream.human_pos()
                )),
            }
        }
    }

    fn parse_block_with_type_and_qualifiers<F>(
        &mut self,
        block_type: String,
        qualifiers: Vec<Value>,
        retain: bool,
        retain_subblock: &F,
    ) -> Result<Option<Block>, String>
    where
        F: Fn(&str, &str) -> bool,
    {
        self.stream.pop_or_error(b"{", "block body start")?;
        let mut members = Vec::new();
        while !self.stream.peek_is(b"}")? {
            if let Some(member) = self.parse_block_member(&block_type, retain, retain_subblock)? {
                members.push(member);
            }
        }
        self.stream.pop_or_error(b"}", "block body end")?;
        Ok(retain.then_some(Block {
            block_type,
            qualifiers,
            members,
        }))
    }

    /// Parses the root block while allowing direct members to be consumed as
    /// they arrive.
    ///
    /// A false `retain_subblock` result discards that subtree after checking
    /// its syntax. The visitor may consume each retained direct member by
    /// returning `None`; only returned members are accumulated in the root
    /// block.
    pub(crate) fn parse_with_member_visitor<F, V>(
        &mut self,
        retain_subblock: F,
        mut visitor: V,
    ) -> Result<Block, ParseWithVisitorError>
    where
        F: Fn(&str, &str) -> bool,
        V: FnMut(BlockMember) -> Result<Option<BlockMember>, String>,
    {
        let block_type = self.stream.pop_identifier_or_error("block type")?;
        self.stream.pop_or_error(b"(", "block name start")?;
        let mut qualifiers = Vec::new();
        while !self.stream.peek_is(b")")? {
            let value = self.pop_value("block qualifier")?;
            qualifiers.push(value);
            if !self.stream.try_pop(b",")? {
                break;
            }
        }
        self.stream.pop_or_error(b")", "block name end")?;
        self.stream.pop_or_error(b"{", "block body start")?;
        let mut members = Vec::new();
        while !self.stream.peek_is(b"}")? {
            if let Some(member) = self.parse_block_member(&block_type, true, &retain_subblock)?
                && let Some(member) = visitor(member).map_err(ParseWithVisitorError::Visitor)?
            {
                members.push(member);
            }
        }
        self.stream.pop_or_error(b"}", "block body end")?;
        Ok(Block {
            block_type,
            qualifiers,
            members,
        })
    }

    pub fn parse(&mut self) -> Result<Block, String> {
        self.parse_with_member_visitor(|_, _| true, |member| Ok(Some(member)))
            .map_err(|error| match error {
                ParseWithVisitorError::Syntax(error) | ParseWithVisitorError::Visitor(error) => {
                    error
                }
            })
    }
}

pub struct CharReader<R: Read> {
    iter: std::io::Bytes<BufReader<R>>,
}

impl<R: Read> CharReader<R> {
    pub fn new(reader: R) -> Self {
        let buf_reader = BufReader::new(reader);
        Self {
            iter: buf_reader.bytes(),
        }
    }
}

impl<R: Read> Iterator for CharReader<R> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        self.iter.next().and_then(|res| res.ok())
    }
}

#[cfg(test)]
mod tests {
    use flate2::Compression;
    use flate2::{read::GzDecoder, write::GzEncoder};
    use std::io::Write;

    use super::*;

    #[test]
    fn test_parse_fake_liberty_file() {
        let _ = env_logger::builder().is_test(true).try_init();
        let text = br#"
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
                    some_table (some_template) {
                        index_1 ("1, 2");
                        values ( \
                            "3.0, 4.0" \
                        );
                    }
                }
            }
        }
        "#;
        let mut parser = LibertyParser::new_from_iter(text.bytes().map(|b| b.unwrap()));
        let library = parser.parse().unwrap();
        assert_eq!(library.block_type, "library");
        assert_eq!(
            library.qualifiers,
            vec![Value::Identifier("my_library".to_string())]
        );
        assert_eq!(library.members.len(), 1);
        let cell = match &library.members[0] {
            BlockMember::SubBlock(sub_block) => sub_block,
            _ => panic!("Expected sub_block"),
        };
        assert_eq!(cell.block_type, "cell");
        assert_eq!(
            cell.qualifiers,
            vec![Value::Identifier("my_and".to_string())]
        );
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
        assert_eq!(pin_y.qualifiers, vec![Value::Identifier("Y".to_string())]);
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
    fn test_parse_whitespace_terminated_attributes() {
        let text = br#"
        library (my_library) {
            cell (my_cell) {
                area: 1.0
                goodness : high ;
            }
        }
        "#;
        let mut parser = LibertyParser::new_from_iter(text.bytes().map(|b| b.unwrap()));
        let library = parser.parse().unwrap();
        assert_eq!(library.block_type, "library");
        assert_eq!(
            library.qualifiers,
            vec![Value::Identifier("my_library".to_string())]
        );
        assert_eq!(library.members.len(), 1);
    }

    #[test]
    fn test_block_with_string_name_instead_of_identifier() {
        let text = br#"
        my_thing("stuff goes here") {
            my_attribute: "wow";
        }
        "#;
        let mut parser = LibertyParser::new_from_iter(text.bytes().map(|b| b.unwrap()));
        let library = parser.parse().unwrap();
        assert_eq!(library.block_type, "my_thing");
        assert_eq!(
            library.qualifiers,
            vec![Value::String("stuff goes here".to_string())]
        );
        assert_eq!(library.members.len(), 1);
    }

    #[test]
    fn test_parse_number_value_scientific_notation() {
        let text = br#"
        some_block(my_name) {
            some_attribute: 1.0e-12;
        }
        "#;
        let mut parser = LibertyParser::new_from_iter(text.bytes().map(|b| b.unwrap()));
        let library = parser.parse().unwrap();
        assert_eq!(library.block_type, "some_block");
        assert_eq!(
            library.qualifiers,
            vec![Value::Identifier("my_name".to_string())]
        );
        assert_eq!(library.members.len(), 1);
        let some_attribute = match &library.members[0] {
            BlockMember::BlockAttr(block_attr) => block_attr,
            _ => panic!("Expected block_attr"),
        };
        assert_eq!(some_attribute.attr_name, "some_attribute");
        assert_eq!(some_attribute.value, Value::Number(1.0e-12));
    }

    // This test compresses some sample text into gzipped form and then shows it
    // streaming into the parser.
    #[test]
    fn test_parse_gzipped_file() {
        let text = br#"
        library (my_library) {
            cell (my_and) {
                area: 1.0;
            }
        }
        "#;
        // gzip the text
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(text).unwrap();
        let compressed = encoder.finish().unwrap();

        // Now we make a char streamer that reads from the gzipped data.
        let streamer = GzDecoder::new(compressed.as_slice());
        let char_reader = CharReader::new(streamer);
        let mut parser = LibertyParser::new_from_iter(char_reader);
        let library = parser.parse().unwrap();
        assert_eq!(library.block_type, "library");
        assert_eq!(
            library.qualifiers,
            vec![Value::Identifier("my_library".to_string())]
        );
    }

    #[test]
    fn test_block_with_multiple_qualifiers_nested() {
        let text = br#"
        outer_block(outer_qualifier,multiple_qualifiers) {
            my_block(stuff,here) {
                my_attribute: 42.0;
            }
        }
        "#;
        let mut parser = LibertyParser::new_from_iter(text.bytes().map(|b| b.unwrap()));
        let block = parser.parse().unwrap();
        assert_eq!(block.block_type, "outer_block");
        assert_eq!(
            block.qualifiers,
            vec![
                Value::Identifier("outer_qualifier".to_string()),
                Value::Identifier("multiple_qualifiers".to_string())
            ]
        );
        assert_eq!(block.members.len(), 1);
        let my_block = match &block.members[0] {
            BlockMember::SubBlock(sub_block) => sub_block,
            _ => panic!("Expected sub_block"),
        };
        assert_eq!(my_block.block_type, "my_block");
        assert_eq!(
            my_block.qualifiers,
            vec![
                Value::Identifier("stuff".to_string()),
                Value::Identifier("here".to_string())
            ]
        );
        let my_attribute = match &my_block.members[0] {
            BlockMember::BlockAttr(block_attr) => block_attr,
            _ => panic!("Expected block_attr"),
        };
        assert_eq!(my_attribute.attr_name, "my_attribute");
        assert_eq!(my_attribute.value, Value::Number(42.0));
    }

    #[test]
    fn test_member_visitor_consumes_cells_and_discards_selected_subtrees() {
        let text = r#"
        library (example) {
            time_unit : "1ns";
            cell (BUF) {
                pin (A) { direction : input; }
                pin (Y) {
                    direction : output;
                    function : "A";
                    timing () {
                        related_pin : "A";
                        cell_rise (scalar) { values ("0.1"); }
                    }
                }
            }
            cell (INV) { area : 1.0; }
        }
        "#;
        let mut parser = LibertyParser::new_from_iter(text.bytes());
        let mut cells = Vec::new();
        let root = parser
            .parse_with_member_visitor(
                |parent, child| parent != "pin" || child != "timing",
                |member| match member {
                    BlockMember::SubBlock(block) if block.block_type == "cell" => {
                        cells.push(*block);
                        Ok(None)
                    }
                    member => Ok(Some(member)),
                },
            )
            .unwrap();
        assert_eq!(root.members.len(), 1);
        assert_eq!(cells.len(), 2);
        let BlockMember::SubBlock(output) = &cells[0].members[1] else {
            panic!("expected output pin");
        };
        assert!(output.members.iter().all(|member| {
            !matches!(member, BlockMember::SubBlock(block) if block.block_type == "timing")
        }));

        let mut full_parser = LibertyParser::new_from_iter(text.bytes());
        let full = full_parser.parse().unwrap();
        assert_eq!(full.members.len(), 3);
        let BlockMember::SubBlock(full_cell) = &full.members[1] else {
            panic!("expected full cell");
        };
        let BlockMember::SubBlock(full_output) = &full_cell.members[1] else {
            panic!("expected full output pin");
        };
        assert!(full_output.members.iter().any(|member| {
            matches!(member, BlockMember::SubBlock(block) if block.block_type == "timing")
        }));
    }

    #[test]
    fn test_discarded_subtree_still_requires_valid_syntax() {
        let text = r#"
        library (example) {
            cell (BUF) {
                pin (Y) {
                    timing () { related_pin : ; }
                }
            }
        }
        "#;
        let mut parser = LibertyParser::new_from_iter(text.bytes());
        let result = parser.parse_with_member_visitor(
            |parent, child| parent != "pin" || child != "timing",
            |member| Ok(Some(member)),
        );
        assert!(matches!(result, Err(ParseWithVisitorError::Syntax(_))));
    }

    #[test]
    fn test_member_visitor_error_is_distinct_from_syntax_error() {
        let text = "library (example) { cell (BUF) {} }";
        let mut parser = LibertyParser::new_from_iter(text.bytes());
        let result =
            parser.parse_with_member_visitor(|_, _| true, |_| Err("visitor failure".to_string()));
        assert_eq!(
            result,
            Err(ParseWithVisitorError::Visitor(
                "visitor failure".to_string()
            ))
        );
    }
}
