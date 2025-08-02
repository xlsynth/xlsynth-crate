// SPDX-License-Identifier: Apache-2.0

//! Parser for XLS IR (just functions for the time being).

use crate::xls_ir::ir::{self, ArrayTypeData, FileTable, operator_to_nary_op};
use crate::xls_ir::ir_node_env::{IrNodeEnv, NameOrId};

pub fn parse_path_to_package(path: &std::path::Path) -> Result<ir::Package, ParseError> {
    let file_content = std::fs::read_to_string(path)
        .map_err(|e| ParseError::new(format!("failed to read file: {}", e)))?;
    let mut parser = Parser::new(&file_content);
    parser.parse_package()
}

#[derive(Debug)]
pub struct ParseError {
    msg: String,
}

impl ParseError {
    fn new(msg: String) -> Self {
        Self { msg }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParseError: {}", self.msg)
    }
}

impl FileTable {
    fn add(&mut self, id: usize, path: String) -> Result<(), ParseError> {
        match self.id_to_path.try_insert(id, path) {
            Err(std::collections::hash_map::OccupiedError { .. }) => Err(ParseError::new(format!(
                "file number {} already exists",
                id
            ))),
            Ok(_) => Ok(()),
        }
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
        self.chars[self.offset..].iter().collect::<String>()
    }

    fn rest_of_line(&self) -> String {
        let rest = self.rest();
        if let Some(pos) = rest.find('\n') {
            rest[..pos].to_string()
        } else {
            rest
        }
    }

    fn current_line(&self) -> String {
        let mut start = 0;
        for i in (0..self.offset).rev() {
            if self.chars[i] == '\n' {
                start = i + 1;
                break;
            }
        }
        let mut end = self.chars.len();
        for i in self.offset..self.chars.len() {
            if self.chars[i] == '\n' {
                end = i;
                break;
            }
        }
        self.chars[start..end].iter().collect::<String>()
    }

    fn at_eof(&mut self) -> bool {
        self.drop_whitespace_and_comments();
        self.offset >= self.chars.len()
    }

    /// Drops a "//" style comment if one is present at the current
    /// offset. Returns `true` if a comment was found and removed, `false`
    /// otherwise.
    fn drop_comment(&mut self) -> bool {
        if self.peek_is("//") {
            // Consume the two '/' characters.
            self.dropc().unwrap();
            self.dropc().unwrap();
            // Skip until end of line or EOF.
            while let Some(c) = self.peekc() {
                self.dropc().unwrap();
                if c == '\n' {
                    break;
                }
            }
            true
        } else {
            false
        }
    }

    /// Repeatedly removes whitespace and "//" comments so that parsing can
    /// resume at the next significant token.
    fn drop_whitespace_and_comments(&mut self) {
        loop {
            // First remove any leading whitespace.
            self.drop_whitespace();
            // Then, if a comment follows, remove it and iterate to drop any
            // whitespace that may come after the comment's terminating newline.
            if !self.drop_comment() {
                break;
            }
        }
    }

    fn drop_whitespace(&mut self) {
        // Consume consecutive whitespace characters.
        while let Some(c) = self.peekc() {
            if c.is_whitespace() {
                // Safe to unwrap because we just peeked a character.
                self.dropc().unwrap();
            } else {
                break;
            }
        }
    }

    fn peekc(&self) -> Option<char> {
        self.chars.get(self.offset).copied()
    }

    fn popc(&mut self) -> Option<char> {
        let c = self.peekc();
        self.offset += 1;
        c
    }

    fn dropc(&mut self) -> Result<(), ParseError> {
        if let Some(_c) = self.popc() {
            Ok(())
        } else {
            Err(ParseError::new("expected character".to_string()))
        }
    }

    fn pop_identifier_or_error(&mut self, ctx: &str) -> Result<String, ParseError> {
        self.drop_whitespace_and_comments();
        let mut identifier = String::new();
        while let Some(c) = self.peekc() {
            if identifier.is_empty() {
                let is_valid_start = c.is_alphabetic() || c == '_';
                if !is_valid_start {
                    return Err(ParseError::new(format!(
                        "in {} expected identifier, got {:?}; rest_of_line: {:?}",
                        ctx,
                        c,
                        self.rest_of_line()
                    )));
                }
                self.dropc()?;
                identifier.push(c);
            } else {
                let is_valid_rest = c.is_alphanumeric() || c == '_';
                if !is_valid_rest {
                    return Ok(identifier);
                }
                self.dropc()?;
                identifier.push(c);
            }
        }
        Ok(identifier)
    }

    fn pop_string_or_error(&mut self) -> Result<String, ParseError> {
        self.drop_whitespace_and_comments();
        self.drop_or_error("\"")?;
        let mut string = String::new();
        while let Some(c) = self.peekc() {
            if c == '"' {
                self.dropc()?;
                break;
            }
            string.push(c);
            self.dropc()?;
        }
        Ok(string)
    }

    fn pop_number_string_or_error(&mut self, ctx: &str) -> Result<String, ParseError> {
        self.drop_whitespace_and_comments();
        let mut number = String::new();

        // Handle radix prefixes.
        if self.peek_is("0x") || self.peek_is("0X") {
            number.push(self.popc().unwrap());
            number.push(self.popc().unwrap());
            while let Some(c) = self.peekc() {
                if c.is_ascii_hexdigit() {
                    number.push(c);
                    self.popc();
                } else if c == '_' {
                    self.popc();
                } else {
                    break;
                }
            }
        } else if self.peek_is("0b") || self.peek_is("0B") {
            number.push(self.popc().unwrap());
            number.push(self.popc().unwrap());
            while let Some(c) = self.peekc() {
                if c == '0' || c == '1' {
                    number.push(c);
                    self.popc();
                } else if c == '_' {
                    self.popc();
                } else {
                    break;
                }
            }
        } else {
            while let Some(c) = self.peekc() {
                if c.is_ascii_digit() {
                    number.push(c);
                    self.popc();
                } else if c == '_' {
                    self.popc();
                } else {
                    break;
                }
            }
        }

        if number.is_empty() {
            Err(ParseError::new(format!(
                "expected number in {}; rest_of_line: {:?}",
                ctx,
                self.rest_of_line()
            )))
        } else {
            Ok(number)
        }
    }

    fn pop_number_usize_or_error(&mut self, ctx: &str) -> Result<usize, ParseError> {
        let number = self.pop_number_string_or_error(ctx)?;
        Ok(number.parse::<usize>().unwrap())
    }

    fn pop_bits_value_or_error(
        &mut self,
        ty: &ir::Type,
        ctx: &str,
    ) -> Result<xlsynth::IrValue, ParseError> {
        let value = self.pop_number_string_or_error(ctx)?;
        Ok(xlsynth::IrValue::parse_typed(&format!("{}:{}", ty, value)).unwrap())
    }

    fn peek_is(&self, s: &str) -> bool {
        for (i, c) in s.chars().enumerate() {
            let char_index = self.offset + i;
            if char_index >= self.chars.len() {
                return false;
            }
            if self.chars[char_index] != c {
                return false;
            }
        }
        true
    }

    fn try_drop(&mut self, s: &str) -> bool {
        self.drop_whitespace_and_comments();
        if self.peek_is(s) {
            self.offset += s.len();
            true
        } else {
            false
        }
    }

    fn drop_or_error_with_ctx(&mut self, s: &str, ctx: &str) -> Result<(), ParseError> {
        self.drop_whitespace_and_comments();
        if self.try_drop(s) {
            Ok(())
        } else {
            Err(ParseError::new(format!(
                "expected {:?} in {}; line: {:?}",
                s,
                ctx,
                self.current_line()
            )))
        }
    }

    fn drop_or_error(&mut self, s: &str) -> Result<(), ParseError> {
        self.drop_whitespace_and_comments();
        if self.try_drop(s) {
            Ok(())
        } else {
            Err(ParseError::new(format!(
                "expected {:?}; line: {:?}",
                s,
                self.current_line()
            )))
        }
    }

    pub fn parse_type(&mut self) -> Result<ir::Type, ParseError> {
        self.drop_whitespace_and_comments();
        // First, parse the *base* type which may be a tuple, bits, or token.
        // Afterwards, we iteratively wrap the base type in `Array` layers for
        // each `[N]` dimension that follows. This mirrors the textual syntax
        // used by XLS IR where multidimensional array types are expressed as a
        // series of bracket-delimited sizes (e.g. `bits[8][4][2]`).

        let mut ty: ir::Type = if self.try_drop("(") {
            let mut members = Vec::new();
            while !self.try_drop(")") {
                let member = self.parse_type()?;
                members.push(Box::new(member));
                if !self.try_drop(",") {
                    self.drop_or_error(")")?;
                    break;
                }
            }
            ir::Type::Tuple(members)
        } else if self.try_drop("bits") {
            self.drop_or_error("[")?;
            let count = self.pop_number_usize_or_error("bit count")?;
            self.drop_or_error("]")?;
            let mut bits_ty = ir::Type::Bits(count);
            // `bits` types can also have array dimensions directly attached –
            // handle those here before we fall through to the common suffix
            // parsing loop below.
            while self.try_drop("[") {
                let count = self.pop_number_usize_or_error("array type size")?;
                self.drop_or_error("]")?;
                bits_ty = ir::Type::new_array(bits_ty, count);
            }
            bits_ty
        } else if self.try_drop("token") {
            ir::Type::Token
        } else {
            return Err(ParseError::new(format!(
                "unexpected start of type; rest_of_line: {:?}",
                self.rest_of_line()
            )));
        };

        // Handle *additional* array dimensions that may follow the base type –
        // this is required for cases like `((bits[2])[7][2]` where a tuple (or
        // any base type) is wrapped in multiple array dimensions.
        while self.try_drop("[") {
            let count = self.pop_number_usize_or_error("array type size")?;
            self.drop_or_error("]")?;
            ty = ir::Type::Array(ArrayTypeData {
                element_type: Box::new(ty),
                element_count: count,
            });
        }

        Ok(ty)
    }

    fn parse_param(&mut self, default_id: usize) -> Result<ir::Param, ParseError> {
        let name = self.pop_identifier_or_error("parameter")?;
        self.drop_or_error(":")?;
        let ty = self.parse_type()?;
        self.drop_whitespace_and_comments();
        let raw_id: usize = if self.peek_is("id=") {
            self.parse_id_attribute()?
        } else {
            default_id
        };
        let id = ir::ParamId::new(raw_id);
        Ok(ir::Param { name, ty, id })
    }

    pub fn parse_params(&mut self) -> Result<Vec<ir::Param>, ParseError> {
        let mut params = Vec::new();
        self.drop_or_error("(")?;
        loop {
            if self.try_drop(")") {
                break;
            }
            let param = self.parse_param(params.len() + 1)?;
            params.push(param);
            if !self.try_drop(",") {
                self.drop_or_error(")")?;
                break;
            }
        }
        Ok(params)
    }

    fn pop_node_name_or_error(&mut self, ctx: &str) -> Result<NameOrId, ParseError> {
        let name: String = self.pop_identifier_or_error(ctx)?;
        if self.try_drop(".") {
            let id = self.pop_number_usize_or_error(ctx)?;
            Ok(NameOrId::Id(id))
        } else {
            Ok(NameOrId::Name(name))
        }
    }

    fn parse_node_ref(
        &mut self,
        node_env: &IrNodeEnv,
        ctx: &str,
    ) -> Result<ir::NodeRef, ParseError> {
        let name_or_id = self.pop_node_name_or_error(ctx)?;
        let maybe_node_ref = node_env.name_id_to_ref(&name_or_id);
        match maybe_node_ref {
            Some(node_ref) => Ok(*node_ref),
            None => Err(ParseError::new(format!(
                "could not resolve node name_or_id in {}: {:?}; rest_of_line: {:?}; available: {:?}",
                ctx,
                name_or_id,
                self.rest_of_line(),
                node_env.keys(),
            ))),
        }
    }

    fn parse_file_number(&mut self, file_table: &mut FileTable) -> Result<(), ParseError> {
        self.drop_or_error("file_number")?;
        let id = self.pop_number_usize_or_error("file_number")?;
        let path = self.pop_string_or_error()?;
        file_table.add(id, path)
    }

    fn parse_id_attribute(&mut self) -> Result<usize, ParseError> {
        self.drop_or_error("id=")?;
        let id = self.pop_number_usize_or_error("id attribute")?;
        Ok(id)
    }

    fn parse_string_attribute(&mut self, attr_name: &str) -> Result<String, ParseError> {
        self.drop_whitespace_and_comments();
        self.drop_or_error(attr_name)?;
        self.drop_or_error("=")?;
        self.pop_string_or_error()
    }

    fn parse_node_ref_array_attribute(
        &mut self,
        attr_name: &str,
        node_env: &IrNodeEnv,
        ctx: &str,
    ) -> Result<Vec<ir::NodeRef>, ParseError> {
        self.drop_or_error(attr_name)?;
        self.drop_or_error("=")?;
        self.drop_or_error("[")?;
        let mut indices = Vec::new();
        while !self.try_drop("]") {
            let index = self.parse_node_ref(&node_env, ctx)?;
            indices.push(index);
            if !self.try_drop(",") {
                self.drop_or_error("]")?;
                break;
            }
        }
        Ok(indices)
    }

    fn maybe_drop_pos_attribute(
        &mut self,
    ) -> Result<Option<Vec<(usize, usize, usize)>>, ParseError> {
        self.drop_whitespace_and_comments();
        if !self.try_drop(", pos=") {
            return Ok(None);
        }
        let mut pos_attr = Vec::new();
        self.drop_or_error("[")?;
        while !self.try_drop("]") {
            self.drop_or_error("(")?;
            let fileno = self.pop_number_usize_or_error("pos fileno")?;
            self.drop_or_error(",")?;
            let lineno = self.pop_number_usize_or_error("pos lineno")?;
            self.drop_or_error(",")?;
            let colno = self.pop_number_usize_or_error("pos colno")?;
            self.drop_or_error(")")?;
            pos_attr.push((fileno, lineno, colno));
            if !self.try_drop(",") {
                self.drop_or_error("]")?;
                break;
            }
        }
        Ok(Some(pos_attr))
    }

    fn parse_usize_attribute(&mut self, attr_name: &str) -> Result<usize, ParseError> {
        self.drop_or_error(attr_name)?;
        self.drop_or_error("=")?;
        self.pop_number_usize_or_error(&format!("usize attribute: {}", attr_name))
    }

    fn parse_bool_attribute(&mut self, attr_name: &str) -> Result<bool, ParseError> {
        self.drop_or_error(attr_name)?;
        self.drop_or_error("=")?;
        let value = self.pop_identifier_or_error(&format!("bool attribute: {}", attr_name))?;
        if value == "true" {
            Ok(true)
        } else if value == "false" {
            Ok(false)
        } else {
            Err(ParseError::new(format!(
                "expected `true` or `false` for bool attribute {:?}; got {:?}; rest_of_line: {:?}",
                attr_name,
                value,
                self.rest_of_line()
            )))
        }
    }

    fn parse_value_with_ty(
        &mut self,
        ty: &ir::Type,
        ctx: &str,
    ) -> Result<xlsynth::IrValue, ParseError> {
        match ty {
            ir::Type::Bits(_width) => self.pop_bits_value_or_error(&ty, ctx),
            ir::Type::Array(ArrayTypeData { element_type, .. }) => {
                self.drop_or_error_with_ctx("[", "start of array literal")?;
                let mut values = Vec::new();
                while !self.try_drop("]") {
                    let value: xlsynth::IrValue = self.parse_value_with_ty(element_type, ctx)?;
                    values.push(value);
                    if !self.try_drop(",") {
                        self.drop_or_error("]")?;
                        break;
                    }
                }
                Ok(xlsynth::IrValue::make_array(&values).unwrap())
            }
            ir::Type::Tuple(element_tys) => {
                self.drop_or_error("(")?;
                let mut values = Vec::new();
                for (i, element_ty) in element_tys.iter().enumerate() {
                    if i > 0 {
                        self.drop_or_error(",")?;
                    }
                    let element = self.parse_value_with_ty(element_ty, ctx)?;
                    values.push(element);
                }
                self.drop_or_error(")")?;
                Ok(xlsynth::IrValue::make_tuple(&values))
            }
            _ => {
                return Err(ParseError::new(format!(
                    "cannot parse value with type: {}; rest: {:?}",
                    ty,
                    self.rest()
                )));
            }
        }
    }

    fn parse_variadic_op(
        &mut self,
        node_env: &IrNodeEnv,
        maybe_id: &mut Option<usize>,
        ctx: &str,
    ) -> Result<Vec<ir::NodeRef>, ParseError> {
        let mut members = Vec::new();
        loop {
            self.drop_whitespace_and_comments();
            if self.peek_is("id=") {
                break;
            }
            if self.peek_is(")") {
                break;
            }
            let member = self.parse_node_ref(&node_env, &format!("variadic op {:?} arg", ctx))?;
            members.push(member);
            if !self.try_drop(",") {
                break;
            }
        }
        if self.peek_is("id=") {
            *maybe_id = Some(self.parse_id_attribute()?);
        }
        if maybe_id.is_none() {
            return Err(ParseError::new(format!(
                "expected id for tuple; rest: {:?}",
                self.rest()
            )));
        }
        Ok(members)
    }

    fn parse_node(&mut self, node_env: &mut IrNodeEnv) -> Result<ir::Node, ParseError> {
        log::debug!("parse_node");
        let name_or_id = self.pop_node_name_or_error("node name")?;
        let mut maybe_id = match name_or_id {
            NameOrId::Id(id) => Some(id),
            NameOrId::Name(_) => None,
        };
        log::debug!("parse_node; name_or_id: {:?}", name_or_id);
        self.drop_or_error(":")?;
        let node_ty = self.parse_type()?;
        self.drop_or_error("=")?;
        let operator = self.pop_identifier_or_error("node operator")?;
        self.drop_or_error("(")?;

        let (payload, id) = match operator.as_str() {
            "tuple" => {
                let members = self.parse_variadic_op(&node_env, &mut maybe_id, "tuple")?;
                (ir::NodePayload::Tuple(members), maybe_id.unwrap())
            }
            "array" => {
                let members = self.parse_variadic_op(&node_env, &mut maybe_id, "array")?;
                (ir::NodePayload::Array(members), maybe_id.unwrap())
            }
            "array_update" => {
                let array = self.parse_node_ref(&node_env, "array_update array")?;
                self.drop_or_error(",")?;
                let value = self.parse_node_ref(&node_env, "array_update value")?;
                self.drop_or_error(",")?;
                let indices = self.parse_node_ref_array_attribute(
                    "indices",
                    &node_env,
                    "array_update indices",
                )?;
                let mut assumed_in_bounds = false;
                if self.try_drop(",") {
                    self.drop_whitespace_and_comments();
                    if self.peek_is("assumed_in_bounds=") {
                        assumed_in_bounds =
                            self.parse_bool_attribute("assumed_in_bounds")?;
                        if self.try_drop(",") {
                            let id_attr = self.parse_id_attribute()?;
                            maybe_id = Some(id_attr);
                        }
                    } else {
                        let id_attr = self.parse_id_attribute()?;
                        maybe_id = Some(id_attr);
                    }
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for array_update; rest: {:?}",
                        self.rest()
                    )));
                }                (
                    ir::NodePayload::ArrayUpdate {
                        array,
                        value,
                        indices,
                        assumed_in_bounds,
                    },
                    maybe_id.unwrap(),
                )
            }
            "dynamic_bit_slice" => {
                let arg_node = self.parse_node_ref(&node_env, "dynamic_bit_slice arg")?;
                self.drop_or_error(",")?;
                let start_node = self.parse_node_ref(&node_env, "dynamic_bit_slice start")?;
                self.drop_or_error(",")?;
                let width = self.parse_usize_attribute("width")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for dynamic_bit_slice; rest: {:?}",
                        self.rest()
                    )));
                }                (
                    ir::NodePayload::DynamicBitSlice {
                        arg: arg_node,
                        start: start_node,
                        width,
                    },
                    maybe_id.unwrap(),
                )
            }
            "array_index" => {
                let array = self.parse_node_ref(&node_env, "array_index array")?;
                self.drop_or_error(",")?;
                let indices = self.parse_node_ref_array_attribute(
                    "indices",
                    &node_env,
                    "array_index indices",
                )?;
                let mut assumed_in_bounds = false;
                if self.try_drop(",") {
                    self.drop_whitespace_and_comments();
                    if self.peek_is("assumed_in_bounds=") {
                        assumed_in_bounds =
                            self.parse_bool_attribute("assumed_in_bounds")?;
                        if self.try_drop(",") {
                            let id_attr = self.parse_id_attribute()?;
                            maybe_id = Some(id_attr);
                        }
                    } else {
                        let id_attr = self.parse_id_attribute()?;
                        maybe_id = Some(id_attr);
                    }
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for array_index; rest: {:?}",
                        self.rest()
                    )));
                }                (
                    ir::NodePayload::ArrayIndex {
                        array,
                        indices,
                        assumed_in_bounds,
                    },
                    maybe_id.unwrap(),
                )
            }
            "tuple_index" => {
                let tuple = self.parse_node_ref(&node_env, "tuple_index tuple")?;
                self.drop_or_error(",")?;
                let index = self.parse_usize_attribute("index")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for tuple_index; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::TupleIndex { tuple, index },
                    maybe_id.unwrap(),
                )
            }
            "sign_ext" => {
                let arg = self.parse_node_ref(&node_env, "sign_ext arg")?;
                self.drop_or_error(",")?;
                let new_bit_count = self.parse_usize_attribute("new_bit_count")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for sign_ext; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::SignExt { arg, new_bit_count },
                    maybe_id.unwrap(),
                )
            }
            "zero_ext" => {
                let arg = self.parse_node_ref(&node_env, "zero_ext arg")?;
                self.drop_or_error(",")?;
                let new_bit_count = self.parse_usize_attribute("new_bit_count")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for zero_ext; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }
                (
                    ir::NodePayload::ZeroExt { arg, new_bit_count },
                    maybe_id.unwrap(),
                )
            }
            "bit_slice" => {
                let arg = self.parse_node_ref(&node_env, "bit_slice arg")?;
                self.drop_or_error(",")?;
                let start = self.parse_usize_attribute("start")?;
                self.drop_or_error(",")?;
                let width = self.parse_usize_attribute("width")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for bit_slice; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::BitSlice { arg, start, width },
                    maybe_id.unwrap(),
                )
            }
            "bit_slice_update" => {
                let arg = self.parse_node_ref(&node_env, "bit_slice_update arg")?;
                self.drop_or_error(",")?;
                let start = self.parse_node_ref(&node_env, "bit_slice_update start")?;
                self.drop_or_error(",")?;
                let update_value =
                    self.parse_node_ref(&node_env, "bit_slice_update update_value")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for bit_slice_update; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::BitSliceUpdate {
                        arg,
                        start,
                        update_value,
                    },
                    maybe_id.unwrap(),
                )
            }
            "assert" => {
                let token = self.parse_node_ref(&node_env, "assert token")?;
                self.drop_or_error(",")?;
                let activate = self.parse_node_ref(&node_env, "assert activate")?;
                self.drop_or_error(",")?;
                let message = self.parse_string_attribute("message")?;
                self.drop_or_error(",")?;
                let label = self.parse_string_attribute("label")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for assert; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::Assert {
                        token,
                        activate,
                        message,
                        label,
                    },
                    maybe_id.unwrap(),
                )
            }
            "cover" => {
                let predicate = self.parse_node_ref(&node_env, "cover predicate")?;
                self.drop_or_error(",")?;
                let label = self.parse_string_attribute("label")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for cover; rest_of_line: {:?}",
                        self.rest()
                    )));
                }                (
                    ir::NodePayload::Cover { predicate, label },
                    maybe_id.unwrap(),
                )
            }
            "trace" => {
                let token = self.parse_node_ref(&node_env, "trace token")?;
                self.drop_or_error(",")?;
                let activated = self.parse_node_ref(&node_env, "trace activated")?;
                self.drop_or_error(",")?;
                let format = self.parse_string_attribute("format")?;
                self.drop_or_error(",")?;
                let operands = self.parse_node_ref_array_attribute(
                    "data_operands",
                    &node_env,
                    "trace data_operands",
                )?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for trace; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::Trace {
                        token,
                        activated,
                        format,
                        operands,
                    },
                    maybe_id.unwrap(),
                )
            }
            "after_all" => {
                // This is a variadic operation that puts the variadic node refs at the top
                // level instead of in an attribute.
                let mut operands = Vec::new();
                loop {
                    self.drop_whitespace_and_comments();
                    if self.peek_is(")") || self.peek_is("id=") {
                        break;
                    }
                    let operand = self.parse_node_ref(&node_env, "after_all operand")?;
                    operands.push(operand);
                    if !self.try_drop(",") {
                        break;
                    }
                }
                if self.peek_is("id=") {
                    maybe_id = Some(self.parse_id_attribute()?);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for after_all; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (ir::NodePayload::AfterAll(operands), maybe_id.unwrap())
            }
            "invoke" => {
                // This is a variadic operands that puts the variadic node refs at the top level
                // instead of in an attribute.
                let mut operands = Vec::new();
                loop {
                    self.drop_whitespace_and_comments();
                    if self.peek_is(")") || self.peek_is("to_apply=") {
                        break;
                    }
                    let operand = self.parse_node_ref(&node_env, "invoke arg")?;
                    operands.push(operand);
                    if !self.try_drop(",") {
                        break;
                    }
                }
                self.drop_or_error("to_apply=")?;
                let to_apply = self.pop_identifier_or_error("invoke to_apply")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for invoke; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::Invoke { to_apply, operands },
                    maybe_id.unwrap(),
                )
            }
            "concat" => {
                let operands =
                    self.parse_variadic_op(&node_env, &mut maybe_id, operator.as_str())?;
                (
                    ir::NodePayload::Nary(ir::NaryOp::Concat, operands),
                    maybe_id.unwrap(),
                )
            }
            "and" | "nor" | "or" | "xor" | "nand" => {
                let operands =
                    self.parse_variadic_op(&node_env, &mut maybe_id, operator.as_str())?;
                (
                    ir::NodePayload::Nary(
                        operator_to_nary_op(operator.as_str()).unwrap(),
                        operands,
                    ),
                    maybe_id.unwrap(),
                )
            }
            "literal" => {
                self.drop_or_error("value=")?;
                let value = self.parse_value_with_ty(&node_ty, "literal value")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for literal; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (ir::NodePayload::Literal(value), maybe_id.unwrap())
            }
            "shll" | "shrl" | "shra"
            // weak arithmetic ops
            | "add" | "sub"
            | "array_concat"
            // partial-product ops
            | "smulp" | "umulp"
            // strong arithmetic ops
            | "umul" | "smul" | "sdiv" | "udiv" | "umod" | "smod"
            // comparison ops
            | "eq" | "ne" | "ugt" | "ult" | "uge" | "ule" | "sgt"
            | "slt" | "sge" | "sle"
            // gate op
            | "gate" => {
                let binop = ir::operator_to_binop(operator.as_str())
                    .expect(format!("operator {:?} should be known binop", operator).as_str());
                let lhs = self.parse_node_ref(&node_env, "binop lhs")?;
                self.drop_or_error(",")?;
                let rhs = self.parse_node_ref(&node_env, "binop rhs")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for binop; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (ir::NodePayload::Binop(binop, lhs, rhs), maybe_id.unwrap())
            }
            "not" | "neg" | "identity" | "reverse" | "or_reduce" | "and_reduce" | "xor_reduce" => {
                let operand = self.parse_node_ref(&node_env, "unop operand")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for not; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                let unop = ir::operator_to_unop(operator.as_str()).unwrap();
                (ir::NodePayload::Unop(unop, operand), maybe_id.unwrap())
            }
            "decode" => {
                let arg = self.parse_node_ref(&node_env, "decode arg")?;
                self.drop_or_error(",")?;
                let width = self.parse_usize_attribute("width")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for decode; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (ir::NodePayload::Decode { arg, width }, maybe_id.unwrap())
            }
            "counted_for" => {
                // counted_for(init, trip_count=..., stride=..., body=...)
                let init = self.parse_node_ref(&node_env, "counted_for init")?;
                self.drop_or_error(",")?;
                self.drop_or_error("trip_count=")?;
                let trip_count = self.pop_number_usize_or_error("trip_count")?;
                self.drop_or_error(",")?;
                self.drop_or_error("stride=")?;
                let stride = self.pop_number_usize_or_error("stride")?;
                self.drop_or_error(",")?;
                self.drop_or_error("body=")?;
                let body = self.pop_identifier_or_error("counted_for body")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for counted_for; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }
                (
                    ir::NodePayload::CountedFor {
                        init,
                        trip_count,
                        stride,
                        body,
                    },
                    maybe_id.unwrap(),
                )
            }
            "encode" => {
                let arg = self.parse_node_ref(&node_env, "encode arg")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for encode; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (ir::NodePayload::Encode { arg }, maybe_id.unwrap())
            }
            "one_hot" => {
                let arg = self.parse_node_ref(&node_env, "one_hot arg")?;
                self.drop_or_error(",")?;
                let lsb_prio = self.parse_bool_attribute("lsb_prio")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for one_hot; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (ir::NodePayload::OneHot { arg, lsb_prio }, maybe_id.unwrap())
            }
            "sel" => {
                let selector = self.parse_node_ref(&node_env, "sel selector")?;
                self.drop_or_error(",")?;
                let cases = self.parse_node_ref_array_attribute("cases", &node_env, "sel cases")?;
                let mut default = None;
                if self.try_drop(", default=") {
                    default = Some(self.parse_node_ref(&node_env, "sel default node ref")?);
                }
                self.drop_whitespace_and_comments();
                if self.peek_is(", id=") {
                    self.drop_or_error(",")?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }

                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for sel; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::Sel {
                        selector,
                        cases,
                        default,
                    },
                    maybe_id.unwrap(),
                )
            }
            "priority_sel" => {
                let selector = self.parse_node_ref(&node_env, "priority_sel selector")?;
                self.drop_or_error(",")?;
                let cases =
                    self.parse_node_ref_array_attribute("cases", &node_env, "priority_sel cases")?;
                self.drop_or_error(",")?;
                self.drop_whitespace_and_comments();
                let default = if self.peek_is("default=") {
                    self.drop_or_error("default=")?;
                    Some(self.parse_node_ref(&node_env, "priority_sel default")?)
                } else {
                    None
                };
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for priority_sel; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::PrioritySel {
                        selector,
                        cases,
                        default,
                    },
                    maybe_id.unwrap(),
                )
            }
            "one_hot_sel" => {
                let selector = self.parse_node_ref(&node_env, "one_hot_sel selector")?;
                self.drop_or_error(",")?;
                let cases =
                    self.parse_node_ref_array_attribute("cases", &node_env, "one_hot_sel cases")?;
                if self.try_drop(",") {
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for one_hot_sel; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }                (
                    ir::NodePayload::OneHotSel { selector, cases },
                    maybe_id.unwrap(),
                )
            }
            "param" => {
                self.drop_or_error("name=")?;
                let _name = self.pop_identifier_or_error("param name")?;
                self.drop_or_error(",")?;
                let raw_id = self.parse_id_attribute()?;
                let pid = ir::ParamId::new(raw_id);
                (ir::NodePayload::GetParam(pid), raw_id)
            }
            _ => {
                return Err(ParseError::new(format!(
                    "unexpected operator {:?}; rest_of_line: {:?}",
                    operator,
                    self.rest_of_line()
                )));
            }
        };

        let pos_attr = self.maybe_drop_pos_attribute()?;
        self.drop_or_error_with_ctx(
            ")",
            &format!("end of node: {:?} operator: {:?}", name_or_id, operator),
        )?;

        let pos_data = pos_attr.map(|p| {
            p.into_iter()
                .map(|(f, l, c)| ir::Pos {
                    fileno: f,
                    lineno: l,
                    colno: c,
                })
                .collect()
        });

        Ok(ir::Node {
            text_id: id,
            name: match name_or_id {
                NameOrId::Id(..) => None,
                NameOrId::Name(name) => Some(name),
            },
            ty: node_ty,
            payload,
            pos: pos_data,
        })
    }

    pub fn add_param_as_node(
        &mut self,
        param: &ir::Param,
        node_env: &mut IrNodeEnv,
        nodes: &mut Vec<ir::Node>,
    ) {
        assert!(!nodes.is_empty(), "nodes should not be empty");
        let node = ir::Node {
            text_id: param.id.get_wrapped_id(),
            name: Some(param.name.clone()),
            ty: param.ty.clone(),
            payload: ir::NodePayload::GetParam(param.id),
            pos: None,
        };
        let node_ref = ir::NodeRef { index: nodes.len() };
        node_env.add(Some(param.name.clone()), node.text_id, node_ref);
        nodes.push(node);
    }

    pub fn parse_fn(&mut self) -> Result<ir::Fn, ParseError> {
        log::debug!("parse_fn");
        self.drop_or_error("fn")?;
        let fn_name = self.pop_identifier_or_error("fn name")?;

        // Parse the parameter text -- these are of the form `name: type id=n`
        let params = self.parse_params()?;
        log::debug!("params: {:?}", params);

        self.drop_or_error("->")?;
        let ret_ty = self.parse_type()?;
        self.drop_or_error("{")?;

        let mut nodes = vec![ir::Node {
            text_id: 0,
            name: Some("reserved_zero_node".to_string()),
            ty: ir::Type::nil(),
            payload: ir::NodePayload::Nil,
            pos: None,
        }];

        let mut node_env = IrNodeEnv::new();

        // For each of the params add it as a node.
        for param in params.iter() {
            self.add_param_as_node(param, &mut node_env, &mut nodes)
        }

        let mut ret_node_ref: Option<ir::NodeRef> = None;
        while !self.try_drop("}") {
            let is_ret = self.try_drop("ret ");
            let node = self.parse_node(&mut node_env)?;
            let node_ref = ir::NodeRef { index: nodes.len() };
            node_env.add(node.name.clone(), node.text_id, node_ref);
            nodes.push(node);
            if is_ret {
                ret_node_ref = Some(node_ref);
            }
        }

        // If the return type is not the same type as the return node, then we flag a
        // validation error.
        if let Some(ret_nr) = ret_node_ref {
            let ret_node = &nodes[ret_nr.index];
            if ret_node.ty != ret_ty {
                return Err(ParseError::new(format!(
                    "return type mismatch; expected: {}, got: {} from node: {}",
                    ret_ty, ret_node.ty, ret_node.text_id
                )));
            }
        }

        Ok(ir::Fn {
            name: fn_name,
            params,
            ret_ty,
            nodes,
            ret_node_ref,
        })
    }

    pub fn parse_package(&mut self) -> Result<ir::Package, ParseError> {
        log::debug!("parse_package");
        let mut fns = Vec::new();
        let mut top_name: Option<String> = None;

        self.drop_or_error("package")?;
        let package_name = self.pop_identifier_or_error("package name")?;
        log::debug!("package_name: {}", package_name);

        let mut file_table = FileTable::new();

        while !self.at_eof() {
            if self.try_drop("top") {
                let f = self.parse_fn()?;
                top_name = Some(f.name.clone());
                fns.push(f);
            } else if self.peek_is("fn") {
                let f = self.parse_fn()?;
                fns.push(f);
            } else if self.peek_is("proc") {
                return Err(ParseError::new(format!(
                    "only functions are supported, got proc; rest: {:?}",
                    self.rest()
                )));
            } else if self.peek_is("chan") {
                return Err(ParseError::new(format!(
                    "only functions are supported, got chan; rest: {:?}",
                    self.rest()
                )));
            } else if self.peek_is("file_number") {
                self.parse_file_number(&mut file_table)?;
            } else {
                return Err(ParseError::new(format!(
                    "expected top or fn, got {:?}; rest: {:?}",
                    self.peekc(),
                    self.rest()
                )));
            }
        }
        Ok(ir::Package {
            name: package_name,
            file_table,
            fns,
            top_name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth::ir_value::IrFormatPreference;

    #[test]
    fn test_parse_simple_fn() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "package simple_fn

top fn main() -> bits[32] {
  ret literal.1: bits[32] = literal(value=1, id=1)
}\n";
        let mut parser = Parser::new(input);
        let package = parser.parse_package().unwrap();
        assert_eq!(package.to_string(), input);
    }

    #[test]
    fn test_parse_sample_package() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "package shll_folding_overlarge

top fn candidate_main() -> bits[11] {
  literal.1305: bits[11] = literal(value=0, id=1305)
  literal.1302: bits[11] = literal(value=1024, id=1302)
  shll.1303: bits[11] = shll(literal.1305, literal.1302, id=1303)
  ret shll.1304: bits[11] = shll(shll.1303, literal.1302, id=1304)
}\n";
        let mut parser = Parser::new(input);
        let package = parser.parse_package().unwrap();
        assert_eq!(package.to_string(), input);
    }

    #[test]
    fn test_parse_package_two_fns() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "package two_functions

fn foo(x: bits[8] id=1) -> bits[8] {
  ret neg.2: bits[8] = neg(x, id=2)
}

fn bar(x: bits[8] id=3) -> bits[8] {
  ret not.4: bits[8] = not(x, id=4)
}\n";
        let mut parser = Parser::new(input);
        let package = parser.parse_package().unwrap();
        assert_eq!(package.to_string(), input);
    }

    #[test]
    fn test_parse_file_number() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "file_number 1 \"foo.x\"";
        let mut parser = Parser::new(input);
        let file_table = parser.parse_file_number(&mut FileTable::new()).unwrap();
        println!("{:?}", file_table);
    }

    #[test]
    fn test_parse_literal_node_with_id() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "literal.1: bits[32] = literal(value=1, id=1)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut IrNodeEnv::new()).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_hex_literal_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "literal.1: bits[8] = literal(value=0x2a, id=1)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut IrNodeEnv::new()).unwrap();
        if let ir::NodePayload::Literal(value) = node.payload {
            assert_eq!(
                value.to_string_fmt(IrFormatPreference::Hex).unwrap(),
                "bits[8]:0x2a"
            );
        } else {
            panic!("expected literal node");
        }
    }

    #[test]
    fn test_parse_hex_literal_with_underscores() {
        let _ = env_logger::builder().is_test(true).try_init();
        let ir = r#"package underscore

fn foo() -> bits[65] {
  ret literal.1: bits[65] = literal(value=0x1_ffff_ffff_ffff_fffe, id=1)
}
"#;

        // Verify the C++ parser accepts this IR.
        let _pkg = xlsynth::IrPackage::parse_ir(ir, None).unwrap();

        // Verify the Rust parser accepts the same IR and parses the literal correctly.
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let f = pkg.get_fn("foo").unwrap();
        if let ir::NodePayload::Literal(v) = &f.nodes[1].payload {
            assert_eq!(
                v.to_string_fmt(IrFormatPreference::Hex).unwrap(),
                "bits[65]:0x1_ffff_ffff_ffff_fffe"
            );
        } else {
            panic!("expected literal node");
        }
    }

    #[test]
    fn test_roundtrip_via_xls_parser() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = r#"package t

fn foo() -> (bits[8], bits[8], bits[8]) {
  literal.1: bits[8] = literal(value=0x2a, id=1)
  literal.2: bits[8] = literal(value=0b1100, id=2)
  literal.3: bits[8] = literal(value=5, id=3)
  ret tuple.4: (bits[8], bits[8], bits[8]) = tuple(literal.1, literal.2, literal.3, id=4)
}
"#;

        let cxx_pkg = xlsynth::IrPackage::parse_ir(input, None).unwrap();
        let formatted = cxx_pkg.to_string();

        let mut parser = Parser::new(&formatted);
        let pkg = parser.parse_package().unwrap();

        let f = pkg.get_fn("foo").unwrap();
        if let ir::NodePayload::Literal(v) = &f.nodes[1].payload {
            assert_eq!(
                v.to_string_fmt(IrFormatPreference::Hex).unwrap(),
                "bits[8]:0x2a"
            );
        } else {
            panic!("expected literal");
        }
        if let ir::NodePayload::Literal(v) = &f.nodes[2].payload {
            assert_eq!(
                v.to_string_fmt(IrFormatPreference::Binary).unwrap(),
                "bits[8]:0b1100"
            );
        } else {
            panic!("expected literal");
        }
        if let ir::NodePayload::Literal(v) = &f.nodes[3].payload {
            assert_eq!(
                v.to_string_fmt(IrFormatPreference::Default).unwrap(),
                "bits[8]:5"
            );
        } else {
            panic!("expected literal");
        }
    }

    #[test]
    fn test_param_with_id() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "fn foo(x: bits[8] id=1) -> bits[8] {
  ret neg.1: bits[8] = neg(x)
}";
        let mut parser = Parser::new(input);
        let f = parser.parse_fn().unwrap();
        println!("{:?}", f);
    }

    #[test]
    fn test_parse_tuple_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("x".to_string()), 1, ir::NodeRef { index: 1 });
        node_env.add(Some("y".to_string()), 2, ir::NodeRef { index: 2 });
        let input = "tuple.7: (token, bits[32]) = tuple(x, y, id=7)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_array_literal_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "literal.4: bits[8][5] = literal(value=[119, 111, 114, 108, 100], id=4)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut IrNodeEnv::new()).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_after_all_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("trace".to_string()), 17, ir::NodeRef { index: 17 });
        let input = "after_all.19: token = after_all(trace.17, id=19)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_after_all_node_with_pos() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("trace".to_string()), 17, ir::NodeRef { index: 17 });
        let input = "after_all.19: token = after_all(trace.17, id=19, pos=[(1,1,1)])";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        assert_eq!(
            node.payload,
            ir::NodePayload::AfterAll(vec![ir::NodeRef { index: 17 }])
        );
    }

    #[test]
    fn test_parse_bit_slice_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("x".to_string()), 1, ir::NodeRef { index: 1 });
        let input = "bit_slice.6: bits[2] = bit_slice(x, start=0, width=2, id=6)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_priority_sel() {
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("selector".to_string()), 1, ir::NodeRef { index: 1 });
        node_env.add(Some("one_case".to_string()), 2, ir::NodeRef { index: 2 });
        node_env.add(Some("my_default".to_string()), 3, ir::NodeRef { index: 3 });
        let input = "priority_sel.8: bits[2] = priority_sel(selector, cases=[one_case], default=my_default, id=8)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_explicit_param_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "x: bits[2][1] = param(name=x, id=1)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut IrNodeEnv::new()).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_cover_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("and".to_string()), 7, ir::NodeRef { index: 7 });
        let input = "cover.8: () = cover(and.7, label=\"x_less_than_0\", id=8)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_cover_node_with_pos() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("and".to_string()), 7, ir::NodeRef { index: 7 });
        let input = "cover.8: () = cover(and.7, label=\"x_less_than_0\", id=8, pos=[(2,42,30)])";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        assert_eq!(
            node.payload,
            ir::NodePayload::Cover {
                predicate: ir::NodeRef { index: 7 },
                label: "x_less_than_0".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_nested_tuple_type() {
        let input = "((bits[1], (bits[6], bits[1]), bits[3])[3], bits[3], bits[1])";
        let mut parser = Parser::new(input);
        let ty = parser.parse_type().unwrap();
        println!("{:?}", ty);
    }

    #[test]
    fn test_parse_multidim_array_type() {
        let input = "bits[32][4][5][6]";
        let mut parser = Parser::new(input);
        let ty = parser.parse_type().unwrap();
        let element_type = ir::Type::Bits(32);
        let element_type = ir::Type::new_array(element_type, 4);
        let element_type = ir::Type::new_array(element_type, 5);
        let element_type = ir::Type::new_array(element_type, 6);
        assert_eq!(ty, element_type);
    }

    #[test]
    fn test_parse_nil_tuple_with_id_and_pos() {
        let input = "tuple.31: () = tuple(id=31, pos=[(0,0,13)])";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut IrNodeEnv::new()).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_sel_fn() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "fn __sel__do_sel(s: bits[1] id=1, on_true: bits[1] id=2, on_false: bits[1] id=3) -> bits[1] {
    ret sel.4: bits[1] = sel(s, cases=[on_false, on_true], id=4, pos=[(0,0,54)])
}";
        let mut parser = Parser::new(input);
        let f = parser.parse_fn().unwrap();
        let ret_node = f.get_node(f.ret_node_ref.unwrap());
        assert_eq!(
            ret_node.payload,
            ir::NodePayload::Sel {
                selector: ir::NodeRef { index: 1 },
                cases: vec![ir::NodeRef { index: 3 }, ir::NodeRef { index: 2 }],
                default: None,
            }
        );
        println!("{:?}", ret_node);
    }

    #[test]
    fn test_parse_pos_attr_with_spaces() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("y".to_string()), 1, ir::NodeRef { index: 1 });
        let input =
            "  y_bexp__2: bits[8] = tuple_index(y, index=1, id=91812, pos=[(6,22,72), (6,58,17)])";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_parse_not_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("ugt".to_string()), 92055, ir::NodeRef { index: 92055 });
        let input =
            "not.92066: bits[1] = not(ugt.92055, id=92066, pos=[(3,39,68), (5,31,32), (5,58,17)])";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        println!("{:?}", node);
    }

    /// Tests the case where one of the arguments has no id suffix.
    #[test]
    fn test_parse_nor_with_name_only_arg() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(
            Some("which_path".to_string()),
            92029,
            ir::NodeRef { index: 92029 },
        );
        node_env.add(Some("eq".to_string()), 92056, ir::NodeRef { index: 92056 });
        node_env.add(Some("not".to_string()), 92066, ir::NodeRef { index: 92066 });
        let input = "  nor.92075: bits[1] = nor(which_path, eq.92056, not.92066, id=92075, pos=[(3,39,68), (5,31,32), (5,58,17)])";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        println!("{:?}", node);
    }

    /// Replaces all instances of `, pos=[...]` in the text with the empty
    /// string.
    fn strip_pos_data(s: &str) -> String {
        let regex = regex::Regex::new(r", pos=\[(\(\d+,\d+,\d+\),?)+\]").unwrap();
        regex.replace_all(s, "").to_string()
    }

    #[test]
    fn test_parse_function_with_named_tuple_index_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let dslx_text = "fn f(x: (u2, u4)) -> u4 { let my_element = x.1; my_element + u4:1 }";
        let fake_path = std::path::Path::new("test.x");
        let dslx_to_ir_result = xlsynth::convert_dslx_to_ir(
            dslx_text,
            fake_path,
            &xlsynth::DslxConvertOptions::default(),
        )
        .unwrap();
        let xlsynth_package_str = dslx_to_ir_result.ir.to_string();
        log::info!("xlsynth package str:\n{}", xlsynth_package_str);
        let mut parser = Parser::new(xlsynth_package_str.as_str());
        let parsed_package = parser.parse_package().unwrap();
        assert_eq!(
            parsed_package.to_string(),
            strip_pos_data(&xlsynth_package_str)
        );
    }

    #[test]
    fn test_parse_sel_with_default() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("input".to_string()), 1, ir::NodeRef { index: 1 });
        let input = "sel.2: bits[8] = sel(input, cases=[input], default=input, id=2)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        println!("{:?}", node);
    }

    #[test]
    fn test_round_trip_assert_ir_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "package sample_package

top fn main(t: token id=1) -> token {
  literal.2: bits[1] = literal(value=1, id=2)
  ret assert.3: token = assert(t, literal.2, message=\"Assertion failure via assert! @ far_path.x:24:12-24:65\", label=\"far_path_x_exp_must_be_ge_y_exp\", id=3)
}
";
        let mut parser = Parser::new(input);
        let package = parser.parse_package().unwrap();
        assert_eq!(package.to_string(), input);
    }

    #[test]
    fn test_round_trip_or_nary_ir_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        // Build a small node environment for the n-ary OR node we want to parse.
        let mut node_env = IrNodeEnv::new();
        node_env.add(
            Some("is_result_nan".to_string()),
            1,
            ir::NodeRef { index: 1 },
        );
        node_env.add(
            Some("is_operand_inf".to_string()),
            2,
            ir::NodeRef { index: 2 },
        );
        node_env.add(
            Some("bit_slice".to_string()),
            90408,
            ir::NodeRef { index: 90408 },
        );
        node_env.add(
            Some("and_reduce".to_string()),
            90409,
            ir::NodeRef { index: 90409 },
        );

        // Input string containing an n-ary OR with four operands and a pos attribute.
        let input = "or.91095: bits[1] = or(is_result_nan, is_operand_inf, bit_slice.90408, and_reduce.90409, id=91095, pos=[(0,2144,26), (2,312,48), (3,2,51)])";
        let mut parser = Parser::new(input);
        let node = parser
            .parse_node(&mut node_env)
            .expect("parse nary or node");

        // The node should be an Nary op with operator OR and four operands.
        if let ir::NodePayload::Nary(op, operands) = &node.payload {
            assert_eq!(*op, ir::NaryOp::Or);
            assert_eq!(operands.len(), 4);
            assert_eq!(operands[0].index, 1);
            assert_eq!(operands[1].index, 2);
            assert_eq!(operands[2].index, 90408);
            assert_eq!(operands[3].index, 90409);
        } else {
            panic!("expected nary or node");
        }
    }

    #[test]
    fn test_parse_counted_for_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        // Provide a mapping for the init literal referenced as `literal.5` in the input
        // string.
        node_env.add(Some("literal".to_string()), 5, ir::NodeRef { index: 5 });
        let input = "counted_for.6: bits[11] = counted_for(literal.5, trip_count=7, stride=1, body=body1, id=6)";
        let mut parser = Parser::new(input);
        let node = parser
            .parse_node(&mut node_env)
            .expect("parse counted_for node");
        match node.payload {
            ir::NodePayload::CountedFor {
                init,
                trip_count,
                stride,
                body,
            } => {
                assert_eq!(init.index, 5);
                assert_eq!(trip_count, 7);
                assert_eq!(stride, 1);
                assert_eq!(body, "body1");
            }
            _ => panic!("expected counted_for payload"),
        }
    }

    #[test]
    fn test_array_index_assumed_in_bounds() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "array_index.2: bits[32] = array_index(x, indices=[literal.1], assumed_in_bounds=true, id=2)";
        let mut parser = Parser::new(input);
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("x".to_string()), 1, ir::NodeRef { index: 1 });
        node_env.add(Some("literal.1".to_string()), 2, ir::NodeRef { index: 2 });
        let node = parser.parse_node(&mut node_env).unwrap();
        assert!(matches!(
            node.payload,
            ir::NodePayload::ArrayIndex {
                assumed_in_bounds: true,
                ..
            }
        ));
    }

    #[test]
    fn test_array_index_assumed_in_bounds_no_space() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "array_index.2: bits[32] = array_index(x, indices=[literal.1],assumed_in_bounds=true, id=2)";
        let mut parser = Parser::new(input);
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("x".to_string()), 1, ir::NodeRef { index: 1 });
        node_env.add(Some("literal.1".to_string()), 2, ir::NodeRef { index: 2 });
        let node = parser.parse_node(&mut node_env).unwrap();
        assert!(matches!(
            node.payload,
            ir::NodePayload::ArrayIndex {
                assumed_in_bounds: true,
                ..
            }
        ));
    }

    #[test]
    fn test_array_update_assumed_in_bounds() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "array_update.2: bits[32][1] = array_update(x, y, indices=[literal.1], assumed_in_bounds=true, id=2)";
        let mut parser = Parser::new(input);
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("x".to_string()), 1, ir::NodeRef { index: 1 });
        node_env.add(Some("y".to_string()), 2, ir::NodeRef { index: 2 });
        node_env.add(Some("literal.1".to_string()), 3, ir::NodeRef { index: 3 });
        let node = parser.parse_node(&mut node_env).unwrap();
        assert!(matches!(
            node.payload,
            ir::NodePayload::ArrayUpdate {
                assumed_in_bounds: true,
                ..
            }
        ));
    }

    #[test]
    fn test_array_update_assumed_in_bounds_no_space() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "array_update.2: bits[32][1] = array_update(x, y, indices=[literal.1],assumed_in_bounds=true, id=2)";
        let mut parser = Parser::new(input);
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("x".to_string()), 1, ir::NodeRef { index: 1 });
        node_env.add(Some("y".to_string()), 2, ir::NodeRef { index: 2 });
        node_env.add(Some("literal.1".to_string()), 3, ir::NodeRef { index: 3 });
        let node = parser.parse_node(&mut node_env).unwrap();
        assert!(matches!(
            node.payload,
            ir::NodePayload::ArrayUpdate {
                assumed_in_bounds: true,
                ..
            }
        ));
    }

    #[test]
    fn test_dynamic_bit_slice() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = " dynamic_bit_slice.1: bits[14] = dynamic_bit_slice(x, y, width=14, id=1)";
        let mut parser = Parser::new(input);
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("x".to_string()), 1, ir::NodeRef { index: 1 });
        node_env.add(Some("y".to_string()), 2, ir::NodeRef { index: 2 });
        let node = parser.parse_node(&mut node_env).unwrap();
        assert!(matches!(
            node.payload,
            ir::NodePayload::DynamicBitSlice { width: 14, .. }
        ));
        println!("{:?}", node);
    }

    #[test]
    fn test_binary_and_with_pos_attr() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "my_name: bits[1] = and(x, y, id=3, pos=[(0,0,13), (0,0,14)])";
        let mut parser = Parser::new(input);
        let mut node_env = IrNodeEnv::new();
        let x_node_ref = ir::NodeRef { index: 1 };
        let y_node_ref = ir::NodeRef { index: 2 };
        node_env.add(Some("x".to_string()), 1, x_node_ref);
        node_env.add(Some("y".to_string()), 2, y_node_ref);
        let node = parser.parse_node(&mut node_env).unwrap();
        let want = ir::NodePayload::Nary(ir::NaryOp::And, vec![x_node_ref, y_node_ref]);
        assert_eq!(node.payload, want);
    }

    #[test]
    fn test_node_name_starts_with_ret() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "  retained_is_odd: bits[1] = bit_slice(x, start=12, width=1, id=905, pos=[(1,327,31), (1,353,27), (1,371,25), (2,12,51)])";
        let mut parser = Parser::new(input);
        let mut node_env = IrNodeEnv::new();
        let x_node_ref = ir::NodeRef { index: 1 };
        node_env.add(Some("x".to_string()), 1, x_node_ref);
        let node = parser.parse_node(&mut node_env).unwrap();
        assert_eq!(
            node.payload,
            ir::NodePayload::BitSlice {
                arg: x_node_ref,
                start: 12,
                width: 1,
            }
        );
    }

    #[test]
    fn test_parse_identity_fn() {
        let input = "fn __f__f(x: bits[8] id=29) -> bits[8] {\n  ret x: bits[8] = param(name=x, id=29)\n}\n";
        let mut parser = Parser::new(input);
        let f = parser.parse_fn().unwrap();
        let ret_node = f.get_node(f.ret_node_ref.unwrap());
        if let crate::xls_ir::ir::NodePayload::GetParam(pid) = &ret_node.payload {
            assert_eq!(pid.get_wrapped_id(), 29);
        } else {
            panic!("Expected GetParam node payload");
        }
    }

    #[test]
    fn test_package_pos_retention() {
        let ir = r#"package pos_pkg
file_number 0 "a.x"
file_number 1 "b.x"

top fn main() -> bits[32] {
  ret literal.1: bits[32] = literal(value=1, id=1, pos=[(0,0,0), (1,1,1)])
}
"#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let main_fn = pkg.get_top().unwrap();
        let data = main_fn.nodes[1].pos.as_ref().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(
            data[0].to_human_string(&pkg.file_table).as_deref(),
            Some("a.x:1:1")
        );
        assert_eq!(
            data[1].to_human_string(&pkg.file_table).as_deref(),
            Some("b.x:2:2")
        );
    }

    #[test]
    fn test_package_collects_pos() {
        let ir = "package nopos\n\nfn main() -> bits[1] {\n  ret literal.1: bits[1] = literal(value=0, id=1, pos=[(0,0,0)])\n}\n";
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        let main_fn = pkg.get_top().unwrap();
        assert!(main_fn.nodes[1].pos.is_some());
    }

    #[test]
    fn test_parse_with_line_comments() {
        let _ = env_logger::builder().is_test(true).try_init();
        let ir = "package cmt_pkg // package comment\n\nfn main() -> bits[8] { // function comment\n  // This literal should be parsed correctly.\n  ret literal.1: bits[8] = literal(value=42, id=1) // trailing comment\n}\n";
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().unwrap();
        assert_eq!(pkg.name, "cmt_pkg");
        let main_fn = pkg.get_fn("main").unwrap();
        assert_eq!(main_fn.ret_ty.to_string(), "bits[8]");
    }

    // Regression test for parsing multidimensional array types whose element
    // type is *itself* a tuple. Prior to the fix for issue #<none>, the parser
    // accepted only a single `[N]` suffix after a tuple, failing on inputs like
    // `((bits[2][2][2], bits[2])[7][2]`. This test ensures such inputs are now
    // handled correctly.
    #[test]
    fn test_parse_tuple_multidim_array_type() {
        let input = "(bits[2], bits[3])[7][2]";
        let mut parser = Parser::new(input);
        let parsed_ty = parser.parse_type().unwrap();

        // Manually construct the expected type: a 2-tuple of bits[2] values,
        // wrapped in a 7-element array, then wrapped in a 2-element array.
        let inner_tuple = ir::Type::Tuple(vec![
            Box::new(ir::Type::Bits(2)),
            Box::new(ir::Type::Bits(3)),
        ]);
        let array_7 = ir::Type::new_array(inner_tuple, 7);
        let want = ir::Type::new_array(array_7, 2);

        assert_eq!(parsed_ty, want);
    }
}
