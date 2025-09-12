// SPDX-License-Identifier: Apache-2.0

//! Parser for XLS IR (just functions for the time being).

use crate::ir::{
    self, ArrayTypeData, BlockPortInfo, FileTable, PackageMember, operator_to_nary_op,
};
use crate::ir_node_env::{IrNodeEnv, NameOrId};
use crate::ir_validate;

pub fn parse_path_to_package(path: &std::path::Path) -> Result<ir::Package, ParseError> {
    let file_content = std::fs::read_to_string(path)
        .map_err(|e| ParseError::new(format!("failed to read file: {}", e)))?;
    let mut parser = Parser::new(&file_content);
    parser.parse_package()
}

/// Parses a package from `path` and validates the resulting IR.
pub fn parse_and_validate_path_to_package(
    path: &std::path::Path,
) -> Result<ir::Package, ParseOrValidateError> {
    let pkg = parse_path_to_package(path)?;
    ir_validate::validate_package(&pkg)?;
    Ok(pkg)
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

/// Unified error for parse-and-validate helpers.
#[derive(Debug)]
pub enum ParseOrValidateError {
    Parse(ParseError),
    Validate(ir_validate::ValidationError),
}

impl std::fmt::Display for ParseOrValidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseOrValidateError::Parse(e) => write!(f, "{}", e),
            ParseOrValidateError::Validate(e) => write!(f, "ValidationError: {}", e),
        }
    }
}

impl std::error::Error for ParseOrValidateError {}

impl From<ParseError> for ParseOrValidateError {
    fn from(e: ParseError) -> Self {
        ParseOrValidateError::Parse(e)
    }
}

impl From<ir_validate::ValidationError> for ParseOrValidateError {
    fn from(e: ir_validate::ValidationError) -> Self {
        ParseOrValidateError::Validate(e)
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
    options: ParseOptions,
}

#[derive(Clone, Copy, Debug)]
pub struct ParseOptions {
    pub retain_pos_data: bool,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            retain_pos_data: true,
        }
    }
}

impl Parser {
    pub fn new(input: &str) -> Self {
        Self::new_with_options(input, ParseOptions::default())
    }

    pub fn new_with_options(input: &str, options: ParseOptions) -> Self {
        Self {
            chars: input.chars().collect(),
            offset: 0,
            options,
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
        // Consume only ASCII whitespace characters: space, tab, CR, LF.
        while let Some(c) = self.peekc() {
            if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
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
                let is_valid_start = Self::is_ident_start(c);
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
                let is_valid_rest = Self::is_ident_rest(c);
                if !is_valid_rest {
                    return Ok(identifier);
                }
                self.dropc()?;
                identifier.push(c);
            }
        }
        if identifier.is_empty() {
            return Err(ParseError::new(format!(
                "in {} expected identifier, got EOF",
                ctx
            )));
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
        match number.parse::<usize>() {
            Ok(v) => Ok(v),
            Err(e) => Err(ParseError::new(format!(
                "in {} expected unsigned integer, got {:?}: {}",
                ctx, number, e
            ))),
        }
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

    fn is_ident_start(c: char) -> bool {
        (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'
    }

    fn is_ident_rest(c: char) -> bool {
        Self::is_ident_start(c) || (c >= '0' && c <= '9')
    }

    fn peek_keyword_is(&self, kw: &str) -> bool {
        if !self.peek_is(kw) {
            return false;
        }
        let next_index = self.offset + kw.len();
        if next_index >= self.chars.len() {
            return true;
        }
        let next_c = self.chars[next_index];
        !Self::is_ident_rest(next_c)
    }

    fn try_drop_keyword(&mut self, kw: &str) -> bool {
        self.drop_whitespace_and_comments();
        if self.peek_keyword_is(kw) {
            self.offset += kw.len();
            true
        } else {
            false
        }
    }

    fn drop_keyword_or_error(&mut self, kw: &str, ctx: &str) -> Result<(), ParseError> {
        self.drop_whitespace_and_comments();
        if self.try_drop_keyword(kw) {
            Ok(())
        } else {
            Err(ParseError::new(format!(
                "expected keyword {:?} in {}; line: {:?}",
                kw,
                ctx,
                self.current_line()
            )))
        }
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
        if raw_id == 0 {
            return Err(ParseError::new(format!(
                "parameter id must be greater than zero, got 0; rest_of_line: {:?}",
                self.rest_of_line()
            )));
        }
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

    fn pop_node_name_or_error_with_dotted(
        &mut self,
        ctx: &str,
    ) -> Result<(NameOrId, Option<String>), ParseError> {
        let name: String = self.pop_identifier_or_error(ctx)?;
        if self.try_drop(".") {
            let id = self.pop_number_usize_or_error(ctx)?;
            Ok((NameOrId::Id(id), Some(name)))
        } else {
            Ok((NameOrId::Name(name), None))
        }
    }

    fn pop_node_name_or_error(&mut self, ctx: &str) -> Result<NameOrId, ParseError> {
        let (name_or_id, _dot) = self.pop_node_name_or_error_with_dotted(ctx)?;
        Ok(name_or_id)
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
        // Accept optional leading comma and whitespace before pos=
        let _ = self.try_drop(",");
        self.drop_whitespace_and_comments();
        if !self.try_drop("pos=") {
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
            ir::Type::Token => {
                // Expect the keyword `token` for token literals.
                let ident = self.pop_identifier_or_error("token literal")?;
                if ident != "token" {
                    return Err(ParseError::new(format!(
                        "expected token literal keyword `token`, got {:?} in {}; rest: {:?}",
                        ident,
                        ctx,
                        self.rest()
                    )));
                }
                Ok(xlsynth::IrValue::make_token())
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
        let (name_or_id, dotted_prefix_opt) =
            self.pop_node_name_or_error_with_dotted("node name")?;
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
            "array_slice" => {
                let array = self.parse_node_ref(&node_env, "array_slice array")?;
                self.drop_or_error(",")?;
                let start = self.parse_node_ref(&node_env, "array_slice start")?;
                self.drop_or_error(",")?;
                let width = self.parse_usize_attribute("width")?;
                if self.peek_is(",") {
                    self.dropc()?;
                    let id_attr = self.parse_id_attribute()?;
                    maybe_id = Some(id_attr);
                }
                if maybe_id.is_none() {
                    return Err(ParseError::new(format!(
                        "expected id for array_slice; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }
                (
                    ir::NodePayload::ArraySlice { array, start, width },
                    maybe_id.unwrap(),
                )
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
                // counted_for(init, trip_count=..., stride=..., body=..., invariant_args=[...])
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

                // Optional attributes: invariant_args=[...] and id=...
                let mut invariant_args: Vec<ir::NodeRef> = Vec::new();
                if self.try_drop(",") {
                    self.drop_whitespace_and_comments();
                    if self.peek_is("invariant_args=") {
                        let inv = self.parse_node_ref_array_attribute(
                            "invariant_args",
                            &node_env,
                            "counted_for invariant_args",
                        )?;
                        invariant_args = inv;
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
                        invariant_args,
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
                let name_attr = self.pop_identifier_or_error("param name")?;
                self.drop_or_error(",")?;
                let raw_id = self.parse_id_attribute()?;
                if raw_id == 0 {
                    return Err(ParseError::new(format!(
                        "param id must be greater than zero, got 0; rest_of_line: {:?}",
                        self.rest_of_line()
                    )));
                }
                // If either the provided name or id refer to an existing parameter
                // (from the function signature), require that they refer to the same
                // node. This prevents mismatches like name=x but id=2 when x is id=1.
                // If neither refer to an existing parameter, this is an unknown
                // parameter name/id reference and should be rejected.
                let by_name = node_env.name_id_to_ref(&NameOrId::Name(name_attr.clone()));
                let by_id = node_env.name_id_to_ref(&NameOrId::Id(raw_id));
                if by_name.is_some() || by_id.is_some() {
                    match (by_name, by_id) {
                        (Some(nr_name), Some(nr_id)) if nr_name == nr_id => {}
                        _ => {
                            return Err(ParseError::new(format!(
                                "param name/id mismatch: name={} id={}",
                                name_attr, raw_id
                            )));
                        }
                    }
                } else {
                    return Err(ParseError::new(format!(
                        "unknown parameter name in param node: {}",
                        name_attr
                    )));
                }
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

        let pos_data = if self.options.retain_pos_data {
            pos_attr.map(|p| {
                p.into_iter()
                    .map(|(f, l, c)| ir::Pos {
                        fileno: f,
                        lineno: l,
                        colno: c,
                    })
                    .collect()
            })
        } else {
            None
        };

        // Enforce dotted LHS consistency: '<prefix>.<digits>' must match operator and
        // id.
        if let Some(lhs_prefix) = dotted_prefix_opt.as_ref() {
            let expected_op = operator.as_str();
            if lhs_prefix != expected_op {
                return Err(ParseError::new(format!(
                    "node name dotted prefix '{}' does not match operator '{}'",
                    lhs_prefix, expected_op
                )));
            }
            if let NameOrId::Id(lhs_id) = name_or_id {
                if lhs_id != id {
                    return Err(ParseError::new(format!(
                        "node name id suffix {} does not match id attribute {}",
                        lhs_id, id
                    )));
                }
            }
        }

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
            // Special handling: avoid duplicating GetParam nodes if we've already
            // created one for this param id from the function signature. If a
            // duplicate param(...) appears (e.g., for a return), just reference
            // the existing node instead of adding a new one.
            let mut node_ref = ir::NodeRef { index: nodes.len() };
            let is_get_param = matches!(node.payload, ir::NodePayload::GetParam(_));
            if is_get_param {
                if let Some(existing) = node_env
                    .name_id_to_ref(&crate::ir_node_env::NameOrId::Id(node.text_id))
                    .copied()
                {
                    // If a GetParam node with this id already exists (from the
                    // function signature), ensure the textual node's type matches
                    // the existing param node type. If not, this is a parse-time
                    // error (mirrors upstream xlsynth behavior).
                    let existing_ty = &nodes[existing.index].ty;
                    if existing_ty != &node.ty {
                        return Err(ParseError::new(format!(
                            "param id={} type mismatch: header {} vs node {}",
                            node.text_id, existing_ty, node.ty
                        )));
                    }
                    // Do not add a duplicate; use the existing node ref.
                    node_ref = existing;
                } else {
                    node_env.add(node.name.clone(), node.text_id, node_ref);
                    nodes.push(node);
                }
            } else {
                node_env.add(node.name.clone(), node.text_id, node_ref);
                nodes.push(node);
            }
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

    /// Parses a combinational `block` and converts it to an equivalent `fn`.
    ///
    /// - `input_port` nodes become parameters (GetParam nodes) with matching
    ///   ids.
    /// - `output_port` nodes are discarded and the referenced value becomes the
    ///   function return.
    /// - Other nodes are parsed as normal IR nodes.
    pub fn parse_block_to_fn(&mut self) -> Result<ir::Fn, ParseError> {
        // Backward-compatible API: discard port info.
        let (f, _) = self.parse_block_to_fn_with_ports()?;
        Ok(f)
    }

    pub fn parse_block_to_fn_with_ports(&mut self) -> Result<(ir::Fn, BlockPortInfo), ParseError> {
        // Skip optional outer attributes like `#[signature("""...""")]`.
        loop {
            self.drop_whitespace_and_comments();
            if self.peek_is("#![") || self.peek_is("#[") {
                if self.try_drop("#![") {
                    // ok
                } else if self.try_drop("#[") {
                    // ok
                } else {
                    unreachable!("attribute must start with '#![' or '#[' after peek");
                }
                // Scan until ']'.
                while let Some(c) = self.peekc() {
                    self.dropc()?;
                    if c == ']' {
                        break;
                    }
                }
                continue;
            }
            break;
        }

        self.drop_or_error("block")?;
        let block_name = self.pop_identifier_or_error("block name")?;

        // Parse port list from the block header: `name: type, ...` (no ids).
        let mut header_ports: Vec<(String, ir::Type)> = Vec::new();
        self.drop_or_error("(")?;
        loop {
            if self.try_drop(")") {
                break;
            }
            let pname = self.pop_identifier_or_error("block port name")?;
            self.drop_or_error(":")?;
            let pty = self.parse_type()?;
            header_ports.push((pname, pty));
            if !self.try_drop(",") {
                self.drop_or_error(")")?;
                break;
            }
        }

        self.drop_or_error("{")?;

        // Nodes start with a reserved zero node.
        let mut nodes = vec![ir::Node {
            text_id: 0,
            name: Some("reserved_zero_node".to_string()),
            ty: ir::Type::nil(),
            payload: ir::NodePayload::Nil,
            pos: None,
        }];
        let mut node_env = IrNodeEnv::new();

        // Track input parameters discovered in the body: name -> (ty, id)
        let mut input_params: Vec<(String, ir::Type, usize)> = Vec::new();
        // Track outputs discovered: (output port name, node ref)
        let mut outputs: Vec<(String, ir::NodeRef)> = Vec::new();
        let mut output_ids_by_name: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        // Helper to maybe skip inner attributes like `#![provenance(...)]`.
        let skip_inner_attribute = |this: &mut Parser| -> Result<bool, ParseError> {
            this.drop_whitespace_and_comments();
            if this.peek_is("#![") || this.peek_is("#[") {
                if this.try_drop("#![") {
                    // ok
                } else if this.try_drop("#[") {
                    // ok
                } else {
                    unreachable!("attribute must start with '#![' or '#[' after peek");
                }
                while let Some(c) = this.peekc() {
                    this.dropc()?;
                    if c == ']' {
                        break;
                    }
                }
                Ok(true)
            } else {
                Ok(false)
            }
        };

        // Parse body lines until '}'.
        while !self.try_drop("}") {
            // Skip attributes and blank/comment lines.
            if skip_inner_attribute(self)? {
                continue;
            }
            self.drop_whitespace_and_comments();
            // If this is an input_port or output_port, handle specially.
            // Peek ahead to capture the operator by temporarily parsing the LHS name and
            // type. Save current offset to backtrack for normal node parsing.
            let saved_offset = self.offset;
            // If we can't parse a node header (name: type = ...), fall back to normal
            // parsing which will error with a helpful message.
            let parse_port_line = || -> Result<Option<()>, ParseError> {
                // name or id.
                let name_or_id = self.pop_node_name_or_error("node name")?;
                self.drop_or_error(":")?;
                let node_ty = self.parse_type()?;
                self.drop_or_error("=")?;
                let operator = self.pop_identifier_or_error("node operator")?;
                if operator == "input_port" {
                    self.drop_or_error("(")?;
                    self.drop_or_error("name=")?;
                    let port_name = self.pop_identifier_or_error("input_port name")?;
                    let mut id_opt: Option<usize> = None;
                    if self.try_drop(",") {
                        self.drop_whitespace_and_comments();
                        if self.peek_is("id=") {
                            id_opt = Some(self.parse_id_attribute()?);
                        }
                    }
                    // Optionally allow a trailing pos attribute, then ')'
                    if self.try_drop(",") {
                        let _ = self.maybe_drop_pos_attribute()?;
                    }
                    self.drop_or_error(")")?;
                    // Validate and construct a GetParam node with the given id.
                    let id_val = id_opt.ok_or_else(|| {
                        ParseError::new(format!(
                            "expected id for input_port; rest_of_line: {:?}",
                            self.rest_of_line()
                        ))
                    })?;
                    // Name must be provided on LHS.
                    let lhs_name = match name_or_id {
                        NameOrId::Name(n) => n,
                        NameOrId::Id(_) => {
                            return Err(ParseError::new(
                                "input_port must have a name on LHS".to_string(),
                            ));
                        }
                    };
                    // Optional consistency check: header declared this port.
                    let _ = header_ports
                        .iter()
                        .find(|(n, _)| n == &lhs_name)
                        .ok_or_else(|| {
                            ParseError::new(format!(
                                "input_port '{}' not found in block header ports",
                                lhs_name
                            ))
                        })?;
                    if id_val == 0 {
                        return Err(ParseError::new(
                            "input_port id must be greater than zero".to_string(),
                        ));
                    }
                    let pid = ir::ParamId::new(id_val);
                    let node = ir::Node {
                        text_id: id_val,
                        name: Some(lhs_name.clone()),
                        ty: node_ty.clone(),
                        payload: ir::NodePayload::GetParam(pid),
                        pos: None,
                    };
                    let node_ref = ir::NodeRef { index: nodes.len() };
                    node_env.add(node.name.clone(), node.text_id, node_ref);
                    nodes.push(node);
                    // Record input param for fn signature.
                    input_params.push((port_name, node_ty, id_val));
                    Ok(Some(()))
                } else if operator == "output_port" {
                    self.drop_or_error("(")?;
                    // First arg is the value being output.
                    let value_ref = self.parse_node_ref(&node_env, "output_port value")?;
                    // Consume any additional attributes in any order until we hit ')'.
                    let mut out_name_opt: Option<String> = None;
                    let mut out_id_opt: Option<usize> = None;
                    loop {
                        self.drop_whitespace_and_comments();
                        if !self.try_drop(",") {
                            break;
                        }
                        self.drop_whitespace_and_comments();
                        if self.peek_is("name=") {
                            self.drop_or_error("name=")?;
                            let nm = self.pop_identifier_or_error("output_port name")?;
                            out_name_opt = Some(nm);
                            continue;
                        }
                        if self.peek_is("id=") {
                            out_id_opt = Some(self.parse_id_attribute()?);
                            continue;
                        }
                        if self.peek_is("pos=") {
                            let _ = self.maybe_drop_pos_attribute()?;
                            continue;
                        }
                        // Unknown attribute; break and let ')' be checked next.
                        break;
                    }
                    self.drop_or_error(")")?;
                    let out_name = out_name_opt.ok_or_else(|| {
                        ParseError::new("output_port missing name attribute".to_string())
                    })?;
                    let out_id = out_id_opt.ok_or_else(|| {
                        ParseError::new("output_port missing id attribute".to_string())
                    })?;
                    outputs.push((out_name.clone(), value_ref));
                    output_ids_by_name.insert(out_name, out_id);
                    Ok(Some(()))
                } else {
                    Ok(None)
                }
            };

            let mut parse_port_line = parse_port_line;
            match parse_port_line() {
                Ok(Some(())) => {
                    // handled specially
                }
                Ok(None) => {
                    // Not a special port op; reset and parse as a normal node.
                    self.offset = saved_offset;
                    let node = self.parse_node(&mut node_env)?;
                    let node_ref = ir::NodeRef { index: nodes.len() };
                    node_env.add(node.name.clone(), node.text_id, node_ref);
                    nodes.push(node);
                }
                Err(e) => return Err(e),
            }
        }

        // Build function parameters in the order they appear in the header, but only
        // for inputs.
        let mut params: Vec<ir::Param> = Vec::new();
        for (hname, hty) in header_ports.iter() {
            if let Some((_, _, id)) = input_params.iter().find(|(n, _, _)| n == hname) {
                if *id == 0 {
                    return Err(ParseError::new(format!(
                        "input_port '{}' id must be greater than zero",
                        hname
                    )));
                }
                params.push(ir::Param {
                    name: hname.clone(),
                    ty: hty.clone(),
                    id: ir::ParamId::new(*id),
                });
            }
        }

        // Outputs are header ports that are not inputs.
        let header_output_names: Vec<String> = header_ports
            .iter()
            .map(|(n, _)| n.clone())
            .filter(|n| input_params.iter().all(|(inn, _, _)| inn != n))
            .collect();
        if header_output_names.is_empty() {
            return Err(ParseError::new(
                "no outputs declared in block header".to_string(),
            ));
        }
        // Map output_port(name=...) by name.
        let mut out_map: std::collections::HashMap<String, ir::NodeRef> =
            std::collections::HashMap::new();
        for (n, r) in outputs.into_iter() {
            out_map.insert(n, r);
        }
        // Build return node refs and types in header order.
        let mut ret_nodes_in_order: Vec<ir::NodeRef> = Vec::new();
        let mut ret_types_in_order: Vec<ir::Type> = Vec::new();
        for (hn, hty) in header_ports.iter() {
            if header_output_names.iter().any(|n| n == hn)
                && input_params.iter().all(|(inn, _, _)| inn != hn)
            {
                let nr = out_map.get(hn).ok_or_else(|| {
                    ParseError::new(format!(
                        "no output_port for header-declared output '{}'",
                        hn
                    ))
                })?;
                ret_nodes_in_order.push(*nr);
                ret_types_in_order.push(hty.clone());
            }
        }

        if ret_nodes_in_order.len() == 1 {
            let ret_node_ref = ret_nodes_in_order[0];
            let ret_ty = ret_types_in_order.remove(0);
            let ret_node_ty = nodes[ret_node_ref.index].ty.clone();
            if ret_node_ty != ret_ty {
                return Err(ParseError::new(format!(
                    "return type mismatch; expected: {}, got: {} from node: {}",
                    ret_ty, ret_node_ty, nodes[ret_node_ref.index].text_id
                )));
            }
            return Ok((
                ir::Fn {
                    name: block_name,
                    params,
                    ret_ty,
                    nodes,
                    ret_node_ref: Some(ret_node_ref),
                },
                BlockPortInfo {
                    input_port_ids: input_params
                        .iter()
                        .map(|(n, _t, id)| (n.clone(), *id))
                        .collect(),
                    output_port_ids: output_ids_by_name,
                    output_names: header_output_names,
                },
            ));
        }

        // Multiple outputs: build a tuple node and return it.
        let ret_ty = ir::Type::Tuple(
            ret_types_in_order
                .into_iter()
                .map(|t| Box::new(t))
                .collect(),
        );
        let next_id = nodes.iter().map(|n| n.text_id).max().unwrap_or(0) + 1;
        let tuple_node = ir::Node {
            text_id: next_id,
            name: None,
            ty: ret_ty.clone(),
            payload: ir::NodePayload::Tuple(ret_nodes_in_order.clone()),
            pos: None,
        };
        let ret_node_ref = ir::NodeRef { index: nodes.len() };
        node_env.add(None, next_id, ret_node_ref);
        nodes.push(tuple_node);

        Ok((
            ir::Fn {
                name: block_name,
                params,
                ret_ty,
                nodes,
                ret_node_ref: Some(ret_node_ref),
            },
            BlockPortInfo {
                input_port_ids: input_params
                    .iter()
                    .map(|(n, _t, id)| (n.clone(), *id))
                    .collect(),
                output_port_ids: output_ids_by_name,
                output_names: header_output_names,
            },
        ))
    }

    pub fn parse_package(&mut self) -> Result<ir::Package, ParseError> {
        log::debug!("parse_package");
        let mut members: Vec<PackageMember> = Vec::new();
        let mut top_name: Option<String> = None;

        self.drop_keyword_or_error("package", "package header")?;
        let package_name = self.pop_identifier_or_error("package name")?;
        log::debug!("package_name: {}", package_name);

        let mut file_table = FileTable::new();

        while !self.at_eof() {
            // Allow standalone attributes between members (commonly preceding a block).
            self.drop_whitespace_and_comments();
            if self.peek_is("#![") || self.peek_is("#[") {
                if self.try_drop("#![") {
                    // ok
                } else if self.try_drop("#[") {
                    // ok
                }
                while let Some(c) = self.peekc() {
                    self.dropc()?;
                    if c == ']' {
                        break;
                    }
                }
                // Continue scanning for the next member after the attribute.
                self.drop_whitespace_and_comments();
            }
            if self.try_drop_keyword("top") {
                // Allow whitespace/comments between `top` and the next keyword.
                self.drop_whitespace_and_comments();
                if self.peek_keyword_is("fn") {
                    let f = self.parse_fn()?;
                    top_name = Some(f.name.clone());
                    members.push(PackageMember::Function(f));
                } else if self.peek_keyword_is("block") {
                    // Allow top block (even if not present in inputs yet).
                    let (f, port_info) = self.parse_block_to_fn_with_ports()?;
                    top_name = Some(f.name.clone());
                    members.push(PackageMember::Block { func: f, port_info });
                } else {
                    return Err(ParseError::new(format!(
                        "expected fn or block after top; rest: {:?}",
                        self.rest()
                    )));
                }
            } else if self.peek_keyword_is("fn") {
                let f = self.parse_fn()?;
                members.push(PackageMember::Function(f));
            } else if self.peek_keyword_is("block") {
                let (f, port_info) = self.parse_block_to_fn_with_ports()?;
                members.push(PackageMember::Block { func: f, port_info });
            } else if self.peek_keyword_is("proc") {
                return Err(ParseError::new(format!(
                    "only functions are supported, got proc; rest: {:?}",
                    self.rest()
                )));
            } else if self.peek_keyword_is("chan") {
                return Err(ParseError::new(format!(
                    "only functions are supported, got chan; rest: {:?}",
                    self.rest()
                )));
            } else if self.peek_keyword_is("file_number") {
                self.parse_file_number(&mut file_table)?;
            } else {
                return Err(ParseError::new(format!(
                    "expected top, fn, block, or file_number, got {:?}; rest: {:?}",
                    self.peekc(),
                    self.rest()
                )));
            }
        }
        Ok(ir::Package {
            name: package_name,
            file_table,
            members,
            top_name,
        })
    }
}

/// Emits a combinational block text from an `ir::Fn`.
///
/// If `output_names` is provided, it determines the names and order of outputs
/// in the block header and `output_port` lines. If not provided, a single
/// output is named `out`, and multiple outputs are named `out0`, `out1`, ...
pub fn emit_fn_as_block(
    f: &ir::Fn,
    output_names: Option<&[String]>,
    port_ids: Option<&BlockPortInfo>,
) -> String {
    // Helper to get reference name for a node as used in operand positions.
    let get_ref_name = |nr: ir::NodeRef| -> String {
        let n = f.get_node(nr);
        match n.payload {
            ir::NodePayload::GetParam(_) => n.name.clone().unwrap(),
            _ => {
                if let Some(ref name) = n.name {
                    name.clone()
                } else {
                    format!("{}.{}", n.payload.get_operator(), n.text_id)
                }
            }
        }
    };

    // Determine outputs, guided by provided port ids if present.
    let (ret_nodes, ret_types, ret_tuple_index): (Vec<ir::NodeRef>, Vec<ir::Type>, Option<usize>) =
        if let Some(pi) = port_ids {
            let expected = pi.output_names.len();
            if expected == 0 {
                (Vec::new(), Vec::new(), None)
            } else if expected == 1 {
                if let Some(ret_nr) = f.ret_node_ref {
                    let ret_node = f.get_node(ret_nr);
                    (vec![ret_nr], vec![ret_node.ty.clone()], None)
                } else {
                    (Vec::new(), Vec::new(), None)
                }
            } else {
                if let Some(ret_nr) = f.ret_node_ref {
                    let ret_node = f.get_node(ret_nr);
                    match &ret_node.payload {
                        ir::NodePayload::Tuple(elems) => {
                            let mut types = Vec::new();
                            if let ir::Type::Tuple(tys) = &ret_node.ty {
                                for t in tys.iter() {
                                    types.push((**t).clone());
                                }
                            } else {
                                panic!("tuple return node must have tuple type");
                            }
                            (elems.clone(), types, Some(ret_nr.index))
                        }
                        _ => panic!(
                            "expected tuple return matching {} outputs, found non-tuple",
                            expected
                        ),
                    }
                } else {
                    (Vec::new(), Vec::new(), None)
                }
            }
        } else if let Some(ret_nr) = f.ret_node_ref {
            let ret_node = f.get_node(ret_nr);
            match &ret_node.payload {
                ir::NodePayload::Tuple(elems) => {
                    let mut types = Vec::new();
                    if let ir::Type::Tuple(tys) = &ret_node.ty {
                        for t in tys.iter() {
                            types.push((**t).clone());
                        }
                    } else {
                        // Strong invariant: tuple payload must have tuple type.
                        panic!("tuple return node must have tuple type");
                    }
                    (elems.clone(), types, Some(ret_nr.index))
                }
                _ => (vec![ret_nr], vec![ret_node.ty.clone()], None),
            }
        } else {
            // No explicit return node; treat as zero outputs.
            (Vec::new(), Vec::new(), None)
        };

    // Decide output names.
    let decided_out_names: Vec<String> = if let Some(names) = output_names {
        assert!(
            names.len() == ret_nodes.len(),
            "output_names length must match number of outputs"
        );
        names.to_vec()
    } else if let Some(pi) = port_ids {
        assert!(
            pi.output_names.len() == ret_nodes.len(),
            "BlockPortInfo.output_names length must match number of outputs"
        );
        pi.output_names.clone()
    } else if ret_nodes.len() == 1 {
        vec!["out".to_string()]
    } else {
        (0..ret_nodes.len()).map(|i| format!("out{}", i)).collect()
    };

    // Construct header: inputs first, then outputs.
    let mut header_parts: Vec<String> = Vec::new();
    for p in f.params.iter() {
        header_parts.push(format!("{}: {}", p.name, p.ty));
    }
    for (i, ty) in ret_types.iter().enumerate() {
        header_parts.push(format!("{}: {}", decided_out_names[i], ty));
    }

    // Emit body lines.
    let mut lines: Vec<String> = Vec::new();
    // input_port lines for each param (in order).
    for p in f.params.iter() {
        let input_id = if let Some(pi) = port_ids {
            *pi.input_port_ids
                .get(&p.name)
                .expect("input id missing in BlockPortInfo for parameter")
        } else {
            p.id.get_wrapped_id()
        };
        lines.push(format!(
            "  {}: {} = input_port(name={}, id={})",
            p.name, p.ty, p.name, input_id
        ));
    }
    // Emit non-param nodes as IR lines.
    for (i, n) in f.nodes.iter().enumerate() {
        if i == 0 {
            continue;
        }
        if matches!(n.payload, ir::NodePayload::GetParam(_)) {
            continue;
        }
        if let Some(idx) = ret_tuple_index {
            if i == idx {
                continue;
            }
        }
        if let Some(s) = n.to_string(f) {
            lines.push(format!("  {}", s));
        }
    }

    // Compute output ids: prefer provided ids; otherwise choose fresh ids after
    // max. If a provided BlockPortInfo is missing a name we decided, allocate
    // a fresh id for that output instead of panicking.
    let mut next_id: usize = f.nodes.iter().map(|n| n.text_id).max().unwrap_or(0) + 1;
    for (i, nr) in ret_nodes.iter().enumerate() {
        let out_name = &decided_out_names[i];
        let val_name = get_ref_name(*nr);
        let out_id = if let Some(pi) = port_ids {
            match pi.output_port_ids.get(out_name) {
                Some(id) => *id,
                None => {
                    let id = next_id;
                    next_id += 1;
                    id
                }
            }
        } else {
            let id = next_id;
            next_id += 1;
            id
        };
        lines.push(format!(
            "  {}: () = output_port({}, name={}, id={})",
            out_name, val_name, out_name, out_id
        ));
    }

    format!(
        "block {}({}) {{\n{}\n}}",
        f.name,
        header_parts.join(", "),
        lines.join("\n")
    )
}

impl Parser {
    /// Parses a package from the input and validates the resulting IR.
    pub fn parse_and_validate_package(&mut self) -> Result<ir::Package, ParseOrValidateError> {
        let pkg = self.parse_package()?;
        ir_validate::validate_package(&pkg)?;
        Ok(pkg)
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
        match node.payload {
            ir::NodePayload::Tuple(elems) => {
                assert_eq!(elems.len(), 2);
                assert_eq!(elems[0], ir::NodeRef { index: 1 });
                assert_eq!(elems[1], ir::NodeRef { index: 2 });
            }
            other => panic!("expected tuple node, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_token_literal_node_value() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        let input = "literal.7: token = literal(value=token, id=7)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        match node.payload {
            ir::NodePayload::Literal(v) => {
                assert_eq!(v.to_string(), "token");
            }
            other => panic!("expected literal node, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_array_literal_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "literal.4: bits[8][5] = literal(value=[119, 111, 114, 108, 100], id=4)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut IrNodeEnv::new()).unwrap();
        if let ir::NodePayload::Literal(v) = node.payload {
            assert_eq!(
                v.to_string(),
                "[bits[8]:119, bits[8]:111, bits[8]:114, bits[8]:108, bits[8]:100]"
            );
            assert_eq!(v.get_element_count().unwrap(), 5);
        } else {
            panic!("expected literal node");
        }
    }

    #[test]
    fn test_parse_after_all_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("trace".to_string()), 17, ir::NodeRef { index: 17 });
        let input = "after_all.19: token = after_all(trace.17, id=19)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        assert_eq!(
            node.payload,
            ir::NodePayload::AfterAll(vec![ir::NodeRef { index: 17 }])
        );
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
    fn test_parse_after_all_node_nullary() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        let input = "after_all.19: token = after_all(id=19)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        assert_eq!(node.payload, ir::NodePayload::AfterAll(Vec::new()));
    }

    #[test]
    fn test_round_trip_fn_with_nullary_after_all() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "fn f() -> token {\n  ret after_all.19: token = after_all(id=19)\n}\n";
        let mut parser = Parser::new(input);
        let f = parser.parse_fn().unwrap();
        assert_eq!(f.to_string(), input.trim_end());
    }

    #[test]
    fn test_parse_bit_slice_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("x".to_string()), 1, ir::NodeRef { index: 1 });
        let input = "bit_slice.6: bits[2] = bit_slice(x, start=0, width=2, id=6)";
        let mut parser = Parser::new(input);
        let node = parser.parse_node(&mut node_env).unwrap();
        match node.payload {
            ir::NodePayload::BitSlice { arg, start, width } => {
                assert_eq!(arg, ir::NodeRef { index: 1 });
                assert_eq!(start, 0);
                assert_eq!(width, 2);
            }
            other => panic!("expected bit_slice node, got {:?}", other),
        }
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
        match node.payload {
            ir::NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                assert_eq!(selector, ir::NodeRef { index: 1 });
                assert_eq!(cases, vec![ir::NodeRef { index: 2 }]);
                assert_eq!(default, Some(ir::NodeRef { index: 3 }));
            }
            other => panic!("expected priority_sel node, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_explicit_param_node() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = "x: bits[2][1] = param(name=x, id=1)";
        let mut parser = Parser::new(input);
        let mut env = IrNodeEnv::new();
        // Seed environment to simulate presence of header param x id=1
        env.add(Some("x".to_string()), 1, ir::NodeRef { index: 1 });
        let node = parser.parse_node(&mut env).unwrap();
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
        assert_eq!(parsed_package.to_string(), xlsynth_package_str);
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
                invariant_args,
            } => {
                assert_eq!(init.index, 5);
                assert_eq!(trip_count, 7);
                assert_eq!(stride, 1);
                assert_eq!(body, "body1");
                assert_eq!(invariant_args.len(), 0);
            }
            _ => panic!("expected counted_for payload"),
        }
    }

    #[test]
    fn test_parse_counted_for_node_with_invariant_args() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("init_lit".to_string()), 5, ir::NodeRef { index: 5 });
        node_env.add(Some("x".to_string()), 7, ir::NodeRef { index: 7 });
        node_env.add(Some("y".to_string()), 8, ir::NodeRef { index: 8 });
        let input = "counted_for.6: bits[11] = counted_for(init_lit.5, trip_count=4, stride=2, body=body2, invariant_args=[x, y], id=6)";
        let mut parser = Parser::new(input);
        let node = parser
            .parse_node(&mut node_env)
            .expect("parse counted_for node with invariant_args");
        match node.payload {
            ir::NodePayload::CountedFor {
                init,
                trip_count,
                stride,
                body,
                invariant_args,
            } => {
                assert_eq!(init.index, 5);
                assert_eq!(trip_count, 4);
                assert_eq!(stride, 2);
                assert_eq!(body, "body2");
                assert_eq!(invariant_args.len(), 2);
                assert_eq!(invariant_args[0].index, 7);
                assert_eq!(invariant_args[1].index, 8);
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
        if let ir::NodePayload::GetParam(pid) = &ret_node.payload {
            assert_eq!(pid.get_wrapped_id(), 29);
        } else {
            panic!("Expected GetParam node payload");
        }

        // Ensure printing retains a non-empty body with the param return.
        let printed = format!("{}\n", f.to_string());
        assert_eq!(printed, input);
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
    fn test_parse_block_to_fn_simple() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = r#"#[signature("""module_name: \"my_main\" data_ports { direction: PORT_DIRECTION_INPUT name: \"x\" width: 8 type { type_enum: BITS bit_count: 8 } } data_ports { direction: PORT_DIRECTION_OUTPUT name: \"out\" width: 8 type { type_enum: BITS bit_count: 8 } } combinational { } """)]
block my_main(x: bits[8], out: bits[8]) {
  #![provenance(name=\"my_main\", kind=\"function\")]
  x: bits[8] = input_port(name=x, id=5)
  one: bits[8] = literal(value=1, id=6)
  add.7: bits[8] = add(x, one, id=7)
  out: () = output_port(add.7, name=out, id=8)
}"#;
        let mut parser = Parser::new(input);
        let f = parser.parse_block_to_fn().unwrap();
        let want = "fn my_main(x: bits[8] id=5) -> bits[8] {\n  one: bits[8] = literal(value=1, id=6)\n  ret add.7: bits[8] = add(x, one, id=7)\n}";
        assert_eq!(f.to_string(), want);
    }

    // -- Test constants for round-trip
    const BLK_ADD_TWO_INPUTS_ONE_OUTPUT: &str = r#"block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  add.3: bits[32] = add(a, b, id=3)
  out: () = output_port(add.3, name=out, id=4)
}"#;

    const BLK_TWO_INPUTS_TWO_OUTPUTS_RT: &str = r#"block my_block(a: bits[32], b: bits[32], a_out: bits[32], b_out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=3)
  a_out: () = output_port(a, name=a_out, id=5)
  b_out: () = output_port(b, name=b_out, id=6)
}"#;

    #[test]
    fn test_roundtrip_block_parse_then_emit_single_output() {
        let _ = env_logger::builder().is_test(true).try_init();
        // Parse -> Fn
        let mut parser = Parser::new(BLK_ADD_TWO_INPUTS_ONE_OUTPUT);
        let f = parser.parse_block_to_fn().unwrap();
        // Emit -> block text (provide output name to match header)
        // When parsing from block, preserve original port ids via BlockPortInfo.
        let mut parser2 = Parser::new(BLK_ADD_TWO_INPUTS_ONE_OUTPUT);
        let (_f2, port_info) = parser2.parse_block_to_fn_with_ports().unwrap();
        let emitted = emit_fn_as_block(&f, Some(&["out".to_string()]), Some(&port_info));
        assert_eq!(emitted, BLK_ADD_TWO_INPUTS_ONE_OUTPUT);
    }

    #[test]
    fn test_roundtrip_block_parse_then_emit_multi_output() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut parser = Parser::new(BLK_TWO_INPUTS_TWO_OUTPUTS_RT);
        let (f, port_info) = parser.parse_block_to_fn_with_ports().unwrap();
        let emitted = emit_fn_as_block(
            &f,
            Some(&["a_out".to_string(), "b_out".to_string()]),
            Some(&port_info),
        );
        assert_eq!(emitted, BLK_TWO_INPUTS_TWO_OUTPUTS_RT);
    }
    #[test]
    fn test_parse_block_to_fn_add_two_inputs_one_output() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = r#"block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  add.3: bits[32] = add(a, b, id=3)
  out: () = output_port(add.3, name=out, id=4)
}"#;
        let mut parser = Parser::new(input);
        let f = parser.parse_block_to_fn().unwrap();
        let want = "fn my_block(a: bits[32] id=1, b: bits[32] id=2) -> bits[32] {\n  ret add.3: bits[32] = add(a, b, id=3)\n}";
        assert_eq!(f.to_string(), want);
    }

    #[test]
    fn test_parse_block_to_fn_multi_output_tuple_return() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = r#"block my_block(a: bits[32], a_out: bits[32], b: bits[32], b_out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=3)
  a_out: () = output_port(a, name=a_out, id=2)
  b_out: () = output_port(b, name=b_out, id=4)
}"#;
        let mut parser = Parser::new(input);
        let f = parser.parse_block_to_fn().unwrap();
        let want = "fn my_block(a: bits[32] id=1, b: bits[32] id=3) -> (bits[32], bits[32]) {
  ret tuple.4: (bits[32], bits[32]) = tuple(a, b, id=4)
}";
        assert_eq!(f.to_string(), want);
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
