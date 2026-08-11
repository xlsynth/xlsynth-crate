// SPDX-License-Identifier: Apache-2.0

//! Deterministic Verilog and SystemVerilog emission for the file-owned AST.

use std::fmt::Write;

use crate::VastFile;
use crate::model::{
    AlwaysKind, BinaryOp, CaseStatement, Conditional, DataKind, Expr, ExprData, FileItem,
    GenerateLoop, Instantiation, IntegerTypeKind, MemberData, ModuleData, ModulePortDirection,
    StatementData, TypeData, UnaryOp, VastAlwaysBase, VastDataType, VastFileType,
    VastStatementBlock,
};

/// Emits all file members in insertion order, with one terminating newline
/// each.
pub(crate) fn emit_file(file: &VastFile) -> String {
    let mut result = String::new();
    for item in &file.ast.file_items {
        match item {
            FileItem::Module(module) => {
                result.push_str(&emit_module(file, &file.ast.modules[module.0.index]));
            }
            FileItem::Include(path) => {
                write!(result, "`include \"{path}\"").expect("writing to String cannot fail");
            }
            FileItem::Comment(text) => result.push_str(&emit_comment(text)),
            FileItem::BlankLine => {
                // The common terminating newline below is the entire blank
                // line.
            }
        }
        result.push('\n');
    }
    result
}

/// Emits one expression with the same parenthesization rules as XLS VAST.
pub(crate) fn emit_expr(file: &VastFile, expr: Expr) -> String {
    match &file.ast.expressions[expr.0.index] {
        ExprData::Name { name, .. } => name.clone(),
        ExprData::Literal { text, .. } => text.clone(),
        ExprData::Unary { op, arg } => {
            let operand = emit_expr(file, *arg);
            let needs_parentheses = precedence(file, *arg) < 12
                || matches!(file.ast.expressions[arg.0.index], ExprData::Unary { .. });
            format!(
                "{}{}",
                unary_spelling(*op),
                parenthesize_if(operand, needs_parentheses)
            )
        }
        ExprData::Binary { op, lhs, rhs } => {
            let current_precedence = binary_precedence(*op);
            let lhs_text = parenthesize_if(
                emit_expr(file, *lhs),
                precedence(file, *lhs) < current_precedence || is_reduction(file, *lhs),
            );
            let rhs_text = parenthesize_if(
                emit_expr(file, *rhs),
                precedence(file, *rhs) <= current_precedence || is_reduction(file, *rhs),
            );
            format!("{lhs_text} {} {rhs_text}", binary_spelling(*op))
        }
        ExprData::Ternary {
            condition,
            consequent,
            alternate,
        } => {
            let render_operand =
                |operand| parenthesize_if(emit_expr(file, operand), precedence(file, operand) <= 0);
            format!(
                "{} ? {} : {}",
                render_operand(*condition),
                render_operand(*consequent),
                render_operand(*alternate)
            )
        }
        ExprData::Index { subject, index } => {
            if is_scalar_logic_ref(file, *subject) {
                assert!(
                    literal_value(file, *index) == Some(0),
                    "a scalar signal can only be indexed with zero"
                );
                emit_expr(file, *subject)
            } else {
                format!("{}[{}]", emit_expr(file, *subject), emit_expr(file, *index))
            }
        }
        ExprData::Slice { subject, hi, lo } => {
            if is_scalar_logic_ref(file, *subject) {
                assert!(
                    literal_value(file, *hi) == Some(0) && literal_value(file, *lo) == Some(0),
                    "a scalar signal can only be sliced at [0:0]"
                );
                emit_expr(file, *subject)
            } else {
                format!(
                    "{}[{}:{}]",
                    emit_expr(file, *subject),
                    emit_expr(file, *hi),
                    emit_expr(file, *lo)
                )
            }
        }
        ExprData::Concat {
            replication,
            elements,
        } => {
            let joined = join_expressions(file, elements);
            match replication {
                Some(count) => format!("{{{}{{{joined}}}}}", emit_expr(file, *count)),
                None => format!("{{{joined}}}"),
            }
        }
        ExprData::ArrayAssignmentPattern(elements) => {
            format!("'{{{}}}", join_expressions(file, elements))
        }
        ExprData::WidthCast { width, value } => emit_width_cast(file, *width, *value),
        ExprData::TypeCast { data_type, value } => emit_data_type_cast(file, *data_type, *value),
        ExprData::PosEdge(value) => format!("posedge {}", emit_expr(file, *value)),
        ExprData::NegEdge(value) => format!("negedge {}", emit_expr(file, *value)),
        ExprData::Macro { name, args } => match args {
            Some(expressions) => format!("`{name}({})", join_expressions(file, expressions)),
            None => format!("`{name}"),
        },
    }
}

/// Emits an expression-width cast, preserving the upstream parenthesis rules.
fn emit_width_cast(file: &VastFile, width: Expr, value: Expr) -> String {
    let rendered_width = emit_expr(file, width);
    let needs_parentheses = !is_digits_or_identifier(&rendered_width);
    let rendered_width = parenthesize_if(rendered_width, needs_parentheses);
    format!("{rendered_width}'({})", emit_expr(file, value))
}

/// Emits a complete, syntactically valid casting type rather than a declaration
/// fragment.
fn emit_data_type_cast(file: &VastFile, data_type: VastDataType, value: Expr) -> String {
    match &file.ast.data_types[data_type.0.index] {
        TypeData::Scalar { signed } => {
            let cast = format!("logic'({})", emit_expr(file, value));
            if *signed {
                format!("signed'({cast})")
            } else {
                cast
            }
        }
        TypeData::BitVector { width, signed, .. } => {
            let sign = if *signed { "signed" } else { "unsigned" };
            format!("{sign}'({})", emit_width_cast(file, *width, value))
        }
        TypeData::Integer { signed, kind } => {
            let keyword = match kind {
                IntegerTypeKind::Integer => "integer",
                IntegerTypeKind::Int => "int",
            };
            let cast = format!("{keyword}'({})", emit_expr(file, value));
            if *signed {
                cast
            } else {
                format!("unsigned'({cast})")
            }
        }
        TypeData::Extern { .. } | TypeData::PackedArray { .. } | TypeData::UnpackedArray { .. } => {
            format!(
                "{}'({})",
                emit_type(file, data_type),
                emit_expr(file, value)
            )
        }
    }
}

/// Returns the textual representation of a data type without its identifier.
pub(crate) fn emit_type(file: &VastFile, data_type: VastDataType) -> String {
    match &file.ast.data_types[data_type.0.index] {
        TypeData::Scalar { signed } => {
            if *signed {
                "signed".to_owned()
            } else {
                String::new()
            }
        }
        TypeData::BitVector {
            width,
            width_value,
            signed,
        } => {
            let signed_prefix = if *signed { "signed " } else { "" };
            format!(
                "{signed_prefix}[{}:0]",
                width_limit(file, *width, *width_value)
            )
        }
        TypeData::Extern { package, name } => match package {
            Some(package) => format!("{package}::{name}"),
            None => name.clone(),
        },
        TypeData::Integer { signed, .. } => {
            if *signed {
                String::new()
            } else {
                "unsigned".to_owned()
            }
        }
        TypeData::PackedArray {
            element,
            dimensions,
        } => {
            let dimension_text = packed_dimensions(dimensions);
            match &file.ast.data_types[element.0.index] {
                TypeData::BitVector {
                    width,
                    width_value,
                    signed,
                } => {
                    let signed_prefix = if *signed { "signed " } else { "" };
                    format!(
                        "{signed_prefix}{dimension_text}[{}:0]",
                        width_limit(file, *width, *width_value)
                    )
                }
                _ => {
                    let element_type = emit_type(file, *element);
                    if element_type.is_empty() {
                        dimension_text
                    } else {
                        format!("{element_type} {dimension_text}")
                    }
                }
            }
        }
        TypeData::UnpackedArray { element, .. } => emit_type(file, *element),
    }
}

/// Emits a type around its identifier, placing unpacked dimensions afterwards.
fn emit_type_with_identifier(file: &VastFile, data_type: VastDataType, name: &str) -> String {
    match &file.ast.data_types[data_type.0.index] {
        TypeData::UnpackedArray {
            element,
            dimensions,
        } => {
            let mut result = emit_type_with_identifier(file, *element, name);
            for dimension in dimensions {
                match file.file_type {
                    VastFileType::SystemVerilog => {
                        write!(result, "[{dimension}]").expect("writing to String cannot fail");
                    }
                    VastFileType::Verilog => {
                        write!(result, "[0:{}]", dimension - 1)
                            .expect("writing to String cannot fail");
                    }
                }
            }
            result
        }
        _ => {
            let rendered_type = emit_type(file, data_type);
            if rendered_type.is_empty() {
                name.to_owned()
            } else {
                format!("{rendered_type} {name}")
            }
        }
    }
}

/// Emits a module header and its ordered module members.
fn emit_module(file: &VastFile, module: &ModuleData) -> String {
    let mut result = format!("module {}", module.name);
    if !module.parameter_ports.is_empty() {
        result.push_str(" #(\n");
        for (index, parameter) in module.parameter_ports.iter().enumerate() {
            if index != 0 {
                result.push_str(",\n");
            }
            result.push_str("  parameter ");
            if let Some(data_type) = parameter.data_type {
                result.push_str(&emit_declaration_without_semicolon(
                    file,
                    parameter_data_kind(file, data_type),
                    data_type,
                    &parameter.name,
                ));
            } else {
                result.push_str(&parameter.name);
            }
            write!(result, " = {}", emit_expr(file, parameter.value))
                .expect("writing to String cannot fail");
        }
        result.push_str("\n)");
        if !module.ports.is_empty() {
            result.push(' ');
        }
    }

    if module.ports.is_empty() {
        result.push_str(";\n");
    } else {
        result.push_str("(\n");
        for (index, handle) in module.ports.iter().enumerate() {
            if index != 0 {
                result.push_str(",\n");
            }
            let port = &file.ast.ports[handle.0.index];
            let direction = match port.direction {
                ModulePortDirection::Input => "input",
                ModulePortDirection::Output => "output",
                ModulePortDirection::InOut => "inout",
            };
            let kind = if is_user_defined(file, port.data_type) {
                DataKind::User
            } else {
                port.kind
            };
            write!(
                result,
                "  {direction} {}",
                emit_declaration_without_semicolon(file, kind, port.data_type, &port.name)
            )
            .expect("writing to String cannot fail");
        }
        result.push_str("\n);\n");
    }

    if module.members.is_empty() {
        result.push('\n');
    } else {
        for member in &module.members {
            result.push_str(&indent(&emit_member(file, member)));
            result.push('\n');
        }
    }
    result.push_str("endmodule");
    result
}

/// Chooses the complete declaration kind corresponding to a parameter type.
fn parameter_data_kind(file: &VastFile, data_type: VastDataType) -> DataKind {
    match &file.ast.data_types[data_type.0.index] {
        TypeData::Scalar { .. } | TypeData::BitVector { .. } => DataKind::Logic,
        TypeData::Extern { .. } => DataKind::User,
        TypeData::Integer { kind, .. } => match kind {
            IntegerTypeKind::Integer => DataKind::Integer,
            IntegerTypeKind::Int => DataKind::Int,
        },
        TypeData::PackedArray { element, .. } | TypeData::UnpackedArray { element, .. } => {
            parameter_data_kind(file, *element)
        }
    }
}

/// Emits one module member without the containing module's indentation.
fn emit_member(file: &VastFile, member: &MemberData) -> String {
    match member {
        MemberData::Declaration {
            kind,
            name,
            data_type,
        } => format!(
            "{};",
            emit_declaration_without_semicolon(file, *kind, *data_type, name)
        ),
        MemberData::ContinuousAssignment { lhs, rhs } => {
            format!(
                "assign {} = {};",
                emit_expr(file, *lhs),
                emit_expr(file, *rhs)
            )
        }
        MemberData::Instantiation(handle) => emit_instantiation(file, *handle),
        MemberData::Always(handle) => emit_always(file, *handle),
        MemberData::Generate(handle) => emit_generate(file, *handle),
        MemberData::Parameter {
            local,
            def,
            name,
            rhs,
            int_kind,
        } => {
            let keyword = if *local { "localparam" } else { "parameter" };
            let declaration = if let Some(definition) = def {
                let definition = &file.ast.defs[definition.0.index];
                emit_declaration_without_semicolon(
                    file,
                    definition.kind,
                    definition.data_type,
                    &definition.name,
                )
            } else if *int_kind {
                format!("int {name}")
            } else {
                name.clone()
            };
            format!("{keyword} {declaration} = {};", emit_expr(file, *rhs))
        }
        MemberData::Conditional(handle) => emit_conditional(file, *handle),
        MemberData::Comment(text) => emit_comment(text),
        MemberData::BlankLine => String::new(),
        MemberData::Inline(text) => text.clone(),
        MemberData::Macro { expr, semicolon } => {
            let suffix = if *semicolon { ";" } else { "" };
            format!("{}{suffix}", emit_expr(file, *expr))
        }
    }
}

/// Combines declaration kind and type without introducing unnecessary spaces.
fn emit_declaration_without_semicolon(
    file: &VastFile,
    kind: DataKind,
    data_type: VastDataType,
    name: &str,
) -> String {
    let kind_spelling = match kind {
        DataKind::Reg => "reg",
        DataKind::Wire => "wire",
        DataKind::Logic => "logic",
        DataKind::Integer => "integer",
        DataKind::Int => "int",
        DataKind::Genvar => "genvar",
        DataKind::User | DataKind::UntypedEnum => "",
    };
    let type_and_name = emit_type_with_identifier(file, data_type, name);
    if kind_spelling.is_empty() {
        type_and_name
    } else {
        format!("{kind_spelling} {type_and_name}")
    }
}

/// Emits a nested generate loop and all of its ordered module members.
fn emit_generate(file: &VastFile, handle: GenerateLoop) -> String {
    let generate = &file.ast.generates[handle.0.index];
    let label = generate
        .label
        .as_ref()
        .map_or_else(String::new, |name| format!(" : {name}"));
    let mut result = format!(
        "for (genvar {} = {}; {} < {}; {} = {} + 1) begin{label}",
        generate.name,
        emit_expr(file, generate.start),
        generate.name,
        emit_expr(file, generate.end),
        generate.name,
        generate.name
    );
    for member in &generate.members {
        result.push('\n');
        result.push_str(&indent(&emit_member(file, member)));
    }
    result.push_str("\nend");
    result
}

/// Emits an always, always_ff, or always_comb procedural block.
fn emit_always(file: &VastFile, handle: VastAlwaysBase) -> String {
    let block = &file.ast.always[handle.0.index];
    match block.kind {
        AlwaysKind::AlwaysComb => {
            format!("always_comb {}", emit_statement_block(file, block.block))
        }
        AlwaysKind::AlwaysFF | AlwaysKind::AlwaysAt => {
            let keyword = match block.kind {
                AlwaysKind::AlwaysFF => "always_ff",
                AlwaysKind::AlwaysAt => "always",
                AlwaysKind::AlwaysComb => unreachable!("always_comb has a dedicated emission"),
            };
            let sensitivity_list = block
                .sensitivity_list
                .iter()
                .map(|expr| emit_expr(file, *expr))
                .collect::<Vec<_>>()
                .join(" or ");
            format!(
                "{keyword} @ ({sensitivity_list}) {}",
                emit_statement_block(file, block.block)
            )
        }
    }
}

/// Emits a begin/end-delimited sequence of procedural statements.
fn emit_statement_block(file: &VastFile, handle: VastStatementBlock) -> String {
    let block = &file.ast.blocks[handle.0.index];
    if block.statements.is_empty() {
        return "begin end".to_owned();
    }
    let mut result = "begin\n".to_owned();
    for statement in &block.statements {
        result.push_str(&indent(&emit_statement(file, statement)));
        result.push('\n');
    }
    result.push_str("end");
    result
}

/// Emits one statement without its containing begin/end block's indentation.
fn emit_statement(file: &VastFile, statement: &StatementData) -> String {
    match statement {
        StatementData::BlockingAssignment { lhs, rhs } => {
            format!("{} = {};", emit_expr(file, *lhs), emit_expr(file, *rhs))
        }
        StatementData::NonblockingAssignment { lhs, rhs } => {
            format!("{} <= {};", emit_expr(file, *lhs), emit_expr(file, *rhs))
        }
        StatementData::ContinuousAssignment { lhs, rhs } => {
            format!(
                "assign {} = {};",
                emit_expr(file, *lhs),
                emit_expr(file, *rhs)
            )
        }
        StatementData::Conditional(handle) => emit_conditional(file, *handle),
        StatementData::Case(handle) => emit_case(file, *handle),
        StatementData::Comment(text) => emit_comment(text),
        StatementData::BlankLine => String::new(),
        StatementData::Inline(text) => text.clone(),
    }
}

/// Emits an if statement and its optional else-if and else clauses.
fn emit_conditional(file: &VastFile, handle: Conditional) -> String {
    let conditional = &file.ast.conditionals[handle.0.index];
    let mut result = format!(
        "if ({}) {}",
        emit_expr(file, conditional.condition),
        emit_statement_block(file, conditional.consequent)
    );
    for (condition, block) in &conditional.alternates {
        result.push_str(" else ");
        if let Some(condition) = condition {
            write!(result, "if ({}) ", emit_expr(file, *condition))
                .expect("writing to String cannot fail");
        }
        result.push_str(&emit_statement_block(file, *block));
    }
    result
}

/// Emits an ordinary case statement and its case/default arms.
fn emit_case(file: &VastFile, handle: CaseStatement) -> String {
    let case = &file.ast.cases[handle.0.index];
    let mut result = format!("case ({})", emit_expr(file, case.selector));
    for (pattern, block) in &case.arms {
        let label = pattern.map_or_else(|| "default".to_owned(), |expr| emit_expr(file, expr));
        result.push('\n');
        result.push_str(&indent(&format!(
            "{label}: {}",
            emit_statement_block(file, *block)
        )));
    }
    result.push_str("\nendcase");
    result
}

/// Emits an instantiation, including multiline named parameters and ports.
fn emit_instantiation(file: &VastFile, handle: Instantiation) -> String {
    let instantiation = &file.ast.instantiations[handle.0.index];
    let mut result = format!("{} ", instantiation.module_name);
    if !instantiation.parameters.is_empty() {
        result.push_str("#(\n");
        for (index, (name, expr)) in instantiation.parameters.iter().enumerate() {
            if index != 0 {
                result.push_str(",\n");
            }
            write!(result, "  .{name}({})", emit_expr(file, *expr))
                .expect("writing to String cannot fail");
        }
        result.push_str("\n) ");
    }
    write!(result, "{} (\n", instantiation.instance_name).expect("writing to String cannot fail");
    for (index, (name, expr)) in instantiation.ports.iter().enumerate() {
        if index != 0 {
            result.push_str(",\n");
        }
        let value = expr.map_or_else(String::new, |expr| emit_expr(file, expr));
        write!(result, "  .{name}({value})").expect("writing to String cannot fail");
    }
    result.push_str("\n);");
    result
}

/// Prefixes every nonempty line with two spaces while preserving blank lines.
fn indent(text: &str) -> String {
    text.split('\n')
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("  {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Emits every line of a comment with the corresponding Verilog prefix.
fn emit_comment(text: &str) -> String {
    text.split('\n')
        .map(|line| {
            if line.is_empty() {
                "//".to_owned()
            } else {
                format!("// {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Joins expressions with the comma spacing used by the C++ VAST emitter.
fn join_expressions(file: &VastFile, expressions: &[Expr]) -> String {
    expressions
        .iter()
        .map(|expr| emit_expr(file, *expr))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Returns whether a named expression directly references a scalar signal.
fn is_scalar_logic_ref(file: &VastFile, expression: Expr) -> bool {
    match &file.ast.expressions[expression.0.index] {
        ExprData::Name {
            data_type: Some(data_type),
            ..
        } => matches!(
            file.ast.data_types[data_type.0.index],
            TypeData::Scalar { .. }
        ),
        _ => false,
    }
}

/// Returns the small integer value of a literal when it is available.
fn literal_value(file: &VastFile, expression: Expr) -> Option<i64> {
    match &file.ast.expressions[expression.0.index] {
        ExprData::Literal { value, .. } => *value,
        _ => None,
    }
}

/// Emits literal widths folded by one and symbolic widths as subtraction.
fn width_limit(file: &VastFile, width: Expr, value: Option<i64>) -> String {
    match value.or_else(|| literal_value(file, width)) {
        Some(value) => (value - 1).to_string(),
        None => {
            let rendered = emit_expr(file, width);
            let rendered = parenthesize_if(rendered, precedence(file, width) < 9);
            format!("{rendered} - 1")
        }
    }
}

/// Emits all packed-array dimensions consecutively and without spaces.
fn packed_dimensions(dimensions: &[i64]) -> String {
    let mut result = String::new();
    for dimension in dimensions {
        write!(result, "[{}:0]", dimension - 1).expect("writing to String cannot fail");
    }
    result
}

/// Reports whether the data type ultimately names a user-defined type.
fn is_user_defined(file: &VastFile, data_type: VastDataType) -> bool {
    match &file.ast.data_types[data_type.0.index] {
        TypeData::Extern { .. } => true,
        TypeData::PackedArray { element, .. } | TypeData::UnpackedArray { element, .. } => {
            is_user_defined(file, *element)
        }
        _ => false,
    }
}

/// Returns Verilog operator precedence, using 13 for atomic expressions.
fn precedence(file: &VastFile, expr: Expr) -> i8 {
    match &file.ast.expressions[expr.0.index] {
        ExprData::Unary { .. } => 12,
        ExprData::Binary { op, .. } => binary_precedence(*op),
        ExprData::Ternary { .. } => 0,
        _ => 13,
    }
}

/// Returns the precedence associated with one binary operator.
fn binary_precedence(op: BinaryOp) -> i8 {
    match op {
        BinaryOp::Power => 11,
        BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 10,
        BinaryOp::Add | BinaryOp::Sub => 9,
        BinaryOp::Shll | BinaryOp::Shra | BinaryOp::Shrl => 8,
        BinaryOp::Ge | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Lt => 7,
        BinaryOp::Ne
        | BinaryOp::CaseNe
        | BinaryOp::Eq
        | BinaryOp::CaseEq
        | BinaryOp::NeX
        | BinaryOp::EqX => 6,
        BinaryOp::BitwiseAnd => 5,
        BinaryOp::BitwiseXor => 4,
        BinaryOp::BitwiseOr => 3,
        BinaryOp::LogicalAnd => 2,
        BinaryOp::LogicalOr => 1,
    }
}

/// Returns the source spelling of one unary operator.
fn unary_spelling(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Negate => "-",
        UnaryOp::BitwiseNot => "~",
        UnaryOp::LogicalNot => "!",
        UnaryOp::AndReduce => "&",
        UnaryOp::OrReduce => "|",
        UnaryOp::XorReduce => "^",
    }
}

/// Returns the source spelling of one binary operator.
fn binary_spelling(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
        BinaryOp::Power => "**",
        BinaryOp::BitwiseAnd => "&",
        BinaryOp::BitwiseOr => "|",
        BinaryOp::BitwiseXor => "^",
        BinaryOp::Shll => "<<",
        BinaryOp::Shra => ">>>",
        BinaryOp::Shrl => ">>",
        BinaryOp::Ne => "!=",
        BinaryOp::CaseNe | BinaryOp::NeX => "!==",
        BinaryOp::Eq => "==",
        BinaryOp::CaseEq | BinaryOp::EqX => "===",
        BinaryOp::Ge => ">=",
        BinaryOp::Gt => ">",
        BinaryOp::Le => "<=",
        BinaryOp::Lt => "<",
        BinaryOp::LogicalAnd => "&&",
        BinaryOp::LogicalOr => "||",
    }
}

/// Detects reductions, which must be parenthesized within binary expressions.
fn is_reduction(file: &VastFile, expr: Expr) -> bool {
    matches!(
        file.ast.expressions[expr.0.index],
        ExprData::Unary {
            op: UnaryOp::AndReduce | UnaryOp::OrReduce | UnaryOp::XorReduce,
            ..
        }
    )
}

/// Adds parentheses only when required by Verilog's precedence rules.
fn parenthesize_if(text: String, condition: bool) -> String {
    if condition { format!("({text})") } else { text }
}

/// Detects width-cast expressions that do not need cosmetic parentheses.
fn is_digits_or_identifier(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    if text.bytes().all(|byte| byte.is_ascii_digit()) {
        return true;
    }
    let mut bytes = text.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'$')
}

#[cfg(test)]
mod tests {
    use crate::{Expr, LiteralFormat, ModulePortDirection, VastFile, VastFileType, VastModule};

    /// Confirms that all lightweight AST references are independently copyable.
    #[test]
    fn expression_and_module_handles_are_copyable() {
        fn assert_copy<T: Copy>() {}

        assert_copy::<Expr>();
        assert_copy::<VastModule>();
    }

    /// Rejects handles from another file instead of silently indexing its
    /// arena.
    #[test]
    #[should_panic(expected = "VAST handle belongs to a different file")]
    fn cross_file_type_handles_are_rejected() {
        let mut first = VastFile::new(VastFileType::Verilog);
        let mut second = VastFile::new(VastFileType::Verilog);
        let module = first.add_module("first");
        let foreign_type = second.make_scalar_type();

        first.add_input(module, "bad", &foreign_type);
    }

    /// Rejects foreign expressions even when they are nested inside a builder.
    #[test]
    #[should_panic(expected = "VAST handle belongs to a different file")]
    fn cross_file_compound_expressions_are_rejected() {
        let mut first = VastFile::new(VastFileType::Verilog);
        let mut second = VastFile::new(VastFileType::Verilog);
        let local = first.make_plain_literal(1, &LiteralFormat::UnsignedDecimal);
        let foreign = second.make_plain_literal(2, &LiteralFormat::UnsignedDecimal);

        first.make_concat(&[&local, &foreign]);
    }

    /// Reports duplicate declarations with the existing stable C++ error text.
    #[test]
    fn duplicate_registers_and_logic_return_deterministic_errors() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("duplicates");
        let scalar = file.make_scalar_type();
        file.add_input(module, "z", &scalar);
        file.add_wire(module, "a", &scalar);
        file.add_wire(module, "middle", &scalar);

        let register_error = file
            .add_reg(module, "a", &scalar)
            .expect_err("register may not reuse an existing wire name");
        assert_eq!(
            register_error.to_string(),
            "FAILED_PRECONDITION: Attempted to declare reg with name 'a' multiple times \
             in the same module. Already defined: [a, middle, z]"
        );

        let logic_error = file
            .add_logic(module, "z", &scalar)
            .expect_err("logic may not reuse an existing port name");
        assert_eq!(
            logic_error.to_string(),
            "FAILED_PRECONDITION: Attempted to declare logic with name 'z' multiple times \
             in the same module. Already defined: [a, middle, z]"
        );
    }

    /// Keeps the former infallible-port behavior for an invalid duplicate.
    #[test]
    #[should_panic(expected = "module ports must have unique names")]
    fn duplicate_ports_panic() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let module = file.add_module("duplicate_port");
        let scalar = file.make_scalar_type();
        file.add_input(module, "signal", &scalar);
        file.add_output(module, "signal", &scalar);
    }

    /// Keeps the former infallible-wire behavior for an invalid duplicate.
    #[test]
    #[should_panic(expected = "module wires must have unique names")]
    fn duplicate_wires_panic() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let module = file.add_module("duplicate_wire");
        let scalar = file.make_scalar_type();
        file.add_wire(module, "signal", &scalar);
        file.add_wire(module, "signal", &scalar);
    }

    /// Matches C++ VAST, where parameter names are outside declaration
    /// tracking.
    #[test]
    fn duplicate_parameters_remain_permitted() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let module = file.add_module("parameters");
        let value = file.make_plain_literal(1, &LiteralFormat::UnsignedDecimal);
        file.add_parameter(module, "P", &value);
        file.add_parameter(module, "P", &value);

        assert_eq!(
            file.emit(),
            "module parameters;\n  parameter P = 1;\n  parameter P = 1;\nendmodule\n"
        );
    }

    /// Keeps all three port directions available through file-owned inspection.
    #[test]
    fn inout_ports_are_introspectable_and_emit_correctly() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let module = file.add_module("tristate");
        let scalar = file.make_scalar_type();
        file.add_inout(module, "bus", &scalar);

        let ports = file.module_ports(module);
        assert_eq!(ports.len(), 1);
        assert_eq!(file.port_direction(ports[0]), ModulePortDirection::InOut);
        assert_eq!(file.port_name(ports[0]), "bus");
        assert_eq!(
            file.emit(),
            "module tristate(\n  inout wire bus\n);\n\nendmodule\n"
        );
    }

    /// Preserves asymmetric associativity and mandatory reduction parentheses.
    #[test]
    fn expression_precedence_matches_cpp_vast() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let module = file.add_module("precedence");
        let scalar = file.make_scalar_type();
        let a = file.add_input(module, "a", &scalar).to_expr();
        let b = file.add_input(module, "b", &scalar).to_expr();
        let c = file.add_input(module, "c", &scalar).to_expr();

        let sum = file.make_add(&a, &b);
        let product = file.make_mul(&sum, &c);
        assert_eq!(file.emit_expression(&product), "(a + b) * c");

        let right_difference = file.make_sub(&b, &c);
        let difference = file.make_sub(&a, &right_difference);
        assert_eq!(file.emit_expression(&difference), "a - (b - c)");

        let reduced_a = file.make_or_reduce(&a);
        let reduced_b = file.make_and_reduce(&b);
        let reductions = file.make_logical_or(&reduced_a, &reduced_b);
        assert_eq!(file.emit_expression(&reductions), "(|a) || (&b)");

        let negated = file.make_logical_not(&a);
        let nested_unary = file.make_bitwise_not(&negated);
        assert_eq!(file.emit_expression(&nested_unary), "~(!a)");

        let inner_ternary = file.make_ternary(&a, &b, &c);
        let outer_ternary = file.make_ternary(&inner_ternary, &b, &c);
        assert_eq!(file.emit_expression(&outer_ternary), "(a ? b : c) ? b : c");
    }

    /// Eliminates illegal bit-selects and part-selects of scalar signals.
    #[test]
    fn scalar_zero_indices_are_elided() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let module = file.add_module("scalar");
        let scalar = file.make_scalar_type();
        let signal = file.add_input(module, "signal", &scalar);

        let index = file.make_index(&signal.to_indexable_expr(), 0);
        let slice = file.make_slice(&signal.to_indexable_expr(), 0, 0);
        assert_eq!(file.emit_expression(&index.to_expr()), "signal");
        assert_eq!(file.emit_expression(&slice.to_expr()), "signal");
    }

    /// Preserves the signed keyword even when a one-bit vector becomes scalar.
    #[test]
    fn signed_single_bit_vectors_remain_signed_scalars() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let module = file.add_module("signed_scalar");
        let scalar = file.make_bit_vector_type(1, true);
        file.add_input(module, "value", &scalar);

        assert!(file.type_is_signed(scalar));
        assert_eq!(
            file.emit(),
            "module signed_scalar(\n  input wire signed value\n);\n\nendmodule\n"
        );
    }

    /// Rejects constant zero widths before they can emit an invalid range.
    #[test]
    #[should_panic(expected = "bit-vector width must be greater than zero")]
    fn zero_expression_vector_widths_are_rejected() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let zero = file.make_plain_literal(0, &LiteralFormat::Default);

        file.make_bit_vector_type_expr(&zero, false);
    }

    /// Rejects negative expression widths with the direct-width error text.
    #[test]
    #[should_panic(expected = "bit-vector width must be greater than zero")]
    fn negative_expression_vector_widths_are_rejected() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let negative = file.make_plain_literal(-1, &LiteralFormat::Default);

        file.make_bit_vector_type_expr(&negative, true);
    }

    /// Keeps explicit one-bit expression vectors distinct from scalar types.
    #[test]
    fn one_bit_expression_vector_widths_preserve_their_explicit_range() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("one_bit_vector");
        let one = file.make_plain_literal(1, &LiteralFormat::Default);
        let vector = file.make_bit_vector_type_expr(&one, false);
        file.add_wire(module, "value", &vector);

        assert_eq!(
            file.emit(),
            "module one_bit_vector;\n  wire [0:0] value;\nendmodule\n"
        );
    }

    /// Chooses integer, int, user, and logic parameter kinds from their type.
    #[test]
    fn typed_parameter_ports_preserve_their_complete_base_types() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("typed_parameters");
        let scalar = file.make_scalar_type();
        let signed_vector = file.make_bit_vector_type(8, true);
        let signed_integer = file.make_integer_type(true);
        let unsigned_integer = file.make_integer_type(false);
        let signed_int = file.make_int_type(true);
        let unsigned_int = file.make_int_type(false);
        let external = file.make_extern_type("payload_t");
        let packaged = file.make_extern_package_type("bus_pkg", "word_t");
        let packed_external = file.make_packed_array_type(packaged, &[2]);
        let value = file.make_plain_literal(7, &LiteralFormat::Default);

        for (name, data_type) in [
            ("Scalar", scalar),
            ("Vector", signed_vector),
            ("Integer", signed_integer),
            ("UnsignedInteger", unsigned_integer),
            ("Int", signed_int),
            ("UnsignedInt", unsigned_int),
            ("External", external),
            ("PackedExternal", packed_external),
        ] {
            file.add_typed_parameter_port(module, name, &data_type, &value);
        }

        assert_eq!(
            file.emit(),
            r#"module typed_parameters #(
  parameter logic Scalar = 7,
  parameter logic signed [7:0] Vector = 7,
  parameter integer Integer = 7,
  parameter integer unsigned UnsignedInteger = 7,
  parameter int Int = 7,
  parameter int unsigned UnsignedInt = 7,
  parameter payload_t External = 7,
  parameter bus_pkg::word_t [1:0] PackedExternal = 7
);

endmodule
"#
        );
    }

    /// Uses legal signing and width casts for every built-in scalar type.
    #[test]
    fn built_in_type_casts_preserve_width_and_signedness() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("casts");
        let byte = file.make_bit_vector_type(8, false);
        let value = file.add_input(module, "value", &byte).to_expr();
        let scalar = file.make_scalar_type();
        let signed_scalar = file.make_bit_vector_type(1, true);
        let signed_byte = file.make_bit_vector_type(8, true);
        let integer = file.make_integer_type(true);
        let unsigned_integer = file.make_integer_type(false);
        let int = file.make_int_type(true);
        let unsigned_int = file.make_int_type(false);
        let external = file.make_extern_package_type("bus_pkg", "payload_t");

        for (data_type, expected) in [
            (scalar, "logic'(value)"),
            (signed_scalar, "signed'(logic'(value))"),
            (byte, "unsigned'(8'(value))"),
            (signed_byte, "signed'(8'(value))"),
            (integer, "integer'(value)"),
            (unsigned_integer, "unsigned'(integer'(value))"),
            (int, "int'(value)"),
            (unsigned_int, "unsigned'(int'(value))"),
            (external, "bus_pkg::payload_t'(value)"),
        ] {
            let cast = file.make_type_cast(&data_type, &value);
            assert_eq!(file.emit_expression(&cast), expected);
        }

        let default_width = file.make_plain_literal(8, &LiteralFormat::Default);
        let width = file
            .add_parameter_port(module, "WIDTH", &default_width)
            .to_expr();
        let symbolic = file.make_bit_vector_type_expr(&width, true);
        let cast = file.make_type_cast(&symbolic, &value);
        assert_eq!(file.emit_expression(&cast), "signed'(WIDTH'(value))");

        let one = file.make_plain_literal(1, &LiteralFormat::Default);
        let incremented_width = file.make_add(&width, &one);
        let compound = file.make_bit_vector_type_expr(&incremented_width, false);
        let compound_cast = file.make_type_cast(&compound, &value);
        assert_eq!(
            file.emit_expression(&compound_cast),
            "unsigned'((WIDTH + 1)'(value))"
        );
    }

    /// Formats arbitrary-width values without truncating them to machine words.
    #[test]
    fn wide_literals_preserve_width_padding_and_digit_grouping() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let wide_hex = file
            .make_literal(
                "bits[128]:0xFFEEDDCCBBAA99887766554433221100",
                &LiteralFormat::Hex,
            )
            .expect("valid 128-bit hexadecimal literal");
        assert_eq!(
            file.emit_expression(&wide_hex),
            "128'hffee_ddcc_bbaa_9988_7766_5544_3322_1100"
        );

        let plain_hex = file
            .make_literal("bits[80]:0x10000", &LiteralFormat::PlainHex)
            .expect("valid unsized hexadecimal literal");
        assert_eq!(file.emit_expression(&plain_hex), "'h1_0000");

        let binary = file
            .make_literal("bits[9]:0b11", &LiteralFormat::Binary)
            .expect("valid binary literal");
        assert_eq!(file.emit_expression(&binary), "9'b0_0000_0011");

        let decimal = file
            .make_literal("bits[96]:42", &LiteralFormat::UnsignedDecimal)
            .expect("valid wide decimal literal");
        assert_eq!(file.emit_expression(&decimal), "96'd42");
    }

    /// Switches unpacked-array spelling according to the selected language
    /// mode.
    #[test]
    fn unpacked_array_dimensions_follow_the_selected_language() {
        let mut verilog = VastFile::new(VastFileType::Verilog);
        let verilog_module = verilog.add_module("arrays");
        let verilog_element = verilog.make_bit_vector_type(8, false);
        let verilog_array = verilog.make_unpacked_array_type(verilog_element, &[2, 3]);
        verilog.add_wire(verilog_module, "values", &verilog_array);
        assert_eq!(
            verilog.emit(),
            "module arrays;\n  wire [7:0] values[0:1][0:2];\nendmodule\n"
        );

        let mut system_verilog = VastFile::new(VastFileType::SystemVerilog);
        let system_verilog_module = system_verilog.add_module("arrays");
        let system_verilog_element = system_verilog.make_bit_vector_type(8, false);
        let system_verilog_array =
            system_verilog.make_unpacked_array_type(system_verilog_element, &[2, 3]);
        system_verilog.add_wire(system_verilog_module, "values", &system_verilog_array);
        assert_eq!(
            system_verilog.emit(),
            "module arrays;\n  wire [7:0] values[2][3];\nendmodule\n"
        );
    }

    /// Emits external packed types without an incorrect wire/logic keyword.
    #[test]
    fn packed_external_types_keep_their_qualified_type_name() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("external");
        let external = file.make_extern_package_type("pkg", "payload_t");
        let packed = file.make_packed_array_type(external, &[2, 4]);
        file.add_input(module, "payload", &packed);

        assert_eq!(
            file.emit(),
            "module external(\n  input pkg::payload_t [1:0][3:0] payload\n);\n\nendmodule\n"
        );
    }

    /// Places parentheses around expression-based width casts when necessary.
    #[test]
    fn width_casts_parenthesize_composite_width_expressions() {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("cast");
        let scalar = file.make_scalar_type();
        let a = file.add_input(module, "a", &scalar).to_expr();
        let b = file.add_input(module, "b", &scalar).to_expr();
        let width = file.make_add(&a, &b);
        let cast = file.make_width_cast(&width, &a);

        assert_eq!(file.emit_expression(&cast), "(a + b)'(a)");
    }
}
