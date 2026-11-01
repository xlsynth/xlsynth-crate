// SPDX-License-Identifier: Apache-2.0

//! A small, checked DSLX AST whose fragments also specify reference IR.

use crate::compound::{CompoundOp, Declarations, LoopKind, PatternOp, resize_unsigned};
use crate::constexpr::{
    ComputedWidth, ConstExpr, ConstantScope, ParametricDefault, bits_literal, metadata_bits,
};
use crate::tuple_pattern::{BindingContext, TuplePattern};
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue};

/// A concrete bit width or the width parameter of a helper function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    Concrete(usize),
    Parametric,
}

/// The homogeneous bits type of one fragment-composition body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitsType {
    pub signed: bool,
    pub width: Width,
}

impl BitsType {
    /// Formats the type without resolving a helper's width parameter.
    pub fn to_dslx(self) -> String {
        let prefix = if self.signed { "sN" } else { "uN" };
        match self.width {
            Width::Concrete(width) => format!("{prefix}[{width}]"),
            Width::Parametric => format!("{prefix}[N]"),
        }
    }
}

/// An earlier value in a body: parameters occupy slots zero and one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRef(usize);

impl ValueRef {
    pub const X: Self = Self(0);
    pub const Y: Self = Self(1);
}

/// A modular binary operation, with signedness supplied by the body's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
    Mul,
    Div,
    Mod,
}

impl BinaryOp {
    pub(crate) fn to_dslx(self, lhs: &str, rhs: &str, ty: BitsType) -> String {
        let symbol = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::And => "&",
            Self::Or => "|",
            Self::Xor => "^",
            Self::ShiftLeft => "<<",
            Self::ShiftRight => ">>",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
        };
        let rhs = if matches!(self, Self::ShiftLeft | Self::ShiftRight) {
            format!(
                "({rhs} as {})",
                BitsType {
                    signed: false,
                    ..ty
                }
                .to_dslx()
            )
        } else if matches!(self, Self::Div | Self::Mod) {
            // An odd divisor avoids interpreter division-by-zero failures.
            let one = bits_literal(
                BitsType {
                    signed: false,
                    width: Width::Concrete(1),
                },
                &IrBits::all_ones(1),
            );
            format!("({rhs} | ({one} as {}))", ty.to_dslx())
        } else {
            rhs.to_string()
        };
        format!("{lhs} {symbol} {rhs}")
    }

    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        lhs: BValue,
        rhs: BValue,
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        match self {
            Self::Add => builder.add(lhs, rhs),
            Self::Sub => builder.sub(lhs, rhs),
            Self::And => builder.and(lhs, rhs),
            Self::Or => builder.or(lhs, rhs),
            Self::Xor => builder.xor(lhs, rhs),
            Self::ShiftLeft => builder.shll(lhs, rhs),
            Self::ShiftRight if signed => builder.shra(lhs, rhs),
            Self::ShiftRight => builder.shrl(lhs, rhs),
            Self::Mul if signed => builder.smul(lhs, rhs),
            Self::Mul => builder.umul(lhs, rhs),
            Self::Div | Self::Mod => {
                let xlsynth_pir::ir::Type::Bits(width) = builder.get_type(rhs)? else {
                    return Err(BuilderError::InvalidOperation(
                        "divisor must be bits".to_string(),
                    ));
                };
                let one = builder
                    .literal(IrValue::from_bits(&IrBits::from_lsb_fn(*width, |index| {
                        index == 0
                    })))?;
                let rhs = builder.or(rhs, one)?;
                match (self, signed) {
                    (Self::Div, true) => builder.sdiv(lhs, rhs),
                    (Self::Div, false) => builder.udiv(lhs, rhs),
                    (_, true) => builder.smod(lhs, rhs),
                    (_, false) => builder.umod(lhs, rhs),
                }
            }
        }
    }
}

/// A comparison whose result controls two same-typed alternatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Comparison {
    fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }

    fn emit_ir(
        self,
        builder: &mut FnBuilder,
        lhs: BValue,
        rhs: BValue,
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        match (self, signed) {
            (Self::Eq, _) => builder.eq(lhs, rhs),
            (Self::Ne, _) => builder.ne(lhs, rhs),
            (Self::Lt, true) => builder.slt(lhs, rhs),
            (Self::Lt, false) => builder.ult(lhs, rhs),
            (Self::Le, true) => builder.sle(lhs, rhs),
            (Self::Le, false) => builder.ule(lhs, rhs),
            (Self::Gt, true) => builder.sgt(lhs, rhs),
            (Self::Gt, false) => builder.ugt(lhs, rhs),
            (Self::Ge, true) => builder.sge(lhs, rhs),
            (Self::Ge, false) => builder.uge(lhs, rhs),
        }
    }
}

/// The two DSLX bit-slicing syntaxes exercised over a concatenation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceKind {
    Fixed,
    Width,
}

/// A unary operation preserving the body's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
}

/// One AST fragment and its hypothesized IR semantics.
///
/// Every fragment consumes earlier values and returns the body's bits type.
/// Conditions and tuples are internal to a fragment, keeping composition typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fragment {
    Literal(IrBits),
    Unary {
        op: UnaryOp,
        arg: ValueRef,
    },
    Binary {
        op: BinaryOp,
        lhs: ValueRef,
        rhs: ValueRef,
    },
    IfCompare {
        op: Comparison,
        lhs: ValueRef,
        rhs: ValueRef,
        on_true: ValueRef,
        on_false: ValueRef,
    },
    TupleIndex {
        elements: [ValueRef; 2],
        index: usize,
    },
    Call {
        module: usize,
        args: [ValueRef; 2],
        explicit_parametric: bool,
    },
    Constant {
        expression: Box<ConstExpr>,
        scope: ConstantScope,
        assert_value: bool,
    },
    ConcatSlice {
        lhs: ValueRef,
        rhs: ValueRef,
        kind: SliceKind,
    },
    WidenShift {
        arg: ValueRef,
        extra: usize,
        shift: usize,
    },
    Compound {
        op: CompoundOp,
        args: [ValueRef; 3],
    },
    TuplePattern {
        pattern: TuplePattern,
        context: BindingContext,
        args: [ValueRef; 3],
    },
    StringLookup {
        bytes: [IrBits; 3],
        index: ValueRef,
    },
}

impl Fragment {
    /// Lists references for validation before appending a fragment.
    fn operands(&self) -> Vec<ValueRef> {
        match self {
            Self::Literal(_) | Self::Constant { .. } => vec![],
            Self::Unary { arg, .. } | Self::WidenShift { arg, .. } => vec![*arg],
            Self::Binary { lhs, rhs, .. } | Self::ConcatSlice { lhs, rhs, .. } => vec![*lhs, *rhs],
            Self::IfCompare {
                lhs,
                rhs,
                on_true,
                on_false,
                ..
            } => {
                vec![*lhs, *rhs, *on_true, *on_false]
            }
            Self::TupleIndex { elements, .. } => elements.to_vec(),
            Self::Call { args, .. } => args.to_vec(),
            Self::Compound { args, .. } | Self::TuplePattern { args, .. } => args.to_vec(),
            Self::StringLookup { index, .. } => vec![*index],
        }
    }

    /// Formats only the DSLX side of the fragment's contract.
    fn to_dslx(
        &self,
        ty: BitsType,
        names: &[String],
        prefix: &str,
        modules: &[HelperModule],
    ) -> Result<String, String> {
        let name = |value: ValueRef| names[value.0].clone();
        let width = || -> Result<String, String> {
            Ok(match ty.width {
                Width::Concrete(width) => bits_literal(
                    BitsType {
                        signed: false,
                        width: Width::Concrete(32),
                    },
                    &metadata_bits(width)?,
                ),
                Width::Parametric => "N".to_string(),
            })
        };
        Ok(match self {
            Self::Literal(bits) => {
                // Hex spells the bit pattern even when the sign bit is set.
                bits_literal(ty, bits)
            }
            Self::Unary { op, arg } => {
                let symbol = match op {
                    UnaryOp::Not => "!",
                    UnaryOp::Neg => "-",
                };
                format!("{symbol}{}", name(*arg))
            }
            Self::Binary { op, lhs, rhs } => op.to_dslx(&name(*lhs), &name(*rhs), ty),
            Self::IfCompare {
                op,
                lhs,
                rhs,
                on_true,
                on_false,
            } => format!(
                "if {} {} {} {{ {} }} else {{ {} }}",
                name(*lhs),
                op.symbol(),
                name(*rhs),
                name(*on_true),
                name(*on_false)
            ),
            Self::TupleIndex { elements, index } => {
                format!("({}, {}).{index}", name(elements[0]), name(elements[1]))
            }
            Self::Call {
                module,
                args,
                explicit_parametric,
            } => {
                let suffix = if ty.signed { "s" } else { "u" };
                let parametric = if *explicit_parametric {
                    format!("<{}>", width()?)
                } else {
                    String::new()
                };
                format!(
                    "m{module}::apply_{suffix}{parametric}({}, {})",
                    name(args[0]),
                    name(args[1])
                )
            }
            Self::Constant { expression, .. } => expression.to_dslx(prefix, modules)?,
            Self::ConcatSlice { lhs, rhs, kind } => {
                let unsigned = BitsType {
                    signed: false,
                    ..ty
                }
                .to_dslx();
                let concat = format!(
                    "(({} as {unsigned}) ++ ({} as {unsigned}))",
                    name(*lhs),
                    name(*rhs)
                );
                let width = width()?;
                let start = format!("({width} / u32:2)");
                let slice = match kind {
                    SliceKind::Fixed => {
                        format!("{concat}[({start} as s32):(({start} + {width}) as s32)]")
                    }
                    SliceKind::Width => format!("{concat}[{start}+:{unsigned}]"),
                };
                format!("({slice} as {})", ty.to_dslx())
            }
            Self::WidenShift { arg, extra, shift } => {
                let unsigned32 = BitsType {
                    signed: false,
                    width: Width::Concrete(32),
                };
                let extra = bits_literal(unsigned32, &metadata_bits(*extra)?);
                let shift = bits_literal(unsigned32, &metadata_bits(*shift)?);
                let prefix = if ty.signed { "sN" } else { "uN" };
                // DSLX permits overshifts through names, but rejects oversized
                // numeric literals directly in the shift expression.
                format!(
                    "{{ let shift_amount = {shift}; ((({} as {prefix}[{} + {extra}]) >> shift_amount) as {}) }}",
                    name(*arg),
                    width()?,
                    ty.to_dslx()
                )
            }
            Self::Compound { op, args } => op.to_dslx(
                ty,
                [&names[args[0].0], &names[args[1].0], &names[args[2].0]],
            ),
            Self::TuplePattern {
                pattern,
                context,
                args,
            } => pattern.to_dslx(
                *context,
                ty,
                [&names[args[0].0], &names[args[1].0], &names[args[2].0]],
            ),
            Self::StringLookup { bytes, index } => {
                let encoded: String = bytes
                    .iter()
                    .map(|bits| format!("\\x{:02x}", bits.to_bytes()[0]))
                    .collect();
                let unsigned = BitsType {
                    signed: false,
                    ..ty
                }
                .to_dslx();
                format!(
                    "(\"{encoded}\"[(({} as {unsigned}) as u32) % u32:3] as {})",
                    name(*index),
                    ty.to_dslx()
                )
            }
        })
    }

    /// Instantiates the reference semantics without consulting the DSLX
    /// compiler.
    fn emit_ir(
        &self,
        builder: &mut FnBuilder,
        values: &[BValue],
        signed: bool,
        width: usize,
        modules: &[HelperModule],
    ) -> Result<BValue, BuilderError> {
        let value = |reference: ValueRef| values[reference.0];
        match self {
            Self::Literal(bits) => builder.literal(IrValue::from_bits(bits)),
            Self::Unary { op, arg } => match op {
                UnaryOp::Not => builder.not(value(*arg)),
                UnaryOp::Neg => builder.neg(value(*arg)),
            },
            Self::Binary { op, lhs, rhs } => {
                let lhs = value(*lhs);
                let rhs = value(*rhs);
                op.emit_ir(builder, lhs, rhs, signed)
            }
            Self::IfCompare {
                op,
                lhs,
                rhs,
                on_true,
                on_false,
            } => {
                let condition = op.emit_ir(builder, value(*lhs), value(*rhs), signed)?;
                builder.select(condition, &[value(*on_false), value(*on_true)], None)
            }
            Self::TupleIndex { elements, index } => {
                let tuple = builder.tuple(&[value(elements[0]), value(elements[1])])?;
                builder.tuple_index(tuple, *index)
            }
            Self::Call { module, args, .. } => {
                // Inline the semantic body with a fresh lexical environment.
                // This also tests DSLX module/name resolution and
                // specialization.
                modules[*module].emit_ir(
                    builder,
                    [value(args[0]), value(args[1])],
                    signed,
                    width,
                    modules,
                )
            }
            Self::Constant { expression, .. } => expression
                .emit_ir(builder, modules, None)
                .map_err(BuilderError::InvalidOperation),
            Self::ConcatSlice { lhs, rhs, .. } => {
                let concatenated = builder.concat(&[value(*lhs), value(*rhs)])?;
                builder.bit_slice(concatenated, width / 2, width)
            }
            Self::WidenShift { arg, extra, shift } => {
                let extended_width = width
                    .checked_add(*extra)
                    .ok_or(BuilderError::WidthOverflow)?;
                let extended = if signed {
                    builder.sign_extend(value(*arg), extended_width)?
                } else {
                    builder.zero_extend(value(*arg), extended_width)?
                };
                let amount = metadata_bits(*shift).map_err(BuilderError::InvalidOperation)?;
                let amount = builder.literal(IrValue::from_bits(&amount))?;
                let shifted = if signed {
                    builder.shra(extended, amount)?
                } else {
                    builder.shrl(extended, amount)?
                };
                builder.bit_slice(shifted, 0, width)
            }
            Self::Compound { op, args } => op.emit_ir(
                builder,
                [value(args[0]), value(args[1]), value(args[2])],
                width,
                signed,
            ),
            Self::TuplePattern { pattern, args, .. } => pattern.emit_ir(
                builder,
                [value(args[0]), value(args[1]), value(args[2])],
                width,
            ),
            Self::StringLookup { bytes, index } => {
                let mut literals = Vec::new();
                for bits in bytes {
                    literals.push(builder.literal(IrValue::from_bits(bits))?);
                }
                let array = builder.array(xlsynth_pir::ir::Type::Bits(8), &literals)?;
                let index = resize_unsigned(builder, value(*index), 32)?;
                let three = builder.literal(IrValue::from_bits(
                    &metadata_bits(3).map_err(BuilderError::InvalidOperation)?,
                ))?;
                let index = builder.umod(index, three)?;
                let element = builder.array_index(array, index)?;
                resize_unsigned(builder, element, width)
            }
        }
    }
}

/// A checked sequence of let-bound fragments with two bits parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Body {
    ty: BitsType,
    fragments: Vec<Fragment>,
    result: ValueRef,
    computed_width: Option<ComputedWidth>,
}

impl Body {
    /// Starts a body whose default result is its first parameter.
    pub fn new(ty: BitsType) -> Result<Self, String> {
        if ty.width == Width::Concrete(0) {
            return Err("DSLX samples require a positive bit width".to_string());
        }
        Ok(Self {
            ty,
            fragments: vec![],
            result: ValueRef::X,
            computed_width: None,
        })
    }

    pub fn bits_type(&self) -> BitsType {
        self.ty
    }

    pub fn fragments(&self) -> &[Fragment] {
        &self.fragments
    }

    pub fn result(&self) -> ValueRef {
        self.result
    }

    pub fn computed_width(&self) -> Option<&ComputedWidth> {
        self.computed_width.as_ref()
    }

    /// Uses a concrete constant expression as the source-level type dimension.
    pub fn set_computed_width(&mut self, computation: ComputedWidth) -> Result<(), String> {
        if self.ty.width == Width::Parametric
            || computation.expression.bits_type()
                != (BitsType {
                    signed: false,
                    width: Width::Concrete(32),
                })
            || computation.expression.requires_bit_width()
        {
            return Err("computed widths require a closed unsigned 32-bit expression".to_string());
        }
        self.computed_width = Some(computation);
        Ok(())
    }

    /// Appends a well-typed fragment and makes it the current result.
    pub fn push(&mut self, fragment: Fragment) -> Result<ValueRef, String> {
        if fragment
            .operands()
            .iter()
            .any(|operand| operand.0 >= self.fragments.len() + 2)
        {
            return Err("fragment references a value that has not been defined".to_string());
        }
        if let Fragment::Literal(bits) = &fragment
            && self.ty.width != Width::Concrete(bits.get_bit_count())
        {
            return Err("literal width must match a concrete body type".to_string());
        }
        if let Fragment::TupleIndex { index, .. } = &fragment
            && *index >= 2
        {
            return Err("tuple fragment index must be zero or one".to_string());
        }
        if let Fragment::Constant { expression, .. } = &fragment
            && (expression.bits_type() != self.ty || expression.requires_bit_width())
        {
            return Err("constant binding must be closed and match its body type".to_string());
        }
        if let Fragment::WidenShift { extra, shift, .. } = &fragment {
            if *extra == 0 {
                return Err("widening requires a positive extension".to_string());
            }
            metadata_bits(*extra)?;
            metadata_bits(*shift)?;
        }
        if let Fragment::Compound { op, .. } = &fragment {
            op.validate()?;
        }
        if let Fragment::StringLookup { bytes, .. } = &fragment
            && bytes.iter().any(|bits| bits.get_bit_count() != 8)
        {
            return Err("string bytes must be 8-bit IrBits values".to_string());
        }
        self.result = ValueRef(self.fragments.len() + 2);
        self.fragments.push(fragment);
        Ok(self.result)
    }

    /// Chooses any earlier value as the body's return expression.
    pub fn set_result(&mut self, result: ValueRef) -> Result<(), String> {
        if result.0 >= self.fragments.len() + 2 {
            return Err("return value has not been defined".to_string());
        }
        self.result = result;
        Ok(())
    }

    /// Checks module references against the acyclic translation environment.
    pub(crate) fn validate_calls(&self, available_modules: usize) -> Result<(), String> {
        if let Some(width) = &self.computed_width {
            width.expression.validate_calls(available_modules)?;
        }
        for fragment in &self.fragments {
            if let Fragment::Call { module, .. } = fragment
                && *module >= available_modules
            {
                return Err("call must reference an earlier helper module".to_string());
            }
            if let Fragment::Constant { expression, .. } = fragment {
                expression.validate_calls(available_modules)?;
            }
        }
        Ok(())
    }

    /// Collects nominal types needed by expressions and nested constants.
    pub(crate) fn declarations(&self) -> Declarations {
        let mut declarations = Declarations::default();
        if let Some(width) = &self.computed_width {
            declarations.extend(width.expression.declarations());
        }
        for fragment in &self.fragments {
            match fragment {
                Fragment::Compound { op, .. } => declarations.extend(op.declarations()),
                Fragment::Constant { expression, .. } => {
                    declarations.extend(expression.declarations())
                }
                _ => { /* Other fragment kinds do not declare nominal types. */ }
            }
        }
        declarations
    }

    /// Loops and pattern-bound names require a constexpr invocation boundary.
    pub(crate) fn needs_const_call(&self) -> bool {
        self.fragments.iter().any(|fragment| {
            matches!(
                fragment,
                Fragment::Compound {
                    op: CompoundOp::Loop {
                        kind: LoopKind::Range | LoopKind::Array,
                        ..
                    },
                    ..
                } | Fragment::Compound {
                    op: CompoundOp::Pattern(PatternOp::Tuple),
                    ..
                } | Fragment::TuplePattern { .. }
            )
        })
    }

    /// Emits concrete functions needed by nested constexpr expressions.
    pub(crate) fn render_const_functions(
        &self,
        prefix: &str,
        modules: &[HelperModule],
    ) -> Result<String, String> {
        let mut text = String::new();
        if let Some(width) = &self.computed_width {
            text.push_str(
                &width
                    .expression
                    .render_definition(&format!("{prefix}width_"), modules)?,
            );
        }
        for (index, fragment) in self.fragments.iter().enumerate() {
            if let Fragment::Constant { expression, .. } = fragment {
                text.push_str(
                    &expression.render_definition(&format!("{prefix}c{index}_"), modules)?,
                );
            }
        }
        Ok(text)
    }

    /// Formats local bindings and leaves the result expression to the caller.
    pub(crate) fn render_bindings(
        &self,
        prefix: &str,
        modules: &[HelperModule],
    ) -> Result<String, String> {
        let mut text = String::new();
        let names: Vec<_> = (0..self.fragments.len() + 2)
            .map(|index| self.value_name(ValueRef(index), prefix))
            .collect();
        for (index, fragment) in self.fragments.iter().enumerate() {
            if matches!(
                fragment,
                Fragment::Constant {
                    scope: ConstantScope::Module,
                    ..
                }
            ) {
                // Module declarations are emitted before function definitions.
                continue;
            }
            let name = &names[index + 2];
            let keyword = if matches!(fragment, Fragment::Constant { .. }) {
                "const"
            } else {
                "let"
            };
            text.push_str(&format!(
                "    {keyword} {name} = {};\n",
                fragment.to_dslx(self.ty, &names, &format!("{prefix}c{index}_"), modules)?
            ));
            if let Fragment::Constant {
                expression,
                assert_value: true,
                ..
            } = fragment
            {
                let expected = expression.evaluate(modules, None)?;
                text.push_str(&format!(
                    "    const_assert!({name} == {});\n",
                    bits_literal(self.ty, &expected)
                ));
            }
        }
        Ok(text)
    }

    fn value_name(&self, reference: ValueRef, prefix: &str) -> String {
        let name = format!("{prefix}v{}", reference.0);
        if reference.0 >= 2 && matches!(self.fragments[reference.0 - 2], Fragment::Constant { .. })
        {
            name.to_ascii_uppercase()
        } else {
            name
        }
    }

    pub(crate) fn result_name(&self, prefix: &str) -> String {
        self.value_name(self.result, prefix)
    }

    pub(crate) fn render_type(&self, prefix: &str) -> String {
        if self.computed_width.is_some() {
            format!(
                "{}N[{}WIDTH]",
                if self.ty.signed { "s" } else { "u" },
                prefix.to_ascii_uppercase()
            )
        } else {
            self.ty.to_dslx()
        }
    }

    /// Emits module constants and type dimensions before the containing
    /// function.
    pub(crate) fn render_module_items(
        &self,
        prefix: &str,
        modules: &[HelperModule],
    ) -> Result<String, String> {
        let mut text = String::new();
        if let Some(computation) = &self.computed_width {
            let name = format!("{}WIDTH", prefix.to_ascii_uppercase());
            text.push_str(&format!(
                "\nconst {name} = {};\n",
                computation
                    .expression
                    .to_dslx(&format!("{prefix}width_"), modules)?
            ));
            if computation.assert_value {
                let Width::Concrete(width) = self.ty.width else {
                    return Err("computed width has no concrete type".to_string());
                };
                text.push_str(&format!(
                    "const_assert!({name} == {});\n",
                    bits_literal(
                        BitsType {
                            signed: false,
                            width: Width::Concrete(32)
                        },
                        &metadata_bits(width)?
                    )
                ));
            }
        }
        let names: Vec<_> = (0..self.fragments.len() + 2)
            .map(|index| self.value_name(ValueRef(index), prefix))
            .collect();
        for (index, fragment) in self.fragments.iter().enumerate() {
            if let Fragment::Constant {
                expression,
                scope: ConstantScope::Module,
                assert_value,
            } = fragment
            {
                let name = &names[index + 2];
                text.push_str(&format!(
                    "\nconst {name} = {};\n",
                    expression.to_dslx(&format!("{prefix}c{index}_"), modules)?
                ));
                if *assert_value {
                    text.push_str(&format!(
                        "const_assert!({name} == {});\n",
                        bits_literal(self.ty, &expression.evaluate(modules, None)?)
                    ));
                }
            }
        }
        Ok(text)
    }

    /// Builds reference nodes bottom up for a particular width instantiation.
    pub(crate) fn emit_ir(
        &self,
        builder: &mut FnBuilder,
        args: [BValue; 2],
        width: usize,
        modules: &[HelperModule],
    ) -> Result<BValue, BuilderError> {
        let mut values = args.to_vec();
        for fragment in &self.fragments {
            values.push(fragment.emit_ir(builder, &values, self.ty.signed, width, modules)?);
        }
        Ok(values[self.result.0])
    }
}

/// A module containing signed and unsigned width-parametric helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelperModule {
    unsigned: Body,
    signed: Body,
    default: Option<ParametricDefault>,
}

impl HelperModule {
    pub(crate) fn declarations(&self) -> Declarations {
        let mut declarations = self.unsigned.declarations();
        declarations.extend(self.signed.declarations());
        if let Some(default) = &self.default {
            declarations.extend(default.expression.declarations());
        }
        declarations
    }
    /// Pairs helpers with the signatures required by module-call fragments.
    pub fn new(unsigned: Body, signed: Body) -> Result<Self, String> {
        if unsigned.ty
            != (BitsType {
                signed: false,
                width: Width::Parametric,
            })
            || signed.ty
                != (BitsType {
                    signed: true,
                    width: Width::Parametric,
                })
        {
            return Err("helpers must have unsigned/signed parametric signatures".to_string());
        }
        Ok(Self {
            unsigned,
            signed,
            default: None,
        })
    }

    pub fn body(&self, signed: bool) -> &Body {
        if signed { &self.signed } else { &self.unsigned }
    }

    pub fn parametric_default(&self) -> Option<&ParametricDefault> {
        self.default.as_ref()
    }

    /// Adds a u32 default parameter, whose cast participates in the result.
    pub fn with_default(mut self, default: ParametricDefault) -> Result<Self, String> {
        if default.expression.bits_type()
            != (BitsType {
                signed: false,
                width: Width::Concrete(32),
            })
        {
            return Err("parametric defaults must have unsigned 32-bit type".to_string());
        }
        self.default = Some(default);
        Ok(self)
    }

    pub(crate) fn validate_calls(&self, available: usize) -> Result<(), String> {
        self.unsigned.validate_calls(available)?;
        self.signed.validate_calls(available)?;
        if let Some(default) = &self.default {
            default.expression.validate_calls(available)?;
        }
        Ok(())
    }

    /// Inlines the body and independently instantiates any omitted default.
    pub(crate) fn emit_ir(
        &self,
        builder: &mut FnBuilder,
        args: [BValue; 2],
        signed: bool,
        width: usize,
        modules: &[HelperModule],
    ) -> Result<BValue, BuilderError> {
        let result = self.body(signed).emit_ir(builder, args, width, modules)?;
        if let Some(default) = &self.default {
            let value = default
                .expression
                .emit_ir(builder, modules, Some(width))
                .map_err(BuilderError::InvalidOperation)?;
            let value = if width >= 32 {
                builder.zero_extend(value, width)?
            } else {
                builder.bit_slice(value, 0, width)?
            };
            default.combine.emit_ir(builder, result, value, signed)
        } else {
            Ok(result)
        }
    }
}
