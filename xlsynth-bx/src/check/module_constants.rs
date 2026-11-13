// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::{IrValue, ir_value_utils::ir_bits_to_usize};

use crate::ast::{Block, BlockStages, ConstDef, Expr, Type};
use crate::dslx_import::ImportResolver;
use crate::infer_literal::has_implicit_constant_type;
use crate::typed::bits;

use super::{Bindings, BodyChecker, CheckedConstant, Declarations, compatible};

/// Module scope visible where a declaration writes its type annotations.
#[derive(Clone, Copy)]
pub(super) struct ModuleDimensions<'a, 'm> {
    pub(super) decls: &'a Declarations<'m>,
    pub(super) position: Option<usize>,
    pub(super) warning_owner: Option<&'m str>,
    zero_binding: ZeroBinding<'a>,
}

/// The declaration that owns the lexical `zero!` binding, or a binding
/// already resolved by the enclosing body checker.
#[derive(Clone, Copy)]
enum ZeroBinding<'a> {
    Owner(&'a str),
    Resolved(&'a str),
}

impl<'m> Declarations<'m> {
    pub(super) fn module_dimensions<'a>(&'a self, owner: &'a str) -> ModuleDimensions<'a, 'm> {
        ModuleDimensions {
            decls: self,
            position: self.positions.get(owner).copied(),
            warning_owner: self.positions.get_key_value(owner).map(|(&name, _)| name),
            zero_binding: ZeroBinding::Owner(owner),
        }
    }

    pub(super) fn dimensions_at<'a>(
        &'a self,
        position: Option<usize>,
        zero_binding: &'a str,
        warning_owner: Option<&'m str>,
    ) -> ModuleDimensions<'a, 'm> {
        ModuleDimensions {
            decls: self,
            position,
            warning_owner,
            zero_binding: ZeroBinding::Resolved(zero_binding),
        }
    }
}

impl ModuleDimensions<'_, '_> {
    /// Module constants already hold evaluated values. Stage inference needs
    /// only the unsigned stage number that the name denotes.
    pub(super) fn block_stages(&self, block: &Block) -> Result<BlockStages<u64>, String> {
        block.map_stages(|name| {
            if !self.decls.visible(name, self.position) {
                return Err(format!(
                    "stage constant '{name}' is declared later in the module"
                ));
            }
            let constant =
                self.decls.constants.get(name).ok_or_else(|| {
                    format!("unknown module constant '{name}' in stage annotation")
                })?;
            if !matches!(bits(&constant.ty), Some((false, _))) {
                return Err(format!(
                    "stage constant '{name}' must be an unsigned integer, got {}",
                    constant.ty
                ));
            }
            constant
                .value
                .to_u64()
                .map_err(|_| format!("stage constant '{name}' does not fit in u64"))
        })
    }

    pub(super) fn zero_binding(&self) -> String {
        match self.zero_binding {
            ZeroBinding::Owner(owner) => self.decls.zero_binding(owner),
            ZeroBinding::Resolved(binding) => binding.to_owned(),
        }
    }

    /// Resolve the constant in its declaring scope and preserve its source
    /// type.
    fn constant(
        &self,
        expr: &Expr,
        expected: &Type,
        is_local: &dyn Fn(&str) -> bool,
        bindings: Option<&Bindings>,
        resolver: &mut ImportResolver,
    ) -> Result<IrValue, String> {
        let Expr::Name(name) = expr else {
            return Err("a concrete type dimension requires a literal or a module constant".into());
        };
        let head = name
            .split_once("::")
            .map_or(name.as_str(), |(head, _)| head);
        if is_local(head) {
            return Err(format!(
                "'{head}' names a local value in this type dimension"
            ));
        }
        if let Some(bindings) = bindings {
            bindings.reject_value(name)?;
        }
        if !self.decls.visible(name, self.position) {
            return Err(format!(
                "module constant '{name}' is declared after this type dimension"
            ));
        }
        let imported;
        let constant = if let Some(constant) = self.decls.constants.get(name.as_str()) {
            constant
        } else {
            imported = resolver
                .resolve_constant(name)?
                .ok_or_else(|| format!("unknown module constant '{name}' in type dimension"))?;
            &imported
        };
        if !compatible(&constant.ty, expected) {
            return Err(format!(
                "type dimension '{name}' expects {expected}, got {}",
                constant.ty
            ));
        }
        Ok(constant.value.clone())
    }

    pub(super) fn size(
        &self,
        expr: &Expr,
        is_local: &dyn Fn(&str) -> bool,
        bindings: Option<&Bindings>,
        resolver: &mut ImportResolver,
    ) -> Result<usize, String> {
        let expected = Type::unsigned(32);
        if let Expr::Literal { ty, text } = expr
            && ty.as_ref().is_none_or(|ty| compatible(ty, &expected))
        {
            let (digits, radix) = crate::parser::radix_digits(text);
            return u32::from_str_radix(digits, radix)
                .map(|width| width as usize)
                .map_err(|_| format!("invalid u32 type dimension '{text}'"));
        }
        let value = self.constant(expr, &expected, is_local, bindings, resolver)?;
        ir_bits_to_usize(value.as_bits().map_err(|e| e.to_string())?)
            .ok_or_else(|| "type dimension does not fit usize".into())
    }

    pub(super) fn signed(
        &self,
        expr: &Expr,
        is_local: &dyn Fn(&str) -> bool,
        bindings: Option<&Bindings>,
        resolver: &mut ImportResolver,
    ) -> Result<bool, String> {
        if let Expr::Bool(value) = expr {
            return Ok(*value);
        }
        self.constant(expr, &Type::Bool, is_local, bindings, resolver)?
            .to_bool()
            .map_err(|e| e.to_string())
    }
}

impl<'a, 'm> BodyChecker<'a, 'm> {
    /// Resolve a pure initializer in declaration order and retain its concrete
    /// type and value for subsequent uses.
    pub(super) fn check_constant(&mut self, decl: &'m ConstDef) -> Result<CheckedConstant, String> {
        self.warning_owner = Some(&decl.name);
        self.zero_binding = self.decls.zero_binding(&decl.name);
        self.function_position = self.decls.positions.get(decl.name.as_str()).copied();
        self.enable_constant_functions();
        let annotation = decl
            .ty
            .as_ref()
            .map(|ty| self.source_type(ty))
            .transpose()?;
        if annotation.is_none() && has_implicit_constant_type(&decl.value)? {
            return Err("constant with an implicit integer type requires a type annotation".into());
        }
        let expr = self.scoped_result(&decl.value, annotation.as_ref(), false)?;
        let value = self
            .checked_constant(&expr)?
            .ok_or("initializer must evaluate to a compile-time constant")?;
        Ok(CheckedConstant {
            name: decl.name.clone(),
            public: decl.public,
            ty: expr.ty,
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::check::TypedExprKind;
    use crate::{check_module, parse_module};
    use std::fmt::Write;
    use xlsynth_pir::IrValue;

    /// Each update materializes the complete array, even when the result
    /// expression selects just one element.
    fn nested_updates() -> String {
        (0..20).fold("A".to_owned(), |array, _| {
            format!("update({array}, u32:0, u1:1)")
        })
    }

    fn check_update_budget(initializer: &str) -> Result<(), String> {
        for size in [8, 65536] {
            let source = format!("const A = zero!<u1[{size}]>(); const C = {initializer};");
            let checked = check_module(&parse_module(&source)?);
            if size == 8 {
                assert_eq!(checked?.constants()[1].value, IrValue::bool(true));
            } else {
                assert_eq!(
                    checked.err().as_deref(),
                    Some("const 'C': constant evaluation exceeds the work limit of 1048576")
                );
            }
        }
        Ok(())
    }

    #[test]
    fn direct_constant_updates_charge_each_intermediate_array() -> Result<(), String> {
        check_update_budget(&format!("{}[u32:0]", nested_updates()))
    }

    #[test]
    fn ordinary_block_caches_a_two_limb_result_of_large_array_updates() -> Result<(), String> {
        let source = format!(
            r#"
block b(clk) -> uN[65] {{
    let A = zero!<u1[65536]>();
    let k = {}[u32:0] as uN[65];
    reg q: uN[65] = k;
    q := uN[65]:0;
    q
}}
"#,
            nested_updates()
        );
        let checked = check_module(&parse_module(&source)?)?;
        let expected = IrValue::make_ubits(65, 1).map_err(|e| e.to_string())?;
        assert!(checked.blocks()[0].statements.iter().any(|stmt|
            matches!(stmt, crate::check::CheckedStmt::Reg { reset: Some(value), .. } if value == &expected)
        ));
        Ok(())
    }

    #[test]
    fn scoped_constant_updates_charge_speculative_and_required_work() -> Result<(), String> {
        let mut initializer = String::from("{ let a0 = A;");
        for i in 1..=20 {
            writeln!(
                initializer,
                "let a{i} = update(a{prior}, u32:0, u1:1);",
                prior = i - 1
            )
            .unwrap();
        }
        initializer.push_str("a20[u32:0] }");
        check_update_budget(&initializer)?;

        // Exhausting speculation on an unused let leaves the demanded let
        // its own allowance to compute the selected element.
        let source = format!(
            r#"
const A = zero!<u1[65536]>();
const C = {{
    let unused = {};
    let selected = update(A, u32:0, u1:1);
    selected[u32:0]
}};
"#,
            nested_updates()
        );
        let checked = check_module(&parse_module(&source)?)?;
        assert_eq!(checked.constants()[1].value, IrValue::bool(true));
        Ok(())
    }

    #[test]
    fn checking_untaken_branches_bounds_owned_wide_constant_reads() -> Result<(), String> {
        let comparisons = std::iter::repeat_n("v == v", 40)
            .collect::<Vec<_>>()
            .join(", ");
        let body = format!("if true {{ true }} else {{ [{comparisons}][u32:0] }}");
        for definition in [
            format!("const C = {};", body.replace('v', "BIG")),
            format!("const C = {{ let v = BIG; {body} }};"),
            format!("fn f() -> bool {{ let v = BIG; {body} }}"),
            format!("block b(clk) -> bool {{ {} }}", body.replace('v', "BIG")),
        ] {
            for width in [64, 1048576] {
                let source = format!("const BIG = zero!<uN[{width}]>(); {definition}");
                let checked = check_module(&parse_module(&source)?);
                if width == 64 {
                    checked?;
                } else {
                    let error = checked
                        .err()
                        .ok_or("wide constant reads exceeded the budget")?;
                    assert!(error.contains("expansion limit"), "{error}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn repeated_reads_bound_owned_tuple_type_trees_in_constants_and_block_lets()
    -> Result<(), String> {
        let mut declarations = String::from("const V0 = u1:0;");
        for i in 1..=14 {
            writeln!(declarations, "const V{i} = (V{n}, V{n});", n = i - 1).unwrap();
        }
        for (leaf, accepted) in [("V4", true), ("V14", false)] {
            let comparisons = std::iter::repeat_n("v == v", 17)
                .collect::<Vec<_>>()
                .join(", ");
            let body = format!("if true {{ true }} else {{ [{comparisons}][u32:0] }}");
            for source in [
                format!("{declarations} const C = {};", body.replace('v', leaf)),
                format!("{declarations} fn f()->bool {{ let v = {leaf}; {body} }}"),
                format!("{declarations} block b(clk)->bool {{ let v = {leaf}; {body} }}"),
            ] {
                let checked = check_module(&parse_module(&source)?);
                if accepted {
                    checked?;
                } else {
                    let error = checked.err().ok_or("unbounded source tuple type copies")?;
                    assert!(error.contains("expansion limit"), "{error}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn shared_tuple_reads_charge_only_the_wide_members_copied_by_patterns() -> Result<(), String> {
        let prefix = "const PAIR = (zero!<uN[1048575]>(),);";
        let comparisons = std::iter::repeat_n("PAIR == PAIR", 80)
            .collect::<Vec<_>>()
            .join(", ");
        let shared =
            format!("{prefix} const C = if true {{ true }} else {{ [{comparisons}][u32:0] }};");
        let checked = check_module(&parse_module(&shared)?)?;
        assert_eq!(checked.constants()[1].value, IrValue::bool(true));

        let bindings = "let (v,) = PAIR;".repeat(70);
        let copies = format!("{prefix} fn f() -> bool {{ {bindings} true }}");
        let error = check_module(&parse_module(&copies)?)
            .err()
            .ok_or("unbounded tuple pattern copies")?;
        assert!(error.contains("expansion limit"), "{error}");
        Ok(())
    }

    #[test]
    fn logical_constant_initializers_need_a_concrete_result_constraint() -> Result<(), String> {
        for expr in ["0 || 1", "!(0 && 1)", "if true { 0 || 1 } else { 1 && 0 }"] {
            let error = check_module(&parse_module(&format!("const C = {expr};"))?).unwrap_err();
            assert!(
                error.contains("requires a type annotation"),
                "{expr}: {error}"
            );
        }
        for expr in [
            "true || 1",
            "0 || (1 < 2)",
            "(0 || 1) == true",
            "(0 || 1) as bool",
        ] {
            let module = check_module(&parse_module(&format!("const C = {expr};"))?)?;
            assert_eq!(module.constants()[0].value, IrValue::bool(true), "{expr}");
        }
        Ok(())
    }

    #[test]
    fn repeated_calls_share_results_across_a_deep_helper_chain() -> Result<(), String> {
        let mut source = String::from("fn f0(x:u64)->u64{x}\n");
        for i in 1..64 {
            writeln!(
                source,
                "fn f{i}(x:u64)->u64 {{ f{prior}(x) + f{prior}(x) }}",
                prior = i - 1
            )
            .unwrap();
        }
        source.push_str("const C=f63(u64:1); fn result()->u64{C}");
        let checked = check_module(&parse_module(&source)?)?;
        let result = checked
            .functions()
            .iter()
            .find(|f| f.name == "result")
            .unwrap();
        let TypedExprKind::Literal(value) = &result.output.kind else {
            panic!("expected the folded constant");
        };
        assert_eq!(
            value,
            &IrValue::make_ubits(64, 1 << 63).map_err(|e| e.to_string())?
        );
        Ok(())
    }
}
