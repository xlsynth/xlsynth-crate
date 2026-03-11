// SPDX-License-Identifier: Apache-2.0

use crate::ast::Expr as VExpr;
use crate::eval::eval_ast_with_calls;
use crate::parser::parse_expr;
use crate::sv_ast::AlwaysFf;
use crate::sv_ast::CasezArm;
use crate::sv_ast::CasezPattern;
use crate::sv_ast::Decl;
use crate::sv_ast::FunctionAssign;
use crate::sv_ast::FunctionBody;
use crate::sv_ast::FunctionDecl;
use crate::sv_ast::GenerateBranch;
use crate::sv_ast::Lhs;
use crate::sv_ast::ModuleItem;
use crate::sv_ast::ParsedModule;
use crate::sv_ast::PortDecl;
use crate::sv_ast::PortDir;
use crate::sv_ast::PortTy;
use crate::sv_ast::Span;
use crate::sv_ast::Stmt;
use crate::sv_lexer::Tok;
use crate::sv_lexer::TokKind;
use crate::Error;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

pub fn parse_combo_module(src: &str) -> Result<ParsedModule> {
    let toks = crate::sv_lexer::lex_all(src)?;
    let mut p = Parser {
        src,
        toks,
        idx: 0,
        params: BTreeMap::new(),
        defines: BTreeSet::new(),
        ifdef_stack: Vec::new(),
    };
    p.parse_combo_module()
}

#[allow(dead_code)]
pub fn parse_pipeline_module(src: &str) -> Result<ParsedModule> {
    let toks = crate::sv_lexer::lex_all(src)?;
    let mut p = Parser {
        src,
        toks,
        idx: 0,
        params: BTreeMap::new(),
        defines: BTreeSet::new(),
        ifdef_stack: Vec::new(),
    };
    p.parse_pipeline_module()
}

pub fn parse_pipeline_module_with_defines(
    src: &str,
    defines: &BTreeSet<String>,
) -> Result<ParsedModule> {
    let toks = crate::sv_lexer::lex_all(src)?;
    let mut p = Parser {
        src,
        toks,
        idx: 0,
        params: BTreeMap::new(),
        defines: defines.clone(),
        ifdef_stack: Vec::new(),
    };
    p.parse_pipeline_module()
}

struct Parser<'a> {
    src: &'a str,
    toks: Vec<Tok>,
    idx: usize,
    params: BTreeMap<String, Value4>,
    defines: BTreeSet<String>,
    ifdef_stack: Vec<IfdefFrame>,
}

#[derive(Debug, Copy, Clone)]
struct IfdefFrame {
    cond_eval: bool,
    outer_active: bool,
    in_else: bool,
    active: bool,
}

impl<'a> Parser<'a> {
    fn cur(&self) -> &TokKind {
        &self.toks[self.idx].kind
    }

    fn bump(&mut self) {
        self.idx += 1;
    }

    fn all_active(&self) -> bool {
        self.ifdef_stack.iter().all(|f| f.active)
    }

    fn handle_preprocessor_directive(&mut self) -> Result<()> {
        if *self.cur() != TokKind::Other('`') {
            return Err(Error::Parse(
                "expected ` for preprocessor directive".to_string(),
            ));
        }
        self.bump(); // '`'
        let dir = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => {
                return Err(Error::Parse(
                    "expected preprocessor directive name".to_string(),
                ));
            }
        };

        match dir.as_str() {
            "ifdef" | "ifndef" => {
                let sym = match self.toks[self.idx].kind.clone() {
                    TokKind::Ident(s) => {
                        self.bump();
                        s
                    }
                    _ => {
                        return Err(Error::Parse(
                            "expected identifier after `ifdef/`ifndef".to_string(),
                        ));
                    }
                };
                let mut cond_eval = self.defines.contains(&sym);
                if dir == "ifndef" {
                    cond_eval = !cond_eval;
                }
                let outer_active = self.all_active();
                let active = outer_active && cond_eval;
                self.ifdef_stack.push(IfdefFrame {
                    cond_eval,
                    outer_active,
                    in_else: false,
                    active,
                });
                Ok(())
            }
            "else" => {
                if let Some(top) = self.ifdef_stack.last_mut() {
                    if top.in_else {
                        return Err(Error::Parse("duplicate `else".to_string()));
                    }
                    top.in_else = true;
                    top.active = top.outer_active && !top.cond_eval;
                }
                Ok(())
            }
            "endif" => {
                let _ = self.ifdef_stack.pop();
                Ok(())
            }
            // Non-conditional directive: best-effort skip tokens until we see another directive
            // or endmodule/EOF. This is conservative given we don't track newlines.
            _ => {
                while self.idx < self.toks.len() {
                    match self.cur() {
                        TokKind::Other('`') | TokKind::KwEndmodule | TokKind::End => break,
                        _ => self.bump(),
                    }
                }
                Ok(())
            }
        }
    }

    fn expect(&mut self, want: TokKind) -> Result<()> {
        if self.toks[self.idx].kind == want {
            self.idx += 1;
            Ok(())
        } else {
            Err(Error::Parse(format!(
                "expected {:?}, got {:?}",
                want, self.toks[self.idx].kind
            )))
        }
    }

    fn params_env(&self) -> crate::Env {
        let mut env = crate::Env::new();
        for (name, value) in &self.params {
            env.insert(name.clone(), value.clone());
        }
        env
    }

    fn eval_const_expr_value(&self, expr: &VExpr, expected_width: Option<u32>) -> Result<Value4> {
        let env = self.params_env();
        eval_ast_with_calls(expr, &env, None, expected_width)
    }

    fn eval_const_u32(&self, expr: &VExpr) -> Result<u32> {
        let value = self.eval_const_expr_value(expr, None)?;
        value.to_u32_if_known().ok_or_else(|| {
            Error::Parse("decl width constant must be known and fit in u32".to_string())
        })
    }

    fn parse_optional_param_list(&mut self) -> Result<()> {
        if *self.cur() != TokKind::Other('#') {
            return Ok(());
        }
        self.bump();
        self.expect(TokKind::LParen)?;
        loop {
            if *self.cur() == TokKind::RParen {
                self.bump();
                break;
            }
            self.parse_parameter_decl()?;
            match self.cur() {
                TokKind::Comma => {
                    self.bump();
                }
                TokKind::RParen => {}
                _ => {
                    return Err(Error::Parse(
                        "expected `,` or `)` in parameter list".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn parse_parameter_decl(&mut self) -> Result<()> {
        self.expect(TokKind::KwParameter)?;

        let mut saw_logic = false;
        let mut is_signed = false;
        loop {
            match self.cur() {
                TokKind::KwLogic if !saw_logic => {
                    saw_logic = true;
                    self.bump();
                }
                TokKind::KwSigned if !is_signed => {
                    is_signed = true;
                    self.bump();
                }
                _ => break,
            }
        }

        let mut declared_width: Option<u32> = None;
        if *self.cur() == TokKind::LBracket {
            self.bump();
            let msb_expr = self.parse_expr_until(&[TokKind::Colon])?;
            self.expect(TokKind::Colon)?;
            let lsb_expr = self.parse_expr_until(&[TokKind::RBracket])?;
            self.expect(TokKind::RBracket)?;
            let msb = self.eval_const_u32(&msb_expr)?;
            let lsb = self.eval_const_u32(&lsb_expr)?;
            if msb < lsb {
                return Err(Error::Parse("decl range msb<lsb not supported".to_string()));
            }
            declared_width = Some(msb - lsb + 1);
        } else if saw_logic {
            declared_width = Some(1);
        }

        let name = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected parameter identifier".to_string())),
        };
        if *self.cur() != TokKind::Eq {
            return Err(Error::Parse(format!(
                "parameter `{name}` must have a default value via `=`; parameter overrides are not supported yet"
            )));
        }
        self.bump();
        let default_expr = self.parse_expr_until(&[TokKind::Comma, TokKind::RParen])?;

        let mut value = self.eval_const_expr_value(&default_expr, declared_width)?;
        if let Some(width) = declared_width {
            let signedness = if is_signed {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            value = value.with_signedness(signedness).resize(width);
        } else if is_signed {
            value = value.with_signedness(Signedness::Signed);
        }

        if self.params.insert(name.clone(), value).is_some() {
            return Err(Error::Parse(format!("duplicate parameter `{name}`")));
        }
        Ok(())
    }

    fn parse_combo_module(&mut self) -> Result<ParsedModule> {
        self.expect(TokKind::KwModule)?;
        let header_start = self.toks[self.idx - 1].start;
        let name = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected module name".to_string())),
        };
        self.parse_optional_param_list()?;

        let ports = if *self.cur() == TokKind::LParen {
            self.parse_port_list_ports()?
        } else {
            Vec::new()
        };
        self.expect(TokKind::Semi)?;
        let header_end = self.toks[self.idx - 1].end;
        let header_span = Span {
            start: header_start,
            end: header_end,
        };

        let mut items: Vec<ModuleItem> = Vec::new();
        let endmodule_span: Span = loop {
            match self.cur().clone() {
                TokKind::KwEndmodule => {
                    let start = self.toks[self.idx].start;
                    let end = self.toks[self.idx].end;
                    self.bump();
                    break Span { start, end };
                }
                TokKind::End => {
                    return Err(Error::Parse(
                        "unexpected EOF (missing endmodule)".to_string(),
                    ));
                }
                TokKind::Semi => {
                    self.bump();
                }
                _ => items.push(self.parse_combo_item()?),
            }
        };

        Ok(ParsedModule {
            name,
            params: self.params.clone(),
            ports,
            header_span,
            endmodule_span,
            items,
        })
    }

    fn parse_combo_item(&mut self) -> Result<ModuleItem> {
        match self.cur().clone() {
            TokKind::KwWire => {
                let (d, span) = self.parse_wire_decl_with_span()?;
                Ok(ModuleItem::Decl { decl: d, span })
            }
            TokKind::KwLogic | TokKind::KwReg => {
                let (d, span) = self.parse_logic_decl_with_span()?;
                Ok(ModuleItem::Decl { decl: d, span })
            }
            TokKind::KwAssign => self.parse_assign_item(),
            TokKind::KwFunction => {
                let start = self.toks[self.idx].start;
                let (f, body_span, begin_span, end_span) =
                    self.parse_combo_function_with_spans()?;
                let end = self.toks[self.idx - 1].end; // endfunction
                Ok(ModuleItem::Function {
                    func: f,
                    span: Span { start, end },
                    body_span,
                    begin_span,
                    end_span,
                })
            }
            TokKind::KwFor => self.parse_combo_generate_for_item(),
            TokKind::KwIf => self.parse_combo_generate_if_item(),
            other => Err(Error::Parse(format!(
                "unsupported combo module item token: {:?}",
                other
            ))),
        }
    }

    fn parse_combo_generate_for_item(&mut self) -> Result<ModuleItem> {
        let start = self.toks[self.idx].start;
        self.expect(TokKind::KwFor)?;
        self.expect(TokKind::LParen)?;
        match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) if s == "genvar" => self.bump(),
            _ => {
                return Err(Error::Parse(
                    "expected `genvar` in generate for".to_string(),
                ));
            }
        }
        let genvar = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected genvar identifier".to_string())),
        };
        self.expect(TokKind::Eq)?;
        let start_expr = self.parse_expr_until(&[TokKind::Semi])?;
        self.expect(TokKind::Semi)?;
        match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) if s == genvar => self.bump(),
            _ => {
                return Err(Error::Parse(format!(
                    "expected loop variable `{genvar}` in generate for condition"
                )));
            }
        }
        if *self.cur() != TokKind::Other('<') {
            return Err(Error::Parse(
                "generate for condition must use `<`".to_string(),
            ));
        }
        self.bump();
        let limit_expr = self.parse_expr_until(&[TokKind::Semi])?;
        self.expect(TokKind::Semi)?;
        match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) if s == genvar => self.bump(),
            _ => {
                return Err(Error::Parse(format!(
                    "expected loop variable `{genvar}` in generate for step"
                )));
            }
        }
        self.expect(TokKind::Eq)?;
        match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) if s == genvar => self.bump(),
            _ => {
                return Err(Error::Parse(format!(
                    "expected loop variable `{genvar}` on generate for rhs"
                )));
            }
        }
        if *self.cur() != TokKind::Other('+') {
            return Err(Error::Parse("generate for step must use `+ 1`".to_string()));
        }
        self.bump();
        match self.toks[self.idx].kind.clone() {
            TokKind::Number(s) if s == "1" => self.bump(),
            _ => {
                return Err(Error::Parse(
                    "generate for step must increment by 1".to_string(),
                ));
            }
        }
        self.expect(TokKind::RParen)?;
        let body = self.parse_combo_item_block()?;
        let end = self.toks[self.idx - 1].end;
        Ok(ModuleItem::GenerateFor {
            genvar,
            start: start_expr,
            limit: limit_expr,
            body,
            span: Span { start, end },
        })
    }

    fn parse_combo_generate_if_item(&mut self) -> Result<ModuleItem> {
        let start = self.toks[self.idx].start;
        let mut branches = Vec::new();
        loop {
            let branch_start = self.toks[self.idx].start;
            let cond = if *self.cur() == TokKind::KwIf {
                self.expect(TokKind::KwIf)?;
                self.expect(TokKind::LParen)?;
                let cond = self.parse_expr_until(&[TokKind::RParen])?;
                self.expect(TokKind::RParen)?;
                Some(cond)
            } else {
                None
            };
            let body = self.parse_combo_item_block()?;
            let branch_end = self.toks[self.idx - 1].end;
            branches.push(GenerateBranch {
                cond,
                body,
                span: Span {
                    start: branch_start,
                    end: branch_end,
                },
            });

            if *self.cur() != TokKind::KwElse {
                break;
            }
            let else_start = self.toks[self.idx].start;
            self.bump();
            if *self.cur() != TokKind::KwIf {
                let body = self.parse_combo_item_block()?;
                let branch_end = self.toks[self.idx - 1].end;
                branches.push(GenerateBranch {
                    cond: None,
                    body,
                    span: Span {
                        start: else_start,
                        end: branch_end,
                    },
                });
                break;
            }
        }
        let end = self.toks[self.idx - 1].end;
        Ok(ModuleItem::GenerateIf {
            branches,
            span: Span { start, end },
        })
    }

    fn parse_combo_item_block(&mut self) -> Result<Vec<ModuleItem>> {
        self.expect(TokKind::KwBegin)?;
        if *self.cur() == TokKind::Colon {
            self.bump();
            match self.toks[self.idx].kind.clone() {
                TokKind::Ident(_) => self.bump(),
                _ => return Err(Error::Parse("expected generate block label".to_string())),
            }
        }
        let mut items = Vec::new();
        while *self.cur() != TokKind::KwEnd {
            if *self.cur() == TokKind::Semi {
                self.bump();
                continue;
            }
            items.push(self.parse_combo_item()?);
        }
        self.expect(TokKind::KwEnd)?;
        Ok(items)
    }

    fn parse_pipeline_module(&mut self) -> Result<ParsedModule> {
        self.expect(TokKind::KwModule)?;
        let header_start = self.toks[self.idx - 1].start;
        let name = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected module name".to_string())),
        };
        self.parse_optional_param_list()?;

        let ports = if *self.cur() == TokKind::LParen {
            self.parse_port_list_ports()?
        } else {
            Vec::new()
        };
        self.expect(TokKind::Semi)?;
        let header_end = self.toks[self.idx - 1].end;
        let header_span = Span {
            start: header_start,
            end: header_end,
        };

        let mut items: Vec<ModuleItem> = Vec::new();
        let endmodule_span: Span = loop {
            // If we're inside a disabled `ifdef branch, skip tokens until we see a
            // directive (which can update nesting) or endmodule/EOF.
            if !self.all_active() {
                match self.cur() {
                    TokKind::Other('`') => {
                        self.handle_preprocessor_directive()?;
                    }
                    TokKind::KwEndmodule | TokKind::End => {}
                    _ => self.bump(),
                }
                continue;
            }
            match self.cur().clone() {
                TokKind::Other('`') => {
                    self.handle_preprocessor_directive()?;
                }
                TokKind::Semi => {
                    self.bump();
                }
                TokKind::KwEndmodule => {
                    let start = self.toks[self.idx].start;
                    let end = self.toks[self.idx].end;
                    self.bump();
                    break Span { start, end };
                }
                TokKind::End => {
                    return Err(Error::Parse(
                        "unexpected EOF (missing endmodule)".to_string(),
                    ));
                }
                _ => items.push(self.parse_pipeline_item()?),
            }
        };

        Ok(ParsedModule {
            name,
            params: self.params.clone(),
            ports,
            header_span,
            endmodule_span,
            items,
        })
    }

    fn parse_pipeline_item(&mut self) -> Result<ModuleItem> {
        match self.cur().clone() {
            TokKind::KwWire => {
                let (d, span) = self.parse_wire_decl_with_span()?;
                Ok(ModuleItem::Decl { decl: d, span })
            }
            TokKind::KwLogic | TokKind::KwReg => {
                let (d, span) = self.parse_logic_decl_with_span()?;
                Ok(ModuleItem::Decl { decl: d, span })
            }
            TokKind::KwAssign => {
                let stmt_start = self.toks[self.idx].start;
                self.expect(TokKind::KwAssign)?;
                let lhs = self.parse_lhs()?;
                self.expect(TokKind::Eq)?;
                let (rhs, rhs_span) = self.parse_expr_with_span_until_semi()?;
                self.expect(TokKind::Semi)?;
                let stmt_end = self.toks[self.idx - 1].end;
                Ok(ModuleItem::Assign {
                    lhs,
                    rhs,
                    rhs_span,
                    span: Span {
                        start: stmt_start,
                        end: stmt_end,
                    },
                })
            }
            TokKind::KwFunction => {
                let start = self.toks[self.idx].start;
                let (f, body_span, begin_span, end_span) =
                    self.parse_combo_function_with_spans()?;
                let end = self.toks[self.idx - 1].end; // endfunction
                Ok(ModuleItem::Function {
                    func: f,
                    span: Span { start, end },
                    body_span,
                    begin_span,
                    end_span,
                })
            }
            TokKind::KwAlwaysFf => {
                let stmt_start = self.toks[self.idx].start;
                let af = self.parse_always_ff()?;
                let stmt_end = self.toks[self.idx - 1].end;
                Ok(ModuleItem::AlwaysFf {
                    always_ff: af,
                    span: Span {
                        start: stmt_start,
                        end: stmt_end,
                    },
                })
            }
            TokKind::KwFor => self.parse_generate_for_item(),
            TokKind::KwIf => self.parse_generate_if_item(),
            other => Err(Error::Parse(format!(
                "unsupported pipeline module item token: {:?}",
                other
            ))),
        }
    }

    fn parse_generate_for_item(&mut self) -> Result<ModuleItem> {
        let start = self.toks[self.idx].start;
        self.expect(TokKind::KwFor)?;
        self.expect(TokKind::LParen)?;
        match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) if s == "genvar" => self.bump(),
            _ => {
                return Err(Error::Parse(
                    "expected `genvar` in generate for".to_string(),
                ));
            }
        }
        let genvar = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected genvar identifier".to_string())),
        };
        self.expect(TokKind::Eq)?;
        let start_expr = self.parse_expr_until(&[TokKind::Semi])?;
        self.expect(TokKind::Semi)?;
        match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) if s == genvar => self.bump(),
            _ => {
                return Err(Error::Parse(format!(
                    "expected loop variable `{genvar}` in generate for condition"
                )));
            }
        }
        if *self.cur() != TokKind::Other('<') {
            return Err(Error::Parse(
                "generate for condition must use `<`".to_string(),
            ));
        }
        self.bump();
        let limit_expr = self.parse_expr_until(&[TokKind::Semi])?;
        self.expect(TokKind::Semi)?;
        match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) if s == genvar => self.bump(),
            _ => {
                return Err(Error::Parse(format!(
                    "expected loop variable `{genvar}` in generate for step"
                )));
            }
        }
        self.expect(TokKind::Eq)?;
        match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) if s == genvar => self.bump(),
            _ => {
                return Err(Error::Parse(format!(
                    "expected loop variable `{genvar}` on generate for rhs"
                )));
            }
        }
        if *self.cur() != TokKind::Other('+') {
            return Err(Error::Parse("generate for step must use `+ 1`".to_string()));
        }
        self.bump();
        match self.toks[self.idx].kind.clone() {
            TokKind::Number(s) if s == "1" => self.bump(),
            _ => {
                return Err(Error::Parse(
                    "generate for step must increment by 1".to_string(),
                ));
            }
        }
        self.expect(TokKind::RParen)?;
        let body = self.parse_pipeline_item_block()?;
        let end = self.toks[self.idx - 1].end;
        Ok(ModuleItem::GenerateFor {
            genvar,
            start: start_expr,
            limit: limit_expr,
            body,
            span: Span { start, end },
        })
    }

    fn parse_generate_if_item(&mut self) -> Result<ModuleItem> {
        let start = self.toks[self.idx].start;
        let mut branches = Vec::new();
        loop {
            let branch_start = self.toks[self.idx].start;
            let cond = if *self.cur() == TokKind::KwIf {
                self.expect(TokKind::KwIf)?;
                self.expect(TokKind::LParen)?;
                let cond = self.parse_expr_until(&[TokKind::RParen])?;
                self.expect(TokKind::RParen)?;
                Some(cond)
            } else {
                None
            };
            let body = self.parse_pipeline_item_block()?;
            let branch_end = self.toks[self.idx - 1].end;
            branches.push(GenerateBranch {
                cond,
                body,
                span: Span {
                    start: branch_start,
                    end: branch_end,
                },
            });

            if *self.cur() != TokKind::KwElse {
                break;
            }
            let else_start = self.toks[self.idx].start;
            self.bump();
            if *self.cur() != TokKind::KwIf {
                let body = self.parse_pipeline_item_block()?;
                let branch_end = self.toks[self.idx - 1].end;
                branches.push(GenerateBranch {
                    cond: None,
                    body,
                    span: Span {
                        start: else_start,
                        end: branch_end,
                    },
                });
                break;
            }
        }
        let end = self.toks[self.idx - 1].end;
        Ok(ModuleItem::GenerateIf {
            branches,
            span: Span { start, end },
        })
    }

    fn parse_pipeline_item_block(&mut self) -> Result<Vec<ModuleItem>> {
        self.expect(TokKind::KwBegin)?;
        if *self.cur() == TokKind::Colon {
            self.bump();
            match self.toks[self.idx].kind.clone() {
                TokKind::Ident(_) => self.bump(),
                _ => return Err(Error::Parse("expected generate block label".to_string())),
            }
        }
        let mut items = Vec::new();
        while *self.cur() != TokKind::KwEnd {
            if *self.cur() == TokKind::Semi {
                self.bump();
                continue;
            }
            items.push(self.parse_pipeline_item()?);
        }
        self.expect(TokKind::KwEnd)?;
        Ok(items)
    }

    fn parse_port_list_ports(&mut self) -> Result<Vec<PortDecl>> {
        self.expect(TokKind::LParen)?;
        let mut ports: Vec<PortDecl> = Vec::new();
        loop {
            if *self.cur() == TokKind::RParen {
                self.bump();
                break;
            }

            let dir = match self.cur() {
                TokKind::KwInput => {
                    self.bump();
                    PortDir::Input
                }
                TokKind::KwOutput => {
                    self.bump();
                    PortDir::Output
                }
                _ => {
                    return Err(Error::Parse(
                        "expected `input` or `output` in port list".to_string(),
                    ));
                }
            };

            let ty = match self.cur() {
                TokKind::KwWire => {
                    self.bump();
                    PortTy::Wire
                }
                TokKind::KwLogic => {
                    self.bump();
                    PortTy::Logic
                }
                _ => {
                    return Err(Error::Parse(
                        "expected `wire` or `logic` in port list".to_string(),
                    ));
                }
            };

            let signed = if *self.cur() == TokKind::KwSigned {
                self.bump();
                true
            } else {
                false
            };

            let packed_dims = self.parse_packed_dims()?;
            let width = packed_dims_width(&packed_dims)?;

            let name = match self.toks[self.idx].kind.clone() {
                TokKind::Ident(s) => {
                    self.bump();
                    s
                }
                _ => return Err(Error::Parse("expected port identifier".to_string())),
            };
            let unpacked_dims = self.parse_unpacked_dims()?;
            let width = dims_total_width(width, &unpacked_dims)?;

            ports.push(PortDecl {
                dir,
                ty,
                signed,
                width,
                packed_dims,
                unpacked_dims,
                name,
            });

            match self.cur() {
                TokKind::Comma => {
                    self.bump();
                    continue;
                }
                TokKind::RParen => continue,
                _ => return Err(Error::Parse("expected `,` or `)` in port list".to_string())),
            }
        }
        Ok(ports)
    }

    fn parse_packed_dims(&mut self) -> Result<Vec<u32>> {
        let mut dims: Vec<u32> = Vec::new();
        while *self.cur() == TokKind::LBracket {
            self.bump();
            let msb_expr = self.parse_expr_until(&[TokKind::Colon])?;
            self.expect(TokKind::Colon)?;
            let lsb_expr = self.parse_expr_until(&[TokKind::RBracket])?;
            self.expect(TokKind::RBracket)?;
            let msb = self.eval_const_u32(&msb_expr)?;
            let lsb = self.eval_const_u32(&lsb_expr)?;
            if lsb != 0 {
                return Err(Error::Parse(
                    "packed declaration ranges must be zero-based `[N:0]`".to_string(),
                ));
            }
            if msb < lsb {
                return Err(Error::Parse("decl range msb<lsb not supported".to_string()));
            }
            dims.push(msb - lsb + 1);
        }
        if dims.is_empty() {
            dims.push(1);
        }
        Ok(dims)
    }

    fn parse_unpacked_dims(&mut self) -> Result<Vec<u32>> {
        let mut dims: Vec<u32> = Vec::new();
        while *self.cur() == TokKind::LBracket {
            self.bump();
            let first = self.parse_expr_until(&[TokKind::Colon, TokKind::RBracket])?;
            let dim = if *self.cur() == TokKind::Colon {
                self.bump();
                let second = self.parse_expr_until(&[TokKind::RBracket])?;
                let first_u = self.eval_const_u32(&first)?;
                let second_u = self.eval_const_u32(&second)?;
                if second_u != 0 {
                    return Err(Error::Parse(
                        "unpacked declaration ranges must be zero-based `[N:0]`".to_string(),
                    ));
                }
                first_u + 1
            } else {
                self.eval_const_u32(&first)?
            };
            self.expect(TokKind::RBracket)?;
            if dim == 0 {
                return Err(Error::Parse(
                    "unpacked array dimension must be > 0".to_string(),
                ));
            }
            dims.push(dim);
        }
        Ok(dims)
    }

    fn parse_wire_decl_with_span(&mut self) -> Result<(Decl, Span)> {
        let start = self.toks[self.idx].start;
        self.expect(TokKind::KwWire)?;

        let signed = if *self.cur() == TokKind::KwSigned {
            self.bump();
            true
        } else {
            false
        };

        let packed_dims = self.parse_packed_dims()?;
        let width = packed_dims_width(&packed_dims)?;

        let name = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected identifier in wire decl".to_string())),
        };
        let unpacked_dims = self.parse_unpacked_dims()?;
        let width = dims_total_width(width, &unpacked_dims)?;
        self.expect(TokKind::Semi)?;
        let end = self.toks[self.idx - 1].end;
        Ok((
            Decl {
                name,
                signed,
                width,
                packed_dims,
                unpacked_dims,
            },
            Span { start, end },
        ))
    }

    fn parse_logic_decl_with_span(&mut self) -> Result<(Decl, Span)> {
        let start = self.toks[self.idx].start;
        let d = self.parse_logic_decl()?;
        let end = self.toks[self.idx - 1].end;
        Ok((d, Span { start, end }))
    }

    fn parse_assign_item(&mut self) -> Result<ModuleItem> {
        let stmt_start = self.toks[self.idx].start;
        self.expect(TokKind::KwAssign)?;
        let lhs = self.parse_lhs()?;
        self.expect(TokKind::Eq)?;
        let (rhs, rhs_span) = self.parse_expr_with_span_until_semi()?;
        self.expect(TokKind::Semi)?;
        let stmt_end = self.toks[self.idx - 1].end;
        Ok(ModuleItem::Assign {
            lhs,
            rhs,
            rhs_span,
            span: Span {
                start: stmt_start,
                end: stmt_end,
            },
        })
    }

    fn parse_combo_function_with_spans(&mut self) -> Result<(FunctionDecl, Span, Span, Span)> {
        self.expect(TokKind::KwFunction)?;
        if *self.cur() == TokKind::KwAutomatic {
            self.bump();
        }

        // Return type: accept optional `logic` and `signed` in either order,
        // followed by an optional range.
        let mut saw_logic = false;
        let mut ret_signed = false;
        loop {
            match self.cur() {
                TokKind::KwLogic if !saw_logic => {
                    saw_logic = true;
                    self.bump();
                }
                TokKind::KwSigned if !ret_signed => {
                    ret_signed = true;
                    self.bump();
                }
                _ => break,
            }
        }

        let mut ret_width: u32 = 1;
        if *self.cur() == TokKind::LBracket {
            self.bump();
            let msb_expr = self.parse_expr_until(&[TokKind::Colon])?;
            self.expect(TokKind::Colon)?;
            let lsb_expr = self.parse_expr_until(&[TokKind::RBracket])?;
            self.expect(TokKind::RBracket)?;
            let msb = self.eval_const_u32(&msb_expr)?;
            let lsb = self.eval_const_u32(&lsb_expr)?;
            if msb < lsb {
                return Err(Error::Parse("decl range msb<lsb not supported".to_string()));
            }
            ret_width = msb - lsb + 1;
        }

        let name = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected function name".to_string())),
        };

        self.expect(TokKind::LParen)?;
        let args = self.parse_function_args_decls()?;
        self.expect(TokKind::RParen)?;
        self.expect(TokKind::Semi)?;
        let locals = self.parse_function_local_decls()?;

        let begin_start = self.toks[self.idx].start;
        let begin_end = self.toks[self.idx].end;
        self.expect(TokKind::KwBegin)?;
        let body = match self.cur().clone() {
            TokKind::KwUnique | TokKind::KwCasez => self.parse_unique_casez_body(&name)?,
            _ => {
                let assigns = self.parse_procedural_function_body()?;
                if assigns.len() == 1 && assigns[0].lhs == name {
                    FunctionBody::Assign {
                        value: assigns[0].value.clone(),
                        value_span: assigns[0].value_span,
                    }
                } else {
                    FunctionBody::Procedure { assigns }
                }
            }
        };
        let end_start = self.toks[self.idx].start;
        let end_tok_end = self.toks[self.idx].end;
        self.expect(TokKind::KwEnd)?;
        let end_end = self.toks[self.idx - 1].end;
        self.expect(TokKind::KwEndfunction)?;
        let body_span = Span {
            start: begin_start,
            end: end_end,
        };
        let begin_span = Span {
            start: begin_start,
            end: begin_end,
        };
        let end_span = Span {
            start: end_start,
            end: end_tok_end,
        };

        Ok((
            FunctionDecl {
                name,
                ret_width,
                ret_signed,
                args,
                locals,
                body,
            },
            body_span,
            begin_span,
            end_span,
        ))
    }

    fn parse_procedural_function_body(&mut self) -> Result<Vec<FunctionAssign>> {
        let mut assigns = Vec::new();
        while *self.cur() != TokKind::KwEnd {
            if *self.cur() == TokKind::Semi {
                self.bump();
                continue;
            }
            let lhs = match self.toks[self.idx].kind.clone() {
                TokKind::Ident(s) => {
                    self.bump();
                    s
                }
                _ => {
                    return Err(Error::Parse(
                        "expected assignment in function body".to_string(),
                    ));
                }
            };
            self.expect(TokKind::Eq)?;
            let (value, value_span) = self.parse_expr_with_span_until_semi()?;
            self.expect(TokKind::Semi)?;
            assigns.push(FunctionAssign {
                lhs,
                value,
                value_span,
            });
        }
        Ok(assigns)
    }

    fn parse_function_local_decls(&mut self) -> Result<Vec<Decl>> {
        let mut out = Vec::new();
        while matches!(self.cur(), TokKind::KwReg | TokKind::KwLogic) {
            out.push(self.parse_var_decl(true)?);
        }
        Ok(out)
    }

    fn parse_function_args_decls(&mut self) -> Result<Vec<Decl>> {
        let mut out: Vec<Decl> = Vec::new();
        loop {
            if *self.cur() == TokKind::RParen {
                break;
            }

            self.expect(TokKind::KwInput)?;
            out.push(self.parse_var_decl(false)?);

            match self.cur() {
                TokKind::Comma => {
                    self.bump();
                    continue;
                }
                TokKind::RParen => break,
                _ => return Err(Error::Parse("expected `,` or `)` in arg list".to_string())),
            }
        }
        Ok(out)
    }

    fn parse_var_decl(&mut self, expect_semi: bool) -> Result<Decl> {
        match self.cur() {
            TokKind::KwReg | TokKind::KwLogic => self.bump(),
            _ if expect_semi => {
                return Err(Error::Parse(
                    "expected `reg` or `logic` in declaration".to_string(),
                ));
            }
            _ => {}
        }

        let signed = if *self.cur() == TokKind::KwSigned {
            self.bump();
            true
        } else {
            false
        };

        let packed_dims = self.parse_packed_dims()?;
        let width = packed_dims_width(&packed_dims)?;

        let name = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected identifier in decl".to_string())),
        };
        let unpacked_dims = self.parse_unpacked_dims()?;
        let width = dims_total_width(width, &unpacked_dims)?;
        if expect_semi {
            self.expect(TokKind::Semi)?;
        }
        Ok(Decl {
            name,
            signed,
            width,
            packed_dims,
            unpacked_dims,
        })
    }

    fn parse_unique_casez_body(&mut self, fn_name: &str) -> Result<FunctionBody> {
        let casez_start = self.toks[self.idx].start;
        if *self.cur() == TokKind::KwUnique {
            self.bump();
        }
        self.expect(TokKind::KwCasez)?;
        self.expect(TokKind::LParen)?;
        let (selector, selector_span) = self.parse_expr_with_span_until(&[TokKind::RParen])?;
        self.expect(TokKind::RParen)?;
        let casez_end = self.toks[self.idx - 1].end;
        let casez_span = Span {
            start: casez_start,
            end: casez_end,
        };

        let mut arms: Vec<CasezArm> = Vec::new();
        let endcase_span: Span = loop {
            match self.cur().clone() {
                TokKind::KwEndcase => {
                    let s0 = self.toks[self.idx].start;
                    let e0 = self.toks[self.idx].end;
                    self.bump();
                    break Span { start: s0, end: e0 };
                }
                TokKind::End => {
                    return Err(Error::Parse("unexpected EOF in casez".to_string()));
                }
                TokKind::Semi => {
                    self.bump();
                    continue;
                }
                _ => {}
            }

            let pat = match self.cur().clone() {
                TokKind::Ident(s) if s == "default" => {
                    let s0 = self.toks[self.idx].start;
                    let e0 = self.toks[self.idx].end;
                    self.bump();
                    (None, Some(Span { start: s0, end: e0 }))
                }
                TokKind::CasezPattern { width, bits_msb } => {
                    let s0 = self.toks[self.idx].start;
                    let e0 = self.toks[self.idx].end;
                    self.bump();
                    (
                        Some(CasezPattern {
                            width,
                            bits_msb,
                            span: Span { start: s0, end: e0 },
                        }),
                        Some(Span { start: s0, end: e0 }),
                    )
                }
                TokKind::Number(n) => {
                    // Back-compat: allow numeric token and parse as a casez pattern (strict
                    // subset).
                    let s0 = self.toks[self.idx].start;
                    let e0 = self.toks[self.idx].end;
                    self.bump();
                    let mut p = parse_casez_pattern(&n)?;
                    p.span = Span { start: s0, end: e0 };
                    (Some(p), Some(Span { start: s0, end: e0 }))
                }
                other => {
                    return Err(Error::Parse(format!(
                        "bad casez arm pattern token: {other:?}"
                    )));
                }
            };
            self.expect(TokKind::Colon)?;
            let arm_start = pat
                .1
                .map(|s| s.start)
                .unwrap_or(self.toks[self.idx - 1].start);
            self.expect(TokKind::KwBegin)?;

            // Expect `<fn_name> = <expr>;` then `end`
            let lhs = match self.toks[self.idx].kind.clone() {
                TokKind::Ident(s) => {
                    self.bump();
                    s
                }
                _ => return Err(Error::Parse("expected assignment in case arm".to_string())),
            };
            if lhs != fn_name {
                return Err(Error::Parse(format!(
                    "expected function result assignment to `{fn_name}`, got `{lhs}`"
                )));
            }
            self.expect(TokKind::Eq)?;
            let (value, value_span) = self.parse_expr_with_span_until_semi()?;
            self.expect(TokKind::Semi)?;
            self.expect(TokKind::KwEnd)?;
            let arm_end = self.toks[self.idx - 1].end;

            arms.push(CasezArm {
                pat: pat.0,
                pat_span: pat.1,
                arm_span: Span {
                    start: arm_start,
                    end: arm_end,
                },
                value,
                value_span,
            });
        };
        Ok(FunctionBody::UniqueCasez {
            casez_span,
            selector,
            selector_span,
            endcase_span,
            arms,
        })
    }

    fn parse_logic_decl(&mut self) -> Result<Decl> {
        match self.cur() {
            TokKind::KwLogic | TokKind::KwReg => self.bump(),
            _ => return Err(Error::Parse("expected `logic` or `reg`".to_string())),
        }
        let signed = if *self.cur() == TokKind::KwSigned {
            self.bump();
            true
        } else {
            false
        };

        let packed_dims = self.parse_packed_dims()?;
        let width = packed_dims_width(&packed_dims)?;

        let name = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected identifier in decl".to_string())),
        };
        let unpacked_dims = self.parse_unpacked_dims()?;
        let width = dims_total_width(width, &unpacked_dims)?;
        self.expect(TokKind::Semi)?;
        Ok(Decl {
            name,
            signed,
            width,
            packed_dims,
            unpacked_dims,
        })
    }

    fn parse_always_ff(&mut self) -> Result<AlwaysFf> {
        self.expect(TokKind::KwAlwaysFf)?;
        self.expect(TokKind::At)?;
        self.expect(TokKind::LParen)?;
        self.expect(TokKind::KwPosedge)?;
        let clk_name = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected clock identifier".to_string())),
        };
        self.expect(TokKind::RParen)?;
        let body = self.parse_stmt()?;
        Ok(AlwaysFf { clk_name, body })
    }

    fn parse_stmt(&mut self) -> Result<Stmt> {
        match self.cur().clone() {
            TokKind::KwBegin => {
                self.bump();
                let mut stmts = Vec::new();
                while *self.cur() != TokKind::KwEnd {
                    if *self.cur() == TokKind::Semi {
                        self.bump();
                        continue;
                    }
                    stmts.push(self.parse_stmt()?);
                }
                self.expect(TokKind::KwEnd)?;
                Ok(Stmt::Begin(stmts))
            }
            TokKind::KwIf => self.parse_if(),
            TokKind::Semi => {
                self.bump();
                Ok(Stmt::Empty)
            }
            TokKind::Other('$') => self.parse_display(),
            TokKind::Ident(_) => self.parse_nba_assign(),
            other => Err(Error::Parse(format!(
                "unsupported statement token: {:?}",
                other
            ))),
        }
    }

    fn parse_display(&mut self) -> Result<Stmt> {
        // Parse a restricted `$display("...", <expr>...) ;` statement.
        let start = self.toks[self.idx].start;
        if *self.cur() != TokKind::Other('$') {
            return Err(Error::Parse("expected `$`".to_string()));
        }
        self.bump(); // '$'
        match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) if s == "display" => self.bump(),
            _ => return Err(Error::Parse("expected `$display`".to_string())),
        }

        // Scan until ';' (respect parentheses depth).
        let mut depth_paren: i32 = 0;
        while self.idx < self.toks.len() {
            let k = self.toks[self.idx].kind.clone();
            if depth_paren == 0 && k == TokKind::Semi {
                break;
            }
            match k {
                TokKind::LParen => depth_paren += 1,
                TokKind::RParen => depth_paren -= 1,
                _ => {}
            }
            self.idx += 1;
        }
        let end = self.toks[self.idx].start;
        self.expect(TokKind::Semi)?;
        let slice = self.src[start..end].trim();

        let (fmt, args) = parse_display_call(slice)?;
        Ok(Stmt::Display { fmt, args })
    }

    fn parse_if(&mut self) -> Result<Stmt> {
        self.expect(TokKind::KwIf)?;
        self.expect(TokKind::LParen)?;
        let cond = self.parse_expr_until(&[TokKind::RParen])?;
        self.expect(TokKind::RParen)?;
        let then_branch = Box::new(self.parse_stmt()?);
        let else_branch = if *self.cur() == TokKind::KwElse {
            self.bump();
            Some(Box::new(self.parse_stmt()?))
        } else {
            None
        };
        Ok(Stmt::If {
            cond,
            then_branch,
            else_branch,
        })
    }

    fn parse_nba_assign(&mut self) -> Result<Stmt> {
        let lhs = self.parse_lhs()?;
        self.expect(TokKind::Leq)?;
        let rhs = self.parse_expr_until(&[TokKind::Semi])?;
        self.expect(TokKind::Semi)?;
        Ok(Stmt::NbaAssign { lhs, rhs })
    }

    fn parse_lhs(&mut self) -> Result<Lhs> {
        let base = match self.toks[self.idx].kind.clone() {
            TokKind::Ident(s) => {
                self.bump();
                s
            }
            _ => return Err(Error::Parse("expected identifier in lhs".to_string())),
        };

        if *self.cur() != TokKind::LBracket {
            return Ok(Lhs::Ident(base));
        }
        let mut indices: Vec<VExpr> = Vec::new();
        loop {
            self.expect(TokKind::LBracket)?;
            let first = self.parse_expr_until(&[TokKind::Colon, TokKind::RBracket])?;
            if *self.cur() == TokKind::Colon {
                if !indices.is_empty() {
                    return Err(Error::Parse(
                        "mixed packed index and slice lhs not supported".to_string(),
                    ));
                }
                self.bump();
                let lsb = self.parse_expr_until(&[TokKind::RBracket])?;
                self.expect(TokKind::RBracket)?;
                if *self.cur() == TokKind::LBracket {
                    return Err(Error::Parse(
                        "multiple bracket groups after slice lhs not supported".to_string(),
                    ));
                }
                return Ok(Lhs::Slice {
                    base,
                    msb: first,
                    lsb,
                });
            }
            self.expect(TokKind::RBracket)?;
            indices.push(first);
            if *self.cur() != TokKind::LBracket {
                break;
            }
        }
        if indices.len() == 1 {
            Ok(Lhs::Index {
                base,
                index: indices.pop().expect("checked len"),
            })
        } else {
            Ok(Lhs::PackedIndex { base, indices })
        }
    }

    fn parse_expr_until(&mut self, end_kinds: &[TokKind]) -> Result<VExpr> {
        self.parse_expr_with_span_until(end_kinds)
            .map(|(expr, _)| expr)
    }

    fn parse_expr_with_span_until(&mut self, end_kinds: &[TokKind]) -> Result<(VExpr, Span)> {
        let span = self.parse_span_until(end_kinds)?;
        self.parse_expr_in_span(span)
    }

    fn parse_expr_with_span_until_semi(&mut self) -> Result<(VExpr, Span)> {
        self.parse_expr_with_span_until(&[TokKind::Semi])
    }

    fn parse_expr_in_span(&self, span: Span) -> Result<(VExpr, Span)> {
        let slice = self.src[span.start..span.end].trim();
        if slice.is_empty() {
            return Err(Error::Parse("empty expression".to_string()));
        }
        let expr = parse_expr(slice)?;
        Ok((expr, span))
    }

    fn parse_span_until(&mut self, end_kinds: &[TokKind]) -> Result<Span> {
        let start = self.toks[self.idx].start;
        let mut depth_paren: i32 = 0;
        let mut depth_brace: i32 = 0;
        let mut depth_bracket: i32 = 0;

        while self.idx < self.toks.len() {
            let k = self.toks[self.idx].kind.clone();
            if depth_paren == 0
                && depth_brace == 0
                && depth_bracket == 0
                && end_kinds.iter().any(|x| *x == k)
            {
                break;
            }
            match &k {
                TokKind::LParen => depth_paren += 1,
                TokKind::RParen => depth_paren -= 1,
                TokKind::LBracket => depth_bracket += 1,
                TokKind::RBracket => depth_bracket -= 1,
                TokKind::Other('{') => depth_brace += 1,
                TokKind::Other('}') => depth_brace -= 1,
                _ => {}
            }
            self.idx += 1;
        }

        let end = self.toks[self.idx].start;
        if start >= end {
            return Err(Error::Parse("empty span".to_string()));
        }
        Ok(Span { start, end })
    }

    // no skip_balanced_parens in v1; we parse a restricted port list.
}

fn parse_casez_pattern(s: &str) -> Result<CasezPattern> {
    // Expect `<w>'b<bits>` where bits may include '?'.
    let compact: String = s.chars().filter(|c| *c != '_').collect();
    let s = compact.as_str();
    let tick = s
        .find('\'')
        .ok_or_else(|| Error::Parse(format!("bad casez pattern: {s}")))?;
    let w_str = s[..tick].trim();
    let width: u32 = w_str
        .parse()
        .map_err(|_| Error::Parse(format!("bad casez width: {s}")))?;
    let base_and_bits = s[tick + 1..].trim();
    let base_and_bits = base_and_bits
        .strip_prefix('b')
        .or_else(|| base_and_bits.strip_prefix('B'))
        .ok_or_else(|| Error::Parse(format!("expected 'b in casez pattern: {s}")))?;
    let bits_msb = base_and_bits.trim().to_string();
    Ok(CasezPattern {
        width,
        bits_msb,
        span: Span { start: 0, end: 0 },
    })
}

fn packed_dims_width(dims: &[u32]) -> Result<u32> {
    dims.iter().try_fold(1u32, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| Error::Parse("packed decl width overflow".to_string()))
    })
}

fn dims_total_width(packed_width: u32, unpacked_dims: &[u32]) -> Result<u32> {
    unpacked_dims.iter().try_fold(packed_width, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| Error::Parse("packed decl width overflow".to_string()))
    })
}

fn parse_display_call(s: &str) -> Result<(String, Vec<VExpr>)> {
    // Very small `$display` parser for forms like:
    // $display("foo %d", expr, expr2)
    //
    // We only support a leading string literal format, followed by zero or more
    // comma-separated expressions.
    let s = s.trim();
    if !s.starts_with("$display") {
        return Err(Error::Parse("expected `$display`".to_string()));
    }
    let open = s
        .find('(')
        .ok_or_else(|| Error::Parse("expected `(` in $display".to_string()))?;
    let close = s
        .rfind(')')
        .ok_or_else(|| Error::Parse("expected `)` in $display".to_string()))?;
    if close <= open {
        return Err(Error::Parse("bad $display parens".to_string()));
    }
    let inside = s[open + 1..close].trim();
    if !inside.starts_with('"') {
        return Err(Error::Parse(
            "expected leading string literal in $display".to_string(),
        ));
    }
    // Find closing quote (no escape handling in v1).
    let mut end_quote: Option<usize> = None;
    for (i, c) in inside.char_indices().skip(1) {
        if c == '"' {
            end_quote = Some(i);
            break;
        }
    }
    let end_quote =
        end_quote.ok_or_else(|| Error::Parse("unterminated string in $display".to_string()))?;
    let fmt = inside[1..end_quote].to_string();
    let rest = inside[end_quote + 1..].trim();
    let rest = rest.strip_prefix(',').map(|s| s.trim()).unwrap_or(rest);
    let mut args: Vec<VExpr> = Vec::new();
    if !rest.is_empty() {
        for part in rest.split(',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            args.push(parse_expr(p)?);
        }
    }
    Ok((fmt, args))
}
