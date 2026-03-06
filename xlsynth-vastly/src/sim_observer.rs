// SPDX-License-Identifier: Apache-2.0

use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::ast::BinaryOp;
use crate::ast::Expr as VExpr;
use crate::eval::eval_ast_with_calls;
use crate::sv_ast::Stmt;

#[derive(Debug, Clone)]
pub struct SimObserver {
    pub clk_name: String,
    pub cond: VExpr,
    pub fmt: String,
    pub args: Vec<VExpr>,
}

pub fn extract_observers(clk_name: &str, stmt: &Stmt) -> Result<Vec<SimObserver>> {
    let mut out: Vec<SimObserver> = Vec::new();
    collect(stmt, None, clk_name, &mut out)?;
    Ok(out)
}

fn collect(
    stmt: &Stmt,
    cur_cond: Option<VExpr>,
    clk_name: &str,
    out: &mut Vec<SimObserver>,
) -> Result<()> {
    match stmt {
        Stmt::Begin(stmts) => {
            for s in stmts {
                collect(s, cur_cond.clone(), clk_name, out)?;
            }
            Ok(())
        }
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let then_cond = and_opt(cur_cond.clone(), cond.clone());
            collect(then_branch, then_cond, clk_name, out)?;

            if let Some(e) = else_branch {
                let else_cond = and_opt(cur_cond, else_taken_cond(cond.clone()));
                collect(e, else_cond, clk_name, out)?;
            }
            Ok(())
        }
        Stmt::Display { fmt, args } => {
            let cond = cur_cond.unwrap_or_else(true_expr);
            out.push(SimObserver {
                clk_name: clk_name.to_string(),
                cond,
                fmt: fmt.clone(),
                args: args.clone(),
            });
            Ok(())
        }
        Stmt::NbaAssign { .. } => Err(Error::Parse(
            "cannot extract observer from stateful assignment".to_string(),
        )),
        Stmt::Empty => Ok(()),
    }
}

fn and_opt(a: Option<VExpr>, b: VExpr) -> Option<VExpr> {
    match a {
        None => Some(b),
        Some(a) => Some(VExpr::Binary {
            op: BinaryOp::LogicalAnd,
            lhs: Box::new(a),
            rhs: Box::new(b),
        }),
    }
}

fn true_expr() -> VExpr {
    VExpr::Literal(Value4::new(1, Signedness::Unsigned, vec![LogicBit::One]))
}

fn else_taken_cond(cond: VExpr) -> VExpr {
    // Procedural Verilog if/else control flow takes else when condition is 0/X/Z.
    let cond_is_zero = VExpr::Binary {
        op: BinaryOp::CaseEq,
        lhs: Box::new(cond.clone()),
        rhs: Box::new(VExpr::Literal(Value4::new(
            1,
            Signedness::Unsigned,
            vec![LogicBit::Zero],
        ))),
    };
    let cond_is_x = VExpr::Binary {
        op: BinaryOp::CaseEq,
        lhs: Box::new(cond.clone()),
        rhs: Box::new(VExpr::Literal(Value4::new(
            1,
            Signedness::Unsigned,
            vec![LogicBit::X],
        ))),
    };
    let cond_is_z = VExpr::Binary {
        op: BinaryOp::CaseEq,
        lhs: Box::new(cond),
        rhs: Box::new(VExpr::Literal(Value4::new(
            1,
            Signedness::Unsigned,
            vec![LogicBit::Z],
        ))),
    };
    VExpr::Binary {
        op: BinaryOp::LogicalOr,
        lhs: Box::new(VExpr::Binary {
            op: BinaryOp::LogicalOr,
            lhs: Box::new(cond_is_zero),
            rhs: Box::new(cond_is_x),
        }),
        rhs: Box::new(cond_is_z),
    }
}

impl SimObserver {
    pub fn eval_and_format(&self, env: &crate::Env) -> Result<Option<String>> {
        let c = eval_ast_with_calls(&self.cond, env, None, None)?;
        if c.to_bool4() != LogicBit::One {
            return Ok(None);
        }
        let mut avs: Vec<Value4> = Vec::with_capacity(self.args.len());
        for a in &self.args {
            avs.push(eval_ast_with_calls(a, env, None, None)?);
        }
        Ok(Some(format_display(&self.fmt, &avs)))
    }
}

fn format_display(fmt: &str, args: &[Value4]) -> String {
    // Minimal formatter: replace `%d` / `%0d` with decimal text (or X if unknown).
    let mut out = String::new();
    let mut i = 0usize;
    let mut arg_i = 0usize;
    let bs = fmt.as_bytes();
    while i < bs.len() {
        if bs[i] == b'%' {
            let mut j = i + 1;
            if j < bs.len() && bs[j] == b'0' {
                j += 1;
            }
            if j < bs.len() && bs[j] == b'd' {
                let v = args.get(arg_i);
                arg_i += 1;
                if let Some(v) = v {
                    if let Some(s) = v.to_decimal_string_if_known() {
                        out.push_str(&s);
                    } else {
                        out.push('X');
                    }
                } else {
                    out.push('X');
                }
                i = j + 1;
                continue;
            }
        }
        out.push(bs[i] as char);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::format_display;
    use crate::LogicBit;
    use crate::Signedness;
    use crate::Value4;

    fn vbits(width: u32, signedness: Signedness, msb: &str) -> Value4 {
        assert_eq!(msb.len(), width as usize);
        let mut bits = Vec::with_capacity(width as usize);
        for c in msb.chars().rev() {
            bits.push(match c {
                '0' => LogicBit::Zero,
                '1' => LogicBit::One,
                'x' | 'X' => LogicBit::X,
                'z' | 'Z' => LogicBit::Z,
                _ => panic!("bad bit char {c}"),
            });
        }
        Value4::new(width, signedness, bits)
    }

    #[test]
    fn format_display_supports_percent_0d() {
        let out = format_display(
            "a=%0d b=%d",
            &[
                vbits(4, Signedness::Unsigned, "0011"),
                vbits(4, Signedness::Unsigned, "0101"),
            ],
        );
        assert_eq!(out, "a=3 b=5");
    }
}
