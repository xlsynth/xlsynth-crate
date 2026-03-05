// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;
use crate::sv_ast::Decl;
use crate::sv_ast::Lhs;
use crate::sv_ast::Module;
use crate::sv_ast::Stmt;
use crate::sv_parser::parse_module;

pub type State = BTreeMap<String, Value4>;

#[derive(Debug, Clone)]
pub struct CompiledModule {
    pub module_name: String,
    pub clk_name: String,
    pub consts: BTreeMap<String, Value4>,
    pub decls: BTreeMap<String, DeclInfo>,
    pub state_regs: BTreeSet<String>,
    pub body: Stmt,
}

#[derive(Debug, Clone)]
pub struct DeclInfo {
    pub width: u32,
    pub signedness: Signedness,
}

pub fn compile_module(src: &str) -> Result<CompiledModule> {
    let m: Module = parse_module(src)?;
    let mut decls: BTreeMap<String, DeclInfo> = BTreeMap::new();
    for Decl {
        name,
        signed,
        width,
    } in m.decls.clone()
    {
        let signedness = if signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        };
        decls.insert(name, DeclInfo { width, signedness });
    }

    let mut state_regs: BTreeSet<String> = BTreeSet::new();
    collect_state_regs(&m.always_ff.body, &mut state_regs);

    // Require state regs declared so we can size initial X state.
    for r in &state_regs {
        if !decls.contains_key(r) {
            return Err(Error::Parse(format!(
                "state reg `{}` must have a `logic` declaration in v1",
                r
            )));
        }
    }

    Ok(CompiledModule {
        module_name: m.name,
        clk_name: m.always_ff.clk_name,
        consts: m.params,
        decls,
        state_regs,
        body: m.always_ff.body,
    })
}

impl CompiledModule {
    /// Time-zero state: all sequential elements are X (your requirement).
    pub fn initial_state_x(&self) -> State {
        let mut s: State = BTreeMap::new();
        for name in &self.state_regs {
            let info = self.decls.get(name).expect("decl checked");
            s.insert(
                name.clone(),
                Value4::new(
                    info.width,
                    info.signedness,
                    vec![LogicBit::X; info.width as usize],
                ),
            );
        }
        s
    }

    pub fn step(&self, inputs: &crate::Env, state: &State) -> Result<State> {
        crate::module_eval::step_module(self, inputs, state)
    }
}

fn collect_state_regs(stmt: &Stmt, out: &mut BTreeSet<String>) {
    match stmt {
        Stmt::Begin(stmts) => {
            for s in stmts {
                collect_state_regs(s, out);
            }
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_state_regs(then_branch, out);
            if let Some(e) = else_branch {
                collect_state_regs(e, out);
            }
        }
        Stmt::NbaAssign { lhs, .. } => match lhs {
            Lhs::Ident(b) => {
                out.insert(b.clone());
            }
            Lhs::Index { base, .. } => {
                out.insert(base.clone());
            }
            Lhs::Slice { base, .. } => {
                out.insert(base.clone());
            }
        },
        Stmt::Display { .. } => {}
        Stmt::Empty => {}
    }
}
