// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::Error;
use crate::LogicBit;
use crate::Result;
use crate::Signedness;
use crate::Value4;
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
    pub packed_dims: Vec<u32>,
    pub unpacked_dims: Vec<u32>,
}

pub fn compile_module(src: &str) -> Result<CompiledModule> {
    // Keep the legacy parser gate for this API surface, but switch lowering to
    // the unified pipeline compiler path.
    let _legacy_shape: Module = parse_module(src)?;
    let pipeline = crate::pipeline_compile::compile_pipeline_module(src)?;
    lower_pipeline_to_legacy(pipeline)
}

fn lower_pipeline_to_legacy(
    pipeline: crate::pipeline_compile::CompiledPipelineModule,
) -> Result<CompiledModule> {
    if !pipeline.combo.assigns.is_empty() || !pipeline.combo.functions.is_empty() {
        return Err(Error::Parse(
            "compile_module only supports declarations and always_ff in v1".to_string(),
        ));
    }
    if pipeline.seqs.len() > 1 {
        return Err(Error::Parse(
            "multiple always_ff blocks not supported".to_string(),
        ));
    }

    let state_regs = pipeline
        .seqs
        .first()
        .map(|seq| seq.state_regs.clone())
        .unwrap_or_default();
    let body = pipeline
        .seqs
        .first()
        .map(|seq| seq.body.clone())
        .unwrap_or(Stmt::Empty);

    for r in &state_regs {
        if !pipeline.combo.decls.contains_key(r) {
            return Err(Error::Parse(format!(
                "state reg `{}` must have a `logic` declaration in v1",
                r
            )));
        }
    }

    Ok(CompiledModule {
        module_name: pipeline.module_name,
        clk_name: pipeline.clk_name,
        consts: pipeline.combo.consts,
        decls: pipeline.combo.decls,
        state_regs,
        body,
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
