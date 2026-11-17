// SPDX-License-Identifier: Apache-2.0

//! Instantiate physical call sites with the source-declared function when its
//! checked signature is available. A compiler-outlined loop body keeps its
//! physical signature, including when it calls a source-declared function.

use super::{CORRESPONDENCE, Presentation, identifier, lean, lean_type, source_type_in};
use crate::block::source_types::HelperSignature;
use anyhow::Result;
use std::{collections::BTreeMap, fmt::Write};

pub(super) struct Calls {
    pub(super) declarations: String,
    pub(super) eval: String,
    proofs: String,
    block_rules: Vec<String>,
}

/// A raw-signature function value for invoking a checked callee, and its
/// theorem of equality to the ordinary raw callee. Both can be substituted
/// into any caller without inspecting that caller's equations.
struct Adapter {
    name: String,
    theorem: String,
}

fn arguments(
    callees: &[&str],
    adapters: &BTreeMap<&str, Adapter>,
) -> Result<(String, Vec<String>)> {
    let mut arguments = String::new();
    let mut rules = Vec::new();
    for &callee in callees {
        if let Some(adapter) = adapters.get(callee) {
            write!(arguments, " {}", adapter.name)?;
            rules.push(adapter.theorem.clone());
        } else {
            write!(arguments, " {}", lean::helper_path("XlsBlock", callee)?)?;
        }
    }
    Ok((arguments, rules))
}

impl Calls {
    pub(super) fn new(model: &Presentation<'_>) -> Result<Self> {
        let mut declarations = String::new();
        let mut proofs = String::new();
        let mut adapters = BTreeMap::new();

        for (index, function) in model.block.helpers.iter().enumerate() {
            let callees = lean::direct_callees(function);
            let (arguments, rules) = arguments(&callees, &adapters)?;
            let signature = model.interface.helpers.get(&function.name);
            if signature.is_none() && rules.is_empty() {
                continue;
            }
            let raw = lean::helper_path("XlsBlock", &function.name)?;
            let body = if callees.is_empty() {
                raw.clone()
            } else {
                lean::body_path("XlsBlock", &function.name)?
            };
            let kernel = format!("{body}{arguments}");
            let name = format!(
                "_root_.XlsReadable.Calls.{}",
                lean::helper_identifier(&function.name)?
            );
            let mut unfold = vec![name.clone()];
            if let Some(signature) = signature {
                let path = &model.callables.helpers[function.name.as_str()];
                model.write_function(&mut declarations, path, &kernel, signature)?;
                write!(declarations, "def {name}")?;
                let mut call = path.clone();
                for (position, (node, (_, ty))) in
                    function.params.iter().zip(&signature.params).enumerate()
                {
                    let arg = format!("a{position}");
                    write!(
                        declarations,
                        " ({arg} : {})",
                        lean_type(function.get_node_ty(*node))?
                    )?;
                    write!(call, " ({})", model.convert(Some(ty), &arg, false)?)?;
                }
                let result = model.convert(Some(&signature.result), "result", true)?;
                writeln!(
                    declarations,
                    " : {} :=\n  let result := {call}\n  {result}\n",
                    lean_type(&function.ret_ty)?
                )?;
                unfold.push(path.clone());
            } else {
                writeln!(declarations, "def {name} := {kernel}\n")?;
            }
            let theorem = format!("eq_call_{index}");
            if !callees.is_empty() {
                unfold.push(raw.clone());
            }
            writeln!(proofs, "private theorem {theorem} : {name} = {raw} := by")?;
            if signature.is_some() && !function.params.is_empty() {
                let params = (0..function.params.len())
                    .map(|i| format!("a{i}"))
                    .collect::<Vec<_>>();
                writeln!(proofs, "  funext {}", params.join(" "))?;
            }
            // The same physical computation appears on each side. Prove the
            // equality for any such function so simplification only sees its
            // arguments and the enclosing result codec, never its equations.
            writeln!(proofs, "  unfold {}", unfold.join(" "))?;
            writeln!(proofs, "  generalize {body} = body")?;
            if rules.is_empty() {
                writeln!(proofs, "  simp\n")?;
            } else {
                writeln!(proofs, "  simp [{}]\n", rules.join(", "))?;
            }
            adapters.insert(function.name.as_str(), Adapter { name, theorem });
        }

        let (arguments, block_rules) =
            arguments(&lean::direct_callees(&model.block.graph), &adapters)?;
        let eval = if block_rules.is_empty() {
            "_root_.XlsBlock.eval".to_owned()
        } else {
            format!("_root_.XlsBlock.evalWith{arguments}")
        };
        Ok(Self {
            declarations,
            eval,
            proofs,
            block_rules,
        })
    }

    pub(super) fn correspondence(&self) -> String {
        let mut source = String::from(
            "import ReadableRepresentation\n\nnamespace XlsReadable\n\nattribute [local simp] L.Function.comp_def\n\n",
        );
        source.push_str(&self.proofs);
        let mut rules = vec!["eval_eq_decoded", "encode_decode_evaluation"];
        if !self.block_rules.is_empty() {
            rules.push("XlsBlock.eval");
            rules.extend(self.block_rules.iter().map(String::as_str));
        }
        writeln!(source, "theorem encoded_eval (q : State) (x : Input) :\n    encodeEvaluation (eval q x) = XlsBlock.eval (encodeState q) (encodeInput x) := by\n  simp only [{}]\n", rules.join(", ")).unwrap();
        source.push_str(CORRESPONDENCE);
        source.push_str("\nend XlsReadable\n");
        source
    }
}

impl Presentation<'_> {
    /// Encode the declared parameters at entry and decode the result at
    /// return. The physical body takes the selected functions as its call
    /// arguments, so calls from a source function to another stay observable.
    fn write_function(
        &self,
        out: &mut String,
        path: &str,
        kernel: &str,
        signature: &HelperSignature,
    ) -> Result<()> {
        // A source module literally named _root_ must be declared from the top.
        let top_level = path.contains(".«_root_».");
        if top_level {
            write!(
                out,
                "end XlsReadable\n\nopen XlsReadable in\ndef {}",
                path.trim_start_matches("_root_.")
            )?;
        } else {
            write!(out, "def {path}")?;
        }
        let mut call = kernel.to_owned();
        let shadowed = |name: &str| signature.params.iter().any(|(param, _)| param == name);
        for (param, ty) in &signature.params {
            let binder = match param.as_str() {
                "_root_" | "XlsCodec" => format!("«xls argument {param}»"),
                _ => identifier(param)?,
            };
            let ty_name = source_type_in(self.bx_namespace, ty, None, &shadowed)?;
            write!(out, " ({binder} : {ty_name})")?;
            write!(call, " ({})", self.convert(Some(ty), &binder, true)?)?;
        }
        writeln!(
            out,
            " : {} :=\n  let result := {call}\n  {}\n",
            source_type_in(self.bx_namespace, &signature.result, None, &shadowed)?,
            self.convert(Some(&signature.result), "result", false)?
        )?;
        if top_level {
            out.push_str("namespace XlsReadable\n\n");
        }
        Ok(())
    }
}
