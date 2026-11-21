// SPDX-License-Identifier: Apache-2.0

//! Emit a checked helper in its source types. The enclosed IR equations and
//! their operands use physical values, including in compiler-outlined helpers.

use super::{BLOCK_NAMESPACE, Presentation, SourceType, identifier, lean, source_type_in};
use anyhow::Result;
use std::fmt::Write;
use xlsynth_pir::ir;

impl lean::CheckedCalls for Presentation<'_> {
    fn call(&self, physical: &str, args: &[String]) -> Result<Option<String>> {
        let Some(signature) = self.interface.helpers.get(physical) else {
            return Ok(None);
        };
        let mut call = self.callables.helpers[&signature.source_name].clone();
        for ((_, ty), arg) in signature.params.iter().zip(args) {
            write!(call, " ({})", self.convert(Some(ty), arg, false)?)?;
        }
        // Tuple encoding projects individual fields. Bind the call once so
        // every projection reads the same result, even for nested products.
        let result = if matches!(signature.result, SourceType::Tuple { .. }) {
            let encoded = self.convert(Some(&signature.result), "result", true)?;
            format!("(let result := {call}; {encoded})")
        } else {
            self.convert(Some(&signature.result), &call, true)?
        };
        Ok(Some(result))
    }

    fn write_helper(&self, out: &mut String, function: &ir::Fn) -> Result<bool> {
        let Some(signature) = self.interface.helpers.get(&function.name) else {
            return Ok(false);
        };
        let path = &self.callables.helpers[&signature.source_name];
        // Lean treats a quoted _root_ component specially in a rooted
        // declaration. Declare this source path directly from the top level.
        let top_level = path.contains(".«_root_».");
        if top_level {
            write!(
                out,
                "end {BLOCK_NAMESPACE}\n\ndef {}",
                path.trim_start_matches("_root_.")
            )?;
        } else {
            write!(out, "def {path}")?;
        }
        let shadowed = |name: &str| signature.params.iter().any(|(param, _)| param == name);
        let mut entry = String::new();
        let mut physical_params = Vec::new();
        for (param, ty) in &signature.params {
            // Lean resolves a local before a qualified declaration, even for
            // _root_. Keep codec and absolute paths usable in the body.
            let binder = match param.as_str() {
                "_root_" | "XlsCodec" => format!("«xls argument {param}»"),
                _ => identifier(param)?,
            };
            let ty_name = source_type_in(self.bx_namespace, ty, None, &shadowed)?;
            write!(out, " ({binder} : {ty_name})")?;
            // The equation renderer introduces its own locals and lambda
            // binders. Spaces keep these wire names distinct from both those
            // names and source parameters such as v0 or xls_node_1_x.
            let wire = format!("«xls wire {param}»");
            let encoded = self.convert(Some(ty), &binder, true)?;
            writeln!(entry, "  let {wire} := {encoded}")?;
            physical_params.push(wire);
        }
        writeln!(
            out,
            " : {} :=",
            source_type_in(self.bx_namespace, &signature.result, None, &shadowed)?
        )?;
        out.push_str(&entry);
        let result = lean::write_helper_equations(
            out,
            BLOCK_NAMESPACE,
            function,
            Some(self),
            &physical_params,
        )?;
        writeln!(
            out,
            "  {}\n",
            self.convert(Some(&signature.result), &result, false)?
        )?;
        if top_level {
            writeln!(out, "namespace {BLOCK_NAMESPACE}\n")?;
        }
        Ok(true)
    }
}
