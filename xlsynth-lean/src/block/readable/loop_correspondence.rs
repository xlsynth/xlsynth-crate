// SPDX-License-Identifier: Apache-2.0

//! Prove the representation change once per helper, then reuse it at each call.
//! Counted loops additionally need the same statement for an arbitrary trip
//! count.

use std::collections::BTreeMap;

use super::*;

impl Presentation<'_> {
    pub(super) fn loop_correspondence(&self) -> Result<String> {
        let mut out = String::from("import ReadableRepresentation\n\nnamespace XlsReadable\n\n");
        out.push_str(BOOL_CORRESPONDENCE);
        self.proof_tactics(&mut out)?;
        out.push_str(r#"private theorem split_prod_ite {α β : Type} (p : Prop) [Decidable p] (x y : α × β) :
    (if p then x else y) = ((if p then x.1 else y.1), (if p then x.2 else y.2)) := by
  split <;> rfl

private theorem countedFor_succ {w : Nat} {α : Type} (n stride : Nat) (f : BitVec w → α → α) (z : α) :
    XlsBlock.countedFor (n+1) stride f z = f (BitVec.ofNat w (n*stride)) (XlsBlock.countedFor n stride f z) := rfl

"#);
        let mut rules = BTreeMap::new();
        let loop_bodies = self
            .block
            .helpers
            .iter()
            .map(|f| &f.graph)
            .chain([self.graph])
            .flat_map(|graph| &graph.nodes)
            .filter_map(|node| match &node.payload {
                NodePayload::CountedFor { body, .. } => Some(body.as_str()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        for (index, function) in self.block.helpers.iter().enumerate() {
            let Some(sig) = self.interface.helpers.get(&function.name) else {
                continue;
            };
            let raw = lean::helper_path(&function.name)?;
            let readable = format!("Helpers.{}", identifier(&sig.name)?);
            let mut binders = Vec::new();
            let mut params = Vec::new();
            let mut decoded = Vec::new();
            for (i, (node, (_, ty))) in function.param_nodes().zip(&sig.params).enumerate() {
                let name = format!("p{i}");
                binders.push(format!("({name} : {})", lean_type(&node.ty)?));
                decoded.push(self.convert(Some(ty), &name, false)?);
                params.push(name);
            }
            let call = format!("({readable} {})", decoded.join(" "));
            let encoded = self.convert(Some(&sig.result), &call, true)?;
            let called = call_rules(&function.graph, &rules);
            writeln!(
                out,
                r#"private theorem helper_{index} {} :
    {raw} {} = {encoded} := by
  unfold {raw} {readable}
  xls_normalize [{called}]
  all_goals xls_codecs [{called}]
  all_goals first | rfl | (simp only [split_prod_ite, prod_fst_ite, prod_snd_ite, ite_self]; rfl)"#,
                binders.join(" "),
                params.join(" ")
            )?;
            // Keep Lean's definitional equality checker from expanding the
            // whole controller when comparing unrelated arguments at a call.
            writeln!(out, "attribute [local irreducible] {raw} {readable}\n")?;
            if loop_bodies.contains(function.name.as_str()) {
                let raw_body = loop_body(&raw, &params[2..]);
                let readable_body = loop_body(&readable, &decoded[2..]);
                let fold = format!(
                    "(XlsBlock.countedFor count stride ({readable_body}) {})",
                    decoded[1]
                );
                let encoded = self.convert(Some(&sig.result), &fold, true)?;
                writeln!(
                    out,
                    r#"private theorem loop_{index} (count stride : Nat) {} :
    XlsBlock.countedFor count stride ({raw_body}) p1 = {encoded} := by
  induction count with
  | zero =>
    xls_normalize [XlsBlock.countedFor, Nat.fold_zero]
    all_goals rfl
  | succ n ih =>
    simp only [countedFor_succ]
    rw [ih]
    xls_normalize [helper_{index}]
    all_goals rfl
"#,
                    binders[1..].join(" ")
                )?;
            }
            rules.insert(function.name.as_str(), index);
        }
        let deps = call_rules(self.graph, &rules);
        // Checked captures retain their underlying declaration even when the
        // caller's input uses an alias. Expose only the alias wrappers so the
        // underlying codec's round-trip lemma can cancel decode and encode.
        let mut aliases = Vec::new();
        for decl in self.declarations {
            if let TypeDecl::Alias { name, .. } = decl {
                let name = identifier(name)?;
                aliases.push(format!("XlsCodec.{name}.encode, XlsCodec.{name}.decode"));
            }
        }
        let aliases = aliases.join(", ");
        writeln!(
            out,
            r#"theorem encoded_eval (q : State) (x : Input) :
    encodeEvaluation (eval q x) = XlsBlock.eval (encodeState q) (encodeInput x) := by
  xls_normalize [eval, XlsBlock.eval, encodeState, encodeInput, encodeOutput, encodeAssertions, encodeEvaluation{}]
  all_goals first | rfl | (xls_normalize [{aliases}] <;> rfl)
"#,
            if deps.is_empty() {
                String::new()
            } else {
                format!(", {deps}")
            }
        )?;
        out.push_str(CORRESPONDENCE);
        out.push_str("\nend XlsReadable\n");
        Ok(out)
    }

    /// Keep the shared codec rule lists in one place in the Lean module.
    fn proof_tactics(&self, out: &mut String) -> Result<()> {
        let mut rules = vec![
            "XlsCodec.raw_select, raw_not, raw_and, raw_or, raw_flag, decode_ofBool".into(),
            BOOL_ROUND_TRIP_RULES.join(", "),
            VECTOR_MAP_RULES.join(", "),
            BOOL_ENCODE_ITE.into(),
            PROJECTION_ITE_RULES.join(", "),
            "ite_self, Vector.map_mk, List.map_toArray, List.map_cons, List.map_nil".into(),
        ];
        let mut codecs = Vec::new();
        for decl in self.declarations {
            let name = identifier(decl.name())?;
            let codec = format!("XlsCodec.{name}");
            rules.extend(codec_round_trip_rules(&name));
            rules.extend(declaration_ite_rules(decl, &name)?);
            codecs.extend([format!("{codec}.encode"), format!("{codec}.decode")]);
        }
        let rules = rules.join(", ");
        for (name, extra) in [
            ("xls_normalize", String::new()),
            ("xls_codecs", codecs.join(", ")),
        ] {
            let prefix = if extra.is_empty() {
                String::new()
            } else {
                format!("{extra}, ")
            };
            writeln!(
                out,
                r#"open Lean.Parser.Tactic in
local macro "{name}" "[" rs:simpLemma,* "]" : tactic => do
  let rs : Lean.Syntax.TSepArray [``simpStar, ``simpErase, ``simpLemma] "," := ⟨rs.elemsAndSeps⟩
  `(tactic| simp (config := {{failIfUnchanged := false}}) only [{prefix}{rules}, $rs,*])
"#
            )?;
        }
        Ok(())
    }
}

fn loop_body(name: &str, captures: &[String]) -> String {
    if captures.is_empty() {
        name.into()
    } else {
        format!("{name} · · {}", captures.join(" "))
    }
}

fn call_rules(graph: &ir::NodeGraph, rules: &BTreeMap<&str, usize>) -> String {
    graph
        .nodes
        .iter()
        .filter_map(|node| {
            let (name, prefix) = match &node.payload {
                NodePayload::CountedFor { body, .. } => (body, "loop"),
                NodePayload::Invoke { to_apply, .. } => (to_apply, "helper"),
                _ => return None,
            };
            rules.get(name.as_str()).map(|i| format!("{prefix}_{i}"))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ")
}
