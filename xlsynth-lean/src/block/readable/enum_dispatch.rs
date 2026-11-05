// SPDX-License-Identifier: Apache-2.0

//! Give each recovered enum match a name so its agreement with the physical
//! priority selector can be proved independently of the expression it matches.

use super::*;

/// Operand order matters when rewriting an equality on symbolic bits.
pub(super) enum Comparison {
    LiteralFirst(u64),
    LiteralLast(u64),
}

/// One comparison per priority arm, lowest selector bit first.
pub(super) struct Dispatch<'a> {
    pub(super) subject: NodeRef,
    pub(super) name: &'a str,
    pub(super) comparisons: Vec<Comparison>,
}

impl Dispatch<'_> {
    fn arm_index(&self, encoding: u64) -> Option<usize> {
        self.comparisons.iter().position(|test| match test {
            Comparison::LiteralFirst(n) | Comparison::LiteralLast(n) => *n == encoding,
        })
    }

    fn arguments(&self) -> String {
        (0..self.comparisons.len())
            .map(|i| format!("arm{i}"))
            .chain(["fallback".into()])
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn physical_selection(&self) -> String {
        let flags = self
            .comparisons
            .iter()
            .rev()
            .map(|test| match test {
                Comparison::LiteralFirst(n) => format!("(BitVec.ofBool (({n}#2) == bits))"),
                Comparison::LiteralLast(n) => format!("(BitVec.ofBool (bits == ({n}#2)))"),
            })
            .reduce(|prefix, flag| format!("(BitVec.append {prefix} {flag})"))
            .unwrap();
        let priorities = (0..self.comparisons.len())
            .rev()
            .fold("fallback".to_owned(), |rest, i| {
                format!("(if flags.getLsbD {i} then arm{i} else {rest})")
            });
        format!("let flags := {flags}\n       {priorities}")
    }
}

impl Presentation<'_> {
    pub(super) fn enum_dispatch_name(&self, id: usize) -> String {
        match self.helper_index {
            Some(index) => format!("helper_{index}_node_{id}"),
            None => format!("node_{id}"),
        }
    }

    fn enum_dispatches(&self) -> Result<Vec<(usize, Dispatch<'_>)>> {
        let mut dispatches = Vec::new();
        for node in &self.graph.nodes {
            if let NodePayload::PrioritySel {
                selector,
                default: Some(_),
                ..
            } = node.payload
                && let Some(dispatch) = self.enum_selector(selector)?
            {
                dispatches.push((node.text_id, dispatch));
            }
        }
        Ok(dispatches)
    }

    pub(super) fn enum_dispatch_definitions(&self) -> Result<String> {
        let mut out = String::new();
        for (id, dispatch) in self.enum_dispatches()? {
            let name = self.enum_dispatch_name(id);
            let ty = identifier(dispatch.name)?;
            writeln!(
                out,
                "def EnumDispatch.{name} {{α : Type}} (s : XlsSource.{ty}) ({} : α) : α :=\n  match s with",
                dispatch.arguments()
            )?;
            for (member, encoding) in self.enum_cases(dispatch.name)? {
                let arm = dispatch
                    .arm_index(encoding)
                    .map_or_else(|| "fallback".into(), |i| format!("arm{i}"));
                writeln!(out, "  | .{} => {arm}", identifier(&member)?)?;
            }
            out.push('\n');
        }
        Ok(out)
    }

    /// Rewrite on the subject itself, including an enum decoded from computed
    /// bits.
    pub(super) fn enum_dispatch_laws(&self, out: &mut String) -> Result<Vec<String>> {
        let mut rules = Vec::new();
        for (id, dispatch) in self.enum_dispatches()? {
            let ty = identifier(dispatch.name)?;
            let args = dispatch.arguments();
            let name = self.enum_dispatch_name(id);
            let rule = format!("{name}_priority");
            writeln!(
                out,
                "private theorem {rule} {{α : Type}} (s : XlsSource.{ty}) ({args} : α) :\n    EnumDispatch.{name} s {args} =\n      (let bits := XlsCodec.{ty}.encode s\n       {}) := by\n  cases s <;> rfl\n",
                dispatch.physical_selection()
            )?;
            rules.push(rule);
        }
        Ok(rules)
    }
}
