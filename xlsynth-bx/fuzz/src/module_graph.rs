// SPDX-License-Identifier: Apache-2.0

//! Compare a Rust module outline and the BX import boundary to native DSLX.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

use xlsynth::dslx::{self, ImportData, MatchableModuleMember};
use xlsynth_bx::ast::Type;
use xlsynth_bx::{
    CheckedModule, ImportOptions, OutlineKind, check_module_with_options, parse_dslx_outline,
    parse_module,
};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;

use crate::eval::expr_eval::{ExpressionEvaluator, bx_source, compile_bundled_module, dslx_source};

/// Two nominally distinct packet declarations, a chain of aliases selecting
/// one of them, and a consumer that embeds the selected declaration in a frame.
struct ModuleGraph {
    sources: BTreeMap<String, String>,
    selected: &'static str,
    other: &'static str,
    public_packet: bool,
    public_enum: bool,
    field_syntax: Vec<(String, String)>,
    enum_members: Vec<(String, u8)>,
    projection: String,
    input_seed: u64,
}

impl ModuleGraph {
    fn generate(data: &[u8]) -> Self {
        let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
        let mut byte = || bytes.next().unwrap();
        let flags = byte();
        let (selected, other) = if flags & 1 == 0 {
            ("left", "right")
        } else {
            ("right", "left")
        };
        let public_packet = flags & 2 != 0;
        let public_enum = flags & 4 != 0;
        let width = 1 + usize::from(byte()) % 129;
        let length = 1 + usize::from(byte()) % 4;
        let mut sources = BTreeMap::from([(
            "sizes".to_owned(),
            format!("pub fn width() -> u32 {{ u32:{width} }} pub const LENGTH = u32:{length};"),
        )]);

        let count = 1 + usize::from(byte()) % 4;
        let first_name = usize::from(byte()) % count;
        let selected_field = usize::from(byte()) % count;
        let mut projection = String::new();
        let mut field_syntax = Vec::new();
        for i in 0..count {
            let name = format!("field_{}", (i + first_name) % count);
            let sign = if byte() & 1 == 0 { "u" } else { "s" };
            let offset = byte() % 8;
            let bits = format!("{sign}N[dim::width() + u32:{offset}]");
            let (ty, suffix) = match byte() % 3 {
                0 => (bits, String::new()),
                1 => (
                    format!("{bits}[dim::LENGTH]"),
                    format!("[u32:{}]", usize::from(byte()) % length),
                ),
                _ => (format!("(u3, {bits})"), ".1".into()),
            };
            if i == selected_field {
                projection = format!("{name}{suffix}");
            }
            field_syntax.push((name, ty));
        }
        let fields = field_syntax
            .iter()
            .map(|(n, t)| format!("{n}: {t}"))
            .collect::<Vec<_>>()
            .join(", ");
        let enum_base = byte() % 248;
        let enum_members: Vec<_> = (0..1 + byte() % 4)
            .map(|i| {
                let value = enum_base + i;
                let expression = if byte() & 1 == 0 {
                    format!("0x{value:x}")
                } else {
                    format!("u8:{enum_base} + u8:{i}")
                };
                (expression, value)
            })
            .collect();
        let variants = enum_members
            .iter()
            .enumerate()
            .map(|(i, (expr, _))| format!("V{i} = {expr}"))
            .collect::<Vec<_>>()
            .join(", ");
        let packet_prefix = if public_packet { "pub " } else { "" };
        let enum_prefix = if public_enum { "pub " } else { "" };
        let packet_source = format!(
            "import sizes as dim;\n{packet_prefix}struct Packet {{ {fields} }}\npub type Exported = Packet;\n{enum_prefix}enum Kind : u8 {{ {variants} }}"
        );
        sources.insert("left".into(), packet_source.clone());
        sources.insert("right".into(), packet_source);
        let mut upstream = selected.to_owned();
        let mut exported = String::from("Exported");
        for i in 0..byte() % 4 {
            let layer = format!("layer_{i}");
            let alias = format!("Alias{i}");
            let source = if byte() & 1 == 0 {
                format!("import {upstream} as prev; pub type {alias} = prev::{exported};")
            } else {
                format!(
                    "#![feature(type_inference_v1)]\n#![feature(use_syntax)]\nuse {upstream}::{exported};\npub type {alias} = {exported};"
                )
            };
            sources.insert(layer.clone(), source);
            upstream = layer;
            exported = alias;
        }
        let identity = if flags & 8 == 0 {
            "pub fn identity(v: Payload) -> Payload { v }"
        } else {
            "pub fn identity<T: type>(v: T) -> T { v }"
        };
        sources.insert(
            "service".into(),
            format!(
                r#"#![feature(generics)]
import {upstream} as source;
pub type Payload = source::{exported};
pub struct Frame {{ item: Payload }}
pub fn wrap(v: Payload) -> Frame {{ Frame {{ item: v }} }}
pub fn make() -> Frame {{ zero!<Frame>() }}
{identity}"#
            ),
        );
        let input_seed = u64::from_le_bytes(std::array::from_fn(|_| byte()));
        Self {
            sources,
            selected,
            other,
            public_packet,
            public_enum,
            field_syntax,
            enum_members,
            projection,
            input_seed,
        }
    }

    fn run(&self, root: &Path) -> Result<(), Box<dyn Error>> {
        for (name, source) in &self.sources {
            fs::write(root.join(format!("{name}.x")), source)?;
        }
        let paths = [root];
        let options = ImportOptions {
            additional_search_paths: paths.to_vec(),
            ..Default::default()
        };
        let body = format!(
            "let frame = service::wrap(value); let p: service::Payload = service::identity(frame.item); p.{} as u32",
            self.projection
        );
        let params = format!("value: {}::Exported", self.selected);
        let bx = format!(
            "use {}; use service;\n{}",
            self.selected,
            bx_source(&params, "u32", &body)
        );
        let native = format!(
            "import {}; import service;\n{}",
            self.selected,
            dslx_source(&params, "u32", &body)
        );
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &options,
            compile_bundled_module(&native, &paths),
        )?;
        for args in generate_argument_sets_from_seed(&evaluator.bx, self.input_seed, 3) {
            evaluator.compare(&args)?;
        }

        // This block has no nominal port or explicit packet type. BX must
        // recover the public spelling of the inferred field before the call.
        let inferred = format!(
            "service::identity(service::make().item).{} as u32",
            self.projection
        );
        let bx = format!("use service; {}", bx_source("", "u32", &inferred));
        let native = format!("import service; {}", dslx_source("", "u32", &inferred));
        let mut data = ImportData::new(None, &paths);
        dslx::parse_and_typecheck(&native, "oracle.x", "oracle", &mut data)?;
        let checked = check_module_with_options(&parse_module(&bx)?, &options)?;
        self.compare_metadata(&checked, root)?;

        // The positive case above verifies the modules and the shared call.
        // Change just the parameter type to exercise nominal identity and
        // visibility.
        for (subject, member, expected) in [
            (self.other, "Exported", false),
            (self.selected, "Packet", self.public_packet),
        ] {
            let params = format!("value: {subject}::{member}");
            let bx = format!(
                "use {subject}; use service;\n{}",
                bx_source(&params, "u32", &body)
            );
            let native = format!(
                "import {subject}; import service;\n{}",
                dslx_source(&params, "u32", &body)
            );
            let checked = check_module_with_options(&parse_module(&bx)?, &options);
            let mut data = ImportData::new(None, &paths);
            let reference = dslx::parse_and_typecheck(&native, "oracle.x", "oracle", &mut data);
            if reference.is_ok() != expected || checked.is_ok() != reference.is_ok() {
                return Err(format!(
                    "acceptance for {params}: expected {expected}, BX error {:?}, DSLX error {:?}",
                    checked.err(),
                    reference.err()
                )
                .into());
            }
        }
        Ok(())
    }

    /// Compare the Rust outline's declarations and BX's inferred field layout
    /// with the selected origin module's independent post-typecheck metadata.
    fn compare_metadata(&self, checked: &CheckedModule, root: &Path) -> Result<(), Box<dyn Error>> {
        let call = checked
            .imported_calls()
            .iter()
            .find(|c| c.module == "service" && c.function == "identity")
            .ok_or("missing checked identity call")?;
        let [Type::Named(name)] = call.args.as_slice() else {
            return Err("expected a nominal packet argument".into());
        };
        let packet = checked
            .structs()
            .iter()
            .find(|s| &s.name == name)
            .ok_or("missing BX packet layout")?;
        let source = &self.sources[self.selected];
        let outline = parse_dslx_outline(source)?;
        let outlined_packet = outline
            .declarations
            .get("Packet")
            .ok_or("missing outlined Packet")?;
        let OutlineKind::Struct { fields } = &outlined_packet.kind else {
            return Err("Packet kind".into());
        };
        let outlined_enum = outline
            .declarations
            .get("Kind")
            .ok_or("missing outlined Kind")?;
        let OutlineKind::Enum {
            underlying,
            members,
        } = &outlined_enum.kind
        else {
            return Err("Kind kind".into());
        };
        if outlined_packet.public != self.public_packet
            || outlined_enum.public != self.public_enum
            || underlying.as_ref().map(|t| t.syntax.as_str()) != Some("u8")
        {
            return Err("visibility or enum annotation differs from generated module".into());
        }
        let mut data = ImportData::new(None, &[root]);
        let module = dslx::parse_and_typecheck(
            source,
            &format!("{}.x", self.selected),
            self.selected,
            &mut data,
        )?;
        let info = module.get_type_info();
        let ast = module.get_module();
        let native_packet = (0..ast.get_member_count())
            .find_map(|i| match ast.get_member(i).to_matchable() {
                Some(MatchableModuleMember::StructDef(s)) if s.get_identifier() == "Packet" => {
                    Some(s)
                }
                _ => None,
            })
            .ok_or("missing native Packet")?;
        let native_ty = info
            .get_type_for_struct_def(&native_packet)
            .ok_or("missing checked Packet type")?;
        if fields.len() != native_packet.get_member_count()
            || fields.len() != packet.fields.len()
            || fields.len() != self.field_syntax.len()
        {
            return Err("packet field count mismatch".into());
        }
        for (i, (name, ty)) in fields.iter().enumerate() {
            let member = native_packet.get_member(i);
            let actual = &packet.fields[i];
            if *name != member.get_name()
                || *name != actual.name
                || (name, &ty.syntax) != (&self.field_syntax[i].0, &self.field_syntax[i].1)
            {
                return Err(format!("field {i} name, order, or written type mismatch").into());
            }
            compare_type(
                &actual.ty,
                &native_ty.get_struct_member_type(i),
                &member.get_type(),
                &info,
            )?;
        }
        let native_enum = (0..ast.get_member_count())
            .find_map(|i| match ast.get_member(i).to_matchable() {
                Some(MatchableModuleMember::EnumDef(e)) if e.get_identifier() == "Kind" => Some(e),
                _ => None,
            })
            .ok_or("missing native Kind")?;
        if members.len() != native_enum.get_member_count()
            || members.len() != self.enum_members.len()
        {
            return Err("enum member count mismatch".into());
        }
        for (i, (name, rhs)) in members.iter().enumerate() {
            let member = native_enum.get_member(i);
            let value = info
                .get_const_expr(&member.get_value())?
                .convert_to_ir()?
                .to_u64()?;
            let (expression, expected) = &self.enum_members[i];
            if *name != member.get_name() || rhs != expression || value != u64::from(*expected) {
                return Err(format!("enum member {i} name, order, or value mismatch").into());
            }
        }
        Ok(())
    }
}

/// Walk native field metadata alongside the checked BX field type. For tuple
/// elements the pinned API exposes their type through the written annotation.
fn compare_type(
    bx: &Type,
    native: &dslx::Type,
    annotation: &dslx::TypeAnnotation,
    info: &dslx::TypeInfo,
) -> Result<(), Box<dyn Error>> {
    match bx {
        Type::Bits { signed, width } if native.is_bits_like() == Some((*signed, *width)) => Ok(()),
        Type::Array { element, size } if native.is_array() && *size == native.get_array_size() => {
            let item = annotation
                .to_array_type_annotation()
                .ok_or("missing array annotation")?
                .get_element_type();
            compare_type(element, &native.get_array_element_type(), &item, info)
        }
        Type::Tuple(fields) => {
            let members = annotation
                .tuple_members()
                .ok_or("missing native tuple annotation")?;
            if members.len() != fields.len() {
                return Err("tuple arity mismatch".into());
            }
            for (bx, member) in fields.iter().zip(members) {
                let ty = info
                    .get_type_for_type_annotation(&member)
                    .ok_or("missing checked tuple element")?;
                compare_type(bx, &ty, &member, info)?;
            }
            Ok(())
        }
        _ => Err(format!(
            "concrete field type mismatch: BX {bx}, DSLX {}",
            native.to_string()?
        )
        .into()),
    }
}

/// Install a generated module graph, then compare its declarations, legal
/// consumers, illegal consumers, and concrete runtime values.
pub fn check_module_graph(data: &[u8]) -> Result<(), String> {
    let graph = ModuleGraph::generate(data);
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    graph
        .run(dir.path())
        .map_err(|e| format!("{e}\nmodules: {:#?}", graph.sources))
}

#[cfg(test)]
mod tests {
    use super::check_module_graph;

    #[test]
    fn replay_module_graph_acceptance_and_layout_checks() -> Result<(), String> {
        check_module_graph(&[])?;
        check_module_graph(&[u8::MAX; 64])
    }
}
