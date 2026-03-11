// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

pub use crate::compiled_module::CasezArm;
pub use crate::compiled_module::CasezPattern;
pub use crate::compiled_module::CompiledFunction;
pub use crate::compiled_module::CompiledFunctionBody;
pub use crate::compiled_module::CompiledModule as CompiledComboModule;
pub use crate::compiled_module::FunctionAssign;
pub use crate::compiled_module::ModuleAssign;
pub use crate::compiled_module::Port;
pub use crate::compiled_module::PortDir;
use crate::sv_ast::ModuleItem;
use crate::sv_ast::ParsedModule;
use crate::Result;

pub fn compile_combo_module(src: &str) -> Result<CompiledComboModule> {
    let parse_src = src;
    let parsed: ParsedModule = crate::sv_parser::parse_combo_module(src)?;
    let items =
        crate::generate_constructs::elaborate_combo_items(src, &parsed.params, &parsed.items)?;

    let module_name = parsed.name;
    let (input_ports, output_ports, mut decls) = crate::compiled_module::lower_ports(&parsed.ports);
    let mut functions: BTreeMap<String, CompiledFunction> = BTreeMap::new();
    let mut assigns: Vec<ModuleAssign> = Vec::new();

    crate::compiled_module::extend_decls_from_items(
        &items, &mut decls, /* reject_duplicates= */ false,
    )?;

    for it in &items {
        match it {
            ModuleItem::Decl { .. } => {}
            ModuleItem::Assign {
                lhs,
                rhs,
                rhs_spanned,
                rhs_span,
                ..
            } => {
                assigns.push(crate::compiled_module::lower_assign(
                    lhs,
                    rhs,
                    rhs_spanned,
                    *rhs_span,
                    &decls,
                    true,
                )?);
            }
            ModuleItem::Function { func: f, .. } => {
                functions.insert(
                    f.name.clone(),
                    crate::compiled_module::lower_function(parse_src, f, &decls, true)?,
                );
            }
            ModuleItem::AlwaysFf { .. } => {
                return Err(crate::Error::Parse(
                    "always_ff is not supported in combo modules".to_string(),
                ));
            }
            ModuleItem::GenerateFor { .. } | ModuleItem::GenerateIf { .. } => {
                unreachable!("combo items should be elaborated")
            }
        }
    }

    Ok(CompiledComboModule {
        module_name,
        consts: parsed.params.clone(),
        input_ports,
        output_ports,
        decls,
        assigns,
        functions,
    })
}
