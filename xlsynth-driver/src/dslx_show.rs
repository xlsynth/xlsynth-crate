// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::common::get_dslx_paths;

fn find_member_by_name(
    module: &xlsynth::dslx::Module,
    name: &str,
) -> Option<xlsynth::dslx::MatchableModuleMember> {
    for i in 0..module.get_member_count() {
        if let Some(m) = module.get_member(i).to_matchable() {
            let is_match = match &m {
                xlsynth::dslx::MatchableModuleMember::EnumDef(e) => e.get_identifier() == name,
                xlsynth::dslx::MatchableModuleMember::StructDef(s) => s.get_identifier() == name,
                xlsynth::dslx::MatchableModuleMember::TypeAlias(t) => t.get_identifier() == name,
                xlsynth::dslx::MatchableModuleMember::ConstantDef(c) => c.get_name() == name,
                xlsynth::dslx::MatchableModuleMember::Function(f) => f.get_identifier() == name,
                xlsynth::dslx::MatchableModuleMember::Quickcheck(qc) => {
                    qc.get_function().get_identifier() == name
                }
            };
            if is_match {
                return Some(m);
            }
        }
    }
    None
}

fn join_path_segments<'a>(segments: impl Iterator<Item = &'a str>) -> String {
    let mut path = std::path::PathBuf::new();
    for s in segments {
        path.push(s);
    }
    path.to_string_lossy().to_string()
}

fn locate_module_file(
    module_segments: &[&str],
    search_roots: &[&std::path::Path],
) -> Option<std::path::PathBuf> {
    if module_segments.is_empty() {
        return None;
    }
    let rel = join_path_segments(module_segments.iter().copied());
    let candidate_rel = std::path::Path::new(&rel).with_extension("x");
    for root in search_roots {
        let candidate = root.join(&candidate_rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn run_show(
    input_path_opt: Option<&std::path::Path>,
    symbol: &str,
    stdlib_path: Option<&std::path::Path>,
    mut search_dirs: Vec<std::path::PathBuf>,
) {
    // Ensure the source file's directory is searched first.
    if let Some(input_path) = input_path_opt {
        if let Some(parent) = input_path.parent() {
            search_dirs.insert(0, parent.to_path_buf());
        }
    }

    // Parse symbol path: module path (optional, dot-separated) + final identifier.
    // We follow DSLX notation for module paths in source files:
    //   import foo.bar.baz;  and references like  baz::Type
    // On the CLI we accept only:
    //   - dotted module path + ::member:  foo.bar.baz::Name
    let (module_path_to_load, module_name, target_ident, module_src_text) = if let Some(idx) =
        symbol.rfind("::")
    {
        let (module_path_str, ident_str) = symbol.split_at(idx);
        // split_at keeps the leading '::' on ident_str; drop it
        let target_ident = ident_str.trim_start_matches("::").to_string();
        // Split module path by '.' only (DSLX-style)
        let module_segments: Vec<&str> = module_path_str
            .split('.')
            .filter(|s| !s.is_empty())
            .collect();

        // Try to locate module file by joining segments under search roots.
        let mut roots: Vec<&std::path::Path> = Vec::new();
        let tmp: Vec<&std::path::Path> = search_dirs.iter().map(|p| p.as_path()).collect();
        roots.extend(tmp);
        if let Some(stdlib) = stdlib_path {
            roots.push(stdlib);
        }
        let module_file = locate_module_file(&module_segments, &roots).unwrap_or_else(|| {
            eprintln!(
                "Error: could not locate module file for '{}' under provided search paths",
                module_path_str
            );
            std::process::exit(1);
        });
        let module_name = module_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module")
            .to_string();
        let text = std::fs::read_to_string(&module_file).unwrap_or_else(|e| {
            panic!(
                "Failed to read DSLX file '{}': {}",
                module_file.display(),
                e
            )
        });
        (module_file, module_name, target_ident, text)
    } else {
        // Unqualified symbol; require an input file context.
        let input_path = input_path_opt.unwrap_or_else(|| {
            eprintln!(
                "Error: --dslx_input_file is required when querying an unqualified symbol '{}'. Use 'path.with.dots::Name' for library symbols.",
                symbol
            );
            std::process::exit(1);
        });
        let module_name = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");
        let text = std::fs::read_to_string(input_path).unwrap_or_else(|e| {
            panic!("Failed to read DSLX file '{}': {}", input_path.display(), e)
        });
        (
            input_path.to_path_buf(),
            module_name.to_string(),
            symbol.to_string(),
            text,
        )
    };

    // Parse/typecheck the chosen module.
    let search_views: Vec<&std::path::Path> = search_dirs.iter().map(|p| p.as_path()).collect();
    let mut import_data = xlsynth::dslx::ImportData::new(stdlib_path, &search_views);
    let tcm = xlsynth::dslx::parse_and_typecheck(
        &module_src_text,
        module_path_to_load.to_str().unwrap(),
        &module_name,
        &mut import_data,
    )
    .unwrap_or_else(|e| panic!("parse_and_typecheck failed: {}", e));

    let module = tcm.get_module();
    match find_member_by_name(&module, &target_ident) {
        Some(m) => print!("{}", m),
        None => {
            eprintln!(
                "Symbol '{}' not found in module '{}'",
                target_ident, module_name
            );
            std::process::exit(1);
        }
    }
}

pub fn handle_dslx_show(
    matches: &ArgMatches,
    config: &Option<crate::toolchain_config::ToolchainConfig>,
) {
    let input_path_opt_buf: Option<std::path::PathBuf> = matches
        .get_one::<String>("dslx_input_file")
        .map(|s| s.into());
    let symbol = matches.get_one::<String>("symbol").unwrap();

    let paths = get_dslx_paths(matches, config);
    let stdlib_opt = paths.stdlib_path.as_ref().map(|p| p.as_path());
    let mut search_dirs: Vec<std::path::PathBuf> = paths.search_paths.clone();
    if let Some(ref input_path) = input_path_opt_buf {
        if let Some(parent) = input_path.parent() {
            search_dirs.insert(0, parent.to_path_buf());
        }
    }
    let input_path_ref = input_path_opt_buf.as_deref();
    run_show(input_path_ref, symbol, stdlib_opt, search_dirs);
}
