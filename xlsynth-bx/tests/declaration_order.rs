// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs};

use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, check_source_with_options, empty_package,
    lower_module, parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;

const STRICT: &str = "#![require_define_before_use]";

/// Evaluate the combinational sample block with a single u8 input of one.
fn sample_result(package: &Package) -> Result<IrValue, Box<dyn Error>> {
    let function = combinational_block_to_fn(package.get_block("sample").ok_or("missing sample")?)?;
    let input = IrValue::make_ubits(8, 1)?;
    Ok(eval_fn_in_package_value(package, &function, &[input]))
}

/// Check the baseline result, then verify the attribute preserves the emitted
/// package.
fn assert_same_ir(
    source: &str,
    options: &ImportOptions<'_>,
    destination: &Package,
    expected: &IrValue,
) -> Result<(), Box<dyn Error>> {
    let lower = |header: &str| {
        let checked = check_source_with_options(&format!("{header}\n{source}"), options)?;
        lower_module(&checked, destination.clone(), ResetBehavior::default())
    };
    let baseline = lower("")?;
    assert_eq!(&sample_result(&baseline)?, expected, "{source}");
    assert_eq!(baseline.to_string(), lower(STRICT)?.to_string(), "{source}");
    Ok(())
}

#[test]
fn strict_modules_require_earlier_block_dependencies() -> Result<(), String> {
    let helper = "fn step(x: u8) -> u8 { x }";
    let stage = "block step(clk, x: u8) -> u8 { x }";
    let external = "extern fn step(x: u8) -> u8;";
    let external_stage = "extern block step(clk, x: u8) -> u8;";
    let call = "block top(clk, x: u8) -> u8 { step(x) }";
    let child = "block top(clk, x: u8) -> u8 { let q = step(x); q }";
    let instance = "block top(clk, x: u8) -> u8 { inst q: step; q.x := x; q.out }";
    let map = "block top(clk, x: u8[2]) -> u8[2] { map(x, step) }";
    let unused = "block generic<T>(clk, x: T, y: u8) -> T { let q = step(y); x }";
    let record = "struct Packet { data: u8 }";
    for (dependency, user, owner, name) in [
        (helper, call, "top", "step"),
        (external, call, "top", "step"),
        (stage, child, "top", "step"),
        (external_stage, child, "top", "step"),
        (stage, instance, "top", "step"),
        (external_stage, instance, "top", "step"),
        (helper, map, "top", "step"),
        (stage, unused, "generic", "step"),
        (
            "extern fn chip::step(x: u8) -> u8;",
            "block top(clk, chip: u8) -> u8 { chip::step(chip) }",
            "top",
            "chip::step",
        ),
        (
            helper,
            "block top(clk, x: u8) -> u8 { unroll_for! (_, acc) in u32:0..u32:0 { step(acc) }(x) }",
            "top",
            "step",
        ),
        (
            helper,
            "fn apply<F: fn(u8) -> u8>(x: u8) -> u8 { F(x) }\nblock top(clk, x: u8) -> u8 { apply<step>(x) }",
            "top",
            "step",
        ),
        (
            record,
            "struct Envelope { packet: Packet }",
            "Envelope",
            "Packet",
        ),
        (
            record,
            "block top(clk, packet: Packet) -> u8 { packet.data }",
            "top",
            "Packet",
        ),
        (
            record,
            "block top(clk, x: u8) -> u8 { Packet { data: x }.data }",
            "top",
            "Packet",
        ),
    ] {
        let forward = format!("{user}\n{dependency}");
        check_module(&parse_module(&forward)?).map_err(|e| format!("{forward}\n{e}"))?;
        let ordered = format!("{STRICT}\n{dependency}\n{user}");
        check_module(&parse_module(&ordered)?).map_err(|e| format!("{ordered}\n{e}"))?;
        let strict_forward = parse_module(&format!("{STRICT}\n{forward}"))?;
        assert_eq!(
            check_module(&strict_forward).unwrap_err(),
            format!("in '{owner}': '{name}' is declared later in the module"),
            "{forward}",
        );
    }
    Ok(())
}

#[test]
fn strict_functions_require_earlier_helpers_and_types() -> Result<(), String> {
    for (dependency, user, owner, name) in [
        (
            "fn step(x: u8) -> u8 { x }",
            "fn apply(x: u8) -> u8 { step(x) }",
            "apply",
            "step",
        ),
        (
            "extern fn step(x: u8) -> u8;",
            "fn apply(x: u8) -> u8 { step(x) }",
            "apply",
            "step",
        ),
        (
            "struct Packet { x: u8 }",
            "fn get(p: Packet) -> u8 { p.x }",
            "get",
            "Packet",
        ),
        (
            "const BIAS = u8:2;",
            "fn add(x: u8) -> u8 { x + BIAS }",
            "add",
            "BIAS",
        ),
        (
            "enum Tag: u1 { A = 0 }",
            "fn tag() -> Tag { Tag::A }",
            "tag",
            "Tag",
        ),
        (
            "fn step(x: u8) -> u8 { x }",
            "fn apply<T>(x: T, y: u8) -> T { let q = step(y); x }",
            "apply",
            "step",
        ),
    ] {
        let ordered = format!("{STRICT}\n{dependency}\n{user}");
        check_module(&parse_module(&ordered)?).map_err(|e| format!("{ordered}\n{e}"))?;
        let forward = format!("{STRICT}\n{user}\n{dependency}");
        assert_eq!(
            check_module(&parse_module(&forward)?).unwrap_err(),
            format!("in '{owner}': '{name}' is declared later in the module"),
            "{forward}",
        );
    }
    Ok(())
}

#[test]
fn strict_module_preserves_local_bindings_and_deferred_wires() -> Result<(), String> {
    let source = r#"#![require_define_before_use]
fn f(port: u8, xs: u8[2]) -> u8 {
    let (left, right) = (port, xs[0]);
    let row = map(xs, |lane| lane + left);
    right + row[0]
}
block delay<T>(clk, data: T) -> T {
    reg q: T;
    q := data;
    q
}
pub block feedback(clk, x: u8) -> u8 {
    let wire: u8;
    let q = delay(wire);
    wire = f(q, [x, x]);
    q
}
const port = u8:1;
const left = u8:2;
const right = u8:3;
const lane = u8:4;
const wire = u8:5;
"#;
    let checked = check_source_with_options(source, &ImportOptions::default())?;
    lower_module(&checked, empty_package("design"), ResetBehavior::default())?;
    Ok(())
}

#[test]
fn each_bx_file_controls_its_own_declaration_order() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    fs::write(dir.path().join("leaf.bx"), "pub fn id(x: u8) -> u8 { x }")?;
    let library = dir.path().join("stages.bx");
    let top = "pub block top(clk, x: u8) -> u8 { let q = stages::stage(x); q }";
    let root = format!("use stages;\n{top}");
    let library_forward = "pub block stage(clk, x: u8) -> u8 { leaf::id(x) }\nuse leaf;";
    fs::write(&library, library_forward)?;
    check_source_with_options(&format!("{STRICT}\n{root}"), &options)?;

    fs::write(&library, format!("{STRICT}\n{library_forward}"))?;
    assert_eq!(
        check_source_with_options(&root, &options).unwrap_err(),
        format!(
            "BX import 'stages' ({}): in 'stage': 'leaf::id' is declared later in the module",
            library.display()
        ),
    );

    fs::write(
        &library,
        format!("{STRICT}\nuse leaf;\npub block stage(clk, x: u8) -> u8 {{ leaf::id(x) }}"),
    )?;
    check_source_with_options(&format!("{STRICT}\n{root}"), &options)?;
    let root_forward = format!("{top}\nuse stages;");
    check_source_with_options(&root_forward, &options)?;
    assert_eq!(
        check_source_with_options(&format!("{STRICT}\n{root_forward}"), &options).unwrap_err(),
        "in 'top': 'stages::stage' is declared later in the module",
    );
    let direct = top.replace("stages::stage", "stage");
    check_source_with_options(
        &format!("{STRICT}\nuse stages::{{stage}};\n{direct}"),
        &options,
    )?;
    assert_eq!(
        check_source_with_options(
            &format!("{STRICT}\n{direct}\nuse stages::{{stage}};"),
            &options
        )
        .unwrap_err(),
        "in 'top': 'stage' is declared later in the module",
    );
    Ok(())
}

#[test]
fn strict_blocks_reject_later_declarations_that_shadow_builtins() -> Result<(), Box<dyn Error>> {
    for (name, declaration, expression, expected) in [
        ("rev", "fn rev(x: u8) -> u8 { x + u8:1 }", "rev(x)", 2),
        (
            "rev",
            "fn rev(x: u8) -> u8 { x + u8:1 }",
            "map([x], rev)[u32:0]",
            2,
        ),
        ("map", "fn map() -> u8 { u8:3 }", "map()", 3),
        ("zero!", "fn zero!() -> u8 { u8:7 }", "zero!<u8>()", 7),
    ] {
        let source = format!("block sample(clk, x: u8) -> u8 {{ {expression} }}\n{declaration}");
        let options = ImportOptions::default();
        let checked = check_source_with_options(&source, &options)?;
        let package = lower_module(&checked, empty_package("design"), ResetBehavior::default())?;
        assert_eq!(
            sample_result(&package)?,
            IrValue::make_ubits(8, expected)?,
            "{expression}",
        );
        assert_eq!(
            check_source_with_options(&format!("{STRICT}\n{source}"), &options).unwrap_err(),
            format!("in 'sample': '{name}' is declared later in the module"),
            "{expression}",
        );
    }
    Ok(())
}

#[test]
fn verification_preserves_builtin_and_visible_local_calls() -> Result<(), Box<dyn Error>> {
    let source = r#"fn before(x: u8) -> (u8, u8, u8) {
    (rev(x), map([x], rev)[u32:0], zero!<u8>())
}
fn rev(x: u8) -> u8 { x + u8:1 }
fn map() -> u8 { u8:3 }
fn zero!() -> u8 { u8:7 }
block sample(clk, x: u8) -> ((u8, u8, u8), u8, u8, u8, u8) {
    (before(x), rev(x), map(), zero!<u8>(), clz(x))
}
"#;
    assert_same_ir(
        source,
        &ImportOptions::default(),
        &empty_package("design"),
        &IrValue::parse_typed(
            "((bits[8]:128, bits[8]:128, bits[8]:0), bits[8]:2, bits[8]:3, bits[8]:7, bits[8]:7)",
        )?,
    )
}

#[test]
fn declaration_type_arguments_keep_function_scope() -> Result<(), Box<dyn Error>> {
    let signature = r#"struct Tag<N: u32> { data: u8 }
struct Envelope { tag: Tag<{rev(u32:0)}> }
block pass(clk, p: Tag<{zero!<u32>()}>, q: Tag<0>) -> (Tag<0>, Tag<{rev(u32:0)}>) { (p, q) }"#;
    let functions = "fn rev(x: u32) -> u32 { u32:7 } fn zero!() -> u32 { u32:7 }";
    let source = format!("{signature}\n{functions}\nblock sample(clk, x: u8) -> u8 {{ x }}");
    assert_same_ir(
        &source,
        &ImportOptions::default(),
        &empty_package("design"),
        &IrValue::make_ubits(8, 1)?,
    )?;

    let externs = r#"extern fn read(p: Tag<{rev(u32:0)}>) -> u8;
extern block pipe(clk, p: Tag<{rev(u32:0)}>) -> u8;"#;
    check_module(&parse_module(&format!(
        "{STRICT}\n{signature}\n{externs}\n{functions}"
    ))?)?;

    for (name, expr) in [("rev", "rev(u32:0)"), ("zero!", "zero!<u32>()")] {
        // The same actual in a block-body annotation calls the later function.
        let source = format!(
            "struct Tag<N:u32> {{ data:u8 }}\nblock sample(clk, x:u8) -> u8 {{ let p: Tag<{{{expr}}}> = Tag<7> {{ data:x }}; p.data }}\n{functions}"
        );
        check_module(&parse_module(&source)?)?;
        assert_eq!(
            check_module(&parse_module(&format!("{STRICT}\n{source}"))?).unwrap_err(),
            format!("in 'sample': '{name}' is declared later in the module"),
        );
    }
    Ok(())
}

#[test]
fn declaration_type_arguments_use_the_resolved_import() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    let definitions = "pub fn rev(x:u32) -> u32 { u32:7 } pub fn zero!() -> u32 { u32:7 }";
    fs::write(dir.path().join("native.x"), definitions)?;
    fs::write(dir.path().join("library.bx"), definitions)?;
    for (name, expr) in [("rev", "rev(u32:0)"), ("zero!", "zero!<u32>()")] {
        for (library, value) in [("native", 0), ("library", 7)] {
            let source = format!(
                "struct Tag<N:u32> {{ data:u8 }}\nblock sample(clk, p: Tag<{{{expr}}}>) -> Tag<{value}> {{ p }}\nuse {library}::{name};"
            );
            check_source_with_options(&source, &options)?;
            let verified = check_source_with_options(&format!("{STRICT}\n{source}"), &options);
            if library == "native" {
                verified?;
            } else {
                assert_eq!(
                    verified.unwrap_err(),
                    format!("in 'sample': '{name}' is declared later in the module")
                );
            }
        }
    }
    Ok(())
}

#[test]
fn verification_uses_the_kind_of_the_imported_member() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    for (path, source) in [
        (
            "enums.x",
            "pub enum rev : u8 { A = 0 } pub enum map : u8 { A = 0 }",
        ),
        ("structs.x", "pub struct rev { value: u8 }"),
        ("aliases.x", "pub type rev = u8;"),
        (
            "funcs.x",
            "pub fn rev(x: u8) -> u8 { x + u8:1 } pub fn map(xs: u8[1], x: u8) -> u8 { xs[u32:0] + x } pub fn zero!() -> u8 { u8:7 }",
        ),
        ("bx_funcs.bx", "pub fn zero!() -> u8 { u8:7 }"),
    ] {
        fs::write(dir.path().join(path), source)?;
    }
    let both_calls = "block sample(clk, x: u8) -> (u8, u8) { (rev(x), map([x], rev)[u32:0]) }";
    let package = empty_package("design");
    for library in ["enums", "structs", "aliases"] {
        assert_same_ir(
            &format!("{both_calls}\nuse {library}::rev;"),
            &options,
            &package,
            &IrValue::parse_typed("(bits[8]:128, bits[8]:128)")?,
        )?;
    }
    assert_same_ir(
        "fn inc(x: u8) -> u8 { x + u8:1 }\nblock sample(clk, x: u8) -> u8 { map([x], inc)[u32:0] }\nuse enums::map;",
        &options,
        &package,
        &IrValue::make_ubits(8, 2)?,
    )?;
    assert_same_ir(
        "block sample(clk, x: u8) -> u8 { zero!<u8>() }\nuse funcs::zero!;",
        &options,
        &package,
        &IrValue::make_ubits(8, 0)?,
    )?;
    for (expression, library, name, expected) in [
        ("rev(x)", "funcs", "rev", 2),
        ("map([x], rev)[u32:0]", "funcs", "rev", 2),
        ("map([x], x)", "funcs", "map", 2),
        ("zero!<u8>()", "bx_funcs", "zero!", 7),
    ] {
        let source =
            format!("block sample(clk, x: u8) -> u8 {{ {expression} }}\nuse {library}::{name};");
        let checked = check_source_with_options(&source, &options)?;
        let baseline = lower_module(&checked, package.clone(), ResetBehavior::default())?;
        assert_eq!(
            sample_result(&baseline)?,
            IrValue::make_ubits(8, expected)?,
            "{source}"
        );
        assert_eq!(
            check_source_with_options(&format!("{STRICT}\n{source}"), &options).unwrap_err(),
            format!("in 'sample': '{name}' is declared later in the module"),
            "{source}",
        );
    }
    Ok(())
}

#[test]
fn typed_zero_in_a_block_ignores_a_later_extern_in_the_ast() -> Result<(), Box<dyn Error>> {
    let destination = Parser::new(
        r#"package design
fn external_zero() -> bits[8] { ret result: bits[8] = literal(value=7, id=1) }
"#,
    )
    .parse_and_verify_package()?;
    let mut module = parse_module(
        r#"block sample(clk, x: u8) -> u8 { zero!<u8>() }
extern fn stub() -> u8 = "external_zero";"#,
    )?;
    // The source grammar reserves the bang spelling for local functions.
    // AST clients can supply an extern with that name and a legal IR symbol.
    module.externs[0].name = "zero!".into();
    module.declaration_order[1] = "zero!".into();
    let checked = check_module(&module)?;
    let baseline = lower_module(&checked, destination.clone(), ResetBehavior::default())?;
    assert_eq!(sample_result(&baseline)?, IrValue::make_ubits(8, 0)?);
    module.require_define_before_use = true;
    let checked = check_module(&module)?;
    let verified = lower_module(&checked, destination, ResetBehavior::default())?;
    assert_eq!(baseline.to_string(), verified.to_string());
    Ok(())
}

#[test]
fn a_block_map_lambda_calls_a_zero_extern_in_its_body() -> Result<(), Box<dyn Error>> {
    let destination = Parser::new(
        r#"package design
fn external_zero() -> bits[8] { ret result: bits[8] = literal(value=7, id=1) }
"#,
    )
    .parse_and_verify_package()?;
    let block = "block sample(clk, x: u8) -> u8 { map([zero!<u8>()], |v| v + zero!<u8>())[u32:0] }";
    let external = r#"extern fn stub() -> u8 = "external_zero";"#;
    for (source, extern_position) in [
        (format!("{external}\n{block}"), 0),
        (format!("{block}\n{external}"), 1),
    ] {
        let mut module = parse_module(&source)?;
        module.externs[0].name = "zero!".into();
        module.declaration_order[extern_position] = "zero!".into();
        let baseline = lower_module(
            &check_module(&module)?,
            destination.clone(),
            ResetBehavior::default(),
        )?;
        // The iterable uses the intrinsic zero, while the lambda body invokes
        // the extern.
        assert_eq!(sample_result(&baseline)?, IrValue::make_ubits(8, 7)?);
        module.require_define_before_use = true;
        if extern_position == 0 {
            let verified = lower_module(
                &check_module(&module)?,
                destination.clone(),
                ResetBehavior::default(),
            )?;
            assert_eq!(baseline.to_string(), verified.to_string());
        } else {
            assert_eq!(
                check_module(&module).unwrap_err(),
                "in 'sample': 'zero!' is declared later in the module"
            );
        }
    }
    Ok(())
}

#[test]
fn an_unused_block_map_lambda_verifies_its_native_zero_import() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    fs::write(dir.path().join("native.x"), "pub fn zero!() -> u8 { u8:7 }")?;
    let template = "pub block lazy<T>(clk, x:T) -> T { let y = map([u8:0], |v| zero!<u8>()); x }";
    let import = "use native::zero!;";
    let library = dir.path().join("stages.bx");
    for source in [
        format!("{template}\n{import}"),
        format!("{import}\n{template}"),
    ] {
        check_source_with_options(&source, &options)?;
        fs::write(&library, &source)?;
        check_source_with_options("use stages;", &options)?;
        let marked = format!("{STRICT}\n{source}");
        let root_result = check_source_with_options(&marked, &options);
        fs::write(&library, marked)?;
        let library_result = check_source_with_options("use stages;", &options);
        if source.starts_with(import) {
            root_result?;
            library_result?;
        } else {
            let expected = "in 'lazy': 'zero!' is declared later in the module";
            assert_eq!(root_result.unwrap_err(), expected);
            assert_eq!(
                library_result.unwrap_err(),
                format!("BX import 'stages' ({}): {expected}", library.display())
            );
        }
    }
    Ok(())
}

#[test]
fn strict_ast_requires_the_order_of_every_declaration() -> Result<(), String> {
    let module = parse_module(&format!("{STRICT}\nfn f() {{}}\nfn g() {{}}"))?;
    for order in [vec![], vec!["f", "f"], vec!["f", "unknown"]] {
        let mut incomplete = module.clone();
        incomplete.declaration_order = order.into_iter().map(str::to_owned).collect();
        assert_eq!(
            check_module(&incomplete).unwrap_err(),
            "require_define_before_use needs a declaration_order containing each declaration and use binding once",
        );
    }
    Ok(())
}

#[test]
fn an_enum_member_and_a_qualified_extern_have_distinct_dependencies() -> Result<(), String> {
    let tag = "enum Tag : u1 { A = 0 }";
    let external = "extern fn Tag::A(x: u8) -> u8;";
    let value = "block sample(clk) -> Tag { Tag::A }";
    let call = "block sample(clk, x: u8) -> u8 { Tag::A(x) }";
    check_module(&parse_module(&format!(
        "{STRICT}\n{tag}\n{value}\n{external}"
    ))?)?;
    check_module(&parse_module(&format!(
        "{STRICT}\n{external}\n{call}\n{tag}"
    ))?)?;
    assert_eq!(
        check_module(&parse_module(&format!(
            "{STRICT}\n{tag}\n{call}\n{external}"
        ))?)
        .unwrap_err(),
        "in 'sample': 'Tag::A' is declared later in the module"
    );
    Ok(())
}

#[test]
fn verification_preserves_deferred_template_checks_and_visits_the_remaining_body()
-> Result<(), String> {
    let template = "block unused<T>(clk, a: T, x: u8) -> T { let q = x::foo(); a }";
    let sample = "block sample(clk, x: u8) -> u8 { x }";
    for header in ["", STRICT] {
        check_module(&parse_module(&format!("{header}\n{template}\n{sample}"))?)?;
    }
    let forward = template.replace("a }", "let later = step(x); a }");
    let source = format!("{STRICT}\n{forward}\nfn step(x: u8) -> u8 {{ x }}\n{sample}");
    assert_eq!(
        check_module(&parse_module(&source)?).unwrap_err(),
        "in 'unused': 'step' is declared later in the module"
    );
    Ok(())
}
