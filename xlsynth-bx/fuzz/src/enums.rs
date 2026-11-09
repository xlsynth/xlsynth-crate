// SPDX-License-Identifier: Apache-2.0

//! Compare nominal numeric enum types, their discriminants and lowered values.

use crate::eval::expr_eval::{BlockEvaluator, bx_source, compile_bundled_module, dslx_source};
use crate::function_oracle::{
    DslxExecution, FunctionProgram, check_sources, compare_bx_program, compare_program,
};
use crate::input_seed;
use std::{fmt::Write, fs};
use tempfile::tempdir;
use xlsynth::dslx::{ImportData, MatchableModuleMember, parse_and_typecheck};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile_with_options, empty_package, parse_module,
};
use xlsynth_pir::libxls_bridge::value_to_libxls;
use xlsynth_pir::{IrBits, IrFormatPreference, IrValue};

/// A finite enum domain. Aliases share the bit pattern of an existing member.
struct Domain {
    width: usize,
    signed: bool,
    ty: String,
    values: Vec<IrBits>,
    alias: bool,
}

impl Domain {
    fn generate(flags: u8, width_selector: u8, count_selector: u8, salt: u8) -> Self {
        let boolean = flags & 4 != 0;
        let width = if boolean {
            1
        } else {
            [0, 1, 2, 3, 5, 8, 17, 32, 64, 65, 97, 129][usize::from(width_selector % 12)]
        };
        let signed = !boolean && width != 0 && flags & 1 != 0;
        let ty = if boolean {
            "bool".to_string()
        } else {
            format!("{}N[{width}]", if signed { 's' } else { 'u' })
        };
        let count = (1 + usize::from(count_selector % 4)).min(1 << width.min(2));
        let values = (0..count)
            .map(|i| {
                IrBits::from_lsb_fn(width, |bit| {
                    // The low two bits keep different generated members
                    // distinct.
                    let mask = if bit < 2 {
                        i as u8
                    } else {
                        (i as u8).wrapping_mul(91)
                    };
                    (salt ^ mask).rotate_right((bit % 8) as u32) & 1 != 0
                })
            })
            .collect();
        Self {
            width,
            signed,
            ty,
            values,
            alias: flags & 2 != 0,
        }
    }

    fn decimal(&self, bits: &IrBits) -> String {
        let format = if self.signed {
            IrFormatPreference::SignedDecimal
        } else {
            IrFormatPreference::UnsignedDecimal
        };
        bits.to_string_fmt(format, false)
    }

    fn literal(&self, bits: &IrBits) -> String {
        if self.ty == "bool" {
            (!bits.is_zero()).to_string()
        } else {
            format!(
                "{}:{}",
                self.ty,
                bits.to_string_fmt(IrFormatPreference::Hex, false)
            )
        }
    }

    /// A typed anchor fixes the representation even when the backing type is
    /// omitted.
    fn declarations(&self, inferred: bool, style: u8) -> String {
        let mut source = format!("fn keep(x: {0}) -> {0} {{ x }}\n", self.ty);
        let mut members = Vec::new();
        let anchor = usize::from(style) % self.values.len();
        for (i, bits) in self.values.iter().enumerate() {
            let literal = self.literal(bits);
            let expr = match (usize::from(style) + i) % 4 {
                0 if !inferred || i != anchor => self.decimal(bits),
                1 => {
                    writeln!(source, "const VALUE_{i}: {} = {literal};", self.ty).unwrap();
                    format!("VALUE_{i}")
                }
                2 => format!("keep({literal})"),
                _ => literal,
            };
            members.push(format!("M{i} = {expr}"));
        }
        if self.alias {
            members.push(format!("Alias = {}", self.literal(&self.values[0])));
        }
        let base = if inferred {
            String::new()
        } else {
            format!(": {}", self.ty)
        };
        writeln!(source, "enum Mode{base} {{ {}, }}", members.join(", ")).unwrap();
        source
    }

    /// Infer the signedness and minimum width entirely from bare literals.
    fn unsized_declaration(&self) -> String {
        let mut members = self
            .values
            .iter()
            .enumerate()
            .map(|(i, bits)| format!("M{i} = {}", self.decimal(bits)))
            .collect::<Vec<_>>();
        if self.alias {
            members.push(format!("Alias = {}", self.decimal(&self.values[0])));
        }
        format!("enum Mode {{ {} }}", members.join(", "))
    }
}

/// Use a complete small domain for enum-typed hardware ports. Every random
/// bit pattern supplied by the shared input generator is then a valid member.
fn compare_ports(
    flags: u8,
    width_selector: u8,
    salt: u8,
    style: u8,
    seed: u64,
) -> Result<(), String> {
    let width = if flags & 4 != 0 {
        1
    } else {
        width_selector % 3
    };
    let domain = Domain::generate(flags, width, (1 << width) - 1, salt);
    let functions = domain.declarations(flags & 64 != 0, style);
    let body = r#"let found = xs[u32:0] == pivot;
        let v = if found { xs[u32:1] } else { zero!<Mode>() };
        (found, v, [v, pivot])"#;
    compare_program(
        FunctionProgram {
            functions,
            params: "xs:Mode[2], pivot:Mode".into(),
            result: "(bool, Mode, Mode[2])".into(),
            body: body.into(),
            valid: Some(true),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

/// Pass enum values across BX and DSLX module boundaries, comparing both
/// import spellings against the native DSLX library.
fn compare_import(
    domain: &Domain,
    mut library: String,
    flags: u8,
    seed: u64,
) -> Result<(), String> {
    library = library.replace("enum Mode", "pub enum Mode");
    let last = domain.values.len() - 1;
    writeln!(
        library,
        "pub fn choose(first:bool)->Mode {{ if first {{ Mode::M0 }} else {{ Mode::M{last} }} }}"
    )
    .unwrap();
    writeln!(
        library,
        "pub fn encode(x:Mode)->{0} {{ x as {0} }}",
        domain.ty
    )
    .unwrap();
    let (read, imports) = if flags & 16 != 0 {
        library.push_str("\npub struct Packet { choices: Mode[2] }\n");
        library.push_str("pub fn pack(x:Mode)->Packet { Packet { choices: [x, Mode::M0] } }");
        (
            "device::pack(device::choose(choose_first)).choices[u32:0]",
            "Mode, choose, encode, pack",
        )
    } else {
        ("device::choose(choose_first)", "Mode, choose, encode")
    };
    let dir = tempdir().map_err(|e| e.to_string())?;
    fs::write(dir.path().join("device.x"), &library).map_err(|e| e.to_string())?;
    let member = if domain.alias && flags & 8 != 0 {
        "Alias"
    } else {
        "M0"
    };
    let params = "choose_first:bool";
    let result = format!("({},bool)", domain.ty);
    let body =
        format!("let x:device::Mode = {read}; (device::encode(x), x == device::Mode::{member})");
    let bx_module = if flags & 32 != 0 {
        fs::write(dir.path().join("bxlib.bx"), &library).map_err(|e| e.to_string())?;
        "bxlib"
    } else {
        "device"
    };
    let (imports, bx_body) = if flags & 128 != 0 {
        (
            format!("use {bx_module}::{{{imports}}};"),
            body.replace("device::", ""),
        )
    } else {
        (
            format!("use {bx_module};"),
            body.replace("device::", &format!("{bx_module}::")),
        )
    };
    let bx = format!("{imports}\n{}", bx_source(params, &result, &bx_body));
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: "import device;".into(),
            params: params.into(),
            result,
            body,
            valid: Some(true),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        vec![vec![IrValue::bool(true)], vec![IrValue::bool(false)]],
        &[dir.path()],
    )
}

/// Place the same comparison on different DSLX call paths in the imported
/// module.
fn ordering_function(name: &str, comparison: &str, shape: u8) -> String {
    let params = "a:Mode,b:Mode";
    match shape % 4 {
        0 => format!("pub fn {name}({params})->bool {{ {comparison} }}"),
        1 => format!(
            "fn {name}_pair(p:(Mode,Mode))->bool {{ let (a,b)=p; {comparison} }}
            pub fn {name}({params})->bool {{ map([(a,b)], {name}_pair)[u32:0] }}"
        ),
        2 => format!(
            "pub fn {name}({params})->bool {{
            map([a], |a:Mode| -> bool {{ {comparison} }})[u32:0]
        }}"
        ),
        _ => format!(
            "fn {name}_leaf({params})->bool {{ {comparison} }}
            pub fn {name}({params})->bool {{
                for (_, carry) in u32:0..u32:1 {{ carry || {name}_leaf(a,b) }}(false)
            }}"
        ),
    }
}

/// Call one generic DSLX function on two distinct declarations named Mode.
/// The native oracle compares the corresponding members directly, so it does
/// not depend on the converter giving the two specializations distinct names.
fn compare_specializations(
    first: &Domain,
    second: &Domain,
    flags: u8,
    style: u8,
    seed: u64,
) -> Result<(), String> {
    let dir = tempdir().map_err(|e| e.to_string())?;
    fs::write(
        dir.path().join("ops.x"),
        "#![feature(generics)]\npub fn classify<T: type>(x:T)->bool { x == T::M0 }",
    )
    .map_err(|e| e.to_string())?;
    let mut setup = String::new();
    let mut calls = Vec::new();
    let mut expected = Vec::new();
    for (module, domain) in [("left", first), ("right", second)] {
        let mut declarations = domain
            .declarations(flags & 64 != 0, style)
            .replace("enum Mode", "pub enum Mode");
        let ty = if flags & 2 != 0 && module == "right" {
            declarations.push_str("pub type Alias = Mode;\n");
            "Alias"
        } else {
            "Mode"
        };
        let last = domain.values.len() - 1;
        writeln!(
            setup,
            "let {module}_mode = if {module}_first {{ {module}::{ty}::M0 }} else {{ {module}::{ty}::M{last} }};"
        )
        .unwrap();
        fs::write(dir.path().join(format!("{module}.x")), &declarations)
            .map_err(|e| e.to_string())?;
        calls.push(format!("ops::classify({module}_mode)"));
        expected.push(format!("{module}_mode == {module}::{ty}::M0"));
    }
    if flags & 8 != 0 {
        calls.reverse();
        expected.reverse();
    }
    let params = "left_first:bool, right_first:bool";
    let result = "(bool,bool)";
    let bx = format!(
        "use left; use right; use ops;\n{}",
        bx_source(params, result, &format!("{setup} ({})", calls.join(", ")))
    );
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: "import left; import right;".into(),
            params: params.into(),
            result: result.into(),
            body: format!("{setup} ({})", expected.join(", ")),
            valid: Some(true),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        vec![
            vec![IrValue::bool(true), IrValue::bool(true)],
            vec![IrValue::bool(false), IrValue::bool(false)],
        ],
        &[dir.path()],
    )
}

/// Pinned DSLX typechecks enum ordering but its bytecode interpreter rejects
/// enum operands and its IR converter aborts. BX diagnoses the operation.
fn check_ordering(
    domain: &Domain,
    source: &str,
    selector: u8,
    shape: u8,
    seed: u64,
) -> Result<(), String> {
    let op = ["<", "<=", ">", ">="][usize::from(selector % 4)];
    let body = format!("a {op} b");
    let params = "a:Mode,b:Mode";
    let bx = format!("{source}\n{}", bx_source(params, "bool", &body));
    let reference = format!("{source}\n{}", dslx_source(params, "bool", &body));
    parse_and_typecheck(&reference, "oracle.x", "oracle", &mut ImportData::default())
        .map_err(|e| format!("DSLX rejected enum ordering: {e}\n{reference}"))?;
    let module = parse_module(&bx)?;
    match check_module(&module) {
        Err(e) if e.contains("ordered comparison of enum values") => { /* Expected local diagnostic. */
        }
        Err(e) => return Err(format!("unexpected BX enum ordering diagnostic: {e}\n{bx}")),
        Ok(_) => return Err(format!("BX accepted unsupported enum ordering\n{bx}")),
    }

    let safe_comparison = format!("(a as {0}) {op} (b as {0})", domain.ty);
    let safe = ordering_function("bit_order", &safe_comparison, shape);
    let bad = ordering_function("enum_order", &body, shape);
    let library = format!(
        "{}\n{bad}\n{safe}",
        source.replace("enum Mode", "pub enum Mode")
    );
    let dir = tempdir().map_err(|e| e.to_string())?;
    fs::write(dir.path().join("device.x"), &library).map_err(|e| e.to_string())?;
    let params = "first:bool, second:bool";
    let last = domain.values.len() - 1;
    let inputs = format!(
        "let a = if first {{ Mode::M0 }} else {{ Mode::M{last} }};
        let b = if second {{ Mode::M0 }} else {{ Mode::M{last} }};"
    );
    let imported = |callee| {
        let body = format!(
            "{} device::{callee}(a,b)",
            inputs.replace("Mode::", "device::Mode::")
        );
        format!("use device;\n{}", bx_source(params, "bool", &body))
    };
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    let bad = imported("enum_order");
    match compile_with_options(
        &bad,
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    ) {
        Err(e) if e.contains("ordered comparison") && e.contains("cast enum operands") => { /* Expected import diagnostic. */
        }
        Err(e) => {
            return Err(format!(
                "unexpected imported enum ordering diagnostic: {e}\n{bad}\n{library}"
            ));
        }
        Ok(_) => {
            return Err(format!(
                "BX accepted imported enum ordering\n{bad}\n{library}"
            ));
        }
    }
    // The independent native IR/bytecode reference contains only the safe
    // sibling. BX must link the safe function from the full imported module.
    compare_bx_program(
        &imported("bit_order"),
        FunctionProgram {
            functions: format!("{source}\n{safe}"),
            params: params.into(),
            result: "bool".into(),
            body: format!("{inputs} bit_order(a,b)"),
            valid: Some(true),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        vec![
            vec![IrValue::bool(true), IrValue::bool(false)],
            vec![IrValue::bool(false), IrValue::bool(true)],
        ],
        &[dir.path()],
    )
}

/// Choose valid discriminants at runtime, exercise enum ports on pure helpers,
/// and compare both backends on every member and on independently sampled
/// inputs.
fn compare_members(
    domain: &Domain,
    mut source: String,
    flags: u8,
    seed: u64,
) -> Result<(), String> {
    let mut picks = Vec::new();
    let mut patterns = Vec::new();
    for i in 0..domain.values.len() {
        let member = if i == 0 && domain.alias {
            "Alias".into()
        } else {
            format!("M{i}")
        };
        if i + 1 == domain.values.len() {
            picks.push(format!("_ => Mode::{member}"));
        } else {
            picks.push(format!("u3:{i} => Mode::{member}"));
        }
        let pat = if flags & 8 != 0 && i + 1 == domain.values.len() {
            "_".to_string()
        } else {
            format!("Mode::{member}")
        };
        patterns.push(pat);
    }
    if flags & 128 != 0 && patterns.get(1).is_some_and(|pat| pat != "_") {
        let second = patterns.remove(1);
        patterns[0].push_str(" | ");
        patterns[0].push_str(&second);
    }
    let labels = patterns
        .iter()
        .enumerate()
        .map(|(i, pat)| format!("{pat} => u8:{}", i * 13 + 7))
        .collect::<Vec<_>>();
    writeln!(
        source,
        "fn select_mode(k: u3) -> Mode {{ match k {{ {} }} }}",
        picks.join(", ")
    )
    .unwrap();
    writeln!(
        source,
        "fn label(x: Mode) -> u8 {{ match x {{ {} }} }}",
        labels.join(", ")
    )
    .unwrap();
    writeln!(source, "const FIRST = Mode::M0;").unwrap();
    writeln!(source, "struct Envelope {{ mode: Mode, pair: Mode[2] }}").unwrap();
    let out_width = if flags & 16 == 0 {
        domain.width + 3
    } else {
        domain.width.saturating_sub(1).max(1)
    };
    let out = format!("{}N[{out_width}]", if flags & 32 == 0 { 'u' } else { 's' });
    let result = format!("(Mode, {out}, bool, bool, u8, Mode[2])");
    writeln!(
        source,
        "fn measure(x: Mode, y: Mode) -> {result} {{
        let p = Envelope {{ mode: x, pair: [y, FIRST] }};
        (p.mode, p.mode as {out}, x == y, x != FIRST, label(x), p.pair)
    }}"
    )
    .unwrap();
    let inputs = (0..domain.values.len())
        .map(|i| {
            Ok(vec![
                IrValue::make_ubits(3, i as u64)?,
                IrValue::make_ubits(3, ((i + 1) % domain.values.len()) as u64)?,
            ])
        })
        .collect::<Result<Vec<_>, xlsynth_pir::ValueError>>()
        .map_err(|e| e.to_string())?;
    compare_program(
        FunctionProgram {
            functions: source,
            params: "a:u3, b:u3".into(),
            result,
            body: "measure(select_mode(a), select_mode(b))".into(),
            valid: Some(true),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        inputs,
    )
}

/// Keep invalid declarations and expressions separate from the value generator.
fn compare_rejection(
    domain: &Domain,
    mut source: String,
    mutation: u8,
    seed: u64,
) -> Result<(), String> {
    let ty = &domain.ty;
    let lit = domain.literal(&domain.values[0]);
    writeln!(
        source,
        "enum Other: {ty} {{ First = {lit}, Alias = {lit} }}"
    )
    .unwrap();
    let body = match mutation % 16 {
        0 => {
            writeln!(source, "enum Bad:{ty} {{ A={lit}, A={lit} }}").unwrap();
            "true"
        }
        1 => {
            writeln!(source, "enum Bad:{ty} {{ A={lit}, B }}").unwrap();
            "true"
        }
        2 => {
            writeln!(source, "enum Bad:{ty} {{ A=uN[{}]:0 }}", domain.width + 1).unwrap();
            "true"
        }
        3 => {
            writeln!(source, "enum Bad:{ty} {{ A=Bad::A as {ty} }}").unwrap();
            "true"
        }
        4 => "Mode::Missing == Mode::M0",
        5 => "Mode::M0 == Other::First",
        6 => "Mode::M0 == 0",
        7 => "(Mode::M0 as Other) == Other::First",
        8 => "(Mode::M0 + Mode::M0) == Mode::M0",
        9 => "(!Mode::M0) == Mode::M0",
        10 => "match Mode::M0 { 0 => true, _ => false }",
        11 => "match Other::First { Other::Alias => true, Other::First => false }",
        12 => {
            writeln!(source, "fn wrong() -> Mode {{ {lit} }}").unwrap();
            "true"
        }
        13 => "Mode:0 == Mode::M0",
        14 => {
            let arms = (1..domain.values.len())
                .map(|i| format!("Mode::M{i} => true"))
                .collect::<Vec<_>>();
            writeln!(
                source,
                "fn partial(x:Mode)->bool {{ match x {{ {} }} }}",
                arms.join(", ")
            )
            .unwrap();
            "partial(Mode::M0)"
        }
        _ => {
            let other_ty = format!(
                "{}N[{}]",
                if domain.signed { 'u' } else { 's' },
                domain.width
            );
            writeln!(source, "enum Bad:{ty} {{ A={other_ty}:0 }}").unwrap();
            "true"
        }
    };
    compare_program(
        FunctionProgram {
            functions: source,
            params: String::new(),
            result: "bool".into(),
            body: body.into(),
            valid: Some(false),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

/// DSLX IR casts resize bits without testing membership. DSLX bytecode
/// requires the original width and a declared value, and rejects enum identity
/// casts.
fn compare_cast(
    domain: &Domain,
    source: String,
    flags: u8,
    width_selector: u8,
) -> Result<(), String> {
    let input_width = match width_selector % 3 {
        1 => domain.width + 2,
        2 => domain.width.saturating_sub(1),
        _ => domain.width,
    };
    let input = format!(
        "{}N[{input_width}]",
        if flags & 16 != 0 && input_width > 0 {
            's'
        } else {
            'u'
        }
    );
    let params = format!("x:{input}");
    let identity = flags & 8 != 0;
    let subject = if identity { "(v as Mode)" } else { "v" };
    let result = if flags & 32 != 0 {
        let mut arms = (0..domain.values.len())
            .map(|i| {
                let next = &domain.values[(i + 1) % domain.values.len()];
                format!("Mode::M{i} => {}", domain.literal(next))
            })
            .collect::<Vec<_>>();
        if flags & 128 != 0 {
            arms.push(format!(
                "_ => {}",
                domain.literal(&IrBits::zero(domain.width))
            ));
        }
        format!("match {subject} {{ {} }}", arms.join(", "))
    } else {
        format!("{subject} as {}", domain.ty)
    };
    let body = format!("let v = x as Mode; {result}");
    let reference = format!("{source}\n{}", dslx_source(&params, &domain.ty, &body));
    let bx = format!("{source}\n{}", bx_source(&params, &domain.ty, &body));
    let checked =
        check_sources(&bx, &reference, Some(true), &[])?.expect("generated cast must typecheck");
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    let native_ir = compile_bundled_module(&reference, &[])?;
    let resized = IrBits::from_lsb_fn(input_width, |i| {
        i < domain.width && domain.values[0].get_bit(i).unwrap()
    });
    for bits in [
        resized,
        IrBits::zero(input_width),
        IrBits::all_ones(input_width),
    ] {
        let actual = evaluator.compare_ir(&native_ir, &[IrValue::Bits(bits.clone())])?;
        let argument = format!(
            "{input}:{}",
            bits.to_string_fmt(IrFormatPreference::Hex, false)
        );
        let constant = format!("{reference}\nconst OBS = sample({argument});");
        let native =
            parse_and_typecheck(&constant, "oracle.x", "oracle", &mut ImportData::default())
                .map_err(|e| format!("{e}\n{constant}"))?;
        let module = native.get_module();
        let Some(MatchableModuleMember::ConstantDef(observed)) = module
            .get_member(module.get_member_count() - 1)
            .to_matchable()
        else {
            return Err("missing DSLX cast observation".into());
        };
        let interpreted = native
            .get_type_info()
            .get_const_expr(&observed.get_value())
            .and_then(|v| v.convert_to_ir());
        let valid = !identity && input_width == domain.width && domain.values.contains(&bits);
        if valid {
            let value = interpreted.map_err(|e| format!("{e}\n{constant}"))?;
            if value != value_to_libxls(&actual).map_err(|e| e.to_string())? {
                return Err(format!(
                    "DSLX constexpr cast returned {value}, BX returned {actual}\n{constant}"
                ));
            }
        } else if interpreted.is_ok() {
            return Err(format!(
                "unexpected native constexpr cast result for {argument}\n{constant}"
            ));
        }
        let bx_const = format!(
            "{source}\nfn probe({params}) -> {ty} {{ {body} }}\nconst OBS = probe({argument});\n{}",
            bx_source("", &domain.ty, &format!("Mode::M0 as {}", domain.ty)),
            ty = domain.ty,
        );
        let bx_result = parse_module(&bx_const).and_then(|m| check_module(&m));
        if valid != bx_result.is_ok() {
            return Err(format!(
                "expected BX constexpr cast acceptance {valid}, got {:?}\n{bx_const}",
                bx_result.err()
            ));
        }
        if let Ok(checked) = bx_result {
            let observed = checked
                .constants()
                .iter()
                .find(|c| c.name == "OBS")
                .ok_or("missing BX enum cast observation")?;
            if observed.value != actual {
                return Err(format!(
                    "BX constexpr cast returned {}, hardware returned {actual}\n{bx_const}",
                    observed.value
                ));
            }
        }
    }
    Ok(())
}

/// Compare generated numeric enum declarations, operations and casts against
/// the pinned DSLX frontend, concrete evaluation and compiled IR.
pub fn check_enums(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let family = bytes.next().unwrap() % 11;
    let flags = bytes.next().unwrap();
    let width_selector = bytes.next().unwrap();
    let count_selector = bytes.next().unwrap();
    let salt = bytes.next().unwrap();
    let domain = Domain::generate(flags, width_selector, count_selector, salt);
    let style = bytes.next().unwrap();
    let source = domain.declarations(flags & 64 != 0, style);
    let outcome = match family {
        0 | 1 => compare_members(&domain, source.clone(), flags, input_seed(data)),
        2 | 3 => compare_rejection(&domain, source.clone(), style, input_seed(data)),
        4 | 5 => compare_cast(&domain, source.clone(), flags, bytes.next().unwrap()),
        6 => compare_ports(flags, width_selector, salt, style, input_seed(data)),
        7 => compare_members(
            &domain,
            domain.unsized_declaration(),
            flags,
            input_seed(data),
        ),
        8 => compare_import(&domain, source.clone(), flags, input_seed(data)),
        9 => check_ordering(
            &domain,
            &source,
            style,
            bytes.next().unwrap(),
            input_seed(data),
        ),
        _ => {
            let other_width = if flags & 16 != 0 {
                bytes.next().unwrap()
            } else {
                width_selector
            };
            let other_salt = if flags & 32 != 0 { salt } else { !salt };
            let other = Domain::generate(flags, other_width, count_selector, other_salt);
            compare_specializations(&domain, &other, flags, style, input_seed(data))
        }
    };
    outcome.map_err(|e| format!("{e}\nGenerated enum declarations:\n{source}"))
}

#[cfg(test)]
mod tests {
    use super::check_enums;

    #[test]
    fn generated_enum_values_and_casts_match_native_dslx() -> Result<(), String> {
        for width in 0..12 {
            for (shape, flags) in [0, 1, 4, 34, 65, 218].into_iter().enumerate() {
                check_enums(&[0, flags, width, 3, 97, width])?;
                for family in 6..10 {
                    check_enums(&[family, flags, width, 3, 97, width, shape as u8])?;
                }
                for cast_width in 0..3 {
                    check_enums(&[4, flags, width, 3, 97, width, cast_width])?;
                }
            }
        }
        for mutation in 0..16 {
            check_enums(&[2, 3, 7, 3, 21, mutation])?;
        }
        for flags in [0, 1, 2, 8, 16, 18, 32, 65] {
            check_enums(&[10, flags, 3, 3, 97, 2, 9])?;
        }
        Ok(())
    }
}
