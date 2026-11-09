// SPDX-License-Identifier: Apache-2.0

//! Compare BX value-parametric functions with the same declarations in DSLX.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program, compare_program};
use crate::input_seed;
use std::fs;
use xlsynth_pir::{IrBits, IrFormatPreference};

/// Dimensions and choices shared by the generated function and its callers.
struct Sample {
    width: usize,
    length: usize,
    value: u8,
    signed: bool,
    nested_call: bool,
    mutation: u8,
}

/// The native module and expressions used for its last value parametric.
struct ImportedBinding {
    module: String,
    first: String,
    second: String,
    wrong: String,
}

impl Sample {
    /// Generate a scalar or nominal binding for a function whose signedness
    /// and width can be inferred independently from its argument.
    fn imported_binding(&self, nominal: bool) -> ImportedBinding {
        let value = self.value;
        let alternate = value.wrapping_add(1);
        let prefix = "pub fn apply<S:bool,N:u32,";
        let signature = ">(x:xN[S][N])->u32";
        // Native sign extension requires a nonempty operand.
        let input = if self.signed && self.width == 0 {
            "((x == xN[S][N]:0) as u32)"
        } else {
            "(x as u32)"
        };
        if !nominal {
            return ImportedBinding {
                module: format!("{prefix}K:u32={{N+u32:1}}{signature} {{ {input} ^ K }}"),
                first: format!("u32:{value}"),
                second: format!("u32:{value} + u32:1"),
                wrong: format!("u8:{value}"),
            };
        }
        if self.signed {
            return ImportedBinding {
                module: format!(
                    r#"pub enum Mode:u8 {{ A={value}, B={alternate} }}
pub enum Other:u8 {{ B={alternate} }}
{prefix}K:Mode={{Mode::A}}{signature} {{ {input} ^ ((K as u8) as u32) }}"#
                ),
                first: "bias::Mode::A".into(),
                second: "bias::Mode::B".into(),
                wrong: "bias::Other::B".into(),
            };
        }
        let fields = |amount: u8| format!("{{offsets:(s8:-1,[u8:{amount},u8:{alternate}])}}");
        ImportedBinding {
            module: format!(
                r#"pub struct Mode {{ offsets:(s8,u8[2]) }}
pub struct Other {{ offsets:(s8,u8[2]) }}
{prefix}K:Mode={{Mode {default}}}{signature} {{ {input} ^ (K.offsets.0 as u32) ^ (K.offsets.1[u32:0] as u32) }}"#,
                default = fields(value)
            ),
            first: format!("bias::Mode {}", fields(value)),
            second: format!("bias::Mode {}", fields(alternate)),
            wrong: format!("bias::Other {}", fields(alternate)),
        }
    }

    /// Compare repeated direct calls or named maps using native declarations,
    /// varying explicit values independently of the hardware signature.
    fn imported(&self, nominal: bool, seed: u64) -> Result<(), String> {
        let binding = self.imported_binding(nominal);
        let signed = self.signed;
        let width = self.width;
        let first = format!("<{signed}>");
        let last = format!("<{signed},{width},first>");
        let middle = match self.mutation {
            0 => String::new(),
            1 => format!("<{signed},{width}>"),
            2 if !nominal => format!("<{signed},{width},{}>", u32::from(self.value) + 1),
            2 | 3 => format!("<{signed},{width},second>"),
            4 => format!("<{}>", !signed),
            5 => format!("<{signed},{}>", width + 1),
            6 => format!("<{signed},{width},wrong>"),
            _ => format!("<{signed},{width},runtime>"),
        };
        let input = format!("{}N[{width}]", if signed { 's' } else { 'u' });
        let (params, result) = if self.nested_call {
            let len = self.length;
            (
                format!("x:{input}[{len}], flag:bool"),
                format!("(u32[{len}],u32[{len}],u32[{len}])"),
            )
        } else {
            (format!("x:{input}, flag:bool"), "(u32,u32,u32)".to_owned())
        };
        let call = |name: &str, actuals: &str| {
            if self.nested_call {
                format!("map(x,{name}{actuals})")
            } else {
                format!("{name}{actuals}(x)")
            }
        };
        let body = |name: &str, middle: &str| {
            format!(
                "let first={}; let second={}; let wrong={}; let runtime=if flag {{ first }} else {{ second }};\n({},{},{})",
                binding.first,
                binding.second,
                binding.wrong,
                call("bias::apply", &first),
                call(name, middle),
                call("bias::apply", &last)
            )
        };
        // v0.61.0 can crash lowering an inline expression in a map callback's
        // angle brackets. Bind the same expression in the native reference.
        let bx_middle = if self.mutation == 3 {
            format!("<{signed},{width},{{{}}}>", binding.second)
        } else {
            middle.clone()
        };
        let bx = format!(
            "use bias; use bias::apply;\n{}",
            bx_source(&params, &result, &body("apply", &bx_middle))
        );
        let native_body = body("bias::apply", &middle);
        compare_imported_program(
            "bias",
            &binding.module,
            &bx,
            FunctionProgram {
                functions: "import bias;".into(),
                params,
                result,
                body: native_body,
                valid: Some(self.mutation < 4),
                dslx_ir: true,
                dynamic_index: None,
            },
            seed,
        )
    }

    /// Supply inline data operands to an imported function under a partial
    /// or complete width binding, preserving typed evidence inside aggregates.
    fn imported_data_operands(&self, seed: u64) -> Result<(), String> {
        let library = r#"
pub fn scalar<K:u32,N:u32>(x:uN[N])->u32 { (x as u32) ^ N ^ K }
pub fn signed_id<N:u32>(x:sN[N])->sN[N] { x }
pub fn pair<K:u32,N:u32>(p:(uN[N],uN[N]))->u32 { ((p.0 - p.1) as u32) ^ N ^ K }
pub fn lanes<K:u32,N:u32>(p:uN[N][2])->u32 { ((p[u32:0] ^ p[u32:1]) as u32) ^ N ^ K }
"#;
        let literal = self.value.max(1);
        let minimum = (u8::BITS - literal.leading_zeros()) as usize;
        let width = minimum + self.width + 1;
        let k = self.length;
        let partial = format!("<{k}>");
        let full = format!("<{k},{width}>");
        let (body, valid) = match self.mutation {
            0 if self.signed && self.nested_call => (
                format!("operands::signed_id<{minimum}>(0x{literal:x})"),
                true,
            ),
            0 if self.signed => (format!("operands::signed_id<{minimum}>({literal})"), false),
            0 => (format!("operands::scalar{partial}({literal})"), false),
            1 if self.signed => (
                format!("let v=0x{literal:x}; operands::signed_id<{minimum}>(v)"),
                true,
            ),
            1 => (
                format!("let v={literal}; operands::scalar{partial}(v)"),
                true,
            ),
            // v0.61.0 can abort converting an untyped let subsequently
            // widened by an explicit N. Type the let at its call width.
            2 if self.nested_call => (
                format!("let v=uN[{width}]:{literal}; operands::scalar{full}(v)"),
                true,
            ),
            2 => (format!("operands::scalar{full}({literal})"), true),
            _ => {
                let tuple = match self.mutation {
                    3 => true,
                    4 => false,
                    _ => self.signed,
                };
                let (function, open, close) = if tuple {
                    ("pair", '(', ')')
                } else {
                    ("lanes", '[', ']')
                };
                let first = format!("{open}{literal},x{close}");
                let other = if self.mutation == 6 { "y" } else { "x" };
                let last = format!("{open}{other},1{close}");
                let operand = match self.mutation {
                    5 | 6 if self.nested_call => {
                        format!("match flag {{ true=>{first}, _=>{last} }}")
                    }
                    5 | 6 => format!("if flag {{ {first} }} else {{ {last} }}"),
                    _ => first,
                };
                // Elements of an array share a type. Tuple fields supply
                // independent evidence when the call leaves N unbound.
                let actuals = if self.mutation == 7 { &partial } else { &full };
                (
                    format!("operands::{function}{actuals}({operand})"),
                    self.mutation != 6 && (self.mutation != 7 || !tuple),
                )
            }
        };
        let params = format!("x:uN[{width}], y:uN[{}], flag:bool", width + 1);
        let result = if self.signed && self.mutation < 2 {
            format!("sN[{minimum}]")
        } else {
            "u32".to_owned()
        };
        let bx = format!("use operands;\n{}", bx_source(&params, &result, &body));
        compare_imported_program(
            "operands",
            library,
            &bx,
            FunctionProgram {
                functions: "import operands;".into(),
                params,
                result,
                body,
                valid: Some(valid),
                dslx_ir: true,
                dynamic_index: None,
            },
            seed,
        )
    }

    /// Infer repeated dimensions inside a tuple, with optional positional
    /// arguments, a dependent default, and forwarding through a second call.
    fn aggregate(&self) -> FunctionProgram {
        let Self {
            width,
            length,
            signed,
            mutation,
            ..
        } = *self;
        let signature = "<S:bool, N:u32, L:u32, R:u32={N + L}>(p:(xN[S][N][L], xN[S][N])) -> uN[R]";
        // Native IR requires a nonempty operand for sign extension, even for
        // a same-width cast. A comparison can observe signed zero-width inputs.
        let operation = if signed && width == 0 { "==" } else { "^" };
        let mut functions = format!(
            r#"fn mix{signature} {{
    let v = p.0[L - u32:1] {operation} p.1;
    (v as uN[R]) + ((if S {{ L }} else {{ N }}) as uN[R])
}}"#
        );
        let callee = if self.nested_call {
            functions.push_str(&format!("\nfn relay{signature} {{ mix<S,N,L,R>(p) }}"));
            "relay"
        } else {
            "mix"
        };
        let prefix = if signed { 's' } else { 'u' };
        let other_width = if mutation == 5 { width + 1 } else { width };
        let other_prefix = if signed ^ (mutation == 6) { 's' } else { 'u' };
        let params = format!("p:({prefix}N[{width}][{length}], {other_prefix}N[{other_width}])");
        let mut result_width = width + length;
        let actuals = match mutation {
            1 => format!("<{signed}>"),
            2 => format!("<{signed}, u32:{width}>"),
            3 => format!("<{signed}, {width}, {length}>"),
            4 => {
                result_width += 1 + usize::from(self.value);
                format!("<{signed}, {width}, {length}, {result_width}>")
            }
            7 => format!("<{signed}, {width}, {}>", length + 1),
            _ => String::new(),
        };
        FunctionProgram {
            functions,
            params,
            result: format!("uN[{result_width}]"),
            body: format!("{callee}{actuals}(p)"),
            valid: Some(mutation < 5),
            dslx_ir: true,
            dynamic_index: None,
        }
    }

    /// Vary value bindings independently of an identical hardware signature,
    /// including values that differ only above bit 63.
    fn bindings(&self) -> FunctionProgram {
        let prefix = if self.signed { 's' } else { 'u' };
        let width = self.width;
        let ty = format!("{prefix}N[{width}]");
        let high_bit = IrBits::signed_min_value(width);
        let value = IrBits::from_lsb_fn(width, |i| {
            i + 1 < width && i < 8 && self.value & (1 << i) != 0
        });
        let hex = |bits: &IrBits| bits.to_string_fmt(IrFormatPreference::Hex, false);
        let base = hex(&value);
        let functions = format!(
            r#"const BASE = {ty}:{base};
fn adjust<K:{ty}, ADD:bool={{true}}>(x:{ty}) -> {ty} {{
    if ADD {{ x + K }} else {{ x ^ K }}
}}"#
        );
        let other = if self.nested_call {
            format!("{{BASE ^ {ty}:{}}}", hex(&high_bit))
        } else {
            format!("{ty}:{}", hex(&value.or(&high_bit)))
        };
        let second = match self.mutation {
            0 => format!("adjust<{other}>(b)"),
            1 => format!("adjust<{other},true>(b)"),
            2 => format!("adjust<{other},false>(b)"),
            3 => "adjust<local_bias,false>(b)".into(),
            4 => "adjust(b)".into(),
            5 => "adjust<b>(b)".into(),
            6 => "adjust<BASE,u16:1>(b)".into(),
            _ => "adjust<BASE,true,false>(b)".into(),
        };
        FunctionProgram {
            functions,
            params: format!("a:{ty}, b:{ty}"),
            result: format!("({ty},{ty},{ty})"),
            body: format!(
                "let local_bias = BASE; (adjust<local_bias>(a), {second}, adjust<BASE>(b))"
            ),
            valid: Some(self.mutation < 4),
            dslx_ir: true,
            dynamic_index: None,
        }
    }

    /// Check a derived width against a binding obtained from a second
    /// argument, an explicit value or a default.
    fn derived_dimension(&self) -> FunctionProgram {
        let width = self.width;
        let extra = self.length;
        let has_other = matches!(self.mutation, 0 | 3 | 4);
        let default = match self.mutation {
            2 | 3 => format!("={{u32:{width}}}"),
            6 => format!("={{u16:{width}}}"),
            _ => String::new(),
        };
        let arg = if has_other { ", y:uN[N]" } else { "" };
        let term = if has_other { " + (y as u32)" } else { "" };
        let functions = format!(
            "fn measure<N:u32{default}>(x:uN[N + u32:{extra}]{arg}) -> u32 {{ (x as u32) + N{term} }}"
        );
        let derived = width + extra + usize::from(self.mutation == 4);
        let mut params = format!("x:uN[{derived}]");
        let mut args = "x".to_string();
        if has_other {
            params.push_str(&format!(", y:uN[{width}]"));
            args.push_str(",y");
        }
        let actual = match self.mutation {
            1 => format!("<u32:{width}>"),
            7 => format!("<u16:{width}>"),
            _ => String::new(),
        };
        FunctionProgram {
            functions,
            params,
            result: "u32".into(),
            body: format!("measure{actual}({args})"),
            valid: Some(self.mutation < 4),
            dslx_ir: true,
            dynamic_index: None,
        }
    }

    /// A scalar parametric's type can depend on an inferred width or a
    /// constant visible at the declaration.
    fn dependent_binding(&self) -> FunctionProgram {
        let width = self.width;
        let module_width = self.nested_call;
        let dimension = if module_width { "WIDTH" } else { "N" };
        let first = if module_width { "" } else { "N:u32, " };
        let value = IrBits::from_lsb_fn(width, |i| i < 8 && self.value & (1 << i) != 0)
            .to_string_fmt(IrFormatPreference::Hex, false);
        let ty = format!("uN[{dimension}]");
        let default_type = if self.mutation == 6 {
            format!("uN[{dimension}+u32:1]")
        } else {
            ty.clone()
        };
        let functions = format!(
            "const WIDTH = u32:{width};\nfn offset<{first}X:{ty}={{{default_type}:{value}}}>(x:{ty})->{ty} {{ x + X }}"
        );
        let input_width = width + usize::from(self.mutation == 5);
        let actual = match (module_width, self.mutation) {
            (true, 7) => format!("<uN[{}]:{value}>", width + 1),
            (true, 1 | 3) => format!("<uN[{width}]:{value}>"),
            (true, 2 | 4) => format!("<{value}>"),
            (false, 7) => format!("<u16:{width}>"),
            (false, 1 | 3 | 5) => "<WIDTH>".into(),
            (false, 2 | 4) => format!("<{width}>"),
            _ => String::new(),
        };
        FunctionProgram {
            functions,
            params: format!("x:uN[{input_width}]"),
            result: format!("uN[{width}]"),
            body: format!("offset{actual}(x)"),
            valid: Some(self.mutation < 5),
            dslx_ir: true,
            dynamic_index: None,
        }
    }

    /// Both tuple fields supply width evidence, including the independently
    /// inferred width of an untyped literal when N was omitted.
    fn tuple_witnesses(&self) -> FunctionProgram {
        let value = self.value.max(1);
        let minimum = (u8::BITS - value.leading_zeros()) as usize;
        let width = if self.nested_call {
            minimum
        } else {
            minimum + self.width + 1
        };
        let ty = format!("uN[{width}]");
        let field = match self.mutation {
            2 => format!("{ty}:{value}"),
            4 => format!("uN[{}]:{value}", width + 1),
            _ => value.to_string(),
        };
        let pair = match self.mutation {
            3 => format!("({value},{value})"),
            _ if self.signed => format!("({field},x)"),
            _ => format!("(x,{field})"),
        };
        let actual = match self.mutation {
            1 | 7 => format!("<{width}>"),
            5 => format!("<{}>", width + 1),
            _ => String::new(),
        };
        let body = if self.mutation == 6 {
            format!("let pair = {pair}; observe(pair)")
        } else {
            format!("observe{actual}({pair})")
        };
        FunctionProgram {
            functions:
                "fn observe<N:u32>(p:(uN[N],uN[N]))->u32 { (p.0 as u32) + (p.1 as u32) + N }".into(),
            params: format!("x:{ty}"),
            result: "u32".into(),
            body,
            valid: Some(match self.mutation {
                0 | 6 => width == minimum,
                4 | 5 => false,
                _ => true,
            }),
            dslx_ir: true,
            dynamic_index: None,
        }
    }

    /// Observe two bindings of one nominal type and reject a value from a
    /// different declaration with the same representation.
    fn nominal_binding(&self) -> FunctionProgram {
        let value = self.value;
        let alternate = value.wrapping_add(1);
        let (declarations, first, second, wrong, computation) = if self.signed {
            (
                format!(
                    "enum Choice: u8 {{ First={value}, Second={alternate} }} enum Other: u8 {{ Second={alternate} }}"
                ),
                "Choice::First".to_owned(),
                "Choice::Second".to_owned(),
                "Other::Second".to_owned(),
                "x + (K as u8)",
            )
        } else {
            let add = self.nested_call;
            (
                "struct Choice { amount:u8, add:bool } struct Other { amount:u8, add:bool }"
                    .to_owned(),
                format!("Choice {{ amount:u8:{value}, add:{add} }}"),
                format!("Choice {{ amount:u8:{alternate}, add:{} }}", !add),
                format!("Other {{ amount:u8:{alternate}, add:{} }}", !add),
                "if K.add { x + K.amount } else { x ^ K.amount }",
            )
        };
        let functions = format!(
            "{declarations}\nconst FIRST = {first};\nconst SECOND = {second};\nfn apply<K:Choice={{FIRST}}>(x:u8)->u8 {{ {computation} }}"
        );
        let call = match self.mutation {
            0 => "apply(x)".to_owned(),
            1 => "apply<SECOND>(x)".to_owned(),
            2 => "apply<local>(x)".to_owned(),
            3 => format!("apply<{{{second}}}>(x)"),
            4 => format!("apply<{{{wrong}}}>(x)"),
            5 => "apply<runtime>(x)".to_owned(),
            6 => format!("apply<u8:{value}>(x)"),
            _ => "apply<FIRST, SECOND>(x)".to_owned(),
        };
        FunctionProgram {
            functions,
            params: "x:u8".into(),
            result: "(u8,u8,u8)".into(),
            body: format!(
                "let local = SECOND; let runtime = if x == u8:0 {{ FIRST }} else {{ SECOND }}; (apply<FIRST>(x), {call}, apply<FIRST>(x))"
            ),
            valid: Some(self.mutation < 4),
            dslx_ir: true,
            dynamic_index: None,
        }
    }

    /// Read bound values as match patterns, loop bounds and width slices.
    /// Defaults and dimensions can call preceding pure helpers.
    fn body_uses(&self) -> FunctionProgram {
        let width = 1 + self.width % 32;
        let length = self.length;
        let tag = self.value;
        let default = match self.mutation {
            5 => String::new(),
            6 => format!("={{u16:{tag}}}"),
            _ => format!("={{u8:{tag}}}"),
        };
        let functions = format!(
            r#"fn increment(n:u32)->u32 {{ n + u32:1 }}
fn observe<N:u32, P:u8{default}, L:u32={{increment(u32:{length})}}>(x:uN[N], tag:u8)->(uN[N],u32,bool) {{
    let field = x[0+:uN[increment(N)-u32:1]];
    let sum = for (i, acc) in u32:0..increment(L) {{ acc + i }}(u32:0);
    (field, sum, match tag {{ P => true, _ => false }})
}}"#
        );
        let supplied_tag = tag.wrapping_add(1);
        let actuals = match self.mutation {
            1 => format!("<{width}>"),
            2 | 6 => format!("<{width}, {supplied_tag}>"),
            3 => format!("<{width}, {supplied_tag}, {length}>"),
            4 => format!("<{}, {supplied_tag}>", width + 1),
            7 => format!("<{width}, tag>"),
            _ => String::new(),
        };
        let matching_tag = if matches!(self.mutation, 2 | 3) {
            supplied_tag
        } else {
            tag
        };
        FunctionProgram {
            functions,
            params: format!("x:uN[{width}], tag:u8, hit:bool"),
            result: format!("(uN[{width}],u32,bool)"),
            body: format!("observe{actuals}(x, if hit {{ u8:{matching_tag} }} else {{ tag }})"),
            valid: Some(self.mutation < 4),
            dslx_ir: true,
            dynamic_index: None,
        }
    }

    /// A bare scalar literal needs a binding from another source. Giving the
    /// literal a type or naming it in a let supplies independent type evidence.
    fn literal_arguments(&self) -> FunctionProgram {
        let value = self.value;
        let minimum = (u8::BITS - value.leading_zeros()).max(1) as usize;
        let width = minimum + self.width % 32;
        let body = match self.mutation {
            0 => format!("width({value})"),
            1 => format!("width(uN[{width}]:{value})"),
            2 => format!("let v={value}; width(v)"),
            3 => format!("width<{width}>({value})"),
            4 => format!("pair({value}, x)"),
            5 => format!("pair(x, {value})"),
            6 => format!("width({value} + 1)"),
            _ => format!("width(if select {{ {value} }} else {{ 1 }})"),
        };
        FunctionProgram {
            functions:
                "fn width<N:u32>(x:uN[N])->u32 { N }\nfn pair<N:u32>(x:uN[N],y:uN[N])->u32 { N }"
                    .into(),
            params: format!("x:uN[{width}], select:bool"),
            result: "u32".into(),
            body,
            valid: Some(matches!(self.mutation, 1..=5)),
            dslx_ir: true,
            dynamic_index: None,
        }
    }

    /// Evaluate conditional actuals directly or via a let. Native constexpr
    /// checks the free names of every match arm and short-circuits an if.
    fn conditional_actuals(&self) -> FunctionProgram {
        let value = self.value;
        let expr = match self.mutation {
            0 if self.signed => format!("({{ let v=u8:{value}; v }})"),
            0 => format!("match true {{ true => u8:{value}, _ => u8:0 }}"),
            1 => format!("match true {{ true => u8:{value}, _ => x }}"),
            2 => format!("match true {{ true => x, _ => u8:{value} }}"),
            3 => format!("if true {{ u8:{value} }} else {{ x }}"),
            4 => format!("if false {{ x }} else {{ u8:{value} }}"),
            5 => format!("if false {{ u8:{value} }} else {{ x }}"),
            6 => {
                format!("if true {{ u8:{value} }} else {{ match true {{ true => u8:0, _ => x }} }}")
            }
            7 if !self.nested_call => format!("{{ u8:{value} }}"),
            _ => format!("if select {{ u8:{value} }} else {{ u8:0 }}"),
        };
        let body = if self.nested_call {
            format!("let bias = {expr}; apply<bias>(x)")
        } else {
            format!("apply<{{{expr}}}>(x)")
        };
        FunctionProgram {
            functions: "fn apply<K:u8>(x:u8)->u8 { x ^ K }".into(),
            params: "x:u8, select:bool".into(),
            result: "u8".into(),
            body,
            valid: Some(matches!(self.mutation, 0 | 3 | 4 | 6)),
            dslx_ir: true,
            dynamic_index: None,
        }
    }

    /// Infer the callback separately for each map element type and then reuse
    /// the first specialization.
    fn map(&self) -> FunctionProgram {
        let first = if self.signed { 's' } else { 'u' };
        let second = if self.signed { 'u' } else { 's' };
        let a = format!("{first}N[{}][{}]", self.width, self.length);
        let b = format!("{second}N[{}][{}]", self.width + 1, self.length + 1);
        let default = if self.mutation == 5 {
            ""
        } else {
            "={N + u32:1}"
        };
        let signed = self.signed;
        let width = self.width;
        let value = self.value;
        let callback = match self.mutation {
            1 => format!("alter<{signed}>"),
            2 => format!("alter<{signed},{width}>"),
            3 | 4 => format!("alter<{signed},{width},u32:{value}>"),
            6 => format!("alter<{}>", !signed),
            7 => format!("alter<{signed},{width},u8:{value}>"),
            _ => "alter".into(),
        };
        let last = if self.mutation == 4 {
            format!("alter<{signed},{width},{}>", u32::from(value) + 1)
        } else {
            callback.clone()
        };
        FunctionProgram {
            functions: format!(
                "fn alter<S:bool,N:u32,K:u32{default}>(x:xN[S][N])->xN[S][N] {{ x ^ (K as xN[S][N]) }}"
            ),
            params: format!("a:{a}, b:{b}"),
            result: format!("({a},{b},{a})"),
            body: format!("(map(a,{callback}),map(b,alter),map(a,{last}))"),
            valid: Some(self.mutation <= 4),
            dslx_ir: true,
            dynamic_index: None,
        }
    }
}

/// Run both frontends and their evaluation oracles against the same native
/// import.
fn compare_imported_program(
    name: &str,
    library: &str,
    bx: &str,
    program: FunctionProgram,
    seed: u64,
) -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    fs::write(dir.path().join(format!("{name}.x")), library).map_err(|e| e.to_string())?;
    compare_bx_program(
        bx,
        program,
        seed,
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[dir.path()],
    )
    .map_err(|e| format!("{e}\n{name}.x:\n{library}"))
}

/// Generate shared parametric declarations and compare source acceptance,
/// concrete DSLX calls, and both hardware evaluations.
pub fn check_function_value_parametrics(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let family = bytes.next().unwrap() % 13;
    let flags = bytes.next().unwrap();
    let sample = Sample {
        width: usize::from(bytes.next().unwrap()),
        length: 1 + usize::from(bytes.next().unwrap() % 4),
        value: bytes.next().unwrap(),
        signed: flags & 1 != 0,
        nested_call: flags & 2 != 0,
        mutation: bytes.next().unwrap() % 8,
    };
    let program = match family {
        0 => sample.aggregate(),
        1 => sample.bindings(),
        2 => sample.derived_dimension(),
        3 => sample.map(),
        4 => sample.dependent_binding(),
        5 => sample.tuple_witnesses(),
        6 => sample.nominal_binding(),
        7 => sample.body_uses(),
        8 => sample.literal_arguments(),
        9 => sample.conditional_actuals(),
        10 | 11 => return sample.imported(family == 11, input_seed(data)),
        _ => return sample.imported_data_operands(input_seed(data)),
    };
    compare_program(
        program,
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::check_function_value_parametrics;

    #[test]
    fn compare_generated_value_parametrics() -> Result<(), String> {
        for family in 0..13 {
            for (flags, width) in [(0, 1), (1, 0), (2, 77), (3, 255)] {
                for mutation in 0..8 {
                    check_function_value_parametrics(&[family, flags, width, 2, 21, mutation])?;
                }
            }
        }
        for value in [1, 127, 128, 255] {
            for flags in [1, 3] {
                for mutation in [0, 1] {
                    check_function_value_parametrics(&[12, flags, 8, 0, value, mutation])?;
                }
            }
        }
        Ok(())
    }
}
