// SPDX-License-Identifier: Apache-2.0

//! Check specialization symbols and the supported subset for concrete struct
//! imports.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, hash_map::Entry};

use xlsynth::dslx::{Function, ModuleMember, ParametricEnv, Type, TypeInfo, TypecheckedModule};
use xlsynth::{DslxCallingConvention, mangle_dslx_name_with_env};

use crate::import_structs::EffectSources;

/// One function declaration under the value/type bindings of a checked call.
#[derive(Clone, Hash, Eq, PartialEq)]
struct Instance {
    function: Function,
    env: ParametricEnv,
}

/// Native call-site metadata indexed by the exact caller specialization.
struct Site {
    callee: Instance,
    context: Option<TypeInfo>,
    /// Repeated typechecks in a shared import graph can record the same AST
    /// call.
    invocation: usize,
}

/// The shared snapshot can contain invocations from earlier, unrelated
/// probes. Index once per callee and select only the current caller's sites.
fn sites(info: &TypeInfo, callee: &Function) -> Result<HashMap<Instance, Vec<Site>>, String> {
    let mut by_caller = HashMap::<_, Vec<_>>::new();
    let entries = info
        .get_all_invocation_callee_data(callee)
        .ok_or("missing DSLX callee data")?;
    for i in 0..entries.len() {
        let site = entries.get(i).ok_or("missing DSLX invocation")?;
        let invocation = site.invocation().ok_or("missing DSLX invocation site")?;
        let Some(caller) = info
            .get_root_invocation_data(&invocation)
            .and_then(|v| v.caller())
        else {
            // Constant initializers have no enclosing runtime function.
            continue;
        };
        let caller = Instance {
            function: caller,
            env: site
                .caller_bindings()
                .ok_or("missing DSLX caller bindings")?,
        };
        let callee = Instance {
            function: callee.clone(),
            env: site
                .callee_bindings()
                .ok_or("missing DSLX callee bindings")?,
        };
        by_caller.entry(caller).or_default().push(Site {
            callee,
            context: site.derived_type_info(),
            invocation: invocation.as_ptr() as usize,
        });
    }
    Ok(by_caller)
}

/// Native names distinguish bits and arrays of bits used as type arguments.
fn scalar_or_array(mut ty: Type) -> bool {
    while ty.is_array() {
        ty = ty.get_array_element_type();
    }
    !ty.is_enum() && ty.is_bits_like().is_some()
}

/// v0.61 can produce invalid symbols or silently merge concrete nominal and
/// tuple type arguments. Value parameters (including u32 struct dimensions)
/// have an IR value and use the converter's ordinary value specialization.
fn unsupported_type_binding(site: &Site) -> Option<String> {
    let bindings = site.callee.env.bindings();
    for (name, actual) in bindings {
        if actual.convert_to_ir().is_ok() {
            continue;
        }
        let annotation = site
            .callee
            .function
            .parametric_bindings()
            .into_iter()
            .find(|binding| binding.get_identifier() == name)
            .and_then(|binding| binding.get_type_annotation());
        let ty = site
            .context
            .as_ref()
            .and_then(|info| info.get_type_for_type_annotation(&annotation?));
        if !ty.is_some_and(scalar_or_array) {
            return Some(format!(
                "DSLX function '{}' specializes type parameter '{name}' as '{}'; BX supports only bits and arrays of bits as type-valued arguments when importing concrete parametric structs with libxls v0.61",
                site.callee.function.get_identifier(),
                actual.to_text()
            ));
        }
    }
    None
}

/// An explicitly named sibling can occupy the spelling of a specialization,
/// such as `foo__2` and `foo<N>` at N=2. Only check reached declarations.
fn named_sibling(instances: &HashSet<Instance>) -> Result<Option<String>, String> {
    let mut explicit = BTreeMap::<String, BTreeSet<String>>::new();
    for instance in instances {
        let name = instance.function.get_identifier();
        if name.contains("__") {
            explicit
                .entry(instance.function.get_body().get_owner_module().get_name())
                .or_default()
                .insert(name);
        }
    }
    if explicit.is_empty() {
        return Ok(None);
    }
    let mut collisions = BTreeSet::new();
    for instance in instances {
        let module = instance.function.get_body().get_owner_module().get_name();
        let Some(siblings) = explicit.get(&module) else {
            continue;
        };
        let name = instance.function.get_identifier();
        let is_method = ModuleMember::try_from(&instance.function).is_err();
        if !is_method && instance.env.is_empty() {
            continue;
        }
        // Ask the released mangler for the local specialization spelling.
        // Impl receiver scopes are unavailable. A matching scoped suffix
        // identifies a potentially colliding declaration in the same module.
        let symbol = mangle_dslx_name_with_env(
            "bx",
            &name,
            DslxCallingConvention::Typical,
            &[],
            Some(&instance.env),
            None,
        )
        .map_err(|e| e.to_string())?;
        let local = symbol
            .strip_prefix("__bx__")
            .ok_or("unexpected native DSLX specialization spelling")?;
        let method_suffix = is_method.then(|| format!("__{local}"));
        for sibling in siblings {
            if sibling != &name
                && (sibling == local
                    || method_suffix
                        .as_ref()
                        .is_some_and(|suffix| sibling.ends_with(suffix)))
            {
                collisions.insert((name.clone(), sibling.clone()));
            }
        }
    }
    Ok(collisions.pop_first().map(|(name, sibling)| format!("DSLX specialization of '{name}' can share a native IR symbol with '{sibling}'; this combination is unsupported with libxls v0.61")))
}

/// Native specialization checks applicable to the current import batch.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum GuardMode {
    ConcreteStructs,
    NumericEnums,
    TypeBindings,
}

/// Guard the checked batch before conversion, where an unsupported symbol
/// can abort the native process before any Rust error can be returned.
pub(super) fn check(
    checked: &TypecheckedModule,
    sources: &EffectSources<'_>,
    mode: GuardMode,
    explicit_values: bool,
) -> Result<(), String> {
    let has_concrete_structs = mode == GuardMode::ConcreteStructs;
    let info = checked.get_type_info();
    let graph = info
        .build_function_call_graph(&checked.get_module())
        .map_err(|e| e.to_string())?;
    let empty = ParametricEnv::empty().map_err(|e| e.to_string())?;
    let mut work: Vec<_> = (0..graph.function_count())
        .filter_map(|i| graph.get_function(i))
        .map(|function| Instance {
            function,
            env: empty.clone(),
        })
        .collect();
    let mut seen = HashSet::new();
    let mut by_callee = HashMap::new();
    let mut written = HashMap::new();
    let mut errors = BTreeSet::new();
    while let Some(caller) = work.pop() {
        if !seen.insert(caller.clone()) {
            continue;
        }
        let inspect_dispatch =
            has_concrete_structs || (mode == GuardMode::NumericEnums && !caller.env.is_empty());
        let mut visible_methods = BTreeMap::<String, usize>::new();
        let mut visible_callees = BTreeSet::new();
        for i in 0..graph.callee_count(&caller.function) {
            let callee = graph
                .get_callee(&caller.function, i)
                .ok_or("missing DSLX callee")?;
            if callee.get_body().get_owner_module().get_name() == "<builtin_stubs>" {
                continue;
            }
            let indexed = match by_callee.entry(callee.clone()) {
                Entry::Vacant(entry) => entry.insert(sites(&info, &callee)?),
                Entry::Occupied(entry) => entry.into_mut(),
            };
            if let Some(sites) = indexed.get(&caller) {
                if inspect_dispatch {
                    visible_callees.insert(callee.get_identifier());
                }
                if inspect_dispatch && ModuleMember::try_from(&callee).is_err() {
                    let count = visible_methods.entry(callee.get_identifier()).or_default();
                    let distinct = sites.iter().map(|s| s.invocation).collect::<HashSet<_>>();
                    *count = count.saturating_add(distinct.len());
                }
                for site in sites {
                    if has_concrete_structs && let Some(error) = unsupported_type_binding(site) {
                        errors.insert(error);
                    }
                    work.push(site.callee.clone());
                }
            } else if !callee.is_parametric() {
                if inspect_dispatch {
                    visible_callees.insert(callee.get_identifier());
                }
                if inspect_dispatch
                    && caller.env.is_empty()
                    && ModuleMember::try_from(&callee).is_err()
                {
                    visible_methods.insert(callee.get_identifier(), usize::MAX);
                }
                work.push(Instance {
                    function: callee,
                    env: empty.clone(),
                });
            }
        }
        if !inspect_dispatch {
            // The native root context supplies the concrete callees of an
            // ordinary caller. A specialized caller can omit impl dispatch.
            continue;
        }
        let calls = match written.entry(caller.function.clone()) {
            Entry::Vacant(entry) => entry.insert(sources.dispatches(&caller.function, &graph)?),
            Entry::Occupied(entry) => entry.into_mut(),
        };
        if has_concrete_structs && calls.indirect {
            errors.insert(format!("DSLX function '{}' uses a map callback or indirect invocation; BX cannot verify its target when converting imported parametric structs with libxls v0.61", caller.function.get_identifier()));
        }
        if !has_concrete_structs
            && (calls.unknown_indirect || !calls.indirect_names.is_subset(&visible_callees))
        {
            errors.insert(format!("libxls v0.61 does not expose the indirect dispatch inside specialized DSLX function '{}'; BX cannot verify its native specialization symbols", caller.function.get_identifier()));
        }
        for (name, count) in &calls.methods {
            if visible_methods.get(name).copied().unwrap_or(0) < *count {
                let detail = if has_concrete_structs {
                    "BX cannot verify this specialization before converting imported parametric structs"
                } else {
                    "BX cannot verify its native specialization symbols"
                };
                errors.insert(format!("libxls v0.61 does not expose the concrete method dispatch to '{name}' inside DSLX function '{}'; {detail}", caller.function.get_identifier()));
            }
        }
    }
    if let Some(error) = errors.pop_first() {
        Err(error)
    } else if (has_concrete_structs || explicit_values)
        && let Some(error) = named_sibling(&seen)?
    {
        Err(error)
    } else {
        check_type_symbols(&seen)
    }
}

/// v0.61 drops closing tuple delimiters from type-valued arguments. Distinct
/// bindings of the same declaration must still have distinct native symbols.
fn check_type_symbols(instances: &HashSet<Instance>) -> Result<(), String> {
    let mut functions = HashMap::<&Function, Vec<&ParametricEnv>>::new();
    for instance in instances {
        if !instance.env.is_empty() {
            functions
                .entry(&instance.function)
                .or_default()
                .push(&instance.env);
        }
    }
    let mut collisions = BTreeSet::new();
    for (function, bindings) in functions {
        if bindings.len() < 2 {
            continue;
        }
        let mut symbols = BTreeSet::new();
        let name = function.get_identifier();
        for env in bindings {
            let symbol = mangle_dslx_name_with_env(
                "bx",
                &name,
                DslxCallingConvention::Typical,
                &[],
                Some(env),
                None,
            )
            .map_err(|e| e.to_string())?;
            if !symbols.insert(symbol) {
                collisions.insert(name.clone());
            }
        }
    }
    if let Some(name) = collisions.first() {
        return Err(format!(
            "distinct specializations of DSLX function '{name}' have the same native IR symbol with libxls v0.61"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fs;
    use tempfile::tempdir;

    use super::{GuardMode, check};
    use crate::import_structs::ImportedStructs;
    use crate::{
        ImportOptions, ResetBehavior, check_module_with_options, compile_with_options,
        empty_package, parse_module,
    };

    #[test]
    fn truncated_native_body_cannot_hide_a_spaced_method_application() -> Result<(), Box<dyn Error>>
    {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("ops.x"),
            r#"#![feature(generics)]
pub enum Mode:u2 { A=0 }
struct Pair<N:u32> { x:uN[N] }
impl Pair<N> { fn choose<K:u32>(x:Mode)->bool { x==Mode::A } }
pub fn via<N:u32>(v:uN[N],x:Mode)->bool {
    let _ = '\x00';
    type P=Pair<N>;
    P::choose <u32:1> (x)
}"#,
        )?;
        let (structs, imports) = ImportedStructs::load(
            &parse_module("use ops;")?.uses,
            xlsynth::default_dslx_stdlib_path(),
            &[dir.path()],
        )?;
        let checked = structs.check_probe(
            "import ops; fn sample(x:ops::Mode)->bool { ops::via(u2:0,x) }",
            "bx_dispatch",
            &mut imports.ok_or("missing native import data")?,
        )?;
        assert_eq!(
            check(
                &checked,
                &structs.effect_sources(),
                GuardMode::NumericEnums,
                false
            )
            .unwrap_err(),
            "libxls v0.61 does not expose the indirect dispatch inside specialized DSLX function 'via'; BX cannot verify its native specialization symbols"
        );
        Ok(())
    }

    #[test]
    fn ordinary_map_and_sum_constructors_remain_direct() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("events.x"),
            r#"
pub enum Event { Idle, Data(u8), }
pub type Message = Event;
pub fn map(x: u8) -> u8 { x + u8:1 }
"#,
        )?;
        fs::write(
            dir.path().join("wire.x"),
            r#"
import events as foreign;
pub struct Tag<N: u32> { x: u8 }
pub type Two = Tag<2>;
enum Event { Idle, Data(u8), }
type Imported = foreign::Message;
fn map(x: u8) -> u8 { x + u8:2 }
pub fn direct(x: Two) -> (u8, u8, u8) {
    type Local = Imported;
    let first = Event::Data(map(x.x));
    let second = Local::Data(foreign::map(x.x));
    let third = foreign::Event::Data(x.x);
    (match first { Event::Data(v) => v, _ => u8:0 },
     match second { Local::Data(v) => v, _ => u8:0 },
     match third { foreign::Event::Data(v) => v, _ => u8:0 })
}
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let source = "use wire; #[allow(combinational_block)] block b(clk, x: wire::Two) -> (u8, u8, u8) { wire::direct(x) }";
        let linked = compile_with_options(
            source,
            empty_package("sample"),
            ResetBehavior::default(),
            &options,
        )?;
        xlsynth::IrPackage::parse_ir(&linked.to_string(), None)?;
        Ok(())
    }

    #[test]
    fn indirect_and_qualified_calls_cannot_hide_nominal_specializations()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        for (module, value) in [("left", 1), ("right", 2)] {
            fs::write(
                dir.path().join(format!("{module}.x")),
                format!(
                    r#"
pub struct Tag<N: u32> {{ value: u8 }}
pub type Item = Tag<2>;
impl Tag<N> {{ fn get(self) -> u8 {{ u8:{value} }} }}
"#
                ),
            )?;
        }
        fs::write(
            dir.path().join("outer.x"),
            r#"#![feature(generics)]
import left;
import right;
import dummy;
pub struct Outer<N: u32> { value: u8 }
pub type Two = Outer<2>;
fn read<T: type>(x: T) -> u8 { x.get() }
impl Outer<N> {
    fn through(x: (left::Item, right::Item)) -> (u8, u8) {
        (read<left::Item>(x.0), read<right::Item>(x.1))
    }
    fn signed<K: s32>(x: (left::Item, right::Item)) -> (u8, u8) {
        (read<left::Item>(x.0), read<right::Item>(x.1))
    }
    fn conditional_method(self, y: left::Item, z: right::Item) -> (u8, u8) {
        type Alias = Outer<N>;
        Alias::signed<{if s32:1 > s32:0 { s32:1 } else { s32:2 }}>((y, z))
    }
    fn less_method(self, y: left::Item, z: right::Item) -> (u8, u8) {
        type Alias = Outer<N>;
        Alias::signed<{if N < u32:1 { s32:1 } else { s32:2 }}>((y, z))
    }
    fn compare(self, x: u8, y: u8) -> u8 {
        let gt = (x > (y)) as u8;
        let lt = (x < (y)) as u8;
        (x >> (y)) + (x << (y)) + gt + lt
    }
}
pub fn direct<N: u32>(s: Outer<N>, y: left::Item, z: right::Item) -> (u8, u8) {
    map([(y, z)], Outer<N>::through)[u32:0]
}
pub fn shadow<N: u32>(s: Outer<N>, y: left::Item, z: right::Item) -> (u8, u8) {
    type dummy = Outer<N>;
    dummy::through((y, z))
}
pub fn ordinary<N: u32>(s: Outer<N>, y: left::Item, z: right::Item) -> (u8, u8) {
    let value = { type dummy = Outer<N>; let v: dummy = s; v.value };
    dummy::pair(value, z.value)
}
pub fn negative<N: u32>(s: Outer<N>, y: left::Item, z: right::Item) -> (u8, u8) {
    type Alias = Outer<N>;
    Alias::signed<s32:-1>((y, z))
}
pub fn character<N: u32>(s: Outer<N>, y: left::Item, z: right::Item) -> (u8, u8) {
    type Alias = Outer<N>;
    Alias::signed<s32:'x'>((y, z))
}
pub fn conditional<N: u32>(s: Outer<N>, y: left::Item, z: right::Item) -> (u8, u8) {
    type Alias = Outer<N>;
    Alias::signed<{if s32:1 > s32:0 { s32:1 } else { s32:2 }}>((y, z))
}
pub fn call_conditional(s: Two, y: left::Item, z: right::Item) -> (u8, u8) {
    s.conditional_method(y, z)
}
pub fn call_less(s: Two, y: left::Item, z: right::Item) -> (u8, u8) {
    s.less_method(y, z)
}
pub fn comparisons(s: Two, y: left::Item, z: right::Item) -> (u8, u8) {
    (s.compare(y.value, z.value), s.value)
}
"#,
        )?;
        fs::write(
            dir.path().join("dummy.x"),
            "pub fn pair(x: u8, y: u8) -> (u8, u8) { (x, y) }",
        )?;
        fs::write(
            dir.path().join("wire.x"),
            r#"
import outer; import left; import right; import dummy as Two;
pub fn nested<N: u32>(s: outer::Outer<N>, y: left::Item, z: right::Item) -> (u8, u8) {
    outer::Two::through((y, z))
}
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let source = "use outer; use left; use right; #[allow(combinational_block)] block b(clk, x: outer::Two, y: left::Item, z: right::Item) -> (u8, u8) { outer::direct(x, y, z) }";
        let error = check_module_with_options(&parse_module(source)?, &options)
            .err()
            .ok_or("expected unsupported map dispatch")?;
        assert_eq!(
            error,
            "compiling DSLX import 'outer': DSLX function 'direct' uses a map callback or indirect invocation; BX cannot verify its target when converting imported parametric structs with libxls v0.61"
        );
        for (module, name, method, caller) in [
            ("outer", "shadow", "through", "shadow"),
            ("wire", "nested", "through", "nested"),
            ("outer", "negative", "signed", "negative"),
            ("outer", "character", "signed", "character"),
            ("outer", "conditional", "signed", "conditional"),
            ("outer", "call_conditional", "signed", "conditional_method"),
            ("outer", "call_less", "signed", "less_method"),
        ] {
            let bx = format!(
                "use wire; {}",
                source.replace("outer::direct", &format!("{module}::{name}"))
            );
            let error = check_module_with_options(&parse_module(&bx)?, &options)
                .err()
                .ok_or("expected unsupported associated method dispatch")?;
            assert_eq!(
                error,
                format!(
                    "compiling DSLX import '{module}': libxls v0.61 does not expose the concrete method dispatch to '{method}' inside DSLX function '{caller}'; BX cannot verify this specialization before converting imported parametric structs"
                )
            );
        }
        for function in ["outer::ordinary", "outer::comparisons"] {
            let ordinary = source.replace("outer::direct", function);
            let package = compile_with_options(
                &ordinary,
                empty_package("sample"),
                ResetBehavior::default(),
                &options,
            )?;
            xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
        }
        Ok(())
    }

    #[test]
    fn concrete_struct_imports_check_only_reachable_converter_cases() -> Result<(), Box<dyn Error>>
    {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("wire.x"),
            r#"#![feature(generics)]
pub struct Tag<N: u32> { value: u8 }
pub type Item = Tag<2>;
fn echo<T: type>(v: T) -> T { v }
impl Tag<N> {
    fn through(self: Self) -> u8 { echo(self).value }
    fn add(self: Self, x: u8) -> u8 { self.value + x }
    fn callback(x: u8) -> u8 { x }
}
pub fn opaque<N: u32>(x: Tag<N>) -> u8 { x.through() }
pub fn mapped<N: u32>(x: Tag<N>) -> u8 { map([x.value], (Tag<N>::callback))[u32:0] }
pub fn grouped<N: u32>(x: Tag<N>) -> u8 { (Tag<N>::callback)(x.value) }
pub fn nominal(x: Item) -> u8 { echo(x).value }
pub fn tuple_arg(x: Item) -> u8 { echo((x.value, x.value)).0 }
fn value<N: u32>(x: Tag<N>) -> u8 { x.value }
fn value__2(x: u8) -> u8 { !x }
pub fn collision(x: Item) -> u8 { value(x) + value__2(x.value) }
struct A { x: u8 }
impl A { fn same(self) -> u8 { self.x } }
fn A__same(v: u8) -> u8 { !v }
pub fn ordinary_collision(v: Item) -> u8 { A { x: v.value }.same() + A__same(v.value) }
"#,
        )?;
        fs::write(
            dir.path().join("arrays.x"),
            r#"import std;
import wire;
pub fn scalar_and_array(x: wire::Item, y: u8[2][2]) -> u8 {
    std::enumerate(y[u32:0])[u32:0].1 + x.add(std::enumerate(y)[u32:0].1[u32:0])
}
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let source = "use wire; use arrays; #[allow(combinational_block)] block b(clk, x: wire::Item, y: u8[2][2]) -> u8 { arrays::scalar_and_array(x, y) }";
        let linked = compile_with_options(
            source,
            empty_package("sample"),
            ResetBehavior::default(),
            &options,
        )?;
        xlsynth::IrPackage::parse_ir(&linked.to_string(), None)?;
        for (function, diagnostic) in [
            (
                "nominal",
                "DSLX function 'echo' specializes type parameter 'T' as 'Tag<u32:2>'; BX supports only bits and arrays of bits as type-valued arguments when importing concrete parametric structs with libxls v0.61",
            ),
            (
                "tuple_arg",
                "DSLX function 'echo' specializes type parameter 'T' as '(u8, u8)'; BX supports only bits and arrays of bits as type-valued arguments when importing concrete parametric structs with libxls v0.61",
            ),
            (
                "opaque",
                "libxls v0.61 does not expose the concrete method dispatch to 'through' inside DSLX function 'opaque'; BX cannot verify this specialization before converting imported parametric structs",
            ),
            (
                "mapped",
                "DSLX function 'mapped' uses a map callback or indirect invocation; BX cannot verify its target when converting imported parametric structs with libxls v0.61",
            ),
            (
                "grouped",
                "DSLX function 'grouped' uses a map callback or indirect invocation; BX cannot verify its target when converting imported parametric structs with libxls v0.61",
            ),
            (
                "collision",
                "DSLX specialization of 'value' can share a native IR symbol with 'value__2'; this combination is unsupported with libxls v0.61",
            ),
            (
                "ordinary_collision",
                "DSLX specialization of 'same' can share a native IR symbol with 'A__same'; this combination is unsupported with libxls v0.61",
            ),
        ] {
            let source =
                format!("use wire; block b(clk, x: wire::Item) -> u8 {{ wire::{function}(x) }}");
            let error = check_module_with_options(&parse_module(&source)?, &options)
                .err()
                .ok_or("expected unsupported DSLX specialization")?;
            assert_eq!(error, format!("compiling DSLX import 'wire': {diagnostic}"));
        }
        Ok(())
    }
}
