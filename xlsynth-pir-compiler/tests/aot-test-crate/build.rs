// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use xlsynth_pir_compiler::aot::{
    AotBuildSpec, DslxConvertOptions, TypedDslxAotBuildSpec, TypedDslxAotPackageBuilder,
    emit_aot_module_from_dslx_file, emit_aot_module_from_pir_text,
};

struct AotCase {
    name: &'static str,
    top: &'static str,
    env_var: &'static str,
    pir_text: &'static str,
}

fn main() {
    let cases = [
        AotCase {
            name: "add_one",
            top: "add_one",
            env_var: "XLSYNTH_PIR_AOT_ADD_ONE_RS",
            pir_text: r#"package native_aot_tests

fn add_one(x: bits[42] id=1) -> bits[42] {
  one: bits[42] = literal(value=1, id=2)
  ret out: bits[42] = add(x, one, id=3)
}
"#,
        },
        AotCase {
            name: "add_inputs",
            top: "add_inputs",
            env_var: "XLSYNTH_PIR_AOT_ADD_INPUTS_RS",
            pir_text: r#"package native_aot_tests

fn add_inputs(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret out: bits[8] = add(a, b, id=3)
}
"#,
        },
        AotCase {
            name: "compound_shapes",
            top: "compound_shapes",
            env_var: "XLSYNTH_PIR_AOT_COMPOUND_SHAPES_RS",
            pir_text: r#"package native_aot_tests

fn compound_shapes(input: (bits[42], bits[65][2]) id=1, increment: bits[42] id=2) -> (bits[42], bits[65][2]) {
  base: bits[42] = tuple_index(input, index=0, id=3)
  limbs: bits[65][2] = tuple_index(input, index=1, id=4)
  sum: bits[42] = add(base, increment, id=5)
  ret out: (bits[42], bits[65][2]) = tuple(sum, limbs, id=6)
}
"#,
        },
        AotCase {
            name: "empty_tuple",
            top: "empty_tuple",
            env_var: "XLSYNTH_PIR_AOT_EMPTY_TUPLE_RS",
            pir_text: r#"package native_aot_tests

fn empty_tuple() -> () {
  ret out: () = tuple(id=1)
}
"#,
        },
        AotCase {
            name: "events",
            top: "events",
            env_var: "XLSYNTH_PIR_AOT_EVENTS_RS",
            pir_text: r#"package native_aot_tests

fn events(x: bits[8] id=1, ok: bits[1] id=2, emit: bits[1] id=3) -> bits[8] {
  t: token = after_all(id=4)
  cv: () = cover(emit, label="covered", id=5)
  a: token = assert(t, ok, message="bad condition", label="A", id=6)
  tr: token = trace(a, emit, format="x={}", data_operands=[x], id=7)
  ret out: bits[8] = identity(x, id=8)
}
"#,
        },
    ];

    for case in cases {
        let output = emit_aot_module_from_pir_text(&AotBuildSpec {
            name: case.name,
            pir_text: case.pir_text,
            top: case.top,
        })
        .unwrap_or_else(|error| {
            panic!(
                "native PIR AOT compilation for {} should succeed: {error}",
                case.top
            )
        });
        println!(
            "cargo:rustc-env={}={}",
            case.env_var,
            output.rust_file.display()
        );
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let widget_types = manifest_dir.join("src/widget_types.x");
    let widget_output = emit_aot_module_from_dslx_file(&TypedDslxAotBuildSpec {
        name: "widget_frob",
        dslx_path: &widget_types,
        top: "frob_widget",
        dslx_options: DslxConvertOptions::default(),
        type_module_paths: vec![],
    })
    .expect("native widget DSLX AOT compilation should succeed");
    println!(
        "cargo:rustc-env=XLSYNTH_PIR_AOT_WIDGET_FROB_RS={}",
        widget_output.rust_file.display()
    );

    let source_dir = manifest_dir.join("src");
    let parametric_imports = source_dir.join("parametric_imports.x");
    let parametric_lib = source_dir.join("parametric_lib.x");
    let parametric_output = emit_aot_module_from_dslx_file(&TypedDslxAotBuildSpec {
        name: "parametric_imports",
        dslx_path: &parametric_imports,
        top: "exercise_parametric_imports",
        dslx_options: DslxConvertOptions {
            additional_search_paths: vec![source_dir.as_path()],
            ..DslxConvertOptions::default()
        },
        type_module_paths: vec![parametric_lib.as_path()],
    })
    .expect("native imported parametric DSLX AOT compilation should succeed");
    println!(
        "cargo:rustc-env=XLSYNTH_PIR_AOT_PARAMETRIC_IMPORTS_RS={}",
        parametric_output.rust_file.display()
    );

    let shared_widget_echo = source_dir.join("shared_widget_echo.x");
    let shared_widget_types = source_dir.join("shared_widget_types.x");
    let shared_output = emit_aot_module_from_dslx_file(&TypedDslxAotBuildSpec {
        name: "shared_widget_echo",
        dslx_path: &shared_widget_echo,
        top: "echo_widget",
        dslx_options: DslxConvertOptions {
            additional_search_paths: vec![source_dir.as_path()],
            ..DslxConvertOptions::default()
        },
        type_module_paths: vec![shared_widget_types.as_path()],
    })
    .expect("native shared nominal DSLX AOT compilation should succeed");
    println!(
        "cargo:rustc-env=XLSYNTH_PIR_AOT_SHARED_WIDGET_ECHO_RS={}",
        shared_output.rust_file.display()
    );

    let shared_widget_bump = source_dir.join("shared_widget_bump.x");
    let shared_package = TypedDslxAotPackageBuilder::new("shared_widget_package")
        .add_entrypoint(TypedDslxAotBuildSpec {
            name: "shared_package_echo",
            dslx_path: &shared_widget_echo,
            top: "echo_widget",
            dslx_options: DslxConvertOptions {
                additional_search_paths: vec![source_dir.as_path()],
                ..DslxConvertOptions::default()
            },
            type_module_paths: vec![shared_widget_types.as_path()],
        })
        .add_entrypoint(TypedDslxAotBuildSpec {
            name: "shared_package_bump",
            dslx_path: &shared_widget_bump,
            top: "bump_widget",
            dslx_options: DslxConvertOptions {
                additional_search_paths: vec![source_dir.as_path()],
                ..DslxConvertOptions::default()
            },
            type_module_paths: vec![shared_widget_types.as_path()],
        })
        .build()
        .expect("native shared nominal DSLX AOT package compilation should succeed");
    println!(
        "cargo:rustc-env=XLSYNTH_PIR_AOT_SHARED_WIDGET_PACKAGE_RS={}",
        shared_package.rust_file.display()
    );

    println!("cargo:rerun-if-changed=build.rs");
}
