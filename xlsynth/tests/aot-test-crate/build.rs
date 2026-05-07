// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use xlsynth::aot_builder::{
    emit_aot_module_from_ir_text, emit_typed_dslx_aot_module_from_file, AotBuildSpec,
    TypedDslxAotBuildSpec, TypedDslxAotPackageBuilder,
};
use xlsynth::DslxConvertOptions;

fn main() {
    // Describes one IR-only AOT wrapper generated for integration tests.
    struct AotCase {
        name: &'static str,
        top: &'static str,
        env_var: &'static str,
        ir_text: &'static str,
    }

    let add_one_ir = r#"package aot_tests

top fn add_one(x: bits[8]) -> bits[8] {
  one: bits[8] = literal(value=1)
  ret out: bits[8] = add(x, one)
}
"#;

    let add_inputs_ir = r#"package aot_tests

top fn add_inputs(a: bits[8], b: bits[8]) -> bits[8] {
  ret out: bits[8] = add(a, b)
}
"#;

    let compound_shapes_ir = r#"package aot_tests

top fn compound_shapes(lhs: (bits[8], bits[16][2]), rhs: (bits[16], bits[8][2])) -> (bits[16][2], (bits[8], bits[16]), bits[8][2]) {
  lhs_base: bits[8] = tuple_index(lhs, index=0)
  lhs_arr: bits[16][2] = tuple_index(lhs, index=1)
  rhs_base: bits[16] = tuple_index(rhs, index=0)
  rhs_arr: bits[8][2] = tuple_index(rhs, index=1)

  i0: bits[1] = literal(value=0)
  i1: bits[1] = literal(value=1)

  l0: bits[16] = array_index(lhs_arr, indices=[i0])
  l1: bits[16] = array_index(lhs_arr, indices=[i1])
  r0: bits[8] = array_index(rhs_arr, indices=[i0])
  r1: bits[8] = array_index(rhs_arr, indices=[i1])

  r0_wide: bits[16] = zero_ext(r0, new_bit_count=16)
  s0: bits[16] = add(l0, r0_wide)
  s1: bits[16] = add(l1, rhs_base)
  out_wide_arr: bits[16][2] = array(s0, s1)

  out_pair: (bits[8], bits[16]) = tuple(r1, rhs_base)

  out_narrow0: bits[8] = add(lhs_base, r0)
  out_narrow1: bits[8] = add(lhs_base, r1)
  out_narrow_arr: bits[8][2] = array(out_narrow0, out_narrow1)

  ret out: (bits[16][2], (bits[8], bits[16]), bits[8][2]) = tuple(out_wide_arr, out_pair, out_narrow_arr)
}
"#;

    let empty_tuple_ir = r#"package aot_tests

top fn make_empty_tuple() -> () {
  ret out: () = tuple()
}
"#;

    let wide_sizes_ir = r#"package aot_tests

top fn wide_sizes(input: (bits[1], bits[7], bits[8], bits[16], bits[32], bits[64], bits[65], bits[73], bits[127], bits[65][2])) -> (bits[1], bits[7], bits[8], bits[16], bits[32], bits[64], bits[65], bits[73], bits[127], bits[65][2]) {
  f0: bits[1] = tuple_index(input, index=0)
  f1: bits[7] = tuple_index(input, index=1)
  f2: bits[8] = tuple_index(input, index=2)
  f3: bits[16] = tuple_index(input, index=3)
  f4: bits[32] = tuple_index(input, index=4)
  f5: bits[64] = tuple_index(input, index=5)
  f6: bits[65] = tuple_index(input, index=6)
  f7: bits[73] = tuple_index(input, index=7)
  f8: bits[127] = tuple_index(input, index=8)
  f9: bits[65][2] = tuple_index(input, index=9)
  ret out: (bits[1], bits[7], bits[8], bits[16], bits[32], bits[64], bits[65], bits[73], bits[127], bits[65][2]) = tuple(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9)
}
"#;

    let large_array_tuple_ir = r#"package aot_tests

top fn large_array_tuple(input: (bits[8], bits[16][128])) -> (bits[8], bits[16][128]) {
  ret out: (bits[8], bits[16][128]) = identity(input)
}
"#;

    let wide_bits_tuple_ir = r#"package aot_tests

top fn wide_bits_tuple(input: (bits[8], bits[257])) -> (bits[8], bits[257]) {
  ret out: (bits[8], bits[257]) = identity(input)
}
"#;

    let trace_assert_ir = r#"package aot_tests

top fn trace_assert_pair(tok: token, pair: (bits[8], bits[8])) -> (bits[8], bits[8]) {
  a: bits[8] = tuple_index(pair, index=0)
  b: bits[8] = tuple_index(pair, index=1)
  sum: bits[8] = add(a, b)
  min_sum: bits[8] = literal(value=5)
  ok: bits[1] = uge(sum, min_sum)
  always_on: bits[1] = literal(value=1)
  asserted: token = assert(tok, ok, message="sum must be >= 5")
  traced: token = trace(asserted, always_on, format="sum: {}", data_operands=[sum])
  ret out: (bits[8], bits[8]) = tuple(sum, b)
}
"#;

    let cases = [
        AotCase {
            name: "add_one",
            top: "add_one",
            env_var: "XLSYNTH_AOT_ADD_ONE_RS",
            ir_text: add_one_ir,
        },
        AotCase {
            name: "add_inputs",
            top: "add_inputs",
            env_var: "XLSYNTH_AOT_ADD_INPUTS_RS",
            ir_text: add_inputs_ir,
        },
        AotCase {
            name: "compound_shapes",
            top: "compound_shapes",
            env_var: "XLSYNTH_AOT_COMPOUND_SHAPES_RS",
            ir_text: compound_shapes_ir,
        },
        AotCase {
            name: "empty_tuple",
            top: "make_empty_tuple",
            env_var: "XLSYNTH_AOT_EMPTY_TUPLE_RS",
            ir_text: empty_tuple_ir,
        },
        AotCase {
            name: "wide_sizes",
            top: "wide_sizes",
            env_var: "XLSYNTH_AOT_WIDE_SIZES_RS",
            ir_text: wide_sizes_ir,
        },
        AotCase {
            name: "large_array_tuple",
            top: "large_array_tuple",
            env_var: "XLSYNTH_AOT_LARGE_ARRAY_TUPLE_RS",
            ir_text: large_array_tuple_ir,
        },
        AotCase {
            name: "wide_bits_tuple",
            top: "wide_bits_tuple",
            env_var: "XLSYNTH_AOT_WIDE_BITS_TUPLE_RS",
            ir_text: wide_bits_tuple_ir,
        },
        AotCase {
            name: "trace_assert",
            top: "trace_assert_pair",
            env_var: "XLSYNTH_AOT_TRACE_ASSERT_RS",
            ir_text: trace_assert_ir,
        },
    ];

    for case in cases {
        let output = emit_aot_module_from_ir_text(&AotBuildSpec {
            name: case.name,
            ir_text: case.ir_text,
            top: case.top,
        })
        .unwrap_or_else(|err| panic!("AOT compile for {} should succeed: {}", case.top, err));
        println!(
            "cargo:rustc-env={}={}",
            case.env_var,
            output.rust_file.display()
        );
    }

    // Typed DSLX cases exercise public signatures that use generated bridge
    // types instead of structural wrapper aliases.
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let widget_dslx_path = manifest_dir.join("src/widget_types.x");
    let widget_output = emit_typed_dslx_aot_module_from_file(&TypedDslxAotBuildSpec {
        name: "widget_frob",
        dslx_path: &widget_dslx_path,
        top: "frob_widget",
        dslx_options: DslxConvertOptions::default(),
        type_module_paths: vec![],
    })
    .unwrap_or_else(|err| panic!("widget-frob typed DSLX AOT compile should succeed: {}", err));
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_WIDGET_FROB_RS={}",
        widget_output.rust_file.display()
    );

    let self_alias_dslx_path = manifest_dir.join("src/self_alias_widget.x");
    let self_alias_output = emit_typed_dslx_aot_module_from_file(&TypedDslxAotBuildSpec {
        name: "self_alias_widget",
        dslx_path: &self_alias_dslx_path,
        top: "echo_widget",
        dslx_options: DslxConvertOptions::default(),
        type_module_paths: vec![],
    })
    .unwrap_or_else(|err| panic!("self-alias widget AOT compile should succeed: {}", err));
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_SELF_ALIAS_WIDGET_RS={}",
        self_alias_output.rust_file.display()
    );

    let parametric_box_dslx_path = manifest_dir.join("src/parametric_box.x");
    let parametric_box_output = emit_typed_dslx_aot_module_from_file(&TypedDslxAotBuildSpec {
        name: "parametric_box",
        dslx_path: &parametric_box_dslx_path,
        top: "echo_box",
        dslx_options: DslxConvertOptions::default(),
        type_module_paths: vec![],
    })
    .unwrap_or_else(|err| panic!("parametric box AOT compile should succeed: {}", err));
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_PARAMETRIC_BOX_RS={}",
        parametric_box_output.rust_file.display()
    );

    let parametric_src_dir = manifest_dir.join("src");
    let parametric_lib_dslx_path = parametric_src_dir.join("parametric_lib.x");
    let parametric_cases = [
        (
            "parametric_shapes",
            "src/parametric_shapes.x",
            "exercise_parametric_shapes",
            "XLSYNTH_AOT_PARAMETRIC_SHAPES_RS",
            Vec::new(),
        ),
        (
            "parametric_arrays",
            "src/parametric_arrays.x",
            "exercise_parametric_arrays",
            "XLSYNTH_AOT_PARAMETRIC_ARRAYS_RS",
            Vec::new(),
        ),
        (
            "parametric_values",
            "src/parametric_values.x",
            "exercise_parametric_values",
            "XLSYNTH_AOT_PARAMETRIC_VALUES_RS",
            Vec::new(),
        ),
        (
            "parametric_imports",
            "src/parametric_imports.x",
            "exercise_parametric_imports",
            "XLSYNTH_AOT_PARAMETRIC_IMPORTS_RS",
            vec![parametric_lib_dslx_path.as_path()],
        ),
    ];
    for (name, path, top, env_var, type_module_paths) in parametric_cases {
        let dslx_path = manifest_dir.join(path);
        let output = emit_typed_dslx_aot_module_from_file(&TypedDslxAotBuildSpec {
            name,
            dslx_path: &dslx_path,
            top,
            dslx_options: DslxConvertOptions {
                additional_search_paths: vec![parametric_src_dir.as_path()],
                ..DslxConvertOptions::default()
            },
            type_module_paths,
        })
        .unwrap_or_else(|err| panic!("{} typed DSLX AOT compile should succeed: {}", name, err));
        println!("cargo:rustc-env={}={}", env_var, output.rust_file.display());
    }

    let dup_root = manifest_dir.join("src/dup");
    let dup_frobber_path = dup_root.join("frobber.x");
    let dup_foo_widget_path = dup_root.join("foo/widget.x");
    let dup_bar_widget_path = dup_root.join("bar/widget.x");
    let dup_output = emit_typed_dslx_aot_module_from_file(&TypedDslxAotBuildSpec {
        name: "duplicate_widget",
        dslx_path: &dup_frobber_path,
        top: "frob_widget",
        dslx_options: DslxConvertOptions {
            additional_search_paths: vec![dup_root.as_path()],
            ..DslxConvertOptions::default()
        },
        type_module_paths: vec![dup_foo_widget_path.as_path(), dup_bar_widget_path.as_path()],
    })
    .unwrap_or_else(|err| {
        panic!(
            "duplicate-widget typed DSLX AOT compile should succeed: {}",
            err
        )
    });
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_DUPLICATE_WIDGET_RS={}",
        dup_output.rust_file.display()
    );

    let shared_widget_types_dslx_path = manifest_dir.join("src/shared_widget_types.x");
    let shared_widget_echo_dslx_path = manifest_dir.join("src/shared_widget_echo.x");
    let shared_widget_bump_dslx_path = manifest_dir.join("src/shared_widget_bump.x");
    let shared_widget_src_dir = manifest_dir.join("src");
    let shared_package_output = TypedDslxAotPackageBuilder::new("shared_widget_package")
        .add_entrypoint(TypedDslxAotBuildSpec {
            name: "echo_widget",
            dslx_path: &shared_widget_echo_dslx_path,
            top: "echo_widget",
            dslx_options: DslxConvertOptions {
                additional_search_paths: vec![shared_widget_src_dir.as_path()],
                ..DslxConvertOptions::default()
            },
            type_module_paths: vec![shared_widget_types_dslx_path.as_path()],
        })
        .add_entrypoint(TypedDslxAotBuildSpec {
            name: "bump_widget",
            dslx_path: &shared_widget_bump_dslx_path,
            top: "bump_widget",
            dslx_options: DslxConvertOptions {
                additional_search_paths: vec![shared_widget_src_dir.as_path()],
                ..DslxConvertOptions::default()
            },
            type_module_paths: vec![shared_widget_types_dslx_path.as_path()],
        })
        .build()
        .unwrap_or_else(|err| {
            panic!(
                "shared-widget typed DSLX AOT package compile should succeed: {}",
                err
            )
        });
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_SHARED_WIDGET_PACKAGE_RS={}",
        shared_package_output.rust_file.display()
    );

    println!("cargo:rerun-if-changed=build.rs");
}
