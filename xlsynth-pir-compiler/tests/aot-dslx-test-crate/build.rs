// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::PathBuf};

use xlsynth_pir_compiler::aot::{
    DslxConvertOptions, TypedDslxAotBuildSpec, TypedDslxAotPackageBuilder,
    build_native_typed_dslx_aot_package_metadata,
};

fn main() {
    let source_dir = PathBuf::from("src");
    let duplicate_widget_root = source_dir.join("dup");
    let common_dslx_options = DslxConvertOptions {
        additional_search_paths: vec![duplicate_widget_root.as_path(), source_dir.as_path()],
        force_implicit_token_calling_convention: true,
        ..DslxConvertOptions::default()
    };

    let gizmo_types = source_dir.join("gizmo_types.x");
    let parametric_forms = source_dir.join("parametric_forms.x");
    let invokes_and_loop = source_dir.join("invokes_and_loop.x");
    let events = source_dir.join("events.x");
    let parametric_imports = source_dir.join("parametric_imports.x");
    let parametric_lib = source_dir.join("parametric_lib.x");
    let duplicate_widget_frobber = duplicate_widget_root.join("frobber.x");
    let duplicate_foo_widget = duplicate_widget_root.join("foo/widget.x");
    let duplicate_bar_widget = duplicate_widget_root.join("bar/widget.x");
    let namespaced_doodle_types = source_dir.join("types/shared_types.x");
    let namespaced_doodle_echo = source_dir.join("foo/my_file.x");
    let namespaced_doodle_bump = source_dir.join("bar/your_file.x");

    let specs = vec![
        TypedDslxAotBuildSpec {
            name: "gizmo_frob",
            dslx_path: &gizmo_types,
            top: "frob_gizmo",
            dslx_options: common_dslx_options.clone(),
            type_module_paths: vec![],
        },
        TypedDslxAotBuildSpec {
            name: "parametric_forms",
            dslx_path: &parametric_forms,
            top: "exercise_parametric_forms",
            dslx_options: common_dslx_options.clone(),
            type_module_paths: vec![],
        },
        TypedDslxAotBuildSpec {
            name: "invokes_and_loop",
            dslx_path: &invokes_and_loop,
            top: "exercise_invokes_and_loop",
            dslx_options: common_dslx_options.clone(),
            type_module_paths: vec![],
        },
        TypedDslxAotBuildSpec {
            name: "events",
            dslx_path: &events,
            top: "exercise_events",
            dslx_options: common_dslx_options.clone(),
            type_module_paths: vec![],
        },
        TypedDslxAotBuildSpec {
            name: "parametric_imports",
            dslx_path: &parametric_imports,
            top: "exercise_parametric_imports",
            dslx_options: common_dslx_options.clone(),
            type_module_paths: vec![parametric_lib.as_path()],
        },
        TypedDslxAotBuildSpec {
            name: "duplicate_widget",
            dslx_path: &duplicate_widget_frobber,
            top: "frob_widget",
            dslx_options: common_dslx_options.clone(),
            type_module_paths: vec![
                duplicate_foo_widget.as_path(),
                duplicate_bar_widget.as_path(),
            ],
        },
        TypedDslxAotBuildSpec {
            name: "namespaced_package_echo",
            dslx_path: &namespaced_doodle_echo,
            top: "echo_doodle",
            dslx_options: common_dslx_options.clone(),
            type_module_paths: vec![namespaced_doodle_types.as_path()],
        },
        TypedDslxAotBuildSpec {
            name: "namespaced_package_bump",
            dslx_path: &namespaced_doodle_bump,
            top: "bump_doodle",
            dslx_options: common_dslx_options,
            type_module_paths: vec![namespaced_doodle_types.as_path()],
        },
    ];
    let metadata = build_native_typed_dslx_aot_package_metadata(&specs)
        .expect("native DSLX AOT package metadata generation should succeed");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let metadata_file = out_dir.join("native_dslx_tests_aot_metadata.json");
    fs::write(
        metadata_file,
        metadata
            .to_json_pretty()
            .expect("native DSLX AOT package metadata should serialize"),
    )
    .expect("native DSLX AOT package metadata should write");
    let mut package_builder = TypedDslxAotPackageBuilder::new("native_dslx_tests");
    for spec in specs {
        package_builder = package_builder.add_entrypoint(spec);
    }
    package_builder
        .build()
        .expect("native DSLX AOT package compilation should succeed");

    println!("cargo:rerun-if-changed=build.rs");
}
