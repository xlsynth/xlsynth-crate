// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use xlsynth_pir_compiler::aot::{
    DslxConvertOptions, TypedDslxAotBuildSpec, TypedDslxAotPackageBuilder,
    emit_aot_module_from_dslx_file,
};

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let gizmo_types = manifest_dir.join("src/gizmo_types.x");
    emit_aot_module_from_dslx_file(&TypedDslxAotBuildSpec {
        name: "gizmo_frob",
        dslx_path: &gizmo_types,
        top: "frob_gizmo",
        dslx_options: DslxConvertOptions::default(),
        type_module_paths: vec![],
    })
    .expect("native gizmo DSLX AOT compilation should succeed");

    let source_dir = manifest_dir.join("src");
    let parametric_forms = source_dir.join("parametric_forms.x");
    emit_aot_module_from_dslx_file(&TypedDslxAotBuildSpec {
        name: "parametric_forms",
        dslx_path: &parametric_forms,
        top: "exercise_parametric_forms",
        dslx_options: DslxConvertOptions::default(),
        type_module_paths: vec![],
    })
    .expect("native parametric forms DSLX AOT compilation should succeed");

    let parametric_imports = source_dir.join("parametric_imports.x");
    let parametric_lib = source_dir.join("parametric_lib.x");
    emit_aot_module_from_dslx_file(&TypedDslxAotBuildSpec {
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

    let duplicate_widget_root = source_dir.join("dup");
    let duplicate_widget_frobber = duplicate_widget_root.join("frobber.x");
    let duplicate_foo_widget = duplicate_widget_root.join("foo/widget.x");
    let duplicate_bar_widget = duplicate_widget_root.join("bar/widget.x");
    emit_aot_module_from_dslx_file(&TypedDslxAotBuildSpec {
        name: "duplicate_widget",
        dslx_path: &duplicate_widget_frobber,
        top: "frob_widget",
        dslx_options: DslxConvertOptions {
            additional_search_paths: vec![duplicate_widget_root.as_path()],
            ..DslxConvertOptions::default()
        },
        type_module_paths: vec![
            duplicate_foo_widget.as_path(),
            duplicate_bar_widget.as_path(),
        ],
    })
    .expect("native duplicate-widget DSLX AOT compilation should succeed");

    let namespaced_doodle_types = source_dir.join("types/shared_types.x");
    let namespaced_doodle_echo = source_dir.join("foo/my_file.x");
    let namespaced_doodle_bump = source_dir.join("bar/your_file.x");
    TypedDslxAotPackageBuilder::new("namespaced_doodle_package")
        .add_entrypoint(TypedDslxAotBuildSpec {
            name: "namespaced_package_echo",
            dslx_path: &namespaced_doodle_echo,
            top: "echo_doodle",
            dslx_options: DslxConvertOptions {
                additional_search_paths: vec![source_dir.as_path()],
                ..DslxConvertOptions::default()
            },
            type_module_paths: vec![namespaced_doodle_types.as_path()],
        })
        .add_entrypoint(TypedDslxAotBuildSpec {
            name: "namespaced_package_bump",
            dslx_path: &namespaced_doodle_bump,
            top: "bump_doodle",
            dslx_options: DslxConvertOptions {
                additional_search_paths: vec![source_dir.as_path()],
                ..DslxConvertOptions::default()
            },
            type_module_paths: vec![namespaced_doodle_types.as_path()],
        })
        .build()
        .expect("native namespaced doodle DSLX AOT package compilation should succeed");

    println!("cargo:rerun-if-changed=build.rs");
}
