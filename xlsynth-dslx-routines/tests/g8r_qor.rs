// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use pretty_assertions::assert_eq;
use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;

#[derive(Clone, Debug, PartialEq, Eq)]
struct QorRow {
    width: u32,
    variant: &'static str,
    live_nodes: usize,
    deepest_path: usize,
}

#[derive(Clone, Copy)]
enum AddWrapperKind {
    SumOnly,
    WithCarry,
}

/// Returns the local DSLX stdlib path when the XLS toolchain is available.
fn dslx_stdlib_path() -> Option<PathBuf> {
    let tool_path = std::env::var_os("XLSYNTH_TOOLS")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join("opt/xlsynth/latest"))
        })?;
    let stdlib_path = tool_path.join("xls/dslx/stdlib");
    stdlib_path.exists().then_some(stdlib_path)
}

/// Converts a generated DSLX wrapper module to unoptimized IR text.
fn wrapper_ir_text(module_name: &str, source: &str) -> String {
    let dslx_stdlib_path = dslx_stdlib_path();
    let result = xlsynth::convert_dslx_to_ir_text(
        source,
        Path::new(module_name),
        &xlsynth::DslxConvertOptions {
            dslx_stdlib_path: dslx_stdlib_path.as_deref(),
            additional_search_paths: vec![xlsynth_dslx_routines::dslx_dir()],
            enable_warnings: None,
            disable_warnings: None,
            force_implicit_token_calling_convention: false,
        },
    )
    .expect("DSLX wrapper should convert to IR");
    result.ir
}

/// Converts, optimizes, and lowers a generated wrapper to g8r gate stats.
fn gate_stats_for_wrapper(module_name: &str, source: &str, top: &str) -> SummaryStats {
    let ir_text = wrapper_ir_text(module_name, source);
    let ir_top = xlsynth::mangle_dslx_name(module_name.trim_end_matches(".x"), top)
        .expect("top should mangle");
    let ir_package =
        xlsynth::IrPackage::parse_ir(&ir_text, Some(&ir_top)).expect("IR should parse");
    let optimized_ir_package =
        xlsynth::optimize_ir(&ir_package, &ir_top).expect("IR should optimize");
    let optimized_ir_text = optimized_ir_package.to_string();
    let out = ir2gates::ir2gates_from_ir_text(
        &optimized_ir_text,
        Some(&ir_top),
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            enable_rewrite_mask_low: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .expect("g8r should lower wrapper IR");
    get_summary_stats(&out.gatify_output.gate_fn)
}

/// Generates a wrapper that imports and calls a canonical CLZ routine.
fn clz_wrapper_source(width: u32, routine: &str, group: Option<u32>) -> String {
    let invocation = match group {
        Some(group) => format!("clz_variants::{routine}<u32:{width}, u32:{group}>(x)"),
        None => format!("clz_variants::{routine}(x)"),
    };
    format!(
        r#"import clz_variants;

const N: u32 = u32:{width};

fn top(x: uN[N]) -> uN[N] {{
  {invocation}
}}
"#
    )
}

/// Generates a wrapper that imports and calls a canonical add routine.
fn add_wrapper_source(
    width: u32,
    routine: &str,
    group: Option<u32>,
    kind: AddWrapperKind,
) -> String {
    let (args, params, return_type) = match kind {
        AddWrapperKind::SumOnly => ("x, y", "x: uN[N], y: uN[N]", "uN[N]"),
        AddWrapperKind::WithCarry => (
            "x, y, carry_in",
            "x: uN[N], y: uN[N], carry_in: u1",
            "(uN[N], u1)",
        ),
    };
    let invocation = match group {
        Some(group) => format!("add_variants::{routine}<u32:{width}, u32:{group}>({args})"),
        None => format!("add_variants::{routine}({args})"),
    };
    format!(
        r#"import add_variants;

const N: u32 = u32:{width};

fn top({params}) -> {return_type} {{
  {invocation}
}}
"#
    )
}

/// Computes QoR rows for the selected CLZ routines and tunings.
fn gather_clz_qor_rows() -> Vec<QorRow> {
    let mut rows = Vec::new();
    for width in [8u32, 16, 32] {
        let cases = [
            ("linear", clz_wrapper_source(width, "clz_linear", None)),
            (
                "prefix_mask",
                clz_wrapper_source(width, "clz_prefix_mask", None),
            ),
            (
                "grouped_default",
                clz_wrapper_source(width, "clz_grouped", None),
            ),
            (
                "grouped_tuned",
                match width {
                    8 => clz_wrapper_source(width, "clz_grouped", Some(2)),
                    16 => clz_wrapper_source(width, "clz_grouped", Some(4)),
                    32 => clz_wrapper_source(width, "clz_grouped", Some(8)),
                    _ => unreachable!(),
                },
            ),
        ];
        for (variant, source) in cases {
            let stats = gate_stats_for_wrapper("clz_qor.x", &source, "top");
            rows.push(QorRow {
                width,
                variant,
                live_nodes: stats.live_nodes,
                deepest_path: stats.deepest_path,
            });
        }
    }
    rows
}

/// Computes QoR rows for the selected add routines and tunings.
fn gather_add_qor_rows() -> Vec<QorRow> {
    let mut rows = Vec::new();
    for width in [8u32, 16, 32] {
        let cases = [
            (
                "ripple",
                add_wrapper_source(width, "add_ripple", None, AddWrapperKind::SumOnly),
            ),
            (
                "ripple_with_carry",
                add_wrapper_source(
                    width,
                    "add_ripple_with_carry",
                    None,
                    AddWrapperKind::WithCarry,
                ),
            ),
            (
                "prefix",
                add_wrapper_source(width, "add_prefix", None, AddWrapperKind::SumOnly),
            ),
            (
                "prefix_with_carry",
                add_wrapper_source(
                    width,
                    "add_prefix_with_carry",
                    None,
                    AddWrapperKind::WithCarry,
                ),
            ),
            (
                "carry_select_default",
                add_wrapper_source(width, "add_carry_select", None, AddWrapperKind::SumOnly),
            ),
            (
                "carry_select_with_carry_default",
                add_wrapper_source(
                    width,
                    "add_carry_select_with_carry",
                    None,
                    AddWrapperKind::WithCarry,
                ),
            ),
            (
                "carry_select_tuned",
                match width {
                    8 => add_wrapper_source(
                        width,
                        "add_carry_select",
                        Some(2),
                        AddWrapperKind::SumOnly,
                    ),
                    16 => add_wrapper_source(
                        width,
                        "add_carry_select",
                        Some(4),
                        AddWrapperKind::SumOnly,
                    ),
                    32 => add_wrapper_source(
                        width,
                        "add_carry_select",
                        Some(8),
                        AddWrapperKind::SumOnly,
                    ),
                    _ => unreachable!(),
                },
            ),
            (
                "carry_select_with_carry_tuned",
                match width {
                    8 => add_wrapper_source(
                        width,
                        "add_carry_select_with_carry",
                        Some(2),
                        AddWrapperKind::WithCarry,
                    ),
                    16 => add_wrapper_source(
                        width,
                        "add_carry_select_with_carry",
                        Some(4),
                        AddWrapperKind::WithCarry,
                    ),
                    32 => add_wrapper_source(
                        width,
                        "add_carry_select_with_carry",
                        Some(8),
                        AddWrapperKind::WithCarry,
                    ),
                    _ => unreachable!(),
                },
            ),
        ];
        for (variant, source) in cases {
            let stats = gate_stats_for_wrapper("add_qor.x", &source, "top");
            rows.push(QorRow {
                width,
                variant,
                live_nodes: stats.live_nodes,
                deepest_path: stats.deepest_path,
            });
        }
    }
    rows
}

#[test]
fn clz_variants_g8r_qor_snapshot() {
    let _ = env_logger::builder().is_test(true).try_init();
    let got = gather_clz_qor_rows();
    #[rustfmt::skip]
    let want: &[QorRow] = &[
        QorRow { width: 8, variant: "linear", live_nodes: 43, deepest_path: 12 },
        QorRow { width: 8, variant: "prefix_mask", live_nodes: 82, deepest_path: 18 },
        QorRow { width: 8, variant: "grouped_default", live_nodes: 40, deepest_path: 8 },
        QorRow { width: 8, variant: "grouped_tuned", live_nodes: 36, deepest_path: 9 },
        QorRow { width: 16, variant: "linear", live_nodes: 110, deepest_path: 24 },
        QorRow { width: 16, variant: "prefix_mask", live_nodes: 196, deepest_path: 32 },
        QorRow { width: 16, variant: "grouped_default", live_nodes: 90, deepest_path: 12 },
        QorRow { width: 16, variant: "grouped_tuned", live_nodes: 90, deepest_path: 12 },
        QorRow { width: 32, variant: "linear", live_nodes: 269, deepest_path: 48 },
        QorRow { width: 32, variant: "prefix_mask", live_nodes: 433, deepest_path: 55 },
        QorRow { width: 32, variant: "grouped_default", live_nodes: 197, deepest_path: 20 },
        QorRow { width: 32, variant: "grouped_tuned", live_nodes: 220, deepest_path: 18 },
    ];
    assert_eq!(got.as_slice(), want);
}

#[test]
fn add_variants_g8r_qor_snapshot() {
    let _ = env_logger::builder().is_test(true).try_init();
    let got = gather_add_qor_rows();
    #[rustfmt::skip]
    let want: &[QorRow] = &[
        QorRow { width: 8, variant: "ripple", live_nodes: 92, deepest_path: 24 },
        QorRow { width: 8, variant: "ripple_with_carry", live_nodes: 105, deepest_path: 26 },
        QorRow { width: 8, variant: "prefix", live_nodes: 102, deepest_path: 12 },
        QorRow { width: 8, variant: "prefix_with_carry", live_nodes: 117, deepest_path: 14 },
        QorRow { width: 8, variant: "carry_select_default", live_nodes: 125, deepest_path: 14 },
        QorRow { width: 8, variant: "carry_select_with_carry_default", live_nodes: 185, deepest_path: 15 },
        QorRow { width: 8, variant: "carry_select_tuned", live_nodes: 127, deepest_path: 11 },
        QorRow { width: 8, variant: "carry_select_with_carry_tuned", live_nodes: 161, deepest_path: 13 },
        QorRow { width: 16, variant: "ripple", live_nodes: 196, deepest_path: 48 },
        QorRow { width: 16, variant: "ripple_with_carry", live_nodes: 209, deepest_path: 50 },
        QorRow { width: 16, variant: "prefix", live_nodes: 257, deepest_path: 15 },
        QorRow { width: 16, variant: "prefix_with_carry", live_nodes: 275, deepest_path: 17 },
        QorRow { width: 16, variant: "carry_select_default", live_nodes: 309, deepest_path: 17 },
        QorRow { width: 16, variant: "carry_select_with_carry_default", live_nodes: 369, deepest_path: 19 },
        QorRow { width: 16, variant: "carry_select_tuned", live_nodes: 309, deepest_path: 17 },
        QorRow { width: 16, variant: "carry_select_with_carry_tuned", live_nodes: 369, deepest_path: 19 },
        QorRow { width: 32, variant: "ripple", live_nodes: 404, deepest_path: 96 },
        QorRow { width: 32, variant: "ripple_with_carry", live_nodes: 417, deepest_path: 98 },
        QorRow { width: 32, variant: "prefix", live_nodes: 618, deepest_path: 17 },
        QorRow { width: 32, variant: "prefix_with_carry", live_nodes: 639, deepest_path: 19 },
        QorRow { width: 32, variant: "carry_select_default", live_nodes: 677, deepest_path: 25 },
        QorRow { width: 32, variant: "carry_select_with_carry_default", live_nodes: 737, deepest_path: 27 },
        QorRow { width: 32, variant: "carry_select_tuned", live_nodes: 673, deepest_path: 29 },
        QorRow { width: 32, variant: "carry_select_with_carry_tuned", live_nodes: 785, deepest_path: 31 },
    ];
    assert_eq!(got.as_slice(), want);
}
