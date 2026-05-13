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
            unsafe_gatify_gate_operation: false,
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

/// Generates a wrapper that imports and calls a canonical multiply routine.
fn mul_wrapper_source(width: u32, routine: &str, group: Option<u32>) -> String {
    let invocation = match group {
        Some(group) => format!("mul_variants::{routine}<u32:{width}, u32:{group}>(x, y)"),
        None => format!("mul_variants::{routine}(x, y)"),
    };
    format!(
        r#"import mul_variants;

const N: u32 = u32:{width};

fn top(x: uN[N], y: uN[N]) -> uN[N] {{
  {invocation}
}}
"#
    )
}

/// Generates a wrapper around one sequential add-state update.
fn add_seq_step_wrapper_source(width: u32) -> String {
    format!(
        r#"import add_seq_variants;

const N: u32 = u32:{width};

fn top(
  remaining_x: uN[N],
  remaining_y: uN[N],
  sum: uN[N],
  bit_mask: uN[N],
  carry: u1,
) -> (uN[N], uN[N], uN[N], uN[N], u1) {{
  let next = add_seq_variants::add_step(add_seq_variants::AddSeqState {{
    remaining_x,
    remaining_y,
    sum,
    bit_mask,
    carry,
  }});
  (next.remaining_x, next.remaining_y, next.sum, next.bit_mask, next.carry)
}}
"#
    )
}

/// Generates a wrapper around one sequential multiply-state update.
fn mul_seq_step_wrapper_source(width: u32) -> String {
    format!(
        r#"import mul_seq_variants;

const N: u32 = u32:{width};

fn top(
  multiplicand: uN[N],
  multiplier: uN[N],
  product: uN[N],
) -> (uN[N], uN[N], uN[N]) {{
  let next = mul_seq_variants::mul_step(mul_seq_variants::MulSeqState {{
    multiplicand,
    multiplier,
    product,
  }});
  (next.multiplicand, next.multiplier, next.product)
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

/// Computes QoR rows for the selected multiply routines and tunings.
fn gather_mul_qor_rows() -> Vec<QorRow> {
    let mut rows = Vec::new();
    for width in [8u32, 16, 32] {
        let cases = [
            (
                "shift_add",
                mul_wrapper_source(width, "mul_shift_add", None),
            ),
            (
                "shift_add_ripple",
                mul_wrapper_source(width, "mul_shift_add_ripple", None),
            ),
            (
                "shift_add_prefix",
                mul_wrapper_source(width, "mul_shift_add_prefix", None),
            ),
            (
                "shift_add_carry_select_default",
                mul_wrapper_source(width, "mul_shift_add_carry_select", None),
            ),
            (
                "shift_add_carry_select_tuned",
                match width {
                    8 => mul_wrapper_source(width, "mul_shift_add_carry_select", Some(2)),
                    16 => mul_wrapper_source(width, "mul_shift_add_carry_select", Some(4)),
                    32 => mul_wrapper_source(width, "mul_shift_add_carry_select", Some(8)),
                    _ => unreachable!(),
                },
            ),
        ];
        for (variant, source) in cases {
            let stats = gate_stats_for_wrapper("mul_qor.x", &source, "top");
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

/// Computes QoR rows for one step of the selected sequential routines.
fn gather_seq_step_qor_rows() -> Vec<QorRow> {
    let mut rows = Vec::new();
    for width in [8u32, 16, 32] {
        let cases = [
            ("add_step", add_seq_step_wrapper_source(width)),
            ("mul_step", mul_seq_step_wrapper_source(width)),
        ];
        for (variant, source) in cases {
            let stats = gate_stats_for_wrapper("seq_step_qor.x", &source, "top");
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

#[test]
fn mul_variants_g8r_qor_snapshot() {
    let _ = env_logger::builder().is_test(true).try_init();
    let got = gather_mul_qor_rows();
    #[rustfmt::skip]
    let want: &[QorRow] = &[
        QorRow { width: 8, variant: "shift_add", live_nodes: 262, deepest_path: 24 },
        QorRow { width: 8, variant: "shift_add_ripple", live_nodes: 276, deepest_path: 40 },
        QorRow { width: 8, variant: "shift_add_prefix", live_nodes: 284, deepest_path: 38 },
        QorRow { width: 8, variant: "shift_add_carry_select_default", live_nodes: 369, deepest_path: 36 },
        QorRow { width: 8, variant: "shift_add_carry_select_tuned", live_nodes: 356, deepest_path: 37 },
        QorRow { width: 16, variant: "shift_add", live_nodes: 1268, deepest_path: 43 },
        QorRow { width: 16, variant: "shift_add_ripple", live_nodes: 1308, deepest_path: 98 },
        QorRow { width: 16, variant: "shift_add_prefix", live_nodes: 1564, deepest_path: 103 },
        QorRow { width: 16, variant: "shift_add_carry_select_default", live_nodes: 2056, deepest_path: 97 },
        QorRow { width: 16, variant: "shift_add_carry_select_tuned", live_nodes: 2056, deepest_path: 97 },
        QorRow { width: 32, variant: "shift_add", live_nodes: 5698, deepest_path: 67 },
        QorRow { width: 32, variant: "shift_add_ripple", live_nodes: 5676, deepest_path: 212 },
        QorRow { width: 32, variant: "shift_add_prefix", live_nodes: 8029, deepest_path: 236 },
        QorRow { width: 32, variant: "shift_add_carry_select_default", live_nodes: 9650, deepest_path: 224 },
        QorRow { width: 32, variant: "shift_add_carry_select_tuned", live_nodes: 9640, deepest_path: 226 },
    ];
    assert_eq!(got.as_slice(), want);
}

#[test]
fn seq_variants_g8r_step_qor_snapshot() {
    let _ = env_logger::builder().is_test(true).try_init();
    let got = gather_seq_step_qor_rows();
    #[rustfmt::skip]
    let want: &[QorRow] = &[
        QorRow { width: 8, variant: "add_step", live_nodes: 61, deepest_path: 7 },
        QorRow { width: 8, variant: "mul_step", live_nodes: 103, deepest_path: 14 },
        QorRow { width: 16, variant: "add_step", live_nodes: 109, deepest_path: 7 },
        QorRow { width: 16, variant: "mul_step", live_nodes: 225, deepest_path: 18 },
        QorRow { width: 32, variant: "add_step", live_nodes: 205, deepest_path: 7 },
        QorRow { width: 32, variant: "mul_step", live_nodes: 475, deepest_path: 22 },
    ];
    assert_eq!(got.as_slice(), want);
}
