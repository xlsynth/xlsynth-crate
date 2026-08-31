// SPDX-License-Identifier: Apache-2.0

//! Stock-XLS discovery and independent public block-codegen comparison.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use xlsynth::external_tool::{ToolError, resolve_executable, run_checked_detailed};
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_g8r_fuzz::external_yosys::ExternalYosysContext;
use xlsynth_pir::ir::{Binop, NodePayload, Package, Type, Unop};

use crate::{emit, top_block};

const BLOCK_TO_VERILOG_PATH_ENV: &str = "XLSYNTH_BLOCK_TO_VERILOG_PATH";
const TOOLS_DIRECTORY_ENV: &str = "XLSYNTH_TOOLS";
const BLOCK_TO_VERILOG_NAME: &str = "block_to_verilog_main";

static STOCK_XLS_PATH: OnceLock<Result<PathBuf, String>> = OnceLock::new();
static STOCK_XLS_PREFLIGHT: OnceLock<Result<(), String>> = OnceLock::new();

/// Finds an optional stock-XLS block-to-Verilog executable once per campaign.
pub fn stock_xls_path() -> Option<&'static Path> {
    match STOCK_XLS_PATH.get_or_init(discover_stock_xls) {
        Ok(path) => Some(path.as_path()),
        Err(_) => None,
    }
}

/// Checks executable discovery and stock codegen before generating samples.
pub fn preflight_stock_xls() -> Result<(), String> {
    STOCK_XLS_PREFLIGHT
        .get_or_init(|| {
            let executable = STOCK_XLS_PATH
                .get_or_init(discover_stock_xls)
                .as_ref()
                .map_err(Clone::clone)?;
            run_stock_codegen(crate::references::COMBINATIONAL, executable)
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
        .clone()
}

/// Checks native/stock Icarus and native mapped gates against one PIR trace.
pub fn assert_stock_xls_semantics(
    package: &Package,
    stock_xls: &Path,
    context: &ExternalYosysContext,
    options: &BlockCodegenOptions,
    trace: &crate::semantics::Trace,
) -> Result<(), ToolError> {
    let ir = package.to_string();
    let native = emit(package, options);
    assert_eq!(
        native,
        emit(package, options),
        "native codegen is nondeterministic"
    );
    let reference = run_stock_codegen(&ir, stock_xls)?;
    crate::iverilog::assert_rtl_trace(
        package,
        &native,
        None,
        crate::iverilog::StateLayout::Packed,
        trace,
    )?;
    crate::iverilog::assert_rtl_trace(
        package,
        &reference,
        None,
        crate::iverilog::StateLayout::StockUnpacked,
        trace,
    )?;
    crate::mapped_semantics::assert_mapped_trace(package, &native, context, trace)
}

/// Identifies valid aggregate operations explicitly unsupported by stock XLS.
pub fn stock_xls_supports(package: &Package) -> bool {
    stock_xls_skip_reason(package).is_none()
}

/// Names known stock-XLS limitations without suppressing unexpected failures.
pub fn stock_xls_skip_reason(package: &Package) -> Option<&'static str> {
    let (block, _) = top_block(package);
    block.nodes.iter().find_map(|node| match &node.payload {
        NodePayload::Binop(Binop::Gate, ..) if node.ty.bit_count() == 0 => Some("zero-bit-gate"),
        NodePayload::Binop(Binop::Gate, ..) if matches!(node.ty, Type::Array(_)) => {
            Some("array-valued-gate")
        }
        NodePayload::ZeroExt { arg, .. } if block.get_node_ty(*arg).bit_count() == 0 => {
            Some("zero-extension-from-zero-bits")
        }
        NodePayload::Binop(Binop::Eq | Binop::Ne, lhs, _) => {
            (block.get_node(*lhs).ty.bit_count() == 0).then_some("zero-bit-comparison")
        }
        NodePayload::Unop(Unop::Identity, _) if matches!(node.ty, Type::Array(_)) => {
            Some("array-valued-identity")
        }
        _ => None,
    })
}

/// Runs the same bounded stock emitter during preflight and sample checks.
fn run_stock_codegen(ir: &str, executable: &Path) -> Result<String, ToolError> {
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let input = directory.path().join("input.ir");
    std::fs::write(&input, ir).map_err(|error| error.to_string())?;
    run_checked_detailed(
        Command::new(executable)
            // Icarus does not support XLS's unpacked-array assignment patterns.
            // Stock Verilog mode emits equivalent element assignments; native RTL
            // is compiled as SystemVerilog without rewriting or parsing it here.
            .arg("--use_system_verilog=false")
            // Keep narrow computations out of genvar-sized comparisons. Stock
            // XLS can otherwise widen an inlined array-update index expression.
            // Native materialization options remain independently randomized.
            .arg("--separate_lines=true")
            .arg("--generator=combinational")
            .arg(&input),
        directory.path(),
        "stock-xls",
        Duration::from_secs(60),
    )
}

/// Resolves explicit executable, configured tools directory, then PATH.
fn discover_stock_xls() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os(BLOCK_TO_VERILOG_PATH_ENV) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return resolve_executable(&path);
        }
        return Err(format!(
            "{BLOCK_TO_VERILOG_PATH_ENV} is not a file: {}",
            path.display()
        ));
    }

    if let Some(tools) = std::env::var_os(TOOLS_DIRECTORY_ENV) {
        let path = Path::new(&tools).join(BLOCK_TO_VERILOG_NAME);
        if path.is_file() {
            return resolve_executable(&path);
        }
    }

    if let Some(path) = std::env::var_os("PATH") {
        for directory in std::env::split_paths(&path) {
            let executable = directory.join(BLOCK_TO_VERILOG_NAME);
            if executable.is_file() {
                return resolve_executable(&executable);
            }
        }
    }

    Err(format!(
        "set {BLOCK_TO_VERILOG_PATH_ENV}, configure {TOOLS_DIRECTORY_ENV}, or add {BLOCK_TO_VERILOG_NAME} to PATH"
    ))
}

#[cfg(test)]
mod tests {
    use crate::{parse_reference, references};
    use xlsynth_codegen::BlockCodegenOptions;
    use xlsynth_g8r_fuzz::external_yosys::required_external_yosys_context;

    use super::{assert_stock_xls_semantics, stock_xls_path, stock_xls_supports};

    #[test]
    fn public_reference_matches_stock_xls() {
        let executable =
            stock_xls_path().expect("selected stock-XLS checks require block_to_verilog_main");
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        for reference in [
            references::COMBINATIONAL,
            references::AGGREGATE,
            references::SEQUENTIAL,
        ] {
            let package = parse_reference(reference);
            assert_stock_xls_semantics(
                &package,
                executable,
                context,
                &BlockCodegenOptions::default(),
                &crate::semantics::Trace::for_package(&package),
            )
            .unwrap();
        }
    }

    #[test]
    fn narrow_div_mod_matches_all_oracles() {
        let executable =
            stock_xls_path().expect("selected stock-XLS checks require block_to_verilog_main");
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        for width in [1, 8, 16] {
            let package = parse_reference(&format!(
                r#"package public_div_mod
top block div_mod(lhs: bits[{width}], rhs: bits[{width}], out: bits[{result_width}]) {{
  lhs: bits[{width}] = input_port(name=lhs, id=1)
  rhs: bits[{width}] = input_port(name=rhs, id=2)
  uq: bits[{width}] = udiv(lhs, rhs, id=3)
  sq: bits[{width}] = sdiv(lhs, rhs, id=4)
  ur: bits[{width}] = umod(lhs, rhs, id=5)
  sr: bits[{width}] = smod(lhs, rhs, id=6)
  results: bits[{result_width}] = concat(uq, sq, ur, sr, id=7)
  out: () = output_port(results, name=out, id=8)
}}
"#,
                result_width = 4 * width
            ));
            let start = std::time::Instant::now();
            assert_stock_xls_semantics(
                &package,
                executable,
                context,
                &BlockCodegenOptions::default(),
                &crate::semantics::Trace::for_package(&package),
            )
            .unwrap();
            eprintln!(
                "{width}-bit div/mod, all four oracles: {:?}",
                start.elapsed()
            );
        }
    }

    #[test]
    fn computed_array_update_indices_match_all_oracles() {
        let executable =
            stock_xls_path().expect("selected stock-XLS checks require block_to_verilog_main");
        let context = required_external_yosys_context()
            .expect("selected Yosys checks require XLSYNTH_YOSYS_PATH and XLSYNTH_LIBERTY_FILES");
        let package = parse_reference(
            r#"package public_computed_index
top block computed_index(values: bits[8][3], index: bits[3], replacement: bits[8], result: bits[8][3]) {
  values: bits[8][3] = input_port(name=values, id=1)
  index: bits[3] = input_port(name=index, id=2)
  replacement: bits[8] = input_port(name=replacement, id=3)
  not.4: bits[3] = not(index, id=4)
  updated: bits[8][3] = array_update(values, replacement, indices=[not.4], id=5)
  result: () = output_port(updated, name=result, id=6)
}
"#,
        );
        assert_stock_xls_semantics(
            &package,
            executable,
            context,
            &BlockCodegenOptions::default(),
            &crate::semantics::Trace::for_package(&package),
        )
        .unwrap();
    }

    #[test]
    fn stock_xls_capability_excludes_zero_width_aggregate_comparisons() {
        let source = r#"package public_empty_comparison

top block public_empty(x: bits[1], result: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  empty: () = literal(value=(), id=2)
  equal: bits[1] = eq(empty, empty, id=3)
  result: () = output_port(equal, name=result, id=4)
}
"#;
        let package = parse_reference(source);
        assert!(!stock_xls_supports(&package));
        assert!(stock_xls_supports(&parse_reference(
            references::COMBINATIONAL
        )));
    }

    #[test]
    fn stock_xls_capability_excludes_array_valued_identity() {
        let source = r#"package public_array_identity

top block public_array(x: bits[3][2], result: bits[3][2]) {
  x: bits[3][2] = input_port(name=x, id=1)
  copied: bits[3][2] = identity(x, id=2)
  result: () = output_port(copied, name=result, id=3)
}
"#;
        assert!(!stock_xls_supports(&parse_reference(source)));
    }

    #[test]
    fn stock_xls_capability_excludes_zero_bit_gate() {
        let package = parse_reference(
            r#"package public_empty_gate
top block empty_gate(enable: bits[1], out: bits[1]) {
  enable: bits[1] = input_port(name=enable, id=1)
  empty: () = tuple(id=2)
  gated: () = gate(enable, empty, id=3)
  out: () = output_port(enable, name=out, id=4)
}
"#,
        );
        assert_eq!(
            super::stock_xls_skip_reason(&package),
            Some("zero-bit-gate")
        );

        crate::check_profile_case(&package, crate::Profile::NativeSemantics);
    }
}
