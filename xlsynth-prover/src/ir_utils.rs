// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for preparing IR functions used across equivalence flows.

use xlsynth_pir::ir;
use xlsynth_pir::ir_parser;

/// Parses IR text into a package and fetches the requested top function with
/// optional parameter drops.
pub fn parse_package_and_drop_params(
    source: &str,
    top: Option<&str>,
    drop_params: &[String],
) -> Result<(ir::Package, ir::Fn), String> {
    let pkg = ir_parser::Parser::new(source)
        .parse_package()
        .map_err(|e| format!("Failed to parse IR package: {}", e))?;

    let func = if let Some(name) = top {
        pkg.get_fn(name)
            .cloned()
            .ok_or_else(|| format!("Top function '{}' not found in package", name))?
    } else {
        pkg.get_top_fn()
            .cloned()
            .ok_or_else(|| "No top function found in package".to_string())?
    };

    let func_name = func.name.clone();
    let func = func.drop_params(drop_params).map_err(|e| {
        format!(
            "Failed to drop parameters in function '{}': {}",
            func_name, e
        )
    })?;

    Ok((pkg, func))
}
