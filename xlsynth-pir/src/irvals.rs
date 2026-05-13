// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use xlsynth::IrValue;

/// Parses `.irvals`-style stimulus text where each line is one typed tuple
/// value.
pub fn parse_irvals_tuple_lines(irvals_text: &str) -> Result<Vec<IrValue>, String> {
    let mut values = Vec::new();
    for (lineno, line) in irvals_text.lines().enumerate() {
        let line_no = lineno + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(format!(
                "empty line {} in stimulus file is not allowed",
                line_no
            ));
        }
        let tuple_val = IrValue::parse_typed(trimmed)
            .map_err(|e| format!("failed to parse stimulus tuple at line {}: {}", line_no, e))?;
        tuple_val
            .get_elements()
            .map_err(|e| format!("stimulus line {} is not a tuple value: {}", line_no, e))?;
        values.push(tuple_val);
    }
    Ok(values)
}

/// Reads and parses a `.irvals` stimulus file.
pub fn parse_irvals_tuple_file(path: &Path) -> Result<Vec<IrValue>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    parse_irvals_tuple_lines(&text)
}
