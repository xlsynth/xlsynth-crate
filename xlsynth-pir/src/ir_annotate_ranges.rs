// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;

use xlsynth::IrAnalysis;
use xlsynth::ir_value::IrFormatPreference;

use crate::ir;
use crate::ir_parser;
use crate::ir_range_info::IrRangeInfo;

fn format_known_bits_binary(k: &xlsynth::KnownBits) -> String {
    let w = k.mask.get_bit_count();
    let mut s = String::with_capacity(2 + w);
    s.push_str("0b");
    for i in (0..w).rev() {
        let known = k.mask.get_bit(i).unwrap_or(false);
        if !known {
            s.push('X');
            continue;
        }
        let bit = k.value.get_bit(i).unwrap_or(false);
        s.push(if bit { '1' } else { '0' });
    }
    s
}

fn format_irbits_unsigned_decimal(bits: &xlsynth::IrBits) -> String {
    bits.to_string_fmt(
        IrFormatPreference::UnsignedDecimal,
        /* include_bit_count= */ false,
    )
}

fn cmp_irbits_unsigned(a: &xlsynth::IrBits, b: &xlsynth::IrBits) -> Ordering {
    if a.ult(b) {
        Ordering::Less
    } else if b.ult(a) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn format_intervals(intervals: &[xlsynth::Interval]) -> String {
    // Ensure deterministic output ordering regardless of libxls enumeration order.
    let mut idxs: Vec<usize> = (0..intervals.len()).collect();
    idxs.sort_by(|&a, &b| {
        let lo_cmp = cmp_irbits_unsigned(&intervals[a].lo, &intervals[b].lo);
        if lo_cmp != Ordering::Equal {
            return lo_cmp;
        }
        cmp_irbits_unsigned(&intervals[a].hi, &intervals[b].hi)
    });

    let parts: Vec<String> = idxs
        .into_iter()
        .map(|i| {
            let it = &intervals[i];
            let lo = format_irbits_unsigned_decimal(&it.lo);
            let hi = format_irbits_unsigned_decimal(&it.hi);
            format!("({}, {})", lo, hi)
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

fn make_node_comment(range_info: &IrRangeInfo, node: &ir::Node) -> Option<String> {
    let info = range_info.get(node.text_id)?;
    let intervals = info.intervals.as_ref()?;
    let known_bits = info.known_bits.as_ref()?;
    let range_str = format_intervals(intervals);
    let known_bits_str = format_known_bits_binary(known_bits);
    Some(format!(
        "range: {} known_bits: {}",
        range_str, known_bits_str
    ))
}

/// Reads an XLS IR *package* text and emits the same package, but with per-node
/// end-of-line comments for the designated top function showing interval ranges
/// and known-bits information from libxls analysis.
pub fn annotate_ranges_in_package_ir_text(
    ir_text: &str,
    top: Option<&str>,
) -> Result<String, String> {
    // Parse with PIR for formatting and to get stable node `text_id`s.
    let mut parser = ir_parser::Parser::new(ir_text);
    let pir_package = parser
        .parse_and_validate_package()
        .map_err(|e| format!("PIR parse/validate failed: {e}"))?;

    let top_fn_name: String = match top {
        Some(name) => name.to_string(),
        None => {
            let pir_top = pir_package.get_top_fn().ok_or_else(|| {
                "PIR package has no top function (top may be a block)".to_string()
            })?;
            pir_top.name.clone()
        }
    };

    let pir_fn = pir_package
        .get_fn(&top_fn_name)
        .ok_or_else(|| format!("PIR package has no function named '{top_fn_name}'"))?;

    // Parse with xlsynth for libxls analysis.
    let mut xlsynth_package = xlsynth::IrPackage::parse_ir(ir_text, None)
        .map_err(|e| format!("xlsynth parse_ir failed: {e}"))?;
    xlsynth_package
        .set_top_by_name(&top_fn_name)
        .map_err(|e| format!("xlsynth set_top_by_name('{top_fn_name}') failed: {e}"))?;
    let analysis: IrAnalysis = xlsynth_package
        .create_ir_analysis()
        .map_err(|e| format!("xlsynth create_ir_analysis failed: {e}"))?;

    let range_info = IrRangeInfo::build_from_analysis(&analysis, pir_fn)
        .map_err(|e| format!("building IrRangeInfo failed: {e}"))?;

    Ok(ir::emit_package_with_fn_override(
        &pir_package,
        |f, is_top| {
            if f.name != top_fn_name {
                return None;
            }
            Some(ir::emit_fn_with_node_comments_full(
                f,
                is_top,
                |func, nr| {
                    let node = func.get_node(nr);
                    make_node_comment(&range_info, node)
                },
            ))
        },
    ))
}
