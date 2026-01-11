// SPDX-License-Identifier: Apache-2.0

//! Loads a binary AIGER ("aig") file into a `GateFn`.
//!
//! Right now we only support purely combinational files (L == 0). Latch
//! support can be added later by materialising registers and clock signals in
//! the `GateFn` representation.
//!
//! The parser is intentionally strict — we fail fast on any structural
//! inconsistency so that upstream tooling can rely on strong invariants when
//! manipulating the resulting `GateFn`.

use crate::aig::{AigNode, AigOperand};
use crate::aig_serdes::load_aiger::LoadAigerResult;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Parses a binary-AIGER file from disk.
pub fn load_aiger_binary_from_path(
    path: &Path,
    opts: GateBuilderOptions,
) -> Result<LoadAigerResult, String> {
    let contents =
        fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    load_aiger_binary(&contents, opts)
}

/// Parses the provided binary-AIGER bytes and yields a `GateFn` built with the
/// supplied `GateBuilderOptions`.
pub fn load_aiger_binary(src: &[u8], opts: GateBuilderOptions) -> Result<LoadAigerResult, String> {
    let (header_line, mut cursor) = read_ascii_line(src, 0)?;
    let header_tokens: Vec<&str> = header_line.split_whitespace().collect();
    if header_tokens.len() != 6 {
        return Err(format!(
            "expected 6 tokens in AIGER header, got {} (\"{}\")",
            header_tokens.len(),
            header_line
        ));
    }
    if header_tokens[0] != "aig" {
        return Err(format!(
            "only binary-AIGER (aig) is supported; got '{}'",
            header_tokens[0]
        ));
    }

    let parse_u32 = |s: &str, field: &str| -> Result<u32, String> {
        s.parse::<u32>()
            .map_err(|e| format!("invalid {} value '{}': {}", field, s, e))
    };

    parse_u32(header_tokens[1], "M")?;
    let i = parse_u32(header_tokens[2], "I")?;
    let l = parse_u32(header_tokens[3], "L")?;
    let o = parse_u32(header_tokens[4], "O")?;
    let a = parse_u32(header_tokens[5], "A")?;

    if l != 0 {
        return Err("latch count (L) must be zero; sequential AIGER not yet supported".to_string());
    }

    let mut gb = GateBuilder::new("loaded_aiger".to_string(), opts);
    let mut var_to_operand: HashMap<u32, AigOperand> = HashMap::new();
    var_to_operand.insert(0, gb.get_false());

    let mut input_names: Vec<String> = Vec::with_capacity(i as usize);
    for idx in 0..i {
        let default_name = format!("i{}", idx);
        let bv = gb.add_input(default_name.clone(), 1);
        let op = *bv.get_lsb(0);
        var_to_operand.insert(idx + 1, op);
        input_names.push(default_name);
    }

    let mut output_literals: Vec<u32> = Vec::with_capacity(o as usize);
    for _ in 0..o {
        let (line, next_cursor) = read_ascii_line(src, cursor)?;
        cursor = next_cursor;
        let lit_val: u32 = line
            .trim()
            .parse()
            .map_err(|e| format!("invalid output literal '{}': {}", line, e))?;
        output_literals.push(lit_val);
    }

    let lit_to_operand = |lit: u32,
                          gb: &mut GateBuilder,
                          var_map: &HashMap<u32, AigOperand>|
     -> Result<AigOperand, String> {
        let var = lit >> 1;
        let neg = (lit & 1) == 1;
        let base_op = var_map
            .get(&var)
            .copied()
            .ok_or_else(|| format!("referenced undefined variable {} (literal {})", var, lit))?;
        Ok(if neg { gb.add_not(base_op) } else { base_op })
    };

    for and_idx in 0..a {
        let delta0 = decode_u32_varint(src, &mut cursor)?;
        let delta1 = decode_u32_varint(src, &mut cursor)?;
        let lhs_var = i + and_idx + 1;
        let lhs_lit = lhs_var << 1;
        let rhs0 = lhs_lit
            .checked_sub(delta0)
            .ok_or_else(|| format!("invalid AIGER delta0 {} for lhs {}", delta0, lhs_lit))?;
        let rhs1 = rhs0
            .checked_sub(delta1)
            .ok_or_else(|| format!("invalid AIGER delta1 {} for rhs0 {}", delta1, rhs0))?;
        let rhs0_op = lit_to_operand(rhs0, &mut gb, &var_to_operand)?;
        let rhs1_op = lit_to_operand(rhs1, &mut gb, &var_to_operand)?;
        let and_op = gb.add_and_binary(rhs0_op, rhs1_op);
        if var_to_operand.contains_key(&lhs_var) {
            return Err(format!("variable {} already defined", lhs_var));
        }
        var_to_operand.insert(lhs_var, and_op);
    }

    let remaining = &src[cursor..];
    if !remaining.is_empty() {
        let tail = std::str::from_utf8(remaining)
            .map_err(|e| format!("invalid UTF-8 in symbol/comment tail: {}", e))?;
        apply_symbol_table(
            tail,
            &mut gb,
            &var_to_operand,
            &input_names,
            &output_literals,
        )?;
    } else {
        apply_symbol_table("", &mut gb, &var_to_operand, &input_names, &output_literals)?;
    }

    let gate_fn = gb.build();

    Ok(LoadAigerResult {
        gate_fn,
        var_to_operand,
    })
}

fn read_ascii_line(src: &[u8], start: usize) -> Result<(String, usize), String> {
    if start >= src.len() {
        return Err("unexpected EOF while reading ASCII line".to_string());
    }
    let end = src[start..]
        .iter()
        .position(|b| *b == b'\n')
        .ok_or_else(|| "unterminated ASCII line in AIGER input".to_string())?;
    let line_bytes = &src[start..start + end];
    let line = std::str::from_utf8(line_bytes)
        .map_err(|e| format!("invalid UTF-8 in AIGER line: {}", e))?;
    Ok((line.to_string(), start + end + 1))
}

fn decode_u32_varint(src: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let mut shift = 0u32;
    let mut acc = 0u32;
    loop {
        if *cursor >= src.len() {
            return Err("unexpected EOF while reading AIGER varint".to_string());
        }
        let byte = src[*cursor];
        *cursor += 1;
        acc |= ((byte & 0x7f) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok(acc);
        }
        shift += 7;
        if shift >= 32 {
            return Err("AIGER varint overflow".to_string());
        }
    }
}

fn apply_symbol_table(
    tail: &str,
    gb: &mut GateBuilder,
    var_to_operand: &HashMap<u32, AigOperand>,
    input_names: &[String],
    output_literals: &[u32],
) -> Result<(), String> {
    let mut sym_input_names: HashMap<usize, String> = HashMap::new();
    let mut sym_output_names: HashMap<usize, String> = HashMap::new();

    let mut iter = tail.lines();
    while let Some(line) = iter.next() {
        if line.starts_with('c') {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let (kind, rest) = line.split_at(1);
        match kind {
            "i" | "o" => {
                let mut parts = rest.trim().split_whitespace();
                let idx_str = parts
                    .next()
                    .ok_or_else(|| format!("malformed symbol '{}': missing index", line))?;
                let idx: usize = idx_str
                    .parse()
                    .map_err(|e| format!("invalid symbol index in '{}': {}", line, e))?;
                let name = parts
                    .next()
                    .ok_or_else(|| format!("malformed symbol '{}': missing name", line))?;
                match kind {
                    "i" => {
                        sym_input_names.insert(idx, name.to_string());
                    }
                    "o" => {
                        sym_output_names.insert(idx, name.to_string());
                    }
                    _ => unreachable!(),
                }
            }
            _ => {}
        }
    }

    for (idx, _name) in input_names.iter().enumerate() {
        if let Some(final_name) = sym_input_names.get(&idx) {
            gb.inputs[idx].name = final_name.clone();
        }
        let full_name = &gb.inputs[idx].name;
        let bit_idx = split_base_bit(full_name).map(|(_, b)| b).unwrap_or(0);
        let op = var_to_operand.get(&((idx as u32) + 1)).unwrap();
        if let Some(AigNode::Input { name: n, lsb_index }) = gb.gates.get_mut(op.node.id) {
            *n = split_base_bit(full_name)
                .map(|(base, _)| base)
                .unwrap_or_else(|| full_name.clone());
            *lsb_index = bit_idx;
        }
    }

    for (out_idx, lit) in output_literals.iter().enumerate() {
        let op = {
            let var = lit >> 1;
            let neg = (lit & 1) == 1;
            let base_op = var_to_operand.get(&var).copied().ok_or_else(|| {
                format!("referenced undefined variable {} (literal {})", var, lit)
            })?;
            if neg { gb.add_not(base_op) } else { base_op }
        };
        let name = sym_output_names
            .get(&out_idx)
            .cloned()
            .unwrap_or_else(|| format!("o{}", out_idx));
        gb.add_output(name, op.into());
    }

    Ok(())
}

fn split_base_bit(name: &str) -> Option<(String, usize)> {
    if let Some(pos) = name.rfind('_') {
        if pos == 0 || pos == name.len() - 1 {
            return None;
        }
        let (base, digits) = name.split_at(pos);
        if let Ok(idx) = digits[1..].parse::<usize>() {
            return Some((base.to_string(), idx));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_serdes::emit_aiger_binary::emit_aiger_binary;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::test_utils::structurally_equivalent;

    #[test]
    fn test_aiger_binary_roundtrip_simple_and() {
        let mut gb = GateBuilder::new("rt_and_bin".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1).try_into().unwrap();
        let b = gb.add_input("b".to_string(), 1).try_into().unwrap();
        let o = gb.add_and_binary(a, b);
        gb.add_output("o".to_string(), o.into());
        let orig = gb.build();

        let aiger = emit_aiger_binary(&orig, true).unwrap();
        let loaded = load_aiger_binary(&aiger, GateBuilderOptions::no_opt()).unwrap();
        assert!(structurally_equivalent(&orig, &loaded.gate_fn));
    }
}
