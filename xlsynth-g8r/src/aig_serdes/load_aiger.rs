// SPDX-License-Identifier: Apache-2.0

//! Loads an ASCII AIGER ("aag") file into a `GateFn`.
//!
//! Right now we only support purely combinational files (L == 0).  Latch
//! support can be added later by materialising registers and clock signals in
//! the `GateFn` representation.
//!
//! The parser is intentionally strict — we fail fast on any structural
//! inconsistency so that upstream tooling can rely on strong invariants when
//! manipulating the resulting `GateFn`.

use crate::aig::{AigNode, AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Resulting `GateFn` together with the mapping from AIGER variable index to
/// operand (useful for debugging / diagnostics).
#[derive(Debug)]
pub struct LoadAigerResult {
    pub gate_fn: GateFn,
    pub var_to_operand: HashMap<u32, AigOperand>,
}

/// Parses an ASCII-AIGER file from disk.
pub fn load_aiger_from_path(
    path: &Path,
    opts: GateBuilderOptions,
) -> Result<LoadAigerResult, String> {
    let contents = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    load_aiger(&contents, opts)
}

/// Parses the provided ASCII-AIGER text (`src`) and yields a `GateFn` built
/// with the supplied `GateBuilderOptions`.
///
/// If `src` violates the (subset of) AIGER spec we support an error is
/// returned.
pub fn load_aiger(src: &str, opts: GateBuilderOptions) -> Result<LoadAigerResult, String> {
    let mut lines = src.lines();

    // ---- Header -----------------------------------------------------------
    let header_line = lines
        .next()
        .ok_or_else(|| "empty AIGER input".to_string())?;
    let header_tokens: Vec<&str> = header_line.split_whitespace().collect();
    if header_tokens.len() != 6 {
        return Err(format!(
            "expected 6 tokens in AIGER header, got {} (\"{}\")",
            header_tokens.len(),
            header_line
        ));
    }
    if header_tokens[0] != "aag" {
        return Err(format!(
            "only ASCII-AIGER (aag) is supported; got '{}'",
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

    // ---- Helper -----------------------------------------------------------
    fn next_non_empty_line<'a>(iter: &mut std::str::Lines<'a>) -> Option<&'a str> {
        while let Some(line) = iter.next() {
            if !line.trim().is_empty() {
                return Some(line);
            }
        }
        None
    }

    // ---- Prepare GateBuilder ---------------------------------------------
    let mut gb = GateBuilder::new("loaded_aiger".to_string(), opts);

    // Map from AIGER variable index -> operand within builder.
    let mut var_to_operand: HashMap<u32, AigOperand> = HashMap::new();
    // Constant 0 and 1  (false / true).
    var_to_operand.insert(0, gb.get_false());

    // ---- Inputs -----------------------------------------------------------
    // For names we may later overwrite using the symbol table.
    let mut input_names: Vec<String> = Vec::with_capacity(i as usize);
    for idx in 0..i {
        let line = next_non_empty_line(&mut lines)
            .ok_or_else(|| format!("expected {} input lines but found fewer", i))?;
        let lit_val: u32 = line
            .trim()
            .parse()
            .map_err(|e| format!("invalid input literal '{}': {}", line, e))?;
        if lit_val & 1 != 0 {
            return Err(format!(
                "input literal must be positive, got negated literal {}",
                lit_val
            ));
        }
        let var = lit_val >> 1;
        if var == 0 {
            return Err("input literal refers to constant false (0)".to_string());
        }
        if var_to_operand.contains_key(&var) {
            return Err(format!("duplicate input variable index {}", var));
        }
        // Create input bit (single-bit vector for now; may regroup later using symbol
        // table).
        let default_name = format!("i{}", idx);
        let bv = gb.add_input(default_name.clone(), 1);
        let op = *bv.get_lsb(0);
        var_to_operand.insert(var, op);
        input_names.push(default_name);
    }

    // ---- Latches (skipped, validated earlier) -----------------------------
    for _ in 0..l {
        // Unreachable but keep parser in sync.
        next_non_empty_line(&mut lines)
            .ok_or_else(|| "unexpected EOF while reading latch lines".to_string())?;
    }

    // ---- Outputs ----------------------------------------------------------
    let mut output_literals: Vec<u32> = Vec::with_capacity(o as usize);
    for _ in 0..o {
        let line = next_non_empty_line(&mut lines)
            .ok_or_else(|| format!("expected {} output lines but found fewer", o))?;
        let lit_val: u32 = line
            .trim()
            .parse()
            .map_err(|e| format!("invalid output literal '{}': {}", line, e))?;
        output_literals.push(lit_val);
    }

    // ---- AND gates --------------------------------------------------------
    // We will store their lines temporarily so we can add them in one pass
    // because GateBuilder must see their operands already mapped.
    #[derive(Debug)]
    struct AndLine {
        lhs_var: u32,
        rhs0_lit: u32,
        rhs1_lit: u32,
    }
    let mut and_lines: Vec<AndLine> = Vec::with_capacity(a as usize);

    for _ in 0..a {
        let line = next_non_empty_line(&mut lines)
            .ok_or_else(|| format!("expected {} AND lines but found fewer", a))?;
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() != 3 {
            return Err(format!("AND line should have 3 fields, got '{}'", line));
        }
        let lhs: u32 = toks[0]
            .parse()
            .map_err(|e| format!("invalid AND lhs '{}': {}", toks[0], e))?;
        let rhs0: u32 = toks[1]
            .parse()
            .map_err(|e| format!("invalid AND rhs '{}': {}", toks[1], e))?;
        let rhs1: u32 = toks[2]
            .parse()
            .map_err(|e| format!("invalid AND rhs '{}': {}", toks[2], e))?;
        if lhs & 1 != 0 {
            return Err(format!("AND lhs literal {} must be positive (even)", lhs));
        }
        let lhs_var = lhs >> 1;
        and_lines.push(AndLine {
            lhs_var,
            rhs0_lit: rhs0,
            rhs1_lit: rhs1,
        });
    }

    // The remaining lines may be symbol table and comments – we parse symbols
    // later.  Gather them now.
    let remaining_lines: Vec<&str> = lines.collect();

    // ---- Helper: convert literal to operand -------------------------------
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

    // ---- Materialise AND gates in builder ---------------------------------
    for al in &and_lines {
        // Evaluate the operands.
        let rhs0 = lit_to_operand(al.rhs0_lit, &mut gb, &var_to_operand)?;
        let rhs1 = lit_to_operand(al.rhs1_lit, &mut gb, &var_to_operand)?;
        let and_op = gb.add_and_binary(rhs0, rhs1);
        // Record mapping.
        if var_to_operand.contains_key(&al.lhs_var) {
            return Err(format!("variable {} already defined", al.lhs_var));
        }
        var_to_operand.insert(al.lhs_var, and_op);
    }

    // ---- Symbol table (optional) -----------------------------------------
    // Build maps idx -> explicit name if provided.
    let mut sym_input_names: HashMap<usize, String> = HashMap::new();
    let mut sym_output_names: HashMap<usize, String> = HashMap::new();

    let mut iter = remaining_lines.into_iter();
    while let Some(line) = iter.next() {
        // Stop at comment section.
        if line.starts_with('c') {
            break;
        }
        if line.is_empty() {
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
            // Ignore other symbol kinds ("l", "d") for now.
            _ => {}
        }
    }

    // ---- Replace input names ----------------------------------------------
    // Preserve symbol table names literally on the raw loaded scalar ports.
    for (idx, _name) in input_names.iter().enumerate() {
        if let Some(final_name) = sym_input_names.get(&idx) {
            gb.inputs[idx].name = final_name.clone();
        }
        let full_name = &gb.inputs[idx].name;
        let op = var_to_operand.get(&((idx as u32) + 1)).unwrap();
        if let Some(AigNode::Input {
            name: n, lsb_index, ..
        }) = gb.gates.get_mut(op.node.id)
        {
            *n = full_name.clone();
            *lsb_index = 0;
        }
    }

    // ---- Create outputs ----------------------------------------------------
    for (out_idx, lit) in output_literals.iter().enumerate() {
        let op = lit_to_operand(*lit, &mut gb, &var_to_operand)?;
        let name = sym_output_names
            .get(&out_idx)
            .cloned()
            .unwrap_or_else(|| format!("o{}", out_idx));
        gb.add_output(name, op.into());
    }

    let gate_fn = gb.build();

    Ok(LoadAigerResult {
        gate_fn,
        var_to_operand,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::gate::AigNode;
    use crate::aig_serdes::emit_aiger::emit_aiger;
    use crate::aig_serdes::gate2ir::{GateFnInterfaceSchema, repack_gate_fn_interface_with_schema};
    use crate::check_equivalence;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::test_utils::{
        interesting_ir_roundtrip_cases, load_interesting_ir_roundtrip_case, structurally_equivalent,
    };

    #[test]
    fn test_aiger_roundtrip_simple_and() {
        let mut gb = GateBuilder::new("rt_and".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1).try_into().unwrap();
        let b = gb.add_input("b".to_string(), 1).try_into().unwrap();
        let o = gb.add_and_binary(a, b);
        gb.add_output("o".to_string(), o.into());
        let orig = gb.build();

        let aiger = emit_aiger(&orig, true).unwrap();
        let loaded = load_aiger(&aiger, GateBuilderOptions::no_opt()).unwrap();
        assert!(structurally_equivalent(&orig, &loaded.gate_fn));
    }

    #[test]
    fn test_aiger_roundtrip_interesting_signatures() {
        for case in interesting_ir_roundtrip_cases() {
            let sample = load_interesting_ir_roundtrip_case(case);
            let aiger = emit_aiger(&sample.gate_fn, true).unwrap();
            let loaded = load_aiger(&aiger, GateBuilderOptions::no_opt()).unwrap();
            let schema = GateFnInterfaceSchema::from_pir_fn(&sample.g8r_fn).unwrap();
            let repacked = repack_gate_fn_interface_with_schema(loaded.gate_fn, &schema).unwrap();
            check_equivalence::validate_same_fn(&sample.g8r_fn, &repacked)
                .unwrap_or_else(|e| panic!("AIGER roundtrip failed for {}: {}", case.name, e));
        }
    }

    #[test]
    fn test_aiger_load_of_structured_ir_gatefn_stays_flat() {
        for case in interesting_ir_roundtrip_cases() {
            let sample = load_interesting_ir_roundtrip_case(case);
            let aiger = emit_aiger(&sample.gate_fn, true).unwrap();
            let loaded = load_aiger(&aiger, GateBuilderOptions::no_opt()).unwrap();
            assert!(
                loaded
                    .gate_fn
                    .inputs
                    .iter()
                    .all(|input| input.get_bit_count() == 1),
                "expected raw loaded AIGER inputs to stay scalar for {}",
                case.name
            );
            assert!(
                loaded
                    .gate_fn
                    .outputs
                    .iter()
                    .all(|output| output.get_bit_count() == 1),
                "expected raw loaded AIGER outputs to stay scalar for {}",
                case.name
            );
        }
    }

    #[test]
    fn test_aiger_load_preserves_suffix_names_literally() {
        let aiger = "aag 2 2 0 2 0\n2\n4\n2\n4\ni0 arg_0\ni1 arg_1\no0 ret_0\no1 ret_1\nc\n";
        let loaded = load_aiger(aiger, GateBuilderOptions::no_opt()).unwrap();
        assert_eq!(loaded.gate_fn.inputs[0].name, "arg_0");
        assert_eq!(loaded.gate_fn.inputs[1].name, "arg_1");
        assert_eq!(loaded.gate_fn.outputs[0].name, "ret_0");
        assert_eq!(loaded.gate_fn.outputs[1].name, "ret_1");

        let input0 = loaded.gate_fn.inputs[0].bit_vector.get_lsb(0).node;
        let input1 = loaded.gate_fn.inputs[1].bit_vector.get_lsb(0).node;
        match loaded.gate_fn.get(input0) {
            AigNode::Input {
                name, lsb_index, ..
            } => {
                assert_eq!(name, "arg_0");
                assert_eq!(*lsb_index, 0);
            }
            other => panic!("expected input node, got {:?}", other),
        }
        match loaded.gate_fn.get(input1) {
            AigNode::Input {
                name, lsb_index, ..
            } => {
                assert_eq!(name, "arg_1");
                assert_eq!(*lsb_index, 0);
            }
            other => panic!("expected input node, got {:?}", other),
        }
    }

    #[test]
    fn test_aiger_roundtrip_native_multi_output_via_gatefn_schema() {
        let mut gb = GateBuilder::new("native_multi".to_string(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        gb.add_output("o0".to_string(), a.into());
        gb.add_output("o1".to_string(), b.into());
        let orig = gb.build();

        let aiger = emit_aiger(&orig, true).unwrap();
        let loaded = load_aiger(&aiger, GateBuilderOptions::no_opt()).unwrap();
        let schema = GateFnInterfaceSchema::from_gate_fn(&orig).unwrap();
        let repacked = repack_gate_fn_interface_with_schema(loaded.gate_fn, &schema).unwrap();

        assert!(structurally_equivalent(&orig, &repacked));
    }
}
