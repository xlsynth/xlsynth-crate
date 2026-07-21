// SPDX-License-Identifier: Apache-2.0

//! Loads ABC's binary AIGER files with structural-choice extensions.
//!
//! ABC serializes GIA choice chains in a binary q extension after the AIGER
//! comment marker. The ordinary AIG remains stored as a GateFn; this loader
//! adds the sibling links while preserving otherwise-dead alternative cones.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::aig::{AigOperand, AigRef, ChoiceAig};
use crate::aig_serdes::load_aiger::LoadAigerResult;
use crate::aig_serdes::load_aiger_auto::load_aiger_auto;
use crate::aig_serdes::load_aiger_binary::load_aiger_binary_with_comment_tail;
use crate::gate_builder::GateBuilderOptions;

/// Parses ASCII or binary AIGER from disk, retaining ABC q choices when
/// present.
pub fn load_abc_choice_aiger_auto_from_path(path: &Path) -> Result<ChoiceAig, String> {
    let contents =
        fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    load_abc_choice_aiger_auto(&contents)
}

/// Parses ordinary AIGER or ABC's binary q-AIGER extension format.
///
/// ASCII AIGER has no ABC q-extension carrier, so it is returned as an
/// ordinary no-choice graph. Binary AIGER is always parsed through the
/// choice-aware path; files without q metadata simply produce no siblings.
pub fn load_abc_choice_aiger_auto(src: &[u8]) -> Result<ChoiceAig, String> {
    let header_line_end = src
        .iter()
        .position(|byte| *byte == b'\n')
        .ok_or_else(|| "unterminated AIGER header".to_string())?;
    let header_line = std::str::from_utf8(&src[..header_line_end])
        .map_err(|e| format!("invalid UTF-8 in AIGER header: {}", e))?;
    let header_kind = header_line
        .split_whitespace()
        .next()
        .ok_or_else(|| "missing AIGER header token".to_string())?;
    match header_kind {
        "aig" => load_abc_choice_aiger_binary(src),
        "aag" => {
            let loaded = load_aiger_auto(src, GateBuilderOptions::no_opt())?;
            Ok(ChoiceAig::without_choices(loaded.gate_fn))
        }
        other => Err(format!("unknown AIGER header '{}'", other)),
    }
}

/// Parses an ABC binary-AIGER file from disk, including its q extension.
pub fn load_abc_choice_aiger_binary_from_path(path: &Path) -> Result<ChoiceAig, String> {
    let contents =
        fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    load_abc_choice_aiger_binary(&contents)
}

/// Parses ABC binary-AIGER bytes, retaining any structural-choice siblings.
///
/// Parsing always disables GateBuilder folding and structural hashing. ABC's q
/// records refer to source AIGER object IDs, so distinct source nodes must
/// remain distinct even when they have identical structure.
pub fn load_abc_choice_aiger_binary(src: &[u8]) -> Result<ChoiceAig, String> {
    let (loaded, comment_tail) =
        load_aiger_binary_with_comment_tail(src, GateBuilderOptions::no_opt())?;
    let sibling_next = parse_q_extension(comment_tail, &loaded)?;
    ChoiceAig::new(loaded.gate_fn, sibling_next)
}

/// Parses the first q extension in ABC's length-prefixed extension stream.
fn parse_q_extension(
    comment_tail: Option<&[u8]>,
    loaded: &LoadAigerResult,
) -> Result<Vec<Option<AigRef>>, String> {
    let mut sibling_next = vec![None; loaded.gate_fn.gates.len()];
    let Some(bytes) = comment_tail else {
        return Ok(sibling_next);
    };

    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let tag = bytes[cursor];
        if tag == b'\n' || tag == b'\r' {
            break;
        }
        if tag == b'q' {
            parse_q_block(
                bytes,
                &mut cursor,
                &loaded.var_to_operand,
                &mut sibling_next,
            )?;
            return Ok(sibling_next);
        }
        if is_length_prefixed_abc_extension(tag) {
            skip_length_prefixed_extension(bytes, &mut cursor, tag)?;
            continue;
        }
        if tag.is_ascii_lowercase() {
            return Err(format!(
                "unsupported ABC AIGER extension '{}' before q extension",
                tag as char
            ));
        }
        break;
    }
    Ok(sibling_next)
}

/// Reads one q block and records its representative-to-sibling links.
fn parse_q_block(
    bytes: &[u8],
    cursor: &mut usize,
    var_to_operand: &HashMap<u32, AigOperand>,
    sibling_next: &mut [Option<AigRef>],
) -> Result<(), String> {
    debug_assert_eq!(bytes[*cursor], b'q');
    *cursor += 1;
    let payload_len = read_be_u32(bytes, cursor, "q payload length")? as usize;
    let payload_end = cursor
        .checked_add(payload_len)
        .ok_or_else(|| "q payload length overflows usize".to_string())?;
    if payload_end > bytes.len() {
        return Err(format!(
            "q payload length {} exceeds remaining input {}",
            payload_len,
            bytes.len() - *cursor
        ));
    }
    if payload_len < 4 {
        return Err(format!(
            "q payload length {} is too small for pair count",
            payload_len
        ));
    }

    let pair_count = read_be_u32(bytes, cursor, "q pair count")? as usize;
    let expected_len = 4usize
        .checked_add(
            pair_count
                .checked_mul(8)
                .ok_or_else(|| "q pair count overflows payload length".to_string())?,
        )
        .ok_or_else(|| "q payload length overflows usize".to_string())?;
    if payload_len != expected_len {
        return Err(format!(
            "q payload length {} does not match {} sibling pairs (expected {})",
            payload_len, pair_count, expected_len
        ));
    }

    for pair_index in 0..pair_count {
        let representative_var = read_be_u32(bytes, cursor, "q representative object ID")?;
        let sibling_var = read_be_u32(bytes, cursor, "q sibling object ID")?;
        if representative_var <= sibling_var {
            return Err(format!(
                "q pair {} must point to an earlier object: {} -> {}",
                pair_index, representative_var, sibling_var
            ));
        }
        let representative = lookup_q_object(
            var_to_operand,
            representative_var,
            "representative",
            pair_index,
        )?;
        let sibling = lookup_q_object(var_to_operand, sibling_var, "sibling", pair_index)?;
        if sibling_next[representative.id].is_some() {
            return Err(format!(
                "q pair {} gives object {} more than one sibling link",
                pair_index, representative_var
            ));
        }
        sibling_next[representative.id] = Some(sibling);
    }
    debug_assert_eq!(*cursor, payload_end);
    Ok(())
}

/// Resolves an ABC object ID to the corresponding GateFn node.
fn lookup_q_object(
    var_to_operand: &HashMap<u32, AigOperand>,
    object_id: u32,
    role: &str,
    pair_index: usize,
) -> Result<AigRef, String> {
    var_to_operand
        .get(&object_id)
        .map(|operand| operand.node)
        .ok_or_else(|| {
            format!(
                "q pair {} references unknown {} object ID {}",
                pair_index, role, object_id
            )
        })
}

/// Skips an ABC extension whose payload starts with a big-endian byte length.
fn skip_length_prefixed_extension(bytes: &[u8], cursor: &mut usize, tag: u8) -> Result<(), String> {
    *cursor += 1;
    let payload_len = read_be_u32(bytes, cursor, "ABC extension payload length")? as usize;
    *cursor = cursor
        .checked_add(payload_len)
        .ok_or_else(|| format!("ABC extension '{}' length overflows usize", tag as char))?;
    if *cursor > bytes.len() {
        return Err(format!(
            "ABC extension '{}' payload exceeds remaining input",
            tag as char
        ));
    }
    Ok(())
}

/// Returns whether a non-choice ABC extension uses the common length prefix.
fn is_length_prefixed_abc_extension(tag: u8) -> bool {
    matches!(
        tag,
        b'a' | b'b'
            | b'c'
            | b'd'
            | b'f'
            | b'g'
            | b'h'
            | b'i'
            | b'j'
            | b'k'
            | b'm'
            | b'o'
            | b'p'
            | b'r'
            | b's'
            | b't'
            | b'u'
            | b'v'
            | b'w'
            | b'y'
    )
}

/// Reads ABC's fixed-width, big-endian integer encoding.
fn read_be_u32(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u32, String> {
    let end = cursor
        .checked_add(4)
        .ok_or_else(|| format!("{} cursor overflows usize", what))?;
    let encoded = bytes
        .get(*cursor..end)
        .ok_or_else(|| format!("unexpected EOF while reading {}", what))?;
    *cursor = end;
    Ok(u32::from_be_bytes(encoded.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::AigNode;
    use crate::aig_serdes::load_aiger_binary::load_aiger_binary;

    fn append_q_block(bytes: &mut Vec<u8>, pairs: &[(u32, u32)]) {
        bytes.extend_from_slice(b"cq");
        let payload_len = 4 + 8 * pairs.len() as u32;
        bytes.extend_from_slice(&payload_len.to_be_bytes());
        bytes.extend_from_slice(&(pairs.len() as u32).to_be_bytes());
        for (representative, sibling) in pairs {
            bytes.extend_from_slice(&representative.to_be_bytes());
            bytes.extend_from_slice(&sibling.to_be_bytes());
        }
    }

    fn binary_aiger_with_dead_duplicate_choice() -> Vec<u8> {
        let mut bytes = b"aig 4 2 0 1 2\n6\n".to_vec();
        // var 3 = i1 & i0; var 4 is a structurally identical, dead choice.
        bytes.extend_from_slice(&[2, 2, 4, 2]);
        append_q_block(&mut bytes, &[(4, 3)]);
        bytes
    }

    fn binary_aiger_with_choice_chain() -> Vec<u8> {
        let mut bytes = b"aig 5 2 0 1 3\n6\n".to_vec();
        // vars 3, 4, and 5 are identical ANDs; only var 3 reaches the output.
        bytes.extend_from_slice(&[2, 2, 4, 2, 6, 2]);
        append_q_block(&mut bytes, &[(5, 4), (4, 3)]);
        bytes
    }

    #[test]
    fn loads_q_link_and_preserves_dead_choice_cone() {
        let bytes = binary_aiger_with_dead_duplicate_choice();

        let choice_aig = load_abc_choice_aiger_binary(&bytes).unwrap();

        assert_eq!(choice_aig.graph().gates.len(), 5);
        assert!(matches!(choice_aig.graph().gates[4], AigNode::And2 { .. }));
        assert!(
            !choice_aig
                .graph()
                .post_order_refs()
                .contains(&AigRef { id: 4 })
        );
        assert_eq!(
            choice_aig.next_sibling(AigRef { id: 4 }),
            Some(AigRef { id: 3 })
        );
        assert_eq!(choice_aig.sibling_link_count(), 1);
    }

    #[test]
    fn ordinary_binary_loader_ignores_binary_q_tail() {
        let bytes = binary_aiger_with_dead_duplicate_choice();

        let loaded = load_aiger_binary(&bytes, GateBuilderOptions::no_opt()).unwrap();

        assert_eq!(loaded.gate_fn.gates.len(), 5);
    }

    #[test]
    fn loads_multi_sibling_chain() {
        let bytes = binary_aiger_with_choice_chain();

        let choice_aig = load_abc_choice_aiger_binary(&bytes).unwrap();
        let chain: Vec<AigRef> = choice_aig.sibling_chain(AigRef { id: 5 }).collect();

        assert_eq!(
            chain,
            vec![AigRef { id: 5 }, AigRef { id: 4 }, AigRef { id: 3 }]
        );
    }

    #[test]
    fn loads_q_after_ascii_symbol_table() {
        let mut bytes = b"aig 4 2 0 1 2\n6\n".to_vec();
        bytes.extend_from_slice(&[2, 2, 4, 2]);
        bytes.extend_from_slice(b"i0 a\ni1 b\no0 y\n");
        append_q_block(&mut bytes, &[(4, 3)]);

        let choice_aig = load_abc_choice_aiger_binary(&bytes).unwrap();

        assert_eq!(choice_aig.graph().inputs[0].name, "a");
        assert_eq!(choice_aig.graph().outputs[0].name, "y");
        assert_eq!(
            choice_aig.next_sibling(AigRef { id: 4 }),
            Some(AigRef { id: 3 })
        );
    }

    #[test]
    fn accepts_binary_aiger_without_q_extension() {
        let bytes = b"aig 3 2 0 1 1\n6\n\x02\x02c\nplain comment\n";

        let choice_aig = load_abc_choice_aiger_binary(bytes).unwrap();

        assert_eq!(choice_aig.sibling_link_count(), 0);
    }

    #[test]
    fn auto_loader_accepts_ascii_aiger_as_no_choice_graph() {
        let bytes = b"aag 1 1 0 1 0\n2\n2\n";

        let choice_aig = load_abc_choice_aiger_auto(bytes).unwrap();

        assert_eq!(choice_aig.graph().inputs.len(), 1);
        assert_eq!(choice_aig.graph().outputs.len(), 1);
        assert_eq!(choice_aig.sibling_link_count(), 0);
    }

    #[test]
    fn rejects_q_payload_with_inconsistent_pair_count() {
        let mut bytes = b"aig 3 2 0 1 1\n6\n\x02\x02cq".to_vec();
        bytes.extend_from_slice(&4u32.to_be_bytes());
        bytes.extend_from_slice(&1u32.to_be_bytes());

        let err = load_abc_choice_aiger_binary(&bytes).unwrap_err();

        assert!(err.contains("does not match"));
    }

    #[test]
    fn rejects_q_link_to_unknown_object() {
        let mut bytes = b"aig 3 2 0 1 1\n6\n\x02\x02".to_vec();
        append_q_block(&mut bytes, &[(4, 3)]);

        let err = load_abc_choice_aiger_binary(&bytes).unwrap_err();

        assert!(err.contains("unknown representative"));
    }
}
