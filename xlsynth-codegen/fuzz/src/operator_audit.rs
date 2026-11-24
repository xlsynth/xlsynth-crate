// SPDX-License-Identifier: Apache-2.0

//! Directed scalar operator coverage across DSLX-derived SV port types.

use std::collections::BTreeSet;
use std::fmt::Write;

use xlsynth_codegen::{BlockCodegenOptions, Layout};

use crate::dslx_types::TYPE_KIND_COUNT;
use crate::semantics::Trace;

/// Checks each scalar operator using every supported input and output type
/// family.
#[test]
fn scalar_operators_preserve_xls_semantics_across_port_types() {
    let package = crate::parse_reference(&scalar_operator_ir());
    let mut observed_signed_difference = false;
    let mut observed_large_index = false;
    let mut observed_zero_divisor = false;
    let mut observed_both_select_arms = [false; 2];

    for kind in 0..TYPE_KIND_COUNT {
        let seed = [kind.wrapping_mul(17).wrapping_add(13); 32];
        let trace = Trace::with_seed(&package, seed);
        for sample in &trace.samples {
            let a = sample.inputs["a"].to_u64_if_known().unwrap() as u8;
            let b = sample.inputs["b"].to_u64_if_known().unwrap() as u8;
            let index = sample.inputs["index"].to_u64_if_known().unwrap();
            observed_signed_difference |= (a < b) != ((a as i8) < (b as i8));
            observed_large_index |= index >= 8;
            observed_zero_divisor |= b == 0;
            observed_both_select_arms[usize::from(a < b)] = true;
        }

        let port_count = crate::top_block(&package).input_ports().count()
            + crate::top_block(&package).output_ports().count();
        let mut choices = Vec::with_capacity(4 * port_count);
        for index in 0..port_count {
            let selected = (kind + (index as u8).wrapping_mul(3)) % TYPE_KIND_COUNT;
            choices.extend([
                selected,
                index as u8,
                kind ^ 1,
                (index as u8).wrapping_mul(7),
            ]);
        }
        let options = BlockCodegenOptions {
            layout: if kind % 2 == 0 {
                Layout::None
            } else {
                Layout::Pipeline
            },
            separate_lines: kind % 3 == 0,
            ..BlockCodegenOptions::default()
        };
        crate::port_types::check_package(package.clone(), &choices, &options, seed)
            .0
            .unwrap_or_else(|error| panic!("kind={kind}: {error}"));
    }

    let block = crate::top_block(&package);
    let live = crate::coverage::live_nodes(block);
    let actual: BTreeSet<_> = block
        .nodes
        .iter()
        .enumerate()
        .filter(|(index, _)| live[*index])
        .map(|(_, node)| node.payload.get_operator())
        .collect();
    let expected = crate::operations(false);
    let missing: Vec<_> = expected
        .iter()
        .map(|operation| operation.name())
        .filter(|name| !actual.contains(name))
        .collect();
    assert!(
        missing.is_empty(),
        "eligible scalar operations missing from checked output paths: {missing:?}"
    );

    assert!(
        observed_signed_difference,
        "no signed/unsigned comparison differed"
    );
    assert!(
        observed_large_index,
        "no dynamic slice or shift exceeded the input width"
    );
    assert!(
        observed_zero_divisor,
        "no division-by-zero input was exercised"
    );
    assert!(
        observed_both_select_arms.into_iter().all(|seen| seen),
        "not all select arms were exercised"
    );
}

/// Bundles many live scalar results so each type family needs one SV
/// compilation.
fn scalar_operator_ir() -> String {
    let operations: &[(&str, usize, &str)] = &[
        ("constant", 8, "literal(value=85"),
        ("copy", 8, "identity(a"),
        ("inverted", 8, "not(a"),
        ("negated", 8, "neg(a"),
        ("any_bit", 1, "or_reduce(a"),
        ("all_bits", 1, "and_reduce(a"),
        ("parity", 1, "xor_reduce(a"),
        ("bitwise_and", 8, "and(a, b"),
        ("bitwise_nand", 8, "nand(a, b"),
        ("bitwise_nor", 8, "nor(a, b"),
        ("bitwise_or", 8, "or(a, b"),
        ("bitwise_xor", 8, "xor(a, b"),
        ("bit_low", 1, "bit_slice(a, start=0, width=1"),
        ("bit_sign", 1, "bit_slice(a, start=7, width=1"),
        ("window", 4, "bit_slice(a, start=2, width=4"),
        ("other_window", 4, "bit_slice(b, start=1, width=4"),
        ("dynamic", 4, "dynamic_bit_slice(a, index, width=4"),
        ("updated", 8, "bit_slice_update(a, index, v_other_window"),
        ("left", 8, "shll(a, index"),
        ("logical_right", 8, "shrl(a, index"),
        ("arithmetic_right", 8, "shra(a, index"),
        ("reversed", 8, "reverse(a"),
        ("joined", 8, "concat(v_window, v_other_window"),
        ("zero_extended", 8, "zero_ext(v_window, new_bit_count=8"),
        ("sign_extended", 8, "sign_ext(v_window, new_bit_count=8"),
        ("sum", 8, "add(a, b"),
        ("difference", 8, "sub(a, b"),
        ("unsigned_product", 8, "umul(a, b"),
        ("signed_product", 8, "smul(a, b"),
        ("unsigned_quotient", 8, "udiv(a, b"),
        ("signed_quotient", 8, "sdiv(a, b"),
        ("unsigned_remainder", 8, "umod(a, b"),
        ("signed_remainder", 8, "smod(a, b"),
        ("equal", 1, "eq(a, b"),
        ("different", 1, "ne(a, b"),
        ("unsigned_less", 1, "ult(a, b"),
        ("signed_less", 1, "slt(a, b"),
        ("unsigned_greater", 1, "ugt(a, b"),
        ("signed_greater", 1, "sgt(a, b"),
        ("unsigned_less_equal", 1, "ule(a, b"),
        ("signed_less_equal", 1, "sle(a, b"),
        ("unsigned_greater_equal", 1, "uge(a, b"),
        ("signed_greater_equal", 1, "sge(a, b"),
        ("gated", 8, "gate(v_bit_low, a"),
        ("one_hot", 9, "one_hot(a, lsb_prio=true"),
        ("encoded", 3, "encode(a"),
        ("decoded", 8, "decode(index, width=8"),
        ("carry", 1, "ext_carry_out(a, b, v_bit_low"),
        ("priority_index", 4, "ext_prio_encode(a, lsb_prio=true"),
        ("leading_zeros", 4, "ext_clz(a, offset=1, new_bit_count=4"),
        (
            "normalized",
            8,
            "ext_normalize_left(a, shift_offset=1, normalized_bit_count=8",
        ),
        ("low_mask", 8, "ext_mask_low(index"),
        (
            "extended_sum",
            10,
            "ext_nary_add(a, b, v_window, signed=[false, true, false], negated=[false, false, true], arch=kogge_stone",
        ),
        ("selected", 8, "sel(v_unsigned_less, cases=[a, b]"),
        (
            "priority_selected",
            8,
            "priority_sel(index, cases=[a, b, v_joined, v_updated], default=a",
        ),
        (
            "one_hot_selected",
            8,
            "one_hot_sel(index, cases=[a, b, v_joined, v_updated]",
        ),
    ];
    let mut ir = String::from(
        "package scalar_audit\n\ntop block scalar_audit(a: bits[8], b: bits[8], index: bits[4]",
    );
    for (name, width, _) in operations {
        write!(ir, ", out_{name}: bits[{width}]").unwrap();
    }
    ir.push_str(") {\n  a: bits[8] = input_port(name=a, id=1)\n  b: bits[8] = input_port(name=b, id=2)\n  index: bits[4] = input_port(name=index, id=3)\n");
    let mut id = 4;
    for (name, width, expression) in operations {
        writeln!(ir, "  v_{name}: bits[{width}] = {expression}, id={id})").unwrap();
        id += 1;
    }
    for (name, _, _) in operations {
        writeln!(
            ir,
            "  out_{name}: () = output_port(v_{name}, name=out_{name}, id={id})"
        )
        .unwrap();
        id += 1;
    }
    ir.push_str("}\n");
    ir
}
