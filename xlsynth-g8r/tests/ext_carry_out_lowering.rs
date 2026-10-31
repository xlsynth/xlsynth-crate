// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::{AigConeStats, get_aig_cone_stats};
use xlsynth_g8r::aig::topo::extract_cone;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_pir::ir::Type;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{FnBuilder, IrValue};

#[test]
fn identical_bit_seeds_the_carry_above_it() {
    let mut builder = FnBuilder::new("carry");
    let s = builder.param("s", Type::Bits(1)).unwrap();
    let x = builder.param("x", Type::Bits(31)).unwrap();
    let y = builder.param("y", Type::Bits(31)).unwrap();
    let a = builder.param("a", Type::Bits(5)).unwrap();
    let b = builder.param("b", Type::Bits(5)).unwrap();
    let zero = builder.literal(IrValue::make_ubits(1, 0).unwrap()).unwrap();
    let low_x = builder.concat(&[s, x]).unwrap();
    let low_y = builder.concat(&[s, y]).unwrap();
    let low = builder.ext_carry_out(low_x, low_y, zero).unwrap();
    let full_x = builder.concat(&[a, low_x]).unwrap();
    let full_y = builder.concat(&[b, low_y]).unwrap();
    let full = builder.ext_carry_out(full_x, full_y, zero).unwrap();
    let upper = builder.ext_carry_out(a, b, s).unwrap();
    let result = builder.concat(&[low, full, upper]).unwrap();
    let function = builder.build(result).unwrap();
    let gates = gatify(&function, GatifyOptions::all_opts_enabled())
        .unwrap()
        .gate_fn;
    let bits = &gates.outputs[0].bit_vector;
    assert_eq!(bits.get_lsb(2), gates.inputs[0].bit_vector.get_lsb(0));
    assert_eq!(bits.get_lsb(1), bits.get_lsb(0));
}

#[test]
fn carry_from_identical_operands_keeps_both_operation_ids() {
    let package = Parser::new(
        r#"package shared
top fn f(u: bits[1], v: bits[1], cin: bits[1]) -> (bits[1], bits[1]) {
  s: bits[1] = and(u, v, id=4)
  carry: bits[1] = ext_carry_out(s, s, cin, id=5)
  ret pair: (bits[1], bits[1]) = tuple(s, carry, id=6)
}"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let lowered = gatify(
        package.get_top_fn().unwrap(),
        GatifyOptions {
            track_pir_node_ids: true,
            ..GatifyOptions::all_opts_enabled()
        },
    )
    .unwrap();
    let bits = &lowered.gate_fn.outputs[0].bit_vector;
    assert_eq!(bits.get_lsb(0), bits.get_lsb(1));
    let ids = lowered.gate_fn.gates[bits.get_lsb(0).node.id].get_pir_node_ids();
    assert_eq!(ids, &[4, 5]);
}

#[test]
fn shared_carry_group_keeps_each_terminals_origin_without_hashing() {
    let package = Parser::new(
        r#"package shared
top fn f(a: bits[12], b: bits[12], cin: bits[1]) -> bits[2] {
  a4: bits[4] = bit_slice(a, start=0, width=4, id=4)
  b4: bits[4] = bit_slice(b, start=0, width=4, id=5)
  low: bits[1] = ext_carry_out(a4, b4, cin, id=6)
  full: bits[1] = ext_carry_out(b, a, cin, id=7)
  ret pair: bits[2] = concat(low, full, id=8)
}"#,
    )
    .parse_and_validate_package()
    .unwrap();
    for hash in [false, true] {
        let gates = gatify(
            package.get_top_fn().unwrap(),
            GatifyOptions {
                hash,
                track_pir_node_ids: true,
                ..GatifyOptions::all_opts_enabled()
            },
        )
        .unwrap()
        .gate_fn;
        // The low carry is also the complete low group in the wider carry.
        let low = gates.outputs[0].bit_vector.get_msb(0);
        let full = gates.outputs[0].bit_vector.get_msb(1);
        let (low_cone, _) = extract_cone(&[low.node], &gates.gates);
        let (full_cone, _) = extract_cone(&[full.node], &gates.gates);
        for node in low_cone {
            assert!(full_cone.contains(&node));
            assert_eq!(gates.gates[node.id].get_pir_node_ids(), &[6, 7]);
        }
    }
}

#[test]
fn independent_carry_terminals_bisect_their_addends() {
    let width = 14;
    let mut builder = FnBuilder::new("independent");
    let a = builder.param("a", Type::Bits(width)).unwrap();
    let b = builder.param("b", Type::Bits(width)).unwrap();
    let x = builder.param("x", Type::Bits(width)).unwrap();
    let y = builder.param("y", Type::Bits(width)).unwrap();
    let seed = builder.param("seed", Type::Bits(1)).unwrap();
    let left = builder.ext_carry_out(a, b, seed).unwrap();
    let right = builder.ext_carry_out(x, y, seed).unwrap();
    let result = builder.concat(&[left, right, left]).unwrap();
    let function = builder.build(result).unwrap();
    let gates = gatify(&function, GatifyOptions::all_opts_enabled())
        .unwrap()
        .gate_fn;
    for root in gates.outputs[0].bit_vector.iter_lsb_to_msb() {
        // At the final OR, high.generate reads both high halves. The other
        // operand also reads the low halves and the carry-in.
        let smaller_support = gates.gates[root.node.id]
            .args()
            .map(|arg| {
                let (_, inputs) = extract_cone(&[arg], &gates.gates);
                inputs.len()
            })
            .min();
        assert_eq!(smaller_support, Some(2 * (width / 2)));
    }
}

#[test]
fn projecting_a_carry_only_builds_the_demanded_tree() {
    let lower = |include_low_carry: bool| -> Result<_, Box<dyn std::error::Error>> {
        let mut builder = FnBuilder::new("carry");
        let a = builder.param("a", Type::Bits(16))?;
        let b = builder.param("b", Type::Bits(16))?;
        let seed = builder.param("seed", Type::Bits(1))?;
        let full = builder.ext_carry_out(a, b, seed)?;
        let x = builder.bit_slice(a, 0, 7)?;
        let y = builder.bit_slice(b, 0, 7)?;
        let discarded = if include_low_carry {
            builder.ext_carry_out(x, y, seed)?
        } else {
            builder.literal(IrValue::make_ubits(1, 0)?)?
        };
        let pair = builder.concat(&[discarded, full])?;
        let result = builder.bit_slice(pair, 0, 1)?;
        let function = builder.build(result)?;
        Ok(gatify(
            &function,
            GatifyOptions {
                track_pir_node_ids: true,
                ..GatifyOptions::all_opts_enabled()
            },
        )?
        .gate_fn)
    };
    let reference = lower(false).unwrap();
    let projected = lower(true).unwrap();
    // Compare all stored gates and their origins, including unreachable gates.
    assert_eq!(projected.gates, reference.gates);
    assert_eq!(
        projected.outputs[0].bit_vector.get_lsb(0),
        reference.outputs[0].bit_vector.get_lsb(0)
    );
}

fn prefix_carry_stats(
    width: usize,
    stride: usize,
) -> Result<AigConeStats, Box<dyn std::error::Error>> {
    let mut builder = FnBuilder::new("prefix_carries");
    let a = builder.param("a", Type::Bits(width))?;
    let b = builder.param("b", Type::Bits(width))?;
    let zero = builder.literal(IrValue::make_ubits(1, 0)?)?;
    let mut carries = Vec::new();
    for end in (stride..=width).step_by(stride) {
        let lhs = builder.bit_slice(a, 0, end)?;
        let rhs = builder.bit_slice(b, 0, end)?;
        carries.push(builder.ext_carry_out(lhs, rhs, zero)?);
    }
    let result = builder.concat(&carries)?;
    let function = builder.build(result)?;
    let gates = gatify(&function, GatifyOptions::all_opts_enabled())?.gate_fn;
    let roots: Vec<_> = gates.outputs[0]
        .bit_vector
        .iter_msb_to_lsb()
        .copied()
        .collect();
    Ok(get_aig_cone_stats(&gates.gates, &roots))
}

#[test]
fn byte_carries_share_aligned_groups_and_have_logarithmic_depth() {
    let width = 256usize;
    let stats = prefix_carry_stats(width, 8).unwrap();
    // Two gates per bit and three per group join. Each byte needs its seven
    // joins just once, even if each prefix pays for all its inter-byte joins.
    let bytes = width / 8;
    let inter_byte_joins: usize = (0..bytes).sum();
    let bound = 2 * width + 3 * (7 * bytes + inter_byte_joins);
    assert!(stats.and_nodes <= bound, "{stats:?}");
    for (depth, end) in stats.root_depths.iter().zip((8..=width).step_by(8)) {
        // A partial next level need only extend the high generate path.
        let bound = 1 + 2 * end.ilog2() as usize + usize::from(!end.is_power_of_two());
        assert!(*depth <= bound, "prefix {end}: {depth}");
    }
}

#[test]
fn all_prefix_carries_share_linear_area() {
    for width in [128, 256, 512] {
        let stats = prefix_carry_stats(width, 1).unwrap();
        assert!(
            stats.and_nodes <= 16 * width,
            "width={width} ANDs={}",
            stats.and_nodes
        );
        for (end, depth) in (1..=width).zip(stats.root_depths) {
            assert!(
                depth <= 4 + 2 * end.ilog2() as usize,
                "prefix {end}: {depth}"
            );
        }
    }
}
