// SPDX-License-Identifier: Apache-2.0

//! Directed semantic audit of aggregate operations at custom-typed ports.

use std::collections::BTreeSet;

use xlsynth_codegen::{BlockCodegenOptions, Layout};
use xlsynth_pir::ir_random::gather_block_stats;

use crate::dslx_types::TYPE_KIND_COUNT;
use crate::semantics::Trace;

/// Checks every eligible aggregate operation across all custom port
/// families.
#[test]
fn custom_typed_nested_aggregates_preserve_xls_order_and_bounds() {
    let source = r#"package aggregate_audit

top block aggregate_audit(data: ((bits[3], bits[5]), bits[4][2]), index: bits[2], replacement: bits[4], result: (bits[3], bits[5], bits[4], bits[4], bits[4], bits[4], bits[4][2], bits[4][2], bits[4][2], bits[4][2], bits[4][4], bits[4][2])) {
  data: ((bits[3], bits[5]), bits[4][2]) = input_port(name=data, id=1)
  index: bits[2] = input_port(name=index, id=2)
  replacement: bits[4] = input_port(name=replacement, id=3)
  fields: (bits[3], bits[5]) = tuple_index(data, index=0, id=4)
  upper: bits[3] = tuple_index(fields, index=0, id=5)
  lower: bits[5] = tuple_index(fields, index=1, id=6)
  lanes: bits[4][2] = tuple_index(data, index=1, id=7)
  zero: bits[2] = literal(value=0, id=8)
  one: bits[2] = literal(value=1, id=9)
  outside: bits[2] = literal(value=3, id=10)
  first: bits[4] = array_index(lanes, indices=[zero], id=11)
  last: bits[4] = array_index(lanes, indices=[one], id=12)
  clamped: bits[4] = array_index(lanes, indices=[outside], id=13)
  selected: bits[4] = array_index(lanes, indices=[index], id=14)
  updated_first: bits[4][2] = array_update(lanes, replacement, indices=[zero], id=15)
  updated_outside: bits[4][2] = array_update(lanes, replacement, indices=[outside], id=16)
  updated_selected: bits[4][2] = array_update(lanes, replacement, indices=[index], id=17)
  constructed: bits[4][2] = array(first, last, id=18)
  concatenated: bits[4][4] = array_concat(lanes, constructed, id=19)
  sliced: bits[4][2] = array_slice(concatenated, index, width=2, id=20)
  packed: (bits[3], bits[5], bits[4], bits[4], bits[4], bits[4], bits[4][2], bits[4][2], bits[4][2], bits[4][2], bits[4][4], bits[4][2]) = tuple(upper, lower, first, last, clamped, selected, updated_first, updated_outside, updated_selected, constructed, concatenated, sliced, id=21)
  result: () = output_port(packed, name=result, id=22)
}
"#;
    let package = crate::parse_reference(source);
    let scalar: BTreeSet<_> = crate::operations(false)
        .iter()
        .map(|operation| operation.name())
        .collect();
    let eligible: BTreeSet<_> = crate::operations(true)
        .iter()
        .map(|operation| operation.name())
        .filter(|operation| !scalar.contains(operation))
        .collect();
    let live = gather_block_stats(crate::top_block(&package)).live_operations;
    let covered: BTreeSet<_> = eligible
        .iter()
        .copied()
        .filter(|operation| live.contains_key(*operation))
        .collect();
    assert_eq!(
        covered, eligible,
        "eligible aggregate operations must be output-live"
    );

    let mut saw_inbounds_index = false;
    let mut saw_outside_index = false;
    let mut saw_distinct_lanes = false;
    for kind in 0..TYPE_KIND_COUNT {
        let seed = [kind.wrapping_mul(13).wrapping_add(0x6a); 32];
        let trace = Trace::with_seed(&package, seed);
        for sample in &trace.samples {
            let index = sample.inputs["index"].to_u64_if_known().unwrap();
            let data = sample.inputs["data"].to_u64_if_known().unwrap();
            saw_inbounds_index |= index < 2;
            saw_outside_index |= index >= 2;
            saw_distinct_lanes |= data & 0xf != (data >> 4) & 0xf;
        }

        // For the nested-struct family these choices partition the input
        // into 3-, 5-, and 8-bit members, matching the IR tuple layout.
        let mut choices = Vec::with_capacity(16);
        choices.extend([kind, 7, 1, 2]);
        // An SV signed index still has unsigned XLS index semantics.
        choices.extend([1, 0, 0, 0]);
        choices.extend([0, 0, 0, 0]);
        choices.extend([(kind + 5) % TYPE_KIND_COUNT, 4, 1, 3]);
        for layout in [Layout::None, Layout::Pipeline] {
            let options = BlockCodegenOptions {
                layout,
                ..BlockCodegenOptions::default()
            };
            crate::port_types::check_package(package.clone(), &choices, &options, seed)
                .0
                .unwrap_or_else(|error| panic!("kind={kind}, {layout:?}: {error}"));
        }
    }
    assert!(saw_inbounds_index, "no in-bounds dynamic index was checked");
    assert!(
        saw_outside_index,
        "no out-of-bounds signed index was checked"
    );
    assert!(saw_distinct_lanes, "array order was not distinguishable");
}
