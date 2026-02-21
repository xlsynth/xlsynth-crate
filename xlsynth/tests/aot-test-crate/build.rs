// SPDX-License-Identifier: Apache-2.0

use xlsynth::aot_builder::{emit_aot_module_from_ir_text, AotBuildSpec};

fn main() {
    let add_one_ir = r#"package aot_tests

top fn add_one(x: bits[8]) -> bits[8] {
  one: bits[8] = literal(value=1)
  ret out: bits[8] = add(x, one)
}
"#;

    let add_inputs_ir = r#"package aot_tests

top fn add_inputs(a: bits[8], b: bits[8]) -> bits[8] {
  ret out: bits[8] = add(a, b)
}
"#;

    let compound_shapes_ir = r#"package aot_tests

top fn compound_shapes(lhs: (bits[8], bits[16][2]), rhs: (bits[16], bits[8][2])) -> (bits[16][2], (bits[8], bits[16]), bits[8][2]) {
  lhs_base: bits[8] = tuple_index(lhs, index=0)
  lhs_arr: bits[16][2] = tuple_index(lhs, index=1)
  rhs_base: bits[16] = tuple_index(rhs, index=0)
  rhs_arr: bits[8][2] = tuple_index(rhs, index=1)

  i0: bits[1] = literal(value=0)
  i1: bits[1] = literal(value=1)

  l0: bits[16] = array_index(lhs_arr, indices=[i0])
  l1: bits[16] = array_index(lhs_arr, indices=[i1])
  r0: bits[8] = array_index(rhs_arr, indices=[i0])
  r1: bits[8] = array_index(rhs_arr, indices=[i1])

  r0_wide: bits[16] = zero_ext(r0, new_bit_count=16)
  s0: bits[16] = add(l0, r0_wide)
  s1: bits[16] = add(l1, rhs_base)
  out_wide_arr: bits[16][2] = array(s0, s1)

  out_pair: (bits[8], bits[16]) = tuple(r1, rhs_base)

  out_narrow0: bits[8] = add(lhs_base, r0)
  out_narrow1: bits[8] = add(lhs_base, r1)
  out_narrow_arr: bits[8][2] = array(out_narrow0, out_narrow1)

  ret out: (bits[16][2], (bits[8], bits[16]), bits[8][2]) = tuple(out_wide_arr, out_pair, out_narrow_arr)
}
"#;

    let empty_tuple_ir = r#"package aot_tests

top fn make_empty_tuple() -> () {
  ret out: () = tuple()
}
"#;

    let wide_sizes_ir = r#"package aot_tests

top fn wide_sizes(input: (bits[1], bits[7], bits[8], bits[16], bits[32], bits[64], bits[65], bits[73], bits[127], bits[65][2])) -> (bits[1], bits[7], bits[8], bits[16], bits[32], bits[64], bits[65], bits[73], bits[127], bits[65][2]) {
  f0: bits[1] = tuple_index(input, index=0)
  f1: bits[7] = tuple_index(input, index=1)
  f2: bits[8] = tuple_index(input, index=2)
  f3: bits[16] = tuple_index(input, index=3)
  f4: bits[32] = tuple_index(input, index=4)
  f5: bits[64] = tuple_index(input, index=5)
  f6: bits[65] = tuple_index(input, index=6)
  f7: bits[73] = tuple_index(input, index=7)
  f8: bits[127] = tuple_index(input, index=8)
  f9: bits[65][2] = tuple_index(input, index=9)
  ret out: (bits[1], bits[7], bits[8], bits[16], bits[32], bits[64], bits[65], bits[73], bits[127], bits[65][2]) = tuple(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9)
}
"#;

    let trace_assert_ir = r#"package aot_tests

top fn trace_assert_pair(tok: token, pair: (bits[8], bits[8])) -> (bits[8], bits[8]) {
  a: bits[8] = tuple_index(pair, index=0)
  b: bits[8] = tuple_index(pair, index=1)
  sum: bits[8] = add(a, b)
  min_sum: bits[8] = literal(value=5)
  ok: bits[1] = uge(sum, min_sum)
  always_on: bits[1] = literal(value=1)
  asserted: token = assert(tok, ok, message="sum must be >= 5")
  traced: token = trace(asserted, always_on, format="sum: {}", data_operands=[sum])
  ret out: (bits[8], bits[8]) = tuple(sum, b)
}
"#;

    let add_one = emit_aot_module_from_ir_text(&AotBuildSpec {
        name: "add_one",
        ir_text: add_one_ir,
        top: "add_one",
    })
    .expect("AOT compile for add_one should succeed");

    let add_inputs = emit_aot_module_from_ir_text(&AotBuildSpec {
        name: "add_inputs",
        ir_text: add_inputs_ir,
        top: "add_inputs",
    })
    .expect("AOT compile for add_inputs should succeed");

    let compound_shapes = emit_aot_module_from_ir_text(&AotBuildSpec {
        name: "compound_shapes",
        ir_text: compound_shapes_ir,
        top: "compound_shapes",
    })
    .expect("AOT compile for compound_shapes should succeed");

    let empty_tuple = emit_aot_module_from_ir_text(&AotBuildSpec {
        name: "empty_tuple",
        ir_text: empty_tuple_ir,
        top: "make_empty_tuple",
    })
    .expect("AOT compile for make_empty_tuple should succeed");

    let wide_sizes = emit_aot_module_from_ir_text(&AotBuildSpec {
        name: "wide_sizes",
        ir_text: wide_sizes_ir,
        top: "wide_sizes",
    })
    .expect("AOT compile for wide_sizes should succeed");

    let trace_assert = emit_aot_module_from_ir_text(&AotBuildSpec {
        name: "trace_assert",
        ir_text: trace_assert_ir,
        top: "trace_assert_pair",
    })
    .expect("AOT compile for trace_assert_pair should succeed");

    println!(
        "cargo:rustc-env=XLSYNTH_AOT_ADD_ONE_RS={}",
        add_one.rust_file.display()
    );
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_ADD_INPUTS_RS={}",
        add_inputs.rust_file.display()
    );
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_COMPOUND_SHAPES_RS={}",
        compound_shapes.rust_file.display()
    );
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_EMPTY_TUPLE_RS={}",
        empty_tuple.rust_file.display()
    );
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_WIDE_SIZES_RS={}",
        wide_sizes.rust_file.display()
    );
    println!(
        "cargo:rustc-env=XLSYNTH_AOT_TRACE_ASSERT_RS={}",
        trace_assert.rust_file.display()
    );

    println!("cargo:rerun-if-changed=build.rs");
}
