// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use xlsynth_pir_compiler::aot::{
    TypedAotPackageMetadata, TypedIrAotBuildSpec, TypedIrAotPackageBuilder,
};

struct AotCase {
    name: &'static str,
    top: &'static str,
    pir_text: &'static str,
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let metadata_path = manifest_dir.join("src/native_aot_tests_aot_metadata.json");
    println!("cargo:rerun-if-changed={}", metadata_path.display());
    let metadata = TypedAotPackageMetadata::from_json_file(&metadata_path)
        .expect("typed IR AOT metadata should parse");

    let cases = [
        AotCase {
            name: "add_one",
            top: "add_one",
            pir_text: r#"package native_aot_tests

fn add_one(x: bits[42] id=1) -> bits[42] {
  one: bits[42] = literal(value=1, id=2)
  ret out: bits[42] = add(x, one, id=3)
}
"#,
        },
        AotCase {
            name: "add_inputs",
            top: "add_inputs",
            pir_text: r#"package native_aot_tests

fn add_inputs(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret out: bits[8] = add(a, b, id=3)
}
"#,
        },
        AotCase {
            name: "compound_shapes",
            top: "compound_shapes",
            pir_text: r#"package native_aot_tests

fn compound_shapes(input: (bits[42], bits[65][2]) id=1, increment: bits[42] id=2) -> (bits[42], bits[65][2]) {
  base: bits[42] = tuple_index(input, index=0, id=3)
  limbs: bits[65][2] = tuple_index(input, index=1, id=4)
  sum: bits[42] = add(base, increment, id=5)
  ret out: (bits[42], bits[65][2]) = tuple(sum, limbs, id=6)
}
"#,
        },
        AotCase {
            name: "empty_tuple",
            top: "empty_tuple",
            pir_text: r#"package native_aot_tests

fn empty_tuple() -> () {
  ret out: () = tuple(id=1)
}
"#,
        },
        AotCase {
            name: "wide_runtime_ops",
            top: "wide_runtime_ops",
            pir_text: r#"package native_aot_tests

fn wide_runtime_ops(x: bits[129] id=1, y: bits[129] id=2, shift: bits[8] id=3, replacement: bits[73] id=4) -> (bits[129], bits[129], bits[129], bits[129], bits[129], bits[96], bits[32], bits[129], bits[130], bits[130], bits[130], bits[8], bits[130]) {
  product: bits[129] = umul(x, y, id=5)
  signed_product: bits[129] = smul(x, y, id=6)
  quotient: bits[129] = udiv(x, y, id=7)
  left: bits[129] = shll(x, shift, id=8)
  right: bits[129] = shra(x, shift, id=9)
  slice: bits[96] = dynamic_bit_slice(x, shift, width=96, id=10)
  low_slice: bits[32] = dynamic_bit_slice(x, shift, width=32, id=11)
  updated: bits[129] = bit_slice_update(x, shift, replacement, id=12)
  unsigned_parts: (bits[130], bits[130]) = umulp(x, y, id=13)
  unsigned_lhs: bits[130] = tuple_index(unsigned_parts, index=0, id=14)
  unsigned_rhs: bits[130] = tuple_index(unsigned_parts, index=1, id=15)
  unsigned_sum: bits[130] = add(unsigned_lhs, unsigned_rhs, id=16)
  signed_parts: (bits[130], bits[130]) = smulp(x, y, id=17)
  signed_lhs: bits[130] = tuple_index(signed_parts, index=0, id=18)
  signed_rhs: bits[130] = tuple_index(signed_parts, index=1, id=19)
  signed_sum: bits[130] = add(signed_lhs, signed_rhs, id=20)
  hot: bits[130] = one_hot(x, lsb_prio=true, id=21)
  encoded: bits[8] = encode(hot, id=22)
  decoded: bits[130] = decode(shift, width=130, id=23)
  ret out: (bits[129], bits[129], bits[129], bits[129], bits[129], bits[96], bits[32], bits[129], bits[130], bits[130], bits[130], bits[8], bits[130]) = tuple(product, signed_product, quotient, left, right, slice, low_slice, updated, unsigned_sum, signed_sum, hot, encoded, decoded, id=24)
}
"#,
        },
        AotCase {
            name: "events",
            top: "events",
            pir_text: r#"package native_aot_tests

fn events(x: bits[8] id=1, y: bits[8] id=2, ok: bits[1] id=3, emit: bits[1] id=4) -> bits[8] {
  t: token = after_all(id=5)
  covered: () = cover(emit, label="covered", id=6)
  accepted: () = cover(ok, label="accepted", id=7)
  a: token = assert(t, ok, message="bad condition", label="A", id=8)
  emitted_trace: token = trace(a, emit, format="x={} y={:x}", data_operands=[x, y], id=9)
  accepted_trace: token = trace(emitted_trace, ok, format="ok={}", data_operands=[ok], id=10)
  ret out: bits[8] = add(x, y, id=11)
}
"#,
        },
        AotCase {
            name: "assumed_in_bounds",
            top: "assumed_in_bounds",
            pir_text: r#"package native_aot_tests

fn assumed_in_bounds(a: bits[8][2] id=1, v: bits[8] id=2, i: bits[2] id=3) -> bits[8] {
  dead_index: bits[8] = array_index(a, indices=[i], assumed_in_bounds=true, id=4)
  dead_update: bits[8][2] = array_update(a, v, indices=[i], assumed_in_bounds=true, id=5)
  ret out: bits[8] = identity(v, id=6)
}
"#,
        },
        AotCase {
            name: "invokes_and_counted_for",
            top: "invokes_and_counted_for",
            pir_text: r#"package native_aot_tests

fn add_one(x: bits[8] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=2)
  ret result: bits[8] = add(x, one, id=3)
}

fn through(x: bits[8] id=4) -> bits[8] {
  ret result: bits[8] = invoke(x, to_apply=add_one, id=5)
}

fn loop_body(i: bits[8] id=6, carry: bits[8] id=7, increment: bits[8] id=8) -> bits[8] {
  with_i: bits[8] = add(carry, i, id=9)
  ret result: bits[8] = add(with_i, increment, id=10)
}

fn invokes_and_counted_for(init: bits[8] id=11, increment: bits[8] id=12) -> bits[8] {
  invoked: bits[8] = invoke(init, to_apply=through, id=13)
  ret result: bits[8] = counted_for(invoked, trip_count=3, stride=2, body=loop_body, invariant_args=[increment], id=14)
}
"#,
        },
    ];

    let mut builder = TypedIrAotPackageBuilder::new("native_aot_tests", metadata);
    for case in cases {
        builder = builder.add_entrypoint(TypedIrAotBuildSpec {
            name: case.name,
            pir_text: case.pir_text,
            top: case.top,
        });
    }
    builder
        .build()
        .unwrap_or_else(|error| panic!("typed native PIR AOT package should build: {error}"));

    println!("cargo:rerun-if-changed=build.rs");
}
