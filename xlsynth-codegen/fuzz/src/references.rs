// SPDX-License-Identifier: Apache-2.0

//! Public, synthetic block references shared by focused codegen targets.

/// Narrowing followed by replacement and concatenation exercises width
/// boundaries.
pub const NARROW_SLICE_UPDATE: &str = r#"package public_narrow_slice_update
top block narrow_slice_update(in0: bits[3], in1: bits[3], out: bits[5]) {
  in0: bits[3] = input_port(name=in0, id=1)
  in1: bits[3] = input_port(name=in1, id=2)
  dynamic_bit_slice.3: bits[1] = dynamic_bit_slice(in0, in1, width=1, id=3)
  literal.4: bits[4] = literal(value=1, id=4)
  decode.5: bits[2] = decode(in1, width=2, id=5)
  bit_slice_update.6: bits[3] = bit_slice_update(in0, decode.5, dynamic_bit_slice.3, id=6)
  sel.7: bits[4] = sel(in1, cases=[literal.4, literal.4, literal.4, literal.4], default=literal.4, id=7)
  bit_slice.8: bits[1] = bit_slice(decode.5, start=1, width=1, id=8)
  ugt.9: bits[1] = ugt(in0, in1, id=9)
  concat.10: bits[5] = concat(decode.5, bit_slice_update.6, id=10)
  encode.11: bits[3] = encode(concat.10, id=11)
  xor.12: bits[4] = xor(sel.7, id=12)
  shrl.13: bits[5] = shrl(concat.10, literal.4, id=13)
  ne.14: bits[1] = ne(in0, in0, id=14)
  out: () = output_port(shrl.13, name=out, id=15)
}
"#;

/// Register-free scalar block with multiple independently observable outputs.
pub const COMBINATIONAL: &str = r#"package public_combinational

top block arithmetic(a: bits[8], b: bits[8], sum: bits[8], difference: bits[8]) {
  a: bits[8] = input_port(name=a, id=1)
  b: bits[8] = input_port(name=b, id=2)
  added: bits[8] = add(a, b, id=3)
  subtracted: bits[8] = sub(a, b, id=4)
  sum: () = output_port(added, name=sum, id=5)
  difference: () = output_port(subtracted, name=difference, id=6)
}
"#;

/// Feedback register with a load enable and an active-low synchronous reset.
pub const SEQUENTIAL: &str = r#"package public_sequential

top block accumulator(clk: clock, rst_n: bits[1], data: bits[8], enable: bits[1], result: bits[8]) {
  #![reset(port="rst_n", asynchronous=false, active_low=true)]
  reg state(bits[8], reset_value=3)
  rst_n: bits[1] = input_port(name=rst_n, id=1)
  data: bits[8] = input_port(name=data, id=2)
  enable: bits[1] = input_port(name=enable, id=3)
  current: bits[8] = register_read(register=state, id=4)
  next_value: bits[8] = add(current, data, id=5)
  update: () = register_write(next_value, register=state, load_enable=enable, reset=rst_n, id=6)
  result: () = output_port(current, name=result, id=7)
}
"#;

/// Array/tuple inputs that exercise dynamic indexing and aggregate flattening.
pub const AGGREGATE: &str = r#"package public_aggregate

top block aggregate(array: bits[8][3], pair: (bits[8], bits[8]), index: bits[2], result: bits[8]) {
  array: bits[8][3] = input_port(name=array, id=1)
  pair: (bits[8], bits[8]) = input_port(name=pair, id=2)
  index: bits[2] = input_port(name=index, id=3)
  element: bits[8] = array_index(array, indices=[index], id=4)
  first: bits[8] = tuple_index(pair, index=0, id=5)
  value: bits[8] = add(element, first, id=6)
  result: () = output_port(value, name=result, id=7)
}
"#;

/// Two instances of one child block, exercising dependency deduplication.
pub const HIERARCHY: &str = r#"package public_hierarchy

block child(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  one: bits[8] = literal(value=1, id=2)
  incremented: bits[8] = add(value, one, id=3)
  result: () = output_port(incremented, name=result, id=4)
}

top block parent(left: bits[8], right: bits[8], result: bits[8]) {
  instantiation left_child(block=child, kind=block)
  instantiation right_child(block=child, kind=block)
  left: bits[8] = input_port(name=left, id=5)
  right: bits[8] = input_port(name=right, id=6)
  left_input: () = instantiation_input(left, instantiation=left_child, port_name=value, id=7)
  right_input: () = instantiation_input(right, instantiation=right_child, port_name=value, id=8)
  left_result: bits[8] = instantiation_output(instantiation=left_child, port_name=result, id=9)
  right_result: bits[8] = instantiation_output(instantiation=right_child, port_name=result, id=10)
  combined: bits[8] = xor(left_result, right_result, id=11)
  result: () = output_port(combined, name=result, id=12)
}
"#;

/// Opaque external instance using a standard XLS foreign-function template.
pub const EXTERN: &str = r#"package public_extern

#[ffi_proto("""code_template: "external_cell {fn} (.x({x}), .y({return}));"
""")]
fn external_identity(x: bits[8] id=1) -> bits[8] {
  ret identity.2: bits[8] = identity(x, id=2)
}

top block wrapper(x: bits[8], result: bits[8]) {
  instantiation external_instance(foreign_function=external_identity, kind=extern)
  x: bits[8] = input_port(name=x, id=3)
  connected: () = instantiation_input(x, instantiation=external_instance, port_name=x, id=4)
  value: bits[8] = instantiation_output(instantiation=external_instance, port_name=return, id=5)
  result: () = output_port(value, name=result, id=6)
}
"#;

/// Constructs a public signed or unsigned product-pair reference.
pub fn partial_product(
    signed: bool,
    lhs_width: usize,
    rhs_width: usize,
    result_width: usize,
) -> String {
    let pair_op = if signed { "smulp" } else { "umulp" };
    let multiply_op = if signed { "smul" } else { "umul" };
    format!(
        r#"package public_partial_product

top block partial_product(lhs: bits[{lhs_width}], rhs: bits[{rhs_width}], first: bits[{result_width}], second: bits[{result_width}], combined: bits[{result_width}], expected: bits[{result_width}]) {{
  lhs: bits[{lhs_width}] = input_port(name=lhs, id=1)
  rhs: bits[{rhs_width}] = input_port(name=rhs, id=2)
  pair: (bits[{result_width}], bits[{result_width}]) = {pair_op}(lhs, rhs, id=3)
  part0: bits[{result_width}] = tuple_index(pair, index=0, id=4)
  part1: bits[{result_width}] = tuple_index(pair, index=1, id=5)
  modular_sum: bits[{result_width}] = add(part0, part1, id=6)
  reference: bits[{result_width}] = {multiply_op}(lhs, rhs, id=7)
  first: () = output_port(part0, name=first, id=8)
  second: () = output_port(part1, name=second, id=9)
  combined: () = output_port(modular_sum, name=combined, id=10)
  expected: () = output_port(reference, name=expected, id=11)
}}
"#
    )
}
