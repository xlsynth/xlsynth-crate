// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{IrBits, IrValue};
use xlsynth_pir_compiler::PirFunctionCompiler;

/// Creates selectors with set bits on either side of native limb boundaries.
fn selector(width: usize, set_bits: &[usize]) -> IrValue {
    let mut bytes = vec![0; width.div_ceil(8)];
    for bit in set_bits {
        assert!(*bit < width);
        bytes[bit / 8] |= 1 << (bit % 8);
    }
    IrValue::Bits(IrBits::from_le_bytes(width, &bytes).unwrap())
}

/// Uses matching leaf patterns so bitwise OR expectations apply to every shape.
fn value_shapes(pattern: u8) -> Vec<IrValue> {
    let scalar = IrValue::make_ubits(8, u64::from(pattern)).unwrap();
    let zero = IrValue::make_ubits(0, 0).unwrap();
    let mut wide_bytes = vec![0; 97usize.div_ceil(8)];
    wide_bytes[0] = pattern;
    wide_bytes[8] = pattern;
    let wide = IrValue::Bits(IrBits::from_le_bytes(97, &wide_bytes).unwrap());
    vec![
        scalar.clone(),
        wide.clone(),
        IrValue::make_array(&[scalar.clone(), scalar.clone()]).unwrap(),
        IrValue::make_tuple(&[scalar, zero.clone(), wide]),
        zero.clone(),
        IrValue::make_array(&[zero.clone(), zero]).unwrap(),
        IrValue::make_tuple(&[]),
    ]
}

/// Checks explicit expectations against both execution engines.
fn assert_results(ir: &str, cases: Vec<(Vec<IrValue>, IrValue)>) {
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let compiler = PirFunctionCompiler::compile_package(&package).unwrap();
    for (args, expected) in cases {
        let FnEvalResult::Success(interpreted) = eval_fn_in_package(&package, function, &args)
        else {
            panic!("interpreter failed for {args:?}\n{ir}");
        };
        assert_eq!(interpreted.value, expected, "interpreter\n{ir}\n{args:?}");
        let actual = compiler.run_ir_values(&args).unwrap();
        assert_eq!(actual, expected, "compiler\n{ir}\n{args:?}");
    }
}

#[test]
fn default_only_sel_accepts_a_wide_computed_selector_and_zero_width_result() {
    assert_results(
        r#"package test
top fn f() -> bits[0] {
  concat.2: bits[0] = concat(id=2)
  zero_ext.3: bits[479] = zero_ext(concat.2, new_bit_count=479, id=3)
  ret sel.4: bits[0] = sel(zero_ext.3, cases=[], default=concat.2, id=4)
}
"#,
        vec![(vec![], IrValue::make_ubits(0, 0).unwrap())],
    );
}

#[test]
fn default_only_sel_preserves_scalar_aggregate_and_zero_sized_values() {
    let mut values = value_shapes(17);
    values.push(IrValue::make_token());
    for value in values {
        let ty = value.type_();
        for width in [0, 1, 8, 64, 65, 479, 1024] {
            let ir = format!(
                r#"package test
top fn f(s: bits[{width}] id=1, d: {ty} id=2) -> {ty} {{
  ret result: {ty} = sel(s, cases=[], default=d, id=3)
}}
"#
            );
            let set_bits = if width == 0 { vec![] } else { vec![width - 1] };
            assert_results(
                &ir,
                vec![(
                    vec![selector(width, &set_bits), value.clone()],
                    value.clone(),
                )],
            );
        }
    }
}

#[test]
fn default_only_scalar_sel_materializes_its_selected_storage_view() {
    for width in [0, 1, 479] {
        let ir = format!(
            r#"package test
top fn f(s: bits[{width}] id=1) -> (bits[8], bits[8][2]) {{
  data: bits[8][2] = literal(value=[17, 29], id=2)
  index: bits[1] = literal(value=0, id=3)
  element: bits[8] = array_index(data, indices=[index], id=4)
  selected: bits[8] = sel(s, cases=[], default=element, id=5)
  replacement: bits[8] = literal(value=34, id=6)
  updated: bits[8][2] = array_update(data, replacement, indices=[index], id=7)
  ret result: (bits[8], bits[8][2]) = tuple(selected, updated, id=8)
}}
"#
        );
        assert_results(
            &ir,
            vec![(
                vec![selector(width, &[])],
                IrValue::parse_typed("(bits[8]:17, [bits[8]:34, bits[8]:29])").unwrap(),
            )],
        );
    }
}

#[test]
fn sel_checks_high_selector_limbs_before_selecting_a_case() {
    for ((a, b), default) in value_shapes(1)
        .into_iter()
        .zip(value_shapes(2))
        .zip(value_shapes(4))
    {
        let ty = a.type_();
        for width in [8, 64, 65, 479, 1024] {
            let ir = format!(
                r#"package test
top fn f(s: bits[{width}] id=1, a: {ty} id=2, b: {ty} id=3, d: {ty} id=4) -> {ty} {{
  ret result: {ty} = sel(s, cases=[a, b], default=d, id=5)
}}
"#
            );
            let cases = [
                (vec![], a.clone()),
                (vec![0], b.clone()),
                (vec![1], default.clone()),
                (vec![width - 1], default.clone()),
                (vec![width - 1, 0], default.clone()),
            ]
            .into_iter()
            .map(|(set_bits, expected)| {
                (
                    vec![
                        selector(width, &set_bits),
                        a.clone(),
                        b.clone(),
                        default.clone(),
                    ],
                    expected,
                )
            })
            .collect();
            assert_results(&ir, cases);
        }
    }
}

#[test]
fn complete_sel_accepts_zero_and_one_bit_selectors() {
    for width in [0, 1] {
        let names = if width == 0 { "a" } else { "a, b" };
        let ir = format!(
            r#"package test
top fn f(s: bits[{width}] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[8] {{
  ret result: bits[8] = sel(s, cases=[{names}], id=4)
}}
"#
        );
        let a = IrValue::make_ubits(8, 17).unwrap();
        let b = IrValue::make_ubits(8, 29).unwrap();
        let mut cases = vec![(vec![selector(width, &[]), a.clone(), b.clone()], a.clone())];
        if width == 1 {
            cases.push((vec![selector(width, &[0]), a, b.clone()], b));
        }
        assert_results(&ir, cases);
    }
}

#[test]
fn priority_and_one_hot_selects_cross_selector_limb_boundaries() {
    for op in ["priority_sel", "one_hot_sel"] {
        for shape in 0..value_shapes(0).len() {
            let a = value_shapes(1).remove(shape);
            let b = value_shapes(2).remove(shape);
            let d = value_shapes(4).remove(shape);
            let z = value_shapes(0).remove(shape);
            let ty = a.type_();
            let case_names = (0..70)
                .map(|index| match index {
                    0 => "a",
                    63 => "b",
                    64 | 69 => "d",
                    _ => "z",
                })
                .collect::<Vec<_>>()
                .join(", ");
            let default = if op == "priority_sel" {
                ", default=d"
            } else {
                ""
            };
            let ir = format!(
                r#"package test
top fn f(s: bits[70] id=1, a: {ty} id=2, b: {ty} id=3, d: {ty} id=4, z: {ty} id=5) -> {ty} {{
  ret result: {ty} = {op}(s, cases=[{case_names}]{default}, id=6)
}}
"#
            );
            let expected_patterns = if op == "priority_sel" {
                [4, 1, 2, 4, 4, 1, 2]
            } else {
                [0, 1, 2, 4, 4, 5, 6]
            };
            let cases = [
                vec![],
                vec![0],
                vec![63],
                vec![64],
                vec![69],
                vec![0, 64],
                vec![63, 69],
            ]
            .into_iter()
            .zip(expected_patterns)
            .map(|(set_bits, expected_pattern)| {
                (
                    vec![
                        selector(70, &set_bits),
                        a.clone(),
                        b.clone(),
                        d.clone(),
                        z.clone(),
                    ],
                    value_shapes(expected_pattern).remove(shape),
                )
            })
            .collect();
            assert_results(&ir, cases);
        }
    }
}
