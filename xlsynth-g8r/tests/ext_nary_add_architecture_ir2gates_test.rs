// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;

fn architecture_name(mapping: AdderMapping) -> &'static str {
    match mapping {
        AdderMapping::RippleCarry => "ripple_carry",
        AdderMapping::KoggeStone => "kogge_stone",
        AdderMapping::BrentKung => "brent_kung",
    }
}

fn different_global_mapping(mapping: AdderMapping) -> AdderMapping {
    match mapping {
        AdderMapping::RippleCarry => AdderMapping::BrentKung,
        AdderMapping::KoggeStone => AdderMapping::RippleCarry,
        AdderMapping::BrentKung => AdderMapping::KoggeStone,
    }
}

fn build_ext_nary_add_ir_text(arch: Option<&str>) -> String {
    let arch_attr = arch
        .map(|arch| format!(", arch={arch}"))
        .unwrap_or_default();
    format!(
        r#"package test

fn f(a: bits[32] id=1, b: bits[32] id=2) -> bits[32] {{
  ret r: bits[32] = ext_nary_add(a, b{arch_attr}, id=3)
}}
"#
    )
}

fn ext_nary_add_tags_for_ir_text(ir_text: &str, global_mapping: AdderMapping) -> Vec<String> {
    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold: false,
            hash: false,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            adder_mapping: global_mapping,
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .expect("ir2gates");
    out.gatify_output
        .gate_fn
        .gates
        .iter()
        .flat_map(|gate| {
            gate.get_tags()
                .unwrap_or(&[])
                .iter()
                .filter(|tag| tag.starts_with("ext_nary_add_"))
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn ext_nary_add_adder_architecture_controls_lowering() {
    for mapping in [
        AdderMapping::RippleCarry,
        AdderMapping::KoggeStone,
        AdderMapping::BrentKung,
    ] {
        let expected_fragment =
            format!("ext_nary_add_3_{}_output_bit_", architecture_name(mapping));
        let explicit = ext_nary_add_tags_for_ir_text(
            &build_ext_nary_add_ir_text(Some(architecture_name(mapping))),
            different_global_mapping(mapping),
        );
        assert!(
            explicit.iter().any(|tag| tag.contains(&expected_fragment)),
            "expected lowering to tag ext_nary_add outputs with explicit architecture {}; tags={explicit:?}",
            architecture_name(mapping)
        );
        let same_arch_with_original_global = ext_nary_add_tags_for_ir_text(
            &build_ext_nary_add_ir_text(Some(architecture_name(mapping))),
            mapping,
        );
        assert_eq!(
            explicit,
            same_arch_with_original_global,
            "expected explicit arch={} to make ext_nary_add lowering independent of the global adder mapping",
            architecture_name(mapping)
        );
        let implicit = ext_nary_add_tags_for_ir_text(&build_ext_nary_add_ir_text(None), mapping);
        assert!(
            implicit.iter().any(|tag| tag.contains(&expected_fragment)),
            "expected omitted arch to use the global adder mapping {}; tags={implicit:?}",
            architecture_name(mapping)
        );
    }
}
