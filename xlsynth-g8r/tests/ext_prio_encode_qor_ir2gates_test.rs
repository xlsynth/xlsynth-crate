// SPDX-License-Identifier: Apache-2.0

use xlsynth::FnBuilder;
use xlsynth::IrPackage;
use xlsynth_g8r::aig::get_summary_stats::get_summary_stats;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;

fn build_encode_one_hot_ir_text(bit_count: u32, lsb_prio: bool) -> String {
    let mut package = IrPackage::new("sample").expect("create package");
    let fn_name = match lsb_prio {
        true => format!("encode_one_hot_lsb_{bit_count}b"),
        false => format!("encode_one_hot_msb_{bit_count}b"),
    };
    let mut fb = FnBuilder::new(&mut package, &fn_name, /* should_verify= */ true);

    let ty_bits = package.get_bits_type(u64::from(bit_count));
    let input = fb.param("input", &ty_bits);
    let one_hot = fb.one_hot(&input, lsb_prio, Some("one_hot"));
    let encoded = fb.encode(&one_hot, Some("encode"));

    let _ = fb
        .build_with_return_value(&encoded)
        .expect("build function");
    package.set_top_by_name(&fn_name).expect("set top");
    package.to_string()
}

fn stats_for_ir_text_via_ir2gates(
    ir_text: &str,
    enable_rewrite_prio_encode: bool,
) -> (usize, usize) {
    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold: true,
            hash: true,
            check_equivalence: true,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .expect("ir2gates");
    let s = get_summary_stats(&out.gatify_output.gate_fn);
    (s.live_nodes, s.deepest_path)
}

#[test]
fn ir2gates_encode_one_hot_qor_improves_with_prio_encode_rewrite() {
    // Sweep a couple representative power-of-two widths to show the win grows
    // with size (while keeping the test stable).
    for bit_count in [8u32, 16u32, 32u32] {
        for lsb_prio in [true, false] {
            let ir_text = build_encode_one_hot_ir_text(bit_count, lsb_prio);
            let (nodes_off, depth_off) = stats_for_ir_text_via_ir2gates(&ir_text, false);
            let (nodes_on, depth_on) = stats_for_ir_text_via_ir2gates(&ir_text, true);

            eprintln!(
                "encode(one_hot) bit_count={} lsb_prio={} : off nodes={} depth={} ; on nodes={} depth={} ; delta nodes={} depth={}",
                bit_count,
                lsb_prio,
                nodes_off,
                depth_off,
                nodes_on,
                depth_on,
                nodes_on as isize - nodes_off as isize,
                depth_on as isize - depth_off as isize
            );

            assert!(
                nodes_on < nodes_off,
                "expected enable_rewrite_prio_encode=true to reduce live_nodes for bit_count={bit_count} lsb_prio={lsb_prio}; off={nodes_off} on={nodes_on} (depth off={depth_off} on={depth_on})"
            );
        }
    }
}
