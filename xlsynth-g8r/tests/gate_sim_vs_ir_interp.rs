// SPDX-License-Identifier: Apache-2.0

//! Tests that compare the results of g8r gate simulation with the results of
//! XLS IR interpretation.
//!
//! This is useful for ensuring the gate simulation is correct.

use std::collections::HashMap;

use half::bf16;
use rand::Rng;
use xlsynth_g8r::aig::AigRef;
use xlsynth_g8r::aig::get_summary_stats::get_gate_depth;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
#[allow(unused_imports)]
use xlsynth_g8r::assert_within;
use xlsynth_g8r::test_utils::{
    Opt, flat_ir_bits_to_ir_value_bf16, ir_value_bf16_to_flat_ir_bits, load_bf16_mul_sample,
    make_bf16,
};
use xlsynth_g8r::use_count::get_id_to_use_count;

#[test]
fn test_bf16_mul_zero_zero() {
    let _ = env_logger::builder().is_test(true).try_init();
    let loaded_sample = load_bf16_mul_sample(Opt::Yes);
    let ir_fn = loaded_sample.ir_fn;
    let gate_fn = loaded_sample.gate_fn;

    let arg0 = make_bf16(bf16::from_f32(0.0));
    let arg1 = make_bf16(bf16::from_f32(0.0));

    let ir_result = ir_fn.interpret(&[arg0.clone(), arg1.clone()]).unwrap();

    let gate_arg0_bits = ir_value_bf16_to_flat_ir_bits(&arg0);
    let gate_arg1_bits = ir_value_bf16_to_flat_ir_bits(&arg1);
    let gate_result_sim =
        gate_sim::eval(&gate_fn, &[gate_arg0_bits, gate_arg1_bits], Collect::None);

    // GateFn outputs are flattened. The mul_bf16 returns a single BF16 tuple.
    assert_eq!(gate_result_sim.outputs.len(), 1);
    let gate_result_bits = &gate_result_sim.outputs[0];
    let gate_result = flat_ir_bits_to_ir_value_bf16(gate_result_bits);

    assert_eq!(ir_result, gate_result);
}

/// Pushes concrete random input vectors through both the IR function and the
/// g8r mapped version and checks they are equivalent.
#[test]
fn test_bf16_mul_random() {
    let _ = env_logger::builder().is_test(true).try_init();
    let loaded_sample = load_bf16_mul_sample(Opt::Yes);
    let ir_fn = loaded_sample.ir_fn;
    let gate_fn = loaded_sample.gate_fn;

    let mut rng = rand::thread_rng();

    for i in 0..256 {
        let f0_bits: u16 = rng.r#gen();
        let f1_bits: u16 = rng.r#gen();

        let f0_bf16 = bf16::from_bits(f0_bits);
        let f1_bf16 = bf16::from_bits(f1_bits);

        log::debug!(
            "Testing iter {} with input bits: {:#06x}, {:#06x} (bf16: {}, {})",
            i,
            f0_bits,
            f1_bits,
            f0_bf16,
            f1_bf16
        );

        let arg0 = make_bf16(f0_bf16);
        let arg1 = make_bf16(f1_bf16);

        let ir_result = ir_fn.interpret(&[arg0.clone(), arg1.clone()]).unwrap();

        let gate_arg0_bits = ir_value_bf16_to_flat_ir_bits(&arg0);
        let gate_arg1_bits = ir_value_bf16_to_flat_ir_bits(&arg1);
        let gate_result_sim =
            gate_sim::eval(&gate_fn, &[gate_arg0_bits, gate_arg1_bits], Collect::None);

        assert_eq!(gate_result_sim.outputs.len(), 1);
        let gate_result_bits = &gate_result_sim.outputs[0];
        let gate_result = flat_ir_bits_to_ir_value_bf16(gate_result_bits);

        assert_eq!(
            ir_result, gate_result,
            "Mismatch at iteration {} with input bits: {:#06x}, {:#06x} (bf16: {}, {})",
            i, f0_bits, f1_bits, f0_bf16, f1_bf16
        );
    }
}

#[test]
fn test_bf16_mul_g8r_stats() {
    let _ = env_logger::builder().is_test(true).try_init();
    log::info!("loading bf16 mul sample");
    let loaded_sample = load_bf16_mul_sample(Opt::Yes);
    let gate_fn = &loaded_sample.gate_fn;

    log::info!("getting id to use count");
    let id_to_use_count: HashMap<AigRef, usize> = get_id_to_use_count(gate_fn);
    let live_node_count = id_to_use_count.len();
    assert_within!(live_node_count as isize, 1153 as isize, 20 as isize);

    log::info!("getting gate depth");
    let live_nodes: Vec<AigRef> = id_to_use_count.keys().cloned().collect();
    let depth_stats = get_gate_depth(gate_fn, &live_nodes);
    let max_depth = depth_stats.deepest_path.len();

    assert_within!(max_depth as isize, 105 as isize, 10 as isize);
}
