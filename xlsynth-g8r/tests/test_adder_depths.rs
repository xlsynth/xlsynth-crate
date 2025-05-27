use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::fraig::{fraig_optimize, IterationBounds};
use xlsynth_g8r::gate::{AigBitVector, GateFn};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::get_summary_stats::get_gate_depth;
use xlsynth_g8r::ir2gate_utils::{
    gatify_add_brent_kung, gatify_add_carry_select, gatify_add_kogge_stone, gatify_add_ripple_carry,
};
use xlsynth_g8r::use_count::get_id_to_use_count;

fn fraig_optimized(g: GateFn) -> GateFn {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    let (opt, _did_converge, _stats) =
        fraig_optimize(&g, 8, IterationBounds::ToConvergence, &mut rng).unwrap();
    opt
}

fn make_ripple(bits: usize) -> GateFn {
    let mut gb = GateBuilder::new("ripple".to_string(), GateBuilderOptions::opt());
    let lhs = gb.add_input("lhs".to_string(), bits);
    let rhs = gb.add_input("rhs".to_string(), bits);
    let c_in = gb.add_input("c_in".to_string(), 1).get_lsb(0).clone();
    let (c_out, sum) = gatify_add_ripple_carry(&lhs, &rhs, c_in, Some("ripple"), &mut gb);
    gb.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
    gb.add_output("sum".to_string(), sum);
    fraig_optimized(gb.build())
}

fn make_carry_select(bits: usize, parts: &[usize]) -> GateFn {
    let mut gb = GateBuilder::new("carry_select".to_string(), GateBuilderOptions::opt());
    let lhs = gb.add_input("lhs".to_string(), bits);
    let rhs = gb.add_input("rhs".to_string(), bits);
    let c_in = gb.add_input("c_in".to_string(), 1).get_lsb(0).clone();
    let (c_out, res) = gatify_add_carry_select(&lhs, &rhs, parts, c_in.into(), "cs", &mut gb);
    gb.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
    gb.add_output("results".to_string(), res);
    fraig_optimized(gb.build())
}

fn make_kogge_stone(bits: usize) -> GateFn {
    let mut gb = GateBuilder::new("ks".to_string(), GateBuilderOptions::opt());
    let lhs = gb.add_input("lhs".to_string(), bits);
    let rhs = gb.add_input("rhs".to_string(), bits);
    let c_in = gb.add_input("c_in".to_string(), 1).get_lsb(0).clone();
    let (c_out, res) = gatify_add_kogge_stone(&lhs, &rhs, c_in, Some("ks"), &mut gb);
    gb.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
    gb.add_output("results".to_string(), res);
    fraig_optimized(gb.build())
}

fn make_brent_kung(bits: usize) -> GateFn {
    let mut gb = GateBuilder::new("bk".to_string(), GateBuilderOptions::opt());
    let lhs = gb.add_input("lhs".to_string(), bits);
    let rhs = gb.add_input("rhs".to_string(), bits);
    let c_in = gb.add_input("c_in".to_string(), 1).get_lsb(0).clone();
    let (c_out, res) = gatify_add_brent_kung(&lhs, &rhs, c_in, Some("bk"), &mut gb);
    gb.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
    gb.add_output("results".to_string(), res);
    fraig_optimized(gb.build())
}

fn default_parts(bits: usize) -> Vec<usize> {
    if bits <= 2 {
        vec![bits]
    } else {
        let mut rem = bits;
        let mut p = Vec::new();
        while rem >= 2 {
            p.push(2);
            rem -= 2;
        }
        if rem > 0 {
            p.push(rem);
        }
        p
    }
}

fn depth(g: &GateFn) -> usize {
    let use_count = get_id_to_use_count(g);
    let live: Vec<_> = use_count.keys().cloned().collect();
    let stats = get_gate_depth(g, &live);
    stats.deepest_path.len()
}

#[test]
fn adder_depths() {
    let _ = env_logger::builder().is_test(true).try_init();
    const RIPPLE: [usize; 8] = [5, 7, 10, 13, 16, 19, 22, 25];
    const CARRY_SELECT: [usize; 8] = [5, 7, 9, 9, 11, 11, 13, 13];
    const KOGGE: [usize; 8] = [5, 7, 9, 11, 14, 15, 17, 19];
    const BRENT: [usize; 8] = [5, 7, 8, 10, 10, 12, 12, 14];

    for bits in 1usize..=8 {
        let ripple = make_ripple(bits);
        let ks = make_kogge_stone(bits);
        let bk = make_brent_kung(bits);
        let cs = make_carry_select(bits, &default_parts(bits));

        check_equivalence::prove_same_gate_fn_via_ir(&ripple, &ks).unwrap();
        check_equivalence::prove_same_gate_fn_via_ir(&ripple, &bk).unwrap();
        check_equivalence::prove_same_gate_fn_via_ir(&ripple, &cs).unwrap();

        let rd = depth(&ripple);
        let ksd = depth(&ks);
        let bkd = depth(&bk);
        let csd = depth(&cs);
        println!("bits {bits}: rd={rd} csd={csd} ksd={ksd} bkd={bkd}");
        assert_eq!(rd, RIPPLE[bits - 1]);
        assert_eq!(csd, CARRY_SELECT[bits - 1]);
        assert_eq!(ksd, KOGGE[bits - 1]);
        assert_eq!(bkd, BRENT[bits - 1]);
    }
}
