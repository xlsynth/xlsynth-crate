// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Proves Yosys-mapped native SystemVerilog equivalent to independent G8R.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::yosys::map_combinational;
use xlsynth_codegen_fuzz::{Profile, generate};
use xlsynth_g8r::prove_gate_fn_equiv_sat::{
    EquivResult, GateFormalBackend, prove_gate_fn_equiv_with_backend_and_options,
};
use xlsynth_g8r_fuzz::external_yosys::required_external_yosys_context;
use xlsynth_g8r_fuzz::fuzz_gate_formal_options;

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::YosysCombinational) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let context = required_external_yosys_context()
        .expect("startup validated Yosys/Liberty setup");
    let mut options = Profile::StockXls.block_options(false, false);
    options.allow_zero_width_ports_and_registers = false;
    options.function_options.allow_zero_width_bits = false;
    options.function_options.max_nodes = 14;
    options.function_options.max_bit_width = 5;
    let package = generate(data, &options);
    let ir = package.to_string();
    let mapped = xlsynth_codegen_fuzz::fuzz_tool!(map_combinational(&package, context));
    let mut source = mapped.source.transition.clone();
    let mut target = mapped.mapped.gate_fn.clone();
    source
        .inputs
        .sort_by(|left, right| left.name.cmp(&right.name));
    source
        .outputs
        .sort_by(|left, right| left.name.cmp(&right.name));
    target
        .inputs
        .sort_by(|left, right| left.name.cmp(&right.name));
    target
        .outputs
        .sort_by(|left, right| left.name.cmp(&right.name));
    assert_eq!(
        source
            .inputs
            .iter()
            .map(|port| (&port.name, port.get_bit_count()))
            .collect::<Vec<_>>(),
        target
            .inputs
            .iter()
            .map(|port| (&port.name, port.get_bit_count()))
            .collect::<Vec<_>>(),
        "mapped input signature changed:\nIR:\n{ir}\nRTL:\n{}\nGV:\n{}",
        mapped.rtl,
        mapped.netlist
    );
    assert_eq!(
        source
            .outputs
            .iter()
            .map(|port| (&port.name, port.get_bit_count()))
            .collect::<Vec<_>>(),
        target
            .outputs
            .iter()
            .map(|port| (&port.name, port.get_bit_count()))
            .collect::<Vec<_>>(),
        "mapped output signature changed:\nIR:\n{ir}\nRTL:\n{}\nGV:\n{}",
        mapped.rtl,
        mapped.netlist
    );
    let result = prove_gate_fn_equiv_with_backend_and_options(
        &source,
        &target,
        GateFormalBackend::Cadical,
        fuzz_gate_formal_options(),
    )
    .unwrap_or_else(|error| {
        panic!(
            "mapped gate equivalence proof failed:\nIR:\n{ir}\nRTL:\n{}\nGV:\n{}\n{error}",
            mapped.rtl, mapped.netlist
        )
    });
    assert_eq!(
        result,
        EquivResult::Proved,
        "mapped gate equivalence found a counterexample:\nIR:\n{ir}\nRTL:\n{}\nGV:\n{}",
        mapped.rtl,
        mapped.netlist
    );
});
