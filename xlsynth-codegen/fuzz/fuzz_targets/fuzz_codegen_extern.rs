// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Checks public foreign-function template and external instance preservation.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_codegen_fuzz::iverilog::assert_extern_semantics;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::{emit, parse_reference, references};

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::Icarus) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let width = usize::from(data.first().copied().unwrap_or(7) % 32) + 1;
    let module_name = format!("external_cell_{}", data.get(1).copied().unwrap_or_default());
    let instance_name = format!(
        "external_instance_{}",
        data.get(2).copied().unwrap_or_default()
    );
    let ir = references::EXTERN
        .replace("bits[8]", &format!("bits[{width}]"))
        .replace("external_cell", &module_name)
        .replace("external_instance", &instance_name);
    let package = parse_reference(&ir);
    let options = BlockCodegenOptions::default();
    let rtl = emit(&package, &options);
    assert_eq!(
        rtl,
        emit(&package, &options),
        "external emission is unstable"
    );
    assert!(
        rtl.contains(&module_name),
        "external template module is missing:\nIR:\n{ir}\nRTL:\n{rtl}"
    );
    assert!(
        rtl.contains(&instance_name),
        "external instance identity is missing:\nIR:\n{ir}\nRTL:\n{rtl}"
    );
    assert!(
        !rtl.contains(&format!("module {module_name}(")),
        "external leaf was emitted as an internal module:\nIR:\n{ir}\nRTL:\n{rtl}"
    );
    for port in [".x(", ".y("] {
        assert!(
            rtl.contains(port),
            "external port binding `{port}` is missing:\nIR:\n{ir}\nRTL:\n{rtl}"
        );
    }

    xlsynth_codegen_fuzz::fuzz_tool!(assert_extern_semantics(&ir, &rtl, width, &module_name));
});
