// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Checks public child-block hierarchy ordering and instance independence.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_codegen_fuzz::iverilog::assert_hierarchy_semantics;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::{emit, parse_reference, references};

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::Icarus) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let width = usize::from(data.first().copied().unwrap_or(7) % 32) + 1;
    let child_name = format!("child_{}", data.get(1).copied().unwrap_or_default());
    let ir = references::HIERARCHY
        .replace("bits[8]", &format!("bits[{width}]"))
        .replace("child", &child_name);
    let package = parse_reference(&ir);
    let options = BlockCodegenOptions::default();
    let rtl = emit(&package, &options);
    assert_eq!(
        rtl,
        emit(&package, &options),
        "hierarchy emission is unstable"
    );

    let child_declaration = format!("module {child_name}(");
    let parent_declaration = "module parent(";
    let child_position = rtl
        .find(&child_declaration)
        .unwrap_or_else(|| panic!("child module was not emitted:\nIR:\n{ir}\nRTL:\n{rtl}"));
    let parent_position = rtl
        .find(parent_declaration)
        .unwrap_or_else(|| panic!("parent module was not emitted:\nIR:\n{ir}\nRTL:\n{rtl}"));
    assert_eq!(
        rtl.matches(&child_declaration).count(),
        1,
        "shared child was emitted multiple times:\nIR:\n{ir}\nRTL:\n{rtl}"
    );
    assert!(
        child_position < parent_position,
        "child dependency follows its parent:\nIR:\n{ir}\nRTL:\n{rtl}"
    );
    for instance in [format!("left_{child_name}"), format!("right_{child_name}")] {
        assert!(
            rtl.contains(&instance),
            "independent child instance `{instance}` is missing:\nIR:\n{ir}\nRTL:\n{rtl}"
        );
    }

    xlsynth_codegen_fuzz::fuzz_tool!(assert_hierarchy_semantics(&ir, &rtl, width));
});
