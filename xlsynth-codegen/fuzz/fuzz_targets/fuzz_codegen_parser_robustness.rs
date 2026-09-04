// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Checks accepted block packages never cause unstable codegen outcomes.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::ir_parser::Parser;

fuzz_target!(|data: &[u8]| {
    // Non-UTF-8 input is outside the textual PIR parser's input contract.
    let Ok(source) = std::str::from_utf8(data) else {
        return;
    };
    // Invalid PIR is expected when arbitrary bytes are interpreted as source.
    let Ok(package) = Parser::new(source).parse_and_validate_package() else {
        return;
    };
    // Packages containing only functions are outside the block-codegen API.
    if package.get_top_block().is_none() {
        return;
    }
    let options = BlockCodegenOptions::default();
    let first = emit_system_verilog(&package, &options);
    let second = emit_system_verilog(&package, &options);
    assert_eq!(
        format!("{first:?}"),
        format!("{second:?}"),
        "block codegen changed result or diagnostic between calls:\n{source}"
    );
});
