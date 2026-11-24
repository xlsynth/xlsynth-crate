// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Checks DSLX-derived SystemVerilog port types against PIR evaluation.

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen_fuzz::port_types;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::Icarus) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let (result, progress) = port_types::check_with_progress(data);
    if let Some(report) = progress {
        eprintln!("port-type-coverage {report}");
    }
    xlsynth_codegen_fuzz::fuzz_tool!(result);
});
