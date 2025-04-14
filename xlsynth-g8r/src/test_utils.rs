// SPDX-License-Identifier: Apache-2.0

use crate::gate::GateFn;
use crate::ir2gate;
use crate::xls_ir::ir_parser;
use std::path::Path;
use xlsynth::{mangle_dslx_name, IrFunction, IrPackage};

pub struct LoadedSample {
    pub ir_package: IrPackage,
    pub ir_fn: IrFunction,
    pub gate_fn: GateFn,
}

pub fn load_bf16_mul_sample() -> LoadedSample {
    let dslx_text = "import bfloat16;

fn mul_bf16_bf16(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 {
        bfloat16::mul(x, y)
}";
    let fake_path = Path::new("test_utils.x");
    let ir_result = xlsynth::convert_dslx_to_ir(
        dslx_text,
        fake_path,
        &xlsynth::DslxConvertOptions::default(),
    )
    .unwrap();
    let fn_name = "mul_bf16_bf16";
    let module_name = "test_utils";
    let mangled_name = mangle_dslx_name(module_name, fn_name).unwrap();

    let opt_ir = xlsynth::optimize_ir(&ir_result.ir, &mangled_name).unwrap();
    let ir_text = opt_ir.to_string();

    // Parse with xlsynth-g8r's parser to get its internal IR representation
    let mut parser = ir_parser::Parser::new(&ir_text);
    let g8r_ir_package = parser.parse_package().unwrap();
    let g8r_ir_fn = g8r_ir_package.get_fn(&mangled_name).unwrap();

    // Convert the internal IR function to a GateFn
    let gatify_output = ir2gate::gatify(
        &g8r_ir_fn,
        ir2gate::GatifyOptions {
            fold: true,
            check_equivalence: false,
        },
    )
    .unwrap();
    let gate_fn = gatify_output.gate_fn;

    // Get the final IrFunction from the optimized package
    let ir_fn = opt_ir.get_function(&mangled_name).unwrap();

    LoadedSample {
        ir_package: opt_ir,
        ir_fn,
        gate_fn,
    }
}
