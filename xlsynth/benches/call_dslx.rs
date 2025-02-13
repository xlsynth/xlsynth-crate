// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lazy_static::lazy_static;

use xlsynth::IrValue;

const DSLX_CODE: &str = "import float32;

fn make_f32(sign: bool, bexp: u8, fraction: u23) -> float32::F32 {
    float32::F32 { sign, bexp, fraction }
}
";

lazy_static! {
    static ref MAKE_F32: xlsynth::IrFunction = {
        let convert_result: xlsynth::DslxToIrPackageResult = xlsynth::convert_dslx_to_ir(
            DSLX_CODE,
            std::path::Path::new("/memfile/make_f32.x"),
            &xlsynth::DslxConvertOptions::default(),
        )
        .expect("convert_dslx_to_ir failed");
        let package: xlsynth::IrPackage = convert_result.ir;
        let mangled =
            xlsynth::mangle_dslx_name("make_f32", "make_f32").expect("mangle_dslx_name failed");
        let function = package.get_function(&mangled).expect("get_function failed");
        function
    };
}

// Benchmark calling the DSLX function with pre-created values.
//
// Notably this still creates a result value that we then are deallocating in
// each trip.
fn bench_call_dslx(c: &mut Criterion) {
    c.bench_function("call_dslx", |b| {
        let s = IrValue::make_ubits(1, 0).unwrap();
        let bexp = IrValue::make_ubits(8, 127).unwrap();
        let frac = IrValue::make_ubits(23, 0).unwrap();
        let args = vec![s, bexp, frac];
        b.iter(|| {
            let result: IrValue = MAKE_F32.interpret(&args).expect("call should succeed");
            black_box(result);
        });
    });
}

criterion_group!(benches, bench_call_dslx);
criterion_main!(benches);
