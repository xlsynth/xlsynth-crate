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
        .unwrap();
        let package: xlsynth::IrPackage = convert_result.ir;
        let mangled = xlsynth::mangle_dslx_name("make_f32", "make_f32").unwrap();
        package.get_function(&mangled).unwrap()
    };
}

// Benchmarks calling the DSLX function with pre-created values.
//
// Notably this still creates a result value that we are then deallocating in
// each trip.
fn bench_call_dslx(c: &mut Criterion) {
    c.bench_function("call_dslx", |b| {
        let s = IrValue::make_ubits(1, 0).unwrap();
        let bexp = IrValue::make_ubits(8, 127).unwrap();
        let frac = IrValue::make_ubits(23, 0).unwrap();
        let args = vec![s, bexp, frac];
        b.iter(|| {
            let result: IrValue = MAKE_F32.interpret(&args).unwrap();
            black_box(result);
        });
    });
}

criterion_group!(benches, bench_call_dslx);
criterion_main!(benches);
