// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lazy_static::lazy_static;
use xlsynth::IrValue;

const DSLX_CODE: &str = "import float32;

fn main(f: float32::F32) -> float32::F32 {
    float32::add(f, f)
}
";

lazy_static! {
    static ref MAIN: xlsynth::IrFunction = {
        let convert_result: xlsynth::DslxToIrPackageResult = xlsynth::convert_dslx_to_ir(
            DSLX_CODE,
            std::path::Path::new("/memfile/main.x"),
            &xlsynth::DslxConvertOptions::default(),
        )
        .unwrap();
        let package: xlsynth::IrPackage = convert_result.ir;
        let mangled = xlsynth::mangle_dslx_name("main", "main").unwrap();
        package.get_function(&mangled).unwrap()
    };
}

fn bench_f32_add_nojit(c: &mut Criterion) {
    let _ = env_logger::builder().is_test(true).try_init();
    let sign = IrValue::make_ubits(1, 0).unwrap();
    let bexp = IrValue::make_ubits(8, 127).unwrap();
    let frac = IrValue::make_ubits(23, 0).unwrap();
    let f = IrValue::make_tuple(&[sign, bexp, frac]);
    c.bench_function("f32_add_nojit", |b| {
        b.iter(|| {
            let result: IrValue = MAIN.interpret(std::slice::from_ref(&f)).unwrap();
            black_box(result);
        });
    });
}

fn bench_f32_add_jit(c: &mut Criterion) {
    let _ = env_logger::builder().is_test(true).try_init();
    let sign = IrValue::make_ubits(1, 0).unwrap();
    let bexp = IrValue::make_ubits(8, 127).unwrap();
    let frac = IrValue::make_ubits(23, 0).unwrap();
    let f = IrValue::make_tuple(&[sign, bexp, frac]);
    let main_jit = xlsynth::IrFunctionJit::new(&MAIN).unwrap();
    c.bench_function("f32_add_jit", |b| {
        b.iter(|| {
            let result: xlsynth::RunResult = main_jit.run(std::slice::from_ref(&f)).unwrap();
            black_box(result);
        });
    });
}

criterion_group!(benches, bench_f32_add_nojit, bench_f32_add_jit);
criterion_main!(benches);
