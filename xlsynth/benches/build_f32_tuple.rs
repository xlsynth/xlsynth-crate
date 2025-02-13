// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xlsynth::IrValue;

fn bench_build_f32_tuple(c: &mut Criterion) {
    c.bench_function("build_f32_tuple", |b| {
        b.iter(|| {
            let sign = IrValue::make_ubits(1, 0).unwrap();
            let bexp = IrValue::make_ubits(8, 127).unwrap();
            let frac = IrValue::make_ubits(23, 0).unwrap();
            let f32_tuple = IrValue::make_tuple(&[sign, bexp, frac]);
            black_box(f32_tuple);
        });
    });
}

fn bench_unpack_f32_tuple(c: &mut Criterion) {
    c.bench_function("unpack_f32_tuple", |b| {
        let orig = IrValue::make_tuple(&[
            IrValue::make_ubits(1, 0).unwrap(),
            IrValue::make_ubits(8, 127).unwrap(),
            IrValue::make_ubits(23, 0).unwrap(),
        ]);
        b.iter(|| {
            let elements = orig.get_elements().unwrap();
            let [sign, bexp, frac] = elements.as_slice() else {
                panic!("expected 3 elements");
            };
            black_box((sign, bexp, frac));
        });
    });
}

criterion_group!(benches, bench_build_f32_tuple, bench_unpack_f32_tuple);
criterion_main!(benches);
