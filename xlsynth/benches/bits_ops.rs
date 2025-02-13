// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use xlsynth::IrBits;

/// Benchmarks creating a single boolean `false` value via the Bits creation
/// machinery.
fn bench_make_bits(c: &mut Criterion) {
    c.bench_function("make_bits", |b| {
        b.iter(|| {
            let bits = IrBits::make_ubits(1, 0).unwrap();
            black_box(bits);
        });
    });
}

criterion_group!(benches, bench_make_bits);
criterion_main!(benches);
