// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use xlsynth::{IrBits, IrValue};

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

/// Benchmarks getting the bit count of a bits value.
fn bench_get_bit_count(c: &mut Criterion) {
    let bits = IrBits::make_ubits(1, 0).unwrap();
    c.bench_function("get_bit_count", |b| {
        b.iter(|| {
            black_box(bits.get_bit_count());
        });
    });
}

/// Benchmarks getting the bit count of an IrValue that holds bits.
fn bench_get_bit_count_of_value(c: &mut Criterion) {
    let value = IrValue::make_ubits(1, 0).unwrap();
    c.bench_function("get_bit_count_of_value", |b| {
        b.iter(|| {
            let bit_count: usize = value.bit_count().unwrap();
            black_box(bit_count);
        });
    });
}

criterion_group!(
    benches,
    bench_make_bits,
    bench_get_bit_count,
    bench_get_bit_count_of_value
);
criterion_main!(benches);
