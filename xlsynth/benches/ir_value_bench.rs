use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xlsynth::ir_value::{IrFormatPreference, IrValue};

fn bench_to_u64_old(c: &mut Criterion) {
    let v = IrValue::parse_typed("bits[32]:305419896").unwrap(); // 0x12345678
    c.bench_function("to_u64_old", |b| {
        b.iter(|| {
            // Old implementation: parse string
            let string = v
                .to_string_fmt(IrFormatPreference::UnsignedDecimal)
                .unwrap();
            let number = string.split(':').nth(1).unwrap();
            let val: u64 = number.parse().unwrap();
            black_box(val)
        })
    });
}

fn bench_to_u64_new(c: &mut Criterion) {
    let v = IrValue::parse_typed("bits[32]:305419896").unwrap(); // 0x12345678
    c.bench_function("to_u64_new", |b| {
        b.iter(|| {
            // New implementation: direct bits
            let val = v.to_u64().unwrap();
            black_box(val)
        })
    });
}

criterion_group!(benches, bench_to_u64_old, bench_to_u64_new);
criterion_main!(benches);
