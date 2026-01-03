// SPDX-License-Identifier: Apache-2.0

use std::io::Cursor;

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use xlsynth_g8r::cut_db::loader::CutDb;

static CUT_DB_BYTES: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/cut_db_v1.bin"));

fn cut_db_load_benchmark(c: &mut Criterion) {
    c.bench_function("cut_db_load_from_reader", |b| {
        b.iter_batched(
            || (),
            |_| {
                let db = CutDb::load_from_reader(Cursor::new(CUT_DB_BYTES)).unwrap();
                black_box(db);
            },
            BatchSize::SmallInput,
        )
    });

    let db = CutDb::load_from_reader(Cursor::new(CUT_DB_BYTES)).unwrap();
    c.bench_function("cut_db_lookup_u16", |b| {
        let mut tt: u16 = 0;
        b.iter(|| {
            tt = tt.wrapping_add(1);
            let (xform, pareto) = db.lookup(tt);
            black_box((xform, pareto.len()));
        })
    });
}

criterion_group!(benches, cut_db_load_benchmark);
criterion_main!(benches);
