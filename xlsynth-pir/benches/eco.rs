// SPDX-License-Identifier: Apache-2.0

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::path::{Path, PathBuf};
use std::time::Duration;
use xlsynth_pir::greedy_matching_ged::GreedyMatchSelector;
use xlsynth_pir::ir::Fn as IrFn;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::matching_ged::compute_fn_edit;

fn top_fn_from_ir_text(ir_text: &str) -> IrFn {
    let pkg = Parser::new(ir_text)
        .parse_and_validate_package()
        .expect("IR parse should succeed");
    let top: IrFn = pkg
        .get_top_fn()
        .expect("package should have a top function")
        .clone();
    top
}

fn bench_greedy_matching_ged(c: &mut Criterion) {
    // Resolve the RISCV IR path located under this crate's benchdata.
    let ir_path: PathBuf =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/benchdata/riscv_simple.opt.ir");
    let ir_text = std::fs::read_to_string(&ir_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", ir_path.display(), e));

    // Use the same IR for both old and new sides (identical graphs on both sides).
    let old_fn = top_fn_from_ir_text(&ir_text);
    let new_fn = top_fn_from_ir_text(&ir_text);

    c.bench_function("greedy_matching_ged_riscv_simple", |b| {
        b.iter(|| {
            let mut selector = GreedyMatchSelector::new(&old_fn, &new_fn);
            let edits = compute_fn_edit(&old_fn, &new_fn, &mut selector)
                .expect("compute_fn_edit (greedy) should succeed");
            black_box(edits);
        })
    });
}

criterion_group! {
    name = eco;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(20))
       .sample_size(10);
    targets = bench_greedy_matching_ged
}
//criterion_group!(eco, bench_greedy_matching_ged);
criterion_main!(eco);
