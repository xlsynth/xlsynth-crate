// SPDX-License-Identifier: Apache-2.0

//! Benchmarks for parsing large synthetic gate-level netlists.
//!
//! These benches exercise `netlist::io::parse_netlist_from_path` on
//! deterministically generated single-module netlists with long chains of
//! `INVX1` instances. The synthetic inputs are intended to approximate the
//! tokenization and parsing workload of large Genus-style gate-level netlists
//! without depending on any external files.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::io::Write as IoWrite;
use tempfile::NamedTempFile;
use xlsynth_g8r::netlist;

fn netlist_parse_benchmark(c: &mut Criterion) {
    // Instance counts chosen to keep runtime reasonable while still stressing
    // the parser on large inputs.
    let sizes: &[usize] = &[5_000, 20_000];

    let mut group = c.benchmark_group("netlist_parse_chain_invx1");

    // Keep temporary files alive for the duration of the benchmarks so their
    // paths remain valid.
    let mut temp_files: Vec<NamedTempFile> = Vec::new();

    for &instance_count in sizes {
        let netlist_text = netlist::bench_synth_netlist::make_chain_netlist(instance_count);
        let mut tmp = NamedTempFile::new().expect("create synthetic netlist temp file");
        IoWrite::write_all(&mut tmp, netlist_text.as_bytes())
            .expect("write synthetic netlist text");
        let path_buf = tmp.path().to_path_buf();
        temp_files.push(tmp);

        group.bench_with_input(
            BenchmarkId::from_parameter(instance_count),
            &path_buf,
            |b, path| {
                b.iter(|| {
                    let parsed = netlist::io::parse_netlist_from_path(black_box(path))
                        .expect("synthetic netlist should parse successfully");
                    black_box(parsed);
                });
            },
        );
    }

    group.finish();

    // Ensure temp_files is not dropped until after the group is finished.
    drop(temp_files);
}

criterion_group!(benches, netlist_parse_benchmark);
criterion_main!(benches);
