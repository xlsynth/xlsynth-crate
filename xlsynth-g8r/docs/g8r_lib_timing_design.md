## Liberty Timing Data Modes (`xlsynth-g8r`)

This crate supports two Liberty load modes:

- `Library` (default): no timing payloads available.
- `LibraryWithTimingData`: full timing arcs/tables available.

The split exists because ASAP7 timing tables are large enough to noticeably hurt load latency for non-STA flows.

### API Surface

- No timing (fast/default): `load_liberty_from_path(...)` / `load_library_from_path(...)` returning `Library`.
- With timing (STA/intended): `load_liberty_with_timing_data_from_path(...)` / `load_library_with_timing_data_from_path(...)` returning `LibraryWithTimingData`.
- Proto generation flag: `liberty-to-proto --no-timing-data` writes a compact library with timing payloads removed.

Timing-required loader behavior:

- `load_library_with_timing_data_from_path(...)` now returns an error if the input proto has no timing payloads.
- This is intentional so STA call-sites fail fast when pointed at a compact/no-timing artifact.

### ASAP7 Observations (Approximate)

Measured on `2026-02-16` with:

```shell
cargo run -q -p xlsynth-g8r --bin liberty-load-bench -- <file> --warmup 2 --iters 10 --timing-table-loading <mode>
```

Artifact sizes observed in local scripts output:

| Artifact | Bytes | Approx Size |
|---|---:|---:|
| `asap7.proto` (no timing) | `252,837` | `0.24 MiB` |
| `asap7-tt.proto` (with timing) | `45,508,465` | `43.4 MiB` |
| `asap7-tt.proto.gz` (with timing, gzipped) | `16,393,913` | `15.6 MiB` |

Load timings observed with benchmark modes:

| File | Mode | p50 | Notes |
|---|---|---:|---|
| `asap7.proto` | `decode` | `~6.9 ms` | no timing tables present |
| `asap7-tt.proto` | `skip` | `~90-110 ms` | omits heavy timing payload decode work |
| `asap7-tt.proto` | `decode` | `~324 ms` | decodes timing table payloads |
| `asap7-tt.proto` | `materialize` | `~328 ms` | similar to `decode`; occasional outliers |
| `asap7-tt.proto.gz` | `skip` | `~629 ms` | gzip decompression dominates |
| `asap7-tt.proto.gz` | `decode` | `~864 ms` | gzip + timing decode |

Key takeaways:

- For interactive non-STA flows, loading without timing data is much faster.
- For STA work, timing data must be loaded explicitly.
- Whole-file gzip improves storage but substantially increases load time.
- Keeping both modes gives a practical default for normal tools and a clear opt-in path for STA.
