// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use flate2::read::MultiGzDecoder;
use prost::Message;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use xlsynth_g8r::liberty::load::{
    TimingTableSummary, count_timing_tables, count_timing_values,
    decode_timing_table_summary_skip_values_from_bytes,
};
use xlsynth_g8r::liberty_proto::Library;

#[derive(Parser, Debug)]
#[command(
    name = "liberty-load-bench",
    about = "Benchmarks Liberty proto load latency (read + optional decompress + decode)"
)]
struct Args {
    /// Input Liberty proto files to benchmark (e.g. .proto, .proto.gz).
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Warmup iterations (per input) that are not included in reported stats.
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Measured iterations (per input).
    #[arg(long, default_value_t = 20)]
    iters: usize,

    /// Controls whether timing table payloads are skipped, decoded, or fully
    /// materialized.
    #[arg(long, value_enum, default_value_t = TimingTableLoading::Decode)]
    timing_table_loading: TimingTableLoading,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum TimingTableLoading {
    /// Skip timing table numeric payload fields while decoding protobuf.
    Skip,
    /// Decode timing table fields as stored in the protobuf.
    Decode,
    /// Decode and materialize timing table values for all tables.
    Materialize,
}

fn read_library_binary(path: &Path) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("opening '{}': {e}", path.display()))?;
    let is_gz = path.extension().map(|e| e == "gz").unwrap_or(false);
    let mut reader: Box<dyn Read> = if is_gz {
        Box::new(MultiGzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut buf = Vec::<u8>::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| format!("reading '{}': {e}", path.display()))?;
    Ok(buf)
}

fn decode_full_library(path: &Path) -> Result<Library, String> {
    let buf = read_library_binary(path)?;
    Library::decode(&buf[..]).map_err(|e| format!("decoding '{}': {e}", path.display()))
}

fn decode_library_skip_timing_payload(path: &Path) -> Result<TimingTableSummary, String> {
    let buf = read_library_binary(path)?;
    decode_timing_table_summary_skip_values_from_bytes(&buf, &path.display().to_string())
        .map_err(|e| format!("{e:#}"))
}

fn percentile(sorted: &[Duration], numer: usize, denom: usize) -> Duration {
    assert!(denom > 0);
    if sorted.is_empty() {
        return Duration::from_secs(0);
    }
    let idx = ((sorted.len() - 1) * numer) / denom;
    sorted[idx]
}

fn fmt_ms(d: Duration) -> String {
    format!("{:.3}", d.as_secs_f64() * 1_000.0)
}

struct BenchSummary {
    cells: usize,
    timing_tables: usize,
    timing_values: usize,
}

fn run_one(
    path: &Path,
    warmup: usize,
    iters: usize,
    timing_table_loading: TimingTableLoading,
) -> Result<(), String> {
    for _ in 0..warmup {
        match timing_table_loading {
            TimingTableLoading::Skip => {
                let _ = decode_library_skip_timing_payload(path)?;
            }
            TimingTableLoading::Decode => {
                let _ = decode_full_library(path)?;
            }
            TimingTableLoading::Materialize => {
                let lib = decode_full_library(path)?;
                let _ = count_timing_values(&lib);
            }
        }
    }

    let mut samples = Vec::<Duration>::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        match timing_table_loading {
            TimingTableLoading::Skip => {
                let _ = decode_library_skip_timing_payload(path)?;
            }
            TimingTableLoading::Decode => {
                let _ = decode_full_library(path)?;
            }
            TimingTableLoading::Materialize => {
                let lib = decode_full_library(path)?;
                let _ = count_timing_values(&lib);
            }
        }
        samples.push(start.elapsed());
    }
    samples.sort();

    let summary = match timing_table_loading {
        TimingTableLoading::Skip => {
            let summary = decode_library_skip_timing_payload(path)?;
            BenchSummary {
                cells: summary.cells,
                timing_tables: summary.timing_tables,
                timing_values: 0,
            }
        }
        TimingTableLoading::Decode => {
            let lib = decode_full_library(path)?;
            BenchSummary {
                cells: lib.cells.len(),
                timing_tables: count_timing_tables(&lib),
                timing_values: 0,
            }
        }
        TimingTableLoading::Materialize => {
            let lib = decode_full_library(path)?;
            BenchSummary {
                cells: lib.cells.len(),
                timing_tables: count_timing_tables(&lib),
                timing_values: count_timing_values(&lib),
            }
        }
    };

    let min = *samples.first().unwrap_or(&Duration::from_secs(0));
    let max = *samples.last().unwrap_or(&Duration::from_secs(0));
    let p50 = percentile(&samples, 50, 100);
    let p95 = percentile(&samples, 95, 100);
    let sum = samples
        .iter()
        .copied()
        .fold(Duration::from_secs(0), |acc, d| acc + d);
    let mean = if iters == 0 {
        Duration::from_secs(0)
    } else {
        sum / (iters as u32)
    };

    println!(
        "file={} bytes={} cells={} timing_tables={} timing_values={} warmup={} iters={} timing_table_loading={}",
        path.display(),
        std::fs::metadata(path)
            .map(|m| m.len())
            .map_err(|e| format!("stat '{}': {e}", path.display()))?,
        summary.cells,
        summary.timing_tables,
        summary.timing_values,
        warmup,
        iters,
        match timing_table_loading {
            TimingTableLoading::Skip => "skip",
            TimingTableLoading::Decode => "decode",
            TimingTableLoading::Materialize => "materialize",
        }
    );
    println!(
        "  min_ms={} p50_ms={} p95_ms={} mean_ms={} max_ms={}",
        fmt_ms(min),
        fmt_ms(p50),
        fmt_ms(p95),
        fmt_ms(mean),
        fmt_ms(max)
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    if args.iters == 0 {
        return Err("--iters must be >= 1".to_string());
    }
    for input in &args.inputs {
        run_one(input, args.warmup, args.iters, args.timing_table_loading)?;
    }
    Ok(())
}
