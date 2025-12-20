// SPDX-License-Identifier: Apache-2.0

//! Generates the 4-input cut database (canonical NPN entries) and writes it to
//! disk.

use std::io;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use rayon::ThreadPoolBuilder;

use xlsynth_g8r::cut_db::enumerate::{EnumerateOptions, enumerate_full_space};
use xlsynth_g8r::cut_db::serdes::{
    CanonDbOnDisk, CanonDbOnDiskV1, canon_from_full_space, upgrade_v1_to_v2,
};

#[derive(Debug, Parser)]
#[command(name = "gen-cut-db")]
#[command(about = "Generate 4-input AIG cut DB (NPN canonical, Pareto frontier)")]
struct Args {
    /// Output path for the bincode DB artifact.
    #[arg(long)]
    out: PathBuf,

    /// Repack an existing bincode artifact into the current on-disk format.
    ///
    /// This is intended for upgrading older in-tree artifacts without rerunning
    /// the (expensive) full-space enumeration.
    #[arg(long)]
    repack_from: Option<PathBuf>,

    /// Optional maximum AND-count to explore (for debugging).
    #[arg(long)]
    max_ands: Option<u16>,

    /// Emit progress every N worklist pops during enumeration.
    #[arg(long, default_value_t = 100_000)]
    progress_every_pops: u64,

    /// Chunk size for parallelizing the p × all_points expansion.
    #[arg(long, default_value_t = 4096)]
    parallel_chunk_size: usize,

    /// Number of rayon worker threads to use (defaults to rayon's global
    /// default).
    #[arg(long)]
    threads: Option<usize>,
}

fn main() -> io::Result<()> {
    env_logger::init();
    let args = Args::parse();

    if let Some(n) = args.threads {
        ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    }
    log::info!("gen-cut-db: rayon threads={}", rayon::current_num_threads());

    let start = Instant::now();

    let canon: CanonDbOnDisk = if let Some(repack_from) = args.repack_from.as_ref() {
        let bytes = std::fs::read(repack_from)?;
        let v2: Result<CanonDbOnDisk, Box<bincode::ErrorKind>> =
            bincode::deserialize(bytes.as_slice());
        match v2 {
            Ok(db) => db,
            Err(_) => {
                let v1: CanonDbOnDiskV1 = bincode::deserialize(bytes.as_slice())
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                upgrade_v1_to_v2(v1).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
            }
        }
    } else {
        let full = enumerate_full_space(EnumerateOptions {
            max_ands: args.max_ands,
            progress_every_pops: Some(args.progress_every_pops),
            parallel_chunk_size: args.parallel_chunk_size,
        });
        let covered = full.covered_count();

        let canon =
            canon_from_full_space(&full).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        eprintln!(
            "gen-cut-db: built {} canonical entries (full covered: {}/65536) in {:?}",
            canon.entries.len(),
            covered,
            start.elapsed()
        );
        canon
    };

    let mut f = std::fs::File::create(&args.out)?;
    bincode::serialize_into(&mut f, &canon).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    eprintln!(
        "gen-cut-db: wrote {} canonical entries (dense={} entries) to {} in {:?}",
        canon.entries.len(),
        canon.dense.len(),
        args.out.display(),
        start.elapsed()
    );

    Ok(())
}
