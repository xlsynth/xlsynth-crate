// SPDX-License-Identifier: Apache-2.0

//! Generates the 4-input cut database (canonical NPN entries) and writes it to disk.

use std::io;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use xlsynth_g8r::cut_db::enumerate::{enumerate_full_space, EnumerateOptions};
use xlsynth_g8r::cut_db::serdes::canon_from_full_space;

#[derive(Debug, Parser)]
#[command(name = "gen-cut-db")]
#[command(about = "Generate 4-input AIG cut DB (NPN canonical, Pareto frontier)")]
struct Args {
    /// Output path for the bincode DB artifact.
    #[arg(long)]
    out: PathBuf,

    /// Optional maximum AND-count to explore (for debugging).
    #[arg(long)]
    max_ands: Option<u16>,

    /// Emit progress every N worklist pops during enumeration.
    #[arg(long, default_value_t = 100_000)]
    progress_every_pops: u64,
}

fn main() -> io::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let start = Instant::now();
    let full = enumerate_full_space(EnumerateOptions {
        max_ands: args.max_ands,
        progress_every_pops: Some(args.progress_every_pops),
    });
    let covered = full.covered_count();

    let canon = canon_from_full_space(&full);

    let mut f = std::fs::File::create(&args.out)?;
    bincode::serialize_into(&mut f, &canon)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    eprintln!(
        "gen-cut-db: wrote {} canonical entries (full covered: {}/65536) to {} in {:?}",
        canon.entries.len(),
        covered,
        args.out.display(),
        start.elapsed()
    );

    Ok(())
}


