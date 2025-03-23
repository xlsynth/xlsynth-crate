// SPDX-License-Identifier: Apache-2.0

//! Simple binary that attempts to parse a given liberty file.

use flate2::bufread::GzDecoder;
use std::io::BufReader;
use std::{fs::File, time::Instant};
use xlsynth_g8r::liberty::{CharReader, LibertyParser};

fn main() {
    let _ = env_logger::builder().try_init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <liberty_file>", args[0]);
        std::process::exit(1);
    }

    let path = args[1].clone();
    let file = File::open(path.clone()).unwrap();
    // Stat the file to figure out how many bytes it is.
    let metadata = file.metadata().unwrap();
    let size = metadata.len();
    println!("File size: {:.3} MiB", size as f64 / 1024.0 / 1024.0);

    let start = Instant::now();

    // The file extension may be either .lib or .lib.gz.
    let streamer: Box<dyn std::io::Read>;
    if path.ends_with(".gz") {
        let buf_reader = BufReader::with_capacity(256 * 1024, file);
        streamer = Box::new(GzDecoder::new(buf_reader));
    } else if path.ends_with(".lib") {
        streamer = Box::new(BufReader::new(file));
    } else {
        eprintln!("Unsupported file extension: {}", path);
        std::process::exit(1);
    }
    let char_reader = CharReader::new(streamer);
    let mut parser = LibertyParser::new_from_iter(char_reader);
    let _library = parser.parse().unwrap();
    let duration = start.elapsed();
    println!("Parsed in {:?}", duration);
    println!(
        "Implied: {:.3} MiB/s",
        size as f64 / duration.as_secs_f64() / 1024.0 / 1024.0
    );
    //println!("{:?}", library);
}
