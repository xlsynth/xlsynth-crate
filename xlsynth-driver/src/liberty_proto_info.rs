// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use xlsynth_g8r::liberty::liberty_proto_info_from_path;

/// Loads a Liberty proto and prints deterministic structural metadata.
pub fn handle_liberty_proto_info(matches: &clap::ArgMatches) -> Result<(), String> {
    let path = matches
        .get_one::<String>("liberty_proto")
        .expect("liberty_proto is required by clap");
    let info = liberty_proto_info_from_path(Path::new(path)).map_err(|e| format!("{e:#}"))?;

    println!(
        "Size: {} bytes ({:.2} MiB)",
        info.size_bytes,
        info.size_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("SHA-256: {}", info.sha256);
    println!(
        "Provenance: {}",
        if info.provenance.is_empty() {
            "<unspecified>"
        } else {
            &info.provenance
        }
    );
    println!("Source files ({}):", info.source_files.len());
    for source_file in &info.source_files {
        println!("  - {source_file}");
    }
    println!("Cells: {}", info.cells);
    println!("  Combinational: {}", info.combinational_cells);
    println!("  Sequential: {}", info.sequential_cells);
    println!("  Native dont-use: {}", info.dont_use_cells);
    println!("LUT templates: {}", info.lut_templates);
    println!("Timing arcs: {}", info.timing_arcs);
    println!("Internal-power groups: {}", info.internal_power_groups);
    println!("Rise power tables: {}", info.rise_power_tables);
    println!("Fall power tables: {}", info.fall_power_tables);
    println!("Both-direction power tables: {}", info.both_power_tables);
    println!(
        "Unknown-direction power tables: {}",
        info.unknown_power_tables
    );
    match info.nominal_voltage {
        Some(voltage) => println!("Nominal voltage: {voltage} V"),
        None => println!("Nominal voltage: <unspecified>"),
    }
    Ok(())
}
