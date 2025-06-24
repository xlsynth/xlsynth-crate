// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::process::Command;
use test_case::test_case;

#[test_case(1)]
#[test_case(2)]
#[test_case(3)]
fn test_simple_signature_compatibility(input_width: u32) {
    let mut temp_file = tempfile::Builder::new().suffix(".ir").tempfile().unwrap();
    let template = "package simple

top fn simple(input: bits[$INPUT_WIDTH] id=1) -> bits[$INPUT_WIDTH] {
    ret result: bits[$INPUT_WIDTH] = identity(input, id=2)
}\n";
    let contents = template.replace("$INPUT_WIDTH", &input_width.to_string());
    write!(temp_file, "{}", contents).unwrap();

    let temp_path = temp_file.into_temp_path();

    // Run the main binary using the temporary DSLX IR file as input
    let output = Command::new(env!("CARGO_BIN_EXE_g8r"))
        .arg(temp_path.to_str().unwrap())
        .env("RUST_LOG", std::env::var("RUST_LOG").unwrap_or_default())
        .output()
        .unwrap();

    // Output for debugging if needed
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    // Ensure the binary exited successfully and produced expected output
    assert!(
        output.status.success(),
        "g8r binary did not exit successfully"
    );
    assert!(
        stdout.contains("Deepest path"),
        "Output does not contain expected 'Deepest path' indication"
    );
}

#[test]
fn test_pack_unpack_signature_compatibility() {
    // Create a temporary DSLX IR file with a .x extension to use DSLX code syntax
    let mut temp_file = tempfile::Builder::new().suffix(".ir").tempfile().unwrap();
    let contents = "package pack_unpack

top fn pack_unpack(input: (bits[1], bits[8], bits[23]) id=1) -> (bits[1], bits[8], bits[23]) {
  a: bits[1] = tuple_index(input, index=0, id=2)
  b: bits[8] = tuple_index(input, index=1, id=3)
  c: bits[23] = tuple_index(input, index=2, id=4)
  packed: bits[32] = concat(a, b, c, id=5)
  a2: bits[1] = bit_slice(packed, start=31, width=1, id=8)
  b2: bits[8] = bit_slice(packed, start=23, width=8, id=11)
  c2: bits[23] = bit_slice(packed, start=0, width=23, id=14)
  ret result: (bits[1], bits[8], bits[23]) = tuple(a2, b2, c2, id=15)
}";
    write!(temp_file, "{}", contents).unwrap();

    let temp_path = temp_file.into_temp_path();

    // Run the main binary using the temporary DSLX IR file as input
    let output = Command::new(env!("CARGO_BIN_EXE_g8r"))
        .arg(temp_path.to_str().unwrap())
        .env("RUST_LOG", std::env::var("RUST_LOG").unwrap_or_default())
        .output()
        .unwrap();

    // Output for debugging if needed
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    // Ensure the binary exited successfully and produced expected output
    assert!(
        output.status.success(),
        "g8r binary did not exit successfully"
    );
    assert!(
        stdout.contains("Deepest path"),
        "Output does not contain expected 'Deepest path' indication"
    );
}

#[test]
fn test_deepest_path_source_display() {
    let ir = "package pos_pkg\nfile_number 0 \"foo.x\"\n\n\
top fn main() -> bits[32] {\n  ret literal.1: bits[32] = literal(value=1, id=1, pos=[(0,0,0)])\n}\n";
    let mut temp_file = tempfile::Builder::new().suffix(".ir").tempfile().unwrap();
    write!(temp_file, "{}", ir).unwrap();
    let temp_path = temp_file.into_temp_path();

    let output = Command::new(env!("CARGO_BIN_EXE_g8r"))
        .arg(temp_path.to_str().unwrap())
        .env("RUST_LOG", std::env::var("RUST_LOG").unwrap_or_default())
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("stdout: {}", stdout);

    assert!(
        output.status.success(),
        "g8r binary did not exit successfully"
    );
    assert!(stdout.contains("source:"), "Deepest path source not shown");
}
