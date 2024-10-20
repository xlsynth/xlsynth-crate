use std::process::Command;

#[test]
fn test_dslx2sv_types_subcommand() {
    let dslx = "enum OpType : u2 { READ = 0, WRITE = 1 }";
    // Make a temporary file to hold the DSLX code.
    let temp_dir = tempfile::tempdir().unwrap();
    let dslx_path = temp_dir.path().join("my_module.x");
    std::fs::write(&dslx_path, dslx).unwrap();
    // Run the dslx2sv subcommand.
    let command_path = env!("CARGO_BIN_EXE_xlsynth-driver");
    let output = Command::new(command_path)
        .arg("dslx2sv-types")
        .arg(dslx_path.to_str().unwrap())
        .output()
        .expect("Failed to run xlsynth-driver");

    assert!(output.status.success(), "stdout: {}\nstderr: {}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), r"typedef enum logic [1:0] {
    READ = 2'd0,
    WRITE = 2'd1
} op_type_e;");
}