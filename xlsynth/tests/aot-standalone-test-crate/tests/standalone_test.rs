// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_aot_standalone_test_crate::add_one_aot;

// Verifies: standalone generated artifacts execute without any runtime
// dependency on the xlsynth crate.
// Catches: generated code paths that still require descriptor-based runtime
// APIs.
#[test]
fn standalone_generated_runner_executes() {
    let mut runner = add_one_aot::new_runner().expect("runner creation should succeed");
    let output = runner.run(&41).expect("run should succeed");
    assert_eq!(output, 42);
}

// Verifies: the minimal standalone runtime consumer binary has no dynamic
// libxls dependency.
// Catches: accidental reintroduction of runtime xlsynth/libxls linkage.
#[test]
fn standalone_consumer_binary_has_no_runtime_libxls_dependency() {
    let executable = std::env::current_exe().expect("current test executable should resolve");
    let output = if cfg!(target_os = "macos") {
        Command::new("otool")
            .arg("-L")
            .arg(executable)
            .output()
            .expect("otool should inspect the test executable")
    } else if cfg!(target_os = "linux") {
        Command::new("ldd")
            .arg(executable)
            .output()
            .expect("ldd should inspect the test executable")
    } else {
        return;
    };
    let dependencies = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "dependency inspection should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !dependencies.contains("libxls"),
        "standalone runtime consumer unexpectedly depends on libxls:\n{}",
        dependencies
    );
}
