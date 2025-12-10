// SPDX-License-Identifier: Apache-2.0

//! Helpers for generating large synthetic gate-level netlists for benchmarks.
//!
//! The generators in this module are intended for performance benchmarking of
//! the netlist parser and related tooling. They produce single-module
//! gate-level netlists with simple topologies (e.g. long chains of INVX1
//! instances) that stress tokenization and parsing without relying on any
//! particular Liberty library.

use std::fmt::Write as FmtWrite;

/// Generates a synthetic gate-level netlist containing a single module `top`
/// with a long chain of `INVX1` instances between scalar ports `a` and `y`.
///
/// The topology is:
///
/// ```text
///   a -> INVX1 u0 -> n0 -> INVX1 u1 -> n1 -> ... -> INVX1 u{N-1} -> y
/// ```
///
/// All intermediate nets `n<i>` are declared as scalar `wire`s. The generated
/// text is valid Verilog in the style used by existing netlist tests and is
/// accepted by the `netlist::parse` module.
pub fn make_chain_netlist(instance_count: usize) -> String {
    assert!(
        instance_count > 0,
        "instance_count must be greater than zero for synthetic chain netlist"
    );

    let mut s = String::new();

    // Module header and port declarations.
    writeln!(&mut s, "module top (a, y);").unwrap();
    writeln!(&mut s, "  input a;").unwrap();
    writeln!(&mut s, "  output y;").unwrap();

    // Scalar net declarations for ports and internal nets.
    writeln!(&mut s, "  wire a;").unwrap();
    writeln!(&mut s, "  wire y;").unwrap();
    for i in 0..instance_count.saturating_sub(1) {
        writeln!(&mut s, "  wire n{};", i).unwrap();
    }

    // Instance chain. Each INVX1 drives the next net in the chain.
    for i in 0..instance_count {
        let inst_name = format!("u{}", i);
        let input_net = if i == 0 {
            "a".to_string()
        } else {
            format!("n{}", i - 1)
        };
        let output_net = if i + 1 == instance_count {
            "y".to_string()
        } else {
            format!("n{}", i)
        };
        writeln!(
            &mut s,
            "  INVX1 {} (.A({}), .Y({}));",
            inst_name, input_net, output_net
        )
        .unwrap();
    }

    writeln!(&mut s, "endmodule").unwrap();

    s
}
