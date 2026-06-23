// SPDX-License-Identifier: Apache-2.0
use std::io::Write;

use xlsynth_vastly::compile_pipeline_module;
use xlsynth_vastly::cycles_from_irvals_file;

#[test]
fn parses_irvals_tuple_and_maps_positionally_to_inputs() {
    let dut = r#"
module m(
  input logic clk,
  input logic [3:0] a,
  input logic [2:0] b,
  output wire [3:0] out
);
  assign out = a;
endmodule
"#;
    let m = compile_pipeline_module(dut).unwrap();

    let td = mk_temp_dir();
    let p = td.join("in.irvals");
    {
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "(bits[4]:0, bits[3]:7)").unwrap();
        writeln!(f, "(bits[4]:15, bits[3]:0)").unwrap();
    }

    let cycles = cycles_from_irvals_file(&m, &p, None).unwrap();
    assert_eq!(cycles.len(), 2);

    assert_eq!(
        cycles[0].inputs.get("a").unwrap().to_bit_string_msb_first(),
        "0000"
    );
    assert_eq!(
        cycles[0].inputs.get("b").unwrap().to_bit_string_msb_first(),
        "111"
    );
    assert_eq!(
        cycles[1].inputs.get("a").unwrap().to_bit_string_msb_first(),
        "1111"
    );
    assert_eq!(
        cycles[1].inputs.get("b").unwrap().to_bit_string_msb_first(),
        "000"
    );
}

#[test]
fn irvals_cycles_flag_truncates_to_first_n_vectors() {
    let dut = r#"
module m(
  input logic clk,
  input logic [3:0] a,
  input logic [2:0] b,
  output wire [3:0] out
);
  assign out = a;
endmodule
"#;
    let m = compile_pipeline_module(dut).unwrap();

    let td = mk_temp_dir();
    let p = td.join("in.irvals");
    {
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "(bits[4]:0, bits[3]:0)").unwrap();
        writeln!(f, "(bits[4]:1, bits[3]:1)").unwrap();
        writeln!(f, "(bits[4]:2, bits[3]:2)").unwrap();
    }

    let cycles = cycles_from_irvals_file(&m, &p, Some(2)).unwrap();
    assert_eq!(cycles.len(), 2);
    assert_eq!(
        cycles[1].inputs.get("a").unwrap().to_bit_string_msb_first(),
        "0001"
    );
    assert_eq!(
        cycles[1].inputs.get("b").unwrap().to_bit_string_msb_first(),
        "001"
    );
}

#[test]
fn parses_named_irvals_by_exact_input_name() {
    let dut = r#"
module m(
  input logic clk,
  input logic [3:0] a,
  input logic [2:0] b,
  output wire [3:0] out
);
  assign out = a;
endmodule
"#;
    let m = compile_pipeline_module(dut).unwrap();
    let td = mk_temp_dir();
    let p = td.join("named.irvals");
    std::fs::write(
        &p,
        "{b: bits[3]:7, a: bits[4]:0}\n{a: bits[4]:15, b: bits[3]:0}\n",
    )
    .unwrap();

    let cycles = cycles_from_irvals_file(&m, &p, None).unwrap();
    assert_eq!(cycles.len(), 2);
    assert_eq!(
        cycles[0].inputs.get("a").unwrap().to_bit_string_msb_first(),
        "0000"
    );
    assert_eq!(
        cycles[0].inputs.get("b").unwrap().to_bit_string_msb_first(),
        "111"
    );

    std::fs::write(
        &p,
        "# Keep original line numbers across comments and blank lines.\n\n{a: bits[4]:0, wrong: bits[3]:0}\n",
    )
    .unwrap();
    let error = cycles_from_irvals_file(&m, &p, None).unwrap_err();
    let message = format!("{error:?}");
    assert!(message.contains("line 3"), "{message}");
    assert!(message.contains("missing [\\\"b\\\"]"), "{message}");
    assert!(message.contains("unknown [\\\"wrong\\\"]"), "{message}");
}

fn mk_temp_dir() -> std::path::PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0u32..1000u32 {
        let p = base.join(format!("vastly_irvals_{pid}_{nanos}_{attempt}"));
        match std::fs::create_dir(&p) {
            Ok(()) => return p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("create temp dir: {e:?}"),
        }
    }
    panic!("failed to create temp dir");
}
