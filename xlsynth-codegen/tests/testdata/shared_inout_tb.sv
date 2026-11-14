// SPDX-License-Identifier: Apache-2.0
module tb;
  reg [3:0] left_data = 4'b1010, right_data = 4'b1001, ext = 4'bzzzz, ext_b = 4'bzzzz;
  reg left_oe = 0, right_oe = 0;
  wire [3:0] pad, pad_b, spare, seen;
  reg [3:0] external_spare = 4'bzzzz;
  assign spare = external_spare;
  assign pad = ext;
  assign pad_b = ext_b;
  pads dut(.*);
  task check(input [3:0] expected);
    #1;
    if (spare !== external_spare) $fatal(1, "open pin: expected %b, saw %b", external_spare, spare);
    if (pad !== expected || pad_b !== expected || seen !== expected ||
        dut.left.observed !== expected || dut.right.observed !== expected)
      $fatal(1, "expected %b; pad=%b pad_b=%b sense=%b left=%b right=%b",
             expected, pad, pad_b, seen, dut.left.observed, dut.right.observed);
  endtask
  initial begin
    check(4'bzzzz);
    left_oe = 1; check(4'b1010);
    external_spare = 4'b0101;
    right_oe = 1; check(4'b10xx);
    right_data = left_data; check(4'b1010);
    left_oe = 0; right_data = 4'b1001; check(4'b1001);
    ext = 4'b1010; check(4'b10xx);
    right_oe = 0; check(4'b1010);
    ext_b = 4'b1001; check(4'b10xx);
    ext = 4'bzzzz; check(4'b1001);
    external_spare = 4'bzzzz;
    ext_b = 4'bzzzz; check(4'bzzzz);
    $finish;
  end
endmodule
