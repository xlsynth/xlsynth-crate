// SPDX-License-Identifier: Apache-2.0
module tb;
  reg [3:0] left_data = 4'b1010, right_data = 4'b1001, ext = 4'bzzzz;
  reg left_oe = 0, right_oe = 0;
  wire [3:0] pad, seen;
  assign pad = ext;
  pads dut(.*);
  task check(input [3:0] expected);
    #1;
    if (pad !== expected || seen !== expected ||
        dut.left.observed !== expected || dut.right.observed !== expected)
      $fatal(1, "expected %b; pin=%b sense=%b left=%b right=%b",
             expected, pad, seen, dut.left.observed, dut.right.observed);
  endtask
  initial begin
    check(4'bzzzz);
    left_oe = 1; check(4'b1010);
    right_oe = 1; check(4'b10xx);
    right_data = left_data; check(4'b1010);
    left_oe = 0; right_data = 4'b1001; check(4'b1001);
    ext = 4'b1010; check(4'b10xx);
    right_oe = 0; check(4'b1010);
    ext = 4'bzzzz; check(4'bzzzz);
    $finish;
  end
endmodule
