// SPDX-License-Identifier: Apache-2.0
module tb;
  reg clk = 0, send = 0, capture = 0, reply = 0;
  reg [3:0] u_device_io = 4'b1010;
  wire [3:0] rx, rx_again, last_request;
  host dut(.*);

  task check(input [3:0] expected);
    #1;
    if (rx !== expected || rx_again !== expected || dut.u_device.io !== expected)
      $fatal(1, "expected %b on the private wire; saw %b, %b, %b",
             expected, rx, rx_again, dut.u_device.io);
  endtask

  task tick;
    #1; clk = 1; #1; clk = 0; #1;
  endtask

  initial begin
    check(4'bzzzz);
    send = 1; capture = 1;
    check(4'b1010);
    tick;
    if (last_request !== 4'b1010) $fatal(1, "lost request: %b", last_request);

    // The host releases the bus for one clock cycle before the device replies.
    send = 0; capture = 0;
    check(4'bzzzz);
    tick;
    check(4'bzzzz);

    reply = 1; check(4'b0101);
    send = 1; check(4'bxxxx);
    send = 0; check(4'b0101);
    reply = 0; check(4'bzzzz);
    $finish;
  end
endmodule
