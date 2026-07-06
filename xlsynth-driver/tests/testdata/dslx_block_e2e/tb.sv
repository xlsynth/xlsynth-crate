// SPDX-License-Identifier: Apache-2.0

module tb;
  logic clk = 0;
  logic rst = 1;
  logic [7:0] x = 8'h10;
  logic ready = 0;
  wire [7:0] transformed;
  wire stream_valid;
  wire [7:0] stream_data;

  composed_top dut (
    .clk(clk), .rst(rst), .x(x), .ready(ready),
    .transformed(transformed), .stream_valid(stream_valid),
    .stream_data(stream_data)
  );

  always #5 clk = ~clk;

  initial begin
    repeat (2) @(posedge clk);
    #1;
    if (transformed !== 8'hb1)
      $fatal(1, "extern/custom result mismatch: %h", transformed);
    if (stream_valid !== 1'b1 || stream_data !== 8'h00)
      $fatal(1, "proc reset result mismatch: valid=%b data=%h", stream_valid, stream_data);
    @(negedge clk);
    rst = 0;
    ready = 1;
    x = 8'h20;
    @(posedge clk);
    #1;
    if (transformed !== 8'h81)
      $fatal(1, "updated extern/custom result mismatch: %h", transformed);
    if (stream_valid !== 1'b1 || stream_data !== 8'h01)
      $fatal(1, "proc cycle 1 mismatch: valid=%b data=%h", stream_valid, stream_data);
    @(posedge clk);
    #1;
    if (stream_data !== 8'h02)
      $fatal(1, "proc cycle 2 mismatch: %h", stream_data);
    $display("BLOCK_E2E_PASS");
    $finish;
  end
endmodule
