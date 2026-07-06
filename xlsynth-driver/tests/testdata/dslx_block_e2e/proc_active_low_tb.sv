// SPDX-License-Identifier: Apache-2.0

module proc_active_low_tb;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic ready = 1'b0;
  wire valid;
  wire [7:0] data;

  always #5 clk = ~clk;

  proc_active_low dut(
    .clk(clk),
    .rst_n(rst_n),
    .ready(ready),
    .valid(valid),
    .data(data)
  );

  initial begin
    repeat (2) @(posedge clk);
    #1;
    if (valid !== 1'b1 || data !== 8'h00)
      $fatal(1, "active-low reset mismatch: valid=%b data=%h", valid, data);

    rst_n = 1'b1;
    repeat (2) begin
      @(posedge clk);
      #1;
      if (valid !== 1'b1 || data !== 8'h00)
        $fatal(1, "stalled proc advanced: valid=%b data=%h", valid, data);
    end

    ready = 1'b1;
    @(posedge clk);
    #1;
    if (valid !== 1'b1 || data !== 8'h01)
      $fatal(1, "proc did not advance after handshake: valid=%b data=%h", valid, data);

    $display("BLOCK_PROC_ACTIVE_LOW_PASS");
    $finish;
  end
endmodule
