module __test__main(
  input wire clk,
  input wire pred,
  input wire x,
  output wire out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg p0_pred;
  reg p0_x;
  always_ff @ (posedge clk) begin
    p0_pred <= pred;
    p0_x <= x;
  end

  // ===== Pipe stage 1:
  wire p1_gated_comb;
  br_gate_buf gated_p1_gated_comb(.in(p0_x), .out(p1_gated_comb));

  // Registers for pipe stage 1:
  reg p1_gated;
  always_ff @ (posedge clk) begin
    p1_gated <= p1_gated_comb;
  end
  assign out = p1_gated;
  `ifdef ASSERT_ON
  `BR_ASSERT(should_be_one, p0_x)
  `endif  // ASSERT_ON
endmodule

