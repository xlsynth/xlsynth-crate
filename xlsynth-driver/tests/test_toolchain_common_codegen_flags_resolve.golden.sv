module __test__main(
  input wire clk,
  input wire pred,
  input wire x,
  output wire out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg pred__input_flop;
  reg x__input_flop;
  always_ff @ (posedge clk) begin
    pred__input_flop <= pred;
    x__input_flop <= x;
  end

  // ===== Pipe stage 1:
  wire p1_gated_comb;
  br_gate_buf gated_p1_gated_comb(.in(x__input_flop), .out(p1_gated_comb));

  // Registers for pipe stage 1:
  reg out__output_flop;
  always_ff @ (posedge clk) begin
    out__output_flop <= p1_gated_comb;
  end
  assign out = out__output_flop;
  `ifdef ASSERT_ON
  `BR_ASSERT(__test__main_0_non_synth___test__main_should_be_one, x__input_flop)
  `endif  // ASSERT_ON
endmodule

