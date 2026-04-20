module packed_flop_outputs(
  input wire clk,
  input wire [1:0] arg0,
  output wire [1:0] output_value
);
  wire [1:0] output_value_comb;
  reg [1:0] p0_output_value;
  wire G0;
  assign G0 = 1'b0;
  assign output_value_comb[0] = ~arg0[0];
  assign output_value_comb[1] = ~arg0[1];
  always_ff @ (posedge clk) begin
    p0_output_value <= output_value_comb;
  end
  assign output_value = p0_output_value;
endmodule
