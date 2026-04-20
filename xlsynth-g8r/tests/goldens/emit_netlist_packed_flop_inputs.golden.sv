module packed_flop_inputs(
  input wire clk,
  input wire [1:0] arg0,
  output wire [1:0] output_value
);
  reg [1:0] p0_arg0;
  wire G0;
  assign G0 = 1'b0;
  assign output_value[0] = ~p0_arg0[0];
  assign output_value[1] = ~p0_arg0[1];
  always_ff @ (posedge clk) begin
    p0_arg0 <= arg0;
  end
endmodule
