module __my_module__main(
  input wire clk,
  input wire [131:0] x,
  output wire [131:0] out,
  output wire idle
);
  wire [32:0] literal_22 = {1'h0, 32'h0000_002a};
  wire [32:0] x_unflattened[4];
  assign x_unflattened[0] = x[32:0];
  assign x_unflattened[1] = x[65:33];
  assign x_unflattened[2] = x[98:66];
  assign x_unflattened[3] = x[131:99];

  // ===== Pipe stage 0:
  wire p0_literal_28_comb;
  assign p0_literal_28_comb = 1'h0;

  // Registers for pipe stage 0:
  reg [32:0] x__input_flop[4];
  always_ff @ (posedge clk) begin
    x__input_flop <= x_unflattened;
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_literal_23_comb;
  wire [32:0] p1_array_update_27_comb[4];
  assign p1_literal_23_comb = 32'h0000_0001;
  assign out = {p1_array_update_27_comb[3], p1_array_update_27_comb[2], p1_array_update_27_comb[1], p1_array_update_27_comb[0]};
  assign idle = p0_literal_28_comb;
  for (genvar __i0 = 0; __i0 < 4; __i0 = __i0 + 1) begin : gen__array_update_27_0
    assign p1_array_update_27_comb[__i0] = p1_literal_23_comb == __i0 ? literal_22 : x__input_flop[__i0];
  end
endmodule

