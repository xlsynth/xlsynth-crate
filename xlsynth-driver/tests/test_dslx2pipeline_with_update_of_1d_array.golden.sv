module __my_module__main(
  input wire clk,
  input wire [131:0] x,
  output wire [131:0] out
);
  wire [32:0] literal_11 = {1'h0, 32'h0000_002a};
  wire [32:0] x_unflattened[4];
  assign x_unflattened[0] = x[32:0];
  assign x_unflattened[1] = x[65:33];
  assign x_unflattened[2] = x[98:66];
  assign x_unflattened[3] = x[131:99];

  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [32:0] p0_x[4];
  always_ff @ (posedge clk) begin
    p0_x <= x_unflattened;
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_literal_12_comb;
  wire [32:0] p1_array_update_16_comb[4];
  assign p1_literal_12_comb = 32'h0000_0001;
  assign p1_array_update_16_comb[0] = p1_literal_12_comb == 32'h0000_0000 ? literal_11 : p0_x[0];
  assign p1_array_update_16_comb[1] = p1_literal_12_comb == 32'h0000_0001 ? literal_11 : p0_x[1];
  assign p1_array_update_16_comb[2] = p1_literal_12_comb == 32'h0000_0002 ? literal_11 : p0_x[2];
  assign p1_array_update_16_comb[3] = p1_literal_12_comb == 32'h0000_0003 ? literal_11 : p0_x[3];
  assign out = {p1_array_update_16_comb[3], p1_array_update_16_comb[2], p1_array_update_16_comb[1], p1_array_update_16_comb[0]};
endmodule

