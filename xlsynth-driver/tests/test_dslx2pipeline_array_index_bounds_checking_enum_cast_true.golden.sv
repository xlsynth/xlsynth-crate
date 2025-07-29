module __enum_index__main(
  input wire clk,
  input wire [1:0] sel,
  input wire [127:0] arr,
  output wire [31:0] out
);
  wire [31:0] arr_unflattened[4];
  assign arr_unflattened[0] = arr[31:0];
  assign arr_unflattened[1] = arr[63:32];
  assign arr_unflattened[2] = arr[95:64];
  assign arr_unflattened[3] = arr[127:96];

  // ===== Pipe stage 0:
  wire p0_literal_26_comb;
  wire [2:0] p0_concat_29_comb;
  wire [2:0] p0_literal_27_comb;
  wire [2:0] p0_add_31_comb;
  wire [31:0] p0_array_index_32_comb;
  assign p0_literal_26_comb = 1'h0;
  assign p0_concat_29_comb = {p0_literal_26_comb, sel};
  assign p0_literal_27_comb = 3'h1;
  assign p0_add_31_comb = p0_concat_29_comb + p0_literal_27_comb;
  assign p0_array_index_32_comb = arr_unflattened[p0_add_31_comb > 3'h3 ? 2'h3 : p0_add_31_comb[1:0]];
  assign out = p0_array_index_32_comb;
endmodule

