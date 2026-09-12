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
  wire p0_literal_30_comb;
  wire [2:0] p0_concat_32_comb;
  wire [2:0] p0_literal_33_comb;
  wire [2:0] p0_add_35_comb;
  wire [31:0] p0_array_index_38_comb;
  assign p0_literal_30_comb = 1'h0;
  assign p0_concat_32_comb = {p0_literal_30_comb, sel};
  assign p0_literal_33_comb = 3'h1;
  assign p0_add_35_comb = p0_concat_32_comb + p0_literal_33_comb;
  assign p0_array_index_38_comb = arr_unflattened[p0_add_35_comb > 3'h3 ? 2'h3 : p0_add_35_comb[1:0]];
  assign out = p0_array_index_38_comb;
endmodule

