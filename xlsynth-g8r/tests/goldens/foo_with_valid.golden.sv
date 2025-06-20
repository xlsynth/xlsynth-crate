module foo_cycle0(
  input wire clk,
  input wire [31:0] x,
  output wire [31:0] out
);
  // ===== Pipe stage 0:
  wire [31:0] p0_literal_5_comb;
  wire [31:0] p0_add_6_comb;
  assign p0_literal_5_comb = 32'h0000_0001;
  assign p0_add_6_comb = x + p0_literal_5_comb;
  assign out = p0_add_6_comb;
endmodule

module foo_cycle1(
  input wire clk,
  input wire [31:0] y,
  output wire [31:0] out
);
  // ===== Pipe stage 0:
  wire [30:0] p0_bit_slice_17_comb;
  wire [30:0] p0_literal_18_comb;
  wire [30:0] p0_add_19_comb;
  wire p0_bit_slice_20_comb;
  wire [31:0] p0_concat_21_comb;
  assign p0_bit_slice_17_comb = y[31:1];
  assign p0_literal_18_comb = 31'h0000_0001;
  assign p0_add_19_comb = p0_bit_slice_17_comb + p0_literal_18_comb;
  assign p0_bit_slice_20_comb = y[0];
  assign p0_concat_21_comb = {p0_add_19_comb, p0_bit_slice_20_comb};
  assign out = p0_concat_21_comb;
endmodule
module foo(
  input wire clk,
  input wire rst,
  input wire input_valid,
  input wire [31:0] x,
  output wire [31:0] out,
  output wire output_valid
);
  reg [31:0] p0_x;
  reg p0_valid;
  always_ff @ (posedge clk) begin
    p0_x <= x;
    p0_valid <= rst ? input_valid : 1'b0;
  end
  wire [31:0] stage0_out_comb;
  foo_cycle0 foo_cycle0_i (
    .clk(clk),
    .x(p0_x),
    .out(stage0_out_comb)
  );
  reg [31:0] p1_out;
  reg p1_valid;
  always_ff @ (posedge clk) begin
    p1_out <= stage0_out_comb;
    p1_valid <= rst ? p0_valid : 1'b0;
  end
  wire [31:0] stage1_out_comb;
  foo_cycle1 foo_cycle1_i (
    .clk(clk),
    .y(p1_out),
    .out(stage1_out_comb)
  );
  reg [31:0] p2_out;
  reg p2_valid;
  always_ff @ (posedge clk) begin
    p2_out <= stage1_out_comb;
    p2_valid <= rst ? p1_valid : 1'b0;
  end
  assign out = p2_out;
  assign output_valid = p2_valid;
endmodule
