module foo_cycle0(
  input wire [31:0] x,
  output wire [31:0] out
);
  wire [31:0] literal_5;
  wire [31:0] add_6;
  assign literal_5 = 32'h0000_0001;
  assign add_6 = x + literal_5;
  assign out = add_6;
endmodule

module foo_cycle1(
  input wire [31:0] y,
  output wire [31:0] out
);
  wire [30:0] bit_slice_17;
  wire [30:0] literal_18;
  wire [30:0] add_19;
  wire bit_slice_20;
  wire [31:0] concat_21;
  assign bit_slice_17 = y[31:1];
  assign literal_18 = 31'h0000_0001;
  assign add_19 = bit_slice_17 + literal_18;
  assign bit_slice_20 = y[0];
  assign concat_21 = {add_19, bit_slice_20};
  assign out = concat_21;
endmodule
module foo(
  input wire clk,
  input wire rst,
  input wire input_valid,
  input wire [31:0] x,
  output wire [31:0] out
);
  reg [31:0] p0_x;
  reg p0_valid;
  always_ff @ (posedge clk) begin
    p0_x <= input_valid ? x : p0_x;
    p0_valid <= rst ? input_valid : 1'b0;
  end
  wire [31:0] stage0_out_comb;
  foo_cycle0 foo_cycle0_i (
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
endmodule

