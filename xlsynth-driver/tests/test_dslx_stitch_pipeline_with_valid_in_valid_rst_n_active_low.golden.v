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
  input wire rst_n,
  input wire [31:0] x,
  input wire in_valid,
  output wire [31:0] out
);
  reg [31:0] p0_x;
  reg p0_valid;
  always @ (posedge clk) begin
    p0_x <= in_valid ? x : p0_x;
    p0_valid <= rst_n ? in_valid : 1'b0;
  end
  wire [31:0] stage_0_out_comb;
  foo_cycle0 stage_0 (
    .x(p0_x),
    .out(stage_0_out_comb)
  );
  reg [31:0] p1_y;
  reg p1_valid;
  always @ (posedge clk) begin
    p1_y <= p0_valid ? stage_0_out_comb : p1_y;
    p1_valid <= rst_n ? p0_valid : 1'b0;
  end
  wire [31:0] stage_1_out_comb;
  foo_cycle1 stage_1 (
    .y(p1_y),
    .out(stage_1_out_comb)
  );
  reg [31:0] p2_out;
  reg p2_valid;
  always @ (posedge clk) begin
    p2_out <= p1_valid ? stage_1_out_comb : p2_out;
    p2_valid <= rst_n ? p1_valid : 1'b0;
  end
  assign out = p2_out;
endmodule
