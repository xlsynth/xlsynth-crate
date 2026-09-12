module foo_cycle0(
  input wire [31:0] x,
  output wire [31:0] out
);
  wire [31:0] literal_12;
  wire [31:0] add_15;
  assign literal_12 = 32'h0000_0001;
  assign add_15 = x + literal_12;
  assign out = add_15;
endmodule

module foo_cycle1(
  input wire [31:0] y,
  output wire [31:0] out
);
  wire [30:0] bit_slice_24;
  wire [30:0] literal_25;
  wire [30:0] add_26;
  wire bit_slice_27;
  wire [31:0] concat_30;
  assign bit_slice_24 = y[31:1];
  assign literal_25 = 31'h0000_0001;
  assign add_26 = bit_slice_24 + literal_25;
  assign bit_slice_27 = y[0];
  assign concat_30 = {add_26, bit_slice_27};
  assign out = concat_30;
endmodule
module foo(
  input wire clk,
  input wire [31:0] x,
  input wire in_valid,
  output wire [31:0] out,
  output wire out_valid
);
  reg [31:0] p0_x;
  reg p0_valid;
  always @ (posedge clk) begin
    p0_x <= in_valid ? x : p0_x;
    p0_valid <= in_valid;
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
    p1_valid <= p0_valid;
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
    p2_valid <= p1_valid;
  end
  assign out = p2_out;
  assign out_valid = p2_valid;
endmodule

