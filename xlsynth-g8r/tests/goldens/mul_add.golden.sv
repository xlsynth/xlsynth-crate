module mul_add_cycle0(
  input wire [31:0] x,
  input wire [31:0] y,
  input wire [31:0] z,
  output wire [63:0] out
);
  // lint_off MULTIPLY
  function automatic [31:0] umul32b_32b_x_32b (input reg [31:0] lhs, input reg [31:0] rhs);
    begin
      umul32b_32b_x_32b = lhs * rhs;
    end
  endfunction
  // lint_on MULTIPLY
  wire [31:0] umul_9;
  wire [63:0] tuple_10;
  assign umul_9 = umul32b_32b_x_32b(x, y);
  assign tuple_10 = {umul_9, z};
  assign out = tuple_10;
endmodule

module mul_add_cycle1(
  input wire [31:0] partial,
  input wire [31:0] z,
  output wire [31:0] out
);
  wire [31:0] add_11;
  assign add_11 = partial + z;
  assign out = add_11;
endmodule
module mul_add(
  input wire clk,
  input wire [31:0] x,
  input wire [31:0] y,
  input wire [31:0] z,
  output wire [31:0] out
);
  reg [31:0] p0_x;
  reg [31:0] p0_y;
  reg [31:0] p0_z;
  always @ (posedge clk) begin
    p0_x <= x;
    p0_y <= y;
    p0_z <= z;
  end
  wire [63:0] stage_0_out_comb;
  mul_add_cycle0 stage_0 (
    .x(p0_x),
    .y(p0_y),
    .z(p0_z),
    .out(stage_0_out_comb)
  );
  wire [31:0] p1_partial_comb;
  assign p1_partial_comb = stage_0_out_comb[31:0];
  wire [31:0] p1_z_comb;
  assign p1_z_comb = stage_0_out_comb[63:32];
  reg [31:0] p1_partial;
  reg [31:0] p1_z;
  always @ (posedge clk) begin
    p1_partial <= p1_partial_comb;
    p1_z <= p1_z_comb;
  end
  wire [31:0] stage_1_out_comb;
  mul_add_cycle1 stage_1 (
    .partial(p1_partial),
    .z(p1_z),
    .out(stage_1_out_comb)
  );
  reg [31:0] p2_out;
  always @ (posedge clk) begin
    p2_out <= stage_1_out_comb;
  end
  assign out = p2_out;
endmodule
