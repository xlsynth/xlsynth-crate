module add_mul_cycle0(
  input wire [31:0] x,
  input wire [31:0] y,
  input wire [31:0] z,
  output wire [63:0] out
);
  wire [31:0] add_9;
  wire [63:0] tuple_10;
  assign add_9 = x + y;
  assign tuple_10 = {add_9, z};
  assign out = tuple_10;
endmodule

module add_mul_cycle1(
  input wire [31:0] sum,
  input wire [31:0] z,
  output wire [31:0] out
);
  // lint_off MULTIPLY
  function automatic [31:0] umul32b_32b_x_32b (input reg [31:0] lhs, input reg [31:0] rhs);
    begin
      umul32b_32b_x_32b = lhs * rhs;
    end
  endfunction
  // lint_on MULTIPLY
  wire [31:0] umul_11;
  assign umul_11 = umul32b_32b_x_32b(sum, z);
  assign out = umul_11;
endmodule
module add_mul(
  input wire clk,
  input wire [31:0] x,
  input wire [31:0] y,
  input wire [31:0] z,
  output wire [31:0] out
);
  reg [31:0] p0_x;
  reg [31:0] p0_y;
  reg [31:0] p0_z;
  always_ff @ (posedge clk) begin
    p0_x <= x;
    p0_y <= y;
    p0_z <= z;
  end
  wire [63:0] p1_next;
  add_mul_cycle0 add_mul_cycle0_i (
    .x(p0_x),
    .y(p0_y),
    .z(p0_z),
    .out(p1_next)
  );
  reg [63:0] p1;
  always_ff @ (posedge clk) begin
    p1 <= p1_next;
  end
  wire [31:0] p2_next;
  add_mul_cycle1 add_mul_cycle1_i (
    .sum(p1[63:32]),
    .z(p1[31:0]),
    .out(p2_next)
  );
  reg [31:0] p2;
  always_ff @ (posedge clk) begin
    p2 <= p2_next;
  end
  assign out = p2;
endmodule

