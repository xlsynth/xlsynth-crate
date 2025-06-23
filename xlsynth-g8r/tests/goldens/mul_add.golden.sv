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
  always_ff @ (posedge clk) begin
    p0_x <= x;
    p0_z <= z;
    p0_y <= y;
  end
  wire [63:0] p1_next;
  mul_add_cycle0 mul_add_cycle0_i (
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
  mul_add_cycle1 mul_add_cycle1_i (
    .partial(p1[63:32]),
    .z(p1[31:0]),
    .out(p2_next)
  );
  reg [31:0] p2;
  always_ff @ (posedge clk) begin
    p2 <= p2_next;
  end
  assign out = p2;
endmodule
