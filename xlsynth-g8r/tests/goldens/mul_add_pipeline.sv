module mul_add_cycle0(
  input wire clk,
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

  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [31:0] p0_x;
  reg [31:0] p0_y;
  reg [31:0] p0_z;
  always_ff @ (posedge clk) begin
    p0_x <= x;
    p0_y <= y;
    p0_z <= z;
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_umul_15_comb;
  wire [63:0] p1_tuple_16_comb;
  assign p1_umul_15_comb = umul32b_32b_x_32b(p0_x, p0_y);
  assign p1_tuple_16_comb = {p1_umul_15_comb, p0_z};
  assign out = p1_tuple_16_comb;
endmodule

module mul_add_cycle1(
  input wire clk,
  input wire [31:0] partial,
  input wire [31:0] z,
  output wire [31:0] out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [31:0] p0_partial;
  reg [31:0] p0_z;
  always_ff @ (posedge clk) begin
    p0_partial <= partial;
    p0_z <= z;
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_add_15_comb;
  assign p1_add_15_comb = p0_partial + p0_z;

  // Registers for pipe stage 1:
  reg [31:0] p1_add_15;
  always_ff @ (posedge clk) begin
    p1_add_15 <= p1_add_15_comb;
  end
  assign out = p1_add_15;
endmodule
module mul_add_pipeline(
  input wire [31:0] clk,
  input wire [31:0] x,
  input wire [31:0] y,
  output wire [31:0] out
);
  wire [31:0] stage0_out;
  wire [31:0] stage1_out;
  mul_add_cycle0 mul_add_cycle0_i (
    .clk(clk),
    .x(x),
    .y(y),
    .out(stage1_out)
  );
  wire [31:0] final_out;
  mul_add_cycle1 mul_add_cycle1_i (
    .clk(clk),
    .x(x),
    .y(y),
    .out(final_out)
  );
  assign out = final_out;
endmodule
