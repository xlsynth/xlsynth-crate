module add32(
  input wire [31:0] a,
  input wire [31:0] b,
  output wire [31:0] c
);
  assign c = a + b;
endmodule
module top(
  input wire clk,
  input wire rst,
  input wire [31:0] a,
  input wire [31:0] b,
  input wire in_valid,
  output wire [31:0] c,
  output wire out_valid
);
  reg [31:0] p0_a;
  reg [31:0] p0_b;
  reg p0_valid;
  always @ (posedge clk) begin
    p0_a <= in_valid ? a : p0_a;
    p0_b <= in_valid ? b : p0_b;
    p0_valid <= rst ? 1'b0 : in_valid;
  end
  wire [31:0] stage_0_out_comb;
  add32 stage_0 (
    .a(p0_a),
    .b(p0_b),
    .c(stage_0_out_comb)
  );
  reg [31:0] p1_c;
  reg p1_valid;
  always @ (posedge clk) begin
    p1_c <= p0_valid ? stage_0_out_comb : p1_c;
    p1_valid <= rst ? 1'b0 : p0_valid;
  end
  assign c = p1_c;
  assign out_valid = p1_valid;
endmodule
