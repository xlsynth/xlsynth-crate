module add32(
  input wire [31:0] a,
  input wire [31:0] b,
  output wire [31:0] c
);
  assign c = a + b;
endmodule
module top(
  input wire clk,
  input wire [31:0] a,
  input wire [31:0] b,
  output wire [31:0] c
);
  reg [31:0] p0_a;
  reg [31:0] p0_b;
  always_ff @ (posedge clk) begin
    p0_a <= a;
    p0_b <= b;
  end
  wire [31:0] stage_0_out_comb;
  add32 stage_0 (
    .a(p0_a),
    .b(p0_b),
    .c(stage_0_out_comb)
  );
  reg [31:0] p1_c;
  always_ff @ (posedge clk) begin
    p1_c <= stage_0_out_comb;
  end
  assign c = p1_c;
endmodule
