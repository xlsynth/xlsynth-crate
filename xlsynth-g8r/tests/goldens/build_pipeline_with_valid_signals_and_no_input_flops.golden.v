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
  wire [31:0] p0_c_comb;
  add32 stage_0 (
    .a(a),
    .b(b),
    .c(p0_c_comb)
  );
  reg [31:0] p0_c;
  reg p0_valid;
  always_ff @ (posedge clk) begin
    p0_c <= in_valid ? p0_c_comb : p0_c;
    p0_valid <= rst ? 1'b0 : in_valid;
  end
  assign c = p0_c;
  assign out_valid = p0_valid;
endmodule
