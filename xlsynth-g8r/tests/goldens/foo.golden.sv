module foo_cycle0(
  input wire [63:0] s,
  output wire [63:0] out
);
  assign out = s;
endmodule

module foo_cycle1(
  input wire [63:0] s,
  output wire [31:0] out
);
  wire [31:0] s_a;
  wire [31:0] s_b;
  wire [31:0] add_9;
  assign s_a = s[63:32];
  assign s_b = s[31:0];
  assign add_9 = s_a + s_b;
  assign out = add_9;
endmodule
module foo(
  input wire clk,
  input wire [63:0] s,
  output wire [31:0] out
);
  reg [63:0] p0_s;
  always_ff @ (posedge clk) begin
    p0_s <= s;
  end
  wire [63:0] stage_0_out_comb;
  foo_cycle0 stage_0 (
    .s(p0_s),
    .out(stage_0_out_comb)
  );
  reg [63:0] p1_s;
  always_ff @ (posedge clk) begin
    p1_s <= stage_0_out_comb;
  end
  wire [31:0] stage_1_out_comb;
  foo_cycle1 stage_1 (
    .s(p1_s),
    .out(stage_1_out_comb)
  );
  reg [31:0] p2_out;
  always_ff @ (posedge clk) begin
    p2_out <= stage_1_out_comb;
  end
  assign out = p2_out;
endmodule
