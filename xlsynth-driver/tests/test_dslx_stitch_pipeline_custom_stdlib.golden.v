module foo_cycle0(
  output wire [31:0] out
);
  wire [31:0] literal_7;
  assign literal_7 = 32'h0000_0007;
  assign out = literal_7;
endmodule

module foo_cycle1(
  input wire [31:0] x,
  output wire [31:0] out
);
  assign out = x;
endmodule
module foo(
  input wire clk,
  output wire [31:0] out
);
  wire [31:0] stage_0_out_comb;
  foo_cycle0 stage_0 (
    .out(stage_0_out_comb)
  );
  reg [31:0] p1_x;
  always @ (posedge clk) begin
    p1_x <= stage_0_out_comb;
  end
  wire [31:0] stage_1_out_comb;
  foo_cycle1 stage_1 (
    .x(p1_x),
    .out(stage_1_out_comb)
  );
  reg [31:0] p2_out;
  always @ (posedge clk) begin
    p2_out <= stage_1_out_comb;
  end
  assign out = p2_out;
endmodule
