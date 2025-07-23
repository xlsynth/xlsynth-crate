module one_cycle0(
  input wire [31:0] x,
  input wire [31:0] y,
  output wire [31:0] out
);
  wire [31:0] add_6;
  assign add_6 = x + y;
  assign out = add_6;
endmodule
module one(
  input wire clk,
  input wire [31:0] x,
  input wire [31:0] y,
  output wire [31:0] out
);
  reg [31:0] p0_x;
  reg [31:0] p0_y;
  always @ (posedge clk) begin
    p0_x <= x;
    p0_y <= y;
  end
  wire [31:0] stage_0_out_comb;
  one_cycle0 stage_0 (
    .x(p0_x),
    .y(p0_y),
    .out(stage_0_out_comb)
  );
  reg [31:0] p1_out;
  always @ (posedge clk) begin
    p1_out <= stage_0_out_comb;
  end
  assign out = p1_out;
endmodule
