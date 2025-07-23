module foo_cycle0(
  output wire [63:0] out
);
  wire [63:0] literal_8 = {32'h0000_002a, 32'h0000_0040};

  assign out = literal_8;
endmodule

module foo_cycle1(
  input wire [31:0] x,
  input wire [31:0] y,
  output wire [31:0] out
);
  wire [31:0] add_9;
  assign add_9 = x + y;
  assign out = add_9;
endmodule
module foo(
  input wire clk,
  output wire [31:0] out
);
  always_ff @ (posedge clk) begin end
  wire [63:0] p1_out_comb;
  wire [31:0] p1_x_comb;
  assign p1_x_comb = p1_out_comb[63:32];
  wire [31:0] p1_y_comb;
  assign p1_y_comb = p1_out_comb[31:0];
  foo_cycle0 stage_0 (
    .out(p1_out_comb)
  );
  reg [63:0] p1_out;
  reg [31:0] p1_x;
  reg [31:0] p1_y;
  always_ff @ (posedge clk) begin
    p1_out <= p1_out_comb;
    p1_x <= p1_x_comb;
    p1_y <= p1_y_comb;
  end
  wire [31:0] p2_out_comb;
  foo_cycle1 stage_1 (
    .x(p1_x),
    .y(p1_y),
    .out(p2_out_comb)
  );
  reg [31:0] p2_out;
  always_ff @ (posedge clk) begin
    p2_out <= p2_out_comb;
  end
  assign out = p2_out;
endmodule

