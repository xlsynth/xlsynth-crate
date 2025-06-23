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
  always_ff @ (posedge clk) begin
    p0_y <= y;
    p0_x <= x;
  end
  wire [31:0] p1_next;
  one_cycle0 one_cycle0_i (
    .x(p0_x),
    .y(p0_y),
    .out(p1_next)
  );
  reg [31:0] p1;
  always_ff @ (posedge clk) begin
    p1 <= p1_next;
  end
  assign out = p1;
endmodule
