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
  wire [63:0] p1_next;
  foo_cycle0 foo_cycle0_i (
    .s(p0_s),
    .out(p1_next)
  );
  reg [63:0] p1;
  always_ff @ (posedge clk) begin
    p1 <= p1_next;
  end
  wire [31:0] p2_next;
  foo_cycle1 foo_cycle1_i (
    .s(p1),
    .out(p2_next)
  );
  reg [31:0] p2;
  always_ff @ (posedge clk) begin
    p2 <= p2_next;
  end
  assign out = p2;
endmodule
