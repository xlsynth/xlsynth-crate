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
  always_ff @ (posedge clk) begin end
  wire [31:0] p1_next;
  foo_cycle0 foo_cycle0_i (
    .out(p1_next)
  );
  reg [31:0] p1;
  always_ff @ (posedge clk) begin
    p1 <= p1_next;
  end
  wire [31:0] p2_next;
  foo_cycle1 foo_cycle1_i (
    .x(p1),
    .out(p2_next)
  );
  reg [31:0] p2;
  always_ff @ (posedge clk) begin
    p2 <= p2_next;
  end
  assign out = p2;
endmodule

