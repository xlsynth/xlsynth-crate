module my_main(
  input wire clk,
  input wire [31:0] x,
  output wire [31:0] out
);
  assign out = x;
endmodule