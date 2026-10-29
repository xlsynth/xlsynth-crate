module leaf(a, y);
  input [3:0] a;
  output y;
  INV gate0 (.A(a[2]), .Y(y));
endmodule
module top(a, y);
  input [1:0] a;
  output y;
  wire [3:0] bus;
  assign bus = {a, 2'b01};
  leaf inst (.a(bus), .y(y));
endmodule
