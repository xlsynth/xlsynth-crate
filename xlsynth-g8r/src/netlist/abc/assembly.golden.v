module top(__abc_n0, data, \wire , clk);
  input [1:0] __abc_n0;
  output [1:0] data;
  output [1:0] \wire ;
  input clk;
  wire __abc_n0_1;
  wire __abc_n1;
  wire __abc_n2;
  wire __abc_n3;
  assign __abc_n0_1 = __abc_n0[0];
  assign __abc_n1 = __abc_n0[1];
  assign data[0] = __abc_n0_1;
  assign data[1] = __abc_n1;
  assign \wire [0] = __abc_n2;
  assign \wire [1] = __abc_n3;
  DFF __reg_0 (.CLK(clk), .D(__abc_n0_1), .Q(__abc_n2));
  DFF __reg_1 (.CLK(clk), .D(__abc_n1), .Q(__abc_n3));
endmodule
