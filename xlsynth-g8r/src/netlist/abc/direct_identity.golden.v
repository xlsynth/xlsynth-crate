module identity(a, y, clk);
  input a;
  output y;
  input clk;
  wire __abc_n0;
  wire __abc_n1;
  assign __abc_n0 = a;
  assign y = __abc_n1;
  DFF __reg_0 (.CLK(clk), .D(__abc_n0), .Q(__abc_n1));
endmodule
