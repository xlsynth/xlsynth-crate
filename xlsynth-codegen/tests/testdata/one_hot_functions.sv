// SPDX-License-Identifier: Apache-2.0

module hot_helpers(
  input logic [3:0] value,
  input logic [3:0] one_hot_lsb_4,
  output logic [4:0] first,
  output logic [4:0] second,
  output logic [4:0] high
);
  function automatic logic [4:0] one_hot_lsb_4__1 (input logic [3:0] value);
    unique casez (value)
      4'b???1: one_hot_lsb_4__1 = 5'h01;
      4'b??10: one_hot_lsb_4__1 = 5'h02;
      4'b?100: one_hot_lsb_4__1 = 5'h04;
      4'b1000: one_hot_lsb_4__1 = 5'h08;
      4'b0000: one_hot_lsb_4__1 = 5'h10;
      default: one_hot_lsb_4__1 = 'X;
    endcase
  endfunction
  function automatic logic [4:0] one_hot_msb_4 (input logic [3:0] value);
    unique casez (value)
      4'b1???: one_hot_msb_4 = 5'h08;
      4'b01??: one_hot_msb_4 = 5'h04;
      4'b001?: one_hot_msb_4 = 5'h02;
      4'b0001: one_hot_msb_4 = 5'h01;
      4'b0000: one_hot_msb_4 = 5'h10;
      default: one_hot_msb_4 = 'X;
    endcase
  endfunction
  logic [4:0] a;
  logic [4:0] b;
  logic [4:0] h;
  assign a = one_hot_lsb_4__1(value);
  assign b = one_hot_lsb_4__1(one_hot_lsb_4);
  assign h = one_hot_msb_4(value);
  assign first = a;
  assign second = b;
  assign high = h;
endmodule
