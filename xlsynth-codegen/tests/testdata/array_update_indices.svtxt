// SPDX-License-Identifier: Apache-2.0

module update_indices(
  input logic [6:0][7:0] values,
  input logic data_bit,
  input logic [7:0] replacement,
  output logic [6:0][7:0] result
);
  logic nor_4;
  logic [6:0][7:0] updated;
  assign nor_4 = ~(data_bit | data_bit | data_bit | data_bit);
  for (genvar __i0 = 0; __i0 < 7; __i0 = __i0 + 1) begin : gen__updated_0
    assign updated[__i0] = nor_4 == __i0 ? replacement : values[__i0];
  end
  assign result = updated;
endmodule
