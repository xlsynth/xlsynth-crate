// SPDX-License-Identifier: Apache-2.0

module nested_update(
  input logic [6:0][2:0] values,
  input logic row_source,
  input logic replacement,
  output logic [6:0][2:0] result
);
  logic not_4;
  logic [1:0] literal_5;
  logic [6:0][2:0] updated;
  assign not_4 = ~row_source;
  assign literal_5 = 2'h2;
  for (genvar __i0 = 0; __i0 < 7; __i0 = __i0 + 1) begin : gen__updated_0
    for (genvar __i1 = 0; __i1 < 3; __i1 = __i1 + 1) begin : gen__updated_1
      assign updated[__i0][__i1] = not_4 == __i0 && literal_5 == __i1 ? replacement : values[__i0][__i1];
    end
  end
  assign result = updated;
endmodule
