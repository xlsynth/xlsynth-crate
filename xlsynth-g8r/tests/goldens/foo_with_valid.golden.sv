module foo_cycle0(
  input wire [31:0] x,
  output wire [31:0] out
);
  wire [31:0] literal_5;
  wire [31:0] add_6;
  assign literal_5 = 32'h0000_0001;
  assign add_6 = x + literal_5;
  assign out = add_6;
endmodule

module foo_cycle1(
  input wire [31:0] y,
  output wire [31:0] out
);
  wire [30:0] bit_slice_17;
  wire [30:0] literal_18;
  wire [30:0] add_19;
  wire bit_slice_20;
  wire [31:0] concat_21;
  assign bit_slice_17 = y[31:1];
  assign literal_18 = 31'h0000_0001;
  assign add_19 = bit_slice_17 + literal_18;
  assign bit_slice_20 = y[0];
  assign concat_21 = {add_19, bit_slice_20};
  assign out = concat_21;
endmodule
module foo(
  input wire clk,
  input wire rst,
  input wire [31:0] x,
  input wire input_valid,
  output wire [31:0] out,
  output wire output_valid
);
  wire [31:0] stage_0_out_comb;
  foo_cycle0 stage_0 (
    .x(x),
    .out(stage_0_out_comb)
  );
  reg [31:0] p0_y;
  reg p0_valid;
  always @ (posedge clk) begin
    p0_y <= input_valid ? stage_0_out_comb : p0_y;
    p0_valid <= rst ? input_valid : 1'b0;
  end
  wire [31:0] stage_1_out_comb;
  foo_cycle1 stage_1 (
    .y(p0_y),
    .out(stage_1_out_comb)
  );
  assign out = stage_1_out_comb;
  assign output_valid = p0_valid;
endmodule
