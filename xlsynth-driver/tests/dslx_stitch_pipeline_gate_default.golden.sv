module foo_cycle0(
  input wire [7:0] x,
  output wire [7:0] out
);
  wire [7:0] literal_16;
  wire [7:0] literal_17;
  wire ne_18;
  wire [7:0] add_19;
  wire [7:0] gate_22;
  assign literal_16 = 8'h00;
  assign literal_17 = 8'h01;
  assign ne_18 = x != literal_16;
  assign add_19 = x + literal_17;
  assign gate_22 = {8{ne_18}} & add_19;
  assign out = gate_22;
endmodule

module foo_cycle1(
  input wire [7:0] x,
  output wire [7:0] out
);
  wire [6:0] bit_slice_34;
  wire [6:0] literal_35;
  wire [7:0] literal_36;
  wire [6:0] add_37;
  wire bit_slice_38;
  wire ne_39;
  wire [7:0] concat_40;
  wire [7:0] gate_43;
  assign bit_slice_34 = x[7:1];
  assign literal_35 = 7'h01;
  assign literal_36 = 8'h00;
  assign add_37 = bit_slice_34 + literal_35;
  assign bit_slice_38 = x[0];
  assign ne_39 = x != literal_36;
  assign concat_40 = {add_37, bit_slice_38};
  assign gate_43 = {8{ne_39}} & concat_40;
  assign out = gate_43;
endmodule
module foo(
  input wire clk,
  input wire [7:0] x,
  output wire [7:0] out
);
  reg [7:0] p0_x;
  always @ (posedge clk) begin
    p0_x <= x;
  end
  wire [7:0] stage_0_out_comb;
  foo_cycle0 stage_0 (
    .x(p0_x),
    .out(stage_0_out_comb)
  );
  reg [7:0] p1_x;
  always @ (posedge clk) begin
    p1_x <= stage_0_out_comb;
  end
  wire [7:0] stage_1_out_comb;
  foo_cycle1 stage_1 (
    .x(p1_x),
    .out(stage_1_out_comb)
  );
  reg [7:0] p2_out;
  always @ (posedge clk) begin
    p2_out <= stage_1_out_comb;
  end
  assign out = p2_out;
endmodule
