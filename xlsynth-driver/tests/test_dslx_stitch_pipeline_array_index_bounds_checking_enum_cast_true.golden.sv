module foo_cycle0(
  input wire [1:0] sel,
  input wire [127:0] arr,
  output wire [31:0] out
);
  wire [31:0] arr_unflattened[4];
  assign arr_unflattened[0] = arr[31:0];
  assign arr_unflattened[1] = arr[63:32];
  assign arr_unflattened[2] = arr[95:64];
  assign arr_unflattened[3] = arr[127:96];
  wire literal_15;
  wire [2:0] concat_16;
  wire [2:0] literal_17;
  wire [2:0] add_18;
  wire [31:0] array_index_19;
  assign literal_15 = 1'h0;
  assign concat_16 = {literal_15, sel};
  assign literal_17 = 3'h1;
  assign add_18 = concat_16 + literal_17;
  assign array_index_19 = arr_unflattened[add_18 > 3'h3 ? 2'h3 : add_18[1:0]];
  assign out = array_index_19;
endmodule

module foo_cycle1(
  input wire [31:0] x,
  output wire [31:0] out
);
  assign out = x;
endmodule
module foo(
  input wire clk,
  input wire [1:0] sel,
  input wire [127:0] arr,
  output wire [31:0] out
);
  reg [127:0] p0_arr;
  reg [1:0] p0_sel;
  always @ (posedge clk) begin
    p0_arr <= arr;
    p0_sel <= sel;
  end
  wire [31:0] stage_0_out_comb;
  foo_cycle0 stage_0 (
    .sel(p0_sel),
    .arr(p0_arr),
    .out(stage_0_out_comb)
  );
  reg [31:0] p1_x;
  always @ (posedge clk) begin
    p1_x <= stage_0_out_comb;
  end
  wire [31:0] stage_1_out_comb;
  foo_cycle1 stage_1 (
    .x(p1_x),
    .out(stage_1_out_comb)
  );
  reg [31:0] p2_out;
  always @ (posedge clk) begin
    p2_out <= stage_1_out_comb;
  end
  assign out = p2_out;
endmodule

