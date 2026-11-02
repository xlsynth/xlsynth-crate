module pipe_cycle0(
  input wire mul_op,
  input wire [3:0] x,
  input wire [3:0] y,
  output wire [10:0] out
);
  // lint_off MULTIPLY
  function automatic [3:0] umul4b_4b_x_4b (input reg [3:0] lhs, input reg [3:0] rhs);
    begin
      umul4b_4b_x_4b = lhs * rhs;
    end
  endfunction
  // lint_on MULTIPLY
  wire not_23;
  wire [3:0] add_24;
  wire [3:0] umul_25;
  wire [4:0] tuple_26;
  wire [4:0] tuple_27;
  wire [10:0] tuple_30;
  assign not_23 = ~mul_op;
  assign add_24 = x + y;
  assign umul_25 = umul4b_4b_x_4b(x, y);
  assign tuple_26 = {not_23, add_24};
  assign tuple_27 = {mul_op, umul_25};
  assign tuple_30 = {mul_op, tuple_26, tuple_27};
  assign out = tuple_30;
endmodule

module pipe_cycle1(
  input wire [10:0] a,
  output wire [10:0] out
);
  assign out = a;
endmodule

module pipe_cycle2(
  input wire [10:0] a,
  output wire [3:0] out
);
  wire [4:0] a_add;
  wire [4:0] a_mul;
  wire a_mul_op;
  wire [3:0] a_add_data;
  wire [3:0] a_mul_data;
  wire [3:0] sel_36;
  assign a_add = a[9:5];
  assign a_mul = a[4:0];
  assign a_mul_op = a[10:10];
  assign a_add_data = a_add[3:0];
  assign a_mul_data = a_mul[3:0];
  assign sel_36 = a_mul_op ? a_mul_data : a_add_data;
  assign out = sel_36;
endmodule
module pipe(
  input wire clk,
  input wire rst,
  input wire mul_op,
  input wire [3:0] x,
  input wire [3:0] y,
  input wire in_valid,
  output wire [3:0] out,
  output wire out_valid
);
  wire [10:0] stage_0_out_comb;
  pipe_cycle0 stage_0 (
    .mul_op(mul_op),
    .x(x),
    .y(y),
    .out(stage_0_out_comb)
  );
  reg p0_bank_at_10;
  reg p0_bank_at_9;
  reg [3:0] p0_bank_at_5;
  reg p0_bank_at_4;
  reg [3:0] p0_bank_at_0;
  always @ (posedge clk) begin
    p0_bank_at_10 <= in_valid ? stage_0_out_comb[10:10] : p0_bank_at_10;
    p0_bank_at_9 <= in_valid ? stage_0_out_comb[9:9] : p0_bank_at_9;
    p0_bank_at_5 <= in_valid & stage_0_out_comb[9] ? stage_0_out_comb[8:5] : p0_bank_at_5;
    p0_bank_at_4 <= in_valid ? stage_0_out_comb[4:4] : p0_bank_at_4;
    p0_bank_at_0 <= in_valid & stage_0_out_comb[4] ? stage_0_out_comb[3:0] : p0_bank_at_0;
  end
  wire [10:0] p0_bank_result;
  assign p0_bank_result = {p0_bank_at_10, p0_bank_at_9, p0_bank_at_5, p0_bank_at_4, p0_bank_at_0};
  reg p0_valid;
  always @ (posedge clk) begin
    p0_valid <= rst ? 1'b0 : in_valid;
  end
  wire [10:0] stage_1_out_comb;
  pipe_cycle1 stage_1 (
    .a(p0_bank_result),
    .out(stage_1_out_comb)
  );
  reg p1_bank_at_10;
  reg [3:0] p1_bank_at_5;
  reg [3:0] p1_bank_at_0;
  always @ (posedge clk) begin
    p1_bank_at_10 <= p0_valid ? stage_1_out_comb[10:10] : p1_bank_at_10;
    p1_bank_at_5 <= p0_valid & stage_1_out_comb[9] ? stage_1_out_comb[8:5] : p1_bank_at_5;
    p1_bank_at_0 <= p0_valid & stage_1_out_comb[4] ? stage_1_out_comb[3:0] : p1_bank_at_0;
  end
  wire [10:0] p1_bank_result;
  assign p1_bank_result = {p1_bank_at_10, 1'b0, p1_bank_at_5, 1'b0, p1_bank_at_0};
  reg p1_valid;
  always @ (posedge clk) begin
    p1_valid <= rst ? 1'b0 : p0_valid;
  end
  wire [3:0] stage_2_out_comb;
  pipe_cycle2 stage_2 (
    .a(p1_bank_result),
    .out(stage_2_out_comb)
  );
  assign out = stage_2_out_comb;
  assign out_valid = p1_valid;
endmodule
