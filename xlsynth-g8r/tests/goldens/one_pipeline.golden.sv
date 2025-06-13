module one_cycle0(
  input wire clk,
  input wire [31:0] x,
  input wire [31:0] y,
  output wire [31:0] out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [31:0] p0_x;
  reg [31:0] p0_y;
  always_ff @ (posedge clk) begin
    p0_x <= x;
    p0_y <= y;
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_add_10_comb;
  assign p1_add_10_comb = p0_x + p0_y;

  // Registers for pipe stage 1:
  reg [31:0] p1_add_10;
  always_ff @ (posedge clk) begin
    p1_add_10 <= p1_add_10_comb;
  end
  assign out = p1_add_10;
endmodule
module one_pipeline(
  input wire [31:0] clk,
  input wire [31:0] x,
  input wire [31:0] y,
  output wire [31:0] out
);
  wire [31:0] stage0_out;
  wire [31:0] final_out;
  one_cycle0 one_cycle0_i (
    .clk(clk),
    .x(x),
    .y(y),
    .out(final_out)
  );
  assign out = final_out;
endmodule
