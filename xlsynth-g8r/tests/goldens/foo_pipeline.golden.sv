module foo_cycle0(
  input wire clk,
  input wire [63:0] s,
  output wire [63:0] out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [63:0] p0_s;
  always_ff @ (posedge clk) begin
    p0_s <= s;
  end
  assign out = p0_s;
endmodule

module foo_cycle1(
  input wire clk,
  input wire [63:0] s,
  output wire [31:0] out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [63:0] p0_s;
  always_ff @ (posedge clk) begin
    p0_s <= s;
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_s_a_comb;
  wire [31:0] p1_s_b_comb;
  wire [31:0] p1_add_11_comb;
  assign p1_s_a_comb = p0_s[63:32];
  assign p1_s_b_comb = p0_s[31:0];
  assign p1_add_11_comb = p1_s_a_comb + p1_s_b_comb;

  // Registers for pipe stage 1:
  reg [31:0] p1_add_11;
  always_ff @ (posedge clk) begin
    p1_add_11 <= p1_add_11_comb;
  end
  assign out = p1_add_11;
endmodule
module foo_pipeline(
  input wire clk,
  input wire [63:0] s,
  output wire [31:0] out
);
  wire [63:0] stage0_out;
  foo_cycle0 foo_cycle0_i (
    .clk(clk),
    .s(s),
    .out(stage0_out)
  );
  wire [31:0] final_out;
  foo_cycle1 foo_cycle1_i (
    .clk(clk),
    .s(stage0_out),
    .out(final_out)
  );
  assign out = final_out;
endmodule
