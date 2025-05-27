module __my_module__main(
  input wire clk,
  input wire rst,
  input wire input_valid,
  input wire [31:0] x,
  output wire output_valid,
  output wire [31:0] out
);
  // ===== Pipe stage 0:
  wire p0_not_18_comb;
  wire p0_load_en_comb;
  assign p0_not_18_comb = ~rst;
  assign p0_load_en_comb = input_valid | p0_not_18_comb;

  // Registers for pipe stage 0:
  reg p0_valid;
  reg [31:0] p0_x;
  always @ (posedge clk) begin
    if (!rst) begin
      p0_valid <= 1'h0;
      p0_x <= 32'h0000_0000;
    end else begin
      p0_valid <= input_valid;
      p0_x <= p0_load_en_comb ? x : p0_x;
    end
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_literal_8_comb;
  wire [31:0] p1_add_9_comb;
  wire p1_load_en_comb;
  assign p1_literal_8_comb = 32'h0000_0001;
  assign p1_add_9_comb = p0_x + p1_literal_8_comb;
  assign p1_load_en_comb = p0_valid | p0_not_18_comb;

  // Registers for pipe stage 1:
  reg p1_valid;
  reg [31:0] p1_add_9;
  always @ (posedge clk) begin
    if (!rst) begin
      p1_valid <= 1'h0;
      p1_add_9 <= 32'h0000_0000;
    end else begin
      p1_valid <= p0_valid;
      p1_add_9 <= p1_load_en_comb ? p1_add_9_comb : p1_add_9;
    end
  end
  assign output_valid = p1_valid;
  assign out = p1_add_9;
endmodule

