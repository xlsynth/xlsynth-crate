module __my_module__main(
  input wire clk,
  input wire rst_n,
  input wire input_valid,
  input wire [31:0] x,
  output wire output_valid,
  output wire [31:0] out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg input_valid__input_flop;
  reg [31:0] x__input_flop;
  always @ (posedge clk) begin
    if (!rst_n) begin
      input_valid__input_flop <= 1'h0;
      x__input_flop <= 32'h0000_0000;
    end else begin
      input_valid__input_flop <= input_valid;
      x__input_flop <= input_valid ? x : x__input_flop;
    end
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_literal_27_comb;
  wire [31:0] p1_add_33_comb;
  assign p1_literal_27_comb = 32'h0000_0001;
  assign p1_add_33_comb = x__input_flop + p1_literal_27_comb;

  // Registers for pipe stage 1:
  reg output_valid__output_flop;
  reg [31:0] out__output_flop;
  always @ (posedge clk) begin
    if (!rst_n) begin
      output_valid__output_flop <= 1'h0;
      out__output_flop <= 32'h0000_0000;
    end else begin
      output_valid__output_flop <= input_valid__input_flop;
      out__output_flop <= input_valid__input_flop ? p1_add_33_comb : out__output_flop;
    end
  end
  assign output_valid = output_valid__output_flop;
  assign out = out__output_flop;
endmodule

