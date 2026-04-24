module __my_module__main(
  input wire clk,
  input wire rst_n,
  input wire input_valid,
  input wire [31:0] x,
  output wire output_valid,
  output wire [31:0] out
);
  // ===== Pipe stage 0:
  wire p0_reset_active_comb;
  wire p0_input_flop_load_enable_comb;
  assign p0_reset_active_comb = ~rst_n;
  assign p0_input_flop_load_enable_comb = input_valid | p0_reset_active_comb;

  // Registers for pipe stage 0:
  reg input_valid__input_flop;
  reg [31:0] x__input_flop;
  always @ (posedge clk) begin
    x__input_flop <= p0_input_flop_load_enable_comb ? x : x__input_flop;
  end
  always @ (posedge clk) begin
    if (!rst_n) begin
      input_valid__input_flop <= 1'h0;
    end else begin
      input_valid__input_flop <= input_valid;
    end
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_literal_34_comb;
  wire [31:0] p1_add_40_comb;
  wire p1_output_flop_load_enable_comb;
  assign p1_literal_34_comb = 32'h0000_0001;
  assign p1_add_40_comb = x__input_flop + p1_literal_34_comb;
  assign p1_output_flop_load_enable_comb = input_valid__input_flop | p0_reset_active_comb;

  // Registers for pipe stage 1:
  reg output_valid__output_flop;
  reg [31:0] out__output_flop;
  always @ (posedge clk) begin
    out__output_flop <= p1_output_flop_load_enable_comb ? p1_add_40_comb : out__output_flop;
  end
  always @ (posedge clk) begin
    if (!rst_n) begin
      output_valid__output_flop <= 1'h0;
    end else begin
      output_valid__output_flop <= input_valid__input_flop;
    end
  end
  assign output_valid = output_valid__output_flop;
  assign out = out__output_flop;
endmodule

