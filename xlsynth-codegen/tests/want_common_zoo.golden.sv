package common_zoo_sv_pkg;
typedef enum logic {
    Read = 1'd0,
    Write = 1'd1
} transaction_type_t;

localparam bit unsigned [31:0] ValuesToHold = 32'h000000ff;

typedef logic [7:0] my_u8_t;

typedef struct packed {
    logic [15:0] x;
    logic [31:0] y;
} point_t;

localparam point_t Origin = '{
    x: 16'h0000,
    y: 32'h00000000
};
endpackage : common_zoo_sv_pkg