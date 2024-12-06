package common_zoo_sv_pkg;
typedef enum logic {
    Read = 1'd0,
    Write = 1'd1
} transaction_type_t;

typedef logic [7:0] my_u8_t;
endpackage : common_zoo_sv_pkg