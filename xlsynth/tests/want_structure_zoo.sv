typedef struct packed {
    logic [31:0] address;
    logic [31:0] data;
    common_zoo_sv_pkg::transaction_type_t ty;
} transaction_t;

typedef struct packed {
    logic enable;
    logic reset;
} control_t;

typedef struct packed {
    control_t control;
    logic [31:0] data;
} interface_t;

typedef enum logic [1:0] {
    Add = 2'd0,
    Sub = 2'd1,
    And = 2'd2
} alu_op_t;

typedef struct packed {
    logic [15:0] operand_a;
    logic [15:0] operand_b;
} operands_t;

typedef struct packed {
    operands_t inputs;
    alu_op_t operation;
} alu_input_t;

typedef enum logic {
    Single = 1'd0,
    Burst = 1'd1
} burst_type_t;

typedef struct packed {
    logic [31:0] address;
    logic [3:0] [63:0] data;
    burst_type_t burst_type;
} mem_request_t;

typedef struct packed {
    logic [31:0] address;
    logic [15:0] data;
} cache_line_t;

typedef struct packed {
    cache_line_t [7:0] lines;
    logic valid;
} cache_t;

typedef enum logic [1:0] {
    OpRead = 2'd0,
    OpWrite = 2'd1
} op_type_t;

typedef struct packed {
    logic valid;
    logic ready;
    op_type_t operation;
} handshake_t;

typedef struct packed {
    logic [7:0] byte_data;
    logic [15:0] word_data;
} data_t;

typedef struct packed {
    data_t [3:0] packets;
    logic [3:0] flags;
} mixed_struct_t;
