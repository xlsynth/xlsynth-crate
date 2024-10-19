typedef enum logic {
    READ = 1'd0,
    WRITE = 1'd1
} transaction_type_e;

typedef struct packed {
    logic [31:0] address;
    logic [31:0] data;
    transaction_type_e ty;
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
    ADD = 2'd0,
    SUB = 2'd1,
    AND = 2'd2
} alu_op_e;

typedef struct packed {
    logic [15:0] operand_a;
    logic [15:0] operand_b;
} operands_t;

typedef struct packed {
    operands_t inputs;
    alu_op_e operation;
} alu_input_t;

typedef enum logic {
    SINGLE = 1'd0,
    BURST = 1'd1
} burst_type_e;

typedef struct packed {
    logic [31:0] address;
    logic [63:0] data[4];
    burst_type_e burst_type;
} mem_request_t;

typedef struct packed {
    logic [31:0] address;
    logic [15:0] data;
} cache_line_t;

typedef struct packed {
    cache_line_t lines[8];
    logic valid;
} cache_t;

typedef enum logic [1:0] {
    READ = 2'd0,
    WRITE = 2'd1
} op_type_e;

typedef struct packed {
    logic valid;
    logic ready;
    op_type_e operation;
} handshake_t;

typedef struct packed {
    logic [7:0] byte_data;
    logic [15:0] word_data;
} data_t;

typedef struct packed {
    data_t packets[4];
    logic [3:0] flags;
} mixed_struct_t;
