// SPDX-License-Identifier: Apache-2.0

//! "Structure Zoo" i.e. a collection of structures with various shapes and forms and properties put together in one place.
//!
//! Note: the structs in here are nonsense, just for testing.

enum TransactionType : u1 {
    READ = 0,
    WRITE = 1,
}

/// Struct with an enum field.
struct Transaction {
    address: u32,
    data: u32,
    ty: TransactionType,
}

struct Control {
    enable: bool,
    reset: bool,
}

/// Struct with a nested struct.
struct Interface {
    control: Control,
    data: u32,
}

enum AluOp : u2 {
    ADD = 0,
    SUB = 1,
    AND = 2,
}

struct Operands {
    operand_a: u16,
    operand_b: u16,
}

/// Struct with nested struct and enum fields.
struct AluInput {
    inputs: Operands,
    operation: AluOp,
}

enum BurstType: u1 {
    SINGLE = 0,
    BURST = 1,
}

/// Struct with a bits type, array of bits type, and an enum.
struct MemRequest {
    address: u32,
    data: u64[4],
    burst_type: BurstType,
}

struct CacheLine {
    address: u32,
    data: u16,
}

struct Cache {
    lines: CacheLine[8],
    valid: bool,
}

// --

enum OpType : u2 {
    OP_READ = 0,
    OP_WRITE = 1,
}

/// Struct with a ready/valid handshake and enum type.
struct Handshake {
    valid: bool,
    ready: bool,
    operation: OpType,
}

// --

struct Data {
    byte_data: u8,
    word_data: u16,
}

/// Struct with mixed width fields and a nested array struct.
struct MixedStruct {
    packets: Data[4],
    flags: u4,
}

