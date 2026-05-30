// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::fmt::Write;
use std::hint::black_box;
use std::time::{Duration, Instant};

use xlsynth::{IrFunctionJit, IrPackage, IrValue};
use xlsynth_pir::ir::Fn;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils::operands;
use xlsynth_pir_compiler::PirFunctionJit;

const LARGE_FUNCTION_INSTRUCTION_COUNT: usize = 10_000;

/// Builds one fully reachable large function using several supported lowering
/// paths.
fn make_large_varied_function(node_count: usize) -> String {
    assert!(node_count >= 4);
    let mut ir = String::from(
        r#"package scaling

fn f(x: bits[32] id=1, y: bits[32] id=2, index: bits[2] id=3) -> bits[32] {
"#,
    );
    let mut next_id = 4;
    let mut current = String::from("x");
    let mut template_index = 0;

    while next_id < node_count {
        let available_before_return = node_count - next_id;
        match template_index % 12 {
            0 => {
                writeln!(
                    ir,
                    "  v{next_id}: bits[32] = add({current}, y, id={next_id})"
                )
                .unwrap();
                current = format!("v{next_id}");
                next_id += 1;
            }
            1 => {
                writeln!(
                    ir,
                    "  v{next_id}: bits[32] = xor({current}, x, id={next_id})"
                )
                .unwrap();
                current = format!("v{next_id}");
                next_id += 1;
            }
            2 => {
                writeln!(
                    ir,
                    "  v{next_id}: bits[32] = sub({current}, x, id={next_id})"
                )
                .unwrap();
                current = format!("v{next_id}");
                next_id += 1;
            }
            3 => {
                writeln!(
                    ir,
                    "  v{next_id}: bits[32] = umul({current}, y, id={next_id})"
                )
                .unwrap();
                current = format!("v{next_id}");
                next_id += 1;
            }
            4 => {
                writeln!(ir, "  v{next_id}: bits[32] = not({current}, id={next_id})").unwrap();
                current = format!("v{next_id}");
                next_id += 1;
            }
            5 => {
                writeln!(
                    ir,
                    "  v{next_id}: bits[32] = shll({current}, index, id={next_id})"
                )
                .unwrap();
                current = format!("v{next_id}");
                next_id += 1;
            }
            6 => {
                writeln!(
                    ir,
                    "  v{next_id}: bits[32] = and({current}, y, id={next_id})"
                )
                .unwrap();
                current = format!("v{next_id}");
                next_id += 1;
            }
            7 => {
                writeln!(
                    ir,
                    "  v{next_id}: bits[32] = or({current}, x, id={next_id})"
                )
                .unwrap();
                current = format!("v{next_id}");
                next_id += 1;
            }
            8 if available_before_return >= 2 => {
                let widened_id = next_id + 1;
                writeln!(
                    ir,
                    "  v{next_id}: bits[1] = ult({current}, y, id={next_id})"
                )
                .unwrap();
                writeln!(
                    ir,
                    "  v{widened_id}: bits[32] = zero_ext(v{next_id}, new_bit_count=32, id={widened_id})"
                )
                .unwrap();
                current = format!("v{widened_id}");
                next_id += 2;
            }
            9 if available_before_return >= 2 => {
                let selected_id = next_id + 1;
                writeln!(
                    ir,
                    "  v{next_id}: bits[32][2] = array({current}, y, id={next_id})"
                )
                .unwrap();
                writeln!(
                    ir,
                    "  v{selected_id}: bits[32] = array_index(v{next_id}, indices=[index], id={selected_id})"
                )
                .unwrap();
                current = format!("v{selected_id}");
                next_id += 2;
            }
            10 if available_before_return >= 3 => {
                let high_id = next_id + 1;
                let joined_id = next_id + 2;
                writeln!(
                    ir,
                    "  v{next_id}: bits[16] = bit_slice({current}, start=0, width=16, id={next_id})"
                )
                .unwrap();
                writeln!(
                    ir,
                    "  v{high_id}: bits[16] = bit_slice(y, start=0, width=16, id={high_id})"
                )
                .unwrap();
                writeln!(
                    ir,
                    "  v{joined_id}: bits[32] = concat(v{high_id}, v{next_id}, id={joined_id})"
                )
                .unwrap();
                current = format!("v{joined_id}");
                next_id += 3;
            }
            11 if available_before_return >= 2 => {
                let widened_id = next_id + 1;
                writeln!(
                    ir,
                    "  v{next_id}: bits[8] = bit_slice({current}, start=0, width=8, id={next_id})"
                )
                .unwrap();
                writeln!(
                    ir,
                    "  v{widened_id}: bits[32] = sign_ext(v{next_id}, new_bit_count=32, id={widened_id})"
                )
                .unwrap();
                current = format!("v{widened_id}");
                next_id += 2;
            }
            _ => {
                writeln!(ir, "  v{next_id}: bits[32] = neg({current}, id={next_id})").unwrap();
                current = format!("v{next_id}");
                next_id += 1;
            }
        }
        template_index += 1;
    }

    writeln!(
        ir,
        "  ret result: bits[32] = identity({current}, id={node_count})\n}}"
    )
    .unwrap();
    ir
}

/// Counts the PIR instructions transitively needed to compute the return value.
fn live_instruction_count(function: &Fn) -> usize {
    let mut pending = vec![
        function
            .ret_node_ref
            .expect("generated function should have a return node"),
    ];
    let mut live = HashSet::new();
    while let Some(node_ref) = pending.pop() {
        if live.insert(node_ref) {
            pending.extend(operands(&function.get_node(node_ref).payload));
        }
    }
    live.len()
}

#[test]
fn compiles_and_executes_varied_ten_thousand_node_function() {
    let ir = make_large_varied_function(LARGE_FUNCTION_INSTRUCTION_COUNT);
    let package = Parser::new(&ir)
        .parse_and_validate_package()
        .expect("large PIR function should parse and validate");
    let function = package.get_fn("f").expect("function f should exist");
    // `nodes[0]` is PIR's reserved Nil sentinel, not a generated instruction.
    assert_eq!(function.nodes.len() - 1, LARGE_FUNCTION_INSTRUCTION_COUNT);
    assert_eq!(
        live_instruction_count(function),
        LARGE_FUNCTION_INSTRUCTION_COUNT
    );

    let jit = PirFunctionJit::compile(function).expect("large PIR function should JIT compile");
    for (x, y, index) in [
        (0_u64, 0_u64, 0_u64),
        (0x1234_5678_u64, 0x89ab_cdef_u64, 1_u64),
        (u32::MAX as u64, 17_u64, 3_u64),
    ] {
        let args = [
            IrValue::make_ubits(32, x).unwrap(),
            IrValue::make_ubits(32, y).unwrap(),
            IrValue::make_ubits(2, index).unwrap(),
        ];
        let expected = match eval_fn(function, &args) {
            FnEvalResult::Success(success) => success.value,
            failure => panic!("PIR evaluator failed on large function: {failure:?}"),
        };
        let actual = jit
            .run_ir_values(&args)
            .expect("large JIT function should execute");
        assert_eq!(actual, expected);
    }
}

struct JitTiming {
    compilation: Duration,
    execution: Duration,
    value: IrValue,
}

/// Compiles and repeatedly executes a large live function through Cranelift.
fn measure_varied_function_cranelift(instruction_count: usize, execution_count: usize) {
    let generation_start = Instant::now();
    let ir = make_large_varied_function(instruction_count);
    let generation = generation_start.elapsed();
    let args = [
        IrValue::make_ubits(32, 0x1234_5678).unwrap(),
        IrValue::make_ubits(32, 0x89ab_cdef).unwrap(),
        IrValue::make_ubits(2, 1).unwrap(),
    ];

    let pir_parse_start = Instant::now();
    let pir_package = Parser::new(&ir)
        .parse_and_validate_package()
        .expect("large PIR function should parse and validate");
    let pir_parse = pir_parse_start.elapsed();
    let pir_function = pir_package.get_fn("f").expect("function f should exist");
    assert_eq!(pir_function.nodes.len() - 1, instruction_count);
    assert_eq!(live_instruction_count(pir_function), instruction_count);

    let timing = {
        let compilation_start = Instant::now();
        let jit =
            PirFunctionJit::compile(pir_function).expect("large PIR function should JIT compile");
        let compilation = compilation_start.elapsed();
        let _ = jit
            .run_ir_values(&args)
            .expect("Cranelift warm-up execution should succeed");
        let execution_start = Instant::now();
        let mut value = None;
        for _ in 0..execution_count {
            value = Some(black_box(
                jit.run_ir_values(black_box(&args))
                    .expect("Cranelift timed execution should succeed"),
            ));
        }
        JitTiming {
            compilation,
            execution: execution_start.elapsed(),
            value: value.expect("execution count should be nonzero"),
        }
    };
    eprintln!(
        "backend=cranelift instructions={instruction_count} live_instructions={instruction_count} \
         ir_bytes={} executions={execution_count} value={:?}\n\
         generation={generation:?} pir_parse={pir_parse:?} compile={:?} \
         execute_total={:?} execute_per_call={:?}",
        ir.len(),
        timing.value,
        timing.compilation,
        timing.execution,
        timing.execution / execution_count as u32,
    );
}

/// Compiles and repeatedly executes the generated function through libxls JIT.
fn measure_varied_function_xls(instruction_count: usize, execution_count: usize) {
    let generation_start = Instant::now();
    let ir = make_large_varied_function(instruction_count);
    let generation = generation_start.elapsed();
    let args = [
        IrValue::make_ubits(32, 0x1234_5678).unwrap(),
        IrValue::make_ubits(32, 0x89ab_cdef).unwrap(),
        IrValue::make_ubits(2, 1).unwrap(),
    ];

    let parse_start = Instant::now();
    let package = IrPackage::parse_ir(&ir, None).expect("XLS should parse large function");
    let parse = parse_start.elapsed();
    let function = package
        .get_function("f")
        .expect("XLS function f should exist");
    let timing = {
        let compilation_start = Instant::now();
        let jit = IrFunctionJit::new(&function).expect("XLS JIT should compile function");
        let compilation = compilation_start.elapsed();
        let _ = jit.run(&args).expect("XLS JIT warm-up should succeed");
        let execution_start = Instant::now();
        let mut value = None;
        for _ in 0..execution_count {
            value = Some(black_box(
                jit.run(black_box(&args))
                    .expect("XLS JIT timed execution should succeed")
                    .value,
            ));
        }
        JitTiming {
            compilation,
            execution: execution_start.elapsed(),
            value: value.expect("execution count should be nonzero"),
        }
    };
    eprintln!(
        "backend=xls instructions={instruction_count} ir_bytes={} executions={execution_count} \
         value={:?}\n\
         generation={generation:?} parse={parse:?} compile={:?} \
         execute_total={:?} execute_per_call={:?}",
        ir.len(),
        timing.value,
        timing.compilation,
        timing.execution,
        timing.execution / execution_count as u32,
    );
}

#[test]
#[ignore = "manual compilation and execution scaling measurement"]
fn measures_cranelift_varied_hundred_thousand_instruction_function() {
    measure_varied_function_cranelift(100_000, 100);
}

#[test]
#[ignore = "manual compilation and execution scaling measurement"]
fn measures_cranelift_varied_one_million_instruction_function() {
    measure_varied_function_cranelift(1_000_000, 10);
}

#[test]
#[ignore = "manual compilation and execution scaling measurement"]
fn measures_xls_varied_hundred_thousand_instruction_function() {
    measure_varied_function_xls(100_000, 100);
}

#[test]
#[ignore = "manual compilation and execution scaling measurement"]
fn measures_xls_varied_one_million_instruction_function() {
    measure_varied_function_xls(1_000_000, 10);
}
