// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use xlsynth::{IrPackage, IrProcRuntime, XlsIrValue, XlsynthError};

// Consume one 32-bit word per tick, least significant bit first, and emit the
// running CRC-32 register before the final XOR.
const CRC32_PROC: &str = r#"package crc32
fn crc_bit(i: bits[5], crc: bits[32]) -> bits[32] {
  low: bits[1] = bit_slice(crc, start=0, width=1)
  one: bits[1] = literal(value=1)
  shifted: bits[32] = shrl(crc, one)
  polynomial: bits[32] = literal(value=0xedb88320)
  feedback: bits[32] = xor(shifted, polynomial)
  ret next_crc: bits[32] = sel(low, cases=[shifted, feedback])
}

top proc crc32<input: bits[32] in, output: bits[32] out>(crc: bits[32], init={0xffffffff}) {
  chan_interface input(direction=receive, kind=streaming, strictness=proven_mutually_exclusive)
  chan_interface output(direction=send, kind=streaming, strictness=proven_mutually_exclusive)
  tok: token = literal(value=token)
  received: (token, bits[32]) = receive(tok, channel=input)
  received_tok: token = tuple_index(received, index=0)
  word: bits[32] = tuple_index(received, index=1)
  current: bits[32] = state_read(state_element=crc)
  mixed: bits[32] = xor(current, word)
  result: bits[32] = counted_for(mixed, trip_count=32, stride=1, body=crc_bit)
  sent: token = send(received_tok, result, channel=output)
  update: () = next_value(state_element=crc, value=result)
}
"#;

type Constructor = fn(IrPackage) -> Result<IrProcRuntime, XlsynthError>;
const BACKENDS: [(&str, Constructor); 2] = [
    ("interpreter", IrProcRuntime::new),
    ("jit", IrProcRuntime::new_jit),
];

/// Performs one complete input, tick, and output transaction.
fn tick_word(runtime: &mut IrProcRuntime, word: &XlsIrValue) -> Result<XlsIrValue, XlsynthError> {
    runtime.write("input", word)?;
    runtime.tick()?;
    runtime
        .try_read("output")?
        .ok_or_else(|| XlsynthError("CRC proc emitted no output".to_owned()))
}

/// Measures runtime construction separately from throughput with channel I/O.
fn bench_crc32(c: &mut Criterion) {
    let mut group = c.benchmark_group("proc_crc32");
    for (backend, create) in BACKENDS {
        group.bench_function(BenchmarkId::new("create", backend), |b| {
            b.iter_batched(
                || IrPackage::parse_ir(CRC32_PROC, None).unwrap(),
                |package| create(package).unwrap(),
                BatchSize::PerIteration,
            );
        });
    }

    let words: Vec<_> = (0..64u32)
        .map(|i| XlsIrValue::u32(i.wrapping_mul(0x9e3779b9)))
        .collect();
    group.throughput(Throughput::Elements(words.len() as u64));
    for (backend, create) in BACKENDS {
        let mut runtime = create(IrPackage::parse_ir(CRC32_PROC, None).unwrap()).unwrap();
        let mut crc = u32::MAX;
        for word in &words {
            crc ^= word.to_u32().unwrap();
            for _ in 0..32 {
                crc = (crc >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(crc & 1));
            }
            assert_eq!(
                tick_word(&mut runtime, word).unwrap().to_u32().unwrap(),
                crc
            );
        }

        group.bench_function(BenchmarkId::new("stream_64_words", backend), |b| {
            b.iter(|| {
                for word in &words {
                    black_box(tick_word(&mut runtime, word).unwrap());
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_crc32);
criterion_main!(benches);
