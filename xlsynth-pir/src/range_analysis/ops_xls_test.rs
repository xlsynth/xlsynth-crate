// SPDX-License-Identifier: Apache-2.0

//! Opt-in differential checks using a locally built raw-XLS transfer oracle.
//!
//! The oracle writes the pinned version header below, then accepts one request
//! per line: `op output_width extra argument_count`, followed by each
//! argument's `width interval_count` and normalized pairs of MSB-first binary
//! endpoints. `-` represents a zero-width endpoint. Responses contain `width
//! count` and endpoint pairs. `extra` is a slice start or the one-hot
//! LSB-priority flag. The local C++ helper calls interval_ops directly, without
//! other engines.

use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

use super::*;

const ORACLE_TIMEOUT: Duration = Duration::from_secs(30);

/// Polls nonblocking completion checks without extending the caller's deadline.
fn wait_until<T>(
    deadline: Instant,
    mut poll: impl FnMut() -> io::Result<Option<T>>,
) -> io::Result<T> {
    loop {
        if let Some(value) = poll()? {
            return Ok(value);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for completion",
            ));
        }
        thread::sleep(remaining.min(Duration::from_millis(10)));
    }
}

/// Requests termination on failure and reaps the child with bounded waiting.
struct Oracle(Child);

impl Oracle {
    fn terminate(&mut self) -> io::Result<()> {
        if self.0.try_wait().is_ok_and(|status| status.is_some()) {
            return Ok(());
        }
        if let Err(error) = self.0.kill() {
            // The child can exit between try_wait and kill; still reap it.
            if self.0.try_wait()?.is_some() {
                return Ok(());
            }
            return Err(error);
        }
        wait_until(Instant::now() + ORACLE_TIMEOUT, || self.0.try_wait()).map(|_| ())
    }
}

impl Drop for Oracle {
    fn drop(&mut self) {
        let _ = self.terminate();
    }
}

/// Runs only when selected by the subprocess guardrail tests below.
#[test]
fn oracle_shutdown_fixture() {
    let Ok(mode) = std::env::var("XLS_RANGE_ORACLE_SHUTDOWN_FIXTURE") else {
        // Normal test runs do not need a fixture process.
        return;
    };
    match mode.as_str() {
        "exit" => {
            // A successful immediate exit exercises the normal shutdown path.
        }
        "wait_for_eof" => {
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
        }
        _ => panic!("unexpected oracle fixture mode: {mode}"),
    }
}

/// Reuses this test executable without an XLS installation, shell, or files.
fn shutdown_fixture(mode: &str) -> Oracle {
    let module = module_path!().split_once("::").unwrap().1;
    Oracle(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("{module}::oracle_shutdown_fixture"),
                "--nocapture",
            ])
            .env("XLS_RANGE_ORACLE_SHUTDOWN_FIXTURE", mode)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    )
}

#[test]
fn oracle_shutdown_reaps_exited_and_terminated_children() {
    let mut exited = shutdown_fixture("exit");
    assert!(
        wait_until(Instant::now() + ORACLE_TIMEOUT, || exited.0.try_wait())
            .unwrap()
            .success()
    );

    let mut stalled = shutdown_fixture("wait_for_eof");
    assert_eq!(
        wait_until(Instant::now(), || stalled.0.try_wait())
            .unwrap_err()
            .kind(),
        io::ErrorKind::TimedOut
    );
    stalled.terminate().unwrap();
    assert!(stalled.0.try_wait().unwrap().is_some());
}

#[test]
fn oracle_reader_completion_wait_is_bounded() {
    let (release, blocked) = mpsc::channel();
    let reader = thread::spawn(move || blocked.recv().unwrap());
    assert_eq!(
        wait_until(Instant::now(), || Ok(reader.is_finished().then_some(())))
            .unwrap_err()
            .kind(),
        io::ErrorKind::TimedOut
    );
    release.send(()).unwrap();
    wait_until(Instant::now() + ORACLE_TIMEOUT, || {
        Ok(reader.is_finished().then_some(()))
    })
    .unwrap();
    reader.join().unwrap();
}

fn binary(value: &IrBits) -> String {
    if value.get_bit_count() == 0 {
        return "-".to_string();
    }
    (0..value.get_bit_count())
        .rev()
        .map(|i| if value.get_bit(i).unwrap() { '1' } else { '0' })
        .collect()
}

fn parse_binary(width: usize, text: &str) -> IrBits {
    if width == 0 {
        assert_eq!(text, "-");
        return IrBits::zero(0);
    }
    assert_eq!(text.len(), width);
    assert!(text.bytes().all(|b| b == b'0' || b == b'1'));
    IrBits::from_lsb_fn(width, |i| text.as_bytes()[width - i - 1] == b'1')
}

fn random_bits(rng: &mut Pcg64Mcg, width: usize) -> IrBits {
    IrBits::from_lsb_fn(width, |_| rng.gen_bool(0.5))
}

/// Generates broad ranges, tiny ranges, and independently fragmented unions.
fn random_set(rng: &mut Pcg64Mcg, width: usize) -> IntervalSet {
    match rng.gen_range(0..6) {
        0 => IntervalSet::full(width),
        1 => IntervalSet::singleton(random_bits(rng, width)),
        mode => {
            let count = if mode == 2 { 1 } else { rng.gen_range(1..=70) };
            let mut intervals = Vec::new();
            for _ in 0..count {
                let lo = random_bits(rng, width);
                let hi = match mode {
                    2 | 3 => lo.add(&index_bits(width, rng.gen_range(0..8))),
                    4 => random_bits(rng, width),
                    _ => lo.clone(),
                };
                intervals.push((lo, hi));
            }
            make(width, intervals)
        }
    }
}

fn run_rust(op: &str, args: &[IntervalSet], width: usize, extra: usize) -> IntervalSet {
    let a = &args[0];
    match op {
        "neg" => unop(Unop::Neg, a, width),
        "not" => unop(Unop::Not, a, width),
        "and_reduce" => unop(Unop::AndReduce, a, width),
        "or_reduce" => unop(Unop::OrReduce, a, width),
        "xor_reduce" => unop(Unop::XorReduce, a, width),
        "encode" => encode(a, width),
        "decode" => decode(a, width),
        "sign_ext" => extend(a, width, true),
        "zero_ext" => extend(a, width, false),
        "truncate" => truncate(a, width),
        "bit_slice" => bit_slice(a, extra, width),
        "one_hot" => one_hot(a, extra != 0),
        "concat" => concat(&args.iter().collect::<Vec<_>>()),
        "and" => nary(NaryOp::And, &[a, &args[1]], width),
        "or" => nary(NaryOp::Or, &[a, &args[1]], width),
        "xor" => nary(NaryOp::Xor, &[a, &args[1]], width),
        "dynamic_bit_slice" => dynamic_bit_slice(a, &args[1], width),
        _ => binop(
            crate::ir::operator_to_binop(op).expect("supported oracle binary operation"),
            a,
            &args[1],
            width,
        ),
    }
}

/// Compares generated interval givens, including fragmentation/work boundaries.
#[test]
#[ignore = "requires XLS_RANGE_TRANSFER_ORACLE pointing to a local raw-XLS interval_ops oracle"]
fn generated_interval_givens_are_at_least_as_precise_as_xls() {
    let binary_path = std::env::var_os("XLS_RANGE_TRANSFER_ORACLE")
        .expect("set XLS_RANGE_TRANSFER_ORACLE to the locally built range_transfer_oracle binary");
    let samples: usize = std::env::var("XLS_RANGE_TRANSFER_SAMPLES")
        .unwrap_or_else(|_| "10000".into())
        .parse()
        .unwrap();
    let seed: u64 = std::env::var("XLS_RANGE_TRANSFER_SEED")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap();
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut child = Oracle(
        Command::new(binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start transfer oracle"),
    );
    let mut stdin = child.0.stdin.take().unwrap();
    let stdout = child.0.stdout.take().unwrap();
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let header = receiver
        .recv_timeout(ORACLE_TIMEOUT)
        .expect("oracle did not emit its version header")
        .expect("read oracle version header");
    assert_eq!(
        header,
        "xls-range-transfer-v1 interval_ops v0.54.7 78446462c07943896edd7d18720b205e3c0e0601",
        "wrong oracle protocol or XLS revision"
    );
    let ops = [
        "add",
        "sub",
        "umul",
        "smul",
        "udiv",
        "sdiv",
        "umod",
        "smod",
        "shll",
        "shrl",
        "shra",
        "eq",
        "ne",
        "ult",
        "ule",
        "ugt",
        "uge",
        "slt",
        "sle",
        "sgt",
        "sge",
        "neg",
        "not",
        "and_reduce",
        "or_reduce",
        "xor_reduce",
        "encode",
        "decode",
        "sign_ext",
        "zero_ext",
        "truncate",
        "bit_slice",
        "one_hot",
        "concat",
        "and",
        "or",
        "xor",
        "gate",
        "dynamic_bit_slice",
    ];
    let widths = [1, 2, 3, 4, 5, 8, 16, 63, 64, 65, 127, 128, 129, 257];
    let mut strictly_tighter = 0;
    for sample in 0..samples {
        let op = ops[rng.gen_range(0..ops.len())];
        let input_width = widths[rng.gen_range(0..widths.len())];
        let mut width = input_width;
        let mut extra = 0;
        let a = random_set(&mut rng, if op == "gate" { 1 } else { input_width });
        let unary = matches!(
            op,
            "neg"
                | "not"
                | "and_reduce"
                | "or_reduce"
                | "xor_reduce"
                | "encode"
                | "decode"
                | "sign_ext"
                | "zero_ext"
                | "truncate"
                | "bit_slice"
                | "one_hot"
        );
        let mut args = vec![a];
        if !unary {
            let bw = if matches!(op, "shll" | "shrl" | "shra" | "dynamic_bit_slice") {
                widths[rng.gen_range(0..widths.len())]
            } else {
                input_width
            };
            args.push(random_set(&mut rng, bw));
        }
        match op {
            "eq" | "ne" | "ult" | "ule" | "ugt" | "uge" | "slt" | "sle" | "sgt" | "sge"
            | "and_reduce" | "or_reduce" | "xor_reduce" => width = 1,
            "umul" | "smul" => {
                width = [1, input_width, input_width + 1, input_width * 2 + 1][rng.gen_range(0..4)]
            }
            "encode" => {
                width = if input_width <= 1 {
                    0
                } else {
                    usize::BITS as usize - (input_width - 1).leading_zeros() as usize
                }
            }
            "decode" => width = rng.gen_range(1..=130),
            "sign_ext" | "zero_ext" => width += rng.gen_range(0..=64),
            "truncate" => width = rng.gen_range(0..=input_width),
            "bit_slice" => {
                extra = rng.gen_range(0..=input_width);
                width = rng.gen_range(0..=input_width - extra);
            }
            "one_hot" => {
                width += 1;
                extra = usize::from(rng.gen_bool(0.5));
            }
            "concat" => width *= 2,
            "dynamic_bit_slice" => width = rng.gen_range(1..=input_width),
            _ => { /* Other operations preserve the input width. */ }
        }
        let mut request = format!("{op} {width} {extra} {}", args.len());
        for a in &args {
            request.push_str(&format!(" {} {}", a.width(), a.intervals().len()));
            for interval in a.intervals() {
                request.push_str(&format!(
                    " {} {}",
                    binary(interval.lower()),
                    binary(interval.upper())
                ));
            }
        }
        writeln!(stdin, "{request}").unwrap();
        stdin.flush().unwrap();
        let response = receiver.recv_timeout(ORACLE_TIMEOUT)
            .unwrap_or_else(|error| panic!("oracle failed/timed out at seed={seed} sample={sample}: {error}; request={request}"))
            .expect("read oracle response");
        let mut fields = response.split_whitespace();
        assert_eq!(
            fields.next().unwrap().parse::<usize>().unwrap(),
            width,
            "{request}"
        );
        let count = fields.next().unwrap().parse::<usize>().unwrap();
        let mut intervals = Vec::new();
        for _ in 0..count {
            intervals.push((
                parse_binary(width, fields.next().unwrap()),
                parse_binary(width, fields.next().unwrap()),
            ));
        }
        assert!(fields.next().is_none(), "extra oracle output: {response}");
        let xls = make(width, intervals);
        let rust = run_rust(op, &args, width, extra);
        assert!(
            rust.is_subset_of(&xls),
            "precision regression at seed={seed} sample={sample}: request={request}; Rust={rust:?}; XLS={xls:?}"
        );
        strictly_tighter += usize::from(!xls.is_subset_of(&rust));
    }
    drop(stdin);
    let deadline = Instant::now() + ORACLE_TIMEOUT;
    assert!(
        wait_until(deadline, || child.0.try_wait())
            .expect("oracle failed or timed out while exiting")
            .success()
    );
    wait_until(deadline, || Ok(reader.is_finished().then_some(())))
        .expect("oracle reader timed out while exiting");
    reader.join().unwrap();
    eprintln!(
        "range transfer differential: seed={seed}, samples={samples}, strictly tighter={strictly_tighter}"
    );
}
