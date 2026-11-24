// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrPackage, IrProcRuntime, TraceMessage, XlsIrValue, XlsynthError};

const ACCUMULATOR: &str = r#"package test
top proc sum<input: bits[8] in, output: bits[8] out>(total: bits[8], init={7}) {
  chan_interface input(direction=receive, kind=streaming, strictness=proven_mutually_exclusive)
  chan_interface output(direction=send, kind=streaming, strictness=proven_mutually_exclusive)
  tok: token = literal(value=token)
  received: (token, bits[8]) = receive(tok, channel=input)
  received_tok: token = tuple_index(received, index=0)
  input_value: bits[8] = tuple_index(received, index=1)
  current: bits[8] = state_read(state_element=total)
  sum_value: bits[8] = add(current, input_value)
  enabled: bits[1] = literal(value=1)
  traced: token = trace(received_tok, enabled, format="total={}", data_operands=[sum_value], verbosity=3)
  sent: token = send(traced, sum_value, channel=output)
  update: () = next_value(state_element=total, value=sum_value)
}
"#;

#[test]
fn ticks_preserve_proc_state_and_transfer_owned_outputs() -> Result<(), XlsynthError> {
    let mut runtime = IrProcRuntime::new(IrPackage::parse_ir(ACCUMULATOR, None)?)?;
    assert_eq!(runtime.try_read("output")?, None);
    assert!(runtime.take_trace_messages().is_empty());

    // Both input values are dropped before the runtime consumes them.
    runtime.write("input", &XlsIrValue::make_ubits(8, 3)?)?;
    runtime.write("input", &XlsIrValue::make_ubits(8, 4)?)?;
    runtime.tick()?;
    runtime.tick()?;
    let first = runtime.try_read("output")?;
    let second = runtime.try_read("output")?;
    assert_eq!(runtime.try_read("output")?, None);
    let traces = runtime.take_trace_messages();
    assert!(runtime.take_trace_messages().is_empty());
    drop(runtime);

    assert_eq!(first, Some(XlsIrValue::make_ubits(8, 10)?));
    assert_eq!(second, Some(XlsIrValue::make_ubits(8, 14)?));
    assert_eq!(
        traces,
        vec![TraceMessage {
            message: "total=14".to_string(),
            verbosity: 3,
        }]
    );
    Ok(())
}

#[test]
fn channel_errors_leave_the_runtime_usable() -> Result<(), XlsynthError> {
    let mut runtime = IrProcRuntime::new(IrPackage::parse_ir(ACCUMULATOR, None)?)?;
    let token = XlsIrValue::make_token();
    let wrong_type = runtime.write("input", &token).unwrap_err();
    assert!(wrong_type.to_string().contains("bits[8]"), "{wrong_type}");
    let missing = runtime.try_read("missing_channel").unwrap_err();
    assert!(missing.to_string().contains("missing_channel"), "{missing}");
    let input = XlsIrValue::make_ubits(8, 2)?;
    assert!(runtime.write("input\0ignored", &input).is_err());
    assert!(runtime.try_read("output\0ignored").is_err());

    runtime.write("input", &input)?;
    runtime.tick()?;
    assert_eq!(
        runtime.try_read("output")?,
        Some(XlsIrValue::make_ubits(8, 9)?)
    );
    Ok(())
}

#[test]
fn failed_tick_retains_traces_for_the_caller() -> Result<(), XlsynthError> {
    let ir = r#"package test
top proc watched<>() {
  tok: token = literal(value=token)
  enabled: bits[1] = literal(value=1)
  traced: token = trace(tok, enabled, format="before failure", data_operands=[])
  disabled: bits[1] = literal(value=0)
  checked: token = assert(traced, disabled, message="expected failure")
}
"#;
    let mut runtime = IrProcRuntime::new(IrPackage::parse_ir(ir, None)?)?;
    let failure = runtime.tick().unwrap_err();
    assert!(
        failure.to_string().contains("expected failure"),
        "{failure}"
    );
    assert_eq!(
        runtime.take_trace_messages(),
        vec![TraceMessage {
            message: "before failure".to_string(),
            verbosity: 0,
        }]
    );
    assert!(runtime.take_trace_messages().is_empty());
    Ok(())
}

#[test]
fn missing_proc_top_reports_the_native_error() -> Result<(), XlsynthError> {
    let package = IrPackage::parse_ir("package test\nproc unselected<>() {}", None)?;
    let Err(error) = IrProcRuntime::new(package) else {
        panic!("a proc-scoped runtime requires a top proc");
    };
    assert!(error.to_string().contains("Top entity not set"), "{error}");
    Ok(())
}
