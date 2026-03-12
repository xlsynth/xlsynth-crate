// SPDX-License-Identifier: Apache-2.0

use xlsynth_vastly::CycleFixture;
use xlsynth_vastly::DriveContext;
use xlsynth_vastly::FixtureRunner;
use xlsynth_vastly::ObserveContext;
use xlsynth_vastly::compile_pipeline_module;

#[test]
fn port_binding_rejects_missing_or_wrong_width() {
    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic [7:0] data_in,
  output logic done
);
  assign done = rst;
endmodule
"#;

    let module = compile_pipeline_module(dut).unwrap();
    let runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();

    let err = ports.input("missing").unwrap_err();
    assert!(format!("{err:?}").contains("input port `missing` was not found"));

    let err = ports.input("clk").unwrap_err();
    assert!(
        format!("{err:?}")
            .contains("clock `clk` is managed internally and is not exposed as a fixture input")
    );

    let err = ports.output_with_width("done", 2).unwrap_err();
    assert!(format!("{err:?}").contains("output port `done` has width 1, expected 2"));
}

#[test]
fn fixture_runner_rejects_clock_dependent_outputs() {
    let dut = r#"
module m(
  input logic clk,
  output logic observed
);
  function automatic logic helper(input logic enable);
    begin
      helper = enable & clk;
    end
  endfunction

  assign observed = helper(1'b1);
endmodule
"#;

    let module = compile_pipeline_module(dut).unwrap();
    let err = match FixtureRunner::new(&module) {
        Ok(_) => panic!("expected clock-sensitive outputs to be rejected"),
        Err(err) => err,
    };
    assert!(format!("{err:?}").contains(
        "fixture runner does not support combinational outputs that depend on clock `clk`: observed"
    ));
}

#[test]
fn fixture_runner_can_drive_and_sample_packed_slices() {
    struct PackedFixture {
        in_bus: Option<xlsynth_vastly::InputPortHandle>,
        out_bus: Option<xlsynth_vastly::OutputPortHandle>,
        observed: Vec<(u64, u64)>,
    }

    impl CycleFixture for PackedFixture {
        fn bind(&mut self, ports: &xlsynth_vastly::PortBindings<'_>) -> xlsynth_vastly::Result<()> {
            self.in_bus = Some(ports.input_with_width("in_bus", 32)?);
            self.out_bus = Some(ports.output_with_width("out_bus", 32)?);
            Ok(())
        }

        fn drive_cycle_inputs(
            &mut self,
            cycle: u64,
            ctx: &mut DriveContext<'_>,
        ) -> xlsynth_vastly::Result<()> {
            if cycle == 0 {
                let in_bus = self.in_bus.as_ref().unwrap();
                ctx.set_packed_u64(in_bus, &[0], 0x11)?;
                ctx.set_packed_u64(in_bus, &[2], 0xab)?;
            }
            Ok(())
        }

        fn observe_cycle_outputs(
            &mut self,
            cycle: u64,
            ctx: &ObserveContext<'_>,
        ) -> xlsynth_vastly::Result<()> {
            let out_bus = self.out_bus.as_ref().unwrap();
            let lane0 = ctx.output_packed_u64_if_known(out_bus, &[0])?.unwrap();
            let lane2 = ctx.output_packed_u64_if_known(out_bus, &[2])?.unwrap();
            self.observed.push((cycle, (lane2 << 8) | lane0));
            Ok(())
        }
    }

    let dut = r#"
module m(
  input logic clk,
  input logic [3:0][7:0] in_bus,
  output logic [3:0][7:0] out_bus
);
  assign out_bus = in_bus;
endmodule
"#;

    let module = compile_pipeline_module(dut).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let mut fixture = PackedFixture {
        in_bus: None,
        out_bus: None,
        observed: Vec::new(),
    };
    runner.bind_fixture(&mut fixture).unwrap();
    let mut fixtures: Vec<&mut dyn CycleFixture> = vec![&mut fixture];
    runner.step_cycle(&mut fixtures).unwrap();

    assert_eq!(fixture.observed, vec![(0, 0xab11)]);
}

#[test]
fn fixture_runner_handles_next_cycle_response_with_multiple_fixtures() {
    struct LaunchFixture {
        launch: Option<xlsynth_vastly::InputPortHandle>,
        payload: Option<xlsynth_vastly::InputPortHandle>,
    }

    impl CycleFixture for LaunchFixture {
        fn bind(&mut self, ports: &xlsynth_vastly::PortBindings<'_>) -> xlsynth_vastly::Result<()> {
            self.launch = Some(ports.input_with_width("launch", 1)?);
            self.payload = Some(ports.input_with_width("launch_data", 8)?);
            Ok(())
        }

        fn drive_cycle_inputs(
            &mut self,
            cycle: u64,
            ctx: &mut DriveContext<'_>,
        ) -> xlsynth_vastly::Result<()> {
            if cycle == 1 {
                ctx.set_bool(self.launch.as_ref().unwrap(), true)?;
                ctx.set_u64(self.payload.as_ref().unwrap(), 0x3c)?;
            }
            Ok(())
        }
    }

    struct ResponderFixture {
        req_valid: Option<xlsynth_vastly::OutputPortHandle>,
        req_data: Option<xlsynth_vastly::OutputPortHandle>,
        resp_valid: Option<xlsynth_vastly::InputPortHandle>,
        resp_data: Option<xlsynth_vastly::InputPortHandle>,
        scheduled_response: Option<u64>,
    }

    impl CycleFixture for ResponderFixture {
        fn bind(&mut self, ports: &xlsynth_vastly::PortBindings<'_>) -> xlsynth_vastly::Result<()> {
            self.req_valid = Some(ports.output_with_width("req_valid", 1)?);
            self.req_data = Some(ports.output_with_width("req_data", 8)?);
            self.resp_valid = Some(ports.input_with_width("resp_valid", 1)?);
            self.resp_data = Some(ports.input_with_width("resp_data", 8)?);
            Ok(())
        }

        fn drive_cycle_inputs(
            &mut self,
            _cycle: u64,
            ctx: &mut DriveContext<'_>,
        ) -> xlsynth_vastly::Result<()> {
            if let Some(value) = self.scheduled_response.take() {
                ctx.set_bool(self.resp_valid.as_ref().unwrap(), true)?;
                ctx.set_u64(self.resp_data.as_ref().unwrap(), value)?;
            }
            Ok(())
        }

        fn observe_cycle_outputs(
            &mut self,
            _cycle: u64,
            ctx: &ObserveContext<'_>,
        ) -> xlsynth_vastly::Result<()> {
            if ctx
                .output_bool_if_known(self.req_valid.as_ref().unwrap())?
                .unwrap_or(false)
            {
                let req_data = ctx
                    .output_u64_if_known(self.req_data.as_ref().unwrap())?
                    .unwrap();
                self.scheduled_response = Some(req_data + 1);
            }
            Ok(())
        }
    }

    struct ScoreFixture {
        done: Option<xlsynth_vastly::OutputPortHandle>,
        result: Option<xlsynth_vastly::OutputPortHandle>,
        observed_results: Vec<u64>,
    }

    impl CycleFixture for ScoreFixture {
        fn bind(&mut self, ports: &xlsynth_vastly::PortBindings<'_>) -> xlsynth_vastly::Result<()> {
            self.done = Some(ports.output_with_width("done", 1)?);
            self.result = Some(ports.output_with_width("result", 8)?);
            Ok(())
        }

        fn observe_cycle_outputs(
            &mut self,
            _cycle: u64,
            ctx: &ObserveContext<'_>,
        ) -> xlsynth_vastly::Result<()> {
            if ctx
                .output_bool_if_known(self.done.as_ref().unwrap())?
                .unwrap_or(false)
            {
                self.observed_results.push(
                    ctx.output_u64_if_known(self.result.as_ref().unwrap())?
                        .unwrap(),
                );
            }
            Ok(())
        }
    }

    let dut = r#"
module m(
  input logic clk,
  input logic rst,
  input logic launch,
  input logic [7:0] launch_data,
  output logic req_valid,
  output logic [7:0] req_data,
  input logic resp_valid,
  input logic [7:0] resp_data,
  output logic done,
  output logic [7:0] result
);
  logic pending;
  logic [7:0] req_data_q;
  logic [7:0] result_q;

  always_ff @(posedge clk) begin
    if (rst) begin
      pending <= 1'b0;
      req_data_q <= 8'h00;
      result_q <= 8'h00;
    end else begin
      if (launch) begin
        pending <= 1'b1;
        req_data_q <= launch_data;
      end else if (resp_valid) begin
        pending <= 1'b0;
        result_q <= resp_data;
      end
    end
  end

  assign req_valid = pending;
  assign req_data = req_data_q;
  assign done = resp_valid;
  assign result = result_q;
endmodule
"#;

    let module = compile_pipeline_module(dut).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let rst = runner.port_bindings().input_with_width("rst", 1).unwrap();

    let mut launch = LaunchFixture {
        launch: None,
        payload: None,
    };
    let mut responder = ResponderFixture {
        req_valid: None,
        req_data: None,
        resp_valid: None,
        resp_data: None,
        scheduled_response: None,
    };
    let mut score = ScoreFixture {
        done: None,
        result: None,
        observed_results: Vec::new(),
    };
    let mut fixtures: Vec<&mut dyn CycleFixture> = vec![&mut launch, &mut responder, &mut score];
    runner.bind_fixtures(&mut fixtures).unwrap();
    runner
        .step_cycle_with_drive(&mut fixtures, |ctx| ctx.set_bool(&rst, true))
        .unwrap();
    for _ in 0..4 {
        runner.step_cycle(&mut fixtures).unwrap();
    }

    assert_eq!(score.observed_results, vec![0x3d]);
}

#[test]
fn fixture_runner_rejects_multiple_fixtures_driving_same_input() {
    struct BoolDriverFixture {
        handle: Option<xlsynth_vastly::InputPortHandle>,
        value: bool,
    }

    impl CycleFixture for BoolDriverFixture {
        fn bind(&mut self, ports: &xlsynth_vastly::PortBindings<'_>) -> xlsynth_vastly::Result<()> {
            self.handle = Some(ports.input_with_width("shared_in", 1)?);
            Ok(())
        }

        fn drive_cycle_inputs(
            &mut self,
            _cycle: u64,
            ctx: &mut DriveContext<'_>,
        ) -> xlsynth_vastly::Result<()> {
            ctx.set_bool(self.handle.as_ref().unwrap(), self.value)
        }
    }

    let dut = r#"
module m(
  input logic clk,
  input logic shared_in,
  output logic observed
);
  assign observed = shared_in;
endmodule
"#;

    let module = compile_pipeline_module(dut).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let mut fixture0 = BoolDriverFixture {
        handle: None,
        value: false,
    };
    let mut fixture1 = BoolDriverFixture {
        handle: None,
        value: true,
    };
    let mut fixtures: Vec<&mut dyn CycleFixture> = vec![&mut fixture0, &mut fixture1];
    runner.bind_fixtures(&mut fixtures).unwrap();

    let err = runner.step_cycle(&mut fixtures).unwrap_err();
    assert!(
        format!("{err:?}")
            .contains("input port `shared_in` was already driven earlier in the cycle")
    );
}

#[test]
fn fixture_runner_rejects_overlapping_packed_slice_drives() {
    struct PackedDriverFixture {
        handle: Option<xlsynth_vastly::InputPortHandle>,
        lane: u32,
        value: u64,
    }

    impl CycleFixture for PackedDriverFixture {
        fn bind(&mut self, ports: &xlsynth_vastly::PortBindings<'_>) -> xlsynth_vastly::Result<()> {
            self.handle = Some(ports.input_with_width("shared_bus", 16)?);
            Ok(())
        }

        fn drive_cycle_inputs(
            &mut self,
            _cycle: u64,
            ctx: &mut DriveContext<'_>,
        ) -> xlsynth_vastly::Result<()> {
            ctx.set_packed_u64(self.handle.as_ref().unwrap(), &[self.lane], self.value)
        }
    }

    let dut = r#"
module m(
  input logic clk,
  input logic [1:0][7:0] shared_bus,
  output logic [1:0][7:0] observed_bus
);
  assign observed_bus = shared_bus;
endmodule
"#;

    let module = compile_pipeline_module(dut).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let mut fixture0 = PackedDriverFixture {
        handle: None,
        lane: 0,
        value: 0x12,
    };
    let mut fixture1 = PackedDriverFixture {
        handle: None,
        lane: 0,
        value: 0x34,
    };
    let mut fixtures: Vec<&mut dyn CycleFixture> = vec![&mut fixture0, &mut fixture1];
    runner.bind_fixtures(&mut fixtures).unwrap();

    let err = runner.step_cycle(&mut fixtures).unwrap_err();
    assert!(
        format!("{err:?}")
            .contains("input port `shared_bus` slice [0] was already driven earlier in the cycle")
    );
}
