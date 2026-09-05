// SPDX-License-Identifier: Apache-2.0

//! External RTL simulation with caller-supplied interfaces, never an SV parser.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::iverilog::{IcarusToolchain, required_iverilog_toolchain};
use tempfile::TempDir;
pub use xlsynth::external_tool::run_checked;
use xlsynth::external_tool::{ToolError, kill_process_group, set_process_group};
use xlsynth_pir::IrBits;

pub type Bindings = BTreeMap<String, LogicValue>;
type WriteRequest = (String, Sender<std::io::Result<()>>);

/// A four-state bit string for transport and comparison, not RTL evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicValue(String);

impl LogicValue {
    pub fn from_bits(bits: &IrBits) -> Self {
        Self(
            (0..bits.get_bit_count())
                .rev()
                .map(|i| if bits.get_bit(i).unwrap() { '1' } else { '0' })
                .collect(),
        )
    }

    pub fn from_u64(width: u32, value: u64) -> Self {
        Self::from_bits(&IrBits::make_ubits(width as usize, value).expect("value fits width"))
    }

    pub fn unknown(width: usize) -> Self {
        Self("x".repeat(width))
    }
    pub fn width(&self) -> usize {
        self.0.len()
    }
    pub fn is_all_known_01(&self) -> bool {
        self.0.bytes().all(|b| b == b'0' || b == b'1')
    }
    pub fn has_unknown(&self) -> bool {
        !self.is_all_known_01()
    }
    pub fn to_bit_string_msb_first(&self) -> String {
        self.0.clone()
    }

    pub fn to_bits(&self) -> Result<IrBits, String> {
        if !self.is_all_known_01() {
            return Err(format!("RTL contains X/Z: {}", self.0));
        }
        Ok(IrBits::from_lsb_is_0(
            &self.0.bytes().rev().map(|b| b == b'1').collect::<Vec<_>>(),
        ))
    }

    pub fn to_u64_if_known(&self) -> Option<u64> {
        if self.width() > 64 || !self.is_all_known_01() {
            return None;
        }
        if self.0.is_empty() {
            Some(0)
        } else {
            u64::from_str_radix(&self.0, 2).ok()
        }
    }

    pub fn parse_binary(text: &str, width: usize) -> Result<Self, String> {
        let text = text.to_ascii_lowercase();
        if text.len() != width || !text.bytes().all(|b| matches!(b, b'0' | b'1' | b'x' | b'z')) {
            return Err(format!("expected {width} binary digits, got `{text}`"));
        }
        Ok(Self(text))
    }
}

#[derive(Clone, Debug)]
pub struct Port {
    pub name: String,
    pub width: usize,
}

/// The expression is a testbench lvalue (e.g. `dut.r`) supplied by the IR
/// adapter.
#[derive(Clone, Debug)]
pub struct StateSignal {
    pub name: String,
    pub width: usize,
    pub expression: String,
}

#[derive(Clone, Debug)]
pub struct Interface {
    pub module: String,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    pub clock: Option<String>,
    pub state: Vec<StateSignal>,
}

#[derive(Debug)]
pub struct Snapshot {
    pub outputs: Bindings,
    pub state: Bindings,
}

/// One compiled design and persistent vvp process; vectors do not recompile
/// RTL.
pub struct Icarus {
    interface: Interface,
    child: Child,
    command_sender: Option<Sender<WriteRequest>>,
    writer: Option<JoinHandle<()>>,
    lines: Receiver<String>,
    reader: Option<JoinHandle<()>>,
    directory: TempDir,
    timeout: Duration,
    response_marker: String,
}

impl Icarus {
    pub fn new(source: &str, interface: Interface) -> Result<Self, ToolError> {
        Self::with_toolchain(source, interface, required_iverilog_toolchain()?)
    }

    /// Compiles once and starts a persistent session using explicit tools.
    pub fn with_toolchain(
        source: &str,
        interface: Interface,
        tools: &IcarusToolchain,
    ) -> Result<Self, ToolError> {
        let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
        let response_marker = session_marker();
        let testbench = render_testbench(&interface, &response_marker)?;
        std::fs::write(directory.path().join("dut.sv"), source).map_err(|e| e.to_string())?;
        std::fs::write(directory.path().join("tb.sv"), &testbench).map_err(|e| e.to_string())?;
        let timeout = Duration::from_secs(60);
        let executable = tools
            .compile(
                directory.path(),
                &["dut.sv", "tb.sv"],
                "rtl_oracle_tb",
                &[],
                timeout,
            )
            .map_err(|e| e.with_context(format!("RTL:\n{source}\nTestbench:\n{testbench}")))?;
        let stderr = std::fs::File::create(directory.path().join("vvp.stderr"))
            .map_err(|e| e.to_string())?;
        let mut command = Command::new(tools.vvp_path());
        command
            .current_dir(directory.path())
            .arg(executable)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr);
        set_process_group(&mut command);
        let mut child = command.spawn().map_err(|e| ToolError::spawn("vvp", e))?;
        let mut stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (command_sender, commands) = mpsc::channel::<WriteRequest>();
        let writer = std::thread::spawn(move || {
            for (command, response) in commands {
                let result = stdin
                    .write_all(command.as_bytes())
                    .and_then(|_| stdin.flush());
                let failed = result.is_err();
                let _ = response.send(result);
                if failed {
                    break;
                }
            }
        });
        let (sender, lines) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(Self {
            interface,
            child,
            command_sender: Some(command_sender),
            writer: Some(writer),
            lines,
            reader: Some(reader),
            directory,
            timeout,
            response_marker,
        })
    }

    /// Drives inputs and settles combinational logic without a clock edge.
    pub fn evaluate(&mut self, inputs: &Bindings) -> Result<Snapshot, ToolError> {
        self.exchange(0, inputs)
    }

    /// Drives inputs, raises the clock, and samples after nonblocking updates.
    pub fn cycle(&mut self, inputs: &Bindings) -> Result<Snapshot, ToolError> {
        if self.interface.clock.is_none() {
            return Err("cycle requires a clock port".into());
        }
        self.exchange(1, inputs)
    }

    /// Initializes state explicitly, including registers without reset.
    pub fn set_state(&mut self, state: &Bindings) -> Result<(), ToolError> {
        let fields: Vec<_> = self
            .interface
            .state
            .iter()
            .map(|s| (s.name.as_str(), s.width))
            .collect();
        let command = render_command(2, &fields, state)?;
        let deadline = self.deadline();
        self.send_command(&command, deadline)?;
        self.response("ACK", deadline).map(|_| ())
    }

    fn exchange(&mut self, op: u8, inputs: &Bindings) -> Result<Snapshot, ToolError> {
        let fields: Vec<_> = self
            .interface
            .inputs
            .iter()
            .filter(|p| p.width != 0)
            .map(|p| (p.name.as_str(), p.width))
            .collect();
        let command = render_command(op, &fields, inputs)?;
        let deadline = self.deadline();
        self.send_command(&command, deadline)?;
        let response = self.response("RESULT", deadline)?;
        let mut words = response.split_whitespace().skip(1);
        let mut outputs = Bindings::new();
        let mut state = Bindings::new();
        for port in self.interface.outputs.iter().filter(|p| p.width != 0) {
            outputs.insert(
                port.name.clone(),
                LogicValue::parse_binary(words.next().ok_or("missing output")?, port.width)?,
            );
        }
        for signal in &self.interface.state {
            state.insert(
                signal.name.clone(),
                LogicValue::parse_binary(words.next().ok_or("missing state")?, signal.width)?,
            );
        }
        if words.next().is_some() {
            return Err(format!("extra simulator output: {response}").into());
        }
        Ok(Snapshot { outputs, state })
    }

    /// Reports an already-exited child before cleanup can change its status.
    fn process_failure(&mut self, message: String) -> ToolError {
        let status = self.child.try_wait().ok().flatten();
        kill_process_group(&mut self.child);
        let stderr =
            std::fs::read_to_string(self.directory.path().join("vvp.stderr")).unwrap_or_default();
        match status {
            Some(status) if !status.success() => {
                ToolError::from_exit_status("vvp", status, "", &stderr).with_context(message)
            }
            _ => ToolError::failure(format!("{message}\n{stderr}")),
        }
    }

    fn deadline(&self) -> Instant {
        Instant::now() + self.timeout
    }

    /// A writer thread lets the same deadline cover a vvp process that stops
    /// consuming stdin; killing the child unblocks the writer's pipe.
    fn send_command(&mut self, command: &str, deadline: Instant) -> Result<(), ToolError> {
        let (sender, response) = mpsc::channel();
        self.command_sender
            .as_ref()
            .ok_or_else(|| ToolError::failure("vvp command writer is unavailable"))?
            .send((command.to_owned(), sender))
            .map_err(|_| self.process_failure("vvp command writer stopped".into()))?;
        if Instant::now() >= deadline {
            return Err(self.timeout_failure("writing vvp command".into()));
        }
        match response.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(self.process_failure(format!("write vvp command: {error}"))),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                Err(self.timeout_failure("writing vvp command".into()))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(self.process_failure("vvp command writer closed".into()))
            }
        }
    }

    fn response(&mut self, kind: &str, deadline: Instant) -> Result<String, ToolError> {
        let prefix = format!("{}_{}", self.response_marker, kind);
        let mut diagnostics = String::new();
        loop {
            if Instant::now() >= deadline {
                return Err(self.timeout_failure(format!("waiting for {prefix}\n{diagnostics}")));
            }
            match self
                .lines
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                Ok(line) if line == prefix || line.starts_with(&format!("{prefix} ")) => {
                    return Ok(line);
                }
                Ok(line) => {
                    if diagnostics.len() < 8192 {
                        diagnostics.push_str(&line);
                        diagnostics.push('\n');
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    return Err(
                        self.timeout_failure(format!("waiting for {prefix}\n{diagnostics}"))
                    );
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(self.process_failure(format!(
                        "vvp closed output waiting for {prefix}\n{diagnostics}"
                    )));
                }
            }
        }
    }

    fn timeout_failure(&mut self, context: String) -> ToolError {
        kill_process_group(&mut self.child);
        let stderr =
            std::fs::read_to_string(self.directory.path().join("vvp.stderr")).unwrap_or_default();
        ToolError::timeout("vvp", self.timeout).with_context(format!("{context}\n{stderr}"))
    }
}

impl Drop for Icarus {
    fn drop(&mut self) {
        kill_process_group(&mut self.child);
        self.command_sender.take();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

fn render_command(op: u8, fields: &[(&str, usize)], values: &Bindings) -> Result<String, String> {
    if fields.len() != values.len() {
        return Err("binding count does not match interface".into());
    }
    let mut result = op.to_string();
    for (name, width) in fields {
        let value = values
            .get(*name)
            .ok_or_else(|| format!("missing `{name}`"))?;
        if value.width() != *width {
            return Err(format!(
                "wrong width for `{name}`: {} vs {width}",
                value.width()
            ));
        }
        result.push(' ');
        result.push_str(&value.0);
    }
    result.push('\n');
    Ok(result)
}

/// Escaped SV identifiers avoid collisions with testbench-local names/keywords.
pub fn identifier(name: &str) -> Result<String, String> {
    if name.is_empty() || name.chars().any(char::is_whitespace) || name.contains('\\') {
        return Err(format!("unsupported identifier `{name}`"));
    }
    Ok(format!("\\{name} "))
}

fn render_testbench(interface: &Interface, response_marker: &str) -> Result<String, String> {
    let mut result = String::from(
        "`timescale 1ns/1ps\nmodule rtl_oracle_tb;\ninteger op, status;\nreg clock = 0;\n",
    );
    let mut connections = Vec::new();
    if let Some(clock) = &interface.clock {
        connections.push(format!(".{}(clock)", identifier(clock)?));
    }
    for (i, port) in interface
        .inputs
        .iter()
        .enumerate()
        .filter(|(_, p)| p.width != 0)
    {
        result.push_str(&format!("reg [{}:0] input_{i} = 0;\n", port.width - 1));
        connections.push(format!(".{}(input_{i})", identifier(&port.name)?));
    }
    for (i, port) in interface
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, p)| p.width != 0)
    {
        result.push_str(&format!("wire [{}:0] output_{i};\n", port.width - 1));
        connections.push(format!(".{}(output_{i})", identifier(&port.name)?));
    }
    for (i, signal) in interface.state.iter().enumerate() {
        if signal.width == 0 {
            return Err("omit zero-bit state from simulator interface".into());
        }
        result.push_str(&format!("reg [{}:0] state_{i};\n", signal.width - 1));
    }
    result.push_str(&format!("{} dut ({});\ninitial begin\nforever begin\nstatus = $fscanf(32'h80000000, \"%d\", op);\nif (status != 1) $finish;\nif (op == 2) begin\n", identifier(&interface.module)?, connections.join(", ")));
    for (i, signal) in interface.state.iter().enumerate() {
        result.push_str(&format!("status = $fscanf(32'h80000000, \"%b\", state_{i});\nif (status != 1) $fatal(1, \"missing state\");\n{} = state_{i};\n", signal.expression));
    }
    result.push_str(&format!(
        "#1; $display(\"\\n{response_marker}_ACK\"); $fflush();\nend else begin\n"
    ));
    for (i, _) in interface
        .inputs
        .iter()
        .enumerate()
        .filter(|(_, p)| p.width != 0)
    {
        result.push_str(&format!("status = $fscanf(32'h80000000, \"%b\", input_{i});\nif (status != 1) $fatal(1, \"missing input\");\n"));
    }
    result.push_str("#1;\nif (op == 1) begin clock = 1; #1; end\n");
    let mut observed: Vec<String> = interface
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, p)| p.width != 0)
        .map(|(i, _)| format!("output_{i}"))
        .collect();
    observed.extend(interface.state.iter().map(|s| s.expression.clone()));
    result.push_str(&format!(
        "$display(\"\\n{response_marker}_RESULT{}\"{});\n$fflush();\nclock = 0; #1;\nend\nend\nend\nendmodule\n",
        " %b".repeat(observed.len()),
        if observed.is_empty() {
            String::new()
        } else {
            format!(", {}", observed.join(", "))
        }
    ));
    Ok(result)
}

/// Per-session text prevents DUT diagnostics from impersonating responses.
fn session_marker() -> String {
    static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);
    let serial = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("RTL_ORACLE_{}_{}_{}", std::process::id(), nanos, serial)
}
