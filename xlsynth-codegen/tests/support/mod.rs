// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

#[cfg(feature = "iverilog-tests")]
use std::cell::RefCell;
use std::fmt;
use std::ops::Deref;
#[cfg(feature = "iverilog-tests")]
use std::rc::Rc;

use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::ir::{Package, PackageMember};
use xlsynth_test_helpers::rtl_sim::{
    Bindings, Interface, LogicValue, Port, StateSignal, identifier,
};

#[cfg(feature = "iverilog-tests")]
use xlsynth_test_helpers::rtl_sim::Icarus;

/// Keeps the independently specified IR interface beside the emitted source.
#[derive(Clone)]
pub struct TestRtl {
    source: String,
    interface: Interface,
    #[cfg(feature = "iverilog-tests")]
    simulator: Rc<RefCell<Option<Icarus>>>,
}

impl TestRtl {
    pub fn emit(package: &Package, options: &BlockCodegenOptions) -> Self {
        let source = emit_system_verilog(package, options)
            .unwrap()
            .system_verilog;
        let selected = options
            .top
            .as_deref()
            .or_else(|| package.top.as_ref().map(|p| p.0.as_str()));
        let (block, metadata) = package
            .members
            .iter()
            .find_map(|m| match m {
                PackageMember::Block { func, metadata }
                    if selected.is_none_or(|s| s == func.name) =>
                {
                    Some((func, metadata))
                }
                _ => None,
            })
            .expect("selected test block");
        let outputs = if metadata.output_names.len() == 1 {
            vec![block.get_node(block.ret_node_ref.unwrap()).ty.clone()]
        } else {
            let xlsynth_pir::ir::Type::Tuple(fields) =
                &block.get_node(block.ret_node_ref.unwrap()).ty
            else {
                panic!("multiple outputs require synthetic tuple");
            };
            fields.iter().map(|f| (**f).clone()).collect()
        };
        let interface = Interface {
            module: options
                .module_name
                .as_deref()
                .unwrap_or(&block.name)
                .to_owned(),
            inputs: block
                .params
                .iter()
                .map(|p| Port {
                    name: p.name.clone(),
                    width: p.ty.bit_count(),
                })
                .collect(),
            outputs: metadata
                .output_names
                .iter()
                .zip(outputs)
                .map(|(name, ty)| Port {
                    name: name.clone(),
                    width: ty.bit_count(),
                })
                .collect(),
            clock: metadata.clock_port_name.clone(),
            state: metadata
                .registers
                .iter()
                .filter(|r| r.ty.bit_count() != 0)
                .map(|r| StateSignal {
                    name: r.name.clone(),
                    width: r.ty.bit_count(),
                    expression: format!("dut.{}", identifier(&r.name).unwrap()),
                })
                .collect(),
        };
        Self {
            source,
            interface,
            #[cfg(feature = "iverilog-tests")]
            simulator: Rc::new(RefCell::new(None)),
        }
    }

    #[cfg(feature = "iverilog-tests")]
    pub fn prepare(&self) -> Result<Self, String> {
        if self.simulator.borrow().is_none() {
            *self.simulator.borrow_mut() =
                Some(Icarus::new(&self.source, self.interface.clone()).map_err(|e| e.to_string())?);
        }
        Ok(self.clone())
    }

    #[cfg(feature = "iverilog-tests")]
    pub fn evaluate(&self, inputs: &Bindings) -> Result<Bindings, String> {
        self.prepare()?;
        self.simulator
            .borrow_mut()
            .as_mut()
            .unwrap()
            .evaluate(inputs)
            .map(|s| s.outputs)
            .map_err(|e| e.to_string())
    }

    pub fn initial_state_x(&self) -> Bindings {
        self.interface
            .state
            .iter()
            .map(|s| (s.name.clone(), LogicValue::unknown(s.width)))
            .collect()
    }

    #[cfg(feature = "iverilog-tests")]
    pub fn cycles(
        &self,
        stimulus: &ClockedInputs,
        initial: &Bindings,
    ) -> Result<Vec<Bindings>, String> {
        self.prepare()?;
        let mut simulator = self.simulator.borrow_mut();
        let simulator = simulator.as_mut().unwrap();
        simulator.set_state(initial).map_err(|e| e.to_string())?;
        stimulus
            .cycles
            .iter()
            .map(|cycle| {
                simulator
                    .cycle(&cycle.inputs)
                    .map(|s| s.outputs)
                    .map_err(|e| e.to_string())
            })
            .collect()
    }
}

impl Deref for TestRtl {
    type Target = str;
    fn deref(&self) -> &str {
        &self.source
    }
}
impl fmt::Display for TestRtl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(f)
    }
}
impl fmt::Debug for TestRtl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(f)
    }
}
impl PartialEq for TestRtl {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}
impl PartialEq<&str> for TestRtl {
    fn eq(&self, other: &&str) -> bool {
        self.source == *other
    }
}
impl PartialEq<String> for TestRtl {
    fn eq(&self, other: &String) -> bool {
        self.source == *other
    }
}

pub struct ClockedInputs {
    pub cycles: Vec<CycleInputs>,
}
pub struct CycleInputs {
    pub inputs: Bindings,
}

#[cfg(feature = "iverilog-tests")]
pub fn run_icarus_cycles(
    design: &TestRtl,
    stimulus: &ClockedInputs,
    initial: &Bindings,
) -> Result<Vec<Bindings>, String> {
    design.cycles(stimulus, initial)
}
