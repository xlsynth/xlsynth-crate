// SPDX-License-Identifier: Apache-2.0

use crate::solver_interface::Uf;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};
use xlsynth::IrValue;

use crate::solver_interface::{BitVec, Solver};
use xlsynth_pir::ir;

#[derive(Clone)]
pub struct IrTypedBitVec<'a, R> {
    pub ir_type: &'a ir::Type,
    pub bitvec: BitVec<R>,
}

impl<'a, R: std::fmt::Debug> std::fmt::Debug for IrTypedBitVec<'a, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IrTypedBitVec {{ ir_type: {:?}, bitvec: {:?} }}",
            self.ir_type, self.bitvec
        )
    }
}

#[derive(Debug, Clone)]
pub struct IrFn<'a> {
    pub fn_ref: &'a ir::Fn,
    // This is allowed to be None for IRs without invoke.
    pub pkg_ref: Option<&'a ir::Package>,
    pub fixed_implicit_activation: bool,
}

impl<'a> IrFn<'a> {
    pub fn new(fn_ref: &'a ir::Fn, pkg_ref: Option<&'a ir::Package>) -> Self {
        Self {
            fn_ref,
            pkg_ref: pkg_ref,
            fixed_implicit_activation: false,
        }
    }

    pub fn name(&self) -> &str {
        &self.fn_ref.name
    }

    pub fn params(&self) -> &'a [ir::Param] {
        &self.fn_ref.params
    }
}

#[derive(Debug, Clone)]
pub struct FnInputs<'a, R> {
    pub ir_fn: IrFn<'a>,
    pub inputs: HashMap<String, IrTypedBitVec<'a, R>>,
}

impl<'a, R> FnInputs<'a, R> {
    pub fn total_width(&self) -> usize {
        self.inputs.values().map(|b| b.bitvec.get_width()).sum()
    }

    pub fn total_free_width(&self) -> usize {
        if self.fixed_implicit_activation() {
            self.total_width() - 1
        } else {
            self.total_width()
        }
    }

    pub fn fixed_implicit_activation(&self) -> bool {
        self.ir_fn.fixed_implicit_activation
    }

    pub fn params(&self) -> &'a [ir::Param] {
        self.ir_fn.params()
    }

    pub fn params_len(&self) -> usize {
        self.params().len()
    }

    pub fn free_params_len(&self) -> usize {
        if self.fixed_implicit_activation() {
            self.params_len() - 2
        } else {
            self.params_len()
        }
    }

    pub fn free_params(&self) -> &'a [ir::Param] {
        if self.fixed_implicit_activation() {
            &self.params()[2..]
        } else {
            self.params()
        }
    }

    pub fn name(&self) -> &str {
        self.ir_fn.name()
    }

    pub fn get_fn(&self, name: &str) -> &'a ir::Fn {
        let pkg = self
            .ir_fn
            .pkg_ref
            .expect("fn lookup requires package context");
        pkg.get_fn(name)
            .unwrap_or_else(|| panic!("Function '{}' not found in package", name))
    }
}

#[derive(Clone)]
pub struct Assertion<'a, R> {
    pub active: BitVec<R>,
    pub message: &'a str,
    pub label: &'a str,
}

pub struct SmtFn<'a, R> {
    pub fn_ref: &'a ir::Fn,
    pub inputs: Vec<IrTypedBitVec<'a, R>>,
    pub output: IrTypedBitVec<'a, R>,
    pub assertions: Vec<Assertion<'a, R>>,
}

/// Semantics for handling `assert` statements when checking functional
/// equivalence.
///
/// Shorthand used in the formulas below:
/// • `r_l` – result of the **l**eft function
/// • `r_r` – result of the **r**ight function
/// • `s_l` – "success" flag of the left (`true` iff no assertion failed)
/// • `s_r` – "success" flag of the right (`true` iff no assertion failed)
///
/// For every variant we list
///  1. **Success condition** – when the equivalence checker should consider the
///     two functions *equivalent*.
///  2. **Failure condition** – negation of the success condition; if *any*
///     model satisfies this predicate, the checker must report a
///     counter-example.
#[derive(Debug, PartialEq, Clone, Copy, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssertionSemantics {
    /// Ignore all assertions.
    ///
    /// 1. Success: `r_l == r_r`
    /// 2. Failure: `r_l != r_r`
    Ignore,

    /// Both sides must succeed and produce the same result – they can **never**
    /// fail.
    ///
    /// 1. Success: `s_l ∧ s_r ∧ (r_l == r_r)`
    /// 2. Failure: `¬s_l ∨ ¬s_r ∨ (r_l != r_r)`
    Never,

    /// The two sides must fail in exactly the same way **or** both succeed with
    /// equal results.
    ///
    /// 1. Success: `(¬s_l ∧ ¬s_r) ∨ (s_l ∧ s_r ∧ (r_l == r_r))`
    /// 2. Failure: `(s_l ⊕ s_r) ∨ (s_l ∧ s_r ∧ (r_l != r_r))`
    Same,

    /// We *assume* both sides do not fail. In other words, we only check that
    /// if they do succeed, their results must be equal.
    ///
    /// 1. Success: `¬(s_l ∧ s_r) ∨ (r_l == r_r)`  (equivalently, `(s_l ∧ s_r) →
    ///    r_l == r_r`)
    /// 2. Failure: `s_l ∧ s_r ∧ (r_l != r_r)`
    Assume,

    /// If the left succeeds, the right must also succeed and match the
    /// result; if the left fails, the right is unconstrained.
    ///
    /// 1. Success: `¬s_l ∨ (s_r ∧ (r_l == r_r))`
    /// 2. Failure: `s_l ∧ (¬s_r ∨ (r_l != r_r))`
    Implies,
}

impl fmt::Display for AssertionSemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AssertionSemantics::Ignore => "ignore",
            AssertionSemantics::Never => "never",
            AssertionSemantics::Same => "same",
            AssertionSemantics::Assume => "assume",
            AssertionSemantics::Implies => "implies",
        };
        write!(f, "{}", s)
    }
}

impl std::str::FromStr for AssertionSemantics {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ignore" => Ok(AssertionSemantics::Ignore),
            "never" => Ok(AssertionSemantics::Never),
            "same" => Ok(AssertionSemantics::Same),
            "assume" => Ok(AssertionSemantics::Assume),
            "implies" => Ok(AssertionSemantics::Implies),
            _ => Err(format!("Invalid assertion semantics: {}", s)),
        }
    }
}

// Map param name -> allowed IrValues for domain constraints.
pub type ParamDomains = HashMap<String, Vec<IrValue>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UfSignature {
    pub arg_widths: Vec<usize>,
    pub ret_width: usize,
}

pub struct UfRegistry<S: Solver> {
    pub ufs: HashMap<String, Uf<S::Term>>,
}

impl<S: Solver> UfRegistry<S> {
    pub fn from_uf_signatures(
        solver: &mut S,
        uf_signatures: &HashMap<String, UfSignature>,
    ) -> Self {
        let mut ufs = HashMap::new();
        for (name, signature) in uf_signatures {
            let uf = solver
                .declare_fresh_uf(&name, &signature.arg_widths, signature.ret_width)
                .unwrap();
            ufs.insert(name.clone(), uf);
        }
        Self { ufs }
    }
}

pub struct ProverFn<'a> {
    pub ir_fn: &'a IrFn<'a>,
    pub domains: Option<ParamDomains>,
    pub uf_map: HashMap<String, String>,
}

// Result types

#[derive(Debug, PartialEq, Clone)]
pub struct AssertionViolation {
    pub message: String,
    pub label: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnInput {
    pub name: String,
    pub value: IrValue,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnOutput {
    pub value: IrValue,
    pub assertion_violation: Option<AssertionViolation>,
}

impl std::fmt::Display for FnOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(violation) = &self.assertion_violation {
            write!(
                f,
                "Value: {:?}, Assertion violation: {} (label: {})",
                self.value, violation.message, violation.label
            )
        } else {
            write!(f, "Value: {:?}", self.value)
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum EquivResult {
    Proved,
    Disproved {
        lhs_inputs: Vec<FnInput>,
        rhs_inputs: Vec<FnInput>,
        lhs_output: FnOutput,
        rhs_output: FnOutput,
    },
    ToolchainDisproved(String),
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuickCheckAssertionSemantics {
    /// Assertions are just dropped entirely
    Ignore,
    /// Prove that assertion conditions can never fire
    Never,
    /// Assume that assertion conditions hold to try to help complete the proof
    Assume,
}

impl fmt::Display for QuickCheckAssertionSemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            QuickCheckAssertionSemantics::Ignore => "ignore",
            QuickCheckAssertionSemantics::Never => "never",
            QuickCheckAssertionSemantics::Assume => "assume",
        };
        write!(f, "{}", s)
    }
}

/// Result of proving that a boolean-returning function is always `true`.
#[derive(Debug, Clone, PartialEq)]
pub enum BoolPropertyResult {
    /// The solver proved that the function returns `true` for **all** possible
    /// inputs (w.r.t. the chosen `assertion_semantics`).
    Proved,
    /// The solver found a counter-example – a concrete set of inputs for which
    /// the function does **not** return `true` (or violates the assertion
    /// semantics).
    Disproved {
        /// Concrete input values leading to failure. Kept in the same order as
        /// the function parameters after potential implicit-token handling.
        inputs: Vec<FnInput>,
        /// Concrete (possibly failing) output value observed for the
        /// counter-example.
        output: FnOutput,
    },
    /// External toolchain reported failure without a structured counterexample.
    ///
    /// This is used by the `ExternalProver` implementation where the
    /// external tools (e.g. `check_ir_equivalence_main`) do not provide a
    /// machine-readable counterexample for QuickCheck-style properties.
    ToolchainDisproved(String),
    /// Internal error encountered before invoking the prover (e.g. package
    /// instrumentation issues). Carries a human-readable explanation.
    Error(String),
}

#[derive(Debug, Clone)]
pub struct QuickCheckRunResult {
    pub name: String,
    pub duration: std::time::Duration,
    pub result: BoolPropertyResult,
}
