// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::time::Instant;

use libfuzzer_sys::fuzz_target;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::aig_serdes::gate2ir;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOptions};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
use xlsynth_pir::ir::{
    self, ExtNaryAddArchitecture, ExtNaryAddTerm, FileTable, MemberType, Node, NodePayload, Package,
    PackageMember, Param, ParamId, Type,
};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_validate;
use xlsynth_prover::prover::types::EquivResult;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::ir_equiv::prove_ir_fn_equiv;
#[cfg(any(feature = "has-bitwuzla", feature = "has-boolector"))]
use xlsynth_prover::prover::types::{AssertionSemantics, ProverFn};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::solver::bitwuzla::{Bitwuzla, BitwuzlaOptions};
#[cfg(all(not(feature = "has-bitwuzla"), feature = "has-boolector"))]
use xlsynth_prover::prover::ir_equiv::prove_ir_fn_equiv;
#[cfg(all(not(feature = "has-bitwuzla"), feature = "has-boolector"))]
use xlsynth_prover::solver::boolector::{Boolector, BoolectorConfig};

const PACKAGE_NAME: &str = "sample";
const FUNCTION_NAME: &str = "main";
const MAX_RESULT_WIDTH: u8 = 10;
const MAX_OPERAND_COUNT: u8 = 5;
const MAX_PARAM_COUNT: u8 = 8;
const MAX_OPERAND_WIDTH: u8 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExtNaryAddFnSample {
    result_width: usize,
    arch: Option<ExtNaryAddArchitecture>,
    params: Vec<ParamSample>,
    terms: Vec<ExtNaryAddTermSample>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParamSample {
    width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExtNaryAddTermSample {
    source: OperandSourceSample,
    signed: bool,
    negated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OperandSourceSample {
    Param { param_index: usize },
    Literal { width: usize, value_bits: u16 },
}

#[cfg(feature = "has-bitwuzla")]
fn prove_exported_vs_gate_equiv(
    exported_fn: &ir::Fn,
    exported_pkg: &ir::Package,
    gate_fn: &ir::Fn,
    gate_pkg: &ir::Package,
) -> EquivResult {
    prove_ir_fn_equiv::<Bitwuzla>(
        &BitwuzlaOptions::new(),
        &ProverFn::new(exported_fn, Some(exported_pkg)),
        &ProverFn::new(gate_fn, Some(gate_pkg)),
        AssertionSemantics::Same,
        None,
        false,
    )
}

#[cfg(all(not(feature = "has-bitwuzla"), feature = "has-boolector"))]
fn prove_exported_vs_gate_equiv(
    exported_fn: &ir::Fn,
    exported_pkg: &ir::Package,
    gate_fn: &ir::Fn,
    gate_pkg: &ir::Package,
) -> EquivResult {
    prove_ir_fn_equiv::<Boolector>(
        &BoolectorConfig::new(),
        &ProverFn::new(exported_fn, Some(exported_pkg)),
        &ProverFn::new(gate_fn, Some(gate_pkg)),
        AssertionSemantics::Same,
        None,
        false,
    )
}

#[cfg(all(not(feature = "has-bitwuzla"), not(feature = "has-boolector")))]
fn prove_exported_vs_gate_equiv(
    _exported_fn: &ir::Fn,
    _exported_pkg: &ir::Package,
    _gate_fn: &ir::Fn,
    _gate_pkg: &ir::Package,
) -> EquivResult {
    panic!(
        "fuzz_ext_nary_add_gatify_equiv requires an in-process solver; \
         build with --features=with-bitwuzla-system, with-bitwuzla-built, \
         with-boolector-system, or with-boolector-built"
    )
}

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let sample = generate_ext_nary_add_fn_sample_without_zero_widths(data);
    let pkg = build_ext_nary_add_package(&sample);
    ir_validate::validate_package(&pkg)
        .unwrap_or_else(|e| panic!("generated ext_nary_add package should validate: {e:?}"));
    let ir_text = pkg.to_string();
    log::info!("generated ext_nary_add IR:\n{}", ir_text);
    let pir_fn = pkg.get_top_fn().expect("generated package should have a top fn");

    let gatify_start = Instant::now();
    let gatify_output = ir2gate::gatify(
        pir_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            array_index_lowering_strategy: Default::default(),
        },
    )
    .expect("gatify should succeed for generated ext_nary_add IR");
    log::info!("gatify completed in {:?}", gatify_start.elapsed());

    // The in-process prover works over PIR functions, so we prove equivalence
    // against the exported/desugared XLS IR form of the same package.
    let export_start = Instant::now();
    let exported_ir_text =
        emit_package_as_xls_ir_text(&pkg).expect("exporting generated ext_nary_add IR should work");
    log::info!(
        "emit_package_as_xls_ir_text completed in {:?} ({} bytes)",
        export_start.elapsed(),
        exported_ir_text.len()
    );
    let gate_ir_start = Instant::now();
    let gate_ir_text = gate2ir::gate_fn_to_xlsynth_ir(
        &gatify_output.gate_fn,
        FUNCTION_NAME,
        &pir_fn.get_type(),
    )
    .expect("gate_fn_to_xlsynth_ir should succeed")
    .to_string();
    log::info!(
        "gate_fn_to_xlsynth_ir completed in {:?} ({} bytes)",
        gate_ir_start.elapsed(),
        gate_ir_text.len()
    );

    let exported_pkg = Parser::new(&exported_ir_text)
        .parse_and_validate_package()
        .unwrap_or_else(|e| {
            panic!("exported ext_nary_add IR should parse:\n{exported_ir_text}\nerror: {e}")
        });
    let gate_pkg = Parser::new(&gate_ir_text)
        .parse_and_validate_package()
        .unwrap_or_else(|e| panic!("gate IR should parse:\n{gate_ir_text}\nerror: {e}"));
    let exported_fn = exported_pkg
        .get_top_fn()
        .expect("exported package should have a top fn");
    let gate_fn = gate_pkg
        .get_top_fn()
        .expect("gate package should have a top fn");

    let prove_start = Instant::now();
    let equiv_result =
        prove_exported_vs_gate_equiv(exported_fn, &exported_pkg, gate_fn, &gate_pkg);

    match equiv_result {
        EquivResult::Proved => {}
        EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } => panic!(
            "gatify equivalence disproved for generated ext_nary_add IR:\n{ir_text}\nexported:\n{exported_ir_text}\ngate:\n{gate_ir_text}\nlhs_inputs={lhs_inputs:?}\nrhs_inputs={rhs_inputs:?}\nlhs_output={lhs_output:?}\nrhs_output={rhs_output:?}"
        ),
        EquivResult::Interrupted => panic!(
            "in-process formal equivalence was interrupted unexpectedly for generated ext_nary_add IR:\n{ir_text}"
        ),
        EquivResult::ToolchainDisproved(msg) | EquivResult::Error(msg) => panic!(
            "in-process formal equivalence failed for generated ext_nary_add IR:\n{ir_text}\nmessage: {msg}"
        ),
    }
    log::info!("in-process equivalence completed in {:?}", prove_start.elapsed());
});

/// Generates a typed `ext_nary_add` fuzz sample from deterministic entropy
/// derived from the raw fuzz input bytes, excluding zero-width types.
fn generate_ext_nary_add_fn_sample_without_zero_widths(data: &[u8]) -> ExtNaryAddFnSample {
    let mut seed = [0u8; 32];
    seed.copy_from_slice(blake3::hash(data).as_bytes());
    let mut rng = StdRng::from_seed(seed);

    let result_width = usize::from(rng.gen_range(1..=MAX_RESULT_WIDTH));
    let operand_count = usize::from(rng.gen_range(0..=MAX_OPERAND_COUNT));
    let arch = match rng.gen_range(0..=3) {
        0 => None,
        1 => Some(ExtNaryAddArchitecture::RippleCarry),
        2 => Some(ExtNaryAddArchitecture::BrentKung),
        _ => Some(ExtNaryAddArchitecture::KoggeStone),
    };

    let mut params = Vec::new();
    let mut terms = Vec::with_capacity(operand_count);
    for _ in 0..operand_count {
        let source_choice = rng.gen_range(0..10);
        let source = if source_choice < 3 {
            make_literal_operand_source(&mut rng)
        } else if source_choice == 3 && !params.is_empty() {
            OperandSourceSample::Param {
                param_index: rng.gen_range(0..params.len()),
            }
        } else if params.len() < usize::from(MAX_PARAM_COUNT) {
            let param_index = params.len();
            params.push(ParamSample {
                width: usize::from(rng.gen_range(1..=MAX_OPERAND_WIDTH)),
            });
            OperandSourceSample::Param { param_index }
        } else if !params.is_empty() {
            OperandSourceSample::Param {
                param_index: rng.gen_range(0..params.len()),
            }
        } else {
            make_literal_operand_source(&mut rng)
        };
        terms.push(ExtNaryAddTermSample {
            source,
            signed: rng.r#gen::<bool>(),
            negated: rng.r#gen::<bool>(),
        });
    }

    ExtNaryAddFnSample {
        result_width,
        arch,
        params,
        terms,
    }
}

fn make_literal_operand_source<R: Rng>(rng: &mut R) -> OperandSourceSample {
    let width = usize::from(rng.gen_range(1..=MAX_OPERAND_WIDTH));
    OperandSourceSample::Literal {
        width,
        value_bits: rng.r#gen::<u16>() & width_mask_u16(width),
    }
}

/// Builds a generated sample as a one-function PIR package.
fn build_ext_nary_add_package(sample: &ExtNaryAddFnSample) -> Package {
    let params = sample
        .params
        .iter()
        .enumerate()
        .map(|(i, param)| Param {
            name: format!("p{i}"),
            ty: Type::Bits(param.width),
            id: ParamId::new(i + 1),
        })
        .collect::<Vec<_>>();

    let mut nodes = Vec::new();
    nodes.push(Node {
        text_id: 0,
        name: None,
        ty: Type::nil(),
        payload: NodePayload::Nil,
        pos: None,
    });
    for param in &params {
        nodes.push(Node {
            text_id: param.id.get_wrapped_id(),
            name: Some(param.name.clone()),
            ty: param.ty.clone(),
            payload: NodePayload::GetParam(param.id),
            pos: None,
        });
    }

    let mut next_text_id = params.len() + 1;
    let mut terms = Vec::with_capacity(sample.terms.len());
    for (term_index, term) in sample.terms.iter().enumerate() {
        let operand = match term.source {
            OperandSourceSample::Param { param_index } => ir::NodeRef {
                index: param_index + 1,
            },
            OperandSourceSample::Literal { width, value_bits } => {
                let literal_ref = ir::NodeRef { index: nodes.len() };
                nodes.push(Node {
                    text_id: next_text_id,
                    name: Some(format!("lit_{term_index}")),
                    ty: Type::Bits(width),
                    payload: NodePayload::Literal(make_bits_value(width, u64::from(value_bits))),
                    pos: None,
                });
                next_text_id += 1;
                literal_ref
            }
        };
        terms.push(ExtNaryAddTerm {
            operand,
            signed: term.signed,
            negated: term.negated,
        });
    }

    let ret_node_ref = ir::NodeRef { index: nodes.len() };
    nodes.push(Node {
        text_id: next_text_id,
        name: Some("r".to_string()),
        ty: Type::Bits(sample.result_width),
        payload: NodePayload::ExtNaryAdd {
            terms,
            arch: sample.arch,
        },
        pos: None,
    });

    let function = ir::Fn {
        name: FUNCTION_NAME.to_string(),
        params,
        ret_ty: Type::Bits(sample.result_width),
        nodes,
        ret_node_ref: Some(ret_node_ref),
        outer_attrs: Vec::new(),
        inner_attrs: Vec::new(),
    };

    Package {
        name: PACKAGE_NAME.to_string(),
        file_table: FileTable::new(),
        members: vec![PackageMember::Function(function)],
        top: Some((FUNCTION_NAME.to_string(), MemberType::Function)),
    }
}

fn make_bits_value(width: usize, raw_value: u64) -> IrValue {
    let bits = IrBits::make_ubits(width, raw_value & width_mask_u64(width))
        .expect("masked literal bits must fit the requested width");
    IrValue::from_bits(&bits)
}

fn width_mask_u16(width: usize) -> u16 {
    if width == 0 {
        0
    } else if width >= u16::BITS as usize {
        u16::MAX
    } else {
        (1u16 << width) - 1
    }
}

fn width_mask_u64(width: usize) -> u64 {
    if width == 0 {
        0
    } else if width >= u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}
