// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig_serdes::gate2ir;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r_fuzz::fuzz_bitwuzla_options;
use xlsynth_pir::ir::NodeRef;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::ir_equiv::prove_ir_fn_equiv;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivResult, ProverFn};
use xlsynth_prover::solver::bitwuzla::Bitwuzla;

/// Shapes of arithmetic inputs and product boundaries exercised by the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArithmeticScenario {
    AddOnly,
    SingleUseProduct,
    SingleUseConstantProduct,
    SharedProduct,
    SlicedProduct,
    NarrowProduct,
}

// Focused generation reaches both add-only rows and deferred product rows.
// Shared, sliced and previously truncated products remain resolved boundaries.
fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();
    let byte = |i: usize| data.get(i).copied().unwrap_or(0) as usize;
    let width = 1 + byte(0) % 12;
    let input_width = 1 + byte(1) % 8;
    let scenario = match byte(2) % 6 {
        0 => ArithmeticScenario::AddOnly,
        1 => ArithmeticScenario::SingleUseProduct,
        2 => ArithmeticScenario::SingleUseConstantProduct,
        3 => ArithmeticScenario::SharedProduct,
        4 => ArithmeticScenario::SlicedProduct,
        // The remainder is 5 after reducing modulo 6.
        _ => ArithmeticScenario::NarrowProduct,
    };
    let product_width = if scenario == ArithmeticScenario::NarrowProduct {
        width.saturating_sub(1).max(1)
    } else {
        width + byte(3) % 3
    };
    let count = 3 + byte(4) % 4;
    let arch = match byte(5) % 3 {
        0 => AdderMapping::BrentKung,
        1 => AdderMapping::RippleCarry,
        _ => AdderMapping::KoggeStone,
    };
    let multiply = if byte(6) & 1 == 0 { "umul" } else { "smul" };
    let mut body = vec![
        format!(
            " k: bits[{input_width}] = literal(value={}, id=5)",
            byte(7) % (1 << input_width)
        ),
        format!(
            " p: bits[{product_width}] = {multiply}(a, {}, id=6)",
            if scenario == ArithmeticScenario::SingleUseConstantProduct {
                "k"
            } else {
                "b"
            }
        ),
    ];
    let mut id = 7;
    let mut late = "c".to_string();
    for _ in 0..byte(8) % 9 {
        body.push(format!(" v{id}: bits[{width}] = reverse({late}, id={id})"));
        let previous = id;
        id += 1;
        body.push(format!(
            " v{id}: bits[{width}] = add(v{previous}, d, id={id})"
        ));
        late = format!("v{id}");
        id += 1;
    }
    let slice_id = (scenario == ArithmeticScenario::SlicedProduct).then_some(id);
    let product_term = if scenario == ArithmeticScenario::SlicedProduct {
        body.push(format!(
            " slice: bits[{width}] = bit_slice(p, start=0, width={width}, id={id})"
        ));
        id += 1;
        "slice"
    } else {
        "p"
    };
    let mut terms = vec![
        if scenario == ArithmeticScenario::AddOnly {
            "c"
        } else {
            product_term
        }
        .to_string(),
        late,
        "d".to_string(),
    ];
    for i in 3..count {
        terms.push(
            if scenario == ArithmeticScenario::SharedProduct && i == 3 {
                "p"
            } else {
                if byte(9 + i) & 1 == 0 { "c" } else { "d" }
            }
            .to_string(),
        );
    }
    // Force an actual shared-product negative control, even for count == 3.
    if scenario == ArithmeticScenario::SharedProduct {
        terms.push("p".to_string());
    }
    let signed: Vec<_> = (0..terms.len())
        .map(|i| (byte(16 + i) & 1 != 0).to_string())
        .collect();
    let negative: Vec<_> = (0..terms.len())
        .map(|i| (byte(24 + i) & 1 != 0).to_string())
        .collect();
    body.push(format!(
        " ret r: bits[{width}] = ext_nary_add({}, signed=[{}], negated=[{}], id={id})",
        terms.join(", "),
        signed.join(", "),
        negative.join(", ")
    ));
    let text = format!(
        "package rows\ntop fn f(a: bits[{input_width}] id=1, b: bits[{input_width}] id=2, c: bits[{width}] id=3, d: bits[{width}] id=4) -> bits[{width}] {{\n{}\n}}",
        body.join("\n")
    );
    let package = Parser::new(&text).parse_and_validate_package().unwrap();
    let source = package.get_top_fn().unwrap();
    let mapped = gatify_prepared_fn(
        source,
        GatifyOptions {
            fold: true,
            hash: byte(9) & 1 == 0,
            enable_rewrite_nary_add: true,
            adder_mapping: arch,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let product = NodeRef {
        index: source.nodes.iter().position(|n| n.text_id == 6).unwrap(),
    };
    let expected_product_entry = match scenario {
        // The product is unused in the add-only case.
        ArithmeticScenario::AddOnly => None,
        ArithmeticScenario::SingleUseProduct | ArithmeticScenario::SingleUseConstantProduct => {
            Some(false)
        }
        ArithmeticScenario::SharedProduct => Some(true),
        // A partially demanded product is omitted even when it is resolved.
        ArithmeticScenario::SlicedProduct => (product_width == width).then_some(true),
        // A one-bit sum cannot produce a narrower product.
        ArithmeticScenario::NarrowProduct => Some(product_width < width),
    };
    if let Some(expected) = expected_product_entry {
        assert_eq!(
            mapped.lowering_map.contains_key(&product),
            expected,
            "unexpected product lowering-map entry for {scenario:?}: {text}"
        );
    }
    if let Some(slice_id) = slice_id {
        let slice = NodeRef {
            index: source
                .nodes
                .iter()
                .position(|n| n.text_id == slice_id)
                .unwrap(),
        };
        // The slice is fully demanded, but the product is omitted from the
        // public lowering map when its upper bits are not demanded. Lowering
        // the slice still requires the intervening product to be resolved.
        assert!(
            mapped.lowering_map.contains_key(&slice),
            "expected resolved product slice: {text}"
        );
    }
    let lifted = gate2ir::gate_fn_to_pir(&mapped.gate_fn, "f", &source.get_type()).unwrap();
    match prove_ir_fn_equiv::<Bitwuzla>(
        &fuzz_bitwuzla_options(),
        &ProverFn::new(source, None),
        &ProverFn::new(lifted.get_top_fn().unwrap(), None),
        AssertionSemantics::Same,
        None,
        false,
    ) {
        EquivResult::Proved => {}
        // Bounded solver work may time out; an inconclusive result is no proof.
        EquivResult::Inconclusive(message) => log::debug!("inconclusive: {message}"),
        other => panic!("common arithmetic rows mismatch: {other:?}\n{text}"),
    }
});
