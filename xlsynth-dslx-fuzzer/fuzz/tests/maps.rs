// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use xlsynth_dslx_fuzzer::Sample;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, HelperModule, ValueRef, Width};
use xlsynth_dslx_fuzzer::compound::{ArrayOp, CompoundOp};
use xlsynth_dslx_fuzzer::constexpr::{ComputedWidth, ConstExpr, ConstInput};
use xlsynth_dslx_fuzzer::map_fragment::{MapBody, MapFragment};
use xlsynth_dslx_fuzzer_fuzz::prove;
use xlsynth_pir::IrBits;
use xlsynth_prover::prover::types::EquivResult;

/// Maps signed or unsigned parametric elements, with distinct capture values.
fn map_helper(signed: bool, mapper: MapBody, length: usize) -> Result<Body, String> {
    let mut body = Body::new(BitsType {
        signed,
        width: Width::Parametric,
    })?;
    let difference = body.push(Fragment::Binary {
        op: BinaryOp::Sub,
        lhs: ValueRef::X,
        rhs: ValueRef::Y,
    })?;
    body.push(Fragment::Compound {
        op: CompoundOp::Map(MapFragment::new(mapper, length)?),
        args: [ValueRef::X, ValueRef::Y, difference],
    })?;
    Ok(body)
}

#[test]
fn specialized_maps_prove_against_elementwise_reference() -> Result<(), String> {
    let mappers = MapBody::BUILTINS.into_iter().chain([
        MapBody::IsEven,
        MapBody::LocalIsEven,
        MapBody::Capture(BinaryOp::Sub),
        MapBody::TwoCaptures(BinaryOp::Add, BinaryOp::Xor),
    ]);
    for (index, mapper) in mappers.enumerate() {
        let length = 1 + index % MapFragment::MAX_LENGTH;
        let helper = HelperModule::new(
            map_helper(false, mapper, length)?,
            map_helper(true, mapper, length)?,
        )?;
        let mut lanes = Vec::new();
        for signed in [false, true] {
            for width in [1, 65] {
                let mut lane = Body::new(BitsType {
                    signed,
                    width: Width::Concrete(width),
                })?;
                lane.push(Fragment::Call {
                    module: 0,
                    args: [ValueRef::X, ValueRef::Y],
                    explicit_parametric: false,
                })?;
                lanes.push(lane);
            }
        }
        let sample = Sample::new(vec![helper], lanes)?;
        let prepared = sample
            .prepare()
            .map_err(|error| format!("{mapper:?}: {error}\n{:?}", sample.sources()))?;
        let result = prove(&prepared)?;
        assert!(
            matches!(result, EquivResult::Proved),
            "{mapper:?} {result:?}"
        );
    }
    Ok(())
}

#[test]
fn computed_width_lambda_and_local_enumerate_round_trip() -> Result<(), String> {
    let dimension = Body::new(BitsType {
        signed: false,
        width: Width::Concrete(32),
    })?;
    let seven = IrBits::from_lsb_fn(32, |index| index < 3);
    let expression = ConstExpr::new(
        dimension,
        [
            ConstInput::Literal(seven.clone()),
            ConstInput::Literal(seven),
        ],
    )?;
    let mut lane = Body::new(BitsType {
        signed: false,
        width: Width::Concrete(7),
    })?;
    lane.set_computed_width(ComputedWidth {
        expression: Box::new(expression),
        assert_value: true,
    })?;
    let operators = [
        CompoundOp::Map(MapFragment::new(
            MapBody::TwoCaptures(BinaryOp::Sub, BinaryOp::Xor),
            2,
        )?),
        CompoundOp::Array(ArrayOp::LocalEnumerate),
    ];
    for op in operators {
        let mut lane = lane.clone();
        lane.push(Fragment::Compound {
            op,
            args: [ValueRef::X, ValueRef::X, ValueRef::Y],
        })?;
        let prepared = Sample::new(vec![], vec![lane])?.prepare()?;
        let result = prove(&prepared)?;
        assert!(matches!(result, EquivResult::Proved), "{op:?}: {result:?}");
    }
    Ok(())
}
