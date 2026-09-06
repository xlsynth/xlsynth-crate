// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir;
use xlsynth_pir::{FnBuilder, IrValue};

/// Adds a true-valued property whose return depends on an invocation of `top`.
pub(crate) fn add_assertion_dependency_property(
    package: &mut ir::Package,
    top_name: &str,
    prefix: &str,
) -> Result<String, String> {
    let top = package
        .get_fn(top_name)
        .ok_or_else(|| format!("IR function '{}' not found", top_name))?;
    let property_name = format!("__{prefix}_property__{top_name}");
    if package.get_fn(&property_name).is_some() {
        return Ok(property_name);
    }

    let mut builder = FnBuilder::new(&property_name);
    let args = top
        .param_nodes()
        .map(|param| builder.param(param.param_name(), param.ty.clone()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let invoked = builder
        .invoke(top, &args)
        .map_err(|error| error.to_string())?;
    builder
        .set_name(invoked, &format!("{prefix}_invoke__{top_name}"))
        .map_err(|error| error.to_string())?;
    let truth = builder
        .literal(IrValue::make_ubits(1, 1).expect("one fits in one bit"))
        .map_err(|error| error.to_string())?;
    builder
        .set_name(truth, &format!("{prefix}_true"))
        .map_err(|error| error.to_string())?;
    // The tuple keeps the invoked function reachable from the return value,
    // so assertion collection cannot discard the call as an unused result.
    let pair = builder
        .tuple(&[invoked, truth])
        .map_err(|error| error.to_string())?;
    builder
        .set_name(pair, &format!("{prefix}_pair"))
        .map_err(|error| error.to_string())?;
    let result = builder
        .tuple_index(pair, 1)
        .map_err(|error| error.to_string())?;
    builder
        .set_name(result, &format!("{prefix}_result"))
        .map_err(|error| error.to_string())?;
    builder
        .build_into_package(result, package)
        .map_err(|error| error.to_string())?;
    Ok(property_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dslx_assertions::add_assertions_property_function;
    use crate::prover::enum_in_bound::add_property_function;
    use xlsynth_pir::ir::{NodePayload, Type};
    use xlsynth_pir::ir_eval::{AssertionFailure, FnEvalResult, eval_fn_in_package};
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_pir::ir_verify::verify_package;

    type AddProperty = fn(&mut ir::Package, &str) -> Result<String, String>;
    const ADD_PROPERTIES: [AddProperty; 2] =
        [add_assertions_property_function, add_property_function];

    #[test]
    fn property_wrappers_preserve_aggregate_signatures_and_package_ids() {
        let text = r#"package properties

top fn target(tok: token id=11, enabled: bits[1] id=19, payload: (bits[65], bits[8][2]) id=27) -> (token, (bits[65], bits[8][2])) {
  ret tuple.35: (token, (bits[65], bits[8][2])) = tuple(tok, payload, id=35)
}
"#;
        let mut package = Parser::new(text)
            .parse_and_validate_package()
            .expect("valid callee");
        let original_top = package.top.clone();
        for add_property in ADD_PROPERTIES {
            let property_name = add_property(&mut package, "target").expect("build property");
            let top = package.get_fn("target").expect("callee");
            let property = package.get_fn(&property_name).expect("property");
            let signature = |function: &ir::Fn| {
                function
                    .param_nodes()
                    .map(|param| (param.name.clone(), param.ty.clone()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(signature(property), signature(top));
            assert_eq!(property.ret_ty, Type::Bits(1));
            let calls = property
                .nodes
                .iter()
                .filter(|node| matches!(node.payload, NodePayload::Invoke { .. }))
                .collect::<Vec<_>>();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].ty, top.ret_ty);
            let NodePayload::Invoke { to_apply, operands } = &calls[0].payload else {
                unreachable!("filtered invoke node");
            };
            assert_eq!(to_apply, "target");
            assert_eq!(operands.len(), property.params.len());
            for (operand, param) in operands.iter().zip(&property.params) {
                assert_eq!(operand, param);
                assert_eq!(property.get_node(*operand).payload, NodePayload::Param);
            }
            assert_eq!(package.top, original_top);
            verify_package(&package).expect("valid property calls and package-unique IDs");

            let before = package.to_string();
            assert_eq!(
                add_property(&mut package, "target").expect("reuse property"),
                property_name
            );
            assert_eq!(package.to_string(), before);
        }
    }

    #[test]
    fn property_wrappers_observe_failing_callee_assertions() {
        let text = r#"package properties

top fn target(tok: token id=1, predicate: bits[1] id=2) -> (token, bits[1]) {
  checked: token = assert(tok, predicate, message="callee failed", label="callee_assert", id=3)
  ret tuple.4: (token, bits[1]) = tuple(checked, predicate, id=4)
}
"#;
        let mut package = Parser::new(text)
            .parse_and_validate_package()
            .expect("valid callee");
        for add_property in ADD_PROPERTIES {
            let property_name = add_property(&mut package, "target").expect("build property");
            let property = package.get_fn(&property_name).expect("property");
            let args = [IrValue::make_token(), IrValue::make_ubits(1, 0).unwrap()];
            let FnEvalResult::Failure(failure) = eval_fn_in_package(&package, property, &args)
            else {
                panic!("property must observe the invoked assertion");
            };
            assert_eq!(failure.value, IrValue::make_ubits(1, 1).unwrap());
            assert_eq!(
                failure.assertion_failures,
                vec![AssertionFailure {
                    message: "callee failed".to_string(),
                    label: "callee_assert".to_string(),
                }]
            );
            let args = [IrValue::make_token(), IrValue::make_ubits(1, 1).unwrap()];
            let FnEvalResult::Success(success) = eval_fn_in_package(&package, property, &args)
            else {
                panic!("true callee predicate must pass");
            };
            assert_eq!(success.value, IrValue::make_ubits(1, 1).unwrap());
        }
    }
}
