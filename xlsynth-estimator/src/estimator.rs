// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2};
use prost::Message;
use prost_reflect::DescriptorPool;
use std::collections::HashSet;
use std::convert::TryFrom;
// Generated code from proto
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/xls.estimator_model.rs"));
}

use proto::estimator_factor::Source as FactorSource;

/// This is the interface for a node for which we want to be able to predict
/// delay.
pub trait Node {
    fn op(&self) -> &str;

    /// Returns the (flat) bit count of the result.
    fn result_bit_count(&self) -> u64;
    /// Returns the (flat) bit count of operand `i`.
    fn operand_bit_count(&self, i: usize) -> u64;

    /// Returns the number of elements in the operand, if it is an aggregate
    /// type.
    fn operand_element_count(&self, i: usize) -> Option<u64>;

    /// Returns the (flat) bit count of the element type of operand `i`, if it
    /// is an array type.
    fn operand_element_bit_count(&self, i: usize) -> Option<u64>;

    /// Returns the number of operands.
    fn operand_count(&self) -> u64;

    /// Returns true if all operands are identical.
    fn all_operands_identical(&self) -> bool;

    /// Returns true if there is at least one literal operand.
    fn has_literal_operand(&self) -> bool;

    /// Returns the literal operands.
    fn literal_operands(&self) -> Vec<bool>;
}

/// Evaluates a factor (i.e. gets the corresponding attribute) for a node.
fn eval_factor_for_node(factor: &proto::EstimatorFactor, node: &dyn Node) -> f64 {
    let source_i32 = factor.source;
    let source_enum = FactorSource::try_from(source_i32).unwrap();
    match source_enum {
        FactorSource::ResultBitCount => node.result_bit_count() as f64,
        FactorSource::OperandBitCount => {
            node.operand_bit_count(factor.operand_number as usize) as f64
        }
        FactorSource::OperandCount => node.operand_count() as f64,
        FactorSource::OperandElementCount => node
            .operand_element_count(factor.operand_number as usize)
            .unwrap() as f64,
        FactorSource::OperandElementBitCount => node
            .operand_element_bit_count(factor.operand_number as usize)
            .unwrap() as f64,
        FactorSource::InvalidVariableSource => panic!("Invalid variable source"),
    }
}

fn eval_factor_for_data_point(
    factor: &proto::EstimatorFactor,
    data_point: &proto::DataPoint,
) -> f64 {
    let source_i32 = factor.source;
    let source_enum = FactorSource::try_from(source_i32).unwrap();
    match source_enum {
        FactorSource::ResultBitCount => data_point.operation.as_ref().unwrap().bit_count as f64,
        FactorSource::OperandBitCount => {
            let operand_number = factor.operand_number as usize;
            let operand = data_point
                .operation
                .as_ref()
                .unwrap()
                .operands
                .get(operand_number)
                .unwrap();
            operand.bit_count as f64
        }
        FactorSource::OperandCount => data_point.operation.as_ref().unwrap().operands.len() as f64,
        FactorSource::OperandElementCount => todo!(),
        FactorSource::OperandElementBitCount => todo!(),
        FactorSource::InvalidVariableSource => panic!("Invalid variable source"),
    }
}

fn eval_expression_for_node(expr: &proto::EstimatorExpression, node: &dyn Node) -> f64 {
    let expr_type = expr.expression_type.as_ref().unwrap();
    match expr_type {
        proto::estimator_expression::ExpressionType::Factor(factor) => {
            eval_factor_for_node(factor, node)
        }
        proto::estimator_expression::ExpressionType::BinOp(binary_op) => {
            let op = proto::estimator_expression::BinaryOperation::try_from(*binary_op).unwrap();
            let lhs_expression = expr.lhs_expression.as_ref().unwrap();
            let rhs_expression = expr.rhs_expression.as_ref().unwrap();
            let lhs = eval_expression_for_node(lhs_expression, node);
            let rhs = eval_expression_for_node(rhs_expression, node);
            match op {
                proto::estimator_expression::BinaryOperation::Add => lhs + rhs,
                proto::estimator_expression::BinaryOperation::Divide => lhs / rhs,
                proto::estimator_expression::BinaryOperation::Max => lhs.max(rhs),
                proto::estimator_expression::BinaryOperation::Min => lhs.min(rhs),
                proto::estimator_expression::BinaryOperation::Multiply => lhs * rhs,
                proto::estimator_expression::BinaryOperation::Power => lhs.powf(rhs),
                proto::estimator_expression::BinaryOperation::Sub => lhs - rhs,
                proto::estimator_expression::BinaryOperation::Invalid => {
                    panic!("Invalid binary operation")
                }
            }
        }
        proto::estimator_expression::ExpressionType::Constant(constant) => *constant as f64,
    }
}

fn eval_expression_for_data_point(
    expr: &proto::EstimatorExpression,
    data_point: &proto::DataPoint,
) -> f64 {
    let expr_type = expr.expression_type.as_ref().unwrap();
    match expr_type {
        proto::estimator_expression::ExpressionType::Factor(factor) => {
            eval_factor_for_data_point(factor, data_point)
        }
        proto::estimator_expression::ExpressionType::BinOp(binary_op) => {
            let op = proto::estimator_expression::BinaryOperation::try_from(*binary_op).unwrap();
            let lhs_expression = expr.lhs_expression.as_ref().unwrap();
            let rhs_expression = expr.rhs_expression.as_ref().unwrap();
            let lhs = eval_expression_for_data_point(lhs_expression, data_point);
            let rhs = eval_expression_for_data_point(rhs_expression, data_point);
            match op {
                proto::estimator_expression::BinaryOperation::Add => lhs + rhs,
                proto::estimator_expression::BinaryOperation::Divide => lhs / rhs,
                proto::estimator_expression::BinaryOperation::Max => lhs.max(rhs),
                proto::estimator_expression::BinaryOperation::Min => lhs.min(rhs),
                proto::estimator_expression::BinaryOperation::Multiply => lhs * rhs,
                proto::estimator_expression::BinaryOperation::Power => lhs.powf(rhs),
                proto::estimator_expression::BinaryOperation::Sub => lhs - rhs,
                proto::estimator_expression::BinaryOperation::Invalid => {
                    panic!("Invalid binary operation")
                }
            }
        }
        proto::estimator_expression::ExpressionType::Constant(constant) => *constant as f64,
    }
}

/// Wraps an underlying node with a new opcode.
struct AliasNode<'a> {
    new_op: &'a str,
    delegate: &'a dyn Node,
}

impl<'a> Node for AliasNode<'a> {
    fn op(&self) -> &str {
        self.new_op
    }
    fn result_bit_count(&self) -> u64 {
        self.delegate.result_bit_count()
    }
    fn operand_bit_count(&self, i: usize) -> u64 {
        self.delegate.operand_bit_count(i)
    }
    fn operand_count(&self) -> u64 {
        self.delegate.operand_count()
    }
    fn all_operands_identical(&self) -> bool {
        self.delegate.all_operands_identical()
    }
    fn has_literal_operand(&self) -> bool {
        self.delegate.has_literal_operand()
    }
    fn literal_operands(&self) -> Vec<bool> {
        self.delegate.literal_operands()
    }
    fn operand_element_count(&self, i: usize) -> Option<u64> {
        self.delegate.operand_element_count(i)
    }
    fn operand_element_bit_count(&self, i: usize) -> Option<u64> {
        self.delegate.operand_element_bit_count(i)
    }
}

fn eval_estimator(
    em: &proto::EstimatorModel,
    estimator: &proto::Estimator,
    node: &dyn Node,
) -> Result<f64, String> {
    match estimator.estimator.as_ref().unwrap() {
        proto::estimator::Estimator::Fixed(fixed) => Ok(*fixed as f64),
        proto::estimator::Estimator::AliasOp(alias_op) => {
            let estimator: Option<&proto::Estimator> = find_estimator_for_op(
                em,
                alias_op,
                node.all_operands_identical(),
                node.has_literal_operand(),
                &node.literal_operands(),
            );
            if let Some(estimator) = estimator {
                eval_estimator(
                    em,
                    estimator,
                    &AliasNode {
                        new_op: alias_op,
                        delegate: node,
                    },
                )
            } else {
                Err(format!(
                    "No estimator found for op: {} aliased to: {}",
                    node.op(),
                    alias_op
                ))
            }
        }
        proto::estimator::Estimator::Regression(regression) => {
            let data_points: &Vec<proto::DataPoint> = &em.data_points;
            let estimator_impl = make_regression_estimator(regression, &data_points, &node.op())?;
            Ok(estimator_impl.predict(node))
        }
        proto::estimator::Estimator::BoundingBox(_bounding_box) => todo!(),
        proto::estimator::Estimator::LogicalEffort(_logical_effort) => todo!(),
        proto::estimator::Estimator::AreaRegression(_area_regression) => todo!(),
    }
}

/// Returns whether the node's literal operands (as given by `literal_operands`)
/// match the literal operand details.
///
/// This is a predicate evaluator to determine if a node matches a given
/// specialization.
fn matches_literal_operand_details(
    literal_operand_details: &proto::LiteralOperandDetails,
    literal_operands: &[bool],
) -> bool {
    log::info!(
        "checking whether literal operand vector {:?} matches operand details: {:?}",
        literal_operands,
        literal_operand_details
    );

    // Note: from the proto definition, it says "if none are specified, then any
    // operand is allowed to be non-literal."
    let allowed_nonliteral_operand: Option<HashSet<usize>> = if literal_operand_details
        .allowed_nonliteral_operand
        .is_empty()
    {
        None
    } else {
        Some(
            literal_operand_details
                .allowed_nonliteral_operand
                .iter()
                .map(|i| *i as usize)
                .collect::<HashSet<_>>(),
        )
    };

    let required_literal_operand: HashSet<usize> = literal_operand_details
        .required_literal_operand
        .iter()
        .map(|i| *i as usize)
        .collect::<HashSet<_>>();
    for (i, is_literal) in literal_operands.iter().enumerate() {
        // If `i` is required to be literal but is not, then we don't match.
        if required_literal_operand.contains(&i) && !*is_literal {
            log::info!("required literal operand {} is not literal", i);
            return false;
        }
        if let Some(ref allowed_nonliteral) = allowed_nonliteral_operand {
            // If `i` is not allowed to be a non-literal but is, then we don't match.
            let is_allowed_nonliteral = allowed_nonliteral.contains(&i);
            let is_nonliteral = !*is_literal;
            if is_nonliteral && !is_allowed_nonliteral {
                log::info!("non-literal operand {} is not allowed", i);
                return false;
            }
        }
    }
    log::info!(
        "literal operand vector {:?} matches operand details: {:?}",
        literal_operands,
        literal_operand_details
    );
    true
}

fn find_estimator_for_op<'a>(
    em: &'a proto::EstimatorModel,
    op: &str,
    all_operands_identical: bool,
    has_literal_operand: bool,
    literal_operands: &[bool],
) -> Option<&'a proto::Estimator> {
    for op_model in em.op_models.iter() {
        if op_model.op != op {
            continue;
        }
        // Reminder: an `OpModel` has a "typical case" estimator and can have any number
        // of "specializations" that match on the predicates we get. The specializations
        // take precedence if they are matched on.
        for specialization in op_model.specializations.iter() {
            log::info!("checking specialization: {:?}", specialization);
            let specialization_kind =
                proto::SpecializationKind::try_from(specialization.kind).unwrap();

            if specialization_kind == proto::SpecializationKind::OperandsIdentical
                && all_operands_identical
            {
                assert!(
                    specialization.details.is_none(),
                    "'operands-identical' specializations should not have details"
                );
                return Some(&specialization.estimator.as_ref().unwrap());
            }
            if specialization_kind == proto::SpecializationKind::HasLiteralOperand
                && has_literal_operand
            {
                // For literal operands there can also be detailed requirements.
                let details = specialization.details.as_ref().unwrap();
                let specialization_details: &proto::op_model::specialization::details::SpecializationDetails = details.specialization_details.as_ref().unwrap();
                let literal_operand_details: &proto::LiteralOperandDetails = match specialization_details {
                    proto::op_model::specialization::details::SpecializationDetails::LiteralOperandDetails(literal_operand_details) => literal_operand_details,
                };
                if matches_literal_operand_details(literal_operand_details, literal_operands) {
                    return Some(&specialization.estimator.as_ref().unwrap());
                }
            }
        }
        return Some(&op_model.estimator.as_ref().unwrap());
    }
    None
}

pub fn eval_estimator_model(em: &proto::EstimatorModel, node: &dyn Node) -> Result<f64, String> {
    // First we have to get the appropriate estimator for this node.
    let estimator = find_estimator_for_op(
        em,
        node.op(),
        node.all_operands_identical(),
        node.has_literal_operand(),
        &node.literal_operands(),
    );
    if let Some(estimator) = estimator {
        eval_estimator(em, estimator, node)
    } else {
        Err(format!("No estimator found for op: {}", node.op()))
    }
}

/// Augment a 2D array of factors `xdata` into a 2D matrix with a leading
/// constant term in addition to raw and log2 terms for each factor.
///
/// This function returns an
/// augmented matrix:
/// - The first column is a constant term (1.0).
/// - The next `n` columns are the raw values from `xdata`.
/// - The final `n` columns are the log2-transformed values of `xdata` (with
///   `log2(1)` for zeros).
fn augment_xdata(xdata: &Array2<f64>) -> Array2<f64> {
    let n = xdata.ncols();
    let mut augmented = Array2::<f64>::zeros((xdata.nrows(), 1 + 2 * n));
    {
        let mut col0 = augmented.column_mut(0);
        col0.fill(1.0);
    }
    for i in 0..n {
        {
            let mut col = augmented.column_mut(1 + i);
            col.assign(&xdata.column(i));
        }
        {
            let mut col = augmented.column_mut(1 + n + i);
            col.assign(&xdata.column(i).map(|&x| (x.max(1.0)).log2()));
        }
    }
    augmented
}

struct RegressionEstimatorImpl {
    exprs: Vec<proto::EstimatorExpression>,
    coeffs: Array1<f64>,
}

impl RegressionEstimatorImpl {
    fn predict(&self, node: &dyn Node) -> f64 {
        let xdata: Array1<f64> = self
            .exprs
            .iter()
            .map(|e| eval_expression_for_node(e, node))
            .collect::<Array1<f64>>();
        let expr_count = xdata.len();
        let x: Array1<f64> = augment_xdata(&xdata.into_shape((1, expr_count)).unwrap())
            .row(0)
            .to_owned();
        let result: f64 = x.dot(&self.coeffs);
        result
    }
}

fn datapoint_is_for_op(dp: &proto::DataPoint, op: &str) -> bool {
    let operation: &Option<proto::Operation> = &dp.operation;
    if let Some(operation) = operation {
        operation.op == op
    } else {
        log::warn!("data point has no operation: {:?}", dp);
        false
    }
}

/// After the characterization process we have a bunch of `y` values from our
/// empirically observed delays and we can construct `x` variables with a vector
/// like the following:
///
/// `[1, x0, log2(x0), x1, log2(x1), ...]`
///
/// The `x` variables are given as expressions in the op model.
///
/// See the `fit_curve` routine for a similar implementation in Python:
/// https://sourcegraph.com/github.com/google/xls@5e5e2e0/-/blob/xls/estimators/estimator_model.py?L574
fn make_regression_estimator(
    spec: &proto::RegressionEstimator,
    data_points: &[proto::DataPoint],
    op: &str,
) -> Result<RegressionEstimatorImpl, String> {
    let data_points: Vec<&proto::DataPoint> = data_points
        .iter()
        .filter(|dp| datapoint_is_for_op(dp, op))
        .collect();

    if data_points.is_empty() {
        return Err(format!("No data points found for op: {}", op));
    }

    // All the ground truth delays.
    let y = Array1::from_vec(data_points.iter().map(|dp| dp.delay as f64).collect());

    log::info!("ground truth delays: {:?}", y);

    // `xdata` is the factors for each data point.
    let mut xdata = Array2::zeros((data_points.len(), spec.expressions.len()));
    for (i, dp) in data_points.iter().enumerate() {
        for (j, expr) in spec.expressions.iter().enumerate() {
            xdata[[i, j]] = eval_expression_for_data_point(expr, dp);
        }
    }
    let x: Array2<f64> = augment_xdata(&xdata);

    let (params, euclidian_norm) = nnls::nnls(x.view(), y.view());
    log::info!("euclidian norm: {}", euclidian_norm);
    Ok(RegressionEstimatorImpl {
        exprs: spec.expressions.clone(),
        coeffs: params,
    })
}

pub fn parse_estimator_model_textproto(
    textproto: &str,
) -> Result<proto::EstimatorModel, Box<dyn std::error::Error>> {
    // Load the descriptor set generated during build
    let descriptor_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/descriptors.bin"));
    let pool = DescriptorPool::decode(descriptor_bytes.as_ref())?;

    // Get the message descriptor for EstimatorModel
    let message_descriptor: prost_reflect::MessageDescriptor = pool
        .get_message_by_name("xls.estimator_model.EstimatorModel")
        .ok_or("Message descriptor not found")?;

    let dynamic_message =
        prost_reflect::DynamicMessage::parse_text_format(message_descriptor, textproto)?;

    // Encode the dynamic message and then decode it into the non-dynamic form.
    let buf = dynamic_message.encode_to_vec();
    let estimator_model = proto::EstimatorModel::decode(&*buf)?;

    Ok(estimator_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeNode {
        op: String,
        result_bit_count: u64,
        operand_bit_count: fn(usize) -> u64,
        operand_count: u64,
        operand_is_literal: fn(usize) -> bool,
        all_operands_identical: bool,
    }

    impl Node for FakeNode {
        fn op(&self) -> &str {
            &self.op
        }
        fn result_bit_count(&self) -> u64 {
            self.result_bit_count
        }
        fn operand_bit_count(&self, operand_number: usize) -> u64 {
            (self.operand_bit_count)(operand_number)
        }
        fn operand_count(&self) -> u64 {
            self.operand_count
        }
        fn operand_element_count(&self, _operand_number: usize) -> Option<u64> {
            None
        }
        fn operand_element_bit_count(&self, _operand_number: usize) -> Option<u64> {
            None
        }
        fn all_operands_identical(&self) -> bool {
            self.all_operands_identical
        }
        fn has_literal_operand(&self) -> bool {
            true
        }
        fn literal_operands(&self) -> Vec<bool> {
            (0..self.operand_count)
                .map(|i| (self.operand_is_literal)(i as usize))
                .collect()
        }
    }

    #[test]
    fn test_eval_estimator_add() {
        let textproto = r#"
op_models {
  op: "kAdd"
  estimator {
    regression {
      expressions {
        factor {
          source: OPERAND_BIT_COUNT
        }
      }
    }
  }
}
data_points {
  operation {
    op: "kAdd"
    bit_count: 5
    operands {
      bit_count: 5
    }
    operands {
      bit_count: 5
    }
  }
  delay: 240
  delay_offset: 94
}
data_points {
  operation {
    op: "kAdd"
    bit_count: 4
    operands {
      bit_count: 4
    }
    operands {
      bit_count: 4
    }
  }
  delay: 226
  delay_offset: 94
}
data_points {
  operation {
    op: "kAdd"
    bit_count: 2
    operands {
      bit_count: 2
    }
    operands {
      bit_count: 2
    }
  }
  delay: 186
  delay_offset: 94
}
        "#;

        let em = parse_estimator_model_textproto(textproto).unwrap();
        let e = &em.op_models[0].estimator;
        let spec = match &e.as_ref().unwrap().estimator {
            Some(proto::estimator::Estimator::Regression(regression)) => regression,
            _ => panic!("Expected regression estimator"),
        };
        let data_points = &em.data_points;

        // Fit the regression estimator
        let estimator = make_regression_estimator(&spec, &data_points, "kAdd").unwrap();

        let node = FakeNode {
            op: "kAdd".to_string(),
            result_bit_count: 3,
            operand_bit_count: |_operand_number: usize| 3,
            operand_count: 2,
            operand_is_literal: |_operand_number: usize| false,
            all_operands_identical: false,
        };

        // Predict delay for operand bit count 3
        let predicted_delay = estimator.predict(&node);

        assert_eq!(format!("{:.2}", predicted_delay), "208.86");
    }

    #[test]
    fn test_eval_estimator_alias() {
        let textproto = r#"
op_models {
  op: "kShll"
  estimator {
    regression {
      expressions {
        factor {
          source: OPERAND_BIT_COUNT
        }
      }
    }
  }
}
op_models {
  op: "kShrl"
  estimator {
    alias_op: "kShll"
  }
}
data_points {
  operation {
    op: "kShll"
    bit_count: 12
    operands {
      bit_count: 12
    }
    operands {
      bit_count: 4
    }
  }
  delay: 300
  delay_offset: 94
}
data_points {
  operation {
    op: "kShll"
    bit_count: 8
    operands {
      bit_count: 8
    }
    operands {
      bit_count: 3
    }
  }
  delay: 255
  delay_offset: 94
}
data_points {
  operation {
    op: "kShll"
    bit_count: 6
    operands {
      bit_count: 6
    }
    operands {
      bit_count: 3
    }
  }
  delay: 223
  delay_offset: 94
}
data_points {
  operation {
    op: "kShll"
    bit_count: 4
    operands {
      bit_count: 4
    }
    operands {
      bit_count: 2
    }
  }
  delay: 202
  delay_offset: 94
}
      "#;

        let em: proto::EstimatorModel = parse_estimator_model_textproto(textproto).unwrap();
        let node = FakeNode {
            op: "kShrl".to_string(),
            result_bit_count: 5,
            operand_bit_count: |operand_number: usize| if operand_number == 0 { 5 } else { 3 },
            operand_count: 2,
            operand_is_literal: |_operand_number: usize| false,
            all_operands_identical: false,
        };
        let predicted_delay = eval_estimator_model(&em, &node).unwrap();
        assert_eq!(format!("{:.2}", predicted_delay), "213.91");
    }

    #[test]
    fn test_two_factors_one_specialization() {
        let textproto = r#"
op_models {
  op: "kSel"
  estimator {
    regression {
      expressions {
        factor {
          source: RESULT_BIT_COUNT
        }
      }
      expressions {
        factor {
          source: OPERAND_COUNT
        }
      }
    }
  }
  specializations {
    kind: HAS_LITERAL_OPERAND
    estimator {
      fixed: 0
    }
    details {
      literal_operand_details {
        required_literal_operand: 0
      }
    }
  }
}
  data_points {
  operation {
    op: "kSel"
    bit_count: 64
    operands {
      bit_count: 2
    }
    operands {
      bit_count: 64
    }
    operands {
      bit_count: 64
    }
    operands {
      bit_count: 64
    }
    operands {
      bit_count: 64
    }
  }
  delay: 290
  delay_offset: 94
}
data_points {
  operation {
    op: "kSel"
    bit_count: 32
    operands {
      bit_count: 2
    }
    operands {
      bit_count: 32
    }
    operands {
      bit_count: 32
    }
    operands {
      bit_count: 32
    }
    operands {
      bit_count: 32
    }
  }
  delay: 271
  delay_offset: 94
}
data_points {
  operation {
    op: "kSel"
    bit_count: 16
    operands {
      bit_count: 2
    }
    operands {
      bit_count: 16
    }
    operands {
      bit_count: 16
    }
    operands {
      bit_count: 16
    }
    operands {
      bit_count: 16
    }
  }
  delay: 276
  delay_offset: 94
}
data_points {
  operation {
    op: "kSel"
    bit_count: 8
    operands {
      bit_count: 2
    }
    operands {
      bit_count: 8
    }
    operands {
      bit_count: 8
    }
    operands {
      bit_count: 8
    }
    operands {
      bit_count: 8
    }
  }
  delay: 265
  delay_offset: 94
}
  data_points {
  operation {
    op: "kSel"
    bit_count: 1024
    operands {
      bit_count: 1
    }
    operands {
      bit_count: 1024
    }
    operands {
      bit_count: 1024
    }
  }
  delay: 309
  delay_offset: 94
}
data_points {
  operation {
    op: "kSel"
    bit_count: 512
    operands {
      bit_count: 1
    }
    operands {
      bit_count: 512
    }
    operands {
      bit_count: 512
    }
  }
  delay: 307
  delay_offset: 94
}
data_points {
  operation {
    op: "kSel"
    bit_count: 128
    operands {
      bit_count: 1
    }
    operands {
      bit_count: 128
    }
    operands {
      bit_count: 128
    }
  }
  delay: 258
  delay_offset: 94
}
data_points {
  operation {
    op: "kSel"
    bit_count: 64
    operands {
      bit_count: 1
    }
    operands {
      bit_count: 64
    }
    operands {
      bit_count: 64
    }
  }
  delay: 235
  delay_offset: 94
}
data_points {
  operation {
    op: "kSel"
    bit_count: 32
    operands {
      bit_count: 1
    }
    operands {
      bit_count: 32
    }
    operands {
      bit_count: 32
    }
  }
  delay: 245
  delay_offset: 94
}
data_points {
  operation {
    op: "kSel"
    bit_count: 16
    operands {
      bit_count: 1
    }
    operands {
      bit_count: 16
    }
    operands {
      bit_count: 16
    }
  }
  delay: 214
  delay_offset: 94
}
data_points {
  operation {
    op: "kSel"
    bit_count: 8
    operands {
      bit_count: 1
    }
    operands {
      bit_count: 8
    }
    operands {
      bit_count: 8
    }
  }
  delay: 209
  delay_offset: 94
}
        "#;

        let _ = env_logger::builder().is_test(true).init();
        let em = parse_estimator_model_textproto(textproto).unwrap();
        let node = FakeNode {
            op: "kSel".to_string(),
            result_bit_count: 17,
            operand_bit_count: |operand_number: usize| if operand_number == 0 { 1 } else { 17 },
            operand_count: 3,
            operand_is_literal: |_operand_number: usize| false,
            all_operands_identical: false,
        };
        let predicted_delay = eval_estimator_model(&em, &node).unwrap();
        assert_eq!(format!("{:.2}", predicted_delay), "222.63");

        let specialized_node = FakeNode {
            op: "kSel".to_string(),
            result_bit_count: 17,
            operand_bit_count: |operand_number: usize| if operand_number == 0 { 1 } else { 17 },
            operand_count: 3,
            operand_is_literal: |operand_number: usize| operand_number == 0,
            all_operands_identical: false,
        };
        let specialized_delay = eval_estimator_model(&em, &specialized_node).unwrap();
        assert_eq!(format!("{:.2}", specialized_delay), "0.00");
    }

    // The op-specific specializations should not affect other operations. This
    // regression test verifies that when we evaluate a node for `kSub`, the
    // `kAdd` specializations are ignored and the default `kSub` estimator is
    // used instead of accidentally matching the `kAdd` specialization.
    #[test]
    fn test_find_estimator_respects_op() {
        let textproto = r#"
op_models {
  op: "kAdd"
  estimator {
    fixed: 111
  }
  specializations {
    kind: OPERANDS_IDENTICAL
    estimator {
      fixed: 222
    }
  }
}
op_models {
  op: "kSub"
  estimator {
    fixed: 333
  }
}
"#;

        let em = parse_estimator_model_textproto(textproto).unwrap();
        let node = FakeNode {
            op: "kSub".to_string(),
            result_bit_count: 32,
            operand_bit_count: |_i| 32,
            operand_count: 2,
            operand_is_literal: |_i| false,
            all_operands_identical: true,
        };
        let delay = eval_estimator_model(&em, &node).unwrap();
        assert_eq!(delay, 333.0);
    }
}
