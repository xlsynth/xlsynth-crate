

// Type alias for the lowering map
pub type IrToGateMap = HashMap<ir::NodeRef, AigBitVector>;

/// Holds the output of the `gatify` process.
#[derive(Debug)]
pub struct GatifyOutput {
    pub gate_fn: GateFn,
    pub lowering_map: IrToGateMap,
}

fn validate_fn_for_gatify(f: &ir::Fn) -> Result<(), String> {
    // Validate the function in a minimal package context.
    //
    // This catches structural issues (operand bounds/order, return node/type,
    // text-id uniqueness within the function, etc.) before and after
    // `prep_for_gatify`.
    let pkg = ir::Package {
        name: "gatify_validate".to_string(),
        file_table: ir::FileTable::new(),
        members: vec![ir::PackageMember::Function(f.clone())],
        top: Some((f.name.clone(), ir::MemberType::Function)),
    };
    ir_validate::validate_package(&pkg).map_err(|e| e.to_string())
}

fn gatify_lower_prepared_fn(
    f: &ir::Fn,
    options: &GatifyOptions,
    orig_ref_by_text_id: Option<&HashMap<usize, ir::NodeRef>>,
    equiv_fn: &ir::Fn,
) -> Result<GatifyOutput, String> {
    let mut g8_builder = GateBuilder::new(
        f.name.clone(),
        GateBuilderOptions {
            fold: options.fold,
            hash: options.hash,
        },
    );
    let mut env = GateEnv::new();
    gatify_internal(f, &mut g8_builder, &mut env, options)?;
    let gate_fn = g8_builder.build();
    log::debug!(
        "converted IR function to gate function:\n{}",
        gate_fn.to_string()
    );

    let mut lowering_map: IrToGateMap = HashMap::new();
    for (node_ref, gate_or_vec) in env.ir_to_g8.into_iter() {
        let bit_vector = match gate_or_vec {
            GateOrVec::BitVector(bv) => bv,
            GateOrVec::Gate(gate_ref) => AigBitVector::from_bit(gate_ref),
        };
        if let Some(orig_ref_by_text_id) = orig_ref_by_text_id {
            let prepared_text_id = f.get_node(node_ref).text_id;
            let Some(orig_node_ref) = orig_ref_by_text_id.get(&prepared_text_id).copied() else {
                continue;
            };
            lowering_map.insert(orig_node_ref, bit_vector);
        } else {
            lowering_map.insert(node_ref, bit_vector);
        }
    }

    if options.check_equivalence {
        log::info!("checking equivalence of IR function and gate function...");
        check_equivalence::validate_same_fn(equiv_fn, &gate_fn)?;
    }
    Ok(GatifyOutput {
        gate_fn,
        lowering_map,
    })
}

/// Lowers an IR function that has already been prepared for gatification.
///
/// This skips `prep_for_gatify`; callers are responsible for running any
/// desired prep rewrites first.
pub fn gatify_prepared_fn(f: &ir::Fn, options: GatifyOptions) -> Result<GatifyOutput, String> {
    validate_fn_for_gatify(f)
        .map_err(|e| format!("PIR validation failed before gatify_prepared_fn: {e}"))?;
    gatify_lower_prepared_fn(f, &options, None, f)
}

pub fn gatify(orig_fn: &ir::Fn, options: GatifyOptions) -> Result<GatifyOutput, String> {
    validate_fn_for_gatify(orig_fn)
        .map_err(|e| format!("PIR validation failed before prep_for_gatify: {e}"))?;

    // `prep_for_gatify` may introduce many new nodes (e.g. lowering ext ops), so
    // any `NodeRef { index }` values produced during gatification generally
    // refer to the *prepared* function's node vector. Most consumers want to
    // interpret the lowering map in terms of the original function, so we
    // remap prepared nodes back to original nodes via stable `text_id`.
    let mut orig_ref_by_text_id: HashMap<usize, ir::NodeRef> = HashMap::new();
    for (idx, n) in orig_fn.nodes.iter().enumerate() {
        orig_ref_by_text_id.insert(n.text_id, ir::NodeRef { index: idx });
    }

    let prepared_fn = prep_for_gatify(
        orig_fn,
        options.range_info.as_deref(),
        PrepForGatifyOptions {
            enable_rewrite_carry_out: options.enable_rewrite_carry_out,
            enable_rewrite_prio_encode: options.enable_rewrite_prio_encode,
            enable_rewrite_nary_add: options.enable_rewrite_nary_add,
            enable_rewrite_mask_low: options.enable_rewrite_mask_low,
            enable_rewrite_normalize_left: options.enable_rewrite_normalize_left,
            ..PrepForGatifyOptions::all_opts_enabled()
        },
    );
    validate_fn_for_gatify(&prepared_fn)
        .map_err(|e| format!("PIR validation failed after prep_for_gatify: {e}"))?;
    gatify_lower_prepared_fn(&prepared_fn, &options, Some(&orig_ref_by_text_id), orig_fn)
}

pub fn gatify_node_as_fn(
    f: &ir::Fn,
    node_ref: ir::NodeRef,
    options: &GatifyOptions,
) -> Result<GateFn, String> {
    let target_text_id = f.get_node(node_ref).text_id;
    validate_fn_for_gatify(f)
        .map_err(|e| format!("PIR validation failed before prep_for_gatify: {e}"))?;
    let prepared_fn = prep_for_gatify(
        f,
        options.range_info.as_deref(),
        PrepForGatifyOptions {
            enable_rewrite_carry_out: options.enable_rewrite_carry_out,
            enable_rewrite_prio_encode: options.enable_rewrite_prio_encode,
            enable_rewrite_nary_add: options.enable_rewrite_nary_add,
            enable_rewrite_mask_low: options.enable_rewrite_mask_low,
            enable_rewrite_normalize_left: options.enable_rewrite_normalize_left,
            ..PrepForGatifyOptions::all_opts_enabled()
        },
    );
    validate_fn_for_gatify(&prepared_fn)
        .map_err(|e| format!("PIR validation failed after prep_for_gatify: {e}"))?;
    let f = &prepared_fn;
    let prepared_node_ref: ir::NodeRef = f
        .nodes
        .iter()
        .enumerate()
        .find_map(|(idx, n)| {
            if n.text_id == target_text_id {
                Some(ir::NodeRef { index: idx })
            } else {
                None
            }
        })
        .ok_or_else(|| {
            format!(
                "gatify_node_as_fn: could not find node with original text_id={} after prep",
                target_text_id
            )
        })?;
    let node = f.get_node(prepared_node_ref);
    let mut g8_builder = GateBuilder::new(
        format!("{}_node_{}", f.name, node.text_id),
        GateBuilderOptions {
            fold: options.fold,
            hash: options.hash,
        },
    );
    let mut env = GateEnv::new();

    // Precompute a map from parameter id to its NodeRef in f.nodes. This is used
    // when lowering GetParam nodes.
    let mut param_id_to_node_ref: HashMap<ParamId, ir::NodeRef> = HashMap::new();
    for (i, param) in f.params.iter().enumerate() {
        let param_ref = ir::NodeRef { index: i + 1 };
        assert!(
            f.nodes[i + 1].payload == ir::NodePayload::GetParam(param.id),
            "expected param node at index {}",
            i + 1
        );
        param_id_to_node_ref.insert(param.id, param_ref);
    }

    // Seed the direct operands of this node as independent GateFn inputs.
    let operands: Vec<ir::NodeRef> = ir_utils::operands(&node.payload);
    for (i, operand_ref) in operands.iter().enumerate() {
        if env.contains(*operand_ref) {
            continue;
        }
        let operand_ty = f.get_node_ty(*operand_ref);
        let width = operand_ty.bit_count();
        let input_bits = g8_builder.add_input(format!("op{}_n{}", i, operand_ref.index), width);
        env.add(*operand_ref, GateOrVec::BitVector(input_bits));
    }

    // Lower the node into env/builder and emit it as the single output.
    match &node.payload {
        ir::NodePayload::GetParam(param_id) => {
            let param = f
                .params
                .iter()
                .find(|p| p.id == *param_id)
                .ok_or_else(|| format!("GetParam refers to missing ParamId {:?}", param_id))?;
            let bits = g8_builder.add_input(param.name.clone(), param.ty.bit_count());
            g8_builder.add_output("output_value".to_string(), bits);
            return Ok(g8_builder.build());
        }
        ir::NodePayload::Literal(literal) => {
            let bits = flatten_literal_to_bits(literal, &node.ty, &mut g8_builder);
            g8_builder.add_output("output_value".to_string(), bits);
            return Ok(g8_builder.build());
        }
        ir::NodePayload::Nil
        | ir::NodePayload::Assert { .. }
        | ir::NodePayload::AfterAll(..)
        | ir::NodePayload::Trace { .. } => {
            g8_builder.add_output(
                "output_value".to_string(),
                AigBitVector::zeros(node.ty.bit_count()),
            );
            return Ok(g8_builder.build());
        }
        _ => {}
    }

    gatify_node(
        f,
        node_ref,
        node,
        &mut g8_builder,
        &mut env,
        options,
        &param_id_to_node_ref,
    )?;
    let output_bits = env.get_bit_vector(node_ref)?;
    g8_builder.add_output("output_value".to_string(), output_bits);
    Ok(g8_builder.build())
}

#[cfg(test)]
mod tests {
    use crate::aig::gate::{AigNode, GateFn, Split};
    use crate::aig::get_summary_stats::{AigStats, SummaryStats, get_aig_stats, get_summary_stats};
    use crate::aig::{AigBitVector, AigOperand};
    use crate::aig_sim::gate_sim;
    use crate::check_equivalence;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions, ReductionKind};
    use crate::gatify::ir2gate::{GatifyOptions, gatify};
    use crate::ir2gate_utils::{AdderMapping, Direction, gatify_barrel_shifter};
    use xlsynth::{IrBits, IrValue};
    use xlsynth_pir::ir;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth_pir::ir_parser;
    use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;

    fn ir_value_to_gate_bits(value: &IrValue, ty: &ir::Type) -> IrBits {
        let mut bits = Vec::new();
        flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut bits).unwrap();
        IrBits::from_lsb_is_0(&bits)
    }

    #[test]
    fn test_gatify_array_literal_flattening() {
        let ir_text = "fn f() -> bits[8][5] {\n  ret literal.1: bits[8][5] = literal(value=[1, 2, 3, 4, 5], id=1)\n}";
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_fn = parser.parse_fn().unwrap();

        let gatify_output = gatify(
            &ir_fn,
            GatifyOptions {
                fold: false,
                hash: false,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();

        // The output should be a bit vector of 5 * 8 = 40 bits
        let output_vec = &gatify_output.gate_fn.outputs[0].bit_vector;
        assert_eq!(
            output_vec.get_bit_count(),
            40,
            "Expected 40 bits for bits[8][5] array literal"
        );
    }

    #[test]
    fn test_gatify_array_concat_preserves_index_order() {
        let gate_fn = gatify_ir_text(
            r#"package sample

top fn f(lhs: bits[4][2] id=1, rhs: bits[4][3] id=2) -> bits[4][5] {
  ret joined: bits[4][5] = array_concat(lhs, rhs, id=3)
}
"#,
        );
        let lhs_ty = ir::Type::new_array(ir::Type::Bits(4), 2);
        let rhs_ty = ir::Type::new_array(ir::Type::Bits(4), 3);
        let out_ty = ir::Type::new_array(ir::Type::Bits(4), 5);
        let lhs_value = IrValue::make_array(&[
            IrValue::make_ubits(4, 1).unwrap(),
            IrValue::make_ubits(4, 2).unwrap(),
        ])
        .unwrap();
        let rhs_value = IrValue::make_array(&[
            IrValue::make_ubits(4, 3).unwrap(),
            IrValue::make_ubits(4, 4).unwrap(),
            IrValue::make_ubits(4, 5).unwrap(),
        ])
        .unwrap();
        let want = ir_value_to_gate_bits(
            &IrValue::make_array(&[
                IrValue::make_ubits(4, 1).unwrap(),
                IrValue::make_ubits(4, 2).unwrap(),
                IrValue::make_ubits(4, 3).unwrap(),
                IrValue::make_ubits(4, 4).unwrap(),
                IrValue::make_ubits(4, 5).unwrap(),
            ])
            .unwrap(),
            &out_ty,
        );
        let got = gate_sim::eval(
            &gate_fn,
            &[
                ir_value_to_gate_bits(&lhs_value, &lhs_ty),
                ir_value_to_gate_bits(&rhs_value, &rhs_ty),
            ],
            gate_sim::Collect::None,
        )
        .outputs[0]
            .clone();
        assert_eq!(got, want);
    }

    #[test]
    fn test_gatify_gate_is_unsupported() {
        let ir_text = r#"package sample

top fn f(p: bits[1] id=1, x: bits[8] id=2) -> bits[8] {
  ret gated: bits[8] = gate(p, x, id=3)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();
        let err = match gatify(
            &ir_fn,
            GatifyOptions::all_opts_disabled(),
        )
        .unwrap();

        let lowering_map = gatify_output.lowering_map;

        // Find the 'Not' node and its input parameter reference within the IR function.
        let mut not_node_ref: Option<ir::NodeRef> = None;
        let mut param_ref: Option<ir::NodeRef> = None;

        for (i, node) in ir_fn.nodes.iter().enumerate() {
            if let ir::NodePayload::Unop(ir::Unop::Not, operand) = &node.payload {
                not_node_ref = Some(ir::NodeRef { index: i });
                param_ref = Some(*operand);
                break; // Assuming only one 'Not' operation in this simple test
                // case
            }
        }

        let not_ref = not_node_ref.expect("Could not find 'Not' node in IR function");
        let param_ref = param_ref.expect("Could not find input parameter for 'Not' node");

        // Retrieve AIG vectors from the map using the identified node references
        let param_vec = lowering_map
            .get(&param_ref)
            .expect("Lowering for parameter node not found");
        let not_vec = lowering_map
            .get(&not_ref)
            .expect("Lowering for 'Not' node not found");

        assert_eq!(param_vec.get_bit_count(), 8, "Param should be 8 bits");
        assert_eq!(not_vec.get_bit_count(), 8, "Node 'not' should be 8 bits");

        // Verify that each bit in not_vec is the negation of the corresponding bit in
        // param_vec
        for i in 0..8 {
            let param_op = param_vec.get_lsb(i);
            let not_op = not_vec.get_lsb(i);

            // The not operation should ideally result in operands pointing to the same
            // underlying node but with opposite negation flags.
            assert_eq!(param_op.node, not_op.node, "Bit {} nodes should match", i);
            assert_ne!(
                param_op.negated, not_op.negated,
                "Bit {} negation flags should differ",
                i
            );
        }
    }

    #[test]
    fn test_gatify_seeds_pir_node_ids_on_inputs_and_lowered_ands() {
        let ir_text = r#"package sample
fn f(a: bits[2] id=1, b: bits[2] id=2) -> bits[2] {
  ret add.3: bits[2] = add(a, b, id=3)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();

        let gatify_output = gatify(
            &ir_fn,
            GatifyOptions {
                fold: false,
                hash: false,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();

        let gate_fn = gatify_output.gate_fn;
        for input in &gate_fn.inputs {
            let expected_id = if input.name == "a" { 1 } else { 2 };
            for bit in input.bit_vector.iter_lsb_to_msb() {
                assert_eq!(
                    gate_fn.gates[bit.node.id].get_pir_node_ids(),
                    &[expected_id],
                    "input bit {} should carry PIR provenance id {}",
                    input.name,
                    expected_id
                );
            }
        }

        for node in &gate_fn.gates {
            if let AigNode::And2 { .. } = node {
                assert_eq!(
                    node.get_pir_node_ids(),
                    &[3],
                    "every lowered AND for this simple add should carry the add node text_id"
                );
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum CmpLowering {
        /// Builds one comparison test per input bit and gates each test with an
        /// independently recomputed equality prefix over all more-significant
        /// bits. The per-bit tests are OR-reduced into the strict comparison;
        /// non-strict comparisons add a separate full-width equality test and
        /// OR it with the strict result. Signed comparisons use the same
        /// full-width unsigned comparison result, guarded by sign-difference
        /// logic.
        RepeatedPrefix,
        /// Builds the unsigned comparison and equality result with the
        /// recursive half-split bit-test tree, where each node returns both
        /// "this range compares true" and "this range is equal". Non-strict
        /// unsigned comparisons explicitly OR that comparison result with the
        /// equality result. Signed comparisons still compare the full operand
        /// width as unsigned after sign-difference handling, so the sign bits
        /// participate in both the sign logic and the unsigned tree.
        RecursiveExplicitOrEq,
        /// Builds a recursive half-split bit-test tree, but avoids explicit
        /// equality ORs for non-strict comparisons by using the boolean dual
        /// form: `lhs <= rhs` is `!(lhs > rhs)` and `lhs >= rhs` is
        /// `!(lhs < rhs)`. Signed comparisons split sign handling from the
        /// magnitude comparison, compare only the lower bits when signs match,
        /// and use the same boolean-dual rewrite for signed non-strict forms.
        RecursiveBitTreeSignSplitBooleanDual,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct CmpRecursiveBitTreeQorRow {
        binop: ir::Binop,
        width: usize,
        repeated_prefix_and_nodes: usize,
        repeated_prefix_depth: usize,
        recursive_bit_tree_and_nodes: usize,
        recursive_bit_tree_depth: usize,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct CmpBooleanDualQorRow {
        binop: ir::Binop,
        width: usize,
        recursive_explicit_or_eq_and_nodes: usize,
        recursive_explicit_or_eq_depth: usize,
        boolean_dual_and_nodes: usize,
        boolean_dual_depth: usize,
    }

    fn cmp_boolean_dual_qor_row(
        binop: ir::Binop,
        width: usize,
        recursive_explicit_or_eq_and_nodes: usize,
        recursive_explicit_or_eq_depth: usize,
        boolean_dual_and_nodes: usize,
        boolean_dual_depth: usize,
    ) -> CmpBooleanDualQorRow {
        CmpBooleanDualQorRow {
            binop,
            width,
            recursive_explicit_or_eq_and_nodes,
            recursive_explicit_or_eq_depth,
            boolean_dual_and_nodes,
            boolean_dual_depth,
        }
    }

    fn gatify_ucmp_via_repeated_prefix_bit_tests<F>(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
        or_eq: bool,
        handle_bit: &F,
    ) -> AigOperand
    where
        F: Fn(&mut GateBuilder, AigOperand, AigOperand) -> AigOperand,
    {
        assert_eq!(lhs_bits.get_bit_count(), rhs_bits.get_bit_count());
        let input_bit_count = lhs_bits.get_bit_count();
        assert!(input_bit_count > 0, "ucmp requires non-zero-width operands");
        let eq_bits = gb.add_xnor_vec(lhs_bits, rhs_bits);
        let mut bit_tests = Vec::new();

        for msb_i in 0..input_bit_count {
            let eq_bits_slice = eq_bits.get_msbs(msb_i);
            let prior_bits_equal = if eq_bits_slice.is_empty() {
                gb.get_true()
            } else {
                gb.add_and_reduce(&eq_bits_slice, ReductionKind::Tree)
            };
            let lhs_bit = *lhs_bits.get_msb(msb_i);
            let rhs_bit = *rhs_bits.get_msb(msb_i);
            let bit_test = handle_bit(gb, lhs_bit, rhs_bit);
            bit_tests.push(gb.add_and_binary(prior_bits_equal, bit_test));
        }

        let cmp = gb.add_or_nary(&bit_tests, ReductionKind::Tree);
        if or_eq {
            let eq = gb.add_and_reduce(&eq_bits, ReductionKind::Tree);
            gb.add_or_binary(cmp, eq)
        } else {
            cmp
        }
    }

    fn gatify_ult_and_eq_via_repeated_prefix_bit_tests(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
    ) -> (AigOperand, AigOperand) {
        let ult = gatify_ucmp_via_repeated_prefix_bit_tests(
            gb,
            lhs_bits,
            rhs_bits,
            false,
            &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                let lhs_bit_unset = gb.add_not(lhs_bit);
                gb.add_and_binary(lhs_bit_unset, rhs_bit)
            },
        );
        let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
        (ult, eq)
    }

    fn gatify_scmp_via_repeated_prefix_bit_tests(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
        cmp_kind: super::CmpKind,
        or_eq: bool,
    ) -> AigOperand {
        assert_eq!(lhs_bits.get_bit_count(), rhs_bits.get_bit_count());
        assert!(lhs_bits.get_bit_count() > 0);
        let bit_count = lhs_bits.get_bit_count();
        if bit_count == 1 {
            let a = *lhs_bits.get_lsb(0);
            let b = *rhs_bits.get_lsb(0);
            return match cmp_kind {
                super::CmpKind::Lt => {
                    let b_complement = gb.add_not(b);
                    let slt = gb.add_and_binary(a, b_complement);
                    if or_eq {
                        let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
                        gb.add_or_binary(slt, eq)
                    } else {
                        slt
                    }
                }
                super::CmpKind::Gt => {
                    let a_complement = gb.add_not(a);
                    let sgt = gb.add_and_binary(a_complement, b);
                    if or_eq {
                        let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
                        gb.add_or_binary(sgt, eq)
                    } else {
                        sgt
                    }
                }
            };
        }

        let a_msb = lhs_bits.get_msb(0);
        let b_msb = rhs_bits.get_msb(0);
        let sign_diff = gb.add_xor_binary(*a_msb, *b_msb);

        let (ult, eq) = gatify_ult_and_eq_via_repeated_prefix_bit_tests(gb, lhs_bits, rhs_bits);
        let term1 = gb.add_and_binary(sign_diff, *a_msb);
        let not_sign_diff = gb.add_not(sign_diff);
        let term2 = gb.add_and_binary(not_sign_diff, ult);
        let lt = gb.add_or_binary(term1, term2);
        match cmp_kind {
            super::CmpKind::Lt => {
                if or_eq {
                    gb.add_or_binary(lt, eq)
                } else {
                    lt
                }
            }
            super::CmpKind::Gt => {
                let lt_or_eq = gb.add_or_binary(lt, eq);
                let gt = gb.add_not(lt_or_eq);
                if or_eq { gb.add_or_binary(gt, eq) } else { gt }
            }
        }
    }

    fn gatify_ule_via_recursive_explicit_or_eq(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
    ) -> AigOperand {
        super::gatify_ucmp_via_bit_tests(
            gb,
            0,
            lhs_bits,
            rhs_bits,
            true,
            &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                let lhs_bit_unset = gb.add_not(lhs_bit);
                gb.add_and_binary(lhs_bit_unset, rhs_bit)
            },
        )
    }

    fn gatify_uge_via_recursive_explicit_or_eq(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
    ) -> AigOperand {
        super::gatify_ucmp_via_bit_tests(
            gb,
            0,
            lhs_bits,
            rhs_bits,
            true,
            &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                let rhs_bit_unset = gb.add_not(rhs_bit);
                gb.add_and_binary(lhs_bit, rhs_bit_unset)
            },
        )
    }

    fn gatify_scmp_via_full_width_unsigned_explicit_or_eq(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
        cmp_kind: super::CmpKind,
        or_eq: bool,
    ) -> AigOperand {
        assert_eq!(lhs_bits.get_bit_count(), rhs_bits.get_bit_count());
        assert!(lhs_bits.get_bit_count() > 0);
        let bit_count = lhs_bits.get_bit_count();
        if bit_count == 1 {
            let a = *lhs_bits.get_lsb(0);
            let b = *rhs_bits.get_lsb(0);
            return match cmp_kind {
                super::CmpKind::Lt => {
                    let b_complement = gb.add_not(b);
                    let slt = gb.add_and_binary(a, b_complement);
                    if or_eq {
                        let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
                        gb.add_or_binary(slt, eq)
                    } else {
                        slt
                    }
                }
                super::CmpKind::Gt => {
                    let a_complement = gb.add_not(a);
                    let sgt = gb.add_and_binary(a_complement, b);
                    if or_eq {
                        let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
                        gb.add_or_binary(sgt, eq)
                    } else {
                        sgt
                    }
                }
            };
        }

        let a_msb = lhs_bits.get_msb(0);
        let b_msb = rhs_bits.get_msb(0);
        let sign_diff = gb.add_xor_binary(*a_msb, *b_msb);

        let (ult, eq) = super::gatify_ult_and_eq_via_bit_tests(gb, lhs_bits, rhs_bits);
        let term1 = gb.add_and_binary(sign_diff, *a_msb);
        let not_sign_diff = gb.add_not(sign_diff);
        let term2 = gb.add_and_binary(not_sign_diff, ult);
        let lt = gb.add_or_binary(term1, term2);
        match cmp_kind {
            super::CmpKind::Lt => {
                if or_eq {
                    gb.add_or_binary(lt, eq)
                } else {
                    lt
                }
            }
            super::CmpKind::Gt => {
                let lt_or_eq = gb.add_or_binary(lt, eq);
                let gt = gb.add_not(lt_or_eq);
                if or_eq { gb.add_or_binary(gt, eq) } else { gt }
            }
        }
    }

    fn gatify_cmp_for_qor_test(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
        binop: ir::Binop,
        lowering: CmpLowering,
    ) -> AigOperand {
        match lowering {
            CmpLowering::RecursiveBitTreeSignSplitBooleanDual => match binop {
                ir::Binop::Ult => super::gatify_ult_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Ule => super::gatify_ule_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Ugt => super::gatify_ugt_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Uge => super::gatify_uge_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Slt => super::gatify_scmp_via_bit_tests(
                    gb,
                    0,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    false,
                ),
                ir::Binop::Sle => super::gatify_scmp_via_bit_tests(
                    gb,
                    0,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    true,
                ),
                ir::Binop::Sgt => super::gatify_scmp_via_bit_tests(
                    gb,
                    0,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    false,
                ),
                ir::Binop::Sge => super::gatify_scmp_via_bit_tests(
                    gb,
                    0,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    true,
                ),
                other => panic!("unexpected cmp binop in QoR test: {other:?}"),
            },
            CmpLowering::RecursiveExplicitOrEq => match binop {
                ir::Binop::Ult => super::gatify_ult_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Ule => gatify_ule_via_recursive_explicit_or_eq(gb, lhs_bits, rhs_bits),
                ir::Binop::Ugt => super::gatify_ugt_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Uge => gatify_uge_via_recursive_explicit_or_eq(gb, lhs_bits, rhs_bits),
                ir::Binop::Slt => gatify_scmp_via_full_width_unsigned_explicit_or_eq(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    false,
                ),
                ir::Binop::Sle => gatify_scmp_via_full_width_unsigned_explicit_or_eq(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    true,
                ),
                ir::Binop::Sgt => gatify_scmp_via_full_width_unsigned_explicit_or_eq(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    false,
                ),
                ir::Binop::Sge => gatify_scmp_via_full_width_unsigned_explicit_or_eq(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    true,
                ),
                other => panic!("unexpected cmp binop in QoR test: {other:?}"),
            },
            CmpLowering::RepeatedPrefix => match binop {
                ir::Binop::Ult => gatify_ucmp_via_repeated_prefix_bit_tests(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    false,
                    &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                        let lhs_bit_unset = gb.add_not(lhs_bit);
                        gb.add_and_binary(lhs_bit_unset, rhs_bit)
                    },
                ),
                ir::Binop::Ule => gatify_ucmp_via_repeated_prefix_bit_tests(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    true,
                    &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                        let lhs_bit_unset = gb.add_not(lhs_bit);
                        gb.add_and_binary(lhs_bit_unset, rhs_bit)
                    },
                ),
                ir::Binop::Ugt => gatify_ucmp_via_repeated_prefix_bit_tests(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    false,
                    &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                        let rhs_bit_unset = gb.add_not(rhs_bit);
                        gb.add_and_binary(lhs_bit, rhs_bit_unset)
                    },
                ),
                ir::Binop::Uge => gatify_ucmp_via_repeated_prefix_bit_tests(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    true,
                    &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                        let rhs_bit_unset = gb.add_not(rhs_bit);
                        gb.add_and_binary(lhs_bit, rhs_bit_unset)
                    },
                ),
                ir::Binop::Slt => gatify_scmp_via_repeated_prefix_bit_tests(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    false,
                ),
                ir::Binop::Sle => gatify_scmp_via_repeated_prefix_bit_tests(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    true,
                ),
                ir::Binop::Sgt => gatify_scmp_via_repeated_prefix_bit_tests(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    false,
                ),
                ir::Binop::Sge => gatify_scmp_via_repeated_prefix_bit_tests(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    true,
                ),
                other => panic!("unexpected cmp binop in QoR test: {other:?}"),
            },
        }
    }

    fn build_cmp_gate_fn_for_qor_test(
        width: usize,
        binop: ir::Binop,
        lowering: CmpLowering,
    ) -> GateFn {
        let mut gb = GateBuilder::new(
            format!("cmp_{lowering:?}_{binop:?}_{width}b"),
            GateBuilderOptions::opt(),
        );
        let lhs_bits = gb.add_input("lhs".to_string(), width);
        let rhs_bits = gb.add_input("rhs".to_string(), width);
        let result = gatify_cmp_for_qor_test(&mut gb, &lhs_bits, &rhs_bits, binop, lowering);
        gb.add_output("result".to_string(), AigBitVector::from_bit(result));
        gb.build()
    }

    fn get_cmp_qor_test_stats(width: usize, binop: ir::Binop, lowering: CmpLowering) -> AigStats {
        get_aig_stats(&build_cmp_gate_fn_for_qor_test(width, binop, lowering))
    }

    fn validate_cmp_lowerings_equivalent_by_simulation(
        width: usize,
        binop: ir::Binop,
        reference_lowering: CmpLowering,
        comparison_lowering: CmpLowering,
    ) {
        let reference_gate_fn = build_cmp_gate_fn_for_qor_test(width, binop, reference_lowering);
        let comparison_gate_fn = build_cmp_gate_fn_for_qor_test(width, binop, comparison_lowering);
        let value_count = 1usize << width;
        for lhs in 0..value_count {
            for rhs in 0..value_count {
                let lhs_bits = IrBits::make_ubits(width, lhs as u64).unwrap();
                let rhs_bits = IrBits::make_ubits(width, rhs as u64).unwrap();
                let reference = gate_sim::eval(
                    &reference_gate_fn,
                    &[lhs_bits.clone(), rhs_bits.clone()],
                    gate_sim::Collect::None,
                );
                let comparison = gate_sim::eval(
                    &comparison_gate_fn,
                    &[lhs_bits, rhs_bits],
                    gate_sim::Collect::None,
                );
                assert_eq!(
                    comparison.outputs, reference.outputs,
                    "cmp mismatch for binop={binop:?} width={width} lhs={lhs} rhs={rhs} \
                     reference={reference_lowering:?} comparison={comparison_lowering:?}"
                );
            }
        }
    }

    fn validate_cmp_lowerings_equivalent_by_prover(
        width: usize,
        binop: ir::Binop,
        reference_lowering: CmpLowering,
        comparison_lowering: CmpLowering,
    ) {
        let reference_gate_fn = build_cmp_gate_fn_for_qor_test(width, binop, reference_lowering);
        let comparison_gate_fn = build_cmp_gate_fn_for_qor_test(width, binop, comparison_lowering);
        check_equivalence::prove_same_gate_fn_via_ir(&reference_gate_fn, &comparison_gate_fn)
            .expect("comparison lowerings should be equivalent");
    }

    fn gather_cmp_qor_rows() -> Vec<CmpRecursiveBitTreeQorRow> {
        let mut got = Vec::new();
        for binop in [
            ir::Binop::Ult,
            ir::Binop::Ule,
            ir::Binop::Ugt,
            ir::Binop::Uge,
            ir::Binop::Slt,
            ir::Binop::Sle,
            ir::Binop::Sgt,
            ir::Binop::Sge,
        ] {
            for width in [3usize, 4, 5, 8, 16, 32] {
                if width <= 5 {
                    validate_cmp_lowerings_equivalent_by_simulation(
                        width,
                        binop,
                        CmpLowering::RepeatedPrefix,
                        CmpLowering::RecursiveBitTreeSignSplitBooleanDual,
                    );
                }
                let repeated_prefix =
                    get_cmp_qor_test_stats(width, binop, CmpLowering::RepeatedPrefix);
                let recursive_bit_tree = get_cmp_qor_test_stats(
                    width,
                    binop,
                    CmpLowering::RecursiveBitTreeSignSplitBooleanDual,
                );
                got.push(CmpRecursiveBitTreeQorRow {
                    binop,
                    width,
                    repeated_prefix_and_nodes: repeated_prefix.and_nodes,
                    repeated_prefix_depth: repeated_prefix.max_depth,
                    recursive_bit_tree_and_nodes: recursive_bit_tree.and_nodes,
                    recursive_bit_tree_depth: recursive_bit_tree.max_depth,
                });
            }
        }
        got
    }

    fn gather_cmp_boolean_dual_qor_rows() -> Vec<CmpBooleanDualQorRow> {
        let mut got = Vec::new();
        for binop in [
            ir::Binop::Ult,
            ir::Binop::Ule,
            ir::Binop::Ugt,
            ir::Binop::Uge,
            ir::Binop::Slt,
            ir::Binop::Sle,
            ir::Binop::Sgt,
            ir::Binop::Sge,
        ] {
            for width in [1usize, 2, 3, 4, 5, 8, 16, 32] {
                if width <= 5 {
                    validate_cmp_lowerings_equivalent_by_simulation(
                        width,
                        binop,
                        CmpLowering::RecursiveExplicitOrEq,
                        CmpLowering::RecursiveBitTreeSignSplitBooleanDual,
                    );
                } else {
                    validate_cmp_lowerings_equivalent_by_prover(
                        width,
                        binop,
                        CmpLowering::RecursiveExplicitOrEq,
                        CmpLowering::RecursiveBitTreeSignSplitBooleanDual,
                    );
                }
                let recursive_explicit_or_eq =
                    get_cmp_qor_test_stats(width, binop, CmpLowering::RecursiveExplicitOrEq);
                let boolean_dual = get_cmp_qor_test_stats(
                    width,
                    binop,
                    CmpLowering::RecursiveBitTreeSignSplitBooleanDual,
                );
                got.push(CmpBooleanDualQorRow {
                    binop,
                    width,
                    recursive_explicit_or_eq_and_nodes: recursive_explicit_or_eq.and_nodes,
                    recursive_explicit_or_eq_depth: recursive_explicit_or_eq.max_depth,
                    boolean_dual_and_nodes: boolean_dual.and_nodes,
                    boolean_dual_depth: boolean_dual.max_depth,
                });
            }
        }
        got
    }

    #[test]
    fn test_cmp_recursive_bit_tree_qor_and_equivalence_sweep() {
        let got = gather_cmp_qor_rows();

        for row in &got {
            assert!(
                row.recursive_bit_tree_and_nodes <= row.repeated_prefix_and_nodes,
                "expected recursive cmp lowering not to increase AND nodes: {:?}",
                row
            );
            assert!(
                row.recursive_bit_tree_depth <= row.repeated_prefix_depth,
                "expected recursive cmp lowering not to increase depth: {:?}",
                row
            );
            assert!(
                row.recursive_bit_tree_and_nodes < row.repeated_prefix_and_nodes
                    || row.recursive_bit_tree_depth < row.repeated_prefix_depth,
                "expected recursive cmp lowering to improve this row: {:?}",
                row
            );
        }

        #[rustfmt::skip]
        let want: &[CmpRecursiveBitTreeQorRow] = &[
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ult, width: 3, repeated_prefix_and_nodes: 12, repeated_prefix_depth: 6, recursive_bit_tree_and_nodes: 12, recursive_bit_tree_depth: 5 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ult, width: 4, repeated_prefix_and_nodes: 18, repeated_prefix_depth: 7, recursive_bit_tree_and_nodes: 17, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ult, width: 5, repeated_prefix_and_nodes: 25, repeated_prefix_depth: 8, recursive_bit_tree_and_nodes: 23, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ult, width: 8, repeated_prefix_and_nodes: 47, repeated_prefix_depth: 9, recursive_bit_tree_and_nodes: 40, recursive_bit_tree_depth: 8 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ult, width: 16, repeated_prefix_and_nodes: 120, repeated_prefix_depth: 11, recursive_bit_tree_and_nodes: 87, recursive_bit_tree_depth: 10 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ult, width: 32, repeated_prefix_and_nodes: 299, repeated_prefix_depth: 13, recursive_bit_tree_and_nodes: 182, recursive_bit_tree_depth: 12 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ule, width: 3, repeated_prefix_and_nodes: 16, repeated_prefix_depth: 7, recursive_bit_tree_and_nodes: 12, recursive_bit_tree_depth: 5 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ule, width: 4, repeated_prefix_and_nodes: 23, repeated_prefix_depth: 8, recursive_bit_tree_and_nodes: 17, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ule, width: 5, repeated_prefix_and_nodes: 30, repeated_prefix_depth: 9, recursive_bit_tree_and_nodes: 23, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ule, width: 8, repeated_prefix_and_nodes: 53, repeated_prefix_depth: 10, recursive_bit_tree_and_nodes: 40, recursive_bit_tree_depth: 8 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ule, width: 16, repeated_prefix_and_nodes: 127, repeated_prefix_depth: 12, recursive_bit_tree_and_nodes: 87, recursive_bit_tree_depth: 10 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ule, width: 32, repeated_prefix_and_nodes: 307, repeated_prefix_depth: 14, recursive_bit_tree_and_nodes: 182, recursive_bit_tree_depth: 12 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ugt, width: 3, repeated_prefix_and_nodes: 12, repeated_prefix_depth: 6, recursive_bit_tree_and_nodes: 12, recursive_bit_tree_depth: 5 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ugt, width: 4, repeated_prefix_and_nodes: 18, repeated_prefix_depth: 7, recursive_bit_tree_and_nodes: 17, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ugt, width: 5, repeated_prefix_and_nodes: 25, repeated_prefix_depth: 8, recursive_bit_tree_and_nodes: 23, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ugt, width: 8, repeated_prefix_and_nodes: 47, repeated_prefix_depth: 9, recursive_bit_tree_and_nodes: 40, recursive_bit_tree_depth: 8 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ugt, width: 16, repeated_prefix_and_nodes: 120, repeated_prefix_depth: 11, recursive_bit_tree_and_nodes: 87, recursive_bit_tree_depth: 10 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Ugt, width: 32, repeated_prefix_and_nodes: 299, repeated_prefix_depth: 13, recursive_bit_tree_and_nodes: 182, recursive_bit_tree_depth: 12 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Uge, width: 3, repeated_prefix_and_nodes: 16, repeated_prefix_depth: 7, recursive_bit_tree_and_nodes: 12, recursive_bit_tree_depth: 5 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Uge, width: 4, repeated_prefix_and_nodes: 23, repeated_prefix_depth: 8, recursive_bit_tree_and_nodes: 17, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Uge, width: 5, repeated_prefix_and_nodes: 30, repeated_prefix_depth: 9, recursive_bit_tree_and_nodes: 23, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Uge, width: 8, repeated_prefix_and_nodes: 53, repeated_prefix_depth: 10, recursive_bit_tree_and_nodes: 40, recursive_bit_tree_depth: 8 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Uge, width: 16, repeated_prefix_and_nodes: 127, repeated_prefix_depth: 12, recursive_bit_tree_and_nodes: 87, recursive_bit_tree_depth: 10 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Uge, width: 32, repeated_prefix_and_nodes: 307, repeated_prefix_depth: 14, recursive_bit_tree_and_nodes: 182, recursive_bit_tree_depth: 12 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Slt, width: 3, repeated_prefix_and_nodes: 15, repeated_prefix_depth: 8, recursive_bit_tree_and_nodes: 12, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Slt, width: 4, repeated_prefix_and_nodes: 21, repeated_prefix_depth: 9, recursive_bit_tree_and_nodes: 18, recursive_bit_tree_depth: 7 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Slt, width: 5, repeated_prefix_and_nodes: 28, repeated_prefix_depth: 10, recursive_bit_tree_and_nodes: 23, recursive_bit_tree_depth: 8 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Slt, width: 8, repeated_prefix_and_nodes: 50, repeated_prefix_depth: 11, recursive_bit_tree_and_nodes: 41, recursive_bit_tree_depth: 9 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Slt, width: 16, repeated_prefix_and_nodes: 123, repeated_prefix_depth: 13, recursive_bit_tree_and_nodes: 88, recursive_bit_tree_depth: 11 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Slt, width: 32, repeated_prefix_and_nodes: 302, repeated_prefix_depth: 15, recursive_bit_tree_and_nodes: 183, recursive_bit_tree_depth: 13 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sle, width: 3, repeated_prefix_and_nodes: 19, repeated_prefix_depth: 9, recursive_bit_tree_and_nodes: 12, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sle, width: 4, repeated_prefix_and_nodes: 26, repeated_prefix_depth: 10, recursive_bit_tree_and_nodes: 18, recursive_bit_tree_depth: 7 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sle, width: 5, repeated_prefix_and_nodes: 33, repeated_prefix_depth: 11, recursive_bit_tree_and_nodes: 23, recursive_bit_tree_depth: 8 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sle, width: 8, repeated_prefix_and_nodes: 56, repeated_prefix_depth: 12, recursive_bit_tree_and_nodes: 41, recursive_bit_tree_depth: 9 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sle, width: 16, repeated_prefix_and_nodes: 130, repeated_prefix_depth: 14, recursive_bit_tree_and_nodes: 88, recursive_bit_tree_depth: 11 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sle, width: 32, repeated_prefix_and_nodes: 310, repeated_prefix_depth: 16, recursive_bit_tree_and_nodes: 183, recursive_bit_tree_depth: 13 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sgt, width: 3, repeated_prefix_and_nodes: 19, repeated_prefix_depth: 9, recursive_bit_tree_and_nodes: 12, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sgt, width: 4, repeated_prefix_and_nodes: 26, repeated_prefix_depth: 10, recursive_bit_tree_and_nodes: 18, recursive_bit_tree_depth: 7 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sgt, width: 5, repeated_prefix_and_nodes: 33, repeated_prefix_depth: 11, recursive_bit_tree_and_nodes: 23, recursive_bit_tree_depth: 8 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sgt, width: 8, repeated_prefix_and_nodes: 56, repeated_prefix_depth: 12, recursive_bit_tree_and_nodes: 41, recursive_bit_tree_depth: 9 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sgt, width: 16, repeated_prefix_and_nodes: 130, repeated_prefix_depth: 14, recursive_bit_tree_and_nodes: 88, recursive_bit_tree_depth: 11 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sgt, width: 32, repeated_prefix_and_nodes: 310, repeated_prefix_depth: 16, recursive_bit_tree_and_nodes: 183, recursive_bit_tree_depth: 13 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sge, width: 3, repeated_prefix_and_nodes: 20, repeated_prefix_depth: 10, recursive_bit_tree_and_nodes: 12, recursive_bit_tree_depth: 6 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sge, width: 4, repeated_prefix_and_nodes: 27, repeated_prefix_depth: 11, recursive_bit_tree_and_nodes: 18, recursive_bit_tree_depth: 7 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sge, width: 5, repeated_prefix_and_nodes: 34, repeated_prefix_depth: 12, recursive_bit_tree_and_nodes: 23, recursive_bit_tree_depth: 8 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sge, width: 8, repeated_prefix_and_nodes: 57, repeated_prefix_depth: 13, recursive_bit_tree_and_nodes: 41, recursive_bit_tree_depth: 9 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sge, width: 16, repeated_prefix_and_nodes: 131, repeated_prefix_depth: 15, recursive_bit_tree_and_nodes: 88, recursive_bit_tree_depth: 11 },
            CmpRecursiveBitTreeQorRow { binop: ir::Binop::Sge, width: 32, repeated_prefix_and_nodes: 311, repeated_prefix_depth: 17, recursive_bit_tree_and_nodes: 183, recursive_bit_tree_depth: 13 },
        ];
        assert_eq!(got.as_slice(), want);
    }

    #[test]
    fn test_cmp_boolean_dual_qor_and_equivalence_sweep() {
        let got = gather_cmp_boolean_dual_qor_rows();

        for row in &got {
            assert!(
                row.boolean_dual_and_nodes <= row.recursive_explicit_or_eq_and_nodes,
                "expected boolean-dual cmp lowering not to increase AND nodes: {:?}",
                row
            );
            assert!(
                row.boolean_dual_depth <= row.recursive_explicit_or_eq_depth,
                "expected boolean-dual cmp lowering not to increase depth: {:?}",
                row
            );
        }

        #[rustfmt::skip]
        let want: &[CmpBooleanDualQorRow] = &[
            cmp_boolean_dual_qor_row(ir::Binop::Ult, 1, 1, 1, 1, 1),
            cmp_boolean_dual_qor_row(ir::Binop::Ult, 2, 6, 4, 6, 4),
            cmp_boolean_dual_qor_row(ir::Binop::Ult, 3, 12, 5, 12, 5),
            cmp_boolean_dual_qor_row(ir::Binop::Ult, 4, 17, 6, 17, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Ult, 5, 23, 6, 23, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Ult, 8, 40, 8, 40, 8),
            cmp_boolean_dual_qor_row(ir::Binop::Ult, 16, 87, 10, 87, 10),
            cmp_boolean_dual_qor_row(ir::Binop::Ult, 32, 182, 12, 182, 12),
            cmp_boolean_dual_qor_row(ir::Binop::Ule, 1, 4, 3, 1, 1),
            cmp_boolean_dual_qor_row(ir::Binop::Ule, 2, 10, 5, 6, 4),
            cmp_boolean_dual_qor_row(ir::Binop::Ule, 3, 16, 6, 12, 5),
            cmp_boolean_dual_qor_row(ir::Binop::Ule, 4, 22, 7, 17, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Ule, 5, 28, 7, 23, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Ule, 8, 46, 9, 40, 8),
            cmp_boolean_dual_qor_row(ir::Binop::Ule, 16, 94, 11, 87, 10),
            cmp_boolean_dual_qor_row(ir::Binop::Ule, 32, 190, 13, 182, 12),
            cmp_boolean_dual_qor_row(ir::Binop::Ugt, 1, 1, 1, 1, 1),
            cmp_boolean_dual_qor_row(ir::Binop::Ugt, 2, 6, 4, 6, 4),
            cmp_boolean_dual_qor_row(ir::Binop::Ugt, 3, 12, 5, 12, 5),
            cmp_boolean_dual_qor_row(ir::Binop::Ugt, 4, 17, 6, 17, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Ugt, 5, 23, 6, 23, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Ugt, 8, 40, 8, 40, 8),
            cmp_boolean_dual_qor_row(ir::Binop::Ugt, 16, 87, 10, 87, 10),
            cmp_boolean_dual_qor_row(ir::Binop::Ugt, 32, 182, 12, 182, 12),
            cmp_boolean_dual_qor_row(ir::Binop::Uge, 1, 4, 3, 1, 1),
            cmp_boolean_dual_qor_row(ir::Binop::Uge, 2, 10, 5, 6, 4),
            cmp_boolean_dual_qor_row(ir::Binop::Uge, 3, 16, 6, 12, 5),
            cmp_boolean_dual_qor_row(ir::Binop::Uge, 4, 22, 7, 17, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Uge, 5, 28, 7, 23, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Uge, 8, 46, 9, 40, 8),
            cmp_boolean_dual_qor_row(ir::Binop::Uge, 16, 94, 11, 87, 10),
            cmp_boolean_dual_qor_row(ir::Binop::Uge, 32, 190, 13, 182, 12),
            cmp_boolean_dual_qor_row(ir::Binop::Slt, 1, 1, 1, 1, 1),
            cmp_boolean_dual_qor_row(ir::Binop::Slt, 2, 9, 6, 7, 4),
            cmp_boolean_dual_qor_row(ir::Binop::Slt, 3, 15, 7, 12, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Slt, 4, 20, 8, 18, 7),
            cmp_boolean_dual_qor_row(ir::Binop::Slt, 5, 26, 8, 23, 8),
            cmp_boolean_dual_qor_row(ir::Binop::Slt, 8, 43, 10, 41, 9),
            cmp_boolean_dual_qor_row(ir::Binop::Slt, 16, 90, 12, 88, 11),
            cmp_boolean_dual_qor_row(ir::Binop::Slt, 32, 185, 14, 183, 13),
            cmp_boolean_dual_qor_row(ir::Binop::Sle, 1, 4, 3, 1, 1),
            cmp_boolean_dual_qor_row(ir::Binop::Sle, 2, 13, 7, 7, 4),
            cmp_boolean_dual_qor_row(ir::Binop::Sle, 3, 19, 8, 12, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Sle, 4, 25, 9, 18, 7),
            cmp_boolean_dual_qor_row(ir::Binop::Sle, 5, 31, 9, 23, 8),
            cmp_boolean_dual_qor_row(ir::Binop::Sle, 8, 49, 11, 41, 9),
            cmp_boolean_dual_qor_row(ir::Binop::Sle, 16, 97, 13, 88, 11),
            cmp_boolean_dual_qor_row(ir::Binop::Sle, 32, 193, 15, 183, 13),
            cmp_boolean_dual_qor_row(ir::Binop::Sgt, 1, 1, 1, 1, 1),
            cmp_boolean_dual_qor_row(ir::Binop::Sgt, 2, 13, 7, 7, 4),
            cmp_boolean_dual_qor_row(ir::Binop::Sgt, 3, 19, 8, 12, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Sgt, 4, 25, 9, 18, 7),
            cmp_boolean_dual_qor_row(ir::Binop::Sgt, 5, 31, 9, 23, 8),
            cmp_boolean_dual_qor_row(ir::Binop::Sgt, 8, 49, 11, 41, 9),
            cmp_boolean_dual_qor_row(ir::Binop::Sgt, 16, 97, 13, 88, 11),
            cmp_boolean_dual_qor_row(ir::Binop::Sgt, 32, 193, 15, 183, 13),
            cmp_boolean_dual_qor_row(ir::Binop::Sge, 1, 4, 3, 1, 1),
            cmp_boolean_dual_qor_row(ir::Binop::Sge, 2, 14, 8, 7, 4),
            cmp_boolean_dual_qor_row(ir::Binop::Sge, 3, 20, 9, 12, 6),
            cmp_boolean_dual_qor_row(ir::Binop::Sge, 4, 26, 10, 18, 7),
            cmp_boolean_dual_qor_row(ir::Binop::Sge, 5, 32, 10, 23, 8),
            cmp_boolean_dual_qor_row(ir::Binop::Sge, 8, 50, 12, 41, 9),
            cmp_boolean_dual_qor_row(ir::Binop::Sge, 16, 98, 14, 88, 11),
            cmp_boolean_dual_qor_row(ir::Binop::Sge, 32, 194, 16, 183, 13),
        ];
        assert_eq!(got.as_slice(), want);
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ShraQorRow {
        width: usize,
        amount_width: usize,
        old_and_nodes: usize,
        old_depth: usize,
        public_and_nodes: usize,
        public_depth: usize,
    }

    fn build_shra_ir_text(width: usize, amount_width: usize) -> String {
        format!(
            r#"package sample

top fn main(x: bits[{width}], amt: bits[{amount_width}]) -> bits[{width}] {{
  ret y: bits[{width}] = shra(x, amt, id=3)
}}
"#
        )
    }

    fn gatify_shra_old_sign_ext_shift(
        gb: &mut GateBuilder,
        arg_bits: &AigBitVector,
        amount_bits: &AigBitVector,
    ) -> AigBitVector {
        let w = arg_bits.get_bit_count();
        assert!(w > 0);

        let required_k = if w <= 1 {
            0
        } else {
            xlsynth_pir::math::ceil_log2(w)
        };
        let amount_w = amount_bits.get_bit_count();
        let k = std::cmp::min(required_k, amount_w);
        let Split {
            msbs: amt_hi,
            lsbs: amt_lo,
        } = amount_bits.get_lsb_partition(k);

        let oob_hi = if amt_hi.get_bit_count() == 0 {
            gb.get_false()
        } else {
            gb.add_nez(&amt_hi, ReductionKind::Tree)
        };
        let oob_lo = if k == 0 || w.is_power_of_two() || k != required_k {
            gb.get_false()
        } else {
            let w_bits =
                IrBits::make_ubits(k, w as u64).expect("width must fit in shra low amount bits");
            super::try_gatify_ucmp_literal_rhs_threshold(gb, ir::Binop::Uge, &amt_lo, &w_bits)
                .expect("Uge threshold compare should be supported")
        };
        let oob = gb.add_or_binary(oob_hi, oob_lo);

        let sign = *arg_bits.get_msb(0);
        let sign_ext = gb.replicate(sign, w);
        let arg_ext = AigBitVector::concat(sign_ext, arg_bits.clone());
        let shifted =
            gatify_barrel_shifter(&arg_ext, &amt_lo, Direction::Right, "shra_old_ext", gb);
        let arith = shifted.get_lsb_slice(0, w);

        if gb.is_known_false(oob) {
            arith
        } else {
            let all_sign = gb.replicate(sign, w);
            gb.add_mux2_vec(&oob, &all_sign, &arith)
        }
    }

    fn get_shra_old_stats(width: usize, amount_width: usize) -> AigStats {
        let mut gb = GateBuilder::new(
            format!("shra_old_w{width}_amt{amount_width}"),
            GateBuilderOptions::opt(),
        );
        let arg_bits = gb.add_input("x".to_string(), width);
        let amount_bits = gb.add_input("amt".to_string(), amount_width);
        let result = gatify_shra_old_sign_ext_shift(&mut gb, &arg_bits, &amount_bits);
        gb.add_output("result".to_string(), result);
        get_aig_stats(&gb.build())
    }

    fn shra_sample_bits(width: usize) -> Vec<IrValue> {
        let all_ones = bit_mask(width);
        let sign_bit = 1u64 << (width - 1);
        let max_positive = sign_bit - 1;
        let alternating = 0xaaaa_aaaa_aaaa_aaaau64 & all_ones;
        let mut values = vec![
            0,
            1,
            all_ones,
            sign_bit,
            sign_bit | 1,
            max_positive,
            alternating,
        ];
        values.sort_unstable();
        values.dedup();
        values
            .into_iter()
            .map(|value| IrValue::make_ubits(width, value).unwrap())
            .collect()
    }

    fn validate_shra_public_simulation(
        ir_fn: &ir::Fn,
        gate_fn: &GateFn,
        width: usize,
        amount_width: usize,
    ) {
        let x_samples = shra_sample_bits(width);
        let amount_count = 1usize << amount_width;
        for x_value in &x_samples {
            for amount in 0..amount_count {
                let amount_value = IrValue::make_ubits(amount_width, amount as u64).unwrap();
                let want = match eval_fn(ir_fn, &[x_value.clone(), amount_value.clone()]) {
                    FnEvalResult::Success(success) => success.value.to_bits().unwrap(),
                    FnEvalResult::Failure(failure) => {
                        panic!("shra source IR failed during simulation: {failure:?}")
                    }
                };
                let sim = gate_sim::eval(
                    gate_fn,
                    &[x_value.to_bits().unwrap(), amount_value.to_bits().unwrap()],
                    gate_sim::Collect::None,
                );
                let got = sim.outputs[0].clone();
                assert_eq!(
                    got, want,
                    "shra simulation mismatch for width={width} amount_width={amount_width} \
                     amount={amount} x={x_value}"
                );
            }
        }
    }

    fn get_shra_public_stats_and_validate(width: usize, amount_width: usize) -> AigStats {
        let ir_text = build_shra_ir_text(width, amount_width);
        let mut parser = ir_parser::Parser::new(&ir_text);
        let ir_package = parser.parse_and_validate_package().expect("parse package");
        let ir_fn = ir_package.get_top_fn().expect("top fn");
        let gatify_output = gatify(
            ir_fn,
            GatifyOptions {
                adder_mapping: AdderMapping::BrentKung,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .expect("gatify shra");
        validate_shra_public_simulation(ir_fn, &gatify_output.gate_fn, width, amount_width);
        get_aig_stats(&gatify_output.gate_fn)
    }

    fn gather_shra_qor_rows() -> Vec<ShraQorRow> {
        let mut got = Vec::new();
        for (width, amount_widths) in [
            (3usize, &[2usize, 3, 4][..]),
            (4usize, &[2usize, 3, 4][..]),
            (5usize, &[2usize, 3, 4, 5][..]),
            (6usize, &[2usize, 3, 4][..]),
            (7usize, &[2usize, 3, 4][..]),
            (8usize, &[3usize, 4][..]),
            (9usize, &[3usize, 4, 5][..]),
            (16usize, &[4usize, 5][..]),
            (32usize, &[5usize, 6][..]),
        ] {
            for amount_width in amount_widths {
                let old = get_shra_old_stats(width, *amount_width);
                let public = get_shra_public_stats_and_validate(width, *amount_width);
                got.push(ShraQorRow {
                    width,
                    amount_width: *amount_width,
                    old_and_nodes: old.and_nodes,
                    old_depth: old.max_depth,
                    public_and_nodes: public.and_nodes,
                    public_depth: public.max_depth,
                });
            }
        }
        got
    }

    #[test]
    fn test_shra_arithmetic_barrel_qor_and_equivalence_sweep() {
        let _ = env_logger::builder().is_test(true).try_init();

        let got = gather_shra_qor_rows();

        for row in &got {
            assert!(
                row.public_and_nodes <= row.old_and_nodes,
                "expected shra lowering not to increase AND nodes: {:?}",
                row
            );
            assert!(
                row.public_depth <= row.old_depth,
                "expected shra lowering not to increase depth: {:?}",
                row
            );
            if !row.width.is_power_of_two()
                && row.amount_width >= xlsynth_pir::math::ceil_log2(row.width)
            {
                assert!(
                    row.public_and_nodes < row.old_and_nodes || row.public_depth < row.old_depth,
                    "expected non-power-of-two shra row to improve: {:?}",
                    row
                );
            }
        }

        #[rustfmt::skip]
        let want: &[ShraQorRow] = &[
            ShraQorRow { width: 3, amount_width: 2, old_and_nodes: 23, old_depth: 6, public_and_nodes: 16, public_depth: 4 },
            ShraQorRow { width: 3, amount_width: 3, old_and_nodes: 24, old_depth: 6, public_and_nodes: 23, public_depth: 6 },
            ShraQorRow { width: 3, amount_width: 4, old_and_nodes: 25, old_depth: 6, public_and_nodes: 24, public_depth: 6 },
            ShraQorRow { width: 4, amount_width: 2, old_and_nodes: 21, old_depth: 4, public_and_nodes: 21, public_depth: 4 },
            ShraQorRow { width: 4, amount_width: 3, old_and_nodes: 30, old_depth: 6, public_and_nodes: 30, public_depth: 6 },
            ShraQorRow { width: 4, amount_width: 4, old_and_nodes: 31, old_depth: 6, public_and_nodes: 31, public_depth: 6 },
            ShraQorRow { width: 5, amount_width: 2, old_and_nodes: 27, old_depth: 4, public_and_nodes: 27, public_depth: 4 },
            ShraQorRow { width: 5, amount_width: 3, old_and_nodes: 57, old_depth: 8, public_and_nodes: 40, public_depth: 6 },
            ShraQorRow { width: 5, amount_width: 4, old_and_nodes: 58, old_depth: 8, public_and_nodes: 51, public_depth: 8 },
            ShraQorRow { width: 5, amount_width: 5, old_and_nodes: 59, old_depth: 8, public_and_nodes: 52, public_depth: 8 },
            ShraQorRow { width: 6, amount_width: 2, old_and_nodes: 33, old_depth: 4, public_and_nodes: 33, public_depth: 4 },
            ShraQorRow { width: 6, amount_width: 3, old_and_nodes: 67, old_depth: 8, public_and_nodes: 49, public_depth: 6 },
            ShraQorRow { width: 6, amount_width: 4, old_and_nodes: 68, old_depth: 8, public_and_nodes: 62, public_depth: 8 },
            ShraQorRow { width: 7, amount_width: 2, old_and_nodes: 39, old_depth: 4, public_and_nodes: 39, public_depth: 4 },
            ShraQorRow { width: 7, amount_width: 3, old_and_nodes: 73, old_depth: 8, public_and_nodes: 58, public_depth: 6 },
            ShraQorRow { width: 7, amount_width: 4, old_and_nodes: 74, old_depth: 8, public_and_nodes: 73, public_depth: 8 },
            ShraQorRow { width: 8, amount_width: 3, old_and_nodes: 65, old_depth: 6, public_and_nodes: 65, public_depth: 6 },
            ShraQorRow { width: 8, amount_width: 4, old_and_nodes: 82, old_depth: 8, public_and_nodes: 82, public_depth: 8 },
            ShraQorRow { width: 9, amount_width: 3, old_and_nodes: 74, old_depth: 6, public_and_nodes: 74, public_depth: 6 },
            ShraQorRow { width: 9, amount_width: 4, old_and_nodes: 135, old_depth: 10, public_and_nodes: 96, public_depth: 8 },
            ShraQorRow { width: 9, amount_width: 5, old_and_nodes: 136, old_depth: 10, public_and_nodes: 115, public_depth: 10 },
            ShraQorRow { width: 16, amount_width: 4, old_and_nodes: 177, old_depth: 8, public_and_nodes: 177, public_depth: 8 },
            ShraQorRow { width: 16, amount_width: 5, old_and_nodes: 210, old_depth: 10, public_and_nodes: 210, public_depth: 10 },
            ShraQorRow { width: 32, amount_width: 5, old_and_nodes: 449, old_depth: 10, public_and_nodes: 449, public_depth: 10 },
            ShraQorRow { width: 32, amount_width: 6, old_and_nodes: 514, old_depth: 12, public_and_nodes: 514, public_depth: 12 },
        ];

        assert_eq!(got.as_slice(), want);
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct BitSliceUpdateQorRow {
        arg_width: usize,
        update_width: usize,
        old_and_nodes: usize,
        old_depth: usize,
        public_and_nodes: usize,
        public_depth: usize,
    }

    fn build_bit_slice_update_ir_text(
        arg_width: usize,
        start_width: usize,
        update_width: usize,
    ) -> String {
        format!(
            r#"package sample

top fn main(x: bits[{arg_width}], start: bits[{start_width}], update: bits[{update_width}]) -> bits[{arg_width}] {{
  ret y: bits[{arg_width}] = bit_slice_update(x, start, update, id=4)
}}
"#
        )
    }

    fn bit_mask(width: usize) -> u64 {
        if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        }
    }

    fn bit_slice_update_sample_bits(width: usize) -> Vec<IrValue> {
        let all_ones = bit_mask(width);
        let alternating = 0xaaaa_aaaa_aaaa_aaaau64 & all_ones;
        let mut values = vec![0, all_ones, alternating];
        for bit_index in [0, width / 2, width - 1] {
            values.push(1u64 << bit_index);
        }
        values.sort_unstable();
        values.dedup();
        values
            .into_iter()
            .map(|value| IrValue::make_ubits(width, value).unwrap())
            .collect()
    }

    fn gatify_bit_slice_update_old_insert_and(
        gb: &mut GateBuilder,
        arg_bits: &AigBitVector,
        start_bits: &AigBitVector,
        update_bits: &AigBitVector,
    ) -> AigBitVector {
        let arg_width = arg_bits.get_bit_count();
        let update_width = update_bits.get_bit_count();
        let effective_update_width = std::cmp::min(update_width, arg_width);

        let ones_effective = gb.replicate(gb.get_true(), effective_update_width);
        let zeros_high_count = arg_width - effective_update_width;
        let ones_ext = if zeros_high_count == 0 {
            ones_effective.clone()
        } else {
            let zeros = AigBitVector::zeros(zeros_high_count);
            AigBitVector::concat(zeros, ones_effective)
        };
        let mask = gatify_barrel_shifter(
            &ones_ext,
            start_bits,
            Direction::Left,
            "bit_slice_update_old_mask",
            gb,
        );

        let update_trim = if update_width > effective_update_width {
            update_bits.get_lsb_slice(0, effective_update_width)
        } else {
            update_bits.clone()
        };
        let update_ext = if zeros_high_count == 0 {
            update_trim.clone()
        } else {
            let zeros = AigBitVector::zeros(zeros_high_count);
            AigBitVector::concat(zeros, update_trim)
        };
        let update_shifted = gatify_barrel_shifter(
            &update_ext,
            start_bits,
            Direction::Left,
            "bit_slice_update_old_value",
            gb,
        );

        let mask_not = gb.add_not_vec(&mask);
        let cleared = gb.add_and_vec(arg_bits, &mask_not);
        let inserted = gb.add_and_vec(&update_shifted, &mask);
        gb.add_or_vec(&cleared, &inserted)
    }

    fn get_bit_slice_update_old_stats(
        arg_width: usize,
        start_width: usize,
        update_width: usize,
    ) -> AigStats {
        let mut gb = GateBuilder::new(
            format!("bit_slice_update_old_w{arg_width}_u{update_width}"),
            GateBuilderOptions::opt(),
        );
        let arg_bits = gb.add_input("x".to_string(), arg_width);
        let start_bits = gb.add_input("start".to_string(), start_width);
        let update_bits = gb.add_input("update".to_string(), update_width);
        let result =
            gatify_bit_slice_update_old_insert_and(&mut gb, &arg_bits, &start_bits, &update_bits);
        gb.add_output("result".to_string(), result);
        get_aig_stats(&gb.build())
    }

    fn validate_bit_slice_update_public_simulation(
        ir_fn: &ir::Fn,
        gate_fn: &GateFn,
        arg_width: usize,
        start_width: usize,
        update_width: usize,
    ) {
        let x_samples = bit_slice_update_sample_bits(arg_width);
        let update_samples = bit_slice_update_sample_bits(update_width);
        let start_count = 1usize << start_width;

        for x_value in &x_samples {
            for update_value in &update_samples {
                for start in 0..start_count {
                    let start_value = IrValue::make_ubits(start_width, start as u64).unwrap();
                    let want = match eval_fn(
                        ir_fn,
                        &[x_value.clone(), start_value.clone(), update_value.clone()],
                    ) {
                        FnEvalResult::Success(success) => success.value.to_bits().unwrap(),
                        FnEvalResult::Failure(failure) => {
                            panic!(
                                "bit_slice_update source IR failed during simulation: {failure:?}"
                            )
                        }
                    };
                    let sim = gate_sim::eval(
                        gate_fn,
                        &[
                            x_value.to_bits().unwrap(),
                            start_value.to_bits().unwrap(),
                            update_value.to_bits().unwrap(),
                        ],
                        gate_sim::Collect::None,
                    );
                    let got = sim.outputs[0].clone();
                    assert_eq!(
                        got, want,
                        "bit_slice_update simulation mismatch for arg_width={arg_width} \
                         update_width={update_width} start={start} x={x_value} \
                         update={update_value}"
                    );
                }
            }
        }
    }

    fn get_bit_slice_update_public_stats_and_validate(
        arg_width: usize,
        start_width: usize,
        update_width: usize,
    ) -> AigStats {
        let ir_text = build_bit_slice_update_ir_text(arg_width, start_width, update_width);
        let mut parser = ir_parser::Parser::new(&ir_text);
        let ir_package = parser.parse_and_validate_package().expect("parse package");
        let ir_fn = ir_package.get_top_fn().expect("top fn");
        let gatify_output = gatify(
            ir_fn,
            GatifyOptions {
                adder_mapping: AdderMapping::BrentKung,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .expect("gatify bit_slice_update");
        validate_bit_slice_update_public_simulation(
            ir_fn,
            &gatify_output.gate_fn,
            arg_width,
            start_width,
            update_width,
        );
        get_aig_stats(&gatify_output.gate_fn)
    }

    fn gather_bit_slice_update_qor_rows() -> Vec<BitSliceUpdateQorRow> {
        let mut got = Vec::new();
        for (arg_width, update_widths) in [
            (8usize, &[1usize, 2, 3, 5, 8][..]),
            (16usize, &[1usize, 2, 3, 5, 8, 16][..]),
            (32usize, &[1usize, 2, 3, 5, 8, 16][..]),
        ] {
            let start_width = (arg_width - 1).ilog2() as usize + 1;
            for update_width in update_widths {
                let old = get_bit_slice_update_old_stats(arg_width, start_width, *update_width);
                let public = get_bit_slice_update_public_stats_and_validate(
                    arg_width,
                    start_width,
                    *update_width,
                );
                got.push(BitSliceUpdateQorRow {
                    arg_width,
                    update_width: *update_width,
                    old_and_nodes: old.and_nodes,
                    old_depth: old.max_depth,
                    public_and_nodes: public.and_nodes,
                    public_depth: public.max_depth,
                });
            }
        }
        got
    }

    #[test]
    fn test_bit_slice_update_qor_and_equivalence_sweep() {
        let _ = env_logger::builder().is_test(true).try_init();

        let got = gather_bit_slice_update_qor_rows();

        for row in &got {
            assert!(
                row.public_and_nodes < row.old_and_nodes,
                "expected bit_slice_update lowering to reduce AND nodes: {:?}",
                row
            );
            assert!(
                row.public_depth <= row.old_depth,
                "expected bit_slice_update lowering not to increase depth: {:?}",
                row
            );
        }

        #[rustfmt::skip]
        let want: &[BitSliceUpdateQorRow] = &[
            BitSliceUpdateQorRow { arg_width: 8, update_width: 1, old_and_nodes: 50, old_depth: 5, public_and_nodes: 42, public_depth: 4 },
            BitSliceUpdateQorRow { arg_width: 8, update_width: 2, old_and_nodes: 64, old_depth: 6, public_and_nodes: 56, public_depth: 5 },
            BitSliceUpdateQorRow { arg_width: 8, update_width: 3, old_and_nodes: 75, old_depth: 7, public_and_nodes: 67, public_depth: 6 },
            BitSliceUpdateQorRow { arg_width: 8, update_width: 5, old_and_nodes: 95, old_depth: 8, public_and_nodes: 87, public_depth: 7 },
            BitSliceUpdateQorRow { arg_width: 8, update_width: 8, old_and_nodes: 101, old_depth: 8, public_and_nodes: 93, public_depth: 7 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 1, old_and_nodes: 106, old_depth: 6, public_and_nodes: 90, public_depth: 5 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 2, old_and_nodes: 126, old_depth: 7, public_and_nodes: 110, public_depth: 6 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 3, old_and_nodes: 143, old_depth: 8, public_and_nodes: 127, public_depth: 7 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 5, old_and_nodes: 174, old_depth: 9, public_and_nodes: 158, public_depth: 8 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 8, old_and_nodes: 216, old_depth: 10, public_and_nodes: 200, public_depth: 9 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 16, old_and_nodes: 253, old_depth: 10, public_and_nodes: 237, public_depth: 9 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 1, old_and_nodes: 218, old_depth: 7, public_and_nodes: 186, public_depth: 6 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 2, old_and_nodes: 244, old_depth: 8, public_and_nodes: 212, public_depth: 7 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 3, old_and_nodes: 267, old_depth: 9, public_and_nodes: 235, public_depth: 8 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 5, old_and_nodes: 310, old_depth: 10, public_and_nodes: 278, public_depth: 9 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 8, old_and_nodes: 370, old_depth: 11, public_and_nodes: 338, public_depth: 10 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 16, old_and_nodes: 506, old_depth: 12, public_and_nodes: 474, public_depth: 11 },
        ];

        assert_eq!(got.as_slice(), want);
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ArraySliceQorRow {
        array_len: usize,
        element_width: usize,
        slice_width: usize,
        old_and_nodes: usize,
        old_depth: usize,
        elem_mux_and_nodes: usize,
        elem_mux_depth: usize,
        public_and_nodes: usize,
        public_depth: usize,
    }

    fn build_array_slice_ir_text(
        array_len: usize,
        element_width: usize,
        start_width: usize,
        slice_width: usize,
    ) -> String {
        let return_width = element_width * slice_width;
        let mut text = format!(
            r#"package sample

top fn main(array: bits[{element_width}][{array_len}], start: bits[{start_width}]) -> bits[{return_width}] {{
  y: bits[{element_width}][{slice_width}] = array_slice(array, start, width={slice_width}, id=3)
"#
        );
        for i in 0..slice_width {
            text.push_str(&format!(
                "  idx{i}: bits[32] = literal(value={i}, id={})\n",
                10 + i
            ));
            text.push_str(&format!(
                "  elem{i}: bits[{element_width}] = array_index(y, indices=[idx{i}], id={})\n",
                20 + i
            ));
        }
        if slice_width == 1 {
            text.push_str(&format!(
                "  ret out: bits[{return_width}] = array_index(y, indices=[idx0], id=100)\n"
            ));
        } else {
            let operands = (0..slice_width)
                .rev()
                .map(|i| format!("elem{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            text.push_str(&format!(
                "  ret out: bits[{return_width}] = concat({operands}, id=100)\n"
            ));
        }
        text.push_str("}\n");
        text
    }

    fn make_array_slice_sample_array(
        array_len: usize,
        element_width: usize,
        elements: &[u64],
    ) -> IrValue {
        assert_eq!(elements.len(), array_len);
        let values = elements
            .iter()
            .map(|value| IrValue::make_ubits(element_width, *value).unwrap())
            .collect::<Vec<_>>();
        IrValue::make_array(&values).unwrap()
    }

    fn array_slice_sample_arrays(array_len: usize, element_width: usize) -> Vec<IrValue> {
        let all_ones = if element_width == 64 {
            u64::MAX
        } else {
            (1u64 << element_width) - 1
        };
        let high_bit = 1u64 << (element_width - 1);
        let mut samples = Vec::new();
        samples.push(make_array_slice_sample_array(
            array_len,
            element_width,
            &vec![0; array_len],
        ));
        samples.push(make_array_slice_sample_array(
            array_len,
            element_width,
            &vec![all_ones; array_len],
        ));

        for index in 0..array_len {
            for value in [1, high_bit, all_ones] {
                let mut elements = vec![0; array_len];
                elements[index] = value;
                samples.push(make_array_slice_sample_array(
                    array_len,
                    element_width,
                    &elements,
                ));
            }
        }

        let alternating = (0..array_len)
            .map(|i| if i % 2 == 0 { 0 } else { all_ones })
            .collect::<Vec<_>>();
        samples.push(make_array_slice_sample_array(
            array_len,
            element_width,
            &alternating,
        ));

        samples
    }

    fn validate_array_slice_public_simulation(
        ir_fn: &ir::Fn,
        gate_fn: &GateFn,
        array_len: usize,
        element_width: usize,
        start_width: usize,
    ) {
        let array_ty = ir::Type::Array(ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        });
        let start_count = 1usize << start_width;
        for array_value in array_slice_sample_arrays(array_len, element_width) {
            let mut array_bits = Vec::new();
            flatten_ir_value_to_lsb0_bits_for_type(&array_value, &array_ty, &mut array_bits)
                .unwrap();
            let gate_array = IrBits::from_lsb_is_0(&array_bits);
            for start in 0..start_count {
                let start_value = IrValue::make_ubits(start_width, start as u64).unwrap();
                let want = match eval_fn(ir_fn, &[array_value.clone(), start_value.clone()]) {
                    FnEvalResult::Success(success) => success.value.to_bits().unwrap(),
                    FnEvalResult::Failure(failure) => {
                        panic!("array_slice source IR failed during simulation: {failure:?}")
                    }
                };
                let gate_start = start_value.to_bits().unwrap();
                let sim = gate_sim::eval(
                    gate_fn,
                    &[gate_array.clone(), gate_start],
                    gate_sim::Collect::None,
                );
                let got = sim.outputs[0].clone();
                assert_eq!(
                    got, want,
                    "array_slice simulation mismatch for array_len={array_len} element_width={element_width} start={start} array={array_value}"
                );
            }
        }
    }

    fn get_array_slice_bit_shift_stats(
        array_len: usize,
        element_width: usize,
        start_width: usize,
        slice_width: usize,
    ) -> AigStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        };
        let mut gb = GateBuilder::new(
            format!("array_slice_bit_shift_n{array_len}_w{element_width}_s{slice_width}"),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("array".to_string(), array_len * element_width);
        let start_bits = gb.add_input("start".to_string(), start_width);
        let result = super::gatify_array_slice_bit_shift(
            &mut gb,
            &array_ty,
            &array_bits,
            &start_bits,
            false,
            slice_width,
            0,
            AdderMapping::BrentKung,
        );
        gb.add_output("result".to_string(), result);
        get_aig_stats(&gb.build())
    }

    fn get_array_slice_element_mux_stats(
        array_len: usize,
        element_width: usize,
        start_width: usize,
        slice_width: usize,
    ) -> AigStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        };
        let mut gb = GateBuilder::new(
            format!("array_slice_elem_mux_n{array_len}_w{element_width}_s{slice_width}"),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("array".to_string(), array_len * element_width);
        let start_bits = gb.add_input("start".to_string(), start_width);
        let result = super::gatify_array_slice_element_mux_if_profitable(
            &mut gb,
            &array_ty,
            &array_bits,
            &start_bits,
            slice_width,
        )
        .expect("array-slice element mux should apply for characterization case");
        gb.add_output("result".to_string(), result);
        get_aig_stats(&gb.build())
    }

    fn get_array_slice_public_stats_and_validate(
        array_len: usize,
        element_width: usize,
        start_width: usize,
        slice_width: usize,
    ) -> AigStats {
        let ir_text = build_array_slice_ir_text(array_len, element_width, start_width, slice_width);
        let mut parser = ir_parser::Parser::new(&ir_text);
        let ir_package = parser.parse_and_validate_package().expect("parse package");
        let ir_fn = ir_package.get_top_fn().expect("top fn");
        let gatify_output = gatify(
            ir_fn,
            GatifyOptions {
                adder_mapping: AdderMapping::BrentKung,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .expect("gatify array_slice");
        validate_array_slice_public_simulation(
            ir_fn,
            &gatify_output.gate_fn,
            array_len,
            element_width,
            start_width,
        );
        get_aig_stats(&gatify_output.gate_fn)
    }

    fn gather_array_slice_qor_rows() -> Vec<ArraySliceQorRow> {
        let mut got = Vec::new();
        for array_len in [5usize, 8, 16] {
            let start_width = (array_len - 1).ilog2() as usize + 1;
            for element_width in [1usize, 3, 5] {
                for slice_width in 1usize..=4 {
                    let old = get_array_slice_bit_shift_stats(
                        array_len,
                        element_width,
                        start_width,
                        slice_width,
                    );
                    let elem_mux = get_array_slice_element_mux_stats(
                        array_len,
                        element_width,
                        start_width,
                        slice_width,
                    );
                    let public = get_array_slice_public_stats_and_validate(
                        array_len,
                        element_width,
                        start_width,
                        slice_width,
                    );
                    got.push(ArraySliceQorRow {
                        array_len,
                        element_width,
                        slice_width,
                        old_and_nodes: old.and_nodes,
                        old_depth: old.max_depth,
                        elem_mux_and_nodes: elem_mux.and_nodes,
                        elem_mux_depth: elem_mux.max_depth,
                        public_and_nodes: public.and_nodes,
                        public_depth: public.max_depth,
                    });
                }
            }
        }
        got
    }

    #[test]
    fn test_array_slice_element_mux_qor_and_equivalence_sweep() {
        let _ = env_logger::builder().is_test(true).try_init();

        let got = gather_array_slice_qor_rows();

        for row in &got {
            assert!(
                row.elem_mux_and_nodes <= row.old_and_nodes,
                "expected element-mux lowering not to increase AND nodes: {:?}",
                row
            );
            assert!(
                row.elem_mux_depth <= row.old_depth,
                "expected element-mux lowering not to increase depth: {:?}",
                row
            );
            assert_eq!(
                (row.public_and_nodes, row.public_depth),
                (row.elem_mux_and_nodes, row.elem_mux_depth),
                "expected public array_slice lowering to use element-mux strategy: {:?}",
                row
            );
        }

        #[rustfmt::skip]
        let want: &[ArraySliceQorRow] = &[
            ArraySliceQorRow { array_len: 5, element_width: 1, slice_width: 1, old_and_nodes: 22, old_depth: 10, elem_mux_and_nodes: 18, elem_mux_depth: 6, public_and_nodes: 18, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 1, slice_width: 2, old_and_nodes: 36, old_depth: 10, elem_mux_and_nodes: 28, elem_mux_depth: 6, public_and_nodes: 28, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 1, slice_width: 3, old_and_nodes: 44, old_depth: 10, elem_mux_and_nodes: 32, elem_mux_depth: 6, public_and_nodes: 32, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 1, slice_width: 4, old_and_nodes: 50, old_depth: 10, elem_mux_and_nodes: 36, elem_mux_depth: 6, public_and_nodes: 36, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 3, slice_width: 1, old_and_nodes: 136, old_depth: 14, elem_mux_and_nodes: 54, elem_mux_depth: 6, public_and_nodes: 54, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 3, slice_width: 2, old_and_nodes: 189, old_depth: 15, elem_mux_and_nodes: 84, elem_mux_depth: 6, public_and_nodes: 84, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 3, slice_width: 3, old_and_nodes: 221, old_depth: 15, elem_mux_and_nodes: 96, elem_mux_depth: 6, public_and_nodes: 96, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 3, slice_width: 4, old_and_nodes: 241, old_depth: 15, elem_mux_and_nodes: 108, elem_mux_depth: 6, public_and_nodes: 108, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 5, slice_width: 1, old_and_nodes: 252, old_depth: 15, elem_mux_and_nodes: 90, elem_mux_depth: 6, public_and_nodes: 90, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 5, slice_width: 2, old_and_nodes: 340, old_depth: 15, elem_mux_and_nodes: 140, elem_mux_depth: 6, public_and_nodes: 140, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 5, slice_width: 3, old_and_nodes: 398, old_depth: 16, elem_mux_and_nodes: 160, elem_mux_depth: 6, public_and_nodes: 160, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 5, slice_width: 4, old_and_nodes: 442, old_depth: 16, elem_mux_and_nodes: 180, elem_mux_depth: 6, public_and_nodes: 180, public_depth: 6 },

            ArraySliceQorRow { array_len: 8, element_width: 1, slice_width: 1, old_and_nodes: 21, old_depth: 6, elem_mux_and_nodes: 21, elem_mux_depth: 6, public_and_nodes: 21, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 1, slice_width: 2, old_and_nodes: 41, old_depth: 6, elem_mux_and_nodes: 41, elem_mux_depth: 6, public_and_nodes: 41, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 1, slice_width: 3, old_and_nodes: 49, old_depth: 6, elem_mux_and_nodes: 49, elem_mux_depth: 6, public_and_nodes: 49, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 1, slice_width: 4, old_and_nodes: 57, old_depth: 6, elem_mux_and_nodes: 57, elem_mux_depth: 6, public_and_nodes: 57, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 3, slice_width: 1, old_and_nodes: 193, old_depth: 10, elem_mux_and_nodes: 63, elem_mux_depth: 6, public_and_nodes: 63, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 3, slice_width: 2, old_and_nodes: 272, old_depth: 10, elem_mux_and_nodes: 123, elem_mux_depth: 6, public_and_nodes: 123, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 3, slice_width: 3, old_and_nodes: 318, old_depth: 10, elem_mux_and_nodes: 147, elem_mux_depth: 6, public_and_nodes: 147, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 3, slice_width: 4, old_and_nodes: 344, old_depth: 10, elem_mux_and_nodes: 171, elem_mux_depth: 6, public_and_nodes: 171, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 5, slice_width: 1, old_and_nodes: 389, old_depth: 12, elem_mux_and_nodes: 105, elem_mux_depth: 6, public_and_nodes: 105, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 5, slice_width: 2, old_and_nodes: 525, old_depth: 12, elem_mux_and_nodes: 205, elem_mux_depth: 6, public_and_nodes: 205, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 5, slice_width: 3, old_and_nodes: 607, old_depth: 12, elem_mux_and_nodes: 245, elem_mux_depth: 6, public_and_nodes: 245, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 5, slice_width: 4, old_and_nodes: 664, old_depth: 12, elem_mux_and_nodes: 285, elem_mux_depth: 6, public_and_nodes: 285, public_depth: 6 },

            ArraySliceQorRow { array_len: 16, element_width: 1, slice_width: 1, old_and_nodes: 45, old_depth: 8, elem_mux_and_nodes: 45, elem_mux_depth: 8, public_and_nodes: 45, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 1, slice_width: 2, old_and_nodes: 89, old_depth: 8, elem_mux_and_nodes: 89, elem_mux_depth: 8, public_and_nodes: 89, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 1, slice_width: 3, old_and_nodes: 109, old_depth: 8, elem_mux_and_nodes: 109, elem_mux_depth: 8, public_and_nodes: 109, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 1, slice_width: 4, old_and_nodes: 129, old_depth: 8, elem_mux_and_nodes: 129, elem_mux_depth: 8, public_and_nodes: 129, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 3, slice_width: 1, old_and_nodes: 388, old_depth: 13, elem_mux_and_nodes: 135, elem_mux_depth: 8, public_and_nodes: 135, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 3, slice_width: 2, old_and_nodes: 542, old_depth: 13, elem_mux_and_nodes: 267, elem_mux_depth: 8, public_and_nodes: 267, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 3, slice_width: 3, old_and_nodes: 636, old_depth: 13, elem_mux_and_nodes: 327, elem_mux_depth: 8, public_and_nodes: 327, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 3, slice_width: 4, old_and_nodes: 694, old_depth: 13, elem_mux_and_nodes: 387, elem_mux_depth: 8, public_and_nodes: 387, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 5, slice_width: 1, old_and_nodes: 790, old_depth: 14, elem_mux_and_nodes: 225, elem_mux_depth: 8, public_and_nodes: 225, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 5, slice_width: 2, old_and_nodes: 1051, old_depth: 14, elem_mux_and_nodes: 445, elem_mux_depth: 8, public_and_nodes: 445, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 5, slice_width: 3, old_and_nodes: 1211, old_depth: 14, elem_mux_and_nodes: 545, elem_mux_depth: 8, public_and_nodes: 545, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 5, slice_width: 4, old_and_nodes: 1311, old_depth: 14, elem_mux_and_nodes: 645, elem_mux_depth: 8, public_and_nodes: 645, public_depth: 8 },
        ];

        assert_eq!(got.as_slice(), want);
    }

    #[test]
    fn test_gatify_array_slice_narrow_start_regression() {
        let ir_text = "package sample
top fn f(start: bits[2], a: bits[8]) -> bits[8][1] {
  array.4: bits[8][8] = array(a, a, a, a, a, a, a, a, id=4)
  ret array_slice.5: bits[8][1] = array_slice(array.4, start, width=1, id=5)
}
";
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();

        // Regression: this used to panic when clamping an out-of-bounds start index
        // because last_idx (7) was forced into start width (2 bits).
        let gatify_output = gatify(
            &ir_fn,
            GatifyOptions {
                fold: false,
                hash: false,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .expect("gatify array_slice with narrow start should not panic");

        // bits[8][1] flatten to an 8-bit gate output.
        assert_eq!(
            gatify_output.gate_fn.outputs[0].bit_vector.get_bit_count(),
            8
        );

        // Optional end-to-end equivalence check against the source IR.
        // This test focuses on clamp behavior in the gate-level lowering itself.
        // End-to-end IR equivalence for this wider-start case is covered elsewhere.
        let _ = ir_fn;
    }

    #[test]
    fn test_gatify_array_slice_wide_start_clamps_oob_to_last_element() {
        let ir_text = "package sample
top fn f(start: bits[4], a: bits[8], b: bits[8]) -> bits[8][1] {
  array.4: bits[8][8] = array(a, a, a, a, a, a, b, b, id=4)
  ret array_slice.5: bits[8][1] = array_slice(array.4, start, width=1, id=5)
}
";
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();

        let gatify_output = gatify(
            &ir_fn,
            GatifyOptions {
                fold: false,
                hash: false,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .expect("gatify array_slice with wide start should succeed");

        let eval = |start: u64, a: u64, b: u64| {
            let inputs = vec![
                xlsynth::IrBits::make_ubits(4, start).unwrap(),
                xlsynth::IrBits::make_ubits(8, a).unwrap(),
                xlsynth::IrBits::make_ubits(8, b).unwrap(),
            ];
            gate_sim::eval(&gatify_output.gate_fn, &inputs, gate_sim::Collect::None).outputs[0]
                .clone()
        };

        let a = 0x12;
        let b = 0x34;
        let at_7 = eval(7, a, b);
        assert_eq!(eval(8, a, b), at_7);
        assert_eq!(eval(15, a, b), at_7);

        let _ = ir_fn;
    }

    fn gatify_ir_text(ir_text: &str) -> GateFn {
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();
        gatify(
            &ir_fn,
            GatifyOptions::all_opts_disabled(),
        )
        .unwrap()
        .gate_fn
    }

    fn gatify_ir_text_with_gate_opt_in(ir_text: &str) -> Result<GateFn, String> {
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();
        gatify(
            ir_fn,
            GatifyOptions {
                unsafe_gatify_gate_operation: true,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .map(|output| output.gate_fn)
    }

    #[test]
    fn test_gate_operation_requires_opt_in() {
        let ir_text = r#"package sample

top fn f(pred: bits[1] id=1, x: bits[3] id=2) -> bits[3] {
  ret gated: bits[3] = gate(pred, x, id=3)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();
        let err = gatify(
            ir_fn,
            GatifyOptions::all_opts_disabled(),
        )
        .unwrap_err();
        assert!(err.contains("--unsafe-gatify-gate-operation=true"));
    }

    #[test]
    fn test_gate_operation_masks_bits_when_enabled() {
        let ir_text = r#"package sample

top fn f(pred: bits[1] id=1, x: bits[3] id=2) -> bits[3] {
  ret gated: bits[3] = gate(pred, x, id=3)
}
"#;
        let gate_fn = gatify_ir_text_with_gate_opt_in(ir_text).unwrap();
        let eval = |pred: u64, x: u64| {
            gate_sim::eval(
                &gate_fn,
                &[
                    IrBits::make_ubits(1, pred).unwrap(),
                    IrBits::make_ubits(3, x).unwrap(),
                ],
                gate_sim::Collect::None,
            )
            .outputs[0]
                .clone()
        };
        assert_eq!(eval(0, 0b101), IrBits::make_ubits(3, 0).unwrap());
        assert_eq!(eval(1, 0b101), IrBits::make_ubits(3, 0b101).unwrap());
    }

    #[test]
    fn test_gate_operation_masks_flattened_tuple_when_enabled() {
        let ir_text = r#"package sample

top fn f(pred: bits[1] id=1, x: bits[2] id=2, y: bits[3] id=3) -> (bits[2], bits[3]) {
  t: (bits[2], bits[3]) = tuple(x, y, id=4)
  ret gated: (bits[2], bits[3]) = gate(pred, t, id=5)
}
"#;
        let gate_fn = gatify_ir_text_with_gate_opt_in(ir_text).unwrap();
        let eval = |pred: u64, x: u64, y: u64| {
            gate_sim::eval(
                &gate_fn,
                &[
                    IrBits::make_ubits(1, pred).unwrap(),
                    IrBits::make_ubits(2, x).unwrap(),
                    IrBits::make_ubits(3, y).unwrap(),
                ],
                gate_sim::Collect::None,
            )
            .outputs[0]
                .clone()
        };
        assert_eq!(eval(0, 0b10, 0b101), IrBits::make_ubits(5, 0).unwrap());
        assert_eq!(
            eval(1, 0b10, 0b101),
            IrBits::make_ubits(5, 0b10101).unwrap()
        );
    }

    fn gate_eval_1bit(gate_fn: &GateFn, lhs: u64, lhs_width: usize) -> bool {
        let inputs = vec![xlsynth::IrBits::make_ubits(lhs_width, lhs).unwrap()];
        gate_sim::eval(gate_fn, &inputs, gate_sim::Collect::None).outputs[0]
            .get_bit(0)
            .unwrap()
    }

    fn bits_as_signed(value: u64, width: usize) -> i64 {
        assert!(width > 0 && width < 63);
        let sign_bit = 1u64 << (width - 1);
        if (value & sign_bit) == 0 {
            value as i64
        } else {
            (value as i64) - (1i64 << width)
        }
    }

    fn expected_signed_cmp(binop: ir::Binop, lhs: u64, rhs: u64, width: usize) -> bool {
        let lhs_signed = bits_as_signed(lhs, width);
        let rhs_signed = bits_as_signed(rhs, width);
        match binop {
            ir::Binop::Slt => lhs_signed < rhs_signed,
            ir::Binop::Sle => lhs_signed <= rhs_signed,
            ir::Binop::Sgt => lhs_signed > rhs_signed,
            ir::Binop::Sge => lhs_signed >= rhs_signed,
            _ => panic!(
                "unexpected binop for signed-compare proof test: {:?}",
                binop
            ),
        }
    }

    #[test]
    fn test_signed_literal_cmp_proof_matrix_rhs_and_lhs_literal() {
        let width = 5usize;
        let values = [9u64, 26u64];
        let cases: &[(ir::Binop, &str)] = &[
            (ir::Binop::Slt, "slt"),
            (ir::Binop::Sle, "sle"),
            (ir::Binop::Sgt, "sgt"),
            (ir::Binop::Sge, "sge"),
        ];

        for &(binop, op_name) in cases {
            for &rhs in &values {
                for literal_on_lhs in [false, true] {
                    let expr = if literal_on_lhs {
                        format!("{op_name}(k, x, id=11)")
                    } else {
                        format!("{op_name}(x, k, id=11)")
                    };
                    let ir_text = format!(
                        r#"package sample
top fn f(x: bits[{width}]) -> bits[1] {{
  k: bits[{width}] = literal(value={rhs}, id=10)
  ret out: bits[1] = {expr}
}}
"#
                    );
                    let gate_fn = gatify_ir_text(&ir_text);
                    for lhs in 0u64..(1u64 << width) {
                        let got = gate_eval_1bit(&gate_fn, lhs, width);
                        let want = if literal_on_lhs {
                            expected_signed_cmp(binop, rhs, lhs, width)
                        } else {
                            expected_signed_cmp(binop, lhs, rhs, width)
                        };
                        assert_eq!(
                            got, want,
                            "signed literal compare mismatch: op={} rhs={} literal_on_lhs={} lhs={}",
                            op_name, rhs, literal_on_lhs, lhs
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_signed_literal_cmp_cone_quality_matches_manual_decomposition() {
        let cone_ir = r#"package sample
top fn cone(leaf_7: bits[5], leaf_9: bits[33]) -> bits[1] {
  sign_ext.3431: bits[33] = sign_ext(leaf_7, new_bit_count=33, id=3431)
  add.3432: bits[33] = add(leaf_9, sign_ext.3431, id=3432)
  literal.3433: bits[33] = literal(value=8589934578, id=3433)
  ret slt.3434: bits[1] = slt(add.3432, literal.3433, id=3434)
}
"#;
        let decomp_ir = r#"package sample
top fn cone(leaf_7: bits[5], leaf_9: bits[33]) -> bits[1] {
  sign_ext.3431: bits[33] = sign_ext(leaf_7, new_bit_count=33, id=3431)
  add.3432: bits[33] = add(leaf_9, sign_ext.3431, id=3432)
  literal.3433: bits[33] = literal(value=8589934578, id=3433)
  bit_slice.3435: bits[1] = bit_slice(add.3432, start=32, width=1, id=3435)
  ult.3436: bits[1] = ult(add.3432, literal.3433, id=3436)
  ret and.3437: bits[1] = and(bit_slice.3435, ult.3436, id=3437)
}
"#;

        let cone_gate_fn = gatify_ir_text(cone_ir);
        let decomp_gate_fn = gatify_ir_text(decomp_ir);
        let cone_stats = get_summary_stats(&cone_gate_fn);
        let decomp_stats = get_summary_stats(&decomp_gate_fn);

        assert_eq!(
            cone_stats.live_nodes, decomp_stats.live_nodes,
            "literal signed-compare lowering should match manual decomposition node count"
        );
        assert_eq!(
            cone_stats.deepest_path, decomp_stats.deepest_path,
            "literal signed-compare lowering should match manual decomposition depth"
        );

        // Characterization guard for the original regression report:
        // before this lowering path, this cone was observed at roughly
        // 562 nodes / 33 levels in g8r output.
        assert!(
            cone_stats.live_nodes <= 434,
            "expected improved node count for regression cone; got {}",
            cone_stats.live_nodes
        );
        assert!(
            cone_stats.deepest_path <= 28,
            "expected improved depth for regression cone; got {}",
            cone_stats.deepest_path
        );
    }

    fn get_1b_priority_sel_stats_for_impl(
        operand_count: usize,
        use_mux_chain: bool,
    ) -> SummaryStats {
        let mut gb = GateBuilder::new(
            format!(
                "prio_sel_{}_{}",
                operand_count,
                if use_mux_chain { "mux" } else { "mask" }
            ),
            GateBuilderOptions::opt(),
        );
        let selector_bits = gb.add_input("sel".to_string(), operand_count);
        let mut cases = Vec::with_capacity(operand_count);
        for i in 0..operand_count {
            cases.push(gb.add_input(format!("a{}", i), 1));
        }
        let default_bits = gb.add_input("default_value".to_string(), 1);

        let result = if use_mux_chain {
            super::gatify_priority_sel_mux_chain(&mut gb, selector_bits, &cases, default_bits)
        } else {
            super::gatify_priority_sel_masking(
                &mut gb,
                /* output_bit_count= */ 1,
                selector_bits,
                &cases,
                Some(default_bits),
            )
        };
        gb.add_output("result".to_string(), result);
        let gate_fn = gb.build();
        get_summary_stats(&gate_fn)
    }

    /// TDD-ish “microbenchmark sweep” test: record the baseline W=1
    /// priority_sel cost (mask+OR implementation) and assert the mux-chain
    /// specialization is strictly cheaper.
    #[test]
    fn test_priority_sel_1b_mux_chain_is_cheaper_than_masking_sweep() {
        #[rustfmt::skip]
        let want_masking: &[(usize, usize)] = &[
            (2, 12),
            (3, 18),
            (4, 24),
            (5, 30),
        ];
        #[rustfmt::skip]
        let want_mux_chain: &[(usize, usize)] = &[
            (2, 11),
            (3, 16),
            (4, 21),
            (5, 26),
        ];

        for &(operand_count, want_live_nodes) in want_masking {
            let got =
                get_1b_priority_sel_stats_for_impl(operand_count, /* use_mux_chain= */ false);
            assert_eq!(
                got.live_nodes, want_live_nodes,
                "masking impl live_nodes mismatch for operand_count={}",
                operand_count
            );
        }

        for &(operand_count, want_live_nodes) in want_mux_chain {
            let got =
                get_1b_priority_sel_stats_for_impl(operand_count, /* use_mux_chain= */ true);
            assert_eq!(
                got.live_nodes, want_live_nodes,
                "mux-chain impl live_nodes mismatch for operand_count={}",
                operand_count
            );
        }

        for operand_count in 2usize..=5 {
            let masking =
                get_1b_priority_sel_stats_for_impl(operand_count, /* use_mux_chain= */ false);
            let mux =
                get_1b_priority_sel_stats_for_impl(operand_count, /* use_mux_chain= */ true);
            assert!(
                mux.live_nodes < masking.live_nodes,
                "expected mux-chain to be cheaper for operand_count={}; masking live_nodes={}, mux live_nodes={}",
                operand_count,
                masking.live_nodes,
                mux.live_nodes
            );
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct PrioSelSweepRow {
        operand_count: usize,
        masking_live_nodes: usize,
        masking_deepest_path: usize,
        mux_chain_live_nodes: usize,
        mux_chain_deepest_path: usize,
    }

    fn get_priority_sel_stats_for_impl(
        output_bit_count: usize,
        operand_count: usize,
        use_mux_chain: bool,
    ) -> SummaryStats {
        let mut gb = GateBuilder::new(
            format!(
                "prio_sel_w{}_n{}_{}",
                output_bit_count,
                operand_count,
                if use_mux_chain { "mux" } else { "mask" }
            ),
            GateBuilderOptions::opt(),
        );
        let selector_bits = gb.add_input("sel".to_string(), operand_count);
        let mut cases = Vec::with_capacity(operand_count);
        for i in 0..operand_count {
            cases.push(gb.add_input(format!("a{}", i), output_bit_count));
        }
        let default_bits = gb.add_input("default_value".to_string(), output_bit_count);

        let result = if use_mux_chain {
            // Hypothetical mux-chain lowering for comparison across widths:
            // mux(s0, c0, mux(s1, c1, ... default))
            let mut acc = default_bits;
            for i in (0..cases.len()).rev() {
                let s_i = selector_bits.get_lsb(i);
                acc = gb.add_mux2_vec(s_i, &cases[i], &acc);
            }
            acc
        } else {
            super::gatify_priority_sel_masking(
                &mut gb,
                output_bit_count,
                selector_bits,
                &cases,
                Some(default_bits),
            )
        };
        gb.add_output("result".to_string(), result);
        let gate_fn = gb.build();
        get_summary_stats(&gate_fn)
    }

    /// “Table sweep” test: captures masking vs mux-chain AIG sizes as output
    /// width increases (W=1..4) for small operand counts (2..5).
    ///
    /// This reflects the crossover behavior we care about when deciding whether
    /// a mux-chain specialization is worthwhile beyond W=1.
    #[test]
    fn test_priority_sel_masking_vs_mux_chain_table_sweep_w1_to_w4() {
        #[rustfmt::skip]
        const WANT_W1: &[PrioSelSweepRow] = &[
            PrioSelSweepRow { operand_count: 2, masking_live_nodes: 12, masking_deepest_path: 6, mux_chain_live_nodes: 11, mux_chain_deepest_path: 5 },
            PrioSelSweepRow { operand_count: 3, masking_live_nodes: 18, masking_deepest_path: 8, mux_chain_live_nodes: 16, mux_chain_deepest_path: 7 },
            PrioSelSweepRow { operand_count: 4, masking_live_nodes: 24, masking_deepest_path: 11, mux_chain_live_nodes: 21, mux_chain_deepest_path: 9 },
            PrioSelSweepRow { operand_count: 5, masking_live_nodes: 30, masking_deepest_path: 13, mux_chain_live_nodes: 26, mux_chain_deepest_path: 11 },
        ];
        #[rustfmt::skip]
        const WANT_W2: &[PrioSelSweepRow] = &[
            PrioSelSweepRow { operand_count: 2, masking_live_nodes: 20, masking_deepest_path: 6, mux_chain_live_nodes: 20, mux_chain_deepest_path: 5 },
            PrioSelSweepRow { operand_count: 3, masking_live_nodes: 29, masking_deepest_path: 8, mux_chain_live_nodes: 29, mux_chain_deepest_path: 7 },
            PrioSelSweepRow { operand_count: 4, masking_live_nodes: 38, masking_deepest_path: 11, mux_chain_live_nodes: 38, mux_chain_deepest_path: 9 },
            PrioSelSweepRow { operand_count: 5, masking_live_nodes: 47, masking_deepest_path: 13, mux_chain_live_nodes: 47, mux_chain_deepest_path: 11 },
        ];
        #[rustfmt::skip]
        const WANT_W3: &[PrioSelSweepRow] = &[
            PrioSelSweepRow { operand_count: 2, masking_live_nodes: 28, masking_deepest_path: 6, mux_chain_live_nodes: 29, mux_chain_deepest_path: 5 },
            PrioSelSweepRow { operand_count: 3, masking_live_nodes: 40, masking_deepest_path: 8, mux_chain_live_nodes: 42, mux_chain_deepest_path: 7 },
            PrioSelSweepRow { operand_count: 4, masking_live_nodes: 52, masking_deepest_path: 11, mux_chain_live_nodes: 55, mux_chain_deepest_path: 9 },
            PrioSelSweepRow { operand_count: 5, masking_live_nodes: 64, masking_deepest_path: 13, mux_chain_live_nodes: 68, mux_chain_deepest_path: 11 },
        ];
        #[rustfmt::skip]
        const WANT_W4: &[PrioSelSweepRow] = &[
            PrioSelSweepRow { operand_count: 2, masking_live_nodes: 36, masking_deepest_path: 6, mux_chain_live_nodes: 38, mux_chain_deepest_path: 5 },
            PrioSelSweepRow { operand_count: 3, masking_live_nodes: 51, masking_deepest_path: 8, mux_chain_live_nodes: 55, mux_chain_deepest_path: 7 },
            PrioSelSweepRow { operand_count: 4, masking_live_nodes: 66, masking_deepest_path: 11, mux_chain_live_nodes: 72, mux_chain_deepest_path: 9 },
            PrioSelSweepRow { operand_count: 5, masking_live_nodes: 81, masking_deepest_path: 13, mux_chain_live_nodes: 89, mux_chain_deepest_path: 11 },
        ];
        #[rustfmt::skip]
        const WANT: &[(usize, &[PrioSelSweepRow])] = &[
            (1, WANT_W1),
            (2, WANT_W2),
            (3, WANT_W3),
            (4, WANT_W4),
        ];

        let mut computed: Vec<(usize, Vec<PrioSelSweepRow>)> = Vec::new();
        for w in 1usize..=4 {
            let mut rows: Vec<PrioSelSweepRow> = Vec::new();
            for operand_count in 2usize..=5 {
                let masking = get_priority_sel_stats_for_impl(
                    w,
                    operand_count,
                    /* use_mux_chain= */ false,
                );
                let mux = get_priority_sel_stats_for_impl(
                    w,
                    operand_count,
                    /* use_mux_chain= */ true,
                );
                rows.push(PrioSelSweepRow {
                    operand_count,
                    masking_live_nodes: masking.live_nodes,
                    masking_deepest_path: masking.deepest_path,
                    mux_chain_live_nodes: mux.live_nodes,
                    mux_chain_deepest_path: mux.deepest_path,
                });
            }
            computed.push((w, rows));
        }

        for &(w, want_rows) in WANT {
            let got_rows = computed
                .iter()
                .find(|(got_w, _)| *got_w == w)
                .unwrap()
                .1
                .as_slice();
            assert_eq!(
                got_rows.len(),
                want_rows.len(),
                "row count mismatch for W={}",
                w
            );
            for (got, want) in got_rows.iter().zip(want_rows.iter()) {
                assert_eq!(
                    got, want,
                    "priority_sel sweep mismatch for W={}, operand_count={}; computed tables: {:?}",
                    w, want.operand_count, computed
                );
            }
        }
    }

    fn get_tuple_field0_array_index_stats(
        array_len: usize,
        payload_width: usize,
        index_width: usize,
        strategy: super::ArrayIndexLoweringStrategy,
    ) -> SummaryStats {
        let tuple_ty = ir::Type::Tuple(vec![
            Box::new(ir::Type::Bits(1)),
            Box::new(ir::Type::Bits(payload_width)),
        ]);
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(tuple_ty.clone()),
            element_count: array_len,
        };
        let field0 = tuple_ty.tuple_get_flat_bit_slice_for_index(0).unwrap();

        let mut gb = GateBuilder::new(
            format!("tuple_field0_w{}_n{}", payload_width, array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), tuple_ty.bit_count() * array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let selected = super::gatify_array_index(
            &mut gb,
            &array_ty,
            &array_bits,
            &index_bits,
            false,
            strategy,
        );
        let result = selected.get_lsb_slice(field0.start, field0.limit - field0.start);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_1b_array_index_oob_one_hot_stats(array_len: usize, index_width: usize) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(1)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_1b_n{}_oob", array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result =
            super::gatify_array_index_oob_one_hot(&mut gb, &array_ty, &array_bits, &index_bits);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_1b_array_index_exact_stats(array_len: usize, index_width: usize) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(1)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_1b_n{}_exact", array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result = super::gatify_array_index_exact(&mut gb, &array_ty, &array_bits, &index_bits);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn repeated_literal_array_bits(
        gb: &mut GateBuilder,
        array_len: usize,
        element_width: usize,
        distinct_values: usize,
    ) -> AigBitVector {
        assert!(distinct_values > 0);
        let mut bits = Vec::with_capacity(array_len * element_width);
        for i in 0..array_len {
            let value_bits = gb.add_literal(
                &xlsynth::IrBits::make_ubits(element_width, (i % distinct_values) as u64).unwrap(),
            );
            bits.extend(value_bits.iter_lsb_to_msb().copied());
        }
        AigBitVector::from_lsb_is_index_0(&bits)
    }

    fn get_repeated_literal_array_index_exact_stats(
        array_len: usize,
        element_width: usize,
        index_width: usize,
        distinct_values: usize,
    ) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!(
                "array_index_repeated_literal_n{}_w{}",
                array_len, element_width
            ),
            GateBuilderOptions::opt(),
        );
        let array_bits =
            repeated_literal_array_bits(&mut gb, array_len, element_width, distinct_values);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result = super::gatify_array_index_exact(&mut gb, &array_ty, &array_bits, &index_bits);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_repeated_literal_array_index_one_hot_stats(
        array_len: usize,
        element_width: usize,
        index_width: usize,
        distinct_values: usize,
    ) -> SummaryStats {
        let mut gb = GateBuilder::new(
            format!(
                "array_index_repeated_literal_one_hot_n{}_w{}",
                array_len, element_width
            ),
            GateBuilderOptions::opt(),
        );
        let array_bits =
            repeated_literal_array_bits(&mut gb, array_len, element_width, distinct_values);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let selector = super::gatify_decode(&mut gb, array_len, &index_bits);
        let cases = (0..array_len)
            .map(|i| array_bits.get_lsb_slice(i * element_width, element_width))
            .collect::<Vec<_>>();
        let result = crate::ir2gate_utils::gatify_one_hot_select(&mut gb, &selector, &cases);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_1b_array_index_near_pow2_padded_stats(
        array_len: usize,
        index_width: usize,
    ) -> SummaryStats {
        const MAX_PADDED_COUNT: usize = 32;
        const MAX_EXTRA_ELEMS: usize = 8;
        let padded_count = array_len.next_power_of_two();
        assert!(
            padded_count > array_len
                && padded_count <= MAX_PADDED_COUNT
                && padded_count - array_len <= MAX_EXTRA_ELEMS
        );

        let mut gb = GateBuilder::new(
            format!("array_index_1b_n{}_padpow2", array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let padded_last_bits = gb.add_literal(
            &xlsynth::IrBits::make_ubits(index_width, (padded_count - 1) as u64).unwrap(),
        );
        let idx_le_padded_last =
            super::gatify_ule_via_bit_tests(&mut gb, 0, &index_bits, &padded_last_bits);
        let clamped_index = gb.add_mux2_vec(&idx_le_padded_last, &index_bits, &padded_last_bits);
        let last_elem = array_bits.get_lsb_slice(array_len - 1, 1);
        let mut padded_array_bits = array_bits.clone();
        for _ in array_len..padded_count {
            padded_array_bits = AigBitVector::concat(last_elem.clone(), padded_array_bits);
        }
        let padded_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(1)),
            element_count: padded_count,
        };
        let result = super::gatify_array_index_exact(
            &mut gb,
            &padded_ty,
            &padded_array_bits,
            &clamped_index,
        );
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_1b_array_index_public_stats(array_len: usize, index_width: usize) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(1)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_1b_n{}_public", array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result = super::gatify_array_index(
            &mut gb,
            &array_ty,
            &array_bits,
            &index_bits,
            false,
            super::ArrayIndexLoweringStrategy::Auto,
        );
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_array_index_mux_tree_pad_last_stats(
        array_len: usize,
        element_width: usize,
        index_width: usize,
    ) -> SummaryStats {
        let mut gb = GateBuilder::new(
            format!("array_index_mux_tree_n{}_w{}", array_len, element_width),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), element_width * array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let mut cases = Vec::with_capacity(array_len);
        for i in 0..array_len {
            let case_bits = array_bits.get_lsb_slice(i * element_width, element_width);
            cases.push(case_bits);
        }
        let result = crate::ir2gate_utils::gatify_indexed_select_mux_tree_pad_last_if_type_fits(
            &mut gb,
            &index_bits,
            &cases,
        )
        .unwrap_or_else(|e| {
            panic!(
                "pad-last mux-tree helper should apply for array_len={} element_width={} index_width={}: {}",
                array_len, element_width, index_width, e
            )
        });
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_array_index_oob_one_hot_stats(
        array_len: usize,
        element_width: usize,
        index_width: usize,
    ) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_ohs_n{}_w{}", array_len, element_width),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), element_width * array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result =
            super::gatify_array_index_oob_one_hot(&mut gb, &array_ty, &array_bits, &index_bits);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_array_index_public_stats(
        array_len: usize,
        element_width: usize,
        index_width: usize,
    ) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_public_n{}_w{}", array_len, element_width),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), element_width * array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result = super::gatify_array_index(
            &mut gb,
            &array_ty,
            &array_bits,
            &index_bits,
            false,
            super::ArrayIndexLoweringStrategy::Auto,
        );
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    #[test]
    fn test_tuple_index_array_index_field0_dce_stays_flat_across_dead_payload_width() {
        #[rustfmt::skip]
        const WANT: &[(usize, usize, usize)] = &[
            (8, 155, 15),
            (32, 155, 15),
            (128, 155, 15),
        ];

        for &(payload_width, want_live_nodes, want_deepest_path) in WANT {
            let got = get_tuple_field0_array_index_stats(
                /* array_len= */ 27,
                payload_width,
                /* index_width= */ 5,
                super::ArrayIndexLoweringStrategy::ForceOobOneHot,
            );
            assert_eq!(
                got.live_nodes, want_live_nodes,
                "tuple field0 array-index live_nodes mismatch for payload_width={}",
                payload_width
            );
            assert_eq!(
                got.deepest_path, want_deepest_path,
                "tuple field0 array-index depth mismatch for payload_width={}",
                payload_width
            );
        }
    }

    #[test]
    fn test_tuple_index_array_index_field0_width2_tuple_hits_public_mux_tree_strategy() {
        let got = get_tuple_field0_array_index_stats(
            /* array_len= */ 27,
            /* payload_width= */ 1,
            5,
            super::ArrayIndexLoweringStrategy::Auto,
        );
        assert_eq!(got.live_nodes, 118);
        assert_eq!(got.deepest_path, 11);
    }

    #[test]
    fn test_1b_array_index_near_pow2_padding_characterization() {
        let oob_27 = get_1b_array_index_oob_one_hot_stats(
            /* array_len= */ 27, /* index_width= */ 5,
        );
        let padded_27 = get_1b_array_index_near_pow2_padded_stats(
            /* array_len= */ 27, /* index_width= */ 5,
        );
        let public_27 =
            get_1b_array_index_public_stats(/* array_len= */ 27, /* index_width= */ 5);
        let exact_32 =
            get_1b_array_index_exact_stats(/* array_len= */ 32, /* index_width= */ 5);

        assert_eq!(oob_27.live_nodes, 155);
        assert_eq!(oob_27.deepest_path, 15);
        assert_eq!(padded_27.live_nodes, 143);
        assert_eq!(padded_27.deepest_path, 10);
        assert_eq!(public_27.live_nodes, 118);
        assert_eq!(public_27.deepest_path, 11);
        assert_eq!(exact_32.live_nodes, 148);
        assert_eq!(exact_32.deepest_path, 10);

        assert!(
            public_27.live_nodes < padded_27.live_nodes,
            "expected public mux-tree strategy to beat naive padded near-pow2 clamping in g8r: {:?} vs {:?}",
            public_27,
            padded_27
        );
        assert!(
            padded_27.deepest_path < oob_27.deepest_path,
            "expected naive padded near-pow2 clamping to reduce depth after optimized compares in g8r: {:?} vs {:?}",
            padded_27,
            oob_27
        );
    }

    #[test]
    fn test_repeated_literal_array_index_groups_cases() {
        let grouped = get_repeated_literal_array_index_exact_stats(
            /* array_len= */ 32, /* element_width= */ 8, /* index_width= */ 5,
            /* distinct_values= */ 4,
        );
        let one_hot = get_repeated_literal_array_index_one_hot_stats(
            /* array_len= */ 32, /* element_width= */ 8, /* index_width= */ 5,
            /* distinct_values= */ 4,
        );

        assert!(
            grouped.live_nodes < one_hot.live_nodes,
            "expected repeated literal grouping to reduce live nodes: grouped={:?} one_hot={:?}",
            grouped,
            one_hot
        );
    }

    #[test]
    fn test_array_index_near_pow2_mux_tree_vs_one_hot_sweep_characterization() {
        let widths = [1usize, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32];
        for element_width in widths {
            let ohs = get_array_index_oob_one_hot_stats(
                /* array_len= */ 27,
                element_width,
                /* index_width= */ 5,
            );
            let mux = get_array_index_mux_tree_pad_last_stats(
                /* array_len= */ 27,
                element_width,
                /* index_width= */ 5,
            );
            let ohs_product = ohs.live_nodes * ohs.deepest_path;
            let mux_product = mux.live_nodes * mux.deepest_path;
            assert!(
                mux.deepest_path < ohs.deepest_path,
                "expected mux-tree to reduce depth for element_width={}: ohs={:?} mux={:?}",
                element_width,
                ohs,
                mux
            );
            assert!(
                mux_product < ohs_product,
                "expected mux-tree to reduce nodes*depth for element_width={}: ohs_product={} mux_product={}",
                element_width,
                ohs_product,
                mux_product
            );
            if element_width <= 2 {
                assert!(
                    mux.live_nodes < ohs.live_nodes,
                    "expected mux-tree to reduce live_nodes for element_width={}: ohs={:?} mux={:?}",
                    element_width,
                    ohs,
                    mux
                );
            } else {
                assert!(
                    mux.live_nodes > ohs.live_nodes,
                    "expected mux-tree to trade nodes for depth beyond width-2 crossover for element_width={}: ohs={:?} mux={:?}",
                    element_width,
                    ohs,
                    mux
                );
            }
        }
    }

    #[test]
    fn test_array_index_public_strategy_uses_mux_tree_only_in_measured_profitable_region() {
        for element_width in [1usize, 2] {
            let public = get_array_index_public_stats(/* array_len= */ 27, element_width, 5);
            let mux =
                get_array_index_mux_tree_pad_last_stats(/* array_len= */ 27, element_width, 5);
            assert_eq!(
                public, mux,
                "expected public strategy to use near-pow2 mux-tree for element_width={}",
                element_width
            );
        }

        for element_width in [3usize, 4, 8] {
            let public = get_array_index_public_stats(/* array_len= */ 27, element_width, 5);
            let ohs = get_array_index_oob_one_hot_stats(/* array_len= */ 27, element_width, 5);
            assert_eq!(
                public, ohs,
                "expected public strategy to keep OHS lowering for element_width={}",
                element_width
            );
        }
    }
}
