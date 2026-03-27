// SPDX-License-Identifier: Apache-2.0

use super::*;
use xlsynth_pir::ir::ExtNaryAddArchitecture;

/// Rewrites `add(a, b)` into `ext_nary_add(a, b, arch=brent_kung)`.
#[derive(Debug)]
pub struct AddToExtNaryAddTransform;

/// Rewrites an explicit `ext_nary_add` architecture to `ripple_carry`.
#[derive(Debug)]
pub struct ChangeExtNaryAddToRippleCarryTransform;

/// Rewrites an explicit `ext_nary_add` architecture to `brent_kung`.
#[derive(Debug)]
pub struct ChangeExtNaryAddToBrentKungTransform;

/// Rewrites an explicit `ext_nary_add` architecture to `kogge_stone`.
#[derive(Debug)]
pub struct ChangeExtNaryAddToKoggeStoneTransform;

/// Rewrites a width-preserving 2-input `ext_nary_add` back to plain `add`.
#[derive(Debug)]
pub struct BinaryExtNaryAddToAddTransform;

fn is_bits_w(f: &IrFn, nr: NodeRef, w: usize) -> bool {
    matches!(&f.get_node(nr).ty, Type::Bits(ow) if *ow == w)
}

fn binary_ext_nary_add_operands_matching_result(
    f: &IrFn,
    nr: NodeRef,
) -> Option<(NodeRef, NodeRef, usize)> {
    let Type::Bits(w) = f.get_node(nr).ty else {
        return None;
    };
    let NodePayload::ExtNaryAdd { operands, .. } = &f.get_node(nr).payload else {
        return None;
    };
    if operands.len() != 2 {
        return None;
    }
    let lhs = operands[0];
    let rhs = operands[1];
    if !is_bits_w(f, lhs, w) || !is_bits_w(f, rhs, w) {
        return None;
    }
    Some((lhs, rhs, w))
}

fn change_ext_nary_add_candidates(
    f: &IrFn,
    target_arch: ExtNaryAddArchitecture,
) -> Vec<TransformLocation> {
    f.node_refs()
        .into_iter()
        .filter(|nr| {
            matches!(
                f.get_node(*nr).payload,
                NodePayload::ExtNaryAdd {
                    arch: Some(current_arch),
                    ..
                } if current_arch != target_arch
            )
        })
        .map(TransformLocation::Node)
        .collect()
}

fn apply_change_ext_nary_add_architecture(
    f: &mut IrFn,
    loc: &TransformLocation,
    target_arch: ExtNaryAddArchitecture,
    transform_name: &str,
) -> Result<(), String> {
    let nr = match loc {
        TransformLocation::Node(nr) => *nr,
        TransformLocation::RewireOperand { .. } => {
            return Err(format!(
                "{transform_name}: expected TransformLocation::Node, got RewireOperand"
            ));
        }
    };
    let node = f.get_node_mut(nr);
    let NodePayload::ExtNaryAdd { arch, .. } = &mut node.payload else {
        return Err(format!("{transform_name}: expected ext_nary_add"));
    };
    let current_arch =
        arch.ok_or_else(|| format!("{transform_name}: ext_nary_add must have an explicit arch"))?;
    if current_arch == target_arch {
        return Err(format!(
            "{transform_name}: target arch must differ from current arch"
        ));
    }
    *arch = Some(target_arch);
    Ok(())
}

impl PirTransform for AddToExtNaryAddTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::AddToExtNaryAdd
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        f.node_refs()
            .into_iter()
            .filter(|nr| {
                let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(*nr).payload else {
                    return false;
                };
                let Type::Bits(w) = f.get_node(*nr).ty else {
                    return false;
                };
                is_bits_w(f, lhs, w) && is_bits_w(f, rhs, w)
            })
            .map(TransformLocation::Node)
            .collect()
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let nr = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "AddToExtNaryAddTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };
        let Type::Bits(w) = f.get_node(nr).ty else {
            return Err("AddToExtNaryAddTransform: add result must be bits-typed".to_string());
        };
        let NodePayload::Binop(Binop::Add, lhs, rhs) = f.get_node(nr).payload else {
            return Err("AddToExtNaryAddTransform: expected add".to_string());
        };
        if !is_bits_w(f, lhs, w) || !is_bits_w(f, rhs, w) {
            return Err(
                "AddToExtNaryAddTransform: add operands must match the result bit width"
                    .to_string(),
            );
        }
        f.get_node_mut(nr).payload = NodePayload::ExtNaryAdd {
            operands: vec![lhs, rhs],
            arch: Some(ExtNaryAddArchitecture::BrentKung),
        };
        Ok(())
    }
}

impl PirTransform for ChangeExtNaryAddToRippleCarryTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ChangeExtNaryAddToRippleCarry
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        change_ext_nary_add_candidates(f, ExtNaryAddArchitecture::RippleCarry)
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        apply_change_ext_nary_add_architecture(
            f,
            loc,
            ExtNaryAddArchitecture::RippleCarry,
            "ChangeExtNaryAddToRippleCarryTransform",
        )
    }
}

impl PirTransform for ChangeExtNaryAddToBrentKungTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ChangeExtNaryAddToBrentKung
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        change_ext_nary_add_candidates(f, ExtNaryAddArchitecture::BrentKung)
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        apply_change_ext_nary_add_architecture(
            f,
            loc,
            ExtNaryAddArchitecture::BrentKung,
            "ChangeExtNaryAddToBrentKungTransform",
        )
    }
}

impl PirTransform for ChangeExtNaryAddToKoggeStoneTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::ChangeExtNaryAddToKoggeStone
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        change_ext_nary_add_candidates(f, ExtNaryAddArchitecture::KoggeStone)
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        apply_change_ext_nary_add_architecture(
            f,
            loc,
            ExtNaryAddArchitecture::KoggeStone,
            "ChangeExtNaryAddToKoggeStoneTransform",
        )
    }
}

impl PirTransform for BinaryExtNaryAddToAddTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::BinaryExtNaryAddToAdd
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        f.node_refs()
            .into_iter()
            .filter(|nr| binary_ext_nary_add_operands_matching_result(f, *nr).is_some())
            .map(TransformLocation::Node)
            .collect()
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let nr = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "BinaryExtNaryAddToAddTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };
        let (lhs, rhs, _w) =
            binary_ext_nary_add_operands_matching_result(f, nr).ok_or_else(|| {
                "BinaryExtNaryAddToAddTransform: expected a width-preserving binary ext_nary_add"
                    .to_string()
            })?;
        f.get_node_mut(nr).payload = NodePayload::Binop(Binop::Add, lhs, rhs);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    fn find_node_ref<F>(f: &IrFn, pred: F) -> NodeRef
    where
        F: Fn(&NodePayload) -> bool,
    {
        f.node_refs()
            .into_iter()
            .find(|nr| pred(&f.get_node(*nr).payload))
            .expect("expected matching node")
    }

    #[test]
    fn add_to_ext_nary_add_rewrites_binary_add() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(a, b, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let add_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::Binop(Binop::Add, _, _))
        });

        let t = AddToExtNaryAddTransform;
        t.apply(&mut f, &TransformLocation::Node(add_ref))
            .expect("apply");

        match &f.get_node(add_ref).payload {
            NodePayload::ExtNaryAdd { operands, arch } => {
                assert_eq!(operands.len(), 2);
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::BrentKung));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn change_ext_nary_add_to_ripple_carry_rewrites_other_architecture() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, arch=brent_kung, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });

        let mut t = ChangeExtNaryAddToRippleCarryTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(candidates[0], TransformLocation::Node(nr) if nr == nary_ref));
        t.apply(&mut f, &TransformLocation::Node(nary_ref))
            .expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { arch, .. } => {
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::RippleCarry));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn change_ext_nary_add_to_brent_kung_rewrites_other_architecture() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, arch=kogge_stone, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });

        let mut t = ChangeExtNaryAddToBrentKungTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(candidates[0], TransformLocation::Node(nr) if nr == nary_ref));
        t.apply(&mut f, &TransformLocation::Node(nary_ref))
            .expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { arch, .. } => {
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::BrentKung));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn change_ext_nary_add_to_kogge_stone_rewrites_other_architecture() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, arch=ripple_carry, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });

        let mut t = ChangeExtNaryAddToKoggeStoneTransform;
        let candidates = t.find_candidates(&f);
        assert_eq!(candidates.len(), 1);
        assert!(matches!(candidates[0], TransformLocation::Node(nr) if nr == nary_ref));
        t.apply(&mut f, &TransformLocation::Node(nary_ref))
            .expect("apply");

        match &f.get_node(nary_ref).payload {
            NodePayload::ExtNaryAdd { arch, .. } => {
                assert_eq!(*arch, Some(ExtNaryAddArchitecture::KoggeStone));
            }
            other => panic!("expected ext_nary_add after rewrite, got {other:?}"),
        }
    }

    #[test]
    fn change_ext_nary_add_to_target_skips_same_architecture() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, arch=brent_kung, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();

        let mut to_ripple = ChangeExtNaryAddToRippleCarryTransform;
        let mut to_brent = ChangeExtNaryAddToBrentKungTransform;
        let mut to_kogge = ChangeExtNaryAddToKoggeStoneTransform;

        assert_eq!(to_ripple.find_candidates(&f).len(), 1);
        assert!(to_brent.find_candidates(&f).is_empty());
        assert_eq!(to_kogge.find_candidates(&f).len(), 1);
    }

    #[test]
    fn binary_ext_nary_add_to_add_rewrites_width_preserving_case() {
        let ir_text = r#"fn t(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, arch=ripple_carry, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let mut f = parser.parse_fn().unwrap();
        let nary_ref = find_node_ref(&f, |payload| {
            matches!(payload, NodePayload::ExtNaryAdd { .. })
        });

        let t = BinaryExtNaryAddToAddTransform;
        t.apply(&mut f, &TransformLocation::Node(nary_ref))
            .expect("apply");

        assert!(matches!(
            f.get_node(nary_ref).payload,
            NodePayload::Binop(Binop::Add, _, _)
        ));
    }

    #[test]
    fn binary_ext_nary_add_to_add_skips_resizing_case() {
        let ir_text = r#"fn t(a: bits[4] id=1, b: bits[6] id=2) -> bits[8] {
  ret ext_nary_add.3: bits[8] = ext_nary_add(a, b, arch=ripple_carry, id=3)
}"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let mut t = BinaryExtNaryAddToAddTransform;

        assert!(
            t.find_candidates(&f).is_empty(),
            "expected resizing ext_nary_add to be excluded from add fallback"
        );
    }
}
