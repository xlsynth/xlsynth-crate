// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use super::{gate, ir, ir2gate};

#[derive(Eq, Hash, PartialEq)]
enum Shape {
    Literal(bool),
    Input(String, usize),
    And([(usize, bool); 2]),
}

/// Source IDs on the original gates, with structural and ordered output
/// witnesses.
pub(super) struct SourcePositions {
    shapes: HashMap<Shape, usize>,
    locations: HashMap<usize, Vec<String>>,
    origins: HashMap<usize, Vec<usize>>,
    gate_by_class: HashMap<usize, usize>,
    outputs: Vec<(String, Vec<usize>)>,
}

impl SourcePositions {
    pub(super) fn capture(
        function: &ir::Fn,
        files: &ir::FileTable,
        graph: &gate::GateFn,
        lowered: &ir2gate::IrToGateMap,
    ) -> Option<Self> {
        if files.id_to_path.is_empty() {
            return None;
        }
        let mut locations = HashMap::new();
        for node in &function.nodes {
            let Some(positions) = &node.pos else { continue };
            let labels: Vec<_> = positions
                .iter()
                .filter_map(|p| p.to_human_string(files))
                .collect();
            if !labels.is_empty() {
                locations.insert(node.text_id, labels);
            }
        }
        if locations.is_empty() {
            return None;
        }
        let mut shapes = HashMap::new();
        let (classes, ambiguous) = structural_classes(graph, &mut shapes);
        let mut origins = HashMap::<usize, Vec<usize>>::new();
        for (node, bits) in lowered {
            let id = function.get_node(*node).text_id;
            if locations.contains_key(&id) {
                for bit in bits.iter_lsb_to_msb() {
                    if classes[bit.node.id].is_some() {
                        // All bits of this IR node are visited together. Record
                        // its ID only on the first visit to each live gate.
                        let ids = origins.entry(bit.node.id).or_default();
                        if ids.last() != Some(&id) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        let outputs = graph
            .outputs
            .iter()
            .map(|out| {
                let bits = out
                    .bit_vector
                    .iter_lsb_to_msb()
                    .map(|b| b.node.id)
                    .collect();
                (out.name.clone(), bits)
            })
            .collect();
        // Duplicate cones can have distinct source locations. Reattach their
        // interior positions only with explicit tracked provenance or an
        // ordered output witness.
        let gate_by_class = origins
            .keys()
            .filter_map(|&node| {
                let class = classes[node]?;
                (!ambiguous.contains(&class)).then_some((class, node))
            })
            .collect();
        Some(Self {
            shapes,
            locations,
            origins,
            gate_by_class,
            outputs,
        })
    }

    pub(super) fn on_final(mut self, graph: &gate::GateFn) -> HashMap<usize, Vec<String>> {
        let (classes, ambiguous) = structural_classes(graph, &mut self.shapes);
        let mut sources = HashMap::<usize, Vec<&String>>::new();
        let mut gate_pairs = HashSet::new();
        let mut locate = |node, id| {
            if let Some(labels) = self.locations.get(&id) {
                sources.entry(node).or_default().extend(labels);
            }
        };
        for (index, class) in classes.iter().enumerate() {
            let Some(class) = class else { continue };
            if !ambiguous.contains(class)
                && let Some(&old) = self.gate_by_class.get(class)
            {
                gate_pairs.insert((old, index));
            }
            for &id in graph.gates[index].get_pir_node_ids() {
                locate(index, id as usize);
            }
        }
        // An output retains its original observation even when its entire cone
        // changes.
        for ((name, old), new) in self.outputs.iter().zip(&graph.outputs) {
            if *name == new.name && old.len() == new.get_bit_count() {
                for (&old, bit) in old.iter().zip(new.bit_vector.iter_lsb_to_msb()) {
                    if self.origins.contains_key(&old) {
                        gate_pairs.insert((old, bit.node.id));
                    }
                }
            }
        }
        for (old, new) in gate_pairs {
            for &id in &self.origins[&old] {
                locate(new, id);
            }
        }
        sources
            .into_iter()
            .map(|(gate, mut labels)| {
                labels.sort_unstable();
                labels.dedup();
                (gate, labels.into_iter().cloned().collect())
            })
            .collect()
    }
}

// Intern the signed child pair itself so structural identity does not depend on
// gate numbering.
fn structural_classes(
    graph: &gate::GateFn,
    shapes: &mut HashMap<Shape, usize>,
) -> (Vec<Option<usize>>, HashSet<usize>) {
    let mut seen = HashSet::new();
    let mut ambiguous = HashSet::new();
    let mut classes = vec![None; graph.gates.len()];
    for operand in graph.post_order_operands(false) {
        let node = operand.node;
        if classes[node.id].is_some() {
            continue;
        }
        let shape = match graph.get(node) {
            gate::AigNode::Input {
                name, lsb_index, ..
            } => Shape::Input(name.clone(), *lsb_index),
            gate::AigNode::Literal { value, .. } => Shape::Literal(*value),
            gate::AigNode::And2 { a, b, .. } => {
                let mut children = [a, b].map(|x| (classes[x.node.id].unwrap(), x.negated));
                children.sort_unstable();
                Shape::And(children)
            }
        };
        let next = shapes.len();
        let id = *shapes.entry(shape).or_insert(next);
        if !seen.insert(id) {
            ambiguous.insert(id);
        }
        classes[node.id] = Some(id);
    }
    (classes, ambiguous)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::dce::dce;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn repeated_literal_output_bits_share_one_origin_and_label() {
        let ir = r#"package locations
file_number 0 "literal.x"
top fn f(x: bits[1] id=1) -> bits[8192] {
  ret zero: bits[8192] = literal(value=0, id=2, pos=[(0, 2, 0)])
}"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        let lowered = ir2gate::gatify(f, ir2gate::GatifyOptions::all_opts_disabled()).unwrap();
        let graph = &lowered.gate_fn;
        let locations =
            SourcePositions::capture(f, &pkg.file_table, graph, &lowered.lowering_map).unwrap();
        let gate = graph.outputs[0].bit_vector.get_lsb(0).node.id;
        assert_eq!(locations.origins, HashMap::from([(gate, vec![2])]));
        assert_eq!(locations.outputs[0].1, vec![gate; 8192]);

        // Split the output between two gates computing zero. The new AND can
        // only receive the location through the ordered output witness.
        let mut builder = GateBuilder::new("f".into(), GateBuilderOptions::no_opt());
        let x = *builder.add_input("x".into(), 1).get_lsb(0);
        let literal = builder.get_false();
        let and = builder.add_and_binary(x, x.negate());
        assert_ne!(literal.node, and.node);
        builder.add_output(
            graph.outputs[0].name.clone(),
            gate::AigBitVector::from_lsb_is_index_0(&[literal, and].repeat(4096)),
        );
        assert_eq!(
            locations.on_final(&builder.build()),
            HashMap::from([
                (literal.node.id, vec!["literal.x:3:1".to_string()]),
                (and.node.id, vec!["literal.x:3:1".to_string()]),
            ])
        );
    }

    #[test]
    fn compaction_cannot_assign_a_dead_nodes_location_to_its_reused_gate_id() {
        let ir = r#"package locations
file_number 0 "source.x"
top fn f(x: bits[1] id=1, y: bits[1] id=2, z: bits[1] id=3) -> bits[1] {
  dead: bits[1] = and(x, y, id=4, pos=[(0, 1, 0)])
  ret live: bits[1] = and(x, z, id=5, pos=[(0, 2, 0)])
}"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        let mut b = GateBuilder::new("f".into(), GateBuilderOptions::opt());
        let x = *b.add_input("x".into(), 1).get_lsb(0);
        let y = *b.add_input("y".into(), 1).get_lsb(0);
        let z = *b.add_input("z".into(), 1).get_lsb(0);
        let dead = b.add_and_binary(x, y);
        let live = b.add_and_binary(x, z);
        let mut lower = ir2gate::IrToGateMap::new();
        for (idx, node) in f.nodes.iter().enumerate() {
            if let Some(bit) = match node.text_id {
                4 => Some(dead),
                5 => Some(live),
                _ => None,
            } {
                lower.insert(
                    ir::NodeRef { index: idx },
                    gate::AigBitVector::from_bit(bit),
                );
            }
        }
        b.add_output("ret".into(), gate::AigBitVector::from_bit(live));
        let pre = b.build();
        let locations = SourcePositions::capture(f, &pkg.file_table, &pre, &lower).unwrap();
        let compact = dce(&pre);
        let id = compact.outputs[0].bit_vector.get_lsb(0).node.id;
        assert_eq!(id, dead.node.id);
        let labels = locations.on_final(&compact);
        assert!(labels[&id].iter().all(|s| s == "source.x:3:1"));
        assert!(!labels[&id].is_empty());
    }

    #[test]
    fn direct_input_output_is_located_without_creating_an_and_gate() {
        let ir = r#"package locations
file_number 0 "pin.x"
top fn f(x: bits[1] id=1) -> bits[1] {
  ret result: bits[1] = identity(x, id=2, pos=[(0, 0, 3)])
}"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        let mut b = GateBuilder::new("f".into(), GateBuilderOptions::opt());
        let pin = b.add_input("x".into(), 1);
        let id = pin.get_lsb(0).node.id;
        let lower = ir2gate::IrToGateMap::from([(f.ret_node_ref.unwrap(), pin.clone())]);
        b.add_output("ret".into(), pin);
        let g = b.build();
        let locations = SourcePositions::capture(f, &pkg.file_table, &g, &lower)
            .unwrap()
            .on_final(&g);
        assert!(locations[&id].iter().all(|s| s == "pin.x:1:4"));
        assert!(!locations[&id].is_empty());
    }

    #[test]
    fn identical_untracked_interior_cones_do_not_cross_attribute_locations() {
        let ir = r#"package locations
file_number 0 "dupe.x"
top fn f(x: bits[1] id=1, y: bits[1] id=2, z: bits[1] id=3) -> (bits[1], bits[1]) {
  a: bits[1] = and(x, y, id=4, pos=[(0, 1, 0)])
  b: bits[1] = and(y, x, id=5, pos=[(0, 2, 0)])
  ret result: (bits[1], bits[1]) = tuple(a, b, id=6)
}"#;
        let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();
        let mut builder = GateBuilder::new("f".into(), GateBuilderOptions::no_opt());
        let x = *builder.add_input("x".into(), 1).get_lsb(0);
        let y = *builder.add_input("y".into(), 1).get_lsb(0);
        let z = *builder.add_input("z".into(), 1).get_lsb(0);
        let a = builder.add_and_binary(x, y);
        let b = builder.add_and_binary(y, x);
        assert_ne!(a.node, b.node);
        let o1 = builder.add_and_binary(a, z);
        let o2 = builder.add_and_binary(b, z.negate());
        builder.add_output("a".into(), gate::AigBitVector::from_bit(o1));
        builder.add_output("b".into(), gate::AigBitVector::from_bit(o2));
        let graph = builder.build();
        let mut lowered = ir2gate::IrToGateMap::new();
        for (index, node) in f.nodes.iter().enumerate() {
            if let Some(bit) = match node.text_id {
                4 => Some(a),
                5 => Some(b),
                _ => None,
            } {
                lowered.insert(ir::NodeRef { index }, gate::AigBitVector::from_bit(bit));
            }
        }
        let locations = SourcePositions::capture(f, &pkg.file_table, &graph, &lowered).unwrap();
        assert!(locations.on_final(&graph).is_empty());
    }
}
