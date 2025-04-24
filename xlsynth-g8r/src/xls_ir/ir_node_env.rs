// SPDX-License-Identifier: Apache-2.0

use crate::xls_ir::ir;
use std::collections::HashMap;

#[derive(Debug)]
pub enum NameOrId {
    Name(String),
    Id(usize),
}

pub struct IrNodeEnv {
    name_to_node: HashMap<String, ir::NodeRef>,
    id_to_node: HashMap<usize, ir::NodeRef>,
}

impl IrNodeEnv {
    pub fn new() -> Self {
        Self {
            name_to_node: HashMap::new(),
            id_to_node: HashMap::new(),
        }
    }

    /// We add nodes by providing their `id` (always) but also an optional
    /// `name` they are also known by.
    pub fn add(&mut self, name: Option<String>, id: usize, node: ir::NodeRef) {
        assert!(id > 0, "Invalid node id {}, must be greater than zero", id);
        log::debug!(
            "NodeEnv::add; name: {:?}; id: {:?}; node: {:?}",
            name,
            id,
            node
        );
        if let Some(name) = name {
            self.name_to_node.insert(name, node);
        }
        self.id_to_node.insert(id, node);
    }

    /// We look up nodes by either their `id` or `name` -- in references either
    /// the a name is provided (e.g. in the case of params or named nodes)
    /// or the referred-to node is unnamed and we refer to it via its
    /// `operator.id` in the text, where the operator can be implicit
    /// so only the `id` is required.
    pub fn name_id_to_ref(&self, key: &NameOrId) -> Option<&ir::NodeRef> {
        match key {
            NameOrId::Id(id) => self.id_to_node.get(&id),
            NameOrId::Name(name) => self.name_to_node.get(name.as_str()),
        }
    }

    pub fn keys(&self) -> Vec<&String> {
        self.name_to_node.keys().collect::<Vec<&String>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_and_and_lookup() {
        let mut node_env = IrNodeEnv::new();
        node_env.add(Some("ugt".to_string()), 92055, ir::NodeRef { index: 92055 });
        let node_ref = node_env.name_id_to_ref(&NameOrId::Id(92055));
        assert!(node_ref.is_some());
        assert_eq!(node_ref.unwrap().index, 92055);
    }
}
