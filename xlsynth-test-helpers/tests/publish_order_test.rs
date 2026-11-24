// SPDX-License-Identifier: Apache-2.0

use cargo_metadata::{DependencyKind, MetadataCommand, Package};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

fn load_publish_order(workspace_root: &Path) -> Vec<String> {
    let contents = std::fs::read_to_string(workspace_root.join("publish_order.toml")).unwrap();
    let value: toml::Value = toml::from_str(&contents).unwrap();
    value["crates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().to_string())
        .collect()
}

/// Returns intra-workspace dependencies that survive crate packaging.
fn publish_time_dependencies(package: &Package, publish_names: &BTreeSet<String>) -> Vec<String> {
    let mut dependencies = package
        .dependencies
        .iter()
        .filter(|dependency| publish_names.contains(&dependency.name))
        .filter(|dependency| dependency.path.is_some())
        // Cargo allows local dev-dependency cycles, and path-only dev-dependencies
        // without a registry version are omitted from the packaged manifest. Those
        // edges matter for local workspace policy, but not for crates.io publish
        // order.
        .filter(|dependency| dependency.req.to_string() != "*")
        .filter(|dependency| {
            matches!(
                dependency.kind,
                DependencyKind::Normal | DependencyKind::Development | DependencyKind::Build
            )
        })
        .map(|dependency| dependency.name.clone())
        .collect::<Vec<_>>();
    dependencies.sort();
    dependencies.dedup();
    dependencies
}

#[test]
fn publish_order_is_topological_for_publish_time_dependencies() {
    let metadata = MetadataCommand::new().exec().unwrap();
    let workspace_root = metadata.workspace_root.as_std_path();
    let publish_order = load_publish_order(workspace_root);
    let publish_names = publish_order.iter().cloned().collect::<BTreeSet<_>>();

    assert_eq!(
        publish_names.len(),
        publish_order.len(),
        "publish_order.toml contains duplicate crate names"
    );

    let workspace_member_ids = metadata.workspace_members.iter().collect::<BTreeSet<_>>();
    let workspace_packages = metadata
        .packages
        .iter()
        .filter(|package| workspace_member_ids.contains(&package.id))
        .map(|package| (package.name.to_string(), package))
        .collect::<BTreeMap<_, _>>();

    let publishable_workspace_crates = workspace_packages
        .iter()
        .filter(|(_, package)| package.publish != Some(vec![]))
        .map(|(crate_name, _)| crate_name.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        publish_names, publishable_workspace_crates,
        "publish_order.toml must list every publishable workspace crate, and omitted workspace crates must set `publish = false`"
    );

    let positions = publish_order
        .iter()
        .enumerate()
        .map(|(index, crate_name)| (crate_name.as_str(), index))
        .collect::<BTreeMap<_, _>>();

    let mut violations = Vec::new();
    for crate_name in &publish_order {
        let package = workspace_packages[crate_name];
        for dependency_name in publish_time_dependencies(package, &publish_names) {
            if positions[dependency_name.as_str()] > positions[crate_name.as_str()] {
                violations.push(format!("{crate_name} depends on {dependency_name}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "publish_order.toml is not topological for publish-time dependencies: {}",
        violations.join("; ")
    );
}

#[test]
fn workspace_path_dependencies_are_acyclic_including_dev_dependencies() {
    let metadata = MetadataCommand::new().no_deps().exec().unwrap();
    let names_by_path = metadata
        .packages
        .iter()
        .map(|package| {
            (
                package.manifest_path.parent().unwrap().to_path_buf(),
                package.name.to_string(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let graph = metadata
        .packages
        .iter()
        .map(|package| {
            let dependencies = package
                .dependencies
                .iter()
                .filter_map(|dependency| dependency.path.as_ref())
                .filter_map(|path| names_by_path.get(path).cloned())
                .collect::<BTreeSet<_>>();
            (package.name.to_string(), dependencies)
        })
        .collect::<BTreeMap<_, _>>();

    let mut visited = BTreeSet::new();
    let mut stack = Vec::new();
    for name in graph.keys() {
        if let Some(cycle) = dependency_cycle(name, &graph, &mut visited, &mut stack) {
            panic!("workspace path dependency cycle: {}", cycle.join(" -> "));
        }
    }
}

/// Reports a cycle in normal, build, or development workspace path
/// dependencies.
fn dependency_cycle<'a>(
    name: &'a str,
    graph: &'a BTreeMap<String, BTreeSet<String>>,
    visited: &mut BTreeSet<&'a str>,
    stack: &mut Vec<&'a str>,
) -> Option<Vec<&'a str>> {
    if let Some(start) = stack.iter().position(|entry| *entry == name) {
        let mut cycle = stack[start..].to_vec();
        cycle.push(name);
        return Some(cycle);
    }
    if !visited.insert(name) {
        return None;
    }
    stack.push(name);
    for dependency in &graph[name] {
        if let Some(cycle) = dependency_cycle(dependency, graph, visited, stack) {
            return Some(cycle);
        }
    }
    stack.pop();
    None
}
