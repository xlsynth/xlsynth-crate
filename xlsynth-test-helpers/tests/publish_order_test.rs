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
        .map(|package| (package.name.clone(), package))
        .collect::<BTreeMap<_, _>>();

    for crate_name in &publish_order {
        let package = workspace_packages.get(crate_name).unwrap_or_else(|| {
            panic!(
                "publish_order.toml names unknown workspace crate `{}`",
                crate_name
            )
        });
        assert_ne!(
            package.publish,
            Some(vec![]),
            "publish_order.toml includes non-publishable crate `{crate_name}`"
        );
    }

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
