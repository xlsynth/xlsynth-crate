// SPDX-License-Identifier: Apache-2.0

use pretty_assertions::assert_eq;
use xlsynth::{dslx, dslx_bridge::convert_leaf_module, sv_bridge_builder::SvBridgeBuilder};

/// Tests that we can convert the whole "structure_zoo.x" file to SystemVerilog.
#[test]
fn test_sv_bridge_structure_zoo() {
    let dslx = std::fs::read_to_string("tests/structure_zoo.x").unwrap();
    let mut import_data = dslx::ImportData::default();
    let path = std::path::PathBuf::from("tests/structure_zoo.x");
    let mut builder = SvBridgeBuilder::new();
    convert_leaf_module(&mut import_data, &dslx, &path, &mut builder).unwrap();
    let structure_zoo_sv_golden = std::fs::read_to_string("tests/want_structure_zoo.sv").unwrap();
    assert_eq!(builder.build(), structure_zoo_sv_golden);
}
