// SPDX-License-Identifier: Apache-2.0

use pretty_assertions::assert_eq;
use xlsynth::{dslx, dslx_bridge::convert_imported_module, sv_bridge_builder::SvBridgeBuilder};

/// Tests that we can convert the whole "structure_zoo.x" file to SystemVerilog.
#[test]
fn test_sv_bridge_structure_zoo() {
    let mut import_data = dslx::ImportData::default();

    let common_zoo_relpath = "tests/common_zoo.x";
    let common_zoo_dslx = std::fs::read_to_string(common_zoo_relpath).unwrap();
    let common_zoo = dslx::parse_and_typecheck(
        &common_zoo_dslx,
        common_zoo_relpath,
        "common_zoo",
        &mut import_data,
    )
    .unwrap();

    let zoo_relpath = "tests/structure_zoo.x";
    let zoo_dslx = std::fs::read_to_string(zoo_relpath).unwrap();
    let zoo = dslx::parse_and_typecheck(&zoo_dslx, zoo_relpath, "structure_zoo", &mut import_data)
        .unwrap();

    // Make a builder to convert the "imported" module.
    let imported_sv = {
        let mut builder = SvBridgeBuilder::new();
        convert_imported_module(&common_zoo, &mut builder).unwrap();
        let contents = builder.build();
        format!(
            "package common_zoo_sv_pkg;\n{}endpackage : common_zoo_sv_pkg",
            contents
        )
    };

    // Check that our generated `imported_sv` matches a golden expectation too,
    // because why not.
    test_helpers::assert_valid_sv(&imported_sv);
    let imported_sv_golden = std::fs::read_to_string("tests/want_common_zoo.golden.sv").unwrap();
    assert_eq!(imported_sv, imported_sv_golden);

    // Make a builder to convert the "importer" module.
    let mut builder = SvBridgeBuilder::new();
    convert_imported_module(&zoo, &mut builder).unwrap();
    let got_sv = builder.build();

    // Check that the SV we got is also valid SV.
    test_helpers::assert_valid_sv_flist(&[
        test_helpers::FlistEntry {
            filename: "common_zoo.sv".to_string(),
            contents: imported_sv,
        },
        test_helpers::FlistEntry {
            filename: "structure_zoo.sv".to_string(),
            contents: got_sv.clone(),
        },
    ]);

    // Check that our generated SV matches the golden (expected) output.
    let structure_zoo_sv_golden =
        std::fs::read_to_string("tests/want_structure_zoo.golden.sv").unwrap();
    assert_eq!(got_sv, structure_zoo_sv_golden);
}
