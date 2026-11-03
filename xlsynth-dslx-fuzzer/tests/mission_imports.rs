// SPDX-License-Identifier: Apache-2.0

use xlsynth_dslx_fuzzer::ast::Fragment;
use xlsynth_dslx_fuzzer::compound::{ArrayOp, CompoundOp};
use xlsynth_dslx_fuzzer::map_fragment::{MapBody, MapFragment};
use xlsynth_dslx_fuzzer::{GenerateOptions, Grammar, generate_with_grammar};

#[test]
fn local_mission_substitutes_both_stdlib_calls_without_changing_reference_ir() -> Result<(), String>
{
    let cases = [
        (
            [0, 0, 0, 22, 0, 1, 1],
            CompoundOp::Map(MapFragment::new(MapBody::IsEven, 1)?),
            CompoundOp::Map(MapFragment::new(MapBody::LocalIsEven, 1)?),
        ),
        (
            [0, 0, 0, 14, 0, 1, 5],
            CompoundOp::Array(ArrayOp::Enumerate),
            CompoundOp::Array(ArrayOp::LocalEnumerate),
        ),
    ];
    for (input, stdlib_op, local_op) in cases {
        let stdlib = generate_with_grammar(&input, GenerateOptions::default(), Grammar::Mission)?;
        let local =
            generate_with_grammar(&input, GenerateOptions::default(), Grammar::MissionLocal)?;
        assert_eq!(
            stdlib.expected_ir()?.to_string(),
            local.expected_ir()?.to_string()
        );
        for (sample, expected) in [(&stdlib, stdlib_op), (&local, local_op)] {
            let [Fragment::Compound { op, .. }] = sample.helpers()[0].body(false).fragments()
            else {
                panic!("expected one compound fragment");
            };
            assert_eq!(*op, expected);
            let imports_std = sample
                .sources()?
                .iter()
                .any(|module| module.text.lines().any(|line| line == "import std;"));
            assert_eq!(imports_std, expected == stdlib_op);
            sample.prepare()?;
        }
    }
    Ok(())
}
