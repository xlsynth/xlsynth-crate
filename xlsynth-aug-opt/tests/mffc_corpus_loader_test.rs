// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::fs;
use std::path::Path;

use mffc_corpus::Expectations;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

pub mod mffc_corpus;

const INPUT: &str = r#"// SPDX-License-Identifier: Apache-2.0

package fixture

top fn entry(x: bits[1] id=1) -> bits[1] {
  ret not.2: bits[1] = not(x, id=2)
}
"#;

const SHIFT: &str = r#"shift {
  require_improvement: false
  limits { and_nodes_max: 0 graph_le_max: 0 depth_max: 0 }
}"#;

const SPLIT_ADDER: &str = r#"split_adder {
  disabled_limits { and_nodes_max: 0 graph_le_max: 0 depth_max: 0 }
  enabled_limits { and_nodes_max: 0 graph_le_max: 0 depth_max: 0 }
}"#;

/// Creates an isolated corpus using the same profile as the checked-in
/// fixtures.
fn corpus_directory() -> Result<TempDir, std::io::Error> {
    let directory = tempfile::tempdir()?;
    fs::write(
        directory.path().join("profile.json"),
        include_str!("fixtures/mffc_regressions/profile.json"),
    )?;
    Ok(directory)
}

/// Writes a fixture and sidecar whose hash covers exactly the supplied IR text.
fn write_case(root: &Path, stem: &str, text: &str, expectations: &str) -> std::io::Result<()> {
    fs::write(root.join(format!("{stem}.ir")), text)?;
    fs::write(
        root.join(format!("{stem}.textproto")),
        format!(
            "# SPDX-License-Identifier: Apache-2.0\nsha256: \"{:x}\"\n{expectations}\n",
            Sha256::digest(text.as_bytes())
        ),
    )
}

#[test]
fn discovers_sorted_pairs_and_derives_names_and_top() -> Result<(), Box<dyn Error>> {
    let directory = corpus_directory()?;
    write_case(directory.path(), "z_last", INPUT, SHIFT)?;
    write_case(directory.path(), "a_first", INPUT, SHIFT)?;

    let corpus = mffc_corpus::load_from_dir(directory.path())?;
    assert_eq!(
        corpus
            .cases
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>(),
        ["a_first", "z_last"]
    );
    for case in corpus.cases {
        assert_eq!(case.text, INPUT);
        assert_eq!(case.original.name, "entry");
        let Expectations::Shift {
            require_improvement,
            limits,
        } = case.expectations
        else {
            panic!("expected shift expectations");
        };
        // Explicit false/zero bounds must remain distinct from omitted fields.
        assert!(!require_improvement);
        assert_eq!(limits.and_nodes_max, 0);
        assert_eq!(limits.graph_le_max, 0.0);
        assert_eq!(limits.depth_max, 0);
    }
    Ok(())
}

#[test]
fn rejects_unpaired_or_empty_corpora() -> Result<(), Box<dyn Error>> {
    for removed in [vec!["textproto"], vec!["ir"], vec!["ir", "textproto"]] {
        let directory = corpus_directory()?;
        write_case(directory.path(), "sample", INPUT, SHIFT)?;
        for extension in &removed {
            fs::remove_file(directory.path().join(format!("sample.{extension}")))?;
        }
        assert!(
            mffc_corpus::load_from_dir(directory.path()).is_err(),
            "accepted corpus after removing {removed:?}"
        );
    }
    Ok(())
}

#[test]
fn rejects_changed_ir_and_undeclared_top() -> Result<(), Box<dyn Error>> {
    let directory = corpus_directory()?;
    write_case(directory.path(), "sample", INPUT, SHIFT)?;
    fs::write(
        directory.path().join("sample.ir"),
        format!("{INPUT}// Changed after recording the hash.\n"),
    )?;
    assert!(mffc_corpus::load_from_dir(directory.path()).is_err());

    // Recompute the hash so only the missing explicit top is invalid.
    write_case(
        directory.path(),
        "sample",
        &INPUT.replace("top fn", "fn"),
        SHIFT,
    )?;
    assert!(mffc_corpus::load_from_dir(directory.path()).is_err());
    Ok(())
}

#[test]
fn requires_explicit_expectations_and_valid_bounds() -> Result<(), Box<dyn Error>> {
    for (description, expectations) in [
        ("missing family", String::new()),
        (
            "missing boolean",
            SHIFT.replace("require_improvement: false", ""),
        ),
        ("missing AND bound", SHIFT.replace("and_nodes_max: 0", "")),
        ("missing LE bound", SHIFT.replace("graph_le_max: 0", "")),
        ("missing depth bound", SHIFT.replace("depth_max: 0", "")),
        ("unknown field", format!("{SHIFT}\nunknown: 1")),
        (
            "negative LE",
            SHIFT.replace("graph_le_max: 0", "graph_le_max: -1"),
        ),
        (
            "infinite LE",
            SHIFT.replace("graph_le_max: 0", "graph_le_max: inf"),
        ),
        (
            "NaN LE",
            SHIFT.replace("graph_le_max: 0", "graph_le_max: nan"),
        ),
    ] {
        let directory = corpus_directory()?;
        write_case(directory.path(), "sample", INPUT, &expectations)?;
        assert!(
            mffc_corpus::load_from_dir(directory.path()).is_err(),
            "accepted {description}"
        );
    }
    Ok(())
}

#[test]
fn rejects_invalid_tolerance() -> Result<(), Box<dyn Error>> {
    for tolerance in ["-1", "1e999", "NaN"] {
        let directory = corpus_directory()?;
        write_case(directory.path(), "sample", INPUT, SHIFT)?;
        let profile_path = directory.path().join("profile.json");
        let mut profile: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&profile_path)?)?;
        profile["graph_le_tolerance"] = serde_json::Value::String("TOLERANCE".into());
        fs::write(
            &profile_path,
            serde_json::to_string(&profile)?.replace("\"TOLERANCE\"", tolerance),
        )?;
        assert!(
            mffc_corpus::load_from_dir(directory.path()).is_err(),
            "accepted tolerance {tolerance}"
        );
    }
    Ok(())
}

#[test]
fn validates_ordering_peers_after_loading_all_cases() -> Result<(), Box<dyn Error>> {
    for (peer, valid) in [
        ("z_target", true),
        ("missing", false),
        ("a_variant", false),
        ("shift", false),
    ] {
        let directory = corpus_directory()?;
        write_case(directory.path(), "z_target", INPUT, SPLIT_ADDER)?;
        write_case(directory.path(), "shift", INPUT, SHIFT)?;
        write_case(
            directory.path(),
            "a_variant",
            INPUT,
            &SPLIT_ADDER.replacen(
                "split_adder {",
                &format!("split_adder {{ same_as: \"{peer}\""),
                1,
            ),
        )?;
        let result = mffc_corpus::load_from_dir(directory.path());
        assert_eq!(result.is_ok(), valid, "same_as={peer}: {result:?}");
    }
    Ok(())
}
