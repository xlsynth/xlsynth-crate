// SPDX-License-Identifier: Apache-2.0

use xlsynth_vast::is_system_verilog_keyword;

#[test]
fn recognizes_keywords_from_each_supported_language_revision() {
    for keyword in [
        "initial",
        "module",
        "endfunction",
        "logic",
        "always_comb",
        "priority",
        "unique",
        "unique0",
        "s_eventually",
        "implements",
        "interconnect",
        "nettype",
        "soft",
    ] {
        assert!(is_system_verilog_keyword(keyword), "keyword={keyword}");
    }
}

#[test]
fn keyword_recognition_is_exact_and_case_sensitive() {
    for identifier in [
        "",
        "Initial",
        "INITIAL",
        "module_",
        "_module",
        "module_name",
        "initial_value",
        "unique1",
        "clock",
        "result",
    ] {
        assert!(
            !is_system_verilog_keyword(identifier),
            "identifier={identifier}"
        );
    }
}
