// SPDX-License-Identifier: Apache-2.0

#[test]
fn clz_variant_quickchecks_prove() {
    xlsynth_test_helpers::assert_dslx_quickchecks_prove(
        xlsynth_dslx_routines::clz_variants_path(),
        &[xlsynth_dslx_routines::dslx_dir()],
    );
}

#[test]
fn add_variant_quickchecks_prove() {
    xlsynth_test_helpers::assert_dslx_quickchecks_prove(
        xlsynth_dslx_routines::add_variants_path(),
        &[xlsynth_dslx_routines::dslx_dir()],
    );
}
