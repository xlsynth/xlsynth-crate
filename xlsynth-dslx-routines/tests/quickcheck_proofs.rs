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

#[test]
fn mul_variant_quickchecks_prove() {
    xlsynth_test_helpers::assert_dslx_quickchecks_prove(
        xlsynth_dslx_routines::mul_variants_path(),
        &[xlsynth_dslx_routines::dslx_dir()],
    );
}

#[test]
fn add_seq_variant_quickchecks_prove() {
    xlsynth_test_helpers::assert_dslx_quickchecks_prove(
        xlsynth_dslx_routines::add_seq_variants_path(),
        &[xlsynth_dslx_routines::dslx_dir()],
    );
}

#[test]
fn mul_seq_variant_quickchecks_prove() {
    xlsynth_test_helpers::assert_dslx_quickchecks_prove(
        xlsynth_dslx_routines::mul_seq_variants_path(),
        &[xlsynth_dslx_routines::dslx_dir()],
    );
}
