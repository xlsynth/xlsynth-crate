// SPDX-License-Identifier: Apache-2.0

//! Relate a checked DSLX type annotation to the concrete type of its value.

use xlsynth::dslx;

use super::ImportedStructs;
use crate::check::MAX_ELABORATION_DEPTH;

/// Follow local aliases so an array or tuple can retain its written member
/// annotations.
pub(super) fn underlying_annotation(mut annotation: dslx::TypeAnnotation) -> dslx::TypeAnnotation {
    for _ in 0..MAX_ELABORATION_DEPTH {
        let Some(reference) = annotation.to_type_ref_type_annotation() else {
            break;
        };
        if reference.get_parametric_count() != 0 {
            break;
        }
        let Some(alias) = reference
            .get_type_ref()
            .get_type_definition()
            .to_type_alias()
        else {
            break;
        };
        annotation = alias.get_type_annotation();
    }
    annotation
}

impl ImportedStructs {
    /// Record the imported spelling of a checked value at the same position in
    /// its written type.
    pub(super) fn observe_typed_annotation(
        &mut self,
        annotation: &dslx::TypeAnnotation,
        ty: &dslx::Type,
    ) {
        let Some(reference) = annotation.to_type_ref_type_annotation() else {
            return;
        };
        let definition = reference.get_type_ref().get_type_definition();
        let Some(colon) = definition.to_colon_ref() else {
            return;
        };
        let Some(import) = colon.resolve_import_subject() else {
            return;
        };
        let subject = import.get_subject().join(".");
        self.subjects.insert(subject.clone());
        // A qualified reference that typechecked across a module boundary is
        // public. Preserve only spellings that can be emitted without
        // parametric arguments.
        if reference.get_parametric_count() == 0 && ty.is_struct() {
            self.record_annotation_path(ty, format!("{subject}::{}", colon.get_attr()));
        }
    }
}
