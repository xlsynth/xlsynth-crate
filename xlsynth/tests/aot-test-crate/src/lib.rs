// SPDX-License-Identifier: Apache-2.0

//! Integration-test crate that includes generated AOT wrappers from its build
//! script.
//!
//! The public modules mirror build-script outputs so tests can exercise the
//! generated Rust API as a downstream crate would see it.

pub mod add_inputs_aot {
    include!(env!("XLSYNTH_AOT_ADD_INPUTS_RS"));
}

pub mod add_one_aot {
    include!(env!("XLSYNTH_AOT_ADD_ONE_RS"));
}

pub mod compound_shapes_aot {
    include!(env!("XLSYNTH_AOT_COMPOUND_SHAPES_RS"));
}

pub mod empty_tuple_aot {
    include!(env!("XLSYNTH_AOT_EMPTY_TUPLE_RS"));
}

pub mod wide_sizes_aot {
    include!(env!("XLSYNTH_AOT_WIDE_SIZES_RS"));
}

pub mod large_array_tuple_aot {
    include!(env!("XLSYNTH_AOT_LARGE_ARRAY_TUPLE_RS"));
}

pub mod wide_bits_tuple_aot {
    include!(env!("XLSYNTH_AOT_WIDE_BITS_TUPLE_RS"));
}

pub mod trace_assert_aot {
    include!(env!("XLSYNTH_AOT_TRACE_ASSERT_RS"));
}

/// Generated typed DSLX AOT wrapper for the widget/frob fixture.
pub mod widget_frob_aot {
    include!(env!("XLSYNTH_AOT_WIDGET_FROB_RS"));
}

/// Generated typed DSLX AOT wrapper for same-module alias resolution.
pub mod self_alias_widget_aot {
    include!(env!("XLSYNTH_AOT_SELF_ALIAS_WIDGET_RS"));
}

/// Generated typed DSLX AOT wrapper for duplicate imported type names.
pub mod duplicate_widget_aot {
    include!(env!("XLSYNTH_AOT_DUPLICATE_WIDGET_RS"));
}

#[cfg(test)]
mod tests {
    use xlsynth::{IrSBits, IrUBits, XlsynthError};

    use super::widget_frob_aot::widget_types::{
        new_runner, FrobNibbles, SignedWidgetMode, Widget, WidgetMode, WidgetOutcome, WidgetTuning,
    };

    fn ub<const BIT_COUNT: usize>(value: u64) -> IrUBits<BIT_COUNT> {
        IrUBits::from_u64(value).unwrap()
    }

    fn sb<const BIT_COUNT: usize>(value: i64) -> IrSBits<BIT_COUNT> {
        IrSBits::from_i64(value).unwrap()
    }

    // Verifies: generated typed DSLX runners accept and return DSLX bridge types.
    // Catches: regressions that expose structural AOT tuple aliases instead.
    #[test]
    fn widget_frob_aot_uses_generated_typed_dslx_types() -> Result<(), XlsynthError> {
        let garnish: FrobNibbles = [ub(5), ub(6), ub(7)];
        let cases = [
            (sb(-3), SignedWidgetMode::Negative, -4),
            (sb(2), SignedWidgetMode::Positive, 1),
        ];

        for (wobble, signed_mode, expected_wobble) in cases {
            let widget = Widget {
                widget_id: ub(7),
                mode: WidgetMode::Frob,
                frobs: [ub(1), ub(2), ub(3)],
                tuning: WidgetTuning {
                    frob_bias: ub(1),
                    wobble_trim: sb(1),
                },
                wobble,
                signed_mode,
            };

            let mut runner = new_runner()?;
            let result: WidgetOutcome = runner.run(&widget, &garnish)?;

            assert_eq!(result.next_widget_id.to_u64()?, 8);
            assert_eq!(result.selected_frob.to_u64()?, 10);
            assert_eq!(result.mode, WidgetMode::Frob);
            assert_eq!(result.adjusted_wobble.to_i64()?, expected_wobble);
            assert_eq!(result.signed_mode, signed_mode);
        }
        Ok(())
    }

    // Verifies: generated source keeps the typed DSLX public runner signature.
    // Catches: regressions that reintroduce structural Input/Output aliases.
    #[test]
    fn widget_frob_generated_source_has_typed_dslx_public_signature() {
        let source = std::fs::read_to_string(env!("XLSYNTH_AOT_WIDGET_FROB_RS")).unwrap();
        assert!(source.contains("pub fn run("));
        assert!(source.contains("widget: &Widget"));
        assert!(source.contains("garnish: &FrobNibbles"));
        assert!(source.contains(") -> Result<WidgetOutcome, xlsynth::XlsynthError>"));
        assert!(!source.contains("WidgetFrobWidget"));
        assert!(!source.contains("WidgetFrobGarnish"));
        assert!(!source.contains("WidgetFrobReturn"));
        assert!(!source.contains("Input0"));
        assert!(!source.contains("Result<Output"));
        assert!(!source.contains("pub type Output"));
    }

    // Verifies: same-module aliases resolve to the canonical generated type.
    // Catches: self-alias regressions that emit redundant local type aliases.
    #[test]
    fn widget_handle_signature_uses_canonical_bridge_type_without_self_alias(
    ) -> Result<(), XlsynthError> {
        use super::self_alias_widget_aot::self_alias_widget::{new_runner, WidgetHandle};

        let widget = WidgetHandle { widget_id: ub(9) };
        let mut runner = new_runner()?;
        let result: WidgetHandle = runner.run(&widget)?;
        assert_eq!(result.widget_id.to_u64()?, 9);

        let source = std::fs::read_to_string(env!("XLSYNTH_AOT_SELF_ALIAS_WIDGET_RS")).unwrap();
        assert!(source.contains("widget: &WidgetHandle"));
        assert!(source.contains(") -> Result<WidgetHandle, xlsynth::XlsynthError>"));
        assert!(!source.contains("pub type WidgetHandle"));
        Ok(())
    }

    // Verifies: duplicate imported type names use canonical nested paths.
    // Catches: bare-name resolution that confuses sibling DSLX modules.
    #[test]
    fn duplicate_widget_modules_use_canonical_nested_paths() -> Result<(), XlsynthError> {
        use super::duplicate_widget_aot::{bar, foo, frobber};

        let widget = foo::widget::Widget { widget_id: ub(41) };
        let mut runner = frobber::new_runner()?;
        let result: bar::widget::Widget = runner.run(&widget)?;
        assert_eq!(result.widget_id.to_u64()?, 42);

        let source = std::fs::read_to_string(env!("XLSYNTH_AOT_DUPLICATE_WIDGET_RS")).unwrap();
        assert!(source.contains("pub mod foo"));
        assert!(source.contains("pub mod bar"));
        assert!(source.contains("widget: &super::foo::widget::Widget"));
        assert!(source.contains(") -> Result<super::bar::widget::Widget, xlsynth::XlsynthError>"));
        Ok(())
    }
}
