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

/// Generated typed DSLX AOT wrapper for concrete parametric structs.
pub mod parametric_box_aot {
    include!(env!("XLSYNTH_AOT_PARAMETRIC_BOX_RS"));
}

/// Generated typed DSLX AOT wrapper for parametric shape examples.
pub mod parametric_shapes_aot {
    include!(env!("XLSYNTH_AOT_PARAMETRIC_SHAPES_RS"));
}

/// Generated typed DSLX AOT wrapper for parametric array examples.
pub mod parametric_arrays_aot {
    include!(env!("XLSYNTH_AOT_PARAMETRIC_ARRAYS_RS"));
}

/// Generated typed DSLX AOT wrapper for parametric value examples.
pub mod parametric_values_aot {
    include!(env!("XLSYNTH_AOT_PARAMETRIC_VALUES_RS"));
}

/// Generated typed DSLX AOT wrapper for imported parametric examples.
pub mod parametric_imports_aot {
    include!(env!("XLSYNTH_AOT_PARAMETRIC_IMPORTS_RS"));
}

/// Generated typed DSLX AOT wrapper for duplicate imported type names.
pub mod duplicate_widget_aot {
    include!(env!("XLSYNTH_AOT_DUPLICATE_WIDGET_RS"));
}

/// Generated typed DSLX AOT package for shared public widget types.
pub mod shared_widget_package_aot {
    include!(env!("XLSYNTH_AOT_SHARED_WIDGET_PACKAGE_RS"));
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

    // Verifies: concrete DSLX parametric structs get generated Rust names
    // suffixed by evaluated parameter values.
    // Catches: regressions that emit unspecialized `Box` or field type `bits[N]`.
    #[test]
    fn parametric_box_aot_uses_concrete_generated_struct() -> Result<(), XlsynthError> {
        use super::parametric_box_aot::parametric_box::{new_runner, Box8};

        let input = Box8 { value: ub(42) };
        let mut runner = new_runner()?;
        let result: Box8 = runner.run(&input)?;
        assert_eq!(result.value.to_u64()?, 42);

        let source = std::fs::read_to_string(env!("XLSYNTH_AOT_PARAMETRIC_BOX_RS")).unwrap();
        assert!(source.contains("pub struct Box__N_8"));
        assert!(source.contains("pub type Box8 = Box__N_8;"));
        assert!(source.contains("x: &Box8"));
        assert!(source.contains(") -> Result<Box8, xlsynth::XlsynthError>"));
        Ok(())
    }

    // Verifies: generated AOT wrappers execute concrete parametric aliases with
    // distinct parameter values.
    // Catches: regressions in concrete struct naming or per-field lowering.
    #[test]
    fn parametric_shapes_aot_executes_concrete_aliases() -> Result<(), XlsynthError> {
        use super::parametric_shapes_aot::parametric_shapes::{new_runner, Box16, Box8, Matrix2x3};

        let box8 = Box8 { value: ub(8) };
        let box16 = Box16 { value: ub(16) };
        let matrix = Matrix2x3 {
            rows: [[ub(1), ub(2), ub(3)], [ub(4), ub(5), ub(6)]],
        };

        let mut runner = new_runner()?;
        let result = runner.run(&box8, &box16, &matrix)?;

        assert_eq!(result.box8.value.to_u64()?, 8);
        assert_eq!(result.box16.value.to_u64()?, 16);
        assert_eq!(result.matrix.rows[0][2].to_u64()?, 3);
        assert_eq!(result.matrix.rows[1][1].to_u64()?, 5);
        Ok(())
    }

    // Verifies: aliases to arrays of concrete parametric structs execute end to
    // end through the typed AOT wrapper.
    // Catches: recursive decode regressions that construct unspecialized `Box`.
    #[test]
    fn parametric_arrays_aot_executes_alias_to_parametric_array() -> Result<(), XlsynthError> {
        use super::parametric_arrays_aot::parametric_arrays::{
            new_runner, ArrayBox4, Box8Array4, Box__N_16, Box__N_8, OuterBox,
        };

        let array_box = ArrayBox4 {
            items: [ub(10), ub(11), ub(12), ub(13)],
        };
        let box_array: Box8Array4 = [
            Box__N_8 { value: ub(20) },
            Box__N_8 { value: ub(21) },
            Box__N_8 { value: ub(22) },
            Box__N_8 { value: ub(23) },
        ];
        let outer = OuterBox {
            inner: Box__N_8 { value: ub(30) },
            wider: Box__N_16 { value: ub(300) },
        };

        let mut runner = new_runner()?;
        let result = runner.run(&array_box, &box_array, &outer)?;

        assert_eq!(result.array_box.items[3].to_u64()?, 13);
        assert_eq!(result.box_array[2].value.to_u64()?, 22);
        assert_eq!(result.outer.inner.value.to_u64()?, 30);
        assert_eq!(result.outer.wider.value.to_u64()?, 300);
        Ok(())
    }

    // Verifies: parametric values from expressions, signed literals, and wide
    // literals all execute after suffix generation.
    // Catches: concrete-name support that compiles but fails AOT conversion.
    #[test]
    fn parametric_values_aot_executes_value_kinds() -> Result<(), XlsynthError> {
        use super::parametric_values_aot::parametric_values::{
            new_runner, ExprBox8, HugeTag, NegativeTag,
        };

        let expr_box = ExprBox8 { value: ub(77) };
        let negative = NegativeTag { payload: ub(88) };
        let huge = HugeTag { payload: ub(99) };

        let mut runner = new_runner()?;
        let result = runner.run(&expr_box, &negative, &huge)?;

        assert_eq!(result.expr_box.value.to_u64()?, 77);
        assert_eq!(result.negative.payload.to_u64()?, 88);
        assert_eq!(result.huge.payload.to_u64()?, 99);
        Ok(())
    }

    // Verifies: direct imported concrete parametric instantiations execute and
    // owner-module emission handles nested arrays plus multi-parameter shapes.
    // Catches: regressions back to alias-only imported parametric support.
    #[test]
    fn parametric_imports_aot_executes_direct_imports() -> Result<(), XlsynthError> {
        use super::parametric_imports_aot::{parametric_imports, parametric_lib};
        use parametric_imports::{Box__N_8, LocalMode, Mixed__N_4};

        let mixed = Mixed__N_4 {
            mode: LocalMode::Busy,
            remote: parametric_lib::RemotePlain { id: ub(40) },
            nested: [[ub(1), ub(2), ub(3), ub(4)], [ub(5), ub(6), ub(7), ub(8)]],
            boxes: [Box__N_8 { value: ub(50) }, Box__N_8 { value: ub(51) }],
            remote_boxes: [
                parametric_lib::RemoteBox__N_8 { value: ub(52) },
                parametric_lib::RemoteBox__N_8 { value: ub(53) },
            ],
        };
        let imported_direct = parametric_lib::RemoteBox__N_8 { value: ub(60) };
        let imported_pair = parametric_lib::RemotePair__A_8__B_23 {
            left: ub(61),
            right: ub(62),
        };

        let mut runner = parametric_imports::new_runner()?;
        let result = runner.run(&mixed, &imported_direct, &imported_pair)?;

        assert_eq!(result.mixed.mode, LocalMode::Busy);
        assert_eq!(result.mixed.remote.id.to_u64()?, 40);
        assert_eq!(result.mixed.nested[1][2].to_u64()?, 7);
        assert_eq!(result.mixed.boxes[1].value.to_u64()?, 51);
        assert_eq!(result.mixed.remote_boxes[1].value.to_u64()?, 53);
        assert_eq!(result.imported_direct.value.to_u64()?, 60);
        assert_eq!(result.imported_pair.left.to_u64()?, 61);
        assert_eq!(result.imported_pair.right.to_u64()?, 62);
        Ok(())
    }

    // Verifies: small golden wrappers capture varied concrete parametric type
    // spellings used in public function signatures.
    // Catches: regressions in suffix generation, arrays, and imported paths.
    #[test]
    fn parametric_generated_sources_have_interesting_type_names() {
        let shapes = std::fs::read_to_string(env!("XLSYNTH_AOT_PARAMETRIC_SHAPES_RS")).unwrap();
        assert!(shapes.contains("pub struct Box__N_8"));
        assert!(shapes.contains("pub struct Box__N_16"));
        assert!(shapes.contains("pub struct Matrix__R_2__C_3"));
        assert!(shapes.contains("box8: &Box8"));

        let arrays = std::fs::read_to_string(env!("XLSYNTH_AOT_PARAMETRIC_ARRAYS_RS")).unwrap();
        assert!(arrays.contains("pub struct ArrayBox__N_4"));
        assert!(arrays.contains("pub type Box8Array4 = [Box__N_8; 4];"));
        assert!(arrays.contains("box_array: &Box8Array4"));
        assert!(arrays.contains("pub inner: Box__N_8"));
        assert!(arrays.contains("pub wider: Box__N_16"));

        let values = std::fs::read_to_string(env!("XLSYNTH_AOT_PARAMETRIC_VALUES_RS")).unwrap();
        assert!(values.contains("pub struct ExprBox__N_8"));
        assert!(values.contains("pub struct SignedTag__S_m3"));
        assert!(values.contains("pub struct WideTag__W_18446744073709551616"));

        let imports = std::fs::read_to_string(env!("XLSYNTH_AOT_PARAMETRIC_IMPORTS_RS")).unwrap();
        assert!(imports.contains("pub struct Mixed__N_4"));
        assert!(imports.contains("pub remote: super::parametric_lib::RemotePlain"));
        assert!(imports.contains("pub boxes: [Box__N_8; 2]"));
        assert!(imports.contains("pub remote_boxes: [super::parametric_lib::RemoteBox__N_8; 2]"));
        assert!(imports.contains("imported_direct: &super::parametric_lib::RemoteBox__N_8"));
        assert!(imports.contains("imported_pair: &super::parametric_lib::RemotePair__A_8__B_23"));
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

    // Verifies: package-generated runners share one nominal Rust type universe.
    // Catches: per-wrapper duplicate bridge types that require adapter rebuilds.
    #[test]
    fn package_builder_shares_nominal_types_across_runners() -> Result<(), XlsynthError> {
        use super::shared_widget_package_aot::{
            shared_widget_bump::aot_bump_widget, shared_widget_echo::aot_echo_widget,
            shared_widget_types::Widget,
        };

        let input = Widget { widget_id: ub(7) };
        let mut echo_runner = aot_echo_widget::new_runner()?;
        let echoed = echo_runner.run(&input)?;
        let mut bump_runner = aot_bump_widget::new_runner()?;
        let bumped = bump_runner.run(&echoed)?;

        assert_eq!(bumped.widget_id.to_u64()?, 8);
        Ok(())
    }
}
