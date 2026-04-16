// SPDX-License-Identifier: Apache-2.0

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

pub mod pretty_route_aot {
    include!(env!("XLSYNTH_AOT_PRETTY_ROUTE_RS"));
}

pub mod self_alias_route_aot {
    include!(env!("XLSYNTH_AOT_SELF_ALIAS_ROUTE_RS"));
}

pub mod duplicate_packet_route_aot {
    include!(env!("XLSYNTH_AOT_DUPLICATE_PACKET_ROUTE_RS"));
}

#[cfg(test)]
mod tests {
    use xlsynth::{IrUBits, XlsynthError};

    use super::pretty_route_aot::pretty_types::{
        new_runner, Nibbles, Packet, RouteKind, RouteResult,
    };

    fn ub<const BIT_COUNT: usize>(value: u64) -> IrUBits<BIT_COUNT> {
        IrUBits::from_u64(value).unwrap()
    }

    #[test]
    fn pretty_route_aot_uses_generated_pretty_types() -> Result<(), XlsynthError> {
        let packet = Packet {
            tag: ub(7),
            kind: RouteKind::Remote,
            lanes: [ub(1), ub(2), ub(3)],
        };
        let salt: Nibbles = [ub(5), ub(6), ub(7)];

        let mut runner = new_runner()?;
        let result: RouteResult = runner.run(&packet, &salt)?;

        assert_eq!(result.next_tag.to_u64()?, 8);
        assert_eq!(result.selected_lane.to_u64()?, 9);
        assert_eq!(result.kind, RouteKind::Remote);
        Ok(())
    }

    #[test]
    fn pretty_route_generated_source_has_pretty_public_signature() {
        let source = std::fs::read_to_string(env!("XLSYNTH_AOT_PRETTY_ROUTE_RS")).unwrap();
        assert!(source.contains("pub fn run("));
        assert!(source.contains("packet: &Packet"));
        assert!(source.contains("salt: &Nibbles"));
        assert!(source.contains(") -> Result<RouteResult, xlsynth::XlsynthError>"));
        assert!(!source.contains("PrettyRoutePacket"));
        assert!(!source.contains("PrettyRouteSalt"));
        assert!(!source.contains("PrettyRouteReturn"));
        assert!(!source.contains("Input0"));
        assert!(!source.contains("Result<Output"));
        assert!(!source.contains("pub type Output"));
    }

    #[test]
    fn route_packet_signature_uses_canonical_bridge_type_without_self_alias(
    ) -> Result<(), XlsynthError> {
        use super::self_alias_route_aot::self_alias_route::{new_runner, RoutePacket};

        let packet = RoutePacket { tag: ub(9) };
        let mut runner = new_runner()?;
        let result: RoutePacket = runner.run(&packet)?;
        assert_eq!(result.tag.to_u64()?, 9);

        let source = std::fs::read_to_string(env!("XLSYNTH_AOT_SELF_ALIAS_ROUTE_RS")).unwrap();
        assert!(source.contains("packet: &RoutePacket"));
        assert!(source.contains(") -> Result<RoutePacket, xlsynth::XlsynthError>"));
        assert!(!source.contains("pub type RoutePacket"));
        Ok(())
    }

    #[test]
    fn duplicate_packet_modules_use_canonical_nested_paths() -> Result<(), XlsynthError> {
        use super::duplicate_packet_route_aot::{bar, foo, route};

        let packet = foo::packet::Packet { tag: ub(41) };
        let mut runner = route::new_runner()?;
        let result: bar::packet::Packet = runner.run(&packet)?;
        assert_eq!(result.tag.to_u64()?, 42);

        let source =
            std::fs::read_to_string(env!("XLSYNTH_AOT_DUPLICATE_PACKET_ROUTE_RS")).unwrap();
        assert!(source.contains("pub mod foo"));
        assert!(source.contains("pub mod bar"));
        assert!(source.contains("packet: &super::foo::packet::Packet"));
        assert!(source.contains(") -> Result<super::bar::packet::Packet, xlsynth::XlsynthError>"));
        Ok(())
    }
}
