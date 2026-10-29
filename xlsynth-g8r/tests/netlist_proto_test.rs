// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use prost::Message;
use xlsynth_g8r::liberty::Library;
use xlsynth_g8r::netlist::bench_synth_netlist::make_chain_netlist;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::{ParsedNetlist, read_gv_from_str};
use xlsynth_g8r::netlist::proto::{decode_netlist, encode_netlist, pb};
use xlsynth_g8r::techmap::area::{AreaMappingLibrary, map_gatefn_to_area_netlist};
use xlsynth_g8r::test_utils::{interesting_ir_roundtrip_cases, load_interesting_ir_roundtrip_case};

#[test]
fn round_trips_mapped_signature_corpus() -> Result<()> {
    // The identity and constant outputs in this corpus need only wiring.
    let library = AreaMappingLibrary::new(&Library::default())?;
    for case in interesting_ir_roundtrip_cases() {
        let loaded = load_interesting_ir_roundtrip_case(case);
        let mapped = map_gatefn_to_area_netlist(&loaded.gate_fn, &library)?.netlist;
        let expected = emit_module_as_netlist_text(&mapped.module, &mapped.nets, &mapped.interner)?;
        let design = ParsedNetlist {
            modules: vec![mapped.module],
            nets: mapped.nets,
            interner: mapped.interner,
        };
        let restored = decode_netlist(&encode_netlist(&design)?)?;
        assert_eq!(
            emit_module_as_netlist_text(&restored.modules[0], &restored.nets, &restored.interner)?,
            expected,
            "mapped signature {}",
            case.name
        );
    }
    Ok(())
}

#[test]
fn repeated_cells_share_their_interface_and_pack_net_ids() -> Result<()> {
    let source = make_chain_netlist(1_000);
    let bytes = encode_netlist(&read_gv_from_str(&source)?)?;
    let mut message = pb::Netlist::decode(bytes.as_slice())?;
    assert_eq!(message.interfaces.len(), 1);
    assert!(
        bytes.len() < source.len(),
        "{} protobuf bytes for {} GV bytes",
        bytes.len(),
        source.len()
    );

    let instance = &mut message.modules[0].instances[0];
    assert_eq!(instance.nets.len(), 2);
    assert!(instance.connections.is_empty());
    instance.connections.push(pb::Ref {
        kind: Some(pb::r#ref::Kind::Net(0)),
    });
    assert!(decode_netlist(&message.encode_to_vec()).is_err());
    Ok(())
}
