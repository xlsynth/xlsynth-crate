// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::combo_compile::CompiledComboModule;
use crate::combo_eval::eval_combo;
use crate::combo_eval::ComboEvalPlan;
use crate::vcd_writer::VcdWriter;
use crate::Error;
use crate::Result;
use crate::Value4;

pub fn run_combo_and_write_vcd(
    m: &CompiledComboModule,
    plan: &ComboEvalPlan,
    vectors: &[BTreeMap<String, Value4>],
    out_path: &std::path::Path,
) -> Result<()> {
    if vectors.is_empty() {
        return Err(Error::Parse("no input vectors provided".to_string()));
    }

    let mut writer = VcdWriter::new("1ns");
    let mut port_names: BTreeSet<String> = BTreeSet::new();
    for p in &m.input_ports {
        port_names.insert(p.name.clone());
    }
    for p in &m.output_ports {
        port_names.insert(p.name.clone());
    }

    for (name, info) in &m.decls {
        if port_names.contains(name) {
            writer.add_var(&format!("tb.{name}"), info.width)?;
            writer.add_var(&format!("tb.dut.{name}"), info.width)?;
        } else {
            writer.add_var(&format!("tb.dut.{name}"), info.width)?;
        }
    }

    let mut f =
        std::fs::File::create(out_path).map_err(|e| Error::Parse(format!("io error: {e}")))?;
    writer.write_header(&mut f)?;

    for (idx, vec_inputs) in vectors.iter().enumerate() {
        let time = idx as u64;
        let values = eval_combo(m, plan, vec_inputs)?;

        // Ensure tb.<port> exists and is driven.
        for pname in &port_names {
            let v = values
                .get(pname)
                .ok_or_else(|| Error::Parse(format!("no value computed for port `{pname}`")))?;
            writer.change(
                &mut f,
                time,
                &format!("tb.{pname}"),
                &v.to_bit_string_msb_first(),
            )?;
            writer.change(
                &mut f,
                time,
                &format!("tb.dut.{pname}"),
                &v.to_bit_string_msb_first(),
            )?;
        }

        // Drive all dut nets (including ports again; redundant but fine).
        for (name, v) in &values {
            writer.change(
                &mut f,
                time,
                &format!("tb.dut.{name}"),
                &v.to_bit_string_msb_first(),
            )?;
        }
    }
    Ok(())
}
