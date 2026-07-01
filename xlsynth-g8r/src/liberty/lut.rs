// SPDX-License-Identifier: Apache-2.0

//! Raw multilinear lookup for Liberty timing-transition and internal-power
//! LUTs.

use crate::liberty::timing_table::TimingTableArrayView;
use crate::liberty_model::{Library, LuTableTemplate, PowerTable, TimingTable};
use anyhow::{Result, anyhow};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
/// Coordinates for a raw Liberty lookup.
pub struct RawLutQuery {
    pub input_transition: f64,
    pub output_load: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
/// Counts lookups that Liberty interpolation clamped to a table boundary.
pub struct RawLutQueryDiagnostics {
    pub below_min_clamp_count: usize,
    pub above_max_clamp_count: usize,
}

struct RawLutLayout<'a> {
    dimensions: &'a [u32],
    values: &'a [f32],
    axes: [&'a [f64]; 3],
    variables: [&'a str; 3],
}

fn effective_axis<'a>(explicit: &'a [f64], template: Option<&'a [f64]>) -> &'a [f64] {
    if explicit.is_empty() {
        template.unwrap_or(&[])
    } else {
        explicit
    }
}

fn template_for_shape<'a>(
    library: &'a Library,
    template_id: u32,
    expected_kind: &str,
    context: &str,
) -> Result<Option<&'a LuTableTemplate>> {
    if template_id == 0 {
        return Ok(None);
    }
    let template = library
        .lu_table_templates
        .get((template_id - 1) as usize)
        .ok_or_else(|| {
            anyhow!(
                "{context}: template_id {template_id} is out of range for {} templates",
                library.lu_table_templates.len()
            )
        })?;
    if template.kind_str(library) != expected_kind {
        return Err(anyhow!(
            "{context}: template_id {template_id} has kind '{}', expected '{}'",
            template.kind_str(library),
            expected_kind
        ));
    }
    Ok(Some(template))
}

fn timing_layout<'a>(
    library: &'a Library,
    table: &'a TimingTable,
    context: &str,
) -> Result<RawLutLayout<'a>> {
    let shape = library.timing_table_shape(table);
    let template = template_for_shape(library, shape.template_id, "lu_table_template", context)?;
    let explicit = library.timing_table_axes(table);
    Ok(RawLutLayout {
        dimensions: &shape.dimensions,
        values: library.timing_table_values(table),
        axes: [
            effective_axis(explicit[0], template.map(|value| value.index_1.as_slice())),
            effective_axis(explicit[1], template.map(|value| value.index_2.as_slice())),
            effective_axis(explicit[2], template.map(|value| value.index_3.as_slice())),
        ],
        variables: [
            template
                .map(|value| value.variable_1_str(library))
                .unwrap_or(""),
            template
                .map(|value| value.variable_2_str(library))
                .unwrap_or(""),
            template
                .map(|value| value.variable_3_str(library))
                .unwrap_or(""),
        ],
    })
}

fn power_layout<'a>(
    library: &'a Library,
    table: &'a PowerTable,
    context: &str,
) -> Result<RawLutLayout<'a>> {
    let shape = library.power_table_shape(table);
    let template = template_for_shape(library, shape.template_id, "power_lut_template", context)?;
    let explicit = library.power_table_axes(table);
    Ok(RawLutLayout {
        dimensions: &shape.dimensions,
        values: library.power_table_values(table),
        axes: [
            effective_axis(explicit[0], template.map(|value| value.index_1.as_slice())),
            effective_axis(explicit[1], template.map(|value| value.index_2.as_slice())),
            effective_axis(explicit[2], template.map(|value| value.index_3.as_slice())),
        ],
        variables: [
            template
                .map(|value| value.variable_1_str(library))
                .unwrap_or(""),
            template
                .map(|value| value.variable_2_str(library))
                .unwrap_or(""),
            template
                .map(|value| value.variable_3_str(library))
                .unwrap_or(""),
        ],
    })
}

fn query_for_axis(variable: &str, axis_index: usize, query: RawLutQuery) -> Result<f64> {
    match variable {
        "input_net_transition"
        | "input_transition_time"
        | "related_pin_transition"
        | "constrained_pin_transition" => Ok(query.input_transition),
        "total_output_net_capacitance"
        | "related_out_total_output_net_capacitance"
        | "output_net_pin_cap"
        | "related_out_output_net_pin_cap" => Ok(query.output_load),
        "" if axis_index == 0 => Ok(query.input_transition),
        "" if axis_index == 1 => Ok(query.output_load),
        _ => Err(anyhow!(
            "unsupported Liberty LUT variable '{}' on axis {}",
            variable,
            axis_index + 1
        )),
    }
}

fn validate_layout(layout: &RawLutLayout<'_>, context: &str) -> Result<()> {
    let array = TimingTableArrayView::from_parts(layout.dimensions, layout.values)
        .map_err(|error| anyhow!("{context}: invalid LUT payload: {error}"))?;
    if array.rank() > 3 {
        return Err(anyhow!(
            "{context}: rank-{} LUT is unsupported",
            array.rank()
        ));
    }
    for (axis_index, axis) in layout.axes.iter().take(array.rank()).enumerate() {
        if axis.len() != layout.dimensions[axis_index] as usize {
            return Err(anyhow!(
                "{context}: axis {} has {} knots but dimension is {}",
                axis_index + 1,
                axis.len(),
                layout.dimensions[axis_index]
            ));
        }
        if axis.iter().any(|value| !value.is_finite()) {
            return Err(anyhow!(
                "{context}: axis {} contains a non-finite knot",
                axis_index + 1
            ));
        }
        if axis.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(anyhow!(
                "{context}: axis {} is not strictly increasing",
                axis_index + 1
            ));
        }
    }
    if layout.values.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!("{context}: LUT contains a non-finite value"));
    }
    Ok(())
}

fn bracket_clamped(axis: &[f64], query: f64) -> (usize, usize, f64) {
    if axis.len() == 1 || query <= axis[0] {
        return (0, 0, 0.0);
    }
    let last = axis.len() - 1;
    if query >= axis[last] {
        return (last, last, 0.0);
    }
    let upper = axis.partition_point(|value| *value < query);
    let lower = upper - 1;
    let weight = (query - axis[lower]) / (axis[upper] - axis[lower]);
    (lower, upper, weight)
}

fn evaluate_layout(
    layout: RawLutLayout<'_>,
    query: RawLutQuery,
    diagnostics: &mut RawLutQueryDiagnostics,
    context: &str,
) -> Result<f64> {
    if !query.input_transition.is_finite() || query.input_transition < 0.0 {
        return Err(anyhow!(
            "{context}: input transition must be finite and non-negative; got {}",
            query.input_transition
        ));
    }
    if !query.output_load.is_finite() || query.output_load < 0.0 {
        return Err(anyhow!(
            "{context}: output load must be finite and non-negative; got {}",
            query.output_load
        ));
    }
    validate_layout(&layout, context)?;
    let array = TimingTableArrayView::from_parts(layout.dimensions, layout.values)
        .map_err(|error| anyhow!("{context}: invalid LUT payload: {error}"))?;
    if array.rank() == 0 {
        return array
            .get(&[])
            .ok_or_else(|| anyhow!("{context}: scalar LUT has no value"));
    }
    let mut bounds = Vec::with_capacity(array.rank());
    for axis_index in 0..array.rank() {
        let axis = layout.axes[axis_index];
        let raw_query = query_for_axis(layout.variables[axis_index], axis_index, query)
            .map_err(|error| anyhow!("{context}: {error}"))?;
        if raw_query < axis[0] {
            diagnostics.below_min_clamp_count += 1;
        } else if raw_query > axis[axis.len() - 1] {
            diagnostics.above_max_clamp_count += 1;
        }
        bounds.push(bracket_clamped(axis, raw_query));
    }

    let mut indices = vec![0usize; array.rank()];
    for (axis_index, (lower, _, _)) in bounds.iter().copied().enumerate() {
        indices[axis_index] = lower;
    }
    let varying: Vec<_> = bounds
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, (lower, upper, _))| lower != upper)
        .collect();
    let mut result = 0.0;
    for corner in 0..(1usize << varying.len()) {
        let mut weight = 1.0;
        for (varying_index, (axis_index, (lower, upper, interpolation))) in
            varying.iter().enumerate()
        {
            if ((corner >> varying_index) & 1) == 1 {
                indices[*axis_index] = *upper;
                weight *= *interpolation;
            } else {
                indices[*axis_index] = *lower;
                weight *= 1.0 - *interpolation;
            }
        }
        let value = array.get(&indices).ok_or_else(|| {
            anyhow!(
                "{context}: could not read LUT value at coordinate {:?}",
                indices
            )
        })?;
        result += weight * value;
    }
    if !result.is_finite() {
        return Err(anyhow!("{context}: LUT interpolation produced {result}"));
    }
    Ok(result)
}

/// Evaluates a timing LUT without STA's monotonic repair or extrapolation.
pub fn evaluate_timing_lut_raw(
    library: &Library,
    table: &TimingTable,
    query: RawLutQuery,
    diagnostics: &mut RawLutQueryDiagnostics,
    context: &str,
) -> Result<f64> {
    evaluate_layout(
        timing_layout(library, table, context)?,
        query,
        diagnostics,
        context,
    )
}

/// Evaluates an internal-power LUT with raw multilinear interpolation.
pub fn evaluate_power_lut(
    library: &Library,
    table: &PowerTable,
    query: RawLutQuery,
    diagnostics: &mut RawLutQueryDiagnostics,
    context: &str,
) -> Result<f64> {
    evaluate_layout(
        power_layout(library, table, context)?,
        query,
        diagnostics,
        context,
    )
}

fn input_transition_range(layout: RawLutLayout<'_>) -> Option<(f64, f64)> {
    layout
        .variables
        .iter()
        .enumerate()
        .find(|(axis_index, variable)| {
            matches!(
                **variable,
                "input_net_transition"
                    | "input_transition_time"
                    | "related_pin_transition"
                    | "constrained_pin_transition"
            ) || (variable.is_empty() && *axis_index == 0)
        })
        .and_then(|(axis_index, _)| {
            let axis = layout.axes[axis_index];
            Some((*axis.first()?, *axis.last()?))
        })
}

/// Returns the minimum and maximum input-transition knots in a timing LUT.
pub fn timing_lut_input_transition_range(
    library: &Library,
    table: &TimingTable,
    context: &str,
) -> Result<Option<(f64, f64)>> {
    Ok(input_transition_range(timing_layout(
        library, table, context,
    )?))
}

/// Returns the minimum and maximum input-transition knots in a power LUT.
pub fn power_lut_input_transition_range(
    library: &Library,
    table: &PowerTable,
    context: &str,
) -> Result<Option<(f64, f64)>> {
    Ok(input_transition_range(power_layout(
        library, table, context,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_model::{LibraryBuilder, LuTableTemplate};
    use crate::liberty_proto::PowerTransition;

    #[test]
    fn raw_power_lookup_is_bilinear_and_clamped() {
        let mut builder = LibraryBuilder::new();
        builder.lu_table_templates.push(LuTableTemplate {
            kind: "power_lut_template".into(),
            name: "power".to_string(),
            variable_1: "input_transition_time".into(),
            variable_2: "total_output_net_capacitance".into(),
            index_1: vec![1.0, 3.0],
            index_2: vec![2.0, 6.0],
            ..Default::default()
        });
        let table = builder
            .add_power_table_f64(
                PowerTransition::Rise,
                1,
                vec![],
                vec![],
                vec![],
                vec![10.0, 20.0, 30.0, 50.0],
                vec![2, 2],
                "",
            )
            .unwrap();
        let library = builder.finish();
        let mut diagnostics = RawLutQueryDiagnostics::default();
        let value = evaluate_power_lut(
            &library,
            &table,
            RawLutQuery {
                input_transition: 2.0,
                output_load: 4.0,
            },
            &mut diagnostics,
            "power",
        )
        .unwrap();
        assert_eq!(value, 27.5);

        let clamped = evaluate_power_lut(
            &library,
            &table,
            RawLutQuery {
                input_transition: 0.0,
                output_load: 10.0,
            },
            &mut diagnostics,
            "power",
        )
        .unwrap();
        assert_eq!(clamped, 20.0);
        assert_eq!(diagnostics.below_min_clamp_count, 1);
        assert_eq!(diagnostics.above_max_clamp_count, 1);
    }
}
