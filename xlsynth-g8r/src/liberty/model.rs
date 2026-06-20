// SPDX-License-Identifier: Apache-2.0

use crate::liberty_proto as wire;
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

/// One-based reference into `Library`'s interned-string pool; zero is empty.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StringId {
    id: u32,
}

impl StringId {
    /// Returns whether this ID is the zero sentinel rather than a pool entry.
    pub fn is_unset(self) -> bool {
        self.id == 0
    }

    fn from_pool_id(id: u32) -> Self {
        Self { id }
    }
}

/// One-based reference into `Library`'s LUT-axis pool; zero is an empty axis.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AxisId(u32);

impl AxisId {
    /// Returns whether this ID is the zero sentinel rather than a pool entry.
    pub fn is_unset(self) -> bool {
        self.0 == 0
    }
}

/// A range in a library-owned float32 LUT-value allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LutValueRange {
    start: u32,
    len: u32,
}

impl LutValueRange {
    /// Returns the number of values in this range.
    pub fn len(self) -> usize {
        self.len as usize
    }

    /// Returns whether this range contains no values.
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Shared geometry and template metadata for timing and power tables.
#[derive(Clone, Debug, PartialEq)]
pub struct LutShape {
    pub template_id: u32,
    pub index_1: AxisId,
    pub index_2: AxisId,
    pub index_3: AxisId,
    pub dimensions: Box<[u32]>,
    pub template_name: StringId,
}

impl Default for LutShape {
    fn default() -> Self {
        Self {
            template_id: 0,
            index_1: AxisId::default(),
            index_2: AxisId::default(),
            index_3: AxisId::default(),
            dimensions: Box::default(),
            template_name: StringId::default(),
        }
    }
}

/// Fully populated in-memory Liberty library used by parsers and evaluators.
#[derive(Debug, Default, PartialEq)]
pub struct Library {
    pub cells: Vec<Cell>,
    pub units: Option<wire::LibraryUnits>,
    pub lu_table_templates: Vec<LuTableTemplate>,
    pub threshold_voltage_groups: Vec<String>,
    pub default_threshold_voltage_group_id: u32,
    pub threshold_voltage_group_class_indices: Vec<i32>,
    pub nominal_voltage: Option<f64>,
    pub provenance: String,
    pub source_files: Vec<String>,
    pub(crate) strings: Vec<Box<str>>,
    pub(crate) lut_axes: Vec<Box<[f64]>>,
    pub(crate) lut_shapes: Vec<LutShape>,
    pub(crate) lut_values: Vec<f32>,
}

/// Mutable construction state for a pooled Liberty runtime model.
#[derive(Debug, Default)]
pub struct LibraryBuilder {
    library: Library,
    string_id_by_value: HashMap<String, StringId>,
    axis_id_by_value: HashMap<Vec<u64>, AxisId>,
}

impl Library {
    /// Resolves an interned string ID.
    pub fn resolve_string(&self, id: &StringId) -> &str {
        if id.id == 0 {
            return "";
        }
        self.strings
            .get((id.id - 1) as usize)
            .expect("validated interned string ID")
    }

    /// Resolves an interned LUT-axis ID.
    pub fn resolve_axis(&self, id: AxisId) -> &[f64] {
        if id.0 == 0 {
            return &[];
        }
        self.lut_axes
            .get((id.0 - 1) as usize)
            .expect("validated LUT axis ID")
    }

    fn lut_shape(&self, shape_id: u32) -> &LutShape {
        self.lut_shapes
            .get((shape_id - 1) as usize)
            .expect("validated LUT shape ID")
    }

    /// Resolves the library-owned shape referenced by a timing table.
    pub fn timing_table_shape<'a>(&'a self, table: &'a TimingTable) -> &'a LutShape {
        self.lut_shape(table.shape_id)
    }

    /// Resolves all explicit axes referenced by a timing table.
    pub fn timing_table_axes<'a>(&'a self, table: &'a TimingTable) -> [&'a [f64]; 3] {
        let shape = self.lut_shape(table.shape_id);
        [
            self.resolve_axis(shape.index_1),
            self.resolve_axis(shape.index_2),
            self.resolve_axis(shape.index_3),
        ]
    }

    /// Resolves the legacy template name referenced by a timing table.
    pub fn timing_table_template_name<'a>(&'a self, table: &'a TimingTable) -> &'a str {
        self.resolve_string(&self.lut_shape(table.shape_id).template_name)
    }

    /// Resolves the library-owned shape referenced by a power table.
    pub fn power_table_shape<'a>(&'a self, table: &'a PowerTable) -> &'a LutShape {
        self.lut_shape(table.shape_id)
    }

    /// Resolves all explicit axes referenced by a power table.
    pub fn power_table_axes<'a>(&'a self, table: &'a PowerTable) -> [&'a [f64]; 3] {
        let shape = self.lut_shape(table.shape_id);
        [
            self.resolve_axis(shape.index_1),
            self.resolve_axis(shape.index_2),
            self.resolve_axis(shape.index_3),
        ]
    }

    /// Resolves the legacy template name referenced by a power table.
    pub fn power_table_template_name<'a>(&'a self, table: &'a PowerTable) -> &'a str {
        self.resolve_string(&self.lut_shape(table.shape_id).template_name)
    }

    fn lut_values(&self, range: LutValueRange) -> &[f32] {
        let start = range.start as usize;
        let end = start + range.len as usize;
        self.lut_values
            .get(start..end)
            .expect("validated LUT value range")
    }

    /// Resolves the library-owned values referenced by a timing table.
    pub fn timing_table_values<'a>(&'a self, table: &'a TimingTable) -> &'a [f32] {
        self.lut_values(table.values)
    }

    /// Resolves the library-owned values referenced by a power table.
    pub fn power_table_values<'a>(&'a self, table: &'a PowerTable) -> &'a [f32] {
        self.lut_values(table.values)
    }
}

impl LibraryBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resumes construction from an existing pooled model.
    pub fn from_library(library: Library) -> Self {
        let string_id_by_value = library
            .strings
            .iter()
            .enumerate()
            .map(|(index, value)| (value.to_string(), StringId::from_pool_id(index as u32 + 1)))
            .collect();
        let axis_id_by_value = library
            .lut_axes
            .iter()
            .enumerate()
            .map(|(index, axis)| (axis_key(axis), AxisId(index as u32 + 1)))
            .collect();
        Self {
            library,
            string_id_by_value,
            axis_id_by_value,
        }
    }

    /// Returns mutable non-pool model data under construction.
    pub fn library_mut(&mut self) -> &mut Library {
        &mut self.library
    }

    /// Finishes construction and discards interning state.
    pub fn finish(self) -> Library {
        self.library
    }

    /// Interns a string for use by a model record.
    pub fn intern_string(&mut self, value: impl AsRef<str>) -> Result<StringId> {
        let value = value.as_ref();
        if value.is_empty() {
            return Ok(StringId::default());
        }
        if let Some(id) = self.string_id_by_value.get(value) {
            return Ok(*id);
        }
        let id = StringId::from_pool_id(
            u32::try_from(self.library.strings.len() + 1)
                .map_err(|_| anyhow::anyhow!("too many interned strings"))?,
        );
        self.library.strings.push(value.into());
        self.string_id_by_value.insert(value.to_string(), id);
        Ok(id)
    }

    fn intern_axis(&mut self, values: Vec<f64>) -> Result<AxisId> {
        if values.is_empty() {
            return Ok(AxisId::default());
        }
        let key = axis_key(&values);
        if let Some(id) = self.axis_id_by_value.get(&key) {
            return Ok(*id);
        }
        let id = AxisId(
            u32::try_from(self.library.lut_axes.len() + 1)
                .map_err(|_| anyhow::anyhow!("too many LUT axes"))?,
        );
        self.library.lut_axes.push(values.into_boxed_slice());
        self.axis_id_by_value.insert(key, id);
        Ok(id)
    }

    fn append_lut_values(&mut self, mut values: Vec<f32>) -> Result<LutValueRange> {
        let start = u32::try_from(self.library.lut_values.len())
            .map_err(|_| anyhow::anyhow!("LUT value storage exceeds u32"))?;
        let len = u32::try_from(values.len())
            .map_err(|_| anyhow::anyhow!("LUT table value count exceeds u32"))?;
        self.library.lut_values.append(&mut values);
        Ok(LutValueRange { start, len })
    }

    fn append_lut_shape(&mut self, shape: LutShape) -> Result<u32> {
        self.library.lut_shapes.push(shape);
        u32::try_from(self.library.lut_shapes.len())
            .map_err(|_| anyhow::anyhow!("too many LUT shapes"))
    }

    /// Builds a timing arc whose selector strings are interned in this library.
    pub fn add_timing_arc(
        &mut self,
        related_pin: impl AsRef<str>,
        timing_sense: impl AsRef<str>,
        timing_type: impl AsRef<str>,
        when: impl AsRef<str>,
        tables: Vec<TimingTable>,
    ) -> Result<TimingArc> {
        let related_pin = self.intern_string(related_pin)?;
        let timing_sense = TimingSense::from_raw_in(self, timing_sense.as_ref())?;
        let timing_type = TimingType::from_raw_in(self, timing_type.as_ref())?;
        let when = self.intern_string(when)?;
        Ok(TimingArc {
            related_pin,
            timing_sense,
            timing_type,
            when,
            tables,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Adds a timing table and returns its pooled table record.
    pub fn add_timing_table_f64(
        &mut self,
        kind: wire::TimingTableKind,
        template_id: u32,
        index_1: Vec<f64>,
        index_2: Vec<f64>,
        index_3: Vec<f64>,
        values: Vec<f64>,
        dimensions: Vec<u32>,
        template_name: impl Into<String>,
    ) -> Result<TimingTable> {
        let axes = [
            self.intern_axis(index_1)?,
            self.intern_axis(index_2)?,
            self.intern_axis(index_3)?,
        ];
        let template_name: String = template_name.into();
        let template_name = self.intern_string(template_name)?;
        let shape_id = self.append_lut_shape(LutShape {
            template_id,
            index_1: axes[0],
            index_2: axes[1],
            index_3: axes[2],
            dimensions: dimensions.into_boxed_slice(),
            template_name,
        })?;
        let values = self.append_lut_values(values_to_f32(values)?)?;
        Ok(TimingTable {
            kind,
            shape_id,
            values,
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Adds a power table and returns its pooled table record.
    pub fn add_power_table_f64(
        &mut self,
        transition: wire::PowerTransition,
        template_id: u32,
        index_1: Vec<f64>,
        index_2: Vec<f64>,
        index_3: Vec<f64>,
        values: Vec<f64>,
        dimensions: Vec<u32>,
        template_name: impl Into<String>,
    ) -> Result<PowerTable> {
        let axes = [
            self.intern_axis(index_1)?,
            self.intern_axis(index_2)?,
            self.intern_axis(index_3)?,
        ];
        let template_name: String = template_name.into();
        let template_name = self.intern_string(template_name)?;
        let shape_id = self.append_lut_shape(LutShape {
            template_id,
            index_1: axes[0],
            index_2: axes[1],
            index_3: axes[2],
            dimensions: dimensions.into_boxed_slice(),
            template_name,
        })?;
        let values = self.append_lut_values(values_to_f32(values)?)?;
        Ok(PowerTable {
            transition,
            shape_id,
            values,
        })
    }
}

impl Deref for LibraryBuilder {
    type Target = Library;

    fn deref(&self) -> &Self::Target {
        &self.library
    }
}

impl DerefMut for LibraryBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.library
    }
}

impl From<LibraryBuilder> for Library {
    fn from(builder: LibraryBuilder) -> Self {
        builder.finish()
    }
}

impl Library {
    pub(crate) fn append_cells_and_luts(&mut self, mut other: Library) -> Result<()> {
        fn offset_string(id: &mut StringId, offset: u32) -> Result<()> {
            if id.id != 0 {
                id.id = id
                    .id
                    .checked_add(offset)
                    .ok_or_else(|| anyhow::anyhow!("interned string ID overflow"))?;
            }
            Ok(())
        }

        fn offset_axis(id: &mut AxisId, offset: u32) -> Result<()> {
            if id.0 != 0 {
                id.0 =
                    id.0.checked_add(offset)
                        .ok_or_else(|| anyhow::anyhow!("LUT axis ID overflow"))?;
            }
            Ok(())
        }

        let string_offset = u32::try_from(self.strings.len())
            .map_err(|_| anyhow::anyhow!("too many interned strings"))?;
        let axis_offset =
            u32::try_from(self.lut_axes.len()).map_err(|_| anyhow::anyhow!("too many LUT axes"))?;
        let shape_offset = u32::try_from(self.lut_shapes.len())
            .map_err(|_| anyhow::anyhow!("too many LUT shapes"))?;
        let value_offset = u32::try_from(self.lut_values.len())
            .map_err(|_| anyhow::anyhow!("LUT value storage exceeds u32"))?;
        for cell in &mut other.cells {
            for pin in &mut cell.pins {
                offset_string(&mut pin.function, string_offset)?;
                offset_string(&mut pin.name, string_offset)?;
                for arc in &mut pin.timing_arcs {
                    offset_string(&mut arc.related_pin, string_offset)?;
                    offset_string(&mut arc.timing_sense.other, string_offset)?;
                    offset_string(&mut arc.timing_type.other, string_offset)?;
                    offset_string(&mut arc.when, string_offset)?;
                    for table in &mut arc.tables {
                        table.shape_id = table
                            .shape_id
                            .checked_add(shape_offset)
                            .ok_or_else(|| anyhow::anyhow!("LUT shape ID overflow"))?;
                        table.values.start = table
                            .values
                            .start
                            .checked_add(value_offset)
                            .ok_or_else(|| anyhow::anyhow!("LUT value offset overflow"))?;
                    }
                }
                for group in &mut pin.internal_power {
                    for related_pin in &mut group.related_pins {
                        offset_string(related_pin, string_offset)?;
                    }
                    offset_string(&mut group.when, string_offset)?;
                    offset_string(&mut group.related_pg_pin, string_offset)?;
                    for table in &mut group.tables {
                        table.shape_id = table
                            .shape_id
                            .checked_add(shape_offset)
                            .ok_or_else(|| anyhow::anyhow!("LUT shape ID overflow"))?;
                        table.values.start = table
                            .values
                            .start
                            .checked_add(value_offset)
                            .ok_or_else(|| anyhow::anyhow!("LUT value offset overflow"))?;
                    }
                }
            }
        }
        for template in &mut other.lu_table_templates {
            offset_string(&mut template.kind.other, string_offset)?;
            offset_string(&mut template.variable_1.other, string_offset)?;
            offset_string(&mut template.variable_2.other, string_offset)?;
            offset_string(&mut template.variable_3.other, string_offset)?;
        }
        for shape in &mut other.lut_shapes {
            offset_axis(&mut shape.index_1, axis_offset)?;
            offset_axis(&mut shape.index_2, axis_offset)?;
            offset_axis(&mut shape.index_3, axis_offset)?;
            offset_string(&mut shape.template_name, string_offset)?;
        }
        self.cells.append(&mut other.cells);
        self.lu_table_templates
            .append(&mut other.lu_table_templates);
        self.strings.append(&mut other.strings);
        self.lut_axes.append(&mut other.lut_axes);
        self.lut_shapes.append(&mut other.lut_shapes);
        self.lut_values.append(&mut other.lut_values);
        Ok(())
    }
}

/// Normalized cell data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cell {
    pub name: String,
    pub pins: Vec<Pin>,
    pub area: f64,
    pub sequential: Vec<wire::Sequential>,
    pub clock_gate: Option<wire::ClockGate>,
    pub threshold_voltage_group_id: u32,
    pub dont_use: Option<bool>,
}

/// Normalized pin data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pin {
    pub direction: i32,
    pub function: StringId,
    pub name: StringId,
    pub is_clocking_pin: bool,
    pub capacitance: Option<f64>,
    pub rise_capacitance: Option<f64>,
    pub fall_capacitance: Option<f64>,
    pub max_capacitance: Option<f64>,
    pub timing_arcs: Vec<TimingArc>,
    pub internal_power: Vec<InternalPower>,
}

macro_rules! liberty_enum_value {
    ($name:ident, $wire:ty, $default:path, $parse:path, $format:path) => {
        #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
        pub struct $name {
            value: $wire,
            other: StringId,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    value: $default,
                    other: StringId::default(),
                }
            }
        }

        impl $name {
            fn from_wire(value: $wire, other: StringId) -> Self {
                Self { value, other }
            }

            pub(crate) fn from_raw_in(builder: &mut LibraryBuilder, raw: &str) -> Result<Self> {
                let value = $parse(raw);
                let other = if value == <$wire>::Other {
                    builder.intern_string(raw)?
                } else {
                    StringId::default()
                };
                Ok(Self { value, other })
            }

            pub fn as_str<'a>(&'a self, library: &'a Library) -> &'a str {
                $format(self.value, library.resolve_string(&self.other))
            }

            pub fn wire_value(&self) -> $wire {
                self.value
            }

            fn fallback<'a>(&'a self, library: &'a Library) -> &'a str {
                library.resolve_string(&self.other)
            }
        }

        #[cfg(test)]
        impl From<String> for $name {
            fn from(raw: String) -> Self {
                let value = $parse(&raw);
                assert_ne!(value, <$wire>::Other, "unknown test enum value '{raw}'");
                Self {
                    value,
                    other: StringId::default(),
                }
            }
        }

        #[cfg(test)]
        impl From<&str> for $name {
            fn from(raw: &str) -> Self {
                raw.to_string().into()
            }
        }
    };
}

liberty_enum_value!(
    LutTemplateKind,
    wire::LutTemplateKind,
    wire::LutTemplateKind::Unknown,
    lut_template_kind_to_wire,
    lut_template_kind_str
);
liberty_enum_value!(
    LutVariable,
    wire::LutVariable,
    wire::LutVariable::Unspecified,
    lut_variable_to_wire,
    lut_variable_str
);
liberty_enum_value!(
    TimingSense,
    wire::TimingSense,
    wire::TimingSense::Unspecified,
    timing_sense_to_wire,
    timing_sense_str
);
liberty_enum_value!(
    TimingType,
    wire::TimingType,
    wire::TimingType::Unspecified,
    timing_type_to_wire,
    timing_type_str
);

/// LUT template metadata; table payloads share canonicalized shapes separately.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LuTableTemplate {
    pub kind: LutTemplateKind,
    pub name: String,
    pub variable_1: LutVariable,
    pub variable_2: LutVariable,
    pub variable_3: LutVariable,
    pub index_1: Vec<f64>,
    pub index_2: Vec<f64>,
    pub index_3: Vec<f64>,
}

impl LuTableTemplate {
    pub fn kind_str<'a>(&'a self, library: &'a Library) -> &'a str {
        self.kind.as_str(library)
    }

    pub fn variable_1_str<'a>(&'a self, library: &'a Library) -> &'a str {
        self.variable_1.as_str(library)
    }

    pub fn variable_2_str<'a>(&'a self, library: &'a Library) -> &'a str {
        self.variable_2.as_str(library)
    }

    pub fn variable_3_str<'a>(&'a self, library: &'a Library) -> &'a str {
        self.variable_3.as_str(library)
    }
}

/// Normalized timing arc.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimingArc {
    pub related_pin: StringId,
    pub timing_sense: TimingSense,
    pub timing_type: TimingType,
    pub when: StringId,
    pub tables: Vec<TimingTable>,
}

impl TimingArc {
    pub fn timing_sense_str<'a>(&'a self, library: &'a Library) -> &'a str {
        self.timing_sense.as_str(library)
    }

    pub fn timing_type_str<'a>(&'a self, library: &'a Library) -> &'a str {
        self.timing_type.as_str(library)
    }
}

/// Timing table referencing shared shape and float32 value storage.
#[derive(Clone, Debug, PartialEq)]
pub struct TimingTable {
    pub kind: wire::TimingTableKind,
    shape_id: u32,
    values: LutValueRange,
}

impl Default for TimingTable {
    fn default() -> Self {
        Self {
            kind: wire::TimingTableKind::Unknown,
            shape_id: 0,
            values: LutValueRange::default(),
        }
    }
}

impl TimingTable {
    /// Returns the one-based ID of this table's library-owned shape.
    pub fn shape_id(&self) -> u32 {
        self.shape_id
    }

    /// Returns this table's range in the library-owned value allocation.
    pub fn value_range(&self) -> LutValueRange {
        self.values
    }

    pub fn kind_str(&self) -> &'static str {
        timing_table_kind_str(self.kind)
    }
}

/// Normalized internal-power selector and table data.
#[derive(Clone, Debug, PartialEq)]
pub struct InternalPower {
    pub related_pins: Vec<StringId>,
    pub when: StringId,
    pub related_pg_pin: StringId,
    pub tables: Vec<PowerTable>,
}

impl Default for InternalPower {
    fn default() -> Self {
        Self {
            related_pins: Vec::new(),
            when: StringId::default(),
            related_pg_pin: StringId::default(),
            tables: Vec::new(),
        }
    }
}

/// Power table referencing shared shape and float32 value storage.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerTable {
    pub transition: wire::PowerTransition,
    shape_id: u32,
    values: LutValueRange,
}

impl Default for PowerTable {
    fn default() -> Self {
        Self {
            transition: wire::PowerTransition::Unknown,
            shape_id: 0,
            values: LutValueRange::default(),
        }
    }
}

impl PowerTable {
    /// Returns the one-based ID of this table's library-owned shape.
    pub fn shape_id(&self) -> u32 {
        self.shape_id
    }
}

fn enum_string<'a, E: Copy + PartialEq>(
    value: E,
    other: &'a str,
    known: &[(E, &'static str)],
) -> &'a str {
    known
        .iter()
        .find_map(|(candidate, text)| (*candidate == value).then_some(*text))
        .unwrap_or(other)
}

fn lut_template_kind_str(value: wire::LutTemplateKind, other: &str) -> &str {
    enum_string(
        value,
        other,
        &[
            (wire::LutTemplateKind::Unknown, ""),
            (wire::LutTemplateKind::Timing, "lu_table_template"),
            (wire::LutTemplateKind::Power, "power_lut_template"),
        ],
    )
}

fn lut_variable_str(value: wire::LutVariable, other: &str) -> &str {
    enum_string(
        value,
        other,
        &[
            (wire::LutVariable::Unspecified, ""),
            (
                wire::LutVariable::InputNetTransition,
                "input_net_transition",
            ),
            (
                wire::LutVariable::InputTransitionTime,
                "input_transition_time",
            ),
            (
                wire::LutVariable::TotalOutputNetCapacitance,
                "total_output_net_capacitance",
            ),
            (
                wire::LutVariable::RelatedPinTransition,
                "related_pin_transition",
            ),
            (
                wire::LutVariable::ConstrainedPinTransition,
                "constrained_pin_transition",
            ),
            (wire::LutVariable::OutputNetLength, "output_net_length"),
            (wire::LutVariable::OutputNetWireCap, "output_net_wire_cap"),
            (wire::LutVariable::OutputNetPinCap, "output_net_pin_cap"),
            (
                wire::LutVariable::RelatedOutTotalOutputNetCapacitance,
                "related_out_total_output_net_capacitance",
            ),
            (
                wire::LutVariable::RelatedOutOutputNetLength,
                "related_out_output_net_length",
            ),
            (
                wire::LutVariable::RelatedOutOutputNetWireCap,
                "related_out_output_net_wire_cap",
            ),
            (
                wire::LutVariable::RelatedOutOutputNetPinCap,
                "related_out_output_net_pin_cap",
            ),
            (wire::LutVariable::ConnectDelay, "connect_delay"),
            (wire::LutVariable::Time, "time"),
            (wire::LutVariable::InputVoltage, "input_voltage"),
            (wire::LutVariable::OutputVoltage, "output_voltage"),
            (wire::LutVariable::PathDepth, "path_depth"),
            (wire::LutVariable::PathDistance, "path_distance"),
            (wire::LutVariable::NormalizedVoltage, "normalized_voltage"),
            (wire::LutVariable::Temperature, "temperature"),
        ],
    )
}

fn timing_type_str(value: wire::TimingType, other: &str) -> &str {
    match value {
        wire::TimingType::Unspecified => "",
        wire::TimingType::Combinational => "combinational",
        wire::TimingType::CombinationalRise => "combinational_rise",
        wire::TimingType::CombinationalFall => "combinational_fall",
        wire::TimingType::RisingEdge => "rising_edge",
        wire::TimingType::FallingEdge => "falling_edge",
        wire::TimingType::Preset => "preset",
        wire::TimingType::Clear => "clear",
        wire::TimingType::HoldRising => "hold_rising",
        wire::TimingType::HoldFalling => "hold_falling",
        wire::TimingType::SetupRising => "setup_rising",
        wire::TimingType::SetupFalling => "setup_falling",
        wire::TimingType::RecoveryRising => "recovery_rising",
        wire::TimingType::RecoveryFalling => "recovery_falling",
        wire::TimingType::RemovalRising => "removal_rising",
        wire::TimingType::RemovalFalling => "removal_falling",
        wire::TimingType::SkewRising => "skew_rising",
        wire::TimingType::SkewFalling => "skew_falling",
        wire::TimingType::MinPulseWidth => "min_pulse_width",
        wire::TimingType::MinimumPeriod => "minimum_period",
        wire::TimingType::ThreeStateEnable => "three_state_enable",
        wire::TimingType::ThreeStateDisable => "three_state_disable",
        wire::TimingType::ThreeStateEnableRise => "three_state_enable_rise",
        wire::TimingType::ThreeStateEnableFall => "three_state_enable_fall",
        wire::TimingType::ThreeStateDisableRise => "three_state_disable_rise",
        wire::TimingType::ThreeStateDisableFall => "three_state_disable_fall",
        wire::TimingType::NonSeqSetupRising => "non_seq_setup_rising",
        wire::TimingType::NonSeqSetupFalling => "non_seq_setup_falling",
        wire::TimingType::NonSeqHoldRising => "non_seq_hold_rising",
        wire::TimingType::NonSeqHoldFalling => "non_seq_hold_falling",
        wire::TimingType::NochangeHighHigh => "nochange_high_high",
        wire::TimingType::NochangeHighLow => "nochange_high_low",
        wire::TimingType::NochangeLowHigh => "nochange_low_high",
        wire::TimingType::NochangeLowLow => "nochange_low_low",
        wire::TimingType::Other => other,
    }
}

fn timing_sense_str(value: wire::TimingSense, other: &str) -> &str {
    enum_string(
        value,
        other,
        &[
            (wire::TimingSense::Unspecified, ""),
            (wire::TimingSense::PositiveUnate, "positive_unate"),
            (wire::TimingSense::NegativeUnate, "negative_unate"),
            (wire::TimingSense::NonUnate, "non_unate"),
        ],
    )
}

pub fn timing_table_kind_str(value: wire::TimingTableKind) -> &'static str {
    match value {
        wire::TimingTableKind::Unknown => "",
        wire::TimingTableKind::CellRise => "cell_rise",
        wire::TimingTableKind::CellFall => "cell_fall",
        wire::TimingTableKind::RiseTransition => "rise_transition",
        wire::TimingTableKind::FallTransition => "fall_transition",
        wire::TimingTableKind::RiseConstraint => "rise_constraint",
        wire::TimingTableKind::FallConstraint => "fall_constraint",
    }
}

/// Wire-format discriminator stored in every serialized Liberty library.
pub const LIBERTY_FORMAT_MAGIC: u64 = 0x4c49_4256_3350_524f;

/// Returns whether a decoded library has the expected wire-format magic.
pub fn has_valid_header(library: &wire::Library) -> bool {
    library.format_magic == LIBERTY_FORMAT_MAGIC
}

fn axis_key(axis: &[f64]) -> Vec<u64> {
    axis.iter().map(|value| value.to_bits()).collect()
}

fn record_axis(axis: &[f64], frequencies: &mut HashMap<Vec<u64>, u64>) {
    if !axis.is_empty() {
        *frequencies.entry(axis_key(axis)).or_default() += 1;
    }
}

fn record_string(value: &str, frequencies: &mut HashMap<String, u64>) {
    if !value.is_empty() {
        *frequencies.entry(value.to_string()).or_default() += 1;
    }
}

fn frequency_order<K: Ord>(frequencies: HashMap<K, u64>) -> Vec<(K, u64)> {
    let mut values: Vec<_> = frequencies.into_iter().collect();
    values.sort_by(|(lhs_value, lhs_frequency), (rhs_value, rhs_frequency)| {
        rhs_frequency
            .cmp(lhs_frequency)
            .then_with(|| lhs_value.cmp(rhs_value))
    });
    values
}

fn axis_id(axis: &[f64], id_by_axis: &HashMap<Vec<u64>, u32>) -> u32 {
    if axis.is_empty() {
        0
    } else {
        *id_by_axis
            .get(&axis_key(axis))
            .expect("every nonempty LUT axis was interned")
    }
}

fn string_id(value: &str, id_by_string: &HashMap<String, u32>) -> u32 {
    if value.is_empty() {
        0
    } else {
        *id_by_string
            .get(value)
            .expect("every nonempty compacted string was interned")
    }
}

pub(crate) fn lut_template_kind_to_wire(value: &str) -> wire::LutTemplateKind {
    match value {
        "" => wire::LutTemplateKind::Unknown,
        "lu_table_template" => wire::LutTemplateKind::Timing,
        "power_lut_template" => wire::LutTemplateKind::Power,
        _ => wire::LutTemplateKind::Other,
    }
}

pub(crate) fn lut_variable_to_wire(value: &str) -> wire::LutVariable {
    match value {
        "" => wire::LutVariable::Unspecified,
        "input_net_transition" => wire::LutVariable::InputNetTransition,
        "input_transition_time" => wire::LutVariable::InputTransitionTime,
        "total_output_net_capacitance" => wire::LutVariable::TotalOutputNetCapacitance,
        "related_pin_transition" => wire::LutVariable::RelatedPinTransition,
        "constrained_pin_transition" => wire::LutVariable::ConstrainedPinTransition,
        "output_net_length" => wire::LutVariable::OutputNetLength,
        "output_net_wire_cap" => wire::LutVariable::OutputNetWireCap,
        "output_net_pin_cap" => wire::LutVariable::OutputNetPinCap,
        "related_out_total_output_net_capacitance" => {
            wire::LutVariable::RelatedOutTotalOutputNetCapacitance
        }
        "related_out_output_net_length" => wire::LutVariable::RelatedOutOutputNetLength,
        "related_out_output_net_wire_cap" => wire::LutVariable::RelatedOutOutputNetWireCap,
        "related_out_output_net_pin_cap" => wire::LutVariable::RelatedOutOutputNetPinCap,
        "connect_delay" => wire::LutVariable::ConnectDelay,
        "time" => wire::LutVariable::Time,
        "input_voltage" => wire::LutVariable::InputVoltage,
        "output_voltage" => wire::LutVariable::OutputVoltage,
        "path_depth" => wire::LutVariable::PathDepth,
        "path_distance" => wire::LutVariable::PathDistance,
        "normalized_voltage" => wire::LutVariable::NormalizedVoltage,
        "temperature" => wire::LutVariable::Temperature,
        _ => wire::LutVariable::Other,
    }
}

pub(crate) fn timing_sense_to_wire(value: &str) -> wire::TimingSense {
    match value {
        "" => wire::TimingSense::Unspecified,
        "positive_unate" => wire::TimingSense::PositiveUnate,
        "negative_unate" => wire::TimingSense::NegativeUnate,
        "non_unate" => wire::TimingSense::NonUnate,
        _ => wire::TimingSense::Other,
    }
}

pub(crate) fn timing_type_to_wire(value: &str) -> wire::TimingType {
    match value {
        "" => wire::TimingType::Unspecified,
        "combinational" => wire::TimingType::Combinational,
        "combinational_rise" => wire::TimingType::CombinationalRise,
        "combinational_fall" => wire::TimingType::CombinationalFall,
        "rising_edge" => wire::TimingType::RisingEdge,
        "falling_edge" => wire::TimingType::FallingEdge,
        "preset" => wire::TimingType::Preset,
        "clear" => wire::TimingType::Clear,
        "hold_rising" => wire::TimingType::HoldRising,
        "hold_falling" => wire::TimingType::HoldFalling,
        "setup_rising" => wire::TimingType::SetupRising,
        "setup_falling" => wire::TimingType::SetupFalling,
        "recovery_rising" => wire::TimingType::RecoveryRising,
        "recovery_falling" => wire::TimingType::RecoveryFalling,
        "removal_rising" => wire::TimingType::RemovalRising,
        "removal_falling" => wire::TimingType::RemovalFalling,
        "skew_rising" => wire::TimingType::SkewRising,
        "skew_falling" => wire::TimingType::SkewFalling,
        "min_pulse_width" => wire::TimingType::MinPulseWidth,
        "minimum_period" => wire::TimingType::MinimumPeriod,
        "three_state_enable" => wire::TimingType::ThreeStateEnable,
        "three_state_disable" => wire::TimingType::ThreeStateDisable,
        "three_state_enable_rise" => wire::TimingType::ThreeStateEnableRise,
        "three_state_enable_fall" => wire::TimingType::ThreeStateEnableFall,
        "three_state_disable_rise" => wire::TimingType::ThreeStateDisableRise,
        "three_state_disable_fall" => wire::TimingType::ThreeStateDisableFall,
        "non_seq_setup_rising" => wire::TimingType::NonSeqSetupRising,
        "non_seq_setup_falling" => wire::TimingType::NonSeqSetupFalling,
        "non_seq_hold_rising" => wire::TimingType::NonSeqHoldRising,
        "non_seq_hold_falling" => wire::TimingType::NonSeqHoldFalling,
        "nochange_high_high" => wire::TimingType::NochangeHighHigh,
        "nochange_high_low" => wire::TimingType::NochangeHighLow,
        "nochange_low_high" => wire::TimingType::NochangeLowHigh,
        "nochange_low_low" => wire::TimingType::NochangeLowLow,
        _ => wire::TimingType::Other,
    }
}

pub(crate) fn timing_table_kind_to_wire(value: &str) -> Result<wire::TimingTableKind> {
    match value {
        "cell_rise" => Ok(wire::TimingTableKind::CellRise),
        "cell_fall" => Ok(wire::TimingTableKind::CellFall),
        "rise_transition" => Ok(wire::TimingTableKind::RiseTransition),
        "fall_transition" => Ok(wire::TimingTableKind::FallTransition),
        "rise_constraint" => Ok(wire::TimingTableKind::RiseConstraint),
        "fall_constraint" => Ok(wire::TimingTableKind::FallConstraint),
        _ => bail!("unsupported evaluator timing-table kind '{value}'"),
    }
}

fn record_enum_fallback(value: &str, is_other: bool, frequencies: &mut HashMap<String, u64>) {
    if is_other {
        record_string(value, frequencies);
    }
}

fn collect_string_frequencies(library: &Library) -> HashMap<String, u64> {
    let mut frequencies = HashMap::new();
    for template in &library.lu_table_templates {
        record_enum_fallback(
            template.kind.fallback(library),
            template.kind.wire_value() == wire::LutTemplateKind::Other,
            &mut frequencies,
        );
        for variable in [
            &template.variable_1,
            &template.variable_2,
            &template.variable_3,
        ] {
            record_enum_fallback(
                variable.fallback(library),
                variable.wire_value() == wire::LutVariable::Other,
                &mut frequencies,
            );
        }
    }
    for cell in &library.cells {
        for pin in &cell.pins {
            record_string(library.resolve_string(&pin.function), &mut frequencies);
            record_string(library.resolve_string(&pin.name), &mut frequencies);
            for arc in &pin.timing_arcs {
                record_string(library.resolve_string(&arc.related_pin), &mut frequencies);
                record_string(library.resolve_string(&arc.when), &mut frequencies);
                record_enum_fallback(
                    arc.timing_sense.fallback(library),
                    arc.timing_sense.wire_value() == wire::TimingSense::Other,
                    &mut frequencies,
                );
                record_enum_fallback(
                    arc.timing_type.fallback(library),
                    arc.timing_type.wire_value() == wire::TimingType::Other,
                    &mut frequencies,
                );
                for table in &arc.tables {
                    if table.kind != wire::TimingTableKind::Unknown {
                        record_string(library.timing_table_template_name(table), &mut frequencies);
                    }
                }
            }
            for group in &pin.internal_power {
                for related_pin in &group.related_pins {
                    record_string(library.resolve_string(related_pin), &mut frequencies);
                }
                record_string(library.resolve_string(&group.when), &mut frequencies);
                record_string(
                    library.resolve_string(&group.related_pg_pin),
                    &mut frequencies,
                );
                for table in &group.tables {
                    record_string(library.power_table_template_name(table), &mut frequencies);
                }
            }
        }
    }
    frequencies
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct LutShapeKey {
    template_id: u32,
    index_1_id: u32,
    index_2_id: u32,
    index_3_id: u32,
    dimensions: Vec<u32>,
    template_name_string_id: u32,
}

fn shape_key(
    template_id: u32,
    index_1: &[f64],
    index_2: &[f64],
    index_3: &[f64],
    dimensions: &[u32],
    template_name: &str,
    id_by_axis: &HashMap<Vec<u64>, u32>,
    id_by_string: &HashMap<String, u32>,
) -> LutShapeKey {
    LutShapeKey {
        template_id,
        index_1_id: axis_id(index_1, id_by_axis),
        index_2_id: axis_id(index_2, id_by_axis),
        index_3_id: axis_id(index_3, id_by_axis),
        dimensions: dimensions.to_vec(),
        template_name_string_id: string_id(template_name, id_by_string),
    }
}

fn shape_id(key: &LutShapeKey, id_by_shape: &HashMap<LutShapeKey, u32>) -> u32 {
    *id_by_shape
        .get(key)
        .expect("every LUT table shape was interned")
}

fn values_to_f32(values: Vec<f64>) -> Result<Vec<f32>> {
    let mut result = Vec::with_capacity(values.len());
    for value in values {
        let compact = value as f32;
        if value.is_finite() && !compact.is_finite() {
            bail!("LUT value {value} is outside the finite float32 range");
        }
        result.push(compact);
    }
    Ok(result)
}

/// Converts normalized data into the compact protobuf representation.
///
/// Timing-table kinds not consumed by the deterministic evaluator are omitted.
pub fn library_to_proto(mut library: Library) -> Result<wire::Library> {
    let mut axis_frequencies = HashMap::new();
    for template in &library.lu_table_templates {
        record_axis(&template.index_1, &mut axis_frequencies);
        record_axis(&template.index_2, &mut axis_frequencies);
        record_axis(&template.index_3, &mut axis_frequencies);
    }
    for cell in &library.cells {
        for pin in &cell.pins {
            for arc in &pin.timing_arcs {
                for table in &arc.tables {
                    if table.kind == wire::TimingTableKind::Unknown {
                        continue;
                    }
                    for axis in library.timing_table_axes(table) {
                        record_axis(axis, &mut axis_frequencies);
                    }
                }
            }
            for group in &pin.internal_power {
                for table in &group.tables {
                    for axis in library.power_table_axes(table) {
                        record_axis(axis, &mut axis_frequencies);
                    }
                }
            }
        }
    }

    let axes = frequency_order(axis_frequencies);
    let mut id_by_axis = HashMap::with_capacity(axes.len());
    let mut lut_axes = Vec::with_capacity(axes.len());
    for (index, (axis, _frequency)) in axes.into_iter().enumerate() {
        let id = u32::try_from(index + 1).map_err(|_| anyhow::anyhow!("too many LUT axes"))?;
        id_by_axis.insert(axis.clone(), id);
        lut_axes.push(wire::LutAxis {
            values: axis.into_iter().map(f64::from_bits).collect(),
        });
    }

    let strings = frequency_order(collect_string_frequencies(&library));
    let mut id_by_string = HashMap::with_capacity(strings.len());
    let mut interned_strings = Vec::with_capacity(strings.len());
    for (index, (value, _frequency)) in strings.into_iter().enumerate() {
        let id = u32::try_from(index + 1).map_err(|_| anyhow::anyhow!("too many strings"))?;
        id_by_string.insert(value.clone(), id);
        interned_strings.push(value);
    }

    let mut shape_frequencies = HashMap::new();
    for cell in &library.cells {
        for pin in &cell.pins {
            for arc in &pin.timing_arcs {
                for table in &arc.tables {
                    if table.kind == wire::TimingTableKind::Unknown {
                        continue;
                    }
                    let shape = library.timing_table_shape(table);
                    let axes = library.timing_table_axes(table);
                    *shape_frequencies
                        .entry(shape_key(
                            shape.template_id,
                            axes[0],
                            axes[1],
                            axes[2],
                            &shape.dimensions,
                            library.timing_table_template_name(table),
                            &id_by_axis,
                            &id_by_string,
                        ))
                        .or_default() += 1;
                }
            }
            for group in &pin.internal_power {
                for table in &group.tables {
                    let shape = library.power_table_shape(table);
                    let axes = library.power_table_axes(table);
                    *shape_frequencies
                        .entry(shape_key(
                            shape.template_id,
                            axes[0],
                            axes[1],
                            axes[2],
                            &shape.dimensions,
                            library.power_table_template_name(table),
                            &id_by_axis,
                            &id_by_string,
                        ))
                        .or_default() += 1;
                }
            }
        }
    }
    let shapes = frequency_order(shape_frequencies);
    let mut id_by_shape = HashMap::with_capacity(shapes.len());
    let mut lut_shapes = Vec::with_capacity(shapes.len());
    for (index, (shape, _frequency)) in shapes.into_iter().enumerate() {
        let id = u32::try_from(index + 1).map_err(|_| anyhow::anyhow!("too many LUT shapes"))?;
        id_by_shape.insert(shape.clone(), id);
        lut_shapes.push(wire::LutShape {
            template_id: shape.template_id,
            index_1_id: shape.index_1_id,
            index_2_id: shape.index_2_id,
            index_3_id: shape.index_3_id,
            dimensions: shape.dimensions,
            template_name_string_id: shape.template_name_string_id,
        });
    }

    let model_templates = std::mem::take(&mut library.lu_table_templates);
    let lu_table_templates = model_templates
        .into_iter()
        .map(|template| {
            let kind = template.kind.wire_value();
            let variable_1 = template.variable_1.wire_value();
            let variable_2 = template.variable_2.wire_value();
            let variable_3 = template.variable_3.wire_value();
            wire::LuTableTemplate {
                kind: kind as i32,
                name: template.name,
                variable_1: variable_1 as i32,
                variable_2: variable_2 as i32,
                variable_3: variable_3 as i32,
                index_1_id: axis_id(&template.index_1, &id_by_axis),
                index_2_id: axis_id(&template.index_2, &id_by_axis),
                index_3_id: axis_id(&template.index_3, &id_by_axis),
                kind_string_id: if kind == wire::LutTemplateKind::Other {
                    string_id(template.kind.fallback(&library), &id_by_string)
                } else {
                    0
                },
                variable_1_string_id: if variable_1 == wire::LutVariable::Other {
                    string_id(template.variable_1.fallback(&library), &id_by_string)
                } else {
                    0
                },
                variable_2_string_id: if variable_2 == wire::LutVariable::Other {
                    string_id(template.variable_2.fallback(&library), &id_by_string)
                } else {
                    0
                },
                variable_3_string_id: if variable_3 == wire::LutVariable::Other {
                    string_id(template.variable_3.fallback(&library), &id_by_string)
                } else {
                    0
                },
            }
        })
        .collect();
    let model_cells = std::mem::take(&mut library.cells);
    let cells = model_cells
        .into_iter()
        .map(|cell| {
            let pins = cell
                .pins
                .into_iter()
                .map(|pin| {
                    let timing_arcs = pin
                        .timing_arcs
                        .into_iter()
                        .map(|arc| {
                            let tables = arc
                                .tables
                                .into_iter()
                                .filter(|table| table.kind != wire::TimingTableKind::Unknown)
                                .map(|table| {
                                    let shape = library.timing_table_shape(&table);
                                    let axes = library.timing_table_axes(&table);
                                    let key = shape_key(
                                        shape.template_id,
                                        axes[0],
                                        axes[1],
                                        axes[2],
                                        &shape.dimensions,
                                        library.timing_table_template_name(&table),
                                        &id_by_axis,
                                        &id_by_string,
                                    );
                                    Ok(wire::TimingTable {
                                        kind: table.kind as i32,
                                        shape_id: shape_id(&key, &id_by_shape),
                                        values: library.timing_table_values(&table).to_vec(),
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let timing_sense = arc.timing_sense.wire_value();
                            let timing_type = arc.timing_type.wire_value();
                            Ok(wire::TimingArc {
                                related_pin_string_id: string_id(
                                    library.resolve_string(&arc.related_pin),
                                    &id_by_string,
                                ),
                                timing_sense: timing_sense as i32,
                                timing_type: timing_type as i32,
                                when_string_id: string_id(
                                    library.resolve_string(&arc.when),
                                    &id_by_string,
                                ),
                                tables,
                                timing_sense_string_id: if timing_sense == wire::TimingSense::Other
                                {
                                    string_id(arc.timing_sense.fallback(&library), &id_by_string)
                                } else {
                                    0
                                },
                                timing_type_string_id: if timing_type == wire::TimingType::Other {
                                    string_id(arc.timing_type.fallback(&library), &id_by_string)
                                } else {
                                    0
                                },
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let internal_power = pin
                        .internal_power
                        .into_iter()
                        .map(|group| {
                            let tables = group
                                .tables
                                .into_iter()
                                .map(|table| {
                                    let shape = library.power_table_shape(&table);
                                    let axes = library.power_table_axes(&table);
                                    let key = shape_key(
                                        shape.template_id,
                                        axes[0],
                                        axes[1],
                                        axes[2],
                                        &shape.dimensions,
                                        library.power_table_template_name(&table),
                                        &id_by_axis,
                                        &id_by_string,
                                    );
                                    Ok(wire::PowerTable {
                                        transition: table.transition as i32,
                                        shape_id: shape_id(&key, &id_by_shape),
                                        values: library.power_table_values(&table).to_vec(),
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?;
                            Ok(wire::InternalPower {
                                related_pin_string_ids: group
                                    .related_pins
                                    .iter()
                                    .map(|value| {
                                        string_id(library.resolve_string(value), &id_by_string)
                                    })
                                    .collect(),
                                when_string_id: string_id(
                                    library.resolve_string(&group.when),
                                    &id_by_string,
                                ),
                                related_pg_pin_string_id: string_id(
                                    library.resolve_string(&group.related_pg_pin),
                                    &id_by_string,
                                ),
                                tables,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(wire::Pin {
                        direction: pin.direction,
                        function_string_id: string_id(
                            library.resolve_string(&pin.function),
                            &id_by_string,
                        ),
                        name_string_id: string_id(library.resolve_string(&pin.name), &id_by_string),
                        is_clocking_pin: pin.is_clocking_pin,
                        capacitance: pin.capacitance,
                        rise_capacitance: pin.rise_capacitance,
                        fall_capacitance: pin.fall_capacitance,
                        max_capacitance: pin.max_capacitance,
                        timing_arcs,
                        internal_power,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(wire::Cell {
                name: cell.name,
                pins,
                area: cell.area,
                sequential: cell.sequential,
                clock_gate: cell.clock_gate,
                threshold_voltage_group_id: cell.threshold_voltage_group_id,
                dont_use: cell.dont_use,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(wire::Library {
        format_magic: LIBERTY_FORMAT_MAGIC,
        cells,
        units: library.units,
        lu_table_templates,
        threshold_voltage_groups: library.threshold_voltage_groups,
        default_threshold_voltage_group_id: library.default_threshold_voltage_group_id,
        threshold_voltage_group_class_indices: library.threshold_voltage_group_class_indices,
        nominal_voltage: library.nominal_voltage,
        provenance: library.provenance,
        source_files: library.source_files,
        lut_axes,
        interned_strings,
        lut_shapes,
    })
}

fn axis_id_from_wire(id: u32, axis_count: usize) -> Result<AxisId> {
    if id as usize > axis_count {
        bail!("LUT axis ID {id} is out of range");
    }
    Ok(AxisId(id))
}

fn string_id_from_wire(id: u32, string_count: usize) -> Result<StringId> {
    if id as usize > string_count {
        bail!("interned string ID {id} is out of range");
    }
    Ok(StringId::from_pool_id(id))
}

fn axis_values(id: AxisId, axes: &[Box<[f64]>]) -> &[f64] {
    if id.is_unset() {
        &[]
    } else {
        &axes[(id.0 - 1) as usize]
    }
}

fn validate_shape_id(shape_id: u32, shape_count: usize) -> Result<()> {
    if shape_id == 0 {
        bail!("LUT table has no shape ID");
    }
    if shape_id as usize > shape_count {
        bail!("LUT shape ID {shape_id} is out of range");
    }
    Ok(())
}

fn timing_table_from_wire(
    mut table: wire::TimingTable,
    shape_count: usize,
    value_storage: &mut Vec<f32>,
) -> Result<TimingTable> {
    validate_shape_id(table.shape_id, shape_count)?;
    let values = append_lut_values(value_storage, std::mem::take(&mut table.values))?;
    Ok(TimingTable {
        kind: wire::TimingTableKind::try_from(table.kind)
            .map_err(|_| anyhow::anyhow!("invalid timing-table kind {}", table.kind))?,
        shape_id: table.shape_id,
        values,
    })
}

fn power_table_from_wire(
    mut table: wire::PowerTable,
    shape_count: usize,
    value_storage: &mut Vec<f32>,
) -> Result<PowerTable> {
    validate_shape_id(table.shape_id, shape_count)?;
    let values = append_lut_values(value_storage, std::mem::take(&mut table.values))?;
    Ok(PowerTable {
        transition: wire::PowerTransition::try_from(table.transition)
            .map_err(|_| anyhow::anyhow!("invalid power transition {}", table.transition))?,
        shape_id: table.shape_id,
        values,
    })
}

fn append_lut_values(storage: &mut Vec<f32>, mut values: Vec<f32>) -> Result<LutValueRange> {
    let start = u32::try_from(storage.len())
        .map_err(|_| anyhow::anyhow!("LUT value storage exceeds u32"))?;
    let len = u32::try_from(values.len())
        .map_err(|_| anyhow::anyhow!("LUT table value count exceeds u32"))?;
    storage.append(&mut values);
    Ok(LutValueRange { start, len })
}

fn other_string_id(
    is_other: bool,
    string_id: u32,
    string_count: usize,
    field: &str,
) -> Result<StringId> {
    if !is_other {
        return Ok(StringId::default());
    }
    if string_id == 0 {
        bail!("{field} uses OTHER without a fallback string ID");
    }
    string_id_from_wire(string_id, string_count)
}

/// Converts a protobuf payload into the fully populated pooled runtime model.
pub fn library_from_proto(mut library: wire::Library) -> Result<Library> {
    if !has_valid_header(&library) {
        bail!(
            "invalid Liberty proto header: magic=0x{:016x}",
            library.format_magic
        );
    }
    if let Some(index) = library.interned_strings.iter().position(String::is_empty) {
        bail!("interned string pool entry {} is empty", index + 1);
    }
    if let Some(index) = library
        .lut_axes
        .iter()
        .position(|axis| axis.values.is_empty())
    {
        bail!("LUT axis pool entry {} is empty", index + 1);
    }
    let strings: Vec<Box<str>> = std::mem::take(&mut library.interned_strings)
        .into_iter()
        .map(String::into_boxed_str)
        .collect();
    let axes: Vec<Box<[f64]>> = std::mem::take(&mut library.lut_axes)
        .into_iter()
        .map(|axis| axis.values.into_boxed_slice())
        .collect();
    let string_count = strings.len();
    let axis_count = axes.len();
    let shapes: Vec<LutShape> = std::mem::take(&mut library.lut_shapes)
        .into_iter()
        .map(|shape| {
            Ok(LutShape {
                template_id: shape.template_id,
                index_1: axis_id_from_wire(shape.index_1_id, axis_count)?,
                index_2: axis_id_from_wire(shape.index_2_id, axis_count)?,
                index_3: axis_id_from_wire(shape.index_3_id, axis_count)?,
                dimensions: shape.dimensions.into_boxed_slice(),
                template_name: string_id_from_wire(shape.template_name_string_id, string_count)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let value_count: usize = library
        .cells
        .iter()
        .flat_map(|cell| &cell.pins)
        .map(|pin| {
            pin.timing_arcs
                .iter()
                .flat_map(|arc| &arc.tables)
                .map(|table| table.values.len())
                .sum::<usize>()
                + pin
                    .internal_power
                    .iter()
                    .flat_map(|group| &group.tables)
                    .map(|table| table.values.len())
                    .sum::<usize>()
        })
        .sum();
    let mut value_storage = Vec::with_capacity(value_count);
    let shape_count = shapes.len();

    let lu_table_templates = std::mem::take(&mut library.lu_table_templates)
        .into_iter()
        .map(|template| {
            let kind = wire::LutTemplateKind::try_from(template.kind)
                .map_err(|_| anyhow::anyhow!("invalid LUT template kind {}", template.kind))?;
            let variable_1 = wire::LutVariable::try_from(template.variable_1)
                .map_err(|_| anyhow::anyhow!("invalid LUT variable {}", template.variable_1))?;
            let variable_2 = wire::LutVariable::try_from(template.variable_2)
                .map_err(|_| anyhow::anyhow!("invalid LUT variable {}", template.variable_2))?;
            let variable_3 = wire::LutVariable::try_from(template.variable_3)
                .map_err(|_| anyhow::anyhow!("invalid LUT variable {}", template.variable_3))?;
            let kind_other = other_string_id(
                kind == wire::LutTemplateKind::Other,
                template.kind_string_id,
                string_count,
                "LUT template kind",
            )?;
            let variable_1_other = other_string_id(
                variable_1 == wire::LutVariable::Other,
                template.variable_1_string_id,
                string_count,
                "LUT variable_1",
            )?;
            let variable_2_other = other_string_id(
                variable_2 == wire::LutVariable::Other,
                template.variable_2_string_id,
                string_count,
                "LUT variable_2",
            )?;
            let variable_3_other = other_string_id(
                variable_3 == wire::LutVariable::Other,
                template.variable_3_string_id,
                string_count,
                "LUT variable_3",
            )?;
            Ok(LuTableTemplate {
                kind: LutTemplateKind::from_wire(kind, kind_other),
                name: template.name,
                variable_1: LutVariable::from_wire(variable_1, variable_1_other),
                variable_2: LutVariable::from_wire(variable_2, variable_2_other),
                variable_3: LutVariable::from_wire(variable_3, variable_3_other),
                index_1: axis_values(axis_id_from_wire(template.index_1_id, axis_count)?, &axes)
                    .to_vec(),
                index_2: axis_values(axis_id_from_wire(template.index_2_id, axis_count)?, &axes)
                    .to_vec(),
                index_3: axis_values(axis_id_from_wire(template.index_3_id, axis_count)?, &axes)
                    .to_vec(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let cells = std::mem::take(&mut library.cells)
        .into_iter()
        .map(|cell| {
            let pins = cell
                .pins
                .into_iter()
                .map(|pin| {
                    let timing_arcs = pin
                        .timing_arcs
                        .into_iter()
                        .map(|arc| {
                            let tables = arc
                                .tables
                                .into_iter()
                                .map(|table| {
                                    timing_table_from_wire(table, shape_count, &mut value_storage)
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let timing_sense = wire::TimingSense::try_from(arc.timing_sense)
                                .map_err(|_| {
                                    anyhow::anyhow!("invalid timing sense {}", arc.timing_sense)
                                })?;
                            let timing_type =
                                wire::TimingType::try_from(arc.timing_type).map_err(|_| {
                                    anyhow::anyhow!("invalid timing type {}", arc.timing_type)
                                })?;
                            let timing_sense_other = other_string_id(
                                timing_sense == wire::TimingSense::Other,
                                arc.timing_sense_string_id,
                                string_count,
                                "timing sense",
                            )?;
                            let timing_type_other = other_string_id(
                                timing_type == wire::TimingType::Other,
                                arc.timing_type_string_id,
                                string_count,
                                "timing type",
                            )?;
                            Ok(TimingArc {
                                related_pin: string_id_from_wire(
                                    arc.related_pin_string_id,
                                    string_count,
                                )?,
                                timing_sense: TimingSense::from_wire(
                                    timing_sense,
                                    timing_sense_other,
                                ),
                                timing_type: TimingType::from_wire(timing_type, timing_type_other),
                                when: string_id_from_wire(arc.when_string_id, string_count)?,
                                tables,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let internal_power = pin
                        .internal_power
                        .into_iter()
                        .map(|group| {
                            let tables = group
                                .tables
                                .into_iter()
                                .map(|table| {
                                    power_table_from_wire(table, shape_count, &mut value_storage)
                                })
                                .collect::<Result<Vec<_>>>()?;
                            Ok(InternalPower {
                                related_pins: group
                                    .related_pin_string_ids
                                    .into_iter()
                                    .map(|id| string_id_from_wire(id, string_count))
                                    .collect::<Result<Vec<_>>>()?,
                                when: string_id_from_wire(group.when_string_id, string_count)?,
                                related_pg_pin: string_id_from_wire(
                                    group.related_pg_pin_string_id,
                                    string_count,
                                )?,
                                tables,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Pin {
                        direction: pin.direction,
                        function: string_id_from_wire(pin.function_string_id, string_count)?,
                        name: string_id_from_wire(pin.name_string_id, string_count)?,
                        is_clocking_pin: pin.is_clocking_pin,
                        capacitance: pin.capacitance,
                        rise_capacitance: pin.rise_capacitance,
                        fall_capacitance: pin.fall_capacitance,
                        max_capacitance: pin.max_capacitance,
                        timing_arcs,
                        internal_power,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Cell {
                name: cell.name,
                pins,
                area: cell.area,
                sequential: cell.sequential,
                clock_gate: cell.clock_gate,
                threshold_voltage_group_id: cell.threshold_voltage_group_id,
                dont_use: cell.dont_use,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Library {
        cells,
        units: library.units,
        lu_table_templates,
        threshold_voltage_groups: library.threshold_voltage_groups,
        default_threshold_voltage_group_id: library.default_threshold_voltage_group_id,
        threshold_voltage_group_class_indices: library.threshold_voltage_group_class_indices,
        nominal_voltage: library.nominal_voltage,
        provenance: library.provenance,
        source_files: library.source_files,
        strings,
        lut_axes: axes,
        lut_shapes: shapes,
        lut_values: value_storage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn string_id_is_a_four_byte_copy_handle() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<StringId>();
        assert_eq!(std::mem::size_of::<StringId>(), 4);
    }

    #[test]
    fn proto_conversion_roundtrips_with_float32_precision() {
        let shared_axis = vec![0.01, 0.02];
        let mut builder = LibraryBuilder::new();
        let rise = builder
            .add_timing_table_f64(
                wire::TimingTableKind::CellRise,
                0,
                shared_axis.clone(),
                vec![],
                vec![],
                vec![0.1, 0.2],
                vec![2],
                "",
            )
            .unwrap();
        let fall = builder
            .add_timing_table_f64(
                wire::TimingTableKind::CellFall,
                0,
                shared_axis.clone(),
                vec![],
                vec![],
                vec![0.5, 0.6],
                vec![2],
                "",
            )
            .unwrap();
        let power = builder
            .add_power_table_f64(
                wire::PowerTransition::Rise,
                0,
                shared_axis.clone(),
                vec![],
                vec![],
                vec![0.3, 0.4],
                vec![2],
                "",
            )
            .unwrap();
        let kind = LutTemplateKind::from_raw_in(&mut builder, "custom_template").unwrap();
        let variable_1 = LutVariable::from_raw_in(&mut builder, "custom_variable").unwrap();
        builder.lu_table_templates = vec![LuTableTemplate {
            kind,
            variable_1,
            index_1: shared_axis.clone(),
            ..Default::default()
        }];
        let function = builder.intern_string("A").unwrap();
        let name = builder.intern_string("Y").unwrap();
        let arc = builder
            .add_timing_arc(
                "A",
                "positive_unate",
                "custom_timing_type",
                "ENABLE",
                vec![rise, fall],
            )
            .unwrap();
        let related_pin = builder.intern_string("A").unwrap();
        let when = builder.intern_string("ENABLE").unwrap();
        let related_pg_pin = builder.intern_string("VDD").unwrap();
        builder.cells = vec![Cell {
            pins: vec![Pin {
                function,
                name,
                timing_arcs: vec![arc],
                internal_power: vec![InternalPower {
                    related_pins: vec![related_pin],
                    when,
                    related_pg_pin,
                    tables: vec![power],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }];

        let proto = library_to_proto(builder.finish()).unwrap();
        let second_proto = library_to_proto(library_from_proto(proto.clone()).unwrap()).unwrap();

        assert!(has_valid_header(&proto));
        assert_eq!(proto.encode_to_vec(), second_proto.encode_to_vec());
        assert_eq!(proto.lut_axes.len(), 1);
        assert_eq!(proto.lut_shapes.len(), 1);
        assert!(proto.interned_strings.iter().any(|value| value == "ENABLE"));
        assert_eq!(proto.lu_table_templates[0].index_1_id, 1);
        assert_eq!(
            proto.lu_table_templates[0].kind,
            wire::LutTemplateKind::Other as i32
        );
        assert_ne!(proto.lu_table_templates[0].kind_string_id, 0);
        assert_eq!(
            proto.cells[0].pins[0].timing_arcs[0].timing_sense,
            wire::TimingSense::PositiveUnate as i32
        );
        assert_eq!(
            proto.cells[0].pins[0].timing_arcs[0].timing_type,
            wire::TimingType::Other as i32
        );
        let timing_shape_id = proto.cells[0].pins[0].timing_arcs[0].tables[0].shape_id;
        assert_eq!(
            timing_shape_id,
            proto.cells[0].pins[0].timing_arcs[0].tables[1].shape_id
        );
        assert_eq!(
            timing_shape_id,
            proto.cells[0].pins[0].internal_power[0].tables[0].shape_id
        );
        assert_eq!(
            proto.cells[0].pins[0].timing_arcs[0].tables[0].values,
            vec![0.1_f32, 0.2_f32]
        );
        let roundtrip = library_from_proto(proto).unwrap();
        assert_eq!(
            roundtrip.lu_table_templates[0].kind_str(&roundtrip),
            "custom_template"
        );
        assert_eq!(
            roundtrip.lu_table_templates[0].variable_1_str(&roundtrip),
            "custom_variable"
        );
        assert_eq!(
            roundtrip.resolve_string(&roundtrip.cells[0].pins[0].function),
            "A"
        );
        assert_eq!(
            roundtrip.resolve_string(&roundtrip.cells[0].pins[0].name),
            "Y"
        );
        assert_eq!(
            roundtrip.cells[0].pins[0].function,
            roundtrip.cells[0].pins[0].timing_arcs[0].related_pin
        );
        assert_eq!(
            roundtrip.cells[0].pins[0].timing_arcs[0].timing_type_str(&roundtrip),
            "custom_timing_type"
        );
        assert_eq!(
            roundtrip.resolve_string(&roundtrip.cells[0].pins[0].timing_arcs[0].when),
            "ENABLE"
        );
        assert_eq!(
            roundtrip.cells[0].pins[0].timing_arcs[0].when,
            roundtrip.cells[0].pins[0].internal_power[0].when
        );
        let rise = &roundtrip.cells[0].pins[0].timing_arcs[0].tables[0];
        let fall = &roundtrip.cells[0].pins[0].timing_arcs[0].tables[1];
        let power = &roundtrip.cells[0].pins[0].internal_power[0].tables[0];
        assert_eq!(rise.shape_id(), fall.shape_id());
        assert_eq!(rise.shape_id(), power.shape_id());
        assert_eq!(roundtrip.timing_table_values(rise), &[0.1_f32, 0.2_f32]);
        assert_eq!(roundtrip.power_table_values(power), &[0.3_f32, 0.4_f32]);
    }

    #[test]
    fn proto_conversion_rejects_invalid_header() {
        let error = library_from_proto(wire::Library::default()).unwrap_err();
        assert!(format!("{error:#}").contains("invalid Liberty proto header"));
    }

    #[test]
    fn proto_conversion_rejects_empty_reference_pool_entries() {
        let empty_string = wire::Library {
            format_magic: LIBERTY_FORMAT_MAGIC,
            interned_strings: vec![String::new()],
            ..Default::default()
        };
        let error = library_from_proto(empty_string).unwrap_err();
        assert!(format!("{error:#}").contains("interned string pool entry 1 is empty"));

        let empty_axis = wire::Library {
            format_magic: LIBERTY_FORMAT_MAGIC,
            lut_axes: vec![wire::LutAxis::default()],
            ..Default::default()
        };
        let error = library_from_proto(empty_axis).unwrap_err();
        assert!(format!("{error:#}").contains("LUT axis pool entry 1 is empty"));
    }
}
