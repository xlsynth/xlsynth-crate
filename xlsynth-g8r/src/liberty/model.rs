// SPDX-License-Identifier: Apache-2.0

use crate::liberty_proto as wire;
use anyhow::{Result, bail};
use std::collections::{HashMap, VecDeque};
use std::ops::Deref;
use std::sync::Arc;

pub type SharedString = String;

pub type SharedAxis = Arc<Vec<f64>>;

fn empty_string() -> SharedString {
    SharedString::default()
}

fn empty_axis() -> SharedAxis {
    Arc::new(Vec::new())
}

pub fn shared_string(value: impl Into<String>) -> SharedString {
    value.into()
}

pub fn shared_axis(values: Vec<f64>) -> SharedAxis {
    Arc::new(values)
}

/// A range in a shared float32 LUT-value allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LutValueRange {
    pub start: u32,
    pub len: u32,
}

/// A slice-like reference into a shared float32 LUT-value allocation.
#[derive(Clone, Debug)]
pub struct LutValues {
    storage: Arc<Vec<f32>>,
    range: LutValueRange,
}

impl Default for LutValues {
    fn default() -> Self {
        Self {
            storage: Arc::new(Vec::new()),
            range: LutValueRange::default(),
        }
    }
}

impl LutValues {
    pub fn from_f32(values: Vec<f32>) -> Self {
        let len = u32::try_from(values.len()).expect("LUT value count exceeds u32");
        Self {
            storage: Arc::new(values),
            range: LutValueRange { start: 0, len },
        }
    }

    pub fn from_f64(values: Vec<f64>) -> Result<Self> {
        Ok(Self::from_f32(values_to_f32(values)?))
    }

    fn from_shared(storage: Arc<Vec<f32>>, range: LutValueRange) -> Self {
        Self { storage, range }
    }

    pub fn range(&self) -> LutValueRange {
        self.range
    }

    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }
}

impl Deref for LutValues {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        let start = self.range.start as usize;
        let end = start + self.range.len as usize;
        &self.storage[start..end]
    }
}

impl<'a> IntoIterator for &'a LutValues {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl PartialEq for LutValues {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl PartialEq<Vec<f32>> for LutValues {
    fn eq(&self, other: &Vec<f32>) -> bool {
        self.deref() == other.as_slice()
    }
}

impl PartialEq<Vec<f64>> for LutValues {
    fn eq(&self, other: &Vec<f64>) -> bool {
        self.iter()
            .copied()
            .map(f64::from)
            .eq(other.iter().copied())
    }
}

impl From<Vec<f64>> for LutValues {
    fn from(values: Vec<f64>) -> Self {
        Self::from_f64(values).expect("test LUT values fit in float32")
    }
}

/// Shared geometry and template metadata for timing and power tables.
#[derive(Clone, Debug, PartialEq)]
pub struct LutShape {
    pub template_id: u32,
    pub index_1: SharedAxis,
    pub index_2: SharedAxis,
    pub index_3: SharedAxis,
    pub dimensions: Arc<Vec<u32>>,
    pub template_name: SharedString,
}

impl Default for LutShape {
    fn default() -> Self {
        Self {
            template_id: 0,
            index_1: empty_axis(),
            index_2: empty_axis(),
            index_3: empty_axis(),
            dimensions: Arc::new(Vec::new()),
            template_name: empty_string(),
        }
    }
}

/// Fully populated in-memory Liberty library used by parsers and evaluators.
#[derive(Clone, Debug, Default, PartialEq)]
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
    pub function: SharedString,
    pub name: SharedString,
    pub is_clocking_pin: bool,
    pub capacitance: Option<f64>,
    pub rise_capacitance: Option<f64>,
    pub fall_capacitance: Option<f64>,
    pub max_capacitance: Option<f64>,
    pub timing_arcs: Vec<TimingArc>,
    pub internal_power: Vec<InternalPower>,
}

/// LUT template metadata; table payloads share canonicalized shapes separately.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LuTableTemplate {
    pub kind: String,
    pub name: String,
    pub variable_1: String,
    pub variable_2: String,
    pub variable_3: String,
    pub index_1: Vec<f64>,
    pub index_2: Vec<f64>,
    pub index_3: Vec<f64>,
}

impl LuTableTemplate {
    pub fn kind_str(&self) -> &str {
        self.kind.as_str()
    }

    pub fn variable_1_str(&self) -> &str {
        self.variable_1.as_str()
    }

    pub fn variable_2_str(&self) -> &str {
        self.variable_2.as_str()
    }

    pub fn variable_3_str(&self) -> &str {
        self.variable_3.as_str()
    }
}

/// Normalized timing arc.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimingArc {
    pub related_pin: SharedString,
    pub timing_sense: String,
    pub timing_type: String,
    pub when: SharedString,
    pub tables: Vec<TimingTable>,
}

impl TimingArc {
    pub fn from_raw(
        related_pin: impl Into<String>,
        timing_sense: impl AsRef<str>,
        timing_type: impl AsRef<str>,
        when: impl Into<String>,
        tables: Vec<TimingTable>,
    ) -> Self {
        Self {
            related_pin: related_pin.into(),
            timing_sense: timing_sense.as_ref().to_string(),
            timing_type: timing_type.as_ref().to_string(),
            when: when.into(),
            tables,
        }
    }

    pub fn timing_sense_str(&self) -> &str {
        self.timing_sense.as_str()
    }

    pub fn timing_type_str(&self) -> &str {
        self.timing_type.as_str()
    }
}

/// Timing table referencing shared shape and float32 value storage.
#[derive(Clone, Debug, PartialEq)]
pub struct TimingTable {
    pub kind: wire::TimingTableKind,
    pub shape: Arc<LutShape>,
    pub values: LutValues,
}

impl Default for TimingTable {
    fn default() -> Self {
        Self {
            kind: wire::TimingTableKind::Unknown,
            shape: Arc::new(LutShape::default()),
            values: LutValues::default(),
        }
    }
}

impl TimingTable {
    #[allow(clippy::too_many_arguments)]
    pub fn from_f64(
        kind: wire::TimingTableKind,
        template_id: u32,
        index_1: Vec<f64>,
        index_2: Vec<f64>,
        index_3: Vec<f64>,
        values: Vec<f64>,
        dimensions: Vec<u32>,
        template_name: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            shape: Arc::new(LutShape {
                template_id,
                index_1: shared_axis(index_1),
                index_2: shared_axis(index_2),
                index_3: shared_axis(index_3),
                dimensions: Arc::new(dimensions),
                template_name: template_name.into(),
            }),
            values: values.into(),
        }
    }

    pub fn kind_str(&self) -> &'static str {
        timing_table_kind_str(self.kind)
    }
}

impl Deref for TimingTable {
    type Target = LutShape;

    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}

impl std::ops::DerefMut for TimingTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.shape)
    }
}

/// Normalized internal-power selector and table data.
#[derive(Clone, Debug, PartialEq)]
pub struct InternalPower {
    pub related_pins: Vec<SharedString>,
    pub when: SharedString,
    pub related_pg_pin: SharedString,
    pub tables: Vec<PowerTable>,
}

impl Default for InternalPower {
    fn default() -> Self {
        Self {
            related_pins: Vec::new(),
            when: empty_string(),
            related_pg_pin: empty_string(),
            tables: Vec::new(),
        }
    }
}

/// Power table referencing shared shape and float32 value storage.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerTable {
    pub transition: wire::PowerTransition,
    pub shape: Arc<LutShape>,
    pub values: LutValues,
}

impl Default for PowerTable {
    fn default() -> Self {
        Self {
            transition: wire::PowerTransition::Unknown,
            shape: Arc::new(LutShape::default()),
            values: LutValues::default(),
        }
    }
}

impl PowerTable {
    #[allow(clippy::too_many_arguments)]
    pub fn from_f64(
        transition: wire::PowerTransition,
        template_id: u32,
        index_1: Vec<f64>,
        index_2: Vec<f64>,
        index_3: Vec<f64>,
        values: Vec<f64>,
        dimensions: Vec<u32>,
        template_name: impl Into<String>,
    ) -> Self {
        Self {
            transition,
            shape: Arc::new(LutShape {
                template_id,
                index_1: shared_axis(index_1),
                index_2: shared_axis(index_2),
                index_3: shared_axis(index_3),
                dimensions: Arc::new(dimensions),
                template_name: template_name.into(),
            }),
            values: values.into(),
        }
    }
}

impl Deref for PowerTable {
    type Target = LutShape;

    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}

impl std::ops::DerefMut for PowerTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.shape)
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

fn retain_templates_and_remap_ids(
    library: &mut Library,
    mut keep: impl FnMut(&LuTableTemplate) -> bool,
) {
    let old_templates = std::mem::take(&mut library.lu_table_templates);
    let mut id_remap = vec![0u32; old_templates.len() + 1];
    for (old_index, template) in old_templates.into_iter().enumerate() {
        if keep(&template) {
            let new_id = library.lu_table_templates.len() as u32 + 1;
            id_remap[old_index + 1] = new_id;
            library.lu_table_templates.push(template);
        }
    }
    let remap = |id: &mut u32| {
        if *id != 0 {
            *id = id_remap.get(*id as usize).copied().unwrap_or(0);
        }
    };
    for cell in &mut library.cells {
        for pin in &mut cell.pins {
            for arc in &mut pin.timing_arcs {
                for table in &mut arc.tables {
                    remap(&mut Arc::make_mut(&mut table.shape).template_id);
                }
            }
            for group in &mut pin.internal_power {
                for table in &mut group.tables {
                    remap(&mut Arc::make_mut(&mut table.shape).template_id);
                }
            }
        }
    }
}

/// Removes timing arcs and timing templates from normalized library data.
pub fn strip_timing_data(library: &mut Library) {
    for cell in &mut library.cells {
        for pin in &mut cell.pins {
            pin.timing_arcs.clear();
        }
    }
    retain_templates_and_remap_ids(library, |template| template.kind != "lu_table_template");
}

/// Removes dynamic-power payloads and templates from normalized library data.
pub fn strip_power_data(library: &mut Library) {
    library.nominal_voltage = None;
    for cell in &mut library.cells {
        for pin in &mut cell.pins {
            pin.internal_power.clear();
        }
    }
    retain_templates_and_remap_ids(library, |template| template.kind != "power_lut_template");
}

/// Wire-format discriminator stored in every serialized Liberty library.
pub const LIBERTY_FORMAT_MAGIC: u64 = 0x4c49_4256_3350_524f;

/// Returns whether a timing-table kind is consumed by the deterministic STA.
pub fn is_evaluator_timing_table_kind(kind: &str) -> bool {
    matches!(
        kind,
        "cell_rise"
            | "cell_fall"
            | "rise_transition"
            | "fall_transition"
            | "rise_constraint"
            | "fall_constraint"
    )
}

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
            &template.kind,
            lut_template_kind_to_wire(&template.kind) == wire::LutTemplateKind::Other,
            &mut frequencies,
        );
        for variable in [
            &template.variable_1,
            &template.variable_2,
            &template.variable_3,
        ] {
            record_enum_fallback(
                variable,
                lut_variable_to_wire(variable) == wire::LutVariable::Other,
                &mut frequencies,
            );
        }
    }
    for cell in &library.cells {
        for pin in &cell.pins {
            record_string(&pin.function, &mut frequencies);
            record_string(&pin.name, &mut frequencies);
            for arc in &pin.timing_arcs {
                record_string(&arc.related_pin, &mut frequencies);
                record_string(&arc.when, &mut frequencies);
                record_enum_fallback(
                    &arc.timing_sense,
                    timing_sense_to_wire(&arc.timing_sense) == wire::TimingSense::Other,
                    &mut frequencies,
                );
                record_enum_fallback(
                    &arc.timing_type,
                    timing_type_to_wire(&arc.timing_type) == wire::TimingType::Other,
                    &mut frequencies,
                );
                for table in &arc.tables {
                    if table.kind != wire::TimingTableKind::Unknown {
                        record_string(&table.shape.template_name, &mut frequencies);
                    }
                }
            }
            for group in &pin.internal_power {
                for related_pin in &group.related_pins {
                    record_string(related_pin, &mut frequencies);
                }
                record_string(&group.when, &mut frequencies);
                record_string(&group.related_pg_pin, &mut frequencies);
                for table in &group.tables {
                    record_string(&table.shape.template_name, &mut frequencies);
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
pub fn library_to_proto(library: Library) -> Result<wire::Library> {
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
                    record_axis(&table.shape.index_1, &mut axis_frequencies);
                    record_axis(&table.shape.index_2, &mut axis_frequencies);
                    record_axis(&table.shape.index_3, &mut axis_frequencies);
                }
            }
            for group in &pin.internal_power {
                for table in &group.tables {
                    record_axis(&table.shape.index_1, &mut axis_frequencies);
                    record_axis(&table.shape.index_2, &mut axis_frequencies);
                    record_axis(&table.shape.index_3, &mut axis_frequencies);
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
                    *shape_frequencies
                        .entry(shape_key(
                            table.shape.template_id,
                            &table.shape.index_1,
                            &table.shape.index_2,
                            &table.shape.index_3,
                            &table.shape.dimensions,
                            &table.shape.template_name,
                            &id_by_axis,
                            &id_by_string,
                        ))
                        .or_default() += 1;
                }
            }
            for group in &pin.internal_power {
                for table in &group.tables {
                    *shape_frequencies
                        .entry(shape_key(
                            table.shape.template_id,
                            &table.shape.index_1,
                            &table.shape.index_2,
                            &table.shape.index_3,
                            &table.shape.dimensions,
                            &table.shape.template_name,
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

    let lu_table_templates = library
        .lu_table_templates
        .into_iter()
        .map(|template| {
            let kind = lut_template_kind_to_wire(&template.kind);
            let variable_1 = lut_variable_to_wire(&template.variable_1);
            let variable_2 = lut_variable_to_wire(&template.variable_2);
            let variable_3 = lut_variable_to_wire(&template.variable_3);
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
                    string_id(&template.kind, &id_by_string)
                } else {
                    0
                },
                variable_1_string_id: if variable_1 == wire::LutVariable::Other {
                    string_id(&template.variable_1, &id_by_string)
                } else {
                    0
                },
                variable_2_string_id: if variable_2 == wire::LutVariable::Other {
                    string_id(&template.variable_2, &id_by_string)
                } else {
                    0
                },
                variable_3_string_id: if variable_3 == wire::LutVariable::Other {
                    string_id(&template.variable_3, &id_by_string)
                } else {
                    0
                },
            }
        })
        .collect();
    let cells = library
        .cells
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
                                    let key = shape_key(
                                        table.shape.template_id,
                                        &table.shape.index_1,
                                        &table.shape.index_2,
                                        &table.shape.index_3,
                                        &table.shape.dimensions,
                                        &table.shape.template_name,
                                        &id_by_axis,
                                        &id_by_string,
                                    );
                                    Ok(wire::TimingTable {
                                        kind: table.kind as i32,
                                        shape_id: shape_id(&key, &id_by_shape),
                                        values: table.values.to_vec(),
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let timing_sense = timing_sense_to_wire(&arc.timing_sense);
                            let timing_type = timing_type_to_wire(&arc.timing_type);
                            Ok(wire::TimingArc {
                                related_pin_string_id: string_id(&arc.related_pin, &id_by_string),
                                timing_sense: timing_sense as i32,
                                timing_type: timing_type as i32,
                                when_string_id: string_id(&arc.when, &id_by_string),
                                tables,
                                timing_sense_string_id: if timing_sense == wire::TimingSense::Other
                                {
                                    string_id(&arc.timing_sense, &id_by_string)
                                } else {
                                    0
                                },
                                timing_type_string_id: if timing_type == wire::TimingType::Other {
                                    string_id(&arc.timing_type, &id_by_string)
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
                                    let key = shape_key(
                                        table.shape.template_id,
                                        &table.shape.index_1,
                                        &table.shape.index_2,
                                        &table.shape.index_3,
                                        &table.shape.dimensions,
                                        &table.shape.template_name,
                                        &id_by_axis,
                                        &id_by_string,
                                    );
                                    Ok(wire::PowerTable {
                                        transition: table.transition as i32,
                                        shape_id: shape_id(&key, &id_by_shape),
                                        values: table.values.to_vec(),
                                    })
                                })
                                .collect::<Result<Vec<_>>>()?;
                            Ok(wire::InternalPower {
                                related_pin_string_ids: group
                                    .related_pins
                                    .iter()
                                    .map(|value| string_id(value, &id_by_string))
                                    .collect(),
                                when_string_id: string_id(&group.when, &id_by_string),
                                related_pg_pin_string_id: string_id(
                                    &group.related_pg_pin,
                                    &id_by_string,
                                ),
                                tables,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(wire::Pin {
                        direction: pin.direction,
                        function_string_id: string_id(&pin.function, &id_by_string),
                        name_string_id: string_id(&pin.name, &id_by_string),
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

fn resolve_shared_axis(axis_id: u32, axes: &[SharedAxis]) -> Result<SharedAxis> {
    if axis_id == 0 {
        return Ok(empty_axis());
    }
    axes.get((axis_id - 1) as usize)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("LUT axis ID {axis_id} is out of range"))
}

fn resolve_shared_string(string_id: u32, strings: &[SharedString]) -> Result<SharedString> {
    if string_id == 0 {
        return Ok(empty_string());
    }
    strings
        .get((string_id - 1) as usize)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("interned string ID {string_id} is out of range"))
}

fn resolve_shape(shape_id: u32, shapes: &[Arc<LutShape>]) -> Result<&Arc<LutShape>> {
    if shape_id == 0 {
        bail!("LUT table has no shape ID");
    }
    shapes
        .get((shape_id - 1) as usize)
        .ok_or_else(|| anyhow::anyhow!("LUT shape ID {shape_id} is out of range"))
}

fn timing_table_from_wire(
    table: wire::TimingTable,
    shapes: &[Arc<LutShape>],
    values: Arc<Vec<f32>>,
    range: LutValueRange,
) -> Result<TimingTable> {
    let shape = resolve_shape(table.shape_id, shapes)?;
    Ok(TimingTable {
        kind: wire::TimingTableKind::try_from(table.kind)
            .map_err(|_| anyhow::anyhow!("invalid timing-table kind {}", table.kind))?,
        shape: shape.clone(),
        values: LutValues::from_shared(values, range),
    })
}

fn power_table_from_wire(
    table: wire::PowerTable,
    shapes: &[Arc<LutShape>],
    values: Arc<Vec<f32>>,
    range: LutValueRange,
) -> Result<PowerTable> {
    let shape = resolve_shape(table.shape_id, shapes)?;
    Ok(PowerTable {
        transition: wire::PowerTransition::try_from(table.transition)
            .map_err(|_| anyhow::anyhow!("invalid power transition {}", table.transition))?,
        shape: shape.clone(),
        values: LutValues::from_shared(values, range),
    })
}

fn append_lut_values(storage: &mut Vec<f32>, values: &mut Vec<f32>) -> Result<LutValueRange> {
    let start = u32::try_from(storage.len())
        .map_err(|_| anyhow::anyhow!("LUT value storage exceeds u32"))?;
    let len = u32::try_from(values.len())
        .map_err(|_| anyhow::anyhow!("LUT table value count exceeds u32"))?;
    storage.append(values);
    Ok(LutValueRange { start, len })
}

fn shared_other_string(
    is_other: bool,
    string_id: u32,
    strings: &[SharedString],
    field: &str,
) -> Result<SharedString> {
    if !is_other {
        return Ok(empty_string());
    }
    if string_id == 0 {
        bail!("{field} uses OTHER without a fallback string ID");
    }
    resolve_shared_string(string_id, strings)
}

/// Converts a protobuf payload into the fully populated pooled runtime model.
pub fn library_from_proto(mut library: wire::Library) -> Result<Library> {
    if !has_valid_header(&library) {
        bail!(
            "invalid Liberty proto header: magic=0x{:016x}",
            library.format_magic
        );
    }
    let strings: Vec<SharedString> = std::mem::take(&mut library.interned_strings)
        .into_iter()
        .map(SharedString::from)
        .collect();
    let axes: Vec<SharedAxis> = std::mem::take(&mut library.lut_axes)
        .into_iter()
        .map(|axis| Arc::new(axis.values))
        .collect();
    let shapes: Vec<Arc<LutShape>> = std::mem::take(&mut library.lut_shapes)
        .into_iter()
        .map(|shape| {
            Ok(Arc::new(LutShape {
                template_id: shape.template_id,
                index_1: resolve_shared_axis(shape.index_1_id, &axes)?,
                index_2: resolve_shared_axis(shape.index_2_id, &axes)?,
                index_3: resolve_shared_axis(shape.index_3_id, &axes)?,
                dimensions: Arc::new(shape.dimensions),
                template_name: resolve_shared_string(shape.template_name_string_id, &strings)?,
            }))
        })
        .collect::<Result<Vec<_>>>()?;

    let table_count: usize = library
        .cells
        .iter()
        .flat_map(|cell| &cell.pins)
        .map(|pin| {
            pin.timing_arcs
                .iter()
                .map(|arc| arc.tables.len())
                .sum::<usize>()
                + pin
                    .internal_power
                    .iter()
                    .map(|group| group.tables.len())
                    .sum::<usize>()
        })
        .sum();
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
    let mut value_ranges = VecDeque::with_capacity(table_count);
    for cell in &mut library.cells {
        for pin in &mut cell.pins {
            for arc in &mut pin.timing_arcs {
                for table in &mut arc.tables {
                    value_ranges
                        .push_back(append_lut_values(&mut value_storage, &mut table.values)?);
                }
            }
            for group in &mut pin.internal_power {
                for table in &mut group.tables {
                    value_ranges
                        .push_back(append_lut_values(&mut value_storage, &mut table.values)?);
                }
            }
        }
    }
    let value_storage = Arc::new(value_storage);

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
            let kind_other = shared_other_string(
                kind == wire::LutTemplateKind::Other,
                template.kind_string_id,
                &strings,
                "LUT template kind",
            )?;
            let variable_1_other = shared_other_string(
                variable_1 == wire::LutVariable::Other,
                template.variable_1_string_id,
                &strings,
                "LUT variable_1",
            )?;
            let variable_2_other = shared_other_string(
                variable_2 == wire::LutVariable::Other,
                template.variable_2_string_id,
                &strings,
                "LUT variable_2",
            )?;
            let variable_3_other = shared_other_string(
                variable_3 == wire::LutVariable::Other,
                template.variable_3_string_id,
                &strings,
                "LUT variable_3",
            )?;
            Ok(LuTableTemplate {
                kind: enum_string(
                    kind,
                    kind_other.as_str(),
                    &[
                        (wire::LutTemplateKind::Unknown, ""),
                        (wire::LutTemplateKind::Timing, "lu_table_template"),
                        (wire::LutTemplateKind::Power, "power_lut_template"),
                    ],
                )
                .to_string(),
                name: template.name,
                variable_1: lut_variable_str(variable_1, variable_1_other.as_str()).to_string(),
                variable_2: lut_variable_str(variable_2, variable_2_other.as_str()).to_string(),
                variable_3: lut_variable_str(variable_3, variable_3_other.as_str()).to_string(),
                index_1: resolve_shared_axis(template.index_1_id, &axes)?
                    .as_ref()
                    .clone(),
                index_2: resolve_shared_axis(template.index_2_id, &axes)?
                    .as_ref()
                    .clone(),
                index_3: resolve_shared_axis(template.index_3_id, &axes)?
                    .as_ref()
                    .clone(),
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
                                    let range = value_ranges.pop_front().ok_or_else(|| {
                                        anyhow::anyhow!("missing pooled timing-table value range")
                                    })?;
                                    timing_table_from_wire(
                                        table,
                                        &shapes,
                                        value_storage.clone(),
                                        range,
                                    )
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
                            let timing_sense_other = shared_other_string(
                                timing_sense == wire::TimingSense::Other,
                                arc.timing_sense_string_id,
                                &strings,
                                "timing sense",
                            )?;
                            let timing_type_other = shared_other_string(
                                timing_type == wire::TimingType::Other,
                                arc.timing_type_string_id,
                                &strings,
                                "timing type",
                            )?;
                            Ok(TimingArc {
                                related_pin: resolve_shared_string(
                                    arc.related_pin_string_id,
                                    &strings,
                                )?,
                                timing_sense: enum_string(
                                    timing_sense,
                                    timing_sense_other.as_str(),
                                    &[
                                        (wire::TimingSense::Unspecified, ""),
                                        (wire::TimingSense::PositiveUnate, "positive_unate"),
                                        (wire::TimingSense::NegativeUnate, "negative_unate"),
                                        (wire::TimingSense::NonUnate, "non_unate"),
                                    ],
                                )
                                .to_string(),
                                timing_type: timing_type_str(
                                    timing_type,
                                    timing_type_other.as_str(),
                                )
                                .to_string(),
                                when: resolve_shared_string(arc.when_string_id, &strings)?,
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
                                    let range = value_ranges.pop_front().ok_or_else(|| {
                                        anyhow::anyhow!("missing pooled power-table value range")
                                    })?;
                                    power_table_from_wire(
                                        table,
                                        &shapes,
                                        value_storage.clone(),
                                        range,
                                    )
                                })
                                .collect::<Result<Vec<_>>>()?;
                            Ok(InternalPower {
                                related_pins: group
                                    .related_pin_string_ids
                                    .into_iter()
                                    .map(|id| resolve_shared_string(id, &strings))
                                    .collect::<Result<Vec<_>>>()?,
                                when: resolve_shared_string(group.when_string_id, &strings)?,
                                related_pg_pin: resolve_shared_string(
                                    group.related_pg_pin_string_id,
                                    &strings,
                                )?,
                                tables,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Pin {
                        direction: pin.direction,
                        function: resolve_shared_string(pin.function_string_id, &strings)?,
                        name: resolve_shared_string(pin.name_string_id, &strings)?,
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

    if !value_ranges.is_empty() {
        bail!("unused pooled LUT-value ranges remain after model conversion");
    }

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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn proto_conversion_roundtrips_with_float32_precision() {
        let shared_axis = vec![0.01, 0.02];
        let library = Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "custom_template".to_string(),
                variable_1: "custom_variable".to_string(),
                index_1: shared_axis.clone(),
                ..Default::default()
            }],
            cells: vec![Cell {
                pins: vec![Pin {
                    function: "A".to_string(),
                    name: "Y".to_string(),
                    timing_arcs: vec![TimingArc::from_raw(
                        "A",
                        "positive_unate",
                        "custom_timing_type",
                        "ENABLE",
                        vec![
                            TimingTable::from_f64(
                                wire::TimingTableKind::CellRise,
                                0,
                                shared_axis.clone(),
                                vec![],
                                vec![],
                                vec![0.1, 0.2],
                                vec![2],
                                "",
                            ),
                            TimingTable::from_f64(
                                wire::TimingTableKind::CellFall,
                                0,
                                shared_axis.clone(),
                                vec![],
                                vec![],
                                vec![0.5, 0.6],
                                vec![2],
                                "",
                            ),
                        ],
                    )],
                    internal_power: vec![InternalPower {
                        related_pins: vec!["A".to_string()],
                        when: "ENABLE".to_string(),
                        related_pg_pin: "VDD".to_string(),
                        tables: vec![PowerTable::from_f64(
                            wire::PowerTransition::Rise,
                            0,
                            shared_axis,
                            vec![],
                            vec![],
                            vec![0.3, 0.4],
                            vec![2],
                            "",
                        )],
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };

        let proto = library_to_proto(library.clone()).unwrap();
        let second_proto = library_to_proto(library).unwrap();

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
        assert_eq!(roundtrip.lu_table_templates[0].kind, "custom_template");
        assert_eq!(
            roundtrip.lu_table_templates[0].variable_1,
            "custom_variable"
        );
        assert_eq!(roundtrip.cells[0].pins[0].function, "A");
        assert_eq!(roundtrip.cells[0].pins[0].name, "Y");
        assert_eq!(
            roundtrip.cells[0].pins[0].timing_arcs[0].timing_type_str(),
            "custom_timing_type"
        );
        assert_eq!(roundtrip.cells[0].pins[0].timing_arcs[0].when, "ENABLE");
        let rise = &roundtrip.cells[0].pins[0].timing_arcs[0].tables[0];
        let fall = &roundtrip.cells[0].pins[0].timing_arcs[0].tables[1];
        let power = &roundtrip.cells[0].pins[0].internal_power[0].tables[0];
        assert!(rise.values.shares_storage_with(&fall.values));
        assert!(rise.values.shares_storage_with(&power.values));
        assert!(Arc::ptr_eq(&rise.shape, &fall.shape));
        assert!(Arc::ptr_eq(&rise.shape, &power.shape));
        assert_eq!(rise.values, vec![f64::from(0.1_f32), f64::from(0.2_f32)]);
        assert_eq!(power.values, vec![f64::from(0.3_f32), f64::from(0.4_f32)]);
    }

    #[test]
    fn proto_conversion_rejects_invalid_header() {
        let error = library_from_proto(wire::Library::default()).unwrap_err();
        assert!(format!("{error:#}").contains("invalid Liberty proto header"));
    }
}
