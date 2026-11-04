// SPDX-License-Identifier: Apache-2.0

//! Typed aggregate values that cross helper calls before the top observes them.

use crate::ast::{BinaryOp, BitsType, Width};
use xlsynth_pir::{BValue, BuilderError, FnBuilder, ir};

/// The DSLX type passed unchanged across producer and updater signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowKind {
    Array,
    Struct,
    Enum,
}

/// Returns the complete value (semantic tag and payload for an enum) or a
/// dynamically selected member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowObservation {
    Whole,
    Select,
}

/// One helper replaces a runtime-selected word and preserves its neighbor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowStep {
    pub op: BinaryOp,
    pub invert_index: bool,
    /// Index into the three scalar values passed by the top.
    pub delta: usize,
}

/// A concrete two-word aggregate built in one helper, transformed in another
/// and returned or inspected in the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateFlow {
    lane: usize,
    width: usize,
    producer: usize,
    updater: usize,
    kind: FlowKind,
    seed_op: BinaryOp,
    steps: Vec<FlowStep>,
    /// Optional later write whose data also depends on the opposite field.
    cross_field_step: Option<usize>,
    observation: FlowObservation,
}

impl AggregateFlow {
    /// Checks operation, dependency and input bounds before rendering or
    /// building the independent scalar reference.
    pub fn new(
        lane: usize,
        width: usize,
        modules: [usize; 2],
        kind: FlowKind,
        seed_op: BinaryOp,
        steps: Vec<FlowStep>,
        observation: FlowObservation,
    ) -> Result<Self, String> {
        let [producer, updater] = modules;
        if width == 0 || producer > updater || !(1..=6).contains(&steps.len()) {
            return Err(
                "aggregate flow requires a positive width, ordered modules and one to six updates"
                    .into(),
            );
        }
        let supported = |op| {
            matches!(
                op,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::And | BinaryOp::Or | BinaryOp::Xor
            )
        };
        if !supported(seed_op)
            || steps
                .iter()
                .any(|step| step.delta >= 3 || !supported(step.op))
        {
            return Err(
                "aggregate flow requires a word combiner and an existing scalar input".into(),
            );
        }
        Ok(Self {
            lane,
            width,
            producer,
            updater,
            kind,
            seed_op,
            steps,
            cross_field_step: None,
            observation,
        })
    }

    pub fn lane(&self) -> usize {
        self.lane
    }

    /// Mixes the preserved field into the delta of a write after the first.
    pub fn with_cross_field_update(mut self, step: usize) -> Result<Self, String> {
        if !(1..self.steps.len()).contains(&step) {
            return Err("cross-field update requires a later step in the flow".into());
        }
        self.cross_field_step = Some(step);
        Ok(self)
    }

    fn word_type(&self) -> BitsType {
        BitsType {
            signed: false,
            width: Width::Concrete(self.width),
        }
    }

    /// A word pair can use either named fields or a reversed tuple payload.
    fn enum_projection(type_name: &str, packet: &str) -> String {
        format!(
            "match {packet} {{
        {type_name}::Front {{ left, right }} => (false, left, right),
        {type_name}::Back((right, left)) => (true, left, right),
    }}"
        )
    }

    /// Emits the actual typed aggregate producer and one updater per step in
    /// their owning modules. Imports are supplied by the sample's module DAG.
    pub fn helper_declarations(&self, module: usize) -> String {
        let word = self.word_type();
        let ty = word.to_dslx();
        let mut text = String::new();
        if module == self.producer {
            let (definition, construct) = match self.kind {
                FlowKind::Array => (format!("pub type FlowPacket = {ty}[2];"), "[left, right]"),
                FlowKind::Struct => (
                    format!("pub struct FlowPacket {{ left: {ty}, right: {ty} }}"),
                    "FlowPacket { right, left }",
                ),
                FlowKind::Enum => (
                    format!(
                        "pub enum FlowPacket {{ Front {{ left: {ty}, right: {ty} }}, Back(({ty}, {ty})), }}"
                    ),
                    "if x as bool { FlowPacket::Back((right, left)) } else { FlowPacket::Front { right, left } }",
                ),
            };
            let left = self.seed_op.to_dslx("computed", "x", word);
            text.push_str(&format!(
                r#"
{definition}

pub fn make_flow(x: {ty}, y: {ty}, computed: {ty}) -> FlowPacket {{
    let left = {left};
    let right = y;
    {construct}
}}
"#
            ));
        }
        if module != self.updater {
            return text;
        }
        if module != self.producer {
            text.push_str(&format!(
                "\ntype FlowPacket = m{}::FlowPacket;\n",
                self.producer
            ));
        }
        for (index, step) in self.steps.iter().enumerate() {
            let index_bit = if step.invert_index {
                "!(delta as bool)"
            } else {
                "delta as bool"
            };
            text.push_str(&format!(
                r#"
pub fn update_flow{index}(packet: FlowPacket, delta: {ty}) -> FlowPacket {{
    let index_bit = {index_bit};
"#
            ));
            let delta = if self.cross_field_step == Some(index) {
                match self.kind {
                    FlowKind::Array => "(delta ^ packet[!index_bit])",
                    FlowKind::Struct => {
                        "(delta ^ (if index_bit { packet.left } else { packet.right }))"
                    }
                    FlowKind::Enum => "(delta ^ (if choose_right { left } else { right }))",
                }
            } else {
                "delta"
            };
            match self.kind {
                FlowKind::Array => {
                    let updated = step.op.to_dslx("packet[index_bit]", delta, word);
                    text.push_str(&format!(
                        "    let updated = {updated};\n    update(packet, index_bit, updated)\n"
                    ));
                }
                FlowKind::Struct => {
                    let updated = step.op.to_dslx("old", delta, word);
                    text.push_str(&format!(
                        r#"    let old = if index_bit {{ packet.right }} else {{ packet.left }};
    let updated = {updated};
    if index_bit {{
        FlowPacket {{ right: updated, ..packet }}
    }} else {{
        FlowPacket {{ left: updated, ..packet }}
    }}
"#
                    ));
                }
                FlowKind::Enum => {
                    let fields = Self::enum_projection("FlowPacket", "packet");
                    let updated = step.op.to_dslx("old", delta, word);
                    text.push_str(&format!(
                        r#"    let (tag, left, right) = {fields};
    let choose_right = tag ^ index_bit;
    let old = if choose_right {{ right }} else {{ left }};
    let updated = {updated};
    if choose_right {{
        FlowPacket::Back((updated, left))
    }} else {{
        FlowPacket::Front {{ left: updated, right }}
    }}
"#
                    ));
                }
            }
            text.push_str("}\n");
        }
        text
    }

    /// Returns the observable DSLX type as spelled in the importing top.
    pub fn output_type(&self) -> String {
        let ty = self.word_type().to_dslx();
        match (self.kind, self.observation) {
            (_, FlowObservation::Select) => ty,
            (FlowKind::Enum, FlowObservation::Whole) => format!("(bool, {ty}, {ty})"),
            (_, FlowObservation::Whole) => format!("m{}::FlowPacket", self.producer),
        }
    }

    /// Chains ordinary aggregate-returning calls and observes the final value.
    pub fn caller_expression(&self, args: [&str; 3]) -> String {
        let word = self.word_type().to_dslx();
        let args = args.map(|arg| format!("({arg} as {word})"));
        let [x, y, computed] = &args;
        let mut body = format!(
            "{{\n    let packet0 = m{}::make_flow({x}, {y}, {computed});\n",
            self.producer
        );
        for (index, step) in self.steps.iter().enumerate() {
            let delta = &args[step.delta];
            body.push_str(&format!(
                "    let packet{} = m{}::update_flow{index}(packet{index}, {delta});\n",
                index + 1,
                self.updater
            ));
        }
        let packet = format!("packet{}", self.steps.len());
        let projection = match (self.kind, self.observation) {
            (FlowKind::Enum, observation) => {
                let fields =
                    Self::enum_projection(&format!("m{}::FlowPacket", self.producer), &packet);
                if observation == FlowObservation::Whole {
                    fields
                } else {
                    format!(
                        "let (tag, left, right) = {fields};\n    if tag ^ ({y} as bool) {{ right }} else {{ left }}"
                    )
                }
            }
            (_, FlowObservation::Whole) => packet,
            (FlowKind::Array, FlowObservation::Select) => format!("{packet}[{y} as bool]"),
            (FlowKind::Struct, FlowObservation::Select) => {
                format!("if {y} as bool {{ {packet}.right }} else {{ {packet}.left }}")
            }
        };
        body.push_str(&format!("    {projection}\n}}"));
        body
    }

    /// Tracks the two scalar words and a logical discriminant. No reference
    /// update uses DSLX's array update, struct packing or enum representation.
    pub fn emit_ir(
        &self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
    ) -> Result<BValue, BuilderError> {
        for arg in args {
            if builder.get_type(arg)? != &ir::Type::Bits(self.width) {
                return Err(BuilderError::InvalidOperation(
                    "aggregate flow input must match its concrete word width".into(),
                ));
            }
        }
        let [x, y, computed] = args;
        let left = self.seed_op.emit_ir(builder, computed, x, false)?;
        let mut fields = [left, y];
        let mut tag = builder.bit_slice(x, 0, 1)?;
        for (number, step) in self.steps.iter().enumerate() {
            let delta = args[step.delta];
            let mut index = builder.bit_slice(delta, 0, 1)?;
            if step.invert_index {
                index = builder.not(index)?;
            }
            if self.kind == FlowKind::Enum {
                index = builder.xor(tag, index)?;
                tag = index;
            }
            // Calculate both possible destinations independently, preserving
            // the value in the other slot regardless of its bit pattern.
            let (left_delta, right_delta) = if self.cross_field_step == Some(number) {
                (
                    builder.xor(delta, fields[1])?,
                    builder.xor(delta, fields[0])?,
                )
            } else {
                (delta, delta)
            };
            let new_left = step.op.emit_ir(builder, fields[0], left_delta, false)?;
            let new_right = step.op.emit_ir(builder, fields[1], right_delta, false)?;
            fields = [
                builder.select(index, &[new_left, fields[0]], None)?,
                builder.select(index, &[fields[1], new_right], None)?,
            ];
        }
        match (self.kind, self.observation) {
            (FlowKind::Array, FlowObservation::Whole) => {
                builder.array(ir::Type::Bits(self.width), &fields)
            }
            (FlowKind::Struct, FlowObservation::Whole) => builder.tuple(&fields),
            (FlowKind::Enum, FlowObservation::Whole) => builder.tuple(&[tag, fields[0], fields[1]]),
            (_, FlowObservation::Select) => {
                let mut selector = builder.bit_slice(y, 0, 1)?;
                if self.kind == FlowKind::Enum {
                    selector = builder.xor(tag, selector)?;
                }
                builder.select(selector, &fields, None)
            }
        }
    }
}
