// SPDX-License-Identifier: Apache-2.0

enum WidgetMode : u2 {
    Idle = 0,
    Frob = 1,
    Quux = 2,
}

enum SignedWidgetMode : s2 {
    Negative = s2:-1,
    Neutral = s2:0,
    Positive = s2:1,
}

type FrobNibbles = u4[3];

struct WidgetTuning {
    frob_bias: u4,
    wobble_trim: s4,
}

struct Widget {
    widget_id: u8,
    mode: WidgetMode,
    frobs: FrobNibbles,
    tuning: WidgetTuning,
    wobble: s4,
    signed_mode: SignedWidgetMode,
}

struct WidgetOutcome {
    next_widget_id: u8,
    selected_frob: u4,
    mode: WidgetMode,
    adjusted_wobble: s4,
    signed_mode: SignedWidgetMode,
}

pub fn frob_widget(widget: Widget, garnish: FrobNibbles) -> WidgetOutcome {
    WidgetOutcome {
        next_widget_id: widget.widget_id + u8:1,
        selected_frob: widget.frobs[u32:1] + garnish[u32:2] + widget.tuning.frob_bias,
        mode: widget.mode,
        adjusted_wobble: widget.wobble - widget.tuning.wobble_trim,
        signed_mode: widget.signed_mode,
    }
}
