// SPDX-License-Identifier: Apache-2.0

enum GizmoMode : u2 {
    Idle = 0,
    Frob = 1,
    Quux = 2,
}

enum SignedGizmoMode : s2 {
    Negative = s2:-1,
    Neutral = s2:0,
    Positive = s2:1,
}

type FrobNibbles = u4[3];

struct GizmoTuning {
    frob_bias: u4,
    wobble_trim: s4,
}

struct Gizmo {
    gizmo_id: u8,
    mode: GizmoMode,
    frobs: FrobNibbles,
    tuning: GizmoTuning,
    wobble: s4,
    signed_mode: SignedGizmoMode,
}

struct GizmoOutcome {
    next_gizmo_id: u8,
    selected_frob: u4,
    mode: GizmoMode,
    adjusted_wobble: s4,
    signed_mode: SignedGizmoMode,
}

pub fn frob_gizmo(gizmo: Gizmo, garnish: FrobNibbles) -> GizmoOutcome {
    GizmoOutcome {
        next_gizmo_id: gizmo.gizmo_id + u8:1,
        selected_frob: gizmo.frobs[u32:1] + garnish[u32:2] + gizmo.tuning.frob_bias,
        mode: gizmo.mode,
        adjusted_wobble: gizmo.wobble - gizmo.tuning.wobble_trim,
        signed_mode: gizmo.signed_mode,
    }
}
