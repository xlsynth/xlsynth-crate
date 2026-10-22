// SPDX-License-Identifier: Apache-2.0

// Metadata fixture. The paired block is hand-authored behavioral authority;
// this file supplies concrete nominal types, not compiler provenance.
type Word = u2;
enum Phase: u2 { Idle = 0, Active = 2 }
struct Flags { armed: bool, queued: bool }
struct Machine { phase: Phase, flags: Flags, sample: Word }

fn selected_sample(machine: Machine, incoming: Word) -> Word {
    let phase_is_active = machine.phase == Phase::Active;
    if phase_is_active { incoming } else { machine.sample }
}
