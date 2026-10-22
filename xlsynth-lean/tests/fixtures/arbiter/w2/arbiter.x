// SPDX-License-Identifier: Apache-2.0

// Producers A=0, B=1 and C=2 share one backpressured output.
// Each input has one pending slot; arbitration includes new arrivals, and
// unselected requests remain in their slots until a later activation.
#![allow(nonstandard_member_naming)]
const PAYLOAD_WIDTH = u32:2;
type Payload = uN[PAYLOAD_WIDTH];

// The producer considered first; the zero reset encoding selects A.
enum Priority: u2 {
    A = 0,
    B = 1,
    C = 2,
}

struct State {
    priority: Priority,
    pendingA: bool,
    pendingB: bool,
    pendingC: bool,
    payloadA: Payload,
    payloadB: Payload,
    payloadC: Payload,
}

proc Arbiter {
    requestA: chan<Payload> in;
    requestB: chan<Payload> in;
    requestC: chan<Payload> in;
    response: chan<(u2, Payload)> out;

    config(
        requestA: chan<Payload> in,
        requestB: chan<Payload> in,
        requestC: chan<Payload> in,
        response: chan<(u2, Payload)> out,
    ) {
        (requestA, requestB, requestC, response)
    }

    init { zero!<State>() }

    next(state: State) {
        let (tokA, dataA, receivedA) =
            recv_if_non_blocking(join(), requestA, !state.pendingA, state.payloadA);
        let (tokB, dataB, receivedB) =
            recv_if_non_blocking(join(), requestB, !state.pendingB, state.payloadB);
        let (tokC, dataC, receivedC) =
            recv_if_non_blocking(join(), requestC, !state.pendingC, state.payloadC);
        let availableA = state.pendingA || receivedA;
        let availableB = state.pendingB || receivedB;
        let availableC = state.pendingC || receivedC;
        let available = availableA || availableB || availableC;

        let (sender, payload) = match state.priority {
            Priority::A => {
                if availableA {
                    (u2:0, dataA)
                } else if availableB {
                    (u2:1, dataB)
                } else {
                    (u2:2, dataC)
                }
            },
            Priority::B => {
                if availableB {
                    (u2:1, dataB)
                } else if availableC {
                    (u2:2, dataC)
                } else {
                    (u2:0, dataA)
                }
            },
            Priority::C => {
                if availableC {
                    (u2:2, dataC)
                } else if availableA {
                    (u2:0, dataA)
                } else {
                    (u2:1, dataB)
                }
            },
        };

        send_if(join(tokA, tokB, tokC), response, available, (sender, payload));
        let priority = if available {
            match sender {
                u2:0 => Priority::B,
                u2:1 => Priority::C,
                _ => Priority::A,
            }
        } else {
            state.priority
        };
        State {
            priority,
            pendingA: availableA && sender != u2:0,
            pendingB: availableB && sender != u2:1,
            pendingC: availableC && sender != u2:2,
            payloadA: dataA,
            payloadB: dataB,
            payloadC: dataC,
        }
    }
}
