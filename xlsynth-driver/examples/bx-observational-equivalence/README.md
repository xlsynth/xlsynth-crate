# Observational equivalence of clocked state machines

Two machines are observationally equivalent when every shared input trace produces the same output trace from their specified initial states. In each BX invariant below, one register holds each implementation's state. On each cycle the harness passes the same fresh input to both DSLX step functions, compares their outputs, and writes each returned next state back to its own register. The machines may use different state widths and encodings. [BX harness](equivalence.bx)

The invariant block itself supplies the circuit to clock repeatedly. The implicit loop for `widened_accumulator` can be written in pseudocode.

```text
Create the narrow and wide registers once.
Clock an asserted reset edge to load (0, 0).
Repeat for each cycle:
    Choose a fresh shared input i and reset level.
    (a, next_narrow) = f(i, narrow)
    (b, next_wide)   = g(i, wide)
    Check a == b.
    Clock the edge, loading (0, 0) on reset or (next_narrow, next_wide) otherwise.
```

These examples import pure DSLX functions, so the BX harness owns their state registers. When both implementations are already clocked BX blocks, each child owns its registers and the harness needs only two instances and one comparison.

```text
#[invariant]
block equivalent(clk, rst, i: u8) -> bool {
    let a = reference(i);
    let b = implementation(i);
    a == b
}
```

Each `let` creates one persistent child instance. The runner clocks both children together. The trailing Boolean is the observation, and induction proves it holds for arbitrarily many repetitions.

With a Bitwuzla-enabled `xlsynth-driver` on your path, run these commands from this directory. The DSLX imports resolve beside the BX file.

```shell
xlsynth-driver bx-test equivalence.bx --seed 42 --quickcheck-trials 8 --invariant-cycles 16
xlsynth-driver bx-prove-invariant equivalence.bx --mode=bmc --max-depth 16
xlsynth-driver bx-prove-invariant equivalence.bx --engine=bitwuzla --max-depth 3
```

The first command samples eight random traces per property. The second checks every legal trace for the first 16 samples after the initial reset edge. The third proves all three properties for every cycle, including traces that assert reset again.

| BX invariant | Smallest closing depth | What the outputs reveal |
| --- | --- | --- |
| `widened_accumulator` | 1 | Only the low eight bits of the widened sum are visible. |
| `latched_detector` | 3 | The output distinguishes the latched state. Idle and armed both report zero. |
| `parity` | 1 with `lhs_is_live` | The parity bit distinguishes the two reachable states. |

## Widening the accumulator

Both machines reset their counter to zero and increment it by one on every edge with reset deasserted. The first adds the input to an eight-bit counter. The second adds it to a sixteen-bit counter and returns the low eight bits. Equal outputs for one shared input imply equal low counter bits. Incrementing both counters preserves that equality, including when the narrow counter wraps. [DSLX step functions](add_vs_widening.x)

To see a reachable failure, change `let sum = i_wide + s;` to `let sum = i_wide + s + u16:1;` in a copy of the example and rerun the induction command. The replay identifies `widened_accumulator` at cycle 0. With input zero, the two outputs are zero and one. This is the mismatch in the original offset-widening demo. [Offset-widening source](https://github.com/xlsynth/xlsynth-kind2/blob/6693bd4756de4fab6a4947e8bab9f157b5dfb245/examples/demos/observational/add_vs_offset_widening.x)

## Binary versus one-hot detection

The detector latches after two consecutive ones and clears on zero. Its binary encoding resets to `0` and its one-hot encoding resets to `0b0001`, both meaning idle. Try `--max-depth 2`. BX reports the detector as inconclusive even though its initialized prefix passes. For example, the induction step may start with binary state `3` and one-hot idle. Both report zero. The first one takes them to idle and armed, which still both report zero. The second one takes them to armed and latched, whose outputs differ. At depth 3 the output property closes without a helper. [DSLX step functions](binary_vs_one_hot_latched_detector.x)

## Removing an unreachable parity state

The left machine has even, odd, and dead states. The right machine keeps a single parity bit. Both start even. Remove only `#[lemma]` from the harness and rerun the proof. Output equality stays open at every finite induction depth. An arbitrary induction state can pair dead with even and idle for as long as needed. A toggle then makes the right machine odd while the left stays dead. [DSLX step functions](unreachable_state_elimination.x)

Restoring `#[lemma] let lhs_is_live = lhs < u2:2;` tells BX to prove that the left machine is always even or odd. Reset establishes the helper and either input preserves it. Once BX has proved the helper, the two output bits agree inductively at depth 1.

The three DSLX modules reproduce the original `xlsynth-kind2` examples at commit `6693bd4` with SPDX license headers added. The BX registers use the initial values from that repository's demo registry. [Original examples and initial states](https://github.com/xlsynth/xlsynth-kind2/blob/6693bd4756de4fab6a4947e8bab9f157b5dfb245/examples/demos/observational/mod.rs)
