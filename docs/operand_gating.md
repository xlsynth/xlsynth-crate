# Preserving operand gating with Yosys and OpenROAD

`gate!(enable, value)` can stop switching before an occasionally enabled
multiplier. Masking only the result gives the same observable values but lets
the multiplier keep switching. **Liberty `use_for_size_only : true` is a cell
selection restriction, not an instance-retention guarantee.** The xlsynth
optimizer's size-only policy does not automatically constrain downstream tools.

Use `gate_format` to emit explicit isolation cells with an enable pin. The
example below produces eight scalar `ISO` instances with function `A & EN`.
For physical implementation, substitute characterized cells with matching
Liberty/LEF views. A buffer without an enable pin does not implement `gate!`.

## A small, reproducible toggle comparison

These optional local commands need Python 3, Yosys, and `xlsynth-driver` with
`dslx-stitch-pipeline --gate_format` support on `PATH`. They add no CI dependency.
Run the blocks in order in a new directory. The toy library supplies Boolean
functions only; it is not a timing or physical library.

```sh
mkdir operand-gating-demo
cd operand-gating-demo
cat > gated.x <<'X'
fn mul_cycle0(en: bool, a: u4, b: u4) -> u8 {
    let ga = gate!(en, a);
    let gb = gate!(en, b);
    if en { (ga as u8) * (gb as u8) } else { u8:0 }
}
X
python3 - <<'PY'
from pathlib import Path
import random

Path("baseline.x").write_text(Path("gated.x").read_text()
    .replace("gate!(en, a)", "a").replace("gate!(en, b)", "b"))
Path("empty.toml").write_text("[toolchain]\n")
cells = [("ISO", "A & EN", ["A", "EN"]), ("AND2", "A & B", ["A", "B"]),
         ("XOR2", "A ^ B", ["A", "B"]), ("INV", "!A", ["A"])]
lib = "library (toy) {\n"
for name, function, pins in cells:
    lib += f"cell ({name}) {{ area : 1;\n"
    if name == "ISO":
        lib += "use_for_size_only : true;\n"
    for pin in pins:
        lib += f"pin ({pin}) {{ direction : input; }}\n"
    lib += f'pin (Y) {{ direction : output; function : "{function}"; }}\n}}\n'
Path("cells.lib").write_text(lib + "}\n")
rng = random.Random(0)
samples = [(int(i % 64 < 16), rng.randrange(16), rng.randrange(16)) for i in range(1024)]
exhaustive = [(en, a, b) for en in range(2) for a in range(16) for b in range(16)]
for name, values in [("stimulus", samples), ("exhaustive", exhaustive)]:
    Path(name + ".irvals").write_text("".join(
        "(bits[1]:%d, bits[4]:%d, bits[4]:%d)\n" % v for v in values))
PY
xlsynth-driver lib2proto cells.lib --output cells.textproto
```

Generate both designs, synthesize their **combinational stage** (excluding the
stitched wrapper's registers), and evaluate the same inputs:

```sh
for design in baseline gated; do
  xlsynth-driver --toolchain empty.toml dslx-stitch-pipeline \
    --dslx_input_file="$design.x" --stages=mul_cycle0 --output_module_name=mul \
    --use_system_verilog=false \
    --gate_format='(* keep = 1 *) ISO iso_{output}[{width}-1:0](.EN({condition}), .A({input}), .Y({output}))' \
    > "$design.v"
  cat > "$design.ys" <<YOSYS
read_liberty -lib cells.lib
read_verilog $design.v
synth -top mul_cycle0 -flatten -noabc
abc -g AND,XOR
chtype -set AND2 t:\$_AND_
chtype -set XOR2 t:\$_XOR_
chtype -set INV t:\$_NOT_
clean -purge
check -assert
write_verilog -noattr $design.gv
YOSYS
  yosys -s "$design.ys" > "$design.yosys.log"
  xlsynth-driver gv-eval --netlist "$design.gv" --liberty_proto cells.textproto \
    --input-irvals stimulus.irvals --toggle-output-json "$design.activity.json" > "$design.outputs"
  xlsynth-driver gv-eval --netlist "$design.gv" --liberty_proto cells.textproto \
    --input-irvals exhaustive.irvals > "$design.exhaustive"
done
cmp baseline.outputs gated.outputs
python3 - <<'PY'
from pathlib import Path
import json

expected = [a * b if en else 0 for en in range(2) for a in range(16) for b in range(16)]
for design in ["baseline", "gated"]:
    values = [int(line.split(":", 1)[1].replace("_", ""), 0)
              for line in Path(design + ".exhaustive").read_text().splitlines()]
    assert values == expected, design
    print(design, json.loads(Path(design + ".activity.json").read_text())["aggregate"])
print("All 512 input combinations match en ? a*b : 0.")
PY
```

The stimulus has 1,024 samples with uniformly drawn independent 4-bit operands
from Python `Random(0)`. Enable is high for 16 then low for 48 samples per period
(25% duty); operands keep changing while disabled. The first sample establishes
the baseline, leaving 1,023 transitions. Both designs mask the result.

| Measured toggles | Baseline | Gated |
| --- | ---: | ---: |
| All cell output pins | 36,767 | 11,888 |
| All cell input pins | 70,608 | 23,576 |
| Output port | 880 | 880 |

Cell totals include isolation and result-mask logic; input counts count each
fanout pin separately. These are settled Boolean pin transitions, **not power**:
no glitches, delays, capacitance, intra-cell activity, or clock tree are modeled.
Duty and ordering matter. With enable always high, isolation adds switching;
it is not automatically an improvement. The counts above apply to the tested
tool versions and mapping, rather than every equivalent implementation.

## Retaining the isolation boundary

The synthesis recipe deliberately uses `read_liberty -lib` plus instance
`(* keep = 1 *)`. An equivalent Yosys-side constraint, after reading the design,
is `setattr -set keep 1 t:ISO`. Do not load functional cell models before
flattening: in a separate probe, flattening absorbed the isolation cells even
with `keep` on the original instances. Black-box import preserved live cells;
`keep` also preserved an otherwise unused instance. Size-only and ordinary
library markings behaved identically. ABC even mapped fresh logic into a
scratch ASAP7 library whose candidates were all marked size-only.

For real library mapping, exclude restricted choices explicitly, e.g.
`abc -liberty real.lib -dont_use ISO` (repeat `-dont_use` for other names), while
retaining full Liberty for linking/timing. The toy example instead maps only
generic AND/XOR/NOT gates, leaving `ISO` outside ABC's logic network.

After OpenROAD `read_verilog` / `link_design`, before resizer optimization,
apply instance constraints explicitly; do not assume Yosys `keep` transfers:

```tcl
set isolation [get_cells -hierarchical *iso_*]
if {[llength $isolation] != 8} { error "Expected eight isolation instances" }
set_dont_touch $isolation
report_dont_touch
```

Adjust selection/count for the linked design. **This Tcl is source-verified,
not runtime-tested here.** `set_dont_touch` also blocks resizing; `set_dont_use`
restricts cell selection, not instance removal. No verified automatic
“retain topology but allow resizing” guarantee was established. That requires
flow-specific sizing controls, with the enable/data/load connections checked
after each optimization stage. Counts or output equivalence alone cannot prove
that operand isolation still drives the intended logic.

Tested: Yosys 0.52 (`fee39a3284`), packaged ABC, and xlsynth-driver at
`36fa93b451` with XLS DSO 0.59.0. The eight emitted isolation cells survived
synthesis. OpenROAD execution was blocked by missing dependencies; no physical
flow validation is claimed. Source inspection at OpenROAD `eb14e40b7b` / OpenSTA
`80fbbb2945` found no modeled `use_for_size_only` restriction. See the
[OpenSTA cell-attribute reader](https://github.com/The-OpenROAD-Project/OpenSTA/blob/80fbbb2945fee2702a84e711a95a0a3bb7050f22/liberty/LibertyReader.cc#L1803)
and [OpenROAD sizing guard](https://github.com/The-OpenROAD-Project/OpenROAD/blob/eb14e40b7b9527a642a07548ed1dc39c4c611cb3/src/rsz/src/Resizer.cc#L2686).
