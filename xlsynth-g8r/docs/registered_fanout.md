# Buffering register outputs

A launch flip-flop may drive many gate inputs, even when combinational
mapping assumed an ideal input source. After register assembly, buffering
can reduce the load on the actual Q output without changing the logic.
Only selected data input pins are reconnected; output ports, aliases, and
other fixed connections stay on the original net.

```text
BEFORE

FF.Q --+--> gate1.A
       +--> gate2.A
       +--> gate3.A
       +--> output port

AFTER

FF.Q --+----------------> output port
       |
       +--> A [BUF] Y --+--> gate1.A
                        +--> gate2.A
                        +--> gate3.A
```

The [buffer selector](../src/netlist/registered_fanout/buffers.rs) chooses
a non-inverting cell with a large supported-output-load to input-capacitance
ratio. The [planner](../src/netlist/registered_fanout/plan.rs) distributes
sink pins across buffers in netlist order. It splits groups that exceed
capacitance or weighted-fanout bounds and adds buffer levels as necessary.
The default limit of 12 children is an empirical delay and area tradeoff.

The proposed netlist is timed with launch clock-to-Q and capture setup
using the same library and load model as the original. A trial must pass
the timing, area, inserted-buffer slew, and timing-table diagnostic checks.
An existing Q slew violation need not disappear. If no acceptable tree
exists, the original netlist is retained.
