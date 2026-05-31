# xlsynth-vastly Fuzz Targets

## `diff_refsim_4value`

Generates bounded four-state Verilog expressions and environments, evaluates
them with Vastly, and compares accepted expressions against an `iverilog`
reference simulation. External compiler and simulator processes use a bounded
polling watchdog so rejected expressions and oracle timeouts do not stall the
campaign.

Failures expose four-state expression parsing or evaluation drift relative to
`iverilog`, excluding oracle-side rejection and timeout cases.

## `xls_ir_codegen_semantics`

Generates bounded standard PIR directly with `xlsynth_pir::ir_random`, creates
an initial typed argument tuple with the shared random value generator, and
optionally expands it through autocov. The emitted PIR is loaded into libxls
for interpretation and Verilog/SystemVerilog codegen, then simulated through
Vastly and compared against the libxls result.

Product-pair operations are excluded from this target while its downstream
interpreter/codegen surface is characterized separately. Failures expose
PIR-to-libxls loading incompatibility, codegen semantic drift, or Vastly
simulation disagreement.
