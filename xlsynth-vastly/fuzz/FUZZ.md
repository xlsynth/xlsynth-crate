# xlsynth-vastly Fuzz Targets

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
