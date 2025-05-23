# Transformation Menu

This directory contains small graph-rewriting operations used by the `xlsynth-g8r` crate during stochastic synthesis.
The mutations are typically applied in a Markov chain Monte Carlo (MCMC) style search where each proposed graph edit is kept only if it survives a functional screen.
Below is a short catalogue of the available transforms.

| slug | sketch of the edit | why it helps in an MCMC search | wild survival rate (256 vector screen) |
|------|-------------------|--------------------------------|----------------------------------------|
| `double-negate` | push or pull a double negation through an `AND` edge | lets the solver explore logically equivalent polarity placements without changing behaviour | ~100% |
| `duplicate` | create a copy of a gate with the same fan‑ins | introduces extra structure that later moves can rewire or delete | ~100% |
| `redundant-and` | wrap an operand with `AND(x,x)` or collapse such a gate | adds or removes trivial gates to vary structure | ~100% |
| `swap-operands` | exchange the left/right inputs of an `AND` | explores symmetrical wiring orders (semantics unchanged) | ~100% |
| `toggle-output` | flip the negation flag on an output bit | probes behaviour changes to escape local optima | ≈0% |
| `true-and` | wrap an operand with `AND(x,true)` or collapse it | introduces a no‑op gate that may unlock later rewrites | ~100% |

The survival rates above are rough estimates assuming the candidate design is screened against 256 random input vectors. Edits that preserve semantics tend to pass every time, whereas edits like `toggle-output` almost always fail the screen.
