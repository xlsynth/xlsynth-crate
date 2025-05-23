# Transformation Menu

This directory contains small graph-rewriting operations used by the `xlsynth-g8r` crate during stochastic synthesis. The mutations are typically applied in a Markov chain Monte Carlo (MCMC) style search where each proposed graph edit is kept only if it survives a functional screen. Below is a short catalogue of the available transforms.

| slug | edit sketch | mcmc benefit |
|------|-------------|--------------|
| `double-negate` | move negation across `AND` | explore polarity options |
| `duplicate` | copy gate with same inputs | add nodes for rewiring |
| `redundant-and` | add/remove `AND(x,x)` | vary trivial gates |
| `swap-operands` | swap `AND` inputs | explore symmetrical wiring |
| `toggle-output` | invert output negation | test new behaviors |
| `true-and` | add/remove `AND(x,true)` | insert no-op to unlock rewrites |
