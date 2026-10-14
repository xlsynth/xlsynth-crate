# Priority predicate regression inputs

These cases use the IR snippet QoR corpus loader and PriorityResult
expectations. The local profile disables FRAIG, cut rewriting, and reassociation
and retains normal gate folds/hash and DCE. The fixtures in `ir_snippet_qor`
keep their original mapping profile.

Origin comments distinguish published k3 inputs from synthetic controls. Tests
cover PIR-only optimization and the full XLS/aug-opt pipeline at one and three
rounds, source equivalence, rewrite counts, and raw AND/graph-LE limits.
