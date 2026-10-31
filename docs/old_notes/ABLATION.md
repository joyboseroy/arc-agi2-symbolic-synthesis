# Neural hypothesis proposer: ablation results

## What was built
- `features.py`: 117 features per task, averaged over training pairs (shape ratios, colour changes, symmetry, containment, object counts, and for each of the 44 primitives whether it gives the right output shape and how many cells already agree).
- `gen_synth.py`: 40,000 synthetic tasks made by running random depth 1 to 3 DSL programs on real ARC training input grids. No outputs and nothing from the evaluation set are used. Label = which primitives the program used.
- `train_proposer.py`: multi-label MLP (256, 128) from scikit-learn.
- `run_guided.py`: the search is restricted to the proposer's top-k primitives, which allows going to depth 3 at modest cost.

## Proposer quality (4,000 held-out synthetic tasks)
Share of tasks where every true primitive is inside the top-k predictions (random in brackets):
top 6: 72% (8%), top 10: 82% (14%), top 16: 90% (25%), top 24: 96% (41%).

## Search ablation on real ARC-AGI-2 public training tasks (fully solved, 1,000 tasks)
| Config | Solved | Sec/task |
|---|---|---|
| All 44 primitives, depth 2 (baseline) | 67 | 0.28 |
| Proposer top 16, depth 2 | 56 | 0.06 |
| Proposer top 16, depth 3 | 60 | 0.70 |
| Random 16, depth 3 | 39 | 0.48 |
| Proposer top 24, depth 3 | 66 | 1.94 |

Reference with no pruning at depth 3 (too slow for the full set, so first 300 tasks only):
| Config (first 300 tasks) | Solved | Sec/task |
|---|---|---|
| All 44, depth 2 | 21 | 0.28 |
| Proposer top 16, depth 3 | 19 | 0.70 |
| Proposer top 24, depth 3 | 22 | 1.94 |
| All 44, depth 3 | 24 | 9.29 |

## Readout
1. The proposer is doing real work: at the same budget it solves 60 tasks versus 39 for random pruning.
2. Guided top 24 at depth 3 recovers 22 of the 24 solves that full depth 3 finds, at about one fifth of the time. Top 16 recovers 19 at about one thirteenth.
3. Pruning alone does not beat the unrestricted depth-2 baseline on accuracy (66 vs 67). The gain is speed that makes deeper search affordable. Accuracy gains need a larger DSL, where unrestricted search stops being feasible.
4. Evaluation set: 0 of 120 for every configuration. The gap to ARC-AGI-2 evaluation difficulty is a DSL expressiveness gap, not a search gap.

## Caveats for the paper
- The proposer is trained on programs from the same DSL, so it learns DSL structure and shortcut features (per-primitive agreement). It has not been shown to generalise to transformations outside the DSL.
- Differences of a few tasks (66 vs 67, 22 vs 24) are within noise for these sample sizes. Only the guided vs random gap is clearly large.
- Only the 1,000 public training tasks were used for development. No tuning on the evaluation set.

## Suggested next step
Grow the DSL with object-level primitives (per-object recolor, move, copy, conditional rules) to about 150 primitives. Full depth-3 search then costs roughly 40x more per task, which is where proposer pruning should move from a speedup to an accuracy gain. That is the experiment the paper's central claim needs.
