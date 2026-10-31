# Extended DSL (135 primitives) and proposer-guided search

## What changed
- `dsl_ext.py` adds 91 primitives to the original 44 (135 total): colour selectors by rank, object recolor and selection, rigid object falls, shifts and rolls, line drawing and connection, boolean ops on halves, grid-cell selection, tiling variants, summaries, morphology, symmetry completion, sorting, per-object flips and bounding-box outlines.
- Fills use an "unused" colour and the solver learns a global colour map afterwards, so no colour is hard-coded.
- `ARC_DSL=small` reproduces the original 44-primitive results exactly (checked: 21 of the first 300 training tasks, as before).
- Proposer retrained on 50,000 synthetic tasks for the 135-primitive DSL: all true primitives are in its top 24 on 81% of held-out tasks (random: 10%), top 16 on 75% (6%).

## Results
First 400 public training tasks, fully solved (evaluation set is the full 120 tasks).

| Config | Training solved (of 400) | Sec/task | Evaluation solved (of 120) |
|---|---|---|---|
| Old DSL (44), all primitives, depth 2 | 32 | 0.3 | 0 |
| New DSL (135), all primitives, depth 2 | 53 | 4.7 | 0 |
| New DSL, proposer top 24, depth 3 | 47 | 4.6 | 0 |
| New DSL, proposer top 32, depth 3 | 52 | 10.8 | 0 |
| New DSL, random 24, depth 3 | 24 | 2.6 | 0 |

## Readout
1. Growing the DSL is what moved the score: 32 to 53 solved on the same 400 tasks (+21, about +5 points), with one task lost.
2. The proposer is clearly informative: top 24 solves 47 against 24 for random pruning at the same size.
3. At equal compute the proposer does not yet beat plain depth-2 search over all 135 primitives (47 at 4.6 s versus 53 at 4.7 s). Full depth 3 over all 135 is about 2.5 million programs per task, which is the regime pruning is meant for, but it is too slow to run here as a reference.
4. Guided depth 3 reaches different tasks from depth 2: top 32 solves 4 tasks that unrestricted depth 2 misses and misses 5 that it finds. Taking either result per task gives 57 of 400.
5. Evaluation set: still 0 of 120 for every configuration. Simple compositions of whole-grid and object primitives do not reach ARC-AGI-2 evaluation tasks.

## Caveats for the paper
- The proposer is trained on synthetic programs from the same DSL and uses per-primitive agreement features, so it has not been shown to generalise beyond the DSL.
- The training comparison uses 400 tasks, so differences of a few tasks (52 vs 53, 47 vs 53) are within noise. The DSL growth (+21) and the guided-versus-random gap (+23) are the clear effects.
- Development used only the public training set. The evaluation set was run once per configuration and not used for any choice.

## Suggested next steps
1. A reference run of unpruned depth 3 on a small sample (about 50 tasks, several hours) to measure how much of full depth-3 the proposer recovers in the 135-primitive regime.
2. Replace depth-3 enumeration with best-first search ordered by proposer probability, so deeper programs are reached without a fixed top-k cut.
3. Add conditional and per-object rule primitives (apply an op to objects selected by a property). Evaluation tasks mostly need these, and none of the current 135 primitives are conditional.
