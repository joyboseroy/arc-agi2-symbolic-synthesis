# Conditional object primitives (201-primitive DSL)

## What changed
- `dsl_cond.py` adds 66 conditional per-object primitives named `if_<predicate>_<action>`. Predicates: largest, smallest, border, interior, has hole, filled rectangle, single pixel, above-mean size, taller than wide, unique shape, repeated shape, most common colour, rarest colour, left-right symmetric. Actions: recolor, remove, keep only, fill bounding box, flip. Four pairs that can never change a grid were dropped.
- `ARC_DSL=small|big|full` selects 44, 135 or 201 primitives (default `full`). `run_ablation_big.sh` now pins `ARC_DSL=big`. The 135-primitive results were re-run on 100 tasks after the refactor and are identical.
- Proposer retrained on 50,000 synthetic tasks for the 201 primitives: all true primitives are in its top 24 on 81% of held-out tasks (random: 7.5%), top 16 on 75% (5%).

## Results
First 300 public training tasks, fully solved. Evaluation set is the full 120 tasks.

| Config | Small 44 | Big 135 | Full 201 | Full 201 sec/task |
|---|---|---|---|---|
| All primitives, depth 2 | 21 | 38 | 40 | 8.2 |
| Proposer top 24, depth 3 | | 33 | 29 | 4.8 |
| Proposer top 32, depth 3 | | 36 | 34 | 10.8 |
| Random 24, depth 3 | | 19 | 9 | 2.6 |
| Evaluation set (any config) | 0 | 0 | 0 | |

## Readout
1. The conditional primitives add almost nothing: 38 to 40 solved at depth 2 (+2 tasks, `3618c87e` and `1b2d62fb`). The union of depth-2 and guided depth-3 results is 41 for both the 135 and the 201 sets.
2. Guided depth 3 got slightly worse with the larger set (33 to 29 at top 24, 36 to 34 at top 32). The proposer's top-k recall did not improve (about 81% at top 24 in both cases), while the pool grew by half, so more true primitives fall outside the cut.
3. Random pruning collapsed (19 to 9), so the proposer's advantage over random pruning widened (29 vs 9), but that is dilution of the random baseline, not a gain.
4. Evaluation set stays at 0 of 120 for every configuration.

## What this says
Adding hand-written primitives has hit diminishing returns. The 44 to 135 step gave +17 tasks on these 300, the 135 to 201 step gave +2. Evaluation tasks are not failing for lack of one more predicate. They need rules that are specific to each task (which property decides what happens to which object, and how), and a fixed menu of predicate-action pairs cannot express that.

## Caveats
- 300 training tasks, so differences of a few tasks are noise. The +17 from the first expansion and the guided-versus-random gap are the clear effects.
- The proposer is trained on synthetic programs from the same DSL and has not been shown to generalise beyond it.
- Development used only the public training set. The evaluation set was run once per configuration.

## Suggested next step
Induce the rule per task instead of enumerating it: for each training pair, match input objects to output objects, record which properties (colour, size, shape, position, count rank) change and which property separates changed from unchanged objects, then fit the smallest consistent rule (MDL) and apply it to the test grid. This uses the same exact-verification and shortest-program selection as the rest of the pipeline, and it gives the paper a concrete mechanism on the "explanation" side of explanation-guided synthesis rather than more primitives.
