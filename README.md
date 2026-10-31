# Symbolic program synthesis for ARC-AGI-2: DSL coverage, search guidance and hypothesis construction

Code, saved results and paper for a controlled study of a small, fully symbolic ARC-AGI-2 solver (ARC Prize 2026, Paper Track).
The solver searches short programs in a hand-written DSL, keeps only programs that reproduce every training pair, and picks the shortest.
It uses no language model, no pretrained network, no test-time training and no internet. It runs on CPU.

This is a lightweight baseline built to study where such a system fails, not a competitive solver.

## Results

| Set | Result |
|---|---|
| Kaggle hidden test (240 tasks), frozen combined solver | score 1.67 (4 of 240 if every task counts equally) |
| 120 public evaluation tasks, combined solver | 1 solved |
| 120 public evaluation tasks, 13 flat-DSL settings (44, 135, 201 primitives) | 0 solved |
| 120 public evaluation tasks, object-relational and object-tree solvers | 0 solved |
| 120 public evaluation tasks, rule-structure (cell-tree) learner | 1 solved (the same task in every variant) |

Experiments on separate public training tasks (DSL size, learned pruning, choice rule, depth 3 and 4 diagnostics) are in the supplement.
They are diagnostics, not performance claims. The paper explains which design choices were fixed in advance and which were exploratory.

Scored Kaggle notebook (Version 2): https://www.kaggle.com/code/joyboseroy/notebookf09093f843?scriptVersionId=362153283
Code dataset used by the notebook: https://www.kaggle.com/datasets/joyboseroy/arc-bundle

## Paper

- `paper/paper.pdf` (main paper, 11 pages) and `paper/paper.tex`
- `paper/supplement.pdf` and `paper/supplement.tex`
- `paper/refs.bib`

## Setup

Python 3.13 was used (earlier 3.x versions should work). Versions used: numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1.

    pip install -r requirements.txt

The ARC-AGI-2 tasks are not included. Get them from https://github.com/arcprize/ARC-AGI-2 (Apache-2.0) and point `ARC_DATA` at the `data` folder:

    git clone https://github.com/arcprize/ARC-AGI-2 ../ARC-AGI-2
    export ARC_DATA=../ARC-AGI-2/data

All commands run from the repository root. Every script reads only the training pairs of a task; test outputs are used only to score.

## Quick start

Flat DSL search (44-primitive tier, depth 2), writes a Kaggle-format `submission.json` and a summary:

    ARC_DSL=small python3 -I run_eval.py $ARC_DATA/evaluation out_demo 10

Combined solver (flat DSL, then object-relational, then cell tree) on a list of task ids:

    ARC_DSL=full python3 -I combined.py $ARC_DATA/evaluation out_combined.json all

`ARC_DSL` selects the primitive tier: `small` (44), `big` (135) or `full` (201).

Kaggle submission file for a challenges JSON (what the notebook runs):

    python3 -I kaggle_run.py        # on Kaggle it finds the challenges file under /kaggle/input
    CHALLENGES=path/to/challenges.json OUT=submission.json python3 -I kaggle_run.py

## What is where

| Path | Contents |
|---|---|
| `dsl.py`, `dsl_ext.py`, `dsl_cond.py` | The three primitive tiers (44, 135, 201). |
| `solver.py`, `depthsearch.py` | Depth-limited exact search, shortest-first choice, merged-state search. |
| `gen_synth.py`, `train_proposer.py`, `run_guided.py`, `run_v2.py` | Synthetic task generator, primitive proposer, guided and random pruning runs. |
| `orel*.py`, `run_orel*.py` | Object-relational solvers (`orel5` is the last variant). |
| `celltree*.py`, `run_celltree*.py` | Cell-tree rule-structure learner and its extensions (`celltree2`, `celltree4`, `celltree5`). |
| `objtree.py`, `objtree2.py` | Object decision trees. |
| `combined.py` | The solver submitted to Kaggle: flat DSL, `orel5`, `celltree2`. |
| `combined_ext.py`, `run_ext.py` | Extended union with three added fallback families. Not submitted. |
| `run_depth4.py` | Depth-4 diagnostic on 100 development tasks. |
| `*_frozen.py` | Frozen copies of solvers as they were when they were run on the evaluation set. |
| `dev700.json`, `subset300.json`, `exclude_ids.json` | The 700 development tasks, the 300-task training sample (fixed seed) and the tasks excluded for the disjoint proposers. |
| `v2/` | Saved per-task results for the experiments in the paper. |
| `models/` | Trained proposers (`.pkl`, scikit-learn). Training sets (`synth*.npz`) are not included; regenerate them with `gen_synth.py`. |
| `kaggle/submitted/` | Exactly the bundle used by the scored notebook. |
| `kaggle/extended/` | Bundle for the extended solver (not submitted). |
| `docs/old_notes/` | Early working notes. The paper supersedes them. |

The `.pkl` files are pickles, so load them only from this repository.

## Evaluation discipline

The solvers were developed on training tasks outside the 300-task sample, frozen, and run once on the sample and once on the 120 public evaluation tasks.
I did not inspect individual public evaluation tasks during development. The hidden Kaggle test set was used once, for the submission above.
The paper lists the exploratory choices and the limits of the evidence.

## Licence

The code is released under MIT-0 (see `LICENSE`), as the competition requires a permissive licence.
The ARC-AGI-2 tasks are Apache-2.0 and are not redistributed here.

## Citation

Joy Bose. "DSL Coverage, Search Guidance and Hypothesis Construction in Symbolic Program Synthesis for ARC-AGI-2." ARC Prize 2026 Paper Track submission, 2026.
