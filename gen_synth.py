"""Generate synthetic tasks by running random DSL programs on real ARC *training inputs*.

Only input grids from the public training set are used (no outputs, nothing from
the evaluation set). Each synthetic task has 3 pairs. Label = multi-hot of the
primitives in the program, plus one extra slot for 'needs a recolor map'.

Usage: python3 -I gen_synth.py <training_dir> <out_npz> <n_tasks> [seed] [exclude_ids.json]
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsl  # noqa: E402
import features  # noqa: E402

NAMES = features.NAMES
GRIDS = []


def load_inputs(task_dir, exclude=()):
    out = []
    for f in sorted(os.listdir(task_dir)):
        if f.endswith(".json") and f[:-5] not in exclude:
            t = json.load(open(os.path.join(task_dir, f)))
            for p in t["train"]:
                out.append(np.array(p["input"]))
    return out


def init(task_dir, exclude):
    global GRIDS
    GRIDS = load_inputs(task_dir, set(exclude))


def one(seed):
    rng = np.random.default_rng(seed)
    for _ in range(40):  # retry until a non-degenerate program is found
        depth = rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20])
        prog = [NAMES[k] for k in rng.integers(0, len(NAMES), size=depth)]
        recolor = rng.random() < 0.15
        perm = rng.permutation(10) if recolor else None
        pairs = []
        for _ in range(3):
            g = GRIDS[rng.integers(0, len(GRIDS))]
            r = g
            for name in prog:
                r = dsl.apply(name, r)
                if r is None:
                    break
            if r is None:
                break
            if recolor:
                r = perm[r]
            if r.shape == g.shape and np.array_equal(r, g):
                break
            pairs.append((g, r))
        if len(pairs) < 3:
            continue
        y = np.zeros(len(NAMES) + 1, dtype=np.float32)
        for name in prog:
            y[NAMES.index(name)] = 1.0
        y[-1] = float(recolor)
        return features.task_features(pairs), y
    return None


def main():
    task_dir, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    exclude = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else []
    with Pool(os.cpu_count() or 2, initializer=init, initargs=(task_dir, exclude)) as p:
        res = p.map(one, range(seed0, seed0 + n), chunksize=50)
    res = [r for r in res if r is not None]
    X = np.stack([r[0] for r in res])
    Y = np.stack([r[1] for r in res])
    np.savez_compressed(out, X=X, Y=Y)
    print("saved", X.shape, Y.shape, "label density", Y[:, :-1].sum(1).mean())


if __name__ == "__main__":
    main()
