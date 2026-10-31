"""Diagnostic: how close does the 201-primitive DSL get on tasks it cannot solve?

For each task, enumerate every program of depth 0 to 2 over the full DSL and record
  - whether any program gives the right output shape on all training pairs,
  - the best mean cell agreement with the training outputs among those programs,
  - the agreement of simply copying the input (identity), and basic task statistics.
No colour map is used here, so agreement is a lower bound for tasks that need recolouring.

Usage: python3 -I diagnose.py <task_dir> <out_json> [task_ids.json]
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsl  # noqa: E402

NAMES = list(dsl.PRIMITIVES)


def fg_agree(r, o):
    """Agreement on foreground cells only: cells that are not background in the prediction or the target."""
    bg = dsl.bg_color(o)
    m = (r != bg) | (o != bg)
    return 1.0 if not m.any() else float((r[m] == o[m]).mean())


def work(path):
    t = json.load(open(path))
    tid = os.path.basename(path)[:-5]
    ins = [np.array(p["input"]) for p in t["train"]]
    outs = [np.array(p["output"]) for p in t["train"]]
    info = {
        "id": tid, "n_pairs": len(ins),
        "same_shape": all(i.shape == o.shape for i, o in zip(ins, outs)),
        "in_cells": float(np.mean([i.size for i in ins])),
        "out_cells": float(np.mean([o.size for o in outs])),
        "out_colors": float(np.mean([len(np.unique(o)) for o in outs])),
        "n_test": len(t["test"]),
    }
    st = {"best": -1.0, "best_fg": -1.0, "shape_ok": False, "exact": False}

    def score(res):
        if any(r is None for r in res):
            return
        if not all(r.shape == o.shape for r, o in zip(res, outs)):
            return
        st["shape_ok"] = True
        a = float(np.mean([(r == o).mean() for r, o in zip(res, outs)]))
        st["best"] = max(st["best"], a)
        st["best_fg"] = max(st["best_fg"], float(np.mean([fg_agree(r, o) for r, o in zip(res, outs)])))
        if a == 1.0:
            st["exact"] = True

    ident = None
    ident_fg = None
    if info["same_shape"]:
        ident = float(np.mean([(i == o).mean() for i, o in zip(ins, outs)]))
        ident_fg = float(np.mean([fg_agree(i, o) for i, o in zip(ins, outs)]))
    score(ins)
    for n1 in NAMES:
        r1 = [dsl.apply(n1, g) for g in ins]
        if any(r is None for r in r1):
            continue
        score(r1)
        for n2 in NAMES:
            score([dsl.apply(n2, g) for g in r1])
    info.update({"shape_reachable": st["shape_ok"], "best_agreement": st["best"] if st["shape_ok"] else None,
                 "exact_no_colormap": st["exact"], "identity_agreement": ident,
                 "best_fg_agreement": st["best_fg"] if st["shape_ok"] else None, "identity_fg_agreement": ident_fg})
    return info


def main():
    task_dir, out = sys.argv[1], sys.argv[2]
    keep = set(json.load(open(sys.argv[3]))) if len(sys.argv) > 3 else None
    files = sorted(os.path.join(task_dir, f) for f in os.listdir(task_dir)
                   if f.endswith(".json") and (keep is None or f[:-5] in keep))
    with Pool(os.cpu_count() or 2) as p:
        rows = p.map(work, files, chunksize=2)
    json.dump(rows, open(out, "w"))
    print("done", out, len(rows))


if __name__ == "__main__":
    main()
