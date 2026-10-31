"""Ablation: proposer-guided vs random vs unrestricted primitive pruning.

Usage: python3 -I run_guided.py <task_dir> <model.pkl> <out_json> <config> [max_tasks]
config: guided<k>_d<depth> | random<k>_d<depth> | all_d<depth>   e.g. guided16_d3
"""
import json
import os
import pickle
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import solver  # noqa: E402

MODEL = None
CFG = None


def init(model_path, cfg):
    global MODEL, CFG
    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)
    CFG = cfg


def parse(cfg):
    kind, depth = cfg.split("_d")
    if kind == "all":
        return "all", None, int(depth)
    for pre in ("guided", "random"):
        if kind.startswith(pre):
            return pre, int(kind[len(pre):]), int(depth)
    raise ValueError(cfg)


def pick_names(task, tid):
    kind, k, _ = parse(CFG)
    if kind == "all":
        return None
    if kind == "random":
        rng = np.random.default_rng(zlib.crc32(tid.encode()))
        return [solver.NAMES[i] for i in rng.choice(len(solver.NAMES), size=k, replace=False)]
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    x = features.task_features(pairs).reshape(1, -1)
    p = MODEL["model"].predict_proba(MODEL["scaler"].transform(x))[0][:-1]
    return [solver.NAMES[i] for i in np.argsort(-p)[:k]]


def work(path):
    task = json.load(open(path))
    tid = os.path.basename(path)[:-5]
    _, _, depth = parse(CFG)
    t0 = time.time()
    names = pick_names(task, tid)
    found, results = solver.solve_task(task, max_depth=depth, names=names)
    scored = [t.get("output") is not None and any(a[2] == t["output"] for a in atts)
              for t, atts in zip(task["test"], results)]
    return tid, sum(scored) / len(scored), len(found) > 0, time.time() - t0


def main():
    task_dir, model, out, cfg = sys.argv[1:5]
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
    files = sorted(os.path.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith(".json"))
    if limit:
        files = files[:limit]
    with Pool(os.cpu_count() or 2, initializer=init, initargs=(model, cfg)) as p:
        rows = p.map(work, files, chunksize=2)
    n = len(rows)
    summary = {
        "config": cfg, "tasks": n,
        "task_score_mean": round(sum(r[1] for r in rows) / n, 4),
        "tasks_fully_solved": sum(1 for r in rows if r[1] == 1.0),
        "tasks_any_solved": sum(1 for r in rows if r[1] > 0),
        "tasks_with_fitting_program": sum(1 for r in rows if r[2]),
        "mean_seconds_per_task": round(sum(r[3] for r in rows) / n, 2),
    }
    json.dump({"summary": summary, "per_task": {r[0]: r[1] for r in rows}}, open(out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
