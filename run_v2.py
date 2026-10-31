"""Instrumented ablation runner (v2).

Usage: python3 -I run_v2.py <task_dir> <model.pkl> <out_json> <config> <task_ids.json> [seed]
config: guided<k>_d<depth> | random<k>_d<depth> | all_d<depth>

Beyond the solved/not-solved score of run_guided.py it records, per task:
  seconds, search nodes (primitive applications), number of fitting programs,
  whether the first and the second guess are right, how often a random choice among
  fitting programs is right (selection ablation), the primitive subset used, and
  (for unpruned runs) the programs whose test output is correct.
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
import dsl  # noqa: E402
import features  # noqa: E402
import solver  # noqa: E402

MODEL = CFG = None
SEED = 0
MAXFIT = 3000  # cap on fitting programs evaluated for the selection ablation


def init(model_path, cfg, seed):
    global MODEL, CFG, SEED
    CFG, SEED = cfg, seed
    if model_path != "none":
        with open(model_path, "rb") as f:
            MODEL = pickle.load(f)


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
        # seed 0 reproduces the earlier single-seed baseline
        rng = np.random.default_rng(zlib.crc32(tid.encode()) ^ (SEED * 2654435761 % 2**32))
        return [solver.NAMES[i] for i in rng.choice(len(solver.NAMES), size=k, replace=False)]
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    x = features.task_features(pairs).reshape(1, -1)
    p = MODEL["model"].predict_proba(MODEL["scaler"].transform(x))[0][:-1]
    return [solver.NAMES[i] for i in np.argsort(-p)[:k]]


def run_fit(prog, cmap, g):
    out = solver.run_program(prog, g)
    if out is None:
        return None
    if cmap:
        out = np.vectorize(lambda x: cmap.get(int(x), int(x)))(out)
    return out


def work(path):
    task = json.load(open(path))
    tid = os.path.basename(path)[:-5]
    _, _, depth = parse(CFG)
    rng = np.random.default_rng(zlib.crc32(tid.encode()) + 17)
    solver.NODES = 0
    t0 = time.time()
    names = pick_names(task, tid)
    found = solver.search(task["train"], max_depth=depth, names=names)
    nodes = solver.NODES
    sec = time.time() - t0
    first_ok, top2_ok, rand1, rand2, any_ok, n_distinct = [], [], [], [], [], []
    good_progs = []
    for t in task["test"]:
        truth = np.array(t["output"]) if t.get("output") is not None else None
        att = solver.predict(found, t["input"])
        top2_ok.append(bool(truth is not None and any(np.array_equal(np.array(a[2]), truth) for a in att)))
        first_ok.append(bool(truth is not None and att and np.array_equal(np.array(att[0][2]), truth)))
        g = np.array(t["input"])
        outs = []
        for cost, prog, cmap in found[:MAXFIT]:
            o = run_fit(prog, cmap, g)
            if o is not None:
                outs.append((prog, o))
        correct = [truth is not None and o.shape == truth.shape and np.array_equal(o, truth) for _, o in outs]
        any_ok.append(bool(any(correct)))
        n_distinct.append(len({(o.shape, o.tobytes()) for _, o in outs}))
        if outs and truth is not None:
            r1 = r2 = 0
            for _ in range(200):
                order = rng.permutation(len(outs))
                a = order[0]
                r1 += correct[a]
                seen = [outs[a][1]]
                pick = [a]
                for b in order[1:]:
                    if not any(outs[b][1].shape == s.shape and np.array_equal(outs[b][1], s) for s in seen):
                        pick.append(b)
                        break
                r2 += any(correct[b] for b in pick)
            rand1.append(r1 / 200)
            rand2.append(r2 / 200)
        else:
            rand1.append(0.0)
            rand2.append(0.0)
        if parse(CFG)[0] == "all" and truth is not None:
            good_progs.append([list(p) for (p, o), c in zip(outs, correct) if c][:40])
    return {"id": tid, "n_test": len(task["test"]),
            "score": sum(top2_ok) / len(top2_ok), "first_score": sum(first_ok) / len(first_ok),
            "rand1_score": sum(rand1) / len(rand1), "rand2_score": sum(rand2) / len(rand2),
            "any_fit_correct": sum(any_ok) / len(any_ok),
            "n_fits": len(found), "n_distinct_outputs": n_distinct,
            "seconds": sec, "nodes": nodes,
            "names": names, "good_progs": good_progs}


def main():
    task_dir, model, out, cfg, ids = sys.argv[1:6]
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    keep = set(json.load(open(ids)))
    files = sorted(os.path.join(task_dir, f) for f in os.listdir(task_dir)
                   if f.endswith(".json") and f[:-5] in keep)
    with Pool(os.cpu_count() or 2, initializer=init, initargs=(model, cfg, seed)) as p:
        rows = p.map(work, files, chunksize=1)
    n = len(rows)
    summary = {
        "config": cfg, "seed": seed, "model": model, "dsl": os.environ.get("ARC_DSL", "full"), "tasks": n,
        "solved": sum(1 for r in rows if r["score"] == 1.0),
        "solved_first_guess": sum(1 for r in rows if r["first_score"] == 1.0),
        "tasks_with_fit": sum(1 for r in rows if r["n_fits"] > 0),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 3),
        "mean_nodes": round(sum(r["nodes"] for r in rows) / n, 1),
        "expected_solved_random_select_2": round(sum(1 for r in rows if r["rand2_score"] == 1.0), 1),
    }
    json.dump({"summary": summary, "rows": rows}, open(out, "w"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
