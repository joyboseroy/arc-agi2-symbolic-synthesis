"""Exhaustive existence search with semantic deduplication of intermediate states.

For every task it asks: is there a program of depth <= D over the library whose output (after one global colour map)
fits every training pair? Programs that give identical grids on all training inputs are merged, so each distinct
intermediate state is expanded once. This answers existence, not which of several equivalent programs is best.

Usage: python3 -I depthsearch.py <task_dir> <out_json> <ids.json> <max_depth>   (ARC_DSL selects the library)
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsl, solver

NAMES = list(dsl.PRIMITIVES)
D = 3


def key_of(res):
    return tuple((r.shape, r.tobytes()) for r in res)


def fits(res, outs):
    if any(r.shape != o.shape for r, o in zip(res, outs)):
        return False
    maps = [solver.learn_colormap(r, o) for r, o in zip(res, outs)]
    if any(m is None for m in maps):
        return False
    return solver.merge_maps(maps) is not None


def work(args):
    path, depth = args
    t = json.load(open(path))
    tid = os.path.basename(path)[:-5]
    ins = [np.array(p["input"]) for p in t["train"]]
    outs = [np.array(p["output"]) for p in t["train"]]
    t0 = time.time()
    seen = {key_of(ins)}
    level = [((), ins)]
    nodes = 0
    first = None
    fit_by_depth = {}
    states = {0: 1}
    if fits(ins, outs):
        first = 0
    for d in range(1, depth + 1):
        nxt = []
        nfit = 0
        for prog, res in level:
            for n in NAMES:
                r = [dsl.apply(n, g) for g in res]
                nodes += len(r)
                if any(x is None for x in r):
                    continue
                if fits(r, outs):
                    nfit += 1
                    if first is None:
                        first = d
                if d < depth:
                    k = key_of(r)
                    if k in seen:
                        continue
                    seen.add(k)
                    nxt.append((prog + (n,), r))
        fit_by_depth[d] = nfit
        states[d] = len(nxt)
        level = nxt
    return {"id": tid, "first_fit_depth": first, "fits_by_depth": fit_by_depth, "states": states,
            "nodes": nodes, "seconds": time.time() - t0}


if __name__ == "__main__":
    d, out, ids, depth = sys.argv[1:5]
    keep = set(json.load(open(ids)))
    files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".json") and f[:-5] in keep]
    with Pool(2) as p:
        rows = p.map(work, [(f, int(depth)) for f in files], chunksize=1)
    json.dump(rows, open(out, "w"))
    print(json.dumps({"tasks": len(rows), "with_fit": sum(r["first_fit_depth"] is not None for r in rows),
                      "mean_s": round(sum(r["seconds"] for r in rows) / len(rows), 1)}))
