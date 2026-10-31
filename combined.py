"""Combined solver: flat DSL search first, then the object-relational solver, then the cell-tree learner.

Attempts for a test input are filled in this order, skipping duplicates, until two are found. If a solver family has a fitting
program or rule it is used before the next family. Usage as a library: solve_task_combined(task) -> list (per test input) of up to two grids.
Usage as a script: ARC_DSL=full python3 -I combined.py <task_dir> <out_json> <ids.json|all>   (reports scores when outputs are present)
"""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ARC_DSL", "full")
import solver, orel5, celltree2 as celltree


def solve_task_combined(task):
    n = len(task["test"])
    fams = []
    try:
        _, res = solver.solve_task(task)
        fams.append([[a[2] for a in atts] for atts in res])
    except Exception:
        fams.append([[] for _ in range(n)])
    try:
        _, p = orel5.solve_any(task)
        fams.append(p)
    except Exception:
        fams.append([[] for _ in range(n)])
    try:
        _, p = celltree.solve(task)
        fams.append(p)
    except Exception:
        fams.append([[] for _ in range(n)])
    out, which = [], []
    for k in range(n):
        cand, src = [], []
        for fi, f in enumerate(fams):
            for g in f[k] if k < len(f) else []:
                if g not in cand:
                    cand.append(g)
                    src.append(fi)
        out.append(cand[:2])
        which.append(src[:2])
    return out, which


def work(path):
    tid = os.path.basename(path)[:-5]
    task = json.load(open(path))
    t0 = time.time()
    preds, which = solve_task_combined(task)
    sc = [bool(t.get("output") is not None and any(a == t["output"] for a in p)) for t, p in zip(task["test"], preds)]
    return {"id": tid, "score": sum(sc) / len(sc), "n_test": len(sc), "which": which, "seconds": time.time() - t0}


if __name__ == "__main__":
    d, out, ids = sys.argv[1:4]
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if ids != "all":
        spec = json.load(open(ids)); spec = spec["include"] if isinstance(spec, dict) else spec
        files = [f for f in files if f[:-5] in set(spec)]
    with Pool(2) as p:
        rows = p.map(work, [os.path.join(d, f) for f in files], chunksize=1)
    json.dump(rows, open(out, "w"))
    print(json.dumps({"tasks": len(rows), "solved": sum(r["score"] == 1.0 for r in rows), "mean_s": round(sum(r["seconds"] for r in rows) / len(rows), 2)}))
