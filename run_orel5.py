"""Run the object-relational solver. Usage: python3 -I run_orel.py <task_dir> <out_json> <ids.json | all> [limit]"""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orel5 as orel

def work(path):
    tid = os.path.basename(path)[:-5]
    task = json.load(open(path))
    t0 = time.time()
    try:
        found, preds = orel.solve_any(task)
    except Exception as e:
        found, preds = [], [[] for _ in task["test"]]
    sc = []
    for t, p in zip(task["test"], preds):
        sc.append(bool(t.get("output") is not None and any(a == t["output"] for a in p)))
    return {"id": tid, "same_size": all(len(a["input"]) == len(a["output"]) and len(a["input"][0]) == len(a["output"][0]) for a in task["train"]),
            "score": sum(sc) / len(sc), "n_fit": len(found), "rules": [(r["mode"], r.get("rules", r.get("lits")), r["cost"]) for r in found][:2],
            "n_test": len(sc), "seconds": time.time() - t0}

if __name__ == "__main__":
    d, out, ids = sys.argv[1:4]
    lim = int(sys.argv[4]) if len(sys.argv) > 4 else None
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if ids != "all":
        spec = json.load(open(ids))
        if isinstance(spec, dict):
            files = [f for f in files if f[:-5] in set(spec["include"])]
        else:
            files = [f for f in files if f[:-5] in set(spec)]
    if lim: files = files[:lim]
    with Pool(2) as p:
        rows = p.map(work, [os.path.join(d, f) for f in files], chunksize=1)
    json.dump(rows, open(out, "w"))
    n = len(rows)
    print(json.dumps({"tasks": n, "solved": sum(r["score"] == 1.0 for r in rows), "any": sum(r["score"] > 0 for r in rows),
                      "with_fit": sum(r["n_fit"] > 0 for r in rows), "same_size": sum(r["same_size"] for r in rows),
                      "mean_s": round(sum(r["seconds"] for r in rows) / n, 2)}))
