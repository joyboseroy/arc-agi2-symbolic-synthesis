"""Write a Kaggle-format submission.json for ARC-AGI-2 with the combined solver (flat DSL, then object-relational, then the cell-tree learner).

Usage: ARC_DSL=full python3 -I kaggle_submit.py <challenges.json | task_dir> <submission.json> [workers]

The challenges file maps task id -> {"train": [...], "test": [{"input": ...}]}. A directory of one-task JSON files also works.
For each test input, attempt_1 is the best flat-solver program, attempt_2 the next distinct prediction from either solver.
If nothing fits, both attempts are the input grid, which is the usual fallback.
"""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ARC_DSL", "full")
import combined


def load(path):
    if os.path.isdir(path):
        return {f[:-5]: json.load(open(os.path.join(path, f))) for f in sorted(os.listdir(path)) if f.endswith(".json")}
    return json.load(open(path))


def work(item):
    tid, task = item
    try:
        preds, _ = combined.solve_task_combined(task)
    except Exception:
        preds = [[] for _ in task["test"]]
    subs = []
    for t, cand in zip(task["test"], preds):
        fb = t["input"]
        a1 = cand[0] if cand else fb
        a2 = cand[1] if len(cand) > 1 else a1
        subs.append({"attempt_1": a1, "attempt_2": a2})
    return tid, subs


if __name__ == "__main__":
    tasks = load(sys.argv[1])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    with Pool(workers) as p:
        rows = p.map(work, list(tasks.items()), chunksize=1)
    sub = dict(rows)
    json.dump(sub, open(sys.argv[2], "w"))
    print("wrote", len(sub), "tasks")
