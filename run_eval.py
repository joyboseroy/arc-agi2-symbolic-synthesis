"""Run the baseline over a directory of ARC tasks and score it.

Usage: python3 -I run_eval.py <task_dir> <out_dir> [max_tasks]
"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver  # noqa: E402


def work(path):
    with open(path) as f:
        task = json.load(f)
    tid = os.path.basename(path)[:-5]
    try:
        found, results = solver.solve_task(task)
    except Exception as e:  # keep the run alive; record failure
        return tid, None, [[] for _ in task["test"]], str(e), 0
    scored = []
    for t, atts in zip(task["test"], results):
        exp = t.get("output")
        ok = exp is not None and any(a[2] == exp for a in atts)
        scored.append(ok)
    return tid, scored, results, None, len(found)


def main():
    task_dir, out_dir = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.path.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith(".json"))
    if limit:
        files = files[:limit]
    with Pool(os.cpu_count() or 2) as p:
        rows = p.map(work, files, chunksize=4)

    submission, per_task, solved_programs = {}, {}, {}
    errors = 0
    for tid, scored, results, err, nfit in rows:
        if err:
            errors += 1
        subs = []
        for atts in results:
            a1 = atts[0][2] if len(atts) > 0 else [[0]]
            a2 = atts[1][2] if len(atts) > 1 else a1
            subs.append({"attempt_1": a1, "attempt_2": a2})
        submission[tid] = subs
        if scored is not None:
            per_task[tid] = sum(scored) / len(scored)
            if any(scored):
                solved_programs[tid] = [
                    {"prog": list(a[1]), "cost": a[0]} for a in results[scored.index(True)]
                ]

    n = len(per_task)
    score = sum(per_task.values()) / max(n, 1)
    full = sum(1 for v in per_task.values() if v == 1.0)
    summary = {
        "tasks": n, "errors": errors,
        "task_score_mean": round(score, 4),
        "tasks_fully_solved": full,
        "tasks_any_test_solved": sum(1 for v in per_task.values() if v > 0),
    }
    with open(os.path.join(out_dir, "submission.json"), "w") as f:
        json.dump(submission, f)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "solved.json"), "w") as f:
        json.dump(solved_programs, f, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
