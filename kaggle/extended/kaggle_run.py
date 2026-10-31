"""Kaggle notebook entry point. Finds the challenges file, runs the combined solver, writes submission.json.

Usage in a notebook cell:  !python3 -I /kaggle/working/bundle/kaggle_run.py
Optional environment: WORKERS (default: all cores), TASK_SECONDS (default 150), CHALLENGES (explicit path), OUT.
Each task has a hard time cap; on timeout or error both attempts are the input grid. Progress is written to OUT as it goes.
"""
import glob, json, multiprocessing as mp, os, signal, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ARC_DSL", "full")
import combined_ext as combined

CAP = int(os.environ.get("TASK_SECONDS", "300"))


def find_challenges():
    p = os.environ.get("CHALLENGES")
    if p:
        return p
    cands = sorted(glob.glob("/kaggle/input/**/*test*challenges*.json", recursive=True))
    cands = cands or sorted(glob.glob("/kaggle/input/**/*challenges*.json", recursive=True))
    cands = [c for c in cands if "solution" not in c]
    if not cands:
        raise SystemExit("no challenges json found under /kaggle/input; set CHALLENGES=<path>")
    return cands[0]


class Timeout(Exception):
    pass


def _alarm(*a):
    raise Timeout()


def work(item):
    tid, task = item
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CAP)
    try:
        preds, _ = combined.solve_task_combined(task)
    except BaseException:
        preds = [[] for _ in task["test"]]
    finally:
        signal.alarm(0)
    subs = []
    for t, cand in zip(task["test"], preds):
        fb = t["input"]
        a1 = cand[0] if cand else fb
        a2 = cand[1] if len(cand) > 1 else a1
        subs.append({"attempt_1": a1, "attempt_2": a2})
    return tid, subs


if __name__ == "__main__":
    path = find_challenges()
    out = os.environ.get("OUT", "/kaggle/working/submission.json" if os.path.isdir("/kaggle/working") else "submission.json")
    tasks = json.load(open(path))
    workers = int(os.environ.get("WORKERS", os.cpu_count() or 2))
    print("challenges:", path, "tasks:", len(tasks), "workers:", workers, "cap/task:", CAP, flush=True)
    sub = {tid: [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]] for tid, task in tasks.items()}
    t0 = time.time()
    with mp.Pool(workers) as p:
        for i, (tid, s) in enumerate(p.imap_unordered(work, list(tasks.items())), 1):
            sub[tid] = s
            if i % 10 == 0 or i == len(tasks):
                json.dump(sub, open(out, "w"))
                print(i, "/", len(tasks), round(time.time() - t0), "s", flush=True)
    json.dump(sub, open(out, "w"))
    print("wrote", out)
