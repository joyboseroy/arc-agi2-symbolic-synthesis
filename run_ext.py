"""Resumable end-to-end run of combined_ext. Usage: CHUNK=n python3 -I run_ext.py <task_dir> <out.jsonl> [ids.json]
Appends one JSON line per task. Prints only counts."""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ARC_DSL", "full")
import combined_ext, signal
D, OUT = sys.argv[1], sys.argv[2]
IDS = sys.argv[3] if len(sys.argv) > 3 else None
def work(i):
    task = json.load(open(os.path.join(D, i + ".json"))); t0 = time.time()
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(300)
    try:
        preds, which = combined_ext.solve_task_combined(task)
    except BaseException:
        preds, which = [[] for _ in task["test"]], [[] for _ in task["test"]]
    finally:
        signal.alarm(0)
    sc = [any(a == t["output"] for a in p) for t, p in zip(task["test"], preds)]
    return {"id": i, "score": sum(sc) / len(sc), "which": which, "seconds": time.time() - t0}
if __name__ == "__main__":
    ids = sorted(f[:-5] for f in os.listdir(D) if f.endswith(".json"))
    if IDS:
        spec = json.load(open(IDS)); spec = spec["include"] if isinstance(spec, dict) else spec
        ids = [i for i in ids if i in set(spec)]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["id"] for l in open(OUT) if l.strip()}
    todo = [i for i in ids if i not in done][:int(os.environ.get("CHUNK", "100000"))]
    with Pool(2) as p, open(OUT, "a") as f:
        for r in p.imap_unordered(work, todo):
            f.write(json.dumps(r) + "\n"); f.flush()
    print("done", len(done) + len(todo), "of", len(ids), "left", len(ids) - len(done) - len(todo), flush=True)
