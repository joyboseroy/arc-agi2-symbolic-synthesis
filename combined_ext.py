"""Extended combined solver: the submitted families first and unchanged (flat DSL, object-relational orel5, cell tree celltree2),
then three added families (object trees, object trees on orel5, colour-canonical and augmented cell trees).
Added families only fill attempt slots left empty by the first three, each under its own time cap.
Usage as a library: solve_task_combined(task) -> (list per test input of up to two grids, source index per grid)."""
import os, signal, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ARC_DSL", "full")
import combined, objtree, objtree2, celltree5

CAPS = {"objtree": 20, "objtree2": 30, "celltree5": 60}


class _T(Exception):
    pass


def _h(*a):
    raise _T()


def guarded(fn, seconds, default):
    prev = signal.alarm(0)
    old = signal.signal(signal.SIGALRM, _h)
    t0 = time.time()
    signal.alarm(max(1, min(seconds, prev) if prev else seconds))
    try:
        return fn()
    except BaseException:
        return default
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        if prev:
            signal.alarm(max(1, prev - int(time.time() - t0)))


def solve_task_combined(task):
    n = len(task["test"])
    out, which = combined.solve_task_combined(task)
    empty = [[] for _ in range(n)]
    extra = [
        ("objtree", lambda: objtree.solve(task)[1]),
        ("objtree2", lambda: objtree2.solve(task)[1]),
        ("celltree5", lambda: celltree5.solve(task)[1]),
    ]
    for fi, (name, fn) in enumerate(extra, start=3):
        if all(len(o) >= 2 for o in out):
            break
        preds = guarded(fn, CAPS[name], empty)
        for k in range(n):
            for g in (preds[k] if k < len(preds) else []):
                if len(out[k]) >= 2:
                    break
                if g not in out[k]:
                    out[k].append(g)
                    which[k].append(fi)
    return out, which
