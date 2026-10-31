"""Cell trees on top of a one-step transformation: a learned residual rule after a DSL primitive.

For a task, each primitive of the 201-primitive DSL whose output has the right size on every training pair is applied to the inputs,
and a cell tree (see celltree.py) is fitted from the transformed input to the output. This lets the cell-tree learner handle tasks whose output
size differs from the input (crop, scale, tile, halves) and tasks where a whole-grid operation must be followed by a local rule.
Primitives are tried in order of how well their output already matches the target, up to K of them.
"""
import os, sys
os.environ.setdefault("ARC_DSL", "full")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsl, features as F0, celltree

K = 12


def _agree(a, b):
    return float((a == b).mean())


def prim_candidates(pairs):
    out = []
    for name in F0.NAMES:
        ok, ag = True, []
        mids = []
        for gi, go in pairs:
            try:
                m = dsl.apply(name, gi)
            except Exception:
                m = None
            if m is None or m.shape != go.shape or m.size == 0:
                ok = False
                break
            mids.append(m)
            ag.append(_agree(m, go))
        if ok and not all(np.array_equal(m, gi) for m, (gi, _) in zip(mids, pairs)):
            out.append((float(np.mean(ag)), name, mids))
    out.sort(key=lambda x: -x[0])
    return out[:K]


def fit_prim(pairs, name, mids):
    new = [(m, go) for m, (_, go) in zip(mids, pairs)]
    best = []
    for pos, ray in ((True, True), (False, True), (False, False)):
        try:
            m = celltree.fit_task(new, pos, 4, ray)
        except Exception:
            m = None
        if m is None:
            continue
        if not all(np.array_equal(celltree.apply_tree(a, m), b) for a, b in new):
            continue
        m["prim"] = name
        loo = 0
        if len(pairs) >= 3:
            for k in range(len(new)):
                tr = new[:k] + new[k + 1:]
                try:
                    mk = celltree.fit_task(tr, pos, 4, ray)
                    loo += int(mk is not None and np.array_equal(celltree.apply_tree(new[k][0], mk), new[k][1]))
                except Exception:
                    pass
            m["loo"] = loo
        else:
            m["loo"] = -1
        best.append(m)
    return best


def solve(task):
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    found = []
    same = all(a.shape == b.shape for a, b in pairs)
    if same:
        f, _ = celltree.solve(task)
        for m in f:
            m["prim"] = None
        found += f
        if any(m["loo"] == len(pairs) for m in f):
            return _predict(task, found)
    for ag, name, mids in prim_candidates(pairs):
        found += fit_prim(pairs, name, mids)
    return _predict(task, found)


def _apply(g, m):
    if m.get("prim"):
        g = dsl.apply(m["prim"], g)
        if g is None:
            return None
    return celltree.apply_tree(g, m)


def _predict(task, found):
    found.sort(key=lambda m: (-(m["loo"] if m["loo"] >= 0 else 0), m["size"], m["depth"]))
    preds = []
    for t in task["test"]:
        g = np.array(t["input"])
        seen, outl = [], []
        for m in found:
            try:
                o = _apply(g, m)
            except Exception:
                o = None
            if o is None:
                continue
            key = (o.shape, o.tobytes())
            if key in seen:
                continue
            seen.append(key)
            outl.append(o.tolist())
        preds.append(outl[:2])
    return found, preds
