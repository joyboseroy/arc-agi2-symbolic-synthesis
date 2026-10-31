"""Object trees: a decision tree over object and relation features whose leaves are object actions (V16).

The object-relational solver of orel.py labels every input object with the actions that explain the output and then picks a short
decision list (at most four rules of at most two literals). Here the same labels are used, but the structure of the rule is learned as a
tree of depth up to four over a richer set of object, rank, count and relation features. A leaf is feasible when one action is admissible
for every object that reaches it, and the leaf must change the object when the object has to change. The tree is verified by exact rendering.
This combines the object-level solver with the structure learner used for cells.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orel5 as orel
from orel5 import Ctx, ACTIONS, RELS, rel_colour, consistent_actions, literals, resolve, render, MODES

MAXD = 4


def extra_literals(ctx):
    n, objs, g = ctx.n, ctx.objs, ctx.g
    H, W = g.shape
    out = {}
    S = ctx.sizes
    for k in (1, 2, 3, 4, 6, 8, 9):
        out[f"size=={k}"] = S == k
    ordr = np.argsort(-S, kind="stable")
    second = np.zeros(n, bool)
    if n > 1:
        second[ordr[1]] = True
    out["second_largest"] = second
    r0 = np.array([o.bbox[0] for o in objs]); r1 = np.array([o.bbox[1] for o in objs])
    c0 = np.array([o.bbox[2] for o in objs]); c1 = np.array([o.bbox[3] for o in objs])
    out["topmost"] = r0 == r0.min(); out["bottommost"] = r1 == r1.max()
    out["leftmost"] = c0 == c0.min(); out["rightmost"] = c1 == c1.max()
    h = r1 - r0 + 1; w = c1 - c0 + 1
    for k in (1, 2, 3):
        out[f"h=={k}"] = h == k; out[f"w=={k}"] = w == k
    out["square"] = h == w
    cnt = np.array([(ctx.cols == c).sum() for c in ctx.cols])
    for k in (1, 2, 3):
        out[f"colcount=={k}"] = cnt == k
    shp = np.array([sum(1 for s in ctx.sigs if s == ctx.sigs[i]) for i in range(n)])
    out["shape_unique"] = shp == 1
    out["shape_repeated"] = shp > 1
    out["dist_nearest<=1"] = (ctx.dist.min(1) <= 1) if n > 1 else np.zeros(n, bool)
    out["dist_nearest>=4"] = (ctx.dist.min(1) >= 4) if n > 1 else np.zeros(n, bool)
    out["touch_border_row"] = (r0 == 0) | (r1 == H - 1)
    out["touch_border_col"] = (c0 == 0) | (c1 == W - 1)
    out["centered_row"] = (r0 + r1) == (H - 1)
    out["centered_col"] = (c0 + c1) == (W - 1)
    out["in_top_half"] = (r0 + r1) < (H - 1)
    out["in_left_half"] = (c0 + c1) < (W - 1)
    return out


def all_literals(ctx):
    d = literals(ctx)
    d.update(extra_literals(ctx))
    d.update({"not " + k: ~v for k, v in list(d.items())})
    return d


def prep(pairs, mode):
    ctxs = []
    for gi, go in pairs:
        c = Ctx(gi, mode)
        if not c.ok:
            return None
        ctxs.append(c)
    rel = [{r: rel_colour(c, r) for r in RELS} for c in ctxs]
    res = [consistent_actions(c, go, rc) for c, (gi, go), rc in zip(ctxs, pairs, rel)]
    A = np.concatenate([r[0] for r in res], 0)
    lits = [all_literals(c) for c in ctxs]
    names = list(lits[0].keys())
    L = np.stack([np.concatenate([l[nm] for l in lits]) for nm in names])
    return ctxs, A, names, L


def choose_action(A, idx):
    """Common admissible action for the objects in idx (bool mask), or None. Must change objects that cannot be kept."""
    sub = A[idx]
    common = sub.all(0)
    must = (~sub[:, 0]).any()
    if must:
        cand = np.nonzero(common[1:])[0] + 1
        return int(cand[0]) if len(cand) else None
    if common[0]:
        return 0
    cand = np.nonzero(common)[0]
    return int(cand[0]) if len(cand) else None


def grow(A, L, idx, depth, budget):
    a = choose_action(A, idx)
    if a is not None:
        return ("leaf", a)
    if depth == 0 or budget[0] <= 0:
        return None
    budget[0] -= 1
    best = None
    nI = int(idx.sum())
    for li in range(L.shape[0]):
        v = L[li] & idx
        nv = int(v.sum())
        if nv == 0 or nv == nI:
            continue
        w = idx & ~L[li]
        sc = 0
        for ch in (v, w):
            sub = A[ch]
            must = (~sub[:, 0]).any()
            cnt = sub.sum(0)
            if must:
                cnt = cnt[1:]
            sc += int(cnt.max())
        if best is None or sc > best[0]:
            best = (sc, li, v, w)
    if best is None:
        return None
    _, li, v, w = best
    t1 = grow(A, L, v, depth - 1, budget)
    if t1 is None:
        return None
    t0 = grow(A, L, w, depth - 1, budget)
    if t0 is None:
        return None
    return ("node", li, t1, t0)


def tsize(t):
    return 1 if t[0] == "leaf" else 1 + tsize(t[2]) + tsize(t[3])


def run_tree(t, names, lits, i):
    while t[0] == "node":
        t = t[2] if lits[names[t[1]]][i] else t[3]
    return ACTIONS[t[1]]


def apply_tree(g, m):
    ctx = Ctx(g, m["mode"])
    if not ctx.ok:
        return None
    lits = all_literals(ctx)
    names = m["names"]
    if any(nm not in lits for nm in names):
        return None
    relcache = {r: rel_colour(ctx, r) for r in RELS}
    conc = []
    for i in range(ctx.n):
        a = run_tree(m["tree"], names, lits, i)
        conc.append(resolve(ctx, i, a, relcache))
    return render(ctx, conc)


def induce(pairs, mode):
    p = prep(pairs, mode)
    if p is None:
        return None
    ctxs, A, names, L = p
    if A[:, 0].all():
        return None
    idx = np.ones(A.shape[0], bool)
    t = grow(A, L, idx, MAXD, [60])
    if t is None:
        return None
    m = {"mode": mode, "tree": t, "names": names, "size": tsize(t)}
    for gi, go in pairs:
        o = apply_tree(gi, m)
        if o is None or o.shape != go.shape or not np.array_equal(o, go):
            return None
    return m


def solve(task):
    pairs = [(np.array(q["input"]), np.array(q["output"])) for q in task["train"]]
    if not all(a.shape == b.shape for a, b in pairs):
        return [], [[] for _ in task["test"]]
    found = []
    for mode in MODES:
        try:
            m = induce(pairs, mode)
        except Exception:
            m = None
        if m:
            found.append(m)
    found.sort(key=lambda m: m["size"])
    preds = []
    for t in task["test"]:
        g = np.array(t["input"])
        seen, outl = [], []
        for m in found:
            try:
                o = apply_tree(g, m)
            except Exception:
                o = None
            if o is None:
                continue
            key = (o.shape, o.tobytes())
            if key in seen:
                continue
            seen.append(key)
            outl.append(o.tolist())
            if len(outl) == 2:
                break
        preds.append(outl)
    return found, preds
