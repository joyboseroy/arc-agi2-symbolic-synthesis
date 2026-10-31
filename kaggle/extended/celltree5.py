"""Cell trees with two ideas borrowed from the test-time-training systems, in symbolic form.

canon: colour canonicalisation. In every pair the input colours are relabelled by frequency rank (0 = most common), colours that occur only in the
output keep a fixed label 10 + colour. A tree fitted in that space can express rules whose colours differ from pair to pair (the symbolic
counterpart of colour-permutation augmentation).
aug: geometric augmentation. The training pairs are extended by the eight rotations/flips applied to input and output together, and the tree
must fit all of them (the symbolic counterpart of augmentation in test-time training). It restricts the hypothesis to rotation/flip-invariant rules.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import celltree5_base as ct


def d8(g, k):
    g = np.rot90(g, k % 4)
    return g[:, ::-1] if k >= 4 else g


def canon_map(g):
    v, c = np.unique(g, return_counts=True)
    order = sorted(zip(-c, v))
    return {int(col): i for i, (_, col) in enumerate(order)}


def to_canon(g, mp, out=False):
    r = np.zeros_like(g)
    for col in range(10):
        r[g == col] = mp.get(col, 10 + col)
    return r


def from_canon(g, mp):
    inv = {v: k for k, v in mp.items()}
    r = np.zeros_like(g)
    for lab in np.unique(g):
        if lab >= 10:
            col = lab - 10
        elif lab in inv:
            col = inv[lab]
        else:
            return None
        r[g == lab] = col
    return r


def fit(pairs, mode):
    """returns list of models, each a dict with the tree and the mode"""
    if mode == "canon":
        maps = [canon_map(gi) for gi, _ in pairs]
        tp = [(to_canon(gi, mp), to_canon(go, mp)) for (gi, go), mp in zip(pairs, maps)]
    elif mode == "aug":
        tp = [(d8(gi, k), d8(go, k)) for gi, go in pairs for k in range(8)]
    else:
        tp = pairs
    out = []
    for pos, ray in ((True, True), (False, True), (False, False)):
        try:
            m = ct.fit_task(tp, pos, 5, ray)
        except Exception:
            m = None
        if m is None:
            continue
        if not all(np.array_equal(ct.apply_tree(a, m), b) for a, b in tp):
            continue
        m["mode"] = mode
        loo = -1
        if len(pairs) >= 3:
            loo = 0
            for k in range(len(pairs)):
                tr = pairs[:k] + pairs[k + 1:]
                try:
                    if mode == "canon":
                        mps = [canon_map(a) for a, _ in tr]
                        ttr = [(to_canon(a, mp), to_canon(b, mp)) for (a, b), mp in zip(tr, mps)]
                    elif mode == "aug":
                        ttr = [(d8(a, kk), d8(b, kk)) for a, b in tr for kk in range(8)]
                    else:
                        ttr = tr
                    mk = ct.fit_task(ttr, pos, 5, ray)
                    if mk is not None:
                        mk["mode"] = mode
                        o = apply(pairs[k][0], mk)
                        loo += int(o is not None and np.array_equal(o, pairs[k][1]))
                except Exception:
                    pass
        m["loo"] = loo
        out.append(m)
    return out


def apply(g, m):
    if m["mode"] == "canon":
        mp = canon_map(g)
        o = ct.apply_tree(to_canon(g, mp), m)
        return None if o is None else from_canon(o, mp)
    return ct.apply_tree(g, m)


def solve(task, modes=("canon", "aug")):
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    if not all(a.shape == b.shape for a, b in pairs):
        return [], [[] for _ in task["test"]]
    found = []
    for mode in modes:
        found += fit(pairs, mode)
    found.sort(key=lambda m: (-(m["loo"] if m["loo"] >= 0 else 0), m["size"], m["depth"]))
    preds = []
    for t in task["test"]:
        g = np.array(t["input"])
        seen, outl = [], []
        for m in found:
            try:
                o = apply(g, m)
            except Exception:
                o = None
            if o is None:
                continue
            key = o.tobytes()
            if key in seen:
                continue
            seen.append(key)
            outl.append(o.tolist())
        preds.append(outl[:2])
    return found, preds
