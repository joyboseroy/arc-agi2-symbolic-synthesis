"""Depth-2 DSL program search with exact train verification and MDL selection."""
import itertools
import numpy as np
import dsl

NAMES = list(dsl.PRIMITIVES.keys())
NODES = 0  # number of primitive applications made by search (hardware-independent cost)


def learn_colormap(pred, exp):
    """Return dict mapping pred colors to exp colors if consistent, else None."""
    if pred.shape != exp.shape:
        return None
    m = {}
    for p, e in zip(pred.ravel().tolist(), exp.ravel().tolist()):
        if m.setdefault(p, e) != e:
            return None
    return m


def merge_maps(maps):
    out = {}
    for m in maps:
        for k, v in m.items():
            if out.setdefault(k, v) != v:
                return None
    return out


def run_program(prog, g):
    for name in prog:
        g = dsl.apply(name, g)
        if g is None:
            return None
    return g


def program_cost(prog, cmap):
    """Description length: one unit per primitive, plus a small charge per non-identity recolor entry."""
    cost = float(len(prog))
    if cmap:
        cost += 0.25 * sum(1 for k, v in cmap.items() if k != v)
    return cost


def search(train_pairs, max_depth=2, names=None):
    """Return (cost, prog, cmap) for every program consistent with all train pairs.

    names restricts the primitives the search may use (default: all of them).
    """
    names = list(names) if names is not None else NAMES
    ins = [np.array(p["input"]) for p in train_pairs]
    outs = [np.array(p["output"]) for p in train_pairs]
    found = []

    # cache depth-1 results per primitive
    cache = {(): ins}
    level = [()]
    for depth in range(1, max_depth + 1):
        nxt = []
        for prefix in level:
            for name in names:
                prog = prefix + (name,)
                prev = cache[prefix]
                res = [dsl.apply(name, g) for g in prev]
                global NODES
                NODES += len(res)
                if any(r is None for r in res):
                    continue
                cache[prog] = res
                nxt.append(prog)
                if all(r.shape == o.shape for r, o in zip(res, outs)):
                    maps = [learn_colormap(r, o) for r, o in zip(res, outs)]
                    if all(m is not None for m in maps):
                        cmap = merge_maps(maps)
                        if cmap is not None:
                            # identity-only map means a pure exact match
                            if all(k == v for k, v in cmap.items()):
                                cmap = {}
                            found.append((program_cost(prog, cmap), prog, cmap))
        level = nxt

    # depth 0: pure recolor of the input
    maps = [learn_colormap(i, o) for i, o in zip(ins, outs)]
    if all(m is not None for m in maps):
        cmap = merge_maps(maps)
        if cmap is not None and any(k != v for k, v in cmap.items()):
            found.append((program_cost((), cmap), (), cmap))
    found.sort(key=lambda t: t[0])
    return found


def predict(found, test_in, k=2):
    """Top-k distinct predictions from the lowest-cost consistent programs."""
    g = np.array(test_in)
    attempts = []
    seen = set()
    for cost, prog, cmap in found:
        out = run_program(prog, g)
        if out is None:
            continue
        if cmap:
            out = np.vectorize(lambda x: cmap.get(int(x), int(x)))(out)
        key = (out.shape, out.tobytes())
        if key in seen:
            continue
        seen.add(key)
        attempts.append((cost, prog, out.tolist()))
        if len(attempts) == k:
            break
    return attempts


def solve_task(task, max_depth=2, names=None):
    found = search(task["train"], max_depth=max_depth, names=names)
    results = []
    for t in task["test"]:
        results.append(predict(found, t["input"]))
    return found, results
