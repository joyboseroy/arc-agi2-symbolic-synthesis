"""Object-relational rule induction, version 4: adds relational placement actions (align with, place next to, mirror across another object).
Extends the frozen solver orel3_frozen.py, which is not changed.

Previous docstring: Object-relational rule induction, version 3 (adds copy, pull-to-object, stamp-at-markers and crop-selection).
This file extends the frozen solver orel_frozen.py; the frozen file is not changed.

Original docstring: Object-relational rule induction for same-size ARC tasks (solver v2).

Idea: split every input grid into objects, label each object with the action that explains the output
(keep, remove, recolour, fill, move), then find the shortest decision rule over object properties and
relations that reproduces the labels on every training pair. The rule is verified by exact rendering.

Usage as a library: solve(task) -> list of up to two predictions per test input, plus the rule description.
"""
import itertools
import numpy as np
from scipy import ndimage

MODES = ("c8", "c4", "m8")
MAXOBJ = 60


class Obj:
    __slots__ = ("cells", "r", "c", "colour", "bbox", "size", "mask")


class Ctx:
    def __init__(self, g, mode):
        self.g = g
        H, W = g.shape
        vals, cnt = np.unique(g, return_counts=True)
        self.bg = int(vals[cnt.argmax()])
        objs = []
        if mode == "m8":
            lab, n = ndimage.label(g != self.bg, structure=np.ones((3, 3)))
            groups = [lab == k for k in range(1, n + 1)]
        else:
            st = np.ones((3, 3)) if mode == "c8" else None
            groups = []
            for col in vals:
                if col == self.bg:
                    continue
                lab, n = ndimage.label(g == col, structure=st)
                groups += [lab == k for k in range(1, n + 1)]
        for m in groups:
            o = Obj()
            rr, cc = np.nonzero(m)
            o.r, o.c, o.mask = rr, cc, m
            o.cells = (rr, cc)
            v, ct = np.unique(g[rr, cc], return_counts=True)
            o.colour = int(v[ct.argmax()])
            o.bbox = (rr.min(), rr.max(), cc.min(), cc.max())
            o.size = len(rr)
            objs.append(o)
        objs.sort(key=lambda o: (o.bbox[0], o.bbox[2], o.size))
        self.objs = objs
        self.n = len(objs)
        self.ok = 0 < self.n <= MAXOBJ
        if self.ok:
            self._relations()

    def shape_sig(self, o):
        return frozenset(zip((o.r - o.bbox[0]).tolist(), (o.c - o.bbox[2]).tolist()))

    def holes(self, o):
        r0, r1, c0, c1 = o.bbox
        sub = o.mask[r0:r1 + 1, c0:c1 + 1]
        filled = ndimage.binary_fill_holes(sub)
        return filled & ~sub  # cells enclosed by the object

    def _relations(self):
        n, objs = self.n, self.objs
        H, W = self.g.shape
        sigs = [self.shape_sig(o) for o in objs]
        R = {k: np.zeros((n, n), bool) for k in ("touch", "samecol", "sameshape", "rowov", "colov", "inside", "contains",
                                                   "left", "right", "above", "below")}
        dist = np.zeros((n, n))
        for i, a in enumerate(objs):
            ad = ndimage.binary_dilation(a.mask, structure=np.ones((3, 3)))
            for j, b in enumerate(objs):
                if i == j:
                    dist[i, j] = 1e9
                    continue
                R["touch"][i, j] = bool((ad & b.mask).any())
                R["samecol"][i, j] = a.colour == b.colour
                R["sameshape"][i, j] = sigs[i] == sigs[j]
                R["rowov"][i, j] = not (a.bbox[1] < b.bbox[0] or b.bbox[1] < a.bbox[0])
                R["colov"][i, j] = not (a.bbox[3] < b.bbox[2] or b.bbox[3] < a.bbox[2])
                R["inside"][i, j] = a.bbox[0] > b.bbox[0] and a.bbox[1] < b.bbox[1] and a.bbox[2] > b.bbox[2] and a.bbox[3] < b.bbox[3]
                R["contains"][i, j] = R["inside"][i, j] if False else False
                R["left"][i, j] = a.bbox[3] < b.bbox[2]
                R["right"][i, j] = a.bbox[2] > b.bbox[3]
                R["above"][i, j] = a.bbox[1] < b.bbox[0]
                R["below"][i, j] = a.bbox[0] > b.bbox[1]
                gr = max(0, max(b.bbox[0] - a.bbox[1], a.bbox[0] - b.bbox[1]) - 1)
                gc = max(0, max(b.bbox[2] - a.bbox[3], a.bbox[2] - b.bbox[3]) - 1)
                dist[i, j] = max(gr, gc) + 0.01 * (gr + gc)
        R["contains"] = R["inside"].T.copy()
        self.R, self.dist, self.sigs = R, dist, sigs
        self.sizes = np.array([o.size for o in objs])
        self.cols = np.array([o.colour for o in objs])
        self.nearest = dist.argmin(1) if n > 1 else np.zeros(n, int)


# ----------------------------------------------------------------------------- literals
def _unary(ctx):
    objs, n, g = ctx.objs, ctx.n, ctx.g
    H, W = g.shape
    S = ctx.sizes
    out = {}
    for c in range(10):
        out[f"col={c}"] = ctx.cols == c
    out["largest"] = S == S.max()
    out["smallest"] = S == S.min()
    out["border"] = np.array([o.bbox[0] == 0 or o.bbox[2] == 0 or o.bbox[1] == H - 1 or o.bbox[3] == W - 1 for o in objs])
    out["hole"] = np.array([ctx.holes(o).any() for o in objs])
    out["rect"] = np.array([o.size == (o.bbox[1] - o.bbox[0] + 1) * (o.bbox[3] - o.bbox[2] + 1) for o in objs])
    out["pixel"] = S == 1
    out["wide"] = np.array([(o.bbox[3] - o.bbox[2]) > (o.bbox[1] - o.bbox[0]) for o in objs])
    out["tall"] = np.array([(o.bbox[1] - o.bbox[0]) > (o.bbox[3] - o.bbox[2]) for o in objs])
    out["bigger_than_mean"] = S > S.mean()
    out["symm_lr"] = np.array([np.array_equal(o.mask[o.bbox[0]:o.bbox[1] + 1, o.bbox[2]:o.bbox[3] + 1],
                                              o.mask[o.bbox[0]:o.bbox[1] + 1, o.bbox[2]:o.bbox[3] + 1][:, ::-1]) for o in objs])
    for k in (2, 3, 4, 5):
        out[f"size<={k}"] = S <= k
    out["size_odd"] = S % 2 == 1
    cnt_col = np.array([(ctx.cols == c).sum() for c in ctx.cols])
    out["col_unique"] = cnt_col == 1
    out["col_repeated"] = cnt_col > 1
    cmax = np.bincount(ctx.cols, minlength=10).max()
    out["col_most_common"] = cnt_col == cmax
    out["col_rarest"] = cnt_col == cnt_col.min()
    return out


def literals(ctx):
    """Dict name -> boolean vector over objects (both the literal and its negation are produced by the caller)."""
    out = _unary(ctx)
    n = ctx.n
    for r, M in ctx.R.items():
        out[f"any_{r}"] = M.any(1)
        for c in range(10):
            sel = ctx.cols == c
            out[f"{r}_col{c}"] = (M & sel[None, :]).any(1) if sel.any() else np.zeros(n, bool)
    out["nearest_samecol"] = ctx.cols[ctx.nearest] == ctx.cols if n > 1 else np.zeros(n, bool)
    out["nearest_sameshape"] = np.array([ctx.sigs[ctx.nearest[i]] == ctx.sigs[i] for i in range(n)]) if n > 1 else np.zeros(n, bool)
    for c in range(10):
        out[f"nearest_col{c}"] = (ctx.cols[ctx.nearest] == c) if n > 1 else np.zeros(n, bool)
    return out


# ----------------------------------------------------------------------------- actions
def rel_colour(ctx, name):
    """Colour of a related object for every object (-1 if none)."""
    n = ctx.n
    res = np.full(n, -1)
    for i in range(n):
        if name == "nearest" and n > 1:
            res[i] = ctx.cols[ctx.nearest[i]]
        elif name == "touch":
            js = np.nonzero(ctx.R["touch"][i])[0]
            if len(js):
                res[i] = ctx.cols[js[np.argmin(ctx.dist[i, js])]]
        elif name == "container":
            js = np.nonzero(ctx.R["inside"][i])[0]
            if len(js):
                res[i] = ctx.cols[js[np.argmin(ctx.sizes[js])]]
        elif name == "largest":
            res[i] = ctx.cols[ctx.sizes.argmax()]
        elif name == "smallest":
            res[i] = ctx.cols[ctx.sizes.argmin()]
        elif name == "same_shape":
            js = np.nonzero(ctx.R["sameshape"][i])[0]
            if len(js):
                res[i] = ctx.cols[js[0]]
    return res


RELS = ("nearest", "touch", "container", "largest", "smallest", "same_shape")
TARGETS = ["nearest", "largest", "smallest"] + [("col", c) for c in range(10)]
MOVES = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if (dy, dx) != (0, 0)]


def all_actions():
    acts = [("keep",), ("remove",)]
    acts += [("recolor", c) for c in range(10)]
    acts += [("recolor_rel", r) for r in RELS]
    acts += [("fillbox", c) for c in range(10)]
    acts += [("fillhole", c) for c in range(10)]
    acts += [("move", dy, dx) for dy, dx in MOVES]
    acts += [("slide", d) for d in ("up", "down", "left", "right")]
    acts += [("flip_h",), ("flip_v",)]
    for kind in ("connect", "hline", "vline", "frame"):
        acts += [(kind, -1)] + [(kind, c) for c in range(10)]
    acts += [("outline", c) for c in range(10)]
    acts += [("copy", dy, dx) for dy, dx in MOVES]
    acts += [("pull", c) for c in range(-1, 10)]
    acts += [("stamp", c) for c in range(-1, 10)] + [("stampc", c) for c in range(10)]
    for T in TARGETS:
        acts += [("align", "x", T), ("align", "y", T), ("align", "xy", T)]
        acts += [("adj", side, cen, T) for side in ("above", "below", "left", "right") for cen in (0, 1)]
        acts += [("reflect", ax, keep, T) for ax in ("x", "y") for keep in (0, 1)]
    for d in ("up", "down", "left", "right", "ul", "ur", "dl", "dr"):
        acts += [("ray", d, -1)] + [("ray", d, c) for c in range(10)]
    return acts


ACTIONS = all_actions()


def _target(ctx, i, T):
    n = ctx.n
    if n < 2:
        return None
    if T == "nearest":
        return int(ctx.nearest[i])
    if T in ("largest", "smallest"):
        order = np.argsort(ctx.sizes if T == "smallest" else -ctx.sizes, kind="stable")
        for j in order:
            if j != i:
                if ctx.sizes[j] == ctx.sizes[order[0]] or True:
                    return int(j) if (ctx.sizes[j] == (ctx.sizes.max() if T == "largest" else ctx.sizes.min())) else None
        return None
    js = [j for j in range(n) if j != i and ctx.cols[j] == T[1]]
    if not js:
        return None
    return min(js, key=lambda j: ctx.dist[i, j])


def resolve(ctx, i, a, relcache):
    """Turn a symbolic action into a concrete one for object i (or None if it does not apply)."""
    if a[0] == "recolor_rel":
        c = relcache[a[1]][i]
        return None if c < 0 else ("recolor", int(c))
    if a[0] == "slide":
        g = ctx.g
        o = ctx.objs[i]
        H, W = g.shape
        block = (g != ctx.bg) & ~o.mask
        dy, dx = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[a[1]]
        k = 0
        while True:
            rr, cc = o.r + dy * (k + 1), o.c + dx * (k + 1)
            if rr.min() < 0 or cc.min() < 0 or rr.max() >= H or cc.max() >= W or block[rr, cc].any():
                break
            k += 1
        return None if k == 0 else ("move", dy * k, dx * k)
    if a[0] in ("align", "adj", "reflect"):
        T = a[-1]
        j = _target(ctx, i, T)
        if j is None:
            return None
        o, q = ctx.objs[i], ctx.objs[j]
        r0, r1, c0, c1 = o.bbox
        tr0, tr1, tc0, tc1 = q.bbox
        cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
        tcy, tcx = (tr0 + tr1) // 2, (tc0 + tc1) // 2
        if a[0] == "align":
            dy = tcy - cy if "y" in a[1] else 0
            dx = tcx - cx if "x" in a[1] else 0
        elif a[0] == "adj":
            side, cen = a[1], a[2]
            if side == "above":
                dy, dx = tr0 - 1 - r1, (tcx - cx if cen else 0)
            elif side == "below":
                dy, dx = tr1 + 1 - r0, (tcx - cx if cen else 0)
            elif side == "left":
                dy, dx = (tcy - cy if cen else 0), tc0 - 1 - c1
            else:
                dy, dx = (tcy - cy if cen else 0), tc1 + 1 - c0
        else:
            return ("reflect", a[1], a[2], (tc0 + tc1) if a[1] == "x" else (tr0 + tr1))
        return None if dy == 0 and dx == 0 else ("move", dy, dx)
    if a[0] == "pull":
        g = ctx.g
        o = ctx.objs[i]
        H, W = g.shape
        best = None
        for j, q in enumerate(ctx.objs):
            if j == i or (a[1] >= 0 and q.colour != a[1]):
                continue
            if ctx.R["touch"][i, j]:
                continue
            if ctx.R["colov"][i, j] and (ctx.R["above"][i, j] or ctx.R["below"][i, j]):
                d = (1, 0) if ctx.R["above"][i, j] else (-1, 0)
            elif ctx.R["rowov"][i, j] and (ctx.R["left"][i, j] or ctx.R["right"][i, j]):
                d = (0, 1) if ctx.R["left"][i, j] else (0, -1)
            else:
                continue
            if best is None or ctx.dist[i, j] < best[0]:
                best = (ctx.dist[i, j], d)
        if best is None:
            return None
        dy, dx = best[1]
        block = (g != ctx.bg) & ~o.mask
        k = 0
        while True:
            rr, cc = o.r + dy * (k + 1), o.c + dx * (k + 1)
            if rr.min() < 0 or cc.min() < 0 or rr.max() >= H or cc.max() >= W or block[rr, cc].any():
                break
            k += 1
        return None if k == 0 else ("move", dy * k, dx * k)
    return a


def render(ctx, concrete):
    """concrete: list of (action tuples or None) per object. Returns new grid."""
    g = ctx.g
    out = g.copy()
    H, W = g.shape
    for o, a in zip(ctx.objs, concrete):
        if a and (a[0] in ("remove", "move", "flip_h", "flip_v") or (a[0] == "reflect" and not a[2])):
            out[o.cells] = ctx.bg
    for o, a in zip(ctx.objs, concrete):
        if not a:
            continue
        if a[0] == "recolor":
            out[o.cells] = a[1]
        elif a[0] == "move":
            rr, cc = o.r + a[1], o.c + a[2]
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            out[rr[ok], cc[ok]] = g[o.r[ok], o.c[ok]]
        elif a[0] in ("flip_h", "flip_v"):
            r0, r1, c0, c1 = o.bbox
            rr = o.r if a[0] == "flip_h" else r0 + r1 - o.r
            cc = c0 + c1 - o.c if a[0] == "flip_h" else o.c
            out[rr, cc] = g[o.r, o.c]
    for o, a in zip(ctx.objs, concrete):
        if a and a[0] == "reflect":
            if a[1] == "x":
                rr, cc = o.r, a[3] - o.c
            else:
                rr, cc = a[3] - o.r, o.c
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            out[rr[ok], cc[ok]] = g[o.r[ok], o.c[ok]]
        elif a and a[0] == "copy":
            rr, cc = o.r + a[1], o.c + a[2]
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            out[rr[ok], cc[ok]] = g[o.r[ok], o.c[ok]]
        elif a and a[0] in ("stamp", "stampc") and o.size > 1:
            cy, cx = (o.bbox[0] + o.bbox[1]) // 2, (o.bbox[2] + o.bbox[3]) // 2
            for q in ctx.objs:
                if q is o or q.size != 1:
                    continue
                if a[0] == "stamp" and a[1] >= 0 and q.colour != a[1]:
                    continue
                if a[0] == "stampc" and q.colour != a[1]:
                    continue
                if a[0] == "stamp" and a[1] < 0 and q.colour == o.colour:
                    continue
                rr, cc = o.r + (int(q.r[0]) - cy), o.c + (int(q.c[0]) - cx)
                ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                col = g[o.r[ok], o.c[ok]] if a[0] == "stamp" else q.colour
                out[rr[ok], cc[ok]] = col
    for o, a in zip(ctx.objs, concrete):
        if not a:
            continue
        r0, r1, c0, c1 = o.bbox
        if a[0] == "fillbox":
            sub = out[r0:r1 + 1, c0:c1 + 1]
            sub[sub == ctx.bg] = a[1]
        elif a[0] == "fillhole":
            hm = ctx.holes(o)
            sub = out[r0:r1 + 1, c0:c1 + 1]
            sub[hm] = a[1]
        elif a[0] == "outline":
            nb = ndimage.binary_dilation(o.mask, structure=np.ones((3, 3))) & ~o.mask & (out == ctx.bg)
            out[nb] = a[1]
        elif a[0] in ("hline", "vline"):
            col = o.colour if a[1] < 0 else a[1]
            if a[0] == "hline":
                for r in set(o.r.tolist()):
                    row = out[r]
                    row[(g[r] == ctx.bg) & (row == ctx.bg)] = col
            else:
                for c in set(o.c.tolist()):
                    cl = out[:, c]
                    cl[(g[:, c] == ctx.bg) & (cl == ctx.bg)] = col
        elif a[0] == "frame":
            col = o.colour if a[1] < 0 else a[1]
            ring = np.zeros_like(o.mask)
            r0_, r1_, c0_, c1_ = max(0, r0 - 1), min(H - 1, r1 + 1), max(0, c0 - 1), min(W - 1, c1 + 1)
            ring[r0_:r1_ + 1, c0_:c1_ + 1] = True
            ring[r0:r1 + 1, c0:c1 + 1] = False
            out[ring & (out == ctx.bg)] = col
        elif a[0] == "connect":
            col = o.colour if a[1] < 0 else a[1]
            for q in ctx.objs:
                if q is o:
                    continue
                for r in set(o.r.tolist()) & set(q.r.tolist()):
                    ca, cb = o.c[o.r == r], q.c[q.r == r]
                    lo, hi = (ca.max(), cb.min()) if ca.max() < cb.min() else ((cb.max(), ca.min()) if cb.max() < ca.min() else (None, None))
                    if lo is not None and hi - lo > 1:
                        seg = out[r, lo + 1:hi]
                        seg[seg == ctx.bg] = col
                for c in set(o.c.tolist()) & set(q.c.tolist()):
                    ra, rb = o.r[o.c == c], q.r[q.c == c]
                    lo, hi = (ra.max(), rb.min()) if ra.max() < rb.min() else ((rb.max(), ra.min()) if rb.max() < ra.min() else (None, None))
                    if lo is not None and hi - lo > 1:
                        seg = out[lo + 1:hi, c]
                        seg[seg == ctx.bg] = col
        elif a[0] == "ray":
            dy, dx = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
                      "ul": (-1, -1), "ur": (-1, 1), "dl": (1, -1), "dr": (1, 1)}[a[1]]
            col = o.colour if a[2] < 0 else a[2]
            for r, c in zip(o.r.tolist(), o.c.tolist()):
                y, x = r + dy, c + dx
                while 0 <= y < H and 0 <= x < W and g[y, x] == ctx.bg:
                    if out[y, x] == ctx.bg:
                        out[y, x] = col
                    y += dy
                    x += dx
    return out


def consistent_actions(ctx, out, relcache):
    """Per object and action: (consistent?, cells of the grid that differ from the input and that the action reproduces)."""
    n, g = ctx.n, ctx.g
    allcells = np.zeros(g.shape, bool)
    for o in ctx.objs:
        allcells |= o.mask
    U = (out != g)
    A = np.zeros((n, len(ACTIONS)), bool)
    E = np.zeros((n, len(ACTIONS), g.size), bool)
    for i, o in enumerate(ctx.objs):
        others = allcells & ~o.mask
        r0, r1, c0, c1 = o.bbox
        for ai, a in enumerate(ACTIONS):
            a2 = resolve(ctx, i, a, relcache)
            if a2 is None:
                continue
            conc = [None] * n
            conc[i] = a2
            r = render(ctx, conc)
            eff = (r != g)
            reg = eff.copy()
            reg[r0:r1 + 1, c0:c1 + 1] = True
            reg &= ~others
            if np.array_equal(r[reg], out[reg]):
                A[i, ai] = True
                E[i, ai] = (eff & U & (r == out)).ravel()
    return A, E


# ----------------------------------------------------------------------------- induction
def induce(pairs, mode, max_rules=4):
    """pairs: list of (in, out) same-shape arrays. Returns a rule set or None."""
    ctxs, outs = [], []
    for gi, go in pairs:
        c = Ctx(gi, mode)
        if not c.ok:
            return None
        ctxs.append(c)
        outs.append(go)
    rel = [{r: rel_colour(c, r) for r in RELS} for c in ctxs]
    res = [consistent_actions(c, o, rc) for c, o, rc in zip(ctxs, outs, rel)]
    A = np.concatenate([r[0] for r in res], 0)
    nobj = A.shape[0]
    # universe of changed cells, one block per pair
    Ucells = []
    for p, (gi, go) in enumerate(pairs):
        Ucells.append((gi != go).ravel())
    offs = np.cumsum([0] + [int(u.sum()) for u in Ucells])
    nU = int(offs[-1])
    nA = len(ACTIONS)
    CM = np.zeros((nobj, nA, nU), bool)
    base = 0
    for p, (c, (Ap, Ep)) in enumerate(zip(ctxs, res)):
        idx = np.nonzero(Ucells[p])[0]
        CM[base:base + c.n, :, offs[p]:offs[p + 1]] = Ep[:, :, idx]
        base += c.n
    must = ~A[:, 0]
    if not must.any() and nU == 0:
        return None
    lits = [literals(c) for c in ctxs]
    names = list(lits[0].keys())
    L = np.stack([np.concatenate([l[nm] for l in lits]) for nm in names])
    L = np.concatenate([L, ~L], 0)
    names = names + ["not " + nm for nm in names]
    unc_o = must.copy()
    unc_u = np.ones(nU, bool)
    rules, nlit = [], 0
    for _ in range(max_rules):
        if not unc_o.any() and not unc_u.any():
            break
        best = None
        for ai in range(1, nA):
            adm = A[:, ai]
            if not adm.any():
                continue
            cm = CM[:, ai, :] & adm[:, None]
            objgain = adm & unc_o
            if not objgain.any() and not (cm.any(0) & unc_u).any():
                continue
            bad = ~adm
            ok1 = ~((L & bad[None, :]).any(1))
            li_list = np.nonzero(ok1)[0]
            if len(li_list):
                Lv = L[li_list]
                gu = ((Lv.astype(np.int32) @ cm.astype(np.int32)) > 0) & unc_u[None, :]
                go = (Lv & objgain[None, :]).sum(1)
                gain = gu.sum(1) + go
                for k in np.nonzero(gain > 0)[0]:
                    cand = (int(gain[k]), -1, Lv[k], (int(li_list[k]),), ai)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
            total_unc = int(unc_o.sum() + unc_u.sum())
            if best is None or best[0] < total_unc:
                useful = np.nonzero(((L & adm[None, :]).any(1)))[0]
                useful = useful[:120]
                for a_, b_ in itertools.combinations(useful, 2):
                    v = L[a_] & L[b_]
                    if not v.any() or (v & bad).any():
                        continue
                    g_ = int((v & objgain).sum()) + int((((v[:, None] & cm).any(0)) & unc_u).sum())
                    if g_ == 0:
                        continue
                    cand = (g_, -2, v, (int(a_), int(b_)), ai)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
        if best is None:
            return None
        gain, nl, vec, li, ai = best
        rules.append((tuple(names[x] for x in li), ACTIONS[ai]))
        nlit += -nl
        adm = A[:, ai]
        unc_o &= ~(vec & adm)
        unc_u &= ~(((vec & adm)[:, None] & CM[:, ai, :]).any(0))
    if unc_o.any():
        return None
    rs = {"mode": mode, "rules": rules, "cost": len(rules) + nlit}
    for gi, go in pairs:
        pr = apply_rules(gi, rs)
        if pr is None or pr.shape != go.shape or not np.array_equal(pr, go):
            return None
    return rs


def apply_rules(g, rs):
    ctx = Ctx(g, rs["mode"])
    if not ctx.ok:
        return None
    lits = literals(ctx)
    lits.update({"not " + k: ~v for k, v in list(lits.items())})
    relcache = {r: rel_colour(ctx, r) for r in RELS}
    concrete = [None] * ctx.n
    taken = np.zeros(ctx.n, bool)
    for names, act in rs["rules"]:
        v = np.ones(ctx.n, bool)
        for nm in names:
            if nm not in lits:
                return None
            v &= lits[nm]
        for i in np.nonzero(v & ~taken)[0]:
            a2 = resolve(ctx, i, act, relcache)
            if a2 is not None:
                concrete[i] = a2
                taken[i] = True
    return render(ctx, concrete)


def solve(task):
    """Returns (rule_sets, predictions) where predictions[t] is a list of up to two output grids."""
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    if not all(a.shape == b.shape for a, b in pairs):
        return [], [[] for _ in task["test"]]
    found = []
    for mode in MODES:
        try:
            rs = induce(pairs, mode)
        except Exception:
            rs = None
        if rs:
            found.append(rs)
    found.sort(key=lambda r: r["cost"])
    preds = []
    for t in task["test"]:
        g = np.array(t["input"])
        seen, outl = [], []
        for rs in found:
            try:
                o = apply_rules(g, rs)
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


# ----------------------------------------------------------------------------- crop selection (output smaller than input)
TRANS = [("id", lambda a: a), ("fliplr", lambda a: a[:, ::-1]), ("flipud", lambda a: a[::-1]), ("rot90", lambda a: np.rot90(a, 1)),
         ("rot180", lambda a: np.rot90(a, 2)), ("rot270", lambda a: np.rot90(a, 3)), ("transpose", lambda a: a.T)]


def _crops(ctx, o):
    r0, r1, c0, c1 = o.bbox
    box = ctx.g[r0:r1 + 1, c0:c1 + 1]
    only = np.where(o.mask[r0:r1 + 1, c0:c1 + 1], box, ctx.bg)
    d = {}
    for nm, arr in (("box", box), ("only", only)):
        for tn, f in TRANS:
            d[nm if tn == "id" else nm + "+" + tn] = f(arr)
    return d


def induce_crop(pairs, mode):
    ctxs = []
    for gi, go in pairs:
        c = Ctx(gi, mode)
        if not c.ok:
            return None
        ctxs.append(c)
    lits = []
    for c in ctxs:
        l = literals(c)
        l.update({"not " + k: ~v for k, v in list(l.items())})
        lits.append(l)
    names = list(lits[0].keys())
    kinds = [k for k in _crops(ctxs[0], ctxs[0].objs[0])]
    for kind in kinds:
        good = []
        for c, (gi, go) in zip(ctxs, pairs):
            good.append(np.array([_crops(c, o)[kind].shape == go.shape and np.array_equal(_crops(c, o)[kind], go) for o in c.objs]))
        if not all(g.any() for g in good):
            continue
        for combo in [(n,) for n in names] + [(a, b) for a, b in itertools.combinations(names, 2)]:
            ok = True
            for l, g in zip(lits, good):
                v = np.ones(len(g), bool)
                for nm in combo:
                    v &= l[nm]
                if not v.any() or (v & ~g).any():
                    ok = False
                    break
            if ok:
                return {"mode": mode, "kind": kind, "lits": combo, "cost": len(combo) + (0 if "+" not in kind else 1)}
    return None


def apply_crop(g, rs):
    ctx = Ctx(g, rs["mode"])
    if not ctx.ok:
        return []
    l = literals(ctx)
    l.update({"not " + k: ~v for k, v in list(l.items())})
    v = np.ones(ctx.n, bool)
    for nm in rs["lits"]:
        v &= l[nm]
    return [_crops(ctx, ctx.objs[i])[rs["kind"]] for i in np.nonzero(v)[0]]


def solve_crop(task):
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    if all(a.shape == b.shape for a, b in pairs) or any(b.size >= a.size for a, b in pairs):
        return [], [[] for _ in task["test"]]
    found = []
    for mode in MODES:
        try:
            rs = induce_crop(pairs, mode)
        except Exception:
            rs = None
        if rs:
            found.append(rs)
    found.sort(key=lambda r: r["cost"])
    preds = []
    for t in task["test"]:
        g = np.array(t["input"])
        seen, outl = [], []
        for rs in found:
            for o in apply_crop(g, rs):
                key = (o.shape, o.tobytes())
                if key in seen:
                    continue
                seen.append(key)
                outl.append(o.tolist())
        preds.append(outl[:2])
    return found, preds


def solve_any(task):
    f, p = solve(task)
    if f:
        return f, p
    return solve_crop(task)
