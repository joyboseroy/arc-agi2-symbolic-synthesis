"""Rule-structure learning for same-size ARC tasks: a decision tree over per-cell features whose leaves copy a colour from a source.

For every cell the solver knows a set of candidate sources (its own colour, a neighbour, the nearest coloured cell along a ray, a mirrored cell,
a constant, ...). A leaf may use a source only if the source gives the right output colour for every training cell that reaches the leaf.
The tree structure (which tests, in which order, which source at each leaf) is learned from the training pairs of the task, then checked by
exact reproduction of every training output. Smaller trees are preferred by iterative deepening.
"""
import itertools
import numpy as np
from scipy import ndimage

DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
DN = ["u", "d", "l", "r", "ul", "ur", "dl", "dr"]


def _shift(g, dy, dx, fill=-1):
    H, W = g.shape
    out = np.full_like(g, fill)
    ys, yd = (slice(max(0, -dy), H - max(0, dy)), slice(max(0, dy), H - max(0, -dy)))
    xs, xd = (slice(max(0, -dx), W - max(0, dx)), slice(max(0, dx), W - max(0, -dx)))
    out[yd, xd] = g[ys, xs]
    return out


def _ray(g, bg, dy, dx):
    """nearest non-bg colour and distance along direction; (-1, 0) if none."""
    H, W = g.shape
    col = np.full((H, W), -1)
    dist = np.zeros((H, W), int)
    ys = range(H) if dy <= 0 else range(H - 1, -1, -1)
    xs = range(W) if dx <= 0 else range(W - 1, -1, -1)
    for y in ys:
        for x in xs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if g[ny, nx] != bg:
                    col[y, x], dist[y, x] = g[ny, nx], 1
                elif col[ny, nx] >= 0:
                    col[y, x], dist[y, x] = col[ny, nx], dist[ny, nx] + 1
    return col, dist


def features(g):
    """Returns (cat, num, src): dicts of flat arrays. cat: colour-valued features, num: numeric features, src: candidate source colours."""
    H, W = g.shape
    vals, cnt = np.unique(g, return_counts=True)
    bg = int(vals[cnt.argmax()])
    cat, num, src = {}, {}, {}
    cat["self"] = g
    src["self"] = g
    for (dy, dx), n in zip(DIRS8, DN):
        nb = _shift(g, dy, dx)
        cat["nb_" + n] = nb
        src["nb_" + n] = np.where(nb >= 0, nb, g)
        col, dist = _ray(g, bg, dy, dx)
        cat["ray_" + n] = col
        num["rayd_" + n] = dist
        src["ray_" + n] = np.where(col >= 0, col, g)
    for (dy, dx), n in zip(DIRS8[:4], DN[:4]):
        cat["nb2_" + n] = _shift(g, 2 * dy, 2 * dx)
    cat["mir_h"] = g[:, ::-1]
    cat["mir_v"] = g[::-1, :]
    cat["mir_hv"] = g[::-1, ::-1]
    for k in ("mir_h", "mir_v", "mir_hv"):
        src[k] = cat[k]
    if H == W:
        cat["mir_t"] = g.T
        src["mir_t"] = g.T
    nonbg = vals[vals != bg]
    if len(nonbg):
        c_nb = cnt[vals != bg]
        src["most"] = np.full((H, W), int(nonbg[c_nb.argmax()]))
        src["least"] = np.full((H, W), int(nonbg[c_nb.argmin()]))
    else:
        src["most"] = src["least"] = np.full((H, W), bg)
    cat["bg"] = np.full((H, W), bg)
    rr, cc = np.indices((H, W))
    num["row"], num["col"] = rr, cc
    num["rrow"], num["rcol"] = H - 1 - rr, W - 1 - cc
    num["row_mod2"], num["col_mod2"] = rr % 2, cc % 2
    num["row_mod3"], num["col_mod3"] = rr % 3, cc % 3
    num["sum_mod2"] = (rr + cc) % 2
    isbg = g == bg
    num["rowcnt"] = np.repeat((~isbg).sum(1)[:, None], W, 1)
    num["colcnt"] = np.repeat((~isbg).sum(0)[None, :], H, 0)
    # neighbourhood counts
    for k in range(10):
        m = (g == k).astype(int)
        num[f"n3_{k}"] = ndimage.convolve(m, np.ones((3, 3), int), mode="constant") - m
    num["n3_nonbg"] = ndimage.convolve((~isbg).astype(int), np.ones((3, 3), int), mode="constant") - (~isbg)
    # objects (same colour, 8-connected)
    osize = np.zeros((H, W), int)
    oborder = np.zeros((H, W), int)
    obb_edge = np.zeros((H, W), int)
    ohole = np.zeros((H, W), int)
    olarge = np.zeros((H, W), int)
    osmall = np.zeros((H, W), int)
    otop = np.zeros((H, W), int)
    oleft = np.zeros((H, W), int)
    obot = np.zeros((H, W), int)
    oright = np.zeros((H, W), int)
    sizes = []
    comps = []
    for k in nonbg:
        lab, n = ndimage.label(g == k, structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            m = lab == i
            comps.append(m)
            sizes.append(int(m.sum()))
    mx = max(sizes) if sizes else 0
    mn = min(sizes) if sizes else 0
    for m, sz in zip(comps, sizes):
        ys, xs = np.nonzero(m)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        osize[m] = sz
        oborder[m] = int(r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1)
        sub = m[r0:r1 + 1, c0:c1 + 1]
        hole = ndimage.binary_fill_holes(sub) & ~sub
        ohole[m] = int(hole.any())
        olarge[m] = int(sz == mx)
        osmall[m] = int(sz == mn)
        otop[m] = (np.arange(H)[:, None] == r0).repeat(W, 1)[m]
        obot[m] = (np.arange(H)[:, None] == r1).repeat(W, 1)[m]
        oleft[m] = (np.arange(W)[None, :] == c0).repeat(H, 0)[m]
        oright[m] = (np.arange(W)[None, :] == c1).repeat(H, 0)[m]
        bb = np.zeros((H, W), bool)
        bb[r0:r1 + 1, c0:c1 + 1] = True
        # cells of this object lying on its bbox edge
    num["osize"], num["oborder"], num["ohole"], num["olarge"], num["osmall"] = osize, oborder, ohole, olarge, osmall
    num["otop"], num["obot"], num["oleft"], num["oright"] = otop, obot, oleft, oright
    # enclosed background and bg component size
    lab, n = ndimage.label(isbg)
    enc = np.zeros((H, W), int)
    bsz = np.zeros((H, W), int)
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        touch = ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1
        enc[m] = 0 if touch else 1
        bsz[m] = int(m.sum())
    num["enclosed"], num["bgsize"] = enc, bsz
    # nearest non-background cell in any direction, colour of enclosing object for enclosed background, colours of the largest / smallest object
    if (~isbg).any():
        dist_, (iy, ix) = ndimage.distance_transform_edt(isbg, return_indices=True)
        near = g[iy, ix]
        cat["near_col"] = near
        src["near_col"] = near
        num["near_dist"] = dist_.astype(int)
    else:
        cat["near_col"] = g
        src["near_col"] = g
        num["near_dist"] = np.zeros((H, W), int)
    bcol = np.full((H, W), -1)
    for i in range(1, n + 1):
        m = lab == i
        ring = ndimage.binary_dilation(m, structure=np.ones((3, 3))) & ~m
        rv = g[ring]
        if len(rv):
            bcol[m] = np.bincount(rv, minlength=10).argmax()
    cat["bgcomp_col"] = bcol
    src["bgcomp_col"] = np.where(bcol >= 0, bcol, g)
    if comps:
        big = comps[int(np.argmax(sizes))]
        small = comps[int(np.argmin(sizes))]
        src["largest_col"] = np.full((H, W), int(g[big][0]))
        src["smallest_col"] = np.full((H, W), int(g[small][0]))
    else:
        src["largest_col"] = src["smallest_col"] = np.full((H, W), bg)
    # relations with the cell's own colour
    for k in list(cat):
        if k != "self":
            num["eq_" + k] = (cat[k] == g).astype(int)
    flat = lambda d: {k: np.asarray(v).reshape(-1) for k, v in d.items()}
    consts = {f"const{c}": np.full(H * W, c) for c in range(10)}
    s = flat(src)
    s.update(consts)
    return flat(cat), flat(num), s


SRC_PRIORITY = ["self"] + [f"const{c}" for c in range(10)] + ["nb_" + n for n in DN] + ["ray_" + n for n in DN] + ["mir_h", "mir_v", "mir_hv", "mir_t", "most", "least", "near_col", "bgcomp_col", "largest_col", "smallest_col"]
POSITION = {"row", "col", "rrow", "rcol", "row_mod2", "col_mod2", "row_mod3", "col_mod3", "sum_mod2"}


def build_predicates(cats, nums, allow_position=True, allow_ray=True):
    """Predicate list built from the training features: (kind, name, value)."""
    preds = []
    for k in cats[0]:
        if not allow_ray and (k.startswith("ray_") or k.startswith("nb2_")):
            continue
        vs = sorted(set(int(v) for c in cats for v in np.unique(c[k])))
        if len(vs) > 12:
            continue
        for v in vs:
            preds.append(("eq", k, v))
    for k in nums[0]:
        if not allow_position and k in POSITION:
            continue
        if not allow_ray and k.startswith("rayd_"):
            continue
        vs = sorted(set(int(v) for n in nums for v in np.unique(n[k])))
        if len(vs) == 1:
            continue
        if len(vs) > 10:
            qs = np.unique(np.quantile(np.concatenate([n[k] for n in nums]), np.linspace(0.1, 0.9, 9)).astype(int))
            vs = list(qs)
        for v in vs[:-1]:
            preds.append(("le", k, int(v)))
    return preds


def eval_preds(preds, cat, num):
    cols = []
    for kind, k, v in preds:
        a = cat[k] if kind == "eq" and k in cat else num[k] if k in num else None
        if a is None:
            cols.append(np.zeros(len(next(iter(cat.values()))), bool))
            continue
        cols.append(a == v if kind == "eq" else a <= v)
    return np.stack(cols, 1)


class Fail(Exception):
    pass


def grow(F, M, idx, depth, node_budget):
    """Returns a tree: ('leaf', source_index) or ('split', pred_index, tree_false, tree_true). Raises Fail."""
    sub = M[idx]
    full = sub.all(0)
    if full.any():
        return ("leaf", int(np.nonzero(full)[0][0]))
    if depth == 0:
        raise Fail
    P = F[idx]
    T = len(idx)
    C1 = P.T.astype(np.int32) @ sub.astype(np.int32)
    C = sub.sum(0).astype(np.int32)
    C0 = C[None, :] - C1
    n1 = P.sum(0)
    valid = (n1 > 0) & (n1 < T)
    score = np.where(valid, C1.max(1) + C0.max(1), -1)
    order = np.argsort(-score, kind="stable")[:3]
    for f in order:
        if score[f] < 0:
            break
        if score[f] <= C.max():
            break
        try:
            t = grow(F, M, idx[~P[:, f]], depth - 1, node_budget)
            tt = grow(F, M, idx[P[:, f]], depth - 1, node_budget)
            return ("split", int(f), t, tt)
        except Fail:
            continue
    raise Fail


def tree_size(t):
    return 1 if t[0] == "leaf" else 1 + tree_size(t[2]) + tree_size(t[3])


def predict_tree(t, F, S, srcnames):
    n = F.shape[0]
    out = np.zeros(n, int)

    def go(t, idx):
        if len(idx) == 0:
            return
        if t[0] == "leaf":
            out[idx] = S[srcnames[t[1]]][idx]
        else:
            m = F[idx, t[1]]
            go(t[2], idx[~m])
            go(t[3], idx[m])
    go(t, np.arange(n))
    return out


def fit_task(pairs, allow_position=True, max_depth=5, allow_ray=True):
    feats = [features(gi) for gi, _ in pairs]
    cats = [f[0] for f in feats]
    nums = [f[1] for f in feats]
    srcs = [f[2] for f in feats]
    preds = build_predicates(cats, nums, allow_position, allow_ray)
    F = np.concatenate([eval_preds(preds, c, n) for c, n in zip(cats, nums)], 0)
    srcnames = [s for s in SRC_PRIORITY if all(s in sd for sd in srcs)]
    Y = np.concatenate([go.reshape(-1) for _, go in pairs])
    M = np.stack([np.concatenate([sd[s] for sd in srcs]) == Y for s in srcnames], 1)
    if M[:, 0].all():
        return None  # identity
    idx = np.arange(len(Y))
    for d in range(1, max_depth + 1):
        try:
            t = grow(F, M, idx, d, 0)
        except Fail:
            continue
        return {"tree": t, "preds": preds, "srcnames": srcnames, "size": tree_size(t), "depth": d, "pos": allow_position, "ray": allow_ray}
    return None


def apply_tree(g, m):
    cat, num, src = features(g)
    for s in m["srcnames"]:
        if s not in src:
            return None
    F = eval_preds(m["preds"], cat, num)
    out = predict_tree(m["tree"], F, src, m["srcnames"])
    return out.reshape(g.shape)


def solve(task):
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    if not all(a.shape == b.shape for a, b in pairs):
        return [], [[] for _ in task["test"]]
    found = []
    for pos, ray in ((True, True), (False, True), (False, False), (True, False)):
        try:
            m = fit_task(pairs, pos, 5, ray)
        except Exception:
            m = None
        if m is None:
            continue
        ok = all(np.array_equal(apply_tree(gi, m), go) for gi, go in pairs)
        if not ok:
            continue
        # leave-one-pair-out check
        loo = 0
        if len(pairs) >= 3:
            for k in range(len(pairs)):
                tr = pairs[:k] + pairs[k + 1:]
                try:
                    mk = fit_task(tr, pos, 5, ray)
                    loo += int(mk is not None and np.array_equal(apply_tree(pairs[k][0], mk), pairs[k][1]))
                except Exception:
                    pass
            m["loo"] = loo
        else:
            m["loo"] = -1
        found.append(m)
    found.sort(key=lambda m: (-(m["loo"] if m["loo"] >= 0 else 0), m["size"], m["depth"]))
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
            key = o.tobytes()
            if key in seen:
                continue
            seen.append(key)
            outl.append(o.tolist())
        preds.append(outl[:2])
    return found, preds
