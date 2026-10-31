"""Extended object-level primitives (about 90) added on top of dsl.py.

Colour-valued outputs never hard-code a colour: fills use an 'unused' colour
(first of 1..9 absent from the grid) because the solver learns a global colour
map afterwards. Colour selectors are by rank (most common non-background colour,
least common, second most common).

Loaded by dsl.py when the environment variable ARC_DSL is not 'small'.
"""
import numpy as np
from scipy import ndimage

import dsl

EIGHT = np.ones((3, 3))


# ---------- helpers ----------
def unused(g):
    present = set(np.unique(g).tolist())
    for c in range(1, 10):
        if c not in present:
            return c
    return 9


def ranked_colors(g):
    bg = dsl.bg_color(g)
    vals, counts = np.unique(g[g != bg], return_counts=True)
    order = sorted(zip(counts.tolist(), vals.tolist()), key=lambda t: (-t[0], t[1]))
    return [c for _, c in order]


def pick(g, which):
    cols = ranked_colors(g)
    if not cols:
        return None
    if which == "most":
        return cols[0]
    if which == "least":
        return cols[-1]
    return cols[1] if len(cols) > 1 else None


def objs_any(g):
    return dsl._objects(g, diag=True, by_color=False)


def objs_col(g):
    return dsl._objects(g, diag=True, by_color=True)


def bg_grid(g):
    return np.full_like(g, dsl.bg_color(g))


def touches_border(m):
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


# ---------- colour selectors ----------
def make_keep(which):
    def f(g):
        c = pick(g, which)
        if c is None:
            return None
        out = bg_grid(g)
        out[g == c] = c
        return out
    return f


def make_remove(which):
    def f(g):
        c = pick(g, which)
        if c is None:
            return None
        out = g.copy()
        out[g == c] = dsl.bg_color(g)
        return out
    return f


def make_crop(which):
    def f(g):
        c = pick(g, which)
        if c is None:
            return None
        b = dsl._bbox(g == c)
        r0, r1, c0, c1 = b
        return g[r0:r1, c0:c1]
    return f


# ---------- object recolor / selection ----------
def recolor_largest_obj(g):
    o = objs_col(g)
    if not o:
        return None
    out = g.copy()
    out[max(o, key=lambda m: m.sum())] = unused(g)
    return out


def recolor_smallest_obj(g):
    o = objs_col(g)
    if not o:
        return None
    out = g.copy()
    out[min(o, key=lambda m: m.sum())] = unused(g)
    return out


def recolor_by_size(g):
    o = objs_col(g)
    if not o:
        return None
    sizes = sorted(set(int(m.sum()) for m in o))
    out = g.copy()
    for m in o:
        out[m] = 1 + sizes.index(int(m.sum())) % 9
    return out


def recolor_has_hole(g):
    o = objs_col(g)
    out = g.copy()
    hit = False
    u = unused(g)
    for m in o:
        if ndimage.binary_fill_holes(m).sum() > m.sum():
            out[m] = u
            hit = True
    return out if hit else None


def recolor_touching_border(g):
    u = unused(g)
    out = g.copy()
    hit = False
    for m in objs_col(g):
        if touches_border(m):
            out[m] = u
            hit = True
    return out if hit else None


def remove_largest_obj(g):
    o = objs_any(g)
    if len(o) < 2:
        return None
    out = g.copy()
    out[max(o, key=lambda m: m.sum())] = dsl.bg_color(g)
    return out


def keep_smallest_obj(g):
    o = objs_any(g)
    if len(o) < 2:
        return None
    out = bg_grid(g)
    m = min(o, key=lambda m: m.sum())
    out[m] = g[m]
    return out


def remove_border_objs(g):
    out = g.copy()
    hit = False
    for m in objs_any(g):
        if touches_border(m):
            out[m] = dsl.bg_color(g)
            hit = True
    return out if hit else None


def keep_border_objs(g):
    out = bg_grid(g)
    hit = False
    for m in objs_any(g):
        if touches_border(m):
            out[m] = g[m]
            hit = True
    return out if hit else None


# ---------- rigid object motion ----------
def _fall_down(g):
    bg = dsl.bg_color(g)
    H = g.shape[0]
    objs = sorted(objs_any(g), key=lambda m: -np.where(m)[0].max())
    out = g.copy()
    for m in objs:
        cells = np.argwhere(m)
        out[m] = bg  # lift
        shift = 0
        while True:
            nr = cells[:, 0] + shift + 1
            if nr.max() >= H or any(out[r, c] != bg for r, c in zip(nr, cells[:, 1])):
                break
            shift += 1
        for (r, c) in cells:
            out[r + shift, c] = g[r, c]
    return out


def make_fall(k):
    def f(g):
        return np.rot90(_fall_down(np.rot90(g, k)), -k)
    return f


def make_shift(dr, dc):
    def f(g):
        out = bg_grid(g)
        H, W = g.shape
        rs, re_ = max(dr, 0), H + min(dr, 0)
        cs, ce = max(dc, 0), W + min(dc, 0)
        if re_ <= rs or ce <= cs:
            return None
        out[rs:re_, cs:ce] = g[rs - dr:re_ - dr, cs - dc:ce - dc]
        return out
    return f


def make_roll(dr, dc):
    def f(g):
        return np.roll(g, (dr, dc), axis=(0, 1))
    return f


# ---------- line drawing ----------
def row_through_pixels(g):
    bg = dsl.bg_color(g)
    out = g.copy()
    for r, c in np.argwhere(g != bg):
        row = out[r]
        row[row == bg] = g[r, c]
    return out


def col_through_pixels(g):
    return row_through_pixels(g.T).T


def cross_through_pixels(g):
    return col_through_pixels(row_through_pixels(g))


def diag_rays(g):
    bg = dsl.bg_color(g)
    out = g.copy()
    H, W = g.shape
    for r, c in np.argwhere(g != bg):
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            rr, cc = r + dr, c + dc
            while 0 <= rr < H and 0 <= cc < W:
                if out[rr, cc] == bg:
                    out[rr, cc] = g[r, c]
                rr += dr
                cc += dc
    return out


def connect_h(g):
    bg = dsl.bg_color(g)
    out = g.copy()
    hit = False
    for r in range(g.shape[0]):
        cols = [c for c in range(g.shape[1]) if g[r, c] != bg]
        for a, b in zip(cols, cols[1:]):
            if g[r, a] == g[r, b] and b - a > 1 and (g[r, a + 1:b] == bg).all():
                out[r, a + 1:b] = g[r, a]
                hit = True
    return out if hit else None


def connect_v(g):
    r = connect_h(g.T)
    return None if r is None else r.T


def connect_hv(g):
    a = connect_h(g)
    base = g if a is None else a
    b = connect_v(base)
    out = base if b is None else b
    return None if (a is None and b is None) else out


def extend_left(g):
    return dsl.extend_lines_h(g[:, ::-1])[:, ::-1]


def extend_up(g):
    return dsl.extend_lines_v(g[::-1, :])[::-1, :]


# ---------- fills and borders ----------
def fill_holes_unused(g):
    bg = dsl.bg_color(g)
    lab, n = ndimage.label(g == bg)
    H, W = g.shape
    out = g.copy()
    u = unused(g)
    hit = False
    for i in range(1, n + 1):
        m = lab == i
        rr, cc = np.where(m)
        if rr.min() == 0 or cc.min() == 0 or rr.max() == H - 1 or cc.max() == W - 1:
            continue
        out[m] = u
        hit = True
    return out if hit else None


def fill_bbox_unused(g):
    bg = dsl.bg_color(g)
    out = g.copy()
    u = unused(g)
    hit = False
    for m in objs_any(g):
        r0, r1, c0, c1 = dsl._bbox(m)
        sub = out[r0:r1, c0:c1]
        mask = sub == bg
        if mask.any():
            sub[mask] = u
            hit = True
    return out if hit else None


def add_border(g):
    u = unused(g)
    if g.shape[0] + 2 > dsl.MAXDIM or g.shape[1] + 2 > dsl.MAXDIM:
        return None
    return np.pad(g, 1, constant_values=u)


def remove_border(g):
    if min(g.shape) <= 2:
        return None
    return g[1:-1, 1:-1]


def frame_grid(g):
    out = g.copy()
    u = unused(g)
    out[0, :] = u
    out[-1, :] = u
    out[:, 0] = u
    out[:, -1] = u
    return out


# ---------- boolean ops on two halves ----------
def _halves(g):
    w = g.shape[1]
    if w < 2:
        return None
    h = w // 2
    bg = dsl.bg_color(g)
    return g[:, :h] != bg, g[:, w - h:] != bg


def make_bool(op):
    def f(g):
        hv = _halves(g)
        if hv is None:
            return None
        a, b = hv
        m = {"and": a & b, "or": a | b, "xor": a ^ b, "nor": ~(a | b)}[op]
        return m.astype(g.dtype)
    return f


def make_bool_v(op):
    h = make_bool(op)

    def f(g):
        r = h(g.T)
        return None if r is None else r.T
    return f


# ---------- grid-of-cells ----------
def _cells(g):
    """Split g into cells by full-length separator lines of one colour.

    Every colour that forms at least one full row or column is a candidate; the
    best candidate has equal-shaped cells, then the most cells.
    """
    H, W = g.shape
    best, best_key = None, None
    for c in np.unique(g):
        rs = [r for r in range(H) if (g[r] == c).all()]
        cs = [k for k in range(W) if (g[:, k] == c).all()]
        if not rs and not cs:
            continue

        def segs(sep, n):
            out, start = [], 0
            for s in sep + [n]:
                if s > start:
                    out.append((start, s))
                start = s + 1
            return out
        rsegs, csegs = segs(rs, H), segs(cs, W)
        if len(rsegs) * len(csegs) < 2:
            continue
        cells = [g[a:b, c0:c1] for a, b in rsegs for c0, c1 in csegs]
        key = (len({x.shape for x in cells}) == 1, len(cells))
        if best_key is None or key > best_key:
            best, best_key = cells, key
    return best


def _nonbg_count(cell, g):
    return int((cell != dsl.bg_color(g)).sum())


def cell_unique(g):
    cells = _cells(g)
    if not cells:
        return None
    keys = [c.tobytes() + bytes(c.shape) for c in cells]
    uniq = [c for c, k in zip(cells, keys) if keys.count(k) == 1]
    return uniq[0] if len(uniq) == 1 else None


def cell_most_filled(g):
    cells = _cells(g)
    return max(cells, key=lambda c: _nonbg_count(c, g)) if cells else None


def cell_least_filled(g):
    cells = _cells(g)
    return min(cells, key=lambda c: _nonbg_count(c, g)) if cells else None


def cells_overlay(g):
    cells = _cells(g)
    if not cells or len({c.shape for c in cells}) != 1:
        return None
    bg = dsl.bg_color(g)
    out = np.full_like(cells[0], bg)
    for c in cells:
        out = np.where(c != bg, c, out)
    return out


# ---------- tiling and concatenation ----------
def hcat_flipud(g): return np.hstack([g, g[::-1, :]])
def vcat_fliplr(g): return np.vstack([g, g[:, ::-1]])
def hcat_rot180(g): return np.hstack([g, np.rot90(g, 2)])
def vcat_rot180(g): return np.vstack([g, np.rot90(g, 2)])


def rot_tile4(g):
    if g.shape[0] != g.shape[1]:
        return None
    top = np.hstack([g, np.rot90(g, 3)])
    bot = np.hstack([np.rot90(g, 1), np.rot90(g, 2)])
    return np.vstack([top, bot])


def tile_flip_alt(g):
    top = np.hstack([g, g[:, ::-1]])
    bot = np.hstack([g[::-1, :], g])
    return np.vstack([top, bot])


def kron_self(g):
    bg = dsl.bg_color(g)
    m = (g != bg).astype(g.dtype)
    out = np.kron(m, g)
    return np.where(np.kron(m, np.ones_like(g)) == 1, out, bg)


def kron_self_inv(g):
    bg = dsl.bg_color(g)
    m = (g == bg).astype(g.dtype)
    out = np.kron(m, g)
    return np.where(np.kron(m, np.ones_like(g)) == 1, out, bg)


def up4(g): return np.kron(g, np.ones((4, 4), dtype=g.dtype))


def down4(g):
    if g.shape[0] % 4 or g.shape[1] % 4:
        return None
    return g[::4, ::4]


def up2_h(g): return np.repeat(g, 2, axis=1)
def up2_v(g): return np.repeat(g, 2, axis=0)


def down2_h(g):
    return g[:, ::2] if g.shape[1] % 2 == 0 else None


def down2_v(g):
    return g[::2, :] if g.shape[0] % 2 == 0 else None


# ---------- summaries ----------
def largest_obj_color_1x1(g):
    o = objs_col(g)
    if not o:
        return None
    m = max(o, key=lambda m: m.sum())
    return np.array([[int(g[m][0])]])


def n_objects_row(g):
    n = len(objs_any(g))
    c = pick(g, "most")
    if n == 0 or n > dsl.MAXDIM or c is None:
        return None
    return np.full((1, n), c, dtype=g.dtype)


def n_colors_row(g):
    k = len(ranked_colors(g))
    c = pick(g, "most")
    if k == 0 or c is None:
        return None
    return np.full((1, k), c, dtype=g.dtype)


# ---------- denoise / morphology ----------
def majority3(g):
    H, W = g.shape
    out = g.copy()
    p = np.pad(g, 1, mode="edge")
    for r in range(H):
        for c in range(W):
            out[r, c] = np.bincount(p[r:r + 3, c:c + 3].ravel(), minlength=10).argmax()
    return out


def dilate(g):
    bg = dsl.bg_color(g)
    out = g.copy()
    H, W = g.shape
    for r, c in np.argwhere(g != bg):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and g[rr, cc] == bg:
                out[rr, cc] = g[r, c]
    return out


def erode(g):
    bg = dsl.bg_color(g)
    nb = g != bg
    keep = ndimage.binary_erosion(nb, structure=ndimage.generate_binary_structure(2, 1), border_value=0)
    out = bg_grid(g)
    out[keep] = g[keep]
    return out


def inner_only(g):
    bg = dsl.bg_color(g)
    nb = g != bg
    keep = ndimage.binary_erosion(nb, structure=EIGHT, border_value=0)
    out = bg_grid(g)
    out[keep] = g[keep]
    return out


# ---------- symmetry completion ----------
def complete_rot4(g):
    if g.shape[0] != g.shape[1]:
        return None
    bg = dsl.bg_color(g)
    out = g.copy()
    for k in (1, 2, 3):
        r = np.rot90(g, k)
        out = np.where((out == bg) & (r != bg), r, out)
    return out


def complete_transpose(g):
    if g.shape[0] != g.shape[1]:
        return None
    bg = dsl.bg_color(g)
    return np.where((g == bg) & (g.T != bg), g.T, g)


# ---------- sorting / masks / colour swaps ----------
def sort_rows_by_count(g):
    bg = dsl.bg_color(g)
    order = np.argsort((g != bg).sum(axis=1), kind="stable")
    return g[order]


def sort_cols_by_count(g):
    return sort_rows_by_count(g.T).T


def swap_bg_fg(g):
    cols = ranked_colors(g)
    if len(cols) != 1:
        return None
    bg, fg = dsl.bg_color(g), cols[0]
    out = g.copy()
    out[g == bg] = fg
    out[g == fg] = bg
    return out


def binarize(g):
    return (g != dsl.bg_color(g)).astype(g.dtype)


def swap_most_least(g):
    cols = ranked_colors(g)
    if len(cols) < 2:
        return None
    a, b = cols[0], cols[-1]
    out = g.copy()
    out[g == a] = b
    out[g == b] = a
    return out


# ---------- thirds ----------
def left_third(g): return g[:, : g.shape[1] // 3] if g.shape[1] >= 3 else None
def right_third(g): return g[:, g.shape[1] - g.shape[1] // 3:] if g.shape[1] >= 3 else None
def top_third(g): return g[: g.shape[0] // 3, :] if g.shape[0] >= 3 else None
def bottom_third(g): return g[g.shape[0] - g.shape[0] // 3:, :] if g.shape[0] >= 3 else None


# ---------- object-wise ----------
def bbox_outline(g):
    out = g.copy()
    u = unused(g)
    H, W = g.shape
    bg = dsl.bg_color(g)
    hit = False
    for m in objs_any(g):
        r0, r1, c0, c1 = dsl._bbox(m)
        r0, r1, c0, c1 = r0 - 1, r1 + 1, c0 - 1, c1 + 1
        for r in range(max(r0, 0), min(r1, H)):
            for c in range(max(c0, 0), min(c1, W)):
                on_edge = r in (r0, r1 - 1) or c in (c0, c1 - 1)
                if on_edge and out[r, c] == bg:
                    out[r, c] = u
                    hit = True
    return out if hit else None


def flip_each_obj_lr(g):
    out = g.copy()
    for m in objs_any(g):
        r0, r1, c0, c1 = dsl._bbox(m)
        out[r0:r1, c0:c1] = g[r0:r1, c0:c1][:, ::-1]
    return out


def flip_each_obj_ud(g):
    out = g.copy()
    for m in objs_any(g):
        r0, r1, c0, c1 = dsl._bbox(m)
        out[r0:r1, c0:c1] = g[r0:r1, c0:c1][::-1, :]
    return out


def build():
    P = {}
    for w in ("most", "least", "second"):
        P[f"keep_color_{w}"] = make_keep(w)
        P[f"remove_color_{w}"] = make_remove(w)
        P[f"crop_color_{w}"] = make_crop(w)
    P.update({
        "recolor_largest_obj": recolor_largest_obj, "recolor_smallest_obj": recolor_smallest_obj,
        "recolor_by_size": recolor_by_size, "recolor_has_hole": recolor_has_hole,
        "recolor_touching_border": recolor_touching_border,
        "remove_largest_obj": remove_largest_obj, "keep_smallest_obj": keep_smallest_obj,
        "remove_border_objs": remove_border_objs, "keep_border_objs": keep_border_objs,
    })
    for name, k in (("down", 0), ("left", 1), ("up", 2), ("right", 3)):
        P[f"obj_fall_{name}"] = make_fall(k)
    for name, d in (("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))):
        P[f"shift_{name}"] = make_shift(*d)
        P[f"roll_{name}"] = make_roll(*d)
    P.update({
        "row_through_pixels": row_through_pixels, "col_through_pixels": col_through_pixels,
        "cross_through_pixels": cross_through_pixels, "diag_rays": diag_rays,
        "connect_h": connect_h, "connect_v": connect_v, "connect_hv": connect_hv,
        "extend_left": extend_left, "extend_up": extend_up,
        "fill_holes_unused": fill_holes_unused, "fill_bbox_unused": fill_bbox_unused,
        "add_border": add_border, "remove_border": remove_border, "frame_grid": frame_grid,
    })
    for op in ("and", "or", "xor", "nor"):
        P[f"{op}_h"] = make_bool(op)
        P[f"{op}_v"] = make_bool_v(op)
    P.update({
        "cell_unique": cell_unique, "cell_most_filled": cell_most_filled,
        "cell_least_filled": cell_least_filled, "cells_overlay": cells_overlay,
        "hcat_flipud": hcat_flipud, "vcat_fliplr": vcat_fliplr,
        "hcat_rot180": hcat_rot180, "vcat_rot180": vcat_rot180,
        "rot_tile4": rot_tile4, "tile_flip_alt": tile_flip_alt,
        "kron_self": kron_self, "kron_self_inv": kron_self_inv,
        "up4": up4, "down4": down4,
        "up2_h": up2_h, "up2_v": up2_v, "down2_h": down2_h, "down2_v": down2_v,
        "largest_obj_color": largest_obj_color_1x1, "n_objects_row": n_objects_row,
        "n_colors_row": n_colors_row,
        "majority3": majority3, "dilate": dilate, "erode": erode, "inner_only": inner_only,
        "complete_rot4": complete_rot4, "complete_transpose": complete_transpose,
        "sort_rows": sort_rows_by_count, "sort_cols": sort_cols_by_count,
        "swap_bg_fg": swap_bg_fg, "binarize": binarize, "swap_most_least": swap_most_least,
        "left_third": left_third, "right_third": right_third,
        "top_third": top_third, "bottom_third": bottom_third,
        "bbox_outline": bbox_outline,
        "flip_each_obj_lr": flip_each_obj_lr, "flip_each_obj_ud": flip_each_obj_ud,
    })
    return P


EXTRA = build()
