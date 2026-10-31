"""Conditional per-object primitives: apply an action to the objects that satisfy a predicate.

14 predicates x 5 actions = 70 primitives, named if_<predicate>_<action>.
Objects are single-colour 8-connected components. A per-grid object table is cached,
so evaluating many primitives on the same grid extracts objects only once.

Recolour and fill actions use an 'unused' colour; the solver's global colour map
turns it into the target colour afterwards.

Loaded by dsl.py when ARC_DSL=full (the default).
"""
import numpy as np
from scipy import ndimage

import dsl
import dsl_ext

_CACHE = {}


def _build(g):
    masks = dsl_ext.objs_col(g)
    objs = []
    shape_keys = []
    for m in masks:
        r0, r1, c0, c1 = dsl._bbox(m)
        crop = m[r0:r1, c0:c1]
        key = (crop.shape, crop.tobytes())
        shape_keys.append(key)
        color = int(g[m][0])
        objs.append({
            "mask": m, "size": int(m.sum()), "color": color,
            "bbox": (r0, r1, c0, c1),
            "border": dsl_ext.touches_border(m),
            "hole": bool(ndimage.binary_fill_holes(m).sum() > m.sum()),
            "rect": bool(crop.all()),
            "sym": bool(np.array_equal(crop, crop[:, ::-1])),
            "tall": (r1 - r0) > (c1 - c0),
            "key": key,
        })
    sizes = [o["size"] for o in objs]
    cols = dsl_ext.ranked_colors(g)
    return {
        "objs": objs,
        "max": max(sizes) if sizes else 0,
        "min": min(sizes) if sizes else 0,
        "mean": float(np.mean(sizes)) if sizes else 0.0,
        "keycount": {k: shape_keys.count(k) for k in set(shape_keys)},
        "most": cols[0] if cols else None,
        "least": cols[-1] if cols else None,
    }


def table(g):
    key = (g.shape, g.tobytes())
    t = _CACHE.get(key)
    if t is None:
        if len(_CACHE) > 4000:
            _CACHE.clear()
        t = _build(g)
        _CACHE[key] = t
    return t


PREDICATES = {
    "largest": lambda o, t: o["size"] == t["max"],
    "smallest": lambda o, t: o["size"] == t["min"],
    "border": lambda o, t: o["border"],
    "interior": lambda o, t: not o["border"],
    "hole": lambda o, t: o["hole"],
    "rect": lambda o, t: o["rect"] and o["size"] > 1,
    "pixel": lambda o, t: o["size"] == 1,
    "big": lambda o, t: o["size"] > t["mean"],
    "tall": lambda o, t: o["tall"],
    "unique": lambda o, t: t["keycount"][o["key"]] == 1,
    "repeated": lambda o, t: t["keycount"][o["key"]] > 1,
    "commonc": lambda o, t: o["color"] == t["most"],
    "rarec": lambda o, t: o["color"] == t["least"],
    "symm": lambda o, t: o["sym"] and o["size"] > 1,
}


def act_recolor(g, sel):
    out = g.copy()
    u = dsl_ext.unused(g)
    for o in sel:
        out[o["mask"]] = u
    return out


def act_remove(g, sel):
    out = g.copy()
    bg = dsl.bg_color(g)
    for o in sel:
        out[o["mask"]] = bg
    return out


def act_keep(g, sel):
    out = dsl_ext.bg_grid(g)
    for o in sel:
        out[o["mask"]] = g[o["mask"]]
    return out


def act_fillbox(g, sel):
    out = g.copy()
    u = dsl_ext.unused(g)
    bg = dsl.bg_color(g)
    for o in sel:
        r0, r1, c0, c1 = o["bbox"]
        sub = out[r0:r1, c0:c1]
        sub[sub == bg] = u
    return out


def act_flip(g, sel):
    out = g.copy()
    for o in sel:
        r0, r1, c0, c1 = o["bbox"]
        out[r0:r1, c0:c1] = g[r0:r1, c0:c1][:, ::-1]
    return out


ACTIONS = {"recolor": act_recolor, "remove": act_remove, "keep": act_keep,
           "fillbox": act_fillbox, "flip": act_flip}


def make(pred, act):
    def f(g):
        t = table(g)
        sel = [o for o in t["objs"] if pred(o, t)]
        if not sel:
            return None  # selecting nothing is a no-op
        out = act(g, sel)
        if out.shape == g.shape and np.array_equal(out, g):
            return None
        return out
    return f


def build():
    P = {}
    # (rect, pixel) x (fillbox, flip) can never change a grid, so they are left out.
    dead = {("rect", "fillbox"), ("rect", "flip"), ("pixel", "fillbox"), ("pixel", "flip")}
    for pn, pf in PREDICATES.items():
        for an, af in ACTIONS.items():
            if (pn, an) in dead:
                continue
            P[f"if_{pn}_{an}"] = make(pf, af)
    return P


EXTRA = build()
