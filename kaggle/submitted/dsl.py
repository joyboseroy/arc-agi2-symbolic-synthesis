"""Small grid DSL for an ARC-AGI-2 symbolic baseline.

Every primitive maps a 2D int numpy array to a 2D int numpy array, or returns
None when it does not apply. Primitives are pure and deterministic.
"""
import numpy as np
from scipy import ndimage

MAXDIM = 30


def bg_color(g):
    """Background = most frequent color."""
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return r0, r1 + 1, c0, c1 + 1


def _ok(g):
    return g is not None and g.ndim == 2 and 0 < g.shape[0] <= MAXDIM and 0 < g.shape[1] <= MAXDIM


# ---- geometric ----
def rot90(g): return np.rot90(g, 1)
def rot180(g): return np.rot90(g, 2)
def rot270(g): return np.rot90(g, 3)
def flipud(g): return g[::-1, :]
def fliplr(g): return g[:, ::-1]
def transpose(g): return g.T
def antitranspose(g): return np.rot90(g, 2).T


# ---- cropping ----
def crop_nonbg(g):
    b = _bbox(g != bg_color(g))
    if b is None:
        return None
    r0, r1, c0, c1 = b
    return g[r0:r1, c0:c1]


def _objects(g, diag=True, by_color=False):
    bg = bg_color(g)
    st = np.ones((3, 3)) if diag else None
    objs = []
    if by_color:
        for c in np.unique(g):
            if c == bg:
                continue
            lab, n = ndimage.label(g == c, structure=st)
            for i in range(1, n + 1):
                objs.append(lab == i)
    else:
        lab, n = ndimage.label(g != bg, structure=st)
        for i in range(1, n + 1):
            objs.append(lab == i)
    return objs


def _crop_obj(g, m):
    r0, r1, c0, c1 = _bbox(m)
    return g[r0:r1, c0:c1]


def crop_largest(g):
    objs = _objects(g)
    if not objs:
        return None
    return _crop_obj(g, max(objs, key=lambda m: m.sum()))


def crop_smallest(g):
    objs = _objects(g)
    if not objs:
        return None
    return _crop_obj(g, min(objs, key=lambda m: m.sum()))


def crop_largest_bycolor(g):
    objs = _objects(g, by_color=True)
    if not objs:
        return None
    return _crop_obj(g, max(objs, key=lambda m: m.sum()))


def crop_smallest_bycolor(g):
    objs = _objects(g, by_color=True)
    if not objs:
        return None
    return _crop_obj(g, min(objs, key=lambda m: m.sum()))


def crop_most_colorful(g):
    objs = _objects(g)
    if not objs:
        return None
    return _crop_obj(g, max(objs, key=lambda m: len(np.unique(g[m]))))


def keep_largest_obj(g):
    objs = _objects(g)
    if not objs:
        return None
    m = max(objs, key=lambda m: m.sum())
    out = np.full_like(g, bg_color(g))
    out[m] = g[m]
    return out


def remove_smallest_objs(g):
    """Remove single-pixel noise objects."""
    bg = bg_color(g)
    out = g.copy()
    for m in _objects(g):
        if m.sum() == 1:
            out[m] = bg
    return out


# ---- scaling / tiling ----
def up2(g): return np.kron(g, np.ones((2, 2), dtype=g.dtype))
def up3(g): return np.kron(g, np.ones((3, 3), dtype=g.dtype))


def down2(g):
    if g.shape[0] % 2 or g.shape[1] % 2:
        return None
    return g[::2, ::2]


def down3(g):
    if g.shape[0] % 3 or g.shape[1] % 3:
        return None
    return g[::3, ::3]


def tile2x2(g): return np.tile(g, (2, 2))
def tile3x3(g): return np.tile(g, (3, 3))


def mirror_h(g): return np.hstack([g, g[:, ::-1]])
def mirror_v(g): return np.vstack([g, g[::-1, :]])


def mirror_4(g):
    top = np.hstack([g, g[:, ::-1]])
    return np.vstack([top, top[::-1, :]])


def left_half(g):
    return g[:, : g.shape[1] // 2] if g.shape[1] > 1 else None


def right_half(g):
    return g[:, (g.shape[1] + 1) // 2:] if g.shape[1] > 1 else None


def top_half(g):
    return g[: g.shape[0] // 2, :] if g.shape[0] > 1 else None


def bottom_half(g):
    return g[(g.shape[0] + 1) // 2:, :] if g.shape[0] > 1 else None


# ---- gravity ----
def _gravity_down(g):
    bg = bg_color(g)
    out = np.full_like(g, bg)
    for c in range(g.shape[1]):
        col = g[:, c][g[:, c] != bg]
        if len(col):
            out[g.shape[0] - len(col):, c] = col
    return out


def gravity_down(g): return _gravity_down(g)
def gravity_up(g): return _gravity_down(g[::-1, :])[::-1, :]

def _grav(g, k):
    """Rotate so the target direction is 'down', apply, rotate back."""
    return np.rot90(_gravity_down(np.rot90(g, k)), -k)


def gravity_left(g): return _grav(g, 1)
def gravity_right2(g): return _grav(g, 3)


# ---- fills / symmetry ----
def fill_holes(g):
    """Fill enclosed background regions with color 10-sentinel -> pick 4 (yellow) by convention is wrong;
    instead fill with the most common non-bg color adjacent to the hole."""
    bg = bg_color(g)
    lab, n = ndimage.label(g == bg)
    out = g.copy()
    H, W = g.shape
    changed = False
    for i in range(1, n + 1):
        m = lab == i
        rr, cc = np.where(m)
        if rr.min() == 0 or cc.min() == 0 or rr.max() == H - 1 or cc.max() == W - 1:
            continue
        dil = ndimage.binary_dilation(m, structure=np.ones((3, 3))) & ~m
        neigh = g[dil]
        neigh = neigh[neigh != bg]
        if len(neigh) == 0:
            continue
        out[m] = int(np.bincount(neigh).argmax())
        changed = True
    return out if changed else None


def symmetrize_lr(g):
    bg = bg_color(g)
    m = g[:, ::-1]
    out = np.where(g == bg, m, g)
    return out


def symmetrize_ud(g):
    bg = bg_color(g)
    m = g[::-1, :]
    return np.where(g == bg, m, g)


def symmetrize_4(g):
    return symmetrize_ud(symmetrize_lr(g))


def outline_objs(g):
    """Keep only the border pixels of each object, hollowing interiors to bg."""
    bg = bg_color(g)
    nb = g != bg
    inner = ndimage.binary_erosion(nb, structure=np.ones((3, 3)), border_value=0)
    out = g.copy()
    out[inner] = bg
    return out


def extend_lines_h(g):
    """Extend every non-bg pixel rightwards until the grid edge or another pixel."""
    bg = bg_color(g)
    out = g.copy()
    for r in range(g.shape[0]):
        cur = None
        for c in range(g.shape[1]):
            if g[r, c] != bg:
                cur = g[r, c]
            elif cur is not None:
                out[r, c] = cur
    return out


def extend_lines_v(g):
    return extend_lines_h(g.T).T


def dedupe_rows_cols(g):
    """Collapse consecutive identical rows and columns."""
    rows = [0] + [i for i in range(1, g.shape[0]) if not np.array_equal(g[i], g[i - 1])]
    g2 = g[rows]
    cols = [0] + [j for j in range(1, g2.shape[1]) if not np.array_equal(g2[:, j], g2[:, j - 1])]
    return g2[:, cols]


def most_common_color_1x1(g):
    bg = bg_color(g)
    nb = g[g != bg]
    if len(nb) == 0:
        return None
    return np.array([[int(np.bincount(nb).argmax())]])


def least_common_color_1x1(g):
    vals, counts = np.unique(g, return_counts=True)
    if len(vals) < 2:
        return None
    return np.array([[int(vals[np.argmin(counts)])]])


def split_overlay_h(g):
    """Overlay left/right halves (separator column dropped if width is odd); right wins on non-bg."""
    w = g.shape[1]
    if w < 3:
        return None
    h = w // 2
    a, b = g[:, :h], g[:, w - h:]
    bg = bg_color(g)
    return np.where(b != bg, b, a)


def split_overlay_v(g):
    r = split_overlay_h(g.T)
    return None if r is None else r.T


PRIMITIVES = {
    "rot90": rot90, "rot180": rot180, "rot270": rot270,
    "flipud": flipud, "fliplr": fliplr, "transpose": transpose, "antitranspose": antitranspose,
    "crop_nonbg": crop_nonbg, "crop_largest": crop_largest, "crop_smallest": crop_smallest,
    "crop_largest_bycolor": crop_largest_bycolor, "crop_smallest_bycolor": crop_smallest_bycolor,
    "crop_most_colorful": crop_most_colorful,
    "keep_largest_obj": keep_largest_obj, "remove_noise": remove_smallest_objs,
    "up2": up2, "up3": up3, "down2": down2, "down3": down3,
    "tile2x2": tile2x2, "tile3x3": tile3x3,
    "mirror_h": mirror_h, "mirror_v": mirror_v, "mirror_4": mirror_4,
    "left_half": left_half, "right_half": right_half, "top_half": top_half, "bottom_half": bottom_half,
    "gravity_down": gravity_down, "gravity_up": gravity_up,
    "gravity_left": gravity_left, "gravity_right": gravity_right2,
    "fill_holes": fill_holes,
    "symmetrize_lr": symmetrize_lr, "symmetrize_ud": symmetrize_ud, "symmetrize_4": symmetrize_4,
    "outline": outline_objs,
    "extend_h": extend_lines_h, "extend_v": extend_lines_v,
    "dedupe": dedupe_rows_cols,
    "common_color": most_common_color_1x1, "rare_color": least_common_color_1x1,
    "overlay_h": split_overlay_h, "overlay_v": split_overlay_v,
}


def apply(name, g):
    try:
        out = PRIMITIVES[name](g)
    except Exception:
        return None
    return out if _ok(out) else None


# Optional extended DSLs. ARC_DSL selects the primitive set:
#   small = original 44, big = 135 (adds dsl_ext), full = 205 (adds dsl_cond too, default).
import os as _os
_MODE = _os.environ.get("ARC_DSL", "full")
if _MODE in ("big", "full"):
    import dsl_ext as _ext
    PRIMITIVES.update(_ext.EXTRA)
if _MODE == "full":
    import dsl_cond as _cond
    PRIMITIVES.update(_cond.EXTRA)
