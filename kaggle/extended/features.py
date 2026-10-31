"""Fixed-length task features for the hypothesis proposer.

A task is summarised by averaging per-pair features over its training pairs.
Per pair we describe how the output relates to the input (shape change, colour
change, symmetry, containment) and, for every DSL primitive, whether applying
it gives the right output shape and how many cells already agree.
"""
import numpy as np
from scipy import ndimage
import dsl

NAMES = list(dsl.PRIMITIVES.keys())
NPRIM = len(NAMES)


def _is_subgrid(small, big):
    h, w = small.shape
    H, W = big.shape
    if h > H or w > W:
        return False
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if np.array_equal(big[r:r + h, c:c + w], small):
                return True
    return False


def _nobj(g):
    bg = dsl.bg_color(g)
    _, n = ndimage.label(g != bg, structure=np.ones((3, 3)))
    return n


def pair_features(i, o):
    ih, iw = i.shape
    oh, ow = o.shape
    f = []
    # shape relations
    f += [oh / ih, ow / iw, float(i.shape == o.shape), float(oh * ow < ih * iw), float(oh * ow > ih * iw)]
    f += [float(oh % ih == 0 and ow % iw == 0), float(ih % oh == 0 and iw % ow == 0),
          float(oh == 1 and ow == 1), float(oh == ow), float(ih == iw)]
    # colours
    ci, co = set(np.unique(i).tolist()), set(np.unique(o).tolist())
    f += [len(ci) / 10, len(co) / 10, len(co - ci) / 10, len(ci - co) / 10]
    f += [float(dsl.bg_color(i) != dsl.bg_color(o)), float((i != dsl.bg_color(i)).mean()),
          float((o != dsl.bg_color(o)).mean())]
    # same-shape change stats
    if i.shape == o.shape:
        ch = i != o
        f += [float(ch.mean()), float((ch & (o != dsl.bg_color(i))).mean()), float((ch & (i != dsl.bg_color(i))).mean())]
    else:
        f += [-1.0, -1.0, -1.0]
    # symmetry of input and output
    f += [float(np.array_equal(i, i[:, ::-1])), float(np.array_equal(i, i[::-1])),
          float(np.array_equal(o, o[:, ::-1])), float(np.array_equal(o, o[::-1]))]
    # containment
    f += [float(_is_subgrid(o, i)), float(oh <= ih and ow <= iw and _is_subgrid(o, i))]
    # objects
    ni, no = _nobj(i), _nobj(o)
    f += [min(ni, 20) / 20, min(no, 20) / 20, float(ni == no)]
    # per-primitive: shape match and cell agreement
    shape_ok, agree = [], []
    for name in NAMES:
        r = dsl.apply(name, i)
        if r is None or r.shape != o.shape:
            shape_ok.append(0.0)
            agree.append(0.0)
        else:
            shape_ok.append(1.0)
            agree.append(float((r == o).mean()))
    return np.array(f + shape_ok + agree, dtype=np.float32)


def task_features(pairs):
    """pairs: list of (input ndarray, output ndarray). Returns mean feature vector."""
    return np.mean([pair_features(i, o) for i, o in pairs], axis=0)


N_FEATURES = len(pair_features(np.zeros((3, 3), dtype=int), np.zeros((3, 3), dtype=int)))
