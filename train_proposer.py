"""Train the neural hypothesis proposer (multi-label MLP) on synthetic tasks.

Usage: python3 -I train_proposer.py <synth.npz> <out_model.pkl> [mlp_seed]
"""
import pickle
import sys

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def recall_at_k(P, Y, k):
    """Fraction of tasks where every true primitive is inside the top-k predictions."""
    top = np.argsort(-P, axis=1)[:, :k]
    hits = []
    for t, y in zip(top, Y):
        need = set(np.where(y > 0)[0].tolist())
        hits.append(need <= set(t.tolist()))
    return float(np.mean(hits))


def main():
    d = np.load(sys.argv[1])
    X, Y = d["X"], d["Y"]
    n = len(X)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    cut = int(n * 0.9)
    tr, va = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    model = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, max_iter=60,
                          early_stopping=True, n_iter_no_change=6, random_state=int(sys.argv[3]) if len(sys.argv) > 3 else 0, verbose=False)
    model.fit(sc.transform(X[tr]), Y[tr])
    P = model.predict_proba(sc.transform(X[va]))
    Yv = Y[va]
    # random baseline: expected recall@k for a uniformly random subset is estimated by simulation
    rand = rng.random(P.shape)
    npr = Y.shape[1]
    print("val tasks", len(va), "primitives+recolor slot", npr)
    for k in (6, 10, 16, 24):
        print(f"all-true-primitives-in-top-{k}: model={recall_at_k(P[:, :-1], Yv[:, :-1], k):.3f} "
              f"random={recall_at_k(rand[:, :-1], Yv[:, :-1], k):.3f}")
    with open(sys.argv[2], "wb") as f:
        pickle.dump({"scaler": sc, "model": model}, f)
    print("saved", sys.argv[2])


if __name__ == "__main__":
    main()
