"""
Multivariate curve resolution of pitch arsenals.

Chemometrics treats a set of mixture spectra as

        X (pitchers × channels)  ≈  W (pitchers × components) · H (components × channels)

with W, H ≥ 0: H holds the pure-component spectra, W the concentrations. Here

  spectrum     a pitcher's pitches as a 2-D density on the Pitch Plane (the normal carries
               1% of the variance and almost no peak separation, 14_resolution), binned
               at 1 inch and lightly smoothed, normalised to sum 1
  component    a region of the plane that recurs across pitchers -- a data-made pitch type,
               allowed to sit between the named ones
  W row        how much of each component a pitcher throws; rows sum to ~1 = usage

Two referencing schemes, as a spectroscopist would test an internal standard:

  abs   in-plane position as is (lefties mirrored)
  ref   minus the pitcher's primary fastball, so every arsenal is read from its own
        fastball -- removes arm slot and arm strength shifts, loses FF-vs-SI of the primary

Number of components K is chosen by (a) reconstruction and (b) split-half reproducibility:
each pitcher's pitches are split by alternate pitch, the two halves are resolved
separately, and components are matched by cosine similarity.

Usage:  python src/16_mcr.py
"""
from __future__ import annotations
import sys, os, importlib.util
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "src", "03_sphere.py"))
sph = importlib.util.module_from_spec(spec); spec.loader.exec_module(sph)

YEAR = 2026
KS = list(range(2, 17))
SMOOTH = 1.0                      # bins (= inches)
z = np.load(os.path.join(ROOT, "output", "plane_%d.npz" % YEAR))
n, S, V0 = z["normal"], float(z["s"]), float(z["v0"])
F1 = np.array([1., 0., 0.]) - n[0] * n; F1 /= np.linalg.norm(F1)       # 横, laid in the plane
F2 = np.cross(n, F1); F2 *= np.sign(F2[1])                             # up, in the plane


def load():
    d = pd.read_parquet(sph.SNAP[YEAR], columns=["game_type", "pitcher", "player_name",
                        "pitch_type", "p_throws", "pfx_x", "pfx_z", "release_speed"])
    d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)].dropna().copy()
    cnt = d.groupby("pitcher").size(); d = d[d.pitcher.isin(cnt[cnt >= sph.MIN_PITCHES].index)]
    for c in ["pfx_x", "pfx_z", "release_speed"]:
        d[c] = d[c].astype(float)
    X = np.c_[np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12, d.pfx_z * 12,
              (d.release_speed - V0) * S]
    d["u"], d["w"] = X @ F1, X @ F2
    # primary fastball = the most-thrown of FF/SI/FC
    f = d[d.pitch_type.isin(["FF", "SI", "FC"])]
    top = f.groupby("pitcher").pitch_type.agg(lambda s: s.value_counts().index[0])
    ref = f[f.pitch_type.values == f.pitcher.map(top).values].groupby("pitcher")[["u", "w"]].mean()
    d = d[d.pitcher.isin(ref.index)]
    d["ur"] = d.u - d.pitcher.map(ref.u)
    d["wr"] = d.w - d.pitcher.map(ref.w)
    d["half"] = d.groupby("pitcher").cumcount() % 2
    return d


def grid_for(d, a, b):
    lo = np.floor(np.quantile(d[[a, b]].values, .001, axis=0)) - 2
    hi = np.ceil(np.quantile(d[[a, b]].values, .999, axis=0)) + 2
    return [np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1)]


def spectra(d, a, b, edges, mask=None):
    P, rows = [], []
    for pid, g in d.groupby("pitcher"):
        if mask is not None:
            g = g[mask.loc[g.index]]
        h, _, _ = np.histogram2d(g[a], g[b], bins=edges)
        h = gaussian_filter(h, SMOOTH)
        P.append((h / h.sum()).ravel()); rows.append(pid)
    return np.array(P), np.array(rows)


def resolve(X, k):
    m = NMF(k, init="nndsvda", solver="mu", beta_loss="frobenius", max_iter=800, tol=1e-5, random_state=0)
    W = m.fit_transform(X); H = m.components_
    s = H.sum(1, keepdims=True); H = H / s; W = W * s.T          # H rows sum to 1; W = usage
    return W, H


def match(Ha, Hb):
    Na = Ha / np.linalg.norm(Ha, axis=1, keepdims=True)
    Nb = Hb / np.linalg.norm(Hb, axis=1, keepdims=True)
    C = Na @ Nb.T
    i, j = linear_sum_assignment(-C)
    return C[i, j]


def main():
    d = load()
    print("%d年 %d投手 / %d球" % (YEAR, d.pitcher.nunique(), len(d)))
    out = {}
    for tag, a, b in (("abs", "u", "w"), ("ref", "ur", "wr")):
        edges = grid_for(d, a, b)
        X, pids = spectra(d, a, b, edges)
        XA, _ = spectra(d, a, b, edges, d.half.eq(0))
        XB, _ = spectra(d, a, b, edges, d.half.eq(1))
        tot = ((X - X.mean(0)) ** 2).sum()
        print("\n■ %s  格子 %d×%d" % ("PP上の絶対位置" if tag == "abs" else "自分の速球を原点", len(edges[0]) - 1, len(edges[1]) - 1))
        print("   %3s %9s %11s %11s %11s" % ("K", "説明率", "再現 平均", "再現 最小", "使用率合計"))
        res = []
        for k in KS:
            W, H = resolve(X, k)
            ev = 1 - ((X - W @ H) ** 2).sum() / tot
            _, Ha = resolve(XA, k); _, Hb = resolve(XB, k)
            sim = match(Ha, Hb)
            res.append((k, ev, sim.mean(), sim.min()))
            print("   %3d %8.1f%% %11.3f %11.3f %11.2f" % (k, 100 * ev, sim.mean(), sim.min(), np.median(W.sum(1))))
        out[tag] = dict(edges=edges, X=X, pids=pids, res=np.array(res))
    np.savez(os.path.join(ROOT, "output", "mcr_scan_%d.npz" % YEAR),
             **{"%s_%s" % (t, k): v for t, o in out.items() for k, v in
                (("res", o["res"]), ("pids", o["pids"]), ("e0", o["edges"][0]), ("e1", o["edges"][1]))})
    d[["pitcher", "player_name", "pitch_type", "u", "w", "ur", "wr", "half"]].to_parquet(
        os.path.join(ROOT, "output", "mcr_pitches_%d.parquet" % YEAR), index=False)


if __name__ == "__main__":
    main()
