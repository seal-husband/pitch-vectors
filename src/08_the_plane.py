"""
Pin down the plane: its normal, how thick the slab is, and how sure we are.

Turned edge-on, the 3D cloud is a slab. 05-07 fitted planes pitcher by pitcher; this file
fixes ONE orientation for the league and asks whether it deserves to be one.

Coordinates are those of the map: (横 in, 縦 in, (球速 − リーグ速球) × s in), s = 0.63.
A normal written in these coordinates depends on s, so every plane is also reported as

        球速[mph] = a × 縦[in] + b × 横[in] + 定数

which does not. Three ways to fit it, because they answer different questions:

  pitch     every pitch once, pitchers pooled. Dominated by fastballs and by where
            each pitcher sits -- a pitcher who throws 98 lifts his whole arsenal.
  type      each pitcher's type means, each type once, pitchers pooled.
  within    each pitcher's type means, minus that pitcher's own centre. Only the SHAPE
            of each arsenal counts; a hard thrower and a soft one can share a plane
            orientation at different heights. This is the definition used from here on.

Then: bootstrap over pitchers, per-season stability, righties vs lefties, how each
pitcher's height along the normal relates to how hard he throws, and the in-plane axes.

Usage:  python src/08_the_plane.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import importlib.util

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "src", "03_sphere.py"))
sph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sph)

S = sph.S_IN_PER_MPH
MIN_SHARE, MIN_N = .03, 30
N_BOOT = 300
SEED = 8
AXN = ["横", "縦", "速度"]


def load(year):
    d = pd.read_parquet(sph.SNAP[year], columns=[
        "game_type", "pitcher", "pitch_type", "p_throws", "pfx_x", "pfx_z", "release_speed"])
    d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)].dropna().copy()
    n = d.groupby("pitcher").size()
    d = d[d.pitcher.isin(n[n >= sph.MIN_PITCHES].index)]
    for c in ["pfx_x", "pfx_z", "release_speed"]:
        d[c] = d[c].astype(float)
    v0 = d[d.pitch_type.isin(sph.FAST)].release_speed.mean()
    d["x"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12
    d["y"] = d.pfx_z * 12
    d["z"] = (d.release_speed - v0) * S
    return d, v0


def type_means(d):
    """Per pitcher: type means for types with enough use."""
    g = d.groupby(["pitcher", "pitch_type"])
    m = g[["x", "y", "z"]].mean()
    m["n"] = g.size()
    m["share"] = m.n / m.groupby(level=0).n.transform("sum")
    m = m[(m.share >= MIN_SHARE) & (m.n >= MIN_N)].reset_index()
    m["p_throws"] = m.pitcher.map(d.groupby("pitcher").p_throws.first())
    k = m.groupby("pitcher").pitch_type.transform("size")
    return m[k >= 3].copy()


def pca(X):
    """Principal axes of X (already centred). Columns of vec: largest -> smallest."""
    val, vec = np.linalg.eigh(X.T @ X / len(X))
    return val[::-1], vec[:, ::-1]


def orient(n):
    return n if n[2] >= 0 else -n            # normal points toward "faster"


def within_normal(m):
    X = m[["x", "y", "z"]].values - m.groupby("pitcher")[["x", "y", "z"]].transform("mean").values
    val, vec = pca(X)
    return orient(vec[:, 2]), val, vec


def natural(n):
    """Plane n·(x, y, z) = k rewritten as mph = a·縦 + b·横 + const."""
    return -n[1] / (n[2] * S), -n[0] / (n[2] * S)


def ang(a, b):
    return np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1)))


def main():
    rng = np.random.default_rng(SEED)
    d, v0 = load(2026)
    m = type_means(d)
    P = m.pitcher.unique()
    print("2026年  %d投手（3球種以上）  リーグ速球 %.1f mph  s = %.2f in/mph" % (len(P), v0, S))

    # ---------------------------------------------------------------- 1. three fits
    print("\n■ 1. 三つの当てはめ方で法線は一致するか")
    X = d[["x", "y", "z"]].values
    _, vp = pca(X - X.mean(0))
    Xt = m[["x", "y", "z"]].values
    _, vt = pca(Xt - Xt.mean(0))
    nW, valW, vecW = within_normal(m)
    fits = {"pitch（一球ずつ）": orient(vp[:, 2]), "type（球種平均）": orient(vt[:, 2]),
            "within（投手内）": nW}
    for nm, n in fits.items():
        a, b = natural(n)
        print("   %-18s 法線 (横 %+.3f, 縦 %+.3f, 速度 %+.3f)   within との差 %4.1f°"
              "   球速 = %+.3f×縦 %+.3f×横"
              % (nm, *n, ang(n, nW), a, b))

    # ---------------------------------------------------------------- 2. thickness
    print("\n■ 2. 板の厚み（within、投手内の球種平均）")
    sd = np.sqrt(valW)
    print("   主軸の標準偏差  第1 %.2f in   第2 %.2f in   法線方向 %.2f in"
          "  （= 球速にして %.2f mph）" % (sd[0], sd[1], sd[2], sd[2] / (nW[2] * S)))
    print("   分散の割合  %.1f%% / %.1f%% / %.2f%%" % tuple(100 * valW / valW.sum()))
    for k, nm in ((0, "第1軸"), (1, "第2軸")):
        e = vecW[:, k] * np.sign(vecW[1, k] if abs(vecW[1, k]) > .1 else vecW[0, k])
        print("   %s (横 %+.3f, 縦 %+.3f, 速度 %+.3f)" % (nm, *e))

    # ---------------------------------------------------------------- 3. bootstrap
    print("\n■ 3. 投手を単位としたブートストラップ（%d回）" % N_BOOT)
    groups = {p: g for p, g in m.groupby("pitcher")}
    B, NA = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(P, len(P), replace=True)
        mb = pd.concat([groups[p].assign(pitcher=i) for i, p in enumerate(pick)])
        nb, _, _ = within_normal(mb)
        B.append(nb); NA.append(natural(nb))
    B, NA = np.array(B), np.array(NA)
    dev = np.array([ang(b, nW) for b in B])
    print("   法線の揺れ  中央 %.2f°   95%%点 %.2f°" % (np.median(dev), np.quantile(dev, .95)))
    for j, nm in enumerate(AXN):
        print("   法線 %s  %+.3f  [%+.3f, %+.3f]" % (nm, nW[j], *np.quantile(B[:, j], [.025, .975])))
    print("   球速 = a×縦 + b×横   a %+.3f [%+.3f, %+.3f]   b %+.3f [%+.3f, %+.3f] mph/in"
          % (natural(nW)[0], *np.quantile(NA[:, 0], [.025, .975]),
             natural(nW)[1], *np.quantile(NA[:, 1], [.025, .975])))

    # ---------------------------------------------------------------- 4. seasons, hands
    print("\n■ 4. 年と利き腕で変わるか（within）")
    for yr in [2024, 2025, 2026]:
        dy, _ = load(yr)
        my = type_means(dy)
        ny, _, _ = within_normal(my)
        print("   %d  法線 (%+.3f, %+.3f, %+.3f)  2026との差 %4.1f°   球速 = %+.3f×縦 %+.3f×横   投手 %d"
              % (yr, *ny, ang(ny, nW), *natural(ny), my.pitcher.nunique()))
    for h in ["R", "L"]:
        nh, _, _ = within_normal(m[m.p_throws == h])
        print("   %s投手  法線 (%+.3f, %+.3f, %+.3f)  全体との差 %4.1f°   球速 = %+.3f×縦 %+.3f×横   投手 %d"
              % (h, *nh, ang(nh, nW), *natural(nh), m[m.p_throws == h].pitcher.nunique()))

    # ---------------------------------------------------------------- 5. per pitcher
    print("\n■ 5. 投手ごとの板：向きのずれと、法線方向の高さ")
    rows = []
    for p, g in groups.items():
        if len(g) < 4:
            continue
        Xg = g[["x", "y", "z"]].values
        _, vg = pca(Xg - Xg.mean(0))
        ff = d[(d.pitcher == p) & d.pitch_type.isin(["FF", "SI"])].release_speed.mean()
        rows.append((ang(orient(vg[:, 2]), nW), Xg.mean(0) @ nW, ff))
    R = np.array(rows)
    print("   投手の平面と共通法線の角度  中央 %.1f°  四分位 %.1f 〜 %.1f°  (4球種以上 n=%d)"
          % (np.median(R[:, 0]), *np.quantile(R[:, 0], [.25, .75]), len(R)))
    ok = ~np.isnan(R[:, 2])
    r, pv = stats.spearmanr(R[ok, 1], R[ok, 2])
    print("   法線方向の高さ  中央 %+.1f in  四分位 %+.1f 〜 %+.1f in" %
          (np.median(R[:, 1]), *np.quantile(R[:, 1], [.25, .75])))
    print("   高さ × 速球の球速  Spearman %+.2f (p=%.1g)" % (r, pv))

    out = os.path.join(ROOT, "output")
    np.savez(os.path.join(out, "plane_2026.npz"), normal=nW, axes=vecW, var=valW,
             boot=B, s=S, v0=v0, height=np.median(R[:, 1]))
    print("\nwrote output/plane_2026.npz")


if __name__ == "__main__":
    main()
