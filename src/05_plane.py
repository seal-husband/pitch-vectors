"""
The arsenal lies on a great circle. Whose, and how tightly?

Seen on the map, every type's mean direction sits near one closed curve; turned the right
way in 3D, they line up. So the arsenal lives on a plane THROUGH ORIGIN E -- a great
circle on the sphere. The expected explanation: the plane is set by the arm slot.

"Expected" is not "shown". This file asks three things in the order that can settle them:

  1. Through the origin, or just flat?  Fit the plane through E and the free (affine)
     plane; if the through-E fit is nearly as tight, the great circle is real.
  2. One plane for the league, or one per pitcher?  Compare each pitcher's out-of-plane
     angle under his own plane, under the league plane, and under a plane fitted to a
     FAKE arsenal assembled from other pitchers' type means (same types, same hand frame).
     If his own plane is not tighter than the fake one, planarity is a property of the
     pitch types, not of him.
  3. What orients it?  The plane's trace on the movement chart (its line at zero speed
     difference) is compared with the release point, which the ball's flight cannot know.

The arm-angle proxy is atan2(release height − 5.0 ft, |release side|). Statcast's own
arm angle needs the shoulder position, which is not in this snapshot; the proxy is
monotone in the same thing and is used only through rank correlations.

Usage:  python src/05_plane.py [season]
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

MIN_TYPE_SHARE = .03
MIN_TYPE_N = 30
SHOULDER_Z = 5.0
SEED = 5


def load(year):
    d = pd.read_parquet(sph.SNAP[year], columns=[
        "game_type", "pitcher", "player_name", "pitch_type", "p_throws",
        "pfx_x", "pfx_z", "release_speed", "release_pos_x", "release_pos_z"])
    d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)
          & d.pfx_x.notna() & d.pfx_z.notna() & d.release_speed.notna()].copy()
    n = d.groupby("pitcher").size()
    d = d[d.pitcher.isin(n[n >= sph.MIN_PITCHES].index)]
    for c in ["pfx_x", "pfx_z", "release_speed", "release_pos_x", "release_pos_z"]:
        d[c] = d[c].astype(float)
    L = d.p_throws.eq("L")
    d["hx"] = np.where(L, -d.pfx_x, d.pfx_x) * 12
    d["vz"] = d.pfx_z * 12
    d["rx"] = np.where(L, -d.release_pos_x, d.release_pos_x)
    return d


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def normal_through_origin(U, w=None):
    """Plane through the origin: least eigenvector of the uncentred second moment."""
    w = np.ones(len(U)) if w is None else w
    M = (U * w[:, None]).T @ U
    val, vec = np.linalg.eigh(M)
    return vec[:, 0]


def normal_affine(P, w=None):
    w = np.ones(len(P)) if w is None else w
    c = (P * w[:, None]).sum(0) / w.sum()
    X = (P - c) * np.sqrt(w)[:, None]
    return np.linalg.svd(X, full_matrices=False)[2][-1], c


def off_deg(U, n):
    return np.degrees(np.arcsin(np.clip(np.abs(U @ n), 0, 1)))


def main():
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(SEED)
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    d = load(year)
    U, L, v0 = sph.vectors_E(d)
    V = U * L[:, None]
    d["i"] = np.arange(len(d))
    print("%d年 %d投手 / %d球" % (year, d.pitcher.nunique(), len(d)))

    # ---- per-pitcher arsenals: type mean vectors with enough use
    ars = {}
    for pid, g in d.groupby("pitcher"):
        vc = g.pitch_type.value_counts()
        ok = vc[(vc / len(g) >= MIN_TYPE_SHARE) & (vc >= MIN_TYPE_N)].index
        if len(ok) < 3:
            continue
        rows = []
        for t in ok:
            ii = g.i.values[g.pitch_type.values == t]
            rows.append((t, V[ii].mean(0), len(ii)))
        ars[pid] = rows
    print("球種3つ以上の投手: %d" % len(ars))

    # ---- 1. league plane from type centroids
    print("\n■ 1. リーグ全体：球種の平均方向は一つの大円に乗るか")
    types = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS"]
    C = np.array([unit(V[d.pitch_type.values == t].mean(0)) for t in types])
    nL = normal_through_origin(C)
    if nL[1] < 0:
        nL = -nL
    for t, c in zip(types, C):
        print("   %-3s 大円からのずれ %5.1f°" % (t, off_deg(c[None], nL)[0]))
    print("   法線 (横 %+.3f, 縦 %+.3f, 速度 %+.3f)" % tuple(nL))
    offL_pitch = off_deg(U, nL)
    print("   一球ずつ：大円からのずれ 中央 %.1f°  90%%点 %.1f°"
          % (np.median(offL_pitch), np.quantile(offL_pitch, .9)))

    # ---- 2. own plane vs league plane vs fake arsenal
    print("\n■ 2. 平面は投手ごとか（球種平均を、各投手の平面・リーグ平面・偽の持ち球で比較）")
    pool = {}
    for pid, rows in ars.items():
        for t, v, n in rows:
            pool.setdefault(t, []).append(v)
    rec = []
    for pid, rows in ars.items():
        Vp = np.array([v for _, v, _ in rows])
        Up = unit(Vp)
        w = np.ones(len(rows))                      # each type counts once
        n_own = normal_through_origin(Up, w)
        n_aff, _ = normal_affine(Vp, w)
        # affine residual in the same angular currency: distance / |v|
        res_aff = np.degrees(np.arcsin(np.clip(
            np.abs((Vp - Vp.mean(0)) @ n_aff) / np.linalg.norm(Vp, axis=1), 0, 1)))
        fake = []
        for _ in range(20):
            Vf = np.array([pool[t][rng.integers(len(pool[t]))] for t, _, _ in rows])
            Uf = unit(Vf)
            fake.append(np.sqrt(np.mean(off_deg(Uf, normal_through_origin(Uf)) ** 2)))
        g = d[d.pitcher == pid]
        rec.append({"pitcher": pid, "name": g.player_name.iloc[0], "k": len(rows),
                    "own": np.sqrt(np.mean(off_deg(Up, n_own) ** 2)),
                    "aff": np.sqrt(np.mean(res_aff ** 2)),
                    "league": np.sqrt(np.mean(off_deg(Up, nL) ** 2)),
                    "fake": np.median(fake),
                    "n": n_own if n_own[1] >= 0 else -n_own,
                    "rx": g.rx.median(), "rz": g.release_pos_z.median(),
                    "types": " ".join(t for t, _, _ in rows)})
    R = pd.DataFrame(rec)
    for k in [3, 4, 5]:
        s = R[R.k == k] if k < 5 else R[R.k >= 5]
        print("   球種%s%d (n=%3d)  自分の平面 %4.1f°  リーグ平面 %4.1f°  偽の持ち球 %4.1f°"
              "   自分<偽 %.0f%%"
              % ("" if k < 5 else "≥", k, len(s), s.own.median(), s.league.median(),
                 s.fake.median(), 100 * (s.own < s.fake).mean()))
    print("   （RMS 角度、投手ごとの中央値。原点Eを通る平面 vs 自由な平面: %.1f° vs %.1f°、球種4以上）"
          % (R[R.k >= 4].own.median(), R[R.k >= 4].aff.median()))

    # ---- 3. what orients the plane
    print("\n■ 3. 平面の向きを決めるもの")
    N = np.vstack(R.n.values)
    # trace on the movement chart: the in-plane line at zero speed difference
    tr = np.cross(N, np.array([0., 0., 1.]))
    tr = tr * np.sign(tr[:, 1:2] + 1e-12)            # point it upward
    R["trace"] = np.degrees(np.arctan2(tr[:, 1], -tr[:, 0]))   # 0 = pure arm side, 90 = pure rise
    R["lean"] = np.degrees(np.arcsin(np.abs(N[:, 2])))         # 0 = plane contains the speed axis
    R["arm"] = np.degrees(np.arctan2(R.rz - SHOULDER_Z, R.rx.abs()))
    print("   平面の軌跡（変化量図上の傾き） 中央 %.0f°  四分位 %.0f 〜 %.0f°"
          % (R.trace.median(), R.trace.quantile(.25), R.trace.quantile(.75)))
    print("   速度軸からの傾き（0=速度軸を含む） 中央 %.0f°  四分位 %.0f 〜 %.0f°"
          % (R.lean.median(), R.lean.quantile(.25), R.lean.quantile(.75)))
    for nm in ["trace", "lean"]:
        r, p = stats.spearmanr(R[nm], R.arm)
        print("   %-6s × 腕の角度（リリース点）  Spearman %+.2f  (p=%.1g)" % (nm, r, p))
    q = pd.qcut(R.arm, 4, labels=["低い腕", "", " ", "高い腕"])
    print("   腕の角度の四分位ごとの平面の軌跡:")
    for lab, g in R.groupby(q, observed=True):
        print("      腕 %5.0f°〜%5.0f°  軌跡 中央 %4.0f°  n=%d"
              % (g.arm.min(), g.arm.max(), g.trace.median(), len(g)))

    # ---- 4. what is left: pitches that leave their own pitcher's plane
    print("\n■ 4. 自分の平面から外れる球種（投手の平面を、その球種を除いて当て直す）")
    out = []
    for pid, rows in ars.items():
        if len(rows) < 4:
            continue
        Up = unit(np.array([v for _, v, _ in rows]))
        for j, (t, _, _) in enumerate(rows):
            m = np.arange(len(rows)) != j
            n = normal_through_origin(Up[m])
            s = np.sign(Up[j] @ n) * np.sign(n[1] if abs(n[1]) > 1e-9 else 1)
            out.append((t, off_deg(Up[j][None], n)[0], pid))
    O = pd.DataFrame(out, columns=["t", "off", "pitcher"])
    print(O.groupby("t").off.describe(percentiles=[.5, .9])[["count", "50%", "90%"]]
          .round(1).loc[[t for t in types if t in set(O.t)]].to_string())

    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    R.drop(columns=["n"]).assign(nx=N[:, 0], ny=N[:, 1], nz=N[:, 2]).to_csv(
        os.path.join(ROOT, "output", "planes_%d.csv" % year), index=False, encoding="utf-8")
    print("\nwrote output/planes_%d.csv" % year)


if __name__ == "__main__":
    main()
