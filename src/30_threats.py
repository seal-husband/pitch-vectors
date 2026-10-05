"""
Threats to the map, checked where the data allow.

  1. Is it just arm angle (or the four-seam's own shape)? Arm slot rotates every pitch's
     movement direction, and the four-seam is the origin, so both could drive the axes
     without any choice by the pitcher. Arm angle proxy: atan2(release height − 5 ft,
     |release side|) -- crude, but monotone in the real thing.
  2. Platoon. Pitchers throw different mixes to same-side and opposite-side batters; a
     pitcher who faces more lefties would move on the map for reasons that are not style.
     The map is recomputed on pitches to RHB only and to LHB only.
  3. Year to year. Split-half reliability (0.99+) is within one season. The map is
     recomputed for 2025 and compared with 2026 for pitchers in both.

Usage:  python src/30_threats.py
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import SNAP
DELTA, EPS = .25, .02


def load(year):
    d = pd.read_parquet(SNAP[year], columns=["game_type", "pitcher", "player_name", "pitch_type", "p_throws", "stand",
                                             "pfx_x", "pfx_z", "release_speed", "release_pos_x", "release_pos_z"])
    d = d[d.game_type == "R"].dropna(subset=["pitch_type", "pfx_x", "pfx_z"]).copy()
    n = d.groupby("pitcher").size(); d = d[d.pitcher.isin(n[n >= 400].index)]
    for c in ["pfx_x", "pfx_z", "release_speed", "release_pos_x", "release_pos_z"]:
        d[c] = d[c].astype(float)
    L = d.p_throws.eq("L")
    d["hx"] = np.where(L, -d.pfx_x, d.pfx_x) * 12
    d["vz"] = d.pfx_z * 12
    d["rx"] = np.where(L, -d.release_pos_x, d.release_pos_x)
    return d


def reference(d):
    tot = d.groupby("pitcher").size()
    fb = d[d.pitch_type.isin(["FF", "SI", "FC"])].groupby(["pitcher", "pitch_type"])[["hx", "vz"]].agg(["mean", "size"])
    fb.columns = ["hx", "n1", "vz", "n"]; fb = fb.drop(columns="n1").reset_index()
    fb["share"] = fb.n / tot.reindex(fb.pitcher).values
    has = fb[(fb.n >= 30) & (fb.share >= .05)]
    W = has.pivot(index="pitcher", columns="pitch_type", values=["hx", "vz"])
    off = {}
    for t in ["SI", "FC"]:
        b = W.dropna(subset=[("hx", "FF"), ("hx", t)])
        off[t] = np.array([(b[(k, "FF")] - b[(k, t)]).median() for k in ["hx", "vz"]])
    ref = {}
    for p, g in has.groupby("pitcher"):
        ff = g[g.pitch_type == "FF"]
        ref[p] = ff[["hx", "vz"]].values[0] if len(ff) else g.sort_values("n").iloc[-1][["hx", "vz"]].values.astype(float) + off[g.sort_values("n").iloc[-1].pitch_type]
    return ref


def mapxyz(d, ref, subset=None):
    d = d[d.pitcher.isin(ref.keys())].copy()
    R = np.array([ref[p] for p in d.pitcher])
    d["fx"], d["fz"] = d.hx - R[:, 0], d.vz - R[:, 1]
    if subset is not None:
        d = d[subset.loc[d.index]]
    tot = d.groupby("pitcher").size()
    g = d[d.pitch_type != "FF"]
    g = g.assign(side=np.where(g.fz <= 2, np.where(g.fx >= 2, "glove", "arm"), "other"))
    out = {}
    for p, h in g.groupby("pitcher"):
        n = tot[p]; a, gl = h[h.side == "arm"], h[h.side == "glove"]
        ah, av, gh, gv = (-a.fx).sum() / n, (-a.fz).sum() / n, gl.fx.sum() / n, (-gl.fz).sum() / n
        ah, av, gh, gv = (max(v, 0) + DELTA for v in (ah, av, gh, gv))
        au, gu = len(a) / n + EPS, len(gl) / n + EPS
        out[p] = (np.log(av / ah) / np.sqrt(2), np.log(gv / gh) / np.sqrt(2), np.log(au / gu) / np.sqrt(2))
    return pd.DataFrame(out, index=["x", "y", "z"]).T


d26 = load(2026); ref26 = reference(d26)
M = mapxyz(d26, ref26)
print("2026年 %d投手" % len(M))

# ---------------------------------------------------------------- 1
print("\n■ 1. 地図の軸は、アームアングルや速球の形をどれだけ映しているか（Spearman、説明率は線形R²）")
pp = d26.groupby("pitcher").agg(rz=("release_pos_z", "median"), rx=("rx", lambda s: s.abs().median()))
arm = np.degrees(np.arctan2(pp.rz - 5.0, pp.rx)).rename("arm")
ffs = d26[d26.pitch_type == "FF"].groupby("pitcher")[["hx", "vz"]].mean().rename(columns={"hx": "ff_hx", "vz": "ff_vz"})
J = M.join(arm).join(ffs)
for c, lab in (("x", "x 腕側 落とす÷流す"), ("y", "y グラブ側 落とす÷曲げる"), ("z", "z 腕側÷グラブ側")):
    ra = stats.spearmanr(J[c], J.arm, nan_policy="omit")[0]
    rv = stats.spearmanr(J[c], J.ff_vz, nan_policy="omit")[0]
    rh = stats.spearmanr(J[c], J.ff_hx, nan_policy="omit")[0]
    ok = J[[c, "arm", "ff_vz", "ff_hx"]].dropna()
    X = np.c_[ok.arm, ok.ff_vz, ok.ff_hx, np.ones(len(ok))]
    b, *_ = np.linalg.lstsq(X, ok[c].values, rcond=None)
    r2 = 1 - ((ok[c] - X @ b) ** 2).sum() / ((ok[c] - ok[c].mean()) ** 2).sum()
    print("   %-20s × 腕の角度 %+.2f   × FFの縦 %+.2f   × FFの横 %+.2f   → 3つで R² %.2f" % (lab, ra, rv, rh, r2))

# ---------------------------------------------------------------- 2
print("\n■ 2. 打者の左右で地図の位置は変わるか")
same = pd.Series(d26.stand.values == d26.p_throws.values, index=d26.index)
Ms, Mo = mapxyz(d26, ref26, same), mapxyz(d26, ref26, ~same)
C = Ms.join(Mo, lsuffix="_同側", rsuffix="_逆側").dropna()
for c in "xyz":
    r = np.corrcoef(C[c + "_同側"], C[c + "_逆側"])[0, 1]
    shift = (C[c + "_逆側"] - C[c + "_同側"])
    print("   %s  同側打者と逆側打者の相関 %.2f   逆側−同側 平均 %+.2f（全体のSD %.2f）"
          % (c, r, shift.mean(), M[c].std()))
dist = np.sqrt(((Ms.values - M.loc[Ms.index].values) ** 2).sum(1))
allp = np.linalg.norm(M.values[:, None] - M.values[None], axis=2)[np.triu_indices(len(M), 1)]
print("   同側打者だけで作った位置と全体の位置のずれ：中央 %.2f（全投手の組の距離の中央 %.2f）" % (np.median(dist), np.median(allp)))

# ---------------------------------------------------------------- 3
print("\n■ 3. 年をまたいだ安定性（2025 → 2026、両年とも400球以上）")
d25 = load(2025); M25 = mapxyz(d25, reference(d25))
Y = M25.join(M, lsuffix="_25", rsuffix="_26").dropna()
for c in "xyz":
    print("   %s  2025と2026の相関 %.2f" % (c, np.corrcoef(Y[c + "_25"], Y[c + "_26"])[0, 1]))
A, B = Y[["x_25", "y_25", "z_25"]].values, Y[["x_26", "y_26", "z_26"]].values
D = np.linalg.norm(A[:, None] - B[None], axis=2)
rank = np.array([(D[i] < D[i, i]).sum() + 1 for i in range(len(Y))])
print("   2025年の位置から見て、2026年の自分が何番目に近いか：1位 %.0f%%、5位以内 %.0f%%（%d投手中）"
      % (100 * (rank == 1).mean(), 100 * (rank <= 5).mean(), len(Y)))
