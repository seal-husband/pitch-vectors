"""
Four numbers per pitcher: how much each side's movement he relies on.

The three principles, applied without further machinery:

  1. arm side and glove side are separate continents   -> two sides
  2. the origin is the four-seam (converted if absent)  -> displacements from it
  3. movement is enough; speed follows (Pitch Plane),
     and the pitcher's plane height is Stuff, not style -> no speed at all

For each side:  horizontal = Σ share × (inches toward that side),
                vertical   = Σ share × (inches of drop)
All four are in the same unit (inches per pitch thrown), so nothing needs scaling.

What the collapse to sums throws away is the peak structure: one pitch at 20 in used 20%
and two pitches at 10 and 30 in used 10% each give the same number. This file measures
the cost against the Earth Mover's Distance on the full peak sets, and the stability of
the four numbers when each pitcher's pitches are split in alternate halves.

Pitch-level version (no peak picking at all): every pitch other than the four-seam is put
on a side by position (Δ縦 ≤ 2 in; glove if Δ横 ≥ 2 in, arm otherwise). The peak-level
version uses the peak table (21). They are compared.

Usage:  python src/23_four_numbers.py
"""
import sys, os
from paths import SNAP
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["腕側の横", "腕側の縦", "グラブ側の横", "グラブ側の縦"]
GLOVE_T = {"FC", "SL", "ST", "SV", "CU", "KC", "CS"}
ARM_T = {"SI", "CH", "FS", "FO"}

# ------------------------------------------------------------------ peak level
Pk = pd.read_parquet(os.path.join(ROOT, "output", "peaks_ff_2026.parquet"))
low = Pk.fz <= 2
Pk["side"] = np.where(low & (Pk.fx >= 2) & Pk.label.isin(GLOVE_T), "glove",
             np.where(low & (Pk.fx < 2) & Pk.label.isin(ARM_T), "arm", "other"))


def four(df, side_col, x, z, w):
    out = {}
    for p, g in df.groupby("pitcher"):
        a, gl = g[g[side_col] == "arm"], g[g[side_col] == "glove"]
        out[p] = [(a[w] * -a[x]).sum(), (a[w] * -a[z]).sum(), (gl[w] * gl[x]).sum(), (gl[w] * -gl[z]).sum()]
    return pd.DataFrame(out, index=COLS).T


F_peak = four(Pk, "side", "fx", "fz", "share")

# ------------------------------------------------------------------ pitch level
raw = pd.read_parquet(SNAP[2026],
                      columns=["game_type", "pitcher", "pitch_type", "p_throws", "pfx_x", "pfx_z", "release_speed"])
raw = raw[(raw.game_type == "R") & raw.pitcher.isin(set(Pk.pitcher))].dropna().copy()
for c in ["pfx_x", "pfx_z"]:
    raw[c] = raw[c].astype(float)
raw["hx"] = np.where(raw.p_throws.eq("L"), -raw.pfx_x, raw.pfx_x) * 12
raw["vz"] = raw.pfx_z * 12
# the same four-seam reference as 21
fb = raw[raw.pitch_type.isin(["FF", "SI", "FC"])].groupby(["pitcher", "pitch_type"])[["hx", "vz"]].agg(["mean", "size"])
fb.columns = ["hx", "n1", "vz", "n"]
fb = fb.drop(columns="n1").reset_index()
tot = raw.groupby("pitcher").size()
fb["share"] = fb.n / tot.reindex(fb.pitcher).values
has = fb[(fb.n >= 30) & (fb.share >= .05)]
W = has.pivot(index="pitcher", columns="pitch_type", values=["hx", "vz"])
off = {t: np.array([(W.dropna(subset=[(k, "FF"), (k, t)])[(k, "FF")] - W.dropna(subset=[(k, "FF"), (k, t)])[(k, t)]).median()
                    for k in ["hx", "vz"]]) for t in ["SI", "FC"]}
ref = {}
for p, g in has.groupby("pitcher"):
    ff = g[g.pitch_type == "FF"]
    ref[p] = ff[["hx", "vz"]].values[0] if len(ff) else g.sort_values("n").iloc[-1][["hx", "vz"]].values.astype(float) + off[g.sort_values("n").iloc[-1].pitch_type]
raw = raw[raw.pitcher.isin(ref.keys())]
R = np.array([ref[p] for p in raw.pitcher])
raw["fx"], raw["fz"] = raw.hx - R[:, 0], raw.vz - R[:, 1]
raw = raw[raw.pitch_type != "FF"]                     # the four-seam is the origin, not a contribution
raw["side"] = np.where(raw.fz <= 2, np.where(raw.fx >= 2, "glove", "arm"), "other")
n_all = tot.reindex(raw.pitcher).values
raw["w"] = 1.0 / n_all                                # each pitch = 1 / (pitcher's total pitches)
raw["half"] = raw.groupby("pitcher").cumcount() % 2
F_pitch = four(raw, "side", "fx", "fz", "w")


def half_four(h):
    s = raw[raw.half == h].copy()
    s["w"] = 1.0 / s.groupby("pitcher").pitcher.transform("size") * (raw.groupby("pitcher").size() / tot).reindex(s.pitcher).values
    return four(s, "side", "fx", "fz", "w")


A_, B_ = half_four(0), half_four(1)
common = F_peak.index.intersection(F_pitch.index)
print("投手 %d" % len(common))
print("\n■ 1. 4つの数（中央値と四分位, インチ／球）")
print(F_pitch.loc[common].describe(percentiles=[.25, .5, .75]).loc[["25%", "50%", "75%"]].round(2).to_string())
print("\n■ 2. ピークから求めた値と、一球ずつ求めた値の一致（Pearson）")
print("   " + "  ".join("%s %.3f" % (c, np.corrcoef(F_peak.loc[common, c], F_pitch.loc[common, c])[0, 1]) for c in COLS))
print("\n■ 3. 折半の再現性（一球おき、Spearman-Brown 補正）")
print("   " + "  ".join("%s %.3f" % (c, (lambda r: 2 * r / (1 + r))(np.corrcoef(A_.loc[common, c], B_.loc[common, c])[0, 1])) for c in COLS))
print("\n■ 4. 4つの数どうしの相関（Pearson）")
print(F_pitch.loc[common].corr().round(2).to_string())

# ------------------------------------------------------------------ vs EMD
UNITS = 60


def pts(p, side):
    g = Pk[Pk.pitcher == p]
    g = g[g.side == side] if side != "both" else g[g.side != "other"]
    if g.empty:
        return None
    u = np.round(g.share.values / g.share.sum() * UNITS).astype(int); u[np.argmax(u)] += UNITS - u.sum()
    return np.repeat(g[["fx", "fz"]].values, u, axis=0), g.share.sum()


def emd(a, b):
    C = np.linalg.norm(a[:, None] - b[None], axis=2); i, j = linear_sum_assignment(C); return C[i, j].mean()


rng = np.random.default_rng(23)
pids = rng.choice(np.array(common), 250, replace=False)
print("\n■ 5. 4つの数の距離は、ピーク集合の EMD をどれだけ保つか（無作為 250 投手、Spearman）")
for side, cols in (("glove", COLS[2:]), ("arm", COLS[:2])):
    S = {p: pts(p, side) for p in pids}
    ok = [p for p in pids if S[p] is not None]
    d_emd, d_4 = [], []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            a, b = S[ok[i]], S[ok[j]]
            # EMD of the shapes, plus the usage difference carried at the typical distance
            d_emd.append(emd(a[0], b[0]))
            d_4.append(np.linalg.norm(F_peak.loc[ok[i], cols].values - F_peak.loc[ok[j], cols].values))
    print("   %-5s 形の EMD × 4つの数の距離  ρ = %.2f   (投手 %d)" % (side, stats.spearmanr(d_emd, d_4)[0], len(ok)))
    # the same with the sums divided by the side's usage = the side's mean position
    m = {p: F_peak.loc[p, cols].values / S[p][1] for p in ok}
    d_m = [np.linalg.norm(m[ok[i]] - m[ok[j]]) for i in range(len(ok)) for j in range(i + 1, len(ok))]
    print("   %-5s 形の EMD × 平均位置（使用率で割る）の距離  ρ = %.2f" % (side, stats.spearmanr(d_emd, d_m)[0]))
F_pitch.loc[common].to_csv(os.path.join(ROOT, "output", "four_numbers_2026.csv"))
