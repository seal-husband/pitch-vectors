"""
Pitch arsenals as spectra: which projection resolves the peaks?

A pitcher's pitches, projected onto one axis, form a spectrum; each pitch type is a peak.
Chromatography's measure of whether two peaks are seen as two:

    Rs = |μ1 − μ2| / (2 (σ1 + σ2))          Rs ≥ 1.5 baseline-separated, Rs < 0.5 merged

Computed WITHIN each pitcher, for every pair of his pitch types, on each candidate axis:
the three raw ones, the Pitch Plane's in-plane axes, and its normal (speed that movement
does not explain). Also: how many of a pitcher's types a 1-D spectrum could show, and
whether any axis is best by a margin.

Usage:  python src/14_resolution.py
"""
import sys, os, importlib.util, itertools
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "src", "03_sphere.py"))
sph = importlib.util.module_from_spec(spec); spec.loader.exec_module(sph)

z = np.load(os.path.join(ROOT, "output", "plane_2026.npz"))
n, E1, E2, S, V0 = z["normal"], z["axes"][:, 0], z["axes"][:, 1], float(z["s"]), float(z["v0"])

d = pd.read_parquet(sph.SNAP[2026], columns=["game_type", "pitcher", "pitch_type", "p_throws",
                    "pfx_x", "pfx_z", "release_speed"])
d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)].dropna().copy()
cnt = d.groupby("pitcher").size(); d = d[d.pitcher.isin(cnt[cnt >= 400].index)]
for c in ["pfx_x", "pfx_z", "release_speed"]:
    d[c] = d[c].astype(float)
X = np.c_[np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12, d.pfx_z * 12,
          (d.release_speed - V0) * S]
AX = {"球速": X[:, 2], "横変化": X[:, 0], "縦変化": X[:, 1],
      "PP第1軸": X @ E1, "PP第2軸": X @ E2, "PP法線": X @ n}
for k, v in AX.items():
    d[k] = v

rows = []
for pid, g in d.groupby("pitcher"):
    vc = g.pitch_type.value_counts()
    T = vc[(vc / len(g) >= .03) & (vc >= 30)].index
    st = g[g.pitch_type.isin(T)].groupby("pitch_type")[list(AX)].agg(["mean", "std"])
    for a_, b_ in itertools.combinations(T, 2):
        rec = {"pitcher": pid, "pair": "-".join(sorted([a_, b_]))}
        for k in AX:
            rec[k] = abs(st.loc[a_, (k, "mean")] - st.loc[b_, (k, "mean")]) / (
                2 * (st.loc[a_, (k, "std")] + st.loc[b_, (k, "std")]))
        rows.append(rec)
R = pd.DataFrame(rows)
print("投手内の球種ペア %d 組（%d投手）" % (len(R), R.pitcher.nunique()))
print("\n■ 1. 軸ごとの分離度 Rs（投手内の全ペア）")
print("   %-8s %6s %10s %10s %12s" % ("", "中央", "Rs≥1.5", "Rs<0.5", "最良の軸"))
best = R[list(AX)].idxmax(axis=1).value_counts(normalize=True)
for k in AX:
    print("   %-8s %6.2f %9.0f%% %9.0f%% %11.0f%%" % (k, R[k].median(), 100 * (R[k] >= 1.5).mean(),
          100 * (R[k] < .5).mean(), 100 * best.get(k, 0)))
R["best1"] = R[list(AX)].max(axis=1)
two = [("球速", "横変化"), ("縦変化", "横変化"), ("球速", "縦変化"), ("PP第1軸", "PP第2軸"), ("PP第2軸", "PP法線")]
print("   どれか1軸で Rs≥1.5 になるペア %.0f%%" % (100 * (R.best1 >= 1.5).mean()))
for a_, b_ in two:
    print("   %s か %s のどちらかで Rs≥1.5  %.0f%%" % (a_, b_, 100 * (R[[a_, b_]].max(axis=1) >= 1.5).mean()))

print("\n■ 2. 分けにくいペア（全軸の最大 Rs の中央値が小さい順）")
pp = R.groupby("pair").agg(n=("best1", "size"), best=("best1", "median"),
                           ax=("best1", lambda s: R.loc[s.index, list(AX)].median().idxmax()))
print(pp[pp.n >= 30].sort_values("best").head(10).round(2).to_string())
R.to_csv(os.path.join(ROOT, "output", "resolution_2026.csv"), index=False)
