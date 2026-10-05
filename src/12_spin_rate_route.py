"""
Is the Pitch Plane just spin rate in disguise?

The alternative: spin rate ↔ movement and spin rate ↔ speed (faster arms spin the ball
harder), so movement and speed correlate only through spin rate.

Spin rate has no direction. If it carried the relation, speed should track the SIZE of
the movement. The plane says speed tracks SIGNED vertical movement: rising is fast,
dropping is slow. A curveball separates the two -- lots of spin, lots of movement, slowest
pitch in the bag.

Usage:  python src/12_spin_rate_route.py
"""
import sys, os, importlib.util
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sph", os.path.join(ROOT, "src", "03_sphere.py"))
sph = importlib.util.module_from_spec(spec); spec.loader.exec_module(sph)

d = pd.read_parquet(sph.SNAP[2026], columns=["game_type", "pitcher", "pitch_type", "p_throws",
                    "pfx_x", "pfx_z", "release_speed", "release_spin_rate"])
d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)].dropna().copy()
n = d.groupby("pitcher").size(); d = d[d.pitcher.isin(n[n >= 400].index)]
for c in ["pfx_x", "pfx_z", "release_speed", "release_spin_rate"]:
    d[c] = d[c].astype(float)
d["x"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12
d["y"] = d.pfx_z * 12
d["mag"] = np.hypot(d.x, d.y)
d["rpm"] = d.release_spin_rate / 1000

print("■ 1. 球種ごとのリーグ平均")
T = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS"]
g = d[d.pitch_type.isin(T)].groupby("pitch_type")[["release_speed", "rpm", "y", "mag"]].mean().loc[T]
g.columns = ["球速", "回転数(千rpm)", "縦変化", "変化の大きさ"]
print(g.round(2).to_string())

# per pitcher × type means, then centred within pitcher (arm strength removed)
g = d.groupby("pitcher").pitch_type.value_counts(normalize=True).rename("share").reset_index()
m = d.groupby(["pitcher", "pitch_type"])[["release_speed", "rpm", "x", "y", "mag"]].mean().reset_index()
m = m.merge(g, on=["pitcher", "pitch_type"])
m = m[m.share >= .03]
m = m[m.groupby("pitcher").pitch_type.transform("size") >= 3]
w = m[["release_speed", "rpm", "x", "y", "mag"]] - m.groupby("pitcher")[["release_speed", "rpm", "x", "y", "mag"]].transform("mean")

def r2(cols):
    X = np.c_[w[cols].values, np.ones(len(w))]
    b, *_ = np.linalg.lstsq(X, w.release_speed.values, rcond=None)
    e = w.release_speed.values - X @ b
    return 1 - e.var() / w.release_speed.var(), b

print("\n■ 2. 投手内で球種の球速差を何が説明するか（%d投手×球種、腕の強さは除去済み）" % len(w))
for lab, cols in [("回転数だけ", ["rpm"]), ("変化の大きさだけ", ["mag"]),
                  ("符号つき縦変化だけ", ["y"]), ("縦＋横（PP）", ["y", "x"]),
                  ("PP＋回転数", ["y", "x", "rpm"])]:
    R, b = r2(cols)
    print("   %-16s R² %.3f   係数 %s" % (lab, R, "  ".join("%s %+.3f" % (c, v) for c, v in zip(cols, b))))
print("   相関（投手内）: 回転数×球速 %+.2f   回転数×縦変化 %+.2f   回転数×変化の大きさ %+.2f"
      % (np.corrcoef(w.rpm, w.release_speed)[0, 1], np.corrcoef(w.rpm, w.y)[0, 1], np.corrcoef(w.rpm, w.mag)[0, 1]))

print("\n■ 3. 同じ球種の中（投手間）：回転数と球速は相関するか")
for t in ["FF", "SI", "SL", "CU", "CH"]:
    s = m[m.pitch_type == t]
    print("   %-3s 回転数×球速 %+.2f   回転数×縦変化 %+.2f   n=%d"
          % (t, np.corrcoef(s.rpm, s.release_speed)[0, 1], np.corrcoef(s.rpm, s.y)[0, 1], len(s)))

print("\n■ 4. PPに回転数を足したときの正の係数は、チェンジ・スプリットだけのせいか")
for lab, keep in [("全球種", m.pitch_type.notna()), ("CH/FS/FO を除く", ~m.pitch_type.isin(["CH", "FS", "FO"]))]:
    mm = m[keep]
    mm = mm[mm.groupby("pitcher").pitch_type.transform("size") >= 3]
    ww = mm[["release_speed", "rpm", "x", "y"]] - mm.groupby("pitcher")[["release_speed", "rpm", "x", "y"]].transform("mean")
    X = np.c_[ww[["y", "x", "rpm"]].values, np.ones(len(ww))]
    b, *_ = np.linalg.lstsq(X, ww.release_speed.values, rcond=None)
    X0 = np.c_[ww[["y", "x"]].values, np.ones(len(ww))]
    b0, *_ = np.linalg.lstsq(X0, ww.release_speed.values, rcond=None)
    e, e0 = ww.release_speed.values - X @ b, ww.release_speed.values - X0 @ b0
    v = ww.release_speed.var()
    print("   %-16s PP R² %.3f → +回転数 %.3f   回転数の係数 %+.2f mph / 千rpm   n=%d"
          % (lab, 1 - e0.var() / v, 1 - e.var() / v, b[2], len(ww)))
